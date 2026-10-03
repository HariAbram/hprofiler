/*
 * Shared event transport for every hprofiler hook library.
 *
 *   hot path    hp_tx_emit()/hp_tx_emitf() copy the record into the CALLING
 *               THREAD's own ring (rb_bytes_t, ringbuffer.h): no lock, no
 *               syscall. Records up to HP_TX_MAX_RECORD travel intact;
 *               larger ones are rejected and counted (oversize).
 *   drain       one background thread per hook library empties the rings
 *               into the collector socket in batches of up to HP_TX_BATCH
 *               bytes. Per-thread order is preserved (each ring is FIFO and
 *               drained in order); records of different threads interleave.
 *   full ring   the producer waits for space (kicking the drain thread), at
 *               most HPROFILER_RING_WAIT_MS per record (default 1000); only
 *               then is the record dropped and counted (dropped_full). It
 *               never blocks indefinitely and never discards silently.
 *   no socket   records that cannot be delivered (no collector, send failure
 *               after one reconnect) are counted (dropped_lost).
 *   status      "xport:1:<pid>:<hook>:k=v,..." reports the counters:
 *               periodically after a loss, and with final=1 at shutdown.
 *               image=<start ns> separates the images before and after an
 *               exec (same pid). A connection that closes without final=1
 *               ended without a clean drain (crash, _exit, kill).
 *   shutdown    hp_tx_shutdown() (hook destructors, MPI_Finalize, OMPT
 *               finalize, exec*) stops the drain thread, drains every ring,
 *               sends the final status and switches to synchronous sends
 *               for anything emitted later (a thread with buffered records
 *               drains its own ring first, keeping its order).
 *   fork        the child drops the parent's buffered records (the parent
 *               sends them), its socket and cached ids, and starts its own
 *               transport on its first event.
 *   exec        preloaded hooks interpose execve/execv/execvp/execvpe/
 *               execl/execlp/execle and drain before the image is replaced.
 *
 * HPROFILER_TRANSPORT=sync sends synchronously instead (same records, same
 * status), for A/B overhead comparisons. HPROFILER_RING_KB sets the
 * per-thread ring size (default 512 KiB, allocated on a thread's first
 * event), HPROFILER_DRAIN_US the idle drain interval (default 500 us).
 *
 * Each hook library includes this header once (static state per library)
 * and calls hp_tx_init("<hook>") from its constructor. Linux-only (futex,
 * gettid), like the hooks themselves.
 */
#ifndef HP_TRANSPORT_H
#define HP_TRANSPORT_H

#include <alloca.h>
#include <dlfcn.h>
#include <errno.h>
#include <linux/futex.h>
#include <pthread.h>
#include <signal.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

#include "ringbuffer.h"

#define HP_TX_WIRE_VERSION 1
#define HP_TX_MAX_RECORD   (64u * 1024u)
#define HP_TX_BATCH        (128u * 1024u)

enum { HP_TX_UNINIT = 0, HP_TX_RING = 1, HP_TX_SYNC = 2 };

/* Producer-side counters live with each thread's ring (written only by
 * that thread, summed when a status record is sent) so the hot path never
 * touches a cache line shared between threads; events emitted without a
 * ring (sync mode, after shutdown) use the global block. */
typedef struct {
    _Atomic uint64_t emitted, dropped_full, oversize, format_errors, blocked_ns, waits, sanitized;
} hp_tx_ctr_t;

typedef struct hp_tx_ring {
    rb_bytes_t          rb;
    hp_tx_ctr_t         ctr;
    _Atomic int         orphan;      /* owning thread exited */
    struct hp_tx_ring  *next;
} hp_tx_ring_t;

static struct {
    pthread_once_t   once;
    const char      *hook;
    int              enabled;        /* HPROFILER_SOCKET set */
    int              want_ring;
    uint64_t         ring_bytes;
    uint64_t         wait_ns;
    uint64_t         drain_ns;
    char             path[108];
    _Atomic int      state;
    pthread_mutex_t  init_mu;        /* starting the drain thread */
    pthread_mutex_t  reg_mu;         /* ring registry */
    pthread_mutex_t  drain_mu;       /* whoever empties rings */
    pthread_mutex_t  sock_mu;        /* socket writes */
    hp_tx_ring_t    *rings;
    int              sock;
    uint64_t         next_connect_ns;
    pthread_t        drain;
    int              drain_started;
    _Atomic uint32_t kick;
    _Atomic int      stop;
    pthread_key_t    key;
    _Atomic pid_t    pid;
    char            *batch;
    /* counters: producer side (global block + freed rings' totals), then
     * delivery side (updated by whoever drains) */
    hp_tx_ctr_t      ctr;
    _Atomic uint64_t sent, bytes, dropped_lost, max_block_ns, threads, reconnects, send_errors;
    _Atomic int      dirty;          /* a loss/oversize counter changed */
    uint64_t         last_status_ns;
    uint64_t         image_ns;       /* start of this process image (fork/exec) */
} hp_tx = {
    .once = PTHREAD_ONCE_INIT, .init_mu = PTHREAD_MUTEX_INITIALIZER,
    .reg_mu = PTHREAD_MUTEX_INITIALIZER, .drain_mu = PTHREAD_MUTEX_INITIALIZER,
    .sock_mu = PTHREAD_MUTEX_INITIALIZER, .sock = -1,
};

static __thread hp_tx_ring_t *hp_tx_tls_ring;
static __thread hp_tx_ctr_t  *hp_tx_tls_ctr;   /* ring's block, or NULL = global */
static __thread pid_t         hp_tx_tls_tid;
static __thread int           hp_tx_tls_busy;   /* re-entrancy guard */

static inline uint64_t hp_tx_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static inline hp_tx_ctr_t *hp_tx_c(void) { return hp_tx_tls_ctr ? hp_tx_tls_ctr : &hp_tx.ctr; }
#define HP_TX_INC(field, n) \
    atomic_fetch_add_explicit(&hp_tx_c()->field, (n), memory_order_relaxed)

static inline void hp_tx_ctr_add(hp_tx_ctr_t *dst, hp_tx_ctr_t *src) {
    atomic_fetch_add(&dst->emitted, atomic_load(&src->emitted));
    atomic_fetch_add(&dst->dropped_full, atomic_load(&src->dropped_full));
    atomic_fetch_add(&dst->oversize, atomic_load(&src->oversize));
    atomic_fetch_add(&dst->format_errors, atomic_load(&src->format_errors));
    atomic_fetch_add(&dst->blocked_ns, atomic_load(&src->blocked_ns));
    atomic_fetch_add(&dst->waits, atomic_load(&src->waits));
    atomic_fetch_add(&dst->sanitized, atomic_load(&src->sanitized));
}

static inline void hp_tx_max(_Atomic uint64_t *slot, uint64_t v) {
    uint64_t cur = atomic_load_explicit(slot, memory_order_relaxed);
    while (v > cur && !atomic_compare_exchange_weak_explicit(slot, &cur, v, memory_order_relaxed,
                                                            memory_order_relaxed)) {}
}

/* ids ──────────────────────────────────────────────────────────────────── */
static inline pid_t hp_tx_pid(void) {
    pid_t p = atomic_load_explicit(&hp_tx.pid, memory_order_relaxed);
    if (p == 0) {
        p = getpid();
        atomic_store_explicit(&hp_tx.pid, p, memory_order_relaxed);
    }
    return p;
}

static inline pid_t hp_tx_tid(void) {
    if (hp_tx_tls_tid == 0) hp_tx_tls_tid = (pid_t)syscall(SYS_gettid);
    return hp_tx_tls_tid;
}

/* socket ───────────────────────────────────────────────────────────────── */
/* Caller holds sock_mu. Connection attempts are rate-limited so a missing
 * collector costs one failed connect() per 100 ms, not one per event. */
static int hp_tx_connect_locked(void) {
    if (hp_tx.sock >= 0) return 1;
    uint64_t now = hp_tx_now();
    if (now < hp_tx.next_connect_ns) return 0;
    int s = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (s >= 0) {
        struct sockaddr_un addr;
        memset(&addr, 0, sizeof(addr));
        addr.sun_family = AF_UNIX;
        memcpy(addr.sun_path, hp_tx.path, sizeof(addr.sun_path) - 1);
        if (connect(s, (struct sockaddr *)&addr, sizeof(addr)) == 0) {
            hp_tx.sock = s;
            return 1;
        }
        close(s);
    }
    hp_tx.next_connect_ns = now + 100000000ull;
    return 0;
}

static int hp_tx_send_raw_locked(const char *buf, size_t n) {
    while (n > 0) {
        ssize_t r = send(hp_tx.sock, buf, n, MSG_NOSIGNAL);
        if (r < 0) {
            if (errno == EINTR) continue;
            return 0;
        }
        buf += r;
        n -= (size_t)r;
    }
    return 1;
}

/* Deliver `n` bytes holding `nrec` complete records: one reconnect on
 * failure (a new connection carries the rest), otherwise they are counted
 * as lost. A partially written record on a broken connection is followed by
 * EOF on that connection, which the collector counts as a partial line. */
static void hp_tx_deliver(const char *buf, size_t n, uint64_t nrec) {
    if (n == 0) return;
    pthread_mutex_lock(&hp_tx.sock_mu);
    int ok = 0;
    for (int attempt = 0; attempt < 2 && !ok; attempt++) {
        if (!hp_tx_connect_locked()) break;
        if (hp_tx_send_raw_locked(buf, n)) {
            ok = 1;
        } else {
            atomic_fetch_add_explicit(&hp_tx.send_errors, 1, memory_order_relaxed);
            close(hp_tx.sock);
            hp_tx.sock = -1;
            hp_tx.next_connect_ns = 0;
            if (attempt == 0) atomic_fetch_add_explicit(&hp_tx.reconnects, 1, memory_order_relaxed);
        }
    }
    pthread_mutex_unlock(&hp_tx.sock_mu);
    if (ok) {
        atomic_fetch_add_explicit(&hp_tx.sent, nrec, memory_order_relaxed);
        atomic_fetch_add_explicit(&hp_tx.bytes, n, memory_order_relaxed);
    } else {
        atomic_fetch_add_explicit(&hp_tx.dropped_lost, nrec, memory_order_relaxed);
        atomic_store_explicit(&hp_tx.dirty, 1, memory_order_relaxed);
    }
}

/* status ───────────────────────────────────────────────────────────────── */
static void hp_tx_send_status(int final) {
    hp_tx_ctr_t sum;
    memset(&sum, 0, sizeof(sum));
    hp_tx_ctr_add(&sum, &hp_tx.ctr);
    pthread_mutex_lock(&hp_tx.reg_mu);
    for (hp_tx_ring_t *r = hp_tx.rings; r; r = r->next) hp_tx_ctr_add(&sum, &r->ctr);
    pthread_mutex_unlock(&hp_tx.reg_mu);
    if (final && atomic_load(&sum.emitted) == 0) return;   /* this image never emitted anything */
    char line[640];
    int n = snprintf(line, sizeof(line),
        "xport:%d:%d:%s:mode=%s,final=%d,emitted=%llu,sent=%llu,bytes=%llu,dropped_full=%llu,"
        "dropped_lost=%llu,oversize=%llu,format_errors=%llu,blocked_ns=%llu,max_block_ns=%llu,"
        "waits=%llu,threads=%llu,ring_kb=%llu,reconnects=%llu,send_errors=%llu,sanitized=%llu,"
        "image=%llu\n",
        HP_TX_WIRE_VERSION, (int)hp_tx_pid(), hp_tx.hook ? hp_tx.hook : "?",
        hp_tx.want_ring ? "ring" : "sync", final,
        (unsigned long long)atomic_load(&sum.emitted), (unsigned long long)atomic_load(&hp_tx.sent),
        (unsigned long long)atomic_load(&hp_tx.bytes), (unsigned long long)atomic_load(&sum.dropped_full),
        (unsigned long long)atomic_load(&hp_tx.dropped_lost), (unsigned long long)atomic_load(&sum.oversize),
        (unsigned long long)atomic_load(&sum.format_errors), (unsigned long long)atomic_load(&sum.blocked_ns),
        (unsigned long long)atomic_load(&hp_tx.max_block_ns), (unsigned long long)atomic_load(&sum.waits),
        (unsigned long long)atomic_load(&hp_tx.threads), (unsigned long long)(hp_tx.ring_bytes / 1024),
        (unsigned long long)atomic_load(&hp_tx.reconnects), (unsigned long long)atomic_load(&hp_tx.send_errors),
        (unsigned long long)atomic_load(&sum.sanitized), (unsigned long long)hp_tx.image_ns);
    if (n <= 0 || n >= (int)sizeof(line)) return;
    pthread_mutex_lock(&hp_tx.sock_mu);
    if (hp_tx_connect_locked() && !hp_tx_send_raw_locked(line, (size_t)n)) {
        close(hp_tx.sock);
        hp_tx.sock = -1;
    }
    pthread_mutex_unlock(&hp_tx.sock_mu);
    atomic_store_explicit(&hp_tx.dirty, 0, memory_order_relaxed);
    hp_tx.last_status_ns = hp_tx_now();
}

/* configuration ────────────────────────────────────────────────────────── */
static uint64_t hp_tx_env_u64(const char *name, uint64_t dflt, uint64_t lo, uint64_t hi) {
    const char *v = getenv(name);
    if (!v || !*v) return dflt;
    char *end = NULL;
    unsigned long long x = strtoull(v, &end, 10);
    if (end == v) return dflt;
    if (x < lo) x = lo;
    if (x > hi) x = hi;
    return (uint64_t)x;
}

static void hp_tx_ring_dtor(void *p);
static void hp_tx_atfork_child(void);

static void hp_tx_configure(void) {
    const char *path = getenv("HPROFILER_SOCKET");
    hp_tx.enabled = path && *path;
    if (hp_tx.enabled) strncpy(hp_tx.path, path, sizeof(hp_tx.path) - 1);
    const char *mode = getenv("HPROFILER_TRANSPORT");
    hp_tx.want_ring = !(mode && strcmp(mode, "sync") == 0);
    hp_tx.ring_bytes = hp_tx_env_u64("HPROFILER_RING_KB", 512, 16, 1u << 20) * 1024u;
    hp_tx.wait_ns = hp_tx_env_u64("HPROFILER_RING_WAIT_MS", 1000, 0, 600000) * 1000000ull;
    hp_tx.drain_ns = hp_tx_env_u64("HPROFILER_DRAIN_US", 500, 50, 1000000) * 1000ull;
    hp_tx.image_ns = hp_tx_now();
    pthread_key_create(&hp_tx.key, hp_tx_ring_dtor);
    pthread_atfork(NULL, NULL, hp_tx_atfork_child);
}

static inline void hp_tx_init(const char *hook) {
    if (!hp_tx.hook) hp_tx.hook = hook;
    pthread_once(&hp_tx.once, hp_tx_configure);
}

static inline int hp_tx_enabled(void) {
    pthread_once(&hp_tx.once, hp_tx_configure);
    return hp_tx.enabled;
}

/* rings ────────────────────────────────────────────────────────────────── */
static void hp_tx_ring_dtor(void *p) {
    hp_tx_ring_t *r = (hp_tx_ring_t *)p;
    if (hp_tx_tls_ring == r) {
        hp_tx_tls_ring = NULL;
        hp_tx_tls_ctr = NULL;
    }
    if (r) atomic_store_explicit(&r->orphan, 1, memory_order_release);
}

static hp_tx_ring_t *hp_tx_my_ring(void) {
    hp_tx_ring_t *r = hp_tx_tls_ring;
    if (r) return r;
    r = (hp_tx_ring_t *)calloc(1, sizeof(*r));
    if (!r) return NULL;
    if (!rb_bytes_init(&r->rb, hp_tx.ring_bytes)) {
        free(r);
        return NULL;
    }
    atomic_init(&r->orphan, 0);
    pthread_mutex_lock(&hp_tx.reg_mu);
    r->next = hp_tx.rings;
    hp_tx.rings = r;
    pthread_mutex_unlock(&hp_tx.reg_mu);
    pthread_setspecific(hp_tx.key, r);
    hp_tx_tls_ring = r;
    hp_tx_tls_ctr = &r->ctr;
    atomic_fetch_add_explicit(&hp_tx.threads, 1, memory_order_relaxed);
    return r;
}

/* Empty one ring into the batch buffer, delivering whenever it fills.
 * Caller holds drain_mu. Returns the number of records taken. */
static uint64_t hp_tx_drain_ring(hp_tx_ring_t *r, size_t *used, uint64_t *nrec) {
    uint64_t taken = 0;
    for (;;) {
        int64_t len = rb_bytes_peek_len(&r->rb);
        if (len < 0) break;
        if (*used + (size_t)len > HP_TX_BATCH) {
            hp_tx_deliver(hp_tx.batch, *used, *nrec);
            *used = 0;
            *nrec = 0;
        }
        rb_bytes_pop(&r->rb, hp_tx.batch + *used, (uint32_t)len);
        *used += (size_t)len;
        *nrec += 1;
        taken++;
    }
    return taken;
}

/* One pass over every ring (caller holds drain_mu). Orphaned rings that are
 * empty are unlinked and freed -- only the drain owner ever frees, and
 * nobody else traverses while drain_mu is held. */
static uint64_t hp_tx_drain_all_locked(void) {
    if (!hp_tx.batch) {
        hp_tx.batch = (char *)malloc(HP_TX_BATCH);
        if (!hp_tx.batch) return 0;
    }
    size_t used = 0;
    uint64_t nrec = 0, taken = 0;
    pthread_mutex_lock(&hp_tx.reg_mu);
    hp_tx_ring_t *r = hp_tx.rings;
    pthread_mutex_unlock(&hp_tx.reg_mu);
    for (; r; r = r->next)
        taken += hp_tx_drain_ring(r, &used, &nrec);
    hp_tx_deliver(hp_tx.batch, used, nrec);
    pthread_mutex_lock(&hp_tx.reg_mu);
    hp_tx_ring_t **pp = &hp_tx.rings;
    while (*pp) {
        hp_tx_ring_t *q = *pp;
        if (atomic_load_explicit(&q->orphan, memory_order_acquire) && rb_bytes_empty(&q->rb)) {
            *pp = q->next;
            hp_tx_ctr_add(&hp_tx.ctr, &q->ctr);
            free(q->rb.buf);
            free(q);
        } else {
            pp = &q->next;
        }
    }
    pthread_mutex_unlock(&hp_tx.reg_mu);
    return taken;
}

static void hp_tx_drain_own_locked(hp_tx_ring_t *r) {
    if (!hp_tx.batch) hp_tx.batch = (char *)malloc(HP_TX_BATCH);
    if (!hp_tx.batch) return;
    size_t used = 0;
    uint64_t nrec = 0;
    hp_tx_drain_ring(r, &used, &nrec);
    hp_tx_deliver(hp_tx.batch, used, nrec);
}

static inline void hp_tx_kick(void) {
    if (atomic_exchange_explicit(&hp_tx.kick, 1, memory_order_acq_rel) == 0)
        syscall(SYS_futex, &hp_tx.kick, FUTEX_WAKE_PRIVATE, 1, NULL, NULL, 0);
}

static void *hp_tx_drain_main(void *arg) {
    (void)arg;
    sigset_t all;
    sigfillset(&all);
    pthread_sigmask(SIG_BLOCK, &all, NULL);   /* signals belong to the application */
    hp_tx_tls_busy = 1;
    while (!atomic_load_explicit(&hp_tx.stop, memory_order_acquire)) {
        pthread_mutex_lock(&hp_tx.drain_mu);
        uint64_t n = hp_tx_drain_all_locked();
        pthread_mutex_unlock(&hp_tx.drain_mu);
        uint64_t now = hp_tx_now();
        if (atomic_load_explicit(&hp_tx.dirty, memory_order_relaxed) &&
            now - hp_tx.last_status_ns > 1000000000ull)
            hp_tx_send_status(0);
        if (n == 0) {
            atomic_store_explicit(&hp_tx.kick, 0, memory_order_release);
            if (atomic_load_explicit(&hp_tx.stop, memory_order_acquire)) break;
            struct timespec ts = { (time_t)(hp_tx.drain_ns / 1000000000ull),
                                   (long)(hp_tx.drain_ns % 1000000000ull) };
            syscall(SYS_futex, &hp_tx.kick, FUTEX_WAIT_PRIVATE, 0, &ts, NULL, 0);
        }
    }
    return NULL;
}

/* Start ring mode (first event of this process image). Falls back to
 * synchronous sends if the drain thread cannot be created. */
static void hp_tx_start(void) {
    pthread_mutex_lock(&hp_tx.init_mu);
    if (atomic_load(&hp_tx.state) == HP_TX_UNINIT) {
        int mode = HP_TX_SYNC;
        if (hp_tx.want_ring) {
            atomic_store(&hp_tx.stop, 0);
            if (pthread_create(&hp_tx.drain, NULL, hp_tx_drain_main, NULL) == 0) {
                hp_tx.drain_started = 1;
                mode = HP_TX_RING;
            } else {
                hp_tx.want_ring = 0;
            }
        }
        atomic_store(&hp_tx.state, mode);
    }
    pthread_mutex_unlock(&hp_tx.init_mu);
}

/* Synchronous delivery (sync mode, after shutdown). Drains the calling
 * thread's own leftover ring first so its records stay in order. */
static void hp_tx_emit_sync(const char *rec, size_t len) {
    hp_tx_ring_t *r = hp_tx_tls_ring;
    pthread_mutex_lock(&hp_tx.drain_mu);
    if (r && !rb_bytes_empty(&r->rb)) hp_tx_drain_own_locked(r);
    pthread_mutex_unlock(&hp_tx.drain_mu);
    hp_tx_deliver(rec, len, 1);
}

/* Replace interior line breaks so one record can never become two lines
 * (or swallow the next one) on the wire. */
static inline void hp_tx_sanitize(char *p, size_t len) {
    int fixed = 0;
    for (size_t i = 0; i + 1 < len; i++)
        if (p[i] == '\n' || p[i] == '\r' || p[i] == '\0') { p[i] = ' '; fixed = 1; }
    if (fixed) HP_TX_INC(sanitized, 1);
}

/* Push a record whose interior has been sanitized and which ends in '\n'. */
static void hp_tx_push(char *rec, size_t len) {
    int st = atomic_load_explicit(&hp_tx.state, memory_order_acquire);
    if (st == HP_TX_UNINIT) {
        hp_tx_start();
        st = atomic_load_explicit(&hp_tx.state, memory_order_acquire);
    }
    hp_tx_ring_t *r = st == HP_TX_RING ? hp_tx_my_ring() : NULL;
    if (!r) {
        hp_tx_emit_sync(rec, len);
        return;
    }
    if (len + 4 > r->rb.cap / 2) {
        HP_TX_INC(oversize, 1);
        atomic_store_explicit(&hp_tx.dirty, 1, memory_order_relaxed);
        return;
    }
    if (!rb_bytes_push(&r->rb, rec, (uint32_t)len)) {
        if (hp_tx.wait_ns == 0) {         /* HPROFILER_RING_WAIT_MS=0: never wait */
            hp_tx_kick();
            HP_TX_INC(dropped_full, 1);
            atomic_store_explicit(&hp_tx.dirty, 1, memory_order_relaxed);
            return;
        }
        /* bounded wait for the drain thread */
        uint64_t t0 = hp_tx_now(), sleep_ns = 20000;
        int pushed = 0;
        HP_TX_INC(waits, 1);
        for (;;) {
            hp_tx_kick();
            if (atomic_load_explicit(&hp_tx.state, memory_order_acquire) != HP_TX_RING) break;
            struct timespec ts = { 0, (long)sleep_ns };
            nanosleep(&ts, NULL);
            if (sleep_ns < 1000000) sleep_ns *= 2;
            if (rb_bytes_push(&r->rb, rec, (uint32_t)len)) { pushed = 1; break; }
            if (hp_tx_now() - t0 >= hp_tx.wait_ns) break;
        }
        uint64_t waited = hp_tx_now() - t0;
        HP_TX_INC(blocked_ns, waited);
        hp_tx_max(&hp_tx.max_block_ns, waited);
        if (!pushed) {
            if (atomic_load_explicit(&hp_tx.state, memory_order_acquire) != HP_TX_RING) {
                hp_tx_emit_sync(rec, len);      /* shut down while we waited */
            } else {
                HP_TX_INC(dropped_full, 1);
                atomic_store_explicit(&hp_tx.dirty, 1, memory_order_relaxed);
            }
            return;
        }
    }
    /* Publish-then-check against a concurrent shutdown (Dekker style):
     * either the final drain sees this record or we drain it ourselves. */
    atomic_thread_fence(memory_order_seq_cst);
    if (atomic_load_explicit(&hp_tx.state, memory_order_seq_cst) != HP_TX_RING) {
        pthread_mutex_lock(&hp_tx.drain_mu);
        hp_tx_drain_own_locked(r);
        pthread_mutex_unlock(&hp_tx.drain_mu);
        return;
    }
    if (r->rb.cap - (atomic_load_explicit(&r->rb.tail, memory_order_relaxed) -
                     atomic_load_explicit(&r->rb.head, memory_order_relaxed)) < r->rb.cap / 2)
        hp_tx_kick();
}

/* public API ───────────────────────────────────────────────────────────── */

/* Emit one complete record. `rec` must end with '\n'; it is copied. */
static void hp_tx_emit(const char *rec, size_t len) {
    if (!hp_tx_enabled() || len == 0 || hp_tx_tls_busy) return;
    hp_tx_tls_busy = 1;
    HP_TX_INC(emitted, 1);
    if (len > HP_TX_MAX_RECORD) {
        HP_TX_INC(oversize, 1);
        atomic_store_explicit(&hp_tx.dirty, 1, memory_order_relaxed);
        hp_tx_tls_busy = 0;
        return;
    }
    char stack[1024];
    char *copy = len <= sizeof(stack) ? stack : (char *)malloc(len);
    if (copy) {
        memcpy(copy, rec, len);
        hp_tx_sanitize(copy, len);
        hp_tx_push(copy, len);
        if (copy != stack) free(copy);
    } else {
        HP_TX_INC(dropped_full, 1);
        atomic_store_explicit(&hp_tx.dirty, 1, memory_order_relaxed);
    }
    hp_tx_tls_busy = 0;
}

/* Format and emit one record (the format must end with "\n"). Records of
 * any length up to HP_TX_MAX_RECORD are formatted completely (heap buffer
 * past 1 KiB); longer ones are counted as oversize, never cut. */
__attribute__((format(printf, 1, 2)))
static void hp_tx_emitf(const char *fmt, ...) {
    if (!hp_tx_enabled() || hp_tx_tls_busy) return;
    hp_tx_tls_busy = 1;
    HP_TX_INC(emitted, 1);
    char stack[1024];
    va_list ap, ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int n = vsnprintf(stack, sizeof(stack), fmt, ap);
    va_end(ap);
    if (n <= 0) {
        if (n < 0) {
            HP_TX_INC(format_errors, 1);
            atomic_store_explicit(&hp_tx.dirty, 1, memory_order_relaxed);
        }
    } else if ((size_t)n > HP_TX_MAX_RECORD) {
        HP_TX_INC(oversize, 1);
        atomic_store_explicit(&hp_tx.dirty, 1, memory_order_relaxed);
    } else if ((size_t)n < sizeof(stack)) {
        hp_tx_sanitize(stack, (size_t)n);
        hp_tx_push(stack, (size_t)n);
    } else {
        char *big = (char *)malloc((size_t)n + 1);
        if (big && vsnprintf(big, (size_t)n + 1, fmt, ap2) == n) {
            hp_tx_sanitize(big, (size_t)n);
            hp_tx_push(big, (size_t)n);
        } else {
            HP_TX_INC(format_errors, 1);
            atomic_store_explicit(&hp_tx.dirty, 1, memory_order_relaxed);
        }
        free(big);
    }
    va_end(ap2);
    hp_tx_tls_busy = 0;
}

/* Drain everything and send the status record. Safe to call more than once
 * (MPI_Finalize, then the destructor); later events go out synchronously.
 * `final`=1 marks the end of this process image. */
static void hp_tx_shutdown(int final) {
    if (!hp_tx_enabled()) return;
    pthread_mutex_lock(&hp_tx.init_mu);
    int prev = atomic_exchange_explicit(&hp_tx.state, HP_TX_SYNC, memory_order_seq_cst);
    atomic_thread_fence(memory_order_seq_cst);
    int joined = 1;
    if (prev == HP_TX_RING && hp_tx.drain_started) {
        atomic_store_explicit(&hp_tx.stop, 1, memory_order_release);
        hp_tx_kick();
        /* bounded: a collector that stopped reading must not hang exit() */
        struct timespec dl;
        clock_gettime(CLOCK_REALTIME, &dl);
        uint64_t ms = hp_tx_env_u64("HPROFILER_SHUTDOWN_MS", 10000, 10, 600000);
        dl.tv_sec += (time_t)(ms / 1000);
        dl.tv_nsec += (long)((ms % 1000) * 1000000);
        if (dl.tv_nsec >= 1000000000L) { dl.tv_sec++; dl.tv_nsec -= 1000000000L; }
        joined = pthread_timedjoin_np(hp_tx.drain, NULL, &dl) == 0;
        if (!joined) pthread_detach(hp_tx.drain);
        hp_tx.drain_started = 0;
    }
    pthread_mutex_unlock(&hp_tx.init_mu);
    struct timespec now_rt;
    clock_gettime(CLOCK_REALTIME, &now_rt);
    now_rt.tv_sec += 2;
    if (pthread_mutex_timedlock(&hp_tx.drain_mu, &now_rt) == 0) {
        hp_tx_drain_all_locked();
        pthread_mutex_unlock(&hp_tx.drain_mu);
    } else {
        /* the detached drain thread is stuck in send(): count what is left */
        atomic_store_explicit(&hp_tx.dirty, 1, memory_order_relaxed);
    }
    (void)joined;
    hp_tx_send_status(final);
}

/* fork: the child starts a fresh transport; the parent's buffered records
 * are the parent's to send. Runs in the (single) forking thread. */
static void hp_tx_atfork_child(void) {
    pthread_mutex_init(&hp_tx.init_mu, NULL);
    pthread_mutex_init(&hp_tx.reg_mu, NULL);
    pthread_mutex_init(&hp_tx.drain_mu, NULL);
    pthread_mutex_init(&hp_tx.sock_mu, NULL);
    if (hp_tx.sock >= 0) close(hp_tx.sock);   /* the child's copy only */
    hp_tx.sock = -1;
    hp_tx.next_connect_ns = 0;
    hp_tx_ring_t *r = hp_tx.rings;
    while (r) {
        hp_tx_ring_t *next = r->next;
        free(r->rb.buf);
        free(r);
        r = next;
    }
    hp_tx.rings = NULL;
    hp_tx_tls_ring = NULL;
    hp_tx_tls_tid = 0;
    hp_tx_tls_busy = 0;
    pthread_setspecific(hp_tx.key, NULL);
    hp_tx.drain_started = 0;
    atomic_store(&hp_tx.stop, 0);
    atomic_store(&hp_tx.kick, 0);
    atomic_store(&hp_tx.pid, 0);
    atomic_store(&hp_tx.state, HP_TX_UNINIT);
    hp_tx_tls_ctr = NULL;
    memset(&hp_tx.ctr, 0, sizeof(hp_tx.ctr));
    atomic_store(&hp_tx.sent, 0); atomic_store(&hp_tx.bytes, 0);
    atomic_store(&hp_tx.dropped_lost, 0); atomic_store(&hp_tx.max_block_ns, 0);
    atomic_store(&hp_tx.threads, 0); atomic_store(&hp_tx.reconnects, 0);
    atomic_store(&hp_tx.send_errors, 0); atomic_store(&hp_tx.dirty, 0);
    hp_tx.last_status_ns = 0;
    hp_tx.image_ns = hp_tx_now();
}

/* exec: drain before the image is replaced (preloaded hooks only -- a
 * dlopen'ed library such as the OMPT tool does not interpose these; its
 * missing final status tells the collector instead). On a failed exec the
 * transport simply restarts on the next event. */
#ifndef HP_TX_NO_EXEC_WRAPPERS
static void hp_tx_before_exec(void) {
    if (!hp_tx_enabled()) return;
    /* A vfork() child shares the parent's memory and runs no atfork
     * handlers: touching the transport there would drain (and stop) the
     * PARENT's. Its pid differs from the cached one -- leave everything. */
    if ((pid_t)syscall(SYS_getpid) != hp_tx_pid()) return;
    hp_tx_shutdown(1);
}
static void hp_tx_after_failed_exec(void) {
    if ((pid_t)syscall(SYS_getpid) != hp_tx_pid()) return;
    pthread_mutex_lock(&hp_tx.init_mu);
    if (atomic_load(&hp_tx.state) == HP_TX_SYNC && hp_tx.want_ring) atomic_store(&hp_tx.state, HP_TX_UNINIT);
    pthread_mutex_unlock(&hp_tx.init_mu);
}

#define HP_TX_EXEC_FN(ret, name, params, args)                                 \
    ret name params {                                                           \
        static ret (*real) params = NULL;                                       \
        if (!real) real = (ret (*) params)dlsym(RTLD_NEXT, #name);              \
        if (!real) { errno = ENOSYS; return -1; }                               \
        hp_tx_before_exec();                                                    \
        ret rc = real args;                                                     \
        hp_tx_after_failed_exec();                                              \
        return rc;                                                              \
    }

HP_TX_EXEC_FN(int, execve, (const char *p, char *const a[], char *const e[]), (p, a, e))
HP_TX_EXEC_FN(int, execv, (const char *p, char *const a[]), (p, a))
HP_TX_EXEC_FN(int, execvp, (const char *f, char *const a[]), (f, a))
HP_TX_EXEC_FN(int, execvpe, (const char *f, char *const a[], char *const e[]), (f, a, e))

#define HP_TX_COLLECT_ARGS(first)                                              \
    va_list ap; size_t n = 1;                                                   \
    va_start(ap, arg); while (va_arg(ap, char *)) n++; va_end(ap);              \
    char **argv = (char **)alloca((n + 1) * sizeof(char *));                     \
    argv[0] = (char *)(first);                                                  \
    va_start(ap, arg); for (size_t i = 1; i <= n; i++) argv[i] = va_arg(ap, char *);

int execl(const char *path, const char *arg, ...) {
    HP_TX_COLLECT_ARGS(arg)
    va_end(ap);
    return execv(path, argv);
}
int execlp(const char *file, const char *arg, ...) {
    HP_TX_COLLECT_ARGS(arg)
    va_end(ap);
    return execvp(file, argv);
}
int execle(const char *path, const char *arg, ...) {
    HP_TX_COLLECT_ARGS(arg)
    char *const *envp = va_arg(ap, char *const *);
    va_end(ap);
    return execve(path, argv, envp);
}
#endif /* HP_TX_NO_EXEC_WRAPPERS */

#endif /* HP_TRANSPORT_H */
