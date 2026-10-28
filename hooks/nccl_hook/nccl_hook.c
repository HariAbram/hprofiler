/*
 * NCCL hook (LD_PRELOAD; no NCCL headers needed, symbols via dlsym).
 *
 * Wraps ncclAllReduce, Broadcast, Reduce, AllGather, ReduceScatter,
 * AllToAll, Send, Recv (category "nccl"; tags type=, bytes= (count x dtype
 * size), stream=, rank=, nranks=, commid=, and seq= on collectives),
 * GroupStart/GroupEnd (one ncclGroup span for the outermost pair) and
 * CommInitRank/InitRankConfig/InitAll/Split/Destroy/Abort.
 *
 * commid= names the communicator identically on every rank: the hash of the
 * ncclUniqueId all ranks passed to ncclCommInitRank (ncclCommSplit: of the
 * parent's id and the color; ncclCommInitAll: of this process and call).
 * seq= numbers the collectives of one communicator; NCCL requires every
 * rank to issue them in the same order, so (commid, seq) is the same call
 * on every rank -- criticalpath.py pairs collectives by it. pseq= numbers
 * the sends to (receives from) one peer on one communicator: NCCL pairs
 * them in that order, so the k-th send from a to b is the k-th receive on b
 * from a. group= names the ncclGroupStart/End an operation was issued in.
 *
 * Timing:
 *   default         a cudaEvent pair on the operation's stream; the wrapper
 *                   waits for the end event before returning, which
 *                   serializes each operation with the host (one span: host
 *                   start, device duration). Without usable CUDA events:
 *                   host wall-clock time, tagged timing=cpu. If the
 *                   end-event wait or elapsed-time query fails, no span.
 *   in a group      NCCL launches grouped operations at ncclGroupEnd, so an
 *                   event pair around one brackets nothing: each is the
 *                   host call (timing=deferred, side=cpu, group=); ncclGroupEnd times
 *                   the group's work with one event pair per stream it used
 *                   (ncclGroup spans with type=group_exec,stream=).
 *   HPROFILER_NCCL_ASYNC=1
 *                   no wait: a host span for the call (side=cpu, sid=) and
 *                   a device span (side=gpu, psid=, timing=async) resolved
 *                   later from the event pair, placed against a reference
 *                   event the hook synchronizes on its own empty stream
 *                   (ref_err= half the host interval around it, in ns).
 *                   cudaEventElapsedTime across streams can exceed the true
 *                   interval (CUDA documents it), so these placements are
 *                   estimates. Pending spans are resolved at later NCCL
 *                   calls and at exit (at most HP_NCCL_ASYNC_CAP pending;
 *                   more wait for the oldest); any not resolved are counted
 *                   in an nccl_async_lost instant.
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <time.h>
#include <pthread.h>
#include <dlfcn.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/syscall.h>

/* ── Minimal NCCL type stubs (no nccl.h required) ───────────────────── */
typedef int ncclResult_t;
typedef void *ncclComm_t;
typedef void *cudaStream_t;
typedef void *cudaEvent_t;
typedef int   ncclDataType_t;
typedef int   ncclRedOp_t;

#define ncclSuccess 0

/* Index = ncclDataType_t; includes the NCCL >= 2.20 FP8 types (1 byte).
 * Unknown types fall back to 4 bytes. */
static const size_t _nccl_dtype_sizes[] = {
    1,  /* ncclInt8    / ncclChar   */
    1,  /* ncclUint8               */
    4,  /* ncclInt32   / ncclInt    */
    4,  /* ncclUint32              */
    8,  /* ncclInt64               */
    8,  /* ncclUint64              */
    2,  /* ncclFloat16 / ncclHalf   */
    4,  /* ncclFloat32 / ncclFloat  */
    8,  /* ncclFloat64 / ncclDouble */
    2,  /* ncclBfloat16            */
    1,  /* ncclFp8E4M3 (NCCL >= 2.20) */
    1,  /* ncclFp8E5M2 (NCCL >= 2.20) */
};
static size_t nccl_dtype_sz(ncclDataType_t dt) {
    if (dt >= 0 && (size_t)dt < sizeof(_nccl_dtype_sizes)/sizeof(_nccl_dtype_sizes[0]))
        return _nccl_dtype_sizes[dt];
    return 4;  /* fallback for datatypes newer than this table */
}

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

/* Events go through the shared per-thread ring transport (its own
 * connection, separate from cuda_hook's). */
#include "../common/hp_transport.h"
#include "../common/callstack.h"
#include "../common/hp_module.h"
#include "../common/hp_ids.h"

static pid_t gettid_compat(void) { return hp_tx_tid(); }

static void emit_span_nostack(const char *cat, pid_t tid, uint64_t start_ns,
                              uint64_t dur_ns, const char *name, const char *extra) {
    if (!hp_tx_enabled()) return;
    hp_tx_emitf("span:%s:%d:%d:%llu:%llu:%s:%s\n", cat, (int)hp_tx_pid(), (int)tid,
                (unsigned long long)start_ns, (unsigned long long)dur_ns,
                name, extra ? extra : "");
}

static void emit_span(const char *cat, pid_t tid, uint64_t start_ns,
                      uint64_t dur_ns, const char *name, const char *extra) {
    if (!hp_tx_enabled()) return;
    emit_span_nostack(cat, tid, start_ns, dur_ns, name, extra);
    emit_callstack(start_ns);
}

/* Appends printf-style text to extra (cap bytes); drops it if it does not fit. */
static void append_tag(char *extra, size_t cap, const char *fmt, ...) __attribute__((format(printf, 3, 4)));
#include <stdarg.h>
static void append_tag(char *extra, size_t cap, const char *fmt, ...) {
    size_t len = strlen(extra);
    if (len + 1 >= cap) return;
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(extra + len, cap - len, fmt, ap);
    va_end(ap);
    if (n < 0 || (size_t)n >= cap - len) extra[len] = '\0';
}

/* Appends ",timing=cpu": the span was timed by the host clock around the
 * call (no usable CUDA events), not by an event pair. */
static void mark_cpu_fallback(char *extra, size_t cap) {
    size_t len = strlen(extra);
    if (len == 0) {
        snprintf(extra, cap, "timing=cpu");
    } else if (len + 12 < cap) {
        snprintf(extra + len, cap - len, ",timing=cpu");
    }
}

/* ── GPU event pair helpers (borrowed from cuda_hook pattern) ────────── */
typedef int (*fn_EvCreate_t)(cudaEvent_t*);
typedef int (*fn_EvRecord_t)(cudaEvent_t, cudaStream_t);
typedef int (*fn_EvElapsed_t)(float*, cudaEvent_t, cudaEvent_t);
typedef int (*fn_EvDestroy_t)(cudaEvent_t);
typedef int (*fn_EvSync_t)(cudaEvent_t);

static fn_EvCreate_t  f_evCreate  = NULL;
static fn_EvRecord_t  f_evRecord  = NULL;
static fn_EvElapsed_t f_evElapsed = NULL;
static fn_EvDestroy_t f_evDestroy = NULL;
static fn_EvSync_t    f_evSync    = NULL;
typedef int (*fn_EvQuery_t)(cudaEvent_t);
typedef int (*fn_StreamCreateWithFlags_t)(cudaStream_t*, unsigned int);
static fn_EvQuery_t               f_evQuery = NULL;
static fn_StreamCreateWithFlags_t f_streamCreate = NULL;

static int ev_ok(void) {
    if (!f_evCreate) {
        f_evCreate  = (fn_EvCreate_t) dlsym(RTLD_DEFAULT, "cudaEventCreate");
        f_evRecord  = (fn_EvRecord_t) dlsym(RTLD_DEFAULT, "cudaEventRecord");
        f_evElapsed = (fn_EvElapsed_t)dlsym(RTLD_DEFAULT, "cudaEventElapsedTime");
        f_evDestroy = (fn_EvDestroy_t)dlsym(RTLD_DEFAULT, "cudaEventDestroy");
        f_evSync    = (fn_EvSync_t)   dlsym(RTLD_DEFAULT, "cudaEventSynchronize");
        f_evQuery   = (fn_EvQuery_t)  dlsym(RTLD_DEFAULT, "cudaEventQuery");
        f_streamCreate = (fn_StreamCreateWithFlags_t)dlsym(RTLD_DEFAULT, "cudaStreamCreateWithFlags");
    }
    return f_evCreate && f_evRecord && f_evElapsed && f_evDestroy && f_evSync;
}

/* ── Grouped operations (per thread) ────────────────────────────────── */
#define HP_NCCL_GROUP_STREAMS 8
static __thread uint64_t     _group_start = 0;
static __thread int          _group_depth = 0;
static __thread uint64_t     _group_id = 0;
static __thread int          _group_nops = 0;
static __thread cudaStream_t _group_streams[HP_NCCL_GROUP_STREAMS];
static __thread int          _group_nstreams = 0;
static __thread int          _group_streams_lost = 0;
static uint64_t              g_group_seq = 0;

static void group_note_stream(cudaStream_t stream) {
    for (int i = 0; i < _group_nstreams; i++)
        if (_group_streams[i] == stream) return;
    if (_group_nstreams < HP_NCCL_GROUP_STREAMS) _group_streams[_group_nstreams++] = stream;
    else _group_streams_lost = 1;
}

/* ── Asynchronous timing (HPROFILER_NCCL_ASYNC=1) ───────────────────── */
static int async_enabled(void) {
    static int on = -1;
    if (on < 0) {
        const char *e = getenv("HPROFILER_NCCL_ASYNC");
        on = e && e[0] == '1';
    }
    return on;
}

/* A reference event: synchronized on the hook's own empty non-blocking
 * stream, so it completes as soon as it is recorded; host time = the
 * midpoint of [before record, after sync]. Renewed every second: float
 * milliseconds from cudaEventElapsedTime lose precision with distance. */
#define HP_NCCL_REFS 16
#define HP_NCCL_REF_RENEW_NS 1000000000ULL
typedef struct { cudaEvent_t ev; uint64_t ns, err; int pending, live; } RefEvent;
static RefEvent        g_refs[HP_NCCL_REFS];
static int             g_ref_cur = -1;
static cudaStream_t    g_ref_stream = NULL;

#define HP_NCCL_ASYNC_CAP 4096
typedef struct { cudaEvent_t s, e; int ref; pid_t tid; char *name; char *extra; } AsyncOp;
static AsyncOp         g_async[HP_NCCL_ASYNC_CAP];
static size_t          g_async_head = 0, g_async_n = 0;
static uint64_t        g_async_lost = 0;
static pthread_mutex_t g_async_mutex = PTHREAD_MUTEX_INITIALIZER;

static void ref_release(int r) {
    if (r < 0) return;
    if (--g_refs[r].pending == 0 && r != g_ref_cur && g_refs[r].live) {
        f_evDestroy(g_refs[r].ev);
        g_refs[r].live = 0;
    }
}

/* The current reference (renewed when old); -1 if none can be made.
 * Caller holds g_async_mutex. */
static int ref_current(void) {
    uint64_t now = now_ns();
    if (g_ref_cur >= 0 && now - g_refs[g_ref_cur].ns < HP_NCCL_REF_RENEW_NS) return g_ref_cur;
    if (!g_ref_stream && (!f_streamCreate || f_streamCreate(&g_ref_stream, 0x1 /* NonBlocking */) != 0))
        return g_ref_cur;
    int slot = -1;
    for (int i = 0; i < HP_NCCL_REFS; i++)
        if (!g_refs[i].live) { slot = i; break; }
    if (slot < 0) return g_ref_cur;                    /* all still referenced: keep the old one */
    cudaEvent_t ev = NULL;
    uint64_t a = now_ns();
    if (f_evCreate(&ev) != 0) return g_ref_cur;
    if (f_evRecord(ev, g_ref_stream) != 0 || f_evSync(ev) != 0) { f_evDestroy(ev); return g_ref_cur; }
    uint64_t b = now_ns();
    int old = g_ref_cur;
    g_refs[slot] = (RefEvent){ev, a + (b - a) / 2, (b - a) / 2, 0, 1};
    g_ref_cur = slot;
    if (old >= 0 && g_refs[old].pending == 0 && g_refs[old].live) {
        f_evDestroy(g_refs[old].ev);
        g_refs[old].live = 0;
    }
    return slot;
}

/* Resolves and emits the oldest pending op. wait: block on its end event;
 * otherwise only if it is complete. Returns 1 if one was taken off the
 * queue. Caller holds g_async_mutex. */
static int async_pop(int wait) {
    if (!g_async_n) return 0;
    AsyncOp *op = &g_async[g_async_head];
    int done = wait ? f_evSync(op->e) == 0 : f_evQuery(op->e) == 0;
    if (!done && !wait) return 0;
    float to_s = 0.0f, dur = 0.0f;
    RefEvent *r = op->ref >= 0 ? &g_refs[op->ref] : NULL;
    if (done && r && f_evElapsed(&to_s, r->ev, op->s) == 0 && f_evElapsed(&dur, op->s, op->e) == 0
        && to_s >= 0.0f && dur >= 0.0f) {
        append_tag(op->extra, 384, ",ref_err=%llu", (unsigned long long)r->err);
        emit_span_nostack("nccl", op->tid, r->ns + (uint64_t)(to_s * 1e6f), (uint64_t)(dur * 1e6f),
                          op->name, op->extra);
    } else {
        g_async_lost++;
    }
    f_evDestroy(op->s);
    f_evDestroy(op->e);
    ref_release(op->ref);
    free(op->name);
    free(op->extra);
    g_async_head = (g_async_head + 1) % HP_NCCL_ASYNC_CAP;
    g_async_n--;
    return 1;
}

static void async_drain(int wait) {
    pthread_mutex_lock(&g_async_mutex);
    while (async_pop(wait)) { }
    pthread_mutex_unlock(&g_async_mutex);
}

/* At exit, from an atexit handler registered at the first asynchronous op
 * (after CUDA's own, so it runs before CUDA is torn down). */
static void async_flush_at_exit(void) {
    async_drain(1);
    if (g_async_lost && hp_tx_enabled()) {
        char extra[48];
        snprintf(extra, sizeof(extra), "count=%llu", (unsigned long long)g_async_lost);
        hp_tx_emitf("inst:nccl:%d:%d:%llu:nccl_async_lost:%s\n", (int)hp_tx_pid(), (int)hp_tx_tid(),
                    (unsigned long long)now_ns(), extra);
    }
}

/* The reference an op's start event will be measured from: taken (and
 * pinned) BEFORE that event is recorded -- a reference recorded after it
 * gives a negative offset. Resolves completed ops first. -1: none. */
static int async_ref_acquire(void) {
    pthread_mutex_lock(&g_async_mutex);
    static int registered = 0;
    if (!registered) { registered = 1; atexit(async_flush_at_exit); }
    while (async_pop(0)) { }
    int ref = ref_current();
    if (ref >= 0) g_refs[ref].pending++;
    pthread_mutex_unlock(&g_async_mutex);
    return ref;
}

static void async_ref_drop(int ref) {
    if (ref < 0) return;
    pthread_mutex_lock(&g_async_mutex);
    ref_release(ref);
    pthread_mutex_unlock(&g_async_mutex);
}

/* Queues an op whose events are recorded; takes ownership of s, e and its
 * pinned reference. */
static void async_push(cudaEvent_t s, cudaEvent_t e, int ref, const char *name, const char *extra) {
    pthread_mutex_lock(&g_async_mutex);
    if (g_async_n == HP_NCCL_ASYNC_CAP) async_pop(1);
    char *n = strdup(name), *x = (char *)malloc(384);
    if (!n || !x) {
        g_async_lost++;
        f_evDestroy(s); f_evDestroy(e);
        ref_release(ref);
        free(n); free(x);
        pthread_mutex_unlock(&g_async_mutex);
        return;
    }
    snprintf(x, 384, "%s", extra);
    g_async[(g_async_head + g_async_n) % HP_NCCL_ASYNC_CAP] = (AsyncOp){s, e, ref, gettid_compat(), n, x};
    g_async_n++;
    pthread_mutex_unlock(&g_async_mutex);
}

static uint64_t next_host_sid(void) {
    static uint64_t seq = 0;
    /* this hook's span ids: pid in the high bits (an id space of its own,
     * see hp_ids.h's HP_ID_NCCL) */
    return hp_span_id(HP_ID_NCCL, (uint32_t)getpid(), (uint32_t)__atomic_add_fetch(&seq, 1, __ATOMIC_RELAXED));
}

/* ── One operation's timing ─────────────────────────────────────────── */
enum { OP_SYNC, OP_DEFERRED, OP_ASYNC, OP_HOST };
typedef struct { uint64_t t0; cudaEvent_t s, e; int mode, ref; } OpTiming;

/* t0 is taken before the start event is recorded, so the host timestamp
 * never follows the GPU start. */
static void op_begin(OpTiming *o, cudaStream_t stream) {
    o->t0 = now_ns();
    o->s = o->e = NULL;
    o->ref = -1;
    if (_group_depth > 0) {
        o->mode = OP_DEFERRED;
        group_note_stream(stream);
        return;
    }
    int async = ev_ok() && async_enabled() && f_evQuery && f_streamCreate;
    if (async) o->ref = async_ref_acquire();
    if (!ev_ok() || f_evCreate(&o->s) != 0 || f_evCreate(&o->e) != 0 || f_evRecord(o->s, stream) != 0) {
        o->mode = OP_HOST;
        return;
    }
    o->mode = async && o->ref >= 0 ? OP_ASYNC : OP_SYNC;
}

static void op_destroy(OpTiming *o) {
    if (o->s) f_evDestroy(o->s);
    if (o->e) f_evDestroy(o->e);
    o->s = o->e = NULL;
    async_ref_drop(o->ref);
    o->ref = -1;
}

static void op_end(OpTiming *o, const char *name, char *extra, size_t cap, cudaStream_t stream,
                   ncclResult_t ret) {
    uint64_t t1 = now_ns();
    switch (o->mode) {
    case OP_DEFERRED:
        append_tag(extra, cap, ",group=%llu,timing=deferred,side=cpu", (unsigned long long)_group_id);
        _group_nops++;
        emit_span("nccl", gettid_compat(), o->t0, t1 - o->t0, name, extra);
        return;
    case OP_SYNC: {
        float ms = 0.0f;
        f_evRecord(o->e, stream);
        if (f_evSync(o->e) == 0 && f_evElapsed(&ms, o->s, o->e) == 0 && ms >= 0.0f)
            emit_span("nccl", gettid_compat(), o->t0, (uint64_t)(ms * 1e6f), name, extra);
        op_destroy(o);
        return;
    }
    case OP_ASYNC:
        if (ret == ncclSuccess && f_evRecord(o->e, stream) == 0) {
            uint64_t sid = next_host_sid();
            char dev[384];
            snprintf(dev, sizeof(dev), "%s,side=gpu,timing=async,psid=%llu", extra, (unsigned long long)sid);
            append_tag(extra, cap, ",side=cpu,sid=%llu", (unsigned long long)sid);
            emit_span("nccl", gettid_compat(), o->t0, t1 - o->t0, name, extra);
            async_push(o->s, o->e, o->ref, name, dev);
            return;
        }
        break;          /* the call failed or its end could not be recorded */
    default:
        break;
    }
    op_destroy(o);
    mark_cpu_fallback(extra, cap);
    emit_span("nccl", gettid_compat(), o->t0, t1 - o->t0, name, extra);
}

#define NCCL_OP_BEGIN(stream)  OpTiming _ot; op_begin(&_ot, (stream));
#define NCCL_OP_END(name, extra, stream)  op_end(&_ot, (name), (extra), sizeof(extra), (stream), ret);

/* ── NCCL comm rank/world-size query ─────────────────────────────────── */
typedef ncclResult_t (*fn_CommUserRank_t)(ncclComm_t, int*);
typedef ncclResult_t (*fn_CommCount_t)   (ncclComm_t, int*);
static fn_CommUserRank_t f_commRank  = NULL;
static fn_CommCount_t    f_commCount = NULL;

static void comm_meta(ncclComm_t comm, int *rank, int *nranks) {
    *rank = -1; *nranks = -1;
    if (!f_commRank)
        f_commRank  = (fn_CommUserRank_t)dlsym(RTLD_DEFAULT, "ncclCommUserRank");
    if (!f_commCount)
        f_commCount = (fn_CommCount_t)   dlsym(RTLD_DEFAULT, "ncclCommCount");
    if (!f_commRank)
        f_commRank  = (fn_CommUserRank_t)dlsym(RTLD_NEXT,    "ncclCommUserRank");
    if (!f_commCount)
        f_commCount = (fn_CommCount_t)   dlsym(RTLD_NEXT,    "ncclCommCount");
    if (f_commRank  && comm) f_commRank(comm, rank);
    if (f_commCount && comm) f_commCount(comm, nranks);
}

/* ── Communicator identity: comm handle -> (commid, collective count) ─── */
/* pseq: per peer, the sends to it [2 * peer] and receives from it
 * [2 * peer + 1] so far (allocated at the first point-to-point call). */
typedef struct { ncclComm_t comm; uint64_t id; uint64_t seq; uint64_t *pseq; int npeers; } CommEntry;
static CommEntry      *g_comms;
static size_t          g_ncomms, g_capcomms;
static pthread_mutex_t g_comm_mutex = PTHREAD_MUTEX_INITIALIZER;

static void comm_register(ncclComm_t comm, uint64_t id) {
    if (!comm) return;
    pthread_mutex_lock(&g_comm_mutex);
    for (size_t i = 0; i < g_ncomms; i++) {
        if (g_comms[i].comm == comm) {          /* a reused handle: a new communicator */
            g_comms[i].id = id;
            g_comms[i].seq = 0;
            free(g_comms[i].pseq);
            g_comms[i].pseq = NULL;
            g_comms[i].npeers = 0;
            pthread_mutex_unlock(&g_comm_mutex);
            return;
        }
    }
    if (g_ncomms == g_capcomms) {
        size_t cap = g_capcomms ? g_capcomms * 2 : 16;
        CommEntry *p = (CommEntry *)realloc(g_comms, cap * sizeof(*p));
        if (!p) { pthread_mutex_unlock(&g_comm_mutex); return; }
        g_comms = p;
        g_capcomms = cap;
    }
    g_comms[g_ncomms++] = (CommEntry){comm, id, 0, NULL, 0};
    pthread_mutex_unlock(&g_comm_mutex);
}

static void comm_forget(ncclComm_t comm) {
    pthread_mutex_lock(&g_comm_mutex);
    for (size_t i = 0; i < g_ncomms; i++) {
        if (g_comms[i].comm == comm) {
            free(g_comms[i].pseq);
            g_comms[i] = g_comms[--g_ncomms];
            break;
        }
    }
    pthread_mutex_unlock(&g_comm_mutex);
}

/* ",commid=<id>[,seq=<n>]" into buf (empty for a communicator created
 * before the hook saw it); collective != 0 counts this call in seq=. */
static const char *comm_ident(ncclComm_t comm, int collective, char *buf, size_t cap) {
    buf[0] = '\0';
    pthread_mutex_lock(&g_comm_mutex);
    for (size_t i = 0; i < g_ncomms; i++) {
        if (g_comms[i].comm != comm) continue;
        if (collective)
            snprintf(buf, cap, ",commid=%016llx,seq=%llu", (unsigned long long)g_comms[i].id,
                     (unsigned long long)++g_comms[i].seq);
        else
            snprintf(buf, cap, ",commid=%016llx", (unsigned long long)g_comms[i].id);
        break;
    }
    pthread_mutex_unlock(&g_comm_mutex);
    return buf;
}

/* Appends ",pseq=<n>": this is the n-th send to (recv 0) / receive from
 * (recv 1) peer on comm. Nothing for a communicator the hook did not see
 * created, or a peer outside [0, nranks). */
static void p2p_seq(ncclComm_t comm, int peer, int recv, int nranks, char *buf, size_t cap) {
    uint64_t n = 0;
    pthread_mutex_lock(&g_comm_mutex);
    for (size_t i = 0; i < g_ncomms; i++) {
        CommEntry *c = &g_comms[i];
        if (c->comm != comm) continue;
        if (!c->pseq && nranks > 0) {
            c->pseq = (uint64_t *)calloc((size_t)nranks * 2, sizeof(uint64_t));
            c->npeers = c->pseq ? nranks : 0;
        }
        if (peer >= 0 && peer < c->npeers) n = ++c->pseq[2 * peer + (recv ? 1 : 0)];
        break;
    }
    pthread_mutex_unlock(&g_comm_mutex);
    if (n) append_tag(buf, cap, ",pseq=%llu", (unsigned long long)n);
}

static uint64_t comm_id_of(ncclComm_t comm) {
    uint64_t id = 0;
    pthread_mutex_lock(&g_comm_mutex);
    for (size_t i = 0; i < g_ncomms; i++)
        if (g_comms[i].comm == comm) { id = g_comms[i].id; break; }
    pthread_mutex_unlock(&g_comm_mutex);
    return id;
}

/* ── NCCL communicator lifecycle ────────────────────────────────────── */
typedef struct { char _opaque[128]; } ncclUniqueId;

ncclResult_t ncclCommInitRank(ncclComm_t *comm, int nranks,
                               ncclUniqueId clique_id, int rank) {
    typedef ncclResult_t (*fn_t)(ncclComm_t*, int, ncclUniqueId, int);
    static fn_t real = NULL;
    if (!real) real = (fn_t)dlsym(RTLD_NEXT, "ncclCommInitRank");
    if (!real) return -1;
    char extra[64];
    snprintf(extra, sizeof(extra), "type=comm_init,nranks=%d,rank=%d", nranks, rank);
    uint64_t t0 = now_ns();
    ncclResult_t ret = real(comm, nranks, clique_id, rank);
    emit_span("nccl", gettid_compat(), t0, now_ns()-t0, "ncclCommInitRank", extra);
    if (ret == ncclSuccess && comm) comm_register(*comm, hp_fnv1a64(&clique_id, sizeof(clique_id)));
    return ret;
}

ncclResult_t ncclCommInitRankConfig(ncclComm_t *comm, int nranks, ncclUniqueId clique_id,
                                     int rank, void *config) {
    typedef ncclResult_t (*fn_t)(ncclComm_t*, int, ncclUniqueId, int, void*);
    static fn_t real = NULL;
    if (!real) real = (fn_t)dlsym(RTLD_NEXT, "ncclCommInitRankConfig");
    if (!real) return -1;
    char extra[64];
    snprintf(extra, sizeof(extra), "type=comm_init,nranks=%d,rank=%d", nranks, rank);
    uint64_t t0 = now_ns();
    ncclResult_t ret = real(comm, nranks, clique_id, rank, config);
    emit_span("nccl", gettid_compat(), t0, now_ns()-t0, "ncclCommInitRankConfig", extra);
    if (ret == ncclSuccess && comm) comm_register(*comm, hp_fnv1a64(&clique_id, sizeof(clique_id)));
    return ret;
}

/* Every rank with the same color gets the same child: id = hash(parent id,
 * color), equal on all of them when the parent's is. */
ncclResult_t ncclCommSplit(ncclComm_t comm, int color, int key, ncclComm_t *newcomm, void *config) {
    typedef ncclResult_t (*fn_t)(ncclComm_t, int, int, ncclComm_t*, void*);
    static fn_t real = NULL;
    if (!real) real = (fn_t)dlsym(RTLD_NEXT, "ncclCommSplit");
    if (!real) return -1;
    uint64_t t0 = now_ns();
    ncclResult_t ret = real(comm, color, key, newcomm, config);
    char extra[48];
    snprintf(extra, sizeof(extra), "type=comm_split,color=%d", color);
    emit_span("nccl", gettid_compat(), t0, now_ns()-t0, "ncclCommSplit", extra);
    uint64_t parent = comm_id_of(comm);
    if (ret == ncclSuccess && newcomm && *newcomm && parent) {
        uint64_t h = hp_fnv1a64_update(HP_FNV1A64_INIT, &parent, sizeof(parent));
        comm_register(*newcomm, hp_fnv1a64_update(h, &color, sizeof(color)));
    }
    return ret;
}

ncclResult_t ncclCommInitAll(ncclComm_t *comm, int ndev, const int *devlist) {
    typedef ncclResult_t (*fn_t)(ncclComm_t*, int, const int*);
    static fn_t real = NULL;
    if (!real) real = (fn_t)dlsym(RTLD_NEXT, "ncclCommInitAll");
    if (!real) return -1;
    char extra[48];
    snprintf(extra, sizeof(extra), "type=comm_init_all,ndev=%d", ndev);
    uint64_t t0 = now_ns();
    ncclResult_t ret = real(comm, ndev, devlist);
    emit_span("nccl", gettid_compat(), t0, now_ns()-t0, "ncclCommInitAll", extra);
    if (ret == ncclSuccess && comm) {
        /* one clique inside this process: (pid, call number) names it */
        static uint64_t calls = 0;
        uint64_t key[2] = {(uint64_t)getpid(), __atomic_add_fetch(&calls, 1, __ATOMIC_RELAXED)};
        uint64_t id = hp_fnv1a64(key, sizeof(key));
        for (int i = 0; i < ndev; i++) comm_register(comm[i], id);
    }
    return ret;
}

ncclResult_t ncclCommDestroy(ncclComm_t comm) {
    typedef ncclResult_t (*fn_t)(ncclComm_t);
    static fn_t real = NULL;
    if (!real) real = (fn_t)dlsym(RTLD_NEXT, "ncclCommDestroy");
    if (!real) return -1;
    uint64_t t0 = now_ns();
    ncclResult_t ret = real(comm);
    emit_span("nccl", gettid_compat(), t0, now_ns()-t0,
              "ncclCommDestroy", "type=comm_destroy");
    comm_forget(comm);
    return ret;
}

ncclResult_t ncclCommAbort(ncclComm_t comm) {
    typedef ncclResult_t (*fn_t)(ncclComm_t);
    static fn_t real = NULL;
    if (!real) real = (fn_t)dlsym(RTLD_NEXT, "ncclCommAbort");
    if (!real) return -1;
    ncclResult_t ret = real(comm);
    comm_forget(comm);
    return ret;
}

/* ── Stream id ───────────────────────────────────────────────────────────
 * Pointer hash identical to cuda_hook.c's get_stream_id(), so both hooks
 * report the same stream=N for the same handle (0 = default stream). */
static int stream_id(cudaStream_t s) {
    if (!s) return 0;
    uint64_t v = (uint64_t)(uintptr_t)s;
    v ^= v >> 33; v *= 0xff51afd7ed558ccdULL;
    v ^= v >> 33; v *= 0xc4ceb9fe1a85ec53ULL;
    v ^= v >> 33;
    return (int)(v % 999983) + 1;
}

/* ── NCCL collectives ────────────────────────────────────────────────── */

ncclResult_t ncclAllReduce(const void *sb, void *rb, size_t count,
                            ncclDataType_t dt, ncclRedOp_t op,
                            ncclComm_t comm, cudaStream_t stream) {
    typedef ncclResult_t (*fn_t)(const void*, void*, size_t, ncclDataType_t,
                                  ncclRedOp_t, ncclComm_t, cudaStream_t);
    static fn_t real = NULL;
    if (!real) real = (fn_t)dlsym(RTLD_NEXT, "ncclAllReduce");
    if (!real) return -1;
    size_t nb = count * nccl_dtype_sz(dt);
    int rank = -1, nranks = -1; comm_meta(comm, &rank, &nranks);
    char cid[64]; comm_ident(comm, 1, cid, sizeof(cid));
    char extra[256]; snprintf(extra, sizeof(extra),
        "type=allreduce,bytes=%zu,stream=%d,rank=%d,nranks=%d%s",
        nb, stream_id(stream), rank, nranks, cid);
    NCCL_OP_BEGIN(stream)
    ncclResult_t ret = real(sb, rb, count, dt, op, comm, stream);
    NCCL_OP_END("ncclAllReduce", extra, stream)
    return ret;
}

ncclResult_t ncclBroadcast(const void *sb, void *rb, size_t count,
                            ncclDataType_t dt, int root,
                            ncclComm_t comm, cudaStream_t stream) {
    typedef ncclResult_t (*fn_t)(const void*, void*, size_t, ncclDataType_t,
                                  int, ncclComm_t, cudaStream_t);
    static fn_t real = NULL;
    if (!real) real = (fn_t)dlsym(RTLD_NEXT, "ncclBroadcast");
    if (!real) return -1;
    size_t nb = count * nccl_dtype_sz(dt);
    int rank = -1, nranks = -1; comm_meta(comm, &rank, &nranks);
    char cid[64]; comm_ident(comm, 1, cid, sizeof(cid));
    char extra[256]; snprintf(extra, sizeof(extra),
        "type=broadcast,bytes=%zu,root=%d,stream=%d,rank=%d,nranks=%d%s",
        nb, root, stream_id(stream), rank, nranks, cid);
    NCCL_OP_BEGIN(stream)
    ncclResult_t ret = real(sb, rb, count, dt, root, comm, stream);
    NCCL_OP_END("ncclBroadcast", extra, stream)
    return ret;
}

ncclResult_t ncclReduce(const void *sb, void *rb, size_t count,
                         ncclDataType_t dt, ncclRedOp_t op, int root,
                         ncclComm_t comm, cudaStream_t stream) {
    typedef ncclResult_t (*fn_t)(const void*, void*, size_t, ncclDataType_t,
                                  ncclRedOp_t, int, ncclComm_t, cudaStream_t);
    static fn_t real = NULL;
    if (!real) real = (fn_t)dlsym(RTLD_NEXT, "ncclReduce");
    if (!real) return -1;
    size_t nb = count * nccl_dtype_sz(dt);
    int rank = -1, nranks = -1; comm_meta(comm, &rank, &nranks);
    char cid[64]; comm_ident(comm, 1, cid, sizeof(cid));
    char extra[256]; snprintf(extra, sizeof(extra),
        "type=reduce,bytes=%zu,root=%d,stream=%d,rank=%d,nranks=%d%s",
        nb, root, stream_id(stream), rank, nranks, cid);
    NCCL_OP_BEGIN(stream)
    ncclResult_t ret = real(sb, rb, count, dt, op, root, comm, stream);
    NCCL_OP_END("ncclReduce", extra, stream)
    return ret;
}

ncclResult_t ncclAllGather(const void *sb, void *rb, size_t sendcount,
                            ncclDataType_t dt,
                            ncclComm_t comm, cudaStream_t stream) {
    typedef ncclResult_t (*fn_t)(const void*, void*, size_t, ncclDataType_t,
                                  ncclComm_t, cudaStream_t);
    static fn_t real = NULL;
    if (!real) real = (fn_t)dlsym(RTLD_NEXT, "ncclAllGather");
    if (!real) return -1;
    size_t nb = sendcount * nccl_dtype_sz(dt);
    int rank = -1, nranks = -1; comm_meta(comm, &rank, &nranks);
    char cid[64]; comm_ident(comm, 1, cid, sizeof(cid));
    char extra[256]; snprintf(extra, sizeof(extra),
        "type=allgather,bytes=%zu,stream=%d,rank=%d,nranks=%d%s",
        nb, stream_id(stream), rank, nranks, cid);
    NCCL_OP_BEGIN(stream)
    ncclResult_t ret = real(sb, rb, sendcount, dt, comm, stream);
    NCCL_OP_END("ncclAllGather", extra, stream)
    return ret;
}

ncclResult_t ncclReduceScatter(const void *sb, void *rb, size_t recvcount,
                                ncclDataType_t dt, ncclRedOp_t op,
                                ncclComm_t comm, cudaStream_t stream) {
    typedef ncclResult_t (*fn_t)(const void*, void*, size_t, ncclDataType_t,
                                  ncclRedOp_t, ncclComm_t, cudaStream_t);
    static fn_t real = NULL;
    if (!real) real = (fn_t)dlsym(RTLD_NEXT, "ncclReduceScatter");
    if (!real) return -1;
    size_t nb = recvcount * nccl_dtype_sz(dt);
    int rank = -1, nranks = -1; comm_meta(comm, &rank, &nranks);
    char cid[64]; comm_ident(comm, 1, cid, sizeof(cid));
    char extra[256]; snprintf(extra, sizeof(extra),
        "type=reduce_scatter,bytes=%zu,stream=%d,rank=%d,nranks=%d%s",
        nb, stream_id(stream), rank, nranks, cid);
    NCCL_OP_BEGIN(stream)
    ncclResult_t ret = real(sb, rb, recvcount, dt, op, comm, stream);
    NCCL_OP_END("ncclReduceScatter", extra, stream)
    return ret;
}

ncclResult_t ncclSend(const void *sb, size_t count, ncclDataType_t dt,
                       int peer, ncclComm_t comm, cudaStream_t stream) {
    typedef ncclResult_t (*fn_t)(const void*, size_t, ncclDataType_t, int,
                                  ncclComm_t, cudaStream_t);
    static fn_t real = NULL;
    if (!real) real = (fn_t)dlsym(RTLD_NEXT, "ncclSend");
    if (!real) return -1;
    size_t nb = count * nccl_dtype_sz(dt);
    int rank = -1, nranks = -1; comm_meta(comm, &rank, &nranks);
    char cid[64]; comm_ident(comm, 0, cid, sizeof(cid));
    char extra[256]; snprintf(extra, sizeof(extra),
        "type=send,bytes=%zu,peer=%d,stream=%d,rank=%d,nranks=%d%s",
        nb, peer, stream_id(stream), rank, nranks, cid);
    p2p_seq(comm, peer, 0, nranks, extra, sizeof(extra));
    NCCL_OP_BEGIN(stream)
    ncclResult_t ret = real(sb, count, dt, peer, comm, stream);
    NCCL_OP_END("ncclSend", extra, stream)
    return ret;
}

ncclResult_t ncclRecv(void *rb, size_t count, ncclDataType_t dt,
                       int peer, ncclComm_t comm, cudaStream_t stream) {
    typedef ncclResult_t (*fn_t)(void*, size_t, ncclDataType_t, int,
                                  ncclComm_t, cudaStream_t);
    static fn_t real = NULL;
    if (!real) real = (fn_t)dlsym(RTLD_NEXT, "ncclRecv");
    if (!real) return -1;
    size_t nb = count * nccl_dtype_sz(dt);
    int rank = -1, nranks = -1; comm_meta(comm, &rank, &nranks);
    char cid[64]; comm_ident(comm, 0, cid, sizeof(cid));
    char extra[256]; snprintf(extra, sizeof(extra),
        "type=recv,bytes=%zu,peer=%d,stream=%d,rank=%d,nranks=%d%s",
        nb, peer, stream_id(stream), rank, nranks, cid);
    p2p_seq(comm, peer, 1, nranks, extra, sizeof(extra));
    NCCL_OP_BEGIN(stream)
    ncclResult_t ret = real(rb, count, dt, peer, comm, stream);
    NCCL_OP_END("ncclRecv", extra, stream)
    return ret;
}

/* ── ncclAllToAll (N3 — NCCL 2.13+) ─────────────────────────────────── */
ncclResult_t ncclAllToAll(const void *sb, void *rb, size_t count,
                           ncclDataType_t dt,
                           ncclComm_t comm, cudaStream_t stream) {
    typedef ncclResult_t (*fn_t)(const void*, void*, size_t, ncclDataType_t,
                                  ncclComm_t, cudaStream_t);
    static fn_t real = NULL;
    if (!real) real = (fn_t)dlsym(RTLD_NEXT, "ncclAllToAll");
    if (!real) return -1;
    size_t nb = count * nccl_dtype_sz(dt);
    int rank = -1, nranks = -1; comm_meta(comm, &rank, &nranks);
    char cid[64]; comm_ident(comm, 1, cid, sizeof(cid));
    char extra[256]; snprintf(extra, sizeof(extra),
        "type=alltoall,bytes=%zu,stream=%d,rank=%d,nranks=%d%s",
        nb, stream_id(stream), rank, nranks, cid);
    NCCL_OP_BEGIN(stream)
    ncclResult_t ret = real(sb, rb, count, dt, comm, stream);
    NCCL_OP_END("ncclAllToAll", extra, stream)
    return ret;
}

/* ── Group boundaries ───────────────────────────────────────────────── */
/* NCCL defers the operations issued inside ncclGroupStart/End to
 * ncclGroupEnd (its documented group semantics), so each of them is the
 * host call (timing=deferred, see op_end) and the work is timed here: one
 * event pair per stream the group used (at most HP_NCCL_GROUP_STREAMS;
 * streams_lost=1 past that) around the real ncclGroupEnd -- with the
 * default timing, which waits for them. The outermost pair is also one
 * host span, type=group. Not checked on multi-GPU hardware. */
ncclResult_t ncclGroupStart(void) {
    typedef ncclResult_t (*fn_t)(void);
    static fn_t real = NULL;
    if (!real) real = (fn_t)dlsym(RTLD_NEXT, "ncclGroupStart");
    if (!real) return -1;
    if (_group_depth++ == 0) {
        _group_start = now_ns();
        _group_id = __atomic_add_fetch(&g_group_seq, 1, __ATOMIC_RELAXED);
        _group_nops = 0;
        _group_nstreams = 0;
        _group_streams_lost = 0;
    }
    return real();
}

ncclResult_t ncclGroupEnd(void) {
    typedef ncclResult_t (*fn_t)(void);
    static fn_t real = NULL;
    if (!real) real = (fn_t)dlsym(RTLD_NEXT, "ncclGroupEnd");
    if (!real) return -1;
    if (_group_depth <= 0) return real();     /* unbalanced: NCCL reports it */
    int outer = _group_depth == 1;
    int ns = outer ? _group_nstreams : 0;
    OpTiming t[HP_NCCL_GROUP_STREAMS];
    if (outer) _group_depth = 0;              /* the per-stream pairs are not grouped ops */
    for (int i = 0; i < ns; i++) op_begin(&t[i], _group_streams[i]);
    if (outer) _group_depth = 1;
    ncclResult_t ret = real();
    if (--_group_depth == 0 && _group_start) {
        uint64_t t1 = now_ns();
        char extra[160];
        for (int i = 0; i < ns; i++) {
            snprintf(extra, sizeof(extra), "type=group_exec,group=%llu,ops=%d,stream=%d%s",
                     (unsigned long long)_group_id, _group_nops, stream_id(_group_streams[i]),
                     _group_streams_lost ? ",streams_lost=1" : "");
            op_end(&t[i], "ncclGroup", extra, sizeof(extra), _group_streams[i], ret);
        }
        snprintf(extra, sizeof(extra), "type=group,group=%llu,ops=%d", (unsigned long long)_group_id,
                 _group_nops);
        emit_span("nccl", gettid_compat(), _group_start, t1 - _group_start, "ncclGroup", extra);
    }
    return ret;
}

/* ── Constructor ─────────────────────────────────────────────────────── */
__attribute__((constructor))
static void hprofiler_nccl_init(void) { hp_tx_init("nccl"); cs_init(); }

__attribute__((destructor))
static void hprofiler_nccl_fini(void) { hp_tx_shutdown(1); }
