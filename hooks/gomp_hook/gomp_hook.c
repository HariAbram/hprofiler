/*
 * GNU libgomp hook: OpenMP events for binaries linked against GCC's libgomp.
 * Typical distro/vendor libgomp builds export no OMPT symbols, so
 * ompt_tool.c never registers there; this hook interposes libgomp's public
 * GOMP_* ABI (stable since GCC 4.9 for these constructs) via LD_PRELOAD.
 *
 * GOMP_parallel: the (fn, data) handed to the real call is replaced by a
 * trampoline and a stack closure, so every participating thread times its
 * own share (one omp_parallel_region span per thread, like OMPT's implicit
 * task). The stack closure is safe because GOMP_parallel returns only after
 * every thread has finished fn and the region's implicit barrier.
 *
 * Covered: GOMP_parallel, loop start/end (dynamic, guided, runtime and their
 * nonmonotonic variants; static too, but GCC computes static ranges inline
 * and calls no GOMP_loop function), GOMP_barrier, critical sections
 * (anonymous and named), GOMP_single_start, tasks (GOMP_task with the
 * 10-argument signature of GCC 11+ -- older libgomp ignore the extra
 * arguments --, GOMP_taskwait, GOMP_taskwait_depend, GOMP_taskgroup_start/
 * end) and the omp_*_lock API. Not covered: the legacy GOMP_parallel_start/
 * _end ABI, taskloop, sections, doacross, ordered, target offload, the _ull
 * loop variants. Tasks and locks are recorded only when the real function
 * is libgomp's (libomp exports the same symbols, and its OMPT callbacks
 * already report them).
 *
 * Names and categories match ompt_tool.c ("openmp"; "sync" with an "omp_"
 * prefix for waits, which criticalpath.py's barrier rendezvous requires).
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <time.h>
#include <pthread.h>
#include <dlfcn.h>
#include <elf.h>
#include <link.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/syscall.h>

/* ── Transport (hooks/common/hp_transport.h) ───────────────────────────── */
#include "../common/hp_transport.h"

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

static pid_t gettid_compat(void) { return hp_tx_tid(); }

#include "../common/callstack.h"
#include "../common/codeptr_resolve.h"

static void emit_span(const char *cat, pid_t tid,
                      uint64_t start_ns, uint64_t dur_ns,
                      const char *name, const char *extra) {
    if (!hp_tx_enabled()) return;
    if (extra && *extra)
        hp_tx_emitf("span:%s:%d:%d:%llu:%llu:%s:%s\n", cat, (int)hp_tx_pid(), (int)tid,
                    (unsigned long long)start_ns, (unsigned long long)dur_ns, name, extra);
    else
        hp_tx_emitf("span:%s:%d:%d:%llu:%llu:%s\n", cat, (int)hp_tx_pid(), (int)tid,
                    (unsigned long long)start_ns, (unsigned long long)dur_ns, name);
    emit_callstack(start_ns);
}

/* ── Real-symbol resolution (dlsym RTLD_NEXT, cached once per symbol) ──── */
static void *real_sym(const char *name) {
    void *p = dlsym(RTLD_NEXT, name);
    if (!p) {
        fprintf(stderr, "hprofiler gomp_hook: could not resolve real %s: %s\n",
                name, dlerror());
    }
    return p;
}

/* Call-site tags (sym=/symfile= or lib=/offset=): the user's call site is
 * what the Source tab disassembles -- no ELF symbol is named
 * "omp_parallel_region". */
#define append_codeptr_tag hprofiler_append_codeptr_tag

/* Recursion guard. Defensive only: the wrapper's own bookkeeping makes no
 * OpenMP calls. */
static __thread int in_hook = 0;

static void emit_ctr(const char *name, int64_t value, const char *unit) {
    if (!hp_tx_enabled()) return;
    hp_tx_emitf("ctr:openmp:%d:%llu:%s:%lld:%s\n", (int)hp_tx_pid(), (unsigned long long)now_ns(), name,
                (long long)value, unit);
}

/* ── Generated tokens for addresses ──────────────────────────────────────
 * The trace never carries an address. A dependence's storage location
 * (depend clauses) and a lock become a token: a number the process hands
 * out the first time it sees the address, the same for the same address
 * (two tasks depending on one location name one token). Lock tokens are
 * retired at omp_init_*lock / omp_destroy_*lock -- a lock initialised later
 * at the same address is another lock and gets a new token; dependence
 * addresses are never retired (OpenMP matches dependences by address among
 * sibling tasks, so a reused address IS the same dependence). Each table
 * has TOK_SHARDS shards, each with its own mutex, of open-addressing slots
 * (16 B), allocated on first use and doubled up to a cap: dependences 16384
 * slots per shard (4 MiB in all), locks 1024 (256 KiB). Beyond the cap an
 * address gets no token: a dependence is counted (dep_unknown=, and
 * gomp_dep_unknown at exit), a lock is lock=?, never a guessed token. */
#define TOK_SHARDS 16
#define TOK_DEAD   UINT64_MAX
typedef struct { uintptr_t key; uint64_t tok; } TokSlot;     /* tok 0: empty, TOK_DEAD: removed */
typedef struct { pthread_mutex_t mu; TokSlot *s; uint32_t cap, live, dead; } TokShard;
typedef struct { TokShard sh[TOK_SHARDS]; uint32_t max_cap; uint64_t next, unknown; } TokTable;
#define TOK_SH { PTHREAD_MUTEX_INITIALIZER, NULL, 0, 0, 0 }
#define TOK_SHARD_INIT { TOK_SH, TOK_SH, TOK_SH, TOK_SH, TOK_SH, TOK_SH, TOK_SH, TOK_SH, \
                         TOK_SH, TOK_SH, TOK_SH, TOK_SH, TOK_SH, TOK_SH, TOK_SH, TOK_SH }
static TokTable g_dep_tok  = { TOK_SHARD_INIT, 16384, 0, 0 };
static TokTable g_lock_tok = { TOK_SHARD_INIT, 1024, 0, 0 };

static uint32_t tok_hash(uintptr_t k) {
    uint64_t v = (uint64_t)k;
    v ^= v >> 33; v *= 0xff51afd7ed558ccdULL; v ^= v >> 33; v *= 0xc4ceb9fe1a85ec53ULL; v ^= v >> 33;
    return (uint32_t)v;
}

/* Caller holds sh->mu. The slot holding `key` (*found = 1), else the slot
 * to insert it into (-1: none free). */
static int tok_find(const TokShard *sh, uintptr_t key, uint32_t h, int *found) {
    *found = 0;
    int dead = -1;
    for (uint32_t i = 0; i < sh->cap; i++) {
        uint32_t j = (h + i) & (sh->cap - 1);
        const TokSlot *t = &sh->s[j];
        if (t->tok == 0) return dead >= 0 ? dead : (int)j;
        if (t->tok == TOK_DEAD) { if (dead < 0) dead = (int)j; continue; }
        if (t->key == key) { *found = 1; return (int)j; }
    }
    return dead;
}

static int tok_grow(TokShard *sh, uint32_t max_cap) {
    uint32_t ncap = sh->cap ? sh->cap * 2 : 256;
    if (ncap > max_cap) return 0;
    TokSlot *ns = (TokSlot *)calloc(ncap, sizeof(TokSlot));
    if (!ns) return 0;
    for (uint32_t i = 0; i < sh->cap; i++) {
        TokSlot t = sh->s[i];
        if (t.tok == 0 || t.tok == TOK_DEAD) continue;
        uint32_t j = (tok_hash(t.key) / TOK_SHARDS) & (ncap - 1);
        while (ns[j].tok) j = (j + 1) & (ncap - 1);
        ns[j] = t;
    }
    free(sh->s);
    sh->s = ns;
    sh->cap = ncap;
    sh->dead = 0;
    return 1;
}

/* The token of `key`, created if new (0: the table is full). */
static uint64_t tok_get(TokTable *tb, uintptr_t key) {
    uint32_t h = tok_hash(key);
    TokShard *sh = &tb->sh[h % TOK_SHARDS];
    h /= TOK_SHARDS;
    uint64_t tok = 0;
    int found = 0;
    pthread_mutex_lock(&sh->mu);
    int j = sh->cap ? tok_find(sh, key, h, &found) : -1;
    if (found) {
        tok = sh->s[j].tok;
    } else {
        if ((sh->live + sh->dead + 1) * 4 > sh->cap * 3 && tok_grow(sh, tb->max_cap))
            j = tok_find(sh, key, h, &found);
        if (j >= 0 && (sh->live + 1) * 8 <= sh->cap * 7) {         /* never past 7/8 full */
            if (sh->s[j].tok == TOK_DEAD) sh->dead--;
            tok = __atomic_add_fetch(&tb->next, 1, __ATOMIC_RELAXED);
            sh->s[j] = (TokSlot){key, tok};
            sh->live++;
        }
    }
    pthread_mutex_unlock(&sh->mu);
    if (!tok) __atomic_add_fetch(&tb->unknown, 1, __ATOMIC_RELAXED);
    return tok;
}

static void tok_forget(TokTable *tb, uintptr_t key) {
    uint32_t h = tok_hash(key);
    TokShard *sh = &tb->sh[h % TOK_SHARDS];
    h /= TOK_SHARDS;
    int found = 0;
    pthread_mutex_lock(&sh->mu);
    int j = sh->cap ? tok_find(sh, key, h, &found) : -1;
    if (found) {
        sh->s[j].tok = TOK_DEAD;
        sh->live--;
        sh->dead++;
    }
    pthread_mutex_unlock(&sh->mu);
}

/* ── Task identity ───────────────────────────────────────────────────────
 * Every task gets an id from one counter (pid in the high bits): explicit
 * tasks at GOMP_task, the implicit task of each thread of a parallel region
 * when the thread enters it, the initial task of a thread on its first
 * need. tls_task is the task this thread is executing -- set and restored
 * by the parallel and task trampolines, so a task executed inside another
 * one's taskwait (libgomp runs it on the same thread's stack) nests
 * correctly. libgomp never moves a started task to another thread (an
 * untied task runs as a tied one), so this per-thread record is the task's
 * identity for its whole execution. */
static uint64_t g_task_seq = 0;
static __thread uint64_t tls_task = 0;

static uint64_t new_task_id(void) {
    return ((uint64_t)(uint32_t)getpid() << 32) | __atomic_add_fetch(&g_task_seq, 1, __ATOMIC_RELAXED);
}

static uint64_t current_task(void) {
    if (!tls_task) tls_task = new_task_id();      /* the thread's initial task */
    return tls_task;
}

/* ── libgomp's ABI level and the depend arrays ───────────────────────────
 * GOMP_task / GOMP_taskwait_depend receive the depend clauses as an array
 * whose layout is libgomp's private ABI (checked on GCC 13.3 / GOMP_5.1.2
 * by dumping the arrays of every dependence type, and against libgomp's
 * own reader):
 *   depend[0] = n > 0     n, then the number of out/inout entries, then n
 *                         addresses: out/inout first, then in (GCC >= 4.9);
 *   depend[0] = 0         then n, the number of out/inout, of
 *                         mutexinoutset and of in entries, then their
 *                         addresses in that order, then the remaining
 *                         entries as pointers to {address, kind} (depobj,
 *                         and inoutset) with kind 1 in, 2 out, 3 inout,
 *                         4 mutexinoutset, 5 inoutset (GOMP_5.0, GCC 9;
 *                         inoutset since GCC 12).
 * A NULL address is omp_all_memory. The first layout is decoded wherever
 * GOMP_4.0 exists; the second only from GOMP_5.0 to GOMP_5.1.2, the newest
 * ABI checked (HPROFILER_GOMP_DEPEND=force decodes it on a newer one;
 * HPROFILER_GOMP_ABI_LEVEL=<n> replaces the detected level, for tests).
 * Anything else -- an older or newer ABI, inconsistent counts, an unknown
 * kind -- records deps_unsupported=<reason> and reads no address: the
 * task's dependences are then unknown, not guessed. */
#define GOMP_ABI_V40          40000
#define GOMP_ABI_NEW_MIN      50000   /* GOMP_5.0 */
#define GOMP_ABI_VERIFIED_MAX 50102   /* GOMP_5.1.2 */
#define DEP_EMIT_MAX          256     /* entries per record; beyond: deps_omitted= */
#define DEP_COUNT_MAX         (1u << 20)

static int g_gomp_abi = -1;           /* x*10000 + y*100 + z of the newest GOMP_x.y[.z]; 0 unknown */
static int g_gomp_force = 0;
static pthread_once_t g_abi_once = PTHREAD_ONCE_INIT;

static int gomp_version_value(const char *s) {
    int x = 0, y = 0, z = 0;
    if (strncmp(s, "GOMP_", 5) != 0 || sscanf(s + 5, "%d.%d.%d", &x, &y, &z) < 1) return 0;
    return x * 10000 + y * 100 + z;
}

/* The newest GOMP_* version node libgomp defines (its ELF version
 * definitions, read from the loaded image). */
static int verdef_cb(struct dl_phdr_info *info, size_t size, void *data) {
    (void)size;
    if (!info->dlpi_name || !strstr(info->dlpi_name, "libgomp")) return 0;
    const ElfW(Dyn) *dyn = NULL;
    for (int i = 0; i < info->dlpi_phnum; i++)
        if (info->dlpi_phdr[i].p_type == PT_DYNAMIC)
            dyn = (const ElfW(Dyn) *)(info->dlpi_addr + info->dlpi_phdr[i].p_vaddr);
    if (!dyn) return 0;
    uintptr_t verdef = 0, strtab = 0;
    size_t nverdef = 0;
    for (; dyn->d_tag != DT_NULL; dyn++) {
        if (dyn->d_tag == DT_VERDEF) verdef = dyn->d_un.d_ptr;
        else if (dyn->d_tag == DT_VERDEFNUM) nverdef = dyn->d_un.d_val;
        else if (dyn->d_tag == DT_STRTAB) strtab = dyn->d_un.d_ptr;
    }
    if (!verdef || !strtab) return 0;
    /* the dynamic linker relocates some entries in place; others are offsets */
    if (verdef < info->dlpi_addr) verdef += info->dlpi_addr;
    if (strtab < info->dlpi_addr) strtab += info->dlpi_addr;
    int best = 0;
    const char *vd = (const char *)verdef;
    for (size_t i = 0; i < nverdef && vd; i++) {
        const ElfW(Verdef) *d = (const ElfW(Verdef) *)vd;
        if (d->vd_cnt > 0) {
            const ElfW(Verdaux) *a = (const ElfW(Verdaux) *)(vd + d->vd_aux);
            int v = gomp_version_value((const char *)strtab + a->vda_name);
            if (v > best) best = v;
        }
        vd = d->vd_next ? vd + d->vd_next : NULL;
    }
    *(int *)data = best;
    return 1;
}

static void gomp_abi_init(void) {
    int level = 0;
    dl_iterate_phdr(verdef_cb, &level);
    /* HPROFILER_GOMP_ABI_LEVEL=<n> stands in for the detected level (tests
     * of the unsupported-ABI path) */
    const char *ov = getenv("HPROFILER_GOMP_ABI_LEVEL");
    if (ov && *ov) level = atoi(ov);
    const char *f = getenv("HPROFILER_GOMP_DEPEND");
    g_gomp_force = f && strcmp(f, "force") == 0;
    g_gomp_abi = level;
    emit_ctr("gomp_abi_level", level, "version");
}

/* ",deps=<k><token>;...,ndeps=N[,deps_omitted=K][,dep_unknown=U]" (k: i in,
 * o out/inout, m mutexinoutset, s inoutset; token * for omp_all_memory), or
 * ",deps_unsupported=<reason>", appended at buf[*len] (cap bytes). */
static void deps_tags(char *buf, size_t cap, size_t *len, void **depend) {
    pthread_once(&g_abi_once, gomp_abi_init);
    size_t n = 0, nout = 0, nmut = 0, normal = 0, base = 0;
    const char *bad = NULL;
    uintptr_t d0 = (uintptr_t)depend[0];
    if (g_gomp_abi < GOMP_ABI_V40) {
        bad = "abi";
    } else if (d0) {
        n = d0;
        nout = (uintptr_t)depend[1];
        normal = n;
        base = 2;
        if (nout > n) bad = "counts";
    } else if (g_gomp_abi < GOMP_ABI_NEW_MIN || (g_gomp_abi > GOMP_ABI_VERIFIED_MAX && !g_gomp_force)) {
        bad = "abi";
    } else {
        n = (uintptr_t)depend[1];
        nout = (uintptr_t)depend[2];
        nmut = (uintptr_t)depend[3];
        normal = nout + nmut + (uintptr_t)depend[4];
        base = 5;
        if (normal > n || nmut > n) bad = "counts";
    }
    if (!bad && n > DEP_COUNT_MAX) bad = "counts";
    /* every pointer entry's kind first: an unknown one leaves the task's
     * dependences undecoded */
    for (size_t i = normal; !bad && i < n; i++) {
        void **o = (void **)depend[base + i];
        uintptr_t k = o ? (uintptr_t)o[1] : 0;
        if (k < 1 || k > 5) bad = "kind";
    }
    if (bad) {
        int w = snprintf(buf + *len, cap - *len, ",deps_unsupported=%s", bad);
        if (w > 0 && *len + (size_t)w < cap) *len += (size_t)w;
        return;
    }
    size_t emitted = 0, omitted = 0, unknown = 0;
    int w = snprintf(buf + *len, cap - *len, ",deps=");
    if (w <= 0 || *len + (size_t)w >= cap) return;
    *len += (size_t)w;
    size_t start = *len;
    for (size_t i = 0; i < n; i++) {
        void *addr;
        char kind;
        if (i < normal) {
            addr = depend[base + i];
            kind = i < nout ? 'o' : (i < nout + nmut ? 'm' : 'i');
        } else {
            void **o = (void **)depend[base + i];
            addr = o[0];
            static const char k2c[6] = {0, 'i', 'o', 'o', 'm', 's'};
            kind = k2c[(uintptr_t)o[1]];
        }
        if (emitted >= DEP_EMIT_MAX) { omitted++; continue; }
        char item[32];
        if (!addr) {
            snprintf(item, sizeof(item), "%c*", kind);
        } else {
            uint64_t tok = tok_get(&g_dep_tok, (uintptr_t)addr);
            if (!tok) { unknown++; continue; }
            snprintf(item, sizeof(item), "%c%llu", kind, (unsigned long long)tok);
        }
        size_t il = strlen(item) + (*len > start ? 1 : 0);
        if (*len + il + 1 >= cap) { omitted++; continue; }
        if (*len > start) buf[(*len)++] = ';';
        memcpy(buf + *len, item, strlen(item) + 1);
        *len += strlen(item);
        emitted++;
    }
    w = snprintf(buf + *len, cap - *len, ",ndeps=%zu", n);
    if (w > 0 && *len + (size_t)w < cap) *len += (size_t)w;
    if (omitted) {
        w = snprintf(buf + *len, cap - *len, ",deps_omitted=%zu", omitted);
        if (w > 0 && *len + (size_t)w < cap) *len += (size_t)w;
    }
    if (unknown) {
        w = snprintf(buf + *len, cap - *len, ",dep_unknown=%zu", unknown);
        if (w > 0 && *len + (size_t)w < cap) *len += (size_t)w;
    }
}

/* Room for deps_tags of `depend` (and the rest of a record's tags). */
static size_t deps_room(void **depend) {
    if (!depend) return 0;
    uintptr_t d0 = (uintptr_t)depend[0];
    size_t n = d0 ? d0 : (uintptr_t)depend[1];
    if (n > DEP_EMIT_MAX) n = DEP_EMIT_MAX;
    return 96 + 24 * n;
}

/* ── GOMP_parallel: per-thread timing via a trampoline + stack closure ─── */
typedef struct {
    void (*real_fn)(void *);
    void *real_data;
    const void *codeptr_ra;
    uint64_t team;          /* this parallel region's team id (team= on its spans) */
} ParallelClosure;

/* Team of the innermost parallel region this thread runs in (0: none).
 * Barriers carry it as team=, so criticalpath.py groups each team's barrier
 * by itself -- two concurrent nested teams' barriers overlap in time but are
 * no common rendezvous. */
static __thread uint64_t tls_team = 0;
static uint32_t g_team_seq = 0;

static void parallel_trampoline(void *arg) {
    ParallelClosure *c = (ParallelClosure *)arg;
    uint64_t outer_team = tls_team, outer_task = tls_task;
    tls_team = c->team;
    uint64_t implicit = new_task_id();          /* this thread's implicit task of the region */
    tls_task = implicit;
    uint64_t t0 = now_ns();
    c->real_fn(c->real_data);
    uint64_t dur = now_ns() - t0;
    tls_team = outer_team;
    tls_task = outer_task;
    char extra[1536];
    snprintf(extra, sizeof(extra), "type=parallel_region,team=%llu,itask=%llu", (unsigned long long)c->team,
             (unsigned long long)implicit);
    append_codeptr_tag(extra, sizeof(extra), c->codeptr_ra);
    emit_span("openmp", gettid_compat(), t0, dur, "omp_parallel_region", extra);
}

typedef void (*fn_GOMP_parallel_t)(void (*)(void *), void *, unsigned, unsigned int);
static fn_GOMP_parallel_t real_GOMP_parallel = NULL;

void GOMP_parallel(void (*fn)(void *), void *data, unsigned num_threads, unsigned int flags) {
    if (!real_GOMP_parallel) real_GOMP_parallel = (fn_GOMP_parallel_t)real_sym("GOMP_parallel");
    if (!real_GOMP_parallel) {
        /* Unreachable in practice (the program could not have started);
         * never call through NULL. Nothing is recorded. */
        fn(data);
        return;
    }
    if (in_hook) { real_GOMP_parallel(fn, data, num_threads, flags); return; }
    ParallelClosure closure = { .real_fn = fn, .real_data = data,
                                .codeptr_ra = __builtin_return_address(0),
                                .team = ((uint64_t)(uint32_t)getpid() << 32)
                                        | __atomic_add_fetch(&g_team_seq, 1, __ATOMIC_RELAXED) };
    real_GOMP_parallel(parallel_trampoline, &closure, num_threads, flags);
}

/* ── Work-sharing loops: static/dynamic/guided/runtime × start/end ─────── */
/* GOMP_loop_*_start's return value (did this thread get iterations) is
 * passed through unchanged. The start time is closed by the next
 * GOMP_loop_end/_end_nowait on this thread, whichever kind started it. */
static __thread uint64_t    tls_loop_start_ns = 0;
static __thread int         tls_loop_active   = 0;
static __thread const void *tls_loop_codeptr  = NULL;

/* (start, end, incr, chunk_size, *istart, *iend); the runtime kind has no
 * chunk_size and is wrapped separately below. */
#define _LOOP_START_WRAPPER(NAME)                                           \
typedef bool (*fn_##NAME##_t)(long, long, long, long, long *, long *);      \
static fn_##NAME##_t real_##NAME = NULL;                                    \
bool NAME(long start, long end, long incr, long chunk_size,                 \
         long *istart, long *iend) {                                        \
    if (!real_##NAME) real_##NAME = (fn_##NAME##_t)real_sym(#NAME);         \
    if (!real_##NAME) return false;                                        \
    tls_loop_codeptr  = __builtin_return_address(0);                       \
    tls_loop_start_ns = now_ns();                                           \
    tls_loop_active = 1;                                                    \
    return real_##NAME(start, end, incr, chunk_size, istart, iend);         \
}

_LOOP_START_WRAPPER(GOMP_loop_static_start)
_LOOP_START_WRAPPER(GOMP_loop_dynamic_start)
_LOOP_START_WRAPPER(GOMP_loop_guided_start)
/* GCC 13 emits the nonmonotonic variants for schedule(dynamic|guided) by
 * default (checked with nm -D -u on a compiled binary); both forms are
 * wrapped. Same signature, same end calls. */
_LOOP_START_WRAPPER(GOMP_loop_nonmonotonic_dynamic_start)
_LOOP_START_WRAPPER(GOMP_loop_nonmonotonic_guided_start)

/* No chunk_size: OMP_SCHEDULE supplies it. GCC 13 calls the
 * maybe_nonmonotonic variant for schedule(runtime); both are wrapped. */
typedef bool (*fn_GOMP_loop_runtime_start_t)(long, long, long, long *, long *);
static fn_GOMP_loop_runtime_start_t real_GOMP_loop_runtime_start = NULL;
bool GOMP_loop_runtime_start(long start, long end, long incr, long *istart, long *iend) {
    if (!real_GOMP_loop_runtime_start)
        real_GOMP_loop_runtime_start = (fn_GOMP_loop_runtime_start_t)real_sym("GOMP_loop_runtime_start");
    if (!real_GOMP_loop_runtime_start) return false;
    tls_loop_codeptr  = __builtin_return_address(0);
    tls_loop_start_ns = now_ns();
    tls_loop_active = 1;
    return real_GOMP_loop_runtime_start(start, end, incr, istart, iend);
}

typedef bool (*fn_GOMP_loop_maybe_nonmonotonic_runtime_start_t)(long, long, long, long *, long *);
static fn_GOMP_loop_maybe_nonmonotonic_runtime_start_t real_GOMP_loop_maybe_nonmonotonic_runtime_start = NULL;
bool GOMP_loop_maybe_nonmonotonic_runtime_start(long start, long end, long incr, long *istart, long *iend) {
    if (!real_GOMP_loop_maybe_nonmonotonic_runtime_start)
        real_GOMP_loop_maybe_nonmonotonic_runtime_start =
            (fn_GOMP_loop_maybe_nonmonotonic_runtime_start_t)real_sym("GOMP_loop_maybe_nonmonotonic_runtime_start");
    if (!real_GOMP_loop_maybe_nonmonotonic_runtime_start) return false;
    tls_loop_codeptr  = __builtin_return_address(0);
    tls_loop_start_ns = now_ns();
    tls_loop_active = 1;
    return real_GOMP_loop_maybe_nonmonotonic_runtime_start(start, end, incr, istart, iend);
}

static void _loop_end_common(const char *name) {
    if (tls_loop_active) {
        uint64_t dur = now_ns() - tls_loop_start_ns;
        char extra[1536];
        snprintf(extra, sizeof(extra), "type=work");
        append_codeptr_tag(extra, sizeof(extra), tls_loop_codeptr);
        emit_span("openmp", gettid_compat(), tls_loop_start_ns, dur, name, extra);
        tls_loop_active = 0;
    }
}

typedef void (*fn_GOMP_loop_end_t)(void);
static fn_GOMP_loop_end_t real_GOMP_loop_end = NULL;
void GOMP_loop_end(void) {
    if (!real_GOMP_loop_end) real_GOMP_loop_end = (fn_GOMP_loop_end_t)real_sym("GOMP_loop_end");
    if (real_GOMP_loop_end) real_GOMP_loop_end();
    _loop_end_common("omp_work_loop");
}

typedef void (*fn_GOMP_loop_end_nowait_t)(void);
static fn_GOMP_loop_end_nowait_t real_GOMP_loop_end_nowait = NULL;
void GOMP_loop_end_nowait(void) {
    if (!real_GOMP_loop_end_nowait)
        real_GOMP_loop_end_nowait = (fn_GOMP_loop_end_nowait_t)real_sym("GOMP_loop_end_nowait");
    if (real_GOMP_loop_end_nowait) real_GOMP_loop_end_nowait();
    _loop_end_common("omp_work_loop_nowait");
}

/* ── Barrier ─────────────────────────────────────────────────────────────
 * The call blocks until every thread arrives, so its duration is the wait.
 * Category "sync" and the "omp_" prefix are what criticalpath.py's barrier
 * rendezvous matches. */
typedef void (*fn_GOMP_barrier_t)(void);
static fn_GOMP_barrier_t real_GOMP_barrier = NULL;
void GOMP_barrier(void) {
    if (!real_GOMP_barrier) real_GOMP_barrier = (fn_GOMP_barrier_t)real_sym("GOMP_barrier");
    const void *ret = __builtin_return_address(0);
    uint64_t t0 = now_ns();
    if (real_GOMP_barrier) real_GOMP_barrier();
    char extra[1536];
    if (tls_team)
        snprintf(extra, sizeof(extra), "type=sync,team=%llu", (unsigned long long)tls_team);
    else
        snprintf(extra, sizeof(extra), "type=sync");
    append_codeptr_tag(extra, sizeof(extra), ret);
    emit_span("sync", gettid_compat(), t0, now_ns() - t0, "omp_barrier", extra);
}

/* ── Critical sections ───────────────────────────────────────────────────
 * omp_critical_wait is the _start call (time to acquire); omp_critical_hold
 * runs from _start returning to _end. One TLS slot, not a stack: with nested
 * critical sections the outer hold is measured from the inner entry. */
static __thread uint64_t tls_critical_enter_ns = 0;
static __thread const void *tls_critical_codeptr = NULL;   /* call site of the critical_start */

static void emit_critical_hold(uint64_t t0, uint64_t now, const char *tags) {
    char extra[1536];
    snprintf(extra, sizeof(extra), "%s", tags);
    append_codeptr_tag(extra, sizeof(extra), tls_critical_codeptr);
    emit_span("openmp", gettid_compat(), t0, now - t0, "omp_critical_hold", extra);
}

typedef void (*fn_GOMP_critical_start_t)(void);
static fn_GOMP_critical_start_t real_GOMP_critical_start = NULL;
void GOMP_critical_start(void) {
    if (!real_GOMP_critical_start)
        real_GOMP_critical_start = (fn_GOMP_critical_start_t)real_sym("GOMP_critical_start");
    const void *ret = __builtin_return_address(0);
    uint64_t t0 = now_ns();
    if (real_GOMP_critical_start) real_GOMP_critical_start();
    uint64_t t1 = now_ns();
    char extra[1536];
    snprintf(extra, sizeof(extra), "type=sync");
    append_codeptr_tag(extra, sizeof(extra), ret);
    emit_span("sync", gettid_compat(), t0, t1 - t0, "omp_critical_wait", extra);
    tls_critical_enter_ns = t1;
    tls_critical_codeptr = ret;
}

typedef void (*fn_GOMP_critical_end_t)(void);
static fn_GOMP_critical_end_t real_GOMP_critical_end = NULL;
void GOMP_critical_end(void) {
    if (!real_GOMP_critical_end)
        real_GOMP_critical_end = (fn_GOMP_critical_end_t)real_sym("GOMP_critical_end");
    uint64_t t0 = tls_critical_enter_ns;
    uint64_t now = now_ns();
    if (real_GOMP_critical_end) real_GOMP_critical_end();
    if (t0) emit_critical_hold(t0, now, "type=critical");
}

typedef void (*fn_GOMP_critical_name_start_t)(void **);
static fn_GOMP_critical_name_start_t real_GOMP_critical_name_start = NULL;
void GOMP_critical_name_start(void **pptr) {
    if (!real_GOMP_critical_name_start)
        real_GOMP_critical_name_start = (fn_GOMP_critical_name_start_t)real_sym("GOMP_critical_name_start");
    const void *ret = __builtin_return_address(0);
    uint64_t t0 = now_ns();
    if (real_GOMP_critical_name_start) real_GOMP_critical_name_start(pptr);
    uint64_t t1 = now_ns();
    char extra[1536];
    snprintf(extra, sizeof(extra), "type=sync,named=1");
    append_codeptr_tag(extra, sizeof(extra), ret);
    emit_span("sync", gettid_compat(), t0, t1 - t0, "omp_critical_wait", extra);
    tls_critical_enter_ns = t1;
    tls_critical_codeptr = ret;
}

typedef void (*fn_GOMP_critical_name_end_t)(void **);
static fn_GOMP_critical_name_end_t real_GOMP_critical_name_end = NULL;
void GOMP_critical_name_end(void **pptr) {
    if (!real_GOMP_critical_name_end)
        real_GOMP_critical_name_end = (fn_GOMP_critical_name_end_t)real_sym("GOMP_critical_name_end");
    uint64_t t0 = tls_critical_enter_ns;
    uint64_t now = now_ns();
    if (real_GOMP_critical_name_end) real_GOMP_critical_name_end(pptr);
    if (t0) emit_critical_hold(t0, now, "type=critical,named=1");
}

/* ── Single ──────────────────────────────────────────────────────────────
 * True only on the thread that executes the region. An instant, not a span:
 * there is no GOMP_single_end, so the region's length is not observable. */
typedef bool (*fn_GOMP_single_start_t)(void);
static fn_GOMP_single_start_t real_GOMP_single_start = NULL;
bool GOMP_single_start(void) {
    if (!real_GOMP_single_start)
        real_GOMP_single_start = (fn_GOMP_single_start_t)real_sym("GOMP_single_start");
    if (!real_GOMP_single_start) return false;
    bool executor = real_GOMP_single_start();
    if (executor && hp_tx_enabled())
        hp_tx_emitf("inst:openmp:%d:%d:%llu:omp_single:type=single\n",
                    (int)hp_tx_pid(), (int)gettid_compat(), (unsigned long long)now_ns());
    return executor;
}

/* ── Is the real function libgomp's? ─────────────────────────────────────
 * libomp exports GOMP_* and omp_*_lock too (for GCC-compiled code); there
 * the OMPT tool reports these constructs, so recording them here as well
 * would count them twice. */
static int from_libgomp(void *fn) {
    Dl_info info;
    return fn && dladdr(fn, &info) && info.dli_fname && strstr(info.dli_fname, "libgomp");
}

/* ── Tasks ───────────────────────────────────────────────────────────────
 * GOMP_task(fn, data, cpyfn, arg_size, arg_align, ...): libgomp copies the
 * task's arguments (cpyfn, else memcpy of arg_size bytes) into the task and
 * later calls fn on the copy -- at once for an undeferred task. The wrapper
 * hands libgomp a trampoline and a copier of its own instead: the copy is a
 * header (the real fn, task id, team) followed by the real arguments, built
 * by the real cpyfn or memcpy, so the trampoline runs the real fn on them
 * and times it: span omp_task (openmp, on the thread that ran it). The
 * creation is a zero-length span omp_task_create (as in ompt_tool.c). */
typedef struct {
    void (*fn)(void *);              /* the task body */
    void (*cpyfn)(void *, void *);   /* source closure only: the real copier */
    void *data;                      /* source closure only: the real arguments */
    long arg_size;
    long hdr;                        /* offset of the arguments in the copy */
    uint64_t id, team;
} TaskClosure;

static void task_trampoline(void *arg) {
    TaskClosure *h = (TaskClosure *)arg;
    void (*fn)(void *) = h->fn;
    uint64_t id = h->id, team = h->team;
    uint64_t outer_team = tls_team, outer_task = tls_task;
    tls_team = team;
    tls_task = id;
    uint64_t t0 = now_ns();
    fn((char *)arg + h->hdr);
    uint64_t t1 = now_ns();
    tls_team = outer_team;
    tls_task = outer_task;
    char extra[96];
    snprintf(extra, sizeof(extra), "type=task,id=%llu,team=%llu", (unsigned long long)id,
             (unsigned long long)team);
    emit_span("openmp", gettid_compat(), t0, t1 - t0, "omp_task", extra);
}

static void task_copier(void *dst, void *src) {
    TaskClosure *s = (TaskClosure *)src, *d = (TaskClosure *)dst;
    *d = *s;
    d->cpyfn = NULL;
    d->data = NULL;
    char *args = (char *)dst + s->hdr;
    if (s->cpyfn) s->cpyfn(args, s->data);
    else if (s->arg_size > 0) memcpy(args, s->data, (size_t)s->arg_size);
}

typedef void (*fn_GOMP_task_t)(void (*)(void *), void *, void (*)(void *, void *), long, long,
                               bool, unsigned, void **, int, void *);
static fn_GOMP_task_t real_GOMP_task = NULL;
void GOMP_task(void (*fn)(void *), void *data, void (*cpyfn)(void *, void *), long arg_size,
               long arg_align, bool if_clause, unsigned flags, void **depend, int priority,
               void *detach) {
    static int gomp = -1;
    if (!real_GOMP_task) real_GOMP_task = (fn_GOMP_task_t)real_sym("GOMP_task");
    if (!real_GOMP_task) return;
    if (gomp < 0) gomp = from_libgomp((void *)real_GOMP_task);
    if (!gomp || in_hook || !hp_tx_enabled()) {
        real_GOMP_task(fn, data, cpyfn, arg_size, arg_align, if_clause, flags, depend, priority, detach);
        return;
    }
    const void *ret = __builtin_return_address(0);
    long align = arg_align > 16 ? arg_align : 16;
    long hdr = ((long)sizeof(TaskClosure) + align - 1) / align * align;
    TaskClosure c = { .fn = fn, .cpyfn = cpyfn, .data = data, .arg_size = arg_size, .hdr = hdr,
                      .id = new_task_id(), .team = tls_team };
    uint64_t t0 = now_ns();
    /* GOMP_TASK_FLAG_DEPEND (8): depend is the clause array (deps_tags) */
    void **deps = (depend && (flags & 8u)) ? depend : NULL;
    size_t cap = 1536 + deps_room(deps);
    char stackbuf[2048];
    char *extra = cap <= sizeof(stackbuf) ? stackbuf : (char *)malloc(cap);
    if (!extra) { extra = stackbuf; cap = sizeof(stackbuf); deps = NULL; }
    int w = snprintf(extra, cap, "type=task_create,id=%llu,team=%llu,parent=%llu%s%s", (unsigned long long)c.id,
                     (unsigned long long)c.team, (unsigned long long)current_task(),
                     if_clause ? "" : ",undeferred=1", depend ? ",depend=1" : "");
    size_t len = w > 0 ? (size_t)w : 0;
    if (deps) deps_tags(extra, cap - 1300, &len, deps);   /* leave room for the call site */
    append_codeptr_tag(extra, cap, ret);
    emit_span("openmp", gettid_compat(), t0, 0, "omp_task_create", extra);
    if (extra != stackbuf) free(extra);
    real_GOMP_task(task_trampoline, &c, task_copier, hdr + (arg_size > 0 ? arg_size : 0), align,
                   if_clause, flags, depend, priority, detach);
}

/* A taskwait / taskgroup span: task= the task that waits (its children,
 * parent=, are what a taskwait waits for), and for GOMP_taskwait_depend its
 * clauses (deps=, as on omp_task_create). */
static void emit_task_wait(const char *name, uint64_t t0, const void *ret, const char *more,
                           uint64_t task, void **depend) {
    size_t cap = 1536 + deps_room(depend) + strlen(more);
    char stackbuf[2048];
    char *extra = cap <= sizeof(stackbuf) ? stackbuf : (char *)malloc(cap);
    if (!extra) { extra = stackbuf; cap = sizeof(stackbuf); depend = NULL; }
    int w = snprintf(extra, cap, "type=sync,team=%llu,task=%llu%s", (unsigned long long)tls_team,
                     (unsigned long long)task, more);
    size_t len = w > 0 ? (size_t)w : 0;
    if (depend) deps_tags(extra, cap - 1300, &len, depend);
    append_codeptr_tag(extra, cap, ret);
    emit_span("sync", gettid_compat(), t0, now_ns() - t0, name, extra);
    if (extra != stackbuf) free(extra);
}

typedef void (*fn_void_t)(void);
static fn_void_t real_GOMP_taskwait = NULL;
void GOMP_taskwait(void) {
    static int gomp = -1;
    if (!real_GOMP_taskwait) real_GOMP_taskwait = (fn_void_t)real_sym("GOMP_taskwait");
    if (!real_GOMP_taskwait) return;
    if (gomp < 0) gomp = from_libgomp((void *)real_GOMP_taskwait);
    const void *ret = __builtin_return_address(0);
    uint64_t task = current_task();
    uint64_t t0 = now_ns();
    real_GOMP_taskwait();
    if (gomp) emit_task_wait("omp_taskwait", t0, ret, "", task, NULL);
}

typedef void (*fn_GOMP_taskwait_depend_t)(void **);
static fn_GOMP_taskwait_depend_t real_GOMP_taskwait_depend = NULL;
void GOMP_taskwait_depend(void **depend) {
    static int gomp = -1;
    if (!real_GOMP_taskwait_depend)
        real_GOMP_taskwait_depend = (fn_GOMP_taskwait_depend_t)real_sym("GOMP_taskwait_depend");
    if (!real_GOMP_taskwait_depend) return;
    if (gomp < 0) gomp = from_libgomp((void *)real_GOMP_taskwait_depend);
    const void *ret = __builtin_return_address(0);
    uint64_t task = current_task();
    uint64_t t0 = now_ns();
    /* the clause array is read before the call: libgomp may free nothing of
     * it, but it is the caller's stack only until the call returns */
    size_t cap = 96 + deps_room(depend);
    char *dtags = gomp && depend ? (char *)malloc(cap) : NULL;
    size_t dlen = 0;
    if (dtags) { dtags[0] = '\0'; deps_tags(dtags, cap, &dlen, depend); }
    real_GOMP_taskwait_depend(depend);
    if (gomp) {
        size_t mcap = 16 + (dtags ? dlen : 0);
        char *more = (char *)malloc(mcap);
        if (more) {
            snprintf(more, mcap, ",depend=1%s", dtags ? dtags : "");
            emit_task_wait("omp_taskwait", t0, ret, more, task, NULL);
            free(more);
        } else {
            emit_task_wait("omp_taskwait", t0, ret, ",depend=1", task, NULL);
        }
    }
    free(dtags);
}

/* The taskgroup span runs from _start to the end of _end (which waits for
 * the group's tasks); one TLS stack for nesting. */
#define MAX_TASKGROUP 32
static __thread uint64_t    tls_tg_start[MAX_TASKGROUP];
static __thread const void *tls_tg_cptr[MAX_TASKGROUP];
static __thread uint64_t    tls_tg_task[MAX_TASKGROUP];
static __thread int         tls_tg_depth = 0;

static fn_void_t real_GOMP_taskgroup_start = NULL;
void GOMP_taskgroup_start(void) {
    if (!real_GOMP_taskgroup_start) real_GOMP_taskgroup_start = (fn_void_t)real_sym("GOMP_taskgroup_start");
    if (!real_GOMP_taskgroup_start) return;
    if (tls_tg_depth < MAX_TASKGROUP) {
        tls_tg_start[tls_tg_depth] = now_ns();
        tls_tg_cptr[tls_tg_depth] = __builtin_return_address(0);
        tls_tg_task[tls_tg_depth] = current_task();
    }
    tls_tg_depth++;
    real_GOMP_taskgroup_start();
}

static fn_void_t real_GOMP_taskgroup_end = NULL;
void GOMP_taskgroup_end(void) {
    static int gomp = -1;
    if (!real_GOMP_taskgroup_end) real_GOMP_taskgroup_end = (fn_void_t)real_sym("GOMP_taskgroup_end");
    if (!real_GOMP_taskgroup_end) return;
    if (gomp < 0) gomp = from_libgomp((void *)real_GOMP_taskgroup_end);
    real_GOMP_taskgroup_end();
    if (tls_tg_depth > 0 && --tls_tg_depth < MAX_TASKGROUP && gomp)
        emit_task_wait("omp_taskgroup", tls_tg_start[tls_tg_depth], tls_tg_cptr[tls_tg_depth], "",
                       tls_tg_task[tls_tg_depth], NULL);
}

/* ── Locks: omp_{set,unset,test}[_nest]_lock ─────────────────────────────
 * omp_lock_wait (sync): the omp_set_*lock call, for an acquisition the
 * thread does not already own; omp_lock_hold (openmp): from the
 * acquisition (set, or a successful test: test=1) to the release that
 * frees the lock (a nest lock: its outermost level, nest=1). The same names
 * as ompt_tool.c's mutex callbacks. */
/* lock=<token> (see "Generated tokens"), "?" when the table is full */
static void lock_token(char *buf, size_t cap, void *lock) {
    uint64_t t = tok_get(&g_lock_tok, (uintptr_t)lock);
    if (t) snprintf(buf, cap, "%llu", (unsigned long long)t);
    else snprintf(buf, cap, "?");
}

#define MAX_LOCKS_HELD 16
typedef struct { void *lock; uint64_t t0; int depth, nest, test; const void *cptr; } HeldLock;
static __thread HeldLock tls_locks[MAX_LOCKS_HELD];
static __thread int      tls_nlocks = 0;

static HeldLock *held_lock(void *lock) {
    for (int i = 0; i < tls_nlocks; i++)
        if (tls_locks[i].lock == lock) return &tls_locks[i];
    return NULL;
}

static void lock_acquired(void *lock, uint64_t t, int nest, int test, const void *cptr) {
    HeldLock *h = held_lock(lock);
    if (h) { h->depth++; return; }
    if (tls_nlocks < MAX_LOCKS_HELD)
        tls_locks[tls_nlocks++] = (HeldLock){lock, t, 1, nest, test, cptr};
}

static void lock_released(void *lock) {
    HeldLock *h = held_lock(lock);
    if (!h || --h->depth > 0) return;
    uint64_t t = now_ns();
    char extra[1536];
    char tok[24];
    lock_token(tok, sizeof(tok), lock);
    snprintf(extra, sizeof(extra), "type=hold,mutex=lock,lock=%s%s%s", tok, h->nest ? ",nest=1" : "",
             h->test ? ",test=1" : "");
    append_codeptr_tag(extra, sizeof(extra), h->cptr);
    emit_span("openmp", gettid_compat(), h->t0, t - h->t0, "omp_lock_hold", extra);
    *h = tls_locks[--tls_nlocks];
}

static void lock_wait_span(void *lock, uint64_t t0, uint64_t t1, int nest, const void *cptr) {
    char extra[1536];
    char tok[24];
    lock_token(tok, sizeof(tok), lock);
    snprintf(extra, sizeof(extra), "type=sync,mutex=lock,lock=%s%s", tok, nest ? ",nest=1" : "");
    append_codeptr_tag(extra, sizeof(extra), cptr);
    emit_span("sync", gettid_compat(), t0, t1 - t0, "omp_lock_wait", extra);
}

/* The libgomp flag is stored before the pointer is published, so a thread
 * that sees the pointer also sees the flag (two threads may resolve it). */
#define LOCK_REAL(var, name) do { \
        if (!__atomic_load_n(&var, __ATOMIC_ACQUIRE)) { \
            void *_f = real_sym(name); \
            __atomic_store_n(&var##_gomp, from_libgomp(_f), __ATOMIC_RELAXED); \
            __atomic_store_n(&var, _f, __ATOMIC_RELEASE); \
        } } while (0)

static void *real_set_lock, *real_unset_lock, *real_test_lock;
static void *real_set_nest_lock, *real_unset_nest_lock, *real_test_nest_lock;
static int real_set_lock_gomp, real_unset_lock_gomp, real_test_lock_gomp;
static int real_set_nest_lock_gomp, real_unset_nest_lock_gomp, real_test_nest_lock_gomp;

void omp_set_lock(void *lock) {
    LOCK_REAL(real_set_lock, "omp_set_lock");
    if (!real_set_lock) return;
    if (!real_set_lock_gomp || !hp_tx_enabled()) { ((void (*)(void *))real_set_lock)(lock); return; }
    const void *ret = __builtin_return_address(0);
    uint64_t t0 = now_ns();
    ((void (*)(void *))real_set_lock)(lock);
    uint64_t t1 = now_ns();
    lock_wait_span(lock, t0, t1, 0, ret);
    lock_acquired(lock, t1, 0, 0, ret);
}

void omp_unset_lock(void *lock) {
    LOCK_REAL(real_unset_lock, "omp_unset_lock");
    if (!real_unset_lock) return;
    if (real_unset_lock_gomp) lock_released(lock);
    ((void (*)(void *))real_unset_lock)(lock);
}

int omp_test_lock(void *lock) {
    LOCK_REAL(real_test_lock, "omp_test_lock");
    if (!real_test_lock) return 0;
    int got = ((int (*)(void *))real_test_lock)(lock);
    if (got && real_test_lock_gomp && hp_tx_enabled())
        lock_acquired(lock, now_ns(), 0, 1, __builtin_return_address(0));
    return got;
}

void omp_set_nest_lock(void *lock) {
    LOCK_REAL(real_set_nest_lock, "omp_set_nest_lock");
    if (!real_set_nest_lock) return;
    if (!real_set_nest_lock_gomp || !hp_tx_enabled()) { ((void (*)(void *))real_set_nest_lock)(lock); return; }
    const void *ret = __builtin_return_address(0);
    int owned = held_lock(lock) != NULL;
    uint64_t t0 = now_ns();
    ((void (*)(void *))real_set_nest_lock)(lock);
    uint64_t t1 = now_ns();
    if (!owned) lock_wait_span(lock, t0, t1, 1, ret);
    lock_acquired(lock, t1, 1, 0, ret);
}

void omp_unset_nest_lock(void *lock) {
    LOCK_REAL(real_unset_nest_lock, "omp_unset_nest_lock");
    if (!real_unset_nest_lock) return;
    if (real_unset_nest_lock_gomp) lock_released(lock);
    ((void (*)(void *))real_unset_nest_lock)(lock);
}

int omp_test_nest_lock(void *lock) {
    LOCK_REAL(real_test_nest_lock, "omp_test_nest_lock");
    if (!real_test_nest_lock) return 0;
    int depth = ((int (*)(void *))real_test_nest_lock)(lock);
    if (depth > 0 && real_test_nest_lock_gomp && hp_tx_enabled())
        lock_acquired(lock, now_ns(), 1, 1, __builtin_return_address(0));
    return depth;
}

/* Initialising or destroying a lock retires its token: the next lock at
 * that address is another lock. Only for libgomp's locks (LOCK_REAL). */
#define LOCK_LIFECYCLE(FN, PROTO, ARGS)                                         \
static void *real_##FN; static int real_##FN##_gomp;                           \
void FN PROTO {                                                                 \
    LOCK_REAL(real_##FN, #FN);                                                  \
    if (!real_##FN) return;                                                     \
    ((void (*) PROTO)real_##FN) ARGS;                                           \
    if (real_##FN##_gomp) tok_forget(&g_lock_tok, (uintptr_t)lock);             \
}
LOCK_LIFECYCLE(omp_init_lock, (void *lock), (lock))
LOCK_LIFECYCLE(omp_init_nest_lock, (void *lock), (lock))
LOCK_LIFECYCLE(omp_destroy_lock, (void *lock), (lock))
LOCK_LIFECYCLE(omp_destroy_nest_lock, (void *lock), (lock))
LOCK_LIFECYCLE(omp_init_lock_with_hint, (void *lock, int hint), (lock, hint))
LOCK_LIFECYCLE(omp_init_nest_lock_with_hint, (void *lock, int hint), (lock, hint))
#undef LOCK_LIFECYCLE

/* Any GOMP_* call can come first: set up at load time. */
__attribute__((constructor))
static void hprofiler_gomp_init(void) {
    hp_tx_init("gomp");
    cs_init();
}

/* libgomp has no shutdown callback: drain every thread's buffered events
 * and send the final transport status at unload. */
__attribute__((destructor))
static void hprofiler_gomp_fini(void) {
    if (__atomic_load_n(&g_dep_tok.next, __ATOMIC_RELAXED) || g_dep_tok.unknown) {
        /* dependence addresses that got a token / that did not (table full) */
        emit_ctr("gomp_dep_tokens", (int64_t)__atomic_load_n(&g_dep_tok.next, __ATOMIC_RELAXED), "tokens");
        emit_ctr("gomp_dep_unknown", (int64_t)__atomic_load_n(&g_dep_tok.unknown, __ATOMIC_RELAXED), "deps");
    }
    hp_tx_shutdown(1);
}
