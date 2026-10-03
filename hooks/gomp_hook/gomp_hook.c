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
 * (anonymous and named), GOMP_single_start. Not covered: the legacy
 * GOMP_parallel_start/_end ABI, GOMP_task/taskwait (their ABI changed across
 * GCC versions; a wrong signature would crash the program), sections,
 * doacross, target offload, the _ull loop variants.
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

/* ── GOMP_parallel: per-thread timing via a trampoline + stack closure ─── */
typedef struct {
    void (*real_fn)(void *);
    void *real_data;
    const void *codeptr_ra;
} ParallelClosure;

static void parallel_trampoline(void *arg) {
    ParallelClosure *c = (ParallelClosure *)arg;
    uint64_t t0 = now_ns();
    c->real_fn(c->real_data);
    uint64_t dur = now_ns() - t0;
    char extra[1536];
    snprintf(extra, sizeof(extra), "type=parallel_region");
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
                                .codeptr_ra = __builtin_return_address(0) };
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
    hp_tx_shutdown(1);
}
