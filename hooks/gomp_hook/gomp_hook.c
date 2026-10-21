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
    uint64_t outer_team = tls_team;
    tls_team = c->team;
    uint64_t t0 = now_ns();
    c->real_fn(c->real_data);
    uint64_t dur = now_ns() - t0;
    tls_team = outer_team;
    char extra[1536];
    snprintf(extra, sizeof(extra), "type=parallel_region,team=%llu", (unsigned long long)c->team);
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

static uint64_t g_task_seq = 0;

static void task_trampoline(void *arg) {
    TaskClosure *h = (TaskClosure *)arg;
    void (*fn)(void *) = h->fn;
    uint64_t id = h->id, team = h->team;
    uint64_t outer_team = tls_team;
    tls_team = team;
    uint64_t t0 = now_ns();
    fn((char *)arg + h->hdr);
    uint64_t t1 = now_ns();
    tls_team = outer_team;
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
                      .id = ((uint64_t)(uint32_t)getpid() << 32)
                            | __atomic_add_fetch(&g_task_seq, 1, __ATOMIC_RELAXED),
                      .team = tls_team };
    uint64_t t0 = now_ns();
    char extra[1536];
    snprintf(extra, sizeof(extra), "type=task_create,id=%llu,team=%llu%s%s", (unsigned long long)c.id,
             (unsigned long long)c.team, if_clause ? "" : ",undeferred=1", depend ? ",depend=1" : "");
    append_codeptr_tag(extra, sizeof(extra), ret);
    emit_span("openmp", gettid_compat(), t0, 0, "omp_task_create", extra);
    real_GOMP_task(task_trampoline, &c, task_copier, hdr + (arg_size > 0 ? arg_size : 0), align,
                   if_clause, flags, depend, priority, detach);
}

static void emit_task_wait(const char *name, uint64_t t0, const void *ret, const char *more) {
    char extra[1536];
    snprintf(extra, sizeof(extra), "type=sync,team=%llu%s", (unsigned long long)tls_team, more);
    append_codeptr_tag(extra, sizeof(extra), ret);
    emit_span("sync", gettid_compat(), t0, now_ns() - t0, name, extra);
}

typedef void (*fn_void_t)(void);
static fn_void_t real_GOMP_taskwait = NULL;
void GOMP_taskwait(void) {
    static int gomp = -1;
    if (!real_GOMP_taskwait) real_GOMP_taskwait = (fn_void_t)real_sym("GOMP_taskwait");
    if (!real_GOMP_taskwait) return;
    if (gomp < 0) gomp = from_libgomp((void *)real_GOMP_taskwait);
    const void *ret = __builtin_return_address(0);
    uint64_t t0 = now_ns();
    real_GOMP_taskwait();
    if (gomp) emit_task_wait("omp_taskwait", t0, ret, "");
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
    uint64_t t0 = now_ns();
    real_GOMP_taskwait_depend(depend);
    if (gomp) emit_task_wait("omp_taskwait", t0, ret, ",depend=1");
}

/* The taskgroup span runs from _start to the end of _end (which waits for
 * the group's tasks); one TLS stack for nesting. */
#define MAX_TASKGROUP 32
static __thread uint64_t    tls_tg_start[MAX_TASKGROUP];
static __thread const void *tls_tg_cptr[MAX_TASKGROUP];
static __thread int         tls_tg_depth = 0;

static fn_void_t real_GOMP_taskgroup_start = NULL;
void GOMP_taskgroup_start(void) {
    if (!real_GOMP_taskgroup_start) real_GOMP_taskgroup_start = (fn_void_t)real_sym("GOMP_taskgroup_start");
    if (!real_GOMP_taskgroup_start) return;
    if (tls_tg_depth < MAX_TASKGROUP) {
        tls_tg_start[tls_tg_depth] = now_ns();
        tls_tg_cptr[tls_tg_depth] = __builtin_return_address(0);
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
        emit_task_wait("omp_taskgroup", tls_tg_start[tls_tg_depth], tls_tg_cptr[tls_tg_depth], "");
}

/* ── Locks: omp_{set,unset,test}[_nest]_lock ─────────────────────────────
 * omp_lock_wait (sync): the omp_set_*lock call, for an acquisition the
 * thread does not already own; omp_lock_hold (openmp): from the
 * acquisition (set, or a successful test: test=1) to the release that
 * frees the lock (a nest lock: its outermost level, nest=1). The same names
 * as ompt_tool.c's mutex callbacks. */
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
    snprintf(extra, sizeof(extra), "type=hold,mutex=lock,lock=%p%s%s", lock, h->nest ? ",nest=1" : "",
             h->test ? ",test=1" : "");
    append_codeptr_tag(extra, sizeof(extra), h->cptr);
    emit_span("openmp", gettid_compat(), h->t0, t - h->t0, "omp_lock_hold", extra);
    *h = tls_locks[--tls_nlocks];
}

static void lock_wait_span(void *lock, uint64_t t0, uint64_t t1, int nest, const void *cptr) {
    char extra[1536];
    snprintf(extra, sizeof(extra), "type=sync,mutex=lock,lock=%p%s", lock, nest ? ",nest=1" : "");
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
