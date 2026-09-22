/*
 * GNU libgomp LD_PRELOAD hook — OpenMP profiling for binaries linked
 * against GCC's libgomp instead of LLVM's libomp.
 *
 * ── Why this exists ──────────────────────────────────────────────────────
 * hooks/ompt_tool/ompt_tool.c uses the OpenMP 5.0 Tools Interface (OMPT),
 * loaded via OMP_TOOL_LIBRARIES. That only works when the profiled binary
 * is linked against an OMPT-capable libomp (LLVM's). GNU's libgomp has no
 * OMPT support in typical distro/vendor builds -- confirmed on this
 * project's own dev machine (Ubuntu 24.04, GCC 13.3): `nm -D
 * libgomp.so.1 | grep ompt` returns nothing at all, no OMPT symbols
 * exported. A real user profiling GROMACS on the Dardel HPC cluster hit
 * exactly this: `ldd gmx_mpi` shows `libgomp.so.1`, so
 * OMP_TOOL_LIBRARIES never gets a chance to register -- `hprofiler
 * backends` correctly reports "openmp available" (an LLVM libomp.so DOES
 * exist somewhere on the system, e.g. bundled with ROCm), but the
 * PROFILED BINARY was never going to use that library at runtime, so zero
 * events were ever going to be captured regardless of detection.
 *
 * Rather than depend on libgomp's OMPT support (unreliable and, per the
 * above, sometimes entirely absent even when GCC's own source tree has
 * some OMPT scaffolding), this hook uses the SAME mechanism every other
 * hook in this codebase already relies on: LD_PRELOAD function
 * interposition, applied to libgomp's own public ABI (the `GOMP_*` symbol
 * family GCC-generated code calls directly for every `#pragma omp`
 * construct) instead of a vendor tools callback API. This ABI has been
 * essentially frozen since GCC 4.9 for the constructs covered here
 * (GOMP_1.0/GOMP_2.0/GOMP_4.0 symbol versions) -- far more stable in
 * practice, on this evidence, than libgomp's OMPT coverage.
 *
 * ── Per-thread visibility for GOMP_parallel ────────────────────────────
 * GOMP_parallel(fn, data, num_threads, flags) itself dispatches `fn(data)`
 * onto libgomp's internal thread pool AND runs it on the calling thread as
 * one of the participants -- wrapping only the outer call (time from
 * before calling the real GOMP_parallel to after it returns) would only
 * ever measure the INITIATING thread's overall time in the construct, not
 * each worker thread's individual contribution (which is what OMPT's
 * per-thread ompt_callback_implicit_task gives, and what this hook aims to
 * match). Instead, this substitutes a trampoline function + a small
 * stack-allocated closure as the (fn, data) actually passed to the real
 * GOMP_parallel: libgomp then calls the TRAMPOLINE on every participating
 * thread, so each one times its own call into the real fn independently.
 * The closure is safe to stack-allocate because GOMP_parallel does not
 * return until every thread has finished running fn (and the region's
 * implicit barrier has completed) -- the stack frame holding it is
 * guaranteed to outlive every thread's read of it.
 *
 * ── Scope of this first version ─────────────────────────────────────────
 * Covers GOMP_parallel, the four common loop scheduling kinds (static/
 * dynamic/guided/runtime) start+end, GOMP_barrier, both critical-section
 * variants (anonymous and named), and GOMP_single_start -- the constructs
 * a typical scientific/HPC code (GROMACS included) actually uses. Not
 * covered: the legacy split GOMP_parallel_start/_end ABI (superseded by
 * combined GOMP_parallel since GCC 4.9, rare in code built by a modern
 * toolchain), GOMP_task/GOMP_taskwait (the task ABI has changed more
 * across GCC versions than the constructs above, and getting a calling-
 * convention wrong here risks a crash in the profiled program -- left out
 * rather than guessed at without the ability to verify the exact
 * signature Dardel's specific GCC build uses), sections, doacross, target
 * offload, and the _ull (unsigned long long trip count) loop variants.
 *
 * Wire format to HPROFILER_SOCKET (identical to every other hook):
 *   span:<cat>:<pid>:<tid>:<start_ns>:<dur_ns>:<name>[:<key=val,...>]\n
 * Categories/names match ompt_tool.c's conventions exactly (category
 * "openmp" for parallel/work-sharing/single, "sync" for barrier/critical
 * with a name prefixed "omp_" -- required for
 * src/analysis/criticalpath.py's _add_omp_barrier_edges, which only
 * recognizes sync-category spans whose name starts with "omp_") so both
 * hooks' output is indistinguishable to every downstream analysis module.
 *
 * ── Verification status ─────────────────────────────────────────────────
 * Unlike this session's GPU/eBPF/multi-rank-MPI work, this COULD be
 * verified with a real compiler and a real multi-threaded run on this
 * development machine (gcc 13.3, libgomp.so.1 both present locally) --
 * see tests/fixtures/gomp_mini.c and tests/integration/run_matrix.sh's
 * gomp entry.
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

/* ── Globals / socket plumbing (identical pattern to every other hook) ── */
static int             g_sock       = -1;
static pthread_mutex_t g_sock_mutex = PTHREAD_MUTEX_INITIALIZER;
static pid_t           g_pid        = 0;

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

static pid_t gettid_compat(void) { return (pid_t)syscall(SYS_gettid); }

static void ensure_connected(void) {
    if (g_sock >= 0) return;
    const char *path = getenv("HPROFILER_SOCKET");
    if (!path) return;
    int s = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (s < 0) return;
    struct sockaddr_un addr = {0};
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, path, sizeof(addr.sun_path) - 1);
    if (connect(s, (struct sockaddr*)&addr, sizeof(addr)) == 0) {
        g_sock = s;
        g_pid  = getpid();
    } else {
        close(s);
    }
}

static void send_all(const char *buf, int n) {
    while (n > 0) {
        ssize_t r = send(g_sock, buf, (size_t)n, MSG_NOSIGNAL);
        if (r < 0) { close(g_sock); g_sock = -1; return; }
        buf += r; n -= (int)r;
    }
}

#include "../common/callstack.h"
#include "../common/codeptr_resolve.h"

static void emit_span(const char *cat, pid_t tid,
                      uint64_t start_ns, uint64_t dur_ns,
                      const char *name, const char *extra) {
    pthread_mutex_lock(&g_sock_mutex);
    ensure_connected();
    if (g_sock >= 0) {
        char buf[512]; int n;
        if (extra && *extra)
            n = snprintf(buf, sizeof(buf), "span:%s:%d:%d:%llu:%llu:%s:%s\n",
                        cat, g_pid, tid, (unsigned long long)start_ns,
                        (unsigned long long)dur_ns, name, extra);
        else
            n = snprintf(buf, sizeof(buf), "span:%s:%d:%d:%llu:%llu:%s\n",
                        cat, g_pid, tid, (unsigned long long)start_ns,
                        (unsigned long long)dur_ns, name);
        if (n > 0 && n < (int)sizeof(buf)) send_all(buf, n);
        emit_callstack(start_ns);
    }
    pthread_mutex_unlock(&g_sock_mutex);
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

/* Appends ",sym=<name>" or ",lib=<path>,offset=0x<hex>" to `buf` (which
 * must already hold the tag string built so far, null-terminated) when
 * `codeptr` resolves to something -- lets src/core/runner.py's
 * _collect_disasm() disassemble the actual user code that made this
 * call. Without this, every span from this hook has no sym=/lib= tag at
 * all, so the Source tab's "No disassembly available" is unconditional
 * for GNU-libgomp-linked binaries -- there's no ELF symbol literally
 * named "omp_parallel_region" or "omp_barrier" for objdump to find; the
 * disassembly that IS meaningful here is of the user's own call site. */
static void append_codeptr_tag(char *buf, size_t bufsz, const void *codeptr) {
    const char *sym = NULL;
    char lib[256];
    uint64_t off = 0;
    if (!hprofiler_resolve_codeptr(codeptr, &sym, lib, sizeof(lib), &off))
        return;
    size_t used = strlen(buf);
    if (used >= bufsz) return;
    if (sym) {
        snprintf(buf + used, bufsz - used, ",sym=%s", sym);
    } else if (lib[0]) {
        snprintf(buf + used, bufsz - used, ",lib=%s,offset=0x%llx",
                 lib, (unsigned long long)off);
    }
}

/* Thread-local recursion guard: our own emit_span()/dlsym() calls never
 * themselves go through OpenMP constructs, so this isn't strictly needed
 * for correctness the way cuda_hook.c's guard is (CUDA calls can trigger
 * more CUDA calls internally) -- kept anyway as a defensive no-op-cost
 * safeguard against any future libgomp internal that happens to call back
 * into one of these entry points during our own wrapper's bookkeeping. */
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
    char extra[512];
    snprintf(extra, sizeof(extra), "type=parallel_region");
    append_codeptr_tag(extra, sizeof(extra), c->codeptr_ra);
    emit_span("openmp", gettid_compat(), t0, dur, "omp_parallel_region", extra);
}

typedef void (*fn_GOMP_parallel_t)(void (*)(void *), void *, unsigned, unsigned int);
static fn_GOMP_parallel_t real_GOMP_parallel = NULL;

void GOMP_parallel(void (*fn)(void *), void *data, unsigned num_threads, unsigned int flags) {
    if (!real_GOMP_parallel) real_GOMP_parallel = (fn_GOMP_parallel_t)real_sym("GOMP_parallel");
    if (!real_GOMP_parallel) {
        /* libgomp itself is missing this symbol -- effectively impossible
         * (the profiled program couldn't have started at all without it),
         * but never call through a NULL pointer regardless: run fn
         * directly, single-threaded, rather than crash. No span emitted
         * since nothing was actually profiled correctly in this branch. */
        fn(data);
        return;
    }
    if (in_hook) { real_GOMP_parallel(fn, data, num_threads, flags); return; }
    ParallelClosure closure = { .real_fn = fn, .real_data = data,
                                .codeptr_ra = __builtin_return_address(0) };
    real_GOMP_parallel(parallel_trampoline, &closure, num_threads, flags);
}

/* ── Work-sharing loops: static/dynamic/guided/runtime × start/end ─────── */
/* GOMP_loop_*_start return true iff this thread got a non-empty iteration
 * range -- must be forwarded exactly as the real function returns it, not
 * assumed, since the caller uses it to decide whether to execute the loop
 * body at all. TLS entry timestamp consumed by whichever of GOMP_loop_end/
 * GOMP_loop_end_nowait this thread calls next -- both close out the same
 * work-sharing region regardless of which scheduling kind started it. */
static __thread uint64_t    tls_loop_start_ns = 0;
static __thread int         tls_loop_active   = 0;
static __thread const void *tls_loop_codeptr  = NULL;

/* long,long,long,long matches (start,end,incr,chunk_size) for the three
 * scheduling kinds below; GOMP_loop_runtime_start has one fewer (no
 * chunk_size -- OMP_SCHEDULE supplies it), handled separately after. */
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
/* The "nonmonotonic" variants -- NOT the plain ones above -- are what gcc
 * actually emits by default for schedule(dynamic)/schedule(guided) as of
 * GCC 13 (verified empirically: `nm -D -u` on a real compiled test binary
 * shows GOMP_loop_nonmonotonic_dynamic_start, never plain
 * GOMP_loop_dynamic_start, for an ordinary `#pragma omp for
 * schedule(dynamic)` with no explicit monotonic: modifier). Same 4-long
 * +2-outptr signature as their plain counterparts, sharing the same
 * GOMP_loop_end/_end_nowait for close-out, so the macro applies as-is.
 * Both variants are intercepted since which one a given compiler/flag
 * combination emits isn't something to assume without re-verifying. */
_LOOP_START_WRAPPER(GOMP_loop_nonmonotonic_dynamic_start)
_LOOP_START_WRAPPER(GOMP_loop_nonmonotonic_guided_start)

/* GOMP_loop_runtime_start has no chunk_size parameter (the runtime-chosen
 * schedule reads chunk size from OMP_SCHEDULE itself) -- one fewer long
 * than the other three kinds, so it can't share the macro above. Same
 * empirical caveat as above: GOMP_loop_maybe_nonmonotonic_runtime_start is
 * what schedule(runtime) actually calls by default on GCC 13; both are
 * intercepted below. */
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
        char extra[512];
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
 * The call itself blocks until every thread arrives -- timing the call
 * directly gives the wait duration, same signal
 * _add_omp_barrier_edges/rendezvous clustering (criticalpath.py) expects
 * from an OMPT-sourced "omp_barrier_implicit" span. Category "sync" and
 * the "omp_" name prefix are required for that recognition. */
typedef void (*fn_GOMP_barrier_t)(void);
static fn_GOMP_barrier_t real_GOMP_barrier = NULL;
void GOMP_barrier(void) {
    if (!real_GOMP_barrier) real_GOMP_barrier = (fn_GOMP_barrier_t)real_sym("GOMP_barrier");
    const void *ret = __builtin_return_address(0);
    uint64_t t0 = now_ns();
    if (real_GOMP_barrier) real_GOMP_barrier();
    char extra[512];
    snprintf(extra, sizeof(extra), "type=sync");
    append_codeptr_tag(extra, sizeof(extra), ret);
    emit_span("sync", gettid_compat(), t0, now_ns() - t0, "omp_barrier", extra);
}

/* ── Critical sections: acquisition-wait and hold-time as separate spans ─
 * _start's OWN call duration (call to return) is how long this thread
 * waited to acquire the section -- a real, distinct signal from how long
 * it then held it (between _start returning and _end being called),
 * which the application's own critical-section body determines. Named
 * and anonymous variants share this pattern; a single TLS slot (not a
 * stack) is used since nested critical sections under different names on
 * the same thread are rare and not distinguished here -- a documented
 * simplification, not a crash risk (worst case: hold-time attribution to
 * the wrong nesting level under that rare pattern). */
static __thread uint64_t tls_critical_enter_ns = 0;

typedef void (*fn_GOMP_critical_start_t)(void);
static fn_GOMP_critical_start_t real_GOMP_critical_start = NULL;
void GOMP_critical_start(void) {
    if (!real_GOMP_critical_start)
        real_GOMP_critical_start = (fn_GOMP_critical_start_t)real_sym("GOMP_critical_start");
    const void *ret = __builtin_return_address(0);
    uint64_t t0 = now_ns();
    if (real_GOMP_critical_start) real_GOMP_critical_start();
    uint64_t t1 = now_ns();
    char extra[512];
    snprintf(extra, sizeof(extra), "type=sync");
    append_codeptr_tag(extra, sizeof(extra), ret);
    emit_span("sync", gettid_compat(), t0, t1 - t0, "omp_critical_wait", extra);
    tls_critical_enter_ns = t1;
}

typedef void (*fn_GOMP_critical_end_t)(void);
static fn_GOMP_critical_end_t real_GOMP_critical_end = NULL;
void GOMP_critical_end(void) {
    if (!real_GOMP_critical_end)
        real_GOMP_critical_end = (fn_GOMP_critical_end_t)real_sym("GOMP_critical_end");
    uint64_t t0 = tls_critical_enter_ns;
    uint64_t now = now_ns();
    if (real_GOMP_critical_end) real_GOMP_critical_end();
    if (t0) emit_span("openmp", gettid_compat(), t0, now - t0, "omp_critical_hold", "type=critical");
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
    char extra[512];
    snprintf(extra, sizeof(extra), "type=sync,named=1");
    append_codeptr_tag(extra, sizeof(extra), ret);
    emit_span("sync", gettid_compat(), t0, t1 - t0, "omp_critical_wait", extra);
    tls_critical_enter_ns = t1;
}

typedef void (*fn_GOMP_critical_name_end_t)(void **);
static fn_GOMP_critical_name_end_t real_GOMP_critical_name_end = NULL;
void GOMP_critical_name_end(void **pptr) {
    if (!real_GOMP_critical_name_end)
        real_GOMP_critical_name_end = (fn_GOMP_critical_name_end_t)real_sym("GOMP_critical_name_end");
    uint64_t t0 = tls_critical_enter_ns;
    uint64_t now = now_ns();
    if (real_GOMP_critical_name_end) real_GOMP_critical_name_end(pptr);
    if (t0) emit_span("openmp", gettid_compat(), t0, now - t0, "omp_critical_hold", "type=critical,named=1");
}

/* ── Single ──────────────────────────────────────────────────────────────
 * Returns true only for the ONE thread that will execute the single
 * region; every other thread gets false and skips it (and typically hits
 * a barrier waiting for the executor, captured separately by
 * GOMP_barrier). Emitted as an instant, not a span -- the SIZE of the
 * single-region body isn't observable from this call site alone (unlike
 * loop start/end, there's no GOMP_single_end to pair with), only that
 * this thread was the one selected. */
typedef bool (*fn_GOMP_single_start_t)(void);
static fn_GOMP_single_start_t real_GOMP_single_start = NULL;
bool GOMP_single_start(void) {
    if (!real_GOMP_single_start)
        real_GOMP_single_start = (fn_GOMP_single_start_t)real_sym("GOMP_single_start");
    if (!real_GOMP_single_start) return false;
    bool executor = real_GOMP_single_start();
    if (executor) {
        pthread_mutex_lock(&g_sock_mutex);
        ensure_connected();
        if (g_sock >= 0) {
            char buf[128];
            int n = snprintf(buf, sizeof(buf), "inst:openmp:%d:%d:%llu:omp_single:type=single\n",
                             g_pid, gettid_compat(), (unsigned long long)now_ns());
            if (n > 0 && n < (int)sizeof(buf)) send_all(buf, n);
        }
        pthread_mutex_unlock(&g_sock_mutex);
    }
    return executor;
}

/* ── Constructor ─────────────────────────────────────────────────────────
 * No single natural entry point exists to hang initialization off (unlike
 * mpi_hook.c's MPI_Init) -- a parallel region could be the very first
 * GOMP_* call. Connect and set up call-stack resolution once at library
 * load, before any GOMP_* wrapper above can run. */
__attribute__((constructor))
static void hprofiler_gomp_init(void) {
    pthread_mutex_lock(&g_sock_mutex);
    ensure_connected();
    pthread_mutex_unlock(&g_sock_mutex);
    cs_init();
}

/* ── Destructor ──────────────────────────────────────────────────────────
 * Unlike mpi_hook.c (MPI_Finalize) or ompt_tool.c (OMPT's finalize
 * callback), there's no runtime-provided "about to shut down" hook to
 * flush on here either -- GOMP_* wrappers each call send() synchronously
 * before returning, so by ordinary process exit every emitted event's
 * send() has already been issued, but a destructor still performs the
 * same orderly half-close/drain handshake mpi_hook.c's MPI_Finalize uses
 * (shutdown(SHUT_WR) then read to EOF) rather than relying only on the
 * OS's implicit flush-on-close -- cheap insurance against the trace's
 * tail being lost if the collector's read races the process's exit. */
__attribute__((destructor))
static void hprofiler_gomp_fini(void) {
    if (g_sock >= 0) {
        shutdown(g_sock, SHUT_WR);
        char drain[64];
        while (recv(g_sock, drain, sizeof(drain), 0) > 0) {}
        close(g_sock);
        g_sock = -1;
    }
}
