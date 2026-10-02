/*
 * ROCm / HIP LD_PRELOAD hook.
 *
 * Changes over initial version (mirrors CUDA hook improvements):
 *   1. Thread-local recursion guard (in_hook).
 *   2. GPU-accurate async memcpy timing via hipEvent pairs.
 *   3. More sync hooks: hipEventSynchronize, hipDeviceReset.
 *      hipMemcpy (sync) flushes pending before the real call.
 *   4. hipModuleGetFunction → kernel name table.
 *   5. More memory hooks: hipMallocAsync/FreeAsync, hipHostMalloc/Free,
 *      hipMallocManaged. Pinned host memory tracked as pinned_memory_bytes.
 *   6. send_all() loop + truncation guard (parity with CUDA hook).
 *   7. 2048-byte span buffer (handles long HIP template kernel names).
 *   8. Dynamic alloc array (realloc-based, no hard cap).
 *   9. Memory leak detection in destructor.
 *  10. ROCTx annotation interception (roctxRangePushA/Pop/MarkA).
 *  11. hipGraphLaunch span with GPU-accurate timing.
 *  12. Host submission vs. device execution: every launch/copy/memset/
 *      graph call emits a host API span (side=cpu) plus a device span
 *      (side=gpu) -- measured by ROCprofiler-SDK when it is running
 *      (rocprof_trace.c), otherwise the hipEvent-pair proxy, labelled
 *      timing=proxy_*.
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
#include <errno.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/syscall.h>

/* ── HIP type stubs (no hip/hip_runtime.h required) ─────────────────────── */
typedef int    hipError_t;
typedef void*  hipStream_t;
typedef void*  hipEvent_t;
typedef void*  hipFunction_t;
typedef void*  hipModule_t;
typedef int    hipMemcpyKind;
typedef struct { int x, y, z; } dim3;

/* ── Globals ────────────────────────────────────────────────────────────── */
static int             g_sock        = -1;
static pthread_mutex_t g_sock_mutex  = PTHREAD_MUTEX_INITIALIZER;
static pid_t           g_pid         = 0;

/* Thread-local recursion guard (shared, hidden, with rocprof_trace.c). */
#include "hp_rocprof.h"
__thread int hp_roc_in_hook HP_HIDDEN = 0;
#define in_hook hp_roc_in_hook

/*
 * Explicit handle to libamdhip64.so opened with RTLD_GLOBAL.
 *
 * AdaptiveCpp SSCP loads libamdhip64.so via dlopen(RTLD_LOCAL), which makes
 * its symbols invisible to _real_hip_sym(...) from within this library.
 * We pre-open it with RTLD_GLOBAL in the constructor so our wrappers can
 * always resolve the real HIP symbols regardless of load order.
 */
static void *g_hip_lib = NULL;

static void *_real_hip_sym(const char *name) {
    /* Use the explicit handle opened with RTLD_GLOBAL in the constructor.
     * dlsym(specific_handle, name) bypasses our LD_PRELOAD wrappers and
     * returns the real symbol from libamdhip64.so directly. */
    if (!g_hip_lib) {
        g_hip_lib = dlopen("libamdhip64.so",   RTLD_LAZY | RTLD_GLOBAL);
        if (!g_hip_lib)
            g_hip_lib = dlopen("libamdhip64.so.5", RTLD_LAZY | RTLD_GLOBAL);
    }
    return g_hip_lib ? dlsym(g_hip_lib, name) : NULL;
}

/* ── Core helpers ───────────────────────────────────────────────────────── */
static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}
static pid_t gettid_compat(void) { return (pid_t)syscall(SYS_gettid); }

static void ensure_connected(void) {
    if (g_sock >= 0) return;
    const char *path = getenv("HPROFILER_SOCKET");
    if (!path) {
        static int warned = 0;
        if (!warned) {
            warned = 1;
            fprintf(stderr, "[hprofiler/rocm] HPROFILER_SOCKET not set — no events will be recorded\n");
        }
        return;
    }
    int s = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (s < 0) return;
    struct sockaddr_un addr = {0};
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, path, sizeof(addr.sun_path) - 1);
    if (connect(s, (struct sockaddr*)&addr, sizeof(addr)) == 0) {
        g_sock = s; g_pid = getpid();
    } else {
        close(s);
        fprintf(stderr, "[hprofiler/rocm] connect(%s) failed: %s\n", path, strerror(errno));
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

static void emit_span(const char *cat, pid_t tid, uint64_t start_ns,
                      uint64_t dur_ns, const char *name, const char *extra) {
    pthread_mutex_lock(&g_sock_mutex);
    ensure_connected();
    if (g_sock >= 0) {
        char buf[2048]; int n;
        if (extra && *extra)
            n = snprintf(buf, sizeof(buf),
                "span:%s:%d:%d:%llu:%llu:%s:%s\n",
                cat, g_pid, (int)tid,
                (unsigned long long)start_ns, (unsigned long long)dur_ns,
                name, extra);
        else
            n = snprintf(buf, sizeof(buf),
                "span:%s:%d:%d:%llu:%llu:%s\n",
                cat, g_pid, (int)tid,
                (unsigned long long)start_ns, (unsigned long long)dur_ns,
                name);
        /* snprintf returns >= sizeof(buf) when truncated. Rather than
         * silently dropping the whole span, retry with a shortened name so
         * the event -- correct timing, full tags -- still reaches the
         * trace (heavily-templated C++ kernel names can alone exceed this
         * buffer). */
        if (n >= (int)sizeof(buf)) {
            char short_name[200];
            size_t name_len = strlen(name);
            if (name_len > sizeof(short_name) - 4) {
                memcpy(short_name, name, sizeof(short_name) - 4);
                memcpy(short_name + sizeof(short_name) - 4, "...", 4);
            } else {
                memcpy(short_name, name, name_len + 1);
            }
            if (extra && *extra)
                n = snprintf(buf, sizeof(buf),
                    "span:%s:%d:%d:%llu:%llu:%s:%s\n",
                    cat, g_pid, (int)tid,
                    (unsigned long long)start_ns, (unsigned long long)dur_ns,
                    short_name, extra);
            else
                n = snprintf(buf, sizeof(buf),
                    "span:%s:%d:%d:%llu:%llu:%s\n",
                    cat, g_pid, (int)tid,
                    (unsigned long long)start_ns, (unsigned long long)dur_ns,
                    short_name);
        }
        if (n > 0 && n < (int)sizeof(buf)) send_all(buf, n);
        emit_callstack(start_ns);
    }
    pthread_mutex_unlock(&g_sock_mutex);
}

static void emit_ctr(const char *cat, const char *name,
                     int64_t value, const char *unit) {
    pthread_mutex_lock(&g_sock_mutex);
    ensure_connected();
    if (g_sock >= 0) {
        char buf[256];
        int n = snprintf(buf, sizeof(buf),
                         "ctr:%s:%d:%llu:%s:%lld:%s\n",
                         cat, g_pid, (unsigned long long)now_ns(),
                         name, (long long)value, unit);
        if (n > 0 && n < (int)sizeof(buf)) send_all(buf, n);
    }
    pthread_mutex_unlock(&g_sock_mutex);
}

/* ── Kernel name table (hipModuleGetFunction → name) ──────────────────────── */
#define KNAME_MAP_CAP 512
typedef struct { hipFunction_t fn; char name[2048]; } KNameEntry;
static KNameEntry      g_knames[KNAME_MAP_CAP];
static int             g_kname_n = 0;
static pthread_mutex_t g_kname_mutex = PTHREAD_MUTEX_INITIALIZER;

static const char *resolve_name(const void *fn) {
    if (!fn) return "<unknown>";
    pthread_mutex_lock(&g_kname_mutex);
    for (int i = 0; i < g_kname_n; i++) {
        if (g_knames[i].fn == (hipFunction_t)fn) {
            const char *p = g_knames[i].name;
            pthread_mutex_unlock(&g_kname_mutex);
            return p;
        }
    }
    pthread_mutex_unlock(&g_kname_mutex);
    Dl_info info;
    if (dladdr(fn, &info) && info.dli_sname) return info.dli_sname;
    return "<jit-kernel>";
}

/* ── GPU device memory tracking ─────────────────────────────────────────── */
typedef struct { uintptr_t ptr; size_t sz; } AllocRec;

static AllocRec       *g_allocs    = NULL;
static int             g_alloc_cap = 0;
static int             g_alloc_n   = 0;
static int64_t         g_gpu_mem   = 0;
static pthread_mutex_t g_alloc_mutex = PTHREAD_MUTEX_INITIALIZER;

static void mem_track_add(void *ptr, size_t sz) {
    int64_t total;
    pthread_mutex_lock(&g_alloc_mutex);
    if (g_alloc_n >= g_alloc_cap) {
        int newcap = g_alloc_cap ? g_alloc_cap * 2 : 256;
        AllocRec *tmp = realloc(g_allocs, (size_t)newcap * sizeof(AllocRec));
        if (tmp) { g_allocs = tmp; g_alloc_cap = newcap; }
    }
    if (g_alloc_n < g_alloc_cap) {
        g_allocs[g_alloc_n].ptr = (uintptr_t)ptr;
        g_allocs[g_alloc_n].sz  = sz;
        g_alloc_n++;
    }
    g_gpu_mem += (int64_t)sz;
    total = g_gpu_mem;
    pthread_mutex_unlock(&g_alloc_mutex);
    emit_ctr("memory", "gpu_memory_bytes", total, "bytes");
}

static void mem_track_rem(void *ptr) {
    int64_t total;
    pthread_mutex_lock(&g_alloc_mutex);
    for (int i = 0; i < g_alloc_n; i++) {
        if (g_allocs[i].ptr == (uintptr_t)ptr) {
            g_gpu_mem -= (int64_t)g_allocs[i].sz;
            g_allocs[i] = g_allocs[--g_alloc_n];
            break;
        }
    }
    total = g_gpu_mem;
    pthread_mutex_unlock(&g_alloc_mutex);
    emit_ctr("memory", "gpu_memory_bytes", total, "bytes");
}

/* ── Pinned host memory tracking ─────────────────────────────────────────── */
static AllocRec       *g_pin_allocs = NULL;
static int             g_pin_cap   = 0;
static int             g_pin_n     = 0;
static int64_t         g_pin_mem   = 0;
static pthread_mutex_t g_pin_mutex  = PTHREAD_MUTEX_INITIALIZER;

static void pin_track_add(void *ptr, size_t sz) {
    int64_t total;
    pthread_mutex_lock(&g_pin_mutex);
    if (g_pin_n >= g_pin_cap) {
        int newcap = g_pin_cap ? g_pin_cap * 2 : 256;
        AllocRec *tmp = realloc(g_pin_allocs, (size_t)newcap * sizeof(AllocRec));
        if (tmp) { g_pin_allocs = tmp; g_pin_cap = newcap; }
    }
    if (g_pin_n < g_pin_cap) {
        g_pin_allocs[g_pin_n].ptr = (uintptr_t)ptr;
        g_pin_allocs[g_pin_n].sz  = sz;
        g_pin_n++;
    }
    g_pin_mem += (int64_t)sz;
    total = g_pin_mem;
    pthread_mutex_unlock(&g_pin_mutex);
    emit_ctr("memory", "pinned_memory_bytes", total, "bytes");
}

static void pin_track_rem(void *ptr) {
    int64_t total;
    pthread_mutex_lock(&g_pin_mutex);
    for (int i = 0; i < g_pin_n; i++) {
        if (g_pin_allocs[i].ptr == (uintptr_t)ptr) {
            g_pin_mem -= (int64_t)g_pin_allocs[i].sz;
            g_pin_allocs[i] = g_pin_allocs[--g_pin_n];
            break;
        }
    }
    total = g_pin_mem;
    pthread_mutex_unlock(&g_pin_mutex);
    emit_ctr("memory", "pinned_memory_bytes", total, "bytes");
}

/* ── Stream ID assignment ────────────────────────────────────────────────── */
/* Deterministic small-integer ID derived from the pointer VALUE (MurmurHash3
 * finalizer/fmix64), not from order-of-first-observation -- same technique
 * and rationale as cuda_hook.c's get_stream_id() (kept identical there and
 * here for consistency even though no AMD-side NCCL-equivalent hook exists
 * yet to cross-reference). Also removes the previous STREAM_MAP_CAP
 * overflow behavior (silently collapsing to stream=-1 past 256 streams). */
static int handle_id(const void *h) {
    if (!h) return 0;
    uint64_t v = (uint64_t)(uintptr_t)h;
    v ^= v >> 33; v *= 0xff51afd7ed558ccdULL;
    v ^= v >> 33; v *= 0xc4ceb9fe1a85ec53ULL;
    v ^= v >> 33;
    return (int)(v % 999983) + 1;
}
static int get_stream_id(const void *stream) { return handle_id(stream); }

/* ── GPU-accurate timing via hipEvent pairs ──────────────────────────────── */
#define MAX_PENDING 512

typedef hipError_t (*fn_EvCreate_t) (hipEvent_t *);
typedef hipError_t (*fn_EvRecord_t) (hipEvent_t, hipStream_t);
typedef hipError_t (*fn_EvElapsed_t)(float *, hipEvent_t, hipEvent_t);
typedef hipError_t (*fn_EvDestroy_t)(hipEvent_t);
typedef hipError_t (*fn_EvSync_t)   (hipEvent_t);

static fn_EvCreate_t  f_evCreate  = NULL;
static fn_EvRecord_t  f_evRecord  = NULL;
static fn_EvElapsed_t f_evElapsed = NULL;
static fn_EvDestroy_t f_evDestroy = NULL;
static fn_EvSync_t    f_evSync    = NULL;

static int ev_api_ok(void) {
    if (!f_evCreate) {
        f_evCreate  = (fn_EvCreate_t) _real_hip_sym("hipEventCreate");
        f_evRecord  = (fn_EvRecord_t) _real_hip_sym("hipEventRecord");
        f_evElapsed = (fn_EvElapsed_t)_real_hip_sym("hipEventElapsedTime");
        f_evDestroy = (fn_EvDestroy_t)_real_hip_sym("hipEventDestroy");
        f_evSync    = (fn_EvSync_t)   _real_hip_sym("hipEventSynchronize");
    }
    return f_evCreate && f_evRecord && f_evElapsed && f_evDestroy && f_evSync;
}

/* ── Exec-start calibration: real GPU-timeline kernel-execution start ────────
 * (xs= tag), distinct from start_ns/t0 (the CPU-side launch-CALL time) ─────
 * Mirrors cuda_hook.c's identical mechanism -- see its comment for the full
 * rationale (hipEvent_t has the same FIFO-completion semantics as
 * cudaEvent_t, and hipEventElapsedTime works across streams the same way).
 * xs= is additive and never changes what start_ns/duration_ns mean.
 *
 * NOT independently verified against real kernel execution on this
 * development machine (no AMD GPU present -- see DOCUMENTATION.md's Known
 * Limitations); compile-checked only. */
static hipEvent_t      g_calib_event  = NULL;
static uint64_t        g_calib_cpu_ns = 0;
static int             g_calib_state  = 0;   /* 0=not tried, 1=ok, -1=failed */
static pthread_mutex_t g_calib_mutex  = PTHREAD_MUTEX_INITIALIZER;

static int exec_start_calibrate_if_needed(void) {
    pthread_mutex_lock(&g_calib_mutex);
    if (g_calib_state != 0) {
        int ok = (g_calib_state == 1);
        pthread_mutex_unlock(&g_calib_mutex);
        return ok;
    }
    int ok = 0;
    if (ev_api_ok() && f_evCreate(&g_calib_event) == 0) {
        if (f_evRecord(g_calib_event, NULL) == 0 && f_evSync(g_calib_event) == 0) {
            g_calib_cpu_ns = now_ns();
            ok = 1;
        } else {
            f_evDestroy(g_calib_event);
            g_calib_event = NULL;
        }
    }
    g_calib_state = ok ? 1 : -1;
    pthread_mutex_unlock(&g_calib_mutex);
    return ok;
}

/* Returns 1 and fills *exec_start_ns from ev_s if computable; 0 if
 * calibration or the elapsed-time query failed. ev_s must have already
 * completed on the GPU (true for any ev_s whose paired ev_e has already
 * been successfully synced, since events on one stream complete in FIFO
 * order). */
static int compute_exec_start_ns(hipEvent_t ev_s, uint64_t *exec_start_ns) {
    if (!exec_start_calibrate_if_needed()) return 0;
    float ms = 0.0f;
    if (f_evElapsed(&ms, g_calib_event, ev_s) != 0 || ms < 0.0f) return 0;
    *exec_start_ns = g_calib_cpu_ns + (uint64_t)(ms * 1e6f);
    return 1;
}

typedef struct {
    hipEvent_t   ev_start;
    hipEvent_t   ev_end;
    hipStream_t  stream;
    char         cat[32];
    char         kname[2048];
    char         extra[256];
    uint64_t     cpu_start_ns;
    pid_t        tid;
} PendingKernel;

static PendingKernel   g_pk[MAX_PENDING];
static int             g_pk_n = 0;
static pthread_mutex_t g_pk_mutex = PTHREAD_MUTEX_INITIALIZER;

static void pk_flush(hipStream_t flush_stream, int all_streams) {
    if (!ev_api_ok()) return;

    typedef struct {
        hipEvent_t ev_s, ev_e;
        char cat[32], kname[2048], extra[256];
        uint64_t t0;
        pid_t tid;
    } Local;
    Local todo[MAX_PENDING];
    int ntodo = 0;

    pthread_mutex_lock(&g_pk_mutex);
    int keep = 0;
    for (int i = 0; i < g_pk_n; i++) {
        PendingKernel *pk = &g_pk[i];
        if (!all_streams && pk->stream != flush_stream) {
            if (keep != i) g_pk[keep] = *pk;
            keep++;
        } else {
            Local *l = &todo[ntodo++];
            l->ev_s = pk->ev_start; l->ev_e = pk->ev_end;
            l->t0   = pk->cpu_start_ns; l->tid  = pk->tid;
            strncpy(l->cat,   pk->cat,   31);  l->cat[31]   = '\0';
            strncpy(l->kname, pk->kname, 2047); l->kname[2047] = '\0';
            strncpy(l->extra, pk->extra, 255); l->extra[255] = '\0';
        }
    }
    g_pk_n = keep;
    pthread_mutex_unlock(&g_pk_mutex);

    for (int i = 0; i < ntodo; i++) {
        Local *l = &todo[i];
        float ms = 0.0f;
        int ok = (f_evSync(l->ev_e) == 0 &&
                  f_evElapsed(&ms, l->ev_s, l->ev_e) == 0 && ms >= 0.0f);
        /* Must compute BEFORE destroying ev_s below -- see cuda_hook.c's
         * matching comment for why this is safe here. */
        uint64_t xs_ns = 0;
        int has_xs = ok && compute_exec_start_ns(l->ev_s, &xs_ns);
        f_evDestroy(l->ev_s);
        f_evDestroy(l->ev_e);
        if (ok) {
            /* timing=proxy_event: GPU-measured duration (event pair),
             * start at the host submission time t0. */
            char final_extra[340];
            const char *sep = l->extra[0] ? "," : "";
            if (has_xs)
                snprintf(final_extra, sizeof(final_extra), "%s%stiming=proxy_event,xs=%llu",
                         l->extra, sep, (unsigned long long)xs_ns);
            else
                snprintf(final_extra, sizeof(final_extra), "%s%stiming=proxy_event",
                         l->extra, sep);
            emit_span(l->cat, l->tid, l->t0, (uint64_t)(ms * 1e6f),
                      l->kname, final_extra);
        } else {
            /* hipEventSynchronize/hipEventElapsedTime failed -- a kernel
             * that ran to completion on the GPU must not simply vanish
             * from the trace. Fall back to wall-clock time from launch to
             * this flush (an upper-bound approximation, since it may
             * include time for other kernels queued after this one), marked
             * distinctly from GPU-accurate timing. */
            char marked[300];
            if (l->extra[0])
                snprintf(marked, sizeof(marked), "%s,timing=proxy_flush", l->extra);
            else
                snprintf(marked, sizeof(marked), "timing=proxy_flush");
            emit_span(l->cat, l->tid, l->t0, now_ns() - l->t0, l->kname, marked);
        }
    }
}

static int pk_try_begin(hipStream_t stream, hipEvent_t *ev_s, hipEvent_t *ev_e) {
    /* Calibrate the xs= reference event HERE, before this kernel's own
     * ev_s is recorded below -- same fix and same rationale as
     * cuda_hook.c's pk_try_begin: exec_start_calibrate_if_needed()
     * SYNCS the calibration event before returning, so deferring it to
     * first-use inside pk_flush() (the previous approach) guaranteed it
     * would postdate every kernel in whatever batch triggered it there,
     * silently omitting xs= for that entire first flushed batch. Calling
     * it here, before this process's very first kernel launch has even
     * been recorded, means it precedes every real kernel's ev_s from
     * then on. Idempotent (guarded by g_calib_state) -- a no-op after
     * the first call. */
    exec_start_calibrate_if_needed();
    *ev_s = *ev_e = NULL;
    if (!ev_api_ok()) return 0;
    if (f_evCreate(ev_s) != 0) return 0;
    if (f_evCreate(ev_e) != 0) { f_evDestroy(*ev_s); *ev_s = NULL; return 0; }
    if (f_evRecord(*ev_s, stream) != 0) {
        f_evDestroy(*ev_s); f_evDestroy(*ev_e);
        *ev_s = *ev_e = NULL; return 0;
    }
    return 1;
}

static void pk_commit(hipEvent_t ev_s, hipEvent_t ev_e,
                      hipStream_t stream,
                      const char *cat, const char *kname, const char *extra,
                      uint64_t t0, pid_t tid) {
    f_evRecord(ev_e, stream);
    pthread_mutex_lock(&g_pk_mutex);
    if (g_pk_n < MAX_PENDING) {
        PendingKernel *pk = &g_pk[g_pk_n++];
        pk->ev_start = ev_s; pk->ev_end = ev_e;
        pk->stream = stream; pk->cpu_start_ns = t0; pk->tid = tid;
        snprintf(pk->cat,   sizeof(pk->cat),   "%s", cat);
        snprintf(pk->kname, sizeof(pk->kname), "%s", kname);
        snprintf(pk->extra, sizeof(pk->extra), "%s", extra);
        pthread_mutex_unlock(&g_pk_mutex);
    } else {
        pthread_mutex_unlock(&g_pk_mutex);
        f_evDestroy(ev_s); f_evDestroy(ev_e);
        /* Pending-kernel queue full -- fall back to CPU-side timing,
         * clearly marked so it isn't mistaken for a GPU-accurate one. */
        char marked[300];
        if (extra && *extra)
            snprintf(marked, sizeof(marked), "%s,timing=proxy_host", extra);
        else
            snprintf(marked, sizeof(marked), "timing=proxy_host");
        emit_span(cat, tid, t0, now_ns() - t0, kname, marked);
    }
}

/* Entry points for rocprof_trace.c (see hp_rocprof.h). */
void hp_roc_emit_span(const char *cat, pid_t tid, uint64_t start_ns,
                      uint64_t dur_ns, const char *name, const char *extra) {
    emit_span(cat, tid, start_ns, dur_ns, name, extra);
}

void hp_roc_emit_line(const char *line) {
    pthread_mutex_lock(&g_sock_mutex);
    ensure_connected();
    if (g_sock >= 0) send_all(line, (int)strlen(line));
    pthread_mutex_unlock(&g_sock_mutex);
}

/* ── Host submission / device activity ──────────────────────────────────────
 * Same model as cuda_hook.c: every intercepted call that submits device
 * work emits a host span (side=cpu, timing=host, lid=) and the device work
 * is a separate side=gpu span -- from ROCprofiler-SDK when it is running
 * (rocprof_trace.c; lid is pushed as the external correlation id around
 * the real call, so its records carry it), otherwise the hipEvent-pair
 * proxy labelled timing=proxy_*. HPROFILER_DEVICE_ACTIVITY=auto|off|both as
 * for CUDA. */
static uint64_t g_lid_counter = 0;
static pthread_once_t g_native_once = PTHREAD_ONCE_INIT;
static int g_device_mode = 0;      /* 0 auto, 1 off, 2 both */

static void native_init_once(void) {
    const char *m = getenv("HPROFILER_DEVICE_ACTIVITY");
    if (m && !strcmp(m, "off")) { g_device_mode = 1; return; }   /* rocprof_trace.c reports it */
    if (m && !strcmp(m, "both")) g_device_mode = 2;
    /* ROCprofiler-SDK configures its tools while the HIP runtime
     * initializes, which HIP does lazily on its first API call. Force that
     * now, so the decision below (native vs. proxy) is made with the tracer
     * already running even when the application's first HIP call is a
     * launch. */
    typedef hipError_t (*fn_t)(int *);
    fn_t count = (fn_t)_real_hip_sym("hipGetDeviceCount");
    int n = 0, saved = in_hook;
    in_hook = 1;
    if (count) count(&n);
    in_hook = saved;
    if (!hp_roc_active()) {
        char line[160];
        snprintf(line, sizeof(line), "gpuact:%d:rocprofiler:status=unavailable,reason=%s\n", (int)getpid(),
                 hp_roc_compiled() ? "sdk_not_loaded" : "built_without_rocprofiler_sdk_headers");
        hp_roc_emit_line(line);
    }
}

static void native_ensure(void) { pthread_once(&g_native_once, native_init_once); }
static int want_proxy(void) { return !hp_roc_active() || g_device_mode == 2; }

typedef struct {
    uint64_t   lid, t0, t1;
    pid_t      tid;
    int        proxy, gpu_ok;
    hipEvent_t ev_s, ev_e;
} Sub;

static void sub_begin(Sub *u, hipStream_t stream, int device_work) {
    native_ensure();
    u->lid    = __atomic_add_fetch(&g_lid_counter, 1, __ATOMIC_RELAXED);
    u->tid    = gettid_compat();
    u->proxy  = device_work && want_proxy();
    u->gpu_ok = u->proxy ? pk_try_begin(stream, &u->ev_s, &u->ev_e) : 0;
    hp_roc_push(u->lid);       /* after pk_try_begin: its events aren't app work */
    u->t0 = now_ns();
}

static void sub_end(Sub *u) {
    u->t1 = now_ns();
    hp_roc_pop();
}

static void sub_host(const Sub *u, const char *cat, const char *api,
                     const char *tags, int ret) {
    char x[512];
    int n = snprintf(x, sizeof(x), "%s,side=cpu,rt=rocm,timing=host,lid=%llu,sid=%llu",
                     tags, (unsigned long long)u->lid,
                     (unsigned long long)(((uint64_t)(uint32_t)getpid() << 32) | (u->lid & 0xffffffffULL)));
    if (ret != 0 && n > 0 && n < (int)sizeof(x))
        snprintf(x + n, sizeof(x) - (size_t)n, ",err=%d", ret);
    emit_span(cat, u->tid, u->t0, u->t1 - u->t0, api, x);
}

static void sub_proxy(Sub *u, hipStream_t stream, const char *cat,
                      const char *name, const char *tags) {
    if (!u->proxy) return;
    char x[256];
    snprintf(x, sizeof(x), "%s,side=gpu,rt=rocm,lid=%llu", tags, (unsigned long long)u->lid);
    if (u->gpu_ok) {
        pk_commit(u->ev_s, u->ev_e, stream, cat, name, x, u->t0, u->tid);
    } else {
        char y[300];
        snprintf(y, sizeof(y), "%s,timing=proxy_host", x);
        emit_span(cat, u->tid, u->t0, u->t1 - u->t0, name, y);
    }
}

/* A failed submission produced no device work. */
static void sub_abort(Sub *u) {
    if (u->gpu_ok) { f_evDestroy(u->ev_s); f_evDestroy(u->ev_e); u->gpu_ok = 0; }
}

static const char *memcpy_dir(hipMemcpyKind kind) {
    static const char *names[] = {"HtoH", "HtoD", "DtoH", "DtoD", "Default"};
    return (kind >= 0 && kind <= 4) ? names[kind] : "Unknown";
}

/* ── HIP Runtime API wrappers ───────────────────────────────────────────── */

hipError_t hipLaunchKernel(const void *fn, dim3 grid, dim3 block,
                            void **args, size_t sharedMem, hipStream_t stream) {
    typedef hipError_t (*fn_t)(const void*, dim3, dim3, void**, size_t, hipStream_t);
    static fn_t real = NULL;
    if (!real) real = (fn_t)_real_hip_sym("hipLaunchKernel");
    if (!real) return -1;
    if (in_hook) return real(fn, grid, block, args, sharedMem, stream);
    in_hook = 1;

    const char *kname = resolve_name(fn);
    int sid = get_stream_id(stream);
    char host[64], dev[160];
    snprintf(host, sizeof(host), "type=launch,op=kernel,stream=%d", sid);
    snprintf(dev, sizeof(dev), "type=kernel,op=kernel,stream=%d,grid=%dx%dx%d,block=%dx%dx%d", sid, grid.x, grid.y, grid.z, block.x, block.y, block.z);

    Sub u;
    sub_begin(&u, stream, 1);
    hipError_t ret = real(fn, grid, block, args, sharedMem, stream);
    sub_end(&u);
    sub_host(&u, "rocm", "hipLaunchKernel", host, (int)ret);
    if (ret == 0) sub_proxy(&u, stream, "rocm", kname, dev);
    else          sub_abort(&u);

    in_hook = 0;
    return ret;
}

hipError_t hipLaunchKernelGGL(hipFunction_t fn,
                               dim3 grid, dim3 block,
                               unsigned int sharedMem, hipStream_t stream,
                               void **kernelParams) {
    typedef hipError_t (*fn_t)(hipFunction_t, dim3, dim3, unsigned int, hipStream_t, void**);
    static fn_t real = NULL;
    if (!real) real = (fn_t)_real_hip_sym("hipLaunchKernelGGL");
    if (!real) return -1;
    if (in_hook) return real(fn, grid, block, sharedMem, stream, kernelParams);
    in_hook = 1;

    const char *kname = resolve_name(fn);
    int sid = get_stream_id(stream);
    char host[64], dev[160];
    snprintf(host, sizeof(host), "type=launch,op=kernel,stream=%d", sid);
    snprintf(dev, sizeof(dev), "type=kernel,op=kernel,stream=%d,grid=%dx%dx%d,block=%dx%dx%d", sid, grid.x, grid.y, grid.z, block.x, block.y, block.z);

    Sub u;
    sub_begin(&u, stream, 1);
    hipError_t ret = real(fn, grid, block, sharedMem, stream, kernelParams);
    sub_end(&u);
    sub_host(&u, "rocm", "hipLaunchKernelGGL", host, (int)ret);
    if (ret == 0) sub_proxy(&u, stream, "rocm", kname, dev);
    else          sub_abort(&u);

    in_hook = 0;
    return ret;
}

hipError_t hipModuleLaunchKernel(hipFunction_t f,
                                  unsigned int gx, unsigned int gy, unsigned int gz,
                                  unsigned int bx, unsigned int by, unsigned int bz,
                                  unsigned int sharedMem, hipStream_t stream,
                                  void **kernelParams, void **extra_params) {
    typedef hipError_t (*fn_t)(hipFunction_t, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, hipStream_t, void**, void**);
    static fn_t real = NULL;
    if (!real) real = (fn_t)_real_hip_sym("hipModuleLaunchKernel");
    if (!real) return -1;
    if (in_hook) return real(f, gx,gy,gz, bx,by,bz, sharedMem, stream, kernelParams, extra_params);
    in_hook = 1;

    const char *kname = resolve_name(f);
    int sid = get_stream_id(stream);
    char host[64], dev[160];
    snprintf(host, sizeof(host), "type=launch,op=kernel,stream=%d", sid);
    snprintf(dev, sizeof(dev), "type=kernel,op=kernel,stream=%d,grid=%ux%ux%u,block=%ux%ux%u", sid, gx, gy, gz, bx, by, bz);

    Sub u;
    sub_begin(&u, stream, 1);
    hipError_t ret = real(f, gx,gy,gz, bx,by,bz, sharedMem, stream, kernelParams, extra_params);
    sub_end(&u);
    sub_host(&u, "rocm", "hipModuleLaunchKernel", host, (int)ret);
    if (ret == 0) sub_proxy(&u, stream, "rocm", kname, dev);
    else          sub_abort(&u);

    in_hook = 0;
    return ret;
}

hipError_t hipMemcpy(void *dst, const void *src, size_t size, hipMemcpyKind kind) {
    typedef hipError_t (*fn_t)(void*, const void*, size_t, hipMemcpyKind);
    static fn_t real = NULL;
    if (!real) real = (fn_t)_real_hip_sym("hipMemcpy");
    if (!real) return -1;
    if (in_hook) return real(dst, src, size, kind);
    in_hook = 1;

    /* Synchronous memcpy implies all prior GPU work is complete — flush pending */
    pk_flush(NULL, 1);

    /* Blocking: the call spans the transfer itself (type=memcpy, category
     * memory); ROCprofiler-SDK reports the device-side copy when active. */
    char host[128];
    snprintf(host, sizeof(host), "type=memcpy,op=memcpy,dir=%s,bytes=%zu", memcpy_dir(kind), size);
    Sub u;
    sub_begin(&u, NULL, 0);
    hipError_t ret = real(dst, src, size, kind);
    sub_end(&u);
    sub_host(&u, "memory", "hipMemcpy", host, (int)ret);

    in_hook = 0;
    return ret;
}

hipError_t hipMemcpyAsync(void *dst, const void *src, size_t size,
                           hipMemcpyKind kind, hipStream_t stream) {
    typedef hipError_t (*fn_t)(void*, const void*, size_t, hipMemcpyKind, hipStream_t);
    static fn_t real = NULL;
    if (!real) real = (fn_t)_real_hip_sym("hipMemcpyAsync");
    if (!real) return -1;
    if (in_hook) return real(dst, src, size, kind, stream);
    in_hook = 1;

    int sid = get_stream_id(stream);
    const char *dir = memcpy_dir(kind);
    char host[128], dev[128], dname[32];
    snprintf(host, sizeof(host), "type=launch,op=memcpy,dir=%s,bytes=%zu,stream=%d", dir, size, sid);
    snprintf(dev, sizeof(dev), "type=memcpy_async,op=memcpy,dir=%s,bytes=%zu,stream=%d", dir, size, sid);
    snprintf(dname, sizeof(dname), "memcpy %s", dir);

    Sub u;
    sub_begin(&u, stream, 1);
    hipError_t ret = real(dst, src, size, kind, stream);
    sub_end(&u);
    sub_host(&u, "memory", "hipMemcpyAsync", host, (int)ret);
    if (ret == 0) sub_proxy(&u, stream, "memory", dname, dev);
    else          sub_abort(&u);

    in_hook = 0;
    return ret;
}

hipError_t hipMemcpyHtoD(void *dst, const void *src, size_t size) {
    typedef hipError_t (*fn_t)(void*, const void*, size_t);
    static fn_t real = NULL;
    if (!real) real = (fn_t)_real_hip_sym("hipMemcpyHtoD");
    if (!real) return -1;
    if (in_hook) return real(dst, src, size);
    in_hook = 1;


    /* Blocking: the call spans the transfer itself (type=memcpy, category
     * memory); ROCprofiler-SDK reports the device-side copy when active. */
    char host[128];
    snprintf(host, sizeof(host), "type=memcpy,op=memcpy,dir=%s,bytes=%zu", "HtoD", size);
    Sub u;
    sub_begin(&u, NULL, 0);
    hipError_t ret = real(dst, src, size);
    sub_end(&u);
    sub_host(&u, "memory", "hipMemcpyHtoD", host, (int)ret);

    in_hook = 0;
    return ret;
}

hipError_t hipMemcpyDtoH(void *dst, const void *src, size_t size) {
    typedef hipError_t (*fn_t)(void*, const void*, size_t);
    static fn_t real = NULL;
    if (!real) real = (fn_t)_real_hip_sym("hipMemcpyDtoH");
    if (!real) return -1;
    if (in_hook) return real(dst, src, size);
    in_hook = 1;


    /* Blocking: the call spans the transfer itself (type=memcpy, category
     * memory); ROCprofiler-SDK reports the device-side copy when active. */
    char host[128];
    snprintf(host, sizeof(host), "type=memcpy,op=memcpy,dir=%s,bytes=%zu", "DtoH", size);
    Sub u;
    sub_begin(&u, NULL, 0);
    hipError_t ret = real(dst, src, size);
    sub_end(&u);
    sub_host(&u, "memory", "hipMemcpyDtoH", host, (int)ret);

    in_hook = 0;
    return ret;
}

hipError_t hipMalloc(void **ptr, size_t size) {
    typedef hipError_t (*fn_t)(void**, size_t);
    static fn_t real = NULL;
    if (!real) real = (fn_t)_real_hip_sym("hipMalloc");
    if (!real) return -1;
    if (in_hook) return real(ptr, size);
    in_hook = 1;

    char extra[64];
    snprintf(extra, sizeof(extra), "type=alloc,bytes=%zu", size);
    uint64_t t0 = now_ns();
    hipError_t ret = real(ptr, size);
    if (ret == 0 && ptr && *ptr) mem_track_add(*ptr, size);
    emit_span("memory", gettid_compat(), t0, now_ns()-t0, "hipMalloc", extra);

    in_hook = 0;
    return ret;
}

hipError_t hipMallocManaged(void **ptr, size_t size, unsigned int flags) {
    typedef hipError_t (*fn_t)(void**, size_t, unsigned int);
    static fn_t real = NULL;
    if (!real) real = (fn_t)_real_hip_sym("hipMallocManaged");
    if (!real) return -1;
    if (in_hook) return real(ptr, size, flags);
    in_hook = 1;

    char extra[64];
    snprintf(extra, sizeof(extra), "type=alloc_managed,bytes=%zu", size);
    uint64_t t0 = now_ns();
    hipError_t ret = real(ptr, size, flags);
    if (ret == 0 && ptr && *ptr) mem_track_add(*ptr, size);
    emit_span("memory", gettid_compat(), t0, now_ns()-t0, "hipMallocManaged", extra);

    in_hook = 0;
    return ret;
}

hipError_t hipMallocAsync(void **ptr, size_t size, hipStream_t stream) {
    typedef hipError_t (*fn_t)(void**, size_t, hipStream_t);
    static fn_t real = NULL;
    if (!real) real = (fn_t)_real_hip_sym("hipMallocAsync");
    if (!real) return -1;
    if (in_hook) return real(ptr, size, stream);
    in_hook = 1;

    char extra[128];
    snprintf(extra, sizeof(extra), "type=alloc_async,bytes=%zu,stream=%d",
             size, get_stream_id(stream));
    uint64_t t0 = now_ns();
    hipError_t ret = real(ptr, size, stream);
    if (ret == 0 && ptr && *ptr) mem_track_add(*ptr, size);
    emit_span("memory", gettid_compat(), t0, now_ns()-t0, "hipMallocAsync", extra);

    in_hook = 0;
    return ret;
}

hipError_t hipFree(void *ptr) {
    typedef hipError_t (*fn_t)(void*);
    static fn_t real = NULL;
    if (!real) real = (fn_t)_real_hip_sym("hipFree");
    if (!real) return -1;
    if (in_hook) return real(ptr);
    in_hook = 1;

    mem_track_rem(ptr);
    uint64_t t0 = now_ns();
    hipError_t ret = real(ptr);
    emit_span("memory", gettid_compat(), t0, now_ns()-t0, "hipFree", "type=free");

    in_hook = 0;
    return ret;
}

hipError_t hipFreeAsync(void *ptr, hipStream_t stream) {
    typedef hipError_t (*fn_t)(void*, hipStream_t);
    static fn_t real = NULL;
    if (!real) real = (fn_t)_real_hip_sym("hipFreeAsync");
    if (!real) return -1;
    if (in_hook) return real(ptr, stream);
    in_hook = 1;

    mem_track_rem(ptr);
    char extra[64];
    snprintf(extra, sizeof(extra), "type=free_async,stream=%d", get_stream_id(stream));
    uint64_t t0 = now_ns();
    hipError_t ret = real(ptr, stream);
    emit_span("memory", gettid_compat(), t0, now_ns()-t0, "hipFreeAsync", extra);

    in_hook = 0;
    return ret;
}

hipError_t hipHostMalloc(void **ptr, size_t size, unsigned int flags) {
    typedef hipError_t (*fn_t)(void**, size_t, unsigned int);
    static fn_t real = NULL;
    if (!real) real = (fn_t)_real_hip_sym("hipHostMalloc");
    if (!real) return -1;
    if (in_hook) return real(ptr, size, flags);
    in_hook = 1;

    char extra[64];
    snprintf(extra, sizeof(extra), "type=alloc_pinned,bytes=%zu", size);
    uint64_t t0 = now_ns();
    hipError_t ret = real(ptr, size, flags);
    if (ret == 0 && ptr && *ptr) pin_track_add(*ptr, size);
    emit_span("memory", gettid_compat(), t0, now_ns()-t0, "hipHostMalloc", extra);

    in_hook = 0;
    return ret;
}

hipError_t hipHostFree(void *ptr) {
    typedef hipError_t (*fn_t)(void*);
    static fn_t real = NULL;
    if (!real) real = (fn_t)_real_hip_sym("hipHostFree");
    if (!real) return -1;
    if (in_hook) return real(ptr);
    in_hook = 1;

    pin_track_rem(ptr);
    uint64_t t0 = now_ns();
    hipError_t ret = real(ptr);
    emit_span("memory", gettid_compat(), t0, now_ns()-t0,
              "hipHostFree", "type=free_pinned");

    in_hook = 0;
    return ret;
}

hipError_t hipDeviceSynchronize(void) {
    typedef hipError_t (*fn_t)(void);
    static fn_t real = NULL;
    if (!real) real = (fn_t)_real_hip_sym("hipDeviceSynchronize");
    if (!real) return -1;
    if (in_hook) return real();
    in_hook = 1;

    char tags[96];
    snprintf(tags, sizeof(tags), "type=sync,op=sync,sync=device");
    Sub u;
    sub_begin(&u, NULL, 0);
    hipError_t ret = real();
    sub_end(&u);
    pk_flush(NULL, 1);
    sub_host(&u, "sync", "hipDeviceSynchronize", tags, (int)ret);

    in_hook = 0;
    return ret;
}

hipError_t hipStreamSynchronize(hipStream_t stream) {
    typedef hipError_t (*fn_t)(hipStream_t);
    static fn_t real = NULL;
    if (!real) real = (fn_t)_real_hip_sym("hipStreamSynchronize");
    if (!real) return -1;
    if (in_hook) return real(stream);
    in_hook = 1;

    char tags[96];
    snprintf(tags, sizeof(tags), "type=sync,op=sync,sync=stream,stream=%d", get_stream_id(stream));
    Sub u;
    sub_begin(&u, NULL, 0);
    hipError_t ret = real(stream);
    sub_end(&u);
    pk_flush(stream, 0);
    sub_host(&u, "sync", "hipStreamSynchronize", tags, (int)ret);

    in_hook = 0;
    return ret;
}

hipError_t hipEventSynchronize(hipEvent_t event) {
    typedef hipError_t (*fn_t)(hipEvent_t);
    static fn_t real = NULL;
    if (!real) real = (fn_t)_real_hip_sym("hipEventSynchronize");
    if (!real) return -1;
    if (in_hook) return real(event);
    in_hook = 1;

    char tags[96];
    snprintf(tags, sizeof(tags), "type=sync,op=sync,sync=event,event=%d", handle_id(event));
    Sub u;
    sub_begin(&u, NULL, 0);
    hipError_t ret = real(event);
    sub_end(&u);
    pk_flush(NULL, 1);
    sub_host(&u, "sync", "hipEventSynchronize", tags, (int)ret);

    in_hook = 0;
    return ret;
}

hipError_t hipDeviceReset(void) {
    typedef hipError_t (*fn_t)(void);
    static fn_t real = NULL;
    if (!real) real = (fn_t)_real_hip_sym("hipDeviceReset");
    if (!real) return -1;
    if (in_hook) return real();
    in_hook = 1;

    pk_flush(NULL, 1);   /* device state is destroyed by the call */
    char tags[96];
    snprintf(tags, sizeof(tags), "type=sync,op=sync,sync=device");
    Sub u;
    sub_begin(&u, NULL, 0);
    hipError_t ret = real();
    sub_end(&u);
    sub_host(&u, "sync", "hipDeviceReset", tags, (int)ret);

    in_hook = 0;
    return ret;
}

/* Event record / stream wait: they define which stream an event sync or a
 * cross-stream wait refers to (the critical path resolves them through
 * these spans). */
hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream) {
    typedef hipError_t (*fn_t)(hipEvent_t, hipStream_t);
    static fn_t real = NULL;
    if (!real) real = (fn_t)_real_hip_sym("hipEventRecord");
    if (!real) return -1;
    if (in_hook) return real(event, stream);
    in_hook = 1;

    char tags[96];
    snprintf(tags, sizeof(tags), "type=event_record,op=event_record,event=%d,stream=%d",
             handle_id(event), get_stream_id(stream));
    Sub u;
    sub_begin(&u, NULL, 0);
    hipError_t ret = real(event, stream);
    sub_end(&u);
    sub_host(&u, "rocm", "hipEventRecord", tags, (int)ret);

    in_hook = 0;
    return ret;
}

hipError_t hipStreamWaitEvent(hipStream_t stream, hipEvent_t event, unsigned int flags) {
    typedef hipError_t (*fn_t)(hipStream_t, hipEvent_t, unsigned int);
    static fn_t real = NULL;
    if (!real) real = (fn_t)_real_hip_sym("hipStreamWaitEvent");
    if (!real) return -1;
    if (in_hook) return real(stream, event, flags);
    in_hook = 1;

    char tags[96];
    snprintf(tags, sizeof(tags), "type=stream_wait,op=stream_wait,event=%d,stream=%d",
             handle_id(event), get_stream_id(stream));
    Sub u;
    sub_begin(&u, NULL, 0);
    hipError_t ret = real(stream, event, flags);
    sub_end(&u);
    sub_host(&u, "rocm", "hipStreamWaitEvent", tags, (int)ret);

    in_hook = 0;
    return ret;
}

hipError_t hipMemsetAsync(void *dst, int value, size_t size, hipStream_t stream) {
    typedef hipError_t (*fn_t)(void*, int, size_t, hipStream_t);
    static fn_t real = NULL;
    if (!real) real = (fn_t)_real_hip_sym("hipMemsetAsync");
    if (!real) return -1;
    if (in_hook) return real(dst, value, size, stream);
    in_hook = 1;

    int sid = get_stream_id(stream);
    char host[96], dev[96];
    snprintf(host, sizeof(host), "type=launch,op=memset,bytes=%zu,stream=%d", size, sid);
    snprintf(dev, sizeof(dev), "type=memset,op=memset,bytes=%zu,stream=%d", size, sid);
    Sub u;
    sub_begin(&u, stream, 1);
    hipError_t ret = real(dst, value, size, stream);
    sub_end(&u);
    sub_host(&u, "memory", "hipMemsetAsync", host, (int)ret);
    if (ret == 0) sub_proxy(&u, stream, "memory", "memset", dev);
    else          sub_abort(&u);

    in_hook = 0;
    return ret;
}

hipError_t hipMemset(void *dst, int value, size_t size) {
    typedef hipError_t (*fn_t)(void*, int, size_t);
    static fn_t real = NULL;
    if (!real) real = (fn_t)_real_hip_sym("hipMemset");
    if (!real) return -1;
    if (in_hook) return real(dst, value, size);
    in_hook = 1;

    char host[96], dev[96];
    snprintf(host, sizeof(host), "type=launch,op=memset,bytes=%zu,stream=0", size);
    snprintf(dev, sizeof(dev), "type=memset,op=memset,bytes=%zu,stream=0", size);
    Sub u;
    sub_begin(&u, NULL, 1);
    hipError_t ret = real(dst, value, size);
    sub_end(&u);
    sub_host(&u, "memory", "hipMemset", host, (int)ret);
    if (ret == 0) sub_proxy(&u, NULL, "memory", "memset", dev);
    else          sub_abort(&u);

    in_hook = 0;
    return ret;
}

/* ── Native HIP AoT kernel registration ─────────────────────────────────── */
static void _save_rocm_bin(const void *image, char *out_path, size_t path_cap);

/*
 * __hipRegisterFatBinary is called once per translation unit during static
 * initialisation.  The `data` pointer points to the embedded clang offload
 * bundle (__CLANG_OFFLOAD_BUNDLE__ magic) containing the AMDGCN HSACOs.
 * Saving it lets the Python disassembler find the device code even for AoT
 * native HIP binaries where hipModuleLoadData is never called.
 */
void **__hipRegisterFatBinary(const void *data) {
    typedef void **(*fn_t)(const void *);
    static fn_t real = NULL;
    if (!real) {
        void *lib = g_hip_lib ? g_hip_lib :
                    dlopen("libamdhip64.so", RTLD_LAZY | RTLD_GLOBAL);
        if (lib) real = (fn_t)dlsym(lib, "__hipRegisterFatBinary");
    }
    void **ret = real ? real(data) : NULL;

    char saved_path[512];
    _save_rocm_bin(data, saved_path, sizeof(saved_path));
    if (saved_path[0]) {
        /* Emit a jit_compile span so runner.py finds the saved path and
         * passes it to collect_disasm via rocm_jit_paths.  The span is also
         * useful for the glob fallback in case the socket isn't yet open. */
        char extra[640];
        snprintf(extra, sizeof(extra), "type=jit_compile,path=%s", saved_path);
        emit_span("jit", gettid_compat(), now_ns(), 0,
                  "__hipRegisterFatBinary", extra);
    }
    return ret;
}

/*
 * __hipRegisterFunction is called by the HIP runtime during static
 * initialisation (before main()) to register each __global__ kernel.
 * Intercepting it lets us build the host-stub-pointer → kernel-name table
 * so that hipLaunchKernel() calls show the real kernel name instead of
 * "<jit-kernel>".
 */
void __hipRegisterFunction(
        void **modules,
        const void *hostFunction,
        char *deviceFunction,
        const char *deviceName,
        unsigned int threadLimit,
        void *tid, void *bid, void *blockDim, void *gridDim, int *wSize)
{
    typedef void (*fn_t)(void **, const void *, char *, const char *,
                         unsigned int, void *, void *, void *, void *, int *);
    static fn_t real = NULL;
    if (!real) {
        void *lib = g_hip_lib ? g_hip_lib :
                    dlopen("libamdhip64.so", RTLD_LAZY | RTLD_GLOBAL);
        if (lib) real = (fn_t)dlsym(lib, "__hipRegisterFunction");
    }
    if (real) real(modules, hostFunction, deviceFunction, deviceName,
                   threadLimit, tid, bid, blockDim, gridDim, wSize);

    const char *kname = (deviceName && deviceName[0]) ? deviceName : deviceFunction;
    if (!hostFunction || !kname || !kname[0]) return;
    pthread_mutex_lock(&g_kname_mutex);
    int found = 0;
    for (int i = 0; i < g_kname_n; i++) {
        if (g_knames[i].fn == (hipFunction_t)(uintptr_t)hostFunction) {
            found = 1; break;
        }
    }
    if (!found && g_kname_n < KNAME_MAP_CAP) {
        g_knames[g_kname_n].fn = (hipFunction_t)(uintptr_t)hostFunction;
        strncpy(g_knames[g_kname_n].name, kname, 2047);
        g_knames[g_kname_n].name[2047] = '\0';
        g_kname_n++;
    }
    pthread_mutex_unlock(&g_kname_mutex);
}

/* ── JIT module load + kernel name table ─────────────────────────────────── */

/* Save the AMDGCN ELF (or PTX text) passed to hipModuleLoad* so the Python
 * disassembler can find it after the process exits.  Returns the saved path
 * in out_path (capacity path_cap) or leaves it empty on failure. */
static int _rocm_bin_counter = 0;
static void _save_rocm_bin(const void *image, char *out_path, size_t path_cap) {
    out_path[0] = '\0';
    if (!image) return;
    const uint8_t *p = (const uint8_t *)image;
    size_t sz = 0;

    if (p[0] == 0x7f && p[1] == 'E' && p[2] == 'L' && p[3] == 'F') {
        int is64 = (p[4] == 2);
        if (is64) {
            /* 64-bit ELF: start with end of section header table, then scan
             * each section header to include section data that may follow. */
            uint64_t shoff; memcpy(&shoff, p + 40, 8);
            uint16_t shesz; memcpy(&shesz, p + 58, 2);
            uint16_t shnum; memcpy(&shnum, p + 60, 2);
            size_t end = (size_t)(shoff + (uint64_t)shesz * shnum);
            if (shoff > 0 && shesz >= 64 && shnum > 0 && shnum <= 32768
                    && shoff < 256ULL*1024*1024) {
                for (uint16_t i = 0; i < shnum; i++) {
                    const uint8_t *shdr = p + shoff + (size_t)i * shesz;
                    uint64_t sec_off, sec_sz;
                    memcpy(&sec_off, shdr + 24, 8);   /* sh_offset */
                    memcpy(&sec_sz,  shdr + 32, 8);   /* sh_size   */
                    size_t sec_end = (size_t)(sec_off + sec_sz);
                    if (sec_end > end && sec_end < 256ULL*1024*1024)
                        end = sec_end;
                }
            }
            if (end > 64 && end < 256ULL * 1024 * 1024) sz = end;
        }
    } else if (memcmp(p, "__CLANG_OFFLOAD_BUNDLE__", 24) == 0) {
        /* Clang offload bundle: parse entry headers to find total data extent. */
        uint64_t num_objs; memcpy(&num_objs, p + 24, 8);
        size_t max_end = 32;
        const uint8_t *hdr = p + 32;
        for (uint64_t i = 0; i < num_objs && i < 1024; i++) {
            uint64_t offset, bundle_sz, triple_sz;
            memcpy(&offset,    hdr,      8);
            memcpy(&bundle_sz, hdr + 8,  8);
            memcpy(&triple_sz, hdr + 16, 8);
            size_t sec_end = (size_t)(offset + bundle_sz);
            if (sec_end > max_end && sec_end < 256ULL*1024*1024)
                max_end = sec_end;
            hdr += 24 + (size_t)triple_sz;
        }
        if (max_end > 24 && max_end < 256ULL*1024*1024) sz = max_end;
    } else if ((p[0] == '/' && p[1] == '/') || p[0] == '.') {
        /* PTX text: include null terminator only if the string ends within
         * the limit (off-by-one fix: sz++ past mapped region at exact 64MB). */
        sz = strnlen((const char *)image, 64 * 1024 * 1024);
        if (sz > 0 && sz < 64 * 1024 * 1024) sz++;
    }
    if (sz == 0) return;

    int idx = __atomic_fetch_add(&_rocm_bin_counter, 1, __ATOMIC_RELAXED);
    snprintf(out_path, path_cap, "/tmp/hprofiler_rocm_%d_%d.bin",
             (int)getpid(), idx);
    FILE *f = fopen(out_path, "wb");
    if (f) {
        fwrite(image, 1, sz, f); fclose(f);
    } else { out_path[0] = '\0'; }
}

hipError_t hipModuleLoadData(hipModule_t *module, const void *image) {
    typedef hipError_t (*fn_t)(hipModule_t *, const void *);
    static fn_t real = NULL;
    if (!real) real = (fn_t)_real_hip_sym("hipModuleLoadData");
    if (!real) return -1;
    char saved_path[512];
    _save_rocm_bin(image, saved_path, sizeof(saved_path));
    uint64_t t0 = now_ns();
    hipError_t ret = real(module, image);
    char extra[640];
    if (saved_path[0])
        snprintf(extra, sizeof(extra), "type=jit_compile,path=%s", saved_path);
    else
        snprintf(extra, sizeof(extra), "type=jit_compile");
    emit_span("jit", gettid_compat(), t0, now_ns()-t0, "hipModuleLoadData", extra);
    return ret;
}

hipError_t hipModuleLoadDataEx(hipModule_t *module, const void *image,
                                unsigned int numOptions,
                                void *options, void *optionValues) {
    typedef hipError_t (*fn_t)(hipModule_t *, const void *, unsigned, void *, void *);
    static fn_t real = NULL;
    if (!real) real = (fn_t)_real_hip_sym("hipModuleLoadDataEx");
    if (!real) return -1;
    char saved_path[512];
    _save_rocm_bin(image, saved_path, sizeof(saved_path));
    uint64_t t0 = now_ns();
    hipError_t ret = real(module, image, numOptions, options, optionValues);
    char extra[640];
    if (saved_path[0])
        snprintf(extra, sizeof(extra), "type=jit_compile,path=%s", saved_path);
    else
        snprintf(extra, sizeof(extra), "type=jit_compile");
    emit_span("jit", gettid_compat(), t0, now_ns()-t0, "hipModuleLoadDataEx", extra);
    return ret;
}

hipError_t hipModuleGetFunction(hipFunction_t *hfunc, hipModule_t hmod,
                                 const char *name) {
    typedef hipError_t (*fn_t)(hipFunction_t *, hipModule_t, const char *);
    static fn_t real = NULL;
    if (!real) real = (fn_t)_real_hip_sym("hipModuleGetFunction");
    if (!real) return -1;
    hipError_t ret = real(hfunc, hmod, name);
    if (ret == 0 && hfunc && *hfunc && name) {
        pthread_mutex_lock(&g_kname_mutex);
        if (g_kname_n < KNAME_MAP_CAP) {
            g_knames[g_kname_n].fn = *hfunc;
            strncpy(g_knames[g_kname_n].name, name, 2047);
            g_knames[g_kname_n].name[2047] = '\0';
            g_kname_n++;
        }
        pthread_mutex_unlock(&g_kname_mutex);
    }
    return ret;
}

hipError_t hipModuleUnload(hipModule_t hmod) {
    typedef hipError_t (*fn_t)(hipModule_t);
    static fn_t real = NULL;
    if (!real) real = (fn_t)_real_hip_sym("hipModuleUnload");
    if (!real) return -1;
    hipError_t ret = real(hmod);
    /* g_knames doesn't track which module each hipFunction_t came from, so
     * we can't selectively invalidate just this module's entries -- and
     * the driver is free to reuse a freed hipFunction_t address for an
     * unrelated kernel in a later-loaded module (relevant for JIT-heavy
     * use, e.g. ACPP/hipSYCL targeting ROCm), which would otherwise make
     * resolve_name() return the OLD kernel's name forever. Clearing the
     * whole table is conservative but correct; still-loaded modules are
     * cheaply repopulated by their next hipModuleGetFunction call. */
    if (ret == 0) {
        pthread_mutex_lock(&g_kname_mutex);
        g_kname_n = 0;
        pthread_mutex_unlock(&g_kname_mutex);
    }
    return ret;
}

/* ── HIP Graph launch ────────────────────────────────────────────────────── */
typedef void *hipGraphExec_t;

hipError_t hipGraphLaunch(hipGraphExec_t graphExec, hipStream_t stream) {
    typedef hipError_t (*fn_t)(hipGraphExec_t, hipStream_t);
    static fn_t real = NULL;
    if (!real) real = (fn_t)_real_hip_sym("hipGraphLaunch");
    if (!real) return -1;
    if (in_hook) return real(graphExec, stream);
    in_hook = 1;

    int sid = get_stream_id(stream);
    char host[64], dev[64];
    snprintf(host, sizeof(host), "type=launch,op=graph,stream=%d", sid);
    snprintf(dev, sizeof(dev), "type=graph_launch,op=graph,stream=%d", sid);

    Sub u;
    sub_begin(&u, stream, 1);
    hipError_t ret = real(graphExec, stream);
    sub_end(&u);
    sub_host(&u, "rocm", "hipGraphLaunch", host, (int)ret);
    if (ret == 0) sub_proxy(&u, stream, "rocm", "graph", dev);
    else          sub_abort(&u);

    in_hook = 0;
    return ret;
}

/* ── ROCTx annotation interception ──────────────────────────────────────── */
typedef int64_t roctx_range_id_t;

#define ROCTX_STACK_DEPTH 64
static __thread uint64_t roctx_stack_ts[ROCTX_STACK_DEPTH];
static __thread char     roctx_stack_nm[ROCTX_STACK_DEPTH][256];
static __thread int      roctx_depth = 0;

roctx_range_id_t roctxRangePushA(const char *message) {
    typedef roctx_range_id_t (*fn_t)(const char *);
    static fn_t real = NULL;
    if (!real) real = (fn_t)_real_hip_sym("roctxRangePushA");
    roctx_range_id_t id = real ? real(message) : (roctx_range_id_t)roctx_depth;
    if (roctx_depth < ROCTX_STACK_DEPTH) {
        roctx_stack_ts[roctx_depth] = now_ns();
        snprintf(roctx_stack_nm[roctx_depth], 256, "%s", message ? message : "");
        roctx_depth++;
    }
    return id;
}

roctx_range_id_t roctxRangePop(void) {
    typedef roctx_range_id_t (*fn_t)(void);
    static fn_t real = NULL;
    if (!real) real = (fn_t)_real_hip_sym("roctxRangePop");
    if (roctx_depth > 0) {
        roctx_depth--;
        emit_span("annotation", gettid_compat(),
                  roctx_stack_ts[roctx_depth],
                  now_ns() - roctx_stack_ts[roctx_depth],
                  roctx_stack_nm[roctx_depth], "type=roctx_range");
    }
    return real ? real() : 0;
}

void roctxMarkA(const char *message) {
    typedef void (*fn_t)(const char *);
    static fn_t real = NULL;
    if (!real) real = (fn_t)_real_hip_sym("roctxMarkA");
    emit_span("annotation", gettid_compat(), now_ns(), 0,
              message ? message : "roctx_mark", "type=roctx_mark");
    if (real) real(message);
}

/* ── Constructor / Destructor ───────────────────────────────────────────── */
__attribute__((constructor)) static void init(void) {
    /* Pre-load libamdhip64 with RTLD_GLOBAL so dlsym(RTLD_NEXT, ...) finds HIP
     * symbols even if AdaptiveCpp SSCP later loads the same library RTLD_LOCAL. */
    if (!g_hip_lib)
        g_hip_lib = dlopen("libamdhip64.so",   RTLD_LAZY | RTLD_GLOBAL);
    if (!g_hip_lib)
        g_hip_lib = dlopen("libamdhip64.so.5", RTLD_LAZY | RTLD_GLOBAL);
    ensure_connected();
    cs_init();
}
__attribute__((destructor))  static void fini(void) {
    pk_flush(NULL, 1);
    /* Detect unreleased GPU allocations (R4: memory leak detection) */
    int64_t leaked = 0;
    pthread_mutex_lock(&g_alloc_mutex);
    for (int i = 0; i < g_alloc_n; i++) leaked += (int64_t)g_allocs[i].sz;
    pthread_mutex_unlock(&g_alloc_mutex);
    if (leaked > 0)
        emit_ctr("memory", "gpu_memory_leaked_bytes", leaked, "bytes");
    if (g_sock >= 0) { close(g_sock); g_sock = -1; }
    if (g_hip_lib)   { dlclose(g_hip_lib); g_hip_lib = NULL; }
}
