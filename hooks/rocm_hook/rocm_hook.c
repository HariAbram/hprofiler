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

/* Thread-local recursion guard */
static __thread int in_hook = 0;

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
    /* Fast path: already in global namespace via RTLD_NEXT */
    void *sym = _real_hip_sym(name);
    if (sym) return sym;
    /* Slow path: libamdhip64 loaded with RTLD_LOCAL — try our explicit handle */
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
    if (!path) return;
    int s = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (s < 0) return;
    struct sockaddr_un addr = {0};
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, path, sizeof(addr.sun_path) - 1);
    if (connect(s, (struct sockaddr*)&addr, sizeof(addr)) == 0) {
        g_sock = s; g_pid = getpid();
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
typedef struct { hipFunction_t fn; char name[256]; } KNameEntry;
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
#define STREAM_MAP_CAP 256
static void           *g_stream_ptrs[STREAM_MAP_CAP];
static int             g_stream_ids[STREAM_MAP_CAP];
static int             g_stream_count = 0;
static pthread_mutex_t g_stream_mutex = PTHREAD_MUTEX_INITIALIZER;

static int get_stream_id(const void *stream) {
    if (!stream) return 0;
    pthread_mutex_lock(&g_stream_mutex);
    for (int i = 0; i < g_stream_count; i++) {
        if (g_stream_ptrs[i] == stream) {
            int id = g_stream_ids[i];
            pthread_mutex_unlock(&g_stream_mutex);
            return id;
        }
    }
    int id = (g_stream_count < STREAM_MAP_CAP) ? (g_stream_count + 1) : -1;
    if (g_stream_count < STREAM_MAP_CAP) {
        g_stream_ptrs[g_stream_count] = (void*)stream;
        g_stream_ids[g_stream_count]  = id;
        g_stream_count++;
    }
    pthread_mutex_unlock(&g_stream_mutex);
    return id;
}

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

typedef struct {
    hipEvent_t   ev_start;
    hipEvent_t   ev_end;
    hipStream_t  stream;
    char         cat[32];
    char         kname[256];
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
        char cat[32], kname[256], extra[256];
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
            strncpy(l->kname, pk->kname, 255); l->kname[255] = '\0';
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
        f_evDestroy(l->ev_s);
        f_evDestroy(l->ev_e);
        if (ok)
            emit_span(l->cat, l->tid, l->t0, (uint64_t)(ms * 1e6f),
                      l->kname, l->extra);
    }
}

static int pk_try_begin(hipStream_t stream, hipEvent_t *ev_s, hipEvent_t *ev_e) {
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
        strncpy(pk->cat,   cat,   31);  pk->cat[31]   = '\0';
        strncpy(pk->kname, kname, 255); pk->kname[255] = '\0';
        strncpy(pk->extra, extra, 255); pk->extra[255] = '\0';
        pthread_mutex_unlock(&g_pk_mutex);
    } else {
        pthread_mutex_unlock(&g_pk_mutex);
        f_evDestroy(ev_s); f_evDestroy(ev_e);
        emit_span(cat, tid, t0, now_ns() - t0, kname, extra);
    }
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
    pid_t tid = gettid_compat();
    int sid = get_stream_id(stream);
    char extra[256];
    snprintf(extra, sizeof(extra),
             "type=kernel,grid=%dx%dx%d,block=%dx%dx%d,stream=%d",
             grid.x,grid.y,grid.z, block.x,block.y,block.z, sid);

    hipEvent_t ev_s, ev_e;
    int gpu_ok = pk_try_begin(stream, &ev_s, &ev_e);
    uint64_t t0 = now_ns();
    hipError_t ret = real(fn, grid, block, args, sharedMem, stream);
    if (gpu_ok) pk_commit(ev_s, ev_e, stream, "rocm", kname, extra, t0, tid);
    else        emit_span("rocm", tid, t0, now_ns() - t0, kname, extra);

    in_hook = 0;
    return ret;
}

hipError_t hipLaunchKernelGGL(hipFunction_t fn,
                               dim3 grid, dim3 block,
                               unsigned int sharedMem, hipStream_t stream,
                               void **kernelParams) {
    typedef hipError_t (*fn_t)(hipFunction_t, dim3, dim3, unsigned int,
                                hipStream_t, void**);
    static fn_t real = NULL;
    if (!real) real = (fn_t)_real_hip_sym("hipLaunchKernelGGL");
    if (!real) return -1;
    if (in_hook) return real(fn, grid, block, sharedMem, stream, kernelParams);
    in_hook = 1;

    const char *kname = resolve_name(fn);
    pid_t tid = gettid_compat();
    int sid = get_stream_id(stream);
    char extra[256];
    snprintf(extra, sizeof(extra),
             "type=kernel,grid=%dx%dx%d,block=%dx%dx%d,stream=%d",
             grid.x,grid.y,grid.z, block.x,block.y,block.z, sid);

    hipEvent_t ev_s, ev_e;
    int gpu_ok = pk_try_begin(stream, &ev_s, &ev_e);
    uint64_t t0 = now_ns();
    hipError_t ret = real(fn, grid, block, sharedMem, stream, kernelParams);
    if (gpu_ok) pk_commit(ev_s, ev_e, stream, "rocm", kname, extra, t0, tid);
    else        emit_span("rocm", tid, t0, now_ns() - t0, kname, extra);

    in_hook = 0;
    return ret;
}

hipError_t hipModuleLaunchKernel(hipFunction_t f,
                                  unsigned int gx, unsigned int gy, unsigned int gz,
                                  unsigned int bx, unsigned int by, unsigned int bz,
                                  unsigned int sharedMem, hipStream_t stream,
                                  void **kernelParams, void **extra_params) {
    typedef hipError_t (*fn_t)(hipFunction_t,
                                unsigned, unsigned, unsigned,
                                unsigned, unsigned, unsigned,
                                unsigned, hipStream_t, void**, void**);
    static fn_t real = NULL;
    if (!real) real = (fn_t)_real_hip_sym("hipModuleLaunchKernel");
    if (!real) return -1;
    if (in_hook) return real(f, gx,gy,gz, bx,by,bz, sharedMem, stream,
                             kernelParams, extra_params);
    in_hook = 1;

    const char *kname = resolve_name(f);
    pid_t tid = gettid_compat();
    int sid = get_stream_id(stream);
    char extra[256];
    snprintf(extra, sizeof(extra),
             "type=kernel,grid=%ux%ux%u,block=%ux%ux%u,stream=%d",
             gx,gy,gz, bx,by,bz, sid);

    hipEvent_t ev_s, ev_e;
    int gpu_ok = pk_try_begin(stream, &ev_s, &ev_e);
    uint64_t t0 = now_ns();
    hipError_t ret = real(f, gx,gy,gz, bx,by,bz, sharedMem, stream,
                          kernelParams, extra_params);
    if (gpu_ok) pk_commit(ev_s, ev_e, stream, "rocm", kname, extra, t0, tid);
    else        emit_span("rocm", tid, t0, now_ns() - t0, kname, extra);

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

    char extra[64];
    snprintf(extra, sizeof(extra), "type=memcpy,bytes=%zu", size);
    uint64_t t0 = now_ns();
    hipError_t ret = real(dst, src, size, kind);
    emit_span("rocm", gettid_compat(), t0, now_ns()-t0, "hipMemcpy", extra);

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

    pid_t tid = gettid_compat();
    int sid = get_stream_id(stream);
    char extra[128];
    snprintf(extra, sizeof(extra), "type=memcpy_async,bytes=%zu,stream=%d", size, sid);

    hipEvent_t ev_s, ev_e;
    int gpu_ok = pk_try_begin(stream, &ev_s, &ev_e);
    uint64_t t0 = now_ns();
    hipError_t ret = real(dst, src, size, kind, stream);
    if (gpu_ok) pk_commit(ev_s, ev_e, stream, "memory", "hipMemcpyAsync", extra, t0, tid);
    else        emit_span("memory", tid, t0, now_ns()-t0, "hipMemcpyAsync", extra);

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

    char extra[64];
    snprintf(extra, sizeof(extra), "type=HtoD,bytes=%zu", size);
    uint64_t t0 = now_ns();
    hipError_t ret = real(dst, src, size);
    emit_span("memory", gettid_compat(), t0, now_ns()-t0, "hipMemcpyHtoD", extra);

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

    char extra[64];
    snprintf(extra, sizeof(extra), "type=DtoH,bytes=%zu", size);
    uint64_t t0 = now_ns();
    hipError_t ret = real(dst, src, size);
    emit_span("memory", gettid_compat(), t0, now_ns()-t0, "hipMemcpyDtoH", extra);

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

    uint64_t t0 = now_ns();
    hipError_t ret = real();
    pk_flush(NULL, 1);
    emit_span("sync", gettid_compat(), t0, now_ns()-t0,
              "hipDeviceSynchronize", "type=sync");

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

    uint64_t t0 = now_ns();
    hipError_t ret = real(stream);
    pk_flush(stream, 0);
    char extra[64];
    snprintf(extra, sizeof(extra), "type=sync,stream=%d", get_stream_id(stream));
    emit_span("sync", gettid_compat(), t0, now_ns()-t0,
              "hipStreamSynchronize", extra);

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

    uint64_t t0 = now_ns();
    hipError_t ret = real(event);
    pk_flush(NULL, 1);
    emit_span("sync", gettid_compat(), t0, now_ns()-t0,
              "hipEventSynchronize", "type=sync");

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

    pk_flush(NULL, 1);
    uint64_t t0 = now_ns();
    hipError_t ret = real();
    emit_span("sync", gettid_compat(), t0, now_ns()-t0,
              "hipDeviceReset", "type=sync");

    in_hook = 0;
    return ret;
}

/* ── JIT binary capture + kernel name table ──────────────────────────────── */
static void _save_rocm_image(const void *image) {
    if (!image) return;
    static int counter = 0;
    char path[256];
    snprintf(path, sizeof(path), "/tmp/hprofiler_rocm_%d_%d.bin",
             (int)getpid(), counter++);
    const uint8_t *p = (const uint8_t *)image;
    size_t sz = 0;
    if (p[0]==0x7f && p[1]=='E' && p[2]=='L' && p[3]=='F' && p[4]==2) {
        uint64_t shoff; memcpy(&shoff, p+40, 8);
        uint16_t shesz; memcpy(&shesz, p+58, 2);
        uint16_t shnum; memcpy(&shnum, p+60, 2);
        size_t end = (size_t)(shoff + (uint64_t)shesz * shnum);
        if (end > 64 && end < 512ULL*1024*1024) sz = end;
    } else if ((p[0]=='/' && p[1]=='/') || p[0]=='.' || p[0]==';') {
        sz = strnlen((const char *)image, 64*1024*1024);
        if (sz > 0) sz++;
    }
    if (sz == 0) return;
    FILE *f = fopen(path, "wb");
    if (f) { fwrite(image, 1, sz, f); fclose(f); }
}

hipError_t hipModuleLoadData(hipModule_t *module, const void *image) {
    typedef hipError_t (*fn_t)(hipModule_t *, const void *);
    static fn_t real = NULL;
    if (!real) real = (fn_t)_real_hip_sym("hipModuleLoadData");
    if (!real) return -1;
    _save_rocm_image(image);
    uint64_t t0 = now_ns();
    hipError_t ret = real(module, image);
    emit_span("jit", gettid_compat(), t0, now_ns()-t0,
              "hipModuleLoadData", "type=jit_compile");
    return ret;
}

hipError_t hipModuleLoadDataEx(hipModule_t *module, const void *image,
                                unsigned int numOptions,
                                void *options, void *optionValues) {
    typedef hipError_t (*fn_t)(hipModule_t *, const void *, unsigned, void *, void *);
    static fn_t real = NULL;
    if (!real) real = (fn_t)_real_hip_sym("hipModuleLoadDataEx");
    if (!real) return -1;
    _save_rocm_image(image);
    uint64_t t0 = now_ns();
    hipError_t ret = real(module, image, numOptions, options, optionValues);
    emit_span("jit", gettid_compat(), t0, now_ns()-t0,
              "hipModuleLoadDataEx", "type=jit_compile");
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
            strncpy(g_knames[g_kname_n].name, name, 255);
            g_knames[g_kname_n].name[255] = '\0';
            g_kname_n++;
        }
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

    pid_t tid = gettid_compat();
    int sid = get_stream_id(stream);
    char extra[64];
    snprintf(extra, sizeof(extra), "type=graph_launch,stream=%d", sid);

    hipEvent_t ev_s, ev_e;
    int gpu_ok = pk_try_begin(stream, &ev_s, &ev_e);
    uint64_t t0 = now_ns();
    hipError_t ret = real(graphExec, stream);
    if (gpu_ok) pk_commit(ev_s, ev_e, stream, "rocm", "hipGraphLaunch", extra, t0, tid);
    else        emit_span("rocm", tid, t0, now_ns()-t0, "hipGraphLaunch", extra);

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
