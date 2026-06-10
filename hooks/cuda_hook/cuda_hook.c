/*
 * CUDA Runtime + Driver API + NVTX LD_PRELOAD hook.
 *
 * Changes over initial version:
 *   1. Thread-local recursion guard (in_hook) — prevents profiling our own
 *      profiling calls (e.g. cudaEventCreate/Record inside the hook).
 *   2. GPU-accurate async memcpy timing — cudaMemcpyAsync and driver async
 *      copies now use event pairs instead of CPU-only timing.
 *   3. More sync hooks — cudaEventSynchronize and cudaDeviceReset now flush
 *      pending GPU spans. cudaMemcpy (sync) flushes before the real call.
 *   4. cuModuleGetFunction — maps CUfunction handles to kernel names so
 *      cuLaunchKernel shows real names instead of <jit-kernel>.
 *   5. More memory hooks — cudaMallocAsync/FreeAsync, cudaHostAlloc,
 *      cudaMallocHost/FreeHost, cuMemAllocManaged, cuMemAllocAsync/FreeAsync.
 *      Pinned host memory tracked separately as pinned_memory_bytes.
 *
 * Wire protocol (newline-delimited ASCII):
 *   span:<cat>:<pid>:<tid>:<start_ns>:<dur_ns>:<name>[:<tag=val,...>]
 *   ctr:<cat>:<pid>:<ts_ns>:<name>:<value>:<unit>
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
#include <wchar.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/syscall.h>

/* ── CUDA / driver type stubs (no cuda.h required) ──────────────────────── */
typedef int              cudaError_t;
typedef void*            cudaStream_t;
typedef void*            cudaEvent_t;
typedef unsigned int     cudaMemcpyKind;
typedef void*            CUfunction;
typedef void*            CUstream;
typedef void*            CUdeviceptr;
typedef unsigned int     CUresult;
typedef void*            CUmodule_t;

typedef struct { unsigned int x, y, z; } dim3_t;

/* ── Globals ────────────────────────────────────────────────────────────── */
static int             g_sock        = -1;
static pthread_mutex_t g_sock_mutex  = PTHREAD_MUTEX_INITIALIZER;
static pid_t           g_pid         = 0;
static void           *g_cudart_handle = NULL;

/* ── dlsym/dlopen intercept for static-cudart support ──────────────────── *
 * When a binary links the CUDA runtime statically (libcudart_static.a),    *
 * the cudaXxx symbols are resolved at compile time and LD_PRELOAD wrappers *
 * are never called.  However, the static runtime still calls               *
 * dlopen("libcuda.so.1") + dlsym(handle, "cuLaunchKernel") to reach the   *
 * driver API.  By intercepting dlopen and dlsym we redirect those lookups  *
 * to our wrappers, giving us full profiling coverage.                       */
static void *(*g_real_dlsym)(void *, const char *) = NULL;
#define CUDA_DRV_HANDLE_CAP 8
static void           *g_cuda_drv_handles[CUDA_DRV_HANDLE_CAP];
static int             g_cuda_drv_n = 0;
static pthread_mutex_t g_cuda_drv_mutex = PTHREAD_MUTEX_INITIALIZER;

/* Thread-local recursion guard: prevents profiling our own CUDA event calls */
static __thread int in_hook = 0;

/* Span ID of the innermost active NVTX range on this thread (0 = none).
 * Set/cleared by nvtxRangePushA / nvtxRangePop.
 * Read by cudaLaunchKernel / cudaMemcpyAsync to stamp child spans. */
static __thread uint64_t tls_nvtx_span_id = 0;

/* ── Core helpers ───────────────────────────────────────────────────────── */
static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

static pid_t gettid_compat(void) {
    return (pid_t)syscall(SYS_gettid);
}

static void *find_cuda_sym(const char *name) {
    /* Use g_real_dlsym directly to bypass our own dlsym override. */
    void *(*real)(void *, const char *) = g_real_dlsym ? g_real_dlsym
        : (void*(*)(void*,const char*))dlsym;   /* fallback before init */
    void *sym = real(RTLD_NEXT, name);
    if (sym) return sym;
    sym = real(RTLD_DEFAULT, name);
    if (sym) return sym;
    if (g_cudart_handle) return real(g_cudart_handle, name);
    return NULL;
}

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
    }
}

/* send_all: write all n bytes; close socket on EPIPE/error so next call
 * reconnects rather than spinning on a broken socket. Caller holds mutex. */
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
        char buf[2048];   /* 2 KB: enough for long kernel names + tags */
        int n;
        if (extra && *extra)
            n = snprintf(buf, sizeof(buf),
                         "span:%s:%d:%d:%llu:%llu:%s:%s\n",
                         cat, g_pid, (int)tid,
                         (unsigned long long)start_ns,
                         (unsigned long long)dur_ns, name, extra);
        else
            n = snprintf(buf, sizeof(buf),
                         "span:%s:%d:%d:%llu:%llu:%s\n",
                         cat, g_pid, (int)tid,
                         (unsigned long long)start_ns,
                         (unsigned long long)dur_ns, name);
        /* snprintf returns >= sizeof(buf) when truncated; drop truncated spans */
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

/* ── Kernel name table (cuModuleGetFunction → name) ─────────────────────── */
#define KNAME_MAP_CAP 512
typedef struct { CUfunction fn; char name[256]; } KNameEntry;
static KNameEntry      g_knames[KNAME_MAP_CAP];
static int             g_kname_n = 0;
static pthread_mutex_t g_kname_mutex = PTHREAD_MUTEX_INITIALIZER;

static const char *resolve_kernel_name(const void *fn) {
    if (!fn) return "<unknown>";
    /* Check cuModuleGetFunction table first — works for JIT/driver-API kernels */
    pthread_mutex_lock(&g_kname_mutex);
    for (int i = 0; i < g_kname_n; i++) {
        if (g_knames[i].fn == (CUfunction)fn) {
            const char *p = g_knames[i].name;   /* stable pointer; array never shrinks */
            pthread_mutex_unlock(&g_kname_mutex);
            return p;
        }
    }
    pthread_mutex_unlock(&g_kname_mutex);
    Dl_info info;
    if (dladdr(fn, &info) && info.dli_sname)
        return info.dli_sname;
    return "<jit-kernel>";
}

/* ── GPU device memory tracking (dynamic growing table) ─────────────────── */
typedef struct { uintptr_t ptr; size_t sz; } AllocRec;

static AllocRec       *g_allocs   = NULL;
static int             g_alloc_n  = 0;
static int             g_alloc_cap = 0;
static int64_t         g_gpu_mem  = 0;
static pthread_mutex_t g_alloc_mutex = PTHREAD_MUTEX_INITIALIZER;

static void _alloc_grow(void) {
    int new_cap = g_alloc_cap ? g_alloc_cap * 2 : 4096;
    AllocRec *p = (AllocRec *)realloc(g_allocs, (size_t)new_cap * sizeof(AllocRec));
    if (p) { g_allocs = p; g_alloc_cap = new_cap; }
    /* If realloc fails, g_allocs/g_alloc_cap are unchanged; the caller's
     * bounds check (g_alloc_n < g_alloc_cap) prevents a write to NULL. */
}

static void mem_track_add(void *ptr, size_t sz) {
    int64_t total;
    pthread_mutex_lock(&g_alloc_mutex);
    if (g_alloc_n >= g_alloc_cap) _alloc_grow();
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

/* ── Pinned host memory tracking (cudaHostAlloc / cudaMallocHost) ─────────── */
#define ALLOC_CAP 4096   /* kept for pin table — pinned allocs are few */
static AllocRec        g_pin_allocs[ALLOC_CAP];
static int             g_pin_n    = 0;
static int64_t         g_pin_mem  = 0;
static pthread_mutex_t g_pin_mutex = PTHREAD_MUTEX_INITIALIZER;

static void pin_track_add(void *ptr, size_t sz) {
    int64_t total;
    pthread_mutex_lock(&g_pin_mutex);
    if (g_pin_n < ALLOC_CAP) {
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

/* ── GPU-accurate timing via cudaEvent pairs ──────────────────────────────── */
#define MAX_PENDING 512

typedef cudaError_t (*fn_EvCreate_t) (cudaEvent_t *);
typedef cudaError_t (*fn_EvRecord_t) (cudaEvent_t, cudaStream_t);
typedef cudaError_t (*fn_EvElapsed_t)(float *, cudaEvent_t, cudaEvent_t);
typedef cudaError_t (*fn_EvDestroy_t)(cudaEvent_t);
typedef cudaError_t (*fn_EvSync_t)   (cudaEvent_t);

static fn_EvCreate_t  f_evCreate  = NULL;
static fn_EvRecord_t  f_evRecord  = NULL;
static fn_EvElapsed_t f_evElapsed = NULL;
static fn_EvDestroy_t f_evDestroy = NULL;
static fn_EvSync_t    f_evSync    = NULL;

static int ev_api_ok(void) {
    if (!f_evCreate) {
        f_evCreate  = (fn_EvCreate_t) find_cuda_sym("cudaEventCreate");
        f_evRecord  = (fn_EvRecord_t) find_cuda_sym("cudaEventRecord");
        f_evElapsed = (fn_EvElapsed_t)find_cuda_sym("cudaEventElapsedTime");
        f_evDestroy = (fn_EvDestroy_t)find_cuda_sym("cudaEventDestroy");
        f_evSync    = (fn_EvSync_t)   find_cuda_sym("cudaEventSynchronize");
    }
    return f_evCreate && f_evRecord && f_evElapsed && f_evDestroy && f_evSync;
}

typedef struct {
    cudaEvent_t  ev_start;
    cudaEvent_t  ev_end;
    cudaStream_t stream;
    char         cat[32];    /* span category: "cuda" for kernels, "memory" for copies */
    char         kname[256];
    char         extra[256];
    uint64_t     cpu_start_ns;
    pid_t        tid;
} PendingKernel;

static PendingKernel   g_pk[MAX_PENDING];
static int             g_pk_n = 0;
static pthread_mutex_t g_pk_mutex = PTHREAD_MUTEX_INITIALIZER;

static void pk_flush(cudaStream_t flush_stream, int all_streams) {
    if (!ev_api_ok()) return;

    typedef struct {
        cudaEvent_t ev_s, ev_e;
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

static int pk_try_begin(cudaStream_t stream,
                        cudaEvent_t *ev_s, cudaEvent_t *ev_e) {
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

static void pk_commit(cudaEvent_t ev_s, cudaEvent_t ev_e,
                      cudaStream_t stream,
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

/* ── CUDA Runtime API wrappers ──────────────────────────────────────────── */

cudaError_t cudaLaunchKernel(
    const void *func, dim3_t gridDim, dim3_t blockDim,
    void **args, size_t sharedMem, cudaStream_t stream)
{
    typedef cudaError_t (*fn_t)(const void*, dim3_t, dim3_t, void**, size_t, cudaStream_t);
    static fn_t real = NULL;
    if (!real) real = (fn_t)find_cuda_sym("cudaLaunchKernel");
    if (!real) return -1;
    if (in_hook) return real(func, gridDim, blockDim, args, sharedMem, stream);
    in_hook = 1;

    const char *kname = resolve_kernel_name(func);
    pid_t tid = gettid_compat();
    int sid = get_stream_id(stream);
    char extra[256];
    if (tls_nvtx_span_id)
        snprintf(extra, sizeof(extra),
                 "type=kernel,grid=%ux%ux%u,block=%ux%ux%u,stream=%d,psid=%llu",
                 gridDim.x, gridDim.y, gridDim.z,
                 blockDim.x, blockDim.y, blockDim.z, sid,
                 (unsigned long long)tls_nvtx_span_id);
    else
        snprintf(extra, sizeof(extra),
                 "type=kernel,grid=%ux%ux%u,block=%ux%ux%u,stream=%d",
                 gridDim.x, gridDim.y, gridDim.z,
                 blockDim.x, blockDim.y, blockDim.z, sid);

    cudaEvent_t ev_s, ev_e;
    int gpu_ok = pk_try_begin(stream, &ev_s, &ev_e);
    uint64_t t0 = now_ns();
    cudaError_t ret = real(func, gridDim, blockDim, args, sharedMem, stream);
    if (gpu_ok)
        pk_commit(ev_s, ev_e, stream, "cuda", kname, extra, t0, tid);
    else
        emit_span("cuda", tid, t0, now_ns() - t0, kname, extra);

    in_hook = 0;
    return ret;
}

cudaError_t cudaMemcpy(void *dst, const void *src, size_t count,
                       cudaMemcpyKind kind) {
    typedef cudaError_t (*fn_t)(void*, const void*, size_t, cudaMemcpyKind);
    static fn_t real = NULL;
    if (!real) real = (fn_t)find_cuda_sym("cudaMemcpy");
    if (!real) return -1;
    if (in_hook) return real(dst, src, count, kind);
    in_hook = 1;

    /* Synchronous memcpy implies all prior GPU work is complete — flush pending */
    pk_flush(NULL, 1);

    const char *names[] = {"HtoH","HtoD","DtoH","DtoD","Default"};
    const char *dir = (kind <= 4) ? names[kind] : "Unknown";
    char extra[64];
    snprintf(extra, sizeof(extra), "type=memcpy,dir=%s,bytes=%zu", dir, count);
    uint64_t t0 = now_ns();
    cudaError_t ret = real(dst, src, count, kind);
    emit_span("cuda", gettid_compat(), t0, now_ns() - t0, "cudaMemcpy", extra);

    in_hook = 0;
    return ret;
}

cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count,
                             cudaMemcpyKind kind, cudaStream_t stream) {
    typedef cudaError_t (*fn_t)(void*, const void*, size_t, cudaMemcpyKind, cudaStream_t);
    static fn_t real = NULL;
    if (!real) real = (fn_t)find_cuda_sym("cudaMemcpyAsync");
    if (!real) return -1;
    if (in_hook) return real(dst, src, count, kind, stream);
    in_hook = 1;

    pid_t tid = gettid_compat();
    int sid = get_stream_id(stream);
    char extra[192];
    if (tls_nvtx_span_id)
        snprintf(extra, sizeof(extra), "type=memcpy_async,bytes=%zu,stream=%d,psid=%llu",
                 count, sid, (unsigned long long)tls_nvtx_span_id);
    else
        snprintf(extra, sizeof(extra), "type=memcpy_async,bytes=%zu,stream=%d", count, sid);

    cudaEvent_t ev_s, ev_e;
    int gpu_ok = pk_try_begin(stream, &ev_s, &ev_e);
    uint64_t t0 = now_ns();
    cudaError_t ret = real(dst, src, count, kind, stream);
    if (gpu_ok)
        pk_commit(ev_s, ev_e, stream, "memory", "cudaMemcpyAsync", extra, t0, tid);
    else
        emit_span("memory", tid, t0, now_ns() - t0, "cudaMemcpyAsync", extra);

    in_hook = 0;
    return ret;
}

cudaError_t cudaMalloc(void **devPtr, size_t size) {
    typedef cudaError_t (*fn_t)(void**, size_t);
    static fn_t real = NULL;
    if (!real) real = (fn_t)find_cuda_sym("cudaMalloc");
    if (!real) return -1;
    if (in_hook) return real(devPtr, size);
    in_hook = 1;

    char extra[64];
    snprintf(extra, sizeof(extra), "type=alloc,bytes=%zu", size);
    uint64_t t0 = now_ns();
    cudaError_t ret = real(devPtr, size);
    if (ret == 0 && devPtr && *devPtr)
        mem_track_add(*devPtr, size);
    emit_span("memory", gettid_compat(), t0, now_ns() - t0, "cudaMalloc", extra);

    in_hook = 0;
    return ret;
}

cudaError_t cudaMallocManaged(void **devPtr, size_t size, unsigned int flags) {
    typedef cudaError_t (*fn_t)(void**, size_t, unsigned int);
    static fn_t real = NULL;
    if (!real) real = (fn_t)find_cuda_sym("cudaMallocManaged");
    if (!real) return -1;
    if (in_hook) return real(devPtr, size, flags);
    in_hook = 1;

    char extra[64];
    snprintf(extra, sizeof(extra), "type=alloc_managed,bytes=%zu", size);
    uint64_t t0 = now_ns();
    cudaError_t ret = real(devPtr, size, flags);
    if (ret == 0 && devPtr && *devPtr)
        mem_track_add(*devPtr, size);
    emit_span("memory", gettid_compat(), t0, now_ns() - t0, "cudaMallocManaged", extra);

    in_hook = 0;
    return ret;
}

cudaError_t cudaMallocAsync(void **devPtr, size_t size, cudaStream_t stream) {
    typedef cudaError_t (*fn_t)(void**, size_t, cudaStream_t);
    static fn_t real = NULL;
    if (!real) real = (fn_t)find_cuda_sym("cudaMallocAsync");
    if (!real) return -1;
    if (in_hook) return real(devPtr, size, stream);
    in_hook = 1;

    char extra[128];
    snprintf(extra, sizeof(extra), "type=alloc_async,bytes=%zu,stream=%d",
             size, get_stream_id(stream));
    uint64_t t0 = now_ns();
    cudaError_t ret = real(devPtr, size, stream);
    if (ret == 0 && devPtr && *devPtr)
        mem_track_add(*devPtr, size);
    emit_span("memory", gettid_compat(), t0, now_ns() - t0, "cudaMallocAsync", extra);

    in_hook = 0;
    return ret;
}

cudaError_t cudaFree(void *devPtr) {
    typedef cudaError_t (*fn_t)(void*);
    static fn_t real = NULL;
    if (!real) real = (fn_t)find_cuda_sym("cudaFree");
    if (!real) return -1;
    if (in_hook) return real(devPtr);
    in_hook = 1;

    mem_track_rem(devPtr);
    uint64_t t0 = now_ns();
    cudaError_t ret = real(devPtr);
    emit_span("memory", gettid_compat(), t0, now_ns() - t0, "cudaFree", "type=free");

    in_hook = 0;
    return ret;
}

cudaError_t cudaFreeAsync(void *devPtr, cudaStream_t stream) {
    typedef cudaError_t (*fn_t)(void*, cudaStream_t);
    static fn_t real = NULL;
    if (!real) real = (fn_t)find_cuda_sym("cudaFreeAsync");
    if (!real) return -1;
    if (in_hook) return real(devPtr, stream);
    in_hook = 1;

    mem_track_rem(devPtr);
    char extra[64];
    snprintf(extra, sizeof(extra), "type=free_async,stream=%d", get_stream_id(stream));
    uint64_t t0 = now_ns();
    cudaError_t ret = real(devPtr, stream);
    emit_span("memory", gettid_compat(), t0, now_ns() - t0, "cudaFreeAsync", extra);

    in_hook = 0;
    return ret;
}

cudaError_t cudaHostAlloc(void **pHost, size_t size, unsigned int flags) {
    typedef cudaError_t (*fn_t)(void**, size_t, unsigned int);
    static fn_t real = NULL;
    if (!real) real = (fn_t)find_cuda_sym("cudaHostAlloc");
    if (!real) return -1;
    if (in_hook) return real(pHost, size, flags);
    in_hook = 1;

    char extra[64];
    snprintf(extra, sizeof(extra), "type=alloc_pinned,bytes=%zu", size);
    uint64_t t0 = now_ns();
    cudaError_t ret = real(pHost, size, flags);
    if (ret == 0 && pHost && *pHost)
        pin_track_add(*pHost, size);
    emit_span("memory", gettid_compat(), t0, now_ns() - t0, "cudaHostAlloc", extra);

    in_hook = 0;
    return ret;
}

cudaError_t cudaMallocHost(void **ptr, size_t size) {
    typedef cudaError_t (*fn_t)(void**, size_t);
    static fn_t real = NULL;
    if (!real) real = (fn_t)find_cuda_sym("cudaMallocHost");
    if (!real) return -1;
    if (in_hook) return real(ptr, size);
    in_hook = 1;

    char extra[64];
    snprintf(extra, sizeof(extra), "type=alloc_pinned,bytes=%zu", size);
    uint64_t t0 = now_ns();
    cudaError_t ret = real(ptr, size);
    if (ret == 0 && ptr && *ptr)
        pin_track_add(*ptr, size);
    emit_span("memory", gettid_compat(), t0, now_ns() - t0, "cudaMallocHost", extra);

    in_hook = 0;
    return ret;
}

cudaError_t cudaFreeHost(void *ptr) {
    typedef cudaError_t (*fn_t)(void*);
    static fn_t real = NULL;
    if (!real) real = (fn_t)find_cuda_sym("cudaFreeHost");
    if (!real) return -1;
    if (in_hook) return real(ptr);
    in_hook = 1;

    pin_track_rem(ptr);
    uint64_t t0 = now_ns();
    cudaError_t ret = real(ptr);
    emit_span("memory", gettid_compat(), t0, now_ns() - t0,
              "cudaFreeHost", "type=free_pinned");

    in_hook = 0;
    return ret;
}

cudaError_t cudaDeviceSynchronize(void) {
    typedef cudaError_t (*fn_t)(void);
    static fn_t real = NULL;
    if (!real) real = (fn_t)find_cuda_sym("cudaDeviceSynchronize");
    if (!real) return -1;
    if (in_hook) return real();
    in_hook = 1;

    uint64_t t0 = now_ns();
    cudaError_t ret = real();
    pk_flush(NULL, 1);
    emit_span("sync", gettid_compat(), t0, now_ns() - t0,
              "cudaDeviceSynchronize", "type=sync");

    in_hook = 0;
    return ret;
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    typedef cudaError_t (*fn_t)(cudaStream_t);
    static fn_t real = NULL;
    if (!real) real = (fn_t)find_cuda_sym("cudaStreamSynchronize");
    if (!real) return -1;
    if (in_hook) return real(stream);
    in_hook = 1;

    uint64_t t0 = now_ns();
    cudaError_t ret = real(stream);
    pk_flush(stream, 0);
    char extra[64];
    snprintf(extra, sizeof(extra), "type=sync,stream=%d", get_stream_id(stream));
    emit_span("sync", gettid_compat(), t0, now_ns() - t0,
              "cudaStreamSynchronize", extra);

    in_hook = 0;
    return ret;
}

cudaError_t cudaEventSynchronize(cudaEvent_t event) {
    typedef cudaError_t (*fn_t)(cudaEvent_t);
    static fn_t real = NULL;
    if (!real) real = (fn_t)find_cuda_sym("cudaEventSynchronize");
    if (!real) return -1;
    if (in_hook) return real(event);
    in_hook = 1;

    uint64_t t0 = now_ns();
    cudaError_t ret = real(event);
    /* Can't know which stream the event was recorded on; flush all pending */
    pk_flush(NULL, 1);
    emit_span("sync", gettid_compat(), t0, now_ns() - t0,
              "cudaEventSynchronize", "type=sync");

    in_hook = 0;
    return ret;
}

cudaError_t cudaDeviceReset(void) {
    typedef cudaError_t (*fn_t)(void);
    static fn_t real = NULL;
    if (!real) real = (fn_t)find_cuda_sym("cudaDeviceReset");
    if (!real) return -1;
    if (in_hook) return real();
    in_hook = 1;

    /* Flush before reset — device state is destroyed after */
    pk_flush(NULL, 1);
    uint64_t t0 = now_ns();
    cudaError_t ret = real();
    emit_span("sync", gettid_compat(), t0, now_ns() - t0,
              "cudaDeviceReset", "type=sync");

    in_hook = 0;
    return ret;
}

/* ── CUDA Driver API ────────────────────────────────────────────────────── */

static void _save_cubin(const void *image) {
    if (!image) return;
    static int cubin_counter = 0;
    char path[256];
    snprintf(path, sizeof(path), "/tmp/hprofiler_cubin_%d_%d.bin",
             (int)getpid(), cubin_counter++);
    size_t sz = 0;
    const uint8_t *p = (const uint8_t *)image;

    if (p[0]==0x50 && p[1]==0xED && p[2]==0x55 && p[3]==0xBA) {
        uint64_t fsz;
        memcpy(&fsz, p + 8, 8);
        if (fsz > 32 && fsz < 512ULL*1024*1024) sz = (size_t)fsz;
    } else if (p[0]==0x7f && p[1]=='E' && p[2]=='L' && p[3]=='F' && p[4]==2) {
        uint64_t shoff; memcpy(&shoff, p+40, 8);
        uint16_t shesz; memcpy(&shesz, p+58, 2);
        uint16_t shnum; memcpy(&shnum, p+60, 2);
        size_t end = (size_t)(shoff + (uint64_t)shesz * shnum);
        if (end > 64 && end < 512ULL*1024*1024) sz = end;
    } else if ((p[0]=='/' && p[1]=='/') || p[0]=='.') {
        sz = strnlen((const char *)image, 64*1024*1024);
        if (sz > 0) sz++;
    }
    FILE *f = fopen(path, "wb");
    if (f) { fwrite(image, 1, sz, f); fclose(f); }
}

CUresult cuModuleLoadData(CUmodule_t *module, const void *image) {
    typedef CUresult (*fn_t)(CUmodule_t *, const void *);
    static fn_t real = NULL;
    if (!real) real = (fn_t)find_cuda_sym("cuModuleLoadData");
    if (!real) return -1;
    _save_cubin(image);
    uint64_t t0 = now_ns();
    CUresult ret = real(module, image);
    emit_span("jit", gettid_compat(), t0, now_ns() - t0,
              "cuModuleLoadData", "type=jit_compile");
    return ret;
}

CUresult cuModuleLoadDataEx(CUmodule_t *module, const void *image,
                             unsigned int numOptions, void *options,
                             void *optionValues) {
    typedef CUresult (*fn_t)(CUmodule_t *, const void *, unsigned, void *, void *);
    static fn_t real = NULL;
    if (!real) real = (fn_t)find_cuda_sym("cuModuleLoadDataEx");
    if (!real) return -1;
    _save_cubin(image);
    uint64_t t0 = now_ns();
    CUresult ret = real(module, image, numOptions, options, optionValues);
    emit_span("jit", gettid_compat(), t0, now_ns() - t0,
              "cuModuleLoadDataEx", "type=jit_compile");
    return ret;
}

CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule_t hmod,
                              const char *name) {
    typedef CUresult (*fn_t)(CUfunction *, CUmodule_t, const char *);
    static fn_t real = NULL;
    if (!real) real = (fn_t)find_cuda_sym("cuModuleGetFunction");
    if (!real) return -1;
    CUresult ret = real(hfunc, hmod, name);
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

CUresult cuLaunchKernel(
    CUfunction f, unsigned int gx, unsigned int gy, unsigned int gz,
    unsigned int bx, unsigned int by, unsigned int bz,
    unsigned int sharedMem, CUstream hStream,
    void **kernelParams, void **extra_params)
{
    typedef CUresult (*fn_t)(CUfunction, unsigned, unsigned, unsigned,
                              unsigned, unsigned, unsigned, unsigned,
                              CUstream, void**, void**);
    static fn_t real = NULL;
    if (!real) real = (fn_t)find_cuda_sym("cuLaunchKernel");
    if (!real) return -1;
    if (in_hook) return real(f, gx,gy,gz, bx,by,bz, sharedMem, hStream,
                             kernelParams, extra_params);
    in_hook = 1;

    const char *kname = resolve_kernel_name(f);
    pid_t tid = gettid_compat();
    cudaStream_t stream = (cudaStream_t)hStream;
    int sid = get_stream_id(stream);
    char extra[256];
    snprintf(extra, sizeof(extra),
             "type=kernel,grid=%ux%ux%u,block=%ux%ux%u,stream=%d",
             gx,gy,gz, bx,by,bz, sid);

    cudaEvent_t ev_s, ev_e;
    int gpu_ok = pk_try_begin(stream, &ev_s, &ev_e);
    uint64_t t0 = now_ns();
    CUresult ret = real(f, gx,gy,gz, bx,by,bz, sharedMem, hStream,
                        kernelParams, extra_params);
    if (gpu_ok)
        pk_commit(ev_s, ev_e, stream, "cuda", kname, extra, t0, tid);
    else
        emit_span("cuda", tid, t0, now_ns() - t0, kname, extra);

    in_hook = 0;
    return ret;
}

CUresult cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src,
                       size_t bytes, CUstream stream) {
    typedef CUresult (*fn_t)(CUdeviceptr, CUdeviceptr, size_t, CUstream);
    static fn_t real = NULL;
    if (!real) real = (fn_t)find_cuda_sym("cuMemcpyAsync");
    if (!real) return -1;
    if (in_hook) return real(dst, src, bytes, stream);
    in_hook = 1;

    pid_t tid = gettid_compat();
    cudaStream_t cstream = (cudaStream_t)stream;
    int sid = get_stream_id(cstream);
    char extra[128];
    snprintf(extra, sizeof(extra), "type=memcpy_async,bytes=%zu,stream=%d", bytes, sid);

    cudaEvent_t ev_s, ev_e;
    int gpu_ok = pk_try_begin(cstream, &ev_s, &ev_e);
    uint64_t t0 = now_ns();
    CUresult ret = real(dst, src, bytes, stream);
    if (gpu_ok)
        pk_commit(ev_s, ev_e, cstream, "memory", "cuMemcpyAsync", extra, t0, tid);
    else
        emit_span("memory", tid, t0, now_ns() - t0, "cuMemcpyAsync", extra);

    in_hook = 0;
    return ret;
}

CUresult cuMemcpyHtoDAsync(CUdeviceptr dst, const void *src,
                            size_t bytes, CUstream stream) {
    typedef CUresult (*fn_t)(CUdeviceptr, const void*, size_t, CUstream);
    static fn_t real = NULL;
    if (!real) {
        real = (fn_t)find_cuda_sym("cuMemcpyHtoDAsync_v2");
        if (!real) real = (fn_t)find_cuda_sym("cuMemcpyHtoDAsync");
    }
    if (!real) return -1;
    if (in_hook) return real(dst, src, bytes, stream);
    in_hook = 1;

    pid_t tid = gettid_compat();
    cudaStream_t cstream = (cudaStream_t)stream;
    int sid = get_stream_id(cstream);
    char extra[128];
    snprintf(extra, sizeof(extra), "type=HtoD,bytes=%zu,stream=%d", bytes, sid);

    cudaEvent_t ev_s, ev_e;
    int gpu_ok = pk_try_begin(cstream, &ev_s, &ev_e);
    uint64_t t0 = now_ns();
    CUresult ret = real(dst, src, bytes, stream);
    if (gpu_ok)
        pk_commit(ev_s, ev_e, cstream, "memory", "cuMemcpyHtoD", extra, t0, tid);
    else
        emit_span("memory", tid, t0, now_ns() - t0, "cuMemcpyHtoD", extra);

    in_hook = 0;
    return ret;
}

CUresult cuMemcpyDtoHAsync(void *dst, CUdeviceptr src,
                            size_t bytes, CUstream stream) {
    typedef CUresult (*fn_t)(void*, CUdeviceptr, size_t, CUstream);
    static fn_t real = NULL;
    if (!real) {
        real = (fn_t)find_cuda_sym("cuMemcpyDtoHAsync_v2");
        if (!real) real = (fn_t)find_cuda_sym("cuMemcpyDtoHAsync");
    }
    if (!real) return -1;
    if (in_hook) return real(dst, src, bytes, stream);
    in_hook = 1;

    pid_t tid = gettid_compat();
    cudaStream_t cstream = (cudaStream_t)stream;
    int sid = get_stream_id(cstream);
    char extra[128];
    snprintf(extra, sizeof(extra), "type=DtoH,bytes=%zu,stream=%d", bytes, sid);

    cudaEvent_t ev_s, ev_e;
    int gpu_ok = pk_try_begin(cstream, &ev_s, &ev_e);
    uint64_t t0 = now_ns();
    CUresult ret = real(dst, src, bytes, stream);
    if (gpu_ok)
        pk_commit(ev_s, ev_e, cstream, "memory", "cuMemcpyDtoH", extra, t0, tid);
    else
        emit_span("memory", tid, t0, now_ns() - t0, "cuMemcpyDtoH", extra);

    in_hook = 0;
    return ret;
}

CUresult cuMemAlloc(CUdeviceptr *dptr, size_t bytes) {
    typedef CUresult (*fn_t)(CUdeviceptr*, size_t);
    static fn_t real = NULL;
    if (!real) {
        real = (fn_t)find_cuda_sym("cuMemAlloc_v2");
        if (!real) real = (fn_t)find_cuda_sym("cuMemAlloc");
    }
    if (!real) return -1;
    if (in_hook) return real(dptr, bytes);
    in_hook = 1;

    char extra[64];
    snprintf(extra, sizeof(extra), "type=alloc,bytes=%zu", bytes);
    uint64_t t0 = now_ns();
    CUresult ret = real(dptr, bytes);
    if (ret == 0 && dptr && *dptr)
        mem_track_add((void*)(uintptr_t)*dptr, bytes);
    emit_span("memory", gettid_compat(), t0, now_ns() - t0, "cuMemAlloc", extra);

    in_hook = 0;
    return ret;
}

CUresult cuMemAllocManaged(CUdeviceptr *dptr, size_t bytes, unsigned int flags) {
    typedef CUresult (*fn_t)(CUdeviceptr*, size_t, unsigned int);
    static fn_t real = NULL;
    if (!real) real = (fn_t)find_cuda_sym("cuMemAllocManaged");
    if (!real) return -1;
    if (in_hook) return real(dptr, bytes, flags);
    in_hook = 1;

    char extra[64];
    snprintf(extra, sizeof(extra), "type=alloc_managed,bytes=%zu", bytes);
    uint64_t t0 = now_ns();
    CUresult ret = real(dptr, bytes, flags);
    if (ret == 0 && dptr && *dptr)
        mem_track_add((void*)(uintptr_t)*dptr, bytes);
    emit_span("memory", gettid_compat(), t0, now_ns() - t0, "cuMemAllocManaged", extra);

    in_hook = 0;
    return ret;
}

CUresult cuMemAllocAsync(CUdeviceptr *dptr, size_t bytes, CUstream stream) {
    typedef CUresult (*fn_t)(CUdeviceptr*, size_t, CUstream);
    static fn_t real = NULL;
    if (!real) real = (fn_t)find_cuda_sym("cuMemAllocAsync");
    if (!real) return -1;
    if (in_hook) return real(dptr, bytes, stream);
    in_hook = 1;

    char extra[128];
    snprintf(extra, sizeof(extra), "type=alloc_async,bytes=%zu,stream=%d",
             bytes, get_stream_id((cudaStream_t)stream));
    uint64_t t0 = now_ns();
    CUresult ret = real(dptr, bytes, stream);
    if (ret == 0 && dptr && *dptr)
        mem_track_add((void*)(uintptr_t)*dptr, bytes);
    emit_span("memory", gettid_compat(), t0, now_ns() - t0, "cuMemAllocAsync", extra);

    in_hook = 0;
    return ret;
}

CUresult cuMemFree(CUdeviceptr dptr) {
    typedef CUresult (*fn_t)(CUdeviceptr);
    static fn_t real = NULL;
    if (!real) {
        real = (fn_t)find_cuda_sym("cuMemFree_v2");
        if (!real) real = (fn_t)find_cuda_sym("cuMemFree");
    }
    if (!real) return -1;
    if (in_hook) return real(dptr);
    in_hook = 1;

    mem_track_rem((void*)(uintptr_t)dptr);
    uint64_t t0 = now_ns();
    CUresult ret = real(dptr);
    emit_span("memory", gettid_compat(), t0, now_ns() - t0, "cuMemFree", "type=free");

    in_hook = 0;
    return ret;
}

CUresult cuMemFreeAsync(CUdeviceptr dptr, CUstream stream) {
    typedef CUresult (*fn_t)(CUdeviceptr, CUstream);
    static fn_t real = NULL;
    if (!real) real = (fn_t)find_cuda_sym("cuMemFreeAsync");
    if (!real) return -1;
    if (in_hook) return real(dptr, stream);
    in_hook = 1;

    mem_track_rem((void*)(uintptr_t)dptr);
    char extra[64];
    snprintf(extra, sizeof(extra), "type=free_async,stream=%d",
             get_stream_id((cudaStream_t)stream));
    uint64_t t0 = now_ns();
    CUresult ret = real(dptr, stream);
    emit_span("memory", gettid_compat(), t0, now_ns() - t0, "cuMemFreeAsync", extra);

    in_hook = 0;
    return ret;
}

CUresult cuStreamSynchronize(CUstream stream) {
    typedef CUresult (*fn_t)(CUstream);
    static fn_t real = NULL;
    if (!real) real = (fn_t)find_cuda_sym("cuStreamSynchronize");
    if (!real) return -1;
    if (in_hook) return real(stream);
    in_hook = 1;

    uint64_t t0 = now_ns();
    CUresult ret = real(stream);
    pk_flush((cudaStream_t)stream, 0);
    char extra[64];
    snprintf(extra, sizeof(extra), "type=sync,stream=%d", get_stream_id(stream));
    emit_span("sync", gettid_compat(), t0, now_ns() - t0,
              "cuStreamSynchronize", extra);

    in_hook = 0;
    return ret;
}

CUresult cuCtxSynchronize(void) {
    typedef CUresult (*fn_t)(void);
    static fn_t real = NULL;
    if (!real) real = (fn_t)find_cuda_sym("cuCtxSynchronize");
    if (!real) return -1;
    if (in_hook) return real();
    in_hook = 1;

    uint64_t t0 = now_ns();
    CUresult ret = real();
    pk_flush(NULL, 1);
    emit_span("sync", gettid_compat(), t0, now_ns() - t0,
              "cuCtxSynchronize", "type=sync");

    in_hook = 0;
    return ret;
}

/* ── NVTX range interception ─────────────────────────────────────────────── */
/*
 * Replaces libnvToolsExt (v1/v2 API) entirely — no forwarding needed.
 * NVTX v3 (header-only, inline) is not intercepted here.
 * No in_hook guard needed: NVTX functions don't call any CUDA Runtime API.
 */

#define MAX_NVTX_DEPTH 64

typedef struct {
    uint64_t start_ns;
    uint64_t span_id;   /* unique ID = start_ns at push time */
    char     name[256];
    pid_t    tid;
} NvtxEntry;

static __thread NvtxEntry nvtx_stack[MAX_NVTX_DEPTH];
static __thread int       nvtx_depth = 0;
/* tls_nvtx_span_id is declared near the top of this file (before the wrappers). */

typedef struct {
    uint16_t version;
    uint16_t size;
    uint32_t category;
    int      colorType;
    uint32_t color;
    int      payloadType;
    int      reserved0;
    union { uint64_t ull; int64_t ll; double d; uint32_t ui; int32_t i; } payload;
    int      messageType;
    union { const char *ascii; const wchar_t *unicode; } message;
} NvtxEvAttr_t;

int nvtxRangePushA(const char *message) {
    if (nvtx_depth < MAX_NVTX_DEPTH) {
        NvtxEntry *e = &nvtx_stack[nvtx_depth];
        e->start_ns = now_ns();
        e->span_id  = e->start_ns;   /* unique: nanosecond timestamp */
        e->tid      = gettid_compat();
        strncpy(e->name, message ? message : "<nvtx>", sizeof(e->name) - 1);
        e->name[sizeof(e->name) - 1] = '\0';
        nvtx_depth++;
        tls_nvtx_span_id = e->span_id;
    }
    return nvtx_depth - 1;
}

int nvtxRangePushW(const wchar_t *message) {
    char narrow[256] = "<nvtx>";
    if (message) {
        int i = 0;
        while (message[i] && i < (int)(sizeof(narrow) - 1)) {
            narrow[i] = (char)message[i]; i++;
        }
        narrow[i] = '\0';
    }
    return nvtxRangePushA(narrow);
}

int nvtxRangePushEx(const void *attr_v) {
    const char *name = "<nvtx>";
    if (attr_v) {
        const NvtxEvAttr_t *a = (const NvtxEvAttr_t *)attr_v;
        if (a->messageType == 1 && a->message.ascii)
            name = a->message.ascii;
        else if (a->messageType == 2 && a->message.unicode)
            return nvtxRangePushW(a->message.unicode);
    }
    return nvtxRangePushA(name);
}

int nvtxRangePop(void) {
    if (nvtx_depth <= 0) return -1;
    nvtx_depth--;
    NvtxEntry *e = &nvtx_stack[nvtx_depth];
    uint64_t dur = now_ns() - e->start_ns;
    char nvtx_extra[64];
    snprintf(nvtx_extra, sizeof(nvtx_extra), "type=nvtx_range,sid=%llu",
             (unsigned long long)e->span_id);
    emit_span("nvtx", e->tid, e->start_ns, dur, e->name, nvtx_extra);
    /* Restore parent NVTX span ID (0 if no enclosing range). */
    tls_nvtx_span_id = (nvtx_depth > 0) ? nvtx_stack[nvtx_depth - 1].span_id : 0;
    return 0;
}

void nvtxMarkA(const char *message) {
    if (!message) return;
    uint64_t ts = now_ns();
    emit_span("nvtx", gettid_compat(), ts, 0, message, "type=nvtx_mark");
}

void nvtxMarkW(const wchar_t *message) {
    char narrow[256] = "<nvtx-mark>";
    if (message) {
        int i = 0;
        while (message[i] && i < (int)(sizeof(narrow) - 1)) {
            narrow[i] = (char)message[i]; i++;
        }
        narrow[i] = '\0';
    }
    nvtxMarkA(narrow);
}

/* ── CUDA Graph launch interception ─────────────────────────────────────── */

/* Runtime API: cudaGraphLaunch */
cudaError_t cudaGraphLaunch(void *graphExec, cudaStream_t stream) {
    typedef cudaError_t (*fn_t)(void*, cudaStream_t);
    static fn_t real = NULL;
    if (!real) real = (fn_t)find_cuda_sym("cudaGraphLaunch");
    if (!real) return -1;
    if (in_hook) return real(graphExec, stream);
    in_hook = 1;

    pid_t tid = gettid_compat();
    int sid = get_stream_id(stream);
    char extra[128];
    snprintf(extra, sizeof(extra), "type=graph_launch,stream=%d", sid);

    cudaEvent_t ev_s, ev_e;
    int gpu_ok = pk_try_begin(stream, &ev_s, &ev_e);
    uint64_t t0 = now_ns();
    cudaError_t ret = real(graphExec, stream);
    if (gpu_ok)
        pk_commit(ev_s, ev_e, stream, "cuda", "cudaGraphLaunch", extra, t0, tid);
    else
        emit_span("cuda", tid, t0, now_ns() - t0, "cudaGraphLaunch", extra);

    in_hook = 0;
    return ret;
}

/* Driver API: cuGraphLaunch */
CUresult cuGraphLaunch(void *hGraphExec, CUstream hStream) {
    typedef CUresult (*fn_t)(void*, CUstream);
    static fn_t real = NULL;
    if (!real) real = (fn_t)find_cuda_sym("cuGraphLaunch");
    if (!real) return -1;
    if (in_hook) return real(hGraphExec, hStream);
    in_hook = 1;

    pid_t tid = gettid_compat();
    cudaStream_t stream = (cudaStream_t)hStream;
    int sid = get_stream_id(stream);
    char extra[128];
    snprintf(extra, sizeof(extra), "type=graph_launch,stream=%d", sid);

    cudaEvent_t ev_s, ev_e;
    int gpu_ok = pk_try_begin(stream, &ev_s, &ev_e);
    uint64_t t0 = now_ns();
    CUresult ret = real(hGraphExec, hStream);
    if (gpu_ok)
        pk_commit(ev_s, ev_e, stream, "cuda", "cuGraphLaunch", extra, t0, tid);
    else
        emit_span("cuda", tid, t0, now_ns() - t0, "cuGraphLaunch", extra);

    in_hook = 0;
    return ret;
}

/* ── dlopen / dlsym overrides for static-cudart interception ────────────── */

void *dlopen(const char *filename, int flags) {
    typedef void *(*fn_t)(const char *, int);
    static fn_t real = NULL;
    if (!real && g_real_dlsym)
        real = (fn_t)g_real_dlsym(RTLD_NEXT, "dlopen");
    void *h = real ? real(filename, flags) : NULL;
    /* Record handles for libcuda.so so dlsym can redirect driver API calls. */
    if (h && filename &&
        (strstr(filename, "libcuda.so") || strstr(filename, "nvcuda.so"))) {
        pthread_mutex_lock(&g_cuda_drv_mutex);
        int found = 0;
        for (int i = 0; i < g_cuda_drv_n; i++)
            if (g_cuda_drv_handles[i] == h) { found = 1; break; }
        if (!found && g_cuda_drv_n < CUDA_DRV_HANDLE_CAP)
            g_cuda_drv_handles[g_cuda_drv_n++] = h;
        pthread_mutex_unlock(&g_cuda_drv_mutex);
    }
    return h;
}

static int is_cuda_drv_handle(void *h) {
    pthread_mutex_lock(&g_cuda_drv_mutex);
    for (int i = 0; i < g_cuda_drv_n; i++)
        if (g_cuda_drv_handles[i] == h) {
            pthread_mutex_unlock(&g_cuda_drv_mutex);
            return 1;
        }
    pthread_mutex_unlock(&g_cuda_drv_mutex);
    return 0;
}

/* cuGetProcAddress / cuGetProcAddress_v2 ─────────────────────────────────
 * CUDA 11+ static runtimes fetch all driver API pointers through this
 * single lookup instead of individual dlsym calls.  We intercept both
 * variants and substitute our wrappers for known symbols.
 *
 * Layout: forward-declare both functions → define intercept table →
 * define function bodies (which reference the now-visible table).      */

typedef unsigned long long cuuint64_t;

/* Forward declarations so the intercept table can hold their addresses. */
CUresult cuGetProcAddress(const char*, void**, int, cuuint64_t);
CUresult cuGetProcAddress_v2(const char*, void**, int, cuuint64_t, void*);

/* Driver API symbol names looked up by the static CUDA runtime.
 * Versioned aliases (_v2) map to the same wrapper.                     */
typedef struct { const char *name; void *fn; } DriveIntercept;
static const DriveIntercept g_drv_intercepts[] = {
    /* cuGetProcAddress itself — so dlsym(handle,"cuGetProcAddress")
     * returns our wrapper and all subsequent lookups go through it.    */
    {"cuGetProcAddress",      (void*)cuGetProcAddress},
    {"cuGetProcAddress_v2",   (void*)cuGetProcAddress_v2},
    {"cuLaunchKernel",        (void*)cuLaunchKernel},
    {"cuMemcpyAsync",         (void*)cuMemcpyAsync},
    {"cuMemcpyHtoDAsync",     (void*)cuMemcpyHtoDAsync},
    {"cuMemcpyHtoDAsync_v2",  (void*)cuMemcpyHtoDAsync},
    {"cuMemcpyDtoHAsync",     (void*)cuMemcpyDtoHAsync},
    {"cuMemcpyDtoHAsync_v2",  (void*)cuMemcpyDtoHAsync},
    {"cuMemAlloc",            (void*)cuMemAlloc},
    {"cuMemAlloc_v2",         (void*)cuMemAlloc},
    {"cuMemAllocManaged",     (void*)cuMemAllocManaged},
    {"cuMemAllocAsync",       (void*)cuMemAllocAsync},
    {"cuMemFree",             (void*)cuMemFree},
    {"cuMemFree_v2",          (void*)cuMemFree},
    {"cuMemFreeAsync",        (void*)cuMemFreeAsync},
    {"cuStreamSynchronize",   (void*)cuStreamSynchronize},
    {"cuModuleLoadData",      (void*)cuModuleLoadData},
    {"cuModuleLoadDataEx",    (void*)cuModuleLoadDataEx},
    {"cuModuleGetFunction",   (void*)cuModuleGetFunction},
    {NULL, NULL},
};

/* cuGetProcAddress function bodies — placed after the table they reference. */

static void *_real_cuGetProcAddress(const char *vname) {
    void *fn = NULL;
    pthread_mutex_lock(&g_cuda_drv_mutex);
    for (int i = 0; i < g_cuda_drv_n && !fn; i++)
        fn = g_real_dlsym ? g_real_dlsym(g_cuda_drv_handles[i], vname) : NULL;
    pthread_mutex_unlock(&g_cuda_drv_mutex);
    /* RTLD_NEXT bypasses our own export and finds the symbol in libcuda.so. */
    if (!fn && g_real_dlsym)
        fn = g_real_dlsym(RTLD_NEXT, vname);
    return fn;
}

CUresult cuGetProcAddress(const char *symbol, void **pfn,
                           int cudaVersion, cuuint64_t flags) {
    typedef CUresult (*fn_t)(const char*, void**, int, cuuint64_t);
    static fn_t real = NULL;
    if (!real) real = (fn_t)_real_cuGetProcAddress("cuGetProcAddress");
    if (!real || !pfn) return -1;
    CUresult ret = real(symbol, pfn, cudaVersion, flags);
    if (ret == 0 && *pfn)
        for (int i = 0; g_drv_intercepts[i].name; i++)
            if (strcmp(symbol, g_drv_intercepts[i].name) == 0)
                { *pfn = g_drv_intercepts[i].fn; break; }
    return ret;
}

CUresult cuGetProcAddress_v2(const char *symbol, void **pfn,
                              int cudaVersion, cuuint64_t flags,
                              void *status) {
    typedef CUresult (*fn_t)(const char*, void**, int, cuuint64_t, void*);
    static fn_t real = NULL;
    if (!real) real = (fn_t)_real_cuGetProcAddress("cuGetProcAddress_v2");
    if (!real || !pfn) return -1;
    CUresult ret = real(symbol, pfn, cudaVersion, flags, status);
    if (ret == 0 && *pfn)
        for (int i = 0; g_drv_intercepts[i].name; i++)
            if (strcmp(symbol, g_drv_intercepts[i].name) == 0)
                { *pfn = g_drv_intercepts[i].fn; break; }
    return ret;
}

void *dlsym(void *handle, const char *symbol) {
    if (!g_real_dlsym) return NULL;
    if (handle == RTLD_NEXT || handle == RTLD_DEFAULT || !symbol)
        return g_real_dlsym(handle, symbol);

    void *real_sym = g_real_dlsym(handle, symbol);

    if (is_cuda_drv_handle(handle) && real_sym)
        for (int i = 0; g_drv_intercepts[i].name; i++)
            if (strcmp(symbol, g_drv_intercepts[i].name) == 0)
                return g_drv_intercepts[i].fn;
    return real_sym;
}

/* Priority 50: initialise g_real_dlsym before any dlsym call is made
 * (including the RTLD_NEXT lookups inside our own wrapper functions).  */
__attribute__((constructor(50)))
static void hprofiler_cuda_dlsym_init(void) {
    static const char *vers[] = {
        "GLIBC_2.2.5", "GLIBC_2.17", "GLIBC_2.34", NULL
    };
    for (int i = 0; vers[i] && !g_real_dlsym; i++)
        g_real_dlsym = (void*(*)(void*, const char*))
            dlvsym(RTLD_NEXT, "dlsym", vers[i]);
}

/* ── Constructor / Destructor ───────────────────────────────────────────── */

__attribute__((constructor))
static void hprofiler_cuda_init(void) {
    ensure_connected();
    cs_init();
    /* libcudart: only use if already loaded — force-loading it alongside a
     * statically-linked CUDA runtime causes dual-runtime conflicts
     * (CUDA_ERROR_INVALID_CONTEXT) because both runtimes try to own the
     * device context independently.  libcuda.so (driver) is safe to load
     * unconditionally; it is always shared and there is only one copy.   */
    static const char *rt_candidates[] = {
        "libcudart.so.12", "libcudart.so.11", "libcudart.so", NULL
    };
    for (int i = 0; rt_candidates[i]; i++) {
        void *h = dlopen(rt_candidates[i], RTLD_LAZY | RTLD_GLOBAL | RTLD_NOLOAD);
        if (h && !g_cudart_handle) g_cudart_handle = h;
    }
    static const char *drv_candidates[] = {
        "libcuda.so.1", "libcuda.so", NULL
    };
    for (int i = 0; drv_candidates[i]; i++) {
        void *h = dlopen(drv_candidates[i], RTLD_LAZY | RTLD_GLOBAL | RTLD_NOLOAD);
        if (!h) h = dlopen(drv_candidates[i], RTLD_LAZY | RTLD_GLOBAL);
        if (h && !g_cudart_handle) g_cudart_handle = h;
    }
}

__attribute__((destructor))
static void hprofiler_cuda_fini(void) {
    pk_flush(NULL, 1);

    /* Report leaked GPU allocations (cudaMalloc without matching cudaFree). */
    pthread_mutex_lock(&g_alloc_mutex);
    if (g_alloc_n > 0) {
        int64_t leaked = 0;
        for (int i = 0; i < g_alloc_n; i++)
            leaked += (int64_t)g_allocs[i].sz;
        pthread_mutex_unlock(&g_alloc_mutex);
        emit_ctr("memory", "gpu_memory_leaked_bytes", leaked, "bytes");
    } else {
        pthread_mutex_unlock(&g_alloc_mutex);
    }

    if (g_sock >= 0) {
        close(g_sock);
        g_sock = -1;
    }
}
