/*
 * NCCL collective operations LD_PRELOAD hook.
 *
 * Intercepts NCCL multi-GPU communication calls and records GPU-accurate
 * spans using cudaEvent pairs (same mechanism as the CUDA hook).
 *
 * Captured calls:
 *   ncclAllReduce, ncclBroadcast, ncclReduce, ncclAllGather, ncclReduceScatter
 *   ncclSend, ncclRecv
 *   ncclGroupStart / ncclGroupEnd  (mark group boundaries as spans)
 *
 * Wire protocol: identical to cuda_hook — spans on HPROFILER_SOCKET.
 * Category: "nccl"
 *
 * Tags:
 *   type=allreduce|broadcast|...
 *   bytes=N          (count × dtype_size)
 *   stream=ID
 *
 * Build requirements:
 *   gcc -shared -fPIC -o libhprofiler_nccl.so nccl_hook.c -ldl -lpthread
 *   (NCCL headers not required — only type stubs below)
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

/* Common NCCL datatype sizes (index matches ncclDataType_t enum order). */
static const size_t _nccl_dtype_sizes[] = {
    1,  /* ncclInt8   / ncclChar  */
    1,  /* ncclUint8              */
    4,  /* ncclInt32  / ncclInt   */
    4,  /* ncclUint32             */
    8,  /* ncclInt64              */
    8,  /* ncclUint64             */
    2,  /* ncclFloat16 / ncclHalf */
    4,  /* ncclFloat32 / ncclFloat*/
    8,  /* ncclFloat64 /ncclDouble*/
    2,  /* ncclBfloat16           */
};
static size_t nccl_dtype_sz(ncclDataType_t dt) {
    if (dt >= 0 && (size_t)dt < sizeof(_nccl_dtype_sizes)/sizeof(_nccl_dtype_sizes[0]))
        return _nccl_dtype_sizes[dt];
    return 4;  /* fallback */
}

/* ── Globals (shared with cuda_hook via socket) ──────────────────────── */
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
        g_sock = s; g_pid = getpid();
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

static void emit_span(const char *cat, pid_t tid, uint64_t start_ns,
                      uint64_t dur_ns, const char *name, const char *extra) {
    pthread_mutex_lock(&g_sock_mutex);
    ensure_connected();
    if (g_sock >= 0) {
        char buf[1024];
        int n = snprintf(buf, sizeof(buf),
            "span:%s:%d:%d:%llu:%llu:%s:%s\n",
            cat, g_pid, (int)tid,
            (unsigned long long)start_ns, (unsigned long long)dur_ns,
            name, extra ? extra : "");
        if (n > 0 && n < (int)sizeof(buf)) send_all(buf, n);
    }
    pthread_mutex_unlock(&g_sock_mutex);
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

static int ev_ok(void) {
    if (!f_evCreate) {
        f_evCreate  = (fn_EvCreate_t) dlsym(RTLD_DEFAULT, "cudaEventCreate");
        f_evRecord  = (fn_EvRecord_t) dlsym(RTLD_DEFAULT, "cudaEventRecord");
        f_evElapsed = (fn_EvElapsed_t)dlsym(RTLD_DEFAULT, "cudaEventElapsedTime");
        f_evDestroy = (fn_EvDestroy_t)dlsym(RTLD_DEFAULT, "cudaEventDestroy");
        f_evSync    = (fn_EvSync_t)   dlsym(RTLD_DEFAULT, "cudaEventSynchronize");
    }
    return f_evCreate && f_evRecord && f_evElapsed && f_evDestroy && f_evSync;
}

/* Record a GPU-accurate span: create events before and after the call. */
#define GPU_SPAN_BEGIN(stream)                          \
    cudaEvent_t _ev_s = NULL, _ev_e = NULL;             \
    int _gpu_ok = ev_ok() &&                            \
        f_evCreate(&_ev_s) == 0 &&                      \
        f_evCreate(&_ev_e) == 0 &&                      \
        f_evRecord(_ev_s, (stream)) == 0;               \
    uint64_t _t0 = now_ns();

#define GPU_SPAN_END(cat, name, extra, stream)                          \
    if (_gpu_ok) {                                                       \
        f_evRecord(_ev_e, (stream));                                     \
        float _ms = 0.0f;                                                \
        if (f_evSync(_ev_e) == 0 &&                                      \
            f_evElapsed(&_ms, _ev_s, _ev_e) == 0 && _ms >= 0.0f)        \
            emit_span((cat), gettid_compat(), _t0,                       \
                      (uint64_t)(_ms * 1e6f), (name), (extra));          \
        f_evDestroy(_ev_s); f_evDestroy(_ev_e);                          \
    } else {                                                             \
        emit_span((cat), gettid_compat(), _t0, now_ns()-_t0,            \
                  (name), (extra));                                       \
        if (_ev_s) f_evDestroy(_ev_s);                                   \
        if (_ev_e) f_evDestroy(_ev_e);                                   \
    }

/* ── NCCL stream ID (simple index) ───────────────────────────────────── */
#define SMAP_CAP 512
static void *_sptrs[SMAP_CAP]; static int _sids[SMAP_CAP], _sn = 0;
static pthread_mutex_t _smtx = PTHREAD_MUTEX_INITIALIZER;
static int stream_id(cudaStream_t s) {
    if (!s) return 0;
    pthread_mutex_lock(&_smtx);
    for (int i = 0; i < _sn; i++) {
        if (_sptrs[i] == s) { int id = _sids[i]; pthread_mutex_unlock(&_smtx); return id; }
    }
    if (_sn >= SMAP_CAP) { pthread_mutex_unlock(&_smtx); return -1; }
    int id = ++_sn;
    _sptrs[_sn - 1] = s;
    _sids[_sn - 1]  = id;
    pthread_mutex_unlock(&_smtx);
    return id;
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
    char extra[128]; snprintf(extra, sizeof(extra),
        "type=allreduce,bytes=%zu,stream=%d", nb, stream_id(stream));
    GPU_SPAN_BEGIN(stream)
    ncclResult_t ret = real(sb, rb, count, dt, op, comm, stream);
    GPU_SPAN_END("nccl", "ncclAllReduce", extra, stream)
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
    char extra[128]; snprintf(extra, sizeof(extra),
        "type=broadcast,bytes=%zu,root=%d,stream=%d", nb, root, stream_id(stream));
    GPU_SPAN_BEGIN(stream)
    ncclResult_t ret = real(sb, rb, count, dt, root, comm, stream);
    GPU_SPAN_END("nccl", "ncclBroadcast", extra, stream)
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
    char extra[128]; snprintf(extra, sizeof(extra),
        "type=reduce,bytes=%zu,root=%d,stream=%d", nb, root, stream_id(stream));
    GPU_SPAN_BEGIN(stream)
    ncclResult_t ret = real(sb, rb, count, dt, op, root, comm, stream);
    GPU_SPAN_END("nccl", "ncclReduce", extra, stream)
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
    char extra[128]; snprintf(extra, sizeof(extra),
        "type=allgather,bytes=%zu,stream=%d", nb, stream_id(stream));
    GPU_SPAN_BEGIN(stream)
    ncclResult_t ret = real(sb, rb, sendcount, dt, comm, stream);
    GPU_SPAN_END("nccl", "ncclAllGather", extra, stream)
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
    char extra[128]; snprintf(extra, sizeof(extra),
        "type=reduce_scatter,bytes=%zu,stream=%d", nb, stream_id(stream));
    GPU_SPAN_BEGIN(stream)
    ncclResult_t ret = real(sb, rb, recvcount, dt, op, comm, stream);
    GPU_SPAN_END("nccl", "ncclReduceScatter", extra, stream)
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
    char extra[128]; snprintf(extra, sizeof(extra),
        "type=send,bytes=%zu,peer=%d,stream=%d", nb, peer, stream_id(stream));
    GPU_SPAN_BEGIN(stream)
    ncclResult_t ret = real(sb, count, dt, peer, comm, stream);
    GPU_SPAN_END("nccl", "ncclSend", extra, stream)
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
    char extra[128]; snprintf(extra, sizeof(extra),
        "type=recv,bytes=%zu,peer=%d,stream=%d", nb, peer, stream_id(stream));
    GPU_SPAN_BEGIN(stream)
    ncclResult_t ret = real(rb, count, dt, peer, comm, stream);
    GPU_SPAN_END("nccl", "ncclRecv", extra, stream)
    return ret;
}

/* ── Group boundaries ───────────────────────────────────────────────── */
static __thread uint64_t _group_start = 0;
static __thread int      _group_depth = 0;

ncclResult_t ncclGroupStart(void) {
    typedef ncclResult_t (*fn_t)(void);
    static fn_t real = NULL;
    if (!real) real = (fn_t)dlsym(RTLD_NEXT, "ncclGroupStart");
    if (!real) return -1;
    if (_group_depth++ == 0) _group_start = now_ns();
    return real();
}

ncclResult_t ncclGroupEnd(void) {
    typedef ncclResult_t (*fn_t)(void);
    static fn_t real = NULL;
    if (!real) real = (fn_t)dlsym(RTLD_NEXT, "ncclGroupEnd");
    if (!real) return -1;
    ncclResult_t ret = real();
    if (--_group_depth == 0 && _group_start)
        emit_span("nccl", gettid_compat(), _group_start,
                  now_ns() - _group_start, "ncclGroup", "type=group");
    return ret;
}

/* ── Constructor ─────────────────────────────────────────────────────── */
__attribute__((constructor))
static void hprofiler_nccl_init(void) { ensure_connected(); }

__attribute__((destructor))
static void hprofiler_nccl_fini(void) {
    if (g_sock >= 0) { close(g_sock); g_sock = -1; }
}
