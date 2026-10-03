/*
 * NCCL hook (LD_PRELOAD; no NCCL headers needed, symbols via dlsym).
 *
 * Wraps ncclAllReduce, Broadcast, Reduce, AllGather, ReduceScatter,
 * AllToAll, Send, Recv (category "nccl"; tags type=, bytes= (count x dtype
 * size), stream=, rank=, nranks=), GroupStart/GroupEnd (one ncclGroup span
 * for the outermost pair) and CommInitRank/InitAll/Destroy.
 *
 * Timing: a cudaEvent pair on the operation's stream; the wrapper waits for
 * the end event before returning, which serializes each operation with the
 * host. Without usable CUDA events: host wall-clock time, tagged timing=cpu.
 * If the end-event wait or elapsed-time query fails, no span is emitted.
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

static pid_t gettid_compat(void) { return hp_tx_tid(); }

static void emit_span(const char *cat, pid_t tid, uint64_t start_ns,
                      uint64_t dur_ns, const char *name, const char *extra) {
    if (!hp_tx_enabled()) return;
    hp_tx_emitf("span:%s:%d:%d:%llu:%llu:%s:%s\n", cat, (int)hp_tx_pid(), (int)tid,
                (unsigned long long)start_ns, (unsigned long long)dur_ns,
                name, extra ? extra : "");
    emit_callstack(start_ns);
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

/* Event-pair timing. _t0 is taken before the start event is recorded, so the
 * host timestamp never follows the GPU start. */
#define GPU_SPAN_BEGIN(stream)                          \
    uint64_t _t0 = now_ns();                            \
    cudaEvent_t _ev_s = NULL, _ev_e = NULL;             \
    int _gpu_ok = ev_ok() &&                            \
        f_evCreate(&_ev_s) == 0 &&                      \
        f_evCreate(&_ev_e) == 0 &&                      \
        f_evRecord(_ev_s, (stream)) == 0;

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
        mark_cpu_fallback((extra), sizeof(extra));                       \
        emit_span((cat), gettid_compat(), _t0, now_ns()-_t0,            \
                  (name), (extra));                                       \
        if (_ev_s) f_evDestroy(_ev_s);                                   \
        if (_ev_e) f_evDestroy(_ev_e);                                   \
    }

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
    char extra[192]; snprintf(extra, sizeof(extra),
        "type=allreduce,bytes=%zu,stream=%d,rank=%d,nranks=%d",
        nb, stream_id(stream), rank, nranks);
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
    int rank = -1, nranks = -1; comm_meta(comm, &rank, &nranks);
    char extra[192]; snprintf(extra, sizeof(extra),
        "type=broadcast,bytes=%zu,root=%d,stream=%d,rank=%d,nranks=%d",
        nb, root, stream_id(stream), rank, nranks);
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
    int rank = -1, nranks = -1; comm_meta(comm, &rank, &nranks);
    char extra[192]; snprintf(extra, sizeof(extra),
        "type=reduce,bytes=%zu,root=%d,stream=%d,rank=%d,nranks=%d",
        nb, root, stream_id(stream), rank, nranks);
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
    int rank = -1, nranks = -1; comm_meta(comm, &rank, &nranks);
    char extra[192]; snprintf(extra, sizeof(extra),
        "type=allgather,bytes=%zu,stream=%d,rank=%d,nranks=%d",
        nb, stream_id(stream), rank, nranks);
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
    int rank = -1, nranks = -1; comm_meta(comm, &rank, &nranks);
    char extra[192]; snprintf(extra, sizeof(extra),
        "type=reduce_scatter,bytes=%zu,stream=%d,rank=%d,nranks=%d",
        nb, stream_id(stream), rank, nranks);
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
    int rank = -1, nranks = -1; comm_meta(comm, &rank, &nranks);
    char extra[192]; snprintf(extra, sizeof(extra),
        "type=send,bytes=%zu,peer=%d,stream=%d,rank=%d,nranks=%d",
        nb, peer, stream_id(stream), rank, nranks);
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
    int rank = -1, nranks = -1; comm_meta(comm, &rank, &nranks);
    char extra[192]; snprintf(extra, sizeof(extra),
        "type=recv,bytes=%zu,peer=%d,stream=%d,rank=%d,nranks=%d",
        nb, peer, stream_id(stream), rank, nranks);
    GPU_SPAN_BEGIN(stream)
    ncclResult_t ret = real(rb, count, dt, peer, comm, stream);
    GPU_SPAN_END("nccl", "ncclRecv", extra, stream)
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
    char extra[192]; snprintf(extra, sizeof(extra),
        "type=alltoall,bytes=%zu,stream=%d,rank=%d,nranks=%d",
        nb, stream_id(stream), rank, nranks);
    GPU_SPAN_BEGIN(stream)
    ncclResult_t ret = real(sb, rb, count, dt, comm, stream);
    GPU_SPAN_END("nccl", "ncclAllToAll", extra, stream)
    return ret;
}

/* ── Group boundaries ───────────────────────────────────────────────── */
/* Unverified: NCCL defers the kernels of operations issued inside
 * ncclGroupStart/End until ncclGroupEnd, so an event pair around such an
 * operation may not bracket its real GPU work. Follows from NCCL's documented
 * group semantics; not checked on multi-GPU hardware. */
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
static void hprofiler_nccl_init(void) { hp_tx_init("nccl"); cs_init(); }

__attribute__((destructor))
static void hprofiler_nccl_fini(void) { hp_tx_shutdown(1); }
