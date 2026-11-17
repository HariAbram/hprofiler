/*
 * Native CUDA device-activity tracing through CUPTI (Activity + Callback
 * APIs), loaded at run time with dlopen -- libhprofiler_cuda.so never links
 * against libcupti, so it still loads and runs (interception-only, "proxy"
 * device timing) where CUPTI is missing.
 *
 * What it produces (wire `span:` lines, see src/core/gpu_activity.py):
 *   kernel   CONCURRENT_KERNEL records  -> category cuda,   type=kernel
 *   memcpy   MEMCPY / MEMCPY2 records   -> category memory, type=memcpy
 *   memset   MEMSET records             -> category memory, type=memset
 *   sync     SYNCHRONIZATION records    -> category sync,   type=sync_wait
 * all tagged side=gpu,rt=cuda,src=cupti,timing=device (sync records:
 * timing=host -- CUPTI timestamps the host-side wait) plus corr/dev/ctx/
 * nstream, and status lines `gpuact:<pid>:cupti:...` (status, clock
 * mapping, dropped records).
 *
 * Correlation: the Callback API reports the correlation id of every
 * runtime/driver API call at entry. cuda_hook.c arms a thread-local capture
 * around each intercepted call (hp_cupti_arm/disarm) and puts the captured
 * ids on its host span, so the Python side can match device records to the
 * exact host call. Kernel launches, copies/memsets, event synchronizations
 * and completed event queries made through calls the LD_PRELOAD layer
 * cannot see (the driver API, libraries with a statically linked CUDA
 * runtime) get a host span from the callback itself (src=cupti_cb), so
 * their device records are still correlated.
 *
 * Clock: CUPTI timestamps default to CLOCK_REALTIME. When available,
 * cuptiActivityRegisterTimestampCallback makes CUPTI use our CLOCK_MONOTONIC
 * directly (clock=monotonic_callback, exact). Otherwise each buffer's
 * records are shifted by an offset measured by bracketing cuptiGetTimestamp
 * between two CLOCK_MONOTONIC reads (clock=offset, error <= half the
 * tightest bracket, reported as clock_err_ns; a wall-clock step while a
 * buffer fills shifts that buffer's records by the step).
 *
 * Record layouts: compiled against the toolkit's own cupti_activity.h
 * (Kernel9 / Memcpy6 or 5 / Memset4 / MemcpyPtoP4 / Synchronization2 or 1,
 * whichever the headers document as current). CMake only
 * defines HP_HAVE_CUPTI after test-compiling exactly these types, so an
 * unfamiliar toolkit disables native tracing instead of breaking the build.
 * At run time a libcupti OLDER than the headers is refused (it could
 * deliver older, shorter record versions); a newer one is accepted on the
 * basis that CUPTI extends records by appending fields -- records with
 * implausible timestamps are counted (bad_records) and skipped rather than
 * emitted. Checked functionally on a GeForce MX550 (CUDA 12.9 CUPTI).
 */
#define _GNU_SOURCE
#include "hp_cupti.h"
#include "../common/hp_ids.h"

#include <dlfcn.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#if defined(HP_HAVE_CUPTI) && !defined(HP_NO_CUPTI)
#include <cupti.h>
#define HP_CUPTI_ENABLED 1
/* The record version CUPTI delivers follows the toolkit; CMake picks the
 * newest these headers define (Memcpy6 = Memcpy5 + copyCount,
 * Synchronization2 = Synchronization + cudaEventSyncId/returnValue: every
 * field read here is at the same offset in both). */
#ifdef HP_CUPTI_MEMCPY6
typedef CUpti_ActivityMemcpy6 hp_memcpy_rec_t;
#else
typedef CUpti_ActivityMemcpy5 hp_memcpy_rec_t;
#endif
#ifdef HP_CUPTI_SYNC2
typedef CUpti_ActivitySynchronization2 hp_sync_rec_t;
#else
typedef CUpti_ActivitySynchronization hp_sync_rec_t;
#endif
#else
#define HP_CUPTI_ENABLED 0
#endif

__attribute__((unused)) static uint64_t mono_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static void emit_status(const char *fields) {
    char line[512];
    int n = snprintf(line, sizeof(line), "gpuact:%d:cupti:%s\n", (int)getpid(), fields);
    if (n > 0 && n < (int)sizeof(line))
        hp_cuda_emit_line(line);
}

int hp_cupti_compiled(void) { return HP_CUPTI_ENABLED; }

#if !HP_CUPTI_ENABLED
/* ── built without usable CUPTI headers ──────────────────────────────────── */

unsigned hp_cupti_start(unsigned want) {
    static int reported = 0;
    if (want && !__atomic_exchange_n(&reported, 1, __ATOMIC_RELAXED))
        emit_status("status=unavailable,reason=built_without_cupti_headers");
    return 0;
}
void hp_cupti_arm(void) {}
void hp_cupti_disarm(uint32_t *corr, uint32_t *corr2) { *corr = 0; *corr2 = 0; }
void hp_cupti_flush(void) {}
void hp_cupti_final_flush(void) {}
int hp_cupti_streams_tracked(void) { return 0; }
uint32_t hp_cupti_graph_exec_id(const void *exec) { (void)exec; return 0; }
uint32_t hp_cupti_graph_exec_version(uint32_t exec_id) { (void)exec_id; return 0; }

#else
/* ── CUPTI available at build time ───────────────────────────────────────── */

/* Function pointers resolved from libcupti with dlsym. Required ones use
 * the header's own prototypes; optional (newer) ones get local typedefs. */
static struct {
    __typeof__(&cuptiActivityRegisterCallbacks) RegisterCallbacks;
    __typeof__(&cuptiActivityEnable)            Enable;
    __typeof__(&cuptiActivityFlushAll)          FlushAll;
    __typeof__(&cuptiActivityGetNextRecord)     GetNextRecord;
    __typeof__(&cuptiActivityGetNumDroppedRecords) GetNumDropped;
    __typeof__(&cuptiGetTimestamp)              GetTimestamp;
    __typeof__(&cuptiGetVersion)                GetVersion;
    __typeof__(&cuptiSubscribe)                 Subscribe;
    __typeof__(&cuptiEnableDomain)              EnableDomain;
    CUptiResult (CUPTIAPI *RegisterTimestampCallback)(uint64_t (CUPTIAPI *)(void));
    CUptiResult (CUPTIAPI *FlushPeriod)(uint32_t);
    CUptiResult (CUPTIAPI *EnableLatencyTimestamps)(uint8_t);
    /* graph and stream ids (CUDA 11.x / 12.x CUPTI; absent: not reported) */
    CUptiResult (CUPTIAPI *GetGraphId)(CUgraph, uint32_t *);
    CUptiResult (CUPTIAPI *GetGraphNodeId)(CUgraphNode, uint64_t *);
    CUptiResult (CUPTIAPI *GetGraphExecId)(CUgraphExec, uint32_t *);
    CUptiResult (CUPTIAPI *GetStreamIdEx)(CUcontext, CUstream, uint8_t, uint32_t *);
    CUptiResult (CUPTIAPI *GetContextId)(CUcontext, uint32_t *);
} p;
static int g_res_ok = 0;                /* resource callbacks enabled (stream lives, graphs) */

static pthread_once_t  g_lib_once  = PTHREAD_ONCE_INIT;
static pthread_mutex_t g_start_mu  = PTHREAD_MUTEX_INITIALIZER;
static int      g_lib_ok        = 0;
static char     g_lib_reason[64] = "";
static unsigned g_active        = 0;
static uint32_t g_lib_version   = 0;
static int      g_callbacks_ok  = 0;
static int      g_latency       = 0;
static int      g_exit_flushed  = 0;
static int      g_pc_enabled    = 0;

/* clock mapping */
static int      g_clock_cb      = 0;   /* 1: CUPTI calls mono_ts_cb */
static int64_t  g_offset_ns     = 0;   /* mono = cupti + offset (offset mode) */
static uint64_t g_offset_err_ns = 0;

/* counters reported as deltas in gpuact lines */
static uint64_t g_notime = 0, g_bad = 0, g_alloc_fail = 0, g_internal = 0;

/* Correlation ids of CUDA calls the hook makes itself (proxy event timing
 * in HPROFILER_DEVICE_ACTIVITY=both mode). CUPTI reports synchronization
 * records for those too; they are not application work. A ring is enough:
 * the records arrive within a flush period of the call. */
#define HP_INTERNAL_RING 8192
static uint32_t g_internal_ring[HP_INTERNAL_RING];
static uint32_t g_internal_pos = 0;

static void note_internal(uint32_t corr) {
    uint32_t i = __atomic_fetch_add(&g_internal_pos, 1, __ATOMIC_RELAXED);
    __atomic_store_n(&g_internal_ring[i % HP_INTERNAL_RING], corr, __ATOMIC_RELAXED);
}

static int is_internal(uint32_t corr) {
    if (!corr || !__atomic_load_n(&g_internal_pos, __ATOMIC_RELAXED)) return 0;
    for (int i = 0; i < HP_INTERNAL_RING; i++)
        if (__atomic_load_n(&g_internal_ring[i], __ATOMIC_RELAXED) == corr) return 1;
    return 0;
}

static uint64_t CUPTIAPI mono_ts_cb(void) { return mono_ns(); }

static void calibrate_offset(void) {
    if (g_clock_cb || !p.GetTimestamp) return;
    uint64_t best = UINT64_MAX;
    int64_t off = g_offset_ns;
    for (int i = 0; i < 5; i++) {
        uint64_t a = mono_ns(), c = 0;
        if (p.GetTimestamp(&c) != CUPTI_SUCCESS) return;
        uint64_t b = mono_ns();
        if (b - a < best) {
            best = b - a;
            off = (int64_t)(a + (b - a) / 2) - (int64_t)c;
        }
    }
    g_offset_ns = off;
    g_offset_err_ns = best / 2 + 1;
}

static uint64_t to_mono(uint64_t cupti_ts) {
    return g_clock_cb ? cupti_ts : (uint64_t)((int64_t)cupti_ts + g_offset_ns);
}

/* ── record -> wire ──────────────────────────────────────────────────────── */

/* 1 hour: no single kernel/copy runs longer; anything beyond is a layout
 * mismatch or an unfinished record, not a measurement. */
#define HP_MAX_PLAUSIBLE_NS 3600000000000ULL

static int plausible(uint64_t start, uint64_t end) {
    if (start == 0 && end == 0) { __atomic_add_fetch(&g_notime, 1, __ATOMIC_RELAXED); return 0; }
    if (end < start || end - start > HP_MAX_PLAUSIBLE_NS) {
        __atomic_add_fetch(&g_bad, 1, __ATOMIC_RELAXED);
        return 0;
    }
    return 1;
}

static const char *copy_dir(uint8_t k) {
    switch (k) {
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD: return "HtoD";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH: return "DtoH";
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOA: return "HtoA";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOH: return "AtoH";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOA: return "AtoA";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOD: return "AtoD";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOA: return "DtoA";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD: return "DtoD";
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOH: return "HtoH";
    case CUPTI_ACTIVITY_MEMCPY_KIND_PTOP: return "PtoP";
    default:                              return "unknown";
    }
}

#define APPEND(buf, n, ...) do { \
    if ((n) >= 0 && (size_t)(n) < sizeof(buf)) \
        (n) += snprintf((buf) + (n), sizeof(buf) - (size_t)(n), __VA_ARGS__); \
} while (0)

static void emit_kernel(const CUpti_ActivityKernel9 *k) {
    if (!plausible(k->start, k->end)) return;
    uint64_t s = to_mono(k->start), e = to_mono(k->end);
    char x[512];
    int n = 0;
    APPEND(x, n, "type=kernel,side=gpu,rt=cuda,op=kernel,timing=device,src=cupti,"
                 "corr=%u,dev=%u,ctx=%u,nstream=%u,grid=%dx%dx%d,block=%dx%dx%d",
           k->correlationId, k->deviceId, k->contextId, k->streamId,
           k->gridX, k->gridY, k->gridZ, k->blockX, k->blockY, k->blockZ);
    if (g_latency && k->queued != CUPTI_TIMESTAMP_UNKNOWN &&
        k->submitted != CUPTI_TIMESTAMP_UNKNOWN && k->submitted <= k->start)
        APPEND(x, n, ",queued=%llu,submitted=%llu",
               (unsigned long long)to_mono(k->queued), (unsigned long long)to_mono(k->submitted));
    if (k->graphId)
        APPEND(x, n, ",graph=%u,gnode=%llu", k->graphId, (unsigned long long)k->graphNodeId);
    hp_cuda_emit_span("cuda", 0, s, e - s, k->name ? k->name : "<unnamed kernel>", x);
}

static void emit_memcpy(const hp_memcpy_rec_t *m) {
    if (!plausible(m->start, m->end)) return;
    uint64_t s = to_mono(m->start), e = to_mono(m->end);
    const char *dir = copy_dir(m->copyKind);
    char name[32], x[512];
    snprintf(name, sizeof(name), "memcpy %s", dir);
    int n = 0;
    /* correlationId is the driver-API call's id; runtimeCorrelationId the
     * runtime call's (0 when the copy came from the driver API). */
    APPEND(x, n, "type=memcpy,side=gpu,rt=cuda,op=memcpy,timing=device,src=cupti,"
                 "dir=%s,bytes=%llu,corr=%u,dev=%u,ctx=%u,nstream=%u",
           dir, (unsigned long long)m->bytes, m->correlationId,
           m->deviceId, m->contextId, m->streamId);
    if (m->runtimeCorrelationId && m->runtimeCorrelationId != m->correlationId)
        APPEND(x, n, ",corr2=%u", m->runtimeCorrelationId);
    if (m->flags & CUPTI_ACTIVITY_FLAG_MEMCPY_ASYNC)
        APPEND(x, n, ",async=1");
#ifdef HP_CUPTI_MEMCPY6
    if (m->graphId)
        APPEND(x, n, ",graph=%u,gnode=%llu", m->graphId, (unsigned long long)m->graphNodeId);
#endif
    hp_cuda_emit_span("memory", 0, s, e - s, name, x);
}

static void emit_p2p(const CUpti_ActivityMemcpyPtoP4 *m) {
    if (!plausible(m->start, m->end)) return;
    uint64_t s = to_mono(m->start), e = to_mono(m->end);
    char x[512];
    int n = 0;
    APPEND(x, n, "type=memcpy,side=gpu,rt=cuda,op=memcpy,timing=device,src=cupti,"
                 "dir=PtoP,bytes=%llu,corr=%u,dev=%u,ctx=%u,nstream=%u,src_dev=%u,dst_dev=%u",
           (unsigned long long)m->bytes, m->correlationId, m->deviceId,
           m->contextId, m->streamId, m->srcDeviceId, m->dstDeviceId);
    hp_cuda_emit_span("memory", 0, s, e - s, "memcpy PtoP", x);
}

static void emit_memset(const CUpti_ActivityMemset4 *m) {
    if (!plausible(m->start, m->end)) return;
    uint64_t s = to_mono(m->start), e = to_mono(m->end);
    char x[384];
    int n = 0;
    APPEND(x, n, "type=memset,side=gpu,rt=cuda,op=memset,timing=device,src=cupti,"
                 "bytes=%llu,corr=%u,dev=%u,ctx=%u,nstream=%u",
           (unsigned long long)m->bytes, m->correlationId,
           m->deviceId, m->contextId, m->streamId);
    if (m->graphId)
        APPEND(x, n, ",graph=%u,gnode=%llu", m->graphId, (unsigned long long)m->graphNodeId);
    hp_cuda_emit_span("memory", 0, s, e - s, "memset", x);
}

static void emit_sync(const hp_sync_rec_t *y) {
    if (is_internal(y->correlationId)) {
        __atomic_add_fetch(&g_internal, 1, __ATOMIC_RELAXED);
        return;
    }
    if (!plausible(y->start, y->end)) return;
    uint64_t s = to_mono(y->start), e = to_mono(y->end);
    const char *kind;
    switch (y->type) {
    case CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_EVENT_SYNCHRONIZE:   kind = "event"; break;
    case CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_STREAM_WAIT_EVENT:   kind = "stream_wait"; break;
    case CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_STREAM_SYNCHRONIZE:  kind = "stream"; break;
    case CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_CONTEXT_SYNCHRONIZE: kind = "context"; break;
    default:                                                      kind = "unknown"; break;
    }
    char name[32], x[384];
    snprintf(name, sizeof(name), "sync %s", kind);
    int n = 0;
    /* timing=host: CUPTI timestamps the host thread's wait, not GPU work. */
    APPEND(x, n, "type=sync_wait,side=gpu,rt=cuda,op=sync,timing=host,src=cupti,"
                 "sync=%s,corr=%u,ctx=%u", kind, y->correlationId, y->contextId);
    if (y->streamId != (uint32_t)CUPTI_SYNCHRONIZATION_INVALID_VALUE)
        APPEND(x, n, ",nstream=%u", y->streamId);
    if (y->cudaEventId != (uint32_t)CUPTI_SYNCHRONIZATION_INVALID_VALUE)
        APPEND(x, n, ",cevent=%u", y->cudaEventId);
    hp_cuda_emit_span("sync", 0, s, e - s, name, x);
}

static void handle_record(const CUpti_Activity *r) {
    switch (r->kind) {
    case CUPTI_ACTIVITY_KIND_KERNEL:
    case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
        emit_kernel((const CUpti_ActivityKernel9 *)r); break;
    case CUPTI_ACTIVITY_KIND_MEMCPY:
        emit_memcpy((const hp_memcpy_rec_t *)r); break;
    case CUPTI_ACTIVITY_KIND_MEMCPY2:
        emit_p2p((const CUpti_ActivityMemcpyPtoP4 *)r); break;
    case CUPTI_ACTIVITY_KIND_MEMSET:
        emit_memset((const CUpti_ActivityMemset4 *)r); break;
    case CUPTI_ACTIVITY_KIND_SYNCHRONIZATION:
        emit_sync((const hp_sync_rec_t *)r); break;
    case CUPTI_ACTIVITY_KIND_PC_SAMPLING:
    case CUPTI_ACTIVITY_KIND_FUNCTION:
        if (g_pc_enabled) hp_cuda_pc_record(r, (uint32_t)r->kind);
        break;
    default:
        break;
    }
}

static void report_counters(uint64_t dropped) {
    static uint64_t last_notime = 0, last_bad = 0, last_alloc = 0, last_internal = 0;
    uint64_t nt = __atomic_load_n(&g_notime, __ATOMIC_RELAXED);
    uint64_t bd = __atomic_load_n(&g_bad, __ATOMIC_RELAXED);
    uint64_t af = __atomic_load_n(&g_alloc_fail, __ATOMIC_RELAXED);
    uint64_t in = __atomic_load_n(&g_internal, __ATOMIC_RELAXED);
    char f[256];
    int n = 0;
    f[0] = '\0';
    if (dropped)          APPEND(f, n, "%sdropped=%llu", n ? "," : "", (unsigned long long)dropped);
    if (nt > last_notime) APPEND(f, n, "%snotime=%llu", n ? "," : "", (unsigned long long)(nt - last_notime));
    if (bd > last_bad)    APPEND(f, n, "%sbad_records=%llu", n ? "," : "", (unsigned long long)(bd - last_bad));
    if (af > last_alloc)  APPEND(f, n, "%sbuffer_alloc_failed=%llu", n ? "," : "", (unsigned long long)(af - last_alloc));
    if (in > last_internal) APPEND(f, n, "%sinternal_records=%llu", n ? "," : "", (unsigned long long)(in - last_internal));
    last_notime = nt; last_bad = bd; last_alloc = af; last_internal = in;
    if (n > 0) emit_status(f);
}

/* ── buffers ─────────────────────────────────────────────────────────────── */

#define HP_CUPTI_BUF_BYTES (8u << 20)

static void CUPTIAPI buf_requested(uint8_t **buf, size_t *size, size_t *max_records) {
    void *b = NULL;
    if (posix_memalign(&b, 8, HP_CUPTI_BUF_BYTES) != 0) b = NULL;
    *buf = (uint8_t *)b;
    *size = b ? HP_CUPTI_BUF_BYTES : 0;   /* 0: CUPTI drops and counts records */
    *max_records = 0;
    if (!b) __atomic_add_fetch(&g_alloc_fail, 1, __ATOMIC_RELAXED);
}

static void CUPTIAPI buf_completed(CUcontext ctx, uint32_t stream_id, uint8_t *buf,
                                   size_t size, size_t valid) {
    (void)size;
    int saved = hp_cuda_in_hook;
    hp_cuda_in_hook = 1;                  /* never trace our own work here */
    calibrate_offset();
    if (buf) {
        CUpti_Activity *rec = NULL;
        while (p.GetNextRecord(buf, valid, &rec) == CUPTI_SUCCESS)
            handle_record(rec);
    }
    size_t dropped = 0;
    if (p.GetNumDropped(ctx, stream_id, &dropped) != CUPTI_SUCCESS)
        dropped = 0;
    report_counters(dropped);
    free(buf);
    hp_cuda_in_hook = saved;
}

/* ── Resource callbacks: stream lives and graph structure ────────────────
 * With the resource domain enabled CUPTI reports every stream's creation
 * and destruction and every graph exec's instantiation, whatever API or
 * library made them (also statically linked runtimes):
 *   STREAM_CREATED     begins a new life of the handle (cuda_hook.c's
 *                      stream= numbers) and emits a zero-length stream_info
 *                      span: stream=, nstream= (CUPTI's id, the one device
 *                      records carry), ctx=, nonblocking=, priority=
 *   STREAM_DESTROY_STARTING  ends the life: the next stream at that address
 *                      gets another number
 *   GRAPHNODE_CLONED   pairs an exec's node with the graph node it copies
 *                      (kept per thread until the exec is created)
 *   GRAPHEXEC_CREATED  the exec's structure, once: see graph_snapshot()
 *   GRAPHEXEC_DESTROY_STARTING  forgets the exec's version.
 * Numeric callback ids (cupti_callbacks.h, stable); the graph data layout
 * is CUpti_GraphData's. */
enum {
    HP_RES_STREAM_CREATED = 3, HP_RES_STREAM_DESTROY_STARTING = 4, HP_RES_GRAPHEXEC_CREATED = 18,
    HP_RES_GRAPHEXEC_DESTROY_STARTING = 19, HP_RES_GRAPHNODE_CLONED = 20,
};
typedef struct {
    CUgraph graph, originalGraph;
    CUgraphNode node, originalNode;
    int nodeType;
    CUgraphNode dependency;
    CUgraphExec graphExec;
} hp_graph_data_t;

int hp_cupti_streams_tracked(void) { return __atomic_load_n(&g_res_ok, __ATOMIC_ACQUIRE); }

/* Driver calls made from the callbacks, resolved from the process's
 * libcuda (also when a static runtime opened it with RTLD_LOCAL). */
static struct {
    CUresult (*GetNodes)(CUgraph, CUgraphNode *, size_t *);
    CUresult (*GetEdges)(CUgraph, CUgraphNode *, CUgraphNode *, size_t *);
    CUresult (*GetEdges2)(CUgraph, CUgraphNode *, CUgraphNode *, void *, size_t *);
    CUresult (*NodeGetType)(CUgraphNode, int *);
    CUresult (*ChildGraph)(CUgraphNode, CUgraph *);
    CUresult (*StreamGetFlags)(CUstream, unsigned int *);
    CUresult (*StreamGetPriority)(CUstream, int *);
} drv;
static pthread_once_t g_drv_once = PTHREAD_ONCE_INIT;

static void drv_init(void) {
    void *h = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_NOLOAD);
    if (!h) return;
#define DSYM(f, n) (drv.f = (__typeof__(drv.f))dlsym(h, n))
    DSYM(GetNodes, "cuGraphGetNodes");
    DSYM(GetEdges, "cuGraphGetEdges");
    DSYM(GetEdges2, "cuGraphGetEdges_v2");
    DSYM(NodeGetType, "cuGraphNodeGetType");
    DSYM(ChildGraph, "cuGraphChildGraphNodeGetGraph");
    DSYM(StreamGetFlags, "cuStreamGetFlags");
    DSYM(StreamGetPriority, "cuStreamGetPriority");
#undef DSYM
}

static void emit_instant(const char *name, const char *tags) {
    char head[160];
    int hn = snprintf(head, sizeof(head), "inst:cuda:%d:%d:%llu:%s:", (int)getpid(), (int)syscall(SYS_gettid),
                      (unsigned long long)mono_ns(), name);
    size_t tl = strlen(tags);
    char *line = (char *)malloc((size_t)hn + tl + 2);
    if (!line) return;
    memcpy(line, head, (size_t)hn);
    memcpy(line + hn, tags, tl);
    line[hn + tl] = '\n';
    line[hn + tl + 1] = '\0';
    hp_cuda_emit_line(line);
    free(line);
}

static void stream_created(const CUpti_ResourceData *rd) {
    CUstream st = rd->resourceHandle.stream;
    int id = hp_cuda_stream_born((const void *)st);
    pthread_once(&g_drv_once, drv_init);
    int saved = hp_cuda_in_hook;
    hp_cuda_in_hook = 1;                      /* the queries below are not the program's */
    unsigned flags = 0;
    int prio = 0, okf = drv.StreamGetFlags && drv.StreamGetFlags(st, &flags) == CUDA_SUCCESS;
    int okp = drv.StreamGetPriority && drv.StreamGetPriority(st, &prio) == CUDA_SUCCESS;
    uint32_t nstream = 0, ctx = 0;
    int oks = p.GetStreamIdEx && p.GetStreamIdEx(rd->context, st, 0, &nstream) == CUPTI_SUCCESS;
    int okc = p.GetContextId && rd->context && p.GetContextId(rd->context, &ctx) == CUPTI_SUCCESS;
    hp_cuda_in_hook = saved;
    char x[256];
    int n = 0;
    APPEND(x, n, "type=stream_info,op=stream_info,side=cpu,rt=cuda,timing=host,src=cupti_res,stream=%d", id);
    if (okf) APPEND(x, n, ",nonblocking=%u", flags & 1u);
    if (okp) APPEND(x, n, ",priority=%d", prio);
    if (oks) APPEND(x, n, ",nstream=%u", nstream);
    if (okc) APPEND(x, n, ",ctx=%u", ctx);
    hp_cuda_emit_span("cuda", (pid_t)syscall(SYS_gettid), mono_ns(), 0, "stream_info", x);
}

/* ── Graph versions ──────────────────────────────────────────────────────
 * An exec's version: 1 at instantiation, +1 per successful driver call that
 * changes what its nodes do (cuGraphExecUpdate*, cuGraphExec*SetParams*,
 * cuGraphNodeSetEnabled, cuGraphExecNodeSetParams). Graph launches carry
 * gexec= (CUPTI's exec id) and gver=; each change is an instant
 * graph_exec_update. GEXEC_CAP execs with a version above 1 (8 B each);
 * beyond, a launch reports the version as 0 (unknown). An exec's topology
 * cannot change after instantiation (cuGraphExecUpdate refuses a different
 * one), so its structure is recorded once. */
#define GEXEC_CAP 4096
typedef struct { uint32_t id, ver; } GexecVer;
static GexecVer        g_gexec[GEXEC_CAP];
static int             g_ngexec = 0, g_gexec_full = 0;
static pthread_mutex_t g_gexec_mu = PTHREAD_MUTEX_INITIALIZER;

uint32_t hp_cupti_graph_exec_id(const void *exec) {
    uint32_t id = 0;
    if (!exec || !p.GetGraphExecId) return 0;
    int saved = hp_cuda_in_hook;
    hp_cuda_in_hook = 1;
    CUptiResult r = p.GetGraphExecId((CUgraphExec)exec, &id);
    hp_cuda_in_hook = saved;
    return r == CUPTI_SUCCESS ? id : 0;
}

uint32_t hp_cupti_graph_exec_version(uint32_t exec_id) {
    if (!exec_id) return 0;
    uint32_t v = 1;
    pthread_mutex_lock(&g_gexec_mu);
    for (int i = 0; i < g_ngexec; i++)
        if (g_gexec[i].id == exec_id) { v = g_gexec[i].ver; break; }
    if (v == 1 && g_gexec_full) v = 0;            /* it may be one that did not fit */
    pthread_mutex_unlock(&g_gexec_mu);
    return v;
}

static uint32_t gexec_bump(uint32_t exec_id) {
    uint32_t v = 0;
    pthread_mutex_lock(&g_gexec_mu);
    int i = 0;
    while (i < g_ngexec && g_gexec[i].id != exec_id) i++;
    if (i < g_ngexec) v = ++g_gexec[i].ver;
    else if (g_ngexec < GEXEC_CAP) { g_gexec[g_ngexec++] = (GexecVer){exec_id, 2}; v = 2; }
    else g_gexec_full = 1;
    pthread_mutex_unlock(&g_gexec_mu);
    return v;
}

static void gexec_forget(uint32_t exec_id) {
    pthread_mutex_lock(&g_gexec_mu);
    for (int i = 0; i < g_ngexec; i++)
        if (g_gexec[i].id == exec_id) { g_gexec[i] = g_gexec[--g_ngexec]; break; }
    pthread_mutex_unlock(&g_gexec_mu);
}

static void graph_exec_updated(const CUpti_CallbackData *cb) {
    const char *f = cb->functionName;
    if (!f || strncmp(f, "cuGraph", 7) != 0) return;
    int update = strncmp(f, "cuGraphExecUpdate", 17) == 0
              || (strncmp(f, "cuGraphExec", 11) == 0 && strstr(f, "SetParams"))
              || strncmp(f, "cuGraphNodeSetEnabled", 21) == 0;
    if (!update || !cb->functionParams || !cb->functionReturnValue) return;
    if (*(const CUresult *)cb->functionReturnValue != CUDA_SUCCESS) return;
    /* every one of these takes the exec first */
    CUgraphExec exec = *(CUgraphExec const *)cb->functionParams;
    uint32_t id = hp_cupti_graph_exec_id((const void *)exec);
    if (!id) return;
    uint32_t v = gexec_bump(id);
    char x[192];
    char name[64];
    snprintf(name, sizeof(name), "%s", f);
    char *suffix = strstr(name, "_v");
    if (suffix && suffix[2] >= '0' && suffix[2] <= '9' && !strchr(suffix + 2, '_')) *suffix = '\0';
    snprintf(x, sizeof(x), "type=graph_exec_update,op=graph_exec_update,rt=cuda,gexec=%u,gver=%u,api=%s", id, v,
             name);
    emit_instant("cudaGraphExecUpdate", x);
}

/* ── Graph structure ─────────────────────────────────────────────────────
 * At GRAPHEXEC_CREATED the source graph is read once (cuGraphGetNodes,
 * cuGraphGetEdges, cuGraphNodeGetType, child graphs recursively) and its
 * edges are translated to the exec's own nodes through the clone pairs --
 * the device records name the exec's nodes (graphNodeId=, gnode= on the
 * span), not the source graph's. Emitted as instants:
 *   cudaGraphExec   type=graph_exec,gexec=<exec id>,cgraph=<graph id>,
 *                   nnodes=,nedges=[,unmapped=N][,lossy=1][,truncated=1]
 *   cudaGraphNodes  type=graph_nodes,gexec=,part=,nodes=<node>/<type>;...
 *   cudaGraphEdges  type=graph_edges,gexec=,part=,edges=<a>-<b>[p];...
 * ('/', not ':': a tag value never holds the record's field separator).
 * <type> is CUgraphNodeType; a child graph becomes two marker nodes (type
 * 100 entry, 101 exit) around its own nodes, so an empty child still orders
 * what is around it. An edge marked p is not a completion dependency
 * (programmatic / launch-order ports, CUDA 12.3): the downstream node may
 * start before the upstream one ends. A node without a clone (unmapped=)
 * keeps a marker id: no device record can name it. Edges read without
 * their edge data where the driver refuses that (lossy=1) are not emitted.
 * At most HP_GRAPH_MAX_NODES nodes / HP_GRAPH_MAX_EDGES edges per exec
 * (truncated=1: none emitted). The exec's later launches only look up its
 * id and version: the structure is never read again. */
#define HP_GRAPH_MAX_NODES (1u << 20)
#define HP_GRAPH_MAX_EDGES (1u << 22)
#define HP_GRAPH_PART      256
#define HP_GRAPH_MAX_DEPTH 8
#define HP_MARK_BIT        (1ULL << 62)

typedef struct { uint32_t graph; uint64_t node, orig; } ClonePair;
static __thread ClonePair *t_clones;
static __thread size_t     t_nclones, t_cclones;

static void note_clone(const hp_graph_data_t *g) {
    if (!g || !p.GetGraphNodeId || !p.GetGraphId) return;
    uint32_t gid = 0;
    uint64_t nid = 0, oid = 0;
    if (!g->graph || p.GetGraphId(g->graph, &gid) != CUPTI_SUCCESS) return;
    if (!g->node || p.GetGraphNodeId(g->node, &nid) != CUPTI_SUCCESS) return;
    if (!g->originalNode || p.GetGraphNodeId(g->originalNode, &oid) != CUPTI_SUCCESS) return;
    if (t_nclones == t_cclones) {
        size_t nc = t_cclones ? t_cclones * 2 : 256;
        if (nc > HP_GRAPH_MAX_NODES) return;
        ClonePair *np = (ClonePair *)realloc(t_clones, nc * sizeof(ClonePair));
        if (!np) return;
        t_clones = np;
        t_cclones = nc;
    }
    t_clones[t_nclones++] = (ClonePair){gid, nid, oid};
}

typedef struct {
    uint32_t exec;
    ClonePair *map;        /* this exec's pairs, sorted by orig */
    size_t nmap;
    uint64_t *nid; int *ntype; size_t nn, cn;
    uint64_t *ea, *eb; char *ep; size_t ne, ce;
    uint64_t marks;
    int unmapped, lossy, truncated;
} Snap;

static int pair_cmp(const void *a, const void *b) {
    uint64_t x = ((const ClonePair *)a)->orig, y = ((const ClonePair *)b)->orig;
    return x < y ? -1 : x > y;
}

static uint64_t exec_node(Snap *S, CUgraphNode n) {
    uint64_t orig = 0;
    if (p.GetGraphNodeId(n, &orig) == CUPTI_SUCCESS && S->nmap) {
        ClonePair key = {0, 0, orig};
        const ClonePair *hit = (const ClonePair *)bsearch(&key, S->map, S->nmap, sizeof(ClonePair), pair_cmp);
        if (hit) return hit->node;
    }
    S->unmapped++;
    return HP_MARK_BIT | ++S->marks;
}

static void snap_node(Snap *S, uint64_t id, int type) {
    if (S->truncated) return;
    if (S->nn == S->cn) {
        size_t nc = S->cn ? S->cn * 2 : 64;
        if (nc > HP_GRAPH_MAX_NODES) { S->truncated = 1; return; }
        uint64_t *a = (uint64_t *)realloc(S->nid, nc * sizeof(uint64_t));
        int *b = a ? (int *)realloc(S->ntype, nc * sizeof(int)) : NULL;
        if (a) S->nid = a;
        if (!b) { S->truncated = 1; return; }
        S->ntype = b;
        S->cn = nc;
    }
    S->nid[S->nn] = id;
    S->ntype[S->nn++] = type;
}

static void snap_edge(Snap *S, uint64_t a, uint64_t b, char weak) {
    if (S->truncated) return;
    if (S->ne == S->ce) {
        size_t nc = S->ce ? S->ce * 2 : 64;
        if (nc > HP_GRAPH_MAX_EDGES) { S->truncated = 1; return; }
        uint64_t *x = (uint64_t *)realloc(S->ea, nc * sizeof(uint64_t));
        if (x) S->ea = x;
        uint64_t *y = x ? (uint64_t *)realloc(S->eb, nc * sizeof(uint64_t)) : NULL;
        if (y) S->eb = y;
        char *z = y ? (char *)realloc(S->ep, nc) : NULL;
        if (!z) { S->truncated = 1; return; }
        S->ep = z;
        S->ce = nc;
    }
    S->ea[S->ne] = a;
    S->eb[S->ne] = b;
    S->ep[S->ne++] = weak;
}

static int handle_cmp(const void *a, const void *b) {
    uintptr_t x = (uintptr_t)*(CUgraphNode const *)a, y = (uintptr_t)*(CUgraphNode const *)b;
    return x < y ? -1 : x > y;
}

/* One graph level: its nodes (a child graph's as two markers around its
 * own), its edges; *roots / *leaves (malloc'd, the caller frees) the
 * entry / exit ids of its sources and sinks, for the level above. */
static void snap_graph(Snap *S, CUgraph g, int depth, uint64_t **roots, size_t *nroots,
                       uint64_t **leaves, size_t *nleaves) {
    *roots = *leaves = NULL;
    *nroots = *nleaves = 0;
    size_t n = 0, ne = 0;
    if (drv.GetNodes(g, NULL, &n) != CUDA_SUCCESS || n > HP_GRAPH_MAX_NODES) { S->truncated = 1; return; }
    CUgraphNode *nodes = (CUgraphNode *)calloc(n ? n : 1, sizeof(CUgraphNode));
    CUgraphNode *sorted = (CUgraphNode *)calloc(n ? n : 1, sizeof(CUgraphNode));
    uint64_t *ent = (uint64_t *)calloc(n ? n : 1, sizeof(uint64_t));
    uint64_t *ext = (uint64_t *)calloc(n ? n : 1, sizeof(uint64_t));
    int *in = (int *)calloc(n ? n : 1, sizeof(int)), *out = (int *)calloc(n ? n : 1, sizeof(int));
    size_t *order = (size_t *)calloc(n ? n : 1, sizeof(size_t));
    if (!nodes || !sorted || !ent || !ext || !in || !out || !order || drv.GetNodes(g, nodes, &n) != CUDA_SUCCESS) {
        S->truncated = 1;
        goto done;
    }
    for (size_t i = 0; i < n; i++) {
        int type = -1;
        drv.NodeGetType(nodes[i], &type);
        CUgraph child = NULL;
        if (type == 4 && drv.ChildGraph && depth < HP_GRAPH_MAX_DEPTH &&      /* CU_GRAPH_NODE_TYPE_GRAPH */
            drv.ChildGraph(nodes[i], &child) == CUDA_SUCCESS && child) {
            ent[i] = HP_MARK_BIT | ++S->marks;
            ext[i] = HP_MARK_BIT | ++S->marks;
            snap_node(S, ent[i], 100);
            snap_node(S, ext[i], 101);
            uint64_t *cr, *cl;
            size_t ncr, ncl;
            snap_graph(S, child, depth + 1, &cr, &ncr, &cl, &ncl);
            for (size_t k = 0; k < ncr; k++) snap_edge(S, ent[i], cr[k], 0);
            for (size_t k = 0; k < ncl; k++) snap_edge(S, cl[k], ext[i], 0);
            if (!ncr) snap_edge(S, ent[i], ext[i], 0);
            free(cr);
            free(cl);
        } else {
            ent[i] = ext[i] = exec_node(S, nodes[i]);
            snap_node(S, ent[i], type);
        }
    }
    /* sorted handles -> index, for the edge list */
    memcpy(sorted, nodes, n * sizeof(CUgraphNode));
    qsort(sorted, n, sizeof(CUgraphNode), handle_cmp);
    for (size_t i = 0; i < n; i++) {
        CUgraphNode *hit = (CUgraphNode *)bsearch(&nodes[i], sorted, n, sizeof(CUgraphNode), handle_cmp);
        if (hit) order[hit - sorted] = i;
    }
    CUresult er = drv.GetEdges ? drv.GetEdges(g, NULL, NULL, &ne) : CUDA_ERROR_NOT_SUPPORTED;
    int with_data = 0;
    if (er != CUDA_SUCCESS && drv.GetEdges2) {
        er = drv.GetEdges2(g, NULL, NULL, NULL, &ne);
        with_data = er == CUDA_SUCCESS;
    }
    if (er != CUDA_SUCCESS) { S->lossy = 1; ne = 0; }
    if (ne > HP_GRAPH_MAX_EDGES) { S->truncated = 1; goto done; }
    if (ne) {
        CUgraphNode *from = (CUgraphNode *)calloc(ne, sizeof(CUgraphNode));
        CUgraphNode *to = (CUgraphNode *)calloc(ne, sizeof(CUgraphNode));
        unsigned char *data = with_data ? (unsigned char *)calloc(ne, 8) : NULL;   /* CUgraphEdgeData: 8 B */
        CUresult r = (!from || !to || (with_data && !data)) ? CUDA_ERROR_OUT_OF_MEMORY
                   : with_data ? drv.GetEdges2(g, from, to, data, &ne) : drv.GetEdges(g, from, to, &ne);
        if (r == CUDA_SUCCESS) {
            for (size_t k = 0; k < ne; k++) {
                CUgraphNode *a = (CUgraphNode *)bsearch(&from[k], sorted, n, sizeof(CUgraphNode), handle_cmp);
                CUgraphNode *b = (CUgraphNode *)bsearch(&to[k], sorted, n, sizeof(CUgraphNode), handle_cmp);
                if (!a || !b) continue;
                size_t ia = order[a - sorted], ib = order[b - sorted];
                /* edge data: from_port, to_port, type -- anything but the default
                 * full dependency lets the downstream node start early */
                char weak = data && (data[8 * k] || data[8 * k + 1] || data[8 * k + 2]) ? 'p' : 0;
                snap_edge(S, ext[ia], ent[ib], weak);
                out[ia]++;
                in[ib]++;
            }
        } else {
            S->lossy = 1;
        }
        free(from);
        free(to);
        free(data);
    }
    *roots = (uint64_t *)calloc(n ? n : 1, sizeof(uint64_t));
    *leaves = (uint64_t *)calloc(n ? n : 1, sizeof(uint64_t));
    for (size_t i = 0; *roots && *leaves && i < n; i++) {
        if (!in[i]) (*roots)[(*nroots)++] = ent[i];
        if (!out[i]) (*leaves)[(*nleaves)++] = ext[i];
    }
done:
    free(nodes); free(sorted); free(ent); free(ext); free(in); free(out); free(order);
}

static void graph_snapshot(const hp_graph_data_t *g) {
    pthread_once(&g_drv_once, drv_init);
    uint32_t exec = 0, gid = 0;
    if (!g || !g->graphExec || !p.GetGraphExecId || p.GetGraphExecId(g->graphExec, &exec) != CUPTI_SUCCESS)
        return;
    if (g->graph && p.GetGraphId) p.GetGraphId(g->graph, &gid);
    Snap S = {0};
    S.exec = exec;
    /* this exec's clone pairs (an exec's internal graph has its id) */
    S.map = (ClonePair *)malloc((t_nclones ? t_nclones : 1) * sizeof(ClonePair));
    for (size_t i = 0; S.map && i < t_nclones; i++)
        if (t_clones[i].graph == exec) S.map[S.nmap++] = t_clones[i];
    t_nclones = 0;
    if (S.map) qsort(S.map, S.nmap, sizeof(ClonePair), pair_cmp);
    int saved = hp_cuda_in_hook;
    hp_cuda_in_hook = 1;                      /* the queries are not the program's */
    if (drv.GetNodes && drv.NodeGetType && p.GetGraphNodeId && g->graph) {
        uint64_t *r, *l;
        size_t nr, nl;
        snap_graph(&S, g->graph, 0, &r, &nr, &l, &nl);
        free(r);
        free(l);
    } else {
        S.truncated = 1;
    }
    hp_cuda_in_hook = saved;
    char head[256];
    int hn = 0;
    APPEND(head, hn, "type=graph_exec,op=graph_exec,rt=cuda,src=cupti_res,gexec=%u,cgraph=%u,nnodes=%zu,nedges=%zu",
           exec, gid, S.truncated ? (size_t)0 : S.nn, S.truncated ? (size_t)0 : S.ne);
    if (S.unmapped) APPEND(head, hn, ",unmapped=%d", S.unmapped);
    if (S.lossy) APPEND(head, hn, ",lossy=1");
    if (S.truncated) APPEND(head, hn, ",truncated=1");
    emit_instant("cudaGraphExec", head);
    if (!S.truncated) {
        char *buf = (char *)malloc(HP_GRAPH_PART * 48 + 128);
        for (size_t base = 0; buf && base < S.nn; base += HP_GRAPH_PART) {
            int n = snprintf(buf, 128, "type=graph_nodes,gexec=%u,part=%zu,nodes=", exec, base / HP_GRAPH_PART);
            for (size_t i = base; i < S.nn && i < base + HP_GRAPH_PART; i++)
                n += snprintf(buf + n, 48, "%s%llu/%d", i > base ? ";" : "", (unsigned long long)S.nid[i], S.ntype[i]);
            emit_instant("cudaGraphNodes", buf);
        }
        for (size_t base = 0; buf && base < S.ne; base += HP_GRAPH_PART) {
            int n = snprintf(buf, 128, "type=graph_edges,gexec=%u,part=%zu,edges=", exec, base / HP_GRAPH_PART);
            for (size_t i = base; i < S.ne && i < base + HP_GRAPH_PART; i++)
                n += snprintf(buf + n, 48, "%s%llu-%llu%s", i > base ? ";" : "", (unsigned long long)S.ea[i],
                              (unsigned long long)S.eb[i], S.ep[i] ? "p" : "");
            emit_instant("cudaGraphEdges", buf);
        }
        free(buf);
    }
    free(S.map); free(S.nid); free(S.ntype); free(S.ea); free(S.eb); free(S.ep);
}

static void resource_callback(CUpti_CallbackId cbid, const CUpti_ResourceData *rd) {
    if (!rd) return;
    switch ((int)cbid) {
    case HP_RES_STREAM_CREATED:
        stream_created(rd);
        break;
    case HP_RES_STREAM_DESTROY_STARTING:
        hp_cuda_stream_died((const void *)rd->resourceHandle.stream);
        break;
    case HP_RES_GRAPHNODE_CLONED:
        note_clone((const hp_graph_data_t *)rd->resourceDescriptor);
        break;
    case HP_RES_GRAPHEXEC_CREATED:
        graph_snapshot((const hp_graph_data_t *)rd->resourceDescriptor);
        break;
    case HP_RES_GRAPHEXEC_DESTROY_STARTING: {
        const hp_graph_data_t *g = (const hp_graph_data_t *)rd->resourceDescriptor;
        uint32_t id = g ? hp_cupti_graph_exec_id((const void *)g->graphExec) : 0;
        if (id) gexec_forget(id);
        break;
    }
    default:
        break;
    }
}

/* ── callback API: correlation capture + un-intercepted submissions ────── */

static __thread int      t_armed;
static __thread uint32_t t_rt_corr, t_drv_corr;
static __thread int      t_rt_open;      /* un-intercepted runtime calls in flight */

#define CB_OPEN_BIT (1ULL << 63)
#define CB_EMIT_BIT (1ULL << 62)
#define CB_TS_MASK  ((1ULL << 62) - 1)

typedef struct { const char *prefix; const char *cat; const char *op; const char *sync; } api_class_t;

/* Calls that submit device work: their host span is what the device record
 * correlates to. */
static const api_class_t k_classes[] = {
    {"cudaGraphLaunch", "cuda",   "graph",  NULL}, {"cuGraphLaunch", "cuda", "graph", NULL},
    {"cudaLaunch",      "cuda",   "kernel", NULL}, {"cuLaunch",      "cuda", "kernel", NULL},
    {"cudaMemcpy",      "memory", "memcpy", NULL}, {"cuMemcpy",    "memory", "memcpy", NULL},
    {"cudaMemset",      "memory", "memset", NULL}, {"cuMemset",    "memory", "memset", NULL},
};
/* Event synchronizations and event queries, always: CUPTI reports both as
 * event synchronization records (a query when it found the event complete
 * -- only those are emitted, as by cuda_hook.c's cudaEventQuery), and such
 * a record needs its host call. A driver-API call (cuEventSynchronize,
 * cuEventQuery) or one through a runtime no wrapper sees has no other.
 * Their first parameter is the event (event=, cuda_hook.c's numbering). */
static const api_class_t k_event_classes[] = {
    {"cudaEventSynchronize", "sync", "sync", "event"}, {"cuEventSynchronize", "sync", "sync", "event"},
    {"cudaEventQuery",       "sync", "sync", "event"}, {"cuEventQuery",       "sync", "sync", "event"},
};
/* Other synchronization calls: only with HP_CUPTI_CB_SYNCS (a static-runtime
 * program, where no LD_PRELOAD wrapper sees them). Otherwise they are left
 * out -- other hprofiler hooks (NCCL) make such calls internally for their
 * own timing and would show up as application calls. */
static const api_class_t k_sync_classes[] = {
    {"cudaDeviceSynchronize", "sync", "sync", "device"}, {"cuCtxSynchronize",    "sync", "sync", "device"},
    {"cudaStreamSynchronize", "sync", "sync", "stream"}, {"cuStreamSynchronize", "sync", "sync", "stream"},
    {"cudaStreamWaitEvent",   "cuda", "stream_wait", NULL}, {"cuStreamWaitEvent", "cuda", "stream_wait", NULL},
    {"cudaEventRecord",       "cuda", "event_record", NULL}, {"cuEventRecord",    "cuda", "event_record", NULL},
};
static int g_cb_syncs = 0;

static const api_class_t *find_class(const api_class_t *tab, size_t n, const char *name) {
    for (size_t i = 0; i < n; i++)
        if (strncmp(name, tab[i].prefix, strlen(tab[i].prefix)) == 0)
            return &tab[i];
    return NULL;
}
#define FIND_CLASS(tab, name) find_class(tab, sizeof(tab) / sizeof(tab[0]), name)

static const api_class_t *classify(const char *name) {
    if (!name) return NULL;
    const api_class_t *c = FIND_CLASS(k_classes, name);
    if (!c) c = FIND_CLASS(k_event_classes, name);
    if (!c && g_cb_syncs) c = FIND_CLASS(k_sync_classes, name);
    return c;
}

static int is_event_class(const api_class_t *c) {
    return c >= k_event_classes && c < k_event_classes + sizeof(k_event_classes) / sizeof(k_event_classes[0]);
}

/* Kernel launches the hook did not intercept (cuLaunchKernelEx obtained with
 * dlsym, as Triton does; a library's own launches): the launched function
 * handle is in the call's parameters, and its module id names the code that
 * ran (mod=, copied to the device span by the correlation). Only the leading
 * members are declared -- layouts from CUPTI's generated_cuda_meta.h and
 * generated_cuda_runtime_api_meta.h; callback ids (stable numbers) from
 * cupti_driver_cbid.h and cupti_runtime_cbid.h. */
typedef struct { const void *f; } hp_launch_f_params;                       /* f first */
typedef struct { const void *config; const void *f; } hp_launch_cfg_f_params;  /* config, f */

enum {
    HP_DRV_cuLaunchKernel = 307, HP_DRV_cuLaunchKernel_ptsz = 442,
    HP_DRV_cuLaunchCooperativeKernel = 477, HP_DRV_cuLaunchCooperativeKernel_ptsz = 478,
    HP_DRV_cuLaunchKernelEx = 652, HP_DRV_cuLaunchKernelEx_ptsz = 653,
    HP_RT_cudaLaunchKernel_v7000 = 211, HP_RT_cudaLaunchKernel_ptsz_v7000 = 214,
    HP_RT_cudaLaunchCooperativeKernel_v9000 = 269, HP_RT_cudaLaunchCooperativeKernel_ptsz_v9000 = 270,
    HP_RT_cudaLaunchKernelExC_v11060 = 430, HP_RT_cudaLaunchKernelExC_ptsz_v11060 = 431,
};

/* The function handle of a kernel-launch callback, or NULL. */
static const void *launch_function(int is_rt, CUpti_CallbackId cbid, const void *params) {
    if (!params) return NULL;
    switch (is_rt ? -(int)cbid : (int)cbid) {
    case HP_DRV_cuLaunchKernel: case HP_DRV_cuLaunchKernel_ptsz:
    case HP_DRV_cuLaunchCooperativeKernel: case HP_DRV_cuLaunchCooperativeKernel_ptsz:
    case -HP_RT_cudaLaunchKernel_v7000: case -HP_RT_cudaLaunchKernel_ptsz_v7000:
    case -HP_RT_cudaLaunchCooperativeKernel_v9000: case -HP_RT_cudaLaunchCooperativeKernel_ptsz_v9000:
        return ((const hp_launch_f_params *)params)->f;
    case HP_DRV_cuLaunchKernelEx: case HP_DRV_cuLaunchKernelEx_ptsz:
    case -HP_RT_cudaLaunchKernelExC_v11060: case -HP_RT_cudaLaunchKernelExC_ptsz_v11060:
        return ((const hp_launch_cfg_f_params *)params)->f;
    default:
        return NULL;
    }
}

static void emit_callback_span(const CUpti_CallbackData *cb, int is_rt, CUpti_CallbackId cbid,
                               uint64_t t0, uint64_t t1) {
    const api_class_t *c = classify(cb->functionName);
    if (!c) return;
    /* an event query that found the event not complete: a poll, not
     * recorded (see k_event_classes) */
    if (strstr(cb->functionName, "EventQuery") && cb->functionReturnValue
        && *(const int *)cb->functionReturnValue != 0)
        return;
    /* "cudaLaunchKernel_v7000" -> "cudaLaunchKernel" */
    char name[128];
    snprintf(name, sizeof(name), "%s", cb->functionName);
    char *v = strstr(name, "_v");
    if (v && v[2] >= '0' && v[2] <= '9') *v = '\0';
    int is_async = strstr(name, "Async") != NULL;
    const char *type = c->sync ? "sync"
                     : (!strcmp(c->op, "stream_wait") || !strcmp(c->op, "event_record")) ? c->op
                     : (!strcmp(c->op, "kernel") || !strcmp(c->op, "graph") || is_async)
                       ? "launch" : c->op;   /* blocking copy/memset: the call is the transfer */
    /* Span id: the CUDA hook's id space (hp_ids.h) with bit 31 of the
     * counter set, so it never collides with cuda_hook.c's host spans. */
    static uint32_t cb_seq = 0;
    uint64_t sid = hp_span_id(HP_ID_CUDA, (uint32_t)getpid(),
                              0x80000000u | (__atomic_add_fetch(&cb_seq, 1, __ATOMIC_RELAXED) & 0x7fffffffu));
    char x[256];
    int n = snprintf(x, sizeof(x), "type=%s,op=%s,side=cpu,rt=cuda,timing=host,src=cupti_cb,corr=%u,sid=%llu",
                     type, c->op, cb->correlationId, (unsigned long long)sid);
    if (c->sync && n > 0 && n < (int)sizeof(x))
        n += snprintf(x + n, sizeof(x) - (size_t)n, ",sync=%s", c->sync);
    if (is_event_class(c) && cb->functionParams && n > 0 && n < (int)sizeof(x))
        n += snprintf(x + n, sizeof(x) - (size_t)n, ",event=%d",
                      hp_cuda_event_id(*(const void *const *)cb->functionParams));
    uint64_t mod = hp_cuda_function_module(launch_function(is_rt, cbid, cb->functionParams));
    if (mod && n > 0 && n < (int)sizeof(x))
        n += snprintf(x + n, sizeof(x) - (size_t)n, ",mod=%016llx", (unsigned long long)mod);
    if (!strcmp(c->op, "graph") && cb->functionParams) {
        /* cudaGraphLaunch / cuGraphLaunch take the exec first */
        uint32_t gexec = hp_cupti_graph_exec_id(*(const void *const *)cb->functionParams);
        if (gexec && n > 0 && n < (int)sizeof(x))
            snprintf(x + n, sizeof(x) - (size_t)n, ",gexec=%u,gver=%u", gexec, hp_cupti_graph_exec_version(gexec));
    }
    hp_cuda_emit_span(c->cat, (pid_t)syscall(SYS_gettid), t0, t1 - t0, name, x);
}

static void CUPTIAPI api_callback(void *ud, CUpti_CallbackDomain dom,
                                  CUpti_CallbackId cbid, const void *data) {
    (void)ud;
    if (dom == CUPTI_CB_DOMAIN_RESOURCE) {
        resource_callback(cbid, (const CUpti_ResourceData *)data);
        return;
    }
    if (dom != CUPTI_CB_DOMAIN_RUNTIME_API && dom != CUPTI_CB_DOMAIN_DRIVER_API) return;
    const CUpti_CallbackData *cb = (const CUpti_CallbackData *)data;
    int is_rt = dom == CUPTI_CB_DOMAIN_RUNTIME_API;
    if (!is_rt && cb->callbackSite == CUPTI_API_EXIT && !hp_cuda_in_hook)
        graph_exec_updated(cb);        /* a graph exec's version (see "Graph versions") */

    if (t_armed) {                      /* inside an intercepted call */
        if (cb->callbackSite == CUPTI_API_ENTER) {
            if (is_rt) { if (!t_rt_corr) t_rt_corr = cb->correlationId; }
            else if (!t_drv_corr) t_drv_corr = cb->correlationId;
        }
        return;
    }
    if (hp_cuda_in_hook) {              /* the hook's own CUDA calls */
        if (cb->callbackSite == CUPTI_API_ENTER) note_internal(cb->correlationId);
        return;
    }

    if (cb->callbackSite == CUPTI_API_ENTER) {
        uint64_t v = 0;
        /* A driver call made by a runtime call is the same submission. */
        if ((is_rt || t_rt_open == 0) && classify(cb->functionName))
            v |= CB_EMIT_BIT | (mono_ns() & CB_TS_MASK);
        if (is_rt) { t_rt_open++; v |= CB_OPEN_BIT; }
        if (cb->correlationData) *cb->correlationData = v;
    } else {
        uint64_t v = cb->correlationData ? *cb->correlationData : 0;
        if (v & CB_OPEN_BIT) t_rt_open--;
        if (v & CB_EMIT_BIT) emit_callback_span(cb, is_rt, cbid, v & CB_TS_MASK, mono_ns());
    }
}

void hp_cupti_arm(void) {
    t_armed = 1;
    t_rt_corr = t_drv_corr = 0;
}

void hp_cupti_disarm(uint32_t *corr, uint32_t *corr2) {
    t_armed = 0;
    if (t_rt_corr) {
        *corr = t_rt_corr;
        *corr2 = (t_drv_corr && t_drv_corr != t_rt_corr) ? t_drv_corr : 0;
    } else {
        *corr = t_drv_corr;
        *corr2 = 0;
    }
}

/* ── library loading and start-up ────────────────────────────────────────── */

/* dlopen one candidate and check its version against the headers. A
 * libcupti older than the headers could deliver older, shorter record
 * versions, so it is refused (and the next candidate tried). */
static void *try_libcupti(const char *path, int flags) {
    void *h = dlopen(path, flags);
    if (!h) return NULL;
    __typeof__(&cuptiGetVersion) gv = (__typeof__(gv))dlsym(h, "cuptiGetVersion");
    uint32_t v = 0;
    if (!gv || gv(&v) != CUPTI_SUCCESS || v < CUPTI_API_VERSION) {
        snprintf(g_lib_reason, sizeof(g_lib_reason), "libcupti_v%u_older_than_headers_v%u",
                 v, (unsigned)CUPTI_API_VERSION);
        if (!(flags & RTLD_NOLOAD)) dlclose(h);
        return NULL;
    }
    return h;
}

static void *load_libcupti(void) {
    static const char *names[] = {
        "libcupti.so.13", "libcupti.so.12", "libcupti.so.11", "libcupti.so", NULL
    };
    /* Already in the process (the application, nsys, ...): a second copy
     * of CUPTI must not be loaded next to it -- use it or nothing. */
    for (int i = 0; names[i]; i++) {
        void *h = dlopen(names[i], RTLD_LAZY | RTLD_NOLOAD);
        if (h) { dlclose(h); return try_libcupti(names[i], RTLD_LAZY | RTLD_NOLOAD); }
    }
    snprintf(g_lib_reason, sizeof(g_lib_reason), "libcupti_not_found");
    const char *override = getenv("HPROFILER_CUPTI_LIB");
    void *h;
    if (override && *override && (h = try_libcupti(override, RTLD_LAZY | RTLD_LOCAL))) return h;
#ifdef HP_CUPTI_LIB_DIR
    /* The toolkit whose headers were compiled in. */
    if ((h = try_libcupti(HP_CUPTI_LIB_DIR "/libcupti.so", RTLD_LAZY | RTLD_LOCAL))) return h;
    if ((h = try_libcupti(HP_CUPTI_LIB_DIR "/../extras/CUPTI/lib64/libcupti.so", RTLD_LAZY | RTLD_LOCAL)))
        return h;
#endif
    const char *home = getenv("CUDA_HOME");
    if (!home) home = getenv("CUDA_PATH");
    if (home) {
        char path[512];
        snprintf(path, sizeof(path), "%s/extras/CUPTI/lib64/libcupti.so", home);
        if ((h = try_libcupti(path, RTLD_LAZY | RTLD_LOCAL))) return h;
        snprintf(path, sizeof(path), "%s/lib64/libcupti.so", home);
        if ((h = try_libcupti(path, RTLD_LAZY | RTLD_LOCAL))) return h;
    }
    for (int i = 0; names[i]; i++)
        if ((h = try_libcupti(names[i], RTLD_LAZY | RTLD_LOCAL))) return h;
    return NULL;
}

#define SYM(field, name) (p.field = (__typeof__(p.field))dlsym(h, name))

static void lib_init(void) {
    void *h = load_libcupti();
    if (!h) return;                    /* g_lib_reason says why */
    if (!SYM(RegisterCallbacks, "cuptiActivityRegisterCallbacks") |
        !SYM(Enable, "cuptiActivityEnable") | !SYM(FlushAll, "cuptiActivityFlushAll") |
        !SYM(GetNextRecord, "cuptiActivityGetNextRecord") |
        !SYM(GetNumDropped, "cuptiActivityGetNumDroppedRecords") |
        !SYM(GetTimestamp, "cuptiGetTimestamp") | !SYM(GetVersion, "cuptiGetVersion") |
        !SYM(Subscribe, "cuptiSubscribe") | !SYM(EnableDomain, "cuptiEnableDomain")) {
        snprintf(g_lib_reason, sizeof(g_lib_reason), "libcupti_symbols_missing");
        return;
    }
    SYM(RegisterTimestampCallback, "cuptiActivityRegisterTimestampCallback");
    SYM(FlushPeriod, "cuptiActivityFlushPeriod");
    SYM(EnableLatencyTimestamps, "cuptiActivityEnableLatencyTimestamps");
    SYM(GetGraphId, "cuptiGetGraphId");
    SYM(GetGraphNodeId, "cuptiGetGraphNodeId");
    SYM(GetGraphExecId, "cuptiGetGraphExecId");
    SYM(GetStreamIdEx, "cuptiGetStreamIdEx");
    SYM(GetContextId, "cuptiGetContextId");

    p.GetVersion(&g_lib_version);      /* checked by try_libcupti() */
    /* Must precede every cuptiActivityEnable so all records use it. */
    if (p.RegisterTimestampCallback && p.RegisterTimestampCallback(mono_ts_cb) == CUPTI_SUCCESS)
        g_clock_cb = 1;
    else
        calibrate_offset();
    if (p.RegisterCallbacks(buf_requested, buf_completed) != CUPTI_SUCCESS) {
        snprintf(g_lib_reason, sizeof(g_lib_reason), "register_buffer_callbacks_failed");
        return;
    }
    g_lib_ok = 1;
}

/* Forced final flush (atexit, before libcupti's own teardown; or the hook
 * destructor if no atexit ran). The final_flush=1 status after it is the
 * collector's evidence that buffered device records were delivered: an
 * active tracer whose process ends without it (crash, _exit, kill) is
 * reported as possibly missing device records. */
static void atexit_flush(void) {
    if (g_lib_ok && !g_exit_flushed) {
        int saved = hp_cuda_in_hook;
        hp_cuda_in_hook = 1;
        p.FlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
        hp_cuda_in_hook = saved;
        if (g_active & HP_CUPTI_ACTIVITY) emit_status("final_flush=1");
    }
    g_exit_flushed = 1;
}

void hp_cupti_final_flush(void) { atexit_flush(); }

unsigned hp_cupti_start(unsigned want) {
    pthread_once(&g_lib_once, lib_init);
    pthread_mutex_lock(&g_start_mu);
    if (!g_lib_ok) {
        static int reported = 0;
        if (want && !reported) {
            char f[160];
            snprintf(f, sizeof(f), "status=unavailable,reason=%s", g_lib_reason);
            emit_status(f);
            reported = 1;
        }
        pthread_mutex_unlock(&g_start_mu);
        return 0;
    }
    if (want & HP_CUPTI_CB_SYNCS) g_cb_syncs = 1;
    if ((want & HP_CUPTI_ACTIVITY) && !(g_active & HP_CUPTI_ACTIVITY)) {
        CUpti_SubscriberHandle sub;
        g_callbacks_ok = p.Subscribe(&sub, api_callback, NULL) == CUPTI_SUCCESS &&
                         p.EnableDomain(1, sub, CUPTI_CB_DOMAIN_RUNTIME_API) == CUPTI_SUCCESS &&
                         p.EnableDomain(1, sub, CUPTI_CB_DOMAIN_DRIVER_API) == CUPTI_SUCCESS;
        /* stream lives and graph structure (see "Resource callbacks") */
        __atomic_store_n(&g_res_ok, g_callbacks_ok &&
                         p.EnableDomain(1, sub, CUPTI_CB_DOMAIN_RESOURCE) == CUPTI_SUCCESS, __ATOMIC_RELEASE);
        if (p.EnableLatencyTimestamps && p.EnableLatencyTimestamps(1) == CUPTI_SUCCESS)
            g_latency = 1;
        CUptiResult rk = p.Enable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
        if (rk == CUPTI_SUCCESS) {
            p.Enable(CUPTI_ACTIVITY_KIND_MEMCPY);
            p.Enable(CUPTI_ACTIVITY_KIND_MEMCPY2);
            p.Enable(CUPTI_ACTIVITY_KIND_MEMSET);
            p.Enable(CUPTI_ACTIVITY_KIND_SYNCHRONIZATION);
            if (p.FlushPeriod) {
                const char *ms = getenv("HPROFILER_CUPTI_FLUSH_MS");
                p.FlushPeriod(ms ? (uint32_t)strtoul(ms, NULL, 10) : 1000u);
            }
            g_active |= HP_CUPTI_ACTIVITY;
            atexit(atexit_flush);
            char f[256];
            if (g_clock_cb)
                snprintf(f, sizeof(f), "status=active,api_version=%u,headers_version=%u,clock=monotonic_callback,"
                         "correlation=%s,latency=%d,final_marker=1", g_lib_version, (unsigned)CUPTI_API_VERSION,
                         g_callbacks_ok ? "callback" : "unavailable", g_latency);
            else
                snprintf(f, sizeof(f), "status=active,api_version=%u,headers_version=%u,clock=offset,offset_ns=%lld,"
                         "clock_err_ns=%llu,correlation=%s,latency=%d,final_marker=1", g_lib_version, (unsigned)CUPTI_API_VERSION,
                         (long long)g_offset_ns, (unsigned long long)g_offset_err_ns,
                         g_callbacks_ok ? "callback" : "unavailable", g_latency);
            emit_status(f);
        } else {
            char f[96];
            snprintf(f, sizeof(f), "status=unavailable,reason=enable_kernel_activity_failed_%d", (int)rk);
            emit_status(f);
        }
    }
    if ((want & HP_CUPTI_PCSAMPLING) && !(g_active & HP_CUPTI_PCSAMPLING)) {
        g_pc_enabled = 1;
        if (p.Enable(CUPTI_ACTIVITY_KIND_FUNCTION) == CUPTI_SUCCESS &&
            p.Enable(CUPTI_ACTIVITY_KIND_PC_SAMPLING) == CUPTI_SUCCESS)
            g_active |= HP_CUPTI_PCSAMPLING;
    }
    unsigned active = g_active;
    pthread_mutex_unlock(&g_start_mu);
    return active;
}

void hp_cupti_flush(void) {
    if (!g_lib_ok || g_exit_flushed) return;
    int saved = hp_cuda_in_hook;
    hp_cuda_in_hook = 1;
    p.FlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
    hp_cuda_in_hook = saved;
}

#ifdef HP_CUPTI_UNIT_TEST
void hp_cupti_test_handle_record(const void *record) {
    g_pc_enabled = 1;
    handle_record((const CUpti_Activity *)record);
}
void hp_cupti_test_report_dropped(uint64_t n) { report_counters(n); }
void hp_cupti_test_set_clock(int monotonic_callback, int64_t offset_ns) {
    g_clock_cb = monotonic_callback;
    g_offset_ns = offset_ns;
}
void hp_cupti_test_set_latency(int on) { g_latency = on; }
void hp_cupti_test_note_internal(uint32_t corr) { note_internal(corr); }
void hp_cupti_test_set_cb_syncs(int on) { g_cb_syncs = on; }
void hp_cupti_test_api_callback(int runtime_domain, const void *callback_data) {
    api_callback(NULL, runtime_domain ? CUPTI_CB_DOMAIN_RUNTIME_API : CUPTI_CB_DOMAIN_DRIVER_API,
                 0, callback_data);
}
void hp_cupti_test_api_callback_id(int runtime_domain, uint32_t cbid, const void *callback_data) {
    api_callback(NULL, runtime_domain ? CUPTI_CB_DOMAIN_RUNTIME_API : CUPTI_CB_DOMAIN_DRIVER_API,
                 (CUpti_CallbackId)cbid, callback_data);
}
void hp_cupti_test_resource_callback(uint32_t cbid, const void *resource_data) {
    api_callback(NULL, CUPTI_CB_DOMAIN_RESOURCE, (CUpti_CallbackId)cbid, resource_data);
}
#endif

#endif /* HP_CUPTI_ENABLED */
