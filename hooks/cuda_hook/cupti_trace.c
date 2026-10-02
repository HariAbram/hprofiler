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
 * exact host call. Kernel launches and copies/memsets made through calls
 * the LD_PRELOAD layer cannot see (e.g. from libraries with a statically
 * linked CUDA runtime) get a host span from the callback itself (src=
 * cupti_cb), so their device work is still correlated.
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
 * emitted. Not validated against a working GPU on the development machine.
 */
#define _GNU_SOURCE
#include "hp_cupti.h"

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
} p;

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
        APPEND(x, n, ",graph=%u", k->graphId);
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
/* Synchronization calls: only with HP_CUPTI_CB_SYNCS (a static-runtime
 * program, where no LD_PRELOAD wrapper sees them). Otherwise they are left
 * out -- other hprofiler hooks (NCCL) make such calls internally for their
 * own timing and would show up as application calls. */
static const api_class_t k_sync_classes[] = {
    {"cudaDeviceSynchronize", "sync", "sync", "device"}, {"cuCtxSynchronize",    "sync", "sync", "device"},
    {"cudaStreamSynchronize", "sync", "sync", "stream"}, {"cuStreamSynchronize", "sync", "sync", "stream"},
    {"cudaEventSynchronize",  "sync", "sync", "event"},  {"cuEventSynchronize",  "sync", "sync", "event"},
    {"cudaStreamWaitEvent",   "cuda", "stream_wait", NULL}, {"cuStreamWaitEvent", "cuda", "stream_wait", NULL},
    {"cudaEventRecord",       "cuda", "event_record", NULL}, {"cuEventRecord",    "cuda", "event_record", NULL},
};
static int g_cb_syncs = 0;

static const api_class_t *classify(const char *name) {
    if (!name) return NULL;
    for (size_t i = 0; i < sizeof(k_classes) / sizeof(k_classes[0]); i++)
        if (strncmp(name, k_classes[i].prefix, strlen(k_classes[i].prefix)) == 0)
            return &k_classes[i];
    if (g_cb_syncs)
        for (size_t i = 0; i < sizeof(k_sync_classes) / sizeof(k_sync_classes[0]); i++)
            if (strncmp(name, k_sync_classes[i].prefix, strlen(k_sync_classes[i].prefix)) == 0)
                return &k_sync_classes[i];
    return NULL;
}

static void emit_callback_span(const CUpti_CallbackData *cb, uint64_t t0, uint64_t t1) {
    const api_class_t *c = classify(cb->functionName);
    if (!c) return;
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
    /* Span id: same (pid << 32 | n) scheme as cuda_hook.c's host spans, in
     * the upper half of the 32-bit range so the two never collide. */
    static uint32_t cb_seq = 0;
    uint64_t sid = ((uint64_t)(uint32_t)getpid() << 32) | 0x80000000u |
                   (__atomic_add_fetch(&cb_seq, 1, __ATOMIC_RELAXED) & 0x7fffffffu);
    char x[256];
    int n = snprintf(x, sizeof(x), "type=%s,op=%s,side=cpu,rt=cuda,timing=host,src=cupti_cb,corr=%u,sid=%llu",
                     type, c->op, cb->correlationId, (unsigned long long)sid);
    if (c->sync && n > 0 && n < (int)sizeof(x))
        snprintf(x + n, sizeof(x) - (size_t)n, ",sync=%s", c->sync);
    hp_cuda_emit_span(c->cat, (pid_t)syscall(SYS_gettid), t0, t1 - t0, name, x);
}

static void CUPTIAPI api_callback(void *ud, CUpti_CallbackDomain dom,
                                  CUpti_CallbackId cbid, const void *data) {
    (void)ud; (void)cbid;
    if (dom != CUPTI_CB_DOMAIN_RUNTIME_API && dom != CUPTI_CB_DOMAIN_DRIVER_API) return;
    const CUpti_CallbackData *cb = (const CUpti_CallbackData *)data;
    int is_rt = dom == CUPTI_CB_DOMAIN_RUNTIME_API;

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
        if (v & CB_EMIT_BIT) emit_callback_span(cb, v & CB_TS_MASK, mono_ns());
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

static void atexit_flush(void) {
    if (g_lib_ok && !g_exit_flushed) {
        int saved = hp_cuda_in_hook;
        hp_cuda_in_hook = 1;
        p.FlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
        hp_cuda_in_hook = saved;
    }
    g_exit_flushed = 1;
}

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
                         "correlation=%s,latency=%d", g_lib_version, (unsigned)CUPTI_API_VERSION,
                         g_callbacks_ok ? "callback" : "unavailable", g_latency);
            else
                snprintf(f, sizeof(f), "status=active,api_version=%u,headers_version=%u,clock=offset,offset_ns=%lld,"
                         "clock_err_ns=%llu,correlation=%s,latency=%d", g_lib_version, (unsigned)CUPTI_API_VERSION,
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
#endif

#endif /* HP_CUPTI_ENABLED */
