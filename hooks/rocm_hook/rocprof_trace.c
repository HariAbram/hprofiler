/*
 * Native ROCm device-activity tracing through ROCprofiler-SDK buffered
 * tracing (kernel dispatches, memory copies). Nothing is linked against
 * librocprofiler-sdk: this file only defines the `rocprofiler_configure`
 * entry point. When the HIP runtime starts, rocprofiler-register finds that
 * symbol in the process (we are LD_PRELOADed), loads librocprofiler-sdk and
 * calls us; every SDK function is then resolved with dlsym. Without the SDK
 * nothing calls rocprofiler_configure and rocm_hook.c keeps its hipEvent
 * proxy timing -- the library still loads and runs.
 *
 * Output (wire `span:` lines, see src/core/gpu_activity.py):
 *   KERNEL_DISPATCH -> category rocm,   type=kernel  (name from the code-
 *                      object kernel-symbol callback, ".kd" stripped)
 *   MEMORY_COPY     -> category memory, type=memcpy
 * tagged side=gpu,rt=rocm,src=rocprofiler,timing=device, corr= (internal
 * correlation id), lid= (the hook's id, pushed as the EXTERNAL correlation
 * id around each intercepted HIP call -- exact host/device matching), dev=
 * (agent handle), queue=, dispatch=, and the launching thread as tid.
 * Status: `gpuact:<pid>:rocprofiler:...` (status, clock, dropped, and
 * final_flush=1 after the forced flush at SDK finalization).
 *
 * Buffer policy is DISCARD: if the buffer fills faster than it is drained,
 * the SDK drops records and reports how many (dropped= in the status line)
 * rather than blocking the runtime's completion handling.
 *
 * Clock: SDK timestamps are mapped onto CLOCK_MONOTONIC by an offset
 * measured around rocprofiler_get_timestamp (re-measured per buffer;
 * clock_err_ns = half the tightest bracket).
 *
 * Compatibility: records carry their own size; fields beyond what the
 * running SDK wrote are never read. The buffer-tracing kind values compiled
 * in are checked against the running SDK's own names before use. Never run
 * on an AMD GPU (decoder tested with synthetic records only).
 */
#define _GNU_SOURCE
#include "hp_rocprof.h"

#include <dlfcn.h>
#include <pthread.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#if defined(HP_HAVE_ROCPROFILER_SDK) && !defined(HP_NO_ROCPROFILER)
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/context.h>
#include <rocprofiler-sdk/external_correlation.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>
#define HP_ROC_ENABLED 1
#else
#define HP_ROC_ENABLED 0
#endif

__attribute__((unused)) static uint64_t mono_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

__attribute__((unused)) static void emit_status(const char *fields) {
    char line[512];
    int n = snprintf(line, sizeof(line), "gpuact:%d:rocprofiler:%s\n", (int)getpid(), fields);
    if (n > 0 && n < (int)sizeof(line))
        hp_roc_emit_line(line);
}

int hp_roc_compiled(void) { return HP_ROC_ENABLED; }

#if !HP_ROC_ENABLED

int  hp_roc_active(void) { return 0; }
void hp_roc_push(uint64_t lid) { (void)lid; }
void hp_roc_pop(void) {}
void hp_roc_flush(void) {}

#else

static struct {
    __typeof__(&rocprofiler_create_context)                     create_context;
    __typeof__(&rocprofiler_start_context)                      start_context;
    __typeof__(&rocprofiler_create_buffer)                      create_buffer;
    __typeof__(&rocprofiler_flush_buffer)                       flush_buffer;
    __typeof__(&rocprofiler_configure_buffer_tracing_service)   buffer_tracing;
    __typeof__(&rocprofiler_configure_callback_tracing_service) callback_tracing;
    __typeof__(&rocprofiler_push_external_correlation_id)       push_ext;
    __typeof__(&rocprofiler_pop_external_correlation_id)        pop_ext;
    __typeof__(&rocprofiler_get_timestamp)                      get_timestamp;
    __typeof__(&rocprofiler_query_buffer_tracing_kind_name)     buffer_kind_name;
} p;

static rocprofiler_context_id_t g_ctx;
static rocprofiler_buffer_id_t  g_buffer;
static int      g_active = 0;
static int64_t  g_offset_ns = 0;
static uint64_t g_offset_err_ns = 0;
static int      g_clock_ok = 0;
static uint64_t g_bad = 0, g_notime = 0;
static __thread int t_pushed;

int hp_roc_active(void) { return __atomic_load_n(&g_active, __ATOMIC_ACQUIRE); }

/* ── kernel-name table (kernel_id -> name), filled by the code-object
 *    callback before any dispatch of that kernel can complete ────────────── */
typedef struct { uint64_t id; char *name; } kname_t;
static kname_t        *g_kn = NULL;
static size_t          g_kn_cap = 0, g_kn_n = 0;
static pthread_mutex_t g_kn_mu = PTHREAD_MUTEX_INITIALIZER;

static size_t kn_slot(kname_t *tab, size_t cap, uint64_t id) {
    size_t i = (size_t)((id * 0x9E3779B97F4A7C15ULL) >> 7) & (cap - 1);
    while (tab[i].name && tab[i].id != id) i = (i + 1) & (cap - 1);
    return i;
}

static void kname_put(uint64_t id, const char *name) {
    if (!name) return;
    size_t len = strlen(name);
    if (len > 3 && !strcmp(name + len - 3, ".kd")) len -= 3;   /* descriptor symbol */
    char *copy = strndup(name, len);
    if (!copy) return;
    pthread_mutex_lock(&g_kn_mu);
    if ((g_kn_n + 1) * 2 > g_kn_cap) {
        size_t ncap = g_kn_cap ? g_kn_cap * 2 : 1024;
        kname_t *nt = calloc(ncap, sizeof(kname_t));
        if (!nt) { pthread_mutex_unlock(&g_kn_mu); free(copy); return; }
        for (size_t i = 0; i < g_kn_cap; i++)
            if (g_kn[i].name) nt[kn_slot(nt, ncap, g_kn[i].id)] = g_kn[i];
        free(g_kn);
        g_kn = nt;
        g_kn_cap = ncap;
    }
    size_t s = kn_slot(g_kn, g_kn_cap, id);
    if (g_kn[s].name) free(g_kn[s].name); else g_kn_n++;
    g_kn[s].id = id;
    g_kn[s].name = copy;
    pthread_mutex_unlock(&g_kn_mu);
}

/* Copies into buf (the table may grow concurrently). */
static const char *kname_get(uint64_t id, char *buf, size_t cap) {
    const char *out = NULL;
    pthread_mutex_lock(&g_kn_mu);
    if (g_kn_cap) {
        size_t s = kn_slot(g_kn, g_kn_cap, id);
        if (g_kn[s].name) { snprintf(buf, cap, "%s", g_kn[s].name); out = buf; }
    }
    pthread_mutex_unlock(&g_kn_mu);
    if (!out) { snprintf(buf, cap, "kernel_%llu", (unsigned long long)id); out = buf; }
    return out;
}

static void code_object_cb(rocprofiler_callback_tracing_record_t record,
                           rocprofiler_user_data_t *user_data, void *data) {
    (void)user_data; (void)data;
    if (record.kind != ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT ||
        record.operation != ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER ||
        record.phase != ROCPROFILER_CALLBACK_PHASE_LOAD || !record.payload)
        return;
    const rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t *d = record.payload;
    kname_put(d->kernel_id, d->kernel_name);
}

/* ── clock ───────────────────────────────────────────────────────────────── */

static void calibrate_offset(void) {
    if (!p.get_timestamp) return;
    uint64_t best = UINT64_MAX;
    int64_t off = g_offset_ns;
    for (int i = 0; i < 5; i++) {
        rocprofiler_timestamp_t t = 0;
        uint64_t a = mono_ns();
        if (p.get_timestamp(&t) != ROCPROFILER_STATUS_SUCCESS) return;
        uint64_t b = mono_ns();
        if (b - a < best) { best = b - a; off = (int64_t)(a + (b - a) / 2) - (int64_t)t; }
    }
    g_offset_ns = off;
    g_offset_err_ns = best / 2 + 1;
    g_clock_ok = 1;
}

static uint64_t to_mono(uint64_t t) { return (uint64_t)((int64_t)t + g_offset_ns); }

/* ── records -> wire ─────────────────────────────────────────────────────── */

#define HP_MAX_PLAUSIBLE_NS 3600000000000ULL
#define HAS_FIELD(rec, type, field) \
    ((rec)->size >= offsetof(type, field) + sizeof(((type *)0)->field))

#define APPEND(buf, n, ...) do { \
    if ((n) >= 0 && (size_t)(n) < sizeof(buf)) \
        (n) += snprintf((buf) + (n), sizeof(buf) - (size_t)(n), __VA_ARGS__); \
} while (0)

static int plausible(uint64_t start, uint64_t end) {
    if (start == 0 && end == 0) { __atomic_add_fetch(&g_notime, 1, __ATOMIC_RELAXED); return 0; }
    if (end < start || end - start > HP_MAX_PLAUSIBLE_NS) {
        __atomic_add_fetch(&g_bad, 1, __ATOMIC_RELAXED);
        return 0;
    }
    return 1;
}

static void emit_dispatch(const rocprofiler_buffer_tracing_kernel_dispatch_record_t *r) {
    typedef rocprofiler_buffer_tracing_kernel_dispatch_record_t R;
    if (!HAS_FIELD(r, R, dispatch_info) ||
        r->dispatch_info.size < offsetof(rocprofiler_kernel_dispatch_info_t, grid_size)
                                + sizeof(rocprofiler_dim3_t)) {
        __atomic_add_fetch(&g_bad, 1, __ATOMIC_RELAXED);
        return;
    }
    if (!plausible(r->start_timestamp, r->end_timestamp)) return;
    const rocprofiler_kernel_dispatch_info_t *d = &r->dispatch_info;
    uint64_t s = to_mono(r->start_timestamp), e = to_mono(r->end_timestamp);
    /* grid_size counts work-items; report HIP-style blocks like the hook. */
    rocprofiler_dim3_t wg = d->workgroup_size, gs = d->grid_size;
    unsigned gx = wg.x ? gs.x / wg.x : gs.x, gy = wg.y ? gs.y / wg.y : gs.y,
             gz = wg.z ? gs.z / wg.z : gs.z;
    char name[512], x[512];
    kname_get(d->kernel_id, name, sizeof(name));
    int n = 0;
    APPEND(x, n, "type=kernel,side=gpu,rt=rocm,op=kernel,timing=device,src=rocprofiler,"
                 "corr=%llu,dev=%llu,queue=%llu,dispatch=%llu,grid=%ux%ux%u,block=%ux%ux%u",
           (unsigned long long)r->correlation_id.internal,
           (unsigned long long)d->agent_id.handle, (unsigned long long)d->queue_id.handle,
           (unsigned long long)d->dispatch_id, gx, gy, gz, wg.x, wg.y, wg.z);
    if (r->correlation_id.external.value)
        APPEND(x, n, ",lid=%llu", (unsigned long long)r->correlation_id.external.value);
    hp_roc_emit_span("rocm", (pid_t)r->thread_id, s, e - s, name, x);
}

static const char *copy_dir(int op) {
    switch (op) {
    case ROCPROFILER_MEMORY_COPY_HOST_TO_HOST:     return "HtoH";
    case ROCPROFILER_MEMORY_COPY_HOST_TO_DEVICE:   return "HtoD";
    case ROCPROFILER_MEMORY_COPY_DEVICE_TO_HOST:   return "DtoH";
    case ROCPROFILER_MEMORY_COPY_DEVICE_TO_DEVICE: return "DtoD";
    default:                                       return "unknown";
    }
}

static void emit_copy(const rocprofiler_buffer_tracing_memory_copy_record_t *r) {
    typedef rocprofiler_buffer_tracing_memory_copy_record_t R;
    if (!HAS_FIELD(r, R, bytes)) { __atomic_add_fetch(&g_bad, 1, __ATOMIC_RELAXED); return; }
    if (!plausible(r->start_timestamp, r->end_timestamp)) return;
    uint64_t s = to_mono(r->start_timestamp), e = to_mono(r->end_timestamp);
    const char *dir = copy_dir((int)r->operation);
    char name[32], x[512];
    snprintf(name, sizeof(name), "memcpy %s", dir);
    int n = 0;
    APPEND(x, n, "type=memcpy,side=gpu,rt=rocm,op=memcpy,timing=device,src=rocprofiler,"
                 "dir=%s,bytes=%llu,corr=%llu,dev=%llu,src_dev=%llu,dst_dev=%llu",
           dir, (unsigned long long)r->bytes, (unsigned long long)r->correlation_id.internal,
           (unsigned long long)r->dst_agent_id.handle, (unsigned long long)r->src_agent_id.handle,
           (unsigned long long)r->dst_agent_id.handle);
    if (r->correlation_id.external.value)
        APPEND(x, n, ",lid=%llu", (unsigned long long)r->correlation_id.external.value);
    hp_roc_emit_span("memory", (pid_t)r->thread_id, s, e - s, name, x);
}

static void report_counters(uint64_t dropped) {
    static uint64_t last_bad = 0, last_notime = 0;
    uint64_t bd = __atomic_load_n(&g_bad, __ATOMIC_RELAXED);
    uint64_t nt = __atomic_load_n(&g_notime, __ATOMIC_RELAXED);
    char f[200];
    int n = 0;
    f[0] = '\0';
    if (dropped)          APPEND(f, n, "%sdropped=%llu", n ? "," : "", (unsigned long long)dropped);
    if (nt > last_notime) APPEND(f, n, "%snotime=%llu", n ? "," : "", (unsigned long long)(nt - last_notime));
    if (bd > last_bad)    APPEND(f, n, "%sbad_records=%llu", n ? "," : "", (unsigned long long)(bd - last_bad));
    last_bad = bd; last_notime = nt;
    if (n > 0) emit_status(f);
}

static void buffer_cb(rocprofiler_context_id_t ctx, rocprofiler_buffer_id_t buf,
                      rocprofiler_record_header_t **headers, size_t num_headers,
                      void *data, uint64_t drop_count) {
    (void)ctx; (void)buf; (void)data;
    int saved = hp_roc_in_hook;
    hp_roc_in_hook = 1;
    calibrate_offset();
    for (size_t i = 0; i < num_headers; i++) {
        const rocprofiler_record_header_t *h = headers[i];
        if (!h || !h->payload || h->category != ROCPROFILER_BUFFER_CATEGORY_TRACING) continue;
        if (h->kind == ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH)
            emit_dispatch(h->payload);
        else if (h->kind == ROCPROFILER_BUFFER_TRACING_MEMORY_COPY)
            emit_copy(h->payload);
    }
    report_counters(drop_count);
    hp_roc_in_hook = saved;
}

/* ── registration ────────────────────────────────────────────────────────── */

static void *sdk_handle(void) {
    static const char *names[] = {"librocprofiler-sdk.so.1", "librocprofiler-sdk.so", NULL};
    for (int i = 0; names[i]; i++) {
        void *h = dlopen(names[i], RTLD_LAZY | RTLD_NOLOAD);
        if (h) return h;
    }
    return RTLD_DEFAULT;
}

/* The compiled-in kind values must mean the same thing in the running SDK. */
static int kind_is(rocprofiler_buffer_tracing_kind_t kind, const char *expect) {
    if (!p.buffer_kind_name) return 1;
    const char *name = NULL;
    uint64_t len = 0;
    if (p.buffer_kind_name(kind, &name, &len) != ROCPROFILER_STATUS_SUCCESS || !name) return 0;
    return strstr(name, expect) != NULL;
}

#define FAIL(reason) do { emit_status("status=unavailable,reason=" reason); return -1; } while (0)
#define SYM(field, name) (p.field = (__typeof__(p.field))dlsym(h, name))

static int tool_init(rocprofiler_client_finalize_t fini, void *tool_data) {
    (void)fini; (void)tool_data;
    void *h = sdk_handle();
    if (!SYM(create_context, "rocprofiler_create_context") |
        !SYM(start_context, "rocprofiler_start_context") |
        !SYM(create_buffer, "rocprofiler_create_buffer") |
        !SYM(flush_buffer, "rocprofiler_flush_buffer") |
        !SYM(buffer_tracing, "rocprofiler_configure_buffer_tracing_service") |
        !SYM(callback_tracing, "rocprofiler_configure_callback_tracing_service") |
        !SYM(push_ext, "rocprofiler_push_external_correlation_id") |
        !SYM(pop_ext, "rocprofiler_pop_external_correlation_id"))
        FAIL("sdk_symbols_missing");
    SYM(get_timestamp, "rocprofiler_get_timestamp");
    SYM(buffer_kind_name, "rocprofiler_query_buffer_tracing_kind_name");
    if (!kind_is(ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH, "KERNEL_DISPATCH") ||
        !kind_is(ROCPROFILER_BUFFER_TRACING_MEMORY_COPY, "MEMORY_COPY"))
        FAIL("sdk_kind_values_differ_from_headers");

    if (p.create_context(&g_ctx) != ROCPROFILER_STATUS_SUCCESS)
        FAIL("create_context_failed");
    rocprofiler_tracing_operation_t sym_op = ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER;
    p.callback_tracing(g_ctx, ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT, &sym_op, 1,
                       code_object_cb, NULL);   /* names only; dispatches still trace without it */
    if (p.create_buffer(g_ctx, 8u << 20, 4u << 20, ROCPROFILER_BUFFER_POLICY_DISCARD,
                        buffer_cb, NULL, &g_buffer) != ROCPROFILER_STATUS_SUCCESS)
        FAIL("create_buffer_failed");
    if (p.buffer_tracing(g_ctx, ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH, NULL, 0, g_buffer)
            != ROCPROFILER_STATUS_SUCCESS)
        FAIL("kernel_dispatch_tracing_failed");
    p.buffer_tracing(g_ctx, ROCPROFILER_BUFFER_TRACING_MEMORY_COPY, NULL, 0, g_buffer);
    calibrate_offset();
    if (p.start_context(g_ctx) != ROCPROFILER_STATUS_SUCCESS)
        FAIL("start_context_failed");
    __atomic_store_n(&g_active, 1, __ATOMIC_RELEASE);

    char f[200];
    if (g_clock_ok)
        snprintf(f, sizeof(f), "status=active,clock=offset,offset_ns=%lld,clock_err_ns=%llu,"
                 "correlation=external,final_marker=1", (long long)g_offset_ns, (unsigned long long)g_offset_err_ns);
    else
        snprintf(f, sizeof(f), "status=active,clock=unmapped,correlation=external,final_marker=1");
    emit_status(f);
    return 0;
}

/* Forced final flush at SDK finalization. The final_flush=1 status after
 * it is the collector's evidence that buffered device records were
 * delivered: an active tracer whose process ends without it (crash,
 * _exit, kill) is reported as possibly missing device records. */
static void tool_fini(void *tool_data) {
    (void)tool_data;
    if (hp_roc_active()) {
        p.flush_buffer(g_buffer);
        emit_status("final_flush=1");
    }
    __atomic_store_n(&g_active, 0, __ATOMIC_RELEASE);
}

__attribute__((visibility("default")))
rocprofiler_tool_configure_result_t *rocprofiler_configure(uint32_t version,
                                                           const char *runtime_version,
                                                           uint32_t priority,
                                                           rocprofiler_client_id_t *client_id) {
    (void)runtime_version; (void)priority;
    const char *m = getenv("HPROFILER_DEVICE_ACTIVITY");
    if (m && !strcmp(m, "off")) {
        emit_status("status=disabled,reason=env_off");
        return NULL;
    }
    if (version / 10000 < 1 && (version % 10000) / 100 < 4) {   /* < 0.4: pre-release API */
        emit_status("status=unavailable,reason=rocprofiler_sdk_too_old");
        return NULL;
    }
    client_id->name = "hprofiler";
    static rocprofiler_tool_configure_result_t cfg = {
        sizeof(rocprofiler_tool_configure_result_t), tool_init, tool_fini, NULL
    };
    return &cfg;
}

void hp_roc_push(uint64_t lid) {
    t_pushed = 0;
    if (!hp_roc_active()) return;
    rocprofiler_user_data_t d = {.value = lid};
    t_pushed = p.push_ext(g_ctx, (rocprofiler_thread_id_t)syscall(SYS_gettid), d)
               == ROCPROFILER_STATUS_SUCCESS;
}

void hp_roc_pop(void) {
    if (!t_pushed) return;
    rocprofiler_user_data_t d;
    p.pop_ext(g_ctx, (rocprofiler_thread_id_t)syscall(SYS_gettid), &d);
    t_pushed = 0;
}

void hp_roc_flush(void) {
    if (hp_roc_active()) p.flush_buffer(g_buffer);
}

#ifdef HP_ROCPROF_UNIT_TEST
void hp_roc_test_buffer(void **headers, uint64_t n, uint64_t drop_count) {
    rocprofiler_context_id_t c = {0};
    rocprofiler_buffer_id_t b = {0};
    buffer_cb(c, b, (rocprofiler_record_header_t **)headers, (size_t)n, NULL, drop_count);
}
void hp_roc_test_kernel_symbol(uint64_t kernel_id, const char *name) {
    rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t d;
    memset(&d, 0, sizeof(d));
    d.size = sizeof(d);
    d.kernel_id = kernel_id;
    d.kernel_name = name;
    rocprofiler_callback_tracing_record_t rec;
    memset(&rec, 0, sizeof(rec));
    rec.kind = ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT;
    rec.operation = ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER;
    rec.phase = ROCPROFILER_CALLBACK_PHASE_LOAD;
    rec.payload = &d;
    code_object_cb(rec, NULL, NULL);
}
void hp_roc_test_set_clock_offset(int64_t offset_ns) { g_offset_ns = offset_ns; }
#endif

#endif /* HP_ROC_ENABLED */
