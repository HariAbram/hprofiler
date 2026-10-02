/*
 * Feeds synthetic CUPTI activity records and Callback-API data -- built with
 * the real cupti.h structs -- through hooks/cuda_hook/cupti_trace.c's own
 * decoding code, and prints the wire lines it emits (one per line, in the
 * exact format the Runner parses). Driven by
 * tests/integration/test_native_gpu_records.py, which asserts on the output.
 *
 * Build: gcc -DHP_HAVE_CUPTI -DHP_CUPTI_UNIT_TEST [-DHP_CUPTI_MEMCPY6 -DHP_CUPTI_SYNC2]
 *        -I<cupti include> -Ihooks/cuda_hook this.c hooks/cuda_hook/cupti_trace.c -ldl -lpthread
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cupti.h>
#include "hp_cupti.h"

__thread int hp_cuda_in_hook = 0;

void hp_cuda_emit_span(const char *cat, pid_t tid, uint64_t start_ns, uint64_t dur_ns,
                       const char *name, const char *extra) {
    printf("span:%s:4242:%d:%llu:%llu:%s:%s\n", cat, (int)tid,
           (unsigned long long)start_ns, (unsigned long long)dur_ns, name, extra);
}
void hp_cuda_emit_line(const char *line) {
    /* status lines carry the real pid; normalise for the test */
    const char *rest = strchr(line + 7, ':');
    printf("gpuact:4242%s", rest ? rest : "\n");
}
void hp_cuda_pc_record(const void *record, uint32_t kind) {
    (void)record;
    printf("pcrecord:%u\n", kind);
}

#ifdef HP_CUPTI_MEMCPY6
typedef CUpti_ActivityMemcpy6 memcpy_rec_t;
#else
typedef CUpti_ActivityMemcpy5 memcpy_rec_t;
#endif
#ifdef HP_CUPTI_SYNC2
typedef CUpti_ActivitySynchronization2 sync_rec_t;
#else
typedef CUpti_ActivitySynchronization sync_rec_t;
#endif

static void kernel(uint32_t corr, uint32_t stream, uint64_t start, uint64_t end,
                   const char *name, uint64_t queued, uint64_t submitted, uint32_t graph) {
    CUpti_ActivityKernel9 k;
    memset(&k, 0, sizeof(k));
    k.kind = CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL;
    k.start = start; k.end = end;
    k.deviceId = 0; k.contextId = 1; k.streamId = stream;
    k.gridX = 64; k.gridY = 1; k.gridZ = 1; k.blockX = 256; k.blockY = 1; k.blockZ = 1;
    k.correlationId = corr;
    k.name = name;
    k.queued = queued; k.submitted = submitted;
    k.graphId = graph;
    hp_cupti_test_handle_record(&k);
}

static void cb(int runtime, CUpti_ApiCallbackSite site, const char *fn, uint32_t corr, uint64_t *slot) {
    CUpti_CallbackData d;
    memset(&d, 0, sizeof(d));
    d.callbackSite = site;
    d.functionName = fn;
    d.correlationId = corr;
    d.correlationData = slot;
    hp_cupti_test_api_callback(runtime, &d);
}

int main(void) {
    /* Clock: CUPTI timestamp callback active -> timestamps pass through. */
    hp_cupti_test_set_clock(1, 0);

    /* Two kernels overlapping on different streams (concurrency). */
    kernel(101, 7, 1000000, 1500000, "_Z4axpyPf", 0, 0, 0);
    kernel(102, 8, 1200000, 1900000, "_Z4scalePf", 0, 0, 0);

    /* Latency timestamps on: queued/submitted reported (mapped). */
    hp_cupti_test_set_latency(1);
    kernel(103, 7, 2000000, 2100000, "_Z4axpyPf", 1900000, 1950000, 0);
    /* submitted after start -> inconsistent, not reported */
    kernel(104, 7, 2200000, 2300000, "_Z4axpyPf", 2150000, 2250000, 0);
    hp_cupti_test_set_latency(0);

    /* Graph node kernel carries the graph id. */
    kernel(105, 9, 2400000, 2450000, "_Z9graphnodev", 0, 0, 77);

    /* Async HtoD copy launched via the runtime: driver id + runtime id. */
    memcpy_rec_t m;
    memset(&m, 0, sizeof(m));
    m.kind = CUPTI_ACTIVITY_KIND_MEMCPY;
    m.copyKind = CUPTI_ACTIVITY_MEMCPY_KIND_HTOD;
    m.flags = CUPTI_ACTIVITY_FLAG_MEMCPY_ASYNC;
    m.bytes = 1048576; m.start = 3000000; m.end = 3400000;
    m.deviceId = 0; m.contextId = 1; m.streamId = 8;
    m.correlationId = 202; m.runtimeCorrelationId = 201;
    hp_cupti_test_handle_record(&m);

    CUpti_ActivityMemset4 s;
    memset(&s, 0, sizeof(s));
    s.kind = CUPTI_ACTIVITY_KIND_MEMSET;
    s.bytes = 4096; s.start = 3500000; s.end = 3510000;
    s.deviceId = 0; s.contextId = 1; s.streamId = 7; s.correlationId = 203;
    hp_cupti_test_handle_record(&s);

    CUpti_ActivityMemcpyPtoP4 pp;
    memset(&pp, 0, sizeof(pp));
    pp.kind = CUPTI_ACTIVITY_KIND_MEMCPY2;
    pp.copyKind = CUPTI_ACTIVITY_MEMCPY_KIND_PTOP;
    pp.bytes = 8192; pp.start = 3600000; pp.end = 3650000;
    pp.deviceId = 0; pp.contextId = 1; pp.streamId = 7;
    pp.srcDeviceId = 0; pp.dstDeviceId = 1; pp.correlationId = 204;
    hp_cupti_test_handle_record(&pp);

    sync_rec_t y;
    memset(&y, 0, sizeof(y));
    y.kind = CUPTI_ACTIVITY_KIND_SYNCHRONIZATION;
    y.type = CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_STREAM_SYNCHRONIZE;
    y.start = 3700000; y.end = 3800000; y.correlationId = 301; y.contextId = 1;
    y.streamId = 7; y.cudaEventId = (uint32_t)CUPTI_SYNCHRONIZATION_INVALID_VALUE;
    hp_cupti_test_handle_record(&y);
    /* A sync record caused by the hook's own (internal) CUDA call is dropped. */
    hp_cupti_test_note_internal(399);
    y.type = CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_EVENT_SYNCHRONIZE;
    y.correlationId = 399; y.cudaEventId = 3;
    hp_cupti_test_handle_record(&y);
    y.cudaEventId = (uint32_t)CUPTI_SYNCHRONIZATION_INVALID_VALUE;
    y.type = CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_CONTEXT_SYNCHRONIZE;
    y.correlationId = 302; y.streamId = (uint32_t)CUPTI_SYNCHRONIZATION_INVALID_VALUE;
    hp_cupti_test_handle_record(&y);

    /* Incomplete (forced-flush) and corrupt records are counted, not emitted. */
    kernel(401, 7, 0, 0, "_Z10unfinishedv", 0, 0, 0);
    kernel(402, 7, 5000000, 4000000, "_Z7corruptv", 0, 0, 0);

    /* PC-sampling records are routed to cuda_hook.c's tables. */
    CUpti_ActivityPCSampling3 pcs;
    memset(&pcs, 0, sizeof(pcs));
    pcs.kind = CUPTI_ACTIVITY_KIND_PC_SAMPLING;
    hp_cupti_test_handle_record(&pcs);

    /* Buffer overflow: CUPTI reports dropped records per buffer. */
    hp_cupti_test_report_dropped(37);
    hp_cupti_test_report_dropped(5);

    /* Offset clock mode: CUPTI (CLOCK_REALTIME) timestamps are shifted. */
    hp_cupti_test_set_clock(0, -1000);
    kernel(501, 7, 9000000, 9100000, "_Z7shiftedv", 0, 0, 0);
    hp_cupti_test_set_clock(1, 0);

    /* ── Callback API ── */
    uint32_t c1 = 0, c2 = 0;
    uint64_t slot_a = 0, slot_b = 0, slot_c = 0;

    /* Intercepted runtime call: runtime id + nested driver id captured. */
    hp_cupti_arm();
    cb(1, CUPTI_API_ENTER, "cudaLaunchKernel_v7000", 600, &slot_a);
    cb(0, CUPTI_API_ENTER, "cuLaunchKernel", 601, &slot_b);
    cb(0, CUPTI_API_EXIT,  "cuLaunchKernel", 601, &slot_b);
    cb(1, CUPTI_API_EXIT,  "cudaLaunchKernel_v7000", 600, &slot_a);
    hp_cupti_disarm(&c1, &c2);
    printf("captured:%u:%u\n", c1, c2);

    /* Intercepted driver call: only a driver id. */
    hp_cupti_arm();
    cb(0, CUPTI_API_ENTER, "cuMemcpyHtoDAsync_v2", 610, &slot_a);
    cb(0, CUPTI_API_EXIT,  "cuMemcpyHtoDAsync_v2", 610, &slot_a);
    hp_cupti_disarm(&c1, &c2);
    printf("captured:%u:%u\n", c1, c2);

    /* Un-intercepted launch (e.g. a library with a static runtime): one host
     * span from the runtime callback, none for its nested driver call. */
    slot_a = slot_b = 0;
    cb(1, CUPTI_API_ENTER, "cudaLaunchKernel_v7000", 700, &slot_a);
    cb(0, CUPTI_API_ENTER, "cuLaunchKernel", 701, &slot_b);
    cb(0, CUPTI_API_EXIT,  "cuLaunchKernel", 701, &slot_b);
    cb(1, CUPTI_API_EXIT,  "cudaLaunchKernel_v7000", 700, &slot_a);

    /* Un-intercepted async and blocking copies. */
    slot_a = 0;
    cb(1, CUPTI_API_ENTER, "cudaMemcpyAsync_v3020", 710, &slot_a);
    cb(1, CUPTI_API_EXIT,  "cudaMemcpyAsync_v3020", 710, &slot_a);
    slot_a = 0;
    cb(1, CUPTI_API_ENTER, "cudaMemcpy_v3020", 711, &slot_a);
    cb(1, CUPTI_API_EXIT,  "cudaMemcpy_v3020", 711, &slot_a);

    /* Sync calls are not emitted from callbacks. */
    slot_a = 0;
    cb(1, CUPTI_API_ENTER, "cudaStreamSynchronize_v3020", 720, &slot_a);
    cb(1, CUPTI_API_EXIT,  "cudaStreamSynchronize_v3020", 720, &slot_a);

    /* The hook's own CUDA calls (in_hook) are never emitted. */
    hp_cuda_in_hook = 1;
    slot_c = 0;
    cb(1, CUPTI_API_ENTER, "cudaLaunchKernel_v7000", 730, &slot_c);
    cb(1, CUPTI_API_EXIT,  "cudaLaunchKernel_v7000", 730, &slot_c);
    hp_cuda_in_hook = 0;

    /* A driver-API launch made directly by the application. */
    slot_a = 0;
    cb(0, CUPTI_API_ENTER, "cuLaunchKernel", 740, &slot_a);
    cb(0, CUPTI_API_EXIT,  "cuLaunchKernel", 740, &slot_a);

    /* Static-runtime mode (HP_CUPTI_CB_SYNCS): syncs and stream waits too. */
    hp_cupti_test_set_cb_syncs(1);
    slot_a = 0;
    cb(1, CUPTI_API_ENTER, "cudaStreamSynchronize_v3020", 750, &slot_a);
    cb(1, CUPTI_API_EXIT,  "cudaStreamSynchronize_v3020", 750, &slot_a);
    slot_a = 0;
    cb(1, CUPTI_API_ENTER, "cudaStreamWaitEvent_v3020", 751, &slot_a);
    cb(1, CUPTI_API_EXIT,  "cudaStreamWaitEvent_v3020", 751, &slot_a);
    return 0;
}
