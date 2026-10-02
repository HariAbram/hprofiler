/*
 * Feeds synthetic ROCprofiler-SDK buffer records (real SDK structs) through
 * hooks/rocm_hook/rocprof_trace.c's buffer callback and kernel-symbol
 * callback, printing the wire lines it emits. Driven by
 * tests/integration/test_native_gpu_records.py.
 *
 * Build: gcc -D__HIP_PLATFORM_AMD__ -DHP_HAVE_ROCPROFILER_SDK -DHP_ROCPROF_UNIT_TEST
 *        -I<rocm>/include -Ihooks/rocm_hook this.c hooks/rocm_hook/rocprof_trace.c -ldl -lpthread
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <rocprofiler-sdk/rocprofiler.h>
#include "hp_rocprof.h"

__thread int hp_roc_in_hook = 0;

void hp_roc_emit_span(const char *cat, pid_t tid, uint64_t start_ns, uint64_t dur_ns,
                      const char *name, const char *extra) {
    printf("span:%s:4242:%d:%llu:%llu:%s:%s\n", cat, (int)tid,
           (unsigned long long)start_ns, (unsigned long long)dur_ns, name, extra);
}
void hp_roc_emit_line(const char *line) {
    const char *rest = strchr(line + 7, ':');
    printf("gpuact:4242%s", rest ? rest : "\n");
}

typedef rocprofiler_buffer_tracing_kernel_dispatch_record_t kd_t;
typedef rocprofiler_buffer_tracing_memory_copy_record_t mc_t;

static kd_t dispatch(uint64_t corr, uint64_t lid, uint64_t tid, uint64_t queue,
                     uint64_t kernel_id, uint64_t start, uint64_t end) {
    kd_t r;
    memset(&r, 0, sizeof(r));
    r.size = sizeof(r);
    r.kind = ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH;
    r.operation = ROCPROFILER_KERNEL_DISPATCH_COMPLETE;
    r.correlation_id.internal = corr;
    r.correlation_id.external.value = lid;
    r.thread_id = tid;
    r.start_timestamp = start;
    r.end_timestamp = end;
    r.dispatch_info.size = sizeof(r.dispatch_info);
    r.dispatch_info.agent_id.handle = 11;
    r.dispatch_info.queue_id.handle = queue;
    r.dispatch_info.kernel_id = kernel_id;
    r.dispatch_info.dispatch_id = corr + 1000;
    r.dispatch_info.workgroup_size.x = 256; r.dispatch_info.workgroup_size.y = 1;
    r.dispatch_info.workgroup_size.z = 1;
    r.dispatch_info.grid_size.x = 256 * 64; r.dispatch_info.grid_size.y = 1;
    r.dispatch_info.grid_size.z = 1;
    return r;
}

int main(void) {
    hp_roc_test_set_clock_offset(500);   /* SDK clock -> CLOCK_MONOTONIC */
    hp_roc_test_kernel_symbol(1, "_Z4axpyPf.kd");
    hp_roc_test_kernel_symbol(2, "_Z5scalePf");

    /* Concurrent dispatches on two queues, out of order in the buffer. */
    kd_t k2 = dispatch(21, 2, 900, 0xB0, 2, 1200000, 1900000);
    kd_t k1 = dispatch(20, 1, 900, 0xA0, 1, 1000000, 1500000);
    kd_t k3 = dispatch(22, 0, 901, 0xA0, 99, 2000000, 2100000);  /* no external id, unknown symbol */

    mc_t m;
    memset(&m, 0, sizeof(m));
    m.size = sizeof(m);
    m.kind = ROCPROFILER_BUFFER_TRACING_MEMORY_COPY;
    m.operation = ROCPROFILER_MEMORY_COPY_HOST_TO_DEVICE;
    m.correlation_id.internal = 23;
    m.correlation_id.external.value = 3;
    m.thread_id = 900;
    m.start_timestamp = 3000000; m.end_timestamp = 3400000;
    m.src_agent_id.handle = 10; m.dst_agent_id.handle = 11;
    m.bytes = 1048576;

    /* A record from an older SDK that is too short for dispatch_info is
     * skipped and counted, never read past its end. */
    kd_t short_rec = dispatch(24, 4, 900, 0xA0, 1, 3500000, 3600000);
    short_rec.size = offsetof(kd_t, dispatch_info);
    kd_t unfinished = dispatch(25, 5, 900, 0xA0, 1, 0, 0);

    rocprofiler_record_header_t h[6];
    void *ptrs[6];
    void *payloads[6] = {&k2, &k1, &k3, &m, &short_rec, &unfinished};
    uint32_t kinds[6] = {ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH, ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH,
                         ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH, ROCPROFILER_BUFFER_TRACING_MEMORY_COPY,
                         ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH, ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH};
    for (int i = 0; i < 6; i++) {
        memset(&h[i], 0, sizeof(h[i]));
        h[i].category = ROCPROFILER_BUFFER_CATEGORY_TRACING;
        h[i].kind = kinds[i];
        h[i].payload = payloads[i];
        ptrs[i] = &h[i];
    }
    /* Buffer overflow: the SDK reports how many records it discarded. */
    hp_roc_test_buffer(ptrs, 6, 12);
    hp_roc_test_buffer(ptrs, 0, 3);
    return 0;
}
