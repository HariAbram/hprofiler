/*
 * Internal interface between rocm_hook.c (LD_PRELOAD interception, hand-
 * rolled HIP type stubs) and rocprof_trace.c (native device activity via
 * ROCprofiler-SDK buffered tracing, compiled against the SDK headers).
 * Everything here is hidden; the only symbol rocprof_trace.c exports is
 * rocprofiler_configure, which ROCprofiler-SDK looks up to find tools.
 */
#ifndef HP_ROCPROF_H
#define HP_ROCPROF_H

#include <stdint.h>
#include <sys/types.h>

#define HP_HIDDEN __attribute__((visibility("hidden")))

/* ── provided by rocm_hook.c (or by the unit-test harness) ─────────────── */
extern __thread int hp_roc_in_hook HP_HIDDEN;
HP_HIDDEN void hp_roc_emit_span(const char *cat, pid_t tid, uint64_t start_ns,
                                uint64_t dur_ns, const char *name, const char *extra);
HP_HIDDEN void hp_roc_emit_line(const char *line);

/* ── provided by rocprof_trace.c ───────────────────────────────────────── */

/* 1 when built with ROCprofiler-SDK headers. */
HP_HIDDEN int hp_roc_compiled(void);
/* 1 once ROCprofiler-SDK has called our rocprofiler_configure/initialize
 * and the tracing context is running. ROCprofiler-SDK does this itself
 * while the HIP runtime initializes. */
HP_HIDDEN int hp_roc_active(void);
/* Tag everything the calling thread submits between push and pop with
 * `lid` (ROCprofiler's external correlation id). No-ops when inactive. */
HP_HIDDEN void hp_roc_push(uint64_t lid);
HP_HIDDEN void hp_roc_pop(void);
HP_HIDDEN void hp_roc_flush(void);

#ifdef HP_ROCPROF_UNIT_TEST
/* Feed synthetic records through the real buffer callback. headers is a
 * rocprofiler_record_header_t** array. */
void hp_roc_test_buffer(void **headers, uint64_t n, uint64_t drop_count);
void hp_roc_test_kernel_symbol(uint64_t kernel_id, const char *name);
void hp_roc_test_set_clock_offset(int64_t offset_ns);
#endif

#endif /* HP_ROCPROF_H */
