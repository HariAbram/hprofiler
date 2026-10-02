/*
 * Internal interface between cuda_hook.c (LD_PRELOAD interception, hand-
 * rolled CUDA type stubs) and cupti_trace.c (native device-activity tracing,
 * compiled against the real CUPTI headers). They are separate translation
 * units because cupti.h pulls in cuda.h, whose types conflict with
 * cuda_hook.c's stubs. Everything here is hidden: none of it is exported
 * from libhprofiler_cuda.so.
 */
#ifndef HP_CUPTI_H
#define HP_CUPTI_H

#include <stdint.h>
#include <sys/types.h>

#define HP_HIDDEN __attribute__((visibility("hidden")))

/* ── provided by cuda_hook.c (or by the unit-test harness) ──────────────── */

/* Recursion guard: nonzero while the hook itself is making CUDA calls. */
extern __thread int hp_cuda_in_hook HP_HIDDEN;

HP_HIDDEN void hp_cuda_emit_span(const char *cat, pid_t tid, uint64_t start_ns,
                                 uint64_t dur_ns, const char *name, const char *extra);
/* One complete newline-terminated wire line (gpuact: status records). */
HP_HIDDEN void hp_cuda_emit_line(const char *line);
/* PC-sampling / function records, routed back to cuda_hook.c's existing
 * PC-sampling tables (CUPTI allows one buffer-callback registration per
 * process, so cupti_trace.c owns it for both features). */
HP_HIDDEN void hp_cuda_pc_record(const void *record, uint32_t kind);

/* ── provided by cupti_trace.c ─────────────────────────────────────────── */

#define HP_CUPTI_ACTIVITY   1u   /* kernel/memcpy/memset/sync activity + callbacks */
#define HP_CUPTI_PCSAMPLING 2u   /* PC sampling + function records */
#define HP_CUPTI_CB_SYNCS   4u   /* also emit host spans for sync calls from callbacks
                                    (static-runtime programs: no wrapper sees them) */

/* Loads libcupti with dlopen on first use and enables the requested
 * features. Returns the feature bits that are active afterwards (0 when
 * libcupti or the headers were unavailable -- a gpuact status line says
 * why). Safe to call repeatedly and from several threads. */
HP_HIDDEN unsigned hp_cupti_start(unsigned want);

/* Correlation capture around one intercepted call: arm immediately before
 * the real call, disarm immediately after. disarm returns the CUPTI
 * correlation ids of the outermost runtime-API and driver-API calls made in
 * between (0 when none / callbacks unavailable): *corr is the runtime id if
 * there was one, else the driver id; *corr2 the other one. */
HP_HIDDEN void hp_cupti_arm(void);
HP_HIDDEN void hp_cupti_disarm(uint32_t *corr, uint32_t *corr2);

/* Deliver every buffered activity record now (forced flush). A no-op after
 * the at-exit flush has run. */
HP_HIDDEN void hp_cupti_flush(void);

/* 1 when cupti_trace.c was compiled with CUPTI headers (it then owns the
 * process's single CUPTI buffer-callback registration). */
HP_HIDDEN int hp_cupti_compiled(void);

#ifdef HP_CUPTI_UNIT_TEST
/* Test entry points: feed one synthetic record / drop report through the
 * same code the buffer-completion callback uses. */
void hp_cupti_test_handle_record(const void *record);
void hp_cupti_test_report_dropped(uint64_t n);
void hp_cupti_test_set_clock(int monotonic_callback, int64_t offset_ns);
void hp_cupti_test_set_latency(int on);
void hp_cupti_test_note_internal(uint32_t corr);
void hp_cupti_test_set_cb_syncs(int on);
/* Invoke the Callback-API handler with a synthetic CUpti_CallbackData. */
void hp_cupti_test_api_callback(int runtime_domain, const void *callback_data);
#endif

#endif /* HP_CUPTI_H */
