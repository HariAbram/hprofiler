/*
 * OMPT tool (OpenMP 5.0 Tools Interface) for LLVM libomp, loaded through
 * OMP_TOOL_LIBRARIES (and LD_PRELOADed, so its dlopen interposer is global).
 *
 * Callbacks: parallel_begin/end (parallel_region, encountering thread),
 * implicit_task (omp_implicit_task: each thread's share of a region), work
 * (loops, sections, ...), sync_region (barriers, taskwait, taskgroup),
 * task_create / task_schedule (omp_task_create, omp_task), target (offload).
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

/* ── Minimal OMPT types matching omp-tools.h ────────────────────────── */
typedef void*    ompt_device_t;
typedef uint64_t ompt_id_t;

typedef union {
    uint64_t value;
    void    *ptr;
} ompt_data_t;

typedef enum {
    ompt_scope_begin = 1,
    ompt_scope_end   = 2,
    ompt_scope_beginend = 3,
} ompt_scope_endpoint_t;

typedef enum {
    ompt_thread_initial = 1,
    ompt_thread_worker  = 2,
    ompt_thread_other   = 3,
    ompt_thread_unknown = 4,
} ompt_thread_t;

typedef enum {
    ompt_sync_region_barrier                  = 1,
    ompt_sync_region_barrier_implicit         = 2,
    ompt_sync_region_barrier_explicit         = 3,
    ompt_sync_region_barrier_implementation   = 4,
    ompt_sync_region_taskwait                 = 5,
    ompt_sync_region_taskgroup                = 6,
    ompt_sync_region_reduction                = 7,
    ompt_sync_region_barrier_implicit_workshare = 8,
    ompt_sync_region_barrier_implicit_parallel  = 9,
    ompt_sync_region_barrier_teams              = 10,
} ompt_sync_region_t;

typedef enum {
    ompt_work_loop         = 1,
    ompt_work_sections     = 2,
    ompt_work_single_executor = 3,
    ompt_work_single_other    = 4,
    ompt_work_workshare    = 5,
    ompt_work_distribute   = 6,
    ompt_work_taskloop     = 7,
    ompt_work_scope        = 8,
    ompt_work_loop_static  = 10,
    ompt_work_loop_dynamic = 11,
    ompt_work_loop_guided  = 12,
    ompt_work_loop_other   = 13,
} ompt_work_t;

typedef enum {
    ompt_target               = 1,
    ompt_target_enter_data    = 2,
    ompt_target_exit_data     = 3,
    ompt_target_update        = 4,
    ompt_target_nowait        = 9,      /* 5.1: +8 for each nowait form */
    ompt_target_enter_data_nowait = 10,
    ompt_target_exit_data_nowait  = 11,
    ompt_target_update_nowait     = 12,
} ompt_target_t;

/* Callback event IDs (from omp-tools.h) */
typedef enum {
    ompt_callback_thread_begin   = 1,
    ompt_callback_thread_end     = 2,
    ompt_callback_parallel_begin = 3,
    ompt_callback_parallel_end   = 4,
    ompt_callback_task_create    = 5,
    ompt_callback_task_schedule  = 6,
    ompt_callback_implicit_task  = 7,
    ompt_callback_target         = 8,
    ompt_callback_mutex_released = 17,
    ompt_callback_dependences    = 18,
    ompt_callback_task_dependence = 19,
    ompt_callback_work           = 20,
    ompt_callback_sync_region    = 23,
    ompt_callback_lock_init      = 24,
    ompt_callback_lock_destroy   = 25,
    ompt_callback_mutex_acquire  = 26,
    ompt_callback_mutex_acquired = 27,
    ompt_callback_nest_lock      = 28,
} ompt_callbacks_t;

typedef enum {
    ompt_mutex_lock           = 1,
    ompt_mutex_test_lock      = 2,
    ompt_mutex_nest_lock      = 3,
    ompt_mutex_test_nest_lock = 4,
    ompt_mutex_critical       = 5,
    ompt_mutex_atomic         = 6,
    ompt_mutex_ordered        = 7,
} ompt_mutex_t;

typedef uint64_t ompt_wait_id_t;

typedef enum {
    ompt_task_complete      = 1,
    ompt_task_yield         = 2,
    ompt_task_cancel        = 3,
    ompt_task_detach        = 4,
    ompt_task_early_fulfill = 5,
    ompt_task_late_fulfill  = 6,
    ompt_task_switch        = 7,
} ompt_task_status_t;

typedef enum {
    ompt_set_error      = 0,
    ompt_set_never      = 1,
    ompt_set_impossible = 2,
    ompt_set_sometimes  = 3,
    ompt_set_sometimes_paired = 4,
    ompt_set_always     = 5,
} ompt_set_result_t;

typedef void (*ompt_interface_fn_t)(void);
typedef ompt_interface_fn_t (*ompt_function_lookup_t)(const char *);
typedef ompt_set_result_t (*ompt_set_callback_t)(ompt_callbacks_t, ompt_interface_fn_t);

typedef void (*ompt_finalize_t)(ompt_data_t *tool_data);
typedef int  (*ompt_initialize_t)(ompt_function_lookup_t lookup,
                                   int initial_device_num,
                                   ompt_data_t *tool_data);

typedef struct {
    ompt_initialize_t initialize;
    ompt_finalize_t   finalize;
    ompt_data_t       tool_data;
} ompt_start_tool_result_t;

/* Frame / codeptr types we don't use deeply */
typedef struct { void *exit_frame; void *enter_frame; } ompt_frame_t;

typedef struct {
    ompt_data_t variable;
    int         dependence_type;   /* ompt_dependence_type_t: 1 in, 2 out, 3 inout, ... */
} ompt_dependence_t;

/* ── Helpers ─────────────────────────────────────────────────────────── */
static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

#include "../common/hp_transport.h"
#include "../common/hp_ids.h"
#include "../common/callstack.h"
#include "../common/codeptr_resolve.h"

static pid_t gettid_compat(void) { return hp_tx_tid(); }

static void emit_span(const char *cat, pid_t tid,
                      uint64_t start_ns, uint64_t dur_ns,
                      const char *name, const char *extra) {
    if (!hp_tx_enabled()) return;
    if (extra && *extra)
        hp_tx_emitf("span:%s:%d:%d:%llu:%llu:%s:%s\n", cat, (int)hp_tx_pid(), (int)tid,
                    (unsigned long long)start_ns, (unsigned long long)dur_ns, name, extra);
    else
        hp_tx_emitf("span:%s:%d:%d:%llu:%llu:%s\n", cat, (int)hp_tx_pid(), (int)tid,
                    (unsigned long long)start_ns, (unsigned long long)dur_ns, name);
    emit_callstack(start_ns);
}

static void emit_instant(const char *cat, uint64_t ts, const char *name, const char *extra) {
    if (!hp_tx_enabled()) return;
    hp_tx_emitf("inst:%s:%d:%d:%llu:%s:%s\n", cat, (int)hp_tx_pid(), (int)gettid_compat(),
                (unsigned long long)ts, name, extra ? extra : "");
}

/* Call-site tags via the shared resolver (codeptr_resolve.h): sym=,symfile=
 * or lib=,offset=. symfile= names the ELF dladdr found the symbol in, which
 * matters when the profiled command is a launcher (srun, mpirun) wrapping
 * the real binary. Tags that do not fit become codeptr=truncated. */
#define EXTRA_SZ 2560

/* ── Per-thread nesting stacks ───────────────────────────────────────── */
#define MAX_DEPTH 32
static __thread uint64_t     tls_parallel_start[MAX_DEPTH];
static __thread ompt_id_t    tls_parallel_id[MAX_DEPTH];
static __thread const void  *tls_parallel_codeptr[MAX_DEPTH];
static __thread int          tls_parallel_depth = 0;
static __thread uint64_t     tls_work_start[MAX_DEPTH];
static __thread ompt_work_t  tls_work_type[MAX_DEPTH];
static __thread const void  *tls_work_codeptr[MAX_DEPTH];
static __thread int          tls_work_depth      = 0;
static __thread uint64_t     tls_sync_start[MAX_DEPTH];
static __thread const void  *tls_sync_codeptr[MAX_DEPTH];
static __thread uint64_t     tls_sync_par[MAX_DEPTH];   /* region id, captured at begin */
static __thread uint64_t     tls_sync_task[MAX_DEPTH];  /* encountering task id */
static __thread int          tls_sync_depth      = 0;
static __thread uint64_t     tls_target_start[MAX_DEPTH];
static __thread int          tls_target_depth    = 0;
static __thread uint64_t     tls_itask_start[MAX_DEPTH];
static __thread uint64_t     tls_itask_par[MAX_DEPTH];
static __thread uint64_t     tls_itask_id[MAX_DEPTH];
static __thread unsigned int tls_itask_index[MAX_DEPTH];
static __thread int          tls_itask_depth     = 0;

/* Open tasks keyed by task id, not a stack: untied tasks and task yields can
 * resume tasks out of nesting order. */
#define MAX_TASK_DEPTH 32
typedef struct { uint64_t id; uint64_t start_ns; int in_use; } TaskSlot;
static __thread TaskSlot     tls_tasks[MAX_TASK_DEPTH];
static uint64_t              g_task_id_seq     = 1;
/* Implicit tasks get ids too (task= on a taskwait names the task whose
 * children it waits for, parent= on a task_create the task that created
 * it), in their own range: HP_TASK_IMPLICIT set. cb_task_schedule never
 * opens an omp_task fragment for them -- their time is omp_implicit_task. */
#define HP_TASK_IMPLICIT (1ULL << 63)
static uint64_t              g_itask_id_seq    = 1;
static int is_explicit_task(uint64_t id) { return id != 0 && !(id & HP_TASK_IMPLICIT); }
static uint64_t              g_parallel_id_seq = 1;

/* ── Callbacks ───────────────────────────────────────────────────────── */

static void cb_thread_begin(ompt_thread_t type, ompt_data_t *thread_data) {
    (void)type; (void)thread_data;
}

static void cb_thread_end(ompt_data_t *thread_data) {
    (void)thread_data;
}

static void cb_parallel_begin(
    ompt_data_t *encountering_task_data,
    const ompt_frame_t *encountering_task_frame,
    ompt_data_t *parallel_data,
    unsigned int requested_parallelism,
    int flags, const void *codeptr_ra)
{
    (void)encountering_task_data; (void)encountering_task_frame; (void)flags;
    if (tls_parallel_depth < MAX_DEPTH) {
        /* Assign a tool-unique ID to this parallel region.  The OMPT spec allows
         * the tool to write parallel_data->value; we store it so cb_work and
         * cb_sync_region callbacks on worker threads can read it back. */
        /* pid in the high bits: region ids stay unique across the ranks
         * of an MPI+OpenMP trace (span ids are joined by value). */
        uint64_t par_unique = hp_span_id(HP_ID_OMPT, (uint32_t)getpid(),
                                         (uint32_t)(__sync_fetch_and_add(&g_parallel_id_seq, 1) & 0xffffffffull));
        if (parallel_data) parallel_data->value = par_unique;
        tls_parallel_start[tls_parallel_depth]   = now_ns();
        tls_parallel_id[tls_parallel_depth]      = par_unique;
        tls_parallel_codeptr[tls_parallel_depth] = codeptr_ra;
    }
    /* Counts every begin, even past MAX_DEPTH, so begin/end stay balanced
     * and end can recognize an overflowed level (see cb_parallel_end). */
    tls_parallel_depth++;
    (void)requested_parallelism;
}

static void cb_parallel_end(
    ompt_data_t *parallel_data,
    ompt_data_t *encountering_task_data,
    int flags, const void *codeptr_ra)
{
    (void)encountering_task_data; (void)flags; (void)codeptr_ra;
    if (tls_parallel_depth > 0) {
        tls_parallel_depth--;
        /* An overflowed level (depth >= MAX_DEPTH) was never recorded: emit
         * nothing rather than pop another, still-open region's slot (which
         * would also desync every later end on this thread). */
        if (tls_parallel_depth < MAX_DEPTH) {
            uint64_t t0         = tls_parallel_start[tls_parallel_depth];
            ompt_id_t pid       = tls_parallel_id[tls_parallel_depth];
            const void *cptr    = tls_parallel_codeptr[tls_parallel_depth];
            char extra[EXTRA_SZ];
            snprintf(extra, sizeof(extra), "type=parallel,id=%llu,sid=%llu",
                     (unsigned long long)pid, (unsigned long long)pid);
            hprofiler_append_codeptr_tag(extra, sizeof(extra), cptr);
            emit_span("openmp", gettid_compat(), t0, now_ns() - t0,
                      "parallel_region", extra);
        }
    }
    (void)parallel_data;
}

/* Per-thread implicit task: the only OMPT event showing a worker thread's
 * share of a region (otherwise workers show only barrier spans). The initial
 * task (the whole program) is skipped. parallel_data may be NULL at
 * scope_end per the spec, so the region id is captured at begin. */
static void cb_implicit_task(
    ompt_scope_endpoint_t endpoint, ompt_data_t *parallel_data,
    ompt_data_t *task_data, unsigned int actual_parallelism,
    unsigned int index, int flags)
{
    (void)actual_parallelism;
    if (flags & 0x1 /* ompt_task_initial */) return;
    if (endpoint == ompt_scope_begin) {
        uint64_t itask = HP_TASK_IMPLICIT | (uint64_t)__sync_fetch_and_add(&g_itask_id_seq, 1);
        if (task_data) task_data->value = itask;
        if (tls_itask_depth < MAX_DEPTH) {
            tls_itask_start[tls_itask_depth] = now_ns();
            tls_itask_par[tls_itask_depth]   = (parallel_data ? parallel_data->value : 0);
            tls_itask_index[tls_itask_depth] = index;
            tls_itask_id[tls_itask_depth]    = itask;
        }
        tls_itask_depth++;  /* unconditional -- see cb_parallel_begin */
    } else if (endpoint == ompt_scope_end && tls_itask_depth > 0) {
        tls_itask_depth--;
        if (tls_itask_depth >= MAX_DEPTH) return;
        uint64_t t0 = tls_itask_start[tls_itask_depth];
        char extra[128];
        if (tls_itask_par[tls_itask_depth])
            snprintf(extra, sizeof(extra), "type=implicit_task,index=%u,psid=%llu,itask=%llu",
                     tls_itask_index[tls_itask_depth],
                     (unsigned long long)tls_itask_par[tls_itask_depth],
                     (unsigned long long)tls_itask_id[tls_itask_depth]);
        else
            snprintf(extra, sizeof(extra), "type=implicit_task,index=%u,itask=%llu",
                     tls_itask_index[tls_itask_depth],
                     (unsigned long long)tls_itask_id[tls_itask_depth]);
        emit_span("openmp", gettid_compat(), t0, now_ns() - t0,
                  "omp_implicit_task", extra);
    }
}

static void cb_work(
    ompt_work_t wstype, ompt_scope_endpoint_t endpoint,
    ompt_data_t *parallel_data, ompt_data_t *task_data,
    uint64_t count, const void *codeptr_ra)
{
    (void)task_data;
    uint64_t par_id = (parallel_data && parallel_data->value) ? parallel_data->value : 0;
    static const char *wnames[] = {
        "", "omp_loop", "omp_sections", "omp_single_exec", "omp_single_other",
        "omp_workshare", "omp_distribute", "omp_taskloop", "omp_scope",
        "", "omp_loop_static", "omp_loop_dynamic", "omp_loop_guided", "omp_loop_other"
    };
    const char *wname = (wstype < 14) ? wnames[wstype] : "omp_work";
    if (*wname == '\0') wname = "omp_work";

    if (endpoint == ompt_scope_begin) {
        if (tls_work_depth < MAX_DEPTH) {
            tls_work_start[tls_work_depth]   = now_ns();
            tls_work_type[tls_work_depth]    = wstype;
            tls_work_codeptr[tls_work_depth] = codeptr_ra;
        }
        /* Unconditional, as in cb_parallel_begin. */
        tls_work_depth++;
    } else if (tls_work_depth > 0) {
        tls_work_depth--;
        if (tls_work_depth >= MAX_DEPTH) return;  /* overflowed level -- never captured, omit */
        char extra[EXTRA_SZ];
        if (par_id)
            snprintf(extra, sizeof(extra), "type=work,count=%llu,psid=%llu",
                     (unsigned long long)count, (unsigned long long)par_id);
        else
            snprintf(extra, sizeof(extra), "type=work,count=%llu",
                     (unsigned long long)count);
        hprofiler_append_codeptr_tag(extra, sizeof(extra), tls_work_codeptr[tls_work_depth]);
        emit_span("openmp", gettid_compat(),
                  tls_work_start[tls_work_depth], now_ns() - tls_work_start[tls_work_depth],
                  wname, extra);
    }
}

static void cb_sync_region(
    ompt_sync_region_t kind, ompt_scope_endpoint_t endpoint,
    ompt_data_t *parallel_data, ompt_data_t *task_data,
    const void *codeptr_ra)
{
    uint64_t par_id = (parallel_data && parallel_data->value) ? parallel_data->value : 0;
    /* indexed by ompt_sync_region_t (OpenMP 5.0: 1-7, 5.1: 8-10). Hooks
     * before this table was corrected named kinds 5-10 one slot late (the
     * implicit barrier of a worksharing loop came out as omp_reduction);
     * sk= records the kind itself. */
    static const char *snames[] = {
        "", "omp_barrier", "omp_barrier_implicit", "omp_barrier_explicit",
        "omp_barrier_impl", "omp_taskwait", "omp_taskgroup", "omp_reduction",
        "omp_barrier_workshare", "omp_barrier_parallel", "omp_barrier_teams"
    };
    const char *sname = (kind >= 0 && kind <= 10) ? snames[kind] : "omp_sync";
    if (*sname == '\0') sname = "omp_sync";

    if (endpoint == ompt_scope_begin) {
        if (tls_sync_depth < MAX_DEPTH) {
            tls_sync_start[tls_sync_depth]   = now_ns();
            tls_sync_codeptr[tls_sync_depth] = codeptr_ra;
            /* The region id at BEGIN: at the end of a parallel region's
             * implicit barrier parallel_data may be NULL (OMPT 5.x), and the
             * barrier's team was then lost (grouped by time overlap). Fall
             * back to the enclosing implicit task's region. */
            if (!par_id && tls_itask_depth > 0 && tls_itask_depth <= MAX_DEPTH)
                par_id = tls_itask_par[tls_itask_depth - 1];
            tls_sync_par[tls_sync_depth]  = par_id;
            tls_sync_task[tls_sync_depth] = task_data ? task_data->value : 0;
        }
        tls_sync_depth++;  /* unconditional -- see cb_parallel_begin's comment */
    } else if (tls_sync_depth > 0) {
        tls_sync_depth--;
        if (tls_sync_depth >= MAX_DEPTH) return;  /* overflowed level -- never captured, omit */
        uint64_t t0      = tls_sync_start[tls_sync_depth];
        const void *cptr = tls_sync_codeptr[tls_sync_depth];
        if (!par_id) par_id = tls_sync_par[tls_sync_depth];
        uint64_t task    = tls_sync_task[tls_sync_depth];
        char extra[EXTRA_SZ];
        int n;
        if (par_id)
            n = snprintf(extra, sizeof(extra), "type=sync,sk=%d,psid=%llu", (int)kind,
                         (unsigned long long)par_id);
        else
            n = snprintf(extra, sizeof(extra), "type=sync,sk=%d", (int)kind);
        /* taskwait / taskgroup: the task whose children (descendants) it
         * waits for. */
        if (task && (kind == ompt_sync_region_taskwait || kind == ompt_sync_region_taskgroup)
            && n > 0 && (size_t)n < sizeof(extra))
            snprintf(extra + n, sizeof(extra) - (size_t)n, ",task=%llu", (unsigned long long)task);
        hprofiler_append_codeptr_tag(extra, sizeof(extra), cptr);
        emit_span("sync", gettid_compat(), t0, now_ns() - t0,
                  sname, extra);
    }
}

static void cb_task_create(
    ompt_data_t *encountering_task_data,
    const ompt_frame_t *encountering_task_frame,
    ompt_data_t *new_task_data,
    int flags, int has_dependences,
    const void *codeptr_ra)
{
    (void)encountering_task_frame;
    uint64_t task_id = (uint64_t)__sync_fetch_and_add(&g_task_id_seq, 1);
    if (new_task_data) new_task_data->value = task_id;
    uint64_t parent = encountering_task_data ? encountering_task_data->value : 0;
    uint64_t region = (tls_itask_depth > 0 && tls_itask_depth <= MAX_DEPTH)
                      ? tls_itask_par[tls_itask_depth - 1] : 0;
    char extra[EXTRA_SZ];
    snprintf(extra, sizeof(extra), "type=task_create,id=%llu,parent=%llu,region=%llu%s%s%s%s",
             (unsigned long long)task_id, (unsigned long long)parent, (unsigned long long)region,
             (flags & 0x8) ? ",target=1" : "",                 /* ompt_task_target */
             (flags & 0x08000000) ? ",undeferred=1" : "",      /* ompt_task_undeferred */
             (flags & 0x10000000) ? ",untied=1" : "",          /* ompt_task_untied */
             has_dependences ? ",deps=1" : "");
    hprofiler_append_codeptr_tag(extra, sizeof(extra), codeptr_ra);
    uint64_t now = now_ns();
    emit_span("openmp", gettid_compat(), now, 0, "omp_task_create", extra);
}

/* Explicit tasks only: their ids come from cb_task_create (from 1); an
 * implicit or initial task's data stays 0, and its time is the
 * omp_implicit_task span -- a fragment of it, cut where an explicit task
 * starts on its thread, is not a task execution.
 *
 * One omp_task span per execution fragment of a task on its thread: the
 * fragment a task is suspended at (taskyield, a task switch at a taskwait
 * or a task scheduling point -- it resumes later, possibly on another
 * thread if untied) carries fragment=1; the last one carries the status
 * that ended it (1 = complete). So the omp_task spans without fragment=1
 * are the tasks executed, one each, and a task's fragments (same id=) add
 * up to its execution time, the time spent in the tasks it waited for not
 * included. */
static void cb_task_schedule(
    ompt_data_t *prior_task_data,
    int prior_task_status,
    ompt_data_t *next_task_data)
{
    uint64_t now = now_ns();
    /* Complete / suspend prior task -- find it by ID among open slots,
     * not by assuming it's the most-recently-pushed one. */
    if (prior_task_data && is_explicit_task(prior_task_data->value)) {
        for (int i = 0; i < MAX_TASK_DEPTH; i++) {
            if (tls_tasks[i].in_use && tls_tasks[i].id == prior_task_data->value) {
                char extra[80];
                int suspended = prior_task_status == ompt_task_yield ||
                                prior_task_status == ompt_task_switch;
                snprintf(extra, sizeof(extra), "type=task,id=%llu,status=%d%s",
                         (unsigned long long)prior_task_data->value, prior_task_status,
                         suspended ? ",fragment=1" : "");
                emit_span("openmp", gettid_compat(),
                          tls_tasks[i].start_ns, now - tls_tasks[i].start_ns,
                          "omp_task", extra);
                tls_tasks[i].in_use = 0;
                break;
            }
        }
    }
    /* Claim any free slot (tasks are found by id). With all MAX_TASK_DEPTH
     * slots in use the task is not tracked. */
    if (next_task_data && is_explicit_task(next_task_data->value)) {
        for (int i = 0; i < MAX_TASK_DEPTH; i++) {
            if (!tls_tasks[i].in_use) {
                tls_tasks[i].id = next_task_data->value;
                tls_tasks[i].start_ns = now;
                tls_tasks[i].in_use = 1;
                break;
            }
        }
    }
}

/* ── Mutual exclusion: critical, locks, ordered (mutex callbacks) ──────────
 * mutex_acquire -> mutex_acquired -> mutex_released, per OMPT mutex kind:
 *   critical          omp_critical_wait (sync)  + omp_critical_hold (openmp)
 *   lock / nest_lock  omp_lock_wait     (sync)  + omp_lock_hold     (openmp)
 *   test_lock (+nest) omp_lock_hold only when the test succeeded (test=1)
 *   ordered           omp_ordered_wait  (sync)  + omp_ordered       (openmp)
 * (the names of the GNU libgomp hook's spans for the same constructs).
 * A nest lock re-acquired by its owner reports ompt_callback_nest_lock
 * instead (instant omp_nest_lock, endpoint=). Atomics are not recorded: one
 * callback per atomic update would dominate the cost of what is measured.
 * wait_id= identifies the mutex. */
#define MAX_HELD 16
typedef struct { ompt_wait_id_t id; int kind; uint64_t t0; const void *cptr; } HeldMutex;
static __thread HeldMutex      tls_held[MAX_HELD];
static __thread int            tls_nheld;
static __thread uint64_t       tls_acq_t0;
static __thread ompt_wait_id_t tls_acq_id;
static __thread int            tls_acq_kind;
static __thread const void    *tls_acq_cptr;

static const char *mutex_base(int kind) {
    switch (kind) {
    case ompt_mutex_critical: return "critical";
    case ompt_mutex_ordered:  return "ordered";
    case ompt_mutex_lock: case ompt_mutex_test_lock:
    case ompt_mutex_nest_lock: case ompt_mutex_test_nest_lock: return "lock";
    default: return NULL;
    }
}

static int mutex_is_test(int kind) {
    return kind == ompt_mutex_test_lock || kind == ompt_mutex_test_nest_lock;
}

static int mutex_is_nest(int kind) {
    return kind == ompt_mutex_nest_lock || kind == ompt_mutex_test_nest_lock;
}

static void cb_mutex_acquire(ompt_mutex_t kind, unsigned int hint, unsigned int impl,
                             ompt_wait_id_t wait_id, const void *codeptr_ra) {
    (void)hint; (void)impl;
    if (!mutex_base(kind)) return;
    tls_acq_t0 = now_ns();
    tls_acq_id = wait_id;
    tls_acq_kind = kind;
    tls_acq_cptr = codeptr_ra;
}

static void cb_mutex_acquired(ompt_mutex_t kind, ompt_wait_id_t wait_id, const void *codeptr_ra) {
    const char *base = mutex_base(kind);
    if (!base) return;
    uint64_t t = now_ns();
    char extra[EXTRA_SZ];
    if (!mutex_is_test(kind) && tls_acq_t0 && tls_acq_id == wait_id && tls_acq_kind == (int)kind) {
        /* the wait: acquire -> acquired */
        char name[32];
        snprintf(name, sizeof(name), "omp_%s_wait", base);
        snprintf(extra, sizeof(extra), "type=sync,mutex=%s,wait_id=%llu%s", base,
                 (unsigned long long)wait_id, mutex_is_nest(kind) ? ",nest=1" : "");
        hprofiler_append_codeptr_tag(extra, sizeof(extra), tls_acq_cptr ? tls_acq_cptr : codeptr_ra);
        emit_span("sync", gettid_compat(), tls_acq_t0, t - tls_acq_t0, name, extra);
    }
    tls_acq_t0 = 0;
    if (tls_nheld < MAX_HELD)
        tls_held[tls_nheld++] = (HeldMutex){wait_id, (int)kind, t, codeptr_ra};
}

static void cb_mutex_released(ompt_mutex_t kind, ompt_wait_id_t wait_id, const void *codeptr_ra) {
    const char *base = mutex_base(kind);
    if (!base) return;
    for (int i = tls_nheld - 1; i >= 0; i--) {
        const char *hb = mutex_base(tls_held[i].kind);
        if (tls_held[i].id != wait_id || !hb || strcmp(hb, base) != 0) continue;
        HeldMutex h = tls_held[i];
        tls_held[i] = tls_held[--tls_nheld];
        uint64_t t = now_ns();
        char extra[EXTRA_SZ], name[32];
        snprintf(name, sizeof(name), kind == ompt_mutex_ordered ? "omp_%s" : "omp_%s_hold", base);
        snprintf(extra, sizeof(extra), "type=%s,mutex=%s,wait_id=%llu%s%s",
                 kind == ompt_mutex_ordered ? "ordered" : "hold", base, (unsigned long long)wait_id,
                 mutex_is_nest(h.kind) ? ",nest=1" : "", mutex_is_test(h.kind) ? ",test=1" : "");
        hprofiler_append_codeptr_tag(extra, sizeof(extra), h.cptr ? h.cptr : codeptr_ra);
        emit_span("openmp", gettid_compat(), h.t0, t - h.t0, name, extra);
        return;
    }
}

static void cb_nest_lock(ompt_scope_endpoint_t endpoint, ompt_wait_id_t wait_id, const void *codeptr_ra) {
    char extra[EXTRA_SZ];
    snprintf(extra, sizeof(extra), "type=nest_lock,endpoint=%s,wait_id=%llu",
             endpoint == ompt_scope_begin ? "begin" : "end", (unsigned long long)wait_id);
    hprofiler_append_codeptr_tag(extra, sizeof(extra), codeptr_ra);
    emit_instant("openmp", now_ns(), "omp_nest_lock", extra);
}

static void cb_lock_init(ompt_mutex_t kind, unsigned int hint, unsigned int impl,
                         ompt_wait_id_t wait_id, const void *codeptr_ra) {
    (void)hint; (void)impl;
    char extra[EXTRA_SZ];
    snprintf(extra, sizeof(extra), "type=lock_init,wait_id=%llu%s", (unsigned long long)wait_id,
             mutex_is_nest(kind) ? ",nest=1" : "");
    hprofiler_append_codeptr_tag(extra, sizeof(extra), codeptr_ra);
    emit_instant("openmp", now_ns(), "omp_lock_init", extra);
}

static void cb_lock_destroy(ompt_mutex_t kind, ompt_wait_id_t wait_id, const void *codeptr_ra) {
    char extra[EXTRA_SZ];
    snprintf(extra, sizeof(extra), "type=lock_destroy,wait_id=%llu%s", (unsigned long long)wait_id,
             mutex_is_nest(kind) ? ",nest=1" : "");
    hprofiler_append_codeptr_tag(extra, sizeof(extra), codeptr_ra);
    emit_instant("openmp", now_ns(), "omp_lock_destroy", extra);
}

/* ── Task dependences ─────────────────────────────────────────────────────
 * ompt_callback_dependences: a task created with depend clauses (instant
 * omp_task_dependences: task=, ndeps=, in=/out=/inout=/other= counts);
 * ompt_callback_task_dependence: the runtime made sink wait for src
 * (instant omp_task_dependence: src=, sink= -- task ids as in
 * omp_task_create's id=). */
static void cb_dependences(ompt_data_t *task_data, const ompt_dependence_t *deps, int ndeps) {
    int in = 0, out = 0, inout = 0, other = 0;
    for (int i = 0; deps && i < ndeps; i++) {
        switch (deps[i].dependence_type) {
        case 1: in++; break;
        case 2: out++; break;
        case 3: inout++; break;
        default: other++;
        }
    }
    char extra[160];
    snprintf(extra, sizeof(extra), "type=task_dependences,task=%llu,ndeps=%d,in=%d,out=%d,inout=%d,other=%d",
             (unsigned long long)(task_data ? task_data->value : 0), ndeps, in, out, inout, other);
    emit_instant("openmp", now_ns(), "omp_task_dependences", extra);
}

static void cb_task_dependence(ompt_data_t *src_task_data, ompt_data_t *sink_task_data) {
    char extra[96];
    snprintf(extra, sizeof(extra), "type=task_dependence,src=%llu,sink=%llu",
             (unsigned long long)(src_task_data ? src_task_data->value : 0),
             (unsigned long long)(sink_task_data ? sink_task_data->value : 0));
    emit_instant("openmp", now_ns(), "omp_task_dependence", extra);
}

static void cb_target(
    ompt_target_t kind, ompt_scope_endpoint_t endpoint,
    int device_num, ompt_data_t *task_data,
    ompt_id_t target_id, const void *codeptr_ra)
{
    (void)task_data; (void)target_id; (void)codeptr_ra;
    /* indexed by ompt_target_t; a nowait form (5.1) is its blocking kind
     * + 8. Hooks before this table was corrected named target regions
     * omp_target_enter_data and enter-data regions omp_target_exit_data. */
    static const char *tnames[] = {
        "", "omp_target", "omp_target_enter_data", "omp_target_exit_data",
        "omp_target_update"
    };
    int nowait = kind >= ompt_target_nowait && kind <= ompt_target_update_nowait;
    int base = nowait ? (int)kind - 8 : (int)kind;
    const char *tname = (base >= 1 && base <= 4) ? tnames[base] : "omp_target";

    if (endpoint == ompt_scope_begin) {
        if (tls_target_depth < MAX_DEPTH)
            tls_target_start[tls_target_depth] = now_ns();
        tls_target_depth++;  /* unconditional -- see cb_parallel_begin's comment */
    } else if (tls_target_depth > 0) {
        tls_target_depth--;
        if (tls_target_depth >= MAX_DEPTH) return;  /* overflowed level -- never captured, omit */
        char extra[64];
        snprintf(extra, sizeof(extra), "type=offload,device=%d%s", device_num,
                 nowait ? ",nowait=1" : "");
        emit_span("openmp", gettid_compat(),
                  tls_target_start[tls_target_depth],
                  now_ns() - tls_target_start[tls_target_depth],
                  tname, extra);
    }
}

/* ── ACPP SSCP .jit.so interception ─────────────────────────────────── */
/* Intercept dlopen for ACPP SSCP .jit.so kernel libraries.
 * Only active when libhprofiler_ompt.so is also LD_PRELOAD-ed (the OpenMP
 * backend sets this automatically).  OMP_TOOL_LIBRARIES alone loads the
 * library as a dlopen plugin, which does NOT put our dlopen() in the global
 * interposition chain — LD_PRELOAD is needed for that.
 */
static int _jit_counter = 0;

void *dlopen(const char *filename, int flags) {
    typedef void *(*real_dlopen_t)(const char *, int);
    static real_dlopen_t real = NULL;
    if (!real) real = (real_dlopen_t)dlsym(RTLD_NEXT, "dlopen");
    void *h = real ? real(filename, flags) : NULL;
    if (!h || !filename || !strstr(filename, ".jit.so"))
        return h;
    int idx = __sync_fetch_and_add(&_jit_counter, 1);
    char saved[512];
    snprintf(saved, sizeof(saved), "/tmp/hprofiler_jit_%d_%d.so",
             (int)getpid(), idx);
    FILE *src = fopen(filename, "rb");
    if (src) {
        int ok = 0;
        FILE *dst = fopen(saved, "wb");
        if (dst) {
            char buf[65536];
            size_t n;
            ok = 1;
            while ((n = fread(buf, 1, sizeof(buf), src)) > 0) {
                if (fwrite(buf, 1, n, dst) != n) { ok = 0; break; }
            }
            fclose(dst);
            if (!ok) remove(saved);
        }
        fclose(src);
        if (ok) {
            const char *base = strrchr(filename, '/');
            char extra[600];
            snprintf(extra, sizeof(extra), "type=jit_load,stage=load,path=%s", saved);
            emit_span("jit", gettid_compat(), now_ns(), 0,
                      base ? base + 1 : filename, extra);
        }
    }
    return h;
}

/* ── OMPT initialize / finalize ──────────────────────────────────────── */

static int tool_initialize(ompt_function_lookup_t lookup,
                            int initial_device_num,
                            ompt_data_t *tool_data)
{
    (void)initial_device_num; (void)tool_data;
    hp_tx_init("ompt");
    cs_init();

    ompt_set_callback_t set_callback =
        (ompt_set_callback_t)lookup("ompt_set_callback");
    if (!set_callback) return 0;

    /* ompt_set_error / ompt_set_never mean the callback will never fire:
     * report those, so a category's zero events are explained. */
    char unsupported[256] = "";
#define REG(event, cb) do { \
        ompt_set_result_t _r = set_callback(event, (ompt_interface_fn_t)(cb)); \
        if (_r == ompt_set_error || _r == ompt_set_never) { \
            size_t _len = strlen(unsupported); \
            if (_len + strlen(#event) + 3 < sizeof(unsupported)) \
                snprintf(unsupported + _len, sizeof(unsupported) - _len, \
                         "%s%s", _len ? "," : "", #event); \
        } \
    } while (0)
    REG(ompt_callback_thread_begin,   cb_thread_begin);
    REG(ompt_callback_thread_end,     cb_thread_end);
    REG(ompt_callback_parallel_begin, cb_parallel_begin);
    REG(ompt_callback_parallel_end,   cb_parallel_end);
    REG(ompt_callback_implicit_task,  cb_implicit_task);
    REG(ompt_callback_task_create,    cb_task_create);
    REG(ompt_callback_task_schedule,  cb_task_schedule);
    REG(ompt_callback_work,           cb_work);
    REG(ompt_callback_sync_region,    cb_sync_region);
    REG(ompt_callback_target,         cb_target);
    REG(ompt_callback_mutex_acquire,  cb_mutex_acquire);
    REG(ompt_callback_mutex_acquired, cb_mutex_acquired);
    REG(ompt_callback_mutex_released, cb_mutex_released);
    REG(ompt_callback_nest_lock,      cb_nest_lock);
    REG(ompt_callback_lock_init,      cb_lock_init);
    REG(ompt_callback_lock_destroy,   cb_lock_destroy);
    REG(ompt_callback_dependences,    cb_dependences);
    REG(ompt_callback_task_dependence, cb_task_dependence);
#undef REG

    if (unsupported[0]) {
        fprintf(stderr,
                "[hprofiler][ompt] this OpenMP runtime does not support: %s "
                "-- those event categories will report zero events, not an error\n",
                unsupported);
    }

    return 1;
}

/* The runtime calls this at exit: drain every thread's buffered events
 * and send the closing status. Events after it go out synchronously; the
 * destructor covers a runtime that never calls finalize. */
static void tool_finalize(ompt_data_t *tool_data) {
    (void)tool_data;
    hp_tx_shutdown(1);
}

__attribute__((destructor)) static void ompt_tool_fini(void) { hp_tx_shutdown(1); }

/* ── Entry point ─────────────────────────────────────────────────────── */

ompt_start_tool_result_t *ompt_start_tool(
    unsigned int omp_version, const char *runtime_version)
{
    (void)omp_version; (void)runtime_version;
    static ompt_start_tool_result_t result = {
        .initialize = tool_initialize,
        .finalize   = tool_finalize,
        .tool_data  = {.value = 0},
    };
    return &result;
}
