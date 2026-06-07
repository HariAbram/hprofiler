/*
 * OMPT Tool — correct OpenMP 5.0 Tools Interface implementation.
 *
 * Loaded via OMP_TOOL_LIBRARIES. Uses the proper ompt_start_tool_result_t
 * ABI with initialize/finalize function pointers.
 *
 * Callbacks registered:
 *   thread_begin/end         → thread lifecycle
 *   parallel_begin/end       → parallel region spans
 *   work (loop/sections)     → work-distribution spans
 *   sync_region (barriers)   → synchronization spans
 *   target begin/end         → GPU offload spans (OMP 5, ACPP)
 *
 * Wire format to HPROFILER_SOCKET:
 *   span:<cat>:<pid>:<tid>:<start_ns>:<dur_ns>:<name>[:<key=val,...>]\n
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

/* Resolve a codeptr_ra:
 *  1. Try dladdr() — works when the symbol is exported.
 *  2. Fall back to /proc/self/maps — finds the library and computes the
 *     static file offset even for non-exported / internal symbols.
 *
 * On success, writes into `out_sym` (symbol name) or `out_lib`+`out_off`
 * (library path + offset).  At most one of sym / lib+off is filled.
 * Returns 1 if anything was resolved, 0 otherwise.
 */
static int resolve_codeptr_full(const void *codeptr,
                                const char **out_sym,   /* dladdr name */
                                char *out_lib, size_t lib_sz,
                                uint64_t *out_off) {
    if (!codeptr) return 0;
    *out_sym = NULL;
    out_lib[0] = '\0';
    *out_off   = 0;

    /* 1. dladdr — fast path for exported symbols */
    Dl_info info;
    if (dladdr(codeptr, &info) && info.dli_sname && info.dli_sname[0]) {
        *out_sym = info.dli_sname;
        return 1;
    }

    /* 2. /proc/self/maps — works for internal symbols in any .so */
    FILE *f = fopen("/proc/self/maps", "r");
    if (!f) return 0;
    uintptr_t addr = (uintptr_t)codeptr;
    char line[512];
    while (fgets(line, sizeof(line), f)) {
        uintptr_t lo, hi, off;
        char perms[8], dev[16], path[256];
        int inode;
        path[0] = '\0';
        if (sscanf(line, "%lx-%lx %7s %lx %15s %d %255s",
                   &lo, &hi, perms, &off, dev, &inode, path) < 6)
            continue;
        if (addr < lo || addr >= hi) continue;
        if (path[0] == '\0' || path[0] == '[') break;  /* anon/stack */
        /* static offset = file_offset_of_segment + (addr - segment_vaddr) */
        uint64_t static_off = (uint64_t)off + (addr - lo);
        strncpy(out_lib, path, lib_sz - 1);
        out_lib[lib_sz - 1] = '\0';
        *out_off = static_off;
        fclose(f);
        return 1;
    }
    fclose(f);
    return 0;
}

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
    ompt_sync_region_taskwait                 = 6,
    ompt_sync_region_taskgroup                = 7,
    ompt_sync_region_reduction                = 8,
    ompt_sync_region_barrier_implicit_workshare = 9,
    ompt_sync_region_barrier_implicit_parallel  = 10,
    ompt_sync_region_barrier_teams              = 11,
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
    ompt_target_submit        = 5,
    ompt_target_enter_data    = 1,
    ompt_target_exit_data     = 2,
    ompt_target               = 3,
    ompt_target_update        = 4,
} ompt_target_t;

/* Callback event IDs (from omp-tools.h) */
typedef enum {
    ompt_callback_thread_begin   = 1,
    ompt_callback_thread_end     = 2,
    ompt_callback_parallel_begin = 3,
    ompt_callback_parallel_end   = 4,
    ompt_callback_task_create    = 5,
    ompt_callback_task_schedule  = 6,
    ompt_callback_target         = 8,
    ompt_callback_work           = 20,
    ompt_callback_sync_region    = 23,
} ompt_callbacks_t;

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

/* ── Globals ─────────────────────────────────────────────────────────── */
static int             g_sock        = -1;
static pthread_mutex_t g_sock_mutex  = PTHREAD_MUTEX_INITIALIZER;
static pid_t           g_pid         = 0;

/* ── Helpers ─────────────────────────────────────────────────────────── */
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
        g_sock = s;
        g_pid  = getpid();
    }
}

static void send_all(const char *buf, int n) {
    while (n > 0) {
        ssize_t r = send(g_sock, buf, (size_t)n, MSG_NOSIGNAL);
        if (r < 0) { close(g_sock); g_sock = -1; return; }
        buf += r; n -= (int)r;
    }
}

#include "../common/callstack.h"

static void emit_span(const char *cat, pid_t tid,
                      uint64_t start_ns, uint64_t dur_ns,
                      const char *name, const char *extra) {
    pthread_mutex_lock(&g_sock_mutex);
    ensure_connected();
    if (g_sock >= 0) {
        char buf[512]; int n;
        if (extra && *extra)
            n = snprintf(buf, sizeof(buf),
                "span:%s:%d:%d:%llu:%llu:%s:%s\n",
                cat, g_pid, tid,
                (unsigned long long)start_ns, (unsigned long long)dur_ns,
                name, extra);
        else
            n = snprintf(buf, sizeof(buf),
                "span:%s:%d:%d:%llu:%llu:%s\n",
                cat, g_pid, tid,
                (unsigned long long)start_ns, (unsigned long long)dur_ns, name);
        if (n > 0 && n < (int)sizeof(buf)) send_all(buf, n);
        emit_callstack(start_ns);
    }
    pthread_mutex_unlock(&g_sock_mutex);
}

/* ── Per-thread nesting stacks ───────────────────────────────────────── */
#define MAX_DEPTH 32
static __thread uint64_t     tls_parallel_start[MAX_DEPTH];
static __thread ompt_id_t    tls_parallel_id[MAX_DEPTH];
static __thread const void  *tls_parallel_codeptr[MAX_DEPTH];
static __thread int          tls_parallel_depth = 0;
static __thread uint64_t     tls_work_start     = 0;
static __thread ompt_work_t  tls_work_type       = 0;
static __thread const void  *tls_work_codeptr    = NULL;
static __thread uint64_t     tls_sync_start[MAX_DEPTH];
static __thread const void  *tls_sync_codeptr[MAX_DEPTH];
static __thread int          tls_sync_depth      = 0;
static __thread uint64_t     tls_target_start    = 0;

/* Task scheduling stacks */
#define MAX_TASK_DEPTH 32
static __thread uint64_t     tls_task_start[MAX_TASK_DEPTH];
static __thread uint64_t     tls_task_id[MAX_TASK_DEPTH];
static __thread int          tls_task_depth = 0;
static uint64_t              g_task_id_seq  = 1;

/* ── Callbacks ───────────────────────────────────────────────────────── */

static void cb_thread_begin(ompt_thread_t type, ompt_data_t *thread_data) {
    ensure_connected();
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
        tls_parallel_start[tls_parallel_depth]   = now_ns();
        tls_parallel_id[tls_parallel_depth]      = parallel_data ? parallel_data->value : 0;
        tls_parallel_codeptr[tls_parallel_depth] = codeptr_ra;
        tls_parallel_depth++;
    }
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
        uint64_t t0         = tls_parallel_start[tls_parallel_depth];
        ompt_id_t pid       = tls_parallel_id[tls_parallel_depth];
        const void *cptr    = tls_parallel_codeptr[tls_parallel_depth];
        const char *sym = NULL; char lib[256]; uint64_t off = 0;
        char extra[512];
        if (resolve_codeptr_full(cptr, &sym, lib, sizeof(lib), &off)) {
            if (sym)
                snprintf(extra, sizeof(extra), "type=parallel,id=%llu,sym=%s",
                         (unsigned long long)pid, sym);
            else
                snprintf(extra, sizeof(extra),
                         "type=parallel,id=%llu,lib=%s,offset=0x%llx",
                         (unsigned long long)pid, lib, (unsigned long long)off);
        } else {
            snprintf(extra, sizeof(extra), "type=parallel,id=%llu",
                     (unsigned long long)pid);
        }
        emit_span("openmp", gettid_compat(), t0, now_ns() - t0,
                  "parallel_region", extra);
    }
    (void)parallel_data;
}

static void cb_work(
    ompt_work_t wstype, ompt_scope_endpoint_t endpoint,
    ompt_data_t *parallel_data, ompt_data_t *task_data,
    uint64_t count, const void *codeptr_ra)
{
    (void)parallel_data; (void)task_data;
    static const char *wnames[] = {
        "", "omp_loop", "omp_sections", "omp_single_exec", "omp_single_other",
        "omp_workshare", "omp_distribute", "omp_taskloop", "omp_scope",
        "", "omp_loop_static", "omp_loop_dynamic", "omp_loop_guided", "omp_loop_other"
    };
    const char *wname = (wstype < 14) ? wnames[wstype] : "omp_work";
    if (*wname == '\0') wname = "omp_work";

    if (endpoint == ompt_scope_begin) {
        tls_work_start   = now_ns();
        tls_work_type    = wstype;
        tls_work_codeptr = codeptr_ra;
    } else if (tls_work_start) {
        const char *sym = NULL; char lib[256]; uint64_t off = 0;
        char extra[512];
        if (resolve_codeptr_full(tls_work_codeptr, &sym, lib, sizeof(lib), &off)) {
            if (sym)
                snprintf(extra, sizeof(extra), "type=work,count=%llu,sym=%s",
                         (unsigned long long)count, sym);
            else
                snprintf(extra, sizeof(extra),
                         "type=work,count=%llu,lib=%s,offset=0x%llx",
                         (unsigned long long)count, lib, (unsigned long long)off);
        } else {
            snprintf(extra, sizeof(extra), "type=work,count=%llu",
                     (unsigned long long)count);
        }
        emit_span("openmp", gettid_compat(),
                  tls_work_start, now_ns() - tls_work_start,
                  wname, extra);
        tls_work_start = 0;
    }
}

static void cb_sync_region(
    ompt_sync_region_t kind, ompt_scope_endpoint_t endpoint,
    ompt_data_t *parallel_data, ompt_data_t *task_data,
    const void *codeptr_ra)
{
    (void)parallel_data; (void)task_data;
    static const char *snames[] = {
        "", "omp_barrier", "omp_barrier_implicit", "omp_barrier_explicit",
        "omp_barrier_impl", "", "omp_taskwait", "omp_taskgroup",
        "omp_reduction", "omp_barrier_workshare", "omp_barrier_parallel",
        "omp_barrier_teams"
    };
    const char *sname = (kind <= 11) ? snames[kind] : "omp_sync";
    if (*sname == '\0') sname = "omp_sync";

    if (endpoint == ompt_scope_begin) {
        if (tls_sync_depth < MAX_DEPTH) {
            tls_sync_start[tls_sync_depth]   = now_ns();
            tls_sync_codeptr[tls_sync_depth] = codeptr_ra;
            tls_sync_depth++;
        }
    } else if (tls_sync_depth > 0) {
        tls_sync_depth--;
        uint64_t t0      = tls_sync_start[tls_sync_depth];
        const void *cptr = tls_sync_codeptr[tls_sync_depth];
        const char *sym = NULL; char lib[256]; uint64_t off = 0;
        char extra[512];
        if (resolve_codeptr_full(cptr, &sym, lib, sizeof(lib), &off)) {
            if (sym)
                snprintf(extra, sizeof(extra), "type=sync,sym=%s", sym);
            else
                snprintf(extra, sizeof(extra),
                         "type=sync,lib=%s,offset=0x%llx",
                         lib, (unsigned long long)off);
        } else {
            snprintf(extra, sizeof(extra), "type=sync");
        }
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
    (void)encountering_task_data; (void)encountering_task_frame;
    (void)flags; (void)has_dependences;
    uint64_t task_id = (uint64_t)__sync_fetch_and_add(&g_task_id_seq, 1);
    if (new_task_data) new_task_data->value = task_id;
    const char *sym = NULL; char lib[256]; uint64_t off = 0;
    char extra[512];
    if (resolve_codeptr_full(codeptr_ra, &sym, lib, sizeof(lib), &off)) {
        if (sym)
            snprintf(extra, sizeof(extra), "type=task_create,id=%llu,sym=%s",
                     (unsigned long long)task_id, sym);
        else
            snprintf(extra, sizeof(extra),
                     "type=task_create,id=%llu,lib=%s,offset=0x%llx",
                     (unsigned long long)task_id, lib, (unsigned long long)off);
    } else {
        snprintf(extra, sizeof(extra), "type=task_create,id=%llu",
                 (unsigned long long)task_id);
    }
    uint64_t now = now_ns();
    emit_span("openmp", gettid_compat(), now, 0, "omp_task_create", extra);
}

static void cb_task_schedule(
    ompt_data_t *prior_task_data,
    int prior_task_status,
    ompt_data_t *next_task_data)
{
    uint64_t now = now_ns();
    /* Complete / suspend prior task */
    if (prior_task_data && tls_task_depth > 0 &&
        prior_task_data->value == tls_task_id[tls_task_depth - 1]) {
        tls_task_depth--;
        char extra[64];
        snprintf(extra, sizeof(extra), "type=task,id=%llu,status=%d",
                 (unsigned long long)prior_task_data->value, prior_task_status);
        emit_span("openmp", gettid_compat(),
                  tls_task_start[tls_task_depth],
                  now - tls_task_start[tls_task_depth],
                  "omp_task", extra);
    }
    /* Start next task */
    if (next_task_data && tls_task_depth < MAX_TASK_DEPTH) {
        tls_task_start[tls_task_depth] = now;
        tls_task_id[tls_task_depth]    = next_task_data->value;
        tls_task_depth++;
    }
}

static void cb_target(
    ompt_target_t kind, ompt_scope_endpoint_t endpoint,
    int device_num, ompt_data_t *task_data,
    ompt_id_t target_id, const void *codeptr_ra)
{
    (void)task_data; (void)target_id; (void)codeptr_ra;
    static const char *tnames[] = {
        "", "omp_target_enter_data", "omp_target_exit_data",
        "omp_target", "omp_target_update", "omp_target_submit"
    };
    const char *tname = (kind <= 5) ? tnames[kind] : "omp_target";
    if (*tname == '\0') tname = "omp_target";

    if (endpoint == ompt_scope_begin) {
        tls_target_start = now_ns();
    } else if (tls_target_start) {
        char extra[64];
        snprintf(extra, sizeof(extra), "type=offload,device=%d", device_num);
        emit_span("openmp", gettid_compat(),
                  tls_target_start, now_ns() - tls_target_start,
                  tname, extra);
        tls_target_start = 0;
    }
}

/* ── OMPT initialize / finalize ──────────────────────────────────────── */

static int tool_initialize(ompt_function_lookup_t lookup,
                            int initial_device_num,
                            ompt_data_t *tool_data)
{
    (void)initial_device_num; (void)tool_data;
    ensure_connected();
    cs_init();

    ompt_set_callback_t set_callback =
        (ompt_set_callback_t)lookup("ompt_set_callback");
    if (!set_callback) return 0;

#define REG(event, cb) set_callback(event, (ompt_interface_fn_t)(cb))
    REG(ompt_callback_thread_begin,   cb_thread_begin);
    REG(ompt_callback_thread_end,     cb_thread_end);
    REG(ompt_callback_parallel_begin, cb_parallel_begin);
    REG(ompt_callback_parallel_end,   cb_parallel_end);
    REG(ompt_callback_task_create,    cb_task_create);
    REG(ompt_callback_task_schedule,  cb_task_schedule);
    REG(ompt_callback_work,           cb_work);
    REG(ompt_callback_sync_region,    cb_sync_region);
    REG(ompt_callback_target,         cb_target);
#undef REG

    return 1;
}

static void tool_finalize(ompt_data_t *tool_data) {
    (void)tool_data;
    if (g_sock >= 0) { close(g_sock); g_sock = -1; }
}

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
