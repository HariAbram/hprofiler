/*
 * OpenCL hook (LD_PRELOAD).
 *
 *   clCreateCommandQueue[WithProperties]  force CL_QUEUE_PROFILING_ENABLE
 *   clEnqueueNDRangeKernel                host span (enqueue latency) plus a
 *                                         device span from the event's
 *                                         completion callback
 *   clEnqueueRead/WriteBuffer, SVMMemcpy  transfer spans (category "memory")
 *   clBuildProgram, clCompileProgram,     JIT compile spans; built binaries
 *   clLinkProgram                         are saved for disassembly, and the
 *                                         kernel spans carry mod=, the id of
 *                                         the binary they ran (hp_module.h)
 *   clFinish, clWaitForEvents             host waits ("sync")
 *   clCreateBuffer, clReleaseMemObject    opencl_memory_bytes counter
 *   dlopen, dlsym                         ACPP SSCP .jit.so CPU kernels, which
 *                                         never go through an enqueue call
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>
#include <pthread.h>
#include <dlfcn.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/syscall.h>

/* ── Minimal OpenCL types (avoid cl.h dependency at build time) ─────── */
typedef void*    cl_platform_id;
typedef void*    cl_device_id;
typedef void*    cl_context;
typedef void*    cl_command_queue;
typedef void*    cl_mem;
typedef void*    cl_program;
typedef void*    cl_kernel;
typedef void*    cl_event;
typedef int32_t  cl_int;
typedef uint32_t cl_uint;
typedef uint64_t cl_ulong;
typedef uint64_t cl_command_queue_properties;
typedef size_t   cl_event_info;
typedef size_t   cl_profiling_info;
typedef int      cl_bool;

#define CL_SUCCESS                 0
#define CL_QUEUE_PROFILING_ENABLE  (1<<1)
#define CL_PROFILING_COMMAND_START 0x1282
#define CL_PROFILING_COMMAND_END   0x1283
#define CL_MEM_READ_WRITE          (1<<0)

/*
 * Device timestamps use the OpenCL device clock, unrelated to
 * CLOCK_MONOTONIC. One (wall, device) pair taken at the first completed
 * command gives the offset for all later timestamps. The pair must use
 * CL_PROFILING_COMMAND_END: the completion callback runs right at the
 * command's end, while its START can be a whole kernel earlier (a slow first
 * kernel would shift every later span).
 *
 * Limitation: one global offset, even when devices have independent clocks
 * (two GPUs, or a GPU and the CPU device).
 */
static int64_t         g_cl_offset_ns   = INT64_MIN;
static pthread_mutex_t g_cl_offset_mutex = PTHREAD_MUTEX_INITIALIZER;

static uint64_t now_ns(void);   /* forward declaration — defined below */

/* Sets the offset once, from a COMMAND_END timestamp read in the CL_COMPLETE
 * callback (see above). Call before converting that event's START. */
static void cl_calibrate_if_needed(cl_ulong gpu_end_ns) {
    pthread_mutex_lock(&g_cl_offset_mutex);
    if (g_cl_offset_ns == INT64_MIN) {
        g_cl_offset_ns = (int64_t)now_ns() - (int64_t)gpu_end_ns;
    }
    pthread_mutex_unlock(&g_cl_offset_mutex);
}

static uint64_t cl_to_wall_ns(cl_ulong cl_ns) {
    pthread_mutex_lock(&g_cl_offset_mutex);
    int64_t offset = (g_cl_offset_ns == INT64_MIN) ? 0 : g_cl_offset_ns;
    pthread_mutex_unlock(&g_cl_offset_mutex);
    int64_t wall = (int64_t)cl_ns + offset;
    return wall > 0 ? (uint64_t)wall : 0;
}

/* ── Helpers ─────────────────────────────────────────────────────────── */
static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

static FILE *g_dbg = NULL;  /* set when HPROFILER_DEBUG=1 */

#include "../common/hp_transport.h"
#include "../common/hp_ids.h"
#include "../common/hp_module.h"
#include "../common/hp_acpp.h"
#include "../common/callstack.h"

static pid_t gettid_compat(void) { return hp_tx_tid(); }

/* Records up to the transport's 64 KiB limit go out intact (long JIT kernel
 * names included). */
static void emit_span(const char *cat, pid_t tid, uint64_t start_ns,
                      uint64_t dur_ns, const char *name, const char *extra) {
    if (!hp_tx_enabled()) return;
    if (g_dbg)
        fprintf(g_dbg, "[opencl-hook] emit_span cat=%s name=%s\n", cat, name);
    if (extra && *extra)
        hp_tx_emitf("span:%s:%d:%d:%llu:%llu:%s:%s\n", cat, (int)hp_tx_pid(), (int)tid,
                    (unsigned long long)start_ns, (unsigned long long)dur_ns, name, extra);
    else
        hp_tx_emitf("span:%s:%d:%d:%llu:%llu:%s\n", cat, (int)hp_tx_pid(), (int)tid,
                    (unsigned long long)start_ns, (unsigned long long)dur_ns, name);
    emit_callstack(start_ns);
}

/* CL_CALLBACK expands to nothing on Linux x86-64 (no special calling conv). */
#ifndef CL_CALLBACK
#define CL_CALLBACK
#endif

/* Context for the event callback. `category` ("opencl" or "memory") comes
 * from the caller, which knows what it timed. `name` is a heap copy of any
 * length (kernel names from SYCL/JIT compilers run to hundreds of chars). */
typedef struct { char *name; char extra[160]; char category[16]; } event_cb_data_t;

static void event_cb_free(event_cb_data_t *d) {
    if (!d) return;
    free(d->name);
    free(d);
}

typedef void (*event_cb_fn_t)(cl_event, cl_int, void *);
typedef cl_int (*SetCB_t)(cl_event, cl_int, event_cb_fn_t, void *);
typedef cl_int (*RetainEvent_t)(cl_event);
typedef cl_int (*ReleaseEvent_t)(cl_event);
typedef cl_int (*GetEventProf_t)(cl_event, cl_profiling_info, size_t, void *, size_t *);

/* File-scope so pthread_once init can populate them safely from any thread.
 * All four pointers are written once (inside g_ecb_once) and read-only after. */
static GetEventProf_t g_real_prof    = NULL;
static ReleaseEvent_t g_real_release = NULL;
static SetCB_t        g_real_setcb   = NULL;
static RetainEvent_t  g_real_retain  = NULL;
static pthread_once_t g_ecb_once     = PTHREAD_ONCE_INIT;

/* ── OpenCL symbol resolver ───────────────────────────────────────────── */
/* ACPP dlopens its backends with RTLD_DEEPBIND, which puts libOpenCL.so.1 in
 * a private namespace where RTLD_NEXT cannot find the real cl* symbols; fall
 * back to a handle obtained with RTLD_NOLOAD (works across namespaces).
 * pthread_once: drivers call in from several threads at startup. */
static void *s_ocl_h = NULL;
static pthread_once_t s_ocl_once = PTHREAD_ONCE_INIT;
static void _init_ocl_handle(void) {
    s_ocl_h = dlopen("libOpenCL.so.1", RTLD_LAZY | RTLD_NOLOAD);
    if (!s_ocl_h) s_ocl_h = dlopen("libOpenCL.so", RTLD_LAZY | RTLD_NOLOAD);
    if (!s_ocl_h) s_ocl_h = dlopen("libOpenCL.so.1", RTLD_LAZY | RTLD_GLOBAL);
}
static void *find_real_ocl(const char *sym) {
    void *f = dlsym(RTLD_NEXT, sym);
    if (f) return f;
    pthread_once(&s_ocl_once, _init_ocl_handle);
    return s_ocl_h ? dlsym(s_ocl_h, sym) : NULL;
}

/* Completion callbacks still to run. clFinish() can return before the
 * driver thread has run a finished command's callback, so a program whose
 * last kernel completes just before it exits would lose that kernel's
 * device span. At exit -- from an atexit handler registered here, after the
 * driver's own initialisation, so it runs before the driver is torn down --
 * the hook waits for them, at most HP_OCL_EXIT_WAIT_NS. */
static int g_cb_inflight = 0;
#define HP_OCL_EXIT_WAIT_NS 2000000000ULL

static void wait_for_callbacks_at_exit(void) {
    uint64_t t0 = now_ns();
    while (__atomic_load_n(&g_cb_inflight, __ATOMIC_ACQUIRE) > 0 && now_ns() - t0 < HP_OCL_EXIT_WAIT_NS) {
        struct timespec ts = {0, 1000000};
        nanosleep(&ts, NULL);
    }
}

/* A forked child runs none of its parent's callbacks: nothing to wait for. */
static void reset_callbacks_in_child(void) { __atomic_store_n(&g_cb_inflight, 0, __ATOMIC_RELEASE); }

static void _ecb_init_fns(void) {
    g_real_prof    = (GetEventProf_t)find_real_ocl("clGetEventProfilingInfo");
    g_real_release = (ReleaseEvent_t)find_real_ocl("clReleaseEvent");
    g_real_setcb   = (SetCB_t)find_real_ocl("clSetEventCallback");
    g_real_retain  = (RetainEvent_t)find_real_ocl("clRetainEvent");
    atexit(wait_for_callbacks_at_exit);
    pthread_atfork(NULL, NULL, reset_callbacks_in_child);
}

/* OpenCL event completion callback — fired on a driver thread when the
   kernel/transfer completes.  Reads GPU timestamps and emits the span.   */
static void CL_CALLBACK on_event_complete(cl_event ev,
                                           cl_int status, void *user_data) {
    event_cb_data_t *d = (event_cb_data_t *)user_data;
    if (status != 0 /* CL_COMPLETE */) {
        event_cb_free(d);
        __atomic_sub_fetch(&g_cb_inflight, 1, __ATOMIC_RELEASE);
        return;
    }

    pthread_once(&g_ecb_once, _ecb_init_fns);

    cl_ulong gpu_start = 0, gpu_end = 0;
    if (g_real_prof &&
        g_real_prof(ev, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &gpu_start, NULL) == CL_SUCCESS &&
        g_real_prof(ev, CL_PROFILING_COMMAND_END,   sizeof(cl_ulong), &gpu_end,   NULL) == CL_SUCCESS) {
        cl_calibrate_if_needed(gpu_end);
        uint64_t wall_start = cl_to_wall_ns(gpu_start);
        uint64_t dur_ns     = (gpu_end >= gpu_start) ? (gpu_end - gpu_start) : 0;
        /* A transfer belongs in "memory", like its host span; only kernels
         * are "opencl". */
        emit_span(d->category, gettid_compat(), wall_start, dur_ns, d->name, d->extra);
    }
    if (g_real_release) g_real_release(ev);
    event_cb_free(d);
    __atomic_sub_fetch(&g_cb_inflight, 1, __ATOMIC_RELEASE);
}

/* Register the completion callback; returns immediately. parent_sid != 0
   adds psid= (link to the host span). category: see event_cb_data_t. */
static void register_event_callback(cl_event ev,
                                    const char *name, const char *extra,
                                    uint64_t parent_sid, const char *category) {
    /* g_real_setcb / g_real_retain are initialised once via g_ecb_once. */
    pthread_once(&g_ecb_once, _ecb_init_fns);
    if (!g_real_setcb || !ev) return;

    event_cb_data_t *d = (event_cb_data_t *)malloc(sizeof(*d));
    if (!d) return;
    d->name = strdup(name ? name : "");
    if (!d->name) { free(d); return; }
    snprintf(d->category, sizeof(d->category), "%s", category ? category : "opencl");
    if (parent_sid)
        snprintf(d->extra, sizeof(d->extra), "%s,psid=%llu",
                 extra ? extra : "", (unsigned long long)parent_sid);
    else
        snprintf(d->extra, sizeof(d->extra), "%s", extra ? extra : "");

    if (g_real_retain) g_real_retain(ev);
    __atomic_add_fetch(&g_cb_inflight, 1, __ATOMIC_ACQ_REL);
    if (g_real_setcb(ev, 0 /* CL_COMPLETE */, on_event_complete, d) != CL_SUCCESS) {
        if (g_real_retain && g_real_release) g_real_release(ev);
        event_cb_free(d);
        __atomic_sub_fetch(&g_cb_inflight, 1, __ATOMIC_RELEASE);
    }
}

/* The kernel's full name as a heap string (free() it): the size is queried
 * first, so names of any length come back whole, and a failed query gives
 * "<unknown>" -- never a previous kernel's name. (A fixed 256-byte buffer
 * made clGetKernelInfo fail with CL_INVALID_VALUE for longer names, leaving
 * the last kernel's name in the buffer: BabelStream's SYCL reduction
 * kernels were reported as "triad".) */
static char *get_kernel_name(cl_kernel kernel) {
    typedef cl_int (*GetKernelInfo_t)(cl_kernel, cl_uint, size_t, void*, size_t*);
    static GetKernelInfo_t real = NULL;
    if (!real) real = (GetKernelInfo_t)find_real_ocl("clGetKernelInfo");
#define CL_KERNEL_FUNCTION_NAME 0x1190
    size_t sz = 0;
    if (real && kernel &&
        real(kernel, CL_KERNEL_FUNCTION_NAME, 0, NULL, &sz) == CL_SUCCESS && sz > 0) {
        char *buf = (char *)malloc(sz + 1);
        if (buf) {
            if (real(kernel, CL_KERNEL_FUNCTION_NAME, sz, buf, NULL) == CL_SUCCESS) {
                buf[sz] = '\0';
                if (buf[0]) return buf;
            }
            free(buf);
        }
    }
    return strdup("<unknown>");
}

/* ── Program binary -> module id ─────────────────────────────────────── */

/* (program, device) -> module id of the binary built for that device, so a
 * kernel launch can name the code it ran (cl_kernel -> CL_KERNEL_PROGRAM,
 * queue -> CL_QUEUE_DEVICE). A rebuilt program, or a new one at a reused
 * address, overwrites its entries. */
typedef struct { cl_program prog; cl_device_id dev; uint64_t mod; } OclModEntry;
static OclModEntry    *g_oclmods;
static size_t          g_oclmod_n, g_oclmod_cap;
static pthread_mutex_t g_oclmod_mutex = PTHREAD_MUTEX_INITIALIZER;

static void oclmod_set(cl_program prog, cl_device_id dev, uint64_t mod) {
    pthread_mutex_lock(&g_oclmod_mutex);
    for (size_t i = 0; i < g_oclmod_n; i++) {
        if (g_oclmods[i].prog == prog && g_oclmods[i].dev == dev) {
            g_oclmods[i].mod = mod;
            pthread_mutex_unlock(&g_oclmod_mutex);
            return;
        }
    }
    if (g_oclmod_n == g_oclmod_cap) {
        size_t cap = g_oclmod_cap ? g_oclmod_cap * 2 : 64;
        OclModEntry *p = (OclModEntry *)realloc(g_oclmods, cap * sizeof(*p));
        if (!p) { pthread_mutex_unlock(&g_oclmod_mutex); return; }
        g_oclmods = p;
        g_oclmod_cap = cap;
    }
    g_oclmods[g_oclmod_n++] = (OclModEntry){prog, dev, mod};
    pthread_mutex_unlock(&g_oclmod_mutex);
}

/* Module id of the code a launch of `kernel` on `q` runs (0: unknown). With
 * no entry for the queue's device, a program built for one device only
 * still answers. */
static uint64_t kernel_module_id(cl_kernel kernel, cl_command_queue q) {
    typedef cl_int (*GetKI_t)(cl_kernel, cl_uint, size_t, void*, size_t*);
    typedef cl_int (*GetQI_t)(cl_command_queue, cl_uint, size_t, void*, size_t*);
    static GetKI_t gki = NULL;
    static GetQI_t gqi = NULL;
    if (!gki) gki = (GetKI_t)find_real_ocl("clGetKernelInfo");
    if (!gqi) gqi = (GetQI_t)find_real_ocl("clGetCommandQueueInfo");
#define CL_KERNEL_PROGRAM 0x1194
#define CL_QUEUE_DEVICE   0x1091
    cl_program prog = NULL;
    cl_device_id dev = NULL;
    if (!gki || !kernel || gki(kernel, CL_KERNEL_PROGRAM, sizeof(prog), &prog, NULL) != CL_SUCCESS || !prog)
        return 0;
    if (gqi && q) gqi(q, CL_QUEUE_DEVICE, sizeof(dev), &dev, NULL);
    uint64_t mod = 0, only = 0;
    int nprog = 0;
    pthread_mutex_lock(&g_oclmod_mutex);
    for (size_t i = 0; i < g_oclmod_n; i++) {
        if (g_oclmods[i].prog != prog) continue;
        if (dev && g_oclmods[i].dev == dev) { mod = g_oclmods[i].mod; break; }
        only = g_oclmods[i].mod;
        nprog++;
    }
    pthread_mutex_unlock(&g_oclmod_mutex);
    return mod ? mod : (nprog == 1 ? only : 0);
}

/* ── clCreateCommandQueue ────────────────────────────────────────────── */

cl_command_queue clCreateCommandQueue(
    cl_context ctx, cl_device_id dev,
    cl_command_queue_properties props, cl_int *err)
{
    typedef cl_command_queue (*fn_t)(cl_context, cl_device_id,
                                     cl_command_queue_properties, cl_int*);
    static fn_t real = NULL;
    if (!real) real = (fn_t)find_real_ocl("clCreateCommandQueue");
    if (!real) { if (err) *err = -6; return NULL; }
    /* Force profiling on. */
    return real(ctx, dev, props | CL_QUEUE_PROFILING_ENABLE, err);
}

/* clCreateCommandQueueWithProperties (OpenCL 2.0+) */
cl_command_queue clCreateCommandQueueWithProperties(
    cl_context ctx, cl_device_id dev,
    const cl_command_queue_properties *props, cl_int *err)
{
    typedef cl_command_queue (*fn_t)(cl_context, cl_device_id,
                                     const cl_command_queue_properties*, cl_int*);
    static fn_t real = NULL;
    if (!real) real = (fn_t)find_real_ocl("clCreateCommandQueueWithProperties");
    if (!real) { if (err) *err = -6; return NULL; }

    /* Build a modified props array with PROFILING_ENABLE inserted.
     * Reserve 3 slots: the profiling pair + the terminating 0. */
    cl_command_queue_properties new_props[32] = {0};
    int n = 0;
    int already_profiling = 0;
    if (props) {
        for (int i = 0; props[i]; i += 2) {
            if (n + 3 > 31) break;  /* leave room for insert pair + terminator */
            new_props[n++] = props[i];
            if ((uint64_t)props[i] == 0x1093 /* CL_QUEUE_PROPERTIES */) {
                new_props[n++] = props[i+1] | CL_QUEUE_PROFILING_ENABLE;
                already_profiling = 1;
            } else {
                new_props[n++] = props[i+1];
            }
        }
    }
    if (!already_profiling && n + 3 <= 32) {
        new_props[n++] = 0x1093; /* CL_QUEUE_PROPERTIES */
        new_props[n++] = CL_QUEUE_PROFILING_ENABLE;
    }
    new_props[n] = 0;
    return real(ctx, dev, new_props, err);
}

/* ── clEnqueueNDRangeKernel ──────────────────────────────────────────── */

cl_int clEnqueueNDRangeKernel(
    cl_command_queue q, cl_kernel kernel, cl_uint work_dim,
    const size_t *global_work_offset, const size_t *global_work_size,
    const size_t *local_work_size, cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list, cl_event *event)
{
    typedef cl_int (*fn_t)(cl_command_queue, cl_kernel, cl_uint,
                            const size_t*, const size_t*, const size_t*,
                            cl_uint, const cl_event*, cl_event*);
    static fn_t real = NULL;
    if (!real) real = (fn_t)find_real_ocl("clEnqueueNDRangeKernel");

    char *kname_heap = get_kernel_name(kernel);
    const char *kname = kname_heap ? kname_heap : "<unknown>";
    if (g_dbg) { fprintf(g_dbg, "[opencl-hook] clEnqueueNDRangeKernel kernel=%s\n", kname); fflush(g_dbg); }

    /* If the caller doesn't want an event, create an internal one. */
    cl_event internal_ev = NULL;
    cl_bool  our_event   = 0;
    if (!event) { event = &internal_ev; our_event = 1; }

    uint64_t cpu_t0 = now_ns();
    cl_int ret = real(q, kernel, work_dim,
                      global_work_offset, global_work_size, local_work_size,
                      num_events_in_wait_list, event_wait_list, event);
    uint64_t cpu_dur = now_ns() - cpu_t0;
    if (g_dbg) { fprintf(g_dbg, "[opencl-hook] enqueue ret=%d event=%p\n", ret, event ? *event : NULL); fflush(g_dbg); }

    if (ret == CL_SUCCESS && *event) {
        /* CPU-side span: time from enqueue call to return (scheduling latency),
           with a span id (hp_ids.h) the GPU execution span links back to. */
        static uint32_t ocl_seq = 0;
        uint64_t sid = hp_span_id(HP_ID_OPENCL, (uint32_t)getpid(),
                                  __atomic_add_fetch(&ocl_seq, 1, __ATOMIC_RELAXED));
        char modtag[24], cpu_extra[96], gpu_extra[64];
        hp_mod_tag(modtag, sizeof(modtag), kernel_module_id(kernel, q));
        snprintf(cpu_extra, sizeof(cpu_extra), "type=kernel,side=cpu,sid=%llu%s",
                 (unsigned long long)sid, modtag);
        snprintf(gpu_extra, sizeof(gpu_extra), "type=kernel,side=gpu%s", modtag);
        emit_span("opencl", gettid_compat(), cpu_t0, cpu_dur, kname, cpu_extra);
        /* GPU-side span: async callback carries psid= linking back to the CPU span. */
        register_event_callback(*event, kname, gpu_extra, sid, "opencl");
        /* Release our internal event reference if the caller didn't want it. */
        if (our_event) {
            typedef cl_int (*ReleaseEvent_t)(cl_event);
            static ReleaseEvent_t real_release = NULL;
            if (!real_release) real_release = (ReleaseEvent_t)find_real_ocl("clReleaseEvent");
            if (real_release) real_release(*event);
        }
    }
    free(kname_heap);
    return ret;
}

/* ── clEnqueueSVMMemcpy (ACPP uses SVM for transfers) ───────────────── */

cl_int clEnqueueSVMMemcpy(
    cl_command_queue q, cl_bool blocking,
    void *dst, const void *src, size_t size,
    cl_uint nwl, const cl_event *ewl, cl_event *ev)
{
    typedef cl_int (*fn_t)(cl_command_queue, cl_bool, void*, const void*,
                            size_t, cl_uint, const cl_event*, cl_event*);
    static fn_t real = NULL;
    if (!real) real = (fn_t)find_real_ocl("clEnqueueSVMMemcpy");
    if (!real) return -1; /* CL_INVALID_OPERATION */
    char extra[64]; snprintf(extra, sizeof(extra), "type=svm_memcpy,bytes=%zu", size);
    cl_event iev = NULL;
    if (!ev) ev = &iev;
    uint64_t t0 = now_ns();
    cl_int ret = real(q, blocking, dst, src, size, nwl, ewl, ev);
    emit_span("memory", gettid_compat(), t0, now_ns()-t0, "clEnqueueSVMMemcpy", extra);
    /* Device callback only for non-blocking calls: a blocking call's host
     * span already covers the whole transfer, and a callback span would
     * count it twice. */
    if (ret == CL_SUCCESS && *ev && !blocking)
        register_event_callback(*ev, "clEnqueueSVMMemcpy", extra, 0, "memory");
    return ret;
}

/* ── Buffer transfers ────────────────────────────────────────────────── */

cl_int clEnqueueReadBuffer(
    cl_command_queue q, cl_mem buf, cl_bool blocking, size_t offset,
    size_t size, void *ptr, cl_uint nwl, const cl_event *ewl, cl_event *ev)
{
    typedef cl_int (*fn_t)(cl_command_queue, cl_mem, cl_bool, size_t, size_t,
                            void*, cl_uint, const cl_event*, cl_event*);
    static fn_t real = NULL;
    if (!real) real = (fn_t)find_real_ocl("clEnqueueReadBuffer");
    if (!real) return -1; /* CL_INVALID_OPERATION */

    cl_event iev = NULL;
    if (!ev) ev = &iev;
    char extra[64]; snprintf(extra, sizeof(extra), "type=read,bytes=%zu", size);
    uint64_t t0 = now_ns();
    cl_int ret = real(q, buf, blocking, offset, size, ptr, nwl, ewl, ev);
    emit_span("memory", gettid_compat(), t0, now_ns()-t0, "clEnqueueReadBuffer", extra);
    /* Non-blocking only (see clEnqueueSVMMemcpy). */
    if (ret == CL_SUCCESS && *ev && !blocking)
        register_event_callback(*ev, "clReadBuffer_gpu", extra, 0, "memory");
    return ret;
}

cl_int clEnqueueWriteBuffer(
    cl_command_queue q, cl_mem buf, cl_bool blocking, size_t offset,
    size_t size, const void *ptr, cl_uint nwl, const cl_event *ewl, cl_event *ev)
{
    typedef cl_int (*fn_t)(cl_command_queue, cl_mem, cl_bool, size_t, size_t,
                            const void*, cl_uint, const cl_event*, cl_event*);
    static fn_t real = NULL;
    if (!real) real = (fn_t)find_real_ocl("clEnqueueWriteBuffer");
    if (!real) return -1; /* CL_INVALID_OPERATION */

    cl_event iev = NULL;
    if (!ev) ev = &iev;
    char extra[64]; snprintf(extra, sizeof(extra), "type=write,bytes=%zu", size);
    uint64_t t0 = now_ns();
    cl_int ret = real(q, buf, blocking, offset, size, ptr, nwl, ewl, ev);
    emit_span("memory", gettid_compat(), t0, now_ns()-t0, "clEnqueueWriteBuffer", extra);
    /* NON-blocking only -- see clEnqueueSVMMemcpy's comment above. */
    if (ret == CL_SUCCESS && *ev && !blocking)
        register_event_callback(*ev, "clWriteBuffer_gpu", extra, 0, "memory");
    return ret;
}

/* ── clBuildProgram: tracks JIT compilation time (ACPP SSCP, etc.) ─── */

/* Extract the compiled ELF binary from a successfully-built OpenCL program.
 * CPU OpenCL backends (POCL, Intel CPU OCL) return an ELF that objdump can
 * disassemble; GPU backends return GPU ISA — we skip those (no ELF magic).
 * Saves to /tmp/hprofiler_ocl_<pid>_<n>.bin and emits a jit_load span so
 * the runner's disasm extractor picks it up automatically. */

/* "[off, off+len) within [0, sz)" without computing off+len, which can wrap
 * on a corrupt ELF header. */
static int _range_ok(uint64_t off, uint64_t len, uint64_t sz)
{
    return len <= sz && off <= sz - len;
}

/* Intel CPU OCL wraps the compiled x86 ELF in a proprietary outer ELF with
 * e_type=0xff04 and e_version=0.  The real relocatable is in ".ocl.obj".
 * Returns 1 and sets *out / *outsz if successfully unwrapped. */
static int _unwrap_intel_ocl(const unsigned char *buf, size_t sz,
                              const unsigned char **out, size_t *outsz)
{
    if (sz < 0x40) return 0;
    uint16_t e_type;
    memcpy(&e_type, buf + 0x10, 2);
    if (e_type != 0xff04u) return 0; /* not Intel OCL wrapper */

    uint64_t e_shoff;
    uint16_t e_shentsize, e_shnum, e_shstrndx;
    memcpy(&e_shoff,     buf + 0x28, 8);
    memcpy(&e_shentsize, buf + 0x3a, 2);
    memcpy(&e_shnum,     buf + 0x3c, 2);
    memcpy(&e_shstrndx,  buf + 0x3e, 2);

    if (e_shnum == 0 || e_shentsize < 64) return 0;
    if (!_range_ok(e_shoff, (uint64_t)e_shnum * e_shentsize, sz)) return 0;
    if (e_shstrndx >= e_shnum) return 0;

    /* String table section */
    const unsigned char *shstr_sh = buf + e_shoff + (uint64_t)e_shstrndx * e_shentsize;
    uint64_t strtab_off, strtab_sz;
    memcpy(&strtab_off, shstr_sh + 0x18, 8);
    memcpy(&strtab_sz,  shstr_sh + 0x20, 8);
    if (!_range_ok(strtab_off, strtab_sz, sz)) return 0;
    const char *strtab = (const char *)(buf + strtab_off);

    for (uint16_t i = 0; i < e_shnum; i++) {
        const unsigned char *sh = buf + e_shoff + (uint64_t)i * e_shentsize;
        uint32_t sh_name;
        memcpy(&sh_name, sh, 4);
        if (sh_name >= (uint32_t)strtab_sz) continue;
        /* strtab[sh_name .. strtab_sz) must contain a NUL before strcmp may
         * read it -- a malformed/truncated string table could otherwise
         * make strcmp read past the mapped binary buffer. */
        if (!memchr(strtab + sh_name, '\0', strtab_sz - sh_name)) continue;
        if (strcmp(strtab + sh_name, ".ocl.obj") != 0) continue;
        uint64_t sh_offset, sh_size;
        memcpy(&sh_offset, sh + 0x18, 8);
        memcpy(&sh_size,   sh + 0x20, 8);
        if (sh_size == 0 || !_range_ok(sh_offset, sh_size, sz)) return 0;
        *out   = buf + sh_offset;
        *outsz = (size_t)sh_size;
        return 1;
    }
    return 0;
}

static void _extract_ocl_binary(cl_program program) {
    typedef cl_int (*GetPI_t)(cl_program, cl_uint, size_t, void*, size_t*);
    static GetPI_t gpi = NULL;
    if (!gpi) gpi = (GetPI_t)find_real_ocl("clGetProgramInfo");
    if (!gpi) return;

#define _OCL_PROG_NUM_DEVICES   0x1162u
#define _OCL_PROG_DEVICES       0x1163u
#define _OCL_PROG_BINARY_SIZES  0x1165u
#define _OCL_PROG_BINARIES      0x1166u

    cl_uint ndev = 0;
    cl_int r1 = gpi(program, _OCL_PROG_NUM_DEVICES, sizeof(ndev), &ndev, NULL);
    if (r1 || ndev == 0) return;

    size_t *bsizes = (size_t *)calloc(ndev, sizeof(size_t));
    cl_device_id *devs = (cl_device_id *)calloc(ndev, sizeof(cl_device_id));
    if (!bsizes || !devs) { free(bsizes); free(devs); return; }
    cl_int r2 = gpi(program, _OCL_PROG_BINARY_SIZES, ndev * sizeof(size_t), bsizes, NULL);
    if (r2) { free(bsizes); free(devs); return; }
    if (gpi(program, _OCL_PROG_DEVICES, ndev * sizeof(cl_device_id), devs, NULL) != CL_SUCCESS)
        memset(devs, 0, ndev * sizeof(cl_device_id));

    unsigned char **bins = (unsigned char **)calloc(ndev, sizeof(unsigned char *));
    if (!bins) { free(bsizes); free(devs); return; }
    for (cl_uint i = 0; i < ndev; i++) {
        if (bsizes[i] > 0) bins[i] = (unsigned char *)malloc(bsizes[i]);
    }

    cl_int r3 = gpi(program, _OCL_PROG_BINARIES, ndev * sizeof(unsigned char *), bins, NULL);
    if (r3 == CL_SUCCESS) {
        static int ocl_bin_n = 0;
        for (cl_uint i = 0; i < ndev; i++) {
            if (!bins[i] || bsizes[i] < 4) continue;
            int is_elf = bins[i][0] == 0x7f && bins[i][1] == 'E' &&
                         bins[i][2] == 'L'  && bins[i][3] == 'F';
            /* NVIDIA's OpenCL returns PTX text: saved too (the extractor
             * parses PTX) */
            int is_ptx = (bins[i][0] == '/' && bins[i][1] == '/') || bins[i][0] == '.';
            if (!is_elf && !is_ptx) {
                /* other GPU ISA: not disassembled, but launches still name
                 * the binary they ran. */
                oclmod_set(program, devs[i], hp_fnv1a64(bins[i], bsizes[i]));
                continue;
            }
            /* Peel Intel CPU OCL outer wrapper (e_type=0xff04) to get the
             * standard x86-64 ELF relocatable stored in .ocl.obj. */
            const unsigned char *obj_data = bins[i];
            size_t obj_sz = bsizes[i];
            const unsigned char *inner = NULL;
            size_t inner_sz = 0;
            if (is_elf && _unwrap_intel_ocl(bins[i], bsizes[i], &inner, &inner_sz)) {
                obj_data = inner;
                obj_sz   = inner_sz;
            }
            /* The module id hashes exactly the bytes saved below, so the
             * disassembler can recompute it from the file. */
            uint64_t mod = hp_fnv1a64(obj_data, obj_sz);
            oclmod_set(program, devs[i], mod);
            int idx = __atomic_fetch_add(&ocl_bin_n, 1, __ATOMIC_RELAXED);
            char saved[512];
            snprintf(saved, sizeof(saved), "/tmp/hprofiler_ocl_%d_%d.bin",
                     (int)getpid(), idx);
            FILE *fp = fopen(saved, "wb");
            if (fp) {
                fwrite(obj_data, 1, obj_sz, fp);
                fclose(fp);
                char extra[640], modtag[24];
                snprintf(extra, sizeof(extra), "type=jit_load,stage=load,path=%s%s", saved,
                         hp_mod_tag(modtag, sizeof(modtag), mod));
                emit_span("jit", gettid_compat(), now_ns(), 0, "ocl_kernel.bin", extra);
            }
        }
    }

    for (cl_uint i = 0; i < ndev; i++) free(bins[i]);
    free(bins);
    free(bsizes);
    free(devs);
}

/* stage= of a build: "source" when the program was created from OpenCL C
 * (the driver runs front and back end), "backend" for IL (SPIR-V, e.g.
 * AdaptiveCpp SSCP, whose own IR generation happens before this call and
 * outside every span) or binaries. */
static const char *ocl_build_stage(cl_program program) {
    typedef cl_int (*GetPI_t)(cl_program, cl_uint, size_t, void*, size_t*);
    static GetPI_t gpi = NULL;
    if (!gpi) gpi = (GetPI_t)find_real_ocl("clGetProgramInfo");
#define _OCL_PROG_SOURCE 0x1164u
    size_t n = 0;
    if (gpi && gpi(program, _OCL_PROG_SOURCE, 0, NULL, &n) == CL_SUCCESS && n > 1)
        return "source";
    return "backend";
}

cl_int clBuildProgram(
    cl_program program, cl_uint num_devices, const cl_device_id *device_list,
    const char *options, void (*pfn_notify)(cl_program, void*), void *user_data)
{
    typedef cl_int (*fn_t)(cl_program, cl_uint, const cl_device_id*,
                            const char*, void(*)(cl_program,void*), void*);
    static fn_t real = NULL;
    if (!real) real = (fn_t)find_real_ocl("clBuildProgram");

    const char *stage = ocl_build_stage(program);
    uint64_t t0 = now_ns();
    cl_int ret = real(program, num_devices, device_list, options, pfn_notify, user_data);
    char extra[48];
    snprintf(extra, sizeof(extra), "status=%s,stage=%s", ret == CL_SUCCESS ? "ok" : "err", stage);
    emit_span("jit", gettid_compat(), t0, now_ns()-t0, "clBuildProgram", extra);
    if (ret == CL_SUCCESS) _extract_ocl_binary(program);
    return ret;
}

cl_int clCompileProgram(
    cl_program program, cl_uint num_devices, const cl_device_id *device_list,
    const char *options, cl_uint num_input_headers,
    const cl_program *input_headers, const char **header_include_names,
    void (*pfn_notify)(cl_program, void*), void *user_data)
{
    typedef cl_int (*fn_t)(cl_program, cl_uint, const cl_device_id*,
                            const char*, cl_uint, const cl_program*,
                            const char**, void(*)(cl_program,void*), void*);
    static fn_t real = NULL;
    if (!real) real = (fn_t)find_real_ocl("clCompileProgram");

    const char *stage = ocl_build_stage(program);
    uint64_t t0 = now_ns();
    cl_int ret = real(program, num_devices, device_list, options,
                      num_input_headers, input_headers, header_include_names,
                      pfn_notify, user_data);
    char extra[48];
    snprintf(extra, sizeof(extra), "type=jit_compile,stage=%s", stage);
    emit_span("jit", gettid_compat(), t0, now_ns()-t0, "clCompileProgram", extra);
    return ret;
}

/* A linked program never goes through clBuildProgram: save its binaries
 * here, so its kernels get disassembly and a module id too. */
cl_program clLinkProgram(
    cl_context ctx, cl_uint num_devices, const cl_device_id *device_list,
    const char *options, cl_uint num_input_programs,
    const cl_program *input_programs,
    void (*pfn_notify)(cl_program, void*), void *user_data, cl_int *errcode_ret)
{
    typedef cl_program (*fn_t)(cl_context, cl_uint, const cl_device_id*,
                                const char*, cl_uint, const cl_program*,
                                void(*)(cl_program,void*), void*, cl_int*);
    static fn_t real = NULL;
    if (!real) real = (fn_t)find_real_ocl("clLinkProgram");
    if (!real) { if (errcode_ret) *errcode_ret = -6; return NULL; }

    cl_int err = CL_SUCCESS;
    uint64_t t0 = now_ns();
    cl_program prog = real(ctx, num_devices, device_list, options,
                           num_input_programs, input_programs, pfn_notify, user_data, &err);
    emit_span("jit", gettid_compat(), t0, now_ns()-t0,
              "clLinkProgram", err == CL_SUCCESS ? "type=jit_compile,stage=backend,status=ok"
                                                 : "type=jit_compile,stage=backend,status=err");
    if (errcode_ret) *errcode_ret = err;
    /* With a callback the link may still be running: its binary is not
     * available yet (like an asynchronous clBuildProgram). */
    if (prog && err == CL_SUCCESS && !pfn_notify) _extract_ocl_binary(prog);
    return prog;
}

/* ── clFinish / clWaitForEvents sync spans ──────────────────────────── */

cl_int clFinish(cl_command_queue queue) {
    typedef cl_int (*fn_t)(cl_command_queue);
    static fn_t real = NULL;
    if (!real) real = (fn_t)find_real_ocl("clFinish");
    uint64_t t0 = now_ns();
    cl_int ret = real ? real(queue) : -1;
    emit_span("sync", gettid_compat(), t0, now_ns()-t0, "clFinish", "type=sync");
    return ret;
}

cl_int clWaitForEvents(cl_uint num_events, const cl_event *event_list) {
    typedef cl_int (*fn_t)(cl_uint, const cl_event*);
    static fn_t real = NULL;
    if (!real) real = (fn_t)find_real_ocl("clWaitForEvents");
    char extra[64]; snprintf(extra, sizeof(extra), "type=sync,count=%u", num_events);
    uint64_t t0 = now_ns();
    cl_int ret = real ? real(num_events, event_list) : -1;
    emit_span("sync", gettid_compat(), t0, now_ns()-t0, "clWaitForEvents", extra);
    return ret;
}

/* ── clCreateBuffer / clReleaseMemObject alloc tracking ─────────────── */
typedef uint64_t cl_mem_flags;

#define CL_ALLOC_CAP 1024
typedef struct { cl_mem handle; size_t sz; } ClAllocRec;
static ClAllocRec      g_cl_allocs[CL_ALLOC_CAP];
static int             g_cl_alloc_n   = 0;
static int64_t         g_cl_mem_bytes = 0;
static pthread_mutex_t g_cl_alloc_mtx = PTHREAD_MUTEX_INITIALIZER;

static void emit_ctr_cl(const char *name, int64_t value) {
    if (!hp_tx_enabled()) return;
    hp_tx_emitf("ctr:memory:%d:%llu:%s:%lld:bytes\n", (int)hp_tx_pid(),
                (unsigned long long)now_ns(), name, (long long)value);
}

cl_mem clCreateBuffer(cl_context ctx, cl_mem_flags flags, size_t size,
                       void *host_ptr, cl_int *err) {
    typedef cl_mem (*fn_t)(cl_context, cl_mem_flags, size_t, void*, cl_int*);
    static fn_t real = NULL;
    if (!real) real = (fn_t)find_real_ocl("clCreateBuffer");
    uint64_t t0 = now_ns();
    cl_mem ret = real ? real(ctx, flags, size, host_ptr, err) : NULL;
    uint64_t dur = now_ns() - t0;
    if (ret) {
        pthread_mutex_lock(&g_cl_alloc_mtx);
        if (g_cl_alloc_n < CL_ALLOC_CAP) {
            g_cl_allocs[g_cl_alloc_n].handle = ret;
            g_cl_allocs[g_cl_alloc_n].sz     = size;
            g_cl_alloc_n++;
        }
        g_cl_mem_bytes += (int64_t)size;
        int64_t total = g_cl_mem_bytes;
        pthread_mutex_unlock(&g_cl_alloc_mtx);
        emit_ctr_cl("opencl_memory_bytes", total);
        char extra[64]; snprintf(extra, sizeof(extra), "type=alloc,bytes=%zu", size);
        emit_span("memory", gettid_compat(), t0, dur, "clCreateBuffer", extra);
    }
    return ret;
}

cl_int clReleaseMemObject(cl_mem memobj) {
    typedef cl_int (*fn_t)(cl_mem);
    static fn_t real = NULL;
    if (!real) real = (fn_t)find_real_ocl("clReleaseMemObject");
    pthread_mutex_lock(&g_cl_alloc_mtx);
    for (int i = 0; i < g_cl_alloc_n; i++) {
        if (g_cl_allocs[i].handle == memobj) {
            g_cl_mem_bytes -= (int64_t)g_cl_allocs[i].sz;
            g_cl_allocs[i] = g_cl_allocs[--g_cl_alloc_n];
            break;
        }
    }
    int64_t total = g_cl_mem_bytes;
    pthread_mutex_unlock(&g_cl_alloc_mtx);
    emit_ctr_cl("opencl_memory_bytes", total);
    uint64_t t0_rel = now_ns();
    cl_int rret = real ? real(memobj) : -1;
    emit_span("memory", gettid_compat(), t0_rel, now_ns() - t0_rel, "clReleaseMemObject", "type=free");
    return rret;
}

/* ── dlopen intercept ────────────────────────────────────────────────────
   Two purposes:
   1. Strip RTLD_DEEPBIND so ACPP's backend plugins resolve CL symbols
      through the global scope (where our wrappers live).
   2. Detect ACPP SSCP ".jit.so" kernel files: emit a jit_load span and
      register the handle so dlsym calls on it are intercepted below.
   ACPP SSCP dispatches CPU kernels entirely through native JIT-compiled
   .so files — clEnqueueNDRangeKernel is never called.  Hooking dlopen
   and dlsym on these files is the only way to capture kernel activity.  */

#ifndef RTLD_DEEPBIND
#define RTLD_DEEPBIND 0x8
#endif

#define MAX_JIT_HANDLES 128
static void     *g_jit_handles[MAX_JIT_HANDLES];
static char      g_jit_names[MAX_JIT_HANDLES][64]; /* short basename */
static uint64_t  g_jit_mods[MAX_JIT_HANDLES];      /* module id of the .jit.so (hp_module.h) */
static int       g_jit_count = 0;
static pthread_mutex_t g_jit_mutex = PTHREAD_MUTEX_INITIALIZER;

/* 1 if h is a registered .jit.so handle; *mod gets its module id. */
static int is_jit_handle(void *h, uint64_t *mod) {
    pthread_mutex_lock(&g_jit_mutex);
    for (int i = 0; i < g_jit_count; i++) {
        if (g_jit_handles[i] == h) {
            if (mod) *mod = g_jit_mods[i];
            pthread_mutex_unlock(&g_jit_mutex);
            return 1;
        }
    }
    pthread_mutex_unlock(&g_jit_mutex);
    return 0;
}

void *dlopen(const char *filename, int flags) {
    typedef void *(*dlopen_t)(const char *, int);
    static dlopen_t real_dlopen = NULL;
    if (!real_dlopen) real_dlopen = (dlopen_t)dlsym(RTLD_NEXT, "dlopen");

    if (g_dbg && filename) {
        fprintf(g_dbg, "[opencl-hook] dlopen %s flags=0x%x deepbind=%d\n",
                filename, flags, !!(flags & RTLD_DEEPBIND));
        fflush(g_dbg);
    }
    /* Strip RTLD_DEEPBIND ONLY for ACPP/SYCL libraries so their OpenCL calls
     * resolve through our wrappers.  Leave it intact for Intel's internal
     * runtime plugins — they depend on RTLD_DEEPBIND for symbol isolation and
     * crash when it is removed (intermittent Intel OpenCL abort). */
    if (filename && (strstr(filename, ".jit.so") ||
                     strstr(filename, "acpp")    ||
                     strstr(filename, "sycl")    ||
                     strstr(filename, "hipsycl") ||
                     strstr(filename, "sscp"))) {
        flags &= ~RTLD_DEEPBIND;
    }

    /* For ACPP SSCP .jit.so files: copy to /tmp BEFORE dlopen so the file
       still exists for the disassembler even if ACPP unlinks it after load. */
    static int jit_copy_n = 0;
    char saved_jit_path[512] = {0};
    uint64_t jit_mod = 0;    /* hash of exactly the bytes copied (hp_module.h) */
    if (filename && strstr(filename, ".jit.so")) {
        FILE *src = fopen(filename, "rb");
        if (src) {
            int jit_idx = __atomic_fetch_add(&jit_copy_n, 1, __ATOMIC_RELAXED);
            snprintf(saved_jit_path, sizeof(saved_jit_path),
                     "/tmp/hprofiler_jit_%d_%d.so", (int)getpid(), jit_idx);
            FILE *dst = fopen(saved_jit_path, "wb");
            if (dst) {
                char buf[65536];
                size_t n;
                int copy_ok = 1;
                uint64_t h = HP_FNV1A64_INIT;
                while ((n = fread(buf, 1, sizeof(buf), src)) > 0) {
                    if (fwrite(buf, 1, n, dst) != n) { copy_ok = 0; break; }
                    h = hp_fnv1a64_update(h, buf, n);
                }
                fclose(dst);
                if (!copy_ok) { remove(saved_jit_path); saved_jit_path[0] = 0; }
                else jit_mod = h;
            } else {
                saved_jit_path[0] = 0;
            }
            fclose(src);
        }
    }

    uint64_t t0 = now_ns();
    void *h = real_dlopen(filename, flags);
    uint64_t dur = now_ns() - t0;

    /* Detect ACPP SSCP kernel .so files by path suffix. */
    if (h && filename && strstr(filename, ".jit.so")) {
        const char *base = strrchr(filename, '/');
        base = base ? base + 1 : filename;

        /* Emit path of the SAVED copy so Python can disassemble it later. */
        const char *disasm_path = saved_jit_path[0] ? saved_jit_path : filename;
        char jit_extra[640], modtag[24];
        snprintf(jit_extra, sizeof(jit_extra), "type=jit_load,stage=load,path=%s%s", disasm_path,
                 hp_mod_tag(modtag, sizeof(modtag), jit_mod));
        emit_span("jit", gettid_compat(), t0, dur, base, jit_extra);

        /* Register this handle so dlsym calls on it are intercepted. */
        pthread_mutex_lock(&g_jit_mutex);
        if (g_jit_count < MAX_JIT_HANDLES) {
            g_jit_handles[g_jit_count] = h;
            g_jit_mods[g_jit_count] = jit_mod;
            snprintf(g_jit_names[g_jit_count], sizeof(g_jit_names[0]),
                     "%s", base);
            g_jit_count++;
        }
        pthread_mutex_unlock(&g_jit_mutex);

        if (g_dbg) {
            fprintf(g_dbg, "[opencl-hook] jit_load registered handle=%p %s dur=%lluns\n",
                    h, base, (unsigned long long)dur);
            fflush(g_dbg);
        }
    }
    return h;
}

/* ── dlsym intercept ─────────────────────────────────────────────────────
   ACPP SSCP looks up compiled CPU kernels with dlsym(jit_handle, name). For
   such handles the real function goes into a slot and a per-slot wrapper
   (trampoline) is returned that times every call of the kernel.
   TRAMPOLINE_N slots exist; beyond that the real function is returned
   unwrapped. */

static void *(*g_real_dlsym)(void *, const char *) = NULL;

/* ── trampoline pool ────────────────────────────────────────────────────── */
#define TRAMPOLINE_N  256

/* Category of a JIT-kernel call span: these are SYCL kernels compiled for
   and run on the host (AdaptiveCpp SSCP .jit.so, e.g. its OpenMP backend);
   this hook only times them -- no OpenCL is involved. Older builds wrote
   "opencl"; hprofiler relabels those (events.is_legacy_host_jit_kernel). */
#define HP_JIT_KERNEL_CATEGORY "sycl"

typedef struct {
    void    (*real_fn)(void);   /* real kernel entry point (cast to fn ptr) */
    char    *name;              /* kernel name for span emission (heap) */
    char     extra[48];         /* span tags: type=jit_kernel,side=cpu[,mod=] */
    int      used;
} trampoline_slot_t;

static trampoline_slot_t  g_slots[TRAMPOLINE_N];
static int                g_slot_next = 0;
static pthread_mutex_t    g_slot_mutex = PTHREAD_MUTEX_INITIALIZER;

/* Forward declaration; defined after the macro expansion. */
static void jit_dispatch(int slot, void *arg0, void *arg1, void *arg2,
                          void *arg3, void *arg4, void *arg5);

/* TRAMPOLINE_N wrappers, each passing its slot index and the first six
   pointer-sized arguments (ACPP CPU kernels take a launch-config pointer and
   an argument array). N is a three-digit token of octal digits (000..377,
   see T64 below), so the slot index is the octal value 0##N -- the
   wrapper's position in g_trampolines (N alone would read "100" as
   decimal). */
#define DEF_TRAMPOLINE(N) \
static void trampoline_##N(void *a0, void *a1, void *a2, \
                            void *a3, void *a4, void *a5) { \
    jit_dispatch(0##N, a0, a1, a2, a3, a4, a5); \
}

/* Expand 256 trampolines. */
#define T8(B)  DEF_TRAMPOLINE(B##0) DEF_TRAMPOLINE(B##1) DEF_TRAMPOLINE(B##2) DEF_TRAMPOLINE(B##3) \
               DEF_TRAMPOLINE(B##4) DEF_TRAMPOLINE(B##5) DEF_TRAMPOLINE(B##6) DEF_TRAMPOLINE(B##7)
#define T64(B) T8(B##0) T8(B##1) T8(B##2) T8(B##3) T8(B##4) T8(B##5) T8(B##6) T8(B##7)
T64(0) T64(1) T64(2) T64(3)
#undef T8
#undef T64
#undef DEF_TRAMPOLINE

/* Table of trampoline function pointers. */
typedef void (*trampoline_fn_t)(void*, void*, void*, void*, void*, void*);
#define TRAMPOLINE_REF(N) trampoline_##N,
static const trampoline_fn_t g_trampolines[TRAMPOLINE_N] = {
#define T8(B) \
    TRAMPOLINE_REF(B##0) TRAMPOLINE_REF(B##1) TRAMPOLINE_REF(B##2) TRAMPOLINE_REF(B##3) \
    TRAMPOLINE_REF(B##4) TRAMPOLINE_REF(B##5) TRAMPOLINE_REF(B##6) TRAMPOLINE_REF(B##7)
#define T64(B) T8(B##0) T8(B##1) T8(B##2) T8(B##3) T8(B##4) T8(B##5) T8(B##6) T8(B##7)
    T64(0) T64(1) T64(2) T64(3)
#undef T8
#undef T64
#undef TRAMPOLINE_REF
};

static void jit_dispatch(int slot, void *a0, void *a1, void *a2,
                          void *a3, void *a4, void *a5) {
    trampoline_slot_t *s = &g_slots[slot];
    if (!s->real_fn) return;

    uint64_t t0 = now_ns();

    /* Call the real kernel function with its original arguments. */
    typedef void (*kern_t)(void*, void*, void*, void*, void*, void*);
    ((kern_t)s->real_fn)(a0, a1, a2, a3, a4, a5);

    uint64_t dur = now_ns() - t0;
    emit_span(HP_JIT_KERNEL_CATEGORY, gettid_compat(), t0, dur, s->name ? s->name : "<jit-kernel>",
              s->extra[0] ? s->extra : "type=jit_kernel,side=cpu");
}

/* Span name for an ACPP SSCP kernel, as a heap string. ACPP SSCP encodes the
   user kernel as a local-in-function lambda:
     _Z18__acpp_sscp_kernel<...ZZ<len><user_func_name>...>
   When the first "ZZ" is followed by <len><name>, that name is used.
   Otherwise the whole symbol is: a nested lambda ("ZZZ8runbench...", as in
   mixbench's templated kernels) gives no short name, and a cut-off prefix
   would merge every kernel into one name (the first 64 characters used to
   be kept, which reported mixbench's 100 SYCL kernels as 2). */
static char *extract_kernel_name(const char *mangled) {
    const char *p = strstr(mangled, "ZZ");
    if (p) {
        p += 2;
        int len = 0;
        while (*p >= '0' && *p <= '9' && len < 100000) { len = len * 10 + (*p - '0'); p++; }
        if (len > 0 && strnlen(p, (size_t)len) == (size_t)len)
            return strndup(p, (size_t)len);
    }
    return strdup(mangled);
}

static void hprofiler_opencl_dlsym_init(void);

void *dlsym(void *handle, const char *symbol) {
    /* Can run before this library's constructors: another preloaded hook's
       constructor may call dlopen() (wrapped above), whose RTLD_NEXT lookup
       lands here. Resolve the real dlsym on demand rather than return NULL. */
    if (!g_real_dlsym) hprofiler_opencl_dlsym_init();
    if (!g_real_dlsym) return NULL;

    /* Don't intercept our own RTLD_NEXT lookups — those chain to the real lib. */
    if (handle == RTLD_NEXT || !symbol) {
        return g_real_dlsym(handle, symbol);
    }

    void *real_sym = g_real_dlsym(handle, symbol);

    /* Only intercept lookups on registered JIT handles. */
    uint64_t mod = 0;
    if (!real_sym || !is_jit_handle(handle, &mod)) {
        return real_sym;
    }

    if (g_dbg) {
        fprintf(g_dbg, "[opencl-hook] dlsym jit kernel '%s' real=%p\n", symbol, real_sym);
        fflush(g_dbg);
    }

    /* Allocate a trampoline slot for this kernel. */
    pthread_mutex_lock(&g_slot_mutex);
    if (g_slot_next >= TRAMPOLINE_N) {
        pthread_mutex_unlock(&g_slot_mutex);
        /* Pool exhausted — return unwrapped function so execution still works. */
        return real_sym;
    }
    int idx = g_slot_next++;
    pthread_mutex_unlock(&g_slot_mutex);

    trampoline_slot_t *s = &g_slots[idx];
    s->name = extract_kernel_name(symbol);
    char modtag[24];
    snprintf(s->extra, sizeof(s->extra), "type=jit_kernel,side=cpu%s",
             hp_mod_tag(modtag, sizeof(modtag), mod));
    s->real_fn = (void (*)(void))real_sym;   /* last: jit_dispatch checks it */
    s->used = 1;

    return (void *)g_trampolines[idx];
}

/* ── Constructor / Destructor ────────────────────────────────────────── */

/* Priority 50: runs first to initialise g_real_dlsym before any dlsym
   call (including the ones our own wrappers make via RTLD_NEXT).        */
__attribute__((constructor(50)))
static void hprofiler_opencl_dlsym_init(void) {
    /* dlvsym has a different name so it is not intercepted by our dlsym
       override above.  Use it to safely obtain the real dlsym pointer.  */
    /* Try versioned dlsym — covers glibc 2.2.5 through 2.34+. */
    static const char *versions[] = {
        "GLIBC_2.2.5", "GLIBC_2.17", "GLIBC_2.34", NULL
    };
    for (int i = 0; versions[i] && !g_real_dlsym; i++) {
        g_real_dlsym = (void*(*)(void*, const char*))
            dlvsym(RTLD_NEXT, "dlsym", versions[i]);
    }
}

__attribute__((constructor(100)))
static void hprofiler_opencl_init(void) {
    if (getenv("HPROFILER_DEBUG")) {
        char dbgpath[128];
        snprintf(dbgpath, sizeof(dbgpath), "/tmp/hprofiler_ocl_%d.log", (int)getpid());
        g_dbg = fopen(dbgpath, "w");
        if (g_dbg) { fprintf(g_dbg, "[opencl-hook] init pid=%d\n", (int)getpid()); fflush(g_dbg); }
    }
    hp_tx_init("opencl");
    cs_init();
}

__attribute__((destructor))
static void hprofiler_opencl_fini(void) {
    hp_tx_shutdown(1);
    if (g_dbg) { fprintf(g_dbg, "[opencl-hook] fini pid=%d\n", (int)getpid()); fclose(g_dbg); g_dbg = NULL; }
}
