/*
 * OpenCL LD_PRELOAD hook.
 *
 * Wraps OpenCL API calls to collect GPU/CPU kernel execution times using
 * the built-in OpenCL profiling infrastructure (CL_QUEUE_PROFILING_ENABLE).
 *
 * Strategy:
 *   1. clCreateCommandQueue / clCreateCommandQueueWithProperties:
 *      force CL_QUEUE_PROFILING_ENABLE so every enqueued command is timed.
 *   2. clEnqueueNDRangeKernel / clEnqueueTask:
 *      capture the cl_event, then after clFinish/clWaitForEvents read
 *      CL_PROFILING_COMMAND_START/END to get GPU-side wall-clock.
 *   3. clEnqueueReadBuffer / clEnqueueWriteBuffer / clEnqueueCopyBuffer:
 *      track memory transfers.
 *   4. clBuildProgram: track JIT compilation time (important for ACPP).
 *
 * Records are emitted as newline-delimited ASCII over HPROFILER_SOCKET.
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

static FILE *g_dbg = NULL;  /* set when HPROFILER_DEBUG=1 */

static void ensure_connected(void) {
    if (g_sock >= 0) return;
    const char *path = getenv("HPROFILER_SOCKET");
    if (g_dbg) fprintf(g_dbg, "[opencl-hook] ensure_connected pid=%d path=%s\n",
                       (int)getpid(), path ? path : "(null)");
    if (!path) return;
    int s = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (s < 0) return;
    struct sockaddr_un addr = {0};
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, path, sizeof(addr.sun_path) - 1);
    if (connect(s, (struct sockaddr*)&addr, sizeof(addr)) == 0) {
        g_sock = s;
        g_pid  = getpid();
        if (g_dbg) fprintf(g_dbg, "[opencl-hook] connected fd=%d\n", s);
    } else {
        if (g_dbg) fprintf(g_dbg, "[opencl-hook] connect FAILED errno=%d\n", errno);
        close(s);
    }
    if (g_dbg) fflush(g_dbg);
}

static void emit_span(const char *cat, pid_t tid, uint64_t start_ns,
                      uint64_t dur_ns, const char *name, const char *extra) {
    pthread_mutex_lock(&g_sock_mutex);
    ensure_connected();
    if (g_dbg) {
        fprintf(g_dbg, "[opencl-hook] emit_span g_sock=%d cat=%s name=%s\n",
                g_sock, cat, name);
        fflush(g_dbg);
    }
    if (g_sock >= 0) {
        char buf[1024];
        int n;
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
                (unsigned long long)start_ns, (unsigned long long)dur_ns,
                name);
        if (n > 0) {
            ssize_t sent = send(g_sock, buf, n, MSG_NOSIGNAL);
            if (g_dbg) { fprintf(g_dbg, "[opencl-hook] send n=%d sent=%zd\n", n, sent); fflush(g_dbg); }
        }
    }
    pthread_mutex_unlock(&g_sock_mutex);
}

/* CL_CALLBACK expands to nothing on Linux x86-64 (no special calling conv). */
#ifndef CL_CALLBACK
#define CL_CALLBACK
#endif

/* Heap-allocated context passed to the OpenCL event callback. */
typedef struct { char name[128]; char extra[64]; } event_cb_data_t;

typedef void (*event_cb_fn_t)(cl_event, cl_int, void *);
typedef cl_int (*SetCB_t)(cl_event, cl_int, event_cb_fn_t, void *);
typedef cl_int (*RetainEvent_t)(cl_event);
typedef cl_int (*ReleaseEvent_t)(cl_event);
typedef cl_int (*GetEventProf_t)(cl_event, cl_profiling_info, size_t, void *, size_t *);

/* OpenCL event completion callback — fired on a driver thread when the
   kernel/transfer completes.  Reads GPU timestamps and emits the span.   */
static void CL_CALLBACK on_event_complete(cl_event ev,
                                           cl_int status, void *user_data) {
    event_cb_data_t *d = (event_cb_data_t *)user_data;
    if (status != 0 /* CL_COMPLETE */) { free(d); return; }

    static GetEventProf_t real_prof    = NULL;
    static ReleaseEvent_t real_release = NULL;
    if (!real_prof)    real_prof    = (GetEventProf_t)dlsym(RTLD_NEXT, "clGetEventProfilingInfo");
    if (!real_release) real_release = (ReleaseEvent_t)dlsym(RTLD_NEXT, "clReleaseEvent");

    cl_ulong gpu_start = 0, gpu_end = 0;
    if (real_prof &&
        real_prof(ev, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &gpu_start, NULL) == CL_SUCCESS &&
        real_prof(ev, CL_PROFILING_COMMAND_END,   sizeof(cl_ulong), &gpu_end,   NULL) == CL_SUCCESS) {
        emit_span("opencl", gettid_compat(),
                  (uint64_t)gpu_start, (uint64_t)(gpu_end - gpu_start),
                  d->name, d->extra);
    }
    if (real_release) real_release(ev);
    free(d);
}

/* Register an async GPU-profiling callback.  Returns immediately. */
static void register_event_callback(cl_event ev,
                                    const char *name, const char *extra) {
    static SetCB_t      real_setcb  = NULL;
    static RetainEvent_t real_retain = NULL;
    if (!real_setcb)  real_setcb  = (SetCB_t)dlsym(RTLD_NEXT, "clSetEventCallback");
    if (!real_retain) real_retain = (RetainEvent_t)dlsym(RTLD_NEXT, "clRetainEvent");
    if (!real_setcb || !ev) return;

    event_cb_data_t *d = (event_cb_data_t *)malloc(sizeof(*d));
    if (!d) return;
    snprintf(d->name,  sizeof(d->name),  "%s", name  ? name  : "");
    snprintf(d->extra, sizeof(d->extra), "%s", extra ? extra : "");

    if (real_retain) real_retain(ev);
    if (real_setcb(ev, 0 /* CL_COMPLETE */, on_event_complete, d) != CL_SUCCESS) {
        static ReleaseEvent_t real_release = NULL;
        if (!real_release) real_release = (ReleaseEvent_t)dlsym(RTLD_NEXT, "clReleaseEvent");
        if (real_retain && real_release) real_release(ev);
        free(d);
    }
}

/* Resolve kernel name. */
static const char *get_kernel_name(cl_kernel kernel) {
    typedef cl_int (*GetKernelInfo_t)(cl_kernel, cl_uint, size_t, void*, size_t*);
    static GetKernelInfo_t real = NULL;
    static __thread char kname[256];
    if (!real) real = (GetKernelInfo_t)dlsym(RTLD_NEXT, "clGetKernelInfo");
    if (!real || !kernel) return "<unknown>";
#define CL_KERNEL_FUNCTION_NAME 0x1190
    real(kernel, CL_KERNEL_FUNCTION_NAME, sizeof(kname), kname, NULL);
    return kname;
}

/* ── clCreateCommandQueue ────────────────────────────────────────────── */

cl_command_queue clCreateCommandQueue(
    cl_context ctx, cl_device_id dev,
    cl_command_queue_properties props, cl_int *err)
{
    typedef cl_command_queue (*fn_t)(cl_context, cl_device_id,
                                     cl_command_queue_properties, cl_int*);
    static fn_t real = NULL;
    if (!real) real = (fn_t)dlsym(RTLD_NEXT, "clCreateCommandQueue");
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
    if (!real) real = (fn_t)dlsym(RTLD_NEXT, "clCreateCommandQueueWithProperties");

    /* Build a modified props array with PROFILING_ENABLE inserted. */
    cl_command_queue_properties new_props[32] = {0};
    int n = 0;
    int already_profiling = 0;
    if (props) {
        for (int i = 0; props[i]; i += 2) {
            new_props[n++] = props[i];
            if ((uint64_t)props[i] == 0x1093 /* CL_QUEUE_PROPERTIES */) {
                new_props[n++] = props[i+1] | CL_QUEUE_PROFILING_ENABLE;
                already_profiling = 1;
            } else {
                new_props[n++] = props[i+1];
            }
            if (n >= 28) break;
        }
    }
    if (!already_profiling) {
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
    if (!real) real = (fn_t)dlsym(RTLD_NEXT, "clEnqueueNDRangeKernel");

    const char *kname = get_kernel_name(kernel);
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
        /* CPU-side span: time from enqueue call to return (scheduling latency). */
        emit_span("opencl", gettid_compat(), cpu_t0, cpu_dur, kname,
                  "type=kernel,side=cpu");
        /* GPU-side span: registered as async callback — does NOT block the queue. */
        register_event_callback(*event, kname, "type=kernel,side=gpu");
        /* Release our internal event reference if the caller didn't want it. */
        if (our_event) {
            typedef cl_int (*ReleaseEvent_t)(cl_event);
            static ReleaseEvent_t real_release = NULL;
            if (!real_release) real_release = (ReleaseEvent_t)dlsym(RTLD_NEXT, "clReleaseEvent");
            if (real_release) real_release(*event);
        }
    }
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
    if (!real) real = (fn_t)dlsym(RTLD_NEXT, "clEnqueueSVMMemcpy");
    char extra[64]; snprintf(extra, sizeof(extra), "type=svm_memcpy,bytes=%zu", size);
    cl_event iev = NULL;
    if (!ev) ev = &iev;
    uint64_t t0 = now_ns();
    cl_int ret = real(q, blocking, dst, src, size, nwl, ewl, ev);
    emit_span("memory", gettid_compat(), t0, now_ns()-t0, "clEnqueueSVMMemcpy", extra);
    if (ret == CL_SUCCESS && *ev)
        register_event_callback(*ev, "clEnqueueSVMMemcpy", extra);
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
    if (!real) real = (fn_t)dlsym(RTLD_NEXT, "clEnqueueReadBuffer");

    cl_event iev = NULL;
    if (!ev) ev = &iev;
    char extra[64]; snprintf(extra, sizeof(extra), "type=read,bytes=%zu", size);
    uint64_t t0 = now_ns();
    cl_int ret = real(q, buf, blocking, offset, size, ptr, nwl, ewl, ev);
    emit_span("memory", gettid_compat(), t0, now_ns()-t0, "clEnqueueReadBuffer", extra);
    if (ret == CL_SUCCESS && *ev)
        register_event_callback(*ev, "clReadBuffer_gpu", extra);
    return ret;
}

cl_int clEnqueueWriteBuffer(
    cl_command_queue q, cl_mem buf, cl_bool blocking, size_t offset,
    size_t size, const void *ptr, cl_uint nwl, const cl_event *ewl, cl_event *ev)
{
    typedef cl_int (*fn_t)(cl_command_queue, cl_mem, cl_bool, size_t, size_t,
                            const void*, cl_uint, const cl_event*, cl_event*);
    static fn_t real = NULL;
    if (!real) real = (fn_t)dlsym(RTLD_NEXT, "clEnqueueWriteBuffer");

    cl_event iev = NULL;
    if (!ev) ev = &iev;
    char extra[64]; snprintf(extra, sizeof(extra), "type=write,bytes=%zu", size);
    uint64_t t0 = now_ns();
    cl_int ret = real(q, buf, blocking, offset, size, ptr, nwl, ewl, ev);
    emit_span("memory", gettid_compat(), t0, now_ns()-t0, "clEnqueueWriteBuffer", extra);
    if (ret == CL_SUCCESS && *ev)
        register_event_callback(*ev, "clWriteBuffer_gpu", extra);
    return ret;
}

/* ── clBuildProgram: tracks JIT compilation time (ACPP SSCP, etc.) ─── */

cl_int clBuildProgram(
    cl_program program, cl_uint num_devices, const cl_device_id *device_list,
    const char *options, void (*pfn_notify)(cl_program, void*), void *user_data)
{
    typedef cl_int (*fn_t)(cl_program, cl_uint, const cl_device_id*,
                            const char*, void(*)(cl_program,void*), void*);
    static fn_t real = NULL;
    if (!real) real = (fn_t)dlsym(RTLD_NEXT, "clBuildProgram");

    uint64_t t0 = now_ns();
    cl_int ret = real(program, num_devices, device_list, options, pfn_notify, user_data);
    emit_span("jit", gettid_compat(), t0, now_ns()-t0,
              "clBuildProgram", ret == CL_SUCCESS ? "status=ok" : "status=err");
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
    if (!real) real = (fn_t)dlsym(RTLD_NEXT, "clCompileProgram");

    uint64_t t0 = now_ns();
    cl_int ret = real(program, num_devices, device_list, options,
                      num_input_headers, input_headers, header_include_names,
                      pfn_notify, user_data);
    emit_span("jit", gettid_compat(), t0, now_ns()-t0,
              "clCompileProgram", "type=jit_compile");
    return ret;
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
static int       g_jit_count = 0;
static pthread_mutex_t g_jit_mutex = PTHREAD_MUTEX_INITIALIZER;

static int is_jit_handle(void *h) {
    pthread_mutex_lock(&g_jit_mutex);
    for (int i = 0; i < g_jit_count; i++) {
        if (g_jit_handles[i] == h) {
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
    flags &= ~RTLD_DEEPBIND;

    /* For ACPP SSCP .jit.so files: copy to /tmp BEFORE dlopen so the file
       still exists for the disassembler even if ACPP unlinks it after load. */
    static int jit_copy_n = 0;
    char saved_jit_path[512] = {0};
    if (filename && strstr(filename, ".jit.so")) {
        FILE *src = fopen(filename, "rb");
        if (src) {
            snprintf(saved_jit_path, sizeof(saved_jit_path),
                     "/tmp/hprofiler_jit_%d_%d.so", (int)getpid(), jit_copy_n++);
            FILE *dst = fopen(saved_jit_path, "wb");
            if (dst) {
                char buf[65536];
                size_t n;
                while ((n = fread(buf, 1, sizeof(buf), src)) > 0)
                    fwrite(buf, 1, n, dst);
                fclose(dst);
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
        char jit_extra[640];
        snprintf(jit_extra, sizeof(jit_extra), "type=jit_load,path=%s", disasm_path);
        emit_span("jit", gettid_compat(), t0, dur, base, jit_extra);

        /* Register this handle so dlsym calls on it are intercepted. */
        pthread_mutex_lock(&g_jit_mutex);
        if (g_jit_count < MAX_JIT_HANDLES) {
            g_jit_handles[g_jit_count] = h;
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
   When ACPP calls dlsym(jit_handle, "kernel_name") to find a compiled
   CPU kernel, we intercept it:
     • emit a span recording the kernel name (useful even if the function
       pointer is cached — it tells us which kernels exist and their names)
     • return a trampoline that times each actual kernel invocation

   Trampoline pool: each trampoline is a small mmap'd executable thunk
   that saves/restores registers, records wall-clock time, calls the real
   kernel function, then emits the span.  We pre-allocate TRAMPOLINE_N
   slots; each slot stores {real_fn, name} and has its own thunk.         */

static void *(*g_real_dlsym)(void *, const char *) = NULL;

/* ── trampoline pool ────────────────────────────────────────────────────── */
#define TRAMPOLINE_N  256

typedef struct {
    void    (*real_fn)(void);   /* real kernel entry point (cast to fn ptr) */
    char     name[128];         /* kernel name for span emission */
    int      used;
} trampoline_slot_t;

static trampoline_slot_t  g_slots[TRAMPOLINE_N];
static int                g_slot_next = 0;
static pthread_mutex_t    g_slot_mutex = PTHREAD_MUTEX_INITIALIZER;
static uint8_t           *g_thunk_page = NULL; /* one executable page */

/*
 * Each thunk is a tiny x86-64 function body that:
 *   1. Calls clock_gettime(CLOCK_MONOTONIC, &ts) to record start time
 *   2. Calls the real kernel function preserving ALL general-purpose
 *      argument registers (rdi, rsi, rdx, rcx, r8, r9) and the stack
 *   3. Calls emit_span with the recorded times
 *
 * Rather than generating raw machine code, we use a C helper that is
 * called FROM the thunk and receives the slot index as an extra hidden
 * first argument (pushed before the real call).
 *
 * Thunk layout (N = slot index, embedded as a 32-bit immediate):
 *   push  rbp
 *   mov   rbp, rsp
 *   and   rsp, -16          ; align stack
 *   ; save argument registers
 *   push  rdi; push rsi; push rdx; push rcx; push r8; push r9
 *   ; call wrapper(slot_index, rdi, rsi, rdx, rcx, r8, r9)
 *   mov   rdi, N            ; slot index
 *   pop   r9; pop r8; pop rcx; pop rdx; pop rsi   ; shift saved regs
 *   ; rdi already set; restore remaining args
 *   ... (this gets complicated - use simpler approach below)
 *
 * Simpler approach: store start_ns in the slot via a pre-call hook,
 * then call the real fn, then a post-call hook reads it.
 * Because the slot is per-trampoline (not per-thread), this is NOT
 * thread-safe for concurrent calls to the same kernel.  For
 * profiling this is acceptable — we emit the span immediately.
 *
 * Assembly thunk (22 bytes each, N slots fit in one 4 KB page):
 *   48 b8 <imm64: slot_ptr>   ; mov rax, &g_slots[N]
 *   ff d0                     ; call rax  ← NO, we need to preserve rdi
 *
 * Cleanest approach: use a single C trampoline with __attribute__((naked))
 * and slot index passed via a thread-local.  But naked functions are not
 * portable.
 *
 * ACTUAL IMPLEMENTATION: use a per-slot wrapper C function selected at
 * registration time.  We generate TRAMPOLINE_N distinct C-callable wrappers
 * using X-macro expansion.  Each wrapper calls jit_dispatch(N, real args).
 */

/* Forward declaration; defined after the macro expansion. */
static void jit_dispatch(int slot, void *arg0, void *arg1, void *arg2,
                          void *arg3, void *arg4, void *arg5);

/* Generate TRAMPOLINE_N distinct wrapper functions.
   Each wrapper passes its fixed slot index as the first argument,
   then forwards the first 6 pointer-sized arguments the kernel received.
   ACPP CPU kernels receive at most a launch-config pointer + arg array,
   so 6 slots covers all realistic cases.                                  */
#define DEF_TRAMPOLINE(N) \
static void trampoline_##N(void *a0, void *a1, void *a2, \
                            void *a3, void *a4, void *a5) { \
    jit_dispatch(N, a0, a1, a2, a3, a4, a5); \
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
    emit_span("opencl", gettid_compat(), t0, dur, s->name, "type=jit_kernel,side=cpu");
}

/* Extract a short human-readable name from ACPP SSCP's mangled kernel names.
   ACPP SSCP encodes the user kernel as a local-in-function lambda:
     _Z18__acpp_sscp_kernel<...ZZ<len><user_func_name>...>
   We find the first "ZZ<digits><name>" occurrence and return "<name>".
   Falls back to the raw (truncated) symbol if the pattern isn't found.    */
static void extract_kernel_name(const char *mangled, char *out, size_t outsz) {
    const char *p = strstr(mangled, "ZZ");
    if (p) {
        p += 2;
        int len = 0;
        while (*p >= '0' && *p <= '9') { len = len * 10 + (*p - '0'); p++; }
        if (len > 0 && len < 128) {
            snprintf(out, outsz, "%.*s", len, p);
            return;
        }
    }
    /* Fallback: first 64 chars of mangled name. */
    snprintf(out, outsz, "%.64s", mangled);
}

void *dlsym(void *handle, const char *symbol) {
    /* Ensure g_real_dlsym is initialised (constructor might not have run yet
       if someone calls dlsym very early). */
    if (!g_real_dlsym) return NULL;

    /* Don't intercept our own RTLD_NEXT lookups — those chain to the real lib. */
    if (handle == RTLD_NEXT || !symbol) {
        return g_real_dlsym(handle, symbol);
    }

    void *real_sym = g_real_dlsym(handle, symbol);

    /* Only intercept lookups on registered JIT handles. */
    if (!real_sym || !is_jit_handle(handle)) {
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
    s->real_fn = (void (*)(void))real_sym;
    extract_kernel_name(symbol, s->name, sizeof(s->name));
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
    ensure_connected();
}

__attribute__((destructor))
static void hprofiler_opencl_fini(void) {
    if (g_dbg) { fprintf(g_dbg, "[opencl-hook] fini pid=%d g_sock=%d\n", (int)getpid(), g_sock); fclose(g_dbg); g_dbg = NULL; }
    if (g_sock >= 0) { close(g_sock); g_sock = -1; }
}
