/*
 * OpenCL event capture for tests/integration/test_opencl_events.py, on every
 * platform's first device (each its own context and in-order queue):
 *   long    40 fills, then one kernel waiting for all 40 (a long wait list)
 *   user    a kernel gated on a user event that the host completes 20 ms
 *           later (clSetUserEventStatus)
 *   reuse   an event released and a new one created until the address
 *           repeats; a marker waits for the new one
 *   more    copy, fill, rectangle read/write, map/unmap, migrate
 *   usm     where the platform has cl_intel_unified_shared_memory: a fill
 *           and a copy through the extension functions (looked up with
 *           clGetExtensionFunctionAddressForPlatform), chained by wait
 *           lists, then a marker waiting for the copy
 *   press   (argv[1] = "press") 300 markers whose events are all kept, then
 *           one kernel waiting for the last 10 -- run with a tiny event table
 * Prints "devices=N" and per device "dev=<i> reused=<0|1> usm=<0|1>".
 */
#define CL_TARGET_OPENCL_VERSION 220
#include <CL/cl.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

static const char *SRC = "__kernel void inc(__global int *p) { p[0] += 1; }";

typedef void *(*host_alloc_t)(cl_context, const void *, size_t, cl_uint, cl_int *);
typedef cl_int (*usm_free_t)(cl_context, void *);
typedef cl_int (*usm_fill_t)(cl_command_queue, void *, const void *, size_t, size_t, cl_uint, const cl_event *,
                             cl_event *);
typedef cl_int (*usm_copy_t)(cl_command_queue, cl_bool, void *, const void *, size_t, cl_uint, const cl_event *,
                             cl_event *);

/* the usm case; 1 when the platform had the extension and it ran */
static int usm(cl_platform_id plat, cl_context ctx, cl_command_queue q) {
    char ext[8192] = "";
    clGetPlatformInfo(plat, CL_PLATFORM_EXTENSIONS, sizeof(ext) - 1, ext, NULL);
    if (!strstr(ext, "cl_intel_unified_shared_memory")) return 0;
    host_alloc_t alloc = (host_alloc_t)clGetExtensionFunctionAddressForPlatform(plat, "clHostMemAllocINTEL");
    usm_free_t release = (usm_free_t)clGetExtensionFunctionAddressForPlatform(plat, "clMemFreeINTEL");
    usm_fill_t fill = (usm_fill_t)clGetExtensionFunctionAddressForPlatform(plat, "clEnqueueMemFillINTEL");
    usm_copy_t copy = (usm_copy_t)clGetExtensionFunctionAddressForPlatform(plat, "clEnqueueMemcpyINTEL");
    if (!alloc || !release || !fill || !copy) return 0;
    cl_int err;
    int *x = alloc(ctx, NULL, 256, 0, &err), *y = alloc(ctx, NULL, 256, 0, &err);
    if (!x || !y) return 0;
    int seven = 7;
    cl_event f, c, m;
    fill(q, x, &seven, sizeof(int), 256, 0, NULL, &f);
    copy(q, CL_FALSE, y, x, 256, 1, &f, &c);
    clEnqueueMarkerWithWaitList(q, 1, &c, &m);
    clFinish(q);
    int ok = y[63] == 7;
    clReleaseEvent(f);
    clReleaseEvent(c);
    clReleaseEvent(m);
    release(ctx, x);
    release(ctx, y);
    return ok;
}

int main(int argc, char **argv) {
    int press = argc > 1 && !strcmp(argv[1], "press");
    cl_platform_id plats[4];
    cl_uint np = 0;
    if (clGetPlatformIDs(4, plats, &np) != CL_SUCCESS || !np) { printf("no OpenCL platform\n"); return 2; }
    int ndev = 0;
    for (cl_uint pi = 0; pi < np; pi++) {
        cl_device_id dev;
        if (clGetDeviceIDs(plats[pi], CL_DEVICE_TYPE_ALL, 1, &dev, NULL) != CL_SUCCESS) continue;
        cl_int err;
        cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
        if (!ctx) continue;
        cl_command_queue q = clCreateCommandQueueWithProperties(ctx, dev, NULL, &err);
        cl_program prog = clCreateProgramWithSource(ctx, 1, &SRC, NULL, &err);
        if (!q || clBuildProgram(prog, 1, &dev, NULL, NULL, NULL) != CL_SUCCESS) continue;
        cl_kernel k = clCreateKernel(prog, "inc", &err);
        cl_mem a = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 4096, NULL, &err);
        cl_mem b = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 4096, NULL, &err);
        clSetKernelArg(k, 0, sizeof(cl_mem), &a);
        size_t one = 1;
        int zero = 0;

        if (press) {
            cl_event evs[300];
            for (int i = 0; i < 300; i++) clEnqueueMarkerWithWaitList(q, 0, NULL, &evs[i]);
            cl_event done;
            clEnqueueNDRangeKernel(q, k, 1, NULL, &one, NULL, 10, &evs[290], &done);
            clFinish(q);
            for (int i = 0; i < 300; i++) clReleaseEvent(evs[i]);
            clReleaseEvent(done);
            ndev++;
            continue;
        }

        /* long: 40 fills, a kernel waiting for all */
        cl_event fills[40], longk;
        for (int i = 0; i < 40; i++)
            clEnqueueFillBuffer(q, b, &zero, sizeof(int), (size_t)i * 4, 4, 0, NULL, &fills[i]);
        clEnqueueNDRangeKernel(q, k, 1, NULL, &one, NULL, 40, fills, &longk);
        clFinish(q);

        /* user: a kernel gated on a user event */
        cl_event user = clCreateUserEvent(ctx, &err), gated;
        clEnqueueNDRangeKernel(q, k, 1, NULL, &one, NULL, 1, &user, &gated);
        usleep(20000);
        clSetUserEventStatus(user, CL_COMPLETE);
        clWaitForEvents(1, &gated);

        /* reuse: an event's address handed to a later event */
        cl_event e1;
        clEnqueueMarkerWithWaitList(q, 0, NULL, &e1);
        clFinish(q);
        void *first = (void *)e1;
        clReleaseEvent(e1);
        int reused = 0;
        cl_event e2 = NULL, spare[64];
        int nspare = 0;
        for (int t = 0; t < 64; t++) {
            clEnqueueMarkerWithWaitList(q, 0, NULL, &e2);
            if ((void *)e2 == first) { reused = 1; break; }
            spare[nspare++] = e2;
        }
        cl_event after;
        clEnqueueBarrierWithWaitList(q, 1, &e2, &after);
        clFinish(q);
        for (int i = 0; i < nspare; i++) clReleaseEvent(spare[i]);

        /* more: copies, rectangles, maps, migration */
        cl_event cp, rect, mp, um, mg;
        clEnqueueCopyBuffer(q, a, b, 0, 0, 64, 0, NULL, &cp);
        size_t origin[3] = {0, 0, 0}, region[3] = {16, 2, 1};
        char host[64];
        clEnqueueReadBufferRect(q, b, CL_FALSE, origin, origin, region, 16, 0, 16, 0, host, 1, &cp, &rect);
        void *m = clEnqueueMapBuffer(q, a, CL_FALSE, CL_MAP_READ, 0, 64, 1, &rect, &mp, &err);
        clEnqueueUnmapMemObject(q, a, m, 1, &mp, &um);
        clEnqueueMigrateMemObjects(q, 1, &b, CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, 1, &um, &mg);
        clFinish(q);

        int u = usm(plats[pi], ctx, q);
        printf("dev=%d reused=%d usm=%d\n", ndev, reused, u);
        ndev++;
    }
    printf("devices=%d\n", ndev);
    return ndev ? 0 : 2;
}
