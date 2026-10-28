/* OpenCL queue / event / wait-list ids under the OpenCL hook
 * (tests/integration/test_opencl_deps.py): an in-order queue q1 and an
 * out-of-order queue q2 (in-order when the device has none), with
 *   k1 on q1                       -> e1
 *   k2 on q2 waiting for e1        -> e2
 *   non-blocking read on q2, e2    -> e3
 *   blocking write on q1
 *   marker on q2                   -> e4
 *   barrier on q1 waiting for e3
 *   clWaitForEvents(e3, e4)
 *   k1 on q1 waiting for a user event (the hook cannot name it)
 *   clFinish(q1), clFinish(q2)
 * Prints "ooo=<0|1>" (whether q2 really is out of order). */
#include <stdio.h>
#include <stdlib.h>
#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>
#define N 4096
static const char *src =
    "__kernel void k1(__global float*a){int i=get_global_id(0);a[i]+=1.0f;}\n"
    "__kernel void k2(__global float*a){int i=get_global_id(0);a[i]*=2.0f;}\n";
int main(void) {
    cl_platform_id plat;
    if (clGetPlatformIDs(1, &plat, NULL) != CL_SUCCESS) { printf("no platform\n"); return 77; }
    cl_device_id dev;
    if (clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, 1, &dev, NULL) != CL_SUCCESS) { printf("no device\n"); return 77; }
    cl_int err;
    cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
    cl_command_queue q1 = clCreateCommandQueueWithProperties(ctx, dev, NULL, &err);
    cl_queue_properties ooo_props[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, 0};
    int ooo = 1;
    cl_command_queue q2 = clCreateCommandQueueWithProperties(ctx, dev, ooo_props, &err);
    if (!q2 || err != CL_SUCCESS) {
        ooo = 0;
        q2 = clCreateCommandQueueWithProperties(ctx, dev, NULL, &err);
    }
    cl_program prog = clCreateProgramWithSource(ctx, 1, &src, NULL, &err);
    if (clBuildProgram(prog, 1, &dev, NULL, NULL, NULL) != CL_SUCCESS) { printf("build failed\n"); return 1; }
    cl_kernel k1 = clCreateKernel(prog, "k1", &err);
    cl_kernel k2 = clCreateKernel(prog, "k2", &err);
    cl_mem a = clCreateBuffer(ctx, CL_MEM_READ_WRITE, N * sizeof(float), NULL, &err);
    float *host = calloc(N, sizeof(float));
    clSetKernelArg(k1, 0, sizeof(cl_mem), &a);
    clSetKernelArg(k2, 0, sizeof(cl_mem), &a);
    size_t global = N;
    cl_event e1, e2, e3, e4;
    clEnqueueNDRangeKernel(q1, k1, 1, NULL, &global, NULL, 0, NULL, &e1);
    clEnqueueNDRangeKernel(q2, k2, 1, NULL, &global, NULL, 1, &e1, &e2);
    clEnqueueReadBuffer(q2, a, CL_FALSE, 0, N * sizeof(float), host, 1, &e2, &e3);
    clEnqueueWriteBuffer(q1, a, CL_TRUE, 0, N * sizeof(float), host, 0, NULL, NULL);
    clEnqueueMarkerWithWaitList(q2, 0, NULL, &e4);
    clEnqueueBarrierWithWaitList(q1, 1, &e3, NULL);
    cl_event both[2] = {e3, e4};
    clWaitForEvents(2, both);
    cl_event u = clCreateUserEvent(ctx, &err);
    clEnqueueNDRangeKernel(q1, k1, 1, NULL, &global, NULL, 1, &u, NULL);
    clSetUserEventStatus(u, CL_COMPLETE);
    clFinish(q1);
    clFinish(q2);
    printf("ooo=%d\n", ooo);
    return 0;
}
