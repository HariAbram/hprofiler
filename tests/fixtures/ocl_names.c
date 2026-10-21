/* Kernel names under the OpenCL hook (tests/integration/test_opencl_names.py):
 * a short and a >256-character kernel launched alternately. A fixed-size
 * name buffer made the long kernel's clGetKernelInfo fail and report the
 * previous (short) kernel's name. */
#include <stdio.h>
#include <stdlib.h>
#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>
#define N 1024
#define LONG_NAME "long_kernel_abcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghij"
static const char *src =
    "__kernel void short_k(__global float*a){int i=get_global_id(0);a[i]+=1.0f;}\n"
    "__kernel void " LONG_NAME "(__global float*a){int i=get_global_id(0);a[i]*=2.0f;}\n";
int main(void) {
    cl_platform_id plat;
    if (clGetPlatformIDs(1, &plat, NULL) != CL_SUCCESS) { printf("no platform\n"); return 77; }
    cl_device_id dev;
    if (clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, 1, &dev, NULL) != CL_SUCCESS) { printf("no device\n"); return 77; }
    cl_int err;
    cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
    cl_command_queue q = clCreateCommandQueueWithProperties(ctx, dev, NULL, &err);
    cl_program prog = clCreateProgramWithSource(ctx, 1, &src, NULL, &err);
    if (clBuildProgram(prog, 1, &dev, NULL, NULL, NULL) != CL_SUCCESS) { printf("build failed\n"); return 1; }
    cl_kernel ks = clCreateKernel(prog, "short_k", &err);
    cl_kernel kl = clCreateKernel(prog, LONG_NAME, &err);
    if (err != CL_SUCCESS) { printf("long kernel not created: %d\n", err); return 1; }
    cl_mem a = clCreateBuffer(ctx, CL_MEM_READ_WRITE, N * sizeof(float), NULL, &err);
    clSetKernelArg(ks, 0, sizeof(cl_mem), &a);
    clSetKernelArg(kl, 0, sizeof(cl_mem), &a);
    size_t global = N;
    for (int r = 0; r < 3; r++) {
        clEnqueueNDRangeKernel(q, ks, 1, NULL, &global, NULL, 0, NULL, NULL);
        if (r < 2) clEnqueueNDRangeKernel(q, kl, 1, NULL, &global, NULL, 0, NULL, NULL);
    }
    clFinish(q);
    printf("ocl_names: done\n");
    return 0;
}
