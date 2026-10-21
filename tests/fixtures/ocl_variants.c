/* Kernel variants under the OpenCL hook (tests/integration/
 * test_opencl_modules.py): two programs both define `variant_k` with
 * different code (program A launched twice, program B three times), and a
 * third program holds two kernels. Disassembly keyed by kernel name kept one
 * `variant_k` (the last program built) for all five launches, and only the
 * first kernel of a multi-kernel program. */
#include <stdio.h>
#include <stdlib.h>
#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>
#define N 1024
static const char *src_a =
    "__kernel void variant_k(__global float*a){int i=get_global_id(0);a[i]+=1.0f;}\n";
static const char *src_b =
    "__kernel void variant_k(__global float*a){int i=get_global_id(0);"
    "float x=a[i];for(int k=0;k<16;k++)x=x*x*0.5f+(float)k;a[i]=x;}\n";
static const char *src_c =
    "__kernel void pair_one(__global float*a){int i=get_global_id(0);a[i]-=2.0f;}\n"
    "__kernel void pair_two(__global float*a){int i=get_global_id(0);a[i]=a[i]*3.0f+a[i];}\n";

static cl_program build(cl_context ctx, cl_device_id dev, const char *src) {
    cl_int err;
    cl_program p = clCreateProgramWithSource(ctx, 1, &src, NULL, &err);
    if (err != CL_SUCCESS || clBuildProgram(p, 1, &dev, NULL, NULL, NULL) != CL_SUCCESS) {
        printf("build failed\n");
        exit(1);
    }
    return p;
}

static void run(cl_command_queue q, cl_program p, const char *name, cl_mem a, int times) {
    cl_int err;
    cl_kernel k = clCreateKernel(p, name, &err);
    if (err != CL_SUCCESS) { printf("kernel %s not created: %d\n", name, err); exit(1); }
    clSetKernelArg(k, 0, sizeof(cl_mem), &a);
    size_t global = N;
    for (int r = 0; r < times; r++)
        clEnqueueNDRangeKernel(q, k, 1, NULL, &global, NULL, 0, NULL, NULL);
    clFinish(q);
    clReleaseKernel(k);
}

int main(void) {
    cl_platform_id plat;
    if (clGetPlatformIDs(1, &plat, NULL) != CL_SUCCESS) { printf("no platform\n"); return 77; }
    cl_device_id dev;
    if (clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, 1, &dev, NULL) != CL_SUCCESS) { printf("no device\n"); return 77; }
    cl_int err;
    cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
    cl_command_queue q = clCreateCommandQueueWithProperties(ctx, dev, NULL, &err);
    cl_mem a = clCreateBuffer(ctx, CL_MEM_READ_WRITE, N * sizeof(float), NULL, &err);
    cl_program pa = build(ctx, dev, src_a);
    cl_program pb = build(ctx, dev, src_b);
    cl_program pc = build(ctx, dev, src_c);
    run(q, pa, "variant_k", a, 2);
    run(q, pb, "variant_k", a, 3);
    run(q, pc, "pair_one", a, 1);
    run(q, pc, "pair_two", a, 1);
    printf("ocl_variants: done\n");
    return 0;
}
