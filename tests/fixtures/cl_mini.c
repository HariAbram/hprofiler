#include <stdio.h>
#include <stdlib.h>
#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>
#define N (1<<18)
static const char *src = "__kernel void add(__global const float*a,__global const float*b,__global float*c){int i=get_global_id(0);c[i]=a[i]+b[i];}";
int main(void) {
    cl_platform_id plat; clGetPlatformIDs(1,&plat,NULL);
    cl_device_id dev; clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, 1, &dev, NULL);
    cl_int err;
    cl_context ctx = clCreateContext(NULL,1,&dev,NULL,NULL,&err);
    cl_command_queue q = clCreateCommandQueueWithProperties(ctx, dev, NULL, &err);
    cl_program prog = clCreateProgramWithSource(ctx,1,&src,NULL,&err);
    clBuildProgram(prog,1,&dev,NULL,NULL,NULL);
    cl_kernel k = clCreateKernel(prog,"add",&err);
    float *ha=malloc(N*sizeof(float)),*hb=malloc(N*sizeof(float)),*hc=malloc(N*sizeof(float));
    for(int i=0;i<N;i++){ha[i]=i;hb[i]=N-i;}
    cl_mem da=clCreateBuffer(ctx,CL_MEM_READ_ONLY,N*sizeof(float),NULL,&err);
    cl_mem db=clCreateBuffer(ctx,CL_MEM_READ_ONLY,N*sizeof(float),NULL,&err);
    cl_mem dc=clCreateBuffer(ctx,CL_MEM_WRITE_ONLY,N*sizeof(float),NULL,&err);
    clEnqueueWriteBuffer(q,da,CL_TRUE,0,N*sizeof(float),ha,0,NULL,NULL);
    clEnqueueWriteBuffer(q,db,CL_TRUE,0,N*sizeof(float),hb,0,NULL,NULL);
    clSetKernelArg(k,0,sizeof(cl_mem),&da); clSetKernelArg(k,1,sizeof(cl_mem),&db); clSetKernelArg(k,2,sizeof(cl_mem),&dc);
    size_t global=N;
    for(int r=0;r<8;r++) clEnqueueNDRangeKernel(q,k,1,NULL,&global,NULL,0,NULL,NULL);
    clEnqueueReadBuffer(q,dc,CL_TRUE,0,N*sizeof(float),hc,0,NULL,NULL);
    clFinish(q);
    printf("cl_mini: c0=%f expect=%f\n", hc[0], ha[0]+hb[0]);
    return 0;
}
