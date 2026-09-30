/* Ground-truth OpenCL program for tests/integration/test_profiling_accuracy.py.
 * Uses the Intel CPU OpenCL device. Blocking write, kernel + clFinish (kernel
 * device time taken from the program's own CL_PROFILING query), non-blocking
 * read + clFinish. Logs CLOCK_MONOTONIC stamps to $TRUTH_OUT. */
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
static inline uint64_t now_ns(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return (uint64_t)t.tv_sec*1000000000ull+t.tv_nsec;}
static const char *src="__kernel void k(__global float*a,int iters){int i=get_global_id(0);float x=a[i];for(int j=0;j<iters;j++)x=x*1.000001f+0.5f;a[i]=x;}";
#define N (4*1024*1024)
int main(void){
  const char *out=getenv("TRUTH_OUT"); FILE *f=out?fopen(out,"w"):stdout;
  cl_uint np; clGetPlatformIDs(0,NULL,&np); cl_platform_id ps[8]; clGetPlatformIDs(np,ps,NULL);
  cl_platform_id p=NULL; char name[256];
  for(cl_uint i=0;i<np;i++){clGetPlatformInfo(ps[i],CL_PLATFORM_NAME,256,name,NULL); if(strstr(name,"Intel")) p=ps[i];}
  cl_device_id d; clGetDeviceIDs(p,CL_DEVICE_TYPE_CPU,1,&d,NULL);
  cl_int e; cl_context c=clCreateContext(NULL,1,&d,NULL,NULL,&e);
  cl_command_queue q=clCreateCommandQueue(c,d,CL_QUEUE_PROFILING_ENABLE,&e);
  cl_program pr=clCreateProgramWithSource(c,1,&src,NULL,&e); clBuildProgram(pr,1,&d,NULL,NULL,NULL);
  cl_kernel k=clCreateKernel(pr,"k",&e);
  float *h=malloc(N*sizeof(float)); for(int i=0;i<N;i++)h[i]=1;
  cl_mem b=clCreateBuffer(c,CL_MEM_READ_WRITE,N*sizeof(float),NULL,&e);
  for(int it=0;it<5;it++){
    uint64_t a=now_ns(); clEnqueueWriteBuffer(q,b,CL_TRUE,0,N*sizeof(float),h,0,NULL,NULL); uint64_t z=now_ns();
    fprintf(f,"write_blocking %d %llu %llu\n",it,(unsigned long long)a,(unsigned long long)z);
    int iters=200*(it+1); clSetKernelArg(k,0,sizeof(b),&b); clSetKernelArg(k,1,sizeof(int),&iters);
    size_t g=N; cl_event ke;
    clEnqueueNDRangeKernel(q,k,1,NULL,&g,NULL,0,NULL,&ke);
    a=now_ns(); clFinish(q); z=now_ns();
    cl_ulong ks,kend; clGetEventProfilingInfo(ke,CL_PROFILING_COMMAND_START,8,&ks,NULL); clGetEventProfilingInfo(ke,CL_PROFILING_COMMAND_END,8,&kend,NULL);
    fprintf(f,"kernel_devdur %d %llu\n",it,(unsigned long long)(kend-ks));
    fprintf(f,"finish %d %llu %llu\n",it,(unsigned long long)a,(unsigned long long)z);
    clReleaseEvent(ke);
    a=now_ns(); clEnqueueReadBuffer(q,b,CL_FALSE,0,N*sizeof(float),h,0,NULL,NULL); clFinish(q); z=now_ns();
    fprintf(f,"read_nonblocking_plus_finish %d %llu %llu\n",it,(unsigned long long)a,(unsigned long long)z);
  }
  if(f!=stdout)fclose(f);
  return 0;
}
