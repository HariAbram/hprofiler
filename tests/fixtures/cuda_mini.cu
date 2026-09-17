#include <cstdio>
#include <cuda_runtime.h>
__global__ void k(float*a){a[0]=1.0f;}
int main(){
    float *d=nullptr;
    cudaError_t e = cudaMalloc((void**)&d, sizeof(float));
    if (d) { k<<<1,1>>>(d); cudaDeviceSynchronize(); cudaFree(d); }
    printf("cuda_mini: cudaMalloc=%s\n", cudaGetErrorString(e));
    return 0;
}
