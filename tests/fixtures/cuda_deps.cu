// Blocking copies, stream flags and host functions under the CUDA hook
// (tests/integration/test_cuda_deps.py):
//   streams    s_block (cudaStreamCreate), s_nb (NonBlocking), s_prio
//              (cudaStreamCreateWithPriority)
//   copies     cudaMemcpy pageable->device (blocking=staged), pinned->device
//              (1), device->host (1), device->device (0), cudaMemcpyDefault
//              pinned->device (1)
//   memset     cudaMemset on device memory (asynchronous: no blocking tag)
//   host fns   on s_nb: kernel, cudaLaunchHostFunc(sleep 2 ms), kernel;
//              on s_block: kernel, cudaStreamAddCallback(sleep 1 ms)
// Prints "host_fn=1 callback=1" once both ran.
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <cuda_runtime.h>

__global__ void spin(long long cycles) {
    long long t0 = clock64();
    while (clock64() - t0 < cycles) { }
}

static volatile int g_host_fn = 0, g_callback = 0;

static void host_fn(void *) { usleep(2000); g_host_fn = 1; }
static void CUDART_CB stream_cb(cudaStream_t, cudaError_t, void *) { usleep(1000); g_callback = 1; }

int main() {
    cudaStream_t s_block, s_nb, s_prio;
    if (cudaStreamCreate(&s_block) != cudaSuccess) { printf("no CUDA device\n"); return 77; }
    cudaStreamCreateWithFlags(&s_nb, cudaStreamNonBlocking);
    int lo = 0, hi = 0;
    cudaDeviceGetStreamPriorityRange(&lo, &hi);
    cudaStreamCreateWithPriority(&s_prio, cudaStreamDefault, hi);
    const size_t n = 1 << 20;
    char *pageable = (char *)malloc(n), *pinned, *d1, *d2;
    cudaMallocHost((void **)&pinned, n);
    cudaMalloc((void **)&d1, n);
    cudaMalloc((void **)&d2, n);
    cudaMemcpy(d1, pageable, n, cudaMemcpyHostToDevice);
    cudaMemcpy(d1, pinned, n, cudaMemcpyHostToDevice);
    cudaMemcpy(pageable, d1, n, cudaMemcpyDeviceToHost);
    cudaMemcpy(d2, d1, n, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d2, pinned, n, cudaMemcpyDefault);
    cudaMemset(d1, 0, n);
    spin<<<1, 1, 0, s_nb>>>(200000);
    cudaLaunchHostFunc(s_nb, host_fn, NULL);
    spin<<<1, 1, 0, s_nb>>>(200000);
    spin<<<1, 1, 0, s_block>>>(200000);
    cudaStreamAddCallback(s_block, stream_cb, NULL, 0);
    spin<<<1, 1, 0, s_prio>>>(100000);
    cudaDeviceSynchronize();
    printf("host_fn=%d callback=%d\n", g_host_fn, g_callback);
    cudaFree(d1);
    cudaFree(d2);
    cudaFreeHost(pinned);
    free(pageable);
    return 0;
}
