// Two streams, async copies/memsets, a cross-stream event dependency and
// every sync flavour -- the shapes the host/device correlation and the
// critical path have to get right. Used by
// tests/integration/test_cuda_native_activity.py.
//
// Per round: s1 copies HtoD and runs a producer kernel; s2 memsets and runs
// an independent kernel concurrently, then waits for s1's event and runs a
// consumer kernel and a DtoH copy. Build with the shared runtime:
//   nvcc -O2 -cudart shared -o cuda_streams cuda_streams.cu
#include <cstdio>
#include <cuda_runtime.h>

__global__ void spin(float *x, int n, int iters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        for (int k = 0; k < iters; k++) v = v * 1.0000001f + 0.5f;
        x[i] = v;
    }
}

#define CHECK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(e_)); return 1; } } while (0)

int main() {
    const int n = 1 << 20;
    const size_t bytes = n * sizeof(float);
    float *h = nullptr, *a = nullptr, *b = nullptr;
    CHECK(cudaMallocHost(&h, bytes));
    CHECK(cudaMalloc(&a, bytes));
    CHECK(cudaMalloc(&b, bytes));
    for (int i = 0; i < n; i++) h[i] = 1.0f;
    cudaStream_t s1, s2;
    CHECK(cudaStreamCreate(&s1));
    CHECK(cudaStreamCreate(&s2));
    cudaEvent_t ev;
    CHECK(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));

    for (int round = 0; round < 3; round++) {
        CHECK(cudaMemcpyAsync(a, h, bytes, cudaMemcpyHostToDevice, s1));
        CHECK(cudaMemsetAsync(b, 0, bytes, s2));
        spin<<<n / 256, 256, 0, s1>>>(a, n, 4000);   // producer
        spin<<<n / 256, 256, 0, s2>>>(b, n, 4000);   // independent, concurrent
        CHECK(cudaEventRecord(ev, s1));
        CHECK(cudaStreamWaitEvent(s2, ev, 0));
        spin<<<n / 256, 256, 0, s2>>>(a, n, 500);    // consumer: after the producer
        CHECK(cudaMemcpyAsync(h, a, bytes, cudaMemcpyDeviceToHost, s2));
        CHECK(cudaEventSynchronize(ev));
        CHECK(cudaStreamSynchronize(s2));
    }
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(h, a, bytes, cudaMemcpyDeviceToHost));
    std::printf("h[0]=%f\n", h[0]);
    cudaEventDestroy(ev);
    cudaStreamDestroy(s1);
    cudaStreamDestroy(s2);
    cudaFree(a);
    cudaFree(b);
    cudaFreeHost(h);
    return 0;
}
