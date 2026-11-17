/*
 * HIP stream lives, host functions, launch and copy variants, for
 * tests/integration/test_rocm_hip_mock.py (run against tests/fixtures/
 * hip_mock.c's libamdhip64.so: no AMD GPU needed, no device execution
 * modelled). Prints the order the host functions ran in and whether the
 * stream / event addresses were reused.
 */
#include <stdio.h>
#include <string.h>

typedef int hipError_t;
typedef void *hipStream_t;
typedef void *hipEvent_t;
typedef struct { int x, y, z; } dim3;
typedef void (*hipHostFn_t)(void *);
typedef void (*hipStreamCallback_t)(hipStream_t, hipError_t, void *);
extern hipError_t hipStreamCreate(hipStream_t *);
extern hipError_t hipStreamCreateWithPriority(hipStream_t *, unsigned, int);
extern hipError_t hipStreamDestroy(hipStream_t);
extern hipError_t hipStreamSynchronize(hipStream_t);
extern hipError_t hipDeviceSynchronize(void);
extern hipError_t hipLaunchHostFunc(hipStream_t, hipHostFn_t, void *);
extern hipError_t hipStreamAddCallback(hipStream_t, hipStreamCallback_t, void *, unsigned);
extern hipError_t hipLaunchKernel(const void *, dim3, dim3, void **, size_t, hipStream_t);
extern hipError_t hipLaunchCooperativeKernel(const void *, dim3, dim3, void **, unsigned, hipStream_t);
extern hipError_t hipExtLaunchKernel(const void *, dim3, dim3, void **, size_t, hipStream_t, hipEvent_t, hipEvent_t, int);
extern hipError_t hipEventCreate(hipEvent_t *);
extern hipError_t hipEventDestroy(hipEvent_t);
extern hipError_t hipEventRecord(hipEvent_t, hipStream_t);
extern hipError_t hipMemcpy2D(void *, size_t, const void *, size_t, size_t, size_t, int);
extern hipError_t hipMemcpy2DAsync(void *, size_t, const void *, size_t, size_t, size_t, int, hipStream_t);
extern hipError_t hipMemcpyPeer(void *, int, const void *, int, size_t);
extern hipError_t hipMemcpyPeerAsync(void *, int, const void *, int, size_t, hipStream_t);
extern hipError_t hipMemcpyDtoD(void *, void *, size_t);
extern hipError_t hipMemcpyDtoDAsync(void *, void *, size_t, hipStream_t);
extern hipError_t hipMemcpyHtoDAsync(void *, void *, size_t, hipStream_t);
extern hipError_t hipMemcpyDtoHAsync(void *, void *, size_t, hipStream_t);

static char order[64];
static void f1(void *u) { (void)u; strcat(order, "f1,"); }
static void f3(void *u) { (void)u; strcat(order, "f3,"); }
static void cb2(hipStream_t s, hipError_t e, void *u) { (void)s; (void)e; (void)u; strcat(order, "cb2,"); }
static void kern(void) {}

int main(void) {
    hipStream_t s1, s2, s3;
    hipStreamCreateWithPriority(&s1, 1 /* hipStreamNonBlocking */, -1);
    hipStreamCreate(&s2);
    hipLaunchHostFunc(s1, f1, NULL);
    hipStreamAddCallback(s2, cb2, NULL, 0);
    hipLaunchHostFunc(s1, f3, NULL);
    hipDeviceSynchronize();

    dim3 g = {1, 1, 1}, b = {32, 1, 1};
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    hipLaunchKernel((const void *)kern, g, b, NULL, 0, s1);
    hipLaunchCooperativeKernel((const void *)kern, g, b, NULL, 0, s2);
    hipExtLaunchKernel((const void *)kern, g, b, NULL, 0, s1, start, stop, 0);

    char a[256], c[256];
    memset(a, 1, sizeof(a));
    hipMemcpy2D(c, 32, a, 32, 16, 4, 1);
    hipMemcpy2DAsync(c, 32, a, 32, 16, 4, 1, s1);
    hipMemcpyPeer(c, 0, a, 0, 64);
    hipMemcpyPeerAsync(c, 0, a, 0, 64, s2);
    hipMemcpyDtoD(c, a, 32);
    hipMemcpyDtoDAsync(c, a, 32, s1);
    hipMemcpyHtoDAsync(c, a, 16, s2);
    hipMemcpyDtoHAsync(c, a, 8, s2);
    hipDeviceSynchronize();

    void *old = s2;
    hipStreamDestroy(s2);
    hipStreamCreate(&s3);
    hipLaunchKernel((const void *)kern, g, b, NULL, 0, s3);
    void *eold = stop;
    hipEventDestroy(stop);
    hipEventCreate(&stop);
    hipEventRecord(start, s3);
    hipEventRecord(stop, s3);
    hipDeviceSynchronize();
    printf("order=%s reused_stream=%d reused_event=%d\n", order, (void *)s3 == old, (void *)stop == eold);
    return 0;
}
