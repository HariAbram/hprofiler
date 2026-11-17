// A CUDA process that vforks and execs children (as Python's subprocess
// does: Triton compiles its launcher with cc this way) and synchronizes on
// events through the runtime and the driver API, for
// tests/integration/test_cuda_vfork_events.py:
//
//   1. vfork + execv(/bin/true) before any CUDA call (Triton's driver init
//      compiles cuda_utils.c with cc before the first tensor exists);
//   2. a kernel, cudaEventRecord, a cudaEventQuery while a ~50 ms kernel
//      still runs (not ready), cudaEventSynchronize, three more queries
//      (ready; CUDA 13's CUPTI reports every query as an event
//      synchronization record);
//   3. vfork + execv(/bin/true) in mid-run;
//   4. the same through the driver API: cuEventRecord, cuEventQuery,
//      cuEventSynchronize;
//   5. cudaDeviceSynchronize.
//
// Prints "pid=<pid> tid=<tid>" first and "notready=<0|1> done" last.
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>

__global__ void spin(long long cycles) {
    long long t0 = clock64();
    while (clock64() - t0 < cycles) {}
}

static void run_child(void) {
    pid_t c = vfork();
    if (c == 0) {
        char *args[] = {(char *)"true", NULL};
        execv("/bin/true", args);
        _exit(127);
    }
    int st = 0;
    waitpid(c, &st, 0);
}

#define CK(x) do { if ((x) != 0) { fprintf(stderr, "%s failed\n", #x); return 2; } } while (0)

int main(void) {
    printf("pid=%d tid=%ld\n", (int)getpid(), (long)syscall(SYS_gettid));
    fflush(stdout);
    run_child();

    int clock_khz = 0;
    CK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0));
    long long ms50 = (long long)clock_khz * 50;           // ~50 ms of spinning
    cudaEvent_t e1, e2;
    CK(cudaEventCreate(&e1));
    CK(cudaEventCreate(&e2));
    spin<<<1, 1>>>(1000);
    CK(cudaEventRecord(e1, 0));
    spin<<<1, 1>>>(ms50);
    CK(cudaEventRecord(e2, 0));
    int notready = cudaEventQuery(e2) == cudaErrorNotReady;
    CK(cudaEventSynchronize(e2));
    for (int i = 0; i < 3; i++) CK(cudaEventQuery(e2));
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, e1, e2));

    run_child();

    CUevent d;
    CK(cuEventCreate(&d, CU_EVENT_DEFAULT));
    spin<<<1, 1>>>(ms50 / 5);
    CK(cuEventRecord(d, 0));
    CK(cuEventSynchronize(d));
    CK(cuEventQuery(d));
    CK(cuEventDestroy(d));
    CK(cudaDeviceSynchronize());
    CK(cudaEventDestroy(e1));
    CK(cudaEventDestroy(e2));
    printf("notready=%d done\n", notready);
    return 0;
}
