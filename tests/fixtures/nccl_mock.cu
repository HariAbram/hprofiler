/* A stand-in libnccl for tests/integration/test_nccl_hook.py (no NCCL or
 * second GPU needed): one process plays rank 0 of 4. Every operation
 * launches a kernel that spins 300k cycles (~200 us) on the caller's stream, so the
 * hook's event pairs time real device work; operations issued inside
 * ncclGroupStart/End are queued and launched at ncclGroupEnd, as NCCL
 * documents for groups. Built as libnccl_mock.so; the program
 * (nccl_prog.c) links it, and the hook, preloaded, reaches it through
 * dlsym(RTLD_NEXT). */
#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>

typedef int ncclResult_t;
typedef struct { int rank, nranks; } MockComm;
typedef MockComm *ncclComm_t;
typedef struct { char internal[128]; } ncclUniqueId;

__global__ void sleep_kernel(long long cycles) {
    long long t0 = clock64();
    while (clock64() - t0 < cycles) { }
}

static int g_depth = 0;
static cudaStream_t g_pending[64];
static int g_npending = 0;

static ncclResult_t launch(cudaStream_t s) {
    if (g_depth > 0) {
        if (g_npending < 64) g_pending[g_npending++] = s;
        return 0;
    }
    sleep_kernel<<<1, 1, 0, s>>>(300000);
    return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

extern "C" {
ncclResult_t ncclGetUniqueId(ncclUniqueId *id) { memset(id, 7, sizeof(*id)); return 0; }
ncclResult_t ncclCommInitRank(ncclComm_t *comm, int nranks, ncclUniqueId id, int rank) {
    (void)id;
    *comm = (MockComm *)malloc(sizeof(MockComm));
    (*comm)->rank = rank;
    (*comm)->nranks = nranks;
    return 0;
}
ncclResult_t ncclCommUserRank(ncclComm_t c, int *r) { *r = c->rank; return 0; }
ncclResult_t ncclCommCount(ncclComm_t c, int *n) { *n = c->nranks; return 0; }
ncclResult_t ncclCommDestroy(ncclComm_t c) { free(c); return 0; }
ncclResult_t ncclAllReduce(const void *sb, void *rb, size_t n, int dt, int op, ncclComm_t c, cudaStream_t s) {
    (void)sb; (void)rb; (void)n; (void)dt; (void)op; (void)c;
    return launch(s);
}
ncclResult_t ncclSend(const void *sb, size_t n, int dt, int peer, ncclComm_t c, cudaStream_t s) {
    (void)sb; (void)n; (void)dt; (void)peer; (void)c;
    return launch(s);
}
ncclResult_t ncclRecv(void *rb, size_t n, int dt, int peer, ncclComm_t c, cudaStream_t s) {
    (void)rb; (void)n; (void)dt; (void)peer; (void)c;
    return launch(s);
}
ncclResult_t ncclGroupStart(void) { g_depth++; return 0; }
ncclResult_t ncclGroupEnd(void) {
    if (--g_depth == 0) {
        for (int i = 0; i < g_npending; i++) sleep_kernel<<<1, 1, 0, g_pending[i]>>>(300000);
        g_npending = 0;
    }
    return 0;
}
}
