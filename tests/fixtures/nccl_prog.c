/* NCCL calls with known structure (tests/integration/test_nccl_hook.py),
 * against nccl_mock.cu, as rank 0 of 4 with two streams:
 *   send to 1, send to 1, recv from 3, send to 2           (outside a group)
 *   group { send to 1 on s1, recv from 3 on s2 }
 *   group { allreduce on s1, group { send to 1 on s2 } }   (nested)
 *   allreduce on s1
 * Prints "done". */
#include <stdio.h>
#include <cuda_runtime.h>

typedef int ncclResult_t;
typedef void *ncclComm_t;
typedef struct { char internal[128]; } ncclUniqueId;
ncclResult_t ncclGetUniqueId(ncclUniqueId *);
ncclResult_t ncclCommInitRank(ncclComm_t *, int, ncclUniqueId, int);
ncclResult_t ncclCommDestroy(ncclComm_t);
ncclResult_t ncclAllReduce(const void *, void *, size_t, int, int, ncclComm_t, cudaStream_t);
ncclResult_t ncclSend(const void *, size_t, int, int, ncclComm_t, cudaStream_t);
ncclResult_t ncclRecv(void *, size_t, int, int, ncclComm_t, cudaStream_t);
ncclResult_t ncclGroupStart(void);
ncclResult_t ncclGroupEnd(void);

int main(void) {
    cudaStream_t s1, s2;
    if (cudaStreamCreate(&s1) != cudaSuccess || cudaStreamCreate(&s2) != cudaSuccess) {
        printf("no CUDA device\n");
        return 77;
    }
    ncclUniqueId id;
    ncclComm_t comm;
    ncclGetUniqueId(&id);
    ncclCommInitRank(&comm, 4, id, 0);
    float *buf;
    cudaMalloc((void **)&buf, 1024 * sizeof(float));
    ncclSend(buf, 256, 7, 1, comm, s1);
    ncclSend(buf, 256, 7, 1, comm, s1);
    ncclRecv(buf, 256, 7, 3, comm, s1);
    ncclSend(buf, 256, 7, 2, comm, s1);
    ncclGroupStart();
    ncclSend(buf, 256, 7, 1, comm, s1);
    ncclRecv(buf, 256, 7, 3, comm, s2);
    ncclGroupEnd();
    ncclGroupStart();
    ncclAllReduce(buf, buf, 1024, 7, 0, comm, s1);
    ncclGroupStart();
    ncclSend(buf, 256, 7, 1, comm, s2);
    ncclGroupEnd();
    ncclGroupEnd();
    ncclAllReduce(buf, buf, 1024, 7, 0, comm, s1);
    cudaDeviceSynchronize();
    ncclCommDestroy(comm);
    printf("done\n");
    return 0;
}
