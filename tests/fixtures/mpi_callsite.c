/*
 * Call-site attribution fixture for tests/integration/test_callsite_e2e.py:
 * every point-to-point and request-based MPI operation the hook wraps,
 * issued from named, non-inlined functions (build with -rdynamic so dladdr
 * resolves them), by one self-communicating rank -- this machine's Hydra
 * cannot form a multi-rank job (see tests/integration/test_mpi_protocol.py).
 *
 * many_requests() completes NREQ receives in one MPI_Waitall: its request-id
 * list is longer than the hook's bounded list, which must report the rest as
 * psid_omitted=N instead of overrunning a buffer.
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define NREQ 2000

__attribute__((noinline)) void blocking_pairs(int rank) {
    int out = 7, in = 0;
    MPI_Request r, s;
    MPI_Irecv(&in, 1, MPI_INT, rank, 1, MPI_COMM_WORLD, &r);
    MPI_Send(&out, 1, MPI_INT, rank, 1, MPI_COMM_WORLD);
    MPI_Wait(&r, MPI_STATUS_IGNORE);

    MPI_Irecv(&in, 1, MPI_INT, rank, 2, MPI_COMM_WORLD, &r);
    MPI_Ssend(&out, 1, MPI_INT, rank, 2, MPI_COMM_WORLD);
    MPI_Wait(&r, MPI_STATUS_IGNORE);

    MPI_Isend(&out, 1, MPI_INT, rank, 3, MPI_COMM_WORLD, &s);
    MPI_Recv(&in, 1, MPI_INT, rank, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Wait(&s, MPI_STATUS_IGNORE);
}

__attribute__((noinline)) void many_requests(int rank) {
    int *in = calloc(NREQ, sizeof(int)), *out = calloc(NREQ, sizeof(int));
    MPI_Request *rr = malloc(NREQ * sizeof(MPI_Request));
    MPI_Request *sr = malloc(NREQ * sizeof(MPI_Request));
    for (int i = 0; i < NREQ; i++)
        MPI_Irecv(&in[i], 1, MPI_INT, rank, 100, MPI_COMM_WORLD, &rr[i]);
    for (int i = 0; i < NREQ; i++) {
        out[i] = i;
        MPI_Isend(&out[i], 1, MPI_INT, rank, 100, MPI_COMM_WORLD, &sr[i]);
    }
    MPI_Waitall(NREQ, sr, MPI_STATUSES_IGNORE);
    MPI_Waitall(NREQ, rr, MPI_STATUSES_IGNORE);
    long sum = 0;
    for (int i = 0; i < NREQ; i++) sum += in[i];
    printf("many_requests sum=%ld\n", sum);
    free(in); free(out); free(rr); free(sr);
}

__attribute__((noinline)) void polling(int rank) {
    int in[3] = {0}, out[3] = {1, 2, 3}, flag = 0, idx = -1, outcount = 0, indices[3];
    MPI_Request r[3], s[3];
    for (int i = 0; i < 3; i++) MPI_Irecv(&in[i], 1, MPI_INT, rank, 200 + i, MPI_COMM_WORLD, &r[i]);
    MPI_Test(&r[0], &flag, MPI_STATUS_IGNORE);
    for (int i = 0; i < 3; i++) MPI_Isend(&out[i], 1, MPI_INT, rank, 200 + i, MPI_COMM_WORLD, &s[i]);
    MPI_Waitany(3, r, &idx, MPI_STATUS_IGNORE);
    MPI_Waitsome(3, r, &outcount, indices, MPI_STATUSES_IGNORE);
    for (int polls = 0; polls < 100000 && !flag; polls++)
        MPI_Testall(3, r, &flag, MPI_STATUSES_IGNORE);
    flag = 0;
    for (int polls = 0; polls < 100000 && !flag; polls++)
        MPI_Testany(3, s, &idx, &flag, MPI_STATUS_IGNORE);
    MPI_Testsome(3, s, &outcount, indices, MPI_STATUSES_IGNORE);
    MPI_Waitall(3, s, MPI_STATUSES_IGNORE);

    int never = 0;
    MPI_Request c;
    MPI_Irecv(&never, 1, MPI_INT, rank, 999, MPI_COMM_WORLD, &c);
    MPI_Cancel(&c);
    MPI_Wait(&c, MPI_STATUS_IGNORE);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    blocking_pairs(rank);
    many_requests(rank);
    polling(rank);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
