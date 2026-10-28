/* Persistent requests, rooted and non-blocking collectives under the MPI
 * hook (tests/integration/test_mpi_persist.py), 4 ranks:
 *   A  a ring of persistent requests (rank r sends to r+1, receives from
 *      r-1): 3 rounds of MPI_Startall + MPI_Waitall, then 2 rounds of one
 *      MPI_Start per request + MPI_Wait each, then MPI_Request_free
 *   B  rooted collectives on MPI_COMM_WORLD (Bcast from 2, Gather to 3) and
 *      on a split, renumbered communicator (Reduce to its rank 1: world
 *      rank 2 in the half {0, 2}... see below)
 *   C  non-blocking collectives (Ibcast from 1, Iallreduce) + MPI_Wait
 * Prints the values each rank received ("A ..", "B ..", "C .."). */
#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 4) {
        if (rank == 0) printf("requires exactly 4 ranks\n");
        MPI_Finalize();
        return 1;
    }
    int right = (rank + 1) % 4, left = (rank + 3) % 4;

    /* A: persistent ring */
    int sendv = 0, recvv = -1;
    MPI_Request reqs[2];
    MPI_Status st[2];
    MPI_Send_init(&sendv, 1, MPI_INT, right, 7, MPI_COMM_WORLD, &reqs[0]);
    MPI_Recv_init(&recvv, 1, MPI_INT, left, 7, MPI_COMM_WORLD, &reqs[1]);
    for (int round = 0; round < 5; round++) {
        sendv = 100 * rank + round;
        if (round < 3) {
            MPI_Startall(2, reqs);
            MPI_Waitall(2, reqs, st);
        } else {
            MPI_Start(&reqs[1]);
            MPI_Start(&reqs[0]);
            MPI_Wait(&reqs[0], MPI_STATUS_IGNORE);
            MPI_Wait(&reqs[1], MPI_STATUS_IGNORE);
        }
        printf("A %d %d %d\n", rank, round, recvv);
    }
    MPI_Request_free(&reqs[0]);
    MPI_Request_free(&reqs[1]);

    /* B: rooted collectives; the split puts world ranks {0, 2} and {1, 3}
     * together, renumbered 0 and 1 -- the root, local rank 1, is world
     * rank 2 resp. 3 */
    int b = rank == 2 ? 42 : 0;
    MPI_Bcast(&b, 1, MPI_INT, 2, MPI_COMM_WORLD);
    int g[4] = {0, 0, 0, 0};
    MPI_Gather(&rank, 1, MPI_INT, g, 1, MPI_INT, 3, MPI_COMM_WORLD);
    MPI_Comm half;
    MPI_Comm_split(MPI_COMM_WORLD, rank % 2, rank, &half);
    int r = 0, one = rank + 1;
    MPI_Reduce(&one, &r, 1, MPI_INT, MPI_SUM, 1, half);
    printf("B %d %d %d %d\n", rank, b, rank == 3 ? g[0] + g[1] + g[2] + g[3] : -1, r);

    /* C: non-blocking collectives */
    int c = rank == 1 ? 9 : 0, s = 0;
    MPI_Request cr[2];
    MPI_Ibcast(&c, 1, MPI_INT, 1, MPI_COMM_WORLD, &cr[0]);
    MPI_Wait(&cr[0], MPI_STATUS_IGNORE);
    MPI_Iallreduce(&rank, &s, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD, &cr[1]);
    MPI_Wait(&cr[1], MPI_STATUS_IGNORE);
    printf("C %d %d %d\n", rank, c, s);

    MPI_Comm_free(&half);
    MPI_Finalize();
    return 0;
}
