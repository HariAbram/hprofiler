/*
 * Fixture exercising the MPI protocol-semantics features added to
 * hooks/mpi_hook/mpi_hook.c for cross-layer causal attribution:
 *   - wildcard MPI_Irecv (ANY_SOURCE/ANY_TAG) resolved via MPI_Waitany
 *     and MPI_Waitsome, including status resolution even when the
 *     original request array has already-completed (NULL) slots.
 *   - MPI_Test polling loop observing flag=0 then flag=1.
 *   - MPI_Cancel on a never-matched wildcard receive.
 *   - MPI_Comm_split into two disjoint sub-communicators, each doing an
 *     MPI_Allreduce, to check commid= agreement/uniqueness.
 *
 * Requires exactly 4 ranks (tests/integration/test_mpi_protocol.py runs
 * it via `mpirun -np 4`).
 */
#include <mpi.h>
#include <stdio.h>
#include <unistd.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 4) {
        if (rank == 0) fprintf(stderr, "mpi_proto requires exactly 4 ranks, got %d\n", size);
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        /* ── Phase A: Waitany + Waitsome over 3 wildcard Irecv ──────── */
        int bufs[3];
        MPI_Request reqs[3];
        for (int i = 0; i < 3; i++)
            MPI_Irecv(&bufs[i], 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &reqs[i]);

        int idx; MPI_Status st;
        MPI_Waitany(3, reqs, &idx, &st);
        printf("phase_a waitany completed_index=%d source=%d tag=%d value=%d\n",
               idx, st.MPI_SOURCE, st.MPI_TAG, bufs[idx]);

        int outcount, indices[3]; MPI_Status stats[3];
        MPI_Waitsome(3, reqs, &outcount, indices, stats);
        for (int k = 0; k < outcount; k++) {
            int i = indices[k];
            printf("phase_a waitsome index=%d source=%d tag=%d value=%d\n",
                   i, stats[k].MPI_SOURCE, stats[k].MPI_TAG, bufs[i]);
        }

        /* ── Phase B: Test-poll loop until a delayed send arrives ───── */
        int buf4; MPI_Request req4;
        MPI_Irecv(&buf4, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &req4);
        int flag = 0; MPI_Status st4;
        int polls = 0;
        while (!flag && polls < 200) {
            MPI_Test(&req4, &flag, &st4);
            if (!flag) { usleep(10000); polls++; }
        }
        printf("phase_b test polls=%d flag=%d source=%d tag=%d value=%d\n",
               polls, flag, st4.MPI_SOURCE, st4.MPI_TAG, buf4);

        /* ── Phase C: Cancel a never-matched wildcard receive ───────── */
        int buf5; MPI_Request req5;
        MPI_Irecv(&buf5, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &req5);
        MPI_Cancel(&req5);
        MPI_Status st5;
        MPI_Wait(&req5, &st5);
        printf("phase_c cancel done\n");

        MPI_Barrier(MPI_COMM_WORLD);

        /* ── Phase D: Comm_split + Allreduce (commid= agreement) ─────── */
        MPI_Comm subcomm;
        MPI_Comm_split(MPI_COMM_WORLD, rank % 2, rank, &subcomm);
        double x = rank, sum = 0.0;
        MPI_Allreduce(&x, &sum, 1, MPI_DOUBLE, MPI_SUM, subcomm);
        printf("phase_d rank=%d color=%d sum=%f\n", rank, rank % 2, sum);
        MPI_Comm_free(&subcomm);

    } else {
        /* Ranks 1,2,3: phase A, one blocking send each, tag = 100+rank so
         * rank 0 can check the resolved wildcard match against known
         * ground truth. */
        int payload = rank;
        MPI_Send(&payload, 1, MPI_INT, 0, 100 + rank, MPI_COMM_WORLD);

        if (rank == 1) {
            /* Phase B: delayed second send, matched by rank 0's Test loop. */
            usleep(300000);
            int payload2 = 999;
            MPI_Send(&payload2, 1, MPI_INT, 0, 200, MPI_COMM_WORLD);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        MPI_Comm subcomm;
        MPI_Comm_split(MPI_COMM_WORLD, rank % 2, rank, &subcomm);
        double x = rank, sum = 0.0;
        MPI_Allreduce(&x, &sum, 1, MPI_DOUBLE, MPI_SUM, subcomm);
        printf("phase_d rank=%d color=%d sum=%f\n", rank, rank % 2, sum);
        MPI_Comm_free(&subcomm);
    }

    MPI_Finalize();
    return 0;
}
