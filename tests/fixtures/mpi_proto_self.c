/*
 * Single-process variant of mpi_proto.c: exercises the same MPI protocol
 * surface (wildcard Irecv resolution, Waitany/Waitsome, Test polling,
 * Cancel, Comm_split) via self-communication (rank 0 sending to itself)
 * instead of real cross-rank traffic.
 *
 * Why this exists alongside mpi_proto.c: this development machine's
 * MPICH/Hydra cannot form a real multi-rank MPI_COMM_WORLD (every rank
 * independently observes MPI_Comm_size == 1 under `mpirun -np N` for
 * N > 1, confirmed via UCX_LOG_LEVEL=info to be a PMI/KVS rank-discovery
 * failure that predates and is unrelated to hprofiler -- reproducible
 * with the pre-existing, unmodified mpi_mini.c fixture too). Self-send
 * still drives the real PMPI_Isend/Irecv/Waitany/Waitsome/Test/Cancel
 * implementation and lets this hook's status-resolution and request-table
 * logic be verified against genuine MPI completion semantics, just
 * without an independent peer process. It does NOT exercise true
 * cross-process commid= agreement (MPI_Comm_split's bootstrap Bcast is
 * only meaningfully distinct from a no-op at size > 1) -- see
 * tests/integration/test_mpi_protocol.py for what is and isn't covered
 * by each fixture.
 */
#include <mpi.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* ── Phase A: Waitany + Waitsome over 3 wildcard Irecv, self-sent ─ */
    int rbufs[3];
    MPI_Request rreqs[3];
    for (int i = 0; i < 3; i++)
        MPI_Irecv(&rbufs[i], 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &rreqs[i]);

    int sbufs[3] = {110, 120, 130};
    MPI_Request sreqs[3];
    int tags[3] = {11, 12, 13};
    for (int i = 0; i < 3; i++)
        MPI_Isend(&sbufs[i], 1, MPI_INT, rank, tags[i], MPI_COMM_WORLD, &sreqs[i]);
    MPI_Waitall(3, sreqs, MPI_STATUSES_IGNORE);

    int idx; MPI_Status st;
    MPI_Waitany(3, rreqs, &idx, &st);
    printf("phase_a waitany completed_index=%d source=%d tag=%d value=%d\n",
           idx, st.MPI_SOURCE, st.MPI_TAG, rbufs[idx]);

    int outcount, indices[3]; MPI_Status stats[3];
    MPI_Waitsome(3, rreqs, &outcount, indices, stats);
    for (int k = 0; k < outcount; k++) {
        int i = indices[k];
        printf("phase_a waitsome index=%d source=%d tag=%d value=%d\n",
               i, stats[k].MPI_SOURCE, stats[k].MPI_TAG, rbufs[i]);
    }

    /* ── Phase B: Test-poll loop, self-sent after observing flag=0 ──── */
    int rbuf4; MPI_Request rreq4;
    MPI_Irecv(&rbuf4, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &rreq4);
    int flag = 0; MPI_Status st4;
    MPI_Test(&rreq4, &flag, &st4);
    printf("phase_b test_before_send flag=%d\n", flag);

    int sbuf4 = 444; MPI_Request sreq4;
    MPI_Isend(&sbuf4, 1, MPI_INT, rank, 200, MPI_COMM_WORLD, &sreq4);
    int polls = 0;
    while (!flag && polls < 1000) { MPI_Test(&rreq4, &flag, &st4); polls++; }
    MPI_Wait(&sreq4, MPI_STATUS_IGNORE);
    printf("phase_b test_after_send polls=%d flag=%d source=%d tag=%d value=%d\n",
           polls, flag, st4.MPI_SOURCE, st4.MPI_TAG, rbuf4);

    /* ── Phase C: Cancel a never-matched wildcard receive ───────────── */
    int rbuf5; MPI_Request rreq5;
    MPI_Irecv(&rbuf5, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &rreq5);
    MPI_Cancel(&rreq5);
    MPI_Status st5;
    MPI_Wait(&rreq5, &st5);
    printf("phase_c cancel done\n");

    /* ── Phase D0: MPI_Waitall over 2 wildcard Irecv, self-sent -- the
     * rmatches= resolution logic here is a separate call site from
     * Waitsome's (same fix, different function), so needs its own check. */
    int rbufsD[2];
    MPI_Request rreqsD[2];
    for (int i = 0; i < 2; i++)
        MPI_Irecv(&rbufsD[i], 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &rreqsD[i]);
    int sbufsD[2] = {510, 520};
    MPI_Request sreqsD[2];
    MPI_Isend(&sbufsD[0], 1, MPI_INT, rank, 51, MPI_COMM_WORLD, &sreqsD[0]);
    MPI_Isend(&sbufsD[1], 1, MPI_INT, rank, 52, MPI_COMM_WORLD, &sreqsD[1]);
    MPI_Waitall(2, sreqsD, MPI_STATUSES_IGNORE);
    MPI_Status statsD[2];
    MPI_Waitall(2, rreqsD, statsD);
    printf("phase_d0 waitall source0=%d tag0=%d source1=%d tag1=%d\n",
           statsD[0].MPI_SOURCE, statsD[0].MPI_TAG, statsD[1].MPI_SOURCE, statsD[1].MPI_TAG);

    /* ── Phase D: Comm_split (trivial at size 1) + two Allreduce calls
     * on the same sub-communicator, to check the emitted commid= tag is
     * at least self-consistent across calls (does not check cross-rank
     * agreement -- needs size > 1, see file header). ─────────────────── */
    MPI_Comm subcomm;
    MPI_Comm_split(MPI_COMM_WORLD, 0, rank, &subcomm);
    double x = 7.0, sum1 = 0.0, sum2 = 0.0;
    MPI_Allreduce(&x, &sum1, 1, MPI_DOUBLE, MPI_SUM, subcomm);
    MPI_Allreduce(&x, &sum2, 1, MPI_DOUBLE, MPI_SUM, subcomm);
    printf("phase_d sum1=%f sum2=%f\n", sum1, sum2);
    MPI_Comm_free(&subcomm);

    printf("mpi_proto_self: rank=%d size=%d done\n", rank, size);
    MPI_Finalize();
    return 0;
}
