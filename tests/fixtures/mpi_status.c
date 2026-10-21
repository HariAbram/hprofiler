/* MPI semantics under the hook with real ranks (tests/integration/
 * test_mpi_multirank.py), run with exactly 4 ranks. Prints what the
 * application observes, so the test can compare it with an unprofiled run.
 *
 *  A  statuses: rank 0 completes 3 exact Irecv with MPI_Waitall, 3 wildcard
 *     Irecv with MPI_Waitsome, 3 with MPI_Testall, 3 with MPI_Testsome --
 *     each with a caller-supplied status array; prints source/tag/error.
 *  B  cancel: rank 0 cancels a never-matched wildcard Irecv, waits on it with
 *     MPI_STATUS_IGNORE (needs the hook's own status), prints cancelled.
 *  C  sub-communicator: split by rank / 2 with key = -rank (local ranks
 *     reversed), local 1 sends tag 7 to local 0 in each half.
 *  D  two MPI_Comm_dup of MPI_COMM_WORLD: rank 1 sends tag 5 on dup X then
 *     dup Y; rank 0 receives on Y first, then X.
 *  E  ring exchange with MPI_Sendrecv and MPI_Sendrecv_replace.
 *  F  MPI_Ibarrier + MPI_Wait, MPI_Probe/MPI_Iprobe, Issend, Rsend (after a
 *     barrier, so the receive is posted), Allgatherv, Alltoallv, Gatherv,
 *     Scatterv, Reduce_scatter, Reduce_scatter_block, PROC_NULL sends.
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 4) {
        if (rank == 0) printf("mpi_status requires exactly 4 ranks\n");
        MPI_Finalize();
        return 1;
    }
    int v[3] = {0, 0, 0}, idx[3], n = 0;
    MPI_Request rq[3];
    MPI_Status st[3];

    /* A: Waitall (exact), Waitsome / Testall / Testsome (wildcard) */
    for (int round = 0; round < 4; round++) {
        int tagbase = 100 * (round + 1);
        if (rank == 0) {
            for (int i = 0; i < 3; i++)
                MPI_Irecv(&v[i], 1, MPI_INT, round == 0 ? i + 1 : MPI_ANY_SOURCE,
                          round == 0 ? tagbase + i + 1 : MPI_ANY_TAG, MPI_COMM_WORLD, &rq[i]);
            for (int i = 0; i < 3; i++) { st[i].MPI_SOURCE = -7; st[i].MPI_TAG = -7; }
            int done = 0;
            if (round == 0) {
                MPI_Waitall(3, rq, st);
                for (int i = 0; i < 3; i++)
                    printf("A waitall %d source=%d tag=%d value=%d\n", i, st[i].MPI_SOURCE, st[i].MPI_TAG, v[i]);
            } else if (round == 1) {
                while (done < 3) {
                    MPI_Waitsome(3, rq, &n, idx, st);
                    for (int k = 0; k < n; k++)
                        printf("A waitsome idx=%d source=%d tag=%d value=%d\n", idx[k], st[k].MPI_SOURCE,
                               st[k].MPI_TAG, v[idx[k]]);
                    done += n;
                }
            } else if (round == 2) {
                int flag = 0;
                while (!flag) MPI_Testall(3, rq, &flag, st);
                for (int i = 0; i < 3; i++)
                    printf("A testall %d source=%d tag=%d value=%d\n", i, st[i].MPI_SOURCE, st[i].MPI_TAG, v[i]);
            } else {
                while (done < 3) {
                    MPI_Testsome(3, rq, &n, idx, st);
                    if (n == MPI_UNDEFINED) break;
                    for (int k = 0; k < n; k++)
                        printf("A testsome idx=%d source=%d tag=%d value=%d\n", idx[k], st[k].MPI_SOURCE,
                               st[k].MPI_TAG, v[idx[k]]);
                    done += n;
                }
            }
        } else {
            int x = rank;
            MPI_Send(&x, 1, MPI_INT, 0, tagbase + rank, MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    /* B: cancel a wildcard receive, wait with MPI_STATUS_IGNORE */
    if (rank == 0) {
        int dummy;
        MPI_Request r;
        MPI_Status s;
        MPI_Irecv(&dummy, 1, MPI_INT, MPI_ANY_SOURCE, 999, MPI_COMM_WORLD, &r);
        MPI_Cancel(&r);
        MPI_Wait(&r, MPI_STATUS_IGNORE);
        MPI_Irecv(&dummy, 1, MPI_INT, MPI_ANY_SOURCE, 998, MPI_COMM_WORLD, &r);
        MPI_Cancel(&r);
        MPI_Wait(&r, &s);
        int c = 0;
        MPI_Test_cancelled(&s, &c);
        printf("B cancelled=%d\n", c);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    /* C: sub-communicator with reversed local ranks */
    MPI_Comm half;
    MPI_Comm_split(MPI_COMM_WORLD, rank / 2, -rank, &half);
    int hr;
    MPI_Comm_rank(half, &hr);
    int msg = rank;
    if (hr == 1) MPI_Send(&msg, 1, MPI_INT, 0, 7, half);
    else {
        MPI_Recv(&msg, 1, MPI_INT, 1, 7, half, MPI_STATUS_IGNORE);
        printf("C world=%d local=%d got=%d\n", rank, hr, msg);
    }
    MPI_Comm_free(&half);

    /* D: two duplicates, same peers and tag, received in the other order */
    MPI_Comm x, y;
    MPI_Comm_dup(MPI_COMM_WORLD, &x);
    MPI_Comm_dup(MPI_COMM_WORLD, &y);
    if (rank == 1) {
        int a = 11, b = 22;
        MPI_Request sr[2];
        MPI_Isend(&a, 1, MPI_INT, 0, 5, x, &sr[0]);
        MPI_Isend(&b, 1, MPI_INT, 0, 5, y, &sr[1]);
        MPI_Waitall(2, sr, MPI_STATUSES_IGNORE);
    } else if (rank == 0) {
        int a = 0, b = 0;
        MPI_Recv(&b, 1, MPI_INT, 1, 5, y, MPI_STATUS_IGNORE);
        MPI_Recv(&a, 1, MPI_INT, 1, 5, x, MPI_STATUS_IGNORE);
        printf("D x=%d y=%d\n", a, b);
    }
    MPI_Comm_free(&x);
    MPI_Comm_free(&y);

    /* E: ring exchanges */
    int right = (rank + 1) % size, left = (rank + size - 1) % size, got = -1, rep = rank;
    MPI_Sendrecv(&rank, 1, MPI_INT, right, 3, &got, 1, MPI_INT, left, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv_replace(&rep, 1, MPI_INT, left, 4, right, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    printf("E rank=%d got=%d replaced=%d\n", rank, got, rep);

    /* F: the rest of the newly hooked calls */
    MPI_Request ib;
    MPI_Ibarrier(MPI_COMM_WORLD, &ib);
    MPI_Wait(&ib, MPI_STATUS_IGNORE);
    if (rank == 2) {
        int t = 5;
        MPI_Request r;
        MPI_Issend(&t, 1, MPI_INT, 3, 8, MPI_COMM_WORLD, &r);
        MPI_Wait(&r, MPI_STATUS_IGNORE);
    } else if (rank == 3) {
        int t = 0, flag = 0;
        MPI_Status s;
        MPI_Probe(2, 8, MPI_COMM_WORLD, &s);
        MPI_Iprobe(2, 8, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
        MPI_Recv(&t, 1, MPI_INT, s.MPI_SOURCE, s.MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("F probe source=%d tag=%d flag=%d value=%d\n", s.MPI_SOURCE, s.MPI_TAG, flag, t);
    }
    int rs_val = 0;
    MPI_Request rr = MPI_REQUEST_NULL;
    if (rank == 0) MPI_Irecv(&rs_val, 1, MPI_INT, 1, 9, MPI_COMM_WORLD, &rr);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 1) { int t = 9; MPI_Rsend(&t, 1, MPI_INT, 0, 9, MPI_COMM_WORLD); }
    if (rank == 0) { MPI_Wait(&rr, MPI_STATUS_IGNORE); printf("F rsend value=%d\n", rs_val); }
    MPI_Send(&rank, 1, MPI_INT, MPI_PROC_NULL, 0, MPI_COMM_WORLD);
    MPI_Recv(&got, 1, MPI_INT, MPI_PROC_NULL, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    int counts[4] = {1, 1, 1, 1}, displs[4] = {0, 1, 2, 3}, all[4], mine = rank, sum[4], red;
    MPI_Allgatherv(&mine, 1, MPI_INT, all, counts, displs, MPI_INT, MPI_COMM_WORLD);
    MPI_Alltoallv(all, counts, displs, MPI_INT, sum, counts, displs, MPI_INT, MPI_COMM_WORLD);
    MPI_Gatherv(&mine, 1, MPI_INT, all, counts, displs, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(all, counts, displs, MPI_INT, &mine, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Reduce_scatter(all, &red, counts, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Reduce_scatter_block(all, &red, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    printf("F collectives rank=%d allgather=%d,%d,%d,%d\n", rank, all[0], all[1], all[2], all[3]);

    MPI_Finalize();
    return 0;
}
