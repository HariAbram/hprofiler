/*
 * Fixture for the MPI hook's sequence numbers (sseq= / rseq= / cseq=) and
 * matched-probe links (mid=): tests/integration/test_mpi_sequence.py runs it
 * on 4 ranks, unprofiled and under the hook.
 *
 * Every message to rank 0 on a communicator carries a value that names its
 * send: 1000 * world rank + n, where n counts that rank's sends to rank 0 on
 * that communicator (the hook's sseq= for the same send). Rank 0 prints
 * what each receive got, in call order -- the ground truth the test checks
 * the dependency graph's pairing against.
 *
 *   A  MPI_COMM_WORLD: each of ranks 1..3 sends three messages with one tag;
 *      rank 0 receives them exactly, with a tag wildcard, exactly again (one
 *      channel, mixed); then one more message each, received by three
 *      ANY_SOURCE/ANY_TAG MPI_Irecv and MPI_Waitall.
 *   B  three more communicators (a dup, a split into {0,1}/{2,3}, a 1-D
 *      cartesian one) with the same tag; a collective of another type on
 *      each, MPI_Iallgatherv and MPI_Iscan on MPI_COMM_WORLD.
 *   C  (MPI_THREAD_MULTIPLE) on a dup: two threads of rank 1 send 20
 *      messages each (tags 100, 101) to two threads of rank 0 receiving one
 *      tag each; on another dup, rank 1 sends 10 messages with tag 102 that
 *      two threads of rank 0 receive concurrently (which thread gets which is
 *      the library's choice).
 *   D  matched probes on MPI_COMM_WORLD: rank 2's five messages through
 *      MPI_Mprobe + MPI_Mrecv in a loop (message handles reused), rank 3's
 *      three through MPI_Improbe polling + MPI_Imrecv + MPI_Wait, a probe of
 *      MPI_PROC_NULL (MPI_MESSAGE_NO_PROC) and its MPI_Mrecv, and one message
 *      of rank 1 probed through PMPI_Mprobe -- which the hook does not see --
 *      and received with MPI_Mrecv.
 *   E  (MPI 4 only) a persistent MPI_Allreduce_init started twice.
 *
 * Requires exactly 4 ranks.
 */
#include <mpi.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/syscall.h>
#include <unistd.h>

static int rank;
static int sent_world = 0;               /* this rank's sends to rank 0 on MPI_COMM_WORLD */

static void send_world(int tag) {
    int v = 1000 * rank + (++sent_world);
    MPI_Send(&v, 1, MPI_INT, 0, tag, MPI_COMM_WORLD);
}

static MPI_Comm comm_d, comm_e;

static void *sender_thread(void *arg) {          /* phase C, rank 1 */
    int t = (int)(long)arg;
    for (int j = 0; j < 20; j++) {
        int v = 6000 + 100 * t + j;
        MPI_Send(&v, 1, MPI_INT, 0, 100 + t, comm_d);
    }
    return NULL;
}

static void *receiver_thread(void *arg) {        /* phase C, rank 0 */
    int t = (int)(long)arg;
    long tid = syscall(SYS_gettid);
    for (int j = 0; j < 20; j++) {
        int v;
        MPI_Recv(&v, 1, MPI_INT, 1, 100 + t, comm_d, MPI_STATUS_IGNORE);
        printf("C2 tid=%ld tag=%d k=%d value=%d\n", tid, 100 + t, j, v);
    }
    /* the contested channel: both threads receive tag 102 at once */
    for (int j = 0; j < 5; j++) {
        int v;
        MPI_Recv(&v, 1, MPI_INT, 1, 102, comm_e, MPI_STATUS_IGNORE);
        printf("C3 tid=%ld k=%d value=%d\n", tid, j, v);
    }
    return NULL;
}

int main(int argc, char **argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 4) {
        if (rank == 0) fprintf(stderr, "mpi_seq requires exactly 4 ranks, got %d\n", size);
        MPI_Finalize();
        return 1;
    }
    int v;
    MPI_Status st;

    /* ── A ─────────────────────────────────────────────────────────────── */
    if (rank == 0) {
        int k = 0;
        for (int r = 1; r <= 3; r++) {
            MPI_Recv(&v, 1, MPI_INT, r, 10 + r, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("A k=%d src=%d value=%d\n", k++, r, v);
            MPI_Request q;
            MPI_Irecv(&v, 1, MPI_INT, r, MPI_ANY_TAG, MPI_COMM_WORLD, &q);
            MPI_Wait(&q, &st);
            printf("A k=%d src=%d value=%d\n", k++, r, v);
            MPI_Recv(&v, 1, MPI_INT, r, 10 + r, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("A k=%d src=%d value=%d\n", k++, r, v);
        }
        int w[3];
        MPI_Request q[3];
        for (int i = 0; i < 3; i++) MPI_Irecv(&w[i], 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &q[i]);
        MPI_Waitall(3, q, MPI_STATUSES_IGNORE);
        for (int i = 0; i < 3; i++) printf("A k=%d src=any value=%d\n", k++, w[i]);
    } else {
        for (int i = 0; i < 3; i++) send_world(10 + rank);
        send_world(30 + rank);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    /* ── B ─────────────────────────────────────────────────────────────── */
    MPI_Comm dup, half, cart;
    MPI_Comm_dup(MPI_COMM_WORLD, &dup);
    MPI_Comm_split(MPI_COMM_WORLD, rank / 2, rank, &half);
    int dims[1] = {4}, periods[1] = {0};
    MPI_Cart_create(MPI_COMM_WORLD, 1, dims, periods, 0, &cart);
    MPI_Comm comms[3] = {dup, half, cart};
    for (int c = 0; c < 3; c++) {
        int lr;
        MPI_Comm_rank(comms[c], &lr);
        if (lr == 1) { v = 2000 + c; MPI_Send(&v, 1, MPI_INT, 0, 5, comms[c]); }
        if (lr == 0) MPI_Recv(&v, 1, MPI_INT, 1, 5, comms[c], MPI_STATUS_IGNORE);
    }
    int x = rank, y = 0;
    MPI_Allreduce(&x, &y, 1, MPI_INT, MPI_SUM, dup);
    MPI_Request rq;
    MPI_Ibcast(&x, 1, MPI_INT, 0, half, &rq);
    MPI_Wait(&rq, MPI_STATUS_IGNORE);
    MPI_Barrier(cart);
    int counts[4] = {1, 1, 1, 1}, displs[4] = {0, 1, 2, 3}, all[4];
    MPI_Iallgatherv(&x, 1, MPI_INT, all, counts, displs, MPI_INT, MPI_COMM_WORLD, &rq);
    MPI_Wait(&rq, MPI_STATUS_IGNORE);
    MPI_Iscan(&x, &y, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD, &rq);
    MPI_Wait(&rq, MPI_STATUS_IGNORE);
    MPI_Comm_free(&cart);
    MPI_Comm_free(&half);
    MPI_Comm_free(&dup);

    /* ── C ─────────────────────────────────────────────────────────────── */
    MPI_Comm_dup(MPI_COMM_WORLD, &comm_d);
    MPI_Comm_dup(MPI_COMM_WORLD, &comm_e);
    if (provided == MPI_THREAD_MULTIPLE) {
        if (rank == 1) {
            pthread_t th[2];
            for (long t = 0; t < 2; t++) pthread_create(&th[t], NULL, sender_thread, (void *)t);
            for (int j = 1; j <= 10; j++) { v = 5000 + j; MPI_Send(&v, 1, MPI_INT, 0, 102, comm_e); }
            for (int t = 0; t < 2; t++) pthread_join(th[t], NULL);
        } else if (rank == 0) {
            pthread_t th[2];
            for (long t = 0; t < 2; t++) pthread_create(&th[t], NULL, receiver_thread, (void *)t);
            for (int t = 0; t < 2; t++) pthread_join(th[t], NULL);
        }
    } else if (rank == 0) {
        printf("C skipped provided=%d\n", provided);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Comm_free(&comm_d);
    MPI_Comm_free(&comm_e);

    /* ── D ─────────────────────────────────────────────────────────────── */
    if (rank == 0) {
        for (int k = 0; k < 5; k++) {
            MPI_Message m;
            MPI_Mprobe(2, MPI_ANY_TAG, MPI_COMM_WORLD, &m, &st);
            printf("D mprobe k=%d handle=%ld tag=%d\n", k, (long)m, st.MPI_TAG);
            MPI_Mrecv(&v, 1, MPI_INT, &m, MPI_STATUS_IGNORE);
            printf("D mrecv k=%d value=%d\n", k, v);
        }
        int got = 0;
        while (got < 3) {
            int flag = 0;
            MPI_Message m;
            MPI_Improbe(3, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &m, &st);
            if (!flag) continue;
            MPI_Request q;
            MPI_Imrecv(&v, 1, MPI_INT, &m, &q);
            MPI_Wait(&q, MPI_STATUS_IGNORE);
            printf("D imrecv k=%d value=%d\n", got++, v);
        }
        MPI_Message np;
        MPI_Mprobe(MPI_PROC_NULL, 0, MPI_COMM_WORLD, &np, &st);
        printf("D noproc=%d\n", np == MPI_MESSAGE_NO_PROC);
        MPI_Mrecv(&v, 1, MPI_INT, &np, MPI_STATUS_IGNORE);
        MPI_Message hidden;
        PMPI_Mprobe(1, 60, MPI_COMM_WORLD, &hidden, &st);    /* not seen by the hook */
        MPI_Mrecv(&v, 1, MPI_INT, &hidden, MPI_STATUS_IGNORE);
        printf("D hidden value=%d\n", v);
    } else if (rank == 2) {
        for (int i = 0; i < 5; i++) send_world(40 + i);
    } else if (rank == 3) {
        for (int i = 0; i < 3; i++) send_world(50 + i);
    } else if (rank == 1) {
        send_world(60);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    /* ── E ─────────────────────────────────────────────────────────────── */
#if MPI_VERSION >= 4
    MPI_Request pr;
    MPI_Allreduce_init(&x, &y, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD, MPI_INFO_NULL, &pr);
    for (int i = 0; i < 2; i++) {
        MPI_Start(&pr);
        MPI_Wait(&pr, MPI_STATUS_IGNORE);
    }
    MPI_Request_free(&pr);
    if (rank == 0) printf("E persistent y=%d\n", y);
#endif
    MPI_Finalize();
    return 0;
}
