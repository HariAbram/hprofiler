/* Exercises mpi_hook.c's one-sided (RMA) synchronization wrappers
 * (MPI_Win_fence/flush/flush_all/lock/lock_all/unlock/unlock_all) against
 * real MPI RMA machinery. Uses MPI_COMM_SELF (one process communicating
 * with itself) rather than a real multi-rank MPI_COMM_WORLD, same reason
 * mpi_proto_self.c does: this dev machine's MPICH/Hydra can't form a real
 * multi-rank world (see tests/integration/test_mpi_protocol.py's header
 * comment) -- this still drives the real PMPI_Win_* calls end to end,
 * which is all that's needed to verify the HOOK's own wrapper logic
 * (correct span name/category/tags), not MPI's own RMA semantics. */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int buf[4] = {1, 2, 3, 4};
    int origin[4] = {10, 20, 30, 40};
    int result[4] = {0, 0, 0, 0};
    MPI_Win win;
    MPI_Win_create(buf, sizeof(buf), sizeof(int), MPI_INFO_NULL, MPI_COMM_SELF, &win);

    /* Active-target (fence-based) epoch: Put to self, fence, Get back. */
    MPI_Win_fence(0, win);
    MPI_Put(origin, 4, MPI_INT, 0, 0, 4, MPI_INT, win);
    MPI_Win_fence(0, win);
    MPI_Get(result, 4, MPI_INT, 0, 0, 4, MPI_INT, win);
    MPI_Win_fence(0, win);

    /* Passive-target (lock-based) epoch. */
    MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win);
    MPI_Accumulate(origin, 4, MPI_INT, 0, 0, 4, MPI_INT, MPI_SUM, win);
    MPI_Win_flush(0, win);
    MPI_Win_unlock(0, win);

    /* Passive-target "all" variants. */
    MPI_Win_lock_all(0, win);
    MPI_Win_flush_all(win);
    MPI_Win_unlock_all(win);

    MPI_Win_free(&win);
    printf("mpi_win_self: result=%d,%d,%d,%d\n", result[0], result[1], result[2], result[3]);
    MPI_Finalize();
    return 0;
}
