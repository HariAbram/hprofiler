/* MPI_Init / MPI_Finalize as spans, and the process-begin markers, for
 * tests/integration/test_mpi_lifecycle.py. Each rank works 20 ms x rank
 * before MPI_Init (thread=1 in argv[1]: MPI_Init_thread) and 20 ms x
 * (size - 1 - rank) before MPI_Finalize; rank 0 works 30 ms right after
 * MPI_Init, before its first message. Prints when each rank's MPI_Init and
 * MPI_Finalize returned (CLOCK_MONOTONIC, ms): equal return times mean the
 * library synchronized the ranks there. */
#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec * 1e-6;
}
static void spin_ms(double ms) {
    double end = now_ms() + ms;
    while (now_ms() < end) { /* busy */ }
}
static int rank_from_env(void) {
    /* the rank before MPI_Init: from the launcher's environment */
    const char *names[] = {"OMPI_COMM_WORLD_RANK", "PMI_RANK", "PMIX_RANK", "MPI_LOCALRANKID"};
    for (unsigned i = 0; i < sizeof(names) / sizeof(names[0]); i++) {
        extern char **environ;
        size_t n = strlen(names[i]);
        for (char **e = environ; *e; e++)
            if (!strncmp(*e, names[i], n) && (*e)[n] == '=') return (int)(*e)[n + 1] - '0';
    }
    return 0;
}

int main(int argc, char **argv) {
    int pre = rank_from_env();
    spin_ms(20.0 * pre);
    double t_in = now_ms();
    if (argc > 1 && !strcmp(argv[1], "thread=1")) {
        int provided;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    } else {
        MPI_Init(&argc, &argv);
    }
    double t_init = now_ms();
    int rank, size, token = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 4) {
        if (rank == 0) fprintf(stderr, "mpi_lifecycle requires exactly 4 ranks, got %d\n", size);
        MPI_Finalize();
        return 1;
    }
    if (rank == 0) {
        spin_ms(30);
        MPI_Send(&token, 1, MPI_INT, 1 % size, 0, MPI_COMM_WORLD);
    } else if (rank == 1) {
        MPI_Recv(&token, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    spin_ms(20.0 * (size - 1 - rank));
    double t_fin_in = now_ms();
    MPI_Finalize();
    double t_fin = now_ms();
    printf("rank=%d pre=%d init_in=%.3f init_out=%.3f fin_in=%.3f fin_out=%.3f\n", rank, pre, t_in, t_init,
           t_fin_in, t_fin);
    return 0;
}
