#include <mpi.h>
#include <stdio.h>
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double x = rank + 1.0, sum = 0.0;
    MPI_Allreduce(&x, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    printf("mpi_mini: rank=%d sum=%f\n", rank, sum);
    MPI_Finalize();
    return 0;
}
