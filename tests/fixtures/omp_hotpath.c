/* Hot-path microbenchmark (DOCUMENTATION.md, "Measured hot-path overhead"):
 * every thread hits N barriers and N critical sections (each an
 * intercepted GOMP_* call -> one event). Prints the wall time of the timed
 * region only.
 *   gcc -O2 -fopenmp -o omp_hotpath tests/fixtures/omp_hotpath.c
 *   OMP_NUM_THREADS=8 ./omp_hotpath 20000
 *   OMP_NUM_THREADS=8 python3 hprofiler run --backend openmp --no-ui --no-json \
 *       -o hot.hpstore -- ./omp_hotpath 20000
 *   (add HPROFILER_TRANSPORT=sync for the synchronous transport) */
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
int main(int argc, char **argv) {
    int n = argc > 1 ? atoi(argv[1]) : 20000;
    volatile long acc = 0;
    double t0 = now();
    #pragma omp parallel
    {
        for (int i = 0; i < n; i++) {
            #pragma omp critical
            acc += i;
            #pragma omp barrier
        }
    }
    double t1 = now();
    printf("threads=%d iters=%d elapsed_s=%.4f acc=%ld\n", omp_get_max_threads(), n, t1 - t0, acc);
    return 0;
}
