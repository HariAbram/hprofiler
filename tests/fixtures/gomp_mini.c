/*
 * Exercises the GNU libgomp constructs hooks/gomp_hook/gomp_hook.c
 * intercepts: parallel region (per-thread work), a static-scheduled
 * work-sharing for loop, an explicit barrier, an anonymous critical
 * section, a named critical section, and a single region. Must be
 * compiled with gcc (not clang) so it actually links against libgomp,
 * not libomp -- the whole point of this fixture.
 */
#include <omp.h>
#include <stdio.h>
#include <unistd.h>

int main(void) {
    long sum = 0;
    int single_ran_on = -1;

    #pragma omp parallel num_threads(4)
    {
        int tid = omp_get_thread_num();

        #pragma omp for schedule(static)
        for (int i = 0; i < 1000; i++) {
            /* trivial work-sharing body */
            (void)i;
        }

        #pragma omp barrier

        #pragma omp critical
        {
            sum += tid;
            usleep(1000);
        }

        #pragma omp critical(named_section)
        {
            usleep(500);
        }

        #pragma omp single
        {
            single_ran_on = tid;
        }
    }

    printf("gomp_mini: sum=%ld single_ran_on_thread=%d\n", sum, single_ran_on);
    return 0;
}
