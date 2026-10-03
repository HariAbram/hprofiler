/*
 * Call-site attribution fixture for tests/integration/test_callsite_e2e.py.
 * Built twice: with clang -fopenmp (LLVM libomp -> the OMPT tool) and with
 * gcc -fopenmp (GNU libgomp -> gomp_hook.c). Each construct lives in a
 * named, non-inlined function (build with -rdynamic) so the hooks' sym=/
 * symfile= tags name a real symbol of this executable, which the test then
 * expects to find disassembled in the saved trace -- including when the
 * program is started through a launcher (`env`), where command[0] is not
 * the profiled binary.
 */
#include <omp.h>
#include <stdio.h>
#include <unistd.h>

static volatile long g_sum;

__attribute__((noinline)) void region_with_sync(void) {
    #pragma omp parallel num_threads(4)
    {
        #pragma omp for schedule(static)
        for (int i = 0; i < 4000; i++)
            g_sum += 0;

        #pragma omp barrier

        #pragma omp critical
        {
            g_sum += omp_get_thread_num();
            usleep(500);
        }
    }
}

int main(void) {
    region_with_sync();
    printf("omp_callsite: sum=%ld\n", (long)g_sum);
    return 0;
}
