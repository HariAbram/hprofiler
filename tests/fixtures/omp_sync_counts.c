/* OpenMP tasks, locks, critical and ordered with known counts
 * (tests/integration/test_openmp_sync.py), for both runtimes: built with gcc
 * (GNU libgomp: gomp_hook.c's GOMP_task trampoline and omp_*_lock wrappers)
 * and with clang (LLVM libomp: ompt_tool.c's mutex/lock/dependence
 * callbacks). The printed results must equal an unprofiled run -- in
 * particular every task must see its own copy of its firstprivate array.
 * 4 threads:
 *   tasks       16 deferred (firstprivate array) + taskwait, 8 in a taskgroup,
 *               1 undeferred (if(0)), 2 with depend(out/in)        = 27 creates
 *   locks       5 x set/unset per thread, 1 nested pair, 1 test_lock loop
 *   critical    10 per thread
 *   ordered     a 16-iteration ordered loop
 */
#include <omp.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

int main(void) {
    long results[32];
    memset(results, 0, sizeof(results));
    long counter = 0, ncounter = 0, crit = 0, dep = 0;   /* ncounter: under the nest lock only */
    int order[16], next = 0;
    omp_lock_t lk;
    omp_nest_lock_t nl;
    omp_init_lock(&lk);
    omp_init_nest_lock(&nl);
    omp_set_dynamic(0);
    #pragma omp parallel num_threads(4)
    {
        #pragma omp single
        {
            for (int i = 0; i < 16; i++) {
                int big[32];
                for (int k = 0; k < 32; k++) big[k] = i * k;
                #pragma omp task firstprivate(big, i) shared(results)
                {
                    long s = 0;
                    for (int k = 0; k < 32; k++) s += big[k];
                    results[i] = s;
                }
            }
            #pragma omp taskwait
            #pragma omp taskgroup
            {
                for (int i = 16; i < 24; i++) {
                    #pragma omp task firstprivate(i) shared(results)
                    results[i] = i * 3;
                }
            }
            #pragma omp task if(0) shared(results)
            results[24] = 99;
            /* slow, so it is still pending when its dependent is created:
             * the runtime records the dependence (deterministically) */
            #pragma omp task depend(out: dep) shared(dep)
            { usleep(20000); dep = 7; }
            #pragma omp task depend(in: dep) shared(dep, results)
            results[25] = dep * 2;
            #pragma omp taskwait
        }
        for (int r = 0; r < 5; r++) {
            omp_set_lock(&lk);
            counter++;
            omp_unset_lock(&lk);
        }
        omp_set_nest_lock(&nl);
        omp_set_nest_lock(&nl);
        ncounter++;
        omp_unset_nest_lock(&nl);
        omp_unset_nest_lock(&nl);
        while (!omp_test_lock(&lk)) { }
        counter++;
        omp_unset_lock(&lk);
        for (int r = 0; r < 10; r++) {
            #pragma omp critical
            crit++;
        }
    }
    #pragma omp parallel for ordered schedule(static, 1) num_threads(4)
    for (int i = 0; i < 16; i++) {
        #pragma omp ordered
        order[next++] = i;
    }
    omp_destroy_lock(&lk);
    omp_destroy_nest_lock(&nl);
    for (int i = 0; i < 26; i++) printf("R %d %ld\n", i, results[i]);
    printf("C counter=%ld crit=%ld\n", counter + ncounter, crit);
    for (int i = 0; i < 16; i++) printf("O %d %d\n", i, order[i]);
    return 0;
}
