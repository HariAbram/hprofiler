/* Explicit tasks with known counts (tests/integration/test_openmp_sync.py,
 * TestTaskCounts), built with clang (LLVM libomp: ompt_tool.c's task
 * callbacks) and gcc (GNU libgomp: gomp_hook.c's GOMP_task trampoline).
 * 4 threads, one single region creating
 *   20 leaf tasks, then a taskwait,
 *    1 parent task that creates 1 nested task and waits for it (a task
 *      switch at its taskwait when its own thread runs the child),
 *    1 task that calls taskyield 3 times while 4 other tasks are pending,
 *    4 tasks for it to yield to,
 * then a taskwait: 27 tasks created, 27 executed.
 */
#include <omp.h>
#include <stdio.h>
#include <unistd.h>

#define N_LEAF 20

int main(void) {
    long done[32] = {0};
    omp_set_dynamic(0);
    #pragma omp parallel num_threads(4)
    {
        #pragma omp single
        {
            for (int i = 0; i < N_LEAF; i++) {
                #pragma omp task firstprivate(i) shared(done)
                { usleep(200); done[i] = 1; }
            }
            #pragma omp taskwait
            #pragma omp task shared(done)
            {
                #pragma omp task shared(done)
                { usleep(2000); done[21] = 1; }
                #pragma omp taskwait
                done[20] = 1;
            }
            #pragma omp task shared(done)
            {
                for (int y = 0; y < 3; y++) {
                    #pragma omp taskyield
                    usleep(500);
                }
                done[22] = 1;
            }
            for (int i = 23; i < 27; i++) {
                #pragma omp task firstprivate(i) shared(done)
                { usleep(1000); done[i] = 1; }
            }
            #pragma omp taskwait
        }
    }
    int n = 0;
    for (int i = 0; i < 27; i++) n += (int)done[i];
    printf("T tasks_done=%d\n", n);
    return n == 27 ? 0 : 1;
}
