/* Two concurrent nested teams under the GNU libgomp hook
 * (tests/integration/test_gomp_teams.py): an outer team of 2 threads each
 * runs an inner team of 2 that meets at an explicit barrier. The two inner
 * barriers overlap in time but are separate rendezvous; the hook's team=
 * tag must keep them apart. Compile with gcc (libgomp). */
#include <omp.h>
#include <stdio.h>
#include <unistd.h>

int main(void) {
    omp_set_max_active_levels(2);
    #pragma omp parallel num_threads(2)
    {
        int outer = omp_get_thread_num();
        #pragma omp parallel num_threads(2)
        {
            int inner = omp_get_thread_num();
            usleep(20000 * (1 + inner + 2 * outer));
            #pragma omp barrier
            usleep(5000);
        }
    }
    printf("gomp_teams: done\n");
    return 0;
}
