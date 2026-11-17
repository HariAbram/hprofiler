/*
 * GNU libgomp task dependences, parent tasks, taskwait/taskgroup membership
 * and lock life cycles, for tests/integration/test_gomp_deps.py. Built with
 * gcc -fopenmp (libgomp). One thread (in a single region of 4) creates the
 * tasks below in this order; the expected dependence pairs are listed by
 * creation index (T0 is the first task created):
 *
 *   T0  out(a)                         -
 *   T1  in(a)                          T0 -> T1
 *   T2  in(a)                          T0 -> T2
 *   T3  inout(a)                       T1, T2 -> T3
 *   T4  mutexinoutset(b)               -
 *   T5  mutexinoutset(b)               T4 -> T5 (libgomp orders them)
 *   T6  inoutset(c)                    -
 *   T7  inoutset(c)                    - (same set: unordered)
 *   T8  in(c)                          T6, T7 -> T8
 *   T9  depobj(o) = inout(d)           -
 *   T10 in(d)                          T9 -> T10
 *   T11 out(omp_all_memory)            T3, T5, T8, T10 -> T11
 *   T12 in(a)                          T11 -> T12
 *   taskwait depend(in: a)             waits for T11 (latest writer) only
 *   T13 (taskgroup) creates T14        taskgroup waits for T13 and T14
 *   taskwait                           waits for T0..T13 (children of the
 *                                      single's implicit task)
 *
 * Then a lock initialised, used, destroyed, initialised again at the same
 * address and used: two lock tokens. Each task sleeps briefly so other
 * threads run some of them (a task executed by another thread than its
 * creator; T14 is created inside T13).
 */
#include <omp.h>
#include <stdio.h>
#include <unistd.h>

int a, b, c, d;

static void work(void) { usleep(2000); }

int main(void) {
    omp_depend_t o;
    #pragma omp depobj(o) depend(inout: d)
    #pragma omp parallel num_threads(4)
    #pragma omp single
    {
        #pragma omp task depend(out: a)
        { work(); a = 1; }
        #pragma omp task depend(in: a)
        { work(); }
        #pragma omp task depend(in: a)
        { work(); }
        #pragma omp task depend(inout: a)
        { work(); a++; }
        #pragma omp task depend(mutexinoutset: b)
        { work(); b++; }
        #pragma omp task depend(mutexinoutset: b)
        { work(); b++; }
        #pragma omp task depend(inoutset: c)
        { work(); }
        #pragma omp task depend(inoutset: c)
        { work(); }
        #pragma omp task depend(in: c)
        { work(); }
        #pragma omp task depend(depobj: o)
        { work(); d++; }
        #pragma omp task depend(in: d)
        { work(); }
        #pragma omp task depend(out: omp_all_memory)
        { work(); }
        #pragma omp task depend(in: a)
        { work(); }
        #pragma omp taskwait depend(in: a)
        #pragma omp taskgroup
        {
            #pragma omp task
            {
                work();
                #pragma omp task
                { work(); }
            }
        }
        #pragma omp taskwait
    }
    #pragma omp depobj(o) destroy
    omp_lock_t l;
    for (int round = 0; round < 2; round++) {
        omp_init_lock(&l);
        #pragma omp parallel num_threads(2)
        {
            omp_set_lock(&l);
            work();
            omp_unset_lock(&l);
        }
        omp_destroy_lock(&l);
    }
    printf("a=%d b=%d d=%d\n", a, b, d);
    return 0;
}
