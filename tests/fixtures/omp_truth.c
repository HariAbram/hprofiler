/* Ground-truth OpenMP program for tests/integration/test_profiling_accuracy.py.
 * 4 threads x 6 regions; thread t spins (t+1)*4ms then waits at an explicit
 * barrier, so compute and barrier-wait per thread are known exactly. Logs its
 * own CLOCK_MONOTONIC stamps (same clock as the hooks) to $TRUTH_OUT so the
 * profiler's timestamps can be compared per event. Build with clang
 * -fopenmp=libomp (OMPT path) and gcc -fopenmp (GOMP hook path). */
#define _GNU_SOURCE
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>
#include <sys/syscall.h>
static inline uint64_t now_ns(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return (uint64_t)t.tv_sec*1000000000ull+t.tv_nsec;}
static void spin_ns(uint64_t d){uint64_t e=now_ns()+d; volatile double x=1; while(now_ns()<e){x*=1.0000001;}}
#define NT 4
#define NREG 6
#define UNIT_NS 4000000ull
int main(void){
  const char *out = getenv("TRUTH_OUT");
  FILE *f = out ? fopen(out,"w") : stdout;
  static uint64_t spin_s[NREG][NT], spin_e[NREG][NT], bar_a[NREG][NT], bar_l[NREG][NT];
  static int tids[NREG][NT];
  uint64_t reg_s[NREG], reg_e[NREG];
  omp_set_num_threads(NT);
  for(int r=0;r<NREG;r++){
    reg_s[r]=now_ns();
    #pragma omp parallel num_threads(NT)
    {
      int t=omp_get_thread_num();
      tids[r][t]=(int)syscall(SYS_gettid);
      spin_s[r][t]=now_ns();
      spin_ns((uint64_t)(t+1)*UNIT_NS);
      spin_e[r][t]=now_ns();
      bar_a[r][t]=now_ns();
      #pragma omp barrier
      bar_l[r][t]=now_ns();
    }
    reg_e[r]=now_ns();
    usleep(2000);
  }
  for(int r=0;r<NREG;r++){
    fprintf(f,"region %d %llu %llu\n",r,(unsigned long long)reg_s[r],(unsigned long long)reg_e[r]);
    for(int t=0;t<NT;t++){
      fprintf(f,"spin %d %d %d %llu %llu\n",r,t,tids[r][t],(unsigned long long)spin_s[r][t],(unsigned long long)spin_e[r][t]);
      fprintf(f,"barrier %d %d %d %llu %llu\n",r,t,tids[r][t],(unsigned long long)bar_a[r][t],(unsigned long long)bar_l[r][t]);
    }
  }
  if(f!=stdout) fclose(f);
  return 0;
}
