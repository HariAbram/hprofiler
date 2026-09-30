/* Ground-truth MPI program for tests/integration/test_profiling_accuracy.py.
 * Single rank talking to itself (this dev machine cannot launch real
 * multi-rank jobs -- see project notes): Barrier, Irecv/Isend + Waitall,
 * a wildcard Irecv resolved by MPI_Wait, Allreduce. Logs its own
 * CLOCK_MONOTONIC stamps around each call to $TRUTH_OUT. */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
static inline uint64_t now_ns(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return (uint64_t)t.tv_sec*1000000000ull+t.tv_nsec;}
static void spin_ns(uint64_t d){uint64_t e=now_ns()+d; volatile double x=1; while(now_ns()<e){x*=1.0000001;}}
#define N 8
#define COUNT (64*1024)
int main(int argc,char**argv){
  MPI_Init(&argc,&argv);
  int rank; MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  const char *out=getenv("TRUTH_OUT"); FILE *f=out?fopen(out,"w"):stdout;
  double *sb=malloc(COUNT*sizeof(double)), *rb=malloc(COUNT*sizeof(double));
  for(int i=0;i<COUNT;i++) sb[i]=i;
  for(int it=0;it<N;it++){
    uint64_t a=now_ns(); MPI_Barrier(MPI_COMM_WORLD); uint64_t b=now_ns();
    fprintf(f,"barrier %d %llu %llu\n",it,(unsigned long long)a,(unsigned long long)b);
    spin_ns(2000000);
    MPI_Request rq[2];
    MPI_Irecv(rb,COUNT,MPI_DOUBLE,rank,it,MPI_COMM_WORLD,&rq[0]);
    MPI_Isend(sb,COUNT,MPI_DOUBLE,rank,it,MPI_COMM_WORLD,&rq[1]);
    spin_ns(1000000);
    a=now_ns(); MPI_Waitall(2,rq,MPI_STATUSES_IGNORE); b=now_ns();
    fprintf(f,"waitall %d %llu %llu\n",it,(unsigned long long)a,(unsigned long long)b);
    /* wildcard receive completed by MPI_Wait */
    MPI_Request r2,s2; MPI_Status st;
    MPI_Irecv(rb,16,MPI_DOUBLE,MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,&r2);
    MPI_Isend(sb,16,MPI_DOUBLE,rank,100+it,MPI_COMM_WORLD,&s2);
    MPI_Wait(&s2,MPI_STATUS_IGNORE);
    a=now_ns(); MPI_Wait(&r2,&st); b=now_ns();
    fprintf(f,"wildwait %d %llu %llu tag=%d src=%d\n",it,(unsigned long long)a,(unsigned long long)b,st.MPI_TAG,st.MPI_SOURCE);
    double x=it,y=0;
    a=now_ns(); MPI_Allreduce(&x,&y,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD); b=now_ns();
    fprintf(f,"allreduce %d %llu %llu\n",it,(unsigned long long)a,(unsigned long long)b);
    spin_ns(500000);
  }
  if(f!=stdout) fclose(f);
  MPI_Finalize();
  return 0;
}
