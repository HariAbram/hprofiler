/*
 * MPI profiling layer hook (PMPI interface).
 *
 * Intercepts common MPI calls by providing PMPI_* wrappers that the MPI
 * standard mandates every conforming implementation must expose.  No
 * dlopen or LD_PRELOAD tricks needed — just link this file alongside the
 * program (via MPI_PRELOAD or mpicc -L...).
 *
 * Captured calls:
 *   Point-to-point : MPI_Send, MPI_Recv, MPI_Isend, MPI_Irecv, MPI_Wait,
 *                    MPI_Waitall, MPI_Ssend, MPI_Bsend
 *   Collectives    : MPI_Bcast, MPI_Reduce, MPI_Allreduce, MPI_Alltoall,
 *                    MPI_Allgather, MPI_Scatter, MPI_Gather, MPI_Barrier,
 *                    MPI_Scan, MPI_Exscan
 *   One-sided      : MPI_Put, MPI_Get, MPI_Accumulate
 *   Lifecycle      : MPI_Init, MPI_Init_thread, MPI_Finalize
 *
 * Wire protocol: identical to other hooks — newline-terminated ASCII to
 * HPROFILER_SOCKET.
 *
 * Build requirements:
 *   mpicc -shared -fPIC -o libhprofiler_mpi.so mpi_hook.c -ldl -lpthread
 * Or via CMake (see CMakeLists.txt in this directory).
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <time.h>
#include <pthread.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/syscall.h>
#include <mpi.h>

/* ── Globals ─────────────────────────────────────────────────────────── */
static int             g_sock       = -1;
static pthread_mutex_t g_sock_mutex = PTHREAD_MUTEX_INITIALIZER;
static pid_t           g_pid        = 0;
static int             g_mpi_rank   = -1;

/* ── Helpers ─────────────────────────────────────────────────────────── */
static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

static pid_t gettid_compat(void) { return (pid_t)syscall(SYS_gettid); }

static void ensure_connected(void) {
    if (g_sock >= 0) return;
    const char *path = getenv("HPROFILER_SOCKET");
    if (!path) return;
    int s = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (s < 0) return;
    struct sockaddr_un addr = {0};
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, path, sizeof(addr.sun_path) - 1);
    if (connect(s, (struct sockaddr*)&addr, sizeof(addr)) == 0) {
        g_sock = s;
        g_pid  = getpid();
    } else {
        close(s);
    }
}

static void send_all(const char *buf, int n) {
    while (n > 0) {
        ssize_t r = send(g_sock, buf, (size_t)n, MSG_NOSIGNAL);
        if (r < 0) { close(g_sock); g_sock = -1; return; }
        buf += r; n -= (int)r;
    }
}

static void emit_span(const char *cat, uint64_t start_ns, uint64_t dur_ns,
                      const char *name, const char *extra) {
    pthread_mutex_lock(&g_sock_mutex);
    ensure_connected();
    if (g_sock >= 0) {
        char buf[1024];
        int n = snprintf(buf, sizeof(buf),
            "span:%s:%d:%d:%llu:%llu:%s:%s\n",
            cat, g_pid, (int)gettid_compat(),
            (unsigned long long)start_ns, (unsigned long long)dur_ns,
            name, extra ? extra : "");
        if (n > 0 && n < (int)sizeof(buf)) send_all(buf, n);
    }
    pthread_mutex_unlock(&g_sock_mutex);
}

/* MPI datatype → byte size (only common types; 0 means unknown). */
static size_t dtype_size(MPI_Datatype t) {
    if (t == MPI_BYTE || t == MPI_CHAR || t == MPI_UNSIGNED_CHAR) return 1;
    if (t == MPI_SHORT || t == MPI_UNSIGNED_SHORT)                 return 2;
    if (t == MPI_INT   || t == MPI_UNSIGNED || t == MPI_FLOAT)     return 4;
    if (t == MPI_LONG  || t == MPI_UNSIGNED_LONG ||
        t == MPI_DOUBLE || t == MPI_LONG_LONG)                     return 8;
    if (t == MPI_LONG_DOUBLE)                                      return 16;
    int sz = 0;
    PMPI_Type_size(t, &sz);
    return (size_t)(sz > 0 ? sz : 0);
}

/* ── MPI lifecycle ──────────────────────────────────────────────────── */

int MPI_Init(int *argc, char ***argv) {
    int ret = PMPI_Init(argc, argv);
    if (ret == MPI_SUCCESS) {
        PMPI_Comm_rank(MPI_COMM_WORLD, &g_mpi_rank);
        ensure_connected();
    }
    return ret;
}

int MPI_Init_thread(int *argc, char ***argv, int required, int *provided) {
    int ret = PMPI_Init_thread(argc, argv, required, provided);
    if (ret == MPI_SUCCESS) {
        PMPI_Comm_rank(MPI_COMM_WORLD, &g_mpi_rank);
        ensure_connected();
    }
    return ret;
}

int MPI_Finalize(void) {
    if (g_sock >= 0) { close(g_sock); g_sock = -1; }
    return PMPI_Finalize();
}

/* ── Point-to-point ─────────────────────────────────────────────────── */

#define _P2P(FNAME, PMPI_CALL, TYPE_STR, ...)                          \
int FNAME(__VA_ARGS__) {                                                \
    char extra[128];                                                    \
    size_t nb = (size_t)count * dtype_size(datatype);                  \
    snprintf(extra, sizeof(extra),                                      \
             "type=%s,bytes=%zu,rank=%d,peer=%d,tag=%d",               \
             TYPE_STR, nb, g_mpi_rank, peer_rank, tag);                \
    uint64_t t0 = now_ns();                                             \
    int ret = PMPI_CALL;                                                \
    emit_span("mpi", t0, now_ns()-t0, #FNAME, extra);                  \
    return ret;                                                         \
}

_P2P(MPI_Send,
     PMPI_Send(buf, count, datatype, dest, tag, comm),
     "send",
     const void *buf, int count, MPI_Datatype datatype,
     int dest, int tag, MPI_Comm comm)
#define peer_rank dest

_P2P(MPI_Ssend,
     PMPI_Ssend(buf, count, datatype, dest, tag, comm),
     "ssend",
     const void *buf, int count, MPI_Datatype datatype,
     int dest, int tag, MPI_Comm comm)

_P2P(MPI_Bsend,
     PMPI_Bsend(buf, count, datatype, dest, tag, comm),
     "bsend",
     const void *buf, int count, MPI_Datatype datatype,
     int dest, int tag, MPI_Comm comm)

#undef peer_rank

int MPI_Recv(void *buf, int count, MPI_Datatype datatype,
             int source, int tag, MPI_Comm comm, MPI_Status *status) {
    size_t nb = (size_t)count * dtype_size(datatype);
    char extra[128];
    snprintf(extra, sizeof(extra),
             "type=recv,bytes=%zu,rank=%d,peer=%d,tag=%d",
             nb, g_mpi_rank, source, tag);
    uint64_t t0 = now_ns();
    int ret = PMPI_Recv(buf, count, datatype, source, tag, comm, status);
    emit_span("mpi", t0, now_ns()-t0, "MPI_Recv", extra);
    return ret;
}

int MPI_Isend(const void *buf, int count, MPI_Datatype datatype,
              int dest, int tag, MPI_Comm comm, MPI_Request *request) {
    size_t nb = (size_t)count * dtype_size(datatype);
    char extra[128];
    snprintf(extra, sizeof(extra),
             "type=isend,bytes=%zu,rank=%d,peer=%d,tag=%d",
             nb, g_mpi_rank, dest, tag);
    uint64_t t0 = now_ns();
    int ret = PMPI_Isend(buf, count, datatype, dest, tag, comm, request);
    emit_span("mpi", t0, now_ns()-t0, "MPI_Isend", extra);
    return ret;
}

int MPI_Irecv(void *buf, int count, MPI_Datatype datatype,
              int source, int tag, MPI_Comm comm, MPI_Request *request) {
    size_t nb = (size_t)count * dtype_size(datatype);
    char extra[128];
    snprintf(extra, sizeof(extra),
             "type=irecv,bytes=%zu,rank=%d,peer=%d,tag=%d",
             nb, g_mpi_rank, source, tag);
    uint64_t t0 = now_ns();
    int ret = PMPI_Irecv(buf, count, datatype, source, tag, comm, request);
    emit_span("mpi", t0, now_ns()-t0, "MPI_Irecv", extra);
    return ret;
}

int MPI_Wait(MPI_Request *request, MPI_Status *status) {
    uint64_t t0 = now_ns();
    int ret = PMPI_Wait(request, status);
    emit_span("mpi", t0, now_ns()-t0, "MPI_Wait", "type=wait");
    return ret;
}

int MPI_Waitall(int count, MPI_Request requests[], MPI_Status statuses[]) {
    uint64_t t0 = now_ns();
    int ret = PMPI_Waitall(count, requests, statuses);
    char extra[64]; snprintf(extra, sizeof(extra), "type=waitall,count=%d", count);
    emit_span("mpi", t0, now_ns()-t0, "MPI_Waitall", extra);
    return ret;
}

/* ── Collectives ────────────────────────────────────────────────────── */

#define _COLL(FNAME, PMPI_CALL, TYPE_STR, BYTES_EXPR, ...)             \
int FNAME(__VA_ARGS__) {                                                \
    char extra[128];                                                    \
    size_t nb = (BYTES_EXPR);                                           \
    snprintf(extra, sizeof(extra),                                      \
             "type=%s,bytes=%zu,rank=%d", TYPE_STR, nb, g_mpi_rank);   \
    uint64_t t0 = now_ns();                                             \
    int ret = PMPI_CALL;                                                \
    emit_span("mpi", t0, now_ns()-t0, #FNAME, extra);                  \
    return ret;                                                         \
}

_COLL(MPI_Bcast,
      PMPI_Bcast(buffer, count, datatype, root, comm),
      "bcast", (size_t)count * dtype_size(datatype),
      void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm)

_COLL(MPI_Reduce,
      PMPI_Reduce(sendbuf, recvbuf, count, datatype, op, root, comm),
      "reduce", (size_t)count * dtype_size(datatype),
      const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
      MPI_Op op, int root, MPI_Comm comm)

_COLL(MPI_Allreduce,
      PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm),
      "allreduce", (size_t)count * dtype_size(datatype),
      const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
      MPI_Op op, MPI_Comm comm)

_COLL(MPI_Alltoall,
      PMPI_Alltoall(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm),
      "alltoall", (size_t)sendcount * dtype_size(sendtype),
      const void *sendbuf, int sendcount, MPI_Datatype sendtype,
      void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm)

_COLL(MPI_Allgather,
      PMPI_Allgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm),
      "allgather", (size_t)sendcount * dtype_size(sendtype),
      const void *sendbuf, int sendcount, MPI_Datatype sendtype,
      void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm)

_COLL(MPI_Scatter,
      PMPI_Scatter(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm),
      "scatter", (size_t)sendcount * dtype_size(sendtype),
      const void *sendbuf, int sendcount, MPI_Datatype sendtype,
      void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm)

_COLL(MPI_Gather,
      PMPI_Gather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm),
      "gather", (size_t)sendcount * dtype_size(sendtype),
      const void *sendbuf, int sendcount, MPI_Datatype sendtype,
      void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm)

int MPI_Barrier(MPI_Comm comm) {
    uint64_t t0 = now_ns();
    int ret = PMPI_Barrier(comm);
    emit_span("mpi", t0, now_ns()-t0, "MPI_Barrier", "type=barrier");
    return ret;
}

_COLL(MPI_Scan,
      PMPI_Scan(sendbuf, recvbuf, count, datatype, op, comm),
      "scan", (size_t)count * dtype_size(datatype),
      const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
      MPI_Op op, MPI_Comm comm)

/* ── One-sided ──────────────────────────────────────────────────────── */

int MPI_Put(const void *origin_addr, int origin_count, MPI_Datatype origin_datatype,
            int target_rank, MPI_Aint target_disp, int target_count,
            MPI_Datatype target_datatype, MPI_Win win) {
    size_t nb = (size_t)origin_count * dtype_size(origin_datatype);
    char extra[128];
    snprintf(extra, sizeof(extra), "type=put,bytes=%zu,rank=%d,peer=%d",
             nb, g_mpi_rank, target_rank);
    uint64_t t0 = now_ns();
    int ret = PMPI_Put(origin_addr, origin_count, origin_datatype,
                       target_rank, target_disp, target_count, target_datatype, win);
    emit_span("mpi", t0, now_ns()-t0, "MPI_Put", extra);
    return ret;
}

int MPI_Get(void *origin_addr, int origin_count, MPI_Datatype origin_datatype,
            int target_rank, MPI_Aint target_disp, int target_count,
            MPI_Datatype target_datatype, MPI_Win win) {
    size_t nb = (size_t)origin_count * dtype_size(origin_datatype);
    char extra[128];
    snprintf(extra, sizeof(extra), "type=get,bytes=%zu,rank=%d,peer=%d",
             nb, g_mpi_rank, target_rank);
    uint64_t t0 = now_ns();
    int ret = PMPI_Get(origin_addr, origin_count, origin_datatype,
                       target_rank, target_disp, target_count, target_datatype, win);
    emit_span("mpi", t0, now_ns()-t0, "MPI_Get", extra);
    return ret;
}
