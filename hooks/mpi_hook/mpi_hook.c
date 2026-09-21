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
 *                    MPI_Waitall, MPI_Waitany, MPI_Waitsome,
 *                    MPI_Test, MPI_Testany, MPI_Testsome, MPI_Testall,
 *                    MPI_Cancel, MPI_Ssend, MPI_Bsend
 *   Non-blocking   : MPI_Ibcast, MPI_Iallreduce, MPI_Ireduce, MPI_Iallgather,
 *                    MPI_Ialltoall, MPI_Iscatter, MPI_Igather
 *   Persistent     : MPI_Send_init, MPI_Recv_init, MPI_Start, MPI_Startall
 *   Collectives    : MPI_Bcast, MPI_Reduce, MPI_Allreduce, MPI_Alltoall,
 *                    MPI_Allgather, MPI_Scatter, MPI_Gather, MPI_Barrier,
 *                    MPI_Scan, MPI_Exscan
 *   One-sided      : MPI_Put, MPI_Get, MPI_Accumulate
 *   Lifecycle      : MPI_Init, MPI_Init_thread, MPI_Finalize
 *   Communicators  : MPI_Comm_dup, MPI_Comm_split, MPI_Comm_create --
 *                    hooked only to assign a cross-rank-agreed commid=,
 *                    see "Communicator identity" below.
 *   Request linking: Isend/Irecv/non-blocking collectives tagged with a
 *                    request id (sid=/req_id=) so Wait/Waitall/Waitany/
 *                    Waitsome/Test* spans can be cross-linked to them.
 *
 * ── Wildcard resolution (MPI_ANY_SOURCE / MPI_ANY_TAG) ────────────────
 * A receive posted with MPI_ANY_SOURCE/MPI_ANY_TAG doesn't know its real
 * peer/tag until the call completes -- the MPI standard guarantees the
 * real values are always written into the resulting MPI_Status, even when
 * the call itself used a wildcard. This hook substitutes its OWN status
 * buffer whenever the caller passes MPI_STATUS_IGNORE (a completely
 * transparent substitution -- the caller never sees either buffer), so a
 * wildcard match's real source/tag is always resolved, never left
 * "unknown" purely because the application didn't ask for its own status.
 * Emitted as the resolved peer=/tag= plus an explicit wildcard=1 marker so
 * downstream causal-graph code can still tell it apart from a match the
 * program requested by exact (source,tag).
 *
 * ── Communicator identity ──────────────────────────────────────────────
 * Collective/p2p spans carry commid=<N>: 0 for MPI_COMM_WORLD (a global
 * constant, needs no agreement), or a globally-unique rank-agreed integer
 * for any communicator created via MPI_Comm_dup/split/create (hooked
 * below to broadcast a freshly-assigned id, from the new communicator's
 * own rank 0, over the new communicator itself right after creation --
 * safe because comm creation is already a collective call every member
 * rank makes together, so a Bcast on the brand-new communicator
 * immediately after is a standard, well-defined pattern; the id itself
 * packs the bootstrapping process's MPI_COMM_WORLD rank with its own
 * local sequence number so two unrelated communicators bootstrapped by
 * two different ranks can never collide -- see comm_id_register()).
 * commid=-1 means "unregistered" --
 * MPI_COMM_SELF or a communicator created via an API this hook doesn't
 * intercept (e.g. MPI_Comm_create_group, MPI_Cart_create, MPI_Intercomm_*)
 * -- collective/p2p matching for those falls back to the pre-existing
 * call-order-only heuristic, same as before this identity mechanism
 * existed.
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

#include "../common/callstack.h"

static void emit_span(const char *cat, uint64_t start_ns, uint64_t dur_ns,
                      const char *name, const char *extra) {
    pthread_mutex_lock(&g_sock_mutex);
    ensure_connected();
    if (g_sock >= 0) {
        char buf[1280];
        int n = snprintf(buf, sizeof(buf),
            "span:%s:%d:%d:%llu:%llu:%s:%s\n",
            cat, g_pid, (int)gettid_compat(),
            (unsigned long long)start_ns, (unsigned long long)dur_ns,
            name, extra ? extra : "");
        if (n > 0 && n < (int)sizeof(buf)) send_all(buf, n);
        emit_callstack(start_ns);
    }
    pthread_mutex_unlock(&g_sock_mutex);
}

static void emit_instant(const char *cat, const char *name, const char *extra) {
    pthread_mutex_lock(&g_sock_mutex);
    ensure_connected();
    if (g_sock >= 0) {
        char buf[640];
        int n = snprintf(buf, sizeof(buf), "inst:%s:%d:%d:%llu:%s:%s\n",
                         cat, g_pid, (int)gettid_compat(),
                         (unsigned long long)now_ns(), name, extra ? extra : "");
        if (n > 0 && n < (int)sizeof(buf)) send_all(buf, n);
    }
    pthread_mutex_unlock(&g_sock_mutex);
}

static void emit_ctr(const char *cat, const char *name, int64_t value, const char *unit) {
    pthread_mutex_lock(&g_sock_mutex);
    ensure_connected();
    if (g_sock >= 0) {
        char buf[256];
        int n = snprintf(buf, sizeof(buf), "ctr:%s:%d:%llu:%s:%lld:%s\n",
                         cat, g_pid, (unsigned long long)now_ns(),
                         name, (long long)value, unit);
        if (n > 0 && n < (int)sizeof(buf)) send_all(buf, n);
    }
    pthread_mutex_unlock(&g_sock_mutex);
}

/* ── Communicator identity ──────────────────────────────────────────── */
#define COMM_TABLE_CAP 256
typedef struct { MPI_Comm comm; int64_t id; } CommRec;
static CommRec         g_comm_table[COMM_TABLE_CAP];
static int              g_comm_n       = 0;
static int64_t          g_comm_id_seq  = 1;   /* 0 reserved for MPI_COMM_WORLD */
static pthread_mutex_t  g_comm_mutex   = PTHREAD_MUTEX_INITIALIZER;

static int64_t comm_id_for(MPI_Comm comm) {
    if (comm == MPI_COMM_WORLD) return 0;
    pthread_mutex_lock(&g_comm_mutex);
    for (int i = 0; i < g_comm_n; i++) {
        if (g_comm_table[i].comm == comm) {
            int64_t id = g_comm_table[i].id;
            pthread_mutex_unlock(&g_comm_mutex);
            return id;
        }
    }
    pthread_mutex_unlock(&g_comm_mutex);
    return -1;  /* unregistered: MPI_COMM_SELF, or created via an un-hooked API */
}

/* Must only be called on a communicator that was JUST collectively created
 * (all member ranks reach this together) -- see file header comment.
 *
 * The id is a (bootstrapping process's MPI_COMM_WORLD rank, that
 * process's own local sequence number) pair packed into one int64. A
 * plain per-process sequence number alone is NOT enough: whichever
 * process ends up as local-rank-0 of the new communicator generates the
 * id, and two *different* communicators bootstrapped by two *different*
 * world ranks (e.g. two disjoint halves of an MPI_Comm_split) would both
 * independently start counting from 1 and collide on the same id despite
 * being semantically unrelated. Packing in the bootstrapping rank's own
 * globally-unique world rank makes collisions impossible without needing
 * any extra cross-rank coordination beyond the Bcast already below. */
static void comm_id_register(MPI_Comm new_comm) {
    if (new_comm == MPI_COMM_NULL) return;
    long long id = 0;
    int my_local_rank = 0;
    PMPI_Comm_rank(new_comm, &my_local_rank);
    if (my_local_rank == 0) {
        int world_rank = 0;
        PMPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        pthread_mutex_lock(&g_comm_mutex);
        long long local_seq = (long long)(g_comm_id_seq++);
        pthread_mutex_unlock(&g_comm_mutex);
        id = ((long long)world_rank << 32) | (local_seq & 0xffffffffLL);
    }
    PMPI_Bcast(&id, 1, MPI_LONG_LONG, 0, new_comm);
    pthread_mutex_lock(&g_comm_mutex);
    if (g_comm_n < COMM_TABLE_CAP) {
        g_comm_table[g_comm_n].comm = new_comm;
        g_comm_table[g_comm_n].id   = (int64_t)id;
        g_comm_n++;
    }
    pthread_mutex_unlock(&g_comm_mutex);
}

int MPI_Comm_dup(MPI_Comm comm, MPI_Comm *newcomm) {
    int ret = PMPI_Comm_dup(comm, newcomm);
    if (ret == MPI_SUCCESS && newcomm) comm_id_register(*newcomm);
    return ret;
}

int MPI_Comm_split(MPI_Comm comm, int color, int key, MPI_Comm *newcomm) {
    int ret = PMPI_Comm_split(comm, color, key, newcomm);
    if (ret == MPI_SUCCESS && newcomm && *newcomm != MPI_COMM_NULL) comm_id_register(*newcomm);
    return ret;
}

int MPI_Comm_create(MPI_Comm comm, MPI_Group group, MPI_Comm *newcomm) {
    int ret = PMPI_Comm_create(comm, group, newcomm);
    if (ret == MPI_SUCCESS && newcomm && *newcomm != MPI_COMM_NULL) comm_id_register(*newcomm);
    return ret;
}

/* ── Async request cross-linking table (M4) ─────────────────────────── */
#define REQ_TABLE_CAP 4096
typedef struct {
    MPI_Request req;
    uint64_t    id;
    char        type[16];
    int         peer;
    int         tag;
    size_t      bytes;
    int         wildcard;   /* 1 if this was an Irecv posted with ANY_SOURCE/ANY_TAG */
} ReqRec;

static ReqRec          g_req_table[REQ_TABLE_CAP];
static int             g_req_n      = 0;
static uint64_t        g_req_seq    = 1;
static pthread_mutex_t g_req_mutex  = PTHREAD_MUTEX_INITIALIZER;

static uint64_t req_register(MPI_Request req, const char *type,
                              int peer, int tag, size_t bytes, int wildcard) {
    pthread_mutex_lock(&g_req_mutex);
    uint64_t id = 0;
    if (g_req_n < REQ_TABLE_CAP) {
        id = g_req_seq++;
        ReqRec *r   = &g_req_table[g_req_n++];
        r->req      = req;
        r->id       = id;
        r->peer     = peer;
        r->tag      = tag;
        r->bytes    = bytes;
        r->wildcard = wildcard;
        strncpy(r->type, type, 15); r->type[15] = '\0';
    }
    /* else: table full (REQ_TABLE_CAP simultaneously outstanding async
     * requests on this rank -- very deep pipelining). Return 0 rather than
     * a fresh-looking sequence id: every call site already treats a 0
     * return as "don't tag sid=", which correctly omits an id that would
     * otherwise dangle forever (req_lookup() could never find a matching
     * table entry for it, so the later MPI_Wait's psid= cross-link would
     * silently never resolve). */
    pthread_mutex_unlock(&g_req_mutex);
    return id;
}

/* Returns 1 and fills out-params if found; removes the entry. */
static int req_lookup(MPI_Request req, uint64_t *id_out, int *wildcard_out) {
    pthread_mutex_lock(&g_req_mutex);
    for (int i = 0; i < g_req_n; i++) {
        if (g_req_table[i].req == req) {
            *id_out = g_req_table[i].id;
            if (wildcard_out) *wildcard_out = g_req_table[i].wildcard;
            g_req_table[i] = g_req_table[--g_req_n];
            pthread_mutex_unlock(&g_req_mutex);
            return 1;
        }
    }
    pthread_mutex_unlock(&g_req_mutex);
    return 0;
}

/* Non-destructive lookup (for Test*, which may report flag=0 -- request
 * must stay in the table since it hasn't completed). */
static int req_peek(MPI_Request req, uint64_t *id_out, int *wildcard_out) {
    pthread_mutex_lock(&g_req_mutex);
    for (int i = 0; i < g_req_n; i++) {
        if (g_req_table[i].req == req) {
            *id_out = g_req_table[i].id;
            if (wildcard_out) *wildcard_out = g_req_table[i].wildcard;
            pthread_mutex_unlock(&g_req_mutex);
            return 1;
        }
    }
    pthread_mutex_unlock(&g_req_mutex);
    return 0;
}

static void req_remove(MPI_Request req) {
    pthread_mutex_lock(&g_req_mutex);
    for (int i = 0; i < g_req_n; i++) {
        if (g_req_table[i].req == req) {
            g_req_table[i] = g_req_table[--g_req_n];
            break;
        }
    }
    pthread_mutex_unlock(&g_req_mutex);
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

/* ── Clock-offset estimation (multi-node design, Cristian's algorithm) ───
 * hprofiler's collector is a local AF_UNIX socket (see criticalpath.py's
 * module docstring): each node runs its own independent collector, so a
 * multi-node run naturally produces one trace file per node. Their
 * CLOCK_MONOTONIC timestamps are NOT comparable across nodes as-is --
 * different machines, different arbitrary epochs, no shared reference.
 * This estimates, once per non-root rank at MPI_Init time, that rank's
 * clock offset relative to rank 0's clock, with an explicit error bound
 * (not just a point estimate) -- src/analysis/multinode.py's merge step
 * (Python side, fully unit-tested against synthetic offset scenarios)
 * consumes this to align multiple saved per-node trace files onto one
 * common timeline.
 *
 * Protocol (classic Cristian's algorithm / NTP-style round-trip
 * estimate): rank R sends a zero-byte ping to rank 0 at its own local
 * time T1, rank 0 records its own local times T2 (ping received) and T3
 * (reply about to be sent) and returns both to R, R records T4 (reply
 * received). Assuming symmetric network latency (the standard, stated
 * simplifying assumption -- not always true, hence the explicit error
 * bound rather than treating the estimate as exact):
 *   round_trip   = T4 - T1                          (measured in R's clock)
 *   offset       = (T2+T3)/2 - (T1 + round_trip/2)   (rank 0's clock minus R's, at the same real instant)
 *   error_bound  = round_trip/2                      (standard bound under the symmetric-latency assumption)
 * R's timestamps can be converted into rank 0's clock frame via
 * `t_in_rank0_frame = t_in_R_frame + offset`.
 *
 * Rank 0 loops over every other rank SEQUENTIALLY (not the fastest
 * possible design -- O(size) round trips -- but the simplest to reason
 * about correctly, and this runs once per job, not per profiled call).
 *
 * OFF BY DEFAULT (HPROFILER_CLOCK_SYNC=1 to enable): this adds a real,
 * blocking round-trip exchange to MPI_Init -- a function every MPI
 * program calls -- so it must never run unless explicitly requested, to
 * guarantee zero effect on any profiling run that doesn't ask for
 * multi-node clock alignment.
 *
 * Verification status: compiled, and the offset/error-bound ARITHMETIC
 * is verified via tests/test_multinode.py (ported to Python, since the
 * formula itself has no MPI dependency) against hand-computed synthetic
 * (T1,T2,T3,T4) scenarios -- but this exact C implementation has never
 * executed a real cross-node round trip, since this machine cannot form
 * a real multi-rank MPI_COMM_WORLD at all (see DOCUMENTATION.md's Known
 * Limitations / project-paper4-benchmark-suite memory). */
#define HPROFILER_CLOCK_SYNC_TAG 30001  /* a small, fixed value safely under
                                          * the MPI standard's guaranteed
                                          * minimum MPI_TAG_UB (32767) --
                                          * not guaranteed collision-free
                                          * with an application's own tags,
                                          * but the best available without
                                          * a dedicated communicator. */

static void clock_sync_if_requested(void) {
    if (!getenv("HPROFILER_CLOCK_SYNC")) return;
    int size = 0;
    PMPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size < 2) return;

    if (g_mpi_rank == 0) {
        for (int r = 1; r < size; r++) {
            PMPI_Recv(NULL, 0, MPI_BYTE, r, HPROFILER_CLOCK_SYNC_TAG,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            uint64_t t2 = now_ns();
            uint64_t t3 = now_ns();  /* negligible local work between recv and reply */
            uint64_t pair[2] = { t2, t3 };
            PMPI_Send(pair, 2, MPI_UNSIGNED_LONG_LONG, r, HPROFILER_CLOCK_SYNC_TAG, MPI_COMM_WORLD);
        }
    } else {
        uint64_t t1 = now_ns();
        PMPI_Send(NULL, 0, MPI_BYTE, 0, HPROFILER_CLOCK_SYNC_TAG, MPI_COMM_WORLD);
        uint64_t pair[2] = {0, 0};
        PMPI_Recv(pair, 2, MPI_UNSIGNED_LONG_LONG, 0, HPROFILER_CLOCK_SYNC_TAG,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        uint64_t t4 = now_ns();
        uint64_t t2 = pair[0], t3 = pair[1];
        uint64_t round_trip = t4 - t1;
        int64_t offset = (int64_t)((t2 + t3) / 2) - (int64_t)(t1 + round_trip / 2);
        uint64_t error_bound = round_trip / 2;
        emit_ctr("mpi", "clock_offset_vs_rank0_ns", offset, "ns");
        emit_ctr("mpi", "clock_offset_error_bound_ns", (int64_t)error_bound, "ns");
        emit_ctr("mpi", "clock_sync_round_trip_ns", (int64_t)round_trip, "ns");
    }
}

int MPI_Init(int *argc, char ***argv) {
    int ret = PMPI_Init(argc, argv);
    if (ret == MPI_SUCCESS) {
        PMPI_Comm_rank(MPI_COMM_WORLD, &g_mpi_rank);
        ensure_connected();
        cs_init();
        clock_sync_if_requested();
    }
    return ret;
}

int MPI_Init_thread(int *argc, char ***argv, int required, int *provided) {
    int ret = PMPI_Init_thread(argc, argv, required, provided);
    if (ret == MPI_SUCCESS) {
        PMPI_Comm_rank(MPI_COMM_WORLD, &g_mpi_rank);
        ensure_connected();
        cs_init();
        clock_sync_if_requested();
    }
    return ret;
}

int MPI_Finalize(void) {
    if (g_sock >= 0) {
        /* Flush the kernel send buffer before closing so no queued events are lost. */
        shutdown(g_sock, SHUT_WR);
        char drain[64];
        while (recv(g_sock, drain, sizeof(drain), 0) > 0) {}
        close(g_sock);
        g_sock = -1;
    }
    return PMPI_Finalize();
}

/* ── Point-to-point ─────────────────────────────────────────────────── */

#define _P2P(FNAME, PMPI_CALL, TYPE_STR, ...)                          \
int FNAME(__VA_ARGS__) {                                                \
    char extra[192];                                                    \
    size_t nb = (size_t)count * dtype_size(datatype);                  \
    int64_t cid = comm_id_for(comm);                                    \
    snprintf(extra, sizeof(extra),                                      \
             "type=%s,bytes=%zu,rank=%d,peer=%d,tag=%d,commid=%lld",   \
             TYPE_STR, nb, g_mpi_rank, peer_rank, tag, (long long)cid); \
    uint64_t t0 = now_ns();                                             \
    int ret = PMPI_CALL;                                                \
    emit_span("mpi", t0, now_ns()-t0, #FNAME, extra);                  \
    return ret;                                                         \
}

#define peer_rank dest
_P2P(MPI_Send,
     PMPI_Send(buf, count, datatype, dest, tag, comm),
     "send",
     const void *buf, int count, MPI_Datatype datatype,
     int dest, int tag, MPI_Comm comm)

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
    int64_t cid = comm_id_for(comm);
    int wildcard = (source == MPI_ANY_SOURCE) || (tag == MPI_ANY_TAG);
    /* Substitute our own status buffer when the caller passes
     * MPI_STATUS_IGNORE, so a wildcard match's real (source,tag) is
     * always resolvable -- see file header comment. Completely
     * transparent: the caller never sees either buffer either way. */
    MPI_Status local_status;
    MPI_Status *use_status = (status == MPI_STATUS_IGNORE) ? &local_status : status;
    uint64_t t0 = now_ns();
    int ret = PMPI_Recv(buf, count, datatype, source, tag, comm, use_status);
    int resolved_source = source, resolved_tag = tag;
    if (ret == MPI_SUCCESS && wildcard) {
        resolved_source = use_status->MPI_SOURCE;
        resolved_tag    = use_status->MPI_TAG;
    }
    char extra[224];
    snprintf(extra, sizeof(extra),
             "type=recv,bytes=%zu,rank=%d,peer=%d,tag=%d,commid=%lld%s",
             nb, g_mpi_rank, resolved_source, resolved_tag, (long long)cid,
             wildcard ? ",wildcard=1" : "");
    emit_span("mpi", t0, now_ns()-t0, "MPI_Recv", extra);
    return ret;
}

int MPI_Isend(const void *buf, int count, MPI_Datatype datatype,
              int dest, int tag, MPI_Comm comm, MPI_Request *request) {
    size_t nb = (size_t)count * dtype_size(datatype);
    int64_t cid = comm_id_for(comm);
    uint64_t t0 = now_ns();
    int ret = PMPI_Isend(buf, count, datatype, dest, tag, comm, request);
    uint64_t req_id = (ret == MPI_SUCCESS && request && *request != MPI_REQUEST_NULL)
                      ? req_register(*request, "isend", dest, tag, nb, 0) : 0;
    char extra[224];
    if (req_id)
        snprintf(extra, sizeof(extra),
                 "type=isend,bytes=%zu,rank=%d,peer=%d,tag=%d,commid=%lld,sid=%llu",
                 nb, g_mpi_rank, dest, tag, (long long)cid, (unsigned long long)req_id);
    else
        snprintf(extra, sizeof(extra),
                 "type=isend,bytes=%zu,rank=%d,peer=%d,tag=%d,commid=%lld",
                 nb, g_mpi_rank, dest, tag, (long long)cid);
    emit_span("mpi", t0, now_ns()-t0, "MPI_Isend", extra);
    return ret;
}

int MPI_Irecv(void *buf, int count, MPI_Datatype datatype,
              int source, int tag, MPI_Comm comm, MPI_Request *request) {
    size_t nb = (size_t)count * dtype_size(datatype);
    int64_t cid = comm_id_for(comm);
    int wildcard = (source == MPI_ANY_SOURCE) || (tag == MPI_ANY_TAG);
    uint64_t t0 = now_ns();
    int ret = PMPI_Irecv(buf, count, datatype, source, tag, comm, request);
    uint64_t req_id = (ret == MPI_SUCCESS && request && *request != MPI_REQUEST_NULL)
                      ? req_register(*request, "irecv", source, tag, nb, wildcard) : 0;
    /* Real peer/tag for a wildcard Irecv aren't known yet -- they only
     * become known when Wait/Waitany/Waitsome/Test* on this request
     * completes (see those functions' rpeer=/rtag= tags below); this span
     * carries wildcard=1 rather than a misleading placeholder value. */
    char extra[224];
    if (req_id)
        snprintf(extra, sizeof(extra),
                 "type=irecv,bytes=%zu,rank=%d,peer=%d,tag=%d,commid=%lld,sid=%llu%s",
                 nb, g_mpi_rank, source, tag, (long long)cid,
                 (unsigned long long)req_id, wildcard ? ",wildcard=1" : "");
    else
        snprintf(extra, sizeof(extra),
                 "type=irecv,bytes=%zu,rank=%d,peer=%d,tag=%d,commid=%lld%s",
                 nb, g_mpi_rank, source, tag, (long long)cid,
                 wildcard ? ",wildcard=1" : "");
    emit_span("mpi", t0, now_ns()-t0, "MPI_Irecv", extra);
    return ret;
}

int MPI_Wait(MPI_Request *request, MPI_Status *status) {
    MPI_Request saved = request ? *request : MPI_REQUEST_NULL;
    MPI_Status local_status;
    MPI_Status *use_status = (status == MPI_STATUS_IGNORE) ? &local_status : status;
    uint64_t t0 = now_ns();
    int ret = PMPI_Wait(request, use_status);
    uint64_t req_id = 0;
    int wildcard = 0;
    req_lookup(saved, &req_id, &wildcard);
    char extra[160];
    if (req_id && wildcard && ret == MPI_SUCCESS)
        snprintf(extra, sizeof(extra), "type=wait,rank=%d,psid=%llu,rpeer=%d,rtag=%d",
                 g_mpi_rank, (unsigned long long)req_id,
                 use_status->MPI_SOURCE, use_status->MPI_TAG);
    else if (req_id)
        snprintf(extra, sizeof(extra), "type=wait,rank=%d,psid=%llu",
                 g_mpi_rank, (unsigned long long)req_id);
    else
        snprintf(extra, sizeof(extra), "type=wait,rank=%d", g_mpi_rank);
    emit_span("mpi", t0, now_ns()-t0, "MPI_Wait", extra);
    return ret;
}

int MPI_Waitall(int count, MPI_Request requests[], MPI_Status statuses[]) {
    /* Save all request handles before PMPI_Waitall nullifies them.
     * Heap-allocate to avoid stack overflow with large request arrays. */
    MPI_Request *saved = (count > 0)
        ? (MPI_Request *)malloc((size_t)count * sizeof(MPI_Request)) : NULL;
    if (saved)
        for (int i = 0; i < count; i++) saved[i] = requests[i];
    /* Always use our own status array (regardless of what caller passed,
     * including MPI_STATUSES_IGNORE) so wildcard-recv resolution works
     * uniformly -- see MPI_Recv's comment. */
    MPI_Status *use_statuses = (count > 0)
        ? (MPI_Status *)malloc((size_t)count * sizeof(MPI_Status)) : NULL;
    uint64_t t0 = now_ns();
    int ret = PMPI_Waitall(count, requests, use_statuses ? use_statuses : statuses);
    /* Collect req IDs for cross-linking, and resolved peer/tag for any
     * wildcard recvs among them, into semicolon-separated lists. */
    char psid_buf[256] = ""; int psid_len = 0;
    char rmatch_buf[384] = ""; int rmatch_len = 0;
    if (saved) {
        for (int i = 0; i < count; i++) {
            uint64_t rid = 0; int wildcard = 0;
            if (req_lookup(saved[i], &rid, &wildcard) && rid) {
                int w = snprintf(psid_buf + psid_len, sizeof(psid_buf) - (size_t)psid_len,
                                 "%s%llu", psid_len ? ";" : "", (unsigned long long)rid);
                if (w > 0) psid_len += w;
                if (wildcard && use_statuses && ret == MPI_SUCCESS) {
                    /* '/' as the intra-triple separator, not ':' -- tag
                     * VALUES must never contain colons (see _split_name_tags
                     * in src/core/runner.py): the wire parser locates the
                     * tags segment by scanning for the *last* colon in the
                     * record, so a colon buried inside a value here would
                     * be misidentified as that boundary and corrupt both
                     * the parsed name and every tag on the line. */
                    int w2 = snprintf(rmatch_buf + rmatch_len, sizeof(rmatch_buf) - (size_t)rmatch_len,
                                      "%s%llu/%d/%d", rmatch_len ? ";" : "",
                                      (unsigned long long)rid,
                                      use_statuses[i].MPI_SOURCE, use_statuses[i].MPI_TAG);
                    if (w2 > 0) rmatch_len += w2;
                }
            }
        }
        free(saved);
    }
    free(use_statuses);
    char extra[768];
    int n = snprintf(extra, sizeof(extra), "type=waitall,count=%d,rank=%d", count, g_mpi_rank);
    if (psid_len && n > 0 && n < (int)sizeof(extra))
        n += snprintf(extra + n, sizeof(extra) - (size_t)n, ",psid=%s", psid_buf);
    if (rmatch_len && n > 0 && n < (int)sizeof(extra))
        n += snprintf(extra + n, sizeof(extra) - (size_t)n, ",rmatches=%s", rmatch_buf);
    emit_span("mpi", t0, now_ns()-t0, "MPI_Waitall", extra);
    return ret;
}

int MPI_Waitany(int count, MPI_Request requests[], int *index, MPI_Status *status) {
    MPI_Request *saved = (count > 0)
        ? (MPI_Request *)malloc((size_t)count * sizeof(MPI_Request)) : NULL;
    if (saved)
        for (int i = 0; i < count; i++) saved[i] = requests[i];
    MPI_Status local_status;
    MPI_Status *use_status = (status == MPI_STATUS_IGNORE) ? &local_status : status;
    uint64_t t0 = now_ns();
    int ret = PMPI_Waitany(count, requests, index, use_status);
    /* Exactly one request completes (index != MPI_UNDEFINED) -- look it up
     * via the SAVED array, since PMPI_Waitany nulls requests[*index] for
     * non-persistent requests. */
    char extra[224];
    if (ret == MPI_SUCCESS && index && *index != MPI_UNDEFINED && saved) {
        uint64_t rid = 0; int wildcard = 0;
        int found = req_lookup(saved[*index], &rid, &wildcard);
        if (found && rid && wildcard)
            snprintf(extra, sizeof(extra),
                     "type=waitany,count=%d,rank=%d,completed_index=%d,psid=%llu,rpeer=%d,rtag=%d",
                     count, g_mpi_rank, *index, (unsigned long long)rid,
                     use_status->MPI_SOURCE, use_status->MPI_TAG);
        else if (found && rid)
            snprintf(extra, sizeof(extra),
                     "type=waitany,count=%d,rank=%d,completed_index=%d,psid=%llu",
                     count, g_mpi_rank, *index, (unsigned long long)rid);
        else
            snprintf(extra, sizeof(extra), "type=waitany,count=%d,rank=%d,completed_index=%d",
                     count, g_mpi_rank, *index);
    } else {
        snprintf(extra, sizeof(extra), "type=waitany,count=%d,rank=%d,completed_index=none",
                 count, g_mpi_rank);
    }
    free(saved);
    emit_span("mpi", t0, now_ns()-t0, "MPI_Waitany", extra);
    return ret;
}

int MPI_Waitsome(int incount, MPI_Request requests[], int *outcount,
                 int indices[], MPI_Status statuses[]) {
    MPI_Request *saved = (incount > 0)
        ? (MPI_Request *)malloc((size_t)incount * sizeof(MPI_Request)) : NULL;
    if (saved)
        for (int i = 0; i < incount; i++) saved[i] = requests[i];
    MPI_Status *use_statuses = (incount > 0)
        ? (MPI_Status *)malloc((size_t)incount * sizeof(MPI_Status)) : NULL;
    uint64_t t0 = now_ns();
    int ret = PMPI_Waitsome(incount, requests, outcount, indices,
                            use_statuses ? use_statuses : statuses);
    char psid_buf[256] = ""; int psid_len = 0;
    char rmatch_buf[384] = ""; int rmatch_len = 0;
    if (ret == MPI_SUCCESS && saved && outcount && indices &&
        *outcount != MPI_UNDEFINED) {
        for (int k = 0; k < *outcount; k++) {
            int idx = indices[k];
            if (idx < 0 || idx >= incount) continue;
            uint64_t rid = 0; int wildcard = 0;
            if (req_lookup(saved[idx], &rid, &wildcard) && rid) {
                int w = snprintf(psid_buf + psid_len, sizeof(psid_buf) - (size_t)psid_len,
                                 "%s%llu", psid_len ? ";" : "", (unsigned long long)rid);
                if (w > 0) psid_len += w;
                if (wildcard && use_statuses) {
                    /* '/' not ':' -- see the matching comment in MPI_Waitall. */
                    int w2 = snprintf(rmatch_buf + rmatch_len, sizeof(rmatch_buf) - (size_t)rmatch_len,
                                      "%s%llu/%d/%d", rmatch_len ? ";" : "",
                                      (unsigned long long)rid,
                                      use_statuses[k].MPI_SOURCE, use_statuses[k].MPI_TAG);
                    if (w2 > 0) rmatch_len += w2;
                }
            }
        }
    }
    free(saved);
    free(use_statuses);
    char extra[768];
    int n = snprintf(extra, sizeof(extra), "type=waitsome,incount=%d,outcount=%d,rank=%d",
                     incount, (ret == MPI_SUCCESS && outcount) ? *outcount : -1, g_mpi_rank);
    if (psid_len && n > 0 && n < (int)sizeof(extra))
        n += snprintf(extra + n, sizeof(extra) - (size_t)n, ",psid=%s", psid_buf);
    if (rmatch_len && n > 0 && n < (int)sizeof(extra))
        n += snprintf(extra + n, sizeof(extra) - (size_t)n, ",rmatches=%s", rmatch_buf);
    emit_span("mpi", t0, now_ns()-t0, "MPI_Waitsome", extra);
    return ret;
}

/* ── Test* (non-blocking completion checks) ─────────────────────────────
 * Modeled as instant events, not spans: a Test* call is meant to be cheap
 * and non-blocking, and its interesting causal signal is binary (did it
 * observe completion or not), not a duration. flag=0 ("checked, not yet
 * ready") is itself a real causal node -- the DAG needs to be able to
 * represent "the program polled and found nothing ready" distinctly from
 * "the program never checked". */

int MPI_Test(MPI_Request *request, int *flag, MPI_Status *status) {
    MPI_Request saved = request ? *request : MPI_REQUEST_NULL;
    MPI_Status local_status;
    MPI_Status *use_status = (status == MPI_STATUS_IGNORE) ? &local_status : status;
    int ret = PMPI_Test(request, flag, use_status);
    uint64_t rid = 0; int wildcard = 0;
    int found = req_peek(saved, &rid, &wildcard);
    char extra[224];
    if (ret == MPI_SUCCESS && flag && *flag) {
        req_remove(saved);
        if (found && rid && wildcard)
            snprintf(extra, sizeof(extra), "type=test,flag=1,rank=%d,psid=%llu,rpeer=%d,rtag=%d",
                     g_mpi_rank, (unsigned long long)rid, use_status->MPI_SOURCE, use_status->MPI_TAG);
        else if (found && rid)
            snprintf(extra, sizeof(extra), "type=test,flag=1,rank=%d,psid=%llu",
                     g_mpi_rank, (unsigned long long)rid);
        else
            snprintf(extra, sizeof(extra), "type=test,flag=1,rank=%d", g_mpi_rank);
    } else {
        snprintf(extra, sizeof(extra), "type=test,flag=0,rank=%d%s%s",
                 g_mpi_rank, found && rid ? ",psid=" : "",
                 found && rid ? "" : "");
        if (found && rid) {
            char idbuf[32]; snprintf(idbuf, sizeof(idbuf), "%llu", (unsigned long long)rid);
            strncat(extra, idbuf, sizeof(extra) - strlen(extra) - 1);
        }
    }
    emit_instant("mpi", "MPI_Test", extra);
    return ret;
}

int MPI_Testany(int count, MPI_Request requests[], int *index, int *flag, MPI_Status *status) {
    MPI_Request *saved = (count > 0)
        ? (MPI_Request *)malloc((size_t)count * sizeof(MPI_Request)) : NULL;
    if (saved)
        for (int i = 0; i < count; i++) saved[i] = requests[i];
    MPI_Status local_status;
    MPI_Status *use_status = (status == MPI_STATUS_IGNORE) ? &local_status : status;
    int ret = PMPI_Testany(count, requests, index, flag, use_status);
    char extra[224];
    if (ret == MPI_SUCCESS && flag && *flag && index && *index != MPI_UNDEFINED && saved) {
        uint64_t rid = 0; int wildcard = 0;
        int found = req_lookup(saved[*index], &rid, &wildcard);
        if (found && rid && wildcard)
            snprintf(extra, sizeof(extra),
                     "type=testany,flag=1,count=%d,rank=%d,completed_index=%d,psid=%llu,rpeer=%d,rtag=%d",
                     count, g_mpi_rank, *index, (unsigned long long)rid,
                     use_status->MPI_SOURCE, use_status->MPI_TAG);
        else if (found && rid)
            snprintf(extra, sizeof(extra),
                     "type=testany,flag=1,count=%d,rank=%d,completed_index=%d,psid=%llu",
                     count, g_mpi_rank, *index, (unsigned long long)rid);
        else
            snprintf(extra, sizeof(extra), "type=testany,flag=1,count=%d,rank=%d,completed_index=%d",
                     count, g_mpi_rank, *index);
    } else {
        snprintf(extra, sizeof(extra), "type=testany,flag=0,count=%d,rank=%d", count, g_mpi_rank);
    }
    free(saved);
    emit_instant("mpi", "MPI_Testany", extra);
    return ret;
}

int MPI_Testsome(int incount, MPI_Request requests[], int *outcount,
                 int indices[], MPI_Status statuses[]) {
    MPI_Request *saved = (incount > 0)
        ? (MPI_Request *)malloc((size_t)incount * sizeof(MPI_Request)) : NULL;
    if (saved)
        for (int i = 0; i < incount; i++) saved[i] = requests[i];
    MPI_Status *use_statuses = (incount > 0)
        ? (MPI_Status *)malloc((size_t)incount * sizeof(MPI_Status)) : NULL;
    int ret = PMPI_Testsome(incount, requests, outcount, indices,
                            use_statuses ? use_statuses : statuses);
    char psid_buf[256] = ""; int psid_len = 0;
    if (ret == MPI_SUCCESS && saved && outcount && indices && *outcount != MPI_UNDEFINED) {
        for (int k = 0; k < *outcount; k++) {
            int idx = indices[k];
            if (idx < 0 || idx >= incount) continue;
            uint64_t rid = 0;
            if (req_lookup(saved[idx], &rid, NULL) && rid) {
                int w = snprintf(psid_buf + psid_len, sizeof(psid_buf) - (size_t)psid_len,
                                 "%s%llu", psid_len ? ";" : "", (unsigned long long)rid);
                if (w > 0) psid_len += w;
            }
        }
    }
    free(saved);
    free(use_statuses);
    char extra[384];
    int n = snprintf(extra, sizeof(extra), "type=testsome,incount=%d,outcount=%d,rank=%d",
                     incount, (ret == MPI_SUCCESS && outcount) ? *outcount : -1, g_mpi_rank);
    if (psid_len && n > 0 && n < (int)sizeof(extra))
        snprintf(extra + n, sizeof(extra) - (size_t)n, ",psid=%s", psid_buf);
    emit_instant("mpi", "MPI_Testsome", extra);
    return ret;
}

int MPI_Testall(int count, MPI_Request requests[], int *flag, MPI_Status statuses[]) {
    MPI_Request *saved = (count > 0)
        ? (MPI_Request *)malloc((size_t)count * sizeof(MPI_Request)) : NULL;
    if (saved)
        for (int i = 0; i < count; i++) saved[i] = requests[i];
    MPI_Status *use_statuses = (count > 0)
        ? (MPI_Status *)malloc((size_t)count * sizeof(MPI_Status)) : NULL;
    int ret = PMPI_Testall(count, requests, flag, use_statuses ? use_statuses : statuses);
    char psid_buf[256] = ""; int psid_len = 0;
    if (ret == MPI_SUCCESS && flag && *flag && saved) {
        for (int i = 0; i < count; i++) {
            uint64_t rid = 0;
            if (req_lookup(saved[i], &rid, NULL) && rid) {
                int w = snprintf(psid_buf + psid_len, sizeof(psid_buf) - (size_t)psid_len,
                                 "%s%llu", psid_len ? ";" : "", (unsigned long long)rid);
                if (w > 0) psid_len += w;
            }
        }
    }
    free(saved);
    free(use_statuses);
    char extra[384];
    int n = snprintf(extra, sizeof(extra), "type=testall,flag=%d,count=%d,rank=%d",
                     (ret == MPI_SUCCESS && flag) ? *flag : -1, count, g_mpi_rank);
    if (psid_len && n > 0 && n < (int)sizeof(extra))
        snprintf(extra + n, sizeof(extra) - (size_t)n, ",psid=%s", psid_buf);
    emit_instant("mpi", "MPI_Testall", extra);
    return ret;
}

/* ── Cancellation ────────────────────────────────────────────────────────
 * A cancelled request never completes normally -- its lifecycle in the
 * causal graph needs an explicit terminal state distinct from a real
 * completion, not silence. */
int MPI_Cancel(MPI_Request *request) {
    MPI_Request saved = request ? *request : MPI_REQUEST_NULL;
    int ret = PMPI_Cancel(request);
    uint64_t rid = 0;
    int found = req_peek(saved, &rid, NULL);
    char extra[96];
    if (found && rid)
        snprintf(extra, sizeof(extra), "type=cancel,rank=%d,psid=%llu",
                 g_mpi_rank, (unsigned long long)rid);
    else
        snprintf(extra, sizeof(extra), "type=cancel,rank=%d", g_mpi_rank);
    emit_instant("mpi", "MPI_Cancel", extra);
    return ret;
}

/* ── Collectives ────────────────────────────────────────────────────── */

#define _COLL(FNAME, PMPI_CALL, TYPE_STR, BYTES_EXPR, ...)             \
int FNAME(__VA_ARGS__) {                                                \
    char extra[160];                                                    \
    size_t nb = (BYTES_EXPR);                                           \
    int64_t cid = comm_id_for(comm);                                    \
    snprintf(extra, sizeof(extra),                                      \
             "type=%s,bytes=%zu,rank=%d,commid=%lld",                   \
             TYPE_STR, nb, g_mpi_rank, (long long)cid);                \
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
    int64_t cid = comm_id_for(comm);
    uint64_t t0 = now_ns();
    int ret = PMPI_Barrier(comm);
    char extra[64];
    snprintf(extra, sizeof(extra), "type=barrier,rank=%d,commid=%lld", g_mpi_rank, (long long)cid);
    emit_span("mpi", t0, now_ns()-t0, "MPI_Barrier", extra);
    return ret;
}

_COLL(MPI_Scan,
      PMPI_Scan(sendbuf, recvbuf, count, datatype, op, comm),
      "scan", (size_t)count * dtype_size(datatype),
      const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
      MPI_Op op, MPI_Comm comm)

_COLL(MPI_Exscan,
      PMPI_Exscan(sendbuf, recvbuf, count, datatype, op, comm),
      "exscan", (size_t)count * dtype_size(datatype),
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

int MPI_Accumulate(const void *origin_addr, int origin_count,
                   MPI_Datatype origin_datatype, int target_rank,
                   MPI_Aint target_disp, int target_count,
                   MPI_Datatype target_datatype, MPI_Op op, MPI_Win win) {
    size_t nb = (size_t)origin_count * dtype_size(origin_datatype);
    char extra[128];
    snprintf(extra, sizeof(extra), "type=accumulate,bytes=%zu,rank=%d,peer=%d",
             nb, g_mpi_rank, target_rank);
    uint64_t t0 = now_ns();
    int ret = PMPI_Accumulate(origin_addr, origin_count, origin_datatype,
                               target_rank, target_disp, target_count,
                               target_datatype, op, win);
    emit_span("mpi", t0, now_ns()-t0, "MPI_Accumulate", extra);
    return ret;
}

/* ── Non-blocking collectives (M1) ──────────────────────────────────── */

#define _ICOLL(FNAME, PMPI_CALL, TYPE_STR, BYTES_EXPR, ...)             \
int FNAME(__VA_ARGS__, MPI_Request *request) {                           \
    size_t nb = (BYTES_EXPR);                                            \
    int64_t cid = comm_id_for(comm);                                     \
    uint64_t t0 = now_ns();                                              \
    int ret = PMPI_CALL;                                                  \
    uint64_t req_id = (ret == MPI_SUCCESS && request &&                  \
                       *request != MPI_REQUEST_NULL)                     \
                      ? req_register(*request, TYPE_STR, -1, 0, nb, 0) : 0;\
    char extra[192];                                                      \
    if (req_id)                                                           \
        snprintf(extra, sizeof(extra),                                   \
                 "type=%s,bytes=%zu,rank=%d,commid=%lld,req_id=%llu",    \
                 TYPE_STR, nb, g_mpi_rank, (long long)cid,               \
                 (unsigned long long)req_id);                            \
    else                                                                  \
        snprintf(extra, sizeof(extra),                                   \
                 "type=%s,bytes=%zu,rank=%d,commid=%lld",                \
                 TYPE_STR, nb, g_mpi_rank, (long long)cid);              \
    emit_span("mpi", t0, now_ns()-t0, #FNAME, extra);                   \
    return ret;                                                           \
}

_ICOLL(MPI_Ibcast,
       PMPI_Ibcast(buffer, count, datatype, root, comm, request),
       "ibcast", (size_t)count * dtype_size(datatype),
       void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm)

_ICOLL(MPI_Iallreduce,
       PMPI_Iallreduce(sendbuf, recvbuf, count, datatype, op, comm, request),
       "iallreduce", (size_t)count * dtype_size(datatype),
       const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
       MPI_Op op, MPI_Comm comm)

_ICOLL(MPI_Ireduce,
       PMPI_Ireduce(sendbuf, recvbuf, count, datatype, op, root, comm, request),
       "ireduce", (size_t)count * dtype_size(datatype),
       const void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype,
       MPI_Op op, int root, MPI_Comm comm)

_ICOLL(MPI_Iallgather,
       PMPI_Iallgather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype,
                       comm, request),
       "iallgather", (size_t)sendcount * dtype_size(sendtype),
       const void *sendbuf, int sendcount, MPI_Datatype sendtype,
       void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm)

_ICOLL(MPI_Ialltoall,
       PMPI_Ialltoall(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype,
                      comm, request),
       "ialltoall", (size_t)sendcount * dtype_size(sendtype),
       const void *sendbuf, int sendcount, MPI_Datatype sendtype,
       void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm)

_ICOLL(MPI_Iscatter,
       PMPI_Iscatter(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype,
                     root, comm, request),
       "iscatter", (size_t)sendcount * dtype_size(sendtype),
       const void *sendbuf, int sendcount, MPI_Datatype sendtype,
       void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm)

_ICOLL(MPI_Igather,
       PMPI_Igather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype,
                    root, comm, request),
       "igather", (size_t)sendcount * dtype_size(sendtype),
       const void *sendbuf, int sendcount, MPI_Datatype sendtype,
       void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm)

/* ── Persistent requests (M2) ───────────────────────────────────────── */

int MPI_Send_init(const void *buf, int count, MPI_Datatype datatype,
                  int dest, int tag, MPI_Comm comm, MPI_Request *request) {
    uint64_t t0 = now_ns();
    int ret = PMPI_Send_init(buf, count, datatype, dest, tag, comm, request);
    size_t nb = (size_t)count * dtype_size(datatype);
    char extra[128];
    snprintf(extra, sizeof(extra), "type=send_init,bytes=%zu,rank=%d,peer=%d,tag=%d",
             nb, g_mpi_rank, dest, tag);
    emit_span("mpi", t0, now_ns()-t0, "MPI_Send_init", extra);
    return ret;
}

int MPI_Recv_init(void *buf, int count, MPI_Datatype datatype,
                  int source, int tag, MPI_Comm comm, MPI_Request *request) {
    uint64_t t0 = now_ns();
    int ret = PMPI_Recv_init(buf, count, datatype, source, tag, comm, request);
    size_t nb = (size_t)count * dtype_size(datatype);
    char extra[128];
    snprintf(extra, sizeof(extra), "type=recv_init,bytes=%zu,rank=%d,peer=%d,tag=%d",
             nb, g_mpi_rank, source, tag);
    emit_span("mpi", t0, now_ns()-t0, "MPI_Recv_init", extra);
    return ret;
}

int MPI_Start(MPI_Request *request) {
    MPI_Request saved = request ? *request : MPI_REQUEST_NULL;
    uint64_t t0 = now_ns();
    int ret = PMPI_Start(request);
    uint64_t req_id = (ret == MPI_SUCCESS && request && *request != MPI_REQUEST_NULL)
                      ? req_register(*request, "start", -1, 0, 0, 0) : 0;
    char extra[96];
    if (req_id)
        snprintf(extra, sizeof(extra), "type=start,rank=%d,req_id=%llu",
                 g_mpi_rank, (unsigned long long)req_id);
    else
        snprintf(extra, sizeof(extra), "type=start,rank=%d", g_mpi_rank);
    emit_span("mpi", t0, now_ns()-t0, "MPI_Start", extra);
    (void)saved;
    return ret;
}

int MPI_Startall(int count, MPI_Request requests[]) {
    uint64_t t0 = now_ns();
    int ret = PMPI_Startall(count, requests);
    char extra[64];
    snprintf(extra, sizeof(extra), "type=startall,count=%d,rank=%d", count, g_mpi_rank);
    emit_span("mpi", t0, now_ns()-t0, "MPI_Startall", extra);
    return ret;
}
