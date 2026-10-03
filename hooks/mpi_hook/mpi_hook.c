/*
 * MPI hook (PMPI interface): wraps MPI calls and forwards to PMPI_*, which
 * every conforming implementation provides. Preloaded by `hprofiler run
 * --backend mpi`, or linked alongside the program.
 *
 * Wrapped: point-to-point (Send, Ssend, Bsend, Recv, Isend, Irecv, Wait*,
 * Test*, Cancel), non-blocking collectives (Ibcast, Iallreduce, Ireduce,
 * Iallgather, Ialltoall, Iscatter, Igather), persistent requests
 * (Send_init, Recv_init, Start, Startall), collectives (Bcast, Reduce,
 * Allreduce, Alltoall, Allgather, Scatter, Gather, Barrier, Scan, Exscan),
 * one-sided (Put, Get, Accumulate, Win_fence/flush/flush_all/lock/lock_all/
 * unlock/unlock_all), Init/Init_thread/Finalize, and Comm_dup/split/create
 * (only to assign commid=).
 *
 * Request linking: a posted request gets an id (sid=); completion calls
 * report the ids they completed (psid=).
 *
 * Wildcards: MPI always writes the matched source/tag into the status, so
 * the hook passes its own status buffer when the caller gives
 * MPI_STATUS(ES)_IGNORE (invisible to the caller). The resolved peer/tag is
 * reported with wildcard=1 so analyses can tell it from an exact match.
 *
 * Communicator identity: commid=0 is MPI_COMM_WORLD; communicators from
 * Comm_dup/split/create get an id assigned by their local rank 0 and
 * broadcast over the new communicator right after creation (safe: creation
 * is collective). -1 = unregistered (MPI_COMM_SELF, or an API not wrapped
 * here such as Comm_create_group or Cart_create); matching then falls back
 * to call order.
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
static int             g_mpi_rank   = -1;

/* ── Helpers ─────────────────────────────────────────────────────────── */
static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

#include "../common/hp_transport.h"
#include "../common/callstack.h"
#include "../common/codeptr_resolve.h"

/* Every MPI span/instant carries the application's call site
 * (hprofiler_append_codeptr_tag: sym=[,symfile=] or lib=,offset=) for the
 * Source tab. emit_span()/emit_instant() are macros so that
 * __builtin_return_address(0) is evaluated in the wrapper, whose caller is
 * the application. */
static void emit_tagged(const char *fmt_kind, const void *site, const char *cat,
                        uint64_t start_ns, uint64_t dur_ns, const char *name,
                        const char *extra) {
    char tag[2400] = "";
    if (site) hprofiler_append_codeptr_tag(tag, sizeof(tag), site);
    if (!extra) extra = "";
    const char *t = (*extra || tag[0] != ',') ? tag : tag + 1;
    if (fmt_kind[0] == 's')
        hp_tx_emitf("span:%s:%d:%d:%llu:%llu:%s:%s%s\n", cat, (int)hp_tx_pid(), (int)hp_tx_tid(),
                    (unsigned long long)start_ns, (unsigned long long)dur_ns, name, extra, t);
    else
        hp_tx_emitf("inst:%s:%d:%d:%llu:%s:%s%s\n", cat, (int)hp_tx_pid(), (int)hp_tx_tid(),
                    (unsigned long long)start_ns, name, extra, t);
}

static void emit_span_at(const void *site, const char *cat, uint64_t start_ns, uint64_t dur_ns,
                         const char *name, const char *extra) {
    if (!hp_tx_enabled()) return;
    emit_tagged("span", site, cat, start_ns, dur_ns, name, extra);
    emit_callstack(start_ns);
}

static void emit_instant_at(const void *site, const char *cat, const char *name, const char *extra) {
    if (!hp_tx_enabled()) return;
    emit_tagged("inst", site, cat, now_ns(), 0, name, extra);
}

#define emit_span(...)    emit_span_at(__builtin_return_address(0), __VA_ARGS__)
#define emit_instant(...) emit_instant_at(__builtin_return_address(0), __VA_ARGS__)

static void emit_ctr(const char *cat, const char *name, int64_t value, const char *unit) {
    if (!hp_tx_enabled()) return;
    hp_tx_emitf("ctr:%s:%d:%llu:%s:%lld:%s\n", cat, (int)hp_tx_pid(),
                (unsigned long long)now_ns(), name, (long long)value, unit);
}

/* Request-id lists (psid=a;b;c, rmatches=id/src/tag;...) for Waitall/
 * Waitsome/Testsome/Testall, bounded at IDLIST_CAP: entries that do not
 * fit are counted and reported as <key>_omitted=N. */
#define IDLIST_CAP 8192
typedef struct { char *p; size_t len; int omitted; } idlist_t;

static void idlist_init(idlist_t *l) { l->p = NULL; l->len = 0; l->omitted = 0; }

__attribute__((format(printf, 2, 3)))
static void idlist_add(idlist_t *l, const char *fmt, ...) {
    if (!l->p) {
        l->p = (char *)malloc(IDLIST_CAP);
        if (!l->p) { l->omitted++; return; }
        l->p[0] = '\0';
    }
    char item[96];
    va_list ap;
    va_start(ap, fmt);
    int w = vsnprintf(item, sizeof(item), fmt, ap);
    va_end(ap);
    if (w <= 0 || (size_t)w >= sizeof(item) || l->len + (size_t)w + 2 > IDLIST_CAP) {
        l->omitted++;
        return;
    }
    if (l->len) l->p[l->len++] = ';';
    memcpy(l->p + l->len, item, (size_t)w + 1);
    l->len += (size_t)w;
}

/* Append ",<key>=<list>[,<key>_omitted=N]" to the heap tag string *x
 * (grown as needed); frees the list. */
static void idlist_append_tag(char **x, size_t *xlen, const char *key, idlist_t *l) {
    if (l->len || l->omitted) {
        size_t need = *xlen + strlen(key) * 2 + l->len + 48;
        char *nx = (char *)realloc(*x, need);
        if (nx) {
            *x = nx;
            int w = 0;
            if (l->len)
                w = snprintf(nx + *xlen, need - *xlen, ",%s=%s", key, l->p);
            if (w > 0) *xlen += (size_t)w;
            if (l->omitted) {
                w = snprintf(nx + *xlen, need - *xlen, ",%s_omitted=%d", key, l->omitted);
                if (w > 0) *xlen += (size_t)w;
            }
        }
    }
    free(l->p);
    idlist_init(l);
}

static char *xstrdup_tags(const char *s, size_t *len) {
    *len = strlen(s);
    char *x = (char *)malloc(*len + 1);
    if (x) memcpy(x, s, *len + 1);
    return x;
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

/* Call only on a communicator that was just collectively created (every
 * member reaches this together). The id packs the bootstrapping process's
 * MPI_COMM_WORLD rank with its local sequence number: a sequence number
 * alone would collide between unrelated communicators bootstrapped by
 * different world ranks (e.g. the two halves of a Comm_split). */
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

/* ── Async request cross-linking table ──────────────────────────────── */
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
    /* Table full (REQ_TABLE_CAP outstanding requests): return 0, which call
     * sites treat as "no sid=" -- an id that can never be looked up would
     * leave its completion's psid= dangling. */
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

/* ── Clock offset vs rank 0 (HPROFILER_CLOCK_SYNC=1, off by default) ─────
 * Each node has its own collector and CLOCK_MONOTONIC epoch, so per-node
 * traces need an offset to be merged (src/analysis/multinode.py). Once, in
 * MPI_Init, each rank R pings rank 0 (Cristian's algorithm): R sends at T1,
 * rank 0 receives at T2 and replies at T3, R receives at T4. Assuming
 * symmetric latency:
 *   round_trip  = T4 - T1
 *   offset      = (T2+T3)/2 - (T1 + round_trip/2)   (rank 0's clock minus R's)
 *   error_bound = round_trip / 2
 * so t_rank0 = t_R + offset. Rank 0 serves the ranks one after another.
 * Off by default because it adds a blocking exchange to MPI_Init. Not yet
 * run with 2+ real ranks; the arithmetic is tested in tests/test_multinode.py. */
#define HPROFILER_CLOCK_SYNC_TAG 30001  /* below the guaranteed minimum
                                          * MPI_TAG_UB (32767); may collide
                                          * with an application tag */

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
        hp_tx_init("mpi");
        cs_init();
        clock_sync_if_requested();
    }
    return ret;
}

int MPI_Init_thread(int *argc, char ***argv, int required, int *provided) {
    int ret = PMPI_Init_thread(argc, argv, required, provided);
    if (ret == MPI_SUCCESS) {
        PMPI_Comm_rank(MPI_COMM_WORLD, &g_mpi_rank);
        hp_tx_init("mpi");
        cs_init();
        clock_sync_if_requested();
    }
    return ret;
}

/* Drain this rank's buffered events while it is certainly alive: launchers
 * may kill ranks soon after MPI_Finalize returns. Data already sent stays in
 * the collector's receive queue even if this process dies. Later events go
 * out synchronously; the destructor sends the closing status again. */
int MPI_Finalize(void) {
    hp_tx_shutdown(1);
    return PMPI_Finalize();
}

__attribute__((constructor)) static void mpi_hook_init(void) { hp_tx_init("mpi"); }
__attribute__((destructor)) static void mpi_hook_fini(void) { hp_tx_shutdown(1); }

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
    /* Own status buffer when the caller passes MPI_STATUS_IGNORE, so a
     * wildcard match is always resolvable (see the file header). */
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
    /* A wildcard Irecv's peer/tag are unknown until completion (rpeer=/
     * rtag= on the completing call); this span carries wildcard=1. */
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
    /* Save the handles before PMPI_Waitall nulls them (heap: arrays can be
     * large). */
    MPI_Request *saved = (count > 0)
        ? (MPI_Request *)malloc((size_t)count * sizeof(MPI_Request)) : NULL;
    if (saved)
        for (int i = 0; i < count; i++) saved[i] = requests[i];
    /* Always our own statuses, so wildcard receives resolve even with
     * MPI_STATUSES_IGNORE. */
    MPI_Status *use_statuses = (count > 0)
        ? (MPI_Status *)malloc((size_t)count * sizeof(MPI_Status)) : NULL;
    uint64_t t0 = now_ns();
    int ret = PMPI_Waitall(count, requests, use_statuses ? use_statuses : statuses);
    uint64_t t1 = now_ns();
    /* Collect req IDs for cross-linking, and resolved peer/tag for any
     * wildcard recvs among them, into semicolon-separated lists. */
    idlist_t psids, rmatches;
    idlist_init(&psids);
    idlist_init(&rmatches);
    if (saved) {
        for (int i = 0; i < count; i++) {
            uint64_t rid = 0; int wildcard = 0;
            if (req_lookup(saved[i], &rid, &wildcard) && rid) {
                idlist_add(&psids, "%llu", (unsigned long long)rid);
                if (wildcard && use_statuses && ret == MPI_SUCCESS) {
                    /* '/' inside the triple: a ':' in a tag value would be
                     * taken for the record's name/tag boundary (the parser
                     * splits at the last ':'). */
                    idlist_add(&rmatches, "%llu/%d/%d", (unsigned long long)rid,
                               use_statuses[i].MPI_SOURCE, use_statuses[i].MPI_TAG);
                }
            }
        }
        free(saved);
    }
    free(use_statuses);
    char head[96];
    snprintf(head, sizeof(head), "type=waitall,count=%d,rank=%d", count, g_mpi_rank);
    size_t xlen;
    char *x = xstrdup_tags(head, &xlen);
    idlist_append_tag(&x, &xlen, "psid", &psids);
    idlist_append_tag(&x, &xlen, "rmatches", &rmatches);
    emit_span("mpi", t0, t1-t0, "MPI_Waitall", x ? x : head);
    free(x);
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
    uint64_t t1 = now_ns();
    idlist_t psids, rmatches;
    idlist_init(&psids);
    idlist_init(&rmatches);
    if (ret == MPI_SUCCESS && saved && outcount && indices &&
        *outcount != MPI_UNDEFINED) {
        for (int k = 0; k < *outcount; k++) {
            int idx = indices[k];
            if (idx < 0 || idx >= incount) continue;
            uint64_t rid = 0; int wildcard = 0;
            if (req_lookup(saved[idx], &rid, &wildcard) && rid) {
                idlist_add(&psids, "%llu", (unsigned long long)rid);
                if (wildcard && use_statuses) {
                    /* '/' not ':' -- see the matching comment in MPI_Waitall. */
                    idlist_add(&rmatches, "%llu/%d/%d", (unsigned long long)rid,
                               use_statuses[k].MPI_SOURCE, use_statuses[k].MPI_TAG);
                }
            }
        }
    }
    free(saved);
    free(use_statuses);
    char head[128];
    snprintf(head, sizeof(head), "type=waitsome,incount=%d,outcount=%d,rank=%d",
             incount, (ret == MPI_SUCCESS && outcount) ? *outcount : -1, g_mpi_rank);
    size_t xlen;
    char *x = xstrdup_tags(head, &xlen);
    idlist_append_tag(&x, &xlen, "psid", &psids);
    idlist_append_tag(&x, &xlen, "rmatches", &rmatches);
    emit_span("mpi", t0, t1-t0, "MPI_Waitsome", x ? x : head);
    free(x);
    return ret;
}

/* ── Test* ───────────────────────────────────────────────────────────────
 * Instants, not spans: the signal is whether completion was observed.
 * flag=0 records that the program polled and found nothing ready. */

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
    idlist_t psids;
    idlist_init(&psids);
    if (ret == MPI_SUCCESS && saved && outcount && indices && *outcount != MPI_UNDEFINED) {
        for (int k = 0; k < *outcount; k++) {
            int idx = indices[k];
            if (idx < 0 || idx >= incount) continue;
            uint64_t rid = 0;
            if (req_lookup(saved[idx], &rid, NULL) && rid)
                idlist_add(&psids, "%llu", (unsigned long long)rid);
        }
    }
    free(saved);
    free(use_statuses);
    char head[128];
    snprintf(head, sizeof(head), "type=testsome,incount=%d,outcount=%d,rank=%d",
             incount, (ret == MPI_SUCCESS && outcount) ? *outcount : -1, g_mpi_rank);
    size_t xlen;
    char *x = xstrdup_tags(head, &xlen);
    idlist_append_tag(&x, &xlen, "psid", &psids);
    emit_instant("mpi", "MPI_Testsome", x ? x : head);
    free(x);
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
    idlist_t psids;
    idlist_init(&psids);
    if (ret == MPI_SUCCESS && flag && *flag && saved) {
        for (int i = 0; i < count; i++) {
            uint64_t rid = 0;
            if (req_lookup(saved[i], &rid, NULL) && rid)
                idlist_add(&psids, "%llu", (unsigned long long)rid);
        }
    }
    free(saved);
    free(use_statuses);
    char head[128];
    snprintf(head, sizeof(head), "type=testall,flag=%d,count=%d,rank=%d",
             (ret == MPI_SUCCESS && flag) ? *flag : -1, count, g_mpi_rank);
    size_t xlen;
    char *x = xstrdup_tags(head, &xlen);
    idlist_append_tag(&x, &xlen, "psid", &psids);
    emit_instant("mpi", "MPI_Testall", x ? x : head);
    free(x);
    return ret;
}

/* ── Cancellation ────────────────────────────────────────────────────────
 * A cancelled request never completes normally; the instant gives it an
 * explicit end state. */
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
    char extra[192];                                                    \
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
    char extra[96];
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

/* ── One-sided synchronization ──────────────────────────────────────────
 * Put/Get/Accumulate may return before the transfer completes; completion
 * is guaranteed only by these synchronization calls, so their spans carry
 * the real RMA wait. Category "mpi", like MPI_Barrier. */

int MPI_Win_fence(int assert, MPI_Win win) {
    char extra[32]; snprintf(extra, sizeof(extra), "type=win_fence,rank=%d", g_mpi_rank);
    uint64_t t0 = now_ns();
    int ret = PMPI_Win_fence(assert, win);
    emit_span("mpi", t0, now_ns()-t0, "MPI_Win_fence", extra);
    return ret;
}

int MPI_Win_flush(int rank, MPI_Win win) {
    char extra[64];
    snprintf(extra, sizeof(extra), "type=win_flush,rank=%d,peer=%d", g_mpi_rank, rank);
    uint64_t t0 = now_ns();
    int ret = PMPI_Win_flush(rank, win);
    emit_span("mpi", t0, now_ns()-t0, "MPI_Win_flush", extra);
    return ret;
}

int MPI_Win_flush_all(MPI_Win win) {
    char extra[32]; snprintf(extra, sizeof(extra), "type=win_flush_all,rank=%d", g_mpi_rank);
    uint64_t t0 = now_ns();
    int ret = PMPI_Win_flush_all(win);
    emit_span("mpi", t0, now_ns()-t0, "MPI_Win_flush_all", extra);
    return ret;
}

int MPI_Win_lock(int lock_type, int rank, int assert, MPI_Win win) {
    char extra[64];
    snprintf(extra, sizeof(extra), "type=win_lock,rank=%d,peer=%d", g_mpi_rank, rank);
    uint64_t t0 = now_ns();
    int ret = PMPI_Win_lock(lock_type, rank, assert, win);
    emit_span("mpi", t0, now_ns()-t0, "MPI_Win_lock", extra);
    return ret;
}

int MPI_Win_lock_all(int assert, MPI_Win win) {
    char extra[32]; snprintf(extra, sizeof(extra), "type=win_lock_all,rank=%d", g_mpi_rank);
    uint64_t t0 = now_ns();
    int ret = PMPI_Win_lock_all(assert, win);
    emit_span("mpi", t0, now_ns()-t0, "MPI_Win_lock_all", extra);
    return ret;
}

int MPI_Win_unlock(int rank, MPI_Win win) {
    char extra[64];
    snprintf(extra, sizeof(extra), "type=win_unlock,rank=%d,peer=%d", g_mpi_rank, rank);
    uint64_t t0 = now_ns();
    int ret = PMPI_Win_unlock(rank, win);
    emit_span("mpi", t0, now_ns()-t0, "MPI_Win_unlock", extra);
    return ret;
}

int MPI_Win_unlock_all(MPI_Win win) {
    char extra[32]; snprintf(extra, sizeof(extra), "type=win_unlock_all,rank=%d", g_mpi_rank);
    uint64_t t0 = now_ns();
    int ret = PMPI_Win_unlock_all(win);
    emit_span("mpi", t0, now_ns()-t0, "MPI_Win_unlock_all", extra);
    return ret;
}

/* ── Non-blocking collectives ───────────────────────────────────────── */

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

/* ── Persistent requests ────────────────────────────────────────────── */

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
