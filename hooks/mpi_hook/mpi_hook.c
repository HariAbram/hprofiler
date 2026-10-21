/*
 * MPI hook (PMPI interface): wraps MPI calls and forwards to PMPI_*, which
 * every conforming implementation provides. Preloaded by `hprofiler run
 * --backend mpi`, or linked alongside the program.
 *
 * Wrapped: point-to-point (Send, Ssend, Bsend, Rsend, Recv, Isend, Issend,
 * Ibsend, Irsend, Irecv, Sendrecv, Sendrecv_replace, Probe, Iprobe,
 * Mprobe/Improbe/Mrecv/Imrecv (as markers), Wait*, Test*, Cancel),
 * non-blocking collectives (Ibarrier, Ibcast, Iallreduce, Ireduce,
 * Iallgather, Ialltoall, Iscatter, Igather), persistent requests
 * (Send_init, Bsend_init, Ssend_init, Rsend_init, Recv_init, Start, Startall), collectives (Bcast, Reduce,
 * Allreduce, Alltoall, Allgather, Scatter, Gather, Allgatherv, Alltoallv,
 * Gatherv, Scatterv, Reduce_scatter, Reduce_scatter_block, Barrier, Scan,
 * Exscan), one-sided (Put, Get, Accumulate, Win_fence/flush/flush_all/lock/
 * lock_all/unlock/unlock_all), Init/Init_thread/Finalize, and
 * Comm_dup/split/create/free (only for commid= and rank translation).
 *
 * Request linking: a posted request gets an id (sid=); completion calls
 * report the ids they completed (psid=).
 *
 * Statuses: a status (array) the caller passes goes to PMPI_* unchanged.
 * Only when the caller passes MPI_STATUS(ES)_IGNORE and the hook needs a
 * status -- a wildcard receive's matched source/tag, or MPI_Test_cancelled
 * for a request with MPI_Cancel pending -- does it pass its own buffer
 * (invisible to the caller). Resolved wildcards are reported with
 * wildcard=1 so analyses can tell them from an exact match; cancelled
 * requests are reported (cancelled=1 / cpsid=) instead of a match.
 *
 * Ranks: rank= is the MPI_COMM_WORLD rank; peer= is the call's
 * communicator-local rank and wpeer= (rwpeer=/rwmatches= when resolved at
 * completion) its MPI_COMM_WORLD rank, so analyses can pair messages on
 * sub-communicators. null=1 marks MPI_PROC_NULL.
 *
 * Communicator identity: commid=0 is MPI_COMM_WORLD; communicators from
 * Comm_dup/split/create get an id assigned by their local rank 0 and
 * broadcast over the new communicator right after creation (safe: creation
 * is collective). -1 = unregistered (MPI_COMM_SELF, or an API not wrapped
 * here such as Comm_create_group or Cart_create).
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
#include "../common/hp_ids.h"
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

/* A freed handle can be reused by the next communicator the library
 * creates; forgetting it keeps that one from inheriting this one's id. */
static void comm_id_forget(MPI_Comm comm) {
    pthread_mutex_lock(&g_comm_mutex);
    for (int i = 0; i < g_comm_n; i++) {
        if (g_comm_table[i].comm == comm) {
            g_comm_table[i] = g_comm_table[--g_comm_n];
            break;
        }
    }
    pthread_mutex_unlock(&g_comm_mutex);
}

/* ── World-rank translation ─────────────────────────────────────────────
 * peer= is a rank of the call's communicator; wpeer= is the same process's
 * MPI_COMM_WORLD rank (for an intercommunicator: a rank of the remote
 * group). The point-to-point matcher keys on world ranks plus commid, so
 * sub-communicators with renumbered ranks match too. One table per
 * communicator, built on first use (MPI_Comm_group or MPI_Comm_remote_group,
 * then MPI_Group_translate_ranks) and dropped in MPI_Comm_free. A wildcard
 * receive holds a reference until it completes, so freeing its
 * communicator first (legal in MPI) cannot invalidate the translation of
 * its resolved source. */
typedef struct {
    MPI_Comm comm;
    int      size;
    int     *world;     /* world[local]: world rank, or MPI_UNDEFINED */
    int      refs;      /* 1 while cached + one per holder */
} Xlate;

#define XLATE_CAP 256
static Xlate          *g_xlate[XLATE_CAP];
static int             g_xlate_n     = 0;
static pthread_mutex_t g_xlate_mutex = PTHREAD_MUTEX_INITIALIZER;

static void xlate_destroy(Xlate *x) {
    if (!x) return;
    free(x->world);
    free(x);
}

static Xlate *xlate_build(MPI_Comm comm) {
    int inter = 0;
    if (PMPI_Comm_test_inter(comm, &inter) != MPI_SUCCESS) return NULL;
    MPI_Group g, wg;
    if ((inter ? PMPI_Comm_remote_group(comm, &g) : PMPI_Comm_group(comm, &g)) != MPI_SUCCESS)
        return NULL;
    if (PMPI_Comm_group(MPI_COMM_WORLD, &wg) != MPI_SUCCESS) {
        PMPI_Group_free(&g);
        return NULL;
    }
    int n = 0;
    PMPI_Group_size(g, &n);
    Xlate *x = (Xlate *)calloc(1, sizeof(*x));
    int *in = (n > 0) ? (int *)malloc((size_t)n * sizeof(int)) : NULL;
    if (x && n > 0) x->world = (int *)malloc((size_t)n * sizeof(int));
    int ok = x && (n == 0 || (in && x->world));
    if (ok && n > 0) {
        for (int i = 0; i < n; i++) in[i] = i;
        ok = PMPI_Group_translate_ranks(g, n, in, wg, x->world) == MPI_SUCCESS;
    }
    free(in);
    PMPI_Group_free(&g);
    PMPI_Group_free(&wg);
    if (!ok) {
        xlate_destroy(x);
        return NULL;
    }
    x->comm = comm;
    x->size = n;
    x->refs = 1;
    return x;
}

/* The communicator's table with one more reference (release with
 * xlate_release), or NULL when it cannot be built. */
static Xlate *xlate_acquire(MPI_Comm comm) {
    if (comm == MPI_COMM_NULL) return NULL;
    pthread_mutex_lock(&g_xlate_mutex);
    for (int i = 0; i < g_xlate_n; i++) {
        if (g_xlate[i]->comm == comm) {
            Xlate *x = g_xlate[i];
            x->refs++;
            pthread_mutex_unlock(&g_xlate_mutex);
            return x;
        }
    }
    pthread_mutex_unlock(&g_xlate_mutex);
    Xlate *x = xlate_build(comm);          /* PMPI calls: outside the lock */
    if (!x) return NULL;
    pthread_mutex_lock(&g_xlate_mutex);
    for (int i = 0; i < g_xlate_n; i++) {  /* another thread built it meanwhile */
        if (g_xlate[i]->comm == comm) {
            Xlate *have = g_xlate[i];
            have->refs++;
            pthread_mutex_unlock(&g_xlate_mutex);
            xlate_destroy(x);
            return have;
        }
    }
    if (g_xlate_n < XLATE_CAP) {
        g_xlate[g_xlate_n++] = x;
        x->refs++;                          /* the cache's reference */
    }
    pthread_mutex_unlock(&g_xlate_mutex);
    return x;
}

static void xlate_release(Xlate *x) {
    if (!x) return;
    pthread_mutex_lock(&g_xlate_mutex);
    int last = (--x->refs == 0);
    pthread_mutex_unlock(&g_xlate_mutex);
    if (last) xlate_destroy(x);
}

static void xlate_forget(MPI_Comm comm) {
    Xlate *drop = NULL;
    pthread_mutex_lock(&g_xlate_mutex);
    for (int i = 0; i < g_xlate_n; i++) {
        if (g_xlate[i]->comm == comm) {
            drop = g_xlate[i];
            g_xlate[i] = g_xlate[--g_xlate_n];
            break;
        }
    }
    pthread_mutex_unlock(&g_xlate_mutex);
    xlate_release(drop);
}

static int xlate_world(const Xlate *x, int local) {
    if (!x || local < 0 || local >= x->size) return -1;
    int w = x->world[local];
    return (w == MPI_UNDEFINED || w < 0) ? -1 : w;
}

/* World rank of `local` in `comm`; -1 for MPI_ANY_SOURCE, MPI_PROC_NULL or
 * when it cannot be translated (then no wpeer= is emitted). */
static int world_rank_of(MPI_Comm comm, int local) {
    if (local < 0) return -1;
    if (comm == MPI_COMM_WORLD) return local;
    Xlate *x = xlate_acquire(comm);
    int w = xlate_world(x, local);
    xlate_release(x);
    return w;
}

/* ",wpeer=<w>" (or ",null=1" for MPI_PROC_NULL) for tag strings. */
static void peer_tags(char *buf, size_t cap, MPI_Comm comm, int local) {
    buf[0] = '\0';
    if (local == MPI_PROC_NULL) {
        snprintf(buf, cap, ",null=1");
        return;
    }
    int w = world_rank_of(comm, local);
    if (w >= 0) snprintf(buf, cap, ",wpeer=%d", w);
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

int MPI_Comm_free(MPI_Comm *comm) {
    MPI_Comm c = comm ? *comm : MPI_COMM_NULL;
    int ret = PMPI_Comm_free(comm);
    if (ret == MPI_SUCCESS && c != MPI_COMM_NULL) {
        comm_id_forget(c);
        xlate_forget(c);
    }
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
    int         wildcard;     /* an Irecv posted with ANY_SOURCE/ANY_TAG */
    int         cancel_req;   /* MPI_Cancel was called on it */
    int         world_comm;   /* posted on MPI_COMM_WORLD (source == world rank) */
    Xlate      *xl;           /* wildcard receive on another comm: its translation */
} ReqRec;

/* What a completion wrapper needs about one request. */
typedef struct {
    uint64_t id;
    int      wildcard;
    int      cancel_req;
    int      world_comm;
    Xlate   *xl;              /* owned by the caller after req_take */
} ReqInfo;

static ReqRec          g_req_table[REQ_TABLE_CAP];
static int             g_req_n      = 0;
static uint64_t        g_req_seq    = 1;
static pthread_mutex_t g_req_mutex  = PTHREAD_MUTEX_INITIALIZER;

/* Registers a posted request and returns its id (0 when the table is full;
 * call sites then emit no sid=, since an id that can never be looked up
 * would leave its completion's psid= dangling). Takes ownership of `xl`. */
static uint64_t req_register_ex(MPI_Request req, const char *type, int peer, int tag,
                                size_t bytes, int wildcard, MPI_Comm comm, Xlate *xl) {
    pthread_mutex_lock(&g_req_mutex);
    uint64_t id = 0;
    if (g_req_n < REQ_TABLE_CAP) {
        /* pid in the high bits: ids stay unique across the ranks of one
         * trace (analyses also scope them by process for older traces). */
        id = hp_span_id(HP_ID_MPI, (uint32_t)hp_tx_pid(), (uint32_t)(g_req_seq++ & 0xffffffffull));
        ReqRec *r     = &g_req_table[g_req_n++];
        r->req        = req;
        r->id         = id;
        r->peer       = peer;
        r->tag        = tag;
        r->bytes      = bytes;
        r->wildcard   = wildcard;
        r->cancel_req = 0;
        r->world_comm = (comm == MPI_COMM_WORLD);
        r->xl         = xl;
        xl            = NULL;
        strncpy(r->type, type, 15); r->type[15] = '\0';
    }
    pthread_mutex_unlock(&g_req_mutex);
    xlate_release(xl);                  /* not stored (table full) */
    return id;
}

static uint64_t req_register(MPI_Request req, const char *type,
                             int peer, int tag, size_t bytes, int wildcard) {
    return req_register_ex(req, type, peer, tag, bytes, wildcard, MPI_COMM_NULL, NULL);
}

static void req_fill(const ReqRec *r, ReqInfo *out) {
    out->id         = r->id;
    out->wildcard   = r->wildcard;
    out->cancel_req = r->cancel_req;
    out->world_comm = r->world_comm;
    out->xl         = r->xl;
}

/* Removes the entry and hands its fields (and its translation reference)
 * to the caller. Returns 1 if found. */
static int req_take(MPI_Request req, ReqInfo *out) {
    if (req == MPI_REQUEST_NULL) return 0;
    pthread_mutex_lock(&g_req_mutex);
    for (int i = 0; i < g_req_n; i++) {
        if (g_req_table[i].req == req) {
            req_fill(&g_req_table[i], out);
            g_req_table[i] = g_req_table[--g_req_n];
            pthread_mutex_unlock(&g_req_mutex);
            return 1;
        }
    }
    pthread_mutex_unlock(&g_req_mutex);
    return 0;
}

/* Non-destructive lookup (a Test* that reports flag=0 leaves the request
 * pending). out->xl stays owned by the table. */
static int req_peek(MPI_Request req, ReqInfo *out) {
    if (req == MPI_REQUEST_NULL) return 0;
    pthread_mutex_lock(&g_req_mutex);
    for (int i = 0; i < g_req_n; i++) {
        if (g_req_table[i].req == req) {
            req_fill(&g_req_table[i], out);
            out->xl = NULL;
            pthread_mutex_unlock(&g_req_mutex);
            return 1;
        }
    }
    pthread_mutex_unlock(&g_req_mutex);
    return 0;
}

static void req_mark_cancel(MPI_Request req) {
    pthread_mutex_lock(&g_req_mutex);
    for (int i = 0; i < g_req_n; i++) {
        if (g_req_table[i].req == req) {
            g_req_table[i].cancel_req = 1;
            break;
        }
    }
    pthread_mutex_unlock(&g_req_mutex);
}

/* Whether completing any of these requests needs its status: a wildcard
 * receive (resolved source/tag) or a request with MPI_Cancel pending
 * (MPI_Test_cancelled). Only then does a wrapper substitute its own status
 * array for MPI_STATUS(ES)_IGNORE. */
static int reqs_need_status(const MPI_Request *reqs, int n) {
    if (!reqs || n <= 0) return 0;
    int need = 0;
    pthread_mutex_lock(&g_req_mutex);
    for (int k = 0; k < n && !need; k++) {
        if (reqs[k] == MPI_REQUEST_NULL) continue;
        for (int i = 0; i < g_req_n; i++) {
            if (g_req_table[i].req == reqs[k]) {
                need = g_req_table[i].wildcard || g_req_table[i].cancel_req;
                break;
            }
        }
    }
    pthread_mutex_unlock(&g_req_mutex);
    return need;
}

static MPI_Request *save_requests(const MPI_Request *reqs, int n) {
    if (!reqs || n <= 0) return NULL;
    MPI_Request *s = (MPI_Request *)malloc((size_t)n * sizeof(MPI_Request));
    if (s) memcpy(s, reqs, (size_t)n * sizeof(MPI_Request));
    return s;
}

/* ── Completion bookkeeping ─────────────────────────────────────────────
 * Every completion wrapper reports, per request the hook saw posted:
 *   psid=<id>                        it completed
 *   cancelled=1 / cpsid=<id;...>     MPI_Test_cancelled says it was
 *                                    cancelled: no match is reported
 *   rpeer=/rtag=/rwpeer= (single) or rmatches=<id>/<src>/<tag> plus
 *   rwmatches=<id>/<world src> (multi): a wildcard receive's resolved source
 *   (comm-local and world) and tag.
 * `st` is NULL when no status is available (caller ignored statuses and no
 * substitute was needed) or the status reports an error. */
typedef struct {
    idlist_t psids, rmatches, rwmatches, cancelled;
} compl_t;

static void compl_init(compl_t *c) {
    idlist_init(&c->psids);
    idlist_init(&c->rmatches);
    idlist_init(&c->rwmatches);
    idlist_init(&c->cancelled);
}

static int status_cancelled(const MPI_Status *st) {
    int flag = 0;
    if (!st) return 0;
    if (PMPI_Test_cancelled(st, &flag) != MPI_SUCCESS) return 0;
    return flag != 0;
}

static int resolved_world(const ReqInfo *ri, int source) {
    if (source < 0) return -1;
    if (ri->world_comm) return source;
    return xlate_world(ri->xl, source);
}

static void compl_add(compl_t *c, MPI_Request saved, const MPI_Status *st) {
    ReqInfo ri;
    if (!req_take(saved, &ri)) return;
    if (ri.id) {
        idlist_add(&c->psids, "%llu", (unsigned long long)ri.id);
        if (status_cancelled(st)) {
            idlist_add(&c->cancelled, "%llu", (unsigned long long)ri.id);
        } else if (ri.wildcard && st) {
            /* '/' inside the triple: a ':' in a tag value would be taken for
             * the record's name/tag boundary (the parser splits at the last
             * ':'). */
            idlist_add(&c->rmatches, "%llu/%d/%d", (unsigned long long)ri.id,
                       st->MPI_SOURCE, st->MPI_TAG);
            int w = resolved_world(&ri, st->MPI_SOURCE);
            if (w >= 0)
                idlist_add(&c->rwmatches, "%llu/%d", (unsigned long long)ri.id, w);
        }
    }
    xlate_release(ri.xl);
}

/* Emits head + the lists as one span or instant. */
static void compl_emit(compl_t *c, const void *site, int span, uint64_t t0, uint64_t t1,
                       const char *name, const char *head) {
    int ncancel = c->cancelled.len ? 1 : 0;
    for (size_t i = 0; i < c->cancelled.len; i++) if (c->cancelled.p[i] == ';') ncancel++;
    size_t xlen;
    char *x = xstrdup_tags(head, &xlen);
    idlist_append_tag(&x, &xlen, "psid", &c->psids);
    idlist_append_tag(&x, &xlen, "rmatches", &c->rmatches);
    idlist_append_tag(&x, &xlen, "rwmatches", &c->rwmatches);
    if (ncancel || c->cancelled.omitted) {
        char tail[48];
        snprintf(tail, sizeof(tail), ",cancelled=%d", ncancel + c->cancelled.omitted);
        size_t need = xlen + strlen(tail) + 1;
        char *nx = x ? (char *)realloc(x, need) : NULL;
        if (nx) { x = nx; memcpy(x + xlen, tail, strlen(tail) + 1); xlen += strlen(tail); }
    }
    idlist_append_tag(&x, &xlen, "cpsid", &c->cancelled);
    if (span) emit_span_at(site, "mpi", t0, t1 - t0, name, x ? x : head);
    else      emit_instant_at(site, "mpi", name, x ? x : head);
    free(x);
}

/* Single completions (Wait, Waitany, Test, Testany): ",psid=..[,cancelled=1 |
 * ,rpeer=..,rtag=..[,rwpeer=..]]" appended to buf. */
static void single_completion_tags(char *buf, size_t cap, MPI_Request saved, const MPI_Status *st) {
    buf[0] = '\0';
    ReqInfo ri;
    if (!req_take(saved, &ri)) return;
    if (ri.id) {
        int n = snprintf(buf, cap, ",psid=%llu", (unsigned long long)ri.id);
        if (n > 0 && (size_t)n < cap) {
            if (status_cancelled(st)) {
                snprintf(buf + n, cap - (size_t)n, ",cancelled=1");
            } else if (ri.wildcard && st) {
                int w = resolved_world(&ri, st->MPI_SOURCE);
                if (w >= 0)
                    snprintf(buf + n, cap - (size_t)n, ",rpeer=%d,rtag=%d,rwpeer=%d",
                             st->MPI_SOURCE, st->MPI_TAG, w);
                else
                    snprintf(buf + n, cap - (size_t)n, ",rpeer=%d,rtag=%d",
                             st->MPI_SOURCE, st->MPI_TAG);
            }
        }
    }
    xlate_release(ri.xl);
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
 * traces need an offset to be merged (src/analysis/multinode.py). Measured
 * twice -- in MPI_Init and in MPI_Finalize -- so the merge can apply offset
 * plus linear drift.
 *
 * Protocol (NTP-style, on a private duplicate of MPI_COMM_WORLD so no
 * application tag can interfere): rank 0 serves one rank at a time and
 * hands it a turn token first, so a rank's pings never queue behind another
 * rank's. Rank R then does HPROFILER_CLOCK_SYNC_ROUNDS (default 16) rounds:
 * R sends at T1, rank 0 receives at T2 and replies at T3, R receives at T4;
 *   delay  = (T4 - T1) - (T3 - T2)
 *   offset = ((T2 - T1) + (T3 - T4)) / 2      (rank 0's clock minus R's)
 *   bound  = delay / 2
 * and keeps the round with the smallest delay (the least queueing). So
 * t_rank0 = t_R + offset, |error| <= bound under any latency asymmetry.
 *
 * Same clock: rank 0 broadcasts its clock domain (kernel boot_id and the
 * time namespace's CLOCK_MONOTONIC offset). A rank in the same domain reads
 * the very same clock, so its offset is exactly 0: the counters say 0 (bound
 * 0) and the estimate is kept only for diagnostics in the instant.
 *
 * Emitted per rank and phase: an instant "clock_sync" (phase=init|finalize,
 * rank=, offset=, bound=, delay=, at= (local midpoint of the kept round),
 * rounds=, same_clock=, est_offset=, est_bound=, boot_id=, clocksource=);
 * rank 0's carries offset=0 as the reference. Ranks other than 0 also emit
 * the counters clock_offset_vs_rank0_ns / clock_offset_error_bound_ns /
 * clock_sync_round_trip_ns (as older hooks did). HPROFILER_CLOCK_SYNC_
 * IGNORE_BOOT_ID=1 treats every rank as a different clock (to exercise the
 * estimate on one host). */
#define HPROFILER_CLOCK_SYNC_TAG 30001

typedef struct {
    char boot_id[48];
    long long timens_mono_ns;     /* this time namespace's CLOCK_MONOTONIC offset */
} ClockDomain;

static MPI_Comm g_clock_comm = MPI_COMM_NULL;

static void read_first_line(const char *path, char *buf, size_t cap) {
    buf[0] = '\0';
    FILE *f = fopen(path, "r");
    if (!f) return;
    if (fgets(buf, (int)cap, f)) buf[strcspn(buf, "\r\n")] = '\0';
    fclose(f);
}

static void read_clock_domain(ClockDomain *d) {
    memset(d, 0, sizeof(*d));
    read_first_line("/proc/sys/kernel/random/boot_id", d->boot_id, sizeof(d->boot_id));
    FILE *f = fopen("/proc/self/timens_offsets", "r");      /* Linux >= 5.6 */
    if (f) {
        char line[128];
        while (fgets(line, sizeof(line), f)) {
            long long sec = 0, nsec = 0;
            if (sscanf(line, "monotonic %lld %lld", &sec, &nsec) == 2)
                d->timens_mono_ns = sec * 1000000000LL + nsec;
        }
        fclose(f);
    }
}

/* An environment switch: set, non-empty and not "0". */
static int env_on(const char *name) {
    const char *v = getenv(name);
    return v && v[0] && strcmp(v, "0") != 0;
}

static int clock_sync_rounds(void) {
    const char *e = getenv("HPROFILER_CLOCK_SYNC_ROUNDS");
    int n = e ? atoi(e) : 16;
    return n < 1 ? 1 : (n > 1000 ? 1000 : n);
}

/* Which point-to-point calls this capture can see, as two counters at
 * MPI_Init: mpi_version (major*100+minor of the runtime) and
 * mpi_p2p_capture_complete -- 1 when every send and receive call of the
 * runtime's MPI version is wrapped here (or recorded as a marker: matched
 * probes and receives, persistent requests), i.e. MPI 3.x; 0 on an MPI-4
 * runtime, whose large-count (_c), partitioned and MPI_Isendrecv calls this
 * hook does not wrap. The critical path's exact-match rule
 * (HPROFILER_EXACT_P2P_HIGH) relies on it. Counters, not an instant: every
 * MPI instant names the application's call site. */
static void emit_capture_info(void) {
    int version = 0, subversion = 0;
    PMPI_Get_version(&version, &subversion);
    emit_ctr("mpi", "mpi_version", (int64_t)(version * 100 + subversion), "version");
    emit_ctr("mpi", "mpi_p2p_capture_complete", version < 4 ? 1 : 0, "flag");
}

static void clock_sync_if_requested(const char *phase) {
    if (!env_on("HPROFILER_CLOCK_SYNC")) return;
    int size = 0;
    PMPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size < 2) return;
    if (g_clock_comm == MPI_COMM_NULL && PMPI_Comm_dup(MPI_COMM_WORLD, &g_clock_comm) != MPI_SUCCESS)
        return;
    MPI_Comm cc = g_clock_comm;

    ClockDomain mine, ref;
    read_clock_domain(&mine);
    ref = mine;
    int rounds = clock_sync_rounds();
    PMPI_Bcast(&ref, (int)sizeof(ref), MPI_BYTE, 0, cc);
    PMPI_Bcast(&rounds, 1, MPI_INT, 0, cc);       /* rank 0's count for everyone */
    int same = mine.boot_id[0] && strcmp(mine.boot_id, ref.boot_id) == 0
               && mine.timens_mono_ns == ref.timens_mono_ns
               && !env_on("HPROFILER_CLOCK_SYNC_IGNORE_BOOT_ID");
    char clocksource[64];
    read_first_line("/sys/devices/system/clocksource/clocksource0/current_clocksource",
                    clocksource, sizeof(clocksource));
    for (char *c = clocksource; *c; c++) if (*c == ',' || *c == ':' || *c == '=') *c = '_';
    char tags[512];

    if (g_mpi_rank == 0) {
        for (int r = 1; r < size; r++) {
            PMPI_Send(NULL, 0, MPI_BYTE, r, HPROFILER_CLOCK_SYNC_TAG, cc);      /* your turn */
            for (int k = 0; k < rounds; k++) {
                PMPI_Recv(NULL, 0, MPI_BYTE, r, HPROFILER_CLOCK_SYNC_TAG, cc, MPI_STATUS_IGNORE);
                uint64_t pair[2];
                pair[0] = now_ns();
                pair[1] = now_ns();
                PMPI_Send(pair, 2, MPI_UNSIGNED_LONG_LONG, r, HPROFILER_CLOCK_SYNC_TAG, cc);
            }
        }
        snprintf(tags, sizeof(tags),
                 "type=clock_sync,phase=%s,rank=0,offset=0,bound=0,reference=1,rounds=%d,"
                 "boot_id=%s,timens=%lld,clocksource=%s",
                 phase, rounds, mine.boot_id[0] ? mine.boot_id : "unknown", mine.timens_mono_ns,
                 clocksource[0] ? clocksource : "unknown");
        emit_instant("mpi", "clock_sync", tags);
        return;
    }

    PMPI_Recv(NULL, 0, MPI_BYTE, 0, HPROFILER_CLOCK_SYNC_TAG, cc, MPI_STATUS_IGNORE);
    uint64_t best_delay = UINT64_MAX, best_t1 = 0, best_t4 = 0;
    int64_t best_offset = 0;
    for (int k = 0; k < rounds; k++) {
        uint64_t t1 = now_ns();
        PMPI_Send(NULL, 0, MPI_BYTE, 0, HPROFILER_CLOCK_SYNC_TAG, cc);
        uint64_t pair[2] = {0, 0};
        PMPI_Recv(pair, 2, MPI_UNSIGNED_LONG_LONG, 0, HPROFILER_CLOCK_SYNC_TAG, cc, MPI_STATUS_IGNORE);
        uint64_t t4 = now_ns();
        int64_t service = (int64_t)(pair[1] - pair[0]);
        int64_t delay = (int64_t)(t4 - t1) - service;
        if (delay < 0) delay = 0;
        if ((uint64_t)delay < best_delay) {
            best_delay = (uint64_t)delay;
            best_offset = (((int64_t)pair[0] - (int64_t)t1) + ((int64_t)pair[1] - (int64_t)t4)) / 2;
            best_t1 = t1;
            best_t4 = t4;
        }
    }
    int64_t est_bound = (int64_t)(best_delay / 2) + 1;   /* +1: integer halving */
    int64_t offset = same ? 0 : best_offset;
    int64_t bound = same ? 0 : est_bound;
    uint64_t at = best_t1 + (best_t4 - best_t1) / 2;
    snprintf(tags, sizeof(tags),
             "type=clock_sync,phase=%s,rank=%d,offset=%lld,bound=%lld,delay=%llu,at=%llu,rounds=%d,"
             "same_clock=%d,est_offset=%lld,est_bound=%lld,boot_id=%s,timens=%lld,clocksource=%s",
             phase, g_mpi_rank, (long long)offset, (long long)bound, (unsigned long long)best_delay,
             (unsigned long long)at, rounds, same, (long long)best_offset, (long long)est_bound,
             mine.boot_id[0] ? mine.boot_id : "unknown", mine.timens_mono_ns,
             clocksource[0] ? clocksource : "unknown");
    emit_instant("mpi", "clock_sync", tags);
    emit_ctr("mpi", "clock_offset_vs_rank0_ns", offset, "ns");
    emit_ctr("mpi", "clock_offset_error_bound_ns", bound, "ns");
    emit_ctr("mpi", "clock_sync_round_trip_ns", (int64_t)best_delay, "ns");
}

int MPI_Init(int *argc, char ***argv) {
    int ret = PMPI_Init(argc, argv);
    if (ret == MPI_SUCCESS) {
        PMPI_Comm_rank(MPI_COMM_WORLD, &g_mpi_rank);
        hp_tx_init("mpi");
        cs_init();
        emit_capture_info();
        clock_sync_if_requested("init");
    }
    return ret;
}

int MPI_Init_thread(int *argc, char ***argv, int required, int *provided) {
    int ret = PMPI_Init_thread(argc, argv, required, provided);
    if (ret == MPI_SUCCESS) {
        PMPI_Comm_rank(MPI_COMM_WORLD, &g_mpi_rank);
        hp_tx_init("mpi");
        cs_init();
        emit_capture_info();
        clock_sync_if_requested("init");
    }
    return ret;
}

/* Drain this rank's buffered events while it is certainly alive: launchers
 * may kill ranks soon after MPI_Finalize returns. Data already sent stays in
 * the collector's receive queue even if this process dies. Later events go
 * out synchronously; the destructor sends the closing status again. */
int MPI_Finalize(void) {
    /* second measurement (drift), while every rank is still here */
    clock_sync_if_requested("finalize");
    if (g_clock_comm != MPI_COMM_NULL) PMPI_Comm_free(&g_clock_comm);
    hp_tx_shutdown(1);
    return PMPI_Finalize();
}

__attribute__((constructor)) static void mpi_hook_init(void) { hp_tx_init("mpi"); }
__attribute__((destructor)) static void mpi_hook_fini(void) { hp_tx_shutdown(1); }

/* ── Point-to-point ─────────────────────────────────────────────────── */

/* Blocking sends: one span, the destination as peer= (comm-local) and
 * wpeer= (world). */
#define _P2P(FNAME, PMPI_CALL, TYPE_STR, ...)                          \
int FNAME(__VA_ARGS__) {                                                \
    char extra[256], pt[48];                                            \
    size_t nb = (size_t)count * dtype_size(datatype);                  \
    int64_t cid = comm_id_for(comm);                                    \
    peer_tags(pt, sizeof(pt), comm, peer_rank);                         \
    snprintf(extra, sizeof(extra),                                      \
             "type=%s,bytes=%zu,rank=%d,peer=%d,tag=%d,commid=%lld%s", \
             TYPE_STR, nb, g_mpi_rank, peer_rank, tag, (long long)cid, pt); \
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

_P2P(MPI_Rsend,
     PMPI_Rsend(buf, count, datatype, dest, tag, comm),
     "rsend",
     const void *buf, int count, MPI_Datatype datatype,
     int dest, int tag, MPI_Comm comm)

#undef peer_rank

/* The tags of a receive whose source/tag are known after the call:
 * peer= is the resolved comm-local source, wpeer= its world rank. */
static void recv_tags(char *buf, size_t cap, const char *type, size_t nb, MPI_Comm comm,
                      int source, int tag, int wildcard) {
    char pt[48];
    peer_tags(pt, sizeof(pt), comm, source);
    snprintf(buf, cap, "type=%s,bytes=%zu,rank=%d,peer=%d,tag=%d,commid=%lld%s%s",
             type, nb, g_mpi_rank, source, tag, (long long)comm_id_for(comm), pt,
             wildcard ? ",wildcard=1" : "");
}

int MPI_Recv(void *buf, int count, MPI_Datatype datatype,
             int source, int tag, MPI_Comm comm, MPI_Status *status) {
    size_t nb = (size_t)count * dtype_size(datatype);
    int wildcard = (source == MPI_ANY_SOURCE) || (tag == MPI_ANY_TAG);
    /* Own status buffer when the caller passes MPI_STATUS_IGNORE, so a
     * wildcard match is always resolvable (see the file header). */
    MPI_Status local_status;
    MPI_Status *use_status = (status == MPI_STATUS_IGNORE) ? &local_status : status;
    uint64_t t0 = now_ns();
    int ret = PMPI_Recv(buf, count, datatype, source, tag, comm, use_status);
    uint64_t t1 = now_ns();
    int resolved_source = source, resolved_tag = tag;
    if (ret == MPI_SUCCESS && wildcard) {
        resolved_source = use_status->MPI_SOURCE;
        resolved_tag    = use_status->MPI_TAG;
    }
    char extra[288];
    recv_tags(extra, sizeof(extra), "recv", nb, comm, resolved_source, resolved_tag, wildcard);
    emit_span("mpi", t0, t1 - t0, "MPI_Recv", extra);
    return ret;
}

/* Non-blocking sends: a request id (sid=) links the call to its completion. */
static int isend_common(const void *site, const char *name, const char *type, int ret,
                        uint64_t t0, uint64_t t1, MPI_Request *request, size_t nb,
                        int dest, int tag, MPI_Comm comm) {
    uint64_t req_id = (ret == MPI_SUCCESS && request && *request != MPI_REQUEST_NULL)
                      ? req_register(*request, type, dest, tag, nb, 0) : 0;
    char extra[288], pt[48], sid[40] = "";
    peer_tags(pt, sizeof(pt), comm, dest);
    if (req_id) snprintf(sid, sizeof(sid), ",sid=%llu", (unsigned long long)req_id);
    snprintf(extra, sizeof(extra), "type=%s,bytes=%zu,rank=%d,peer=%d,tag=%d,commid=%lld%s%s",
             type, nb, g_mpi_rank, dest, tag, (long long)comm_id_for(comm), pt, sid);
    emit_span_at(site, "mpi", t0, t1 - t0, name, extra);
    return ret;
}

int MPI_Isend(const void *buf, int count, MPI_Datatype datatype,
              int dest, int tag, MPI_Comm comm, MPI_Request *request) {
    size_t nb = (size_t)count * dtype_size(datatype);
    uint64_t t0 = now_ns();
    int ret = PMPI_Isend(buf, count, datatype, dest, tag, comm, request);
    uint64_t t1 = now_ns();
    return isend_common(__builtin_return_address(0), "MPI_Isend", "isend", ret, t0, t1,
                        request, nb, dest, tag, comm);
}

int MPI_Issend(const void *buf, int count, MPI_Datatype datatype,
               int dest, int tag, MPI_Comm comm, MPI_Request *request) {
    size_t nb = (size_t)count * dtype_size(datatype);
    uint64_t t0 = now_ns();
    int ret = PMPI_Issend(buf, count, datatype, dest, tag, comm, request);
    uint64_t t1 = now_ns();
    return isend_common(__builtin_return_address(0), "MPI_Issend", "issend", ret, t0, t1,
                        request, nb, dest, tag, comm);
}

int MPI_Ibsend(const void *buf, int count, MPI_Datatype datatype,
               int dest, int tag, MPI_Comm comm, MPI_Request *request) {
    size_t nb = (size_t)count * dtype_size(datatype);
    uint64_t t0 = now_ns();
    int ret = PMPI_Ibsend(buf, count, datatype, dest, tag, comm, request);
    uint64_t t1 = now_ns();
    return isend_common(__builtin_return_address(0), "MPI_Ibsend", "ibsend", ret, t0, t1,
                        request, nb, dest, tag, comm);
}

int MPI_Irsend(const void *buf, int count, MPI_Datatype datatype,
               int dest, int tag, MPI_Comm comm, MPI_Request *request) {
    size_t nb = (size_t)count * dtype_size(datatype);
    uint64_t t0 = now_ns();
    int ret = PMPI_Irsend(buf, count, datatype, dest, tag, comm, request);
    uint64_t t1 = now_ns();
    return isend_common(__builtin_return_address(0), "MPI_Irsend", "irsend", ret, t0, t1,
                        request, nb, dest, tag, comm);
}

/* Matched probes and receives: messages received through an MPI_Message
 * handle are not paired by the critical path (instants, no edges); the
 * instant on the probe's communicator tells the analysis that its message
 * order there is incomplete. MPI_Mrecv has no communicator argument. */
static void mprobe_instant(const void *site, const char *name, const char *type, MPI_Comm comm,
                           int source, int tag, int found, const MPI_Status *st) {
    char extra[288], res[64] = "";
    if (found && st)
        snprintf(res, sizeof(res), ",rpeer=%d,rtag=%d", st->MPI_SOURCE, st->MPI_TAG);
    snprintf(extra, sizeof(extra), "type=%s,rank=%d,peer=%d,tag=%d,commid=%lld,flag=%d%s", type,
             g_mpi_rank, source, tag, (long long)comm_id_for(comm), found, res);
    emit_instant_at(site, "mpi", name, extra);
}

int MPI_Mprobe(int source, int tag, MPI_Comm comm, MPI_Message *message, MPI_Status *status) {
    MPI_Status local_status;
    MPI_Status *use_status = (status == MPI_STATUS_IGNORE) ? &local_status : status;
    int ret = PMPI_Mprobe(source, tag, comm, message, use_status);
    mprobe_instant(__builtin_return_address(0), "MPI_Mprobe", "mprobe", comm, source, tag,
                   ret == MPI_SUCCESS, use_status);
    return ret;
}

int MPI_Improbe(int source, int tag, MPI_Comm comm, int *flag, MPI_Message *message, MPI_Status *status) {
    MPI_Status local_status;
    MPI_Status *use_status = (status == MPI_STATUS_IGNORE) ? &local_status : status;
    int ret = PMPI_Improbe(source, tag, comm, flag, message, use_status);
    mprobe_instant(__builtin_return_address(0), "MPI_Improbe", "improbe", comm, source, tag,
                   ret == MPI_SUCCESS && flag && *flag, use_status);
    return ret;
}

int MPI_Mrecv(void *buf, int count, MPI_Datatype datatype, MPI_Message *message, MPI_Status *status) {
    int ret = PMPI_Mrecv(buf, count, datatype, message, status);
    char extra[96];
    snprintf(extra, sizeof(extra), "type=mrecv,rank=%d,bytes=%zu", g_mpi_rank,
             (size_t)count * dtype_size(datatype));
    emit_instant("mpi", "MPI_Mrecv", extra);
    return ret;
}

int MPI_Imrecv(void *buf, int count, MPI_Datatype datatype, MPI_Message *message, MPI_Request *request) {
    int ret = PMPI_Imrecv(buf, count, datatype, message, request);
    char extra[96];
    snprintf(extra, sizeof(extra), "type=imrecv,rank=%d,bytes=%zu", g_mpi_rank,
             (size_t)count * dtype_size(datatype));
    emit_instant("mpi", "MPI_Imrecv", extra);
    return ret;
}

int MPI_Irecv(void *buf, int count, MPI_Datatype datatype,
              int source, int tag, MPI_Comm comm, MPI_Request *request) {
    size_t nb = (size_t)count * dtype_size(datatype);
    int wildcard = (source == MPI_ANY_SOURCE) || (tag == MPI_ANY_TAG);
    /* A wildcard receive is resolved at completion: keep the communicator's
     * translation with the request (see "World-rank translation"). */
    Xlate *xl = (wildcard && comm != MPI_COMM_WORLD) ? xlate_acquire(comm) : NULL;
    uint64_t t0 = now_ns();
    int ret = PMPI_Irecv(buf, count, datatype, source, tag, comm, request);
    uint64_t t1 = now_ns();
    uint64_t req_id = 0;
    if (ret == MPI_SUCCESS && request && *request != MPI_REQUEST_NULL)
        req_id = req_register_ex(*request, "irecv", source, tag, nb, wildcard, comm, xl);
    else
        xlate_release(xl);
    /* A wildcard Irecv's peer/tag are unknown until completion (rpeer=/
     * rtag=/rwpeer= on the completing call); this span carries wildcard=1. */
    char extra[320], sid[40] = "";
    recv_tags(extra, sizeof(extra), "irecv", nb, comm, source, tag, wildcard);
    if (req_id) {
        snprintf(sid, sizeof(sid), ",sid=%llu", (unsigned long long)req_id);
        strncat(extra, sid, sizeof(extra) - strlen(extra) - 1);
    }
    emit_span("mpi", t0, t1 - t0, "MPI_Irecv", extra);
    return ret;
}

/* MPI_Sendrecv / MPI_Sendrecv_replace: a pair the point-to-point matcher
 * already understands --
 *   MPI_Sendrecv.send  zero-length span at the call's start, type=send:
 *                      the outgoing message (the send half cannot wait for
 *                      the receive half, so it gates nothing on this rank);
 *   MPI_Sendrecv       the whole call, type=recv: what the call waited for.
 * Two spans rather than one keep the dependency graph acyclic when two
 * ranks exchange with each other (each half would otherwise both feed and
 * wait for the other rank's call). */
static void sendrecv_emit(const void *site, const char *name, uint64_t t0, uint64_t t1,
                          MPI_Comm comm, int dest, int sendtag, size_t sbytes,
                          int source, int recvtag, size_t rbytes, int wildcard) {
    char sname[64], pt[48], extra[320];
    snprintf(sname, sizeof(sname), "%s.send", name);
    peer_tags(pt, sizeof(pt), comm, dest);
    snprintf(extra, sizeof(extra),
             "type=send,op=sendrecv,bytes=%zu,rank=%d,peer=%d,tag=%d,commid=%lld%s",
             sbytes, g_mpi_rank, dest, sendtag, (long long)comm_id_for(comm), pt);
    emit_span_at(site, "mpi", t0, 0, sname, extra);
    recv_tags(extra, sizeof(extra), "recv", rbytes, comm, source, recvtag, wildcard);
    strncat(extra, ",op=sendrecv", sizeof(extra) - strlen(extra) - 1);
    emit_span_at(site, "mpi", t0, t1 - t0, name, extra);
}

int MPI_Sendrecv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, int dest, int sendtag,
                 void *recvbuf, int recvcount, MPI_Datatype recvtype, int source, int recvtag,
                 MPI_Comm comm, MPI_Status *status) {
    int wildcard = (source == MPI_ANY_SOURCE) || (recvtag == MPI_ANY_TAG);
    MPI_Status local_status;
    MPI_Status *use_status = (status == MPI_STATUS_IGNORE) ? &local_status : status;
    uint64_t t0 = now_ns();
    int ret = PMPI_Sendrecv(sendbuf, sendcount, sendtype, dest, sendtag,
                            recvbuf, recvcount, recvtype, source, recvtag, comm, use_status);
    uint64_t t1 = now_ns();
    int rs = source, rt = recvtag;
    if (ret == MPI_SUCCESS && wildcard) { rs = use_status->MPI_SOURCE; rt = use_status->MPI_TAG; }
    sendrecv_emit(__builtin_return_address(0), "MPI_Sendrecv", t0, t1, comm,
                  dest, sendtag, (size_t)sendcount * dtype_size(sendtype),
                  rs, rt, (size_t)recvcount * dtype_size(recvtype), wildcard);
    return ret;
}

int MPI_Sendrecv_replace(void *buf, int count, MPI_Datatype datatype, int dest, int sendtag,
                         int source, int recvtag, MPI_Comm comm, MPI_Status *status) {
    int wildcard = (source == MPI_ANY_SOURCE) || (recvtag == MPI_ANY_TAG);
    MPI_Status local_status;
    MPI_Status *use_status = (status == MPI_STATUS_IGNORE) ? &local_status : status;
    uint64_t t0 = now_ns();
    int ret = PMPI_Sendrecv_replace(buf, count, datatype, dest, sendtag, source, recvtag,
                                    comm, use_status);
    uint64_t t1 = now_ns();
    int rs = source, rt = recvtag;
    if (ret == MPI_SUCCESS && wildcard) { rs = use_status->MPI_SOURCE; rt = use_status->MPI_TAG; }
    size_t nb = (size_t)count * dtype_size(datatype);
    sendrecv_emit(__builtin_return_address(0), "MPI_Sendrecv_replace", t0, t1, comm,
                  dest, sendtag, nb, rs, rt, nb, wildcard);
    return ret;
}

/* Probes: recorded (blocking probe as a span, MPI_Iprobe as an instant),
 * with the probed source resolved; they create no dependency edges. */
int MPI_Probe(int source, int tag, MPI_Comm comm, MPI_Status *status) {
    MPI_Status local_status;
    MPI_Status *use_status = (status == MPI_STATUS_IGNORE) ? &local_status : status;
    uint64_t t0 = now_ns();
    int ret = PMPI_Probe(source, tag, comm, use_status);
    uint64_t t1 = now_ns();
    char extra[288], res[96] = "";
    if (ret == MPI_SUCCESS) {
        int w = world_rank_of(comm, use_status->MPI_SOURCE);
        snprintf(res, sizeof(res), ",rpeer=%d,rtag=%d", use_status->MPI_SOURCE, use_status->MPI_TAG);
        if (w >= 0) {
            char wb[32];
            snprintf(wb, sizeof(wb), ",rwpeer=%d", w);
            strncat(res, wb, sizeof(res) - strlen(res) - 1);
        }
    }
    snprintf(extra, sizeof(extra), "type=probe,rank=%d,peer=%d,tag=%d,commid=%lld%s",
             g_mpi_rank, source, tag, (long long)comm_id_for(comm), res);
    emit_span("mpi", t0, t1 - t0, "MPI_Probe", extra);
    return ret;
}

int MPI_Iprobe(int source, int tag, MPI_Comm comm, int *flag, MPI_Status *status) {
    MPI_Status local_status;
    MPI_Status *use_status = (status == MPI_STATUS_IGNORE) ? &local_status : status;
    int ret = PMPI_Iprobe(source, tag, comm, flag, use_status);
    int found = (ret == MPI_SUCCESS && flag && *flag);
    char extra[288], res[96] = "";
    if (found) {
        int w = world_rank_of(comm, use_status->MPI_SOURCE);
        snprintf(res, sizeof(res), ",rpeer=%d,rtag=%d", use_status->MPI_SOURCE, use_status->MPI_TAG);
        if (w >= 0) {
            char wb[32];
            snprintf(wb, sizeof(wb), ",rwpeer=%d", w);
            strncat(res, wb, sizeof(res) - strlen(res) - 1);
        }
    }
    snprintf(extra, sizeof(extra), "type=iprobe,flag=%d,rank=%d,peer=%d,tag=%d,commid=%lld%s",
             found, g_mpi_rank, source, tag, (long long)comm_id_for(comm), res);
    emit_instant("mpi", "MPI_Iprobe", extra);
    return ret;
}

/* ── Completions ─────────────────────────────────────────────────────────
 * Status arrays: when the caller passes one, it goes to PMPI_* unchanged, so
 * the application sees exactly what the library writes (MPI_SOURCE, MPI_TAG,
 * MPI_ERROR under MPI_ERR_IN_STATUS, the cancelled flag). Only when the
 * caller passes MPI_STATUS(ES)_IGNORE and a completed request needs its
 * status (a wildcard receive, or a request with MPI_Cancel pending -- see
 * reqs_need_status) does the hook substitute its own array, invisible to the
 * caller. In that one case a failing request reports MPI_ERR_IN_STATUS
 * (the library's return value for a call with statuses) rather than
 * whatever the library returns when statuses are ignored. */

int MPI_Wait(MPI_Request *request, MPI_Status *status) {
    MPI_Request saved = request ? *request : MPI_REQUEST_NULL;
    MPI_Status local_status;
    MPI_Status *use_status = (status == MPI_STATUS_IGNORE) ? &local_status : status;
    uint64_t t0 = now_ns();
    int ret = PMPI_Wait(request, use_status);
    uint64_t t1 = now_ns();
    char extra[224], tail[160] = "";
    if (ret == MPI_SUCCESS) single_completion_tags(tail, sizeof(tail), saved, use_status);
    snprintf(extra, sizeof(extra), "type=wait,rank=%d%s", g_mpi_rank, tail);
    emit_span("mpi", t0, t1 - t0, "MPI_Wait", extra);
    return ret;
}

/* The requests a multi-completion call completed, from its return value
 * (MPI_ERR_IN_STATUS: per status; MPI_ERR_PENDING entries did not complete,
 * failed ones did but report no match). */
static void collect_all(compl_t *c, const MPI_Request *saved, int count, int ret,
                        const MPI_Status *st) {
    if (!saved) return;
    for (int i = 0; i < count; i++) {
        if (ret == MPI_SUCCESS) {
            compl_add(c, saved[i], st ? &st[i] : NULL);
        } else if (ret == MPI_ERR_IN_STATUS && st) {
            if (st[i].MPI_ERROR == MPI_ERR_PENDING) continue;
            compl_add(c, saved[i], st[i].MPI_ERROR == MPI_SUCCESS ? &st[i] : NULL);
        }
        /* any other error: the requests' state is unknown -- keep them */
    }
}

static void collect_some(compl_t *c, const MPI_Request *saved, int incount, int ret,
                         const int *outcount, const int *indices, const MPI_Status *st) {
    if (!saved || !outcount || !indices || *outcount == MPI_UNDEFINED) return;
    if (ret != MPI_SUCCESS && ret != MPI_ERR_IN_STATUS) return;
    for (int k = 0; k < *outcount; k++) {
        int idx = indices[k];
        if (idx < 0 || idx >= incount) continue;
        const MPI_Status *s = st ? &st[k] : NULL;
        if (s && ret == MPI_ERR_IN_STATUS && s->MPI_ERROR != MPI_SUCCESS) s = NULL;
        compl_add(c, saved[idx], s);
    }
}

int MPI_Waitall(int count, MPI_Request requests[], MPI_Status statuses[]) {
    MPI_Request *saved = save_requests(requests, count);
    MPI_Status *priv = NULL, *use = statuses;
    if (statuses == MPI_STATUSES_IGNORE && reqs_need_status(saved, count)) {
        priv = (MPI_Status *)malloc((size_t)count * sizeof(MPI_Status));
        if (priv) use = priv;
    }
    uint64_t t0 = now_ns();
    int ret = PMPI_Waitall(count, requests, use);
    uint64_t t1 = now_ns();
    compl_t c;
    compl_init(&c);
    collect_all(&c, saved, count, ret, use == MPI_STATUSES_IGNORE ? NULL : use);
    free(saved);
    free(priv);
    char head[96];
    snprintf(head, sizeof(head), "type=waitall,count=%d,rank=%d", count, g_mpi_rank);
    compl_emit(&c, __builtin_return_address(0), 1, t0, t1, "MPI_Waitall", head);
    return ret;
}

int MPI_Waitany(int count, MPI_Request requests[], int *index, MPI_Status *status) {
    MPI_Request *saved = save_requests(requests, count);
    MPI_Status local_status;
    MPI_Status *use_status = (status == MPI_STATUS_IGNORE) ? &local_status : status;
    uint64_t t0 = now_ns();
    int ret = PMPI_Waitany(count, requests, index, use_status);
    uint64_t t1 = now_ns();
    /* Exactly one request completes (index != MPI_UNDEFINED) -- look it up
     * via the SAVED array, since PMPI_Waitany nulls requests[*index] for
     * non-persistent requests. */
    char extra[288], tail[160] = "";
    if (ret == MPI_SUCCESS && index && *index != MPI_UNDEFINED && saved) {
        single_completion_tags(tail, sizeof(tail), saved[*index], use_status);
        snprintf(extra, sizeof(extra), "type=waitany,count=%d,rank=%d,completed_index=%d%s",
                 count, g_mpi_rank, *index, tail);
    } else {
        snprintf(extra, sizeof(extra), "type=waitany,count=%d,rank=%d,completed_index=none",
                 count, g_mpi_rank);
    }
    free(saved);
    emit_span("mpi", t0, t1 - t0, "MPI_Waitany", extra);
    return ret;
}

int MPI_Waitsome(int incount, MPI_Request requests[], int *outcount,
                 int indices[], MPI_Status statuses[]) {
    MPI_Request *saved = save_requests(requests, incount);
    MPI_Status *priv = NULL, *use = statuses;
    if (statuses == MPI_STATUSES_IGNORE && reqs_need_status(saved, incount)) {
        priv = (MPI_Status *)malloc((size_t)incount * sizeof(MPI_Status));
        if (priv) use = priv;
    }
    uint64_t t0 = now_ns();
    int ret = PMPI_Waitsome(incount, requests, outcount, indices, use);
    uint64_t t1 = now_ns();
    compl_t c;
    compl_init(&c);
    collect_some(&c, saved, incount, ret, outcount, indices,
                 use == MPI_STATUSES_IGNORE ? NULL : use);
    free(saved);
    free(priv);
    char head[128];
    snprintf(head, sizeof(head), "type=waitsome,incount=%d,outcount=%d,rank=%d", incount,
             ((ret == MPI_SUCCESS || ret == MPI_ERR_IN_STATUS) && outcount) ? *outcount : -1,
             g_mpi_rank);
    compl_emit(&c, __builtin_return_address(0), 1, t0, t1, "MPI_Waitsome", head);
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
    char extra[224], tail[160] = "";
    if (ret == MPI_SUCCESS && flag && *flag) {
        single_completion_tags(tail, sizeof(tail), saved, use_status);
        snprintf(extra, sizeof(extra), "type=test,flag=1,rank=%d%s", g_mpi_rank, tail);
    } else {
        ReqInfo ri;
        if (req_peek(saved, &ri) && ri.id)
            snprintf(tail, sizeof(tail), ",psid=%llu", (unsigned long long)ri.id);
        snprintf(extra, sizeof(extra), "type=test,flag=0,rank=%d%s", g_mpi_rank, tail);
    }
    emit_instant("mpi", "MPI_Test", extra);
    return ret;
}

int MPI_Testany(int count, MPI_Request requests[], int *index, int *flag, MPI_Status *status) {
    MPI_Request *saved = save_requests(requests, count);
    MPI_Status local_status;
    MPI_Status *use_status = (status == MPI_STATUS_IGNORE) ? &local_status : status;
    int ret = PMPI_Testany(count, requests, index, flag, use_status);
    char extra[288], tail[160] = "";
    if (ret == MPI_SUCCESS && flag && *flag && index && *index != MPI_UNDEFINED && saved) {
        single_completion_tags(tail, sizeof(tail), saved[*index], use_status);
        snprintf(extra, sizeof(extra), "type=testany,flag=1,count=%d,rank=%d,completed_index=%d%s",
                 count, g_mpi_rank, *index, tail);
    } else {
        snprintf(extra, sizeof(extra), "type=testany,flag=0,count=%d,rank=%d", count, g_mpi_rank);
    }
    free(saved);
    emit_instant("mpi", "MPI_Testany", extra);
    return ret;
}

int MPI_Testsome(int incount, MPI_Request requests[], int *outcount,
                 int indices[], MPI_Status statuses[]) {
    MPI_Request *saved = save_requests(requests, incount);
    MPI_Status *priv = NULL, *use = statuses;
    if (statuses == MPI_STATUSES_IGNORE && reqs_need_status(saved, incount)) {
        priv = (MPI_Status *)malloc((size_t)incount * sizeof(MPI_Status));
        if (priv) use = priv;
    }
    int ret = PMPI_Testsome(incount, requests, outcount, indices, use);
    compl_t c;
    compl_init(&c);
    collect_some(&c, saved, incount, ret, outcount, indices,
                 use == MPI_STATUSES_IGNORE ? NULL : use);
    free(saved);
    free(priv);
    char head[128];
    snprintf(head, sizeof(head), "type=testsome,incount=%d,outcount=%d,rank=%d", incount,
             ((ret == MPI_SUCCESS || ret == MPI_ERR_IN_STATUS) && outcount) ? *outcount : -1,
             g_mpi_rank);
    compl_emit(&c, __builtin_return_address(0), 0, 0, 0, "MPI_Testsome", head);
    return ret;
}

int MPI_Testall(int count, MPI_Request requests[], int *flag, MPI_Status statuses[]) {
    MPI_Request *saved = save_requests(requests, count);
    MPI_Status *priv = NULL, *use = statuses;
    if (statuses == MPI_STATUSES_IGNORE && reqs_need_status(saved, count)) {
        priv = (MPI_Status *)malloc((size_t)count * sizeof(MPI_Status));
        if (priv) use = priv;
    }
    int ret = PMPI_Testall(count, requests, flag, use);
    compl_t c;
    compl_init(&c);
    /* Statuses (and completions) only when flag is true. */
    if (flag && *flag)
        collect_all(&c, saved, count, ret, use == MPI_STATUSES_IGNORE ? NULL : use);
    free(saved);
    free(priv);
    char head[128];
    snprintf(head, sizeof(head), "type=testall,flag=%d,count=%d,rank=%d",
             ((ret == MPI_SUCCESS || ret == MPI_ERR_IN_STATUS) && flag) ? *flag : -1, count,
             g_mpi_rank);
    compl_emit(&c, __builtin_return_address(0), 0, 0, 0, "MPI_Testall", head);
    return ret;
}

/* ── Cancellation ────────────────────────────────────────────────────────
 * A cancelled request never completes normally; the instant gives it an
 * explicit end state, and the completion that follows checks
 * MPI_Test_cancelled before reporting a match. */
int MPI_Cancel(MPI_Request *request) {
    MPI_Request saved = request ? *request : MPI_REQUEST_NULL;
    int ret = PMPI_Cancel(request);
    ReqInfo ri;
    int found = req_peek(saved, &ri);
    if (found) req_mark_cancel(saved);
    char extra[96];
    if (found && ri.id)
        snprintf(extra, sizeof(extra), "type=cancel,rank=%d,psid=%llu",
                 g_mpi_rank, (unsigned long long)ri.id);
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

/* Vector collectives and reduce-scatter: same rendezvous semantics as their
 * fixed-size forms. bytes= is this rank's contribution (sent data; for
 * in-place calls the received part of its own buffer). */
static size_t count_sum(const int *counts, MPI_Comm comm) {
    int n = 0;
    if (!counts) return 0;
    if (PMPI_Comm_size(comm, &n) != MPI_SUCCESS || n <= 0) return 0;
    size_t s = 0;
    for (int i = 0; i < n; i++) if (counts[i] > 0) s += (size_t)counts[i];
    return s;
}

static size_t own_count(const int *counts, MPI_Comm comm) {
    int r = 0;
    if (!counts || PMPI_Comm_rank(comm, &r) != MPI_SUCCESS || r < 0) return 0;
    return counts[r] > 0 ? (size_t)counts[r] : 0;
}

_COLL(MPI_Allgatherv,
      PMPI_Allgatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, comm),
      "allgatherv",
      sendbuf == MPI_IN_PLACE ? own_count(recvcounts, comm) * dtype_size(recvtype)
                              : (size_t)sendcount * dtype_size(sendtype),
      const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
      const int recvcounts[], const int displs[], MPI_Datatype recvtype, MPI_Comm comm)

_COLL(MPI_Alltoallv,
      PMPI_Alltoallv(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls,
                     recvtype, comm),
      "alltoallv",
      sendbuf == MPI_IN_PLACE ? count_sum(recvcounts, comm) * dtype_size(recvtype)
                              : count_sum(sendcounts, comm) * dtype_size(sendtype),
      const void *sendbuf, const int sendcounts[], const int sdispls[], MPI_Datatype sendtype,
      void *recvbuf, const int recvcounts[], const int rdispls[], MPI_Datatype recvtype,
      MPI_Comm comm)

_COLL(MPI_Gatherv,
      PMPI_Gatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype, root, comm),
      "gatherv",
      sendbuf == MPI_IN_PLACE ? own_count(recvcounts, comm) * dtype_size(recvtype)
                              : (size_t)sendcount * dtype_size(sendtype),
      const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf,
      const int recvcounts[], const int displs[], MPI_Datatype recvtype, int root, MPI_Comm comm)

_COLL(MPI_Scatterv,
      PMPI_Scatterv(sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount, recvtype, root, comm),
      "scatterv",
      recvbuf == MPI_IN_PLACE ? own_count(sendcounts, comm) * dtype_size(sendtype)
                              : (size_t)recvcount * dtype_size(recvtype),
      const void *sendbuf, const int sendcounts[], const int displs[], MPI_Datatype sendtype,
      void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm)

_COLL(MPI_Reduce_scatter,
      PMPI_Reduce_scatter(sendbuf, recvbuf, recvcounts, datatype, op, comm),
      "reduce_scatter", count_sum(recvcounts, comm) * dtype_size(datatype),
      const void *sendbuf, void *recvbuf, const int recvcounts[], MPI_Datatype datatype,
      MPI_Op op, MPI_Comm comm)

static size_t block_total(int recvcount, MPI_Comm comm) {
    int n = 0;
    if (recvcount <= 0 || PMPI_Comm_size(comm, &n) != MPI_SUCCESS || n <= 0) return 0;
    return (size_t)recvcount * (size_t)n;
}

_COLL(MPI_Reduce_scatter_block,
      PMPI_Reduce_scatter_block(sendbuf, recvbuf, recvcount, datatype, op, comm),
      "reduce_scatter_block", block_total(recvcount, comm) * dtype_size(datatype),
      const void *sendbuf, void *recvbuf, int recvcount, MPI_Datatype datatype,
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
    if (req_id)   /* sid=: the completing call links back (psid=) */      \
        snprintf(extra, sizeof(extra),                                   \
                 "type=%s,bytes=%zu,rank=%d,commid=%lld,sid=%llu",       \
                 TYPE_STR, nb, g_mpi_rank, (long long)cid,               \
                 (unsigned long long)req_id);                            \
    else                                                                  \
        snprintf(extra, sizeof(extra),                                   \
                 "type=%s,bytes=%zu,rank=%d,commid=%lld",                \
                 TYPE_STR, nb, g_mpi_rank, (long long)cid);              \
    emit_span("mpi", t0, now_ns()-t0, #FNAME, extra);                   \
    return ret;                                                           \
}

int MPI_Ibarrier(MPI_Comm comm, MPI_Request *request) {
    int64_t cid = comm_id_for(comm);
    uint64_t t0 = now_ns();
    int ret = PMPI_Ibarrier(comm, request);
    uint64_t t1 = now_ns();
    uint64_t req_id = (ret == MPI_SUCCESS && request && *request != MPI_REQUEST_NULL)
                      ? req_register(*request, "ibarrier", -1, 0, 0, 0) : 0;
    char extra[128];
    if (req_id)
        snprintf(extra, sizeof(extra), "type=ibarrier,rank=%d,commid=%lld,sid=%llu",
                 g_mpi_rank, (long long)cid, (unsigned long long)req_id);
    else
        snprintf(extra, sizeof(extra), "type=ibarrier,rank=%d,commid=%lld", g_mpi_rank, (long long)cid);
    emit_span("mpi", t0, t1 - t0, "MPI_Ibarrier", extra);
    return ret;
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

/* Persistent requests are recorded (with commid=) but their messages are not
 * paired by the critical path: a communicator with persistent traffic has
 * an incomplete message order there. */
static int persistent_init(const void *site, const char *name, const char *type, int ret,
                           uint64_t t0, size_t nb, int peer, int tag, MPI_Comm comm) {
    char extra[192];
    snprintf(extra, sizeof(extra), "type=%s,bytes=%zu,rank=%d,peer=%d,tag=%d,commid=%lld",
             type, nb, g_mpi_rank, peer, tag, (long long)comm_id_for(comm));
    emit_span_at(site, "mpi", t0, now_ns() - t0, name, extra);
    return ret;
}

#define _PERSISTENT_SEND(FNAME, TYPE) \
int FNAME(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, \
          MPI_Request *request) { \
    uint64_t t0 = now_ns(); \
    int ret = P##FNAME(buf, count, datatype, dest, tag, comm, request); \
    return persistent_init(__builtin_return_address(0), #FNAME, TYPE, ret, t0, \
                           (size_t)count * dtype_size(datatype), dest, tag, comm); \
}
_PERSISTENT_SEND(MPI_Send_init,  "send_init")
_PERSISTENT_SEND(MPI_Bsend_init, "bsend_init")
_PERSISTENT_SEND(MPI_Ssend_init, "ssend_init")
_PERSISTENT_SEND(MPI_Rsend_init, "rsend_init")
#undef _PERSISTENT_SEND

int MPI_Recv_init(void *buf, int count, MPI_Datatype datatype,
                  int source, int tag, MPI_Comm comm, MPI_Request *request) {
    uint64_t t0 = now_ns();
    int ret = PMPI_Recv_init(buf, count, datatype, source, tag, comm, request);
    return persistent_init(__builtin_return_address(0), "MPI_Recv_init", "recv_init", ret, t0,
                           (size_t)count * dtype_size(datatype), source, tag, comm);
}

int MPI_Start(MPI_Request *request) {
    MPI_Request saved = request ? *request : MPI_REQUEST_NULL;
    uint64_t t0 = now_ns();
    int ret = PMPI_Start(request);
    uint64_t req_id = (ret == MPI_SUCCESS && request && *request != MPI_REQUEST_NULL)
                      ? req_register(*request, "start", -1, 0, 0, 0) : 0;
    char extra[96];
    if (req_id)
        snprintf(extra, sizeof(extra), "type=start,rank=%d,sid=%llu",
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
