/*
 * A stand-in HIP runtime (built as libamdhip64.so) for testing the ROCm
 * hook's wrappers on a machine without an AMD GPU:
 * tests/integration/test_rocm_hip_mock.py puts it first on LD_LIBRARY_PATH,
 * where the hook's dlopen("libamdhip64.so") finds it. Only the calls the
 * hook and tests/fixtures/hip_streams.c make, with the behaviour that
 * matters to the hook:
 *   streams     objects whose addresses are reused: a destroyed stream's
 *               (or event's) object is the next one created (last freed
 *               first), as real runtimes hand freed handles out again
 *   host funcs  run in submission order by ONE callback thread, like the
 *               real runtime; hipStreamSynchronize / hipDeviceSynchronize
 *               wait for them
 *   events      a CLOCK_MONOTONIC timestamp at record; elapsed time from it
 *   kernels     nothing (the launch is recorded by the caller)
 *   copies      memcpy between host buffers
 * It is not a model of device execution.
 */
#define _GNU_SOURCE
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef int hipError_t;
typedef struct { int x, y, z; } dim3;
typedef struct { unsigned flags; int prio; } MockStream;
typedef struct { uint64_t t; } MockEvent;
typedef MockStream *hipStream_t;
typedef MockEvent *hipEvent_t;
typedef void (*hipHostFn_t)(void *);
typedef void (*hipStreamCallback_t)(hipStream_t, hipError_t, void *);

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/* ── the callback thread ─────────────────────────────────────────────── */
typedef struct Job { hipHostFn_t fn; hipStreamCallback_t cb; hipStream_t s; void *user; struct Job *next; } Job;
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  g_cv = PTHREAD_COND_INITIALIZER, g_done = PTHREAD_COND_INITIALIZER;
static Job *g_head, *g_tail;
static uint64_t g_queued, g_ran;
static int g_started;

static void *worker(void *arg) {
    (void)arg;
    pthread_mutex_lock(&g_mu);
    for (;;) {
        while (!g_head) pthread_cond_wait(&g_cv, &g_mu);
        Job *j = g_head;
        g_head = j->next;
        if (!g_head) g_tail = NULL;
        pthread_mutex_unlock(&g_mu);
        if (j->fn) j->fn(j->user);
        else j->cb(j->s, 0, j->user);
        free(j);
        pthread_mutex_lock(&g_mu);
        g_ran++;
        pthread_cond_broadcast(&g_done);
    }
    return NULL;
}

static hipError_t enqueue(hipHostFn_t fn, hipStreamCallback_t cb, hipStream_t s, void *user) {
    Job *j = calloc(1, sizeof(*j));
    if (!j) return 2;
    *j = (Job){fn, cb, s, user, NULL};
    pthread_mutex_lock(&g_mu);
    if (!g_started) {
        pthread_t t;
        pthread_create(&t, NULL, worker, NULL);
        pthread_detach(t);
        g_started = 1;
    }
    if (g_tail) g_tail->next = j; else g_head = j;
    g_tail = j;
    g_queued++;
    pthread_cond_signal(&g_cv);
    pthread_mutex_unlock(&g_mu);
    return 0;
}

static hipError_t drain(void) {
    pthread_mutex_lock(&g_mu);
    uint64_t want = g_queued;
    while (g_ran < want) pthread_cond_wait(&g_done, &g_mu);
    pthread_mutex_unlock(&g_mu);
    return 0;
}

/* ── device, errors ──────────────────────────────────────────────────── */
hipError_t hipGetDeviceCount(int *n) { if (n) *n = 1; return 0; }
hipError_t hipGetDevice(int *d) { if (d) *d = 0; return 0; }
hipError_t hipSetDevice(int d) { (void)d; return 0; }
const char *hipGetErrorString(hipError_t e) { return e ? "mock error" : "hipSuccess"; }
hipError_t hipDeviceSynchronize(void) { return drain(); }

/* ── streams ─────────────────────────────────────────────────────────── */
static void *g_free_streams[64], *g_free_events[64];
static int g_nfree_streams, g_nfree_events;
static pthread_mutex_t g_alloc_mu = PTHREAD_MUTEX_INITIALIZER;

static void *take(void **list, int *n, size_t size) {
    pthread_mutex_lock(&g_alloc_mu);
    void *p = *n ? list[--*n] : NULL;
    pthread_mutex_unlock(&g_alloc_mu);
    return p ? memset(p, 0, size) : calloc(1, size);
}

static void give(void **list, int *n, void *p) {
    pthread_mutex_lock(&g_alloc_mu);
    if (*n < 64) { list[(*n)++] = p; p = NULL; }
    pthread_mutex_unlock(&g_alloc_mu);
    free(p);
}

hipError_t hipStreamCreateWithPriority(hipStream_t *s, unsigned flags, int prio) {
    MockStream *m = take(g_free_streams, &g_nfree_streams, sizeof(MockStream));
    if (!m) return 2;
    m->flags = flags;
    m->prio = prio;
    *s = m;
    return 0;
}
hipError_t hipStreamCreateWithFlags(hipStream_t *s, unsigned flags) { return hipStreamCreateWithPriority(s, flags, 0); }
hipError_t hipStreamCreate(hipStream_t *s) { return hipStreamCreateWithPriority(s, 0, 0); }
hipError_t hipStreamDestroy(hipStream_t s) { drain(); give(g_free_streams, &g_nfree_streams, s); return 0; }
hipError_t hipStreamGetFlags(hipStream_t s, unsigned *f) { if (!s) return 1; *f = s->flags; return 0; }
hipError_t hipStreamGetPriority(hipStream_t s, int *p) { if (!s) return 1; *p = s->prio; return 0; }
hipError_t hipStreamSynchronize(hipStream_t s) { (void)s; return drain(); }
hipError_t hipLaunchHostFunc(hipStream_t s, hipHostFn_t fn, void *user) { return enqueue(fn, NULL, s, user); }
hipError_t hipStreamAddCallback(hipStream_t s, hipStreamCallback_t cb, void *user, unsigned flags) {
    (void)flags;
    return enqueue(NULL, cb, s, user);
}

/* ── events ──────────────────────────────────────────────────────────── */
hipError_t hipEventCreate(hipEvent_t *e) { *e = take(g_free_events, &g_nfree_events, sizeof(MockEvent)); return *e ? 0 : 2; }
hipError_t hipEventCreateWithFlags(hipEvent_t *e, unsigned flags) { (void)flags; return hipEventCreate(e); }
hipError_t hipEventDestroy(hipEvent_t e) { give(g_free_events, &g_nfree_events, e); return 0; }
hipError_t hipEventRecord(hipEvent_t e, hipStream_t s) { (void)s; if (e) e->t = now_ns(); return 0; }
hipError_t hipEventSynchronize(hipEvent_t e) { (void)e; return 0; }
hipError_t hipEventElapsedTime(float *ms, hipEvent_t a, hipEvent_t b) {
    *ms = (float)((double)(b->t - a->t) / 1e6);
    return 0;
}
hipError_t hipStreamWaitEvent(hipStream_t s, hipEvent_t e, unsigned flags) { (void)s; (void)e; (void)flags; return 0; }

/* ── kernels ─────────────────────────────────────────────────────────── */
hipError_t hipLaunchKernel(const void *f, dim3 g, dim3 b, void **a, size_t sh, hipStream_t s) {
    (void)f; (void)g; (void)b; (void)a; (void)sh; (void)s; return 0;
}
hipError_t hipLaunchCooperativeKernel(const void *f, dim3 g, dim3 b, void **a, unsigned sh, hipStream_t s) {
    (void)f; (void)g; (void)b; (void)a; (void)sh; (void)s; return 0;
}
hipError_t hipExtLaunchKernel(const void *f, dim3 g, dim3 b, void **a, size_t sh, hipStream_t s, hipEvent_t start,
                              hipEvent_t stop, int flags) {
    (void)f; (void)g; (void)b; (void)a; (void)sh; (void)flags;
    if (start) hipEventRecord(start, s);
    if (stop) hipEventRecord(stop, s);
    return 0;
}

/* ── memory ──────────────────────────────────────────────────────────── */
hipError_t hipMalloc(void **p, size_t n) { *p = malloc(n); return *p ? 0 : 2; }
hipError_t hipFree(void *p) { free(p); return 0; }
static void copy2d(void *d, size_t dp, const void *s, size_t sp, size_t w, size_t h) {
    for (size_t r = 0; r < h; r++) memcpy((char *)d + r * dp, (const char *)s + r * sp, w);
}
hipError_t hipMemcpy(void *d, const void *s, size_t n, int k) { (void)k; memcpy(d, s, n); return 0; }
hipError_t hipMemcpyAsync(void *d, const void *s, size_t n, int k, hipStream_t st) { (void)k; (void)st; memcpy(d, s, n); return 0; }
hipError_t hipMemcpy2D(void *d, size_t dp, const void *s, size_t sp, size_t w, size_t h, int k) {
    (void)k; copy2d(d, dp, s, sp, w, h); return 0;
}
hipError_t hipMemcpy2DAsync(void *d, size_t dp, const void *s, size_t sp, size_t w, size_t h, int k, hipStream_t st) {
    (void)k; (void)st; copy2d(d, dp, s, sp, w, h); return 0;
}
hipError_t hipMemcpyPeer(void *d, int dd, const void *s, int sd, size_t n) { (void)dd; (void)sd; memcpy(d, s, n); return 0; }
hipError_t hipMemcpyPeerAsync(void *d, int dd, const void *s, int sd, size_t n, hipStream_t st) {
    (void)dd; (void)sd; (void)st; memcpy(d, s, n); return 0;
}
hipError_t hipMemcpyDtoD(void *d, void *s, size_t n) { memcpy(d, s, n); return 0; }
hipError_t hipMemcpyDtoDAsync(void *d, void *s, size_t n, hipStream_t st) { (void)st; memcpy(d, s, n); return 0; }
hipError_t hipMemcpyHtoDAsync(void *d, void *s, size_t n, hipStream_t st) { (void)st; memcpy(d, s, n); return 0; }
hipError_t hipMemcpyDtoHAsync(void *d, void *s, size_t n, hipStream_t st) { (void)st; memcpy(d, s, n); return 0; }
