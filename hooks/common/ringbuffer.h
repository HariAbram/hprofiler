/*
 * Lock-free per-thread SPSC ring buffer + append-only arena + name
 * interning table, for the profiling-collection hot path.
 *
 * ── Why this exists ─────────────────────────────────────────────────────
 * Every hook's current emit_span() does, synchronously, on the profiled
 * program's own calling thread, for every single intercepted call:
 *   1. snprintf() the record into a stack buffer
 *   2. lock a process-wide mutex shared by every thread using this hook
 *   3. lazily connect() the socket if not already connected
 *   4. send() -- a blocking syscall, subject to backpressure from however
 *      fast (or slow) the Python collector's reader thread drains it
 *   5. unlock
 * That is real, unavoidable-today overhead added to the profiled program's
 * own critical path on every intercepted call -- exactly what low-overhead
 * tracing systems (Score-P, HPCToolkit, etc.) avoid via asynchronous,
 * buffered collection. This header provides that alternative: the hot path
 * becomes a bump-allocate + memcpy into the CALLING THREAD'S OWN ring
 * buffer (no lock, no syscall, wait-free from the producer's perspective),
 * and the actual socket write happens later, off a separate drain path.
 *
 * ── Design: per-thread SPSC, not a shared MPSC structure ───────────────
 * Multiple threads DO concurrently call into a hook (e.g. every OpenMP
 * worker thread), so naively this would need a multi-producer structure --
 * which is real work to get right lock-free. Sharding by thread sidesteps
 * that entirely: each OS thread gets its OWN ring buffer (rb_get_local()),
 * written to ONLY by that thread (the producer) and drained by ONE
 * separate consumer (a background thread, or an existing flush point) --
 * a true single-producer/single-consumer buffer per shard, which is a
 * well-understood, provably-correct pattern with a much smaller surface
 * for concurrency bugs than a general MPSC queue.
 *
 * ── What this header does NOT do ────────────────────────────────────────
 * It does not open a socket, does not spawn a drain thread, and is not
 * wired into any hook's actual emit_span() path. That integration --
 * background-thread lifecycle across fork/exec, ordering with the
 * process's own exit and the existing MPI_Finalize-style final-flush
 * pattern, per-hook socket reuse -- is real additional work with its own
 * failure modes (a stuck or crashed drain thread silently losing the
 * tail of a trace is a much worse failure than today's synchronous-but-
 * simple path), and is deliberately left as a designed-but-not-yet-wired-
 * up next step rather than rushed into hook code without the ability to
 * stress-test it under this machine's real GPU/multi-node workloads. This
 * header is verified in isolation (see tests/native/ringbuffer_stress.c):
 * concurrent producer/consumer correctness (no lost or corrupted events
 * short of a full buffer), the drop counter accounting exactly for events
 * dropped once it does overflow, and a standalone latency comparison
 * against the mutex+send pattern it would replace.
 *
 * ── Drop policy ──────────────────────────────────────────────────────────
 * rb_push() never blocks. If the buffer is full, the new event is dropped
 * and rb->dropped is atomically incremented -- "drop newest", the simplest
 * policy that can never stall the profiled program regardless of how slow
 * the consumer is. rb_dropped_count() lets a periodic counter/summary flush
 * report how many events were lost, rather than the loss being silent.
 */
#ifndef HPROFILER_RINGBUFFER_H
#define HPROFILER_RINGBUFFER_H

#include <stdatomic.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <pthread.h>

#ifndef RB_SLOT_SIZE
#define RB_SLOT_SIZE 512   /* bytes per record; matches existing hooks' typical span-line buffer sizes */
#endif
#ifndef RB_CAPACITY
#define RB_CAPACITY 1024   /* must be a power of 2; 1024*512B = 512KB per thread's ring buffer */
#endif

typedef struct {
    char     data[RB_SLOT_SIZE];
    uint16_t len;
} rb_slot_t;

typedef struct {
    rb_slot_t         slots[RB_CAPACITY];
    _Atomic uint64_t  head;     /* next slot the consumer will read */
    _Atomic uint64_t  tail;     /* next slot the producer will write */
    _Atomic uint64_t  dropped;  /* events dropped because the buffer was full */
} ringbuffer_t;

static inline void rb_init(ringbuffer_t *rb) {
    atomic_init(&rb->head, 0);
    atomic_init(&rb->tail, 0);
    atomic_init(&rb->dropped, 0);
}

/* Producer side -- call only from the single thread that owns this ring
 * buffer. Never blocks. Returns 1 on success, 0 if the buffer was full
 * (the event was dropped; rb->dropped was incremented). Truncates any
 * record longer than RB_SLOT_SIZE, same truncation risk the existing
 * fixed-size snprintf buffers already have -- not a new regression. */
static inline int rb_push(ringbuffer_t *rb, const char *data, uint16_t len) {
    uint64_t tail = atomic_load_explicit(&rb->tail, memory_order_relaxed);
    uint64_t head = atomic_load_explicit(&rb->head, memory_order_acquire);
    if (tail - head >= RB_CAPACITY) {
        atomic_fetch_add_explicit(&rb->dropped, 1, memory_order_relaxed);
        return 0;
    }
    rb_slot_t *slot = &rb->slots[tail % RB_CAPACITY];
    uint16_t n = (uint16_t)(len < RB_SLOT_SIZE ? len : RB_SLOT_SIZE);
    memcpy(slot->data, data, n);
    slot->len = n;
    /* release: publishes the slot's contents to the consumer atomically
     * with the tail bump -- the consumer's acquire load of tail (in
     * rb_pop) cannot observe the new tail without also observing this
     * write to slot->data/len. */
    atomic_store_explicit(&rb->tail, tail + 1, memory_order_release);
    return 1;
}

/* Consumer side -- call only from the single thread draining this ring
 * buffer. Returns 1 and fills out_data/out_len if an item was available,
 * 0 if the buffer was empty. out_data must have room for RB_SLOT_SIZE
 * bytes. */
static inline int rb_pop(ringbuffer_t *rb, char *out_data, uint16_t *out_len) {
    uint64_t head = atomic_load_explicit(&rb->head, memory_order_relaxed);
    uint64_t tail = atomic_load_explicit(&rb->tail, memory_order_acquire);
    if (head >= tail) return 0;
    rb_slot_t *slot = &rb->slots[head % RB_CAPACITY];
    memcpy(out_data, slot->data, slot->len);
    *out_len = slot->len;
    atomic_store_explicit(&rb->head, head + 1, memory_order_release);
    return 1;
}

static inline uint64_t rb_dropped_count(const ringbuffer_t *rb) {
    return atomic_load_explicit((_Atomic uint64_t *)&rb->dropped, memory_order_relaxed);
}

/* Approximate depth (items currently buffered, not yet drained). Racy by
 * nature if called concurrently with push/pop (head/tail may be read at
 * slightly different instants) -- fine for a monitoring/diagnostic read,
 * not meant for anything requiring an exact count. */
static inline uint64_t rb_depth(const ringbuffer_t *rb) {
    uint64_t tail = atomic_load_explicit((_Atomic uint64_t *)&rb->tail, memory_order_relaxed);
    uint64_t head = atomic_load_explicit((_Atomic uint64_t *)&rb->head, memory_order_relaxed);
    return tail - head;
}

/* ── Append-only arena ────────────────────────────────────────────────────
 * A simple bump allocator for variable-length data (e.g. long call stacks
 * or tag blobs) that doesn't fit a fixed-size ring slot. Single fixed-size
 * block, thread-safe via one atomic fetch-add (no lock) -- allocation
 * itself is wait-free; the arena as a whole is meant to be reset (or
 * simply left to fill and rotated to a fresh block) by the consumer/drain
 * side, not the hot path. Returns NULL if the arena is full -- caller must
 * handle that (e.g. fall back to truncating into a ring slot directly)
 * rather than this header silently growing/reallocating, which would
 * reintroduce a lock on the hot path.
 */
typedef struct {
    char             *base;
    uint32_t          capacity;
    _Atomic uint32_t  offset;
} arena_t;

static inline void arena_init(arena_t *a, char *base, uint32_t capacity) {
    a->base = base;
    a->capacity = capacity;
    atomic_init(&a->offset, 0);
}

/* Wait-free bump allocation. size 0 returns a valid non-NULL pointer
 * (an empty allocation), never NULL, so callers don't need a special case. */
static inline char *arena_alloc(arena_t *a, uint32_t size) {
    uint32_t off = atomic_fetch_add_explicit(&a->offset, size, memory_order_relaxed);
    if (off + size > a->capacity) return NULL;
    return a->base + off;
}

static inline void arena_reset(arena_t *a) {
    atomic_store_explicit(&a->offset, 0, memory_order_relaxed);
}

/* ── Name interning ──────────────────────────────────────────────────────
 * Maps each unique string (typically a kernel/function name repeated
 * across many events) to a small integer id, assigned the first time it's
 * seen -- so repeat occurrences can reference the id instead of resending
 * the full string. Backed by a simple mutex-protected open-addressing
 * table: unlike the ring buffer/arena, this does NOT need to be lock-free
 * to deliver the intended benefit -- interning a given name only touches
 * the lock ONCE (the first time that name is seen), not once per event,
 * so contention is proportional to the number of *distinct* names, not
 * the number of *events* -- orders of magnitude less hot than the
 * per-event path this whole header exists to get off a lock.
 */
#define INTERN_TABLE_CAP 4096   /* must be a power of 2; distinct names per process */

typedef struct {
    const char     *names[INTERN_TABLE_CAP];  /* NULL = empty slot */
    uint32_t        n;
    pthread_mutex_t mutex;
} intern_table_t;

static inline void intern_init(intern_table_t *t) {
    memset(t->names, 0, sizeof(t->names));
    t->n = 0;
    pthread_mutex_init(&t->mutex, NULL);
}

static inline uint32_t _intern_hash(const char *s) {
    uint32_t h = 2166136261u;   /* FNV-1a */
    for (; *s; s++) { h ^= (unsigned char)*s; h *= 16777619u; }
    return h;
}

/* Returns the interned id for `name` (assigning a new one on first sight),
 * or UINT32_MAX if the table is full. `name` must remain valid for the
 * lifetime of the table (this stores the pointer, does not copy the
 * string) -- callers should intern process-lifetime-constant strings
 * (e.g. a kernel name already cached elsewhere), not stack buffers. */
static inline uint32_t intern_id(intern_table_t *t, const char *name) {
    pthread_mutex_lock(&t->mutex);
    uint32_t h = _intern_hash(name) & (INTERN_TABLE_CAP - 1);
    for (uint32_t probe = 0; probe < INTERN_TABLE_CAP; probe++) {
        uint32_t idx = (h + probe) & (INTERN_TABLE_CAP - 1);
        if (t->names[idx] == NULL) {
            t->names[idx] = name;
            t->n++;
            pthread_mutex_unlock(&t->mutex);
            return idx;
        }
        if (strcmp(t->names[idx], name) == 0) {
            pthread_mutex_unlock(&t->mutex);
            return idx;
        }
    }
    pthread_mutex_unlock(&t->mutex);
    return UINT32_MAX;  /* table full */
}

static inline const char *intern_lookup(const intern_table_t *t, uint32_t id) {
    if (id >= INTERN_TABLE_CAP) return NULL;
    return t->names[id];
}

#endif /* HPROFILER_RINGBUFFER_H */
