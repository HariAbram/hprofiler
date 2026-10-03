/*
 * Lock-free single-producer/single-consumer rings, an append-only arena and
 * a name-interning table for the collection hot path.
 *
 * One ring per producing thread (sharding by thread avoids a multi-producer
 * structure): only the owning thread pushes, one consumer drains.
 *
 * rb_bytes_t (variable-length records) is what hooks/common/hp_transport.h
 * uses for every hook. The fixed-slot ringbuffer_t (truncates records longer
 * than RB_SLOT_SIZE; when full, rb_push drops the new event and counts it in
 * rb->dropped), the arena and the interning table are exercised by
 * tests/native/ringbuffer_stress.c (also under ThreadSanitizer) and are not
 * used by the hooks.
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

/* Producer side -- call only from the single thread that owns this ring.
 * Never blocks. Returns 1 on success, 0 if the ring was full (the event was
 * dropped; rb->dropped was incremented). Records longer than RB_SLOT_SIZE
 * are truncated. */
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

/* Approximate depth; racy against concurrent push/pop (diagnostics only). */
static inline uint64_t rb_depth(const ringbuffer_t *rb) {
    uint64_t tail = atomic_load_explicit((_Atomic uint64_t *)&rb->tail, memory_order_relaxed);
    uint64_t head = atomic_load_explicit((_Atomic uint64_t *)&rb->head, memory_order_relaxed);
    return tail - head;
}

/* ── Variable-length byte ring (used by hp_transport.h) ───────────────────
 * Same single-producer / single-consumer protocol as ringbuffer_t above,
 * over a power-of-two byte array holding length-prefixed records
 * ([uint32 len][len bytes]), so records of any size up to half the ring
 * travel intact -- no fixed slot, no truncation. head/tail are byte
 * offsets that only grow; positions wrap with `mask`. The producer
 * publishes a record with one release store of tail after copying it; the
 * consumer releases the space with one release store of head after
 * copying it out. */
typedef struct {
    char             *buf;
    uint64_t          cap;      /* power of two */
    uint64_t          mask;
    _Atomic uint64_t  head;     /* consumer position */
    _Atomic uint64_t  tail;     /* producer position */
} rb_bytes_t;

static inline int rb_bytes_init(rb_bytes_t *r, uint64_t cap) {
    uint64_t c = 1;
    while (c < cap) c <<= 1;
    r->buf = (char *)malloc(c);
    if (!r->buf) return 0;
    r->cap = c;
    r->mask = c - 1;
    atomic_init(&r->head, 0);
    atomic_init(&r->tail, 0);
    return 1;
}

static inline void _rb_bytes_in(rb_bytes_t *r, uint64_t pos, const void *src, uint64_t n) {
    uint64_t off = pos & r->mask, first = r->cap - off;
    if (first >= n) { memcpy(r->buf + off, src, n); return; }
    memcpy(r->buf + off, src, first);
    memcpy(r->buf, (const char *)src + first, n - first);
}

static inline void _rb_bytes_out(const rb_bytes_t *r, uint64_t pos, void *dst, uint64_t n) {
    uint64_t off = pos & r->mask, first = r->cap - off;
    if (first >= n) { memcpy(dst, r->buf + off, n); return; }
    memcpy(dst, r->buf + off, first);
    memcpy((char *)dst + first, r->buf, n - first);
}

/* Free bytes as seen by the producer (exact for the producer, which owns
 * tail; head only grows, so the true free space can only be larger). */
static inline uint64_t rb_bytes_free(rb_bytes_t *r) {
    uint64_t t = atomic_load_explicit(&r->tail, memory_order_relaxed);
    uint64_t h = atomic_load_explicit(&r->head, memory_order_acquire);
    return r->cap - (t - h);
}

/* Producer: append one record. Returns 1, or 0 when it does not fit right
 * now (caller decides whether to wait or count a drop). */
static inline int rb_bytes_push(rb_bytes_t *r, const char *data, uint32_t len) {
    uint64_t need = 4 + (uint64_t)len;
    uint64_t t = atomic_load_explicit(&r->tail, memory_order_relaxed);
    uint64_t h = atomic_load_explicit(&r->head, memory_order_acquire);
    if (r->cap - (t - h) < need) return 0;
    _rb_bytes_in(r, t, &len, 4);
    _rb_bytes_in(r, t + 4, data, len);
    atomic_store_explicit(&r->tail, t + need, memory_order_release);
    return 1;
}

/* Consumer: length of the next record, or -1 when empty. */
static inline int64_t rb_bytes_peek_len(rb_bytes_t *r) {
    uint64_t h = atomic_load_explicit(&r->head, memory_order_relaxed);
    uint64_t t = atomic_load_explicit(&r->tail, memory_order_acquire);
    if (h >= t) return -1;
    uint32_t len;
    _rb_bytes_out(r, h, &len, 4);
    return (int64_t)len;
}

/* Consumer: copy the next record (whose length rb_bytes_peek_len returned)
 * into dst and release its space. */
static inline void rb_bytes_pop(rb_bytes_t *r, char *dst, uint32_t len) {
    uint64_t h = atomic_load_explicit(&r->head, memory_order_relaxed);
    _rb_bytes_out(r, h + 4, dst, len);
    atomic_store_explicit(&r->head, h + 4 + len, memory_order_release);
}

static inline int rb_bytes_empty(rb_bytes_t *r) {
    return atomic_load_explicit(&r->head, memory_order_acquire) >=
           atomic_load_explicit(&r->tail, memory_order_acquire);
}

/* ── Append-only arena ────────────────────────────────────────────────────
 * Bump allocator: one atomic fetch-add per allocation (wait-free), reset by
 * the consumer side. Returns NULL when full; it never grows, since that
 * would need a lock on the hot path. */
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
 * Maps repeated names to small ids. A mutex-protected open-addressing table
 * is enough: the lock is taken once per distinct name, not per event. */
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
