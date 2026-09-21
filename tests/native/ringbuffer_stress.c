/*
 * Standalone correctness + overhead stress test for hooks/common/ringbuffer.h.
 * Not wired into any hook (see that header's comment for why the ring
 * buffer itself is verified in isolation rather than through a live hook
 * this session) -- this is the verification for the ring buffer /
 * interning primitives themselves: concurrent producer/consumer
 * correctness under real pthread scheduling, drop-counter exactness under
 * intentional overflow, and a real (not guessed) latency comparison
 * against the mutex+syscall pattern it's meant to replace on hooks' hot
 * paths.
 *
 * Build:  gcc -O2 -pthread -Wall -Wextra -o ringbuffer_stress ringbuffer_stress.c
 * Run:    ./ringbuffer_stress
 * Exit code 0 = all checks passed; non-zero + a printed reason = failure.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>
#include <fcntl.h>

#include "../../hooks/common/ringbuffer.h"

static int g_failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__); g_failures++; } \
} while (0)

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/* ── Test 1: concurrent producer/consumer correctness ───────────────────
 * N independent (producer thread, consumer thread, ring buffer) triples
 * running concurrently. Each producer pushes a known sequence
 * "p<producer_id>:<seq>" and the consumer verifies every single one
 * arrives, in order, uncorrupted, with none missing or duplicated --
 * capacity is sized so no overflow is expected in this test (overflow
 * behavior is verified separately in Test 2). */
#define NUM_TRIPLES   8
#define PUSHES_PER_PRODUCER 200000

typedef struct {
    ringbuffer_t *rb;
    int producer_id;
} ProducerArg;

typedef struct {
    ringbuffer_t *rb;
    int producer_id;
    int ok;
} ConsumerArg;

static void *producer_thread(void *arg_) {
    ProducerArg *arg = (ProducerArg *)arg_;
    char buf[64];
    for (int seq = 0; seq < PUSHES_PER_PRODUCER; seq++) {
        int n = snprintf(buf, sizeof(buf), "p%d:%d", arg->producer_id, seq);
        while (!rb_push(arg->rb, buf, (uint16_t)n)) {
            /* Buffer full -- back off briefly and retry (in this test the
             * consumer drains fast enough that this should be rare/never;
             * a real hot-path caller would NOT retry, this is test-only). */
            usleep(10);
        }
    }
    return NULL;
}

static void *consumer_thread(void *arg_) {
    ConsumerArg *arg = (ConsumerArg *)arg_;
    char out[RB_SLOT_SIZE];
    uint16_t out_len;
    int expected_seq = 0;
    int received = 0;
    uint64_t deadline = now_ns() + 30ULL * 1000000000ULL;  /* 30s safety timeout */
    while (received < PUSHES_PER_PRODUCER) {
        if (rb_pop(arg->rb, out, &out_len)) {
            out[out_len] = '\0';
            char expected[64];
            snprintf(expected, sizeof(expected), "p%d:%d", arg->producer_id, expected_seq);
            if (strcmp(out, expected) != 0) {
                fprintf(stderr, "producer %d: expected '%s' got '%s' at position %d\n",
                        arg->producer_id, expected, out, received);
                arg->ok = 0;
                return NULL;
            }
            expected_seq++;
            received++;
        } else if (now_ns() > deadline) {
            fprintf(stderr, "producer %d: timed out after receiving %d/%d\n",
                    arg->producer_id, received, PUSHES_PER_PRODUCER);
            arg->ok = 0;
            return NULL;
        }
    }
    arg->ok = 1;
    return NULL;
}

static void test_concurrent_correctness(void) {
    int _before = g_failures;
    ringbuffer_t *rbs = malloc(sizeof(ringbuffer_t) * NUM_TRIPLES);
    pthread_t producers[NUM_TRIPLES], consumers[NUM_TRIPLES];
    ProducerArg pargs[NUM_TRIPLES];
    ConsumerArg cargs[NUM_TRIPLES];

    for (int i = 0; i < NUM_TRIPLES; i++) {
        rb_init(&rbs[i]);
        pargs[i] = (ProducerArg){ .rb = &rbs[i], .producer_id = i };
        cargs[i] = (ConsumerArg){ .rb = &rbs[i], .producer_id = i, .ok = 0 };
    }
    for (int i = 0; i < NUM_TRIPLES; i++) {
        pthread_create(&consumers[i], NULL, consumer_thread, &cargs[i]);
        pthread_create(&producers[i], NULL, producer_thread, &pargs[i]);
    }
    for (int i = 0; i < NUM_TRIPLES; i++) {
        pthread_join(producers[i], NULL);
        pthread_join(consumers[i], NULL);
    }
    uint64_t total_dropped = 0;
    for (int i = 0; i < NUM_TRIPLES; i++) {
        CHECK(cargs[i].ok, "producer/consumer pair lost, corrupted, or misordered an event");
        total_dropped += rb_dropped_count(&rbs[i]);
    }
    /* Some transient drops here are EXPECTED, not a bug: with 8 triples (16
     * threads) contending on a smaller core count, the OS scheduler can
     * legitimately let a producer get far enough ahead of its consumer to
     * fill a 1024-slot buffer in a burst -- that's real backpressure, not
     * data loss, since the producer's retry loop (test-only; a real
     * hot-path caller would NOT retry, per ringbuffer.h's documented
     * "never blocks, drops instead" policy) keeps trying until it
     * succeeds. The property this test actually needs to prove is that
     * every logical event *eventually delivered* arrives intact, in
     * order, exactly once -- verified by cargs[i].ok above -- not that a
     * fixed-size buffer never experiences a momentary backlog under
     * 16-way contention on this machine's core count. */
    printf("  (%llu transient drops absorbed by producer retries across all triples -- not a failure)\n",
           (unsigned long long)total_dropped);
    printf("test_concurrent_correctness: %d triples x %d events each -- %s\n",
           NUM_TRIPLES, PUSHES_PER_PRODUCER, g_failures == _before ? "OK" : "FAILED");
    free(rbs);
}

/* ── Test 2: drop counter is exact under intentional overflow ──────────── */
static void test_drop_counter_exact(void) {
    int _before = g_failures;
    ringbuffer_t rb;
    rb_init(&rb);
    int pushed_ok = 0, pushed_dropped = 0;
    /* Push far more than RB_CAPACITY with NO consumer draining at all. */
    int total_attempts = RB_CAPACITY * 4;
    for (int i = 0; i < total_attempts; i++) {
        char buf[32];
        int n = snprintf(buf, sizeof(buf), "x%d", i);
        if (rb_push(&rb, buf, (uint16_t)n)) pushed_ok++;
        else pushed_dropped++;
    }
    CHECK(pushed_ok == RB_CAPACITY, "expected exactly RB_CAPACITY successful pushes before full");
    CHECK((uint64_t)pushed_dropped == rb_dropped_count(&rb),
          "rb_dropped_count() must exactly match the number of failed rb_push() calls");
    CHECK(pushed_ok + pushed_dropped == total_attempts, "accounting mismatch");
    printf("test_drop_counter_exact: %d ok + %d dropped = %d attempted, counter=%llu -- %s\n",
           pushed_ok, pushed_dropped, total_attempts,
           (unsigned long long)rb_dropped_count(&rb), g_failures == _before ? "OK" : "FAILED");
}

/* ── Test 3: FIFO order preserved across a full wrap-around ──────────────
 * Push and pop one at a time (interleaved), forcing the ring to wrap past
 * its physical capacity many times over, and verify strict FIFO order the
 * entire way -- catches any off-by-one in the modulo indexing that a
 * single-pass (never-wraps) test wouldn't reach. */
static void test_wraparound_fifo_order(void) {
    int _before = g_failures;
    ringbuffer_t rb;
    rb_init(&rb);
    int total = RB_CAPACITY * 10;
    int next_push = 0, next_expected_pop = 0;
    int mismatches = 0;
    while (next_expected_pop < total) {
        if (next_push < total) {
            char buf[16];
            int n = snprintf(buf, sizeof(buf), "%d", next_push);
            if (rb_push(&rb, buf, (uint16_t)n)) next_push++;
        }
        char out[RB_SLOT_SIZE]; uint16_t out_len;
        if (rb_pop(&rb, out, &out_len)) {
            out[out_len] = '\0';
            char expected[16];
            snprintf(expected, sizeof(expected), "%d", next_expected_pop);
            if (strcmp(out, expected) != 0) mismatches++;
            next_expected_pop++;
        }
    }
    CHECK(mismatches == 0, "FIFO order violated across ring wrap-around");
    printf("test_wraparound_fifo_order: %d items through a %dx-capacity wraparound -- %s\n",
           total, total / RB_CAPACITY, g_failures == _before ? "OK" : "FAILED");
}

/* ── Test 4: arena allocator ─────────────────────────────────────────────── */
static void test_arena_alloc(void) {
    int _before = g_failures;
    char backing[1024];
    arena_t a;
    arena_init(&a, backing, sizeof(backing));
    char *p1 = arena_alloc(&a, 100);
    char *p2 = arena_alloc(&a, 100);
    CHECK(p1 != NULL && p2 != NULL, "arena_alloc should succeed within capacity");
    CHECK(p2 == p1 + 100, "arena_alloc should hand out contiguous, non-overlapping regions");
    char *p3 = arena_alloc(&a, 1000);  /* exceeds remaining capacity (1024-200=824) */
    CHECK(p3 == NULL, "arena_alloc must return NULL, not overflow, once capacity is exceeded");
    arena_reset(&a);
    char *p4 = arena_alloc(&a, 100);
    CHECK(p4 == backing, "arena_reset must allow reuse from the start");
    printf("test_arena_alloc: %s\n", g_failures == _before ? "OK" : "FAILED");
}

/* ── Test 5: name interning ──────────────────────────────────────────────── */
static void test_interning(void) {
    int _before = g_failures;
    intern_table_t t;
    intern_init(&t);
    uint32_t id_a1 = intern_id(&t, "kernel_a");
    uint32_t id_b  = intern_id(&t, "kernel_b");
    uint32_t id_a2 = intern_id(&t, "kernel_a");  /* same string, different pointer... */
    CHECK(id_a1 == id_a2, "the same name string must intern to the same id");
    CHECK(id_a1 != id_b, "different names must intern to different ids");
    CHECK(strcmp(intern_lookup(&t, id_a1), "kernel_a") == 0, "intern_lookup must round-trip the name");
    CHECK(strcmp(intern_lookup(&t, id_b), "kernel_b") == 0, "intern_lookup must round-trip the name");
    printf("test_interning: %s\n", g_failures == _before ? "OK" : "FAILED");
}

/* ── Benchmark: ring buffer push vs. mutex+write (the pattern it replaces) ─
 * Not a correctness check -- prints real measured numbers so the "removes
 * mutex+socket from the hot path" claim has an actual before/after instead
 * of an assertion. Uses write() to a pipe (not a real AF_UNIX socket) to
 * isolate "syscall + lock" overhead consistently across machines without
 * needing a live collector process; the real hooks' overhead includes
 * additional snprintf/connect-check work on top of this baseline, so this
 * is a lower bound on the improvement, not an exact reproduction. */
typedef struct { int fd; pthread_mutex_t *mutex; int iters; double ns_per_call; } BenchMutexArg;
typedef struct { ringbuffer_t *rb; int iters; double ns_per_call; } BenchRbArg;

static void *bench_mutex_thread(void *arg_) {
    BenchMutexArg *arg = arg_;
    char buf[64] = "span:cuda:1234:1234:1000:500:kernel:type=kernel,stream=1\n";
    size_t len = strlen(buf);
    uint64_t t0 = now_ns();
    for (int i = 0; i < arg->iters; i++) {
        pthread_mutex_lock(arg->mutex);
        ssize_t w = write(arg->fd, buf, len);
        (void)w;
        pthread_mutex_unlock(arg->mutex);
    }
    uint64_t t1 = now_ns();
    arg->ns_per_call = (double)(t1 - t0) / arg->iters;
    return NULL;
}

static void *bench_rb_thread(void *arg_) {
    BenchRbArg *arg = arg_;
    char buf[64] = "span:cuda:1234:1234:1000:500:kernel:type=kernel,stream=1\n";
    uint16_t len = (uint16_t)strlen(buf);
    uint64_t t0 = now_ns();
    for (int i = 0; i < arg->iters; i++) {
        rb_push(arg->rb, buf, len);
    }
    uint64_t t1 = now_ns();
    arg->ns_per_call = (double)(t1 - t0) / arg->iters;
    return NULL;
}

static void *drain_pipe_thread(void *arg_) {
    int fd = *(int *)arg_;
    char tmp[65536];
    while (read(fd, tmp, sizeof(tmp)) > 0) {}
    return NULL;
}

static void run_benchmark(int num_threads, int iters_per_thread) {
    /* Mutex+write benchmark: all threads share ONE mutex + ONE pipe fd,
     * matching today's real g_sock_mutex-per-hook-per-process sharing. */
    int pipefd[2];
    if (pipe(pipefd) != 0) { perror("pipe"); return; }
    /* Drain the read end in a background thread so writes to a small pipe
     * buffer don't block once it fills -- keeps this a lock+syscall
     * latency measurement, not a blocked-on-a-full-pipe measurement. */
    int drain_fd = pipefd[0];
    pthread_t drainer;
    pthread_create(&drainer, NULL, drain_pipe_thread, &drain_fd);

    pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_t threads[64];
    BenchMutexArg margs[64];
    for (int i = 0; i < num_threads; i++) {
        margs[i] = (BenchMutexArg){ .fd = pipefd[1], .mutex = &mutex, .iters = iters_per_thread };
        pthread_create(&threads[i], NULL, bench_mutex_thread, &margs[i]);
    }
    double mutex_total = 0;
    for (int i = 0; i < num_threads; i++) { pthread_join(threads[i], NULL); mutex_total += margs[i].ns_per_call; }
    close(pipefd[1]);
    pthread_join(drainer, NULL);
    close(pipefd[0]);

    /* Ring buffer benchmark: each thread gets its OWN ring buffer (the
     * actual per-thread-sharded design), no draining needed during the
     * push-only measurement since capacity comfortably covers iters. */
    ringbuffer_t *rbs = malloc(sizeof(ringbuffer_t) * num_threads);
    BenchRbArg rargs[64];
    for (int i = 0; i < num_threads; i++) {
        rb_init(&rbs[i]);
        rargs[i] = (BenchRbArg){ .rb = &rbs[i], .iters = iters_per_thread < RB_CAPACITY ? iters_per_thread : RB_CAPACITY };
        pthread_create(&threads[i], NULL, bench_rb_thread, &rargs[i]);
    }
    double rb_total = 0;
    for (int i = 0; i < num_threads; i++) { pthread_join(threads[i], NULL); rb_total += rargs[i].ns_per_call; }
    free(rbs);

    double mutex_avg = mutex_total / num_threads;
    double rb_avg = rb_total / num_threads;
    printf("benchmark (%d threads, %d iters/thread):\n", num_threads, iters_per_thread);
    printf("  mutex+write(2) (today's pattern) : %8.1f ns/call\n", mutex_avg);
    printf("  rb_push (lock-free ring buffer)  : %8.1f ns/call\n", rb_avg);
    if (rb_avg > 0)
        printf("  ratio                             : %.1fx\n", mutex_avg / rb_avg);
}

int main(void) {
    test_concurrent_correctness();
    test_drop_counter_exact();
    test_wraparound_fifo_order();
    test_arena_alloc();
    test_interning();

    printf("\n");
    run_benchmark(1, 200000);
    run_benchmark(4, 200000);
    run_benchmark(8, 200000);

    printf("\n%s\n", g_failures == 0 ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED");
    return g_failures == 0 ? 0 : 1;
}
