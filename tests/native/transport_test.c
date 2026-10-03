/*
 * Scenario tests for hooks/common/hp_transport.h, driven by
 * tests/test_transport_native.py (which builds this file plainly and under
 * ThreadSanitizer / AddressSanitizer+UBSan and runs one scenario per
 * process, since the transport reads its configuration once).
 *
 * The process hosts its own collector: a thread accepting connections on a
 * temporary AF_UNIX socket and checking every line it receives. Output is
 * one "RESULT k=v ..." line the Python side asserts on.
 *
 *   order     T threads x N records: every thread's sequence arrives
 *             complete and in order; final status accounts for everything
 *   long      10 KiB and 60 KiB records arrive intact; a 70 KiB record is
 *             rejected and counted (oversize), never truncated
 *   newline   a record with embedded line breaks stays one line
 *   overflow  16 KiB rings, no waiting, collector stalled: drops are counted
 *             exactly (received + dropped_full == emitted)
 *   wait      16 KiB rings, collector stalled briefly: producers wait
 *             (bounded) and nothing is dropped
 *   threads   short-lived threads: their rings are drained after they exit
 *   fork      parent and child both emit; the child reports its own pid and
 *             the parent's pre-fork records arrive exactly once
 *   nocollector  no listener: emitting never hangs, records counted as lost
 *   exec      records emitted before execv() arrive (drain before exec)
 */
#define _GNU_SOURCE
#include "../../hooks/common/hp_transport.h"

#include <sys/wait.h>

static char g_path[108];
static int g_listen = -1;
static _Atomic int g_stall = 0;          /* collector pauses reading */
static _Atomic int g_collector_done = 0;

#define MAXT 64
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static long g_lines, g_bad_order, g_gaps, g_xport_final, g_status_lines;
static long g_seq_next[MAXT];
static long g_long_ok, g_long_bad, g_newline_ok, g_pids_seen[8];
static int g_npids;
static unsigned long long g_images[8];
static int g_nimages;
static char g_last_status[700];

static void note_pid(long pid) {
    for (int i = 0; i < g_npids; i++) if (g_pids_seen[i] == pid) return;
    if (g_npids < 8) g_pids_seen[g_npids++] = pid;
}

static void handle_line(char *line, size_t len) {
    pthread_mutex_lock(&g_mu);
    if (strncmp(line, "xport:", 6) == 0) {
        g_status_lines++;
        if (strstr(line, "final=1")) g_xport_final++;
        const char *im = strstr(line, "image=");
        if (im) {
            unsigned long long v = strtoull(im + 6, NULL, 10);
            int known = 0;
            for (int i = 0; i < g_nimages; i++) known |= g_images[i] == v;
            if (!known && g_nimages < 8) g_images[g_nimages++] = v;
        }
        snprintf(g_last_status, sizeof(g_last_status), "%.*s", (int)len, line);
        pthread_mutex_unlock(&g_mu);
        return;
    }
    g_lines++;
    long pid = 0, t = 0, seq = 0;
    if (sscanf(line, "rec:%ld:%ld:%ld:", &pid, &t, &seq) == 3) {
        note_pid(pid);
        if (t >= 0 && t < MAXT) {
            /* a gap is a dropped record (overflow scenario); going
             * backwards is a reordering or duplicate -- never allowed */
            if (seq < g_seq_next[t]) g_bad_order++;
            else if (seq > g_seq_next[t]) g_gaps++;
            if (seq >= g_seq_next[t]) g_seq_next[t] = seq + 1;
        }
    } else if (strncmp(line, "long:", 5) == 0) {
        size_t want = (size_t)atol(line + 5);
        /* "long:<n>:" + 'x' * n */
        char *x = strchr(line + 5, ':');
        size_t got = x ? len - (size_t)(x + 1 - line) : 0;
        int ok = x && got == want;
        for (size_t i = 0; ok && i < got; i++) ok = x[1 + i] == 'x';
        if (ok) g_long_ok++; else g_long_bad++;
    } else if (strncmp(line, "nl:", 3) == 0) {
        if (strcmp(line, "nl:a b c") == 0) g_newline_ok++;
    }
    pthread_mutex_unlock(&g_mu);
}

static void *conn_main(void *arg) {
    int fd = (int)(intptr_t)arg;
    size_t cap = 1 << 20, used = 0;
    char *buf = malloc(cap);
    for (;;) {
        while (atomic_load(&g_stall)) usleep(1000);
        ssize_t r = recv(fd, buf + used, cap - used, 0);
        if (r <= 0) break;
        used += (size_t)r;
        size_t start = 0;
        for (size_t i = 0; i < used; i++) {
            if (buf[i] == '\n') {
                buf[i] = 0;
                handle_line(buf + start, i - start);
                start = i + 1;
            }
        }
        memmove(buf, buf + start, used - start);
        used -= start;
    }
    free(buf);
    close(fd);
    return NULL;
}

static void *collector_main(void *arg) {
    (void)arg;
    for (;;) {
        int fd = accept(g_listen, NULL, NULL);
        if (fd < 0) break;
        pthread_t t;
        pthread_create(&t, NULL, conn_main, (void *)(intptr_t)fd);
        pthread_detach(t);
    }
    atomic_store(&g_collector_done, 1);
    return NULL;
}

static void start_collector(int listen_ok) {
    snprintf(g_path, sizeof(g_path), "/tmp/hptx_%d.sock", (int)getpid());
    unlink(g_path);
    setenv("HPROFILER_SOCKET", g_path, 1);
    if (!listen_ok) return;
    g_listen = socket(AF_UNIX, SOCK_STREAM, 0);
    struct sockaddr_un a;
    memset(&a, 0, sizeof(a));
    a.sun_family = AF_UNIX;
    memcpy(a.sun_path, g_path, sizeof(a.sun_path) - 1);
    if (bind(g_listen, (struct sockaddr *)&a, sizeof(a)) || listen(g_listen, 64)) {
        perror("bind/listen");
        exit(2);
    }
    pthread_t t;
    pthread_create(&t, NULL, collector_main, NULL);
    pthread_detach(t);
}

/* wait until the collector has seen `n` data lines or `ms` elapsed */
static void wait_lines(long n, int ms) {
    for (int i = 0; i < ms; i++) {
        pthread_mutex_lock(&g_mu);
        long have = g_lines;
        pthread_mutex_unlock(&g_mu);
        if (have >= n) return;
        usleep(1000);
    }
}

static long status_field(const char *key) {
    pthread_mutex_lock(&g_mu);
    char pat[64];
    snprintf(pat, sizeof(pat), "%s=", key);
    char *p = strstr(g_last_status, pat);
    long v = p ? atol(p + strlen(pat)) : -1;
    pthread_mutex_unlock(&g_mu);
    return v;
}

static void finish(const char *scenario) {
    hp_tx_shutdown(1);
    usleep(200000);
    pthread_mutex_lock(&g_mu);
    printf("RESULT scenario=%s lines=%ld bad_order=%ld gaps=%ld final=%ld status_lines=%ld long_ok=%ld "
           "long_bad=%ld newline_ok=%ld pids=%d images=%d",
           scenario, g_lines, g_bad_order, g_gaps, g_xport_final, g_status_lines, g_long_ok, g_long_bad,
           g_newline_ok, g_npids, g_nimages);
    pthread_mutex_unlock(&g_mu);
    printf(" emitted=%ld sent=%ld dropped_full=%ld dropped_lost=%ld oversize=%ld waits=%ld sanitized=%ld\n",
           status_field("emitted"), status_field("sent"), status_field("dropped_full"),
           status_field("dropped_lost"), status_field("oversize"), status_field("waits"),
           status_field("sanitized"));
    fflush(stdout);
}

static void *release_later(void *p) {
    (void)p;
    usleep(300000);
    atomic_store(&g_stall, 0);
    return NULL;
}

typedef struct { int t, n; } worker_arg;
static void *worker(void *p) {
    worker_arg *a = (worker_arg *)p;
    for (int i = 0; i < a->n; i++)
        hp_tx_emitf("rec:%d:%d:%d:payload-%d\n", (int)hp_tx_pid(), a->t, i, i * 7);
    return NULL;
}

static void run_workers_from(int first, int threads, int n) {
    pthread_t th[MAXT];
    worker_arg args[MAXT];
    for (int t = 0; t < threads; t++) {
        args[t].t = first + t;
        args[t].n = n;
        pthread_create(&th[t], NULL, worker, &args[t]);
    }
    for (int t = 0; t < threads; t++) pthread_join(th[t], NULL);
}

static void run_workers(int threads, int n) { run_workers_from(0, threads, n); }

static void emit_long(size_t n) {
    char *b = malloc(n + 32);
    int k = snprintf(b, 32, "long:%zu:", n);
    memset(b + k, 'x', n);
    b[k + n] = '\n';
    hp_tx_emit(b, (size_t)k + n + 1);
    free(b);
}

int main(int argc, char **argv) {
    const char *sc = argc > 1 ? argv[1] : "order";
    if (strcmp(sc, "overflow") == 0) {
        setenv("HPROFILER_RING_KB", "16", 1);
        setenv("HPROFILER_RING_WAIT_MS", "0", 1);
    } else if (strcmp(sc, "wait") == 0) {
        setenv("HPROFILER_RING_KB", "16", 1);
        setenv("HPROFILER_RING_WAIT_MS", "5000", 1);
    }
    int exec_image = strcmp(sc, "exec") == 0 && argc > 2;
    if (!exec_image) start_collector(strcmp(sc, "nocollector") != 0);
    hp_tx_init("test");

    if (strcmp(sc, "order") == 0) {
        run_workers(8, 20000);
        wait_lines(160000, 20000);
    } else if (strcmp(sc, "long") == 0) {
        emit_long(10 * 1024);
        emit_long(60 * 1024);
        emit_long(70 * 1024);
        wait_lines(2, 5000);
    } else if (strcmp(sc, "newline") == 0) {
        hp_tx_emitf("nl:a\nb\rc\n");
        wait_lines(1, 5000);
    } else if (strcmp(sc, "overflow") == 0 || strcmp(sc, "wait") == 0) {
        atomic_store(&g_stall, 1);
        hp_tx_emitf("rec:%d:63:0:warmup\n", (int)hp_tx_pid());   /* connect first */
        usleep(100000);
        if (strcmp(sc, "wait") == 0) {
            pthread_t t;
            pthread_create(&t, NULL, release_later, NULL);   /* un-stall after 300 ms */
            pthread_detach(t);
        }
        run_workers(4, 20000);
        atomic_store(&g_stall, 0);
        wait_lines(80001, 20000);
    } else if (strcmp(sc, "threads") == 0) {
        for (int round = 0; round < 15; round++) run_workers_from(round * 4, 4, 500);
        wait_lines(30000, 10000);
    } else if (strcmp(sc, "fork") == 0) {
        for (int i = 0; i < 1000; i++) hp_tx_emitf("rec:%d:0:%d:parent\n", (int)hp_tx_pid(), i);
        pid_t c = fork();
        if (c == 0) {
            for (int i = 0; i < 1000; i++) hp_tx_emitf("rec:%d:1:%d:child\n", (int)hp_tx_pid(), i);
            hp_tx_shutdown(1);
            _exit(0);
        }
        for (int i = 1000; i < 2000; i++) hp_tx_emitf("rec:%d:0:%d:parent\n", (int)hp_tx_pid(), i);
        int st;
        waitpid(c, &st, 0);
        wait_lines(3000, 10000);
    } else if (strcmp(sc, "nocollector") == 0) {
        run_workers(2, 2000);
        hp_tx_shutdown(1);
        printf("RESULT scenario=nocollector returned=1\n");
        return 0;
    } else if (strcmp(sc, "exec") == 0) {
        if (argc > 2) {   /* the exec'd image: one more record, then exit */
            hp_tx_emitf("rec:%d:1:0:after-exec\n", (int)hp_tx_pid());
            hp_tx_shutdown(1);
            return 0;
        }
        pid_t c = fork();
        if (c == 0) {
            for (int i = 0; i < 500; i++) hp_tx_emitf("rec:%d:0:%d:before-exec\n", (int)hp_tx_pid(), i);
            char *args[] = { argv[0], "exec", "child", NULL };
            execv(argv[0], args);
            _exit(3);
        }
        int st;
        waitpid(c, &st, 0);
        wait_lines(501, 10000);
    }
    finish(sc);
    return 0;
}
