/*
 * Userspace loader for sched_trace.bpf.c: opens, loads and attaches the eBPF
 * program through its bpftool-generated skeleton, polls its ring buffer and
 * forwards events over HPROFILER_SOCKET in the hooks' wire format
 * (DOCUMENTATION.md, "Wire protocol").
 *
 * Not an LD_PRELOAD library: loading BPF needs CAP_BPF/root, a property of
 * how this process is started, so it runs as a separate privileged process:
 *   sudo HPROFILER_SOCKET=/tmp/hprofiler.sock ./os_tracer &
 *   hprofiler run --backend mpi -- mpirun -np 4 ./my_app
 *   kill %1
 *
 * Never loaded into a kernel: the development machine has
 * kernel.unprivileged_bpf_disabled=2 and no root, so open_and_load()/attach()
 * have only been reached up to the EPERM privilege check.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <time.h>
#include <sys/socket.h>
#include <sys/un.h>

#include <bpf/libbpf.h>
#include "sched_trace.skel.h"

#include "sched_event_fmt.h"

static volatile sig_atomic_t g_stop = 0;
static void on_signal(int sig) { (void)sig; g_stop = 1; }

static int g_sock = -1;

static void connect_socket(void) {
    const char *path = getenv("HPROFILER_SOCKET");
    if (!path) {
        fprintf(stderr, "os_tracer: HPROFILER_SOCKET not set, exiting\n");
        exit(1);
    }
    int s = socket(AF_UNIX, SOCK_STREAM, 0);
    if (s < 0) { perror("socket"); exit(1); }
    struct sockaddr_un addr = {0};
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, path, sizeof(addr.sun_path) - 1);
    if (connect(s, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        perror("connect");
        exit(1);
    }
    g_sock = s;
}

static void send_line(const char *buf, size_t len) {
    while (len > 0) {
        ssize_t n = send(g_sock, buf, len, MSG_NOSIGNAL);
        if (n <= 0) return;
        buf += n; len -= (size_t)n;
    }
}

static int handle_event(void *ctx, void *data, size_t data_sz) {
    (void)ctx;
    if (data_sz < sizeof(struct sched_event)) return 0;
    char line[384];
    int n = format_sched_event((const struct sched_event *)data, line, sizeof(line));
    if (n > 0) send_line(line, (size_t)n);
    return 0;
}

static int libbpf_print_fn(enum libbpf_print_level level, const char *fmt, va_list args) {
    if (level == LIBBPF_DEBUG) return 0;
    return vfprintf(stderr, fmt, args);
}

int main(void) {
    libbpf_set_print(libbpf_print_fn);
    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);

    connect_socket();

    struct sched_trace_bpf *skel = sched_trace_bpf__open_and_load();
    if (!skel) {
        fprintf(stderr, "os_tracer: failed to open/load BPF skeleton "
                "(needs root/CAP_BPF -- see DOCUMENTATION.md)\n");
        return 1;
    }
    if (sched_trace_bpf__attach(skel) != 0) {
        fprintf(stderr, "os_tracer: failed to attach BPF programs\n");
        sched_trace_bpf__destroy(skel);
        return 1;
    }

    struct ring_buffer *rb = ring_buffer__new(bpf_map__fd(skel->maps.events),
                                              handle_event, NULL, NULL);
    if (!rb) {
        fprintf(stderr, "os_tracer: failed to create ring buffer\n");
        sched_trace_bpf__destroy(skel);
        return 1;
    }

    fprintf(stderr, "os_tracer: attached, forwarding sched events to %s\n",
            getenv("HPROFILER_SOCKET"));
    while (!g_stop) {
        int err = ring_buffer__poll(rb, 200 /* ms */);
        if (err < 0 && err != -4 /* -EINTR */) {
            fprintf(stderr, "os_tracer: ring_buffer__poll error %d\n", err);
            break;
        }
    }

    ring_buffer__free(rb);
    sched_trace_bpf__destroy(skel);
    if (g_sock >= 0) close(g_sock);
    return 0;
}
