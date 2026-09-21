/*
 * Userspace loader for sched_trace.bpf.c -- opens/loads/attaches the eBPF
 * program via its bpftool-generated skeleton, polls its ring buffer, and
 * forwards events into hprofiler's existing wire protocol (the same
 * newline-delimited ASCII over HPROFILER_SOCKET every other hook uses --
 * see DOCUMENTATION.md §12).
 *
 * Unlike every other hook, this is NOT an LD_PRELOAD library -- eBPF
 * loading needs CAP_BPF/root, which is a property of how THIS process is
 * launched, not something an LD_PRELOAD shim injected into an arbitrary
 * unprivileged profiled program could obtain. Run alongside the profiled
 * program as a separate, explicitly-privileged process:
 *   sudo HPROFILER_SOCKET=/tmp/hprofiler.sock ./os_tracer &
 *   hprofiler run --backend mpi -- mpirun -np 4 ./my_app
 *   kill %1
 *
 * ── Verification status ─────────────────────────────────────────────────
 * Compile-checked only. This development machine has
 * kernel.unprivileged_bpf_disabled=2, which blocks loading/attaching any
 * BPF program without root/CAP_BPF -- not available/authorized in this
 * session, so sched_trace_bpf__open_and_load()/__attach() below have
 * never actually executed here; only the surrounding C compiles. See
 * DOCUMENTATION.md's Known Limitations.
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

/* Mirrors sched_trace.bpf.c's struct sched_event exactly -- kept in sync
 * manually since the BPF side can't #include a shared userspace header
 * (different compilation target). */
#define EV_OFFCPU  1
#define EV_WAKEUP  2
#define EV_MIGRATE 3

struct sched_event {
    unsigned int kind;
    unsigned int pid;
    unsigned long long ts_ns;
    unsigned long long dur_ns;
    int orig_cpu;
    int dest_cpu;
    char comm[16];
};

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

/* Replaces any wire-protocol-significant character (':' -- the top-level
 * field separator; ',' -- the tag separator; '=' -- the tag key/value
 * separator) with '_' before a comm string is embedded in a tag value.
 * Linux thread names (comm, max 16 bytes) are usually plain identifiers
 * but are NOT guaranteed to be -- a userspace program can set an
 * arbitrary comm via prctl(PR_SET_NAME) or pthread_setname_np(), so this
 * cannot be skipped. Without it, a comm containing e.g. ':' could corrupt
 * the emitted record's name/tags boundary exactly the way an unsanitized
 * rmatches= value did in mpi_hook.c earlier in this same redesign (see
 * DOCUMENTATION.md §12's "Tag values must never contain ':'" note) --
 * this hook produces the record directly (no snprintf-time truncation
 * retry like the LD_PRELOAD hooks have), so sanitizing at the source is
 * the only guard here. */
static void sanitize_comm(char *out, const char *in, size_t cap) {
    size_t i = 0;
    for (; in[i] != '\0' && i < cap - 1; i++) {
        char c = in[i];
        out[i] = (c == ':' || c == ',' || c == '=') ? '_' : c;
    }
    out[i] = '\0';
}

static int handle_event(void *ctx, void *data, size_t data_sz) {
    (void)ctx;
    if (data_sz < sizeof(struct sched_event)) return 0;
    const struct sched_event *ev = data;
    char comm[17];
    sanitize_comm(comm, ev->comm, sizeof(comm));

    char line[256];
    int n = 0;
    switch (ev->kind) {
        case EV_OFFCPU:
            n = snprintf(line, sizeof(line),
                "span:sched:0:%u:%llu:%llu:off_cpu:comm=%s\n",
                ev->pid, ev->ts_ns, ev->dur_ns, comm);
            break;
        case EV_WAKEUP:
            n = snprintf(line, sizeof(line),
                "inst:sched:0:%u:%llu:wakeup:comm=%s,target_cpu=%d\n",
                ev->pid, ev->ts_ns, comm, ev->dest_cpu);
            break;
        case EV_MIGRATE:
            n = snprintf(line, sizeof(line),
                "inst:sched:0:%u:%llu:migrate:comm=%s,orig_cpu=%d,dest_cpu=%d\n",
                ev->pid, ev->ts_ns, comm, ev->orig_cpu, ev->dest_cpu);
            break;
        default:
            return 0;
    }
    if (n > 0 && n < (int)sizeof(line)) send_line(line, (size_t)n);
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
