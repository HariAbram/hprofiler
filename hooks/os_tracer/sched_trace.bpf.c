/*
 * eBPF CO-RE (Compile Once - Run Everywhere) OS-level scheduler tracer.
 *
 * Answers a question hprofiler's other hooks structurally cannot: when a
 * thread's span shows an idle gap (e.g. inside MPI_Wait, or between two
 * kernel launches), was that gap *caused* by the thing the causal DAG
 * thinks it's waiting on, or was the OS scheduler simply not running that
 * thread on a CPU during that window (preempted by another process,
 * waiting for a free core, migrated between NUMA nodes)? None of the
 * LD_PRELOAD/OMPT/PMPI hooks can see this -- it's below the userspace
 * boundary entirely.
 *
 * Attaches to three sched tracepoints (kernel-standardized ABI, present on
 * any Linux kernel with CONFIG_SCHED_TRACER, essentially universal):
 *   sched_switch        -- a CPU stopped running one task and started
 *                          running another. This program tracks, per
 *                          pid, how long it was off-CPU between being
 *                          switched out and switched back in, and emits
 *                          ONE event per off-CPU period (not per
 *                          switch) -- directly comparable to an
 *                          hprofiler span: an "off-cpu" duration a
 *                          Python-side consumer can overlap against any
 *                          instrumented span's idle gap.
 *   sched_wakeup        -- a sleeping task became runnable. Emitted as a
 *                          raw instant; pairing it with the NEXT
 *                          sched_switch that switches the same pid in
 *                          gives run-queue (scheduling) latency, i.e.
 *                          "wanted to run but no core was free yet" --
 *                          left as a documented follow-on Python-side
 *                          computation rather than decomposed in-kernel,
 *                          to keep this program's per-event kernel-side
 *                          work minimal.
 *   sched_migrate_task  -- a task moved to a different CPU (often across
 *                          NUMA nodes) -- a common, otherwise invisible
 *                          cause of an unexplained slowdown.
 *
 * Events are delivered to userspace via a BPF ring buffer (the modern,
 * simpler replacement for the older perf-buffer API) -- see os_tracer.c
 * for the loader/consumer.
 *
 * ── Verification status ─────────────────────────────────────────────────
 * Compile-checked only (`clang -target bpf`, using this machine's own
 * BTF-derived vmlinux.h for CO-RE field access -- see DOCUMENTATION.md).
 * `kernel.unprivileged_bpf_disabled=2` on this development machine blocks
 * loading/attaching without root/CAP_BPF, which was not available/
 * authorized in this session -- the kernel BPF *verifier* (a distinct,
 * additional pass beyond compilation that rejects programs violating
 * memory-safety/termination rules the C compiler itself doesn't check)
 * has never actually run on this program. Treat as unverified beyond
 * "clang accepted the C" until confirmed loadable on a machine where BPF
 * loading is permitted.
 */
#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>

char LICENSE[] SEC("license") = "GPL";

/* Event kinds delivered through the ring buffer -- mirrors the wire
 * protocol's span/inst distinction (see os_tracer.c's forwarding). */
#define EV_OFFCPU  1   /* span: pid was off-CPU for dur_ns */
#define EV_WAKEUP  2   /* instant: pid became runnable */
#define EV_MIGRATE 3   /* instant: pid moved from cpu_a to cpu_b (orig_cpu/dest_cpu) */

struct sched_event {
    __u32 kind;
    __u32 pid;
    __u64 ts_ns;
    __u64 dur_ns;     /* EV_OFFCPU only */
    __s32 orig_cpu;   /* EV_MIGRATE only */
    __s32 dest_cpu;   /* EV_MIGRATE only */
    char  comm[16];
};

/* Ring buffer for kernel -> userspace event delivery. 256KB is generous
 * headroom for burst scheduling activity between userspace poll
 * iterations; sized independently of hooks/common/ringbuffer.h (that one
 * is userspace-only, for the LD_PRELOAD hooks' own hot path -- this is
 * the separate kernel/userspace boundary ring buffer BPF itself provides). */
struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 256 * 1024);
} events SEC(".maps");

/* Per-pid off-CPU start timestamp, populated when a task is switched OUT,
 * consumed (and cleared) when that same pid is next switched IN. A task
 * that's switched out and never switched back in within the trace window
 * (e.g. it exits) simply leaves a stale entry that's naturally evicted by
 * LRU -- not a leak, and not surfaced as a spurious event, since no
 * matching switch-IN ever reads it. */
struct {
    __uint(type, BPF_MAP_TYPE_LRU_HASH);
    __uint(max_entries, 65536);
    __type(key, __u32);    /* pid */
    __type(value, __u64);  /* off-cpu start ts_ns */
} offcpu_start SEC(".maps");

SEC("tp/sched/sched_switch")
int handle_sched_switch(struct trace_event_raw_sched_switch *ctx) {
    __u64 now = bpf_ktime_get_ns();
    __u32 prev_pid = ctx->prev_pid;
    __u32 next_pid = ctx->next_pid;

    /* prev task is going off-CPU now. */
    if (prev_pid != 0) {  /* skip the swapper/idle task */
        bpf_map_update_elem(&offcpu_start, &prev_pid, &now, BPF_ANY);
    }

    /* next task is coming ON-CPU now -- if we recorded when it went off,
     * that period just ended; emit it and clear the entry. */
    if (next_pid != 0) {
        __u64 *start = bpf_map_lookup_elem(&offcpu_start, &next_pid);
        if (start) {
            struct sched_event *ev = bpf_ringbuf_reserve(&events, sizeof(*ev), 0);
            if (ev) {
                ev->kind = EV_OFFCPU;
                ev->pid = next_pid;
                ev->ts_ns = *start;
                ev->dur_ns = now - *start;
                ev->orig_cpu = 0;
                ev->dest_cpu = 0;
                __builtin_memcpy(ev->comm, ctx->next_comm, sizeof(ev->comm));
                bpf_ringbuf_submit(ev, 0);
            }
            bpf_map_delete_elem(&offcpu_start, &next_pid);
        }
    }
    return 0;
}

SEC("tp/sched/sched_wakeup")
int handle_sched_wakeup(struct trace_event_raw_sched_wakeup_template *ctx) {
    struct sched_event *ev = bpf_ringbuf_reserve(&events, sizeof(*ev), 0);
    if (!ev) return 0;
    ev->kind = EV_WAKEUP;
    ev->pid = ctx->pid;
    ev->ts_ns = bpf_ktime_get_ns();
    ev->dur_ns = 0;
    ev->orig_cpu = 0;
    ev->dest_cpu = ctx->target_cpu;
    __builtin_memcpy(ev->comm, ctx->comm, sizeof(ev->comm));
    bpf_ringbuf_submit(ev, 0);
    return 0;
}

SEC("tp/sched/sched_migrate_task")
int handle_sched_migrate_task(struct trace_event_raw_sched_migrate_task *ctx) {
    struct sched_event *ev = bpf_ringbuf_reserve(&events, sizeof(*ev), 0);
    if (!ev) return 0;
    ev->kind = EV_MIGRATE;
    ev->pid = ctx->pid;
    ev->ts_ns = bpf_ktime_get_ns();
    ev->dur_ns = 0;
    ev->orig_cpu = ctx->orig_cpu;
    ev->dest_cpu = ctx->dest_cpu;
    __builtin_memcpy(ev->comm, ctx->comm, sizeof(ev->comm));
    bpf_ringbuf_submit(ev, 0);
    return 0;
}
