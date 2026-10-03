/*
 * eBPF CO-RE scheduler tracer: shows whether an idle gap in a thread's spans
 * was the OS not running the thread (preempted, waiting for a core,
 * migrated) rather than the dependency the critical path assumes -- below
 * what any LD_PRELOAD/OMPT/PMPI hook can observe.
 *
 * Tracepoints (stable kernel ABI):
 *   sched_switch        one event per off-CPU period of a task (switched out
 *                       -> switched back in), comparable to a span's gap
 *   sched_wakeup        instant; pairing it with the task's next switch-in
 *                       would give run-queue latency (not computed yet)
 *   sched_migrate_task  instant; a task moved to another CPU
 *
 * Events reach userspace through a BPF ring buffer (os_tracer.c).
 *
 * Compiled only (clang -target bpf against this machine's BTF-generated
 * vmlinux.h); the kernel verifier has never run on it (no CAP_BPF here).
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

/* Kernel -> userspace ring buffer; 256 KB of headroom for scheduling bursts
 * between polls. */
struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 256 * 1024);
} events SEC(".maps");

/* Off-CPU start per pid, set at switch-out and consumed at the next
 * switch-in. A task that never comes back leaves an entry the LRU map
 * evicts; no event is emitted for it. */
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
