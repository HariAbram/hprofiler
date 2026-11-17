/*
 * eBPF CO-RE scheduler tracer: shows whether an idle gap in a thread's spans
 * was the OS not running the thread (preempted, waiting for a core,
 * migrated) rather than the dependency the critical path assumes -- below
 * what any LD_PRELOAD/OMPT/PMPI hook can observe.
 *
 * Tracepoints (stable kernel ABI):
 *   sched_switch        one event per off-CPU period of a task (switched out
 *                       -> switched back in), comparable to a span's gap;
 *                       sleep=1 when the task left the CPU to sleep, 0 when
 *                       it was preempted or yielded while runnable. A sleep
 *                       carries the waking that ended it (below)
 *   sched_waking        (BTF tracepoint) instant; runs in the WAKER's
 *                       context: the woken task (tid, tgid) and the waking
 *                       one (bpf_get_current_pid_tgid, comm), with ctx=task
 *                       when the waker is a task, irq when the wakeup runs
 *                       in interrupt context (then "current" is whatever was
 *                       interrupted, not the waker), unknown when the
 *                       kernel's preempt count cannot be read (no pcpu_hot)
 *   sched_migrate_task  instant; a task moved to another CPU
 *
 * Which waking ended a sleep is joined here, not by timestamps later: a
 * task that slept becomes runnable again only through try_to_wake_up,
 * which fires sched_waking once (later wakers find it running), so the
 * last waking recorded for the task when it is switched back in is the one
 * that ended the sleep -- also when the waking ran just before the
 * switch-out was traced, and whether or not ring-buffer records were
 * dropped. The record is lost only if the LRU map evicts it first (the
 * sleep then has no waker fields).
 *
 * Events reach userspace through a BPF ring buffer (os_tracer.c,
 * sched_event_fmt.h).
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

#define OFF_SLEEP 1    /* EV_OFFCPU flags: left the CPU to sleep */
#define OFF_WAKER 2    /*   the waking that ended the sleep is recorded */

#define WAKE_CTX_TASK    0
#define WAKE_CTX_IRQ     1
#define WAKE_CTX_UNKNOWN 2

/* sched_event_fmt.h has the userspace copy */
struct sched_event {
    __u32 kind;
    __u32 pid;        /* the task (thread id) the event is about */
    __u64 ts_ns;
    __u64 dur_ns;     /* EV_OFFCPU only */
    __s32 orig_cpu;   /* EV_MIGRATE only */
    __s32 dest_cpu;   /* EV_MIGRATE; EV_WAKEUP: the task's wake_cpu (the CPU the
                       * kernel tries first -- a migrate event follows if it
                       * picks another) */
    char  comm[16];
    __u32 tgid;       /* the task's process (EV_OFFCPU, EV_WAKEUP) */
    __u32 waker_tid;  /* EV_WAKEUP; EV_OFFCPU with OFF_WAKER: the waking task */
    __u32 waker_tgid;
    __u32 flags;      /* EV_OFFCPU: OFF_* */
    char  waker_comm[16];
    __u64 wake_ns;    /* EV_OFFCPU with OFF_WAKER: when the waking ran */
    __u32 wake_ctx;   /* EV_WAKEUP; EV_OFFCPU with OFF_WAKER: WAKE_CTX_* */
    __u32 pad;
};
_Static_assert(sizeof(struct sched_event) == 96 && __builtin_offsetof(struct sched_event, wake_ns) == 80,
               "sched_event_fmt.h mirrors this layout");

/* this CPU's preempt count (x86, kernel >= 6.2: pcpu_hot); weak: on
 * kernels without it the context is reported unknown */
extern struct pcpu_hot pcpu_hot __ksym __weak;

#define HARDIRQ_MASK_  0x000f0000
#define SOFTIRQ_OFFSET_ 0x00000100
#define NMI_MASK_      0x00f00000

static __always_inline __u32 wake_context(void) {
    if (!bpf_ksym_exists(&pcpu_hot))
        return WAKE_CTX_UNKNOWN;
    struct pcpu_hot *h = (struct pcpu_hot *)bpf_this_cpu_ptr(&pcpu_hot);
    int pc = BPF_CORE_READ(h, preempt_count);
    return (pc & (HARDIRQ_MASK_ | SOFTIRQ_OFFSET_ | NMI_MASK_)) ? WAKE_CTX_IRQ : WAKE_CTX_TASK;
}

/* Kernel -> userspace ring buffer; 256 KB of headroom for scheduling bursts
 * between polls. */
struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 256 * 1024);
} events SEC(".maps");

/* Off-CPU start per pid, set at switch-out and consumed at the next
 * switch-in. A task that never comes back leaves an entry the LRU map
 * evicts; no event is emitted for it. */
struct offcpu {
    __u64 ts;      /* off-cpu start ts_ns */
    __u32 sleep;   /* left to sleep (not preempted) */
    __u32 tgid;    /* its process: current is still the task at switch-out */
};
struct {
    __uint(type, BPF_MAP_TYPE_LRU_HASH);
    __uint(max_entries, 65536);
    __type(key, __u32);    /* pid */
    __type(value, struct offcpu);
} offcpu_start SEC(".maps");

/* The last waking of each pid, set by sched_waking and consumed when the
 * task is switched in after a sleep (see the file comment). */
struct waking {
    __u64 ts;
    __u32 waker_tid;
    __u32 waker_tgid;
    __u32 ctx;
    __u32 pad;
    char  comm[16];
};
struct {
    __uint(type, BPF_MAP_TYPE_LRU_HASH);
    __uint(max_entries, 65536);
    __type(key, __u32);    /* the woken pid */
    __type(value, struct waking);
} last_waking SEC(".maps");

SEC("tp/sched/sched_switch")
int handle_sched_switch(struct trace_event_raw_sched_switch *ctx) {
    __u64 now = bpf_ktime_get_ns();
    __u32 prev_pid = ctx->prev_pid;
    __u32 next_pid = ctx->next_pid;

    /* prev task is going off-CPU now. The tracepoint reports its state as
     * one TASK_REPORT bit (0x01..0x40: interruptible, uninterruptible,
     * stopped, ...) when it left to sleep, 0 when it yielded still
     * runnable, and TASK_REPORT_MAX (0x80) when it was preempted (kernels
     * >= 4.14; TASK_REPORT is 0x7f there). */
    if (prev_pid != 0) {  /* skip the swapper/idle task */
        struct offcpu o = { .ts = now, .sleep = (ctx->prev_state & 0x7f) != 0,
                            .tgid = (__u32)(bpf_get_current_pid_tgid() >> 32) };
        bpf_map_update_elem(&offcpu_start, &prev_pid, &o, BPF_ANY);
    }

    /* next task is coming ON-CPU now -- if we recorded when it went off,
     * that period just ended; emit it and clear the entry. */
    if (next_pid != 0) {
        struct offcpu *start = bpf_map_lookup_elem(&offcpu_start, &next_pid);
        if (start) {
            struct sched_event *ev = bpf_ringbuf_reserve(&events, sizeof(*ev), 0);
            if (ev) {
                __builtin_memset(ev, 0, sizeof(*ev));
                ev->kind = EV_OFFCPU;
                ev->pid = next_pid;
                ev->tgid = start->tgid;
                ev->ts_ns = start->ts;
                ev->dur_ns = now - start->ts;
                __builtin_memcpy(ev->comm, ctx->next_comm, sizeof(ev->comm));
                if (start->sleep) {
                    ev->flags = OFF_SLEEP;
                    struct waking *w = bpf_map_lookup_elem(&last_waking, &next_pid);
                    if (w) {
                        ev->flags |= OFF_WAKER;
                        ev->wake_ns = w->ts;
                        ev->waker_tid = w->waker_tid;
                        ev->waker_tgid = w->waker_tgid;
                        ev->wake_ctx = w->ctx;
                        __builtin_memcpy(ev->waker_comm, w->comm, sizeof(ev->waker_comm));
                    }
                }
                bpf_ringbuf_submit(ev, 0);
            }
            if (start->sleep)
                bpf_map_delete_elem(&last_waking, &next_pid);
            bpf_map_delete_elem(&offcpu_start, &next_pid);
        }
    }
    return 0;
}

SEC("tp_btf/sched_waking")
int BPF_PROG(handle_sched_waking, struct task_struct *p) {
    __u64 waker = bpf_get_current_pid_tgid();
    __u32 pid = BPF_CORE_READ(p, pid);
    struct waking w = { .ts = bpf_ktime_get_ns(), .waker_tid = (__u32)waker,
                        .waker_tgid = (__u32)(waker >> 32), .ctx = wake_context() };
    bpf_get_current_comm(w.comm, sizeof(w.comm));
    bpf_map_update_elem(&last_waking, &pid, &w, BPF_ANY);

    struct sched_event *ev = bpf_ringbuf_reserve(&events, sizeof(*ev), 0);
    if (!ev) return 0;
    __builtin_memset(ev, 0, sizeof(*ev));
    ev->kind = EV_WAKEUP;
    ev->pid = pid;
    ev->tgid = BPF_CORE_READ(p, tgid);
    ev->ts_ns = w.ts;
    ev->dest_cpu = BPF_CORE_READ(p, wake_cpu);
    BPF_CORE_READ_STR_INTO(&ev->comm, p, comm);
    ev->waker_tid = w.waker_tid;
    ev->waker_tgid = w.waker_tgid;
    ev->wake_ctx = w.ctx;
    __builtin_memcpy(ev->waker_comm, w.comm, sizeof(ev->waker_comm));
    bpf_ringbuf_submit(ev, 0);
    return 0;
}

SEC("tp/sched/sched_migrate_task")
int handle_sched_migrate_task(struct trace_event_raw_sched_migrate_task *ctx) {
    struct sched_event *ev = bpf_ringbuf_reserve(&events, sizeof(*ev), 0);
    if (!ev) return 0;
    __builtin_memset(ev, 0, sizeof(*ev));
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
