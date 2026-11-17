/*
 * The scheduler events sched_trace.bpf.c delivers and their wire lines
 * (DOCUMENTATION.md, "OS scheduler tracer (eBPF)"). Shared by os_tracer.c
 * and tests/native/sched_event_fmt_test.c; no libbpf here.
 *
 *   off_cpu  span:sched:0:<tid>:<start>:<dur>:off_cpu:comm=..,task_pid=..,
 *            sleep=0|1[,waker_pid=..,waker_tid=..,waker_comm=..,ctx=..,
 *            wake_ns=..]
 *            sleep=1: the task left the CPU to sleep; 0: it was preempted,
 *            or yielded, while runnable. A sleep names the waking that
 *            ended it (joined in the kernel, see sched_trace.bpf.c), unless
 *            that record was evicted
 *   wakeup   inst:sched:0:<tid>:<ts>:wakeup:comm=..,wake_cpu=..,
 *            wakee_pid=..,waker_pid=..,waker_tid=..,waker_comm=..,ctx=..
 *   migrate  inst:sched:0:<tid>:<ts>:migrate:comm=..,orig_cpu=..,dest_cpu=..
 *
 * ctx=task|irq|unknown: the waker is the task the kernel ran the waking in,
 * a real waker for ctx=task only (irq: whatever the interrupt hit; unknown:
 * the preempt count could not be read). wake_cpu: the CPU the kernel tries
 * first (task_struct.wake_cpu). The process field is 0: the kernel's ids
 * are thread ids (the hooks' tid field); task_pid / wakee_pid / waker_pid
 * name the processes -- in the kernel's initial pid namespace.
 *
 * Before 2026-10 (sched_wakeup): off_cpu had comm= only, wakeup comm= and
 * target_cpu= only -- no sleep flag, no processes, no waker.
 */
#ifndef SCHED_EVENT_FMT_H
#define SCHED_EVENT_FMT_H

#include <stdio.h>

/* Must match sched_trace.bpf.c's struct sched_event (kept in sync by hand:
 * the BPF side cannot include a userspace header). */
#define EV_OFFCPU  1
#define EV_WAKEUP  2
#define EV_MIGRATE 3

#define OFF_SLEEP 1
#define OFF_WAKER 2

#define WAKE_CTX_TASK    0
#define WAKE_CTX_IRQ     1
#define WAKE_CTX_UNKNOWN 2

struct sched_event {
    unsigned int kind;
    unsigned int pid;
    unsigned long long ts_ns;
    unsigned long long dur_ns;
    int orig_cpu;
    int dest_cpu;
    char comm[16];
    unsigned int tgid;
    unsigned int waker_tid;
    unsigned int waker_tgid;
    unsigned int flags;
    char waker_comm[16];
    unsigned long long wake_ns;
    unsigned int wake_ctx;
    unsigned int pad;
};
_Static_assert(sizeof(struct sched_event) == 96 && __builtin_offsetof(struct sched_event, wake_ns) == 80,
               "the layout of sched_trace.bpf.c's struct sched_event");

/* Replace ':' (field separator), ',' (tag separator) and '=' (key/value)
 * with '_' before embedding a comm in a tag value: a thread can set an
 * arbitrary comm (prctl(PR_SET_NAME)), and an unsanitized ':' would move the
 * record's name/tag boundary. Reads at most 16 bytes (a kernel comm). */
static void sanitize_comm(char *out, const char *in, size_t cap) {
    size_t i = 0;
    for (; i < 16 && in[i] != '\0' && i < cap - 1; i++) {
        char c = in[i];
        out[i] = (c == ':' || c == ',' || c == '=' || c == '\n') ? '_' : c;
    }
    out[i] = '\0';
}

static const char *wake_ctx_name(unsigned int c) {
    return c == WAKE_CTX_TASK ? "task" : c == WAKE_CTX_IRQ ? "irq" : "unknown";
}

/* The wire line of one event into line (cap bytes); its length, 0 for an
 * unknown kind or a line that does not fit. */
static int format_sched_event(const struct sched_event *ev, char *line, size_t cap) {
    char comm[17], waker[17];
    sanitize_comm(comm, ev->comm, sizeof(comm));
    sanitize_comm(waker, ev->waker_comm, sizeof(waker));
    int n = 0;
    switch (ev->kind) {
        case EV_OFFCPU:
            if ((ev->flags & (OFF_SLEEP | OFF_WAKER)) == (OFF_SLEEP | OFF_WAKER))
                n = snprintf(line, cap,
                             "span:sched:0:%u:%llu:%llu:off_cpu:comm=%s,task_pid=%u,sleep=1,waker_pid=%u,"
                             "waker_tid=%u,waker_comm=%s,ctx=%s,wake_ns=%llu\n",
                             ev->pid, ev->ts_ns, ev->dur_ns, comm, ev->tgid, ev->waker_tgid, ev->waker_tid,
                             waker, wake_ctx_name(ev->wake_ctx), ev->wake_ns);
            else
                n = snprintf(line, cap, "span:sched:0:%u:%llu:%llu:off_cpu:comm=%s,task_pid=%u,sleep=%u\n",
                             ev->pid, ev->ts_ns, ev->dur_ns, comm, ev->tgid, (ev->flags & OFF_SLEEP) ? 1u : 0u);
            break;
        case EV_WAKEUP:
            n = snprintf(line, cap,
                         "inst:sched:0:%u:%llu:wakeup:comm=%s,wake_cpu=%d,wakee_pid=%u,waker_pid=%u,"
                         "waker_tid=%u,waker_comm=%s,ctx=%s\n",
                         ev->pid, ev->ts_ns, comm, ev->dest_cpu, ev->tgid, ev->waker_tgid, ev->waker_tid,
                         waker, wake_ctx_name(ev->wake_ctx));
            break;
        case EV_MIGRATE:
            n = snprintf(line, cap, "inst:sched:0:%u:%llu:migrate:comm=%s,orig_cpu=%d,dest_cpu=%d\n",
                         ev->pid, ev->ts_ns, comm, ev->orig_cpu, ev->dest_cpu);
            break;
        default:
            return 0;
    }
    return (n > 0 && (size_t)n < cap) ? n : 0;
}

#endif /* SCHED_EVENT_FMT_H */
