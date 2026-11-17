/*
 * hooks/os_tracer/sched_event_fmt.h: the wire lines os_tracer forwards for
 * each BPF event, without a kernel (the eBPF program itself cannot be loaded
 * on this machine). Prints one line per event below; tests/test_sched_wakeup.py
 * parses them with the collector's own parser and checks every field.
 */
#include <string.h>

#include "../../hooks/os_tracer/sched_event_fmt.h"

static void emit(const struct sched_event *ev) {
    char line[384];
    int n = format_sched_event(ev, line, sizeof(line));
    if (n > 0) fputs(line, stdout);
    else printf("none kind=%u\n", ev->kind);
}

int main(void) {
    struct sched_event ev;

    memset(&ev, 0, sizeof(ev));               /* slept 2 ms, woken by a task */
    ev.kind = EV_OFFCPU; ev.pid = 4242; ev.tgid = 4200; ev.ts_ns = 1000; ev.dur_ns = 2000000;
    ev.flags = OFF_SLEEP | OFF_WAKER; ev.wake_ns = 1990000; ev.wake_ctx = WAKE_CTX_TASK;
    ev.waker_tid = 4243; ev.waker_tgid = 4200;
    strcpy(ev.comm, "worker"); strcpy(ev.waker_comm, "a:b,c=d");
    emit(&ev);

    ev.flags = OFF_SLEEP; ev.ts_ns = 3000000;  /* slept, its waking evicted */
    emit(&ev);

    ev.flags = 0; ev.ts_ns = 5000000;          /* preempted (waker fields ignored) */
    emit(&ev);

    ev.flags = OFF_WAKER; ev.ts_ns = 5500000;  /* not a combination the BPF side emits */
    emit(&ev);

    memset(&ev, 0, sizeof(ev));               /* wakings from each context */
    ev.kind = EV_WAKEUP; ev.pid = 4242; ev.tgid = 4200; ev.ts_ns = 6000000; ev.dest_cpu = 3;
    ev.waker_tid = 4243; ev.waker_tgid = 4200; ev.wake_ctx = WAKE_CTX_TASK;
    strcpy(ev.comm, "worker"); strcpy(ev.waker_comm, "main");
    emit(&ev);
    ev.wake_ctx = WAKE_CTX_IRQ; ev.ts_ns = 6100000;
    emit(&ev);
    ev.wake_ctx = WAKE_CTX_UNKNOWN; ev.ts_ns = 6200000;
    emit(&ev);
    ev.wake_ctx = 7; ev.ts_ns = 6300000;      /* not a context the BPF side emits */
    emit(&ev);

    memset(&ev, 0, sizeof(ev));               /* a full 16-byte comm, no NUL */
    ev.kind = EV_MIGRATE; ev.pid = 4242; ev.ts_ns = 7000000; ev.orig_cpu = 1; ev.dest_cpu = 2;
    memcpy(ev.comm, "0123456789abcdef", 16);
    emit(&ev);

    ev.kind = 99;
    emit(&ev);

    char tiny[16];                            /* a line that does not fit: nothing */
    ev.kind = EV_MIGRATE;
    printf("tiny=%d\n", format_sched_event(&ev, tiny, sizeof(tiny)));
    return 0;
}
