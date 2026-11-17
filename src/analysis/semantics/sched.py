"""
OS scheduler wakeups (hooks/os_tracer: an off_cpu span with sleep=1 names
the waking that ended the sleep -- waker_pid=, waker_tid=, ctx=, wake_ns=,
joined in the kernel, sched_trace.bpf.c): which task released a blocking
call that slept.

  waker span -> blocking call "wakeup" (cp_model: the waker span's start
  before the call's end, as for a message's send): blocking call B on
  thread T returned after T slept inside it, and the kernel ran the waking
  that ended T's last sleep in B from traced thread W while W was inside
  span X (the innermost one covering the waking).

Tier "medium". The kernel recorded the waking, but that it is what released
B is inferred: a thread is also woken by signals, timeouts and other waits
inside the same call, and a call can return for a reason the scheduler never
saw -- the wakeup is not proof of causality.

An edge needs every one of the following; anything else gives none and is
counted (dependency_diagnostics, sched_*):
  * B is a blocking call (semantics.blocking_wait) and the tracer saw its
    thread (sched_threads_unseen: none of the thread's scheduler events
    were recorded -- tracer not running then, or another pid namespace);
  * T slept inside B: an off_cpu span with sleep=1 within B; the last one
    is taken. No sleep (sched_no_sleep): a spin wait, or a call that never
    left the CPU or was only preempted -- the scheduler says nothing about
    what released it. A trace from the tracer before sleep= cannot tell a
    sleep from a preemption (sched_sleep_unflagged);
  * the sleep is B's process's (task_pid=; sched_pid_mismatch: the tracer
    and the hooks saw different pid namespaces);
  * it names its waking (sched_waker_unrecorded: the record was evicted, or
    the tracer predates it) from task context (sched_waker_not_task:
    ctx=irq -- the "waker" is whatever the interrupt hit -- or unknown);
  * the waker is a traced thread with a span covering the waking
    (sched_waker_untraced: a runtime helper thread, the driver, another
    program).
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict

from . import blocking_wait

KIND, TIER = "wakeup", "medium"

_REASONS = ("sched_threads_unseen", "sched_no_sleep", "sched_sleep_unflagged", "sched_pid_mismatch",
            "sched_waker_unrecorded", "sched_waker_not_task", "sched_waker_untraced")


def _node(s) -> bool:
    """A host span the event graph keeps as a node (cp_model._excluded)."""
    from ..activity_buckets import is_annotation, is_sampled
    from ...core import gpu_activity as ga
    return not (s.category.value == "sched" or s.tags.get("side") == "gpu" or ga.is_device_span(s)
                or is_annotation(s) or is_sampled(s))


def _int(v) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def has_sched(trace) -> bool:
    return next(iter(trace.iter_spans(categories=("sched",))), None) is not None


def _sleeps(trace, tids: set[int]) -> dict[int, tuple[list[int], list]]:
    """off_cpu spans of `tids` by tid, sorted by start: (their starts,
    (start, end, tags) each)."""
    out: dict[int, list] = defaultdict(list)
    for s in trace.iter_spans(categories=("sched",)):
        if s.name == "off_cpu" and s.tid in tids:
            out[s.tid].append((s.start_ns, s.end_ns, s.tags))
    res = {}
    for tid, v in out.items():
        v.sort(key=lambda x: (x[0], x[1]))
        res[tid] = ([x[0] for x in v], v)
    return res


def _within(sl, a: int, b: int) -> list:
    """The off_cpu spans of one thread (_sleeps) inside [a, b]."""
    starts, v = sl
    out = []
    for k in range(bisect_left(starts, a), bisect_right(starts, b)):
        if v[k][1] <= b:
            out.append(v[k])
    return out


def _innermost(spans, ts: list[int]) -> list[int | None]:
    """For each time in `ts`: the ordinal of the innermost node span of
    `spans` ((ordinal, span), one thread) covering it, None if none. One
    pass; each span visits only the times it covers."""
    order = sorted(range(len(ts)), key=ts.__getitem__)
    sts = [ts[k] for k in order]
    best: list[tuple | None] = [None] * len(ts)
    for i, s in spans:
        if not _node(s):
            continue
        a, b = s.start_ns, s.end_ns
        key = (a, -b, i)                       # latest start, then shortest, then latest ordinal
        for j in range(bisect_left(sts, a), bisect_right(sts, b)):
            k = order[j]
            if best[k] is None or key > best[k]:
                best[k] = key
    return [None if b is None else b[2] for b in best]


def wakeup_edges(trace, blocking, thread_spans) -> tuple[list[tuple[int, int]], dict]:
    """(waker ordinal, blocking-call ordinal) pairs in blocking-call order,
    and the counts of calls without one (module docstring).

    blocking: (ordinal, span) of candidate spans in ordinal order (the
    blocking host calls are picked here); thread_spans(pid, tid): a
    thread's (ordinal, span) pairs."""
    if not has_sched(trace):
        return [], {}
    calls = [(v, s) for v, s in blocking if s.tags.get("side") != "gpu" and blocking_wait(s)]
    if not calls:
        return [], {}
    sleeps = _sleeps(trace, {s.tid for _v, s in calls})
    diag = dict.fromkeys(_REASONS, 0)
    want: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for v, b in calls:
        sl = sleeps.get(b.tid)
        if not sl:
            diag["sched_threads_unseen"] += 1
            continue
        inside = _within(sl, b.start_ns, b.end_ns)
        if not inside:
            diag["sched_no_sleep"] += 1
            continue
        mine = [x for x in inside if _int(x[2].get("task_pid")) in (None, b.pid)]
        if not mine:
            diag["sched_pid_mismatch"] += 1
            continue
        slept = [x for x in mine if x[2].get("sleep") == "1"]
        if not slept:
            diag["sched_sleep_unflagged" if any("sleep" not in x[2] for x in mine) else "sched_no_sleep"] += 1
            continue
        tags = max(slept, key=lambda x: x[1])[2]
        wpid, wtid, wns = _int(tags.get("waker_pid")), _int(tags.get("waker_tid")), _int(tags.get("wake_ns"))
        if wtid is None or wpid is None or wns is None:
            diag["sched_waker_unrecorded"] += 1
            continue
        if tags.get("ctx") != "task":
            diag["sched_waker_not_task"] += 1
            continue
        want[(wpid, wtid)].append((wns, v))
    found: dict[int, int] = {}
    for (wpid, wtid), reqs in want.items():
        for (_t, v), u in zip(reqs, _innermost(thread_spans(wpid, wtid), [t for t, _v in reqs])):
            if u is None or u == v:
                diag["sched_waker_untraced"] += 1
            else:
                found[v] = u
    edges = [(found[v], v) for v, _s in calls if v in found]
    diag = {k: n for k, n in diag.items() if n}
    diag["sched_blocking_calls"] = len(calls)
    diag["sched_wakeup_edges"] = len(edges)
    return edges, diag


def add_edges(preds, trace, blocking, thread_spans) -> None:
    """wakeup_edges() into `preds` (criticalpath's dict-of-lists or _EdgeSink)."""
    for u, v in wakeup_edges(trace, blocking, thread_spans)[0]:
        preds[v].append((u, KIND, TIER))


def stored_inputs(trace, ordinal_of):
    """wakeup_edges()'s blocking / thread_spans arguments streamed from the
    trace's store (criticalpath.build_edges; ordinal_of: seq -> ordinal)."""
    blocking = ((ordinal_of(s.seq), s) for s in trace.iter_spans(categories=("mpi", "sync")))

    def thread_spans(pid, tid):
        return ((ordinal_of(s.seq), s) for s in trace.iter_spans(pid=pid, tid=tid))
    return blocking, thread_spans


def list_inputs(spans):
    """wakeup_edges()'s arguments over a span list (ordinal = index)."""
    by_thread: dict[tuple[int, int], list] | None = None

    def thread_spans(pid, tid):
        nonlocal by_thread
        if by_thread is None:
            by_thread = defaultdict(list)
            for i, s in enumerate(spans):
                by_thread[(s.pid, s.tid)].append(i)
        return ((i, spans[i]) for i in by_thread.get((pid, tid), ()))
    return enumerate(spans), thread_spans


def dependency_diagnostics(trace) -> dict:
    if not has_sched(trace):
        return {}
    seqs = trace.store.span_seqs()
    import numpy as np
    return wakeup_edges(trace, *stored_inputs(trace, lambda q: int(np.searchsorted(seqs, q))))[1]


def dependency_warnings(diag: dict) -> list[str]:
    if not diag.get("sched_blocking_calls"):
        return []
    out = []
    n, edges = diag["sched_blocking_calls"], diag.get("sched_wakeup_edges", 0)
    parts = [f"{diag[k]} {label}" for k, label in (
        ("sched_no_sleep", "never slept (spin waits, or only preempted)"),
        ("sched_threads_unseen", "on threads the scheduler tracer did not report"),
        ("sched_sleep_unflagged", "from a tracer without sleep= (sleep and preemption look alike)"),
        ("sched_pid_mismatch", "whose sleeps name another process (pid namespaces differ?)"),
        ("sched_waker_unrecorded", "whose waking was not recorded"),
        ("sched_waker_not_task", "woken from interrupt or unknown context"),
        ("sched_waker_untraced", "woken by a thread without a traced span then"),
    ) if diag.get(k)]
    if parts:
        out.append(f"{edges} of {n} blocking call(s) got an OS wakeup edge ('medium': the waking is "
                   f"recorded, that it released the call is inferred); no edge for " + ", ".join(parts))
    return out
