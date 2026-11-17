"""
OpenCL command-queue semantics (the hook's queue= / ooo= / ev= / wait= /
blocking= tags, opencl_hook.c):

  * an in-order queue runs its commands one after another, in enqueue
    order: each device command starts after the previous one ends; a
    blocking transfer (no device span: the host call is the transfer)
    waits for the queue's earlier commands;
  * an out-of-order queue orders commands only by wait lists and
    barriers: clEnqueueBarrierWithWaitList holds back every later command
    until every earlier one is done (edge from the latest-ending earlier
    one); a marker completes once its prerequisites have (its wait list,
    else every earlier command of the queue);
  * wait lists: a command starts after each event it waits for completed;
    an event the hook could not name (wait_unknown=, or beyond the
    recorded ids: wait_omitted=) gives no edge -- the command's wait is
    then incomplete, never guessed;
  * clEnqueueWaitForEvents (OpenCL 1.1, type=wait_events): the queue's
    later commands wait for its events;
  * user events (clCreateUserEvent, ev=): a command or clWaitForEvents
    waiting for one waits for the clSetUserEventStatus call that completed
    it (an explicit_span_id edge, host call -> waiter);
  * clFinish(queue): every command enqueued to it before the call is done
    when it returns; clWaitForEvents: each listed event is.

Device spans are the event-completion callbacks' spans (side=gpu), linked
to their enqueue by psid= -> sid= (the launch itself is an
explicit_span_id edge). Edges: device -> device "sequential" (cp_model:
stream order on one queue, an event wait across queues), device -> host
call "device_wait". Traces from hooks before queue= get none of these.
"""
from __future__ import annotations

from collections import defaultdict


class _Cmd:
    __slots__ = ("host", "dev", "ev", "wait", "kind", "queue", "start", "blocking")

    def __init__(self, host, dev, ev, wait, kind, queue, start, blocking) -> None:
        self.host, self.dev, self.ev, self.wait = host, dev, ev, wait
        self.kind, self.queue, self.start, self.blocking = kind, queue, start, blocking


def _is_device(s) -> bool:
    return s.tags.get("side") == "gpu" or s.name.endswith("_gpu")


def _ids(value: str | None) -> list[str]:
    return [v for v in (value or "").split(";") if v]


def add_edges(preds, spans, items) -> None:
    """OpenCL queue edges into `preds` (criticalpath's dict-of-lists or
    _EdgeSink); spans: {ordinal: span} or a list, iterated with `items`."""
    by_pid: dict[int, dict] = defaultdict(lambda: {"host": {}, "dev": {}, "finish": [], "waits": [],
                                                   "user_set": {}})
    for i, s in items(spans):
        cat = s.category.value
        if cat not in ("opencl", "memory", "sync"):
            continue
        t = s.tags
        p = by_pid[s.pid]
        if _is_device(s):
            if s.parent_span_id and "queue" in t:
                p["dev"][s.parent_span_id] = i
        elif cat == "sync" and s.name == "clFinish" and "queue" in t:
            p["finish"].append(i)
        elif cat == "sync" and s.name == "clWaitForEvents" and "nwait" in t:
            p["waits"].append(i)
        elif t.get("type") == "user_event_status" and t.get("ev") and t.get("status") == "0":
            p["user_set"][t["ev"]] = i               # CL_COMPLETE: what its waiters wait for
        elif "queue" in t:
            p["host"][i] = s
    for pid in sorted(by_pid):
        _add_process(preds, spans, by_pid[pid])


def _add_process(preds, spans, p) -> None:
    queues: dict[str, list[_Cmd]] = defaultdict(list)
    ooo: set[str] = set()
    by_ev: dict[str, _Cmd] = {}
    for i, s in sorted(p["host"].items()):
        t = s.tags
        q = t["queue"]
        if t.get("ooo") == "1":
            ooo.add(q)
        dev = p["dev"].get(s.span_id) if s.span_id else None
        typ = t.get("type", "")
        kind = typ if typ in ("marker", "barrier", "wait_events") else "command"
        c = _Cmd(i, dev, t.get("ev"), _ids(t.get("wait")), kind, q, s.start_ns, t.get("blocking") == "1")
        queues[q].append(c)
        if c.ev:
            by_ev[c.ev] = c
    for cmds in queues.values():
        cmds.sort(key=lambda c: c.start)

    def end(j: int) -> int:
        return spans[j].start_ns + max(0, spans[j].duration_ns)

    done_by: dict[str, int | None] = {}       # a marker's / barrier's event -> what it stands for

    user_set = p["user_set"]

    def resolve(ev: str) -> int | None:
        """The device span whose completion completes event `ev` (a user
        event: the host call that completed it)."""
        src = by_ev.get(ev)
        if src is None:
            return user_set.get(ev)
        return src.dev if src.dev is not None else done_by.get(ev)

    def kind_from(j: int, default: str) -> str:
        return "explicit_span_id" if j in user_hosts else default
    user_hosts = set(user_set.values())

    def walk(q: str, cmds: list[_Cmd], emit: bool) -> None:
        in_order = q not in ooo
        last = None          # in-order: the device span the next command follows
        barrier = None       # out-of-order: what the last barrier stood for
        since: list[int] = []   # out-of-order: device spans since the last barrier
        later: list[int] = []      # clEnqueueWaitForEvents: what the queue's later commands wait for
        for c in cmds:
            waits = [j for j in (resolve(ev) for ev in c.wait) if j is not None]
            if c.kind == "wait_events":
                later = later + waits
                continue
            waits += later
            if c.kind != "command":
                if c.wait:
                    pre = waits + ([last] if in_order and last is not None else [])
                else:
                    pre = ([last] if last is not None else []) if in_order else since + \
                        ([barrier] if barrier is not None else [])
                m = max(pre, key=end) if pre else None
                if c.ev:
                    done_by[c.ev] = m
                if in_order:
                    last = m if m is not None else last
                elif c.kind == "barrier":
                    barrier = m if m is not None else barrier
                    since = []
                continue
            if emit:
                sources = waits + ([last] if in_order and last is not None else []) + \
                    ([barrier] if not in_order and barrier is not None else [])
                for j in dict.fromkeys(sources):     # once each, in order
                    if c.dev is not None and j != c.dev:
                        preds[c.dev].append((j, kind_from(j, "sequential"), "certain"))
                    elif c.dev is None and c.blocking:
                        preds[c.host].append((j, kind_from(j, "device_wait"), "certain"))
            if c.dev is not None:
                if in_order:
                    last = c.dev
                since.append(c.dev)

    for _pass in range(2):                    # a wait list may name another queue's marker
        for q, cmds in queues.items():
            walk(q, cmds, emit=False)
    for q in sorted(queues):
        walk(q, queues[q], emit=True)

    def latest(cmds) -> int | None:
        devs = [c.dev for c in cmds if c.dev is not None]
        return max(devs, key=end) if devs else None

    # clFinish: the latest-ending command enqueued to its queue before it
    for f in p["finish"]:
        s = spans[f]
        j = latest([c for c in queues.get(s.tags["queue"], ()) if c.start <= s.start_ns])
        if j is not None:
            preds[f].append((j, "device_wait", "certain"))
    # clWaitForEvents: each event it names
    for w in p["waits"]:
        for j in dict.fromkeys(resolve(ev) for ev in _ids(spans[w].tags.get("wait"))):
            if j is not None and j != w:
                preds[w].append((j, kind_from(j, "device_wait"), "certain"))


def dependency_diagnostics(trace) -> dict:
    """OpenCL dependence-capture holes (flat keys, ocl_*):

      ocl_wait_unknown      waited-for events the hook could not name (made by
                            a call it does not record, already released, or
                            the event table full): no edge for them
      ocl_wait_omitted      wait-list entries beyond the per-record limit
      ocl_event_table_full  events that got no id (their waiters: unknown)
      ocl_clock_unbounded   devices whose clock mapping has no bound
                            (callback_end: clGetDeviceAndHostTimer missing)
    Empty when there is nothing to report."""
    out: dict = {}
    for s in trace.iter_spans(categories=("opencl", "memory", "sync")):
        t = s.tags
        for key, tag in (("ocl_wait_unknown", "wait_unknown"), ("ocl_wait_omitted", "wait_omitted")):
            if t.get(tag):
                out[key] = out.get(key, 0) + int(t[tag])
    unbounded = set()
    for i in trace.iter_instants():
        if i.name == "ocl_clock" and i.tags.get("method") == "callback_end":
            unbounded.add((i.pid, i.tags.get("dev")))
    if unbounded:
        out["ocl_clock_unbounded"] = len(unbounded)
    for c in trace.iter_counters():
        if c.name == "ocl_event_table_full" and c.value:
            out["ocl_event_table_full"] = out.get("ocl_event_table_full", 0) + int(c.value)
    return out


def dependency_warnings(diag: dict) -> list[str]:
    out = []
    if diag.get("ocl_wait_unknown") or diag.get("ocl_wait_omitted"):
        out.append(f"{diag.get('ocl_wait_unknown', 0)} OpenCL wait-list event(s) could not be named (made by a call "
                   "the hook does not record, released, or the event table full)"
                   + (f" and {diag['ocl_wait_omitted']} were beyond the per-record limit" if diag.get(
                       "ocl_wait_omitted") else "")
                   + ": those waits have no edge")
    if diag.get("ocl_event_table_full"):
        out.append(f"{diag['ocl_event_table_full']} OpenCL event(s) found the hook's event table full: they got no "
                   "id, and what waited for them is unknown")
    if diag.get("ocl_clock_unbounded"):
        out.append(f"{diag['ocl_clock_unbounded']} OpenCL device(s) have no clGetDeviceAndHostTimer: their clock is "
                   "mapped at the first completion callback and may be late by its delivery latency (unbounded)")
    return out
