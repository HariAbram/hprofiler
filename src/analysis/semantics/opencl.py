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
    by_pid: dict[int, dict] = defaultdict(lambda: {"host": {}, "dev": {}, "finish": [], "waits": []})
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
        kind = typ if typ in ("marker", "barrier") else "command"
        c = _Cmd(i, dev, t.get("ev"), _ids(t.get("wait")), kind, q, s.start_ns, t.get("blocking") == "1")
        queues[q].append(c)
        if c.ev:
            by_ev[c.ev] = c
    for cmds in queues.values():
        cmds.sort(key=lambda c: c.start)

    def end(j: int) -> int:
        return spans[j].start_ns + max(0, spans[j].duration_ns)

    done_by: dict[str, int | None] = {}       # a marker's / barrier's event -> what it stands for

    def resolve(ev: str) -> int | None:
        """The device span whose completion completes event `ev`."""
        src = by_ev.get(ev)
        if src is None:
            return None
        return src.dev if src.dev is not None else done_by.get(ev)

    def walk(q: str, cmds: list[_Cmd], emit: bool) -> None:
        in_order = q not in ooo
        last = None          # in-order: the device span the next command follows
        barrier = None       # out-of-order: what the last barrier stood for
        since: list[int] = []   # out-of-order: device spans since the last barrier
        for c in cmds:
            waits = [j for j in (resolve(ev) for ev in c.wait) if j is not None]
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
                        preds[c.dev].append((j, "sequential", "certain"))
                    elif c.dev is None and c.blocking:
                        preds[c.host].append((j, "device_wait", "certain"))
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
            if j is not None:
                preds[w].append((j, "device_wait", "certain"))
