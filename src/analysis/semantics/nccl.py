"""
NCCL semantics (nccl_hook.c's pseq= / group= / timing= tags):

  * point-to-point: NCCL pairs the sends to and the receives from one peer
    on one communicator in issue order -- the k-th send from a to b
    (pseq=k on a) is the k-th receive on b from a (pseq=k on b): the
    receive completes after the send started;
  * collectives: the k-th collective of a communicator on every rank
    (commid= + seq=) is one rendezvous (criticalpath's grouping);
  * the work of an operation: itself when its span is the event-pair
    measurement (the default); its device span (psid= -> its sid=) with
    asynchronous timing (timing=async); for an operation issued inside
    ncclGroupStart/End (timing=deferred, group=) the group's work on its
    stream (type=group_exec, timed at ncclGroupEnd). Edges join those
    spans, never the host calls of grouped or asynchronous operations.
"""
from __future__ import annotations


def effective(spans, items) -> dict[int, int]:
    """ordinal -> the ordinal of the span timing that operation's work, for
    the operations whose work is another span."""
    dev_of: dict[tuple, int] = {}
    group_exec: dict[tuple, int] = {}
    nccl = [(i, s) for i, s in items(spans) if s.category.value == "nccl"]
    for i, s in nccl:
        t = s.tags
        if t.get("side") == "gpu" and s.parent_span_id:
            dev_of[(s.pid, s.parent_span_id)] = i
    for i, s in nccl:
        t = s.tags
        if t.get("type") == "group_exec":
            j = dev_of.get((s.pid, s.span_id), i) if t.get("side") == "cpu" and s.span_id else i
            key = (s.pid, t.get("group"), t.get("stream"))
            if key not in group_exec or spans[j].tags.get("side") == "gpu":
                group_exec[key] = j
    out: dict[int, int] = {}
    for i, s in nccl:
        t = s.tags
        if t.get("type") == "group_exec":
            continue
        if t.get("timing") == "deferred":
            j = group_exec.get((s.pid, t.get("group"), t.get("stream")))
        elif t.get("side") == "cpu" and s.span_id:
            j = dev_of.get((s.pid, s.span_id))
        else:
            continue
        if j is not None:
            out[i] = j
    return out


def operations(spans, items):
    """(ordinal, span) of the NCCL operations as issued: never the device
    copies asynchronous timing adds (side=gpu with the same tags) or the
    group spans."""
    for i, s in items(spans):
        if s.category.value == "nccl" and s.tags.get("side") != "gpu" \
                and s.tags.get("type") not in ("group", "group_exec"):
            yield i, s


def add_p2p_edges(preds, spans, items, eff: dict[int, int], lossy=None) -> None:
    from ..criticalpath import _lossy
    sends: dict[tuple, int] = {}
    recvs: list[tuple[int, tuple]] = []
    for i, s in operations(spans, items):
        t = s.tags
        typ = t.get("type")
        if typ not in ("send", "recv") or not t.get("pseq") or not t.get("commid"):
            continue
        if typ == "send":
            sends[(t["commid"], t.get("rank"), t.get("peer"), t["pseq"])] = i
        else:
            recvs.append((i, (t["commid"], t.get("peer"), t.get("rank"), t["pseq"])))
    for r, key in recvs:
        snd = sends.get(key)
        if snd is None:
            continue
        a, b = eff.get(snd, snd), eff.get(r, r)
        conf = "low" if _lossy(lossy, spans[snd].pid, spans[r].pid) else "certain"
        if a != b:
            preds[b].append((a, "p2p", conf))


def rendezvous_items(spans, items, eff: dict[int, int]) -> list[tuple]:
    """criticalpath._rendezvous_groups items of the collectives, timed by
    the span doing their work."""
    from ..criticalpath import _NCCL_COLLECTIVE_TYPES
    out = []
    for i, s in operations(spans, items):
        t = s.tags
        typ = t.get("type", "")
        if typ not in _NCCL_COLLECTIVE_TYPES:
            continue
        j = eff.get(i, i)
        w = spans[j]
        if w.duration_ns <= 0:
            continue
        commid = t.get("commid")
        out.append((w.start_ns, w.start_ns + w.duration_ns, j, ("pid", s.pid), s.tid, typ,
                    None if commid is None else str(commid), t.get("seq"), None))
    return out
