"""
Attribution of critical-path time: three questions, answered separately on
the event-point graph (cp_model.py).

  witness       what the observed path passes through: the time each span
                owns on it (cp_model.walk / path_rows) -- where the path
                runs, not why.
  blame         who made others wait. Each gated call's waiting time -- from
                its start to its release, the latest source event of the
                dependencies gating it -- is charged to the work that delayed
                that release: the busy time on the releasing source's lane
                inside the waiting window (CASITA-style blame). Joint
                release: sources within joint_ns of the latest (or whose
                order the clocks cannot tell, cp_clock.py) share the waiting
                time equally -- the call waited for all of them. Time the
                source itself spent waiting inside the window passes the
                blame on (to depth `depth`); unrecorded, unknown and device
                idle time in the window is charged to "(unrecorded)" /
                "(unknown)" / "(device idle)". Every waiting nanosecond is
                charged once.
  intervention  what would change if some work shrank: the trace replayed
                (replay()) with those spans' busy time scaled. Every event
                happens when its latest predecessor allows, plus its
                residual -- the time it had after that predecessor in the
                recording (post-release processing, launch latency,
                queueing...). Busy, unrecorded and unknown time keep their
                durations; waiting, device idle time and dependency latency
                shrink to what the predecessors allow. Lanes stay serial
                (their edges chain), streams FIFO. Unchanged inputs reproduce
                the recording exactly. A replay under fixed residuals, not a
                prediction: contention, adaptive algorithms and anything else
                the trace did not record are not modeled.
"""
from __future__ import annotations

import array
from collections import defaultdict
from dataclasses import dataclass, field

from . import cp_model as cm

UNRECORDED, UNKNOWN, IDLE = "(unrecorded)", "(unknown)", "(device idle)"
_LANE_KINDS = frozenset({cm._K["thread"], cm._K["device_work"], cm._K["stream_order"]})


# ── intervention ─────────────────────────────────────────────────────────────

def _delta(g: cm.EventGraph, e: int, scale: dict[int, float] | None) -> float:
    m = g.mode[e]
    if m == cm.BUSY:
        f = scale.get(g.owner[e], 1.0) if scale else 1.0
        return g.d[e] * f
    if m in (cm.GAP, cm.UNKNOWN):
        return g.d[e]
    return 0.0                                  # waiting, device idle, dependency latency


def residuals(g: cm.EventGraph) -> array.array:
    """Per node: its recorded time minus what its predecessors alone
    require (>= 0; nodes without predecessors keep their time: -1)."""
    t, src, enabled = g.t, g.src, g.enabled
    out = array.array("q", [-1]) * (2 * g.n)
    for v in g.order:
        best = None
        for k in range(g.in_ptr[v], g.in_ptr[v + 1]):
            e = g.in_idx[k]
            if enabled[e]:
                x = t[src[e]] + _delta(g, e, None)
                if best is None or x > best:
                    best = x
        if best is not None:
            out[v] = max(0, int(round(t[v] - best)))
    return out


def replay(g: cm.EventGraph, scale: dict[int, float] | None = None,
           res: array.array | None = None) -> tuple[list[float], float]:
    """Event times with the busy time of spans in `scale` multiplied by its
    factor, and the end of the run (the latest event)."""
    if res is None:
        res = residuals(g)
    t, src, enabled = g.t, g.src, g.enabled
    tp = [0.0] * (2 * g.n)
    end = 0.0
    for v in g.order:
        if res[v] < 0:
            x = float(t[v])
        else:
            x = None
            for k in range(g.in_ptr[v], g.in_ptr[v + 1]):
                e = g.in_idx[k]
                if enabled[e]:
                    y = tp[src[e]] + _delta(g, e, scale)
                    if x is None or y > x:
                        x = y
            x = (x or 0.0) + res[v]
        tp[v] = x
        if x > end:
            end = x
    return tp, end


def run_end(g: cm.EventGraph) -> int:
    return max((g.t[v] for v in g.order), default=0)


# ── blame ────────────────────────────────────────────────────────────────────

@dataclass
class Blame:
    by_span: dict[int, float] = field(default_factory=dict)        # span -> waiting time charged to it
    other: dict[str, float] = field(default_factory=dict)          # (unrecorded) / (unknown) / (device idle)
    waiting_ns: float = 0.0                                         # every gated call's waiting time
    joint_releases: int = 0                                         # waits shared between several sources
    truncated: int = 0                                              # charges cut short (steps / depth)
    # every charged wait: (waiting span, its start, its release, waiting ns,
    # the spans whose source events released it -- several: a joint release)
    waits: list[tuple[int, int, int, float, tuple[int, ...]]] = field(default_factory=list)
    passed_on: dict[int, float] = field(default_factory=dict)       # span -> waiting it passed to its releasers

    def by_category(self, g: cm.EventGraph) -> dict[str, float]:
        out: dict[str, float] = defaultdict(float)
        for j, ns in self.by_span.items():
            out[g.cat[j]] += ns
        for k, ns in self.other.items():
            out[k] += ns
        return dict(out)

    def top(self, g: cm.EventGraph, k: int = 10) -> list[dict]:
        rows = sorted(self.by_span.items(), key=lambda kv: -kv[1])[:k]
        return [{"span": j, "category": g.cat[j], "name": g.name(j), "start_ns": g.t[2 * j],
                 "blame_ns": int(round(ns))} for j, ns in rows]


def _lane_pred(g: cm.EventGraph, v: int) -> int:
    """The enabled lane edge into v (a thread's, a device operation's own
    edge, or its stream's order), -1 if none."""
    for k in range(g.in_ptr[v], g.in_ptr[v + 1]):
        e = g.in_idx[k]
        if g.enabled[e] and g.kind[e] in _LANE_KINDS:
            return e
    return -1


def _sources(g: cm.EventGraph, j: int) -> list[int]:
    v = 2 * j + 1
    return [g.in_idx[k] for k in range(g.in_ptr[v], g.in_ptr[v + 1])
            if g.enabled[g.in_idx[k]] and g.mode[g.in_idx[k]] == cm.DEP]


def _joint(g: cm.EventGraph, srcs: list[int], joint_ns: int) -> list[int]:
    """The sources that released jointly: within joint_ns of the latest,
    or within the clocks' uncertainty of it."""
    t = g.t
    latest = max(srcs, key=lambda e: t[g.src[e]])
    r = t[g.src[latest]]
    ca = g.clock
    out = []
    for e in srcs:
        gap = r - t[g.src[e]]
        if gap <= joint_ns:
            out.append(e)
        elif ca is not None:
            a, b = g.domain[g.src[e] >> 1], g.domain[g.src[latest] >> 1]
            if a != b and ca.offset_range(a, b)[1] >= gap:
                out.append(e)
    return out


def blame(g: cm.EventGraph, *, joint_ns: int = 1000, depth: int = 4, max_steps: int = 4096) -> Blame:
    """Waiting blame (module docstring) over every gated call of g."""
    out = Blame()
    by_span: dict[int, float] = defaultdict(float)
    other: dict[str, float] = defaultdict(float)
    passed: dict[int, float] = defaultdict(float)

    def charge(node: int, a: int, b: int, share: float, level: int) -> None:
        """Charge `share` ns for the window [a, b] that ended at event
        `node` to what the node's lane did in that window."""
        if share <= 0:
            return
        if b <= a:                                   # released no later than the window opened
            by_span[node >> 1] += share
            return
        pieces: list[tuple[int, int]] = []          # (edge, overlap)
        v, steps = node, 0
        while steps < max_steps:
            e = _lane_pred(g, v)
            if e < 0:
                break
            u = g.src[e]
            lo, hi = max(a, g.t[u]), min(b, g.t[v])
            if hi > lo:
                pieces.append((e, hi - lo))
            if g.t[u] <= a:
                break
            v = u
            steps += 1
        else:
            out.truncated += 1
        covered = sum(o for _e, o in pieces)
        if covered <= 0:
            other[UNRECORDED] += share
            return
        for e, o in pieces:
            part = share * o / covered
            m = g.mode[e]
            if m == cm.BUSY:
                by_span[g.owner[e]] += part
            elif m == cm.WAIT:
                # inside a gated call of the source: until that call's own
                # release it was waiting too -- passed on to what released
                # it; after, it was processing -- its own
                j = g.owner[e]
                lo, hi = max(a, g.t[g.src[e]]), min(b, g.t[g.dst[e]])
                r = g.release[j]
                wait_part = part * max(0, min(hi, r) - lo) / (hi - lo)
                by_span[j] += part - wait_part
                srcs = _sources(g, j) if wait_part > 0 else []
                if wait_part <= 0:
                    continue
                if level >= depth or not srcs:
                    if level >= depth:
                        out.truncated += 1
                        by_span[j] += wait_part
                    else:
                        other[UNKNOWN] += wait_part
                    continue
                joint = _joint(g, srcs, joint_ns)
                passed[j] += wait_part
                for s_e in joint:
                    charge(g.src[s_e], lo, min(hi, max(lo, g.t[g.src[s_e]])), wait_part / len(joint), level + 1)
            elif m == cm.IDLE:
                other[IDLE] += part
            elif m == cm.UNKNOWN:
                other[UNKNOWN] += part
            else:
                other[UNRECORDED] += part

    for j in range(g.n):
        if not g.gated[j] or g.excluded[j]:
            continue
        srcs = _sources(g, j)
        if not srcs:
            continue
        S, E = g.t[2 * j], g.t[2 * j + 1]
        r = max(g.t[g.src[e]] for e in srcs)
        w = min(r, E) - S
        if w <= 0:
            continue
        out.waiting_ns += w
        joint = _joint(g, srcs, joint_ns)
        if len(joint) > 1:
            out.joint_releases += 1
        out.waits.append((j, S, min(r, E), w, tuple(g.src[e] >> 1 for e in joint)))
        for e in joint:
            charge(g.src[e], S, min(E, max(S, g.t[g.src[e]])), w / len(joint), 0)
    out.by_span = dict(by_span)
    out.other = dict(other)
    out.passed_on = dict(passed)
    return out
