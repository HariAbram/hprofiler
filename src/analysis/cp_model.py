"""
Event-point critical-path model.

The span-level graph of criticalpath.py (one node per call) cannot keep
time consistent: a blocking call's span contains its waiting, which overlaps
the work of whatever it waited for, so a path through both counts that
interval twice, and a path picked by such a score can be the wrong one.
This module splits every span into its start and end events and keeps time
on the edges, so a path never covers an instant twice.

Graph
-----
Nodes. Every span s has a start node S(s) and an end node E(s) at its start
and end timestamps (device work: its measured execution interval,
criticalpath._effective_start_ns) -- except annotation ranges (NVTX/ROCTX:
labels, not work), perf samples (an estimated weight, not an interval) and
scheduler intervals (thread states, not calls).

Each edge u -> v has an observed duration d = t(v) - t(u) >= 0 and a
required part w, 0 <= w <= d: the time that must pass between the two
events in the model however the dependencies had been timed.

Lane edges -- a thread's or a device operation's own time:
  * Host threads: the start and end events of the thread's spans in time
    order (an enclosing span's start before the start of a span nested in
    it, a nested span's end before the enclosing end), consecutive events
    linked. Each interval is classified by the innermost span open during
    it:
      busy  inside a recorded call                                  w = d
      wait  inside a call whose end a dependency from another lane gates
            (a receive completion, a collective, a barrier, a sync call,
            a parallel region's join): the trace cannot split its time
            into waiting and processing, so none of it is required  w = 0
      gap   no recorded call open: untraced work or idle time, the trace
            cannot tell -- both are computed (see "Bounds")
  * Device work (and other device-timed spans): one edge S -> E, busy, or
    wait when a dependency gates its end (an NCCL collective).

Dependency edges -- between lanes, w = 0: the source event must have
happened; how long the dependency took to take effect is not required time.
They come from the persisted span-level edges (criticalpath.build_edges):
    p2p               S(send)          -> E(receive completion)
    arrival           S(last arriver)  -> E(every other member)
    launch            S(host call)     -> S(device work)
    device_wait       E(device work)   -> E(sync call)
    device_sync       E(gpu span)      -> E(sync call)   pre-split traces
    stream_order      E(device work)   -> S(next work on the same stream):
                      the interval between them is device idle time, w = 0
    event_wait        E(producer work) -> S(consumer work), across streams
    fork / join       OpenMP region -> a task on another thread inside it
                      (psid=): S(region) -> S(task), E(task) -> E(region)
    explicit_span_id  other cross-thread sid=/psid= links: E(parent) ->
                      S(child), or S(parent) -> S(child) / E(child) when
                      the child does not start after the parent ended
    sequential        a pre-split device span and its launching thread
  Same-thread links of the span-level graph (program order, a request and
  its Wait on one thread) are the thread's own event order and are not
  repeated. An edge whose source event lies after its destination (beyond
  the clock tolerance for host/device edges) contradicts the timestamps:
  it is dropped and counted, never bent.

Observed critical path
----------------------
Followed backwards from the last event of the trace: at each node, the
incoming edge with the latest release time t(source) + w is the one that
released it. A busy edge always wins (its release time is the node's own
time). A gap edge releases at its source time (the thread was free from
then on; "unrecorded_as_work" below assumes the opposite), so a dependency
that arrived during untraced time on a thread binds. Ties are counted.
Every step covers [t(source), t(destination)] once (a running frontier
trims the overlap that a clock-tolerance edge can cause), so the path length
is the extent it covers and at most the wall time. Each step is time in a
call (busy), time in a gated call after its last dependency, unrecorded
time on a thread, device idle time between two operations of a stream, or
the latency of a dependency (from its source event to the event it
released).

Modeled longest path
--------------------
The longest path by required time w (longest-path values forward and
backward in topological order): the longest chain of recorded work under
the recorded dependencies with waiting, dependency latency and time inside
gated calls set to zero. A graph model, not a measurement; at most the wall
time. From it:
  * slack of each span: how much one of its busy segments could grow before
    it would lengthen the longest path;
  * spans with work on every / on some / on no longest path -- exact: a
    tight edge (F[u] + w + B[v] = L) lies on some longest path, and on every
    one iff it is the only tight edge crossing some cut of the topological
    order (one O(V + E) sweep; paths start and end anywhere, through a
    virtual source and sink);
  * counterfactual reductions: the longest path recomputed with a span's
    (or every span of one name's) exclusive busy time set to zero. Graph
    estimates under the model's assumptions, not measured speedups: real
    dependencies the trace did not record, contention and the timing of
    the dependencies themselves are not modeled.

Bounds and alternatives
-----------------------
Where the data does not decide, the model is computed both ways instead of
enumerating paths: unrecorded thread time as work and as idle (modeled
length), gap edges releasing at their end instead of their start
(observed path, "unrecorded_as_work"), and with direct-evidence dependency
edges only. Spans on the observed path of every variant vs. only some show
how much the path rests on those choices. A few walks and linear passes:
no exponential search.

Evidence
--------
Every dependency edge carries its kind, runtime, tier (criticalpath.py's
certain/high/medium/low -- evidence classes, not probabilities: nothing
calibrates them) and a provenance from PROVENANCE, with its class: direct
(an id, a correlation id, a runtime ordering rule on one clock) or inferred
(posting order, call order, time overlap, and any edge whose tier is medium
or low: the builder chose among candidates or the process lost events).
edge_reason() spells out why an edge exists and has its tier.
"""
from __future__ import annotations

import array
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..core import gpu_activity as ga
from .activity_buckets import is_annotation, is_device_timed, is_sampled

if TYPE_CHECKING:
    from ..core.trace import Trace

# ── edge vocabulary ───────────────────────────────────────────────────────────

BUSY, WAIT, GAP, IDLE, DEP = range(5)
MODES = ("busy", "wait", "gap", "idle", "dependency")

KINDS = ("thread", "device_work", "stream_order", "event_wait", "launch", "device_wait",
         "device_sync", "p2p", "arrival", "fork", "join", "explicit_span_id", "sequential")
_K = {k: i for i, k in enumerate(KINDS)}

CONFS = ("certain", "high", "medium", "low")          # criticalpath.EDGE_CONFS order
_C = {c: i for i, c in enumerate(CONFS)}

# provenance -> (class, what established the edge)
PROVENANCE: dict[str, tuple[str, str]] = {
    "thread_timeline": ("direct", "consecutive events of one thread, on one clock"),
    "device_interval": ("direct", "the device work's measured execution interval"),
    "stream_fifo": ("direct", "one device stream runs its work in submission order"),
    "event_record": ("direct", "a stream wait waits for the latest record of its event"),
    "sync_target": ("direct", "a sync call returns after the last work submitted to what it synchronizes"),
    "correlation_id": ("direct", "the runtime's correlation id links the host call to its device work"),
    "explicit_id": ("direct", "an id the hook recorded (sid= / psid=)"),
    "fork_join": ("direct", "an OpenMP parallel-region id (psid=): the region begins before and ends "
                            "after each task of its team"),
    "posting_order": ("inferred", "the k-th send paired with the k-th receive of (sender, receiver, "
                                  "tag, communicator)"),
    "status_resolved": ("inferred", "a wildcard receive: source and tag from MPI_Status, then posting "
                                    "order"),
    "call_order": ("inferred", "the k-th collective / barrier call of every member (MPI / OpenMP "
                               "calling-order rule)"),
    "nccl_seq": ("direct", "the NCCL communicator id and sequence number recorded by the hook"),
    "time_overlap": ("inferred", "calls grouped because their time intervals overlap"),
}
PROVS = tuple(PROVENANCE)
_P = {p: i for i, p in enumerate(PROVS)}

_TIER_REASON = {
    ("p2p", "high"): "the receive's source and tag came from MPI_Status",
    ("p2p", "medium"): "matched by posting order alone",
    ("arrival", "high"): "every member made the same calls in the same order",
    ("arrival", "medium"): "members' call sequences differ, or no communicator / team id: grouped by "
                           "overlapping intervals",
    ("device_wait", "medium"): "which stream or device the call synchronized is not recorded "
                               "unambiguously; every candidate gets an edge",
    ("launch", "medium"): "the correlation id matched more than one candidate",
}


def edge_class(prov: str, conf: str) -> str:
    """direct | inferred: the provenance's class, inferred whenever the tier
    is medium or low (the builder chose among candidates, or the process
    lost events)."""
    cls = PROVENANCE[prov][0]
    return "inferred" if cls == "direct" and conf in ("medium", "low") else cls


def edge_reason(kind: str, prov: str, conf: str) -> str:
    """Why the edge exists and has its tier, in words."""
    why = PROVENANCE[prov][1]
    if conf == "low":
        return why + "; the process lost events, so the pairing may be shifted"
    extra = _TIER_REASON.get((kind, conf))
    return f"{why}; {extra}" if extra else why


def runtime_of_edge(kind: str, src_cat: str, dst_cat: str) -> str:
    if kind == "p2p":
        return "mpi"
    if kind in ("fork", "join"):
        return "openmp"
    if kind == "arrival":
        return dst_cat if dst_cat in ("mpi", "nccl") else "openmp"
    if kind == "launch":
        return dst_cat
    if kind in ("device_wait", "device_sync", "stream_order", "event_wait", "sequential", "explicit_span_id"):
        return src_cat
    return dst_cat


# ── the graph ─────────────────────────────────────────────────────────────────

def _i32() -> str:
    return "i" if array.array("i").itemsize == 4 else "l"


class EventGraph:
    """Nodes 2i (start) and 2i+1 (end) of span ordinal i; edges in typed
    arrays (src, dst, mode, kind, prov, conf, owner span, observed d)."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.t = array.array("q", bytes(16 * n))
        self.active = bytearray(2 * n)
        I32 = _i32()                              # node and span ids: < 2^31
        self.src = array.array(I32)
        self.dst = array.array(I32)
        self.mode = array.array("b")
        self.kind = array.array("b")
        self.prov = array.array("b")
        self.conf = array.array("b")
        self.owner = array.array(I32)
        self.d = array.array("q")
        self.enabled = bytearray()
        self.order = array.array(_i32())          # active nodes, topological order
        # per span
        self.cat: list[str] = [""] * n
        self.name_id = array.array(_i32(), bytes(4 * n))
        self.names: list[str] = []
        self.device = bytearray(n)
        self.gated = bytearray(n)
        self.rooted = bytearray(n)
        self.excluded = bytearray(n)
        # diagnostics
        self.dropped: Counter = Counter()        # dependency kind -> edges contradicting timestamps
        self.improper_nesting = 0
        self.cycle_edges_dropped = 0
        self.unmapped: Counter = Counter()

    def add(self, u: int, v: int, mode: int, kind: int, prov: int, conf: int, owner: int,
            tol: int = 0) -> bool:
        d = self.t[v] - self.t[u]
        if d < 0:
            if -d > tol:
                self.dropped[KINDS[kind]] += 1
                return False
            d = 0
        self.src.append(u)
        self.dst.append(v)
        self.mode.append(mode)
        self.kind.append(kind)
        self.prov.append(prov)
        self.conf.append(conf)
        self.owner.append(owner)
        self.d.append(d)
        self.active[u] = self.active[v] = 1
        return True

    def __len__(self) -> int:
        return len(self.src)

    def finish(self) -> None:
        """In/out adjacency (CSR by destination and by source)."""
        N = 2 * self.n
        m = len(self.src)
        self.enabled = bytearray(b"\x01") * m
        self.in_ptr, self.in_idx = self._csr(self.dst, N, m)
        self.out_ptr, self.out_idx = self._csr(self.src, N, m)

    @staticmethod
    def _csr(keys, N: int, m: int):
        I32 = _i32()
        ptr = array.array(I32, bytes(4 * (N + 1)))
        for v in keys:
            ptr[v + 1] += 1
        for i in range(N):
            ptr[i + 1] += ptr[i]
        fill = array.array(I32, ptr)
        idx = array.array(I32, bytes(4 * m))
        for e, v in enumerate(keys):
            idx[fill[v]] = e
            fill[v] += 1
        return ptr, idx

    def inferred(self, e: int) -> bool:
        return self.mode[e] == DEP and edge_class(PROVS[self.prov[e]], CONFS[self.conf[e]]) == "inferred"

    def name(self, j: int) -> str:
        return self.names[self.name_id[j]]

    def runtime(self, e: int) -> str:
        if self.mode[e] in (BUSY, WAIT):
            return self.cat[self.owner[e]]
        return runtime_of_edge(KINDS[self.kind[e]], self.cat[self.src[e] >> 1], self.cat[self.dst[e] >> 1])


def _excluded(s) -> bool:
    return is_annotation(s) or is_sampled(s) or s.category.value == "sched"


def build_event_graph(trace: "Trace", graph) -> EventGraph:
    """The event-point graph of a trace from its persisted span-level edges
    (criticalpath.Graph). Spans are read once, in ordinal (= seq) order."""
    from . import criticalpath as cp

    n = len(graph.seqs)
    g = EventGraph(n)
    stream_key: dict[int, tuple] = {}
    lane = array.array(_i32(), [-1]) * n           # thread lane of each host span
    resolved_wildcard = bytearray(n)
    name_ids: dict[str, int] = {}
    lanes: dict[tuple[int, int], array.array] = defaultdict(lambda: array.array("q"))
    count = 0
    for i, s in enumerate(trace.iter_spans()):
        count += 1
        st = cp._effective_start_ns(s)
        g.t[2 * i] = st
        g.t[2 * i + 1] = st + max(0, s.duration_ns)
        g.cat[i] = s.category.value
        nid = name_ids.get(s.name)
        if nid is None:
            nid = name_ids[s.name] = len(g.names)
            g.names.append(s.name)
        g.name_id[i] = nid
        tags = s.tags
        if _excluded(s):
            g.excluded[i] = 1
            continue
        if is_device_timed(s):
            g.device[i] = 1
            stream_key[i] = (s.pid, tags.get("rt") or s.category.value, tags.get("stream"), tags.get("dev"))
        else:
            lanes[(s.pid, s.tid)].append(i)
        if s.category.value == "mpi" and ("rpeer" in tags or "rmatches" in tags):
            resolved_wildcard[i] = 1
        if tags.get("type") in cp.ROOTED_COLLECTIVE_TYPES:
            g.rooted[i] = 1
    if count != n:
        raise ValueError(f"{count} spans, but the graph has {n} ordinals")
    for li, members in enumerate(lanes.values()):
        for j in members:
            lane[j] = li

    _add_dependency_edges(g, graph.edges, cp.EDGE_KINDS, stream_key, lane, resolved_wildcard)
    del stream_key, lane, resolved_wildcard
    # a call whose end a dependency from another lane gates can wait inside
    for e in range(len(g)):
        if g.mode[e] == DEP and g.dst[e] & 1:
            g.gated[g.dst[e] >> 1] = 1
    for key in list(lanes):
        _add_thread_lane(g, lanes.pop(key))
    kind, prov = _K["device_work"], _P["device_interval"]
    for j in range(n):
        if g.device[j] and not g.excluded[j]:
            g.add(2 * j, 2 * j + 1, WAIT if g.gated[j] else BUSY, kind, prov, 0, j)
    g.finish()
    _break_cycles(g)
    return g


def _chunks(edges, size: int = 65536):
    for a in range(0, len(edges), size):
        part = edges[a:a + size]
        yield from zip(part["dst"].tolist(), part["src"].tolist(), part["kind"].tolist(),
                       part["conf"].tolist())


def _add_dependency_edges(g: EventGraph, edges, edge_kinds, stream_key, lane, resolved_wildcard) -> None:
    tol = ga.CLOCK_TOLERANCE_NS
    high = _C["high"]
    for v, u, k, c in _chunks(edges):
        if g.excluded[u] or g.excluded[v]:
            continue
        kind = edge_kinds[k]
        S_u, E_u, S_v, E_v = 2 * u, 2 * u + 1, 2 * v, 2 * v + 1
        same_lane = lane[u] >= 0 and lane[u] == lane[v]
        if kind == "sequential":
            du, dv = g.device[u], g.device[v]
            if du and dv:
                if stream_key.get(u) == stream_key.get(v):
                    g.add(E_u, S_v, IDLE, _K["stream_order"], _P["stream_fifo"], c, -1, tol)
                else:
                    g.add(E_u, S_v, DEP, _K["event_wait"], _P["event_record"], c, -1, tol)
            elif du and g.cat[v] == "sync":           # pre-split: a stream sync waits for its stream
                g.add(E_u, E_v, DEP, _K["device_sync"], _P["stream_fifo"], c, -1, tol)
            elif du or dv:                            # pre-split: a device span and its launching thread
                g.add(E_u, S_v, DEP, _K["sequential"], _P["stream_fifo"] if g.cat[u] == "sync"
                      else _P["thread_timeline"], c, -1, tol)
            elif not same_lane:
                g.unmapped["sequential across threads"] += 1
        elif kind == "launch":
            g.add(S_u, S_v, DEP, _K["launch"], _P["correlation_id"], c, -1, tol)
        elif kind == "device_wait":
            g.add(E_u, E_v, DEP, _K["device_wait"], _P["sync_target"], c, -1, tol)
        elif kind == "device_sync":
            g.add(E_u, E_v, DEP, _K["device_sync"], _P["sync_target"], c, -1, tol)
        elif kind == "p2p":
            prov = "status_resolved" if resolved_wildcard[v] and c == high else "posting_order"
            g.add(S_u, E_v, DEP, _K["p2p"], _P[prov], c, -1)
        elif kind == "arrival":
            if c == high:
                prov = "nccl_seq" if g.cat[v] == "nccl" else "call_order"
            else:
                prov = "time_overlap"
            g.add(S_u, E_v, DEP, _K["arrival"], _P[prov], c, -1)
        elif kind == "explicit_span_id":
            if same_lane:
                continue
            tu0, tu1, tv0 = g.t[S_u], g.t[E_u], g.t[S_v]
            if tu0 <= tv0 <= tu1 and g.cat[u] == "openmp":
                g.add(S_u, S_v, DEP, _K["fork"], _P["fork_join"], c, -1)
                g.add(E_v, E_u, DEP, _K["join"], _P["fork_join"], c, -1)
            elif tu1 <= tv0:
                g.add(E_u, S_v, DEP, _K["explicit_span_id"], _P["explicit_id"], c, -1)
            elif tu0 <= tv0:
                g.add(S_u, S_v, DEP, _K["explicit_span_id"], _P["explicit_id"], c, -1)
            else:
                g.add(S_u, E_v, DEP, _K["explicit_span_id"], _P["explicit_id"], c, -1)
        else:
            g.unmapped[kind] += 1


def _add_thread_lane(g: EventGraph, members) -> None:
    evs = []
    for j in members:
        st, en = g.t[2 * j], g.t[2 * j + 1]
        if en > st:
            evs.append((st, 2, -en, j, 2 * j))           # enclosing starts first
            evs.append((en, 0, -st, -j, 2 * j + 1))      # nested ends first
        else:
            evs.append((st, 1, 0, j, -1))                # zero length: S then E together
    evs.sort()
    kind, prov = _K["thread"], _P["thread_timeline"]
    open_: list[int] = []
    prev = -1
    for _t, _o, _k, _j, node in evs:
        j = node >> 1 if node >= 0 else _j
        for nd in ((2 * j, 2 * j + 1) if node < 0 else (node,)):
            if prev >= 0:
                inner = open_[-1] if open_ else -1
                if inner < 0:
                    g.add(prev, nd, GAP, kind, prov, 0, -1)
                else:
                    g.add(prev, nd, WAIT if g.gated[inner] else BUSY, kind, prov, 0, inner)
            if node >= 0:
                if nd & 1:
                    if open_ and open_[-1] == j:
                        open_.pop()
                    elif j in open_:
                        open_.remove(j)
                        g.improper_nesting += 1
                else:
                    open_.append(j)
            prev = nd


def _topological_order(g: EventGraph) -> array.array:
    N = 2 * g.n
    indeg = array.array("q", bytes(8 * N))
    for e in range(len(g)):
        if g.enabled[e]:
            indeg[g.dst[e]] += 1
    dq = deque(v for v in range(N) if g.active[v] and indeg[v] == 0)
    order = array.array(_i32())
    out_ptr, out_idx, dst, enabled = g.out_ptr, g.out_idx, g.dst, g.enabled
    while dq:
        u = dq.popleft()
        order.append(u)
        for k in range(out_ptr[u], out_ptr[u + 1]):
            e = out_idx[k]
            if enabled[e]:
                v = dst[e]
                indeg[v] -= 1
                if indeg[v] == 0:
                    dq.append(v)
    return order


def _break_cycles(g: EventGraph) -> None:
    """Every edge points forward in time or between events at one instant
    (zero-length, or within the clock tolerance), so a cycle can only close
    among such events. Disable the dependency edges between nodes a
    topological sort cannot place (counted), then lane edges if needed."""
    order = _topological_order(g)
    n_active = sum(g.active)
    for dep_only in (True, False):
        if len(order) == n_active:
            break
        placed = bytearray(2 * g.n)
        for v in order:
            placed[v] = 1
        for e in range(len(g)):
            if g.enabled[e] and not placed[g.src[e]] and not placed[g.dst[e]] \
                    and (not dep_only or g.mode[e] == DEP):
                g.enabled[e] = 0
                g.cycle_edges_dropped += 1
        order = _topological_order(g)
    g.order = order


def _w(mode: int, d: int, gap_as_work: bool) -> int:
    return d if mode == BUSY or (mode == GAP and gap_as_work) else 0


# ── observed critical path ──────────────────────────────────────────────────

@dataclass
class Walk:
    steps: array.array               # edge ids, forward order
    cover: array.array               # time each step covers (its d, unless a clamped edge overlapped)
    first: int                       # first node (-1: empty)
    sink: int
    ties: int = 0
    tie_samples: list[tuple[int, tuple[str, ...]]] = field(default_factory=list)

    @property
    def length(self) -> int:
        return sum(self.cover)

    def spans(self, g: EventGraph) -> set[int]:
        """Spans that own time on the path, or whose events it passes."""
        out = {self.first >> 1} if self.first >= 0 else set()
        for e in self.steps:
            out.add(g.dst[e] >> 1)
            if g.owner[e] >= 0:
                out.add(g.owner[e])
        return out


def last_events(g: EventGraph) -> list[int]:
    """The events at the latest timestamp that nothing follows (an
    enclosing span's end, not the nested end at the same instant), highest
    node first. Several: the trace ends on several lanes at once."""
    if not g.order:
        return []
    tmax = max(g.t[v] for v in g.order)
    at = [v for v in g.order if g.t[v] == tmax]
    free = [v for v in at if not any(g.enabled[g.out_idx[k]]
                                     for k in range(g.out_ptr[v], g.out_ptr[v + 1]))]
    return sorted(free or at, reverse=True)


def walk(g: EventGraph, *, gap_as_work: bool = False, direct_only: bool = False) -> Walk:
    """The chain of last-releasing edges, backwards from the last event
    (several at the latest instant: the highest node, counted as a tie).
    Release time of an edge: t(source) + w, gap edges counted as work only
    when gap_as_work. Tie order: stay on the lane, the stronger tier, the
    later source node."""
    ends = last_events(g)
    steps = array.array(_i32())
    cover = array.array("q")
    if not ends:
        return Walk(steps, cover, -1, -1)
    sink = ends[0]
    ties = 0
    samples: list[tuple[int, tuple[str, ...]]] = []
    if len(ends) > 1:
        ties = 1
        samples.append((g.t[sink], ("end of trace",)))
    t, src, mode, d, conf, enabled = g.t, g.src, g.mode, g.d, g.conf, g.enabled
    seen = bytearray(2 * g.n)
    front = t[sink]
    v = sink
    while not seen[v]:
        seen[v] = 1
        best_key = None
        best = -1
        cands: list[tuple[int, int]] = []
        for k in range(g.in_ptr[v], g.in_ptr[v + 1]):
            e = g.in_idx[k]
            if not enabled[e] or (direct_only and g.inferred(e)):
                continue
            rel = t[src[e]] + _w(mode[e], d[e], gap_as_work)
            cands.append((rel, e))
            key = (rel, mode[e] != DEP, -conf[e], src[e])
            if best_key is None or key > best_key:
                best_key, best = key, e
        if best < 0:
            break
        if any(rel == best_key[0] and src[e] != src[best] for rel, e in cands):
            ties += 1
            if len(samples) < 8:
                samples.append((t[v], tuple(sorted({KINDS[g.kind[e]] for rel, e in cands
                                                    if rel == best_key[0]}))))
        u = src[best]
        lo = min(t[u], front)
        cover.append(max(0, min(t[v], front) - lo))
        front = lo
        steps.append(best)
        v = u
    steps.reverse()
    cover.reverse()
    return Walk(steps, cover, v, sink, ties, samples)


# ── modeled longest path ─────────────────────────────────────────────────────

def longest(g: EventGraph, *, gap_as_work: bool = True, direct_only: bool = False,
            zero_owners: set[int] | None = None) -> tuple[array.array, int]:
    """Forward longest-path values by required time, and the maximum."""
    F = array.array("q", bytes(16 * g.n))
    in_ptr, in_idx, src, enabled, mode, d, owner = (g.in_ptr, g.in_idx, g.src, g.enabled, g.mode,
                                                     g.d, g.owner)
    best_all = 0
    for v in g.order:
        best = 0
        for k in range(in_ptr[v], in_ptr[v + 1]):
            e = in_idx[k]
            if not enabled[e]:
                continue
            m = mode[e]
            if m == BUSY:
                w = 0 if zero_owners is not None and owner[e] in zero_owners else d[e]
            elif m == GAP and gap_as_work:
                w = d[e]
            elif m == DEP and direct_only and g.inferred(e):
                continue
            else:
                w = 0
            val = F[src[e]] + w
            if val > best:
                best = val
        F[v] = best
        if best > best_all:
            best_all = best
    return F, best_all


def _backward(g: EventGraph, gap_as_work: bool = True) -> array.array:
    B = array.array("q", bytes(16 * g.n))
    out_ptr, out_idx, dst, enabled, mode, d = g.out_ptr, g.out_idx, g.dst, g.enabled, g.mode, g.d
    for u in reversed(g.order):
        best = 0
        for k in range(out_ptr[u], out_ptr[u + 1]):
            e = out_idx[k]
            if enabled[e]:
                val = B[dst[e]] + _w(mode[e], d[e], gap_as_work)
                if val > best:
                    best = val
        B[u] = best
    return B


class SpanValues:
    """A read-only {span: value} over a per-span array (-1: no value)."""

    def __init__(self, values: array.array) -> None:
        self._v = values

    def get(self, j: int, default=None):
        x = self._v[j]
        return default if x < 0 else x

    def __getitem__(self, j: int) -> int:
        x = self._v[j]
        if x < 0:
            raise KeyError(j)
        return x

    def __contains__(self, j: int) -> bool:
        return 0 <= j < len(self._v) and self._v[j] >= 0

    def items(self):
        return ((j, x) for j, x in enumerate(self._v) if x >= 0)

    def __len__(self) -> int:
        return sum(1 for x in self._v if x >= 0)


@dataclass
class Criticality:
    length: int
    on_some: set[int]                # spans with busy time on some longest path
    on_all: set[int]                 # ... on every longest path
    slack: SpanValues                # span -> least slack of its busy segments
    work_on_some: dict[int, int]     # span -> its busy time on longest paths
    work_on_all_ns: int              # busy time on every longest path
    work_on_some_ns: int             # busy time on at least one


def criticality(g: EventGraph, F: array.array, length: int) -> Criticality:
    """Tight edges and the cut test (module docstring), on the modeled path
    with unrecorded thread time counted as work."""
    B = _backward(g)
    order = g.order
    pos = array.array(_i32(), bytes(8 * g.n))
    for p, v in enumerate(order):
        pos[v] = p
    M = len(order)
    # cuts 0..M; cut c separates order positions < c from >= c; the virtual
    # source sits before position 0, the virtual sink after M - 1
    diff = array.array("q", bytes(8 * (M + 2)))
    tight = bytearray(len(g))
    for e in range(len(g)):
        if not g.enabled[e]:
            continue
        u, v = g.src[e], g.dst[e]
        if F[u] + _w(g.mode[e], g.d[e], True) + B[v] == length:
            tight[e] = 1
            diff[pos[u] + 1] += 1
            diff[pos[v] + 1] -= 1
    for v in order:
        if B[v] == length:                     # virtual source -> v
            diff[0] += 1
            diff[pos[v] + 1] -= 1
        if F[v] == length:                     # v -> virtual sink
            diff[pos[v] + 1] += 1
            diff[M + 1] -= 1
    ones = array.array("q", bytes(8 * (M + 2)))   # ones[c + 1]: cuts 0..c crossed by one tight edge
    run = acc = 0
    for c in range(M + 1):
        run += diff[c]
        if run == 1:
            acc += 1
        ones[c + 1] = acc
    on_some: set[int] = set()
    on_all: set[int] = set()
    slack = array.array("q", [-1]) * g.n
    work: dict[int, int] = defaultdict(int)
    all_ns = some_ns = 0
    for e in range(len(g)):
        if not g.enabled[e] or g.mode[e] != BUSY or g.d[e] <= 0:
            continue
        o, u, v, d = g.owner[e], g.src[e], g.dst[e], g.d[e]
        s = length - (F[u] + d + B[v])
        if slack[o] < 0 or s < slack[o]:
            slack[o] = s
        if tight[e]:
            on_some.add(o)
            work[o] += d
            some_ns += d
            if ones[pos[v] + 1] - ones[pos[u] + 1] > 0:      # a cut in pos[u]+1 .. pos[v]
                on_all.add(o)
                all_ns += d
    return Criticality(length, on_some, on_all, SpanValues(slack), dict(work), all_ns, some_ns)


# ── result ──────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class PathRow:
    """One run of steps on the observed path: time owned by a span (busy or
    wait), or time before an event (gap, device idle, dependency latency)."""
    span: int                # owner, or the span whose event the step reached
    own_ns: int
    other_ns: int
    start_ns: int
    end_ns: int
    mode: str
    kind: str
    conf: str
    source: int = -1         # dependency steps: the span whose event released it


@dataclass
class ModelResult:
    graph: EventGraph
    walk: Walk
    covered_ns: int
    work_by_category: dict[str, int]
    in_call_by_category: dict[str, int]
    unrecorded_ns: int
    device_idle_ns: int
    dependency_by_kind: dict[str, int]
    dependency_by_source_category: dict[str, int]
    by_tier: dict[str, int]
    by_class: dict[str, int]
    by_provenance: dict[str, tuple[int, int]]          # prov -> (ns, steps)
    rows: list[PathRow]
    # span-level view: spans in first-appearance order, time each owns on
    # the path, other time before it, and the step that reached it
    path_spans: list[int]
    credited: list[int]
    gaps: list[int]
    entering_kinds: list[str]
    entering_confs: list[str]
    modeled_ns: int = 0
    modeled_unrecorded_idle_ns: int = 0
    modeled_direct_only_ns: int = 0
    crit: Criticality | None = None
    sensitivity: list[dict] = field(default_factory=list)
    sensitivity_note: str = ""
    variants: dict[str, dict] = field(default_factory=dict)
    stable_spans: int = 0
    variant_only_spans: int = 0
    variant_only_ns: int = 0
    graph_evidence: dict = field(default_factory=dict)

    def dependency_steps(self, limit: int | None = None) -> list[dict]:
        """The dependency steps of the observed path with their evidence,
        longest first."""
        g = self.graph
        out = []
        for e, ns in zip(self.walk.steps, self.walk.cover):
            if g.mode[e] != DEP:
                continue
            kind, prov, conf = KINDS[g.kind[e]], PROVS[g.prov[e]], CONFS[g.conf[e]]
            u, v = g.src[e] >> 1, g.dst[e] >> 1
            out.append({"ns": ns, "kind": kind, "runtime": g.runtime(e), "tier": conf,
                        "provenance": prov, "class": edge_class(prov, conf),
                        "reason": edge_reason(kind, prov, conf),
                        "from": (g.cat[u], g.name(u), u), "to": (g.cat[v], g.name(v), v),
                        "at_ns": g.t[g.dst[e]]})
        out.sort(key=lambda r: -r["ns"])
        return out[:limit] if limit is not None else out


_LEGACY_KIND = {"thread": "sequential", "device_work": "sequential", "stream_order": "sequential",
                "event_wait": "sequential", "fork": "explicit_span_id", "join": "explicit_span_id"}


def legacy_kind(kind: str) -> str:
    """The span-level edge kind (criticalpath.EDGE_KINDS) a step corresponds to."""
    return _LEGACY_KIND.get(kind, kind)


def path_rows(g: EventGraph, w: Walk) -> list[PathRow]:
    """Steps merged into runs: consecutive lane steps of one owner in one
    mode, and every other step on its own."""
    rows: list[PathRow] = []
    for e, ns in zip(w.steps, w.cover):
        m = g.mode[e]
        u, v = g.src[e], g.dst[e]
        kind, conf = KINDS[g.kind[e]], CONFS[g.conf[e]]
        if m in (BUSY, WAIT):
            o = g.owner[e]
            last = rows[-1] if rows else None
            if last is not None and last.span == o and last.mode == MODES[m] and last.end_ns == g.t[u]:
                last.own_ns += ns
                last.end_ns = g.t[v]
                continue
            rows.append(PathRow(o, ns, 0, g.t[u], g.t[v], MODES[m], kind, conf))
        else:
            rows.append(PathRow(v >> 1, 0, ns, g.t[u], g.t[v], MODES[m], kind, conf,
                                (u >> 1) if m == DEP else -1))
    return rows


def _span_view(g: EventGraph, w: Walk):
    order: list[int] = []
    index: dict[int, int] = {}
    credited: list[int] = []
    gaps: list[int] = []
    kinds: list[str] = []
    confs: list[str] = []

    def intro(j: int, kind: str, conf: str) -> int:
        k = index.get(j)
        if k is None:
            k = index[j] = len(order)
            order.append(j)
            credited.append(0)
            gaps.append(0)
            kinds.append(kind)
            confs.append(conf)
        return k

    if w.first >= 0:
        intro(w.first >> 1, "", "")
    for e, ns in zip(w.steps, w.cover):
        m = g.mode[e]
        kind, conf = legacy_kind(KINDS[g.kind[e]]), CONFS[g.conf[e]]
        if m in (BUSY, WAIT):
            credited[intro(g.owner[e], kind, conf)] += ns
            intro(g.dst[e] >> 1, kind, conf)
        else:
            gaps[intro(g.dst[e] >> 1, kind, conf)] += ns
    return order, credited, gaps, kinds[1:], confs[1:]


def observed_path(trace: "Trace", graph) -> tuple[EventGraph, Walk]:
    g = build_event_graph(trace, graph)
    return g, walk(g)


def analyze_model(trace: "Trace", graph, *, sensitivity: int | None = None,
                  sensitivity_node_limit: int = 4_000_000) -> ModelResult:
    """The observed path, its composition and evidence, the modeled longest
    path, criticality, alternatives and (top `sensitivity` candidates)
    counterfactual reductions. sensitivity=None: 5 when the graph has at
    most `sensitivity_node_limit` event nodes, else none (each candidate is
    a full longest-path pass); 0: none."""
    g, w = observed_path(trace, graph)
    work: dict[str, int] = defaultdict(int)
    in_call: dict[str, int] = defaultdict(int)
    dep_kind: dict[str, int] = defaultdict(int)
    dep_src: dict[str, int] = defaultdict(int)
    tier: dict[str, int] = defaultdict(int)
    cls: dict[str, int] = defaultdict(int)
    prov: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    unrecorded = idle = 0
    for e, ns in zip(w.steps, w.cover):
        m = g.mode[e]
        if m == BUSY:
            work[g.cat[g.owner[e]]] += ns
        elif m == WAIT:
            in_call[g.cat[g.owner[e]]] += ns
        elif m == GAP:
            unrecorded += ns
        elif m == IDLE:
            idle += ns
        else:
            dep_kind[KINDS[g.kind[e]]] += ns
            dep_src[g.cat[g.src[e] >> 1]] += ns
        p, c = PROVS[g.prov[e]], CONFS[g.conf[e]]
        tier[c] += ns
        cls[edge_class(p, c)] += ns
        prov[p][0] += ns
        prov[p][1] += 1
    spans, credited, gaps, kinds, confs = _span_view(g, w)
    res = ModelResult(
        graph=g, walk=w, covered_ns=w.length,
        work_by_category=dict(work), in_call_by_category=dict(in_call),
        unrecorded_ns=unrecorded, device_idle_ns=idle,
        dependency_by_kind=dict(dep_kind), dependency_by_source_category=dict(dep_src),
        by_tier=dict(tier), by_class=dict(cls), by_provenance={k: (v[0], v[1]) for k, v in prov.items()},
        rows=path_rows(g, w), path_spans=spans, credited=credited, gaps=gaps,
        entering_kinds=kinds, entering_confs=confs,
    )
    res.graph_evidence = _graph_evidence(g)
    if g.order:
        F, L = longest(g)
        res.modeled_ns = L
        res.modeled_unrecorded_idle_ns = longest(g, gap_as_work=False)[1]
        res.modeled_direct_only_ns = longest(g, direct_only=True)[1]
        res.crit = criticality(g, F, L)
        k = sensitivity
        if k is None:
            if 2 * g.n > sensitivity_node_limit:
                res.sensitivity_note = (f"counterfactual reductions skipped: {2 * g.n} event nodes exceed "
                                        f"{sensitivity_node_limit} (each candidate is a full longest-path "
                                        "pass; --sensitivity N forces N)")
                k = 0
            else:
                k = 5
        _sensitivity(res, g, L, k)
    _alternatives(res, g, w)
    return res


def _graph_evidence(g: EventGraph) -> dict:
    by_prov: Counter = Counter()
    by_cls: Counter = Counter()
    by_tier: Counter = Counter()
    by_kind: Counter = Counter()
    rooted = 0
    for e in range(len(g)):
        if not g.enabled[e] or g.mode[e] not in (DEP, IDLE):
            continue
        p, c = PROVS[g.prov[e]], CONFS[g.conf[e]]
        by_prov[p] += 1
        by_cls[edge_class(p, c)] += 1
        by_tier[c] += 1
        by_kind[KINDS[g.kind[e]]] += 1
        if g.kind[e] == _K["arrival"] and g.rooted[g.dst[e] >> 1]:
            rooted += 1
    return {"by_provenance": dict(by_prov), "by_class": dict(by_cls), "by_tier": dict(by_tier),
            "by_kind": dict(by_kind), "rooted_collective_edges": rooted,
            "dropped_contradicting_time": dict(g.dropped), "dropped_in_cycles": g.cycle_edges_dropped,
            "improper_nesting": g.improper_nesting, "unmapped": dict(g.unmapped)}


def _alternatives(res: ModelResult, g: EventGraph, w: Walk) -> None:
    variants = {"default": w, "unrecorded_as_work": walk(g, gap_as_work=True),
                "direct_evidence_only": walk(g, direct_only=True)}
    sets = {k: v.spans(g) for k, v in variants.items()}
    every = set.intersection(*sets.values())
    some = set.union(*sets.values())
    res.stable_spans = len(every)
    res.variant_only_spans = len(some - every)
    owned: dict[int, int] = defaultdict(int)
    for e, ns in zip(w.steps, w.cover):
        if g.mode[e] in (BUSY, WAIT):
            owned[g.owner[e]] += ns
    res.variant_only_ns = sum(ns for j, ns in owned.items() if j not in every)
    for name, v in variants.items():
        res.variants[name] = {"covered_ns": v.length, "spans": len(sets[name]),
                              "first_ns": g.t[v.first] if v.first >= 0 else None,
                              "same_spans_as_default": sets[name] == sets["default"]}


def _sensitivity(res: ModelResult, g: EventGraph, L: int, k: int) -> None:
    """Counterfactuals: the modeled longest path with the busy time of all
    spans of one (category, name), or of one span, set to zero. Candidates:
    the top k by busy time on longest paths (single spans on every longest
    path first); one longest-path pass each."""
    crit = res.crit
    if k <= 0 or crit is None or not crit.work_on_some:
        return
    work_by_name: dict[tuple[str, int], int] = defaultdict(int)
    for j, ns in crit.work_on_some.items():
        work_by_name[(g.cat[j], g.name_id[j])] += ns
    top = sorted(work_by_name.items(), key=lambda kv: -kv[1])[:k]
    members: dict[tuple[str, int], set[int]] = {key: set() for key, _ns in top}
    for j in range(g.n):
        key = (g.cat[j], g.name_id[j])
        if key in members and not g.excluded[j]:
            members[key].add(j)
    for key, ns in top:
        L2 = longest(g, zero_owners=members[key])[1]
        res.sensitivity.append({"scope": "name", "category": key[0], "name": g.names[key[1]],
                                "spans": len(members[key]), "work_on_longest_ns": ns,
                                "modeled_reduction_ns": L - L2})
    singles = sorted(crit.work_on_some.items(), key=lambda kv: (kv[0] not in crit.on_all, -kv[1]))[:k]
    for j, ns in singles:
        L2 = longest(g, zero_owners={j})[1]
        res.sensitivity.append({"scope": "span", "category": g.cat[j], "name": g.name(j), "span": j,
                                "start_ns": g.t[2 * j], "work_on_longest_ns": ns,
                                "on_every_longest_path": j in crit.on_all, "slack_ns": crit.slack.get(j, 0),
                                "modeled_reduction_ns": L - L2})
