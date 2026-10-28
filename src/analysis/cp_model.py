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
            into waiting and processing. Required time w in [0, d_post],
            d_post the part after the latest source event of the
            dependencies gating it (processing can only follow the
            release; the whole call when the source came first, and for
            the last arriver of a rendezvous): w = 0 by default, w =
            d_post in the "gated_tail_as_work" variant -- where each
            dependency that gates the call carries the call's required
            time after its release too, since that processing follows
            whichever source arrived last
      unknown  inside a call that waits for another lane by its runtime's
            semantics (semantics.blocking_wait: a receive, a collective, a
            barrier, a sync call...) but no recorded dependency gates it --
            an unmatched receive, a lost message: waiting for something the
            trace did not record, or processing. w in [0, d], treated as
            "gap" (never as plain work)
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

Clocks
------
Timestamps come from several clock domains (cp_clock.py): each node's host
clock, mapped device clocks, OpenCL's calibrated device clock, NCCL's
asynchronous reference events. Their unknown offsets are bounded, and the
dependencies constrain them further (source no later than destination): a
simple temporal network solved over the domains. Edges no offset within the
bounds can satisfy together with the rest are contradicted and disabled
(counted); a domain with no bound and no correction of its own (OpenCL)
whose recorded times contradict its dependencies is shifted by the smallest
amount that satisfies them before the graph is built. The walk compares two
releases from different domains by their offset interval: when the clocks
cannot tell which came last, the step is "undecidable" (counted, with the
time at stake).

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
    the dependencies themselves are not modeled. Alongside, the same
    intervention replayed (cp_attribution.replay: every event as early as
    its predecessors allow plus its recorded residual) -- how much earlier
    the run would end with waits and dependency latencies adapting.

Attribution (cp_attribution.py) answers three questions separately: the
observed path is the witness (where it runs), blame charges every gated
call's waiting time to the work that delayed its release (joint releases
shared), and the replay above is the intervention.

Bounds and alternatives
-----------------------
Where the data does not decide, the model is computed both ways instead of
enumerating paths. VARIANTS name the choices, and each variant's observed
walk and modeled longest path use the same parameters:
  gaps      unrecorded / unknown time as idle (released at its start) or as
            work ("unrecorded_as_work")
  evidence  every dependency edge, or direct-evidence ones only
            ("direct_evidence_only")
  gated     a gated call's tail after its release as waiting (w = 0) or as
            processing ("gated_tail_as_work")
"default" is the observed path as recorded (gaps idle, every edge, tails
waiting); the headline modeled length keeps counting unrecorded time as
work (modeled_ns), with the range over all variants alongside. Spans on the
observed path of every variant vs. only some show how much the path rests
on those choices. A few walks and linear passes: no exponential search.

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

BUSY, WAIT, GAP, IDLE, DEP, UNKNOWN = range(6)
MODES = ("busy", "wait", "gap", "idle", "dependency", "unknown")

# name -> (gaps as work, direct evidence only, gated tails as work)
VARIANTS: dict[str, tuple[bool, bool, bool]] = {
    "default": (False, False, False),
    "unrecorded_as_work": (True, False, False),
    "direct_evidence_only": (False, True, False),
    "gated_tail_as_work": (False, False, True),
}

KINDS = ("thread", "device_work", "stream_order", "event_wait", "launch", "device_wait",
         "device_sync", "p2p", "arrival", "fork", "join", "explicit_span_id", "sequential",
         "task_join", "task_dependence", "lock_handoff")
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
    "task_id": ("direct", "the OMPT task ids on a task's creation (parent=), its execution (id=) and the "
                          "taskwait / taskgroup of its parent (task=), or a recorded task dependence"),
    "lock_handoff": ("inferred", "the release of the same lock (wait_id= / lock=) just before a contended "
                                 "acquisition, by another thread, on one clock"),
    "nccl_pseq": ("direct", "NCCL pairs the sends and receives between two ranks of a communicator in "
                            "issue order; the hook numbered them (pseq=)"),
    "rooted_flow": ("inferred", "a rooted collective's data flow with the root its members recorded "
                                "(wroot=): from the root (bcast, scatter) or to it (reduce, gather), "
                                "members grouped by call order"),
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
        return "nccl" if dst_cat == "nccl" else "mpi"
    if kind in ("task_join", "task_dependence", "lock_handoff"):
        return "openmp"
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
    arrays (src, dst, mode, kind, prov, conf, owner span, observed d, and
    origin: the index of the span-level edge it came from, -1 for lane
    edges)."""

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
        self.whi = array.array("q")               # required time's upper bound (see "Lane edges")
        self.origin = array.array("q")
        self._origin = -1                          # span-level edge being added
        self.enabled = bytearray()
        self.order = array.array(_i32())          # active nodes, topological order
        # per span
        self.cat: list[str] = [""] * n
        self.name_id = array.array(_i32(), bytes(4 * n))
        self.names: list[str] = []
        self.device = bytearray(n)
        self.gated = bytearray(n)
        self.blocking = bytearray(n)              # semantics.blocking_wait
        self.release = array.array("q", [-1]) * n  # gated call: latest source event of its gates
        self.rooted = bytearray(n)                # rooted collective, root not recorded
        self.rootflow = bytearray(n)              # rooted collective with its root recorded
        self.excluded = bytearray(n)
        # diagnostics
        self.dropped: Counter = Counter()        # dependency kind -> edges contradicting timestamps
        self.dropped_edges = array.array("q")     # their span-level edge indices
        self.dropped_pairs: list[tuple[int, int, int, int]] = []   # (u, v, kind, conf) event nodes
        self.domain = array.array(_i32(), bytes(4 * n))   # span -> clock domain id (cp_clock)
        self.clock = None                         # cp_clock.ClockAnalysis
        self.clock_contradicted = 0               # edges disabled: no clock offsets satisfy them
        self.clock_disabled: set[int] = set()     # their event edge ids
        self.skeleton = None                      # Skeleton (build_event_graph(keep_skeleton=True))
        self.skipped_edges = array.array("q")     # span-level edges with no event edge (same lane, unmapped)
        self.improper_nesting = 0
        self.cycle_edges_dropped = 0
        self.unmapped: Counter = Counter()

    def add(self, u: int, v: int, mode: int, kind: int, prov: int, conf: int, owner: int,
            tol: int = 0) -> bool:
        d = self.t[v] - self.t[u]
        if d < 0:
            if -d > tol:
                self.dropped[KINDS[kind]] += 1
                if self._origin >= 0:
                    self.dropped_edges.append(self._origin)
                self.dropped_pairs.append((u, v, kind, conf))
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
        self.origin.append(self._origin)
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


@dataclass
class Skeleton:
    """What build_event_graph reads from the spans -- everything but the
    dependency edges -- kept (keep_skeleton=True) so the graphs of
    alternative worlds (cp_alternatives) are assembled without reading the
    spans again (graph_with_edges)."""
    n: int
    t: array.array
    cat: list
    name_id: array.array
    names: list
    device: bytearray
    excluded: bytearray
    rooted: bytearray
    rootflow: bytearray
    blocking: bytearray
    domain: array.array
    stream_key: dict
    lane: array.array
    resolved_wildcard: bytearray
    lanes: list
    dom_keys: list
    dom_bounds: list


def _read_spans(trace: "Trace", graph) -> Skeleton:
    from . import cp_clock
    from . import criticalpath as cp
    from .semantics import blocking_wait

    n = len(graph.seqs)
    node_bounds = cp_clock.node_bounds_of(getattr(trace, "metadata", None))
    dom_ids: dict[tuple, int] = {}
    sk = Skeleton(n=n, t=array.array("q", bytes(16 * n)), cat=[""] * n,
                  name_id=array.array(_i32(), bytes(4 * n)), names=[], device=bytearray(n),
                  excluded=bytearray(n), rooted=bytearray(n), rootflow=bytearray(n), blocking=bytearray(n),
                  domain=array.array(_i32(), bytes(4 * n)), stream_key={},
                  lane=array.array(_i32(), [-1]) * n, resolved_wildcard=bytearray(n), lanes=[],
                  dom_keys=[], dom_bounds=[])
    name_ids: dict[str, int] = {}
    lanes: dict[tuple[int, int], array.array] = defaultdict(lambda: array.array("q"))
    count = 0
    for i, s in enumerate(trace.iter_spans()):
        count += 1
        st = cp._effective_start_ns(s)
        key, bound = cp_clock.domain_of(s, node_bounds)
        di = dom_ids.get(key)
        if di is None:
            di = dom_ids[key] = len(sk.dom_keys)
            sk.dom_keys.append(key)
            sk.dom_bounds.append(bound)
        elif bound > sk.dom_bounds[di]:
            sk.dom_bounds[di] = bound
        sk.domain[i] = di
        sk.t[2 * i] = st
        sk.t[2 * i + 1] = st + max(0, s.duration_ns)
        sk.cat[i] = s.category.value
        nid = name_ids.get(s.name)
        if nid is None:
            nid = name_ids[s.name] = len(sk.names)
            sk.names.append(s.name)
        sk.name_id[i] = nid
        tags = s.tags
        if _excluded(s):
            sk.excluded[i] = 1
            continue
        if is_device_timed(s):
            sk.device[i] = 1
            if "queue" in tags and not tags.get("rt"):      # an OpenCL command queue
                sk.stream_key[i] = (s.pid, "opencl", "q" + str(tags["queue"]), None)
            else:
                sk.stream_key[i] = (s.pid, tags.get("rt") or s.category.value, tags.get("stream"),
                                    tags.get("dev"))
        else:
            lanes[(s.pid, s.tid)].append(i)
        if s.category.value == "mpi" and ("rpeer" in tags or "rmatches" in tags):
            sk.resolved_wildcard[i] = 1
        typ = tags.get("type")
        if typ in cp.ROOTED_COLLECTIVE_TYPES or typ in cp._FROM_ROOT or typ in cp._TO_ROOT:
            if "wroot" in tags and typ not in ("scan", "exscan"):
                sk.rootflow[i] = 1
            elif typ in cp.ROOTED_COLLECTIVE_TYPES:
                sk.rooted[i] = 1
        if blocking_wait(s):
            sk.blocking[i] = 1
    if count != n:
        raise ValueError(f"{count} spans, but the graph has {n} ordinals")
    sk.lanes = list(lanes.values())
    for li, members in enumerate(sk.lanes):
        for j in members:
            sk.lane[j] = li
    if cp_clock.REFERENCE not in dom_ids:
        sk.dom_keys.append(cp_clock.REFERENCE)
        sk.dom_bounds.append(0.0)
    return sk


def _shifted(sk: Skeleton, shift: dict) -> Skeleton:
    """sk with the timestamps of the shifted domains moved."""
    from dataclasses import replace
    t = array.array("q", sk.t)
    by_id = {i: shift[k] for i, k in enumerate(sk.dom_keys) if k in shift}
    for j in range(sk.n):
        d = by_id.get(sk.domain[j])
        if d:
            t[2 * j] += d
            t[2 * j + 1] += d
    return replace(sk, t=t)


def _assemble(sk: Skeleton, edges, *, clocks: bool, prior_clock=None) -> EventGraph:
    """The event graph of the skeleton's spans with these span-level edges.
    The per-span arrays are shared with the skeleton (read only)."""
    from . import criticalpath as cp
    n = sk.n
    g = EventGraph(n)
    g.t = array.array("q", sk.t)
    g.cat, g.name_id, g.names = sk.cat, sk.name_id, sk.names
    g.device, g.excluded, g.rooted, g.rootflow, g.blocking = sk.device, sk.excluded, sk.rooted, sk.rootflow, \
        sk.blocking
    g.domain = sk.domain
    _add_dependency_edges(g, edges, cp.EDGE_KINDS, sk.stream_key, sk.lane, sk.resolved_wildcard)
    # a call whose end a dependency from another lane gates can wait inside;
    # it is released at the latest source event of those dependencies. The
    # last arriver of a rendezvous gates the others and is released by its
    # own arrival: all of its time may be processing.
    arrival = _K["arrival"]
    for e in range(len(g)):
        if g.mode[e] != DEP:
            continue
        if g.dst[e] & 1:
            j = g.dst[e] >> 1
            g.gated[j] = 1
            ts = g.t[g.src[e]]
            if ts > g.release[j]:
                g.release[j] = ts
        if g.kind[e] == arrival:
            u = g.src[e] >> 1
            if not g.gated[u] and g.blocking[u]:
                g.gated[u] = 1
                g.release[u] = max(g.release[u], g.t[2 * u])
    # a rooted collective's member that, by the data flow, waits for nobody
    for j in range(n):
        if g.rootflow[j] and g.blocking[j] and not g.gated[j]:
            g.gated[j] = 1
            g.release[j] = g.t[2 * j]
    for members in sk.lanes:
        _add_thread_lane(g, members)
    kind, prov = _K["device_work"], _P["device_interval"]
    for j in range(n):
        if g.device[j] and not g.excluded[j]:
            g.add(2 * j, 2 * j + 1, WAIT if g.gated[j] else BUSY, kind, prov, 0, j)
    _required_upper_bounds(g)
    g.finish()
    _gating_dependency_bounds(g)
    if clocks:
        g.clock = _clock_analysis(g, sk.dom_keys, sk.dom_bounds)
        _disable_clock_contradicted(g, g.clock)
    else:
        g.clock = prior_clock
    _break_cycles(g)
    return g


def build_event_graph(trace: "Trace", graph, *, clocks: bool = True, keep_skeleton: bool = False) -> EventGraph:
    """The event-point graph of a trace from its persisted span-level edges
    (criticalpath.Graph). Spans are read once, in ordinal (= seq) order.
    clocks: analyze the clock domains (see "Clocks"); a shiftable domain
    whose times contradict its dependencies is shifted and the graph
    assembled once more. keep_skeleton: keep what was read from the spans
    (g.skeleton) for graph_with_edges."""
    sk = _read_spans(trace, graph)
    g = _assemble(sk, graph.edges, clocks=clocks)
    if clocks:
        ca = g.clock
        shift = {sk.dom_keys[i]: sft for i, sft in enumerate(ca.shift) if sft}
        if shift:
            sk = _shifted(sk, shift)
            g = _assemble(sk, graph.edges, clocks=True)
            g.clock.shift = [shift.get(k, 0) for k in g.clock.keys]
            g.clock.feasible_as_recorded = False
            g.clock.relaxed = ca.relaxed + g.clock.relaxed
    if keep_skeleton:
        g.skeleton = sk
    return g


def graph_with_edges(base: EventGraph, edges) -> EventGraph:
    """The event graph of base's spans with other span-level edges (an
    alternative world), from base's skeleton (its timestamps already
    shifted where base's clocks needed it; the offsets are solved again for
    the world's edges). base must be built with keep_skeleton=True."""
    return _assemble(base.skeleton, edges, clocks=True)


def _clock_analysis(g: EventGraph, keys: list[tuple], bounds: list[float]):
    """cp_clock over every dependency edge of g, those add() dropped as
    contradicting the fixed tolerance included (constraint ids len(g)+k)."""
    from . import cp_clock
    cons = []
    for e in range(len(g)):
        if g.mode[e] in (DEP, IDLE):
            cons.append((e, g.src[e], g.dst[e], 0 if g.inferred(e) else 1))
    m = len(g)
    for k, (u, v, kind, conf) in enumerate(g.dropped_pairs):
        cons.append((m + k, u, v, 0 if conf >= _C["medium"] else 1))
    dom = g.domain
    return cp_clock.analyze_clocks(lambda v: dom[v >> 1], g.t, cons, keys, bounds)


def _disable_clock_contradicted(g: EventGraph, ca) -> None:
    for cid in ca.contradicted:
        if cid < len(g) and g.enabled[cid]:
            g.enabled[cid] = 0
            g.clock_contradicted += 1
            g.clock_disabled.add(cid)


def _required_upper_bounds(g: EventGraph) -> None:
    """whi per edge: d for busy, gap and unknown time; for time in a gated
    call, its overlap with [release, end] of that call (all of it when the
    release came before the call); 0 for dependency and device-idle edges
    (_gating_dependency_bounds sets the dependencies into a gated call's
    end once the adjacency exists)."""
    whi = g.whi
    t, src, dst, mode, d, owner, release = g.t, g.src, g.dst, g.mode, g.d, g.owner, g.release
    for e in range(len(src)):
        m = mode[e]
        if m == WAIT:
            j = owner[e]
            lo = max(t[src[e]], release[j])
            whi.append(max(0, min(t[dst[e]], t[2 * j + 1]) - lo) if lo < t[dst[e]] else 0)
        elif m in (BUSY, GAP, UNKNOWN):
            whi.append(d[e])
        else:
            whi.append(0)


def _lane_in(g: EventGraph, v: int) -> int:
    """The lane edge into node v (-1: none)."""
    for k in range(g.in_ptr[v], g.in_ptr[v + 1]):
        e = g.in_idx[k]
        if g.mode[e] in (BUSY, WAIT, GAP, UNKNOWN):
            return e
    return -1


def _tail_after_release(g: EventGraph, j: int) -> int:
    """Gated call j's required time after its release, at most: the upper
    bounds of its lane edges from the release to its end (nested calls
    included; an edge across the release counts its part after it)."""
    t, src, mode, whi = g.t, g.src, g.mode, g.whi
    rel = g.release[j]
    v, total = 2 * j + 1, 0
    while v != 2 * j:
        e = _lane_in(g, v)
        if e < 0:
            break
        w = whi[e] if mode[e] in (BUSY, WAIT, UNKNOWN) else 0
        if t[src[e]] >= rel:
            total += w
            v = src[e]
            continue
        total += min(w, max(0, t[v] - rel))
        break
    return total


def _gating_dependency_bounds(g: EventGraph) -> None:
    """whi of a dependency into a gated call's end: the call's required time
    after its release (_tail_after_release), within the dependency's own
    duration. Counted only in the gated_tail_as_work variant, where the
    processing after the release follows whichever gating source came
    last -- a chain through the dependency carries it, not only the call's
    own lane."""
    tails: dict[int, int] = {}
    whi, d, dst, mode = g.whi, g.d, g.dst, g.mode
    for e in range(len(g)):
        if mode[e] != DEP or not dst[e] & 1:
            continue
        j = dst[e] >> 1
        if not g.gated[j]:
            continue
        r = tails.get(j)
        if r is None:
            r = tails[j] = _tail_after_release(g, j)
        whi[e] = min(r, d[e])


def _chunks(edges, size: int = 65536):
    for a in range(0, len(edges), size):
        part = edges[a:a + size]
        yield from zip(part["dst"].tolist(), part["src"].tolist(), part["kind"].tolist(),
                       part["conf"].tolist())


def _add_dependency_edges(g: EventGraph, edges, edge_kinds, stream_key, lane, resolved_wildcard) -> None:
    tol = ga.CLOCK_TOLERANCE_NS
    high = _C["high"]
    for idx, (v, u, k, c) in enumerate(_chunks(edges)):
        g._origin = idx
        n_before = len(g.src) + len(g.dropped_edges)
        _add_dependency_edge(g, v, u, k, c, edge_kinds, stream_key, lane, resolved_wildcard, tol, high)
        if len(g.src) + len(g.dropped_edges) == n_before:
            g.skipped_edges.append(idx)
    g._origin = -1


def _add_dependency_edge(g: EventGraph, v, u, k, c, edge_kinds, stream_key, lane, resolved_wildcard,
                         tol, high) -> None:
    if g.excluded[u] or g.excluded[v]:
        return
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
        prov = "nccl_pseq" if g.cat[v] == "nccl" else \
            "status_resolved" if resolved_wildcard[v] and c == high else "posting_order"
        g.add(S_u, E_v, DEP, _K["p2p"], _P[prov], c, -1)
    elif kind == "task_join":
        g.add(E_u, E_v, DEP, _K["task_join"], _P["task_id"], c, -1)
    elif kind == "task_dep":
        g.add(E_u, S_v, DEP, _K["task_dependence"], _P["task_id"], c, -1)
    elif kind == "lock_order":
        g.add(E_u, E_v, DEP, _K["lock_handoff"], _P["lock_handoff"], c, -1, tol)
    elif kind == "arrival":
        if c == high:
            prov = "nccl_seq" if g.cat[v] == "nccl" else \
                "rooted_flow" if g.rootflow[u] or g.rootflow[v] else "call_order"
        else:
            prov = "time_overlap"
        g.add(S_u, E_v, DEP, _K["arrival"], _P[prov], c, -1)
    elif kind == "explicit_span_id":
        if same_lane:
            return
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
                    g.add(prev, nd, WAIT if g.gated[inner] else UNKNOWN if g.blocking[inner] else BUSY,
                          kind, prov, 0, inner)
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


def _w(mode: int, d: int, gap_as_work: bool, whi: int = 0, gated_hi: bool = False) -> int:
    """Required time of an edge: busy time always; unrecorded / unknown time
    when counted as work; a gated call's tail (whi), on its lane and on the
    dependencies that release it, when counted as processing."""
    if mode == BUSY:
        return d
    if mode in (GAP, UNKNOWN):
        return d if gap_as_work else 0
    if mode in (WAIT, DEP) and gated_hi:
        return whi
    return 0


# ── observed critical path ──────────────────────────────────────────────────

@dataclass
class Walk:
    steps: array.array               # edge ids, forward order
    cover: array.array               # time each step covers (its d, unless a clamped edge overlapped)
    first: int                       # first node (-1: empty)
    sink: int
    ties: int = 0
    tie_samples: list[tuple[int, tuple[str, ...]]] = field(default_factory=list)
    # steps whose release the clocks cannot order against another candidate
    undecidable: int = 0
    undecidable_ns: int = 0          # how much later the chosen release was, summed
    undecidable_samples: list[dict] = field(default_factory=list)
    undecidable_steps: list[tuple[int, int]] = field(default_factory=list)   # (chosen, alternative) edge

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


def walk(g: EventGraph, *, gap_as_work: bool = False, direct_only: bool = False,
         gated_hi: bool = False) -> Walk:
    """The chain of last-releasing edges, backwards from the last event
    (several at the latest instant: the highest node, counted as a tie).
    Release time of an edge: t(source) + w, gap / unknown edges counted as
    work only when gap_as_work, a gated call's tail only when gated_hi.
    Tie order: stay on the lane, the stronger tier, the later source
    node."""
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
    t, src, mode, d, conf, enabled, whi = g.t, g.src, g.mode, g.d, g.conf, g.enabled, g.whi
    ca = g.clock
    dom = g.domain
    dlo = ca.lo if ca is not None else None
    dhi = ca.hi if ca is not None else None
    undecidable = undecidable_ns = 0
    u_samples: list[dict] = []
    u_steps: list[tuple[int, int]] = []
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
            rel = t[src[e]] + _w(mode[e], d[e], gap_as_work, whi[e], gated_hi)
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
        if dlo is not None and len(cands) > 1:
            # could another candidate's release, on its own clock, have
            # come after the chosen one's?
            a = dom[src[best] >> 1]
            alt, margin = -1, 0
            for rel, e in cands:
                if e == best or rel >= best_key[0]:
                    continue
                b = dom[src[e] >> 1]
                if a == b:
                    continue
                gap = best_key[0] - rel
                if gap + (dlo[a] - dhi[b]) < 0 and (alt < 0 or gap < margin):
                    alt, margin = e, gap
            if alt >= 0:
                undecidable += 1
                undecidable_ns += margin
                u_steps.append((best, alt))
                if len(u_samples) < 8:
                    u_samples.append({"at_ns": t[v], "chosen": KINDS[g.kind[best]],
                                      "alternative": KINDS[g.kind[alt]], "margin_ns": margin})
        u = src[best]
        lo = min(t[u], front)
        cover.append(max(0, min(t[v], front) - lo))
        front = lo
        steps.append(best)
        v = u
    steps.reverse()
    cover.reverse()
    return Walk(steps, cover, v, sink, ties, samples, undecidable, undecidable_ns, u_samples, u_steps)


# ── modeled longest path ─────────────────────────────────────────────────────

def longest(g: EventGraph, *, gap_as_work: bool = True, direct_only: bool = False,
            zero_owners: set[int] | None = None, gated_hi: bool = False) -> tuple[array.array, int]:
    """Forward longest-path values by required time, and the maximum."""
    F = array.array("q", bytes(16 * g.n))
    in_ptr, in_idx, src, enabled, mode, d, owner, whi = (g.in_ptr, g.in_idx, g.src, g.enabled, g.mode,
                                                          g.d, g.owner, g.whi)
    dst = g.dst
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
            elif m in (GAP, UNKNOWN) and gap_as_work:
                w = d[e]
            elif m == WAIT and gated_hi:
                w = 0 if zero_owners is not None and owner[e] in zero_owners else whi[e]
            elif m == DEP and direct_only and g.inferred(e):
                continue
            elif m == DEP and gated_hi:
                w = 0 if zero_owners is not None and dst[e] >> 1 in zero_owners else whi[e]
            else:
                w = 0
            val = F[src[e]] + w
            if val > best:
                best = val
        F[v] = best
        if best > best_all:
            best_all = best
    return F, best_all


def _backward(g: EventGraph, gap_as_work: bool = True, gated_hi: bool = False) -> array.array:
    B = array.array("q", bytes(16 * g.n))
    out_ptr, out_idx, dst, enabled, mode, d, whi = g.out_ptr, g.out_idx, g.dst, g.enabled, g.mode, g.d, g.whi
    for u in reversed(g.order):
        best = 0
        for k in range(out_ptr[u], out_ptr[u + 1]):
            e = out_idx[k]
            if enabled[e]:
                val = B[dst[e]] + _w(mode[e], d[e], gap_as_work, whi[e], gated_hi)
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
    blame: object = None                     # cp_attribution.Blame
    variants: dict[str, dict] = field(default_factory=dict)
    variant_spans: dict[str, set] = field(default_factory=dict)   # variant -> spans on its observed path
    stable_spans: int = 0
    variant_only_spans: int = 0
    variant_only_ns: int = 0
    graph_evidence: dict = field(default_factory=dict)
    # time in blocking calls no recorded dependency gates ("unknown" mode)
    unmatched_by_category: dict[str, int] = field(default_factory=dict)
    modeled_gated_tail_ns: int = 0
    modeled_range_ns: tuple[int, int] = (0, 0)
    # the observed path's extent when it starts and ends on different clocks
    covered_range_ns: tuple[int, int] = (0, 0)

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
        if m in (BUSY, WAIT, UNKNOWN):
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
        if m in (BUSY, WAIT, UNKNOWN):
            credited[intro(g.owner[e], kind, conf)] += ns
            intro(g.dst[e] >> 1, kind, conf)
        else:
            gaps[intro(g.dst[e] >> 1, kind, conf)] += ns
    return order, credited, gaps, kinds[1:], confs[1:]


def observed_path(trace: "Trace", graph, *, keep_skeleton: bool = False) -> tuple[EventGraph, Walk]:
    g = build_event_graph(trace, graph, keep_skeleton=keep_skeleton)
    return g, walk(g)


def analyze_model(trace: "Trace", graph, *, sensitivity: int | None = None,
                  sensitivity_node_limit: int = 4_000_000, keep_skeleton: bool = False) -> ModelResult:
    """The observed path, its composition and evidence, the modeled longest
    path, criticality, alternatives and (top `sensitivity` candidates)
    counterfactual reductions. sensitivity=None: 5 when the graph has at
    most `sensitivity_node_limit` event nodes, else none (each candidate is
    a full longest-path pass); 0: none."""
    g, w = observed_path(trace, graph, keep_skeleton=keep_skeleton)
    work: dict[str, int] = defaultdict(int)
    in_call: dict[str, int] = defaultdict(int)
    dep_kind: dict[str, int] = defaultdict(int)
    dep_src: dict[str, int] = defaultdict(int)
    tier: dict[str, int] = defaultdict(int)
    cls: dict[str, int] = defaultdict(int)
    prov: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    unmatched: dict[str, int] = defaultdict(int)
    unrecorded = idle = 0
    for e, ns in zip(w.steps, w.cover):
        m = g.mode[e]
        if m == BUSY:
            work[g.cat[g.owner[e]]] += ns
        elif m == WAIT:
            in_call[g.cat[g.owner[e]]] += ns
        elif m == UNKNOWN:
            unmatched[g.cat[g.owner[e]]] += ns
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
        entering_kinds=kinds, entering_confs=confs, unmatched_by_category=dict(unmatched),
    )
    res.graph_evidence = _graph_evidence(g)
    res.covered_range_ns = (w.length, w.length)
    if g.clock is not None and w.first >= 0:
        a, b = g.domain[w.sink >> 1], g.domain[w.first >> 1]
        if a != b:
            lo, hi = g.clock.offset_range(a, b)
            res.covered_range_ns = (max(0, int(w.length + lo)) if lo != float("-inf") else 0,
                                    int(w.length + hi) if hi != float("inf") else -1)
    if g.order:
        F, L = longest(g)
        res.modeled_ns = L
        res.modeled_unrecorded_idle_ns = longest(g, gap_as_work=False)[1]
        res.modeled_direct_only_ns = longest(g, direct_only=True)[1]
        res.modeled_gated_tail_ns = longest(g, gated_hi=True)[1]
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
        if 2 * g.n <= sensitivity_node_limit:
            from .cp_attribution import blame
            res.blame = blame(g)
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
            "clock_contradicted": g.clock_contradicted,
            "clocks": g.clock.to_dict() if g.clock is not None else None,
            "improper_nesting": g.improper_nesting, "unmapped": dict(g.unmapped)}


def _alternatives(res: ModelResult, g: EventGraph, w: Walk) -> None:
    """Every VARIANTS entry: its observed walk and its modeled longest path,
    with the same parameters."""
    variants = {}
    modeled = {}
    for name, (gaps, direct, gated) in VARIANTS.items():
        variants[name] = w if name == "default" else walk(g, gap_as_work=gaps, direct_only=direct,
                                                           gated_hi=gated)
        modeled[name] = longest(g, gap_as_work=gaps, direct_only=direct, gated_hi=gated)[1] if g.order else 0
    sets = {k: v.spans(g) for k, v in variants.items()}
    res.variant_spans = sets
    every = set.intersection(*sets.values())
    some = set.union(*sets.values())
    res.stable_spans = len(every)
    res.variant_only_spans = len(some - every)
    owned: dict[int, int] = defaultdict(int)
    for e, ns in zip(w.steps, w.cover):
        if g.mode[e] in (BUSY, WAIT, UNKNOWN):
            owned[g.owner[e]] += ns
    res.variant_only_ns = sum(ns for j, ns in owned.items() if j not in every)
    for name, v in variants.items():
        gaps, direct, gated = VARIANTS[name]
        res.variants[name] = {"covered_ns": v.length, "spans": len(sets[name]),
                              "first_ns": g.t[v.first] if v.first >= 0 else None,
                              "same_spans_as_default": sets[name] == sets["default"],
                              "modeled_ns": modeled[name],
                              "gaps_as_work": gaps, "direct_evidence_only": direct, "gated_tail_as_work": gated}
    vals = list(modeled.values()) + ([res.modeled_ns] if g.order else [])
    res.modeled_range_ns = (min(vals), max(vals)) if vals else (0, 0)


def _sensitivity(res: ModelResult, g: EventGraph, L: int, k: int) -> None:
    """Counterfactuals: the modeled longest path with the busy time of all
    spans of one (category, name), or of one span, set to zero. Candidates:
    the top k by busy time on longest paths (single spans on every longest
    path first); one longest-path pass each."""
    crit = res.crit
    if k <= 0 or crit is None or not crit.work_on_some:
        return
    from .cp_attribution import replay, residuals, run_end
    resid = residuals(g)
    T = run_end(g)

    def replayed(zero: set[int]) -> int:
        return int(round(T - replay(g, {j: 0.0 for j in zero}, resid)[1]))
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
                                "modeled_reduction_ns": L - L2, "replay_reduction_ns": replayed(members[key])})
    singles = sorted(crit.work_on_some.items(), key=lambda kv: (kv[0] not in crit.on_all, -kv[1]))[:k]
    for j, ns in singles:
        L2 = longest(g, zero_owners={j})[1]
        res.sensitivity.append({"scope": "span", "category": g.cat[j], "name": g.name(j), "span": j,
                                "start_ns": g.t[2 * j], "work_on_longest_ns": ns,
                                "on_every_longest_path": j in crit.on_all, "slack_ns": crit.slack.get(j, 0),
                                "modeled_reduction_ns": L - L2, "replay_reduction_ns": replayed({j})})
