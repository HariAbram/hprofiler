"""
Multi-node trace alignment and merging.

hprofiler's collector is a local AF_UNIX socket (see criticalpath.py's
module docstring for why that makes today's traces inherently single-node
regardless of clock synchronization) -- a multi-node job naturally produces
one independent trace file per node, each on that node's own
CLOCK_MONOTONIC timeline with its own arbitrary epoch. This module aligns
several such per-node traces onto one common timeline and combines them
into a single Trace, so criticalpath.py / pop_efficiency.py can operate
across node boundaries instead of just within one.

── Clock offset estimation ──────────────────────────────────────────────
estimate_clock_offset() implements Cristian's algorithm (the classic
round-trip clock-offset estimate, the same technique NTP itself is built
on): given four timestamps from a single round-trip ping/reply exchange
between a node and a reference node, it returns both a point estimate AND
an explicit error bound (half the round-trip time, the standard bound
under the assumption of symmetric network latency) -- never claiming more
precision than the measurement actually supports.

offset_from_counters() reads the estimate of hooks/mpi_hook/mpi_hook.c's
opt-in HPROFILER_CLOCK_SYNC exchange back out of a trace (the counters older
hooks wrote); clock_samples() reads the richer "clock_sync" instants the
hook now emits at MPI_Init and MPI_Finalize (minimum-delay sample of R
rounds, its bound, the local time it was taken at, and whether the rank
shares rank 0's clock domain).

plan_offsets() decides each node's offset before merging (see its
docstring): the node holding rank 0 is the reference; a node in the same
clock domain (kernel boot_id and time namespace: the same CLOCK_MONOTONIC)
gets exactly 0; otherwise the estimate -- offset plus linear drift when
both measurements exist -- tightened with the matched MPI send/receive
pairs (a send cannot start after its matched receive ended), choosing
offsets inside [estimate ± bound] that satisfy them, as close to the
estimate as possible.

── Merging ───────────────────────────────────────────────────────────────
merge_traces() combines N per-node traces into one, applying each node's
clock offset to its timestamps and remapping `pid` into a per-node-unique
namespace (pid 1234 on node A and pid 1234 on node B are unrelated
processes -- not remapping would let unrelated nodes' events get grouped
together by every (pid, tid)-keyed piece of existing analysis code). `tid`
is deliberately left unchanged: everywhere in this codebase that groups by
thread already groups by the (pid, tid) pair together, so a per-node-
unique pid alone is sufficient to keep those tuples unique post-merge.
MPI `rank=`/`peer=` tags are also left unchanged -- MPI_COMM_WORLD ranks
are already globally unique across an entire job regardless of which
physical node a rank happens to run on, so criticalpath.py's cross-rank
P2P/collective matching (built on those tags, not pid/tid) works
transparently across a merge with no further change needed.

validate_causality() is a sanity check for the result: after merging, a
matched MPI send can never causally complete after its receive already
finished. A violation means either a bad offset estimate or a genuine
clock anomaly -- surfaced explicitly rather than silently trusted.

Selective aggregation: merge_traces() takes a plain list, so a caller
merges only the NodeTrace entries they want included (e.g. a subset of
ranks/nodes) simply by not including the others -- no separate filtering
API needed beyond ordinary list slicing/filtering before the call.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.trace import Trace

# mpi_hook.c's HPROFILER_CLOCK_SYNC counters carry these exact names.
_COUNTER_OFFSET = "clock_offset_vs_rank0_ns"
_COUNTER_ERROR_BOUND = "clock_offset_error_bound_ns"
_COUNTER_ROUND_TRIP = "clock_sync_round_trip_ns"

# pid namespace stride for merge_traces -- large enough that no real
# process's pid (max ~4*10^6 on a default-configured Linux system, per
# /proc/sys/kernel/pid_max) could collide with the next node's offset.
PID_NAMESPACE_STRIDE = 10_000_000


@dataclass
class ClockOffsetEstimate:
    offset_ns: int
    error_bound_ns: int
    round_trip_ns: int


def estimate_clock_offset(t1: int, t2: int, t3: int, t4: int) -> ClockOffsetEstimate:
    """Cristian's algorithm. t1 = local send time, t2 = remote receive
    time, t3 = remote reply time, t4 = local reply-received time -- all in
    their own clock's units (e.g. ns since CLOCK_MONOTONIC boot on their
    respective machine, not comparable to each other before this
    computation). Returns the estimated offset (remote clock minus local
    clock, at the same real instant -- add it to a local timestamp to
    express it in the remote clock's frame) and its error bound (half the
    round trip; assumes symmetric network latency, which is a real
    approximation, not an exact guarantee -- hence returning a bound, not
    pretending the point estimate alone is precise)."""
    round_trip = t4 - t1
    offset = (t2 + t3) // 2 - (t1 + round_trip // 2)
    return ClockOffsetEstimate(offset_ns=offset, error_bound_ns=round_trip // 2,
                               round_trip_ns=round_trip)


def offset_from_counters(trace: "Trace") -> ClockOffsetEstimate | None:
    """Reads back the HPROFILER_CLOCK_SYNC counters mpi_hook.c emits (see
    module docstring). Returns None if absent -- either HPROFILER_CLOCK_SYNC
    wasn't set for the run that produced this trace, or this trace IS the
    reference (rank 0), which never emits these (its offset is 0 by
    construction, not measured)."""
    offset = error_bound = round_trip = None
    for c in trace.counters:
        if c.name == _COUNTER_OFFSET:
            offset = int(c.value)
        elif c.name == _COUNTER_ERROR_BOUND:
            error_bound = int(c.value)
        elif c.name == _COUNTER_ROUND_TRIP:
            round_trip = int(c.value)
    if offset is None or error_bound is None:
        return None
    return ClockOffsetEstimate(offset_ns=offset, error_bound_ns=error_bound,
                               round_trip_ns=round_trip or 0)


@dataclass
class OffsetModel:
    """t_reference = t + offset_at(t): offset_ns at local time t0_ns, drifting
    by drift (ns per ns, i.e. 1e-6 = 1 ppm), plus a constant correction."""
    offset_ns: float = 0.0
    t0_ns: float = 0.0
    drift: float = 0.0
    correction_ns: float = 0.0
    t1_ns: float | None = None   # drift is measured over [t0, t1]: not extrapolated beyond

    def offset_at(self, t: int) -> int:
        if self.drift:
            t = max(self.t0_ns, t if self.t1_ns is None else min(t, self.t1_ns))
            return int(round(self.offset_ns + self.drift * (t - self.t0_ns) + self.correction_ns))
        return int(round(self.offset_ns + self.correction_ns))


@dataclass
class NodeTrace:
    trace: "Trace"
    offset_ns: int = 0
    error_bound_ns: int = 0
    label: str = ""   # e.g. hostname -- diagnostics only, not load-bearing
    model: "OffsetModel | None" = None   # time-dependent offset (overrides offset_ns)


def merge_traces(nodes: list[NodeTrace], into: "Trace | None" = None) -> tuple["Trace", list[str]]:
    """Combines several per-node traces into one Trace on a common
    timeline -- see module docstring for the offset/pid-remapping/rank-
    tag-preservation rules. Returns (merged_trace, warnings); a node with
    no distinguishing offset data merges at an uncorrected 0 offset and is
    called out explicitly in warnings (unless it's node 0, the assumed
    reference, for which offset=0 is the expected, correct value, not a
    missing measurement)."""
    from ..core.trace import Trace, TraceMetadata

    # Streams every node's events into `into` (e.g. a disk-backed Trace),
    # one event at a time -- no node is ever fully materialized.
    merged = into if into is not None else Trace(TraceMetadata(command="(multi-node merge)"))
    if into is not None:
        merged.metadata.command = merged.metadata.command or "(multi-node merge)"
    warnings: list[str] = []

    for node_idx, node in enumerate(nodes):
        pid_offset = node_idx * PID_NAMESPACE_STRIDE
        if node_idx != 0 and node.offset_ns == 0 and node.error_bound_ns == 0 and node.model is None:
            warnings.append(
                f"node {node_idx} ({node.label or 'unlabeled'}): no clock-offset "
                f"data provided -- merged at an uncorrected 0 offset; timestamps "
                f"are not meaningfully comparable to other nodes unless this node "
                f"genuinely is the reference clock"
            )
        m = node.model
        if m is None or (m.drift == 0.0):
            const = node.offset_ns if m is None else m.offset_at(0)

            def at(t: int, _c=const) -> int:
                return t + _c
        else:
            def at(t: int, _m=m) -> int:
                return t + _m.offset_at(t)
        for s in node.trace.iter_spans():
            start = at(s.start_ns)
            merged.add(replace(s, pid=s.pid + pid_offset, start_ns=start,
                               duration_ns=max(0, at(s.start_ns + s.duration_ns) - start),
                               tags={**s.tags, "node": str(node_idx)}))
        for i in node.trace.iter_instants():
            merged.add(replace(i, pid=i.pid + pid_offset,
                               timestamp_ns=at(i.timestamp_ns),
                               tags={**i.tags, "node": str(node_idx)}))
        for c in node.trace.iter_counters():
            merged.add(replace(c, pid=c.pid + pid_offset,
                               timestamp_ns=at(c.timestamp_ns)))

    merged.store.flush()
    ext = merged.store.event_extent()
    if ext is not None:
        merged.metadata.start_time_ns, merged.metadata.end_time_ns = ext
    return merged, warnings


# ── Offset planning: clock domains, drift, message-order tightening ────────

@dataclass
class ClockSample:
    """One clock_sync measurement (mpi_hook.c): rank 0's clock minus this
    rank's, its bound, and the local time it was taken at."""
    phase: str
    rank: int
    offset_ns: int
    bound_ns: int
    at_ns: int | None
    same_clock: bool


@dataclass
class NodeClock:
    """What a trace says about its clock."""
    boot_id: str = ""            # kernel boot_id: one CLOCK_MONOTONIC per boot
    timens_ns: int = 0           # time-namespace CLOCK_MONOTONIC offset
    has_rank0: bool = False      # holds the reference rank (MPI_COMM_WORLD rank 0)
    ranks: set = None            # MPI ranks seen
    samples: dict = None         # phase -> best ClockSample (smallest bound)
    counters: "ClockOffsetEstimate | None" = None   # older hooks' counters


def node_clock(trace: "Trace") -> NodeClock:
    """The clock facts of one trace: metadata.clock (written by the Runner),
    the hook's clock_sync instants, and the older counters."""
    md = getattr(trace.metadata, "clock", None) or {}
    nc = NodeClock(boot_id=str(md.get("boot_id") or ""), timens_ns=int(md.get("timens_monotonic_ns") or 0),
                   ranks=set(), samples={})
    for i in trace.iter_instants():
        if i.name != "clock_sync" or i.tags.get("type") != "clock_sync":
            continue
        t = i.tags
        try:
            rank = int(t.get("rank", -1))
        except ValueError:
            continue
        nc.ranks.add(rank)
        if not nc.boot_id and t.get("boot_id") not in (None, "", "unknown"):
            nc.boot_id = str(t["boot_id"])
            nc.timens_ns = int(t.get("timens", 0) or 0)
        if t.get("reference") == "1" or rank == 0:
            nc.has_rank0 = True
            continue
        try:
            smp = ClockSample(phase=str(t.get("phase", "init")), rank=rank, offset_ns=int(t["offset"]),
                              bound_ns=int(t["bound"]), at_ns=int(t["at"]) if t.get("at") else None,
                              same_clock=t.get("same_clock") == "1")
        except (KeyError, ValueError):
            continue
        best = nc.samples.get(smp.phase)
        if best is None or smp.bound_ns < best.bound_ns:
            nc.samples[smp.phase] = smp
    nc.counters = offset_from_counters(trace)
    return nc


@dataclass
class NodePlan:
    """How one node's offset was decided (merge-nodes prints and stores it)."""
    label: str
    method: str                  # reference | same-clock | explicit | estimate | estimate+drift
                                 # | estimate (counters) | message order | none
    estimate_ns: int = 0         # offset at t0 before tightening
    bound_ns: float = 0.0        # +- around the estimate (inf: unknown)
    drift_ppm: float = 0.0
    correction_ns: int = 0       # tightening applied
    constraints: int = 0         # matched send/receive pairs with another node
    model: OffsetModel = None

    @property
    def offset_ns(self) -> int:
        return self.model.offset_at(int(self.model.t0_ns)) if self.model else 0

    def to_dict(self) -> dict:
        return {"label": self.label, "method": self.method, "estimate_ns": self.estimate_ns,
                "bound_ns": None if self.bound_ns == float("inf") else int(self.bound_ns),
                "drift_ppm": round(self.drift_ppm, 4), "correction_ns": self.correction_ns,
                "offset_ns": self.offset_ns, "constraints": self.constraints}


@dataclass
class MergePlan:
    reference: int
    nodes: list
    tightening: str = ""         # what the message-order step did
    lam: float = 0.0             # fraction of each bound the corrections needed

    def to_dict(self) -> dict:
        return {"reference": self.reference, "tightening": self.tightening, "lambda": round(self.lam, 4),
                "nodes": [n.to_dict() for n in self.nodes]}


def _bellman_ford(n: int, src, dst, w, source: int):
    """Shortest distances from `source` (x_dst <= x_src + w edges); None on a
    negative cycle (the difference constraints are infeasible)."""
    import numpy as np
    d = np.full(n, np.inf)
    d[source] = 0.0
    for _ in range(n + 1):
        nd = d.copy()
        np.minimum.at(nd, dst, d[src] + w)
        if not np.any(nd < d - 0.5):           # converged to within half a nanosecond
            return nd
        d = nd
    return None


def _solve_corrections(n: int, ref: int, pairs: dict, bounds: list[float]):
    """Corrections c (c[ref] = 0, |c_i| <= lam*bounds[i]) with c_A - c_B <= w
    for every pair constraint {(B, A): w}, for the smallest lam in [0, 1].
    Returns (c, lam) or (None, None) when no lam <= 1 is feasible. The
    feasible set is convex, so the midpoint of its smallest and largest
    points is feasible too: that is the answer."""
    import numpy as np
    if all(w >= 0 for w in pairs.values()):
        return np.zeros(n), 0.0
    psrc = np.array([b for (b, a) in pairs], dtype=np.int64)
    pdst = np.array([a for (b, a) in pairs], dtype=np.int64)
    pw = np.array(list(pairs.values()), dtype=float)
    finite = [i for i in range(n) if i != ref and np.isfinite(bounds[i])]

    def region(lam):
        bw = [lam * bounds[i] for i in finite]
        src = np.concatenate([psrc, np.full(len(finite), ref), np.array(finite, dtype=np.int64)])
        dst = np.concatenate([pdst, np.array(finite, dtype=np.int64), np.full(len(finite), ref)])
        w = np.concatenate([pw, np.array(bw), np.array(bw)])
        hi = _bellman_ford(n, src, dst, w, ref)
        if hi is None:
            return None
        lo = _bellman_ford(n, dst, src, w, ref)       # reversed edges: -x_i <= dist
        if lo is None:
            return None
        return -lo, hi

    if region(1.0) is None:
        return None, None
    a, b = 0.0, 1.0
    for _ in range(30):
        mid = (a + b) / 2
        if region(mid) is None:
            a = mid
        else:
            b = mid
    lam = min(1.0, b * 1.02 + 1e-9)                    # a little room inside
    lo, hi = region(lam) or region(1.0)
    c = np.where(np.isfinite(lo) & np.isfinite(hi), (lo + hi) / 2,
                 np.where(np.isfinite(lo), lo, np.where(np.isfinite(hi), hi, 0.0)))
    c[ref] = 0.0
    return c, lam


def plan_offsets(traces: list, labels: list[str] | None = None, explicit: list | None = None,
                 ignore_boot_id: bool = False) -> MergePlan:
    """Each node's offset onto the reference clock, before merging.

    * The reference is the trace holding MPI_COMM_WORLD rank 0 (the clock
      sync measures against it), else the first trace.
    * explicit[i] (merge-nodes --offset-ns) is used as given.
    * A node in the reference's clock domain (same kernel boot_id and time
      namespace offset: the same CLOCK_MONOTONIC) gets exactly 0 -- no
      estimate is applied (ignore_boot_id: never assume this).
    * Otherwise the clock_sync estimate: offset plus linear drift when both
      the MPI_Init and the MPI_Finalize measurement exist, or the older
      counters; a node with neither starts from 0 with an unknown bound.
    * Then the matched MPI send/receive pairs between nodes (the critical
      path's own matcher; pairing depends on each side's own order only,
      not on offsets) constrain the offsets: a send cannot start after its
      matched receive ended. If the estimates violate any, the smallest
      corrections within [estimate ± bound] that satisfy all of them are
      applied; a node without an estimate gets an offset from those
      constraints alone. When no offsets within the bounds satisfy them,
      the estimates stay and the residual violations are reported -- an
      offset outside the measured bound is never applied."""
    import math
    from dataclasses import replace as _replace
    from ..analysis import criticalpath as cp

    n = len(traces)
    labels = labels or [f"node{i}" for i in range(n)]
    explicit = list(explicit or []) + [None] * n
    clocks = [node_clock(t) for t in traces]

    # MPI spans of every node through one matcher (pids namespaced exactly as
    # merge_traces does: request ids are per process)
    matcher = cp._P2PMatcher()
    where: list[tuple[int, int, int]] = []          # ordinal -> (node, start, end)
    for k, t in enumerate(traces):
        for s in t.iter_spans(categories=("mpi",)):
            r = s.tags.get("rank")
            if r == "0":
                clocks[k].has_rank0 = True
            if r is not None and r.lstrip("-").isdigit():
                clocks[k].ranks.add(int(r))
            matcher.add(len(where), _replace(s, pid=s.pid + k * PID_NAMESPACE_STRIDE))
            where.append((k, s.start_ns, s.end_ns))
    edges = matcher.edges()

    ref = next((k for k in range(n) if clocks[k].has_rank0), 0)
    rc = clocks[ref]
    plans: list[NodePlan] = []
    for k in range(n):
        c = clocks[k]
        if explicit[k] is not None:
            plans.append(NodePlan(labels[k], "explicit", int(explicit[k]), 0.0,
                                  model=OffsetModel(offset_ns=float(explicit[k]))))
        elif k == ref:
            plans.append(NodePlan(labels[k], "reference", 0, 0.0, model=OffsetModel()))
        elif (not ignore_boot_id and c.boot_id and c.boot_id == rc.boot_id
              and c.timens_ns == rc.timens_ns):
            plans.append(NodePlan(labels[k], "same-clock", 0, 0.0, model=OffsetModel()))
        elif c.samples:
            a, b = c.samples.get("init"), c.samples.get("finalize")
            first = a or b
            if a and b and a.at_ns is not None and b.at_ns is not None and b.at_ns != a.at_ns:
                drift = (b.offset_ns - a.offset_ns) / (b.at_ns - a.at_ns)
                plans.append(NodePlan(labels[k], "estimate+drift", a.offset_ns,
                                      float(max(a.bound_ns, b.bound_ns)), drift_ppm=drift * 1e6,
                                      model=OffsetModel(offset_ns=a.offset_ns, t0_ns=a.at_ns, drift=drift,
                                                        t1_ns=b.at_ns)))
            else:
                plans.append(NodePlan(labels[k], "estimate", first.offset_ns, float(first.bound_ns),
                                      model=OffsetModel(offset_ns=first.offset_ns,
                                                        t0_ns=first.at_ns or 0)))
        elif c.counters is not None:
            plans.append(NodePlan(labels[k], "estimate (counters)", c.counters.offset_ns,
                                  float(c.counters.error_bound_ns),
                                  model=OffsetModel(offset_ns=c.counters.offset_ns)))
        else:
            plans.append(NodePlan(labels[k], "none", 0, math.inf, model=OffsetModel()))

    # message-order constraints on the corrections: c_A - c_B <= w for a send
    # on A matched with a receive completion on B
    pairs: dict[tuple[int, int], float] = {}
    for target, src, _conf in edges:
        a, s_start, _ = where[src]
        b, _, r_end = where[target]
        if a == b:
            continue
        w = (r_end + plans[b].model.offset_at(r_end)) - (s_start + plans[a].model.offset_at(s_start))
        key = (b, a)
        pairs[key] = min(pairs.get(key, math.inf), w)
        plans[a].constraints += 1
        plans[b].constraints += 1
    fixed = {k for k, p in enumerate(plans) if p.method in ("explicit", "reference", "same-clock")}
    if all(p.method == "explicit" for p in plans) or len(fixed) == n:
        return MergePlan(ref, plans, "offsets fixed (explicit / same clock): no tightening")
    if not pairs:
        return MergePlan(ref, plans, "no matched messages between nodes: estimates applied as measured")

    # nodes without an estimate: from the message order alone, the other
    # nodes held at their estimates
    bounds = [0.0 if k in fixed else plans[k].bound_ns for k in range(n)]
    unknown = [k for k in range(n) if math.isinf(bounds[k])]
    if unknown:
        held = [math.inf if k in unknown else 0.0 for k in range(n)]
        cu, _lam = _solve_corrections(n, ref, pairs, held)
        for k in unknown:
            v = 0.0 if cu is None else float(cu[k])
            plans[k].model.correction_ns = v
            plans[k].method = "message order"
            plans[k].correction_ns = int(round(v))
            bounds[k] = 0.0
        for (b_, a_), w in list(pairs.items()):          # re-express with those offsets
            shift = (plans[b_].model.correction_ns if b_ in unknown else 0.0) - \
                    (plans[a_].model.correction_ns if a_ in unknown else 0.0)
            pairs[(b_, a_)] = w + shift

    corr, lam = _solve_corrections(n, ref, pairs, bounds)
    violated = sum(1 for w in pairs.values() if w < 0)
    if corr is None:
        return MergePlan(ref, plans, f"no offsets within the error bounds satisfy the message order "
                                     f"({violated} node pair(s) out of order): estimates applied as "
                                     f"measured", lam=float("nan"))
    for k in range(n):
        if k in unknown or k in fixed:
            continue
        plans[k].model.correction_ns = float(corr[k])
        plans[k].correction_ns = int(round(corr[k]))
    if lam == 0.0:
        note = "estimates already satisfy the message order: no correction"
    else:
        note = (f"corrected within the error bounds to satisfy the message order "
                f"(largest correction {max(abs(p.correction_ns) for p in plans)} ns, "
                f"{lam:.0%} of a bound)")
    return MergePlan(ref, plans, note, lam=lam)


def validate_causality(trace: "Trace") -> list[str]:
    """After merging, a matched MPI send can never causally complete after
    its receive already finished -- checks this holds for every p2p edge
    criticalpath.py's own matching would build (reusing
    build_dependency_graph so this validates exactly the pairing the
    critical-path/blame-attribution engine itself will use, not a separate
    approximation of it). A violation means either a bad clock-offset
    estimate or a genuine anomaly -- surfaced explicitly, not silently
    accepted into a critical-path report that would then misattribute
    blame across the false ordering."""
    from ..analysis import criticalpath as cp

    # Persisted/streamed edges; only the p2p edges' endpoint spans are
    # fetched (one streaming pass over the MPI spans).
    graph = cp.load_or_build_graph(trace)
    edges = graph.edges[graph.edges["kind"] == cp.EDGE_KINDS.index("p2p")]
    if not len(edges):
        return []
    needed = set(edges["dst"].tolist()) | set(edges["src"].tolist())
    ends = graph.endpoints(needed, categories=("mpi",))
    problems: list[str] = []
    for v, u, _kind_code, conf_code in edges.tolist():
        conf = cp.EDGE_CONFS[conf_code]
        send, recv = ends.get(u) or graph.spans[u], ends.get(v) or graph.spans[v]
        if send.start_ns > recv.end_ns:
            problems.append(
                f"causality violation ({conf} confidence match): send on "
                f"pid={send.pid} starts at {send.start_ns}ns, AFTER its "
                f"matched receive on pid={recv.pid} already ended at "
                f"{recv.end_ns}ns -- clock offset estimate is likely wrong "
                f"for one of these nodes"
            )
    return problems
