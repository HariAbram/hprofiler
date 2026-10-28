"""
Clock domains and timestamp bounds for the event-point model (cp_model.py).

Timestamps come from several clocks: each node's host clock, a GPU's clock
mapped onto the host by CUPTI / ROCprofiler-SDK, the OpenCL device clock the
hook maps with one offset taken at the first completion callback, the
reference event an asynchronously timed NCCL operation is measured from.
The events of one clock domain share one unknown offset delta_D from the
reference (node 0's host clock, delta = 0), bounded by what is known about
the mapping: |delta_D| <= b_D.

  domain                          b_D
  host of node 0                  0 (the reference)
  host of node k (merge-nodes)    the merge plan's bound for the node
  device (CUPTI / ROCprofiler)    max(the clock tolerance, any drift
                                  correction applied to its spans)
  OpenCL device                   unbounded: the hook's single offset can be
                                  late by the first callback's latency
  NCCL asynchronous timing        the reference event's uncertainty (ref_err=)

A dependency from event u to event v says u happened no later than v:
t(u) + delta_D(u) <= t(v) + delta_D(v), i.e. delta_D(v) - delta_D(u) >=
t(u) - t(v). Within one domain no offset can help: the edge holds or the
timestamps contradict it. Across domains these are difference constraints
-- a simple temporal network over the domains' offsets, solved with
Bellman-Ford: feasible iff no negative cycle, and then each offset has a
tightest interval [lo, hi]. When the constraints are infeasible, the
constraint on a negative cycle with the weakest evidence is relaxed to what
the rest allows, and the edges that needed more are contradicted (counted,
kept by the evidence model, never edges of the event graph).

Uses:
  * contradicted edges beyond what the domains' bounds allow;
  * for a domain with no a-priori bound and no correction of its own
    (OpenCL) whose recorded times the dependencies contradict: the
    smallest shift that satisfies them (applied before the graph is built,
    reported, tagged nowhere);
  * the interval of each event's time relative to another domain, for the
    walk's "undecidable" steps (two releases whose order the clocks cannot
    tell) and the bounds of times that cross domains.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


from ..core import gpu_activity as ga
from .activity_buckets import is_device_timed

INF = math.inf
REFERENCE = ("host", "0")


def domain_of(span, node_bounds: dict[str, float]) -> tuple[tuple, float]:
    """(domain key, bound in ns) of a span's timestamps."""
    tags = span.tags
    node = tags.get("node", "0")
    if is_device_timed(span):
        timing = tags.get("timing")
        cat = span.category.value
        if timing == "async":                                     # NCCL, HPROFILER_NCCL_ASYNC=1
            return ("nccl_async", node, span.pid), float(tags.get("ref_err", 0) or 0)
        if cat in ("opencl", "memory") and tags.get("rt") not in ga.RUNTIMES and (
                tags.get("side") == "gpu" or span.name.endswith("_gpu")):
            return ("opencl", node, span.pid), INF
        if ga.is_device_span(span) and ga.timing_source(span) == ga.TIMING_DEVICE:
            corr = float(tags.get("clock_corr_ns", 0) or 0)
            return ("device", node, span.pid, tags.get("rt"), tags.get("dev") or tags.get("ctx") or ""), \
                max(float(ga.CLOCK_TOLERANCE_NS), corr)
        # proxies (host submission time + event duration), host-timed work
        # on a device lane (CUDA host functions), pre-split spans: host clock
    return ("host", node), (0.0 if node == "0" else node_bounds.get(node, INF))


def node_bounds_of(metadata) -> dict[str, float]:
    """node index -> offset bound of a merged trace's nodes (merge-nodes'
    plan); a node without a known bound is unbounded."""
    out: dict[str, float] = {}
    clock = getattr(metadata, "clock", None) or {}
    nodes = (clock.get("merge") or {}).get("nodes") if isinstance(clock, dict) else None
    for i, n in enumerate(nodes or []):
        b = n.get("bound_ns")
        out[str(i)] = INF if b is None else float(b)
    return out


# shiftable domains: no a-priori bound and no correction step of their own
SHIFTABLE = frozenset({"opencl"})


@dataclass
class ClockAnalysis:
    keys: list[tuple]                      # domain id -> key
    bounds: list[float]                    # domain id -> |delta| bound
    lo: list[float]                        # tightest offset interval (relative to the reference)
    hi: list[float]
    shift: list[int]                       # applied to the domain's timestamps (shiftable ones)
    feasible_as_recorded: bool = True
    constraints: int = 0                   # cross-domain dependency constraints
    relaxed: list[dict] = field(default_factory=list)   # what was given up to restore feasibility
    contradicted: set[int] = field(default_factory=set)  # constraint indices ruled out

    def offset_range(self, a: int, b: int) -> tuple[float, float]:
        """Interval of delta_a - delta_b."""
        return self.lo[a] - self.hi[b], self.hi[a] - self.lo[b]

    def to_dict(self) -> dict:
        def num(x):
            return None if x in (INF, -INF) else int(round(x))
        return {"domains": [{"domain": "/".join(str(p) for p in k), "bound_ns": num(b), "lo_ns": num(lo),
                             "hi_ns": num(hi), "shift_ns": s}
                            for k, b, lo, hi, s in zip(self.keys, self.bounds, self.lo, self.hi, self.shift)],
                "feasible_as_recorded": self.feasible_as_recorded, "constraints": self.constraints,
                "relaxed": self.relaxed, "contradicted_edges": len(self.contradicted)}


def _shortest(n: int, edges: list[tuple[int, int, float]], source: int | None):
    """x_v <= x_u + w for each (u, v, w). From `source` (None: a virtual
    source at distance 0 to every node): (distances, None), or (None, the
    edge indices of one negative cycle)."""
    d = [INF] * n if source is not None else [0.0] * n
    if source is not None:
        d[source] = 0.0
    pred = [-1] * n
    x = -1
    for _ in range(n + 1):
        x = -1
        for k, (u, v, w) in enumerate(edges):
            if d[u] + w < d[v] - 0.5:            # half a nanosecond: integer timestamps
                d[v] = d[u] + w
                pred[v] = k
                x = v
        if x < 0:
            return d, None
    for _ in range(n):                           # step back onto the cycle itself
        x = edges[pred[x]][0]
    cyc, v = [], x
    while True:
        k = pred[v]
        cyc.append(k)
        v = edges[k][0]
        if v == x or len(cyc) > n:
            break
    return None, cyc


@dataclass
class _Pair:
    a: int                      # the source events' domain
    b: int                      # the destination events' domain
    needs: list[tuple[float, int]]   # (t(u) - t(v), constraint id), largest first
    strength: int               # 0 inferred, 1 direct: the weakest is relaxed first

    @property
    def need(self) -> float:
        return self.needs[0][0]


def solve(keys: list[tuple], bounds: list[float], pairs: list[_Pair]) -> ClockAnalysis:
    """Offsets of every domain: delta_b - delta_a >= need for each pair,
    |delta_i| <= bounds[i], delta_ref = 0. While infeasible, the weakest
    pair on a negative cycle is relaxed by the cycle's deficit; its
    constraints above the new value are contradicted."""
    n = len(keys)
    ref = keys.index(REFERENCE) if REFERENCE in keys else 0
    out = ClockAnalysis(keys=keys, bounds=bounds, lo=[0.0] * n, hi=[0.0] * n, shift=[0] * n,
                        constraints=sum(len(p.needs) for p in pairs))
    pairs = [p for p in pairs if p.needs]

    def system() -> tuple[list[tuple[int, int, float]], int]:
        edges = [(p.b, p.a, -p.need) for p in pairs]        # delta_a <= delta_b - need
        npairs = len(edges)
        for i, bd in enumerate(bounds):
            if i != ref and bd != INF:
                edges.append((ref, i, bd))                  # delta_i <= b
                edges.append((i, ref, bd))                  # 0 <= delta_i + b
        return edges, npairs

    for _round in range(sum(len(p.needs) for p in pairs) + 1):
        edges, npairs = system()
        dist, cyc = _shortest(n, edges, None)
        if dist is not None:
            break
        out.feasible_as_recorded = False
        on = [k for k in cyc if k < npairs]
        if not on:                                          # bounds alone never conflict
            break
        deficit = -sum(edges[k][2] for k in cyc)            # > 0: how much the cycle lacks
        k = min(on, key=lambda k: (pairs[k].strength, -pairs[k].need))
        p = pairs[k]
        allowed = p.need - deficit
        gone = [cid for need, cid in p.needs if need > allowed]
        out.contradicted.update(gone)
        out.relaxed.append({"from": "/".join(map(str, keys[p.a])), "to": "/".join(map(str, keys[p.b])),
                            "need_ns": int(round(p.need)), "allowed_ns": int(math.floor(allowed)),
                            "edges_contradicted": len(gone),
                            "strength": "direct" if p.strength else "inferred"})
        p.needs = [(need, cid) for need, cid in p.needs if need <= allowed]
        if not p.needs:
            pairs.pop(k)
    edges, _ = system()
    hi, _ = _shortest(n, edges, ref)
    rev = [(v, u, w) for u, v, w in edges]
    to_ref, _ = _shortest(n, rev, ref)
    if hi is None or to_ref is None:                        # unreachable: give up on bounds
        out.lo = [-b for b in bounds]
        out.hi = list(bounds)
        return out
    out.hi = [float(x) for x in hi]
    out.lo = [-float(x) for x in to_ref]
    for i in range(n):
        if keys[i][0] in SHIFTABLE and not (out.lo[i] <= 0.0 <= out.hi[i]):
            # the smallest shift into the feasible interval
            out.shift[i] = int(math.ceil(out.lo[i])) if out.lo[i] > 0 else int(math.floor(out.hi[i]))
    return out


def analyze_clocks(node_domain, t, constraints, keys: list[tuple], bounds: list[float]) -> ClockAnalysis:
    """constraints: (constraint id, source node, destination node, strength)
    of dependency edges; node_domain(v) -> domain id; t: node timestamps."""
    by_pair: dict[tuple[int, int], _Pair] = {}
    for cid, u, v, strength in constraints:
        a, b = node_domain(u), node_domain(v)
        if a == b:
            continue
        p = by_pair.get((a, b))
        if p is None:
            p = by_pair[(a, b)] = _Pair(a, b, [], 1)
        p.needs.append((float(t[u] - t[v]), cid))
        p.strength = min(p.strength, strength)
    for p in by_pair.values():
        p.needs.sort(reverse=True)
    return solve(list(keys), list(bounds), list(by_pair.values()))
