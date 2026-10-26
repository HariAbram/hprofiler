"""
Analysis overlays for the Timeline -- the critical path, dependency edges
and their evidence, waits and their causes -- read from the existing
analyses (criticalpath.analyze / cp_model, the persisted dependency edges,
the span-level DP), never recomputed with other rules: this module only
indexes their results by lane and time so a viewport can ask for what is
visible. Qt-free; built once per trace on a background thread
(TimelineModel.startAnalysis).

Everything is keyed by the store's event id (eid) and the Timeline's lane
index; times are absolute ns on the trace's clock.
"""
from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ..analysis import cp_model as cm
from ..analysis import criticalpath as cp
from ..analysis import dashboard as dash

# Overlay families (the dependency filter's groups).
FAMILIES = ("critical", "mpi_p2p", "mpi_collective", "openmp", "submission", "gpu_sync", "nccl", "other")
FAMILY_LABELS = {
    "critical": "Critical-path hops", "mpi_p2p": "MPI point-to-point", "mpi_collective": "MPI collectives",
    "openmp": "OpenMP barriers / fork-join", "submission": "CPU → GPU submission",
    "gpu_sync": "GPU synchronization / stream order", "nccl": "NCCL collectives", "other": "Other explicit ids",
}


def edge_family(kind: str, dst_cat: str) -> str:
    if kind == "p2p":
        return "mpi_p2p"
    if kind == "arrival":
        if dst_cat == "mpi":
            return "mpi_collective"
        if dst_cat == "nccl":
            return "nccl"
        return "openmp"
    if kind in ("fork", "join"):
        return "openmp"
    if kind == "launch":
        return "submission"
    if kind in ("device_wait", "device_sync", "stream_order", "event_wait", "sequential"):
        return "gpu_sync"
    return "other"


@dataclass
class Edge:
    id: int
    kind: str
    family: str
    runtime: str
    tier: str
    provenance: str
    cls: str
    src_lane: int
    dst_lane: int
    src_eid: int
    dst_eid: int
    t_src: int               # the source event (start or end of the source span)
    t_dst: int               # the event it gates
    src_end: str             # "start" / "end": which event of the source span
    dst_end: str
    on_path: bool = False


@dataclass
class PathStep:
    lane: int
    t0: int
    t1: int
    mode: str                # busy / wait / gap / idle / dependency
    kind: str
    tier: str
    eid: int                 # owner (busy/wait) or the span reached
    src_lane: int = -1       # dependency steps
    src_eid: int = -1
    edge: int = -1


@dataclass
class Transition:
    t: int
    from_lane: int
    to_lane: int
    switch: list[str]        # what changed: process / rank / thread / runtime / device / stream
    kind: str                # the dependency kind of the hop
    tier: str


@dataclass
class WaitCause:
    lane: int
    eid: int
    name: str
    t0: int                  # the gated call's start
    t1: int                  # when its last dependency's source event happened (clipped to the call)
    call_end: int
    edge: int                # the binding dependency


@dataclass
class AnalysisData:
    ok: bool = False
    error: str = ""
    seconds: float = 0.0
    report: Any = None
    summary: dict = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)
    edge_t: np.ndarray = None            # sorted min(t_src, t_dst) of edges (window queries)
    edge_order: np.ndarray = None
    edges_of_eid: dict[int, list[int]] = field(default_factory=dict)
    steps: list[PathStep] = field(default_factory=list)
    steps_of_lane: dict[int, list[int]] = field(default_factory=dict)
    path_eids: set[int] = field(default_factory=set)
    path_lanes: set[int] = field(default_factory=set)
    credit_of_eid: dict[int, int] = field(default_factory=dict)
    on_all: set[int] = field(default_factory=set)
    on_some: set[int] = field(default_factory=set)
    slack_of_eid: dict[int, int] = field(default_factory=dict)
    sensitivity: list[dict] = field(default_factory=list)
    transitions: list[Transition] = field(default_factory=list)
    waits: list[WaitCause] = field(default_factory=list)
    waits_of_lane: dict[int, list[int]] = field(default_factory=dict)
    legacy: dict = field(default_factory=dict)
    lane_of_ord: np.ndarray = None
    eid_of_ord: np.ndarray = None

    long_edges: list[int] = field(default_factory=list)   # longer than short_max_ns
    short_max_ns: float = 0.0

    def edges_in(self, a: float, b: float) -> list[int]:
        """Edges with an endpoint in [a, b) or spanning it: short edges by
        binary search on their earlier endpoint, long ones scanned."""
        if self.edge_t is None or not len(self.edges):
            return []
        lo = int(np.searchsorted(self.edge_t, a - self.short_max_ns, side="left"))
        hi = int(np.searchsorted(self.edge_t, b, side="left"))
        out = []
        for k in self.edge_order[lo:hi].tolist():
            e = self.edges[k]
            if max(e.t_src, e.t_dst) >= a and k not in self._long_set:
                out.append(k)
        for k in self.long_edges:
            e = self.edges[k]
            if min(e.t_src, e.t_dst) < b and max(e.t_src, e.t_dst) >= a:
                out.append(k)
        return out

    @property
    def _long_set(self) -> set:
        s = getattr(self, "_long_set_cache", None)
        if s is None:
            s = self._long_set_cache = set(self.long_edges)
        return s


def _lane_meta_switch(meta: Callable[[int], dict], a: int, b: int) -> list[str]:
    ma, mb = meta(a), meta(b)
    out = []
    if ma.get("pid") != mb.get("pid"):
        out.append("rank" if ma.get("rank") is not None or mb.get("rank") is not None else "process")
    elif ma.get("kind") != mb.get("kind"):
        out.append("device" if "stream" in (ma.get("kind"), mb.get("kind")) or "device" in (
            ma.get("kind"), mb.get("kind")) else "thread")
    else:
        if ma.get("tid") != mb.get("tid"):
            out.append("thread")
        if ma.get("device") != mb.get("device"):
            out.append("device")
        if ma.get("kind") == "stream" and ma.get("ident") != mb.get("ident"):
            out.append("stream")
    if ma.get("category") != mb.get("category"):
        out.append("runtime")
    return out or ["lane"]


def build(trace, lane_of_span: Callable, lane_meta: Callable[[int], dict], *,
          cancel: Callable[[], bool] = lambda: False, progress: Callable[[str], None] = lambda s: None
          ) -> AnalysisData:
    t_start = time.time()
    out = AnalysisData()
    progress("dependency graph and critical path")
    report = cp.analyze(trace, sensitivity=None)
    res = report.model
    g = res.graph
    out.report = report
    if cancel():
        return out
    progress("mapping events to lanes")
    n = g.n
    lane_of_ord = np.full(n, -1, dtype=np.int32)
    eid_of_ord = np.full(n, -1, dtype=np.int64)
    names: list[str] = [""] * n
    for i, s in enumerate(trace.iter_spans()):
        if i >= n:
            break
        ln = lane_of_span(s)
        lane_of_ord[i] = -1 if ln is None else ln
        eid_of_ord[i] = s.eid
        names[i] = s.name
        if i % 200_000 == 0 and cancel():
            return out
    out.lane_of_ord, out.eid_of_ord = lane_of_ord, eid_of_ord

    # ── dependency edges (between lanes) with their evidence ─────────────
    progress("indexing dependency edges")
    walk_edges = set(res.walk.steps.tolist())
    edge_id_of: dict[int, int] = {}
    for e in range(len(g)):
        if not g.enabled[e] or g.mode[e] not in (cm.DEP, cm.IDLE):
            continue
        u, v = g.src[e], g.dst[e]
        ou, ov = u >> 1, v >> 1
        kind = cm.KINDS[g.kind[e]]
        prov = cm.PROVS[g.prov[e]]
        tier = cm.CONFS[g.conf[e]]
        k = len(out.edges)
        edge_id_of[e] = k
        out.edges.append(Edge(
            id=k, kind=kind, family=edge_family(kind, g.cat[ov]), runtime=g.runtime(e), tier=tier,
            provenance=prov, cls=cm.edge_class(prov, tier), src_lane=int(lane_of_ord[ou]),
            dst_lane=int(lane_of_ord[ov]), src_eid=int(eid_of_ord[ou]), dst_eid=int(eid_of_ord[ov]),
            t_src=int(g.t[u]), t_dst=int(g.t[v]), src_end="end" if u & 1 else "start",
            dst_end="end" if v & 1 else "start", on_path=e in walk_edges))
    if out.edges:
        lo = np.asarray([min(e.t_src, e.t_dst) for e in out.edges], dtype=np.float64)
        length = np.asarray([abs(e.t_dst - e.t_src) for e in out.edges], dtype=np.float64)
        ext = trace.store.span_extent(timed_only=True)
        cut = 0.01 * ((ext[1] - ext[0]) if ext else 1.0)
        out.long_edges = np.nonzero(length > cut)[0].tolist()
        short = length[length <= cut]
        out.short_max_ns = float(short.max()) if len(short) else 0.0
        out.edge_order = np.argsort(lo, kind="stable")
        out.edge_t = lo[out.edge_order]
    for ed in out.edges:
        out.edges_of_eid.setdefault(ed.src_eid, []).append(ed.id)
        out.edges_of_eid.setdefault(ed.dst_eid, []).append(ed.id)

    # ── the observed critical path ───────────────────────────────────────
    progress("critical-path overlay")
    for e, ns in zip(res.walk.steps.tolist(), res.walk.cover.tolist()):
        mode = g.mode[e]
        u, v = g.src[e], g.dst[e]
        kind = cm.KINDS[g.kind[e]]
        tier = cm.CONFS[g.conf[e]]
        if mode in (cm.BUSY, cm.WAIT):
            o = g.owner[e]
            lane = int(lane_of_ord[o])
            step = PathStep(lane, int(g.t[u]), int(g.t[v]), cm.MODES[mode], kind, tier, int(eid_of_ord[o]))
            out.credit_of_eid[step.eid] = out.credit_of_eid.get(step.eid, 0) + int(ns)
        elif mode == cm.DEP:
            step = PathStep(int(lane_of_ord[v >> 1]), int(g.t[u]), int(g.t[v]), "dependency", kind, tier,
                            int(eid_of_ord[v >> 1]), int(lane_of_ord[u >> 1]), int(eid_of_ord[u >> 1]),
                            edge_id_of.get(e, -1))
        else:
            step = PathStep(int(lane_of_ord[v >> 1]), int(g.t[u]), int(g.t[v]), cm.MODES[mode], kind, tier,
                            int(eid_of_ord[v >> 1]), edge=edge_id_of.get(e, -1))
        k = len(out.steps)
        out.steps.append(step)
        if step.lane >= 0:
            out.steps_of_lane.setdefault(step.lane, []).append(k)
            out.path_lanes.add(step.lane)
        if step.src_lane >= 0:
            out.path_lanes.add(step.src_lane)
    for j in res.path_spans:
        out.path_eids.add(int(eid_of_ord[j]))
    prev = None
    for st in out.steps:
        if st.mode == "dependency" and st.src_lane >= 0 and st.lane >= 0 and st.src_lane != st.lane:
            out.transitions.append(Transition(st.t0, st.src_lane, st.lane,
                                              _lane_meta_switch(lane_meta, st.src_lane, st.lane), st.kind, st.tier))
        elif prev is not None and prev.lane >= 0 and st.lane >= 0 and prev.lane != st.lane and st.mode != "dependency":
            out.transitions.append(Transition(st.t0, prev.lane, st.lane,
                                              _lane_meta_switch(lane_meta, prev.lane, st.lane), st.kind, st.tier))
        prev = st
    crit = res.crit
    if crit is not None:
        out.on_all = {int(eid_of_ord[j]) for j in crit.on_all}
        out.on_some = {int(eid_of_ord[j]) for j in crit.on_some}
        out.slack_of_eid = {int(eid_of_ord[j]): int(s) for j, s in crit.slack.items()}
    out.sensitivity = []
    for r in res.sensitivity:
        r2 = dict(r)
        if "span" in r2:
            r2["eid"] = int(eid_of_ord[r2["span"]])
            r2["lane"] = int(lane_of_ord[r2["span"]])
        out.sensitivity.append(r2)

    # ── waits: a gated call's time before its last dependency arrived ────
    progress("waits and their causes")
    for j in range(n):
        if not g.gated[j] or g.excluded[j]:
            continue
        v = 2 * j + 1
        best_e, best_t = -1, None
        for kk in range(g.in_ptr[v], g.in_ptr[v + 1]):
            e = g.in_idx[kk]
            if g.enabled[e] and g.mode[e] == cm.DEP and (best_t is None or g.t[g.src[e]] > best_t):
                best_e, best_t = e, g.t[g.src[e]]
        if best_e < 0:
            continue
        t0, t_end = int(g.t[2 * j]), int(g.t[v])
        t1 = min(int(best_t), t_end)
        if t1 <= t0:
            continue
        out.waits.append(WaitCause(int(lane_of_ord[j]), int(eid_of_ord[j]), names[j], t0, t1, t_end,
                                   edge_id_of.get(best_e, -1)))
    out.waits.sort(key=lambda w: w.t0)
    for k, w in enumerate(out.waits):
        out.waits_of_lane.setdefault(w.lane, []).append(k)

    # ── the span-level DP (legacy) and what it counts ────────────────────
    progress("span-level search value")
    out.legacy = _legacy_dp(trace, report)
    out.summary = _summary(report, res, out)
    out.ok = True
    out.seconds = time.time() - t_start
    return out


def _legacy_dp(trace, report) -> dict:
    """The span-level longest-path DP (criticalpath.Graph.critical_path, what
    analyze() used before cp-5): its value, and per hop the interval it
    counts -- each hop adds gap + duration ending at the hop's end, so
    hops whose counted intervals overlap count that wall time more than
    once. Shown so the search value is never mistaken for elapsed time."""
    try:
        graph = cp.load_or_build_graph(trace)
        path, confs, kinds = graph.critical_path()
        if not path:
            return {"available": False, "reason": "empty path"}
        spans = graph.spans
        ext = trace.store.span_extent(timed_only=True)
        wall = max(1, ext[1] - ext[0]) if ext else 1
        rep = cp.attribute_blame(spans, path, wall, confs, kinds)
        intervals = []
        prev = None
        for i, o in enumerate(path):
            s = spans[o]
            if prev is None:
                inc = s.duration_ns
            else:
                _gate, _limit, gap = cp._edge_gap_and_gate(spans, prev, o, kinds[i - 1])
                inc = gap + s.duration_ns
            end = cp._effective_end_ns(s)
            if inc > 0:
                intervals.append((int(end - inc), int(end), s.name, kinds[i - 1] if i else ""))
            prev = o
        # multiplicity: sweep
        pts = sorted([(a, 1) for a, _b, _n, _k in intervals] + [(b, -1) for _a, b, _n, _k in intervals])
        over = []
        depth = 0
        last = None
        union = 0
        for t, d in pts:
            if last is not None and t > last:
                if depth >= 1:
                    union += t - last
                if depth >= 2:
                    if over and over[-1][1] == last and over[-1][2] == depth:
                        over[-1][1] = t
                    else:
                        over.append([last, t, depth])
            depth += d
            last = t
        return {"available": True, "value_ns": int(rep.dp_accounted_ns), "covered_ns": int(rep.total_path_ns),
                "union_ns": int(union), "counted_ns": int(sum(b - a for a, b, _n, _k in intervals)),
                "intervals": intervals, "overlaps": [tuple(o) for o in over], "path_len": len(path)}
    except Exception as exc:          # display only: never cost the analysis
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}


def _summary(report, res, data: AnalysisData) -> dict:
    total = report.total_path_ns
    comp = []
    for cat, ns in sorted(report.path_work_by_category.items(), key=lambda kv: -kv[1]):
        comp.append({"group": "busy", "key": cat, "ns": ns})
    for cat, ns in sorted(report.path_in_call_by_category.items(), key=lambda kv: -kv[1]):
        comp.append({"group": "in gated call", "key": cat, "ns": ns})
    for kind, ns in sorted(report.path_dependency_by_kind.items(), key=lambda kv: -kv[1]):
        comp.append({"group": "dependency latency", "key": kind, "ns": ns})
    if report.path_unrecorded_ns:
        comp.append({"group": "unrecorded", "key": "thread time", "ns": report.path_unrecorded_ns})
    if report.path_device_idle_ns:
        comp.append({"group": "device idle", "key": "stream", "ns": report.path_device_idle_ns})
    ev = res.graph_evidence
    crit = res.crit
    tiers = {t: int(report.path_time_by_tier.get(t, 0)) for t in cm.CONFS}
    return {
        "wallNs": int(report.wall_ns), "observedNs": int(total), "pathSpans": len(report.path_span_indices),
        "modeledNs": int(report.modeled_path_ns), "modeledIdleNs": int(report.modeled_unrecorded_idle_ns),
        "modeledDirectNs": int(report.modeled_direct_only_ns),
        "legacyValueNs": int(data.legacy.get("value_ns", 0)) if data.legacy.get("available") else -1,
        "legacyUnionNs": int(data.legacy.get("union_ns", 0)) if data.legacy.get("available") else -1,
        "composition": comp,
        "workByCategory": dict(report.path_work_by_category),
        "waitedByCategory": dict(report.wait_caused_by_category),
        "tiers": tiers, "classes": {k: int(v) for k, v in report.path_time_by_class.items()},
        "workOnAllNs": int(crit.work_on_all_ns) if crit else 0, "workOnSomeNs": int(crit.work_on_some_ns) if crit else 0,
        "spansOnAll": len(crit.on_all) if crit else 0, "spansOnSome": len(crit.on_some) if crit else 0,
        "ties": int(res.walk.ties), "variants": res.variants, "stableSpans": res.stable_spans,
        "variantOnlySpans": res.variant_only_spans, "variantOnlyNs": int(res.variant_only_ns),
        "graph": {"byKind": ev["by_kind"], "byTier": ev["by_tier"], "byClass": ev["by_class"],
                  "byProvenance": ev["by_provenance"], "dropped": ev["dropped_contradicting_time"],
                  "cycles": ev["dropped_in_cycles"], "rooted": ev["rooted_collective_edges"]},
        "diagnostics": dict(report.diagnostics or {}), "notes": list(report.notes),
        "transitions": len(data.transitions),
        "transitionKinds": dict(Counter("/".join(t.switch) for t in data.transitions)),
    }


def fmt(ns: float) -> str:
    return dash.fmt_ns(ns)
