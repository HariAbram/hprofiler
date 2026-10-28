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

Alternative worlds (cp_alternatives) are part of the analysis only up to
criticalpath.ALTERNATIVES_AUTO_NODE_LIMIT event nodes; beyond, they are
"not evaluated" until asked for (evaluate_alternatives, on a background
thread) -- never presented as "none" or as "every world".
"""
from __future__ import annotations

import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ..analysis import cp_model as cm
from ..analysis import criticalpath as cp
from ..analysis import dashboard as dash

# Alternative-world evaluation states (AnalysisData.alternatives_state).
ALT_STATE_LABELS = {
    "not_evaluated": "not evaluated", "none_found": "none found",
    "truncated": "evaluation truncated", "evaluated": "evaluated",
}

# What each path variant assumes (cp_model.VARIANTS), in words.
VARIANT_LABELS = {
    "default": "as recorded",
    "unrecorded_as_work": "unrecorded / unknown time as work",
    "direct_evidence_only": "direct-evidence edges only",
    "gated_tail_as_work": "gated calls' tails as processing",
}

# Span-level edge kinds whose alternatives the generators produce, and which
# event each endpoint is (cp_model's dependency edges).
_ALT_ENDPOINTS = {"p2p": ("start", "end"), "arrival": ("start", "end"), "launch": ("start", "start")}

# Overlay families (the dependency filter's groups).
FAMILIES = ("critical", "mpi_p2p", "mpi_collective", "openmp", "submission", "gpu_sync", "nccl", "other")
FAMILY_LABELS = {
    "critical": "Critical-path hops", "mpi_p2p": "MPI point-to-point", "mpi_collective": "MPI collectives",
    "openmp": "OpenMP barriers / tasks / locks", "submission": "CPU → GPU submission",
    "gpu_sync": "GPU / OpenCL synchronization, stream and queue order", "nccl": "NCCL",
    "other": "Other explicit ids",
}


def edge_family(kind: str, dst_cat: str) -> str:
    if kind == "p2p":
        return "nccl" if dst_cat == "nccl" else "mpi_p2p"
    if kind in ("task_join", "task_dependence", "lock_handoff"):
        return "openmp"
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
    status: str = ""         # evidence.STATUSES (observed / inferred / ambiguous ...)
    alternatives: int = 0    # other candidates answering the same question
    candidate: int = -1      # its row in AnalysisData.evidence


@dataclass
class ExtraEdge:
    """A dependency that is NOT an active edge of the graph, drawn only on
    request: one the clocks' bounds or the timestamps contradict
    (`disabled`), or an alternative another world uses (`alternative`)."""
    kind: str
    family: str
    tier: str
    role: str                # "disabled" | "alternative"
    reason: str              # clock_bound / time_contradiction / the alternative's evidence
    src_lane: int
    dst_lane: int
    src_eid: int
    dst_eid: int
    t_src: int
    t_dst: int
    candidate: int = -1      # its row in AnalysisData.evidence (-1: none)
    worlds: tuple = ()       # alternative: the evaluated worlds that use it


@dataclass
class PathStep:
    lane: int
    t0: int
    t1: int
    mode: str                # busy / wait / unknown / gap / idle / dependency
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
    evidence: Any = None                 # evidence.EvidenceSet (with alternatives when computed)
    robust_every: set[int] = field(default_factory=set)   # on the path in every alternative world
    robust_some: set[int] = field(default_factory=set)    # ... in some
    alternatives_evaluated: bool = False
    alternatives_state: str = "not_evaluated"             # ALT_STATE_LABELS
    alternatives_reason: str = ""
    worlds: list[dict] = field(default_factory=list)      # evaluated worlds ([0]: as recorded)
    worst_world: int = -1
    blame_of_eid: dict[int, int] = field(default_factory=dict)   # waiting time its work delayed
    replay_of_eid: dict[int, int] = field(default_factory=dict)  # replayed reduction of its removal
    lane_of_ord: np.ndarray = None
    eid_of_ord: np.ndarray = None
    start_of_ord: np.ndarray = None      # the store span's start / end (what the lanes draw)
    end_of_ord: np.ndarray = None
    # path variants (cp_model.VARIANTS): spans on each variant's observed path
    variant_eids: dict[str, set[int]] = field(default_factory=dict)
    stable_eids: set[int] = field(default_factory=set)        # on it in every variant
    variant_only_eids: set[int] = field(default_factory=set)  # in some variants only
    near_eids: set[int] = field(default_factory=set)          # near-critical busy work
    near_basis: str = ""
    # blocking calls: no recorded dependency releases them (unknown), or the
    # gated ones' release and the upper bound of their processing after it
    unknown_eids: set[int] = field(default_factory=set)
    unknown_of_lane: dict[int, list[tuple[int, int, int]]] = field(default_factory=dict)
    gated_of_eid: dict[int, dict] = field(default_factory=dict)
    extra_edges: list[ExtraEdge] = field(default_factory=list)     # ids len(edges) + k
    undecidable: list[dict] = field(default_factory=list)
    blame_waits: list[dict] = field(default_factory=list)          # joint releases first
    joint_of_eid: dict[int, list[int]] = field(default_factory=dict)   # eid -> its joint waits (indices)
    passed_on_of_eid: dict[int, int] = field(default_factory=dict)
    passed_rows: list[dict] = field(default_factory=list)       # the largest, for the blame section
    sensitivity_eids: set[int] = field(default_factory=set)       # spans the replay was run for

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
    start_of_ord = np.zeros(n, dtype=np.int64)
    end_of_ord = np.zeros(n, dtype=np.int64)
    names: list[str] = [""] * n
    for i, s in enumerate(trace.iter_spans()):
        if i >= n:
            break
        ln = lane_of_span(s)
        lane_of_ord[i] = -1 if ln is None else ln
        eid_of_ord[i] = s.eid
        start_of_ord[i], end_of_ord[i] = s.start_ns, s.start_ns + max(s.duration_ns, 0)
        names[i] = s.name
        if i % 200_000 == 0 and cancel():
            return out
    out.lane_of_ord, out.eid_of_ord = lane_of_ord, eid_of_ord
    out.start_of_ord, out.end_of_ord = start_of_ord, end_of_ord

    # ── dependency edges (between lanes) with their evidence ─────────────
    progress("indexing dependency edges")
    from ..analysis import evidence as ev
    alts = report.alternatives
    es = alts.evidence if alts is not None and alts.evidence is not None else \
        ev.evidence_from_graph(cp.load_or_build_graph(trace).edges, g)
    out.evidence = es
    group_size = np.bincount(es.cand["group"], minlength=len(es.groups)) if len(es) else np.zeros(0, int)
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
        c = int(g.origin[e]) if g.origin[e] >= 0 and g.origin[e] < es.n_selected else -1
        out.edges.append(Edge(
            id=k, kind=kind, family=edge_family(kind, g.cat[ov]), runtime=g.runtime(e), tier=tier,
            provenance=prov, cls=cm.edge_class(prov, tier), src_lane=int(lane_of_ord[ou]),
            dst_lane=int(lane_of_ord[ov]), src_eid=int(eid_of_ord[ou]), dst_eid=int(eid_of_ord[ov]),
            t_src=int(g.t[u]), t_dst=int(g.t[v]), src_end="end" if u & 1 else "start",
            dst_end="end" if v & 1 else "start", on_path=e in walk_edges,
            status=es.status(c) if c >= 0 else "",
            alternatives=int(group_size[int(es.cand["group"][c])]) - 1 if c >= 0 else 0, candidate=c))
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
        if mode in (cm.BUSY, cm.WAIT, cm.UNKNOWN):
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
            r2["startNs"] = float(start_of_ord[r2["span"]])
            out.replay_of_eid[r2["eid"]] = int(r2.get("replay_reduction_ns", 0))
            out.sensitivity_eids.add(r2["eid"])
        out.sensitivity.append(r2)
    out._edge_id_of = edge_id_of
    _index_variants(out, res, g)
    _index_blocking(out, g)
    _index_extra_edges(out, g, es)
    _index_undecidable(out, res, g)
    _index_blame(out, res, g)
    apply_alternatives(out, alts, n_nodes=2 * n)

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


def _eid(out: AnalysisData, j: int) -> int:
    return int(out.eid_of_ord[j]) if 0 <= j < len(out.eid_of_ord) else -1


def _lane(out: AnalysisData, j: int) -> int:
    return int(out.lane_of_ord[j]) if 0 <= j < len(out.lane_of_ord) else -1


def _index_variants(out: AnalysisData, res, g) -> None:
    """Spans on each variant's observed path; stable (every variant) and
    variant-only (some); near-critical in the recorded world (busy work
    with slack within 1% of the modeled length -- cp_alternatives' rule)."""
    sets = getattr(res, "variant_spans", None) or {}
    out.variant_eids = {k: {_eid(out, j) for j in v} for k, v in sets.items()}
    if out.variant_eids:
        vals = list(out.variant_eids.values())
        out.stable_eids = set.intersection(*vals)
        out.variant_only_eids = set.union(*vals) - out.stable_eids
    crit = res.crit
    if crit is not None and res.modeled_ns > 0:
        lim = 0.01 * res.modeled_ns
        out.near_eids = {_eid(out, j) for j, sl in crit.slack.items() if sl <= lim}
        out.near_basis = "recorded world"


def _index_blocking(out: AnalysisData, g) -> None:
    """Blocking calls no recorded dependency releases (unknown: never drawn
    or described as a gated wait), and every gated call's release with the
    upper bound of its processing after it (cp_model._tail_after_release)."""
    for j in range(g.n):
        if g.excluded[j]:
            continue
        eid = _eid(out, j)
        if g.blocking[j] and not g.gated[j]:
            lane = _lane(out, j)
            out.unknown_eids.add(eid)
            if lane >= 0:
                out.unknown_of_lane.setdefault(lane, []).append(
                    (int(out.start_of_ord[j]), int(out.end_of_ord[j]), eid))
        elif g.gated[j] and g.release[j] >= 0:
            v = 2 * j + 1
            best, best_t = -1, None
            for kk in range(g.in_ptr[v], g.in_ptr[v + 1]):
                e = g.in_idx[kk]
                if g.enabled[e] and g.mode[e] == cm.DEP and (best_t is None or g.t[g.src[e]] > best_t):
                    best, best_t = e, g.t[g.src[e]]
            out.gated_of_eid[eid] = {
                "releaseNs": int(g.release[j]), "callStartNs": int(g.t[2 * j]), "callEndNs": int(g.t[v]),
                "tailNs": int(cm._tail_after_release(g, j)),
                "edge": out._edge_id_of.get(best, -1) if best >= 0 else -1,
                "selfReleased": best < 0,       # a rendezvous's last arriver / a rooted member nobody gates
            }
    for lst in out.unknown_of_lane.values():
        lst.sort()


def _extra_endpoints(out: AnalysisData, g, src: int, dst: int, kind: str) -> tuple[int, int]:
    a, b = _ALT_ENDPOINTS.get(kind, ("start", "end"))
    return (int(g.t[2 * src + (a == "end")]), int(g.t[2 * dst + (b == "end")]))


def _index_extra_edges(out: AnalysisData, g, es) -> None:
    """Edges the graph does not use as dependencies: disabled because no
    clock offsets within the bounds satisfy them, or dropped because their
    source lies after their destination beyond the tolerance."""
    for e in sorted(g.clock_disabled):
        u, v = g.src[e], g.dst[e]
        kind = cm.KINDS[g.kind[e]]
        out.extra_edges.append(ExtraEdge(
            kind, edge_family(kind, g.cat[v >> 1]), cm.CONFS[g.conf[e]], "disabled", "clock_bound",
            _lane(out, u >> 1), _lane(out, v >> 1), _eid(out, u >> 1), _eid(out, v >> 1), int(g.t[u]), int(g.t[v]),
            int(g.origin[e]) if 0 <= g.origin[e] < es.n_selected else -1))
    for u, v, k, c in g.dropped_pairs:
        kind = cm.KINDS[k]
        out.extra_edges.append(ExtraEdge(
            kind, edge_family(kind, g.cat[v >> 1]), cm.CONFS[c], "disabled", "time_contradiction",
            _lane(out, u >> 1), _lane(out, v >> 1), _eid(out, u >> 1), _eid(out, v >> 1), int(g.t[u]), int(g.t[v])))


def _index_undecidable(out: AnalysisData, res, g) -> None:
    """Walk steps whose release the clocks cannot order against another
    candidate: where, the chosen and the other release, and the margin."""
    for best, alt in res.walk.undecidable_steps:
        v = g.dst[best]
        j = v >> 1
        out.undecidable.append({
            "atNs": float(g.t[v]), "laneIndex": _lane(out, j), "spanIdx": float(_eid(out, j)),
            "chosenKind": cm.KINDS[g.kind[best]], "alternativeKind": cm.KINDS[g.kind[alt]],
            "chosenEdge": out._edge_id_of.get(best, -1), "alternativeEdge": out._edge_id_of.get(alt, -1),
            "chosenSrcLane": _lane(out, g.src[best] >> 1), "chosenSrcIdx": float(_eid(out, g.src[best] >> 1)),
            "altSrcLane": _lane(out, g.src[alt] >> 1), "altSrcIdx": float(_eid(out, g.src[alt] >> 1)),
            "chosenSrcNs": float(g.t[g.src[best]]), "altSrcNs": float(g.t[g.src[alt]]),
            "marginNs": float(g.t[g.src[best]] - g.t[g.src[alt]]),
            "chosenDomain": _domain_label(g, g.src[best] >> 1), "altDomain": _domain_label(g, g.src[alt] >> 1),
        })


def _domain_label(g, j: int) -> str:
    ca = g.clock
    if ca is None or not (0 <= g.domain[j] < len(ca.keys)):
        return ""
    return "/".join(str(p) for p in ca.keys[g.domain[j]])


def _index_blame(out: AnalysisData, res, g) -> None:
    """Blame per span, the waits it was computed from (joint releases first),
    and what a waiting call passed on to its own releasers."""
    b = res.blame
    if b is None:
        return
    out.blame_of_eid = {_eid(out, j): int(round(ns)) for j, ns in b.by_span.items() if ns >= 1}
    passed = {j: ns for j, ns in getattr(b, "passed_on", {}).items() if ns >= 1}
    out.passed_on_of_eid = {_eid(out, j): int(round(ns)) for j, ns in passed.items()}
    out.passed_rows = [{"spanIdx": float(_eid(out, j)), "laneIndex": _lane(out, j), "name": g.name(j),
                        "startNs": float(out.start_of_ord[j]), "ns": float(ns)}
                       for j, ns in sorted(passed.items(), key=lambda kv: -kv[1])[:12]]
    waits = sorted(getattr(b, "waits", ()), key=lambda w: (len(w[4]) < 2, -w[3]))
    for j, a, r, w, rel in waits[:2000]:
        k = len(out.blame_waits)
        out.blame_waits.append({
            "laneIndex": _lane(out, j), "spanIdx": float(_eid(out, j)), "name": g.name(j),
            "startNs": float(a), "releaseNs": float(r), "waitNs": float(w), "joint": len(rel) > 1,
            "releasers": [{"laneIndex": _lane(out, s), "spanIdx": float(_eid(out, s)), "name": g.name(s),
                           "startNs": float(out.start_of_ord[s])} for s in rel],
        })
        if len(rel) > 1:
            for s in rel:
                out.joint_of_eid.setdefault(_eid(out, s), []).append(k)


def alternatives_state(alts, n_nodes: int) -> tuple[str, str]:
    """(state, why) for an alternatives report -- None: not evaluated."""
    if alts is None:
        return "not_evaluated", (
            f"not evaluated: {n_nodes:,} event nodes exceed the automatic limit of "
            f"{cp.ALTERNATIVES_AUTO_NODE_LIMIT:,} (each world is one more graph assembly); "
            "evaluate them from Critical path ▸ Alternative worlds")
    if alts.skipped:
        return "not_evaluated", alts.skipped
    if alts.verdict == "no alternatives":
        return "none_found", ("the trace leaves open no dependency the generators consider (point-to-point "
                              "alignment, last arriver within the clocks' uncertainty, a shared correlation id)")
    if alts.truncated:
        if alts.cancelled:
            return "truncated", f"stopped early after {alts.worlds} world(s): the sets cover only those"
        return "truncated", (f"{alts.touching_units} choice units touch the recorded path; "
                             f"{alts.one_change_limit} got a one-change world")
    return "evaluated", (f"{alts.worlds} worlds ({alts.one_change_worlds} one-change, {alts.sampled_worlds} "
                         "sampled) -- a sample of the worlds the trace allows, not all of them")


def apply_alternatives(out: AnalysisData, alts, *, n_nodes: int) -> None:
    """Everything that depends on the alternatives report: edge statuses and
    alternative counts, the robust sets, the evaluated worlds and their
    alternative edges, the near-critical set, the state and the summary
    blocks. Called by build() and after an on-demand evaluation."""
    report = out.report
    res = report.model
    g = res.graph
    report.alternatives = alts
    out.alternatives_state, out.alternatives_reason = alternatives_state(alts, n_nodes)
    evaluated = out.alternatives_state in ("none_found", "truncated", "evaluated")
    out.alternatives_evaluated = evaluated
    if alts is not None and alts.evidence is not None:
        out.evidence = alts.evidence
    es = out.evidence           # build() always sets it (the default view without alternatives)
    group_size = np.bincount(es.cand["group"], minlength=len(es.groups)) if es is not None and len(es) else None
    for e in out.edges:
        c = e.candidate
        if es is not None and 0 <= c < len(es):
            e.status = es.status(c)
            e.alternatives = int(group_size[int(es.cand["group"][c])]) - 1
    out.extra_edges = [x for x in out.extra_edges if x.role != "alternative"]
    out.worlds, out.worst_world = [], -1
    out.robust_every, out.robust_some = set(), set()
    if evaluated:
        out.robust_every = {_eid(out, j) for j in alts.every}
        out.robust_some = {_eid(out, j) for j in alts.some}
        if alts.near is not None and alts.worlds > 1:
            out.near_eids = {_eid(out, j) for j in alts.near}
            out.near_basis = f"every one of {alts.worlds} evaluated worlds"
        _index_worlds(out, alts, g, es)
    cal = getattr(report, "calibration", None)
    if cal is not None and es is not None:
        # the statuses changed (inferred -> ambiguous): so do the path's probabilities
        from ..analysis import edge_calibration as ec
        report.edge_probabilities = ec.path_probabilities(cal, res, es)
    out.summary = _summary(report, res, out) if out.ok or out.summary else out.summary


def _index_worlds(out: AnalysisData, alts, g, es) -> None:
    from ..analysis import evidence as ev
    from ..analysis.criticalpath import EDGE_KINDS
    world_list = getattr(alts, "world_list", None) or []
    used_by: dict[int, list[int]] = defaultdict(list)
    for wi, w in enumerate(world_list):
        for c in w.added:
            used_by[c].append(wi)
    alt_ids: dict[int, int] = {}
    for c, ws in sorted(used_by.items()):
        r = es.cand[c]
        dst, src, kind = int(r["dst"]), int(r["src"]), EDGE_KINDS[int(r["kind"])]
        if not (0 <= src < g.n and 0 <= dst < g.n):
            continue
        ts, td = _extra_endpoints(out, g, src, dst, kind)
        why = ", ".join(ev.kinds_of(int(r["ev_for"]))) or "alternative"
        alt_ids[c] = len(out.edges) + len(out.extra_edges)
        out.extra_edges.append(ExtraEdge(
            kind, edge_family(kind, g.cat[dst]), cm.CONFS[int(r["conf"])], "alternative", why,
            _lane(out, src), _lane(out, dst), _eid(out, src), _eid(out, dst), ts, td, c, tuple(ws)))
    rec = world_list[0].spans if world_list else set()
    for wi, w in enumerate(world_list):
        share = w.robust_in_call_ns / w.in_call_ns if w.in_call_ns else 1.0
        out.worlds.append({
            "index": wi, "kind": w.kind, "label": w.label,
            "coveredNs": float(w.covered_ns), "modeledNs": float(w.modeled_ns),
            "inCallNs": float(w.in_call_ns), "robustInCallNs": float(w.robust_in_call_ns), "robustShare": share,
            "joined": len(w.spans - rec), "left": len(rec - w.spans),
            "eids": {_eid(out, j) for j in w.spans},
            "altEdges": [alt_ids[c] for c in w.added if c in alt_ids], "removed": len(w.removed),
        })
    out.worst_world = getattr(alts, "worst_world", -1)


def evaluate_alternatives(out: AnalysisData, trace, samples: int, *, progress=None, cancel=None):
    """Alternative worlds on request (the GUI's equivalent of
    `critical-path --alternatives N`): N sampled worlds and up to 16
    one-change worlds, on the analysis already computed. Qt-free and
    read-only on `out` -- run it on a worker thread, then hand the report
    to apply_alternatives on the main thread."""
    from ..analysis import cp_alternatives
    res = out.report.model
    graph = cp.load_or_build_graph(trace)
    base = res.graph
    if base.skeleton is None:
        # what the default build read from the spans, so each world is
        # assembled without reading them again (cp_model.graph_with_edges)
        base.skeleton = cm.build_event_graph(trace, graph, keep_skeleton=True).skeleton
    try:
        return cp_alternatives.analyze_alternatives(trace, graph, res, samples=max(0, int(samples)),
                                                    one_change=16, progress=progress, cancel=cancel)
    finally:
        base.skeleton = None


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
    for cat, ns in sorted(report.path_unmatched_by_category.items(), key=lambda kv: -kv[1]):
        comp.append({"group": "in an ungated blocking call", "key": cat, "ns": ns})
    if report.path_device_idle_ns:
        comp.append({"group": "device idle", "key": "stream", "ns": report.path_device_idle_ns})
    ev = res.graph_evidence
    alt_summary = _alternatives_summary(report, data)
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
        "unmatchedByCategory": dict(report.path_unmatched_by_category),
        "modeledRangeNs": [int(x) for x in report.modeled_range_ns],
        "modeledGatedTailNs": int(report.modeled_gated_tail_ns),
        "undecidable": int(res.walk.undecidable), "undecidableNs": int(res.walk.undecidable_ns),
        "clocks": ev.get("clocks"), "clockContradicted": int(ev.get("clock_contradicted", 0)),
        "alternatives": alt_summary,
        "evidence": data.evidence.summary() if data.evidence is not None else None,
        "blameByCategory": dict(report.blame_by_category), "probabilities": _probabilities(report, data),
        "transitions": len(data.transitions),
        "transitionKinds": dict(Counter("/".join(t.switch) for t in data.transitions)),
        "variantsTable": _variants_table(res, data), "modeledRange": _modeled_range(res),
        "headlineVariant": "unrecorded_as_work",
        "stableCount": len(data.stable_eids), "variantOnlyCount": len(data.variant_only_eids),
        "nearCount": len(data.near_eids), "nearBasis": data.near_basis,
        "statusSummary": status_summary(data),
        "clock": _clock_summary(res.graph, data),
        "undecidableSteps": data.undecidable[:200],
        "blame": _blame_summary(report, res, data),
        "sensitivityRows": _sensitivity_rows(res, data),
        "sensitivityNote": res.sensitivity_note,
        "gatedCalls": len(data.gated_of_eid), "unknownCalls": len(data.unknown_eids),
    }


def _variants_table(res, data: AnalysisData) -> list[dict]:
    """The four variants by what they assume, each with its observed and
    modeled lengths; which of them set the modeled range's ends."""
    rows = []
    vals = [v["modeled_ns"] for v in res.variants.values()]
    lo, hi = (min(vals), max(vals)) if vals else (0, 0)
    for name, (gaps, direct, gated) in cm.VARIANTS.items():
        v = res.variants.get(name)
        if v is None:
            continue
        rows.append({
            "name": name, "label": VARIANT_LABELS.get(name, name),
            "unrecorded": "work" if gaps else "idle", "edges": "direct evidence only" if direct else "all",
            "tails": "processing" if gated else "waiting",
            "observedNs": float(v["covered_ns"]), "modeledNs": float(v["modeled_ns"]), "spans": int(v["spans"]),
            "sameAsDefault": bool(v["same_spans_as_default"]),
            "isMin": v["modeled_ns"] == lo, "isMax": v["modeled_ns"] == hi,
            "headline": name == "unrecorded_as_work",
        })
    return rows


def _modeled_range(res) -> dict:
    """The modeled length's range over the four variants -- the only values
    labelled minimum / maximum (none is a bound by its semantics alone)."""
    vals = {k: v["modeled_ns"] for k, v in res.variants.items()}
    if not vals:
        return {}
    lo, hi = min(vals.values()), max(vals.values())
    return {"minNs": float(lo), "maxNs": float(hi), "headlineNs": float(res.modeled_ns),
            "minVariants": [VARIANT_LABELS.get(k, k) for k, v in vals.items() if v == lo],
            "maxVariants": [VARIANT_LABELS.get(k, k) for k, v in vals.items() if v == hi]}


def _alternatives_summary(report, data: AnalysisData) -> dict:
    alts = report.alternatives
    st = data.alternatives_state
    out = {"state": st, "stateLabel": ALT_STATE_LABELS.get(st, st), "reason": data.alternatives_reason,
           "evaluated": data.alternatives_evaluated, "nodes": int(2 * report.model.graph.n),
           "autoLimit": int(cp.ALTERNATIVES_AUTO_NODE_LIMIT), "skipped": alts.skipped if alts else "not evaluated"}
    if alts is None or not data.alternatives_evaluated:
        return out
    worst = data.worlds[data.worst_world] if 0 <= data.worst_world < len(data.worlds) else None
    out.update({
        "verdict": alts.verdict, "reasons": list(alts.reasons), "units": dict(alts.units),
        "worlds": alts.worlds, "oneChange": alts.one_change_worlds, "sampled": alts.sampled_worlds,
        "touching": alts.touching_units, "oneChangeLimit": alts.one_change_limit, "samples": alts.samples,
        "every": len(alts.every), "some": len(alts.some), "near": len(alts.near),
        "alternativeEdges": sum(1 for x in data.extra_edges if x.role == "alternative"),
        "coveredRangeNs": [float(x) for x in alts.covered_range_ns],
        "modeledRangeNs": [float(x) for x in alts.modeled_range_ns],
        "worstWorld": data.worst_world,
        "worstShare": worst["robustShare"] if worst else 1.0,
        "worldRows": [{k: v for k, v in w.items() if k != "eids"} | {"spans": len(w["eids"])}
                      for w in data.worlds],
    })
    return out


def status_summary(data: AnalysisData) -> list[dict]:
    """Per evidence status: candidates, active graph edges, their summed
    source-to-target time, and the observed path's time on dependency steps
    with that status."""
    from ..analysis.evidence import STATUSES
    es = data.evidence
    cand = Counter(es.status(i) for i in range(len(es))) if es is not None else Counter()
    edges, latency, path = Counter(), Counter(), Counter()
    for e in data.edges:
        s = e.status or "untracked"
        edges[s] += 1
        latency[s] += abs(e.t_dst - e.t_src)
    for st in data.steps:
        if st.mode == "dependency" and 0 <= st.edge < len(data.edges):
            path[data.edges[st.edge].status or "untracked"] += st.t1 - st.t0
    disabled = Counter(x.reason for x in data.extra_edges if x.role == "disabled")
    rows = []
    for s in STATUSES + ("untracked",):
        if not (cand[s] or edges[s]):
            continue
        rows.append({"status": s, "candidates": int(cand[s]), "edges": int(edges[s]),
                     "latencyNs": float(latency[s]), "pathNs": float(path[s]),
                     "disabled": int(disabled["clock_bound"] + disabled["time_contradiction"])
                     if s == "contradicted" else 0})
    if (disabled["clock_bound"] or disabled["time_contradiction"]) and not cand["contradicted"]:
        rows.append({"status": "contradicted", "candidates": 0, "edges": 0, "latencyNs": 0.0, "pathNs": 0.0,
                     "disabled": int(disabled["clock_bound"] + disabled["time_contradiction"])})
    return rows


def _clock_summary(g, data: AnalysisData) -> dict:
    ca = g.clock
    disabled = Counter(x.reason for x in data.extra_edges if x.role == "disabled")
    out = {"available": ca is not None, "clockContradicted": int(g.clock_contradicted),
           "disabledByClock": int(disabled["clock_bound"]), "droppedByTime": int(disabled["time_contradiction"]),
           "undecidable": len(data.undecidable)}
    if ca is None:
        return out
    d = ca.to_dict()
    out.update({"feasibleAsRecorded": bool(d["feasible_as_recorded"]), "constraints": int(d["constraints"]),
                # each assembly (before and after a shift) reports what it gave up: once each
                "relaxed": list(dict.fromkeys(
                    f"{r.get('from')} → {r.get('to')}: needed {_signed_ns(r.get('need_ns', 0))}, the bounds "
                    f"allow {_signed_ns(r.get('allowed_ns', 0))}; {r.get('edges_contradicted', 0)} "
                    f"{r.get('strength', '')} edge(s) contradicted"
                    if isinstance(r, dict) else str(r) for r in d["relaxed"]))[:20],
                "domains": [{"domain": x["domain"],
                             "bound": "none" if x["bound_ns"] is None else dash.fmt_ns(x["bound_ns"]),
                             "lo": "−∞" if x["lo_ns"] is None else _signed_ns(x["lo_ns"]),
                             "hi": "+∞" if x["hi_ns"] is None else _signed_ns(x["hi_ns"]),
                             "shift": _signed_ns(x["shift_ns"]) if x["shift_ns"] else "none",
                             "shifted": bool(x["shift_ns"])} for x in d["domains"]]})
    return out


def _signed_ns(ns: float) -> str:
    return ("+" if ns > 0 else "−" if ns < 0 else "") + dash.fmt_ns(abs(ns))


def _blame_summary(report, res, data: AnalysisData) -> dict:
    b = res.blame
    if b is None:
        return {"available": False, "reason": res.sensitivity_note or "not computed for this graph size"}
    total = max(1.0, b.waiting_ns)
    rows = [{"key": k, "ns": float(v), "share": v / total, "bucket": k.startswith("(")}
            for k, v in sorted(report.blame_by_category.items(), key=lambda kv: -kv[1])]
    joint = [w for w in data.blame_waits if w["joint"]]
    return {"available": True, "waitingNs": float(b.waiting_ns), "jointReleases": int(b.joint_releases),
            "truncated": int(b.truncated), "byCategory": rows, "jointWaits": joint[:50],
            "passedOn": list(data.passed_rows)}


def _sensitivity_rows(res, data: AnalysisData) -> list[dict]:
    out = []
    for r in data.sensitivity:
        out.append({"scope": r.get("scope", ""), "category": r.get("category", ""), "name": r.get("name", ""),
                    "spans": int(r.get("spans", 1)), "workOnLongestNs": float(r.get("work_on_longest_ns", 0)),
                    "modeledNs": float(r.get("modeled_reduction_ns", 0)),
                    "replayNs": float(r.get("replay_reduction_ns", 0)),
                    "every": bool(r.get("every", False)),
                    "laneIndex": int(r.get("lane", -1)), "spanIdx": float(r.get("eid", -1)),
                    "startNs": float(r.get("startNs", 0))})
    return out


def _probabilities(report, data: AnalysisData) -> dict:
    """With no calibration: unquantified. With one: where it came from, the
    evidence version it was fitted for, its model and coverage, the path's
    quantified steps and expected wrong ones."""
    from ..analysis import edge_calibration as ec
    from ..analysis.evidence import evidence_version
    p = dict(report.edge_probabilities or {})
    cal = getattr(report, "calibration", None)
    if cal is None or p.get("status") != "calibrated":
        return {"status": ec.UNQUANTIFIED, "calibrated": False,
                "reason": "no calibration given (HPROFILER_EDGE_CALIBRATION=FILE): tiers and statuses are "
                          "evidence classes, not probabilities"}
    quantified = sum(1 for e in data.edges if edge_probability(data, e) is not None)
    cur = evidence_version()
    return {"status": "calibrated", "calibrated": True, "source": getattr(report, "calibration_source", ""),
            "dataset": cal.dataset or "(unnamed)", "evidenceVersion": cal.evidence_version or "(not recorded)",
            "currentVersion": cur, "versionMatches": cal.evidence_version == cur,
            "model": f"isotonic regression (pool-adjacent-violators) of an ordinal evidence score, per "
                     f"(edge kind, provenance) stratum with at least {cal.min_examples} labels",
            "strata": len(cal.strata), "examples": int(cal.examples),
            "edgesQuantified": quantified, "edgesTotal": len(data.edges),
            "quantifiedSteps": int(p.get("quantified_steps", 0)), "unquantifiedSteps": int(p.get("unquantified_steps", 0)),
            "expectedWrongSteps": float(p.get("expected_wrong_steps", 0.0)),
            "pAllQuantifiedRight": p.get("p_all_quantified_right")}


def edge_probability(data: AnalysisData, e) -> float | None:
    """The calibrated probability of an active edge (None: no calibration,
    or its stratum has none)."""
    from ..analysis import edge_calibration as ec
    cal = getattr(data.report, "calibration", None) if data.report is not None else None
    es = data.evidence
    if cal is None or es is None or not (0 <= e.candidate < len(es)):
        return None
    r = es.cand[e.candidate]
    prov = cm.PROVS[int(r["prov"])] if r["prov"] >= 0 else e.provenance
    tier = cm.CONFS[int(r["conf"])]
    return cal.probability(_span_kind(r), prov,
                           ec.evidence_score(es.status(e.candidate), tier, cm.edge_class(prov, tier)))


def _span_kind(r) -> str:
    from ..analysis.criticalpath import EDGE_KINDS
    return EDGE_KINDS[int(r["kind"])]


def fmt(ns: float) -> str:
    return dash.fmt_ns(ns)
