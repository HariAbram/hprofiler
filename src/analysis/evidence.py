"""
Dependency evidence: what the trace says about each possible dependency.

The edge builders (criticalpath.py) answer one question per edge -- this
send matches that receive, these calls form one collective, this sync call
waited for that stream -- and persist only the answer. This module keeps
the question, the candidate answers, what supports and what contradicts
each, and a status, so a report can say how much of a critical path rests
on which kind of evidence and where the trace does not decide.

Vocabulary
----------
  Evidence     one kind of fact (EVIDENCE): an id the hook recorded on both
               ends, a runtime correlation id, a runtime ordering rule on
               one clock, the source MPI_Status reported, a posting-order or
               call-order position, an overlap in time -- with its class
               (direct, inferred, or against) and the kinds it is derived
               from (a status-resolved match rests on MPI_Status AND posting
               order). Per candidate, a bitmask of the kinds that support it
               and of those against it; explain() renders the tree with the
               raw tags of the two spans.
  Candidate    a possible dependency src -> dst of a span-level edge kind
               (criticalpath.EDGE_KINDS), with its tier, its provenance
               (cp_model.PROVENANCE), its evidence, a status, whether the
               builders selected it (the default view) and whether it is an
               edge of the event graph (in_model).
  MatchGroup   the candidates answering one question, with a cardinality:
                 exactly_one   a device operation has one submission; a
                               collective member one last arriver
                 at_most_one   exactly one, unless the process lost events
                 any_subset    each candidate independently (a sync call
                               waits for every stream it synchronizes;
                               program order, explicit ids)

Statuses
--------
  observed      direct evidence, the only candidate its group needs,
                timestamps consistent
  inferred      chosen by a rule (posting order, call order, time overlap)
                or with a medium / low tier, not contradicted
  ambiguous     its group needs exactly one candidate but has several the
                evidence does not separate
  contradicted  the timestamps rule it out: its source event lies after its
                destination beyond the clock tolerance, no offsets of the
                clock domains within their bounds allow it together with the
                other dependencies, or it closed a cycle among simultaneous
                events. Kept (and counted), never an edge of the event graph
  unsupported   an alternative no builder chose, with no evidence beyond
                being possible in time (cp_alternatives.py adds these)

The default view is the selected candidates in insertion order: exactly
criticalpath.build_edges(), so every result computed from the edges is
unchanged. The statuses are evidence classes, not probabilities: nothing
calibrates them (edge_calibration.py says how that would be done and
reports "unquantified" until it is).
"""
from __future__ import annotations

import io
from collections import Counter
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..core.trace import Trace

# ── vocabulary ───────────────────────────────────────────────────────────────

OBSERVED, INFERRED, AMBIGUOUS, CONTRADICTED, UNSUPPORTED = range(5)
STATUSES = ("observed", "inferred", "ambiguous", "contradicted", "unsupported")

EXACTLY_ONE, AT_MOST_ONE, ANY_SUBSET = range(3)
CARDINALITIES = ("exactly_one", "at_most_one", "any_subset")

# kind -> (class, what it is, kinds it is derived from)
EVIDENCE: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "hook_id": ("direct", "an id the hook recorded on both ends (sid= / psid=)", ()),
    "correlation_id": ("direct", "the runtime's correlation id (lid= / corr=) on the host call and its "
                                 "device work", ()),
    "runtime_order": ("direct", "a runtime ordering rule over events on one clock (a thread's order, a "
                                "stream's FIFO, the work a sync call synchronizes)", ()),
    "region_id": ("direct", "an OpenMP parallel-region id on the region and on its team's tasks",
                  ("hook_id",)),
    "mpi_status": ("direct", "the source and tag MPI_Status reported for a wildcard receive", ()),
    "posting_order": ("inferred", "the k-th send of a (sender, receiver, tag, communicator) channel paired "
                                  "with its k-th receive: MPI's non-overtaking rule over the recorded "
                                  "calls", ()),
    "status_then_order": ("inferred", "a wildcard receive's channel from MPI_Status, then posting order "
                                      "within it", ("mpi_status", "posting_order")),
    "call_order": ("inferred", "the k-th collective / barrier call of every member: the calling-order "
                               "rule over the recorded calls", ()),
    "root_data_flow": ("inferred", "a rooted collective's data flow from or to the root its members "
                                   "recorded (wroot=)", ("call_order",)),
    "nccl_seq": ("direct", "the NCCL communicator id and per-communicator sequence number the hook "
                           "recorded", ("hook_id",)),
    "time_overlap": ("inferred", "calls grouped because their time intervals overlap", ()),
    "task_id": ("direct", "the OMPT task ids on a task's creation, its execution and the join that waits "
                          "for it, or a recorded task dependence", ("hook_id",)),
    "lock_release": ("inferred", "the same lock's release just before a contended acquisition, by another "
                                 "thread, on one clock", ()),
    "nccl_pseq": ("direct", "the per-peer send / receive numbers the NCCL hook recorded (pseq=) and NCCL's "
                            "in-order pairing", ("hook_id", "runtime_order")),
    "several_candidates": ("against", "the identifying data matched more than one candidate", ()),
    "event_loss": ("against", "the process lost events: an ordinal pairing may be shifted", ()),
    "time_contradiction": ("against", "the source event lies after the destination event beyond the "
                                      "clock tolerance", ()),
    "cycle": ("against", "the edge closed a cycle among simultaneous events", ()),
    "clock_bound": ("against", "no offsets of the clock domains within their bounds satisfy it together with "
                               "the other dependencies (cp_clock.py)", ()),
    "time_feasible": ("inferred", "an alternative no builder chose: possible in time, nothing else for it", ()),
    "alignment_shift": ("inferred", "a point-to-point channel whose sends and receives do not pair one to one as "
                                    "recorded: the pairing shifted by one", ("posting_order",)),
    "clock_order": ("inferred", "a rendezvous member that arrived within the clocks' uncertainty of the "
                                "last arriver: it may have been the last", ()),
    "shared_correlation": ("inferred", "another host submission with the device operation's correlation id",
                           ("correlation_id",)),
}
EVIDENCE_KINDS = tuple(EVIDENCE)
assert len(EVIDENCE_KINDS) <= 32, "ev_for / ev_against are 32-bit masks"
_BIT = {k: 1 << i for i, k in enumerate(EVIDENCE_KINDS)}

# cp_model provenance -> evidence kinds supporting an edge with it
PROV_EVIDENCE: dict[str, tuple[str, ...]] = {
    "thread_timeline": ("runtime_order",),
    "device_interval": ("runtime_order",),
    "stream_fifo": ("runtime_order",),
    "event_record": ("runtime_order",),
    "sync_target": ("runtime_order",),
    "correlation_id": ("correlation_id",),
    "explicit_id": ("hook_id",),
    "fork_join": ("region_id",),
    "posting_order": ("posting_order",),
    "status_resolved": ("status_then_order",),
    "call_order": ("call_order",),
    "rooted_flow": ("root_data_flow",),
    "task_id": ("task_id",),
    "lock_handoff": ("lock_release",),
    "nccl_pseq": ("nccl_pseq",),
    "nccl_seq": ("nccl_seq",),
    "time_overlap": ("time_overlap",),
}

GROUP_KINDS = ("single", "p2p", "launch", "device_wait", "arrival")
_G = {k: i for i, k in enumerate(GROUP_KINDS)}

CAND_DTYPE = np.dtype([("dst", np.int64), ("src", np.int64), ("kind", np.int8), ("conf", np.int8),
                       ("prov", np.int8), ("status", np.int8), ("selected", np.int8), ("in_model", np.int8),
                       ("group", np.int64), ("ev_for", np.uint32), ("ev_against", np.uint32)])
GROUP_DTYPE = np.dtype([("kind", np.int8), ("card", np.int8), ("key", np.int64)])

VERSION = "ev-1"


def mask(*kinds: str) -> int:
    m = 0
    for k in kinds:
        m |= _BIT[k]
    return m


def kinds_of(m: int) -> list[str]:
    return [k for k in EVIDENCE_KINDS if m & _BIT[k]]


def _fallback_prov(kind: str, conf: str) -> str:
    """Provenance of a span-level edge the event graph did not map."""
    return {"p2p": "posting_order", "launch": "correlation_id", "device_wait": "sync_target",
            "device_sync": "sync_target", "sequential": "thread_timeline",
            "explicit_span_id": "explicit_id", "task_join": "task_id", "task_dep": "task_id",
            "lock_order": "lock_handoff"}.get(kind) or (
        "call_order" if conf in ("certain", "high") else "time_overlap")


# ── the set ──────────────────────────────────────────────────────────────────

class EvidenceSet:
    """Candidates (CAND_DTYPE) and match groups (GROUP_DTYPE). The first
    n_selected candidates are the builders' edges, in their order."""

    def __init__(self, cand: np.ndarray, groups: np.ndarray, n_selected: int) -> None:
        self.cand = cand
        self.groups = groups
        self.n_selected = n_selected

    def __len__(self) -> int:
        return len(self.cand)

    # the default view
    def default_view(self) -> np.ndarray:
        """The builders' edges (criticalpath EDGE_DTYPE), in insertion order."""
        from ..core.store import EDGE_DTYPE
        sel = self.cand[:self.n_selected]
        out = np.empty(len(sel), dtype=EDGE_DTYPE)
        for f in ("dst", "src", "kind", "conf"):
            out[f] = sel[f]
        return out

    def status(self, i: int) -> str:
        return STATUSES[int(self.cand["status"][i])]

    def group_members(self, gid: int) -> np.ndarray:
        return np.flatnonzero(self.cand["group"] == gid)

    def summary(self) -> dict:
        """Counts by status, by edge kind and status, and of the groups."""
        from .criticalpath import EDGE_KINDS
        c = self.cand
        by_status = Counter(STATUSES[s] for s in c["status"].tolist())
        by_kind: dict[str, Counter] = {}
        for k, s in zip(c["kind"].tolist(), c["status"].tolist()):
            by_kind.setdefault(EDGE_KINDS[k], Counter())[STATUSES[s]] += 1
        sizes = np.bincount(c["group"], minlength=len(self.groups)) if len(c) else np.zeros(0, int)
        multi = int(np.sum(sizes > 1))
        return {"candidates": len(c), "selected": self.n_selected,
                "by_status": dict(by_status),
                "by_kind": {k: dict(v) for k, v in sorted(by_kind.items())},
                "groups": len(self.groups),
                "groups_by_cardinality": dict(Counter(CARDINALITIES[x] for x in self.groups["card"].tolist())),
                "groups_with_alternatives": multi,
                "against": dict(Counter(k for m in c["ev_against"].tolist() if m for k in kinds_of(m)))}

    def explain(self, i: int, spans=None) -> dict:
        """Candidate i in words: its edge, status, group, and the evidence
        tree for and against it. With `spans` (ordinal -> SpanEvent), each
        supporting kind also lists the raw tags it was read from."""
        from .cp_model import PROVS
        from .criticalpath import EDGE_CONFS, EDGE_KINDS
        r = self.cand[i]
        g = self.groups[int(r["group"])]
        src, dst = int(r["src"]), int(r["dst"])
        records = _raw_records(spans, src, dst) if spans is not None else {}

        def node(k: str) -> dict:
            cls, what, derived = EVIDENCE[k]
            out = {"kind": k, "class": cls, "what": what,
                   "derived_from": [node(x) for x in derived]}
            if k in records:
                out["record"] = records[k]
            return out

        return {"candidate": i, "src": src, "dst": dst, "kind": EDGE_KINDS[int(r["kind"])],
                "tier": EDGE_CONFS[int(r["conf"])],
                "provenance": PROVS[int(r["prov"])] if r["prov"] >= 0 else None,
                "status": STATUSES[int(r["status"])], "selected": bool(r["selected"]),
                "in_model": bool(r["in_model"]),
                "group": {"id": int(r["group"]), "kind": GROUP_KINDS[int(g["kind"])],
                          "cardinality": CARDINALITIES[int(g["card"])],
                          "members": len(self.group_members(int(r["group"])))},
                "for": [node(k) for k in kinds_of(int(r["ev_for"]))],
                "against": [node(k) for k in kinds_of(int(r["ev_against"]))]}

    # persistence
    def to_bytes(self) -> bytes:
        buf = io.BytesIO()
        np.savez(buf, cand=self.cand, groups=self.groups, n_selected=np.array([self.n_selected]))
        return buf.getvalue()

    @classmethod
    def from_bytes(cls, data: bytes) -> "EvidenceSet":
        z = np.load(io.BytesIO(data))
        return cls(z["cand"], z["groups"], int(z["n_selected"][0]))


_RECORD_TAGS = {
    "hook_id": ("sid", "psid"), "region_id": ("psid",), "correlation_id": ("lid", "corr", "corr2"),
    "mpi_status": ("rpeer", "rtag", "rmatches", "rwmatches", "rwpeer"),
    "posting_order": ("peer", "wpeer", "tag", "commid"), "status_then_order": ("rpeer", "rtag", "commid"),
    "call_order": ("type", "commid", "team"), "nccl_seq": ("commid", "seq"),
    "runtime_order": ("stream", "queue", "dev", "hdev"),
}


def _raw_records(spans, src: int, dst: int) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for k, keys in _RECORD_TAGS.items():
        rows = []
        for role, j in (("src", src), ("dst", dst)):
            s = spans[j]
            vals = {t: s.tags[t] for t in keys if t in s.tags}
            if "sid" in keys and s.span_id:
                vals["sid"] = s.span_id
            if "psid" in keys and s.parent_span_id:
                vals["psid"] = s.parent_span_id
            if vals:
                rows.append({"role": role, "span": s.name, **vals})
        if rows:
            out[k] = rows
    return out


# ── building ─────────────────────────────────────────────────────────────────

class _Groups:
    def __init__(self) -> None:
        self.kind: list[int] = []
        self.card: list[int] = []
        self.key: list[int] = []
        self._by: dict[tuple, int] = {}

    def get(self, kind: str, card: int, key: int, shared: bool) -> int:
        if shared:
            gid = self._by.get((kind, key))
            if gid is not None:
                return gid
        gid = len(self.kind)
        self.kind.append(_G[kind])
        self.card.append(card)
        self.key.append(key)
        if shared:
            self._by[(kind, key)] = gid
        return gid

    def array(self) -> np.ndarray:
        out = np.empty(len(self.kind), dtype=GROUP_DTYPE)
        out["kind"], out["card"], out["key"] = self.kind, self.card, self.key
        return out


def evidence_from_graph(edges, g) -> EvidenceSet:
    """The evidence for a trace's persisted span-level edges (`edges`,
    EDGE_DTYPE) given the event graph `g` built from them
    (cp_model.build_event_graph): provenance from the event edges each one
    became, contradictions from the ones the graph dropped (timestamps) or
    disabled (cycles), groups by what each edge answers."""
    from .cp_model import CONFS, PROVS, edge_class, _P
    from .criticalpath import EDGE_CONFS, EDGE_KINDS
    m = len(edges)
    prov = np.full(m, -1, dtype=np.int8)
    in_model = np.zeros(m, dtype=np.int8)
    cyc = np.zeros(m, dtype=bool)
    origin = np.frombuffer(g.origin, dtype=np.int64) if len(g.origin) else np.zeros(0, np.int64)
    eprov = np.frombuffer(g.prov, dtype=np.int8) if len(g.prov) else np.zeros(0, np.int8)
    enabled = np.frombuffer(bytes(g.enabled), dtype=np.uint8) if len(g.enabled) else np.zeros(len(origin), np.uint8)
    dep = origin >= 0
    if dep.any():
        o = origin[dep]
        # the first event edge of each span-level edge gives its provenance
        first = np.unique(o, return_index=True)
        prov[first[0]] = eprov[dep][first[1]]
        on = enabled[dep].astype(bool)
        in_model[np.unique(o[on])] = 1
        off = np.unique(o[~on])
        cyc[off] = True
    dropped = np.zeros(m, dtype=bool)
    if len(g.dropped_edges):
        dropped[np.frombuffer(g.dropped_edges, dtype=np.int64)] = True
    cyc &= in_model == 0                     # every event edge of it disabled
    clock = np.zeros(m, dtype=bool)
    for e in getattr(g, "clock_disabled", ()):
        if origin[e] >= 0:
            clock[origin[e]] = True
    clock &= cyc
    cyc &= ~clock

    kinds = edges["kind"].astype(np.int64)
    confs = edges["conf"].astype(np.int64)
    dsts = edges["dst"].astype(np.int64)
    groups = _Groups()
    cand = np.zeros(m, dtype=CAND_DTYPE)
    cand["dst"], cand["src"], cand["kind"], cand["conf"] = edges["dst"], edges["src"], edges["kind"], edges["conf"]
    cand["selected"] = 1
    cand["in_model"] = in_model
    # launch / arrival: several selected edges into one destination compete
    launch_k, arrival_k = EDGE_KINDS.index("launch"), EDGE_KINDS.index("arrival")
    p2p_k, wait_k = EDGE_KINDS.index("p2p"), EDGE_KINDS.index("device_wait")
    low = EDGE_CONFS.index("low")
    for i in range(m):
        k, c, v = int(kinds[i]), int(confs[i]), int(dsts[i])
        kind, conf = EDGE_KINDS[k], EDGE_CONFS[c]
        p = int(prov[i])
        if p < 0:
            p = _P[_fallback_prov(kind, conf)]
            prov[i] = p
        pname = PROVS[p]
        if k == launch_k:
            gid = groups.get("launch", EXACTLY_ONE, v, True)
        elif k == arrival_k:
            gid = groups.get("arrival", EXACTLY_ONE, v, True)
        elif k == wait_k:
            gid = groups.get("device_wait", ANY_SUBSET, v, True)
        elif k == p2p_k:
            gid = groups.get("p2p", AT_MOST_ONE if c == low else EXACTLY_ONE, v, False)
        else:
            gid = groups.get("single", ANY_SUBSET, v, False)
        cand["group"][i] = gid
        ev_for = mask(*PROV_EVIDENCE[pname])
        against = 0
        if c == low:
            against |= _BIT["event_loss"]
        if dropped[i]:
            against |= _BIT["time_contradiction"]
        if cyc[i]:
            against |= _BIT["cycle"]
        if clock[i]:
            against |= _BIT["clock_bound"]
        cand["ev_for"][i] = ev_for
        cand["ev_against"][i] = against
        if dropped[i] or cyc[i] or clock[i]:
            st = CONTRADICTED
        elif edge_class(pname, CONFS[c]) == "direct":
            st = OBSERVED
        else:
            st = INFERRED
        cand["status"][i] = st
    cand["prov"] = prov
    garr = groups.array()
    _mark_ambiguous(cand, garr)
    return EvidenceSet(cand, garr, m)


def _mark_ambiguous(cand: np.ndarray, groups: np.ndarray) -> None:
    """A group that needs one candidate but holds several live ones: each
    observed / inferred one of those is ambiguous (alternatives stay
    unsupported, contradicted ones contradicted)."""
    if not len(cand):
        return
    live = cand["status"] != CONTRADICTED
    sizes = np.bincount(cand["group"][live], minlength=len(groups))
    one = (groups["card"] == EXACTLY_ONE) | (groups["card"] == AT_MOST_ONE)
    crowded = one & (sizes > 1)
    hit = live & crowded[cand["group"]] & ((cand["status"] == OBSERVED) | (cand["status"] == INFERRED))
    cand["status"][hit] = AMBIGUOUS
    cand["ev_against"][hit] |= _BIT["several_candidates"]


def with_alternatives(es: EvidenceSet, alts: list[tuple[int, int, int, int, int, str]],
                      contested: list[int] = ()) -> EvidenceSet:
    """es plus alternative candidates (dst, src, kind code, conf code, group
    id, evidence kind): unsupported, not selected, not in the model; the
    groups they join become ambiguous, and so do the `contested` candidates
    (an at_most_one group whose answer may be "none of them")."""
    if not alts and not contested:
        return es
    extra = np.zeros(len(alts), dtype=CAND_DTYPE)
    for i, (dst, src, kind, conf, gid, why) in enumerate(alts):
        extra[i] = (dst, src, kind, conf, -1, UNSUPPORTED, 0, 0, gid, mask(why, "time_feasible"), 0)
    cand = np.concatenate([es.cand, extra])
    _mark_ambiguous(cand, es.groups)
    for i in contested:
        if cand["status"][i] in (OBSERVED, INFERRED):
            cand["status"][i] = AMBIGUOUS
            cand["ev_against"][i] |= _BIT["several_candidates"]
    return EvidenceSet(cand, es.groups, es.n_selected)


def build_evidence(trace: "Trace", graph=None, g=None) -> EvidenceSet:
    """The trace's dependency evidence (persisted span-level edges + the
    event graph built from them)."""
    from . import cp_model
    from .criticalpath import load_or_build_graph
    if graph is None:
        graph = load_or_build_graph(trace)
    if g is None:
        g = cp_model.build_event_graph(trace, graph)
    return evidence_from_graph(graph.edges, g)


def evidence_version() -> str:
    from .criticalpath import edges_version
    return f"{VERSION}|{edges_version()}"


def load_or_build_evidence(trace: "Trace", graph=None, g=None) -> EvidenceSet:
    """Persisted evidence when the store has it for the current events and
    versions, else built and persisted."""
    store = trace.store
    version = evidence_version()
    data = store.load_blob("evidence", version)
    if data is not None:
        return EvidenceSet.from_bytes(data)
    ev = build_evidence(trace, graph, g)
    store.save_blob("evidence", version, ev.to_bytes())
    return ev
