"""
Structure- and causality-aware comparison of two runs (BEFORE = baseline,
AFTER = candidate), built on src/analysis/projection.py.

    1. Project both traces (normalized call paths, roles, phases, measured
       node values, node-level dependency edges, critical-path credit).
    2. Align the phase sequences (prologue / iterations / epilogue, or
       top-level segments) with a monotone alignment that maximizes phase
       similarity -- inserted or removed phases become gaps.
    3. Within every aligned phase pair, match nodes: identical identity
       first (path + roles), then by normalized call path, source location,
       roles and graph neighborhood (never by name alone).
    4. Aggregate matched nodes across phases into identities and explain
       each change: measured deltas (count, own time, queue delay, overlap)
       are split into increased work / more invocations / queueing /
       synchronization / communication; graph-derived evidence adds lost
       overlap, changed dependency edges and movement onto the critical
       path, and follows dependency edges upstream to the node that
       actually moved (wait propagation).
    5. Rank causal contributors by the change of critical-path time they
       account for (time credited on the path + idle time on the path
       blamed on them) -- the graph-derived decomposition of the wall-time
       change.

Every value is labeled by how it was obtained: "measured" (timestamps of
both runs), "derived" (critical path / dependency graph), "heuristic"
(phase detection, alignment, node matching -- each with a confidence) or
"unavailable" (with the reason). The alignment confidence decides the
method: phase-aligned comparison, else whole-run structural comparison,
else the (category, name) aggregate comparison of src/analysis/compare.py,
which is always included for compatibility. Single-run traces carry no
variance information: deltas are screened by the disclosed noise floor of
compare.py, never by a significance test, and no LLM is involved.
"""
from __future__ import annotations

import difflib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import compare as cmp
from . import dashboard as dash
from .projection import (
    WAIT_BUCKETS, NodeKey, Phase, ProjectedNode, TraceProjection, build_projection, source_key,
)

COMPARE_SCHEMA = "hprofiler-compare/1"
DEFAULT_MIN_CONFIDENCE = 0.5
MIN_PHASE_SIMILARITY = 0.4
MIN_NODE_SCORE = 0.55
MAX_CHAIN = 6

CAUSES = ("increased_work", "more_invocations", "queueing", "synchronization", "communication",
          "lost_overlap", "changed_dependency", "moved_onto_critical_path")
CAUSE_LABELS = {
    "increased_work": "increased work",
    "more_invocations": "more invocations",
    "queueing": "queueing",
    "synchronization": "synchronization",
    "communication": "communication",
    "lost_overlap": "lost overlap",
    "changed_dependency": "changed dependency edge",
    "moved_onto_critical_path": "moved onto the critical path",
    "new_work": "new work",
}
IMPROVEMENT_LABELS = {
    "increased_work": "less work",
    "more_invocations": "fewer invocations",
    "queueing": "less queueing",
    "synchronization": "less synchronization",
    "communication": "less communication",
    "lost_overlap": "gained overlap",
    "changed_dependency": "changed dependency edge",
    "moved_onto_critical_path": "moved off the critical path",
    "new_work": "removed work",
}
_CONF_RANK = {c: i for i, c in enumerate(("certain", "high", "medium", "low"))}
_MEASURED_CAUSES = ("increased_work", "more_invocations", "queueing", "synchronization", "communication")
_GATING_KINDS = frozenset({"p2p", "arrival", "device_wait", "device_sync", "launch", "explicit_span_id",
                           "sequential"})


# ── phase alignment ──────────────────────────────────────────────────────────

def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def phase_similarity(a: Phase, b: Phase) -> float:
    s = 0.5 * _jaccard(a.driver_tokens, b.driver_tokens) + 0.5 * _jaccard(a.node_paths, b.node_paths)
    if a.kind != b.kind and not {a.kind, b.kind} <= {"iteration", "segment", "whole"}:
        s *= 0.5
    return s


@dataclass
class PhasePair:
    index: int
    a: int | None                 # baseline phase index
    b: int | None                 # candidate phase index
    similarity: float
    status: str                   # matched / inserted / removed
    ambiguous: bool = False       # identical neighbours: which one was inserted is arbitrary


def align_phases(pa: list[Phase], pb: list[Phase]) -> tuple[list[PhasePair], float]:
    """Monotone alignment maximizing total similarity (+ a small duration-
    similarity bonus that only decides between otherwise equal placements);
    pairs below MIN_PHASE_SIMILARITY are never matched. Returns the pairs in
    order and the phase-alignment confidence: the time-weighted similarity
    of matched pairs, where an unmatched phase of a shape that also occurs
    among matched phases (an extra iteration) counts as explained and an
    unmatched phase of a new shape does not."""
    cls_a, cls_b = _classes(pa), _classes(pb)
    sim_cache: dict[tuple[int, int], float] = {}

    def sim(i: int, j: int) -> float:
        k = (cls_a[i], cls_b[j])
        v = sim_cache.get(k)
        if v is None:
            v = sim_cache[k] = phase_similarity(pa[i], pb[j])
        return v

    n, m = len(pa), len(pb)
    score = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        row, prev = score[i], score[i - 1]
        a = pa[i - 1]
        for j in range(1, m + 1):
            best = row[j - 1] if row[j - 1] >= prev[j] else prev[j]
            s = sim(i - 1, j - 1)
            if s >= MIN_PHASE_SIMILARITY:
                b = pb[j - 1]
                da, db = a.duration_ns, b.duration_ns
                bonus = 0.01 * (min(da, db) / max(da, db) if max(da, db) > 0 else 1.0)
                d = prev[j - 1] + s + bonus
                if d > best:
                    best = d
            row[j] = best
    pairs: list[tuple[int | None, int | None, float]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            s = sim(i - 1, j - 1)
            # gaps first on ties: an ambiguous extra phase is reported at the end
            if j > 0 and score[i][j] == score[i][j - 1]:
                pairs.append((None, j - 1, 0.0))
                j -= 1
                continue
            if score[i][j] == score[i - 1][j]:
                pairs.append((i - 1, None, 0.0))
                i -= 1
                continue
            pairs.append((i - 1, j - 1, s))
            i, j = i - 1, j - 1
        elif j > 0:
            pairs.append((None, j - 1, 0.0))
            j -= 1
        else:
            pairs.append((i - 1, None, 0.0))
            i -= 1
    pairs.reverse()
    out: list[PhasePair] = []
    matched_cls_a = {cls_a[a] for a, b, _ in pairs if a is not None and b is not None}
    matched_cls_b = {cls_b[b] for a, b, _ in pairs if a is not None and b is not None}
    total = explained = 0.0
    for k, (a, b, s) in enumerate(pairs):
        if a is not None and b is not None:
            w = pa[a].duration_ns + pb[b].duration_ns
            total += w
            explained += s * w
            out.append(PhasePair(k, a, b, s, "matched"))
        elif a is not None:
            w = pa[a].duration_ns
            total += w
            known = cls_a[a] in matched_cls_a
            explained += w if known else 0.0
            out.append(PhasePair(k, a, None, 0.0, "removed",
                                 ambiguous=known and _same_class_neighbour(pairs, k, cls_a, side=0)))
        else:
            w = pb[b].duration_ns
            total += w
            known = cls_b[b] in matched_cls_b
            explained += w if known else 0.0
            out.append(PhasePair(k, None, b, 0.0, "inserted",
                                 ambiguous=known and _same_class_neighbour(pairs, k, cls_b, side=1)))
    conf = explained / total if total > 0 else (1.0 if out else 0.0)
    return out, conf


def _classes(phases: list[Phase]) -> list[int]:
    ids: dict[tuple, int] = {}
    return [ids.setdefault((p.kind, p.driver_tokens, p.node_paths), len(ids)) for p in phases]


def _same_class_neighbour(pairs, k: int, cls: list[int], side: int) -> bool:
    me = pairs[k][side]
    for nb in (k - 1, k + 1):
        if 0 <= nb < len(pairs) and pairs[nb][side] is not None and cls[pairs[nb][side]] == cls[me]:
            return True
    return False


# ── node matching ────────────────────────────────────────────────────────────

def _neighbourhood(proj: TraceProjection, nid: int) -> frozenset:
    ins, outs = proj.adjacency()
    nodes = proj.nodes
    return frozenset([("in", kind, nodes[o].key.leaf) for o, kind, _c, _cf in ins.get(nid, ())] +
                     [("out", kind, nodes[o].key.leaf) for o, kind, _c, _cf in outs.get(nid, ())])


def node_score(a: ProjectedNode, b: ProjectedNode, na: frozenset, nb: frozenset) -> float:
    """Structural similarity of two nodes (0..1): normalized call path 40%,
    roles 20%, source location 20%, graph neighbourhood 20%; a different
    leaf name only matches through the same source location (a rename)."""
    ka, kb = a.key, b.key
    if ka.backend != kb.backend:
        return 0.0
    path = 1.0 if ka.path == kb.path else difflib.SequenceMatcher(None, ka.path, kb.path).ratio()
    ra, rb = ka.roles(), kb.roles()
    roles = sum(1 for k in ("rank", "thread", "stream", "device", "comm") if ra[k] == rb[k]) / 5
    if a.source and b.source:
        src = 1.0 if source_key(a.source) == source_key(b.source) else 0.0
    else:
        src = 0.5
    neigh = _jaccard(na, nb) if (na or nb) else 0.5
    score = 0.4 * path + 0.2 * roles + 0.2 * src + 0.2 * neigh
    if ka.leaf != kb.leaf:
        score = score * 0.8 if (a.source and src == 1.0) else 0.0
    return score


def match_nodes(pa: TraceProjection, nodes_a: list[ProjectedNode], pb: TraceProjection,
                nodes_b: list[ProjectedNode]) -> list[tuple[ProjectedNode | None, ProjectedNode | None, float]]:
    """Pairs (a, b, score) covering every node of both lists once; a side is
    None for a node with no counterpart."""
    out = []
    by_key_b = {n.key: n for n in nodes_b}
    used_b: set[int] = set()
    rest_a = []
    for a in nodes_a:
        b = by_key_b.get(a.key)
        if b is not None and b.id not in used_b:
            out.append((a, b, 1.0))
            used_b.add(b.id)
        else:
            rest_a.append(a)
    rest_b = [b for b in nodes_b if b.id not in used_b]
    if rest_a and rest_b:
        by_leaf: dict[str, list[ProjectedNode]] = defaultdict(list)
        by_src: dict[str, list[ProjectedNode]] = defaultdict(list)
        for b in rest_b:
            by_leaf[b.key.leaf].append(b)
            if b.source:
                by_src[source_key(b.source)].append(b)
        nh_a: dict[int, frozenset] = {}
        nh_b: dict[int, frozenset] = {}
        cands = []
        for a in rest_a:
            pool = {b.id: b for b in by_leaf.get(a.key.leaf, ())}
            if a.source:
                pool.update((b.id, b) for b in by_src.get(source_key(a.source), ()))
            for b in pool.values():
                if a.id not in nh_a:
                    nh_a[a.id] = _neighbourhood(pa, a.id)
                if b.id not in nh_b:
                    nh_b[b.id] = _neighbourhood(pb, b.id)
                s = node_score(a, b, nh_a[a.id], nh_b[b.id])
                if s >= MIN_NODE_SCORE:
                    cands.append((s, a.self_ns + b.self_ns, a.id, b.id, a, b))
        cands.sort(key=lambda c: (-c[0], -c[1], c[2], c[3]))
        used_a: set[int] = set()
        for s, _w, ai, bi, a, b in cands:
            if ai in used_a or bi in used_b:
                continue
            used_a.add(ai)
            used_b.add(bi)
            out.append((a, b, s))
        rest_a = [a for a in rest_a if a.id not in used_a]
    for a in rest_a:
        out.append((a, None, 0.0))
    for b in nodes_b:
        if b.id not in used_b:
            out.append((None, b, 0.0))
    return out


# ── identities ───────────────────────────────────────────────────────────────

_FIELDS = ("count", "total_ns", "self_ns", "queue_ns", "queued", "overlap_ns", "cp_ns", "blame_ns")


@dataclass
class _Side:
    count: int = 0
    total_ns: int = 0
    self_ns: int = 0
    queue_ns: int = 0
    queued: int = 0
    overlap_ns: int = 0
    cp_ns: int = 0
    blame_ns: int = 0
    first_start: int | None = None
    last_end: int | None = None

    def add(self, n: ProjectedNode) -> None:
        for f in _FIELDS:
            setattr(self, f, getattr(self, f) + getattr(n, f))
        self.first_start = n.first_start if self.first_start is None else min(self.first_start, n.first_start)
        self.last_end = n.last_end if self.last_end is None else max(self.last_end, n.last_end)

    @property
    def own_ns(self) -> int:
        return self.self_ns + self.queue_ns

    @property
    def critical_ns(self) -> int:
        return self.cp_ns + self.blame_ns


@dataclass
class Identity:
    id: int
    ka: NodeKey | None
    kb: NodeKey | None
    a: _Side = field(default_factory=_Side)
    b: _Side = field(default_factory=_Side)
    score_w: float = 0.0
    weight: float = 0.0
    pairs: dict[int, tuple[ProjectedNode | None, ProjectedNode | None]] = field(default_factory=dict)
    extra_a: int = 0              # invocations in baseline phases with no counterpart
    extra_b: int = 0
    node_a: ProjectedNode | None = None   # representative nodes (display, source, bucket)
    node_b: ProjectedNode | None = None

    @property
    def key(self) -> NodeKey:
        return self.kb or self.ka

    @property
    def node(self) -> ProjectedNode:
        return self.node_b or self.node_a

    @property
    def match_score(self) -> float:
        if self.ka is None or self.kb is None:
            return 0.0
        return self.score_w / self.weight if self.weight else 1.0


class _Comparison:
    def __init__(self, pa: TraceProjection, pb: TraceProjection, pairs: list[PhasePair],
                 noise_pct: float, noise_ns: float) -> None:
        self.pa, self.pb, self.pairs = pa, pb, pairs
        self.noise_pct, self.noise_ns = noise_pct, noise_ns
        self.identities: list[Identity] = []
        self.by_pair: dict[tuple, Identity] = {}
        self.ident_a: dict[int, Identity] = {}     # baseline node id -> identity
        self.ident_b: dict[int, Identity] = {}
        self.match_conf = 1.0

    def ident(self, ka: NodeKey | None, kb: NodeKey | None) -> Identity:
        k = (ka, kb)
        i = self.by_pair.get(k)
        if i is None:
            i = Identity(len(self.identities), ka, kb)
            self.identities.append(i)
            self.by_pair[k] = i
        return i

    def run(self) -> None:
        pa, pb = self.pa, self.pb
        cache: dict[tuple, list] = {}
        cls_a = _node_classes(pa)
        cls_b = _node_classes(pb)
        matched_w = total_w = 0.0
        for pp in self.pairs:
            if pp.status != "matched":
                continue
            na, nb = pa.phase_nodes(pp.a), pb.phase_nodes(pp.b)
            ck = (cls_a[pp.a], cls_b[pp.b])
            mapping = cache.get(ck)
            if mapping is None:
                pairs = match_nodes(pa, na, pb, nb)
                mapping = cache[ck] = [(a.key if a else None, b.key if b else None, s) for a, b, s in pairs]
            idx_a = {n.key: n for n in na}
            idx_b = {n.key: n for n in nb}
            for ka, kb, s in mapping:
                a = idx_a.get(ka) if ka is not None else None
                b = idx_b.get(kb) if kb is not None else None
                ident = self.ident(ka, kb)
                w = (a.self_ns if a else 0) + (b.self_ns if b else 0)
                total_w += w
                if a is not None and b is not None:
                    matched_w += s * w
                    ident.score_w += s * w
                    ident.weight += w
                self.attach(ident, pp.index, a, b)
        has_nodes = any(pa.nodes) or any(pb.nodes)
        self.match_conf = matched_w / total_w if total_w > 0 else (0.0 if has_nodes else 1.0)
        # phases without a counterpart: their nodes join the identity with
        # the same key (an extra iteration adds invocations of known work)
        by_ka: dict[NodeKey, Identity] = {}
        by_kb: dict[NodeKey, Identity] = {}
        for ident in self.identities:
            if ident.ka is not None:
                by_ka.setdefault(ident.ka, ident)
            if ident.kb is not None:
                by_kb.setdefault(ident.kb, ident)
        for pp in self.pairs:
            if pp.status == "removed":
                for a in pa.phase_nodes(pp.a):
                    ident = by_ka.get(a.key) or self.ident(a.key, None)
                    by_ka.setdefault(a.key, ident)
                    ident.extra_a += a.count
                    self.attach(ident, pp.index, a, None)
            elif pp.status == "inserted":
                for b in pb.phase_nodes(pp.b):
                    ident = by_kb.get(b.key) or self.ident(None, b.key)
                    by_kb.setdefault(b.key, ident)
                    ident.extra_b += b.count
                    self.attach(ident, pp.index, None, b)

    def attach(self, ident: Identity, pair: int, a: ProjectedNode | None, b: ProjectedNode | None) -> None:
        prev = ident.pairs.get(pair)
        if prev is not None:   # a removed/inserted phase merged into an identity twice
            a = a or prev[0]
            b = b or prev[1]
        ident.pairs[pair] = (a, b)
        if a is not None and (prev is None or prev[0] is not a):
            ident.a.add(a)
            self.ident_a[a.id] = ident
            if ident.node_a is None or a.self_ns > ident.node_a.self_ns:
                ident.node_a = a
        if b is not None and (prev is None or prev[1] is not b):
            ident.b.add(b)
            self.ident_b[b.id] = ident
            if ident.node_b is None or b.self_ns > ident.node_b.self_ns:
                ident.node_b = b

    # identity-level dependency edges
    def edge_sets(self) -> tuple[dict, dict]:
        def lift(proj: TraceProjection, ident_of: dict[int, Identity]) -> dict:
            out: dict[tuple[int, int, str], str] = {}
            for s, d, kind, _count, conf in proj.edges:
                i, j = ident_of.get(s), ident_of.get(d)
                if i is None or j is None or i is j:
                    continue
                out.setdefault((i.id, j.id, kind), conf)
            return out
        return lift(self.pa, self.ident_a), lift(self.pb, self.ident_b)


def _node_classes(proj: TraceProjection) -> dict[int, int]:
    ids: dict[frozenset, int] = {}
    return {p.index: ids.setdefault(frozenset(n.key for n in proj.phase_nodes(p.index)), len(ids))
            for p in proj.phases}


# ── whole-run collapse (fallback level 2) ────────────────────────────────────

def collapse(proj: TraceProjection) -> TraceProjection:
    """The projection with every phase merged into one: nodes with the same
    identity summed across phases, edges and critical path remapped."""
    whole = Phase(0, "whole", "whole run", proj.start_ns, proj.end_ns,
                  critical_ns=proj.critical_total_ns)
    nodes: list[ProjectedNode] = []
    by_key: dict[NodeKey, ProjectedNode] = {}
    remap: dict[int, int] = {}
    for n in proj.nodes:
        m = by_key.get(n.key)
        if m is None:
            m = ProjectedNode(len(nodes), 0, n.key, n.category, n.raw_name, n.bucket)
            m.source = n.source
            m.min_ns, m.max_ns = n.min_ns, n.max_ns
            m.first_start, m.last_end = n.first_start, n.last_end
            nodes.append(m)
            by_key[n.key] = m
        else:
            m.min_ns = min(m.min_ns, n.min_ns)
            m.max_ns = max(m.max_ns, n.max_ns)
            m.first_start = min(m.first_start, n.first_start)
            m.last_end = max(m.last_end, n.last_end)
            m.source = m.source or n.source
        for f in _FIELDS + ("cp_count",):
            setattr(m, f, getattr(m, f) + getattr(n, f))
        m.threads |= n.threads
        m.timing = n.timing if m.timing in ("", n.timing) else "mixed"
        remap[n.id] = m.id
    edges: dict[tuple[int, int, str], list] = {}
    for s, d, kind, count, conf in proj.edges:
        ms, md = remap[s], remap[d]
        if ms == md:
            continue
        e = edges.setdefault((ms, md, kind), [0, conf])
        e[0] += count
        if _CONF_RANK.get(conf, 9) < _CONF_RANK.get(e[1], 9):
            e[1] = conf
    from .projection import CPSegment
    segs = [CPSegment(remap.get(sg.node, -1) if sg.node >= 0 else -1, sg.credited_ns, sg.gap_before_ns,
                      sg.start_ns, sg.end_ns, sg.edge_kind, sg.confidence) for sg in proj.critical_path]
    whole.driver_tokens = frozenset().union(*(p.driver_tokens for p in proj.phases)) if proj.phases else frozenset()
    whole.node_paths = frozenset(n.key.path for n in nodes)
    return TraceProjection(proj.command, proj.start_ns, proj.end_ns, [whole], "whole", proj.anchor,
                           proj.driver, nodes, [(s, d, k, c, cf) for (s, d, k), (c, cf) in edges.items()],
                           segs, proj.critical_total_ns, proj.span_count, dict(proj.availability),
                           list(proj.notes))


# ── analysis of one identity ─────────────────────────────────────────────────

def _cls(a: float | None, b: float | None, noise_pct: float, noise_ns: float) -> str:
    return cmp.classify(a, b, noise_pct=noise_pct, noise_ns=noise_ns)[0]


class _Explainer:
    def __init__(self, c: _Comparison) -> None:
        self.c = c
        self.edges_a, self.edges_b = c.edge_sets()
        self.preds_b: dict[int, list[tuple[int, str]]] = defaultdict(list)
        for (s, d, kind) in self.edges_b:
            self.preds_b[d].append((s, kind))
        self.preds_a: dict[int, set[tuple[int, str]]] = defaultdict(set)
        for (s, d, kind) in self.edges_a:
            self.preds_a[d].add((s, kind))
        self.status: dict[int, str] = {}
        self.cause: dict[int, str] = {}
        self.shift: dict[int, tuple[float | None, float | None]] = {}
        self.end_b: dict[int, float] = {}      # mean end offset within its candidate phase
        self.end_a: dict[int, float] = {}      # ... and within its baseline phase
        both = lambda i: i.ka is not None and i.kb is not None  # noqa: E731
        self.paired = {i.id for i in c.identities if both(i)}
        for ident in c.identities:
            self.shift[ident.id], self.end_b[ident.id], self.end_a[ident.id] = self._offsets(ident)

    def _offsets(self, ident: Identity) -> tuple[tuple[float | None, float | None], float | None, float | None]:
        """Mean change of (start, end) offset within aligned phases, and the
        mean end offset in the candidate and in the baseline."""
        pa, pb = self.c.pa, self.c.pb
        ds, de, eb, ea = [], [], [], []
        for pair, (a, b) in ident.pairs.items():
            if a is None or b is None:
                continue
            p = self.c.pairs[pair]
            a0, b0 = pa.phases[p.a].start_ns, pb.phases[p.b].start_ns
            ds.append((b.first_start - b0) - (a.first_start - a0))
            de.append((b.last_end - b0) - (a.last_end - a0))
            eb.append(b.last_end - b0)
            ea.append(a.last_end - a0)
        if not ds:
            return (None, None), None, None
        return (sum(ds) / len(ds), sum(de) / len(de)), sum(eb) / len(eb), sum(ea) / len(ea)

    def components(self, ident: Identity) -> dict[str, float]:
        a, b = ident.a, ident.b
        node = ident.node
        comp: dict[str, float] = {}
        if ident.ka is None:
            comp["new_work"] = float(b.own_ns)
            return comp
        if ident.kb is None:
            comp["new_work"] = -float(a.own_ns)
            return comp
        avg = a.self_ns / a.count if a.count else 0.0
        inv = (b.count - a.count) * avg
        per_call = (b.self_ns - a.self_ns) - inv
        comp["more_invocations"] = inv
        if node.bucket == "Synchronization":
            comp["synchronization"] = per_call
        elif node.bucket == "Communication":
            comp["communication"] = per_call
        else:
            comp["increased_work"] = per_call
        if a.queued and b.queued:
            comp["queueing"] = float(b.queue_ns - a.queue_ns)
        comp["lost_overlap"] = float(a.overlap_ns - b.overlap_ns)
        comp["moved_onto_critical_path"] = float(b.critical_ns - a.critical_ns)
        return comp

    def classify(self, ident: Identity) -> None:
        c = self.c
        a, b = ident.a, ident.b
        own = _cls(a.own_ns if ident.ka else None, b.own_ns if ident.kb else None, c.noise_pct, c.noise_ns)
        self.status[ident.id] = own
        comp = self.components(ident)
        if ident.ka is None or ident.kb is None:
            self.cause[ident.id] = "new_work"
            return
        if own in (cmp.STATUS_REGRESSED, cmp.STATUS_IMPROVED):
            sign = 1 if own == cmp.STATUS_REGRESSED else -1
            measured = {k: v * sign for k, v in comp.items() if k in _MEASURED_CAUSES}
            cause = max(measured, key=lambda k: measured[k]) if measured else "increased_work"
            # longer queueing because the work is now serialized behind other
            # work (moved to a busy stream / a new stream-order edge) is the
            # loss of overlap showing up as queue time
            if cause == "queueing" and sign * comp.get("lost_overlap", 0.0) >= c.noise_ns and (
                    ident.ka.stream != ident.kb.stream or
                    (self.added_in_edges(ident) if sign > 0 else self.removed_in_edges(ident))):
                cause = "lost_overlap"
            self.cause[ident.id] = cause
            return
        impact = _cls(float(a.critical_ns), float(b.critical_ns), c.noise_pct, c.noise_ns)
        if impact not in (cmp.STATUS_REGRESSED, cmp.STATUS_IMPROVED):
            self.cause[ident.id] = ""
            return
        sign = 1 if impact == cmp.STATUS_REGRESSED else -1
        added = self.added_in_edges(ident) if sign > 0 else self.removed_in_edges(ident)
        stream_changed = ident.ka.stream != ident.kb.stream
        overlap_loss = sign * comp.get("lost_overlap", 0.0)
        before = a.critical_ns if sign > 0 else b.critical_ns
        if overlap_loss >= c.noise_ns and (added or stream_changed):
            cause = "lost_overlap"
        elif added:
            cause = "changed_dependency"
        elif before < c.noise_ns:
            cause = "moved_onto_critical_path"
        elif overlap_loss >= c.noise_ns:
            cause = "lost_overlap"
        else:
            cause = "moved_onto_critical_path"
        self.cause[ident.id] = cause

    def added_in_edges(self, ident: Identity) -> list[tuple[int, str]]:
        """Edges into `ident` in the candidate whose both ends exist in both
        runs but that the baseline lacks."""
        have = self.preds_a.get(ident.id, set())
        return [(s, k) for s, k in self.preds_b.get(ident.id, ())
                if s in self.paired and (s, k) not in have]

    def removed_in_edges(self, ident: Identity) -> list[tuple[int, str]]:
        now = set(self.preds_b.get(ident.id, ()))
        return [(s, k) for s, k in self.preds_a.get(ident.id, ()) if s in self.paired and (s, k) not in now]

    def chain(self, ident: Identity, improved: bool = False) -> list[int]:
        """Upstream chain explaining a wait or a move onto/off the critical
        path by what it waited for: repeatedly the predecessor whose end
        moved latest (a regression: candidate graph) or earliest (an
        improvement: baseline graph, mirror image) relative to its phase,
        until one whose own measured time regressed / improved -- the
        origin. Symmetric so that swapping baseline and candidate swaps the
        attribution instead of changing it."""
        c = self.c
        preds = self.preds_a if improved else self.preds_b
        ends = self.end_a if improved else self.end_b
        sign = -1.0 if improved else 1.0
        origin_status = cmp.STATUS_IMPROVED if improved else cmp.STATUS_REGRESSED
        out: list[int] = []
        seen = {ident.id}
        cur = ident.id
        tol = 50_000
        for _ in range(MAX_CHAIN):
            best, best_shift = None, 0.0
            cur_end = ends.get(cur)
            for s, kind in sorted(preds.get(cur, ())):
                if s in seen or kind not in _GATING_KINDS:
                    continue
                # only work that finishes within the node's own span can have
                # delayed it (identity edges also join iterations: the
                # previous iteration's sync is not what this kernel waited for)
                s_end = ends.get(s)
                if cur_end is not None and s_end is not None and s_end > cur_end + tol:
                    continue
                end_shift = self.shift.get(s, (None, None))[1]
                if end_shift is not None and sign * end_shift > best_shift:
                    best, best_shift = s, sign * end_shift
            if best is None or best_shift < c.noise_ns / 2:
                break
            out.append(best)
            seen.add(best)
            if self.status.get(best) == origin_status and \
                    self.cause.get(best) in ("increased_work", "more_invocations", "queueing"):
                break
            cur = best
        return out


# ── report ───────────────────────────────────────────────────────────────────

def _range(side: _Side) -> dict[str, int] | None:
    if side.first_start is None:
        return None
    return {"startNs": int(side.first_start), "endNs": int(side.last_end)}


def _ident_label(ident: Identity) -> str:
    key = ident.key
    return key.leaf.split(":", 1)[-1] if key else "?"


def compare_traces(trace_a, trace_b, *, noise_pct: float = cmp.NOISE_FLOOR_PCT,
                   noise_ns: float = cmp.NOISE_FLOOR_NS,
                   min_confidence: float = DEFAULT_MIN_CONFIDENCE, top_n: int = 25) -> dict[str, Any]:
    """BEFORE (`trace_a`) vs AFTER (`trace_b`): the full comparison as one
    JSON-serializable dict (schema COMPARE_SCHEMA)."""
    pa, pb = build_projection(trace_a), build_projection(trace_b)
    return compare_projections(pa, pb, trace_a=trace_a, trace_b=trace_b, noise_pct=noise_pct,
                               noise_ns=noise_ns, min_confidence=min_confidence, top_n=top_n)


def compare_projections(pa: TraceProjection, pb: TraceProjection, *, trace_a=None, trace_b=None,
                        noise_pct: float = cmp.NOISE_FLOOR_PCT, noise_ns: float = cmp.NOISE_FLOOR_NS,
                        min_confidence: float = DEFAULT_MIN_CONFIDENCE, top_n: int = 25) -> dict[str, Any]:
    notes: list[str] = []
    pairs, phase_conf = align_phases(pa.phases, pb.phases)
    comp = _Comparison(pa, pb, pairs, noise_pct, noise_ns)
    comp.run()
    confidence = phase_conf * comp.match_conf
    method = "phase-aligned"
    phased = (pa.phase_method, pb.phase_method)
    if confidence < min_confidence or phased == ("whole", "whole"):
        # whole runs are compared with each other whatever their phase
        # structure; how alike the runs are overall scales the confidence
        ca, cb = collapse(pa), collapse(pb)
        wsim = phase_similarity(ca.phases[0], cb.phases[0])
        wpairs = [PhasePair(0, 0, 0, wsim, "matched")]
        wcomp = _Comparison(ca, cb, wpairs, noise_pct, noise_ns)
        wcomp.run()
        wconfidence = wsim * wcomp.match_conf
        if phased != ("whole", "whole"):
            notes.append(f"phase alignment confidence {confidence:.2f} below {min_confidence:.2f}: "
                         "compared as whole runs")
        if wconfidence >= min_confidence or phased == ("whole", "whole"):
            pa, pb, pairs, comp, confidence, method = ca, cb, wpairs, wcomp, wconfidence, "whole-run"
            phase_conf = wsim
        elif wconfidence > confidence:
            confidence, phase_conf, comp, method = wconfidence, wsim, wcomp, "aggregate"
        else:
            method = "aggregate"
    report: dict[str, Any] = {
        "schema": COMPARE_SCHEMA,
        "baseline": _side_summary(pa, trace_a),
        "candidate": _side_summary(pb, trace_b),
        "noiseFloor": {"pct": noise_pct, "ns": noise_ns,
                       "note": "fixed disclosed heuristic threshold (both the % and the absolute time "
                               "must be exceeded) -- not a statistical significance test: each side is "
                               "a single run with no variance information"},
        "wallTime": _wall(pa, pb, noise_pct, noise_ns),
        "alignment": {
            "method": method, "confidence": round(confidence, 4), "kind": "heuristic",
            "minConfidence": min_confidence, "phaseConfidence": round(phase_conf, 4),
            "matchConfidence": round(comp.match_conf, 4),
            "baselinePhases": [p.to_dict() for p in pa.phases],
            "candidatePhases": [p.to_dict() for p in pb.phases],
            "baselineMethod": pa.phase_method, "candidateMethod": pb.phase_method,
            "baselineAnchor": pa.anchor, "candidateAnchor": pb.anchor,
            "pairs": [], "notes": notes,
        },
        "contributors": [], "offCriticalPath": [], "improvements": [], "newWork": [], "removedWork": [],
        "phases": [], "criticalPath": {}, "unavailable": _unavailable(pa, pb),
    }
    if trace_a is not None and trace_b is not None:
        report["aggregate"] = cmp.report_dict(trace_a, trace_b, noise_pct=noise_pct, noise_ns=noise_ns,
                                              top_n=top_n)
    if method == "aggregate":
        report["alignment"]["fallbackReason"] = (
            f"alignment confidence {confidence:.2f} is below {min_confidence:.2f}: the runs' structure "
            "differs too much for node-level explanations; showing the (category, name) aggregate "
            "comparison instead")
        report["alignment"]["pairs"] = [_pair_dict(p, pa, pb) for p in pairs]
        return report
    _describe_alignment(report["alignment"], pairs, pa, pb)
    ex = _Explainer(comp)
    for ident in comp.identities:
        ex.classify(ident)
    rows = {ident.id: _ident_row(ident, comp, ex, confidence) for ident in comp.identities}
    _attribute_to_origins(rows, comp, noise_pct, noise_ns)
    contributors, off_path, improvements, new_work, removed, propagated = [], [], [], [], [], []
    for ident in comp.identities:
        r = rows[ident.id]
        if ident.ka is None:
            if r["status"] == cmp.STATUS_NEW and b_own(ident) >= noise_ns:
                new_work.append(r)
            continue
        if ident.kb is None:
            if a_own(ident) >= noise_ns:
                removed.append(r)
            continue
        if r["propagatedFrom"] is not None and (
                r["status"] in (cmp.STATUS_REGRESSED, cmp.STATUS_IMPROVED) or abs(r["criticalDeltaNs"]) >= noise_ns):
            propagated.append(r)
        if r["impactStatus"] == cmp.STATUS_REGRESSED:
            contributors.append(r)
        elif r["status"] == cmp.STATUS_REGRESSED and r["propagatedFrom"] is None:
            off_path.append(r)
        if r["status"] == cmp.STATUS_IMPROVED or r["impactStatus"] == cmp.STATUS_IMPROVED:
            improvements.append(r)
    propagated.sort(key=lambda r: -max(abs(r["ownDeltaNs"]), abs(r["criticalDeltaNs"])))
    d_cp = pb.critical_total_ns - pa.critical_total_ns
    contrib_ns = sum(r["impactNs"] for r in contributors)
    off_ns = sum(r["impactNs"] for r in rows.values()
                 if r["impactStatus"] == cmp.STATUS_IMPROVED and r["impactNs"] < 0)
    report["criticalPathChange"] = {"deltaNs": d_cp, "contributorsNs": contrib_ns, "offPathNs": off_ns,
                                    "otherNs": d_cp - contrib_ns - off_ns, "kind": "derived"}
    report["propagated"] = propagated[:top_n]
    # origins before the waits they caused, on equal impact
    contributors.sort(key=lambda r: (-r["impactNs"], r["propagatedFrom"] is not None))
    for i, r in enumerate(contributors):
        r["rank"] = i + 1
    off_path.sort(key=lambda r: -r["ownDeltaNs"])
    improvements.sort(key=lambda r: min(r["impactNs"], r["ownDeltaNs"]))
    new_work.sort(key=lambda r: -r["ownDeltaNs"])
    removed.sort(key=lambda r: r["ownDeltaNs"])
    report["contributors"] = contributors[:top_n]
    report["offCriticalPath"] = off_path[:top_n]
    report["improvements"] = improvements[:top_n]
    report["newWork"] = new_work[:top_n]
    report["removedWork"] = removed[:top_n]
    report["phases"] = _phase_rows(comp, rows, noise_pct, noise_ns)
    report["criticalPath"] = {"baseline": _cp_view(pa, comp.ident_a, comp),
                              "candidate": _cp_view(pb, comp.ident_b, comp)}
    if not contributors and report["wallTime"]["status"] == cmp.STATUS_REGRESSED:
        report["unavailable"].append({
            "conclusion": "wall-time regression attribution",
            "reason": "no single matched node's critical-path time changed by more than the noise "
                      "floor: the change is spread over many small contributions"})
    return report


def _attribute_to_origins(rows: dict[int, dict], comp: _Comparison, noise_pct: float,
                          noise_ns: float) -> None:
    """Move the critical-path change of a wait (or of work pushed onto the
    path) to the origin its upstream chain ends at, capped at the origin's
    own measured regression -- an origin cannot account for more wall time
    than the extra work it did; what exceeds the cap stays with the
    symptom. The critical path credits a blocking wait's whole duration
    (e.g. an MPI_Recv waiting for a late sender) to the wait itself."""
    for r in rows.values():
        r["criticalDeltaNs"] = r["impactNs"]
        r["receivedNs"] = 0
        r["passedOnNs"] = 0
    # regressions (sign +1) and, mirrored, improvements (sign -1): an
    # improved wait is credited to the origin whose work shrank
    for sign in (1, -1):
        groups: dict[int, list[dict]] = defaultdict(list)
        for r in rows.values():
            o = r["propagatedFrom"]
            if o is not None and o in rows and sign * r["criticalDeltaNs"] > 0:
                groups[o].append(r)
        for o, symptoms in groups.items():
            ro = rows[o]
            base = max(sign * ro["criticalDeltaNs"], 0)
            total = sum(sign * r["criticalDeltaNs"] for r in symptoms)
            give = max(0, min(total, max(base, sign * ro["ownDeltaNs"]) - base))
            ro["impactNs"] = sign * (base + give)
            ro["receivedNs"] = sign * give
            residual = total - give
            for r in symptoms:
                r["impactNs"] = sign * int(round(residual * sign * r["criticalDeltaNs"] / total)) if total else 0
                r["passedOnNs"] = r["criticalDeltaNs"] - r["impactNs"]
    idents = comp.identities
    for r in rows.values():
        ident = idents[r["id"]]
        if ident.ka is None or ident.kb is None or r["impactKind"] == "unavailable":
            continue
        base = float(ident.a.critical_ns)
        r["impactStatus"] = _cls(base, base + r["impactNs"], noise_pct, noise_ns)


def a_own(ident: Identity) -> int:
    return ident.a.own_ns


def b_own(ident: Identity) -> int:
    return ident.b.own_ns


def _side_summary(p: TraceProjection, trace) -> dict[str, Any]:
    return {"command": p.command, "wallNs": p.wall_ns, "spans": p.span_count,
            "phaseMethod": p.phase_method, "phases": len(p.phases), "driver": p.driver,
            "criticalPathNs": p.critical_total_ns, "availability": p.availability,
            "notes": p.notes}


def _wall(pa: TraceProjection, pb: TraceProjection, noise_pct: float, noise_ns: float) -> dict[str, Any]:
    status, delta = cmp.classify(float(pa.wall_ns), float(pb.wall_ns), noise_pct=noise_pct, noise_ns=noise_ns)
    return {"baselineNs": pa.wall_ns, "candidateNs": pb.wall_ns, "deltaNs": pb.wall_ns - pa.wall_ns,
            "deltaPct": (100.0 * (pb.wall_ns - pa.wall_ns) / pa.wall_ns) if pa.wall_ns else None,
            "status": status, "kind": "measured", "reason": delta["reason"],
            "baselineCriticalNs": pa.critical_total_ns, "candidateCriticalNs": pb.critical_total_ns}


def _pair_dict(p: PhasePair, pa: TraceProjection, pb: TraceProjection) -> dict[str, Any]:
    return {"index": p.index, "baseline": p.a, "candidate": p.b, "status": p.status,
            "similarity": round(p.similarity, 4), "ambiguous": p.ambiguous,
            "baselineLabel": pa.phases[p.a].label if p.a is not None else None,
            "candidateLabel": pb.phases[p.b].label if p.b is not None else None}


def _describe_alignment(al: dict, pairs: list[PhasePair], pa: TraceProjection, pb: TraceProjection) -> None:
    al["pairs"] = [_pair_dict(p, pa, pb) for p in pairs]
    ins = [p for p in pairs if p.status == "inserted"]
    rem = [p for p in pairs if p.status == "removed"]

    def shape(p: TraceProjection) -> str:
        kinds = Counter(ph.kind for ph in p.phases)
        if p.phase_method == "iterations":
            its = sum(ph.iterations for ph in p.phases if ph.kind == "iteration")
            return (f"{'prologue + ' if kinds['prologue'] else ''}{its} iteration(s)"
                    f"{' + epilogue' if kinds['epilogue'] else ''} (anchor {p.anchor.split(':', 1)[-1]})")
        if p.phase_method == "segments":
            return f"{kinds['segment']} top-level segment(s)"
        return "one whole-run phase"
    al["notes"].append(f"baseline: {shape(pa)}; candidate: {shape(pb)}")
    for kind, lst, side in (("inserted in the candidate", ins, pb), ("removed from the baseline", rem, pa)):
        if lst:
            labels = ", ".join(side.phases[p.b if p.b is not None else p.a].label for p in lst[:5])
            amb = " (identical neighbours: which one is the extra phase is arbitrary)" \
                if any(p.ambiguous for p in lst) else ""
            al["notes"].append(f"{len(lst)} phase(s) {kind}: {labels}{amb}")


def _ident_row(ident: Identity, comp: _Comparison, ex: _Explainer, alignment_conf: float) -> dict[str, Any]:
    a, b = ident.a, ident.b
    has_a, has_b = ident.ka is not None, ident.kb is not None
    noise_pct, noise_ns = comp.noise_pct, comp.noise_ns
    node = ident.node
    key = ident.key
    status = ex.status.get(ident.id, cmp.STATUS_UNAVAILABLE)
    cps = comp.pa.availability.get("criticalPath") and comp.pb.availability.get("criticalPath")
    impact_status = _cls(float(a.critical_ns), float(b.critical_ns), noise_pct, noise_ns) \
        if (has_a and has_b and cps) else cmp.STATUS_UNAVAILABLE
    components = ex.components(ident)
    cause = ex.cause.get(ident.id, "")
    regress = status == cmp.STATUS_REGRESSED or impact_status == cmp.STATUS_REGRESSED
    labels = CAUSE_LABELS if regress or status == cmp.STATUS_NEW else IMPROVEMENT_LABELS
    sign = 1 if regress else -1
    also = [k for k, v in sorted(components.items(), key=lambda kv: -kv[1] * sign)
            if k != cause and k in CAUSES and v * sign >= noise_ns]
    queue_ok = bool(a.queued and b.queued)
    measured = {
        "count": [a.count if has_a else None, b.count if has_b else None],
        "totalNs": [a.total_ns if has_a else None, b.total_ns if has_b else None],
        "selfNs": [a.self_ns if has_a else None, b.self_ns if has_b else None],
        "queueNs": [a.queue_ns if queue_ok else None, b.queue_ns if queue_ok else None],
        "overlapNs": [a.overlap_ns if has_a else None, b.overlap_ns if has_b else None],
        "waitNs": [a.self_ns if has_a and node.bucket in WAIT_BUCKETS else None,
                   b.self_ns if has_b and node.bucket in WAIT_BUCKETS else None],
    }
    derived = {
        "criticalNs": [a.cp_ns if has_a else None, b.cp_ns if has_b else None],
        "blameNs": [a.blame_ns if has_a else None, b.blame_ns if has_b else None],
        "onCriticalPath": [a.cp_ns > 0 if has_a else None, b.cp_ns > 0 if has_b else None],
    }
    evidence: list[dict[str, Any]] = []
    idents = comp.identities
    if has_a and has_b:
        for s, kind in ex.added_in_edges(ident)[:5]:
            evidence.append({"kind": "derived", "type": "edge_added",
                             "text": f"new {kind} dependency from {_ident_label(idents[s])} "
                                     f"({idents[s].key.role_text()})", "identity": s})
        for s, kind in ex.removed_in_edges(ident)[:5]:
            evidence.append({"kind": "derived", "type": "edge_removed",
                             "text": f"{kind} dependency from {_ident_label(idents[s])} "
                                     f"({idents[s].key.role_text()}) no longer present", "identity": s})
        if ident.ka.stream != ident.kb.stream:
            evidence.append({"kind": "measured", "type": "stream_changed",
                             "text": f"runs on stream {ident.kb.stream or '-'} (was {ident.ka.stream or '-'})"})
        if a.cp_ns == 0 and b.cp_ns > 0:
            evidence.append({"kind": "derived", "type": "critical_path",
                             "text": f"now on the critical path ({dash.fmt_ns(b.cp_ns)} credited; "
                                     "it was off the path)"})
        elif a.cp_ns > 0 and b.cp_ns == 0:
            evidence.append({"kind": "derived", "type": "critical_path",
                             "text": "no longer on the critical path"})
        if ident.extra_b or ident.extra_a:
            evidence.append({"kind": "heuristic", "type": "phase_count",
                             "text": f"{ident.extra_b or ident.extra_a} invocation(s) in phases "
                                     f"{'inserted in' if ident.extra_b else 'removed from'} the run"})
    improve = not regress and cmp.STATUS_IMPROVED in (status, impact_status)
    chain_ids = ex.chain(ident, improved=improve) if (has_a and has_b and (regress or improve) and cause in (
        "synchronization", "communication", "moved_onto_critical_path", "lost_overlap",
        "changed_dependency", "queueing")) else []
    chain = [{"identity": c, "label": _ident_label(idents[c]), "roles": idents[c].key.role_text(),
              "cause": ex.cause.get(c, ""), "status": ex.status.get(c, ""),
              "endShiftNs": ex.shift.get(c, (None, None))[1],
              "ownDeltaNs": idents[c].b.own_ns - idents[c].a.own_ns} for c in chain_ids]
    origin = chain[-1] if chain and chain[-1]["status"] == (
        cmp.STATUS_IMPROVED if improve else cmp.STATUS_REGRESSED) else None
    match_kind = "exact" if (has_a and has_b and ident.ka == ident.kb) else \
        ("structural" if has_a and has_b else ("new" if has_b else "removed"))
    explanation = _explain(ident, cause, components, measured, chain, origin, evidence, regress, noise_ns)
    impact = (b.critical_ns - a.critical_ns) if (has_a and has_b and cps) else 0
    pairs = _pair_details(ident)
    focus = pairs[0] if pairs else None
    src_full = (ident.node_b.source if ident.node_b else "") or (ident.node_a.source if ident.node_a else "")
    return {
        "id": ident.id, "label": _ident_label(ident), "path": key.path_text(), "fullPath": list(key.path),
        "roles": key.role_text(), "roleMap": key.roles(), "category": node.category,
        "rawName": (ident.node_b or ident.node_a).raw_name, "bucket": node.bucket,
        "source": source_key(src_full), "sourcePath": src_full,
        "status": status, "impactStatus": impact_status,
        "ownDeltaNs": (b.own_ns if has_b else 0) - (a.own_ns if has_a else 0),
        "impactNs": impact, "impactKind": "derived" if cps else "unavailable",
        "cause": cause, "causeLabel": labels.get(cause, CAUSE_LABELS.get(cause, "")),
        "alsoCauses": [{"cause": k, "label": labels.get(k, k), "ns": components[k] * sign} for k in also],
        "components": {k: v for k, v in components.items()},
        "measured": measured, "derived": derived, "evidence": evidence, "chain": chain,
        "propagatedFrom": origin["identity"] if origin else None,
        "explanation": explanation,
        "match": {"kind": match_kind, "score": round(ident.match_score, 4) if has_a and has_b else None},
        "confidence": round(ident.match_score * alignment_conf, 4) if has_a and has_b else None,
        # where to look: the aligned phase where it changed most (whole run
        # in `timelineWhole`); absolute ns of each run's own clock
        "timeline": {"baseline": focus["baseline"], "candidate": focus["candidate"],
                     "pair": focus["pair"]} if focus else {"baseline": _range(a), "candidate": _range(b),
                                                          "pair": None},
        "timelineWhole": {"baseline": _range(a), "candidate": _range(b)},
        "phasePairs": sorted(ident.pairs),
        "pairs": pairs[:64],
    }


def _pair_details(ident: Identity) -> list[dict[str, Any]]:
    """Per aligned phase: the node's range on each side and its changes,
    most changed first."""
    out = []
    for pair, (na, nb) in ident.pairs.items():
        crit = ((nb.cp_ns + nb.blame_ns) if nb else 0) - ((na.cp_ns + na.blame_ns) if na else 0)
        own = ((nb.self_ns + nb.queue_ns) if nb else 0) - ((na.self_ns + na.queue_ns) if na else 0)
        out.append({"pair": pair, "criticalDeltaNs": crit, "ownDeltaNs": own,
                    "baseline": {"startNs": na.first_start, "endNs": na.last_end} if na else None,
                    "candidate": {"startNs": nb.first_start, "endNs": nb.last_end} if nb else None})
    out.sort(key=lambda x: (-max(abs(x["criticalDeltaNs"]), abs(x["ownDeltaNs"])), x["pair"]))
    return out


def _explain(ident: Identity, cause: str, comp: dict, measured: dict, chain: list, origin: dict | None,
             evidence: list, regress: bool, noise_ns: float) -> str:
    a, b = ident.a, ident.b
    f = dash.fmt_ns
    if ident.ka is None:
        return f"new in the candidate: {b.count} call(s), {f(b.own_ns)} own time (measured)"
    if ident.kb is None:
        return f"absent from the candidate: {a.count} call(s), {f(a.own_ns)} own time in the baseline (measured)"
    parts = []
    d_own = b.own_ns - a.own_ns
    if cause in ("increased_work",):
        per_a = a.self_ns / a.count if a.count else 0
        per_b = b.self_ns / b.count if b.count else 0
        parts.append(f"own time per call {f(per_a)} → {f(per_b)} over {b.count} call(s) "
                     f"({f(abs(comp.get('increased_work', 0)))} {'more' if regress else 'less'}, measured)")
    elif cause == "more_invocations":
        parts.append(f"calls {a.count} → {b.count} at ~{f(a.self_ns / a.count if a.count else 0)} each "
                     f"({'+' if d_own >= 0 else ''}{f(comp.get('more_invocations', 0))}, measured)")
    elif cause == "queueing":
        parts.append(f"queue delay before execution {f(a.queue_ns)} → {f(b.queue_ns)} (measured, native "
                     "device timing)")
    elif cause in ("synchronization", "communication"):
        parts.append(f"{'waiting' if cause == 'synchronization' else 'communication'} time "
                     f"{f(a.self_ns)} → {f(b.self_ns)} (measured)")
    elif cause == "lost_overlap":
        txt = f"time running concurrently with other work {f(a.overlap_ns)} → {f(b.overlap_ns)} (measured)"
        if a.queued and b.queued and b.queue_ns - a.queue_ns >= noise_ns:
            txt += f"; it now queues {f(b.queue_ns - a.queue_ns)} longer behind the work it is serialized with"
        elif abs(b.self_ns - a.self_ns) < noise_ns:
            txt += "; its own execution time is unchanged"
        parts.append(txt)
    elif cause == "changed_dependency":
        parts.append("its incoming dependencies changed (graph-derived); own time unchanged within the "
                     "noise floor")
    elif cause == "moved_onto_critical_path":
        parts.append(f"critical-path time {f(a.critical_ns)} → {f(b.critical_ns)} (graph-derived); own "
                     "time unchanged within the noise floor" if abs(d_own) < noise_ns else
                     f"critical-path time {f(a.critical_ns)} → {f(b.critical_ns)} (graph-derived)")
    if chain:
        hops = " ← ".join(f"{c['label']} ({c['roles']})" for c in chain)
        labels = CAUSE_LABELS if regress else IMPROVEMENT_LABELS
        if origin is not None:
            d = origin["ownDeltaNs"]
            parts.append(f"it waited for {hops}; origin: {origin['label']} "
                         f"{labels.get(origin['cause'], origin['cause'])} "
                         f"({'+' if d >= 0 else '-'}{f(abs(d))}, graph-derived, heuristic phase offsets)")
        else:
            parts.append(f"it waited for {hops}, which finished {'later' if regress else 'earlier'} "
                         "(graph-derived, heuristic phase offsets)")
    if not parts:
        parts.append(f"own time {f(a.own_ns)} → {f(b.own_ns)}")
    return "; ".join(parts)


def _phase_rows(comp: _Comparison, rows: dict[int, dict], noise_pct: float, noise_ns: float) -> list[dict]:
    pa, pb = comp.pa, comp.pb
    crit: dict[int, dict[int, float]] = defaultdict(dict)
    own: dict[int, dict[int, float]] = defaultdict(dict)
    for ident in comp.identities:
        if ident.ka is None or ident.kb is None:
            continue
        for pair, (a, b) in ident.pairs.items():
            if a is None or b is None:
                continue
            crit[pair][ident.id] = (b.cp_ns + b.blame_ns) - (a.cp_ns + a.blame_ns)
            own[pair][ident.id] = (b.self_ns + b.queue_ns) - (a.self_ns + a.queue_ns)
    # same symptom -> origin transfer as the whole-run ranking, per phase
    per_pair: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for pair, d in crit.items():
        attributed = dict(d)
        groups: dict[int, list[int]] = defaultdict(list)
        for i, v in d.items():
            o = rows[i]["propagatedFrom"] if i in rows else None
            if o is not None and v > 0:
                groups[o].append(i)
        for o, symptoms in groups.items():
            base = max(attributed.get(o, 0.0), 0.0)
            total = sum(d[i] for i in symptoms)
            give = max(0.0, min(total, max(base, own[pair].get(o, 0.0)) - base))
            attributed[o] = base + give
            for i in symptoms:
                attributed[i] = (total - give) * d[i] / total if total else 0.0
        per_pair[pair] = [(v, i) for i, v in attributed.items() if v >= noise_ns]
    out = []
    for p in comp.pairs:
        A = pa.phases[p.a] if p.a is not None else None
        B = pb.phases[p.b] if p.b is not None else None
        da = A.duration_ns if A else None
        db = B.duration_ns if B else None
        status = _cls(float(da) if da is not None else None, float(db) if db is not None else None,
                      noise_pct, noise_ns)
        top = sorted(per_pair.get(p.index, []), reverse=True)[:10]
        out.append({
            "index": p.index, "status": p.status, "similarity": round(p.similarity, 4),
            "ambiguous": p.ambiguous, "label": (B or A).label,
            "baseline": A.to_dict() if A else None, "candidate": B.to_dict() if B else None,
            "deltaNs": (db or 0) - (da or 0), "deltaStatus": status,
            "criticalDeltaNs": (B.critical_ns if B else 0) - (A.critical_ns if A else 0),
            "topContributors": [{"identity": i, "label": rows[i]["label"], "impactNs": d,
                                 "cause": rows[i]["cause"]} for d, i in top],
        })
    return out


def _cp_view(proj: TraceProjection, ident_of: dict[int, Identity], comp: _Comparison,
             limit: int = 300) -> dict[str, Any]:
    """Critical path of one side: composition by identity (whole run and
    per phase pair) and the ordered path segments, merged per identity."""
    comp_total: Counter = Counter()
    by_pair: dict[int, Counter] = defaultdict(Counter)
    side = 0 if proj is comp.pa else 1
    phase_to_pair = {(p.a if side == 0 else p.b): p.index for p in comp.pairs
                     if (p.a if side == 0 else p.b) is not None}
    segments: list[dict[str, Any]] = []
    for sg in proj.critical_path:
        ident = ident_of.get(sg.node) if sg.node >= 0 else None
        iid = ident.id if ident else -1
        ns = sg.credited_ns + sg.gap_before_ns
        comp_total[iid] += ns
        pair = phase_to_pair.get(proj.nodes[sg.node].phase) if sg.node >= 0 else None
        if pair is not None:
            by_pair[pair][iid] += ns
        if segments and segments[-1]["identity"] == iid and segments[-1]["pair"] == pair:
            s = segments[-1]
            s["ns"] += ns
            s["endNs"] = sg.end_ns
            s["spans"] += 1
        elif len(segments) < limit:
            segments.append({"identity": iid, "label": _ident_label(ident) if ident else "(unattributed)",
                             "roles": ident.key.role_text() if ident else "", "ns": ns,
                             "startNs": sg.start_ns, "endNs": sg.end_ns, "edge": sg.edge_kind,
                             "confidence": sg.confidence, "pair": pair, "spans": 1})

    def composition(counter: Counter) -> list[dict[str, Any]]:
        idents = comp.identities
        return [{"identity": i, "ns": ns,
                 "label": _ident_label(idents[i]) if i >= 0 else "(unattributed)",
                 "roles": idents[i].key.role_text() if i >= 0 else ""}
                for i, ns in counter.most_common()]
    return {"totalNs": proj.critical_total_ns, "composition": composition(comp_total),
            "byPair": {str(k): composition(v) for k, v in by_pair.items()},
            "segments": segments, "kind": "derived",
            "truncated": len(proj.critical_path) > 0 and len(segments) >= limit}


def _unavailable(pa: TraceProjection, pb: TraceProjection) -> list[dict[str, str]]:
    out = [{"conclusion": "statistical significance",
            "reason": "each side is one run: deltas are screened by the disclosed noise floor only"}]
    av_a, av_b = pa.availability, pb.availability
    if not (av_a.get("criticalPath") and av_b.get("criticalPath")):
        out.append({"conclusion": "critical-path impact and movement onto the critical path",
                    "reason": "no critical path could be built for "
                              + ("either run" if not av_a.get("criticalPath") and not av_b.get("criticalPath")
                                 else ("the baseline" if not av_a.get("criticalPath") else "the candidate"))})
    has_dev = any(n.key.backend.endswith(".device") for p in (pa, pb) for n in p.nodes)
    if has_dev and not (av_a.get("queue") and av_b.get("queue")):
        out.append({"conclusion": "queueing",
                    "reason": "queue delays need native device timing (CUPTI / ROCprofiler-SDK) in both "
                              "runs; missing in " + ("both" if not av_a.get("queue") and not av_b.get("queue")
                                                    else ("the baseline" if not av_a.get("queue") else "the candidate"))})
    if not (av_a.get("source") or av_b.get("source")):
        out.append({"conclusion": "source locations",
                    "reason": "no file=/line= or sym= tags and no call stacks in either run"})
    if pa.phase_method == "whole" and pb.phase_method == "whole":
        out.append({"conclusion": "phase alignment",
                    "reason": "no recurring top-level pattern on the driver thread: compared as whole runs"})
    unranked = [name for name, p in (("baseline", pa), ("candidate", pb))
                if any(n.category == "mpi" for n in p.nodes) and not p.availability.get("ranks")]
    if unranked:
        out.append({"conclusion": "rank roles",
                    "reason": f"MPI spans without rank= tags in the {' and '.join(unranked)}: processes "
                              "matched by order of first activity"})
    return out


# ── source snippets (for the GUI) ────────────────────────────────────────────

def read_source(location: str, cwd: str = "", context: int = 3) -> dict[str, Any] | None:
    """{"path", "line", "lines": [(no, text)]} around a file:line source
    location, if the file exists on this machine (relative paths resolved
    against the trace's recorded cwd)."""
    if ":" not in location:
        return None
    file_part, _, line_part = location.rpartition(":")
    try:
        line = int(line_part)
    except ValueError:
        return None
    candidates = [Path(file_part)]
    if cwd and not Path(file_part).is_absolute():
        candidates.insert(0, Path(cwd) / file_part)
    for p in candidates:
        try:
            if p.is_file():
                lines = p.read_text(errors="replace").splitlines()
                if 1 <= line <= len(lines):
                    lo, hi = max(1, line - context), min(len(lines), line + context)
                    return {"path": str(p), "line": line,
                            "lines": [(i, lines[i - 1]) for i in range(lo, hi + 1)]}
        except OSError:
            continue
    return None


# ── text rendering ───────────────────────────────────────────────────────────

def _pct(a: float | None, b: float | None) -> str:
    if not a or b is None:
        return ""
    return f", {100.0 * (b - a) / a:+.1f}%"


def _signed(ns: float) -> str:
    return ("+" if ns >= 0 else "-") + dash.fmt_ns(abs(ns))


def render_text(report: dict[str, Any], top: int = 10) -> str:
    f = dash.fmt_ns
    L: list[str] = []
    a, b = report["baseline"], report["candidate"]
    w = report["wallTime"]
    al = report["alignment"]
    L.append("=" * 72)
    L.append(f"  Comparison: {a['command']}  →  {b['command']}")
    L.append("=" * 72)
    L.append(f"  Wall time       : {f(w['baselineNs'])} → {f(w['candidateNs'])} "
             f"({_signed(w['deltaNs'])}{_pct(w['baselineNs'], w['candidateNs'])}) [measured, {w['status']}]")
    L.append(f"  Critical path   : {f(w['baselineCriticalNs'])} → {f(w['candidateCriticalNs'])} [graph-derived]")
    L.append(f"  Alignment       : {al['method']}, confidence {al['confidence']:.2f} [heuristic] "
             f"(phases {al['phaseConfidence']:.2f} × node matching {al['matchConfidence']:.2f})")
    for n in al["notes"]:
        L.append(f"    - {n}")
    if al["method"] == "aggregate":
        L.append(f"\n  {al.get('fallbackReason', '')}")
        agg = report.get("aggregate")
        if agg:
            L.append("\n  Aggregate comparison by (category, name) [measured]:")
            for r in agg["topRegressions"][:top]:
                L.append(f"    {_signed(r['deltaNs']):>10}  {r['name'][:50]}  ({r['category']})")
    else:
        L.append("\n  Ranked causal contributors (Δ critical-path time, graph-derived):")
        dc = report.get("criticalPathChange")
        if dc:
            L.append(f"    critical path {_signed(dc['deltaNs'])} = {_signed(dc['contributorsNs'])} from the "
                     f"contributors below, {_signed(dc['offPathNs'])} from work that left or shrank on the "
                     f"path, {_signed(dc['otherNs'])} below the noise floor / unattributed")
        if not report["contributors"]:
            L.append("    (none above the noise floor)")
        for r in report["contributors"][:top]:
            _render_row(L, r, impact=True)
        if report.get("propagated"):
            L.append("\n  Waits and delays caused upstream (their critical-path change is credited to the origin):")
            for r in report["propagated"][:top]:
                origin = f"  ← {r['chain'][-1]['label']} ({r['chain'][-1]['roles']})" if r["chain"] else ""
                d = max(r["ownDeltaNs"], r["criticalDeltaNs"])
                L.append(f"    {_signed(d):>10}  {r['label']}  [{r['roles']}]  {r['causeLabel']}{origin}")
        if report["offCriticalPath"]:
            L.append("\n  Regressions off the critical path (no measured wall-time effect):")
            for r in report["offCriticalPath"][:top]:
                _render_row(L, r, impact=False)
        if report["newWork"]:
            L.append("\n  New in the candidate:")
            for r in report["newWork"][:top]:
                L.append(f"    {_signed(r['ownDeltaNs']):>10}  {r['label']}  [{r['roles']}]  {r['path']}")
        if report["removedWork"]:
            L.append("\n  Removed from the candidate:")
            for r in report["removedWork"][:top]:
                L.append(f"    {_signed(r['ownDeltaNs']):>10}  {r['label']}  [{r['roles']}]  {r['path']}")
        if report["improvements"]:
            L.append("\n  Improvements:")
            for r in report["improvements"][:top]:
                d = r["impactNs"] if r["impactStatus"] == cmp.STATUS_IMPROVED else r["ownDeltaNs"]
                L.append(f"    {_signed(d):>10}  {r['label']}  [{r['roles']}]  {r['causeLabel']}")
        changed = [p for p in report["phases"] if p["status"] != "matched" or
                   p["deltaStatus"] in (cmp.STATUS_REGRESSED, cmp.STATUS_IMPROVED)]
        if changed:
            L.append("\n  Phases that changed:")
            for p in changed[:top]:
                tail = ""
                if p["topContributors"]:
                    t = p["topContributors"][0]
                    tail = f"  top: {t['label']} {_signed(t['impactNs'])}"
                L.append(f"    {p['label']:<28} {p['status']:<9} {_signed(p['deltaNs']):>10}{tail}")
    if report["unavailable"]:
        L.append("\n  Unavailable conclusions:")
        for u in report["unavailable"]:
            L.append(f"    - {u['conclusion']}: {u['reason']}")
    nf = report["noiseFloor"]
    L.append(f"\n  Noise floor: {nf['pct']:.0f}% and {f(nf['ns'])} -- {nf['note']}.")
    L.append("=" * 72)
    return "\n".join(L)


def _render_row(L: list[str], r: dict, impact: bool) -> None:
    f = dash.fmt_ns
    d = r["impactNs"] if impact else r["ownDeltaNs"]
    rank = f"{r.get('rank', '')}." if impact and r.get("rank") else "  "
    L.append(f"  {rank:>3} {_signed(d):>10}  {r['label']}  [{r['roles']}]  {r['causeLabel']}")
    m = r["measured"]
    sa, sb = m["selfNs"]
    ca, cb = m["count"]
    qa, qb = m["queueNs"]
    queue = f", queued {f(qa)} → {f(qb)}" if qa is not None and qb is not None and (qa or qb) else ""
    L.append(f"        measured: self time {f(sa or 0)} → {f(sb or 0)}{queue}, calls {ca} → {cb}")
    if impact and r.get("receivedNs"):
        L.append(f"        includes {f(r['receivedNs'])} of critical-path time observed on the waits it caused")
    L.append(f"        path: {r['path']}" + (f"   source: {r['source']}" if r["source"] else ""))
    L.append(f"        why: {r['explanation']}")
    for e in r["evidence"][:3]:
        L.append(f"        evidence [{e['kind']}]: {e['text']}")
    if r["confidence"] is not None:
        L.append(f"        confidence: {r['confidence']:.2f} (node match {r['match']['score']:.2f}, "
                 f"{r['match']['kind']})")
