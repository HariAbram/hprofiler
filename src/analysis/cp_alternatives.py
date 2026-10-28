"""
Alternative worlds: where the trace does not decide a dependency, the
critical path under each consistent alternative.

The builders pick one answer per question (evidence.py). Most answers are
forced -- a correlation id names one submission, a non-wildcard receive on
a channel whose every call was recorded pairs by MPI's non-overtaking rule
-- but some are not, and only there are alternatives generated:

  p2p_alignment   a point-to-point channel (sender, receiver, tag,
                  communicator) whose sends and receives do not pair one to
                  one as recorded (more sends than paired receives, or the
                  process lost events). One unit per channel: MPI's
                  non-overtaking rule pairs a channel's receives with its
                  sends in order, each send completing at most one
                  receive, so an option moves a run of consecutive
                  pairings together -- its tail to the next sends when the
                  send after the run is unpaired (a receive record lost, or
                  a send never received), its head to the previous sends
                  when the send before the run is unpaired, and, after lost
                  events, one receive to no recorded send at all with the
                  later ones of its run moved back by one (its send was the
                  lost record)
  last_arriver    a rendezvous whose last arriver (by start) arrived within
                  the clocks' uncertainty of another member (cp_clock.py):
                  the other one as the last, for every member at once
  launch_corr     device operations whose correlation ids matched several
                  host submissions. One unit per group sharing candidate
                  submissions: a submission launches one operation, so an
                  option moves an operation to a submission no operation
                  uses, or swaps two operations' submissions

Each is a choice unit: options that remove some selected edges and add
alternatives; options change together (a rendezvous's edges all come from
its last arriver). Units never share a source or a destination, so any
combination of their options is a world the trace allows. Alternatives join
the evidence set (unsupported; the groups they contest become ambiguous).
An alternative whose source event lies after its destination is never
generated.

Worlds: the default (the builders' edges); one world per unit that touches
the default observed path (that unit's first alternative changing an edge
of a span on the path, everything else as recorded: "one change"); then up
to `samples` seeded random worlds, every unit picking uniformly among its
options. Each world's event graph is built, walked and its longest path
computed exactly like the default (cp_model.analyze_model's parts).

Robust sets over the worlds evaluated:
  every   spans on the observed path in every world
  some    spans on it in at least one
  near    spans whose modeled slack is within near_frac of the modeled
          length in every world (busy work)
and the range of the observed and modeled lengths. The verdict is "not
enough evidence" when, in some world, less than robust_frac of its path's
time in calls belongs to spans on the path in every world (the default
world's path or another's: a span that joins the path in one world counts
against it), or the observed or the modeled length varies by more than
range_frac of the default's. These are worlds the trace allows, not a
probability distribution over them: nothing weighs one against another.
"""
from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from . import cp_model as cm
from . import evidence as ev

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..core.trace import Trace

_SEND_TYPES = frozenset({"send", "isend", "ssend", "issend", "bsend", "ibsend", "rsend", "irsend"})


@dataclass
class Option:
    remove: "Sequence[int]"      # selected candidate indices
    add: "Sequence[int]"         # alternative candidate indices (a channel's options share arrays)


@dataclass
class ChoiceUnit:
    generator: str
    options: list[Option]        # options[0] is "as recorded" (nothing removed or added)
    touches: set[int] = field(default_factory=set)   # span ordinals involved


@dataclass
class World:
    """One evaluated world: how it differs from the recorded one, its paths
    and how much of its path's time in calls is robust (on spans on the path
    in every world)."""
    kind: str                    # recorded | one_change | sample
    label: str
    unit: int = -1               # one_change: the unit changed
    removed: list[int] = field(default_factory=list)   # selected candidates it drops
    added: list[int] = field(default_factory=list)     # alternative candidates it uses
    spans: set[int] = field(default_factory=set)       # on its observed path
    covered_ns: int = 0
    modeled_ns: int = 0
    in_call_ns: int = 0
    robust_in_call_ns: int = 0


@dataclass
class AlternativesReport:
    units: dict[str, int] = field(default_factory=dict)       # generator -> units
    alternatives: int = 0
    worlds: int = 0
    one_change_worlds: int = 0
    sampled_worlds: int = 0
    skipped: str = ""
    every: set[int] = field(default_factory=set)
    some: set[int] = field(default_factory=set)
    near: set[int] = field(default_factory=set)
    covered_range_ns: tuple[int, int] = (0, 0)
    modeled_range_ns: tuple[int, int] = (0, 0)
    robust_in_call_ns: int = 0
    default_in_call_ns: int = 0
    worst_robust_in_call_ns: int = 0         # the least robust world's path: robust / all time in calls
    worst_in_call_ns: int = 0
    verdict: str = "decided"                # decided | not enough evidence | no alternatives
    reasons: list[str] = field(default_factory=list)
    changes: list[dict] = field(default_factory=list)       # one-change worlds that moved the path
    evidence: "ev.EvidenceSet | None" = None
    touching_units: int = 0          # units touching the recorded path ...
    one_change_limit: int = 0        # ... of which this many got a one-change world
    samples: int = 0                 # random worlds requested
    cancelled: bool = False          # stopped early (cancel): the sets cover the worlds evaluated so far
    world_list: list[World] = field(default_factory=list)   # [0] is the recorded world
    worst_world: int = -1            # index into world_list of the least robust world

    @property
    def truncated(self) -> bool:
        """Fewer one-change worlds than units touching the recorded path, or
        stopped early: the robust sets are computed over a subset of the
        worlds the generators would evaluate."""
        return self.cancelled or self.touching_units > self.one_change_limit

    def to_dict(self) -> dict:
        return {"units": self.units, "alternatives": self.alternatives, "worlds": self.worlds,
                "one_change_worlds": self.one_change_worlds, "sampled_worlds": self.sampled_worlds,
                "skipped": self.skipped, "spans_on_every_path": len(self.every),
                "spans_on_some_path": len(self.some), "near_critical_spans": len(self.near),
                "covered_range_ns": list(self.covered_range_ns), "modeled_range_ns": list(self.modeled_range_ns),
                "robust_in_call_ns": self.robust_in_call_ns, "default_in_call_ns": self.default_in_call_ns,
                "worst_robust_in_call_ns": self.worst_robust_in_call_ns,
                "worst_in_call_ns": self.worst_in_call_ns,
                "touching_units": self.touching_units, "one_change_limit": self.one_change_limit,
                "truncated": self.truncated, "cancelled": self.cancelled,
                "verdict": self.verdict, "reasons": self.reasons, "changes": self.changes[:20]}


# ── generators ───────────────────────────────────────────────────────────────

class _Builder:
    def __init__(self, es: ev.EvidenceSet) -> None:
        self.es = es
        self.alts: list[tuple[int, int, int, int, int, str]] = []
        self.units: list[ChoiceUnit] = []
        self.contested: list[int] = []      # selected candidates with a "none of them" option

    def alt(self, dst: int, src: int, kind: int, conf: int, gid: int, why: str) -> int:
        self.alts.append((dst, src, kind, conf, gid, why))
        return len(self.es) + len(self.alts) - 1


def _p2p_alignment(b: _Builder, spans_by_cat, edges, g) -> None:
    from .criticalpath import _CONF_CODE, _KIND_CODE
    p2p, low = _KIND_CODE["p2p"], _CONF_CODE["low"]
    sends: dict[tuple, list[int]] = defaultdict(list)
    for i, s in spans_by_cat.get("mpi", {}).items():
        t = s.tags
        if t.get("type") in _SEND_TYPES or (t.get("type") == "start" and t.get("ptype") == "send"):
            key = (t.get("rank"), t.get("wpeer", t.get("peer")), t.get("tag"), t.get("commid"))
            sends[key].append(i)
    for lst in sends.values():
        lst.sort(key=lambda i: g.t[2 * i])
    pos = {i: (key, k) for key, lst in sends.items() for k, i in enumerate(lst)}
    by_key: dict[tuple, list[int]] = defaultdict(list)
    for c in np.flatnonzero(edges["kind"] == p2p).tolist():
        src = int(edges["src"][c])
        if src in pos:
            by_key[pos[src][0]].append(c)
    for key, cs in by_key.items():
        lst = sends[key]
        lost = any(int(edges["conf"][c]) == low for c in cs)
        if not (len(lst) > len(cs) or lost):
            continue
        cs.sort(key=lambda c: pos[int(edges["src"][c])][1])
        ks = [pos[int(edges["src"][c])][1] for c in cs]
        if any(x >= y for x, y in zip(ks, ks[1:])):
            continue                                  # a send paired twice: no order to shift
        dsts = [int(edges["dst"][c]) for c in cs]
        gids = [int(b.es.cand["group"][c]) for c in cs]

        def to(m: int, kk: int) -> int:
            return b.alt(dsts[m], lst[kk], p2p, low, gids[m], "alignment_shift")
        opts = [Option([], [])]
        a = 0
        while a < len(cs):
            e = a                                     # a run: consecutive sends ks[a]..ks[e]
            while e + 1 < len(cs) and ks[e + 1] == ks[e] + 1:
                e += 1
            run = np.array(cs[a:e + 1], dtype=np.int64)
            if ks[e] + 1 < len(lst):                  # the send after the run is unpaired
                f = e + 1                             # the suffix [f, e] fits the next sends
                while f > a and g.t[2 * lst[ks[f - 1] + 1]] <= g.t[2 * dsts[f - 1] + 1]:
                    f -= 1
                if f <= e:
                    nxt = np.array([to(m, ks[m] + 1) for m in range(f, e + 1)], dtype=np.int64)
                    for i in range(e, f - 1, -1):     # the shortest move first
                        opts.append(Option(run[i - a:], nxt[i - f:]))
            before = ks[a] >= 1                       # the send before the run is unpaired
            if before or (lost and e > a):            # an earlier send always started earlier
                prv = np.array([to(m, ks[m] - 1) if ks[m] >= 1 else -1 for m in range(a, e + 1)],
                               dtype=np.int64)
                if before:
                    for i in range(a, e + 1):
                        opts.append(Option(run[:i - a + 1], prv[:i - a + 1]))
                if lost:                              # receive i's send was the lost record
                    for i in range(e, a - 1, -1):
                        opts.append(Option(run[i - a:], prv[i - a + 1:]))
            elif lost:
                opts.append(Option(run, []))
            if lost:
                b.contested.extend(cs[a:e + 1])
            a = e + 1
        if len(opts) > 1:
            b.units.append(ChoiceUnit("p2p_alignment", opts, set(dsts) | set(lst)))


def _last_arriver(b: _Builder, edges, g) -> None:
    from .criticalpath import _KIND_CODE
    arrival = _KIND_CODE["arrival"]
    ca = g.clock
    if ca is None:
        return
    by_src: dict[int, list[int]] = defaultdict(list)
    for c in np.flatnonzero(edges["kind"] == arrival).tolist():
        by_src[int(edges["src"][c])].append(c)
    for last, cs in by_src.items():
        members = [int(edges["dst"][c]) for c in cs]
        if not members:
            continue
        runner = max(members, key=lambda j: g.t[2 * j])
        gap = g.t[2 * last] - g.t[2 * runner]        # >= 0: the builder took the latest start
        a, d = g.domain[runner], g.domain[last]
        if a == d:
            if gap > 0:                               # one clock: only a tie is undecided
                continue
        elif ca.offset_range(a, d)[1] <= gap:         # largest delta_runner - delta_last
            continue
        conf = int(edges["conf"][cs[0]])
        adds = []
        for j in [last] + [m for m in members if m != runner]:
            gid = int(b.es.cand["group"][cs[0]]) if j == last else int(b.es.cand["group"][cs[members.index(j)]])
            adds.append(b.alt(j, runner, arrival, conf, gid, "clock_order"))
        b.units.append(ChoiceUnit("last_arriver", [Option([], []), Option(list(cs), adds)], {last, *members}))


def _launch_corr(b: _Builder, spans_by_cat, edges, g) -> None:
    from ..core import gpu_activity as ga
    from .criticalpath import _CONF_CODE, _KIND_CODE
    launch, high = _KIND_CODE["launch"], _CONF_CODE["high"]
    cs = np.flatnonzero((edges["kind"] == launch) & (edges["conf"] == high)).tolist()
    if not cs:
        return
    hosts: dict[tuple, list[int]] = defaultdict(list)
    for cat in ("cuda", "rocm", "memory", "sync"):
        for i, s in spans_by_cat.get(cat, {}).items():
            if ga.is_host_submission(s):
                for kind, val in ga._host_keys(s):
                    if kind == "corr":
                        hosts[(s.pid, s.tags["rt"], val)].append(i)
    used = {int(x) for x in edges["src"][edges["kind"] == launch].tolist()}
    cand: dict[int, list[int]] = {}
    for c in cs:
        dst, src = int(edges["dst"][c]), int(edges["src"][c])
        d = spans_by_cat["_all"][dst]
        others = set()
        for k in ("corr", "corr2"):
            v = d.tags.get(k)
            if v and v != "0":
                others.update(hosts.get((d.pid, d.tags.get("rt"), v), ()))
        others.discard(src)
        ok = sorted(h for h in others if g.t[2 * h] <= g.t[2 * dst] + ga.CLOCK_TOLERANCE_NS)
        if ok:
            cand[c] = ok
    # groups of launches that share a candidate submission (union-find over submissions)
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for c, hs in cand.items():
        for h in hs:
            parent[find(h)] = find(int(edges["src"][c]))
    groups: dict[int, list[int]] = defaultdict(list)
    for c in cand:
        groups[find(int(edges["src"][c]))].append(c)
    made: dict[tuple[int, int], int] = {}

    def to(c: int, h: int) -> int:
        if (c, h) not in made:
            made[(c, h)] = b.alt(int(edges["dst"][c]), h, launch, high, int(b.es.cand["group"][c]),
                                 "shared_correlation")
        return made[(c, h)]
    for members in groups.values():
        members.sort()
        opts = [Option([], [])]
        for c in members:                              # to a submission no launch uses
            opts.extend(Option([c], [to(c, h)]) for h in cand[c] if h not in used)
        for x, c1 in enumerate(members):               # two launches trade submissions
            s1 = int(edges["src"][c1])
            for c2 in members[x + 1:]:
                s2 = int(edges["src"][c2])
                if s1 != s2 and s2 in cand[c1] and s1 in cand.get(c2, ()):
                    opts.append(Option([c1, c2], [to(c1, s2), to(c2, s1)]))
        if len(opts) > 1:
            touches = {int(edges["dst"][c]) for c in members} | {h for c in members for h in cand[c]} | \
                {int(edges["src"][c]) for c in members}
            b.units.append(ChoiceUnit("launch_corr", opts, touches))


def generate(trace: "Trace", graph, g, es: ev.EvidenceSet) -> tuple[ev.EvidenceSet, list[ChoiceUnit]]:
    """Alternatives and choice units for a trace (graph: criticalpath.Graph,
    g: its default event graph, es: its evidence)."""
    edges = graph.edges
    spans_by_cat: dict[str, dict[int, object]] = defaultdict(dict)
    need = {int(x) for x in np.concatenate([edges["src"], edges["dst"]]).tolist()} if len(edges) else set()
    for i, s in enumerate(trace.iter_spans()):
        cat = s.category.value
        if cat in ("mpi", "cuda", "rocm", "memory", "sync"):
            spans_by_cat[cat][i] = s
        if i in need:
            spans_by_cat["_all"][i] = s
    b = _Builder(es)
    _p2p_alignment(b, spans_by_cat, edges, g)
    _last_arriver(b, edges, g)
    _launch_corr(b, spans_by_cat, edges, g)
    return ev.with_alternatives(es, b.alts, contested=b.contested), b.units


# ── worlds ───────────────────────────────────────────────────────────────────

def _world_edges(es: ev.EvidenceSet, units: list[ChoiceUnit], choice: dict[int, int]) -> np.ndarray:
    from ..core.store import EDGE_DTYPE
    keep = np.zeros(len(es), dtype=bool)
    keep[:es.n_selected] = True
    for u, k in choice.items():
        opt = units[u].options[k]
        keep[opt.remove] = False
        keep[opt.add] = True
    sel = es.cand[keep]
    out = np.empty(len(sel), dtype=EDGE_DTYPE)
    for f in ("dst", "src", "kind", "conf"):
        out[f] = sel[f]
    return out


def _evaluate(trace, graph, edges, base: cm.EventGraph) -> tuple[cm.EventGraph, cm.Walk, int,
                                                                  "cm.Criticality | None"]:
    """One world's graph, walk and longest path: from the default graph's
    skeleton when it kept one (no second read of the spans), else built."""
    from .criticalpath import Graph
    if base.skeleton is not None:
        g = cm.graph_with_edges(base, edges)
    else:
        g = cm.build_event_graph(trace, Graph(trace, graph.seqs, edges))
    w = cm.walk(g)
    if not g.order:
        return g, w, 0, None
    F, L = cm.longest(g)
    return g, w, L, cm.criticality(g, F, L)


def _first_touching(u: ChoiceUnit, es: ev.EvidenceSet, spans: set[int], limit: int = 64) -> int:
    """The first alternative of u that changes an edge of a span in `spans`
    (the first one when none of the first `limit` does)."""
    dst, src = es.cand["dst"], es.cand["src"]
    for k in range(1, min(len(u.options), limit + 1)):
        o = u.options[k]
        ids = np.concatenate([np.asarray(o.remove, dtype=np.int64), np.asarray(o.add, dtype=np.int64)])
        if ids.size and not spans.isdisjoint(np.concatenate([dst[ids], src[ids]]).tolist()):
            return k
    return 1


def _in_call(g: cm.EventGraph, w: cm.Walk) -> dict[int, int]:
    owned: dict[int, int] = defaultdict(int)
    for e, ns in zip(w.steps, w.cover):
        if g.mode[e] in (cm.BUSY, cm.WAIT, cm.UNKNOWN):
            owned[g.owner[e]] += ns
    return owned


def analyze_alternatives(trace: "Trace", graph, res: "cm.ModelResult", *, samples: int = 8,
                         one_change: int = 16, seed: int = 0, near_frac: float = 0.01,
                         robust_frac: float = 0.8, range_frac: float = 0.05,
                         node_limit: int = 2_000_000, progress=None, cancel=None) -> AlternativesReport:
    """Alternatives, worlds and robust sets (module docstring) for a trace
    whose default model is `res`. progress(done, total) is called after each
    world; cancel() returning True stops before the next one (the report then
    covers the worlds evaluated so far, `cancelled` set)."""
    g0, w0 = res.graph, res.walk
    out = AlternativesReport()
    es = ev.evidence_from_graph(graph.edges, g0)
    if 2 * g0.n > node_limit:
        out.skipped = (f"alternatives not generated: {2 * g0.n} event nodes exceed {node_limit} (each world "
                       "is a full model build)")
        out.verdict = "not evaluated"
        out.evidence = es
        out.every = out.some = set(w0.spans(g0))
        return out
    es, units = generate(trace, graph, g0, es)
    out.evidence = es
    out.alternatives = len(es) - es.n_selected
    for u in units:
        out.units[u.generator] = out.units.get(u.generator, 0) + 1
    default_spans = w0.spans(g0)
    owned0 = _in_call(g0, w0)
    out.default_in_call_ns = sum(owned0.values())
    crit0 = res.crit
    L0 = res.modeled_ns

    def near_set(g, crit, L) -> set[int]:
        if crit is None:
            return set()
        lim = near_frac * L
        return {j for j, sl in crit.slack.items() if sl <= lim}

    worlds = [(default_spans, w0.length, L0, near_set(g0, crit0, L0), owned0)]
    out.worlds = 1
    out.samples = samples
    out.one_change_limit = one_change
    out.world_list = [World("recorded", "as recorded", spans=set(default_spans), covered_ns=w0.length,
                            modeled_ns=L0)]
    if not units:
        out.verdict = "no alternatives"
        out.every = out.some = set(default_spans)
        out.near = worlds[0][3]
        out.covered_range_ns = (w0.length, w0.length)
        out.modeled_range_ns = (L0, L0)
        out.robust_in_call_ns = out.worst_robust_in_call_ns = out.worst_in_call_ns = out.default_in_call_ns
        out.world_list[0].in_call_ns = out.world_list[0].robust_in_call_ns = out.default_in_call_ns
        out.worst_world = 0
        return out
    plans: list[tuple[str, dict[int, int], int]] = []
    touching = [k for k, u in enumerate(units) if u.touches & default_spans]
    out.touching_units = len(touching)
    for k in touching[:one_change]:
        plans.append(("one_change", {k: _first_touching(units[k], es, default_spans)}, k))
    rng = random.Random(seed)
    for _ in range(samples):
        plans.append(("sample", {k: rng.randrange(len(u.options)) for k, u in enumerate(units)}, -1))
    n_samples = 0
    for i, (kind, choice, unit) in enumerate(plans):
        if cancel is not None and cancel():
            out.cancelled = True
            break
        if kind == "sample":
            n_samples += 1
        if all(v == 0 for v in choice.values()):
            if progress is not None:
                progress(i + 1, len(plans))
            continue
        g, w, L, crit = _evaluate(trace, graph, _world_edges(es, units, choice), g0)
        spans = w.spans(g)
        owned = _in_call(g, w)
        worlds.append((spans, w.length, L, near_set(g, crit, L), owned))
        removed = [int(c) for u, k in choice.items() for c in units[u].options[k].remove]
        added = [int(c) for u, k in choice.items() for c in units[u].options[k].add]
        label = (f"one change: {units[unit].generator} #{unit}, option {choice[unit]}" if kind == "one_change"
                 else f"sample {n_samples}")
        out.world_list.append(World(kind, label, unit, removed, added, set(spans), w.length, L,
                                    sum(owned.values())))
        out.worlds += 1
        if progress is not None:
            progress(i + 1, len(plans))
        if kind == "one_change":
            out.one_change_worlds += 1
            if spans != default_spans:
                u = units[unit]
                out.changes.append({"generator": u.generator, "spans_left": len(default_spans - spans),
                                    "spans_joined": len(spans - default_spans),
                                    "covered_ns": w.length, "modeled_ns": L})
        else:
            out.sampled_worlds += 1
    out.every = set.intersection(*(x[0] for x in worlds))
    out.some = set.union(*(x[0] for x in worlds))
    out.near = set.intersection(*(x[3] for x in worlds))
    covered = [x[1] for x in worlds]
    modeled = [x[2] for x in worlds]
    out.covered_range_ns = (min(covered), max(covered))
    out.modeled_range_ns = (min(modeled), max(modeled))
    out.robust_in_call_ns = sum(ns for j, ns in owned0.items() if j in out.every)
    # the least robust world: its path's time in calls on spans on the path in every world
    for i, x in enumerate(worlds):
        tot = sum(x[4].values())
        rob = sum(ns for j, ns in x[4].items() if j in out.every)
        out.world_list[i].in_call_ns, out.world_list[i].robust_in_call_ns = tot, rob
        if tot and (not out.worst_in_call_ns or rob * out.worst_in_call_ns < out.worst_robust_in_call_ns * tot):
            out.worst_in_call_ns, out.worst_robust_in_call_ns = tot, rob
            out.worst_world = i
    reasons = []
    if out.default_in_call_ns and out.robust_in_call_ns < robust_frac * out.default_in_call_ns:
        reasons.append(f"only {100 * out.robust_in_call_ns / out.default_in_call_ns:.0f}% of the path's time in "
                       f"calls belongs to spans on it in every one of {out.worlds} world(s)")
    elif out.worst_in_call_ns and out.worst_robust_in_call_ns < robust_frac * out.worst_in_call_ns:
        reasons.append(f"in one of the {out.worlds} worlds only "
                       f"{100 * out.worst_robust_in_call_ns / out.worst_in_call_ns:.0f}% of its path's time in calls "
                       "belongs to spans on the path in every world: other spans join it there")
    span = out.covered_range_ns[1] - out.covered_range_ns[0]
    if w0.length and span > range_frac * w0.length:
        reasons.append(f"the observed path's length varies by {100 * span / w0.length:.0f}% across the worlds")
    mspan = out.modeled_range_ns[1] - out.modeled_range_ns[0]
    if L0 and mspan > range_frac * L0:
        from .dashboard import fmt_ns
        reasons.append(f"the modeled longest path's length varies by {100 * mspan / L0:.0f}% across the worlds "
                       f"({fmt_ns(out.modeled_range_ns[0])} to {fmt_ns(out.modeled_range_ns[1])})")
    out.verdict = "not enough evidence" if reasons else "decided"
    out.reasons = reasons
    return out
