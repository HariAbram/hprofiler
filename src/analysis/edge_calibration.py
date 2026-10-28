"""
Calibrating dependency evidence into probabilities -- the method, not a
calibration.

The evidence statuses and tiers (evidence.py) are classes, not
probabilities: nothing in a trace says how often an "inferred, medium"
point-to-point match is wrong. A probability needs labelled data -- traces
whose true dependencies are known: synthetic scenarios with constructed
ground truth (tests/validation), or application traces checked against a
ground-truth instrumentation run. This module holds the method; hprofiler
ships no calibration, so every probability is "unquantified" unless a
calibration file is given (HPROFILER_EDGE_CALIBRATION=FILE or
`critical-path --calibration FILE`), and a report then names the dataset it
was fitted on.

  score     an ordinal evidence score per candidate (evidence_score):
            status first (observed > inferred > ambiguous > unsupported >
            contradicted), then tier (certain > high > medium > low), then
            class (direct > inferred)
  strata    candidates are calibrated per (edge kind, provenance) -- "a
            posting-order point-to-point match" --, since one tier means
            different things for different evidence; a stratum with fewer
            than min_examples labelled candidates stays unquantified
  fit       isotonic regression of the labels on the score within a
            stratum (pool-adjacent-violators): the closest monotone mapping
            in squared error, so more evidence never means less probable;
            scores between fitted ones take the next lower fitted value
  quality   expected calibration error (equal-width bins), Brier score and
            log loss on labelled candidates -- held out from the fit when
            judging a calibration (evaluate)
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

STATUS_ORDER = {"contradicted": 0, "unsupported": 1, "ambiguous": 2, "inferred": 3, "observed": 4}
TIER_ORDER = {"low": 0, "medium": 1, "high": 2, "certain": 3}
UNQUANTIFIED = "unquantified"


def evidence_score(status: str, tier: str, cls: str) -> int:
    """Ordinal evidence score: status, then tier, then class."""
    return 100 * STATUS_ORDER[status] + 10 * TIER_ORDER[tier] + (1 if cls == "direct" else 0)


@dataclass(frozen=True)
class Labeled:
    kind: str
    provenance: str
    score: int
    label: bool

    @property
    def stratum(self) -> str:
        return f"{self.kind}/{self.provenance}"


# ── isotonic regression ───────────────────────────────────────────────────────

def pav(xs: list[float], ys: list[float], ws: list[float] | None = None) -> tuple[list[float], list[float]]:
    """Pool-adjacent-violators: the non-decreasing fit of ys on xs (ties in
    xs pooled first). Returns (distinct xs, fitted value at each)."""
    if not xs:
        return [], []
    ws = ws or [1.0] * len(xs)
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    # pool equal xs
    pts: list[list[float]] = []                  # [x, sum w*y, sum w]
    for i in order:
        if pts and pts[-1][0] == xs[i]:
            pts[-1][1] += ws[i] * ys[i]
            pts[-1][2] += ws[i]
        else:
            pts.append([xs[i], ws[i] * ys[i], ws[i]])
    blocks: list[list[float]] = []               # [sum w*y, sum w, first point index, last]
    for k, (_x, sy, sw) in enumerate(pts):
        blocks.append([sy, sw, k, k])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            sy2, sw2, _a, b = blocks.pop()
            blocks[-1][0] += sy2
            blocks[-1][1] += sw2
            blocks[-1][3] = b
    fit = [0.0] * len(pts)
    for sy, sw, a, b in blocks:
        for k in range(int(a), int(b) + 1):
            fit[k] = sy / sw
    return [p[0] for p in pts], fit


def _step(xs: list[float], fit: list[float], x: float) -> float:
    import bisect
    k = bisect.bisect_right(xs, x) - 1
    return fit[max(0, k)]


# ── quality ──────────────────────────────────────────────────────────────────

def brier(probs: list[float], labels: list[bool]) -> float:
    return sum((p - float(y)) ** 2 for p, y in zip(probs, labels)) / len(probs) if probs else 0.0


def log_loss(probs: list[float], labels: list[bool], eps: float = 1e-6) -> float:
    if not probs:
        return 0.0
    tot = 0.0
    for p, y in zip(probs, labels):
        p = min(1 - eps, max(eps, p))
        tot -= math.log(p) if y else math.log(1 - p)
    return tot / len(probs)


def ece(probs: list[float], labels: list[bool], bins: int = 10) -> float:
    """Expected calibration error: the count-weighted mean, over equal-width
    probability bins, of |mean prediction - observed frequency|."""
    if not probs:
        return 0.0
    acc = [[0.0, 0.0, 0] for _ in range(bins)]
    for p, y in zip(probs, labels):
        b = min(bins - 1, int(p * bins))
        acc[b][0] += p
        acc[b][1] += float(y)
        acc[b][2] += 1
    return sum(abs(sp / n - sy / n) * n for sp, sy, n in acc if n) / len(probs)


# ── the calibration ──────────────────────────────────────────────────────────

@dataclass
class Calibration:
    strata: dict[str, dict] = field(default_factory=dict)   # stratum -> {"x": [...], "p": [...], "n": int}
    dataset: str = ""
    examples: int = 0
    min_examples: int = 20
    evidence_version: str = ""

    def probability(self, kind: str, provenance: str, score: int) -> float | None:
        st = self.strata.get(f"{kind}/{provenance}")
        if st is None:
            return None
        return _step(st["x"], st["p"], score)

    def to_dict(self) -> dict:
        return {"dataset": self.dataset, "examples": self.examples, "min_examples": self.min_examples,
                "evidence_version": self.evidence_version, "strata": self.strata}

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=1, sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict) -> "Calibration":
        return cls(strata=d.get("strata", {}), dataset=d.get("dataset", ""), examples=d.get("examples", 0),
                   min_examples=d.get("min_examples", 20), evidence_version=d.get("evidence_version", ""))

    @classmethod
    def load(cls, path: str) -> "Calibration":
        with open(path) as f:
            return cls.from_dict(json.load(f))


def fit(examples: list[Labeled], *, dataset: str, min_examples: int = 20) -> Calibration:
    """Isotonic fit per stratum; strata with fewer than min_examples
    examples are left out (unquantified)."""
    from .evidence import evidence_version
    by: dict[str, list[Labeled]] = {}
    for e in examples:
        by.setdefault(e.stratum, []).append(e)
    cal = Calibration(dataset=dataset, examples=len(examples), min_examples=min_examples,
                      evidence_version=evidence_version())
    for st, lst in sorted(by.items()):
        if len(lst) < min_examples:
            continue
        xs, p = pav([float(e.score) for e in lst], [1.0 if e.label else 0.0 for e in lst])
        cal.strata[st] = {"x": xs, "p": p, "n": len(lst)}
    return cal


def evaluate(cal: Calibration, examples: list[Labeled]) -> dict:
    """ECE, Brier score and log loss of a calibration on labelled examples
    (only those it quantifies), and how many it does not."""
    probs, labels, unq = [], [], 0
    for e in examples:
        p = cal.probability(e.kind, e.provenance, e.score)
        if p is None:
            unq += 1
            continue
        probs.append(p)
        labels.append(e.label)
    return {"quantified": len(probs), "unquantified": unq, "ece": ece(probs, labels),
            "brier": brier(probs, labels), "log_loss": log_loss(probs, labels)}


def labeled_examples(es, truth: set[tuple[int, int, str]]) -> list[Labeled]:
    """Every candidate of an evidence set (evidence.EvidenceSet, with
    alternatives) labelled by whether (src, dst, edge kind) is a true
    dependency."""
    from .cp_model import PROVS, edge_class
    from .criticalpath import EDGE_CONFS, EDGE_KINDS
    from .evidence import STATUSES, _fallback_prov
    out = []
    for r in es.cand.tolist():
        dst, src, k, c, p, status = r[0], r[1], r[2], r[3], r[4], r[5]
        kind, tier = EDGE_KINDS[k], EDGE_CONFS[c]
        prov = PROVS[p] if p >= 0 else _fallback_prov(kind, tier)
        st = STATUSES[status]
        out.append(Labeled(kind, prov, evidence_score(st, tier, edge_class(prov, tier)),
                           (src, dst, kind) in truth))
    return out


def from_env() -> Calibration | None:
    path = os.environ.get("HPROFILER_EDGE_CALIBRATION")
    return Calibration.load(path) if path else None


def path_probabilities(cal: Calibration | None, res, es) -> dict:
    """The observed path's dependency steps under a calibration: how many are
    quantified, the expected number of wrong ones among those, and the
    probability that all quantified ones are right (independence assumed:
    a statement about the calibration's strata, not about this trace). With
    no calibration: {"status": "unquantified"}."""
    if cal is None:
        return {"status": UNQUANTIFIED}
    from . import cp_model as cm
    from .criticalpath import EDGE_CONFS, EDGE_KINDS
    from .evidence import STATUSES
    g, w = res.graph, res.walk
    probs, unq = [], 0
    for e in w.steps:
        if g.mode[e] != cm.DEP or g.origin[e] < 0:
            continue
        r = es.cand[int(g.origin[e])]
        kind = EDGE_KINDS[int(r["kind"])]
        prov = cm.PROVS[int(r["prov"])]
        tier = EDGE_CONFS[int(r["conf"])]
        p = cal.probability(kind, prov, evidence_score(STATUSES[int(r["status"])], tier,
                                                       cm.edge_class(prov, tier)))
        if p is None:
            unq += 1
        else:
            probs.append(p)
    all_right = math.prod(probs) if probs else None
    return {"status": "calibrated", "dataset": cal.dataset, "quantified_steps": len(probs),
            "unquantified_steps": unq, "expected_wrong_steps": sum(1 - p for p in probs),
            "p_all_quantified_right": all_right}
