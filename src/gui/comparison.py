"""
Backs the Compare screen (CompareScreen.qml, always tab index 9 per this
project's "GUI tabs are always visible" convention). Registered as the
"Compare" singleton UNCONDITIONALLY, even when no comparison trace was
loaded: an unregistered QML singleton that a screen binds against is a load
error.

Roles: `hprofiler gui AFTER --compare BEFORE` -- the opened trace (the one
every other tab shows) is the **candidate**, the `--compare` trace is the
**baseline**. That way a regression's "Show in Timeline" lands on the
trace the Timeline actually displays.

Two layers, both from src/analysis:

  * the structure- and causality-aware comparison (causal_compare.py):
    verdict and alignment confidence, ranked causal contributors with their
    measured / graph-derived / heuristic evidence, phase navigation, before/
    after critical-path composition, and per-contributor details with
    timeline ranges and source locations for click-through;
  * the (category, name) aggregate comparison (compare.py) -- the
    compatibility view and the fallback the causal layer itself reports
    when its alignment confidence is too low.

The comparison is computed off the main thread by the loader
(compute_comparison_data()); only plain dicts and the two projections are
kept, never the traces.
"""
from __future__ import annotations

import csv
import json
from typing import Any

from PySide6.QtCore import QObject, Property, Signal, Slot

from ..core.trace import Trace
from ..analysis import causal_compare as cc
from ..analysis import compare as cmp
from ..analysis import dashboard as dash
from . import columns
from .tablemodel import TableBundle

_COVERAGE_BUCKETS = 60
_CP_SLICES = 8          # critical-path composition entries shown before "other"


def _summary_fields(trace: Trace | None, *, available: bool, reason: str = "") -> list[dict[str, Any]]:
    if not available or trace is None:
        return [
            {"label": label, "value": "", "kind": "unavailable", "reason": reason}
            for label in ("Executable", "Wall time", "Spans", "Kernels/functions")
        ]
    meta = trace.metadata
    stats = trace.aggregated_stats()
    ext = trace.store.span_extent(timed_only=True)
    wall_ns = (ext[1] - ext[0]) if ext else 0
    return [
        {"label": "Executable", "value": meta.command or "(unknown)", "kind": "measured", "reason": ""},
        {"label": "Wall time", "value": dash.fmt_ns(wall_ns), "kind": "measured", "reason": ""},
        {"label": "Spans", "value": dash.fmt_count(trace.span_count()), "kind": "measured", "reason": ""},
        {"label": "Kernels/functions", "value": dash.fmt_count(len(stats)), "kind": "measured", "reason": ""},
    ]


def compute_comparison_data(candidate: Trace, baseline: Trace, *,
                            noise_pct: float = cmp.NOISE_FLOOR_PCT,
                            noise_ns: float = cmp.NOISE_FLOOR_NS) -> dict[str, Any]:
    """The Qt-free part of ComparisonBridge -- everything that reads the
    traces, so the loader's worker thread can run it."""
    from ..analysis.projection import build_projection
    return {
        "baselineFields": _summary_fields(baseline, available=True),
        "candidateFields": _summary_fields(candidate, available=True),
        "aggregate": cmp.compare_aggregates(baseline, candidate, noise_pct=noise_pct, noise_ns=noise_ns),
        "buckets": cmp.compare_buckets(baseline, candidate, noise_pct=noise_pct, noise_ns=noise_ns),
        "baselineCoverage": cmp.normalized_coverage(baseline, _COVERAGE_BUCKETS),
        "candidateCoverage": cmp.normalized_coverage(candidate, _COVERAGE_BUCKETS),
        "projections": (build_projection(baseline), build_projection(candidate)),
        "candidateCwd": candidate.metadata.cwd or "",
        "baselineCwd": baseline.metadata.cwd or "",
    }


def _wait_delta(r: dict[str, Any]) -> float | None:
    """A wait's measured change in waiting time (self time of a wait node)."""
    a, b = r["measured"]["waitNs"]
    if a is None or b is None:
        return None
    return float(b - a)


def _pair_delta(pair) -> float | None:
    a, b = pair
    return None if a is None or b is None else float(b - a)


def _compact(r: dict[str, Any], joint: dict[int, list[dict]] | None = None) -> dict[str, Any]:
    """A contributor row as the ranked lists show it."""
    origin = r["chain"][-1] if r.get("chain") else None
    joint = joint or {}
    return {
        "jointOrigins": 1 + len(r.get("jointFrom") or []) if r.get("propagatedFrom") is not None and r.get("jointFrom") else 0,
        "jointWaits": len(joint.get(r["id"], [])),
        "waitBlameDeltaNs": _pair_delta(r["derived"].get("waitBlameNs", (None, None))),
        "id": r["id"], "rank": r.get("rank", 0), "label": dash.fmt_kernel_name(r["label"]),
        "roles": r["roles"], "path": r["path"], "cause": r["cause"], "causeLabel": r["causeLabel"],
        "impactNs": float(r["impactNs"]), "criticalDeltaNs": float(r.get("criticalDeltaNs", r["impactNs"])),
        "ownDeltaNs": float(r["ownDeltaNs"]), "status": r["status"], "impactStatus": r["impactStatus"],
        "confidence": -1.0 if r["confidence"] is None else float(r["confidence"]),
        "propagated": r.get("propagatedFrom") is not None,
        "origin": f"{origin['label']} ({origin['roles']})" if origin else "",
        "matchKind": r["match"]["kind"],
    }


class ComparisonBridge(QObject):
    thresholdsChanged = Signal()
    causalChanged = Signal()
    selectionChanged = Signal()

    def __init__(self, trace: Trace, compare_trace: Trace | None, theme, parent: QObject | None = None,
                 *, precomputed: dict[str, Any] | None = None) -> None:
        super().__init__(parent)
        self._theme = theme
        self._available = compare_trace is not None
        self._noise_pct = cmp.NOISE_FLOOR_PCT
        self._noise_ns = cmp.NOISE_FLOOR_NS
        self._selected_phase = -1
        self._report: dict[str, Any] = {}
        self._rows_by_id: dict[int, dict[str, Any]] = {}
        self._all_rows: dict[int, dict[str, Any]] = {}
        self._joint_waits: dict[int, list[dict[str, Any]]] = {}
        self._rows: list[dict[str, Any]] = []
        self._report_rows: list[dict[str, Any]] = []
        self._bucket_rows: list[dict[str, Any]] = []
        self._top_improvements: list[dict[str, Any]] = []
        self._top_regressions: list[dict[str, Any]] = []
        self._baseline_coverage: list[float] = [0.0] * _COVERAGE_BUCKETS
        self._comparison_coverage: list[float] = [0.0] * _COVERAGE_BUCKETS
        self._new_count = 0
        self._removed_count = 0
        self._projections = None
        self._cwd = {"baseline": "", "candidate": ""}
        # side-by-side timelines (compare_timeline.py): both traces, the
        # matcher over the comparison's identities, the baseline's own model
        self._traces = {"baseline": compare_trace, "candidate": trace}
        self._detail: dict[str, Any] = {}
        self._matcher = None
        self._timelines: dict[str, Any] = {"baseline": None, "candidate": None}

        reason = ("no comparison trace loaded -- run "
                  "`hprofiler gui AFTER.json --compare BEFORE.json`")
        if self._available:
            data = precomputed if precomputed is not None else compute_comparison_data(trace, compare_trace)
            self._baseline_fields = data["baselineFields"]
            self._comparison_fields = data["candidateFields"]
            self._set_aggregate(data["aggregate"])
            self._bucket_rows = data["buckets"]
            self._baseline_coverage = data["baselineCoverage"]
            self._comparison_coverage = data["candidateCoverage"]
            self._projections = data["projections"]
            self._cwd = {"baseline": data["baselineCwd"], "candidate": data["candidateCwd"]}
            self._recompute_causal()
        else:
            self._baseline_fields = _summary_fields(None, available=False, reason=reason)
            self._comparison_fields = _summary_fields(trace, available=True)

        self._table = TableBundle(self._rows, columns.COMPARE_COLUMNS, self)

    # ── aggregate layer ──────────────────────────────────────────────
    def _set_aggregate(self, rows: list[dict[str, Any]]) -> None:
        # Missing sides/undefined percentages stay NaN ("—" in the table),
        # never 0: a function absent from one run did not take 0ns there.
        nan = float("nan")
        self._rows = [
            {
                "name": dash.fmt_kernel_name(r["name"]), "category": r["category"], "status": r["status"],
                "baselineNs": r["baseNs"] if r["baseNs"] is not None else nan,
                "comparisonNs": r["compNs"] if r["compNs"] is not None else nan,
                "deltaNs": r["deltaNs"] if r["deltaNs"] is not None else nan,
                "deltaPct": r["deltaPct"] if r["deltaPct"] is not None else nan,
                "matchKind": r["matchKind"],
            }
            for r in rows
        ]
        self._report_rows = rows
        self._new_count = sum(1 for r in rows if r["status"] == cmp.STATUS_NEW)
        self._removed_count = sum(1 for r in rows if r["status"] == cmp.STATUS_REMOVED)
        self._refresh_top_changes()

    def _refresh_top_changes(self) -> None:
        self._top_improvements = cmp.top_changes(self._report_rows, status=cmp.STATUS_IMPROVED, limit=10)
        self._top_regressions = cmp.top_changes(self._report_rows, status=cmp.STATUS_REGRESSED, limit=10)

    # ── causal layer ─────────────────────────────────────────────────
    def _recompute_causal(self) -> None:
        if self._projections is None:
            return
        pa, pb = self._projections
        self._detail = {}
        self._report = cc.compare_projections(pa, pb, noise_pct=self._noise_pct, noise_ns=self._noise_ns,
                                              detail=self._detail)
        from .compare_timeline import EventMatcher
        self._matcher = EventMatcher(self._detail, self._traces["baseline"], self._traces["candidate"]) \
            if self._detail.get("pa") is not None and pa.span_node is not None else None
        self._rows_by_id = {}
        for key in ("contributors", "propagated", "offCriticalPath", "improvements", "newWork", "removedWork"):
            for r in self._report.get(key, []):
                self._rows_by_id.setdefault(r["id"], r)
        # joint releases: every wait whose change traces to several origins,
        # indexed by origin -- an origin off the path (direct impact 0) still
        # jointly caused the wait
        # every identity's row (the lists hold only those above the noise
        # floor): a chain step or a joint origin below it is still reachable
        self._all_rows: dict[int, dict[str, Any]] = dict(self._detail.get("rows") or {})
        for i, r in self._rows_by_id.items():
            self._all_rows.setdefault(i, r)
        self._joint_waits: dict[int, list[dict[str, Any]]] = {}
        for r in self._all_rows.values():
            o = r.get("propagatedFrom")
            if o is None:
                continue
            origins = [o] + [x for x in (r.get("jointFrom") or []) if x != o]
            dw = _wait_delta(r)
            for x in origins:
                self._joint_waits.setdefault(x, []).append({
                    "wait": r["id"], "origins": origins, "waitDeltaNs": dw,
                    "shareNs": None if dw is None else dw / len(origins), "joint": len(origins) > 1})
        if self._selected_phase >= len(self._report.get("phases", [])):
            self._selected_phase = -1

    def _list(self, key: str) -> list[dict[str, Any]]:
        return [_compact(r, self._joint_waits) for r in self._report.get(key, [])]

    @Property(bool, constant=True)
    def available(self) -> bool:
        return self._available

    @Property(bool, notify=causalChanged)
    def causalAvailable(self) -> bool:
        return bool(self._report) and self._report["alignment"]["method"] != "aggregate"

    @Property('QVariantMap', notify=causalChanged)
    def verdict(self) -> dict[str, Any]:
        r = self._report
        if not r:
            return {}
        w, al = r["wallTime"], r["alignment"]
        pct = f", {w['deltaPct']:+.1f}%" if w["deltaPct"] is not None else ""
        f = dash.fmt_ns
        method = {"phase-aligned": "phase-aligned", "whole-run": "whole run (phases not aligned)",
                  "aggregate": "aggregate fallback"}.get(al["method"], al["method"])
        dc = r.get("criticalPathChange")
        decomposition = ""
        if dc:
            decomposition = (f"critical path {_signed(dc['deltaNs'])} = {_signed(dc['contributorsNs'])} from "
                             f"ranked contributors, {_signed(dc['offPathNs'])} from work that left the path, "
                             f"{_signed(dc['otherNs'])} other")
        return {
            "wallText": f"{f(w['baselineNs'])} → {f(w['candidateNs'])} ({_signed(w['deltaNs'])}{pct})",
            "wallStatus": w["status"],
            "criticalText": f"{f(w['baselineCriticalNs'])} → {f(w['candidateCriticalNs'])}",
            "method": al["method"], "methodLabel": method,
            "confidence": al["confidence"],
            "confidenceText": f"{al['confidence']:.2f} (phases {al['phaseConfidence']:.2f} × "
                              f"node matching {al['matchConfidence']:.2f})",
            "notes": list(al["notes"]), "fallbackReason": al.get("fallbackReason", ""),
            "decomposition": decomposition,
            "baselinePhases": len(al["baselinePhases"]), "candidatePhases": len(al["candidatePhases"]),
        }

    @Property('QVariantList', notify=causalChanged)
    def unavailableConclusions(self) -> list[dict[str, Any]]:
        return list(self._report.get("unavailable", [])) if self._report else []

    @Property('QVariantList', notify=causalChanged)
    def phasePairs(self) -> list[dict[str, Any]]:
        out = []
        for p in self._report.get("phases", []) if self._report else []:
            out.append({"index": p["index"], "label": p["label"], "status": p["status"],
                        "deltaNs": float(p["deltaNs"]), "deltaStatus": p["deltaStatus"],
                        "similarity": p["similarity"], "ambiguous": p["ambiguous"],
                        "contributors": len(p["topContributors"])})
        return out

    @Property(int, notify=selectionChanged)
    def selectedPhase(self) -> int:
        return self._selected_phase

    @Slot(int)
    def selectPhase(self, index: int) -> None:
        n = len(self._report.get("phases", [])) if self._report else 0
        index = index if 0 <= index < n else -1
        if index != self._selected_phase:
            self._selected_phase = index
            self.selectionChanged.emit()

    @Property('QVariantMap', notify=selectionChanged)
    def selectedPhaseInfo(self) -> dict[str, Any]:
        if self._selected_phase < 0:
            return {}
        p = self._report["phases"][self._selected_phase]
        return {"index": p["index"], "label": p["label"], "status": p["status"],
                "deltaText": _signed(p["deltaNs"]), "criticalDeltaText": _signed(p["criticalDeltaNs"]),
                "similarity": p["similarity"], "ambiguous": p["ambiguous"],
                "baselineLabel": p["baseline"]["label"] if p["baseline"] else "—",
                "candidateLabel": p["candidate"]["label"] if p["candidate"] else "—",
                "hasCandidate": p["candidate"] is not None}

    @Property('QVariantList', notify=selectionChanged)
    def contributors(self) -> list[dict[str, Any]]:
        if not self._report:
            return []
        if self._selected_phase < 0:
            return self._list("contributors")
        p = self._report["phases"][self._selected_phase]
        out = []
        for i, t in enumerate(p["topContributors"]):
            r = self._rows_by_id.get(t["identity"])
            if r is None:
                continue
            c = _compact(r, self._joint_waits)
            c["rank"] = i + 1
            c["impactNs"] = float(t["impactNs"])
            out.append(c)
        return out

    @Property('QVariantList', notify=causalChanged)
    def propagated(self) -> list[dict[str, Any]]:
        return self._list("propagated") if self._report else []

    @Property('QVariantList', notify=causalChanged)
    def offCriticalPath(self) -> list[dict[str, Any]]:
        return self._list("offCriticalPath") if self._report else []

    @Property('QVariantList', notify=causalChanged)
    def causalImprovements(self) -> list[dict[str, Any]]:
        return self._list("improvements") if self._report else []

    @Property('QVariantList', notify=causalChanged)
    def newWork(self) -> list[dict[str, Any]]:
        return self._list("newWork") if self._report else []

    @Property('QVariantList', notify=causalChanged)
    def removedWork(self) -> list[dict[str, Any]]:
        return self._list("removedWork") if self._report else []

    def _composition(self, side: str) -> list[dict[str, Any]]:
        if not self._report or not self._report.get("criticalPath"):
            return []
        view = self._report["criticalPath"][side]
        comp = view["composition"] if self._selected_phase < 0 else \
            view["byPair"].get(str(self._selected_phase), [])
        total = sum(c["ns"] for c in comp) or 1
        shown = comp[:_CP_SLICES]
        dup = {c["label"] for c in shown if sum(1 for d in shown if d["label"] == c["label"]) > 1}

        def label(c):
            # same name in several roles (rank0 / rank1 ...): say which
            roles = c["roles"].split(", ")
            extra = next((r for r in roles[1:] if r), "") if c["label"] in dup else ""
            return dash.fmt_kernel_name(c["label"]) + (f" ({extra})" if extra else "")
        out = [{"identity": c["identity"], "label": label(c), "roles": c["roles"],
                "ns": float(c["ns"]), "frac": c["ns"] / total} for c in shown]
        rest = sum(c["ns"] for c in comp[_CP_SLICES:])
        if rest:
            out.append({"identity": -1, "label": f"{len(comp) - _CP_SLICES} other", "roles": "",
                        "ns": float(rest), "frac": rest / total})
        return out

    @Property('QVariantList', notify=selectionChanged)
    def criticalBefore(self) -> list[dict[str, Any]]:
        return self._composition("baseline")

    @Property('QVariantList', notify=selectionChanged)
    def criticalAfter(self) -> list[dict[str, Any]]:
        return self._composition("candidate")

    @Slot(int, result='QVariantMap')
    def contributorDetail(self, identity: int) -> dict[str, Any]:
        """Everything the details panel shows for one contributor, incl.
        the timeline ranges to jump to (candidate = the trace on screen) and
        the source location with a snippet when the file exists here."""
        r = self._rows_by_id.get(identity) or self._all_rows.get(identity)
        if r is None:
            return {}
        f = dash.fmt_ns
        m, d = r["measured"], r["derived"]

        def row(label, pair, kind, fmt=f):
            a, b = pair
            if a is None and b is None:
                return {"label": label, "before": "—", "after": "—", "delta": "—", "kind": "unavailable"}
            delta = _signed((b or 0) - (a or 0)) if fmt is f else f"{(b or 0) - (a or 0):+}"
            return {"label": label, "before": "—" if a is None else fmt(a), "after": "—" if b is None else fmt(b),
                    "delta": delta, "kind": kind}
        count_fmt = lambda v: str(v)  # noqa: E731
        rows = [row("calls", m["count"], "measured", count_fmt), row("self time", m["selfNs"], "measured"),
                row("total (inclusive)", m["totalNs"], "measured"),
                row("queue delay", m["queueNs"], "measured" if m["queueNs"][0] is not None else "unavailable"),
                row("overlapped with other work", m["overlapNs"], "measured"),
                row("critical-path time", d["criticalNs"], "derived"),
                row("idle on the path blamed on it", d["blameNs"], "derived"),
                row("waiting it caused (wait blame)", d.get("waitBlameNs", (None, None)), "derived")]
        if m["waitNs"][0] is not None or m["waitNs"][1] is not None:
            rows.insert(2, row("waiting", m["waitNs"], "measured"))
        pair = None
        tl = r.get("timeline") or {}
        if self._selected_phase >= 0:
            for p in r.get("pairs", []):
                if p["pair"] == self._selected_phase:
                    tl = {"baseline": p["baseline"], "candidate": p["candidate"], "pair": p["pair"]}
                    break
        pair = tl.get("pair")
        phase_label = self._report["phases"][pair]["label"] if pair is not None and pair < len(
            self._report["phases"]) else "whole run"
        snippet = None
        source = r.get("sourcePath") or ""
        if source:
            ctx = cc.read_source(source, self._cwd["candidate"]) or cc.read_source(source, self._cwd["baseline"])
            if ctx is not None:
                snippet = {"path": ctx["path"], "line": ctx["line"],
                           "lines": [{"line": n, "text": t, "hot": n == ctx["line"]} for n, t in ctx["lines"]]}
        cand = tl.get("candidate")
        base = tl.get("baseline")
        return {
            "id": r["id"], "label": dash.fmt_kernel_name(r["label"]), "roles": r["roles"], "path": r["path"],
            "category": r["category"], "rawName": r["rawName"], "bucket": r["bucket"],
            "cause": r["cause"], "causeLabel": r["causeLabel"], "status": r["status"],
            "impactNs": float(r["impactNs"]), "criticalDeltaNs": float(r.get("criticalDeltaNs", 0)),
            "receivedNs": float(r.get("receivedNs", 0)), "ownDeltaNs": float(r["ownDeltaNs"]),
            "explanation": r["explanation"],
            "alsoCauses": [f"{c['label']} ({_signed(c['ns'])})" for c in r["alsoCauses"]],
            "rows": rows, "evidence": list(r["evidence"]),
            "chain": [{"identity": c["identity"], "label": c["label"], "roles": c["roles"], "cause": c["cause"],
                       "selectable": c["identity"] in self._all_rows,
                       "causeLabel": cc.CAUSE_LABELS.get(c["cause"], c["cause"] or "—")} for c in r["chain"]],
            "attribution": self._attribution(r),
            "jointOrigins": self._joint_origins(r),
            "jointWaits": self._waits_released(r["id"]),
            "confidence": -1.0 if r["confidence"] is None else float(r["confidence"]),
            "matchKind": r["match"]["kind"],
            "matchScore": -1.0 if r["match"]["score"] is None else float(r["match"]["score"]),
            "source": r["source"], "snippet": snippet,
            "hasTimeline": cand is not None,
            "timelineStartNs": float(cand["startNs"]) if cand else 0.0,
            "timelineEndNs": float(cand["endNs"]) if cand else 0.0,
            "phaseLabel": phase_label,
            "baselineRangeText": (f"{f(base['endNs'] - base['startNs'])} window in the baseline"
                                  if base else "not in the baseline"),
        }

    def _attribution(self, r: dict[str, Any]) -> dict[str, Any]:
        """Three separate quantities: its own critical-path change (direct),
        its share of the waits it released with others (jointly caused),
        and the change of the waiting cp_attribution charges to its work
        (propagated wait blame)."""
        released = self._joint_waits.get(r["id"], [])
        joint = [w for w in released if w["shareNs"] is not None]
        wb = _pair_delta(r["derived"].get("waitBlameNs", (None, None)))
        return {
            "directNs": float(r.get("criticalDeltaNs", r["impactNs"])), "receivedNs": float(r.get("receivedNs", 0)),
            "impactNs": float(r["impactNs"]),
            "jointWaitNs": float(sum(w["shareNs"] for w in joint)) if joint else None,
            "jointWaits": len(released),
            "waitBlameDeltaNs": wb,
        }

    def _identity_brief(self, ident: int) -> dict[str, Any]:
        r = self._all_rows.get(ident)
        if r is None:
            return {"id": ident, "label": f"identity {ident}", "roles": "", "available": False, "hasTimeline": False}
        tl = (r.get("timeline") or {}).get("candidate")
        return {"id": ident, "label": dash.fmt_kernel_name(r["label"]), "roles": r["roles"], "available": True,
                "directNs": float(r.get("criticalDeltaNs", r["impactNs"])), "impactNs": float(r["impactNs"]),
                "waitBlameDeltaNs": _pair_delta(r["derived"].get("waitBlameNs", (None, None))),
                "hasTimeline": tl is not None, "timelineStartNs": float(tl["startNs"]) if tl else 0.0,
                "timelineEndNs": float(tl["endNs"]) if tl else 0.0,
                "category": r["category"], "rawName": r["rawName"]}

    def _joint_origins(self, r: dict[str, Any]) -> list[dict[str, Any]]:
        """A wait released jointly: every origin, the chain's and the joint
        ones, each with its own direct impact and its share of the wait."""
        o = r.get("propagatedFrom")
        joint = [x for x in (r.get("jointFrom") or []) if x != o]
        if o is None or not joint:
            return []
        dw = _wait_delta(r)
        n = 1 + len(joint)
        out = []
        for x, role in [(o, "chain origin")] + [(j, "joint origin") for j in joint]:
            b = self._identity_brief(x)
            b["role"] = role
            b["shareNs"] = None if dw is None else dw / n
            out.append(b)
        return out

    def _waits_released(self, ident: int) -> list[dict[str, Any]]:
        out = []
        for w in self._joint_waits.get(ident, []):
            b = self._identity_brief(w["wait"])
            b["shareNs"] = w["shareNs"]
            b["waitDeltaNs"] = w["waitDeltaNs"]
            b["joint"] = w["joint"]
            b["coOrigins"] = [self._identity_brief(x)["label"] + " (" + self._identity_brief(x)["roles"] + ")"
                              for x in w["origins"] if x != ident]
            out.append(b)
        return out

    @Slot(int, result='QVariantMap')
    def identityTimeline(self, identity: int) -> dict[str, Any]:
        """Where any compared identity ran in the candidate (for "Show in
        Timeline" from a chain step, a joint origin or a released wait)."""
        return self._identity_brief(int(identity))

    @Slot(int, result='QVariantMap')
    def phaseRange(self, index: int) -> dict[str, Any]:
        """The candidate's time window of an aligned phase (absolute ns), or
        {} for a phase that exists only in the baseline."""
        if not self._report or not (0 <= index < len(self._report["phases"])):
            return {}
        c = self._report["phases"][index]["candidate"]
        return {"startNs": float(c["startNs"]), "endNs": float(c["endNs"])} if c else {}

    # ── aggregate layer properties (compatibility view) ──────────────
    @Property('QVariantList', constant=True)
    def baselineFields(self) -> list[dict[str, Any]]:
        return self._baseline_fields

    @Property('QVariantList', constant=True)
    def comparisonFields(self) -> list[dict[str, Any]]:
        return self._comparison_fields

    @Property(QObject, constant=True)
    def table(self) -> TableBundle:
        return self._table

    @Property('QVariantList', notify=thresholdsChanged)
    def bucketDeltas(self) -> list[dict[str, Any]]:
        return self._bucket_rows

    @Property('QVariantList', notify=thresholdsChanged)
    def topImprovements(self) -> list[dict[str, Any]]:
        return self._top_improvements

    @Property('QVariantList', notify=thresholdsChanged)
    def topRegressions(self) -> list[dict[str, Any]]:
        return self._top_regressions

    @Property('QVariantList', constant=True)
    def baselineCoverage(self) -> list[float]:
        return self._baseline_coverage

    @Property('QVariantList', constant=True)
    def comparisonCoverage(self) -> list[float]:
        return self._comparison_coverage

    @Property(int, constant=True)
    def newCount(self) -> int:
        return self._new_count

    @Property(int, constant=True)
    def removedCount(self) -> int:
        return self._removed_count

    @Property('QVariantList', constant=True)
    def statusLegend(self) -> list[dict[str, Any]]:
        return [
            {"status": cmp.STATUS_IMPROVED, "label": "Improved"},
            {"status": cmp.STATUS_REGRESSED, "label": "Regressed"},
            {"status": cmp.STATUS_UNCHANGED, "label": "Unchanged (within noise floor)"},
            {"status": cmp.STATUS_NEW, "label": "New"},
            {"status": cmp.STATUS_REMOVED, "label": "Removed"},
        ]

    @Property('QVariantMap', notify=thresholdsChanged)
    def noiseFloor(self) -> dict[str, Any]:
        return {
            "pct": self._noise_pct, "ns": self._noise_ns,
            "note": "fixed disclosed heuristic threshold -- both a minimum percentage AND a "
                    "minimum absolute time must be cleared for a delta to count as a real "
                    "change, NOT a statistical significance test (hprofiler captures single-run "
                    "traces only, with no repeated-trial variance data to test against)",
        }

    # ── Mutators ─────────────────────────────────────────────────────
    @Slot(float, float)
    def setChangeThresholds(self, min_abs_ns: float, min_pct: float) -> None:
        """Re-classifies everything against new noise-floor thresholds --
        the aggregate rows without re-matching, the causal report by
        re-running the comparison on the cached projections."""
        if not self._available:
            return
        self._noise_ns = max(0.0, min_abs_ns)
        self._noise_pct = max(0.0, min_pct)
        # Everything derived from a status must be re-derived together.
        for row, raw in zip(self._rows, self._report_rows):
            status, delta = cmp.classify(raw["baseNs"], raw["compNs"], noise_pct=self._noise_pct, noise_ns=self._noise_ns)
            row["status"] = status
            raw["status"] = status
            raw["deltaKind"], raw["deltaReason"] = delta["kind"], delta["reason"]
        for b in self._bucket_rows:
            b["status"] = cmp.classify(b["baseNs"], b["compNs"], noise_pct=self._noise_pct, noise_ns=self._noise_ns)[0]
        self._refresh_top_changes()
        self._table.setRows(self._rows)
        self._recompute_causal()
        self.thresholdsChanged.emit()
        self.causalChanged.emit()
        self.selectionChanged.emit()

    @Slot(str, result=bool)
    def exportReport(self, path: str) -> bool:
        """Writes the comparison as JSON: the aggregate tables (unchanged
        keys) plus the full causal report under "causal"."""
        if not self._available:
            return False
        try:
            with open(path, "w") as f:
                json.dump({
                    "noiseFloor": self.noiseFloor,
                    "aggregates": self._report_rows,
                    "buckets": self._bucket_rows,
                    "topImprovements": self._top_improvements,
                    "topRegressions": self._top_regressions,
                    "newCount": self._new_count,
                    "removedCount": self._removed_count,
                    "causal": self._report,
                }, f, indent=2)
            return True
        except OSError:
            return False

    @Slot(str, result=bool)
    def exportCsv(self, path: str) -> bool:
        """Writes the function/kernel comparison table as CSV -- the
        RAW (unformatted) values."""
        if not self._available:
            return False
        try:
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["category", "name", "status", "matchKind",
                                  "baselineNs", "comparisonNs", "deltaNs", "deltaPct"])
                for r in self._report_rows:
                    writer.writerow([r["category"], r["name"], r["status"], r["matchKind"],
                                      r["baseNs"], r["compNs"], r["deltaNs"], r["deltaPct"]])
            return True
        except OSError:
            return False

    # ── side-by-side timelines ───────────────────────────────────────
    timelinesChanged = Signal()

    def attach_candidate_timeline(self, model) -> None:
        """The opened trace's TimelineModel (the candidate); the baseline
        gets its own. Both learn to colour events by their comparison
        status and to show their counterpart in the event details."""
        if not self._available:
            return
        from .models import TimelineModel
        self._timelines["candidate"] = model
        if self._timelines["baseline"] is None:
            self._timelines["baseline"] = TimelineModel(self._traces["baseline"], self._theme)
        for side, m in self._timelines.items():
            m._comparison_hook = (lambda lane, eid, side=side: self._detail_fields(side, eid))
            m._status_of = (lambda span, side=side: self._matcher.status_of(side, span)
                            if self._matcher is not None else "unmatched")
        self.timelinesChanged.emit()

    @Property(QObject, notify=timelinesChanged)
    def baselineTimeline(self):
        return self._timelines["baseline"]

    @Property(QObject, notify=timelinesChanged)
    def candidateTimeline(self):
        return self._timelines["candidate"]

    @Property(bool, notify=causalChanged)
    def eventMatching(self) -> bool:
        return self._matcher is not None and self._matcher.ok

    def _detail_fields(self, side: str, eid: int) -> list[dict[str, Any]]:
        m = self.matchEvent(side, float(eid))
        f = dash.fmt_ns
        if m.get("status") == "unavailable":
            return [{"label": "Counterpart", "value": "", "kind": "unavailable", "reason": m.get("basis", "")}]
        out = [{"label": "Comparison status", "value": m.get("status", ""), "kind": "measured", "reason": ""}]
        cp = m.get("counterpart")
        other = "baseline" if side == "candidate" else "candidate"
        if cp:
            out.append({"label": f"In the {other}", "value": f"{cp['name']}  dur {f(cp['durNs'])}", "kind": "derived",
                        "reason": "matched by comparison identity, aligned phase and occurrence index"})
            a, b = m["durationNs"]
            rel = f" ({m['relDelta'] * 100:+.1f}%)" if m.get("relDelta") is not None else ""
            out.append({"label": "Duration baseline → candidate", "value": f"{f(a)} → {f(b)}  Δ {dash.fmt_signed_ns(b - a)}{rel}",
                        "kind": "measured", "reason": ""})
        else:
            out.append({"label": f"In the {other}", "value": "", "kind": "unavailable", "reason": m.get("basis", "")})
        if m.get("confidence") is not None:
            out.append({"label": "Match confidence", "value": f"{m['confidence']:.2f}", "kind": "estimated",
                        "reason": "identity match score × phase alignment confidence (a heuristic)"})
        out.append({"label": "Basis", "value": m.get("basis", ""), "kind": "derived", "reason": ""})
        if m.get("explanation"):
            out.append({"label": "Why it changed", "value": m["explanation"], "kind": "derived",
                        "reason": "causal_compare: measured deltas + graph-derived dependencies"})
        return out

    @Slot(str, float, result='QVariantMap')
    def matchEvent(self, side: str, span_idx: float) -> dict[str, Any]:
        """The counterpart of an event of `side` ("baseline" / "candidate")
        in the other run, with durations, deltas, the match basis and its
        confidence, and the identity's causal explanation and chain."""
        if self._matcher is None:
            return {"status": "unavailable",
                    "basis": "no node-level comparison: " + (self._report.get("alignment", {}).get("fallbackReason")
                                                             or "comparison not loaded")}
        out = self._matcher.match(side, int(span_idx))
        cp = out.get("counterpart")
        other = "baseline" if side == "candidate" else "candidate"
        om = self._timelines.get(other)
        if cp and om is not None:
            span = self._traces[other].store.event_by_id(int(cp["eid"]))
            lane = om._lane_of_span(span) if span is not None else None
            cp["laneIndex"] = -1 if lane is None else int(lane)
        from .compare_timeline import regression_chain
        row = self._detail.get("rows", {}).get(out.get("identity"))
        out["steps"] = regression_chain(row) if row else []
        for k in ("measured", "derived"):
            out.pop(k, None)
        return out

    @Property('QVariantMap', notify=timelinesChanged)
    def timelineStarts(self) -> dict[str, float]:
        out = {}
        for side, m in self._timelines.items():
            if m is not None:
                out[side] = float(m.viewStartNs)
                out[side + "Duration"] = float(m.traceDurationNs)
        return out

    @Slot(str, int, int, float, float, result='QVariantMap')
    def alignmentOffset(self, mode: str, phase: int, identity: int, base_ns: float, cand_ns: float) -> dict[str, Any]:
        from .compare_timeline import alignment_offset
        st = self.timelineStarts
        row = self._detail.get("rows", {}).get(identity) if identity >= 0 else None
        return alignment_offset(mode, self._report or {}, phase=phase, identity_row=row,
                                event_pair=(base_ns, cand_ns) if base_ns > 0 and cand_ns > 0 else None,
                                starts=(st.get("baseline", 0.0), st.get("candidate", 0.0)))

    @Property('QVariantList', notify=causalChanged)
    def deltaStrip(self) -> list[dict[str, Any]]:
        from .compare_timeline import delta_strip
        return delta_strip(self._report or {})

    @Property('QVariantList', notify=timelinesChanged)
    def structureDiff(self) -> list[dict[str, Any]]:
        from .compare_timeline import structure_diff
        a, b = self._timelines["baseline"], self._timelines["candidate"]
        if a is None or b is None:
            return []
        return structure_diff(a._lane_extra, b._lane_extra)

    @Slot(int, result='QVariantList')
    def regressionSteps(self, identity: int) -> list[dict[str, Any]]:
        from .compare_timeline import regression_chain
        row = self._detail.get("rows", {}).get(identity)
        return regression_chain(row) if row else []

    @Slot(int, result='QVariantMap')
    def identitySpans(self, identity: int) -> dict[str, Any]:
        """What to highlight for an identity on each timeline: its raw name
        (both runs) -- the highlight is by name, the identity's own roles
        are in the label."""
        row = self._detail.get("rows", {}).get(identity)
        if not row:
            return {}
        return {"names": [row["rawName"]], "label": f"{row['label']} ({row['roles']})"}

    @Property('QVariantMap', notify=timelinesChanged)
    def evidenceChange(self) -> dict[str, Any]:
        """Critical-path time by evidence tier in each run (needs both
        runs' analyses; TimelineModel.startAnalysis on each)."""
        out = {}
        for side, m in self._timelines.items():
            if m is None:
                continue
            sm = m.criticalSummary
            out[side] = sm.get("tiers", {}) if sm else {}
            out[side + "State"] = m.analysisState
        return out

    @Slot()
    def refreshTimelines(self) -> None:
        self.timelinesChanged.emit()


def _signed(ns: float) -> str:
    return ("+" if ns >= 0 else "-") + dash.fmt_ns(abs(ns))
