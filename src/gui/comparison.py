"""
Backs the Compare screen (Main.qml's CompareScreen.qml, always tab index 9
per this project's "GUI tabs are always visible" convention -- see
DOCUMENTATION.md). Registered as the "Compare" singleton UNCONDITIONALLY,
even when no comparison trace was loaded (`trace_b=None`): an unregistered
QML singleton that a screen tries to bind against is a load error, and the
whole test suite's shared-engine class asserts zero warnings on every tab.

Reuses src/analysis/compare.py for all matching/classification (this file's
job, like every other bridge, is turning that data into QML-consumable
shapes -- see bridge.py's own docstring). Scope decision, disclosed here
rather than silently: this bridge compares FUNCTIONS/KERNELS (the `table`
below) and activity BUCKETS (`bucketDeltas`), not call-tree nodes or
system/PMU metrics -- those would each need their own matching scheme
(path-based for the tree, a differently-shaped table for scalar metrics)
and hierarchical/typed diff rendering, a second phase roughly the size of
this one for comparatively narrower value than the function-level and
bucket-level views, which already cover "communication/synchronization/
memory activity" (via buckets) and "kernels/functions" (via the table)
from the comparison requirements.
"""
from __future__ import annotations

import csv
import json
from typing import Any

from PySide6.QtCore import QObject, Property, Signal, Slot

from ..core.trace import Trace
from ..analysis import compare as cmp
from ..analysis import dashboard as dash
from . import columns
from .tablemodel import TableBundle

_COVERAGE_BUCKETS = 60


def _summary_fields(trace: Trace, *, available: bool, reason: str = "") -> list[dict[str, Any]]:
    if not available:
        return [
            {"label": label, "value": "", "kind": "unavailable", "reason": reason}
            for label in ("Executable", "Wall time", "Spans", "Kernels/functions")
        ]
    meta = trace.metadata
    stats = trace.aggregated_stats()
    timed = [s for s in trace.spans if s.duration_ns > 0]
    wall_ns = (max(s.end_ns for s in timed) - min(s.start_ns for s in timed)) if timed else 0
    return [
        {"label": "Executable", "value": meta.command or "(unknown)", "kind": "measured", "reason": ""},
        {"label": "Wall time", "value": dash.fmt_ns(wall_ns), "kind": "measured", "reason": ""},
        {"label": "Spans", "value": dash.fmt_count(len(trace.spans)), "kind": "measured", "reason": ""},
        {"label": "Kernels/functions", "value": dash.fmt_count(len(stats)), "kind": "measured", "reason": ""},
    ]


class ComparisonBridge(QObject):
    thresholdsChanged = Signal()

    def __init__(self, trace_a: Trace, trace_b: Trace | None, theme, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._theme = theme
        self._available = trace_b is not None
        self._noise_pct = cmp.NOISE_FLOOR_PCT
        self._noise_ns = cmp.NOISE_FLOOR_NS

        self._baseline_fields = _summary_fields(trace_a, available=True)
        self._comparison_fields = _summary_fields(
            trace_b, available=self._available,
            reason="no comparison trace loaded -- run "
                   "`hprofiler gui trace1.json --compare trace2.json`")

        if self._available:
            self._recompute(trace_a, trace_b)
        else:
            self._rows: list[dict[str, Any]] = []
            self._bucket_rows: list[dict[str, Any]] = []
            self._top_improvements: list[dict[str, Any]] = []
            self._top_regressions: list[dict[str, Any]] = []
            self._baseline_coverage: list[float] = [0.0] * _COVERAGE_BUCKETS
            self._comparison_coverage: list[float] = [0.0] * _COVERAGE_BUCKETS
            self._new_count = 0
            self._removed_count = 0

        self._table = TableBundle(self._rows, columns.COMPARE_COLUMNS, self)

        # trace_a/trace_b are NOT kept as instance attributes beyond
        # __init__ -- loading two full traces roughly doubles peak
        # memory, and every comparison this bridge exposes is
        # precomputed above; nothing after __init__ needs to hold either
        # trace alive (the comparison trace is also never loaded with
        # collect_disasm=True, for the same reason -- see app.py's argv
        # parsing).

    def _recompute(self, trace_a: Trace, trace_b: Trace) -> None:
        rows = cmp.compare_aggregates(trace_a, trace_b, noise_pct=self._noise_pct, noise_ns=self._noise_ns)
        self._rows = [
            {
                "name": dash.fmt_kernel_name(r["name"]), "category": r["category"], "status": r["status"],
                "baselineNs": r["baseNs"] if r["baseNs"] is not None else 0.0,
                "comparisonNs": r["compNs"] if r["compNs"] is not None else 0.0,
                "deltaNs": r["deltaNs"] if r["deltaNs"] is not None else 0.0,
                "deltaPct": r["deltaPct"] if r["deltaPct"] is not None else 0.0,
                "matchKind": r["matchKind"],
            }
            for r in rows
        ]
        self._bucket_rows = cmp.compare_buckets(trace_a, trace_b, noise_pct=self._noise_pct, noise_ns=self._noise_ns)
        self._top_improvements = cmp.top_changes(rows, status=cmp.STATUS_IMPROVED, limit=10)
        self._top_regressions = cmp.top_changes(rows, status=cmp.STATUS_REGRESSED, limit=10)
        self._baseline_coverage = cmp.normalized_coverage(trace_a, _COVERAGE_BUCKETS)
        self._comparison_coverage = cmp.normalized_coverage(trace_b, _COVERAGE_BUCKETS)
        self._new_count = sum(1 for r in rows if r["status"] == cmp.STATUS_NEW)
        self._removed_count = sum(1 for r in rows if r["status"] == cmp.STATUS_REMOVED)
        self._report_rows = rows   # raw (unformatted) rows, for exportReport/exportCsv

    # ── Properties ───────────────────────────────────────────────────
    @Property(bool, constant=True)
    def available(self) -> bool:
        return self._available

    @Property('QVariantList', constant=True)
    def baselineFields(self) -> list[dict[str, Any]]:
        return self._baseline_fields

    @Property('QVariantList', constant=True)
    def comparisonFields(self) -> list[dict[str, Any]]:
        return self._comparison_fields

    @Property(QObject, constant=True)
    def table(self) -> TableBundle:
        return self._table

    @Property('QVariantList', constant=True)
    def bucketDeltas(self) -> list[dict[str, Any]]:
        return self._bucket_rows

    @Property('QVariantList', constant=True)
    def topImprovements(self) -> list[dict[str, Any]]:
        return self._top_improvements

    @Property('QVariantList', constant=True)
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
        """Re-classifies every row against new noise-floor thresholds --
        does NOT re-match rows (matching is threshold-independent), so
        this is cheap even on a trace with many distinct kernels."""
        if not self._available:
            return
        self._noise_ns = max(0.0, min_abs_ns)
        self._noise_pct = max(0.0, min_pct)
        for row, raw in zip(self._rows, self._report_rows):
            status, delta = cmp.classify(raw["baseNs"], raw["compNs"], noise_pct=self._noise_pct, noise_ns=self._noise_ns)
            row["status"] = status
        self._table.setRows(self._rows)
        self.thresholdsChanged.emit()

    @Slot(str, result=bool)
    def exportReport(self, path: str) -> bool:
        """Writes the full comparison (aggregates + buckets + top
        changes + noise-floor disclosure) as JSON -- same json.dump
        pattern InspectorBridge.exportTo() already uses."""
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
                }, f, indent=2)
            return True
        except OSError:
            return False

    @Slot(str, result=bool)
    def exportCsv(self, path: str) -> bool:
        """Writes the function/kernel comparison table as CSV -- the
        RAW (unformatted) values, same "export what's underneath the
        display formatting" convention TableBundle.exportCsv() uses."""
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
