"""
The details inspector's data layer -- given the current selection (see
src/gui/nav.py's Selection singleton), computes Summary/Context/Metrics/
Relationships/Recommendations content by QUERYING the other bridges'
already-computed data, not recomputing aggregated_stats()/_ct_build()/
etc. a second time.

Every field is tagged "measured" (a real number straight off the trace),
"derived" (computed from measured data, e.g. a merged-interval
percentage), "estimated" (a static-analysis heuristic, e.g. asm_advisor
hints), or "unavailable" (nothing to show, with a `reason` explaining
why) -- see this module's _field() helper. This is what makes "clearly
distinguish measured, derived, estimated, and unavailable values" a
concrete UI contract instead of an aspiration: nothing here is ever
silently blank.
"""
from __future__ import annotations

import json
from typing import Any

from PySide6.QtCore import QObject, Property, Signal, Slot
from PySide6.QtGui import QGuiApplication

from ..core.trace import Trace
from ..analysis import dashboard as dash


def _field(label: str, value: str, kind: str = "measured", reason: str = "") -> dict[str, str]:
    return {"label": label, "value": value, "kind": kind, "reason": reason}


def _node_in_tree(nodes: list[dict[str, Any]], category: str, name: str) -> bool:
    for n in nodes:
        if n["category"] == category and n["name"] == name:
            return True
        if _node_in_tree(n["children"], category, name):
            return True
    return False


class InspectorBridge(QObject):
    """Registered as the "Inspector" singleton. Recomputes `content`
    whenever the shared Selection changes -- the one bridge in this
    layer (besides KernelsBridge's disasm-polling) that's genuinely live
    rather than computed once at construction, since its whole job is to
    react to what the user just selected."""

    contentChanged = Signal()

    def __init__(self, trace: Trace, selection, kernels_bridge, call_tree_bridge,
                 roofline_bridge, source_bridge, timeline_model,
                 parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._trace = trace
        self._selection = selection
        self._kernels = kernels_bridge
        self._call_tree = call_tree_bridge
        self._roofline = roofline_bridge
        self._source = source_bridge
        self._timeline = timeline_model
        self._content: dict[str, list] = self._empty_content()

        selection.selectionChanged.connect(self._recompute)
        selection.callPathChanged.connect(self._recompute)

    @staticmethod
    def _empty_content() -> dict[str, list]:
        return {"summary": [], "context": [], "metrics": [], "relationships": [], "recommendations": []}

    def _recompute(self) -> None:
        category = self._selection.selectedCategory
        name = self._selection.selectedName
        if not name:
            self._content = self._empty_content()
            self.contentChanged.emit()
            return

        summary: list[dict] = [_field("Name", name), _field("Category", category)]
        context: list[dict] = []
        metrics: list[dict] = []
        relationships: list[dict] = []
        recommendations: list[dict] = []

        # Same (category,name) key trace.aggregated_stats() itself
        # builds -- reused directly, not recomputed a second time.
        row = None
        for r in self._trace.aggregated_stats():
            if r["category"] == category and r["name"] == name:
                row = r
                break

        if row is not None:
            summary.append(_field("Total time", dash.fmt_ns(row["total_ns"])))
            summary.append(_field("Call count", str(row["count"])))
            summary.append(_field("Share of total", f"{row['pct']:.1f}%", "derived"))
            metrics.append(_field("Avg duration", dash.fmt_ns(row["avg_ns"])))
            metrics.append(_field("Min duration", dash.fmt_ns(row["min_ns"])))
            metrics.append(_field("Max duration", dash.fmt_ns(row["max_ns"])))
        else:
            summary.append(_field("Total time", "", "unavailable", "no matching spans in this trace"))

        threads = sorted({
            (s.pid, s.tid) for s in self._trace.spans
            if s.category.value == category and s.name == name
        })
        if threads:
            # GPU spans get a synthetic tid far above any real OS tid
            # (chrome_trace.py's _GPU_BASE_TID, so GPU streams get their
            # own lanes in Perfetto/chrome://tracing) -- showing that raw
            # number here ("tid 2000000000") would be technically correct
            # but meaningless, so GPU categories are labeled by stream
            # instead of a fake thread id.
            if category in dash._GPU_CATS:
                shown = ", ".join(f"pid {p} (GPU stream {i})" for i, (p, t) in enumerate(threads[:5]))
            else:
                shown = ", ".join(f"pid {p} / tid {t}" for p, t in threads[:5])
            if len(threads) > 5:
                shown += f"  (+{len(threads) - 5} more)"
            context.append(_field("Runs on", shown))
        else:
            context.append(_field("Runs on", "", "unavailable", "no matching spans in this trace"))

        call_path = self._selection.selectedCallPath
        if call_path:
            context.append(_field("Call path", " → ".join(p.get("name", "") for p in call_path), "derived"))
        else:
            context.append(_field("Call path", "", "unavailable", "not selected from the Call Tree tab"))

        # Timeline occurrences -- reuses TimelineModel.findByName(), the
        # same lookup the "jump to Timeline" navigation action itself
        # uses, so the count shown here always matches what navigating
        # there actually finds.
        matches = self._timeline.findByName(category, name, 50)
        if matches:
            relationships.append(_field(
                "Timeline", f"{len(matches)}{'+' if len(matches) >= 50 else ''} occurrence(s)"))
        else:
            relationships.append(_field("Timeline", "", "unavailable", "no matching spans in this trace"))

        roofline_point = None
        if self._roofline.available:
            for p in self._roofline.points:
                if p["name"] == name:
                    roofline_point = p
                    break
        if roofline_point is not None:
            metrics.append(_field("Arithmetic intensity", f"{roofline_point['ai']:.2f} FLOP/B"))
            metrics.append(_field("Achieved throughput", f"{roofline_point['tflops']:.2f} TFLOP/s"))
            metrics.append(_field("Roofline bound", roofline_point["bound"], "derived"))
            relationships.append(_field("Roofline", "has a plotted point"))
        else:
            relationships.append(_field(
                "Roofline", "", "unavailable",
                "no roofline data for this trace" if not self._roofline.available
                else "no matching kernel in the roofline data"))

        if _node_in_tree(self._call_tree.roots, category, name):
            relationships.append(_field("Call Tree", "appears in the call tree"))
        else:
            relationships.append(_field(
                "Call Tree", "", "unavailable",
                "no call-stack data captured -- run with --call-tree or --perf-callgraph"))

        has_disasm = any(k.get("rawName") == name for k in self._source.kernels)
        if has_disasm:
            hints = self._source.advisorHints(name)
            if hints:
                for h in hints:
                    recommendations.append(_field(
                        h.get("category", "Analysis"), h.get("message", ""), "estimated",
                        "static analysis of the disassembled instruction stream"))
            else:
                recommendations.append(_field("Static analysis", "no notable issues found in this function's assembly"))
        else:
            recommendations.append(_field(
                "Static analysis", "", "unavailable",
                self._source.noDisasmReason(name) or "no disassembly collected for this function"))

        self._content = {
            "summary": summary, "context": context, "metrics": metrics,
            "relationships": relationships, "recommendations": recommendations,
        }
        self.contentChanged.emit()

    @Property('QVariantMap', notify=contentChanged)
    def content(self) -> dict[str, list]:
        return self._content

    @Slot(str)
    def copyToClipboard(self, text: str) -> None:
        clipboard = QGuiApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(text)

    @Slot(str, result=bool)
    def exportTo(self, path: str) -> bool:
        """Writes the CURRENT inspector content as JSON. Same json.dump
        pattern src/output/chrome_trace.py's write() already uses, not a
        new serialization approach."""
        try:
            with open(path, "w") as f:
                json.dump(self._content, f, indent=2)
            return True
        except OSError:
            return False
