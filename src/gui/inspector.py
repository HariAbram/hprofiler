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

from ..core.trace import Trace
from ..core import gpu_activity as ga
from ..analysis import dashboard as dash
from . import clipboard


_SAMPLE = 50_000


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

        # A bounded sample (first _SAMPLE occurrences in arrival order) --
        # one name can have millions of spans in a large trace; totals above
        # come from the store's aggregates, not from this sample.
        from ..core.store import SpanFilter
        here = []
        for s in self._trace.iter_spans(filt=SpanFilter(names=frozenset({name}),
                                                       categories=frozenset({category}))):
            here.append(s)
            if len(here) >= _SAMPLE:
                break
        sampled = len(here) >= _SAMPLE

        # Device-side spans (a kernel/copy on the GPU) ran on a stream, not
        # on the thread that launched them; host calls -- including CUDA/
        # HIP launch calls and OpenCL enqueues -- ran on their thread.
        def _on_device(s) -> bool:
            side = s.tags.get("side")
            return side == "gpu" or (side is None and category in dash._GPU_CATS)

        places = sorted({
            (s.pid, "stream " + str(s.tags["stream"]) if "stream" in s.tags else "device")
            if _on_device(s) else (s.pid, f"tid {s.tid}")
            for s in here
        }, key=str)
        if places:
            shown = ", ".join(f"pid {p} / {where}" for p, where in places[:5])
            if len(places) > 5:
                shown += f"  (+{len(places) - 5} more)"
            context.append(_field("Runs on", shown))
        else:
            context.append(_field("Runs on", "", "unavailable", "no matching spans in this trace"))

        self._gpu_timing_fields(here, metrics, relationships)
        if sampled:
            metrics.append(_field("Sample", f"first {_SAMPLE:,} occurrences", "derived",
                                  "per-span details above use a sample; totals use every span"))

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

        # allRoots, NOT roots -- the table-upgrade round made `roots` a
        # live, filterable view (CallTreeBridge.setFilter()); querying
        # that instead would mean an unrelated Call Tree text filter
        # could silently flip this relationship to "unavailable".
        if _node_in_tree(self._call_tree.allRoots, category, name):
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

    def _gpu_timing_fields(self, here: list, metrics: list, relationships: list) -> None:
        """Where a GPU span's numbers come from: device-measured (CUPTI /
        ROCprofiler-SDK), a host-side proxy, or the host call itself --
        plus the host-call / queueing / execution split when measured."""
        device = [s for s in here if s.tags.get("side") == "gpu" or ga.is_device_kernel(s)]
        if device:
            sources = {ga.timing_source(s) for s in device}
            if sources == {"device"}:
                tracer = sorted({s.tags.get("src", "native") for s in device})
                metrics.append(_field("Device timing", "measured on the device (" + ", ".join(tracer) + ")"))
            elif sources <= {"proxy_event"}:
                metrics.append(_field(
                    "Device timing", "duration from GPU events; start = submission time", "estimated",
                    "no CUPTI / ROCprofiler-SDK records: queued work appears to start early"))
            elif sources <= ga.PROXY_TIMINGS:
                metrics.append(_field(
                    "Device timing", "host-side proxy only (no device measurement)", "estimated",
                    "GPU event timing was unavailable for these launches"))
            elif "host" in sources and len(sources) == 1:
                metrics.append(_field("Wait timing", "host wait recorded by the native tracer"))
            else:
                metrics.append(_field("Device timing", "mixed: " + ", ".join(sorted(sources)), "estimated"))

            def _median(key: str) -> int | None:
                vals = sorted(int(s.tags[key]) for s in device if key in s.tags)
                return vals[len(vals) // 2] if vals else None
            api, queue = _median("api_ns"), _median("queue_ns")
            if api is not None:
                metrics.append(_field("Host call (median)", dash.fmt_ns(api)))
            if queue is not None:
                metrics.append(_field("Queued before start (median)", dash.fmt_ns(queue), "derived"))
            if api is not None or queue is not None:
                dur = sorted(s.duration_ns for s in device)
                metrics.append(_field("Device execution (median)", dash.fmt_ns(dur[len(dur) // 2])))
            linked = sum(1 for s in device if s.parent_span_id)
            if any(ga.is_device_span(s) for s in device):
                relationships.append(_field("Submitted by", f"{linked} of {len(device)} correlated to a host call",
                                            "derived"))
        hosts = [s for s in here if ga.is_host_submission(s)]
        if hosts:
            sids = {s.span_id for s in hosts if s.span_id}
            n_dev = sum(1 for s in self._trace.iter_spans(gpu_model=True)
                        if s.parent_span_id in sids and ga.is_device_span(s))
            relationships.append(_field("Device work", f"{n_dev} device span(s) from {len(hosts)} call(s)",
                                        "derived"))

    @Property('QVariantMap', notify=contentChanged)
    def content(self) -> dict[str, list]:
        return self._content

    @Slot(str)
    def copyToClipboard(self, text: str) -> None:
        clipboard.copy_text(text)

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
