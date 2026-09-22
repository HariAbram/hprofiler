"""
QObject models backing the Timeline screen (Phase 4) and, later, the
Kernels/Call Tree screens (Phase 5). Unlike bridge.py's DashboardBridge
(small, fixed-size data exposed as plain QVariantList properties), the
Timeline is span-count-sensitive -- a real GROMACS trace easily has tens
of thousands of spans (this project's own Dardel test run captured
60381) -- so lane/span data is fetched ON DEMAND via @Slot methods the
QML Canvas calls with the current viewport (start_ns, end_ns, pixel
width), not all at once as a constant Property. This mirrors
TimelineWidget's own numpy-vectorized spatial-index approach
(src/ui/app.py's _density_row) for the same reason: only compute what's
actually visible.
"""
from __future__ import annotations

import zlib
from typing import Any

import numpy as np
from PySide6.QtCore import QObject, Property, Slot

from ..core.trace import Trace
from ..analysis import criticalpath as _cp

# Same 16-hue palette concept as the TUI's _SPAN_PALETTE (src/ui/app.py),
# independently expressed as hex since QML wants CSS colors, not Rich
# style names. Order/hues deliberately mirror the TUI's so the same
# function tends to land in a visually similar slot on both UIs, though
# exact index parity isn't guaranteed (different palette sizes are fine
# either way -- both use the same crc32-modulo-then-probe algorithm).
_SPAN_HEX_PALETTE = [
    "#f87171", "#22d3ee", "#4ade80", "#e879f9", "#fbbf24", "#60a5fa",
    "#fb923c", "#c084fc", "#f472b6", "#2dd4bf", "#a3e635", "#facc15",
    "#818cf8", "#fca5a5", "#34d399", "#f0abfc",
]

_CONFIDENCE_HEX = {"certain": "#e6edf3", "high": "#22d3ee", "medium": "#8b949e"}


def _assign_span_colors(names: list[str]) -> dict[str, str]:
    """Deterministic per-function color, stable across runs (same
    crc32-then-open-addressing scheme as TimelineWidget._func_colors in
    src/ui/app.py -- see that code's comment for why crc32 and not
    Python's salted builtin hash())."""
    assigned: dict[str, int] = {}
    taken: set[int] = set()
    for name in sorted(set(names)):
        idx = zlib.crc32(name.encode("utf-8")) % len(_SPAN_HEX_PALETTE)
        while idx in taken and len(taken) < len(_SPAN_HEX_PALETTE):
            idx = (idx + 1) % len(_SPAN_HEX_PALETTE)
        assigned[name] = idx
        taken.add(idx)
    return {name: _SPAN_HEX_PALETTE[idx] for name, idx in assigned.items()}


class TimelineModel(QObject):
    """Backs the Timeline screen. Lane list is small (tens, not
    thousands) so it's a constant Property; span data within a lane is
    fetched per-viewport via visibleSpans()."""

    def __init__(self, trace: Trace, theme, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._trace = trace
        self._theme = theme
        self._lanes = trace.lanes()

        all_tids = sorted({
            int(ln.split("/thread-")[1]) for ln in self._lanes if "/thread-" in ln
        })
        tid_seq = {tid: i + 1 for i, tid in enumerate(all_tids)}

        lane_rank: dict[str, str] = {}
        for lane_name, spans in self._lanes.items():
            if not lane_name.startswith("mpi/"):
                continue
            for s in spans:
                rank = s.tags.get("rank")
                if rank is not None:
                    lane_rank[lane_name] = rank
                    break

        def _sort_key(name: str) -> tuple[int, str]:
            cat, _, suffix = name.partition("/")
            if suffix.startswith("thread-"):
                try:
                    return (tid_seq.get(int(suffix.removeprefix("thread-")), 9999), cat)
                except ValueError:
                    pass
            elif suffix.startswith("stream-"):
                try:
                    return (10000 + int(suffix.removeprefix("stream-")), cat)
                except ValueError:
                    pass
            return (0, cat)

        self._lane_names: list[str] = sorted(self._lanes.keys(), key=_sort_key)

        self._lane_meta: list[dict[str, Any]] = []
        for lane_name in self._lane_names:
            cat = lane_name.split("/")[0]
            suffix = lane_name.split("/", 1)[1] if "/" in lane_name else ""
            if lane_name in lane_rank:
                label = f"{cat} rank{lane_rank[lane_name]}"
            elif suffix.startswith("thread-"):
                try:
                    label = f"{cat} T{tid_seq.get(int(suffix.removeprefix('thread-')), '?')}"
                except ValueError:
                    label = cat
            elif suffix.startswith("stream-"):
                label = f"{cat} {suffix.replace('stream-', 'S')}"
            else:
                label = cat
            self._lane_meta.append({
                "name": lane_name,
                "label": label,
                "color": self._theme.categoryColor(cat),
                "count": len(self._lanes[lane_name]),
            })

        distinct_names = [s.name for spans in self._lanes.values() for s in spans]
        self._func_colors = _assign_span_colors(distinct_names)

        self._sorted_spans: dict[str, list] = {
            lane: sorted(spans, key=lambda s: s.start_ns)
            for lane, spans in self._lanes.items()
        }
        self._starts: dict[str, np.ndarray] = {}
        self._ends: dict[str, np.ndarray] = {}
        self._max_dur: dict[str, int] = {}
        for lane, slist in self._sorted_spans.items():
            if slist:
                self._starts[lane] = np.array([s.start_ns for s in slist], dtype=np.int64)
                self._ends[lane] = np.array([s.end_ns for s in slist], dtype=np.int64)
                self._max_dur[lane] = int(self._ends[lane].max() - self._starts[lane].min())
            else:
                self._starts[lane] = np.empty(0, dtype=np.int64)
                self._ends[lane] = np.empty(0, dtype=np.int64)
                self._max_dur[lane] = 0

        timed = [s for s in trace.spans if s.duration_ns > 0]
        if timed:
            self._view_start = min(s.start_ns for s in timed)
            self._view_end = max(s.end_ns for s in timed)
        else:
            self._view_start = trace.metadata.start_time_ns
            self._view_end = self._view_start + 1
        self._trace_dur = max(self._view_end - self._view_start, 1)

        # Cross-rank MPI/NCCL connectors -- same source data as the TUI's
        # TimelineWidget (criticalpath.py's resolved dependency graph),
        # different rendering (a real Canvas line here, not Braille dots).
        self._connectors: list[dict[str, Any]] = []
        try:
            cp_spans, cp_preds = _cp.build_dependency_graph(trace)
            span_lane: dict[int, str] = {
                id(s): lane for lane, spans in self._lanes.items() for s in spans
            }
            span_idx_in_lane: dict[int, int] = {}
            for lane, slist in self._sorted_spans.items():
                for i, s in enumerate(slist):
                    span_idx_in_lane[id(s)] = i
            for succ_idx, edges in cp_preds.items():
                succ = cp_spans[succ_idx]
                if succ.category.value not in ("mpi", "nccl"):
                    continue
                succ_lane = span_lane.get(id(succ))
                if succ_lane is None:
                    continue
                for pred_idx, kind, confidence in edges:
                    if kind not in ("p2p", "arrival"):
                        continue
                    pred = cp_spans[pred_idx]
                    pred_lane = span_lane.get(id(pred))
                    if pred_lane is None or pred_lane == succ_lane:
                        continue
                    self._connectors.append({
                        "predLane": self._lane_names.index(pred_lane),
                        "predSpanIdx": span_idx_in_lane.get(id(pred), -1),
                        "predMidNs": (pred.start_ns + pred.end_ns) / 2.0,
                        "succLane": self._lane_names.index(succ_lane),
                        "succSpanIdx": span_idx_in_lane.get(id(succ), -1),
                        "succMidNs": (succ.start_ns + succ.end_ns) / 2.0,
                        "color": _CONFIDENCE_HEX.get(confidence, "#8b949e"),
                    })
        except Exception:
            self._connectors = []

    # ── Constant properties ─────────────────────────────────────────────
    @Property('QVariantList', constant=True)
    def lanes(self) -> list[dict[str, Any]]:
        return self._lane_meta

    @Property(float, constant=True)
    def viewStartNs(self) -> float:
        return float(self._view_start)

    @Property(float, constant=True)
    def traceDurationNs(self) -> float:
        return float(self._trace_dur)

    @Property('QVariantList', constant=True)
    def connectors(self) -> list[dict[str, Any]]:
        return self._connectors

    # ── On-demand span queries ──────────────────────────────────────────
    @Slot(int, float, float, int, result='QVariantList')
    def visibleSpans(self, lane_index: int, view_start_ns: float, view_end_ns: float,
                     max_spans: int = 2000) -> list[dict[str, Any]]:
        """Spans in `lane_index` overlapping [view_start_ns, view_end_ns),
        via the same numpy searchsorted spatial-index trick
        TimelineWidget._density_row uses (src/ui/app.py) -- cheap even
        for a lane with tens of thousands of spans, since only the
        visible slice is ever materialized into Python dicts. Capped at
        `max_spans`: past that, the caller is zoomed out far enough that
        individual rectangles would be sub-pixel anyway (a coarse
        bucketed fallback, like the Overview preview's, is a possible
        future improvement, not implemented for this first pass)."""
        if lane_index < 0 or lane_index >= len(self._lane_names):
            return []
        lane = self._lane_names[lane_index]
        starts = self._starts[lane]
        if len(starts) == 0:
            return []
        max_dur = self._max_dur[lane]
        lo = int(np.searchsorted(starts, view_start_ns - max_dur, side="left"))
        hi = int(np.searchsorted(starts, view_end_ns, side="right"))
        if lo >= hi:
            return []
        slist = self._sorted_spans[lane]
        candidates = slist[lo:hi]
        if len(candidates) > max_spans:
            step = len(candidates) // max_spans + 1
            candidates = candidates[::step]
        return [
            {
                "startNs": float(s.start_ns),
                "durNs": float(max(s.duration_ns, 1)),
                "name": s.name,
                "color": self._func_colors.get(s.name, "#9ca3af"),
                "spanIdx": lo + i * (step if len(slist[lo:hi]) > max_spans else 1),
            }
            for i, s in enumerate(candidates)
        ]

    @Slot(int, int, result='QVariantMap')
    def spanAt(self, lane_index: int, span_idx: int) -> dict[str, Any]:
        """Full detail for one span (hover tooltip) by its index within
        the lane's time-sorted list -- the same index visibleSpans()
        returns per row, avoiding a second linear search."""
        if lane_index < 0 or lane_index >= len(self._lane_names):
            return {}
        lane = self._lane_names[lane_index]
        slist = self._sorted_spans[lane]
        if span_idx < 0 or span_idx >= len(slist):
            return {}
        s = slist[span_idx]
        return {
            "name": s.name,
            "category": s.category.value,
            "startNs": float(s.start_ns - self._view_start),
            "durNs": float(s.duration_ns),
            "tags": dict(s.tags),
        }
