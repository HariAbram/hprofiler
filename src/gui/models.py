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

import math
import re
import zlib
from typing import Any

import numpy as np
from PySide6.QtCore import QObject, Property, Signal, Slot

from ..core.trace import Trace
from ..analysis import criticalpath as _cp
from ..analysis import activity_buckets
from ..analysis import dashboard as dash

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
    fetched per-viewport via visibleSpans().

    `rows` (live, notify=rowsChanged) is the NEW visual row list added for
    filtering/grouping -- group headers + filtered/ordered/hidden-aware
    lane references -- that QML iterates instead of `lanes` directly. A
    "lane" row always carries its ORIGINAL `laneIndex`, so `lanes`/
    `visibleSpans`/`spanAt`/`findByName`/`callGraph` all keep their
    existing physical-lane-index addressing completely unchanged --
    Round 16's cross-tab navigation (Nav/Inspector, double-click-to-zoom)
    is built on top of that addressing and must not need to change."""

    rowsChanged = Signal()
    filtersChanged = Signal()
    groupingChanged = Signal()
    colorModeChanged = Signal()
    searchChanged = Signal()
    bookmarksChanged = Signal()
    namedRangesChanged = Signal()

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
        self._lane_rank = lane_rank

        # Per-lane process/thread/stream identity for filtering
        # (applyFilters()) -- a "lane" is one category/thread-or-stream
        # combination, so these are derived once here rather than
        # re-parsed per filter application. pid is "whichever process this
        # lane's spans belong to" (first-seen, same convention as
        # lane_rank above) -- in practice a lane never mixes pids since
        # lane keys already partition by thread/stream id, which doesn't
        # repeat across a single merged trace's processes in the traces
        # this codebase has seen so far.
        self._lane_pid: dict[str, int] = {}
        self._lane_tid: dict[str, int | None] = {}
        self._lane_stream: dict[str, str | None] = {}
        for lane_name, spans in self._lanes.items():
            self._lane_pid[lane_name] = spans[0].pid if spans else 0
            suffix = lane_name.split("/", 1)[1] if "/" in lane_name else ""
            tid = None
            stream = None
            if suffix.startswith("thread-"):
                try:
                    tid = int(suffix.removeprefix("thread-"))
                except ValueError:
                    tid = None
            elif suffix.startswith("stream-"):
                stream = suffix.removeprefix("stream-")
            self._lane_tid[lane_name] = tid
            self._lane_stream[lane_name] = stream

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
        self._lane_index: dict[str, int] = {name: i for i, name in enumerate(self._lane_names)}

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

        # Filter state (applyFilters()/clearFilters()) -- empty dict means
        # "no active filter", so _rebuild_rows() below (called before
        # _sorted_spans/_starts/_ends exist -- it only needs lane metadata)
        # can run safely with self._event_mask empty: every lane's
        # filteredCount just falls back to its full count until a real
        # filter is applied later, well after __init__ completes.
        self._filters: dict[str, Any] = {}
        self._event_mask: dict[str, np.ndarray] = {}

        # Grouping/hide/isolate/reorder state (Phase B3) -- also needs no
        # span data, safe to initialize before _sorted_spans/_starts/
        # _ends exist, same reasoning as the filter state just above.
        self._grouping: str = "none"
        self._group_collapsed: dict[str, bool] = {}
        self._hidden_lanes: set[str] = set()
        # None = "not isolating" (every non-hidden lane shows); a set,
        # even an empty one, means "isolate mode is active, show only
        # these lanes".
        self._isolated_lanes: set[str] | None = None
        # None = default (self._lane_names) order; a list = a custom
        # order, always covering every lane name (moveRow/setRowOrder
        # both maintain that invariant, see their own docstrings).
        self._row_order: list[str] | None = None

        self._rows: list[dict[str, Any]] = []
        self._rebuild_rows()

        distinct_names = [s.name for spans in self._lanes.values() for s in spans]
        self._func_colors = _assign_span_colors(distinct_names)
        # Precomputed once here (not per-repaint, unlike the cheap
        # frozenset-membership bucket_of_span() lookup itself) since
        # Theme.bucketColor()/categoryColor() do a palette dict lookup --
        # visibleSpans() is a hot path (called per lane on every pan/zoom
        # frame), so setColorMode("bucket"/"category") must stay just as
        # cheap as the existing function-name coloring, which _func_colors
        # already precomputes the same way.
        self._color_mode: str = "function"
        self._bucket_colors: dict[str, str] = {
            b: self._theme.bucketColor(b) for b in activity_buckets.BUCKETS
        }
        distinct_categories = {s.category.value for spans in self._lanes.values() for s in spans}
        self._category_colors: dict[str, str] = {
            c: self._theme.categoryColor(c) for c in distinct_categories
        }

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
                # Longest INDIVIDUAL span in this lane -- the look-back margin
                # visibleSpans() needs so a span starting just before the
                # viewport but extending into it isn't missed. NOT the lane's
                # overall first-to-last time range (ends.max()-starts.min()),
                # which for a lane whose spans are spread across most of the
                # trace pins searchsorted's lower bound near index 0
                # regardless of how far into the trace the viewport actually
                # is -- flooding candidates with spans nowhere near the
                # visible window (see visibleSpans()'s docstring).
                self._max_dur[lane] = int((self._ends[lane] - self._starts[lane]).max())
            else:
                self._starts[lane] = np.empty(0, dtype=np.int64)
                self._ends[lane] = np.empty(0, dtype=np.int64)
                self._max_dur[lane] = 0

        # (category, name) -> [(laneIndex, spanIdx), ...], built once here
        # so both findByName() and search() below stop being O(total
        # spans) linear scans -- iterating this index's KEYS (one per
        # distinct (category,name) pair, typically far fewer than the
        # trace's total span count) instead. Occurrence order within each
        # list matches _sorted_spans' time order (lanes visited in
        # _lane_names order, spans in time order within each), so
        # findByName()'s existing output ordering is unchanged.
        self._name_index: dict[tuple[str, str], list[tuple[int, int]]] = {}
        for lane_idx, lane in enumerate(self._lane_names):
            for span_idx, s in enumerate(self._sorted_spans[lane]):
                self._name_index.setdefault((s.category.value, s.name), []).append((lane_idx, span_idx))

        # Search state (Phase B4) -- also safe to init before anything
        # else below needs it, same reasoning as filter/grouping state.
        self._search_active: bool = False
        self._search_matches: list[tuple[int, int]] = []
        self._search_match_set: set[tuple[int, int]] = set()
        self._search_cursor: int = -1

        # Bookmarks / named ranges (Phase B5) -- Timeline-specific view
        # state, lives here rather than on Nav (cross-tab selection), same
        # reasoning as the filter/grouping state above.
        self._bookmarks: list[dict[str, Any]] = []
        self._named_ranges: list[dict[str, Any]] = []
        self._next_bookmark_id: int = 1
        self._next_range_id: int = 1

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

    # ── Rows (live: filtering/grouping/reordering) ──────────────────────
    def _lane_passes_static_filters(self, lane_name: str) -> bool:
        """Lane-level (whole-row) filter dimensions -- rank/process/
        thread/runtime/stream are properties of the LANE itself (every
        span in a lane shares them), so a lane either passes entirely or
        is dropped from `rows` entirely. Event-level dimensions (name/
        duration/time-range/bucket) are handled separately by
        `_event_mask`, since individual spans within a passing lane can
        still differ on those."""
        f = self._filters
        if f.get("ranks") and self._lane_rank.get(lane_name) not in f["ranks"]:
            return False
        if f.get("processes") and self._lane_pid.get(lane_name) not in f["processes"]:
            return False
        if f.get("threads") and self._lane_tid.get(lane_name) not in f["threads"]:
            return False
        if f.get("runtimes") and lane_name.split("/")[0] not in f["runtimes"]:
            return False
        if f.get("streams") and self._lane_stream.get(lane_name) not in f["streams"]:
            return False
        return True

    def _effective_lane_order(self) -> list[str]:
        """Base lane name order `_rebuild_rows()` starts from: the custom
        `_row_order` if one was set (moveRow/setRowOrder), else the
        original `_lane_names` order -- with hidden lanes and, when
        isolate mode is active, everything except the isolated set,
        dropped. Filtering (rank/process/etc.) is applied separately in
        `_rebuild_rows()`, not here, so this stays reusable for both the
        flat and grouped row-building paths without duplicating that
        logic."""
        order = self._row_order if self._row_order is not None else self._lane_names
        out = []
        for lane in order:
            if lane not in self._lane_index:
                continue
            if lane in self._hidden_lanes:
                continue
            if self._isolated_lanes is not None and lane not in self._isolated_lanes:
                continue
            out.append(lane)
        return out

    def _lane_row(self, lane: str) -> dict[str, Any] | None:
        """One `kind: "lane"` row dict for `lane`, honoring the active
        event-level filter's `activeOnly` toggle (returns None to mean
        "drop this row entirely") -- shared by both the flat (no
        grouping) and grouped row-building paths in `_rebuild_rows()`."""
        i = self._lane_index[lane]
        m = self._lane_meta[i]
        mask = self._event_mask.get(lane)
        filtered_count = int(mask.sum()) if mask is not None else m["count"]
        if self._filters.get("activeOnly") and filtered_count == 0:
            return None
        return {
            "kind": "lane",
            "laneIndex": i,
            "name": lane,
            "label": m["label"],
            "color": m["color"],
            "count": m["count"],
            "filteredCount": filtered_count,
        }

    def _group_key_for_lane(self, lane_name: str, grouping: str) -> Any:
        """The value `lane_name` takes for grouping dimension `grouping`
        -- None means "no data for this dimension" (e.g. every "device"
        lookup, or "rank" for a non-MPI lane), which `_rebuild_rows()`
        collects into one honestly-labeled "(unavailable)" group rather
        than silently omitting those lanes."""
        if grouping == "rank":
            return self._lane_rank.get(lane_name)
        if grouping == "process":
            return self._lane_pid.get(lane_name)
        if grouping == "runtime":
            return lane_name.split("/")[0]
        if grouping == "stream":
            return self._lane_stream.get(lane_name)
        if grouping == "thread":
            return self._lane_tid.get(lane_name)
        # "device": no per-span device field exists anywhere in this
        # trace format (see filterDimensions' own docstring) -- every
        # lane falls into the one "(unavailable)" group rather than a
        # fabricated per-lane value.
        return None

    @staticmethod
    def _group_label(grouping: str, key: Any) -> str:
        if key is None:
            return "(unavailable)"
        prefix = {
            "rank": "Rank", "process": "Process", "runtime": "Runtime",
            "stream": "Stream", "thread": "Thread", "device": "Device",
        }.get(grouping, grouping)
        return f"{prefix} {key}"

    def _rebuild_rows(self) -> None:
        """Recomputes the visual `rows` list -- lane-level filters +
        hide/isolate + custom order (via `_effective_lane_order()`),
        event-level filters (via `_lane_row()`), then, if `_grouping` is
        active, bucketed into group headers (each carrying every member
        lane's original `laneIndex` in `laneIndexes`, for
        `groupCoverage()`) with member lane rows following unless that
        group is collapsed."""
        lane_names = [ln for ln in self._effective_lane_order()
                      if self._lane_passes_static_filters(ln)]

        if self._grouping == "none":
            rows = []
            for lane in lane_names:
                r = self._lane_row(lane)
                if r is not None:
                    rows.append(r)
            self._rows = rows
            self.rowsChanged.emit()
            return

        groups: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        for lane in lane_names:
            r = self._lane_row(lane)
            if r is None:
                continue
            key = self._group_key_for_lane(lane, self._grouping)
            group_id = f"{self._grouping}:{key}"
            if group_id not in groups:
                groups[group_id] = {"label": self._group_label(self._grouping, key),
                                     "laneIndexes": [], "laneRows": []}
                order.append(group_id)
            groups[group_id]["laneIndexes"].append(r["laneIndex"])
            groups[group_id]["laneRows"].append(r)

        rows = []
        for group_id in order:
            g = groups[group_id]
            collapsed = self._group_collapsed.get(group_id, False)
            rows.append({
                "kind": "group",
                "groupId": group_id,
                "label": g["label"],
                "laneIndexes": g["laneIndexes"],
                "laneCount": len(g["laneRows"]),
                "count": sum(lr["count"] for lr in g["laneRows"]),
                "filteredCount": sum(lr["filteredCount"] for lr in g["laneRows"]),
                "collapsed": collapsed,
            })
            if not collapsed:
                rows.extend(g["laneRows"])
        self._rows = rows
        self.rowsChanged.emit()

    @Property('QVariantList', notify=rowsChanged)
    def rows(self) -> list[dict[str, Any]]:
        return self._rows

    # ── Filtering ────────────────────────────────────────────────────
    @Property('QVariantList', constant=True)
    def filterDimensions(self) -> list[dict[str, Any]]:
        """Describes what can be filtered on and, for each dimension,
        the values actually present in this trace -- QML builds
        TimelineFilterBar's controls off this instead of hardcoding a
        list, so a dimension with no data (e.g. no MPI ranks in a
        single-process CPU trace) can honestly report `available: false`
        with a `reason` rather than showing an empty, confusing control.
        `device` always reports unavailable -- no per-span device field
        exists anywhere in this codebase (only per-run DevicePeak
        capability data and per-stream lane naming), so it would be
        fabricated rather than derived."""
        ranks = sorted({r for r in self._lane_rank.values()}, key=str)
        processes = sorted({p for p in self._lane_pid.values()})
        threads = sorted({t for t in self._lane_tid.values() if t is not None})
        runtimes = sorted({name.split("/")[0] for name in self._lane_names})
        streams = sorted({s for s in self._lane_stream.values() if s is not None})

        def _dim(key: str, label: str, values: list, supported: bool = True, reason: str = "") -> dict[str, Any]:
            is_available = supported and bool(values)
            return {
                "key": key,
                "label": label,
                "available": is_available,
                "reason": "" if is_available else reason,
                "values": [{"value": v, "label": str(v)} for v in values],
            }

        return [
            _dim("rank", "MPI rank", ranks, reason="no MPI rank tags in this trace"),
            _dim("process", "Process", processes, reason="only one process in this trace"),
            _dim("thread", "Thread", threads, reason="no per-thread lanes in this trace"),
            _dim("runtime", "Runtime", runtimes, reason="no runtime categories in this trace"),
            _dim("stream", "Stream", streams, reason="no GPU stream tags in this trace"),
            _dim("device", "Device", [], supported=False,
                 reason="no per-span device field exists in this trace format -- "
                        "only per-run device capability data and per-stream lane naming"),
            _dim("bucket", "Activity", list(activity_buckets.BUCKETS)),
        ]

    @Property('QVariantMap', notify=filtersChanged)
    def activeFilters(self) -> dict[str, Any]:
        return self._filters

    @Property(int, notify=filtersChanged)
    def hiddenRowCount(self) -> int:
        return len(self._lane_meta) - len(self._rows)

    @Property(int, notify=filtersChanged)
    def filteredSpanCount(self) -> int:
        return sum(r["filteredCount"] for r in self._rows)

    @Property(int, constant=True)
    def totalSpanCount(self) -> int:
        return sum(m["count"] for m in self._lane_meta)

    def _recompute_event_mask(self) -> None:
        """Per-lane boolean mask (True = this span passes every active
        EVENT-level filter) -- computed once here, on filter change, not
        per repaint frame. visibleSpans() ANDs the relevant slice of this
        into its own numpy mask (cheap, vectorized); `_rebuild_rows()`
        sums a lane's full mask for its `filteredCount`."""
        f = self._filters
        name_query = f.get("nameQuery") or ""
        is_regex = bool(f.get("nameIsRegex"))
        min_dur = float(f.get("minDurationNs") or 0)
        buckets = set(f.get("buckets") or [])
        time_range_only = bool(f.get("timeRangeOnly"))
        range_start = float(f.get("rangeStartNs") or 0)
        range_end = float(f.get("rangeEndNs") or 0)

        regex = None
        if is_regex and name_query:
            try:
                regex = re.compile(name_query, re.IGNORECASE)
            except re.error:
                regex = None
        name_query_lower = name_query.lower()

        needs_per_event = bool(name_query) or bool(buckets)

        self._event_mask = {}
        if not (min_dur > 0 or needs_per_event or time_range_only):
            return

        for lane in self._lane_names:
            slist = self._sorted_spans[lane]
            n = len(slist)
            if n == 0:
                continue
            mask = np.ones(n, dtype=bool)
            if min_dur > 0:
                mask &= (self._ends[lane] - self._starts[lane]) >= min_dur
            if time_range_only:
                mask &= (self._ends[lane] > range_start) & (self._starts[lane] < range_end)
            if needs_per_event:
                for i, s in enumerate(slist):
                    if not mask[i]:
                        continue
                    if regex is not None:
                        if not regex.search(s.name):
                            mask[i] = False
                            continue
                    elif name_query_lower and name_query_lower not in s.name.lower():
                        mask[i] = False
                        continue
                    if buckets and activity_buckets.bucket_of_span(s) not in buckets:
                        mask[i] = False
            self._event_mask[lane] = mask

    @Slot('QVariantMap')
    def applyFilters(self, spec: dict) -> None:
        """Replaces the active filter spec wholesale (QML always passes
        the FULL current filter-bar state, not a partial patch -- keeps
        this the single source of truth instead of needing get-then-
        merge semantics on both sides). Recognized keys: ranks/
        processes/threads/runtimes/streams/buckets (lists), nameQuery
        (str), nameIsRegex (bool), minDurationNs (number), timeRangeOnly/
        rangeStartNs/rangeEndNs, activeOnly (bool, hides lanes with zero
        matching events). Missing/falsy keys mean "no constraint" on
        that dimension."""
        self._filters = dict(spec)
        self._recompute_event_mask()
        self._rebuild_rows()
        self.filtersChanged.emit()

    @Slot()
    def clearFilters(self) -> None:
        self._filters = {}
        self._event_mask = {}
        self._rebuild_rows()
        self.filtersChanged.emit()

    # ── Grouping / collapse ──────────────────────────────────────────
    @Property(str, notify=groupingChanged)
    def grouping(self) -> str:
        return self._grouping

    @Slot(str)
    def setGrouping(self, key: str) -> None:
        """`key` one of none/rank/process/runtime/device/stream/thread
        (anything else is treated as "none"). Grouping is a LANE-level
        concept (each lane keeps one value per dimension), unlike
        filtering's `buckets`, which is event-level -- a single thread
        lane can carry both Computation and Idle spans, so "group by
        activity bucket" isn't offered here the way it is as a filter."""
        self._grouping = key if key in (
            "rank", "process", "runtime", "device", "stream", "thread") else "none"
        self._rebuild_rows()
        self.groupingChanged.emit()

    @Slot(str, bool)
    def setGroupCollapsed(self, group_id: str, collapsed: bool) -> None:
        self._group_collapsed[group_id] = collapsed
        self._rebuild_rows()

    @Slot()
    def collapseAllGroups(self) -> None:
        for r in self._rows:
            if r["kind"] == "group":
                self._group_collapsed[r["groupId"]] = True
        self._rebuild_rows()

    @Slot()
    def expandAllGroups(self) -> None:
        for r in self._rows:
            if r["kind"] == "group":
                self._group_collapsed[r["groupId"]] = False
        self._rebuild_rows()

    @Slot('QVariantList', float, float, int, result='QVariantList')
    def groupCoverage(self, lane_indexes: list, view_start_ns: float, view_end_ns: float,
                       n_buckets: int) -> list[float]:
        """Fraction (0..1) of each of `n_buckets` equal-width time
        buckets across [view_start_ns, view_end_ns) covered by ANY span
        across every lane in `lane_indexes` -- a collapsed group row's
        aggregated activity strip. Same bucket-coverage TECHNIQUE
        dash.bucket_coverage() already uses for Overview's
        execution-timeline preview, reimplemented here in numpy: that
        Python version is fine to run once at Overview's startup, but
        this can run on every pan/zoom frame while a group stays
        collapsed, over potentially many lanes' spans at once. Boolean
        coverage (covered or not), not a duration-weighted density --
        matches what a collapsed row can honestly show at a glance."""
        if n_buckets <= 0 or view_end_ns <= view_start_ns:
            return []
        width = (view_end_ns - view_start_ns) / n_buckets
        covered = np.zeros(n_buckets, dtype=bool)
        for lane_idx in lane_indexes:
            if lane_idx < 0 or lane_idx >= len(self._lane_names):
                continue
            lane = self._lane_names[lane_idx]
            starts = self._starts.get(lane)
            ends = self._ends.get(lane)
            if starts is None or len(starts) == 0:
                continue
            max_dur = self._max_dur.get(lane, 0)
            lo = int(np.searchsorted(starts, view_start_ns - max_dur, side="left"))
            hi = int(np.searchsorted(starts, view_end_ns, side="right"))
            if lo >= hi:
                continue
            s = starts[lo:hi]
            e = ends[lo:hi]
            overlap = e > view_start_ns
            if not overlap.any():
                continue
            s = np.clip(s[overlap], view_start_ns, view_end_ns)
            e = np.clip(e[overlap], view_start_ns, view_end_ns)
            b0 = np.clip(np.floor((s - view_start_ns) / width).astype(np.int64), 0, n_buckets - 1)
            b1 = np.clip(np.ceil((e - view_start_ns) / width).astype(np.int64), 0, n_buckets)
            for a, b in zip(b0.tolist(), b1.tolist()):
                if b > a:
                    covered[a:b] = True
        return covered.astype(float).tolist()

    # ── Hide / isolate / reorder ─────────────────────────────────────
    @Property('QVariantList', notify=rowsChanged)
    def hiddenLanes(self) -> list[str]:
        return sorted(self._hidden_lanes)

    @Property('QVariantList', notify=rowsChanged)
    def isolatedLanes(self) -> list[str]:
        return sorted(self._isolated_lanes) if self._isolated_lanes is not None else []

    @Slot(str)
    def hideLane(self, lane_name: str) -> None:
        self._hidden_lanes.add(lane_name)
        self._rebuild_rows()

    @Slot(str)
    def showLane(self, lane_name: str) -> None:
        self._hidden_lanes.discard(lane_name)
        self._rebuild_rows()

    @Slot(str)
    def isolateLane(self, lane_name: str) -> None:
        """Solo semantics: REPLACES the isolated set with just this one
        lane (matching a per-row "isolate" button, the common case), not
        additive -- showAllLanes() is the only way back to "every lane",
        so a second isolateLane() call narrows to a different single
        lane rather than accumulating one."""
        self._isolated_lanes = {lane_name}
        self._rebuild_rows()

    @Slot()
    def showAllLanes(self) -> None:
        self._hidden_lanes = set()
        self._isolated_lanes = None
        self._rebuild_rows()

    @Slot(str, int)
    def moveRow(self, lane_name: str, new_index: int) -> None:
        """Moves `lane_name` to position `new_index` within the FULL lane
        order (not the currently-visible `rows` position, which can
        already be a filtered/grouped subset) -- establishes a custom
        `_row_order` the first time it's called, covering every lane
        (not just visible ones), so a later filter/isolate change can't
        silently lose the custom position of a lane that's temporarily
        hidden."""
        order = list(self._row_order) if self._row_order is not None else list(self._lane_names)
        if lane_name not in self._lane_index or lane_name not in order:
            return
        order.remove(lane_name)
        new_index = max(0, min(new_index, len(order)))
        order.insert(new_index, lane_name)
        self._row_order = order
        self._rebuild_rows()

    @Slot('QVariantList')
    def setRowOrder(self, lane_names: list) -> None:
        """Replaces the custom order wholesale. `lane_names` need not
        list every lane -- any this call omits keep their existing
        relative order, appended after the given ones, so a caller that
        only reorders a filtered/visible subset doesn't silently drop
        the rest of the trace's lanes from `rows`."""
        given = [ln for ln in lane_names if ln in self._lane_index]
        given_set = set(given)
        remaining = [ln for ln in self._lane_names if ln not in given_set]
        self._row_order = given + remaining
        self._rebuild_rows()

    @Slot(int, result=int)
    def rowIndexForLane(self, lane_index: int) -> int:
        """Current POSITION within `rows` of the lane row carrying this
        original `laneIndex` -- -1 if that lane isn't currently visible
        (hidden, filtered out, or inside a collapsed group). Connector-
        overlay/any other "row N is at pixel Y" calculation must use this
        instead of the raw laneIndex now that grouping/hide/isolate/
        reorder can make a row's visual position differ from its lane's
        original index."""
        for pos, r in enumerate(self._rows):
            if r["kind"] == "lane" and r["laneIndex"] == lane_index:
                return pos
        return -1

    # ── On-demand span queries ──────────────────────────────────────────
    @Slot(int, float, float, int, result='QVariantList')
    def visibleSpans(self, lane_index: int, view_start_ns: float, view_end_ns: float,
                     max_spans: int = 2000) -> list[dict[str, Any]]:
        """Spans in `lane_index` truly overlapping [view_start_ns,
        view_end_ns), via the same numpy searchsorted spatial-index trick
        TimelineWidget._density_row uses (src/ui/app.py) -- cheap even
        for a lane with tens of thousands of spans, since only the
        visible slice is ever materialized into Python dicts. Capped at
        `max_spans`: past that, the caller is zoomed out far enough that
        individual rectangles would be sub-pixel anyway (a coarse
        bucketed fallback, like the Overview preview's, is a possible
        future improvement, not implemented for this first pass).

        `lo` uses `max_dur` (the longest INDIVIDUAL span in this lane, not
        the lane's overall time range -- see its computation in __init__)
        as a look-back margin, then `mask` narrows the [lo:hi) candidate
        window down to spans that genuinely END after view_start_ns --
        without this, a span starting well before the window but NOT
        actually reaching into it (there can be many between `lo` and the
        first truly-visible span, once `lo` only needs to look back one
        span's worth of margin rather than the whole lane) would still be
        returned, and at high zoom/deep offsets could dominate decimation's
        `max_spans` budget with spans nowhere near the viewport."""
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
        ends = self._ends[lane]
        mask = ends[lo:hi] > view_start_ns
        event_mask = self._event_mask.get(lane)
        if event_mask is not None:
            mask = mask & event_mask[lo:hi]
        overlap_idx = (np.nonzero(mask)[0] + lo).tolist()
        if len(overlap_idx) > max_spans:
            step = len(overlap_idx) // max_spans + 1
            overlap_idx = overlap_idx[::step]
        slist = self._sorted_spans[lane]
        result = []
        for i in overlap_idx:
            s = slist[i]
            d = {
                "startNs": float(s.start_ns),
                "durNs": float(max(s.duration_ns, 1)),
                "name": s.name,
                "color": self._span_color(s),
                "spanIdx": i,
            }
            # "matched" only exists on this dict while a search is
            # active (see search()'s own docstring) -- not a permanent
            # per-span key, so a caller not using search pays nothing
            # extra per repaint frame.
            if self._search_active:
                d["matched"] = (lane_index, i) in self._search_match_set
            result.append(d)
        return result

    def _span_color(self, span) -> str:
        """The color visibleSpans() paints `span` with, per the active
        `_color_mode` -- bucket/category modes use dicts precomputed once
        in __init__ (see their construction there for why), so switching
        modes doesn't change visibleSpans()'s per-frame cost."""
        if self._color_mode == "bucket":
            return self._bucket_colors.get(activity_buckets.bucket_of_span(span), "#9ca3af")
        if self._color_mode == "category":
            return self._category_colors.get(span.category.value, "#9ca3af")
        return self._func_colors.get(span.name, "#9ca3af")

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
            # pid/tid: additive, for cross-tab navigation's
            # Nav.selectThread() (see src/gui/nav.py) -- SpanEvent carries
            # these as real dataclass fields, not tags entries, so they
            # weren't reachable from a click handler without this.
            "pid": s.pid,
            "tid": s.tid,
        }

    @Slot(str, str, int, result='QVariantList')
    def findByName(self, category: str, name: str, max_results: int = 50) -> list[dict[str, Any]]:
        """All spans matching (category, name) across every lane --
        "occurrences of this kernel/function" for cross-tab navigation
        (see src/gui/nav.py's docstring for why (category,name) is the
        correlation key rather than a true per-instance id). Backed by
        `_name_index` (built once in __init__): an O(1) dict lookup
        followed by a slice, not a scan over every span in the trace.
        Capped at `max_results` so a name appearing thousands of times (a
        worker-thread loop body, say) doesn't build a huge QML list.
        startNs is ABSOLUTE (matching visibleSpans()'s convention), NOT
        relative to viewStartNs the way spanAt()'s own startNs field is
        -- a real, deliberate asymmetry between these two methods (see
        spanAt()'s docstring); callers jumping the view to a match need
        an absolute timestamp to assign directly to viewStartNs."""
        occurrences = self._name_index.get((category, name), [])
        out = []
        for lane_idx, span_idx in occurrences[:max_results]:
            s = self._sorted_spans[self._lane_names[lane_idx]][span_idx]
            out.append({"laneIndex": lane_idx, "spanIdx": span_idx, "startNs": float(s.start_ns)})
        return out

    # ── Color mode (Phase B4) ────────────────────────────────────────
    @Property(str, notify=colorModeChanged)
    def colorMode(self) -> str:
        return self._color_mode

    @Slot(str)
    def setColorMode(self, mode: str) -> None:
        self._color_mode = mode if mode in ("function", "bucket", "category") else "function"
        self.colorModeChanged.emit()

    # ── Search (Phase B4) ────────────────────────────────────────────
    @Property(int, notify=searchChanged)
    def searchMatchCount(self) -> int:
        return len(self._search_matches)

    @Property(int, notify=searchChanged)
    def searchCursor(self) -> int:
        return self._search_cursor

    @Slot(str, bool, result=int)
    def search(self, query: str, is_regex: bool = False) -> int:
        """Matches `query` (substring, case-insensitive, unless
        `is_regex`) against every DISTINCT (category,name) key in
        `_name_index` -- not a scan of every span -- then flattens
        matched keys' occurrence lists into `_search_matches`, sorted by
        (laneIndex, spanIdx) for a stable, meaningful "next match" order
        (dict-iteration order is stable in Python but not meaningful to
        a user stepping through results). Returns the match count. An
        empty `query` clears the search (same effect as clearSearch())."""
        query = query or ""
        self._search_active = bool(query)
        if not query:
            self._search_matches = []
            self._search_match_set = set()
            self._search_cursor = -1
            self.searchChanged.emit()
            return 0

        regex = None
        if is_regex:
            try:
                regex = re.compile(query, re.IGNORECASE)
            except re.error:
                self._search_matches = []
                self._search_match_set = set()
                self._search_cursor = -1
                self.searchChanged.emit()
                return 0
        query_lower = query.lower()

        matches: list[tuple[int, int]] = []
        for (_category, name), occurrences in self._name_index.items():
            if regex is not None:
                if not regex.search(name):
                    continue
            elif query_lower not in name.lower():
                continue
            matches.extend(occurrences)
        matches.sort()

        self._search_matches = matches
        self._search_match_set = set(matches)
        self._search_cursor = 0 if matches else -1
        self.searchChanged.emit()
        return len(matches)

    @Slot()
    def clearSearch(self) -> None:
        self._search_active = False
        self._search_matches = []
        self._search_match_set = set()
        self._search_cursor = -1
        self.searchChanged.emit()

    def _match_result(self) -> dict[str, Any]:
        lane_idx, span_idx = self._search_matches[self._search_cursor]
        s = self._sorted_spans[self._lane_names[lane_idx]][span_idx]
        return {
            "laneIndex": lane_idx, "spanIdx": span_idx, "startNs": float(s.start_ns),
            "matchIndex": self._search_cursor, "matchCount": len(self._search_matches),
        }

    @Slot(result='QVariantMap')
    def nextMatch(self) -> dict[str, Any]:
        if not self._search_matches:
            return {}
        self._search_cursor = (self._search_cursor + 1) % len(self._search_matches)
        return self._match_result()

    @Slot(result='QVariantMap')
    def previousMatch(self) -> dict[str, Any]:
        if not self._search_matches:
            return {}
        self._search_cursor = (self._search_cursor - 1) % len(self._search_matches)
        return self._match_result()

    @Slot(float, float, result='QVariantMap')
    def callGraph(self, view_start_ns: float, view_end_ns: float) -> dict[str, Any]:
        """Node-and-edge call graph (analysis/call_graph.py) for whatever
        spans overlap [view_start_ns, view_end_ns) -- the Timeline's
        currently visible window, not the whole trace, so the graph
        reflects "what's on screen right now" and updates as the user
        pans/zooms (throttled QML-side, this rebuild -- an O(visible
        spans) pass twice over, once for the graph, once for layout --
        isn't cheap enough to run on every single pixel of drag).
        Silently returns an empty graph if no visible spans have
        captured stack frames (HPROFILER_CALLSTACK/--call-tree wasn't
        enabled for this run) -- same "no data" convention as the Call
        Tree tab, not an error."""
        visible = [
            s for s in self._trace.spans
            if s.duration_ns > 0 and s.start_ns < view_end_ns
            and s.start_ns + s.duration_ns > view_start_ns
        ]
        from ..analysis.call_graph import build_call_graph, layout_call_graph
        nodes, edges = build_call_graph(visible)
        # 60, not the module default of 30: the panel now scrolls (a
        # dynamically-sized Flickable canvas, sized off numLayers/
        # maxLayerSize) instead of squeezing every node into one fixed
        # 210px box, so a less aggressive cap no longer costs readability.
        layout = layout_call_graph(nodes, edges, max_nodes=60)
        for n in layout["nodes"]:
            n["color"] = self._theme.categoryColor(n["category"])
        return layout

    # ── Time ruler (Phase B5) ────────────────────────────────────────
    @Slot(float, float, int, result='QVariantList')
    def timeTicks(self, view_start_ns: float, view_end_ns: float, target_ticks: int = 8) -> list[dict[str, Any]]:
        """"Nice" round-number tick positions across [view_start_ns,
        view_end_ns) -- the classic 1/2/5-times-a-power-of-ten interval
        selection (same idea most plotting libraries use for axis
        ticks), so the ruler reads "0, 10ms, 20ms, ..." rather than
        whatever raw fraction `span/target_ticks` happens to produce.
        Each tick's `ns` is ABSOLUTE (matching visibleSpans()'s
        convention, for direct use as an x-position); `label` is
        formatted relative to the trace's own start (dash.fmt_ns()),
        matching the existing status line's "offset" convention."""
        if view_end_ns <= view_start_ns or target_ticks <= 0:
            return []
        raw_interval = (view_end_ns - view_start_ns) / target_ticks
        if raw_interval <= 0:
            return []
        magnitude = 10 ** math.floor(math.log10(raw_interval))
        normalized = raw_interval / magnitude
        if normalized < 1.5:
            nice = 1
        elif normalized < 3.5:
            nice = 2
        elif normalized < 7.5:
            nice = 5
        else:
            nice = 10
        interval = nice * magnitude
        if interval <= 0:
            return []
        ticks = []
        t = math.ceil(view_start_ns / interval) * interval
        # Defensive cap -- protects against an unexpected pathological
        # interval (e.g. floating-point edge cases) looping far more
        # times than the requested tick density.
        max_ticks = max(target_ticks * 3, 1)
        while t <= view_end_ns and len(ticks) < max_ticks:
            ticks.append({"ns": float(t), "label": dash.fmt_ns(max(t - self._view_start, 0))})
            t += interval
        return ticks

    # ── Bookmarks / named ranges (Phase B5) ──────────────────────────
    @Property('QVariantList', notify=bookmarksChanged)
    def bookmarks(self) -> list[dict[str, Any]]:
        return self._bookmarks

    @Property('QVariantList', notify=namedRangesChanged)
    def namedRanges(self) -> list[dict[str, Any]]:
        return self._named_ranges

    @Slot(float, str, result=int)
    def addBookmark(self, ns: float, name: str = "") -> int:
        bookmark_id = self._next_bookmark_id
        self._next_bookmark_id += 1
        label = name or dash.fmt_ns(max(ns - self._view_start, 0))
        self._bookmarks.append({"id": bookmark_id, "ns": float(ns), "name": label})
        self._bookmarks.sort(key=lambda b: b["ns"])
        self.bookmarksChanged.emit()
        return bookmark_id

    @Slot(int)
    def removeBookmark(self, bookmark_id: int) -> None:
        self._bookmarks = [b for b in self._bookmarks if b["id"] != bookmark_id]
        self.bookmarksChanged.emit()

    @Slot(float, float, str, result=int)
    def addNamedRange(self, start_ns: float, end_ns: float, name: str = "") -> int:
        if end_ns < start_ns:
            start_ns, end_ns = end_ns, start_ns
        range_id = self._next_range_id
        self._next_range_id += 1
        label = name or f"range {dash.fmt_ns(max(start_ns - self._view_start, 0))}"
        self._named_ranges.append({
            "id": range_id, "startNs": float(start_ns), "endNs": float(end_ns), "name": label,
        })
        self._named_ranges.sort(key=lambda r: r["startNs"])
        self.namedRangesChanged.emit()
        return range_id

    @Slot(int)
    def removeNamedRange(self, range_id: int) -> None:
        self._named_ranges = [r for r in self._named_ranges if r["id"] != range_id]
        self.namedRangesChanged.emit()
