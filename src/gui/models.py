"""
QObject models backing the Timeline screen. Unlike bridge.py's
DashboardBridge (small, fixed-size data exposed as plain properties), the
Timeline is span-count-sensitive -- a real GROMACS trace easily has tens of
thousands of spans -- so lane/span data is fetched on demand via @Slot
methods the QML Canvas calls with the current viewport (start_ns, end_ns,
pixel width). Only what is visible is computed.
"""
from __future__ import annotations

import math
import re
import zlib
from typing import Any

import numpy as np
from PySide6.QtCore import QObject, Property, Signal, Slot

from ..core.trace import Trace, parse_lane_name
from ..analysis import criticalpath as _cp
from ..analysis import activity_buckets
from ..analysis import dashboard as dash

# Same 16-hue palette concept as the TUI's _SPAN_PALETTE (src/ui/app.py),
# expressed as hex since QML wants CSS colors. Hue order mirrors the TUI's so
# a function tends to land in a similar slot on both UIs; exact index parity
# isn't guaranteed (both use crc32-modulo-then-probe over different sizes).
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

    `rows` (live, notify=rowsChanged) is the visual row list -- group
    headers + filtered/ordered/hidden-aware lane references -- that QML
    iterates instead of `lanes` directly. A "lane" row always carries its
    ORIGINAL `laneIndex`, so `lanes`/`visibleSpans`/`spanAt`/`findByName`/
    `callGraph` keep physical-lane-index addressing, which cross-tab
    navigation (Nav/Inspector, double-click-to-zoom) relies on."""

    rowsChanged = Signal()
    filtersChanged = Signal()
    groupingChanged = Signal()
    colorModeChanged = Signal()
    searchChanged = Signal()
    bookmarksChanged = Signal()
    namedRangesChanged = Signal()
    viewNoted = Signal()

    def __init__(self, trace: Trace, theme, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._trace = trace
        self._theme = theme
        self._store = trace.store
        # Lane metadata only (name, pid, count, extent, longest span) -- the
        # spans themselves stay in the store and are fetched per visible
        # window (visibleSpans/laneView).
        self._infos = {ln.name: ln for ln in self._store.lane_infos()}
        self._lanes = self._infos
        parsed = {ln: parse_lane_name(ln) for ln in self._lanes}

        # T1, T2, ... numbering per (pid, tid) -- the same tid from two
        # processes (merge-nodes) is two different threads.
        def _thread_key(ln: str) -> tuple[int, int] | None:
            cat, kind, ident, pid = parsed[ln]
            if kind != "thread":
                return None
            try:
                return (pid if pid is not None else -1, int(ident))
            except ValueError:
                return None
        all_tids = sorted({k for ln in self._lanes if (k := _thread_key(ln)) is not None})
        tid_seq = {k: i + 1 for i, k in enumerate(all_tids)}

        # rank for any lane whose process made MPI calls -- labels
        # pid-disambiguated GPU stream lanes ("cuda S0 r2") too.
        pid_rank: dict[int, str] = {}
        lane_rank: dict[str, str] = {}
        for lane_name, info in self._infos.items():
            if not lane_name.startswith("mpi/"):
                continue
            for s in self._store.iter_spans(lane=lane_name):
                rank = s.tags.get("rank")
                if rank is not None:
                    lane_rank[lane_name] = rank
                    pid_rank.setdefault(info.pid, rank)
                    break
        self._lane_rank = lane_rank

        # Per-lane process/thread/stream identity for filtering
        # (applyFilters()), derived once here rather than per filter
        # application. pid is the first-seen process of the lane's spans
        # (same convention as lane_rank above); lane keys partition by
        # thread/stream id, so a lane is not expected to mix pids.
        self._lane_pid: dict[str, int] = {}
        self._lane_tid: dict[str, int | None] = {}
        self._lane_stream: dict[str, str | None] = {}
        for lane_name, info in self._infos.items():
            self._lane_pid[lane_name] = info.pid
            _cat, kind, ident, _pid = parsed[lane_name]
            tid = None
            stream = None
            if kind == "thread":
                try:
                    tid = int(ident)
                except ValueError:
                    tid = None
            elif kind == "stream":
                stream = ident
            self._lane_tid[lane_name] = tid
            self._lane_stream[lane_name] = stream

        def _sort_key(name: str) -> tuple[int, int, str]:
            cat, kind, ident, pid = parsed[name]
            if kind == "thread":
                k = _thread_key(name)
                if k is not None:
                    return (tid_seq.get(k, 9999), 0, cat)
            elif kind == "stream":
                try:
                    return (10000 + int(ident), pid or 0, cat)
                except ValueError:
                    pass
            elif kind == "device":
                return (20000, pid or 0, cat)
            return (0, pid or 0, cat)

        self._lane_names: list[str] = sorted(self._lanes.keys(), key=_sort_key)
        self._lane_index: dict[str, int] = {name: i for i, name in enumerate(self._lane_names)}

        self._lane_meta: list[dict[str, Any]] = []
        for lane_name in self._lane_names:
            cat, kind, ident, pid = parsed[lane_name]
            if lane_name in lane_rank:
                label = f"{cat} rank{lane_rank[lane_name]}"
            elif kind == "thread":
                k = _thread_key(lane_name)
                label = f"{cat} T{tid_seq.get(k, '?')}" if k is not None else cat
            elif kind == "stream":
                label = f"{cat} S{ident}"
            elif kind == "device":
                label = f"{cat} device"
            else:
                label = cat
            if pid is not None and lane_name not in lane_rank:
                label += f" r{pid_rank[pid]}" if pid in pid_rank else f" p{pid}"
            self._lane_meta.append({
                "name": lane_name,
                "label": label,
                "color": self._theme.categoryColor(cat),
                "count": self._infos[lane_name].count,
            })

        # Filter state (applyFilters()/clearFilters()) -- empty dict means
        # "no active filter"; every lane's filteredCount is then its full
        # count.
        self._filters: dict[str, Any] = {}
        self._span_filt = None          # SpanFilter for the event-level filters

        # Grouping/hide/isolate/reorder state -- lane metadata only.
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

        stats = self._store.aggregate_stats()          # store-side, one row per (category, name)
        self._func_colors = _assign_span_colors([r["name"] for r in stats])
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
        self._category_colors: dict[str, str] = {
            c: self._theme.categoryColor(c) for c in {r["category"] for r in stats}
        }
        # (pid, lane base) -> lane index, to place a span found by a name
        # query (findByName/search) on its lane.
        from ..core.store.common import lane_base
        self._lane_base = lane_base
        self._lane_of_key = {(info.pid, (info.cat, info.kind, info.ident)): i
                             for i, info in ((self._lane_index[n], self._infos[n]) for n in self._lane_names)}

        # Search state.
        self._search_active: bool = False
        self._search_matches: list[tuple[int, int]] = []
        self._search_match_set: set[tuple[int, int]] = set()
        self._search_cursor: int = -1
        self._search_starts: dict[tuple[int, int], int] = {}

        # Bookmarks / named ranges -- Timeline-specific view state, kept here
        # rather than on Nav (cross-tab selection), like filter/grouping.
        self._bookmarks: list[dict[str, Any]] = []
        self._named_ranges: list[dict[str, Any]] = []
        self._next_bookmark_id: int = 1
        self._next_range_id: int = 1
        # zoom / pan as last reported by the QML view (noteView), and the
        # saved one to apply once when the view appears (restoredView)
        self._noted_view: dict[str, float] = {}
        self._restored_view: dict[str, float] = {}

        ext = self._store.span_extent(timed_only=True)
        if ext is not None:
            self._view_start, self._view_end = ext
        else:
            self._view_start = trace.metadata.start_time_ns
            self._view_end = self._view_start + 1
        self._trace_dur = max(self._view_end - self._view_start, 1)

        # Cross-rank MPI/NCCL connectors -- same source data as the TUI's
        # TimelineWidget (criticalpath.py's resolved dependency graph),
        # different rendering (a real Canvas line here, not Braille dots).
        self._connectors: list[dict[str, Any]] = []
        try:
            self._connectors = self._build_connectors()
        except Exception:
            self._connectors = []

    # Building the dependency graph for connectors costs a pass over every
    # span; above this size it is only done when the store already holds
    # persisted edges (e.g. after `hprofiler critical-path`).
    CONNECTOR_SPAN_LIMIT = 500_000

    def _build_connectors(self) -> list[dict[str, Any]]:
        if not ({"mpi", "nccl"} & {info.cat for info in self._infos.values()}):
            return []
        graph = _cp.load_comm_graph(self._trace, max_build_spans=self.CONNECTOR_SPAN_LIMIT)
        if graph is None or not len(graph.edges):
            return []
        edges = graph.edges
        comm = graph.endpoints(set(edges["dst"].tolist()) | set(edges["src"].tolist()),
                               categories=("mpi", "nccl"))
        out = []
        for succ_idx, pred_idx, kind, conf in edges.tolist():
            succ, pred = comm.get(succ_idx), comm.get(pred_idx)
            if succ is None or pred is None:
                continue
            succ_lane = self._lane_of_span(succ)
            pred_lane = self._lane_of_span(pred)
            if succ_lane is None or pred_lane is None or pred_lane == succ_lane:
                continue
            out.append({
                "predLane": pred_lane,
                "predSpanIdx": pred.eid,
                "predMidNs": (pred.start_ns + pred.end_ns) / 2.0,
                "succLane": succ_lane,
                "succSpanIdx": succ.eid,
                "succMidNs": (succ.start_ns + succ.end_ns) / 2.0,
                "color": _CONFIDENCE_HEX.get(_cp.EDGE_CONFS[conf], "#8b949e"),
            })
        return out

    def _lane_of_span(self, span) -> int | None:
        return self._lane_of_key.get((span.pid, self._lane_base(span)))

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
        `_span_filt`, since individual spans within a passing lane can
        still differ on those."""
        f = self._filters
        if f.get("ranks") and self._lane_rank.get(lane_name) not in f["ranks"]:
            return False
        if f.get("processes") and self._lane_pid.get(lane_name) not in f["processes"]:
            return False
        if f.get("threads") and self._lane_tid.get(lane_name) not in f["threads"]:
            return False
        if f.get("runtimes") and parse_lane_name(lane_name)[0] not in f["runtimes"]:
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
        filtered_count = (self._store.lane_count(lane, self._span_filt)
                          if self._span_filt is not None else m["count"])
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
            return parse_lane_name(lane_name)[0]
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
        runtimes = sorted({parse_lane_name(name)[0] for name in self._lane_names})
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
        """The active EVENT-level filters (name / duration / time range /
        bucket) as one store SpanFilter, applied store-side by every window
        and count query; None when there are none."""
        from ..core.store import SpanFilter
        f = self._filters
        name_query = f.get("nameQuery") or ""
        is_regex = bool(f.get("nameIsRegex"))
        min_dur = float(f.get("minDurationNs") or 0)
        buckets = frozenset(f.get("buckets") or [])
        time_range_only = bool(f.get("timeRangeOnly"))
        filt = SpanFilter(
            name_query=name_query, name_is_regex=is_regex, min_dur_ns=int(min_dur),
            buckets=buckets,
            range_start_ns=int(float(f.get("rangeStartNs") or 0)) if time_range_only else None,
            range_end_ns=int(float(f.get("rangeEndNs") or 0)) if time_range_only else None,
        )
        self._span_filt = None if filt.is_empty() else filt

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
        self._span_filt = None
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
        """Fraction-free coverage (1.0 = some span of some lane in
        `lane_indexes` is active in that bucket) of `n_buckets` equal
        buckets across [view_start_ns, view_end_ns) -- a collapsed group
        row's activity strip, from each lane's occupancy bins
        (_lane_occupancy) rather than from individual spans."""
        if n_buckets <= 0 or view_end_ns <= view_start_ns:
            return []
        covered = np.zeros(n_buckets, dtype=bool)
        for lane_idx in lane_indexes:
            if lane_idx < 0 or lane_idx >= len(self._lane_names):
                continue
            occ = self._lane_occupancy(self._lane_names[lane_idx], view_start_ns, view_end_ns, n_buckets)
            covered |= occ > 0
        return covered.astype(float).tolist()

    def _lane_occupancy(self, lane: str, view_start_ns: float, view_end_ns: float, n: int) -> np.ndarray:
        """Occupancy per bin: the precomputed multiresolution index when no
        event filter is active and it resolves this zoom, else binned from
        the window's spans (streamed, never all materialized)."""
        if self._span_filt is None:
            occ = self._store.occupancy(lane, view_start_ns, view_end_ns, n)
            if occ is not None:
                return occ
        return self._store.occupancy_exact(lane, view_start_ns, view_end_ns, n, self._span_filt)

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
        (hidden, filtered out, or inside a collapsed group). Any "row N is
        at pixel Y" calculation (e.g. the connector overlay) must use this
        rather than the raw laneIndex, since grouping/hide/isolate/reorder
        make a row's visual position differ from its lane's index."""
        for pos, r in enumerate(self._rows):
            if r["kind"] == "lane" and r["laneIndex"] == lane_index:
                return pos
        return -1

    # ── On-demand span queries ──────────────────────────────────────────
    @Slot(int, float, float, int, result='QVariantList')
    def visibleSpans(self, lane_index: int, view_start_ns: float, view_end_ns: float,
                     max_spans: int = 2000) -> list[dict[str, Any]]:
        """Spans in `lane_index` truly overlapping [view_start_ns,
        view_end_ns), from the store's per-lane time index (only the visible
        window is read). Past `max_spans` the result is decimated (every
        k-th span) -- the Timeline itself calls laneView(), which switches
        to occupancy bins instead; this method stays for callers that want
        individual spans regardless. `spanIdx` is the span's store event
        id (spanAt() takes it back)."""
        if lane_index < 0 or lane_index >= len(self._lane_names):
            return []
        lane = self._lane_names[lane_index]
        a, b = int(view_start_ns), int(view_end_ns)
        n = self._store.count_window(lane, a, b, filt=self._span_filt)
        step = n // max_spans + 1 if n > max_spans else 1
        out = []
        for i, s in enumerate(self._store.iter_spans(order="start", lane=lane, window=(a, b),
                                                     filt=self._span_filt)):
            if i % step == 0:
                out.append(self._span_dict(lane_index, s))
        return out

    def _span_dict(self, lane_index: int, s) -> dict[str, Any]:
        d = {
            "startNs": float(s.start_ns),
            "durNs": float(max(s.duration_ns, 1)),
            "name": s.name,
            "color": self._span_color(s),
            "spanIdx": s.eid,
        }
        # "matched" only exists on this dict while a search is
        # active (see search()'s own docstring) -- not a permanent
        # per-span key, so a caller not using search pays nothing
        # extra per repaint frame.
        if self._search_active:
            d["matched"] = (lane_index, s.eid) in self._search_match_set
        return d

    @Slot(int, float, float, int, int, result='QVariantMap')
    def laneView(self, lane_index: int, view_start_ns: float, view_end_ns: float,
                 pixel_width: int, max_spans: int = 2000) -> dict[str, Any]:
        """What the Timeline paints for one lane: {"mode": "spans", "spans":
        [...]} when the window holds at most `max_spans` spans (fetched
        exactly), else {"mode": "bins", "bins": [occupancy 0..1 per pixel
        column], "color": lane color} from the activity index built at
        finalization -- zoomed-out views never create one object per
        event."""
        if lane_index < 0 or lane_index >= len(self._lane_names) or view_end_ns <= view_start_ns:
            return {"mode": "spans", "spans": []}
        lane = self._lane_names[lane_index]
        a, b = int(view_start_ns), int(view_end_ns)
        n_px = max(1, min(int(pixel_width), 4096))
        many = False
        if self._span_filt is None:
            est = self._store.estimate_starts(lane, a, b)
            many = est is not None and est > 4 * max_spans
        if not many:
            many = self._store.count_window(lane, a, b, filt=self._span_filt, cap=max_spans) > max_spans
        if not many:
            return {"mode": "spans", "spans": [self._span_dict(lane_index, s) for s in self._store.iter_spans(
                order="start", lane=lane, window=(a, b), filt=self._span_filt)]}
        occ = self._lane_occupancy(lane, view_start_ns, view_end_ns, n_px)
        return {"mode": "bins", "bins": [round(float(v), 3) for v in occ],
                "color": self._lane_meta[lane_index]["color"]}

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

    @Slot(int, float, result='QVariantMap')
    def spanAt(self, lane_index: int, span_idx: float) -> dict[str, Any]:
        """Full detail for one span (hover tooltip) by the `spanIdx` (store
        event id) visibleSpans()/laneView() returned."""
        if lane_index < 0 or lane_index >= len(self._lane_names):
            return {}
        s = self._store.event_by_id(int(span_idx)) if span_idx is not None and span_idx >= 0 else None
        if s is None or not hasattr(s, "duration_ns"):
            return {}
        return {
            "name": s.name,
            "category": s.category.value,
            "startNs": float(s.start_ns - self._view_start),
            "durNs": float(s.duration_ns),
            "tags": dict(s.tags),
            # pid/tid for Nav.selectThread() (src/gui/nav.py) -- SpanEvent
            # carries these as dataclass fields, not tags.
            "pid": s.pid,
            "tid": s.tid,
        }

    @Slot(str, str, int, result='QVariantList')
    def findByName(self, category: str, name: str, max_results: int = 50) -> list[dict[str, Any]]:
        """Spans matching (category, name) across every lane --
        "occurrences of this kernel/function" for cross-tab navigation
        (see src/gui/nav.py's docstring for why (category,name) is the
        correlation key rather than a true per-instance id). A store query
        (name/category filter), stopping after `max_results`, in lane
        order then time order. startNs is ABSOLUTE (matching
        visibleSpans()'s convention), NOT relative to viewStartNs the way
        spanAt()'s own startNs field is."""
        out = self._occurrences(frozenset({name}), category, max_results)
        return [{"laneIndex": li, "spanIdx": eid, "startNs": float(start)} for li, eid, start in out]

    def _occurrences(self, names: frozenset, category: str | None, limit: int) -> list[tuple[int, int, int]]:
        from ..core.store import SpanFilter
        filt = SpanFilter(names=names, categories=frozenset({category}) if category else frozenset())
        out: list[tuple[int, int, int]] = []
        for lane_idx, lane in enumerate(self._lane_names):
            info = self._infos[lane]
            if category and info.cat != category:
                continue
            for s in self._store.iter_spans(order="start", lane=lane, filt=filt):
                out.append((lane_idx, s.eid, s.start_ns))
                if len(out) >= limit:
                    return out
        return out

    # ── Color mode ───────────────────────────────────────────────────
    @Property(str, notify=colorModeChanged)
    def colorMode(self) -> str:
        return self._color_mode

    @Slot(str)
    def setColorMode(self, mode: str) -> None:
        self._color_mode = mode if mode in ("function", "bucket", "category") else "function"
        self.colorModeChanged.emit()

    # ── Search ───────────────────────────────────────────────────────
    @Property(int, notify=searchChanged)
    def searchMatchCount(self) -> int:
        return len(self._search_matches)

    @Property(int, notify=searchChanged)
    def searchCursor(self) -> int:
        return self._search_cursor

    # Matches beyond this are not collected (the count says so).
    SEARCH_LIMIT = 10_000

    @Slot(str, bool, result=int)
    def search(self, query: str, is_regex: bool = False) -> int:
        """Matches `query` (substring, case-insensitive, unless
        `is_regex`) against every DISTINCT name in the trace (the store's
        name dictionary -- not a scan of every span), then collects the
        matching spans lane by lane in time order, at most SEARCH_LIMIT,
        into `_search_matches` ((laneIndex, spanIdx) pairs, sorted). An
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
        names = frozenset(r["name"] for r in self._store.aggregate_stats()
                          if (regex.search(r["name"]) if regex is not None
                              else query_lower in r["name"].lower()))
        found = self._occurrences(names, None, self.SEARCH_LIMIT) if names else []
        self._search_starts = {(li, eid): start for li, eid, start in found}
        matches = sorted((li, eid) for li, eid, _ in found)

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
        start = self._search_starts.get((lane_idx, span_idx), 0)
        return {
            "laneIndex": lane_idx, "spanIdx": span_idx, "startNs": float(start),
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
            s for s in self._trace.iter_spans(window=(int(view_start_ns), int(view_end_ns)), has_stack=True)
            if s.duration_ns > 0 and s.start_ns < view_end_ns
            and s.start_ns + s.duration_ns > view_start_ns
        ]
        from ..analysis.call_graph import build_call_graph, layout_call_graph
        nodes, edges = build_call_graph(visible)
        # 60, not the module default of 30: a scrolling consumer can show
        # more nodes without losing readability.
        layout = layout_call_graph(nodes, edges, max_nodes=60)
        for n in layout["nodes"]:
            n["color"] = self._theme.categoryColor(n["category"])
        return layout

    # ── Time ruler ───────────────────────────────────────────────────
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

    # ── Bookmarks / named ranges ─────────────────────────────────────
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

    # ── Persisted view state (settings.ViewStatePersister) ───────────
    # Presentation state only: what is filtered, grouped, hidden and where
    # the view is -- never span data. Restoring tolerates stale entries
    # (lanes that no longer exist are dropped, malformed values ignored).
    _FILTER_LISTS = ("ranks", "processes", "threads", "runtimes", "streams", "buckets")

    @Slot(float, float)
    def noteView(self, zoom: float, view_start_ns: float) -> None:
        self._noted_view = {"zoom": float(zoom), "startOffsetNs": float(view_start_ns) - self._view_start}
        self.viewNoted.emit()

    @Property('QVariantMap', constant=True)
    def restoredView(self) -> dict[str, float]:
        """{"zoom", "viewStartNs"} saved for this profile ({} if none) --
        TimelineScreen applies it once when it is created."""
        v = self._restored_view
        if not v:
            return {}
        return {"zoom": v["zoom"], "viewStartNs": self._view_start + v["startOffsetNs"]}

    def export_view_state(self) -> dict[str, Any]:
        return {
            "filters": dict(self._filters),
            "grouping": self._grouping,
            "groupCollapsed": dict(self._group_collapsed),
            "colorMode": self._color_mode,
            "hiddenLanes": sorted(self._hidden_lanes),
            "isolatedLanes": sorted(self._isolated_lanes) if self._isolated_lanes is not None else None,
            "rowOrder": list(self._row_order) if self._row_order is not None else None,
            "bookmarks": [dict(b) for b in self._bookmarks],
            "namedRanges": [dict(r) for r in self._named_ranges],
            "view": dict(self._noted_view or self._restored_view),
        }

    def restore_view_state(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            return
        lanes = set(self._lane_names)

        def _lane_list(v) -> list[str]:
            return [x for x in v if isinstance(x, str) and x in lanes] if isinstance(v, list) else []

        f = state.get("filters")
        if isinstance(f, dict):
            clean: dict[str, Any] = {}
            for k in self._FILTER_LISTS:
                if isinstance(f.get(k), list):
                    clean[k] = [x for x in f[k] if isinstance(x, (str, int, float))]
            if isinstance(f.get("nameQuery"), str):
                clean["nameQuery"] = f["nameQuery"]
            for k in ("nameIsRegex", "timeRangeOnly", "activeOnly"):
                if isinstance(f.get(k), bool):
                    clean[k] = f[k]
            for k in ("minDurationNs", "rangeStartNs", "rangeEndNs"):
                if isinstance(f.get(k), (int, float)) and f[k] >= 0:
                    clean[k] = f[k]
            if clean.get("nameIsRegex") and clean.get("nameQuery"):
                import re
                try:
                    re.compile(clean["nameQuery"])
                except re.error:
                    clean.pop("nameIsRegex")
            self._filters = clean
            self._recompute_event_mask()
        if isinstance(state.get("grouping"), str):
            g = state["grouping"]
            self._grouping = g if g in ("rank", "process", "runtime", "device", "stream", "thread") else "none"
        if isinstance(state.get("groupCollapsed"), dict):
            self._group_collapsed = {str(k): bool(v) for k, v in state["groupCollapsed"].items()}
        if state.get("colorMode") in ("function", "category", "bucket"):
            self._color_mode = state["colorMode"]
        self._hidden_lanes = set(_lane_list(state.get("hiddenLanes")))
        iso = _lane_list(state.get("isolatedLanes"))
        self._isolated_lanes = set(iso) if iso else None
        order = _lane_list(state.get("rowOrder"))
        in_order = set(order)
        self._row_order = (order + [ln for ln in self._lane_names if ln not in in_order]) if order else None
        lo, hi = self._view_start, self._view_end
        bms = []
        for b in state.get("bookmarks") or []:
            if isinstance(b, dict) and isinstance(b.get("ns"), (int, float)) and lo <= b["ns"] <= hi:
                bms.append({"id": len(bms) + 1, "ns": float(b["ns"]), "name": str(b.get("name", ""))[:200]})
        self._bookmarks = sorted(bms, key=lambda b: b["ns"])
        self._next_bookmark_id = len(bms) + 1
        rngs = []
        for r in state.get("namedRanges") or []:
            if (isinstance(r, dict) and isinstance(r.get("startNs"), (int, float))
                    and isinstance(r.get("endNs"), (int, float)) and lo <= r["startNs"] <= r["endNs"] <= hi):
                rngs.append({"id": len(rngs) + 1, "startNs": float(r["startNs"]), "endNs": float(r["endNs"]),
                             "name": str(r.get("name", ""))[:200]})
        self._named_ranges = sorted(rngs, key=lambda r: r["startNs"])
        self._next_range_id = len(rngs) + 1
        v = state.get("view")
        self._restored_view = {}
        if isinstance(v, dict) and isinstance(v.get("zoom"), (int, float)) \
                and isinstance(v.get("startOffsetNs"), (int, float)):
            zoom = min(max(float(v["zoom"]), 1.0), 1e9)
            off = min(max(float(v["startOffsetNs"]), 0.0), float(self._trace_dur))
            self._restored_view = {"zoom": zoom, "startOffsetNs": off}
        self._rebuild_rows()
        self.filtersChanged.emit()
        self.groupingChanged.emit()
        self.colorModeChanged.emit()
        self.bookmarksChanged.emit()
        self.namedRangesChanged.emit()
