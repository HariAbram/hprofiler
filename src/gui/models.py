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
from . import theme as theme_mod

# Same 16-hue palette concept as the TUI's _SPAN_PALETTE (src/ui/app.py),
# expressed as hex since QML wants CSS colors. Hue order mirrors the TUI's so
# a function tends to land in a similar slot on both UIs; exact index parity
# isn't guaranteed (both use crc32-modulo-then-probe over different sizes).
_SPAN_HEX_PALETTE = [
    "#f87171", "#22d3ee", "#4ade80", "#e879f9", "#fbbf24", "#60a5fa",
    "#fb923c", "#c084fc", "#f472b6", "#2dd4bf", "#a3e635", "#facc15",
    "#818cf8", "#fca5a5", "#34d399", "#f0abfc",
]



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
    # The light/dark theme was toggled and the theme-dependent colours
    # (lane labels and occupancy, bucket/category span colours,
    # connectors) were recomputed: lanes repaint.
    colorsChanged = Signal()

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
                "category": cat,
                "color": self._theme.categoryColor(cat),
                "count": self._infos[lane_name].count,
            })

        # ── Hierarchy: node / process (rank) / thread or device / stream ──
        # Per-lane identity for the hierarchical rows (timeline_hierarchy.py)
        # and for path-switch classification: the device of a stream lane is
        # the hook's hdev= (else the native dev=) of its first span; the node
        # of a merged multi-node trace is the `node` tag merge-nodes adds.
        from . import timeline_hierarchy as th
        merged = bool((trace.metadata.clock or {}).get("merge"))
        node_of_pid: dict[int, str | None] = {}
        self._lane_extra: list[dict[str, Any]] = []
        metas: list[Any] = []
        for i, lane_name in enumerate(self._lane_names):
            cat, kind, ident, pid_suffix = parsed[lane_name]
            info = self._infos[lane_name]
            pid = info.pid
            device = None
            if kind in ("stream", "device") or (merged and pid not in node_of_pid):
                first = next(iter(self._store.iter_spans(lane=lane_name)), None)
                if first is not None:
                    if kind in ("stream", "device"):
                        dv = first.tags.get("hdev", first.tags.get("dev"))
                        device = None if dv is None else str(dv)
                    if merged and pid not in node_of_pid:
                        node_of_pid[pid] = first.tags.get("node")
            tid = self._lane_tid.get(lane_name)
            k = _thread_key(lane_name)
            extra = {"pid": pid, "tid": tid, "rank": pid_rank.get(pid), "node": node_of_pid.get(pid),
                     "device": device, "kind": kind, "ident": ident, "category": cat,
                     "threadLabel": f"T{tid_seq.get(k, '?')} · tid {tid}" if k is not None else ""}
            self._lane_extra.append(extra)
            metas.append(th.LaneMeta(i, lane_name, cat, kind, ident, pid, info.count, tid=tid,
                                     rank=extra["rank"], node=extra["node"], device=device,
                                     thread_label=extra["threadLabel"]))
        host = trace.metadata.hostname
        self._tree = th.build_tree(metas, system_label=f"System · {host}" if host else "System")
        self._tree_nodes: dict[str, Any] = {}

        def _index(n) -> None:
            self._tree_nodes[n.id] = n
            for c in n.children:
                _index(c)
        _index(self._tree)
        # Each non-sync thread lane's explicit waits come from the sync lane
        # of the same (pid, tid): they are cut out of its spans (a parallel
        # region covers the barrier waits inside it).
        sync_of: dict[tuple[int, int], int] = {}
        for i, ex in enumerate(self._lane_extra):
            if ex["kind"] == "thread" and ex["category"] == "sync" and ex["tid"] is not None:
                sync_of[(ex["pid"], ex["tid"])] = i
        self._sync_pair: dict[int, int] = {}
        for i, ex in enumerate(self._lane_extra):
            if ex["kind"] == "thread" and ex["category"] != "sync" and ex["tid"] is not None:
                j = sync_of.get((ex["pid"], ex["tid"]))
                if j is not None:
                    self._sync_pair[i] = j

        # Rendering, selection, highlight (linked views), overlays, analysis,
        # view history -- see the sections near the end of this class.
        self._render_cache: dict[tuple, Any] = {}
        self._group_cache: dict[tuple, Any] = {}
        self._group_approx: dict[tuple, bool] = {}
        self._paint_rev = 0
        self._paint_errors = 0
        self._selected: tuple[int, int] = (-1, -1)          # (lane, eid)
        self._selected_edge = -1
        self._highlight: dict[str, Any] = {}
        self._overlay: dict[str, Any] = {"criticalPath": False, "focusPath": False, "families": [],
                                         "tiers": ["certain", "high", "medium", "low"],
                                         "classes": ["direct", "inferred"], "selectedOnly": True,
                                         "inferredWaits": False, "scoreTrack": False}
        self._analysis = None
        self._analysis_state = "idle"
        self._analysis_message = ""
        self._analysis_thread = None
        self._overlay_geometry: list[Any] = []
        self._history: list[tuple[float, float]] = []
        self._history_pos = -1
        self._minimap_cache: dict[tuple, Any] = {}
        self._row_pos: dict[int, int] = {}
        self._anchor_pos: dict[int, int] = {}
        self._visible_lanes: set[int] = set()
        self._comparison_hook = None
        self._status_of = None              # set by ComparisonBridge: span -> comparison status
        from .timeline_items import register_types
        register_types()

        # Filter state (applyFilters()/clearFilters()) -- empty dict means
        # "no active filter"; every lane's filteredCount is then its full
        # count.
        self._filters: dict[str, Any] = {}
        self._span_filt = None          # SpanFilter for the event-level filters

        # Grouping/hide/isolate/reorder state -- lane metadata only.
        # "hierarchy" (node / process / thread / device / stream) by default.
        self._grouping: str = "hierarchy"
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
        self._jit_color = theme_mod.activity_color("jit", self._theme.dark)
        self._category_colors: dict[str, str] = {
            c: self._theme.categoryColor(c) for c in {r["category"] for r in stats}
        }
        theme.themeChanged.connect(self._on_theme_changed)
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
        # (lane, spanIdx) -> the connectors that start or end at that span:
        # what the overlay draws for a hovered span (connectorsOf()).
        self._connectors_of: dict[tuple[int, int], list[dict[str, Any]]] = {}
        for c in self._connectors:
            self._connectors_of.setdefault((c["predLane"], c["predSpanIdx"]), []).append(c)
            self._connectors_of.setdefault((c["succLane"], c["succSpanIdx"]), []).append(c)

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
                "confidence": _cp.EDGE_CONFS[conf],
                "color": theme_mod.confidence_color(_cp.EDGE_CONFS[conf], self._theme.dark),
            })
        return out

    def _on_theme_changed(self) -> None:
        """Recompute every theme-dependent colour (computed once, they kept
        the starting theme's: e.g. the dark theme's near-white "sync" lane
        label on the light background)."""
        dark = self._theme.dark
        for m in self._lane_meta:
            m["color"] = theme_mod.category_color(m["category"], dark)
        self._bucket_colors = {b: theme_mod.bucket_color(b, dark) for b in self._bucket_colors}
        self._jit_color = theme_mod.activity_color("jit", dark)
        self._category_colors = {c: theme_mod.category_color(c, dark) for c in self._category_colors}
        for c in self._connectors:
            c["color"] = theme_mod.confidence_color(c["confidence"], dark)
        # rows carry the lane colour too: updated in place, without
        # rowsChanged, which would reset the lane list (and its scroll
        # position) -- QML colours the labels by category for the theme
        for r in self._rows:
            if r.get("kind") == "lane":
                r["color"] = self._lane_meta[r["laneIndex"]]["color"]
        self.colorsChanged.emit()
        self._repaint(data=True)

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

    @Slot(int, float, result='QVariantList')
    def connectorsOf(self, lane: int, span_idx: float) -> list[dict[str, Any]]:
        """The connectors that start or end at one span (`spanIdx`, a store
        event id: a float, as in spanAt(), since it passes 2^31). The hover
        overlay asks for these instead of reading `connectors`: QML re-read
        that whole list from Python on each element access, so one repaint
        over an OSU multi-pair bandwidth run's 44k connectors blocked the
        GUI for minutes."""
        if span_idx is None or span_idx < 0:
            return []
        return self._connectors_of.get((lane, int(span_idx)), [])

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
        if f.get("devices") and self._lane_extra[self._lane_index[lane_name]]["device"] not in \
                [str(d) for d in f["devices"]]:
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
            "hierLabel": m["label"],
            "level": 0,
            "category": m["category"],
            "color": m["color"],
            "count": m["count"],
            "filteredCount": filtered_count,
            "syncLane": self._sync_pair.get(i, -1),
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
        if grouping == "device":
            # device lanes / streams: the hook's hdev= (else the native
            # dev=) of the lane's first span; host lanes have none
            return self._lane_extra[self._lane_index[lane_name]]["device"]
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

        if self._grouping == "hierarchy":
            from . import timeline_hierarchy as th
            lane_rows: dict[int, dict[str, Any]] = {}
            for lane in lane_names:
                r = self._lane_row(lane)
                if r is not None:
                    lane_rows[r["laneIndex"]] = r
            focus = None
            if self._overlay.get("focusPath") and self._analysis is not None and self._analysis.ok:
                focus = self._analysis.path_lanes
            rows = th.flatten(self._tree, self._group_collapsed,
                              lane_row=lambda i: lane_rows.get(i) if focus is None or i in focus else None)
            if self._row_order is not None:
                # a custom order (moveRow/setRowOrder) is a flat concept:
                # keep the hierarchy's headers, order lanes inside each
                order = {n: k for k, n in enumerate(self._row_order)}
                rows = self._reorder_within_groups(rows, order)
            self._set_rows(rows)
            return

        if self._grouping == "none":
            rows = []
            for lane in lane_names:
                r = self._lane_row(lane)
                if r is not None:
                    rows.append(r)
            self._set_rows(rows)
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
                "nodeKind": "group",
                "level": 0,
                "label": g["label"],
                "laneIndexes": g["laneIndexes"],
                "laneCount": len(g["laneRows"]),
                "count": sum(lr["count"] for lr in g["laneRows"]),
                "filteredCount": sum(lr["filteredCount"] for lr in g["laneRows"]),
                "collapsed": collapsed,
            })
            if not collapsed:
                rows.extend(dict(lr, level=1) for lr in g["laneRows"])
        self._set_rows(rows)

    @staticmethod
    def _reorder_within_groups(rows: list[dict[str, Any]], order: dict[str, int]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        run: list[dict[str, Any]] = []
        for r in rows + [None]:
            if r is not None and r["kind"] == "lane" and (not run or r["level"] == run[0]["level"]):
                run.append(r)
                continue
            out.extend(sorted(run, key=lambda x: order.get(x["name"], 1 << 30)))
            run = []
            if r is None:
                break
            if r["kind"] == "lane":
                run.append(r)
            else:
                out.append(r)
        return out

    def _set_rows(self, rows: list[dict[str, Any]]) -> None:
        """The visible rows, plus where each lane is: its own row, or the
        collapsed header it is folded into (overlays anchor there)."""
        self._rows = rows
        self._row_pos = {}
        self._anchor_pos = {}
        visible: set[int] = set()
        for pos, r in enumerate(rows):
            if r["kind"] == "lane":
                self._row_pos[r["laneIndex"]] = pos
                self._anchor_pos.setdefault(r["laneIndex"], pos)
                visible.add(r["laneIndex"])
            elif r.get("collapsed"):
                for li in r["laneIndexes"]:
                    self._anchor_pos.setdefault(li, pos)
                    visible.add(li)
        self._visible_lanes = visible
        self._group_cache.clear()
        self.rowsChanged.emit()
        self._paint_rev += 1
        self.paintRevisionChanged.emit()

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
        devices = sorted({ex["device"] for ex in self._lane_extra if ex["device"] is not None})

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
            _dim("device", "Device", devices,
                 reason="no hdev=/dev= tag on device work in this trace (older hooks, host-only runs)"),
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
        self._repaint(data=True)

    @Slot()
    def clearFilters(self) -> None:
        self._filters = {}
        self._span_filt = None
        self._rebuild_rows()
        self.filtersChanged.emit()
        self._repaint(data=True)

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
            "hierarchy", "rank", "process", "runtime", "device", "stream", "thread") else "none"
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
        return self._row_pos.get(lane_index, -1)

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
        if self._color_mode == "comparison" and self._status_of is not None:
            return theme_mod.change_color(self._status_of(span), self._theme.dark)
        if self._color_mode == "bucket":
            # the Activity mode: buckets, with JIT compilation apart from
            # runtime overhead (timeline_lod.activity_of)
            if span.category.value == "jit":
                return self._jit_color
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

    @Property(bool, notify=colorModeChanged)
    def hasComparison(self) -> bool:
        return self._status_of is not None

    @Slot(str)
    def setColorMode(self, mode: str) -> None:
        modes = ("function", "bucket", "category") + (("comparison",) if self._status_of is not None else ())
        self._color_mode = mode if mode in modes else "function"
        self.colorModeChanged.emit()
        self._repaint(data=True)

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
        self._repaint()
        return len(matches)

    @Slot()
    def clearSearch(self) -> None:
        self._search_active = False
        self._search_matches = []
        self._search_match_set = set()
        self._search_cursor = -1
        self.searchChanged.emit()
        self._repaint()

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
    _FILTER_LISTS = ("ranks", "processes", "threads", "runtimes", "streams", "devices", "buckets")

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
            self._grouping = g if g in ("hierarchy", "rank", "process", "runtime", "device", "stream",
                                        "thread") else "none"
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
        self._repaint(data=True)

    # ══ Rendering (timeline_lod.py decides, timeline_paint.py draws) ══════
    paintRevisionChanged = Signal()
    eventSelectionChanged = Signal()
    highlightChanged = Signal()
    overlayChanged = Signal()
    analysisChanged = Signal()
    historyChanged = Signal()
    _analysisDone = Signal(object)

    @Property(int, notify=paintRevisionChanged)
    def paintRevision(self) -> int:
        """Bumped whenever something every painted item shows changed (data
        filters, colours, selection, highlight, overlays): the lane items
        bind their `revision` to it."""
        return self._paint_rev

    def _repaint(self, *, data: bool = False) -> None:
        if data:
            self._render_cache.clear()
            self._group_cache.clear()
            self._minimap_cache.clear()
        self._paint_rev += 1
        self.paintRevisionChanged.emit()

    def note_paint_error(self, exc: Exception) -> None:
        self._paint_errors += 1
        if self._paint_errors <= 3:
            import traceback
            traceback.print_exception(exc)

    def _style(self):
        from .timeline_paint import Style
        st = getattr(self, "_style_cache", None)
        if st is None or st.dark != self._theme.dark:
            st = self._style_cache = Style.for_theme(self._theme.dark)
        return st

    def _data_key(self) -> tuple:
        return (self._color_mode, self._theme.dark, id(self._span_filt), self._search_active)

    def render_lane(self, lane_index: int, a: float, b: float, px: int):
        """The level-of-detail render of one lane for a viewport (cached)."""
        key = (lane_index, float(a), float(b), int(px)) + self._data_key()
        lr = self._render_cache.get(key)
        if lr is None:
            lr = self._compute_render(lane_index, float(a), float(b), max(1, min(int(px), 8192)))
            if len(self._render_cache) > 600:
                self._render_cache.clear()
            self._render_cache[key] = lr
        return lr

    def _compute_render(self, lane_index: int, a: float, b: float, px: int):
        from . import timeline_lod as lod
        lane = self._lane_names[lane_index]
        extent = self._observed_extent(lane_index)
        A, B = int(math.floor(a)), int(math.ceil(b))
        sync_idx = self._sync_pair.get(lane_index, -1)
        sync_lane = self._lane_names[sync_idx] if sync_idx >= 0 else None
        est = None
        many = False
        if self._span_filt is None:
            est = self._store.estimate_starts(lane, a, b)
            many = est is not None and est > 4 * lod.EXACT_LIMIT
        if not many:
            many = self._store.count_window(lane, A, B, filt=self._span_filt, cap=lod.EXACT_LIMIT) > lod.EXACT_LIMIT
        if not many and sync_lane is not None:
            many = self._store.count_window(sync_lane, A, B, cap=lod.EXACT_LIMIT) > lod.EXACT_LIMIT
        lr = None
        if not many:
            waits = list(self._window_spans(sync_lane, A, B, None)[0]) if sync_lane is not None else None
            spans, act = self._window_spans(lane, A, B, self._span_filt)
            lr = lod.render_spans(
                spans, a, b, px, color_of=self._span_color, waits=waits,
                wait_color=theme_mod.category_color("sync", self._theme.dark), extent=extent, activity=act)
        if lr is None:
            occ = self._lane_occupancy(lane, a, b, px)
            wait_occ = self._lane_occupancy(sync_lane, a, b, px) if sync_lane is not None else None
            levels = self._store.activity(lane) if self._span_filt is None else None
            starts = lod.starts_per_column(levels[0], a, b, px) if levels else None
            bin_ns = 0.0
            if levels:
                usable = [lv["bin_ns"] for lv in levels if lv["bin_ns"] <= (b - a) / px]
                bin_ns = min(usable) if usable else 0.0
            lr = lod.render_occupancy(occ, starts, a, b, px, self._lane_meta[lane_index]["color"],
                                      wait_occupancy=wait_occ, index_bin_ns=bin_ns, extent=extent,
                                      events_in_view=int(est or 0))
        return lr

    _PREFETCH_LIMIT = 3          # x EXACT_LIMIT spans per prefetched lane window
    _PREFETCH_TOTAL = 400_000    # spans held over all lanes

    def _window_spans(self, lane: str, A: int, B: int, filt):
        """The spans of `lane` the store returns for window [A, B] (start <=
        B and end > A, start order), and a memoized activity_of for them.
        A miss reads a window three views wide when it holds few enough
        spans, so a pan or small zoom is served from memory instead of
        reading and decoding the same rows again."""
        import bisect
        from . import timeline_lod as lod
        cache = self.__dict__.setdefault("_span_window_cache", {})
        ent = cache.get(lane)
        if ent is None or ent["filt"] is not filt or not (ent["lo"] <= A and B <= ent["hi"]):
            w = max(B - A, 1)
            lo, hi = A - w, B + w
            cap = self._PREFETCH_LIMIT * lod.EXACT_LIMIT
            if self._store.count_window(lane, lo, hi, filt=filt, cap=cap) > cap:
                lo, hi = A, B
            spans = list(self._store.iter_spans(order="start", lane=lane, window=(lo, hi), filt=filt))
            ent = {"filt": filt, "lo": lo, "hi": hi, "spans": spans,
                   "starts": [s.start_ns for s in spans],
                   "max_dur": max((max(s.duration_ns, 0) for s in spans), default=0), "act": {}}
            cache.pop(lane, None)
            cache[lane] = ent
            total = sum(len(e["spans"]) for e in cache.values())
            while total > self._PREFETCH_TOTAL and len(cache) > 1:
                total -= len(cache.pop(next(iter(cache)))["spans"])
        spans, starts = ent["spans"], ent["starts"]
        i0 = bisect.bisect_left(starts, A - ent["max_dur"])
        i1 = bisect.bisect_right(starts, B)
        memo = ent["act"]

        def act(s, memo=memo):
            v = memo.get(id(s))
            if v is None:
                v = memo[id(s)] = lod.activity_of(s)
            return v
        return (s for s in spans[i0:i1] if s.start_ns + s.duration_ns > A), act

    def _observed_extent(self, lane_index: int) -> tuple[int, int]:
        """Where this lane's thread / stream has data: first to last event
        over all of its runtime lanes -- outside it is "no data" (the trace
        cannot say whether the thread existed); inside, a lane's empty time
        is a real gap (the thread was observed doing something else)."""
        c = getattr(self, "_extent_cache", None)
        if c is None:
            groups: dict[tuple, list[int]] = {}
            for i, ex in enumerate(self._lane_extra):
                if ex["kind"] == "thread" and ex["tid"] is not None:
                    key = ("t", ex["pid"], ex["tid"])
                elif ex["kind"] == "stream":
                    key = ("s", ex["pid"], ex["device"], ex["ident"])
                else:
                    key = ("l", i)
                groups.setdefault(key, []).append(i)
            c = {}
            for members in groups.values():
                infos = [self._infos[self._lane_names[i]] for i in members]
                ext = (min(x.min_start for x in infos), max(x.max_end for x in infos))
                for i in members:
                    c[i] = ext
            self._extent_cache = c
        return c[lane_index]

    def paint_lane_item(self, painter, lane_index: int, a: float, b: float, w: float, h: float) -> None:
        from .timeline_paint import paint_lane
        if lane_index < 0 or lane_index >= len(self._lane_names):
            return
        lr = self.render_lane(lane_index, a, b, int(round(w)))
        paint_lane(painter, lr, w, h, self._style(), self._lane_options(lane_index, lr))

    def _lane_options(self, lane_index: int, lr):
        from .timeline_paint import LaneOptions
        from . import timeline_lod as lod
        sel_lane, sel_eid = self._selected
        opts = LaneOptions(selected_eid=sel_eid if sel_lane == lane_index else -1)
        if self._search_active:
            ms = self._search_match_set
            opts.matched = lambda eid: (lane_index, eid) in ms
        hl = self._highlight
        if hl:
            cat = self._lane_extra[lane_index]["category"]
            lane_ok = (not hl.get("lanes") or lane_index in hl["lanes"]) and \
                      (not hl.get("categories") or cat in hl["categories"])
            acts = hl.get("activities")
            eids = hl.get("eids")
            names = hl.get("names")

            def dim(r, lane_ok=lane_ok, acts=acts, eids=eids, names=names) -> bool:
                if not lane_ok:
                    return True
                if acts and lod.ACTIVITY[r.activity] not in acts:
                    return True
                if eids and r.eid not in eids:
                    return True
                if names and r.name not in names:
                    return True
                return False
            opts.dim = dim
        an = self._analysis
        if an is not None and an.ok:
            if self._overlay.get("criticalPath"):
                opts.cp_steps = self._cp_steps_px(lane_index, lr)
            if self._overlay.get("inferredWaits"):
                opts.inferred_waits = self._waits_px(lane_index, lr)
        return opts

    def _per_lane_times(self, cache_name: str, index: dict, items: list, t_of) -> dict:
        c = getattr(self, cache_name, None)
        if c is None:
            c = {lane: np.asarray([t_of(items[k]) for k in ks], dtype=np.float64) for lane, ks in index.items()}
            setattr(self, cache_name, c)
        return c

    def _cp_arrays(self) -> dict:
        """The path steps as numpy columns (t0, t1, lane, mode code: 0 busy,
        1 waiting, 2 dependency hop), built once per analysis -- the
        per-frame step selection is then vectorized."""
        c = getattr(self, "_cp_arr_cache", None)
        if c is None:
            steps = self._analysis.steps
            c = {"t0": np.fromiter((s.t0 for s in steps), np.float64, len(steps)),
                 "t1": np.fromiter((s.t1 for s in steps), np.float64, len(steps)),
                 "lane": np.fromiter((s.lane for s in steps), np.int64, len(steps)),
                 "mode": np.fromiter((0 if s.mode == "busy" else 2 if s.mode == "dependency" else 1
                                      for s in steps), np.int8, len(steps)),
                 "by_lane": {lane: np.asarray(ks, dtype=np.int64)
                             for lane, ks in self._analysis.steps_of_lane.items()}}
            self._cp_arr_cache = c
        return c

    _CP_DENSE_FACTOR = 2         # path steps per pixel above which the band is per column

    @staticmethod
    def _cp_columns(x0a: np.ndarray, x1a: np.ndarray, mode: np.ndarray, width: int) -> list:
        """Zoomed out (many path steps per pixel): one band state per column
        -- busy or waiting, whichever holds more of the column's path time
        -- as runs of equal columns, instead of one rectangle per step."""
        from .timeline_lod import _coverage
        cov = []
        for code in (0, 1):
            sel = mode == code
            lo = x0a[sel]
            hi = np.maximum(x1a[sel], lo + 1e-6)          # a zero-length step still marks its column
            order = np.argsort(lo, kind="stable")
            cov.append(_coverage(lo[order], hi[order], width) if len(lo) else np.zeros(width))
        col = np.where((cov[0] > 0) & (cov[0] >= cov[1]), 0, np.where(cov[1] > 0, 1, -1))
        cut = np.flatnonzero(np.diff(col)) + 1
        starts = np.concatenate(([0], cut)).tolist()
        ends = np.concatenate((cut, [width])).tolist()
        return [(float(c0), float(c1), "busy" if col[c0] == 0 else "wait", "")
                for c0, c1 in zip(starts, ends) if col[c0] >= 0]

    def _cp_steps_px(self, lane_index: int, lr) -> list:
        an = self._analysis
        ks = an.steps_of_lane.get(lane_index)
        if not ks:
            return []
        t0s = self._per_lane_times("_cp_t0_cache", an.steps_of_lane, an.steps, lambda s: s.t0)[lane_index]
        a, b = lr.view_start, lr.view_end
        k = lr.scale
        lo = max(0, int(np.searchsorted(t0s, a, side="left")) - 1)
        hi = int(np.searchsorted(t0s, b, side="right"))
        arr = self._cp_arrays()
        idx = arr["by_lane"][lane_index][lo:hi]
        t0, t1, mode = arr["t0"][idx], arr["t1"][idx], arr["mode"][idx]
        keep = (t1 >= a) & (t0 <= b) & (mode != 2)
        idx, t0, t1, mode = idx[keep], t0[keep], t1[keep], mode[keep]
        x0a = np.maximum(0.0, (t0 - a) * k)
        x1a = np.minimum(float(lr.width), (t1 - a) * k)
        if len(idx) > self._CP_DENSE_FACTOR * lr.width:
            return self._cp_columns(x0a, x1a, mode, int(lr.width))
        x0s, x1s = x0a.tolist(), x1a.tolist()
        steps = an.steps
        out: list[list] = []
        for i, x0, x1, md in zip(idx.tolist(), x0s, x1s, mode.tolist()):
            m = "busy" if md == 0 else "wait"
            if out and out[-1][2] == m and x0 - out[-1][1] < 0.5:
                if x1 > out[-1][1]:
                    out[-1][1] = x1
            else:
                out.append([x0, x1, m, steps[i].tier])
        return [tuple(o) for o in out]

    def _waits_px(self, lane_index: int, lr) -> list:
        an = self._analysis
        ks = an.waits_of_lane.get(lane_index)
        if not ks:
            return []
        a, b, k = lr.view_start, lr.view_end, lr.scale
        out = []
        for idx in ks:
            w = an.waits[idx]
            if w.t1 <= a or w.t0 >= b:
                continue
            out.append((max(0.0, (w.t0 - a) * k), min(float(lr.width), (w.t1 - a) * k)))
            if len(out) > 4000:
                break
        return out

    # ── header rows: aggregate strips ────────────────────────────────────
    def paint_group_item(self, painter, group_id: str, lane_indexes: list, a: float, b: float,
                         w: float, h: float) -> None:
        from .timeline_paint import paint_group
        busy, wait, colors, cp = self.render_group(group_id, lane_indexes, a, b, int(round(w)))
        paint_group(painter, busy, wait, colors, w, h, self._style(), cp_cols=cp)

    def render_group(self, group_id: str, lane_indexes: list, a: float, b: float, px: int):
        """(busy, wait, colour per column, critical-path coverage per column)
        of a header row. Lanes of one thread or stream combine by MAX (they
        nest: a barrier wait inside a parallel region), and a process,
        device, node or the system by the MEAN of their members -- the
        share of threads (streams) busy / waiting in that column. Busy
        excludes the explicit waits of the thread's sync lane."""
        px = max(1, min(int(px), 8192))
        key = (group_id, tuple(lane_indexes), float(a), float(b), px) + self._data_key()
        hit = self._group_cache.get(key)
        if hit is not None:
            return hit
        palette: list[str] = []
        pal_idx: dict[str, int] = {}

        def code_of(color: str) -> int:
            i = pal_idx.get(color)
            if i is None:
                i = pal_idx[color] = len(palette)
                palette.append(color)
            return i

        visible = set(lane_indexes)

        from .timeline_lod import column_fractions
        approx = [False]

        def lane_vals(li: int):
            ex = self._lane_extra[li]
            if ex["category"] in self._NOT_WORK_CATEGORIES:
                return None          # annotation ranges are labels, not activity
            # the lane's own (cached) render: the same activity classes the
            # lane row shows -- an MPI_Wait is waiting here too
            lr = self.render_lane(li, a, b, px)
            busy, wait = column_fractions(lr)
            if lr.source == "index":
                approx[0] = True
                if ex["category"] == "sync":
                    busy, wait = np.zeros(px), np.clip(busy + wait, 0.0, 1.0)
            code = np.full(px, code_of(self._lane_meta[li]["color"]), dtype=np.int32)
            return busy, wait, code

        def combine(kids, how: str):
            kids = [k for k in kids if k is not None]
            if not kids:
                return np.zeros(px), np.zeros(px), np.zeros(px, dtype=np.int32)
            B = np.stack([k[0] for k in kids])
            W = np.stack([k[1] for k in kids])
            C = np.stack([k[2] for k in kids])
            arg = np.argmax(B, axis=0)
            code = C[arg, np.arange(px)]
            if how == "max":
                wait = W.max(axis=0)
                busy = np.clip(B.max(axis=0) - wait, 0.0, 1.0)
            else:
                busy, wait = B.mean(axis=0), W.mean(axis=0)
            return busy, np.minimum(wait, 1.0 - busy), code

        def agg(n):
            if n.lane >= 0:
                return lane_vals(n.lane) if n.lane in visible else None
            kids = [(c, agg(c)) for c in n.children]
            kids = [(c, v) for c, v in kids if v is not None]
            if not kids:
                return None
            if n.kind in ("thread", "stream"):
                return combine([v for _c, v in kids], "max")
            if n.kind == "process":
                host = [v for c, v in kids if c.kind == "thread" or (c.lane >= 0 and c.kind == "lane")]
                return combine(host or [v for _c, v in kids], "mean")
            return combine([v for _c, v in kids], "mean")

        node = self._tree_nodes.get(group_id)
        if node is None:            # a flat grouping (by rank, runtime, ...): mean of its lanes
            vals = [lane_vals(li) for li in lane_indexes if 0 <= li < len(self._lane_names)]
            res = combine(vals, "mean")
        else:
            res = agg(node) or (np.zeros(px), np.zeros(px), np.zeros(px, dtype=np.int32))
        busy, wait, code = res
        colors = [palette[c] if busy[i] > 0 and palette else None for i, c in enumerate(code.tolist())]
        cp = None
        an = self._analysis
        if an is not None and an.ok and self._overlay.get("criticalPath"):
            cp = self._cp_coverage(set(lane_indexes), a, b, px)
        out = (busy, wait, colors, cp)
        self._group_approx[key] = approx[0]
        if len(self._group_cache) > 300:
            self._group_cache.clear()
        self._group_cache[key] = out
        return out

    def _cp_coverage(self, lanes: set | None, a: float, b: float, px: int) -> np.ndarray:
        from .timeline_lod import _coverage
        arr = self._cp_arrays()
        k = px / max(b - a, 1e-9)
        keep = (arr["mode"] != 2) & (arr["t1"] > a) & (arr["t0"] < b)
        if lanes is not None:
            keep &= np.isin(arr["lane"], np.fromiter(lanes, np.int64, len(lanes)))
        if not keep.any():
            return np.zeros(px)
        lo = np.maximum(0.0, (arr["t0"][keep] - a) * k)
        hi = np.minimum(float(px), (arr["t1"][keep] - a) * k)
        order = np.argsort(lo, kind="stable")
        return _coverage(lo[order], hi[order], px)

    # ── hover / hit testing ──────────────────────────────────────────────
    @Slot(int, float, float, float, float, result='QVariantMap')
    def hitTest(self, lane_index: int, x: float, view_start_ns: float, view_end_ns: float,
                width: float) -> dict[str, Any]:
        """What is under pixel x of a lane: an exact span ({"kind": "span",
        spanIdx, name, ...}) or an aggregated column ({"kind": "column",
        count, busy, wait, ...: never an invented event}), else {}."""
        if lane_index < 0 or lane_index >= len(self._lane_names):
            return {}
        lr = self.render_lane(lane_index, view_start_ns, view_end_ns, int(round(width)))
        r = lr.hit(x)
        if r is not None and r.eid >= 0:
            return {"kind": "span", "spanIdx": float(r.eid), "name": r.name, "startNs": float(r.start_ns),
                    "durNs": float(r.end_ns - r.start_ns), "activity": r.activity}
        if r is not None:
            return {"kind": "wait", "startNs": float(r.start_ns), "durNs": float(r.end_ns - r.start_ns),
                    "name": "explicit wait (sync lane)"}
        info = lr.column_info(x)
        if info is not None:
            info["kind"] = "column"
            return info
        return {}

    # ── selection (one event) and its details ────────────────────────────
    @Property('QVariantMap', notify=eventSelectionChanged)
    def selectedEvent(self) -> dict[str, Any]:
        lane, eid = self._selected
        return {"laneIndex": lane, "spanIdx": float(eid), "edge": self._selected_edge}

    @Slot(int, float)
    def selectEvent(self, lane_index: int, span_idx: float) -> None:
        sel = (int(lane_index), int(span_idx)) if span_idx is not None and span_idx >= 0 else (-1, -1)
        if sel == self._selected and self._selected_edge < 0:
            return
        self._selected = sel
        self._selected_edge = -1
        self.eventSelectionChanged.emit()
        self._repaint()

    @Slot()
    def clearEventSelection(self) -> None:
        self._selected = (-1, -1)
        self._selected_edge = -1
        self.eventSelectionChanged.emit()
        self._repaint()

    @Slot(int, float, result='QVariantMap')
    def eventDetails(self, lane_index: int, span_idx: float) -> dict[str, Any]:
        from .timeline_details import event_details
        return event_details(self, int(lane_index), int(span_idx))

    @Slot(int, result='QVariantMap')
    def edgeDetails(self, edge_id: int) -> dict[str, Any]:
        from .timeline_details import edge_details
        return edge_details(self, int(edge_id))

    @Slot(float, float, result=int)
    def edgeAt(self, x: float, y: float) -> int:
        """The overlay edge within 5 px of (x, y) in the overlay's own
        coordinates (as last painted), else -1."""
        best, best_d = -1, 5.0
        for e in self._overlay_geometry:
            dx, dy = e.x1 - e.x0, e.y1 - e.y0
            n2 = dx * dx + dy * dy
            t = 0.0 if n2 <= 0 else max(0.0, min(1.0, ((x - e.x0) * dx + (y - e.y0) * dy) / n2))
            px, py = e.x0 + t * dx, e.y0 + t * dy
            d = ((x - px) ** 2 + (y - py) ** 2) ** 0.5
            if d < best_d:
                best, best_d = e.id, d
        return best

    @Slot(int)
    def selectEdge(self, edge_id: int) -> None:
        self._selected_edge = int(edge_id)
        self.eventSelectionChanged.emit()
        self._repaint()

    # ── highlight (linked summary views) ─────────────────────────────────
    @Property('QVariantMap', notify=highlightChanged)
    def highlight(self) -> dict[str, Any]:
        h = self._highlight
        return {k: (sorted(v) if isinstance(v, (set, frozenset)) else v) for k, v in h.items()}

    @Slot('QVariantMap')
    def setHighlight(self, spec: dict) -> None:
        """Dim everything except events matching ALL given keys: activities
        (timeline_lod.ACTIVITY names), categories, lanes, eids, names; plus
        an optional `label` shown in the screen's highlight chip."""
        clean: dict[str, Any] = {}
        for k in ("activities", "categories", "names"):
            if spec.get(k):
                clean[k] = frozenset(str(x) for x in spec[k])
        for k in ("lanes", "eids"):
            if spec.get(k):
                clean[k] = frozenset(int(x) for x in spec[k])
        if spec.get("label"):
            clean["label"] = str(spec["label"])
        self._highlight = clean
        self.highlightChanged.emit()
        self._repaint()

    @Slot()
    def clearHighlight(self) -> None:
        self._highlight = {}
        self.highlightChanged.emit()
        self._repaint()

    # ── overlays: dependencies, critical path ────────────────────────────
    @Property('QVariantMap', notify=overlayChanged)
    def overlay(self) -> dict[str, Any]:
        return dict(self._overlay)

    @Slot('QVariantMap')
    def setOverlay(self, spec: dict) -> None:
        before_focus = self._overlay.get("focusPath")
        for k in ("criticalPath", "focusPath", "inferredWaits", "scoreTrack"):
            if k in spec:
                self._overlay[k] = bool(spec[k])
        for k in ("families", "tiers", "classes"):
            if k in spec and isinstance(spec[k], (list, tuple)):
                self._overlay[k] = [str(x) for x in spec[k]]
        if (self._overlay.get("criticalPath") or self._overlay.get("families")
                or self._overlay.get("inferredWaits")) and self._analysis is None:
            self.startAnalysis()
        self.overlayChanged.emit()
        if self._overlay.get("focusPath") != before_focus:
            self._rebuild_rows()
        self._repaint(data=True)

    @Property('QVariantList', constant=True)
    def edgeFamilies(self) -> list[dict[str, Any]]:
        from .timeline_analysis import FAMILIES, FAMILY_LABELS
        return [{"key": f, "label": FAMILY_LABELS[f]} for f in FAMILIES]

    def _visible_edge_ids(self, a: float, b: float, hover_eid: int) -> list[int]:
        an = self._analysis
        ov = self._overlay
        fams = set(ov.get("families") or ())
        if ov.get("criticalPath"):
            fams.add("critical")
        tiers, classes = set(ov.get("tiers") or ()), set(ov.get("classes") or ())
        focus = {e for e in (self._selected[1], int(hover_eid)) if e >= 0}
        out = []
        for k in an.edges_in(a, b):
            e = an.edges[k]
            if e.src_eid in focus or e.dst_eid in focus or k == self._selected_edge:
                out.append(k)
                continue
            if not (e.family in fams or ("critical" in fams and e.on_path)):
                continue
            if e.on_path and "critical" in fams:
                out.append(k)
                continue
            if e.tier in tiers and e.cls in classes:
                out.append(k)
        if len(out) > 4000:
            out.sort(key=lambda k: (not an.edges[k].on_path, ("certain", "high", "medium", "low").index(an.edges[k].tier)))
            out = out[:4000]
        self._edges_capped = len(out) >= 4000
        return out

    def paint_overlay_item(self, painter, w: float, h: float, a: float, b: float, scroll_y: float,
                           row_h: float, hover_lane: int, hover_eid: float) -> None:
        from .timeline_paint import OverlayEdge, paint_edges, paint_transition_markers
        k = w / max(b - a, 1e-9)
        geo: list[OverlayEdge] = []

        def y_of(lane: int):
            pos = self._anchor_pos.get(lane)
            if pos is None:
                return None
            return pos * row_h + row_h / 2 - scroll_y

        an = self._analysis
        if an is None or not an.ok:
            # before the analysis: the hovered span's MPI/NCCL partners
            # from the persisted communication edges (as before)
            if hover_lane >= 0 and hover_eid is not None and hover_eid >= 0:
                for c in self._connectors_of.get((hover_lane, int(hover_eid)), []):
                    y0, y1 = y_of(c["predLane"]), y_of(c["succLane"])
                    if y0 is None or y1 is None:
                        continue
                    tier = c["confidence"]
                    geo.append(OverlayEdge(-1, (c["predMidNs"] - a) * k, y0, (c["succMidNs"] - a) * k, y1,
                                           "mpi_p2p", tier, "inferred" if tier in ("medium", "low") else "direct",
                                           False))
        else:
            for kk in self._visible_edge_ids(a, b, int(hover_eid if hover_eid is not None else -1)):
                e = an.edges[kk]
                y0, y1 = y_of(e.src_lane), y_of(e.dst_lane)
                if y0 is None or y1 is None:
                    continue
                if (y0 < -row_h and y1 < -row_h) or (y0 > h + row_h and y1 > h + row_h):
                    continue
                geo.append(OverlayEdge(kk, (e.t_src - a) * k, y0, (e.t_dst - a) * k, y1, e.family, e.tier,
                                       e.cls, e.on_path,
                                       (e.src_lane in self._row_pos, e.dst_lane in self._row_pos)))
        self._overlay_geometry = geo
        paint_edges(painter, geo, self._style(), selected=self._selected_edge,
                    path_mode=bool(self._overlay.get("criticalPath")))
        if an is not None and an.ok and self._overlay.get("criticalPath"):
            # one chevron per 4 px of a row (they are 10 px wide): N / Shift+N
            # and the CP panel still step through every switch
            import bisect
            tc = getattr(self, "_transition_t", None)
            if tc is None or len(tc[1]) != len(an.transitions):
                srt = sorted(an.transitions, key=lambda t: t.t)
                tc = self._transition_t = ([t.t for t in srt], srt)
            ts, srt = tc
            pts, seen = [], set()
            for t in srt[bisect.bisect_left(ts, a):bisect.bisect_right(ts, b)]:
                y = y_of(t.to_lane)
                if y is None or not -row_h < y < h + row_h:
                    continue
                x = (t.t - a) * k
                cell = (int(x // 4), y)
                if cell not in seen:
                    seen.add(cell)
                    pts.append((x, y - row_h / 2 + 9))
            paint_transition_markers(painter, pts, self._style())

    @Property(bool, notify=paintRevisionChanged)
    def edgesCapped(self) -> bool:
        return bool(getattr(self, "_edges_capped", False))

    # ── critical-path navigation and summary ─────────────────────────────
    @Slot(float, int, result='QVariantMap')
    def nextTransition(self, ns: float, direction: int) -> dict[str, Any]:
        """The next (direction > 0) or previous path switch after / before
        `ns`: where the path moves to another rank, thread, runtime,
        device or stream."""
        an = self._analysis
        if an is None or not an.ok or not an.transitions:
            return {}
        ts = [t.t for t in an.transitions]
        import bisect
        if direction > 0:
            i = bisect.bisect_right(ts, ns + 0.5)
            if i >= len(ts):
                return {}
        else:
            i = bisect.bisect_left(ts, ns - 0.5) - 1
            if i < 0:
                return {}
        t = an.transitions[i]
        return {"ns": float(t.t), "fromLane": t.from_lane, "toLane": t.to_lane,
                "fromLabel": self._lane_display(t.from_lane), "toLabel": self._lane_display(t.to_lane),
                "switch": " + ".join(t.switch), "kind": t.kind, "tier": t.tier, "index": i,
                "count": len(ts)}

    def _lane_display(self, lane: int) -> str:
        if lane < 0 or lane >= len(self._lane_meta):
            return "?"
        ex = self._lane_extra[lane]
        where = f"rank {ex['rank']}" if ex["rank"] is not None else f"pid {ex['pid']}"
        if ex["kind"] == "thread":
            where += f" {ex['threadLabel'].split(' · ')[0]}"
        elif ex["kind"] == "stream":
            where += f" stream {ex['ident']}" + (f" (device {ex['device']})" if ex["device"] else "")
        return f"{ex['category']} · {where}"

    @Property('QVariantMap', notify=analysisChanged)
    def criticalSummary(self) -> dict[str, Any]:
        an = self._analysis
        if an is None or not an.ok:
            return {}
        return an.summary

    @Property(str, notify=analysisChanged)
    def analysisState(self) -> str:
        return self._analysis_state

    @Property(str, notify=analysisChanged)
    def analysisMessage(self) -> str:
        return self._analysis_message

    # Lanes that are not activity: annotation ranges (NVTX / ROCTX).
    _NOT_WORK_CATEGORIES = frozenset({"nvtx"})

    # Above this many spans the analysis starts only when asked for.
    AUTO_ANALYSIS_SPANS = 300_000

    @Slot()
    def maybeStartAnalysis(self) -> None:
        if self._analysis is None and self._analysis_state == "idle" and \
                self._store.span_count() <= self.AUTO_ANALYSIS_SPANS:
            self.startAnalysis()

    @Slot()
    def startAnalysis(self) -> None:
        """Builds the overlays (timeline_analysis.build) on a background
        thread; the GUI stays interactive and repaints when they are in."""
        if self._analysis_state == "running":
            return
        import threading
        from . import timeline_analysis as ta
        self._analysis_state = "running"
        self._analysis_message = "computing dependencies and the critical path…"
        self.analysisChanged.emit()
        if not getattr(self, "_analysis_connected", False):
            self._analysisDone.connect(self._on_analysis_done)
            self._analysis_connected = True

        def meta(lane: int) -> dict:
            return self._lane_extra[lane] if 0 <= lane < len(self._lane_extra) else {}

        def run() -> None:
            try:
                data = ta.build(self._trace, self._lane_of_span, meta,
                                progress=lambda s: setattr(self, "_analysis_message", s + "…"))
            except Exception as exc:          # reported in the GUI, never raised into Qt
                data = ta.AnalysisData(ok=False, error=f"{type(exc).__name__}: {exc}")
            self._analysisDone.emit(data)

        self._analysis_thread = threading.Thread(target=run, name="hprofiler-timeline-analysis", daemon=True)
        self._analysis_thread.start()

    def _on_analysis_done(self, data) -> None:
        self._analysis = data
        for name in ("_cp_t0_cache", "_cp_arr_cache", "_transition_t"):
            if hasattr(self, name):
                delattr(self, name)
        if data.ok:
            self._analysis_state = "ready"
            self._analysis_message = f"ready ({data.seconds:.1f} s)"
        else:
            self._analysis_state = "failed"
            self._analysis_message = data.error or "analysis failed"
        self.analysisChanged.emit()
        if self._overlay.get("focusPath"):
            self._rebuild_rows()
        self._repaint(data=True)

    def wait_for_analysis(self, timeout: float = 120.0) -> bool:
        """Tests and export: block until the background analysis finished
        (processing the queued completion)."""
        import time as _t
        from PySide6.QtCore import QCoreApplication
        t_end = _t.time() + timeout
        while self._analysis_state == "running" and _t.time() < t_end:
            QCoreApplication.processEvents()
            if self._analysis_thread is not None:
                self._analysis_thread.join(0.02)
        QCoreApplication.processEvents()
        return self._analysis_state == "ready"

    # ── minimap and score track ──────────────────────────────────────────
    def minimap_columns(self, n: int):
        key = (n,) + self._data_key() + (bool(self._analysis and self._analysis.ok),
                                          bool(self._overlay.get("criticalPath")))
        hit = self._minimap_cache.get(key)
        if hit is not None:
            return hit
        t0, t1 = float(self._view_start), float(self._view_end)
        busy_parts, wait_parts = [], []
        for i, ln in enumerate(self._lane_names):
            ex = self._lane_extra[i]
            if ex["kind"] not in ("thread", "") or ex["category"] in self._NOT_WORK_CATEGORIES:
                continue
            occ = self._lane_occupancy(ln, t0, t1, n)
            (wait_parts if ex["category"] == "sync" else busy_parts).append(occ)
        busy = np.max(busy_parts, axis=0) if busy_parts else np.zeros(n)
        wait = np.max(wait_parts, axis=0) if wait_parts else np.zeros(n)
        busy = np.clip(busy - wait, 0.0, 1.0)
        cp = None
        an = self._analysis
        if an is not None and an.ok and self._overlay.get("criticalPath"):
            cp = self._cp_coverage(None, t0, t1, n)
        out = (busy, np.minimum(wait, 1.0 - busy), cp)
        self._minimap_cache[key] = out
        return out

    def paint_minimap_item(self, painter, w: float, h: float, a: float, b: float) -> None:
        from .timeline_paint import paint_minimap
        n = max(1, min(int(w), 4096))
        busy, wait, cp = self.minimap_columns(n)
        paint_minimap(painter, busy, wait, cp, float(self._view_start), float(self._view_end), a, b, w, h,
                      self._style())

    def paint_score_item(self, painter, w: float, h: float, a: float, b: float) -> None:
        from .timeline_paint import paint_score_track
        an = self._analysis
        if an is None or not an.ok:
            return
        paint_score_track(painter, an.legacy, a, b, w, h, self._style())

    @Slot(float, float, result='QVariantMap')
    def scoreOverlapAt(self, ns: float, tolerance_ns: float) -> dict[str, Any]:
        """Hover on the score track: the counted intervals at `ns` and how
        many times the span-level search counts that instant."""
        an = self._analysis
        if an is None or not an.ok or not an.legacy.get("available"):
            return {}
        hits = [iv for iv in an.legacy["intervals"] if iv[0] - tolerance_ns <= ns <= iv[1] + tolerance_ns]
        return {"count": len(hits), "names": [iv[2] for iv in hits[:6]],
                "kinds": [iv[3] for iv in hits[:6]]}

    # ── view history, jumps ──────────────────────────────────────────────
    @Slot(float, float)
    def pushView(self, start_ns: float, end_ns: float) -> None:
        """Records a view (after a discrete navigation step) for Back /
        Forward; consecutive near-identical views collapse into one."""
        v = (float(start_ns), float(end_ns))
        if self._history_pos >= 0:
            cs, ce = self._history[self._history_pos]
            if abs(cs - v[0]) <= 1e-6 * max(1.0, ce - cs) and abs(ce - v[1]) <= 1e-6 * max(1.0, ce - cs):
                return
        del self._history[self._history_pos + 1:]
        self._history.append(v)
        if len(self._history) > 200:
            self._history.pop(0)
        self._history_pos = len(self._history) - 1
        self.historyChanged.emit()

    @Property(bool, notify=historyChanged)
    def canGoBack(self) -> bool:
        return self._history_pos > 0

    @Property(bool, notify=historyChanged)
    def canGoForward(self) -> bool:
        return 0 <= self._history_pos < len(self._history) - 1

    @Slot(result='QVariantMap')
    def historyBack(self) -> dict[str, float]:
        if self._history_pos <= 0:
            return {}
        self._history_pos -= 1
        self.historyChanged.emit()
        s, e = self._history[self._history_pos]
        return {"startNs": s, "endNs": e}

    @Slot(result='QVariantMap')
    def historyForward(self) -> dict[str, float]:
        if self._history_pos >= len(self._history) - 1:
            return {}
        self._history_pos += 1
        self.historyChanged.emit()
        s, e = self._history[self._history_pos]
        return {"startNs": s, "endNs": e}

    @Slot(str, result='QVariantMap')
    def resolveJump(self, text: str) -> dict[str, Any]:
        from .timeline_details import resolve_jump
        return resolve_jump(self, text)

    @Slot(float, result=str)
    def formatNs(self, ns: float) -> str:
        return dash.fmt_ns(ns)

    # ── coordinated summary views (timeline_details.py) ──────────────────
    @Slot(result='QVariantMap')
    def summaries(self) -> dict[str, Any]:
        from .timeline_details import summaries
        return summaries(self)

    @Slot(int, result='QVariantMap')
    def heatmap(self, n_bins: int) -> dict[str, Any]:
        from .timeline_details import heatmap
        return heatmap(self, n_bins)

    # ── figure export ────────────────────────────────────────────────────
    @Slot(str, float, float, int, result=str)
    def exportFigure(self, path: str, view_start_ns: float, view_end_ns: float, width: int) -> str:
        """Writes the visible rows of [start, end) as a PNG or SVG figure
        (by extension) with time axis, row labels and a legend of the
        encodings drawn. Returns "" on success, else the error."""
        from .timeline_details import export_figure
        try:
            export_figure(self, path, view_start_ns, view_end_ns, int(width))
            return ""
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"
