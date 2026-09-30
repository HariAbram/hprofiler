"""
QObject bridges exposing Trace/analysis data to QML. Each bridge computes
once at construction (a loaded trace never changes afterwards, same
assumption the TUI's widgets already make) and exposes `constant=True`
Properties -- no change notification machinery needed for data that's
fixed for the lifetime of the window. SourceBridge is the one exception
(see its own docstring): background disassembly collection can still be
running when the window opens, so its kernel list needs real change
notification, polled the same way the TUI's DisasmWidget does.

Reuses analysis/dashboard.py (shared with the TUI's Overview tab) and the
existing analysis modules directly -- this file's job is turning that
data into QML-consumable shapes, not computing anything new. Kernels/
Call Tree data is a small QVariantList here (aggregated by function name/
call-tree node, not per-span -- even a trace with tens of thousands of
spans typically aggregates to a few hundred distinct rows at most, well
within what a plain list + QML-side sort/filter handles fine); see
models.py for the Timeline's per-span data instead, which genuinely does
need on-demand/viewport-culled fetching.
"""
from __future__ import annotations

import csv
from typing import Any

from PySide6.QtCore import QObject, Property, Signal, Slot, QTimer

from ..core.trace import Trace
from ..analysis import dashboard as dash
from ..analysis import activity_buckets
from ..disasm.classifier import InsnType
from . import columns
from . import theme as theme_mod
from .tablemodel import TableBundle


_DASHBOARD_TIMELINE_BUCKETS = 60


def compute_dashboard_data(trace: Trace, dark: bool) -> dict[str, Any]:
    """The expensive, Qt-free part of DashboardBridge's construction --
    several full passes over `trace.spans` plus a handful of analysis-
    module calls -- as a module-level function the async-loading worker
    thread can call directly (see compute_call_tree_data()'s docstring
    for why: no QObject touched off its owning thread). Returns a flat
    dict whose keys match DashboardBridge's own `_foo` attribute names
    (minus the leading underscore) -- __init__ applies it via a single
    mechanical `setattr` loop instead of hand-duplicating each field
    twice, which would be its own source of transcription bugs.

    Does NOT build `top_bottlenecks`' TableBundle (a QObject) -- that
    stays in DashboardBridge.__init__, built from this dict's
    `top_bottlenecks` list, since TableBundle itself must be constructed
    on the thread that will own it, never on the worker thread."""
    meta = trace.metadata
    wall_ns = dash.trace_wall_ns(trace)
    result: dict[str, Any] = {}
    # One exclusive-time pass shared by diagnosis, findings, wait % and
    # the time breakdown (each used to redo it: 2.8x slower at 1M spans).
    et = activity_buckets.ExclusiveTime(trace.spans)
    findings = dash.top_findings(trace, et)

    diag_label, diag_severity = dash.diagnose(trace, et)
    result["diagnosis_label"] = diag_label
    result["diagnosis_severity"] = diag_severity
    result["wall_time"] = dash.fmt_ns(wall_ns)

    # Kept as the full dict (not just gpu_active_pct) -- the time-
    # breakdown/investigate-next logic below reuses launch_gap_pct/
    # sync_stall_pct from the SAME gpu_starvation() call rather than
    # invoking it a second time.
    gpu_stats: dict[str, float] | None = None
    if any(b in dash._GPU_CATS for b in (meta.backends_used or [])):
        try:
            from ..analysis.cct import gpu_starvation
            gpu_stats = gpu_starvation(trace)
        except Exception:
            gpu_stats = None
    result["gpu_active_available"] = gpu_stats is not None
    result["gpu_active_pct"] = gpu_stats["gpu_active_pct"] if gpu_stats else 0.0

    result["wait_label"], result["wait_pct"] = dash.wait_fraction(trace, et)

    ctrs = {c.name: c.value for c in trace.counters}
    rss = ctrs.get("process_max_rss_bytes", 0.0)
    if rss > 0:
        result["peak_memory"] = dash.fmt_bytes(rss)
    else:
        vram_total = sum(v for k, v in ctrs.items() if k.startswith("gpu_mem_used_bytes"))
        result["peak_memory"] = dash.fmt_bytes(vram_total) if vram_total > 0 else "n/a"

    result["findings"] = [
        {"icon": icon, "color": theme_mod.severity_color(severity, dark),
         "title": title, "metric": metric}
        for icon, severity, title, metric in findings
    ]

    stats = trace.aggregated_stats()
    total_all = sum(r["total_ns"] for r in stats) or 1
    result["hot_kernels"] = [
        {
            "name": dash.fmt_kernel_name(r["name"])[:40],
            "category": r["category"],
            "color": theme_mod.category_color(r["category"], dark),
            "calls": r["count"],
            "total": dash.fmt_ns(r["total_ns"]),
            "share": f"{r['total_ns']/total_all*100:.1f}%",
        }
        for r in stats[:8]
    ]

    by_cat: dict[str, list] = {}
    for s in trace.spans:
        if s.duration_ns > 0:
            by_cat.setdefault(s.category.value, []).append(s)
    top_cats = sorted(by_cat.items(), key=lambda kv: -sum(s.duration_ns for s in kv[1]))[:3]
    timed = [s for s in trace.spans if s.duration_ns > 0]
    view_start = min((s.start_ns for s in timed), default=0)
    view_end = max((s.end_ns for s in timed), default=1)
    view_dur = max(view_end - view_start, 1)
    result["timeline_preview"] = [
        {
            "category": cat,
            "color": theme_mod.category_color(cat, dark),
            "coverage": dash.bucket_coverage(cspans, _DASHBOARD_TIMELINE_BUCKETS, view_start, view_dur),
        }
        for cat, cspans in top_cats
    ]

    ctx = dash.find_source_context(trace)
    if ctx is not None:
        result["has_source"] = True
        result["source_display_name"] = ctx.display_name
        result["source_hotspot_name"] = ctx.hotspot_name
        result["source_hot_line"] = ctx.hot_line
        result["source_lines"] = [{"line": ln, "text": text} for ln, text in ctx.lines]
    else:
        result["has_source"] = False
        result["source_display_name"] = ""
        result["source_hotspot_name"] = ""
        result["source_hot_line"] = -1
        result["source_lines"] = []

    # ── Overview redesign: run summary, breakdown, "investigate next" ──
    result["executable"] = meta.command
    result["host"] = meta.hostname or "—"
    result["backends"] = list(meta.backends_used or [])
    result["devices"] = [f"{d.name} ({d.backend})" for d in trace.devices]
    result["process_count"] = len({s.pid for s in trace.spans})
    result["thread_count"] = len({(s.pid, s.tid) for s in trace.spans})
    result["profiling_duration"] = result["wall_time"]
    result["capture_time"] = meta.capture_time_iso

    # Same merged-interval technique as waitPct/gpuActivePct above -- a
    # plain sum would double-count overlapping spans on different CPU
    # threads and could read well over 100%.
    cpu_ns = dash.merged_ns([s for s in trace.spans if s.category.value == "cpu"])
    result["cpu_util_pct"] = 100.0 * cpu_ns / wall_ns if wall_ns else 0.0

    # Shared with Timeline's own bucket legend/grouping (activity_buckets.py)
    # so the two views can never disagree about what counts as
    # Computation vs Runtime overhead vs Annotation -- see that module's
    # docstring for why the type= tag (already this codebase's
    # established compute-vs-overhead discriminator) drives this instead
    # of category alone.
    idle_ns = 0
    if gpu_stats is not None:
        idle_ns = max(int(gpu_stats["launch_gap_pct"] / 100.0 * wall_ns), 0)
    buckets = activity_buckets.bucket_totals(et.spans, idle_ns=idle_ns, et=et)
    grand = sum(buckets.values()) or 1
    _order = list(activity_buckets.BUCKETS)
    result["time_breakdown"] = [
        {"label": label, "pct": buckets[label] / grand * 100, "ns": buckets[label], "kind": "derived"}
        for label in _order if buckets.get(label, 0) > 0
    ]

    # No overhead-measurement instrumentation exists anywhere in this
    # codebase (confirmed during design) -- reported honestly as
    # unavailable rather than invented from an unrelated proxy number.
    result["profiling_overhead"] = {
        "label": "Profiling overhead", "value": "", "kind": "unavailable",
        "reason": "not measured by this build -- no overhead-instrumentation exists yet",
    }

    result["top_bottlenecks"] = [
        {"label": title, "value": metric, "kind": "measured", "reason": "",
         "icon": icon, "color": theme_mod.severity_color(severity, dark)}
        for icon, severity, title, metric in findings
    ]

    # Tab indices match Main.qml's TabBar order (Overview=0 .. Compare=9).
    _TAB = {"timeline": 1, "kernels": 2, "call_tree": 3, "roofline": 5, "source": 6}
    actions: list[dict[str, Any]] = []
    if gpu_stats is not None:
        if gpu_stats["launch_gap_pct"] > 30:
            actions.append({"label": "Find idle GPU gaps on the Timeline",
                             "tab": _TAB["timeline"], "category": "", "name": ""})
            actions.append({"label": "Check compute intensity on the Roofline",
                             "tab": _TAB["roofline"], "category": "", "name": ""})
        if gpu_stats["sync_stall_pct"] > 20:
            actions.append({"label": "Find synchronization hotspots in the Call Tree",
                             "tab": _TAB["call_tree"], "category": "", "name": ""})
    if result["wait_pct"] > 20 and not actions:
        actions.append({"label": "Find synchronization hotspots in the Call Tree",
                         "tab": _TAB["call_tree"], "category": "", "name": ""})
    if stats and stats[0]["pct"] > 30:
        top_cat, top_name = stats[0]["category"], stats[0]["name"]
        short = dash.fmt_kernel_name(top_name)[:30]
        actions.append({"label": f"Investigate dominant kernel '{short}' in Kernels",
                         "tab": _TAB["kernels"], "category": top_cat, "name": top_name})
        if result["has_source"]:
            actions.append({"label": f"View source for '{short}'",
                             "tab": _TAB["source"], "category": top_cat, "name": top_name})
    if not actions and stats:
        actions.append({"label": "Browse hot functions in Kernels", "tab": _TAB["kernels"],
                         "category": stats[0]["category"], "name": stats[0]["name"]})
    result["investigate_next"] = actions[:5]

    return result


class DashboardBridge(QObject):
    """Backs the Overview screen (Main.qml's OverviewScreen.qml)."""

    _TIMELINE_BUCKETS = _DASHBOARD_TIMELINE_BUCKETS

    def __init__(self, trace: Trace, theme, parent: QObject | None = None, *, comparison=None,
                 precomputed: dict[str, Any] | None = None) -> None:
        super().__init__(parent)
        self._trace = trace
        self._theme = theme
        # Optional, additive -- every existing caller passes nothing here
        # and gets byte-identical behavior to before this parameter
        # existed (asserted directly in tests, not just assumed). When
        # given, `comparison` is a ComparisonBridge (already computed its
        # own topImprovements/topRegressions; this bridge just surfaces
        # them for Overview's "largest improvements/regressions" ranking
        # instead of recomputing anything).
        self._comparison = comparison
        self._compute(precomputed)

    def _compute(self, precomputed: dict[str, Any] | None = None) -> None:
        # `precomputed`: the async-loading worker already ran
        # compute_dashboard_data() off-thread; every existing caller
        # (including every test) still computes it right here instead,
        # unchanged. Either way, every value lands on `self` via the same
        # `_<key>` attribute-name convention the Property getters below
        # already expect.
        data = precomputed if precomputed is not None else compute_dashboard_data(self._trace, self._theme.dark)
        for key, value in data.items():
            setattr(self, f"_{key}", value)
        self._findings_table = TableBundle(self._top_bottlenecks, columns.FINDINGS_COLUMNS, self)

    # ── Stat cards ───────────────────────────────────────────────────────
    @Property(str, constant=True)
    def diagnosisLabel(self) -> str:
        return self._diagnosis_label

    @Property(str, constant=True)
    def diagnosisColor(self) -> str:
        return self._theme.severityColor(self._diagnosis_severity)

    @Property(str, constant=True)
    def wallTime(self) -> str:
        return self._wall_time

    @Property(bool, constant=True)
    def gpuActiveAvailable(self) -> bool:
        return self._gpu_active_available

    @Property(float, constant=True)
    def gpuActivePct(self) -> float:
        return self._gpu_active_pct

    @Property(str, constant=True)
    def waitLabel(self) -> str:
        return self._wait_label

    @Property(float, constant=True)
    def waitPct(self) -> float:
        return self._wait_pct

    @Property(str, constant=True)
    def peakMemory(self) -> str:
        return self._peak_memory

    # ── Panels ───────────────────────────────────────────────────────────
    @Property('QVariantList', constant=True)
    def findings(self) -> list[dict[str, Any]]:
        return self._findings

    @Property('QVariantList', constant=True)
    def hotKernels(self) -> list[dict[str, Any]]:
        return self._hot_kernels

    @Property('QVariantList', constant=True)
    def timelinePreview(self) -> list[dict[str, Any]]:
        return self._timeline_preview

    @Property(bool, constant=True)
    def hasSourceContext(self) -> bool:
        return self._has_source

    @Property(str, constant=True)
    def sourceDisplayName(self) -> str:
        return self._source_display_name

    @Property(str, constant=True)
    def sourceHotspotName(self) -> str:
        return self._source_hotspot_name

    @Property(int, constant=True)
    def sourceHotLine(self) -> int:
        return self._source_hot_line

    @Property('QVariantList', constant=True)
    def sourceLines(self) -> list[dict[str, Any]]:
        return self._source_lines

    # ── Overview redesign: run summary, breakdown, "investigate next" ──────
    @Property(str, constant=True)
    def executable(self) -> str:
        return self._executable

    @Property(str, constant=True)
    def host(self) -> str:
        return self._host

    @Property(str, constant=True)
    def backends(self) -> str:
        return "  ".join(self._backends) or "none"

    @Property('QVariantList', constant=True)
    def devices(self) -> list[str]:
        return self._devices

    @Property(int, constant=True)
    def processCount(self) -> int:
        return self._process_count

    @Property(int, constant=True)
    def threadCount(self) -> int:
        return self._thread_count

    @Property(str, constant=True)
    def profilingDuration(self) -> str:
        return self._profiling_duration

    @Property(str, constant=True)
    def captureTime(self) -> str:
        return self._capture_time

    @Property(float, constant=True)
    def cpuUtilPct(self) -> float:
        return self._cpu_util_pct

    @Property('QVariantList', constant=True)
    def timeBreakdown(self) -> list[dict[str, Any]]:
        return self._time_breakdown

    @Property('QVariantMap', constant=True)
    def profilingOverhead(self) -> dict[str, str]:
        return self._profiling_overhead

    @Property('QVariantList', constant=True)
    def topBottlenecks(self) -> list[dict[str, Any]]:
        return self._top_bottlenecks

    @Property(QObject, constant=True)
    def findingsTable(self) -> QObject:
        return self._findings_table

    @Property('QVariantList', constant=True)
    def investigateNext(self) -> list[dict[str, Any]]:
        return self._investigate_next

    @Property(bool, constant=True)
    def hasComparison(self) -> bool:
        return self._comparison is not None and self._comparison.available

    @Property('QVariantList', constant=True)
    def comparisonTopChanges(self) -> list[dict[str, Any]]:
        """The largest improvements/regressions from Compare mode, for
        Overview's own ranking -- empty (not an error) when no comparison
        trace is loaded, same "additive, no comparison" case
        hasComparison covers."""
        if not self.hasComparison:
            return []
        return list(self._comparison.topImprovements) + list(self._comparison.topRegressions)


class KernelsBridge(QObject):
    """Backs the Kernels screen -- the GUI's equivalent of the TUI's
    HotspotsWidget (src/ui/app.py). Sort/filter happen client-side in
    QML over this one full list (aggregated by function name, so even a
    trace with tens of thousands of spans is at most a few hundred rows
    here -- no server-side pagination needed)."""

    def __init__(self, trace: Trace, theme, parent: QObject | None = None) -> None:
        super().__init__(parent)
        stats = trace.aggregated_stats()
        total_all = sum(r["total_ns"] for r in stats) or 1
        self._rows: list[dict[str, Any]] = [
            {
                "name": dash.fmt_kernel_name(r["name"]),
                # Untruncated -- the correlation key cross-tab navigation
                # uses (see src/gui/nav.py), never the display name,
                # which fmt_kernel_name() can truncate/reformat. Same
                # raw/display split SourceBridge.kernels already uses.
                "rawName": r["name"],
                "category": r["category"],
                "color": theme.categoryColor(r["category"]),
                "count": r["count"],
                "totalNs": r["total_ns"],
                "avgNs": r["avg_ns"],
                "minNs": r["min_ns"],
                "maxNs": r["max_ns"],
                "total": dash.fmt_ns(r["total_ns"]),
                "avg": dash.fmt_ns(r["avg_ns"]),
                "min": dash.fmt_ns(r["min_ns"]),
                "max": dash.fmt_ns(r["max_ns"]),
                "share": f"{r['total_ns']/total_all*100:.1f}%",
                "sharePct": r["total_ns"] / total_all * 100,
            }
            for r in stats
        ]
        self._table = TableBundle(self._rows, columns.KERNEL_COLUMNS, self)

    @Property('QVariantList', constant=True)
    def rows(self) -> list[dict[str, Any]]:
        return self._rows

    @Property(QObject, constant=True)
    def table(self) -> QObject:
        return self._table


def _ct_node_to_dict(node, dark: bool) -> dict[str, Any]:
    """`dark: bool`, not a Theme QObject -- so this (and the tree build
    it wraps, analysis/call_tree.py's _ct_build(), already Qt-free) is
    safe to call from the async-loading worker thread (src/gui/loader.py).
    See theme.py's category_color()/etc. for why a bool is passed
    instead of the Theme instance."""
    return {
        "name": node.name,
        "category": node.category,
        "color": theme_mod.category_color(node.category, dark),
        "totalNs": node.total_ns,
        "selfNs": node.self_ns,
        "avgNs": node.avg_ns,
        "count": node.count,
        "total": dash.fmt_ns(node.total_ns),
        "self": dash.fmt_ns(node.self_ns),
        "avg": dash.fmt_ns(node.avg_ns),
        "children": [_ct_node_to_dict(c, dark) for c in node.children],
    }


def _ct_filter_tree(nodes: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    """Prunes to nodes whose name matches `query` PLUS their ancestor
    chain (a match with its ancestors pruned away would float with no
    indication of where it's actually called from -- the call tree's
    whole point is that context). Case-insensitive substring match, same
    convention DataTable's own text filter uses."""
    if not query:
        return nodes
    q = query.lower()
    out = []
    for n in nodes:
        kept_children = _ct_filter_tree(n["children"], query)
        if q in n["name"].lower() or kept_children:
            out.append({**n, "children": kept_children})
    return out


def _ct_sort_tree(nodes: list[dict[str, Any]], key: str, desc: bool) -> list[dict[str, Any]]:
    if not key:
        return nodes
    ordered = sorted(nodes, key=lambda n: n.get(key, 0), reverse=desc)
    return [{**n, "children": _ct_sort_tree(n["children"], key, desc)} for n in ordered]


def compute_call_tree_data(trace: Trace, dark: bool) -> list[dict[str, Any]]:
    """The expensive, Qt-free part of CallTreeBridge's construction (tree
    build + node-to-dict conversion) -- a module-level function so the
    async-loading worker thread (src/gui/loader.py) can call it directly
    without ever touching a QObject off its owning thread. `dark`, not a
    Theme instance, for the same reason (see theme.py's category_color()
    docstring)."""
    try:
        from ..analysis.call_tree import _ct_build
        roots = _ct_build(trace.spans)
    except Exception:
        roots = []
    return [_ct_node_to_dict(r, dark) for r in roots]


class CallTreeBridge(QObject):
    """Backs the Call Tree screen. Reuses analysis/call_tree.py's _ct_build
    (stack-based when HPROFILER_CALLSTACK data is present, temporal-
    containment fallback otherwise -- see that function's own docstring)
    directly rather than re-deriving call-tree construction here; this
    class's only job is reshaping _CTNode's dataclass tree into plain
    nested dicts a QML recursive component can walk. Also used by the
    TUI's CallTreeWidget (src/ui/app.py, via a re-export) -- extracted to
    analysis/call_tree.py specifically so importing it here doesn't pull
    in the whole Textual-based TUI module.

    Deliberately NOT rebuilt on DataTable/TableBundle -- flattening a call
    TREE into a flat table would destroy the hierarchy that's the entire
    point of it. `roots` is now live (filter/sort applied) instead of
    `constant=True`; `allRoots` stays the permanently-unfiltered tree so
    InspectorBridge's "appears in the call tree" relationship can never be
    silently flipped to "unavailable" by an unrelated text filter here."""

    rootsChanged = Signal()

    def __init__(self, trace: Trace, theme, parent: QObject | None = None, *,
                 precomputed: list[dict[str, Any]] | None = None) -> None:
        super().__init__(parent)
        # `precomputed`: the async-loading worker already ran
        # compute_call_tree_data() off-thread; the common (non-worker)
        # construction path -- every existing caller, including every
        # test -- still computes it right here, unchanged.
        self._all_roots: list[dict[str, Any]] = (
            precomputed if precomputed is not None else compute_call_tree_data(trace, theme.dark)
        )
        self._filter_query = ""
        self._sort_key = ""
        self._sort_desc = True
        self._roots = self._all_roots

    def _recompute(self) -> None:
        nodes = self._all_roots
        if self._filter_query:
            nodes = _ct_filter_tree(nodes, self._filter_query)
        if self._sort_key:
            nodes = _ct_sort_tree(nodes, self._sort_key, self._sort_desc)
        self._roots = nodes
        self.rootsChanged.emit()

    @Property('QVariantList', notify=rootsChanged)
    def roots(self) -> list[dict[str, Any]]:
        return self._roots

    @Property('QVariantList', constant=True)
    def allRoots(self) -> list[dict[str, Any]]:
        return self._all_roots

    @Slot(str)
    def setFilter(self, query: str) -> None:
        self._filter_query = query
        self._recompute()

    @Slot(str, bool)
    def sortChildren(self, key: str, desc: bool) -> None:
        self._sort_key = key
        self._sort_desc = desc
        self._recompute()

    @Slot(str, result=bool)
    def exportCsv(self, path: str) -> bool:
        """Flattens the CURRENTLY visible (filtered+sorted) tree, one row
        per node, with a "Path" column (root-to-node names) standing in
        for the frozen identifier column a flat table would have --
        same open()+csv.writer+try/except OSError pattern TableBundle's
        own exportCsv uses."""
        rows: list[list[Any]] = []

        def walk(nodes: list[dict[str, Any]], path_parts: list[str]) -> None:
            for n in nodes:
                full_path = path_parts + [n["name"]]
                rows.append([n["name"], n["category"], " > ".join(full_path),
                             n["count"], n["totalNs"], n["selfNs"], n["avgNs"]])
                walk(n["children"], full_path)

        walk(self._roots, [])
        try:
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Function", "Category", "Path", "Calls",
                                  "Total (ns)", "Self (ns)", "Avg (ns)"])
                writer.writerows(rows)
            return True
        except OSError:
            return False


def _flame_node_with_color(node: dict[str, Any], dark: bool) -> dict[str, Any]:
    """analysis/flamegraph_tree.py's build_flame_tree() already returns
    the exact {name, value, category, children} shape the Flame Graph
    screen needs -- this just adds a theme-resolved "color" field
    (recursively), the same pattern _ct_node_to_dict above uses for Call
    Tree, so the two tabs share one color language (category -> hex)
    instead of the flame graph introducing its own separate per-function
    hash-coloring scheme the way the now-removed standalone
    `hprofiler flamegraph --gui` popup did in isolation. `dark: bool`,
    not a Theme QObject -- see theme.py's category_color() docstring for
    why (this runs on the async-loading worker thread)."""
    return {
        "name": node["name"],
        "value": node["value"],
        "category": node["category"],
        "color": theme_mod.category_color(node["category"], dark),
        "children": [_flame_node_with_color(c, dark) for c in node["children"]],
    }


def compute_flame_graph_data(trace: Trace, dark: bool) -> dict[str, Any]:
    """The expensive, Qt-free part of FlameGraphBridge's construction --
    a module-level function so the async-loading worker thread can call
    it directly (see compute_call_tree_data()'s docstring for why)."""
    from ..analysis.flamegraph_tree import build_flame_tree
    spans = [s for s in trace.spans if s.duration_ns > 0]
    tree = build_flame_tree(spans)
    return _flame_node_with_color(tree, dark)


class FlameGraphBridge(QObject):
    """Backs the Flame Graph screen. Reuses analysis/flamegraph_tree.py's
    build_flame_tree() (itself a thin wrapper over the SAME _ct_build
    CallTreeBridge above uses), so the two tabs can never disagree about
    the underlying call structure -- only the rendering differs
    (indented list there, proportional icicle here)."""

    def __init__(self, trace: Trace, theme, parent: QObject | None = None, *,
                 precomputed: dict[str, Any] | None = None) -> None:
        super().__init__(parent)
        self._tree = precomputed if precomputed is not None else compute_flame_graph_data(trace, theme.dark)

    @Property('QVariant', constant=True)
    def tree(self) -> dict[str, Any]:
        return self._tree

    @Property(int, constant=True)
    def totalNs(self) -> int:
        """Inclusive nanoseconds at the tree's root -- NOT a raw sample
        count (unlike the removed standalone popup's own totalSamples,
        which counted perf-collected folded-stack samples directly; this
        tree's "value" is real time, from build_flame_tree()'s reuse of
        _ct_build's inclusive-time accumulation)."""
        return self._tree.get("value", 0)


class RooflineBridge(QObject):
    """Backs the Roofline screen -- reuses analysis/roofline.py's
    analyze_trace() exactly like the TUI's RooflineWidget does (see
    src/ui/app.py); this class only reshapes the result into plain
    dicts + precomputed log-space plot coordinates (0..1 within the
    data's own bounds) for a QML Canvas to draw, so the QML side needs
    no log-scale math of its own."""

    _BOUND_COLOR = {"compute": "#22d3ee", "memory": "#e879f9", "unknown": "#9ca3af"}

    def __init__(self, trace: Trace, parent: QObject | None = None) -> None:
        super().__init__(parent)
        import math
        points: list[dict[str, Any]] = []
        try:
            from ..analysis.roofline import analyze_trace
            metrics = analyze_trace(trace)
        except Exception:
            metrics = []

        pts = [(d, m) for d, m in metrics if m.arith_intensity > 0 and m.achieved_tflops > 0]
        self._available = bool(pts)
        self._lines: list[dict[str, Any]] = []
        if pts:
            ai_vals = [m.arith_intensity for _, m in pts]
            tf_vals = [m.achieved_tflops for _, m in pts]
            ridge_vals = [m.ridge for _, m in pts if m.ridge > 0]
            peak_vals = [d.fp32_tflops for d, _ in pts if d.fp32_tflops > 0]

            lo_x = math.log10(max(min(ai_vals) * 0.5, 1e-3))
            hi_x = math.log10(max(max(ai_vals + ridge_vals) * 2.0, 10 ** (lo_x + 1)))
            lo_y = math.log10(max(min(tf_vals) * 0.3, 1e-4))
            hi_y = math.log10(max(max(tf_vals + peak_vals) * 1.3, 10 ** (lo_y + 1)))
            self._lo_x, self._hi_x, self._lo_y, self._hi_y = lo_x, hi_x, lo_y, hi_y

            def _pos(ai: float, tf: float) -> tuple[float, float]:
                fx = (math.log10(max(ai, 1e-6)) - lo_x) / (hi_x - lo_x)
                fy = (math.log10(max(tf, 1e-6)) - lo_y) / (hi_y - lo_y)
                return min(1.0, max(0.0, fx)), min(1.0, max(0.0, fy))

            for dev, m in pts:
                x, y = _pos(m.arith_intensity, m.achieved_tflops)
                points.append({
                    "x": x, "y": 1.0 - y,  # canvas y grows downward
                    "name": m.kernel_name, "bound": m.bound,
                    "color": self._BOUND_COLOR.get(m.bound, "#9ca3af"),
                    "ai": m.arith_intensity, "tflops": m.achieved_tflops,
                    "flopsPct": m.flops_pct, "bwPct": m.bw_pct,
                })

            seen: set[str] = set()
            for dev, m in pts:
                key = f"{dev.name}:{dev.fp32_tflops:.3f}"
                if key in seen or dev.fp32_tflops <= 0 or dev.bandwidth_gbs <= 0 or m.ridge <= 0:
                    continue
                seen.add(key)
                ai_lo = 10 ** lo_x
                tf_lo = max(dev.bandwidth_gbs * ai_lo / 1000.0, 1e-6)
                x0, y0 = _pos(ai_lo, tf_lo)
                x1, y1 = _pos(m.ridge, dev.fp32_tflops)
                x2, y2 = _pos(10 ** hi_x, dev.fp32_tflops)
                self._lines.append({
                    "device": dev.name,
                    "points": [
                        {"x": x0, "y": 1.0 - y0}, {"x": x1, "y": 1.0 - y1}, {"x": x2, "y": 1.0 - y2},
                    ],
                })
        else:
            self._lo_x = self._hi_x = self._lo_y = self._hi_y = 0.0

        self._points = points

    @Property(bool, constant=True)
    def available(self) -> bool:
        return self._available

    @Property('QVariantList', constant=True)
    def points(self) -> list[dict[str, Any]]:
        return self._points

    @Property('QVariantList', constant=True)
    def rooflineLines(self) -> list[dict[str, Any]]:
        return self._lines

    @Property(str, constant=True)
    def aiRangeLabel(self) -> str:
        return f"{10**self._lo_x:.2g}–{10**self._hi_x:.2g} FLOP/B"

    @Property(str, constant=True)
    def tflopsRangeLabel(self) -> str:
        return f"{10**self._lo_y:.2g}–{10**self._hi_y:.2g} TFLOP/s"


# Same instruction-type semantics as disasm/classifier.py's ITYPE_COLOR,
# independently expressed as hex for the same Rich-name-vs-QML-hex reason
# as theme.py's category colors.
_ITYPE_HEX = {
    "vec_sp": "#4ade80", "vec_dp": "#22d3ee", "vec_mem": "#fb923c", "vector": "#16a34a",
    "scalar": "#7dd3fc", "memory": "#fbbf24", "control": "#e879f9", "sync": "#f87171",
    "compute": "#60a5fa", "int_compute": "#3b82f6", "tensor": "#c084fc", "other": "#9ca3af",
}

# Fuller labels than the TUI's ITYPE_LABEL (classifier.py) -- that one is
# terse (3 chars: "vsp", "mem", ...) to fit the TUI's narrow bar; this
# panel has room for real words. Same instruction-type set as _ITYPE_HEX
# (classifier.py's ITYPE_LABEL is missing int_compute/tensor entirely).
_ITYPE_MIX_LABEL = {
    "vec_sp": "Vector (FP32)", "vec_dp": "Vector (FP64)", "vec_mem": "Vector load/store",
    "vector": "Vector (int/misc)", "scalar": "Scalar", "memory": "Memory",
    "control": "Control flow", "sync": "Sync/barrier", "compute": "FMA/compute",
    "int_compute": "Int compute", "tensor": "Tensor core", "other": "Other",
}
_ITYPE_MIX_ORDER = [
    "vec_sp", "vec_dp", "vec_mem", "vector", "tensor", "compute", "int_compute",
    "scalar", "memory", "control", "sync", "other",
]


class SourceBridge(QObject):
    """Backs the Source screen -- the GUI's equivalent of the TUI's
    DisasmWidget (src/ui/app.py). Kernel list is small (one row per
    profiled/disassembled function); each kernel's actual instruction
    lines are fetched on demand via disasmLines() so a trace with many
    disassembled kernels doesn't pay to reshape all of them upfront.

    Unlike this file's other bridges, `kernels` is NOT a constant
    Property: `hprofiler gui --disasm` (or `run --gui --disasm`) starts
    disassembly collection as a background thread (see
    output/chrome_trace.py's load_trace_from_json) that can still be
    running when this window opens -- a real trace's disasm often isn't
    fully resolved yet at load time. A constant snapshot from __init__
    would permanently show "disassembly still failed" for any function
    that resolves a few seconds later, which is exactly the bug a user
    reported (disasm worked in the TUI -- which polls -- but never
    appeared in the GUI, which didn't poll at all). Mirrors the TUI's
    DisasmWidget._poll_disasm_ready: a 0.5s QTimer watches
    trace._disasm_version (bumped by Trace.add_disasm) and rebuilds/
    re-emits only when it actually changes."""

    kernelsChanged = Signal()

    def __init__(self, trace: Trace, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._trace = trace
        self._last_disasm_version = -1
        self._kernels: list[dict[str, Any]] = []
        self._rebuild_kernels()

        self._poll = QTimer(self)
        self._poll.setInterval(500)
        self._poll.timeout.connect(self._rebuild_kernels)
        self._poll.start()

    def _rebuild_kernels(self) -> None:
        cur_version = self._trace._disasm_version
        if cur_version == self._last_disasm_version:
            return
        self._last_disasm_version = cur_version

        trace = self._trace
        stats = trace.aggregated_stats()
        profiled_names = [r["name"] for r in stats]
        stats_by_name = {r["name"]: r for r in stats}
        disasm = trace.disasm
        disasm_only = [n for n in disasm if n not in profiled_names]
        names = profiled_names + disasm_only

        kernels: list[dict[str, Any]] = []
        for name in names:
            kd = disasm.get(name)
            stat = stats_by_name.get(name)
            # kd.name (== `name` here) is the span/event label hprofiler
            # itself invents ("omp_barrier", "MPI_Bcast") -- there is no
            # real ELF symbol by that name. kd.mangled_name, when set, is
            # the REAL function that was actually disassembled: for an
            # OMP/MPI-hook-resolved kernel this is the call site in the
            # PROFILED PROGRAM's own code that triggered the event (the
            # OpenMP/MPI runtime's own implementation is never what gets
            # shown -- see hooks/common/codeptr_resolve.h), i.e. "which of
            # YOUR functions hit this barrier/collective". Demangled here
            # so the Source screen can tell the user what they're actually
            # looking at instead of just the event label.
            symbol = dash.demangle(kd.mangled_name) if kd and kd.mangled_name else ""
            kernels.append({
                "name": dash.fmt_kernel_name(name),
                "rawName": name,
                "hasDisasm": kd is not None,
                "arch": kd.arch if kd else "—",
                "total": dash.fmt_ns(stat["total_ns"]) if stat else "—",
                "symbol": symbol,
            })
        self._kernels = kernels
        self.kernelsChanged.emit()

    @Property('QVariantList', notify=kernelsChanged)
    def kernels(self) -> list[dict[str, Any]]:
        return self._kernels

    @Slot(str, result='QVariantList')
    def disasmLines(self, raw_name: str) -> list[dict[str, Any]]:
        kd = self._trace.disasm.get(raw_name)
        if kd is None:
            return []
        out = []
        prev_loc = (None, None)
        for ln in kd.lines:
            itype = ln.itype.value if hasattr(ln.itype, "value") else str(ln.itype)
            # Source location (populated by disasm/source_ann.py, which
            # runs unconditionally after every disasm collection -- same
            # data the TUI's DisasmWidget already interleaves as "//
            # file:line" comment rows). sourceChanged marks only the
            # first instruction at a given file:line, so the QML delegate
            # can show the label once per source line instead of on every
            # single instruction row.
            loc = (ln.source_file, ln.source_line)
            source_changed = bool(ln.source_file) and loc != prev_loc
            prev_loc = loc
            out.append({
                "addr": f"{ln.addr:x}",
                "mnemonic": ln.mnemonic,
                "operands": ln.operands,
                "comment": ln.comment,
                "itype": itype,
                "color": _ITYPE_HEX.get(itype, "#9ca3af"),
                "samplePct": ln.sample_pct,
                "sourceFile": ln.source_file,
                "sourceLine": ln.source_line,
                "sourceChanged": source_changed,
            })
        return out

    @Slot(str, result=str)
    def noDisasmReason(self, raw_name: str) -> str:
        """Mirrors the TUI's smarter "No disassembly available" message
        (src/ui/app.py's DisasmWidget._show_disasm): distinguishes a
        genuinely-missing call-site tag (stale trace / unrebuilt hooks)
        from a tag that resolved but disassembly still failed (missing
        objdump/nm) -- see that method's own comment for why this
        distinction matters (a real user report)."""
        for s in self._trace.spans:
            if s.name != raw_name:
                continue
            if s.tags.get("sym"):
                return f"Call site resolved (sym={s.tags['sym']}) but disassembly still " \
                       f"failed -- likely a missing tool (objdump/nm) or the symbol/library " \
                       f"wasn't found where expected."
            if s.tags.get("lib"):
                return f"Call site resolved (lib={s.tags['lib']}) but disassembly still " \
                       f"failed -- likely a missing tool (objdump/nm) or the symbol/library " \
                       f"wasn't found where expected."
            if s.tags.get("type") == "kernel":
                return "No disassembly for this GPU kernel -- install cuobjdump " \
                       "(CUDA) or llvm-objdump (ROCm)."
        return "No resolved call-site symbol at all for this event -- most likely a " \
               "trace captured before rebuilding the hooks, or a construct that doesn't " \
               "resolve one yet (point-to-point MPI, omp_critical_hold). Not a missing-tool problem."

    @Slot(str, result='QVariantList')
    def instructionMix(self, raw_name: str) -> list[dict[str, Any]]:
        """Instruction-type breakdown (vector/memory/scalar/control/...)
        for the disassembled kernel -- the GUI's equivalent of the TUI's
        DisasmWidget._show_mix. KernelDisasm.itype_counts() already
        existed and was already used by the TUI; the GUI's Source screen
        never called it at all, so this was blank-but-should-have-been-
        computed, not a new metric being invented here."""
        kd = self._trace.disasm.get(raw_name)
        if kd is None or not kd.lines:
            return []
        counts = kd.itype_counts()
        total = sum(counts.values()) or 1
        rows = []
        for itype in _ITYPE_MIX_ORDER:
            count = counts.get(InsnType(itype), 0)
            if count <= 0:
                continue
            rows.append({
                "type": itype,
                "label": _ITYPE_MIX_LABEL.get(itype, itype),
                "color": _ITYPE_HEX.get(itype, "#9ca3af"),
                "count": count,
                "pct": 100.0 * count / total,
            })
        return rows

    @Slot(str, result='QVariantList')
    def advisorHints(self, raw_name: str) -> list[dict[str, Any]]:
        """Static-analysis hints for the disassembled kernel (missed
        vectorization, register spills, memory-bound sections, ...) --
        the GUI's equivalent of the TUI's DisasmWidget._show_hints.
        analysis/asm_advisor.py already existed, already purely static
        (no runtime data needed) and already used by the TUI; the GUI
        never called it at all. No LLM/network call involved -- pure
        pattern analysis of the instruction stream, same as the TUI,
        so this works identically on an air-gapped HPC compute node."""
        kd = self._trace.disasm.get(raw_name)
        if kd is None or not kd.lines:
            return []
        try:
            from ..analysis.asm_advisor import advise
            advices = advise(kd)
        except Exception:
            return []
        _SEV_HEX = {"crit": "#f87171", "warn": "#fbbf24", "info": "#22d3ee"}
        return [
            {
                "severity": adv.severity,
                "category": adv.category,
                "message": adv.message,
                "detail": adv.detail,
                "icon": adv.icon,
                "color": _SEV_HEX.get(adv.severity, "#e6edf3"),
            }
            for adv in advices
        ]


class SystemBridge(QObject):
    """Backs the System screen -- device specs + CPU microarch counters,
    the GUI's equivalent of the TUI's SystemWidget."""

    def __init__(self, trace: Trace, parent: QObject | None = None) -> None:
        super().__init__(parent)
        meta = trace.metadata
        self._command = f"{meta.command} {' '.join(meta.args[:6])}".strip()
        self._host = meta.hostname or "—"
        self._duration = dash.fmt_ns(dash.trace_wall_ns(trace))
        self._backends = list(meta.backends_used or [])

        self._devices: list[dict[str, Any]] = []
        for dev in trace.devices:
            peaks = []
            if dev.fp16_tflops: peaks.append(f"FP16 {dash.fmt_tf(dev.fp16_tflops)}")
            if dev.fp32_tflops: peaks.append(f"FP32 {dash.fmt_tf(dev.fp32_tflops)}")
            if dev.fp64_tflops: peaks.append(f"FP64 {dash.fmt_tf(dev.fp64_tflops)}")
            if dev.tensor_tflops: peaks.append(f"Tensor {dash.fmt_tf(dev.tensor_tflops)}")
            ridge_hint = ""
            if dev.ridge_point > 0:
                ridge_hint = ("memory-bound" if dev.ridge_point < 5 else
                              "balanced" if dev.ridge_point < 30 else "compute-bound")
            self._devices.append({
                "backend": dev.backend, "name": dev.name, "computeCap": dev.compute_cap,
                "smCount": dev.sm_count, "clockGhz": dev.core_clock_ghz,
                "peaks": "  ".join(peaks),
                "bandwidth": f"{dev.bandwidth_gbs:.0f} GB/s" if dev.bandwidth_gbs else "",
                "vram": f"{dev.vram_gb:.1f} GB" if dev.vram_gb > 0 else "",
                "ridgeHint": ridge_hint,
            })

        ctrs = {c.name: c.value for c in trace.counters}
        self._ipc = ctrs.get("ipc", 0.0)
        self._cacheMiss = ctrs.get("cache_miss_pct", -1.0)
        self._branchMiss = ctrs.get("branch_miss_pct", -1.0)
        rss = ctrs.get("process_max_rss_bytes", 0.0)
        self._rss = dash.fmt_bytes(rss) if rss > 0 else ""

        self._device_table = TableBundle(self._devices, columns.DEVICE_COLUMNS, self)

        # Field-shaped rows -- an absent PMU counter becomes an honest
        # "unavailable" row instead of the old `visible: System.ipc > 0`-
        # style silent disappearance (same style QML used to gate these).
        no_perf_reason = "no PMU counters captured -- run with `perf` available, or --no-perf wasn't used"
        metric_rows = [
            {"label": "IPC", "value": f"{self._ipc:.2f}" if self._ipc > 0 else "",
             "kind": "measured" if self._ipc > 0 else "unavailable",
             "reason": "" if self._ipc > 0 else no_perf_reason},
            {"label": "Cache miss", "value": f"{self._cacheMiss:.1f}%" if self._cacheMiss >= 0 else "",
             "kind": "measured" if self._cacheMiss >= 0 else "unavailable",
             "reason": "" if self._cacheMiss >= 0 else no_perf_reason},
            {"label": "Branch miss", "value": f"{self._branchMiss:.1f}%" if self._branchMiss >= 0 else "",
             "kind": "measured" if self._branchMiss >= 0 else "unavailable",
             "reason": "" if self._branchMiss >= 0 else no_perf_reason},
            {"label": "Peak RSS", "value": self._rss,
             "kind": "measured" if self._rss else "unavailable",
             "reason": "" if self._rss else "process_max_rss_bytes counter not present in this trace"},
        ]
        self._metric_table = TableBundle(metric_rows, columns.SYSTEM_METRIC_COLUMNS, self)

    @Property(str, constant=True)
    def command(self) -> str: return self._command

    @Property(str, constant=True)
    def host(self) -> str: return self._host

    @Property(str, constant=True)
    def duration(self) -> str: return self._duration

    @Property(str, constant=True)
    def backends(self) -> str: return "  ".join(self._backends) or "none"

    @Property('QVariantList', constant=True)
    def devices(self) -> list[dict[str, Any]]: return self._devices

    @Property(float, constant=True)
    def ipc(self) -> float: return self._ipc

    @Property(float, constant=True)
    def cacheMissPct(self) -> float: return self._cacheMiss

    @Property(float, constant=True)
    def branchMissPct(self) -> float: return self._branchMiss

    @Property(str, constant=True)
    def peakRss(self) -> str: return self._rss

    @Property(QObject, constant=True)
    def deviceTable(self) -> QObject: return self._device_table

    @Property(QObject, constant=True)
    def metricTable(self) -> QObject: return self._metric_table


class ProfileBridge(QObject):
    """Backs the Profile screen -- per-backend activity, time breakdown,
    hotspots, insight tips. The GUI's equivalent of the TUI's
    ProfileWidget; shares _bottleneck_analysis with it via
    analysis/dashboard.py so the tips never disagree."""

    def __init__(self, trace: Trace, theme, parent: QObject | None = None) -> None:
        super().__init__(parent)
        wall_ns = dash.trace_wall_ns(trace)
        spans = trace.spans

        self._gpu_activity: list[dict[str, Any]] = []
        for cat_val, label in (("cuda", "CUDA"), ("rocm", "ROCm")):
            kspans = [s for s in spans if s.category.value == cat_val and s.tags.get("type") == "kernel"]
            if not kspans:
                continue
            kern_ns = dash.merged_ns(kspans)
            kern_acc = sum(s.duration_ns for s in kspans)
            sync_ns = dash.merged_ns([s for s in spans if s.category.value == "sync"])
            pct = 100.0 * kern_ns / wall_ns if wall_ns else 0.0
            sync_pct = 100.0 * sync_ns / wall_ns if wall_ns else 0.0
            eff = pct / (pct + sync_pct) * 100 if (pct + sync_pct) > 0 else 0
            self._gpu_activity.append({
                "label": label, "activePct": pct, "syncPct": sync_pct, "efficiency": eff,
                "launches": len(kspans), "total": dash.fmt_ns(kern_acc),
                "color": theme.categoryColor(cat_val),
            })

        by_cat: dict[str, dict[str, Any]] = {}
        for s in spans:
            d = by_cat.setdefault(s.category.value, {"ns": 0, "n": 0})
            d["ns"] += s.duration_ns
            d["n"] += 1
        grand = sum(v["ns"] for v in by_cat.values()) or 1
        self._breakdown: list[dict[str, Any]] = [
            {
                "category": cat, "color": theme.categoryColor(cat),
                "pct": info["ns"] / grand * 100, "total": dash.fmt_ns(info["ns"]),
                "count": info["n"],
            }
            for cat, info in sorted(by_cat.items(), key=lambda kv: -kv[1]["ns"])
        ]

        stats = trace.aggregated_stats()
        total_all = sum(r["total_ns"] for r in stats) or 1
        self._hotspots: list[dict[str, Any]] = [
            {
                "name": dash.fmt_kernel_name(r["name"]), "category": r["category"],
                "color": theme.categoryColor(r["category"]),
                "share": r["total_ns"] / total_all * 100,
                "total": dash.fmt_ns(r["total_ns"]), "count": r["count"],
            }
            for r in stats[:12]
        ]

        ctrs = {c.name: c.value for c in trace.counters}
        ctr_sub = {k: ctrs[k] for k in ("ipc", "cache_miss_pct", "branch_miss_pct") if k in ctrs}
        try:
            tips = dash.bottleneck_analysis(trace, ctr_sub)
        except Exception:
            tips = []
        self._insight: list[dict[str, Any]] = [
            {"icon": t[:1], "text": t[2:].strip()} for t in tips
        ]

    @Property('QVariantList', constant=True)
    def gpuActivity(self) -> list[dict[str, Any]]: return self._gpu_activity

    @Property('QVariantList', constant=True)
    def breakdown(self) -> list[dict[str, Any]]: return self._breakdown

    @Property('QVariantList', constant=True)
    def hotspots(self) -> list[dict[str, Any]]: return self._hotspots

    @Property('QVariantList', constant=True)
    def insight(self) -> list[dict[str, Any]]: return self._insight
