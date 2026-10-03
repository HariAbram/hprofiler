"""
Tests for src/gui/theme.py and src/gui/bridge.py -- the Qt/QML GUI's
Python-side data layer. Skipped entirely if PySide6 isn't installed
(it's an optional dependency, `pip install hprofiler[gui]`), matching
this project's existing pattern for optional-tool-dependent tests (e.g.
test_gomp_hook.py skipping when gcc isn't available).

Runs headless (QT_QPA_PLATFORM=offscreen) -- these tests check computed
Property values, not rendering (tests/test_gui_timeline_hover.py covers
real QML interaction), so a real display is not needed; a
QGuiApplication instance IS needed since
PySide6's QObject/Property machinery requires one to exist.
"""
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtGui import QGuiApplication
    _PYSIDE6_AVAILABLE = True
except ImportError:
    _PYSIDE6_AVAILABLE = False

from src.core.trace import Trace, TraceMetadata
from src.core.events import SpanEvent, CounterEvent, Category
from src.analysis.device import DevicePeak


def _span(pid, tid, cat, start_ns, dur_ns, name, tags=None):
    return SpanEvent(name=name, category=cat, start_ns=start_ns, duration_ns=dur_ns,
                     pid=pid, tid=tid, tags=dict(tags or {}))


def _dev() -> DevicePeak:
    return DevicePeak(name="A100", backend="cuda", fp32_tflops=19.5, fp64_tflops=9.7,
                      fp16_tflops=78.0, bandwidth_gbs=1555, sm_count=108,
                      core_clock_ghz=1.41, mem_clock_ghz=1.2, mem_bus_bits=5120,
                      vram_gb=40.0, compute_cap="8.0")


def _mk_trace(spans, backends=None, command="./a.out"):
    meta = TraceMetadata(command=command, args=[], backends_used=backends or [])
    t = Trace(meta)
    for s in spans:
        t.add(s)
    return t


@unittest.skipUnless(_PYSIDE6_AVAILABLE, "PySide6 not installed")
class TestGuiBridge(unittest.TestCase):
    _app = None

    @classmethod
    def setUpClass(cls):
        # One QGuiApplication for the whole test class -- PySide6 doesn't
        # allow more than one live at a time in a single process.
        cls._app = QGuiApplication.instance() or QGuiApplication([])

    def _theme(self):
        from src.gui.theme import Theme
        return Theme(dark=True)

    # ── Theme ────────────────────────────────────────────────────────────

    def test_dark_and_light_backgrounds_differ(self):
        theme = self._theme()
        dark_bg = theme.background
        theme.dark = False
        light_bg = theme.background
        self.assertNotEqual(dark_bg, light_bg)
        self.assertEqual(dark_bg, "#0d1117")
        self.assertEqual(light_bg, "#ffffff")

    def test_toggle_flips_dark_flag(self):
        theme = self._theme()
        self.assertTrue(theme.dark)
        theme.toggle()
        self.assertFalse(theme.dark)
        theme.toggle()
        self.assertTrue(theme.dark)

    def test_category_color_known_and_unknown(self):
        theme = self._theme()
        self.assertEqual(theme.categoryColor("cuda"), "#f87171")
        # Falls back to "other" rather than raising for a category this
        # palette doesn't know about (forward-compatible with any future
        # category the trace format adds).
        self.assertEqual(theme.categoryColor("nonexistent"), theme.categoryColor("other"))

    def test_severity_color_matches_tui_semantic_families(self):
        theme = self._theme()
        # Must accept exactly the severity family strings
        # analysis/dashboard.py's diagnose()/top_findings() emit.
        for sev in ("red", "yellow", "green", "cyan"):
            color = theme.severityColor(sev)
            self.assertTrue(color.startswith("#"))

    def test_category_and_severity_colors_change_with_theme(self):
        theme = self._theme()
        dark_color = theme.categoryColor("cuda")
        theme.dark = False
        light_color = theme.categoryColor("cuda")
        self.assertNotEqual(dark_color, light_color)

    # ── Layout tokens (visual-consistency audit) ────────────────────────
    # spacing/radius/typography/row/button scales added so screens stop
    # each picking their own ad hoc numbers -- see theme.py's module
    # docstring. constant=True (not notify=themeChanged): these never
    # vary with the dark/light toggle, only colors do.

    def test_spacing_scale_is_ascending_ints(self):
        theme = self._theme()
        values = [theme.spacingXs, theme.spacingSm, theme.spacingMd,
                  theme.spacingLg, theme.spacingXl]
        self.assertEqual(values, sorted(values))
        self.assertTrue(all(isinstance(v, int) for v in values))
        # spacingMd is the specific value the audit found already
        # dominant (most common ad hoc panel-padding number) -- the
        # token was chosen to match it, not invent a new one.
        self.assertEqual(theme.spacingMd, 8)

    def test_radius_scale(self):
        theme = self._theme()
        self.assertLess(theme.radiusSmall, theme.radiusPanel)
        # radiusPanel is the radius every panel block uses.
        self.assertEqual(theme.radiusPanel, 6)

    def test_typography_scale_is_ascending_by_role(self):
        theme = self._theme()
        # caption < label < body < title < heading -- typeValue
        # (StatCard's headline numbers) is deliberately the largest and
        # not part of this ascending body-text progression.
        values = [theme.typeCaption, theme.typeLabel, theme.typeBody,
                  theme.typeTitle, theme.typeHeading]
        self.assertEqual(values, sorted(values))
        self.assertGreater(theme.typeValue, theme.typeHeading)

    def test_row_and_button_and_field_tokens_exist(self):
        theme = self._theme()
        self.assertLess(theme.rowCompact, theme.rowComfortable)
        self.assertGreater(theme.statRowHeight, theme.rowComfortable)
        self.assertGreater(theme.buttonHeight, 0)
        self.assertGreater(theme.iconButtonWidth, 0)
        self.assertGreater(theme.fieldWidth, 0)

    def test_semantic_severity_aliases_match_underlying_family(self):
        # Additive, not a replacement -- severityColor()'s "red"/"yellow"/
        # "green"/"cyan" family-name contract (depended on by
        # analysis/dashboard.py's diagnose() and 9 bridge.py call sites)
        # is untouched; these are just clearer names for exactly the same
        # colors, in both theme states.
        theme = self._theme()
        for dark in (True, False):
            theme.dark = dark
            self.assertEqual(theme.errorColor, theme.severityColor("red"))
            self.assertEqual(theme.warningColor, theme.severityColor("yellow"))
            self.assertEqual(theme.successColor, theme.severityColor("green"))
            self.assertEqual(theme.infoColor, theme.severityColor("cyan"))

    def test_revised_category_colors_for_colorblind_safety(self):
        # mpi/memory/jit/nvtx values were chosen with a quantitative
        # deuteranopia/protanopia/tritanopia simulation so they aren't
        # confusable with other categories (e.g. mpi vs. memory under both
        # common red-green CVD forms). Exact values pinned so an edit can't
        # silently drift back to confusable ones.
        theme = self._theme()
        theme.dark = True
        self.assertEqual(theme.categoryColor("mpi"), "#2563eb")
        self.assertEqual(theme.categoryColor("memory"), "#a78bfa")
        self.assertEqual(theme.categoryColor("jit"), "#8b5cf6")
        self.assertEqual(theme.categoryColor("nvtx"), "#ea580c")
        theme.dark = False
        self.assertEqual(theme.categoryColor("mpi"), "#1d4ed8")
        self.assertEqual(theme.categoryColor("memory"), "#7c3aed")
        self.assertEqual(theme.categoryColor("jit"), "#581c87")
        self.assertEqual(theme.categoryColor("nvtx"), "#9a3412")

    def test_bucket_color_covers_every_bucket_and_varies_with_theme(self):
        from src.analysis.activity_buckets import BUCKETS
        theme = self._theme()
        for bucket in BUCKETS:
            theme.dark = True
            dark_color = theme.bucketColor(bucket)
            self.assertTrue(dark_color.startswith("#"))
            theme.dark = False
            light_color = theme.bucketColor(bucket)
            self.assertTrue(light_color.startswith("#"))

    def test_bucket_color_unknown_bucket_falls_back_gracefully(self):
        theme = self._theme()
        self.assertTrue(theme.bucketColor("NotARealBucket").startswith("#"))

    def test_change_color_covers_every_status(self):
        theme = self._theme()
        for status in ("improved", "regressed", "unchanged", "new", "removed",
                       "unavailable", "zero"):
            self.assertTrue(theme.changeColor(status).startswith("#"))
        self.assertEqual(theme.changeColor("improved"), theme.severityColor("green"))
        self.assertEqual(theme.changeColor("regressed"), theme.severityColor("red"))
        self.assertEqual(theme.changeColor("unchanged"), theme.textMuted)

    def test_no_raw_hex_colors_outside_theme_py(self):
        # Regression guard for the visual-consistency audit's core
        # finding: FlameGraphScreen.qml alone had 8 raw hex literals that
        # silently stopped repainting on the light/dark toggle (color
        # bindings to a literal string aren't reactive the way
        # AppTheme.* bindings are) -- this single grep-based check would
        # have caught all 12 hits (3 files) across the whole GUI
        # instantly, before it shipped. Any legitimate new raw color
        # belongs in theme.py as a named token, not inline in a screen.
        import re
        qml_dir = Path(__file__).resolve().parent.parent / "src" / "gui" / "qml"
        hex_re = re.compile(r'#[0-9a-fA-F]{3,8}\b')
        offenders = []
        for qml_file in qml_dir.rglob("*.qml"):
            text = qml_file.read_text()
            for lineno, line in enumerate(text.splitlines(), start=1):
                if hex_re.search(line):
                    offenders.append(f"{qml_file.relative_to(qml_dir)}:{lineno}: {line.strip()}")
        self.assertEqual(offenders, [], "raw hex color literal(s) found outside theme.py:\n" + "\n".join(offenders))

    def test_light_mpi_no_longer_collides_with_old_dark_mpi(self):
        # Light-mode mpi must not equal #2563eb (the dark-mode mpi value):
        # light/dark palettes are checked independently, so both sides
        # must keep distinct values.
        theme = self._theme()
        theme.dark = False
        self.assertNotEqual(theme.categoryColor("mpi"), "#60a5fa")
        self.assertEqual(theme.categoryColor("mpi"), "#1d4ed8")

    # ── DashboardBridge ──────────────────────────────────────────────────

    def _bridge(self, trace):
        from src.gui.bridge import DashboardBridge
        return DashboardBridge(trace, self._theme())

    def test_rich_trace_populates_headline_stats(self):
        spans = [_span(1, 1, Category.MPI, i * 1_000_000, 200_000, "MPI_Allreduce",
                       tags={"rank": "0"}) for i in range(5)]
        spans += [_span(2, 2, Category.GPU_CUDA, i * 400_000, 150_000, "k",
                        tags={"type": "kernel"}) for i in range(10)]
        trace = _mk_trace(spans, backends=["cuda", "mpi"], command="gmx_mpi")
        trace.add(CounterEvent(name="process_max_rss_bytes", category=Category.OTHER,
                               timestamp_ns=0, value=2_000_000_000, pid=1))
        bridge = self._bridge(trace)

        self.assertTrue(bridge.diagnosisLabel)
        self.assertTrue(bridge.diagnosisColor.startswith("#"))
        self.assertIn("ms", bridge.wallTime)
        self.assertTrue(bridge.gpuActiveAvailable)
        self.assertEqual(bridge.waitLabel, "MPI WAIT")  # MPI spans present
        self.assertIn("GB", bridge.peakMemory)

    def test_no_mpi_falls_back_to_sync_wait_label(self):
        spans = [_span(1, 1, Category.CPU, 0, 100, "x"),
                _span(1, 1, Category.SYNC, 100, 50, "cudaDeviceSynchronize")]
        trace = _mk_trace(spans, backends=["cpu"])
        bridge = self._bridge(trace)
        self.assertEqual(bridge.waitLabel, "SYNC WAIT")

    def test_no_gpu_backend_reports_unavailable(self):
        trace = _mk_trace([_span(1, 1, Category.CPU, 0, 100, "x")], backends=["cpu"])
        bridge = self._bridge(trace)
        self.assertFalse(bridge.gpuActiveAvailable)

    def test_hot_kernels_sorted_by_total_time_desc(self):
        spans = [_span(1, 1, Category.CPU, i, 100, "hot") for i in range(10)]
        spans += [_span(1, 1, Category.CPU, 2000 + i, 5, "cold") for i in range(2)]
        trace = _mk_trace(spans, backends=["cpu"])
        bridge = self._bridge(trace)
        kernels = bridge.hotKernels
        self.assertGreaterEqual(len(kernels), 2)
        self.assertEqual(kernels[0]["name"], "hot")
        self.assertEqual(kernels[-1]["name"], "cold")

    def test_hot_kernels_have_hex_colors_from_theme(self):
        trace = _mk_trace([_span(1, 1, Category.GPU_CUDA, 0, 100, "k",
                                 tags={"type": "kernel"})], backends=["cuda"])
        bridge = self._bridge(trace)
        for k in bridge.hotKernels:
            self.assertTrue(k["color"].startswith("#"))

    def test_findings_list_shape(self):
        spans = [_span(1, 1, Category.GPU_CUDA, i * 1_000_000, 50_000, "k",
                       tags={"type": "kernel"}) for i in range(20)]
        trace = _mk_trace(spans, backends=["cuda"])
        bridge = self._bridge(trace)
        for f in bridge.findings:
            self.assertIn("icon", f)
            self.assertIn("color", f)
            self.assertIn("title", f)
            self.assertIn("metric", f)
            self.assertTrue(f["color"].startswith("#"))

    def test_timeline_preview_has_at_most_three_categories(self):
        spans = []
        for cat in (Category.CPU, Category.MPI, Category.GPU_CUDA, Category.SYNC, Category.MEMORY):
            spans.append(_span(1, 1, cat, 0, 1000, f"x_{cat.value}"))
        trace = _mk_trace(spans, backends=["cpu", "mpi", "cuda"])
        bridge = self._bridge(trace)
        self.assertLessEqual(len(bridge.timelinePreview), 3)
        for row in bridge.timelinePreview:
            self.assertIn("category", row)
            self.assertIn("color", row)
            self.assertIn("coverage", row)
            self.assertEqual(len(row["coverage"]), DashboardBridgeBuckets())

    def test_no_source_context_reports_unavailable(self):
        trace = _mk_trace([_span(1, 1, Category.CPU, 0, 100, "hot_fn")], backends=["cpu"])
        bridge = self._bridge(trace)
        self.assertFalse(bridge.hasSourceContext)
        self.assertEqual(bridge.sourceLines, [])

    def test_source_context_when_file_and_line_present(self):
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".c", delete=False) as f:
            f.write("\n".join(f"line{i}" for i in range(1, 11)) + "\n")
            path = f.name
        try:
            trace = _mk_trace(
                [_span(1, 1, Category.CPU, 0, 100, "hot_fn", tags={"file": path, "line": "5"})],
                backends=["cpu"])
            bridge = self._bridge(trace)
            self.assertTrue(bridge.hasSourceContext)
            self.assertEqual(bridge.sourceHotLine, 5)
            self.assertTrue(any(ln["line"] == 5 and ln["text"] == "line5"
                                for ln in bridge.sourceLines))
        finally:
            Path(path).unlink()

    def test_minimal_single_span_trace_does_not_crash(self):
        trace = _mk_trace([_span(1, 1, Category.CPU, 0, 10, "main")], backends=["cpu"])
        self._bridge(trace)  # must not raise

    # ── KernelsBridge ────────────────────────────────────────────────────

    def test_kernels_bridge_rows_sorted_by_total_time(self):
        from src.gui.bridge import KernelsBridge
        spans = [_span(1, 1, Category.CPU, i, 100, "hot") for i in range(10)]
        spans += [_span(1, 1, Category.CPU, 2000 + i, 5, "cold") for i in range(2)]
        trace = _mk_trace(spans, backends=["cpu"])
        kb = KernelsBridge(trace, self._theme())
        self.assertEqual(kb.rows[0]["name"], "hot")
        for row in kb.rows:
            self.assertTrue(row["color"].startswith("#"))

    def test_kernels_bridge_table_matches_rows(self):
        from src.gui.bridge import KernelsBridge
        spans = [_span(1, 1, Category.CPU, i, 100, "hot") for i in range(10)]
        spans += [_span(1, 1, Category.CPU, 2000 + i, 5, "cold") for i in range(2)]
        trace = _mk_trace(spans, backends=["cpu"])
        kb = KernelsBridge(trace, self._theme())
        self.assertIsNotNone(kb.table)
        self.assertEqual(kb.table.rows.sourceCount, len(kb.rows))
        kb.table.filters.textFilter = "cold"
        self.assertEqual(kb.table.filters.matchCount, 1)

    # ── CallTreeBridge ───────────────────────────────────────────────────

    def test_call_tree_bridge_empty_without_stacks(self):
        from src.gui.bridge import CallTreeBridge
        trace = _mk_trace([_span(1, 1, Category.CPU, 0, 100, "x")], backends=["cpu"])
        ctb = CallTreeBridge(trace, self._theme())
        # Falls back to temporal containment (from _ct_build), not
        # necessarily empty -- must not crash either way.
        self.assertIsInstance(ctb.roots, list)

    def test_call_tree_bridge_with_stacks_reshapes_nested_dicts(self):
        from src.gui.bridge import CallTreeBridge
        span = SpanEvent(name="work", category=Category.CPU, start_ns=0, duration_ns=100,
                         pid=1, tid=1, tags={}, stack_frames=["main"])
        trace = _mk_trace([span], backends=["cpu"])
        ctb = CallTreeBridge(trace, self._theme())
        self.assertEqual(len(ctb.roots), 1)
        root = ctb.roots[0]
        self.assertEqual(root["name"], "main")
        self.assertTrue(root["color"].startswith("#"))
        self.assertEqual(len(root["children"]), 1)
        self.assertEqual(root["children"][0]["name"], "work")

    def _deep_call_tree_bridge(self):
        from src.gui.bridge import CallTreeBridge
        spans = [
            SpanEvent(name="alpha_fn", category=Category.CPU, start_ns=0, duration_ns=100,
                      pid=1, tid=1, tags={}, stack_frames=["main"]),
            SpanEvent(name="beta_fn", category=Category.CPU, start_ns=200, duration_ns=50,
                      pid=1, tid=1, tags={}, stack_frames=["main"]),
        ]
        trace = _mk_trace(spans, backends=["cpu"])
        return CallTreeBridge(trace, self._theme())

    def test_call_tree_bridge_all_roots_never_changes(self):
        ctb = self._deep_call_tree_bridge()
        before = ctb.allRoots
        ctb.setFilter("alpha")
        self.assertEqual(ctb.allRoots, before)
        self.assertNotEqual(ctb.roots, before)

    def test_call_tree_bridge_filter_keeps_ancestors_of_a_match(self):
        ctb = self._deep_call_tree_bridge()
        ctb.setFilter("alpha_fn")
        # "main" (the ancestor) must survive even though it doesn't match
        # "alpha_fn" itself -- otherwise the match would float with no
        # indication of where it's actually called from.
        self.assertEqual(len(ctb.roots), 1)
        self.assertEqual(ctb.roots[0]["name"], "main")
        names = {c["name"] for c in ctb.roots[0]["children"]}
        self.assertEqual(names, {"alpha_fn"})

    def test_call_tree_bridge_clear_filter_restores_full_tree(self):
        ctb = self._deep_call_tree_bridge()
        full = ctb.roots
        ctb.setFilter("alpha_fn")
        ctb.setFilter("")
        self.assertEqual(ctb.roots, full)

    def test_call_tree_bridge_sort_children_orders_by_key(self):
        ctb = self._deep_call_tree_bridge()
        ctb.sortChildren("totalNs", True)   # descending
        names = [c["name"] for c in ctb.roots[0]["children"]]
        self.assertEqual(names, ["alpha_fn", "beta_fn"])   # 100ns > 50ns
        ctb.sortChildren("totalNs", False)
        names = [c["name"] for c in ctb.roots[0]["children"]]
        self.assertEqual(names, ["beta_fn", "alpha_fn"])

    def test_call_tree_bridge_export_csv_includes_path_column(self):
        import csv
        import tempfile
        ctb = self._deep_call_tree_bridge()
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            self.assertTrue(ctb.exportCsv(path))
            with open(path, newline="") as fh:
                rows = list(csv.reader(fh))
            self.assertIn("Path", rows[0])
            path_col = rows[0].index("Path")
            paths = [r[path_col] for r in rows[1:]]
            self.assertIn("main > alpha_fn", paths)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_inspector_call_tree_relationship_unaffected_by_filter(self):
        from src.gui.bridge import KernelsBridge, RooflineBridge, SourceBridge, CallTreeBridge
        from src.gui.models import TimelineModel
        from src.gui.nav import Selection
        from src.gui.inspector import InspectorBridge
        span = SpanEvent(name="alpha_fn", category=Category.CPU, start_ns=0, duration_ns=100,
                         pid=1, tid=1, tags={}, stack_frames=["main"])
        trace = _mk_trace([span], backends=["cpu"])
        theme = self._theme()
        ctb = CallTreeBridge(trace, theme)
        # Filter that would remove "alpha_fn" from the FILTERED roots.
        ctb.setFilter("something_else_entirely")
        kernels = KernelsBridge(trace, theme)
        roofline = RooflineBridge(trace)
        source = SourceBridge(trace)
        timeline = TimelineModel(trace, theme)
        selection = Selection()
        inspector = InspectorBridge(trace, selection, kernels, ctb, roofline, source, timeline)
        selection.selectFunction("cpu", "alpha_fn")
        relationships = {f["label"]: f for f in inspector.content["relationships"]}
        self.assertEqual(relationships["Call Tree"]["kind"], "measured")
        self.assertEqual(relationships["Call Tree"]["value"], "appears in the call tree")

    # ── FlameGraphBridge ─────────────────────────────────────────────────
    # Trace-sourced data (analysis/flamegraph_tree.py's build_flame_tree(),
    # built on the same _ct_build CallTreeBridge uses).

    def test_flame_graph_bridge_empty_trace(self):
        from src.gui.bridge import FlameGraphBridge
        trace = _mk_trace([])
        fgb = FlameGraphBridge(trace, self._theme())
        self.assertEqual(fgb.totalNs, 0)
        self.assertEqual(fgb.tree["children"], [])

    def test_flame_graph_bridge_tree_shape_and_color(self):
        from src.gui.bridge import FlameGraphBridge
        span = SpanEvent(name="work", category=Category.CPU, start_ns=0, duration_ns=100,
                         pid=1, tid=1, tags={}, stack_frames=["main"])
        trace = _mk_trace([span], backends=["cpu"])
        fgb = FlameGraphBridge(trace, self._theme())
        self.assertEqual(fgb.totalNs, 100)
        tree = fgb.tree
        self.assertEqual(tree["name"], "all")
        self.assertEqual(tree["value"], 100)
        self.assertTrue(tree["color"].startswith("#"))
        main = tree["children"][0]
        self.assertEqual(main["name"], "main")
        self.assertTrue(main["color"].startswith("#"))
        work = main["children"][0]
        self.assertEqual(work["name"], "work")
        self.assertEqual(work["value"], 100)

    def test_flame_graph_bridge_matches_call_tree_bridge_structure(self):
        # Same underlying spans, same _ct_build call -- CallTreeBridge and
        # FlameGraphBridge must agree on names/values, just reshaped
        # differently (totalNs/value vs totalNs-per-node/children).
        from src.gui.bridge import CallTreeBridge, FlameGraphBridge
        spans = [
            SpanEvent(name="leaf_a", category=Category.CPU, start_ns=0, duration_ns=70,
                      pid=1, tid=1, stack_frames=["main"]),
            SpanEvent(name="leaf_b", category=Category.CPU, start_ns=70, duration_ns=30,
                      pid=1, tid=1, stack_frames=["main"]),
        ]
        trace = _mk_trace(spans, backends=["cpu"])
        theme = self._theme()
        ctb = CallTreeBridge(trace, theme)
        fgb = FlameGraphBridge(trace, theme)
        ct_main = ctb.roots[0]
        fg_main = fgb.tree["children"][0]
        self.assertEqual(ct_main["name"], fg_main["name"])
        self.assertEqual(ct_main["totalNs"], fg_main["value"])
        self.assertEqual({c["name"] for c in ct_main["children"]},
                         {c["name"] for c in fg_main["children"]})

    # ── RooflineBridge ───────────────────────────────────────────────────

    def test_roofline_bridge_unavailable_without_gpu_metrics(self):
        from src.gui.bridge import RooflineBridge
        trace = _mk_trace([_span(1, 1, Category.CPU, 0, 100, "x")], backends=["cpu"])
        rb = RooflineBridge(trace)
        self.assertFalse(rb.available)
        self.assertEqual(rb.points, [])

    def test_roofline_bridge_does_not_crash_on_empty_trace(self):
        from src.gui.bridge import RooflineBridge
        RooflineBridge(_mk_trace([]))  # must not raise

    # ── SourceBridge ─────────────────────────────────────────────────────

    def test_source_bridge_lists_kernels_without_disasm(self):
        from src.gui.bridge import SourceBridge
        trace = _mk_trace([_span(1, 1, Category.CPU, 0, 100, "hot_fn")], backends=["cpu"])
        sb = SourceBridge(trace)
        self.assertTrue(any(k["rawName"] == "hot_fn" for k in sb.kernels))
        self.assertFalse(sb.kernels[0]["hasDisasm"])
        self.assertEqual(sb.disasmLines("hot_fn"), [])

    def test_source_bridge_no_disasm_reason_distinguishes_tag_presence(self):
        from src.gui.bridge import SourceBridge
        no_tag = _span(1, 1, Category.CPU, 0, 100, "a")
        with_sym = _span(1, 1, Category.OPENMP, 0, 100, "b", tags={"sym": "main"})
        trace = _mk_trace([no_tag, with_sym], backends=["cpu", "openmp"])
        sb = SourceBridge(trace)
        self.assertIn("Not a missing-tool", sb.noDisasmReason("a"))
        self.assertIn("sym=main", sb.noDisasmReason("b"))

    def test_source_bridge_picks_up_disasm_added_after_construction(self):
        # `hprofiler gui --disasm` collects disassembly in the background,
        # so a function can resolve after the window opens. SourceBridge
        # must poll trace._disasm_version (every 0.5s, like the TUI) rather
        # than build its kernel list once, or it stays stuck on
        # "disassembly still failed".
        from src.gui.bridge import SourceBridge
        from src.disasm.extractor import KernelDisasm
        span = _span(1, 1, Category.OPENMP, 0, 100, "hot_fn", tags={"sym": "hot_fn"})
        trace = _mk_trace([span], backends=["openmp"])
        sb = SourceBridge(trace)
        self.assertFalse(sb.kernels[0]["hasDisasm"])

        received = []
        sb.kernelsChanged.connect(lambda: received.append(True))
        trace.add_disasm(KernelDisasm(name="hot_fn", arch="x86_64", source="objdump", lines=[]))
        sb._poll.timeout.emit()  # simulate one poll tick without waiting 0.5s in the test

        self.assertTrue(received, "kernelsChanged did not fire after add_disasm")
        self.assertTrue(sb.kernels[0]["hasDisasm"])

    def test_source_bridge_poll_is_a_noop_when_disasm_version_unchanged(self):
        # The poll runs every 0.5s for the lifetime of the window; it must
        # not rebuild/re-emit on every tick regardless of whether anything
        # actually changed, or every open Source screen would repaint 2x/
        # sec forever for no reason.
        from src.gui.bridge import SourceBridge
        trace = _mk_trace([_span(1, 1, Category.CPU, 0, 100, "hot_fn")], backends=["cpu"])
        sb = SourceBridge(trace)
        received = []
        sb.kernelsChanged.connect(lambda: received.append(True))
        sb._poll.timeout.emit()
        sb._poll.timeout.emit()
        self.assertEqual(received, [])

    def test_source_bridge_exposes_demangled_call_site_symbol(self):
        # The kernel list shows the span/event label ("omp_barrier"), not
        # the function that was disassembled; `symbol` carries the
        # demangled resolved call site (KernelDisasm.mangled_name).
        from src.gui.bridge import SourceBridge
        from src.disasm.extractor import KernelDisasm
        span = _span(1, 1, Category.SYNC, 0, 100, "omp_barrier", tags={"sym": "caller_fn"})
        trace = _mk_trace([span], backends=["openmp"])
        trace.add_disasm(KernelDisasm(
            name="omp_barrier", arch="x86-64", source="/bin/a.out",
            mangled_name="_Z9caller_fnv", lines=[],
        ))
        sb = SourceBridge(trace)
        row = next(k for k in sb.kernels if k["rawName"] == "omp_barrier")
        self.assertIn("caller_fn", row["symbol"])

    def test_source_bridge_symbol_empty_when_mangled_name_unset(self):
        from src.gui.bridge import SourceBridge
        from src.disasm.extractor import KernelDisasm
        span = _span(1, 1, Category.CPU, 0, 100, "hot_fn")
        trace = _mk_trace([span], backends=["cpu"])
        trace.add_disasm(KernelDisasm(name="hot_fn", arch="x86-64", source="/bin/a.out", lines=[]))
        sb = SourceBridge(trace)
        row = next(k for k in sb.kernels if k["rawName"] == "hot_fn")
        self.assertEqual(row["symbol"], "")

    def test_source_bridge_disasm_lines_carry_source_correlation(self):
        # source_file/source_line come from disasm/source_ann.py, which
        # runs unconditionally after every disasm collection -- the data
        # already existed, it just wasn't threaded through to QML before.
        # sourceChanged marks only the first instruction at a given
        # file:line so the delegate shows the label once, not per line.
        from src.gui.bridge import SourceBridge
        from src.disasm.extractor import KernelDisasm, DisasmLine
        span = _span(1, 1, Category.CPU, 0, 100, "hot_fn")
        trace = _mk_trace([span], backends=["cpu"])
        trace.add_disasm(KernelDisasm(
            name="hot_fn", arch="x86-64", source="/bin/a.out",
            lines=[
                DisasmLine(addr=0x10, mnemonic="push", operands="rbp",
                          source_file="gmx.cpp", source_line=42),
                DisasmLine(addr=0x14, mnemonic="mov", operands="rbp, rsp",
                          source_file="gmx.cpp", source_line=42),
                DisasmLine(addr=0x18, mnemonic="call", operands="0x20",
                          source_file="gmx.cpp", source_line=43),
            ],
        ))
        sb = SourceBridge(trace)
        lines = sb.disasmLines("hot_fn")
        self.assertEqual([ln["sourceLine"] for ln in lines], [42, 42, 43])
        self.assertEqual([ln["sourceChanged"] for ln in lines], [True, False, True])
        self.assertTrue(all(ln["sourceFile"] == "gmx.cpp" for ln in lines))

    def test_source_bridge_disasm_lines_source_fields_empty_without_debug_info(self):
        from src.gui.bridge import SourceBridge
        from src.disasm.extractor import KernelDisasm, DisasmLine
        span = _span(1, 1, Category.CPU, 0, 100, "hot_fn")
        trace = _mk_trace([span], backends=["cpu"])
        trace.add_disasm(KernelDisasm(
            name="hot_fn", arch="x86-64", source="/bin/a.out",
            lines=[DisasmLine(addr=0x10, mnemonic="push", operands="rbp")],
        ))
        sb = SourceBridge(trace)
        lines = sb.disasmLines("hot_fn")
        self.assertEqual(lines[0]["sourceFile"], "")
        self.assertFalse(lines[0]["sourceChanged"])

    def test_source_bridge_instruction_mix_counts_and_percentages(self):
        # The Source screen exposes KernelDisasm.itype_counts()/itype_pcts(),
        # the same instruction mix the TUI's DisasmWidget._show_mix shows.
        from src.gui.bridge import SourceBridge
        from src.disasm.extractor import KernelDisasm, DisasmLine
        from src.disasm.classifier import InsnType
        span = _span(1, 1, Category.CPU, 0, 100, "hot_fn")
        trace = _mk_trace([span], backends=["cpu"])
        lines = (
            [DisasmLine(addr=i, mnemonic="vaddps", operands="ymm0, ymm1, ymm2",
                        itype=InsnType.VEC_SP) for i in range(3)]
            + [DisasmLine(addr=100 + i, mnemonic="mov", operands="rax, [rbx]",
                          itype=InsnType.MEMORY) for i in range(1)]
        )
        trace.add_disasm(KernelDisasm(name="hot_fn", arch="x86-64", source="/bin/a.out", lines=lines))
        sb = SourceBridge(trace)
        mix = sb.instructionMix("hot_fn")
        by_type = {row["type"]: row for row in mix}
        self.assertEqual(by_type["vec_sp"]["count"], 3)
        self.assertEqual(by_type["memory"]["count"], 1)
        self.assertAlmostEqual(by_type["vec_sp"]["pct"], 75.0)
        self.assertAlmostEqual(by_type["memory"]["pct"], 25.0)

    def test_source_bridge_instruction_mix_empty_for_unknown_kernel(self):
        from src.gui.bridge import SourceBridge
        trace = _mk_trace([_span(1, 1, Category.CPU, 0, 100, "hot_fn")], backends=["cpu"])
        sb = SourceBridge(trace)
        self.assertEqual(sb.instructionMix("does_not_exist"), [])

    def test_source_bridge_advisor_hints_flags_low_vectorisation(self):
        # Regression test for the same report: analysis/asm_advisor.py
        # already existed (pure static analysis, no LLM/network call) and
        # was already used by the TUI -- the GUI never called it either.
        from src.gui.bridge import SourceBridge
        from src.disasm.extractor import KernelDisasm, DisasmLine
        from src.disasm.classifier import InsnType
        span = _span(1, 1, Category.CPU, 0, 100, "hot_fn")
        trace = _mk_trace([span], backends=["cpu"])
        # 20 scalar instructions, zero vector -- triggers asm_advisor's
        # "low vectorisation" rule (vec_pct < 10%, n >= 20).
        lines = [DisasmLine(addr=i, mnemonic="add", operands="rax, rbx",
                            itype=InsnType.SCALAR) for i in range(20)]
        trace.add_disasm(KernelDisasm(name="hot_fn", arch="x86-64", source="/bin/a.out", lines=lines))
        sb = SourceBridge(trace)
        hints = sb.advisorHints("hot_fn")
        self.assertTrue(any(h["category"] == "vectorize" for h in hints))
        vec_hint = next(h for h in hints if h["category"] == "vectorize")
        self.assertEqual(vec_hint["severity"], "warn")
        self.assertTrue(vec_hint["color"].startswith("#"))

    def test_source_bridge_advisor_hints_empty_for_unknown_kernel(self):
        from src.gui.bridge import SourceBridge
        trace = _mk_trace([_span(1, 1, Category.CPU, 0, 100, "hot_fn")], backends=["cpu"])
        sb = SourceBridge(trace)
        self.assertEqual(sb.advisorHints("does_not_exist"), [])

    # ── SystemBridge ─────────────────────────────────────────────────────

    def test_system_bridge_exposes_run_info_and_devices(self):
        from src.gui.bridge import SystemBridge
        trace = _mk_trace([_span(1, 1, Category.CPU, 0, 100, "x")],
                          backends=["cuda"], command="gmx_mpi")
        trace.set_devices([_dev()])
        trace.add(CounterEvent(name="process_max_rss_bytes", category=Category.OTHER,
                               timestamp_ns=0, value=2_000_000_000, pid=1))
        sysb = SystemBridge(trace)
        self.assertIn("gmx_mpi", sysb.command)
        self.assertIn("cuda", sysb.backends)
        self.assertEqual(len(sysb.devices), 1)
        self.assertEqual(sysb.devices[0]["name"], "A100")
        self.assertIn("GB", sysb.peakRss)

    def test_system_bridge_no_devices_or_counters_does_not_crash(self):
        from src.gui.bridge import SystemBridge
        SystemBridge(_mk_trace([_span(1, 1, Category.CPU, 0, 10, "x")]))

    def test_system_bridge_device_table_matches_devices(self):
        from src.gui.bridge import SystemBridge
        trace = _mk_trace([_span(1, 1, Category.CPU, 0, 100, "x")], backends=["cuda"])
        trace.set_devices([_dev()])
        sysb = SystemBridge(trace)
        self.assertEqual(sysb.deviceTable.rows.sourceCount, 1)

    def test_system_bridge_metric_table_reports_unavailable_without_counters(self):
        from src.gui.bridge import SystemBridge
        sysb = SystemBridge(_mk_trace([_span(1, 1, Category.CPU, 0, 10, "x")]))
        self.assertEqual(sysb.metricTable.rows.sourceCount, 4)
        rows = [sysb.metricTable._model.rowDict(i) for i in range(4)]
        by_label = {r["label"]: r for r in rows}
        self.assertEqual(by_label["IPC"]["kind"], "unavailable")
        self.assertTrue(by_label["IPC"]["reason"])

    def test_system_bridge_metric_table_reports_measured_with_counters(self):
        from src.gui.bridge import SystemBridge
        trace = _mk_trace([_span(1, 1, Category.CPU, 0, 100, "x")])
        trace.add(CounterEvent(name="ipc", category=Category.OTHER, timestamp_ns=0, value=1.85, pid=1))
        sysb = SystemBridge(trace)
        rows = [sysb.metricTable._model.rowDict(i) for i in range(4)]
        by_label = {r["label"]: r for r in rows}
        self.assertEqual(by_label["IPC"]["kind"], "measured")
        self.assertIn("1.85", by_label["IPC"]["value"])

    # ── ProfileBridge ────────────────────────────────────────────────────

    def test_profile_bridge_gpu_activity_and_breakdown(self):
        from src.gui.bridge import ProfileBridge
        spans = [_span(1, 1, Category.GPU_CUDA, i * 1000, 500, "k",
                       tags={"type": "kernel"}) for i in range(10)]
        spans += [_span(1, 1, Category.CPU, 0, 2000, "cpu_work")]
        trace = _mk_trace(spans, backends=["cuda", "cpu"])
        pb = ProfileBridge(trace, self._theme())
        self.assertEqual(len(pb.gpuActivity), 1)
        self.assertEqual(pb.gpuActivity[0]["label"], "CUDA")
        self.assertTrue(pb.breakdown)
        self.assertTrue(pb.hotspots)

    def test_profile_bridge_insight_shares_bottleneck_analysis(self):
        from src.gui.bridge import ProfileBridge
        spans = [_span(1, 1, Category.GPU_CUDA, i * 1_000_000, 50_000, "k",
                       tags={"type": "kernel"}) for i in range(20)]
        trace = _mk_trace(spans, backends=["cuda"])
        pb = ProfileBridge(trace, self._theme())
        self.assertTrue(any("occupancy" in t["text"].lower() for t in pb.insight))

    def test_profile_bridge_minimal_trace_does_not_crash(self):
        from src.gui.bridge import ProfileBridge
        ProfileBridge(_mk_trace([_span(1, 1, Category.CPU, 0, 10, "x")]), self._theme())

    # ── DashboardBridge: Overview run summary / breakdown / next steps ──

    def test_dashboard_bridge_run_summary_fields(self):
        trace = _mk_trace([_span(1, 1, Category.CPU, 0, 100, "x"), _span(1, 2, Category.CPU, 0, 100, "y")],
                          backends=["cpu"], command="./a.out")
        trace.metadata.hostname = "node01"
        trace.metadata.capture_time_iso = "2026-09-25T10:00:00"
        bridge = self._bridge(trace)
        self.assertEqual(bridge.executable, "./a.out")
        self.assertEqual(bridge.host, "node01")
        self.assertEqual(bridge.captureTime, "2026-09-25T10:00:00")
        self.assertEqual(bridge.processCount, 1)
        self.assertEqual(bridge.threadCount, 2)

    def test_dashboard_bridge_capture_time_empty_when_unset(self):
        # "" (TraceMetadata.capture_time_iso's own default) -- the GUI
        # renders this as "unavailable", not a fabricated time.
        bridge = self._bridge(_mk_trace([_span(1, 1, Category.CPU, 0, 100, "x")]))
        self.assertEqual(bridge.captureTime, "")

    def test_dashboard_bridge_time_breakdown_sums_to_100_pct(self):
        spans = [_span(1, 1, Category.CPU, 0, 100, "cpu_work"),
                _span(1, 1, Category.MPI, 100, 50, "MPI_Send"),
                _span(1, 1, Category.SYNC, 150, 20, "sync"),
                _span(1, 1, Category.MEMORY, 170, 10, "memcpy")]
        trace = _mk_trace(spans, backends=["cpu", "mpi"])
        breakdown = self._bridge(trace).timeBreakdown
        self.assertTrue(breakdown)
        self.assertAlmostEqual(sum(b["pct"] for b in breakdown), 100.0, delta=0.5)
        self.assertTrue(all(b["kind"] == "derived" for b in breakdown))

    def test_dashboard_bridge_profiling_overhead_always_unavailable(self):
        # No overhead-measurement instrumentation exists anywhere in this
        # codebase -- reported honestly, not invented from a proxy number.
        overhead = self._bridge(_mk_trace([_span(1, 1, Category.CPU, 0, 100, "x")])).profilingOverhead
        self.assertEqual(overhead["kind"], "unavailable")
        self.assertTrue(overhead["reason"])

    def test_dashboard_bridge_investigate_next_targets_dominant_kernel(self):
        spans = [_span(1, 1, Category.CPU, i, 100, "dominant_fn") for i in range(20)]
        spans += [_span(1, 1, Category.CPU, 3000 + i, 5, "minor_fn") for i in range(2)]
        actions = self._bridge(_mk_trace(spans, backends=["cpu"])).investigateNext
        self.assertTrue(actions)
        self.assertTrue(any(a["name"] == "dominant_fn" and a["category"] == "cpu" for a in actions))

    def test_dashboard_bridge_top_bottlenecks_reshapes_top_findings(self):
        from src.analysis import dashboard as dash
        spans = [_span(1, 1, Category.GPU_CUDA, i * 1_000_000, 50_000, "k",
                       tags={"type": "kernel"}) for i in range(5)]
        trace = _mk_trace(spans, backends=["cuda"])
        bottlenecks = self._bridge(trace).topBottlenecks
        findings = dash.top_findings(trace)
        self.assertEqual(len(bottlenecks), len(findings))
        for field, (_icon, _severity, title, metric) in zip(bottlenecks, findings):
            self.assertEqual(field["label"], title)
            self.assertEqual(field["value"], metric)
            self.assertEqual(field["kind"], "measured")

    def test_dashboard_bridge_findings_table_matches_top_bottlenecks(self):
        spans = [_span(1, 1, Category.GPU_CUDA, i * 1_000_000, 50_000, "k",
                       tags={"type": "kernel"}) for i in range(5)]
        trace = _mk_trace(spans, backends=["cuda"])
        bridge = self._bridge(trace)
        self.assertEqual(bridge.findingsTable.rows.sourceCount, len(bridge.topBottlenecks))

    # ── InspectorBridge ──────────────────────────────────────────────────
    # Field shape/kind-tagging is what makes "clearly distinguish
    # measured, derived, estimated, and unavailable values" a concrete
    # contract rather than an aspiration -- see src/gui/inspector.py.

    def _inspector(self, trace):
        from src.gui.bridge import KernelsBridge, CallTreeBridge, RooflineBridge, SourceBridge
        from src.gui.models import TimelineModel
        from src.gui.nav import Selection
        from src.gui.inspector import InspectorBridge
        theme = self._theme()
        kernels = KernelsBridge(trace, theme)
        call_tree = CallTreeBridge(trace, theme)
        roofline = RooflineBridge(trace)
        source = SourceBridge(trace)
        timeline = TimelineModel(trace, theme)
        selection = Selection()
        inspector = InspectorBridge(trace, selection, kernels, call_tree, roofline, source, timeline)
        return selection, inspector

    def _gpu_trace(self, native=True):
        from src.core.runner import _parse_record
        from src.core import gpu_activity as ga
        host = ("span:cuda:1:100:1000:50:cudaLaunchKernel:type=launch,op=kernel,stream=11,"
                "side=cpu,rt=cuda,timing=host,lid=1,sid=4294967297,corr=101")
        if native:
            dev = ("span:cuda:1:0:1200:4000:_Z4bigv:type=kernel,side=gpu,rt=cuda,op=kernel,"
                   "timing=device,src=cupti,corr=101,dev=0,ctx=1,nstream=7")
        else:
            dev = ("span:cuda:1:100:1000:4000:_Z4bigv:type=kernel,op=kernel,stream=11,side=gpu,"
                   "rt=cuda,lid=1,timing=proxy_event")
        trace = _mk_trace([_parse_record(host), _parse_record(dev)], backends=["cuda"])
        ga.assemble(trace)
        return trace

    def test_inspector_gpu_device_span_provenance(self):
        from src.analysis import dashboard as dash
        selection, inspector = self._inspector(self._gpu_trace())
        selection.selectFunction("cuda", "_Z4bigv")
        metrics = {f["label"]: f for f in inspector.content["metrics"]}
        self.assertEqual(metrics["Device timing"]["kind"], "measured")
        self.assertIn("cupti", metrics["Device timing"]["value"])
        self.assertEqual(metrics["Host call (median)"]["value"], dash.fmt_ns(50))
        self.assertEqual(metrics["Queued before start (median)"]["kind"], "derived")
        self.assertEqual(metrics["Queued before start (median)"]["value"], dash.fmt_ns(150))
        context = {f["label"]: f for f in inspector.content["context"]}
        self.assertEqual(context["Runs on"]["value"], "pid 1 / stream 11")
        rel = {f["label"]: f for f in inspector.content["relationships"]}
        self.assertEqual(rel["Submitted by"]["value"], "1 of 1 correlated to a host call")

    def test_inspector_gpu_host_call_and_proxy(self):
        selection, inspector = self._inspector(self._gpu_trace())
        selection.selectFunction("cuda", "cudaLaunchKernel")
        context = {f["label"]: f for f in inspector.content["context"]}
        self.assertEqual(context["Runs on"]["value"], "pid 1 / tid 100")
        rel = {f["label"]: f for f in inspector.content["relationships"]}
        self.assertEqual(rel["Device work"]["value"], "1 device span(s) from 1 call(s)")

        selection, inspector = self._inspector(self._gpu_trace(native=False))
        selection.selectFunction("cuda", "_Z4bigv")
        metrics = {f["label"]: f for f in inspector.content["metrics"]}
        self.assertEqual(metrics["Device timing"]["kind"], "estimated")
        self.assertIn("submission time", metrics["Device timing"]["value"])
        self.assertNotIn("Queued before start (median)", metrics)

    def test_inspector_empty_selection_yields_empty_content(self):
        _selection, inspector = self._inspector(_mk_trace([_span(1, 1, Category.CPU, 0, 100, "x")]))
        for section in ("summary", "context", "metrics", "relationships", "recommendations"):
            self.assertEqual(inspector.content[section], [])

    def test_inspector_matched_selection_has_measured_and_derived_fields(self):
        spans = [_span(1, 1, Category.CPU, i * 100, 50, "hot_fn") for i in range(5)]
        selection, inspector = self._inspector(_mk_trace(spans, backends=["cpu"]))

        selection.selectFunction("cpu", "hot_fn")
        content = inspector.content

        self.assertTrue(content["summary"])
        kinds = {f["label"]: f["kind"] for f in content["summary"]}
        self.assertEqual(kinds["Name"], "measured")
        self.assertEqual(kinds["Share of total"], "derived")
        self.assertTrue(content["metrics"])
        self.assertTrue(all(f["kind"] in ("measured", "derived") for f in content["metrics"]))

    def test_inspector_unmatched_selection_reports_unavailable_with_reasons(self):
        selection, inspector = self._inspector(_mk_trace([_span(1, 1, Category.CPU, 0, 100, "x")]))

        selection.selectFunction("other", "nothing_matches_this")
        content = inspector.content

        summary_kinds = {f["label"]: f["kind"] for f in content["summary"]}
        self.assertEqual(summary_kinds["Total time"], "unavailable")
        for section in ("context", "relationships", "recommendations"):
            for field in content[section]:
                if field["kind"] == "unavailable":
                    self.assertTrue(field["reason"], f"{section}.{field['label']} unavailable with no reason")

    def test_inspector_every_field_has_the_full_field_shape(self):
        spans = [_span(1, 1, Category.CPU, i * 100, 50, "hot_fn") for i in range(3)]
        selection, inspector = self._inspector(_mk_trace(spans, backends=["cpu"]))
        selection.selectFunction("cpu", "hot_fn")

        for section_fields in inspector.content.values():
            for field in section_fields:
                self.assertEqual(set(field.keys()), {"label", "value", "kind", "reason"})
                self.assertIn(field["kind"], ("measured", "derived", "estimated", "unavailable"))

    def test_inspector_copy_and_export(self):
        import json
        import tempfile
        selection, inspector = self._inspector(_mk_trace(
            [_span(1, 1, Category.CPU, 0, 100, "hot_fn")], backends=["cpu"]))
        selection.selectFunction("cpu", "hot_fn")

        inspector.copyToClipboard(json.dumps(inspector.content))  # must not raise

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            self.assertTrue(inspector.exportTo(path))
            with open(path) as fh:
                exported = json.load(fh)
            self.assertEqual(exported, inspector.content)
        finally:
            Path(path).unlink(missing_ok=True)


@unittest.skipUnless(_PYSIDE6_AVAILABLE, "PySide6 not installed")
class TestDataTableModels(unittest.TestCase):
    """src/gui/tablemodel.py -- the shared table infrastructure (real Qt
    model/view: QAbstractListModel + QSortFilterProxyModel). Built as a
    reference implementation against KERNEL_COLUMNS, but this module
    itself is table-agnostic."""

    _app = None

    @classmethod
    def setUpClass(cls):
        cls._app = QGuiApplication.instance() or QGuiApplication([])

    def _rows(self):
        return [
            {"name": "matmul", "rawName": "matmul", "category": "cuda", "color": "#f87171",
             "count": 100, "totalNs": 500_000, "avgNs": 5_000, "minNs": 1_000, "maxNs": 9_000, "sharePct": 60.0},
            {"name": "reduce", "rawName": "reduce", "category": "cuda", "color": "#f87171",
             "count": 20, "totalNs": 200_000, "avgNs": 10_000, "minNs": 5_000, "maxNs": 15_000, "sharePct": 24.0},
            {"name": "copy_9x_kernel", "rawName": "copy_9x_kernel", "category": "memory", "color": "#a78bfa",
             "count": 5, "totalNs": 133_333, "avgNs": 26_666, "minNs": 10_000, "maxNs": 40_000, "sharePct": 16.0},
        ]

    def _bundle(self):
        from src.gui.tablemodel import TableBundle
        from src.gui.columns import KERNEL_COLUMNS
        return TableBundle(self._rows(), KERNEL_COLUMNS)

    def test_role_names_cover_every_column_plus_row(self):
        bundle = self._bundle()
        names = {v.decode() if isinstance(v, (bytes, bytearray)) else v
                  for v in bundle._model.roleNames().values()}
        for col in bundle._model.columnSpecs():
            self.assertIn(col.key, names)
        self.assertIn("row", names)

    def test_numeric_sort_is_correct_not_lexicographic(self):
        # A naive string/JS sort would order "1000" < "200" < "9000"
        # lexicographically if a numeric field were ever stringified;
        # TableFilterProxy's
        # lessThan() must sort by real numeric value regardless.
        bundle = self._bundle()
        bundle.filters.toggleSort("count")   # 100, 20, 5 -> ascending: 5, 20, 100
        names = []
        for r in range(bundle.rows.rowCount()):
            idx = bundle.rows.index(r, 0)
            names.append(bundle.rows.data(idx, bundle._model.roleForKey("name")))
        self.assertEqual(names, ["copy_9x_kernel", "reduce", "matmul"])

    def test_toggle_sort_same_key_flips_direction(self):
        bundle = self._bundle()
        bundle.filters.toggleSort("count")
        self.assertFalse(bundle.filters.sortDescending)
        bundle.filters.toggleSort("count")
        self.assertTrue(bundle.filters.sortDescending)

    def test_text_filter_matches_name(self):
        bundle = self._bundle()
        bundle.filters.textFilter = "copy"
        self.assertEqual(bundle.filters.matchCount, 1)
        bundle.filters.clearFilters()
        self.assertEqual(bundle.filters.matchCount, 3)

    def test_min_max_value_filters(self):
        bundle = self._bundle()
        bundle.filters.setMin("totalNs", 150_000)
        self.assertEqual(bundle.filters.matchCount, 2)
        bundle.filters.clearMin("totalNs")
        bundle.filters.setMax("totalNs", 150_000)
        self.assertEqual(bundle.filters.matchCount, 1)

    def test_category_filter(self):
        bundle = self._bundle()
        bundle.filters.setCategoryEnabled("memory", True)
        self.assertEqual(bundle.filters.matchCount, 1)
        bundle.filters.setCategoryEnabled("memory", False)
        self.assertEqual(bundle.filters.matchCount, 3)

    def test_percent_mode_toggle_does_not_change_stored_role_value(self):
        # percentMode is a DISPLAY concern (DataTableCell.qml computes the
        # percentage at render time) -- the underlying role value must
        # stay the raw number so sorting/filtering/export are unaffected
        # by whether percent mode happens to be on.
        bundle = self._bundle()
        idx = bundle.rows.index(0, 0)
        role = bundle._model.roleForKey("totalNs")
        before = bundle.rows.data(idx, role)
        bundle.config.togglePercentMode()
        after = bundle.rows.data(idx, role)
        self.assertEqual(before, after)

    def test_table_config_width_and_visibility_round_trip(self):
        bundle = self._bundle()
        bundle.config.setColumnWidth("name", 300)
        bundle.config.setColumnVisible("minNs", True)
        cols = {c["key"]: c for c in bundle.config.columns}
        self.assertEqual(cols["name"]["width"], 300)
        self.assertTrue(cols["minNs"]["visible"])
        bundle.config.resetLayout()
        cols = {c["key"]: c for c in bundle.config.columns}
        self.assertNotEqual(cols["name"]["width"], 300)
        self.assertFalse(cols["minNs"]["visible"])

    def test_move_column_reorders(self):
        bundle = self._bundle()
        order_before = [c["key"] for c in bundle.config.columns]
        bundle.config.moveColumn(0, len(order_before) - 1)
        order_after = [c["key"] for c in bundle.config.columns]
        self.assertNotEqual(order_before, order_after)
        self.assertEqual(order_after[-1], order_before[0])

    def test_bar_maxima_computed_from_full_row_set(self):
        bundle = self._bundle()
        self.assertEqual(bundle.barMaxima["totalNs"], 500_000)
        self.assertEqual(bundle.barMaxima["sharePct"], 60.0)

    def test_export_csv_writes_visible_filtered_sorted_rows(self):
        import csv
        import tempfile
        bundle = self._bundle()
        bundle.config.setColumnVisible("minNs", False)
        bundle.filters.setMin("totalNs", 150_000)
        bundle.filters.toggleSort("totalNs")
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            self.assertTrue(bundle.exportCsv(path))
            with open(path, newline="") as fh:
                reader = list(csv.reader(fh))
            header, rows = reader[0], reader[1:]
            self.assertNotIn("Min", header)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0][0], "reduce")   # ascending totalNs first
            self.assertEqual(rows[1][0], "matmul")
        finally:
            Path(path).unlink(missing_ok=True)

    def test_copy_row_and_copy_all_do_not_raise(self):
        bundle = self._bundle()
        bundle.copyRow(0)   # clipboard access under offscreen QPA -- must not raise
        bundle.copyAll()

    def test_sort_and_filter_never_reset_the_source_model(self):
        # The literal "avoid unnecessary full-table rebuilding" requirement:
        # sorting/filtering go through QSortFilterProxyModel's incremental
        # sort()/invalidateFilter(), never beginResetModel() on the SOURCE
        # DictListTableModel.
        bundle = self._bundle()
        resets = []
        bundle._model.modelAboutToBeReset.connect(lambda: resets.append(1))
        bundle.filters.toggleSort("count")
        bundle.filters.toggleSort("totalNs")
        bundle.filters.textFilter = "cuda"
        bundle.filters.setMin("count", 1)
        bundle.filters.clearFilters()
        bundle.config.setColumnVisible("minNs", True)
        self.assertEqual(resets, [])

    def test_set_rows_does_reset_the_source_model(self):
        # The one legitimate case -- a genuine data replacement, distinct
        # from sort/filter above.
        bundle = self._bundle()
        resets = []
        bundle._model.modelAboutToBeReset.connect(lambda: resets.append(1))
        bundle._model.setRows(self._rows())
        self.assertEqual(resets, [1])

    def test_row_as_text_reflects_visible_columns_only(self):
        bundle = self._bundle()
        bundle.config.setColumnVisible("category", False)
        text = bundle.rowAsText(0)
        self.assertNotIn("cuda", text.split("\t"))

    def test_format_bridge_matches_dashboard_fmt_helpers(self):
        from src.gui.tablemodel import FormatBridge
        from src.analysis import dashboard as dash
        fmt = FormatBridge()
        self.assertEqual(fmt.formatNumber("time_ns", 500_000), dash.fmt_ns(500_000))
        self.assertEqual(fmt.formatNumber("count", 1234), dash.fmt_count(1234))
        self.assertEqual(fmt.formatNumber("pct", 12.3), dash.fmt_pct(12.3))
        self.assertEqual(fmt.signedNs(-500_000), dash.fmt_signed_ns(-500_000))
        self.assertEqual(fmt.signedPct(12.3), dash.fmt_signed_pct(12.3))


def DashboardBridgeBuckets() -> int:
    from src.gui.bridge import DashboardBridge
    return DashboardBridge._TIMELINE_BUCKETS


@unittest.skipUnless(_PYSIDE6_AVAILABLE, "PySide6 not installed")
class TestComparisonBridge(unittest.TestCase):
    _app = None

    @classmethod
    def setUpClass(cls):
        cls._app = QGuiApplication.instance() or QGuiApplication([])

    def _theme(self):
        from src.gui.theme import Theme
        return Theme(dark=True)

    def _bridge(self, trace_a, trace_b):
        """trace_a = baseline (BEFORE), trace_b = candidate (AFTER): the
        GUI's opened trace is the candidate, --compare names the baseline."""
        from src.gui.comparison import ComparisonBridge
        if trace_b is None:
            return ComparisonBridge(trace_a, None, self._theme())
        return ComparisonBridge(trace_b, trace_a, self._theme())

    def test_unavailable_when_no_comparison_trace(self):
        trace_a = _mk_trace([_span(1, 1, Category.GPU_CUDA, 0, 1_000_000, "k")])
        bridge = self._bridge(trace_a, None)
        self.assertFalse(bridge.available)
        self.assertEqual(bridge.table.rows.rowCount(), 0)
        self.assertEqual(bridge.bucketDeltas, [])
        self.assertEqual(bridge.topImprovements, [])
        self.assertEqual(bridge.topRegressions, [])
        # The opened trace is the candidate: without --compare only the
        # BASELINE side is unavailable.
        for field in bridge.baselineFields:
            self.assertEqual(field["kind"], "unavailable")
            self.assertTrue(field["reason"])
        for field in bridge.comparisonFields:
            self.assertEqual(field["kind"], "measured")
        self.assertFalse(bridge.causalAvailable)
        self.assertEqual(bridge.contributors, [])

    def test_export_methods_fail_cleanly_when_unavailable(self):
        trace_a = _mk_trace([_span(1, 1, Category.GPU_CUDA, 0, 1_000_000, "k")])
        bridge = self._bridge(trace_a, None)
        self.assertFalse(bridge.exportReport("/tmp/should_not_be_written.json"))
        self.assertFalse(bridge.exportCsv("/tmp/should_not_be_written.csv"))

    def test_real_comparison_reports_statuses_and_match_kinds(self):
        trace_a = _mk_trace([
            _span(1, 1, Category.GPU_CUDA, 0, 10_000_000, "matmul", tags={"type": "kernel"}),
            _span(1, 1, Category.GPU_CUDA, 0, 5_000_000, "old_only", tags={"type": "kernel"}),
        ])
        trace_b = _mk_trace([
            _span(1, 1, Category.GPU_CUDA, 0, 20_000_000, "matmul", tags={"type": "kernel"}),
            _span(1, 1, Category.GPU_CUDA, 0, 3_000_000, "new_only", tags={"type": "kernel"}),
        ])
        bridge = self._bridge(trace_a, trace_b)
        self.assertTrue(bridge.available)
        # Read directly from the underlying rows the bridge computed
        # (the same data table.rows' proxy wraps), matching how other
        # bridges' tests in this file verify table CONTENT rather than
        # poking at TableFilterProxy internals.
        by_name = {r["name"]: r for r in bridge._rows}
        self.assertEqual(by_name["matmul"]["status"], "regressed")
        self.assertEqual(by_name["matmul"]["matchKind"], "exact")
        self.assertEqual(by_name["old_only"]["status"], "removed")
        self.assertEqual(by_name["new_only"]["status"], "new")
        self.assertEqual(bridge.newCount, 1)
        self.assertEqual(bridge.removedCount, 1)

    def _pair(self):
        trace_a = _mk_trace([
            _span(1, 1, Category.GPU_CUDA, 0, 10_000_000, "matmul", tags={"type": "kernel"}),
            _span(1, 1, Category.GPU_CUDA, 0, 5_000_000, "old_only", tags={"type": "kernel"}),
        ])
        trace_b = _mk_trace([
            _span(1, 1, Category.GPU_CUDA, 0, 10_800_000, "matmul", tags={"type": "kernel"}),
            _span(1, 1, Category.GPU_CUDA, 0, 3_000_000, "new_only", tags={"type": "kernel"}),
        ])
        return self._bridge(trace_a, trace_b)

    def test_threshold_change_updates_every_derived_summary(self):
        # +8% / +0.8ms on matmul: unchanged under the default 5%/1ms
        # floor, regressed once the floor drops. The table, topRegressions,
        # bucketDeltas and the export must all switch verdict together.
        import json, tempfile, os
        bridge = self._pair()
        self.assertNotIn("matmul", [r["name"] for r in bridge.topRegressions])
        bridge.setChangeThresholds(100_000.0, 1.0)
        self.assertEqual({r["name"]: r["status"] for r in bridge._rows}["matmul"], "regressed")
        self.assertIn("matmul", [r["name"] for r in bridge.topRegressions])
        fd, path = tempfile.mkstemp(suffix=".json"); os.close(fd)
        try:
            self.assertTrue(bridge.exportReport(path))
            exported = {r["name"]: r["status"] for r in json.loads(Path(path).read_text())["aggregates"]}
        finally:
            os.unlink(path)
        self.assertEqual(exported["matmul"], "regressed")
        comp = {b["bucket"]: b["status"] for b in bridge.bucketDeltas}
        from src.analysis import compare as cmp
        self.assertEqual(comp["Computation"], cmp.classify(15_000_000.0, 13_800_000.0,
                                                            noise_pct=1.0, noise_ns=100_000.0)[0])

    def _mpi_pair(self):
        from tests import compare_scenarios as S
        return self._bridge(S.mpi_wait(False), S.mpi_wait(True))

    def test_causal_layer_ranks_contributors_and_explains_them(self):
        b = self._mpi_pair()
        self.assertTrue(b.causalAvailable)
        v = b.verdict
        self.assertEqual(v["method"], "phase-aligned")
        self.assertEqual(v["wallStatus"], "regressed")
        self.assertIn("confidence", b.verdict["confidenceText"] + " confidence")
        top = b.contributors[0]
        self.assertEqual((top["label"], top["cause"]), ("compute", "increased_work"))
        self.assertTrue(any(r["label"] == "MPI_Recv" and r["propagated"] for r in b.propagated))
        d = b.contributorDetail(top["id"])
        self.assertEqual(d["rawName"], "compute")
        self.assertTrue(d["hasTimeline"])
        self.assertLess(d["timelineStartNs"], d["timelineEndNs"])
        kinds = {r["label"]: r["kind"] for r in d["rows"]}
        self.assertEqual(kinds["self time"], "measured")
        self.assertEqual(kinds["critical-path time"], "derived")
        self.assertEqual(b.contributorDetail(10**6), {})

    def test_phase_selection_filters_contributors_and_critical_path(self):
        b = self._mpi_pair()
        self.assertGreater(len(b.phasePairs), 1)
        self.assertEqual(b.selectedPhase, -1)
        b.selectPhase(1)
        self.assertEqual(b.selectedPhase, 1)
        self.assertEqual([c["label"] for c in b.contributors], ["compute"])
        self.assertTrue(b.criticalAfter)
        self.assertAlmostEqual(sum(c["frac"] for c in b.criticalAfter), 1.0, places=6)
        r = b.phaseRange(1)
        self.assertLess(r["startNs"], r["endNs"])
        b.selectPhase(999)
        self.assertEqual(b.selectedPhase, -1)

    def test_export_includes_causal_report_and_thresholds_recompute_it(self):
        import json, tempfile, os
        b = self._mpi_pair()
        fd, path = tempfile.mkstemp(suffix=".json"); os.close(fd)
        try:
            self.assertTrue(b.exportReport(path))
            data = json.loads(Path(path).read_text())
        finally:
            os.unlink(path)
        self.assertEqual(data["causal"]["contributors"][0]["label"], "compute")
        self.assertIn("aggregates", data)
        b.setChangeThresholds(1e15, 5.0)
        self.assertEqual(b.contributors, [])

    def test_missing_side_is_nan_not_zero(self):
        import math
        bridge = self._pair()
        by_name = {r["name"]: r for r in bridge._rows}
        self.assertTrue(math.isnan(by_name["new_only"]["baselineNs"]))
        self.assertTrue(math.isnan(by_name["new_only"]["deltaPct"]))
        self.assertTrue(math.isnan(by_name["old_only"]["comparisonNs"]))
        from src.gui.tablemodel import FormatBridge
        fmt = FormatBridge()
        self.assertEqual(fmt.formatNumber("time_ns", by_name["new_only"]["baselineNs"]), "—")
        self.assertEqual(fmt.formatNumber("pct", by_name["new_only"]["deltaPct"]), "—")

    def test_missing_values_sort_consistently_and_fail_range_filters(self):
        bridge = self._pair()
        proxy = bridge.table.rows
        proxy.toggleSort("baselineNs")  # ascending: missing first
        names = [proxy.data(proxy.index(i, 0), proxy.sourceModel().roleForKey("name"))
                 for i in range(proxy.rowCount())]
        self.assertEqual(names[0], "new_only")
        proxy.setMin("baselineNs", 1.0)
        self.assertNotIn("new_only", [proxy.data(proxy.index(i, 0), proxy.sourceModel().roleForKey("name"))
                                      for i in range(proxy.rowCount())])

    def test_noise_floor_note_present_and_not_a_statistical_claim(self):
        trace_a = _mk_trace([_span(1, 1, Category.GPU_CUDA, 0, 1_000_000, "k")])
        trace_b = _mk_trace([_span(1, 1, Category.GPU_CUDA, 0, 1_000_000, "k")])
        bridge = self._bridge(trace_a, trace_b)
        note = bridge.noiseFloor["note"]
        self.assertIn("not a statistical", note.lower())
        self.assertEqual(bridge.noiseFloor["pct"], bridge.noiseFloor["pct"])   # present, no KeyError

    def test_set_change_thresholds_reclassifies_without_rematching(self):
        trace_a = _mk_trace([_span(1, 1, Category.GPU_CUDA, 0, 1_000_000, "k")])
        trace_b = _mk_trace([_span(1, 1, Category.GPU_CUDA, 0, 1_060_000, "k")])   # +6%, +60us
        bridge = self._bridge(trace_a, trace_b)
        self.assertEqual(bridge._rows[0]["status"], "unchanged")   # 60us < 1ms floor
        bridge.setChangeThresholds(1000.0, 1.0)   # 1us / 1% floor -- now clears both
        self.assertEqual(bridge._rows[0]["status"], "regressed")

    def test_export_report_round_trips(self):
        import json
        import tempfile
        trace_a = _mk_trace([_span(1, 1, Category.GPU_CUDA, 0, 10_000_000, "k")])
        trace_b = _mk_trace([_span(1, 1, Category.GPU_CUDA, 0, 20_000_000, "k")])
        bridge = self._bridge(trace_a, trace_b)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            self.assertTrue(bridge.exportReport(path))
            with open(path) as fh:
                data = json.load(fh)
            self.assertIn("aggregates", data)
            self.assertIn("noiseFloor", data)
            self.assertEqual(data["aggregates"][0]["name"], "k")
        finally:
            Path(path).unlink(missing_ok=True)

    def test_export_csv_round_trips(self):
        import csv
        import tempfile
        trace_a = _mk_trace([_span(1, 1, Category.GPU_CUDA, 0, 10_000_000, "k")])
        trace_b = _mk_trace([_span(1, 1, Category.GPU_CUDA, 0, 20_000_000, "k")])
        bridge = self._bridge(trace_a, trace_b)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            self.assertTrue(bridge.exportCsv(path))
            with open(path, newline="") as fh:
                rows = list(csv.reader(fh))
            self.assertEqual(rows[1][1], "k")
        finally:
            Path(path).unlink(missing_ok=True)

    def test_coverage_strips_are_independently_normalized(self):
        trace_a = _mk_trace([_span(1, 1, Category.GPU_CUDA, 0, 1_000_000, "k")])
        trace_b = _mk_trace([_span(1, 1, Category.GPU_CUDA, 5_000_000, 1_000_000, "k")])
        bridge = self._bridge(trace_a, trace_b)
        self.assertEqual(len(bridge.baselineCoverage), 60)
        self.assertEqual(len(bridge.comparisonCoverage), 60)

    def test_dashboard_bridge_without_comparison_param_is_unchanged(self):
        # Without comparison=, DashboardBridge behaves exactly as a plain
        # single-profile Overview.
        from src.gui.bridge import DashboardBridge
        trace = _mk_trace([_span(1, 1, Category.GPU_CUDA, 0, 1_000_000, "k")])
        bridge = DashboardBridge(trace, self._theme())
        self.assertFalse(bridge.hasComparison)
        self.assertEqual(bridge.comparisonTopChanges, [])

    def test_dashboard_bridge_with_comparison_exposes_top_changes(self):
        from src.gui.bridge import DashboardBridge
        trace_a = _mk_trace([_span(1, 1, Category.GPU_CUDA, 0, 10_000_000, "k")])
        trace_b = _mk_trace([_span(1, 1, Category.GPU_CUDA, 0, 20_000_000, "k")])
        comparison = self._bridge(trace_a, trace_b)
        bridge = DashboardBridge(trace_a, self._theme(), comparison=comparison)
        self.assertTrue(bridge.hasComparison)
        self.assertEqual(len(bridge.comparisonTopChanges), 1)
        self.assertEqual(bridge.comparisonTopChanges[0]["name"], "k")




class TestNavFocusRange(unittest.TestCase):
    def test_focus_requests_are_serialized_and_select_the_range(self):
        app = QGuiApplication.instance() or QGuiApplication([])  # noqa: F841
        from src.gui.nav import Selection
        nav = Selection()
        fired = []
        nav.focusRangeChanged.connect(lambda: fired.append(dict(nav.focusRange)))
        self.assertEqual(nav.focusRange, {})
        nav.focusTimeRange(10.0, 20.0)
        nav.focusTimeRange(10.0, 20.0)       # the same range again is a new request
        self.assertEqual([f["serial"] for f in fired], [1, 2])
        self.assertEqual(nav.selectedTimeRange, {"startNs": 10.0, "endNs": 20.0})


if __name__ == "__main__":
    unittest.main()
