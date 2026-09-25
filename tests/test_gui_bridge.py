"""
Tests for src/gui/theme.py and src/gui/bridge.py -- the Qt/QML GUI's
Python-side data layer. Skipped entirely if PySide6 isn't installed
(it's an optional dependency, `pip install hprofiler[gui]`), matching
this project's existing pattern for optional-tool-dependent tests (e.g.
test_gomp_hook.py skipping when gcc isn't available).

Runs headless (QT_QPA_PLATFORM=offscreen) -- these tests check computed
Property values, not actual rendering (see the manual screenshot-based
verification in project_qml_gui memory for that level of check), so a
real display is not needed; a QGuiApplication instance IS needed since
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
        # radiusPanel matches the value that already dominated every
        # panel block across the GUI before this token existed.
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
        # mpi/memory/jit/nvtx were changed after a quantitative
        # deuteranopia/protanopia/tritanopia simulation found them
        # confusable with other categories (worst: mpi/memory confusable
        # under both common red-green CVD forms). Exact values pinned so
        # a future edit can't silently drift back to the old, confusable
        # ones -- the other 8 categories are deliberately NOT asserted
        # here since they were untouched by this fix.
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
        # The exact regression this fix closes: mpi's dark value used to
        # be #60a5fa and its light value #2563eb -- distinct at the time,
        # but #2563eb is mpi's NEW dark value, so a naive fix that only
        # changed the dark side would have made light-mode mpi equal the
        # OLD dark-mode mpi, not a real fix. Confirms both sides moved.
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

    # ── FlameGraphBridge ─────────────────────────────────────────────────
    # Moved here (from the now-removed standalone `hprofiler flamegraph
    # --gui` popup's own tests/test_flamegraph_gui.py) since the bridge
    # itself moved into bridge.py alongside CallTreeBridge -- same trace-
    # sourced data now (analysis/flamegraph_tree.py's build_flame_tree(),
    # itself built on the same _ct_build CallTreeBridge uses), not a
    # folded-stacks-text constructor argument the way the popup's version was.

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
        # Regression test for a real user report: `hprofiler gui --disasm`
        # started background disassembly collection correctly, but
        # SourceBridge built its kernel list ONCE at construction time
        # (a constant Property) and never looked again -- so a function
        # that resolved a second later than the window opening (the
        # common case: collection takes a moment) stayed stuck showing
        # "disassembly still failed" forever, even though the exact same
        # trace's TUI (which polls trace._disasm_version every 0.5s)
        # showed it correctly. SourceBridge now polls the same way.
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
        # Regression test for a real user question: the kernel list shows
        # the span/event label ("omp_barrier"), not the real function that
        # was disassembled -- with no way to tell what code they're
        # actually looking at. `symbol` now carries the demangled
        # resolved call site (KernelDisasm.mangled_name).
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
        # Regression test: KernelDisasm.itype_counts()/itype_pcts()
        # already existed and were already used by the TUI's
        # DisasmWidget._show_mix -- the GUI's Source screen never called
        # them at all, so a real user reported "no vector/memory
        # instruction counts" with no way to see them.
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

    # ── DashboardBridge: Overview redesign (cross-tab-navigation round) ──

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


def DashboardBridgeBuckets() -> int:
    from src.gui.bridge import DashboardBridge
    return DashboardBridge._TIMELINE_BUCKETS


if __name__ == "__main__":
    unittest.main()
