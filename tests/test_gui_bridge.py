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


def DashboardBridgeBuckets() -> int:
    from src.gui.bridge import DashboardBridge
    return DashboardBridge._TIMELINE_BUCKETS


if __name__ == "__main__":
    unittest.main()
