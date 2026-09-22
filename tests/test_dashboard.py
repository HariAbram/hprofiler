"""
Tests for the redesigned TUI's Overview dashboard (DashboardWidget),
Roofline tab (RooflineWidget), and the app-chrome changes that went with
them (TopBar/BottomBar, numbered/conditional tabs, digit-key jump) --
src/ui/app.py. Built when the Timeline/Hotspots/System/Profile/Call-Tree
tabs were restyled into a card-based dashboard layout at the user's
request, modelled on a reference screenshot.

Also covers two real bugs the redesign surfaced along the way:
  - _bottleneck_analysis was a dead import (output.summary never defined
    it), so the Profile tab's "Insight" section silently rendered
    nothing; it's now a real, local, tested implementation.
  - Two static hint strings (HotspotsWidget's "[s] cycle sort..." and
    CallTreeWidget's "[u] collapse all...") relied on UNESCAPED brackets
    that collide with Rich markup syntax ([s]=strikethrough, [u]=
    underline, anything else bracketed is silently eaten as an unknown
    style tag) -- confirmed visually via an SVG screenshot before fixing.
"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.text import Text
from textual.app import App, ComposeResult

from src.core.trace import Trace, TraceMetadata
from src.core.events import SpanEvent, CounterEvent, Category
from src.analysis.device import DevicePeak
from src.analysis.roofline import KernelMetrics
from src.ui.app import (
    ProfilerApp, DashboardWidget, RooflineWidget, HotspotsWidget, CallTreeWidget,
    SystemWidget, TopBar, BottomBar,
    _diagnose, _top_findings, _bottleneck_analysis, _mini_row, _source_snippet,
    _has_roofline_data, _trace_wall_ns,
)


def _span(pid, tid, cat, start_ns, dur_ns, name, tags=None):
    return SpanEvent(name=name, category=cat, start_ns=start_ns, duration_ns=dur_ns,
                     pid=pid, tid=tid, tags=dict(tags or {}))


def _mk_trace(spans, backends=None, command="./a.out"):
    meta = TraceMetadata(command=command, args=[], backends_used=backends or [])
    t = Trace(meta)
    for s in spans:
        t.add(s)
    return t


def _gpu_starved_trace() -> Trace:
    """50us kernel every 1ms -- launch_gap_pct should land well above the
    30% "Low GPU occupancy" threshold."""
    spans = [
        _span(1, 1, Category.GPU_CUDA, i * 1_000_000, 50_000, "kern", tags={"type": "kernel"})
        for i in range(20)
    ]
    return _mk_trace(spans, backends=["cuda"])


def _imbalanced_mpi_trace() -> Trace:
    """pid 1 does 10x the useful CPU work of pids 2/3 -- load_balance
    should land well under the 0.85 "Load imbalance" threshold."""
    spans = [_span(1, 1, Category.CPU, 0, 900_000, "busy")]
    for pid in (2, 3):
        spans.append(_span(pid, pid, Category.CPU, 0, 90_000, "busy"))
    return _mk_trace(spans, backends=["mpi", "cpu"])


class _HostApp(App):
    """Minimal Textual host for mounting one widget under test."""

    def __init__(self, widget, **kwargs):
        super().__init__(**kwargs)
        self._widget = widget

    def compose(self) -> ComposeResult:
        yield self._widget


# ── _bottleneck_analysis ────────────────────────────────────────────────────

class TestBottleneckAnalysis(unittest.TestCase):
    def test_returns_real_tips_for_a_gpu_starved_trace(self):
        # Regression test for the dead `from ..output.summary import
        # _bottleneck_analysis` import: that function was never defined
        # there, so this call site always silently produced an empty list.
        tips = _bottleneck_analysis(_gpu_starved_trace(), {})
        self.assertTrue(any("occupancy" in t.lower() for t in tips))

    def test_tips_use_plain_ascii_icons_not_emoji(self):
        # Emoji glyph coverage over a bare SSH session to an HPC cluster is
        # unreliable (confirmed via a screenshot where an unsupported emoji
        # silently fell back to an unrelated letter) -- every tip's icon
        # must come from the same safe symbol set used elsewhere in this
        # file (!, ▲, ◆, ...), not an emoji-range codepoint.
        tips = _bottleneck_analysis(_gpu_starved_trace(), {"ipc": 0.5, "cache_miss_pct": 30})
        self.assertTrue(tips)
        for t in tips:
            icon = t[0]
            self.assertLess(ord(icon), 0x2600, f"tip {t!r} has a likely-emoji icon {icon!r}")

    def test_low_ipc_and_high_cache_miss_each_produce_a_tip(self):
        tips = _bottleneck_analysis(_mk_trace([_span(1, 1, Category.CPU, 0, 1000, "x")]),
                                    {"ipc": 0.4, "cache_miss_pct": 25})
        self.assertTrue(any("ipc" in t.lower() for t in tips))
        self.assertTrue(any("llc miss" in t.lower() for t in tips))

    def test_healthy_trace_produces_no_tips(self):
        # Dense, gap-free kernel launches and clean counters -- nothing
        # actionable to report.
        spans = [_span(1, 1, Category.GPU_CUDA, i * 100, 100, "k", tags={"type": "kernel"})
                for i in range(50)]
        tips = _bottleneck_analysis(_mk_trace(spans, backends=["cuda"]),
                                    {"ipc": 3.0, "cache_miss_pct": 1.0})
        self.assertEqual(tips, [])


# ── _diagnose ────────────────────────────────────────────────────────────────

class TestDiagnose(unittest.TestCase):
    def test_gpu_starvation_diagnosed_as_red(self):
        label, color = _diagnose(_gpu_starved_trace())
        self.assertEqual(label, "GPU starvation")
        self.assertEqual(color, "red")

    def test_load_imbalance_diagnosed_for_mpi_trace(self):
        label, color = _diagnose(_imbalanced_mpi_trace())
        self.assertIn("imbalance", label.lower())

    def test_plain_cpu_trace_gets_some_diagnosis_not_a_crash(self):
        t = _mk_trace([_span(1, 1, Category.CPU, 0, 1000, "x")], backends=["cpu"])
        label, color = _diagnose(t)
        self.assertIsInstance(label, str)
        self.assertTrue(label)


# ── _top_findings ────────────────────────────────────────────────────────────

class TestTopFindings(unittest.TestCase):
    def test_single_dominant_function_always_surfaces_a_finding(self):
        # The cheap always-available fallback -- must fire even for a
        # trace with no GPU/MPI backends at all, so the Dashboard's "Top
        # findings" panel is never silently empty for a plain CPU trace.
        spans = [_span(1, 1, Category.CPU, i, 100, "hot_fn") for i in range(10)]
        spans.append(_span(1, 1, Category.CPU, 2000, 5, "cold_fn"))
        findings = _top_findings(_mk_trace(spans, backends=["cpu"]))
        self.assertTrue(findings)
        self.assertTrue(any("hot_fn" in f[2] for f in findings))

    def test_gpu_starvation_produces_low_occupancy_finding(self):
        findings = _top_findings(_gpu_starved_trace())
        titles = [f[2] for f in findings]
        self.assertTrue(any("occupancy" in t.lower() for t in titles))

    def test_findings_capped_at_four(self):
        t = _gpu_starved_trace()
        findings = _top_findings(t)
        self.assertLessEqual(len(findings), 4)

    def test_each_finding_is_a_4_tuple(self):
        for f in _top_findings(_gpu_starved_trace()):
            self.assertEqual(len(f), 4)
            icon, color, title, metric = f
            self.assertIsInstance(icon, str)
            self.assertIsInstance(color, str)
            self.assertIsInstance(title, str)
            self.assertIsInstance(metric, str)


# ── _mini_row ────────────────────────────────────────────────────────────────

class TestMiniRow(unittest.TestCase):
    def test_full_coverage_span_fills_every_column(self):
        span = _span(1, 1, Category.CPU, 0, 1000, "x")
        row = _mini_row([span], width=10, view_start=0, view_dur=1000, color="cyan")
        self.assertEqual(row.plain, "█" * 10)

    def test_no_spans_renders_all_blank(self):
        row = _mini_row([], width=10, view_start=0, view_dur=1000, color="cyan")
        self.assertEqual(row.plain, " " * 10)

    def test_zero_duration_view_returns_empty_without_crashing(self):
        row = _mini_row([_span(1, 1, Category.CPU, 0, 10, "x")],
                        width=10, view_start=0, view_dur=0, color="cyan")
        self.assertEqual(row.plain, "")


# ── _source_snippet ──────────────────────────────────────────────────────────

class TestSourceSnippet(unittest.TestCase):
    def test_no_file_tag_anywhere_returns_none(self):
        t = _mk_trace([_span(1, 1, Category.CPU, 0, 100, "hot_fn")])
        self.assertIsNone(_source_snippet(t))

    def test_file_that_does_not_exist_returns_none(self):
        t = _mk_trace([_span(1, 1, Category.CPU, 0, 100, "hot_fn",
                             tags={"file": "/no/such/path/here.c", "line": "3"})])
        self.assertIsNone(_source_snippet(t))

    def test_line_number_out_of_range_returns_none(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".c", delete=False) as f:
            f.write("int main() {\n    return 0;\n}\n")
            path = f.name
        try:
            t = _mk_trace([_span(1, 1, Category.CPU, 0, 100, "hot_fn",
                                 tags={"file": path, "line": "999"})])
            self.assertIsNone(_source_snippet(t))
        finally:
            Path(path).unlink()

    def test_valid_file_and_line_returns_annotated_snippet(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".c", delete=False) as f:
            f.write("\n".join(f"line{i}" for i in range(1, 11)) + "\n")
            path = f.name
        try:
            t = _mk_trace([_span(1, 1, Category.CPU, 0, 100, "hot_fn",
                                 tags={"file": path, "line": "5"})])
            snippet = _source_snippet(t)
            self.assertIsNotNone(snippet)
            self.assertIn("line5", snippet.plain)
            self.assertIn("▶", snippet.plain)
        finally:
            Path(path).unlink()


# ── DashboardWidget ──────────────────────────────────────────────────────────

class TestDashboardWidget(unittest.IsolatedAsyncioTestCase):
    async def test_rich_trace_populates_all_five_stat_cards(self):
        spans = [_span(1, 1, Category.MPI, i * 1_000_000, 200_000, "MPI_Allreduce",
                       tags={"rank": "0"}) for i in range(5)]
        spans += [_span(2, 2, Category.GPU_CUDA, i * 400_000, 150_000, "k",
                        tags={"type": "kernel"}) for i in range(10)]
        trace = _mk_trace(spans, backends=["cuda", "mpi"], command="gmx_mpi")
        trace.add(CounterEvent(name="process_max_rss_bytes", category=Category.OTHER,
                               timestamp_ns=0, value=2_000_000_000, pid=1))
        app = _HostApp(DashboardWidget(trace))
        async with app.run_test(size=(140, 45)):
            dash = app.query_one(DashboardWidget)
            diag = dash.query_one("#stat-diag").render().plain
            wait = dash.query_one("#stat-wait").render().plain
            mem  = dash.query_one("#stat-mem").render().plain
            self.assertIn("DIAGNOSIS", diag)
            self.assertIn("MPI WAIT", wait)     # MPI spans present -> MPI label, not SYNC
            self.assertIn("1.86 GB", mem)

    async def test_no_mpi_falls_back_to_sync_wait_label(self):
        spans = [_span(1, 1, Category.CPU, 0, 100, "x"),
                _span(1, 1, Category.SYNC, 100, 50, "cudaDeviceSynchronize")]
        trace = _mk_trace(spans, backends=["cpu"])
        app = _HostApp(DashboardWidget(trace))
        async with app.run_test(size=(140, 45)):
            wait = app.query_one(DashboardWidget).query_one("#stat-wait").render().plain
            self.assertIn("SYNC WAIT", wait)
            self.assertNotIn("MPI WAIT", wait)

    async def test_no_gpu_backend_shows_na_for_gpu_active(self):
        trace = _mk_trace([_span(1, 1, Category.CPU, 0, 100, "x")], backends=["cpu"])
        app = _HostApp(DashboardWidget(trace))
        async with app.run_test(size=(140, 45)):
            gpu = app.query_one(DashboardWidget).query_one("#stat-gpu").render().plain
            self.assertIn("n/a", gpu)

    async def test_minimal_single_span_trace_does_not_crash(self):
        trace = _mk_trace([_span(1, 1, Category.CPU, 0, 10, "main")], backends=["cpu"])
        app = _HostApp(DashboardWidget(trace))
        async with app.run_test(size=(100, 30)):
            pass  # on_mount()'s _populate() must complete without raising

    async def test_hot_kernels_table_has_rows_sorted_by_total_time(self):
        spans = [_span(1, 1, Category.CPU, i, 100, "hot") for i in range(10)]
        spans += [_span(1, 1, Category.CPU, 2000 + i, 5, "cold") for i in range(2)]
        trace = _mk_trace(spans, backends=["cpu"])
        app = _HostApp(DashboardWidget(trace))
        async with app.run_test(size=(140, 45)):
            from textual.widgets import DataTable
            dt = app.query_one(DashboardWidget).query_one("#dash-kernels", DataTable)
            self.assertGreaterEqual(dt.row_count, 2)


# ── RooflineWidget ───────────────────────────────────────────────────────────

def _dev() -> DevicePeak:
    return DevicePeak(name="A100", backend="cuda", fp32_tflops=19.5, fp64_tflops=9.7,
                      fp16_tflops=78.0, bandwidth_gbs=1555, sm_count=108,
                      core_clock_ghz=1.41, mem_clock_ghz=1.2, mem_bus_bits=5120,
                      vram_gb=40.0, compute_cap="8.0")


def _kmetric(bound: str, ai: float, tflops: float, ridge: float) -> KernelMetrics:
    return KernelMetrics(kernel_name="k", arch="sm_80", duration_ns=1000, threads=1024,
                         est_flops=1.0, est_bytes=1.0, arith_intensity=ai,
                         achieved_tflops=tflops, achieved_gbs=1.0, flops_pct=1.0,
                         bw_pct=1.0, bound=bound, ridge=ridge)


class TestHasRooflineData(unittest.TestCase):
    def test_false_for_a_trace_with_no_gpu_spans(self):
        t = _mk_trace([_span(1, 1, Category.CPU, 0, 100, "x")], backends=["cpu"])
        self.assertFalse(_has_roofline_data(t))


class TestRooflineWidget(unittest.IsolatedAsyncioTestCase):
    async def test_no_metrics_shows_explanatory_message_not_a_crash(self):
        trace = _mk_trace([_span(1, 1, Category.CPU, 0, 100, "x")])
        rw = RooflineWidget(trace)
        self.assertEqual(rw._metrics, [])
        app = _HostApp(rw)
        async with app.run_test(size=(100, 30)):
            out = rw.render()
            self.assertIn("No roofline data", out.plain)

    async def test_synthetic_metrics_render_braille_scatter_and_legend(self):
        dev = _dev()
        m_compute = _kmetric("compute", ai=1000.0, tflops=15.0, ridge=dev.ridge_point)
        m_memory  = _kmetric("memory",  ai=1.0,    tflops=1.2,  ridge=dev.ridge_point)
        trace = _mk_trace([_span(1, 1, Category.GPU_CUDA, 0, 100, "k", tags={"type": "kernel"})])
        rw = RooflineWidget(trace)
        rw._metrics = [(dev, m_compute), (dev, m_memory)]
        app = _HostApp(rw)
        async with app.run_test(size=(100, 30)):
            out = rw.render()
            braille_chars = sum(1 for ch in out.plain if 0x2800 <= ord(ch) <= 0x28FF)
            self.assertGreater(braille_chars, 0, "expected at least the roofline knee line drawn")
            self.assertIn("compute 1", out.plain)
            self.assertIn("memory 1", out.plain)


# ── ProfilerApp chrome: tabs, numbering, digit-key jump ─────────────────────

class TestProfilerAppTabs(unittest.IsolatedAsyncioTestCase):
    async def test_minimal_trace_gets_exactly_five_base_tabs_in_order(self):
        trace = _mk_trace([_span(1, 1, Category.CPU, 0, 10, "main")], backends=["cpu"])
        app = ProfilerApp(trace)
        async with app.run_test(size=(140, 45)):
            self.assertEqual(
                app._tab_ids,
                ["tab-overview", "tab-timeline", "tab-kernels", "tab-system", "tab-profile"],
            )

    async def test_calltree_tab_appears_only_when_stacks_present(self):
        with_stack = _mk_trace([SpanEvent(name="main", category=Category.CPU, start_ns=0,
                                          duration_ns=10, pid=1, tid=1, tags={},
                                          stack_frames=["main"])])
        app = ProfilerApp(with_stack)
        async with app.run_test(size=(140, 45)):
            self.assertIn("tab-calltree", app._tab_ids)

        without_stack = _mk_trace([_span(1, 1, Category.CPU, 0, 10, "main")])
        app2 = ProfilerApp(without_stack)
        async with app2.run_test(size=(140, 45)):
            self.assertNotIn("tab-calltree", app2._tab_ids)

    async def test_digit_key_jump_activates_the_right_tab(self):
        trace = _mk_trace([_span(1, 1, Category.CPU, 0, 10, "main")], backends=["cpu"])
        app = ProfilerApp(trace)
        async with app.run_test(size=(140, 45)) as pilot:
            await pilot.press("3")
            await pilot.pause()
            from textual.widgets import TabbedContent
            self.assertEqual(app.query_one("#main-tabs", TabbedContent).active, "tab-kernels")

    async def test_goto_tab_out_of_range_is_a_noop_not_a_crash(self):
        trace = _mk_trace([_span(1, 1, Category.CPU, 0, 10, "main")], backends=["cpu"])
        app = ProfilerApp(trace)
        async with app.run_test(size=(140, 45)):
            app.action_goto_tab(99)  # only 5 tabs exist -- must not raise


# ── Markup-collision regression (HotspotsWidget / CallTreeWidget hints) ────

class TestHintTextMarkupSafety(unittest.IsolatedAsyncioTestCase):
    """Regression test for two hint strings that used unescaped brackets
    colliding with Rich markup ([s]=strikethrough, [u]=underline, any
    other bracketed text silently eaten as an unrecognised style tag) --
    caught via a rendered SVG screenshot showing "cycle sort" struck
    through and "[u] collapse all" missing its key label entirely."""

    async def test_hotspots_hint_shows_literal_brackets(self):
        trace = _mk_trace([_span(1, 1, Category.CPU, 0, 10, "x")])
        app = _HostApp(HotspotsWidget(trace))
        async with app.run_test(size=(100, 30)):
            hint = app.query_one("#hs-sort-hint").render()
            self.assertIn("[s]", hint.plain)
            self.assertIn("[/]", hint.plain)
            self.assertIn("[j/k", hint.plain)

    async def test_calltree_hint_shows_literal_brackets(self):
        trace = _mk_trace([SpanEvent(name="main", category=Category.CPU, start_ns=0,
                                     duration_ns=10, pid=1, tid=1, tags={},
                                     stack_frames=["main"])])
        app = _HostApp(CallTreeWidget(trace))
        async with app.run_test(size=(100, 30)):
            hint = app.query_one("#ct-hint").render()
            self.assertIn("[u]", hint.plain)
            self.assertIn("[e]", hint.plain)
            self.assertIn("[enter/space]", hint.plain)


# ── TopBar / BottomBar chrome ────────────────────────────────────────────────

class TestTopBar(unittest.IsolatedAsyncioTestCase):
    async def test_shows_command_and_omits_absent_fields(self):
        trace = _mk_trace([_span(1, 1, Category.CPU, 0, 10, "main")],
                          backends=["cpu"], command="./myapp")
        app = _HostApp(TopBar(trace))
        async with app.run_test(size=(100, 5)):
            tb = app.query_one(TopBar)
            left  = tb.query_one("#topbar-left").render().plain
            right = tb.query_one("#topbar-right").render().plain
            self.assertIn("hprofiler", left)
            self.assertIn("myapp", left)
            self.assertNotIn("ranks", right)   # no MPI spans -> no rank count shown

    async def test_shows_rank_count_and_device_when_present(self):
        spans = [_span(p, p, Category.MPI, 0, 10, "MPI_Init", tags={"rank": str(p)})
                for p in range(4)]
        trace = _mk_trace(spans, backends=["mpi"])
        trace.set_devices([_dev()])
        app = _HostApp(TopBar(trace))
        async with app.run_test(size=(100, 5)):
            right = app.query_one(TopBar).query_one("#topbar-right").render().plain
            self.assertIn("4 ranks", right)
            self.assertIn("A100", right)


class TestBottomBar(unittest.IsolatedAsyncioTestCase):
    async def test_show_hints_renders_extra_then_global_hints(self):
        app = _HostApp(BottomBar())
        async with app.run_test(size=(100, 3)):
            bb = app.query_one(BottomBar)
            bb.show_hints([("x", "do a thing")])
            text = bb.render().plain
            self.assertIn("x do a thing", text)
            self.assertIn("q quit", text)


# ── Wall-time consistency across tabs ────────────────────────────────────────

class TestWallTimeConsistency(unittest.IsolatedAsyncioTestCase):
    """Regression test: SystemWidget's "Duration" line used raw
    trace.duration_ns (meaningless for anything but a live in-process
    trace -- see tests/README.md's "A real bug this test suite caught")
    while every other tab already derived wall time from the spans
    themselves, so the two could show different numbers for the same
    trace. Both must now agree."""

    async def test_system_duration_matches_dashboard_wall_time(self):
        spans = [_span(1, 1, Category.CPU, i * 1000, 500, "x") for i in range(20)]
        trace = _mk_trace(spans, backends=["cpu"])
        expected = _trace_wall_ns(trace)

        sys_app = _HostApp(SystemWidget(trace))
        async with sys_app.run_test(size=(120, 40)):
            sys_text = sys_app.query_one(SystemWidget).render()

        dash_app = _HostApp(DashboardWidget(trace))
        async with dash_app.run_test(size=(140, 45)):
            wall_text = dash_app.query_one(DashboardWidget).query_one("#stat-wall").render().plain

        from src.ui.app import _fmt_ns
        formatted = _fmt_ns(expected)
        self.assertIn(formatted, sys_text)
        self.assertIn(formatted, wall_text)


if __name__ == "__main__":
    unittest.main()
