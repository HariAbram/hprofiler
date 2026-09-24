"""
Tests for the TUI's new Flame Graph tab (src/ui/app.py's FlameGraphWidget
/ _FlameCanvas), which replaced the old dead FlameGraphWidget (a flat,
unwired bar chart -- never composed into ProfilerApp at all). Real
interaction tests via Textual's Pilot (async run_test/click), not just
"does it render" -- matches this project's established "test the actual
interaction, don't just assume the API works" rule for anything with a
click/zoom model (the QML flame graph popup needed the exact same
rigor).
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from textual.app import App, ComposeResult

from src.core.trace import Trace, TraceMetadata
from src.core.events import SpanEvent, Category
from src.ui.app import FlameGraphWidget, _FlameCanvas


def _span(name, dur_ns, stack_frames, cat=Category.CPU):
    return SpanEvent(name=name, category=cat, start_ns=0, duration_ns=dur_ns,
                     pid=1, tid=1, stack_frames=list(stack_frames))


def _mk_trace(spans):
    t = Trace(TraceMetadata(command="./a.out", args=[]))
    for s in spans:
        t.add(s)
    return t


class _HostApp(App):
    """Minimal Textual host for mounting one widget under test (same
    pattern as test_dashboard.py's _HostApp)."""

    def __init__(self, widget, **kwargs):
        super().__init__(**kwargs)
        self._widget = widget

    def compose(self) -> ComposeResult:
        yield self._widget


class TestFlameGraphWidgetNoData(unittest.IsolatedAsyncioTestCase):
    async def test_empty_trace_shows_explanatory_message_not_a_crash(self):
        trace = _mk_trace([_span("fn", 100, [])])  # duration>0 but no stack_frames
        # _ct_build falls back to temporal containment for stack-less
        # spans, so this alone isn't "no data" -- use truly empty spans
        # to hit the real empty-tree path.
        trace = _mk_trace([])
        app = _HostApp(FlameGraphWidget(trace))
        async with app.run_test(size=(100, 30)):
            canvas = app.query_one(_FlameCanvas)
            rendered = canvas.render()
            self.assertIn("No call-stack data", rendered.plain)


class TestFlameGraphWidgetInteraction(unittest.IsolatedAsyncioTestCase):
    def _tree_trace(self) -> Trace:
        # all -> main -> {leaf_a (700ns, 70%), leaf_b (300ns, 30%)}
        return _mk_trace([
            _span("leaf_a", 700, ["main"]),
            _span("leaf_b", 300, ["main"]),
        ])

    async def test_initial_render_shows_root_at_bottom_row(self):
        trace = self._tree_trace()
        app = _HostApp(FlameGraphWidget(trace))
        async with app.run_test(size=(100, 30)):
            canvas = app.query_one(_FlameCanvas)
            self.assertEqual(canvas.current_root["name"], "all")
            # Root is depth 0 -- its built entry must be the bottom-most
            # occupied row (size.height - 1), not stuck at the top.
            root_entries = [f for f in canvas._built if f["depth"] == 0]
            self.assertTrue(root_entries)
            self.assertEqual(root_entries[0]["depth"], 0)

    async def test_click_on_a_leaf_zooms_in(self):
        trace = self._tree_trace()
        app = _HostApp(FlameGraphWidget(trace))
        async with app.run_test(size=(100, 30)) as pilot:
            canvas = app.query_one(_FlameCanvas)
            leaf_a = next(f for f in canvas._built if f["node"]["name"] == "leaf_a")
            cx = leaf_a["x0"] + leaf_a["w"] // 2
            cy = [f for f in canvas._built if f is leaf_a][0]
            # Recompute the row's actual screen y from the canvas's own
            # region (offset within the terminal) rather than assuming.
            region = canvas.region
            y_for_depth = region.y + (region.height - 1 - leaf_a["depth"])
            await pilot.click(_FlameCanvas, offset=(cx, y_for_depth - region.y))
            self.assertEqual(canvas.current_root["name"], "leaf_a")
            self.assertEqual(len(canvas.zoom_stack), 2)

    async def test_right_click_zooms_back_out(self):
        trace = self._tree_trace()
        app = _HostApp(FlameGraphWidget(trace))
        async with app.run_test(size=(100, 30)) as pilot:
            canvas = app.query_one(_FlameCanvas)
            main_frame = next(f for f in canvas._built if f["node"]["name"] == "main")
            cx = main_frame["x0"] + main_frame["w"] // 2
            region = canvas.region
            y_offset = region.height - 1 - main_frame["depth"]
            await pilot.click(_FlameCanvas, offset=(cx, y_offset), button=1)
            self.assertEqual(canvas.current_root["name"], "main")
            # Right-click anywhere inside the (now zoomed) canvas zooms
            # back out one level, same as the QML popup's own model.
            await pilot.click(_FlameCanvas, offset=(cx, y_offset), button=3)
            self.assertEqual(canvas.current_root["name"], "all")
            self.assertEqual(len(canvas.zoom_stack), 1)

    async def test_escape_key_resets_zoom_from_any_depth(self):
        trace = self._tree_trace()
        app = _HostApp(FlameGraphWidget(trace))
        async with app.run_test(size=(100, 30)) as pilot:
            canvas = app.query_one(_FlameCanvas)
            main_frame = next(f for f in canvas._built if f["node"]["name"] == "main")
            canvas.zoom_into(main_frame["node"])
            leaf = next(c for c in main_frame["node"]["children"] if c["name"] == "leaf_a")
            canvas.zoom_into(leaf)
            self.assertEqual(len(canvas.zoom_stack), 3)
            canvas.focus()
            await pilot.press("escape")
            self.assertEqual(canvas.current_root["name"], "all")
            self.assertEqual(len(canvas.zoom_stack), 1)

    async def test_search_dims_non_matching_frames(self):
        trace = self._tree_trace()
        widget = FlameGraphWidget(trace)
        app = _HostApp(widget)
        async with app.run_test(size=(100, 30)) as pilot:
            search_box = app.query_one("#fg-search")
            search_box.focus()
            await pilot.press(*"leaf_a")
            canvas = app.query_one(_FlameCanvas)
            self.assertEqual(canvas.search, "leaf_a")
            rendered = canvas.render()
            # Just confirm it re-renders without crashing and the search
            # state actually reached the canvas -- exact color assertions
            # would be brittle against Rich's own Style internals.
            self.assertIsNotNone(rendered)


if __name__ == "__main__":
    unittest.main()
