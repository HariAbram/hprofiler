"""
Tests for TimelineWidget's cross-rank communication connector overlay
(src/ui/app.py + src/ui/braille_canvas.py) -- draws MPI/NCCL send/recv and
collective-rendezvous connector lines across lanes, using
criticalpath.py's already-resolved dependency-graph edges (wildcard
matching, commid=-scoped rendezvous, confidence tiers) rather than
re-deriving matching logic in the UI layer.

Uses Textual's headless run_test() harness (App.run_test()) since
TimelineWidget.render() reads self.size, which is only meaningfully set
inside a mounted widget context -- this project's first UI-level test,
previously everything exercised only the analysis/hook layers directly.
"""
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.trace import Trace, TraceMetadata
from src.core.events import SpanEvent, Category
from src.ui.app import TimelineWidget
from textual.app import App, ComposeResult


def _span(pid, tid, cat, start_ns, dur_ns, name, tags=None):
    return SpanEvent(name=name, category=cat, start_ns=start_ns, duration_ns=dur_ns,
                     pid=pid, tid=tid, tags=dict(tags or {}))


def _mk_trace(spans):
    t = Trace(TraceMetadata())
    for s in spans:
        t.add(s)
    return t


def _braille_count(text_plain: str) -> int:
    return sum(1 for ch in text_plain if 0x2800 <= ord(ch) <= 0x28FF)


class _HarnessApp(App):
    def __init__(self, trace, **kwargs):
        super().__init__(**kwargs)
        self._trace = trace

    def compose(self) -> ComposeResult:
        yield TimelineWidget(self._trace)


class TestConnectorComputation(unittest.TestCase):
    """These don't need a mounted widget -- __init__ computes
    self._connectors independent of render()/self.size."""

    def test_cross_lane_p2p_produces_one_connector(self):
        send = _span(1, 101, Category.MPI, 1000, 50, "MPI_Send",
                     tags={"type": "send", "rank": "0", "peer": "1", "tag": "7"})
        recv = _span(1, 201, Category.MPI, 1100, 80, "MPI_Recv",
                     tags={"type": "recv", "rank": "1", "peer": "0", "tag": "7"})
        w = TimelineWidget(_mk_trace([send, recv]))
        self.assertEqual(len(w._connectors), 1)
        pred_lane, _pred_ns, succ_lane, _succ_ns, confidence, pred_sid, succ_sid = w._connectors[0]
        self.assertEqual(pred_lane, "mpi/thread-101")
        self.assertEqual(succ_lane, "mpi/thread-201")
        self.assertEqual(confidence, "medium")
        self.assertEqual(pred_sid, id(send))
        self.assertEqual(succ_sid, id(recv))
        # Precomputed hover-text link-count index (on_mouse_move's "⇄N" hint).
        self.assertEqual(w._connector_count[id(send)], 1)
        self.assertEqual(w._connector_count[id(recv)], 1)

    def test_same_lane_edges_produce_no_connector(self):
        # Both spans on the SAME thread -- already visually adjacent in one
        # row, no cross-lane connector needed (this is what
        # _add_program_order_edges would link, not p2p/arrival anyway, but
        # confirms the same-lane skip explicitly).
        a = _span(1, 101, Category.MPI, 1000, 50, "MPI_Send",
                 tags={"type": "send", "rank": "0", "peer": "0", "tag": "1"})
        b = _span(1, 101, Category.MPI, 1100, 50, "MPI_Recv",
                 tags={"type": "recv", "rank": "0", "peer": "0", "tag": "1"})
        w = TimelineWidget(_mk_trace([a, b]))
        self.assertEqual(len(w._connectors), 0)

    def test_non_mpi_nccl_categories_produce_no_connector(self):
        a = _span(1, 101, Category.CPU, 1000, 50, "work_a")
        b = _span(1, 201, Category.CPU, 1100, 50, "work_b")
        w = TimelineWidget(_mk_trace([a, b]))
        self.assertEqual(len(w._connectors), 0)

    def test_commid_scoped_rendezvous_produces_high_confidence_connector(self):
        a1 = _span(1, 101, Category.MPI, 1000, 100, "MPI_Allreduce",
                  tags={"type": "allreduce", "commid": "1"})
        a2 = _span(1, 201, Category.MPI, 1010, 80, "MPI_Allreduce",
                  tags={"type": "allreduce", "commid": "1"})
        w = TimelineWidget(_mk_trace([a1, a2]))
        self.assertEqual(len(w._connectors), 1)
        self.assertEqual(w._connectors[0][4], "high")

    def test_empty_trace_produces_no_connectors_and_no_crash(self):
        w = TimelineWidget(_mk_trace([]))
        self.assertEqual(w._connectors, [])


class TestConnectorRendering(unittest.IsolatedAsyncioTestCase):
    async def test_hovering_an_endpoint_renders_braille_overlay(self):
        send = _span(100, 101, Category.MPI, 1_000_000, 50_000, "MPI_Send",
                     tags={"type": "send", "rank": "0", "peer": "1", "tag": "7"})
        recv = _span(200, 201, Category.MPI, 1_100_000, 80_000, "MPI_Recv",
                     tags={"type": "recv", "rank": "1", "peer": "0", "tag": "7"})
        app = _HarnessApp(_mk_trace([send, recv]))
        async with app.run_test(size=(120, 30)):
            widget = app.query_one(TimelineWidget)
            self.assertEqual(len(widget._connectors), 1)
            widget._hover_span_id = id(send)
            plain = widget.render().plain
            self.assertGreater(_braille_count(plain), 0,
                              "expected connector overlay characters once an endpoint is hovered")

    async def test_connectors_present_but_nothing_hovered_renders_no_braille(self):
        # The headline change from the raw always-on version: connectors
        # existing is no longer sufficient to draw them -- a span must
        # actually be hovered, to avoid a hairball of every edge at once
        # on a busy trace.
        send = _span(100, 101, Category.MPI, 1_000_000, 50_000, "MPI_Send",
                     tags={"type": "send", "rank": "0", "peer": "1", "tag": "7"})
        recv = _span(200, 201, Category.MPI, 1_100_000, 80_000, "MPI_Recv",
                     tags={"type": "recv", "rank": "1", "peer": "0", "tag": "7"})
        app = _HarnessApp(_mk_trace([send, recv]))
        async with app.run_test(size=(120, 30)):
            widget = app.query_one(TimelineWidget)
            self.assertEqual(len(widget._connectors), 1)
            self.assertEqual(widget._hover_span_id, 0)  # nothing hovered yet
            plain = widget.render().plain
            self.assertEqual(_braille_count(plain), 0)

    async def test_hovering_an_unrelated_span_renders_no_braille(self):
        send = _span(100, 101, Category.MPI, 1_000_000, 50_000, "MPI_Send",
                     tags={"type": "send", "rank": "0", "peer": "1", "tag": "7"})
        recv = _span(200, 201, Category.MPI, 1_100_000, 80_000, "MPI_Recv",
                     tags={"type": "recv", "rank": "1", "peer": "0", "tag": "7"})
        unrelated = _span(300, 301, Category.CPU, 1_000_000, 10_000, "other_work")
        app = _HarnessApp(_mk_trace([send, recv, unrelated]))
        async with app.run_test(size=(120, 30)):
            widget = app.query_one(TimelineWidget)
            widget._hover_span_id = id(unrelated)
            plain = widget.render().plain
            self.assertEqual(_braille_count(plain), 0)

    async def test_no_connectors_means_no_braille_chars_rendered(self):
        # Regression guard: a trace with nothing to connect must render
        # IDENTICALLY to how TimelineWidget behaved before this feature
        # existed -- zero Braille characters anywhere in the output.
        a = _span(1, 101, Category.CPU, 0, 100, "work")
        app = _HarnessApp(_mk_trace([a]))
        async with app.run_test(size=(120, 30)):
            widget = app.query_one(TimelineWidget)
            self.assertEqual(len(widget._connectors), 0)
            plain = widget.render().plain
            self.assertEqual(_braille_count(plain), 0)

    async def test_scrolled_view_does_not_crash_with_connectors_present(self):
        send = _span(100, 101, Category.MPI, 1_000_000, 50_000, "MPI_Send",
                     tags={"type": "send", "rank": "0", "peer": "1", "tag": "7"})
        recv = _span(200, 201, Category.MPI, 1_100_000, 80_000, "MPI_Recv",
                     tags={"type": "recv", "rank": "1", "peer": "0", "tag": "7"})
        app = _HarnessApp(_mk_trace([send, recv]))
        async with app.run_test(size=(120, 30)):
            widget = app.query_one(TimelineWidget)
            widget._hover_span_id = id(send)  # exercise the overlay path, not just computation
            # Zoom/pan far enough that both endpoints fall outside [0,
            # width) -- must not raise, just render without that connector
            # (or clipped) rather than crash on out-of-range dot coordinates.
            widget.zoom = 50.0
            widget.view_x = 100000
            widget.render()  # must not raise
            widget.view_y = 999999  # far beyond any real lane count
            widget.render()  # must not raise

    async def test_many_connectors_render_within_reasonable_time(self):
        # A real multi-rank MPI trace could have far more than 2
        # connectors -- confirm this scales reasonably, not just
        # correctness on a toy example. 64 ranks x a send/recv pair each,
        # all on one commid=-scoped rendezvous too.
        spans = []
        for r in range(64):
            spans.append(_span(r, 1000 + r, Category.MPI, 1000 + r * 10, 50,
                               "MPI_Send", tags={"type": "send", "rank": str(r),
                                                 "peer": str((r + 1) % 64), "tag": "1"}))
            spans.append(_span(r, 2000 + r, Category.MPI, 1500 + r * 10, 50,
                               "MPI_Recv", tags={"type": "recv", "rank": str(r),
                                                 "peer": str((r - 1) % 64), "tag": "1"}))
            spans.append(_span(r, 3000 + r, Category.MPI, 2000 + r * 5, 100,
                               "MPI_Allreduce", tags={"type": "allreduce", "commid": "1"}))
        app = _HarnessApp(_mk_trace(spans))
        async with app.run_test(size=(150, 60)):
            widget = app.query_one(TimelineWidget)
            self.assertGreater(len(widget._connectors), 0)
            # Hover the busiest span (the allreduce cluster's "last
            # arriver" has ~63 edges all pointing to it -- the worst case
            # for the overlay path, not just connector computation).
            busiest_sid = max(widget._connector_count, key=widget._connector_count.get)
            widget._hover_span_id = busiest_sid
            t0 = time.monotonic()
            widget.render()
            elapsed = time.monotonic() - t0
            self.assertLess(elapsed, 2.0,
                            f"render() took {elapsed:.2f}s for {widget._connector_count[busiest_sid]} "
                            f"connectors on one hovered span -- likely a real performance problem, "
                            f"not just slow CI")


if __name__ == "__main__":
    unittest.main()
