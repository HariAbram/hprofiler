"""
Tests for src/gui/models.py's TimelineModel -- the Timeline screen's data
layer (Phase 4). Skipped if PySide6 isn't installed. See
tests/test_gui_bridge.py's docstring for why these run headless/offscreen.
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
from src.core.events import SpanEvent, Category


def _span(pid, tid, cat, start_ns, dur_ns, name, tags=None):
    return SpanEvent(name=name, category=cat, start_ns=start_ns, duration_ns=dur_ns,
                     pid=pid, tid=tid, tags=dict(tags or {}))


def _mk_trace(spans):
    t = Trace(TraceMetadata())
    for s in spans:
        t.add(s)
    return t


@unittest.skipUnless(_PYSIDE6_AVAILABLE, "PySide6 not installed")
class TestTimelineModel(unittest.TestCase):
    _app = None

    @classmethod
    def setUpClass(cls):
        cls._app = QGuiApplication.instance() or QGuiApplication([])

    def _model(self, trace):
        from src.gui.theme import Theme
        from src.gui.models import TimelineModel
        return TimelineModel(trace, Theme(dark=True))

    def test_lanes_labeled_by_mpi_rank_not_generic_thread_number(self):
        a = _span(100, 101, Category.MPI, 0, 10, "MPI_Send", tags={"rank": "5"})
        m = self._model(_mk_trace([a]))
        labels = [l["label"] for l in m.lanes]
        self.assertTrue(any("rank5" in l for l in labels))

    def test_non_mpi_lane_uses_sequential_thread_label(self):
        a = _span(100, 101, Category.CPU, 0, 10, "kernel")
        m = self._model(_mk_trace([a]))
        labels = [l["label"] for l in m.lanes]
        self.assertTrue(any("T1" in l for l in labels))

    def test_lane_colors_are_hex(self):
        a = _span(100, 101, Category.GPU_CUDA, 0, 10, "k")
        m = self._model(_mk_trace([a]))
        for lane in m.lanes:
            self.assertTrue(lane["color"].startswith("#"))

    def test_visible_spans_culls_to_viewport(self):
        # 100 spans spread across a wide range -- a narrow viewport must
        # only return the ones actually overlapping it.
        spans = [_span(1, 1, Category.CPU, i * 1_000_000, 1000, f"s{i}") for i in range(100)]
        m = self._model(_mk_trace(spans))
        lane_idx = 0
        all_visible = m.visibleSpans(lane_idx, 0, 100_000_000, 1000)
        self.assertEqual(len(all_visible), 100)
        narrow = m.visibleSpans(lane_idx, 0, 5_000_000, 1000)
        self.assertLess(len(narrow), 100)
        self.assertGreaterEqual(len(narrow), 5)

    def test_visible_spans_out_of_range_lane_index_returns_empty(self):
        m = self._model(_mk_trace([_span(1, 1, Category.CPU, 0, 10, "x")]))
        self.assertEqual(m.visibleSpans(99, 0, 1000, 100), [])
        self.assertEqual(m.visibleSpans(-1, 0, 1000, 100), [])

    def test_visible_spans_lo_bound_uses_per_span_duration_not_lane_span(self):
        # Regression test for a real bug: _max_dur used to be the lane's
        # FULL first-to-last time range (ends.max()-starts.min()), not the
        # longest INDIVIDUAL span's own duration -- so searchsorted's
        # look-back margin was wildly oversized for any lane whose spans
        # spread across most of the trace, pinning `lo` near index 0
        # regardless of how far into the trace the query window actually
        # was. One long early span (duration 1000) followed by a cluster
        # of short spans far later -- querying a window that starts well
        # after the early span's TRUE end (but still within the old,
        # bogus, whole-lane-range look-back) must not resurrect it.
        early = _span(1, 1, Category.OPENMP, 0, 1000, "early_long_span")
        late = [_span(1, 1, Category.OPENMP, 10_000_000 + i * 100, 50, f"late{i}")
                for i in range(5)]
        m = self._model(_mk_trace([early] + late))
        lane_idx = 0
        result = m.visibleSpans(lane_idx, 9_000_000, 9_500_000, 2000)
        self.assertEqual(result, [])
        names = {r["name"] for r in m.visibleSpans(lane_idx, 10_000_000, 10_001_000, 2000)}
        self.assertNotIn("early_long_span", names)
        self.assertTrue(any(n.startswith("late") for n in names))

    def test_visible_spans_preserves_idle_gap_between_two_clusters(self):
        # Two clusters of spans with a genuine, deliberate idle gap
        # between them (no span at all covers that time) -- querying
        # squarely inside the gap must return nothing, not spans smeared
        # in from either cluster.
        cluster_a = [_span(1, 1, Category.OPENMP, i * 1000, 500, f"a{i}") for i in range(20)]
        cluster_b = [_span(1, 1, Category.OPENMP, 100_000 + i * 1000, 500, f"b{i}") for i in range(20)]
        m = self._model(_mk_trace(cluster_a + cluster_b))
        lane_idx = 0
        gap = m.visibleSpans(lane_idx, 40_000, 60_000, 2000)
        self.assertEqual(gap, [])

    def test_visible_spans_max_dur_is_longest_single_span_not_lane_range(self):
        a = _span(1, 1, Category.CPU, 0, 500, "a")
        b = _span(1, 1, Category.CPU, 1_000_000, 30, "b")
        m = self._model(_mk_trace([a, b]))
        self.assertEqual(m._max_dur["cpu/thread-1"], 500)

    def test_visible_spans_respects_max_spans_cap(self):
        spans = [_span(1, 1, Category.CPU, i * 100, 50, f"s{i}") for i in range(500)]
        m = self._model(_mk_trace(spans))
        capped = m.visibleSpans(0, 0, 100_000, 50)
        self.assertLessEqual(len(capped), 60)  # cap=50 plus stepping slack

    def test_span_at_returns_full_detail(self):
        a = _span(1, 1, Category.MPI, 1000, 500, "MPI_Bcast", tags={"rank": "2"})
        m = self._model(_mk_trace([a]))
        detail = m.spanAt(0, 0)
        self.assertEqual(detail["name"], "MPI_Bcast")
        self.assertEqual(detail["category"], "mpi")
        self.assertEqual(detail["durNs"], 500.0)
        self.assertEqual(detail["tags"]["rank"], "2")

    def test_span_at_out_of_range_returns_empty_dict(self):
        m = self._model(_mk_trace([_span(1, 1, Category.CPU, 0, 10, "x")]))
        self.assertEqual(m.spanAt(0, 99), {})
        self.assertEqual(m.spanAt(99, 0), {})

    def test_cross_lane_connector_reshaped_with_lane_and_span_indices(self):
        # Same matched send/recv pattern as tests/test_timeline_connectors.py
        # (the TUI's equivalent test) -- proves this model consumes the
        # exact same criticalpath.py dependency graph, just reshaped for
        # Canvas rendering (lane/span indices) instead of Braille coords.
        send = _span(1, 101, Category.MPI, 1000, 50, "MPI_Send",
                     tags={"type": "send", "rank": "0", "peer": "1", "tag": "7"})
        recv = _span(1, 201, Category.MPI, 1100, 80, "MPI_Recv",
                     tags={"type": "recv", "rank": "1", "peer": "0", "tag": "7"})
        m = self._model(_mk_trace([send, recv]))
        self.assertEqual(len(m.connectors), 1)
        c = m.connectors[0]
        self.assertIn("predLane", c)
        self.assertIn("succLane", c)
        self.assertNotEqual(c["predLane"], c["succLane"])
        self.assertTrue(c["color"].startswith("#"))

    def test_minimal_single_span_trace_does_not_crash(self):
        self._model(_mk_trace([_span(1, 1, Category.CPU, 0, 10, "main")]))

    def test_call_graph_scoped_to_visible_window(self):
        # Two spans with captured stacks, one inside the queried window,
        # one entirely before it -- only the visible one should
        # contribute nodes/edges.
        visible = SpanEvent(name="in_view", category=Category.CPU,
                            start_ns=1000, duration_ns=100, pid=1, tid=1,
                            stack_frames=["caller_a"])
        offscreen = SpanEvent(name="out_of_view", category=Category.CPU,
                              start_ns=0, duration_ns=10, pid=1, tid=1,
                              stack_frames=["caller_b"])
        m = self._model(_mk_trace([visible, offscreen]))
        result = m.callGraph(900.0, 1200.0)
        names = {n["name"] for n in result["nodes"]}
        self.assertIn("in_view", names)
        self.assertIn("caller_a", names)
        self.assertNotIn("out_of_view", names)
        self.assertNotIn("caller_b", names)

    def test_call_graph_nodes_have_normalized_positions_and_colors(self):
        span = SpanEvent(name="leaf", category=Category.CPU,
                         start_ns=0, duration_ns=100, pid=1, tid=1,
                         stack_frames=["root"])
        m = self._model(_mk_trace([span]))
        result = m.callGraph(0.0, 1000.0)
        self.assertEqual(len(result["nodes"]), 2)
        for n in result["nodes"]:
            self.assertGreaterEqual(n["x"], 0.0)
            self.assertLessEqual(n["x"], 1.0)
            self.assertTrue(n["color"].startswith("#"))

    def test_call_graph_empty_when_no_spans_have_stack_frames(self):
        m = self._model(_mk_trace([_span(1, 1, Category.CPU, 0, 100, "fn")]))
        result = m.callGraph(0.0, 1000.0)
        self.assertEqual(result["nodes"], [])
        self.assertEqual(result["edges"], [])


if __name__ == "__main__":
    unittest.main()
