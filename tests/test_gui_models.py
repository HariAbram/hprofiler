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


if __name__ == "__main__":
    unittest.main()
