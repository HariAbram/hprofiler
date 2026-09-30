"""
Trace.lanes() naming and parse_lane_name() -- the display-lane key both
Timelines (TUI src/ui/app.py, GUI src/gui/models.py) build rows from.

Regression: lanes were keyed by stream id or tid WITHOUT the process, so in
a multi-rank GPU job every rank's default-stream kernels (stream id 0 in
every process) were drawn overlapping in ONE "cuda/stream-0" lane, and
merge-nodes output (pids remapped, tids not) merged same-tid threads from
different nodes. Device-timed OpenCL spans (side=gpu) landed in whatever
driver callback thread's lane fired -- sometimes the main thread's.
"""
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.trace import Trace, TraceMetadata, parse_lane_name
from src.core.events import SpanEvent, Category


def _s(cat, pid, tid, start=0, dur=10, name="x", **tags):
    return SpanEvent(name=name, category=cat, start_ns=start, duration_ns=dur, pid=pid, tid=tid, tags=tags)


def _trace(spans):
    t = Trace(TraceMetadata(command="a.out"))
    for s in spans:
        t.add(s)
    return t


class TestLaneNaming(unittest.TestCase):
    def test_single_process_names_unchanged(self):
        t = _trace([_s(Category.GPU_CUDA, 1, 10, type="kernel", stream="0"),
                    _s(Category.CPU, 1, 10), _s(Category.MPI, 1, 11)])
        self.assertEqual(set(t.lanes()), {"cuda/stream-0", "cpu/thread-10", "mpi/thread-11"})

    def test_default_stream_of_two_ranks_is_two_lanes(self):
        t = _trace([_s(Category.GPU_CUDA, 100, 100, type="kernel", stream="0"),
                    _s(Category.GPU_CUDA, 200, 200, type="kernel", stream="0")])
        lanes = t.lanes()
        self.assertEqual(set(lanes), {"cuda/stream-0@100", "cuda/stream-0@200"})
        self.assertEqual([s.pid for s in lanes["cuda/stream-0@100"]], [100])

    def test_same_tid_in_two_merged_nodes_is_two_lanes(self):
        t = _trace([_s(Category.CPU, 5, 777), _s(Category.CPU, 1_000_005, 777)])
        self.assertEqual(len(t.lanes()), 2)

    def test_device_timed_opencl_spans_get_a_device_lane(self):
        t = _trace([_s(Category.GPU_OPENCL, 1, 50, type="kernel", side="cpu"),
                    _s(Category.GPU_OPENCL, 1, 50, start=20, type="kernel", side="gpu"),
                    _s(Category.MEMORY, 1, 61, name="clReadBuffer_gpu", type="read")])
        self.assertEqual(set(t.lanes()), {"opencl/thread-50", "opencl/device", "memory/device"})

    def test_async_copy_goes_to_its_stream_lane(self):
        t = _trace([_s(Category.MEMORY, 1, 9, type="memcpy_async", stream="3"),
                    _s(Category.MEMORY, 1, 9, type="memcpy")])
        self.assertEqual(set(t.lanes()), {"memory/stream-3", "memory/thread-9"})

    def test_parse_round_trips_every_form(self):
        cases = {
            "cuda/stream-0": ("cuda", "stream", "0", None),
            "cuda/stream-0@42": ("cuda", "stream", "0", 42),
            "cpu/thread-7": ("cpu", "thread", "7", None),
            "cpu/thread-7@3": ("cpu", "thread", "7", 3),
            "opencl/device": ("opencl", "device", "", None),
            "opencl/device@9": ("opencl", "device", "", 9),
            "cpu": ("cpu", "", "", None),
            "cpu@12": ("cpu", "", "", 12),
        }
        for name, want in cases.items():
            self.assertEqual(parse_lane_name(name), want, name)


try:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QCoreApplication
    _PYSIDE6 = True
except ImportError:
    _PYSIDE6 = False


@unittest.skipUnless(_PYSIDE6, "PySide6 not installed")
class TestGuiTimelineLanes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QCoreApplication.instance() or QCoreApplication([])

    def test_two_ranks_default_stream_distinct_rows_and_labels(self):
        from src.gui.theme import Theme
        from src.gui.models import TimelineModel
        t = _trace([_s(Category.MPI, 100, 100, name="MPI_Barrier", type="barrier", rank="0"),
                    _s(Category.MPI, 200, 200, name="MPI_Barrier", type="barrier", rank="1"),
                    _s(Category.GPU_CUDA, 100, 100, start=20, type="kernel", stream="0"),
                    _s(Category.GPU_CUDA, 200, 200, start=20, type="kernel", stream="0"),
                    _s(Category.CPU, 300, 0)])  # bare-category lane, also multi-pid-free
        m = TimelineModel(t, Theme(dark=True))
        labels = [lane["label"] for lane in m.lanes]
        cuda = [lb for lb in labels if lb.startswith("cuda")]
        self.assertEqual(sorted(cuda), ["cuda S0 r0", "cuda S0 r1"])
        self.assertEqual(len(labels), len(set(labels)))


class TestTuiTimelineLanes(unittest.TestCase):
    def test_pid_disambiguated_lanes_render_distinct_labels(self):
        from src.ui.app import TimelineWidget
        t = _trace([_s(Category.GPU_CUDA, 100, 100, type="kernel", stream="0"),
                    _s(Category.GPU_CUDA, 200, 200, start=5, type="kernel", stream="0"),
                    _s(Category.CPU, 100, 7), _s(Category.CPU, 1_000_100, 7, start=3)])
        w = TimelineWidget(t)
        labels = [w._lane_label(ln) for ln in w._lane_names]
        self.assertEqual(len(labels), len(set(labels)), labels)
        self.assertEqual(len(w._lane_names), 4)


if __name__ == "__main__":
    unittest.main()
