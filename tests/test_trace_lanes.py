"""
Trace.lanes() naming and parse_lane_name() -- the display-lane key both
Timelines (TUI src/ui/app.py, GUI src/gui/models.py) build rows from.

Lanes are keyed by process as well as stream id or tid: in a multi-rank GPU
job every rank's default-stream kernels share stream id 0, and merge-nodes
output (pids remapped, tids not) reuses tids across nodes, so without the
process they would overlap in one lane. Device-timed OpenCL spans
(side=gpu) get a per-category device lane rather than whichever driver
callback thread reported them.
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

    def _cupti_trace(self):
        """pid 100: streams 900001 (first work at 10), 5 (at 50) and the
        default stream, CUPTI sync records that name only their context
        (ctx=1, no dev=) -- one without a stream, one on stream 5; pid
        200 uses the same two ids, first seen the other way round."""
        k = dict(type="kernel", side="gpu", rt="cuda", timing="device", ctx="1", dev="0")
        sync = dict(type="sync_wait", side="gpu", rt="cuda", op="sync", timing="host", src="cupti", ctx="1")
        return _trace([
            _s(Category.GPU_CUDA, 100, 100, start=10, stream="900001", nstream="16", **k),
            _s(Category.GPU_CUDA, 100, 100, start=50, stream="5", nstream="15", **k),
            _s(Category.MEMORY, 100, 100, start=60, name="memcpy DtoH", stream="5", nstream="15",
               **dict(k, type="memcpy")),
            _s(Category.GPU_CUDA, 100, 100, start=70, stream="0", nstream="7", **k),
            _s(Category.SYNC, 100, 100, start=80, name="sync event", sync="event", **sync),
            _s(Category.SYNC, 100, 100, start=90, name="sync stream", sync="stream", stream="5", nstream="15", **sync),
            _s(Category.GPU_CUDA, 200, 200, start=1, stream="5", **k),
            _s(Category.GPU_CUDA, 200, 200, start=5, stream="900001", **k),
        ])

    @staticmethod
    def _where(m, name, **match):
        sp = next(s for s in m._store.iter_spans() if s.name == name
                  and all(getattr(s, k, None) == v or s.tags.get(k) == v for k, v in match.items()))
        d = m.eventDetails(m._lane_of_span(sp), float(sp.eid))
        return next(x for x in d["sections"] if x["title"] == "Where")["fields"]

    def test_cupti_sync_records_join_their_contexts_device_as_host_waits(self):
        from src.gui.theme import Theme
        from src.gui.models import TimelineModel
        m = TimelineModel(self._cupti_trace(), Theme(dark=True))
        rows = [(r["level"], r.get("hierLabel") or r["label"]) for r in m.rows]
        p100 = rows[[lb for _l, lb in rows].index("pid 100") + 1:[lb for _l, lb in rows].index("pid 200")]
        self.assertEqual(p100, [
            (2, "device 0"), (3, "stream 0 (default) · cuda"), (3, "stream 1 · cuda"), (3, "stream 2"),
            (4, "cuda"), (4, "memory"), (4, "sync (host waits)"), (3, "sync (host waits)")])
        self.assertNotIn("device", [lb for _l, lb in rows])           # no unnumbered device
        where = {f["label"]: f for f in self._where(m, "sync event")}
        self.assertEqual((where["Device"]["value"], where["Device"]["kind"]), ("0", "derived"))
        self.assertIn("Waiting thread", where)
        self.assertNotIn("Launched by thread", where)

    def test_a_context_on_two_devices_names_no_device(self):
        from src.gui.theme import Theme
        from src.gui.models import TimelineModel
        k = dict(type="kernel", side="gpu", rt="cuda", timing="device", ctx="3")
        t = _trace([_s(Category.GPU_CUDA, 1, 1, stream="7", dev="0", **k),
                    _s(Category.GPU_CUDA, 1, 1, stream="8", dev="1", **k),
                    _s(Category.SYNC, 1, 1, name="sync context", type="sync_wait", side="gpu", rt="cuda",
                       timing="host", ctx="3")])
        m = TimelineModel(t, Theme(dark=True))
        self.assertIsNone(m._lane_extra[m._lane_index["sync/device"]]["device"])

    def test_streams_numbered_per_process_by_first_work_ids_in_details(self):
        from src.gui.theme import Theme
        from src.gui.models import TimelineModel
        from src.gui import timeline_details as td
        m = TimelineModel(self._cupti_trace(), Theme(dark=True))
        ex = {(e["pid"], e["ident"]): e["streamLabel"] for e in m._lane_extra if e["kind"] == "stream"}
        self.assertEqual(ex, {(100, "900001"): "stream 1", (100, "5"): "stream 2", (100, "0"): "stream 0 (default)",
                              (200, "5"): "stream 1", (200, "900001"): "stream 2"})
        labels = [lane["label"] for lane in m.lanes]
        self.assertIn("cuda S1 p100", labels)
        self.assertFalse(any("900001" in lb for lb in labels))
        dim = next(d for d in m.filterDimensions if d["key"] == "stream")
        self.assertEqual(dim["values"], [{"value": "0", "label": "stream 0 (default)"},
                                         {"value": "5", "label": "stream 1 / stream 2"},
                                         {"value": "900001", "label": "stream 1 / stream 2"}])
        m.setGrouping("stream")
        self.assertIn("Stream 0 (default)", [r["label"] for r in m.rows if r["kind"] == "group"])
        where = {f["label"]: f["value"] for f in self._where(m, "x", pid=100, stream="900001")}
        self.assertEqual((where["Stream / queue"], where["Stream id"], where["CUPTI stream id"]),
                         ("stream 1", "900001", "16"))
        jump = td.resolve_jump(m, "stream 2")
        self.assertEqual(sorted(m._lane_names[i] for i in jump["laneIndexes"]),
                         ["cuda/stream-5@100", "cuda/stream-900001@200", "memory/stream-5", "sync/stream-5"])
        self.assertEqual(td.resolve_jump(m, "stream 900001")["laneIndexes"],
                         [m._lane_index[n] for n in m._lane_names if n.startswith("cuda/stream-900001")])


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
