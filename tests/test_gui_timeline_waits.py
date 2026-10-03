"""
Timeline thread lanes show real activity, not a solid "busy" bar: a span
such as omp_parallel_region covers a whole region on a thread, including
the barrier/critical waits the paired sync lane reports. Those waits must
be cut out of the thread lane and painted in the neutral Idle style, while
work executed inside a wait (a task run during a barrier) stays visible on
top (an overlay in a saturated colour would still read as busy and would
hide such tasks).

Renders the real Main.qml offscreen in a SUBPROCESS (one QML engine per
process: a second engine in the test process corrupts Qt Quick Controls)
and samples pixels of the thread lane.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

import importlib.util

_HAVE_PYSIDE = importlib.util.find_spec("PySide6") is not None

_SCRIPT = r'''
import json, os, sys
os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, sys.argv[1])
from PySide6.QtTest import QTest
from PySide6.QtQuick import QQuickItem
import tests.test_gui_timeline_hover as H
from src.core.events import Category, SpanEvent
from src.core.trace import Trace, TraceMetadata

def build():
    t = Trace(TraceMetadata(command="a.out", args=[]))
    # one OpenMP worker thread: a region covering everything, a barrier wait
    # in its second half, and a task executed inside that barrier
    t.add(SpanEvent("omp_parallel_region", Category.OPENMP, 0, 1_000_000, 1, 10, {"type": "parallel"}))
    t.add(SpanEvent("omp_barrier", Category.SYNC, 500_000, 500_000, 1, 10, {"type": "barrier"}))
    t.add(SpanEvent("omp_task", Category.OPENMP, 700_000, 100_000, 1, 10, {"type": "task"}))
    # stretches the trace to 2.5 ms so the events above sit in the left 40%
    # of the lane, clear of the docked Inspector
    t.add(SpanEvent("elsewhere", Category.CPU, 2_400_000, 100_000, 1, 99, {}))
    return t

H._build_trace = build
H.TestTimelineHover.setUpClass()
C = H.TestTimelineHover
def walk(item):                       # delegates are only reachable via the visual tree
    yield item
    for ch in item.childItems():
        yield from walk(ch)
C._selection.navigateTo(1)
QTest.qWait(400)
C._workspace.dismissTimelineOverlay()      # the first-use card covers part of the lanes
QTest.qWait(400)
canvases = [i for i in walk(C._window.contentItem())
            if i.property("laneName") is not None and i.property("syncOverlayLaneIndex") is not None]
lane = next(c for c in canvases if str(c.property("laneName")).startswith("openmp/"))
img = C._window.grabWindow()
p0 = lane.mapToScene(lane.boundingRect().topLeft())
w, h = lane.width(), lane.height()
tl = next(i for i in C._root.findChildren(QQuickItem)
          if i.property("hoverText") is not None and i.property("visibleNs") is not None)
def px(t_ns):
    x = (t_ns - float(tl.property("viewStartNs"))) * w / float(tl.property("visibleNs"))
    c = img.pixelColor(int(p0.x() + x), int(p0.y() + h / 2))
    return [c.red(), c.green(), c.blue()]
tm = C._timeline_model
colors = {s["name"]: s["color"] for li in range(len(tm.lanes))
          for s in tm.laneView(li, tm.viewStartNs, tm.viewStartNs + tm.traceDurationNs, 800, 2000)["spans"]}
print(json.dumps({"busy": px(250_000), "wait": px(600_000), "task": px(750_000), "wait_late": px(900_000),
                  "colors": colors, "lane": str(lane.property("laneName")),
                  "paired": int(lane.property("syncOverlayLaneIndex")), "warnings": C._warnings}))
'''


def _spread(rgb) -> int:
    """0 for a pure grey; colours have a large spread between channels."""
    return max(rgb) - min(rgb)


def _hex(c: str) -> list[int]:
    c = c.lstrip("#")
    return [int(c[i:i + 2], 16) for i in (0, 2, 4)]


def _dist(a, b) -> int:
    return max(abs(x - y) for x, y in zip(a, b))


@unittest.skipUnless(_HAVE_PYSIDE, "PySide6 not installed (optional gui extra)")
class TestTimelineWaitsRender(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        env = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}
        p = subprocess.run([sys.executable, "-c", _SCRIPT, str(REPO)], capture_output=True, text=True,
                           timeout=180, env=env, cwd=str(REPO))
        line = next((l for l in p.stdout.splitlines() if l.startswith("{")), None)
        if line is None:
            raise AssertionError(f"render script failed:\n{p.stdout[-2000:]}\n{p.stderr[-4000:]}")
        cls.r = json.loads(line)

    def test_lane_is_paired_with_its_sync_lane(self):
        self.assertTrue(self.r["lane"].startswith("openmp/thread-"))
        self.assertGreaterEqual(self.r["paired"], 0)
        self.assertEqual(self.r["warnings"], [])

    def test_working_part_is_painted_busy(self):
        region = _hex(self.r["colors"]["omp_parallel_region"])
        self.assertLessEqual(_dist(self.r["busy"], region), 12, self.r)

    def test_wait_is_painted_idle_not_busy(self):
        region = _hex(self.r["colors"]["omp_parallel_region"])
        barrier = _hex(self.r["colors"]["omp_barrier"])
        for key in ("wait", "wait_late"):
            px = self.r[key]
            self.assertGreater(_dist(px, region), 60, f"{key} still looks like the busy region: {self.r}")
            self.assertGreater(_dist(px, barrier), 60, f"{key} painted in the wait's vivid colour: {self.r}")
            self.assertLessEqual(_spread(px), 30, f"{key} is not neutral (idle): {self.r}")

    def test_task_executed_inside_the_wait_stays_visible(self):
        task = _hex(self.r["colors"]["omp_task"])
        self.assertLessEqual(_dist(self.r["task"], task), 12, self.r)


if __name__ == "__main__":
    unittest.main()
