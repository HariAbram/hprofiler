"""
Per-profile presentation state and table layouts survive a relaunch
(settings.ViewStatePersister): Timeline filters, grouping, colour mode,
hidden/isolated lanes, row order, bookmarks, named ranges, zoom/pan, and
table column layouts. Stale or malformed saved state degrades safely. The
settings file never contains the profiled command line, the trace path, or
span data.

The QML side (TimelineScreen applying the saved zoom/pan) is checked by
rendering Main.qml in a subprocess (one QML engine per process).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

try:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QSettings
    from PySide6.QtGui import QGuiApplication
    _HAVE_PYSIDE = True
except ImportError:
    _HAVE_PYSIDE = False

from src.core.events import Category, SpanEvent
from src.core.trace import Trace, TraceMetadata

SECRET_CMD = "./solver --token=sk-DO-NOT-PERSIST-123"


def _trace() -> Trace:
    t = Trace(TraceMetadata(command=SECRET_CMD, args=["--password=hunter2"]))
    for tid in (1, 2, 3):
        t.add(SpanEvent(f"secret_kernel_{tid}", Category.CPU, 0, 1_000_000, 7, tid, {}))
        t.add(SpanEvent("omp_barrier", Category.SYNC, 500_000, 100_000, 7, tid, {}))
    return t


@unittest.skipUnless(_HAVE_PYSIDE, "PySide6 not installed (optional gui extra)")
class TestViewStatePersistence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QGuiApplication.instance() or QGuiApplication([sys.argv[0]])

    def setUp(self):
        from src.gui.models import TimelineModel
        from src.gui.settings import WorkspaceSettings
        from src.gui.theme import Theme
        self.tmp = tempfile.TemporaryDirectory()
        self.ini = os.path.join(self.tmp.name, "hprofiler.conf")
        self.trace_path = os.path.join(self.tmp.name, "private-project", "run.hprofiler.json")
        self.settings = lambda: WorkspaceSettings(QSettings(self.ini, QSettings.Format.IniFormat))
        self.model = lambda: TimelineModel(_trace(), Theme(dark=True))

    def tearDown(self):
        self.tmp.cleanup()

    def _tables(self):
        from src.gui import columns
        from src.gui.tablemodel import TableConfig
        return {"kernels": TableConfig(columns.KERNEL_COLUMNS)}

    def _configure(self, m, tables):
        lanes = [l["name"] for l in m.lanes]
        m.applyFilters({"nameQuery": "kernel", "minDurationNs": 10, "buckets": ["Computation"]})
        m.setGrouping("process")
        m.setColorMode("bucket")
        m.hideLane(lanes[0])
        m.setRowOrder(list(reversed(lanes)))
        m.addBookmark(250_000.0, "spike")
        m.addNamedRange(100_000.0, 400_000.0, "phase 1")
        m.noteView(8.0, 300_000.0)
        cfg = tables["kernels"]
        cfg.setColumnVisible(cfg.columns[1]["key"], False)
        cfg.moveColumn(0, 2)
        cfg.togglePercentMode()
        return lanes

    def test_round_trip_through_settings(self):
        from src.gui.settings import ViewStatePersister
        m, tables = self.model(), self._tables()
        p = ViewStatePersister(self.settings(), self.trace_path, m, tables)
        lanes = self._configure(m, tables)
        before_cols = tables["kernels"].columns
        p.save()

        m2, tables2 = self.model(), self._tables()
        ViewStatePersister(self.settings(), self.trace_path, m2, tables2)
        self.assertEqual(m2.activeFilters["nameQuery"], "kernel")
        self.assertEqual(m2.activeFilters["buckets"], ["Computation"])
        self.assertEqual(m2.grouping, "process")
        self.assertEqual(m2.colorMode, "bucket")
        self.assertEqual(m2.hiddenLanes, [lanes[0]])
        self.assertEqual([b["name"] for b in m2.bookmarks], ["spike"])
        self.assertEqual([(r["startNs"], r["endNs"], r["name"]) for r in m2.namedRanges],
                         [(100_000.0, 400_000.0, "phase 1")])
        self.assertEqual(m2.restoredView, {"zoom": 8.0, "viewStartNs": 300_000.0})
        self.assertEqual(tables2["kernels"].columns, before_cols)
        self.assertTrue(tables2["kernels"].percentMode)

    def test_state_is_per_profile(self):
        from src.gui.settings import ViewStatePersister
        m = self.model()
        p = ViewStatePersister(self.settings(), self.trace_path, m, {})
        m.setGrouping("process")
        p.save()
        other = self.model()
        ViewStatePersister(self.settings(), self.trace_path + ".other", other, {})
        self.assertEqual(other.grouping, "none")
        self.assertEqual(other.restoredView, {})

    def test_nothing_sensitive_written(self):
        from src.gui.settings import ViewStatePersister
        m, tables = self.model(), self._tables()
        p = ViewStatePersister(self.settings(), self.trace_path, m, tables)
        self._configure(m, tables)
        p.save()
        text = Path(self.ini).read_text()
        for needle in ("sk-DO-NOT-PERSIST", "hunter2", "--token", "solver", "private-project",
                       "run.hprofiler.json", "secret_kernel"):
            self.assertNotIn(needle, text)

    def test_stale_or_malformed_state_degrades_safely(self):
        from src.gui.settings import _storage_key, ViewStatePersister
        qs = QSettings(self.ini, QSettings.Format.IniFormat)
        qs.setValue(f"profiles/{_storage_key(self.trace_path)}/state", json.dumps({"timeline": {
            "filters": {"nameQuery": "(unclosed", "nameIsRegex": True, "minDurationNs": "abc",
                        "threads": "not-a-list"},
            "grouping": "bogus", "colorMode": 42,
            "hiddenLanes": ["no/such-lane", 5], "rowOrder": ["no/such-lane"],
            "bookmarks": [{"ns": 10**18, "name": "outside"}, {"ns": "x"}, {"ns": 5.0, "name": "ok"}],
            "namedRanges": [{"startNs": 9, "endNs": 1}],
            "view": {"zoom": -5, "startOffsetNs": 10**15},
        }}))
        qs.setValue("tables/kernels/config", json.dumps({"order": ["gone", 3], "widths": {"name": -1}}))
        qs.sync()
        m, tables = self.model(), self._tables()
        ViewStatePersister(self.settings(), self.trace_path, m, tables)
        self.assertNotIn("nameIsRegex", m.activeFilters, "an invalid regex must not be re-applied")
        self.assertNotIn("minDurationNs", m.activeFilters)
        self.assertNotIn("threads", m.activeFilters)
        self.assertEqual(m.grouping, "none")
        self.assertEqual(m.hiddenLanes, [])
        self.assertEqual([b["name"] for b in m.bookmarks], ["ok"])
        self.assertEqual(m.namedRanges, [])
        self.assertEqual(m.restoredView, {"zoom": 1.0, "viewStartNs": 1_000_000.0})   # clamped
        from src.gui import columns
        self.assertEqual([c["key"] for c in tables["kernels"].columns], [c.key for c in columns.KERNEL_COLUMNS])

    def test_corrupt_json_means_no_saved_state(self):
        from src.gui.settings import _storage_key, ViewStatePersister
        qs = QSettings(self.ini, QSettings.Format.IniFormat)
        qs.setValue(f"profiles/{_storage_key(self.trace_path)}/state", "{not json")
        qs.sync()
        m = self.model()
        ViewStatePersister(self.settings(), self.trace_path, m, {})
        self.assertEqual(m.grouping, "none")

    def test_restoring_does_not_trigger_a_save(self):
        from src.gui.settings import ViewStatePersister
        m = self.model()
        p = ViewStatePersister(self.settings(), self.trace_path, m, {})
        self.assertFalse(p._timer.isActive())
        m.setGrouping("process")
        self.assertTrue(p._timer.isActive(), "a change schedules a (debounced) save")


_SCRIPT = r'''
import json, os, sys
os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, sys.argv[1])
from PySide6.QtTest import QTest
from PySide6.QtQuick import QQuickItem
import tests.test_gui_timeline_hover as H
from src.gui.models import TimelineModel

class Restored(TimelineModel):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.restore_view_state({"view": {"zoom": 4.0, "startOffsetNs": 250000.0}})

H.TimelineModel = Restored
H.TestTimelineHover.setUpClass()
C = H.TestTimelineHover
C._selection.navigateTo(1)
QTest.qWait(500)
tl = next(i for i in C._root.findChildren(QQuickItem)
          if i.property("hoverText") is not None and i.property("visibleNs") is not None)
noted = C._timeline_model.export_view_state()["view"]
print(json.dumps({"zoom": tl.property("zoom"), "viewStartNs": tl.property("viewStartNs"),
                  "traceStart": C._timeline_model.viewStartNs, "noted": noted}))
'''


@unittest.skipUnless(_HAVE_PYSIDE, "PySide6 not installed (optional gui extra)")
class TestTimelineAppliesSavedView(unittest.TestCase):
    def test_saved_zoom_and_pan_applied_when_the_tab_opens(self):
        p = subprocess.run([sys.executable, "-c", _SCRIPT, str(REPO)], capture_output=True, text=True,
                           timeout=180, env={**os.environ, "QT_QPA_PLATFORM": "offscreen"}, cwd=str(REPO))
        line = next((l for l in p.stdout.splitlines() if l.startswith("{")), None)
        self.assertIsNotNone(line, p.stderr[-3000:])
        r = json.loads(line)
        self.assertEqual(r["zoom"], 4.0)
        self.assertEqual(r["viewStartNs"] - r["traceStart"], 250_000.0)
        self.assertEqual(r["noted"], {"zoom": 4.0, "startOffsetNs": 250_000.0})


if __name__ == "__main__":
    unittest.main()
