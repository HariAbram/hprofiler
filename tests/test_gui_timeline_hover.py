"""
Regression test for a real bug in src/gui/qml/screens/TimelineScreen.qml:
hovering over the Timeline never showed anything, in any GUI session,
since the screen was first built. Root cause: the per-lane Canvas's
`laneIndex` property was referenced UNQUALIFIED from its child
MouseArea's onPositionChanged/onExited handlers -- QML does not resolve
a parent item's custom properties by bare name from a nested child's
scope, only via the parent's own `id` (here: `laneCanvas.laneIndex`).
The bare reference threw a JS ReferenceError on every single hover move,
reported only via engine.warnings (which nothing at runtime was
watching), so it was completely silent: no crash, no visible error, the
hover callback just never did anything useful.

This was never caught by the project's existing static/screenshot-based
QML verification (see project_qml_gui memory) because none of it
synthesized a real mouse move -- this test does, via QTest.mouseMove,
which is the only way this specific bug class (working QML that still
throws at a nested-scope property reference) is actually observable.
Loads the real Main.qml (not TimelineScreen.qml standalone) so the
Timeline gets the same StackLayout-driven sizing it has in the real app,
rather than reproducing that sizing by hand in the test.

Requires a real Qt platform plugin (offscreen is enough -- QTest event
delivery works the same as a real window, just without visible pixels).
Skipped entirely if PySide6 isn't installed, matching this project's
other GUI tests.
"""
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QUrl, QObject, QPoint, Property
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtQml import QQmlApplicationEngine, qmlRegisterSingletonInstance
    from PySide6.QtQuick import QQuickWindow, QQuickItem
    from PySide6.QtTest import QTest
    _PYSIDE6_AVAILABLE = True
except ImportError:
    _PYSIDE6_AVAILABLE = False

if _PYSIDE6_AVAILABLE:
    from src.core.trace import Trace, TraceMetadata
    from src.core.events import SpanEvent, Category
    from src.gui.theme import Theme
    from src.gui.models import TimelineModel
    from src.gui.bridge import (
        DashboardBridge, KernelsBridge, CallTreeBridge, RooflineBridge, SourceBridge,
        SystemBridge, ProfileBridge,
    )

    class _AppInfo(QObject):
        @Property(str, constant=True)
        def commandLine(self) -> str:
            return "test"

        @Property(str, constant=True)
        def tracePath(self) -> str:
            return "test"

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MAIN_QML = _REPO_ROOT / "src" / "gui" / "qml" / "Main.qml"


def _build_trace() -> "Trace":
    # Two lanes, one span each, each spanning the ENTIRE trace duration --
    # at the default zoom (1.0x) each covers the full canvas width, so any
    # x position within a lane's row reliably hits its span. No fragile
    # pixel-perfect math needed to pick a hover point.
    trace = Trace(TraceMetadata(command="a.out", args=[]))
    trace.add(SpanEvent(name="fn_on_thread_1", category=Category.CPU,
                         start_ns=0, duration_ns=1000, pid=1, tid=1))
    trace.add(SpanEvent(name="fn_on_thread_2", category=Category.CPU,
                         start_ns=0, duration_ns=1000, pid=1, tid=2))
    return trace


@unittest.skipUnless(_PYSIDE6_AVAILABLE, "PySide6 not installed (optional gui extra)")
class TestTimelineHover(unittest.TestCase):
    """One shared QGuiApplication + one engine/window for the whole class
    -- qmlRegisterSingletonInstance registers into the process-global QML
    type system and PySide6 only tolerates one QGuiApplication per
    process, so re-registering per test caused cross-test interference."""

    @classmethod
    def setUpClass(cls):
        cls._app = QGuiApplication.instance() or QGuiApplication([sys.argv[0]])
        cls._trace = _build_trace()

        # Bridge instances MUST be kept alive via a real Python reference
        # (here: class attributes) for as long as the engine/window uses
        # them -- passing them inline to qmlRegisterSingletonInstance with
        # no stored reference lets them get garbage-collected almost
        # immediately, which is exactly what "singleton has already been
        # deleted" / "returns a null pointer" below means. This is the
        # same PySide6 lifetime gotcha documented in project_qml_gui
        # memory for src/gui/app.py, reproduced here by first getting it
        # wrong the same way.
        cls._theme = Theme(dark=True)
        cls._app_info = _AppInfo()
        cls._dashboard = DashboardBridge(cls._trace, cls._theme)
        cls._timeline_model = TimelineModel(cls._trace, cls._theme)
        cls._kernels = KernelsBridge(cls._trace, cls._theme)
        cls._call_tree = CallTreeBridge(cls._trace, cls._theme)
        cls._roofline = RooflineBridge(cls._trace)
        cls._source = SourceBridge(cls._trace)
        cls._system = SystemBridge(cls._trace)
        cls._profile = ProfileBridge(cls._trace, cls._theme)

        qmlRegisterSingletonInstance(Theme, "Hprofiler", 1, 0, "AppTheme", cls._theme)
        qmlRegisterSingletonInstance(_AppInfo, "Hprofiler", 1, 0, "AppInfo", cls._app_info)
        qmlRegisterSingletonInstance(DashboardBridge, "Hprofiler", 1, 0, "Dashboard", cls._dashboard)
        qmlRegisterSingletonInstance(TimelineModel, "Hprofiler", 1, 0, "TimelineModel", cls._timeline_model)
        qmlRegisterSingletonInstance(KernelsBridge, "Hprofiler", 1, 0, "Kernels", cls._kernels)
        qmlRegisterSingletonInstance(CallTreeBridge, "Hprofiler", 1, 0, "CallTree", cls._call_tree)
        qmlRegisterSingletonInstance(RooflineBridge, "Hprofiler", 1, 0, "Roofline", cls._roofline)
        qmlRegisterSingletonInstance(SourceBridge, "Hprofiler", 1, 0, "Source", cls._source)
        qmlRegisterSingletonInstance(SystemBridge, "Hprofiler", 1, 0, "System", cls._system)
        qmlRegisterSingletonInstance(ProfileBridge, "Hprofiler", 1, 0, "Profile", cls._profile)

        cls._engine = QQmlApplicationEngine()
        cls._warnings: list[str] = []
        cls._engine.warnings.connect(lambda ws: cls._warnings.extend(str(w) for w in ws))
        cls._engine.load(QUrl.fromLocalFile(str(_MAIN_QML)))
        assert cls._engine.rootObjects(), f"Main.qml failed to load: {cls._warnings}"
        cls._root = cls._engine.rootObjects()[0]

        cls._window = None
        for w in cls._app.allWindows():
            if isinstance(w, QQuickWindow):
                cls._window = w
                break
        assert cls._window is not None

        tab_bar = cls._root.findChild(QObject, "tabBar")
        tab_bar.setProperty("currentIndex", 1)  # "2 Timeline"
        for _ in range(5):
            cls._app.processEvents()

        cls._timeline_root = None
        for child in cls._root.findChildren(QQuickItem):
            if child.property("hoverText") is not None and child.property("visibleNs") is not None:
                cls._timeline_root = child
                break
        assert cls._timeline_root is not None, "could not locate TimelineScreen's root item"

    def setUp(self):
        self._warnings.clear()

    def _move_to(self, point: QPoint) -> None:
        QTest.mouseMove(self._window, point)
        for _ in range(3):
            self._app.processEvents()

    def _find_hover_point(self) -> QPoint | None:
        # Sweep x/y over a generous range rather than compute one exact
        # pixel -- robust to header/tab-bar/margin sizing this test
        # shouldn't need to know precisely, while still proving the
        # real hover path (mouse event -> MouseArea -> laneIndex ->
        # TimelineModel.spanAt -> hoverText) works end to end. Both
        # lanes' spans cover the FULL trace duration at zoom 1.0x, so
        # any x within a lane row hits something.
        for y in range(70, 200, 10):
            for x in (150, 400, 700, 1000):
                self._move_to(QPoint(x, y))
                if self._timeline_root.property("hoverLane") >= 0:
                    return QPoint(x, y)
        return None

    def test_hovering_over_a_span_reports_it_with_no_warnings(self):
        self.assertEqual(self._timeline_root.property("hoverLane"), -1)

        hit_point = self._find_hover_point()

        self.assertEqual(
            self._warnings, [],
            f"QML runtime warnings during hover (this is how the laneIndex "
            f"ReferenceError bug showed up): {self._warnings}",
        )
        self.assertIsNotNone(hit_point, "hovering never registered a hit anywhere in the swept area")
        self.assertIn("fn_on_thread_", self._timeline_root.property("hoverText"))

    def test_moving_off_the_lanes_clears_hover(self):
        hit_point = self._find_hover_point()
        self.assertIsNotNone(hit_point, "setup: hovering never registered a hit anywhere")
        self.assertGreaterEqual(self._timeline_root.property("hoverLane"), 0)

        # Off the bottom of the window entirely -- past both lane rows.
        self._move_to(QPoint(hit_point.x(), 780))

        self.assertEqual(self._warnings, [])
        self.assertEqual(self._timeline_root.property("hoverLane"), -1)
        self.assertEqual(self._timeline_root.property("hoverText"), "")


if __name__ == "__main__":
    unittest.main()
