"""
Real-interaction tests for src/gui/qml/screens/TimelineScreen.qml, using a
single shared Main.qml load for the whole module (see TestTimelineHover's
own docstring for why: a second independent QQmlApplicationEngine loading
Main.qml later in the same process was found, empirically, to corrupt Qt
Quick Controls' component resolution under this offscreen QPA platform --
not specific to any one control type, confirmed by reproducing it with two
unrelated components (Fusion style's ToolButton/ButtonPanel, then Basic
style's StatCard) depending on which style got resolved first. All Timeline
QML-interaction coverage therefore lives in ONE test class/engine here,
not split across files the way tests/test_gui_*.py otherwise are.

Originally just the hover regression below. Root cause of THAT bug: the
per-lane Canvas's `laneIndex` property was referenced UNQUALIFIED from its
child MouseArea's onPositionChanged/onExited handlers -- QML does not
resolve a parent item's custom properties by bare name from a nested
child's scope, only via the parent's own `id` (here: `laneCanvas.laneIndex`).
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
    from PySide6.QtCore import QUrl, QObject, QPoint, QPointF, Property, Qt
    from PySide6.QtGui import QGuiApplication, QWheelEvent
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
    # pixel-perfect math needed to pick a hover point. (Scaled up from the
    # original 1000ns to 1_000_000ns, still fully covering the trace
    # either way -- purely to give the nav tests below a large enough
    # range to fit a distinctly-short "target_event" span, see tid=3.)
    trace = Trace(TraceMetadata(command="a.out", args=[]))
    trace.add(SpanEvent(name="fn_on_thread_1", category=Category.CPU,
                         start_ns=0, duration_ns=1_000_000, pid=1, tid=1))
    trace.add(SpanEvent(name="fn_on_thread_2", category=Category.CPU,
                         start_ns=0, duration_ns=1_000_000, pid=1, tid=2))
    # A short, distinct span for the double-click-to-zoom-to-event tests --
    # its own lane so it doesn't interfere with the two full-width hover
    # lanes above.
    trace.add(SpanEvent(name="target_event", category=Category.CPU,
                         start_ns=490_000, duration_ns=20_000, pid=1, tid=3))
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

        cls._flick = cls._root.findChild(QObject, "timelineFlick")
        assert cls._flick is not None

    def setUp(self):
        self._warnings.clear()
        # Nav tests mutate zoom/viewStartNs/contentY/hover state -- start
        # every test from the same known view state regardless of
        # execution order (unittest runs a class's tests alphabetically by
        # default, and e.g. test_double_click_on_span_zooms_to_it leaves
        # the synthesized cursor sitting on target_event's span, which
        # would otherwise leak a nonzero hoverLane into whatever test
        # happens to sort right after it).
        self._timeline_root.setProperty("zoom", 1.0)
        self._timeline_root.setProperty("viewStartNs", self._timeline_model.viewStartNs)
        self._flick.setProperty("contentY", 0)
        self._timeline_root.setProperty("hoverLane", -1)
        self._timeline_root.setProperty("hoverSpanIdx", -1)
        self._timeline_root.setProperty("hoverText", "")
        for _ in range(3):
            self._app.processEvents()

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

    # ── Navigation overhaul: call-graph panel removal, zoom-to-cursor,
    #    keyboard controls, double-click-to-event, vertical-scroll
    #    preservation. Same shared engine/window as the hover tests above
    #    (see module docstring for why this can't be a separate file's
    #    own QQmlApplicationEngine). ──────────────────────────────────────

    def _give_keyboard_focus(self):
        self._timeline_root.forceActiveFocus()
        for _ in range(3):
            self._app.processEvents()

    def test_call_graph_panel_is_gone(self):
        self.assertIsNone(self._root.findChild(QObject, "callGraphCanvas"))

    def test_wheel_zoom_keeps_timestamp_under_cursor_fixed(self):
        # A real synthesized QWheelEvent, not a property write -- proves
        # the actual event wiring (MouseArea.onWheel -> zoomAtFraction),
        # not just the math function in isolation.
        view_start_before = self._timeline_root.property("viewStartNs")
        visible_ns_before = self._timeline_root.property("visibleNs")
        cursor = QPointF(300, 100)
        canvas_w = self._window.width() - 130 - 13
        ns_under_cursor_before = view_start_before + (cursor.x() - 130) / canvas_w * visible_ns_before

        ev = QWheelEvent(cursor, cursor, QPoint(0, 0), QPoint(0, 120),
                          Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False)
        QGuiApplication.sendEvent(self._window, ev)
        for _ in range(5):
            self._app.processEvents()

        zoom_after = self._timeline_root.property("zoom")
        self.assertGreater(zoom_after, 1.0, "wheel-up should have zoomed in")

        view_start_after = self._timeline_root.property("viewStartNs")
        visible_ns_after = self._timeline_root.property("visibleNs")
        ns_under_cursor_after = view_start_after + (cursor.x() - 130) / canvas_w * visible_ns_after
        # The timestamp under the cursor before the zoom should still be
        # (approximately) under the cursor after it -- the whole point of
        # zoom-to-cursor, unlike the old center-anchored behavior.
        self.assertAlmostEqual(ns_under_cursor_before, ns_under_cursor_after, delta=visible_ns_before * 0.02)
        self.assertEqual(self._warnings, [])

    def test_keyboard_right_then_left_pans(self):
        # At zoom 1.0 the whole trace is already visible, so
        # clampViewStart() pins any pan attempt right back -- zoom in
        # first so there's actually room to pan.
        self._timeline_root.setProperty("zoom", 4.0)
        self._app.processEvents()
        self._give_keyboard_focus()
        start0 = self._timeline_root.property("viewStartNs")
        QTest.keyClick(self._window, Qt.Key_Right)
        self._app.processEvents()
        start1 = self._timeline_root.property("viewStartNs")
        self.assertGreater(start1, start0)

        QTest.keyClick(self._window, Qt.Key_Left)
        self._app.processEvents()
        start2 = self._timeline_root.property("viewStartNs")
        self.assertAlmostEqual(start2, start0, delta=1.0)
        self.assertEqual(self._warnings, [])

    def test_keyboard_plus_zooms_in_centered(self):
        self._give_keyboard_focus()
        zoom0 = self._timeline_root.property("zoom")
        QTest.keyClick(self._window, Qt.Key_Plus)
        self._app.processEvents()
        self.assertGreater(self._timeline_root.property("zoom"), zoom0)
        self.assertEqual(self._warnings, [])

    def test_keyboard_0_resets_view_and_vertical_scroll(self):
        self._give_keyboard_focus()
        QTest.keyClick(self._window, Qt.Key_Plus)
        self._flick.setProperty("contentY", 5)
        self._app.processEvents()

        QTest.keyClick(self._window, Qt.Key_0)
        self._app.processEvents()

        self.assertEqual(self._timeline_root.property("zoom"), 1.0)
        self.assertEqual(self._flick.property("contentY"), 0)

    def test_keyboard_home_end_jump_to_trace_bounds(self):
        self._give_keyboard_focus()
        QTest.keyClick(self._window, Qt.Key_Plus)  # zoom in first so Home/End are meaningful
        self._app.processEvents()

        QTest.keyClick(self._window, Qt.Key_End)
        self._app.processEvents()
        visible_ns = self._timeline_root.property("visibleNs")
        expected_end_start = self._timeline_model.viewStartNs + self._timeline_model.traceDurationNs - visible_ns
        self.assertAlmostEqual(self._timeline_root.property("viewStartNs"), expected_end_start, delta=1.0)

        QTest.keyClick(self._window, Qt.Key_Home)
        self._app.processEvents()
        self.assertAlmostEqual(
            self._timeline_root.property("viewStartNs"), self._timeline_model.viewStartNs, delta=1.0)
        self.assertEqual(self._warnings, [])

    def test_zoom_buttons_present_and_work(self):
        # QML Controls types get a synthesized metaobject class name (e.g.
        # "ToolButton_QMLTYPE_12", confirmed empirically -- NOT the C++
        # base class name "QQuickToolButton") with a numeric suffix that
        # isn't stable across runs/Qt versions, hence startswith().
        buttons = [c for c in self._root.findChildren(QObject)
                   if c.metaObject().className().startswith("ToolButton")]
        texts = {b.property("text") for b in buttons}
        self.assertIn("+", texts)
        self.assertIn("−", texts)  # "−"
        self.assertIn("Fit", texts)
        self.assertIn("Reset", texts)

        zoom0 = self._timeline_root.property("zoom")
        plus_btn = next(b for b in buttons if b.property("text") == "+")
        center = plus_btn.mapToScene(QPointF(plus_btn.property("width") / 2, plus_btn.property("height") / 2))
        QTest.mouseClick(self._window, Qt.LeftButton, Qt.NoModifier, center.toPoint())
        self._app.processEvents()
        self.assertGreater(self._timeline_root.property("zoom"), zoom0)

    def test_double_click_on_empty_area_resets(self):
        # 25% across: clearly inside the [1_000_000, 490_000) gap before
        # target_event's lane even starts to matter -- reset baseline is
        # zoom 1.0, so this only needs to avoid the OTHER two lanes' full-
        # width spans, which it does since it targets the SAME x fraction
        # each lane shares (the two full-width spans would be hit too,
        # but this point is deliberately below both lane rows: y=400 is
        # well past the 3-lane block near the top of the canvas).
        empty_point = QPoint(int(130 + (self._window.width() - 130 - 13) * 0.25), 400)
        QTest.mouseDClick(self._window, Qt.LeftButton, Qt.NoModifier, empty_point)
        self._app.processEvents()
        self.assertEqual(self._timeline_root.property("zoom"), 1.0)

    def test_double_click_on_span_zooms_to_it(self):
        # Sweep for the short "target_event" span the same way
        # _find_hover_point sweeps for the hover tests above -- robust to
        # exact label/margin/row-position pixel sizing. Wider y range
        # than _find_hover_point's since target_event is the 3rd lane row.
        found = None
        for y in range(70, 260, 10):
            for x in range(140, int(self._window.width() * 0.9), 20):
                self._move_to(QPoint(x, y))
                if self._timeline_root.property("hoverText").startswith("target_event"):
                    found = QPoint(x, y)
                    break
            if found:
                break
        self.assertIsNotNone(found, "could not hover the target_event span to set up the double-click test")

        zoom_before = self._timeline_root.property("zoom")
        QTest.mouseDClick(self._window, Qt.LeftButton, Qt.NoModifier, found)
        self._app.processEvents()
        self.assertGreater(self._timeline_root.property("zoom"), zoom_before)
        # Centered roughly on the span: its midpoint (500_000ns) should
        # now be within the new (much narrower) visible window.
        view_start = self._timeline_root.property("viewStartNs")
        visible_ns = self._timeline_root.property("visibleNs")
        self.assertTrue(view_start <= 500_000 <= view_start + visible_ns)
        self.assertEqual(self._warnings, [])

    def test_vertical_scroll_preserved_across_zoom_and_pan(self):
        self._flick.setProperty("contentY", 7)
        self._app.processEvents()

        self._give_keyboard_focus()
        QTest.keyClick(self._window, Qt.Key_Plus)
        QTest.keyClick(self._window, Qt.Key_Right)
        self._app.processEvents()

        self.assertEqual(self._flick.property("contentY"), 7)


if __name__ == "__main__":
    unittest.main()
