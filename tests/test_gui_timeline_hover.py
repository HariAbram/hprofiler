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
    from PySide6.QtGui import QGuiApplication, QWheelEvent, QColor
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
        SystemBridge, ProfileBridge, FlameGraphBridge,
    )
    from src.gui.nav import Selection
    from src.gui.inspector import InspectorBridge

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
    # stack_frames so FlameGraphBridge/CallTreeBridge build a real
    # (non-empty) tree -- needed for the FlameGraph theme-toggle
    # regression test, which has to hover an actual frame.
    trace.add(SpanEvent(name="flame_leaf", category=Category.CPU,
                         start_ns=0, duration_ns=1_000_000, pid=1, tid=4,
                         stack_frames=["flame_root"]))
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
        cls._flame_graph = FlameGraphBridge(cls._trace, cls._theme)
        cls._selection = Selection()
        cls._inspector = InspectorBridge(
            cls._trace, cls._selection, cls._kernels, cls._call_tree,
            cls._roofline, cls._source, cls._timeline_model,
        )

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
        qmlRegisterSingletonInstance(FlameGraphBridge, "Hprofiler", 1, 0, "FlameGraph", cls._flame_graph)
        qmlRegisterSingletonInstance(Selection, "Hprofiler", 1, 0, "Nav", cls._selection)
        qmlRegisterSingletonInstance(InspectorBridge, "Hprofiler", 1, 0, "Inspector", cls._inspector)

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

        cls._tab_bar = cls._root.findChild(QObject, "tabBar")
        cls._tab_bar.setProperty("currentIndex", 1)  # "2 Timeline"
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
        # Some tests (e.g. the tab-cycling and FlameGraph theme-toggle
        # checks below) switch tabs away from Timeline -- restore it
        # first so every OTHER test's real QTest mouse/keyboard events
        # (which only hit whatever StackLayout page is actually visible)
        # land on TimelineScreen regardless of what a previous test left
        # active. Cheap even when already on Timeline.
        self._tab_bar.setProperty("currentIndex", 1)
        for _ in range(3):
            self._app.processEvents()
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
        # Nav/Inspector are the SAME kind of shared, class-level state as
        # the Timeline view properties above -- cross-tab-navigation tests
        # mutate selection/breadcrumbs/inspectorOpen, so those need
        # resetting between tests too, or execution order would leak
        # state (e.g. a leftover breadcrumb from one test changing
        # goBack()'s behavior in the next). clearSelection() covers
        # selection/call-path/time-range/thread; breadcrumbs has no public
        # reset (Selection never needs one outside tests), so it's reset
        # directly here.
        self._selection.clearSelection()
        self._selection._breadcrumbs = []
        self._selection.breadcrumbsChanged.emit()
        if not self._selection.inspectorOpen:
            self._selection.toggleInspector()
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

    # ── Visual-consistency audit: cross-screen smoke test + the
    #    FlameGraph theme-toggle regression, direct not just visual ────

    def test_every_tab_loads_without_warnings(self):
        # Cheap, catches a broken "../components" import or missing
        # property across every migrated screen at once -- each of the
        # 9 tabs' Loader activates for the first time here (most were
        # never visited by any other test in this class).
        for i in range(9):
            self._warnings.clear()
            self._tab_bar.setProperty("currentIndex", i)
            for _ in range(8):
                self._app.processEvents()
            self.assertEqual(self._warnings, [], f"tab {i} produced QML warnings: {self._warnings}")

    def test_flame_graph_tooltip_repaints_on_theme_toggle(self):
        # Direct regression test for the audit's headline bug:
        # FlameGraphScreen's tooltip used to be built from 8 hardcoded
        # hex literals that never repainted on the light/dark toggle.
        # Checks the actual bound QColor property, not a screenshot --
        # screenshots of this exact tooltip were visually misjudged
        # (misread as "still dark") during this fix's own verification,
        # while property introspection was unambiguous; this test uses
        # the reliable method.
        self._tab_bar.setProperty("currentIndex", 4)  # Flame Graph
        for _ in range(10):
            self._app.processEvents()

        canvas = self._root.findChild(QQuickItem, "flameGraphCanvas")
        self.assertIsNotNone(canvas, "flameGraphCanvas not found")
        fg_root = None
        for child in self._root.findChildren(QQuickItem):
            if child.property("zoomStack") is not None:
                fg_root = child
                break
        self.assertIsNotNone(fg_root, "FlameGraphScreen root (zoomStack) not found")

        tooltip = None
        for child in self._root.findChildren(QQuickItem):
            if child.property("followCursor") is not None:
                tooltip = child
                break
        self.assertIsNotNone(tooltip, "Tooltip instance not found")

        # Bottom row, well inside the canvas -- the root frame spans the
        # full width at any zoom, so this reliably hits something.
        target_local = QPoint(int(canvas.width() * 0.3), int(canvas.height() - 8))
        # A SINGLE mouseMove teleporting straight to the target does not
        # reliably register Qt Quick hover-enter as the first-ever
        # synthetic mouse event in the process -- confirmed real and
        # reproducible (not a one-off), and NOT fixed by more
        # processEvents() alone, explicit window activation, or a
        # trajectory through unrelated screen regions; only a trajectory
        # that stays within the target item's own area reliably worked,
        # and even that was not 100% deterministic across repeated runs
        # under the offscreen QPA platform. Retrying the sweep a few
        # times (cheap, and this loop exits the instant a real hover
        # registers) makes this robust without weakening what's actually
        # asserted -- it still requires a real, successful hover.
        for attempt in range(5):
            for step in range(1, 6):
                local = QPoint(target_local.x(), int(target_local.y() * step / 5))
                self._move_to(canvas.mapToScene(local).toPoint())
            if tooltip.property("visible"):
                break
        self.assertTrue(tooltip.property("visible"),
                         "tooltip never became visible after repeated hover attempts")

        dark_color = tooltip.property("color")
        self._theme.toggle()
        for _ in range(10):
            self._app.processEvents()
        light_color = tooltip.property("color")

        self.assertNotEqual(dark_color, light_color)
        self.assertEqual(light_color, QColor(self._theme.background))
        self._theme.toggle()  # back to dark for whatever test runs next
        for _ in range(5):
            self._app.processEvents()
        self.assertEqual(self._warnings, [])

    # ── Cross-tab navigation: shared selection, inspector, breadcrumbs ──
    #    Same shared engine/window as every test above -- see module
    #    docstring for why. Nav/Inspector state is reset in setUp().

    def test_kernel_click_updates_nav_selection(self):
        self._tab_bar.setProperty("currentIndex", 2)  # Kernels
        for _ in range(5):
            self._app.processEvents()

        # Scope to tabLoader2's OWN loaded item, not the whole root tree:
        # a bare root.findChildren(QQuickListView) match TabBar's own
        # internal ListView first under the Basic style (its contentItem
        # is a ListView of TabButtons) -- a real false positive that's
        # visible and populated (count=9) just like Kernels' actual data
        # list, found the hard way while building this test.
        kernels_loader = self._root.findChild(QObject, "tabLoader2")
        self.assertIsNotNone(kernels_loader)
        kernels_screen = kernels_loader.property("item")
        self.assertIsNotNone(kernels_screen)
        lists = [c for c in kernels_screen.findChildren(QQuickItem)
                 if c.metaObject().className().startswith("QQuickListView")
                 and c.property("visible") and (c.property("count") or 0) > 0]
        self.assertTrue(lists, "could not find Kernels' populated ListView")
        kernels_list = lists[0]

        top_left = kernels_list.mapToScene(kernels_list.boundingRect().topLeft()).toPoint()
        QTest.mouseClick(self._window, Qt.LeftButton, Qt.NoModifier, top_left + QPoint(60, 14))
        for _ in range(5):
            self._app.processEvents()

        self.assertNotEqual(self._selection.selectedName, "")
        self.assertEqual(self._warnings, [])
        # Inspector reacts to the same selection -- Summary always has at
        # least Name/Category, confirming the signal chain (QML click ->
        # Nav.selectFunction -> InspectorBridge._recompute) fired for real,
        # not just that the Python-level Selection object changed.
        self.assertTrue(len(self._inspector.content["summary"]) >= 2)

    def test_navigateTo_pushes_breadcrumb_and_goBack_restores(self):
        self._selection.selectFunction("cpu", "fn_on_thread_1")
        self.assertEqual(self._selection.breadcrumbs, [])

        self._selection.navigateTo(7)  # System
        for _ in range(5):
            self._app.processEvents()
        self.assertEqual(self._selection.currentTab, 7)
        self.assertEqual(len(self._selection.breadcrumbs), 1)
        self.assertEqual(self._tab_bar.property("currentIndex"), 7,
                          "TabBar didn't follow Nav.currentTab after navigateTo()")

        back_btn = None
        for c in self._root.findChildren(QObject):
            if c.metaObject().className().startswith("ToolButton") and c.property("text") == "← Back":
                back_btn = c
                break
        self.assertIsNotNone(back_btn, "Inspector's Back button not found")
        self.assertTrue(back_btn.property("visible"))

        self._selection.goBack()
        for _ in range(5):
            self._app.processEvents()
        self.assertEqual(self._selection.currentTab, 1)
        self.assertEqual(self._selection.selectedName, "fn_on_thread_1")
        self.assertEqual(self._selection.breadcrumbs, [])
        self.assertEqual(self._tab_bar.property("currentIndex"), 1)
        self.assertEqual(self._warnings, [])

    def test_plain_tab_click_does_not_push_breadcrumb(self):
        # The established rule (see src/gui/nav.py's Selection.navigateTo
        # docstring): only an explicit navigateTo() call records a
        # breadcrumb. Ordinary browsing -- a direct TabBar click, wired
        # as a plain currentIndex write, not routed through navigateTo --
        # must not.
        self._tab_bar.setProperty("currentIndex", 3)
        for _ in range(5):
            self._app.processEvents()
        self.assertEqual(self._selection.currentTab, 3)
        self.assertEqual(self._selection.breadcrumbs, [])

    def test_calltree_node_click_selects_and_keeps_expand_working(self):
        self._tab_bar.setProperty("currentIndex", 3)  # Call Tree
        for _ in range(5):
            self._app.processEvents()

        ct_loader = self._root.findChild(QObject, "tabLoader3")
        self.assertIsNotNone(ct_loader)
        ct_screen = ct_loader.property("item")
        self.assertIsNotNone(ct_screen)

        # Repeater-created TreeNode delegates are QObject-parented to the
        # Repeater itself for lifecycle management, not to their visual
        # parent Item (QQuickItem::parentItem() and QObject::parent() are
        # separate trees in Qt Quick) -- findChildren() walks the QOBJECT
        # tree and structurally cannot reach them, confirmed by direct
        # comparison against a real screenshot showing fully-rendered rows
        # while findChildren() returned none. A coordinate sweep (the same
        # technique _find_hover_point already uses for Timeline) sidesteps
        # this rather than hunting for an Item handle that can't be found
        # this way.
        #
        # Priming mouseMoves stepping up to the point, THEN an explicit
        # mousePress + processEvents + mouseRelease -- confirmed by direct
        # repro that a bare QTest.mouseClick (and even mousePress/
        # mouseRelease with no priming move) unreliably fails to register
        # on this Flickable-wrapped, Loader/Repeater-recursed row, while
        # this exact combination (the same family as this file's
        # documented first-mouseMove hover lesson, extended here to click
        # delivery) registers reliably across repeated attempts.
        def _click(pt: QPoint) -> None:
            start = pt - QPoint(0, 20)
            for step in range(1, 5):
                self._move_to(start + (pt - start) * step / 4)
            QTest.mousePress(self._window, Qt.LeftButton, Qt.NoModifier, pt)
            self._app.processEvents()
            QTest.mouseRelease(self._window, Qt.LeftButton, Qt.NoModifier, pt)

        panel_top_left = ct_screen.mapToScene(QPointF(0, 0)).toPoint()
        hit = False
        # Outer retry around the whole sweep, same rationale as the
        # FlameGraph tooltip test's repeated hover attempts: this specific
        # synthetic-event path was observed to succeed reliably in an
        # interactive repro but still occasionally miss every point on a
        # single pass under the offscreen QPA platform when run through
        # the full unittest harness -- cheap, and exits the instant a
        # click actually registers.
        for _attempt in range(3):
            for y_off in range(30, 140, 6):
                self._selection.clearSelection()
                for _ in range(2):
                    self._app.processEvents()
                _click(panel_top_left + QPoint(80, y_off))
                for _ in range(3):
                    self._app.processEvents()
                if self._selection.selectedName != "":
                    hit = True
                    break
            if hit:
                break
        self.assertTrue(hit, "clicking never selected a call-tree row anywhere in the swept area")
        self.assertEqual(self._warnings, [])

        # Root-level call-tree data built from a single-frame stack (see
        # _build_trace's flame_leaf span): the outermost row is the
        # synthetic frame name, not the span's own name/category.
        selected_before = (self._selection.selectedCategory, self._selection.selectedName)

        # Clicking the SAME point again toggles expand/collapse (existing
        # behavior, unchanged) while re-selecting the same node -- proves
        # the new Nav.selectFunction() call didn't replace the old
        # root.expanded = !root.expanded toggle, both still fire together.
        _click(panel_top_left + QPoint(80, y_off))
        for _ in range(3):
            self._app.processEvents()
        self.assertEqual((self._selection.selectedCategory, self._selection.selectedName), selected_before)
        self.assertEqual(self._warnings, [])

    def test_timeline_click_selects_and_double_click_still_zooms(self):
        # Reuses the same target_event sweep as
        # test_double_click_on_span_zooms_to_it.
        found = None
        for y in range(70, 260, 10):
            for x in range(140, int(self._window.width() * 0.9), 20):
                self._move_to(QPoint(x, y))
                if self._timeline_root.property("hoverText").startswith("target_event"):
                    found = QPoint(x, y)
                    break
            if found:
                break
        self.assertIsNotNone(found, "could not hover target_event to set up this test")

        QTest.mouseClick(self._window, Qt.LeftButton, Qt.NoModifier, found)
        for _ in range(5):
            self._app.processEvents()
        self.assertEqual(self._selection.selectedName, "target_event")
        self.assertEqual(self._selection.selectedThread.get("tid"), 3)

        zoom_before = self._timeline_root.property("zoom")
        QTest.mouseDClick(self._window, Qt.LeftButton, Qt.NoModifier, found)
        for _ in range(5):
            self._app.processEvents()
        self.assertGreater(self._timeline_root.property("zoom"), zoom_before,
                            "double-click-to-zoom should still work after adding click-to-select")
        self.assertEqual(self._warnings, [])

    def test_inspector_panel_toggles(self):
        panel = self._root.findChild(QObject, "inspectorPanel")
        self.assertIsNotNone(panel)
        reopen_btn = None
        for c in panel.findChildren(QObject):
            if c.metaObject().className().startswith("ToolButton") and c.property("text") == "◀":
                reopen_btn = c
                break
        self.assertIsNotNone(reopen_btn, "collapsed-state reopen button not found")

        self.assertTrue(self._selection.inspectorOpen)
        self.assertFalse(reopen_btn.property("visible"))

        self._selection.toggleInspector()
        for _ in range(5):
            self._app.processEvents()
        self.assertFalse(self._selection.inspectorOpen)
        self.assertTrue(reopen_btn.property("visible"))

        self._selection.toggleInspector()
        for _ in range(5):
            self._app.processEvents()
        self.assertTrue(self._selection.inspectorOpen)
        self.assertFalse(reopen_btn.property("visible"))
        self.assertEqual(self._warnings, [])

    def test_uncorrelated_selection_reports_unavailable_gracefully(self):
        # A (category,name) guaranteed to match nothing in the fixture
        # trace -- every section should degrade to an honest
        # "unavailable" with a specific reason, never a crash/warning or
        # a silently blank field.
        self._selection.selectFunction("other", "totally_fake_uncorrelated_name")
        for _ in range(5):
            self._app.processEvents()
        self.assertEqual(self._warnings, [])

        content = self._inspector.content
        summary_kinds = {f["label"]: f["kind"] for f in content["summary"]}
        self.assertEqual(summary_kinds.get("Total time"), "unavailable")

        for section in ("context", "relationships", "recommendations"):
            for field in content[section]:
                self.assertIn(field["kind"], ("measured", "derived", "estimated", "unavailable"))
                if field["kind"] == "unavailable":
                    self.assertTrue(field["reason"], f"{section}.{field['label']} unavailable with no reason given")


if __name__ == "__main__":
    unittest.main()
