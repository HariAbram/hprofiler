"""
Real-interaction tests for src/gui/qml/screens/TimelineScreen.qml (and the
other screens' QML interaction), using a single shared Main.qml load for
the whole module: a second independent QQmlApplicationEngine loading
Main.qml in the same process corrupts Qt Quick Controls' component
resolution under the offscreen QPA platform (reproduced with unrelated
components -- Fusion's ToolButton/ButtonPanel, Basic's StatCard -- depending
on which style resolved first). All QML-interaction coverage therefore
lives in ONE test class/engine here, not split across files the way
tests/test_gui_*.py otherwise are.

The hover tests guard a nested-scope bug class: the per-lane Canvas's
`laneIndex` referenced UNQUALIFIED from its child MouseArea's handlers.
QML does not resolve a parent item's custom properties by bare name from a
nested child's scope, only via the parent's `id` (`laneCanvas.laneIndex`);
the bare reference throws a JS ReferenceError on every hover move,
reported only via engine.warnings -- no crash, no visible error. Only a
synthesized mouse move (QTest.mouseMove) makes this observable, so these
tests use real input, not static rendering checks. Loads the real Main.qml
(not TimelineScreen.qml standalone) so the Timeline gets the same
StackLayout-driven sizing as the real app.

Requires a Qt platform plugin (offscreen is enough -- QTest event delivery
works the same as a real window, just without visible pixels). Skipped if
PySide6 isn't installed, like the other GUI tests.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QUrl, QObject, QPoint, QPointF, Property, Qt, QSettings
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
    from src.gui.tablemodel import FormatBridge
    from src.gui.comparison import ComparisonBridge
    from src.gui.settings import WorkspaceSettings, WorkspaceBridge
    from src.gui.controller import AppController
    from src.gui.shortcuts import ShortcutsBridge

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
    # Many extra lanes for the row-virtualization test -- STRICTLY LARGER
    # tids than the ones above (100+), since TimelineModel's tid_seq
    # renumbering sorts all thread tids numerically before assigning
    # sequence numbers, so these always sort AFTER tid 1-4 and never
    # shift the existing hover-dependent lanes' row positions.
    for extra_tid in range(100, 140):
        trace.add(SpanEvent(name=f"extra_fn_{extra_tid}", category=Category.CPU,
                             start_ns=0, duration_ns=1000, pid=1, tid=extra_tid))
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
        # same PySide6 lifetime requirement src/gui/app.py follows.
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
        cls._formatter = FormatBridge()
        # trace_b=None here -- this shared engine can only ever hold ONE
        # trace pairing for the whole test run, so it proves the "no
        # comparison loaded" case (all 10 tabs load clean with zero
        # warnings, the regression that matters most for ordinary single-
        # profile users). A real second-trace comparison is covered
        # separately by tests/test_gui_compare_launch.py, which spawns a
        # genuine subprocess instead.
        cls._comparison = ComparisonBridge(cls._trace, None, cls._theme)
        # Backed by a throwaway temp-file QSettings, never the real user
        # config -- matches test_gui_settings.py's own convention. Main.qml
        # reads Workspace.hasSavedGeometry/windowX/etc unconditionally at
        # startup now (window geometry persistence), so this singleton
        # MUST be registered for Main.qml to load without warnings here,
        # same as in the real app.py bootstrap.
        cls._settings_tmpdir = tempfile.TemporaryDirectory()
        cls._workspace_settings = WorkspaceSettings(
            QSettings(os.path.join(cls._settings_tmpdir.name, "settings.ini"), QSettings.Format.IniFormat))
        cls._workspace = WorkspaceBridge(cls._workspace_settings)
        cls._app_controller = AppController("test-trace.json")
        cls._shortcuts = ShortcutsBridge()

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
        qmlRegisterSingletonInstance(FormatBridge, "Hprofiler", 1, 0, "Format", cls._formatter)
        qmlRegisterSingletonInstance(ComparisonBridge, "Hprofiler", 1, 0, "Compare", cls._comparison)
        qmlRegisterSingletonInstance(WorkspaceBridge, "Hprofiler", 1, 0, "Workspace", cls._workspace)
        qmlRegisterSingletonInstance(AppController, "Hprofiler", 1, 0, "App", cls._app_controller)
        qmlRegisterSingletonInstance(ShortcutsBridge, "Hprofiler", 1, 0, "Shortcuts", cls._shortcuts)

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

    @classmethod
    def tearDownClass(cls):
        cls._settings_tmpdir.cleanup()

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
        # DataTable's own hover-definition tooltip (see DataTableHeader's
        # onHoverDefinition) has no automatic "mouse left the window"
        # reset the way a real cursor move away would give it -- a prior
        # test's hover-priming can leave it visible for whatever test
        # sorts next alphabetically. Reset directly rather than relying on
        # every future hover test to scrupulously move the mouse away.
        kernels_tooltip = self._root.findChild(QObject, "dataTableTooltip_Kernels")
        if kernels_tooltip is not None:
            kernels_tooltip.setProperty("visible", False)
        # Timeline filters are process-lifetime model state too -- a test
        # that applies one and forgets to clear it would leak into e.g.
        # test_row_list_scales_to_many_lanes...'s hardcoded len(rows)==44
        # assertion, depending on alphabetical test order.
        self._timeline_model.clearFilters()
        filter_popup = self._root.findChild(QObject, "timelineFilterPopup")
        if filter_popup is not None and filter_popup.property("visible"):
            filter_popup.setProperty("visible", False)
        # Grouping/hide/isolate: same process-lifetime state, same reset.
        self._timeline_model.setGrouping("none")
        self._timeline_model.showAllLanes()
        group_popup = self._root.findChild(QObject, "timelineGroupPopup")
        if group_popup is not None and group_popup.property("visible"):
            group_popup.setProperty("visible", False)
        # Search/color-mode: same process-lifetime state reset.
        self._timeline_model.clearSearch()
        self._timeline_model.setColorMode("function")
        search_field = self._root.findChild(QObject, "timelineSearchField")
        if search_field is not None and search_field.property("text"):
            search_field.setProperty("text", "")
        # Bookmarks/named ranges: same process-lifetime state reset.
        for b in list(self._timeline_model.bookmarks):
            self._timeline_model.removeBookmark(b["id"])
        for r in list(self._timeline_model.namedRanges):
            self._timeline_model.removeNamedRange(r["id"])
        for _ in range(3):
            self._app.processEvents()

    def _move_to(self, point: QPoint) -> None:
        QTest.mouseMove(self._window, point)
        for _ in range(3):
            self._app.processEvents()

    def _stable_center(self, item: QObject) -> QPoint:
        """`item`'s scene-space center, waited for RowLayout/ColumnLayout
        geometry polish to settle first: reading mapToScene() right after a
        sibling's visibility/text change (e.g. a status Text becoming
        visible, which resizes the row and shifts every button after it)
        can return a STALE pre-layout position where several siblings
        report the SAME point. processEvents() calls (however many) don't
        reliably flush it -- Qt Quick's layout polish is tied to the
        render/timer loop, which advances only when wall-clock time passes
        (QTest.qWait). A click computed from the stale position can land on
        a different, overlapping neighbor (e.g. "Clear" instead of "Next"
        in the search bar; ~50% of runs with a processEvents()-only wait,
        0/5 with qWait). Polls until two consecutive reads agree -- a real
        settle condition, not a guessed iteration count."""
        prev = None
        for _ in range(40):
            QTest.qWait(10)
            cur = item.mapToScene(QPointF(item.property("width") / 2,
                                           item.property("height") / 2)).toPoint()
            if prev is not None and cur == prev:
                return cur
            prev = cur
        return prev

    def _find_hover_point(self) -> QPoint | None:
        # Sweep x/y over a generous range rather than compute one exact
        # pixel -- robust to header/tab-bar/margin sizing this test
        # shouldn't need to know precisely, while still proving the
        # real hover path (mouse event -> MouseArea -> laneIndex ->
        # TimelineModel.spanAt -> hoverText) works end to end. Both
        # lanes' spans cover the FULL trace duration at zoom 1.0x, so
        # any x within a lane row hits something. Range starts at 150:
        # the lanes area sits below the bookmarks/legend row and TimeRuler.
        for y in range(150, 300, 10):
            for x in (150, 400, 700, 1000):
                self._move_to(QPoint(x, y))
                if self._timeline_root.property("hoverLane") >= 0:
                    return QPoint(x, y)
        return None

    def _find_target_event_point(self) -> QPoint | None:
        """Sweeps for the short "target_event" span (3rd lane row),
        shared by both the double-click-to-zoom and click-to-select
        tests. Wrapped in a bounded outer retry: with the fixture's many
        lanes (see _build_trace()), the per-move Canvas repaint cost makes
        a single-pass sweep occasionally miss -- synthetic input isn't 100%
        deterministic on the first pass under offscreen QPA (same remedy as
        the FlameGraph-tooltip and Call-Tree-click tests). The range starts
        at 150 and extends to 420: the lanes area sits below the
        bookmarks/legend row + TimeRuler, and under grouping target_event's
        row can sit several rows deeper than in the flat case."""
        for _attempt in range(3):
            for y in range(150, 420, 10):
                for x in range(140, int(self._window.width() * 0.9), 20):
                    self._move_to(QPoint(x, y))
                    if self._timeline_root.property("hoverText").startswith("target_event"):
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
        # y=200: the lanes area sits below the bookmarks/legend row and
        # TimeRuler.
        cursor = QPointF(300, 200)
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
        # zoom-to-cursor, as opposed to center-anchored zoom.
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
        found = self._find_target_event_point()
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

    # ── Table/Timeline-upgrade round: row-list virtualization ───────────

    def test_row_list_scales_to_many_lanes_and_scrolls_past_the_viewport(self):
        # _build_trace() carries 44 lanes total (4 original + 40 extra,
        # tids 100-139, added specifically for this test -- see that
        # function's own comment on why strictly-larger tids never shift
        # the other tests' lane positions). A plain Repeater-per-lane
        # would still render all 54 Canvases regardless of viewport size;
        # this doesn't assert an exact instantiated-delegate count
        # (Repeater/ListView-created items are not reliably reachable via
        # findChildren() under PySide6 -- confirmed repeatedly elsewhere
        # in this file) but does assert the real, structurally meaningful
        # thing: the model has far more rows than fit on screen, the
        # ListView's contentHeight reflects ALL of them, scrolling past
        # the viewport works, and none of this produces a single warning
        # -- exactly the condition under which a non-virtualized Repeater
        # would visibly slow down or, at minimum, do a lot of unnecessary
        # paint work.
        self.assertEqual(len(self._timeline_model.rows), 44)
        row_height = self._timeline_root.property("rowHeight")
        content_height = self._flick.property("contentHeight")
        viewport_height = self._flick.property("height")
        self.assertAlmostEqual(content_height, 44 * row_height, delta=1.0)
        self.assertLess(viewport_height, content_height,
                         "fixture should have more lanes than fit in one viewport")

        max_scroll = content_height - viewport_height
        self._flick.setProperty("contentY", max_scroll)
        for _ in range(5):
            self._app.processEvents()
        self.assertAlmostEqual(self._flick.property("contentY"), max_scroll, delta=1.0)
        self.assertEqual(self._warnings, [])
        self._flick.setProperty("contentY", 0)

    # ── Timeline filters ─────────────────────────────────────────────

    def test_timeline_filter_button_opens_and_closes_popup(self):
        # The filter button itself is a static child (not Repeater-
        # created), so it's reliably reachable via findChild the same way
        # the zoom/reset ToolButtons already proven clickable elsewhere in
        # this file are -- unlike the checkboxes INSIDE the popup, which
        # are two levels of nested Repeater deep and hit the same
        # findChildren()-can't-reach-delegate-items limitation already
        # documented for ColumnMenu/DataTable in this file (see
        # test_kernels_column_hide_survives_tab_switch, which drives that
        # case through the Python bridge directly instead).
        button = self._root.findChild(QObject, "timelineFilterButton")
        popup = self._root.findChild(QObject, "timelineFilterPopup")
        self.assertIsNotNone(button)
        self.assertIsNotNone(popup)
        self.assertFalse(popup.property("visible"))

        center = button.mapToScene(QPointF(button.property("width") / 2,
                                            button.property("height") / 2)).toPoint()
        self._move_to(center)
        QTest.mouseClick(self._window, Qt.LeftButton, Qt.NoModifier, center)
        for _ in range(5):
            self._app.processEvents()
        self.assertTrue(popup.property("visible"))
        self.assertEqual(self._warnings, [])

        QTest.mouseClick(self._window, Qt.LeftButton, Qt.NoModifier, center)
        for _ in range(5):
            self._app.processEvents()
        self.assertFalse(popup.property("visible"))

    def test_timeline_filter_popup_summary_reflects_active_filters(self):
        popup = self._root.findChild(QObject, "timelineFilterPopup")
        summary = self._root.findChild(QObject, "timelineFilterSummary")
        self.assertIsNotNone(summary)
        popup.setProperty("visible", True)
        for _ in range(3):
            self._app.processEvents()
        self.assertIn(str(self._timeline_model.totalSpanCount), summary.property("text"))

        self._timeline_model.applyFilters({"nameQuery": "extra_fn_100"})
        for _ in range(3):
            self._app.processEvents()
        # Exactly one span in the whole 44-lane fixture is literally named
        # "extra_fn_100"; nameQuery is a substring match, not a regex, so
        # nothing else in the fixture (extra_fn_10, extra_fn_101, etc. all
        # differ) matches it either.
        self.assertEqual(self._timeline_model.filteredSpanCount, 1)
        self.assertEqual(self._timeline_model.hiddenRowCount, 0)
        text = summary.property("text")
        self.assertIn("1/44", text)
        self.assertEqual(self._warnings, [])
        popup.setProperty("visible", False)
        self._timeline_model.clearFilters()

    def test_timeline_listview_reacts_to_applied_filters(self):
        # ListView.count/contentHeight are real Qt Quick properties bound
        # to TimelineModel.rows -- this is the genuinely QML-specific
        # thing worth proving here (the filtering LOGIC itself is already
        # covered thoroughly at the model layer in test_gui_models.py):
        # that rowsChanged actually propagates through the live ListView,
        # not just the Python-side list.
        row_height = self._timeline_root.property("rowHeight")
        self.assertEqual(self._flick.property("count"), 44)

        # Keeps only the 3 lanes whose single span is >= 500us (tid 1/2/4,
        # each 1_000_000ns) -- excludes target_event (20_000ns) and all 40
        # extra_fn_* lanes (1000ns each), a genuinely partial reduction
        # rather than an all-or-nothing one.
        self._timeline_model.applyFilters({"minDurationNs": 500_000, "activeOnly": True})
        for _ in range(5):
            self._app.processEvents()
        self.assertEqual(self._flick.property("count"), len(self._timeline_model.rows))
        self.assertEqual(len(self._timeline_model.rows), 3)
        self.assertAlmostEqual(self._flick.property("contentHeight"),
                                len(self._timeline_model.rows) * row_height, delta=1.0)
        self.assertEqual(self._warnings, [])

        self._timeline_model.clearFilters()
        for _ in range(5):
            self._app.processEvents()
        self.assertEqual(self._flick.property("count"), 44)

    # ── Timeline grouping / hide / isolate ──────────────────────────

    def test_timeline_group_button_opens_popup(self):
        button = self._root.findChild(QObject, "timelineGroupButton")
        popup = self._root.findChild(QObject, "timelineGroupPopup")
        self.assertIsNotNone(button)
        self.assertIsNotNone(popup)
        self.assertFalse(popup.property("visible"))

        center = button.mapToScene(QPointF(button.property("width") / 2,
                                            button.property("height") / 2)).toPoint()
        self._move_to(center)
        QTest.mouseClick(self._window, Qt.LeftButton, Qt.NoModifier, center)
        for _ in range(5):
            self._app.processEvents()
        self.assertTrue(popup.property("visible"))
        self.assertEqual(self._warnings, [])
        popup.setProperty("visible", False)

    def test_timeline_collapse_all_expand_all_buttons_react_and_work(self):
        # These two ToolButtons are static (not Repeater-created), so
        # unlike the group-option Buttons inside timelineGroupPopup
        # (nested Repeater delegates -- the same reachability limitation
        # documented throughout this file), real synthesized clicks on
        # them are reliable.
        collapse_btn = self._root.findChild(QObject, "timelineCollapseAllButton")
        expand_btn = self._root.findChild(QObject, "timelineExpandAllButton")
        self.assertIsNotNone(collapse_btn)
        self.assertIsNotNone(expand_btn)
        self.assertFalse(collapse_btn.property("visible"))

        self._timeline_model.setGrouping("thread")
        for _ in range(5):
            self._app.processEvents()
        self.assertTrue(collapse_btn.property("visible"))
        self.assertTrue(expand_btn.property("visible"))

        flat_count = self._flick.property("count")   # groups + all member lanes
        # _stable_center(), not a bare mapToScene() read -- these two
        # buttons just became visible this same frame (their `visible`
        # binding flipped when grouping activated), and reading their
        # position before RowLayout's geometry polish has actually run
        # can return a stale pre-layout point that overlaps a sibling
        # (see _stable_center's own docstring for the real bug this
        # caused in the search bar's Next/Clear buttons).
        center = self._stable_center(collapse_btn)
        self._move_to(center)
        QTest.mouseClick(self._window, Qt.LeftButton, Qt.NoModifier, center)
        for _ in range(5):
            self._app.processEvents()
        # Collapsed: only group header rows remain, one per (single-lane,
        # since grouping by thread) group -- 44 of them, same as the
        # lane count, just no member rows beneath any of them.
        self.assertEqual(self._flick.property("count"), 44)
        self.assertLess(self._flick.property("count"), flat_count)
        self.assertEqual(self._warnings, [])

        center = self._stable_center(expand_btn)
        self._move_to(center)
        QTest.mouseClick(self._window, Qt.LeftButton, Qt.NoModifier, center)
        for _ in range(5):
            self._app.processEvents()
        self.assertEqual(self._flick.property("count"), flat_count)
        self.assertEqual(self._warnings, [])

    def test_timeline_show_all_lanes_button_appears_and_works(self):
        button = self._root.findChild(QObject, "timelineShowAllLanesButton")
        self.assertIsNotNone(button)
        self.assertFalse(button.property("visible"))

        target = self._timeline_model.lanes[0]["name"]
        self._timeline_model.hideLane(target)
        for _ in range(5):
            self._app.processEvents()
        self.assertTrue(button.property("visible"))
        self.assertEqual(self._flick.property("count"), 43)

        center = self._stable_center(button)   # button just became visible -- see _stable_center's docstring
        self._move_to(center)
        QTest.mouseClick(self._window, Qt.LeftButton, Qt.NoModifier, center)
        for _ in range(5):
            self._app.processEvents()
        self.assertFalse(button.property("visible"))
        self.assertEqual(self._flick.property("count"), 44)
        self.assertEqual(self._warnings, [])

    def test_timeline_grouping_reshapes_listview_reactively(self):
        row_height = self._timeline_root.property("rowHeight")
        self._timeline_model.setGrouping("thread")
        for _ in range(5):
            self._app.processEvents()
        self.assertEqual(self._flick.property("count"), len(self._timeline_model.rows))
        self.assertEqual(len(self._timeline_model.rows), 88)   # 44 groups + 44 members
        self.assertAlmostEqual(self._flick.property("contentHeight"),
                                88 * row_height, delta=1.0)
        self.assertEqual(self._warnings, [])

    def test_timeline_hover_and_connectors_still_work_under_grouping(self):
        # The connector overlay's y-position math was rewritten to use
        # TimelineModel.rowIndexForLane() instead of the raw lane index
        # once grouping could make a row's visual position diverge from
        # its lane's original index (see TimelineScreen.qml's overlay
        # onPaint) -- this exercises that exact code path for real: group
        # by thread (so every row shifts down by at least one group-
        # header row), hover the same target_event span the ungrouped
        # hover tests use, and confirm it's still found with zero
        # warnings (a JS error in the rewritten y0/y1 math would surface
        # here as a warning, not necessarily a crash).
        self._timeline_model.setGrouping("thread")
        for _ in range(5):
            self._app.processEvents()
        point = self._find_target_event_point()
        self.assertIsNotNone(point, "target_event span not found under thread grouping")
        self.assertGreaterEqual(self._timeline_root.property("hoverSpanIdx"), 0)
        self.assertEqual(self._warnings, [])

    # ── Timeline search / color mode ────────────────────────────────

    def test_timeline_search_field_updates_status_and_model(self):
        field = self._root.findChild(QObject, "timelineSearchField")
        status = self._root.findChild(QObject, "timelineSearchStatus")
        self.assertIsNotNone(field)
        self.assertIsNotNone(status)
        self.assertFalse(status.property("visible"))

        # "target_event" is unique to tid=3's span in _build_trace().
        field.setProperty("text", "target_event")
        for _ in range(5):
            self._app.processEvents()
        self.assertEqual(self._timeline_model.searchMatchCount, 1)
        self.assertTrue(status.property("visible"))
        self.assertEqual(status.property("text"), "1 / 1")
        self.assertEqual(self._warnings, [])

    def test_timeline_search_no_matches_shows_no_matches_status(self):
        field = self._root.findChild(QObject, "timelineSearchField")
        status = self._root.findChild(QObject, "timelineSearchStatus")
        field.setProperty("text", "no_such_event_exists")
        for _ in range(5):
            self._app.processEvents()
        self.assertEqual(self._timeline_model.searchMatchCount, 0)
        self.assertEqual(status.property("text"), "no matches")
        self.assertEqual(self._warnings, [])

    def test_timeline_search_next_button_jumps_and_sets_hover(self):
        field = self._root.findChild(QObject, "timelineSearchField")
        next_btn = self._root.findChild(QObject, "timelineSearchNext")
        self.assertIsNotNone(next_btn)
        field.setProperty("text", "target_event")
        for _ in range(5):
            self._app.processEvents()
        self.assertTrue(next_btn.property("visible"))
        self.assertTrue(next_btn.property("enabled"))

        self._timeline_root.setProperty("hoverLane", -1)
        self._timeline_root.setProperty("hoverSpanIdx", -1)
        center = self._stable_center(next_btn)
        self._move_to(center)
        QTest.mouseClick(self._window, Qt.LeftButton, Qt.NoModifier, center)
        for _ in range(5):
            self._app.processEvents()
        self.assertGreaterEqual(self._timeline_root.property("hoverLane"), 0)
        self.assertGreaterEqual(self._timeline_root.property("hoverSpanIdx"), 0)
        self.assertIn("target_event", self._timeline_root.property("hoverText"))
        self.assertEqual(self._warnings, [])

    def test_timeline_search_clear_button_resets_everything(self):
        field = self._root.findChild(QObject, "timelineSearchField")
        clear_btn = self._root.findChild(QObject, "timelineSearchClear")
        field.setProperty("text", "target_event")
        for _ in range(5):
            self._app.processEvents()
        self.assertTrue(clear_btn.property("visible"))

        center = self._stable_center(clear_btn)   # button just became visible -- see _stable_center's docstring
        self._move_to(center)
        QTest.mouseClick(self._window, Qt.LeftButton, Qt.NoModifier, center)
        for _ in range(5):
            self._app.processEvents()
        self.assertEqual(field.property("text"), "")
        self.assertEqual(self._timeline_model.searchMatchCount, 0)
        self.assertEqual(self._warnings, [])

    def test_timeline_color_mode_button_cycles_through_modes_without_warnings(self):
        button = self._root.findChild(QObject, "timelineColorModeButton")
        self.assertIsNotNone(button)
        self.assertEqual(self._timeline_model.colorMode, "function")

        center = button.mapToScene(QPointF(button.property("width") / 2,
                                            button.property("height") / 2)).toPoint()
        expected = ["bucket", "category", "function"]
        for want in expected:
            self._move_to(center)
            QTest.mouseClick(self._window, Qt.LeftButton, Qt.NoModifier, center)
            for _ in range(5):
                self._app.processEvents()
            self.assertEqual(self._timeline_model.colorMode, want)
        self.assertEqual(self._warnings, [])

    def test_timeline_first_use_overlay_dismisses_and_persists(self):
        # A fresh temp-file-backed WorkspaceSettings (this shared engine's
        # own setup) means Workspace.timelineOverlayDismissed starts
        # False -- the overlay should be visible by default here, exactly
        # as a genuinely first-ever launch would show it.
        card = self._root.findChild(QObject, "timelineFirstUseCard")
        self.assertIsNotNone(card)
        try:
            self.assertFalse(self._workspace.timelineOverlayDismissed)
            self.assertTrue(card.property("visible"))

            got_it = self._root.findChild(QObject, "timelineFirstUseGotIt")
            self.assertIsNotNone(got_it)
            center = self._stable_center(got_it)
            self._move_to(center)
            QTest.mouseClick(self._window, Qt.LeftButton, Qt.NoModifier, center)
            for _ in range(5):
                self._app.processEvents()

            self.assertTrue(self._workspace.timelineOverlayDismissed)
            self.assertFalse(card.property("visible"))
            self.assertEqual(self._warnings, [])
        finally:
            if self._workspace.timelineOverlayDismissed:
                # Reset directly through settings -- WorkspaceBridge
                # itself has no "undismiss" Slot (dismissal is meant to
                # be permanent for a user; only test
                # cleanup needs to undo it so later tests in this shared-
                # engine class see the overlay in its original state).
                self._workspace_settings.save_timeline_overlay_dismissed(False)
                self._workspace._timeline_overlay_dismissed = False
                self._workspace.timelineOverlayDismissedChanged.emit()
                for _ in range(5):
                    self._app.processEvents()

    def test_legend_collapse_toggle_hides_swatches_and_persists(self):
        color_btn = self._root.findChild(QObject, "timelineColorModeButton")
        self.assertIsNotNone(color_btn)
        self.assertEqual(self._timeline_model.colorMode, "function")

        def _click(item):
            center = self._stable_center(item)
            self._move_to(center)
            QTest.mouseClick(self._window, Qt.LeftButton, Qt.NoModifier, center)
            for _ in range(5):
                self._app.processEvents()

        try:
            _click(color_btn)  # function -> bucket, makes the legend visible
            self.assertEqual(self._timeline_model.colorMode, "bucket")

            toggle = self._root.findChild(QObject, "legendCollapseToggle")
            legend = self._root.findChild(QObject, "legendSwatchRepeater").parent()
            self.assertIsNotNone(toggle)
            self.assertIsNotNone(legend)
            self.assertFalse(self._workspace.legendCollapsed)
            for _ in range(5):
                self._app.processEvents()
            expanded_width = legend.property("width")

            # Repeater DELEGATE items are QObject-parented to the
            # Repeater itself, not reachable via findChildren() or a
            # working itemAt() call from Python despite rendering
            # correctly (a known gotcha in this codebase; confirmed
            # again here via a standalone probe: itemAt() raises
            # "Unknown return type" through PySide's dynamic meta-method
            # invocation) -- verified indirectly instead, via the legend
            # RowLayout's own total width collapsing once its swatch
            # children lose visibility (a RowLayout's width is the sum
            # of its VISIBLE children's widths), a real, observable
            # consequence of the binding actually working, not just
            # Workspace's own flag flipping.
            _click(toggle)
            self.assertTrue(self._workspace.legendCollapsed)
            for _ in range(20):
                QTest.qWait(10)
                if legend.property("width") < expanded_width:
                    break
            self.assertLess(legend.property("width"), expanded_width)

            # Restored via the Python-side Slot directly here, not a
            # second real click: collapsing shrinks the legend, which
            # (right-aligned via its parent row's fillWidth spacer)
            # shifts the WHOLE legend, toggle included, further right --
            # a second _stable_center() click on the now-narrower layout
            # intermittently computed a position past the window's own
            # edge (offscreen QPA logs "Mouse event ... occurs outside
            # target window" and the click is dropped). The one real
            # click above already proves the button IS wired to
            # Workspace.setLegendCollapsed(); this proves the swatch
            # visibility binding reacts correctly in both directions.
            self._workspace.setLegendCollapsed(False)
            for _ in range(20):
                QTest.qWait(10)
                if legend.property("width") == expanded_width:
                    break
            self.assertEqual(legend.property("width"), expanded_width)
            self.assertEqual(self._warnings, [])
        finally:
            if self._workspace.legendCollapsed:
                self._workspace.setLegendCollapsed(False)
            _click(color_btn)  # bucket -> category
            _click(color_btn)  # category -> function, restore default

    # ── Timeline ruler / bookmarks / named ranges ───────────────────

    def test_timeline_add_bookmark_button_creates_bookmark_at_view_center(self):
        button = self._root.findChild(QObject, "timelineAddBookmarkButton")
        self.assertIsNotNone(button)
        before = len(self._timeline_model.bookmarks)
        expected_ns = self._timeline_root.property("viewStartNs") + self._timeline_root.property("visibleNs") / 2

        center = self._stable_center(button)
        self._move_to(center)
        QTest.mouseClick(self._window, Qt.LeftButton, Qt.NoModifier, center)
        for _ in range(5):
            self._app.processEvents()
        self.assertEqual(len(self._timeline_model.bookmarks), before + 1)
        self.assertAlmostEqual(self._timeline_model.bookmarks[-1]["ns"], expected_ns, delta=1.0)
        self.assertEqual(self._warnings, [])

    def test_timeline_shift_drag_selects_time_range(self):
        self._selection.clearSelection()
        pan_area = self._root.findChild(QObject, "timelinePanMouseArea")
        self.assertIsNotNone(pan_area)
        top_left = pan_area.mapToScene(QPointF(0, 0)).toPoint()
        p1 = top_left + QPoint(100, 30)
        p2 = top_left + QPoint(400, 30)

        self._move_to(p1)
        QTest.mousePress(self._window, Qt.LeftButton, Qt.ShiftModifier, p1)
        for _ in range(3):
            self._app.processEvents()
        QTest.mouseMove(self._window, p2)
        for _ in range(3):
            self._app.processEvents()
        QTest.mouseRelease(self._window, Qt.LeftButton, Qt.ShiftModifier, p2)
        for _ in range(5):
            self._app.processEvents()

        rng = self._selection.selectedTimeRange
        self.assertIn("startNs", rng)
        self.assertLess(rng["startNs"], rng["endNs"])
        self.assertEqual(self._warnings, [])
        # A shift-drag must NOT also pan the view -- distinct gesture.
        self.assertEqual(self._timeline_root.property("zoom"), 1.0)

    def test_timeline_add_range_button_enabled_after_selection_and_promotes_it(self):
        add_range_btn = self._root.findChild(QObject, "timelineAddRangeButton")
        self.assertIsNotNone(add_range_btn)
        self._selection.clearSelection()
        for _ in range(3):
            self._app.processEvents()
        self.assertFalse(add_range_btn.property("enabled"))

        self._selection.selectTimeRange(100_000, 300_000)
        for _ in range(3):
            self._app.processEvents()
        self.assertTrue(add_range_btn.property("enabled"))

        before = len(self._timeline_model.namedRanges)
        center = self._stable_center(add_range_btn)
        self._move_to(center)
        QTest.mouseClick(self._window, Qt.LeftButton, Qt.NoModifier, center)
        for _ in range(5):
            self._app.processEvents()
        self.assertEqual(len(self._timeline_model.namedRanges), before + 1)
        self.assertEqual(self._timeline_model.namedRanges[-1]["startNs"], 100_000.0)
        self.assertEqual(self._timeline_model.namedRanges[-1]["endNs"], 300_000.0)
        self.assertEqual(self._warnings, [])
        self._selection.clearSelection()

    def test_timeline_ruler_repaints_across_pan_zoom_without_warnings(self):
        ruler = self._root.findChild(QObject, "timelineRuler")
        self.assertIsNotNone(ruler)
        self._timeline_root.setProperty("zoom", 2.0)
        self._timeline_root.setProperty(
            "viewStartNs", self._timeline_model.viewStartNs + self._timeline_model.traceDurationNs * 0.1)
        for _ in range(5):
            self._app.processEvents()
        self.assertEqual(self._warnings, [])
        self._timeline_root.setProperty("zoom", 1.0)
        self._timeline_root.setProperty("viewStartNs", self._timeline_model.viewStartNs)

    def test_timeline_bookmark_marker_click_jumps_view(self):
        ruler = self._root.findChild(QObject, "timelineRuler")
        # Zoom in and park the view mid-trace (not at either edge) --
        # at the default zoom=1.0, the ENTIRE trace is already visible,
        # so "jumping" anywhere is a no-op (clampViewStart() immediately
        # snaps it right back to where it started); this reproduces that
        # exact false-negative once already, see git history for the
        # diagnosis. The bookmark itself sits near the LEFT edge of the
        # current view (not center), so recentering on it is a real,
        # clamp-surviving move given enough trace on both sides.
        self._timeline_root.setProperty("zoom", 4.0)
        base_start = self._timeline_model.viewStartNs + self._timeline_model.traceDurationNs * 0.4
        self._timeline_root.setProperty("viewStartNs", base_start)
        for _ in range(5):
            self._app.processEvents()
        target_ns = self._timeline_root.property("viewStartNs") + self._timeline_root.property("visibleNs") * 0.1
        self._timeline_model.addBookmark(target_ns, "diag")
        for _ in range(5):
            self._app.processEvents()

        # The bookmark marker's hit-test box is populated by the ruler's
        # own Canvas paint pass (its x-position depends on the current
        # zoom/pan, computed the same way the lane bars are) -- read it
        # back rather than recomputing that math independently here,
        # which would risk silently testing against the wrong pixel if
        # the two computations ever drifted apart. Canvas.requestPaint()
        # is tied to the render loop, not the Python event queue, so
        # polling with QTest.qWait() (real settle time, same fix as
        # _stable_center) rather than bare processEvents() here too.
        boxes = []
        for _ in range(20):
            QTest.qWait(10)
            raw = ruler.property("_bookmarkHitboxes")
            # A QML `property var` holding a JS array comes back from
            # PySide6's .property() as a QJSValue, not a plain Python
            # list -- .toVariant() does that conversion explicitly.
            boxes = raw.toVariant() if hasattr(raw, "toVariant") else raw
            if boxes:
                break
        self.assertTrue(boxes, "ruler painted no bookmark hitboxes")
        box = boxes[-1]
        ruler_top_left = ruler.mapToScene(QPointF(0, 0)).toPoint()
        click_point = ruler_top_left + QPoint(int(box["x"]), int(ruler.property("height")) - 3)

        before_view_start = self._timeline_root.property("viewStartNs")
        self._move_to(click_point)
        QTest.mouseClick(self._window, Qt.LeftButton, Qt.NoModifier, click_point)
        for _ in range(5):
            self._app.processEvents()
        self.assertNotAlmostEqual(self._timeline_root.property("viewStartNs"), before_view_start, delta=1.0)
        self.assertEqual(self._warnings, [])
        self._timeline_root.setProperty("zoom", 1.0)
        self._timeline_root.setProperty("viewStartNs", self._timeline_model.viewStartNs)

    # ── Cross-screen smoke test + FlameGraph theme-toggle regression ──

    def test_every_tab_loads_without_warnings(self):
        # Cheap, catches a broken "../components" import or missing
        # property across every migrated screen at once -- each of the
        # 10 tabs' Loader activates for the first time here (most were
        # never visited by any other test in this class). Tab 9 (Compare)
        # renders its EmptyState in this class's fixture (Compare.available
        # is False -- see setUpClass's cls._comparison), so this also
        # covers "every tab clean with no comparison trace loaded".
        for i in range(10):
            self._warnings.clear()
            self._tab_bar.setProperty("currentIndex", i)
            for _ in range(8):
                self._app.processEvents()
            self.assertEqual(self._warnings, [], f"tab {i} produced QML warnings: {self._warnings}")

    def test_flame_graph_tooltip_repaints_on_theme_toggle(self):
        # Regression guard: FlameGraphScreen's tooltip colors must follow
        # the light/dark toggle (no hardcoded hex). Checks the bound QColor
        # property rather than a screenshot -- screenshots of this tooltip
        # are easy to misjudge, property introspection is unambiguous.
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

        # objectName, NOT a bare "has a followCursor property" duck-typed
        # search -- the shared Tooltip.qml component (src/gui/qml/
        # components/Tooltip.qml) is now instantiated by more than one
        # screen (FlameGraph AND the table-upgrade round's DataTable), and
        # a non-specific search finds whichever instance happens to come
        # first in tree order, which is exactly what broke this test the
        # first time DataTable's own Tooltip existed in the same window
        # (Kernels stays loaded once visited, same as every other tab --
        # "first match" silently picked Kernels' tooltip instead of
        # FlameGraph's, and it never became visible from a FlameGraph
        # hover for the obvious reason: it isn't FlameGraph's tooltip).
        tooltip = self._root.findChild(QQuickItem, "flameGraphTooltip")
        self.assertIsNotNone(tooltip, "flameGraphTooltip instance not found")

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

    # ── Table upgrade round: DataTable on the Kernels screen ────────────
    #    Same shared engine/window as every test above.

    def _kernels_datatable(self):
        self._tab_bar.setProperty("currentIndex", 2)  # Kernels
        for _ in range(5):
            self._app.processEvents()
        dt = self._root.findChild(QObject, "dataTable_Kernels")
        self.assertIsNotNone(dt, "Kernels' DataTable (objectName dataTable_Kernels) not found")
        return dt

    def _header_click_point(self, header_item, column_key):
        cols = [c for c in self._kernels.table.config.columns if c["visible"]]
        frozen_width = sum(c["width"] for c in cols if c["frozen"])
        x_in_scroll = 0
        target_x = None
        for c in cols:
            if c["frozen"]:
                continue
            if c["key"] == column_key:
                target_x = frozen_width + x_in_scroll + c["width"] / 2
                break
            x_in_scroll += c["width"]
        self.assertIsNotNone(target_x, f"column {column_key} not visible")
        top_left = header_item.mapToScene(QPointF(0, 0)).toPoint()
        return top_left + QPoint(int(target_x), int(header_item.property("height") / 2))

    def test_kernels_header_click_sorts_numerically(self):
        self._kernels.table.config.resetLayout()
        self._kernels.table.filters.clearFilters()
        dt = self._kernels_datatable()

        header = None
        for c in dt.findChildren(QQuickItem):
            if c.metaObject().className().startswith("DataTableHeader"):
                header = c
                break
        self.assertIsNotNone(header, "DataTableHeader not found")

        pt = self._header_click_point(header, "totalNs")
        # Move-priming before press/release -- a bare click on a
        # Flickable/Repeater-recursed item is unreliable under offscreen
        # QPA (see the Call Tree click test's note).
        start = pt - QPoint(0, 15)
        for step in range(1, 5):
            self._move_to(start + (pt - start) * step / 4)
        QTest.mousePress(self._window, Qt.LeftButton, Qt.NoModifier, pt)
        self._app.processEvents()
        QTest.mouseRelease(self._window, Qt.LeftButton, Qt.NoModifier, pt)
        for _ in range(5):
            self._app.processEvents()

        self.assertEqual(self._kernels.table.filters.sortKey, "totalNs")
        self.assertEqual(self._warnings, [])

    def test_kernels_column_hide_survives_tab_switch(self):
        self._kernels.table.config.resetLayout()
        self._kernels.table.config.setColumnVisible("avgNs", False)
        self._tab_bar.setProperty("currentIndex", 0)  # Overview
        for _ in range(5):
            self._app.processEvents()
        self._tab_bar.setProperty("currentIndex", 2)  # back to Kernels
        for _ in range(5):
            self._app.processEvents()

        cols = {c["key"]: c for c in self._kernels.table.config.columns}
        self.assertFalse(cols["avgNs"]["visible"])
        self.assertEqual(self._warnings, [])
        self._kernels.table.config.resetLayout()

    def test_kernels_csv_export_writes_filtered_rows(self):
        import csv
        import tempfile
        self._kernels_datatable()
        self._kernels.table.filters.clearFilters()
        self._kernels.table.filters.textFilter = "target_event"
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            self.assertTrue(self._kernels.table.exportCsv(path))
            with open(path, newline="") as fh:
                rows = list(csv.reader(fh))
            self.assertEqual(len(rows) - 1, 1)   # header + exactly the filtered row
            self.assertEqual(rows[1][0], "target_event")
        finally:
            Path(path).unlink(missing_ok=True)
            self._kernels.table.filters.clearFilters()

    def test_kernels_header_definition_tooltip(self):
        self._kernels.table.config.resetLayout()
        dt = self._kernels_datatable()
        tooltip = self._root.findChild(QObject, "dataTableTooltip_Kernels")
        self.assertIsNotNone(tooltip)
        # NOT asserting the tooltip starts invisible here -- switching
        # back to this tab under an already-stationary cursor (left over
        # from a previous test's own hover) can re-trigger Qt Quick's
        # hover re-evaluation before this test moves the mouse at all,
        # independent of whatever setUp() already reset. What actually
        # matters, and IS asserted below: a real hover motion ending on
        # the header produces the right tooltip content.

        header = None
        for c in dt.findChildren(QQuickItem):
            if c.metaObject().className().startswith("DataTableHeader"):
                header = c
                break
        self.assertIsNotNone(header)
        pt = self._header_click_point(header, "totalNs")

        found = False
        for attempt in range(3):
            start = pt - QPoint(0, 15)
            for step in range(1, 5):
                self._move_to(start + (pt - start) * step / 4)
            if tooltip.property("visible"):
                found = True
                break
        self.assertTrue(found, "header definition tooltip never became visible on hover")
        self.assertIn("Sum of every call's duration", tooltip.property("text"))
        self.assertEqual(self._warnings, [])

        # Move away -- leaves state clean for whatever test sorts next,
        # belt-and-suspenders alongside setUp()'s own reset above.
        self._move_to(QPoint(5, 780))
        tooltip.setProperty("visible", False)

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
        # tree and structurally cannot reach them, even though the rows
        # render correctly. A coordinate sweep (the same technique
        # _find_hover_point uses for Timeline) sidesteps this.
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
        found = self._find_target_event_point()
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

    def test_command_palette_opens_via_ctrl_k_filters_and_navigates(self):
        palette = self._root.findChild(QObject, "commandPalette")
        self.assertIsNotNone(palette)
        self.assertFalse(palette.property("visible"))

        QTest.keyClick(self._window, Qt.Key_K, Qt.ControlModifier)
        for _ in range(5):
            self._app.processEvents()
        self.assertTrue(palette.property("visible"))

        field = self._root.findChild(QObject, "commandPaletteField")
        self.assertIsNotNone(field)
        field.setProperty("text", "Go to 3 Kernels")
        for _ in range(5):
            self._app.processEvents()

        results = self._root.findChild(QObject, "commandPaletteResults")
        self.assertIsNotNone(results)
        self.assertEqual(results.property("count"), 1)

        QTest.keyClick(self._window, Qt.Key_Return)
        for _ in range(5):
            self._app.processEvents()

        self.assertFalse(palette.property("visible"))
        self.assertEqual(self._selection.currentTab, 2)
        self.assertEqual(self._warnings, [])
        self._selection.currentTab = 1  # restore Timeline for later tests

    def test_command_palette_escape_closes_without_acting(self):
        palette = self._root.findChild(QObject, "commandPalette")
        QTest.keyClick(self._window, Qt.Key_K, Qt.ControlModifier)
        for _ in range(5):
            self._app.processEvents()
        self.assertTrue(palette.property("visible"))

        QTest.keyClick(self._window, Qt.Key_Escape)
        for _ in range(5):
            self._app.processEvents()
        self.assertFalse(palette.property("visible"))
        self.assertEqual(self._selection.currentTab, 1)  # unchanged
        self.assertEqual(self._warnings, [])

    def test_shortcuts_dialog_opens_via_f1_and_lists_real_shortcuts(self):
        dialog = self._root.findChild(QObject, "shortcutsDialog")
        self.assertIsNotNone(dialog)
        self.assertFalse(dialog.property("visible"))

        QTest.keyClick(self._window, Qt.Key_F1)
        for _ in range(5):
            self._app.processEvents()
        self.assertTrue(dialog.property("visible"))
        self.assertEqual(self._warnings, [])

        dialog.close()
        for _ in range(5):
            self._app.processEvents()
        self.assertFalse(dialog.property("visible"))

    def test_icon_only_controls_have_real_accessible_names(self):
        # Spot-checks both icon-only control patterns: real ToolButtons
        # (which get Accessible.role for free) and MouseArea-based
        # pseudo-buttons (which need Accessible.role set explicitly).
        # Accessible.name is read back through QAccessible because plain
        # .property("Accessible.name") doesn't return the attached value.
        from PySide6.QtGui import QAccessible

        def accessible_name(item):
            self.assertIsNotNone(item)
            iface = QAccessible.queryAccessibleInterface(item)
            self.assertIsNotNone(iface, "no accessible interface for this item")
            return iface.text(QAccessible.Text.Name)

        zoom_out = self._root.findChild(QObject, "timelineZoomOutButton")
        zoom_in = self._root.findChild(QObject, "timelineZoomInButton")
        reset_btn = self._root.findChild(QObject, "timelineResetButton")
        self.assertEqual(accessible_name(zoom_out), "Zoom out")
        self.assertEqual(accessible_name(zoom_in), "Zoom in")
        self.assertEqual(accessible_name(reset_btn), "Reset view")

        # Inspector reopen button: only present (non-empty accessible
        # interface) while the panel is actually collapsed.
        if self._selection.inspectorOpen:
            self._selection.toggleInspector()
            for _ in range(5):
                self._app.processEvents()
        reopen_btn = self._root.findChild(QObject, "inspectorReopenButton")
        self.assertEqual(accessible_name(reopen_btn), "Open inspector")
        self._selection.toggleInspector()
        for _ in range(5):
            self._app.processEvents()
        collapse_btn = self._root.findChild(QObject, "inspectorCollapseButton")
        self.assertEqual(accessible_name(collapse_btn), "Collapse inspector")

        self.assertEqual(self._warnings, [])

    def test_reset_all_confirm_dialog_accept_wipes_workspace_settings(self):
        # Menu > File > Reset All UI Settings opens this same dialog
        # (Menu/MenuBar popup interaction itself isn't exercised here --
        # QtQuick.Controls menu popups are known-fragile to drive via
        # QTest under the offscreen QPA platform in this project; this
        # instead verifies the dialog's own accept()->Workspace.resetAll()
        # wiring directly, which is the part any menu click would
        # ultimately trigger).
        dialog = self._root.findChild(QObject, "resetAllConfirmDialog")
        self.assertIsNotNone(dialog)

        self._workspace.setLegendCollapsed(True)
        self.assertTrue(self._workspace.legendCollapsed)

        dialog.open()
        for _ in range(5):
            self._app.processEvents()
        self.assertTrue(dialog.property("visible"))

        dialog.accept()
        for _ in range(5):
            self._app.processEvents()

        self.assertFalse(dialog.property("visible"))
        self.assertFalse(self._workspace.legendCollapsed)
        self.assertEqual(self._warnings, [])

    def test_reset_all_confirm_dialog_reject_leaves_settings_untouched(self):
        dialog = self._root.findChild(QObject, "resetAllConfirmDialog")
        self._workspace.setLegendCollapsed(True)

        dialog.open()
        for _ in range(5):
            self._app.processEvents()
        dialog.reject()
        for _ in range(5):
            self._app.processEvents()

        self.assertFalse(dialog.property("visible"))
        self.assertTrue(self._workspace.legendCollapsed)  # untouched by reject()
        self._workspace.setLegendCollapsed(False)  # restore for later tests
        self.assertEqual(self._warnings, [])

    def test_open_profile_overlay_reacts_to_app_controller_signals(self):
        # No real subprocess spawn here (see test_gui_controller.py's
        # mocked-Popen tests for that) -- this exercises the RECEIVING
        # side of the handshake against the real rendered Main.qml:
        # AppController.profileOpening/profileOpenFailed driving the
        # overlay's visible state, exactly as a real "Open Profile"
        # attempt would (the File-menu trigger calls App.openProfile(),
        # which emits the same two signals).
        overlay = self._root.findChild(QObject, "openProfileOverlay")
        self.assertIsNotNone(overlay, "could not locate the Open Profile overlay")

        self.assertEqual(overlay.property("state"), "idle")
        self.assertFalse(overlay.property("visible"))

        self._app_controller.profileOpening.emit()
        for _ in range(5):
            self._app.processEvents()
        self.assertEqual(overlay.property("state"), "opening")
        self.assertTrue(overlay.property("visible"))

        self._app_controller.profileOpenFailed.emit({
            "kind": "internal_error", "message": "Could not open the new profile.",
            "detail": "exit code 1", "file": "", "stage": "opening_profile", "tracebackText": "",
        })
        for _ in range(5):
            self._app.processEvents()
        self.assertEqual(overlay.property("state"), "error")
        self.assertTrue(overlay.property("visible"))
        self.assertEqual(self._warnings, [])

        dismiss_btn = self._root.findChild(QObject, "openProfileOverlayDismiss")
        self.assertIsNotNone(dismiss_btn)
        center = self._stable_center(dismiss_btn)
        self._move_to(center)
        QTest.mouseClick(self._window, Qt.LeftButton, Qt.NoModifier, center)
        for _ in range(5):
            self._app.processEvents()
        self.assertEqual(overlay.property("state"), "idle")
        self.assertFalse(overlay.property("visible"))
        self.assertEqual(self._warnings, [])

    def test_compare_screen_state_shows_empty_when_no_comparison_loaded(self):
        # This shared engine only ever loads ONE trace pairing for the
        # whole test run (see class docstring) -- trace_b is always None
        # here, so Compare.available is always False, making this the
        # one screen where ScreenState's "empty" branch is exercised for
        # real without imperatively faking anything.
        self._tab_bar.setProperty("currentIndex", 9)  # "10 Compare"
        for _ in range(5):
            self._app.processEvents()
        state_item = self._root.findChild(QObject, "compareScreenState")
        self.assertIsNotNone(state_item)
        self.assertEqual(state_item.property("state"), "empty")
        self.assertEqual(self._warnings, [])

    def test_roofline_screen_state_shows_unsupported_for_a_cpu_only_trace(self):
        # The fixture trace has no cuda/rocm/opencl spans, so Roofline
        # has nothing to plot -- Roofline.available is False for real.
        self._tab_bar.setProperty("currentIndex", 5)  # "6 Roofline"
        for _ in range(5):
            self._app.processEvents()
        state_item = self._root.findChild(QObject, "rooflineScreenState")
        self.assertIsNotNone(state_item)
        self.assertEqual(state_item.property("state"), "unsupported")
        self.assertEqual(self._warnings, [])

    def test_error_state_technical_details_copy_and_open_log_actions(self):
        # No screen in this fixture reaches a real "error" ScreenState
        # today (see loader.py/controller.py's docstring: every failure
        # this GUI can classify happens before any tab exists at all) --
        # imperatively driving CompareScreen's already-loaded ScreenState
        # into "error" is the only way to exercise ErrorState's new
        # technical-details/copy/open-log UI against the REAL rendered
        # tree rather than a separate throwaway engine. Restores the
        # real Compare.available-driven state afterward (in `finally`) so
        # later tests (this class shares ONE engine/state across the
        # whole run) see CompareScreen exactly as they'd otherwise expect.
        self._tab_bar.setProperty("currentIndex", 9)  # "10 Compare"
        for _ in range(5):
            self._app.processEvents()
        state_item = self._root.findChild(QObject, "compareScreenState")
        self.assertIsNotNone(state_item)

        try:
            state_item.setProperty("state", "error")
            state_item.setProperty("errorMessage", "Something went wrong.")
            state_item.setProperty("errorDetail", "a detail line")
            state_item.setProperty("errorTracebackText", "Traceback (most recent call last):\n  boom")
            state_item.setProperty("errorStage", "computing_dashboard")
            state_item.setProperty("errorFile", "/tmp/fake.json")
            for _ in range(5):
                self._app.processEvents()

            details_toggle = self._root.findChild(QObject, "errorDetailsToggle")
            copy_btn = self._root.findChild(QObject, "errorCopyDiagnostics")
            open_log_btn = self._root.findChild(QObject, "errorOpenLog")
            details_pane = self._root.findChild(QObject, "errorDetailsPane")
            self.assertIsNotNone(details_toggle)
            self.assertIsNotNone(copy_btn)
            self.assertIsNotNone(open_log_btn)
            self.assertIsNotNone(details_pane)

            # Accessible name/tooltip -- "every icon-only control has a
            # tooltip and accessible name" applies just as much to these
            # text buttons carrying non-obvious behavior (copy/open-log).
            # Accessible.name/description are QML ATTACHED properties --
            # plain QObject.property("Accessible.name") and even
            # QQmlProperty(obj, "Accessible.name").read() both silently
            # return None for them (confirmed via a standalone probe
            # before trusting either); the actual accessibility backend
            # (QAccessible.queryAccessibleInterface) is the only thing
            # that resolves the real value.
            from PySide6.QtGui import QAccessible
            copy_iface = QAccessible.queryAccessibleInterface(copy_btn)
            open_log_iface = QAccessible.queryAccessibleInterface(open_log_btn)
            self.assertIsNotNone(copy_iface)
            self.assertIsNotNone(open_log_iface)
            self.assertTrue(copy_iface.text(QAccessible.Text.Name))
            self.assertTrue(open_log_iface.text(QAccessible.Text.Name))

            def _click(item):
                # _stable_center, not a raw mapToScene() read -- this
                # click follows right after an imperative state/property
                # change that reflows the whole ErrorState layout, the
                # exact scenario _stable_center's own docstring warns
                # stale-position reads happen in.
                center = self._stable_center(item)
                self._move_to(center)
                QTest.mouseClick(self._window, Qt.LeftButton, Qt.NoModifier, center)
                for _ in range(5):
                    self._app.processEvents()

            self.assertFalse(details_pane.property("visible"))
            _click(details_toggle)
            self.assertTrue(details_pane.property("visible"))

            # The button calls the REAL Inspector.copyToClipboard Slot,
            # which writes to the actual system clipboard -- assert on
            # that directly rather than trying to intercept the call.
            _click(copy_btn)
            clipboard_text = QGuiApplication.clipboard().text()
            self.assertIn("Something went wrong.", clipboard_text)
            self.assertIn("a detail line", clipboard_text)
            self.assertIn("computing_dashboard", clipboard_text)
            self.assertIn("boom", clipboard_text)

            self.assertEqual(self._warnings, [])
        finally:
            state_item.setProperty("state", "ready" if self._comparison.available else "empty")
            for _ in range(5):
                self._app.processEvents()

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
