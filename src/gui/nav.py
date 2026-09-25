"""
Cross-tab navigation and selection state -- the single source of truth
every screen reads from and writes to, so tabs never reach into each
other directly. Main.qml's TabBar/StackLayout are both bound to this
singleton's `currentTab` instead of the TabBar solely owning tab state
(previously: `StackLayout.currentIndex: tabBar.currentIndex`, a one-way
convenience binding with nothing else able to drive it -- see this
module's own history for why that had to change).

Correlation key is (category, name) -- NOT true per-instance span
identity. SpanEvent.span_id/parent_span_id are real per-instance IDs
assigned at hook-capture time, but chrome_trace.write() never
serializes them and load_trace_from_json() never restores them, so
every span any GUI bridge sees has span_id=="". Fixing that end-to-end
(new JSON fields + threading real IDs through every bridge) was
evaluated and explicitly deferred as too large/invasive for this
feature -- (category, name) is what every aggregating bridge (Kernels,
CallTree, Dashboard, Profile) already natively groups spans by, so this
needs no new data-model plumbing, just one shared place to hold the
CURRENT selection. Practical consequence, intentional and disclosed
(see InspectorBridge): selectFunction() means "select occurrences of
this raw (category, name)", not "select the exact span instance
clicked elsewhere in some other tab".
"""
from __future__ import annotations

from typing import Any

from PySide6.QtCore import QObject, Property, Signal, Slot


class Selection(QObject):
    """Registered as the "Nav" singleton (module "Hprofiler" 1.0, same
    as every other bridge). Properties are live (notify=), not
    constant=True unlike most bridges in this codebase -- this is the
    one object in the bridge layer genuinely mutated at runtime by user
    interaction rather than computed once from the trace at
    construction."""

    currentTabChanged = Signal()
    selectionChanged = Signal()
    callPathChanged = Signal()
    timeRangeChanged = Signal()
    threadChanged = Signal()
    inspectorOpenChanged = Signal()
    breadcrumbsChanged = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._current_tab = 0
        self._category = ""
        self._name = ""
        self._call_path: list[dict[str, str]] = []
        self._time_range: dict[str, float] = {}
        self._thread: dict[str, int] = {}
        self._inspector_open = True
        self._breadcrumbs: list[dict[str, Any]] = []

    # ── currentTab ───────────────────────────────────────────────────
    @Property(int, notify=currentTabChanged)
    def currentTab(self) -> int:
        return self._current_tab

    @currentTab.setter
    def currentTab(self, value: int) -> None:
        if value != self._current_tab:
            self._current_tab = value
            self.currentTabChanged.emit()

    # ── selection ────────────────────────────────────────────────────
    @Property(str, notify=selectionChanged)
    def selectedCategory(self) -> str:
        return self._category

    @Property(str, notify=selectionChanged)
    def selectedName(self) -> str:
        return self._name

    @Property('QVariantList', notify=callPathChanged)
    def selectedCallPath(self) -> list[dict[str, str]]:
        return self._call_path

    @Property('QVariantMap', notify=timeRangeChanged)
    def selectedTimeRange(self) -> dict[str, float]:
        return self._time_range

    @Property('QVariantMap', notify=threadChanged)
    def selectedThread(self) -> dict[str, int]:
        return self._thread

    @Property(bool, notify=inspectorOpenChanged)
    def inspectorOpen(self) -> bool:
        return self._inspector_open

    @Property('QVariantList', notify=breadcrumbsChanged)
    def breadcrumbs(self) -> list[dict[str, Any]]:
        return self._breadcrumbs

    # ── mutators, called from QML ───────────────────────────────────
    @Slot(str, str)
    def selectFunction(self, category: str, name: str) -> None:
        if category == self._category and name == self._name:
            return
        self._category = category
        self._name = name
        self.selectionChanged.emit()

    @Slot('QVariantList')
    def selectCallPath(self, path: list) -> None:
        """path: [{category,name}, ...] root-to-selected. Call-tree
        nodes aren't independently addressable by index across screens
        the way list rows are -- a name path is the only stable way to
        describe "this specific position in the tree"."""
        self._call_path = [dict(p) for p in path]
        self.callPathChanged.emit()
        if self._call_path:
            last = self._call_path[-1]
            self.selectFunction(last.get("category", ""), last.get("name", ""))

    @Slot(float, float)
    def selectTimeRange(self, start_ns: float, end_ns: float) -> None:
        self._time_range = {"startNs": start_ns, "endNs": end_ns}
        self.timeRangeChanged.emit()

    @Slot(int, int)
    def selectThread(self, pid: int, tid: int) -> None:
        self._thread = {"pid": pid, "tid": tid}
        self.threadChanged.emit()

    @Slot()
    def clearSelection(self) -> None:
        self._category = ""
        self._name = ""
        self._call_path = []
        self._time_range = {}
        self._thread = {}
        self.selectionChanged.emit()
        self.callPathChanged.emit()
        self.timeRangeChanged.emit()
        self.threadChanged.emit()

    @Slot(int)
    def navigateTo(self, tab_index: int) -> None:
        """Switches tabs AND records where we came from, so goBack() can
        undo it -- the ONE mutator that pushes a breadcrumb. A plain
        manual TabBar click (bound directly to `currentTab`, not routed
        through this slot) never does: ordinary browsing isn't a "path"
        worth retracing, only an explicit "go look at this related
        thing" jump is."""
        self._breadcrumbs.append({
            "tab": self._current_tab, "category": self._category, "name": self._name,
        })
        self.breadcrumbsChanged.emit()
        self.currentTab = tab_index

    @Slot()
    def goBack(self) -> None:
        if not self._breadcrumbs:
            return
        prev = self._breadcrumbs.pop()
        self.breadcrumbsChanged.emit()
        self.currentTab = prev["tab"]
        self.selectFunction(prev["category"], prev["name"])

    @Slot()
    def toggleInspector(self) -> None:
        self._inspector_open = not self._inspector_open
        self.inspectorOpenChanged.emit()
