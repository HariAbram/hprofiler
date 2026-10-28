"""
Subprocess driver for tests/test_gui_evidence_interaction.py's Compare
checks: loads the real Main.qml with a populated Compare tab on the joint-
release scenario (tests/compare_scenarios.mpi_waitall_two_senders: both
senders slowed, one of them off the critical path), selects the
MPI_Waitall, reads the joint-origin rows and the three attributions, clicks
through to an origin and to the Timeline, and prints one JSON line.

  python3 tests/gui_compare_joint_driver.py [SHOT_DIR]
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtCore import QPointF, Qt
from PySide6.QtQuick import QQuickItem
from PySide6.QtTest import QTest

import tests.compare_scenarios as S
import tests.test_gui_timeline_hover as H
from src.gui.comparison import ComparisonBridge

SHOTS = Path(sys.argv[1]) if len(sys.argv) > 1 else None


def walk(item):
    yield item
    for ch in item.childItems():
        yield from walk(ch)


def find(win, name: str):
    return next((i for i in walk(win.contentItem()) if i.objectName() == name), None)


def find_all(win, prefix: str) -> list:
    return [i for i in walk(win.contentItem()) if i.objectName().startswith(prefix)]


def ensure_visible(item: QQuickItem) -> None:
    a = item.parentItem()
    while a is not None:
        if a.property("contentY") is not None and a.property("contentItem") is not None:
            ci = a.property("contentItem")
            y = item.mapToItem(ci, QPointF(0, 0)).y()
            cy, h = a.property("contentY"), a.height()
            if y + item.height() > cy + h:
                a.setProperty("contentY", y + item.height() - h + 10)
            elif y < cy:
                a.setProperty("contentY", max(0.0, y - 10))
        a = a.parentItem()
    QTest.qWait(50)


def click(win, item: QQuickItem) -> None:
    ensure_visible(item)
    p = item.mapToScene(QPointF(item.width() / 2, item.height() / 2))
    QTest.mouseMove(win, p.toPoint())
    QTest.mouseClick(win, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, p.toPoint())
    QTest.qWait(80)


def scroll_to(win, name: str) -> None:
    """Scroll the Compare page so the named panel is at the top."""
    flick, item = find(win, "compareFlick"), find(win, name)
    if flick is not None and item is not None:
        flick.setProperty("contentY", max(0.0, item.mapToItem(flick.property("contentItem"), QPointF(0, 0)).y() - 8))
        QTest.qWait(200)


def main() -> None:
    candidate, baseline = S.mpi_waitall_two_senders(True), S.mpi_waitall_two_senders(False)
    H._build_trace = lambda: candidate

    class PopulatedBridge(ComparisonBridge):
        def __init__(self, trace, _none, theme, parent=None):
            super().__init__(trace, baseline, theme, parent)
    H.ComparisonBridge = PopulatedBridge
    H.TestTimelineHover.setUpClass()
    C = H.TestTimelineHover
    nav, cmpb, win, root = C._selection, C._comparison, C._window, C._root
    if SHOTS:
        win.resize(1500, 1000)
    C._warnings.clear()
    nav.navigateTo(9)
    QTest.qWait(300)
    screen = next(i for i in root.findChildren(QQuickItem)
                  if i.property("selectedId") is not None and i.property("cpPalette") is not None)
    out: dict = {}
    rows = cmpb.contributors + cmpb.propagated + cmpb.offCriticalPath
    wait = next(r for r in rows if r["label"] == "MPI_Waitall")
    rank0 = next(r for r in rows if r["label"] == "compute" and "rank0" in r["roles"])
    screen.setProperty("selectedId", wait["id"])
    QTest.qWait(250)
    out["jointRows"] = [i.property("summary") for i in find_all(win, "compareJointOrigin_")
                        if i.property("summary") is not None]
    out["jointPanelVisible"] = bool(find(win, "compareJointOrigins").property("visible"))
    out["waitAttribution"] = [find(win, f"compareAttributionValue_{k}").property("text") for k in range(3)]
    if SHOTS:
        C._selection.toggleInspector()              # the Inspector closed: room for the details
        QTest.qWait(150)
        scroll_to(win, "compareDetail")
        win.grabWindow().save(str(SHOTS / "compare_joint_wait.png"))
    # select the off-path origin from the wait's joint-origin list
    sel = next(i for i in find_all(win, "compareJointOriginSelect_")
               if "rank0" in str(i.parentItem().property("summary")))
    click(win, sel)
    QTest.qWait(250)
    out["selectedAfterClick"] = screen.property("selectedId")
    out["rank0Id"] = rank0["id"]
    out["rank0Attribution"] = [find(win, f"compareAttributionValue_{k}").property("text") for k in range(3)]
    out["rank0JointWaitsVisible"] = bool(find(win, "compareJointWaits").property("visible"))
    detail = screen.property("detail")
    detail = detail.toVariant() if hasattr(detail, "toVariant") else detail
    out["rank0Rows"] = {r["label"]: [r["before"], r["after"], r["delta"]] for r in detail.get("rows", [])}
    if SHOTS:
        scroll_to(win, "compareDetail")
        win.grabWindow().save(str(SHOTS / "compare_joint_origin.png"))
    # back to the wait; "Timeline" on its joint origin goes to that origin's range
    screen.setProperty("selectedId", wait["id"])
    QTest.qWait(250)
    tl_btn = next(i for i in find_all(win, "compareJointOriginTimeline_")
                  if "rank0" in str(i.parentItem().property("summary")))
    click(win, tl_btn)
    QTest.qWait(300)
    out["afterTimelineTab"] = nav.currentTab
    out["focusRange"] = dict(nav.focusRange)
    t = cmpb.identityTimeline(rank0["id"])
    out["rank0Range"] = [t["timelineStartNs"], t["timelineEndNs"]]
    out["selectedName"] = nav.selectedName
    out["warnings"] = list(C._warnings)
    print(json.dumps(out, default=str))


if __name__ == "__main__":
    main()
