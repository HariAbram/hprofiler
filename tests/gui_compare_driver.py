"""
Subprocess driver for tests/test_gui_compare_interaction.py: loads the real
Main.qml with a POPULATED Compare tab (candidate = the opened trace,
baseline = --compare) through the shared GUI fixture of
tests/test_gui_timeline_hover.py, clicks through it with synthesized mouse
input and prints one JSON line with what happened. Runs in its own process
because one process can only ever hold one QML engine / trace pairing.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtCore import QObject, QPointF, Qt
from PySide6.QtQuick import QQuickItem
from PySide6.QtTest import QTest

import tests.test_gui_timeline_hover as H
from src.core.events import Category, SpanEvent
from src.core.trace import Trace, TraceMetadata
from src.gui.comparison import ComparisonBridge

MS = 1_000_000
SRC = Path(tempfile.mkdtemp(prefix="hprofiler_cmpgui_")) / "halo.c"
SRC.write_text("\n".join(f"/* line {i} */" for i in range(1, 30)) + "\n")


def mpi_trace(slow: bool) -> Trace:
    t = Trace(TraceMetadata(command="mpirun -np 2 ./halo", cwd=str(SRC.parent)))
    c = MS
    w0 = (20 if slow else 10) * MS
    for _ in range(5):
        t.add(SpanEvent("compute", Category.OPENMP, c, w0, 100, 100,
                        {"type": "work", "file": str(SRC), "line": "12"}))
        t.add(SpanEvent("MPI_Send", Category.MPI, c + w0, MS // 10, 100, 100,
                        {"type": "send", "rank": "0", "peer": "1", "tag": "7"}))
        t.add(SpanEvent("compute", Category.OPENMP, c, 10 * MS, 200, 200, {"type": "work"}))
        end = c + w0 + 150_000
        t.add(SpanEvent("MPI_Recv", Category.MPI, c + 10 * MS, end - (c + 10 * MS), 200, 200,
                        {"type": "recv", "rank": "1", "peer": "0", "tag": "7"}))
        c = end + 200_000
    return t


def ensure_visible(item: QQuickItem) -> None:
    """Scroll every enclosing Flickable so `item` is on screen, as a user would."""
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


def click(window, item: QQuickItem) -> None:
    ensure_visible(item)
    p = item.mapToScene(QPointF(item.width() / 2, item.height() / 2))
    QTest.mouseMove(window, p.toPoint())
    QTest.mouseClick(window, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, p.toPoint())
    QTest.qWait(50)


def main() -> None:
    candidate, baseline = mpi_trace(True), mpi_trace(False)
    H._build_trace = lambda: candidate
    class PopulatedBridge(ComparisonBridge):
        def __init__(self, trace, _none, theme, parent=None):
            super().__init__(trace, baseline, theme, parent)
    H.ComparisonBridge = PopulatedBridge
    H.TestTimelineHover.setUpClass()
    C = H.TestTimelineHover
    out: dict = {}
    nav, cmpb, win, root = C._selection, C._comparison, C._window, C._root
    C._warnings.clear()
    nav.navigateTo(9)
    QTest.qWait(300)
    screen = next(i for i in root.findChildren(QQuickItem)
                  if i.property("selectedId") is not None and i.property("cpPalette") is not None)
    lst = root.findChild(QObject, "compareContributorList")
    out["available"] = cmpb.available
    out["causal"] = cmpb.causalAvailable
    out["contributors"] = [c["label"] for c in cmpb.contributors]
    out["listCount"] = lst.property("count")
    first = None
    for _ in range(5):
        first = lst.itemAtIndex(0) if hasattr(lst, "itemAtIndex") else None
        if first is None:
            from PySide6.QtCore import QMetaObject, Q_RETURN_ARG, Q_ARG
            first = QMetaObject.invokeMethod(lst, "itemAtIndex", Q_RETURN_ARG(QQuickItem), Q_ARG(int, 0))
        if first is not None:
            break
        QTest.qWait(100)
    for _ in range(3):        # synthetic input may need priming under offscreen
        click(win, first)
        if screen.property("selectedId") >= 0:
            break
    out["selectedId"] = screen.property("selectedId")
    out["firstId"] = cmpb.contributors[0]["id"] if cmpb.contributors else None
    expl = root.findChild(QObject, "compareDetailExplanation")
    out["explanation"] = expl.property("text") if expl else ""
    detail = screen.property("detail")
    detail = detail.toVariant() if hasattr(detail, "toVariant") else detail
    out["detailLabel"] = detail.get("label")
    out["snippetLines"] = len((detail.get("snippet") or {}).get("lines", []))
    out["rangeNs"] = [detail.get("timelineStartNs"), detail.get("timelineEndNs")]

    btn = root.findChild(QObject, "compareShowInTimelineButton")
    click(win, btn)
    QTest.qWait(300)
    out["afterTimelineTab"] = nav.currentTab
    out["focusRange"] = dict(nav.focusRange)
    out["selectedName"] = nav.selectedName
    tl = next(i for i in root.findChildren(QQuickItem)
              if i.property("hoverText") is not None and i.property("visibleNs") is not None)
    out["timelineView"] = [tl.property("viewStartNs"), tl.property("viewStartNs") + tl.property("visibleNs")]

    nav.navigateTo(9)
    QTest.qWait(200)
    cmpb.selectPhase(2)
    QTest.qWait(100)
    out["phaseSelected"] = cmpb.selectedPhase
    out["phaseContributors"] = [c["label"] for c in cmpb.contributors]
    out["cpBefore"] = [c["label"] for c in cmpb.criticalBefore]
    out["cpAfter"] = [c["label"] for c in cmpb.criticalAfter]
    pbtn = root.findChild(QObject, "compareShowPhaseButton")
    click(win, pbtn)
    QTest.qWait(300)
    out["afterPhaseTab"] = nav.currentTab
    out["phaseFocus"] = dict(nav.focusRange)
    out["phaseRange"] = dict(cmpb.phaseRange(2))

    nav.navigateTo(9)
    QTest.qWait(200)
    cmpb.selectPhase(-1)
    src_btn = root.findChild(QObject, "compareOpenSourceButton")
    click(win, src_btn)
    QTest.qWait(200)
    out["afterSourceTab"] = nav.currentTab
    out["sourceSelected"] = nav.selectedName

    nav.navigateTo(9)
    QTest.qWait(100)
    cmpb.setChangeThresholds(1e12, 5.0)       # nothing clears this floor
    QTest.qWait(100)
    out["contributorsAtHugeFloor"] = len(cmpb.contributors)
    out["warnings"] = list(C._warnings)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
