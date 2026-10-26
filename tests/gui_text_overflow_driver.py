"""
Subprocess driver for tests/test_gui_text_overflow.py: loads the real
Main.qml (through the shared GUI fixture of tests/test_gui_timeline_hover.py)
on a Comb-like MPI + CUDA trace with the application font set to
argv[1] points and the window resized to argv[2] (WIDTHxHEIGHT), opens the
Timeline's popups, the bottom panel's tabs and the details panel, and prints
one JSON line: for each frame (popup, summary card, panel), the visible
texts painted past its right edge. Its own process because the font has to
be set before the one QML engine of a process is created.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtCore import QMetaObject, QObject, QPointF
from PySide6.QtGui import QGuiApplication
from PySide6.QtTest import QTest

FONT_PT = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
W, H_ = (int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "1280x800").split("x"))

app = QGuiApplication.instance() or QGuiApplication([sys.argv[0]])
if FONT_PT:
    f = app.font()
    f.setPointSizeF(FONT_PT)
    app.setFont(f)

import tests.test_gui_timeline_hover as H          # noqa: E402
from tests.test_criticalpath_comb import comb_cycle  # noqa: E402


def walk(item):
    yield item
    for ch in item.childItems():
        yield from walk(ch)


def shown(it) -> bool:
    a = it
    while a is not None:
        if not a.isVisible() or a.opacity() == 0:
            return False
        a = a.parentItem()
    return True


def overflowing(frame, right_limit=None) -> list[dict]:
    """Visible texts under `frame` whose painted extent crosses its right edge."""
    left = frame.mapToScene(QPointF(0, 0)).x()
    right = right_limit if right_limit is not None else left + frame.width()
    out = []
    for it in walk(frame):
        mo = it.metaObject()
        if it is frame or any(mo.indexOfProperty(k) < 0 for k in ("contentWidth", "wrapMode", "elide", "text")):
            continue
        text = it.property("text") or ""
        if not text.strip() or not shown(it):
            continue
        x0 = it.mapToScene(QPointF(0, 0)).x()
        painted = x0 + max(0.0, it.property("contentWidth") or 0.0)
        if painted > right + 1.0:
            out.append({"text": text[:80], "over_px": round(painted - right, 1)})
    return out


def main() -> None:
    H._build_trace = lambda: comb_cycle(cycles=2)
    H.TestTimelineHover.setUpClass()
    C = H.TestTimelineHover
    win, root, m, nav = C._window, C._root, C._timeline_model, C._selection
    win.resize(W, H_)
    nav.navigateTo(1)
    QTest.qWait(300)
    m.startAnalysis()
    res = {"font": f"{app.font().pointSizeF()}pt", "size": f"{W}x{H_}", "frames": {},
           "analysis": m.wait_for_analysis(120)}
    QTest.qWait(150)
    items = list(walk(win.contentItem()))

    def find_qobject(name):
        for it in items:
            p = it.findChild(QObject, name)
            if p is not None:
                return p
        return None

    for name in ("timelineCriticalPathPopup", "timelineDependenciesPopup", "timelineExportPopup"):
        p = find_qobject(name)
        p.setProperty("visible", True)
        QTest.qWait(200)
        bg = p.property("background")
        res["frames"][name] = overflowing(p.property("contentItem"),
                                          bg.mapToScene(QPointF(0, 0)).x() + bg.width())
        p.setProperty("visible", False)
        QTest.qWait(80)

    panel = next(i for i in items if i.objectName() == "timelineBottomPanel")
    panel.setProperty("expanded", True)
    for tab, tag in ((0, "summary"), (1, "criticalPathTab"), (2, "heatmap")):
        panel.setProperty("tab", tab)
        QMetaObject.invokeMethod(panel, "refresh")
        QTest.qWait(300)
        if tab == 0:
            flow = next(i for i in walk(panel) if i.metaObject().className().startswith("QQuickFlow") and i.isVisible())
            cards = [c for c in flow.childItems() if c.isVisible() and c.width() > 0]
            res["summaryCards"] = len(cards)
            for k, card in enumerate(cards):
                res["frames"][f"summaryCard{k}"] = overflowing(card)
        res["frames"][tag] = overflowing(panel)

    lane = next(i for i, ex in enumerate(m._lane_extra) if ex["category"] == "mpi")
    s = next(iter(m._store.iter_spans(lane=m._lane_names[lane])))
    m.selectEvent(lane, float(s.eid))
    QTest.qWait(250)
    dp = next((i for i in walk(win.contentItem()) if i.objectName() == "timelineDetailsPanel"), None)
    res["detailsPanel"] = dp is not None and dp.isVisible()
    if dp is not None:
        res["frames"]["detailsPanel"] = overflowing(dp)
    res["warnings"] = list(C._warnings)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
