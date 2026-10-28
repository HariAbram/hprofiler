"""
Subprocess driver for tests/test_gui_evidence_interaction.py: loads the real
Main.qml on tests/evidence_scenarios.py's trace through the shared GUI
fixture of tests/test_gui_timeline_hover.py, computes the analysis, opens
the bottom panel's Critical path tab and the details panel, and prints one
JSON line with what the QML shows. One QML engine / trace per process.

  python3 tests/gui_evidence_driver.py [default|big|calibrated] [SHOT_DIR]

big         the alternatives' automatic limit lowered below the trace: "not
            evaluated" until "Evaluate alternative worlds" is clicked
calibrated  HPROFILER_EDGE_CALIBRATION set to a synthetic calibration
SHOT_DIR    also save screenshots there
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtCore import QPointF, Qt
from PySide6.QtQuick import QQuickItem
from PySide6.QtTest import QTest

MODE = sys.argv[1] if len(sys.argv) > 1 else "default"
SHOTS = Path(sys.argv[2]) if len(sys.argv) > 2 else None
if MODE == "calibrated":
    from tests import evidence_scenarios as _E
    os.environ["HPROFILER_EDGE_CALIBRATION"] = _E.calibration_file(
        str(Path(tempfile.mkdtemp(prefix="hprofiler_cal_")) / "cal.json"))

import tests.test_gui_timeline_hover as H
from src.analysis import criticalpath as cp
from tests import evidence_scenarios as E


def walk(item):
    yield item
    for ch in item.childItems():
        yield from walk(ch)


def find(win, name: str):
    return next((i for i in walk(win.contentItem()) if i.objectName() == name), None)


def find_all(win, prefix: str) -> list:
    return [i for i in walk(win.contentItem()) if i.objectName().startswith(prefix)]


def text_of(win, name: str) -> str:
    it = find(win, name)
    return it.property("text") if it is not None else None


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


def details_fields(root_panel_details) -> dict:
    """{label: value} of the event details' Critical path section."""
    d = root_panel_details.toVariant() if hasattr(root_panel_details, "toVariant") else root_panel_details
    for sec in d.get("sections", []):
        if sec["title"] == "Critical path":
            return {f["label"]: (f["value"] if f["kind"] != "unavailable" else "— " + f["reason"]) for f in sec["fields"]}
    return {}


def main() -> None:
    trace = E.big_trace() if MODE == "big" else E.evidence_trace()
    if MODE == "big":
        cp.ALTERNATIVES_AUTO_NODE_LIMIT = 50          # below this trace's event nodes
    H._build_trace = lambda: trace
    H.TestTimelineHover.setUpClass()
    C = H.TestTimelineHover
    m, win, root = C._timeline_model, C._window, C._root
    C._workspace_settings.save_timeline_overlay_dismissed(True)
    C._workspace._timeline_overlay_dismissed = True
    C._workspace.timelineOverlayDismissedChanged.emit()
    C._warnings.clear()
    out: dict = {"mode": MODE}
    m.setOverlay({"criticalPath": True, "families": ["mpi_p2p", "mpi_collective", "gpu_sync", "submission"]})
    out["ready"] = m.wait_for_analysis(120)
    QTest.qWait(200)
    an = m._analysis
    tl = next(i for i in root.findChildren(QQuickItem)
              if i.property("hoverText") is not None and i.property("visibleNs") is not None)
    bottom = find(win, "timelineBottomPanel")
    bottom.setProperty("expanded", True)
    bottom.setProperty("tab", 1)
    QTest.qWait(300)
    out["altState"] = text_of(win, "cpAltState")
    out["altReason"] = text_of(win, "cpAltReason")
    out["rangeText"] = text_of(win, "cpRangeText")
    out["legacyText"] = text_of(win, "cpLegacyText")
    out["variantModeled"] = {n.objectName().replace("cpVariantModeled_", ""): n.property("text")
                             for n in find_all(win, "cpVariantModeled_")}
    cells = [i.property("text") for i in walk(find(win, "timelineCpNumbers")) if i.property("text") is not None]
    out["cpNumbers"] = cells
    out["probText"] = text_of(win, "cpProbText")
    for key in ("evidence", "clocks", "blame", "sensitivity", "probabilities"):
        hdr = find(win, "cpSectionHeader_" + key)
        if hdr is not None:
            click(win, hdr)
    QTest.qWait(200)
    body = find(win, "cpSectionBody_evidence")
    out["evidenceOpen"] = bool(body and body.property("visible"))
    out["statusCells"] = [i.property("text") for i in walk(find(win, "cpStatusTable")) if i.property("text") is not None]
    out["clockCells"] = [i.property("text") for i in walk(find(win, "cpClockTable")) if i.property("text") is not None]
    out["undecidableRows"] = len(find_all(win, "cpUndecidableRow_"))
    out["worldRows"] = len(find_all(win, "cpWorldRow_"))
    out["sensitivityCells"] = [i.property("text") for i in walk(find(win, "cpSensitivityTable"))
                               if i.property("text") is not None]
    out["passedRows"] = [i.property("label") for i in find_all(win, "cpPassedRow_")]

    # an ambiguous / inferred point-to-point edge in the details panel
    p2p = [k for k, e in enumerate(an.edges) if e.kind == "p2p"]
    amb = next((k for k in p2p if an.edges[k].status == "ambiguous"), p2p[0] if p2p else -1)
    out["edgeStatusModel"] = an.edges[amb].status if amb >= 0 else ""
    if amb >= 0:
        m.selectEdge(amb)
        QTest.qWait(250)
        fields = {i.objectName().replace("timelineEdgeField_", ""): i.property("value")
                  for i in find_all(win, "timelineEdgeField_")}
        out["edgeFields"] = fields
        out["edgeCandidates"] = len([i for i in walk(find(win, "timelineDetailsPanel"))
                                     if i.property("text") and "(chosen)" not in str(i.property("text"))
                                     and str(i.property("text")).startswith(("unsupported", "contradicted"))])
        out["edgeTreeLines"] = [str(i.property("text")) for i in walk(find(win, "timelineDetailsPanel"))
                                if str(i.property("text") or "").startswith(("✓", "✗", "↳"))]
    m.clearEventSelection()

    # events: an unknown blocking call, a gated receive, a span on the path
    spans = list(trace.iter_spans())
    unk = next((s for s in spans if s.eid in an.unknown_eids), None)
    gated = next((s for s in spans if s.eid in an.gated_of_eid and s.name == "MPI_Recv"), None)
    on_path = next((s for s in spans if s.eid in an.path_eids and s.name not in ("MPI_Recv", "MPI_Send")), None)
    for key, s in (("unknown", unk), ("gated", gated), ("onPath", on_path)):
        if s is None:
            continue
        lane = m._lane_of_span(s)
        m.selectEvent(lane, float(s.eid))
        QTest.qWait(200)
        out[key + "Fields"] = details_fields(tl.property("selDetails"))
        out[key + "Hover"] = m.spanAt(lane, float(s.eid)).get("analysisNote", "")
    m.clearEventSelection()

    # unknown time has its own band code; membership marks and the chip
    out["cpModes"] = sorted({st.mode for st in an.steps})
    lane_unk = m._lane_of_span(unk) if unk is not None else -1
    marks = {}
    for mode in ("variants", "near", "worlds"):
        m.setOverlay({"membership": mode})
        QTest.qWait(120)
        out.setdefault("chip", {})[mode] = text_of(win, "timelineMembershipChipText")
        n = 0
        for li in range(len(m._lane_names)):
            lr = m.render_lane(li, m.viewStartNs, m.viewStartNs + m.traceDurationNs, 800)
            n += len(m._lane_options(li, lr).marks or [])
        marks[mode] = n
    out["marks"] = marks
    m.setOverlay({"membership": "", "unknownBlocking": True})
    if lane_unk >= 0:
        lr = m.render_lane(lane_unk, m.viewStartNs, m.viewStartNs + m.traceDurationNs, 800)
        out["unknownVeils"] = len(m._lane_options(lane_unk, lr).unknown_calls or [])
    m.setOverlay({"unknownBlocking": False})

    if MODE == "big":
        # never "every world" before the evaluation; then evaluate on request
        out["beforeEvalOnPath"] = out.get("onPathFields", {}).get("Across alternatives", "")
        m.setOverlay({"membership": "worlds"})
        QTest.qWait(100)
        out["chipNotEvaluated"] = text_of(win, "timelineMembershipChipText")
        m.setOverlay({"membership": ""})
        if SHOTS:
            # slow each world down so the busy state can be seen and captured
            import time
            from src.gui import timeline_analysis as ta
            orig = ta.evaluate_alternatives

            def slow(an_, trace_, samples, progress=None, cancel=None):
                def p(d, t):
                    time.sleep(0.6)
                    progress(d, t)
                return orig(an_, trace_, samples, progress=p, cancel=cancel)
            ta.evaluate_alternatives = slow
            win.resize(1600, 1050)
            if C._selection.inspectorOpen:
                C._selection.toggleInspector()
            bottom.setProperty("expanded", True)
            bottom.setProperty("tab", 1)
            QTest.qWait(300)
            scroll(win, "timelineCpScroll", "cpSectionHeader_alternatives")
            win.grabWindow().save(str(SHOTS / "bottom_panel_not_evaluated.png"))
        btn = find(win, "cpAltEvaluate")
        hdr = find(win, "cpSectionBody_alternatives")
        out["evaluateButton"] = btn.property("text") if btn else None
        click(win, btn)
        out["busySeen"] = bool(m.alternativesJob.get("running"))
        if SHOTS:
            QTest.qWait(900)
            busy = find(win, "cpAltBusy")
            out["busyIndicatorRunning"] = bool(busy and busy.property("running"))
            win.grabWindow().save(str(SHOTS / "bottom_panel_evaluating.png"))
        out["evalOk"] = m.wait_for_alternatives(120)
        QTest.qWait(300)
        out["altStateAfter"] = text_of(win, "cpAltState")
        out["altReasonAfter"] = text_of(win, "cpAltReason")
        out["worldRowsAfter"] = len(find_all(win, "cpWorldRow_"))
        s = on_path
        if s is not None:
            m.selectEvent(m._lane_of_span(s), float(s.eid))
            QTest.qWait(200)
            out["afterEvalOnPath"] = details_fields(tl.property("selDetails")).get("Across alternatives", "")
        out["jobMessage"] = m.alternativesJob.get("message")
        if hdr is not None:
            out["altBodyVisible"] = bool(hdr.property("visible"))
    out["warnings"] = list(C._warnings)
    if SHOTS:
        screenshots(C, m, win, tl, bottom, amb)
    print(json.dumps(out, default=str))


def scroll(win, view: str, target: str | None) -> None:
    """Scroll a ScrollView (objectName `view`) to the top, or so the item
    named `target` is at its top."""
    sv = find(win, view)
    if sv is None:
        return
    fl = sv.property("contentItem")
    y = 0.0
    if target is not None:
        it = find(win, target)
        if it is not None:
            y = max(0.0, it.mapToItem(fl.property("contentItem"), QPointF(0, 0)).y() - 4)
    fl.setProperty("contentY", y)
    QTest.qWait(150)


def screenshots(C, m, win, tl, bottom, edge) -> None:
    """The corrected views, framed for reading: the path's lanes, the
    bottom panel's sections, an edge's details."""
    win.resize(1600, 1050)
    if C._selection.inspectorOpen:
        C._selection.toggleInspector()
    QTest.qWait(200)
    tag = MODE
    m.setOverlay({"criticalPath": True, "focusPath": True, "unknownBlocking": True, "membership": "variants",
                  "disabledEdges": True})
    bottom.setProperty("expanded", False)
    tl.setProperty("viewStartNs", m.viewStartNs)
    tl.setProperty("zoom", 1.0)
    QTest.qWait(400)
    win.grabWindow().save(str(SHOTS / f"timeline_{tag}.png"))
    m.setOverlay({"membership": ""})
    bottom.setProperty("expanded", True)
    bottom.setProperty("tab", 1)
    QTest.qWait(300)
    for key in ("variants", "alternatives", "evidence", "clocks", "blame", "sensitivity", "probabilities"):
        hdr = find(win, "cpSectionBody_" + key)
        if hdr is not None and not hdr.property("visible"):
            click(win, find(win, "cpSectionHeader_" + key))
    scroll(win, "timelineCpScroll", None)
    win.grabWindow().save(str(SHOTS / f"bottom_panel_{tag}.png"))
    scroll(win, "timelineCpScroll", "cpSectionHeader_variants")
    win.grabWindow().save(str(SHOTS / f"bottom_panel_variants_{tag}.png"))
    scroll(win, "timelineCpScroll", "cpSectionHeader_alternatives")
    win.grabWindow().save(str(SHOTS / f"bottom_panel_alternatives_{tag}.png"))
    scroll(win, "timelineCpScroll", "cpSectionHeader_evidence")
    win.grabWindow().save(str(SHOTS / f"bottom_panel_evidence_clocks_{tag}.png"))
    scroll(win, "timelineCpScroll", "cpSectionHeader_blame")
    win.grabWindow().save(str(SHOTS / f"bottom_panel_blame_{tag}.png"))
    if edge >= 0:
        bottom.setProperty("expanded", False)
        m.selectEdge(edge)
        QTest.qWait(300)
        win.grabWindow().save(str(SHOTS / f"edge_details_{tag}.png"))
        scroll(win, "timelineDetailsScroll", "timelineEdgeField_alternatives")
        win.grabWindow().save(str(SHOTS / f"edge_details_evidence_{tag}.png"))


if __name__ == "__main__":
    main()
