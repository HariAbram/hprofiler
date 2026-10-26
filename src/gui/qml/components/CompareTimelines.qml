import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import Hprofiler 1.0

// Two runs' timelines, baseline above candidate. Events are matched by
// hprofiler's comparison (identity, aligned phase, occurrence index --
// never by timestamp), and the basis of every match is shown. The two
// views move together (synchronized zoom/pan, shifted by the alignment
// offset) unless "Sync" is off. A delta strip on top shows where the runs
// diverge (per aligned phase); selecting an event highlights its
// counterpart, gives both durations and the change, and traces a
// regression back along the comparison's upstream chain -- observed
// changes and inferred links labelled apart.
ColumnLayout {
    id: root
    objectName: "compareTimelines"
    spacing: AppTheme.spacingXs

    readonly property var base: Compare.baselineTimeline
    readonly property var cand: Compare.candidateTimeline
    readonly property var starts: Compare.timelineStarts
    property bool sync: true
    property string alignMode: "start"
    // the combo follows the mode however it is set (a user pick breaks a
    // plain currentIndex binding, so it is kept in sync imperatively)
    onAlignModeChanged: {
        for (var i = 0; i < alignBox.model.length; i++)
            if (alignBox.model[i].k === alignMode) { alignBox.currentIndex = i; break }
    }
    property real offsetNs: (starts.candidate || 0) - (starts.baseline || 0)
    property string offsetBasis: "application start (each run's first event)"
    property real candStart: starts.candidate || 0
    property real candVisible: Math.max(1, starts.candidateDuration || 1)
    property real baseStart: starts.baseline || 0
    property real baseVisible: Math.max(1, starts.baselineDuration || 1)
    property real cursorCand: -1
    property var match: ({})
    property string matchSide: ""
    property int chainIdentity: -1

    function setCand(start, visible) {
        candStart = start
        candVisible = visible
        if (sync) { baseStart = start - offsetNs; baseVisible = visible }
    }
    function setBase(start, visible) {
        baseStart = start
        baseVisible = visible
        if (sync) { candStart = start + offsetNs; candVisible = visible }
    }
    function fit() {
        setCand(starts.candidate || 0, Math.max(1, starts.candidateDuration || 1))
        if (!sync) { baseStart = starts.baseline || 0; baseVisible = Math.max(1, starts.baselineDuration || 1) }
    }
    function applyAlignment() {
        var phase = Compare.selectedPhase
        var ident = root.chainIdentity
        var b = -1, c = -1
        if (root.match.counterpart && root.matchSide.length) {
            var cp = root.match.counterpart
            var mine = root.matchSide === "candidate" ? root.cand.spanAt(root.match.myLane, root.match.myEid)
                                                      : root.base.spanAt(root.match.myLane, root.match.myEid)
            var mineAbs = (root.matchSide === "candidate" ? root.cand.viewStartNs : root.base.viewStartNs) + (mine.startNs || 0)
            if (root.matchSide === "candidate") { c = mineAbs; b = cp.startNs } else { b = mineAbs; c = cp.startNs }
        }
        var r = Compare.alignmentOffset(root.alignMode, phase, ident, b, c)
        root.offsetNs = r.offsetNs
        root.offsetBasis = r.basis
        if (root.alignMode === "phase" && phase >= 0) {
            var pr = Compare.phaseRange(phase)
            if (pr.startNs !== undefined) { setCand(pr.startNs - (pr.endNs - pr.startNs) * 0.05, (pr.endNs - pr.startNs) * 1.1); return }
        }
        if (root.alignMode === "event" && c > 0) { setCand(c - candVisible / 2, candVisible); return }
        setCand(candStart, candVisible)
    }
    function selectIn(side, laneIndex, spanIdx) {
        var m = Compare.matchEvent(side, spanIdx)
        m.myLane = laneIndex
        m.myEid = spanIdx
        root.match = m
        root.matchSide = side
        root.chainIdentity = m.identity !== undefined ? m.identity : -1
        var mine = side === "candidate" ? root.cand : root.base
        var other = side === "candidate" ? root.base : root.cand
        mine.selectEvent(laneIndex, spanIdx)
        if (m.counterpart && m.counterpart.laneIndex >= 0) {
            other.selectEvent(m.counterpart.laneIndex, m.counterpart.eid)
            if (side === "candidate") basePane.revealLane(m.counterpart.laneIndex)
            else candPane.revealLane(m.counterpart.laneIndex)
        } else {
            other.clearEventSelection()
        }
        if (root.alignMode === "event") applyAlignment()
    }

    Component.onCompleted: fit()

    RowLayout {
        Layout.fillWidth: true
        spacing: AppTheme.spacingSm
        CheckBox {
            objectName: "compareTimelinesSync"
            text: "Sync zoom/pan"
            checked: root.sync
            onToggled: { root.sync = checked; if (checked) root.setCand(root.candStart, root.candVisible) }
        }
        Text { text: "Align by"; color: AppTheme.textMuted; font.pixelSize: AppTheme.typeCaption }
        ComboBox {
            id: alignBox
            objectName: "compareTimelinesAlign"
            implicitHeight: AppTheme.buttonHeight
            model: [{"k": "start", "t": "application start"}, {"k": "event", "t": "selected event"},
                    {"k": "phase", "t": "selected phase"}, {"k": "region", "t": "contributor region"}]
            textRole: "t"
            onActivated: { root.alignMode = model[currentIndex].k; root.applyAlignment() }
        }
        Text {
            Layout.fillWidth: true
            Layout.minimumWidth: 0
            text: "offset " + Format.signedNs(root.offsetNs) + " — " + root.offsetBasis
            color: AppTheme.textMuted
            font.pixelSize: AppTheme.typeCaption
            elide: Text.ElideRight
        }
        ToolButton {
            text: "Fit"
            implicitHeight: AppTheme.buttonHeight
            onClicked: root.fit()
        }
        CheckBox {
            objectName: "compareTimelinesStatusColors"
            text: "Colour by change"
            checked: root.cand && root.cand.colorMode === "comparison"
            onToggled: {
                root.cand.setColorMode(checked ? "comparison" : "function")
                root.base.setColorMode(checked ? "comparison" : "function")
            }
            ToolTip.visible: hovered
            ToolTip.text: "red slower / green faster / cyan new / yellow removed / grey unchanged or unmatched (comparison status of each event's identity)"
        }
        CheckBox {
            objectName: "compareTimelinesCriticalPath"
            text: "Critical path (both)"
            onToggled: {
                root.cand.setOverlay({"criticalPath": checked})
                root.base.setOverlay({"criticalPath": checked})
            }
        }
    }

    // ── where the runs diverge: aligned phases on the candidate's clock ──
    Item {
        Layout.fillWidth: true
        Layout.preferredHeight: 14
        Repeater {
            objectName: "compareDeltaStrip"
            model: Compare.deltaStrip
            delegate: Rectangle {
                readonly property real a: modelData.side === "candidate" ? modelData.startNs : modelData.startNs + root.offsetNs
                readonly property real b: modelData.side === "candidate" ? modelData.endNs : modelData.endNs + root.offsetNs
                x: candPane.labelWidth + Math.max(0, (a - root.candStart) / root.candVisible) * (parent.width - candPane.labelWidth - 12)
                width: Math.max(1, Math.min(b - Math.max(a, root.candStart), root.candStart + root.candVisible - Math.max(a, root.candStart))
                                / root.candVisible * (parent.width - candPane.labelWidth - 12))
                visible: b > root.candStart && a < root.candStart + root.candVisible
                height: 12
                y: 1
                color: AppTheme.changeColor(modelData.pairStatus !== "matched" ? (modelData.side === "candidate" ? "new" : "removed")
                                            : modelData.deltaStatus, AppTheme.dark)
                opacity: Compare.selectedPhase === modelData.index ? 1.0 : 0.7
                border.width: Compare.selectedPhase === modelData.index ? 2 : 0
                border.color: AppTheme.text
                MouseArea {
                    anchors.fill: parent
                    hoverEnabled: true
                    onClicked: { Compare.selectPhase(modelData.index); if (root.alignMode === "phase") root.applyAlignment() }
                    ToolTip.visible: containsMouse
                    ToolTip.text: modelData.label + ": " + modelData.pairStatus + ", Δ " + Format.signedNs(modelData.deltaNs)
                                  + ", critical path Δ " + Format.signedNs(modelData.criticalDeltaNs)
                }
            }
        }
        Text {
            x: 4
            text: "Δ per phase"
            color: AppTheme.textMuted
            font.pixelSize: AppTheme.typeCaption
        }
    }

    TimelinePane {
        id: basePane
        objectName: "compareBaselinePane"
        Layout.fillWidth: true
        Layout.preferredHeight: 190
        side: "baseline"
        title: "Baseline (--compare)" + (AppInfo.comparePath ? ": " + AppInfo.comparePath : "")
        model: root.base
        viewStartNs: root.baseStart
        visibleNs: root.baseVisible
        cursorNs: root.cursorCand >= 0 ? root.cursorCand - root.offsetNs : -1
        onViewRequested: (s, v) => root.setBase(s, v)
        onEventClicked: (lane, eid) => root.selectIn("baseline", lane, eid)
        onCursorMoved: (ns) => root.cursorCand = ns >= 0 ? ns + root.offsetNs : -1
    }
    TimelinePane {
        id: candPane
        objectName: "compareCandidatePane"
        Layout.fillWidth: true
        Layout.preferredHeight: 190
        side: "candidate"
        title: "Candidate (opened)" + (AppInfo.tracePath ? ": " + AppInfo.tracePath : "")
        model: root.cand
        viewStartNs: root.candStart
        visibleNs: root.candVisible
        cursorNs: root.cursorCand
        onViewRequested: (s, v) => root.setCand(s, v)
        onEventClicked: (lane, eid) => root.selectIn("candidate", lane, eid)
        onCursorMoved: (ns) => root.cursorCand = ns
    }
    Text {
        Layout.fillWidth: true
        visible: basePane.hoverText.length > 0 || candPane.hoverText.length > 0
        text: candPane.hoverText.length > 0 ? "candidate: " + candPane.hoverText : "baseline: " + basePane.hoverText
        color: AppTheme.text
        font.pixelSize: AppTheme.typeCaption
        elide: Text.ElideRight
    }

    // ── the selected event and its counterpart ──────────────────────────
    RowLayout {
        Layout.fillWidth: true
        visible: root.matchSide.length > 0
        spacing: AppTheme.spacingLg
        ColumnLayout {
            Layout.preferredWidth: 460
            Layout.minimumWidth: 0
            spacing: 1
            Text {
                objectName: "compareMatchTitle"
                Layout.fillWidth: true
                wrapMode: Text.WrapAtWordBoundaryOrAnywhere
                text: (root.match.label || "") + "  [" + (root.match.roles || "") + "]  — " + (root.match.status || "")
                color: AppTheme.changeColor(root.match.status || "unchanged", AppTheme.dark)
                font.bold: true
                font.pixelSize: AppTheme.typeLabel
            }
            Text {
                objectName: "compareMatchDurations"
                Layout.fillWidth: true
                wrapMode: Text.Wrap
                visible: root.match.durationNs !== undefined && root.match.durationNs[0] !== null && root.match.durationNs[1] !== null
                text: visible ? ("baseline " + Format.duration(root.match.durationNs[0]) + " → candidate "
                                 + Format.duration(root.match.durationNs[1]) + "   Δ " + Format.signedNs(root.match.deltaNs)
                                 + (root.match.relDelta !== null && root.match.relDelta !== undefined
                                    ? " (" + (100 * root.match.relDelta).toFixed(1) + "%)" : "") + "   [measured]") : ""
                color: AppTheme.text
                font.pixelSize: AppTheme.typeCaption
            }
            Text {
                Layout.fillWidth: true
                objectName: "compareMatchBasis"
                text: "match: " + (root.match.basis || "") + (root.match.confidence !== undefined && root.match.confidence !== null
                      ? "   confidence " + root.match.confidence.toFixed(2) + " [heuristic]" : "")
                color: AppTheme.textMuted
                font.pixelSize: AppTheme.typeCaption
                wrapMode: Text.Wrap
            }
            Text {
                Layout.fillWidth: true
                visible: (root.match.explanation || "").length > 0
                text: "why: " + (root.match.explanation || "") + "   [derived]"
                color: AppTheme.text
                font.pixelSize: AppTheme.typeCaption
                wrapMode: Text.Wrap
            }
        }
        ColumnLayout {
            Layout.fillWidth: true
            Layout.minimumWidth: 0
            spacing: 1
            Text {
                Layout.fillWidth: true
                wrapMode: Text.Wrap
                text: "Trace back (observed change first, inferred links labelled)"
                color: AppTheme.text
                font.bold: true
                font.pixelSize: AppTheme.typeCaption
                visible: (root.match.steps || []).length > 1
            }
            Repeater {
                objectName: "compareRegressionSteps"
                model: root.match.steps || []
                delegate: Text {
                    Layout.fillWidth: true
                    text: (index === 0 ? "" : "→ ") + modelData.label + " [" + modelData.roles + "]: " + modelData.text
                          + "   (" + modelData.kind + ")"
                    color: index === 0 ? AppTheme.text : AppTheme.textMuted
                    font.pixelSize: AppTheme.typeCaption
                    wrapMode: Text.Wrap
                    MouseArea {
                        anchors.fill: parent
                        cursorShape: Qt.PointingHandCursor
                        onClicked: {
                            var h = Compare.identitySpans(modelData.identity)
                            if (h.names) { root.cand.setHighlight(h); root.base.setHighlight(h) }
                        }
                    }
                }
            }
        }
    }

    Connections {
        target: root.cand
        function onAnalysisChanged() { Compare.refreshTimelines() }
    }
    Connections {
        target: root.base
        function onAnalysisChanged() { Compare.refreshTimelines() }
    }
    Text {
        objectName: "compareEvidenceChange"
        readonly property var ev: Compare.evidenceChange
        visible: ev.baselineState === "ready" && ev.candidateState === "ready"
        text: {
            if (!visible) return ""
            var tiers = ["certain", "high", "medium", "low"], parts = []
            for (var i = 0; i < tiers.length; i++) {
                var t = tiers[i], a = ev.baseline[t] || 0, b = ev.candidate[t] || 0
                if (a || b) parts.push(t + " " + Format.duration(a) + " → " + Format.duration(b))
            }
            return "Critical-path time by evidence tier: " + parts.join(" · ") + "  (tiers are evidence classes, not probabilities)"
        }
        color: AppTheme.textMuted
        font.pixelSize: AppTheme.typeCaption
        Layout.fillWidth: true
        wrapMode: Text.Wrap
    }

    // ── structural differences (ranks / threads / streams / devices) ────
    Text {
        visible: Compare.structureDiff.length > 0
        text: "Structure: " + Compare.structureDiff.map(function(d) { return d.rank + ": " + d.change }).join(" · ")
        color: AppTheme.textMuted
        font.pixelSize: AppTheme.typeCaption
        Layout.fillWidth: true
        wrapMode: Text.Wrap
    }
}
