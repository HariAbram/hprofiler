import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import Hprofiler 1.0
import "../components"

// Comparison mode -- always tab index 9 ("GUI tabs are always visible"
// convention), backed by the "Compare" singleton (always registered, even
// with no comparison trace loaded -- see comparison.py). The opened trace is
// the CANDIDATE, `--compare` names the BASELINE.
//
// Top to bottom: verdict (wall time, alignment method + confidence),
// phase navigation, ranked causal contributors with a details panel
// (measured / derived / heuristic evidence, click-through to the Timeline
// range and the source location), before/after critical-path composition,
// propagated waits and off-path changes, unavailable conclusions, then the
// (category, name) aggregate view kept for compatibility.
Item {
    id: root

    property int selectedId: -1
    readonly property var detail: {
        var _deps = [Compare.verdict, Compare.selectedPhase]   // refresh on recompute / phase change
        return root.selectedId >= 0 ? Compare.contributorDetail(root.selectedId) : ({})
    }
    readonly property var cpPalette: ["openmp", "cuda", "mpi", "sync", "memory", "nvtx", "jit", "rocm", "opencl"]

    function causeChipColor(cause) { return AppTheme.causeColor(cause || "") }
    function cpColor(identity) {
        return identity < 0 ? AppTheme.textMuted : AppTheme.categoryColor(cpPalette[identity % cpPalette.length])
    }
    function showInTimeline(d) {
        if (!d || !d.hasTimeline) return
        if (d.category && d.rawName) Nav.selectFunction(d.category, d.rawName)
        Nav.focusTimeRange(d.timelineStartNs, d.timelineEndNs)
        Nav.navigateTo(1)
    }
    function openSource(d) {
        if (!d || !d.rawName) return
        Nav.selectFunction(d.category, d.rawName)
        Nav.navigateTo(6)
    }
    function showPhase(index) {
        var r = Compare.phaseRange(index)
        if (r.startNs === undefined) return
        Nav.focusTimeRange(r.startNs, r.endNs)
        Nav.navigateTo(1)
    }

    ScreenState {
        objectName: "compareScreenState"
        anchors.fill: parent
        state: Compare.available ? "ready" : "empty"
        emptyMonospace: true
        emptyMessage: "No comparison trace loaded.\n\n" +
                 "hprofiler gui AFTER.json --compare BEFORE.json\n\n" +
                 "The opened trace is the candidate; --compare names the baseline.\n" +
                 "Terminal: hprofiler compare BEFORE.json AFTER.json"
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: AppTheme.spacingSm
        visible: Compare.available

        Toolbar {
            title: "Compare"
            statusText: (Compare.verdict.methodLabel || "") +
                        (Compare.verdict.confidence !== undefined
                         ? " · confidence " + Compare.verdict.confidence.toFixed(2) : "")

            ToolButton {
                text: "Export report"
                onClicked: {
                    var path = (AppInfo.tracePath || "compare") + ".compare.json"
                    exportStatus.text = Compare.exportReport(path) ? ("Exported to " + path) : "Export failed"
                }
            }
            ToolButton {
                text: "Export CSV"
                onClicked: {
                    var path = (AppInfo.tracePath || "compare") + ".compare.csv"
                    exportStatus.text = Compare.exportCsv(path) ? ("Exported to " + path) : "Export failed"
                }
            }
            Text {
                id: exportStatus
                color: AppTheme.textMuted
                font.pixelSize: AppTheme.typeCaption
            }
        }

        Flickable {
            objectName: "compareFlick"
            Layout.fillWidth: true
            Layout.fillHeight: true
            contentWidth: width
            contentHeight: mainColumn.height
            clip: true
            boundsBehavior: Flickable.StopAtBounds

            ColumnLayout {
                id: mainColumn
                width: parent.width
                spacing: AppTheme.spacingMd

                // ── Summary + verdict ─────────────────────────────────
                RowLayout {
                    Layout.fillWidth: true
                    spacing: AppTheme.spacingMd

                    Panel {
                        title: ""
                        Layout.preferredWidth: 420
                        Layout.preferredHeight: Math.max(summary.implicitHeight, verdictColumn.implicitHeight)
                                                + AppTheme.spacingMd * 2 + 24
                        ComparisonSummary {
                            id: summary
                            anchors.fill: parent
                            baselineFields: Compare.baselineFields
                            comparisonFields: Compare.comparisonFields
                        }
                    }
                    Panel {
                        objectName: "compareVerdict"
                        title: "Verdict"
                        Layout.fillWidth: true
                        Layout.preferredHeight: Math.max(summary.implicitHeight, verdictColumn.implicitHeight)
                                                + AppTheme.spacingMd * 2 + 24
                        ColumnLayout {
                            id: verdictColumn
                            anchors.fill: parent
                            spacing: AppTheme.spacingXs
                            RowLayout {
                                spacing: AppTheme.spacingSm
                                Text { text: "Wall time"; color: AppTheme.textMuted; font.pixelSize: AppTheme.typeCaption; Layout.preferredWidth: 110 }
                                Text { text: Compare.verdict.wallText || ""; color: AppTheme.text; font.pixelSize: AppTheme.typeBody; font.bold: true }
                                ChangeBadge { status: Compare.verdict.wallStatus || "unavailable"; visible: !!Compare.verdict.wallStatus }
                                Text { text: "measured"; color: AppTheme.textMuted; font.pixelSize: AppTheme.typeCaption; font.italic: true }
                            }
                            RowLayout {
                                spacing: AppTheme.spacingSm
                                Text { text: "Critical path"; color: AppTheme.textMuted; font.pixelSize: AppTheme.typeCaption; Layout.preferredWidth: 110 }
                                Text { text: Compare.verdict.criticalText || ""; color: AppTheme.text; font.pixelSize: AppTheme.typeBody }
                                Text { text: "graph-derived"; color: AppTheme.textMuted; font.pixelSize: AppTheme.typeCaption; font.italic: true }
                            }
                            RowLayout {
                                spacing: AppTheme.spacingSm
                                Text { text: "Alignment"; color: AppTheme.textMuted; font.pixelSize: AppTheme.typeCaption; Layout.preferredWidth: 110 }
                                Text {
                                    objectName: "compareAlignmentText"
                                    text: (Compare.verdict.methodLabel || "") + ", confidence " + (Compare.verdict.confidenceText || "—")
                                    color: AppTheme.text
                                    font.pixelSize: AppTheme.typeBody
                                    Layout.fillWidth: true
                                    wrapMode: Text.WordWrap
                                }
                                Text { text: "heuristic"; color: AppTheme.textMuted; font.pixelSize: AppTheme.typeCaption; font.italic: true }
                            }
                            Text {
                                visible: (Compare.verdict.fallbackReason || "").length > 0
                                text: Compare.verdict.fallbackReason || ""
                                color: AppTheme.warningColor
                                font.pixelSize: AppTheme.typeCaption
                                wrapMode: Text.WordWrap
                                Layout.fillWidth: true
                            }
                            Text {
                                visible: (Compare.verdict.decomposition || "").length > 0
                                text: Compare.verdict.decomposition || ""
                                color: AppTheme.textMuted
                                font.pixelSize: AppTheme.typeCaption
                                wrapMode: Text.WordWrap
                                Layout.fillWidth: true
                            }
                            Repeater {
                                model: Compare.verdict.notes || []
                                delegate: Text {
                                    text: "• " + modelData
                                    color: AppTheme.textMuted
                                    font.pixelSize: AppTheme.typeCaption
                                    wrapMode: Text.WordWrap
                                    Layout.fillWidth: true
                                }
                            }
                        }
                    }
                }

                // ── Phase navigation ──────────────────────────────────
                Panel {
                    objectName: "comparePhases"
                    title: "Phases — " + Compare.phasePairs.length + " aligned pair(s); click one to focus"
                    visible: Compare.causalAvailable && Compare.phasePairs.length > 1
                    Layout.fillWidth: true
                    Layout.preferredHeight: phaseColumn.implicitHeight + AppTheme.spacingMd * 2 + 24
                    ColumnLayout {
                        id: phaseColumn
                        anchors.fill: parent
                        spacing: AppTheme.spacingSm
                        Flow {
                            Layout.fillWidth: true
                            spacing: 2
                            Repeater {
                                objectName: "comparePhaseRepeater"
                                model: Compare.phasePairs
                                delegate: Rectangle {
                                    width: 14
                                    height: 18
                                    radius: AppTheme.radiusSmall
                                    color: modelData.status === "inserted" ? AppTheme.changeColor("new")
                                         : modelData.status === "removed" ? AppTheme.changeColor("removed")
                                         : AppTheme.changeColor(modelData.deltaStatus)
                                    opacity: modelData.status === "matched" && modelData.deltaStatus === "unchanged" ? 0.35 : 0.9
                                    border.width: Compare.selectedPhase === modelData.index ? 2 : 0
                                    border.color: AppTheme.text
                                    Accessible.role: Accessible.Button
                                    Accessible.name: modelData.label + ", " + modelData.status + ", " + Format.signedNs(modelData.deltaNs)
                                    Accessible.onPressAction: Compare.selectPhase(Compare.selectedPhase === modelData.index ? -1 : modelData.index)
                                    MouseArea {
                                        id: chipMouse
                                        anchors.fill: parent
                                        hoverEnabled: true
                                        onClicked: Compare.selectPhase(Compare.selectedPhase === modelData.index ? -1 : modelData.index)
                                    }
                                    ToolTip.visible: chipMouse.containsMouse
                                    ToolTip.text: modelData.label + " — " + modelData.status +
                                                  (modelData.ambiguous ? " (position arbitrary)" : "") + ", " +
                                                  Format.signedNs(modelData.deltaNs) +
                                                  ", similarity " + modelData.similarity.toFixed(2)
                                }
                            }
                        }
                        RowLayout {
                            spacing: AppTheme.spacingSm
                            ToolButton {
                                objectName: "compareAllPhasesButton"
                                text: "All phases"
                                enabled: Compare.selectedPhase >= 0
                                onClicked: Compare.selectPhase(-1)
                            }
                            Text {
                                text: Compare.selectedPhase < 0 ? "Showing the whole run"
                                      : (Compare.selectedPhaseInfo.label || "") + " — " + (Compare.selectedPhaseInfo.status || "") +
                                        ", duration " + (Compare.selectedPhaseInfo.deltaText || "") +
                                        ", critical path " + (Compare.selectedPhaseInfo.criticalDeltaText || "") +
                                        " (baseline: " + (Compare.selectedPhaseInfo.baselineLabel || "—") +
                                        ", candidate: " + (Compare.selectedPhaseInfo.candidateLabel || "—") + ")"
                                color: AppTheme.textMuted
                                font.pixelSize: AppTheme.typeCaption
                                Layout.fillWidth: true
                                elide: Text.ElideRight
                            }
                            ToolButton {
                                objectName: "compareShowPhaseButton"
                                text: "Show phase in Timeline"
                                enabled: Compare.selectedPhase >= 0 && !!Compare.selectedPhaseInfo.hasCandidate
                                onClicked: root.showPhase(Compare.selectedPhase)
                            }
                        }
                    }
                }

                // ── Ranked causal contributors + details ──────────────
                RowLayout {
                    Layout.fillWidth: true
                    spacing: AppTheme.spacingMd
                    visible: Compare.causalAvailable

                    Panel {
                        title: "Ranked causal contributors" +
                               (Compare.selectedPhase >= 0 ? " — " + (Compare.selectedPhaseInfo.label || "") : "")
                        Layout.preferredWidth: Math.max(360, mainColumn.width * 0.45)
                        Layout.preferredHeight: 380
                        ColumnLayout {
                            anchors.fill: parent
                            spacing: AppTheme.spacingXs
                            Text {
                                text: "Δ critical-path time each accounts for (graph-derived); click for evidence"
                                color: AppTheme.textMuted
                                font.pixelSize: AppTheme.typeCaption
                                Layout.fillWidth: true
                                elide: Text.ElideRight
                            }
                            Text {
                                visible: Compare.contributors.length === 0
                                text: "No contributor above the noise floor."
                                color: AppTheme.textMuted
                                font.pixelSize: AppTheme.typeBody
                            }
                            ListView {
                                id: contributorList
                                objectName: "compareContributorList"
                                Layout.fillWidth: true
                                Layout.fillHeight: true
                                clip: true
                                model: Compare.contributors
                                delegate: Rectangle {
                                    width: ListView.view.width
                                    height: 46
                                    radius: AppTheme.radiusSmall
                                    color: root.selectedId === modelData.id ? AppTheme.panelBorder : "transparent"
                                    Accessible.role: Accessible.Button
                                    Accessible.name: modelData.rank + ". " + modelData.label + ", " + modelData.causeLabel +
                                                     ", " + Format.signedNs(modelData.impactNs)
                                    Accessible.onPressAction: root.selectedId = modelData.id
                                    MouseArea { anchors.fill: parent; onClicked: root.selectedId = modelData.id }
                                    ColumnLayout {
                                        anchors.fill: parent
                                        anchors.margins: AppTheme.spacingXs
                                        spacing: 2
                                        RowLayout {
                                            Layout.fillWidth: true
                                            Text { text: modelData.rank + "."; color: AppTheme.textMuted; font.pixelSize: AppTheme.typeCaption }
                                            Text {
                                                text: modelData.label
                                                color: AppTheme.text
                                                font.bold: true
                                                font.pixelSize: AppTheme.typeBody
                                                elide: Text.ElideRight
                                                Layout.fillWidth: true
                                            }
                                            Text {
                                                text: Format.signedNs(modelData.impactNs)
                                                color: AppTheme.changeColor("regressed")
                                                font.pixelSize: AppTheme.typeBody
                                            }
                                        }
                                        RowLayout {
                                            Layout.fillWidth: true
                                            spacing: AppTheme.spacingXs
                                            Rectangle {
                                                implicitWidth: causeText.implicitWidth + AppTheme.spacingSm
                                                implicitHeight: causeText.implicitHeight + 2
                                                radius: height / 2
                                                color: Qt.alpha(root.causeChipColor(modelData.cause), 0.2)
                                                border.color: root.causeChipColor(modelData.cause)
                                                Text {
                                                    id: causeText
                                                    anchors.centerIn: parent
                                                    text: modelData.causeLabel
                                                    color: root.causeChipColor(modelData.cause)
                                                    font.pixelSize: AppTheme.typeCaption
                                                }
                                            }
                                            Text {
                                                text: modelData.roles + (modelData.propagated ? "  ← " + modelData.origin : "")
                                                color: AppTheme.textMuted
                                                font.pixelSize: AppTheme.typeCaption
                                                elide: Text.ElideRight
                                                Layout.fillWidth: true
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    Panel {
                        objectName: "compareDetail"
                        title: root.selectedId >= 0 ? "Details — " + (root.detail.label || "") : "Details"
                        Layout.fillWidth: true
                        Layout.preferredHeight: 380
                        Text {
                            visible: root.selectedId < 0
                            anchors.centerIn: parent
                            text: "Select a contributor to see its measured deltas,\n" +
                                  "graph evidence and where to look."
                            color: AppTheme.textMuted
                            font.pixelSize: AppTheme.typeBody
                            horizontalAlignment: Text.AlignHCenter
                        }
                        Flickable {
                            visible: root.selectedId >= 0
                            anchors.fill: parent
                            contentWidth: width
                            contentHeight: detailColumn.implicitHeight
                            clip: true
                            boundsBehavior: Flickable.StopAtBounds
                            ColumnLayout {
                                id: detailColumn
                                width: parent.width
                                spacing: AppTheme.spacingXs
                                RowLayout {
                                    spacing: AppTheme.spacingSm
                                    ToolButton {
                                        objectName: "compareShowInTimelineButton"
                                        text: "Show in Timeline"
                                        enabled: !!root.detail.hasTimeline
                                        onClicked: root.showInTimeline(root.detail)
                                        ToolTip.visible: hovered
                                        ToolTip.text: "Zoom the Timeline to where it ran in " + (root.detail.phaseLabel || "the run")
                                    }
                                    ToolButton {
                                        objectName: "compareOpenSourceButton"
                                        text: "Open in Source"
                                        enabled: !!root.detail.rawName
                                        onClicked: root.openSource(root.detail)
                                    }
                                    Text {
                                        text: (root.detail.phaseLabel || "") + " · " + (root.detail.baselineRangeText || "")
                                        color: AppTheme.textMuted
                                        font.pixelSize: AppTheme.typeCaption
                                        elide: Text.ElideRight
                                        Layout.fillWidth: true
                                    }
                                }
                                Text {
                                    text: (root.detail.roles || "") + "   " + (root.detail.path || "")
                                    color: AppTheme.textMuted
                                    font.pixelSize: AppTheme.typeCaption
                                    wrapMode: Text.WordWrap
                                    Layout.fillWidth: true
                                }
                                RowLayout {
                                    spacing: AppTheme.spacingSm
                                    Rectangle {
                                        implicitWidth: detailCause.implicitWidth + AppTheme.spacingMd
                                        implicitHeight: detailCause.implicitHeight + AppTheme.spacingXs
                                        radius: height / 2
                                        color: Qt.alpha(root.causeChipColor(root.detail.cause), 0.2)
                                        border.color: root.causeChipColor(root.detail.cause)
                                        Text {
                                            id: detailCause
                                            anchors.centerIn: parent
                                            text: root.detail.causeLabel || "—"
                                            color: root.causeChipColor(root.detail.cause)
                                            font.bold: true
                                            font.pixelSize: AppTheme.typeCaption
                                        }
                                    }
                                    Text {
                                        text: "critical-path impact " + Format.signedNs(root.detail.impactNs || 0) +
                                              ((root.detail.receivedNs || 0) > 0
                                               ? " (incl. " + Format.signedNs(root.detail.receivedNs) + " seen on waits it caused)" : "")
                                        color: AppTheme.text
                                        font.pixelSize: AppTheme.typeCaption
                                    }
                                }
                                Text {
                                    objectName: "compareDetailExplanation"
                                    text: root.detail.explanation || ""
                                    color: AppTheme.text
                                    font.pixelSize: AppTheme.typeBody
                                    wrapMode: Text.WordWrap
                                    Layout.fillWidth: true
                                }
                                RowLayout {
                                    spacing: AppTheme.spacingMd
                                    Repeater {
                                        model: [["", 150], ["before", 72], ["after", 72], ["Δ", 72], ["how", 60]]
                                        delegate: Text {
                                            text: modelData[0]
                                            color: AppTheme.textMuted
                                            font.pixelSize: AppTheme.typeCaption
                                            font.bold: true
                                            Layout.preferredWidth: modelData[1]
                                        }
                                    }
                                }
                                Repeater {
                                    model: root.detail.rows || []
                                    delegate: RowLayout {
                                        spacing: AppTheme.spacingMd
                                        Text { text: modelData.label; color: AppTheme.textMuted; font.pixelSize: AppTheme.typeCaption; Layout.preferredWidth: 150; elide: Text.ElideRight }
                                        Text { text: modelData.before; color: AppTheme.text; font.pixelSize: AppTheme.typeCaption; Layout.preferredWidth: 72 }
                                        Text { text: modelData.after; color: AppTheme.text; font.pixelSize: AppTheme.typeCaption; Layout.preferredWidth: 72 }
                                        Text { text: modelData.delta; color: AppTheme.text; font.pixelSize: AppTheme.typeCaption; Layout.preferredWidth: 72 }
                                        Text { text: modelData.kind; color: AppTheme.textMuted; font.pixelSize: AppTheme.typeCaption; font.italic: true }
                                    }
                                }
                                Repeater {
                                    model: root.detail.evidence || []
                                    delegate: Text {
                                        text: "• " + modelData.text + "  [" + modelData.kind + "]"
                                        color: AppTheme.text
                                        font.pixelSize: AppTheme.typeCaption
                                        wrapMode: Text.WordWrap
                                        Layout.fillWidth: true
                                    }
                                }
                                Text {
                                    visible: (root.detail.chain || []).length > 0
                                    text: {
                                        var c = root.detail.chain || []
                                        var parts = []
                                        for (var i = 0; i < c.length; i++)
                                            parts.push(c[i].label + " (" + c[i].roles + ", " + c[i].causeLabel + ")")
                                        return "Waited for: " + parts.join(" ← ") + "  [graph-derived, heuristic phase offsets]"
                                    }
                                    color: AppTheme.text
                                    font.pixelSize: AppTheme.typeCaption
                                    wrapMode: Text.WordWrap
                                    Layout.fillWidth: true
                                }
                                Text {
                                    visible: (root.detail.alsoCauses || []).length > 0
                                    text: "Also: " + (root.detail.alsoCauses || []).join(", ")
                                    color: AppTheme.textMuted
                                    font.pixelSize: AppTheme.typeCaption
                                    wrapMode: Text.WordWrap
                                    Layout.fillWidth: true
                                }
                                Text {
                                    text: "Match: " + (root.detail.matchKind || "—") +
                                          ((root.detail.matchScore || -1) >= 0 ? " " + root.detail.matchScore.toFixed(2) : "") +
                                          " · confidence " + ((root.detail.confidence || -1) >= 0 ? root.detail.confidence.toFixed(2) : "—") +
                                          "  [heuristic]"
                                    color: AppTheme.textMuted
                                    font.pixelSize: AppTheme.typeCaption
                                }
                                Text {
                                    text: "Source: " + ((root.detail.source || "").length > 0 ? root.detail.source : "not recorded")
                                    color: AppTheme.textMuted
                                    font.pixelSize: AppTheme.typeCaption
                                }
                                Rectangle {
                                    visible: !!root.detail.snippet
                                    Layout.fillWidth: true
                                    Layout.preferredHeight: snippetColumn.implicitHeight + AppTheme.spacingSm
                                    color: AppTheme.background
                                    radius: AppTheme.radiusSmall
                                    Column {
                                        id: snippetColumn
                                        anchors.left: parent.left
                                        anchors.right: parent.right
                                        anchors.top: parent.top
                                        anchors.margins: AppTheme.spacingXs
                                        Repeater {
                                            model: root.detail.snippet ? root.detail.snippet.lines : []
                                            delegate: Text {
                                                text: modelData.line + "  " + modelData.text
                                                color: modelData.hot ? AppTheme.accent : AppTheme.textMuted
                                                font.family: "monospace"
                                                font.pixelSize: AppTheme.typeCaption
                                                font.bold: modelData.hot
                                                elide: Text.ElideRight
                                                width: snippetColumn.width
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // ── Critical path before / after ──────────────────────
                Panel {
                    objectName: "compareCriticalPath"
                    title: "Critical path composition — before vs after" +
                           (Compare.selectedPhase >= 0 ? " — " + (Compare.selectedPhaseInfo.label || "") : "") +
                           " (graph-derived)"
                    visible: Compare.causalAvailable
                    Layout.fillWidth: true
                    Layout.preferredHeight: cpColumn.implicitHeight + AppTheme.spacingMd * 2 + 24
                    ColumnLayout {
                        id: cpColumn
                        anchors.fill: parent
                        spacing: AppTheme.spacingXs
                        Repeater {
                            model: [{"label": "Baseline", "items": Compare.criticalBefore},
                                    {"label": "Candidate", "items": Compare.criticalAfter}]
                            delegate: ColumnLayout {
                                Layout.fillWidth: true
                                spacing: 2
                                property var side: modelData
                                Text { text: side.label; color: AppTheme.textMuted; font.pixelSize: AppTheme.typeCaption }
                                Row {
                                    id: bar
                                    Layout.fillWidth: true
                                    height: 16
                                    Repeater {
                                        model: side.items
                                        delegate: Rectangle {
                                            width: Math.max(1, bar.width * modelData.frac)
                                            height: bar.height
                                            color: root.cpColor(modelData.identity)
                                            border.color: AppTheme.surface
                                            border.width: 1
                                            MouseArea { id: segMouse; anchors.fill: parent; hoverEnabled: true
                                                onClicked: if (modelData.identity >= 0) root.selectedId = modelData.identity }
                                            ToolTip.visible: segMouse.containsMouse
                                            ToolTip.text: modelData.label + " " + modelData.roles + " — " + Format.formatNumber("time_ns", modelData.ns)
                                        }
                                    }
                                }
                                Flow {
                                    Layout.fillWidth: true
                                    spacing: AppTheme.spacingMd
                                    Repeater {
                                        model: side.items
                                        delegate: Row {
                                            spacing: AppTheme.spacingXs
                                            Rectangle { width: 8; height: 8; radius: 2; color: root.cpColor(modelData.identity); anchors.verticalCenter: parent.verticalCenter }
                                            Text {
                                                text: modelData.label + " " + Format.formatNumber("time_ns", modelData.ns)
                                                color: AppTheme.textMuted
                                                font.pixelSize: AppTheme.typeCaption
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // ── Propagated waits / off-path changes / new & removed ─
                RowLayout {
                    Layout.fillWidth: true
                    spacing: AppTheme.spacingMd
                    visible: Compare.causalAvailable

                    Repeater {
                        model: [
                            {"title": "Waits caused upstream", "items": Compare.propagated},
                            {"title": "Off the critical path", "items": Compare.offCriticalPath},
                            {"title": "New / removed", "items": Compare.newWork.concat(Compare.removedWork)},
                            {"title": "Improvements", "items": Compare.causalImprovements}
                        ]
                        delegate: Panel {
                            title: modelData.title
                            Layout.fillWidth: true
                            Layout.preferredHeight: 22 * Math.max(1, Math.min(8, modelData.items.length)) + AppTheme.spacingMd * 2 + 30
                            property var items: modelData.items
                            ColumnLayout {
                                anchors.fill: parent
                                spacing: 1
                                Text {
                                    visible: items.length === 0
                                    text: "none"
                                    color: AppTheme.textMuted
                                    font.pixelSize: AppTheme.typeCaption
                                }
                                Repeater {
                                    model: items.slice(0, 8)
                                    delegate: RowLayout {
                                        Layout.fillWidth: true
                                        Text {
                                            text: modelData.label + (modelData.propagated ? " ← " + modelData.origin : "")
                                            color: AppTheme.text
                                            font.pixelSize: AppTheme.typeCaption
                                            elide: Text.ElideRight
                                            Layout.fillWidth: true
                                            MouseArea { anchors.fill: parent; onClicked: root.selectedId = modelData.id }
                                        }
                                        Text {
                                            text: Format.signedNs(Math.abs(modelData.ownDeltaNs) >= Math.abs(modelData.criticalDeltaNs)
                                                                  ? modelData.ownDeltaNs : modelData.criticalDeltaNs)
                                            color: AppTheme.changeColor(modelData.ownDeltaNs + modelData.criticalDeltaNs >= 0 ? "regressed" : "improved")
                                            font.pixelSize: AppTheme.typeCaption
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                Panel {
                    title: "Conclusions not available"
                    visible: Compare.unavailableConclusions.length > 0
                    Layout.fillWidth: true
                    Layout.preferredHeight: 20 * Compare.unavailableConclusions.length + AppTheme.spacingMd * 2 + 30
                    ColumnLayout {
                        anchors.fill: parent
                        spacing: 1
                        Repeater {
                            model: Compare.unavailableConclusions
                            delegate: Text {
                                text: "• " + modelData.conclusion + ": " + modelData.reason
                                color: AppTheme.textMuted
                                font.pixelSize: AppTheme.typeCaption
                                elide: Text.ElideRight
                                Layout.fillWidth: true
                            }
                        }
                    }
                }

                // ── (category, name) aggregate view ───────────────────
                Panel {
                    title: "Legend & noise floor"
                    Layout.fillWidth: true
                    Layout.preferredHeight: legendColumn.implicitHeight + AppTheme.spacingMd * 2 + 30
                    ColumnLayout {
                        id: legendColumn
                        anchors.fill: parent
                        spacing: AppTheme.spacingSm

                        RowLayout {
                            spacing: AppTheme.spacingSm
                            Repeater {
                                model: Compare.statusLegend
                                delegate: ChangeBadge { status: modelData.status; text: modelData.label }
                            }
                        }
                        Text {
                            Layout.fillWidth: true
                            text: Compare.noiseFloor.note
                            color: AppTheme.textMuted
                            font.pixelSize: AppTheme.typeCaption
                            font.italic: true
                            wrapMode: Text.WordWrap
                        }
                        RowLayout {
                            spacing: AppTheme.spacingSm
                            Text {
                                text: "Thresholds: min Δ%"
                                color: AppTheme.textMuted
                                font.pixelSize: AppTheme.typeCaption
                            }
                            TextField {
                                id: minPctField
                                objectName: "compareMinPctField"
                                text: Compare.noiseFloor.pct.toFixed(0)
                                implicitWidth: 50
                                validator: DoubleValidator { bottom: 0 }
                                font.pixelSize: AppTheme.typeCaption
                                color: AppTheme.text
                            }
                            Text {
                                text: "min Δ time (ns)"
                                color: AppTheme.textMuted
                                font.pixelSize: AppTheme.typeCaption
                            }
                            TextField {
                                id: minNsField
                                objectName: "compareMinNsField"
                                text: Compare.noiseFloor.ns.toFixed(0)
                                implicitWidth: 90
                                validator: DoubleValidator { bottom: 0 }
                                font.pixelSize: AppTheme.typeCaption
                                color: AppTheme.text
                            }
                            ToolButton {
                                objectName: "compareApplyThresholdsButton"
                                text: "Apply"
                                onClicked: Compare.setChangeThresholds(
                                    parseFloat(minNsField.text) || 0, parseFloat(minPctField.text) || 0)
                            }
                        }
                    }
                }

                RowLayout {
                    Layout.fillWidth: true
                    spacing: AppTheme.spacingMd

                    Panel {
                        title: "Activity bucket deltas"
                        Layout.fillWidth: true
                        Layout.preferredHeight: 26 * Math.max(1, Compare.bucketDeltas.length) + AppTheme.spacingMd * 2 + 24
                        ColumnLayout {
                            anchors.fill: parent
                            spacing: 2
                            Repeater {
                                model: Compare.bucketDeltas
                                delegate: RowLayout {
                                    Layout.fillWidth: true
                                    Rectangle { width: 8; height: 8; radius: 4; color: AppTheme.bucketColor(modelData.bucket) }
                                    Text {
                                        text: modelData.bucket
                                        color: AppTheme.text
                                        font.pixelSize: AppTheme.typeCaption
                                        Layout.preferredWidth: 130
                                    }
                                    Text {
                                        text: Format.signedNs(modelData.deltaNs)
                                        color: AppTheme.textMuted
                                        font.pixelSize: AppTheme.typeCaption
                                        Layout.preferredWidth: 90
                                    }
                                    ChangeBadge { status: modelData.status }
                                    Item { Layout.fillWidth: true }
                                }
                            }
                        }
                    }

                    Panel {
                        title: "Execution coverage (own wall time)"
                        Layout.preferredWidth: 320
                        Layout.preferredHeight: 90
                        ColumnLayout {
                            anchors.fill: parent
                            spacing: AppTheme.spacingXs
                            Text { text: "Baseline"; color: AppTheme.textMuted; font.pixelSize: AppTheme.typeCaption }
                            Row {
                                Layout.fillWidth: true
                                height: 14
                                Repeater {
                                    model: Compare.baselineCoverage
                                    delegate: Rectangle {
                                        width: 4; height: 14
                                        color: modelData > 0 ? AppTheme.accent : "transparent"
                                        opacity: 0.3 + 0.7 * modelData
                                    }
                                }
                            }
                            Text { text: "Candidate"; color: AppTheme.textMuted; font.pixelSize: AppTheme.typeCaption }
                            Row {
                                Layout.fillWidth: true
                                height: 14
                                Repeater {
                                    model: Compare.comparisonCoverage
                                    delegate: Rectangle {
                                        width: 4; height: 14
                                        color: modelData > 0 ? AppTheme.warningColor : "transparent"
                                        opacity: 0.3 + 0.7 * modelData
                                    }
                                }
                            }
                        }
                    }
                }

                RowLayout {
                    Layout.fillWidth: true
                    spacing: AppTheme.spacingMd

                    Panel {
                        title: "Aggregate by (category, name): largest regressions"
                        Layout.fillWidth: true
                        Layout.preferredHeight: 22 * Math.max(1, Compare.topRegressions.length) + AppTheme.spacingMd * 2 + 24
                        ColumnLayout {
                            anchors.fill: parent
                            spacing: 1
                            Repeater {
                                model: Compare.topRegressions
                                delegate: RowLayout {
                                    Layout.fillWidth: true
                                    Text {
                                        text: modelData.name
                                        color: AppTheme.text
                                        font.pixelSize: AppTheme.typeCaption
                                        Layout.fillWidth: true
                                        elide: Text.ElideRight
                                    }
                                    Text {
                                        text: Format.signedNs(modelData.deltaNs)
                                        color: AppTheme.changeColor("regressed")
                                        font.pixelSize: AppTheme.typeCaption
                                    }
                                }
                            }
                        }
                    }
                    Panel {
                        title: "Aggregate by (category, name): largest improvements"
                        Layout.fillWidth: true
                        Layout.preferredHeight: 22 * Math.max(1, Compare.topImprovements.length) + AppTheme.spacingMd * 2 + 24
                        ColumnLayout {
                            anchors.fill: parent
                            spacing: 1
                            Repeater {
                                model: Compare.topImprovements
                                delegate: RowLayout {
                                    Layout.fillWidth: true
                                    Text {
                                        text: modelData.name
                                        color: AppTheme.text
                                        font.pixelSize: AppTheme.typeCaption
                                        Layout.fillWidth: true
                                        elide: Text.ElideRight
                                    }
                                    Text {
                                        text: Format.signedNs(modelData.deltaNs)
                                        color: AppTheme.changeColor("improved")
                                        font.pixelSize: AppTheme.typeCaption
                                    }
                                }
                            }
                        }
                    }
                }

                DataTable {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 420
                    title: "Aggregate by (category, name) — compatibility view"
                    table: Compare.table
                }
            }
        }
    }
}
