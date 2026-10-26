import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import Hprofiler 1.0

// Exact values for the selected event or dependency edge (TimelineModel.
// eventDetails / edgeDetails): timing, place, source / stack / kernel /
// disassembly, critical-path membership, dependencies with their evidence,
// the comparison counterpart. Non-measured values carry their kind
// ("derived", "estimated", "unavailable" + reason) -- never a silent blank.
Rectangle {
    id: panel
    objectName: "timelineDetailsPanel"
    color: AppTheme.surface
    border.color: AppTheme.panelBorder
    border.width: 1
    radius: AppTheme.radiusPanel

    // set by TimelineScreen
    property var details: ({})
    property var edge: ({})
    signal closeRequested()
    signal jumpTo(int laneIndex, real spanIdx, real startNs)
    signal selectEdge(int edgeId)
    signal openSource(string category, string name)

    readonly property bool showingEdge: edge && edge.id !== undefined

    function kindColor(kind) {
        if (kind === "derived") return AppTheme.accent
        if (kind === "estimated") return AppTheme.warningColor
        if (kind === "unavailable") return AppTheme.textMuted
        return AppTheme.text
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: AppTheme.spacingSm
        spacing: AppTheme.spacingXs

        RowLayout {
            Layout.fillWidth: true
            Text {
                objectName: "timelineDetailsTitle"
                Layout.fillWidth: true
                Layout.minimumWidth: 0
                text: panel.showingEdge ? ("Dependency: " + panel.edge.kind)
                                        : (panel.details.title || "")
                color: AppTheme.text
                font.bold: true
                font.pixelSize: AppTheme.typeLabel
                elide: Text.ElideMiddle
            }
            ToolButton {
                objectName: "timelineDetailsClose"
                text: "✕"
                implicitWidth: AppTheme.iconButtonWidth
                implicitHeight: AppTheme.buttonHeight
                onClicked: panel.closeRequested()
                ToolTip.visible: hovered
                ToolTip.text: "Close details (Esc)"
                Accessible.name: "Close details"
            }
        }

        ScrollView {
            id: scroll
            Layout.fillWidth: true
            Layout.fillHeight: true
            clip: true
            contentWidth: availableWidth

            ColumnLayout {
                width: scroll.availableWidth
                spacing: AppTheme.spacingXs

                // ── an edge ───────────────────────────────────────────
                Repeater {
                    model: panel.showingEdge ? [
                        {"l": "Family", "v": panel.edge.familyLabel},
                        {"l": "Runtime", "v": panel.edge.runtime},
                        {"l": "Evidence tier", "v": panel.edge.tier + "  (an evidence class, not a probability)"},
                        {"l": "Provenance", "v": panel.edge.provenance + " — " + panel.edge.provenanceText},
                        {"l": "Direct or inferred", "v": panel.edge["class"]},
                        {"l": "Why", "v": panel.edge.reason},
                        {"l": "From", "v": panel.edge.source.name + " (" + panel.edge.source.event + ") @ " + panel.edge.source.place + "  " + panel.edge.source.at},
                        {"l": "To", "v": panel.edge.target.name + " (" + panel.edge.target.event + ") @ " + panel.edge.target.place + "  " + panel.edge.target.at},
                        {"l": "Source → target", "v": Format.duration(panel.edge.latencyNs)},
                        {"l": "On the critical path", "v": panel.edge.onPath ? "yes" : "no"}
                    ] : []
                    delegate: ColumnLayout {
                        Layout.fillWidth: true
                        spacing: 0
                        Text { text: modelData.l; color: AppTheme.textMuted; font.pixelSize: AppTheme.typeCaption }
                        Text {
                            Layout.fillWidth: true
                            text: modelData.v
                            color: AppTheme.text
                            font.pixelSize: AppTheme.typeLabel
                            wrapMode: Text.Wrap
                        }
                    }
                }
                RowLayout {
                    visible: panel.showingEdge
                    Button {
                        objectName: "timelineEdgeGotoSource"
                        text: "Go to source"
                        onClicked: panel.jumpTo(panel.edge.source.laneIndex, panel.edge.source.spanIdx,
                                                panel.edge.source.atNs)
                    }
                    Button {
                        objectName: "timelineEdgeGotoTarget"
                        text: "Go to target"
                        onClicked: panel.jumpTo(panel.edge.target.laneIndex, panel.edge.target.spanIdx,
                                                panel.edge.target.atNs)
                    }
                }

                // ── an event ──────────────────────────────────────────
                Repeater {
                    model: panel.showingEdge ? [] : (panel.details.sections || [])
                    delegate: ColumnLayout {
                        id: section
                        Layout.fillWidth: true
                        spacing: 1
                        property var sec: modelData
                        Text {
                            text: section.sec.title
                            color: AppTheme.accent
                            font.bold: true
                            font.pixelSize: AppTheme.typeCaption
                            topPadding: AppTheme.spacingXs
                        }
                        Repeater {
                            model: section.sec.fields
                            delegate: RowLayout {
                                Layout.fillWidth: true
                                spacing: AppTheme.spacingXs
                                Text {
                                    Layout.preferredWidth: 110
                                    Layout.alignment: Qt.AlignTop
                                    text: modelData.label
                                    color: AppTheme.textMuted
                                    font.pixelSize: AppTheme.typeCaption
                                    elide: Text.ElideRight
                                }
                                Text {
                                    Layout.fillWidth: true
                                    text: modelData.kind === "unavailable" ? ("— " + modelData.reason)
                                          : modelData.value + (modelData.kind !== "measured" && modelData.kind !== undefined
                                                               ? "  [" + modelData.kind + "]" : "")
                                    color: panel.kindColor(modelData.kind)
                                    font.pixelSize: AppTheme.typeCaption
                                    font.underline: modelData.action !== undefined
                                    wrapMode: Text.Wrap
                                    ToolTip.visible: tipArea.containsMouse && modelData.reason !== undefined
                                                     && modelData.reason.length > 0 && modelData.kind !== "unavailable"
                                    ToolTip.text: modelData.reason || ""
                                    MouseArea {
                                        id: tipArea
                                        anchors.fill: parent
                                        hoverEnabled: true
                                        cursorShape: modelData.action !== undefined ? Qt.PointingHandCursor : Qt.ArrowCursor
                                        onClicked: {
                                            var a = modelData.action
                                            if (a && a.kind === "source") panel.openSource(a.category, a.name)
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                Text {
                    visible: !panel.showingEdge && (panel.details.dependencies || []).length > 0
                    text: "Dependencies (" + (panel.details.dependencies || []).length + ")"
                    color: AppTheme.accent
                    font.bold: true
                    font.pixelSize: AppTheme.typeCaption
                    topPadding: AppTheme.spacingXs
                }
                Repeater {
                    objectName: "timelineDetailsDependencies"
                    model: panel.showingEdge ? [] : (panel.details.dependencies || [])
                    delegate: Rectangle {
                        Layout.fillWidth: true
                        implicitHeight: depCol.implicitHeight + 6
                        color: depArea.containsMouse ? AppTheme.control : "transparent"
                        radius: AppTheme.radiusSmall
                        ColumnLayout {
                            id: depCol
                            x: 3; y: 3
                            width: parent.width - 6
                            spacing: 0
                            Text {
                                Layout.fillWidth: true
                                text: (modelData.direction === "in" ? "← " : "→ ") + modelData.kind
                                      + (modelData.onPath ? "  [critical path]" : "") + "   "
                                      + modelData.partner + " @ " + modelData.partnerPlace
                                color: AppTheme.familyColor(modelData.family, AppTheme.dark)
                                font.pixelSize: AppTheme.typeCaption
                                elide: Text.ElideRight
                            }
                            Text {
                                Layout.fillWidth: true
                                text: modelData.tier + " · " + modelData.provenance + " · " + modelData["class"]
                                      + " · " + Format.duration(modelData.latencyNs)
                                color: AppTheme.textMuted
                                font.pixelSize: AppTheme.typeCaption
                                elide: Text.ElideRight
                            }
                        }
                        MouseArea {
                            id: depArea
                            anchors.fill: parent
                            hoverEnabled: true
                            cursorShape: Qt.PointingHandCursor
                            onClicked: panel.selectEdge(modelData.edge)
                            onDoubleClicked: panel.jumpTo(modelData.partnerLane, modelData.partnerSpanIdx,
                                                          modelData.partnerStartNs)
                        }
                        ToolTip.visible: depArea.containsMouse
                        ToolTip.text: modelData.reason + "\nclick: show the edge · double-click: go to the partner"
                    }
                }
            }
        }
    }
}
