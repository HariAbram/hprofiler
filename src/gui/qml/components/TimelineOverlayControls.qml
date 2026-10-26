import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import Hprofiler 1.0

// The Timeline's analysis overlays: the critical path (highlight, focus on
// its lanes, its switches, the span-level score track, inferred waiting)
// and dependency edges by family / evidence tier / direct-or-inferred.
// Never every edge at once by default: the hovered or selected event's own
// dependencies are always drawn; whole families only when ticked here.
Item {
    id: root
    implicitWidth: row.implicitWidth
    implicitHeight: AppTheme.buttonHeight

    signal transitionRequested(int direction)

    // A check box whose label wraps to the width its layout gives it: the
    // default label keeps its one-line width and runs past a popup's edge
    // (the system font, which controls use, can be larger than the theme's).
    component WrapCheckBox: CheckBox {
        id: control
        Layout.fillWidth: true
        Layout.minimumWidth: 0
        contentItem: Text {
            text: control.text
            font: control.font
            color: control.enabled ? control.palette.windowText : control.palette.disabled.windowText
            wrapMode: Text.Wrap
            verticalAlignment: Text.AlignVCenter
            leftPadding: control.indicator && !control.mirrored ? control.indicator.width + control.spacing : 0
            rightPadding: control.indicator && control.mirrored ? control.indicator.width + control.spacing : 0
        }
    }

    readonly property var ov: TimelineModel.overlay
    readonly property bool ready: TimelineModel.analysisState === "ready"

    function setKey(key, value) {
        var spec = {}
        spec[key] = value
        TimelineModel.setOverlay(spec)
    }
    function toggleIn(key, value) {
        var list = (root.ov[key] || []).slice()
        var i = list.indexOf(value)
        if (i >= 0) list.splice(i, 1); else list.push(value)
        root.setKey(key, list)
    }

    RowLayout {
        id: row
        anchors.fill: parent
        spacing: AppTheme.spacingXs

        ToolButton {
            id: cpButton
            objectName: "timelineCriticalPathButton"
            text: "Critical path" + (root.ov.criticalPath ? " ●" : "")
            implicitHeight: AppTheme.buttonHeight
            onClicked: cpPopup.visible ? cpPopup.close() : cpPopup.open()
            ToolTip.visible: hovered
            ToolTip.text: "Highlight the observed critical path, focus on its lanes, step through its switches (C)"
        }
        ToolButton {
            id: depButton
            objectName: "timelineDependenciesButton"
            text: "Dependencies" + ((root.ov.families || []).length ? " (" + root.ov.families.length + ")" : "")
            implicitHeight: AppTheme.buttonHeight
            onClicked: depPopup.visible ? depPopup.close() : depPopup.open()
            ToolTip.visible: hovered
            ToolTip.text: "Overlay dependency edges by kind and evidence; a selected event's own edges are always shown"
        }
        Text {
            objectName: "timelineAnalysisStatus"
            Layout.maximumWidth: 260
            elide: Text.ElideRight
            visible: TimelineModel.analysisState !== "idle"
            text: TimelineModel.analysisState === "running" ? "⏳ " + TimelineModel.analysisMessage
                  : TimelineModel.analysisState === "failed" ? "⚠ " + TimelineModel.analysisMessage : ""
            color: TimelineModel.analysisState === "failed" ? AppTheme.warningColor : AppTheme.textMuted
            font.pixelSize: AppTheme.typeCaption
        }
    }

    Popup {
        id: cpPopup
        objectName: "timelineCriticalPathPopup"
        y: cpButton.height + 2
        width: 360
        modal: false
        focus: true
        closePolicy: Popup.CloseOnEscape | Popup.CloseOnPressOutsideParent
        background: Rectangle {
            color: AppTheme.surface
            border.color: AppTheme.panelBorder
            radius: AppTheme.radiusPanel
        }
        ColumnLayout {
            width: cpPopup.availableWidth
            spacing: AppTheme.spacingXs
            Text {
                Layout.fillWidth: true
                text: TimelineModel.analysisState === "ready" ? "Observed critical path — each instant counted once"
                      : TimelineModel.analysisState === "running" ? TimelineModel.analysisMessage
                      : "Not computed yet (runs in the background)"
                color: AppTheme.textMuted
                font.pixelSize: AppTheme.typeCaption
                wrapMode: Text.Wrap
            }
            Button {
                objectName: "timelineComputeAnalysisButton"
                Layout.maximumWidth: cpPopup.availableWidth
                visible: TimelineModel.analysisState === "idle" || TimelineModel.analysisState === "failed"
                text: "Compute dependencies and critical path"
                onClicked: TimelineModel.startAnalysis()
            }
            WrapCheckBox {
                objectName: "timelineCpHighlight"
                text: "Highlight the path (band + hops), keep everything else"
                checked: !!root.ov.criticalPath
                onToggled: root.setKey("criticalPath", checked)
            }
            WrapCheckBox {
                objectName: "timelineCpFocus"
                text: "Focus: only lanes the path touches"
                checked: !!root.ov.focusPath
                enabled: root.ready
                onToggled: root.setKey("focusPath", checked)
            }
            WrapCheckBox {
                objectName: "timelineCpScoreTrack"
                text: "Span-level search score track (what it counts twice)"
                checked: !!root.ov.scoreTrack
                enabled: root.ready
                onToggled: root.setKey("scoreTrack", checked)
            }
            WrapCheckBox {
                objectName: "timelineInferredWaits"
                text: "Inferred waiting inside gated calls"
                checked: !!root.ov.inferredWaits
                onToggled: root.setKey("inferredWaits", checked)
                ToolTip.visible: hovered
                ToolTip.text: "Dotted veil from a blocking call's start to its last dependency's source event (from the dependency graph)"
            }
            Flow {
                Layout.fillWidth: true
                enabled: root.ready
                spacing: AppTheme.spacingXs
                Button {
                    objectName: "timelinePrevTransition"
                    text: "◀ previous switch"
                    onClicked: root.transitionRequested(-1)
                }
                Button {
                    objectName: "timelineNextTransition"
                    text: "next switch ▶"
                    onClicked: root.transitionRequested(1)
                }
            }
            Text {
                Layout.fillWidth: true
                text: "A switch is where the path moves to another rank, thread, runtime, device or stream (N / Shift+N)."
                color: AppTheme.textMuted
                font.pixelSize: AppTheme.typeCaption
                wrapMode: Text.Wrap
            }
        }
    }

    Popup {
        id: depPopup
        objectName: "timelineDependenciesPopup"
        y: depButton.height + 2
        x: depButton.x
        width: 360
        modal: false
        focus: true
        closePolicy: Popup.CloseOnEscape | Popup.CloseOnPressOutsideParent
        background: Rectangle {
            color: AppTheme.surface
            border.color: AppTheme.panelBorder
            radius: AppTheme.radiusPanel
        }
        ColumnLayout {
            width: depPopup.availableWidth
            spacing: 2
            Text {
                Layout.fillWidth: true
                text: "The hovered or selected event's own dependencies are always drawn. Tick a family to draw all of its edges in view."
                color: AppTheme.textMuted
                font.pixelSize: AppTheme.typeCaption
                wrapMode: Text.Wrap
            }
            Repeater {
                model: TimelineModel.edgeFamilies
                delegate: RowLayout {
                    visible: modelData.key !== "critical"
                    Layout.fillWidth: true
                    spacing: AppTheme.spacingXs
                    Rectangle {
                        Layout.preferredWidth: 14
                        Layout.preferredHeight: 3
                        color: AppTheme.familyColor(modelData.key, AppTheme.dark)
                    }
                    WrapCheckBox {
                        objectName: "timelineFamily_" + modelData.key
                        text: modelData.label
                        checked: (root.ov.families || []).indexOf(modelData.key) >= 0
                        onToggled: root.toggleIn("families", modelData.key)
                    }
                }
            }
            Text {
                Layout.fillWidth: true
                text: "Evidence tier (width / opacity)"
                color: AppTheme.text
                font.pixelSize: AppTheme.typeCaption
                font.bold: true
                wrapMode: Text.Wrap
            }
            Flow {
                Layout.fillWidth: true
                Repeater {
                    model: ["certain", "high", "medium", "low"]
                    delegate: CheckBox {
                        objectName: "timelineTier_" + modelData
                        text: modelData
                        checked: (root.ov.tiers || []).indexOf(modelData) >= 0
                        onToggled: root.toggleIn("tiers", modelData)
                    }
                }
            }
            Text {
                Layout.fillWidth: true
                text: "Evidence class (line style)"
                color: AppTheme.text
                font.pixelSize: AppTheme.typeCaption
                font.bold: true
                wrapMode: Text.Wrap
            }
            Flow {
                Layout.fillWidth: true
                CheckBox {
                    objectName: "timelineClass_direct"
                    text: "direct (solid)"
                    checked: (root.ov.classes || []).indexOf("direct") >= 0
                    onToggled: root.toggleIn("classes", "direct")
                }
                CheckBox {
                    objectName: "timelineClass_inferred"
                    text: "inferred (dashed)"
                    checked: (root.ov.classes || []).indexOf("inferred") >= 0
                    onToggled: root.toggleIn("classes", "inferred")
                }
            }
            Text {
                Layout.fillWidth: true
                visible: TimelineModel.edgesCapped
                text: "More than 4000 edges in view: showing the critical-path and strongest ones — zoom in or narrow the filters."
                color: AppTheme.warningColor
                font.pixelSize: AppTheme.typeCaption
                wrapMode: Text.Wrap
            }
            Text {
                Layout.fillWidth: true
                text: "Tiers are evidence classes, not probabilities. Click an edge for its provenance and reason."
                color: AppTheme.textMuted
                font.pixelSize: AppTheme.typeCaption
                wrapMode: Text.Wrap
            }
        }
    }
}
