import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import Hprofiler 1.0
import "."

// Dismissible first-use explainer for the Timeline tab -- shown once
// (gated by Workspace.timelineOverlayDismissed, which starts false and
// is permanently set true on dismiss, persisted via QSettings so it
// never reappears after the first dismissal across relaunches too).
// Deliberately NOT a modal/blocking dialog: a semi-transparent panel
// anchored to a corner, the Timeline underneath stays fully visible and
// interactive the whole time -- explaining a screen shouldn't require
// hiding it first.
Item {
    id: root
    anchors.fill: parent
    z: 500
    visible: !Workspace.timelineOverlayDismissed

    Rectangle {
        objectName: "timelineFirstUseCard"
        anchors.top: parent.top
        anchors.right: parent.right
        anchors.margins: AppTheme.spacingLg
        width: 320
        radius: AppTheme.radiusPanel
        color: AppTheme.surface
        border.color: AppTheme.panelBorderFocus
        border.width: 1
        height: content.implicitHeight + AppTheme.spacingLg * 2

        ColumnLayout {
            id: content
            anchors.fill: parent
            anchors.margins: AppTheme.spacingLg
            spacing: AppTheme.spacingSm

            RowLayout {
                Layout.fillWidth: true
                Text {
                    text: "Exploring the Timeline"
                    color: AppTheme.text
                    font.bold: true
                    font.pixelSize: AppTheme.typeBody
                    Layout.fillWidth: true
                }
                ToolButton {
                    objectName: "timelineFirstUseDismiss"
                    text: "✕"
                    implicitWidth: 24
                    implicitHeight: 24
                    onClicked: Workspace.dismissTimelineOverlay()
                    ToolTip.visible: hovered
                    ToolTip.text: "Dismiss this tip (won't show again)"
                    Accessible.name: "Dismiss tip"
                    Accessible.description: "Dismisses this Timeline introduction and hides it permanently"
                }
            }

            Text {
                Layout.fillWidth: true
                wrapMode: Text.WordWrap
                color: AppTheme.textMuted
                font.pixelSize: AppTheme.typeLabel
                text: "• Scroll to zoom, drag to pan\n" +
                      "• Shift+drag to select a time range\n" +
                      "• Click a span to select it; double-click to zoom to it\n" +
                      "• Use Filter/Group to narrow or reorganize lanes\n" +
                      "• Press 0 or the reset button to return to the full view"
            }

            Button {
                objectName: "timelineFirstUseGotIt"
                Layout.alignment: Qt.AlignRight
                text: "Got it"
                flat: true
                onClicked: Workspace.dismissTimelineOverlay()
            }
        }
    }
}
