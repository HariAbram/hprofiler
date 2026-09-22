import QtQuick
import Hprofiler 1.0

// Small headline stat card (DIAGNOSIS / WALL TIME / GPU ACTIVE / ... on
// the Overview screen) -- label on top, big value below, value color
// configurable per-card (diagnosis/wait cards recolor by severity).
Rectangle {
    property string label: ""
    property string value: ""
    property string valueColor: AppTheme.text

    color: AppTheme.surface
    border.color: AppTheme.panelBorder
    border.width: 1
    radius: 6

    Column {
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.verticalCenter: parent.verticalCenter
        anchors.margins: 12
        spacing: 4

        Text {
            text: label
            color: AppTheme.textMuted
            font.pixelSize: 11
            font.letterSpacing: 1
        }
        Text {
            text: value
            color: valueColor
            font.pixelSize: 20
            font.bold: true
            elide: Text.ElideRight
            width: parent.width
        }
    }
}
