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
    radius: AppTheme.radiusPanel

    Column {
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.verticalCenter: parent.verticalCenter
        anchors.margins: AppTheme.spacingLg
        spacing: AppTheme.spacingXs

        Text {
            text: label
            color: AppTheme.textMuted
            font.pixelSize: AppTheme.typeLabel
            font.letterSpacing: 1
        }
        Text {
            text: value
            color: valueColor
            font.pixelSize: AppTheme.typeValue
            font.bold: true
            elide: Text.ElideRight
            width: parent.width
        }
    }
}
