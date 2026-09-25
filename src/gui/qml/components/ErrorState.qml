import QtQuick
import QtQuick.Layouts
import Hprofiler 1.0

// Shared error-message presentation. Like LoadingState, nothing
// equivalent existed anywhere in the GUI before this (QML-load failures
// happen before the engine exists at all, out of reach of a QML
// component) -- ships as ready-to-use infrastructure for any future
// screen-level error condition, not a consolidation of an existing
// pattern. Deliberately uses errorColor (not a red background fill or
// icon) to stay consistent with "avoid excessive... saturated colors".
ColumnLayout {
    id: root
    property string message: ""
    property string detail: ""

    anchors.centerIn: parent
    spacing: AppTheme.spacingXs

    Text {
        Layout.alignment: Qt.AlignHCenter
        text: root.message
        color: AppTheme.errorColor
        font.pixelSize: AppTheme.typeBody
        font.bold: true
        horizontalAlignment: Text.AlignHCenter
        wrapMode: Text.WordWrap
    }
    Text {
        Layout.alignment: Qt.AlignHCenter
        visible: root.detail.length > 0
        text: root.detail
        color: AppTheme.textMuted
        font.pixelSize: AppTheme.typeLabel
        horizontalAlignment: Text.AlignHCenter
        wrapMode: Text.WordWrap
    }
}
