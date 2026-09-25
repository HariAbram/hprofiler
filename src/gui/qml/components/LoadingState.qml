import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import Hprofiler 1.0

// Shared "actively working, not just empty" indicator. Nothing like this
// existed anywhere in the GUI before this component (confirmed: zero
// spinner/busy-indicator usage across every screen) -- ships as ready-
// to-use infrastructure per the visual-consistency audit's explicit ask,
// not a consolidation of an existing pattern. Deliberately plain (a
// stock BusyIndicator, no custom skinning) to match this project's
// "avoid excessive gradients/shadows" preference.
ColumnLayout {
    id: root
    property string message: "Loading…"

    anchors.centerIn: parent
    spacing: AppTheme.spacingSm

    BusyIndicator {
        Layout.alignment: Qt.AlignHCenter
        running: true
        implicitWidth: 28
        implicitHeight: 28
    }
    Text {
        Layout.alignment: Qt.AlignHCenter
        text: root.message
        color: AppTheme.textMuted
        font.pixelSize: AppTheme.typeBody
    }
}
