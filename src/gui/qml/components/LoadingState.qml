import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import Hprofiler 1.0
import "."

// Shared "actively working, not just empty" indicator. Deliberately
// plain (a stock BusyIndicator, no custom skinning) to match this
// project's "avoid excessive gradients/shadows" preference.
//
// `showProgress`/`progressPct`/`stageText`/`elapsedText` (added for the
// async-loading overlay -- see loader.py's stage/progress signals and
// controller.py's AppController): when showProgress is true, a
// ProgressBar replaces the plain spinner (determinate if progressPct
// is given a real value, indeterminate otherwise) and stageText/
// elapsedText render as small secondary lines below the main message --
// satisfies "show the current operation, processing stage, elapsed
// time, and measurable progress" without every caller having to
// reassemble the same three-line layout by hand.
ColumnLayout {
    id: root
    property string message: "Loading…"
    property bool showProgress: false
    property real progressPct: -1   // -1 = indeterminate
    property string stageText: ""
    property string elapsedText: ""

    anchors.centerIn: parent
    spacing: AppTheme.spacingSm

    BusyIndicator {
        Layout.alignment: Qt.AlignHCenter
        visible: !root.showProgress
        running: visible
        implicitWidth: 28
        implicitHeight: 28
    }

    ProgressBar {
        Layout.alignment: Qt.AlignHCenter
        Layout.preferredWidth: 220
        visible: root.showProgress
        indeterminate: root.progressPct < 0
        pct: root.progressPct
    }

    Text {
        Layout.alignment: Qt.AlignHCenter
        text: root.message
        color: AppTheme.textMuted
        font.pixelSize: AppTheme.typeBody
    }
    Text {
        Layout.alignment: Qt.AlignHCenter
        visible: root.stageText.length > 0
        text: root.stageText
        color: AppTheme.textMuted
        font.pixelSize: AppTheme.typeLabel
    }
    Text {
        Layout.alignment: Qt.AlignHCenter
        visible: root.elapsedText.length > 0
        text: root.elapsedText
        color: AppTheme.textMuted
        font.pixelSize: AppTheme.typeCaption
    }
}
