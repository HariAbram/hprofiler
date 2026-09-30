import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import Hprofiler 1.0

// Shared error-message presentation. Deliberately uses errorColor (not a
// red background fill or icon) to stay consistent with "avoid
// excessive... saturated colors".
//
// `tracebackText`/`stage`/`file` (added alongside errors.py's
// HprofilerLoadError/classify_load_exception) back an expandable
// "technical details" section -- collapsed by default (concise-by-
// default matches HprofilerLoadError's own message/detail split), with
// "copy diagnostics" (reuses Inspector.copyToClipboard -- the same
// clipboard path the Inspector panel's own copy action already uses,
// not a second clipboard mechanism) and "open log" (calls
// App.openLogFile()) actions. Both action buttons are hidden when
// nothing backs them (no traceback AND the caller never wired an
// onOpenLog handler) so a plain message+detail usage (unchanged from
// before) shows nothing extra.
ColumnLayout {
    id: root
    property string message: ""
    property string detail: ""
    property string tracebackText: ""
    property string stage: ""
    property string file: ""
    property bool showOpenLog: true

    signal openLogRequested()

    anchors.centerIn: parent
    width: Math.min(560, parent ? parent.width - AppTheme.spacingXl * 2 : 560)
    spacing: AppTheme.spacingXs

    Text {
        Layout.alignment: Qt.AlignHCenter
        Layout.fillWidth: true
        text: root.message
        color: AppTheme.errorColor
        font.pixelSize: AppTheme.typeBody
        font.bold: true
        horizontalAlignment: Text.AlignHCenter
        wrapMode: Text.WordWrap
    }
    Text {
        Layout.alignment: Qt.AlignHCenter
        Layout.fillWidth: true
        visible: root.detail.length > 0
        text: root.detail
        color: AppTheme.textMuted
        font.pixelSize: AppTheme.typeLabel
        horizontalAlignment: Text.AlignHCenter
        wrapMode: Text.WordWrap
    }

    readonly property string diagnosticsText:
        "message: " + root.message + "\n" +
        (root.detail ? "detail: " + root.detail + "\n" : "") +
        (root.stage ? "stage: " + root.stage + "\n" : "") +
        (root.file ? "file: " + root.file + "\n" : "") +
        (root.tracebackText ? "\n" + root.tracebackText : "")

    RowLayout {
        Layout.alignment: Qt.AlignHCenter
        Layout.topMargin: AppTheme.spacingSm
        spacing: AppTheme.spacingSm

        Button {
            objectName: "errorDetailsToggle"
            text: detailsPane.visible ? "Hide technical details" : "Show technical details"
            flat: true
            visible: root.tracebackText.length > 0 || root.stage.length > 0 || root.file.length > 0
            onClicked: detailsPane.visible = !detailsPane.visible
        }
        Button {
            objectName: "errorCopyDiagnostics"
            text: "Copy diagnostics"
            flat: true
            onClicked: Inspector.copyToClipboard(root.diagnosticsText)
            ToolTip.visible: hovered
            ToolTip.text: "Copy the full error message, detail, stage, file, and traceback to the clipboard"
            Accessible.name: "Copy diagnostics"
            Accessible.description: "Copies the full error message, detail, stage, file, and traceback to the clipboard"
        }
        Button {
            objectName: "errorOpenLog"
            text: "Open log"
            flat: true
            visible: root.showOpenLog
            onClicked: root.openLogRequested()
            ToolTip.visible: hovered
            ToolTip.text: "Open the hprofiler GUI log file"
            Accessible.name: "Open log"
            Accessible.description: "Opens the hprofiler GUI log file in the default text viewer"
        }
    }

    ScrollView {
        id: detailsPane
        objectName: "errorDetailsPane"
        visible: false
        Layout.alignment: Qt.AlignHCenter
        Layout.preferredWidth: root.width
        Layout.preferredHeight: Math.min(180, contentHeight + 16)
        clip: true
        ScrollBar.horizontal.policy: ScrollBar.AlwaysOff

        Rectangle {
            width: detailsPane.availableWidth
            height: detailsText.implicitHeight + AppTheme.spacingMd * 2
            color: AppTheme.surface
            radius: AppTheme.radiusSmall
            border.color: AppTheme.panelBorder
            border.width: 1

            Text {
                id: detailsText
                anchors.fill: parent
                anchors.margins: AppTheme.spacingMd
                text: (root.stage ? "stage: " + root.stage + "\n" : "") +
                      (root.file ? "file: " + root.file + "\n" : "") +
                      root.tracebackText
                color: AppTheme.textMuted
                font.family: "monospace"
                font.pixelSize: AppTheme.typeCaption
                wrapMode: Text.WrapAnywhere
            }
        }
    }
}
