import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import Hprofiler 1.0

// Help > Keyboard Shortcuts reference -- renders src/gui/shortcuts.py's
// SHORTCUTS table directly (Shortcuts.rows), so this can never drift
// from what the app actually binds; there's exactly one source of truth
// for the shortcut list, not a hand-typed copy here plus real bindings
// elsewhere.
Dialog {
    id: root
    objectName: "shortcutsDialog"
    title: "Keyboard Shortcuts"
    modal: true
    standardButtons: Dialog.Close
    width: 420
    height: 420
    anchors.centerIn: parent ? Overlay.overlay : undefined

    ColumnLayout {
        anchors.fill: parent
        spacing: AppTheme.spacingMd

        ListView {
            Layout.fillWidth: true
            Layout.fillHeight: true
            clip: true
            model: Shortcuts.rows
            spacing: AppTheme.spacingXs

            section.property: "scope"
            section.criteria: ViewSection.FullString
            section.delegate: Text {
                width: ListView.view.width
                text: section
                color: AppTheme.text
                font.bold: true
                font.pixelSize: AppTheme.typeBody
                topPadding: AppTheme.spacingMd
                bottomPadding: AppTheme.spacingXs
            }

            delegate: RowLayout {
                width: ListView.view.width
                spacing: AppTheme.spacingMd

                Rectangle {
                    Layout.preferredWidth: 130
                    Layout.preferredHeight: 22
                    radius: AppTheme.radiusSmall
                    color: AppTheme.surface
                    border.color: AppTheme.panelBorder
                    border.width: 1
                    Text {
                        anchors.centerIn: parent
                        text: modelData.sequence
                        color: AppTheme.text
                        font.family: "monospace"
                        font.pixelSize: AppTheme.typeCaption
                    }
                }
                Text {
                    Layout.fillWidth: true
                    text: modelData.label
                    color: AppTheme.textMuted
                    font.pixelSize: AppTheme.typeLabel
                    wrapMode: Text.WordWrap
                }
            }
        }
    }
}
