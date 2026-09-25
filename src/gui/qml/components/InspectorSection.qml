import QtQuick
import QtQuick.Layouts
import Hprofiler 1.0

// One collapsible group (Summary/Context/Metrics/Relationships/
// Recommendations) inside InspectorPanel -- click the header to toggle,
// same expand/collapse affordance TreeNode.qml already uses for call-
// tree rows, so collapsing behaves the same way everywhere in the GUI.
ColumnLayout {
    id: root
    property string title: ""
    property var fields: []
    property bool expanded: true
    Layout.fillWidth: true
    spacing: AppTheme.spacingXs

    MouseArea {
        Layout.fillWidth: true
        Layout.preferredHeight: headerRow.implicitHeight
        onClicked: root.expanded = !root.expanded

        RowLayout {
            id: headerRow
            width: parent.width
            spacing: AppTheme.spacingXs

            Text {
                text: root.expanded ? "▾" : "▸"
                color: AppTheme.textMuted
                font.pixelSize: AppTheme.typeLabel
            }
            Text {
                text: root.title + " (" + root.fields.length + ")"
                color: AppTheme.accent
                font.bold: true
                font.pixelSize: AppTheme.typeLabel
            }
        }
    }

    ColumnLayout {
        Layout.fillWidth: true
        visible: root.expanded
        spacing: AppTheme.spacingSm

        Repeater {
            model: root.fields
            delegate: InspectorField { field: modelData; Layout.fillWidth: true }
        }
        EmptyState {
            Layout.fillWidth: true
            visible: root.fields.length === 0
            message: "Nothing to show."
        }
    }
}
