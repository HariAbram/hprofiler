import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import Hprofiler 1.0

// Column visibility toggle for DataTable.qml -- a checkbox per column plus
// "Reset layout" (restores default widths/visibility/order).
Popup {
    id: root
    property var table: null   // TableBundle
    modal: false
    focus: true
    width: 220
    height: Math.min(360, contentColumn.implicitHeight + AppTheme.spacingMd * 2)

    background: Rectangle {
        color: AppTheme.surface
        border.color: AppTheme.panelBorder
        border.width: 1
        radius: AppTheme.radiusPanel
    }

    ColumnLayout {
        id: contentColumn
        anchors.fill: parent
        spacing: AppTheme.spacingXs

        Text {
            text: "Columns"
            color: AppTheme.accent
            font.bold: true
            font.pixelSize: AppTheme.typeTitle
        }

        Flickable {
            Layout.fillWidth: true
            Layout.fillHeight: true
            contentHeight: checkColumn.height
            clip: true

            Column {
                id: checkColumn
                width: parent.width
                Repeater {
                    model: root.table ? root.table.config.columns : []
                    delegate: CheckBox {
                        width: checkColumn.width
                        text: modelData.title
                        checked: modelData.visible
                        onToggled: root.table.config.setColumnVisible(modelData.key, checked)
                    }
                }
            }
        }

        Button {
            text: "Reset layout"
            Layout.fillWidth: true
            onClicked: if (root.table) root.table.config.resetLayout()
        }
    }
}
