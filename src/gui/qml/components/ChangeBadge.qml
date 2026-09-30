import QtQuick
import Hprofiler 1.0

// Small colored pill for a comparison row's status (improved/regressed/
// unchanged/new/removed) -- AppTheme.changeColor() drives the color,
// same palette DataTableCell.qml's status column already uses, so a
// badge here and the table's own status cell always agree.
Rectangle {
    id: root
    property string status: ""
    property string text: status

    implicitWidth: label.implicitWidth + AppTheme.spacingMd
    implicitHeight: label.implicitHeight + AppTheme.spacingXs
    radius: height / 2
    color: Qt.alpha(AppTheme.changeColor(root.status), 0.18)
    border.color: AppTheme.changeColor(root.status)
    border.width: 1

    Text {
        id: label
        anchors.centerIn: parent
        text: root.text
        color: AppTheme.changeColor(root.status)
        font.bold: true
        font.pixelSize: AppTheme.typeCaption
    }
}
