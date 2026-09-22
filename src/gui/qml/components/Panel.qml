import QtQuick
import QtQuick.Layouts
import Hprofiler 1.0

// Reusable bordered card with a title -- the QML equivalent of the TUI's
// border_title-carrying panels (src/ui/app.py's DashboardWidget etc.),
// used for every card/panel across every screen so they all look the
// same and the border color reacts to the theme in one place.
Rectangle {
    id: root
    property string title: ""
    default property alias content: contentItem.children

    color: AppTheme.surface
    border.color: AppTheme.panelBorder
    border.width: 1
    radius: 6

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 10
        spacing: 6

        Text {
            visible: root.title.length > 0
            text: root.title
            color: AppTheme.accent
            font.pixelSize: 13
            font.bold: true
        }

        Item {
            id: contentItem
            Layout.fillWidth: true
            Layout.fillHeight: true
        }
    }
}
