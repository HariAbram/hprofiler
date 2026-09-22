import QtQuick
import Hprofiler 1.0

// Temporary stand-in for a screen not built yet in this phase --
// replaced one at a time as later phases land, never left silently
// blank (a blank screen reads as broken; this reads as "not built yet").
Item {
    property string screenName: ""

    Column {
        anchors.centerIn: parent
        spacing: 8
        Text {
            text: screenName
            color: AppTheme.textMuted
            font.pixelSize: 18
            anchors.horizontalCenter: parent.horizontalCenter
        }
        Text {
            text: "Not built yet in this phase"
            color: AppTheme.textMuted
            font.pixelSize: 12
            anchors.horizontalCenter: parent.horizontalCenter
        }
    }
}
