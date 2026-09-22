import QtQuick
import QtQuick.Layouts
import Hprofiler 1.0
import "../components"

ColumnLayout {
    spacing: 6

    Text {
        text: "Call Tree"
        color: AppTheme.accent
        font.bold: true
        font.pixelSize: 13
    }

    Rectangle {
        Layout.fillWidth: true
        Layout.fillHeight: true
        color: AppTheme.surface
        border.color: AppTheme.panelBorder
        border.width: 1
        radius: 6
        clip: true

        Flickable {
            id: flick
            anchors.fill: parent
            anchors.margins: 8
            contentHeight: col.height
            boundsBehavior: Flickable.StopAtBounds

            readonly property real totalWallNs: {
                var m = 0
                for (var i = 0; i < CallTree.roots.length; i++)
                    m = Math.max(m, CallTree.roots[i].totalNs)
                return m
            }

            Column {
                id: col
                width: flick.width
                Repeater {
                    model: CallTree.roots
                    delegate: TreeNode {
                        node: modelData
                        depth: 0
                        wallNs: flick.totalWallNs
                    }
                }
            }
        }

        Text {
            anchors.centerIn: parent
            visible: CallTree.roots.length === 0
            text: "No call-stack data in this trace.\nRun with --call-tree to capture it."
            horizontalAlignment: Text.AlignHCenter
            color: AppTheme.textMuted
        }
    }
}
