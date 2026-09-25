import QtQuick
import QtQuick.Layouts
import Hprofiler 1.0
import "../components"

ColumnLayout {
    spacing: AppTheme.spacingSm

    Panel {
        Layout.fillWidth: true
        Layout.fillHeight: true
        title: "Call Tree"
        clip: true

        Flickable {
            id: flick
            anchors.fill: parent
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

        EmptyState {
            centered: true
            visible: CallTree.roots.length === 0
            message: "No call-stack data in this trace.\nRun with --call-tree to capture it."
        }
    }
}
