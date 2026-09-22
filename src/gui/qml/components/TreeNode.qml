import QtQuick
import QtQuick.Layouts
import Hprofiler 1.0

// Recursive call-tree row: a node's own line, plus (when expanded) one
// TreeNode per child, indented -- avoids needing a full
// QAbstractItemModel for what's typically a few hundred nodes at most
// (see bridge.py's CallTreeBridge docstring).
Column {
    id: root
    property var node
    property int depth: 0
    property real wallNs: 1
    width: parent ? parent.width : 0

    property bool expanded: depth < 2   // auto-expand the first couple of levels

    Rectangle {
        width: root.width
        height: 24
        color: rowMouse.containsMouse ? AppTheme.panelBorder : "transparent"

        RowLayout {
            anchors.fill: parent
            anchors.leftMargin: root.depth * 18
            anchors.rightMargin: 8
            spacing: 6

            Text {
                text: root.node.children.length > 0 ? (root.expanded ? "▾" : "▸") : " "
                color: AppTheme.textMuted
                font.pixelSize: 11
                Layout.preferredWidth: 14
            }
            Rectangle { width: 8; height: 8; radius: 4; color: root.node.color }
            Text {
                text: root.node.name
                color: AppTheme.text
                font.pixelSize: 12
                elide: Text.ElideRight
                Layout.fillWidth: true
            }
            Text {
                text: (root.wallNs > 0 ? (100.0 * root.node.totalNs / root.wallNs).toFixed(1) : "0.0") + "%"
                color: AppTheme.textMuted
                font.pixelSize: 11
                Layout.preferredWidth: 50
                horizontalAlignment: Text.AlignRight
            }
            Text {
                text: root.node.total
                color: AppTheme.text
                font.pixelSize: 12
                Layout.preferredWidth: 70
                horizontalAlignment: Text.AlignRight
            }
            Text {
                text: root.node.count + "×"
                color: AppTheme.textMuted
                font.pixelSize: 11
                Layout.preferredWidth: 50
                horizontalAlignment: Text.AlignRight
            }
        }

        MouseArea {
            id: rowMouse
            anchors.fill: parent
            hoverEnabled: true
            onClicked: root.expanded = !root.expanded
        }
    }

    Column {
        id: childrenColumn
        width: root.width
        visible: root.expanded
        Repeater {
            model: root.expanded ? root.node.children : []
            // A .qml file can't directly instantiate its own type
            // recursively (compile-time self-reference) -- Loader defers
            // resolution to runtime, which sidesteps that restriction.
            delegate: Loader {
                // Without an explicit width, this Loader and its loaded
                // TreeNode (which sets its own width from `parent.width`,
                // i.e. THIS Loader) reference each other circularly --
                // resolved to 0 in practice, collapsing every recursive
                // level to invisible. The whole subtree silently vanished
                // this way (no QML error/warning at all) until traced
                // down to this one missing binding.
                width: childrenColumn.width
                asynchronous: false
                // setSource's property map passes initial values to the
                // component's constructor (like arguments), evaluated
                // BEFORE the item's own bindings first run -- more
                // reliable for a recursive self-load than creating with
                // defaults via `source:` and mutating via onLoaded
                // afterwards, which left this tree rendering only its
                // root row with every child silently missing (no error,
                // just an empty result) when tried first.
                Component.onCompleted: setSource("TreeNode.qml", {
                    node: modelData,
                    depth: root.depth + 1,
                    wallNs: root.wallNs,
                })
            }
        }
    }
}
