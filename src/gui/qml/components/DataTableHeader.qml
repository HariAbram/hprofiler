import QtQuick
import Hprofiler 1.0

// Header row for DataTable.qml -- same frozen+scroll layout as
// DataTableRow.qml (must stay pixel-identical so columns line up), plus
// click-to-sort and a right-edge resize handle per column. Supersedes
// TableHeaderRow.qml for tables built on this component (that file stays,
// it's still used by screens not yet migrated).
Item {
    id: root
    property var table: null   // TableBundle
    property real hScrollX: 0
    signal hoverDefinition(string text)

    readonly property var allColumns: table ? table.config.columns : []
    readonly property var frozenColumns: allColumns.filter(function(c) { return c.frozen && c.visible })
    readonly property var scrollColumns: allColumns.filter(function(c) { return !c.frozen && c.visible })
    readonly property real frozenWidth: {
        var w = 0
        for (var i = 0; i < frozenColumns.length; i++) w += frozenColumns[i].width
        return w
    }

    height: AppTheme.rowCompact + AppTheme.spacingXs

    Rectangle { anchors.fill: parent; color: AppTheme.surface }

    Row {
        id: frozenRow
        x: 0
        height: parent.height
        Repeater {
            model: root.frozenColumns
            delegate: headerCellComp
        }
    }

    Item {
        x: frozenRow.width
        width: Math.max(0, root.width - frozenRow.width)
        height: parent.height
        clip: true
        Row {
            x: -root.hScrollX
            height: parent.height
            Repeater {
                model: root.scrollColumns
                delegate: headerCellComp
            }
        }
    }

    Rectangle {
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.bottom: parent.bottom
        height: 1
        color: AppTheme.panelBorder
    }

    Component {
        id: headerCellComp
        Item {
            id: cell
            width: modelData.width
            height: root.height
            property var spec: modelData

            Text {
                anchors.left: parent.left
                anchors.right: resizeHandle.left
                anchors.verticalCenter: parent.verticalCenter
                anchors.leftMargin: 4
                text: cell.spec.title
                    + (table && table.filters.sortKey === cell.spec.key
                       ? (table.filters.sortDescending ? " ▼" : " ▲") : "")
                color: AppTheme.textMuted
                font.bold: true
                font.pixelSize: AppTheme.typeLabel
                horizontalAlignment: cell.spec.align === "left" ? Text.AlignLeft : Text.AlignRight
                elide: Text.ElideRight
            }

            MouseArea {
                anchors.fill: parent
                anchors.rightMargin: 6
                hoverEnabled: true
                onClicked: if (table) table.filters.toggleSort(cell.spec.key)
                onEntered: root.hoverDefinition(cell.spec.definition || "")
                onExited: root.hoverDefinition("")

                Accessible.role: Accessible.Button
                Accessible.name: "Sort by " + cell.spec.title
                Accessible.description: (table && table.filters.sortKey === cell.spec.key)
                    ? ("Currently sorted " + (table.filters.sortDescending ? "descending" : "ascending") + "; click to reverse")
                    : "Sorts the table by this column"
                Accessible.onPressAction: if (table) table.filters.toggleSort(cell.spec.key)
            }

            Rectangle {
                id: resizeHandle
                anchors.right: parent.right
                anchors.top: parent.top
                anchors.bottom: parent.bottom
                width: 4
                color: resizeMouse.containsMouse || resizeMouse.drag.active ? AppTheme.panelBorderFocus : "transparent"

                MouseArea {
                    id: resizeMouse
                    anchors.fill: parent
                    cursorShape: Qt.SizeHorCursor
                    hoverEnabled: true
                    property real startSceneX: 0
                    property real startWidth: 0

                    Accessible.role: Accessible.Button
                    Accessible.name: "Resize " + cell.spec.title + " column"
                    Accessible.description: "Drag to resize this column's width"
                    // Track drag distance in `root`'s (stable) coordinate
                    // space, NOT this handle's own local mouse.x -- the
                    // handle's position shifts as the column resizes out
                    // from under it mid-drag (spec.width is a reactive
                    // Python-backed property), so a naive local mouse.x
                    // delta would use a moving reference frame and jitter.
                    // Same technique as the horizontal-scrollbar thumb.
                    onPressed: (mouse) => {
                        startSceneX = resizeMouse.mapToItem(root, mouse.x, mouse.y).x
                        startWidth = cell.spec.width
                    }
                    onPositionChanged: (mouse) => {
                        if (pressed && table) {
                            var sceneX = resizeMouse.mapToItem(root, mouse.x, mouse.y).x
                            var newWidth = Math.max(40, startWidth + (sceneX - startSceneX))
                            table.config.setColumnWidth(cell.spec.key, newWidth)
                        }
                    }
                }
            }
        }
    }
}
