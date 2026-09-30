import QtQuick
import Hprofiler 1.0

// One ListView delegate for DataTable.qml -- frozen (identifier) columns
// render fixed at x=0, the rest scroll together under a single shared
// hScrollX (bound identically across every row AND the header), the same
// fixed-strip-plus-offset-content idea TimelineScreen.qml already uses for
// its lane labels, applied here to multiple narrow columns instead of one
// wide canvas.
Item {
    id: root
    property var tableRow: ({})       // the "row" role: whole source dict
    property var columns: []          // TableConfig.columns, already ordered
    property var barMaxima: ({})
    property bool percentMode: false
    property real hScrollX: 0
    property int rowIndex: 0
    property bool selected: false
    signal rowClicked()
    signal rowRightClicked()

    readonly property var frozenColumns: columns.filter(function(c) { return c.frozen && c.visible })
    readonly property var scrollColumns: columns.filter(function(c) { return !c.frozen && c.visible })
    readonly property real frozenWidth: {
        var w = 0
        for (var i = 0; i < frozenColumns.length; i++) w += frozenColumns[i].width
        return w
    }

    height: AppTheme.rowComfortable

    Accessible.role: Accessible.Button
    Accessible.name: root.frozenColumns.length > 0
                      ? String(root.tableRow[root.frozenColumns[0].key])
                      : ("Row " + (root.rowIndex + 1))
    Accessible.description: "Selects this row; right-click for more actions"
    Accessible.onPressAction: root.rowClicked()

    Rectangle {
        anchors.fill: parent
        color: root.selected ? AppTheme.panelBorder
               : (rowMouse.containsMouse ? Qt.alpha(AppTheme.panelBorder, 0.5)
                  : (root.rowIndex % 2 === 0 ? "transparent" : AppTheme.background))
    }
    Rectangle {
        visible: root.selected
        width: 2
        height: parent.height
        color: AppTheme.panelBorderFocus
    }

    Row {
        id: frozenRow
        x: 0
        height: parent.height
        Repeater {
            model: root.frozenColumns
            delegate: DataTableCell {
                width: modelData.width
                height: root.height
                column: modelData
                row: root.tableRow
                barMaxima: root.barMaxima
                percentMode: root.percentMode
            }
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
                delegate: DataTableCell {
                    width: modelData.width
                    height: root.height
                    column: modelData
                    row: root.tableRow
                    barMaxima: root.barMaxima
                    percentMode: root.percentMode
                }
            }
        }
    }

    MouseArea {
        id: rowMouse
        anchors.fill: parent
        hoverEnabled: true
        acceptedButtons: Qt.LeftButton | Qt.RightButton
        onClicked: (mouse) => {
            if (mouse.button === Qt.RightButton) root.rowRightClicked()
            else root.rowClicked()
        }
    }
}
