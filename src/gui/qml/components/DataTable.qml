import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import Hprofiler 1.0
import "."

// Generic sortable/filterable table -- built on a real Qt model/view
// (src/gui/tablemodel.py's TableBundle: QAbstractListModel +
// QSortFilterProxyModel driving a plain ListView), not per-screen
// hand-rolled JS sort/filter. `table` is a TableBundle exposed by a
// bridge property (e.g. Kernels.table, System.deviceTable).
Item {
    id: root
    // Unique per instance (derived from `title`, which every real usage
    // sets) since this component is instantiated more than once across
    // the GUI (Kernels, System devices/metrics, Findings, ...) -- a
    // hardcoded objectName here would repeat the exact bug this round
    // already found and fixed for FlameGraphScreen's Tooltip (a
    // non-specific "first match wins" lookup silently grabbing the WRONG
    // instance's tooltip once a second one existed in the same window).
    objectName: "dataTable_" + title
    property var table: null            // TableBundle
    property string title: ""
    property bool showFilterField: true
    property bool showColumnMenu: true
    property bool showExport: true
    property bool compact: false        // small/dense mode (e.g. Overview's findings table)
    // Optional function(row) -> bool, lets each screen define its own
    // selection semantics (e.g. Kernels compares against Nav.selectedName)
    // without DataTable itself knowing what "selected" means for every table.
    property var isRowSelected: null
    signal rowClicked(var row)

    property real hScrollX: 0
    readonly property real scrollableWidth: {
        if (!table) return 0
        var cols = table.config.columns.filter(function(c) { return !c.frozen && c.visible })
        var w = 0
        for (var i = 0; i < cols.length; i++) w += cols[i].width
        return w
    }
    readonly property real scrollableViewport: Math.max(0, width - header.frozenWidth)
    readonly property real maxScrollX: Math.max(0, scrollableWidth - scrollableViewport)

    onMaxScrollXChanged: if (hScrollX > maxScrollX) hScrollX = maxScrollX

    implicitHeight: compact
        ? Math.min(360, toolbar.height + header.height + listView.contentHeight + 40)
        : 360

    ColumnLayout {
        anchors.fill: parent
        spacing: AppTheme.spacingXs

        Toolbar {
            id: toolbar
            title: root.title
            statusText: root.table ? (root.table.filters.matchCount + " / " + root.table.filters.sourceCount) : ""

            TextField {
                visible: root.showFilterField
                Layout.preferredWidth: AppTheme.fieldWidth
                placeholderText: "filter…"
                color: AppTheme.text
                font.pixelSize: AppTheme.typeLabel
                onTextChanged: if (root.table) root.table.filters.textFilter = text
                background: Rectangle {
                    color: AppTheme.background
                    border.color: parent.activeFocus ? AppTheme.panelBorderFocus : AppTheme.panelBorder
                    border.width: 1
                    radius: AppTheme.radiusSmall
                }
            }
            ToolButton {
                visible: root.showColumnMenu
                text: "Columns"
                font.pixelSize: AppTheme.typeLabel
                onClicked: columnMenu.visible ? columnMenu.close() : columnMenu.open()
            }
            ToolButton {
                objectName: "dataTablePercentModeToggle"
                text: root.table && root.table.config.percentMode ? "%" : "abs"
                font.pixelSize: AppTheme.typeLabel
                onClicked: if (root.table) root.table.config.togglePercentMode()
                ToolTip.visible: hovered
                ToolTip.text: "Toggle absolute values / percentage of a reference column"
                Accessible.name: root.table && root.table.config.percentMode
                                  ? "Showing percentages, switch to absolute values"
                                  : "Showing absolute values, switch to percentages"
                Accessible.description: "Toggles this table's numeric columns between absolute values and percentage of a reference column"
            }
            ToolButton {
                visible: root.showExport
                text: "Copy all"
                font.pixelSize: AppTheme.typeLabel
                onClicked: if (root.table) root.table.copyAll()
            }
            ToolButton {
                visible: root.showExport
                text: "Export CSV"
                font.pixelSize: AppTheme.typeLabel
                onClicked: {
                    if (!root.table) return
                    var path = (AppInfo.tracePath || "table") + "." + (root.title || "table").toLowerCase().replace(/\s+/g, "_") + ".csv"
                    var ok = root.table.exportCsv(path)
                    exportStatus.text = ok ? ("Exported to " + path) : "Export failed"
                }
            }
            Text {
                id: exportStatus
                color: AppTheme.textMuted
                font.pixelSize: AppTheme.typeCaption
            }
        }

        ColumnMenu {
            id: columnMenu
            table: root.table
            x: toolbar.width - width
            y: toolbar.height
        }

        DataTableHeader {
            id: header
            Layout.fillWidth: true
            table: root.table
            hScrollX: root.hScrollX
            onHoverDefinition: (text) => {
                if (text.length > 0) { hoverTip.text = text; hoverTip.visible = true }
                else hoverTip.visible = false
            }
        }

        Item {
            Layout.fillWidth: true
            Layout.fillHeight: true

            ListView {
                id: listView
                anchors.fill: parent
                anchors.bottomMargin: root.maxScrollX > 0 ? 12 : 0
                clip: true
                model: root.table ? root.table.rows : null
                delegate: DataTableRow {
                    width: listView.width
                    tableRow: row
                    columns: root.table ? root.table.config.columns : []
                    barMaxima: root.table ? root.table.barMaxima : ({})
                    percentMode: root.table ? root.table.config.percentMode : false
                    hScrollX: root.hScrollX
                    rowIndex: index
                    selected: root.isRowSelected ? root.isRowSelected(tableRow) : false
                    onRowClicked: root.rowClicked(tableRow)
                }

                // Mouse wheel over the body also scrolls horizontally when
                // shift is held -- vertical scroll (the common case) stays
                // ListView's own native wheel handling; this only adds a
                // NEW gesture, it doesn't take one away.
                MouseArea {
                    anchors.fill: parent
                    acceptedButtons: Qt.NoButton
                    onWheel: (wheel) => {
                        if (wheel.modifiers & Qt.ShiftModifier) {
                            root.hScrollX = Math.max(0, Math.min(root.maxScrollX, root.hScrollX - wheel.angleDelta.y / 2))
                            wheel.accepted = true
                        } else {
                            wheel.accepted = false
                        }
                    }
                }
            }

            EmptyState {
                anchors.centerIn: parent
                visible: root.table && root.table.filters.matchCount === 0
                message: root.table && root.table.filters.sourceCount === 0
                         ? "No rows." : "No rows match the current filter."
            }

            // Hand-rolled horizontal scrollbar -- panning is driven by
            // the plain hScrollX property above, not a real Flickable
            // contentX, so a stock attached ScrollBar doesn't apply (same
            // reasoning as TimelineScreen.qml's own horizontal scrollbar).
            // The drag MouseArea covers the whole FIXED track, not the
            // thumb itself, to avoid a moving-reference-frame bug (Round 8).
            Rectangle {
                id: hTrack
                visible: root.maxScrollX > 0
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.bottom: parent.bottom
                anchors.leftMargin: header.frozenWidth
                height: 10
                color: AppTheme.panelBorder
                radius: 4

                Rectangle {
                    id: hThumb
                    height: parent.height
                    radius: 4
                    color: AppTheme.accent
                    width: root.scrollableViewport > 0
                           ? Math.max(20, parent.width * root.scrollableViewport / root.scrollableWidth) : 0
                    x: root.maxScrollX > 0 ? (parent.width - width) * root.hScrollX / root.maxScrollX : 0
                }

                MouseArea {
                    anchors.fill: parent
                    onPressed: (mouse) => dragTo(mouse.x)
                    onPositionChanged: (mouse) => { if (pressed) dragTo(mouse.x) }
                    function dragTo(x) {
                        var usable = Math.max(1, hTrack.width - hThumb.width)
                        var frac = Math.max(0, Math.min(1, (x - hThumb.width / 2) / usable))
                        root.hScrollX = frac * root.maxScrollX
                    }
                }
            }
        }
    }

    Tooltip {
        id: hoverTip
        objectName: "dataTableTooltip_" + root.title
        property string text: ""
        visible: false
        followCursor: false
        parent: root
        Text {
            text: hoverTip.text
            color: AppTheme.text
            font.pixelSize: AppTheme.typeLabel
            wrapMode: Text.WordWrap
            width: Math.min(implicitWidth, 320)
        }
    }
}
