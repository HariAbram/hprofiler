import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import Hprofiler 1.0
import "../components"

// GUI equivalent of the TUI's HotspotsWidget -- sortable/filterable
// function table. Sort/filter happen client-side over Kernels.rows
// (see bridge.py's KernelsBridge docstring for why that's fine at this
// data size: aggregated by function name, not per-span).
ColumnLayout {
    id: root
    spacing: AppTheme.spacingSm

    property int sortKey: 0   // index into sortKeys below
    property bool sortDesc: true
    property string filterText: ""
    readonly property var sortKeys: ["totalNs", "avgNs", "count", "minNs", "maxNs"]
    readonly property var sortLabels: ["Total", "Avg", "Count", "Min", "Max"]

    property var displayRows: []

    function recompute() {
        var rows = Kernels.rows
        var key = sortKeys[sortKey]
        var ft = filterText.toLowerCase()
        var filtered = ft.length === 0 ? rows : rows.filter(function(r) {
            return r.name.toLowerCase().indexOf(ft) !== -1
        })
        var sorted = filtered.slice().sort(function(a, b) {
            return sortDesc ? (b[key] - a[key]) : (a[key] - b[key])
        })
        displayRows = sorted
    }

    Component.onCompleted: recompute()
    onFilterTextChanged: recompute()
    onSortKeyChanged: recompute()
    onSortDescChanged: recompute()

    Toolbar {
        statusText: root.displayRows.length + " / " + Kernels.rows.length + " functions"

        TextField {
            id: filterField
            Layout.preferredWidth: AppTheme.fieldWidth
            placeholderText: "filter by name…"
            color: AppTheme.text
            font.pixelSize: AppTheme.typeLabel
            onTextChanged: root.filterText = text
            background: Rectangle {
                color: AppTheme.background
                border.color: filterField.activeFocus ? AppTheme.panelBorderFocus : AppTheme.panelBorder
                border.width: 1
                radius: AppTheme.radiusSmall
            }
        }
        Repeater {
            model: root.sortLabels
            delegate: Button {
                text: modelData + (root.sortKey === index ? (root.sortDesc ? " ▼" : " ▲") : "")
                font.pixelSize: AppTheme.typeLabel
                onClicked: {
                    if (root.sortKey === index) root.sortDesc = !root.sortDesc
                    else { root.sortKey = index; root.sortDesc = true }
                }
            }
        }
    }

    Panel {
        Layout.fillWidth: true
        Layout.fillHeight: true
        clip: true

        ColumnLayout {
            anchors.fill: parent
            spacing: AppTheme.spacingXs

            TableHeaderRow {
                Layout.fillWidth: true
                Text { text: "Function"; color: AppTheme.textMuted; font.bold: true; font.pixelSize: AppTheme.typeLabel; Layout.fillWidth: true }
                Text { text: "Category"; color: AppTheme.textMuted; font.bold: true; font.pixelSize: AppTheme.typeLabel; Layout.preferredWidth: 80 }
                Text { text: "Calls"; color: AppTheme.textMuted; font.bold: true; font.pixelSize: AppTheme.typeLabel; Layout.preferredWidth: 60; horizontalAlignment: Text.AlignRight }
                Text { text: "Total"; color: AppTheme.textMuted; font.bold: true; font.pixelSize: AppTheme.typeLabel; Layout.preferredWidth: 80; horizontalAlignment: Text.AlignRight }
                Text { text: "Avg"; color: AppTheme.textMuted; font.bold: true; font.pixelSize: AppTheme.typeLabel; Layout.preferredWidth: 80; horizontalAlignment: Text.AlignRight }
                Text { text: "Share"; color: AppTheme.textMuted; font.bold: true; font.pixelSize: AppTheme.typeLabel; Layout.preferredWidth: 60; horizontalAlignment: Text.AlignRight }
            }

            ListView {
                Layout.fillWidth: true
                Layout.fillHeight: true
                model: root.displayRows
                clip: true
                delegate: Rectangle {
                    id: kernelRow
                    width: ListView.view.width
                    height: AppTheme.rowComfortable
                    property bool selected: Nav.selectedCategory === modelData.category
                                             && Nav.selectedName === modelData.rawName
                    // A thin border, not a saturated fill, marks the
                    // selected row -- panelBorderFocus is a bright accent
                    // meant for 1px outlines elsewhere in this theme, not
                    // a full-row background (would fight the row text for
                    // contrast and read as an alert, not a selection).
                    color: rowMouse.containsMouse ? AppTheme.panelBorder
                           : (index % 2 === 0 ? "transparent" : AppTheme.background)
                    border.width: selected ? 1 : 0
                    border.color: AppTheme.panelBorderFocus

                    RowLayout {
                        anchors.fill: parent
                        anchors.rightMargin: AppTheme.spacingXs
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: AppTheme.spacingSm
                            Rectangle { width: 8; height: 8; radius: 4; color: modelData.color }
                            Text {
                                text: modelData.name
                                color: AppTheme.text
                                font.pixelSize: AppTheme.typeBody
                                elide: Text.ElideRight
                                Layout.fillWidth: true
                            }
                        }
                        Text { text: modelData.category; color: modelData.color; font.pixelSize: AppTheme.typeLabel; Layout.preferredWidth: 80 }
                        Text { text: modelData.count; color: AppTheme.textMuted; font.pixelSize: AppTheme.typeBody; Layout.preferredWidth: 60; horizontalAlignment: Text.AlignRight }
                        Text { text: modelData.total; color: AppTheme.text; font.pixelSize: AppTheme.typeBody; Layout.preferredWidth: 80; horizontalAlignment: Text.AlignRight }
                        Text { text: modelData.avg; color: AppTheme.textMuted; font.pixelSize: AppTheme.typeBody; Layout.preferredWidth: 80; horizontalAlignment: Text.AlignRight }
                        Text { text: modelData.share; color: AppTheme.text; font.pixelSize: AppTheme.typeBody; Layout.preferredWidth: 60; horizontalAlignment: Text.AlignRight }
                    }

                    MouseArea {
                        id: rowMouse
                        anchors.fill: parent
                        hoverEnabled: true
                        cursorShape: Qt.PointingHandCursor
                        onClicked: Nav.selectFunction(modelData.category, modelData.rawName)
                    }
                }

                EmptyState {
                    centered: true
                    visible: root.displayRows.length === 0
                    message: "No functions match this filter."
                }
            }
        }
    }
}
