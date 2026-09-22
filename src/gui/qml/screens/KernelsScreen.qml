import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import Hprofiler 1.0

// GUI equivalent of the TUI's HotspotsWidget -- sortable/filterable
// function table. Sort/filter happen client-side over Kernels.rows
// (see bridge.py's KernelsBridge docstring for why that's fine at this
// data size: aggregated by function name, not per-span).
ColumnLayout {
    id: root
    spacing: 6

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

    RowLayout {
        Layout.fillWidth: true
        spacing: 8

        TextField {
            id: filterField
            Layout.preferredWidth: 240
            placeholderText: "filter by name…"
            color: AppTheme.text
            onTextChanged: root.filterText = text
        }
        Repeater {
            model: root.sortLabels
            delegate: Button {
                text: modelData + (root.sortKey === index ? (root.sortDesc ? " ▼" : " ▲") : "")
                onClicked: {
                    if (root.sortKey === index) root.sortDesc = !root.sortDesc
                    else { root.sortKey = index; root.sortDesc = true }
                }
            }
        }
        Item { Layout.fillWidth: true }
        Text {
            text: root.displayRows.length + " / " + Kernels.rows.length + " functions"
            color: AppTheme.textMuted
            font.pixelSize: 11
        }
    }

    Rectangle {
        Layout.fillWidth: true
        Layout.fillHeight: true
        color: AppTheme.surface
        border.color: AppTheme.panelBorder
        border.width: 1
        radius: 6
        clip: true

        ColumnLayout {
            anchors.fill: parent
            anchors.margins: 8
            spacing: 2

            RowLayout {
                Layout.fillWidth: true
                Text { text: "Function"; color: AppTheme.textMuted; font.bold: true; font.pixelSize: 11; Layout.fillWidth: true }
                Text { text: "Category"; color: AppTheme.textMuted; font.bold: true; font.pixelSize: 11; Layout.preferredWidth: 80 }
                Text { text: "Calls"; color: AppTheme.textMuted; font.bold: true; font.pixelSize: 11; Layout.preferredWidth: 60; horizontalAlignment: Text.AlignRight }
                Text { text: "Total"; color: AppTheme.textMuted; font.bold: true; font.pixelSize: 11; Layout.preferredWidth: 80; horizontalAlignment: Text.AlignRight }
                Text { text: "Avg"; color: AppTheme.textMuted; font.bold: true; font.pixelSize: 11; Layout.preferredWidth: 80; horizontalAlignment: Text.AlignRight }
                Text { text: "Share"; color: AppTheme.textMuted; font.bold: true; font.pixelSize: 11; Layout.preferredWidth: 60; horizontalAlignment: Text.AlignRight }
            }

            Rectangle { Layout.fillWidth: true; height: 1; color: AppTheme.panelBorder }

            ListView {
                Layout.fillWidth: true
                Layout.fillHeight: true
                model: root.displayRows
                clip: true
                delegate: Rectangle {
                    width: ListView.view.width
                    height: 26
                    color: index % 2 === 0 ? "transparent" : AppTheme.background

                    RowLayout {
                        anchors.fill: parent
                        anchors.rightMargin: 4
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 6
                            Rectangle { width: 8; height: 8; radius: 4; color: modelData.color }
                            Text {
                                text: modelData.name
                                color: AppTheme.text
                                font.pixelSize: 12
                                elide: Text.ElideRight
                                Layout.fillWidth: true
                            }
                        }
                        Text { text: modelData.category; color: modelData.color; font.pixelSize: 11; Layout.preferredWidth: 80 }
                        Text { text: modelData.count; color: AppTheme.textMuted; font.pixelSize: 12; Layout.preferredWidth: 60; horizontalAlignment: Text.AlignRight }
                        Text { text: modelData.total; color: AppTheme.text; font.pixelSize: 12; Layout.preferredWidth: 80; horizontalAlignment: Text.AlignRight }
                        Text { text: modelData.avg; color: AppTheme.textMuted; font.pixelSize: 12; Layout.preferredWidth: 80; horizontalAlignment: Text.AlignRight }
                        Text { text: modelData.share; color: AppTheme.text; font.pixelSize: 12; Layout.preferredWidth: 60; horizontalAlignment: Text.AlignRight }
                    }
                }

                Text {
                    anchors.centerIn: parent
                    visible: root.displayRows.length === 0
                    text: "No functions match this filter."
                    color: AppTheme.textMuted
                }
            }
        }
    }
}
