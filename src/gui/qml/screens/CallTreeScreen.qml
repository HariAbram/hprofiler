import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import Hprofiler 1.0
import "../components"

ColumnLayout {
    id: root
    spacing: AppTheme.spacingSm

    property int sortKey: 0
    property bool sortDesc: true
    readonly property var sortKeys: ["", "totalNs", "selfNs", "count"]
    readonly property var sortLabels: ["Tree order", "Total", "Self", "Calls"]

    function applySort() {
        CallTree.sortChildren(root.sortKeys[root.sortKey], root.sortDesc)
    }

    Toolbar {
        title: "Call Tree"
        statusText: CallTree.roots.length + " root(s)"

        TextField {
            id: filterField
            Layout.preferredWidth: AppTheme.fieldWidth
            placeholderText: "filter (keeps ancestors of a match)…"
            color: AppTheme.text
            font.pixelSize: AppTheme.typeLabel
            onTextChanged: CallTree.setFilter(text)
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
                text: modelData + (root.sortKey === index && index > 0 ? (root.sortDesc ? " ▼" : " ▲") : "")
                font.pixelSize: AppTheme.typeLabel
                onClicked: {
                    if (root.sortKey === index) root.sortDesc = !root.sortDesc
                    else { root.sortKey = index; root.sortDesc = true }
                    root.applySort()
                }
            }
        }
        ToolButton {
            text: "Export CSV"
            font.pixelSize: AppTheme.typeLabel
            onClicked: {
                var path = (AppInfo.tracePath || "calltree") + ".calltree.csv"
                var ok = CallTree.exportCsv(path)
                exportStatus.text = ok ? ("Exported to " + path) : "Export failed"
            }
        }
        Text {
            id: exportStatus
            color: AppTheme.textMuted
            font.pixelSize: AppTheme.typeCaption
        }
    }

    Panel {
        Layout.fillWidth: true
        Layout.fillHeight: true
        title: ""
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
            visible: CallTree.allRoots.length === 0
            message: "No call-stack data in this trace.\nRun with --call-tree to capture it."
        }
        EmptyState {
            centered: true
            visible: CallTree.allRoots.length > 0 && CallTree.roots.length === 0
            message: "No call-tree nodes match this filter."
        }
    }
}
