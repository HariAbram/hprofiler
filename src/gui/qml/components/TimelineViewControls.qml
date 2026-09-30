import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import Hprofiler 1.0

// Timeline's grouping/collapse/show-all-lanes control (Phase B3) -- a
// separate control from TimelineFilterBar.qml on purpose: filters narrow
// WHAT's shown, this changes HOW the rows are organized. Per-lane hide/
// isolate live on each lane row itself (TimelineScreen.qml's laneRowComponent,
// right-click on the label), not here -- they're a per-row action, not a
// screen-level setting.
Item {
    id: root
    implicitWidth: rowLayout.implicitWidth
    implicitHeight: AppTheme.buttonHeight

    readonly property var groupOptions: ["none", "rank", "process", "runtime", "stream", "thread", "device"]
    readonly property var groupLabels: ({
        "none": "None", "rank": "Rank", "process": "Process", "runtime": "Runtime",
        "stream": "Stream", "thread": "Thread", "device": "Device"
    })

    function dimAvailable(key) {
        if (key === "none") return true
        var dims = TimelineModel.filterDimensions
        for (var i = 0; i < dims.length; i++)
            if (dims[i].key === key) return dims[i].available
        return true
    }

    RowLayout {
        id: rowLayout
        anchors.fill: parent
        spacing: AppTheme.spacingXs

        ToolButton {
            id: groupButton
            objectName: "timelineGroupButton"
            text: "Group: " + root.groupLabels[TimelineModel.grouping]
            onClicked: groupPopup.visible ? groupPopup.close() : groupPopup.open()
            ToolTip.visible: hovered
            ToolTip.text: "Group Timeline rows by rank, process, runtime, stream, or thread"
        }

        ToolButton {
            objectName: "timelineCollapseAllButton"
            visible: TimelineModel.grouping !== "none"
            text: "Collapse all"
            onClicked: TimelineModel.collapseAllGroups()
        }
        ToolButton {
            objectName: "timelineExpandAllButton"
            visible: TimelineModel.grouping !== "none"
            text: "Expand all"
            onClicked: TimelineModel.expandAllGroups()
        }
        ToolButton {
            objectName: "timelineShowAllLanesButton"
            visible: TimelineModel.hiddenLanes.length > 0 || TimelineModel.isolatedLanes.length > 0
            text: "Show all lanes"
            onClicked: TimelineModel.showAllLanes()
            ToolTip.visible: hovered
            ToolTip.text: TimelineModel.isolatedLanes.length > 0
                          ? "Isolating " + TimelineModel.isolatedLanes.length + " lane(s)"
                          : TimelineModel.hiddenLanes.length + " lane(s) hidden"
        }
        ToolButton {
            objectName: "timelineColorModeButton"
            readonly property var modes: ["function", "bucket", "category"]
            readonly property var modeLabels: ({"function": "Function", "bucket": "Activity", "category": "Category"})
            text: "Color: " + modeLabels[TimelineModel.colorMode]
            onClicked: {
                var idx = modes.indexOf(TimelineModel.colorMode)
                TimelineModel.setColorMode(modes[(idx + 1) % modes.length])
            }
            ToolTip.visible: hovered
            ToolTip.text: "Cycle span coloring: by function, activity bucket, or runtime category"
        }
    }

    Popup {
        id: groupPopup
        objectName: "timelineGroupPopup"
        y: groupButton.height + 2
        width: 180
        height: Math.min(260, optionColumn.implicitHeight + AppTheme.spacingMd * 2)
        modal: false
        focus: true
        closePolicy: Popup.CloseOnEscape | Popup.CloseOnPressOutsideParent

        background: Rectangle {
            color: AppTheme.surface
            border.color: AppTheme.panelBorder
            border.width: 1
            radius: AppTheme.radiusPanel
        }

        Column {
            id: optionColumn
            width: parent.width
            Repeater {
                model: root.groupOptions
                delegate: Button {
                    property string key: modelData
                    width: optionColumn.width
                    flat: true
                    visible: root.dimAvailable(key)
                    text: root.groupLabels[key] + (TimelineModel.grouping === key ? "  ✓" : "")
                    onClicked: { TimelineModel.setGrouping(key); groupPopup.close() }
                }
            }
        }
    }
}
