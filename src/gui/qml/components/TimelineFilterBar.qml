import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import Hprofiler 1.0

// Timeline's filter control -- a single button that opens a Popup with
// one section per TimelineModel.filterDimensions entry (rank/process/
// thread/runtime/stream/bucket, each only shown when `available`, so a
// trace with no MPI ranks or GPU streams doesn't show a dead control for
// them) plus event-level controls (name/regex, min duration, active-only,
// selected-time-range-only). Every change calls TimelineModel.applyFilters()
// immediately (no separate "Apply" step) -- filtering is cheap (computed
// once per call, not per repaint frame, per TimelineModel's own docs), so
// there's no debounce/staging benefit to a deferred apply here, only extra
// clicks for the user.
Item {
    id: root
    implicitWidth: filterButton.implicitWidth
    implicitHeight: AppTheme.buttonHeight

    property var ranks: []
    property var processes: []
    property var threads: []
    property var runtimes: []
    property var streams: []
    property var buckets: []
    property string nameQuery: ""
    property bool nameIsRegex: false
    property real minDurationNs: 0
    property bool activeOnly: false
    property bool timeRangeOnly: false

    // Maps a filterDimensions `key` to this component's own (plural)
    // list property name -- lets the generic per-dimension Repeater
    // below read/write the right property via root[...] bracket access
    // instead of one hand-written section per dimension.
    readonly property var _dimListKey: ({
        "rank": "ranks", "process": "processes", "thread": "threads",
        "runtime": "runtimes", "stream": "streams", "bucket": "buckets"
    })

    readonly property int activeCount: {
        var n = 0
        if (ranks.length) n++
        if (processes.length) n++
        if (threads.length) n++
        if (runtimes.length) n++
        if (streams.length) n++
        if (buckets.length) n++
        if (nameQuery.length) n++
        if (minDurationNs > 0) n++
        if (activeOnly) n++
        if (timeRangeOnly) n++
        return n
    }

    function _toggled(list, value) {
        var idx = list.indexOf(value)
        var copy = list.slice()
        if (idx >= 0) copy.splice(idx, 1)
        else copy.push(value)
        return copy
    }

    function dimValues(dimKey) {
        var propName = root._dimListKey[dimKey]
        return propName ? root[propName] : []
    }

    function toggleDim(dimKey, value) {
        var propName = root._dimListKey[dimKey]
        if (!propName) return
        root[propName] = root._toggled(root[propName], value)
        root.apply()
    }

    function apply() {
        var range = Nav.selectedTimeRange
        TimelineModel.applyFilters({
            "ranks": root.ranks, "processes": root.processes, "threads": root.threads,
            "runtimes": root.runtimes, "streams": root.streams, "buckets": root.buckets,
            "nameQuery": root.nameQuery, "nameIsRegex": root.nameIsRegex,
            "minDurationNs": root.minDurationNs, "activeOnly": root.activeOnly,
            "timeRangeOnly": root.timeRangeOnly,
            "rangeStartNs": range.startNs !== undefined ? range.startNs : 0,
            "rangeEndNs": range.endNs !== undefined ? range.endNs : 0
        })
    }

    function clearAll() {
        ranks = []; processes = []; threads = []; runtimes = []; streams = []
        buckets = []; nameQuery = ""; nameIsRegex = false; minDurationNs = 0
        activeOnly = false; timeRangeOnly = false
        TimelineModel.clearFilters()
    }

    ToolButton {
        id: filterButton
        objectName: "timelineFilterButton"
        anchors.fill: parent
        text: "Filters" + (root.activeCount > 0 ? " (" + root.activeCount + ")" : "")
        onClicked: popup.visible ? popup.close() : popup.open()
        ToolTip.visible: hovered
        ToolTip.text: "Filter Timeline rows and events"
    }

    Popup {
        id: popup
        objectName: "timelineFilterPopup"
        y: filterButton.height + 2
        width: 340
        height: Math.min(460, contentCol.implicitHeight + AppTheme.spacingMd * 2)
        modal: false
        focus: true
        closePolicy: Popup.CloseOnEscape | Popup.CloseOnPressOutsideParent

        background: Rectangle {
            color: AppTheme.surface
            border.color: AppTheme.panelBorder
            border.width: 1
            radius: AppTheme.radiusPanel
        }

        Flickable {
            anchors.fill: parent
            contentHeight: contentCol.implicitHeight
            clip: true

            ColumnLayout {
                id: contentCol
                width: parent.width
                spacing: AppTheme.spacingSm

                RowLayout {
                    Layout.fillWidth: true
                    Text {
                        text: "Filters"
                        color: AppTheme.accent
                        font.bold: true
                        font.pixelSize: AppTheme.typeTitle
                    }
                    Item { Layout.fillWidth: true }
                    Text {
                        objectName: "timelineFilterSummary"
                        text: TimelineModel.hiddenRowCount + " row(s) hidden, " +
                              TimelineModel.filteredSpanCount + "/" + TimelineModel.totalSpanCount + " events shown"
                        color: AppTheme.textMuted
                        font.pixelSize: AppTheme.typeCaption
                    }
                }

                Repeater {
                    model: TimelineModel.filterDimensions
                    delegate: ColumnLayout {
                        id: dimBlock
                        property var dim: modelData
                        Layout.fillWidth: true
                        visible: dim.available
                        spacing: 2

                        Text {
                            text: dimBlock.dim.label
                            color: AppTheme.textMuted
                            font.bold: true
                            font.pixelSize: AppTheme.typeCaption
                        }
                        Flow {
                            Layout.fillWidth: true
                            spacing: 2
                            Repeater {
                                model: dimBlock.dim.values
                                delegate: RowLayout {
                                    property var val: modelData
                                    spacing: 2
                                    Rectangle {
                                        visible: dimBlock.dim.key === "bucket"
                                        width: 8
                                        height: 8
                                        radius: 2
                                        color: AppTheme.bucketColor(val.value)
                                    }
                                    CheckBox {
                                        objectName: "filterCheck_" + dimBlock.dim.key + "_" + val.value
                                        text: val.label
                                        font.pixelSize: AppTheme.typeCaption
                                        checked: root.dimValues(dimBlock.dim.key).indexOf(val.value) >= 0
                                        onToggled: root.toggleDim(dimBlock.dim.key, val.value)
                                    }
                                }
                            }
                        }
                    }
                }

                Rectangle { Layout.fillWidth: true; height: 1; color: AppTheme.panelBorder }

                RowLayout {
                    Layout.fillWidth: true
                    spacing: AppTheme.spacingXs
                    TextField {
                        id: nameField
                        objectName: "filterNameQuery"
                        Layout.fillWidth: true
                        placeholderText: "event name…"
                        text: root.nameQuery
                        font.pixelSize: AppTheme.typeCaption
                        color: AppTheme.text
                        background: Rectangle {
                            color: AppTheme.background
                            border.color: nameField.activeFocus ? AppTheme.panelBorderFocus : AppTheme.panelBorder
                            border.width: 1
                            radius: AppTheme.radiusSmall
                        }
                        onTextChanged: { root.nameQuery = text; root.apply() }
                    }
                    CheckBox {
                        objectName: "filterRegex"
                        text: "regex"
                        font.pixelSize: AppTheme.typeCaption
                        checked: root.nameIsRegex
                        onToggled: { root.nameIsRegex = checked; root.apply() }
                    }
                }

                RowLayout {
                    Layout.fillWidth: true
                    spacing: AppTheme.spacingXs
                    Text {
                        text: "min duration (ns)"
                        color: AppTheme.textMuted
                        font.pixelSize: AppTheme.typeCaption
                    }
                    TextField {
                        id: durField
                        objectName: "filterMinDuration"
                        Layout.fillWidth: true
                        placeholderText: "0"
                        validator: DoubleValidator { bottom: 0 }
                        font.pixelSize: AppTheme.typeCaption
                        color: AppTheme.text
                        background: Rectangle {
                            color: AppTheme.background
                            border.color: durField.activeFocus ? AppTheme.panelBorderFocus : AppTheme.panelBorder
                            border.width: 1
                            radius: AppTheme.radiusSmall
                        }
                        onTextChanged: { root.minDurationNs = text.length ? parseFloat(text) : 0; root.apply() }
                    }
                }

                CheckBox {
                    objectName: "filterActiveOnly"
                    text: "Show only active rows"
                    font.pixelSize: AppTheme.typeCaption
                    checked: root.activeOnly
                    onToggled: { root.activeOnly = checked; root.apply() }
                }

                CheckBox {
                    objectName: "filterTimeRangeOnly"
                    text: "Show only selected time range"
                    font.pixelSize: AppTheme.typeCaption
                    enabled: Nav.selectedTimeRange.startNs !== undefined
                    checked: root.timeRangeOnly
                    onToggled: { root.timeRangeOnly = checked; root.apply() }
                    ToolTip.visible: hovered && !enabled
                    ToolTip.text: "No time range selected yet -- shift-drag on the lanes area to pick one"
                }

                Button {
                    objectName: "filterClearButton"
                    text: "Clear filters"
                    Layout.fillWidth: true
                    enabled: root.activeCount > 0
                    onClicked: root.clearAll()
                }
            }
        }
    }
}
