import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import Hprofiler 1.0

// A compact Timeline for any TimelineModel (`model`): the same rows,
// painters, hover and selection as the Timeline tab, without its toolbars
// -- the Compare tab stacks two of these (baseline above candidate). The
// owner holds the view (viewStartNs / visibleNs) and decides whether the
// two panes move together; this pane only asks (viewRequested).
Rectangle {
    id: pane
    property var model: null
    property string side: "candidate"
    property string title: ""
    property real viewStartNs: 0
    property real visibleNs: 1
    property real rowHeight: 20
    property real labelWidth: 170
    property real cursorNs: -1
    property int hoverLane: -1
    property real hoverSpanIdx: -1
    property string hoverText: ""

    signal viewRequested(real startNs, real visibleNs)
    signal eventClicked(int laneIndex, real spanIdx)
    signal cursorMoved(real ns)

    color: AppTheme.surface
    border.color: AppTheme.panelBorder
    border.width: 1
    radius: AppTheme.radiusPanel
    clip: true

    function zoomAt(factor, frac) {
        if (!pane.model) return
        var dur = pane.model.traceDurationNs
        var newVisible = Math.max(200, Math.min(dur, pane.visibleNs / factor))
        var anchor = pane.viewStartNs + frac * pane.visibleNs
        pane.viewRequested(anchor - frac * newVisible, newVisible)
    }

    Text {
        id: titleText
        x: 6
        y: 2
        width: parent.width - 12
        text: pane.title
        color: AppTheme.textMuted
        font.pixelSize: AppTheme.typeCaption
        elide: Text.ElideMiddle
    }

    ListView {
        id: list
        objectName: "timelinePaneList_" + pane.side
        anchors.fill: parent
        anchors.topMargin: 16
        anchors.margins: 1
        anchors.rightMargin: 12
        clip: true
        boundsBehavior: Flickable.StopAtBounds
        model: pane.model ? pane.model.rows : []
        ScrollBar.vertical: ScrollBar { policy: ScrollBar.AsNeeded }
        delegate: Item {
            id: row
            width: list.width
            height: pane.rowHeight
            property var rowData: modelData
            readonly property bool isLane: rowData.kind === "lane"
            readonly property int level: rowData.level !== undefined ? rowData.level : 0
            Text {
                x: 4 + 10 * row.level
                width: pane.labelWidth - x - 2
                height: parent.height
                verticalAlignment: Text.AlignVCenter
                text: (row.isLane ? "" : (row.rowData.collapsed ? "▸ " : "▾ ")) +
                      (row.isLane ? (row.rowData.hierLabel || row.rowData.label) : row.rowData.label)
                color: row.isLane ? AppTheme.categoryColor(row.rowData.category, AppTheme.dark) : AppTheme.text
                font.pixelSize: AppTheme.typeCaption
                font.bold: !row.isLane
                elide: Text.ElideRight
                MouseArea {
                    anchors.fill: parent
                    enabled: !row.isLane
                    onClicked: pane.model.setGroupCollapsed(row.rowData.groupId, !row.rowData.collapsed)
                }
            }
            LanePainter {
                id: painter
                objectName: "paneLane_" + pane.side + "_" + (row.isLane ? row.rowData.laneIndex : row.rowData.groupId)
                x: pane.labelWidth
                width: parent.width - pane.labelWidth
                height: parent.height
                model: pane.model
                kind: row.isLane ? "lane" : "group"
                laneIndex: row.isLane ? row.rowData.laneIndex : -1
                groupId: row.isLane ? "" : row.rowData.groupId
                laneIndexes: row.isLane ? [] : row.rowData.laneIndexes
                laneName: row.isLane ? row.rowData.name : ""
                viewStartNs: pane.viewStartNs
                viewEndNs: pane.viewStartNs + pane.visibleNs
                revision: pane.model ? pane.model.paintRevision : 0
                MouseArea {
                    anchors.fill: parent
                    hoverEnabled: true
                    acceptedButtons: Qt.LeftButton
                    property real pressX: 0
                    property real pressStart: 0
                    onPositionChanged: (mouse) => {
                        var ns = pane.viewStartNs + mouse.x / painter.width * pane.visibleNs
                        pane.cursorMoved(ns)
                        if (pressed) {
                            var scale = painter.width / pane.visibleNs
                            pane.viewRequested(pressStart - (mouse.x - pressX) / scale, pane.visibleNs)
                            return
                        }
                        if (!row.isLane) { pane.hoverText = row.rowData.label + " (aggregate)"; return }
                        var hit = pane.model.hitTest(row.rowData.laneIndex, mouse.x, pane.viewStartNs,
                                                     pane.viewStartNs + pane.visibleNs, painter.width)
                        if (hit.kind === "span") {
                            pane.hoverLane = row.rowData.laneIndex
                            pane.hoverSpanIdx = hit.spanIdx
                            pane.hoverText = hit.name + "  dur " + Format.duration(hit.durNs)
                        } else {
                            pane.hoverLane = -1
                            pane.hoverSpanIdx = -1
                            pane.hoverText = hit.kind === "column" ? ("aggregated: " + hit.count + " event(s) < 1 px, busy "
                                                                      + (100 * hit.busy).toFixed(1) + "%") : ""
                        }
                    }
                    onPressed: (mouse) => { pressX = mouse.x; pressStart = pane.viewStartNs }
                    onClicked: (mouse) => {
                        if (Math.abs(mouse.x - pressX) > 4) return
                        if (pane.hoverSpanIdx >= 0) pane.eventClicked(pane.hoverLane, pane.hoverSpanIdx)
                    }
                    onExited: { pane.hoverText = ""; pane.cursorMoved(-1) }
                    onWheel: (wheel) => {
                        if (wheel.modifiers & Qt.ShiftModifier) {
                            list.contentY = Math.max(0, Math.min(Math.max(0, list.contentHeight - list.height),
                                                                 list.contentY + (wheel.angleDelta.y > 0 ? -3 : 3) * pane.rowHeight))
                            return
                        }
                        pane.zoomAt(wheel.angleDelta.y > 0 ? 1.25 : 0.8, wheel.x / painter.width)
                    }
                }
            }
        }
    }

    Rectangle {
        // the synchronized cursor (same instant after the alignment offset)
        visible: pane.cursorNs >= pane.viewStartNs && pane.cursorNs <= pane.viewStartNs + pane.visibleNs
        x: pane.labelWidth + 1 + (pane.cursorNs - pane.viewStartNs) / pane.visibleNs * (list.width - pane.labelWidth)
        y: 16
        width: 1
        height: parent.height - 16
        color: AppTheme.accent
        opacity: 0.7
    }

    function revealLane(laneIndex) {
        if (!pane.model) return
        var pos = pane.model.rowIndexForLane(laneIndex)
        if (pos >= 0) list.positionViewAtIndex(pos, ListView.Contain)
    }
}
