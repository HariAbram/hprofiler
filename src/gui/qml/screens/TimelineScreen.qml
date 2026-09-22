import QtQuick
import QtQuick.Layouts
import Hprofiler 1.0

// Paraver-style Gantt view -- the QML/Canvas counterpart of the TUI's
// TimelineWidget (src/ui/app.py). Real vector rectangles at whatever
// zoom level (not the TUI's fixed character-cell bucketing), smooth
// wheel-zoom and drag-pan, hover-gated MPI/NCCL connector lines drawn on
// a single overlay Canvas spanning every lane (same hover-gating
// rationale as the TUI: drawing every connector at once on a busy trace
// is a hairball -- see TimelineModel's docstring for the data side).
Item {
    id: root
    readonly property real rowHeight: 26
    readonly property real labelWidth: 130

    property real zoom: 1.0
    property real viewStartNs: TimelineModel.viewStartNs
    readonly property real visibleNs: TimelineModel.traceDurationNs / zoom

    property int hoverLane: -1
    property int hoverSpanIdx: -1
    property string hoverText: ""

    function resetView() {
        zoom = 1.0
        viewStartNs = TimelineModel.viewStartNs
    }

    function clampViewStart() {
        var maxStart = TimelineModel.viewStartNs + TimelineModel.traceDurationNs - visibleNs
        var minStart = TimelineModel.viewStartNs
        if (viewStartNs > maxStart) viewStartNs = Math.max(minStart, maxStart)
        if (viewStartNs < minStart) viewStartNs = minStart
    }

    function fmtNs(ns) {
        if (ns >= 1e9) return (ns / 1e9).toFixed(3) + "s"
        if (ns >= 1e6) return (ns / 1e6).toFixed(2) + "ms"
        if (ns >= 1e3) return (ns / 1e3).toFixed(1) + "µs"
        return ns.toFixed(0) + "ns"
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 4

        // ── Status / controls row ───────────────────────────────────────
        RowLayout {
            Layout.fillWidth: true
            Layout.preferredHeight: 24
            Layout.maximumHeight: 24
            spacing: 12

            Text {
                text: "zoom " + root.zoom.toFixed(1) + "×   offset " +
                      root.fmtNs(root.viewStartNs - TimelineModel.viewStartNs) +
                      "   window " + root.fmtNs(root.visibleNs)
                color: AppTheme.textMuted
                font.pixelSize: 11
            }
            Item { Layout.fillWidth: true }
            Text {
                visible: root.hoverText.length > 0
                text: root.hoverText
                color: AppTheme.text
                font.pixelSize: 11
                font.bold: true
            }
            Item { Layout.fillWidth: true }
            Text {
                text: "wheel: zoom · drag: pan · double-click: reset"
                color: AppTheme.textMuted
                font.pixelSize: 10
            }
        }

        // ── Lanes ────────────────────────────────────────────────────────
        Rectangle {
            Layout.fillWidth: true
            Layout.fillHeight: true
            color: AppTheme.surface
            border.color: AppTheme.panelBorder
            border.width: 1
            radius: 6
            clip: true

            Flickable {
                id: flick
                anchors.fill: parent
                anchors.margins: 1
                contentHeight: laneColumn.height
                boundsBehavior: Flickable.StopAtBounds

                Column {
                    id: laneColumn
                    width: flick.width

                    Repeater {
                        model: TimelineModel.lanes
                        delegate: Item {
                            width: laneColumn.width
                            height: root.rowHeight

                            Text {
                                width: root.labelWidth
                                height: parent.height
                                verticalAlignment: Text.AlignVCenter
                                text: modelData.label + "  (" + modelData.count + ")"
                                color: modelData.color
                                font.bold: true
                                font.pixelSize: 11
                                elide: Text.ElideRight
                                leftPadding: 6
                            }

                            Canvas {
                                id: laneCanvas
                                objectName: "laneCanvas_" + index
                                x: root.labelWidth
                                width: parent.width - root.labelWidth
                                height: parent.height
                                property int laneIndex: index
                                property var cachedSpans: []
                                property real boundZoom: root.zoom
                                property real boundStart: root.viewStartNs
                                onBoundZoomChanged: requestPaint()
                                onBoundStartChanged: requestPaint()

                                onPaint: {
                                    var ctx = getContext("2d")
                                    ctx.reset()
                                    var spans = TimelineModel.visibleSpans(
                                        laneIndex, root.viewStartNs, root.viewStartNs + root.visibleNs, 2000)
                                    cachedSpans = spans
                                    var scale = width / root.visibleNs
                                    for (var i = 0; i < spans.length; i++) {
                                        var sp = spans[i]
                                        var x0 = (sp.startNs - root.viewStartNs) * scale
                                        var w = Math.max(1, sp.durNs * scale)
                                        ctx.fillStyle = sp.color
                                        ctx.fillRect(Math.max(0, x0), 3, Math.min(w, width - x0), height - 6)
                                    }
                                }

                                MouseArea {
                                    anchors.fill: parent
                                    hoverEnabled: true
                                    acceptedButtons: Qt.NoButton
                                    onPositionChanged: (mouse) => {
                                        var ns = root.viewStartNs + mouse.x / laneCanvas.width * root.visibleNs
                                        var found = -1
                                        var spans = laneCanvas.cachedSpans
                                        for (var i = 0; i < spans.length; i++) {
                                            if (ns >= spans[i].startNs && ns <= spans[i].startNs + spans[i].durNs) {
                                                found = spans[i].spanIdx
                                                break
                                            }
                                        }
                                        if (found >= 0) {
                                            root.hoverLane = laneIndex
                                            root.hoverSpanIdx = found
                                            var detail = TimelineModel.spanAt(laneIndex, found)
                                            root.hoverText = detail.name + "  @" + root.fmtNs(detail.startNs) +
                                                             "  dur " + root.fmtNs(detail.durNs)
                                        } else {
                                            root.hoverLane = -1
                                            root.hoverSpanIdx = -1
                                            root.hoverText = ""
                                        }
                                        overlay.requestPaint()
                                    }
                                    onExited: {
                                        root.hoverLane = -1
                                        root.hoverSpanIdx = -1
                                        root.hoverText = ""
                                        overlay.requestPaint()
                                    }
                                }
                            }
                        }
                    }

                    // ── Connector overlay: spans every lane, draws only
                    // the hovered span's own MPI/NCCL edges ──────────────
                    Canvas {
                        id: overlay
                        x: root.labelWidth
                        y: 0
                        width: laneColumn.width - root.labelWidth
                        height: TimelineModel.lanes.length * root.rowHeight
                        z: 10

                        onPaint: {
                            var ctx = getContext("2d")
                            ctx.reset()
                            if (root.hoverLane < 0) return
                            var scale = width / root.visibleNs
                            var conns = TimelineModel.connectors
                            for (var i = 0; i < conns.length; i++) {
                                var c = conns[i]
                                if (c.predLane !== root.hoverLane && c.succLane !== root.hoverLane)
                                    continue
                                if ((c.predSpanIdx !== root.hoverSpanIdx || c.predLane !== root.hoverLane) &&
                                    (c.succSpanIdx !== root.hoverSpanIdx || c.succLane !== root.hoverLane))
                                    continue
                                var x0 = (c.predMidNs - root.viewStartNs) * scale
                                var x1 = (c.succMidNs - root.viewStartNs) * scale
                                var y0 = c.predLane * root.rowHeight + root.rowHeight / 2
                                var y1 = c.succLane * root.rowHeight + root.rowHeight / 2
                                ctx.strokeStyle = c.color
                                ctx.lineWidth = 2
                                ctx.beginPath()
                                ctx.moveTo(x0, y0)
                                ctx.lineTo(x1, y1)
                                ctx.stroke()
                                ctx.fillStyle = c.color
                                ctx.beginPath(); ctx.arc(x0, y0, 3, 0, 2 * Math.PI); ctx.fill()
                                ctx.beginPath(); ctx.arc(x1, y1, 3, 0, 2 * Math.PI); ctx.fill()
                            }
                        }
                    }
                }
            }

            // ── Zoom (wheel) + pan (drag) ─────────────────────────────────
            MouseArea {
                anchors.fill: parent
                anchors.leftMargin: root.labelWidth
                propagateComposedEvents: true
                acceptedButtons: Qt.LeftButton

                property real dragStartNs: 0
                property real dragStartX: 0

                onPressed: (mouse) => {
                    dragStartNs = root.viewStartNs
                    dragStartX = mouse.x
                }
                onPositionChanged: (mouse) => {
                    if (pressed) {
                        var scale = width / root.visibleNs
                        root.viewStartNs = dragStartNs - (mouse.x - dragStartX) / scale
                        root.clampViewStart()
                    }
                }
                onDoubleClicked: root.resetView()
                onWheel: (wheel) => {
                    var factor = wheel.angleDelta.y > 0 ? 1.25 : 0.8
                    root.zoom = Math.max(1.0, Math.min(256.0, root.zoom * factor))
                    root.clampViewStart()
                }
            }
        }
    }
}
