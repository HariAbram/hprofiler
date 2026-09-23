import QtQuick
import QtQuick.Controls
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
                objectName: "timelineFlick"
                anchors.fill: parent
                anchors.margins: 1
                anchors.rightMargin: 13
                anchors.bottomMargin: 13
                contentHeight: laneColumn.height
                boundsBehavior: Flickable.StopAtBounds

                // Real lanes (rows), for a trace with more of them than
                // fit the window -- Flickable already supported this
                // (contentHeight was always bound correctly), it just had
                // no way to actually GET there: the pan/zoom MouseArea
                // below sits on top of the whole area and only reads
                // mouse.x, so a vertical drag silently did nothing at
                // all. A real ScrollBar thumb, in its own reserved strip
                // the pan MouseArea explicitly excludes (rightMargin
                // below), fixes that without the two drag gestures
                // (Flickable's native drag-to-scroll vs. the pan
                // MouseArea's drag-to-pan-in-time) fighting each other.
                ScrollBar.vertical: ScrollBar {
                    policy: TimelineModel.lanes.length * root.rowHeight > flick.height
                            ? ScrollBar.AlwaysOn : ScrollBar.AlwaysOff
                }

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
                                // Throttled, not immediate: a drag/wheel gesture fires
                                // dozens of these changes per second, and each repaint
                                // means a Python round-trip (TimelineModel.visibleSpans)
                                // per lane -- with 17 lanes, painting on every single
                                // change made dragging feel very sluggish, worse still
                                // over X11 forwarding where each composited frame also
                                // pays network latency. Capped to ~60fps via a THROTTLE
                                // (start-if-not-already-running), not a debounce/restart
                                // -- it keeps repainting periodically throughout a long
                                // continuous drag instead of only once movement pauses.
                                onBoundZoomChanged: if (!repaintThrottle.running) repaintThrottle.start()
                                onBoundStartChanged: if (!repaintThrottle.running) repaintThrottle.start()

                                Timer {
                                    id: repaintThrottle
                                    interval: 16
                                    repeat: false
                                    onTriggered: laneCanvas.requestPaint()
                                }

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
                                        // Skip the Python round-trip (spanAt) and repaint
                                        // entirely when the hovered span hasn't actually
                                        // changed -- onPositionChanged fires on every
                                        // single mouse-moved pixel, not just on entering a
                                        // new span, so without this guard a 500px-wide
                                        // hover over one long span meant ~500 redundant
                                        // Python calls + overlay repaints for no visible
                                        // change at all.
                                        //
                                        // NOTE: laneIndex is a property of the enclosing
                                        // Canvas (laneCanvas), not of this MouseArea --
                                        // QML does not resolve a parent item's custom
                                        // properties unqualified from a nested child's
                                        // scope, only via its id. Referencing bare
                                        // `laneIndex` here throws a silent JS
                                        // ReferenceError on every hover move (visible only
                                        // via engine.warnings, which nothing was reading
                                        // at runtime) -- hover was completely non-
                                        // functional from when this screen was first
                                        // built; only ever verified via static screenshots,
                                        // never a real synthesized mouse move, so this
                                        // never surfaced until tested with QTest.mouseMove.
                                        if (found === root.hoverSpanIdx && laneCanvas.laneIndex === root.hoverLane) return
                                        if (found >= 0) {
                                            root.hoverLane = laneCanvas.laneIndex
                                            root.hoverSpanIdx = found
                                            var detail = TimelineModel.spanAt(laneCanvas.laneIndex, found)
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
                                        if (root.hoverLane !== laneCanvas.laneIndex) return
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
            // rightMargin/bottomMargin match the Flickable's above --
            // reserves the vertical/horizontal scrollbar strips so this
            // (which covers everything else and consumes all drag input
            // for time-panning) never overlaps and steals their clicks.
            MouseArea {
                anchors.fill: parent
                anchors.leftMargin: root.labelWidth
                anchors.rightMargin: 13
                anchors.bottomMargin: 13
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

            // ── Horizontal time-scrollbar ───────────────────────────────
            // A real QtQuick.Controls ScrollBar wouldn't work here the way
            // the vertical one above does -- panning isn't driven by a
            // real Flickable's contentX, it's the custom viewStartNs/zoom
            // state the wheel-zoom/drag-pan MouseArea manages, so this is
            // a small custom thumb driven by that same state instead.
            // Doubles as an at-a-glance "how much of the trace am I
            // looking at" indicator, which wheel-zoom + drag-pan alone
            // don't give you.
            Rectangle {
                id: hScrollTrack
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.bottom: parent.bottom
                anchors.leftMargin: root.labelWidth + 1
                anchors.rightMargin: 14
                anchors.bottomMargin: 1
                height: 12
                radius: 4
                color: AppTheme.background
                visible: root.visibleNs < TimelineModel.traceDurationNs

                Rectangle {
                    id: hScrollThumb
                    readonly property real thumbFrac: Math.min(1.0, root.visibleNs / TimelineModel.traceDurationNs)
                    readonly property real scrollRangeNs: Math.max(TimelineModel.traceDurationNs - root.visibleNs, 1)
                    readonly property real scrollFrac: Math.max(0.0, Math.min(1.0,
                        (root.viewStartNs - TimelineModel.viewStartNs) / scrollRangeNs))
                    // Bindings, never imperatively assigned (e.g. via
                    // drag.target) -- this stays a pure function of
                    // root.viewStartNs/zoom so it's always correct
                    // regardless of whether THIS thumb, the main pan
                    // drag, or the wheel is what last changed the view.
                    x: (hScrollTrack.width - width) * scrollFrac
                    width: Math.max(20, hScrollTrack.width * thumbFrac)
                    height: parent.height
                    radius: 4
                    color: hDrag.pressed ? AppTheme.accent : AppTheme.panelBorder
                }

                // Fills the whole (fixed, non-moving) track rather than
                // just the thumb, so mouse.x during a drag is measured
                // against a stable reference frame -- a MouseArea on the
                // thumb itself would be measuring against a target that's
                // moving out from under the cursor as a RESULT of that
                // same drag, corrupting the delta.
                MouseArea {
                    id: hDrag
                    anchors.fill: parent
                    property real dragStartX: 0
                    property real dragStartViewNs: 0

                    onPressed: (mouse) => {
                        dragStartX = mouse.x
                        dragStartViewNs = root.viewStartNs
                    }
                    onPositionChanged: (mouse) => {
                        if (!pressed) return
                        var trackSpan = hScrollTrack.width - hScrollThumb.width
                        if (trackSpan <= 0) return
                        var deltaFrac = (mouse.x - dragStartX) / trackSpan
                        root.viewStartNs = dragStartViewNs + deltaFrac * hScrollThumb.scrollRangeNs
                        root.clampViewStart()
                    }
                }
            }
        }

        // ── Call graph: who-calls-whom for whatever's currently visible ────
        // A node-and-edge diagram (analysis/call_graph.py), NOT the same
        // thing as the Call Tree tab: that shows time breakdown down each
        // specific call PATH (the same function under two different
        // callers is two separate rows there, by design); this merges
        // every occurrence of a function into ONE box regardless of
        // caller, with edges showing the distinct call relationships --
        // answering "which functions call which, overall" for the
        // CURRENT TIME WINDOW, not "how expensive is this specific path"
        // for the whole trace.
        Rectangle {
            id: callGraphPanel
            Layout.fillWidth: true
            Layout.preferredHeight: 210
            Layout.maximumHeight: 210
            color: AppTheme.surface
            border.color: AppTheme.panelBorder
            border.width: 1
            radius: 6
            clip: true

            property var graphData: ({nodes: [], edges: [], truncated: 0})
            property var hoveredNode: null
            property var hoverPos: Qt.point(0, 0)

            function refresh() {
                graphData = TimelineModel.callGraph(root.viewStartNs, root.viewStartNs + root.visibleNs)
                graphCanvas.requestPaint()
            }

            // Rebuilding the graph is a real Python round-trip over
            // every visible span (twice: once to aggregate, once to
            // lay out) -- much heavier than a single lane's
            // visibleSpans() call, so this uses a slower, separate
            // throttle (~4/sec) rather than the lane canvases' ~60fps
            // one; still feels live while panning/zooming without
            // costing a rebuild on every single pixel of drag.
            Timer {
                id: graphRefreshThrottle
                interval: 250
                repeat: false
                onTriggered: callGraphPanel.refresh()
            }
            // NOTE: zoom/viewStartNs are root's properties, not this
            // item's -- bare onZoomChanged/onViewStartNsChanged handlers
            // declared directly here would silently never fire (exactly
            // the class of bug already found once this session in the
            // per-lane hover handlers: an unqualified reference to an
            // ancestor's property). Connections{target: root} is the
            // correct way to listen to a DIFFERENT item's property
            // changes from here.
            Connections {
                target: root
                function onViewStartNsChanged() {
                    if (!graphRefreshThrottle.running) graphRefreshThrottle.start()
                }
                function onZoomChanged() {
                    if (!graphRefreshThrottle.running) graphRefreshThrottle.start()
                }
            }
            Component.onCompleted: refresh()

            Text {
                anchors.top: parent.top
                anchors.left: parent.left
                anchors.margins: 6
                text: "Call graph (visible window)" +
                      (callGraphPanel.graphData.truncated > 0
                       ? "  ·  +" + callGraphPanel.graphData.truncated + " more not shown"
                       : "")
                color: AppTheme.textMuted
                font.pixelSize: 10
                z: 5
            }

            Text {
                anchors.centerIn: parent
                visible: callGraphPanel.graphData.nodes.length === 0
                text: "No call-stack data for the current view.\nRun with --call-tree to capture it."
                horizontalAlignment: Text.AlignHCenter
                color: AppTheme.textMuted
                font.pixelSize: 12
            }

            Canvas {
                id: graphCanvas
                anchors.fill: parent
                anchors.topMargin: 20
                anchors.margins: 8

                readonly property real boxW: 120
                readonly property real boxH: 30

                onPaint: {
                    var ctx = getContext("2d")
                    ctx.reset()
                    var data = callGraphPanel.graphData
                    var nodes = data.nodes
                    var edges = data.edges
                    if (nodes.length === 0) return

                    var W = width, H = height
                    function px(n) { return n.x * (W - graphCanvas.boxW) + graphCanvas.boxW / 2 }
                    function py(n) { return n.y * (H - graphCanvas.boxH) + graphCanvas.boxH / 2 }

                    // Edges first (under the node boxes), thickness/alpha
                    // scaled by relative time weight so hot call paths
                    // visually stand out.
                    var maxEdgeNs = 1
                    for (var i = 0; i < edges.length; i++)
                        maxEdgeNs = Math.max(maxEdgeNs, edges[i].totalNs)
                    for (i = 0; i < edges.length; i++) {
                        var e = edges[i]
                        var a = nodes[e.callerIdx], b = nodes[e.calleeIdx]
                        var x0 = px(a), y0 = py(a), x1 = px(b), y1 = py(b)
                        var weight = e.totalNs / maxEdgeNs
                        ctx.strokeStyle = AppTheme.dark
                            ? "rgba(139,148,158," + (0.3 + 0.6 * weight) + ")"
                            : "rgba(101,109,118," + (0.3 + 0.6 * weight) + ")"
                        ctx.lineWidth = 1 + 2 * weight
                        ctx.beginPath()
                        ctx.moveTo(x0, y0)
                        ctx.lineTo(x1, y1)
                        ctx.stroke()
                        // Arrowhead at the callee end
                        var ang = Math.atan2(y1 - y0, x1 - x0)
                        var ah = 6
                        var tx = x1 - Math.cos(ang) * (graphCanvas.boxW / 2)
                        var ty = y1 - Math.sin(ang) * (graphCanvas.boxH / 2)
                        ctx.beginPath()
                        ctx.moveTo(tx, ty)
                        ctx.lineTo(tx - ah * Math.cos(ang - Math.PI / 6), ty - ah * Math.sin(ang - Math.PI / 6))
                        ctx.lineTo(tx - ah * Math.cos(ang + Math.PI / 6), ty - ah * Math.sin(ang + Math.PI / 6))
                        ctx.closePath()
                        ctx.fill()
                    }

                    // Nodes on top.
                    for (i = 0; i < nodes.length; i++) {
                        var n = nodes[i]
                        var cx = px(n), cy = py(n)
                        var hovered = callGraphPanel.hoveredNode === n
                        ctx.fillStyle = n.color
                        ctx.globalAlpha = hovered ? 1.0 : 0.85
                        ctx.beginPath()
                        // Plain rect, not roundedRect -- QML's Canvas 2D
                        // context doesn't implement that method (a
                        // relatively recent addition to the HTML5 Canvas
                        // spec); calling it threw and silently aborted
                        // the rest of this paint (everything after the
                        // FIRST node in the loop), which is why edges
                        // rendered correctly but no node boxes ever did.
                        ctx.rect(cx - graphCanvas.boxW / 2, cy - graphCanvas.boxH / 2,
                                 graphCanvas.boxW, graphCanvas.boxH)
                        ctx.fill()
                        ctx.globalAlpha = 1.0
                        if (hovered) {
                            ctx.strokeStyle = AppTheme.text
                            ctx.lineWidth = 2
                            ctx.stroke()
                        }
                        ctx.fillStyle = "#111111"
                        ctx.font = "11px sans-serif"
                        ctx.textAlign = "center"
                        var label = n.name
                        var maxCh = Math.floor(graphCanvas.boxW / 6.5)
                        if (label.length > maxCh) label = label.slice(0, maxCh - 1) + "…"
                        ctx.fillText(label, cx, cy + 4)
                    }
                    ctx.textAlign = "left"
                }

                function hitTest(mx, my) {
                    var nodes = callGraphPanel.graphData.nodes
                    for (var i = nodes.length - 1; i >= 0; i--) {
                        var n = nodes[i]
                        var cx = n.x * (width - boxW) + boxW / 2
                        var cy = n.y * (height - boxH) + boxH / 2
                        if (Math.abs(mx - cx) <= boxW / 2 && Math.abs(my - cy) <= boxH / 2)
                            return n
                    }
                    return null
                }

                MouseArea {
                    anchors.fill: parent
                    hoverEnabled: true
                    onPositionChanged: (mouse) => {
                        var hit = graphCanvas.hitTest(mouse.x, mouse.y)
                        if (hit === callGraphPanel.hoveredNode) return
                        callGraphPanel.hoveredNode = hit
                        callGraphPanel.hoverPos = Qt.point(mouse.x, mouse.y)
                        graphCanvas.requestPaint()
                    }
                    onExited: {
                        callGraphPanel.hoveredNode = null
                        graphCanvas.requestPaint()
                    }
                }
            }

            // Tooltip for the hovered node.
            Rectangle {
                visible: !!callGraphPanel.hoveredNode
                color: "#000000"
                opacity: 0.92
                radius: 5
                border.color: "#444444"
                border.width: 1
                width: tipCol.width + 18
                height: tipCol.height + 12
                x: Math.min(callGraphPanel.hoverPos.x + 30, callGraphPanel.width - width - 6)
                y: Math.min(callGraphPanel.hoverPos.y + 28, callGraphPanel.height - height - 6)
                z: 20

                ColumnLayout {
                    id: tipCol
                    anchors.centerIn: parent
                    spacing: 2
                    Text {
                        text: callGraphPanel.hoveredNode ? callGraphPanel.hoveredNode.name : ""
                        color: "#eeeeee"
                        font.bold: true
                        font.pixelSize: 11
                    }
                    Text {
                        text: callGraphPanel.hoveredNode
                              ? (callGraphPanel.hoveredNode.count > 0
                                 ? root.fmtNs(callGraphPanel.hoveredNode.totalNs) + " · " +
                                   callGraphPanel.hoveredNode.count + " calls"
                                 : "(ancestor frame only -- not directly measured)")
                              : ""
                        color: "#cccccc"
                        font.pixelSize: 10
                    }
                }
            }
        }
    }
}
