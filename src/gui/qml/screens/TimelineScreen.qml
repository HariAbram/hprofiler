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

    // True when at least one lane pairs with a "sync" lane on the same
    // thread (see each lane Canvas's syncOverlayLaneIndex) -- gates the
    // legend below so traces with no OpenMP/sync data (e.g. pure MPI,
    // pure GPU) don't show an explanation for an overlay that never
    // appears anywhere in them.
    readonly property bool hasSyncOverlay: {
        var lanes = TimelineModel.lanes
        var names = {}
        for (var i = 0; i < lanes.length; i++) names[lanes[i].name] = true
        for (i = 0; i < lanes.length; i++) {
            var n = lanes[i].name
            if (n.indexOf("sync/") !== 0 && n.indexOf("/thread-") > 0 &&
                names["sync/" + n.split("/")[1]]) return true
        }
        return false
    }

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
            RowLayout {
                visible: root.hasSyncOverlay
                spacing: 4
                Rectangle {
                    width: 10
                    height: 10
                    radius: 2
                    color: AppTheme.categoryColor("sync")
                    opacity: 0.8
                }
                Text {
                    text: "= blocked at a nested sync event"
                    color: AppTheme.textMuted
                    font.pixelSize: 10
                }
            }
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
                                property string laneName: modelData.name
                                property var cachedSpans: []
                                property real boundZoom: root.zoom
                                property real boundStart: root.viewStartNs

                                // Which OTHER lane (if any) holds the "sync" events
                                // nested inside this lane's own spans -- e.g.
                                // "openmp/thread-5" pairs with "sync/thread-5".
                                // Resolved once (lanes is a constant list), not
                                // per-paint: a span like omp_parallel_region times
                                // its ENTIRE call including any nested
                                // GOMP_barrier()/critical-section wait that's
                                // ALSO separately reported as its own "sync" span
                                // on the same thread -- so the same wall-clock
                                // interval gets drawn once here (as "busy") and
                                // once on the sync lane (as "waiting"), which is
                                // exactly what made a thread deep in barrier waits
                                // still look continuously busy. This lane draws
                                // that paired lane's spans as a dimmed overlay ON
                                // TOP of its own bars afterward, so the idle
                                // portion is visible without needing to
                                // cross-reference a separate row.
                                property int syncOverlayLaneIndex: {
                                    if (laneName.indexOf("/thread-") < 0) return -1
                                    if (laneName.indexOf("sync/") === 0) return -1
                                    var pairedName = "sync/" + laneName.split("/")[1]
                                    var lanes = TimelineModel.lanes
                                    for (var i = 0; i < lanes.length; i++)
                                        if (lanes[i].name === pairedName) return i
                                    return -1
                                }
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

                                // Shared by both the lane's own spans and the sync
                                // overlay below -- fills `spans` left-to-right with
                                // the same gap-aware 1px floor (see the comment this
                                // replaced): every rect gets at least 1px for
                                // visibility, but never so wide it eats the real gap
                                // before the next span in the SAME list.
                                function _paintSpans(ctx, spans, scale, colorOf, alpha) {
                                    ctx.globalAlpha = alpha
                                    for (var i = 0; i < spans.length; i++) {
                                        var sp = spans[i]
                                        var x0 = (sp.startNs - root.viewStartNs) * scale
                                        var wReal = sp.durNs * scale
                                        var w = Math.max(1, wReal)
                                        if (i + 1 < spans.length) {
                                            var nextX0 = (spans[i + 1].startNs - root.viewStartNs) * scale
                                            w = Math.max(wReal, Math.min(w, nextX0 - x0))
                                        }
                                        ctx.fillStyle = colorOf(sp)
                                        ctx.fillRect(Math.max(0, x0), 3, Math.min(w, width - x0), height - 6)
                                    }
                                    ctx.globalAlpha = 1.0
                                }

                                onPaint: {
                                    var ctx = getContext("2d")
                                    ctx.reset()
                                    var spans = TimelineModel.visibleSpans(
                                        laneIndex, root.viewStartNs, root.viewStartNs + root.visibleNs, 2000)
                                    cachedSpans = spans
                                    var scale = width / root.visibleNs
                                    _paintSpans(ctx, spans, scale, function(sp) { return sp.color }, 1.0)

                                    // Overlay the paired sync lane's spans ON TOP,
                                    // dimmed, so a barrier/critical-section wait
                                    // nested inside one of the spans just painted
                                    // above reads as visibly idle instead of being
                                    // silently absorbed into the parent's solid
                                    // "busy" color -- see syncOverlayLaneIndex's
                                    // comment for why the same time interval can
                                    // legitimately belong to both lanes at once.
                                    if (syncOverlayLaneIndex >= 0) {
                                        var syncSpans = TimelineModel.visibleSpans(
                                            syncOverlayLaneIndex, root.viewStartNs, root.viewStartNs + root.visibleNs, 2000)
                                        // Each span's OWN per-function color (sp.color,
                                        // the same field the sync lane's own bars use),
                                        // not a flat category color -- so a given
                                        // function (e.g. "omp_barrier") overlays here in
                                        // the exact same hue it renders as on the sync
                                        // lane directly below, letting the two be
                                        // visually correlated at a glance.
                                        _paintSpans(ctx, syncSpans, scale, function(sp) { return sp.color }, 0.8)
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
        //
        // The canvas is wrapped in a Flickable and sized from the
        // layout's raw numLayers/maxLayerSize (fixed per-node pixel
        // pitch, not "squeeze everything into 210px") so nodes never
        // visually overlap regardless of how wide/tall the graph is --
        // a real crowding bug seen on a live GROMACS trace where a
        // layer of ~8-10 nodes packed into one fixed-height column made
        // labels overlap and become unreadable. Scrolling (not just a
        // bigger fixed canvas) is what makes raising max_nodes from 30
        // to 60 in TimelineModel.callGraph() viable instead of just
        // moving the crowding problem to a bigger box.
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

            readonly property real boxW: 130
            readonly property real boxH: 32
            readonly property real colGap: 46
            readonly property real rowGap: 14

            property var graphData: ({nodes: [], edges: [], numLayers: 0, maxLayerSize: 0, truncated: 0})
            property var hoveredNode: null
            property real hoverViewX: 0
            property real hoverViewY: 0

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
                id: graphTitle
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

            Flickable {
                id: graphFlick
                anchors.top: graphTitle.bottom
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.bottom: parent.bottom
                anchors.margins: 4
                contentWidth: graphCanvas.width
                contentHeight: graphCanvas.height
                clip: true
                boundsBehavior: Flickable.StopAtBounds
                ScrollBar.horizontal: ScrollBar {
                    policy: graphFlick.contentWidth > graphFlick.width ? ScrollBar.AlwaysOn : ScrollBar.AsNeeded
                }
                ScrollBar.vertical: ScrollBar {
                    policy: graphFlick.contentHeight > graphFlick.height ? ScrollBar.AlwaysOn : ScrollBar.AsNeeded
                }

                Canvas {
                    id: graphCanvas
                    objectName: "callGraphCanvas"

                    // NOT named "data" -- that's QtQuick's built-in
                    // Item.data default property (holds child objects);
                    // shadowing it with a custom property of the same
                    // name is a real footgun, not just a style nit.
                    readonly property var layoutData: callGraphPanel.graphData
                    width: Math.max(layoutData.numLayers * (callGraphPanel.boxW + callGraphPanel.colGap) + callGraphPanel.colGap,
                                     graphFlick.width)
                    height: Math.max(layoutData.maxLayerSize * (callGraphPanel.boxH + callGraphPanel.rowGap) + callGraphPanel.rowGap,
                                      graphFlick.height)

                    // Raw layer/layerIndex (not the normalized x/y) give
                    // every node the SAME pixel size/spacing no matter
                    // how many layers/nodes exist -- this is what
                    // actually fixes the overlap, not just a bigger canvas.
                    function px(n) { return colGapHalf + n.layer * (callGraphPanel.boxW + callGraphPanel.colGap) + callGraphPanel.boxW / 2 }
                    function py(n) { return rowGapHalf + n.layerIndex * (callGraphPanel.boxH + callGraphPanel.rowGap) + callGraphPanel.boxH / 2 }
                    readonly property real colGapHalf: callGraphPanel.colGap / 2
                    readonly property real rowGapHalf: callGraphPanel.rowGap / 2

                    onPaint: {
                        var ctx = getContext("2d")
                        ctx.reset()
                        var nodes = layoutData.nodes
                        var edges = layoutData.edges
                        if (nodes.length === 0) return

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
                            var tx = x1 - Math.cos(ang) * (callGraphPanel.boxW / 2)
                            var ty = y1 - Math.sin(ang) * (callGraphPanel.boxH / 2)
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
                            ctx.rect(cx - callGraphPanel.boxW / 2, cy - callGraphPanel.boxH / 2,
                                     callGraphPanel.boxW, callGraphPanel.boxH)
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
                            var maxCh = Math.floor(callGraphPanel.boxW / 6.5)
                            if (label.length > maxCh) label = label.slice(0, maxCh - 1) + "…"
                            ctx.fillText(label, cx, cy + 4)
                        }
                        ctx.textAlign = "left"
                    }

                    function hitTest(mx, my) {
                        var nodes = layoutData.nodes
                        for (var i = nodes.length - 1; i >= 0; i--) {
                            var n = nodes[i]
                            var cx = px(n), cy = py(n)
                            if (Math.abs(mx - cx) <= callGraphPanel.boxW / 2 && Math.abs(my - cy) <= callGraphPanel.boxH / 2)
                                return n
                        }
                        return null
                    }

                    Connections {
                        target: callGraphPanel
                        function onGraphDataChanged() { graphCanvas.requestPaint() }
                    }

                    MouseArea {
                        anchors.fill: parent
                        hoverEnabled: true
                        onPositionChanged: (mouse) => {
                            var hit = graphCanvas.hitTest(mouse.x, mouse.y)
                            callGraphPanel.hoverViewX = mouse.x - graphFlick.contentX
                            callGraphPanel.hoverViewY = mouse.y - graphFlick.contentY
                            if (hit === callGraphPanel.hoveredNode) return
                            callGraphPanel.hoveredNode = hit
                            graphCanvas.requestPaint()
                        }
                        onExited: {
                            callGraphPanel.hoveredNode = null
                            graphCanvas.requestPaint()
                        }
                    }
                }
            }

            // Tooltip for the hovered node -- follows the cursor, clamped
            // to stay within the panel (not the scrollable canvas, which
            // can be much bigger than the visible viewport).
            Rectangle {
                visible: !!callGraphPanel.hoveredNode
                color: "#000000"
                opacity: 0.92
                radius: 5
                border.color: "#444444"
                border.width: 1
                width: tipCol.width + 18
                height: tipCol.height + 12
                x: Math.min(callGraphPanel.hoverViewX + 14, callGraphPanel.width - width - 6)
                y: Math.min(graphTitle.height + callGraphPanel.hoverViewY + 8, callGraphPanel.height - height - 6)
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
