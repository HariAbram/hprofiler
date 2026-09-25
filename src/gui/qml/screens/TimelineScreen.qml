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

    // Keyboard needs a real focused Item -- Keys only attaches to Item,
    // not Window -- and this screen is one of several StackLayout pages
    // in Main.qml that get shown/hidden (not destroyed/recreated) as the
    // user switches tabs, so focus has to be actively reclaimed every
    // time this page becomes the visible one, not just set once at
    // startup.
    focus: true
    // Covers the FIRST ever visit to this tab: an item born already-
    // visible (the current StackLayout page at the moment its Loader
    // first instantiates it) never actually transitions false->true, so
    // onVisibleChanged below -- which fires correctly on every SUBSEQUENT
    // tab switch -- can't be relied on for that initial activation.
    Component.onCompleted: jumpToSelectionIfNew()
    onVisibleChanged: {
        if (!visible) return
        forceActiveFocus()
        // Catches the common real path a Loader-based tab otherwise
        // misses entirely: select a kernel in Kernels, THEN switch to
        // Timeline (e.g. via the Inspector's "Open in Timeline" action,
        // which only calls Nav.navigateTo -- it does not re-fire
        // selectFunction, since the selection itself didn't change).
        // Connections.onSelectionChanged below can't see that jump: this
        // screen's Loader isn't even active yet while the click happens
        // on a different, already-visible tab, so nothing here is alive
        // to receive the signal.
        jumpToSelectionIfNew()
    }

    property real zoom: 1.0
    property real viewStartNs: TimelineModel.viewStartNs
    readonly property real visibleNs: TimelineModel.traceDurationNs / zoom

    property int hoverLane: -1
    property int hoverSpanIdx: -1
    property string hoverText: ""
    // How many spans share the currently cross-tab-selected (category,
    // name) -- 0 when nothing's selected or it has no occurrences here.
    property int matchCount: 0
    // (category,name) key last auto-jumped to, so revisiting this tab
    // with the SAME selection (e.g. after manually panning elsewhere and
    // tabbing back) doesn't yank the view back every time -- only an
    // actual selection change re-triggers the jump.
    property string lastJumpedKey: ""

    function jumpToSelectionIfNew() {
        if (Nav.selectedName.length === 0) { root.matchCount = 0; return }
        var key = Nav.selectedCategory + "::" + Nav.selectedName
        var matches = TimelineModel.findByName(Nav.selectedCategory, Nav.selectedName, 50)
        root.matchCount = matches.length
        if (matches.length === 0) return
        if (key === root.lastJumpedKey) return
        // Skip the auto-jump when the selection just came from clicking
        // a span on THIS screen (see the pan/zoom MouseArea's onClicked
        // below) -- root.hoverLane/hoverSpanIdx already point at it, so
        // re-jumping to the first same-named match could yank the view
        // away from the exact span the user just clicked whenever it
        // isn't chronologically first. Still records the key so a later
        // revisit doesn't jump either.
        if (root.hoverLane >= 0 && root.hoverSpanIdx >= 0) {
            var hovered = TimelineModel.spanAt(root.hoverLane, root.hoverSpanIdx)
            if (hovered.category === Nav.selectedCategory && hovered.name === Nav.selectedName) {
                root.lastJumpedKey = key
                return
            }
        }
        root.lastJumpedKey = key
        root.zoomToSpan(matches[0].laneIndex, matches[0].spanIdx)
    }

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
        flick.contentY = 0
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

    // Zooms by `factor` (>1 in, <1 out) while keeping the timestamp at
    // horizontal fraction `xFrac` (0=left edge of the lanes area, 1=right
    // edge) fixed under that same fraction afterward -- i.e. zoom toward
    // the cursor (wheel/double-click) or the canvas center (keyboard/
    // buttons, xFrac=0.5), not always toward the trace start. Captures
    // the anchor timestamp BEFORE mutating zoom, and computes the new
    // visibleNs by direct division rather than reading root.visibleNs
    // back after the write, to not depend on binding-reevaluation order.
    function zoomAtFraction(factor, xFrac) {
        var newZoom = Math.max(1.0, Math.min(256.0, zoom * factor))
        if (newZoom === zoom) return
        var nsAtAnchor = viewStartNs + xFrac * visibleNs
        var newVisibleNs = TimelineModel.traceDurationNs / newZoom
        zoom = newZoom
        viewStartNs = nsAtAnchor - xFrac * newVisibleNs
        clampViewStart()
    }

    // Zooms to and centers a specific span (double-click target) -- the
    // span fills ~20% of the new window (clamped to the normal 1-256x
    // zoom range, so a very short span just zooms as far as allowed
    // rather than producing an absurd zoom value).
    //
    // NOTE: TimelineModel.spanAt() returns startNs RELATIVE to
    // TimelineModel.viewStartNs (the hover tooltip already relies on
    // this), unlike visibleSpans()'s ABSOLUTE startNs -- the offset has
    // to be re-added here, this isn't a bug to "fix" in the model.
    function zoomToSpan(laneIndex, spanIdx) {
        var d = TimelineModel.spanAt(laneIndex, spanIdx)
        if (!d.name) return
        var absStart = TimelineModel.viewStartNs + d.startNs
        var centerNs = absStart + d.durNs / 2.0
        var targetVisibleNs = Math.max(d.durNs / 0.2, 1)
        zoom = Math.max(1.0, Math.min(256.0, TimelineModel.traceDurationNs / targetVisibleNs))
        viewStartNs = centerNs - (TimelineModel.traceDurationNs / zoom) / 2.0
        clampViewStart()
    }

    // Covers a selection click ON this screen itself (see the pan/zoom
    // MouseArea's onClicked below, which calls Nav.selectFunction) --
    // onVisibleChanged above handles the "selected elsewhere, THEN
    // switched to Timeline" path; this handles "already on Timeline,
    // selection changes right here" so matchCount/lastJumpedKey stay
    // correct without needing a tab switch to refresh them.
    Connections {
        target: Nav
        function onSelectionChanged() { root.jumpToSelectionIfNew() }
    }

    Keys.onPressed: (event) => {
        switch (event.key) {
        case Qt.Key_Left:  viewStartNs -= visibleNs * 0.1; clampViewStart(); break
        case Qt.Key_Right: viewStartNs += visibleNs * 0.1; clampViewStart(); break
        case Qt.Key_Up:
            flick.contentY = Math.max(0, flick.contentY - rowHeight * 3)
            break
        case Qt.Key_Down:
            flick.contentY = Math.min(Math.max(0, flick.contentHeight - flick.height),
                                       flick.contentY + rowHeight * 3)
            break
        case Qt.Key_Plus: case Qt.Key_Equal:
            zoomAtFraction(1.25, 0.5); break
        case Qt.Key_Minus: case Qt.Key_Underscore:
            zoomAtFraction(0.8, 0.5); break
        case Qt.Key_Home:
            viewStartNs = TimelineModel.viewStartNs; clampViewStart(); break
        case Qt.Key_End:
            viewStartNs = TimelineModel.viewStartNs + TimelineModel.traceDurationNs - visibleNs
            clampViewStart()
            break
        case Qt.Key_0:
            resetView(); break
        default:
            return
        }
        event.accepted = true
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: AppTheme.spacingXs

        // ── Status / controls row ───────────────────────────────────────
        RowLayout {
            Layout.fillWidth: true
            Layout.preferredHeight: 24
            Layout.maximumHeight: 24
            spacing: AppTheme.spacingMd

            Text {
                text: "zoom " + root.zoom.toFixed(1) + "×   offset " +
                      root.fmtNs(root.viewStartNs - TimelineModel.viewStartNs) +
                      "   window " + root.fmtNs(root.visibleNs)
                color: AppTheme.textMuted
                font.pixelSize: AppTheme.typeLabel
            }
            Item { Layout.fillWidth: true }
            Text {
                visible: root.hoverText.length > 0
                text: root.hoverText
                color: AppTheme.text
                font.pixelSize: AppTheme.typeLabel
                font.bold: true
            }
            Text {
                visible: root.matchCount > 1
                text: root.matchCount + (root.matchCount >= 50 ? "+" : "") + " matches"
                color: AppTheme.accent
                font.pixelSize: AppTheme.typeLabel
            }
            Item { Layout.fillWidth: true }
            RowLayout {
                visible: root.hasSyncOverlay
                spacing: AppTheme.spacingXs
                Rectangle {
                    width: 10
                    height: 10
                    radius: AppTheme.radiusSmall / 2
                    color: AppTheme.categoryColor("sync")
                    opacity: 0.8
                }
                Text {
                    text: "= blocked at a nested sync event"
                    color: AppTheme.textMuted
                    font.pixelSize: AppTheme.typeCaption
                }
            }
            RowLayout {
                spacing: 2
                ToolButton {
                    text: "−"
                    implicitWidth: AppTheme.iconButtonWidth
                    implicitHeight: AppTheme.buttonHeight
                    onClicked: { root.zoomAtFraction(0.8, 0.5); root.forceActiveFocus() }
                    ToolTip.visible: hovered
                    ToolTip.text: "Zoom out (-)"
                }
                ToolButton {
                    text: "+"
                    implicitWidth: AppTheme.iconButtonWidth
                    implicitHeight: AppTheme.buttonHeight
                    onClicked: { root.zoomAtFraction(1.25, 0.5); root.forceActiveFocus() }
                    ToolTip.visible: hovered
                    ToolTip.text: "Zoom in (+)"
                }
                ToolButton {
                    text: "Fit"
                    implicitHeight: AppTheme.buttonHeight
                    onClicked: { root.resetView(); root.forceActiveFocus() }
                    ToolTip.visible: hovered
                    ToolTip.text: "Fit entire trace"
                }
                ToolButton {
                    text: "Reset"
                    implicitHeight: AppTheme.buttonHeight
                    onClicked: { root.resetView(); root.forceActiveFocus() }
                    ToolTip.visible: hovered
                    ToolTip.text: "Reset view (0)"
                }
            }
            Text {
                text: "wheel: zoom@cursor · drag: pan · dbl-click event: zoom to it · arrows/+/-/Home/End/0: keyboard"
                color: AppTheme.textMuted
                font.pixelSize: AppTheme.typeCaption
            }
        }

        // ── Lanes ────────────────────────────────────────────────────────
        Rectangle {
            Layout.fillWidth: true
            Layout.fillHeight: true
            color: AppTheme.surface
            border.color: AppTheme.panelBorder
            border.width: 1
            radius: AppTheme.radiusPanel
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
                                font.pixelSize: AppTheme.typeLabel
                                elide: Text.ElideRight
                                leftPadding: AppTheme.spacingSm
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
                                    var viewEndNs = root.viewStartNs + root.visibleNs
                                    for (var i = 0; i < spans.length; i++) {
                                        var sp = spans[i]
                                        // Defense-in-depth: never paint a span that
                                        // doesn't truly overlap the visible window,
                                        // regardless of what the model handed back --
                                        // guarantees this class of bug (an incorrectly
                                        // windowed Python query smearing off-screen
                                        // spans across the canvas) can't resurface here
                                        // even if visibleSpans()'s own windowing ever
                                        // regresses.
                                        if (sp.startNs + sp.durNs <= root.viewStartNs || sp.startNs >= viewEndNs)
                                            continue
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
                // Single click ON a span selects it for cross-tab
                // navigation (Nav.selectFunction/selectThread), distinct
                // from the drag-to-pan handled above and the
                // double-click-to-zoom handled below -- guarded by the
                // same movement threshold pan itself doesn't use, since a
                // plain click-release fires `clicked` in Qt Quick
                // regardless of how far the mouse moved in between
                // (MouseArea.clicked isn't drag-aware unless drag.target
                // is set, which this pan implementation deliberately
                // doesn't use -- see onPositionChanged above).
                onClicked: (mouse) => {
                    if (Math.abs(mouse.x - dragStartX) > 4) return
                    if (root.hoverSpanIdx < 0) return
                    var d = TimelineModel.spanAt(root.hoverLane, root.hoverSpanIdx)
                    if (!d.name) return
                    Nav.selectFunction(d.category, d.name)
                    Nav.selectThread(d.pid, d.tid)
                }
                // Double-click ON a span (root.hoverSpanIdx is kept live by
                // each lane's own hover MouseArea, which passes clicks
                // through via acceptedButtons: Qt.NoButton) zooms to and
                // centers that event; double-click on empty canvas falls
                // back to the original reset-view behavior.
                onDoubleClicked: {
                    if (root.hoverSpanIdx >= 0)
                        root.zoomToSpan(root.hoverLane, root.hoverSpanIdx)
                    else
                        root.resetView()
                }
                onWheel: (wheel) => {
                    var factor = wheel.angleDelta.y > 0 ? 1.25 : 0.8
                    root.zoomAtFraction(factor, wheel.x / width)
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
                radius: AppTheme.radiusSmall
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
                    radius: AppTheme.radiusSmall
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
    }
}
