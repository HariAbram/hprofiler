import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import Hprofiler 1.0
import "../components"

// The Timeline: the whole heterogeneous run, one row per lane under a
// hierarchy (node / process or rank / thread -> runtime, device -> stream),
// painted by Python (TimelineModel + LanePainter: timeline_lod decides what
// is an exact interval and what an honest per-column density). Over it:
// dependency edges and the critical path (OverlayPainter), a synchronized
// cursor, the selection's details (right) and the linked summaries
// (bottom). Wheel: zoom at the cursor; drag: pan; shift-drag: select a
// range; Shift+wheel: scroll rows; double-click: zoom to an event (empty:
// back to the whole run).
Item {
    id: root
    readonly property real rowHeight: 26
    readonly property real labelWidth: 190

    // Keyboard needs a real focused Item -- Keys only attaches to Item, not
    // Window -- and this page is shown/hidden (not recreated) by the tab
    // StackLayout, so focus is reclaimed every time it becomes visible.
    focus: true
    Component.onCompleted: {
        var saved = TimelineModel.restoredView
        if (saved && saved.zoom) {
            zoom = saved.zoom
            viewStartNs = saved.viewStartNs
        }
        jumpToSelectionIfNew()
        applyFocusIfNew()
        TimelineModel.pushView(viewStartNs, viewStartNs + visibleNs)
        TimelineModel.maybeStartAnalysis()
    }
    onZoomChanged: { TimelineModel.noteView(zoom, viewStartNs); historyTimer.restart() }
    onViewStartNsChanged: { TimelineModel.noteView(zoom, viewStartNs); historyTimer.restart() }
    onVisibleChanged: {
        if (!visible) return
        forceActiveFocus()
        jumpToSelectionIfNew()
        applyFocusIfNew()
    }

    property real zoom: 1.0
    property real viewStartNs: TimelineModel.viewStartNs
    readonly property real visibleNs: TimelineModel.traceDurationNs / zoom
    // down to a 200 ns window (it was a fixed 256x: microsecond kernels in a
    // seconds-long run could not be reached)
    readonly property real maxZoom: Math.max(256.0, TimelineModel.traceDurationNs / 200.0)

    property int hoverLane: -1
    // A store event id, (shard << 38) | row -- past 2^31 for every span of a
    // disk store, so a real (double, exact to 2^53), never an int.
    property real hoverSpanIdx: -1
    property string hoverText: ""
    property real hoverX: 0
    property real hoverY: 0
    property real cursorNs: -1
    property int matchCount: 0
    property string lastJumpedKey: ""
    property bool _restoringHistory: false

    // the selected event / edge, shown in the details panel
    readonly property var sel: TimelineModel.selectedEvent
    property var selDetails: ({})
    property var selEdge: ({})
    Connections {
        target: TimelineModel
        function onEventSelectionChanged() { root.refreshDetails() }
        function onAnalysisChanged() { root.refreshDetails() }
    }
    function refreshDetails() {
        var s = TimelineModel.selectedEvent
        selEdge = s.edge >= 0 ? TimelineModel.edgeDetails(s.edge) : ({})
        selDetails = (s.spanIdx >= 0 && s.edge < 0) ? TimelineModel.eventDetails(s.laneIndex, s.spanIdx) : ({})
    }

    // Debounced view history: a gesture (wheel, drag) is one Back step.
    Timer {
        id: historyTimer
        interval: 700
        onTriggered: if (!root._restoringHistory) TimelineModel.pushView(root.viewStartNs, root.viewStartNs + root.visibleNs)
    }

    property int lastFocusSerial: 0
    function applyFocusIfNew() {
        var f = Nav.focusRange
        if (f.serial === undefined || f.serial <= root.lastFocusSerial) return
        root.lastFocusSerial = f.serial
        root.zoomToRange(f.startNs, f.endNs)
    }
    function setView(startNs, endNs) {
        var span = Math.max(endNs - startNs, 1)
        zoom = Math.max(1.0, Math.min(root.maxZoom, TimelineModel.traceDurationNs / span))
        viewStartNs = startNs
        clampViewStart()
    }
    function zoomToRange(startNs, endNs) {
        var span = Math.max(endNs - startNs, 1)
        var pad = span * 0.1
        zoom = Math.max(1.0, Math.min(root.maxZoom, TimelineModel.traceDurationNs / (span + 2 * pad)))
        viewStartNs = startNs - pad
        clampViewStart()
        TimelineModel.pushView(viewStartNs, viewStartNs + visibleNs)
    }
    function zoomToSelection() {
        var r = Nav.selectedTimeRange
        if (r.startNs !== undefined && r.endNs > r.startNs) { zoomToRange(r.startNs, r.endNs); return }
        if (root.sel.spanIdx >= 0) zoomToSpan(root.sel.laneIndex, root.sel.spanIdx)
    }
    function historyStep(back) {
        var v = back ? TimelineModel.historyBack() : TimelineModel.historyForward()
        if (v.startNs === undefined) return
        _restoringHistory = true
        setView(v.startNs, v.endNs)
        _restoringHistory = false
    }

    function jumpToSelectionIfNew() {
        if (Nav.selectedName.length === 0) { root.matchCount = 0; return }
        var key = Nav.selectedCategory + "::" + Nav.selectedName
        var matches = TimelineModel.findByName(Nav.selectedCategory, Nav.selectedName, 50)
        root.matchCount = matches.length
        if (matches.length === 0) return
        if (key === root.lastJumpedKey) return
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

    // True when some thread lane pairs with a sync lane (its waits are cut
    // out of its spans) -- gates the waiting legend.
    readonly property bool hasSyncOverlay: {
        var rows = TimelineModel.rows
        for (var i = 0; i < rows.length; i++)
            if (rows[i].kind === "lane" && rows[i].syncLane !== undefined && rows[i].syncLane >= 0) return true
        return false
    }

    function resetView() {
        zoom = 1.0
        viewStartNs = TimelineModel.viewStartNs
        flick.contentY = 0
        TimelineModel.pushView(viewStartNs, viewStartNs + visibleNs)
    }
    function clampViewStart() {
        var maxStart = TimelineModel.viewStartNs + TimelineModel.traceDurationNs - visibleNs
        var minStart = TimelineModel.viewStartNs
        if (viewStartNs > maxStart) viewStartNs = Math.max(minStart, maxStart)
        if (viewStartNs < minStart) viewStartNs = minStart
    }
    function fmtNs(ns) { return Format.duration(ns) }

    // Zooms by `factor` keeping the timestamp at horizontal fraction xFrac
    // fixed (zoom toward the cursor, or the centre for keys / buttons).
    function zoomAtFraction(factor, xFrac) {
        var newZoom = Math.max(1.0, Math.min(root.maxZoom, zoom * factor))
        if (newZoom === zoom) return
        var nsAtAnchor = viewStartNs + xFrac * visibleNs
        var newVisibleNs = TimelineModel.traceDurationNs / newZoom
        zoom = newZoom
        viewStartNs = nsAtAnchor - xFrac * newVisibleNs
        clampViewStart()
    }

    // Zooms to and centres one event (it fills ~20% of the window).
    // TimelineModel.spanAt() returns startNs RELATIVE to viewStartNs.
    function zoomToSpan(laneIndex, spanIdx) {
        var d = TimelineModel.spanAt(laneIndex, spanIdx)
        if (!d.name) return
        var absStart = TimelineModel.viewStartNs + d.startNs
        var centerNs = absStart + d.durNs / 2.0
        var targetVisibleNs = Math.max(d.durNs / 0.2, 200)
        zoom = Math.max(1.0, Math.min(root.maxZoom, TimelineModel.traceDurationNs / targetVisibleNs))
        viewStartNs = centerNs - (TimelineModel.traceDurationNs / zoom) / 2.0
        clampViewStart()
        TimelineModel.pushView(viewStartNs, viewStartNs + visibleNs)
    }
    function revealLane(laneIndex) {
        var rowPos = TimelineModel.rowIndexForLane(laneIndex)
        if (rowPos >= 0) flick.positionViewAtIndex(rowPos, ListView.Contain)
    }
    // Go to one event: reveal its row, keep it in view, select it.
    function focusEvent(laneIndex, spanIdx, startNs) {
        if (laneIndex < 0) return
        root.revealLane(laneIndex)
        if (startNs !== undefined && startNs > 0 && (startNs < viewStartNs || startNs > viewStartNs + visibleNs)) {
            viewStartNs = startNs - visibleNs / 2.0
            clampViewStart()
            TimelineModel.pushView(viewStartNs, viewStartNs + visibleNs)
        }
        TimelineModel.selectEvent(laneIndex, spanIdx)
    }

    // Search "next/previous match": pans (keeps the zoom) to centre the
    // match, scrolls its row into view and points the tooltip at it.
    function jumpToMatch(match) {
        if (!match || match.laneIndex === undefined) return
        viewStartNs = match.startNs - visibleNs / 2.0
        clampViewStart()
        TimelineModel.pushView(viewStartNs, viewStartNs + visibleNs)
        var rowPos = TimelineModel.rowIndexForLane(match.laneIndex)
        if (rowPos >= 0) flick.positionViewAtIndex(rowPos, ListView.Contain)
        var detail = TimelineModel.spanAt(match.laneIndex, match.spanIdx)
        if (detail.name) {
            root.hoverLane = match.laneIndex
            root.hoverSpanIdx = match.spanIdx
            root.hoverText = detail.name + "  @" + root.fmtNs(detail.startNs) + "  dur " + root.fmtNs(detail.durNs)
            var laneW = flick.width - root.labelWidth
            var row = rowPos >= 0 ? flick.itemAtIndex(rowPos) : null
            var p = row ? row.mapToItem(root, 0, row.height / 2) : flick.mapToItem(root, 0, flick.height / 2)
            root.hoverX = p.x + root.labelWidth + Math.max(0, Math.min(1, (match.startNs - viewStartNs) / visibleNs)) * laneW
            root.hoverY = p.y
        }
        overlay.update()
    }

    function gotoTransition(direction) {
        var ref = root.viewStartNs + root.visibleNs / 2
        var t = TimelineModel.nextTransition(ref, direction)
        if (t.ns === undefined) return
        root.viewStartNs = t.ns - root.visibleNs / 2
        root.clampViewStart()
        TimelineModel.pushView(viewStartNs, viewStartNs + visibleNs)
        root.revealLane(t.toLane)
        transitionLabel.text = "switch " + (t.index + 1) + "/" + t.count + ": " + t.fromLabel + " → " + t.toLabel
                               + "  (" + t["switch"] + ", " + t.kind + ", " + t.tier + ")"
    }

    function runJump(text) {
        var r = TimelineModel.resolveJump(text)
        jumpField.status = r.message || ""
        if (r.kind === "time") {
            viewStartNs = r.ns - visibleNs / 2
            clampViewStart()
            root.cursorNs = r.ns
            TimelineModel.pushView(viewStartNs, viewStartNs + visibleNs)
        } else if (r.kind === "range") {
            zoomToRange(r.startNs, r.endNs)
        } else if (r.kind === "lane") {
            if (r.rowPos >= 0) flick.positionViewAtIndex(r.rowPos, ListView.Beginning)
            TimelineModel.setHighlight({"lanes": r.laneIndexes, "label": r.label})
        } else if (r.kind === "event") {
            zoomToSpan(r.laneIndex, r.spanIdx)
            root.revealLane(r.laneIndex)
            TimelineModel.selectEvent(r.laneIndex, r.spanIdx)
        }
    }

    Connections {
        target: Nav
        function onSelectionChanged() { root.jumpToSelectionIfNew() }
        function onFocusRangeChanged() { if (root.visible) root.applyFocusIfNew() }
    }

    Keys.onPressed: (event) => {
        var alt = (event.modifiers & Qt.AltModifier) !== 0
        var shift = (event.modifiers & Qt.ShiftModifier) !== 0
        switch (event.key) {
        case Qt.Key_Left:
            if (alt) { historyStep(true); break }
            viewStartNs -= visibleNs * 0.1; clampViewStart(); break
        case Qt.Key_Right:
            if (alt) { historyStep(false); break }
            viewStartNs += visibleNs * 0.1; clampViewStart(); break
        case Qt.Key_Up:
            flick.contentY = Math.max(0, flick.contentY - rowHeight * 3)
            break
        case Qt.Key_Down:
            flick.contentY = Math.min(Math.max(0, flick.contentHeight - flick.height), flick.contentY + rowHeight * 3)
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
        case Qt.Key_Z:
            zoomToSelection(); break
        case Qt.Key_N:
            gotoTransition(shift ? -1 : 1); break
        case Qt.Key_C:
            TimelineModel.setOverlay({"criticalPath": !TimelineModel.overlay.criticalPath}); break
        case Qt.Key_G:
            if (event.modifiers & Qt.ControlModifier) { jumpField.forceActiveFocus(); jumpField.selectAll(); break }
            return
        case Qt.Key_Escape:
            TimelineModel.clearEventSelection(); TimelineModel.clearHighlight(); break
        default:
            return
        }
        event.accepted = true
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: AppTheme.spacingXs

        // ── Status / controls row ───────────────────────────────────────
        // Neither toolbar row may be wider than the screen: a ColumnLayout
        // widens EVERY child to its widest child's minimum (minimumWidth 0,
        // eliding texts, hover info in a floating tooltip).
        RowLayout {
            objectName: "timelineStatusRow"
            Layout.fillWidth: true
            Layout.minimumWidth: 0
            Layout.preferredHeight: 24
            Layout.maximumHeight: 24
            spacing: AppTheme.spacingXs

            ToolButton {
                objectName: "timelineBackButton"
                text: "◀"
                enabled: TimelineModel.canGoBack
                implicitWidth: AppTheme.iconButtonWidth
                implicitHeight: AppTheme.buttonHeight
                onClicked: { root.historyStep(true); root.forceActiveFocus() }
                ToolTip.visible: hovered
                ToolTip.text: "Back to the previous view (Alt+←)"
                Accessible.name: "Back"
            }
            ToolButton {
                objectName: "timelineForwardButton"
                text: "▶"
                enabled: TimelineModel.canGoForward
                implicitWidth: AppTheme.iconButtonWidth
                implicitHeight: AppTheme.buttonHeight
                onClicked: { root.historyStep(false); root.forceActiveFocus() }
                ToolTip.visible: hovered
                ToolTip.text: "Forward (Alt+→)"
                Accessible.name: "Forward"
            }
            Text {
                Layout.fillWidth: true
                Layout.minimumWidth: 0
                Layout.preferredWidth: Math.ceil(implicitWidth) + 1
                Layout.maximumWidth: Math.ceil(implicitWidth) + 1
                text: "zoom " + root.zoom.toFixed(1) + "×   offset " +
                      root.fmtNs(root.viewStartNs - TimelineModel.viewStartNs) + "   window " + root.fmtNs(root.visibleNs)
                color: AppTheme.textMuted
                font.pixelSize: AppTheme.typeLabel
                elide: Text.ElideRight
            }
            TextField {
                id: jumpField
                objectName: "timelineJumpField"
                property string status: ""
                Layout.fillWidth: true
                Layout.preferredWidth: 104
                Layout.maximumWidth: 140
                Layout.minimumWidth: 36
                implicitHeight: AppTheme.buttonHeight
                placeholderText: "Jump: 12.5ms, rank 3, T2, kernel…"
                font.pixelSize: AppTheme.typeCaption
                onAccepted: { root.runJump(text); root.forceActiveFocus() }
                ToolTip.visible: (hovered || activeFocus) && status.length > 0
                ToolTip.text: status
                Accessible.name: "Jump to a time, rank, thread, stream or event"
            }
            Text {
                visible: root.matchCount > 1
                text: root.matchCount + (root.matchCount >= 50 ? "+" : "") + " matches"
                color: AppTheme.accent
                font.pixelSize: AppTheme.typeLabel
            }
            Item { Layout.fillWidth: true }
            TimelineFilterBar {
                id: filterBar
                objectName: "timelineFilterBar"
            }
            TimelineViewControls {
                id: viewControls
                objectName: "timelineViewControls"
            }
            TimelineSearchBar {
                id: searchBar
                objectName: "timelineSearchBar"
                onMatchJumped: (match) => root.jumpToMatch(match)
            }
            RowLayout {
                spacing: 2
                ToolButton {
                    objectName: "timelineZoomOutButton"
                    text: "−"
                    implicitWidth: AppTheme.iconButtonWidth
                    implicitHeight: AppTheme.buttonHeight
                    onClicked: { root.zoomAtFraction(0.8, 0.5); root.forceActiveFocus() }
                    ToolTip.visible: hovered
                    ToolTip.text: "Zoom out (-)"
                    Accessible.name: "Zoom out"
                    Accessible.description: "Zooms the Timeline out, centered on the current view"
                }
                ToolButton {
                    objectName: "timelineZoomInButton"
                    text: "+"
                    implicitWidth: AppTheme.iconButtonWidth
                    implicitHeight: AppTheme.buttonHeight
                    onClicked: { root.zoomAtFraction(1.25, 0.5); root.forceActiveFocus() }
                    ToolTip.visible: hovered
                    ToolTip.text: "Zoom in (+)"
                    Accessible.name: "Zoom in"
                    Accessible.description: "Zooms the Timeline in, centered on the current view"
                }
                ToolButton {
                    objectName: "timelineZoomSelectionButton"
                    text: "⤢"
                    implicitWidth: AppTheme.iconButtonWidth
                    implicitHeight: AppTheme.buttonHeight
                    enabled: Nav.selectedTimeRange.startNs !== undefined || root.sel.spanIdx >= 0
                    onClicked: { root.zoomToSelection(); root.forceActiveFocus() }
                    ToolTip.visible: hovered
                    ToolTip.text: "Zoom to the selected range or event (Z)"
                    Accessible.name: "Zoom to selection"
                }
                ToolButton {
                    objectName: "timelineFitButton"
                    text: "Fit"
                    implicitHeight: AppTheme.buttonHeight
                    onClicked: { root.resetView(); root.forceActiveFocus() }
                    ToolTip.visible: hovered
                    ToolTip.text: "Fit entire trace"
                    Accessible.name: "Fit entire trace"
                    Accessible.description: "Zooms and pans so the whole trace duration is visible"
                }
                ToolButton {
                    objectName: "timelineResetButton"
                    text: "Reset"
                    implicitHeight: AppTheme.buttonHeight
                    onClicked: { root.resetView(); root.forceActiveFocus() }
                    ToolTip.visible: hovered
                    ToolTip.text: "Reset view (0)"
                    Accessible.name: "Reset view"
                    Accessible.description: "Resets zoom and pan to the full trace view"
                }
            }
        }

        // ── Bookmarks / overlays / legend / help row ────────────────────
        RowLayout {
            objectName: "timelineSecondRow"
            Layout.fillWidth: true
            Layout.minimumWidth: 0
            Layout.preferredHeight: 22
            Layout.maximumHeight: 22
            spacing: AppTheme.spacingSm

            ToolButton {
                objectName: "timelineAddBookmarkButton"
                text: "+ Bookmark"
                implicitHeight: AppTheme.buttonHeight
                onClicked: TimelineModel.addBookmark(root.viewStartNs + root.visibleNs / 2, "")
                ToolTip.visible: hovered
                ToolTip.text: "Bookmark the center of the current view"
            }
            ToolButton {
                objectName: "timelineAddRangeButton"
                text: "+ Range"
                implicitHeight: AppTheme.buttonHeight
                enabled: Nav.selectedTimeRange.startNs !== undefined
                onClicked: {
                    var r = Nav.selectedTimeRange
                    TimelineModel.addNamedRange(r.startNs, r.endNs, "")
                }
                ToolTip.visible: hovered
                ToolTip.text: enabled ? "Save the selected time range" : "Shift-drag on the lanes area to select a range first"
            }
            Repeater {
                model: TimelineModel.bookmarks
                delegate: RowLayout {
                    spacing: 2
                    Rectangle { width: 8; height: 8; radius: 4; color: AppTheme.warningColor }
                    Text {
                        text: modelData.name
                        color: AppTheme.textMuted
                        font.pixelSize: AppTheme.typeCaption
                        Accessible.role: Accessible.Button
                        Accessible.name: "Bookmark: " + modelData.name
                        Accessible.description: "Click to jump the view to this bookmark; right-click to remove it"
                        Accessible.onPressAction: {
                            root.viewStartNs = modelData.ns - root.visibleNs / 2
                            root.clampViewStart()
                        }
                        MouseArea {
                            anchors.fill: parent
                            acceptedButtons: Qt.LeftButton | Qt.RightButton
                            onClicked: (mouse) => {
                                if (mouse.button === Qt.RightButton) TimelineModel.removeBookmark(modelData.id)
                                else { root.viewStartNs = modelData.ns - root.visibleNs / 2; root.clampViewStart() }
                            }
                        }
                    }
                }
            }
            TimelineOverlayControls {
                id: overlayControls
                objectName: "timelineOverlayControls"
                onTransitionRequested: (dir) => root.gotoTransition(dir)
            }
            ToolButton {
                objectName: "timelineExportButton"
                text: "Export"
                implicitHeight: AppTheme.buttonHeight
                onClicked: exportPopup.open()
                ToolTip.visible: hovered
                ToolTip.text: "Save the visible window as a PNG or SVG figure with axis and legend"
                Accessible.name: "Export figure"
            }
            Rectangle {
                // the active highlight from a summary view, with a way out
                objectName: "timelineHighlightChip"
                visible: TimelineModel.highlight.label !== undefined
                implicitWidth: chipText.implicitWidth + 26
                implicitHeight: 18
                radius: 9
                color: AppTheme.control
                border.color: AppTheme.accent
                Text {
                    id: chipText
                    x: 8
                    anchors.verticalCenter: parent.verticalCenter
                    text: "highlight: " + (TimelineModel.highlight.label || "")
                    color: AppTheme.text
                    font.pixelSize: AppTheme.typeCaption
                }
                Text {
                    anchors.right: parent.right
                    anchors.rightMargin: 6
                    anchors.verticalCenter: parent.verticalCenter
                    text: "✕"
                    color: AppTheme.textMuted
                    font.pixelSize: AppTheme.typeCaption
                }
                MouseArea {
                    anchors.fill: parent
                    onClicked: TimelineModel.clearHighlight()
                    cursorShape: Qt.PointingHandCursor
                }
            }
            Text {
                id: transitionLabel
                objectName: "timelineTransitionLabel"
                visible: TimelineModel.overlay.criticalPath && text.length > 0
                Layout.fillWidth: true
                Layout.minimumWidth: 0
                Layout.preferredWidth: Math.ceil(implicitWidth) + 1
                Layout.maximumWidth: Math.ceil(implicitWidth) + 1
                color: AppTheme.criticalColor
                font.pixelSize: AppTheme.typeCaption
                elide: Text.ElideRight
            }
            Item { Layout.fillWidth: true }
            BucketLegend {
                visible: TimelineModel.colorMode === "bucket"
            }
            RowLayout {
                visible: root.hasSyncOverlay
                spacing: AppTheme.spacingXs
                Rectangle {
                    objectName: "timelineWaitLegendSwatch"
                    width: 10
                    height: 10
                    radius: AppTheme.radiusSmall / 2
                    color: "transparent"
                    border.width: 1
                    border.color: AppTheme.bucketColor("Idle", AppTheme.dark)
                    Rectangle {
                        anchors.fill: parent
                        anchors.margins: 1
                        color: AppTheme.bucketColor("Idle", AppTheme.dark)
                        opacity: 0.28
                    }
                    Rectangle {
                        anchors.left: parent.left
                        anchors.right: parent.right
                        anchors.bottom: parent.bottom
                        height: 2
                        color: AppTheme.categoryColor("sync", AppTheme.dark)
                    }
                }
                Text {
                    text: "hatched = waiting, not working"
                    color: AppTheme.textMuted
                    font.pixelSize: AppTheme.typeCaption
                }
            }
            Text {
                objectName: "timelineInputHint"
                Layout.fillWidth: true
                Layout.minimumWidth: 0
                Layout.preferredWidth: Math.ceil(implicitWidth) + 1
                Layout.maximumWidth: Math.ceil(implicitWidth) + 1
                text: "wheel: zoom@cursor · drag: pan · shift-wheel: rows · shift-drag: range · dbl-click: zoom to event / empty: whole run · bar height = busy share where events are < 1 px"
                color: AppTheme.textMuted
                font.pixelSize: AppTheme.typeCaption
                elide: Text.ElideRight
            }
        }

        // ── Timeline body (one time axis) + details ─────────────────────
        // The minimap, ruler, score track and lanes share ONE column, so
        // the details panel beside them can never shift the axis off the
        // lanes.
        RowLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: AppTheme.spacingXs

        ColumnLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: AppTheme.spacingXs

        // ── Minimap: the whole run, the visible window ─────────────────
        MinimapPainter {
            id: minimap
            objectName: "timelineMinimap"
            Layout.fillWidth: true
            Layout.preferredHeight: 16
            // the lanes start 1 px in (the row list's inset): the same here,
            // so one instant is one x in every strip
            Layout.leftMargin: root.labelWidth + 1
            Layout.rightMargin: 13
            model: TimelineModel
            viewStartNs: root.viewStartNs
            viewEndNs: root.viewStartNs + root.visibleNs
            revision: TimelineModel.paintRevision
            MouseArea {
                anchors.fill: parent
                property bool moved: false
                function centerAt(x) {
                    var ns = TimelineModel.viewStartNs + (x / width) * TimelineModel.traceDurationNs
                    root.viewStartNs = ns - root.visibleNs / 2
                    root.clampViewStart()
                }
                onPressed: (mouse) => { moved = false; centerAt(mouse.x) }
                onPositionChanged: (mouse) => { if (pressed) { moved = true; centerAt(mouse.x) } }
                onDoubleClicked: root.resetView()
                ToolTip.visible: containsMouse
                ToolTip.text: "Overview of the whole run: click or drag to move the window; double-click: whole run"
                hoverEnabled: true
            }
        }

        // ── Time ruler ───────────────────────────────────────────────────
        TimeRuler {
            id: timeRuler
            objectName: "timelineRuler"
            Layout.fillWidth: true
            Layout.preferredHeight: 26
            Layout.maximumHeight: 26
            Layout.leftMargin: root.labelWidth + 1
            Layout.rightMargin: 13
            viewStartNs: root.viewStartNs
            visibleNs: root.visibleNs
            bookmarks: TimelineModel.bookmarks
            namedRanges: TimelineModel.namedRanges
            selectedRange: Nav.selectedTimeRange
            onBookmarkClicked: (ns) => { root.viewStartNs = ns - root.visibleNs / 2; root.clampViewStart() }
            onBookmarkRemoveRequested: (id) => TimelineModel.removeBookmark(id)
        }

        // ── Span-level score track (critical-path mode) ─────────────────
        Item {
            Layout.fillWidth: true
            Layout.preferredHeight: 22
            visible: TimelineModel.overlay.scoreTrack && TimelineModel.analysisState === "ready"
            Text {
                width: root.labelWidth - 6
                anchors.verticalCenter: parent.verticalCenter
                x: 6
                text: "span-level score"
                color: AppTheme.overlapColor
                font.pixelSize: AppTheme.typeCaption
                elide: Text.ElideRight
                ToolTip.visible: scoreHover.containsMouse
                ToolTip.text: "The span-level search adds each hop's whole duration: red = wall time it counts more than once"
                MouseArea { id: scoreHover; anchors.fill: parent; hoverEnabled: true }
            }
            ScoreTrackPainter {
                objectName: "timelineScoreTrack"
                x: root.labelWidth + 1
                width: parent.width - root.labelWidth - 14
                height: parent.height
                model: TimelineModel
                viewStartNs: root.viewStartNs
                viewEndNs: root.viewStartNs + root.visibleNs
                revision: TimelineModel.paintRevision
            }
        }

        Rectangle {
            id: lanesFrame
            Layout.fillWidth: true
            Layout.fillHeight: true
            color: AppTheme.surface
            border.color: AppTheme.panelBorder
            border.width: 1
            radius: AppTheme.radiusPanel
            clip: true

            // ListView, not Flickable+Repeater: only rows near the viewport
            // are instantiated. Iterates `rows` (hierarchy headers + lanes);
            // a lane row keeps its ORIGINAL laneIndex.
            ListView {
                id: flick
                objectName: "timelineFlick"
                anchors.fill: parent
                anchors.margins: 1
                anchors.rightMargin: 13
                anchors.bottomMargin: 13
                boundsBehavior: Flickable.StopAtBounds
                model: TimelineModel.rows

                ScrollBar.vertical: ScrollBar {
                    policy: TimelineModel.rows.length * root.rowHeight > flick.height
                            ? ScrollBar.AlwaysOn : ScrollBar.AlwaysOff
                }

                delegate: Item {
                    id: rowItem
                    width: flick.width
                    height: root.rowHeight
                    property var rowData: modelData
                    property int rowPos: index
                    readonly property bool isLane: rowData.kind === "lane"
                    readonly property int level: rowData.level !== undefined ? rowData.level : 0

                    Rectangle {
                        // header rows: a faint band so the hierarchy reads
                        anchors.fill: parent
                        visible: !rowItem.isLane
                        color: AppTheme.background
                        opacity: groupMouse.containsMouse ? 1.0 : 0.6
                    }

                    // ── label (stays put while panning in time) ─────────
                    Item {
                        id: labelBox
                        width: root.labelWidth
                        height: parent.height
                        Text {
                            id: chevron
                            visible: !rowItem.isLane
                            x: 4 + 12 * rowItem.level
                            anchors.verticalCenter: parent.verticalCenter
                            text: rowItem.rowData.collapsed ? "▸" : "▾"
                            color: AppTheme.textMuted
                            font.pixelSize: AppTheme.typeLabel
                        }
                        Text {
                            id: laneLabel
                            objectName: rowItem.isLane ? "laneLabel_" + rowItem.rowData.laneIndex
                                                       : "groupLabel_" + rowItem.rowData.groupId
                            x: 16 + 12 * rowItem.level
                            width: root.labelWidth - x - 2
                            height: parent.height
                            verticalAlignment: Text.AlignVCenter
                            text: rowItem.isLane
                                  ? (rowItem.rowData.hierLabel || rowItem.rowData.label) + "  (" +
                                    (rowItem.rowData.filteredCount !== rowItem.rowData.count
                                     ? rowItem.rowData.filteredCount + "/" + rowItem.rowData.count
                                     : rowItem.rowData.count) + ")"
                                  : rowItem.rowData.label + "  (" + rowItem.rowData.laneCount + ")"
                            color: !rowItem.isLane ? AppTheme.text
                                   : rowItem.rowData.filteredCount === 0 ? AppTheme.textMuted
                                   : AppTheme.categoryColor(rowItem.rowData.category, AppTheme.dark)
                            font.bold: rowItem.isLane || rowItem.level <= 1
                            font.pixelSize: AppTheme.typeLabel
                            elide: Text.ElideRight
                        }
                        MouseArea {
                            id: groupMouse
                            anchors.fill: parent
                            hoverEnabled: !rowItem.isLane
                            acceptedButtons: rowItem.isLane ? Qt.RightButton : (Qt.LeftButton | Qt.RightButton)
                            onClicked: (mouse) => {
                                if (rowItem.isLane) { laneContextMenu.open(); return }
                                TimelineModel.setGroupCollapsed(rowItem.rowData.groupId, !rowItem.rowData.collapsed)
                            }
                            Accessible.role: rowItem.isLane ? Accessible.StaticText : Accessible.Button
                            Accessible.name: rowItem.isLane ? laneLabel.text
                                             : (rowItem.rowData.collapsed ? "Expand " : "Collapse ") + rowItem.rowData.label
                        }
                        Popup {
                            id: laneContextMenu
                            objectName: rowItem.isLane ? "laneContextMenu_" + rowItem.rowData.laneIndex : ""
                            y: laneLabel.height
                            width: 170
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
                                width: parent.width
                                Button {
                                    width: parent.width
                                    flat: true
                                    text: "Hide this lane"
                                    onClicked: { TimelineModel.hideLane(rowItem.rowData.name); laneContextMenu.close() }
                                }
                                Button {
                                    width: parent.width
                                    flat: true
                                    text: "Isolate this lane"
                                    onClicked: { TimelineModel.isolateLane(rowItem.rowData.name); laneContextMenu.close() }
                                }
                                Button {
                                    width: parent.width
                                    flat: true
                                    enabled: TimelineModel.hiddenLanes.length > 0 || TimelineModel.isolatedLanes.length > 0
                                    text: "Show all lanes"
                                    onClicked: { TimelineModel.showAllLanes(); laneContextMenu.close() }
                                }
                            }
                        }
                    }

                    // ── the lane (or the header's aggregate strip) ──────
                    LanePainter {
                        id: laneCanvas
                        objectName: rowItem.isLane ? "laneCanvas_" + rowItem.rowData.laneIndex
                                                   : "groupCoverage_" + rowItem.rowData.groupId
                        x: root.labelWidth
                        width: parent.width - root.labelWidth
                        height: parent.height
                        model: TimelineModel
                        kind: rowItem.isLane ? "lane" : "group"
                        laneIndex: rowItem.isLane ? rowItem.rowData.laneIndex : -1
                        groupId: rowItem.isLane ? "" : rowItem.rowData.groupId
                        laneIndexes: rowItem.isLane ? [] : rowItem.rowData.laneIndexes
                        laneName: rowItem.isLane ? rowItem.rowData.name : ""
                        syncOverlayLaneIndex: rowItem.isLane && rowItem.rowData.syncLane !== undefined ? rowItem.rowData.syncLane : -1
                        viewStartNs: root.viewStartNs
                        viewEndNs: root.viewStartNs + root.visibleNs
                        revision: TimelineModel.paintRevision

                        MouseArea {
                            anchors.fill: parent
                            hoverEnabled: true
                            acceptedButtons: Qt.NoButton
                            onPositionChanged: (mouse) => {
                                var at = mapToItem(root, mouse.x, mouse.y)
                                root.hoverX = at.x
                                root.hoverY = at.y
                                root.cursorNs = root.viewStartNs + mouse.x / laneCanvas.width * root.visibleNs
                                if (!rowItem.isLane) {
                                    root.hoverLane = -1
                                    root.hoverSpanIdx = -1
                                    root.hoverText = rowItem.rowData.label + ": aggregate of " + rowItem.rowData.laneCount +
                                        " lanes (" + (rowItem.rowData.nodeKind === "thread" || rowItem.rowData.nodeKind === "stream"
                                                      ? "busiest lane per column" : "mean busy share of its threads / streams") +
                                        "; hatched = waiting)"
                                    return
                                }
                                var hit = TimelineModel.hitTest(laneCanvas.laneIndex, mouse.x, root.viewStartNs,
                                                                root.viewStartNs + root.visibleNs, laneCanvas.width)
                                var found = hit.kind === "span" ? hit.spanIdx : -1
                                if (found === root.hoverSpanIdx && laneCanvas.laneIndex === root.hoverLane && found >= 0) return
                                if (found >= 0) {
                                    root.hoverLane = laneCanvas.laneIndex
                                    root.hoverSpanIdx = found
                                    var detail = TimelineModel.spanAt(laneCanvas.laneIndex, found)
                                    root.hoverText = detail.name + "  @" + root.fmtNs(detail.startNs) + "  dur " + root.fmtNs(detail.durNs)
                                } else {
                                    root.hoverLane = -1
                                    root.hoverSpanIdx = -1
                                    if (hit.kind === "column")
                                        root.hoverText = "aggregated: " + (hit.count > 0 ? hit.count + " event(s) < 1 px, " : "") +
                                            "busy " + (100 * hit.busy).toFixed(1) + "%" +
                                            (hit.wait > 0 ? ", waiting " + (100 * hit.wait).toFixed(1) + "%" : "") +
                                            " of " + root.fmtNs(hit.endNs - hit.startNs) +
                                            (hit.source === "index" ? (hit.approx ? " (activity index, approximate)" : " (activity index)") : "") +
                                            " — zoom in for the events"
                                    else if (hit.kind === "wait")
                                        root.hoverText = "waiting (sync lane)  @" + root.fmtNs(hit.startNs - TimelineModel.viewStartNs) +
                                                         "  dur " + root.fmtNs(hit.durNs)
                                    else
                                        root.hoverText = ""
                                }
                                overlay.update()
                            }
                            onExited: {
                                root.cursorNs = -1
                                if (root.hoverLane !== laneCanvas.laneIndex && rowItem.isLane) return
                                root.hoverLane = -1
                                root.hoverSpanIdx = -1
                                root.hoverText = ""
                                overlay.update()
                            }
                        }
                    }
                    Rectangle {
                        anchors.bottom: parent.bottom
                        width: parent.width
                        height: 1
                        color: AppTheme.panelBorder
                        opacity: 0.35
                    }
                }
            }

            // ── Dependency / critical-path overlay over the visible rows ──
            OverlayPainter {
                id: overlay
                objectName: "timelineOverlay"
                x: root.labelWidth + 1
                y: 1
                width: flick.width - root.labelWidth
                height: flick.height
                z: 10
                model: TimelineModel
                viewStartNs: root.viewStartNs
                viewEndNs: root.viewStartNs + root.visibleNs
                scrollY: flick.contentY
                rowHeight: root.rowHeight
                hoverLane: root.hoverLane
                hoverSpanIdx: root.hoverSpanIdx
                revision: TimelineModel.paintRevision
            }

            // ── synchronized cursor ──────────────────────────────────────
            Rectangle {
                objectName: "timelineCursor"
                visible: root.cursorNs >= root.viewStartNs && root.cursorNs <= root.viewStartNs + root.visibleNs
                x: root.labelWidth + 1 + (root.cursorNs - root.viewStartNs) / root.visibleNs * (flick.width - root.labelWidth)
                y: 1
                width: 1
                height: flick.height
                z: 11
                color: AppTheme.textMuted
                opacity: 0.7
            }

            // ── Zoom (wheel) + pan (drag) ─────────────────────────────────
            MouseArea {
                objectName: "timelinePanMouseArea"
                anchors.fill: parent
                anchors.leftMargin: root.labelWidth
                anchors.rightMargin: 13
                anchors.bottomMargin: 13
                propagateComposedEvents: true
                acceptedButtons: Qt.LeftButton
                z: 12

                property real dragStartNs: 0
                property real dragStartX: 0
                property bool selecting: false
                property real selectStartNs: 0

                onPressed: (mouse) => {
                    root.forceActiveFocus()
                    dragStartNs = root.viewStartNs
                    dragStartX = mouse.x
                    selecting = (mouse.modifiers & Qt.ShiftModifier) !== 0
                    if (selecting) {
                        var scale0 = width / root.visibleNs
                        selectStartNs = root.viewStartNs + mouse.x / scale0
                    }
                }
                onPositionChanged: (mouse) => {
                    if (!pressed) return
                    var scale = width / root.visibleNs
                    if (selecting) {
                        var curNs = root.viewStartNs + mouse.x / scale
                        Nav.selectTimeRange(Math.min(selectStartNs, curNs), Math.max(selectStartNs, curNs))
                    } else {
                        root.viewStartNs = dragStartNs - (mouse.x - dragStartX) / scale
                        root.clampViewStart()
                    }
                }
                onReleased: { selecting = false }
                // A click ON an event selects it (details panel; cross-tab
                // selection); on an edge, the edge; elsewhere clears.
                onClicked: (mouse) => {
                    if (Math.abs(mouse.x - dragStartX) > 4) return
                    if (mouse.modifiers & Qt.ShiftModifier) return
                    if (root.hoverSpanIdx >= 0) {
                        var d = TimelineModel.spanAt(root.hoverLane, root.hoverSpanIdx)
                        if (!d.name) return
                        TimelineModel.selectEvent(root.hoverLane, root.hoverSpanIdx)
                        Nav.selectFunction(d.category, d.name)
                        Nav.selectThread(d.pid, d.tid)
                        return
                    }
                    var edge = TimelineModel.edgeAt(mouse.x - 1, mouse.y - 1)
                    if (edge >= 0) TimelineModel.selectEdge(edge)
                    else TimelineModel.clearEventSelection()
                }
                onDoubleClicked: {
                    if (root.hoverSpanIdx >= 0)
                        root.zoomToSpan(root.hoverLane, root.hoverSpanIdx)
                    else
                        root.resetView()
                }
                onWheel: (wheel) => {
                    if (wheel.modifiers & Qt.ShiftModifier) {
                        // rows: independent of the time axis
                        var dy = (wheel.angleDelta.y !== 0 ? wheel.angleDelta.y : wheel.angleDelta.x) > 0 ? -3 : 3
                        flick.contentY = Math.max(0, Math.min(Math.max(0, flick.contentHeight - flick.height),
                                                              flick.contentY + dy * root.rowHeight))
                        return
                    }
                    if (wheel.angleDelta.y === 0 && wheel.angleDelta.x !== 0) {
                        root.viewStartNs -= (wheel.angleDelta.x / 120) * root.visibleNs * 0.1
                        root.clampViewStart()
                        return
                    }
                    var factor = wheel.angleDelta.y > 0 ? 1.25 : 0.8
                    root.zoomAtFraction(factor, wheel.x / width)
                }
            }

            // ── Horizontal time scrollbar ───────────────────────────────
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
                    x: (hScrollTrack.width - width) * scrollFrac
                    width: Math.max(20, hScrollTrack.width * thumbFrac)
                    height: parent.height
                    radius: AppTheme.radiusSmall
                    color: hDrag.pressed ? AppTheme.accent : AppTheme.panelBorder
                }
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

        TimelineDetailsPanel {
            id: detailsPanel
            // shrinks before the row overflows (a narrow window with the
            // Inspector docked leaves ~550 px for the whole Timeline)
            Layout.preferredWidth: 340
            Layout.minimumWidth: 160
            Layout.maximumWidth: 340
            Layout.fillWidth: true
            Layout.fillHeight: true
            visible: root.sel.spanIdx >= 0 || root.sel.edge >= 0
            details: root.selDetails
            edge: root.selEdge
            onCloseRequested: TimelineModel.clearEventSelection()
            onJumpTo: (laneIndex, spanIdx, startNs) => root.focusEvent(laneIndex, spanIdx, startNs)
            onSelectEdge: (edgeId) => TimelineModel.selectEdge(edgeId)
            onOpenSource: (category, name) => { Nav.selectFunction(category, name); Nav.navigateTo(6) }
        }
        }

        TimelineBottomPanel {
            id: bottomPanel
            Layout.fillWidth: true
            Layout.preferredHeight: implicitHeight
            onFocusRange: (startNs, endNs) => root.zoomToRange(startNs, endNs)
            onSelectEvent: (laneIndex, spanIdx, startNs, durNs) => {
                if (durNs > 0) root.zoomToRange(startNs, startNs + durNs)
                root.focusEvent(laneIndex, spanIdx, startNs)
            }
            onRevealLanes: (laneIndexes) => { if (laneIndexes && laneIndexes.length) root.revealLane(laneIndexes[0]) }
            onTransitionRequested: (dir) => root.gotoTransition(dir)
        }
    }

    // ── Export ──────────────────────────────────────────────────────────
    Popup {
        id: exportPopup
        objectName: "timelineExportPopup"
        anchors.centerIn: parent
        width: 420
        modal: true
        focus: true
        background: Rectangle {
            color: AppTheme.surface
            border.color: AppTheme.panelBorder
            radius: AppTheme.radiusPanel
        }
        ColumnLayout {
            width: exportPopup.availableWidth
            Text {
                Layout.fillWidth: true
                text: "Save the visible window and rows as a figure (.png or .svg)"
                color: AppTheme.text
                font.pixelSize: AppTheme.typeLabel
                wrapMode: Text.Wrap
            }
            TextField {
                id: exportPath
                objectName: "timelineExportPath"
                Layout.fillWidth: true
                text: AppInfo.tracePath.replace(/\.(json|hpstore)$/, "") + ".timeline.png"
            }
            Flow {
                Layout.fillWidth: true
                spacing: AppTheme.spacingXs
                Button {
                    objectName: "timelineExportSave"
                    text: "Save"
                    onClicked: {
                        var err = TimelineModel.exportFigure(exportPath.text, root.viewStartNs,
                                                             root.viewStartNs + root.visibleNs, 1600)
                        exportStatus.text = err.length ? "⚠ " + err : "saved " + exportPath.text
                    }
                }
                Button { text: "Close"; onClicked: exportPopup.close() }
            }
            Text {
                id: exportStatus
                Layout.fillWidth: true
                color: AppTheme.textMuted
                font.pixelSize: AppTheme.typeCaption
                wrapMode: Text.Wrap
            }
        }
    }

    // The hovered span / column (or the search match jumped to). Floats
    // over the lanes, outside every layout, so it never moves them.
    Tooltip {
        objectName: "timelineHoverTooltip"
        visible: root.hoverText.length > 0
        anchorX: root.hoverX
        anchorY: root.hoverY
        Text {
            objectName: "timelineHoverTooltipText"
            Layout.maximumWidth: Math.max(0, root.width - 3 * AppTheme.spacingLg)
            text: root.hoverText
            color: AppTheme.text
            font.pixelSize: AppTheme.typeLabel
            font.bold: true
            elide: Text.ElideMiddle
        }
    }

    TimelineFirstUseOverlay {}
}
