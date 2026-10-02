import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import Hprofiler 1.0
import "../components"

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
    Component.onCompleted: { jumpToSelectionIfNew(); applyFocusIfNew() }
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
        applyFocusIfNew()     // an explicit range request wins over the selection jump
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

    // Nav.focusTimeRange() requests (e.g. the Compare tab's "Show in
    // Timeline"): zoom to the range once per request -- checked on load and
    // on every visit, since the request is usually made while this tab's
    // Loader is not active yet.
    property int lastFocusSerial: 0
    function applyFocusIfNew() {
        var f = Nav.focusRange
        if (f.serial === undefined || f.serial <= root.lastFocusSerial) return
        root.lastFocusSerial = f.serial
        root.zoomToRange(f.startNs, f.endNs)
    }
    function zoomToRange(startNs, endNs) {
        var span = Math.max(endNs - startNs, 1)
        var pad = span * 0.1
        zoom = Math.max(1.0, Math.min(256.0, TimelineModel.traceDurationNs / (span + 2 * pad)))
        viewStartNs = startNs - pad
        clampViewStart()
    }

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

    // Search "next/previous match" (Phase B4) jump: pans (does NOT zoom,
    // unlike zoomToSpan/double-click -- stepping through matches
    // shouldn't also yank the zoom level around) to center the match at
    // the CURRENT zoom, scrolls its row into view (rowIndexForLane(), not
    // the raw laneIndex -- the row may sit under a group header now), and
    // sets hover state so the connector overlay highlights it the same
    // way a real mouse hover would.
    function jumpToMatch(match) {
        if (!match || match.laneIndex === undefined) return
        viewStartNs = match.startNs - visibleNs / 2.0
        clampViewStart()
        var rowPos = TimelineModel.rowIndexForLane(match.laneIndex)
        if (rowPos >= 0) flick.positionViewAtIndex(rowPos, ListView.Contain)
        var detail = TimelineModel.spanAt(match.laneIndex, match.spanIdx)
        if (detail.name) {
            root.hoverLane = match.laneIndex
            root.hoverSpanIdx = match.spanIdx
            root.hoverText = detail.name + "  @" + root.fmtNs(detail.startNs) +
                             "  dur " + root.fmtNs(detail.durNs)
        }
        overlay.requestPaint()
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
        function onFocusRangeChanged() { if (root.visible) root.applyFocusIfNew() }
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
            Text {
                text: "wheel: zoom@cursor · drag: pan · shift-drag: select range · dbl-click event: zoom to it · arrows/+/-/Home/End/0: keyboard"
                color: AppTheme.textMuted
                font.pixelSize: AppTheme.typeCaption
            }
        }

        // ── Bookmarks / named ranges / bucket legend row ────────────────
        RowLayout {
            Layout.fillWidth: true
            Layout.preferredHeight: 20
            Layout.maximumHeight: 20
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
            Item { Layout.fillWidth: true }
            BucketLegend {
                visible: TimelineModel.colorMode === "bucket"
            }
        }

        // ── Time ruler ───────────────────────────────────────────────────
        TimeRuler {
            id: timeRuler
            objectName: "timelineRuler"
            Layout.fillWidth: true
            Layout.preferredHeight: 26
            Layout.maximumHeight: 26
            Layout.leftMargin: root.labelWidth
            Layout.rightMargin: 14
            viewStartNs: root.viewStartNs
            visibleNs: root.visibleNs
            bookmarks: TimelineModel.bookmarks
            namedRanges: TimelineModel.namedRanges
            selectedRange: Nav.selectedTimeRange
            onBookmarkClicked: (ns) => { root.viewStartNs = ns - root.visibleNs / 2; root.clampViewStart() }
            onBookmarkRemoveRequested: (id) => TimelineModel.removeBookmark(id)
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

            // ListView, not a plain Flickable+Column+Repeater -- only
            // instantiates delegates near the viewport (+ a small cache
            // margin), recycling them while scrolling, instead of
            // building a Canvas for every lane unconditionally regardless
            // of whether it's ever visible (a real perf risk already for
            // a many-rank MPI trace, independent of the table-upgrade/
            // grouping round this was built alongside). `rows`, not
            // `lanes` directly: TimelineModel.rows is the NEW visual row
            // list (group headers + filtered/ordered/hidden-aware lane
            // references) -- each "lane" row still carries its own
            // original `laneIndex`, so visibleSpans()/spanAt() calls
            // below are completely unaffected by this switch.
            ListView {
                id: flick
                objectName: "timelineFlick"
                anchors.fill: parent
                anchors.margins: 1
                anchors.rightMargin: 13
                anchors.bottomMargin: 13
                boundsBehavior: Flickable.StopAtBounds
                model: TimelineModel.rows

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
                    policy: TimelineModel.rows.length * root.rowHeight > flick.height
                            ? ScrollBar.AlwaysOn : ScrollBar.AlwaysOff
                }

                delegate: Loader {
                    id: rowLoader
                    width: flick.width
                    height: root.rowHeight
                    // Explicit properties, not relying on modelData/index
                    // propagating implicitly into a Component instantiated
                    // via sourceComponent -- a Component's creation context
                    // follows where it was DECLARED, not where the Loader
                    // that instantiates it happens to sit, so the loaded
                    // item can't reliably see the delegate's own context
                    // properties by bare name. Every loaded component below
                    // reads `parent.rowData`/`parent.rowPos` instead (the
                    // Loader IS the loaded item's parent) -- same
                    // "always id/parent-qualify, never rely on implicit
                    // nested-scope resolution" lesson as laneCanvas's own
                    // laneIndex property further down.
                    property var rowData: modelData
                    property int rowPos: index
                    sourceComponent: rowData.kind === "group" ? groupRowComponent : laneRowComponent
                }

                // ── Group header row (Phase B3): label + lane count +
                // aggregated count, click-to-collapse/expand, and (only
                // while collapsed) a coverage strip standing in for the
                // member lanes' own bars -- "show meaningful aggregated
                // activity when collapsed" from the Timeline requirements.
                Component {
                    id: groupRowComponent
                    Item {
                        id: groupRow
                        property var rowData: parent.rowData
                        width: parent.width
                        height: parent.height

                        Rectangle {
                            anchors.fill: parent
                            color: groupMouse.containsMouse ? AppTheme.panelBorder : AppTheme.background

                            Accessible.role: Accessible.Button
                            Accessible.name: (groupRow.rowData.collapsed ? "Expand " : "Collapse ") + groupRow.rowData.label
                            Accessible.description: "Toggles whether the \"" + groupRow.rowData.label + "\" lane group is collapsed"
                            Accessible.onPressAction: TimelineModel.setGroupCollapsed(
                                groupRow.rowData.groupId, !groupRow.rowData.collapsed)
                        }

                        RowLayout {
                            x: AppTheme.spacingXs
                            width: root.labelWidth - AppTheme.spacingXs
                            height: parent.height
                            spacing: AppTheme.spacingXs
                            Text {
                                text: groupRow.rowData.collapsed ? "▸" : "▾"
                                color: AppTheme.textMuted
                                font.pixelSize: AppTheme.typeLabel
                            }
                            Text {
                                Layout.fillWidth: true
                                text: groupRow.rowData.label + "  (" + groupRow.rowData.laneCount + ")"
                                color: AppTheme.text
                                font.bold: true
                                font.pixelSize: AppTheme.typeLabel
                                elide: Text.ElideRight
                            }
                        }

                        Canvas {
                            id: coverageCanvas
                            objectName: "groupCoverage_" + groupRow.rowData.groupId
                            visible: groupRow.rowData.collapsed
                            x: root.labelWidth
                            width: parent.width - root.labelWidth
                            height: parent.height
                            property real boundZoom: root.zoom
                            property real boundStart: root.viewStartNs
                            onBoundZoomChanged: if (visible) requestPaint()
                            onBoundStartChanged: if (visible) requestPaint()
                            onVisibleChanged: if (visible) requestPaint()
                            onPaint: {
                                var ctx = getContext("2d")
                                ctx.reset()
                                if (!visible || width <= 0) return
                                var nBuckets = Math.max(1, Math.floor(width / 3))
                                var cov = TimelineModel.groupCoverage(
                                    groupRow.rowData.laneIndexes, root.viewStartNs,
                                    root.viewStartNs + root.visibleNs, nBuckets)
                                var bw = width / nBuckets
                                ctx.fillStyle = AppTheme.accent
                                for (var i = 0; i < cov.length; i++) {
                                    if (cov[i] <= 0) continue
                                    ctx.globalAlpha = 0.35 + 0.45 * cov[i]
                                    ctx.fillRect(i * bw, 4, Math.max(1, bw), height - 8)
                                }
                                ctx.globalAlpha = 1.0
                            }
                        }

                        Text {
                            visible: !groupRow.rowData.collapsed
                            x: root.labelWidth + AppTheme.spacingSm
                            anchors.verticalCenter: parent.verticalCenter
                            text: groupRow.rowData.filteredCount !== groupRow.rowData.count
                                  ? groupRow.rowData.filteredCount + "/" + groupRow.rowData.count + " events"
                                  : groupRow.rowData.count + " events"
                            color: AppTheme.textMuted
                            font.pixelSize: AppTheme.typeCaption
                        }

                        MouseArea {
                            id: groupMouse
                            anchors.fill: parent
                            hoverEnabled: true
                            onClicked: TimelineModel.setGroupCollapsed(
                                groupRow.rowData.groupId, !groupRow.rowData.collapsed)
                        }
                    }
                }

                Component {
                    id: laneRowComponent
                    Item {
                        id: laneRow
                        property var rowData: parent.rowData
                        width: parent.width
                        height: parent.height

                            Text {
                                id: laneLabel
                                width: root.labelWidth
                                height: parent.height
                                verticalAlignment: Text.AlignVCenter
                                text: laneRow.rowData.label + "  (" +
                                      (laneRow.rowData.filteredCount !== laneRow.rowData.count
                                       ? laneRow.rowData.filteredCount + "/" + laneRow.rowData.count
                                       : laneRow.rowData.count) + ")"
                                color: laneRow.rowData.filteredCount === 0 ? AppTheme.textMuted : laneRow.rowData.color
                                font.bold: true
                                font.pixelSize: AppTheme.typeLabel
                                elide: Text.ElideRight
                                leftPadding: AppTheme.spacingSm

                                // Right-click: per-lane hide/isolate -- a
                                // per-ROW action, distinct from
                                // TimelineViewControls' screen-level
                                // grouping/show-all-lanes controls.
                                MouseArea {
                                    anchors.fill: parent
                                    acceptedButtons: Qt.RightButton
                                    onClicked: laneContextMenu.open()
                                }
                                Popup {
                                    id: laneContextMenu
                                    objectName: "laneContextMenu_" + laneRow.rowData.laneIndex
                                    y: laneLabel.height
                                    width: 160
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
                                            onClicked: {
                                                TimelineModel.hideLane(laneRow.rowData.name)
                                                laneContextMenu.close()
                                            }
                                        }
                                        Button {
                                            width: parent.width
                                            flat: true
                                            text: "Isolate this lane"
                                            onClicked: {
                                                TimelineModel.isolateLane(laneRow.rowData.name)
                                                laneContextMenu.close()
                                            }
                                        }
                                        Button {
                                            width: parent.width
                                            flat: true
                                            enabled: TimelineModel.hiddenLanes.length > 0 || TimelineModel.isolatedLanes.length > 0
                                            text: "Show all lanes"
                                            onClicked: {
                                                TimelineModel.showAllLanes()
                                                laneContextMenu.close()
                                            }
                                        }
                                    }
                                }
                            }

                            Canvas {
                                id: laneCanvas
                                objectName: "laneCanvas_" + laneRow.rowData.laneIndex
                                x: root.labelWidth
                                width: parent.width - root.labelWidth
                                height: parent.height
                                // laneRow.rowData.laneIndex, NOT the bare
                                // ListView `index`/rowPos -- a row's visual
                                // position and its underlying lane's real
                                // index diverge once grouping/hiding/
                                // reordering are in play (B3), and every
                                // visibleSpans()/spanAt() call below needs
                                // the latter, stable one.
                                property int laneIndex: laneRow.rowData.laneIndex
                                property string laneName: laneRow.rowData.name
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

                                // Search query/cursor changes and color-mode toggles
                                // (Phase B4) don't move boundZoom/boundStart, so they
                                // need their own repaint trigger -- a direct signal
                                // connection, not a dirty-checked property, since e.g. a
                                // query change that happens to keep the same match COUNT
                                // (different matches, same total) would otherwise be missed.
                                Connections {
                                    target: TimelineModel
                                    function onSearchChanged() {
                                        if (!repaintThrottle.running) repaintThrottle.start()
                                    }
                                    function onColorModeChanged() {
                                        if (!repaintThrottle.running) repaintThrottle.start()
                                    }
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

                                // Outlines every span visibleSpans() marked "matched"
                                // (only present while a search is active, see
                                // TimelineModel.search()'s docstring) -- a thin accent-
                                // colored stroke on top of the normal fill, so a match
                                // reads as highlighted without hiding its own color.
                                function _paintMatchHighlights(ctx, spans, scale) {
                                    var viewEndNs = root.viewStartNs + root.visibleNs
                                    ctx.strokeStyle = AppTheme.accent
                                    ctx.lineWidth = 2
                                    for (var i = 0; i < spans.length; i++) {
                                        var sp = spans[i]
                                        if (!sp.matched) continue
                                        if (sp.startNs + sp.durNs <= root.viewStartNs || sp.startNs >= viewEndNs)
                                            continue
                                        var x0 = (sp.startNs - root.viewStartNs) * scale
                                        var w = Math.max(1, sp.durNs * scale)
                                        ctx.strokeRect(Math.max(0, x0) + 0.5, 1.5,
                                                        Math.min(w, width - x0) - 1, height - 3)
                                    }
                                }

                                // Zoomed out past ~2000 spans per lane, the model returns
                                // occupancy bins from the store's precomputed activity
                                // index (one value per pixel column) instead of spans --
                                // painted as the lane colour with opacity = how busy
                                // that column is. Individual spans (and hover detail)
                                // come back as soon as the window is narrow enough.
                                function _paintBins(ctx, view, alpha) {
                                    var bins = view.bins
                                    if (!bins || bins.length === 0) return
                                    var bw = width / bins.length
                                    ctx.fillStyle = view.color
                                    for (var i = 0; i < bins.length; i++) {
                                        var v = bins[i]
                                        if (v <= 0) continue
                                        ctx.globalAlpha = alpha * (0.25 + 0.75 * v)
                                        ctx.fillRect(i * bw, 3, Math.max(1, bw), height - 6)
                                    }
                                    ctx.globalAlpha = 1.0
                                }

                                onPaint: {
                                    var ctx = getContext("2d")
                                    ctx.reset()
                                    var view = TimelineModel.laneView(
                                        laneIndex, root.viewStartNs, root.viewStartNs + root.visibleNs,
                                        Math.max(1, Math.round(width)), 2000)
                                    var scale = width / root.visibleNs
                                    if (view.mode === "bins") {
                                        cachedSpans = []
                                        _paintBins(ctx, view, 1.0)
                                    } else {
                                        cachedSpans = view.spans
                                        _paintSpans(ctx, view.spans, scale, function(sp) { return sp.color }, 1.0)
                                        _paintMatchHighlights(ctx, view.spans, scale)
                                    }

                                    // Overlay the paired sync lane's spans ON TOP,
                                    // dimmed, so a barrier/critical-section wait
                                    // nested inside one of the spans just painted
                                    // above reads as visibly idle instead of being
                                    // silently absorbed into the parent's solid
                                    // "busy" color -- see syncOverlayLaneIndex's
                                    // comment for why the same time interval can
                                    // legitimately belong to both lanes at once.
                                    if (syncOverlayLaneIndex >= 0) {
                                        var syncView = TimelineModel.laneView(
                                            syncOverlayLaneIndex, root.viewStartNs, root.viewStartNs + root.visibleNs,
                                            Math.max(1, Math.round(width)), 2000)
                                        if (syncView.mode === "bins") {
                                            _paintBins(ctx, syncView, 0.8)
                                            return
                                        }
                                        var syncSpans = syncView.spans
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
                    // Reparented to flick.contentItem (a ListView's own
                    // internal scrolling Item, holding every delegate) so
                    // this stays aligned with the lane rows as the user
                    // scrolls -- a plain child of the ListView itself
                    // would stay fixed to the viewport instead, verified
                    // with an isolated repro (a Rectangle reparented to
                    // contentItem, contentY moved, scenePos confirmed to
                    // shift by the same amount) before relying on it here.
                    Canvas {
                        id: overlay
                        parent: flick.contentItem
                        x: root.labelWidth
                        y: 0
                        width: flick.width - root.labelWidth
                        height: TimelineModel.rows.length * root.rowHeight
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
                                // rowIndexForLane(), NOT the raw lane index, as the
                                // row-position multiplier -- a row's visual position
                                // and its underlying lane's real index diverge once
                                // grouping/hiding/isolating/reordering (B3) are in
                                // play; -1 means that lane isn't currently a visible
                                // row at all (hidden, filtered out, or collapsed
                                // inside a group), so its edges simply aren't drawn
                                // rather than drawn at a wrong/stale position.
                                var predRow = TimelineModel.rowIndexForLane(c.predLane)
                                var succRow = TimelineModel.rowIndexForLane(c.succLane)
                                if (predRow < 0 || succRow < 0) continue
                                var x0 = (c.predMidNs - root.viewStartNs) * scale
                                var x1 = (c.succMidNs - root.viewStartNs) * scale
                                var y0 = predRow * root.rowHeight + root.rowHeight / 2
                                var y1 = succRow * root.rowHeight + root.rowHeight / 2
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

                    // Row POSITIONS (not just which lanes exist) can shift
                    // under grouping/hide/isolate/collapse/reorder (B3) even
                    // while a connector is being shown for an already-
                    // hovered span -- repaint so it never freezes at a
                    // stale y-position after e.g. a group gets collapsed.
                    Connections {
                        target: TimelineModel
                        function onRowsChanged() { overlay.requestPaint() }
                    }
                }

            // ── Zoom (wheel) + pan (drag) ─────────────────────────────────
            // rightMargin/bottomMargin match the Flickable's above --
            // reserves the vertical/horizontal scrollbar strips so this
            // (which covers everything else and consumes all drag input
            // for time-panning) never overlaps and steals their clicks.
            MouseArea {
                objectName: "timelinePanMouseArea"
                anchors.fill: parent
                anchors.leftMargin: root.labelWidth
                anchors.rightMargin: 13
                anchors.bottomMargin: 13
                propagateComposedEvents: true
                acceptedButtons: Qt.LeftButton

                property real dragStartNs: 0
                property real dragStartX: 0
                // Shift-drag selects a time range (Nav.selectedTimeRange,
                // dormant since Round 16 -- feeds TimeRuler's highlighted
                // band and "+ Range" below) instead of panning -- decided
                // once at press time from the modifier held THEN, not
                // re-evaluated mid-drag, so releasing/re-pressing Shift
                // partway through a drag can't switch modes underneath it.
                property bool selecting: false
                property real selectStartNs: 0

                onPressed: (mouse) => {
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
                    if (mouse.modifiers & Qt.ShiftModifier) return
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

    TimelineFirstUseOverlay {}
}
