import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import Hprofiler 1.0

// Coordinated views under the Timeline. Every bar, row and cell leads back
// to the events it was computed from: a click highlights them on the
// timeline (TimelineModel.setHighlight), selects them or zooms to them.
//   Summary       time by activity (host, device), per process, waiting by
//                 runtime, busiest lanes, evidence, dependency coverage,
//                 top waits and their upstream causes, path switches
//   Critical path wall time vs. the observed path vs. the modeled path vs.
//                 the span-level search score -- one scale, reconciled
//   Heatmap       busy share per process / device over the whole run
Rectangle {
    id: panel
    objectName: "timelineBottomPanel"
    color: AppTheme.surface
    border.color: AppTheme.panelBorder
    border.width: 1
    radius: AppTheme.radiusPanel

    property bool expanded: false
    property int tab: 0
    property var sum: ({})
    property var hm: ({})
    readonly property var cps: TimelineModel.criticalSummary
    readonly property bool ready: TimelineModel.analysisState === "ready"

    signal focusRange(real startNs, real endNs)
    signal selectEvent(int laneIndex, real spanIdx, real startNs, real durNs)
    signal revealLanes(var laneIndexes)
    signal transitionRequested(int direction)

    implicitHeight: expanded ? 250 : 26

    function refresh() {
        if (!expanded) return
        if (tab === 0) sum = TimelineModel.summaries()
        else if (tab === 2) hm = TimelineModel.heatmap(Math.max(20, Math.floor((width - 170) / 6)))
    }
    onExpandedChanged: refresh()
    onTabChanged: refresh()
    Connections {
        target: TimelineModel
        function onAnalysisChanged() { panel.refresh() }
        function onFiltersChanged() { panel.refresh() }
    }

    function fmt(ns) { return Format.duration(ns) }
    function pct(x) { return (100 * x).toFixed(1) + "%" }
    // `t` shortened with an ellipsis to at most `maxW` pixels in ctx's font
    function fitText(ctx, t, maxW) {
        if (ctx.measureText(t).width <= maxW) return t
        var lo = 0, hi = t.length
        while (lo < hi) {
            var mid = Math.ceil((lo + hi) / 2)
            if (ctx.measureText(t.slice(0, mid) + "…").width <= maxW) lo = mid
            else hi = mid - 1
        }
        return t.slice(0, lo) + "…"
    }
    // a summary card as wide as its widest row asks for, within [lo, hi]
    // and never wider than the panel
    function cardWidth(implicit, lo, hi) {
        return Math.min(sumScroll.availableWidth, Math.max(lo, Math.min(hi, implicit)))
    }

    // One summary row: an optional bar, a label that shrinks (elided) and a
    // value that never does -- the number is what the row is for.
    component BarRow: Rectangle {
        id: br
        property string label: ""
        property string value: ""
        property real share: 0
        property color barColor: AppTheme.accent
        property bool hatched: false
        property real barWidth: 70
        property color labelColor: AppTheme.text
        property string tip: ""
        signal activated()
        Layout.fillWidth: true
        Layout.minimumWidth: 0
        implicitWidth: (barWidth > 0 ? barWidth + 6 : 0) + labelText.implicitWidth + 10 + valueText.implicitWidth
        implicitHeight: 16
        color: brArea.containsMouse ? AppTheme.control : "transparent"
        Rectangle {
            visible: br.barWidth > 0
            x: 0; y: 3; height: 10
            width: Math.max(1, br.barWidth * Math.max(0, Math.min(1, br.share)))
            color: br.barColor
            opacity: br.hatched ? 0.45 : 0.9
            border.width: br.hatched ? 1 : 0
            border.color: AppTheme.textMuted
        }
        Text {
            id: valueText
            anchors.right: parent.right
            anchors.verticalCenter: parent.verticalCenter
            text: br.value
            color: AppTheme.text
            font.pixelSize: AppTheme.typeCaption
        }
        Text {
            id: labelText
            x: br.barWidth > 0 ? br.barWidth + 6 : 0
            width: Math.max(0, valueText.x - x - 10)
            anchors.verticalCenter: parent.verticalCenter
            text: br.label
            color: br.labelColor
            font.pixelSize: AppTheme.typeCaption
            elide: Text.ElideRight
        }
        MouseArea {
            id: brArea
            anchors.fill: parent
            hoverEnabled: true
            cursorShape: Qt.PointingHandCursor
            onClicked: br.activated()
        }
        ToolTip.visible: brArea.containsMouse && labelText.truncated
        ToolTip.text: br.label + "  " + br.value
    }
    // a card title or note: wraps inside its card
    component CardText: Text {
        Layout.fillWidth: true
        Layout.minimumWidth: 0
        color: AppTheme.text
        font.pixelSize: AppTheme.typeCaption
        wrapMode: Text.Wrap
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 2
        spacing: 2

        RowLayout {
            Layout.fillWidth: true
            Layout.minimumWidth: 0
            Layout.preferredHeight: 22
            spacing: AppTheme.spacingXs
            ToolButton {
                objectName: "timelineBottomToggle"
                text: panel.expanded ? "▾" : "▸"
                implicitHeight: 22
                onClicked: panel.expanded = !panel.expanded
                Accessible.name: panel.expanded ? "Collapse summaries" : "Expand summaries"
            }
            Repeater {
                model: ["Summary", "Critical path", "Heatmap"]
                delegate: ToolButton {
                    objectName: "timelineBottomTab_" + index
                    text: modelData
                    implicitHeight: 22
                    checkable: true
                    checked: panel.expanded && panel.tab === index
                    onClicked: { panel.tab = index; panel.expanded = true; panel.refresh() }
                }
            }
            Item { Layout.fillWidth: true }
            Text {
                // the headline numbers, always visible
                objectName: "timelineBottomHeadline"
                text: panel.ready && panel.cps.wallNs !== undefined
                      ? "wall " + panel.fmt(panel.cps.wallNs) + " [measured] · observed path " + panel.fmt(panel.cps.observedNs)
                        + " [each instant once] · modeled " + panel.fmt(panel.cps.modeledNs) + " [model]"
                      : "Critical path: " + (TimelineModel.analysisState === "running" ? "computing…" : "not computed")
                color: AppTheme.textMuted
                font.pixelSize: AppTheme.typeCaption
                elide: Text.ElideLeft
                // fill + min 0: a non-fill item is fixed at its full width,
                // which widened the whole panel past its frame
                Layout.fillWidth: true
                Layout.minimumWidth: 0
                // whole pixels: a layout rounds down, and a fraction short of
                // the text's width already elides it
                Layout.preferredWidth: Math.ceil(implicitWidth) + 1
                Layout.maximumWidth: Math.ceil(implicitWidth) + 1
            }
        }

        // ── Summary ───────────────────────────────────────────────────
        ScrollView {
            id: sumScroll
            // a visible scrollbar when the content is taller than the panel,
            // so rows below the fold read as "scroll", not "cut off"
            ScrollBar.vertical.policy: contentHeight > availableHeight ? ScrollBar.AlwaysOn : ScrollBar.AsNeeded
            visible: panel.expanded && panel.tab === 0
            Layout.fillWidth: true
            Layout.fillHeight: true
            clip: true
            contentWidth: availableWidth
            Flow {
                width: sumScroll.availableWidth
                spacing: AppTheme.spacingMd

                Repeater {
                    // stacked-bar cards: host / device activity
                    model: [{"title": "Host threads: exclusive time by activity", "rows": panel.sum.host || []},
                            {"title": "Device work by activity", "rows": panel.sum.device || []},
                            {"title": "Explicit waiting by runtime (click: highlight)", "rows": panel.sum.waits || []}]
                    delegate: ColumnLayout {
                        id: barCard
                        width: panel.cardWidth(implicitWidth, 260, 420)
                        spacing: 1
                        visible: modelData.rows.length > 0
                        CardText { text: modelData.title; font.bold: true }
                        Repeater {
                            model: modelData.rows
                            delegate: BarRow {
                                label: modelData.label
                                value: panel.fmt(modelData.ns) + (modelData.share !== undefined ? "  (" + panel.pct(modelData.share) + ")" : "")
                                share: modelData.share !== undefined ? modelData.share : 0
                                barColor: modelData.colorKind === "activity"
                                          ? AppTheme.activityColor(modelData.key, AppTheme.dark)
                                          : AppTheme.categoryColor(modelData.key, AppTheme.dark)
                                hatched: !!modelData.hatched
                                onActivated: TimelineModel.setHighlight(modelData.highlight)
                            }
                        }
                    }
                }

                ColumnLayout {
                    width: panel.cardWidth(implicitWidth, 250, 320)
                    spacing: 1
                    visible: (panel.sum.processes || []).length > 0
                    CardText { text: "Per process (click: highlight its lanes)"; font.bold: true }
                    Repeater {
                        objectName: "timelineSummaryProcesses"
                        model: panel.sum.processes || []
                        delegate: Item {
                            Layout.fillWidth: true
                            implicitWidth: 74 + 160
                            implicitHeight: 15
                            Text {
                                width: 70
                                anchors.verticalCenter: parent.verticalCenter
                                text: modelData.label
                                color: AppTheme.textMuted
                                font.pixelSize: AppTheme.typeCaption
                                elide: Text.ElideRight
                            }
                            Row {
                                x: 74; y: 3
                                Repeater {
                                    model: modelData.parts
                                    delegate: Rectangle {
                                        width: Math.max(1, 160 * modelData.share)
                                        height: 9
                                        color: AppTheme.activityColor(modelData.key, AppTheme.dark)
                                        opacity: modelData.hatched ? 0.45 : 0.9
                                    }
                                }
                            }
                            MouseArea {
                                anchors.fill: parent
                                cursorShape: Qt.PointingHandCursor
                                onClicked: { TimelineModel.setHighlight(modelData.highlight); panel.revealLanes(modelData.highlight.lanes) }
                            }
                        }
                    }
                }

                ColumnLayout {
                    width: panel.cardWidth(implicitWidth, 250, 380)
                    spacing: 1
                    visible: (panel.sum.streams || []).length + (panel.sum.threads || []).length > 0
                    CardText { text: "Busiest lanes (union busy time)"; font.bold: true }
                    Repeater {
                        model: (panel.sum.streams || []).slice(0, 5).concat((panel.sum.threads || []).slice(0, 5))
                        delegate: BarRow {
                            barWidth: 0
                            label: modelData.label
                            value: panel.fmt(modelData.ns)
                            labelColor: AppTheme.categoryColor(modelData.category, AppTheme.dark)
                            onActivated: { TimelineModel.setHighlight(modelData.highlight); panel.revealLanes([modelData.laneIndex]) }
                        }
                    }
                }

                ColumnLayout {
                    width: panel.cardWidth(implicitWidth, 250, 340)
                    spacing: 1
                    CardText { text: "Evidence on the critical path"; font.bold: true }
                    CardText {
                        visible: !panel.ready
                        text: "needs the dependency graph (Critical path ▸ Compute)"
                        color: AppTheme.textMuted
                    }
                    Repeater {
                        model: panel.ready ? (panel.sum.tiers || []).concat(panel.sum.classes || []) : []
                        delegate: BarRow {
                            barWidth: 80
                            label: modelData.key
                            value: panel.fmt(modelData.ns) + "  (" + panel.pct(modelData.share) + ")"
                            share: modelData.share
                            barColor: AppTheme.confidenceColor(modelData.key, AppTheme.dark)
                        }
                    }
                    CardText {
                        visible: panel.ready
                        text: "graph edges: " + (panel.sum.graphTiers || []).map(function(t) { return t.key + " " + t.count }).join(", ")
                        color: AppTheme.textMuted
                    }
                    CardText {
                        visible: panel.ready
                        text: "tiers are evidence classes, not probabilities"
                        color: AppTheme.textMuted
                        font.italic: true
                    }
                }

                ColumnLayout {
                    width: panel.cardWidth(implicitWidth, 250, 340)
                    spacing: 1
                    visible: panel.ready && (panel.sum.coverage || []).length > 0
                    CardText { text: "Dependency coverage"; font.bold: true }
                    Repeater {
                        model: panel.sum.coverage || []
                        delegate: CardText { text: modelData.label + ": " + modelData.value }
                    }
                }

                ColumnLayout {
                    width: panel.cardWidth(implicitWidth, 320, 440)
                    spacing: 1
                    visible: panel.ready && (panel.sum.topWaits || []).length > 0
                    CardText { text: "Top waits and their upstream cause (derived from the dependency graph)"; font.bold: true }
                    Repeater {
                        objectName: "timelineSummaryTopWaits"
                        model: panel.sum.topWaits || []
                        delegate: Rectangle {
                            Layout.fillWidth: true
                            Layout.minimumWidth: 0
                            implicitHeight: 28
                            color: waitArea.containsMouse ? AppTheme.control : "transparent"
                            Text {
                                x: 2; y: 1
                                width: parent.width - 4
                                text: panel.fmt(modelData.waitNs) + "  " + modelData.name + " @ " + modelData.place
                                color: AppTheme.text
                                font.pixelSize: AppTheme.typeCaption
                                elide: Text.ElideRight
                            }
                            Text {
                                x: 12; y: 14
                                width: parent.width - 14
                                text: "← " + modelData.cause + "  [" + modelData.tier + ", " + modelData.provenance + "]"
                                color: AppTheme.textMuted
                                font.pixelSize: AppTheme.typeCaption
                                elide: Text.ElideRight
                            }
                            MouseArea {
                                id: waitArea
                                anchors.fill: parent
                                hoverEnabled: true
                                cursorShape: Qt.PointingHandCursor
                                acceptedButtons: Qt.LeftButton | Qt.RightButton
                                onClicked: (mouse) => {
                                    if (mouse.button === Qt.RightButton && modelData.causeLaneIndex >= 0)
                                        panel.selectEvent(modelData.causeLaneIndex, modelData.causeSpanIdx, modelData.startNs, 0)
                                    else
                                        panel.selectEvent(modelData.laneIndex, modelData.spanIdx, modelData.startNs,
                                                          modelData.endNs - modelData.startNs)
                                }
                            }
                            ToolTip.visible: waitArea.containsMouse
                            ToolTip.text: "click: the waiting call · right-click: its cause"
                        }
                    }
                }

                ColumnLayout {
                    width: panel.cardWidth(implicitWidth, 250, 360)
                    spacing: 1
                    visible: panel.ready && (panel.sum.transitions || []).length > 0
                    CardText { text: "Critical-path runtime switches"; font.bold: true }
                    Repeater {
                        model: panel.sum.transitions || []
                        delegate: BarRow {
                            barWidth: 0
                            label: modelData.key
                            value: modelData.count + " ×"
                        }
                    }
                }
            }
        }

        // ── Critical path ─────────────────────────────────────────────
        ScrollView {
            id: cpScroll
            // a visible scrollbar when the content is taller than the panel,
            // so rows below the fold read as "scroll", not "cut off"
            ScrollBar.vertical.policy: contentHeight > availableHeight ? ScrollBar.AlwaysOn : ScrollBar.AsNeeded
            visible: panel.expanded && panel.tab === 1
            Layout.fillWidth: true
            Layout.fillHeight: true
            clip: true
            contentWidth: availableWidth

            ColumnLayout {
                width: cpScroll.availableWidth
                spacing: AppTheme.spacingXs

                Text {
                    Layout.fillWidth: true
                    wrapMode: Text.Wrap
                    visible: !panel.ready
                    text: TimelineModel.analysisState === "running" ? TimelineModel.analysisMessage
                          : "Not computed. Critical path ▸ Compute (runs in the background)."
                    color: AppTheme.textMuted
                    font.pixelSize: AppTheme.typeLabel
                }

                Flow {
                    id: cpRow
                    visible: panel.ready
                    Layout.fillWidth: true
                    spacing: AppTheme.spacingLg

                    // the numbers, each with what kind of value it is: one
                    // grid cell per text, so columns size to their content
                    GridLayout {
                        id: cpNumbers
                        objectName: "timelineCpNumbers"
                        columns: 3
                        rowSpacing: 0
                        columnSpacing: AppTheme.spacingSm
                        Repeater {
                            model: {
                                if (!panel.ready) return []
                                var rows = [
                                    ["Wall time", panel.fmt(panel.cps.wallNs), "measured"],
                                    ["Observed path (unique time)", panel.fmt(panel.cps.observedNs), "each instant once"],
                                    ["Path length", panel.cps.pathSpans + " spans", "derived"],
                                    ["Modeled longest path", panel.fmt(panel.cps.modeledNs), "model: required work only"],
                                    ["  unrecorded time as idle", panel.fmt(panel.cps.modeledIdleNs), "lower bound"],
                                    ["  direct evidence only", panel.fmt(panel.cps.modeledDirectNs), "model"],
                                    ["Span-level search value", panel.cps.legacyValueNs >= 0 ? panel.fmt(panel.cps.legacyValueNs) : "—",
                                     "a search score, not elapsed time"],
                                    ["Busy on every / some longest path", panel.fmt(panel.cps.workOnAllNs) + " / " + panel.fmt(panel.cps.workOnSomeNs), "model"],
                                    ["Ties (equally consistent choices)", "" + panel.cps.ties, "derived"]
                                ]
                                var cells = []
                                for (var i = 0; i < rows.length; i++)
                                    cells.push(rows[i][0], rows[i][1], "[" + rows[i][2] + "]")
                                return cells
                            }
                            delegate: Text {
                                readonly property int col: index % 3
                                text: modelData
                                color: col === 0 ? AppTheme.textMuted : col === 1 ? AppTheme.text : AppTheme.accent
                                font.pixelSize: AppTheme.typeCaption
                                font.bold: col === 1
                                elide: Text.ElideRight
                                Layout.maximumWidth: col === 0 ? 240 : col === 1 ? 180 : 230
                            }
                        }
                    }

                    // time reconciliation: one scale for all four quantities
                    Canvas {
                        id: recon
                        objectName: "timelineReconciliation"
                        // beside the numbers when there is room, else under them
                        readonly property real besideWidth: cpRow.width - cpNumbers.width - cpRow.spacing
                        width: besideWidth >= 360 ? besideWidth : cpRow.width
                        height: 120
                        property var c: panel.cps
                        onCChanged: requestPaint()
                        onWidthChanged: requestPaint()
                        Connections {
                            target: AppTheme
                            function onThemeChanged() { recon.requestPaint() }
                        }
                        onPaint: {
                            var ctx = getContext("2d")
                            ctx.reset()
                            var c = panel.cps
                            if (!c || c.wallNs === undefined) return
                            ctx.font = AppTheme.typeCaption + "px sans-serif"
                            // room for the widest label, so none is clipped at the right edge
                            var labels = ["wall " + panel.fmt(c.wallNs), "observed " + panel.fmt(c.observedNs),
                                          "modeled " + panel.fmt(c.modeledNs),
                                          c.legacyValueNs > 0 ? "span-level score " + panel.fmt(c.legacyValueNs) : ""]
                            var labelW = 0
                            for (var li = 0; li < labels.length; li++)
                                labelW = Math.max(labelW, ctx.measureText(labels[li]).width)
                            var left = 4, w = Math.max(40, width - labelW - 20)
                            var maxv = Math.max(c.wallNs, c.observedNs, c.modeledNs, c.legacyValueNs > 0 ? c.legacyValueNs : 0) * 1.04
                            var k = w / Math.max(maxv, 1)
                            var rowH = 18, y = 6
                            function label(t, yy) { ctx.fillStyle = AppTheme.text; ctx.fillText(t, left + w + 8, yy + 12) }
                            // wall
                            ctx.strokeStyle = AppTheme.text
                            ctx.lineWidth = 1
                            ctx.strokeRect(left, y, c.wallNs * k, rowH - 4)
                            label("wall " + panel.fmt(c.wallNs), y); y += rowH + 4
                            // observed path: its composition
                            var x = left
                            var comp = c.composition || []
                            for (var i = 0; i < comp.length; i++) {
                                var part = comp[i], pw = part.ns * k
                                if (part.group === "busy") ctx.fillStyle = AppTheme.categoryColor(part.key, AppTheme.dark)
                                else if (part.group === "dependency latency") ctx.fillStyle = AppTheme.criticalColor
                                else ctx.fillStyle = AppTheme.textMuted
                                ctx.globalAlpha = part.group === "busy" ? 0.9 : 0.45
                                ctx.fillRect(x, y, Math.max(0.5, pw), rowH - 4)
                                x += pw
                            }
                            ctx.globalAlpha = 1
                            label("observed " + panel.fmt(c.observedNs), y); y += rowH + 4
                            // modeled path and its lower bound
                            ctx.fillStyle = AppTheme.accent
                            ctx.globalAlpha = 0.75
                            ctx.fillRect(left, y, c.modeledNs * k, rowH - 4)
                            ctx.globalAlpha = 1
                            ctx.strokeStyle = AppTheme.text
                            ctx.beginPath(); ctx.moveTo(left + c.modeledIdleNs * k, y - 2); ctx.lineTo(left + c.modeledIdleNs * k, y + rowH - 2); ctx.stroke()
                            label("modeled " + panel.fmt(c.modeledNs), y); y += rowH + 4
                            // span-level score: covered once + counted again
                            if (c.legacyValueNs > 0) {
                                var once = Math.min(c.legacyUnionNs, c.legacyValueNs)
                                ctx.fillStyle = AppTheme.textMuted
                                ctx.globalAlpha = 0.6
                                ctx.fillRect(left, y, once * k, rowH - 4)
                                ctx.globalAlpha = 1
                                ctx.fillStyle = AppTheme.overlapColor
                                ctx.fillRect(left + once * k, y, Math.max(0, (c.legacyValueNs - once) * k), rowH - 4)
                                label("span-level score " + panel.fmt(c.legacyValueNs), y)
                            }
                            // the wall-time line across every bar
                            ctx.strokeStyle = AppTheme.warningColor
                            ctx.setLineDash([4, 3])
                            ctx.beginPath(); ctx.moveTo(left + c.wallNs * k, 2); ctx.lineTo(left + c.wallNs * k, y + rowH); ctx.stroke()
                        }
                    }
                }

                Text {
                    visible: panel.ready
                    Layout.fillWidth: true
                    text: "Observed path bar: busy time by runtime (solid), in gated calls / unrecorded / device idle (grey), "
                          + "dependency latency (yellow). The span-level search adds each hop's full duration, so where a "
                          + "start-gated hop's spans overlap it counts that wall time twice (red) and can exceed the wall line. "
                          + "Turn on the score track to see where on the timeline."
                    color: AppTheme.textMuted
                    font.pixelSize: AppTheme.typeCaption
                    wrapMode: Text.Wrap
                }

                RowLayout {
                    visible: panel.ready
                    Layout.fillWidth: true
                    Button { text: "◀ previous switch"; onClicked: panel.transitionRequested(-1) }
                    Button { text: "next switch ▶"; onClicked: panel.transitionRequested(1) }
                    Text {
                        Layout.fillWidth: true
                        Layout.minimumWidth: 0
                        text: (panel.cps.transitions || 0) + " switches (rank / thread / runtime / device / stream)"
                        color: AppTheme.textMuted
                        font.pixelSize: AppTheme.typeCaption
                        elide: Text.ElideRight
                    }
                }

                Text {
                    visible: panel.ready
                    text: "Composition of the observed path (click: highlight)"
                    color: AppTheme.text
                    font.bold: true
                    font.pixelSize: AppTheme.typeCaption
                }
                Flow {
                    visible: panel.ready
                    Layout.fillWidth: true
                    spacing: AppTheme.spacingSm
                    Repeater {
                        model: panel.ready ? (panel.cps.composition || []) : []
                        delegate: Text {
                            text: modelData.group + ": " + modelData.key + " " + panel.fmt(modelData.ns)
                            color: modelData.group === "busy" ? AppTheme.categoryColor(modelData.key, AppTheme.dark) : AppTheme.textMuted
                            font.pixelSize: AppTheme.typeCaption
                            MouseArea {
                                anchors.fill: parent
                                cursorShape: Qt.PointingHandCursor
                                onClicked: {
                                    if (modelData.group === "busy")
                                        TimelineModel.setHighlight({"categories": [modelData.key], "activities": ["work", "communication", "memory", "overhead", "jit"], "label": "path: busy " + modelData.key})
                                    else if (modelData.group === "in gated call")
                                        TimelineModel.setHighlight({"categories": [modelData.key], "activities": ["wait"], "label": "path: waiting in " + modelData.key})
                                }
                            }
                        }
                    }
                }
                Text {
                    Layout.fillWidth: true
                    wrapMode: Text.Wrap
                    visible: panel.ready && (TimelineModel.criticalSummary.diagnostics || {}).receives !== undefined
                    text: "MPI dependency coverage: " + ((TimelineModel.criticalSummary.diagnostics || {}).matched || 0)
                          + " / " + ((TimelineModel.criticalSummary.diagnostics || {}).receives || 0)
                          + " receive completions have a sender edge"
                    color: AppTheme.text
                    font.pixelSize: AppTheme.typeCaption
                }
                Repeater {
                    model: panel.ready ? (panel.cps.notes || []) : []
                    delegate: Text {
                        Layout.fillWidth: true
                        text: "• " + modelData
                        color: AppTheme.textMuted
                        font.pixelSize: AppTheme.typeCaption
                        wrapMode: Text.Wrap
                    }
                }
            }
        }

        // ── Heatmap ───────────────────────────────────────────────────
        Item {
            visible: panel.expanded && panel.tab === 2
            Layout.fillWidth: true
            Layout.fillHeight: true
            Canvas {
                id: heat
                objectName: "timelineHeatmap"
                anchors.fill: parent
                property var h: panel.hm
                onHChanged: requestPaint()
                Connections {
                    target: AppTheme
                    function onThemeChanged() { heat.requestPaint() }
                }
                onPaint: {
                    var ctx = getContext("2d")
                    ctx.reset()
                    var d = panel.hm
                    if (!d || !d.rows || d.rows.length === 0) return
                    var labelW = 160, rows = d.rows, n = d.bins
                    var rh = Math.max(6, Math.min(16, (height - 4) / rows.length))
                    var cw = (width - labelW - 4) / n
                    ctx.font = AppTheme.typeCaption + "px sans-serif"
                    for (var r = 0; r < rows.length; r++) {
                        var y = 2 + r * rh
                        ctx.fillStyle = AppTheme.textMuted
                        ctx.fillText(panel.fitText(ctx, rows[r].label, labelW - 8), 2, y + rh - 3)
                        for (var i = 0; i < n; i++) {
                            var b = rows[r].busy[i], wv = rows[r].wait[i]
                            if (b > 0) {
                                ctx.globalAlpha = 0.15 + 0.85 * b
                                ctx.fillStyle = AppTheme.accent
                                ctx.fillRect(labelW + i * cw, y, Math.max(1, cw - 0.5), rh - 1)
                            }
                            if (wv > 0.05) {
                                ctx.globalAlpha = 0.6
                                ctx.fillStyle = AppTheme.textMuted
                                ctx.fillRect(labelW + i * cw, y + rh - 3, Math.max(1, cw - 0.5), 2)
                            }
                        }
                        ctx.globalAlpha = 1
                    }
                }
                MouseArea {
                    anchors.fill: parent
                    cursorShape: Qt.PointingHandCursor
                    onClicked: (mouse) => {
                        var d = panel.hm
                        if (!d || !d.rows || d.rows.length === 0) return
                        var labelW = 160
                        var rh = Math.max(6, Math.min(16, (height - 4) / d.rows.length))
                        var r = Math.floor((mouse.y - 2) / rh)
                        var i = Math.floor((mouse.x - labelW) / ((width - labelW - 4) / d.bins))
                        if (r < 0 || r >= d.rows.length || i < 0 || i >= d.bins) return
                        var span = (d.t1 - d.t0) / d.bins
                        TimelineModel.setHighlight({"lanes": d.rows[r].laneIndexes, "label": d.rows[r].label})
                        panel.revealLanes(d.rows[r].laneIndexes)
                        panel.focusRange(d.t0 + i * span, d.t0 + (i + 1) * span)
                    }
                }
            }
            Text {
                anchors.right: parent.right
                anchors.bottom: parent.bottom
                width: Math.min(implicitWidth, parent.width)
                horizontalAlignment: Text.AlignRight
                text: "cell: busy share of the process's threads (device: streams); grey tick: waiting · click: zoom + highlight"
                color: AppTheme.textMuted
                font.pixelSize: AppTheme.typeCaption
                elide: Text.ElideLeft
            }
        }
    }
}
