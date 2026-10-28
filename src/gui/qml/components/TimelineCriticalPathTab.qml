import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import Hprofiler 1.0

// The bottom panel's "Critical path" tab: the headline numbers and their
// reconciliation, then sections disclosed on demand -- the four path
// variants and the modeled range, alternative worlds (with their
// evaluation on request), edge evidence by status, clock domains and
// undecidable steps, blame, sensitivity / replay, probabilities. Every
// value says what kind it is; nothing alternative-world-dependent is shown
// as "every world" before the worlds were evaluated.
ColumnLayout {
    id: tab
    objectName: "timelineCriticalPathTab"
    spacing: AppTheme.spacingXs

    property var cps: ({})
    property bool ready: false
    readonly property var alt: cps.alternatives || ({})
    readonly property var job: TimelineModel.alternativesJob
    readonly property string membership: TimelineModel.overlay.membership || ""

    signal selectEvent(int laneIndex, real spanIdx, real startNs, real durNs)
    signal selectEdge(int edgeId, real atNs)
    signal transitionRequested(int direction)

    function fmt(ns) { return Format.duration(ns) }
    function pct(x) { return (100 * x).toFixed(0) + "%" }
    function setMembership(mode, world) {
        TimelineModel.setOverlay({"membership": tab.membership === mode && (world === undefined || world < 0) ? "" : mode,
                                  "world": world === undefined ? -1 : world})
    }
    function stateColor(st) {
        if (st === "evaluated") return AppTheme.evidenceColor("robust", AppTheme.dark)
        if (st === "none_found") return AppTheme.textMuted
        if (st === "truncated") return AppTheme.evidenceColor("partial", AppTheme.dark)
        return AppTheme.warningColor
    }

    // A collapsible section: the header names it and summarises it; the
    // body is disclosed on click.
    component Section: ColumnLayout {
        id: sec
        property string key: ""
        property string title: ""
        property string summary: ""
        property bool open: false
        default property alias content: body.data
        Layout.fillWidth: true
        Layout.minimumWidth: 0
        spacing: 2
        Rectangle {
            objectName: "cpSectionHeader_" + sec.key
            Layout.fillWidth: true
            Layout.minimumWidth: 0
            implicitHeight: 20
            radius: AppTheme.radiusSmall
            color: hdr.containsMouse ? AppTheme.control : "transparent"
            Text {
                x: 4
                width: parent.width - 8
                anchors.verticalCenter: parent.verticalCenter
                text: (sec.open ? "▾ " : "▸ ") + sec.title + (sec.summary.length ? "  —  " + sec.summary : "")
                color: AppTheme.text
                font.bold: true
                font.pixelSize: AppTheme.typeCaption
                elide: Text.ElideRight
            }
            MouseArea {
                id: hdr
                anchors.fill: parent
                hoverEnabled: true
                cursorShape: Qt.PointingHandCursor
                onClicked: sec.open = !sec.open
            }
            Accessible.role: Accessible.Button
            Accessible.name: (sec.open ? "Collapse " : "Expand ") + sec.title
        }
        ColumnLayout {
            id: body
            objectName: "cpSectionBody_" + sec.key
            visible: sec.open
            Layout.fillWidth: true
            Layout.minimumWidth: 0
            Layout.leftMargin: 12
            spacing: 2
        }
    }
    // a table cell: as wide as its text, shrinking (elided) when the panel
    // is narrower than the table -- a grid must never widen its column
    component Cell: Text {
        Layout.minimumWidth: 0
        Layout.fillWidth: true
        Layout.preferredWidth: Math.ceil(implicitWidth) + 1
        Layout.maximumWidth: Math.ceil(implicitWidth) + 1
        color: AppTheme.text
        font.pixelSize: AppTheme.typeCaption
        elide: Text.ElideRight
    }
    // a wrapped caption line
    component Note: Text {
        Layout.fillWidth: true
        Layout.minimumWidth: 0
        color: AppTheme.textMuted
        font.pixelSize: AppTheme.typeCaption
        wrapMode: Text.Wrap
    }
    // a clickable one-line row: elided label, value never elided
    component LinkRow: Rectangle {
        id: lr
        property string label: ""
        property string value: ""
        property color labelColor: AppTheme.text
        property string tip: ""
        signal activated()
        Layout.fillWidth: true
        Layout.minimumWidth: 0
        implicitHeight: 16
        implicitWidth: lrLabel.implicitWidth + lrValue.implicitWidth + 16
        color: lrArea.containsMouse ? AppTheme.control : "transparent"
        Text {
            id: lrValue
            anchors.right: parent.right
            anchors.verticalCenter: parent.verticalCenter
            text: lr.value
            color: AppTheme.text
            font.pixelSize: AppTheme.typeCaption
        }
        Text {
            id: lrLabel
            x: 2
            width: Math.max(0, lrValue.x - 10)
            anchors.verticalCenter: parent.verticalCenter
            text: lr.label
            color: lr.labelColor
            font.pixelSize: AppTheme.typeCaption
            elide: Text.ElideRight
        }
        MouseArea {
            id: lrArea
            anchors.fill: parent
            hoverEnabled: true
            cursorShape: Qt.PointingHandCursor
            onClicked: lr.activated()
        }
        ToolTip.visible: lrArea.containsMouse && (lr.tip.length > 0 || lrLabel.truncated)
        ToolTip.text: lr.tip.length ? lr.tip : lr.label + "  " + lr.value
    }

    Text {
        Layout.fillWidth: true
        wrapMode: Text.Wrap
        visible: !tab.ready
        text: TimelineModel.analysisState === "running" ? TimelineModel.analysisMessage
              : "Not computed. Critical path ▸ Compute (runs in the background)."
        color: AppTheme.textMuted
        font.pixelSize: AppTheme.typeLabel
    }

    Flow {
        id: cpRow
        visible: tab.ready
        Layout.fillWidth: true
        spacing: AppTheme.spacingLg

        // the numbers, each with what kind of value it is: one grid cell
        // per text, so columns size to their content
        GridLayout {
            id: cpNumbers
            objectName: "timelineCpNumbers"
            columns: 3
            rowSpacing: 0
            columnSpacing: AppTheme.spacingSm
            Repeater {
                model: {
                    if (!tab.ready) return []
                    var c = tab.cps, r = c.modeledRange || {}
                    var rows = [
                        ["Wall time", tab.fmt(c.wallNs), "measured"],
                        ["Observed path (unique time)", tab.fmt(c.observedNs), "each instant once"],
                        ["Path length", c.pathSpans + " spans", "derived"],
                        ["Modeled longest path", tab.fmt(c.modeledNs), "model: unrecorded time as work"],
                        ["Modeled range (4 variants)", r.minNs !== undefined ? tab.fmt(r.minNs) + " – " + tab.fmt(r.maxNs) : "—",
                         "model: see Variants"],
                        ["Ties (equally consistent choices)", "" + c.ties, "derived"],
                        ["Undecidable by the clocks", "" + (c.undecidable || 0), "see Clocks"],
                        ["Alternative worlds", tab.alt.stateLabel || "—", tab.alt.state === "not_evaluated" ? "see Alternative worlds" : "derived"],
                        ["Edge probabilities", (c.probabilities || {}).calibrated ? "calibrated" : "unquantified",
                         (c.probabilities || {}).calibrated ? "see Probabilities" : "no calibration given"]
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

        // time reconciliation: one scale for wall, observed, modeled (with
        // its range over the variants) and the span-level score
        Canvas {
            id: recon
            objectName: "timelineReconciliation"
            readonly property real besideWidth: cpRow.width - cpNumbers.width - cpRow.spacing
            width: besideWidth >= 360 ? besideWidth : cpRow.width
            height: 120
            property var c: tab.cps
            onCChanged: requestPaint()
            onWidthChanged: requestPaint()
            Connections {
                target: AppTheme
                function onThemeChanged() { recon.requestPaint() }
            }
            onPaint: {
                var ctx = getContext("2d")
                ctx.reset()
                var c = tab.cps
                if (!c || c.wallNs === undefined) return
                var r = c.modeledRange || {}
                ctx.font = AppTheme.typeCaption + "px sans-serif"
                var labels = ["wall " + tab.fmt(c.wallNs), "observed " + tab.fmt(c.observedNs),
                              "modeled " + tab.fmt(c.modeledNs) + (r.minNs !== undefined ? "  [" + tab.fmt(r.minNs) + " – " + tab.fmt(r.maxNs) + "]" : ""),
                              c.legacyValueNs > 0 ? "span-level score " + tab.fmt(c.legacyValueNs) : ""]
                var labelW = 0
                for (var li = 0; li < labels.length; li++)
                    labelW = Math.max(labelW, ctx.measureText(labels[li]).width)
                var left = 4, w = Math.max(40, width - labelW - 20)
                var maxv = Math.max(c.wallNs, c.observedNs, c.modeledNs, r.maxNs || 0, c.legacyValueNs > 0 ? c.legacyValueNs : 0) * 1.04
                var k = w / Math.max(maxv, 1)
                var rowH = 18, y = 6
                function label(t, yy) { ctx.fillStyle = AppTheme.text; ctx.fillText(t, left + w + 8, yy + 12) }
                ctx.strokeStyle = AppTheme.text
                ctx.lineWidth = 1
                ctx.strokeRect(left, y, c.wallNs * k, rowH - 4)
                label(labels[0], y); y += rowH + 4
                var x = left
                var comp = c.composition || []
                for (var i = 0; i < comp.length; i++) {
                    var part = comp[i], pw = part.ns * k
                    if (part.group === "busy") ctx.fillStyle = AppTheme.categoryColor(part.key, AppTheme.dark)
                    else if (part.group === "dependency latency") ctx.fillStyle = AppTheme.criticalColor
                    else if (part.group === "in an ungated blocking call") ctx.fillStyle = AppTheme.evidenceColor("unknown", AppTheme.dark)
                    else ctx.fillStyle = AppTheme.textMuted
                    ctx.globalAlpha = part.group === "busy" ? 0.9 : 0.45
                    ctx.fillRect(x, y, Math.max(0.5, pw), rowH - 4)
                    x += pw
                }
                ctx.globalAlpha = 1
                label(labels[1], y); y += rowH + 4
                // modeled path, and its range over the variants as a bracket
                ctx.fillStyle = AppTheme.accent
                ctx.globalAlpha = 0.75
                ctx.fillRect(left, y, c.modeledNs * k, rowH - 4)
                ctx.globalAlpha = 1
                if (r.minNs !== undefined) {
                    ctx.strokeStyle = AppTheme.text
                    var y0 = y + rowH - 2
                    ctx.beginPath()
                    ctx.moveTo(left + r.minNs * k, y0 - 6); ctx.lineTo(left + r.minNs * k, y0)
                    ctx.lineTo(left + r.maxNs * k, y0); ctx.lineTo(left + r.maxNs * k, y0 - 6)
                    ctx.stroke()
                }
                label(labels[2], y); y += rowH + 4
                if (c.legacyValueNs > 0) {
                    var once = Math.min(c.legacyUnionNs, c.legacyValueNs)
                    ctx.fillStyle = AppTheme.textMuted
                    ctx.globalAlpha = 0.6
                    ctx.fillRect(left, y, once * k, rowH - 4)
                    ctx.globalAlpha = 1
                    ctx.fillStyle = AppTheme.overlapColor
                    ctx.fillRect(left + once * k, y, Math.max(0, (c.legacyValueNs - once) * k), rowH - 4)
                    label(labels[3], y)
                }
                ctx.strokeStyle = AppTheme.warningColor
                ctx.setLineDash([4, 3])
                ctx.beginPath(); ctx.moveTo(left + c.wallNs * k, 2); ctx.lineTo(left + c.wallNs * k, y + rowH); ctx.stroke()
            }
        }
    }

    Note {
        visible: tab.ready
        text: "Observed path bar: busy time by runtime (solid), in gated calls / unrecorded / device idle (grey), "
              + "in blocking calls nobody records releasing (pink), dependency latency (yellow). The bracket under the "
              + "modeled bar is its range over the four variants. The span-level search adds each hop's full duration, "
              + "so it can count wall time twice (red) and exceed the wall line."
    }

    RowLayout {
        visible: tab.ready
        Layout.fillWidth: true
        Button { text: "◀ previous switch"; onClicked: tab.transitionRequested(-1) }
        Button { text: "next switch ▶"; onClicked: tab.transitionRequested(1) }
        Text {
            Layout.fillWidth: true
            Layout.minimumWidth: 0
            text: (tab.cps.transitions || 0) + " switches (rank / thread / runtime / device / stream)"
            color: AppTheme.textMuted
            font.pixelSize: AppTheme.typeCaption
            elide: Text.ElideRight
        }
    }

    Text {
        visible: tab.ready
        text: "Composition of the observed path (click: highlight)"
        color: AppTheme.text
        font.bold: true
        font.pixelSize: AppTheme.typeCaption
    }
    Flow {
        visible: tab.ready
        Layout.fillWidth: true
        spacing: AppTheme.spacingSm
        Repeater {
            model: tab.ready ? (tab.cps.composition || []) : []
            delegate: Text {
                text: modelData.group + ": " + modelData.key + " " + tab.fmt(modelData.ns)
                color: modelData.group === "busy" ? AppTheme.categoryColor(modelData.key, AppTheme.dark)
                       : modelData.group === "in an ungated blocking call" ? AppTheme.evidenceColor("unknown", AppTheme.dark)
                       : AppTheme.textMuted
                font.pixelSize: AppTheme.typeCaption
                MouseArea {
                    anchors.fill: parent
                    cursorShape: Qt.PointingHandCursor
                    onClicked: {
                        if (modelData.group === "busy")
                            TimelineModel.setHighlight({"categories": [modelData.key], "activities": ["work", "communication", "memory", "overhead", "jit"], "label": "path: busy " + modelData.key})
                        else if (modelData.group === "in gated call")
                            TimelineModel.setHighlight({"categories": [modelData.key], "activities": ["wait"], "label": "path: waiting in " + modelData.key})
                        else if (modelData.group === "in an ungated blocking call")
                            TimelineModel.setOverlay({"unknownBlocking": true})
                    }
                }
            }
        }
    }

    // ── Variants and the modeled range ──────────────────────────────
    Section {
        key: "variants"
        title: "Variants and the modeled range"
        summary: tab.ready && tab.cps.modeledRange ? "modeled " + tab.fmt(tab.cps.modeledRange.minNs) + " – "
                 + tab.fmt(tab.cps.modeledRange.maxNs) + "; stable " + tab.cps.stableCount + ", variant-only "
                 + tab.cps.variantOnlyCount + ", near-critical " + tab.cps.nearCount : ""
        open: true
        visible: tab.ready
        Note { text: "Where the trace cannot decide how to count time, the walk and the modeled path are recomputed four ways. "
                     + "None of them is a bound by its semantics alone: the range's ends are whichever variants give them." }
        GridLayout {
            objectName: "cpVariantsTable"
            Layout.fillWidth: true
            Layout.minimumWidth: 0
            columns: 7
            rowSpacing: 1
            columnSpacing: AppTheme.spacingSm
            Repeater {
                model: ["variant", "unrecorded / unknown", "edges", "gated tails", "observed", "modeled", "path spans (vs recorded)"]
                delegate: Cell { text: modelData; color: AppTheme.textMuted; font.pixelSize: AppTheme.typeCaption; font.bold: true }
            }
            Repeater {
                model: {
                    var out = []
                    var rows = tab.cps.variantsTable || []
                    for (var i = 0; i < rows.length; i++) {
                        var v = rows[i]
                        var mark = (v.headline ? " · headline" : "") + (v.isMin ? " · range min" : "") + (v.isMax ? " · range max" : "")
                        out.push({"t": v.label, "b": true, "n": "cpVariantRow_" + v.name, "v": v.name},
                                 {"t": v.unrecorded}, {"t": v.edges}, {"t": v.tails},
                                 {"t": tab.fmt(v.observedNs)}, {"t": tab.fmt(v.modeledNs) + mark, "n": "cpVariantModeled_" + v.name},
                                 {"t": v.spans + (v.sameAsDefault ? ", same" : ", different")})
                    }
                    return out
                }
                delegate: Cell {
                    objectName: modelData.n || ""
                    text: modelData.t
                    color: modelData.v !== undefined ? AppTheme.accent : AppTheme.text
                    font.pixelSize: AppTheme.typeCaption
                    font.bold: !!modelData.b
                    font.underline: modelData.v !== undefined
                    MouseArea {
                        // a variant's name: highlight the spans on its observed path
                        anchors.fill: parent
                        enabled: modelData.v !== undefined
                        cursorShape: enabled ? Qt.PointingHandCursor : Qt.ArrowCursor
                        onClicked: TimelineModel.setOverlay(
                            tab.membership === "variant" && TimelineModel.overlay.variant === modelData.v
                            ? {"membership": "", "variant": ""} : {"membership": "variant", "variant": modelData.v})
                    }
                }
            }
        }
        Note { text: "Click a variant's name to mark the spans on its observed path." }
        Text {
            objectName: "cpRangeText"
            Layout.fillWidth: true
            Layout.minimumWidth: 0
            wrapMode: Text.Wrap
            visible: !!tab.cps.modeledRange
            text: {
                var r = tab.cps.modeledRange || {}
                if (r.minNs === undefined) return ""
                return "Modeled range: " + tab.fmt(r.minNs) + " (" + r.minVariants.join(", ") + ") to "
                       + tab.fmt(r.maxNs) + " (" + r.maxVariants.join(", ") + ").  Headline "
                       + tab.fmt(r.headlineNs) + ": unrecorded time counted as work.  [model]"
            }
            color: AppTheme.text
            font.pixelSize: AppTheme.typeCaption
        }
        Text {
            objectName: "cpLegacyText"
            Layout.fillWidth: true
            Layout.minimumWidth: 0
            wrapMode: Text.Wrap
            text: "Span-level search value (the pre-cp-5 DP, kept for comparison): "
                  + (tab.cps.legacyValueNs >= 0 ? tab.fmt(tab.cps.legacyValueNs) : "—")
                  + "  [a search score, not elapsed time]"
            color: AppTheme.textMuted
            font.pixelSize: AppTheme.typeCaption
        }
        Flow {
            Layout.fillWidth: true
            spacing: AppTheme.spacingXs
            Button {
                objectName: "cpHighlightVariants"
                text: tab.membership === "variants" ? "Clear variant highlight" : "Highlight stable / variant-only"
                onClicked: tab.setMembership("variants")
                ToolTip.visible: hovered
                ToolTip.text: "Strip under spans on the observed path in every variant (solid) or some (hatched); others dimmed"
            }
            Button {
                objectName: "cpHighlightNear"
                text: tab.membership === "near" ? "Clear near-critical" : "Highlight near-critical"
                onClicked: tab.setMembership("near")
                ToolTip.visible: hovered
                ToolTip.text: "Busy work whose modeled slack is within 1% of the modeled length (" + (tab.cps.nearBasis || "") + ")"
            }
        }
    }

    // ── Alternative worlds ─────────────────────────────────────────
    Section {
        key: "alternatives"
        title: "Alternative worlds"
        summary: (tab.alt.stateLabel || "") + (tab.alt.verdict ? " · " + tab.alt.verdict : "")
        open: true
        visible: tab.ready
        RowLayout {
            Layout.fillWidth: true
            spacing: AppTheme.spacingSm
            Rectangle {
                objectName: "cpAltStateChip"
                implicitWidth: altStateText.implicitWidth + 14
                implicitHeight: 18
                radius: 9
                color: "transparent"
                border.color: tab.stateColor(tab.alt.state)
                Text {
                    id: altStateText
                    objectName: "cpAltState"
                    anchors.centerIn: parent
                    text: tab.alt.stateLabel || "—"
                    color: tab.stateColor(tab.alt.state)
                    font.pixelSize: AppTheme.typeCaption
                    font.bold: true
                }
            }
            Text {
                objectName: "cpAltReason"
                Layout.fillWidth: true
                Layout.minimumWidth: 0
                text: tab.alt.reason || ""
                color: AppTheme.textMuted
                font.pixelSize: AppTheme.typeCaption
                wrapMode: Text.Wrap
            }
        }
        RowLayout {
            Layout.fillWidth: true
            spacing: AppTheme.spacingSm
            Text { text: "random worlds"; color: AppTheme.textMuted; font.pixelSize: AppTheme.typeCaption }
            SpinBox {
                id: altSamples
                objectName: "cpAltSamples"
                from: 0
                to: 64
                value: 2
                editable: true
                enabled: !tab.job.running
                Accessible.name: "Random alternative worlds to evaluate"
            }
            Button {
                objectName: "cpAltEvaluate"
                text: tab.alt.evaluated ? "Re-evaluate" : "Evaluate alternative worlds"
                enabled: !tab.job.running
                onClicked: TimelineModel.evaluateAlternatives(altSamples.value)
            }
            Button {
                objectName: "cpAltCancel"
                text: "Cancel"
                visible: !!tab.job.running
                onClicked: TimelineModel.cancelAlternatives()
            }
            BusyIndicator {
                objectName: "cpAltBusy"
                running: !!tab.job.running
                visible: running
                implicitWidth: 20
                implicitHeight: 20
            }
            Text {
                objectName: "cpAltJob"
                Layout.fillWidth: true
                Layout.minimumWidth: 0
                text: (tab.job.message || "") + (tab.job.error ? "  ⚠ " + tab.job.error : "")
                color: tab.job.error ? AppTheme.warningColor : AppTheme.textMuted
                font.pixelSize: AppTheme.typeCaption
                elide: Text.ElideRight
            }
        }
        Note {
            text: "Each world is one more assembly of the event graph (" + (tab.alt.nodes || 0).toLocaleString()
                  + " event nodes here): up to 16 one-change worlds plus the random ones — seconds on small traces, "
                  + "minutes and more memory on large ones. The worlds are a sample of those the trace allows, not a "
                  + "probability distribution. The rest of the GUI stays usable meanwhile."
        }
        ColumnLayout {
            visible: !!tab.alt.evaluated
            Layout.fillWidth: true
            spacing: 2
            Text {
                objectName: "cpAltVerdict"
                Layout.fillWidth: true
                Layout.minimumWidth: 0
                wrapMode: Text.Wrap
                text: "Verdict: " + (tab.alt.verdict || "—") + " · " + (tab.alt.worlds || 0) + " worlds ("
                      + (tab.alt.oneChange || 0) + " one-change of " + (tab.alt.touching || 0) + " units touching the path, "
                      + (tab.alt.sampled || 0) + " sampled) · on the path in every / some world: " + (tab.alt.every || 0)
                      + " / " + (tab.alt.some || 0) + " spans · near-critical in every world: " + (tab.alt.near || 0)
                color: AppTheme.text
                font.pixelSize: AppTheme.typeCaption
            }
            Repeater {
                model: tab.alt.reasons || []
                delegate: Note { text: "• " + modelData; color: AppTheme.text }
            }
            Note {
                text: "Observed length " + tab.fmt((tab.alt.coveredRangeNs || [0, 0])[0]) + " – " + tab.fmt((tab.alt.coveredRangeNs || [0, 0])[1])
                      + ", modeled " + tab.fmt((tab.alt.modeledRangeNs || [0, 0])[0]) + " – " + tab.fmt((tab.alt.modeledRangeNs || [0, 0])[1])
                      + " across the worlds; " + (tab.alt.alternativeEdges || 0) + " alternative edge(s) used."
            }
            Flow {
                Layout.fillWidth: true
                spacing: AppTheme.spacingXs
                Button {
                    objectName: "cpHighlightWorlds"
                    text: tab.membership === "worlds" ? "Clear world highlight" : "Highlight every / some / no world"
                    onClicked: tab.setMembership("worlds")
                    ToolTip.visible: hovered
                    ToolTip.text: "Strip under spans on the path in every evaluated world (solid) or some (hatched); spans on no world's path dimmed"
                }
                Button {
                    objectName: "cpShowWorstWorld"
                    text: "Least robust world"
                    enabled: tab.alt.worstWorld !== undefined && tab.alt.worstWorld >= 0
                    onClicked: TimelineModel.setOverlay({"membership": "world", "world": tab.alt.worstWorld})
                }
            }
            Text {
                text: "Evaluated worlds (click: highlight its path and draw its alternative edges)"
                color: AppTheme.text
                font.bold: true
                font.pixelSize: AppTheme.typeCaption
            }
            Repeater {
                objectName: "cpWorldRows"
                model: tab.alt.worldRows || []
                delegate: LinkRow {
                    objectName: "cpWorldRow_" + index
                    label: (modelData.index === tab.alt.worstWorld ? "◆ " : "") + modelData.label
                           + (tab.membership === "world" && TimelineModel.overlay.world === modelData.index ? "  [shown]" : "")
                    value: "observed " + tab.fmt(modelData.coveredNs) + " · modeled " + tab.fmt(modelData.modeledNs)
                           + " · robust " + tab.pct(modelData.robustShare) + " · +" + modelData.joined + " / −" + modelData.left
                           + " spans · " + modelData.altEdges.length + " alt edge(s)"
                    labelColor: modelData.index === tab.alt.worstWorld ? AppTheme.evidenceColor("partial", AppTheme.dark) : AppTheme.text
                    tip: modelData.index === tab.alt.worstWorld ? "the least robust world: the smallest share of its path's time in calls on spans on the path in every world" : ""
                    onActivated: TimelineModel.setOverlay({"membership": "world", "world": modelData.index})
                }
            }
        }
    }

    // ── Edge evidence by status ────────────────────────────────────
    Section {
        key: "evidence"
        title: "Edge evidence by status"
        summary: {
            var rows = tab.cps.statusSummary || [], parts = []
            for (var i = 0; i < rows.length; i++) parts.push(rows[i].status + " " + rows[i].candidates)
            return parts.join(", ")
        }
        visible: tab.ready
        GridLayout {
            objectName: "cpStatusTable"
            Layout.fillWidth: true
            Layout.minimumWidth: 0
            columns: 6
            rowSpacing: 1
            columnSpacing: AppTheme.spacingSm
            Repeater {
                model: ["status", "candidates", "graph edges", "summed source → target", "on the observed path", "disabled"]
                delegate: Cell { text: modelData; color: AppTheme.textMuted; font.pixelSize: AppTheme.typeCaption; font.bold: true }
            }
            Repeater {
                model: {
                    var out = [], rows = tab.cps.statusSummary || []
                    for (var i = 0; i < rows.length; i++) {
                        var r = rows[i]
                        out.push({"t": r.status, "s": r.status}, {"t": "" + r.candidates}, {"t": "" + r.edges},
                                 {"t": tab.fmt(r.latencyNs)}, {"t": tab.fmt(r.pathNs)}, {"t": r.disabled ? "" + r.disabled : "—"})
                    }
                    return out
                }
                delegate: Cell {
                    text: modelData.t
                    color: modelData.s === "ambiguous" ? AppTheme.evidenceColor("ambiguous", AppTheme.dark)
                           : modelData.s === "contradicted" ? AppTheme.evidenceColor("contradicted", AppTheme.dark)
                           : modelData.s === "unsupported" ? AppTheme.evidenceColor("alternative", AppTheme.dark)
                           : AppTheme.text
                    font.pixelSize: AppTheme.typeCaption
                    font.bold: modelData.s !== undefined
                }
            }
        }
        CheckBox {
            objectName: "cpShowDisabledEdges"
            text: "Draw contradicted / disabled edges (never active dependencies)"
            checked: !!TimelineModel.overlay.disabledEdges
            onToggled: TimelineModel.setOverlay({"disabledEdges": checked})
        }
        Note {
            text: "Statuses: observed = direct, the only candidate; inferred = chosen by a rule; ambiguous = other candidates "
                  + "remain; contradicted = the timestamps or clock bounds rule it out; unsupported = an alternative no builder "
                  + "chose. " + (tab.alt.evaluated ? "" : "Alternative worlds were not evaluated: no edge is called unambiguous.")
        }
    }

    // ── Clocks ─────────────────────────────────────────────────────
    Section {
        key: "clocks"
        title: "Clock domains"
        summary: {
            var c = tab.cps.clock || {}
            if (!c.available) return "single clock"
            return (c.domains || []).length + " domains · " + (c.disabledByClock || 0) + " contradicted by the bounds · "
                   + (c.droppedByTime || 0) + " by the timestamps · " + (c.undecidable || 0) + " undecidable step(s)"
        }
        visible: tab.ready
        GridLayout {
            objectName: "cpClockTable"
            Layout.fillWidth: true
            Layout.minimumWidth: 0
            columns: 4
            rowSpacing: 1
            columnSpacing: AppTheme.spacingSm
            visible: !!(tab.cps.clock || {}).available
            Repeater {
                model: ["domain", "bound", "offset interval", "applied shift"]
                delegate: Cell { text: modelData; color: AppTheme.textMuted; font.pixelSize: AppTheme.typeCaption; font.bold: true }
            }
            Repeater {
                model: {
                    var out = [], ds = (tab.cps.clock || {}).domains || []
                    for (var i = 0; i < ds.length; i++)
                        out.push({"t": ds[i].domain}, {"t": ds[i].bound}, {"t": ds[i].lo + " … " + ds[i].hi},
                                 {"t": ds[i].shift, "w": ds[i].shifted})
                    return out
                }
                delegate: Cell {
                    text: modelData.t
                    color: modelData.w ? AppTheme.warningColor : AppTheme.text
                    font.pixelSize: AppTheme.typeCaption
                }
            }
        }
        Note {
            visible: !!(tab.cps.clock || {}).available
            text: ((tab.cps.clock || {}).feasibleAsRecorded ? "The recorded times satisfy every cross-clock dependency within the bounds."
                   : "The recorded times do not satisfy every cross-clock dependency: what was given up —")
        }
        Repeater {
            model: (tab.cps.clock || {}).relaxed || []
            delegate: Note { text: "• " + modelData }
        }
        Text {
            visible: (tab.cps.undecidableSteps || []).length > 0
            text: "Undecidable steps of the observed path (click: select the step's edge and go there)"
            color: AppTheme.text
            font.bold: true
            font.pixelSize: AppTheme.typeCaption
        }
        Repeater {
            objectName: "cpUndecidableRows"
            model: tab.cps.undecidableSteps || []
            delegate: LinkRow {
                objectName: "cpUndecidableRow_" + index
                label: "@ " + tab.fmt(modelData.atNs - TimelineModel.viewStartNs) + "  " + modelData.chosenKind + " ("
                       + modelData.chosenDomain + ") chosen over " + modelData.alternativeKind + " (" + modelData.altDomain + ")"
                value: "margin " + tab.fmt(modelData.marginNs)
                tip: "the clocks' offsets cannot order these two releases: the path may run through the other one"
                onActivated: {
                    if (modelData.chosenEdge >= 0) tab.selectEdge(modelData.chosenEdge, modelData.atNs)
                    else tab.selectEvent(modelData.laneIndex, modelData.spanIdx, modelData.atNs, 0)
                }
            }
        }
        Flow {
            Layout.fillWidth: true
            spacing: AppTheme.spacingXs
            visible: ((tab.cps.clock || {}).disabledByClock || 0) + ((tab.cps.clock || {}).droppedByTime || 0) > 0
            Button {
                objectName: "cpShowContradicted"
                text: TimelineModel.overlay.disabledEdges ? "Hide contradicted edges" : "Draw contradicted edges"
                onClicked: TimelineModel.setOverlay({"disabledEdges": !TimelineModel.overlay.disabledEdges})
            }
        }
    }

    // ── Blame ──────────────────────────────────────────────────────
    Section {
        key: "blame"
        title: "Blame: who made others wait"
        summary: (tab.cps.blame || {}).available ? tab.fmt(tab.cps.blame.waitingNs) + " waiting · "
                 + tab.cps.blame.jointReleases + " joint release(s)" : "not computed"
        visible: tab.ready
        Note {
            visible: !(tab.cps.blame || {}).available
            text: (tab.cps.blame || {}).reason || ""
        }
        Repeater {
            objectName: "cpBlameRows"
            model: (tab.cps.blame || {}).byCategory || []
            delegate: LinkRow {
                label: modelData.key + (modelData.bucket ? "  (not a call: time the trace records no work for)" : "")
                value: tab.fmt(modelData.ns) + "  (" + tab.pct(modelData.share) + ")"
                labelColor: modelData.bucket ? AppTheme.textMuted : AppTheme.categoryColor(modelData.key, AppTheme.dark)
                onActivated: if (!modelData.bucket) TimelineModel.setHighlight({"categories": [modelData.key], "label": "blame: " + modelData.key})
            }
        }
        Text {
            visible: ((tab.cps.blame || {}).jointWaits || []).length > 0
            text: "Joint releases: waits released by several sources at once (their blame is split equally)"
            color: AppTheme.text
            font.bold: true
            font.pixelSize: AppTheme.typeCaption
        }
        Repeater {
            objectName: "cpJointRows"
            model: (tab.cps.blame || {}).jointWaits || []
            delegate: ColumnLayout {
                id: jointWait
                readonly property int nRel: modelData.releasers.length
                Layout.fillWidth: true
                spacing: 0
                LinkRow {
                    label: "◇ " + modelData.name + " waited " + tab.fmt(modelData.waitNs) + " — released by " + modelData.releasers.length + " sources"
                    value: "@ " + tab.fmt(modelData.releaseNs - TimelineModel.viewStartNs)
                    onActivated: tab.selectEvent(modelData.laneIndex, modelData.spanIdx, modelData.startNs, modelData.releaseNs - modelData.startNs)
                }
                Repeater {
                    model: modelData.releasers
                    delegate: LinkRow {
                        Layout.leftMargin: 14
                        label: "← " + modelData.name
                        value: "share " + (100 / Math.max(1, jointWait.nRel)).toFixed(0) + "%"
                        onActivated: tab.selectEvent(modelData.laneIndex, modelData.spanIdx, modelData.startNs, 0)
                    }
                }
            }
        }
        Text {
            visible: ((tab.cps.blame || {}).passedOn || []).length > 0
            text: "Waiting passed on: blame charged to calls that were themselves waiting, forwarded to what released them"
            color: AppTheme.text
            font.bold: true
            font.pixelSize: AppTheme.typeCaption
        }
        Repeater {
            objectName: "cpPassedRows"
            model: (tab.cps.blame || {}).passedOn || []
            delegate: LinkRow {
                objectName: "cpPassedRow_" + index
                label: "↪ " + modelData.name
                value: tab.fmt(modelData.ns) + " passed on"
                onActivated: tab.selectEvent(modelData.laneIndex, modelData.spanIdx, modelData.startNs, 0)
            }
        }
        Note {
            visible: (tab.cps.blame || {}).available === true
            text: "A waiting call's time is charged to the busy time on its releasing source's lane in the waiting window; "
                  + "time the source spent waiting itself is passed on to what released it; (unrecorded), (unknown) and "
                  + "(device idle) are the parts of those windows with no recorded work."
        }
    }

    // ── Sensitivity and replay ─────────────────────────────────────
    Section {
        key: "sensitivity"
        title: "Sensitivity and replay"
        summary: (tab.cps.sensitivityRows || []).length + " candidate(s)"
        visible: tab.ready
        Note {
            text: "Replay is computed only for these candidates (the names and spans with the most busy time on longest paths): "
                  + "modeled cut = how much shorter the modeled path gets with that busy time removed; replayed = how much "
                  + "earlier the run ends replaying the trace without it, every recorded residual kept. Graph estimates, "
                  + "not measured speedups." + (tab.cps.sensitivityNote ? "  " + tab.cps.sensitivityNote : "")
        }
        GridLayout {
            objectName: "cpSensitivityTable"
            Layout.fillWidth: true
            Layout.minimumWidth: 0
            columns: 4
            rowSpacing: 1
            columnSpacing: AppTheme.spacingSm
            Repeater {
                model: ["busy time set to zero for", "on longest paths", "modeled cut", "replayed"]
                delegate: Cell { text: modelData; color: AppTheme.textMuted; font.pixelSize: AppTheme.typeCaption; font.bold: true }
            }
            Repeater {
                model: {
                    var out = [], rows = tab.cps.sensitivityRows || []
                    for (var i = 0; i < rows.length; i++) {
                        var r = rows[i]
                        var who = (r.scope === "name" ? "all " + r.spans + " " : "one ") + r.category + ":" + r.name
                        out.push({"t": who, "r": r}, {"t": tab.fmt(r.workOnLongestNs)}, {"t": tab.fmt(r.modeledNs)},
                                 {"t": tab.fmt(r.replayNs)})
                    }
                    return out
                }
                delegate: Cell {
                    text: modelData.t
                    color: modelData.r && modelData.r.laneIndex >= 0 ? AppTheme.accent : AppTheme.text
                    font.pixelSize: AppTheme.typeCaption
                    font.underline: !!(modelData.r && modelData.r.laneIndex >= 0)
                    MouseArea {
                        anchors.fill: parent
                        enabled: !!(modelData.r && modelData.r.laneIndex >= 0)
                        cursorShape: enabled ? Qt.PointingHandCursor : Qt.ArrowCursor
                        onClicked: tab.selectEvent(modelData.r.laneIndex, modelData.r.spanIdx, modelData.r.startNs, 0)
                    }
                }
            }
        }
    }

    // ── Probabilities ──────────────────────────────────────────────
    Section {
        key: "probabilities"
        title: "Probabilities and calibration"
        summary: (tab.cps.probabilities || {}).calibrated ? "calibrated: " + tab.cps.probabilities.dataset : "unquantified"
        visible: tab.ready
        Text {
            objectName: "cpProbText"
            Layout.fillWidth: true
            Layout.minimumWidth: 0
            wrapMode: Text.Wrap
            color: AppTheme.text
            font.pixelSize: AppTheme.typeCaption
            text: {
                var p = tab.cps.probabilities || {}
                if (!p.calibrated) return "unquantified — " + (p.reason || "")
                return "calibrated · source " + (p.source || "—") + " · dataset " + p.dataset
                       + " · evidence version " + p.evidenceVersion + (p.versionMatches ? " (current)" : " (current: " + p.currentVersion + " — refit)")
                       + "\nmodel: " + p.model + " · " + p.strata + " strata from " + p.examples + " labelled candidates"
                       + "\ncoverage: " + p.edgesQuantified + " of " + p.edgesTotal + " graph edges quantified; the observed path's "
                       + p.quantifiedSteps + " quantified / " + p.unquantifiedSteps + " unquantified dependency steps, expected wrong "
                       + p.expectedWrongSteps.toFixed(2)
                       + (p.pAllQuantifiedRight !== null && p.pAllQuantifiedRight !== undefined
                          ? ", P(all quantified right) " + p.pAllQuantifiedRight.toFixed(3) + " (independence assumed)" : "")
            }
        }
    }

    Text {
        Layout.fillWidth: true
        wrapMode: Text.Wrap
        visible: tab.ready && (TimelineModel.criticalSummary.diagnostics || {}).receives !== undefined
        text: "MPI dependency coverage: " + ((TimelineModel.criticalSummary.diagnostics || {}).matched || 0)
              + " / " + ((TimelineModel.criticalSummary.diagnostics || {}).receives || 0)
              + " receive completions have a sender edge"
        color: AppTheme.text
        font.pixelSize: AppTheme.typeCaption
    }
    Repeater {
        model: tab.ready ? (tab.cps.notes || []) : []
        delegate: Note { text: "• " + modelData }
    }
}
