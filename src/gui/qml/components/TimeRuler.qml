import QtQuick
import Hprofiler 1.0

// Timeline's time axis (Phase B5) -- tick marks/labels (TimelineModel.
// timeTicks(), a "nice" round-number interval, same idea as any plotting
// library's axis ticks), named-range bands, bookmark markers, and the
// current Nav.selectedTimeRange band, all drawn on one Canvas so they
// share exactly the same x-position math as the lanes below (`scale`).
// Sits directly above the lanes area; the caller positions/offsets it by
// labelWidth, matching the connector overlay's own convention.
Item {
    id: root
    property real viewStartNs: 0
    property real visibleNs: 1
    property var bookmarks: []      // TimelineModel.bookmarks
    property var namedRanges: []    // TimelineModel.namedRanges
    property var selectedRange: ({})   // Nav.selectedTimeRange ({startNs,endNs} or {})

    signal bookmarkClicked(real ns)
    signal bookmarkRemoveRequested(int id)

    implicitHeight: 26

    function fmtNs(ns) {
        if (ns >= 1e9) return (ns / 1e9).toFixed(3) + "s"
        if (ns >= 1e6) return (ns / 1e6).toFixed(2) + "ms"
        if (ns >= 1e3) return (ns / 1e3).toFixed(1) + "µs"
        return ns.toFixed(0) + "ns"
    }

    property var _bookmarkHitboxes: []   // [{x, ns, id}], refreshed each paint

    Canvas {
        id: canvas
        objectName: "timeRulerCanvas"
        anchors.fill: parent

        onPaint: {
            var ctx = getContext("2d")
            ctx.reset()
            if (root.visibleNs <= 0 || width <= 0) return
            var scale = width / root.visibleNs
            var viewEndNs = root.viewStartNs + root.visibleNs

            // Named-range bands, drawn first so ticks/bookmarks paint on
            // top of them.
            ctx.fillStyle = AppTheme.infoColor
            for (var i = 0; i < root.namedRanges.length; i++) {
                var r = root.namedRanges[i]
                if (r.endNs <= root.viewStartNs || r.startNs >= viewEndNs) continue
                var rx0 = Math.max(0, (r.startNs - root.viewStartNs) * scale)
                var rx1 = Math.min(width, (r.endNs - root.viewStartNs) * scale)
                ctx.globalAlpha = 0.18
                ctx.fillRect(rx0, 0, Math.max(1, rx1 - rx0), height)
            }
            ctx.globalAlpha = 1.0

            // The live shift-drag selection / Nav.selectedTimeRange, drawn
            // distinctly (accent border) so it reads as "current", not just
            // another saved range.
            if (root.selectedRange && root.selectedRange.startNs !== undefined) {
                var s0 = Math.max(0, (root.selectedRange.startNs - root.viewStartNs) * scale)
                var s1 = Math.min(width, (root.selectedRange.endNs - root.viewStartNs) * scale)
                if (s1 > s0 && root.selectedRange.endNs > root.viewStartNs && root.selectedRange.startNs < viewEndNs) {
                    ctx.fillStyle = AppTheme.accent
                    ctx.globalAlpha = 0.15
                    ctx.fillRect(s0, 0, s1 - s0, height)
                    ctx.globalAlpha = 1.0
                    ctx.strokeStyle = AppTheme.accent
                    ctx.lineWidth = 1
                    ctx.strokeRect(s0 + 0.5, 0.5, Math.max(1, s1 - s0 - 1), height - 1)
                }
            }

            // Ticks + labels.
            var ticks = TimelineModel.timeTicks(root.viewStartNs, viewEndNs, Math.max(2, Math.floor(width / 90)))
            ctx.strokeStyle = AppTheme.panelBorder
            ctx.fillStyle = AppTheme.textMuted
            ctx.font = "10px sans-serif"
            ctx.textBaseline = "top"
            for (i = 0; i < ticks.length; i++) {
                var tx = (ticks[i].ns - root.viewStartNs) * scale
                ctx.beginPath()
                ctx.moveTo(tx + 0.5, height - 6)
                ctx.lineTo(tx + 0.5, height)
                ctx.stroke()
                ctx.fillText(ticks[i].label, tx + 3, 1)
            }

            // Bookmark markers -- small downward-pointing accent triangles
            // sitting on the baseline, tall enough to be a comfortable
            // click target despite the ruler's modest height.
            var hitboxes = []
            ctx.fillStyle = AppTheme.warningColor
            for (i = 0; i < root.bookmarks.length; i++) {
                var b = root.bookmarks[i]
                if (b.ns < root.viewStartNs || b.ns > viewEndNs) continue
                var bx = (b.ns - root.viewStartNs) * scale
                ctx.beginPath()
                ctx.moveTo(bx - 4, height)
                ctx.lineTo(bx + 4, height)
                ctx.lineTo(bx, height - 8)
                ctx.closePath()
                ctx.fill()
                hitboxes.push({x: bx, ns: b.ns, id: b.id, name: b.name})
            }
            root._bookmarkHitboxes = hitboxes
        }
    }

    onViewStartNsChanged: canvas.requestPaint()
    onVisibleNsChanged: canvas.requestPaint()
    onBookmarksChanged: canvas.requestPaint()
    onNamedRangesChanged: canvas.requestPaint()
    onSelectedRangeChanged: canvas.requestPaint()

    MouseArea {
        id: rulerMouse
        anchors.fill: parent
        hoverEnabled: true
        acceptedButtons: Qt.LeftButton | Qt.RightButton
        property var hoveredBookmark: null

        function _hit(mouseX, mouseY) {
            if (mouseY < root.height - 10) return null
            var boxes = root._bookmarkHitboxes
            for (var i = 0; i < boxes.length; i++) {
                if (Math.abs(boxes[i].x - mouseX) <= 5) return boxes[i]
            }
            return null
        }

        onPositionChanged: (mouse) => { rulerMouse.hoveredBookmark = _hit(mouse.x, mouse.y) }
        onExited: { rulerMouse.hoveredBookmark = null }
        onClicked: (mouse) => {
            var hit = _hit(mouse.x, mouse.y)
            if (!hit) return
            if (mouse.button === Qt.RightButton) root.bookmarkRemoveRequested(hit.id)
            else root.bookmarkClicked(hit.ns)
        }

        Rectangle {
            id: bookmarkTip
            visible: rulerMouse.hoveredBookmark !== null
            x: Math.max(0, (rulerMouse.hoveredBookmark ? rulerMouse.hoveredBookmark.x : 0) - width / 2)
            y: -22
            width: tipLabel.implicitWidth + 8
            height: 18
            radius: AppTheme.radiusSmall
            color: AppTheme.surface
            border.color: AppTheme.panelBorder
            border.width: 1
            Text {
                id: tipLabel
                anchors.centerIn: parent
                text: rulerMouse.hoveredBookmark ? rulerMouse.hoveredBookmark.name : ""
                color: AppTheme.text
                font.pixelSize: AppTheme.typeCaption
            }
        }
    }
}
