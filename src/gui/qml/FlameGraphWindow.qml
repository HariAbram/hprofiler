import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import Hprofiler 1.0

// Standalone popup for `hprofiler flamegraph --gui` -- NOT part of the
// main 8-tab dashboard (Main.qml), since this window's data source is a
// single perf-collected folded-stacks tree (FlameGraphBridge), not a
// Trace object. The layout algorithm, color function, and interaction
// model (click = zoom in, right-click = zoom out, search = highlight,
// Esc = reset) are ported near-verbatim from output/flamegraph.py's
// existing, already-shipped HTML flame graph JS -- not redesigned from
// scratch, so this window behaves the same way `hprofiler flamegraph
// --html` already does, just natively.
ApplicationWindow {
    id: window
    width: 1280
    height: 800
    visible: true
    title: "hprofiler — " + FlameGraph.title
    color: AppTheme.background

    readonly property int frameH: 20
    readonly property int minPx: 1

    property var zoomStack: [FlameGraph.tree]
    readonly property var currentRoot: zoomStack[zoomStack.length - 1]
    property string searchText: ""
    property var hoveredFrame: null  // {node, x, y, w, depth} or null
    // Mouse position in the viewport's (not canvas-content's) coordinate
    // space -- i.e. already adjusted for Flickable scroll -- so the
    // tooltip floats correctly next to the cursor regardless of scroll
    // position. Tracked separately from hoveredFrame since hitTest()'s
    // result describes the FRAME under the cursor, not the cursor itself.
    property real hoverViewX: 0
    property real hoverViewY: 0

    function resetZoom() {
        zoomStack = [FlameGraph.tree]
    }
    function zoomUp() {
        if (zoomStack.length > 1)
            zoomStack = zoomStack.slice(0, zoomStack.length - 1)
    }
    function zoomInto(node) {
        if (node !== currentRoot)
            zoomStack = zoomStack.concat([node])
    }
    function treeDepth(node) {
        var m = 0
        for (var i = 0; i < node.children.length; i++)
            m = Math.max(m, 1 + treeDepth(node.children[i]))
        return m
    }
    // Same hash-to-warm-color function as flamegraph.py's colorFor() --
    // kept identical so the same function tends to land on the same
    // color whether you're looking at the HTML, TUI, or this window.
    function colorFor(name) {
        var h = 0
        for (var i = 0; i < name.length; i++)
            h = (Math.imul(h, 31) + name.charCodeAt(i)) | 0
        h = h >>> 0
        return [200 + (h & 0x37), 80 + ((h >> 6) & 0x5F), 20 + ((h >> 14) & 0x3F)]
    }
    function pctTotal(node) {
        return FlameGraph.totalSamples > 0
            ? (100 * node.value / FlameGraph.totalSamples).toFixed(1) : "0.0"
    }
    function pctView(node) {
        return currentRoot.value > 0
            ? (100 * node.value / currentRoot.value).toFixed(1) : "0.0"
    }
    function fmtCount(n) {
        return n.toLocaleString()
    }

    header: ColumnLayout {
        spacing: 0
        Rectangle {
            Layout.fillWidth: true
            height: 40
            color: AppTheme.surface

            RowLayout {
                anchors.fill: parent
                anchors.leftMargin: 12
                anchors.rightMargin: 12
                spacing: 8

                Text {
                    text: FlameGraph.title
                    color: AppTheme.text
                    font.bold: true
                    font.pixelSize: 13
                    elide: Text.ElideRight
                    Layout.maximumWidth: 400
                }
                TextField {
                    id: searchField
                    Layout.preferredWidth: 220
                    placeholderText: "Search (regex)…"
                    color: AppTheme.text
                    onTextChanged: window.searchText = text
                }
                Button {
                    text: "↑ Up"
                    enabled: window.zoomStack.length > 1
                    onClicked: window.zoomUp()
                }
                Button {
                    text: "⟲ Reset"
                    onClicked: window.resetZoom()
                }
                Item { Layout.fillWidth: true }
                Text {
                    text: window.hoveredFrame
                          ? (window.hoveredFrame.node.name + "  —  " +
                             window.fmtCount(window.hoveredFrame.node.value) + " samples  (" +
                             window.pctTotal(window.hoveredFrame.node) + "% total, " +
                             window.pctView(window.hoveredFrame.node) + "% of view)")
                          : window.fmtCount(FlameGraph.totalSamples) + " samples total"
                    color: AppTheme.textMuted
                    font.pixelSize: 11
                    elide: Text.ElideLeft
                    Layout.maximumWidth: 420
                }
                ToolButton {
                    text: AppTheme.dark ? "☀" : "☾"
                    font.pixelSize: 16
                    onClicked: AppTheme.toggle()
                }
            }
        }
        Rectangle {
            Layout.fillWidth: true
            height: 18
            color: AppTheme.surface
            Text {
                anchors.left: parent.left
                anchors.leftMargin: 12
                anchors.verticalCenter: parent.verticalCenter
                text: "Click: zoom in   ·   Right-click / Up: zoom out   ·   " +
                      "Esc: reset   ·   Backspace: zoom out   ·   Search supports regex"
                color: AppTheme.textMuted
                font.pixelSize: 10
            }
        }
    }

    Rectangle {
        anchors.fill: parent
        color: AppTheme.background
        // Keys.onPressed needs an Item, not a Window -- ApplicationWindow
        // itself isn't one (Window and Item are separate QML type
        // hierarchies), so keyboard handling lives here instead, on the
        // content root, with explicit focus since nothing grabs it by
        // default.
        focus: true

        Keys.onPressed: (event) => {
            if (event.key === Qt.Key_Escape) {
                window.resetZoom()
                event.accepted = true
            } else if (event.key === Qt.Key_Backspace && !searchField.activeFocus) {
                window.zoomUp()
                event.accepted = true
            }
        }

        Text {
            anchors.centerIn: parent
            visible: FlameGraph.totalSamples === 0
            text: "No stacks collected."
            color: AppTheme.textMuted
        }

        Flickable {
            id: flick
            anchors.fill: parent
            contentWidth: canvas.width
            contentHeight: canvas.height
            clip: true
            boundsBehavior: Flickable.StopAtBounds

            Canvas {
                id: canvas
                property var frames: []
                width: Math.max(flick.width, 200)
                height: (window.treeDepth(window.currentRoot) + 1) * window.frameH + 4

                Connections {
                    target: window
                    function onSearchTextChanged() { canvas.requestPaint() }
                    function onZoomStackChanged() { canvas.requestPaint() }
                }

                onPaint: {
                    var ctx = getContext("2d")
                    ctx.reset()
                    ctx.font = "11px monospace"
                    var root = window.currentRoot
                    var H = height
                    var W = width

                    var searchRe = null
                    if (window.searchText.length > 0) {
                        try { searchRe = new RegExp(window.searchText, "i") }
                        catch (e) { searchRe = null }
                    }

                    var built = []
                    function walk(node, x, w, depth) {
                        if (w < window.minPx) return
                        built.push({ node: node, x: x, depth: depth, w: w, y: 0 })
                        var cx = x
                        for (var i = 0; i < node.children.length; i++) {
                            var child = node.children[i]
                            var cw = node.value > 0 ? w * child.value / node.value : 0
                            walk(child, cx, cw, depth + 1)
                            cx += cw
                        }
                    }
                    walk(root, 0, W, 0)

                    for (var i = 0; i < built.length; i++) {
                        var f = built[i]
                        var y = H - (f.depth + 1) * window.frameH
                        f.y = y

                        var rgb = window.colorFor(f.node.name)
                        var r = rgb[0], g = rgb[1], b = rgb[2]
                        if (searchRe) {
                            if (searchRe.test(f.node.name)) {
                                r = 255; g = 210; b = 20
                            } else {
                                r = Math.round(r * 0.22)
                                g = Math.round(g * 0.22)
                                b = Math.round(b * 0.22)
                            }
                        }
                        ctx.fillStyle = "rgb(" + r + "," + g + "," + b + ")"
                        ctx.fillRect(f.x, f.y, Math.max(f.w - 0.5, 0), window.frameH - 1)

                        if (f.w > 32) {
                            ctx.fillStyle = "#111111"
                            var maxCh = Math.floor((f.w - 6) / 6.5)
                            var lbl = f.node.name
                            if (lbl.length > maxCh) lbl = lbl.slice(0, Math.max(maxCh - 1, 0)) + "…"
                            ctx.fillText(lbl, f.x + 3, f.y + window.frameH - 5)
                        }
                    }
                    frames = built
                }

                function hitTest(mx, my) {
                    for (var i = frames.length - 1; i >= 0; i--) {
                        var f = frames[i]
                        if (mx >= f.x && mx < f.x + f.w && my >= f.y && my < f.y + window.frameH - 1)
                            return f
                    }
                    return null
                }

                MouseArea {
                    anchors.fill: parent
                    hoverEnabled: true
                    acceptedButtons: Qt.LeftButton | Qt.RightButton

                    onPositionChanged: (mouse) => {
                        window.hoveredFrame = canvas.hitTest(mouse.x, mouse.y)
                        window.hoverViewX = mouse.x - flick.contentX
                        window.hoverViewY = mouse.y - flick.contentY
                    }
                    onExited: window.hoveredFrame = null
                    onClicked: (mouse) => {
                        var f = canvas.hitTest(mouse.x, mouse.y)
                        if (!f) return
                        if (mouse.button === Qt.RightButton) {
                            window.zoomUp()
                        } else {
                            window.zoomInto(f.node)
                        }
                    }
                }
            }
        }

        // Tooltip -- follows the cursor, clamped to stay within the window.
        Rectangle {
            visible: !!window.hoveredFrame
            color: "#000000"
            opacity: 0.92
            radius: 5
            border.color: "#444444"
            border.width: 1
            width: tooltipCol.width + 22
            height: tooltipCol.height + 14
            x: Math.min(window.hoverViewX + 14, parent.width - width - 10)
            y: Math.max(window.hoverViewY - height - 10, 0)
            z: 100

            ColumnLayout {
                id: tooltipCol
                anchors.centerIn: parent
                spacing: 3
                Text {
                    text: window.hoveredFrame ? window.hoveredFrame.node.name : ""
                    color: "#eeeeee"
                    font.family: "monospace"
                    font.pixelSize: 12
                    font.bold: true
                }
                Text {
                    text: window.hoveredFrame
                          ? (window.fmtCount(window.hoveredFrame.node.value) + " samples")
                          : ""
                    color: "#cccccc"
                    font.family: "monospace"
                    font.pixelSize: 11
                }
                Text {
                    text: window.hoveredFrame
                          ? (window.pctTotal(window.hoveredFrame.node) + "% of total    " +
                             window.pctView(window.hoveredFrame.node) + "% of view")
                          : ""
                    color: "#cccccc"
                    font.family: "monospace"
                    font.pixelSize: 11
                }
            }
        }
    }
}
