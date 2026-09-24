import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import Hprofiler 1.0

// Flame Graph tab -- proportional-width icicle chart of FlameGraph.tree
// (src/gui/bridge.py's FlameGraphBridge, itself analysis/
// flamegraph_tree.py's build_flame_tree() -- the SAME _ct_build tree the
// Call Tree screen shows, just rendered as an icicle instead of an
// indented list, so the two screens can never disagree about the
// underlying call structure). The layout algorithm, and the click-zoom/
// search interaction model, are adapted from the now-removed standalone
// `hprofiler flamegraph --gui` popup's FlameGraphWindow.qml -- ported,
// not copy-pasted blind, applying every lesson that window needed two
// rounds of real-screenshot bug reports to find:
//   - the canvas must be BOTTOM-anchored (y: Math.max(0, viewport height
//     - content height)) or a shallow tree renders stuck at the top with
//     dead space below instead of the root sitting at the actual bottom;
//   - once the canvas has a nonzero y offset, hover/tooltip position
//     tracking MUST add canvas.y back in (mouse.y is canvas-LOCAL) or the
//     tooltip lands at the wrong height;
//   - the tooltip's name Text needs a bounded Layout.maximumWidth +
//     wrapMode, or a long (C++ template) name balloons the tooltip box
//     and gets clamped off-screen;
//   - never name a property "data" (collides with Item's own default
//     property).
// Color comes directly from the bridge (theme.categoryColor, resolved in
// Python) instead of the popup's own per-function hash palette -- ties
// this screen's color language to the same one Call Tree/Timeline/every
// other screen already uses, rather than a fourth, inconsistent scheme.
ColumnLayout {
    id: root
    spacing: 6

    readonly property int frameH: 20
    readonly property int minPx: 1
    property var zoomStack: [FlameGraph.tree]
    readonly property var currentRoot: zoomStack.length > 0 ? zoomStack[zoomStack.length - 1] : FlameGraph.tree
    property string searchText: ""
    property var hoveredFrame: null  // {node, x, y, w, depth} or null
    property real hoverViewX: 0
    property real hoverViewY: 0

    function resetZoom() { zoomStack = [FlameGraph.tree] }
    function zoomUp() { if (zoomStack.length > 1) zoomStack = zoomStack.slice(0, zoomStack.length - 1) }
    function zoomInto(node) { if (node !== currentRoot) zoomStack = zoomStack.concat([node]) }
    function treeDepth(node) {
        var m = 0
        for (var i = 0; i < node.children.length; i++)
            m = Math.max(m, 1 + treeDepth(node.children[i]))
        return m
    }
    function pctTotal(node) {
        return FlameGraph.totalNs > 0
            ? (100 * node.value / FlameGraph.totalNs).toFixed(1) : "0.0"
    }
    function fmtNs(ns) {
        if (ns >= 1e9) return (ns / 1e9).toFixed(3) + "s"
        if (ns >= 1e6) return (ns / 1e6).toFixed(2) + "ms"
        if (ns >= 1e3) return (ns / 1e3).toFixed(1) + "µs"
        return ns.toFixed(0) + "ns"
    }

    RowLayout {
        Layout.fillWidth: true
        spacing: 8
        Text {
            text: "Flame Graph"
            color: AppTheme.accent
            font.bold: true
            font.pixelSize: 13
        }
        Item { Layout.fillWidth: true }
        TextField {
            Layout.preferredWidth: 220
            placeholderText: "search (regex)…"
            font.pixelSize: 11
            onTextChanged: root.searchText = text
        }
        Button {
            text: "↑ Up"
            font.pixelSize: 11
            enabled: root.zoomStack.length > 1
            onClicked: root.zoomUp()
        }
        Button {
            text: "⟲ Reset"
            font.pixelSize: 11
            onClicked: root.resetZoom()
        }
        Text {
            text: root.hoveredFrame
                  ? (root.hoveredFrame.node.name + "  —  " + root.fmtNs(root.hoveredFrame.node.value) +
                     "  (" + root.pctTotal(root.hoveredFrame.node) + "% of total)")
                  : root.fmtNs(FlameGraph.totalNs) + " total"
            color: AppTheme.textMuted
            font.pixelSize: 11
            elide: Text.ElideLeft
            Layout.maximumWidth: 320
        }
    }

    Rectangle {
        Layout.fillWidth: true
        Layout.fillHeight: true
        color: AppTheme.surface
        border.color: AppTheme.panelBorder
        border.width: 1
        radius: 6
        clip: true

        Text {
            anchors.centerIn: parent
            visible: FlameGraph.totalNs === 0
            text: "No call-stack data in this trace.\nRun with --perf-callgraph fp|dwarf|lbr (or --call-tree) to capture it."
            horizontalAlignment: Text.AlignHCenter
            color: AppTheme.textMuted
            z: 5
        }

        Flickable {
            id: flick
            anchors.fill: parent
            anchors.margins: 4
            contentWidth: canvas.width
            contentHeight: canvas.height
            clip: true
            boundsBehavior: Flickable.StopAtBounds

            Canvas {
                id: canvas
                objectName: "flameGraphCanvas"
                property var frames: []
                width: Math.max(flick.width, 200)
                height: (root.treeDepth(root.currentRoot) + 1) * root.frameH + 4
                // Bottom-anchor when the tree is shallower than the
                // viewport (the common case) -- see this file's header
                // comment for why this is required, not cosmetic.
                y: Math.max(0, flick.height - height)

                Connections {
                    target: root
                    function onSearchTextChanged() { canvas.requestPaint() }
                    function onZoomStackChanged() { canvas.requestPaint() }
                }

                onPaint: {
                    var ctx = getContext("2d")
                    ctx.reset()
                    ctx.font = "11px monospace"
                    var rootNode = root.currentRoot
                    var H = height, W = width

                    var searchRe = null
                    if (root.searchText.length > 0) {
                        try { searchRe = new RegExp(root.searchText, "i") }
                        catch (e) { searchRe = null }
                    }

                    var built = []
                    function walk(node, x, w, depth) {
                        if (w < root.minPx) return
                        built.push({ node: node, x: x, depth: depth, w: w, y: 0 })
                        var cx = x
                        for (var i = 0; i < node.children.length; i++) {
                            var child = node.children[i]
                            var cw = node.value > 0 ? w * child.value / node.value : 0
                            walk(child, cx, cw, depth + 1)
                            cx += cw
                        }
                    }
                    walk(rootNode, 0, W, 0)

                    for (var i = 0; i < built.length; i++) {
                        var f = built[i]
                        var y = H - (f.depth + 1) * root.frameH
                        f.y = y

                        var matched = !searchRe || searchRe.test(f.node.name)
                        ctx.fillStyle = matched ? f.node.color : "#2a2a2a"
                        // Plain fillRect, not roundedRect -- QML's Canvas
                        // 2D context doesn't implement that method
                        // (throws and silently aborts the rest of the
                        // paint call -- see the Timeline call-graph
                        // panel's own history with this exact bug).
                        ctx.fillRect(f.x, f.y, Math.max(f.w - 0.5, 0), root.frameH - 1)

                        if (f.w > 32) {
                            ctx.fillStyle = matched ? "#111111" : "#888888"
                            var maxCh = Math.floor((f.w - 6) / 6.5)
                            var lbl = f.node.name
                            if (lbl.length > maxCh) lbl = lbl.slice(0, Math.max(maxCh - 1, 0)) + "…"
                            ctx.fillText(lbl, f.x + 3, f.y + root.frameH - 5)
                        }
                    }
                    frames = built
                }

                function hitTest(mx, my) {
                    for (var i = frames.length - 1; i >= 0; i--) {
                        var f = frames[i]
                        if (mx >= f.x && mx < f.x + f.w && my >= f.y && my < f.y + root.frameH - 1)
                            return f
                    }
                    return null
                }

                MouseArea {
                    anchors.fill: parent
                    hoverEnabled: true
                    acceptedButtons: Qt.LeftButton | Qt.RightButton

                    onPositionChanged: (mouse) => {
                        root.hoveredFrame = canvas.hitTest(mouse.x, mouse.y)
                        root.hoverViewX = mouse.x - flick.contentX
                        // + canvas.y: mouse.y is canvas-LOCAL, canvas.y is
                        // nonzero whenever bottom-anchored -- see header
                        // comment. Omitting this reproduces the exact
                        // wrong-tooltip-position bug the standalone popup
                        // had after its own bottom-anchoring fix.
                        root.hoverViewY = mouse.y + canvas.y - flick.contentY
                    }
                    onExited: root.hoveredFrame = null
                    onClicked: (mouse) => {
                        var f = canvas.hitTest(mouse.x, mouse.y)
                        if (!f) return
                        if (mouse.button === Qt.RightButton) {
                            root.zoomUp()
                        } else {
                            root.zoomInto(f.node)
                        }
                    }
                }
            }
        }

        // Tooltip -- follows the cursor, clamped to stay within the panel.
        Rectangle {
            visible: !!root.hoveredFrame
            color: "#000000"
            opacity: 0.92
            radius: 5
            border.color: "#444444"
            border.width: 1
            width: tooltipCol.width + 22
            height: tooltipCol.height + 14
            x: Math.min(root.hoverViewX + 14, parent.width - width - 10)
            y: Math.max(root.hoverViewY - height - 10, 0)
            z: 100

            ColumnLayout {
                id: tooltipCol
                anchors.centerIn: parent
                spacing: 3
                Text {
                    // Bounded width + wrap so a long (C++ template) name
                    // can't balloon this tooltip past the panel's own
                    // width -- see this file's header comment.
                    Layout.maximumWidth: 420
                    text: root.hoveredFrame ? root.hoveredFrame.node.name : ""
                    color: "#eeeeee"
                    font.family: "monospace"
                    font.pixelSize: 12
                    font.bold: true
                    wrapMode: Text.WrapAnywhere
                }
                Text {
                    text: root.hoveredFrame ? root.fmtNs(root.hoveredFrame.node.value) : ""
                    color: "#cccccc"
                    font.family: "monospace"
                    font.pixelSize: 11
                }
                Text {
                    text: root.hoveredFrame ? (root.pctTotal(root.hoveredFrame.node) + "% of total") : ""
                    color: "#cccccc"
                    font.pixelSize: 11
                }
            }
        }
    }
}
