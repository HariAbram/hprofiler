import QtQuick
import QtQuick.Layouts
import Hprofiler 1.0
import "../components"

ColumnLayout {
    spacing: AppTheme.spacingSm

    Toolbar {
        title: "Roofline"
        statusText: Roofline.available
            ? (Roofline.aiRangeLabel + "   " + Roofline.tflopsRangeLabel) : ""
    }

    Rectangle {
        Layout.fillWidth: true
        Layout.fillHeight: true
        color: AppTheme.surface
        border.color: AppTheme.panelBorder
        border.width: 1
        radius: AppTheme.radiusPanel
        clip: true

        ScreenState {
            objectName: "rooflineScreenState"
            anchors.fill: parent
            state: Roofline.available ? "ready" : "unsupported"
            unsupportedMonospace: true
            unsupportedMessage: "No roofline data for this trace -- needs GPU hardware-counter\n" +
                     "or disassembly-estimated kernel metrics. Try:\n\n" +
                     "  hprofiler roofline --backend <backend> -- ./app"
        }

        Item {
            anchors.fill: parent
            anchors.margins: 16
            visible: Roofline.available

            Canvas {
                id: canvas
                anchors.fill: parent
                property real margin: 8

                onPaint: {
                    var ctx = getContext("2d")
                    ctx.reset()
                    var pw = width - 2 * margin
                    var ph = height - 2 * margin

                    var lines = Roofline.rooflineLines
                    ctx.strokeStyle = AppTheme.textMuted
                    ctx.lineWidth = 1.5
                    for (var i = 0; i < lines.length; i++) {
                        var pts = lines[i].points
                        ctx.beginPath()
                        for (var j = 0; j < pts.length; j++) {
                            var px = margin + pts[j].x * pw
                            var py = margin + pts[j].y * ph
                            if (j === 0) ctx.moveTo(px, py)
                            else ctx.lineTo(px, py)
                        }
                        ctx.stroke()
                    }

                    var pts2 = Roofline.points
                    for (var k = 0; k < pts2.length; k++) {
                        var p = pts2[k]
                        var x = margin + p.x * pw
                        var y = margin + p.y * ph
                        ctx.fillStyle = p.color
                        ctx.beginPath()
                        ctx.arc(x, y, 4, 0, 2 * Math.PI)
                        ctx.fill()
                    }
                }

                Component.onCompleted: requestPaint()
            }

            MouseArea {
                id: hoverArea
                anchors.fill: parent
                hoverEnabled: true
                property string hoverText: ""
                onPositionChanged: (mouse) => {
                    var pw = width - 2 * canvas.margin
                    var ph = height - 2 * canvas.margin
                    var pts = Roofline.points
                    var found = null
                    for (var i = 0; i < pts.length; i++) {
                        var px = canvas.margin + pts[i].x * pw
                        var py = canvas.margin + pts[i].y * ph
                        if (Math.abs(mouse.x - px) < 6 && Math.abs(mouse.y - py) < 6) {
                            found = pts[i]
                            break
                        }
                    }
                    hoverText = found ? (found.name + "  AI=" + found.ai.toFixed(2) +
                                        "  " + found.tflops.toFixed(2) + " TFLOP/s  (" +
                                        found.bound + "-bound)") : ""
                }

                Tooltip {
                    visible: hoverArea.hoverText.length > 0
                    followCursor: false
                    Text {
                        text: hoverArea.hoverText
                        color: AppTheme.text
                        font.pixelSize: AppTheme.typeLabel
                    }
                }
            }
        }
    }

    RowLayout {
        Layout.fillWidth: true
        visible: Roofline.available
        spacing: AppTheme.spacingXl
        // Uses the cpu/rocm category colors, which match the dark-theme
        // values of the scatter points' colors -- those are fixed server-side
        // (bridge.py's RooflineBridge._BOUND_COLOR), independent of AppTheme.
        LegendSwatch { swatchColor: AppTheme.categoryColor("cpu"); label: "compute-bound" }
        LegendSwatch { swatchColor: AppTheme.categoryColor("rocm"); label: "memory-bound" }
        Item { Layout.fillWidth: true }
    }
}
