import QtQuick
import QtQuick.Layouts
import Hprofiler 1.0

ColumnLayout {
    spacing: 6

    RowLayout {
        Layout.fillWidth: true
        Text { text: "Roofline"; color: AppTheme.accent; font.bold: true; font.pixelSize: 13 }
        Item { Layout.fillWidth: true }
        Text {
            visible: Roofline.available
            text: Roofline.aiRangeLabel + "   " + Roofline.tflopsRangeLabel
            color: AppTheme.textMuted
            font.pixelSize: 11
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
            visible: !Roofline.available
            horizontalAlignment: Text.AlignHCenter
            text: "No roofline data for this trace -- needs GPU hardware-counter\n" +
                  "or disassembly-estimated kernel metrics. Try:\n\n" +
                  "  hprofiler roofline --backend <backend> -- ./app"
            color: AppTheme.textMuted
            font.family: "monospace"
            font.pixelSize: 12
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

                Rectangle {
                    visible: parent.hoverText.length > 0
                    color: AppTheme.background
                    border.color: AppTheme.panelBorder
                    radius: 4
                    width: hoverLabel.width + 12
                    height: hoverLabel.height + 8
                    x: 8; y: 8
                    Text {
                        id: hoverLabel
                        anchors.centerIn: parent
                        text: parent.parent.hoverText
                        color: AppTheme.text
                        font.pixelSize: 11
                    }
                }
            }
        }
    }

    RowLayout {
        Layout.fillWidth: true
        visible: Roofline.available
        spacing: 16
        Rectangle { width: 10; height: 10; radius: 5; color: "#22d3ee" }
        Text { text: "compute-bound"; color: AppTheme.textMuted; font.pixelSize: 11 }
        Rectangle { width: 10; height: 10; radius: 5; color: "#e879f9" }
        Text { text: "memory-bound"; color: AppTheme.textMuted; font.pixelSize: 11 }
        Item { Layout.fillWidth: true }
    }
}
