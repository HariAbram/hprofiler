import QtQuick
import QtQuick.Layouts
import Hprofiler 1.0
import "../components"

// Mirrors the TUI's DashboardWidget (src/ui/app.py) -- same underlying
// data (Dashboard context property, from src/gui/bridge.py's
// DashboardBridge), different presentation.
ColumnLayout {
    spacing: 10

    // ── Stat card row ──────────────────────────────────────────────────
    RowLayout {
        Layout.fillWidth: true
        Layout.fillHeight: false
        Layout.preferredHeight: 76
        Layout.maximumHeight: 76
        spacing: 10

        StatCard {
            Layout.fillWidth: true
            Layout.fillHeight: true
            label: "DIAGNOSIS"
            value: Dashboard.diagnosisLabel
            valueColor: Dashboard.diagnosisColor
        }
        StatCard {
            Layout.fillWidth: true
            Layout.fillHeight: true
            label: "WALL TIME"
            value: Dashboard.wallTime
        }
        StatCard {
            Layout.fillWidth: true
            Layout.fillHeight: true
            label: "GPU ACTIVE"
            value: Dashboard.gpuActiveAvailable
                   ? Dashboard.gpuActivePct.toFixed(0) + "%" : "n/a"
            valueColor: Dashboard.gpuActiveAvailable
                        ? (Dashboard.gpuActivePct >= 70 ? AppTheme.severityColor("green")
                           : Dashboard.gpuActivePct >= 40 ? AppTheme.severityColor("yellow")
                           : AppTheme.severityColor("red"))
                        : AppTheme.textMuted
        }
        StatCard {
            Layout.fillWidth: true
            Layout.fillHeight: true
            label: Dashboard.waitLabel
            value: Dashboard.waitPct.toFixed(0) + "%"
            valueColor: Dashboard.waitPct >= 40 ? AppTheme.severityColor("red")
                        : Dashboard.waitPct >= 20 ? AppTheme.severityColor("yellow")
                        : AppTheme.severityColor("green")
        }
        StatCard {
            Layout.fillWidth: true
            Layout.fillHeight: true
            label: "PEAK MEMORY"
            value: Dashboard.peakMemory
        }
    }

    // ── 2x2 grid ──────────────────────────────────────────────────────
    GridLayout {
        Layout.fillWidth: true
        Layout.fillHeight: true
        columns: 2
        rows: 2
        columnSpacing: 10
        rowSpacing: 10

        // Execution timeline preview
        Panel {
            Layout.fillWidth: true
            Layout.fillHeight: true
            title: "Execution timeline"

            Column {
                anchors.fill: parent
                spacing: 6

                Repeater {
                    model: Dashboard.timelinePreview
                    delegate: RowLayout {
                        id: previewRow
                        width: parent.width
                        spacing: 6
                        property string rowColor: modelData.color

                        Text {
                            text: modelData.category
                            color: previewRow.rowColor
                            font.pixelSize: 11
                            font.bold: true
                            Layout.preferredWidth: 60
                        }
                        Row {
                            Layout.fillWidth: true
                            height: 14
                            Repeater {
                                model: modelData.coverage
                                delegate: Rectangle {
                                    width: 6
                                    height: 14
                                    color: modelData > 0.05 ? previewRow.rowColor : "transparent"
                                }
                            }
                        }
                    }
                }

                Text {
                    visible: Dashboard.timelinePreview.length === 0
                    text: "No timed spans recorded."
                    color: AppTheme.textMuted
                    font.pixelSize: 12
                }
            }
        }

        // Top findings
        Panel {
            Layout.fillWidth: true
            Layout.fillHeight: true
            title: "Top findings"

            Column {
                anchors.fill: parent
                spacing: 8

                Repeater {
                    model: Dashboard.findings
                    delegate: Column {
                        width: parent.width
                        spacing: 2
                        RowLayout {
                            width: parent.width
                            Text { text: modelData.icon; color: modelData.color; font.bold: true }
                            Text { text: modelData.title; color: AppTheme.text; font.bold: true }
                        }
                        Text {
                            text: modelData.metric
                            color: modelData.color
                            font.pixelSize: 12
                            leftPadding: 20
                        }
                    }
                }

                Text {
                    visible: Dashboard.findings.length === 0
                    text: "No actionable findings — looks balanced."
                    color: AppTheme.textMuted
                    font.pixelSize: 12
                }
            }
        }

        // Hot kernels
        Panel {
            Layout.fillWidth: true
            Layout.fillHeight: true
            title: "Hot kernels"

            ListView {
                anchors.fill: parent
                model: Dashboard.hotKernels
                clip: true
                delegate: RowLayout {
                    width: ListView.view.width
                    height: 24
                    Rectangle { width: 8; height: 8; radius: 4; color: modelData.color }
                    Text {
                        text: modelData.name
                        color: AppTheme.text
                        font.pixelSize: 12
                        Layout.fillWidth: true
                        elide: Text.ElideRight
                    }
                    Text { text: modelData.calls; color: AppTheme.textMuted; font.pixelSize: 12 }
                    Text {
                        text: modelData.total; color: AppTheme.textMuted; font.pixelSize: 12
                        Layout.preferredWidth: 70; horizontalAlignment: Text.AlignRight
                    }
                    Text {
                        text: modelData.share; color: AppTheme.text; font.pixelSize: 12
                        Layout.preferredWidth: 50; horizontalAlignment: Text.AlignRight
                    }
                }
            }
        }

        // Source correlation
        Panel {
            Layout.fillWidth: true
            Layout.fillHeight: true
            title: "Source correlation"

            ColumnLayout {
                anchors.fill: parent
                visible: Dashboard.hasSourceContext
                spacing: 4

                Text {
                    text: Dashboard.sourceDisplayName + "  ·  " + Dashboard.sourceHotspotName
                    color: AppTheme.accent
                    font.pixelSize: 11
                }
                ListView {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    model: Dashboard.sourceLines
                    clip: true
                    delegate: Row {
                        width: ListView.view.width
                        property bool hot: modelData.line === Dashboard.sourceHotLine
                        Text {
                            text: (hot ? "▶" : " ") + modelData.line
                            color: hot ? AppTheme.accent : AppTheme.textMuted
                            font.family: "monospace"
                            font.pixelSize: 12
                            width: 40
                        }
                        Text {
                            text: modelData.text
                            color: hot ? AppTheme.text : AppTheme.textMuted
                            font.family: "monospace"
                            font.pixelSize: 12
                            font.bold: hot
                        }
                    }
                }
            }

            Text {
                anchors.fill: parent
                visible: !Dashboard.hasSourceContext
                wrapMode: Text.WordWrap
                text: "No source correlation available — either the hottest " +
                      "function carries no file/line tag, or its source file " +
                      "isn't present on this machine."
                color: AppTheme.textMuted
                font.pixelSize: 12
            }
        }
    }
}
