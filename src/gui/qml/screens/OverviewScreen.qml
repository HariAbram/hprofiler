import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import Hprofiler 1.0
import "../components"

// Mirrors the TUI's DashboardWidget (src/ui/app.py) -- same underlying
// data (the Dashboard singleton, src/gui/bridge.py's DashboardBridge),
// different presentation. Wrapped in a ScrollView since the
// run-summary/breakdown/investigate-next sections exceed a typical
// window's height.
ScrollView {
    id: root
    clip: true
    ScrollBar.horizontal.policy: ScrollBar.AlwaysOff

    function gotoAction(action) {
        if (action.name && action.name.length > 0) Nav.selectFunction(action.category, action.name)
        Nav.navigateTo(action.tab)
    }

    ColumnLayout {
        width: root.availableWidth
        spacing: AppTheme.spacingLg

        // ── Run summary ──────────────────────────────────────────────────
        Panel {
            Layout.fillWidth: true
            Layout.preferredHeight: summaryFlow.implicitHeight + AppTheme.spacingMd * 2 +
                                     AppTheme.typeTitle + AppTheme.spacingSm
            title: "Run summary"

            Flow {
                id: summaryFlow
                width: parent.width
                spacing: AppTheme.spacingXl

                Repeater {
                    model: [
                        { label: "Executable", value: Dashboard.executable.length > 0 ? Dashboard.executable : "—" },
                        { label: "Backend", value: Dashboard.backends },
                        { label: "Host", value: Dashboard.host },
                        { label: "Device", value: Dashboard.devices.length > 0 ? Dashboard.devices.join(", ") : "none" },
                        { label: "Processes", value: String(Dashboard.processCount) },
                        { label: "Threads", value: String(Dashboard.threadCount) },
                        { label: "Duration", value: Dashboard.profilingDuration },
                        { label: "Captured", value: Dashboard.captureTime.length > 0 ? Dashboard.captureTime : "unavailable" },
                    ]
                    delegate: Column {
                        spacing: 2
                        Text {
                            text: modelData.label
                            color: AppTheme.textMuted
                            font.pixelSize: AppTheme.typeCaption
                        }
                        Text {
                            text: modelData.value
                            color: AppTheme.text
                            font.pixelSize: AppTheme.typeBody
                            elide: Text.ElideRight
                            width: Math.min(implicitWidth, 260)
                        }
                    }
                }
            }
        }

        // ── Capture warnings (only when the trace is incomplete or partly
        //    estimated: dropped/lost events, no final drain or GPU flush,
        //    proxy device timing) ─────────────────────────────────────────
        Panel {
            id: warningsPanel
            objectName: "captureWarningsPanel"
            visible: Dashboard.captureWarnings.length > 0
            Layout.fillWidth: true
            Layout.preferredHeight: visible ? warningsCol.implicitHeight + AppTheme.spacingMd * 2 +
                                              AppTheme.typeTitle + AppTheme.spacingSm : 0
            title: "Capture warnings"
            Accessible.name: "Capture warnings"
            Accessible.description: Dashboard.captureWarnings.join("\n")

            Column {
                id: warningsCol
                width: parent.width
                spacing: AppTheme.spacingXs

                Repeater {
                    model: Dashboard.captureWarnings
                    delegate: Row {
                        width: warningsCol.width
                        spacing: AppTheme.spacingSm
                        Text {
                            id: warnMark
                            text: "!"
                            font.bold: true
                            color: AppTheme.warningColor
                            font.pixelSize: AppTheme.typeBody
                        }
                        Text {
                            text: modelData
                            color: AppTheme.text
                            font.pixelSize: AppTheme.typeBody
                            wrapMode: Text.Wrap
                            width: warningsCol.width - warnMark.width - AppTheme.spacingSm
                        }
                    }
                }
            }
        }

        // ── Stat card row ──────────────────────────────────────────────
        RowLayout {
            Layout.fillWidth: true
            Layout.preferredHeight: AppTheme.statRowHeight
            Layout.maximumHeight: AppTheme.statRowHeight
            spacing: AppTheme.spacingLg

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
                label: "TOTAL RUNTIME"
                value: Dashboard.profilingDuration
            }
            StatCard {
                Layout.fillWidth: true
                Layout.fillHeight: true
                label: "CPU ACTIVE"
                value: Dashboard.cpuUtilPct.toFixed(0) + "%"
                valueColor: Dashboard.cpuUtilPct >= 70 ? AppTheme.severityColor("green")
                            : Dashboard.cpuUtilPct >= 40 ? AppTheme.severityColor("yellow")
                            : AppTheme.severityColor("red")
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

        // ── Time breakdown ────────────────────────────────────────────
        Panel {
            Layout.fillWidth: true
            Layout.preferredHeight: 150
            title: "Time breakdown"

            ColumnLayout {
                anchors.fill: parent
                spacing: AppTheme.spacingSm

                Item {
                    id: breakdownBar
                    Layout.fillWidth: true
                    Layout.preferredHeight: 18
                    visible: Dashboard.timeBreakdown.length > 0

                    Row {
                        anchors.fill: parent
                        Repeater {
                            model: Dashboard.timeBreakdown
                            delegate: Rectangle {
                                width: breakdownBar.width * modelData.pct / 100
                                height: breakdownBar.height
                                color: AppTheme.bucketColor(modelData.label)
                            }
                        }
                    }
                }

                Flow {
                    Layout.fillWidth: true
                    spacing: AppTheme.spacingMd
                    Repeater {
                        model: Dashboard.timeBreakdown
                        delegate: LegendSwatch {
                            circular: false
                            swatchColor: AppTheme.bucketColor(modelData.label)
                            label: modelData.label + "  " + modelData.pct.toFixed(1) + "%  (" + modelData.ns.toLocaleString() + " ns, derived)"
                        }
                    }
                }

                EmptyState {
                    Layout.fillWidth: true
                    visible: Dashboard.timeBreakdown.length === 0
                    message: "No timed spans recorded."
                }

                Item { Layout.fillHeight: true }

                Text {
                    Layout.fillWidth: true
                    text: "Profiling overhead: unavailable — " + Dashboard.profilingOverhead.reason
                    color: AppTheme.textMuted
                    font.pixelSize: AppTheme.typeCaption
                    font.italic: true
                    elide: Text.ElideRight
                }
            }
        }

        // ── Where to investigate next ───────────────────────────────────
        Panel {
            Layout.fillWidth: true
            Layout.preferredHeight: Math.max(70, investigateCol.implicitHeight + AppTheme.spacingMd * 2 +
                                              AppTheme.typeTitle + AppTheme.spacingSm)
            title: "Where to investigate next"

            ColumnLayout {
                id: investigateCol
                anchors.fill: parent
                spacing: 2

                Repeater {
                    model: Dashboard.investigateNext
                    delegate: Rectangle {
                        id: actionRow
                        Layout.fillWidth: true
                        Layout.preferredHeight: AppTheme.buttonHeight
                        color: actionMouse.containsMouse ? AppTheme.panelBorder : "transparent"
                        radius: AppTheme.radiusSmall

                        Accessible.role: Accessible.Button
                        Accessible.name: modelData.label
                        Accessible.description: "Navigates to " + modelData.label
                        Accessible.onPressAction: root.gotoAction(modelData)

                        RowLayout {
                            anchors.fill: parent
                            anchors.leftMargin: AppTheme.spacingSm
                            anchors.rightMargin: AppTheme.spacingSm
                            spacing: AppTheme.spacingSm

                            Text { text: "→"; color: AppTheme.accent; font.bold: true }
                            Text {
                                text: modelData.label
                                color: AppTheme.text
                                font.pixelSize: AppTheme.typeBody
                                Layout.fillWidth: true
                                elide: Text.ElideRight
                            }
                        }

                        MouseArea {
                            id: actionMouse
                            anchors.fill: parent
                            hoverEnabled: true
                            cursorShape: Qt.PointingHandCursor
                            onClicked: root.gotoAction(modelData)
                        }
                    }
                }

                EmptyState {
                    Layout.fillWidth: true
                    visible: Dashboard.investigateNext.length === 0
                    message: "Nothing stands out — this run looks balanced."
                }
            }
        }

        // ── 2x2 grid ──────────────────────────────────────────────────────
        GridLayout {
            Layout.fillWidth: true
            Layout.preferredHeight: 520
            columns: 2
            rows: 2
            columnSpacing: AppTheme.spacingLg
            rowSpacing: AppTheme.spacingLg

            // Execution timeline preview
            Panel {
                Layout.fillWidth: true
                Layout.fillHeight: true
                title: "Execution timeline"

                Column {
                    anchors.fill: parent
                    spacing: AppTheme.spacingSm

                    Repeater {
                        model: Dashboard.timelinePreview
                        delegate: RowLayout {
                            id: previewRow
                            width: parent.width
                            spacing: AppTheme.spacingSm
                            property string rowColor: modelData.color

                            Text {
                                text: modelData.category
                                color: previewRow.rowColor
                                font.pixelSize: AppTheme.typeLabel
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

                    EmptyState {
                        visible: Dashboard.timelinePreview.length === 0
                        message: "No timed spans recorded."
                    }
                }
            }

            // Top bottlenecks -- a real (if deliberately small, at most 4
            // rows: dash.top_findings() itself caps there) DataTable, not
            // bespoke styling, so this reuses the same sort/tooltip/copy
            // chrome as every other table for one consistent look.
            Panel {
                Layout.fillWidth: true
                Layout.fillHeight: true
                title: ""

                DataTable {
                    anchors.fill: parent
                    title: "Top bottlenecks"
                    table: Dashboard.findingsTable
                    compact: true
                    showFilterField: false
                    showColumnMenu: false
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
                        height: AppTheme.rowCompact
                        Rectangle { width: 8; height: 8; radius: 4; color: modelData.color }
                        Text {
                            text: modelData.name
                            color: AppTheme.text
                            font.pixelSize: AppTheme.typeBody
                            Layout.fillWidth: true
                            elide: Text.ElideRight
                        }
                        Text { text: modelData.calls; color: AppTheme.textMuted; font.pixelSize: AppTheme.typeBody }
                        Text {
                            text: modelData.total; color: AppTheme.textMuted; font.pixelSize: AppTheme.typeBody
                            Layout.preferredWidth: 70; horizontalAlignment: Text.AlignRight
                        }
                        Text {
                            text: modelData.share; color: AppTheme.text; font.pixelSize: AppTheme.typeBody
                            Layout.preferredWidth: 50; horizontalAlignment: Text.AlignRight
                        }
                    }
                }

                EmptyState {
                    centered: true
                    visible: Dashboard.hotKernels.length === 0
                    message: "No hot kernels recorded."
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
                    spacing: AppTheme.spacingXs

                    Text {
                        text: Dashboard.sourceDisplayName + "  ·  " + Dashboard.sourceHotspotName
                        color: AppTheme.accent
                        font.pixelSize: AppTheme.typeLabel
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
                                font.pixelSize: AppTheme.typeBody
                                width: 40
                            }
                            Text {
                                text: modelData.text
                                color: hot ? AppTheme.text : AppTheme.textMuted
                                font.family: "monospace"
                                font.pixelSize: AppTheme.typeBody
                                font.bold: hot
                            }
                        }
                    }
                }

                EmptyState {
                    centered: true
                    visible: !Dashboard.hasSourceContext
                    message: "No source correlation available — either the hottest " +
                             "function carries no file/line tag, or its source file " +
                             "isn't present on this machine."
                }
            }
        }
    }
}
