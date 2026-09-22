import QtQuick
import QtQuick.Layouts
import Hprofiler 1.0
import "../components"

ColumnLayout {
    spacing: 10

    Panel {
        Layout.fillWidth: true
        Layout.preferredHeight: 110
        title: "Run"
        ColumnLayout {
            anchors.fill: parent
            spacing: 2
            Text { text: "Command:  " + System.command; color: AppTheme.text; font.pixelSize: 12 }
            Text { text: "Host:  " + System.host; color: AppTheme.textMuted; font.pixelSize: 12 }
            Text { text: "Duration:  " + System.duration; color: AppTheme.text; font.pixelSize: 12 }
            Text { text: "Backends:  " + System.backends; color: AppTheme.textMuted; font.pixelSize: 12 }
        }
    }

    Panel {
        Layout.fillWidth: true
        Layout.preferredHeight: Math.max(80, System.devices.length * 90)
        title: "Devices"
        ColumnLayout {
            anchors.fill: parent
            spacing: 10
            Repeater {
                model: System.devices
                delegate: ColumnLayout {
                    Layout.fillWidth: true
                    spacing: 1
                    Text {
                        text: modelData.backend.toUpperCase() + "  " + modelData.name +
                              "   cap " + modelData.computeCap + "   " + modelData.smCount + " SMs"
                        color: AppTheme.accent
                        font.bold: true
                        font.pixelSize: 12
                    }
                    Text { text: modelData.peaks; color: AppTheme.text; font.pixelSize: 11 }
                    Text {
                        text: [modelData.bandwidth, modelData.vram, modelData.ridgeHint]
                              .filter(function(x) { return x.length > 0 }).join("   ·   ")
                        color: AppTheme.textMuted
                        font.pixelSize: 11
                    }
                }
            }
            Text {
                visible: System.devices.length === 0
                text: "No device info captured."
                color: AppTheme.textMuted
                font.pixelSize: 12
            }
        }
    }

    Panel {
        Layout.fillWidth: true
        Layout.fillHeight: true
        title: "CPU"
        ColumnLayout {
            anchors.fill: parent
            spacing: 4
            Text { visible: System.ipc > 0; text: "IPC:  " + System.ipc.toFixed(2); color: AppTheme.text; font.pixelSize: 12 }
            Text { visible: System.cacheMissPct >= 0; text: "LLC miss rate:  " + System.cacheMissPct.toFixed(1) + "%"; color: AppTheme.text; font.pixelSize: 12 }
            Text { visible: System.branchMissPct >= 0; text: "Branch miss:  " + System.branchMissPct.toFixed(1) + "%"; color: AppTheme.text; font.pixelSize: 12 }
            Text { visible: System.peakRss.length > 0; text: "Peak RSS:  " + System.peakRss; color: AppTheme.text; font.pixelSize: 12 }
            Item { Layout.fillHeight: true }
        }
    }
}
