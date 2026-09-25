import QtQuick
import QtQuick.Layouts
import Hprofiler 1.0
import "../components"

ColumnLayout {
    spacing: AppTheme.spacingLg

    Panel {
        Layout.fillWidth: true
        Layout.preferredHeight: 110
        title: "Run"
        ColumnLayout {
            anchors.fill: parent
            spacing: AppTheme.spacingXs
            Text { text: "Command:  " + System.command; color: AppTheme.text; font.pixelSize: AppTheme.typeBody }
            Text { text: "Host:  " + System.host; color: AppTheme.textMuted; font.pixelSize: AppTheme.typeBody }
            Text { text: "Duration:  " + System.duration; color: AppTheme.text; font.pixelSize: AppTheme.typeBody }
            Text { text: "Backends:  " + System.backends; color: AppTheme.textMuted; font.pixelSize: AppTheme.typeBody }
        }
    }

    Panel {
        Layout.fillWidth: true
        Layout.preferredHeight: Math.max(80, System.devices.length * 90)
        title: "Devices"
        ColumnLayout {
            anchors.fill: parent
            spacing: AppTheme.spacingLg
            Repeater {
                model: System.devices
                delegate: ColumnLayout {
                    Layout.fillWidth: true
                    spacing: AppTheme.spacingXs
                    Text {
                        text: modelData.backend.toUpperCase() + "  " + modelData.name +
                              "   cap " + modelData.computeCap + "   " + modelData.smCount + " SMs"
                        color: AppTheme.accent
                        font.bold: true
                        font.pixelSize: AppTheme.typeBody
                    }
                    Text { text: modelData.peaks; color: AppTheme.text; font.pixelSize: AppTheme.typeLabel }
                    Text {
                        text: [modelData.bandwidth, modelData.vram, modelData.ridgeHint]
                              .filter(function(x) { return x.length > 0 }).join("   ·   ")
                        color: AppTheme.textMuted
                        font.pixelSize: AppTheme.typeLabel
                    }
                }
            }
            EmptyState {
                visible: System.devices.length === 0
                message: "No device info captured."
            }
        }
    }

    Panel {
        Layout.fillWidth: true
        Layout.fillHeight: true
        title: "CPU"
        ColumnLayout {
            anchors.fill: parent
            spacing: AppTheme.spacingXs
            Text { visible: System.ipc > 0; text: "IPC:  " + System.ipc.toFixed(2); color: AppTheme.text; font.pixelSize: AppTheme.typeBody }
            Text { visible: System.cacheMissPct >= 0; text: "LLC miss rate:  " + System.cacheMissPct.toFixed(1) + "%"; color: AppTheme.text; font.pixelSize: AppTheme.typeBody }
            Text { visible: System.branchMissPct >= 0; text: "Branch miss:  " + System.branchMissPct.toFixed(1) + "%"; color: AppTheme.text; font.pixelSize: AppTheme.typeBody }
            Text { visible: System.peakRss.length > 0; text: "Peak RSS:  " + System.peakRss; color: AppTheme.text; font.pixelSize: AppTheme.typeBody }
            Item { Layout.fillHeight: true }
        }
    }
}
