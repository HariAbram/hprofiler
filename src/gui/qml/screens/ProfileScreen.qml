import QtQuick
import QtQuick.Layouts
import Hprofiler 1.0
import "../components"

ColumnLayout {
    spacing: AppTheme.spacingLg

    RowLayout {
        Layout.fillWidth: true
        Layout.fillHeight: false
        Layout.preferredHeight: AppTheme.statRowHeight
        Layout.maximumHeight: AppTheme.statRowHeight
        visible: Profile.gpuActivity.length > 0
        spacing: AppTheme.spacingLg
        Repeater {
            model: Profile.gpuActivity
            delegate: Panel {
                Layout.fillWidth: true
                Layout.fillHeight: true
                title: modelData.label + " Activity"
                ColumnLayout {
                    anchors.fill: parent
                    spacing: AppTheme.spacingXs
                    Text {
                        text: modelData.activePct.toFixed(1) + "% active   " +
                              modelData.syncPct.toFixed(1) + "% sync   eff " + modelData.efficiency.toFixed(0) + "%"
                        color: modelData.color
                        font.bold: true
                        font.pixelSize: AppTheme.typeBody
                    }
                    Text {
                        text: modelData.launches + " launches  ·  " + modelData.total + " total"
                        color: AppTheme.textMuted
                        font.pixelSize: AppTheme.typeLabel
                    }
                }
            }
        }
    }

    RowLayout {
        Layout.fillWidth: true
        Layout.fillHeight: true
        spacing: AppTheme.spacingLg

        Panel {
            Layout.fillWidth: true
            Layout.fillHeight: true
            title: "Time breakdown"
            ColumnLayout {
                anchors.fill: parent
                spacing: AppTheme.spacingSm
                Repeater {
                    model: Profile.breakdown
                    delegate: ColumnLayout {
                        Layout.fillWidth: true
                        spacing: AppTheme.spacingXs
                        RowLayout {
                            Text { text: modelData.category; color: modelData.color; font.bold: true; font.pixelSize: AppTheme.typeBody; Layout.preferredWidth: 80 }
                            Text { text: modelData.pct.toFixed(1) + "%"; color: AppTheme.text; font.pixelSize: AppTheme.typeBody; Layout.preferredWidth: 50 }
                            Text { text: modelData.total; color: AppTheme.textMuted; font.pixelSize: AppTheme.typeLabel }
                        }
                        ProgressBar {
                            Layout.fillWidth: true
                            pct: modelData.pct
                            barColor: modelData.color
                        }
                    }
                }
                Item { Layout.fillHeight: true }
            }
        }

        Panel {
            Layout.fillWidth: true
            Layout.fillHeight: true
            title: "Insight"
            ColumnLayout {
                anchors.fill: parent
                spacing: AppTheme.spacingMd
                Repeater {
                    model: Profile.insight
                    delegate: RowLayout {
                        Layout.fillWidth: true
                        Text { text: modelData.icon; color: AppTheme.accent; font.bold: true }
                        Text {
                            text: modelData.text
                            color: AppTheme.text
                            font.pixelSize: AppTheme.typeBody
                            wrapMode: Text.WordWrap
                            Layout.fillWidth: true
                        }
                    }
                }
                EmptyState {
                    visible: Profile.insight.length === 0
                    message: "No actionable insight — looks balanced."
                }
                Item { Layout.fillHeight: true }
            }
        }
    }
}
