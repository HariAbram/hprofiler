import QtQuick
import QtQuick.Layouts
import Hprofiler 1.0
import "../components"

ColumnLayout {
    spacing: 10

    RowLayout {
        Layout.fillWidth: true
        Layout.fillHeight: false
        Layout.preferredHeight: 90
        Layout.maximumHeight: 90
        visible: Profile.gpuActivity.length > 0
        spacing: 10
        Repeater {
            model: Profile.gpuActivity
            delegate: Panel {
                Layout.fillWidth: true
                Layout.fillHeight: true
                title: modelData.label + " Activity"
                ColumnLayout {
                    anchors.fill: parent
                    spacing: 2
                    Text {
                        text: modelData.activePct.toFixed(1) + "% active   " +
                              modelData.syncPct.toFixed(1) + "% sync   eff " + modelData.efficiency.toFixed(0) + "%"
                        color: modelData.color
                        font.bold: true
                        font.pixelSize: 12
                    }
                    Text {
                        text: modelData.launches + " launches  ·  " + modelData.total + " total"
                        color: AppTheme.textMuted
                        font.pixelSize: 11
                    }
                }
            }
        }
    }

    RowLayout {
        Layout.fillWidth: true
        Layout.fillHeight: true
        spacing: 10

        Panel {
            Layout.fillWidth: true
            Layout.fillHeight: true
            title: "Time breakdown"
            ColumnLayout {
                anchors.fill: parent
                spacing: 6
                Repeater {
                    model: Profile.breakdown
                    delegate: ColumnLayout {
                        Layout.fillWidth: true
                        spacing: 1
                        RowLayout {
                            Text { text: modelData.category; color: modelData.color; font.bold: true; font.pixelSize: 12; Layout.preferredWidth: 80 }
                            Text { text: modelData.pct.toFixed(1) + "%"; color: AppTheme.text; font.pixelSize: 12; Layout.preferredWidth: 50 }
                            Text { text: modelData.total; color: AppTheme.textMuted; font.pixelSize: 11 }
                        }
                        Rectangle {
                            Layout.fillWidth: true
                            height: 6
                            radius: 3
                            color: AppTheme.background
                            Rectangle {
                                width: parent.width * modelData.pct / 100
                                height: parent.height
                                radius: 3
                                color: modelData.color
                            }
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
                spacing: 8
                Repeater {
                    model: Profile.insight
                    delegate: RowLayout {
                        Layout.fillWidth: true
                        Text { text: modelData.icon; color: AppTheme.accent; font.bold: true }
                        Text {
                            text: modelData.text
                            color: AppTheme.text
                            font.pixelSize: 12
                            wrapMode: Text.WordWrap
                            Layout.fillWidth: true
                        }
                    }
                }
                Text {
                    visible: Profile.insight.length === 0
                    text: "No actionable insight — looks balanced."
                    color: AppTheme.textMuted
                    font.pixelSize: 12
                }
                Item { Layout.fillHeight: true }
            }
        }
    }
}
