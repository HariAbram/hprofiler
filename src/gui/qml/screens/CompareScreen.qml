import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import Hprofiler 1.0
import "../components"

// Comparison mode (Phase C) -- always tab index 9 ("GUI tabs are always
// visible" convention, see DOCUMENTATION.md), backed by the "Compare"
// singleton (always registered, even with no comparison trace loaded --
// see comparison.py's own docstring on why). Empty state shows the exact
// CLI syntax to load one instead of a generic "nothing here" message.
Item {
    id: root

    ScreenState {
        objectName: "compareScreenState"
        anchors.fill: parent
        state: Compare.available ? "ready" : "empty"
        emptyMonospace: true
        emptyMessage: "No comparison trace loaded.\n\n" +
                 "hprofiler gui trace1.json --compare trace2.json\n\n" +
                 "trace1.json is the baseline, trace2.json is the comparison run."
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: AppTheme.spacingSm
        visible: Compare.available

        Toolbar {
            title: "Compare"
            statusText: Compare.newCount + " new · " + Compare.removedCount + " removed"

            ToolButton {
                text: "Export report"
                onClicked: {
                    var path = (AppInfo.tracePath || "compare") + ".compare.json"
                    exportStatus.text = Compare.exportReport(path) ? ("Exported to " + path) : "Export failed"
                }
            }
            ToolButton {
                text: "Export CSV"
                onClicked: {
                    var path = (AppInfo.tracePath || "compare") + ".compare.csv"
                    exportStatus.text = Compare.exportCsv(path) ? ("Exported to " + path) : "Export failed"
                }
            }
            Text {
                id: exportStatus
                color: AppTheme.textMuted
                font.pixelSize: AppTheme.typeCaption
            }
        }

        Flickable {
            Layout.fillWidth: true
            Layout.fillHeight: true
            contentWidth: width
            contentHeight: mainColumn.height
            clip: true
            boundsBehavior: Flickable.StopAtBounds

            ColumnLayout {
                id: mainColumn
                width: parent.width
                spacing: AppTheme.spacingMd

                Panel {
                    title: ""
                    Layout.fillWidth: true
                    Layout.preferredHeight: summary.implicitHeight + AppTheme.spacingMd * 2
                    ComparisonSummary {
                        id: summary
                        anchors.fill: parent
                        baselineFields: Compare.baselineFields
                        comparisonFields: Compare.comparisonFields
                    }
                }

                Panel {
                    title: "Legend & noise floor"
                    Layout.fillWidth: true
                    Layout.preferredHeight: legendColumn.implicitHeight + AppTheme.spacingMd * 2
                    ColumnLayout {
                        id: legendColumn
                        anchors.fill: parent
                        spacing: AppTheme.spacingSm

                        RowLayout {
                            spacing: AppTheme.spacingSm
                            Repeater {
                                model: Compare.statusLegend
                                delegate: ChangeBadge { status: modelData.status; text: modelData.label }
                            }
                        }
                        Text {
                            Layout.fillWidth: true
                            text: Compare.noiseFloor.note
                            color: AppTheme.textMuted
                            font.pixelSize: AppTheme.typeCaption
                            font.italic: true
                            wrapMode: Text.WordWrap
                        }
                        RowLayout {
                            spacing: AppTheme.spacingSm
                            Text {
                                text: "Thresholds: min Δ%"
                                color: AppTheme.textMuted
                                font.pixelSize: AppTheme.typeCaption
                            }
                            TextField {
                                id: minPctField
                                objectName: "compareMinPctField"
                                text: Compare.noiseFloor.pct.toFixed(0)
                                implicitWidth: 50
                                validator: DoubleValidator { bottom: 0 }
                                font.pixelSize: AppTheme.typeCaption
                                color: AppTheme.text
                            }
                            Text {
                                text: "min Δ time (ns)"
                                color: AppTheme.textMuted
                                font.pixelSize: AppTheme.typeCaption
                            }
                            TextField {
                                id: minNsField
                                objectName: "compareMinNsField"
                                text: Compare.noiseFloor.ns.toFixed(0)
                                implicitWidth: 90
                                validator: DoubleValidator { bottom: 0 }
                                font.pixelSize: AppTheme.typeCaption
                                color: AppTheme.text
                            }
                            ToolButton {
                                objectName: "compareApplyThresholdsButton"
                                text: "Apply"
                                onClicked: Compare.setChangeThresholds(
                                    parseFloat(minNsField.text) || 0, parseFloat(minPctField.text) || 0)
                            }
                        }
                    }
                }

                RowLayout {
                    Layout.fillWidth: true
                    spacing: AppTheme.spacingMd

                    Panel {
                        title: "Activity bucket deltas"
                        Layout.fillWidth: true
                        Layout.preferredHeight: 26 * Math.max(1, Compare.bucketDeltas.length) + AppTheme.spacingMd * 2 + 24
                        ColumnLayout {
                            anchors.fill: parent
                            spacing: 2
                            Repeater {
                                model: Compare.bucketDeltas
                                delegate: RowLayout {
                                    Layout.fillWidth: true
                                    Rectangle { width: 8; height: 8; radius: 4; color: AppTheme.bucketColor(modelData.bucket) }
                                    Text {
                                        text: modelData.bucket
                                        color: AppTheme.text
                                        font.pixelSize: AppTheme.typeCaption
                                        Layout.preferredWidth: 130
                                    }
                                    Text {
                                        text: Format.signedNs(modelData.deltaNs)
                                        color: AppTheme.textMuted
                                        font.pixelSize: AppTheme.typeCaption
                                        Layout.preferredWidth: 90
                                    }
                                    ChangeBadge { status: modelData.status }
                                    Item { Layout.fillWidth: true }
                                }
                            }
                        }
                    }

                    Panel {
                        title: "Execution coverage (each normalized to its own wall time)"
                        Layout.preferredWidth: 320
                        Layout.preferredHeight: 90
                        ColumnLayout {
                            anchors.fill: parent
                            spacing: AppTheme.spacingXs
                            Text { text: "Baseline"; color: AppTheme.textMuted; font.pixelSize: AppTheme.typeCaption }
                            Row {
                                Layout.fillWidth: true
                                height: 14
                                Repeater {
                                    model: Compare.baselineCoverage
                                    delegate: Rectangle {
                                        width: 4; height: 14
                                        color: modelData > 0 ? AppTheme.accent : "transparent"
                                        opacity: 0.3 + 0.7 * modelData
                                    }
                                }
                            }
                            Text { text: "Comparison"; color: AppTheme.textMuted; font.pixelSize: AppTheme.typeCaption }
                            Row {
                                Layout.fillWidth: true
                                height: 14
                                Repeater {
                                    model: Compare.comparisonCoverage
                                    delegate: Rectangle {
                                        width: 4; height: 14
                                        color: modelData > 0 ? AppTheme.warningColor : "transparent"
                                        opacity: 0.3 + 0.7 * modelData
                                    }
                                }
                            }
                        }
                    }
                }

                RowLayout {
                    Layout.fillWidth: true
                    spacing: AppTheme.spacingMd

                    Panel {
                        title: "Largest regressions"
                        Layout.fillWidth: true
                        Layout.preferredHeight: 22 * Math.max(1, Compare.topRegressions.length) + AppTheme.spacingMd * 2 + 24
                        ColumnLayout {
                            anchors.fill: parent
                            spacing: 1
                            Repeater {
                                model: Compare.topRegressions
                                delegate: RowLayout {
                                    Layout.fillWidth: true
                                    Text {
                                        text: modelData.name
                                        color: AppTheme.text
                                        font.pixelSize: AppTheme.typeCaption
                                        Layout.fillWidth: true
                                        elide: Text.ElideRight
                                    }
                                    Text {
                                        text: Format.signedNs(modelData.deltaNs)
                                        color: AppTheme.changeColor("regressed")
                                        font.pixelSize: AppTheme.typeCaption
                                    }
                                }
                            }
                        }
                    }
                    Panel {
                        title: "Largest improvements"
                        Layout.fillWidth: true
                        Layout.preferredHeight: 22 * Math.max(1, Compare.topImprovements.length) + AppTheme.spacingMd * 2 + 24
                        ColumnLayout {
                            anchors.fill: parent
                            spacing: 1
                            Repeater {
                                model: Compare.topImprovements
                                delegate: RowLayout {
                                    Layout.fillWidth: true
                                    Text {
                                        text: modelData.name
                                        color: AppTheme.text
                                        font.pixelSize: AppTheme.typeCaption
                                        Layout.fillWidth: true
                                        elide: Text.ElideRight
                                    }
                                    Text {
                                        text: Format.signedNs(modelData.deltaNs)
                                        color: AppTheme.changeColor("improved")
                                        font.pixelSize: AppTheme.typeCaption
                                    }
                                }
                            }
                        }
                    }
                }

                DataTable {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 420
                    title: "All matched kernels/functions"
                    table: Compare.table
                }
            }
        }
    }
}
