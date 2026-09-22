import QtQuick
import QtQuick.Layouts
import Hprofiler 1.0

// GUI equivalent of the TUI's DisasmWidget -- split-pane disassembly
// viewer. Left: kernel list. Right: annotated assembly (instruction
// type colored, matching disasm/classifier.py's scheme).
RowLayout {
    id: root
    spacing: 8

    property int selectedIndex: 0
    readonly property var selectedKernel: Source.kernels.length > selectedIndex
                                          ? Source.kernels[selectedIndex] : null

    Rectangle {
        Layout.preferredWidth: 260
        Layout.fillHeight: true
        color: AppTheme.surface
        border.color: AppTheme.panelBorder
        border.width: 1
        radius: 6
        clip: true

        ListView {
            anchors.fill: parent
            anchors.margins: 4
            model: Source.kernels
            clip: true
            delegate: Rectangle {
                width: ListView.view.width
                height: 40
                color: root.selectedIndex === index ? AppTheme.panelBorder : "transparent"
                radius: 4

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 6
                    spacing: 1
                    RowLayout {
                        Text {
                            text: (modelData.hasDisasm ? "✓ " : "  ") + modelData.name
                            color: AppTheme.text
                            font.pixelSize: 12
                            elide: Text.ElideRight
                            Layout.fillWidth: true
                        }
                    }
                    Text {
                        text: modelData.arch + "  ·  " + modelData.total
                        color: AppTheme.textMuted
                        font.pixelSize: 10
                    }
                }

                MouseArea {
                    anchors.fill: parent
                    onClicked: root.selectedIndex = index
                }
            }

            Text {
                anchors.centerIn: parent
                visible: Source.kernels.length === 0
                text: "No kernels profiled."
                color: AppTheme.textMuted
            }
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

        ListView {
            id: asmView
            anchors.fill: parent
            anchors.margins: 8
            clip: true
            model: root.selectedKernel && root.selectedKernel.hasDisasm
                   ? Source.disasmLines(root.selectedKernel.rawName) : []
            delegate: RowLayout {
                width: ListView.view.width
                height: 18
                spacing: 8
                Text {
                    text: modelData.addr
                    color: AppTheme.textMuted
                    font.family: "monospace"
                    font.pixelSize: 11
                    Layout.preferredWidth: 60
                }
                Text {
                    text: modelData.mnemonic
                    color: modelData.color
                    font.family: "monospace"
                    font.bold: true
                    font.pixelSize: 11
                    Layout.preferredWidth: 90
                }
                Text {
                    text: modelData.operands
                    color: AppTheme.text
                    font.family: "monospace"
                    font.pixelSize: 11
                    Layout.fillWidth: true
                    elide: Text.ElideRight
                }
                Text {
                    visible: modelData.samplePct > 0
                    text: modelData.samplePct.toFixed(1) + "%"
                    color: modelData.samplePct >= 10 ? "#f87171" : (modelData.samplePct >= 1 ? "#fbbf24" : AppTheme.textMuted)
                    font.pixelSize: 10
                    Layout.preferredWidth: 40
                }
            }
        }

        ColumnLayout {
            anchors.centerIn: parent
            visible: !root.selectedKernel || !root.selectedKernel.hasDisasm
            spacing: 8
            Text {
                Layout.alignment: Qt.AlignHCenter
                text: root.selectedKernel ? root.selectedKernel.name : ""
                color: AppTheme.text
                font.bold: true
            }
            Text {
                Layout.maximumWidth: 500
                wrapMode: Text.WordWrap
                horizontalAlignment: Text.AlignHCenter
                text: root.selectedKernel ? Source.noDisasmReason(root.selectedKernel.rawName) : ""
                color: AppTheme.textMuted
                font.pixelSize: 12
            }
        }
    }
}
