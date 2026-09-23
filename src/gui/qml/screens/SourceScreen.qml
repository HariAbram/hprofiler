import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import Hprofiler 1.0

// GUI equivalent of the TUI's DisasmWidget -- split-pane disassembly
// viewer. Left: kernel list. Middle: annotated assembly (instruction
// type colored, matching disasm/classifier.py's scheme). Right:
// instruction-mix breakdown + static analysis hints (analysis/
// asm_advisor.py) -- both already existed and were already shown in the
// TUI (_show_mix/_show_hints); this screen just never called them.
RowLayout {
    id: root
    spacing: 8

    property int selectedIndex: 0
    readonly property var selectedKernel: Source.kernels.length > selectedIndex
                                          ? Source.kernels[selectedIndex] : null
    readonly property bool hasSelection: !!selectedKernel && selectedKernel.hasDisasm
    // Computed once per kernel selection (not once per delegate/binding
    // evaluation) -- both are read from multiple places in the panel below.
    readonly property var mixData: hasSelection ? Source.instructionMix(selectedKernel.rawName) : []
    readonly property var hintsData: hasSelection ? Source.advisorHints(selectedKernel.rawName) : []

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

        ColumnLayout {
            anchors.fill: parent
            anchors.margins: 8
            spacing: 4
            visible: root.selectedKernel && root.selectedKernel.hasDisasm

            // What function this actually is: the span list on the left
            // shows event labels hprofiler invents ("omp_barrier",
            // "MPI_Bcast") -- there's no ELF symbol by that name. This is
            // the real resolved call site that got disassembled: for an
            // OpenMP/MPI event, that's the function in YOUR OWN profiled
            // program that triggered it (the runtime library's own
            // implementation is never what's shown -- see
            // hooks/common/codeptr_resolve.h). Empty when nothing
            // resolved (e.g. plain perf-sampled-by-name CPU functions,
            // where the kernel list name already IS the real symbol).
            RowLayout {
                Layout.fillWidth: true
                Layout.preferredHeight: 18
                Layout.maximumHeight: 18
                spacing: 8
                visible: !!(root.selectedKernel && root.selectedKernel.symbol)
                Text {
                    text: "call site:"
                    color: AppTheme.textMuted
                    font.pixelSize: 11
                }
                Text {
                    text: root.selectedKernel ? root.selectedKernel.symbol : ""
                    color: AppTheme.text
                    font.bold: true
                    font.pixelSize: 11
                    elide: Text.ElideRight
                    Layout.fillWidth: true
                }
            }

            ListView {
                id: asmView
                Layout.fillWidth: true
                Layout.fillHeight: true
                clip: true
                model: root.selectedKernel && root.selectedKernel.hasDisasm
                       ? Source.disasmLines(root.selectedKernel.rawName) : []
                delegate: ColumnLayout {
                    width: ListView.view.width
                    spacing: 0

                    // Source correlation: shown once per source line (not
                    // once per instruction) via sourceChanged, computed
                    // Python-side in SourceBridge.disasmLines(). Empty
                    // whenever the binary has no debug info for this
                    // function, or addr2line/llvm-symbolizer isn't
                    // installed -- silently absent, not an error.
                    Text {
                        visible: modelData.sourceChanged
                        text: "// " + modelData.sourceFile + ":" + modelData.sourceLine
                        color: AppTheme.textMuted
                        font.family: "monospace"
                        font.italic: true
                        font.pixelSize: 10
                        Layout.topMargin: 4
                        elide: Text.ElideLeft
                        Layout.fillWidth: true
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        Layout.preferredHeight: 18
                        Layout.maximumHeight: 18
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

    // ── Analysis panel: instruction mix + static advisor hints ─────────────
    Rectangle {
        Layout.preferredWidth: 300
        Layout.fillHeight: true
        color: AppTheme.surface
        border.color: AppTheme.panelBorder
        border.width: 1
        radius: 6
        clip: true
        visible: root.hasSelection

        ScrollView {
            anchors.fill: parent
            anchors.margins: 8
            clip: true
            ScrollBar.horizontal.policy: ScrollBar.AlwaysOff

            ColumnLayout {
                width: parent.width
                spacing: 10

                Text {
                    text: "Instruction mix"
                    color: AppTheme.text
                    font.bold: true
                    font.pixelSize: 12
                }
                Text {
                    text: {
                        var total = 0
                        for (var i = 0; i < root.mixData.length; i++) total += root.mixData[i].count
                        return total + " instructions"
                    }
                    color: AppTheme.textMuted
                    font.pixelSize: 10
                    Layout.bottomMargin: 2
                }
                Repeater {
                    model: root.mixData
                    delegate: ColumnLayout {
                        Layout.fillWidth: true
                        spacing: 1
                        RowLayout {
                            Layout.fillWidth: true
                            Text {
                                text: modelData.label
                                color: modelData.color
                                font.pixelSize: 11
                                Layout.fillWidth: true
                            }
                            Text {
                                text: modelData.count
                                color: AppTheme.text
                                font.pixelSize: 11
                            }
                            Text {
                                text: modelData.pct.toFixed(0) + "%"
                                color: AppTheme.textMuted
                                font.pixelSize: 10
                                Layout.preferredWidth: 30
                                horizontalAlignment: Text.AlignRight
                            }
                        }
                        Rectangle {
                            Layout.fillWidth: true
                            height: 5
                            radius: 2
                            color: AppTheme.background
                            Rectangle {
                                width: parent.width * modelData.pct / 100
                                height: parent.height
                                radius: 2
                                color: modelData.color
                            }
                        }
                    }
                }
                Text {
                    visible: root.mixData.length === 0
                    text: "No instruction data."
                    color: AppTheme.textMuted
                    font.pixelSize: 11
                }

                Rectangle { Layout.fillWidth: true; height: 1; color: AppTheme.panelBorder; Layout.topMargin: 4; Layout.bottomMargin: 4 }

                Text {
                    text: "Analysis"
                    color: AppTheme.text
                    font.bold: true
                    font.pixelSize: 12
                }
                Repeater {
                    model: root.hintsData
                    delegate: ColumnLayout {
                        Layout.fillWidth: true
                        Layout.bottomMargin: 6
                        spacing: 2
                        RowLayout {
                            spacing: 6
                            Text { text: modelData.icon; color: modelData.color; font.pixelSize: 11; font.bold: true }
                            Text { text: modelData.category; color: AppTheme.textMuted; font.pixelSize: 10 }
                        }
                        Text {
                            text: modelData.message
                            color: modelData.color
                            font.pixelSize: 11
                            font.bold: true
                            wrapMode: Text.WordWrap
                            Layout.fillWidth: true
                        }
                        Text {
                            visible: modelData.detail.length > 0
                            text: modelData.detail
                            color: AppTheme.textMuted
                            font.pixelSize: 10
                            wrapMode: Text.WordWrap
                            Layout.fillWidth: true
                        }
                    }
                }
                Text {
                    visible: root.hasSelection && root.hintsData.length === 0
                    text: "No notable issues found in this function's assembly."
                    color: AppTheme.textMuted
                    font.pixelSize: 11
                    wrapMode: Text.WordWrap
                    Layout.fillWidth: true
                }
            }
        }
    }
}
