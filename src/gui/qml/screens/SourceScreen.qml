import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import Hprofiler 1.0
import "../components"

// GUI equivalent of the TUI's DisasmWidget -- split-pane disassembly
// viewer. Left: kernel list. Middle: annotated assembly (instruction
// type colored, matching disasm/classifier.py's scheme). Right:
// instruction-mix breakdown + static analysis hints (analysis/
// asm_advisor.py) -- both already existed and were already shown in the
// TUI (_show_mix/_show_hints); this screen just never called them.
RowLayout {
    id: root
    spacing: AppTheme.spacingMd

    property int selectedIndex: 0
    readonly property var selectedKernel: Source.kernels.length > selectedIndex
                                          ? Source.kernels[selectedIndex] : null
    readonly property bool hasSelection: !!selectedKernel && selectedKernel.hasDisasm
    // Computed once per kernel selection (not once per delegate/binding
    // evaluation) -- both are read from multiple places in the panel below.
    readonly property var mixData: hasSelection ? Source.instructionMix(selectedKernel.rawName) : []
    readonly property var hintsData: hasSelection ? Source.advisorHints(selectedKernel.rawName) : []

    Panel {
        Layout.preferredWidth: 260
        Layout.fillHeight: true
        clip: true

        ListView {
            anchors.fill: parent
            model: Source.kernels
            clip: true
            delegate: Rectangle {
                width: ListView.view.width
                // 40, not the shared row-height scale -- genuine 2-line
                // content (name + arch/total sub-line), not drift.
                height: 40
                color: root.selectedIndex === index ? AppTheme.panelBorder : "transparent"
                radius: AppTheme.radiusSmall

                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: AppTheme.spacingSm
                    spacing: AppTheme.spacingXs
                    RowLayout {
                        Text {
                            text: (modelData.hasDisasm ? "✓ " : "  ") + modelData.name
                            color: AppTheme.text
                            font.pixelSize: AppTheme.typeBody
                            elide: Text.ElideRight
                            Layout.fillWidth: true
                        }
                    }
                    Text {
                        text: modelData.arch + "  ·  " + modelData.total
                        color: AppTheme.textMuted
                        font.pixelSize: AppTheme.typeCaption
                    }
                }

                MouseArea {
                    anchors.fill: parent
                    onClicked: root.selectedIndex = index
                }
            }

            EmptyState {
                centered: true
                visible: Source.kernels.length === 0
                message: "No kernels profiled."
            }
        }
    }

    Panel {
        Layout.fillWidth: true
        Layout.fillHeight: true
        clip: true

        ColumnLayout {
            anchors.fill: parent
            spacing: AppTheme.spacingXs
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
                spacing: AppTheme.spacingMd
                visible: !!(root.selectedKernel && root.selectedKernel.symbol)
                Text {
                    text: "call site:"
                    color: AppTheme.textMuted
                    font.pixelSize: AppTheme.typeLabel
                }
                Text {
                    text: root.selectedKernel ? root.selectedKernel.symbol : ""
                    color: AppTheme.text
                    font.bold: true
                    font.pixelSize: AppTheme.typeLabel
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
                        font.pixelSize: AppTheme.typeCaption
                        Layout.topMargin: AppTheme.spacingXs
                        elide: Text.ElideLeft
                        Layout.fillWidth: true
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        Layout.preferredHeight: 18
                        Layout.maximumHeight: 18
                        spacing: AppTheme.spacingMd
                        Text {
                            text: modelData.addr
                            color: AppTheme.textMuted
                            font.family: "monospace"
                            font.pixelSize: AppTheme.typeLabel
                            Layout.preferredWidth: 60
                        }
                        Text {
                            text: modelData.mnemonic
                            color: modelData.color
                            font.family: "monospace"
                            font.bold: true
                            font.pixelSize: AppTheme.typeLabel
                            Layout.preferredWidth: 90
                        }
                        Text {
                            text: modelData.operands
                            color: AppTheme.text
                            font.family: "monospace"
                            font.pixelSize: AppTheme.typeLabel
                            Layout.fillWidth: true
                            elide: Text.ElideRight
                        }
                        Text {
                            visible: modelData.samplePct > 0
                            text: modelData.samplePct.toFixed(1) + "%"
                            // >=10% "hot": error-family red: this is a
                            // severity/attention signal (heavily sampled
                            // instruction), not literally an error, but
                            // reuses the same red/yellow "how much should
                            // this worry you" scale as everywhere else.
                            color: modelData.samplePct >= 10 ? AppTheme.errorColor
                                   : modelData.samplePct >= 1 ? AppTheme.warningColor
                                   : AppTheme.textMuted
                            font.pixelSize: AppTheme.typeCaption
                            Layout.preferredWidth: 40
                        }
                    }
                }
            }
        }

        ColumnLayout {
            anchors.centerIn: parent
            visible: !root.selectedKernel || !root.selectedKernel.hasDisasm
            spacing: AppTheme.spacingMd
            Text {
                Layout.alignment: Qt.AlignHCenter
                text: root.selectedKernel ? root.selectedKernel.name : ""
                color: AppTheme.text
                font.bold: true
            }
            EmptyState {
                Layout.maximumWidth: 500
                Layout.alignment: Qt.AlignHCenter
                message: root.selectedKernel ? Source.noDisasmReason(root.selectedKernel.rawName) : ""
            }
        }
    }

    // ── Analysis panel: instruction mix + static advisor hints ─────────────
    Panel {
        Layout.preferredWidth: 300
        Layout.fillHeight: true
        clip: true
        visible: root.hasSelection

        ScrollView {
            id: analysisScroll
            anchors.fill: parent
            clip: true
            ScrollBar.horizontal.policy: ScrollBar.AlwaysOff

            ColumnLayout {
                // NOT parent.width -- inside a ScrollView, the direct
                // child's "parent" is an internal Flickable whose own
                // width is sized to CONTENT, not the visible viewport
                // (that's the whole mechanism that lets it scroll) -- so
                // binding to it is circular and settles wider than the
                // panel, which is why wrapMode: Text.WordWrap below had
                // no effect (a Text never wraps until something gives it
                // a real bounded width to wrap AT). availableWidth is
                // ScrollView's own documented "content area, viewport-
                // bounded" property, made for exactly this.
                width: analysisScroll.availableWidth
                spacing: AppTheme.spacingLg

                Text {
                    text: "Instruction mix"
                    color: AppTheme.text
                    font.bold: true
                    font.pixelSize: AppTheme.typeBody
                }
                Text {
                    text: {
                        var total = 0
                        for (var i = 0; i < root.mixData.length; i++) total += root.mixData[i].count
                        return total + " instructions"
                    }
                    color: AppTheme.textMuted
                    font.pixelSize: AppTheme.typeCaption
                    Layout.bottomMargin: 2
                }
                Repeater {
                    model: root.mixData
                    delegate: ColumnLayout {
                        Layout.fillWidth: true
                        spacing: AppTheme.spacingXs
                        RowLayout {
                            Layout.fillWidth: true
                            Text {
                                text: modelData.label
                                color: modelData.color
                                font.pixelSize: AppTheme.typeLabel
                                Layout.fillWidth: true
                            }
                            Text {
                                text: modelData.count
                                color: AppTheme.text
                                font.pixelSize: AppTheme.typeLabel
                            }
                            Text {
                                text: modelData.pct.toFixed(0) + "%"
                                color: AppTheme.textMuted
                                font.pixelSize: AppTheme.typeCaption
                                Layout.preferredWidth: 30
                                horizontalAlignment: Text.AlignRight
                            }
                        }
                        ProgressBar {
                            Layout.fillWidth: true
                            pct: modelData.pct
                            barColor: modelData.color
                        }
                    }
                }
                EmptyState {
                    visible: root.mixData.length === 0
                    message: "No instruction data."
                }

                Rectangle { Layout.fillWidth: true; height: 1; color: AppTheme.panelBorder; Layout.topMargin: 4; Layout.bottomMargin: 4 }

                Text {
                    text: "Analysis"
                    color: AppTheme.text
                    font.bold: true
                    font.pixelSize: AppTheme.typeBody
                }
                Repeater {
                    model: root.hintsData
                    delegate: ColumnLayout {
                        Layout.fillWidth: true
                        Layout.bottomMargin: AppTheme.spacingSm
                        spacing: AppTheme.spacingXs
                        RowLayout {
                            spacing: AppTheme.spacingSm
                            Text { text: modelData.icon; color: modelData.color; font.pixelSize: AppTheme.typeLabel; font.bold: true }
                            Text { text: modelData.category; color: AppTheme.textMuted; font.pixelSize: AppTheme.typeCaption }
                        }
                        Text {
                            text: modelData.message
                            color: modelData.color
                            font.pixelSize: AppTheme.typeLabel
                            font.bold: true
                            wrapMode: Text.WordWrap
                            Layout.fillWidth: true
                        }
                        Text {
                            visible: modelData.detail.length > 0
                            text: modelData.detail
                            color: AppTheme.textMuted
                            font.pixelSize: AppTheme.typeCaption
                            wrapMode: Text.WordWrap
                            Layout.fillWidth: true
                        }
                    }
                }
                EmptyState {
                    Layout.fillWidth: true
                    visible: root.hasSelection && root.hintsData.length === 0
                    message: "No notable issues found in this function's assembly."
                }
            }
        }
    }
}
