import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import Hprofiler 1.0

// Collapsible right-side details panel -- the one place every tab's
// selection ends up rendered, so "kernel -> its timeline events/call
// path/source/roofline data" etc. all resolve to "look over here"
// instead of six bespoke per-screen detail views. Backed by Nav
// (src/gui/nav.py, current selection + breadcrumbs) and Inspector
// (src/gui/inspector.py, the Summary/Context/Metrics/Relationships/
// Recommendations content computed FROM that selection).
Rectangle {
    id: root
    color: AppTheme.surface
    border.color: AppTheme.panelBorder
    border.width: 1
    radius: AppTheme.radiusPanel
    // Driven by Nav.inspectorOpen, not a local toggle -- so any future
    // "open the inspector" action elsewhere in the GUI (not just this
    // panel's own button) can drive the same collapse state.
    implicitWidth: Nav.inspectorOpen ? 320 : 36

    Behavior on implicitWidth { NumberAnimation { duration: 120 } }

    // ── Collapsed: a thin strip with just a reopen button ──────────────
    ToolButton {
        visible: !Nav.inspectorOpen
        anchors.top: parent.top
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.topMargin: AppTheme.spacingSm
        text: "◀"
        onClicked: Nav.toggleInspector()
        ToolTip.visible: hovered
        ToolTip.text: "Open inspector"
    }

    // ── Expanded content ────────────────────────────────────────────────
    ColumnLayout {
        visible: Nav.inspectorOpen
        anchors.fill: parent
        anchors.margins: AppTheme.spacingMd
        spacing: AppTheme.spacingSm
        clip: true

        RowLayout {
            Layout.fillWidth: true
            Text {
                text: "Inspector"
                color: AppTheme.accent
                font.bold: true
                font.pixelSize: AppTheme.typeTitle
                Layout.fillWidth: true
                elide: Text.ElideRight
            }
            ToolButton {
                text: "▶"
                implicitWidth: AppTheme.iconButtonWidth
                onClicked: Nav.toggleInspector()
                ToolTip.visible: hovered
                ToolTip.text: "Collapse inspector"
            }
        }

        RowLayout {
            Layout.fillWidth: true
            visible: Nav.breadcrumbs.length > 0
            spacing: AppTheme.spacingXs

            ToolButton {
                text: "← Back"
                implicitHeight: AppTheme.buttonHeight
                onClicked: Nav.goBack()
                ToolTip.visible: hovered
                ToolTip.text: "Return to the previous analytical context"
            }
            Text {
                text: Nav.breadcrumbs.length + (Nav.breadcrumbs.length === 1 ? " step back" : " steps back")
                color: AppTheme.textMuted
                font.pixelSize: AppTheme.typeCaption
                Layout.fillWidth: true
            }
        }

        ColumnLayout {
            Layout.fillWidth: true
            visible: Nav.selectedName.length > 0
            spacing: 2

            Text {
                text: Nav.selectedName
                color: AppTheme.text
                font.bold: true
                font.pixelSize: AppTheme.typeBody
                elide: Text.ElideRight
                Layout.fillWidth: true
            }
            Text {
                text: Nav.selectedCategory
                color: AppTheme.textMuted
                font.pixelSize: AppTheme.typeCaption
            }
        }

        RowLayout {
            Layout.fillWidth: true
            visible: Nav.selectedName.length > 0
            spacing: AppTheme.spacingXs

            ToolButton {
                text: "Copy"
                implicitHeight: AppTheme.buttonHeight
                onClicked: {
                    Inspector.copyToClipboard(JSON.stringify(Inspector.content, null, 2))
                    exportStatus.text = "Copied"
                }
                ToolTip.visible: hovered
                ToolTip.text: "Copy all inspector details as JSON"
            }
            ToolButton {
                text: "Export"
                implicitHeight: AppTheme.buttonHeight
                onClicked: {
                    var ok = Inspector.exportTo(AppInfo.tracePath + ".inspector.json")
                    exportStatus.text = ok ? "Exported" : "Export failed"
                }
                ToolTip.visible: hovered
                ToolTip.text: "Export inspector details as JSON next to the trace file"
            }
            Text {
                id: exportStatus
                color: AppTheme.textMuted
                font.pixelSize: AppTheme.typeCaption
                Layout.fillWidth: true
                elide: Text.ElideRight
            }
        }

        EmptyState {
            // NOT centered:true -- this is a direct child of the real
            // ColumnLayout above (Layout-managed placement), and
            // centered's anchors.centerIn would conflict with that (see
            // this component's own docstring on the two placement modes).
            Layout.fillWidth: true
            visible: Nav.selectedName.length === 0
            message: "Select a kernel, function, call-tree node, or timeline event to inspect it here."
        }

        ScrollView {
            id: detailScroll
            Layout.fillWidth: true
            Layout.fillHeight: true
            clip: true
            visible: Nav.selectedName.length > 0
            ScrollBar.horizontal.policy: ScrollBar.AlwaysOff

            ColumnLayout {
                // availableWidth, not parent.width -- see SourceScreen.qml's
                // analysisScroll for why (a ScrollView child's `parent` is
                // an internal content Flickable sized to CONTENT, not the
                // viewport, so binding to it directly never wraps/elides).
                width: detailScroll.availableWidth
                spacing: AppTheme.spacingMd

                InspectorSection { title: "Summary"; fields: Inspector.content.summary || [] }
                InspectorSection { title: "Context"; fields: Inspector.content.context || [] }
                InspectorSection { title: "Metrics"; fields: Inspector.content.metrics || [] }
                InspectorSection { title: "Relationships"; fields: Inspector.content.relationships || [] }
                InspectorSection { title: "Recommendations"; fields: Inspector.content.recommendations || [] }
            }
        }

        Connections {
            target: Nav
            function onSelectionChanged() { exportStatus.text = "" }
        }
    }
}
