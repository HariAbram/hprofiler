import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import Hprofiler 1.0
import "."

// Ctrl+K command palette -- opens tabs, runs a small fixed set of
// commands, and locates profiler entities (functions/kernels), all
// through ONE filtered list. Deliberately reuses existing data/
// navigation rather than aggregating anything new: entity search is a
// client-side filter over Kernels.rows (already trace.aggregated_stats(),
// already a bounded few-hundred-row list -- see bridge.py's
// KernelsBridge), "jump to it" is the existing Nav.selectFunction() +
// Nav.navigateTo() contract every other cross-tab-selection path in this
// GUI already uses.
Popup {
    id: root
    objectName: "commandPalette"
    modal: true
    focus: true
    width: 480
    height: 420
    anchors.centerIn: parent ? Overlay.overlay : undefined
    padding: 0

    property string query: ""

    onOpened: {
        query = ""
        searchField.forceActiveFocus()
        resultsList.currentIndex = 0
    }

    readonly property var tabLabels: [
        "1 Overview", "2 Timeline", "3 Kernels", "4 Call Tree", "5 Flame Graph",
        "6 Roofline", "7 Source", "8 System", "9 Profile", "10 Compare",
    ]

    function tabEntries() {
        var out = []
        for (var i = 0; i < tabLabels.length; i++) {
            out.push({
                label: "Go to " + tabLabels[i],
                sub: "Tab",
                action: (function (idx) { return function () { Nav.navigateTo(idx) } })(i),
            })
        }
        return out
    }

    function commandEntries() {
        return [
            { label: "Toggle light / dark theme", sub: "Command", action: function () { AppTheme.toggle() } },
            { label: "Reset current view", sub: "Command", action: function () { Workspace.resetCurrentView(App.tracePath) } },
            { label: "Reset all UI settings", sub: "Command", action: function () { Workspace.resetAll() } },
            { label: "Open log file", sub: "Command", action: function () { App.openLogFile() } },
            { label: "Open a different profile…", sub: "Command", action: function () { openProfileDialog.open() } },
            { label: "Show keyboard shortcuts", sub: "Command", action: function () { shortcutsDialog.open() } },
        ]
    }

    // Kernels.rows: {name, rawName, category, ...} -- see bridge.py's
    // KernelsBridge. Matched only once the query is at least 2 chars,
    // same "don't show noise for a single keystroke" threshold this
    // GUI's other search fields already use.
    function entityEntries() {
        if (query.length < 2) return []
        var q = query.toLowerCase()
        var rows = Kernels.rows
        var out = []
        for (var i = 0; i < rows.length && out.length < 20; i++) {
            if (rows[i].name.toLowerCase().indexOf(q) !== -1) {
                out.push({
                    label: rows[i].name,
                    sub: "Kernel · " + rows[i].category,
                    action: (function (cat, name) {
                        return function () { Nav.selectFunction(cat, name); Nav.navigateTo(2) }
                    })(rows[i].category, rows[i].rawName),
                })
            }
        }
        return out
    }

    readonly property var filteredResults: {
        var q = query.toLowerCase()
        var out = []
        var tabs = tabEntries()
        var commands = commandEntries()
        for (var i = 0; i < tabs.length; i++)
            if (q === "" || tabs[i].label.toLowerCase().indexOf(q) !== -1) out.push(tabs[i])
        for (var j = 0; j < commands.length; j++)
            if (q === "" || commands[j].label.toLowerCase().indexOf(q) !== -1) out.push(commands[j])
        out = out.concat(entityEntries())
        return out
    }

    function run(entry) {
        if (!entry) return
        entry.action()
        root.close()
    }

    contentItem: ColumnLayout {
        spacing: 0

        TextField {
            id: searchField
            objectName: "commandPaletteField"
            Layout.fillWidth: true
            Layout.margins: AppTheme.spacingMd
            placeholderText: "Go to a tab, run a command, or find a function…"
            text: root.query
            onTextChanged: root.query = text
            Keys.onDownPressed: resultsList.incrementCurrentIndex()
            Keys.onUpPressed: resultsList.decrementCurrentIndex()
            Keys.onReturnPressed: root.run(root.filteredResults[resultsList.currentIndex])
            Keys.onEnterPressed: root.run(root.filteredResults[resultsList.currentIndex])
            Keys.onEscapePressed: root.close()
        }

        ListView {
            id: resultsList
            objectName: "commandPaletteResults"
            Layout.fillWidth: true
            Layout.fillHeight: true
            clip: true
            model: root.filteredResults
            currentIndex: 0
            highlightMoveDuration: 0

            delegate: Rectangle {
                width: ListView.view.width
                height: 36
                color: ListView.isCurrentItem ? AppTheme.panelBorder : "transparent"

                Accessible.role: Accessible.Button
                Accessible.name: modelData.label + ", " + modelData.sub
                Accessible.onPressAction: root.run(modelData)

                RowLayout {
                    anchors.fill: parent
                    anchors.leftMargin: AppTheme.spacingMd
                    anchors.rightMargin: AppTheme.spacingMd
                    Text {
                        text: modelData.label
                        color: AppTheme.text
                        font.pixelSize: AppTheme.typeBody
                        Layout.fillWidth: true
                        elide: Text.ElideRight
                    }
                    Text {
                        text: modelData.sub
                        color: AppTheme.textMuted
                        font.pixelSize: AppTheme.typeCaption
                    }
                }

                MouseArea {
                    anchors.fill: parent
                    hoverEnabled: true
                    onEntered: resultsList.currentIndex = index
                    onClicked: root.run(modelData)
                }
            }

            EmptyState {
                anchors.centerIn: parent
                centered: true
                visible: root.filteredResults.length === 0
                message: "No matches."
            }
        }
    }

    OpenProfileFileDialog {
        id: openProfileDialog
    }

    ShortcutsDialog {
        id: shortcutsDialog
    }
}
