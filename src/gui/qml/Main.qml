import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Window
import Hprofiler 1.0
import "screens"
import "components"

ApplicationWindow {
    id: window
    x: Workspace.hasSavedGeometry ? Workspace.windowX : (Screen.width - width) / 2
    y: Workspace.hasSavedGeometry ? Workspace.windowY : (Screen.height - height) / 2
    width: Workspace.hasSavedGeometry ? Workspace.windowWidth : 1280
    height: Workspace.hasSavedGeometry ? Workspace.windowHeight : 800
    visibility: Workspace.hasSavedGeometry && Workspace.windowMaximized
                ? ApplicationWindow.Maximized : ApplicationWindow.Windowed
    visible: true
    title: "hprofiler — " + AppInfo.commandLine
    color: AppTheme.background

    // Persist geometry on every resize/move, and once more on close --
    // deliberately NOT throttled/debounced: these are cheap QSettings
    // writes (a handful of ints), not per-frame work, and saving on
    // every change (not just close) means a hard kill (SIGKILL, a crash)
    // still leaves the LAST known-good geometry on disk rather than
    // whatever was there from the session before.
    function saveGeometry() {
        Workspace.saveWindowGeometry(window.x, window.y, window.width, window.height,
                                      window.visibility === ApplicationWindow.Maximized)
    }
    onXChanged: saveGeometry()
    onYChanged: saveGeometry()
    onWidthChanged: saveGeometry()
    onHeightChanged: saveGeometry()
    onVisibilityChanged: saveGeometry()
    onClosing: saveGeometry()

    // ── Menu bar + global shortcuts ─────────────────────────────────────
    // Sequences read from Shortcuts (src/gui/shortcuts.py's SHORTCUTS
    // table), never re-typed here -- the Help > Keyboard Shortcuts
    // dialog renders that exact same table, so the two can never
    // disagree about what key does what.
    menuBar: MenuBar {
        Menu {
            title: "File"
            MenuItem {
                // "\t<sequence>" -- the standard Qt convention for a
                // right-aligned shortcut hint in a menu item; used
                // instead of MenuItem's own `shortcut` property, which
                // this build's QtQuick.Controls style doesn't expose
                // ("Cannot assign to non-existent property" at QML load
                // time, confirmed directly, not assumed) -- the actual
                // key handling still comes from the real Shortcut{}
                // items below, this is purely the visual hint text.
                text: "Open Profile…\t" + Shortcuts.openProfileSequence
                onTriggered: openProfileDialog.open()
            }
            MenuSeparator {}
            MenuItem {
                text: "Reset Current View"
                onTriggered: Workspace.resetCurrentView(App.tracePath)
            }
            MenuItem {
                text: "Reset All UI Settings…"
                onTriggered: resetAllConfirmDialog.open()
            }
        }
        Menu {
            title: "View"
            MenuItem {
                text: "Command Palette\t" + Shortcuts.commandPaletteSequence
                onTriggered: commandPalette.open()
            }
        }
        Menu {
            title: "Help"
            MenuItem {
                text: "Keyboard Shortcuts\t" + Shortcuts.shortcutsReferenceSequence
                onTriggered: shortcutsDialog.open()
            }
            MenuItem {
                text: "Open Log File"
                onTriggered: App.openLogFile()
            }
        }
    }

    Shortcut {
        sequence: Shortcuts.commandPaletteSequence
        onActivated: commandPalette.open()
    }
    Shortcut {
        sequence: Shortcuts.openProfileSequence
        onActivated: openProfileDialog.open()
    }
    Shortcut {
        sequence: Shortcuts.shortcutsReferenceSequence
        onActivated: shortcutsDialog.open()
    }

    CommandPalette {
        id: commandPalette
    }

    OpenProfileFileDialog {
        id: openProfileDialog
    }

    ShortcutsDialog {
        id: shortcutsDialog
    }

    Dialog {
        id: resetAllConfirmDialog
        objectName: "resetAllConfirmDialog"
        title: "Reset All UI Settings?"
        modal: true
        standardButtons: Dialog.Yes | Dialog.No
        anchors.centerIn: parent ? Overlay.overlay : undefined
        onAccepted: Workspace.resetAll()

        Text {
            text: "This clears window geometry, theme, table configuration,\n" +
                  "and every saved profile's filters/bookmarks/zoom state.\n" +
                  "This cannot be undone."
            color: AppTheme.textMuted
            font.pixelSize: AppTheme.typeLabel
            wrapMode: Text.WordWrap
        }
    }

    // ── Top bar: app name + command (left), tab strip (right below) ────
    header: ColumnLayout {
        spacing: 0

        Rectangle {
            Layout.fillWidth: true
            height: 32
            color: AppTheme.surface

            RowLayout {
                anchors.fill: parent
                anchors.leftMargin: AppTheme.spacingLg
                anchors.rightMargin: AppTheme.spacingLg

                Text {
                    text: "hprofiler"
                    color: AppTheme.text
                    font.bold: true
                    font.pixelSize: AppTheme.typeHeading
                }
                Text {
                    text: AppInfo.commandLine
                    color: AppTheme.textMuted
                    font.pixelSize: AppTheme.typeBody
                    Layout.fillWidth: true
                    elide: Text.ElideRight
                }
                ToolButton {
                    objectName: "themeToggleButton"
                    text: AppTheme.dark ? "☀" : "☾"
                    font.pixelSize: 16
                    onClicked: AppTheme.toggle()
                    ToolTip.visible: hovered
                    ToolTip.text: "Toggle light/dark theme"
                    Accessible.name: "Toggle theme"
                    Accessible.description: "Switches between light and dark color themes"
                }
            }
        }

        TabBar {
            id: tabBar
            objectName: "tabBar"
            Layout.fillWidth: true
            background: Rectangle { color: AppTheme.surface }

            // Two-way link to Nav.currentTab, NOT a plain one-way binding:
            // TabBar sets currentIndex itself when a TabButton is clicked,
            // which silently breaks a declarative `currentIndex:
            // Nav.currentTab` binding the first time that happens (a plain
            // write to a bound property tears the binding out in QML).
            // Re-asserting the value imperatively via Connections below
            // sidesteps that -- it never relies on the binding surviving
            // past the first click.
            currentIndex: Nav.currentTab
            onCurrentIndexChanged: {
                if (Nav.currentTab !== currentIndex) Nav.currentTab = currentIndex
            }
            Connections {
                target: Nav
                function onCurrentTabChanged() {
                    if (tabBar.currentIndex !== Nav.currentTab) tabBar.currentIndex = Nav.currentTab
                }
            }

            TabButton { text: "1 Overview" }
            TabButton { text: "2 Timeline" }
            TabButton { text: "3 Kernels" }
            TabButton { text: "4 Call Tree" }
            TabButton { text: "5 Flame Graph" }
            TabButton { text: "6 Roofline" }
            TabButton { text: "7 Source" }
            TabButton { text: "8 System" }
            TabButton { text: "9 Profile" }
            TabButton { text: "10 Compare" }
        }
    }

    // ── Tab content ──────────────────────────────────────────────────────
    // Each tab is behind a Loader, active only once selected (then stays
    // loaded, so revisiting a tab doesn't rebuild it) -- a StackLayout
    // with plain screen children instead builds and paints EVERY tab
    // eagerly at startup, including the Timeline's per-lane Canvases
    // (each doing a real Python round-trip via TimelineModel.visibleSpans
    // on first paint), which was pure wasted work for the 7 tabs the user
    // hasn't opened yet and a real, measured contributor to slow startup
    // on a large trace.
    RowLayout {
        anchors.fill: parent
        anchors.margins: AppTheme.spacingLg
        spacing: AppTheme.spacingLg

        StackLayout {
            id: stack
            Layout.fillWidth: true
            Layout.fillHeight: true
            // One-way is safe here (unlike TabBar above): nothing ever
            // assigns to StackLayout.currentIndex directly, so this
            // binding never gets torn out.
            currentIndex: Nav.currentTab

            Loader { objectName: "tabLoader0"; active: stack.currentIndex === 0 || item !== null; sourceComponent: overviewComp }
            Loader { objectName: "tabLoader1"; active: stack.currentIndex === 1 || item !== null; sourceComponent: timelineComp }
            Loader { objectName: "tabLoader2"; active: stack.currentIndex === 2 || item !== null; sourceComponent: kernelsComp }
            Loader { objectName: "tabLoader3"; active: stack.currentIndex === 3 || item !== null; sourceComponent: callTreeComp }
            Loader { objectName: "tabLoader4"; active: stack.currentIndex === 4 || item !== null; sourceComponent: flameGraphComp }
            Loader { objectName: "tabLoader5"; active: stack.currentIndex === 5 || item !== null; sourceComponent: rooflineComp }
            Loader { objectName: "tabLoader6"; active: stack.currentIndex === 6 || item !== null; sourceComponent: sourceComp }
            Loader { objectName: "tabLoader7"; active: stack.currentIndex === 7 || item !== null; sourceComponent: systemComp }
            Loader { objectName: "tabLoader8"; active: stack.currentIndex === 8 || item !== null; sourceComponent: profileComp }
            Loader { objectName: "tabLoader9"; active: stack.currentIndex === 9 || item !== null; sourceComponent: compareComp }
        }

        InspectorPanel {
            objectName: "inspectorPanel"
            Layout.fillHeight: true
        }
    }

    Component { id: overviewComp; OverviewScreen {} }
    Component { id: timelineComp; TimelineScreen {} }
    Component { id: kernelsComp; KernelsScreen {} }
    Component { id: callTreeComp; CallTreeScreen {} }
    Component { id: flameGraphComp; FlameGraphScreen {} }
    Component { id: rooflineComp; RooflineScreen {} }
    Component { id: sourceComp; SourceScreen {} }
    Component { id: systemComp; SystemScreen {} }
    Component { id: profileComp; ProfileScreen {} }
    Component { id: compareComp; CompareScreen {} }

    // ── "Open Profile" overlay ──────────────────────────────────────────
    // Non-blocking: this window's own content underneath is completely
    // untouched (never hidden, never a Loader gate) the entire time --
    // "Open Profile" always spawns a genuinely new OS process (see
    // controller.py's module docstring for why: a second Controls-
    // loading QQmlApplicationEngine corrupts Controls resolution in this
    // PySide6 build), and this window only ever closes itself once that
    // new process signals it's actually showing something. A failure
    // there leaves this workspace exactly as it was -- visually, not
    // just architecturally.
    Item {
        id: openProfileOverlay
        objectName: "openProfileOverlay"
        anchors.fill: parent
        z: 1000
        visible: state !== "idle"
        property string state: "idle"   // idle | opening | error
        property var lastError: ({})

        Rectangle {
            anchors.fill: parent
            color: AppTheme.scrimColor
            opacity: 0.55
        }

        Rectangle {
            anchors.centerIn: parent
            width: 480
            height: openProfileOverlay.state === "error" ? 320 : 160
            radius: AppTheme.radiusPanel
            color: AppTheme.surface
            border.color: AppTheme.panelBorder
            border.width: 1

            ScreenState {
                anchors.fill: parent
                anchors.margins: AppTheme.spacingLg
                state: openProfileOverlay.state === "opening" ? "loading"
                       : openProfileOverlay.state === "error" ? "error" : "ready"
                loadingMessage: "Opening new profile…"
                errorMessage: openProfileOverlay.lastError.message || ""
                errorDetail: openProfileOverlay.lastError.detail || ""
                errorTracebackText: openProfileOverlay.lastError.tracebackText || ""
                errorStage: openProfileOverlay.lastError.stage || ""
                errorFile: openProfileOverlay.lastError.file || ""
            }

            Button {
                objectName: "openProfileOverlayDismiss"
                visible: openProfileOverlay.state === "error"
                anchors.bottom: parent.bottom
                anchors.horizontalCenter: parent.horizontalCenter
                anchors.bottomMargin: AppTheme.spacingSm
                flat: true
                text: "Dismiss"
                onClicked: openProfileOverlay.state = "idle"
                Accessible.name: "Dismiss"
                Accessible.description: "Closes this error message; your current profile is unaffected"
            }
        }
    }

    Connections {
        target: App
        function onProfileOpening() { openProfileOverlay.state = "opening" }
        function onProfileOpenFailed(err) {
            openProfileOverlay.lastError = err
            openProfileOverlay.state = "error"
        }
    }

    // ── Footer ───────────────────────────────────────────────────────────
    footer: Rectangle {
        height: 24
        color: AppTheme.surface
        Text {
            anchors.left: parent.left
            anchors.verticalCenter: parent.verticalCenter
            anchors.leftMargin: AppTheme.spacingLg
            text: AppInfo.tracePath
            color: AppTheme.textMuted
            font.pixelSize: AppTheme.typeLabel
        }
    }
}
