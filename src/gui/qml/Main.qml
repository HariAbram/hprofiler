import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import Hprofiler 1.0
import "screens"
import "components"

ApplicationWindow {
    id: window
    width: 1280
    height: 800
    visible: true
    title: "hprofiler — " + AppInfo.commandLine
    color: AppTheme.background

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
                    text: AppTheme.dark ? "☀" : "☾"
                    font.pixelSize: 16
                    onClicked: AppTheme.toggle()
                    ToolTip.visible: hovered
                    ToolTip.text: "Toggle light/dark theme"
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
