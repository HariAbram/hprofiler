import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import Hprofiler 1.0
import "screens"

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
                anchors.leftMargin: 12
                anchors.rightMargin: 12

                Text {
                    text: "hprofiler"
                    color: AppTheme.text
                    font.bold: true
                    font.pixelSize: 14
                }
                Text {
                    text: AppInfo.commandLine
                    color: AppTheme.textMuted
                    font.pixelSize: 12
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

            TabButton { text: "1 Overview" }
            TabButton { text: "2 Timeline" }
            TabButton { text: "3 Kernels" }
            TabButton { text: "4 Call Tree" }
            TabButton { text: "5 Roofline" }
            TabButton { text: "6 Source" }
            TabButton { text: "7 System" }
            TabButton { text: "8 Profile" }
        }
    }

    // ── Tab content ──────────────────────────────────────────────────────
    StackLayout {
        anchors.fill: parent
        anchors.margins: 10
        currentIndex: tabBar.currentIndex

        OverviewScreen {}
        TimelineScreen {}
        KernelsScreen {}
        CallTreeScreen {}
        RooflineScreen {}
        SourceScreen {}
        SystemScreen {}
        ProfileScreen {}
    }

    // ── Footer ───────────────────────────────────────────────────────────
    footer: Rectangle {
        height: 24
        color: AppTheme.surface
        Text {
            anchors.left: parent.left
            anchors.verticalCenter: parent.verticalCenter
            anchors.leftMargin: 12
            text: AppInfo.tracePath
            color: AppTheme.textMuted
            font.pixelSize: 11
        }
    }
}
