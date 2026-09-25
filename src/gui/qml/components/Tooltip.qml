import QtQuick
import QtQuick.Layouts
import Hprofiler 1.0

// Shared hover-info box -- replaces two independent implementations
// (FlameGraphScreen's cursor-following tooltip, fully hardcoded hex, and
// RooflineScreen's fixed-corner hover box, already AppTheme-driven) with
// one component covering both via `followCursor`. Always theme-token-
// driven internally, which is itself the fix for the hardcoded-hex
// version's real bug: it never repainted on the light/dark toggle.
// (TimelineScreen's inline status-row hover text is a deliberately
// different, simpler design for that data-dense screen and is NOT
// folded into this component.)
Rectangle {
    id: root
    // View-relative cursor position, used only when followCursor is true.
    property real anchorX: 0
    property real anchorY: 0
    property bool followCursor: true
    default property alias content: contentCol.children

    color: AppTheme.background
    border.color: AppTheme.panelBorder
    border.width: 1
    radius: AppTheme.radiusSmall
    width: contentCol.implicitWidth + AppTheme.spacingLg
    height: contentCol.implicitHeight + AppTheme.spacingMd
    z: 100

    x: followCursor
        ? Math.min(anchorX + 14, (parent ? parent.width : width) - width - 10)
        : 8
    y: followCursor
        ? Math.max(anchorY - height - 10, 0)
        : 8

    ColumnLayout {
        id: contentCol
        anchors.centerIn: parent
        spacing: AppTheme.spacingXs
    }
}
