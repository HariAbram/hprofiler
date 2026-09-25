import QtQuick
import Hprofiler 1.0

// Shared horizontal percentage bar -- replaces two independently-sized
// implementations (SourceScreen's instruction-mix bar at height 5/
// radius 2, ProfileScreen's breakdown bar at height 6/radius 3; same
// widget, two ad hoc scales, no shared component).
Rectangle {
    id: root
    property real pct: 0        // 0-100
    property color barColor: AppTheme.accent
    property int barHeight: 6

    height: barHeight
    radius: AppTheme.radiusSmall
    color: AppTheme.panelBorder

    Rectangle {
        anchors.left: parent.left
        anchors.top: parent.top
        anchors.bottom: parent.bottom
        width: parent.width * Math.max(0, Math.min(100, root.pct)) / 100
        radius: AppTheme.radiusSmall
        color: root.barColor
    }
}
