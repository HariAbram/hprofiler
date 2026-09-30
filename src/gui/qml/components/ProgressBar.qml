import QtQuick
import Hprofiler 1.0

// Shared horizontal percentage bar -- replaces two independently-sized
// implementations (SourceScreen's instruction-mix bar at height 5/
// radius 2, ProfileScreen's breakdown bar at height 6/radius 3; same
// widget, two ad hoc scales, no shared component).
//
// `indeterminate: true` (added for the async-loading overlay, where a
// real total isn't always known up front -- see loader.py's
// progress(0, 0) convention) swaps the fixed-fraction fill for a short
// segment sweeping back and forth, the standard "working, but no ETA"
// treatment -- same visual language QtQuick.Controls' own indeterminate
// ProgressBar uses, reimplemented here (not that stock control) only so
// it stays visually consistent with this bar's determinate mode.
Rectangle {
    id: root
    property real pct: 0        // 0-100, ignored when indeterminate
    property color barColor: AppTheme.accent
    property int barHeight: 6
    property bool indeterminate: false

    height: barHeight
    radius: AppTheme.radiusSmall
    color: AppTheme.panelBorder
    clip: true

    Rectangle {
        id: fill
        anchors.top: parent.top
        anchors.bottom: parent.bottom
        radius: AppTheme.radiusSmall
        color: root.barColor

        anchors.left: root.indeterminate ? undefined : parent.left
        width: root.indeterminate ? root.width * 0.28
                                   : root.width * Math.max(0, Math.min(100, root.pct)) / 100

        SequentialAnimation on x {
            running: root.indeterminate
            loops: Animation.Infinite
            NumberAnimation { from: 0; to: root.width - fill.width; duration: 900; easing.type: Easing.InOutQuad }
            NumberAnimation { from: root.width - fill.width; to: 0; duration: 900; easing.type: Easing.InOutQuad }
        }
    }
}
