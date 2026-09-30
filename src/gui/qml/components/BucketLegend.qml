import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import Hprofiler 1.0

// Activity-bucket color key for Timeline's "Color: Activity" mode --
// only meaningful (and only shown) while that mode is active, since
// "function" mode has too many distinct colors for a legend to be useful
// and "category" mode's colors already match the existing per-category
// convention used everywhere else in the GUI.
//
// Collapsible (persisted globally via Workspace.legendCollapsed, not
// per-tab -- there's only ever this one legend today, and a global flag
// is one fewer thing to key by trace/tab if a second legend is ever
// added later) -- collapsed to just a small disclosure toggle so it
// stays out of the way once a user already knows the color mapping.
RowLayout {
    id: root
    spacing: AppTheme.spacingSm

    // Annotation/Other excluded here: Annotation is excluded from
    // bucket_totals() (see activity_buckets.py's module docstring -- it
    // overlaps real work by design) so it never actually colors a span
    // in practice, and "Other" is a rare fallback, not a category a user
    // needs a permanent legend entry for.
    readonly property var buckets: [
        "Computation", "Communication", "Synchronization",
        "Memory transfer", "Runtime overhead", "Idle",
    ]

    ToolButton {
        objectName: "legendCollapseToggle"
        text: Workspace.legendCollapsed ? "▸ Legend" : "▾ Legend"
        font.pixelSize: AppTheme.typeCaption
        implicitHeight: 22
        onClicked: Workspace.setLegendCollapsed(!Workspace.legendCollapsed)
        ToolTip.visible: hovered
        ToolTip.text: Workspace.legendCollapsed ? "Show the activity-color legend" : "Hide the activity-color legend"
        Accessible.name: Workspace.legendCollapsed ? "Show legend" : "Hide legend"
        Accessible.description: "Toggles the activity-color legend below the Timeline"
    }

    Repeater {
        id: swatchRepeater
        objectName: "legendSwatchRepeater"
        model: root.buckets
        delegate: LegendSwatch {
            visible: !Workspace.legendCollapsed
            circular: false
            swatchColor: AppTheme.bucketColor(modelData)
            label: modelData
        }
    }
}
