import QtQuick
import Hprofiler 1.0

// Shared "no data yet" message -- replaces 12 independently-styled
// empty-state Texts found across 8 screens (real drift: some centered
// via anchors inside a bare panel Rectangle, some left-aligned as a
// plain row inside a Column/ColumnLayout, some with no font.pixelSize
// override at all, manual "\n" vs wrapMode, one file uniquely using a
// monospace font). Deliberately does NOT self-position via
// `anchors.centerIn` here -- real usage is a genuine mix of
// Layout-managed placement (most instances: just another row in a
// Column/ColumnLayout, no anchors wanted or QtQuick logs an "anchors on
// a layout-managed item" warning) and anchor-based placement (a few:
// centered inside an otherwise-empty panel Rectangle). `centered: true`
// opts into the latter at the call site instead of baking one placement
// assumption into every use.
Text {
    property string message: ""
    property bool wrap: true
    property bool centered: false
    // Roofline's empty-state uniquely embeds a literal CLI command
    // example ("hprofiler roofline --backend <backend> -- ./app") --
    // monospace is a deliberate, genuine exception for that one case,
    // not left as a one-off custom Text outside this shared component.
    property bool monospace: false

    text: message
    anchors.centerIn: centered ? parent : undefined
    horizontalAlignment: Text.AlignHCenter
    color: AppTheme.textMuted
    font.family: monospace ? "monospace" : ""
    font.pixelSize: AppTheme.typeBody
    wrapMode: wrap ? Text.WordWrap : Text.NoWrap
    width: wrap && parent ? Math.min(implicitWidth, parent.width - AppTheme.spacingXl * 2) : implicitWidth
}
