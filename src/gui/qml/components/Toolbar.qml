import QtQuick
import QtQuick.Layouts
import Hprofiler 1.0

// Shared screen-level control strip: optional title (section identity,
// e.g. "Flame Graph"), optional trailing status/stats readout, and a
// default-property slot for whatever interactive controls a screen
// needs (search fields, buttons) -- replaces 3 independently copy-
// pasted title-row recipes and 4 structurally different toolbar rows
// found across screens (Kernels/FlameGraph/Timeline/Roofline each
// re-decided field width, which properties got themed, and button
// styling on their own). A screen that doesn't need a title (most of
// them -- the TabBar already names the tab) simply doesn't set one,
// rather than every screen growing one just for consistency's sake.
RowLayout {
    id: root
    property string title: ""
    property string statusText: ""
    default property alias controls: controlsRow.children

    Layout.fillWidth: true
    spacing: AppTheme.spacingSm

    Text {
        visible: root.title.length > 0
        text: root.title
        color: AppTheme.accent
        font.bold: true
        font.pixelSize: AppTheme.typeTitle
    }

    Item { Layout.fillWidth: true }

    RowLayout {
        id: controlsRow
        spacing: AppTheme.spacingSm
    }

    Text {
        visible: root.statusText.length > 0
        text: root.statusText
        color: AppTheme.textMuted
        font.pixelSize: AppTheme.typeLabel
    }
}
