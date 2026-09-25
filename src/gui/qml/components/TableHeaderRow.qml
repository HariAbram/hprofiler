import QtQuick
import QtQuick.Layouts
import Hprofiler 1.0

// Shared table column-header chrome (bold label row + divider beneath).
// The caller still places its own labeled Text cells (column sets differ
// genuinely per table -- this only standardizes the surrounding chrome,
// not a column-spec DSL). Extracted from KernelsScreen's hand-built
// header, the only true table-header implementation in the GUI so far,
// before a second table reimplements it independently.
ColumnLayout {
    default property alias cells: row.children
    spacing: AppTheme.spacingXs

    RowLayout {
        id: row
        Layout.fillWidth: true
        spacing: AppTheme.spacingSm
    }
    Rectangle {
        Layout.fillWidth: true
        height: 1
        color: AppTheme.panelBorder
    }
}
