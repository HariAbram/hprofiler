import QtQuick
import QtQuick.Layouts
import Hprofiler 1.0

// Shared "colored swatch + explanatory label" row -- replaces two
// independent implementations (RooflineScreen's circular dots with 2
// hardcoded hex colors, TimelineScreen's token-driven rounded-square
// overlay swatch). `circular` preserves both shapes rather than forcing
// one: Timeline's square deliberately signals "this is an overlay on
// another lane," not a category, so it stays visually distinct from a
// true category-color dot.
RowLayout {
    id: root
    property color swatchColor: AppTheme.accent
    property string label: ""
    property int size: 10
    property bool circular: true

    spacing: AppTheme.spacingXs

    Rectangle {
        width: root.size
        height: root.size
        radius: root.circular ? root.size / 2 : AppTheme.radiusSmall / 2
        color: root.swatchColor
        opacity: root.circular ? 1.0 : 0.8
    }
    Text {
        text: root.label
        color: AppTheme.textMuted
        font.pixelSize: AppTheme.typeLabel
    }
}
