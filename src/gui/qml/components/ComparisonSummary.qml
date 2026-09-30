import QtQuick
import QtQuick.Layouts
import Hprofiler 1.0

// Side-by-side baseline/comparison headline fields, clearly labeled which
// is which (the comparison requirements' own "clearly identify the
// baseline and comparison profiles") -- reuses InspectorField.qml for
// each field's measured/derived/estimated/unavailable badge, same as
// every other field list in this GUI.
RowLayout {
    id: root
    property var baselineFields: []
    property var comparisonFields: []
    spacing: AppTheme.spacingLg

    ColumnLayout {
        Layout.fillWidth: true
        Layout.alignment: Qt.AlignTop
        spacing: AppTheme.spacingXs
        Text {
            text: "Baseline"
            color: AppTheme.textMuted
            font.bold: true
            font.pixelSize: AppTheme.typeLabel
        }
        Repeater {
            model: root.baselineFields
            delegate: InspectorField { field: modelData }
        }
    }

    Rectangle { Layout.preferredWidth: 1; Layout.fillHeight: true; color: AppTheme.panelBorder }

    ColumnLayout {
        Layout.fillWidth: true
        Layout.alignment: Qt.AlignTop
        spacing: AppTheme.spacingXs
        Text {
            text: "Comparison"
            color: AppTheme.textMuted
            font.bold: true
            font.pixelSize: AppTheme.typeLabel
        }
        Repeater {
            model: root.comparisonFields
            delegate: InspectorField { field: modelData }
        }
    }
}
