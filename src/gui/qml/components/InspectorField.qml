import QtQuick
import QtQuick.Layouts
import Hprofiler 1.0

// Renders one Field ({label,value,kind,reason} -- see
// src/gui/inspector.py's _field()) with a small kind badge, so
// measured/derived/estimated/unavailable is always visually
// distinguishable instead of every value looking the same. "measured"
// (the common case) gets no badge at all -- badging only what's NOT
// a plain trace-sourced number keeps the panel from being wall-to-wall
// badges when most fields are perfectly ordinary.
ColumnLayout {
    id: root
    property var field: ({})
    Layout.fillWidth: true
    spacing: 2

    function kindColor() {
        switch (root.field.kind) {
        case "unavailable": return AppTheme.textMuted
        case "estimated": return AppTheme.warningColor
        case "derived": return AppTheme.infoColor
        default: return AppTheme.textMuted
        }
    }

    RowLayout {
        Layout.fillWidth: true
        spacing: AppTheme.spacingSm

        Text {
            text: root.field.label || ""
            color: AppTheme.textMuted
            font.pixelSize: AppTheme.typeCaption
            Layout.fillWidth: true
            elide: Text.ElideRight
        }
        Rectangle {
            visible: (root.field.kind || "measured") !== "measured"
            radius: AppTheme.radiusSmall
            color: "transparent"
            border.width: 1
            border.color: root.kindColor()
            implicitWidth: kindLabel.implicitWidth + 10
            implicitHeight: kindLabel.implicitHeight + 4

            Text {
                id: kindLabel
                anchors.centerIn: parent
                text: root.field.kind || ""
                font.pixelSize: AppTheme.typeCaption
                color: root.kindColor()
            }
        }
    }

    Text {
        visible: (root.field.value || "").length > 0
        text: root.field.value || ""
        color: AppTheme.text
        font.pixelSize: AppTheme.typeBody
        Layout.fillWidth: true
        wrapMode: Text.WordWrap
    }
    Text {
        visible: (root.field.reason || "").length > 0
        text: root.field.reason || ""
        color: AppTheme.textMuted
        font.pixelSize: AppTheme.typeCaption
        font.italic: true
        Layout.fillWidth: true
        wrapMode: Text.WordWrap
    }
}
