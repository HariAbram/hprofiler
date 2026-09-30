import QtQuick
import Hprofiler 1.0

// One cell in a DataTable row or header -- formats `value` by `column.kind`
// via the "Format" singleton (src/gui/tablemodel.py's FormatBridge, which
// wraps analysis/dashboard.py's fmt_* helpers, so every table's numbers
// are formatted the SAME way instead of each screen reimplementing it in
// QML JS) and optionally draws an in-cell bar/heatmap behind the text.
Item {
    id: root
    property var column: ({})      // one entry from TableConfig.columns
    property var row: ({})          // the whole source row dict
    property var barMaxima: ({})    // TableBundle.barMaxima
    property bool percentMode: false
    signal hovered(string text, string definition)
    signal unhovered()

    readonly property real rawValue: {
        var v = root.row ? root.row[root.column.key] : undefined
        return (typeof v === "number") ? v : 0
    }
    readonly property bool showPercent: root.percentMode && root.column.pctOf && root.row
    readonly property real percentValue: {
        if (!showPercent) return 0
        var of = root.row[root.column.pctOf]
        return of > 0 ? rawValue / of * 100 : 0
    }
    // Field-shaped rows ({label,value,kind,reason} -- see inspector.py's
    // convention, reused here for System's metric table and similar
    // "might not be measured" tables) -- the "value" column shows the
    // muted/warning/info kind coloring an Inspector field already uses,
    // distinct from a comparison row's "status" (improved/regressed/...),
    // a different vocabulary entirely despite the visual similarity.
    readonly property bool isFieldValueCell: root.row && root.row.kind !== undefined && root.column.key === "value"
    readonly property color fieldKindColor: {
        if (!isFieldValueCell) return AppTheme.text
        switch (root.row.kind) {
        case "unavailable": return AppTheme.textMuted
        case "estimated": return AppTheme.warningColor
        case "derived": return AppTheme.infoColor
        default: return AppTheme.text
        }
    }
    readonly property string displayText: {
        if (!root.row) return ""
        if (root.column.kind === "text" || root.column.kind === "category"
            || root.column.kind === "status" || root.column.kind === "badge") {
            var raw = root.row[root.column.key]
            if (isFieldValueCell && (raw === undefined || raw === "") && root.row.reason) {
                return root.row.reason
            }
            return String(raw !== undefined ? raw : "")
        }
        if (showPercent) return Format.formatNumber("pct", percentValue)
        return Format.formatNumber(root.column.kind, rawValue)
    }
    readonly property real barFraction: {
        if (!root.column.bar || !root.row) return 0
        var maxV = root.barMaxima[root.column.key] || 0
        return maxV > 0 ? Math.max(0, Math.min(1, rawValue / maxV)) : 0
    }

    // In-cell bar/heatmap -- a translucent fill behind the text, width
    // proportional to this cell's own value over the table-wide max for
    // that column (barMaxima, computed once from the full row set).
    Rectangle {
        visible: root.column.bar && root.column.bar.length > 0
        anchors.left: parent.left
        anchors.top: parent.top
        anchors.bottom: parent.bottom
        width: parent.width * root.barFraction
        color: Qt.alpha(root.row && root.row.color ? root.row.color : AppTheme.accent, 0.18)
    }

    Rectangle {
        visible: root.column.kind === "category"
        anchors.left: parent.left
        anchors.verticalCenter: parent.verticalCenter
        anchors.leftMargin: 2
        width: 8
        height: 8
        radius: 4
        color: root.row && root.row.color ? root.row.color : AppTheme.categoryColor(root.displayText)
    }

    Rectangle {
        visible: root.column.kind === "status" && root.row && root.row.status !== undefined
        anchors.left: parent.left
        anchors.verticalCenter: parent.verticalCenter
        anchors.leftMargin: 2
        width: 8
        height: 8
        radius: 4
        color: root.row ? AppTheme.changeColor(root.row.status) : AppTheme.textMuted
    }

    Text {
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.verticalCenter: parent.verticalCenter
        anchors.leftMargin: (root.column.kind === "category"
                             || (root.column.kind === "status" && root.row && root.row.status !== undefined)) ? 14 : 4
        anchors.rightMargin: 4
        text: root.displayText
        color: root.isFieldValueCell ? root.fieldKindColor
               : (root.column.kind === "status" && root.row && root.row.status !== undefined
                  ? AppTheme.changeColor(root.row.status) : AppTheme.text)
        font.italic: root.isFieldValueCell && root.row.kind === "unavailable"
        font.pixelSize: AppTheme.typeBody
        horizontalAlignment: root.column.align === "left" ? Text.AlignLeft : Text.AlignRight
        elide: Text.ElideRight
    }

    MouseArea {
        anchors.fill: parent
        hoverEnabled: true
        acceptedButtons: Qt.NoButton
        onEntered: {
            var full = root.column.kind === "text" ? root.displayText
                       : root.displayText + (root.row && root.row[root.column.key] !== undefined
                                              ? "  (" + root.row[root.column.key] + ")" : "")
            root.hovered(full, root.column.definition || "")
        }
        onExited: root.unhovered()
    }
}
