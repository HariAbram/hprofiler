import QtQuick
import QtQuick.Layouts
import Hprofiler 1.0
import "../components"

ColumnLayout {
    spacing: AppTheme.spacingLg

    Panel {
        Layout.fillWidth: true
        Layout.preferredHeight: 110
        title: "Run"
        ColumnLayout {
            anchors.fill: parent
            spacing: AppTheme.spacingXs
            Text { text: "Command:  " + System.command; color: AppTheme.text; font.pixelSize: AppTheme.typeBody }
            Text { text: "Host:  " + System.host; color: AppTheme.textMuted; font.pixelSize: AppTheme.typeBody }
            Text { text: "Duration:  " + System.duration; color: AppTheme.text; font.pixelSize: AppTheme.typeBody }
            Text { text: "Backends:  " + System.backends; color: AppTheme.textMuted; font.pixelSize: AppTheme.typeBody }
        }
    }

    DataTable {
        Layout.fillWidth: true
        Layout.preferredHeight: 220
        title: "Devices"
        table: System.deviceTable
        compact: true
        showFilterField: false
    }

    DataTable {
        Layout.fillWidth: true
        Layout.fillHeight: true
        title: "CPU metrics"
        table: System.metricTable
        compact: false
        showFilterField: false
    }
}
