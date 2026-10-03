import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import Hprofiler 1.0

// Timeline's event search: a name/regex query, match count,
// next/previous navigation. Jumping the view to a match is left to
// TimelineScreen.qml (matchJumped signal) -- this component only owns the
// query/navigation UI, not view-centering, matching TimelineFilterBar's
// same "one focused concern" split.
Item {
    id: root
    implicitWidth: rowLayout.implicitWidth
    implicitHeight: AppTheme.buttonHeight

    property bool isRegex: false
    // Emitted with the match dict TimelineModel.nextMatch()/previousMatch()
    // returned ({laneIndex, spanIdx, startNs, matchIndex, matchCount}) --
    // empty object means "no matches to jump to".
    signal matchJumped(var match)

    function runSearch() {
        TimelineModel.search(queryField.text, root.isRegex)
    }

    RowLayout {
        id: rowLayout
        anchors.fill: parent
        spacing: AppTheme.spacingXs

        TextField {
            id: queryField
            objectName: "timelineSearchField"
            Layout.preferredWidth: AppTheme.fieldWidth
            placeholderText: "search events…"
            color: AppTheme.text
            font.pixelSize: AppTheme.typeLabel
            onTextChanged: root.runSearch()
            background: Rectangle {
                color: AppTheme.background
                border.color: queryField.activeFocus ? AppTheme.panelBorderFocus : AppTheme.panelBorder
                border.width: 1
                radius: AppTheme.radiusSmall
            }
        }
        CheckBox {
            objectName: "timelineSearchRegex"
            text: "regex"
            font.pixelSize: AppTheme.typeCaption
            checked: root.isRegex
            onToggled: { root.isRegex = checked; root.runSearch() }
        }
        Text {
            objectName: "timelineSearchStatus"
            visible: queryField.text.length > 0
            text: TimelineModel.searchMatchCount === 0 ? "no matches"
                  : (TimelineModel.searchCursor + 1) + " / " + TimelineModel.searchMatchCount
            color: TimelineModel.searchMatchCount === 0 ? AppTheme.textMuted : AppTheme.accent
            font.pixelSize: AppTheme.typeLabel
        }
        ToolButton {
            objectName: "timelineSearchPrev"
            text: "◂"
            visible: queryField.text.length > 0
            enabled: TimelineModel.searchMatchCount > 0
            onClicked: root.matchJumped(TimelineModel.previousMatch())
            ToolTip.visible: hovered
            ToolTip.text: "Previous match"
            Accessible.name: "Previous match"
            Accessible.description: "Jumps to the previous search match on the Timeline"
        }
        ToolButton {
            objectName: "timelineSearchNext"
            text: "▸"
            visible: queryField.text.length > 0
            enabled: TimelineModel.searchMatchCount > 0
            onClicked: root.matchJumped(TimelineModel.nextMatch())
            ToolTip.visible: hovered
            ToolTip.text: "Next match"
            Accessible.name: "Next match"
            Accessible.description: "Jumps to the next search match on the Timeline"
        }
        ToolButton {
            objectName: "timelineSearchClear"
            text: "✕"
            visible: queryField.text.length > 0
            onClicked: { queryField.text = ""; TimelineModel.clearSearch() }
            ToolTip.visible: hovered
            ToolTip.text: "Clear search"
            Accessible.name: "Clear search"
            Accessible.description: "Clears the search query and search results"
        }
    }
}
