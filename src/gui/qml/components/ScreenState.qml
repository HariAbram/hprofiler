import QtQuick
import Hprofiler 1.0

// Consistent WHOLE-TAB state overlay -- loading/empty/unsupported/
// cancelled/error/ready, one shared visual language for "this entire
// screen has nothing else useful to show right now" across every tab.
// Deliberately distinct from EmptyState (used independently ~18 times
// across screens already, left untouched): EmptyState is a smaller,
// PANEL-level "this one section has no rows" message that coexists
// alongside other populated content on the SAME screen (e.g. Kernels
// list empty while Inspector still shows something); ScreenState is
// for when there's nothing ELSE on the tab at all.
//
// An OVERLAY, not a gate: placed as a sibling on top of a screen's own
// real content (same "anchors.fill/centerIn + visible" pattern every
// existing EmptyState/LoadingState/ErrorState usage already follows in
// this codebase), not a Loader wrapping/deferring that content's
// construction -- keeps each screen's own layout exactly as it already
// is, avoids adding another nested Loader/Component indirection layer
// (a real, previously-hit bug class here: an extra Loader with no
// explicit sizing can collapse content to zero size with zero warning).
Item {
    id: root
    // "loading" | "empty" | "unsupported" | "cancelled" | "error" | "ready"
    property string state: "ready"

    property string emptyMessage: "Nothing to show yet."
    property bool emptyMonospace: false

    property string unsupportedMessage: "This metric isn't available for this trace."
    property bool unsupportedMonospace: false

    property string loadingMessage: "Loading…"
    property bool loadingShowProgress: false
    property real loadingProgressPct: 0
    property string loadingStageText: ""
    property string loadingElapsedText: ""

    property string cancelledMessage: "Cancelled."

    property string errorMessage: ""
    property string errorDetail: ""
    property string errorTracebackText: ""
    property string errorStage: ""
    property string errorFile: ""

    visible: state !== "ready"

    EmptyState {
        anchors.centerIn: parent
        centered: true
        monospace: root.emptyMonospace
        visible: root.state === "empty"
        message: root.emptyMessage
    }

    // "Unsupported" reuses EmptyState's presentation but with a warning-
    // toned leading icon so it reads as "this trace doesn't support
    // this" rather than plain "nothing found" -- a real, if small,
    // distinction: one means "nothing to report", the other "hprofiler
    // couldn't measure this for this run".
    Column {
        anchors.centerIn: parent
        spacing: AppTheme.spacingXs
        visible: root.state === "unsupported"

        Text {
            anchors.horizontalCenter: parent.horizontalCenter
            text: "⚠"
            color: AppTheme.warningColor
            font.pixelSize: AppTheme.typeTitle
        }
        EmptyState {
            centered: false
            monospace: root.unsupportedMonospace
            message: root.unsupportedMessage
        }
    }

    LoadingState {
        anchors.fill: parent
        visible: root.state === "loading"
        message: root.loadingMessage
        showProgress: root.loadingShowProgress
        progressPct: root.loadingProgressPct
        stageText: root.loadingStageText
        elapsedText: root.loadingElapsedText
    }

    EmptyState {
        anchors.centerIn: parent
        centered: true
        visible: root.state === "cancelled"
        message: root.cancelledMessage
    }

    ErrorState {
        anchors.fill: parent
        visible: root.state === "error"
        message: root.errorMessage
        detail: root.errorDetail
        tracebackText: root.errorTracebackText
        stage: root.errorStage
        file: root.errorFile
        onOpenLogRequested: App.openLogFile()
    }
}
