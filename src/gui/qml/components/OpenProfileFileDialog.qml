import QtQuick
import QtQuick.Dialogs
import Hprofiler 1.0

// "Open a different profile" file picker -- backs both the command
// palette's "Open a different profile…" command and Main.qml's File
// menu item. Converts the platform file dialog's QUrl result to a plain
// local path string before calling App.openProfile(str), which never
// touches this process's own QML engine (see controller.py's module
// docstring) -- spawns a new subprocess and watches it instead.
FileDialog {
    id: root
    objectName: "openProfileFileDialog"
    title: "Open Profile"
    nameFilters: ["Trace files (*.json)", "All files (*)"]

    function urlToLocalPath(u) {
        var s = u.toString()
        if (s.indexOf("file://") === 0) s = s.substring(7)
        return decodeURIComponent(s)
    }

    onAccepted: App.openProfile(urlToLocalPath(selectedFile))
}
