"""
Qt/QML GUI bootstrap -- run as a SEPARATE PROCESS by launch.py (see that
module's docstring for why: isolating a GLX crash from the CLI process).

Usage: python3 app.py <trace.json> [--disasm]

Exit code 0 means the window opened and closed normally (the user closed
it) -- launch.py's caller treats that as "the GUI was shown", not
"nothing went wrong internally"; a window that opened but rendered
garbage due to a software/driver quirk still exits 0, since a Qt-level
crash (the actual failure mode this process-isolation guards against) is
what a nonzero/killed exit communicates back, not rendering quality.

Python objects are exposed to QML via qmlRegisterSingletonInstance under
the "Hprofiler" 1.0 module (import Hprofiler 1.0 in any .qml file that
uses AppTheme/Dashboard/AppInfo) -- NOT QQmlContext.setContextProperty,
which was found to silently fail in this PySide6 build: the object
registers successfully (readable back from Python via
rootContext().contextProperty()) but QML-side bindings still evaluate it
as null at runtime, with no reported error. qmlRegisterSingletonInstance
does not have this problem; verified with a minimal repro before
adopting it project-wide (see project_gui_qml_context_property_bug memory).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: app.py <trace.json>", file=sys.stderr)
        return 2
    trace_path = sys.argv[1]

    from PySide6.QtCore import QUrl, QObject, Property
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtQml import QQmlApplicationEngine, qmlRegisterSingletonInstance

    # Absolute imports (not relative -- this file is run directly as a
    # script by launch.py's subprocess.run, not imported as part of the
    # src.gui package, so relative imports would fail with "attempted
    # relative import with no known parent package"; the sys.path insert
    # above makes the repo root importable as `src.*` instead).
    from src.ui.app import load_trace_from_json
    from src.gui.theme import Theme
    from src.gui.bridge import (
        DashboardBridge, KernelsBridge, CallTreeBridge, RooflineBridge, SourceBridge,
        SystemBridge, ProfileBridge,
    )
    from src.gui.models import TimelineModel

    trace = load_trace_from_json(trace_path)

    app = QGuiApplication(sys.argv[:1])
    app.setApplicationName("hprofiler")
    app.setOrganizationName("hprofiler")

    theme = Theme(dark=True)
    dashboard = DashboardBridge(trace, theme)
    timeline = TimelineModel(trace, theme)
    kernels = KernelsBridge(trace, theme)
    call_tree = CallTreeBridge(trace, theme)
    roofline = RooflineBridge(trace)
    source = SourceBridge(trace)
    system = SystemBridge(trace)
    profile = ProfileBridge(trace, theme)

    meta = trace.metadata
    cmd_line = f"{meta.command} {' '.join(meta.args[:4])}".strip() or "(no command)"

    class AppInfo(QObject):
        @Property(str, constant=True)
        def commandLine(self) -> str:
            return cmd_line

        @Property(str, constant=True)
        def tracePath(self) -> str:
            return str(trace_path)

    app_info = AppInfo()

    qmlRegisterSingletonInstance(Theme, "Hprofiler", 1, 0, "AppTheme", theme)
    qmlRegisterSingletonInstance(DashboardBridge, "Hprofiler", 1, 0, "Dashboard", dashboard)
    qmlRegisterSingletonInstance(AppInfo, "Hprofiler", 1, 0, "AppInfo", app_info)
    qmlRegisterSingletonInstance(TimelineModel, "Hprofiler", 1, 0, "TimelineModel", timeline)
    qmlRegisterSingletonInstance(KernelsBridge, "Hprofiler", 1, 0, "Kernels", kernels)
    qmlRegisterSingletonInstance(CallTreeBridge, "Hprofiler", 1, 0, "CallTree", call_tree)
    qmlRegisterSingletonInstance(RooflineBridge, "Hprofiler", 1, 0, "Roofline", roofline)
    qmlRegisterSingletonInstance(SourceBridge, "Hprofiler", 1, 0, "Source", source)
    qmlRegisterSingletonInstance(SystemBridge, "Hprofiler", 1, 0, "System", system)
    qmlRegisterSingletonInstance(ProfileBridge, "Hprofiler", 1, 0, "Profile", profile)

    engine = QQmlApplicationEngine()
    errors: list[str] = []
    engine.warnings.connect(lambda ws: errors.extend(str(w) for w in ws))

    qml_main = Path(__file__).resolve().parent / "qml" / "Main.qml"
    engine.load(QUrl.fromLocalFile(str(qml_main)))

    if not engine.rootObjects():
        for w in errors:
            print(f"[hprofiler][gui] QML error: {w}", file=sys.stderr)
        print("[hprofiler][gui] failed to load the QML UI", file=sys.stderr)
        return 1

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
