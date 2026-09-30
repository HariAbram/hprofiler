"""
Qt/QML GUI bootstrap -- run as a SEPARATE PROCESS by launch.py (see that
module's docstring for why: isolating a GLX crash from the CLI process).

Usage: python3 app.py <trace.json> [--compare <trace_b.json>] [--disasm]

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

HPROFILER_GUI_SELFTEST=1 (set by tests only): loads the QML, prints a
one-line JSON summary of rootObjects()/warnings, and returns BEFORE
app.exec() instead of showing a real window -- backs a subprocess-based
test for the populated (real second trace) Compare tab, which the
shared-engine in-process test class (tests/test_gui_timeline_hover.py)
structurally can't do: that engine is process-global and can only ever
have ONE trace pairing loaded in it for the whole test run, so a SECOND,
different trace_a/trace_b pairing needs a genuinely separate process.

Loading (Phase 3 of the usability/persistence/loading overhaul): the
actual parse + the four most expensive bridge computations now run on a
background QThread (src/gui/controller.py's LoadController), not this
thread -- but no QML engine loads, and no window appears, until that
finishes. This is a deliberate, empirically-forced scope boundary, not
an oversight: a SECOND QQmlApplicationEngine loading Qt Quick Controls
types anywhere in this process corrupts Controls resolution in this
PySide6 build, confirmed even when a FIRST, completely Controls-free
"splash" engine is loaded and thoroughly torn down first (see project
memory project_qml_gui.md, Round 17) -- so there is no safe way to show
a loading-progress WINDOW before the one-and-only real engine loads.
Progress is instead printed to stderr (`[hprofiler][gui] ...`, the same
convention launch.py already uses for its tier-fallback logging), which
is genuinely visible since this process is always launched from a
terminal. HPROFILER_READY_MARKER (set by controller.py's
open_profile_subprocess when this process was spawned as an "Open
Profile" replacement for an already-running window): once the QML
engine successfully loads, this process touches that path so the OLD
process's ProfileOpenWatcher knows to close itself -- see
controller.py's module docstring for the full handshake design.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def _parse_argv(argv: list[str]) -> tuple[str, str | None, bool]:
    """argv[0] is this script's own path (sys.argv convention) -- returns
    (trace_path, compare_path, disasm)."""
    if len(argv) < 2:
        raise SystemExit("usage: app.py <trace.json> [--compare <trace_b.json>] [--disasm]")
    trace_path = argv[1]
    rest = argv[2:]
    disasm = "--disasm" in rest
    compare_path = None
    if "--compare" in rest:
        i = rest.index("--compare")
        if i + 1 >= len(rest):
            raise SystemExit("--compare requires a trace path argument")
        compare_path = rest[i + 1]
    return trace_path, compare_path, disasm


def main() -> int:
    trace_path, compare_path, disasm = _parse_argv(sys.argv)

    from PySide6.QtCore import QUrl, QObject, Property
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtQml import QQmlApplicationEngine, qmlRegisterSingletonInstance

    # Absolute imports (not relative -- this file is run directly as a
    # script by launch.py's subprocess.run, not imported as part of the
    # src.gui package, so relative imports would fail with "attempted
    # relative import with no known parent package"; the sys.path insert
    # above makes the repo root importable as `src.*` instead).
    from src.gui.theme import Theme
    from src.gui.bridge import (
        DashboardBridge, KernelsBridge, CallTreeBridge, FlameGraphBridge, RooflineBridge,
        SourceBridge, SystemBridge, ProfileBridge,
    )
    from src.gui.models import TimelineModel
    from src.gui.nav import Selection
    from src.gui.inspector import InspectorBridge
    from src.gui.tablemodel import FormatBridge
    from src.gui.comparison import ComparisonBridge
    from src.gui.settings import WorkspaceSettings, WorkspaceBridge
    from src.gui.controller import AppController, LoadController, run_load_blocking
    from src.gui.logging_setup import setup_logging, get_logger, log_error, current_log_path
    from src.gui.shortcuts import ShortcutsBridge

    app = QGuiApplication(sys.argv[:1])
    app.setApplicationName("hprofiler")
    app.setOrganizationName("hprofiler")

    setup_logging()
    log = get_logger()
    log.info("loading %s", trace_path)

    settings = WorkspaceSettings()
    dark = settings.load_theme()

    verbose = os.environ.get("HPROFILER_GUI_SELFTEST") != "1"

    def _on_stage(stage: str, label: str) -> None:
        if verbose:
            print(f"[hprofiler][gui] {label}", file=sys.stderr)

    last_progress_pct = [-1]

    def _on_progress(current: int, total: int) -> None:
        if not verbose or total <= 0:
            return
        pct = (current * 100) // total
        if pct != last_progress_pct[0]:
            last_progress_pct[0] = pct
            print(f"[hprofiler][gui]   {pct}%", file=sys.stderr)

    import signal
    from PySide6.QtCore import QTimer

    # Python runs signal handlers only when the main thread executes Python
    # bytecode, which barely happens inside Qt's event loop -- Ctrl+C was
    # ignored for the whole load (and then the window opened anyway). A
    # cheap periodic no-op slot gives the handler a chance to run.
    py_tick = QTimer()
    py_tick.timeout.connect(lambda: None)
    py_tick.start(100)

    loader = LoadController(trace_path, compare_path=compare_path, disasm=disasm, dark=dark)

    def _cancel_on_sigint(_signum, _frame):
        log.info("load interrupted (Ctrl+C) -- cancelling")
        loader.cancel()   # plain flag write; the worker polls it

    prev_sigint = signal.signal(signal.SIGINT, _cancel_on_sigint)
    try:
        outcome, value = run_load_blocking(loader, on_stage=_on_stage, on_progress=_on_progress)
    finally:
        signal.signal(signal.SIGINT, prev_sigint)

    if outcome == "cancelled":
        print("[hprofiler][gui] cancelled", file=sys.stderr)
        return 130
    if outcome == "failed":
        err = value
        log_error(err)
        print(f"[hprofiler][gui] {err.message}", file=sys.stderr)
        if err.detail:
            print(f"[hprofiler][gui] detail: {err.detail}", file=sys.stderr)
        log_path = current_log_path()
        if log_path:
            print(f"[hprofiler][gui] log: {log_path}", file=sys.stderr)
        return 1

    result = value  # LoadResult
    trace = result.trace
    trace_b = result.trace_b

    theme = Theme(dark=dark)
    theme.themeChanged.connect(lambda: settings.save_theme(theme.dark))
    comparison = ComparisonBridge(trace, trace_b, theme)
    dashboard = DashboardBridge(trace, theme, comparison=comparison, precomputed=result.dashboard_data)
    timeline = TimelineModel(trace, theme)
    kernels = KernelsBridge(trace, theme)
    call_tree = CallTreeBridge(trace, theme, precomputed=result.call_tree_data)
    flame_graph = FlameGraphBridge(trace, theme, precomputed=result.flame_graph_data)
    roofline = RooflineBridge(trace)
    source = SourceBridge(trace)
    system = SystemBridge(trace)
    profile = ProfileBridge(trace, theme)
    selection = Selection()
    inspector = InspectorBridge(trace, selection, kernels, call_tree, roofline, source, timeline)
    formatter = FormatBridge()
    workspace_bridge = WorkspaceBridge(settings)
    app_controller = AppController(str(trace_path), compare_path)
    shortcuts_bridge = ShortcutsBridge()

    meta = trace.metadata
    cmd_line = f"{meta.command} {' '.join(meta.args[:4])}".strip() or "(no command)"

    class AppInfo(QObject):
        @Property(str, constant=True)
        def commandLine(self) -> str:
            return cmd_line

        @Property(str, constant=True)
        def tracePath(self) -> str:
            return str(trace_path)

        @Property(str, constant=True)
        def comparePath(self) -> str:
            return str(compare_path) if compare_path else ""

        @Property(bool, constant=True)
        def hasComparison(self) -> bool:
            return trace_b is not None

    app_info = AppInfo()

    qmlRegisterSingletonInstance(Theme, "Hprofiler", 1, 0, "AppTheme", theme)
    qmlRegisterSingletonInstance(DashboardBridge, "Hprofiler", 1, 0, "Dashboard", dashboard)
    qmlRegisterSingletonInstance(AppInfo, "Hprofiler", 1, 0, "AppInfo", app_info)
    qmlRegisterSingletonInstance(TimelineModel, "Hprofiler", 1, 0, "TimelineModel", timeline)
    qmlRegisterSingletonInstance(KernelsBridge, "Hprofiler", 1, 0, "Kernels", kernels)
    qmlRegisterSingletonInstance(CallTreeBridge, "Hprofiler", 1, 0, "CallTree", call_tree)
    qmlRegisterSingletonInstance(FlameGraphBridge, "Hprofiler", 1, 0, "FlameGraph", flame_graph)
    qmlRegisterSingletonInstance(RooflineBridge, "Hprofiler", 1, 0, "Roofline", roofline)
    qmlRegisterSingletonInstance(SourceBridge, "Hprofiler", 1, 0, "Source", source)
    qmlRegisterSingletonInstance(SystemBridge, "Hprofiler", 1, 0, "System", system)
    qmlRegisterSingletonInstance(ProfileBridge, "Hprofiler", 1, 0, "Profile", profile)
    qmlRegisterSingletonInstance(Selection, "Hprofiler", 1, 0, "Nav", selection)
    qmlRegisterSingletonInstance(InspectorBridge, "Hprofiler", 1, 0, "Inspector", inspector)
    qmlRegisterSingletonInstance(FormatBridge, "Hprofiler", 1, 0, "Format", formatter)
    qmlRegisterSingletonInstance(WorkspaceBridge, "Hprofiler", 1, 0, "Workspace", workspace_bridge)
    qmlRegisterSingletonInstance(AppController, "Hprofiler", 1, 0, "App", app_controller)
    qmlRegisterSingletonInstance(ShortcutsBridge, "Hprofiler", 1, 0, "Shortcuts", shortcuts_bridge)
    # Always registered, even with no comparison trace -- an unregistered
    # QML singleton that CompareScreen.qml tries to bind against is a
    # load error, not just an empty screen (see comparison.py's docstring).
    qmlRegisterSingletonInstance(ComparisonBridge, "Hprofiler", 1, 0, "Compare", comparison)

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

    # Signal a watching parent process (see controller.py's
    # open_profile_subprocess/ProfileOpenWatcher) that this window is up
    # and it can now close itself -- a plain empty file, its EXISTENCE is
    # the whole signal. Absent for a normal (non-"Open Profile") launch.
    ready_marker = os.environ.get("HPROFILER_READY_MARKER")
    if ready_marker:
        try:
            Path(ready_marker).touch()
        except OSError:
            pass

    if os.environ.get("HPROFILER_GUI_SELFTEST") == "1":
        import json
        print(json.dumps({
            "rootObjects": len(engine.rootObjects()),
            "warnings": errors,
            "hasComparison": trace_b is not None,
        }))
        return 0

    # Ctrl+C in the launching terminal closes the window normally (same
    # path as closing it) instead of being ignored by the event loop.
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
