"""
Qt/QML flame graph popup bootstrap -- run as a SEPARATE PROCESS by
launch.py's launch_flamegraph_gui(), same crash-isolation rationale as
src/gui/app.py (the main dashboard's bootstrap): a hard GLX crash here
must not take the `hprofiler flamegraph` CLI process down with it.

Usage: python3 flamegraph_app.py <folded_stacks_file> <title>

Reads raw folded-stacks text (Brendan Gregg's stackcollapse format,
"func1;func2;func3 count" per line) from a file rather than inline argv
-- a real HPC binary's collected stacks can be many MB of text, well
past comfortable argv/environment size limits on some systems.

Exit code 0 means the window opened and closed normally; see app.py's
docstring for why a nonzero/killed exit (not rendering quality) is what
launch.py's tier-retry logic keys off.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: flamegraph_app.py <folded_stacks_file> <title>", file=sys.stderr)
        return 2
    folded_path = sys.argv[1]
    title = sys.argv[2]

    from PySide6.QtCore import QUrl
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtQml import QQmlApplicationEngine, qmlRegisterSingletonInstance

    from src.output.flamegraph import build_qml_tree
    from src.gui.theme import Theme
    from src.gui.flamegraph_bridge import FlameGraphBridge

    folded_text = Path(folded_path).read_text(errors="replace")
    tree = build_qml_tree(folded_text)
    if tree is None:
        print("[hprofiler][gui] no stacks to render in the collected flame graph data",
              file=sys.stderr)
        return 1

    app = QGuiApplication(sys.argv[:1])
    app.setApplicationName("hprofiler")
    app.setOrganizationName("hprofiler")

    theme = Theme(dark=True)
    flamegraph = FlameGraphBridge(tree, title)

    qmlRegisterSingletonInstance(Theme, "Hprofiler", 1, 0, "AppTheme", theme)
    qmlRegisterSingletonInstance(FlameGraphBridge, "Hprofiler", 1, 0, "FlameGraph", flamegraph)

    engine = QQmlApplicationEngine()
    errors: list[str] = []
    engine.warnings.connect(lambda ws: errors.extend(str(w) for w in ws))

    qml_main = Path(__file__).resolve().parent / "qml" / "FlameGraphWindow.qml"
    engine.load(QUrl.fromLocalFile(str(qml_main)))

    if not engine.rootObjects():
        for w in errors:
            print(f"[hprofiler][gui] QML error: {w}", file=sys.stderr)
        print("[hprofiler][gui] failed to load the flame graph QML UI", file=sys.stderr)
        return 1

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
