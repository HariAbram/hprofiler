"""Qt/QML GUI viewer -- optional, PySide6-based. Pops up after profiling
completes (or via `hprofiler gui <trace>`), falls back to the TUI
when the X server isn't reachable or PySide6 isn't installed. See
launch.py's launch_gui() for the actual fallback decision, and
DOCUMENTATION.md's "GUI" section for the full design."""
