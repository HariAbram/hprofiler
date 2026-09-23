"""
QObject bridge for the standalone `hprofiler flamegraph --gui` popup --
kept separate from bridge.py (which is entirely Trace-object-oriented,
backing the main 8-tab dashboard from `hprofiler run --gui`/`hprofiler
gui`) since this window has a completely different data source: raw
folded-stacks text from a single `perf record --call-graph` pass
(src/output/flamegraph.py's collect_folded_stacks()), not a Trace/JSON
file at all. Reuses that same module's tree-building logic
(build_qml_tree(), which wraps the exact _build_tree() the HTML/TUI
flame graph outputs already use) so parsing can't drift between the
three renderers.
"""
from __future__ import annotations

from typing import Any

from PySide6.QtCore import QObject, Property


class FlameGraphBridge(QObject):
    """One instance per window. `tree` is the whole {name, value,
    children} structure handed to QML in one shot -- unlike the main
    dashboard's Timeline (viewport-culled per-frame fetching), a folded-
    stacks flame graph tree is bounded by distinct call-stack shapes in
    the profiled run, not by wall-clock span count, and in practice is
    small enough (hundreds to low thousands of nodes even for a busy
    HPC binary) to hand over whole and let QML's Canvas lay out
    on-the-fly per zoom level -- exactly what the existing HTML flame
    graph's JS already does (see flamegraph.py's _JS, ported near-
    verbatim into FlameGraph.qml's Canvas.onPaint)."""

    def __init__(self, tree: dict, title: str, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._tree = tree
        self._title = title

    @Property('QVariant', constant=True)
    def tree(self) -> dict[str, Any]:
        return self._tree

    @Property(str, constant=True)
    def title(self) -> str:
        return self._title

    @Property(int, constant=True)
    def totalSamples(self) -> int:
        return self._tree.get("value", 0)
