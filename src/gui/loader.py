"""
Async profile loading -- runs the expensive, Qt-free parts of opening a
trace (the parse, plus the pure-compute functions extracted from
DashboardBridge/CallTreeBridge/FlameGraphBridge in bridge.py) on a
dedicated QThread, so the GUI process's main thread stays responsive and
reports progress.

Design: a QObject worker moved to a QThread via moveToThread(), driven by
queued signals, with a done/error/progress/cancel contract (the
fire-and-forget background-Thread pattern used by SourceBridge and
chrome_trace.py has none of these). The worker constructs NO QObject
bridge instances -- only plain Python (Trace, dicts, dataclasses) crosses
the thread boundary; every bridge QObject is constructed on the main
thread afterward from the LoadResult payload, as Qt requires a QObject to
live on the thread that constructs it.

Scope: only the highest-cost computations run here -- the trace parse
(the dominant cost for a large trace) and the three bridges that do
multi-pass/O(n log n) work (Dashboard, Call Tree, Flame Graph).
TimelineModel and the smaller, bounded bridges (Kernels, Roofline, Source,
System, Profile, Inspector, Comparison) are constructed on the main
thread.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from PySide6.QtCore import QObject, Signal, Slot

from ..core.trace import Trace
from ..output.chrome_trace import LoadCancelled
from ..core.trace_io import open_trace
from .errors import HprofilerLoadError, classify_load_exception
from .logging_setup import get_logger, log_error


class LoadStage(str, Enum):
    READING_FILE = "reading_file"
    PARSING_JSON = "parsing_json"
    COMPUTING_DASHBOARD = "computing_dashboard"
    COMPUTING_CALL_TREE = "computing_call_tree"
    COMPUTING_FLAME_GRAPH = "computing_flame_graph"
    LOADING_COMPARISON = "loading_comparison"
    COMPUTING_COMPARISON = "computing_comparison"
    DONE = "done"


_STAGE_LABELS: dict[LoadStage, str] = {
    LoadStage.READING_FILE: "Reading trace file…",
    LoadStage.PARSING_JSON: "Parsing trace events…",
    LoadStage.COMPUTING_DASHBOARD: "Computing overview…",
    LoadStage.COMPUTING_CALL_TREE: "Building call tree…",
    LoadStage.COMPUTING_FLAME_GRAPH: "Building flame graph…",
    LoadStage.LOADING_COMPARISON: "Loading comparison trace…",
    LoadStage.COMPUTING_COMPARISON: "Comparing runs…",
    LoadStage.DONE: "Done",
}


@dataclass
class LoadResult:
    """Plain-data payload handed from the worker thread to the main
    thread via ProfileLoadWorker.finished -- every field here is either
    a Trace (a plain Python object, no Qt dependency -- see core/trace.py)
    or a dict/list built by one of bridge.py's pure compute_*_data()
    functions. Never a QObject."""

    trace: Trace
    trace_b: Trace | None
    dark: bool
    dashboard_data: dict[str, Any] = field(default_factory=dict)
    call_tree_data: list[dict[str, Any]] = field(default_factory=list)
    flame_graph_data: dict[str, Any] = field(default_factory=dict)
    comparison_data: dict[str, Any] | None = None


class ProfileLoadWorker(QObject):
    """Runs on a dedicated QThread (see AppController for the
    moveToThread()/lifecycle wiring). One instance per load attempt --
    never reused across two different loads.

    Every signal here crosses the thread boundary via Qt's automatic
    queued-connection marshalling (the receiving slot runs on whichever
    thread it was connected from -- ordinarily the main thread). `cancel()`
    is a Slot specifically so calling it from the main thread is itself
    marshalled safely onto the worker thread rather than racing with
    `run()`'s own reads of `_cancel_requested`."""

    stageChanged = Signal(str, str)      # stage value, human label
    progress = Signal(int, int)          # current, total (0, 0 = indeterminate)
    finished = Signal(object)            # LoadResult
    failed = Signal(object)              # HprofilerLoadError
    cancelled = Signal()

    def __init__(self, trace_path: str, *, compare_path: str | None = None,
                 disasm: bool = False, dark: bool = True) -> None:
        super().__init__()
        self._trace_path = trace_path
        self._compare_path = compare_path
        self._disasm = disasm
        self._dark = dark
        self._cancel_requested = False
        self._current_stage = LoadStage.READING_FILE

    @Slot()
    def cancel(self) -> None:
        self._cancel_requested = True

    def _is_cancelled(self) -> bool:
        return self._cancel_requested

    def _emit_stage(self, stage: LoadStage) -> None:
        self._current_stage = stage
        self.stageChanged.emit(stage.value, _STAGE_LABELS[stage])
        get_logger().info("load stage: %s (%s)", stage.value, self._trace_path)

    @Slot()
    def run(self) -> None:
        try:
            self._emit_stage(LoadStage.READING_FILE)
            if self._is_cancelled():
                self.cancelled.emit()
                return

            self._emit_stage(LoadStage.PARSING_JSON)
            # A .hpstore directory opens instantly (indexed, on disk); a
            # JSON file loads into memory, or is imported into a disk
            # store first when large (core/trace_io.open_trace).
            trace = open_trace(
                self._trace_path, collect_disasm=self._disasm,
                progress_cb=lambda i, t: self.progress.emit(i, t),
                cancel_check=self._is_cancelled,
            )

            trace_b: Trace | None = None
            if self._compare_path:
                self._emit_stage(LoadStage.LOADING_COMPARISON)
                if self._is_cancelled():
                    self.cancelled.emit()
                    return
                # Never collect_disasm=True for the comparison trace --
                # disassembly is a debugging aid for the run being
                # inspected, not for a comparison-only trace.
                trace_b = open_trace(
                    self._compare_path, collect_disasm=False,
                    cancel_check=self._is_cancelled,
                )

            if self._is_cancelled():
                self.cancelled.emit()
                return
            self._emit_stage(LoadStage.COMPUTING_DASHBOARD)
            from .bridge import compute_dashboard_data
            dashboard_data = compute_dashboard_data(trace, self._dark)

            if self._is_cancelled():
                self.cancelled.emit()
                return
            self._emit_stage(LoadStage.COMPUTING_CALL_TREE)
            from .bridge import compute_call_tree_data
            call_tree_data = compute_call_tree_data(trace, self._dark)

            if self._is_cancelled():
                self.cancelled.emit()
                return
            self._emit_stage(LoadStage.COMPUTING_FLAME_GRAPH)
            from .bridge import compute_flame_graph_data
            flame_graph_data = compute_flame_graph_data(trace, self._dark)

            comparison_data = None
            if trace_b is not None:
                if self._is_cancelled():
                    self.cancelled.emit()
                    return
                self._emit_stage(LoadStage.COMPUTING_COMPARISON)
                # The opened trace is the candidate, --compare names the
                # baseline (see src/gui/comparison.py).
                from .comparison import compute_comparison_data
                comparison_data = compute_comparison_data(trace, trace_b)

        except LoadCancelled:
            self.cancelled.emit()
            return
        except HprofilerLoadError as err:
            log_error(err)
            self.failed.emit(err)
            return
        except Exception as exc:
            err = classify_load_exception(exc, file=str(self._trace_path), stage=self._current_stage.value)
            log_error(err)
            self.failed.emit(err)
            return

        if self._is_cancelled():
            self.cancelled.emit()
            return
        self._emit_stage(LoadStage.DONE)
        self.finished.emit(LoadResult(
            trace=trace, trace_b=trace_b, dark=self._dark,
            dashboard_data=dashboard_data, call_tree_data=call_tree_data,
            flame_graph_data=flame_graph_data, comparison_data=comparison_data,
        ))
