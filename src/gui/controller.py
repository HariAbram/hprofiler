"""
Orchestration for the usability/persistence/loading overhaul (Phase 3).
Two independent concerns live here:

1. `LoadController` -- owns exactly one `ProfileLoadWorker` + `QThread`
   lifecycle (starting it, forwarding its signals, and guaranteeing the
   thread is stopped and cleaned up no matter how the load ends: success,
   failure, cancellation, or the whole application quitting mid-load).
   Used both for a first launch (app.py's main()) and, inside a freshly
   spawned "Open Profile" child process, for that process's own first
   launch -- there is only ever ONE loading code path, never a separate
   "initial load" vs. "switch profile" implementation to keep in sync.

   IMPORTANT (cost real debugging time, see project memory
   project_qml_gui.md Round 17): `LoadController.cancel()` calls
   `self._worker.cancel()` as a PLAIN Python method call, never via a
   Qt signal connected directly to the worker's bound `cancel` slot.
   Connecting a cross-thread signal straight to a `@Slot`-decorated
   method of a QObject living on another thread makes PySide
   auto-detect a queued connection -- which can only be delivered once
   that thread's event loop starts pumping, but `QThread`'s default
   `run()` doesn't call `exec()` (starting that loop) until the
   `started`-connected `worker.run()` -- the long synchronous parse --
   already RETURNS. That's a deadlock (the queued cancel can never
   arrive in time to matter) that also corrupts process shutdown later.
   A plain, undecorated-at-the-call-site Python call bypasses all of
   that -- it's just a GIL-safe attribute write, exactly what
   `ProfileLoadWorker.cancel()` already does.

2. `AppController` -- registered as the "Hprofiler" 1.0 "App" singleton
   in the real (only) QML engine once it's up. Owns "Open Profile":
   spawning a genuinely NEW OS process rather than ever rebuilding this
   process's own QML engine in-process. Verified empirically (see the
   approved plan's Context section and project memory) that a SECOND
   QQmlApplicationEngine loading Qt Quick Controls types anywhere in one
   process corrupts Controls resolution in this PySide6 build, with no
   teardown sequence found that avoids it -- so profile-switching must
   never touch this process's engine at all. The OLD window stays fully
   open and interactive for as long as the new process takes to either
   show its own window or fail; a small ready-marker-file handshake
   (`HPROFILER_READY_MARKER`) tells this process when the child has
   succeeded, at which point (and only then) this process quits. A
   failure in the child cannot touch the old window's state at all --
   the two processes share no memory.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from PySide6.QtCore import QCoreApplication, QObject, QThread, QTimer, QUrl, Property, Signal, Slot
from PySide6.QtGui import QDesktopServices

from .errors import ErrorKind, HprofilerLoadError, classify_load_exception
from .loader import ProfileLoadWorker
from .logging_setup import current_log_path, get_logger, log_error

READY_MARKER_ENV = "HPROFILER_READY_MARKER"

# How long the OLD window waits for a spawned child to become ready
# (write its marker file) or exit, before giving up on WATCHING it --
# the child keeps running either way; this only bounds how long the old
# window's polling loop waits before reporting "no response" rather than
# hanging forever.
_OPEN_PROFILE_TIMEOUT_S = 30.0
_WATCH_POLL_MS = 150


class LoadController(QObject):
    """See module docstring. One instance per load attempt; never reused
    across two different loads."""

    stageChanged = Signal(str, str)
    progress = Signal(int, int)
    finished = Signal(object)   # LoadResult
    failed = Signal(object)     # HprofilerLoadError
    cancelled = Signal()

    def __init__(self, trace_path: str, *, compare_path: str | None = None,
                 disasm: bool = False, dark: bool = True, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._worker = ProfileLoadWorker(trace_path, compare_path=compare_path,
                                          disasm=disasm, dark=dark)
        self._thread = QThread()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.cancelled.connect(self._on_cancelled)
        self._worker.stageChanged.connect(self.stageChanged)
        self._worker.progress.connect(self.progress)
        self._stopped = False

        app = QCoreApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._on_about_to_quit)

    def start(self) -> None:
        self._thread.start()

    def cancel(self) -> None:
        # A plain call -- see module docstring for why this must never
        # be wired up via a Qt signal/slot connection instead.
        self._worker.cancel()

    def _teardown(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._thread.quit()
        self._thread.wait(10_000)

    def _on_finished(self, result: Any) -> None:
        self._teardown()
        self.finished.emit(result)

    def _on_failed(self, err: Any) -> None:
        self._teardown()
        self.failed.emit(err)

    def _on_cancelled(self) -> None:
        self._teardown()
        self.cancelled.emit()

    def _on_about_to_quit(self) -> None:
        # The whole application is quitting (window closed mid-load,
        # Ctrl+C, ...) -- ask the worker to stop and bound how long we
        # wait for it, rather than hanging shutdown indefinitely on a
        # large trace.
        if self._stopped:
            return
        self._worker.cancel()
        self._teardown()


def run_load_blocking(loader: LoadController, *, on_stage=None, on_progress=None):
    """Drives `loader` to completion using a real (but strictly local,
    non-reentrant-with-anything-else) QEventLoop -- the correct,
    idiomatic Qt way to block-with-signals in code that isn't itself
    inside a slot, used here so app.py's main() can await the async
    load before ever constructing a QML engine (there is nothing for a
    QML engine to show yet -- see AppController's docstring and project
    memory for why a loading-progress WINDOW isn't possible before the
    one-and-only real engine loads). `on_stage`/`on_progress` are
    optional plain callables invoked synchronously as those signals
    arrive, e.g. to print progress to the terminal.

    Returns ("finished", LoadResult) / ("failed", HprofilerLoadError) /
    ("cancelled", None). Raises nothing -- a KeyboardInterrupt during
    the load is the CALLER's responsibility to catch and turn into
    `loader.cancel()` if desired (this function doesn't install its own
    signal handler, matching Python's normal SIGINT behavior of
    interrupting whichever line is currently executing)."""
    from PySide6.QtCore import QEventLoop

    outcome: dict[str, Any] = {}
    loop = QEventLoop()

    if on_stage is not None:
        loader.stageChanged.connect(lambda s, label: on_stage(s, label))
    if on_progress is not None:
        loader.progress.connect(lambda i, t: on_progress(i, t))

    def _finished(result: Any) -> None:
        outcome["kind"] = "finished"
        outcome["value"] = result
        loop.quit()

    def _failed(err: Any) -> None:
        outcome["kind"] = "failed"
        outcome["value"] = err
        loop.quit()

    def _cancelled() -> None:
        outcome["kind"] = "cancelled"
        outcome["value"] = None
        loop.quit()

    loader.finished.connect(_finished)
    loader.failed.connect(_failed)
    loader.cancelled.connect(_cancelled)

    loader.start()
    loop.exec()

    return outcome.get("kind", "failed"), outcome.get("value")


# ── "Open Profile": spawn a new process, never rebuild this one's engine ──

def validate_profile_path(trace_path: str) -> HprofilerLoadError | None:
    """Cheap, synchronous pre-flight check run BEFORE spawning a
    subprocess for "Open Profile" -- catches the common mistakes (a
    typo'd path, a directory, no read permission, an empty/non-JSON
    file) without paying the cost of a Python/Qt subprocess startup just
    to fail immediately inside it. Returns None if the path is
    plausible (the real parse/validation still happens inside the child
    via load_trace_from_json -- this is a fast pre-filter, not a
    replacement), or a classified, ready-to-display error otherwise."""
    p = Path(trace_path)
    from ..core.trace_io import is_store
    if p.is_dir() and is_store(p):
        return None                      # a .hpstore trace store
    try:
        if not p.exists():
            raise FileNotFoundError(str(trace_path))
        if p.is_dir():
            raise IsADirectoryError(str(trace_path))
        with open(p, "rb") as f:
            head = f.read(2048)
    except Exception as exc:
        return classify_load_exception(exc, file=str(trace_path), stage="validating")

    # Not JSON at all (empty file, plain text, ...) -- constructed
    # directly as INVALID_INPUT rather than raising a generic ValueError
    # through classify_load_exception, which deliberately maps
    # ValueError to UNSUPPORTED_DATA (matching what chrome_trace.py's
    # parser raises for a well-formed-JSON-but-foreign-schema file).
    # Those are different situations: this is "not JSON", that's "JSON,
    # wrong shape" -- worth keeping distinct for an accurate message.
    stripped = head.lstrip()
    if not stripped or stripped[:1] not in (b"{", b"["):
        return HprofilerLoadError(
            kind=ErrorKind.INVALID_INPUT,
            message=f"{trace_path} doesn't look like a trace file.",
            file=str(trace_path), stage="validating",
        )
    return None


def build_profile_argv(trace_path: str, *, compare_path: str | None = None,
                        disasm: bool = False,
                        ready_marker: str | None = None) -> tuple[list[str], dict[str, str]]:
    """argv + env for a new GUI subprocess -- mirrors launch.py's
    existing argv shape exactly (trace_path, then --compare/--disasm) so
    app.py's own _parse_argv keeps working completely unchanged; the
    ready-marker path is passed via the environment (additive, never
    read by _parse_argv) rather than argv, so it can never collide with
    positional-argument parsing."""
    app_main = os.path.join(os.path.dirname(__file__), "app.py")
    argv = [sys.executable, app_main, str(trace_path)]
    if compare_path:
        argv += ["--compare", str(compare_path)]
    if disasm:
        argv += ["--disasm"]
    env = dict(os.environ)
    if ready_marker:
        env[READY_MARKER_ENV] = str(ready_marker)
    return argv, env


class ProfileOpenWatcher(QObject):
    """Watches a just-spawned "Open Profile" subprocess without blocking
    the current window -- polls (QTimer, never a blocking wait) for
    either the child's own ready-marker file appearing (it successfully
    showed its window -- app.py writes this right after
    engine.rootObjects() succeeds) or the child process exiting early (a
    real failure). The CURRENT window and everything in it are never
    touched while this runs, and untouched if the child fails -- the two
    processes share no memory, so there is nothing here that COULD
    corrupt the old workspace."""

    ready = Signal()
    failed = Signal(object)  # HprofilerLoadError

    def __init__(self, popen: subprocess.Popen, marker_path: str, *,
                 timeout_s: float = _OPEN_PROFILE_TIMEOUT_S,
                 parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._popen = popen
        self._marker_path = Path(marker_path)
        self._timeout_s = timeout_s
        self._deadline = time.monotonic() + timeout_s
        self._timer = QTimer(self)
        self._timer.setInterval(_WATCH_POLL_MS)
        self._timer.timeout.connect(self._poll)
        self._done = False

    def start(self) -> None:
        self._timer.start()

    def _stop(self) -> None:
        self._timer.stop()
        try:
            self._marker_path.unlink(missing_ok=True)
        except OSError:
            pass

    def _poll(self) -> None:
        if self._done:
            return

        if self._marker_path.exists():
            self._done = True
            self._stop()
            self.ready.emit()
            return

        rc = self._popen.poll()
        if rc is not None:
            self._done = True
            self._stop()
            stderr_tail = ""
            try:
                if self._popen.stderr is not None:
                    stderr_tail = self._popen.stderr.read().decode("utf-8", "replace")[-2000:]
            except Exception:
                pass
            err = HprofilerLoadError(
                kind=ErrorKind.INTERNAL_ERROR,
                message="The new profile window didn't open.",
                detail=stderr_tail or f"the GUI process exited with code {rc}",
                stage="opening_profile",
            )
            self.failed.emit(err)
            return

        if time.monotonic() > self._deadline:
            self._done = True
            self._stop()
            # Deliberately NOT killed -- it may still succeed a moment
            # later (e.g. a genuinely slow first load of a huge trace);
            # giving up on WATCHING it is not the same as giving up on
            # it. The old window just stops waiting so the user isn't
            # left staring at nothing indefinitely.
            err = HprofilerLoadError(
                kind=ErrorKind.INTERNAL_ERROR,
                message="The new profile window is taking unusually long to open.",
                detail=f"no response after {self._timeout_s:.0f}s -- it may still appear "
                       "in a moment; check for a second hprofiler window before retrying",
                stage="opening_profile",
            )
            self.failed.emit(err)


def open_profile_subprocess(
    trace_path: str, *, compare_path: str | None = None, disasm: bool = False,
    timeout_s: float = _OPEN_PROFILE_TIMEOUT_S,
) -> tuple[subprocess.Popen | None, ProfileOpenWatcher | None, HprofilerLoadError | None]:
    """Fail-fast validates `trace_path`; only spawns a subprocess if that
    passes. Never blocks. Returns (popen, watcher, None) on a successful
    spawn (caller must keep both alive and call watcher.start()), or
    (None, None, error) if validation failed and nothing was spawned."""
    err = validate_profile_path(trace_path)
    if err is not None:
        return None, None, err

    marker_fd, marker_path = tempfile.mkstemp(prefix="hprofiler-ready-", suffix=".marker")
    os.close(marker_fd)
    os.unlink(marker_path)  # must NOT exist yet -- its creation is the ready signal

    argv, env = build_profile_argv(trace_path, compare_path=compare_path,
                                    disasm=disasm, ready_marker=marker_path)
    try:
        popen = subprocess.Popen(argv, env=env, stderr=subprocess.PIPE)
    except OSError as exc:
        return None, None, classify_load_exception(exc, file=str(trace_path), stage="opening_profile")

    watcher = ProfileOpenWatcher(popen, marker_path, timeout_s=timeout_s)
    return popen, watcher, None


class AppController(QObject):
    """Registered as the "Hprofiler" 1.0 "App" singleton in the real
    engine. Owns "Open Profile" from a running window -- see module
    docstring for why this is always a new subprocess, never an
    in-process engine rebuild."""

    profileOpening = Signal()
    # QVariantMap (HprofilerLoadError.to_dict()'s shape: kind, message,
    # detail, file, stage, tracebackText), NOT the raw HprofilerLoadError
    # object -- QML can read plain dict/QVariantMap keys as properties
    # directly (err.message, err.detail, ...), but has no way to access
    # attributes of an opaque wrapped Python object passed through
    # Signal(object). See errors.py's HprofilerLoadError.to_dict()
    # docstring: this shape was designed for exactly this consumption.
    profileOpenFailed = Signal('QVariantMap')

    def __init__(self, trace_path: str, compare_path: str | None = None,
                 parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._trace_path = trace_path
        self._compare_path = compare_path or ""
        self._watcher: ProfileOpenWatcher | None = None
        self._popen: subprocess.Popen | None = None

    @Property(str, constant=True)
    def tracePath(self) -> str:
        return self._trace_path

    @Property(str, constant=True)
    def comparePath(self) -> str:
        return self._compare_path

    @Slot(result=str)
    def logFilePath(self) -> str:
        path = current_log_path()
        return str(path) if path is not None else ""

    @Slot()
    def openLogFile(self) -> None:
        """Backs the "open log" action on a diagnostics panel -- opens
        the GUI's own rotating log file (see logging_setup.py) in
        whatever the OS considers the default handler for a .log file
        (a text editor, typically). A no-op (not an error) if logging
        hasn't produced a file yet -- there is nothing to open."""
        path = current_log_path()
        if path is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    @Slot(str)
    def openProfile(self, path: str) -> None:
        popen, watcher, err = open_profile_subprocess(path)
        if err is not None:
            log_error(err, extra_context="Open Profile")
            self.profileOpenFailed.emit(err.to_dict())
            return

        self._popen = popen
        self._watcher = watcher
        watcher.ready.connect(self._on_open_ready)
        watcher.failed.connect(self._on_open_failed)
        watcher.start()
        self.profileOpening.emit()
        get_logger().info("opening profile in a new process: %s", path)

    def _on_open_ready(self) -> None:
        get_logger().info("new profile process is ready -- closing this window")
        app = QCoreApplication.instance()
        if app is not None:
            app.quit()

    def _on_open_failed(self, err: Any) -> None:
        log_error(err, extra_context="Open Profile")
        self.profileOpenFailed.emit(err.to_dict())
