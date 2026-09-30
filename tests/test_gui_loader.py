"""
Tests for src/gui/loader.py's ProfileLoadWorker -- stage sequencing,
progress reporting, cancellation (before start and mid-parse), and error
classification. Most tests call worker.run() SYNCHRONOUSLY (directly, not
via a real QThread) for deterministic, fast assertions -- matches how
tests/test_gui_launch.py already avoids real subprocess/thread
nondeterminism by mocking the boundary. A small number of tests use a
real QThread specifically to prove the cross-thread signal marshalling
itself works (Qt's own well-tested machinery, but worth one direct
check given this is the first QThread usage anywhere in this codebase).

Needs a QCoreApplication (Signal/Slot machinery) but no QQmlApplicationEngine
anywhere in this file -- safe alongside tests/test_gui_timeline_hover.py's
one-engine-per-process constraint.
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QCoreApplication, QThread
    _PYSIDE6_AVAILABLE = True
except ImportError:
    _PYSIDE6_AVAILABLE = False

if _PYSIDE6_AVAILABLE:
    from src.gui.loader import ProfileLoadWorker, LoadStage, LoadResult
    from src.gui.errors import ErrorKind

from src.core.trace import Trace, TraceMetadata
from src.core.events import SpanEvent, Category
from src.output import chrome_trace


def _write_trace(path: str, *, n_spans: int = 5, command: str = "./a.out") -> None:
    trace = Trace(TraceMetadata(command=command, args=[]))
    for i in range(n_spans):
        trace.add(SpanEvent(name=f"fn{i}", category=Category.GPU_CUDA, start_ns=i * 1000,
                            duration_ns=100, pid=1, tid=1, tags={"type": "kernel"}))
    chrome_trace.write(trace, path)


@unittest.skipUnless(_PYSIDE6_AVAILABLE, "PySide6 not installed (optional gui extra)")
class TestProfileLoadWorkerSynchronous(unittest.TestCase):
    _app = None

    @classmethod
    def setUpClass(cls):
        cls._app = QCoreApplication.instance() or QCoreApplication([])

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._trace_path = os.path.join(self._tmpdir.name, "a.hprofiler.json")
        _write_trace(self._trace_path)

    def tearDown(self):
        self._tmpdir.cleanup()

    def _run(self, **kwargs):
        worker = ProfileLoadWorker(self._trace_path, **kwargs)
        stages, progress, finished, failed, cancelled = [], [], [], [], []
        worker.stageChanged.connect(lambda s, l: stages.append((s, l)))
        worker.progress.connect(lambda i, t: progress.append((i, t)))
        worker.finished.connect(lambda r: finished.append(r))
        worker.failed.connect(lambda e: failed.append(e))
        worker.cancelled.connect(lambda: cancelled.append(1))
        worker.run()
        return worker, stages, progress, finished, failed, cancelled

    def test_stage_sequence_in_order_on_success(self):
        _, stages, _, finished, failed, cancelled = self._run()
        stage_ids = [s for s, _ in stages]
        self.assertEqual(stage_ids, [
            LoadStage.READING_FILE.value, LoadStage.PARSING_JSON.value,
            LoadStage.COMPUTING_DASHBOARD.value, LoadStage.COMPUTING_CALL_TREE.value,
            LoadStage.COMPUTING_FLAME_GRAPH.value, LoadStage.DONE.value,
        ])
        self.assertEqual(len(finished), 1)
        self.assertEqual(failed, [])
        self.assertEqual(cancelled, [])

    def test_stage_labels_are_human_readable_non_empty_strings(self):
        _, stages, *_ = self._run()
        for _, label in stages:
            self.assertTrue(label)
            self.assertIsInstance(label, str)

    def test_finished_payload_has_expected_shape(self):
        _, _, _, finished, _, _ = self._run()
        result: LoadResult = finished[0]
        self.assertEqual(len(result.trace.spans), 5)
        self.assertIsNone(result.trace_b)
        self.assertIn("diagnosis_label", result.dashboard_data)
        self.assertIsInstance(result.call_tree_data, list)
        self.assertIn("name", result.flame_graph_data)

    def test_progress_reported_during_parse(self):
        _, _, progress, *_ = self._run()
        self.assertTrue(progress)

    def test_compare_path_loads_second_trace(self):
        compare_path = os.path.join(self._tmpdir.name, "b.hprofiler.json")
        _write_trace(compare_path, n_spans=3)
        _, stages, _, finished, failed, _ = self._run(compare_path=compare_path)
        self.assertIn(LoadStage.LOADING_COMPARISON.value, [s for s, _ in stages])
        self.assertEqual(len(finished), 1)
        self.assertIsNotNone(finished[0].trace_b)
        self.assertEqual(len(finished[0].trace_b.spans), 3)

    def test_no_compare_path_leaves_trace_b_none(self):
        _, stages, _, finished, _, _ = self._run()
        self.assertNotIn(LoadStage.LOADING_COMPARISON.value, [s for s, _ in stages])
        self.assertIsNone(finished[0].trace_b)

    def test_dark_flag_threaded_into_dashboard_data_colors(self):
        _, _, _, finished_dark, _, _ = self._run(dark=True)
        _, _, _, finished_light, _, _ = self._run(dark=False)
        dark_colors = [f["color"] for f in finished_dark[0].dashboard_data["findings"]]
        light_colors = [f["color"] for f in finished_light[0].dashboard_data["findings"]]
        if dark_colors and light_colors:
            self.assertNotEqual(dark_colors, light_colors)

    # ── Cancellation ──────────────────────────────────────────────────

    def test_cancel_before_run_emits_cancelled_not_finished(self):
        w = ProfileLoadWorker(self._trace_path)
        finished_list, cancelled_list = [], []
        w.finished.connect(lambda r: finished_list.append(r))
        w.cancelled.connect(lambda: cancelled_list.append(1))
        w.cancel()
        w.run()
        self.assertEqual(cancelled_list, [1])
        self.assertEqual(finished_list, [])

    def test_cancel_mid_parse_stops_before_finishing(self):
        # A large-enough trace that cancellation-before-completion is
        # actually exercised (not a race that happens to finish first).
        big_path = os.path.join(self._tmpdir.name, "big.hprofiler.json")
        _write_trace(big_path, n_spans=20_000)
        w = ProfileLoadWorker(big_path)
        finished_list, cancelled_list = [], []
        w.finished.connect(lambda r: finished_list.append(r))
        w.cancelled.connect(lambda: cancelled_list.append(1))

        call_count = [0]
        def cancel_after_first_progress(i, t):
            call_count[0] += 1
            if call_count[0] == 1:
                w.cancel()
        w.progress.connect(cancel_after_first_progress)
        w.run()
        self.assertEqual(cancelled_list, [1])
        self.assertEqual(finished_list, [])

    # ── Error classification ─────────────────────────────────────────

    def test_missing_file_produces_invalid_input_error(self):
        w = ProfileLoadWorker("/tmp/hprofiler_test_does_not_exist_12345.json")
        failed_list = []
        w.failed.connect(lambda e: failed_list.append(e))
        w.run()
        self.assertEqual(len(failed_list), 1)
        self.assertEqual(failed_list[0].kind, ErrorKind.INVALID_INPUT)

    def test_malformed_json_produces_invalid_input_error(self):
        bad_path = os.path.join(self._tmpdir.name, "bad.json")
        with open(bad_path, "w") as f:
            f.write("{not valid json")
        w = ProfileLoadWorker(bad_path)
        failed_list = []
        w.failed.connect(lambda e: failed_list.append(e))
        w.run()
        self.assertEqual(failed_list[0].kind, ErrorKind.INVALID_INPUT)

    def test_foreign_schema_produces_unsupported_data_error(self):
        foreign_path = os.path.join(self._tmpdir.name, "foreign.json")
        with open(foreign_path, "w") as f:
            json.dump({"traceEvents": ["not", "a", "dict"]}, f)
        w = ProfileLoadWorker(foreign_path)
        failed_list = []
        w.failed.connect(lambda e: failed_list.append(e))
        w.run()
        self.assertEqual(len(failed_list), 1)
        self.assertEqual(failed_list[0].kind, ErrorKind.UNSUPPORTED_DATA)

    def test_permission_denied_produces_permission_denied_error(self):
        if os.geteuid() == 0:
            self.skipTest("running as root -- permission bits don't block root")
        restricted_path = os.path.join(self._tmpdir.name, "restricted.json")
        _write_trace(restricted_path)
        os.chmod(restricted_path, 0o000)
        try:
            w = ProfileLoadWorker(restricted_path)
            failed_list = []
            w.failed.connect(lambda e: failed_list.append(e))
            w.run()
            self.assertEqual(len(failed_list), 1)
            self.assertEqual(failed_list[0].kind, ErrorKind.PERMISSION_DENIED)
        finally:
            os.chmod(restricted_path, 0o644)

    def test_error_records_stage_and_file(self):
        w = ProfileLoadWorker("/tmp/hprofiler_test_does_not_exist_12345.json")
        failed_list = []
        w.failed.connect(lambda e: failed_list.append(e))
        w.run()
        err = failed_list[0]
        self.assertTrue(err.stage)
        self.assertIn("does_not_exist", err.file)

    def test_unexpected_exception_becomes_internal_error_not_raised(self):
        # Simulate a bug deep in one of the compute_* functions by
        # pointing at a valid trace but monkeypatching the dashboard
        # compute function to blow up -- proves the worker's own
        # try/except catches genuinely unexpected exceptions too, not
        # just the load_trace_from_json ones.
        import src.gui.bridge as bridge_mod
        original = bridge_mod.compute_dashboard_data
        def boom(*a, **kw):
            raise RuntimeError("surprise bug")
        bridge_mod.compute_dashboard_data = boom
        try:
            w = ProfileLoadWorker(self._trace_path)
            failed_list = []
            w.failed.connect(lambda e: failed_list.append(e))
            w.run()   # must not raise
            self.assertEqual(len(failed_list), 1)
            self.assertEqual(failed_list[0].kind, ErrorKind.INTERNAL_ERROR)
            self.assertIn("surprise bug", failed_list[0].detail)
            self.assertTrue(failed_list[0].traceback_text)
        finally:
            bridge_mod.compute_dashboard_data = original


@unittest.skipUnless(_PYSIDE6_AVAILABLE, "PySide6 not installed (optional gui extra)")
class TestProfileLoadWorkerRealThread(unittest.TestCase):
    """A small number of real-QThread tests proving the cross-thread
    signal marshalling works, plus (see
    test_cancel_called_from_main_thread_reaches_worker_thread below) a
    real cancel-from-the-main-thread test. Uses a real QEventLoop + a
    QTimer timeout (the correct, idiomatic Qt way to block-with-a-
    deadline in a test) instead of a manual processEvents()+time.sleep()
    polling loop -- an earlier version of this test used that polling
    pattern and it produced a real, ugly failure mode: a "QThread:
    Destroyed while thread is still running" warning immediately
    followed by a hard process crash (core dump), because the polling
    loop could exit (deadline reached) without the thread having
    actually been told to quit()/given a chance to wait() for long
    enough, leaving a live QThread whose Python wrapper then got
    garbage-collected out from under the still-running C++ thread.
    QEventLoop.quit() driven directly off the worker's own terminal
    signals (not a polled condition) doesn't have that gap, and every
    test below still explicitly quit()s+wait()s the thread before
    returning as defense in depth."""

    _app = None

    @classmethod
    def setUpClass(cls):
        cls._app = QCoreApplication.instance() or QCoreApplication([])

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._trace_path = os.path.join(self._tmpdir.name, "a.hprofiler.json")
        _write_trace(self._trace_path)

    def tearDown(self):
        self._tmpdir.cleanup()

    def _run_on_thread(self, worker: "ProfileLoadWorker", *, timeout_ms: int = 10_000):
        """Starts `worker` on a real QThread, blocks (via a real
        QEventLoop, not manual polling) until a terminal signal fires or
        `timeout_ms` elapses, then ALWAYS quits+waits the thread before
        returning -- so a dangling still-running QThread can never
        outlive this helper regardless of what happened. Returns
        (results, failures, cancellations, timed_out)."""
        from PySide6.QtCore import QEventLoop, QTimer

        thread = QThread()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)

        results, failures, cancellations = [], [], []
        loop = QEventLoop()
        worker.finished.connect(lambda r: (results.append(r), loop.quit()))
        worker.failed.connect(lambda e: (failures.append(e), loop.quit()))
        worker.cancelled.connect(lambda: (cancellations.append(1), loop.quit()))

        timed_out = []
        timeout_timer = QTimer()
        timeout_timer.setSingleShot(True)
        timeout_timer.timeout.connect(lambda: (timed_out.append(1), loop.quit()))
        timeout_timer.start(timeout_ms)

        thread.start()
        loop.exec()
        timeout_timer.stop()

        try:
            thread.quit()
            thread.wait(5000)
        finally:
            pass

        return results, failures, cancellations, bool(timed_out)

    def test_finished_signal_arrives_via_real_qthread(self):
        worker = ProfileLoadWorker(self._trace_path)
        results, failures, cancellations, timed_out = self._run_on_thread(worker)
        self.assertFalse(timed_out, "worker never signaled completion within the timeout")
        self.assertEqual(len(results), 1)
        self.assertEqual(failures, [])
        self.assertEqual(cancellations, [])
        self.assertGreater(len(results[0].trace.spans), 0)

    def test_cancel_called_from_main_thread_reaches_worker_thread(self):
        # Root cause of an earlier crash-prone version of this test
        # (a hard "QThread: Destroyed while thread is still running"
        # abort): connecting a QTimer.timeout SIGNAL directly to
        # worker.cancel -- a bound @Slot method of a QObject whose
        # thread affinity is the WORKER thread -- makes PySide
        # auto-detect a cross-thread QUEUED connection (same as
        # explicitly using QMetaObject.invokeMethod(..., QueuedConnection)).
        # That queued call can only be DELIVERED once the worker
        # thread's own event loop starts pumping -- but QThread's
        # default run() only calls exec() (starting that loop) AFTER
        # the started-signal-connected worker.run() (the long
        # synchronous parse) returns. That's a deadlock: the queued
        # cancel can't be delivered until the very call it's meant to
        # interrupt finishes on its own, so it never actually cancels
        # anything, worker.run() eventually returns on its own terms
        # only for a leftover queued invocation to land on a thread
        # that's mid-teardown -- corrupting process shutdown later.
        #
        # Fix, confirmed via a standalone probe (5/5 clean runs) before
        # landing here: wrap the call in a plain lambda instead of
        # connecting the bound slot directly. A lambda has no QObject
        # receiver for PySide's thread-affinity detection to key off
        # of, so the connection is NOT auto-queued -- it fires
        # synchronously on the timer's own (main) thread, a plain,
        # immediate, GIL-safe write to worker._cancel_requested. This
        # is exactly the shape a real cancel button's clicked signal
        # must use in production code (see controller.py) -- never
        # `signal.connect(worker.cancel)` directly, always
        # `signal.connect(lambda: worker.cancel())`.
        from PySide6.QtCore import QTimer

        big_path = os.path.join(self._tmpdir.name, "big.hprofiler.json")
        _write_trace(big_path, n_spans=50_000)
        worker = ProfileLoadWorker(big_path)

        thread = QThread()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)

        from PySide6.QtCore import QEventLoop
        results, failures, cancellations = [], [], []
        loop = QEventLoop()
        worker.finished.connect(lambda r: (results.append(r), loop.quit()))
        worker.failed.connect(lambda e: (failures.append(e), loop.quit()))
        worker.cancelled.connect(lambda: (cancellations.append(1), loop.quit()))

        timed_out = []
        timeout_timer = QTimer()
        timeout_timer.setSingleShot(True)
        timeout_timer.timeout.connect(lambda: (timed_out.append(1), loop.quit()))
        timeout_timer.start(10_000)

        fire_cancel = QTimer()
        fire_cancel.setSingleShot(True)
        fire_cancel.timeout.connect(lambda: worker.cancel())
        fire_cancel.start(20)

        thread.start()
        loop.exec()
        timeout_timer.stop()

        thread.quit()
        wait_ok = thread.wait(5000)

        self.assertFalse(timed_out, "worker never reached a terminal state within the timeout")
        self.assertTrue(wait_ok, "worker thread did not stop cleanly after cancellation")
        self.assertEqual(cancellations, [1])
        self.assertEqual(results, [])
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
