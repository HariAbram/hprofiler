"""
Tests for src/gui/controller.py -- LoadController's QThread lifecycle,
the "Open Profile" fail-fast validation + spawn-and-watch subprocess
path (mocking subprocess.Popen, matching test_gui_launch.py's
established convention), and AppController's wiring of the two
together. No QQmlApplicationEngine anywhere in this file -- safe
alongside tests/test_gui_timeline_hover.py's one-engine-per-process
constraint (see that file's docstring).
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QCoreApplication
    _PYSIDE6_AVAILABLE = True
except ImportError:
    _PYSIDE6_AVAILABLE = False

if _PYSIDE6_AVAILABLE:
    from src.gui.controller import (
        AppController, LoadController, ProfileOpenWatcher, READY_MARKER_ENV,
        build_profile_argv, open_profile_subprocess, run_load_blocking,
        validate_profile_path,
    )
    from src.gui.errors import ErrorKind

from src.core.trace import Trace, TraceMetadata
from src.core.events import SpanEvent, Category
from src.output import chrome_trace


def _write_trace(path: str, *, n_spans: int = 5) -> None:
    trace = Trace(TraceMetadata(command="a.out", args=[]))
    for i in range(n_spans):
        trace.add(SpanEvent(name=f"fn{i}", category=Category.CPU, start_ns=i * 1000,
                            duration_ns=100, pid=1, tid=1))
    chrome_trace.write(trace, path)


@unittest.skipUnless(_PYSIDE6_AVAILABLE, "PySide6 not installed (optional gui extra)")
class TestValidateProfilePath(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_valid_trace_passes(self):
        path = os.path.join(self._tmpdir.name, "a.hprofiler.json")
        _write_trace(path)
        self.assertIsNone(validate_profile_path(path))

    def test_missing_file_is_invalid_input(self):
        err = validate_profile_path(os.path.join(self._tmpdir.name, "nope.json"))
        self.assertIsNotNone(err)
        self.assertEqual(err.kind, ErrorKind.INVALID_INPUT)

    def test_directory_is_invalid_input(self):
        err = validate_profile_path(self._tmpdir.name)
        self.assertIsNotNone(err)
        self.assertEqual(err.kind, ErrorKind.INVALID_INPUT)

    def test_non_json_content_is_invalid_input(self):
        path = os.path.join(self._tmpdir.name, "notjson.txt")
        with open(path, "w") as f:
            f.write("this is not json at all")
        err = validate_profile_path(path)
        self.assertIsNotNone(err)
        self.assertEqual(err.kind, ErrorKind.INVALID_INPUT)

    def test_empty_file_is_invalid_input(self):
        path = os.path.join(self._tmpdir.name, "empty.json")
        Path(path).touch()
        err = validate_profile_path(path)
        self.assertIsNotNone(err)
        self.assertEqual(err.kind, ErrorKind.INVALID_INPUT)

    def test_permission_denied_is_classified(self):
        if os.geteuid() == 0:
            self.skipTest("running as root -- permission bits don't block root")
        path = os.path.join(self._tmpdir.name, "restricted.json")
        _write_trace(path)
        os.chmod(path, 0o000)
        try:
            err = validate_profile_path(path)
            self.assertIsNotNone(err)
            self.assertEqual(err.kind, ErrorKind.PERMISSION_DENIED)
        finally:
            os.chmod(path, 0o644)

    def test_error_records_the_file(self):
        path = os.path.join(self._tmpdir.name, "nope.json")
        err = validate_profile_path(path)
        self.assertEqual(err.file, path)


@unittest.skipUnless(_PYSIDE6_AVAILABLE, "PySide6 not installed (optional gui extra)")
class TestBuildProfileArgv(unittest.TestCase):
    def test_basic_argv_shape(self):
        argv, env = build_profile_argv("/tmp/trace.json")
        self.assertEqual(argv[0], sys.executable)
        self.assertTrue(argv[1].endswith("app.py"))
        self.assertEqual(argv[2], "/tmp/trace.json")

    def test_compare_and_disasm_appended_like_launch_py(self):
        argv, env = build_profile_argv("/tmp/a.json", compare_path="/tmp/b.json", disasm=True)
        self.assertIn("--compare", argv)
        self.assertIn("/tmp/b.json", argv)
        self.assertEqual(argv[-1], "--disasm")

    def test_ready_marker_goes_in_env_not_argv(self):
        argv, env = build_profile_argv("/tmp/trace.json", ready_marker="/tmp/marker.txt")
        self.assertNotIn("/tmp/marker.txt", argv)
        self.assertEqual(env[READY_MARKER_ENV], "/tmp/marker.txt")

    def test_no_ready_marker_means_env_var_absent(self):
        argv, env = build_profile_argv("/tmp/trace.json")
        self.assertNotIn(READY_MARKER_ENV, env)

    def test_env_inherits_current_environment(self):
        os.environ["_HPROFILER_TEST_SENTINEL"] = "1"
        try:
            argv, env = build_profile_argv("/tmp/trace.json")
            self.assertEqual(env.get("_HPROFILER_TEST_SENTINEL"), "1")
        finally:
            del os.environ["_HPROFILER_TEST_SENTINEL"]


@unittest.skipUnless(_PYSIDE6_AVAILABLE, "PySide6 not installed (optional gui extra)")
class TestOpenProfileSubprocess(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_bad_path_spawns_nothing(self):
        with patch("src.gui.controller.subprocess.Popen") as mock_popen:
            popen, watcher, err = open_profile_subprocess(
                os.path.join(self._tmpdir.name, "nope.json"))
        mock_popen.assert_not_called()
        self.assertIsNone(popen)
        self.assertIsNone(watcher)
        self.assertIsNotNone(err)
        self.assertEqual(err.kind, ErrorKind.INVALID_INPUT)

    def test_good_path_spawns_a_subprocess(self):
        path = os.path.join(self._tmpdir.name, "a.hprofiler.json")
        _write_trace(path)
        with patch("src.gui.controller.subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock(poll=MagicMock(return_value=None))
            popen, watcher, err = open_profile_subprocess(path)
        mock_popen.assert_called_once()
        self.assertIsNone(err)
        self.assertIsNotNone(popen)
        self.assertIsInstance(watcher, ProfileOpenWatcher)

    def test_popen_oserror_is_classified_not_raised(self):
        path = os.path.join(self._tmpdir.name, "a.hprofiler.json")
        _write_trace(path)
        with patch("src.gui.controller.subprocess.Popen", side_effect=OSError("no such executable")):
            popen, watcher, err = open_profile_subprocess(path)
        self.assertIsNone(popen)
        self.assertIsNone(watcher)
        self.assertIsNotNone(err)


@unittest.skipUnless(_PYSIDE6_AVAILABLE, "PySide6 not installed (optional gui extra)")
class TestProfileOpenWatcher(unittest.TestCase):
    _app = None

    @classmethod
    def setUpClass(cls):
        cls._app = QCoreApplication.instance() or QCoreApplication([])

    def setUp(self):
        fd, self._marker_path = tempfile.mkstemp(prefix="hprofiler-test-marker-")
        os.close(fd)
        os.unlink(self._marker_path)  # watcher expects it absent until "ready"

    def tearDown(self):
        Path(self._marker_path).unlink(missing_ok=True)

    def _pump_until(self, predicate, *, timeout_s=5.0):
        from PySide6.QtCore import QElapsedTimer
        t = QElapsedTimer()
        t.start()
        while not predicate() and t.elapsed() < timeout_s * 1000:
            self._app.processEvents()

    def test_marker_appearing_emits_ready(self):
        popen = MagicMock(poll=MagicMock(return_value=None))
        watcher = ProfileOpenWatcher(popen, self._marker_path, timeout_s=5.0)
        results = []
        watcher.ready.connect(lambda: results.append("ready"))
        watcher.failed.connect(lambda e: results.append(("failed", e)))
        watcher.start()

        Path(self._marker_path).touch()
        self._pump_until(lambda: results)

        self.assertEqual(results, ["ready"])
        self.assertFalse(Path(self._marker_path).exists(), "watcher should clean up its marker file")

    def test_process_exiting_early_emits_failed(self):
        popen = MagicMock(poll=MagicMock(return_value=1), stderr=None)
        watcher = ProfileOpenWatcher(popen, self._marker_path, timeout_s=5.0)
        results = []
        watcher.ready.connect(lambda: results.append("ready"))
        watcher.failed.connect(lambda e: results.append(("failed", e)))
        watcher.start()

        self._pump_until(lambda: results)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "failed")

    def test_timeout_emits_failed_without_killing_the_process(self):
        popen = MagicMock(poll=MagicMock(return_value=None))
        watcher = ProfileOpenWatcher(popen, self._marker_path, timeout_s=0.05)
        results = []
        watcher.failed.connect(lambda e: results.append(e))
        watcher.start()

        self._pump_until(lambda: results, timeout_s=3.0)

        self.assertEqual(len(results), 1)
        popen.kill.assert_not_called()
        popen.terminate.assert_not_called()

    def test_only_one_terminal_signal_ever_fires(self):
        # Marker appears AND the process happens to still show poll()
        # is None (still running, as expected once its own window is
        # up) -- ready must win, and nothing should double-fire on a
        # later poll tick.
        popen = MagicMock(poll=MagicMock(return_value=None))
        watcher = ProfileOpenWatcher(popen, self._marker_path, timeout_s=5.0)
        results = []
        watcher.ready.connect(lambda: results.append("ready"))
        watcher.failed.connect(lambda e: results.append("failed"))
        watcher.start()

        Path(self._marker_path).touch()
        self._pump_until(lambda: results)
        # Let a few more poll ticks happen -- must stay at exactly one signal.
        from PySide6.QtCore import QElapsedTimer
        t = QElapsedTimer()
        t.start()
        while t.elapsed() < 500:
            self._app.processEvents()

        self.assertEqual(results, ["ready"])


@unittest.skipUnless(_PYSIDE6_AVAILABLE, "PySide6 not installed (optional gui extra)")
class TestLoadController(unittest.TestCase):
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

    def test_run_load_blocking_returns_finished_with_a_result(self):
        loader = LoadController(self._trace_path)
        kind, value = run_load_blocking(loader)
        self.assertEqual(kind, "finished")
        self.assertGreater(len(value.trace.spans), 0)

    def test_run_load_blocking_reports_stage_changes(self):
        loader = LoadController(self._trace_path)
        stages = []
        kind, value = run_load_blocking(loader, on_stage=lambda s, label: stages.append(s))
        self.assertEqual(kind, "finished")
        self.assertIn("done", stages)

    def test_run_load_blocking_missing_file_fails_cleanly(self):
        loader = LoadController(os.path.join(self._tmpdir.name, "nope.json"))
        kind, value = run_load_blocking(loader)
        self.assertEqual(kind, "failed")
        self.assertEqual(value.kind, ErrorKind.INVALID_INPUT)

    def test_cancel_before_start_produces_cancelled_outcome(self):
        loader = LoadController(self._trace_path)
        loader.cancel()
        kind, value = run_load_blocking(loader)
        self.assertEqual(kind, "cancelled")
        self.assertIsNone(value)


@unittest.skipUnless(_PYSIDE6_AVAILABLE, "PySide6 not installed (optional gui extra)")
class TestAppController(unittest.TestCase):
    _app = None

    @classmethod
    def setUpClass(cls):
        cls._app = QCoreApplication.instance() or QCoreApplication([])

    def test_trace_path_and_compare_path_properties(self):
        ctrl = AppController("/tmp/a.json", "/tmp/b.json")
        self.assertEqual(ctrl.tracePath, "/tmp/a.json")
        self.assertEqual(ctrl.comparePath, "/tmp/b.json")

    def test_compare_path_defaults_to_empty_string(self):
        ctrl = AppController("/tmp/a.json")
        self.assertEqual(ctrl.comparePath, "")

    def test_open_profile_validation_failure_emits_failed_signal(self):
        ctrl = AppController("/tmp/a.json")
        results = []
        ctrl.profileOpenFailed.connect(lambda e: results.append(e))
        opening = []
        ctrl.profileOpening.connect(lambda: opening.append(1))

        with patch("src.gui.controller.subprocess.Popen") as mock_popen:
            ctrl.openProfile("/tmp/does_not_exist_at_all.json")

        mock_popen.assert_not_called()
        self.assertEqual(len(results), 1)
        self.assertEqual(opening, [])
        # A plain dict (QVariantMap), not the raw HprofilerLoadError --
        # QML reads err.message/err.kind/etc as ordinary properties.
        self.assertIsInstance(results[0], dict)
        self.assertEqual(results[0]["kind"], ErrorKind.INVALID_INPUT.value)
        self.assertIn("message", results[0])

    def test_open_profile_success_emits_opening_and_wires_watcher(self):
        ctrl = AppController("/tmp/a.json")
        opening = []
        ctrl.profileOpening.connect(lambda: opening.append(1))

        fake_watcher = MagicMock()
        with patch("src.gui.controller.open_profile_subprocess",
                   return_value=(MagicMock(), fake_watcher, None)):
            ctrl.openProfile("/tmp/some_trace.json")

        self.assertEqual(opening, [1])
        fake_watcher.start.assert_called_once()
        fake_watcher.ready.connect.assert_called_once()
        fake_watcher.failed.connect.assert_called_once()

    def test_open_profile_ready_quits_the_application(self):
        ctrl = AppController("/tmp/a.json")
        fake_watcher = MagicMock()
        with patch("src.gui.controller.open_profile_subprocess",
                   return_value=(MagicMock(), fake_watcher, None)):
            ctrl.openProfile("/tmp/some_trace.json")

        with patch("src.gui.controller.QCoreApplication.instance") as mock_instance:
            mock_app = MagicMock()
            mock_instance.return_value = mock_app
            ctrl._on_open_ready()
            mock_app.quit.assert_called_once()

    def test_log_file_path_reflects_current_log_path(self):
        ctrl = AppController("/tmp/a.json")
        with patch("src.gui.controller.current_log_path", return_value=Path("/tmp/hprofiler-gui.log")):
            self.assertEqual(ctrl.logFilePath(), "/tmp/hprofiler-gui.log")

    def test_log_file_path_empty_when_not_yet_configured(self):
        ctrl = AppController("/tmp/a.json")
        with patch("src.gui.controller.current_log_path", return_value=None):
            self.assertEqual(ctrl.logFilePath(), "")

    def test_open_log_file_opens_the_current_log_path(self):
        ctrl = AppController("/tmp/a.json")
        with patch("src.gui.controller.current_log_path", return_value=Path("/tmp/hprofiler-gui.log")), \
             patch("src.gui.controller.QDesktopServices") as mock_dsvc:
            ctrl.openLogFile()
        mock_dsvc.openUrl.assert_called_once()

    def test_open_log_file_is_a_noop_when_no_log_yet(self):
        ctrl = AppController("/tmp/a.json")
        with patch("src.gui.controller.current_log_path", return_value=None), \
             patch("src.gui.controller.QDesktopServices") as mock_dsvc:
            ctrl.openLogFile()
        mock_dsvc.openUrl.assert_not_called()

    def test_open_profile_watcher_failure_emits_profile_open_failed(self):
        ctrl = AppController("/tmp/a.json")
        results = []
        ctrl.profileOpenFailed.connect(lambda e: results.append(e))
        fake_watcher = MagicMock()
        with patch("src.gui.controller.open_profile_subprocess",
                   return_value=(MagicMock(), fake_watcher, None)):
            ctrl.openProfile("/tmp/some_trace.json")

        from src.gui.errors import HprofilerLoadError
        err = HprofilerLoadError(kind=ErrorKind.INTERNAL_ERROR, message="boom")
        ctrl._on_open_failed(err)

        # A QVariantMap (err.to_dict()'s shape), not the raw
        # HprofilerLoadError object -- QML needs plain dict keys to read
        # err.message/err.detail/etc as properties.
        self.assertEqual(results, [err.to_dict()])


if __name__ == "__main__":
    unittest.main()
