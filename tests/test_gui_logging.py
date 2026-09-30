"""
Tests for src/gui/logging_setup.py -- log file creation, idempotent
configuration (repeated setup_logging() calls don't duplicate handlers or
lose the established path), and log_error()'s HprofilerLoadError
formatting. Needs no QGuiApplication for the temp-dir-overridden path
this file exercises exclusively (the QStandardPaths default path DOES
need one, but every test here passes an explicit log_dir, matching
settings.py's tests never touching the real user config location).
"""
import logging
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.gui import logging_setup
from src.gui.errors import HprofilerLoadError, ErrorKind


class TestLoggingSetup(unittest.TestCase):
    def setUp(self):
        # logging_setup module state (_configured/_log_path) is
        # process-global by design (mirrors real usage: setup_logging()
        # is meant to be called once per process) -- reset it before
        # each test so tests don't see each other's configuration.
        logging_setup._configured = False
        logging_setup._log_path = None
        logger = logging.getLogger(logging_setup._LOGGER_NAME)
        for h in list(logger.handlers):
            logger.removeHandler(h)
            h.close()

    def test_creates_log_file_in_given_directory(self):
        with tempfile.TemporaryDirectory() as d:
            path = logging_setup.setup_logging(log_dir=d)
            self.assertTrue(path.exists())
            self.assertEqual(path.parent, Path(d))

    def test_creates_directory_if_missing(self):
        with tempfile.TemporaryDirectory() as d:
            nested = Path(d) / "nested" / "dir"
            path = logging_setup.setup_logging(log_dir=nested)
            self.assertTrue(path.exists())

    def test_second_call_returns_same_path_and_does_not_duplicate_handlers(self):
        with tempfile.TemporaryDirectory() as d:
            path1 = logging_setup.setup_logging(log_dir=d)
            logger = logging_setup.get_logger()
            handler_count_after_first = len(logger.handlers)
            path2 = logging_setup.setup_logging(log_dir=d)
            self.assertEqual(path1, path2)
            self.assertEqual(len(logger.handlers), handler_count_after_first)

    def test_get_logger_before_setup_does_not_raise(self):
        logger = logging_setup.get_logger()
        logger.info("this goes nowhere, but must not raise")

    def test_current_log_path_none_before_setup(self):
        self.assertIsNone(logging_setup.current_log_path())

    def test_current_log_path_set_after_setup(self):
        with tempfile.TemporaryDirectory() as d:
            path = logging_setup.setup_logging(log_dir=d)
            self.assertEqual(logging_setup.current_log_path(), path)

    def test_logged_messages_appear_in_the_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = logging_setup.setup_logging(log_dir=d)
            logging_setup.get_logger().info("distinctive marker 12345")
            self.assertIn("distinctive marker 12345", path.read_text())

    def test_log_error_writes_message_and_stage(self):
        with tempfile.TemporaryDirectory() as d:
            path = logging_setup.setup_logging(log_dir=d)
            err = HprofilerLoadError(kind=ErrorKind.INVALID_INPUT, message="bad file", stage="parsing_json")
            logging_setup.log_error(err)
            text = path.read_text()
            self.assertIn("bad file", text)
            self.assertIn("parsing_json", text)

    def test_log_error_includes_traceback_text_when_present(self):
        with tempfile.TemporaryDirectory() as d:
            path = logging_setup.setup_logging(log_dir=d)
            try:
                raise RuntimeError("distinctive failure")
            except RuntimeError as exc:
                err = HprofilerLoadError.wrap_unexpected(exc, stage="computing_dashboard")
            logging_setup.log_error(err)
            text = path.read_text()
            self.assertIn("distinctive failure", text)
            self.assertIn("RuntimeError", text)

    def test_log_error_works_without_traceback_text(self):
        with tempfile.TemporaryDirectory() as d:
            path = logging_setup.setup_logging(log_dir=d)
            err = HprofilerLoadError(kind=ErrorKind.METRIC_UNAVAILABLE, message="no PMU counters")
            logging_setup.log_error(err)   # must not raise
            self.assertIn("no PMU counters", path.read_text())


if __name__ == "__main__":
    unittest.main()
