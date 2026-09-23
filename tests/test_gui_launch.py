"""
Tests for src/gui/launch.py's launch_gui() -- the 3-tier fallback chain
(GPU QML -> software QML -> TUI). Mocks check_x11() and subprocess.run so
these run with no real X11/PySide6/GUI process needed, matching this
project's existing pattern for CLI-wiring tests.
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.gui.x11_check import X11Status


def _available() -> X11Status:
    return X11Status(True, "$DISPLAY=':1' is reachable", ":1")


class TestLaunchGuiDisasmThreading(unittest.TestCase):
    """Regression tests for a real bug: `hprofiler gui trace.json --disasm`
    accepted the flag but silently dropped it whenever the GUI actually
    launched (launch_gui() had no `disasm` parameter at all, so the flag
    only reached the TUI-fallback code path) -- disasm never appeared in
    a working GUI regardless of the flag. Fixed by threading `disasm`
    through launch_gui() into the subprocess argv; see also
    SourceBridge's live-refresh tests in test_gui_bridge.py for the other
    half of this fix (the GUI process actually noticing collected disasm)."""

    def _run_with_mocked_subprocess(self, disasm: bool):
        from src.gui.launch import launch_gui
        with patch("src.gui.launch.check_x11", return_value=_available()), \
             patch("src.gui.launch.subprocess.run") as mock_run, \
             patch.dict("sys.modules", {"PySide6": MagicMock()}):
            mock_run.return_value = MagicMock(returncode=0)
            launch_gui("/tmp/trace.json", disasm=disasm)
        return mock_run.call_args

    def test_disasm_flag_appended_to_subprocess_argv_when_true(self):
        call_args = self._run_with_mocked_subprocess(disasm=True)
        argv = call_args.args[0]
        self.assertIn("--disasm", argv)
        self.assertEqual(argv[-1], "--disasm")
        self.assertIn("/tmp/trace.json", argv)

    def test_disasm_flag_absent_from_subprocess_argv_when_false(self):
        call_args = self._run_with_mocked_subprocess(disasm=False)
        argv = call_args.args[0]
        self.assertNotIn("--disasm", argv)

    def test_disasm_defaults_to_false(self):
        from src.gui.launch import launch_gui
        with patch("src.gui.launch.check_x11", return_value=_available()), \
             patch("src.gui.launch.subprocess.run") as mock_run, \
             patch.dict("sys.modules", {"PySide6": MagicMock()}):
            mock_run.return_value = MagicMock(returncode=0)
            launch_gui("/tmp/trace.json")  # no disasm= passed at all
        argv = mock_run.call_args.args[0]
        self.assertNotIn("--disasm", argv)


class TestLaunchGuiFallback(unittest.TestCase):
    def test_no_x11_skips_subprocess_entirely(self):
        from src.gui.launch import launch_gui
        with patch("src.gui.launch.check_x11",
                    return_value=X11Status(False, "$DISPLAY is not set", "")), \
             patch("src.gui.launch.subprocess.run") as mock_run:
            shown = launch_gui("/tmp/trace.json", verbose=False)
        self.assertFalse(shown)
        mock_run.assert_not_called()

    def test_gpu_tier_success_skips_software_tier(self):
        from src.gui.launch import launch_gui
        with patch("src.gui.launch.check_x11", return_value=_available()), \
             patch("src.gui.launch.subprocess.run") as mock_run, \
             patch.dict("sys.modules", {"PySide6": MagicMock()}):
            mock_run.return_value = MagicMock(returncode=0)
            shown = launch_gui("/tmp/trace.json", verbose=False)
        self.assertTrue(shown)
        self.assertEqual(mock_run.call_count, 1)

    def test_gpu_tier_failure_retries_with_software_backend(self):
        from src.gui.launch import launch_gui
        with patch("src.gui.launch.check_x11", return_value=_available()), \
             patch("src.gui.launch.subprocess.run") as mock_run, \
             patch.dict("sys.modules", {"PySide6": MagicMock()}):
            mock_run.side_effect = [MagicMock(returncode=1), MagicMock(returncode=0)]
            shown = launch_gui("/tmp/trace.json", verbose=False)
        self.assertTrue(shown)
        self.assertEqual(mock_run.call_count, 2)
        second_call_env = mock_run.call_args_list[1].kwargs["env"]
        self.assertEqual(second_call_env.get("QT_QUICK_BACKEND"), "software")

    def test_both_tiers_failing_falls_back_to_tui(self):
        from src.gui.launch import launch_gui
        with patch("src.gui.launch.check_x11", return_value=_available()), \
             patch("src.gui.launch.subprocess.run") as mock_run, \
             patch.dict("sys.modules", {"PySide6": MagicMock()}):
            mock_run.return_value = MagicMock(returncode=1)
            shown = launch_gui("/tmp/trace.json", verbose=False)
        self.assertFalse(shown)
        self.assertEqual(mock_run.call_count, 2)


class TestLaunchFlamegraphGui(unittest.TestCase):
    """launch_flamegraph_gui() -- the standalone `hprofiler flamegraph
    --gui` popup's own launch path, sharing _run_gui_tiers() with
    launch_gui() but pointed at flamegraph_app.py instead of app.py, and
    taking a folded-stacks file path + title instead of a trace path."""

    def test_argv_points_at_flamegraph_app_with_path_and_title(self):
        from src.gui.launch import launch_flamegraph_gui
        with patch("src.gui.launch.check_x11", return_value=_available()), \
             patch("src.gui.launch.subprocess.run") as mock_run, \
             patch.dict("sys.modules", {"PySide6": MagicMock()}):
            mock_run.return_value = MagicMock(returncode=0)
            shown = launch_flamegraph_gui("/tmp/stacks.txt", "myprog — flame graph (fp)",
                                          verbose=False)
        self.assertTrue(shown)
        argv = mock_run.call_args.args[0]
        self.assertIn("flamegraph_app.py", argv[1])
        self.assertIn("/tmp/stacks.txt", argv)
        self.assertIn("myprog — flame graph (fp)", argv)

    def test_no_x11_skips_subprocess_entirely(self):
        from src.gui.launch import launch_flamegraph_gui
        with patch("src.gui.launch.check_x11",
                    return_value=X11Status(False, "$DISPLAY is not set", "")), \
             patch("src.gui.launch.subprocess.run") as mock_run:
            shown = launch_flamegraph_gui("/tmp/stacks.txt", "title", verbose=False)
        self.assertFalse(shown)
        mock_run.assert_not_called()

    def test_both_tiers_failing_falls_back_to_tui(self):
        from src.gui.launch import launch_flamegraph_gui
        with patch("src.gui.launch.check_x11", return_value=_available()), \
             patch("src.gui.launch.subprocess.run") as mock_run, \
             patch.dict("sys.modules", {"PySide6": MagicMock()}):
            mock_run.return_value = MagicMock(returncode=1)
            shown = launch_flamegraph_gui("/tmp/stacks.txt", "title", verbose=False)
        self.assertFalse(shown)
        self.assertEqual(mock_run.call_count, 2)


if __name__ == "__main__":
    unittest.main()
