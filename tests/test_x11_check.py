"""
Tests for src/gui/x11_check.py -- the preflight check the GUI's 3-tier
fallback (GPU QML -> software QML -> TUI) uses to decide whether
attempting the Qt GUI is worth it at all, before paying the cost of
importing PySide6.
"""
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.gui.x11_check import _parse_display, _socket_reachable, check_x11


class TestParseDisplay(unittest.TestCase):
    def test_local_display_no_host(self):
        self.assertEqual(_parse_display(":1"), (None, 1))

    def test_local_display_with_screen(self):
        self.assertEqual(_parse_display(":1.0"), (None, 1))

    def test_localhost_treated_as_local(self):
        # sshd commonly sets DISPLAY to "localhost:N.0" for X11
        # forwarding -- must resolve to the Unix-socket path, not a TCP
        # connection to literal "localhost", since that's not always
        # where the X11 forwarding listener actually is.
        self.assertEqual(_parse_display("localhost:10.0"), (None, 10))

    def test_remote_host_display(self):
        self.assertEqual(_parse_display("myhost:0"), ("myhost", 0))

    def test_invalid_display_raises(self):
        with self.assertRaises(ValueError):
            _parse_display("not-a-display")


class TestCheckX11(unittest.TestCase):
    def test_unset_display_is_unavailable(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DISPLAY", None)
            status = check_x11()
            self.assertFalse(status.available)
            self.assertIn("not set", status.reason)

    def test_empty_display_is_unavailable(self):
        with patch.dict(os.environ, {"DISPLAY": ""}):
            status = check_x11()
            self.assertFalse(status.available)

    def test_display_set_but_nothing_listening(self):
        # A display number essentially guaranteed not to have a real X
        # server (and no /tmp/.X11-unix/X9999 socket file either).
        with patch.dict(os.environ, {"DISPLAY": ":9999"}):
            status = check_x11(timeout=0.5)
            self.assertFalse(status.available)
            self.assertIn(":9999", status.reason)

    def test_socket_reachable_false_for_nonexistent_socket(self):
        self.assertFalse(_socket_reachable(":9999", timeout=0.5))

    def test_reason_is_always_a_nonempty_string(self):
        # Whatever the outcome, callers show `reason` to the user --
        # must never be blank.
        with patch.dict(os.environ, {"DISPLAY": ":9999"}):
            status = check_x11(timeout=0.5)
            self.assertTrue(status.reason)


if __name__ == "__main__":
    unittest.main()
