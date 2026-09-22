"""
Tests for src/gui/x11_check.py -- the preflight check the GUI's 3-tier
fallback (GPU QML -> software QML -> TUI) uses to decide whether
attempting the Qt GUI is worth it at all, before paying the cost of
importing PySide6.
"""
import os
import socket
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
        # forwarding. host=None means "try local transports" -- it does
        # NOT mean "Unix socket only" (see test_socket_reachable_*
        # below): SSH X11 forwarding is usually TCP loopback, not a Unix
        # socket, since there's no real local X server involved.
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

    def test_socket_reachable_via_tcp_loopback_when_no_unix_socket(self):
        # Reproduces a real `ssh -Y` session onto an HPC login node:
        # sshd sets DISPLAY=localhost:N.0 and proxies X11 over a plain
        # TCP listener on 127.0.0.1:(6000+N) -- there is no
        # /tmp/.X11-unix/XN file at all, since no real X server is
        # involved. Bug: _socket_reachable used to check ONLY the Unix
        # socket for a "local" (host=None) display and never tried the
        # TCP loopback fallback, so it reported a perfectly-working
        # SSH-forwarded display as unreachable.
        num = 9876
        sock_path = f"/tmp/.X11-unix/X{num}"
        self.assertFalse(
            os.path.exists(sock_path),
            f"test assumption violated: {sock_path} exists on this machine",
        )
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            srv.bind(("127.0.0.1", 6000 + num))
            srv.listen(1)
            self.assertTrue(_socket_reachable(f"localhost:{num}.0", timeout=0.5))
        finally:
            srv.close()

    def test_check_x11_available_via_tcp_loopback(self):
        # Same scenario as above, through the full check_x11() entry
        # point (with xdpyinfo's real-handshake step skipped -- our fake
        # listener isn't a real X server and would fail that handshake,
        # which is a separate, correctly-detected failure mode, not the
        # one this test is about).
        num = 9877
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            srv.bind(("127.0.0.1", 6000 + num))
            srv.listen(1)
            with patch.dict(os.environ, {"DISPLAY": f"localhost:{num}.0"}), \
                 patch("src.gui.x11_check.shutil.which", return_value=None):
                status = check_x11(timeout=0.5)
            self.assertTrue(status.available, status.reason)
        finally:
            srv.close()

    def test_reason_is_always_a_nonempty_string(self):
        # Whatever the outcome, callers show `reason` to the user --
        # must never be blank.
        with patch.dict(os.environ, {"DISPLAY": ":9999"}):
            status = check_x11(timeout=0.5)
            self.assertTrue(status.reason)


if __name__ == "__main__":
    unittest.main()
