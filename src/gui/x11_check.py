"""
Preflight check for whether a usable X11 display is actually reachable --
deliberately NOT just "is $DISPLAY set" (that's necessary but not
sufficient: a stale/broken forwarding tunnel, a dropped-then-reconnected
SSH session without -X, or X11Forwarding disabled server-side all leave
assorted environment states that are easy to get wrong). Fast and cheap
on purpose: this runs before anything imports PySide6 (a large,
multi-hundred-MB optional dependency), so a cluster job with no GUI
intent at all pays no cost for checking.

Used by launch.py's three-tier fallback (GPU-rendered QML -> software-
rendered QML -> TUI); see that module for how the result here is used.
"""
from __future__ import annotations

import os
import shutil
import socket
import subprocess
from dataclasses import dataclass


@dataclass
class X11Status:
    available: bool
    reason: str          # human-readable, always set (explains why, even when available=True)
    display: str = ""    # the $DISPLAY value that was checked, if any


def _parse_display(display: str) -> tuple[str | None, int]:
    """Parse a $DISPLAY string ("[host]:display[.screen]") into
    (host_or_None, display_number). host is None for a local (Unix
    socket) display -- the common case for SSH X11 forwarding, where
    sshd sets DISPLAY to "localhost:N.0" or "hostname/unix:N.0", both of
    which should be treated as local for socket-path purposes."""
    if ":" not in display:
        raise ValueError(f"not a valid DISPLAY string: {display!r}")
    host_part, rest = display.rsplit(":", 1)
    num_part = rest.split(".", 1)[0]
    display_num = int(num_part)
    host_part = host_part.strip()
    if host_part in ("", "localhost", "unix") or host_part.endswith("/unix"):
        return None, display_num
    return host_part, display_num


def _socket_reachable(display: str, timeout: float) -> bool:
    """Low-level connect test -- succeeds only if something is actually
    listening, without speaking the X11 protocol itself (which would
    need Xlib or a real Qt/X11 client). A closed/refused/timed-out
    connection means the display is not usable regardless of what
    $DISPLAY claims."""
    try:
        host, num = _parse_display(display)
    except ValueError:
        return False

    if host is None:
        sock_path = f"/tmp/.X11-unix/X{num}"
        if not os.path.exists(sock_path):
            return False
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                s.settimeout(timeout)
                s.connect(sock_path)
            return True
        except OSError:
            return False

    try:
        with socket.create_connection((host, 6000 + num), timeout=timeout):
            return True
    except OSError:
        return False


def check_x11(timeout: float = 2.0) -> X11Status:
    """
    Three checks, cheapest first, any failure is conclusive:
      1. $DISPLAY set and non-empty.
      2. A real socket connect succeeds (rules out a stale/broken
         forwarding tunnel that left $DISPLAY set but nothing listening).
      3. If `xdpyinfo` is on PATH, run it -- a real X11 handshake, not
         just a TCP/Unix-socket accept, so it also catches an X server
         that's listening but rejecting the connection (e.g. a stale
         xauth cookie after a reconnect). Skipped, not treated as a
         failure, when xdpyinfo isn't installed -- steps 1-2 are already
         a reasonable signal on their own.
    """
    display = os.environ.get("DISPLAY", "")
    if not display:
        return X11Status(False, "$DISPLAY is not set", display)

    if not _socket_reachable(display, timeout):
        return X11Status(
            False,
            f"$DISPLAY={display!r} is set but nothing is listening there "
            f"(stale or broken X11 forwarding?)",
            display,
        )

    xdpyinfo = shutil.which("xdpyinfo")
    if xdpyinfo:
        try:
            r = subprocess.run(
                [xdpyinfo], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=timeout,
            )
            if r.returncode != 0:
                return X11Status(
                    False,
                    f"$DISPLAY={display!r} accepted a connection but "
                    f"xdpyinfo's handshake failed (exit {r.returncode}) -- "
                    f"likely a stale xauth cookie",
                    display,
                )
        except (subprocess.TimeoutExpired, OSError):
            return X11Status(
                False, f"$DISPLAY={display!r}: xdpyinfo did not respond in time", display)

    return X11Status(True, f"$DISPLAY={display!r} is reachable", display)
