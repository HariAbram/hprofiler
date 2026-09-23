"""
Entry point the CLI calls to show a trace: launch_gui(trace) -- the
three-tier fallback chain:

  1. GPU-rendered QML   (Qt Quick's default scene graph, OpenGL/GLX)
  2. Software-rendered QML (QT_QUICK_BACKEND=software -- no GPU/GLX
     needed; the answer to indirect-GLX-over-X11-forwarding being slow
     or outright broken, a well-known Qt Quick pain point that would
     otherwise undermine the whole point of offering a GUI over ssh -X)
  3. The existing Textual TUI (src/ui/app.py's launch_viewer) -- always
     works over a bare SSH session, no X11 needed at all.

Tier 1/2 are only attempted when x11_check.check_x11() says a display is
actually reachable AND `import PySide6` succeeds -- both checked BEFORE
committing to a tier, so a cluster job with neither never pays for either.
Tier 1 vs 2 isn't separately probed in advance (there's no cheap, fast
way to know if GLX will actually render well without just trying it) --
tier 1 is attempted first every time a display is reachable, and if the
whole GUI process exits non-zero or raises, tier 2 is retried once with
the software backend before giving up to the TUI. This means a hard GLX
crash costs one extra (fast) subprocess attempt, not a silent black
window -- acceptable given a broken-GLX failure is exactly the scenario
this fallback exists for.
"""
from __future__ import annotations

import os
import subprocess
import sys

from .x11_check import check_x11


def launch_gui(trace_path: str, verbose: bool = True, disasm: bool = False) -> bool:
    """
    Attempts the Qt/QML GUI on a trace JSON file already written to disk
    (both `hprofiler run --gui` and `hprofiler gui <trace.json>` go
    through this same path -- see hprofiler's CLI). Returns True if the
    GUI was shown (regardless of tier), False if every tier was skipped
    or failed and the caller should fall back to launch_viewer() (TUI).

    Runs the actual Qt application in a SEPARATE PROCESS, not this one,
    specifically so a GLX crash (tier 1 failing) can't take down the CLI
    process itself -- QGuiApplication + an OpenGL context crashing hard
    (a real possibility with broken/partial GLX over some X11 forwarding
    setups) is a segfault-class failure in the underlying Qt/driver
    stack, not a catchable Python exception.

    `disasm`: forwarded to the GUI subprocess as `--disasm` so it starts
    background disassembly collection for any function whose call site
    resolved (sym=/lib= tag) but wasn't already disassembled in the trace
    file -- mirrors `hprofiler view --disasm`'s TUI behavior. Previously
    this flag existed on the `hprofiler gui`/`run --gui` CLI commands but
    was silently dropped whenever the GUI actually launched (only used on
    the TUI-fallback path), so `--disasm` had no effect on a working GUI
    -- a real bug, not by design.
    """
    def log(msg: str) -> None:
        if verbose:
            print(f"[hprofiler][gui] {msg}", file=sys.stderr)

    status = check_x11()
    if not status.available:
        log(f"no usable X11 display ({status.reason}) -- falling back to the TUI")
        return False

    try:
        import PySide6  # noqa: F401
    except ImportError:
        log("PySide6 is not installed (pip install 'hprofiler[gui]') -- "
            "falling back to the TUI")
        return False

    log(f"{status.reason} -- attempting the GUI (GPU-rendered)")
    app_main = os.path.join(os.path.dirname(__file__), "app.py")
    argv = [sys.executable, app_main, trace_path] + (["--disasm"] if disasm else [])

    for tier_name, extra_env in (
        ("GPU-rendered", {}),
        ("software-rendered", {"QT_QUICK_BACKEND": "software"}),
    ):
        env = dict(os.environ)
        env.update(extra_env)
        try:
            rc = subprocess.run(argv, env=env).returncode
        except OSError as e:
            log(f"could not launch the GUI process ({tier_name}): {e}")
            rc = 1

        if rc == 0:
            return True
        log(f"{tier_name} GUI attempt exited with code {rc}"
            + (" -- retrying with software rendering" if not extra_env else
               " -- falling back to the TUI"))

    return False
