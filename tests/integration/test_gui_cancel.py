"""
Ctrl+C during the GUI's async trace load must cancel it (exit 130) instead
of being ignored. Catching KeyboardInterrupt around the load is not enough:
Python signal handlers only run when the main thread executes bytecode,
which barely happens inside Qt's event loop, so SIGINT mid-load would be
ignored and the window would open anyway.

Runs src/gui/app.py as a real subprocess (as launch.py does) on a trace
large enough to still be loading when the signal arrives.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

try:
    import PySide6  # noqa: F401
    _PYSIDE6 = True
except ImportError:
    _PYSIDE6 = False


@unittest.skipUnless(_PYSIDE6, "PySide6 not installed")
class TestGuiLoadCancellation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.core.trace import Trace, TraceMetadata
        from src.core.events import SpanEvent, Category
        from src.output import chrome_trace
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = os.path.join(cls.tmp.name, "big.hprofiler.json")
        t = Trace(TraceMetadata(command="big"))
        for i in range(600_000):
            t.add(SpanEvent(name=f"f{i % 300}", category=Category.CPU, start_ns=10**12 + i * 100,
                            duration_ns=50, pid=1, tid=1 + i % 8))
        chrome_trace.write(t, cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_sigint_mid_load_cancels_with_exit_130(self):
        env = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}
        proc = subprocess.Popen([sys.executable, str(REPO / "src" / "gui" / "app.py"), self.path],
                                env=env, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True)
        try:
            # Wait until parsing has visibly started, then interrupt.
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                line = proc.stderr.readline()
                if "%" in line or not line:
                    break
            proc.send_signal(signal.SIGINT)
            rest = proc.communicate(timeout=60)[1]
        finally:
            if proc.poll() is None:
                proc.kill()
        self.assertEqual(proc.returncode, 130, rest[-2000:])
        self.assertIn("cancelled", rest)


if __name__ == "__main__":
    unittest.main()
