"""
Subprocess-based test for the POPULATED Compare tab (Phase C) -- a real
second trace loaded via `--compare`. This can't be covered by the shared-
engine in-process test class (tests/test_gui_timeline_hover.py): that
engine is process-global and only one trace pairing can ever be live in
it for the whole test run (already registers ComparisonBridge(trace,
None, theme) there, proving the "no comparison loaded" case instead --
see that file's setUpClass). A real trace_a/trace_b pairing needs a
genuinely separate process, so this spawns `python3 src/gui/app.py
trace_a.json --compare trace_b.json` for real, using the
HPROFILER_GUI_SELFTEST=1 hook (see app.py's own docstring) to load the
QML and report back without opening a real window or blocking on
app.exec().

Matches this project's existing subprocess-isolation philosophy
(src/gui/launch.py's own docstring on why the GUI runs out-of-process at
all): QT_QPA_PLATFORM=offscreen, no real display needed.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.trace import Trace, TraceMetadata
from src.core.events import SpanEvent, Category
from src.output.chrome_trace import write as write_trace

try:
    import PySide6  # noqa: F401
    _PYSIDE6_AVAILABLE = True
except ImportError:
    _PYSIDE6_AVAILABLE = False

_REPO_ROOT = Path(__file__).resolve().parent.parent
_APP_PY = _REPO_ROOT / "src" / "gui" / "app.py"


def _write_trace(path: Path, *, command: str, total_ns: int) -> None:
    trace = Trace(TraceMetadata(command=command, args=[]))
    trace.add(SpanEvent(name="kernel_a", category=Category.GPU_CUDA, start_ns=0,
                         duration_ns=total_ns, pid=1, tid=1, tags={"type": "kernel"}))
    with open(path, "w") as f:
        write_trace(trace, f)


@unittest.skipUnless(_PYSIDE6_AVAILABLE, "PySide6 not installed (optional gui extra)")
class TestGuiCompareLaunchSubprocess(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._trace_a = Path(self._tmpdir.name) / "a.hprofiler.json"
        self._trace_b = Path(self._tmpdir.name) / "b.hprofiler.json"
        _write_trace(self._trace_a, command="./a.out", total_ns=1_000_000)
        _write_trace(self._trace_b, command="./a.out", total_ns=2_000_000)

    def tearDown(self):
        self._tmpdir.cleanup()

    def _run_selftest(self, argv_tail: list[str]) -> dict:
        env = dict(os.environ)
        env["QT_QPA_PLATFORM"] = "offscreen"
        env["HPROFILER_GUI_SELFTEST"] = "1"
        proc = subprocess.run(
            [sys.executable, str(_APP_PY)] + argv_tail,
            env=env, capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(proc.returncode, 0,
                          f"app.py exited {proc.returncode}\nstdout: {proc.stdout}\nstderr: {proc.stderr}")
        # The selftest hook prints exactly one JSON line; stdout could in
        # principle carry other Qt/platform chatter around it, so take
        # the last line rather than assuming it's the only one.
        last_line = proc.stdout.strip().splitlines()[-1]
        return json.loads(last_line)

    def test_populated_compare_tab_loads_clean(self):
        result = self._run_selftest([str(self._trace_a), "--compare", str(self._trace_b)])
        self.assertGreater(result["rootObjects"], 0)
        self.assertEqual(result["warnings"], [])
        self.assertTrue(result["hasComparison"])

    def test_without_compare_still_loads_clean(self):
        result = self._run_selftest([str(self._trace_a)])
        self.assertGreater(result["rootObjects"], 0)
        self.assertEqual(result["warnings"], [])
        self.assertFalse(result["hasComparison"])


if __name__ == "__main__":
    unittest.main()
