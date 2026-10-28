"""
`hprofiler critical-path` end to end on a saved trace whose point-to-point
pairing the trace does not decide: the text report's new sections (variants
with modeled lengths, alternative worlds, blame, "probabilities:
unquantified"), --json (the whole analysis as data) and --export (the
annotated Chrome trace's path_robust= / blame_ns= tags).
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.core.trace_io import write_json_export
from tests import test_cp_alternatives as tca


class TestCriticalPathCli(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        d = Path(cls.tmp.name)
        trace = tca.TestPointToPointAlignment().trace()
        src = d / "t.json"
        write_json_export(trace, str(src))
        cls.json_out, cls.export = d / "cp.json", d / "cp_export.json"
        r = subprocess.run([sys.executable, str(ROOT / "hprofiler"), "critical-path", str(src),
                            "--json", str(cls.json_out), "--export", str(cls.export)],
                           capture_output=True, text=True, timeout=180)
        cls.r = r

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_text_report(self):
        self.assertEqual(self.r.returncode, 0, self.r.stderr[-2000:])
        out = self.r.stdout
        for needle in ("Variants (where the trace cannot decide how to count time)",
                       "Alternative worlds (dependencies the trace does not decide)",
                       "choices: p2p_alignment 1", "probabilities: unquantified",
                       "Blame (every gated call's waiting time", "gated tails as processing"):
            self.assertIn(needle, out)

    def test_json(self):
        d = json.loads(self.json_out.read_text())
        self.assertEqual(d["schema"], "hprofiler.critical_path/1")
        self.assertEqual(d["alternatives"]["units"], {"p2p_alignment": 1})
        self.assertIn("ambiguous", d["evidence"]["by_status"])
        self.assertEqual(d["probabilities"], {"status": "unquantified"})
        self.assertEqual(set(d["variants"]), {"default", "unrecorded_as_work", "direct_evidence_only",
                                              "gated_tail_as_work"})
        self.assertTrue(d["dependencySteps"])
        self.assertIn("clocks", d)

    def test_export_tags(self):
        events = json.loads(self.export.read_text())
        events = events["traceEvents"] if isinstance(events, dict) else events
        args = [e.get("args", {}) for e in events if isinstance(e, dict)]
        self.assertTrue(any(a.get("path_robust") in ("every", "some") for a in args))
        self.assertTrue(any("blame_ns" in a for a in args))


if __name__ == "__main__":
    unittest.main()
