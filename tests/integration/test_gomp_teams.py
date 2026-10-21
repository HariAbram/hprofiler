"""
OpenMP barriers of concurrent teams through the real GNU libgomp hook
(tests/fixtures/gomp_teams.c): two nested inner teams of 2 threads each
reach an explicit barrier at overlapping times. Each barrier must carry its
team's id (team=) and the dependency graph must link each thread only to
the last arriver of its own team -- grouping the barriers by name alone made
one rendezvous of all four. Skips without gcc / the built hook.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


class TestGompTeams(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not shutil.which("gcc") or not (ROOT / "build/lib/libhprofiler_gomp.so").exists():
            raise unittest.SkipTest("needs gcc and the built GNU libgomp hook")
        cls.tmp = tempfile.mkdtemp()
        exe = os.path.join(cls.tmp, "gomp_teams")
        subprocess.run(["gcc", "-fopenmp", "-O1", str(ROOT / "tests/fixtures/gomp_teams.c"), "-o", exe],
                       check=True, capture_output=True)
        from src.core.runner import Runner
        cls.trace = Runner([exe], backends=["openmp"]).run()
        cls.barriers = [s for s in cls.trace.iter_spans(categories=("sync",)) if s.name == "omp_barrier"]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(getattr(cls, "tmp", ""), ignore_errors=True)

    def test_barriers_carry_their_team(self):
        self.assertEqual(len(self.barriers), 4)
        teams = [s.tags.get("team") for s in self.barriers]
        self.assertTrue(all(teams), teams)
        self.assertEqual(sorted(teams.count(t) for t in set(teams)), [2, 2])

    def test_arrivals_stay_inside_each_team(self):
        from src.analysis import criticalpath as cp
        spans, preds = cp.build_dependency_graph(self.trace)
        arrivals = [(spans[u], spans[v]) for v, pl in preds.items() for u, k, _c in pl if k == "arrival"]
        self.assertEqual(len(arrivals), 2)
        for last, other in arrivals:
            self.assertEqual(last.tags["team"], other.tags["team"])
            self.assertGreaterEqual(last.start_ns, other.start_ns)


if __name__ == "__main__":
    unittest.main()
