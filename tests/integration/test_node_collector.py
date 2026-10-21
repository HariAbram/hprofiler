"""
Node-collector mode: one `hprofiler collect` for every process of the node,
the processes started with `hprofiler exec` (the hooks preloaded, no Python
left in the process):

  * 4 real MPI ranks (tests/fixtures/mpi_status.c, Intel MPI as in
    test_mpi_multirank.py) end up in one trace -- every rank, every receive
    paired -- and the collector ends by itself after --idle-exit;
  * `hprofiler exec` started before its collector waits for it to listen,
    so nothing is lost (a GNU libgomp program).

Skips without the built hooks, a compiler or a 4-rank MPI.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
HP = [sys.executable, str(ROOT / "hprofiler")]


def _collect(sock, out, backends, env=None):
    return subprocess.Popen(HP + ["collect", "-b", backends, "--socket", sock, "-o", out, "--idle-exit", "1"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)


class TestNodeCollector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_mpi_ranks_in_one_node_trace(self):
        from tests.integration.test_mpi_multirank import _mpi_env
        env = _mpi_env()
        mpicc = shutil.which("mpicc", path=env.get("PATH"))
        mpirun = shutil.which("mpirun", path=env.get("PATH"))
        if not mpicc or not mpirun or not (ROOT / "build/lib/libhprofiler_mpi.so").exists():
            self.skipTest("needs mpicc/mpirun and the MPI hook")
        exe = os.path.join(self.tmp, "mpi_status")
        if subprocess.run([mpicc, "-O1", str(ROOT / "tests/fixtures/mpi_status.c"), "-o", exe], env=env,
                          capture_output=True).returncode:
            self.skipTest("cannot build the fixture")
        sock = f"@hp-test-node-{os.getpid()}"
        out = os.path.join(self.tmp, "node.json")
        col = _collect(sock, out, "mpi", env)
        try:
            app = subprocess.run([mpirun, "-np", "4"] + HP + ["exec", "-b", "mpi", "--socket", sock, "--", exe],
                                 env=env, capture_output=True, text=True, timeout=180)
            if "requires exactly 4 ranks" in app.stdout:
                self.skipTest("the MPI launcher does not start 4 real ranks")
            self.assertEqual(app.returncode, 0, app.stderr[-1500:])
            log, _ = col.communicate(timeout=120)              # ends by itself (idle exit)
        finally:
            if col.poll() is None:
                col.kill()
        self.assertEqual(col.returncode, 0, log)
        from src.core.trace_io import open_trace
        from src.analysis import criticalpath as cp
        t = open_trace(out)
        self.assertEqual(len(t.store.pids()), 4)
        self.assertEqual(sorted({s.tags.get("rank") for s in t.iter_spans(categories=("mpi",))}),
                         ["0", "1", "2", "3"])
        diag = cp.mpi_dependency_diagnostics(t)
        self.assertEqual(diag["receives"], diag["matched"])
        self.assertEqual(t.metadata.command, "(node collector)")

    def test_exec_waits_for_its_collector(self):
        if not shutil.which("gcc") or not (ROOT / "build/lib/libhprofiler_gomp.so").exists():
            self.skipTest("needs gcc and the GNU libgomp hook")
        exe = os.path.join(self.tmp, "gomp_mini")
        subprocess.run(["gcc", "-fopenmp", "-O1", str(ROOT / "tests/fixtures/gomp_mini.c"), "-o", exe],
                       check=True, capture_output=True)
        sock = f"@hp-test-wait-{os.getpid()}"
        app = subprocess.Popen(HP + ["exec", "-b", "openmp", "--socket", sock, "--", exe],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        time.sleep(1.5)                                        # no collector yet: exec waits
        self.assertIsNone(app.poll())
        out = os.path.join(self.tmp, "late.json")
        col = _collect(sock, out, "openmp")
        try:
            app.communicate(timeout=120)
            log, _ = col.communicate(timeout=120)
        finally:
            for p in (app, col):
                if p.poll() is None:
                    p.kill()
        self.assertEqual(app.returncode, 0)
        from src.core.trace_io import open_trace
        t = open_trace(out)
        self.assertGreater(t.span_count(), 0, log)
        self.assertGreater(sum(1 for s in t.iter_spans() if s.name == "omp_parallel_region"), 0)


if __name__ == "__main__":
    unittest.main()
