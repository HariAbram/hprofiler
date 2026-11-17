"""
A CUDA process that vforks and execs children, and synchronizes on events
through the runtime and the driver API, through the real CUDA hook and
CUPTI on a GPU (tests/fixtures/cuda_vfork_events.cu: a child before the
first CUDA call -- as Triton's driver compiles cuda_utils.c with cc -- and
another in mid-run):

  * process identity: every span carries the program's own pid and the
    trace has one process-begin marker, the program's. The vfork child used
    to run the hook's execv wrapper in the parent's memory, fill the
    transport's pid cache with its own pid (the parent had recorded nothing
    yet) and shut the parent's transport down: every span had the child's
    pid and no marker was sent (campaign smoke2, triton/vector-add);
  * event synchronizations: every CUPTI synchronization record of an event
    (sync=event) has its host call -- cudaEventSynchronize, cudaEventQuery
    (recorded when the event was complete; the not-ready query is not, and
    CUPTI reports none for it), cuEventSynchronize and cuEventQuery (driver
    API: from the CUPTI callback, src=cupti_cb, with event=);
  * the critical path's window starts at the marker and nothing is left
    unconnected after the walk.

Skips without nvcc, a usable GPU, CUPTI or a hook that predates recorded
event queries (point HPROFILER_HOOKS_DIR at a fresh build).
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

NVCC = shutil.which("nvcc") or ("/usr/local/cuda/bin/nvcc" if Path("/usr/local/cuda/bin/nvcc").exists() else None)


class TestCudaVforkEvents(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.core.hookdir import hooks_dir
        if not NVCC or not (hooks_dir() / "libhprofiler_cuda.so").exists():
            raise unittest.SkipTest("needs nvcc and the built CUDA hook")
        cls.tmp = tempfile.mkdtemp()
        cls.exe = os.path.join(cls.tmp, "cuda_vfork_events")
        r = subprocess.run([NVCC, "-O1", "-Wno-deprecated-gpu-targets", "-cudart", "shared",
                            str(ROOT / "tests/fixtures/cuda_vfork_events.cu"), "-o", cls.exe, "-lcuda"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise unittest.SkipTest(f"cannot build the fixture: {r.stderr[:300]}")
        base = subprocess.run([cls.exe], capture_output=True, text=True, timeout=120)
        if base.returncode != 0 or "done" not in base.stdout:
            raise unittest.SkipTest(f"no usable GPU ({(base.stdout + base.stderr).strip()[:120]})")
        from src.core.runner import Runner
        rf, wf = os.pipe()
        saved = os.dup(1)
        os.dup2(wf, 1)
        try:
            cls.trace = Runner([cls.exe], backends=["cuda"]).run()
        finally:
            os.dup2(saved, 1)
            os.close(wf)
        cls.out = os.read(rf, 65536).decode()
        os.close(rf)
        m = re.search(r"pid=(\d+) tid=(\d+)", cls.out)
        if m is None:
            raise AssertionError(f"no pid line in the fixture's output: {cls.out[:300]}")
        cls.pid, cls.tid = int(m.group(1)), int(m.group(2))
        cls.spans = sorted(cls.trace.iter_spans(), key=lambda s: s.start_ns)
        if not any(s.tags.get("src") == "cupti" for s in cls.spans):
            raise unittest.SkipTest("no CUPTI records in the trace")
        if not any(s.name == "cudaEventQuery" for s in cls.spans):
            raise unittest.SkipTest("the CUDA hook predates recorded event queries -- point HPROFILER_HOOKS_DIR "
                                    "at a fresh build")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(getattr(cls, "tmp", ""), ignore_errors=True)

    def test_spans_carry_the_programs_pid(self):
        self.assertEqual({s.pid for s in self.spans}, {self.pid})
        host = [s for s in self.spans if s.tags.get("side") == "cpu"]
        self.assertTrue(host)
        self.assertEqual({s.tid for s in host}, {self.tid})

    def test_one_process_begin_marker_the_programs(self):
        marks = [i for i in self.trace.iter_instants() if i.name == "process_begin"]
        self.assertEqual([(m.pid, m.tags.get("hook")) for m in marks], [(self.pid, "cuda")])
        self.assertLessEqual(marks[0].timestamp_ns, self.spans[0].start_ns)

    def test_every_event_sync_record_has_its_host_call(self):
        host_corr = {(s.tags.get(k)) for s in self.spans if s.tags.get("side") == "cpu" for k in ("corr", "corr2")
                     if s.tags.get(k)}
        recs = [s for s in self.spans if s.tags.get("type") == "sync_wait" and s.tags.get("sync") == "event"]
        self.assertGreaterEqual(len(recs), 6, [s.tags for s in recs])  # sync, 3 queries, cuEventSynchronize, cuEventQuery
        self.assertEqual([s.tags["corr"] for s in recs if s.tags["corr"] not in host_corr], [])
        by_corr = {s.tags["corr"]: s for s in self.spans if s.tags.get("side") == "cpu" and s.tags.get("corr")}
        names = sorted({by_corr[s.tags["corr"]].name for s in recs})
        self.assertEqual(names, ["cuEventQuery", "cuEventSynchronize", "cudaEventQuery", "cudaEventSynchronize"])
        for s in recs:
            h = by_corr[s.tags["corr"]]
            self.assertEqual((h.category.value, h.tags.get("type"), h.tags.get("sync")), ("sync", "sync", "event"))
            self.assertIn("event", h.tags)
            self.assertNotIn("err", h.tags)
        self.assertEqual({by_corr[s.tags["corr"]].tags.get("src") for s in recs
                          if by_corr[s.tags["corr"]].name.startswith("cuEvent")}, {"cupti_cb"})

    def test_the_not_ready_query_is_not_recorded(self):
        self.assertIn("notready=1", self.out)
        queries = [s for s in self.spans if s.name == "cudaEventQuery"]
        self.assertEqual(len(queries), 3, [s.tags for s in queries])      # the three after the sync

    def test_window_from_the_marker_nothing_unconnected(self):
        from src.analysis import criticalpath as cp
        r = cp.analyze(self.trace, sensitivity=0, alternatives=0)
        self.assertEqual(r.window_start, "process_begin")
        self.assertEqual((r.unconnected_tail_records, r.unmatched_sync_records), (0, {}))
        self.assertEqual(r.total_path_ns, r.wall_ns)


if __name__ == "__main__":
    unittest.main()
