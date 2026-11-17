"""
MPI_Init / MPI_Finalize as spans and the process-begin markers on 4 real
ranks (tests/fixtures/mpi_lifecycle.c: rank r works 20 ms x r before
MPI_Init, 20 ms x (3 - r) before MPI_Finalize; rank 0 works 30 ms before
its send to rank 1):

  * every rank has one MPI_Init (MPI_Init_thread with thread=1, with
    required=/provided=) and one MPI_Finalize span (category mpi,
    type=init|finalize, rank=, csize=4) inside the times the program saw
    around the calls;
  * one process-begin marker per rank, before the rank's work before
    MPI_Init, none from the launcher's processes (they record nothing),
    and one only with two hooks preloaded (the MPI and NCCL hooks: the
    first to load claims it, hp_transport.h);
  * the critical path from process begin: wall time = observed path =
    the sum of the path's spans' time; with the MPI_Init rendezvous the
    last rank's 60 ms before MPI_Init are on it (absent with
    --window first-event).

Needs Intel MPI or another launcher that starts 4 real ranks (see
test_mpi_multirank.py) and an MPI hook built from this source
(HPROFILER_HOOKS_DIR); skips otherwise.
"""
import os
import re
import shutil
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from tests.integration import test_mpi_multirank as mr     # the module, not its TestCases

MS = 1_000_000
US = 1_000          # the fixture prints its times to the microsecond


def _times(out: str) -> dict[int, dict]:
    """rank -> the fixture's printed times (CLOCK_MONOTONIC, converted to ns)."""
    rows = {}
    for ln in out.splitlines():
        if ln.startswith("rank="):
            kv = dict(re.findall(r"(\w+)=([\d.]+)", ln))
            rows[int(kv["rank"])] = {k: (round(float(v) * MS) if k not in ("rank", "pre") else int(v))
                                     for k, v in kv.items()}
    return rows


def _run(cls, args, backends):
    from src.core.runner import Runner
    out = os.path.join(cls.tmp, "prof_%d.out" % len(os.listdir(cls.tmp)))
    with mock.patch.dict(os.environ, cls.env, clear=True), open(out, "w") as f:
        saved = os.dup(1)
        os.dup2(f.fileno(), 1)
        try:
            trace = Runner([cls.mpirun, "-np", "4", cls.exe] + args, backends=backends).run()
        finally:
            os.dup2(saved, 1)
            os.close(saved)
    return trace, Path(out).read_text()


class TestMPILifecycle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mr._build_and_profile(cls, "mpi_lifecycle")
        if not any(s.name == "MPI_Init" for s in cls.trace.iter_spans(categories=("mpi",))):
            raise unittest.SkipTest("the MPI hook predates MPI_Init spans -- point HPROFILER_HOOKS_DIR at a "
                                    "fresh build")
        cls.thread_trace, cls.thread_out = _run(cls, ["thread=1"], ["mpi", "nccl"])

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(getattr(cls, "tmp", ""), ignore_errors=True)

    def lifecycle(self, trace, out, init_name):
        times = _times(out)
        self.assertEqual(sorted(times), [0, 1, 2, 3], out[-500:])
        spans = list(trace.iter_spans())
        rank_of = {s.pid: int(s.tags["rank"]) for s in spans if s.category.value == "mpi" and "rank" in s.tags}
        self.assertEqual(sorted(rank_of.values()), [0, 1, 2, 3])
        for pid, rank in rank_of.items():
            t = times[rank]
            mine = [s for s in spans if s.pid == pid]
            inits = [s for s in mine if s.name.startswith("MPI_Init")]
            fins = [s for s in mine if s.name == "MPI_Finalize"]
            self.assertEqual(([s.name for s in inits], len(fins)), ([init_name], 1), rank)
            init, fin = inits[0], fins[0]
            for s, typ in ((init, "init"), (fin, "finalize")):
                self.assertEqual(s.category.value, "mpi")
                self.assertEqual((s.tags.get("type"), s.tags.get("rank"), s.tags.get("csize")),
                                 (typ, str(rank), "4"), s.tags)
            # inside what the program saw around the calls (the hook's own
            # bookkeeping before PMPI_Finalize lies before the span)
            self.assertTrue(t["init_in"] - US <= init.start_ns and init.end_ns <= t["init_out"] + US, (rank, t, init))
            self.assertTrue(t["fin_in"] - US <= fin.start_ns and fin.end_ns <= t["fin_out"] + US, (rank, t, fin))
            # MPI_Init is the rank's first span, MPI_Finalize its last
            self.assertEqual(min(mine, key=lambda s: s.start_ns).name, init_name)
            self.assertEqual(max(mine, key=lambda s: s.end_ns).name, "MPI_Finalize")
        return rank_of, times

    def markers(self, trace, rank_of, times):
        marks = [i for i in trace.iter_instants() if i.name == "process_begin"]
        self.assertEqual(sorted(m.pid for m in marks), sorted(rank_of), "one per rank, none from the launcher")
        for m in marks:
            t = times[rank_of[m.pid]]
            self.assertEqual(m.tags.get("type"), "process_begin")
            # before the rank's 20 ms x rank of work before MPI_Init
            self.assertLessEqual(m.timestamp_ns, t["init_in"] - 20 * MS * t["pre"], (m.pid, t))
        return marks

    def test_mpi_init_spans_and_markers(self):
        rank_of, times = self.lifecycle(self.trace, self.prof_out, "MPI_Init")
        self.assertEqual({m.tags.get("hook") for m in self.markers(self.trace, rank_of, times)}, {"mpi"})

    def test_mpi_init_thread_with_two_hooks_one_marker(self):
        rank_of, times = self.lifecycle(self.thread_trace, self.thread_out, "MPI_Init_thread")
        self.markers(self.thread_trace, rank_of, times)
        for s in self.thread_trace.iter_spans(categories=("mpi",)):
            if s.name == "MPI_Init_thread":
                # MPI_THREAD_* as 0..3 (single, funneled, serialized, multiple)
                self.assertEqual(s.tags.get("required"), "1", s.tags)
                self.assertIn(s.tags.get("provided"), ("1", "2", "3"), s.tags)

    def test_critical_path_window(self):
        from src.analysis import cp_model as cm
        from src.analysis import criticalpath as cp
        pb = cp.analyze(self.trace, sensitivity=0, alternatives=0)
        fe = cp.analyze(self.trace, sensitivity=0, alternatives=0, window="first_event")
        self.assertEqual((pb.window_start, fe.window_start), ("process_begin", "first_event"))
        self.assertGreater(pb.window_lead_ns, 0)
        self.assertEqual(pb.total_path_ns, pb.wall_ns)
        for r in (pb, fe):
            self.assertEqual(sum(c + g for c, g in zip(r.path_credited_ns, r.path_gap_ns)), r.total_path_ns)
        self.assertEqual(pb.wall_ns - fe.wall_ns, pb.window_lead_ns)
        g = pb.model.graph
        inits = sum(1 for e in range(len(g)) if g.enabled[e] and cm.PROVS[g.prov[e]] == "lifecycle_sync"
                    and g.name(g.dst[e] >> 1).startswith("MPI_Init"))
        rank_of = {s.pid: int(s.tags["rank"]) for s in pb.spans if "rank" in s.tags}
        first = pb.spans[pb.path_span_indices[0]]
        if first.name == "MPI_Init" and rank_of.get(first.pid) == 3:
            # the library held every rank in MPI_Init until the last (rank
            # 3, 60 ms late) arrived: its time before MPI_Init is on the path
            self.assertGreaterEqual(pb.path_gap_ns[0], 55 * MS)
            self.assertGreaterEqual(pb.total_path_ns - fe.total_path_ns, 55 * MS)
        else:
            self.skipTest(f"this MPI's MPI_Init did not hold the ranks (the path starts at {first.name}, "
                          f"rank {rank_of.get(first.pid)}; {inits} MPI_Init rendezvous edges)")


if __name__ == "__main__":
    unittest.main()
