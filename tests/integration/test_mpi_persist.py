"""
Persistent requests, rooted and non-blocking collectives through the MPI
hook with 4 real ranks (tests/fixtures/mpi_persist.c; the launcher and
skips as in test_mpi_multirank.py):

  * every MPI_Start -- alone or one per request of an MPI_Startall -- is a
    post: ptype=send|recv with peer=/wpeer=/tag=/commid= and a sid= its
    completion names (psid=);
  * rooted collectives name their root as a communicator rank (root=) and a
    world rank (wroot=), also on a split, renumbered communicator;
  * non-blocking rooted collectives too, with the sid= their MPI_Wait names.

Skips with a hook library that predates these tags.
"""
import shutil
import sys
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from tests.integration.test_mpi_multirank import _build_and_profile, _lines  # noqa: E402


class TestMPIPersistentAndRooted(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _build_and_profile(cls, "mpi_persist")
        cls.mpi = [s for s in cls.trace.iter_spans(categories=("mpi",))]
        if not any(s.name == "MPI_Start" and "ptype" in s.tags for s in cls.mpi):
            raise unittest.SkipTest("the hook library predates MPI_Start's ptype= -- rebuild build/ "
                                    "or point HPROFILER_HOOKS_DIR at a fresh build")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(getattr(cls, "tmp", ""), ignore_errors=True)

    def test_application_output_equals_the_unprofiled_run(self):
        self.assertEqual(_lines(self.prof_out), _lines(self.base_out))
        self.assertEqual(len([ln for ln in _lines(self.prof_out) if ln.startswith("A ")]), 20)

    def test_every_start_is_a_post(self):
        starts = [s for s in self.mpi if s.name == "MPI_Start"]
        # 4 ranks x (3 Startall rounds x 2 requests + 2 rounds x 2 Starts)
        self.assertEqual(len(starts), 4 * (3 * 2 + 2 * 2))
        self.assertEqual(sum(1 for s in starts if s.tags.get("startall") == "1"), 4 * 3 * 2)
        for s in starts:
            rank = int(s.tags["rank"])
            want = (rank + 1) % 4 if s.tags["ptype"] == "send" else (rank + 3) % 4
            self.assertEqual((int(s.tags["peer"]), int(s.tags["wpeer"]), s.tags["tag"]), (want, want, "7"), s.tags)
            self.assertTrue(s.span_id)
        self.assertEqual(Counter(s.tags["ptype"] for s in starts), Counter(send=20, recv=20))

    def test_completions_name_the_starts(self):
        started = {s.span_id for s in self.mpi if s.name == "MPI_Start"}
        completed = set()
        for s in self.mpi:
            if s.name in ("MPI_Wait", "MPI_Waitall") and s.parent_span_id:
                completed.update(x for x in s.parent_span_id.split(";") if x)
        self.assertTrue(started <= completed, started - completed)

    def test_rooted_collectives_name_the_world_root(self):
        def roots(name):
            return {(s.tags["rank"], s.tags.get("root"), s.tags.get("wroot")) for s in self.mpi if s.name == name}
        self.assertEqual(roots("MPI_Bcast"), {(str(r), "2", "2") for r in range(4)})
        self.assertEqual(roots("MPI_Gather"), {(str(r), "3", "3") for r in range(4)})
        # split halves {0, 2} and {1, 3}: local rank 1 is world rank 2 / 3
        self.assertEqual(roots("MPI_Reduce"), {("0", "1", "2"), ("2", "1", "2"), ("1", "1", "3"), ("3", "1", "3")})
        self.assertEqual(roots("MPI_Ibcast"), {(str(r), "1", "1") for r in range(4)})

    def test_the_graph_pairs_persistent_messages(self):
        starts = [(self.spans[u], self.spans[v]) for u, v, _c in
                  [(u, v, c) for v, pl in self.preds.items() for u, k, c in pl if k == "p2p"]]
        persistent = [(a, b) for a, b in starts if a.name == "MPI_Start"]
        self.assertEqual(len(persistent), 20)              # 4 ranks x 5 rounds
        for a, b in persistent:
            self.assertEqual(a.tags["ptype"], "send")
            self.assertIn(b.name, ("MPI_Wait", "MPI_Waitall"))
            self.assertEqual(int(b.tags["rank"]), (int(a.tags["rank"]) + 1) % 4)

    def test_the_graph_follows_root_data_flow(self):
        arr = [(self.spans[u], self.spans[v]) for v, pl in self.preds.items() for u, k, _c in pl if k == "arrival"]
        bcast = [(a.tags["rank"], b.tags["rank"]) for a, b in arr if a.name == b.name == "MPI_Bcast"]
        self.assertEqual(sorted(bcast), [("2", "0"), ("2", "1"), ("2", "3")])
        gather = [(a.tags["rank"], b.tags["rank"]) for a, b in arr if a.name == b.name == "MPI_Gather"]
        self.assertEqual([r for _a, r in gather], ["3"] * len(gather))
        self.assertLessEqual(len(gather), 1)
        ibcast = [(a.tags["rank"], b.name) for a, b in arr if a.name == "MPI_Ibcast"]
        self.assertEqual(sorted(ibcast), [("1", "MPI_Wait")] * 4)

    def test_nonblocking_collectives_are_completed_by_their_wait(self):
        posted = {s.span_id: s for s in self.mpi if s.name in ("MPI_Ibcast", "MPI_Iallreduce")}
        self.assertEqual(len(posted), 8)
        waits = [s for s in self.mpi if s.name == "MPI_Wait" and s.parent_span_id in posted]
        self.assertEqual(len(waits), 8)


if __name__ == "__main__":
    unittest.main()
