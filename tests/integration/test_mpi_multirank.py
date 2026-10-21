"""
The MPI hook with 4 real ranks.

  * tests/fixtures/mpi_status.c (TestMPIMultiRank): what the application
    observes must equal the unprofiled run (statuses from Waitall/Waitsome/
    Testall/Testsome, cancellation), and the dependency graph must pair
    messages on renumbered sub-communicators, on concurrent duplicated
    communicators and through MPI_Sendrecv, with no unmatched receive;
  * tests/fixtures/mpi_proto.c (TestMPIProto): wildcard receives resolved by
    MPI_Waitany/MPI_Waitsome (with completed slots in the array), a Test
    polling loop, a cancelled receive and two MPI_Comm_split halves -- the
    program's output against its ground truth, unprofiled and profiled, and
    the trace: resolved matches, Test instants, the cancellation, commid=
    agreement across processes, and the dependency edges.

Needs an MPI whose launcher really starts 4 ranks. Ubuntu's MPICH package
starts singletons (see DOCUMENTATION.md, "Verification status"), so Intel
MPI (oneAPI, MPICH ABI) is used when installed: HPROFILER_TEST_MPIVARS
names the vars.sh to source (default /opt/intel/oneapi/mpi/latest/env/
vars.sh); without it the mpicc/mpirun on PATH are tried. Skips when no
launcher gives 4 ranks.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


def _mpi_env() -> dict:
    vars_sh = os.environ.get("HPROFILER_TEST_MPIVARS", "/opt/intel/oneapi/mpi/latest/env/vars.sh")
    env = dict(os.environ)
    if Path(vars_sh).exists():
        out = subprocess.run(["bash", "-c", f'source "{vars_sh}" >/dev/null 2>&1; env -0'],
                             capture_output=True, text=True, check=True).stdout
        env = dict(kv.split("=", 1) for kv in out.split("\0") if "=" in kv)
        env.setdefault("I_MPI_FABRICS", "shm")
    return env


def _lines(text: str) -> list[str]:
    return sorted(ln for ln in text.splitlines() if ln[:2] in ("A ", "B ", "C ", "D ", "E ", "F "))


def _build_and_profile(cls, fixture: str, strict: bool = False) -> None:
    """Build tests/fixtures/<fixture>.c with the MPI environment's mpicc and
    run it on 4 ranks, unprofiled (cls.base_out) and under the MPI hook
    (cls.prof_out, cls.trace), then build its dependency graph (cls.spans,
    cls.preds, cls.p2p). Skips without the hook, mpicc/mpirun or a launcher
    that starts 4 real ranks (the fixture says so itself: "requires exactly
    4 ranks"); with strict=True any other unprofiled failure is a failure,
    not a skip."""
    if not (ROOT / "build/lib/libhprofiler_mpi.so").exists():
        raise unittest.SkipTest("MPI hook not built")
    cls.env = _mpi_env()
    mpicc = shutil.which("mpicc", path=cls.env.get("PATH"))
    cls.mpirun = shutil.which("mpirun", path=cls.env.get("PATH"))
    if not mpicc or not cls.mpirun:
        raise unittest.SkipTest("no mpicc/mpirun")
    cls.tmp = tempfile.mkdtemp()
    cls.exe = os.path.join(cls.tmp, fixture)
    r = subprocess.run([mpicc, "-O1", str(ROOT / f"tests/fixtures/{fixture}.c"), "-o", cls.exe],
                       env=cls.env, capture_output=True, text=True)
    if r.returncode != 0:
        raise unittest.SkipTest(f"cannot build the fixture: {r.stderr[:200]}")
    base = subprocess.run([cls.mpirun, "-np", "4", cls.exe], env=cls.env, capture_output=True,
                          text=True, timeout=120)
    said = next((ln.strip() for ln in (base.stdout + base.stderr).splitlines()
                 if "requires exactly 4 ranks" in ln), None)
    if said:
        raise unittest.SkipTest(f"the MPI launcher does not start 4 real ranks ({said})")
    if base.returncode != 0:
        if strict:
            raise AssertionError(f"unprofiled {fixture} failed ({base.returncode}): {base.stderr[-1500:]}")
        raise unittest.SkipTest("the MPI launcher does not start 4 real ranks")
    cls.base_out = base.stdout
    from src.core.runner import Runner
    out = os.path.join(cls.tmp, "prof.out")
    with mock.patch.dict(os.environ, cls.env, clear=True), open(out, "w") as f:
        saved = os.dup(1)
        os.dup2(f.fileno(), 1)
        try:
            runner = Runner([cls.mpirun, "-np", "4", cls.exe], backends=["mpi"])
            cls.trace = runner.run()
        finally:
            os.dup2(saved, 1)
            os.close(saved)
    cls.prof_status = runner.exit_status
    cls.prof_out = Path(out).read_text()
    from src.analysis import criticalpath as cp
    cls.cp = cp
    cls.spans, cls.preds = cp.build_dependency_graph(cls.trace)
    cls.p2p = [(cls.spans[u], cls.spans[v], c) for v, pl in cls.preds.items()
               for u, k, c in pl if k == "p2p"]


class TestMPIMultiRank(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _build_and_profile(cls, "mpi_status")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(getattr(cls, "tmp", ""), ignore_errors=True)

    # ── what the application sees ──────────────────────────────────────
    def test_application_output_equals_the_unprofiled_run(self):
        prof = [ln for ln in _lines(self.prof_out) if not ln.startswith("A ")]
        base = [ln for ln in _lines(self.base_out) if not ln.startswith("A ")]
        self.assertEqual(prof, base)

    def test_statuses_reach_the_application(self):
        a = [ln for ln in _lines(self.prof_out) if ln.startswith("A ")]
        self.assertEqual(len(a), 12, self.prof_out)
        for ln in a:
            f = dict(kv.split("=") for kv in ln.split()[2:] if "=" in kv)
            src, tag, val = int(f["source"]), int(f["tag"]), int(f["value"])
            self.assertIn(src, (1, 2, 3), ln)
            self.assertEqual(val, src, ln)
            self.assertEqual(tag % 100, src, ln)

    def test_cancelled_waits_report_no_match(self):
        self.assertIn("B cancelled=1", self.prof_out)
        waits = [s for s in self.trace.iter_spans(categories=("mpi",))
                 if s.name == "MPI_Wait" and s.tags.get("cancelled") == "1"]
        self.assertEqual(len(waits), 2)
        for w in waits:
            self.assertNotIn("rpeer", w.tags)
            self.assertFalse([1 for _s, d, _c in self.p2p if d is w])

    # ── dependency graph ───────────────────────────────────────────────
    def _edges_into(self, name, rank, **tags):
        return [(s, c) for s, d, c in self.p2p if d.name == name and d.tags.get("rank") == str(rank)
                and all(d.tags.get(k) == str(v) for k, v in tags.items())]

    def test_renumbered_sub_communicator(self):
        # half {0,1}: world 1 is local 0 and receives from world 0; half {2,3}:
        # world 3 is local 0 and receives from world 2
        for receiver, sender in ((1, 0), (3, 2)):
            e = self._edges_into("MPI_Recv", receiver, tag=7)
            self.assertEqual([s.tags.get("rank") for s, _c in e], [str(sender)])

    def test_concurrent_duplicated_communicators(self):
        e = self._edges_into("MPI_Recv", 0, tag=5)
        self.assertEqual(len(e), 2)
        for s, _c in e:
            recv = next(d for ss, d, _cc in self.p2p if ss is s)
            self.assertEqual(s.tags.get("commid"), recv.tags.get("commid"))

    def test_sendrecv_ring(self):
        for name, tag, step in (("MPI_Sendrecv", 3, -1), ("MPI_Sendrecv_replace", 4, 1)):
            for r in range(4):
                e = self._edges_into(name, r, tag=tag)
                self.assertEqual([s.tags.get("rank") for s, _c in e], [str((r + step) % 4)], (name, r))
                self.assertEqual(e[0][0].name, name + ".send")

    def test_rsend_issend_probe_ibarrier_and_v_collectives_recorded(self):
        names = Counter(s.name for s in self.trace.iter_spans(categories=("mpi",)))
        for n, k in (("MPI_Rsend", 1), ("MPI_Issend", 1), ("MPI_Probe", 1), ("MPI_Ibarrier", 4),
                     ("MPI_Allgatherv", 4), ("MPI_Alltoallv", 4), ("MPI_Gatherv", 4), ("MPI_Scatterv", 4),
                     ("MPI_Reduce_scatter", 4), ("MPI_Reduce_scatter_block", 4)):
            self.assertEqual(names[n], k, n)
        self.assertEqual(len(self._edges_into("MPI_Recv", 3, tag=8)), 1)       # from Issend
        self.assertEqual(len(self._edges_into("MPI_Wait", 0)), 1)              # Rsend -> Irecv's Wait
        # every Ibarrier request is completed by a Wait that links back to it
        ib = {s.span_id for s in self.trace.iter_spans(categories=("mpi",)) if s.name == "MPI_Ibarrier"}
        self.assertEqual(len(ib), 4)
        waited = {p for s in self.trace.iter_spans(categories=("mpi",)) if s.name == "MPI_Wait"
                  for p in (s.parent_span_id or "").split(";")}
        self.assertTrue(ib <= waited)

    def test_no_unmatched_receives_and_counted_gaps(self):
        diag = self.cp.mpi_dependency_diagnostics(self.trace)
        self.assertEqual(diag.get("key"), "world+commid")
        self.assertNotIn("unmatched_by_comm", diag)
        self.assertNotIn("orphan_completions", diag)
        self.assertEqual(diag.get("test_only_completions"), 6)   # Testall + Testsome rounds
        self.assertEqual(diag.get("cancelled"), 2)
        self.assertEqual(self.trace.metadata.dependency_diagnostics, diag)


_PHASE_A = re.compile(r"phase_a (waitany|waitsome) \S+=(\d+) source=(-?\d+) tag=(-?\d+) value=(-?\d+)")
_PHASE_B = re.compile(r"phase_b test polls=(\d+) flag=(\d+) source=(-?\d+) tag=(-?\d+)")
_PHASE_D = re.compile(r"phase_d rank=(\d+) color=(\d+) sum=(\S+)")


class TestMPIProto(unittest.TestCase):
    """tests/fixtures/mpi_proto.c on 4 ranks. Ground truth: ranks 1-3 each send
    their rank to rank 0 with tag 100+rank (phase A: 3 wildcard receives,
    one completed by MPI_Waitany, the rest by MPI_Waitsome); rank 1 sends
    999 with tag 200 after 300 ms (phase B: rank 0 polls MPI_Test); rank 0
    cancels a never-matched wildcard receive (phase C); each half of an
    MPI_Comm_split by rank % 2 sums its ranks (phase D: 0+2, 1+3)."""

    @classmethod
    def setUpClass(cls):
        _build_and_profile(cls, "mpi_proto", strict=True)
        cls.rank0 = [s for s in cls.trace.iter_spans(categories=("mpi",)) if s.tags.get("rank") == "0"]
        cls.inst0 = [i for i in cls.trace.iter_instants() if i.category.value == "mpi"
                     and i.tags.get("rank") == "0"]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(getattr(cls, "tmp", ""), ignore_errors=True)

    # ── what the application sees (unprofiled and profiled alike) ──────────
    def _check_output(self, out: str) -> int:
        """Asserts the ground truth on one run's stdout; returns phase B's polls."""
        a = _PHASE_A.findall(out)
        self.assertEqual([k for k, *_ in a].count("waitany"), 1, out)
        self.assertEqual([k for k, *_ in a].count("waitsome"), 2, out)
        self.assertEqual(sorted((int(s), int(t), int(v)) for _k, _i, s, t, v in a),
                         [(1, 101, 1), (2, 102, 2), (3, 103, 3)], out)
        b = _PHASE_B.findall(out)
        self.assertEqual(len(b), 1, out)
        polls, flag, src, tag = (int(x) for x in b[0])
        self.assertEqual((flag, src, tag), (1, 1, 200), out)
        self.assertLess(polls, 200, out)
        self.assertRegex(out, r"phase_b test .* value=999\b")
        self.assertIn("phase_c cancel done", out)
        self.assertEqual(sorted((int(r), int(c), s) for r, c, s in _PHASE_D.findall(out)),
                         [(0, 0, "2.000000"), (1, 1, "4.000000"), (2, 0, "2.000000"), (3, 1, "4.000000")], out)
        return polls

    def test_unprofiled_output(self):
        self._check_output(self.base_out)

    def test_profiled_output(self):
        self._check_output(self.prof_out)
        self.assertEqual(self.prof_status, 0)

    # ── the trace ───────────────────────────────────────────────────────
    def test_every_rank_traced(self):
        ranks = {s.tags.get("rank") for s in self.trace.iter_spans(categories=("mpi",))}
        self.assertEqual(ranks, {"0", "1", "2", "3"})

    def test_wildcard_receives_resolved(self):
        waitany = [s for s in self.rank0 if s.name == "MPI_Waitany"]
        self.assertEqual(len(waitany), 1)
        resolved = [(int(waitany[0].tags["rpeer"]), int(waitany[0].tags["rtag"]))]
        waitsome = [s for s in self.rank0 if s.name == "MPI_Waitsome"]
        self.assertGreaterEqual(len(waitsome), 1)
        for s in waitsome:
            for entry in s.tags.get("rmatches", "").split(";"):
                if entry:
                    _req, src, tag = entry.split("/")
                    resolved.append((int(src), int(tag)))
        self.assertEqual(sorted(resolved), [(1, 101), (2, 102), (3, 103)])

    def test_test_polling(self):
        polls = self._check_output(self.prof_out)
        tests = [i for i in self.inst0 if i.name == "MPI_Test"]
        self.assertEqual(len(tests), polls + 1)
        done = [i for i in tests if i.tags.get("flag") == "1"]
        self.assertEqual(len(done), 1)
        self.assertEqual((done[0].tags.get("rpeer"), done[0].tags.get("rtag")), ("1", "200"))
        self.assertTrue(all(i.tags.get("flag") == "0" for i in tests if i is not done[0]))

    def _cancelled_wait(self):
        cancels = [i for i in self.trace.iter_instants() if i.name == "MPI_Cancel"]
        self.assertEqual(len(cancels), 1)
        psid = cancels[0].tags.get("psid")
        self.assertTrue(psid)
        waits = [s for s in self.rank0 if s.name == "MPI_Wait" and psid in (s.parent_span_id or "").split(";")]
        self.assertEqual(len(waits), 1)
        return waits[0]

    def test_cancelled_receive(self):
        wait = self._cancelled_wait()
        self.assertNotIn("rpeer", wait.tags)
        self.assertNotIn("rtag", wait.tags)

    def test_communicator_ids_agree_across_processes(self):
        by_color: dict[int, set] = {0: set(), 1: set()}
        ranks: dict[int, set] = {0: set(), 1: set()}
        for s in self.trace.iter_spans(categories=("mpi",)):
            if s.name == "MPI_Allreduce":
                color = int(s.tags["rank"]) % 2
                by_color[color].add(s.tags.get("commid"))
                ranks[color].add(s.tags["rank"])
        self.assertEqual(ranks, {0: {"0", "2"}, 1: {"1", "3"}})
        self.assertEqual([len(by_color[0]), len(by_color[1])], [1, 1], by_color)
        self.assertNotEqual(by_color[0], by_color[1])
        self.assertNotIn(None, by_color[0] | by_color[1])

    # ── the dependency graph ────────────────────────────────────────────
    def test_phase_a_edges(self):
        into = [(s, d, c) for s, d, c in self.p2p
                if d.tags.get("rank") == "0" and d.name in ("MPI_Waitany", "MPI_Waitsome")]
        self.assertEqual(len(into), 3, [(s.name, s.tags, d.name) for s, d, _c in into])
        self.assertEqual(sorted((s.name, s.tags.get("rank"), s.tags.get("tag")) for s, _d, _c in into),
                         [("MPI_Send", "1", "101"), ("MPI_Send", "2", "102"), ("MPI_Send", "3", "103")])
        self.assertTrue(all(c in ("high", "certain") for _s, _d, c in into), [c for *_x, c in into])

    def test_no_edge_into_the_cancelled_wait(self):
        wait = self._cancelled_wait()
        self.assertFalse([1 for _s, d, _c in self.p2p if d is wait])

    def test_test_only_completion_has_no_edge_by_design(self):
        # DOCUMENTATION.md: a request completed only by MPI_Test* gets no
        # cross-rank edge; it is counted instead
        late = [s for s in self.trace.iter_spans(categories=("mpi",))
                if s.name == "MPI_Send" and s.tags.get("tag") == "200"]
        self.assertEqual(len(late), 1)
        self.assertFalse([1 for s, _d, _c in self.p2p if s is late[0]])
        diag = self.cp.mpi_dependency_diagnostics(self.trace)
        self.assertEqual(diag.get("test_only_completions"), 1)


if __name__ == "__main__":
    unittest.main()
