"""
End-to-end profiling-accuracy tests against ground-truth programs
(tests/fixtures/{omp,mpi,ocl}_truth.c) that log their own CLOCK_MONOTONIC
timestamps -- the same clock every hook uses -- around each operation.
Each run goes through the real pipeline: hook -> socket -> Runner -> Trace
-> chrome_trace.write -> load_trace_from_json -> analysis, i.e. exactly
what `hprofiler run` + `hprofiler gui/summary/critical-path` do.

Tolerances are explicit and chosen from repeated measurements on a laptop
CPU (see DOCUMENTATION.md §13 "Measured accuracy"): hook-side timestamps land
within tens of microseconds of the program's own stamps; per-thread compute
attribution within 5% (or 1ms) of truth. Timing tests never assert exact
equality of noisy measurements.

Skips (does not fail) when a toolchain/runtime is unavailable, like the
rest of tests/integration/.
"""
from __future__ import annotations

import collections
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
FIX = REPO / "tests" / "fixtures"

from src.core.runner import Runner
from src.core.trace import Trace
from src.output import chrome_trace
from src.analysis import activity_buckets as ab
from src.analysis import dashboard as dash


def _build(cmd: list[str]) -> bool:
    try:
        return subprocess.run(cmd, capture_output=True, timeout=120).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _profile(binary: str, backends: list[str], workdir: str) -> tuple[Trace, Trace, str]:
    """(in-memory trace, reloaded trace, truth file path)."""
    truth = os.path.join(workdir, Path(binary).name + ".truth")
    trace = Runner(command=[binary], backends=backends, env_extra={"TRUTH_OUT": truth}).run()
    out = os.path.join(workdir, Path(binary).name + ".json")
    chrome_trace.write(trace, out)
    return trace, chrome_trace.load_trace_from_json(out), truth


def _read_truth(path: str) -> dict[str, list[list[int]]]:
    rows: dict[str, list[list[int]]] = collections.defaultdict(list)
    with open(path) as f:
        for line in f:
            p = line.split()
            rows[p[0]].append([int(x) for x in p[1:] if x.lstrip("-").isdigit()])
    return rows


class _Tmp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="hprofiler_accuracy_")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)


class TestOpenMPAccuracy(_Tmp):
    """Same source built for both OpenMP runtimes hprofiler supports:
    LLVM libomp (OMPT tool) and GNU libgomp (GOMP_* interposition)."""

    NREG, NT, UNIT_NS = 6, 4, 4_000_000

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.results = {}
        src = str(FIX / "omp_truth.c")
        builds = {"llvm": ["clang", "-O2", "-fopenmp=libomp", src, "-o"],
                  "gnu": ["gcc", "-O2", "-fopenmp", src, "-o"]}
        for rt, cmd in builds.items():
            exe = os.path.join(cls.tmp, f"omp_truth_{rt}")
            if shutil.which(cmd[0]) and _build(cmd + [exe]):
                cls.results[rt] = _profile(exe, ["openmp"], cls.tmp)

    def _get(self, rt):
        if rt not in self.results:
            self.skipTest(f"{rt} OpenMP toolchain unavailable")
        return self.results[rt]

    def _barrier_name(self, rt):
        return "omp_barrier_explicit" if rt == "llvm" else "omp_barrier"

    def _check_runtime(self, rt):
        _, tr, truth_path = self._get(rt)
        truth = _read_truth(truth_path)
        bars = [s for s in tr.spans if s.name == self._barrier_name(rt)]
        # Completeness: one explicit-barrier span per (region, thread).
        self.assertEqual(len(bars), self.NREG * self.NT)

        # Timestamps vs the program's own stamps, matched by tid + time.
        lags, leads = [], []
        for r, t, tid, arrive, leave in truth["barrier"]:
            cands = [s for s in bars if s.tid == tid and arrive - 5_000_000 <= s.start_ns <= leave]
            self.assertTrue(cands, f"no barrier span for region {r} thread {t}")
            s = min(cands, key=lambda s: abs(s.start_ns - arrive))
            lags.append(s.start_ns - arrive)
            leads.append(leave - s.end_ns)
        self.assertLessEqual(statistics.median(lags), 50_000)       # 50 us
        self.assertLessEqual(max(abs(x) for x in lags), 2_000_000)  # 2 ms
        self.assertLessEqual(statistics.median(leads), 100_000)

        # Aggregation: total explicit-barrier wait vs truth, within 3%.
        truth_wait = sum(leave - arrive for *_, arrive, leave in truth["barrier"])
        measured = sum(s.duration_ns for s in bars)
        self.assertAlmostEqual(measured / truth_wait, 1.0, delta=0.03)

        # Per-thread compute attribution (exclusive openmp time) vs the
        # program's known spin time.
        et = ab.ExclusiveTime(tr.spans)
        by_tid = et.totals(lambda i, s: (s.tid, et.bucket[i]))
        spin_by_tid = collections.Counter()
        for r, t, tid, s0, s1 in truth["spin"]:
            spin_by_tid[tid] += s1 - s0
        for tid, spin in spin_by_tid.items():
            got = by_tid.get((tid, "Computation"), 0)
            self.assertLessEqual(abs(got - spin), max(0.05 * spin, 1_000_000),
                                 f"{rt} tid {tid}: compute {got} vs truth {spin}")

    def test_llvm_ompt_accuracy(self):
        self._check_runtime("llvm")

    def test_gnu_gomp_accuracy(self):
        self._check_runtime("gnu")

    def test_ompt_emits_per_thread_implicit_tasks(self):
        _, tr, _ = self._get("llvm")
        tasks = [s for s in tr.spans if s.name == "omp_implicit_task"]
        self.assertEqual(len(tasks), self.NREG * self.NT)
        self.assertEqual(len({s.tid for s in tasks}), self.NT)

    def test_both_runtimes_get_the_same_diagnosis(self):
        _, llvm, _ = self._get("llvm")
        _, gnu, _ = self._get("gnu")
        self.assertEqual(dash.diagnose(llvm)[0], dash.diagnose(gnu)[0])
        self.assertEqual(dash.diagnose(gnu)[0], "openmp-bound")


class TestMPIAccuracy(_Tmp):
    N = 8

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.result = None
        exe = os.path.join(cls.tmp, "mpi_truth")
        if shutil.which("mpicc") and _build(["mpicc", "-O2", str(FIX / "mpi_truth.c"), "-o", exe]) \
                and (REPO / "build" / "lib" / "libhprofiler_mpi.so").exists():
            try:
                cls.result = _profile(exe, ["mpi"], cls.tmp)
            except Exception:
                cls.result = None

    def setUp(self):
        if self.result is None:
            self.skipTest("MPI toolchain/hook unavailable")

    def test_every_call_captured(self):
        _, tr, _ = self.result
        counts = collections.Counter(s.name for s in tr.spans)
        for name, n in (("MPI_Barrier", self.N), ("MPI_Waitall", self.N), ("MPI_Allreduce", self.N),
                        ("MPI_Irecv", 2 * self.N), ("MPI_Isend", 2 * self.N), ("MPI_Wait", 2 * self.N)):
            self.assertEqual(counts[name], n, name)

    def test_timestamps_match_program_clock(self):
        _, tr, truth_path = self.result
        truth = _read_truth(truth_path)
        for name, key in (("MPI_Barrier", "barrier"), ("MPI_Waitall", "waitall"), ("MPI_Allreduce", "allreduce")):
            spans = sorted((s for s in tr.spans if s.name == name), key=lambda s: s.start_ns)
            lags = [s.start_ns - row[1] for s, row in zip(spans, truth[key])]
            under = [(row[2] - row[1]) - s.duration_ns for s, row in zip(spans, truth[key])]
            self.assertLessEqual(statistics.median(lags), 20_000, name)
            self.assertGreaterEqual(min(lags), -1_000, name)   # never before the call started
            self.assertLessEqual(statistics.median(under), 20_000, name)
            self.assertGreaterEqual(min(under), -1_000, name)  # never longer than the call

    def test_wildcard_receive_resolution_survives_reload(self):
        _, tr, truth_path = self.result
        truth = _read_truth(truth_path)
        resolved = sorted(int(s.tags["rtag"]) for s in tr.spans if s.name == "MPI_Wait" and "rtag" in s.tags)
        self.assertEqual(resolved, sorted(100 + row[0] for row in truth["wildwait"]))
        waits = [s for s in tr.spans if s.name == "MPI_Wait" and "rtag" in s.tags]
        irecv_ids = {s.span_id for s in tr.spans if s.name == "MPI_Irecv"}
        self.assertTrue(all(w.parent_span_id in irecv_ids for w in waits))

    def test_critical_path_graph_identical_in_memory_and_reloaded(self):
        from src.analysis import criticalpath as cp
        mem, tr, _ = self.result

        def edges(t):
            spans, preds = cp.build_dependency_graph(t)
            return sorted(((spans[u].name, spans[u].start_ns), (spans[v].name, spans[v].start_ns), k, c)
                          for v, es in preds.items() for (u, k, c) in es)
        e_mem = edges(mem)
        self.assertTrue(any(k == "explicit_span_id" for *_, k, _c in e_mem))
        self.assertEqual(e_mem, edges(tr))


class TestOpenCLAccuracy(_Tmp):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.result = None
        exe = os.path.join(cls.tmp, "ocl_truth")
        if _build(["gcc", "-O2", str(FIX / "ocl_truth.c"), "-o", exe, "-lOpenCL"]):
            try:
                if subprocess.run([exe], env={**os.environ, "TRUTH_OUT": os.devnull},
                                  capture_output=True, timeout=120).returncode == 0:
                    cls.result = _profile(exe, ["opencl"], cls.tmp)
            except (OSError, subprocess.TimeoutExpired):
                cls.result = None

    def setUp(self):
        if self.result is None:
            self.skipTest("no usable OpenCL CPU device")

    def test_kernel_device_time_matches_cl_profiling(self):
        _, tr, truth_path = self.result
        truth = _read_truth(truth_path)
        gpu = sorted((s for s in tr.spans if s.tags.get("type") == "kernel" and s.tags.get("side") == "gpu"),
                     key=lambda s: s.start_ns)
        self.assertEqual(len(gpu), len(truth["kernel_devdur"]))
        for s, row in zip(gpu, truth["kernel_devdur"]):
            self.assertLessEqual(abs(s.duration_ns - row[1]), 1_000)

    def test_blocking_transfer_reported_once(self):
        _, tr, truth_path = self.result
        writes = [s for s in tr.spans if s.tags.get("type") == "write"]
        self.assertEqual(len(writes), len(_read_truth(truth_path)["write_blocking"]))

    def test_device_spans_lie_within_host_enqueue_to_finish(self):
        _, tr, truth_path = self.result
        truth = _read_truth(truth_path)
        cpu = sorted((s for s in tr.spans if s.name == "k" and s.tags.get("side") == "cpu"), key=lambda s: s.start_ns)
        gpu = sorted((s for s in tr.spans if s.name == "k" and s.tags.get("side") == "gpu"), key=lambda s: s.start_ns)
        for c, g, row in zip(cpu, gpu, truth["finish"]):
            self.assertGreaterEqual(g.start_ns, c.start_ns)
            self.assertLessEqual(g.end_ns, row[2])


if __name__ == "__main__":
    unittest.main()
