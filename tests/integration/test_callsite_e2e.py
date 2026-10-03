"""
End-to-end call-site attribution: hook -> socket -> saved trace store ->
disassembly attached by `hprofiler run --disasm`, for:

  * MPI point-to-point and request operations (Send/Ssend/Recv/Isend/Irecv/
    Wait/Waitall/Waitany/Waitsome and the Test*/Cancel instants), not just
    collectives;
  * GNU libgomp omp_critical_hold (gomp_hook.c);
  * LLVM libomp through the OMPT tool, whose resolver emits symfile=;

each launched through a wrapper (`env <binary>`), so command[0] is not the
profiled program and disassembly only works if symfile= names the real ELF.
Also checks that a request list longer than the hook's bounded list is
reported exactly (listed + psid_omitted == completed requests) and that the
transport's capture health shows a complete, lossless run.

Runs the real CLI against the hook libraries in build/lib (rebuilt first
when a CMake build directory exists). Skips when a toolchain is missing.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
FIX = REPO / "tests" / "fixtures"
LIB = REPO / "build" / "lib"

from src.core.trace_io import open_trace


def _sh(cmd: list[str], timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _rebuild(*targets: str) -> None:
    if (REPO / "build" / "CMakeCache.txt").exists() and shutil.which("cmake"):
        _sh(["cmake", "--build", str(REPO / "build"), "--target", *targets], timeout=600)


def _profile(backend: str, binary: str, out_json: str) -> tuple[subprocess.CompletedProcess, object]:
    proc = _sh([sys.executable, str(REPO / "hprofiler"), "run", "--backend", backend, "--disasm",
                "--no-ui", "--no-summary", "-o", out_json, "--", "env", binary])
    store = out_json[:-len(".json")] + ".hpstore"
    trace = open_trace(store) if os.path.isdir(store) else None
    return proc, trace


class _Base(unittest.TestCase):
    tmp: str

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="hprofiler_callsite_")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def assert_lossless(self, trace, hook: str):
        health = trace.metadata.capture_health
        self.assertEqual(health.get("state"), "complete")
        entries = [v for k, v in health.get("transport", {}).items() if k.endswith("/" + hook)]
        self.assertTrue(entries, f"no transport status from the {hook} hook: {health!r}")
        for e in entries:
            self.assertEqual(e.get("final"), 1, e)
            self.assertFalse(e.get("ended_without_final"), e)
            self.assertEqual(e.get("received"), e.get("emitted"), e)
            for k in ("dropped_full", "dropped_lost", "oversize", "format_errors"):
                self.assertEqual(e.get(k), 0, (k, e))

    def assert_symfile_is(self, ev, binary: str):
        self.assertEqual(os.path.realpath(ev.tags.get("symfile", "")), os.path.realpath(binary),
                         f"{ev.name}: symfile must name the profiled binary, not the launcher: {ev.tags!r}")


@unittest.skipUnless(shutil.which("mpicc") and (LIB / "libhprofiler_mpi.so").exists(),
                     "mpicc or build/lib/libhprofiler_mpi.so not available")
class TestMpiCallsites(_Base):
    P2P = ("MPI_Send", "MPI_Ssend", "MPI_Recv", "MPI_Isend", "MPI_Irecv", "MPI_Wait",
           "MPI_Waitall", "MPI_Waitany", "MPI_Waitsome")
    INSTANTS = ("MPI_Test", "MPI_Testall", "MPI_Testany", "MPI_Testsome", "MPI_Cancel")

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        _rebuild("hprofiler_mpi")
        cls.binary = os.path.join(cls.tmp, "mpi_callsite")
        b = _sh(["mpicc", "-O1", "-g", "-rdynamic", "-o", cls.binary, str(FIX / "mpi_callsite.c")])
        if b.returncode:
            raise unittest.SkipTest(f"mpicc failed: {b.stderr[-1000:]}")
        cls.proc, cls.trace = _profile("mpi", cls.binary, os.path.join(cls.tmp, "m.json"))
        if cls.trace is None:
            raise AssertionError(f"no trace store written: {cls.proc.stdout}\n{cls.proc.stderr}")

    def test_every_p2p_and_request_span_carries_the_application_call_site(self):
        spans = [s for s in self.trace.iter_spans() if s.category.value == "mpi"]
        names = {s.name for s in spans}
        for n in self.P2P:
            self.assertIn(n, names)
        expected_fn = {"MPI_Send": "blocking_pairs", "MPI_Ssend": "blocking_pairs",
                       "MPI_Recv": "blocking_pairs", "MPI_Waitany": "polling",
                       "MPI_Waitsome": "polling", "MPI_Barrier": "main"}
        for s in spans:
            self.assertTrue(s.tags.get("sym"), f"{s.name} has no sym= tag: {s.tags!r}")
            self.assert_symfile_is(s, self.binary)
            if s.name in expected_fn:
                self.assertEqual(s.tags["sym"], expected_fn[s.name], s.tags)
        self.assertEqual({s.tags["sym"] for s in spans if s.name == "MPI_Isend"},
                         {"blocking_pairs", "many_requests", "polling"})

    def test_test_and_cancel_instants_carry_the_call_site(self):
        insts = [i for i in self.trace.iter_instants() if i.category.value == "mpi"]
        self.assertEqual({i.name for i in insts}, set(self.INSTANTS))
        for i in insts:
            self.assertEqual(i.tags.get("sym"), "polling", i.tags)
            self.assert_symfile_is(i, self.binary)

    def test_disassembly_attached_for_each_name_even_when_functions_are_shared(self):
        # MPI_Send/MPI_Recv/MPI_Ssend all come from blocking_pairs(): each
        # name gets that function's listing.
        for n in self.P2P + ("MPI_Barrier",):
            kd = self.trace.disasm.get(n)
            self.assertIsNotNone(kd, f"no disassembly for {n}; have {sorted(self.trace.disasm)}")
            self.assertTrue(kd.lines)
            self.assertEqual(os.path.realpath(kd.source), os.path.realpath(self.binary))
        self.assertEqual(self.trace.disasm["MPI_Recv"].mangled_name, "blocking_pairs")
        self.assertEqual(self.trace.disasm["MPI_Waitsome"].mangled_name, "polling")

    def test_long_request_lists_are_bounded_and_counted_exactly(self):
        big = [s for s in self.trace.iter_spans()
               if s.name == "MPI_Waitall" and s.tags.get("count") == "2000"]
        self.assertEqual(len(big), 2)
        for s in big:
            listed = [x for x in s.parent_span_id.split(";") if x]
            omitted = int(s.tags.get("psid_omitted", "0"))
            self.assertGreater(omitted, 0, "fixture must exceed the bounded list")
            self.assertEqual(len(listed) + omitted, 2000)
            self.assertTrue(all(x.isdigit() for x in listed))

    def test_capture_is_lossless(self):
        self.assertEqual(self.proc.returncode, 0, self.proc.stderr)
        self.assertIn("many_requests sum=1999000", self.proc.stdout)
        self.assert_lossless(self.trace, "mpi")


def _cc_openmp(cc: str) -> bool:
    if not shutil.which(cc):
        return False
    p = subprocess.run([cc, "-fopenmp", "-x", "c", "-o", os.devnull, "-"],
                       input="int main(void){return 0;}", capture_output=True, text=True)
    return p.returncode == 0


@unittest.skipUnless(_cc_openmp("gcc") and (LIB / "libhprofiler_gomp.so").exists(),
                     "gcc -fopenmp or build/lib/libhprofiler_gomp.so not available")
class TestGompCallsites(_Base):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        _rebuild("hprofiler_gomp", "hprofiler_ompt")
        cls.binary = os.path.join(cls.tmp, "omp_gcc")
        b = _sh(["gcc", "-fopenmp", "-O1", "-g", "-rdynamic", "-o", cls.binary,
                 str(FIX / "omp_callsite.c")])
        if b.returncode:
            raise unittest.SkipTest(f"gcc failed: {b.stderr[-1000:]}")
        cls.proc, cls.trace = _profile("openmp", cls.binary, os.path.join(cls.tmp, "g.json"))
        if cls.trace is None:
            raise AssertionError(f"no trace store written: {cls.proc.stdout}\n{cls.proc.stderr}")

    def test_critical_hold_carries_a_call_site_and_gets_disassembly(self):
        holds = [s for s in self.trace.iter_spans() if s.name == "omp_critical_hold"]
        self.assertEqual(len(holds), 4)
        for s in holds:
            self.assertTrue(s.tags.get("sym") or s.tags.get("lib"), s.tags)
        kd = self.trace.disasm.get("omp_critical_hold")
        self.assertIsNotNone(kd, sorted(self.trace.disasm))
        self.assertIn("region_with_sync", kd.mangled_name)

    def test_parallel_region_resolves_into_the_wrapped_binary(self):
        regions = [s for s in self.trace.iter_spans() if s.name == "omp_parallel_region"]
        self.assertTrue(regions)
        for s in regions:
            self.assertEqual(s.tags.get("sym"), "region_with_sync", s.tags)
            self.assert_symfile_is(s, self.binary)
        self.assertIn("omp_parallel_region", self.trace.disasm)

    def test_capture_is_lossless(self):
        self.assertIn("sum=6", self.proc.stdout)
        self.assert_lossless(self.trace, "gomp")


@unittest.skipUnless(_cc_openmp("clang") and (LIB / "libhprofiler_ompt.so").exists(),
                     "clang -fopenmp (libomp) or build/lib/libhprofiler_ompt.so not available")
class TestOmptCallsites(_Base):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        _rebuild("hprofiler_gomp", "hprofiler_ompt")
        cls.binary = os.path.join(cls.tmp, "omp_clang")
        b = _sh(["clang", "-fopenmp", "-O1", "-g", "-rdynamic", "-o", cls.binary,
                 str(FIX / "omp_callsite.c")])
        if b.returncode:
            raise unittest.SkipTest(f"clang failed: {b.stderr[-1000:]}")
        if "libomp" not in _sh(["ldd", cls.binary]).stdout:
            raise unittest.SkipTest("clang -fopenmp did not link LLVM libomp")
        cls.proc, cls.trace = _profile("openmp", cls.binary, os.path.join(cls.tmp, "o.json"))
        if cls.trace is None:
            raise AssertionError(f"no trace store written: {cls.proc.stdout}\n{cls.proc.stderr}")

    def test_launcher_wrapped_ompt_spans_name_the_real_binary(self):
        tagged = [s for s in self.trace.iter_spans() if s.tags.get("sym")]
        self.assertTrue(tagged, "no OMPT span resolved to an exported symbol")
        for s in tagged:
            self.assert_symfile_is(s, self.binary)
        region = [s for s in self.trace.iter_spans() if s.name == "parallel_region"]
        self.assertEqual(len(region), 1)
        self.assertTrue(region[0].tags.get("sym"), region[0].tags)
        self.assertIn("parallel_region", self.trace.disasm)
        self.assertEqual(os.path.realpath(self.trace.disasm["parallel_region"].source),
                         os.path.realpath(self.binary))

    def test_worksharing_and_barrier_spans_get_disassembly(self):
        for n in ("omp_loop", "omp_barrier_explicit"):
            self.assertIn(n, self.trace.disasm, sorted(self.trace.disasm))
            self.assertIn("omp_outlined", self.trace.disasm[n].mangled_name)

    def test_capture_is_lossless(self):
        self.assertIn("sum=6", self.proc.stdout)
        self.assert_lossless(self.trace, "ompt")


if __name__ == "__main__":
    unittest.main()
