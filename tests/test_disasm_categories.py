"""
Regression tests for src/core/runner.py's _collect_disasm(): the loop that
extracts sym=/lib= codeptr tags into `omp_syms` (later passed to
disasm/extractor.py's collect_disasm()) must include "mpi" spans alongside
"openmp"/"sync"/"cpu" -- mpi_hook.c emits sym=/lib= tags on its
collective-call spans (MPI_Bcast/MPI_Allreduce/MPI_Barrier), and without
them the Source tab shows "No disassembly available" for every MPI span.

Also covers launcher-wrapped commands (`hprofiler run -- srun -n 4 gmx_mpi
...`, where `command[0]` is `srun`, not the binary holding the resolved
symbol): the hooks emit `symfile=<path>` (dladdr's `dli_fname` -- the ELF
the symbol was found in) alongside `sym=`, `omp_syms["name"]`'s "sym"
payload is `(sym_name, symfile)`, and collect_disasm() uses `symfile` when
present, falling back to `command[0]` only for tags from a hook build
without symfile= (symfile is None).

Mocks disasm/extractor.collect_disasm() (a real implementation shells out
to nm/objdump against a real binary, which this test has no need for --
only the `omp_syms` dict _collect_disasm builds and hands it matters
here) rather than exercising the full extraction pipeline, which is
already covered at the wire-protocol level by
tests/integration/test_gomp_hook.py and test_mpi_protocol.py's
"...carry_a_resolved_codeptr_tag_for_disasm" tests, and end-to-end
(hook -> wire protocol -> this function -> real objdump output) by
tests/integration/test_callsite_e2e.py.
"""
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.trace import Trace, TraceMetadata
from src.core.events import SpanEvent, Category
from src.core.runner import _collect_disasm
from src.disasm.extractor import collect_disasm


def _span(cat, name, tags):
    return SpanEvent(name=name, category=cat, start_ns=0, duration_ns=1000,
                     pid=1, tid=1, tags=dict(tags))


class TestCollectDisasmCategoryFilter(unittest.TestCase):
    def _omp_syms_seen(self, trace: Trace) -> dict:
        with patch("src.disasm.extractor.collect_disasm") as mock_collect:
            mock_collect.return_value = {}
            _collect_disasm(trace, ["./app"], ["mpi"])
            self.assertTrue(mock_collect.called, "collect_disasm was never invoked")
            _command, _backends, _jit_spans, omp_syms, *_rest = mock_collect.call_args[0]
            return omp_syms

    def test_mpi_span_with_sym_tag_is_included(self):
        trace = Trace(TraceMetadata(command="./app", cwd=""))
        trace.add(_span(Category.MPI, "MPI_Allreduce", {"sym": "compute_forces"}))
        omp_syms = self._omp_syms_seen(trace)
        self.assertIn("MPI_Allreduce", omp_syms)
        # payload is (sym_name, symfile) -- symfile is None here since no
        # symfile= tag was present (older hook build / dladdr didn't
        # report dli_fname).
        self.assertEqual(omp_syms["MPI_Allreduce"], ("sym", ("compute_forces", None)))

    def test_mpi_span_with_sym_and_symfile_tags_is_included(self):
        # symfile= is the ELF dladdr() actually found the symbol in --
        # NOT necessarily command[0], since the profiled command is
        # routinely a launcher (srun/mpirun) wrapping the real binary.
        trace = Trace(TraceMetadata(command="srun", cwd=""))
        trace.add(_span(Category.MPI, "MPI_Allreduce",
                        {"sym": "compute_forces", "symfile": "/opt/app/gmx_mpi"}))
        omp_syms = self._omp_syms_seen(trace)
        self.assertEqual(omp_syms["MPI_Allreduce"], ("sym", ("compute_forces", "/opt/app/gmx_mpi")))

    def test_mpi_span_with_lib_offset_tag_is_included(self):
        trace = Trace(TraceMetadata(command="./app", cwd=""))
        trace.add(_span(Category.MPI, "MPI_Barrier", {"lib": "/opt/app/app", "offset": "0x1a2b"}))
        omp_syms = self._omp_syms_seen(trace)
        self.assertIn("MPI_Barrier", omp_syms)
        self.assertEqual(omp_syms["MPI_Barrier"], ("lib", ("/opt/app/app", 0x1a2b)))

    def test_mpi_span_with_no_codeptr_tags_is_not_included(self):
        # No sym=/lib= at all (e.g. an older hook build, or resolution
        # genuinely failed) -- must not crash or add a bogus entry.
        trace = Trace(TraceMetadata(command="./app", cwd=""))
        trace.add(_span(Category.MPI, "MPI_Send", {"type": "send", "rank": "0"}))
        omp_syms = self._omp_syms_seen(trace)
        self.assertNotIn("MPI_Send", omp_syms)

    def test_openmp_span_still_included_unaffected_by_the_fix(self):
        # The "openmp"/"sync"/"cpu" categories keep working.
        trace = Trace(TraceMetadata(command="./app", cwd=""))
        trace.add(_span(Category.OPENMP, "omp_parallel_region", {"sym": "main"}))
        omp_syms = self._omp_syms_seen(trace)
        self.assertIn("omp_parallel_region", omp_syms)
        self.assertEqual(omp_syms["omp_parallel_region"], ("sym", ("main", None)))


@unittest.skipUnless(shutil.which("gcc"), "gcc not available")
class TestCollectDisasmUsesSymfileNotLauncher(unittest.TestCase):
    """
    Real (non-mocked) end-to-end check of the launcher-wrapped scenario
    (`hprofiler run --disasm -- srun -n 1 gmx_mpi ...`, command[0] =
    "srun"): builds a REAL binary with a known function and confirms
    collect_disasm() finds it via `symfile=`, not `command[0]` --
    command[0] here deliberately has no such symbol, standing in for
    "srun".
    """

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="hprofiler_symfile_test_")
        cls.real_binary = str(Path(cls.tmp) / "real_program")
        src = Path(cls.tmp) / "real_program.c"
        src.write_text(
            "int hprofiler_test_target_function(int x) { return x * 2; }\n"
            "int main(void) { return hprofiler_test_target_function(21) == 42 ? 0 : 1; }\n"
        )
        r = subprocess.run(["gcc", "-O0", "-g", "-o", cls.real_binary, str(src)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise unittest.SkipTest(f"failed to build test binary: {r.stderr}")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_disassembles_symfile_not_command0(self):
        fake_launcher = "/definitely/not/a/real/path/srun"
        omp_syms = {
            "omp_parallel_region": (
                "sym", ("hprofiler_test_target_function", self.real_binary)
            ),
        }
        result = collect_disasm(
            command=[fake_launcher], backends=["openmp"], jit_spans=[],
            omp_syms=omp_syms,
        )
        self.assertIn("omp_parallel_region", result,
                      "collect_disasm did not disassemble the symbol via symfile= "
                      "-- it likely fell back to (nonexistent) command[0] instead")
        kd = result["omp_parallel_region"]
        self.assertTrue(kd.lines, "disasm entry present but has no instruction lines")

    def test_falls_back_to_command0_when_symfile_is_none(self):
        # Old-format tag (no symfile=, from a hook build without it) --
        # must still work when command[0] genuinely IS the right binary.
        omp_syms = {
            "omp_parallel_region": ("sym", ("hprofiler_test_target_function", None)),
        }
        result = collect_disasm(
            command=[self.real_binary], backends=["openmp"], jit_spans=[],
            omp_syms=omp_syms,
        )
        self.assertIn("omp_parallel_region", result)

    def test_real_binary_param_overrides_command0_with_no_tags_at_all(self):
        # The CPU-perf-sampled-by-name path (cpu_names) has no sym=/lib=
        # tag to fall back on at all -- it just does nm/objdump on
        # `binary` directly, same as CUDA/ROCm AoT disasm (which needs
        # real CUDA/ROCm hardware+toolchain to test end-to-end here, but
        # goes through this exact same `binary` variable -- see
        # collect_disasm()'s real_binary override, applied once, before
        # any backend-specific branch). Proves the override actually
        # takes effect for this "no per-span tag at all" case, not just
        # the OpenMP sym=/lib= paths the other tests here cover.
        fake_launcher = "/definitely/not/a/real/path/srun"
        result = collect_disasm(
            command=[fake_launcher], backends=["cpu"], jit_spans=[],
            cpu_names={"hprofiler_test_target_function"},
            real_binary=self.real_binary,
        )
        self.assertIn("hprofiler_test_target_function", result,
                      "collect_disasm did not use real_binary -- it likely tried "
                      "(nonexistent) command[0] instead")

    def test_real_binary_absent_falls_back_to_command0_unchanged(self):
        # No real_binary passed at all (e.g. no hook ever connected, or
        # SO_PEERCRED isn't available) -- command[0] is used.
        result = collect_disasm(
            command=[self.real_binary], backends=["cpu"], jit_spans=[],
            cpu_names={"hprofiler_test_target_function"},
        )
        self.assertIn("hprofiler_test_target_function", result)

    def test_symfile_still_takes_priority_over_real_binary(self):
        # symfile= (dladdr-resolved, exact) is more specific than
        # real_binary (SO_PEERCRED-resolved, whole-process) -- must not
        # regress: when both are available, symfile= still wins. In this
        # test they happen to point at the same real binary, so this
        # mainly proves passing real_binary doesn't break the existing
        # symfile= path at all.
        omp_syms = {
            "omp_parallel_region": (
                "sym", ("hprofiler_test_target_function", self.real_binary)
            ),
        }
        result = collect_disasm(
            command=["/definitely/not/a/real/path/srun"], backends=["openmp"],
            jit_spans=[], omp_syms=omp_syms, real_binary=self.real_binary,
        )
        self.assertIn("omp_parallel_region", result)
        self.assertTrue(result["omp_parallel_region"].lines)

    def test_mangled_name_is_the_real_symbol_not_the_span_label(self):
        # kd.name stays the span/event label ("omp_parallel_region") for
        # display grouping, but
        # kd.mangled_name must carry the REAL resolved symbol that was
        # actually disassembled (here: hprofiler_test_target_function),
        # not be left empty. Two things depend on this: annotate_with_perf
        # filtering `perf annotate` by a symbol that actually exists (see
        # the sibling test below), and UIs showing the user what function
        # they're really looking at.
        omp_syms = {
            "omp_parallel_region": (
                "sym", ("hprofiler_test_target_function", self.real_binary)
            ),
        }
        result = collect_disasm(
            command=[self.real_binary], backends=["openmp"], jit_spans=[],
            omp_syms=omp_syms,
        )
        kd = result["omp_parallel_region"]
        self.assertEqual(kd.name, "omp_parallel_region")
        self.assertEqual(kd.mangled_name, "hprofiler_test_target_function")


class TestAnnotateWithPerfSymbolFilter(unittest.TestCase):
    """annotate_with_perf() must filter `perf annotate -s` by the REAL
    resolved symbol (kd.mangled_name), not kd.name: for an OMP/MPI-hook-
    resolved kernel kd.name is an hprofiler event label ("omp_barrier",
    "MPI_Bcast") that no ELF symbol is named, so filtering by it matches
    nothing and every DisasmLine.sample_pct stays 0 even though perf
    recorded samples -- disassembly shown, but no sample statistics.

    Mocks the subprocess call and asserts on the constructed argv, since
    a real `perf record` is blocked where perf_event_paranoid > 2 (as on
    the development machine)."""

    def test_uses_mangled_name_as_the_symbol_filter_when_present(self):
        from src.disasm.extractor import annotate_with_perf, KernelDisasm, DisasmLine
        kd = KernelDisasm(
            name="omp_barrier", arch="x86-64", source="/bin/gmx_mpi",
            mangled_name="hprofiler_test_target_function",
            lines=[DisasmLine(addr=0x1000, mnemonic="push", operands="rbp")],
        )
        with patch("src.disasm.extractor.shutil.which", return_value="/usr/bin/perf"), \
             patch("src.disasm.extractor.Path.exists", return_value=True), \
             patch("src.disasm.extractor._run", return_value="") as mock_run:
            annotate_with_perf(kd, "/tmp/fake_perf.data")
        first_call_argv = mock_run.call_args_list[0].args[0]
        self.assertIn("hprofiler_test_target_function", first_call_argv)
        self.assertNotIn("omp_barrier", first_call_argv)

    def test_falls_back_to_kd_name_when_mangled_name_is_unset(self):
        # The OTHER caller of annotate_with_perf (perf-sampled-by-name CPU
        # kernels, src/core/runner.py) never sets mangled_name at all --
        # kd.name there already IS the real symbol, so it must still be
        # used as the filter in that case.
        from src.disasm.extractor import annotate_with_perf, KernelDisasm, DisasmLine
        kd = KernelDisasm(
            name="hot_cpu_function", arch="x86-64", source="/bin/a.out",
            lines=[DisasmLine(addr=0x1000, mnemonic="push", operands="rbp")],
        )
        with patch("src.disasm.extractor.shutil.which", return_value="/usr/bin/perf"), \
             patch("src.disasm.extractor.Path.exists", return_value=True), \
             patch("src.disasm.extractor._run", return_value="") as mock_run:
            annotate_with_perf(kd, "/tmp/fake_perf.data")
        first_call_argv = mock_run.call_args_list[0].args[0]
        self.assertIn("hot_cpu_function", first_call_argv)


class TestDemangle(unittest.TestCase):
    """analysis/dashboard.py's demangle() -- added alongside the
    mangled_name fix above so both UIs can show the user a readable
    function name (e.g. "gmx::ThreadedForceBuffer<...>::ThreadedForceBuffer(...)")
    instead of the raw mangled symbol or, worse, nothing at all."""

    @unittest.skipUnless(shutil.which("c++filt"), "c++filt not installed")
    def test_demangles_a_real_mangled_cpp_symbol(self):
        from src.analysis.dashboard import demangle
        # The exact symbol from a real Dardel/GROMACS trace (see
        # hooks/common/codeptr_resolve.h's own docstring example).
        result = demangle("_ZN3gmx19ThreadedForceBufferIA4_fEC2Eibi")
        self.assertIn("ThreadedForceBuffer", result)
        self.assertNotEqual(result, "_ZN3gmx19ThreadedForceBufferIA4_fEC2Eibi")

    def test_returns_input_unchanged_when_cxxfilt_unavailable(self):
        from src.analysis.dashboard import demangle
        demangle.cache_clear()
        with patch("src.analysis.dashboard.subprocess.run",
                   side_effect=FileNotFoundError):
            self.assertEqual(demangle("some_plain_c_function"), "some_plain_c_function")
        demangle.cache_clear()

    def test_empty_name_returns_empty_without_shelling_out(self):
        from src.analysis.dashboard import demangle
        self.assertEqual(demangle(""), "")


if __name__ == "__main__":
    unittest.main()
