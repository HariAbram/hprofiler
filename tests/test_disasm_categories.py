"""
Regression test for src/core/runner.py's _collect_disasm(): the loop that
extracts sym=/lib= codeptr tags into `omp_syms` (later passed to
disasm/extractor.py's collect_disasm()) filtered on
`span.category.value in ("openmp", "sync", "cpu")` -- "mpi" was missing,
so even after mpi_hook.c started emitting sym=/lib= tags on its
collective-call spans (MPI_Bcast/MPI_Allreduce/MPI_Barrier, see
hooks/mpi_hook/mpi_hook.c), this loop never looked at them. The Source
tab's kernel list includes every profiled span name, not just GPU
kernels, so those MPI spans showed "No disassembly available"
unconditionally -- not because objdump was missing, but because nothing
ever tried.

Also covers a second, subtler bug found immediately after the first fix
shipped: a real user's Source tab showed "Call site resolved (sym=...)
but disassembly still failed" -- the sym= tag WAS present and correct,
but collect_disasm() always disassembled `command[0]` for a "sym" entry,
and the profiled command was `hprofiler run -- srun -n 4 gmx_mpi ...`,
so `command[0]` was `srun`, not `gmx_mpi` (where the resolved symbol
actually lives). Fixed by having the hooks also emit `symfile=<path>`
(dladdr's own `dli_fname` -- the ELF the symbol was actually found in)
alongside `sym=`, and `omp_syms["name"]`'s "sym" payload is now
`(sym_name, symfile)` instead of a bare string; collect_disasm() uses
`symfile` when present, falling back to `command[0]` only for tags from
an older hook build (symfile is None).

Mocks disasm/extractor.collect_disasm() (a real implementation shells out
to nm/objdump against a real binary, which this test has no need for --
only the `omp_syms` dict _collect_disasm builds and hands it matters
here) rather than exercising the full extraction pipeline, which is
already covered at the wire-protocol level by
tests/integration/test_gomp_hook.py and test_mpi_protocol.py's
"...carry_a_resolved_codeptr_tag_for_disasm" tests, and end-to-end
(hook -> wire protocol -> this exact function -> real objdump output) by
manual verification during development (see project_disasm_codeptr_fix
memory).
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
        # Non-regression: the pre-existing "openmp"/"sync"/"cpu" categories
        # must keep working exactly as before.
        trace = Trace(TraceMetadata(command="./app", cwd=""))
        trace.add(_span(Category.OPENMP, "omp_parallel_region", {"sym": "main"}))
        omp_syms = self._omp_syms_seen(trace)
        self.assertIn("omp_parallel_region", omp_syms)
        self.assertEqual(omp_syms["omp_parallel_region"], ("sym", ("main", None)))


@unittest.skipUnless(shutil.which("gcc"), "gcc not available")
class TestCollectDisasmUsesSymfileNotLauncher(unittest.TestCase):
    """
    Real (non-mocked) end-to-end reproduction of the launcher-wrapped
    scenario: a real user ran `hprofiler run --disasm -- srun -n 1
    gmx_mpi ...`, so command[0] was "srun", not "gmx_mpi" -- their
    Source tab showed a correctly-resolved `sym=_ZN3gmx19...` but still
    "No disassembly available", because collect_disasm() always
    disassembled command[0] for a "sym" entry regardless of where the
    symbol actually was. Builds a REAL binary with a known function and
    confirms collect_disasm() finds it via `symfile=`, not `command[0]`
    -- command[0] here is deliberately something with no such symbol at
    all, standing in for "srun".
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
        # Old-format tag (no symfile=, e.g. from a hook build that
        # predates this fix) -- must still work exactly as before when
        # command[0] genuinely IS the right binary.
        omp_syms = {
            "omp_parallel_region": ("sym", ("hprofiler_test_target_function", None)),
        }
        result = collect_disasm(
            command=[self.real_binary], backends=["openmp"], jit_spans=[],
            omp_syms=omp_syms,
        )
        self.assertIn("omp_parallel_region", result)


if __name__ == "__main__":
    unittest.main()
