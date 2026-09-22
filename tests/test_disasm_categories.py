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

Mocks disasm/extractor.collect_disasm() (a real implementation shells out
to nm/objdump against a real binary, which this test has no need for --
only the `omp_syms` dict _collect_disasm builds and hands it matters
here) rather than exercising the full extraction pipeline, which is
already covered at the wire-protocol level by
tests/integration/test_gomp_hook.py and test_mpi_protocol.py's
"...carry_a_resolved_codeptr_tag_for_disasm" tests, and end-to-end
(hook -> wire protocol -> this exact function -> real objdump output) by
manual verification during development (see project_dashboard_redesign
memory / this fix's commit).
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.trace import Trace, TraceMetadata
from src.core.events import SpanEvent, Category
from src.core.runner import _collect_disasm


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
        self.assertEqual(omp_syms["MPI_Allreduce"], ("sym", "compute_forces"))

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
        self.assertEqual(omp_syms["omp_parallel_region"], ("sym", "main"))


if __name__ == "__main__":
    unittest.main()
