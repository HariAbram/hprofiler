"""
Unit tests for src/analysis/pop_efficiency.py against hand-computed expected
values on small synthetic traces -- verifies the formulas themselves, not
just "it runs".
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.trace import Trace, TraceMetadata
from src.core.events import SpanEvent, CounterEvent, Category
from src.analysis import pop_efficiency as pe


def _mk_trace(spans, counters=None, start=0, end=None):
    meta = TraceMetadata(start_time_ns=start, end_time_ns=end or 0)
    t = Trace(meta)
    for s in spans:
        t.add(s)
    for c in counters or []:
        t.add(c)
    return t


def _span(pid, cat, start_ns, dur_ns, tags=None, tid=1, name="s"):
    return SpanEvent(name=name, category=cat, start_ns=start_ns, duration_ns=dur_ns,
                      pid=pid, tid=tid, tags=dict(tags or {}))


class TestLoadBalance(unittest.TestCase):
    def test_perfect_balance(self):
        useful = {1: 100, 2: 100, 3: 100}
        self.assertAlmostEqual(pe.load_balance(useful), 1.0)

    def test_imbalanced(self):
        useful = {1: 100, 2: 50}
        # avg=75, max=100 -> 0.75
        self.assertAlmostEqual(pe.load_balance(useful), 0.75)

    def test_empty_is_none(self):
        self.assertIsNone(pe.load_balance({}))


class TestCommEfficiency(unittest.TestCase):
    def test_known_ratio(self):
        useful = {1: 80, 2: 90}
        # max(useful)=90, wall=100 -> 0.9
        self.assertAlmostEqual(pe.comm_efficiency(useful, 100), 0.9)

    def test_capped_at_one(self):
        useful = {1: 150}
        self.assertLessEqual(pe.comm_efficiency(useful, 100), 1.0)


class TestUsefulTimeByPid(unittest.TestCase):
    def test_merges_overlapping_gpu_streams_no_double_count(self):
        # Two overlapping CUDA kernel spans on the same pid, different
        # streams -- true busy wall-clock time is the union, not the sum.
        spans = [
            _span(1, Category.GPU_CUDA, 0, 100, tags={"stream": "1"}),
            _span(1, Category.GPU_CUDA, 50, 100, tags={"stream": "2"}),  # overlaps [0,100) by 50ns
        ]
        trace = _mk_trace(spans)
        useful = pe.useful_time_by_pid(trace)
        # union of [0,100) and [50,150) = [0,150) = 150, NOT 100+100=200
        self.assertEqual(useful[1], 150)

    def test_excludes_mpi_and_sync(self):
        spans = [
            _span(1, Category.GPU_CUDA, 0, 100),
            _span(1, Category.MPI, 100, 500, tags={"type": "barrier"}),
            _span(1, Category.SYNC, 600, 50, name="cudaDeviceSynchronize"),
        ]
        trace = _mk_trace(spans)
        useful = pe.useful_time_by_pid(trace)
        self.assertEqual(useful[1], 100)  # only the cuda span counts


class TestAlphaBetaFit(unittest.TestCase):
    def test_recovers_known_model_exactly(self):
        true_alpha, true_beta = 1000.0, 2.0  # ns latency, bytes/ns bandwidth
        spans = []
        for b in (64, 1024, 8192, 65536, 1_000_000):
            dur = true_alpha + b / true_beta
            spans.append(_span(1, Category.MPI, 0, int(dur), tags={"bytes": str(b), "type": "send"}))
        alpha, beta = pe.fit_alpha_beta(spans)
        self.assertAlmostEqual(alpha, true_alpha, delta=1.0)
        self.assertAlmostEqual(beta, true_beta, delta=0.01)

    def test_insufficient_points_returns_none(self):
        spans = [_span(1, Category.MPI, 0, 100, tags={"bytes": "64"})]
        self.assertIsNone(pe.fit_alpha_beta(spans))

    def test_no_size_variance_returns_none(self):
        spans = [_span(1, Category.MPI, 0, 100, tags={"bytes": "64"}) for _ in range(5)]
        self.assertIsNone(pe.fit_alpha_beta(spans))

    def test_transfer_efficiency_perfect_when_actual_matches_ideal(self):
        alpha, beta = 100.0, 4.0
        spans = [_span(1, Category.MPI, 0, int(alpha + b / beta), tags={"bytes": str(b)})
                 for b in (100, 10_000, 1_000_000)]
        eff = pe.transfer_efficiency_proxy(spans, alpha, beta)
        self.assertAlmostEqual(eff, 1.0, delta=1e-6)

    def test_transfer_efficiency_detects_inflated_duration(self):
        alpha, beta = 100.0, 4.0
        # Actual duration is 2x the ideal -> efficiency should be ~0.5
        spans = [_span(1, Category.MPI, 0, int(2 * (alpha + b / beta)), tags={"bytes": str(b)})
                 for b in (100, 10_000, 1_000_000)]
        eff = pe.transfer_efficiency_proxy(spans, alpha, beta)
        self.assertAlmostEqual(eff, 0.5, delta=0.05)


class TestNCCLBusBandwidth(unittest.TestCase):
    def test_known_formula(self):
        # 4 ranks, 1 GiB message, 0.1s -> busBW = 2*(4-1)/4 * bytes/time
        n, b, secs = 4, 1024**3, 0.1
        expected_gbs = 2 * (n - 1) / n * b / secs / 1e9
        spans = [_span(1, Category.NCCL, 0, int(secs * 1e9),
                        tags={"type": "allreduce", "bytes": str(b), "nranks": str(n)})]
        trace = _mk_trace(spans)
        got = pe.nccl_bus_bandwidth_gbs(trace)
        self.assertAlmostEqual(got, expected_gbs, delta=1e-6)

    def test_no_nccl_spans_returns_none(self):
        trace = _mk_trace([_span(1, Category.MPI, 0, 100)])
        self.assertIsNone(pe.nccl_bus_bandwidth_gbs(trace))


class TestComputationalScaling(unittest.TestCase):
    def test_ratio_capped_at_one(self):
        base = _mk_trace([], counters=[CounterEvent(name="ipc", category=Category.CPU,
                                                      timestamp_ns=0, value=2.0)])
        now = _mk_trace([], counters=[CounterEvent(name="ipc", category=Category.CPU,
                                                     timestamp_ns=0, value=3.0)])
        self.assertAlmostEqual(pe.computational_scaling(now, base), 1.0)  # capped

    def test_degraded_ipc_below_one(self):
        base = _mk_trace([], counters=[CounterEvent(name="ipc", category=Category.CPU,
                                                      timestamp_ns=0, value=2.0)])
        now = _mk_trace([], counters=[CounterEvent(name="ipc", category=Category.CPU,
                                                     timestamp_ns=0, value=1.0)])
        self.assertAlmostEqual(pe.computational_scaling(now, base), 0.5)

    def test_missing_counters_is_none(self):
        base = _mk_trace([])
        now = _mk_trace([])
        self.assertIsNone(pe.computational_scaling(now, base))


class TestDurationWeightedPct(unittest.TestCase):
    """Regression tests for a self-audit bug: gpu_efficiency() originally
    weighted by achieved_tflops (a RATE) instead of duration_ns, and used
    `x or 1.0` which silently treated a genuine 0.0-tflops kernel as full
    weight (0.0 is falsy in Python)."""

    def test_weighted_by_duration_not_by_rate(self):
        # A long, low-utilization kernel should dominate a short,
        # high-utilization one -- not the other way around.
        pairs = [(1000, 10.0), (10, 90.0)]  # (duration_ns, flops_pct)
        result = pe.duration_weighted_pct(pairs)
        expected = (1000 * 10.0 + 10 * 90.0) / 1010 / 100.0
        self.assertAlmostEqual(result, expected, places=6)
        self.assertLess(result, 0.15)  # dominated by the long low-pct kernel

    def test_zero_pct_kernel_with_real_duration_pulls_average_down(self):
        # A kernel achieving genuinely 0% (e.g. pure memory-bound, no FP
        # ops) must count with its real duration as weight, not be
        # excluded/mis-weighted just because its pct (or, in the old bug,
        # its rate-based weight) was 0.
        pairs = [(100, 0.0), (100, 100.0)]
        result = pe.duration_weighted_pct(pairs)
        self.assertAlmostEqual(result, 0.5, places=6)

    def test_empty_is_none(self):
        self.assertIsNone(pe.duration_weighted_pct([]))

    def test_all_zero_duration_is_none(self):
        self.assertIsNone(pe.duration_weighted_pct([(0, 50.0), (0, 90.0)]))


class TestPerCategoryTransferEfficiency(unittest.TestCase):
    """Regression test for a self-audit bug: fitting one alpha/beta model
    across MPI and NCCL spans mixed together lets whichever category
    dominates the regression corrupt the "ideal" time for the other."""

    def test_mpi_and_nccl_fit_independently(self):
        # MPI: slow network-like model. NCCL: fast NVLink-like model.
        mpi_alpha, mpi_beta = 5000.0, 0.5      # ns latency, bytes/ns (slow)
        nccl_alpha, nccl_beta = 100.0, 50.0    # much faster

        def mk(cat, alpha, beta, b, tid):
            dur = alpha + b / beta
            return _span(1, cat, 0, int(dur), tid=tid, tags={"bytes": str(b), "type": "send" if cat == Category.MPI else "allreduce"})

        spans = []
        for i, b in enumerate((64, 1024, 65536, 1_000_000)):
            spans.append(mk(Category.MPI, mpi_alpha, mpi_beta, b, tid=1))
            spans.append(mk(Category.NCCL, nccl_alpha, nccl_beta, b, tid=2))
        trace = _mk_trace(spans)

        per_cat, overall, notes = pe.transfer_efficiency_by_category(trace)
        # Each category was fit against its OWN (exactly-matching) model,
        # so each should score ~perfect efficiency independently -- if they
        # were wrongly fit together, at least one would be far from 1.0.
        self.assertIn("mpi", per_cat)
        self.assertIn("nccl", per_cat)
        self.assertAlmostEqual(per_cat["mpi"], 1.0, delta=0.01)
        self.assertAlmostEqual(per_cat["nccl"], 1.0, delta=0.01)
        self.assertIsNotNone(overall)


class TestUsefulTimeExcludesDataMovement(unittest.TestCase):
    """Regression test: memcpy/alloc/free spans are tagged with a
    'compute' category (cuda/rocm) but are data-movement, not computation
    -- POP's methodology means them to count as overhead, not useful time."""

    def test_memcpy_excluded_from_useful_time(self):
        spans = [
            _span(1, Category.GPU_CUDA, 0, 100, name="kernel", tags={"type": "kernel"}),
            _span(1, Category.GPU_CUDA, 100, 500, name="cudaMemcpy", tags={"type": "memcpy"}),
        ]
        trace = _mk_trace(spans)
        useful = pe.useful_time_by_pid(trace)
        self.assertEqual(useful[1], 100)  # only the kernel counts, not the 500ns memcpy


class TestAnalyzeEndToEnd(unittest.TestCase):
    def test_two_rank_trace_bounds_and_notes(self):
        spans = [
            _span(1, Category.GPU_CUDA, 0, 900, name="k"),
            _span(1, Category.MPI, 900, 100, tags={"type": "barrier"}),
            _span(2, Category.GPU_CUDA, 0, 700, name="k"),
            _span(2, Category.MPI, 700, 300, tags={"type": "barrier"}),
        ]
        trace = _mk_trace(spans)
        report = pe.analyze(trace)

        for val in (report.load_balance, report.comm_efficiency, report.parallel_efficiency):
            self.assertIsNotNone(val)
            self.assertGreaterEqual(val, 0.0)
            self.assertLessEqual(val, 1.0 + 1e-9)

        # rank1 useful=900, rank2 useful=700 -> load_balance = avg/max = 800/900
        self.assertAlmostEqual(report.load_balance, 800 / 900, places=6)

        # No baseline / critical path supplied -> both explicitly omitted with a note.
        self.assertIsNone(report.computational_scaling)
        self.assertIsNone(report.serialization_efficiency)
        joined_notes = " ".join(report.notes)
        self.assertIn("baseline", joined_notes)
        self.assertIn("critical-path", joined_notes)

    def test_single_rank_gets_explanatory_note(self):
        trace = _mk_trace([_span(1, Category.GPU_CUDA, 0, 100)])
        report = pe.analyze(trace)
        self.assertTrue(any("Only one rank/process" in n for n in report.notes))

    def test_empty_trace_does_not_crash_and_gets_correct_note(self):
        trace = _mk_trace([])
        report = pe.analyze(trace)
        self.assertIsNone(report.load_balance)
        self.assertTrue(any("No spans" in n for n in report.notes))
        self.assertFalse(any("Only one rank/process" in n for n in report.notes))


if __name__ == "__main__":
    unittest.main()
