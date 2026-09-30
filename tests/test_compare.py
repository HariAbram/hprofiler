"""
Unit tests for src/analysis/compare.py -- pure Python, no Qt (mirrors the
module's own docstring on why). Covers exact/normalized/unmatched matching
tiers, classify()'s dual noise-floor threshold, true-zero vs missing-data
distinction, and the aggregate/bucket/top-changes/report_dict rollups.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.trace import Trace, TraceMetadata
from src.core.events import SpanEvent, Category
from src.analysis import compare as cmp


def _span(cat, name, dur_ns, start_ns=0, tags=None):
    return SpanEvent(name=name, category=cat, start_ns=start_ns, duration_ns=dur_ns,
                      pid=1, tid=1, tags=dict(tags or {}))


def _trace(spans):
    t = Trace(TraceMetadata())
    for s in spans:
        t.add(s)
    return t


class TestMatchRows(unittest.TestCase):
    def test_exact_match(self):
        rows_a = [{"category": "cuda", "name": "matmul", "total_ns": 100}]
        rows_b = [{"category": "cuda", "name": "matmul", "total_ns": 120}]
        matched = cmp.match_rows(rows_a, rows_b)
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0]["matchKind"], "exact")
        self.assertEqual(matched[0]["base"]["total_ns"], 100)
        self.assertEqual(matched[0]["comp"]["total_ns"], 120)

    def test_baseline_only(self):
        rows_a = [{"category": "cuda", "name": "old_kernel", "total_ns": 100}]
        rows_b = []
        matched = cmp.match_rows(rows_a, rows_b)
        self.assertEqual(matched[0]["matchKind"], "baseline_only")
        self.assertIsNone(matched[0]["comp"])

    def test_comparison_only(self):
        rows_a = []
        rows_b = [{"category": "cuda", "name": "new_kernel", "total_ns": 50}]
        matched = cmp.match_rows(rows_a, rows_b)
        self.assertEqual(matched[0]["matchKind"], "comparison_only")
        self.assertIsNone(matched[0]["base"])

    def test_normalized_fallback_for_jit_hash_names(self):
        # Different raw JIT hash-named kernels whose fmt_kernel_name()
        # normalization collapses to the same form (see dash.fmt_kernel_name's
        # regex: last 6 digits of the first group, last 4 of the second).
        rows_a = [{"category": "cuda", "name": "111123456.789.jit.so", "total_ns": 100}]
        rows_b = [{"category": "cuda", "name": "222123456.789.jit.so", "total_ns": 110}]
        matched = cmp.match_rows(rows_a, rows_b)
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0]["matchKind"], "normalized")

    def test_normalized_fallback_never_crosses_category(self):
        rows_a = [{"category": "cuda", "name": "111123456.789.jit.so", "total_ns": 100}]
        rows_b = [{"category": "rocm", "name": "222123456.789.jit.so", "total_ns": 110}]
        matched = cmp.match_rows(rows_a, rows_b)
        kinds = {m["matchKind"] for m in matched}
        self.assertEqual(kinds, {"baseline_only", "comparison_only"})

    def test_comparison_row_used_at_most_once(self):
        # Two baseline rows that would BOTH normalize-match the same
        # single comparison row -- only one may claim it.
        rows_a = [
            {"category": "cuda", "name": "111123456.789.jit.so", "total_ns": 100},
            {"category": "cuda", "name": "333123456.789.jit.so", "total_ns": 90},
        ]
        rows_b = [{"category": "cuda", "name": "222123456.789.jit.so", "total_ns": 110}]
        matched = cmp.match_rows(rows_a, rows_b)
        normalized = [m for m in matched if m["matchKind"] == "normalized"]
        self.assertEqual(len(normalized), 1)
        baseline_only = [m for m in matched if m["matchKind"] == "baseline_only"]
        self.assertEqual(len(baseline_only), 1)


class TestClassify(unittest.TestCase):
    def test_both_missing_is_unavailable(self):
        status, delta = cmp.classify(None, None)
        self.assertEqual(status, "unavailable")
        self.assertEqual(delta["kind"], "unavailable")

    def test_missing_baseline_is_new(self):
        status, delta = cmp.classify(None, 500.0)
        self.assertEqual(status, "new")
        self.assertEqual(delta["value"], 500.0)

    def test_missing_comparison_is_removed(self):
        status, delta = cmp.classify(500.0, None)
        self.assertEqual(status, "removed")
        self.assertEqual(delta["value"], -500.0)

    def test_true_zero_is_distinct_from_unavailable(self):
        status, delta = cmp.classify(0.0, 0.0)
        self.assertEqual(status, "zero")
        self.assertEqual(delta["value"], 0.0)
        self.assertEqual(delta["kind"], "measured")

    def test_large_regression_clears_both_thresholds(self):
        status, delta = cmp.classify(1_000_000.0, 2_000_000.0)   # +100%, +1ms
        self.assertEqual(status, "regressed")
        self.assertEqual(delta["kind"], "measured")

    def test_large_improvement_clears_both_thresholds(self):
        status, delta = cmp.classify(2_000_000.0, 1_000_000.0)
        self.assertEqual(status, "improved")

    def test_big_percent_but_tiny_absolute_time_is_unchanged(self):
        # +200% but only 200ns absolute -- fails the absolute-time floor.
        status, delta = cmp.classify(100.0, 300.0)
        self.assertEqual(status, "unchanged")
        self.assertEqual(delta["kind"], "derived")
        self.assertIn("noise-floor", delta["reason"])

    def test_big_absolute_time_but_tiny_percent_is_unchanged(self):
        # +2% on a huge baseline -- large absolute delta_ns but fails the
        # percentage floor.
        base = 1_000_000_000.0
        status, delta = cmp.classify(base, base * 1.02)
        self.assertEqual(status, "unchanged")

    def test_zero_baseline_nonzero_comparison_has_no_percent_but_is_real(self):
        status, delta = cmp.classify(0.0, 5_000_000.0)
        self.assertEqual(status, "regressed")
        self.assertEqual(delta["value"], 5_000_000.0)

    def test_custom_noise_floor_thresholds_respected(self):
        status, _ = cmp.classify(1_000_000.0, 1_100_000.0, noise_pct=20.0, noise_ns=1.0)
        self.assertEqual(status, "unchanged")   # 10% < 20% floor


class TestCompareAggregates(unittest.TestCase):
    def test_ranks_by_absolute_impact_descending(self):
        trace_a = _trace([
            _span(Category.GPU_CUDA, "big_kernel", 10_000_000),
            _span(Category.GPU_CUDA, "small_kernel", 2_000_000),
        ])
        trace_b = _trace([
            _span(Category.GPU_CUDA, "big_kernel", 20_000_000),    # +10ms
            _span(Category.GPU_CUDA, "small_kernel", 2_500_000),   # +0.5ms
        ])
        rows = cmp.compare_aggregates(trace_a, trace_b)
        self.assertEqual(rows[0]["name"], "big_kernel")
        self.assertEqual(rows[0]["status"], "regressed")

    def test_matches_real_behavior_with_new_and_removed_rows(self):
        trace_a = _trace([_span(Category.GPU_CUDA, "old_fn", 5_000_000)])
        trace_b = _trace([_span(Category.GPU_CUDA, "new_fn", 5_000_000)])
        rows = cmp.compare_aggregates(trace_a, trace_b)
        statuses = {r["name"]: r["status"] for r in rows}
        self.assertEqual(statuses["old_fn"], "removed")
        self.assertEqual(statuses["new_fn"], "new")


class TestCompareBuckets(unittest.TestCase):
    def test_excludes_annotation_bucket(self):
        trace_a = _trace([_span(Category.NVTX, "range", 1_000_000, tags={"type": "nvtx_range"})])
        trace_b = _trace([_span(Category.NVTX, "range", 1_000_000, tags={"type": "nvtx_range"})])
        rows = cmp.compare_buckets(trace_a, trace_b)
        buckets = {r["bucket"] for r in rows}
        self.assertNotIn("Annotation", buckets)

    def test_computation_bucket_delta(self):
        trace_a = _trace([_span(Category.GPU_CUDA, "k", 1_000_000, tags={"type": "kernel"})])
        trace_b = _trace([_span(Category.GPU_CUDA, "k", 3_000_000, tags={"type": "kernel"})])
        rows = cmp.compare_buckets(trace_a, trace_b)
        comp_row = next(r for r in rows if r["bucket"] == "Computation")
        self.assertEqual(comp_row["status"], "regressed")
        self.assertEqual(comp_row["baseNs"], 1_000_000.0)
        self.assertEqual(comp_row["compNs"], 3_000_000.0)


class TestTopChanges(unittest.TestCase):
    def test_returns_only_matching_status_ranked_by_impact(self):
        rows = [
            {"name": "a", "status": "regressed", "deltaNs": 5_000_000},
            {"name": "b", "status": "improved", "deltaNs": -1_000_000},
            {"name": "c", "status": "regressed", "deltaNs": 20_000_000},
        ]
        top = cmp.top_changes(rows, status="regressed", limit=10)
        self.assertEqual([r["name"] for r in top], ["c", "a"])

    def test_respects_limit(self):
        rows = [{"name": str(i), "status": "regressed", "deltaNs": i} for i in range(20)]
        self.assertEqual(len(cmp.top_changes(rows, status="regressed", limit=5)), 5)


class TestNormalizedCoverage(unittest.TestCase):
    def test_length_matches_requested_buckets(self):
        trace = _trace([_span(Category.CPU, "f", 1000, start_ns=0)])
        cov = cmp.normalized_coverage(trace, n_buckets=20)
        self.assertEqual(len(cov), 20)

    def test_empty_trace_returns_all_zero(self):
        trace = _trace([])
        cov = cmp.normalized_coverage(trace, n_buckets=10)
        self.assertEqual(cov, [0.0] * 10)


class TestReportDict(unittest.TestCase):
    def test_shape(self):
        trace_a = _trace([_span(Category.GPU_CUDA, "k", 1_000_000)])
        trace_b = _trace([_span(Category.GPU_CUDA, "k", 5_000_000)])
        report = cmp.report_dict(trace_a, trace_b)
        for key in ("noiseFloor", "aggregates", "buckets", "topImprovements",
                    "topRegressions", "newCount", "removedCount"):
            self.assertIn(key, report)
        self.assertIn("not a statistical", report["noiseFloor"]["note"])


if __name__ == "__main__":
    unittest.main()
