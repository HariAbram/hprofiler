"""
Unit tests for src/analysis/activity_buckets.py -- confirms every `type` tag
value the C hooks actually emit (grepped from hooks/*/*.c's `type=...`
snprintf calls, not guessed) maps to a sensible bucket, that the established
type=="kernel" compute discriminator (already used elsewhere in this
codebase) lands in Computation, that Annotation spans are excluded from
bucket_totals(), and that a span with no `type` tag falls back to a
category-based bucket.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.trace import Trace, TraceMetadata
from src.core.events import SpanEvent, Category
from src.analysis import activity_buckets as ab


def _span(cat, dur_ns, span_type="", name="s", start_ns=0):
    tags = {"type": span_type} if span_type else {}
    return SpanEvent(name=name, category=cat, start_ns=start_ns, duration_ns=dur_ns,
                      pid=1, tid=1, tags=tags)


class TestBucketOf(unittest.TestCase):
    def test_every_hook_emitted_type_maps_to_a_real_bucket(self):
        # Every type= value grepped from hooks/*/*.c -- must not fall
        # through to "Other" (that would mean this module forgot one).
        emitted_types = [
            "accumulate", "allgather", "alloc", "alloc_async", "alloc_managed",
            "alloc_pinned", "allreduce", "alltoall", "barrier", "broadcast",
            "cancel", "comm_destroy", "comm_init", "comm_init_all", "critical",
            "DtoH", "free", "free_async", "free_pinned", "get", "graph_launch",
            "group", "HtoD", "irecv", "isend", "jit_compile", "jit_kernel",
            "jit_load", "kernel", "memcpy", "memcpy_async", "nvtx_mark",
            "nvtx_range", "offload", "parallel", "parallel_region", "put",
            "read", "recv", "recv_init", "reduce", "reduce_scatter",
            "roctx_mark", "roctx_range", "send", "send_init", "single",
            "start", "startall", "svm_memcpy", "sync", "task", "task_create",
            "test", "testall", "testany", "testsome", "wait", "waitall",
            "waitany", "waitsome", "win_fence", "win_flush", "win_flush_all",
            "win_lock", "win_lock_all", "win_unlock", "win_unlock_all",
            "work", "write",
        ]
        for t in emitted_types:
            bucket = ab.bucket_of("cuda", t)
            self.assertNotEqual(bucket, "Other", f"type={t} fell through to Other")
            self.assertIn(bucket, ab.BUCKETS)

    def test_kernel_is_computation_matching_established_discriminator(self):
        # type=="kernel" is already the compute-vs-overhead split used in
        # bridge.py/output/summary.py/analysis/context.py -- must agree.
        self.assertEqual(ab.bucket_of("cuda", "kernel"), "Computation")
        self.assertEqual(ab.bucket_of("rocm", "kernel"), "Computation")
        self.assertEqual(ab.bucket_of("opencl", "kernel"), "Computation")

    def test_gpu_api_overhead_types_are_not_computation(self):
        for t in ("alloc", "free", "graph_launch", "jit_compile", "jit_load"):
            self.assertEqual(ab.bucket_of("cuda", t), "Runtime overhead")

    def test_memory_transfer_types(self):
        for t in ("DtoH", "HtoD", "memcpy", "memcpy_async", "svm_memcpy"):
            self.assertEqual(ab.bucket_of("cuda", t), "Memory transfer")

    def test_communication_and_synchronization_are_distinct(self):
        self.assertEqual(ab.bucket_of("mpi", "allreduce"), "Communication")
        self.assertEqual(ab.bucket_of("mpi", "send"), "Communication")
        self.assertEqual(ab.bucket_of("mpi", "wait"), "Synchronization")
        self.assertEqual(ab.bucket_of("mpi", "barrier"), "Synchronization")

    def test_annotation_types_are_their_own_bucket(self):
        for t in ("nvtx_mark", "nvtx_range", "roctx_mark", "roctx_range"):
            self.assertEqual(ab.bucket_of("nvtx", t), "Annotation")

    def test_no_type_tag_falls_back_to_category(self):
        self.assertEqual(ab.bucket_of("cpu", ""), "Computation")
        self.assertEqual(ab.bucket_of("mpi", ""), "Communication")
        self.assertEqual(ab.bucket_of("sync", ""), "Synchronization")
        self.assertEqual(ab.bucket_of("memory", ""), "Memory transfer")
        self.assertEqual(ab.bucket_of("jit", ""), "Runtime overhead")
        self.assertEqual(ab.bucket_of("nvtx", ""), "Annotation")
        self.assertEqual(ab.bucket_of("sched", ""), "Idle")

    def test_unknown_category_and_type_falls_back_to_other(self):
        self.assertEqual(ab.bucket_of("other", "something_unrecognized"), "Other")


class TestBucketOfSpan(unittest.TestCase):
    def test_reads_type_tag_off_a_real_span(self):
        s = _span(Category.GPU_CUDA, 1000, span_type="kernel")
        self.assertEqual(ab.bucket_of_span(s), "Computation")

    def test_missing_type_tag_uses_category(self):
        s = _span(Category.MPI, 1000)
        self.assertEqual(ab.bucket_of_span(s), "Communication")


class TestBucketTotals(unittest.TestCase):
    def test_sums_duration_per_bucket(self):
        spans = [
            _span(Category.GPU_CUDA, 100, "kernel"),
            _span(Category.GPU_CUDA, 50, "kernel"),
            _span(Category.MPI, 30, "allreduce"),
            _span(Category.SYNC, 10, "barrier"),
        ]
        totals = ab.bucket_totals(spans)
        self.assertEqual(totals["Computation"], 150)
        self.assertEqual(totals["Communication"], 30)
        self.assertEqual(totals["Synchronization"], 10)

    def test_annotation_spans_excluded_from_totals(self):
        spans = [
            _span(Category.GPU_CUDA, 100, "kernel"),
            # An nvtx range wrapping (overlapping) that same kernel --
            # must not add its own 500ns on top, that would double-count.
            _span(Category.NVTX, 500, "nvtx_range"),
        ]
        totals = ab.bucket_totals(spans)
        self.assertEqual(totals["Computation"], 100)
        self.assertNotIn("Annotation", totals)

    def test_zero_duration_spans_excluded(self):
        spans = [_span(Category.GPU_CUDA, 0, "kernel")]
        self.assertEqual(ab.bucket_totals(spans), {})

    def test_idle_ns_added_to_idle_bucket(self):
        spans = [_span(Category.GPU_CUDA, 100, "kernel")]
        totals = ab.bucket_totals(spans, idle_ns=250)
        self.assertEqual(totals["Idle"], 250)
        self.assertEqual(totals["Computation"], 100)

    def test_empty_input(self):
        self.assertEqual(ab.bucket_totals([]), {})


if __name__ == "__main__":
    unittest.main()
