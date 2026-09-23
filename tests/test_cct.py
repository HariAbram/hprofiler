"""
Regression tests for src/analysis/cct.py's gpu_starvation():
1. wall_ns must be derived from span timestamps, not trace.duration_ns
   (meaningless for a trace loaded from JSON).
2. launch_gap_ns must be computed from the UNION of kernel-active and
   sync-wait intervals, not gpu_active_ns + sync_ns added separately --
   overlap between a sync call and the kernel it waits on was previously
   double-counted, under-reporting (or, in extreme cases, zeroing) a real
   launch gap elsewhere in the trace.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.trace import Trace, TraceMetadata
from src.core.events import SpanEvent, Category
from src.analysis import cct


def _mk_trace(spans):
    t = Trace(TraceMetadata())
    for s in spans:
        t.add(s)
    return t


def _span(cat, start, dur, name="k", tags=None):
    return SpanEvent(name=name, category=cat, start_ns=start, duration_ns=dur, tags=dict(tags or {}))


class TestGPUStarvation(unittest.TestCase):
    def test_overlapping_sync_does_not_hide_a_real_gap(self):
        # kernel [0,100), sync [90,110) (overlaps the kernel's tail by 10ns),
        # then a genuine 90ns gap [110,200), then another kernel [200,250).
        spans = [
            _span(Category.GPU_CUDA, 0, 100, name="k1", tags={"type": "kernel"}),
            _span(Category.SYNC, 90, 20, name="cudaStreamSynchronize"),
            _span(Category.GPU_CUDA, 200, 50, name="k2", tags={"type": "kernel"}),
        ]
        trace = _mk_trace(spans)
        sv = cct.gpu_starvation(trace)

        self.assertEqual(sv["wall_ns"], 250)
        # union([0,100),[90,110)) = 110; plus [200,250) = 50 -> busy=160
        # -> real gap = 250-160 = 90 (the actual uncovered [110,200) window)
        self.assertEqual(sv["launch_gap_ns"], 90)

    def test_wall_ns_from_spans_not_trace_duration(self):
        spans = [_span(Category.GPU_CUDA, 1000, 500, tags={"type": "kernel"})]
        trace = _mk_trace(spans)
        # TraceMetadata.start_time_ns defaults to time.monotonic_ns() at
        # construction -- a real, large value -- while our one span sits
        # at start_ns=1000. A wall_ns derived from trace.duration_ns would
        # be wildly different from 500 (the span's own duration).
        sv = cct.gpu_starvation(trace)
        self.assertEqual(sv["wall_ns"], 500)

    def test_opencl_cpu_side_span_not_double_counted_as_kernel_active(self):
        # opencl_hook.c emits TWO spans per kernel launch: a CPU-side
        # enqueue-latency span (type=kernel,side=cpu) and a GPU-side
        # execution span (type=kernel,side=gpu) -- both carry
        # type=="kernel", distinguished only by `side`. An `or` here
        # (instead of `and`) would count the CPU-side span's own interval
        # as ADDITIONAL kernel-active time even where it doesn't overlap
        # the real GPU-side span, inflating gpu_active_ns/pct.
        # CPU-side: enqueue latency [0,10). GPU-side (the real execution):
        # [50,150). They don't overlap -- if the CPU-side span were wrongly
        # counted, gpu_active_ns would include [0,10) too (110 total
        # instead of the correct 100).
        spans = [
            _span(Category.GPU_OPENCL, 0, 10, name="k", tags={"type": "kernel", "side": "cpu"}),
            _span(Category.GPU_OPENCL, 50, 100, name="k", tags={"type": "kernel", "side": "gpu"}),
        ]
        trace = _mk_trace(spans)
        sv = cct.gpu_starvation(trace)
        self.assertEqual(sv["wall_ns"], 150)
        self.assertEqual(sv["gpu_active_ns"], 100)

    def test_opencl_memory_transfer_not_counted_as_kernel_active(self):
        # A "memory"-category span (e.g. a buffer read/write, correctly
        # NOT in _GPU_CATS) must never contribute to kernel_intervals
        # regardless of its tags.
        spans = [
            _span(Category.MEMORY, 0, 100, name="clEnqueueReadBuffer",
                  tags={"type": "read", "bytes": "1024"}),
        ]
        trace = _mk_trace(spans)
        sv = cct.gpu_starvation(trace)
        self.assertEqual(sv["gpu_active_ns"], 0)


if __name__ == "__main__":
    unittest.main()
