"""
OTLP timestamps: each event's absolute CLOCK_MONOTONIC timestamp is
converted with the monotonic->epoch offset directly. Adding the trace's
start_time_ns as well would double-count an absolute monotonic reading and
shift every timestamp by the machine's uptime (hours to months on a
persistent HPC node).
"""
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.trace import Trace, TraceMetadata
from src.core.events import SpanEvent, CounterEvent, Category
from src.output import otlp


class TestOTLPTimeAlignment(unittest.TestCase):
    def test_span_epoch_time_matches_real_now_not_offset_by_uptime(self):
        # Simulate a machine that has been up a long time: trace start's
        # absolute CLOCK_MONOTONIC reading is large (e.g. ~30 days of
        # uptime). The bug's signature is an exported timestamp that is
        # off by roughly this amount.
        thirty_days_ns = 30 * 24 * 3600 * 1_000_000_000
        mono_now = time.monotonic_ns()
        trace_start_mono = mono_now - 1_000_000  # trace "started" 1ms ago
        # Pretend the machine has been up a long time by using a start
        # value far in the monotonic past -- but still consistent with the
        # real clock (this is what a real long-uptime machine's
        # CLOCK_MONOTONIC would look like relative to "now").
        assert thirty_days_ns > 0  # sanity, not actually used as an offset here

        meta = TraceMetadata(start_time_ns=trace_start_mono, end_time_ns=mono_now)
        trace = Trace(meta)
        # A span that started 500us after trace start (still in absolute
        # CLOCK_MONOTONIC terms, per SpanEvent.start_ns's real semantics).
        span_start_mono = trace_start_mono + 500_000
        trace.add(SpanEvent(name="k", category=Category.GPU_CUDA,
                             start_ns=span_start_mono, duration_ns=100_000))

        real_epoch_now = time.time_ns()
        payload = otlp.build_traces_payload(trace)

        spans = payload["resourceSpans"][0]["scopeSpans"][0]["spans"]
        exported_start_ns = int(spans[0]["startTimeUnixNano"])

        # The exported epoch timestamp must be close to real wall-clock
        # "now" (within a couple seconds of test-execution slop) -- NOT
        # off by the trace's own monotonic start reading, which in a
        # buggy version would be many hours/days off on a long-uptime
        # machine, and even here (trace started ~1ms before "now") would
        # incorrectly double that ~1ms gap.
        self.assertLess(abs(exported_start_ns - real_epoch_now), 3_000_000_000)

    def test_metrics_epoch_time_matches_real_now(self):
        mono_now = time.monotonic_ns()
        meta = TraceMetadata(start_time_ns=mono_now - 1_000_000, end_time_ns=mono_now)
        trace = Trace(meta)
        trace.add(CounterEvent(name="ipc", category=Category.CPU,
                                timestamp_ns=mono_now, value=1.5))

        real_epoch_now = time.time_ns()
        payload = otlp.build_metrics_payload(trace)

        dp = payload["resourceMetrics"][0]["scopeMetrics"][0]["metrics"][0]["gauge"]["dataPoints"][0]
        exported_ts = int(dp["timeUnixNano"])
        self.assertLess(abs(exported_ts - real_epoch_now), 3_000_000_000)


if __name__ == "__main__":
    unittest.main()
