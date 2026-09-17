"""
Regression test for the cross-hook stk: correlation race (see
audit-fixes-june-2026 memory / DOCUMENTATION.md §12): each LD_PRELOAD hook
opens its own socket connection, so span: and stk: records from *different*
hooks but the same OS tid can interleave. A single last-write-wins slot
keyed by (pid, tid) could then have the wrong span attached, or drop the
stack entirely, when a different hook's span for the same tid arrived
between the original span: and its stk: record.

_remember_recent_span / _find_recent_span (src/core/runner.py) replace that
single slot with a small ring matched by exact start_ns, closing this
window. This test drives them directly (not through real sockets) since
handle_client() itself is a private closure inside Runner.run().
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.runner import _remember_recent_span, _find_recent_span, RECENT_SPANS_PER_THREAD
from src.core.events import SpanEvent, Category


def _span(pid, tid, start_ns, name="k", cat=Category.GPU_CUDA, dur=100):
    return SpanEvent(name=name, category=cat, start_ns=start_ns, duration_ns=dur, pid=pid, tid=tid)


class TestRecentSpanRing(unittest.TestCase):
    def test_simple_match(self):
        recent = {}
        s = _span(1, 100, 1000)
        _remember_recent_span(recent, s)
        found = _find_recent_span(recent, 1, 100, 1000)
        self.assertIs(found, s)

    def test_no_match_returns_none(self):
        recent = {}
        _remember_recent_span(recent, _span(1, 100, 1000))
        self.assertIsNone(_find_recent_span(recent, 1, 100, 9999))
        self.assertIsNone(_find_recent_span(recent, 1, 200, 1000))  # wrong tid
        self.assertIsNone(_find_recent_span(recent, 2, 100, 1000))  # wrong pid

    def test_interleaved_hooks_same_tid_both_recoverable(self):
        """The exact race this fix targets: two different hook libraries
        (e.g. cuda_hook and ompt_tool) emit spans from the same OS tid in
        close succession, on two different socket connections. With a
        single last-write-wins slot, hookB's span would overwrite hookA's
        before hookA's stk: record arrives, silently dropping hookA's
        stack. With the ring, both are still findable by start_ns."""
        recent = {}
        span_a = _span(1, 100, 1000, name="cudaLaunchKernel", cat=Category.GPU_CUDA)
        span_b = _span(1, 100, 1050, name="omp_task", cat=Category.OPENMP)

        # Connection A: span: record for hook A arrives.
        _remember_recent_span(recent, span_a)
        # Connection B interleaves: span: record for hook B arrives before
        # hook A's stk: record does.
        _remember_recent_span(recent, span_b)
        # Now hook A's stk: record (for start_ns=1000) arrives.
        found_a = _find_recent_span(recent, 1, 100, 1000)
        # And hook B's stk: record (for start_ns=1050) arrives.
        found_b = _find_recent_span(recent, 1, 100, 1050)

        self.assertIs(found_a, span_a, "hook A's span must still be findable after hook B's span interleaved")
        self.assertIs(found_b, span_b)

    def test_ring_is_bounded(self):
        """More than RECENT_SPANS_PER_THREAD spans on one thread must evict
        the oldest, not grow unbounded (this dict lives for the whole
        profiling run)."""
        recent = {}
        spans = [_span(1, 100, i) for i in range(RECENT_SPANS_PER_THREAD + 3)]
        for s in spans:
            _remember_recent_span(recent, s)
        ring = recent[(1, 100)]
        self.assertEqual(len(ring), RECENT_SPANS_PER_THREAD)
        # oldest 3 evicted -> not findable any more
        self.assertIsNone(_find_recent_span(recent, 1, 100, spans[0].start_ns))
        self.assertIsNone(_find_recent_span(recent, 1, 100, spans[1].start_ns))
        # newest still there
        self.assertIs(_find_recent_span(recent, 1, 100, spans[-1].start_ns), spans[-1])

    def test_stale_duplicate_start_ns_prefers_most_recent(self):
        """If two spans on the same thread happen to share a start_ns
        (possible with coarse timers), matching walks newest-first, so a
        stk: record pairs with whichever span was emitted right before it
        (the common case) rather than an older namesake."""
        recent = {}
        old = _span(1, 100, 5000, name="old")
        new = _span(1, 100, 5000, name="new")
        _remember_recent_span(recent, old)
        _remember_recent_span(recent, new)
        found = _find_recent_span(recent, 1, 100, 5000)
        self.assertIs(found, new)


if __name__ == "__main__":
    unittest.main()
