"""
Tests for TimelineWidget readability/consistency improvements
(src/ui/app.py) made in response to user feedback that the Timeline
"doesn't look pretty": deterministic per-function color assignment
(instead of thread-scheduling-order-dependent), MPI lanes labeled by
actual rank instead of a generic sequential thread number, and a quieter
idle-column representation. Connector-specific behavior (hover-gating,
elbow routing) is covered separately in test_timeline_connectors.py.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.trace import Trace, TraceMetadata
from src.core.events import SpanEvent, Category
from src.ui.app import TimelineWidget


def _span(pid, tid, cat, start_ns, dur_ns, name, tags=None):
    return SpanEvent(name=name, category=cat, start_ns=start_ns, duration_ns=dur_ns,
                     pid=pid, tid=tid, tags=dict(tags or {}))


def _mk_trace(spans):
    t = Trace(TraceMetadata())
    for s in spans:
        t.add(s)
    return t


class TestDeterministicColorAssignment(unittest.TestCase):
    def test_same_function_name_gets_same_color_regardless_of_insertion_order(self):
        # Simulates the real-world case this fixes: the same program's
        # threads can be scheduled in a different order from one run to
        # the next, so encounter-order color assignment gave the same
        # kernel a different color across runs -- bad for building muscle
        # memory across repeated profiling sessions.
        names_order_a = ["kernel_a", "kernel_b", "kernel_c"]
        names_order_b = ["kernel_c", "kernel_a", "kernel_b"]
        w1 = TimelineWidget(_mk_trace(
            [_span(1, 1, Category.GPU_CUDA, i, 10, n) for i, n in enumerate(names_order_a)]))
        w2 = TimelineWidget(_mk_trace(
            [_span(1, 1, Category.GPU_CUDA, i, 10, n) for i, n in enumerate(names_order_b)]))
        for name in names_order_a:
            self.assertEqual(w1._func_colors[name], w2._func_colors[name],
                            f"{name} got a different color depending on insertion order")

    def test_distinct_functions_within_palette_size_get_distinct_colors(self):
        # Open-addressing collision resolution must still guarantee this
        # for small function counts, matching the old encounter-order
        # scheme's guarantee -- a hash alone (without probing) could
        # collide even with very few distinct names.
        names = [f"func_{i}" for i in range(16)]  # == len(_SPAN_PALETTE)
        w = TimelineWidget(_mk_trace(
            [_span(1, 1, Category.GPU_CUDA, i, 10, n) for i, n in enumerate(names)]))
        colors = [w._func_colors[n] for n in names]
        self.assertEqual(len(set(colors)), 16, "16 distinct functions must get 16 distinct colors")

    def test_uses_stable_hash_not_randomized_builtin_hash(self):
        # zlib.crc32 (not Python's str hash(), which is randomly salted
        # per-process by default) -- verified indirectly: running the
        # SAME assignment logic twice in THIS process must agree (a
        # necessary but not sufficient check for determinism; the
        # insertion-order test above is the stronger cross-order check).
        w1 = TimelineWidget(_mk_trace([_span(1, 1, Category.MPI, 0, 10, "MPI_Allreduce")]))
        w2 = TimelineWidget(_mk_trace([_span(1, 1, Category.MPI, 0, 10, "MPI_Allreduce")]))
        self.assertEqual(w1._func_colors["MPI_Allreduce"], w2._func_colors["MPI_Allreduce"])


class TestMpiRankLabels(unittest.TestCase):
    def test_mpi_lane_shows_actual_rank_not_sequential_thread_number(self):
        a = _span(100, 101, Category.MPI, 0, 10, "MPI_Send", tags={"rank": "5"})
        w = TimelineWidget(_mk_trace([a]))
        label = w._lane_label("mpi/thread-101")
        self.assertIn("rank5", label)
        self.assertNotIn("T1", label)

    def test_non_mpi_lane_unaffected_still_shows_thread_sequence(self):
        a = _span(100, 101, Category.GPU_CUDA, 0, 10, "kernel")
        w = TimelineWidget(_mk_trace([a]))
        label = w._lane_label("cuda/thread-101")
        self.assertIn("T1", label)

    def test_mpi_lane_with_no_rank_tag_falls_back_to_thread_sequence(self):
        # Defensive fallback -- every real mpi_hook.c span does carry
        # rank=, but nothing should crash if one somehow didn't.
        a = _span(100, 101, Category.MPI, 0, 10, "some_mpi_span")
        w = TimelineWidget(_mk_trace([a]))
        label = w._lane_label("mpi/thread-101")
        self.assertIn("T1", label)

    def test_different_mpi_lanes_show_their_own_distinct_ranks(self):
        a = _span(100, 101, Category.MPI, 0, 10, "MPI_Send", tags={"rank": "0"})
        b = _span(200, 201, Category.MPI, 0, 10, "MPI_Recv", tags={"rank": "7"})
        w = TimelineWidget(_mk_trace([a, b]))
        self.assertIn("rank0", w._lane_label("mpi/thread-101"))
        self.assertIn("rank7", w._lane_label("mpi/thread-201"))


class TestIdleRendering(unittest.TestCase):
    def test_idle_columns_are_blank_not_a_visible_dot(self):
        # A single short span leaves most of a wide view idle -- idle
        # columns must render as plain unstyled space, not the previous
        # visible "·" dot (which read as noise/static on sparse traces).
        a = _span(1, 101, Category.CPU, 0, 10, "brief_work")
        w = TimelineWidget(_mk_trace([a]))
        chars, styles, _util = w._density_row("cpu/thread-101", width=50, _unused="")
        # The span itself is tiny relative to the trace window (only view
        # start/end are seeded from this one span, so it may fill more
        # than expected) -- the real invariant is just that wherever it
        # IS idle, the character is blank, not a dot.
        for ch, st in zip(chars, styles):
            self.assertIn(ch, ("█", " "))
            if ch == " ":
                self.assertEqual(st, "")


if __name__ == "__main__":
    unittest.main()
