"""
Unit tests for src/analysis/criticalpath.py -- verifies the dependency-graph
edge builders and the backward critical-path walk against hand-computed
expected answers on small synthetic traces, including a 2-backend (MPI+CUDA)
scenario modeled on the classic CASITA-style case this generalizes.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.trace import Trace, TraceMetadata
from src.core.events import SpanEvent, Category
from src.analysis import criticalpath as cp


def _mk_trace(spans):
    t = Trace(TraceMetadata())
    for s in spans:
        t.add(s)
    return t


def _span(pid, tid, cat, start_ns, dur_ns, name="s", tags=None, span_id="", parent_span_id=""):
    return SpanEvent(name=name, category=cat, start_ns=start_ns, duration_ns=dur_ns,
                      pid=pid, tid=tid, tags=dict(tags or {}),
                      span_id=span_id, parent_span_id=parent_span_id)


class TestProgramOrderEdges(unittest.TestCase):
    def test_three_sequential_spans_no_gaps(self):
        spans = [
            _span(1, 1, Category.CPU, 0, 100, name="a"),
            _span(1, 1, Category.CPU, 100, 100, name="b"),
            _span(1, 1, Category.CPU, 200, 100, name="c"),
        ]
        trace = _mk_trace(spans)
        report = cp.analyze(trace)
        names = [report.spans[i].name for i in report.path_span_indices]
        self.assertEqual(names, ["a", "b", "c"])
        self.assertEqual(report.total_path_ns, 300)  # no idle gaps
        self.assertEqual(report.wait_caused_by_category, {})

    def test_gap_between_spans_attributed_to_earlier_spans_category(self):
        spans = [
            _span(1, 1, Category.MPI, 0, 100, name="a", tags={"type": "barrier"}),
            _span(1, 1, Category.CPU, 300, 100, name="b"),  # 200ns idle gap after a
        ]
        trace = _mk_trace(spans)
        report = cp.analyze(trace)
        self.assertEqual(report.wait_caused_by_category.get("mpi"), 200)


class TestStreamOrderAndDeviceSync(unittest.TestCase):
    def test_device_sync_gates_on_latest_finishing_stream(self):
        # Two CUDA streams launch concurrently; device sync must wait on
        # whichever finishes LAST (stream 2, ending at 500).
        spans = [
            _span(1, 1, Category.GPU_CUDA, 0, 200, name="k1", tags={"stream": "1"}),
            _span(1, 1, Category.GPU_CUDA, 0, 500, name="k2", tags={"stream": "2"}),
            _span(1, 1, Category.SYNC, 500, 10, name="cudaDeviceSynchronize"),
        ]
        trace = _mk_trace(spans)
        spans_list, preds = cp.build_dependency_graph(trace)
        sync_idx = next(i for i, s in enumerate(spans_list) if s.name == "cudaDeviceSynchronize")
        path = cp.compute_critical_path(spans_list, preds)
        # k2 (the later-finishing stream) must be the sync's blamed predecessor on the path
        self.assertIn(sync_idx, path)
        pos = path.index(sync_idx)
        self.assertGreater(pos, 0)
        self.assertEqual(spans_list[path[pos - 1]].name, "k2")


class TestExplicitSpanIdEdges(unittest.TestCase):
    def test_wait_depends_on_matching_isend(self):
        spans = [
            _span(1, 1, Category.MPI, 0, 50, name="MPI_Isend", span_id="sid1"),
            _span(1, 1, Category.MPI, 100, 20, name="MPI_Wait", parent_span_id="sid1"),
        ]
        trace = _mk_trace(spans)
        spans_list, preds = cp.build_dependency_graph(trace)
        wait_idx = 1
        pred_ids = [p for p, kind in preds.get(wait_idx, []) if kind == "explicit_span_id"]
        self.assertEqual(pred_ids, [0])


class TestMPIPointToPoint(unittest.TestCase):
    def test_recv_depends_on_matching_send(self):
        send = _span(0, 1, Category.MPI, 0, 10, name="MPI_Send",
                      tags={"type": "send", "rank": "0", "peer": "1", "tag": "7"})
        recv = _span(1, 1, Category.MPI, 5, 10, name="MPI_Recv",
                      tags={"type": "recv", "rank": "1", "peer": "0", "tag": "7"})
        trace = _mk_trace([send, recv])
        spans_list, preds = cp.build_dependency_graph(trace)
        recv_idx = 1
        pred_ids = [p for p, kind in preds.get(recv_idx, []) if kind == "p2p"]
        self.assertEqual(pred_ids, [0])

    def test_early_posted_receive_still_matches(self):
        """Regression test for a self-audit bug: the receiver commonly
        posts MPI_Recv well before the matching MPI_Send even starts (an
        intentional HPC pattern -- 'post an early receive to overlap with
        compute'), so the recv's span can legitimately start before the
        send's. The old matching required send.start_ns <= recv.start_ns
        as a precondition, which silently found NO edge at all in this
        (common) case instead of the real send->recv dependency."""
        recv = _span(1, 1, Category.MPI, 0, 600, name="MPI_Recv",
                     tags={"type": "recv", "rank": "1", "peer": "0", "tag": "3"})
        send = _span(0, 1, Category.MPI, 500, 10, name="MPI_Send",
                     tags={"type": "send", "rank": "0", "peer": "1", "tag": "3"})
        trace = _mk_trace([recv, send])
        spans_list, preds = cp.build_dependency_graph(trace)
        recv_idx = 0
        pred_ids = [p for p, kind in preds.get(recv_idx, []) if kind == "p2p"]
        self.assertEqual(pred_ids, [1])  # send is spans_list[1]

        # And the backward walk must actually be able to use this edge: the
        # send started (500) well before the recv ended (600), so it's a
        # valid (start-gated) predecessor even though it started AFTER the
        # recv itself started.
        path = cp.compute_critical_path(spans_list, {recv_idx: preds[recv_idx]})
        self.assertEqual(path, [1, 0])  # send, then recv


class TestRendezvousClustering(unittest.TestCase):
    def test_last_arriver_is_the_chosen_predecessor(self):
        # 3 ranks call allreduce; rank 3 arrives latest (start=300).
        # A downstream span on rank 1 that depends on the whole rendezvous
        # should trace back through rank 3's call, not rank 1's own (earlier)
        # arrival or rank 2's.
        r1 = _span(1, 1, Category.MPI, 0, 400, name="MPI_Allreduce", tags={"type": "allreduce"})
        r2 = _span(2, 1, Category.MPI, 100, 250, name="MPI_Allreduce", tags={"type": "allreduce"})
        r3 = _span(3, 1, Category.MPI, 300, 50, name="MPI_Allreduce", tags={"type": "allreduce"})
        trace = _mk_trace([r1, r2, r3])
        spans_list, preds = cp.build_dependency_graph(trace)
        # r1's only "arrival" predecessor should be r3 -- the single last
        # arriver -- not a full mutual clique with r2 as well (see
        # _add_last_arriver_edges docstring for why a clique is wrong here).
        r1_idx = 0
        arrival_preds = {p for p, kind in preds.get(r1_idx, []) if kind == "arrival"}
        self.assertEqual(arrival_preds, {2})  # r3's index only

        # r3 (the last arriver) itself has no arrival-predecessor from this
        # cluster -- nothing here explains why it started when it did.
        self.assertEqual([kind for _, kind in preds.get(2, []) if kind == "arrival"], [])

        path = cp.compute_critical_path(spans_list, {r1_idx: preds[r1_idx]})
        self.assertEqual(path, [2, 0])  # r3 (idx2) then r1 (idx0)


class TestCausalityEnforcement(unittest.TestCase):
    """Regression test for a bug caught while validating this module against
    a real trace: the backward walk must never pick a predecessor whose
    gate time is *after* the point it's supposed to explain -- otherwise
    the resulting 'critical path' isn't chronologically ordered and its
    per-category time sums can wildly exceed wall-clock time."""

    def test_wide_rendezvous_cluster_stays_bounded_and_self_reports(self):
        """Regression test: caught on a real OpenMP trace where a full
        mutual clique between rendezvous participants (pre-fix) let the
        backward walk chain through arrival edges repeatedly within one
        cluster, producing a category-time sum >1000x the actual wall-clock
        time (8.2s attributed inside a 9ms run). With edges restricted to
        "every member -> the single last arriver" (_add_last_arriver_edges)
        plus the current-node-relative causality check in
        compute_critical_path, a 5-thread rendezvous cluster must no longer
        blow up like that -- and if the span-level (not event-level)
        approximation still overshoots wall time at all, it must say so."""
        barrier_spans = [
            _span(1, tid, Category.SYNC, start, 1000 - start, name="omp_barrier_implicit")
            for tid, start in enumerate([0, 50, 300, 310, 900])
        ]
        tail = _span(1, 0, Category.OPENMP, 1000, 100, name="after_barrier")
        trace = _mk_trace(barrier_spans + [tail])

        report = cp.analyze(trace)
        # The old (mutual-clique) bug produced a >100x blowup on scenarios
        # like this; the fix keeps it within a small constant factor of
        # wall time (bounded by the overlap of at most a couple of spans,
        # not a chain through every cluster member).
        self.assertLess(report.total_path_ns, report.wall_ns * 3)
        if report.total_path_ns > report.wall_ns * 1.05:
            self.assertTrue(any("exceeds wall time" in n for n in report.notes))


class TestOMPBarrier(unittest.TestCase):
    def test_two_threads_barrier_clustered(self):
        # thread 2's barrier (start=20) arrives later than thread 1's
        # (start=0), so it's the "last arriver": thread 1's barrier depends
        # on it (arrival edge), but thread 2's barrier -- being the last
        # arriver -- gets no arrival-predecessor from this cluster itself.
        spans = [
            _span(1, 1, Category.SYNC, 0, 100, name="omp_barrier_implicit"),
            _span(1, 2, Category.SYNC, 20, 90, name="omp_barrier_implicit"),  # overlaps thread 1's
        ]
        trace = _mk_trace(spans)
        spans_list, preds = cp.build_dependency_graph(trace)
        self.assertTrue(any(kind == "arrival" for _, kind in preds.get(0, [])))
        self.assertFalse(any(kind == "arrival" for _, kind in preds.get(1, [])))


class TestTwoBackendKnownAnswer(unittest.TestCase):
    """Models the classic CASITA case: a CUDA kernel launch on rank 0,
    followed by an MPI Send that (via program order on rank 0's single
    thread) can't start until the kernel launch call returns, paired with a
    Recv on rank 1 that can't complete until the Send has happened. This is
    the 2-backend (MPI+CUDA) case this module generalizes to N-way; the
    critical path and blame breakdown here should match hand computation
    exactly before trusting the engine on larger N-way traces."""

    def test_known_path_and_blame(self):
        # rank 0, thread 1: cuda kernel launch [0,50), then MPI_Send [50,60)
        # rank 1, thread 1: MPI_Recv starts at 200 (rank 1 was busy until
        # then), takes 10ns, and is the last event in the trace.
        k = _span(0, 1, Category.GPU_CUDA, 0, 50, name="kernel")
        send = _span(0, 1, Category.MPI, 50, 10, name="MPI_Send",
                     tags={"type": "send", "rank": "0", "peer": "1", "tag": "1"})
        busy = _span(1, 1, Category.OPENMP, 0, 200, name="other_work")
        recv = _span(1, 1, Category.MPI, 200, 10, name="MPI_Recv",
                     tags={"type": "recv", "rank": "1", "peer": "0", "tag": "1"})
        trace = _mk_trace([k, send, busy, recv])
        report = cp.analyze(trace)

        names = [report.spans[i].name for i in report.path_span_indices]
        # recv ends last (210) -> critical path terminus is recv.
        # recv's predecessors: program-order (busy, since same thread) and
        # p2p (send). Program-order gate = busy.end_ns=200; p2p gate =
        # send.end_ns=60. busy's gate (200) is later -> busy is blamed, not
        # the MPI send/kernel chain -- this is the correct "who actually
        # gated the recv" answer: rank 1 was still busy until t=200, the
        # send had been ready since t=60.
        self.assertEqual(names[-1], "MPI_Recv")
        self.assertEqual(names[-2], "other_work")
        # No idle gap between busy ending at 200 and recv starting at 200.
        self.assertEqual(report.wait_caused_by_category, {})


class TestSerializationEfficiencyBridge(unittest.TestCase):
    def test_ratio_of_on_path_comm_time(self):
        a = _span(1, 1, Category.MPI, 0, 100, name="a", tags={"type": "barrier"})
        b = _span(1, 1, Category.MPI, 100, 50, name="b", tags={"type": "barrier"})
        trace = _mk_trace([a, b])
        report = cp.analyze(trace)  # both on path (program order, no gap)
        eff = cp.serialization_efficiency_from_path(report, [a, b])
        self.assertAlmostEqual(eff, 1.0)  # both comm spans are on the critical path

        c = _span(2, 1, Category.MPI, 0, 500, name="c", tags={"type": "barrier"})  # different pid, off path
        eff2 = cp.serialization_efficiency_from_path(report, [a, b, c])
        # on-path comm ns = 150 (a+b), total comm ns = 150+500=650
        self.assertAlmostEqual(eff2, 150 / 650, places=6)

    def test_works_across_separately_loaded_trace_instances(self):
        """Regression test for a self-audit bug: the bridge originally
        matched spans by raw object identity (id()), which silently returns
        wrong (near-zero) results if `comm_spans` comes from a DIFFERENT
        Trace object than the one the CriticalPathReport was built from --
        e.g. two separate load_trace_from_json() calls on the same file.
        _span_identity_key (value-based) must make this work correctly
        regardless of object identity."""
        a = _span(1, 1, Category.MPI, 0, 100, name="a", tags={"type": "barrier"})
        b = _span(1, 1, Category.MPI, 100, 50, name="b", tags={"type": "barrier"})
        trace = _mk_trace([a, b])
        report = cp.analyze(trace)

        # Simulate a second, independently-constructed Trace with logically
        # identical (but NOT object-identical) spans -- as if reloaded from
        # the same JSON file a second time.
        a2 = _span(1, 1, Category.MPI, 0, 100, name="a", tags={"type": "barrier"})
        b2 = _span(1, 1, Category.MPI, 100, 50, name="b", tags={"type": "barrier"})
        self.assertIsNot(a, a2)  # genuinely different objects

        eff = cp.serialization_efficiency_from_path(report, [a2, b2])
        self.assertAlmostEqual(eff, 1.0)  # must still match, not silently return ~0


class TestEmptyTrace(unittest.TestCase):
    def test_no_crash_and_correct_note(self):
        trace = _mk_trace([])
        report = cp.analyze(trace)
        self.assertEqual(report.path_span_indices, [])
        self.assertTrue(any("No spans" in n for n in report.notes))
        self.assertFalse(any("Only one process" in n for n in report.notes))


if __name__ == "__main__":
    unittest.main()
