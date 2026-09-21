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
        matches = [(p, conf) for p, kind, conf in preds.get(wait_idx, []) if kind == "explicit_span_id"]
        self.assertEqual([p for p, _ in matches], [0])
        self.assertEqual(matches[0][1], "certain")


class TestMPIPointToPoint(unittest.TestCase):
    def test_recv_depends_on_matching_send(self):
        send = _span(0, 1, Category.MPI, 0, 10, name="MPI_Send",
                      tags={"type": "send", "rank": "0", "peer": "1", "tag": "7"})
        recv = _span(1, 1, Category.MPI, 5, 10, name="MPI_Recv",
                      tags={"type": "recv", "rank": "1", "peer": "0", "tag": "7"})
        trace = _mk_trace([send, recv])
        spans_list, preds = cp.build_dependency_graph(trace)
        recv_idx = 1
        matches = [(p, conf) for p, kind, conf in preds.get(recv_idx, []) if kind == "p2p"]
        self.assertEqual([p for p, _ in matches], [0])
        # Exact (non-wildcard) tag match, no commid= in this synthetic
        # trace -- same evidence class this module always had.
        self.assertEqual(matches[0][1], "medium")

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
        pred_ids = [p for p, kind, _conf in preds.get(recv_idx, []) if kind == "p2p"]
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
        arrival_edges = [(p, conf) for p, kind, conf in preds.get(r1_idx, []) if kind == "arrival"]
        self.assertEqual({p for p, _ in arrival_edges}, {2})  # r3's index only
        # No commid= in this synthetic trace -- falls back to per-type-only
        # clustering, same evidence class this module always had.
        self.assertEqual(arrival_edges[0][1], "medium")

        # r3 (the last arriver) itself has no arrival-predecessor from this
        # cluster -- nothing here explains why it started when it did.
        self.assertEqual([kind for _, kind, _conf in preds.get(2, []) if kind == "arrival"], [])

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
        self.assertTrue(any(kind == "arrival" for _, kind, _conf in preds.get(0, [])))
        self.assertFalse(any(kind == "arrival" for _, kind, _conf in preds.get(1, [])))
        # OpenMP barrier rendezvous is hardware/runtime-enforced -- "certain",
        # not "medium" the way MPI/NCCL rendezvous can be without a commid=.
        self.assertEqual(
            [conf for _, kind, conf in preds.get(0, []) if kind == "arrival"], ["certain"]
        )


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


class TestWildcardResolution(unittest.TestCase):
    """mpi_hook.c now resolves MPI_ANY_SOURCE/MPI_ANY_TAG via the real
    MPI_Status on every completion path (see its file header) -- these
    tests check criticalpath.py actually uses that resolved data instead
    of a sentinel/unmatchable value, and marks it 'high' confidence."""

    def test_blocking_wildcard_recv_uses_resolved_peer_tag(self):
        # mpi_hook.c already writes the *resolved* peer/tag directly into
        # the MPI_Recv span's own tags (not the input ANY_SOURCE/ANY_TAG
        # sentinel) -- wildcard=1 just flags that this is what happened.
        recv = _span(1, 1, Category.MPI, 100, 50, name="MPI_Recv",
                     tags={"type": "recv", "rank": "1", "peer": "0", "tag": "9", "wildcard": "1"})
        send = _span(0, 1, Category.MPI, 90, 5, name="MPI_Send",
                     tags={"type": "send", "rank": "0", "peer": "1", "tag": "9"})
        trace = _mk_trace([recv, send])
        spans_list, preds = cp.build_dependency_graph(trace)
        recv_idx = 0
        edges = [(p, conf) for p, kind, conf in preds.get(recv_idx, []) if kind == "p2p"]
        self.assertEqual([p for p, _ in edges], [1])  # send is spans_list[1]
        self.assertEqual(edges[0][1], "high")  # resolved wildcard match

    def test_async_wildcard_irecv_resolved_via_wait_rpeer_rtag(self):
        """The key new capability: a non-blocking MPI_Irecv posted with
        MPI_ANY_SOURCE/MPI_ANY_TAG has no known peer/tag on its own span --
        the resolution only appears on the MPI_Wait that later completes it
        (rpeer=/rtag=). Before this phase, non-blocking Isend/Irecv pairs
        got NO cross-rank edge at all (only the same-rank Irecv->Wait
        explicit_span_id link existed, which says nothing about when the
        remote sender's data arrived)."""
        isend = _span(0, 1, Category.MPI, 0, 10, name="MPI_Isend",
                      tags={"type": "isend", "rank": "0", "peer": "1", "tag": "5"})
        irecv = _span(1, 1, Category.MPI, 0, 3, name="MPI_Irecv", span_id="r2",
                      tags={"type": "irecv", "rank": "1", "peer": "-2", "tag": "-1", "wildcard": "1"})
        wait = _span(1, 1, Category.MPI, 3, 200, name="MPI_Wait", parent_span_id="r2",
                     tags={"rpeer": "0", "rtag": "5"})
        trace = _mk_trace([isend, irecv, wait])
        spans_list, preds = cp.build_dependency_graph(trace)
        wait_idx = 2
        p2p_edges = [(p, conf) for p, kind, conf in preds.get(wait_idx, []) if kind == "p2p"]
        self.assertEqual([p for p, _ in p2p_edges], [0])  # isend is spans_list[0]
        self.assertEqual(p2p_edges[0][1], "high")

    def test_async_exact_tag_irecv_wait_still_linked(self):
        """Non-wildcard non-blocking pair: no rpeer=/rtag= needed since the
        Irecv's own (non-wildcard) tags are already exact -- still must
        route the cross-rank edge to the Wait (which actually blocks), not
        the Irecv (which returns almost instantly)."""
        isend = _span(0, 1, Category.MPI, 0, 10, name="MPI_Isend",
                      tags={"type": "isend", "rank": "0", "peer": "1", "tag": "5"})
        irecv = _span(1, 1, Category.MPI, 0, 3, name="MPI_Irecv", span_id="r1",
                      tags={"type": "irecv", "rank": "1", "peer": "0", "tag": "5"})
        wait = _span(1, 1, Category.MPI, 3, 200, name="MPI_Wait", parent_span_id="r1")
        trace = _mk_trace([isend, irecv, wait])
        spans_list, preds = cp.build_dependency_graph(trace)
        wait_idx = 2
        p2p_edges = [(p, conf) for p, kind, conf in preds.get(wait_idx, []) if kind == "p2p"]
        self.assertEqual([p for p, _ in p2p_edges], [0])
        self.assertEqual(p2p_edges[0][1], "medium")  # exact tag, no wildcard resolution needed


class TestWaitallMultiRequestLinking(unittest.TestCase):
    def test_semicolon_separated_psid_links_each_request(self):
        """Before this phase, MPI_Waitall/MPI_Waitsome's psid='4;6' (multiple
        requests completing in one call) never matched anything: the
        generic explicit_span_id lookup tried an exact-string match against
        span_id, and no span has span_id '4;6' -- only '4' and '6'
        separately do. Waitall/Waitsome were silently unlinked to their own
        originating Isend/Irecv calls."""
        isend_a = _span(1, 1, Category.MPI, 0, 5, name="MPI_Isend", span_id="a")
        isend_b = _span(1, 1, Category.MPI, 5, 5, name="MPI_Isend", span_id="b")
        waitall = _span(1, 1, Category.MPI, 10, 100, name="MPI_Waitall", parent_span_id="a;b")
        trace = _mk_trace([isend_a, isend_b, waitall])
        spans_list, preds = cp.build_dependency_graph(trace)
        waitall_idx = 2
        edges = {(p, conf) for p, kind, conf in preds.get(waitall_idx, []) if kind == "explicit_span_id"}
        self.assertEqual(edges, {(0, "certain"), (1, "certain")})


class TestCommidScopedRendezvous(unittest.TestCase):
    def test_different_communicators_not_falsely_clustered(self):
        """Real bug this closes: two UNRELATED communicators each running an
        allreduce at overlapping wall-clock times used to be clustered into
        ONE rendezvous group (_cluster_rendezvous only looks at time
        overlap), creating a bogus arrival edge between completely
        unrelated ranks/communicators. With commid= now available, spans
        are bucketed by (type, commid) before clustering, so overlapping-
        but-different communicators stay separate."""
        # comm id=1: A and B overlap (A:[0,100), B:[10,90)) -> B is the last arriver.
        a = _span(1, 1, Category.MPI, 0, 100, name="MPI_Allreduce",
                  tags={"type": "allreduce", "commid": "1"})
        b = _span(2, 1, Category.MPI, 10, 80, name="MPI_Allreduce",
                  tags={"type": "allreduce", "commid": "1"})
        # comm id=2: C and D overlap (C:[5,95), D:[50,150)) -> D is the last arriver.
        # C/D's wall-clock range also heavily overlaps A/B's, which is
        # exactly what would have caused a false 4-way cluster pre-commid.
        c = _span(3, 1, Category.MPI, 5, 90, name="MPI_Allreduce",
                  tags={"type": "allreduce", "commid": "2"})
        d = _span(4, 1, Category.MPI, 50, 100, name="MPI_Allreduce",
                  tags={"type": "allreduce", "commid": "2"})
        trace = _mk_trace([a, b, c, d])
        spans_list, preds = cp.build_dependency_graph(trace)
        a_idx, b_idx, c_idx, d_idx = 0, 1, 2, 3

        a_arrivals = {(p, conf) for p, kind, conf in preds.get(a_idx, []) if kind == "arrival"}
        self.assertEqual(a_arrivals, {(b_idx, "high")})  # only B, never C or D

        c_arrivals = {(p, conf) for p, kind, conf in preds.get(c_idx, []) if kind == "arrival"}
        self.assertEqual(c_arrivals, {(d_idx, "high")})  # only D, never A or B

    def test_missing_commid_falls_back_to_medium_confidence(self):
        r1 = _span(1, 1, Category.MPI, 0, 100, name="MPI_Allreduce", tags={"type": "allreduce"})
        r2 = _span(2, 1, Category.MPI, 10, 80, name="MPI_Allreduce", tags={"type": "allreduce"})
        trace = _mk_trace([r1, r2])
        spans_list, preds = cp.build_dependency_graph(trace)
        arrivals = [(p, conf) for p, kind, conf in preds.get(0, []) if kind == "arrival"]
        self.assertEqual(arrivals, [(1, "medium")])


class TestFormalDPBeatsGreedy(unittest.TestCase):
    """Hand-verified case where the DP's globally-optimal choice differs
    from what the old greedy walk (picking the single locally-tightest-gate
    predecessor at each step) would have picked -- concretely demonstrating
    the DP is not just a refactor but a real correctness improvement for
    competing-predecessor scenarios. See criticalpath.py's module docstring
    for why greedy's local choice isn't guaranteed to reach the same answer
    as the proven-optimal DAG longest-path DP.

    Layout (see inline comments for the exact hand-computed numbers):
      Q  [0,500)   pid2/tid1, solo, dur=500
      P2 [500,550) pid2/tid1, same thread as Q -> program-order pred, dur=50
      P1 [590,600) pid3/tid1, solo, dur=10
      V  [600,650) pid1/tid1, parent_span_id="p1;p2" (explicit_span_id from BOTH)
    P1's gate (600) is *tighter* than P2's (550) -- greedy picks P1, total=60.
    But P2's own chain (Q+P2) accounts for 550ns before V even starts --
    the DP correctly picks P2, total=650 (Q+P2+the 50ns gap to V+V itself).
    """

    def _build(self):
        q = _span(2, 1, Category.CPU, 0, 500, name="Q")
        p2 = _span(2, 1, Category.CPU, 500, 50, name="P2", span_id="p2")
        p1 = _span(3, 1, Category.CPU, 590, 10, name="P1", span_id="p1")
        v = _span(1, 1, Category.CPU, 600, 50, name="V", parent_span_id="p1;p2")
        trace = _mk_trace([q, p2, p1, v])
        return cp.build_dependency_graph(trace)

    def test_dp_finds_the_globally_longer_explanatory_chain(self):
        spans_list, preds = self._build()
        path, confidences = cp.compute_critical_path_with_confidence(spans_list, preds)
        names = [spans_list[i].name for i in path]
        self.assertEqual(names, ["Q", "P2", "V"])
        report = cp.attribute_blame(spans_list, path, cp._wall_ns(spans_list), confidences)
        self.assertEqual(report.total_path_ns, 650)

    def test_old_greedy_walk_would_have_picked_the_worse_predecessor(self):
        """Confirms the premise: calling the retained (fallback-only) greedy
        implementation directly on the same graph reaches a different,
        objectively worse (shorter-accounted) answer -- proving the DP
        change in compute_critical_path is load-bearing, not cosmetic."""
        spans_list, preds = self._build()
        greedy_path, _greedy_conf = cp._compute_critical_path_greedy(spans_list, preds)
        names = [spans_list[i].name for i in greedy_path]
        self.assertEqual(names, ["P1", "V"])
        greedy_report = cp.attribute_blame(spans_list, greedy_path, cp._wall_ns(spans_list))
        self.assertEqual(greedy_report.total_path_ns, 60)
        self.assertLess(greedy_report.total_path_ns, 650)  # strictly worse than the DP's answer


class TestCycleFallback(unittest.TestCase):
    def test_artificial_cycle_falls_back_without_hanging(self):
        """The edge builders never produce a cycle by construction (every
        edge points from an earlier-enabling event to a later-gated one --
        see the module docstring), but the DP defensively verifies this via
        topological sort rather than assuming it, and must not hang or
        crash if one somehow exists -- falls back to the old greedy walk,
        which is cycle-safe by construction (its `visited` set)."""
        a = _span(1, 1, Category.CPU, 0, 10, name="A")
        b = _span(1, 1, Category.CPU, 20, 10, name="B")
        preds = {
            0: [(1, "sequential", "certain")],  # A depends on B
            1: [(0, "sequential", "certain")],  # B depends on A -- a direct 2-cycle
        }
        path, confidences = cp.compute_critical_path_with_confidence([a, b], preds)
        self.assertIn(len(path), (1, 2))  # terminates; exact shape doesn't matter, just must not hang


class TestConfidenceBreakdown(unittest.TestCase):
    def test_breakdown_sums_duration_by_edge_confidence_tier(self):
        a = _span(1, 1, Category.CPU, 0, 100, name="a")
        b = _span(1, 1, Category.CPU, 100, 50, name="b")
        c = _span(1, 1, Category.CPU, 150, 30, name="c")
        report = cp.CriticalPathReport(
            path_span_indices=[0, 1, 2],
            spans=[a, b, c],
            wall_ns=180,
            path_edge_confidence=["certain", "medium"],
        )
        breakdown = report.confidence_breakdown_ns()
        # b's duration (50) credited to the a->b edge's tier ("certain");
        # c's duration (30) credited to the b->c edge's tier ("medium").
        self.assertEqual(breakdown, {"certain": 50, "medium": 30})


class TestExecStartCalibration(unittest.TestCase):
    """cuda_hook.c/rocm_hook.c now emit xs=<ns> on GPU kernel spans: the
    real GPU-timeline execution-start time (from a reference-event
    calibration -- see their file comments), which can differ significantly
    from start_ns (the CPU-side launch-CALL time) under stream queue
    backlog. _effective_start_ns/_effective_end_ns prefer it when present;
    this is what the DP's gate/gap computation (_edge_gap_and_gate) uses."""

    def test_effective_start_prefers_xs_tag(self):
        s = _span(1, 1, Category.GPU_CUDA, 5, 200, tags={"xs": "1000"})
        self.assertEqual(cp._effective_start_ns(s), 1000)
        self.assertEqual(cp._effective_end_ns(s), 1200)  # xs + duration, not start_ns + duration

    def test_effective_start_falls_back_without_xs_tag(self):
        s = _span(1, 1, Category.GPU_CUDA, 5, 200)
        self.assertEqual(cp._effective_start_ns(s), 5)
        self.assertEqual(cp._effective_end_ns(s), 205)

    def test_effective_start_falls_back_on_malformed_xs_tag(self):
        s = _span(1, 1, Category.GPU_CUDA, 5, 200, tags={"xs": "not-a-number"})
        self.assertEqual(cp._effective_start_ns(s), 5)

    def test_xs_corrects_gap_attribution_under_queue_backlog(self):
        """Hand-verified scenario: two kernels launched back-to-back (CPU
        launch calls 5ns apart) on the same busy stream. k1 occupies the
        GPU until t=1000; k2's CPU launch call happened at t=5, but it
        can't actually start executing until k1 finishes at t=1000 --
        xs=1000 reflects that. A downstream span D (e.g. linked via
        explicit_span_id) starts at t=1250.

        Without xs=, the k2->D gap would be computed from k2's raw
        start_ns+duration (5+200=205) to D's start (1250): a bogus 1045ns
        "idle" gap, when k2 was actually still busy running until t=1200.
        With xs=1000, k2's effective end is 1000+200=1200, giving the
        correct, much smaller 50ns gap."""
        k1 = _span(1, 1, Category.GPU_CUDA, 0, 1000, name="k1", span_id="k1")
        k2 = _span(1, 1, Category.GPU_CUDA, 5, 200, name="k2", span_id="k2",
                  tags={"xs": "1000"})
        d = _span(1, 1, Category.SYNC, 1250, 10, name="D", parent_span_id="k2")
        trace = _mk_trace([k1, k2, d])
        spans_list, preds = cp.build_dependency_graph(trace)
        d_idx = 2
        gate, limit, gap = cp._edge_gap_and_gate(spans_list, 1, d_idx, "explicit_span_id")
        self.assertEqual(gate, 1200)  # k2's effective end (xs=1000 + duration=200), not 205
        self.assertEqual(gap, 50)     # 1250 - 1200, not the bogus 1045


if __name__ == "__main__":
    unittest.main()
