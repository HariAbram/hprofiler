"""
MPI point-to-point matching (src/analysis/criticalpath.py _P2PMatcher) on
synthetic traces in the wire format hooks/mpi_hook/mpi_hook.c emits:
world-rank + communicator keys (sub-communicators with renumbered ranks,
concurrent duplicated communicators, MPI_COMM_WORLD), the legacy key for
traces without wpeer=, cancelled requests, send variants, MPI_Sendrecv
pairs, MPI_PROC_NULL, and the dependency-graph diagnostics. Every case is
checked with both the in-memory builder and the streamed builder.
"""
import io
import sys
import unittest
from contextlib import redirect_stderr
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.events import Category, InstantEvent, SpanEvent
from src.core.trace import Trace, TraceMetadata
from src.analysis import criticalpath as cp


def _trace(events):
    t = Trace(TraceMetadata())
    for e in events:
        t.add(e)
    return t


def _mpi(name, pid, start, dur, sid="", psid="", **tags):
    return SpanEvent(name=name, category=Category.MPI, start_ns=start, duration_ns=dur,
                     pid=pid, tid=pid, tags={k: str(v) for k, v in tags.items()},
                     span_id=sid, parent_span_id=psid)


def send(pid, start, rank, peer, tag, commid, wpeer=None, typ="send", name="MPI_Send", sid="", **extra):
    t = dict(type=typ, rank=rank, peer=peer, tag=tag, commid=commid, **extra)
    if wpeer is not None:
        t["wpeer"] = wpeer
    return _mpi(name, pid, start, 10, sid=sid, **t)


def recv(pid, start, rank, peer, tag, commid, wpeer=None, dur=50, **extra):
    t = dict(type="recv", rank=rank, peer=peer, tag=tag, commid=commid, **extra)
    if wpeer is not None:
        t["wpeer"] = wpeer
    return _mpi("MPI_Recv", pid, start, dur, **t)


def p2p_edges(trace):
    """{(src name, src pid, dst name, dst pid, tier)} from both builders,
    which must agree."""
    spans, preds = cp.build_dependency_graph(trace)
    mem = sorted((u, v, c) for v, pl in preds.items() for u, k, c in pl if k == "p2p")
    g = cp.Graph(trace, trace.store.span_seqs(), cp.build_edges(trace))
    streamed = sorted((u, v, c) for v, pl in g.preds_dict().items() for u, k, c in pl if k == "p2p")
    assert mem == streamed, (mem, streamed)
    return [(spans[u], spans[v], c) for u, v, c in mem]


class TestWorldKey(unittest.TestCase):
    def test_comm_split_with_renumbered_ranks_pairs_on_world_ranks(self):
        # World ranks 0..3, split into {0,1} (commid 11) and {2,3} (commid 22):
        # comm-local ranks 0/1 in both halves. Each half sends local 0 -> 1.
        ev = [
            send(0, 0, rank=0, peer=1, wpeer=1, tag=5, commid=11),
            send(2, 0, rank=2, peer=1, wpeer=3, tag=5, commid=22),
            recv(1, 0, rank=1, peer=0, wpeer=0, tag=5, commid=11),
            recv(3, 0, rank=3, peer=0, wpeer=2, tag=5, commid=22),
        ]
        edges = p2p_edges(_trace(ev))
        pairs = sorted((s.pid, d.pid) for s, d, _c in edges)
        self.assertEqual(pairs, [(0, 1), (2, 3)])

    def test_old_key_would_have_lost_the_second_half(self):
        # Same trace without wpeer=: the legacy key (sender world rank,
        # comm-local peer, tag) pairs the first half only -- the documented
        # behaviour of traces from older hooks, with a warning.
        ev = [
            send(0, 0, rank=0, peer=1, tag=5, commid=11),
            send(2, 0, rank=2, peer=1, tag=5, commid=22),
            recv(1, 0, rank=1, peer=0, tag=5, commid=11),
            recv(3, 0, rank=3, peer=0, tag=5, commid=22),
        ]
        cp._legacy_key_warned = False
        err = io.StringIO()
        with redirect_stderr(err):
            edges = p2p_edges(_trace(ev))
        self.assertEqual(sorted((s.pid, d.pid) for s, d, _c in edges), [(0, 1)])
        self.assertIn("no wpeer= tags", err.getvalue())
        self.assertEqual(err.getvalue().count("no wpeer= tags"), 1)   # warned once
        diag = cp.mpi_dependency_diagnostics(_trace(ev))
        self.assertTrue(diag["legacy_key"])
        self.assertEqual(diag["unmatched_by_comm"], {"22": 1})

    def test_two_duplicated_communicators_with_the_same_peers_and_tags(self):
        # Rank 0 sends tag 7 on dup X (commid 41) then on dup Y (commid 42);
        # rank 1 posts its receives in the opposite order (Y first). Without
        # the communicator in the key the first receive would take the
        # first send.
        ev = [
            send(0, 0, rank=0, peer=1, wpeer=1, tag=7, commid=41, name="sendX"),
            send(0, 20, rank=0, peer=1, wpeer=1, tag=7, commid=42, name="sendY"),
            _mpi("recvY", 1, 0, 60, type="recv", rank=1, peer=0, wpeer=0, tag=7, commid=42),
            _mpi("recvX", 1, 70, 10, type="recv", rank=1, peer=0, wpeer=0, tag=7, commid=41),
        ]
        edges = p2p_edges(_trace(ev))
        self.assertEqual(sorted((s.name, d.name) for s, d, _c in edges),
                         [("sendX", "recvX"), ("sendY", "recvY")])

    def test_comm_world(self):
        ev = [
            send(0, 0, rank=0, peer=1, wpeer=1, tag=1, commid=0),
            send(1, 0, rank=1, peer=0, wpeer=0, tag=1, commid=0),
            recv(1, 20, rank=1, peer=0, wpeer=0, tag=1, commid=0),
            recv(0, 20, rank=0, peer=1, wpeer=1, tag=1, commid=0),
        ]
        edges = p2p_edges(_trace(ev))
        self.assertEqual(sorted((s.pid, d.pid, c) for s, d, c in edges),
                         [(0, 1, "medium"), (1, 0, "medium")])

    def test_wildcard_irecv_resolved_to_a_world_rank(self):
        # Rank 3 (local 1 of commid 22) receives with ANY_SOURCE; the Wait
        # reports the comm-local source (rpeer=0) and its world rank (rwpeer=2).
        ev = [
            send(2, 0, rank=2, peer=1, wpeer=3, tag=9, commid=22),
            _mpi("MPI_Irecv", 3, 0, 1, sid="r1", type="irecv", rank=3, peer=-2, tag=-1,
                 commid=22, wildcard=1),
            _mpi("MPI_Wait", 3, 5, 40, psid="r1", type="wait", rank=3, rpeer=0, rtag=9, rwpeer=2),
        ]
        edges = p2p_edges(_trace(ev))
        self.assertEqual([(s.pid, d.name, c) for s, d, c in edges], [(2, "MPI_Wait", "high")])

    def test_waitall_rwmatches(self):
        ev = [
            send(1, 0, rank=1, peer=0, wpeer=0, tag=3, commid=0),
            send(2, 0, rank=2, peer=0, wpeer=0, tag=4, commid=0),
            _mpi("MPI_Irecv", 0, 0, 1, sid="a", type="irecv", rank=0, peer=-2, tag=-1, commid=0, wildcard=1),
            _mpi("MPI_Irecv", 0, 1, 1, sid="b", type="irecv", rank=0, peer=-2, tag=-1, commid=0, wildcard=1),
            _mpi("MPI_Waitall", 0, 5, 40, psid="a;b", type="waitall", rank=0,
                 rmatches="a/2/4;b/1/3", rwmatches="a/2;b/1"),
        ]
        edges = p2p_edges(_trace(ev))
        self.assertEqual(sorted((s.pid, d.name) for s, d, _c in edges),
                         [(1, "MPI_Waitall"), (2, "MPI_Waitall")])


class TestPerProcessRequestIds(unittest.TestCase):
    def test_colliding_request_ids_of_two_ranks_stay_apart(self):
        # Older MPI hooks numbered requests 1, 2, ... in every process. Rank 0
        # posts Irecv "1"; rank 1 posts Isend "1" (to rank 0) and waits on it.
        # Each Wait must complete its own process's request.
        ev = [
            _mpi("MPI_Irecv", 0, 0, 1, sid="1", type="irecv", rank=0, peer=1, wpeer=1, tag=4, commid=0),
            send(1, 5, rank=1, peer=0, wpeer=0, tag=4, commid=0, typ="isend", name="MPI_Isend", sid="1"),
            _mpi("MPI_Wait", 1, 20, 5, psid="1", type="wait", rank=1),
            _mpi("MPI_Wait", 0, 10, 30, psid="1", type="wait", rank=0),
        ]
        trace = _trace(ev)
        edges = p2p_edges(trace)
        self.assertEqual([(s.pid, d.name, d.pid) for s, d, _c in edges], [(1, "MPI_Wait", 0)])
        spans, preds = cp.build_dependency_graph(trace)
        explicit = sorted((spans[u].name, spans[u].pid, spans[v].pid)
                          for v, pl in preds.items() for u, k, _c in pl if k == "explicit_span_id")
        self.assertEqual(explicit, [("MPI_Irecv", 0, 0), ("MPI_Isend", 1, 1)])


class TestCancellation(unittest.TestCase):
    def test_wait_on_a_cancelled_receive_gets_no_edge(self):
        # mpi_proto phase C: a wildcard Irecv is cancelled; its Wait used to
        # carry stale rpeer/rtag from the status buffer and get a high edge.
        ev = [
            send(1, 0, rank=1, peer=0, wpeer=0, tag=200, commid=0),
            _mpi("MPI_Irecv", 0, 0, 1, sid="c", type="irecv", rank=0, peer=-2, tag=-1, commid=0, wildcard=1),
            _mpi("MPI_Wait", 0, 5, 10, psid="c", type="wait", rank=0, cancelled=1),
            # even a stale resolution next to cancelled=1 must not pair
            _mpi("MPI_Irecv", 0, 20, 1, sid="d", type="irecv", rank=0, peer=-2, tag=-1, commid=0, wildcard=1),
            _mpi("MPI_Wait", 0, 25, 10, psid="d", type="wait", rank=0, cancelled=1, rpeer=1, rtag=200, rwpeer=1),
        ]
        self.assertEqual(p2p_edges(_trace(ev)), [])
        self.assertEqual(cp.mpi_dependency_diagnostics(_trace(ev))["cancelled"], 2)

    def test_cancelled_send_does_not_shift_later_pairs(self):
        ev = [
            send(0, 0, rank=0, peer=1, wpeer=1, tag=1, commid=0, typ="isend", name="isend_cancelled", sid="s1"),
            _mpi("MPI_Wait", 0, 5, 5, psid="s1", type="wait", rank=0, cancelled=1),
            send(0, 20, rank=0, peer=1, wpeer=1, tag=1, commid=0, name="real_send"),
            recv(1, 0, rank=1, peer=0, wpeer=0, tag=1, commid=0),
        ]
        edges = p2p_edges(_trace(ev))
        self.assertEqual([s.name for s, _d, _c in edges], ["real_send"])

    def test_cancelled_entry_of_a_multi_completion(self):
        ev = [
            send(1, 0, rank=1, peer=0, wpeer=0, tag=3, commid=0),
            _mpi("MPI_Irecv", 0, 0, 1, sid="a", type="irecv", rank=0, peer=1, wpeer=1, tag=3, commid=0),
            _mpi("MPI_Irecv", 0, 1, 1, sid="b", type="irecv", rank=0, peer=1, wpeer=1, tag=3, commid=0),
            _mpi("MPI_Waitall", 0, 5, 40, psid="a;b", type="waitall", rank=0, cancelled=1, cpsid="b"),
        ]
        edges = p2p_edges(_trace(ev))
        self.assertEqual(len(edges), 1)


class TestSendVariantsAndSendrecv(unittest.TestCase):
    def test_ssend_bsend_rsend_issend_are_senders(self):
        ev = []
        for i, (typ, name) in enumerate((("ssend", "MPI_Ssend"), ("bsend", "MPI_Bsend"),
                                         ("rsend", "MPI_Rsend"), ("issend", "MPI_Issend"))):
            ev.append(send(0, 100 * i, rank=0, peer=1, wpeer=1, tag=i, commid=0, typ=typ, name=name))
            ev.append(recv(1, 100 * i, rank=1, peer=0, wpeer=0, tag=i, commid=0))
        edges = p2p_edges(_trace(ev))
        self.assertEqual(sorted(s.name for s, _d, _c in edges),
                         ["MPI_Bsend", "MPI_Issend", "MPI_Rsend", "MPI_Ssend"])

    def test_sendrecv_exchange_is_paired_both_ways_and_stays_acyclic(self):
        # Ranks 0 and 1 exchange with MPI_Sendrecv: each call is a zero-length
        # send half plus the receive half (the hook's representation).
        def sr(pid, start, dur, rank, peer):
            return [
                _mpi("MPI_Sendrecv.send", pid, start, 0, type="send", op="sendrecv", rank=rank,
                     peer=peer, wpeer=peer, tag=0, commid=0),
                _mpi("MPI_Sendrecv", pid, start, dur, type="recv", op="sendrecv", rank=rank,
                     peer=peer, wpeer=peer, tag=0, commid=0),
            ]
        # rank 1 computes for 60 before its Sendrecv; rank 0 enters at 0 and waits
        ev = [_mpi("compute", 1, 0, 60, type="work")] + sr(0, 0, 100, 0, 1) + sr(1, 60, 40, 1, 0) + \
            [_mpi("after", 0, 110, 10, type="work"), _mpi("after", 1, 110, 10, type="work")]
        trace = _trace(ev)
        edges = p2p_edges(trace)
        self.assertEqual(sorted((s.name, s.pid, d.name, d.pid) for s, d, _c in edges),
                         [("MPI_Sendrecv.send", 0, "MPI_Sendrecv", 1),
                          ("MPI_Sendrecv.send", 1, "MPI_Sendrecv", 0)])
        spans, preds = cp.build_dependency_graph(trace)
        self.assertIsNotNone(cp._topological_order(len(spans), preds))
        report = cp.analyze(trace)
        path = [(report.spans[i].name, report.spans[i].pid) for i in report.path_span_indices]
        # rank 0 waited in its Sendrecv for rank 1's late send half
        self.assertEqual(path[:3], [("compute", 1), ("MPI_Sendrecv.send", 1), ("MPI_Sendrecv", 0)])

    def test_proc_null_is_not_matched_or_reported(self):
        ev = [
            send(0, 0, rank=0, peer=-1, tag=0, commid=0, null=1),
            recv(0, 20, rank=0, peer=-1, tag=0, commid=0, null=1),
        ]
        self.assertEqual(p2p_edges(_trace(ev)), [])
        diag = cp.mpi_dependency_diagnostics(_trace(ev))
        self.assertNotIn("unmatched_by_comm", diag)


class TestDiagnostics(unittest.TestCase):
    def test_counts_unmatched_test_only_and_orphan_completions(self):
        ev = [
            # a receive whose sender is not in the trace (communicator 5)
            recv(1, 0, rank=1, peer=0, wpeer=0, tag=1, commid=5),
            # an Irecv completed by MPI_Test only
            _mpi("MPI_Irecv", 1, 100, 1, sid="t", type="irecv", rank=1, peer=0, wpeer=0, tag=2, commid=0),
            # a Wait on a request the hook never saw (e.g. an unhooked call)
            _mpi("MPI_Wait", 1, 200, 10, type="wait", rank=1),
            # a Wait naming a request id never posted in this trace
            _mpi("MPI_Wait", 1, 300, 10, psid="ghost", type="wait", rank=1),
        ]
        trace = _trace(ev)
        trace.add(InstantEvent("MPI_Test", Category.MPI, 150, 1, 1,
                               {"type": "test", "flag": "1", "rank": "1", "psid": "t"}))
        diag = cp.mpi_dependency_diagnostics(trace)
        self.assertEqual(diag["unmatched_by_comm"], {"5": 1})
        self.assertEqual(diag["test_only_completions"], 1)
        self.assertEqual(diag["orphan_completions"], 2)
        warnings = cp.dependency_warnings(diag)
        self.assertTrue(any("got no point-to-point edge" in w for w in warnings))
        self.assertTrue(any("MPI_Test" in w for w in warnings))
        self.assertTrue(any("never saw initiated" in w for w in warnings))
        report = cp.analyze(trace)
        self.assertEqual(report.diagnostics, diag)
        self.assertTrue(any("never saw initiated" in n for n in report.notes))

    def test_no_mpi_no_diagnostics(self):
        trace = _trace([SpanEvent("k", Category.CPU, 0, 10, 1, 1)])
        self.assertEqual(cp.mpi_dependency_diagnostics(trace), {})

    def test_cli_commands_print_the_coverage(self):
        # `efficiency` once read the diagnostics from the wrong report and
        # crashed after printing its numbers
        import subprocess
        import tempfile
        from src.core.trace_io import write_json_export
        root = Path(__file__).resolve().parent.parent
        trace = _trace([send(10, 100, rank=0, peer=1, wpeer=1, tag=3, commid=0),
                        recv(11, 50, rank=1, peer=0, wpeer=0, tag=3, commid=0),
                        recv(11, 400, rank=1, peer=0, wpeer=0, tag=9, commid=0)])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.json"
            write_json_export(trace, str(p))
            for cmd in ("efficiency", "critical-path"):
                r = subprocess.run([sys.executable, str(root / "hprofiler"), cmd, str(p)],
                                   capture_output=True, text=True, timeout=120)
                self.assertEqual(r.returncode, 0, (cmd, r.stderr[-1500:]))
                self.assertIn("MPI dependency coverage", r.stdout, cmd)
                self.assertIn("no sender found     : 1 on communicator id 0", r.stdout, cmd)


if __name__ == "__main__":
    unittest.main()
