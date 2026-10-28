"""
The dependency evidence model (src/analysis/evidence.py):

  * the default view is exactly the builders' edges, in order, on MPI,
    CUDA, OpenMP and mixed traces -- nothing computed from edges changes;
  * statuses: a correlation-id launch and stream order are observed,
    posting order and a status-resolved wildcard inferred, an edge whose
    source lies after its destination contradicted -- kept, selected, not an
    edge of the event graph -- and a lossy process's pairing at_most_one
    with event_loss against it;
  * groups: one launch group per device operation (exactly_one), one
    device_wait group per sync call (any_subset), several live candidates
    in an exactly_one group are ambiguous;
  * explain(): the evidence tree, derived_from included (status_then_order
    rests on mpi_status and posting_order), with the raw tags;
  * persistence: memory and disk stores, stale after the events change.
"""
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis import cp_model as cm
from src.analysis import criticalpath as cp
from src.analysis import evidence as ev
from src.core.store import EDGE_DTYPE
from tests.test_cp_model import _trace, mpi, random_trace, recv, send, work


def evidence(t):
    graph = cp.load_or_build_graph(t)
    return graph, ev.build_evidence(t, graph)


def statuses(e, kind):
    k = cp._KIND_CODE[kind]
    return sorted(ev.STATUSES[s] for s, kk in zip(e.cand["status"].tolist(), e.cand["kind"].tolist()) if kk == k)


class TestDefaultView(unittest.TestCase):
    def assert_same(self, t):
        graph, e = evidence(t)
        view = e.default_view()
        self.assertEqual(view.dtype, EDGE_DTYPE)
        self.assertEqual(view.tolist(), graph.edges.tolist())
        self.assertEqual(view.tolist(), cp.build_edges(t).tolist())
        self.assertEqual(e.n_selected, len(graph.edges))
        self.assertTrue(np.all(e.cand["selected"][:e.n_selected] == 1))

    def test_random_mpi_openmp_traces(self):
        for seed in range(6):
            with self.subTest(seed=seed):
                self.assert_same(random_trace(seed))

    def test_gpu_trace(self):
        from tests.test_gpu_activity import build, concurrent_streams
        self.assert_same(build(concurrent_streams()))

    def test_mixed_mpi_cuda_trace(self):
        from tests.test_criticalpath_comb import comb_cycle
        self.assert_same(comb_cycle(cycles=2))

    def test_empty_trace(self):
        _graph, e = evidence(_trace([]))
        self.assertEqual(len(e), 0)
        self.assertEqual(e.summary()["candidates"], 0)


class TestStatuses(unittest.TestCase):
    def test_gpu_edges_are_observed(self):
        from tests.test_gpu_activity import build, concurrent_streams
        _graph, e = evidence(build(concurrent_streams()))
        self.assertEqual(set(statuses(e, "launch")), {"observed"})
        self.assertEqual(set(statuses(e, "device_wait")), {"observed"})
        # one launch group per device operation, one wait group per sync call
        for kind, card in (("launch", ev.EXACTLY_ONE), ("device_wait", ev.ANY_SUBSET)):
            sel = e.cand[e.cand["kind"] == cp._KIND_CODE[kind]]
            gids = set(sel["group"].tolist())
            self.assertEqual({int(e.groups["card"][g]) for g in gids}, {card})
            self.assertEqual(len(gids), len(set(sel["dst"].tolist())))

    def test_point_to_point(self):
        _graph, e = evidence(_trace([send(2, 10, 1, 0, 9), recv(1, 0, 40, 0, 1, 9)]))
        self.assertEqual(statuses(e, "p2p"), ["inferred"])
        i = int(np.flatnonzero(e.cand["kind"] == cp._KIND_CODE["p2p"])[0])
        self.assertEqual(ev.kinds_of(int(e.cand["ev_for"][i])), ["posting_order"])
        self.assertEqual(ev.CARDINALITIES[int(e.groups["card"][int(e.cand["group"][i])])], "exactly_one")

    def test_status_resolved_wildcard_explains_both_sources(self):
        t = _trace([
            send(2, 20, 1, 0, 9),
            mpi("MPI_Irecv", 1, 0, 1, sid="r1", type="irecv", rank=0, peer=-2, tag=-1, commid=0, wildcard=1),
            mpi("MPI_Wait", 1, 5, 40, psid="r1", type="wait", rank=0, rpeer=1, rtag=9, rwpeer=1)])
        graph, e = evidence(t)
        i = int(np.flatnonzero(e.cand["kind"] == cp._KIND_CODE["p2p"])[0])
        x = e.explain(i, graph.spans)
        self.assertEqual((x["status"], x["provenance"], x["tier"]), ("inferred", "status_resolved", "high"))
        root = x["for"][0]
        self.assertEqual(root["kind"], "status_then_order")
        self.assertEqual({d["kind"] for d in root["derived_from"]}, {"mpi_status", "posting_order"})
        rec = {r["role"]: r for r in root["record"]}
        self.assertEqual((rec["dst"]["rpeer"], rec["dst"]["rtag"]), ("1", "9"))

    def test_a_contradicted_edge_is_kept_not_modelled(self):
        t = _trace([work("a", 1, 0, 100), work("b", 2, 200, 300)])
        g0 = cp.load_or_build_graph(t)
        edges = np.array([(0, 1, cp._KIND_CODE["p2p"], cp._CONF_CODE["medium"])], dtype=EDGE_DTYPE)
        graph = cp.Graph(t, g0.seqs, edges)
        e = ev.build_evidence(t, graph)
        self.assertEqual(statuses(e, "p2p"), ["contradicted"])
        self.assertEqual((int(e.cand["selected"][0]), int(e.cand["in_model"][0])), (1, 0))
        self.assertIn("time_contradiction", ev.kinds_of(int(e.cand["ev_against"][0])))
        self.assertEqual(e.default_view().tolist(), edges.tolist())     # still in the view
        self.assertEqual(e.summary()["by_status"], {"contradicted": 1})

    def test_event_loss_weakens_to_at_most_one(self):
        edges = np.array([(1, 0, cp._KIND_CODE["p2p"], cp._CONF_CODE["low"])], dtype=EDGE_DTYPE)
        t = _trace([send(2, 10, 1, 0, 9), recv(1, 0, 40, 0, 1, 9)])
        g0 = cp.load_or_build_graph(t)
        e = ev.build_evidence(t, cp.Graph(t, g0.seqs, edges))
        self.assertEqual(ev.CARDINALITIES[int(e.groups["card"][0])], "at_most_one")
        self.assertIn("event_loss", ev.kinds_of(int(e.cand["ev_against"][0])))
        self.assertEqual(e.status(0), "inferred")

    def test_two_launches_of_one_device_operation_are_ambiguous(self):
        from tests.test_gpu_activity import build, concurrent_streams
        t = build(concurrent_streams())
        g0 = cp.load_or_build_graph(t)
        launch = g0.edges[g0.edges["kind"] == cp._KIND_CODE["launch"]][0]
        other = next(int(s) for s in g0.edges["src"].tolist() if s != launch["src"])
        dup = np.array([(launch["dst"], other, launch["kind"], launch["conf"])], dtype=EDGE_DTYPE)
        e = ev.build_evidence(t, cp.Graph(t, g0.seqs, np.concatenate([g0.edges, dup])))
        both = np.flatnonzero((e.cand["kind"] == cp._KIND_CODE["launch"]) & (e.cand["dst"] == launch["dst"]))
        self.assertEqual(len(both), 2)
        live = [e.status(int(i)) for i in both]
        self.assertIn("ambiguous", live)
        self.assertEqual(len({int(e.cand["group"][i]) for i in both}), 1)

    def test_rendezvous_is_call_order_or_overlap(self):
        def coll(pid, a, b):
            return mpi("MPI_Allreduce", pid, a, b, type="allreduce", rank=pid - 1, commid=4)
        _graph, e = evidence(_trace([coll(1, 0, 100), coll(2, 50, 120)]))
        i = int(np.flatnonzero(e.cand["kind"] == cp._KIND_CODE["arrival"])[0])
        self.assertEqual(ev.kinds_of(int(e.cand["ev_for"][i])), ["call_order"])
        _graph, e = evidence(_trace([coll(1, 0, 100), coll(1, 100, 300), coll(2, 120, 310)]))
        kinds = {k for m in e.cand["ev_for"][e.cand["kind"] == cp._KIND_CODE["arrival"]].tolist()
                 for k in ev.kinds_of(m)}
        self.assertEqual(kinds, {"time_overlap"})


class TestPersistence(unittest.TestCase):
    def test_memory_store_round_trip(self):
        t = random_trace(2)
        a = ev.load_or_build_evidence(t)
        b = ev.load_or_build_evidence(t)
        self.assertEqual(a.cand.tolist(), b.cand.tolist())
        self.assertIsNotNone(t.store.load_blob("evidence", ev.evidence_version()))
        c = ev.EvidenceSet.from_bytes(a.to_bytes())
        self.assertEqual((c.cand.tolist(), c.groups.tolist(), c.n_selected),
                         (a.cand.tolist(), a.groups.tolist(), a.n_selected))

    def test_disk_store_round_trip_and_staleness(self):
        from src.core import trace_io
        from src.core.store import DiskTraceStore
        from src.core.trace import Trace
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "t.hpstore")
        try:
            t = Trace(store=DiskTraceStore(path))
            for s in random_trace(1).iter_spans():
                t.add(s)
            t.finalize()
            a = ev.load_or_build_evidence(t)
            t.close()
            r = trace_io.open_trace(path)
            self.assertIsNotNone(r.store.load_blob("evidence", ev.evidence_version()))   # survives reopen
            self.assertEqual(ev.load_or_build_evidence(r).cand.tolist(), a.cand.tolist())
            self.assertIsNone(r.store.load_blob("evidence", "ev-0|other"))
            s = next(iter(r.iter_spans()))
            s.tags["changed"] = "1"
            r.update_span(s)
            self.assertIsNone(r.store.load_blob("evidence", ev.evidence_version()))   # stale after a change
            r.close()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
