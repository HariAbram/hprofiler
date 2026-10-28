"""
Clock domains and timestamp bounds (src/analysis/cp_clock.py, cp_model's
"Clocks"):

  * the solver: an unbounded domain gets the offset interval its
    dependencies allow (and a shift when 0 is outside it); a bounded one
    that cannot satisfy a dependency has it contradicted, the weakest
    evidence first; offsets chain through domains;
  * domains: node 0's host clock is the reference, merged nodes carry the
    merge plan's bound, CUPTI device clocks the tolerance (or the drift
    correction applied), OpenCL device clocks no bound, NCCL asynchronous
    timing its ref_err; proxies are on the host clock;
  * in the model: OpenCL device work recorded late enough to contradict
    the waits on it is shifted back before the graph is built, the edges
    kept; a message between nodes needing more than their bound is
    contradicted (disabled, counted, clock_bound in the evidence); two
    releases from different clocks closer than their offsets allow are an
    undecidable step.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis import cp_clock as cc
from src.analysis import cp_model as cm
from src.analysis import criticalpath as cp
from src.analysis import evidence as ev
from src.core.events import Category, SpanEvent
from src.core.store import EDGE_DTYPE
from tests.test_cp_model import _trace, recv, send, work


def span(name, cat, start, end, pid=1, tid=1, **tags):
    return SpanEvent(name, Category(cat), start, end - start, pid, tid, tags={k: str(v) for k, v in tags.items()})


class TestSolver(unittest.TestCase):
    keys = [("host", "0"), ("opencl", "0", 1)]

    def test_unbounded_domain_gets_its_feasible_interval_and_a_shift(self):
        # launch: host 100 -> device start 50 (delta >= 50); wait: device end
        # 300 -> host 400 (delta <= 100)
        t, dom = [100, 50, 300, 400], [0, 1, 1, 0]
        r = cc.analyze_clocks(dom.__getitem__, t, [(0, 0, 1, 1), (1, 2, 3, 1)], self.keys, [0, cc.INF])
        self.assertTrue(r.feasible_as_recorded)
        self.assertEqual((r.lo[1], r.hi[1], r.shift[1]), (50, 100, 50))
        self.assertEqual(r.contradicted, set())

    def test_a_bounded_domain_contradicts_what_it_cannot_satisfy(self):
        t, dom = [100, 50, 300, 400], [0, 1, 1, 0]
        keys = [("host", "0"), ("device", "0", 1, "cuda", "0")]
        r = cc.analyze_clocks(dom.__getitem__, t, [(0, 0, 1, 1), (1, 2, 3, 1)], keys, [0, 20.0])
        self.assertFalse(r.feasible_as_recorded)
        self.assertEqual(r.contradicted, {0})
        self.assertEqual(r.shift, [0, 0])                  # a bounded domain is never shifted
        self.assertEqual(r.relaxed[0]["edges_contradicted"], 1)

    def test_the_weakest_evidence_gives_way(self):
        # needs delta >= 50 (direct) and delta <= -10 (inferred): the inferred one goes
        t, dom = [100, 50, 300, 290], [0, 1, 1, 0]
        r = cc.analyze_clocks(dom.__getitem__, t, [(0, 0, 1, 1), (1, 2, 3, 0)], self.keys, [0, cc.INF])
        self.assertEqual(r.contradicted, {1})
        self.assertEqual(r.relaxed[0]["strength"], "inferred")
        self.assertEqual(r.shift[1], 50)

    def test_offsets_chain_through_domains(self):
        # host -> A needs dA >= 30; A -> B needs dB - dA >= 20: dB >= 50
        keys = [("host", "0"), ("opencl", "0", 1), ("opencl", "0", 2)]
        t, dom = [100, 70, 50, 30], [0, 1, 1, 2]
        r = cc.analyze_clocks(dom.__getitem__, t, [(0, 0, 1, 1), (1, 2, 3, 1)], keys, [0, cc.INF, cc.INF])
        self.assertEqual(r.lo[1], 30)
        self.assertEqual(r.lo[2], 50)
        self.assertEqual(r.hi[2], cc.INF)


class TestDomains(unittest.TestCase):
    def test_kinds_of_clock(self):
        nb = {"1": 2000.0}
        host = span("MPI_Send", "mpi", 0, 10, type="send")
        node1 = span("MPI_Send", "mpi", 0, 10, type="send", node=1)
        cupti = span("k", "cuda", 0, 10, side="gpu", rt="cuda", op="kernel", timing="device", dev=0)
        drift = span("k", "cuda", 0, 10, side="gpu", rt="cuda", op="kernel", timing="device", dev=0,
                     clock_corr_ns=90_000)
        proxy = span("k", "cuda", 0, 10, side="gpu", rt="cuda", op="kernel", timing="proxy_event")
        ocl = span("k", "opencl", 0, 10, side="gpu", type="kernel")
        ocl_copy = span("clReadBuffer_gpu", "memory", 0, 10, type="read")
        nccl = span("ncclSend", "nccl", 0, 10, side="gpu", timing="async", ref_err=3000)
        self.assertEqual(cc.domain_of(host, nb), (("host", "0"), 0.0))
        self.assertEqual(cc.domain_of(node1, nb), (("host", "1"), 2000.0))
        self.assertEqual(cc.domain_of(cupti, nb)[1], float(cc.ga.CLOCK_TOLERANCE_NS))
        self.assertEqual(cc.domain_of(drift, nb)[1], 90_000.0)
        self.assertEqual(cc.domain_of(proxy, nb)[0], ("host", "0"))
        self.assertEqual(cc.domain_of(ocl, nb), (("opencl", "0", 1), cc.INF))
        self.assertEqual(cc.domain_of(ocl_copy, nb), (("opencl", "0", 1), cc.INF))
        self.assertEqual(cc.domain_of(nccl, nb), (("nccl_async", "0", 1), 3000.0))
        self.assertEqual(cc.domain_of(span("x", "mpi", 0, 1, node=7), nb)[1], cc.INF)

    def test_node_bounds_from_the_merge_plan(self):
        class M:
            clock = {"merge": {"nodes": [{"bound_ns": 0}, {"bound_ns": 1500}, {"bound_ns": None}]}}
        self.assertEqual(cc.node_bounds_of(M()), {"0": 0.0, "1": 1500.0, "2": cc.INF})
        self.assertEqual(cc.node_bounds_of(None), {})


def _graph(t, edges):
    g0 = cp.load_or_build_graph(t)
    return cp.Graph(t, g0.seqs, np.array(edges, dtype=EDGE_DTYPE))


class TestInTheModel(unittest.TestCase):
    def test_late_opencl_device_clock_is_shifted_back(self):
        # host enqueue [0, 10], device kernel recorded at [500k, 600k] (its clock
        # late by ~490 us), host wait [20, 200k]: the wait cannot have ended
        # before the kernel did
        late = 490_000
        t = _trace([span("k", "opencl", 0, 10, side="cpu", type="kernel"),
                    span("k", "opencl", 15 + late, 100_000 + late, tid=99, side="gpu", type="kernel"),
                    span("clWaitForEvents", "sync", 20, 200_000, type="sync")])
        L, W = cp._KIND_CODE["launch"], cp._KIND_CODE["device_wait"]
        c = cp._CONF_CODE["certain"]
        graph = _graph(t, [(1, 0, L, c), (2, 1, W, c)])
        g = cm.build_event_graph(t, graph)
        ca = g.clock
        k = ca.keys.index(("opencl", "0", 1))
        self.assertFalse(ca.feasible_as_recorded)
        self.assertEqual(ca.shift[k], -390_000)              # the least that satisfies the wait
        self.assertEqual(dict(g.dropped), {})                # the wait edge kept after the shift
        self.assertLessEqual(g.t[3], g.t[5])                  # kernel end <= wait end
        self.assertGreaterEqual(g.t[2], g.t[0])               # kernel start >= enqueue start
        res = cm.analyze_model(t, graph)
        self.assertEqual(res.graph_evidence["clocks"]["domains"][k]["shift_ns"], ca.shift[k])

    def test_a_message_between_nodes_beyond_their_bound_is_contradicted(self):
        class Meta:
            clock = {"merge": {"nodes": [{"bound_ns": 0}, {"bound_ns": 1000}]}}
        # the send on node 1 starts 5 us after the receive on node 0 ended
        s = send(2, 105_000, 1, 0, 9)
        s.tags["node"] = "1"
        r = recv(1, 0, 100_000, 0, 1, 9)
        t = _trace([r, s, work("w", 1, 100_000, 300_000)])
        t.metadata = Meta()
        P, m = cp._KIND_CODE["p2p"], cp._CONF_CODE["medium"]
        graph = _graph(t, [(0, 1, P, m)])
        g = cm.build_event_graph(t, graph)
        self.assertEqual(g.clock_contradicted + sum(g.dropped.values()), 1)
        e = ev.evidence_from_graph(graph.edges, g)
        self.assertEqual(e.status(0), "contradicted")
        against = ev.kinds_of(int(e.cand["ev_against"][0]))
        self.assertTrue({"clock_bound", "time_contradiction"} & set(against), against)

    def test_releases_closer_than_the_clocks_allow_are_undecidable(self):
        # a sync call gated by two device operations on different devices
        # (each clock +-50 us) that ended 10 us apart
        k1 = span("k1", "cuda", 0, 100_000, tid=50, side="gpu", rt="cuda", op="kernel", timing="device",
                  dev=0, stream=1)
        k2 = span("k2", "cuda", 0, 110_000, tid=51, side="gpu", rt="cuda", op="kernel", timing="device",
                  dev=1, stream=1)
        sync = span("cudaDeviceSynchronize", "sync", 0, 200_000, type="sync")
        t = _trace([k1, k2, sync])
        W, c = cp._KIND_CODE["device_wait"], cp._CONF_CODE["certain"]
        graph = _graph(t, [(2, 0, W, c), (2, 1, W, c)])
        g = cm.build_event_graph(t, graph)
        w = cm.walk(g)
        self.assertEqual(w.undecidable, 1)
        self.assertEqual(w.undecidable_ns, 10_000)
        self.assertEqual(w.undecidable_samples[0]["alternative"], "device_wait")
        # on one device the order is exact
        k2.tags["dev"] = "0"
        t = _trace([k1, k2, sync])
        g = cm.build_event_graph(t, _graph(t, [(2, 0, W, c), (2, 1, W, c)]))
        self.assertEqual(cm.walk(g).undecidable, 0)


if __name__ == "__main__":
    unittest.main()
