"""
Alternative worlds (src/analysis/cp_alternatives.py):

  * a trace whose every answer is forced (complete channels, one clock) has
    no alternatives: one world, everything robust;
  * a point-to-point channel with more sends than paired receives gets
    shifted alignments, one choice unit per channel; they join the
    evidence set as unsupported and make the builder's pairing ambiguous;
    one-change and sampled worlds are evaluated and the robust sets are
    their intersection / union;
  * every world a channel's options allow pairs each send with at most one
    receive, in order (MPI's non-overtaking rule): a receive only one send
    can have completed keeps it, and after lost events a receive may have
    had no recorded send, the later ones of its run moving back by one;
  * a rendezvous whose members started closer than the two nodes' clock
    bound may have had the other last arriver; on one clock it may not;
  * a reused correlation id offers the other submission, as a swap: no
    world gives one submission two launches;
  * the verdict says "not enough evidence" when the path's time rests on
    spans that leave it in some world, when spans join it in another, or
    when the modeled length varies.
"""
import itertools
import random
import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis import cp_alternatives as alt
from src.analysis import cp_model as cm
from src.analysis import criticalpath as cp
from src.analysis import evidence as ev
from tests.test_cp_model import _trace, mpi, random_trace, recv, send, work


def run(t, **kw):
    graph = cp.load_or_build_graph(t)
    res = cm.analyze_model(t, graph, sensitivity=0)
    return graph, res, alt.analyze_alternatives(t, graph, res, **kw)


def units_of(t):
    graph = cp.load_or_build_graph(t)
    g = cm.build_event_graph(t, graph)
    es, units = alt.generate(t, graph, g, ev.evidence_from_graph(graph.edges, g))
    return es, units


def every_world(es, units):
    """The edges of every combination of the units' options."""
    for choice in itertools.product(*(range(len(u.options)) for u in units)):
        yield alt._world_edges(es, units, dict(enumerate(choice)))


def p2p_pairs(t, edges):
    """(send start, receive end) of a world's point-to-point edges."""
    spans = list(t.iter_spans())
    p2p = edges[edges["kind"] == cp._KIND_CODE["p2p"]]
    return [(spans[int(s)].start_ns, spans[int(d)].end_ns) for s, d in zip(p2p["src"].tolist(), p2p["dst"].tolist())]


class Lossy:
    """Metadata of a capture in which pid 1's MPI hook dropped records."""
    capture_health = {"transport": {"1/mpi": {"dropped_full": 3}}}
    clock = None


class TestNoAlternatives(unittest.TestCase):
    def test_forced_answers_give_one_world(self):
        _g, res, a = run(random_trace(0))
        self.assertEqual(a.verdict, "no alternatives")
        self.assertEqual(a.worlds, 1)
        self.assertEqual(a.every, res.walk.spans(res.graph))
        self.assertEqual(a.alternatives, 0)

    def test_a_tie_on_one_clock_is_undecided(self):
        # random_trace(3): two members of an allreduce start at one instant
        graph, res, a = run(random_trace(3))
        self.assertEqual(a.units, {"last_arriver": 1})
        g = res.graph
        arr = graph.edges[graph.edges["kind"] == cp._KIND_CODE["arrival"]]
        starts = sorted({g.t[2 * int(d)] for d in arr["dst"].tolist()} | {g.t[2 * int(u)] for u in arr["src"].tolist()})
        self.assertTrue(any(g.t[2 * int(u)] == g.t[2 * int(d)] for u, d in zip(arr["src"].tolist(), arr["dst"].tolist())),
                        starts)


class TestPointToPointAlignment(unittest.TestCase):
    def trace(self):
        # rank 1 sends three messages on one channel, rank 0 recorded two receives
        return _trace([work("a1", 2, 0, 100), send(2, 100, 1, 0, 9), work("a2", 2, 110, 200),
                       send(2, 200, 1, 0, 9), work("a3", 2, 210, 300), send(2, 300, 1, 0, 9),
                       recv(1, 50, 400, 0, 1, 9), recv(1, 400, 450, 0, 1, 9), work("b", 1, 450, 600)])

    def test_shifted_alignments_are_alternatives(self):
        _g, _res, a = run(self.trace(), samples=4)
        self.assertEqual(a.units, {"p2p_alignment": 1})          # one unit per channel
        e = a.evidence
        alts = range(e.n_selected, len(e))
        self.assertTrue(alts)
        self.assertEqual({e.status(i) for i in alts}, {"unsupported"})
        for i in alts:
            self.assertIn("alignment_shift", ev.kinds_of(int(e.cand["ev_for"][i])))
            self.assertEqual(int(e.cand["selected"][i]), 0)
        p2p = cp._KIND_CODE["p2p"]
        contested = [i for i in range(e.n_selected) if int(e.cand["kind"][i]) == p2p]
        self.assertEqual({e.status(i) for i in contested}, {"ambiguous"})
        self.assertGreater(a.worlds, 1)
        self.assertEqual(a.worlds, 1 + a.one_change_worlds + a.sampled_worlds)
        self.assertTrue(a.every <= a.some)
        self.assertLessEqual(a.covered_range_ns[0], a.covered_range_ns[1])

    def test_a_complete_channel_has_none(self):
        t = _trace([send(2, 100, 1, 0, 9), recv(1, 50, 400, 0, 1, 9)])
        _g, _res, a = run(t)
        self.assertEqual(a.units, {})

    def test_every_world_pairs_each_send_once_in_order(self):
        t = self.trace()
        es, units = units_of(t)
        self.assertEqual(len(units), 1)
        # as recorded; the second receive moved to the third send; both moved on by one
        self.assertEqual(len(units[0].options), 3)
        for edges in every_world(es, units):
            pairs = p2p_pairs(t, edges)
            sends = [s for s, _ in pairs]
            self.assertEqual(len(sends), len(set(sends)), pairs)
            self.assertEqual(sorted(pairs, key=lambda p: p[1]), sorted(pairs), pairs)   # order kept

    def test_random_channels_give_admissible_worlds_only(self):
        rng = random.Random(7)
        for case in range(60):
            with self.subTest(case=case):
                spans, ts = [], 0
                for _k in range(rng.randrange(2, 7)):
                    ts += rng.choice([5, 20, 60])
                    spans.append(send(2, ts, 1, 0, 9))
                    ts += 10
                n_recv = rng.randrange(1, len([s for s in spans]) + 1)
                tr = rng.choice([0, 30])
                for _k in range(n_recv):                  # sequential receives on one thread
                    a, tr = tr, tr + rng.choice([10, 40, 120])
                    spans.append(recv(1, a, tr, 0, 1, 9))
                t = _trace(spans)
                if rng.random() < 0.5:
                    t.metadata = Lossy()
                es, units = units_of(t)
                for edges in every_world(es, units):
                    pairs = sorted(p2p_pairs(t, edges), key=lambda p: p[1])
                    sends = [s for s, _ in pairs]
                    self.assertEqual(len(sends), len(set(sends)), pairs)
                    self.assertEqual(sends, sorted(sends), pairs)

    def test_a_receive_only_one_send_can_have_completed_keeps_it(self):
        # the first receive ends before the second send starts: only the
        # second receive may have had the third send, and never the first
        t = _trace([work("c1", 2, 0, 300), send(2, 300, 1, 0, 9, end=350), work("c2", 2, 350, 800),
                    send(2, 800, 1, 0, 9, end=850), work("c3", 2, 850, 1300), send(2, 1300, 1, 0, 9, end=1350),
                    work("prep", 1, 0, 200), recv(1, 200, 400, 0, 1, 9), work("mid", 1, 400, 600),
                    recv(1, 600, 1400, 0, 1, 9), work("solve", 1, 1400, 1800)])
        es, units = units_of(t)
        worlds = [sorted(p2p_pairs(t, e)) for e in every_world(es, units)]
        self.assertEqual(worlds, [[(300, 400), (800, 1400)], [(300, 400), (1300, 1400)]])

    def test_after_lost_events_a_receive_may_have_had_no_recorded_send(self):
        t = _trace([send(2, 100, 1, 0, 9), send(2, 200, 1, 0, 9),
                    recv(1, 50, 150, 0, 1, 9), recv(1, 150, 300, 0, 1, 9)])
        t.metadata = Lossy()
        es, units = units_of(t)
        self.assertEqual([u.generator for u in units], ["p2p_alignment"])
        worlds = {tuple(sorted(p2p_pairs(t, e))) for e in every_world(es, units)}
        # as recorded; the second receive's send lost; the first's lost, the second on the first send
        self.assertEqual(worlds, {((100, 150), (200, 300)), ((100, 150),), ((100, 300),)})
        _g, _res, a = run(t)
        p2p = cp._KIND_CODE["p2p"]
        self.assertEqual({a.evidence.status(i) for i in range(a.evidence.n_selected)
                          if int(a.evidence.cand["kind"][i]) == p2p}, {"ambiguous"})


class TestLastArriver(unittest.TestCase):
    def coll(self, pid, a, b, node):
        s = mpi("MPI_Allreduce", pid, a, b, type="allreduce", rank=pid - 1, commid=0)
        s.tags["node"] = str(node)
        return s

    def test_members_within_the_clock_bound_may_swap(self):
        class Meta:
            clock = {"merge": {"nodes": [{"bound_ns": 0}, {"bound_ns": 5_000}]}}
        t = _trace([work("w0", 1, 0, 100_000), self.coll(1, 100_000, 200_000, 0),
                    work("w1", 2, 0, 101_000), self.coll(2, 101_000, 200_000, 1)])
        t.metadata = Meta()
        _g, _res, a = run(t)
        self.assertEqual(a.units, {"last_arriver": 1})
        self.assertGreaterEqual(a.worlds, 2)
        self.assertEqual(a.some - a.every, set(a.some - a.every))

    def test_one_clock_decides(self):
        t = _trace([work("w0", 1, 0, 100_000), self.coll(1, 100_000, 200_000, 0),
                    work("w1", 2, 0, 101_000), self.coll(2, 101_000, 200_000, 0)])
        _g, _res, a = run(t)
        self.assertEqual(a.units, {})


class TestLaunchCorrelation(unittest.TestCase):
    def test_a_reused_correlation_id_offers_the_other_submission(self):
        from tests.test_gpu_activity import build, cupti_kernel, host
        t = build([host("cudaLaunchKernel", 1_000_000, 1_000, lid=1, corr=7, stream=1),
                   host("cudaLaunchKernel", 1_010_000, 1_000, lid=2, corr=7, stream=2),
                   cupti_kernel(7, 5, 1_005_000, 1_006_000, name="_Z5firstv"),
                   cupti_kernel(7, 6, 1_020_000, 1_030_000, name="_Z6secondv")])
        _g, _res, a = run(t)
        self.assertEqual(a.units, {"launch_corr": 1})            # one group: the two launches trade
        e = a.evidence
        self.assertEqual({ev.kinds_of(int(e.cand["ev_for"][i]))[-1] for i in range(e.n_selected, len(e))},
                         {"shared_correlation"})
        es, units = units_of(t)
        self.assertEqual([(len(o.remove), len(o.add)) for o in units[0].options], [(0, 0), (2, 2)])
        launch = cp._KIND_CODE["launch"]
        for edges in every_world(es, units):          # never one submission for both kernels
            srcs = Counter(edges["src"][edges["kind"] == launch].tolist())
            self.assertEqual(max(srcs.values()), 1, srcs)


class TestWorldGraphs(unittest.TestCase):
    def test_a_world_from_the_skeleton_equals_a_full_build(self):
        for seed in range(6):
            with self.subTest(seed=seed):
                t = random_trace(seed)
                graph = cp.load_or_build_graph(t)
                full = cm.build_event_graph(t, graph, keep_skeleton=True)
                world = cm.graph_with_edges(full, graph.edges)
                for f in ("src", "dst", "mode", "kind", "prov", "conf", "owner", "d", "whi", "origin"):
                    self.assertEqual(getattr(world, f).tolist(), getattr(full, f).tolist(), f)
                self.assertEqual(bytes(world.enabled), bytes(full.enabled))
                self.assertEqual(list(world.order), list(full.order))
                self.assertEqual(cm.walk(world).steps.tolist(), cm.walk(full).steps.tolist())


class TestVerdict(unittest.TestCase):
    def test_not_enough_evidence_when_the_path_moves(self):
        # the second receive pairs either with the 2nd send (path through
        # a2) or the 3rd (path through a3): a different chain each way
        t = _trace([work("a1", 2, 0, 100), send(2, 100, 1, 0, 9), work("a2", 2, 110, 2_000),
                    send(2, 2_000, 1, 0, 9), work("a3", 2, 2_010, 9_000), send(2, 9_000, 1, 0, 9),
                    recv(1, 50, 150, 0, 1, 9), recv(1, 150, 9_500, 0, 1, 9), work("b", 1, 9_500, 9_600)])
        _g, _res, a = run(t, samples=0)
        self.assertGreaterEqual(a.one_change_worlds, 1)
        self.assertTrue(a.changes, a.to_dict())
        self.assertEqual(a.verdict, "not enough evidence")
        self.assertTrue(a.reasons)
        d = a.to_dict()
        self.assertEqual(d["verdict"], "not enough evidence")

    def test_spans_that_join_the_path_in_another_world_count(self):
        # the recorded path's calls are on it in both worlds, but in the
        # second one c3 joins it and the modeled length grows by 350
        t = _trace([work("c1", 2, 0, 300), send(2, 300, 1, 0, 9, end=350), work("c2", 2, 350, 800),
                    send(2, 800, 1, 0, 9, end=850), work("c3", 2, 850, 1300), send(2, 1300, 1, 0, 9, end=1350),
                    work("prep", 1, 0, 200), recv(1, 200, 400, 0, 1, 9), work("mid", 1, 400, 600),
                    recv(1, 600, 1400, 0, 1, 9), work("solve", 1, 1400, 1800)])
        _g, res, a = run(t, samples=0)
        self.assertEqual(a.robust_in_call_ns, a.default_in_call_ns)        # the recorded path is robust
        self.assertEqual((a.worst_robust_in_call_ns, a.worst_in_call_ns), (1250, 1700))
        self.assertEqual(a.modeled_range_ns, (1350, 1700))
        self.assertEqual(a.verdict, "not enough evidence")
        self.assertEqual(len(a.reasons), 2, a.reasons)
        self.assertIn("join", a.reasons[0])
        self.assertIn("modeled longest path", a.reasons[1])
        names = {res.graph.name(j) for j in a.some - a.every}
        self.assertEqual(names, {"c3", "MPI_Send"})


if __name__ == "__main__":
    unittest.main()
