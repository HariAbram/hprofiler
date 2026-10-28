"""
The event-point critical-path model (src/analysis/cp_model.py): graph
invariants and time accounting.

  * invariants on random multi-rank traces (nesting, partial overlap, MPI
    point-to-point, collectives, barriers): every edge's time is >= 0 and
    its required time <= it; each thread's lane covers its extent once; the
    observed path is contiguous and covers each instant once, at most the
    wall time; every composition sums to it; the modeled longest path, its
    bounds, slack, every/some-path sets and counterfactuals agree with brute
    force over all paths;
  * overlapping spans: nesting, partial overlap, host/device overlap, a
    clock-skewed launch;
  * start-gated dependencies: an early-posted receive, a rendezvous, fork
    and join of an OpenMP region;
  * ambiguous edges: inferred edges and the direct-evidence alternative, a
    dependency contradicting the timestamps, a rooted collective;
  * multiple equal-length paths: ties, every vs some, counterfactuals that
    another path absorbs;
  * evidence: provenance, class, tier and reason of each edge kind;
  * compatibility: edges persisted by the previous builder are rebuilt,
    the report's span-level fields keep their invariants.
"""
import random
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis import cp_model as cm
from src.analysis import criticalpath as cp
from src.core.events import Category, SpanEvent
from src.core.store import EDGE_DTYPE
from src.core.trace import Trace, TraceMetadata


def _trace(spans):
    t = Trace(TraceMetadata())
    for s in spans:
        t.add(s)
    return t


def work(name, pid, start, end, tid=None, **kw):
    return SpanEvent(name, Category.OPENMP, start, end - start, pid, pid if tid is None else tid,
                     tags={"type": "work"}, **kw)


def mpi(name, pid, start, end, sid="", psid="", **tags):
    return SpanEvent(name, Category.MPI, start, end - start, pid, pid,
                     tags={k: str(v) for k, v in tags.items()}, span_id=sid, parent_span_id=psid)


def send(pid, start, rank, peer, tag, end=None):
    return mpi("MPI_Send", pid, start, start + 10 if end is None else end, type="send", rank=rank, peer=peer,
               wpeer=peer, tag=tag, commid=0)


def recv(pid, start, end, rank, peer, tag, **extra):
    return mpi("MPI_Recv", pid, start, end, type="recv", rank=rank, peer=peer, wpeer=peer, tag=tag, commid=0,
               **extra)


def model(t, **kw):
    return cm.analyze_model(t, cp.load_or_build_graph(t), **kw)


def names(res, spans):
    return sorted(res.graph.name(j) for j in spans)


def path_names(res):
    return [res.graph.name(j) for j in res.path_spans]


# ── brute force over every path of the event DAG ───────────────────────────

def all_paths(g):
    """Every path (as an edge list) of the enabled edges, from every node."""
    memo = {}

    def from_(u):
        if u not in memo:
            out = [[]]
            for k in range(g.out_ptr[u], g.out_ptr[u + 1]):
                e = g.out_idx[k]
                if g.enabled[e]:
                    out += [[e] + p for p in from_(g.dst[e])]
            memo[u] = out
        return memo[u]

    paths = []
    for v in g.order:
        paths += from_(v)
    return paths


def weight(g, path, gap_as_work=True, zero=(), gated_hi=False):
    tot = 0
    for e in path:
        if g.mode[e] in (cm.BUSY, cm.WAIT) and g.owner[e] in zero:
            continue
        tot += cm._w(g.mode[e], g.d[e], gap_as_work, g.whi[e], gated_hi)
    return tot


def random_trace(seed, n_ranks=3):
    """Ranks doing rounds of work (nested calls, partial overlaps, sends),
    each round ending in an allreduce that completes on every rank at one
    instant; receives posted at random times. Durations come from a small
    set, so equal-length paths and ties occur."""
    rng = random.Random(seed)
    spans = []
    tag = 0
    clock = [rng.choice([0, 10, 20]) for _ in range(n_ranks)]
    for _round in range(2):
        for r in range(n_ranks):
            pid, t = r + 1, clock[r]
            for _k in range(rng.randrange(1, 4)):
                dur = rng.choice([50, 100, 150])
                spans.append(work(f"w{r}", pid, t, t + dur))
                if rng.random() < 0.4:                       # a nested call
                    a = t + rng.choice([0, 10, 25])
                    spans.append(work(f"inner{r}", pid, a, a + rng.choice([10, 25])))
                if rng.random() < 0.1:                       # a partial overlap on the thread
                    spans.append(work(f"odd{r}", pid, t + dur // 2, t + dur + 20))
                t += dur + rng.choice([0, 0, 10, 40])
                if rng.random() < 0.5:                       # a message to another rank
                    peer = rng.choice([q for q in range(n_ranks) if q != r])
                    spans.append(send(pid, t, r, peer, tag))
                    t += 10
                    tag += 1
            clock[r] = t
        end = max(clock) + rng.choice([5, 20])
        for r in range(n_ranks):
            spans.append(mpi("MPI_Allreduce", r + 1, clock[r], end, type="allreduce", rank=r, commid=0))
            clock[r] = end + rng.choice([0, 10])
    # receives matching the sends: posted at a random time, ending later
    for s in [s for s in spans if s.name == "MPI_Send"]:
        r, peer, tg = int(s.tags["rank"]), int(s.tags["peer"]), s.tags["tag"]
        a = max(0, s.start_ns + rng.choice([-100, -20, 0, 30]))
        spans.append(recv(peer + 1, a, max(a, s.start_ns) + rng.choice([5, 30]), peer, r, tg))
    rng.shuffle(spans)
    return _trace(spans)


class TestInvariants(unittest.TestCase):
    def check(self, t):
        res = model(t, sensitivity=3)
        g, w = res.graph, res.walk
        self.assertEqual(g.cycle_edges_dropped, 0)
        for e in range(len(g)):
            self.assertGreaterEqual(g.d[e], 0)
            self.assertEqual(g.d[e], g.t[g.dst[e]] - g.t[g.src[e]])     # host-only: no clamps
            self.assertLessEqual(cm._w(g.mode[e], g.d[e], True), g.d[e])
            if g.mode[e] == cm.DEP:
                self.assertEqual(cm._w(g.mode[e], g.d[e], True), 0)
        # each thread's lane edges cover its extent exactly once
        thread_of = [(s.pid, s.tid) for s in t.iter_spans()]
        lanes = {}
        for e in range(len(g)):
            if g.kind[e] == cm._K["thread"]:
                lanes.setdefault(thread_of[g.src[e] >> 1], []).append(e)
        for pid, es in lanes.items():
            ts = [g.t[g.src[e]] for e in es] + [g.t[g.dst[e]] for e in es]
            self.assertEqual(sum(g.d[e] for e in es), max(ts) - min(ts), pid)
        # the observed path: contiguous, monotone, each instant once
        for a, b in zip(w.steps, w.steps[1:]):
            self.assertEqual(g.dst[a], g.src[b])
        self.assertEqual(list(w.cover), [g.d[e] for e in w.steps])
        if w.steps:
            self.assertEqual(g.src[w.steps[0]], w.first)
            self.assertEqual(g.dst[w.steps[-1]], w.sink)
        self.assertEqual(res.covered_ns, g.t[w.sink] - g.t[w.first])
        ext = t.store.span_extent(timed_only=True)
        wall = ext[1] - ext[0]
        self.assertLessEqual(res.covered_ns, wall)
        self.assertEqual(g.t[w.sink], ext[1])
        # compositions
        self.assertEqual(sum(res.work_by_category.values()) + sum(res.in_call_by_category.values())
                         + sum(res.unmatched_by_category.values())
                         + res.unrecorded_ns + res.device_idle_ns + sum(res.dependency_by_kind.values()),
                         res.covered_ns)
        self.assertEqual(sum(res.by_tier.values()), res.covered_ns)
        self.assertEqual(sum(res.by_class.values()), res.covered_ns)
        self.assertEqual(sum(ns for ns, _n in res.by_provenance.values()), res.covered_ns)
        self.assertEqual(sum(res.credited) + sum(res.gaps), res.covered_ns)
        self.assertEqual(sum(r.own_ns + r.other_ns for r in res.rows), res.covered_ns)
        self.assertEqual(len(set(res.path_spans)), len(res.path_spans))
        # the report keeps the span-level fields' invariants
        rep = cp.analyze(t, sensitivity=0)
        self.assertEqual(rep.total_path_ns, res.covered_ns)
        self.assertEqual(sum(rep.time_on_path_by_category.values()) + sum(rep.wait_caused_by_category.values()),
                         rep.total_path_ns)
        self.assertEqual(sum(rep.confidence_breakdown_ns().values()), rep.total_path_ns)
        self.assertEqual(len(rep.path_edge_kinds), max(0, len(rep.path_span_indices) - 1))
        self.assertTrue(set(rep.path_edge_kinds) <= set(cp.EDGE_KINDS), rep.path_edge_kinds)
        # modeled longest path vs brute force over every path
        paths = all_paths(g)
        L = max(weight(g, p) for p in paths)
        self.assertEqual(res.modeled_ns, L)
        self.assertEqual(res.modeled_unrecorded_idle_ns, max(weight(g, p, gap_as_work=False) for p in paths))
        direct = [p for p in paths if not any(g.inferred(e) for e in p)]
        self.assertEqual(res.modeled_direct_only_ns, max(weight(g, p) for p in direct))
        self.assertLessEqual(res.modeled_unrecorded_idle_ns, res.modeled_ns)
        self.assertLessEqual(res.modeled_direct_only_ns, res.modeled_ns)
        self.assertEqual(res.modeled_gated_tail_ns, max(weight(g, p, gated_hi=True) for p in paths))
        for e in range(len(g)):                 # required time's upper bound: within the edge
            self.assertLessEqual(0, g.whi[e])
            self.assertLessEqual(g.whi[e], g.d[e])
        # every variant: its walk and its longest path with the same parameters
        for name, (gaps, direct, gated) in cm.VARIANTS.items():
            ok = [p for p in paths if not direct or not any(g.inferred(e) for e in p)]
            self.assertEqual(res.variants[name]["modeled_ns"],
                             max(weight(g, p, gap_as_work=gaps, gated_hi=gated) for p in ok), name)
        lo, hi = res.modeled_range_ns
        self.assertLessEqual(lo, res.modeled_ns)
        self.assertLessEqual(res.modeled_ns, hi)
        self.assertLessEqual(res.modeled_ns, wall)
        longest_paths = [p for p in paths if weight(g, p) == L]
        busy_spans = lambda p: {g.owner[e] for e in p if g.mode[e] == cm.BUSY and g.d[e] > 0}
        some = set().union(*map(busy_spans, longest_paths))
        every = set.intersection(*map(busy_spans, longest_paths))
        c = res.crit
        self.assertEqual(c.on_some, some)
        self.assertEqual(c.on_all, every)
        for j, s in c.slack.items():
            self.assertGreaterEqual(s, 0)
            self.assertEqual(s == 0, j in c.on_some, j)
        for r in res.sensitivity:
            zero = {r["span"]} if r["scope"] == "span" else \
                {j for j in range(g.n) if g.cat[j] == r["category"] and g.name(j) == r["name"]}
            self.assertEqual(L - max(weight(g, p, zero=zero) for p in paths), r["modeled_reduction_ns"])
            self.assertGreaterEqual(r["modeled_reduction_ns"], 0)
            # the replay is monotone (max-plus) and 1-Lipschitz in the work removed
            removed = sum(g.d[e] for e in range(len(g)) if g.mode[e] == cm.BUSY and g.owner[e] in zero)
            self.assertGreaterEqual(r["replay_reduction_ns"], 0)
            self.assertLessEqual(r["replay_reduction_ns"], removed)
            self.assertLessEqual(r["modeled_reduction_ns"], r["work_on_longest_ns"] if r["scope"] == "span"
                                 else sum(g.d[e] for e in range(len(g))
                                          if g.mode[e] == cm.BUSY and g.owner[e] in zero))
        return res

    def test_random_traces(self):
        for seed in range(25):
            with self.subTest(seed=seed):
                self.check(random_trace(seed, n_ranks=2 + seed % 2))

    def test_path_never_exceeds_wall_where_the_span_level_score_did(self):
        # rank 1 arrives last at the reduce; the span-level DP valued the
        # path A1, R1, R0, B0 at 1210 on an 800 ns run
        t = _trace([work("A0", 1, 0, 300), mpi("MPI_Reduce", 1, 300, 700, type="reduce", rank=0, commid=0),
                    work("B0", 1, 700, 800),
                    work("A1", 2, 0, 690), mpi("MPI_Reduce", 2, 690, 710, type="reduce", rank=1, commid=0)])
        spans, preds = cp.build_dependency_graph(t)
        legacy = cp._compute_path_full(spans, preds)
        self.assertEqual(cp.attribute_blame(spans, legacy[0], 800, legacy[1], legacy[2]).dp_accounted_ns, 1210)
        res = self.check(t)
        self.assertEqual(res.covered_ns, 800)
        self.assertEqual(res.modeled_ns, 790)


# ── overlapping spans ──────────────────────────────────────────────────────

class TestOverlap(unittest.TestCase):
    def test_nested_call_owns_its_own_time_and_the_parent_only_the_rest(self):
        t = _trace([work("outer", 1, 0, 100), work("inner", 1, 20, 60)])
        res = model(t)
        self.assertEqual(res.covered_ns, 100)
        own = dict(zip(path_names(res), res.credited))
        self.assertEqual(own, {"outer": 60, "inner": 40})
        self.assertEqual([(r.mode, res.graph.name(r.span), r.own_ns) for r in res.rows],
                         [("busy", "outer", 20), ("busy", "inner", 40), ("busy", "outer", 40)])
        self.assertEqual(res.modeled_ns, 100)

    def test_identical_intervals_nest_whatever_their_arrival_order(self):
        # the child often arrives (ends) first
        t = _trace([work("child", 1, 0, 100), work("parent", 1, 0, 100)])
        res = model(t)
        self.assertEqual(res.graph.improper_nesting, 0)
        self.assertEqual(res.covered_ns, 100)

    def test_partial_overlap_on_a_thread_is_counted_once_and_reported(self):
        t = _trace([work("a", 1, 0, 100), work("b", 1, 50, 150)])
        res = model(t)
        self.assertEqual(res.covered_ns, 150)
        self.assertEqual(res.graph.improper_nesting, 1)
        self.assertEqual(sum(res.work_by_category.values()), 150)
        self.assertTrue(any("did not nest" in n for n in cp._model_notes(res, res.covered_ns)))

    def test_a_sync_call_overlapping_the_kernel_counts_only_its_tail(self):
        from tests.test_gpu_activity import build, concurrent_streams
        t = build(concurrent_streams())
        res = model(t)
        self.assertEqual(res.covered_ns, 4300)
        self.assertEqual(res.work_by_category, {"cuda": 4000})
        self.assertEqual(res.dependency_by_kind, {"launch": 200, "device_wait": 100})
        self.assertEqual(names(res, res.crit.on_all), ["_Z4bigv"])
        self.assertEqual(res.modeled_ns, 4000)

    def test_an_overlap_within_the_clock_tolerance_is_counted_once(self):
        from tests.test_gpu_activity import build, cupti_kernel, host
        # B starts 10 ns before A (same stream) ends: the stream-order edge is
        # kept (clock tolerance) with no time, and the walk's frontier trims
        # A's busy step so 1990..2000 counts once
        t = build([host("cudaLaunchKernel", 1000, 20, lid=1, corr=1, stream=1),
                   host("cudaLaunchKernel", 1030, 20, lid=2, corr=2, stream=1),
                   cupti_kernel(1, 7, 1100, 2000, name="A"),
                   cupti_kernel(2, 7, 1990, 2500, name="B")])
        res = model(t)
        g, w = res.graph, res.walk
        self.assertEqual([cm.KINDS[g.kind[e]] for e in w.steps], ["launch", "device_work", "stream_order",
                                                                   "device_work"])
        self.assertEqual(list(w.cover), [100, 890, 0, 510])
        self.assertEqual([g.d[e] for e in w.steps], [100, 900, 0, 510])
        self.assertEqual(res.covered_ns, 2500 - 1000)
        self.assertEqual(sum(res.work_by_category.values()) + sum(res.dependency_by_kind.values())
                         + res.device_idle_ns, res.covered_ns)

    def test_device_work_recorded_before_its_launch_is_moved_by_assembly(self):
        from tests.test_gpu_activity import build, cupti_kernel, host
        # 10 ns before the call's start: the device-clock correction moves
        # it to the call's start (core/gpu_activity), so no edge is clamped
        t = build([host("cudaLaunchKernel", 1000, 500, lid=1, corr=1, stream=1),
                   cupti_kernel(1, 7, 990, 3000),
                   host("cudaStreamSynchronize", 1600, 1500, cat="sync", typ="sync", op="sync", lid=2,
                        sync="stream", stream=1)])
        res = model(t)
        g, w = res.graph, res.walk
        self.assertEqual(list(w.cover), [g.d[e] for e in w.steps])
        self.assertEqual(res.covered_ns, 3100 - 1000)
        self.assertEqual(res.dependency_by_kind, {"launch": 0, "device_wait": 90})

    def test_sched_and_perf_sample_spans_are_no_nodes(self):
        t = _trace([work("a", 1, 0, 100),
                    SpanEvent("offcpu", Category.SCHED, 10, 50, 1, 1, tags={}),
                    SpanEvent("sample", Category.CPU, 20, 5, 1, 1, tags={})])
        res = model(t)
        self.assertEqual(path_names(res), ["a"])
        self.assertEqual(res.graph.improper_nesting, 0)


# ── start-gated dependencies ───────────────────────────────────────────────

class TestStartGated(unittest.TestCase):
    def test_an_early_posted_receive_waits_without_counting_its_wait_as_work(self):
        t = _trace([work("compute", 2, 0, 900), send(2, 900, 1, 0, 5),
                    recv(1, 0, 1000, 0, 1, 5), work("after", 1, 1000, 1100)])
        res = model(t)
        self.assertEqual(path_names(res), ["compute", "MPI_Send", "MPI_Recv", "after"])
        self.assertEqual(res.covered_ns, 1100)
        self.assertEqual(res.dependency_by_kind, {"p2p": 100})       # send start -> receive return
        self.assertEqual(res.work_by_category, {"openmp": 1000})
        self.assertNotIn("mpi", res.in_call_by_category)             # its first 900 ns are off the path
        self.assertEqual(res.modeled_ns, 1000)
        self.assertEqual(names(res, res.crit.on_all), ["after", "compute"])
        # the receive's own interval is required time nowhere
        self.assertNotIn(res.path_spans[2], res.crit.on_some)

    def test_a_receive_whose_message_was_there_first_stays_on_its_thread(self):
        t = _trace([send(2, 0, 1, 0, 5), work("compute", 1, 0, 500), recv(1, 500, 520, 0, 1, 5)])
        res = model(t)
        self.assertEqual(path_names(res), ["compute", "MPI_Recv"])
        self.assertEqual(res.in_call_by_category, {"mpi": 20})

    def test_last_arriver_releases_the_rendezvous(self):
        t = _trace([work("A0", 1, 0, 300), mpi("MPI_Allreduce", 1, 300, 720, type="allreduce", rank=0, commid=0),
                    work("A1", 2, 0, 690), mpi("MPI_Allreduce", 2, 690, 720, type="allreduce", rank=1, commid=0),
                    work("B0", 1, 720, 800)])
        res = model(t)
        self.assertEqual(path_names(res)[:2], ["A1", "MPI_Allreduce"])
        self.assertEqual(res.dependency_by_kind, {"arrival": 30})
        self.assertEqual(res.covered_ns, 800)
        self.assertEqual(res.graph_evidence["by_provenance"], {"call_order": 1})

    def test_parallel_region_fork_and_join(self):
        # the span-level graph dropped region -> task edges (end-gated); here
        # the region's start forks each worker task and its end joins them
        region = SpanEvent("parallel", Category.OPENMP, 0, 1000, 1, 1, tags={"type": "parallel_region"},
                           span_id="7")
        tasks = [SpanEvent("implicit_task", Category.OPENMP, a, b - a, 1, tid, tags={"type": "implicit_task"},
                           parent_span_id="7") for tid, a, b in ((1, 10, 400), (2, 20, 900), (3, 30, 500))]
        t = _trace([region, *tasks])
        res = model(t)
        g = res.graph
        kinds = [cm.KINDS[g.kind[e]] for e in res.walk.steps if g.mode[e] == cm.DEP]
        self.assertEqual(kinds, ["fork", "join"])
        self.assertEqual(res.covered_ns, 1000)
        self.assertEqual(res.dependency_by_kind, {"fork": 20, "join": 100})
        self.assertEqual(res.work_by_category, {"openmp": 880})
        worker = {g.t[2 * j]: j for j in range(g.n)}[20]
        self.assertEqual(res.crit.on_all, {worker})
        self.assertEqual(res.modeled_ns, 880)
        self.assertEqual(res.graph_evidence["by_kind"], {"fork": 2, "join": 2})


# ── ambiguous edges ─────────────────────────────────────────────────────────

class TestAmbiguity(unittest.TestCase):
    def test_an_inferred_edge_shows_in_the_direct_evidence_alternative(self):
        t = _trace([work("compute", 2, 0, 900), send(2, 900, 1, 0, 5),
                    recv(1, 0, 1000, 0, 1, 5), work("after", 1, 1000, 1100)])
        res = model(t)
        self.assertEqual(res.by_class, {"direct": 1000, "inferred": 100})
        alt = res.variants["direct_evidence_only"]
        self.assertFalse(alt["same_spans_as_default"])
        self.assertGreater(res.variant_only_spans, 0)
        self.assertEqual(res.variant_only_ns, 900)                     # rank 1's compute
        # without the message: rank 1's compute + send, unconnected to rank 0
        self.assertEqual(res.modeled_direct_only_ns, 910)
        self.assertEqual(res.modeled_ns, 1000)
        notes = cp._model_notes(res, res.covered_ns)
        self.assertTrue(any("not on it under every alternative" in n for n in notes))

    def test_unrecorded_time_as_work_or_idle(self):
        # rank 0's worker idles (untraced) until the fork; as work, that gap
        # would hold the task back instead of the fork
        region = SpanEvent("parallel", Category.OPENMP, 500, 500, 1, 1, tags={"type": "parallel_region"},
                           span_id="7")
        task = SpanEvent("implicit_task", Category.OPENMP, 520, 380, 1, 2, tags={"type": "implicit_task"},
                         parent_span_id="7")
        t = _trace([work("setup", 1, 0, 500), work("before", 1, 0, 100, tid=2), region, task])
        res = model(t)
        self.assertEqual(res.dependency_by_kind, {"fork": 20, "join": 100})
        # as work, the worker's 420 ns gap makes before + gap + task the
        # longest chain; as idle, setup + task is
        self.assertEqual(res.modeled_ns, 100 + 420 + 380)
        self.assertEqual(res.modeled_unrecorded_idle_ns, 500 + 380)
        alt = res.variants["unrecorded_as_work"]
        self.assertFalse(alt["same_spans_as_default"])                 # back through the gap to "before"

    def test_a_dependency_contradicting_the_timestamps_is_dropped_and_counted(self):
        t = _trace([work("a", 1, 0, 100), work("b", 2, 200, 300)])
        g0 = cp.load_or_build_graph(t)
        a, b = 0, 1
        # an explicit-id edge from b (starting at 200) to a (ending at 100)
        edges = np.array([(a, b, cp._KIND_CODE["p2p"], cp._CONF_CODE["medium"])], dtype=EDGE_DTYPE)
        res = cm.analyze_model(t, cp.Graph(t, g0.seqs, edges))
        self.assertEqual(res.graph_evidence["dropped_contradicting_time"], {"p2p": 1})
        self.assertTrue(any("contradict" in n for n in cp._model_notes(res, res.covered_ns)))
        self.assertEqual(path_names(res), ["b"])

    def test_rooted_collectives_are_flagged(self):
        t = _trace([mpi("MPI_Bcast", 1, 0, 100, type="bcast", rank=0, commid=0),
                    mpi("MPI_Bcast", 2, 50, 100, type="bcast", rank=1, commid=0)])
        res = model(t)
        self.assertEqual(res.graph_evidence["rooted_collective_edges"], 1)
        self.assertTrue(any("rooted collective" in n for n in cp._model_notes(res, res.covered_ns)))

    def test_call_order_grouping_and_its_overlap_fallback(self):
        def coll(pid, a, b):
            return mpi("MPI_Allreduce", pid, a, b, type="allreduce", rank=pid - 1, commid=4)
        # consistent: both ranks make two calls -> k-th with k-th, "high",
        # even where the intervals of different rounds overlap
        t = _trace([coll(1, 0, 100), coll(1, 100, 300), coll(2, 50, 120), coll(2, 120, 310)])
        confs = {c for _v, pl in cp.build_dependency_graph(t)[1].items() for _u, k, c in pl if k == "arrival"}
        self.assertEqual(confs, {"high"})
        # inconsistent: rank 2 lost a call -> overlap grouping, "medium"
        t = _trace([coll(1, 0, 100), coll(1, 100, 300), coll(2, 120, 310)])
        confs = {c for _v, pl in cp.build_dependency_graph(t)[1].items() for _u, k, c in pl if k == "arrival"}
        self.assertEqual(confs, {"medium"})


# ── gated and unmatched blocking calls ─────────────────────────────────────

class TestGatedAndUnmatched(unittest.TestCase):
    def modes(self, res, name):
        g = res.graph
        return {cm.MODES[g.mode[e]] for e in range(len(g)) if g.owner[e] >= 0 and g.name(g.owner[e]) == name}

    def test_an_unmatched_receive_is_unknown_not_work(self):
        res = model(_trace([work("a", 1, 0, 100), recv(1, 100, 300, 0, 1, 9)]))
        self.assertEqual(self.modes(res, "MPI_Recv"), {"unknown"})
        self.assertNotIn("mpi", res.work_by_category)
        self.assertEqual(res.unmatched_by_category, {"mpi": 200})
        self.assertEqual(res.covered_ns, 300)
        # counted as work only where unrecorded time is
        self.assertEqual(res.modeled_unrecorded_idle_ns, 100)
        self.assertEqual(res.modeled_ns, 300)
        notes = cp._model_notes(res, res.covered_ns)
        self.assertTrue(any("not counted as work" in n for n in notes), notes)
        rep = cp.analyze(_trace([work("a", 1, 0, 100), recv(1, 100, 300, 0, 1, 9)]))
        self.assertEqual(rep.path_unmatched_by_category, {"mpi": 200})
        self.assertEqual(sum(rep.time_on_path_by_category.values()) + sum(rep.wait_caused_by_category.values()),
                         rep.total_path_ns)

    def test_a_matched_receive_is_waiting_with_a_processing_tail(self):
        # the message's source event at 50: [50, 200] may be processing
        res = model(_trace([send(2, 50, 1, 0, 9), recv(1, 0, 200, 0, 1, 9)]))
        self.assertEqual(self.modes(res, "MPI_Recv"), {"wait"})
        g = res.graph
        tail = [g.whi[e] for e in range(len(g)) if g.mode[e] == cm.WAIT]
        self.assertEqual(tail, [150])
        self.assertEqual(res.modeled_ns, 10)                 # the send's own time
        self.assertEqual(res.modeled_gated_tail_ns, 150)
        self.assertEqual(res.variants["gated_tail_as_work"]["modeled_ns"], 150)
        self.assertEqual(res.modeled_range_ns, (10, 150))

    def test_the_tail_follows_the_releasing_dependency(self):
        # rank 2's work, then the message (send at 900, receive returns at
        # 1000), then solve: as processing, the receive's last 100 follow
        # the send, so the chain through the message is 900 + 100 + 600
        res = model(_trace([work("work", 2, 0, 900), send(2, 900, 1, 0, 9, end=950), work("post", 2, 1100, 1400),
                            work("prep", 1, 0, 200), recv(1, 200, 1000, 0, 1, 9), work("solve", 1, 1000, 1600)]))
        g = res.graph
        self.assertEqual([g.whi[e] for e in range(len(g)) if g.mode[e] == cm.DEP], [100])
        self.assertEqual(res.variants["default"]["modeled_ns"], 1500)
        self.assertEqual(res.variants["gated_tail_as_work"]["modeled_ns"], 1600)
        self.assertEqual(res.modeled_gated_tail_ns, 1600)
        self.assertEqual(res.modeled_range_ns, (1250, 1600))

    def test_the_tail_after_the_release_includes_nested_calls(self):
        # released at 100; after it: waiting 100..200, a nested call 200..250,
        # waiting 250..300 -- 200 of required time at most, on the lane and
        # on the message
        res = model(_trace([work("w", 2, 0, 100), send(2, 100, 1, 0, 9), recv(1, 0, 300, 0, 1, 9),
                            work("cb", 1, 200, 250)]))
        g = res.graph
        self.assertEqual([g.whi[e] for e in range(len(g)) if g.mode[e] == cm.DEP], [200])
        self.assertEqual(res.variants["gated_tail_as_work"]["modeled_ns"], 300)

    def test_a_message_there_first_leaves_the_whole_call_as_tail(self):
        res = model(_trace([send(2, 0, 1, 0, 9), recv(1, 50, 120, 0, 1, 9)]))
        g = res.graph
        self.assertEqual([g.whi[e] for e in range(len(g)) if g.mode[e] == cm.WAIT], [70])

    def test_the_last_arriver_of_a_rendezvous_is_released_by_itself(self):
        def coll(pid, a, b):
            return mpi("MPI_Allreduce", pid, a, b, type="allreduce", rank=pid - 1, commid=0)
        res = model(_trace([coll(1, 0, 100), coll(2, 80, 100)]))
        g = res.graph
        for name in ("MPI_Allreduce",):
            self.assertEqual(self.modes(res, name), {"wait"})
        self.assertEqual(sorted(g.whi[e] for e in range(len(g)) if g.mode[e] == cm.WAIT), [20, 20])
        self.assertEqual(res.unmatched_by_category, {})

    def test_variants_name_the_same_choices_for_walk_and_longest_path(self):
        res = model(random_trace(4))
        self.assertEqual(set(res.variants), set(cm.VARIANTS))
        for name, (gaps, direct, gated) in cm.VARIANTS.items():
            v = res.variants[name]
            self.assertEqual((v["gaps_as_work"], v["direct_evidence_only"], v["gated_tail_as_work"]),
                             (gaps, direct, gated))
            self.assertLessEqual(v["covered_ns"], res.variants["default"]["covered_ns"] + 10 ** 9)
        self.assertEqual(res.variants["default"]["modeled_ns"], res.modeled_unrecorded_idle_ns)
        self.assertEqual(res.variants["unrecorded_as_work"]["modeled_ns"], res.modeled_ns)


# ── multiple equal-length paths ────────────────────────────────────────────

class TestEqualPaths(unittest.TestCase):
    def region(self, ends):
        region = SpanEvent("parallel", Category.OPENMP, 500, 1000, 1, 1, tags={"type": "parallel_region"},
                           span_id="7")
        tasks = [SpanEvent(f"task{tid}", Category.OPENMP, 520, end - 520, 1, tid,
                           tags={"type": "implicit_task"}, parent_span_id="7") for tid, end in zip((2, 3), ends)]
        return _trace([work("serial", 1, 0, 500), region, *tasks])

    def test_two_equal_tasks_are_on_some_not_every_longest_path(self):
        res = model(self.region((1400, 1400)))
        g, c = res.graph, res.crit
        self.assertEqual(names(res, c.on_some), ["serial", "task2", "task3"])
        self.assertEqual(names(res, c.on_all), ["serial"])
        self.assertEqual(c.work_on_all_ns, 500)
        self.assertEqual(c.work_on_some_ns, 500 + 2 * 880)
        self.assertEqual(res.modeled_ns, 500 + 880)
        self.assertEqual(res.walk.ties, 1)                     # both joins release the region's end at once
        self.assertEqual(res.walk.tie_samples[0][1], ("join",))
        self.assertTrue(any("equally consistent alternative" in n for n in cp._model_notes(res, res.covered_ns)))
        # a counterfactual on one task: the other one takes over
        by = {(r["scope"], r["name"]): r for r in res.sensitivity}
        self.assertEqual(by[("span", "task2")]["modeled_reduction_ns"], 0)
        self.assertFalse(by[("span", "task2")]["on_every_longest_path"])
        self.assertEqual(by[("span", "serial")]["modeled_reduction_ns"], 500)
        self.assertEqual(by[("name", "task2")]["modeled_reduction_ns"], 0)      # one span of that name
        self.assertEqual(c.slack[[j for j in range(g.n) if g.name(j) == "task3"][0]], 0)

    def test_a_shorter_task_has_slack_and_no_reduction(self):
        res = model(self.region((1400, 1200)))
        g, c = res.graph, res.crit
        t3 = [j for j in range(g.n) if g.name(j) == "task3"][0]
        self.assertEqual(c.slack[t3], 200)
        self.assertEqual(names(res, c.on_all), ["serial", "task2"])
        self.assertEqual(res.walk.ties, 0)

    def test_two_ranks_ending_at_once_is_a_tie(self):
        t = _trace([work("a", 1, 0, 100), work("b", 2, 0, 100)])
        res = model(t)
        self.assertEqual(res.walk.ties, 1)
        self.assertEqual(res.walk.tie_samples, [(100, ("end of trace",))])
        self.assertEqual(names(res, res.crit.on_some), ["a", "b"])
        self.assertEqual(res.crit.on_all, set())

    def test_name_scope_counterfactual_removes_every_alternative(self):
        t = _trace([work("solve", 1, 0, 100), work("solve", 2, 0, 100), work("io", 3, 0, 40)])
        res = model(t)
        by = {(r["scope"], r["name"]): r for r in res.sensitivity}
        self.assertEqual(by[("name", "solve")]["modeled_reduction_ns"], 60)   # io becomes the longest
        self.assertEqual(by[("span", "solve")]["modeled_reduction_ns"], 0)


# ── evidence ────────────────────────────────────────────────────────────────

class TestEvidence(unittest.TestCase):
    def test_classes_follow_provenance_and_tier(self):
        self.assertEqual(cm.edge_class("correlation_id", "certain"), "direct")
        self.assertEqual(cm.edge_class("sync_target", "medium"), "inferred")
        self.assertEqual(cm.edge_class("posting_order", "high"), "inferred")
        self.assertEqual(cm.edge_class("nccl_seq", "high"), "direct")
        for prov in cm.PROVENANCE:
            for conf in cm.CONFS:
                self.assertTrue(cm.edge_reason("p2p", prov, conf))
        self.assertIn("lost events", cm.edge_reason("p2p", "posting_order", "low"))

    def test_point_to_point_provenance(self):
        resolved = _trace([
            send(2, 20, 1, 0, 9),                  # after the Wait began: the message binds
            mpi("MPI_Irecv", 1, 0, 1, sid="r1", type="irecv", rank=0, peer=-2, tag=-1, commid=0, wildcard=1),
            mpi("MPI_Wait", 1, 5, 40, psid="r1", type="wait", rank=0, rpeer=1, rtag=9, rwpeer=1)])
        res = model(resolved)
        self.assertEqual(res.graph_evidence["by_provenance"], {"status_resolved": 1})
        step = res.dependency_steps()[0]
        self.assertEqual((step["kind"], step["runtime"], step["tier"], step["class"]),
                         ("p2p", "mpi", "high", "inferred"))
        self.assertIn("MPI_Status", step["reason"])
        plain = model(_trace([send(2, 10, 1, 0, 9), recv(1, 0, 40, 0, 1, 9)]))
        self.assertEqual(plain.graph_evidence["by_provenance"], {"posting_order": 1})
        self.assertEqual(plain.dependency_steps()[0]["tier"], "medium")

    def test_gpu_provenance_is_direct(self):
        from tests.test_gpu_activity import build, concurrent_streams
        res = model(build(concurrent_streams()))
        ev = res.graph_evidence
        self.assertEqual(set(ev["by_provenance"]), {"correlation_id", "sync_target"})
        self.assertEqual(set(ev["by_class"]), {"direct"})
        self.assertEqual({d["runtime"] for d in res.dependency_steps()}, {"cuda"})


# ── compatibility ───────────────────────────────────────────────────────────

class TestCompatibility(unittest.TestCase):
    def test_edges_persisted_by_cp4_are_rebuilt(self):
        t = _trace([work("a", 1, 0, 100), work("b", 2, 0, 50), work("c", 1, 100, 200)])
        t.store.save_edges("cp-4", np.empty(0, dtype=EDGE_DTYPE))
        self.assertNotEqual(cp.edges_version(), "cp-4")
        r = cp.analyze(t)
        self.assertIsNotNone(t.store.load_edges(cp.edges_version()))
        self.assertEqual(r.total_path_ns, 200)

    def test_sensitivity_is_bounded(self):
        t = random_trace(3)
        self.assertEqual(model(t, sensitivity=0).sensitivity, [])
        res = cm.analyze_model(t, cp.load_or_build_graph(t), sensitivity_node_limit=4)
        self.assertEqual(res.sensitivity, [])
        self.assertIn("skipped", res.sensitivity_note)
        self.assertTrue(model(t, sensitivity=2).sensitivity)

    def test_empty_and_annotation_only_traces(self):
        self.assertEqual(cp.analyze(_trace([])).total_path_ns, 0)
        t = _trace([SpanEvent("range", Category.NVTX, 0, 100, 1, 1, tags={"type": "nvtx_range"})])
        r = cp.analyze(t)
        self.assertEqual((r.total_path_ns, r.path_span_indices, r.modeled_path_ns), (0, [], 0))


if __name__ == "__main__":
    unittest.main()
