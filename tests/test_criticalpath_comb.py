"""
Critical-path defects found with LLNL Comb (MPI + OpenMP + CUDA, NVTX phase
ranges), on synthetic two-rank traces:

  * annotation ranges (NVTX/ROCTX) are no nodes of the dependency graph or
    of the DP: a closed top-level range used to become the program-order
    predecessor of everything after it (and the psid= parent of the calls
    inside it), and the chain of enclosing ranges -- each as long as all
    its children -- hid the program's own cross-runtime path. With them,
    the path is the one of the same trace without them; it runs through
    p2p, launch and device_wait hops; no path time is on `nvtx`; both
    builders agree; edges persisted by the previous builder version are
    rebuilt; the ranges stay available as display (path_time_by_annotation);
  * the path's length is the time it covers: on the event-point model
    (cp_model.py) every step covers [source event, destination event], so a
    start-gated hop whose successor started before its predecessor -- or an
    earlier path span on another rank -- ended counts that wall time once.
    total <= extent <= wall, credited + gaps = total, the composition sums
    to total, in an arrival hop, a p2p hop into a completion that started
    before the send, and a ping-pong zig-zag.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis import criticalpath as cp
from src.core.events import Category, SpanEvent
from src.core.trace import Trace, TraceMetadata
from tests.test_gpu_activity import _line, build, cupti_kernel, host

TID = 100


# ── a Comb-like cycle: NVTX phases around pack kernel / stream sync / MPI ───

CYCLE = 10_000_000


def _rank(r, c, kernel_end, send_at, wait, unpack_end, nvtx=True):
    """Rank r (pid r+1), cycle c (offset c * CYCLE): post a receive, pack
    (launch + kernel + stream sync), send, wait for both requests, unpack
    (launch + kernel + sync), all on one thread -- inside a top-level NVTX
    range per cycle with phase ranges in it, as Comb's, when nvtx."""
    pid, peer, o = r + 1, 1 - r, c * CYCLE
    nv = (pid << 32) | (900 + 10 * c)
    lid = 10 * c
    mpi = dict(rank=r, peer=peer, wpeer=peer, tag=7 + c, commid=0)
    ln = []
    if nvtx:
        ln += [_line("nvtx", o, CYCLE, "Comm mpi Mesh cuda Buffers cuda", {"type": "nvtx_range", "sid": nv},
                     pid, TID),
               _line("nvtx", o + 40_000, send_at + 20_000 - 40_000, "post-send",
                     {"type": "nvtx_range", "sid": nv + 1, "psid": nv}, pid, TID),
               _line("nvtx", o + wait[0], unpack_end + 30_000 - wait[0], "wait-recv",
                     {"type": "nvtx_range", "sid": nv + 2, "psid": nv}, pid, TID)]
    psid = {"psid": nv + 1} if nvtx else {}
    ln += [_line("mpi", o + 50_000, 10_000, "MPI_Irecv", dict(type="irecv", sid=f"r{r}.{c}", **mpi), pid, TID),
           host("cudaLaunchKernel", o + 100_000, 10_000, lid=lid + 1, corr=lid + 1, stream=1, pid=pid, tid=TID,
                **psid),
           cupti_kernel(lid + 1, 7, o + 120_000, o + kernel_end, name="_Z4packv", pid=pid),
           host("cudaStreamSynchronize", o + 115_000, kernel_end + 10_000 - 115_000, cat="sync", typ="sync",
                op="sync", lid=lid + 2, corr=lid + 2, stream=1, sync="stream", pid=pid, tid=TID, **psid),
           _line("mpi", o + send_at, 10_000, "MPI_Isend", dict(type="isend", sid=f"s{r}.{c}", **mpi), pid, TID),
           _line("mpi", o + wait[0], wait[1] - wait[0], "MPI_Waitall",
                 dict(type="waitall", rank=r, psid=f"r{r}.{c};s{r}.{c}"), pid, TID),
           host("cudaLaunchKernel", o + wait[1] + 10_000, 10_000, lid=lid + 3, corr=lid + 3, stream=1, pid=pid,
                tid=TID),
           cupti_kernel(lid + 3, 7, o + wait[1] + 30_000, o + unpack_end, name="_Z6unpackv", pid=pid),
           host("cudaStreamSynchronize", o + wait[1] + 25_000, unpack_end + 10_000 - wait[1] - 25_000, cat="sync",
                typ="sync", op="sync", lid=lid + 4, corr=lid + 4, stream=1, sync="stream", pid=pid, tid=TID)]
    return ln


def comb_cycle(nvtx=True, cycles=2):
    # rank 1 packs for longer, so rank 0's Waitall waits for rank 1's send;
    # back-to-back cycles: a closed top-level range used to be the only
    # program-order exit into the next one
    lines = []
    for c in range(cycles):
        lines += _rank(0, c, 1_000_000, 1_020_000, (3_000_000, 5_600_000), 8_000_000, nvtx)
        lines += _rank(1, c, 5_000_000, 5_520_000, (5_540_000, 5_600_000), 6_000_000, nvtx)
    return build(lines)


def _path(trace):
    r = cp.analyze(trace)
    return r, [(r.spans[i].pid, r.spans[i].name, r.spans[i].start_ns) for i in r.path_span_indices]


class TestAnnotationsAreNoNodes(unittest.TestCase):
    def test_path_runs_through_the_program_not_its_annotations(self):
        t = comb_cycle()
        r, path = _path(t)
        self.assertTrue(path)
        self.assertFalse([p for p in path if p[1] in ("post-send", "wait-recv") or p[1].startswith("Comm")])
        self.assertNotIn("nvtx", r.time_on_path_by_category)
        for kind in ("p2p", "launch", "device_wait"):
            self.assertIn(kind, r.path_edge_kinds)
        # the cross-rank chain: rank 1's pack -> sync -> Isend -p2p-> rank 0's Waitall
        hops = [i for i, k in enumerate(r.path_edge_kinds) if k == "p2p"]
        self.assertTrue(hops)
        for hop in hops:
            self.assertEqual((path[hop][:2], path[hop + 1][:2]), ((2, "MPI_Isend"), (1, "MPI_Waitall")))

    def test_same_path_as_the_trace_without_annotations(self):
        self.assertEqual(_path(comb_cycle())[1], _path(comb_cycle(nvtx=False))[1])
        self.assertEqual(_path(comb_cycle())[0].total_path_ns, _path(comb_cycle(nvtx=False))[0].total_path_ns)

    def test_no_edge_touches_an_annotation_in_either_builder(self):
        t = comb_cycle()
        spans, preds = cp.build_dependency_graph(t)
        nv = {i for i, s in enumerate(spans) if s.category.value == "nvtx"}
        self.assertEqual(len(nv), 12)
        self.assertFalse([(u, v) for v, pl in preds.items() for u, _k, _c in pl if u in nv or v in nv])
        edges = cp.build_edges(t)
        self.assertFalse([1 for d, s_ in zip(edges["dst"], edges["src"]) if int(d) in nv or int(s_) in nv])

    def test_full_and_streaming_builders_agree(self):
        # same edges, and the same span-level (legacy) DP result over them
        t = comb_cycle()
        spans, preds = cp.build_dependency_graph(t)
        graph = cp.load_or_build_graph(t)
        self.assertEqual(graph.preds_dict(), {k: v for k, v in preds.items() if v})
        full = cp._compute_path_full(spans, preds)
        self.assertEqual(tuple(map(list, full)), tuple(map(list, graph.critical_path())))

    def test_an_annotation_never_ends_the_path(self):
        # a range longer than any chain of work, with no edges at all
        t = Trace(TraceMetadata())
        t.add(SpanEvent(name="whole run", category=Category.NVTX, start_ns=0, duration_ns=10_000, pid=1, tid=1,
                        tags={"type": "nvtx_range"}, span_id="9"))
        t.add(SpanEvent(name="work", category=Category.OPENMP, start_ns=100, duration_ns=50, pid=1, tid=1,
                        tags={"type": "parallel_region"}))
        r, path = _path(t)
        self.assertEqual([p[1] for p in path], ["work"])
        spans, preds = cp.build_dependency_graph(t)
        self.assertEqual([spans[i].name for i in cp._compute_path_full(spans, preds)[0]], ["work"])
        # and a roctx range, by its type (category "other")
        t.add(SpanEvent(name="roctx", category=Category.OTHER, start_ns=0, duration_ns=20_000, pid=1, tid=1,
                        tags={"type": "roctx_range"}))
        self.assertEqual([p[1] for p in _path(t)[1]], ["work"])

    def test_edges_of_the_previous_builder_version_are_rebuilt(self):
        import numpy as np
        from src.core.store import EDGE_DTYPE
        t = comb_cycle()
        spans, _preds = cp.build_dependency_graph(t)
        nv = next(i for i, s in enumerate(spans) if s.name == "post-send")
        work = next(i for i, s in enumerate(spans) if s.name == "MPI_Isend")
        stale = np.array([(work, nv, cp._KIND_CODE["sequential"], cp._CONF_CODE["certain"])], dtype=EDGE_DTYPE)
        t.store.save_edges("cp-2", stale)                     # what the old builder persisted
        self.assertEqual(t.store.load_edges("cp-2").tolist(), stale.tolist())
        g = cp.load_or_build_graph(t)
        self.assertNotEqual(cp.EDGES_VERSION, "cp-2")
        self.assertFalse([1 for s_ in g.edges["src"] if int(s_) == nv])
        self.assertIsNotNone(t.store.load_edges(cp.edges_version()))

    def test_persisted_edges_are_reused_while_the_store_is_unchanged(self):
        from unittest import mock
        t = comb_cycle()
        first = cp.load_or_build_graph(t).edges
        with mock.patch.object(cp, "build_edges", side_effect=AssertionError("rebuilt")):
            again = cp.load_or_build_graph(t).edges
        self.assertEqual(again.tolist(), first.tolist())

    def test_ranges_stay_available_for_display(self):
        r, _path_ = _path(comb_cycle())
        self.assertGreater(r.path_time_by_annotation.get("wait-recv", 0), 0)
        self.assertGreater(r.path_time_by_annotation.get("post-send", 0), 0)
        outer = r.path_time_by_annotation["Comm mpi Mesh cuda Buffers cuda"]
        self.assertLessEqual(outer, r.total_path_ns)
        self.assertNotIn("nvtx", r.time_on_path_by_category)


# ── path length = the time the path covers ─────────────────────────────────

def _mpi(name, pid, start, end, sid="", psid="", **tags):
    return SpanEvent(name=name, category=Category.MPI, start_ns=start, duration_ns=end - start, pid=pid, tid=pid,
                     tags={k: str(v) for k, v in tags.items()}, span_id=sid, parent_span_id=psid)


def _work(name, pid, start, end):
    return SpanEvent(name=name, category=Category.OPENMP, start_ns=start, duration_ns=end - start, pid=pid,
                     tid=pid, tags={"type": "parallel_region"})


def _trace(spans):
    t = Trace(TraceMetadata())
    for s in spans:
        t.add(s)
    return t


def _p2p(src_pid, dst_pid, send_at, post_at, wait, tag, sid):
    """src sends (Isend + Wait), dst posted its receive at post_at and
    waits over `wait`."""
    a, b = src_pid - 1, dst_pid - 1
    return [_mpi("MPI_Isend", src_pid, send_at, send_at + 10, sid=f"s{sid}", type="isend", rank=a, peer=b,
                 wpeer=b, tag=tag, commid=0),
            _mpi("MPI_Irecv", dst_pid, post_at, post_at + 5, sid=f"r{sid}", type="irecv", rank=b, peer=a,
                 wpeer=a, tag=tag, commid=0),
            _mpi("MPI_Wait", dst_pid, wait[0], wait[1], psid=f"r{sid}", type="wait", rank=b)]


class TestPathCoversEachInstantOnce(unittest.TestCase):
    def _check(self, t, expect_kind, expect_total=None):
        r = cp.analyze(t)
        spans = [r.spans[i] for i in r.path_span_indices]
        extent = max(s.end_ns for s in spans) - spans[0].start_ns
        self.assertIn(expect_kind, r.path_edge_kinds)
        self.assertLessEqual(r.total_path_ns, extent)
        self.assertLessEqual(extent, r.wall_ns)
        self.assertEqual(sum(r.path_credited_ns) + sum(r.path_gap_ns), r.total_path_ns)
        self.assertEqual(sum(r.time_on_path_by_category.values()) + sum(r.wait_caused_by_category.values()),
                         r.total_path_ns)
        self.assertEqual(sum(r.path_work_by_category.values()) + sum(r.path_in_call_by_category.values())
                         + r.path_unrecorded_ns + r.path_device_idle_ns
                         + sum(r.path_dependency_by_kind.values()), r.total_path_ns)
        self.assertEqual(r.dp_accounted_ns, r.total_path_ns)
        self.assertLessEqual(r.modeled_path_ns, r.wall_ns)
        if expect_total is not None:
            self.assertEqual(r.total_path_ns, expect_total)
        return r

    def test_arrival_hop_whose_successor_started_before_its_predecessor_ended(self):
        # rank 1 arrives last at the reduce; rank 0's reduce started 390 ns
        # earlier and ends before rank 1's does -- the path A1, R1, R0, B0
        # covers 0..800, it used to count 1210
        t = _trace([_work("A0", 1, 0, 300), _mpi("MPI_Reduce", 1, 300, 700, type="reduce", rank=0, commid=0),
                    _work("B0", 1, 700, 800),
                    _work("A1", 2, 0, 690), _mpi("MPI_Reduce", 2, 690, 710, type="reduce", rank=1, commid=0)])
        r = self._check(t, "arrival", expect_total=800)
        self.assertEqual([r.spans[i].name for i in r.path_span_indices], ["A1", "MPI_Reduce", "MPI_Reduce", "B0"])
        # A1's work, rank 1's arrival releasing rank 0's reduce 10 ns later
        # (dependency latency), B0's work; the span-level DP valued this
        # path at 1210
        self.assertEqual(r.path_work_by_category, {"openmp": 690 + 100})
        self.assertEqual(r.path_dependency_by_kind, {"arrival": 10})
        self.assertEqual(r.modeled_path_ns, 790)

    def test_p2p_hop_into_a_completion_that_started_before_the_send(self):
        # rank 0 waits from 100; rank 1 sends at 600 after computing
        t = _trace([_work("compute", 2, 0, 600), *_p2p(2, 1, 600, 50, (100, 650), 1, 1),
                    _work("after", 1, 650, 700)])
        r = self._check(t, "p2p", expect_total=700)
        self.assertEqual([r.spans[i].name for i in r.path_span_indices], ["compute", "MPI_Isend", "MPI_Wait", "after"])
        # the Wait returned 50 ns after the send began: dependency latency
        # (the trace cannot split it into the Isend call, the transfer and
        # the Wait's own processing) -- not 10 + 40 credited to the calls
        self.assertEqual(r.path_credited_ns, [600, 0, 0, 50])
        self.assertEqual(r.path_gap_ns, [0, 0, 50, 0])
        self.assertEqual(r.path_dependency_by_kind, {"p2p": 50})

    def test_ping_pong_zig_zag(self):
        # each rank waits for the other's reply with its receive posted (and
        # the wait begun) long before: 0 -> 1 -> 0 -> 1
        t = _trace([_work("w0", 1, 0, 100), *_p2p(1, 2, 100, 0, (10, 150), 1, 1),
                    _work("w1", 2, 150, 300), *_p2p(2, 1, 300, 120, (130, 320), 2, 2),
                    _work("w0b", 1, 320, 400), *_p2p(1, 2, 400, 310, (315, 450), 3, 3),
                    _work("end", 2, 450, 500)])
        r = self._check(t, "p2p")
        self.assertLessEqual(r.total_path_ns, 500)
        self.assertGreaterEqual(r.path_edge_kinds.count("p2p"), 2)

    def test_a_sync_calls_tail_after_the_work_is_device_wait_latency(self):
        # the sync call overlaps the work it waited for: only its tail after
        # the work ended is on the path, as the device_wait step
        t = comb_cycle(nvtx=False)
        r = cp.analyze(t)
        hops = 0
        for i, k in enumerate(r.path_edge_kinds):
            if k == "device_wait":
                hops += 1
                sync, work = r.spans[r.path_span_indices[i + 1]], r.spans[r.path_span_indices[i]]
                # the sync call's tail after the work ended: device_wait latency
                self.assertEqual(r.path_gap_ns[i + 1], max(0, sync.end_ns - work.end_ns))
                self.assertEqual(r.path_credited_ns[i + 1], 0)
        self.assertTrue(hops)
        self.assertLessEqual(r.total_path_ns, r.wall_ns)


if __name__ == "__main__":
    unittest.main()
