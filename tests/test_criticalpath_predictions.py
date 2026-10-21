"""
Dependency-graph defects predicted from reading src/analysis/criticalpath.py
(benchmark harness planning documents), each reproduced on a synthetic
trace and checked through both edge builders (build_dependency_graph and
the streamed build_edges must agree):

  1. streams are keyed without the device: the default (NULL) stream of two
     GPUs is one stream, so their kernels were chained as "certain"
     sequential work; a device-wide synchronize waited for every device's
     streams
  2. kernels of one CUDA graph launch were chained as "certain" same-stream
     work although the graph may run them concurrently
  3. the OpenMP rendezvous treated omp_critical_wait / omp_taskwait /
     omp_taskgroup / omp_reduction as barriers and grouped barriers of
     different teams (parallel regions) together
  4. after event loss, order-matched MPI messages kept their tiers although
     a lost send shifts every later pairing on that channel
  5. NCCL collectives were grouped by operation type only, so concurrent
     collectives on different communicators formed one rendezvous

The GPU and NCCL cases are synthetic only: no usable CUDA/ROCm/NCCL here.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analysis import criticalpath as cp
from src.core.events import Category, SpanEvent
from src.core.trace import Trace, TraceMetadata
from tests.test_gpu_activity import build, cupti_kernel, host, _line


def edges(trace):
    """[(src span, dst span, kind, confidence)] from both builders."""
    spans, preds = cp.build_dependency_graph(trace)
    mem = sorted((u, v, k, c) for v, pl in preds.items() for u, k, c in pl)
    g = cp.Graph(trace, trace.store.span_seqs(), cp.build_edges(trace))
    streamed = sorted((u, v, k, c) for v, pl in g.preds_dict().items() for u, k, c in pl)
    assert mem == streamed, (mem, streamed)
    return [(spans[u], spans[v], k, c) for u, v, k, c in mem]


def _span(name, cat, start, dur, pid=1, tid=1, psid="", sid="", **tags):
    return SpanEvent(name=name, category=Category(cat), start_ns=start, duration_ns=dur, pid=pid, tid=tid,
                     tags={k: str(v) for k, v in tags.items()}, parent_span_id=psid, span_id=sid)


def _trace(spans, health=None):
    t = Trace(TraceMetadata())
    if health:
        t.metadata.capture_health = health
    for s in spans:
        t.add(s)
    return t


def sync_host(api, start, dur, lid, kind, hdev=None, **extra):
    tags = dict(extra)
    if hdev is not None:
        tags["hdev"] = hdev
    return host(api, start, dur, typ="sync", op="sync", lid=lid, sync=kind, **tags)


class TestStreamsHaveADevice(unittest.TestCase):
    def _two_gpus(self, with_hdev=True):
        h0 = {"hdev": 0} if with_hdev else {}
        h1 = {"hdev": 1} if with_hdev else {}
        return [
            host("cudaLaunchKernel", 1000, 20, lid=1, corr=1, stream=0, **h0),
            host("cudaLaunchKernel", 1100, 20, lid=2, corr=2, stream=0, **h1),
            cupti_kernel(1, 7, 2000, 9000, name="k_dev0", dev=0),
            cupti_kernel(2, 7, 2100, 9100, name="k_dev1", dev=1),
        ]

    def test_default_streams_of_two_devices_are_not_chained(self):
        e = edges(build(self._two_gpus()))
        seq = [(s.name, d.name) for s, d, k, _c in e if k == "sequential"]
        self.assertNotIn(("k_dev0", "k_dev1"), seq)
        # native records alone (no hdev= from the hook) still name the device
        e = edges(build(self._two_gpus(with_hdev=False)))
        self.assertNotIn(("k_dev0", "k_dev1"), [(s.name, d.name) for s, d, k, _c in e if k == "sequential"])

    def test_device_sync_waits_for_its_own_device_only(self):
        lines = self._two_gpus() + [sync_host("cudaDeviceSynchronize", 9500, 100, lid=3, kind="device", hdev=0)]
        waits = [(s.name, c) for s, d, k, c in edges(build(lines)) if k == "device_wait"]
        self.assertEqual(waits, [("k_dev0", "certain")])

    def test_device_sync_of_unknown_device_is_not_certain(self):
        # an older hook: no hdev= on the sync -- either device may be meant
        lines = self._two_gpus(with_hdev=False) + [
            host("cudaDeviceSynchronize", 9500, 100, typ="sync", op="sync", lid=3, sync="device")]
        waits = sorted((s.name, c) for s, d, k, c in edges(build(lines)) if k == "device_wait")
        self.assertEqual(waits, [("k_dev0", "medium"), ("k_dev1", "medium")])

    def test_single_device_sync_stays_certain(self):
        lines = [host("cudaLaunchKernel", 1000, 20, lid=1, corr=1, stream=0),
                 cupti_kernel(1, 7, 2000, 9000, name="k", dev=0),
                 host("cudaDeviceSynchronize", 9500, 100, typ="sync", op="sync", lid=3, sync="device")]
        waits = [(s.name, c) for s, d, k, c in edges(build(lines)) if k == "device_wait"]
        self.assertEqual(waits, [("k", "certain")])


class TestGraphKernels(unittest.TestCase):
    def test_graph_kernels_are_not_chained(self):
        lines = [
            host("cudaLaunchKernel", 100, 10, lid=1, corr=1, stream=5),
            cupti_kernel(1, 9, 200, 400, name="before"),
            host("cudaGraphLaunch", 500, 10, op="graph", lid=2, corr=2, stream=5),
            cupti_kernel(2, 9, 600, 900, name="g_a", graph=1),
            cupti_kernel(2, 9, 650, 950, name="g_b", graph=1),     # concurrent branch
            cupti_kernel(2, 9, 960, 990, name="g_c", graph=1),
            host("cudaLaunchKernel", 1000, 10, lid=3, corr=3, stream=5),
            cupti_kernel(3, 9, 1100, 1200, name="after"),
        ]
        seq = {(s.name, d.name) for s, d, k, _c in edges(build(lines)) if k == "sequential"}
        graph = {"g_a", "g_b", "g_c"}
        self.assertFalse({(a, b) for a, b in seq if a in graph and b in graph}, seq)
        # the stream still orders the graph after earlier work, and later work
        # after the graph -- after its last kernel to finish
        for g in graph:
            self.assertIn(("before", g), seq)
        self.assertEqual({a for a, b in seq if b == "after"}, {"g_c"})


class TestOpenMPRendezvous(unittest.TestCase):
    def _arrivals(self, spans):
        return {(s.tid, d.tid, c) for s, d, k, c in edges(_trace(spans)) if k == "arrival"}

    def test_non_barrier_sync_regions_are_no_rendezvous(self):
        for name in ("omp_critical_wait", "omp_taskwait", "omp_taskgroup", "omp_reduction"):
            spans = [_span(name, "sync", 100, 500, tid=1, psid="7"),
                     _span(name, "sync", 200, 300, tid=2, psid="7")]
            self.assertEqual(self._arrivals(spans), set(), name)

    def test_barriers_still_rendezvous(self):
        spans = [_span("omp_barrier_implicit", "sync", 100, 500, tid=1, psid="7"),
                 _span("omp_barrier_implicit", "sync", 400, 200, tid=2, psid="7")]
        self.assertEqual(self._arrivals(spans), {(2, 1, "certain")})

    def test_barriers_of_different_teams_are_separate(self):
        # two concurrent parallel regions (psid 7: threads 1-2, psid 8: 3-4)
        spans = [_span("omp_barrier_implicit", "sync", 100, 500, tid=1, psid="7"),
                 _span("omp_barrier_implicit", "sync", 300, 300, tid=2, psid="7"),
                 _span("omp_barrier_implicit", "sync", 150, 600, tid=3, psid="8"),
                 _span("omp_barrier_implicit", "sync", 700, 50, tid=4, psid="8")]
        self.assertEqual(self._arrivals(spans), {(2, 1, "certain"), (4, 3, "certain")})

    def test_gnu_libgomp_team_ids_separate_teams(self):
        spans = [_span("omp_barrier", "sync", 100, 500, tid=1, team=5),
                 _span("omp_barrier", "sync", 300, 300, tid=2, team=5),
                 _span("omp_barrier", "sync", 150, 600, tid=3, team=6),
                 _span("omp_barrier", "sync", 700, 50, tid=4, team=6)]
        self.assertEqual(self._arrivals(spans), {(2, 1, "certain"), (4, 3, "certain")})

    def test_barriers_without_a_team_id_keep_the_per_process_grouping(self):
        # traces from hooks before team= / psid=: one group per name, as before
        spans = [_span("omp_barrier", "sync", 100, 500, tid=1),
                 _span("omp_barrier", "sync", 400, 200, tid=2)]
        self.assertEqual(self._arrivals(spans), {(2, 1, "certain")})


class TestEventLoss(unittest.TestCase):
    def _sends_recvs(self, lost_first_send):
        sp = []
        for k in range(3):
            if not (lost_first_send and k == 0):
                sp.append(_span("MPI_Send", "mpi", 100 + 1000 * k, 10, pid=10, tid=10, type="send", rank=0,
                                peer=1, wpeer=1, tag=5, commid=0))
            sp.append(_span("MPI_Recv", "mpi", 50 + 1000 * k, 200, pid=11, tid=11, type="recv", rank=1,
                            peer=0, wpeer=0, tag=5, commid=0))
        return sp

    def test_order_matched_edges_after_loss_are_low(self):
        health = {"transport": {"10/mpi": {"final": 1, "emitted": 4, "sent": 3, "received": 3,
                                           "dropped_full": 1}}}
        t = _trace(self._sends_recvs(True), health)
        p2p = [c for _s, _d, k, c in edges(t) if k == "p2p"]
        self.assertEqual(p2p, ["low", "low"])
        diag = cp.mpi_dependency_diagnostics(t)
        self.assertEqual(diag.get("matched_after_loss"), 2)
        self.assertTrue(any("lost" in w and "order" in w for w in cp.dependency_warnings(diag)))

    def test_clean_capture_keeps_tiers(self):
        t = _trace(self._sends_recvs(False), {"transport": {"10/mpi": {"final": 1, "sent": 3, "received": 3}}})
        self.assertEqual([c for _s, _d, k, c in edges(t) if k == "p2p"], ["medium"] * 3)
        self.assertNotIn("matched_after_loss", cp.mpi_dependency_diagnostics(t))

    def test_loss_in_an_unrelated_hook_keeps_tiers(self):
        t = _trace(self._sends_recvs(False), {"transport": {"10/ompt": {"dropped_full": 5}}})
        self.assertEqual([c for _s, _d, k, c in edges(t) if k == "p2p"], ["medium"] * 3)

    def test_loss_in_a_process_not_in_the_trace_downgrades_all(self):
        # e.g. merged traces whose pids were renumbered
        t = _trace(self._sends_recvs(False), {"transport": {"999/mpi": {"dropped_lost": 2}}})
        self.assertEqual([c for _s, _d, k, c in edges(t) if k == "p2p"], ["low"] * 3)


class TestGpuEventLoss(unittest.TestCase):
    def _lines(self):
        return [host("cudaLaunchKernel", 100, 10, lid=1, corr=1, stream=5, hdev=0),
                cupti_kernel(1, 9, 200, 800, name="k", dev=0),
                host("cudaEventRecord", 120, 5, typ="event_record", op="event_record", lid=2, event=9,
                     stream=5, hdev=0),
                host("cudaEventSynchronize", 300, 600, typ="sync", op="sync", lid=3, sync="event", event=9,
                     hdev=0)]

    def _waits(self, trace):
        return [(s.name, c) for s, d, k, c in edges(trace) if k == "device_wait"]

    def test_event_sync_after_cuda_loss_is_low(self):
        trace = build(self._lines())
        self.assertEqual(self._waits(trace), [("k", "certain")])
        trace.metadata.capture_health = {"transport": {"1/cuda": {"dropped_full": 3}}}
        self.assertEqual(self._waits(trace), [("k", "low")])


class TestNcclCommunicators(unittest.TestCase):
    def _coll(self, pid, start, dur, **tags):
        return _span("ncclAllReduce", "nccl", start, dur, pid=pid, tid=pid, type="allreduce", **tags)

    def test_concurrent_communicators_are_separate(self):
        spans = [self._coll(1, 100, 500, commid="aa", seq=1), self._coll(2, 400, 100, commid="aa", seq=1),
                 self._coll(3, 150, 600, commid="bb", seq=1), self._coll(4, 300, 100, commid="bb", seq=1)]
        arr = {(s.pid, d.pid) for s, d, k, _c in edges(_trace(spans)) if k == "arrival"}
        # the last arriver of each communicator's call (2 on aa, 4 on bb) is
        # the predecessor of the other rank's -- never across communicators
        self.assertEqual(arr, {(2, 1), (4, 3)})

    def test_sequence_numbers_pair_calls_that_do_not_overlap(self):
        # rank 2 is late: its 1st call starts after rank 1's 1st ended, while
        # rank 1's 2nd call overlaps it -- the per-communicator sequence
        # number pairs 1st with 1st
        spans = [self._coll(1, 100, 100, commid="aa", seq=1), self._coll(1, 300, 400, commid="aa", seq=2),
                 self._coll(2, 350, 100, commid="aa", seq=1), self._coll(2, 800, 50, commid="aa", seq=2)]
        arr = {(s.start_ns, d.start_ns, c) for s, d, k, c in edges(_trace(spans)) if k == "arrival"}
        self.assertEqual(arr, {(350, 100, "high"), (800, 300, "high")})

    def test_traces_without_communicator_ids_keep_the_type_fallback(self):
        spans = [self._coll(1, 100, 500), self._coll(2, 400, 100)]
        arr = {(s.pid, d.pid, c) for s, d, k, c in edges(_trace(spans)) if k == "arrival"}
        self.assertEqual(arr, {(2, 1, "medium")})


if __name__ == "__main__":
    unittest.main()
