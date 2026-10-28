"""
Runtime semantics in the dependency graph (src/analysis/semantics/), on
synthetic spans carrying the hooks' tags:

  OpenMP   a taskwait waits for its task's children created before it (not
           later ones), a taskgroup for the tasks created inside it and
           their descendants, a dependence orders source before sink, a
           contended lock acquisition follows the previous holder's release
           (OMPT wait_id=, GNU lock=, GNU unnamed critical sections); the
           waits are "wait" in the model, never "unknown";
  OpenCL   an in-order queue chains its device commands, an out-of-order
           one only through wait lists and barriers; a wait list across
           queues, a marker's prerequisites, clFinish and clWaitForEvents
           wait for the right device spans; a blocking transfer waits for
           its queue; device spans are on no host thread's program order;
  NCCL     the k-th send to a peer pairs with the k-th receive from it
           (pseq=), between the spans timing the work: a grouped
           operation's group_exec span, an asynchronous one's device span;
           collectives rendezvous on those spans, not on host calls.

Both pipelines (dict and streamed) build the same edges.
"""
import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis import cp_model as cm
from src.analysis import criticalpath as cp
from src.core.events import Category, InstantEvent, SpanEvent
from src.core.trace import Trace, TraceMetadata


def S(name, cat, start, end, pid=1, tid=1, sid="", psid="", **tags):
    return SpanEvent(name, Category(cat), start, end - start, pid, tid,
                     tags={k: str(v) for k, v in tags.items()}, span_id=sid, parent_span_id=psid)


def trace(spans, instants=()):
    t = Trace(TraceMetadata())
    for s in spans:
        t.add(s)
    for i in instants:
        t.add(i)
    return t


def edges(t, kinds):
    spans, preds = cp.build_dependency_graph(t)
    full = [(v, u, cp._KIND_CODE[k], cp._CONF_CODE[c]) for v, pl in preds.items() for u, k, c in pl]
    assert sorted(cp.build_edges(t).tolist()) == sorted(full), "pipelines differ"
    return Counter((k, spans[u].name, spans[v].name) for v, pl in preds.items() for u, k, _c in pl if k in kinds)


def model(t):
    return cm.analyze_model(t, cp.load_or_build_graph(t), sensitivity=0)


# ── OpenMP ───────────────────────────────────────────────────────────────────

def task(tid_, start, end, thread, **kw):
    return S("omp_task", "openmp", start, end, tid=thread, type="task", id=tid_, **kw)


def create(tid_, parent, at, thread=1):
    return S("omp_task_create", "openmp", at, at, tid=thread, type="task_create", id=tid_, parent=parent)


class TestOpenMPTasks(unittest.TestCase):
    def test_taskwait_waits_for_children_created_before_it(self):
        t = trace([create("1", "100", 10), create("2", "100", 20), create("3", "100", 600),
                   task("1", 30, 300, 2), task("2", 40, 500, 3), task("3", 610, 700, 2),
                   S("omp_taskwait", "sync", 50, 510, type="sync", sk=5, task="100"),
                   S("omp_task", "openmp", 0, 0, tid=9, type="task", id="0")])
        self.assertEqual(edges(t, ("task_join",)), Counter({("task_join", "omp_task", "omp_taskwait"): 2}))
        res = model(t)
        self.assertEqual(res.unmatched_by_category, {})
        self.assertEqual(res.graph_evidence["by_provenance"].get("task_id"), 2)

    def test_taskgroup_waits_for_descendants_created_inside_it(self):
        t = trace([create("1", "100", 5),                       # before the group: not waited for
                   create("2", "100", 20), create("3", "2", 60),   # inside, and a grandchild
                   task("1", 10, 100, 2), task("2", 25, 200, 3), task("3", 70, 400, 2),
                   S("omp_taskgroup", "sync", 15, 410, type="sync", sk=6, task="100")])
        spans, preds = cp.build_dependency_graph(t)
        joined = sorted(spans[u].tags["id"] for v, pl in preds.items() for u, k, _c in pl if k == "task_join")
        self.assertEqual(joined, ["2", "3"])

    def test_a_suspended_task_completes_with_its_last_fragment(self):
        t = trace([create("1", "100", 5), task("1", 10, 50, 2, fragment=1), task("1", 80, 200, 3),
                   S("omp_taskwait", "sync", 20, 210, type="sync", sk=5, task="100")])
        spans, preds = cp.build_dependency_graph(t)
        src = [spans[u] for v, pl in preds.items() for u, k, _c in pl if k == "task_join"]
        self.assertEqual([(s.start_ns, s.tags.get("fragment")) for s in src], [(80, None)])

    def test_a_dependence_orders_source_before_sink(self):
        dep = InstantEvent("omp_task_dependence", Category.OPENMP, 15, 1, 1, tags={"src": "1", "sink": "2"})
        t = trace([create("1", "100", 5), create("2", "100", 6), task("1", 10, 100, 2), task("2", 120, 200, 3)],
                  [dep])
        self.assertEqual(edges(t, ("task_dep",)), Counter({("task_dep", "omp_task", "omp_task"): 1}))


class TestLocks(unittest.TestCase):
    def test_ompt_lock_handoff(self):
        t = trace([S("omp_lock_hold", "openmp", 0, 100, tid=1, type="hold", mutex="lock", wait_id=7),
                   S("omp_lock_wait", "sync", 20, 101, tid=2, type="sync", mutex="lock", wait_id=7),
                   S("omp_lock_hold", "openmp", 101, 150, tid=2, type="hold", mutex="lock", wait_id=7),
                   S("omp_lock_wait", "sync", 30, 60, tid=3, type="sync", mutex="lock", wait_id=8)])
        self.assertEqual(edges(t, ("lock_order",)),
                         Counter({("lock_order", "omp_lock_hold", "omp_lock_wait"): 1}))
        g = model(t).graph
        modes = {(g.name(g.owner[e]), g.t[2 * g.owner[e]]): cm.MODES[g.mode[e]] for e in range(len(g))
                 if g.owner[e] >= 0 and g.name(g.owner[e]) == "omp_lock_wait"}
        self.assertEqual(modes, {("omp_lock_wait", 20): "wait", ("omp_lock_wait", 30): "unknown"})

    def test_gnu_unnamed_critical_sections_share_one_lock(self):
        t = trace([S("omp_critical_hold", "openmp", 0, 100, tid=1, type="sync"),
                   S("omp_critical_wait", "sync", 20, 100, tid=2, type="sync"),
                   S("omp_critical_wait", "sync", 30, 70, tid=3, type="sync", named=1)])
        self.assertEqual(edges(t, ("lock_order",)),
                         Counter({("lock_order", "omp_critical_hold", "omp_critical_wait"): 1}))


# ── OpenCL ───────────────────────────────────────────────────────────────────

def enq(name, start, sid, queue, ev="", wait="", ooo=False, typ="kernel", **kw):
    tags = {"type": typ, "side": "cpu", "queue": queue}
    if ooo:
        tags["ooo"] = 1
    if ev:
        tags["ev"] = ev
    if wait:
        tags["wait"] = wait
        tags["nwait"] = len(wait.split(";"))
    tags.update(kw)
    return S(name, "opencl" if typ in ("kernel", "marker", "barrier") else "memory", start, start + 5,
             sid=sid, **tags)


def dev(name, start, end, psid, queue, ev="", cat="opencl"):
    """The device span of a command (named name+"'" here, to tell it from
    its enqueue call in the assertions)."""
    tags = {"type": "kernel", "side": "gpu", "queue": queue}
    if ev:
        tags["ev"] = ev
    return S(name + "'", cat, start, end, tid=99, psid=psid, **tags)


class TestOpenCLQueues(unittest.TestCase):
    def test_in_order_queue_wait_lists_and_waits(self):
        t = trace([enq("k1", 0, "h1", "1", ev="1"), dev("k1", 10, 100, "h1", "1", ev="1"),
                   enq("k2", 6, "h2", "1"), dev("k2", 100, 200, "h2", "1"),
                   enq("k3", 12, "h3", "2", ooo=True, ev="3", wait="1"), dev("k3", 110, 300, "h3", "2", ev="3"),
                   S("clWaitForEvents", "sync", 20, 310, type="sync", count=1, wait="3", nwait=1),
                   S("clFinish", "sync", 320, 330, type="sync", queue="1")])
        e = edges(t, ("sequential", "device_wait"))
        self.assertEqual(e[("sequential", "k1'", "k2'")], 1)               # in-order queue 1
        self.assertEqual(e[("sequential", "k1'", "k3'")], 1)               # wait list across queues
        self.assertEqual(e[("device_wait", "k3'", "clWaitForEvents")], 1)
        self.assertEqual(e[("device_wait", "k2'", "clFinish")], 1)
        res = model(t)
        g = res.graph
        modes = {g.name(g.owner[e]): cm.MODES[g.mode[e]] for e in range(len(g)) if g.owner[e] >= 0}
        self.assertEqual(modes["clWaitForEvents"], "wait")
        # queue 1's commands are one stream: k1 -> k2 is device idle, not a dependency
        self.assertIn("stream_order", res.graph_evidence["by_kind"])
        # device spans are no part of the callback thread's program order
        # (tid 99): it would have added k1' -> k2' a second time
        self.assertEqual(sum(n for k, n in e.items() if k[1].endswith("'") and k[2].endswith("'")), 2)

    def test_out_of_order_queue_orders_only_by_barriers(self):
        t = trace([enq("a", 0, "h1", "5", ooo=True), dev("a", 10, 100, "h1", "5"),
                   enq("b", 5, "h2", "5", ooo=True), dev("b", 10, 150, "h2", "5"),
                   enq("clEnqueueBarrierWithWaitList", 8, "", "5", ooo=True, typ="barrier"),
                   enq("c", 9, "h3", "5", ooo=True), dev("c", 160, 200, "h3", "5")])
        e = {k: n for k, n in edges(t, ("sequential",)).items() if k[1].endswith("'")}
        self.assertEqual(e, {("sequential", "b'", "c'"): 1})               # the latest-ending before the barrier

    def test_a_marker_stands_for_its_prerequisites(self):
        t = trace([enq("a", 0, "h1", "5", ooo=True), dev("a", 10, 300, "h1", "5"),
                   enq("clEnqueueMarkerWithWaitList", 3, "", "5", ooo=True, typ="marker", ev="9"),
                   S("clWaitForEvents", "sync", 20, 310, type="sync", count=1, wait="9", nwait=1)])
        self.assertEqual(edges(t, ("device_wait",)), Counter({("device_wait", "a'", "clWaitForEvents"): 1}))

    def test_a_blocking_transfer_waits_for_its_queue(self):
        t = trace([enq("k1", 0, "h1", "1"), dev("k1", 10, 100, "h1", "1"),
                   enq("clEnqueueReadBuffer", 6, "", "1", typ="read", blocking=1)])
        self.assertEqual(edges(t, ("device_wait",)),
                         Counter({("device_wait", "k1'", "clEnqueueReadBuffer"): 1}))


# ── NCCL ─────────────────────────────────────────────────────────────────────

def nccl(name, pid, start, end, typ, **tags):
    tags = {"type": typ, "commid": "c0", **tags}
    return S(name, "nccl", start, end, pid=pid, tid=pid, **tags)


class TestNCCL(unittest.TestCase):
    def test_point_to_point_pairs_by_sequence(self):
        t = trace([nccl("ncclSend", 1, 0, 100, "send", rank=0, peer=1, pseq=1),
                   nccl("ncclSend", 1, 100, 200, "send", rank=0, peer=1, pseq=2),
                   nccl("ncclRecv", 2, 50, 150, "recv", rank=1, peer=0, pseq=1),
                   nccl("ncclRecv", 2, 150, 260, "recv", rank=1, peer=0, pseq=2)])
        spans, preds = cp.build_dependency_graph(t)
        pairs = sorted((spans[u].tags["pseq"], spans[v].tags["pseq"]) for v, pl in preds.items()
                       for u, k, _c in pl if k == "p2p")
        self.assertEqual(pairs, [("1", "1"), ("2", "2")])
        self.assertEqual(model(t).graph_evidence["by_provenance"].get("nccl_pseq"), 2)

    def test_grouped_operations_pair_their_groups_work(self):
        t = trace([nccl("ncclSend", 1, 0, 1, "send", rank=0, peer=1, pseq=1, group=4, stream=7,
                        timing="deferred", side="cpu"),
                   nccl("ncclGroup", 1, 2, 300, "group_exec", group=4, ops=1, stream=7),
                   nccl("ncclRecv", 2, 0, 1, "recv", rank=1, peer=0, pseq=1, group=9, stream=3,
                        timing="deferred", side="cpu"),
                   nccl("ncclGroup", 2, 2, 320, "group_exec", group=9, ops=1, stream=3)])
        self.assertEqual(edges(t, ("p2p",)), Counter({("p2p", "ncclGroup", "ncclGroup"): 1}))

    def test_asynchronous_collectives_rendezvous_on_device_spans(self):
        def coll(pid, h0, d0, d1):
            return [nccl("ncclAllReduce", pid, h0, h0 + 2, "allreduce", seq=1, side="cpu", sid=f"s{pid}"),
                    S("ncclAllReduce", "nccl", d0, d1, pid=pid, tid=pid, psid=f"s{pid}", type="allreduce",
                      commid="c0", seq="1", side="gpu", timing="async", stream="7")]
        t = trace(coll(1, 0, 10, 300) + coll(2, 5, 200, 300))
        spans, preds = cp.build_dependency_graph(t)
        arr = [(spans[u].tags["side"], spans[v].tags["side"]) for v, pl in preds.items()
               for u, k, _c in pl if k == "arrival"]
        self.assertEqual(arr, [("gpu", "gpu")])


if __name__ == "__main__":
    unittest.main()
