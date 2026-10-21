"""
Unit tests for src/analysis/multinode.py -- clock-offset estimation
(Cristian's algorithm), trace merging with pid remapping, and post-merge
causality validation. All fully testable without real multi-node hardware
(the arithmetic and data-structure logic have no MPI/network dependency) --
see multinode.py's module docstring for what remains unverified on the
C-side (hooks/mpi_hook/mpi_hook.c's actual round-trip capture, which this
development machine cannot exercise: it can't form a real multi-rank
MPI_COMM_WORLD at all).
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.trace import Trace, TraceMetadata
from src.core.events import SpanEvent, InstantEvent, CounterEvent, Category
from src.analysis import multinode as mn


def _span(pid, tid, cat, start_ns, dur_ns, name="s", tags=None):
    return SpanEvent(name=name, category=cat, start_ns=start_ns, duration_ns=dur_ns,
                     pid=pid, tid=tid, tags=dict(tags or {}))


def _mk_trace(spans=None, instants=None, counters=None):
    t = Trace(TraceMetadata())
    for s in spans or []:
        t.add(s)
    for i in instants or []:
        t.add(i)
    for c in counters or []:
        t.add(c)
    return t


class TestEstimateClockOffset(unittest.TestCase):
    def test_symmetric_latency_gives_exact_offset(self):
        # Hand-derived: reference clock reads TRUE_OFFSET=5000ns ahead of
        # the local clock at any given real instant; 200ns symmetric
        # one-way network latency each direction.
        # T1=1000 (local send), T2=6200, T3=6200 (reference recv+reply,
        # both 200ns after the real send instant in reference's frame),
        # T4=1400 (local receipt, 200ns after reference's reply).
        est = mn.estimate_clock_offset(t1=1000, t2=6200, t3=6200, t4=1400)
        self.assertEqual(est.round_trip_ns, 400)
        self.assertEqual(est.offset_ns, 5000)     # exact under symmetric latency
        self.assertEqual(est.error_bound_ns, 200)  # round_trip / 2

    def test_asymmetric_latency_error_stays_within_bound(self):
        # Same TRUE_OFFSET=5000ns, same 400ns round trip total, but split
        # asymmetrically: 100ns forward, 300ns return (see file's module
        # docstring derivation). The point estimate is now off by exactly
        # half the forward/return difference (100ns) from the true value,
        # but the returned error_bound (200ns) still correctly BRACKETS
        # that real error -- the whole point of returning a bound instead
        # of presenting the point estimate as exact.
        est = mn.estimate_clock_offset(t1=1000, t2=6100, t3=6100, t4=1400)
        self.assertEqual(est.round_trip_ns, 400)
        self.assertEqual(est.offset_ns, 4900)
        self.assertEqual(est.error_bound_ns, 200)
        true_offset = 5000
        self.assertLessEqual(abs(est.offset_ns - true_offset), est.error_bound_ns)

    def test_zero_offset_reference_node(self):
        # A node measuring against itself (or genuinely synchronized)
        # sees t1==t2==t3==t4 apart from negligible local processing.
        est = mn.estimate_clock_offset(t1=1000, t2=1000, t3=1000, t4=1000)
        self.assertEqual(est.offset_ns, 0)
        self.assertEqual(est.error_bound_ns, 0)


class TestOffsetFromCounters(unittest.TestCase):
    def test_extracts_all_three_counters(self):
        trace = _mk_trace(counters=[
            CounterEvent(name="clock_offset_vs_rank0_ns", category=Category.MPI,
                        timestamp_ns=0, value=4900.0, unit="ns"),
            CounterEvent(name="clock_offset_error_bound_ns", category=Category.MPI,
                        timestamp_ns=0, value=200.0, unit="ns"),
            CounterEvent(name="clock_sync_round_trip_ns", category=Category.MPI,
                        timestamp_ns=0, value=400.0, unit="ns"),
        ])
        est = mn.offset_from_counters(trace)
        self.assertIsNotNone(est)
        self.assertEqual(est.offset_ns, 4900)
        self.assertEqual(est.error_bound_ns, 200)
        self.assertEqual(est.round_trip_ns, 400)

    def test_absent_counters_returns_none(self):
        # HPROFILER_CLOCK_SYNC wasn't set for this run, or this trace IS
        # rank 0 (which never emits these -- its offset is 0 by
        # construction, not measured).
        trace = _mk_trace()
        self.assertIsNone(mn.offset_from_counters(trace))


class TestMergeTraces(unittest.TestCase):
    def test_pid_remapped_to_avoid_cross_node_collision(self):
        # Both nodes use the SAME raw pid (1234) -- unrelated processes on
        # different physical machines that happen to share a pid number,
        # exactly the collision merge_traces must prevent.
        node_a = _mk_trace(spans=[_span(1234, 1, Category.CPU, 0, 100, name="a")])
        node_b = _mk_trace(spans=[_span(1234, 1, Category.CPU, 0, 100, name="b")])
        merged, warnings = mn.merge_traces([
            mn.NodeTrace(trace=node_a, offset_ns=0, error_bound_ns=0, label="node0"),
            mn.NodeTrace(trace=node_b, offset_ns=0, error_bound_ns=1, label="node1"),
        ])
        pids = {s.name: s.pid for s in merged.spans}
        self.assertNotEqual(pids["a"], pids["b"], "different nodes' pids must not collide after merge")
        self.assertEqual(pids["a"], 1234)  # node 0 keeps its raw pid (offset 0*stride)
        self.assertEqual(pids["b"], 1234 + mn.PID_NAMESPACE_STRIDE)

    def test_timestamps_shifted_by_node_offset(self):
        node_a = _mk_trace(spans=[_span(1, 1, Category.CPU, 1000, 50, name="a")])
        node_b = _mk_trace(spans=[_span(1, 1, Category.CPU, 1000, 50, name="b")])
        merged, _ = mn.merge_traces([
            mn.NodeTrace(trace=node_a, offset_ns=0, error_bound_ns=0),
            mn.NodeTrace(trace=node_b, offset_ns=5000, error_bound_ns=200),
        ])
        starts = {s.name: s.start_ns for s in merged.spans}
        self.assertEqual(starts["a"], 1000)          # node 0: no shift
        self.assertEqual(starts["b"], 1000 + 5000)    # node 1: shifted by its offset

    def test_mpi_rank_tags_not_remapped(self):
        # rank=/peer= are MPI_COMM_WORLD-global identities, already unique
        # across the whole job regardless of physical node -- must survive
        # the merge completely unchanged for criticalpath.py's cross-rank
        # matching to keep working transparently post-merge.
        node_a = _mk_trace(spans=[_span(1, 1, Category.MPI, 0, 10, name="send",
                                        tags={"type": "send", "rank": "0", "peer": "1", "tag": "5"})])
        node_b = _mk_trace(spans=[_span(1, 1, Category.MPI, 0, 10, name="recv",
                                        tags={"type": "recv", "rank": "1", "peer": "0", "tag": "5"})])
        merged, _ = mn.merge_traces([
            mn.NodeTrace(trace=node_a),
            mn.NodeTrace(trace=node_b, offset_ns=0, error_bound_ns=1),
        ])
        by_name = {s.name: s for s in merged.spans}
        self.assertEqual(by_name["send"].tags["rank"], "0")
        self.assertEqual(by_name["send"].tags["peer"], "1")
        self.assertEqual(by_name["recv"].tags["rank"], "1")
        self.assertEqual(by_name["recv"].tags["peer"], "0")

    def test_node_tag_added(self):
        node_a = _mk_trace(spans=[_span(1, 1, Category.CPU, 0, 10, name="a")])
        node_b = _mk_trace(spans=[_span(1, 1, Category.CPU, 0, 10, name="b")])
        merged, _ = mn.merge_traces([
            mn.NodeTrace(trace=node_a),
            mn.NodeTrace(trace=node_b, offset_ns=0, error_bound_ns=1),
        ])
        by_name = {s.name: s for s in merged.spans}
        self.assertEqual(by_name["a"].tags["node"], "0")
        self.assertEqual(by_name["b"].tags["node"], "1")

    def test_missing_offset_data_warns_for_non_reference_node(self):
        node_a = _mk_trace(spans=[_span(1, 1, Category.CPU, 0, 10)])
        node_b = _mk_trace(spans=[_span(1, 1, Category.CPU, 0, 10)])
        _merged, warnings = mn.merge_traces([
            mn.NodeTrace(trace=node_a),  # node 0 -- the reference; 0 offset is correct, not missing
            mn.NodeTrace(trace=node_b, label="node1"),  # offset_ns=0, error_bound_ns=0 -- genuinely missing
        ])
        self.assertEqual(len(warnings), 1)
        self.assertIn("node1", warnings[0])

    def test_no_warning_for_reference_node_or_real_zero_offset(self):
        node_a = _mk_trace(spans=[_span(1, 1, Category.CPU, 0, 10)])
        node_b = _mk_trace(spans=[_span(1, 1, Category.CPU, 0, 10)])
        # node 1 has a genuinely measured (if tiny) error_bound, distinguishing
        # "measured and happened to be ~0" from "never measured at all".
        _merged, warnings = mn.merge_traces([
            mn.NodeTrace(trace=node_a),
            mn.NodeTrace(trace=node_b, offset_ns=0, error_bound_ns=5),
        ])
        self.assertEqual(warnings, [])

    def test_instants_and_counters_also_shifted_and_remapped(self):
        node_b = _mk_trace(
            instants=[InstantEvent(name="wakeup", category=Category.SCHED,
                                   timestamp_ns=2000, pid=99, tid=1)],
            counters=[CounterEvent(name="mem", category=Category.MEMORY,
                                   timestamp_ns=2000, value=1.0, unit="B", pid=99)],
        )
        merged, _ = mn.merge_traces([
            mn.NodeTrace(trace=_mk_trace()),
            mn.NodeTrace(trace=node_b, offset_ns=500, error_bound_ns=10),
        ])
        inst = merged.instants[0]
        ctr = merged.counters[0]
        self.assertEqual(inst.pid, 99 + mn.PID_NAMESPACE_STRIDE)
        self.assertEqual(inst.timestamp_ns, 2500)
        self.assertEqual(ctr.pid, 99 + mn.PID_NAMESPACE_STRIDE)
        self.assertEqual(ctr.timestamp_ns, 2500)


class TestValidateCausality(unittest.TestCase):
    def test_correct_offset_no_violations(self):
        # Send on node 0 at t=1000; matching recv on node 1, which (after
        # applying its correct +5000 offset) starts at 1000+5000+50=6050,
        # comfortably after the send -- no violation.
        node_a = _mk_trace(spans=[_span(1, 1, Category.MPI, 1000, 10, name="send",
                                        tags={"type": "send", "rank": "0", "peer": "1", "tag": "1"})])
        node_b = _mk_trace(spans=[_span(1, 1, Category.MPI, 50, 100, name="recv",
                                        tags={"type": "recv", "rank": "1", "peer": "0", "tag": "1"})])
        merged, _ = mn.merge_traces([
            mn.NodeTrace(trace=node_a),
            mn.NodeTrace(trace=node_b, offset_ns=5000, error_bound_ns=200),
        ])
        self.assertEqual(mn.validate_causality(merged), [])

    def test_wrong_offset_flags_violation(self):
        # Same spans, but node 1's offset is wrong (way too small/negative),
        # making the merged recv appear to finish BEFORE the send even
        # starts -- a real causality violation the check must catch.
        node_a = _mk_trace(spans=[_span(1, 1, Category.MPI, 100000, 10, name="send",
                                        tags={"type": "send", "rank": "0", "peer": "1", "tag": "1"})])
        node_b = _mk_trace(spans=[_span(1, 1, Category.MPI, 50, 100, name="recv",
                                        tags={"type": "recv", "rank": "1", "peer": "0", "tag": "1"})])
        merged, _ = mn.merge_traces([
            mn.NodeTrace(trace=node_a),
            mn.NodeTrace(trace=node_b, offset_ns=0, error_bound_ns=1),  # should be ~+100000
        ])
        problems = mn.validate_causality(merged)
        self.assertEqual(len(problems), 1)
        self.assertIn("causality violation", problems[0])


# ── offset planning (plan_offsets): same clock, drift, message-order tightening ──

def _ping_pong(true_offset, drift=0.0, t0=10**9, n=40, gap=25_000_000, latency=2_000, local_epoch=0):
    """Rank 0 (reference) and rank 1 exchange n ping-pongs on a true
    timeline; rank 1's clock reads t_local with t_ref = t_local + offset(t_local),
    offset(t) = true_offset + drift * (t - t0). Returns (node0, node1) spans."""
    def local(t_true):
        # invert t_true = t + O + D (t - t0) for t
        return int(round((t_true - true_offset + drift * t0) / (1 + drift)))
    a, b = [], []
    for k in range(n):
        t = t0 + true_offset + k * gap                  # reference time of the ping
        a.append(_span(100, 1, Category.MPI, t, 500, "MPI_Send",
                       {"type": "send", "rank": "0", "peer": "1", "wpeer": "1", "tag": "7", "commid": "0"}))
        recv_start, recv_end = t - 30_000, t + latency
        b.append(_span(200, 1, Category.MPI, local(recv_start), local(recv_end) - local(recv_start), "MPI_Recv",
                       {"type": "recv", "rank": "1", "peer": "0", "wpeer": "0", "tag": "7", "commid": "0"}))
        reply = t + latency + 1_000
        b.append(_span(200, 1, Category.MPI, local(reply), 500, "MPI_Send",
                       {"type": "send", "rank": "1", "peer": "0", "wpeer": "0", "tag": "8", "commid": "0"}))
        a.append(_span(100, 1, Category.MPI, t + 600, reply + latency - (t + 600), "MPI_Recv",
                       {"type": "recv", "rank": "0", "peer": "1", "wpeer": "1", "tag": "8", "commid": "0"}))
    return a, b


def _sync(rank, phase, offset, bound, at, same=False):
    return InstantEvent("clock_sync", Category.MPI, at, 200, 1,
                        {"type": "clock_sync", "phase": phase, "rank": str(rank), "offset": str(offset),
                         "bound": str(bound), "at": str(at), "same_clock": "1" if same else "0"})


def _ref_sync():
    return InstantEvent("clock_sync", Category.MPI, 1, 100, 1,
                        {"type": "clock_sync", "phase": "init", "rank": "0", "offset": "0", "bound": "0",
                         "reference": "1"})


def _pair(true_offset, est, bound, drift=0.0, est_fin=None, boot=("a", "b")):
    a, b = _ping_pong(true_offset, drift)
    t0 = 10**9
    t1 = t0 + 40 * 25_000_000
    inst_b = [_sync(1, "init", est, bound, t0)]
    if est_fin is not None:
        inst_b.append(_sync(1, "finalize", est_fin, bound, t1))
    node0 = _mk_trace(spans=a, instants=[_ref_sync()])
    node1 = _mk_trace(spans=b, instants=inst_b)
    node0.metadata.clock = {"boot_id": boot[0]}
    node1.metadata.clock = {"boot_id": boot[1]}
    return node0, node1


def _merge(plan, traces):
    nodes = [mn.NodeTrace(trace=t, label=p.label, model=p.model) for t, p in zip(traces, plan.nodes)]
    merged, _ = mn.merge_traces(nodes)
    return merged


class TestPlanOffsets(unittest.TestCase):
    def test_same_boot_id_is_offset_zero_without_estimate(self):
        n0, n1 = _pair(0, est=4_000, bound=5_000, boot=("same", "same"))
        plan = mn.plan_offsets([n0, n1])
        self.assertEqual(plan.nodes[1].method, "same-clock")
        self.assertEqual(plan.nodes[1].offset_ns, 0)
        self.assertEqual(mn.validate_causality(_merge(plan, [n0, n1])), [])
        # ignore_boot_id: the estimate (within its bound) is used instead
        plan = mn.plan_offsets([n0, n1], ignore_boot_id=True)
        self.assertNotEqual(plan.nodes[1].method, "same-clock")
        self.assertLessEqual(abs(plan.nodes[1].offset_ns), 5_000)

    def test_exact_estimate_needs_no_correction(self):
        n0, n1 = _pair(7_000_000, est=7_000_000, bound=3_000)
        plan = mn.plan_offsets([n0, n1])
        self.assertEqual(plan.nodes[1].method, "estimate")
        self.assertEqual(plan.nodes[1].correction_ns, 0)
        self.assertEqual(mn.validate_causality(_merge(plan, [n0, n1])), [])

    def test_asymmetric_latency_bias_is_corrected_within_the_bound(self):
        # the estimate is off by +2.6 us (asymmetric path, still inside its
        # 3 us bound): every reply would land before its send -- tightening
        # moves it back inside the bound, near the truth
        true = 7_000_000
        n0, n1 = _pair(true, est=true + 2_600, bound=3_000)
        before = mn.validate_causality(_merge(mn.plan_offsets([n0, n1], explicit=[None, true + 2_600]),
                                              [n0, n1]))
        self.assertGreater(len(before), 0)
        plan = mn.plan_offsets([n0, n1])
        off = plan.nodes[1].offset_ns
        self.assertLessEqual(abs(off - (true + 2_600)), 3_000)        # inside [est +- bound]
        self.assertLessEqual(abs(off - true), 3_000)                  # recovered within the bound
        self.assertNotEqual(plan.nodes[1].correction_ns, 0)
        self.assertEqual(mn.validate_causality(_merge(plan, [n0, n1])), [])

    def test_drift_is_recovered_from_the_two_measurements(self):
        # 50 ppm over a 1 s run: 50 us of drift against a 3 us bound
        true, drift = 7_000_000, 50e-6
        t0, t1 = 10**9, 10**9 + 40 * 25_000_000
        n0, n1 = _pair(true, est=true, bound=3_000, drift=drift, est_fin=int(true + drift * (t1 - t0)))
        plan = mn.plan_offsets([n0, n1])
        p = plan.nodes[1]
        self.assertEqual(p.method, "estimate+drift")
        self.assertAlmostEqual(p.drift_ppm, 50.0, places=3)
        for t in (t0, (t0 + t1) // 2, t1):
            self.assertLessEqual(abs(p.model.offset_at(t) - (true + drift * (t - t0))), 3_000)
        self.assertEqual(mn.validate_causality(_merge(plan, [n0, n1])), [])
        # a constant offset (the MPI_Init estimate alone) breaks message order
        const = mn.plan_offsets([n0, n1], explicit=[None, true])
        self.assertGreater(len(mn.validate_causality(_merge(const, [n0, n1]))), 0)

    def test_no_offset_within_the_bound_keeps_the_estimate_and_says_so(self):
        n0, n1 = _pair(7_000_000, est=7_050_000, bound=1_000)       # 50 us off, 1 us bound
        plan = mn.plan_offsets([n0, n1])
        self.assertEqual(plan.nodes[1].offset_ns, 7_050_000)
        self.assertIn("no offsets within the error bounds", plan.tightening)

    def test_reference_is_the_trace_holding_rank_0(self):
        n0, n1 = _pair(7_000_000, est=7_000_000, bound=3_000)
        plan = mn.plan_offsets([n1, n0])                              # rank 0's trace second
        self.assertEqual(plan.reference, 1)
        self.assertEqual(plan.nodes[1].method, "reference")
        self.assertEqual(plan.nodes[0].offset_ns, 7_000_000)

    def test_older_counters_are_an_estimate(self):
        a, b = _ping_pong(7_000_000)
        n0 = _mk_trace(spans=a)
        n1 = _mk_trace(spans=b, counters=[
            CounterEvent("clock_offset_vs_rank0_ns", Category.MPI, 5, 7_000_000.0, 200),
            CounterEvent("clock_offset_error_bound_ns", Category.MPI, 5, 3_000.0, 200)])
        plan = mn.plan_offsets([n0, n1])
        self.assertEqual(plan.nodes[1].method, "estimate (counters)")
        self.assertEqual(mn.validate_causality(_merge(plan, [n0, n1])), [])

    def test_no_clock_data_uses_the_message_order(self):
        a, b = _ping_pong(7_000_000)
        n0, n1 = _mk_trace(spans=a), _mk_trace(spans=b)
        plan = mn.plan_offsets([n0, n1])
        self.assertEqual(plan.nodes[1].method, "message order")
        # ping and reply bracket the offset: within one latency of the truth
        self.assertLessEqual(abs(plan.nodes[1].offset_ns - 7_000_000), 2_000)
        self.assertEqual(mn.validate_causality(_merge(plan, [n0, n1])), [])

    def test_three_nodes_with_a_coupled_constraint(self):
        # node 2 talks only to node 1; its estimate is biased the wrong way
        a, b = _ping_pong(7_000_000)
        c_spans = []
        for k in range(20):
            t_ref = 10**9 + 7_000_000 + k * 25_000_000 + 10_000_000       # node 1 sends to node 2
            b.append(_span(200, 1, Category.MPI, t_ref - 7_000_000, 300, "MPI_Send",
                           {"type": "send", "rank": "1", "peer": "2", "wpeer": "2", "tag": "9", "commid": "0"}))
            c_spans.append(_span(300, 1, Category.MPI, t_ref - 3_000_000 - 20_000, 20_000 + 1_500, "MPI_Recv",
                                 {"type": "recv", "rank": "2", "peer": "1", "wpeer": "1", "tag": "9",
                                  "commid": "0"}))
        n0 = _mk_trace(spans=a, instants=[_ref_sync()])
        n1 = _mk_trace(spans=b, instants=[_sync(1, "init", 7_000_000, 3_000, 10**9)])
        n2 = _mk_trace(spans=c_spans, instants=[_sync(2, "init", 3_000_000 - 2_500, 3_000, 10**9)])
        for t, bid in ((n0, "a"), (n1, "b"), (n2, "c")):
            t.metadata.clock = {"boot_id": bid}
        plan = mn.plan_offsets([n0, n1, n2])
        self.assertLessEqual(abs(plan.nodes[2].offset_ns - 3_000_000), 3_000)
        self.assertEqual(mn.validate_causality(_merge(plan, [n0, n1, n2])), [])


class TestMergeNodesCli(unittest.TestCase):
    def test_same_host_per_rank_traces_merge_at_zero(self):
        import subprocess
        import tempfile
        from src.core.trace_io import write_json_export
        root = Path(__file__).resolve().parent.parent
        # estimates far off (like the old one-ping protocol) but one boot_id
        n0, n1 = _pair(0, est=3_000_000, bound=4_000_000, boot=("same", "same"))
        with tempfile.TemporaryDirectory() as d:
            paths = []
            for k, t in enumerate((n0, n1)):
                paths.append(str(Path(d) / f"rank{k}.json"))
                write_json_export(t, paths[-1])
            r = subprocess.run([sys.executable, str(root / "hprofiler"), "merge-nodes", *paths,
                                "-o", str(Path(d) / "m.json")], capture_output=True, text=True, timeout=120)
            self.assertEqual(r.returncode, 0, r.stderr[-1500:])
            self.assertIn("offset=0ns (same clock as the reference", r.stdout)
            self.assertNotIn("causality violation", r.stdout)
            from src.core.trace_io import open_trace
            m = open_trace(str(Path(d) / "m.json"))
            self.assertEqual([x["method"] for x in m.metadata.clock["merge"]["nodes"]],
                             ["reference", "same-clock"])

