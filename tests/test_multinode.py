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


if __name__ == "__main__":
    unittest.main()
