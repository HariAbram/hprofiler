"""
The critical path's window (criticalpath.analyze(window=...), `critical-path
--window`): the measured wall time and the observed path start at the
earliest process-begin marker (hp_transport.h: a hook's constructor, before
main), so work before a process's first recorded call is on the path, like
Score-P's; the time from the window's start to the path's first event is
unrecorded thread time on that event's span. Traces without markers (and
--window first-event) keep the first recorded event.

Also MPI_Init / MPI_Finalize as spans (mpi_hook.c): one rendezvous of every
process each ("medium", provenance lifecycle_sync, inferred), an edge the
timestamps contradict dropped.

The trace: two ranks; rank 0 works 20 ms between MPI_Init and its send,
rank 1 starts MPI_Init last and works 28 ms after the receive.
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analysis import cp_model as cm
from src.analysis import criticalpath as cp
from src.core.events import Category, InstantEvent, SpanEvent
from src.core.trace import Trace, TraceMetadata
from src.core.trace_io import write_json_export

MS = 1_000_000


def mpi(name, pid, a, b, **tags):
    return SpanEvent(name, Category.MPI, round(a * MS), round((b - a) * MS), pid, pid,
                     tags={k: str(v) for k, v in tags.items()})


def lifecycle(name, pid, rank, a, b):
    return mpi(name, pid, a, b, type="init" if name.startswith("MPI_Init") else "finalize", rank=rank,
               commid=0, csize=2)


def marker(pid, at):
    return InstantEvent("process_begin", Category.OTHER, round(at * MS), pid, pid,
                        tags={"type": "process_begin", "hook": "mpi"})


def build(markers=True, fin1=(100.5, 120.0)):
    t = Trace(TraceMetadata())
    events = [
        lifecycle("MPI_Init", 1, 0, 1, 50),
        mpi("MPI_Send", 1, 70, 71, type="send", rank=0, peer=1, wpeer=1, tag=0, commid=0),
        lifecycle("MPI_Finalize", 1, 0, 100, 120),
        lifecycle("MPI_Init_thread", 2, 1, 2, 50),
        mpi("MPI_Recv", 2, 51, 72, type="recv", rank=1, peer=0, wpeer=0, tag=0, commid=0),
        lifecycle("MPI_Finalize", 2, 1, *fin1),
    ]
    if markers:
        events += [marker(1, 0), marker(2, 0.5), marker(77, -50)]   # pid 77: a launcher, no spans
    for e in events:
        t.add(e)
    return t


def path_ns(rep):
    return sum(c + g for c, g in zip(rep.path_credited_ns, rep.path_gap_ns))


class TestWindow(unittest.TestCase):
    def test_from_process_begin(self):
        t = build()
        self.assertEqual(cp.process_begin(t), {1: 0, 2: round(0.5 * MS)})    # the launcher's is ignored
        rep = cp.analyze(t, sensitivity=0, alternatives=0)
        self.assertEqual((rep.window_start, rep.window_start_ns, rep.window_lead_ns), ("process_begin", 0, MS))
        self.assertEqual(rep.wall_ns, 120 * MS)                 # from process begin, not from the first span
        self.assertEqual(rep.total_path_ns, 120 * MS)
        self.assertEqual(path_ns(rep), rep.total_path_ns)       # what --export writes sums to observedNs
        # the path starts in the last arriver at MPI_Init (rank 1): the 2 ms
        # since the window's start are its unrecorded thread time
        first = rep.spans[rep.path_span_indices[0]]
        self.assertEqual((first.name, rep.path_gap_ns[0]), ("MPI_Init_thread", 2 * MS))
        self.assertEqual(rep.model.walk.lead, 2 * MS)
        self.assertEqual(rep.model.by_provenance["process_begin"], (2 * MS, 1))
        self.assertGreaterEqual(rep.path_unrecorded_ns, 2 * MS)
        # rank 0's 20 ms before its send are on the path (the gap before the send)
        k = [rep.spans[i].name for i in rep.path_span_indices].index("MPI_Send")
        self.assertEqual(rep.path_gap_ns[k], 20 * MS)
        d = cp.report_to_dict(rep)
        self.assertEqual((d["windowStart"], d["windowStartNs"], d["windowLeadNs"], d["wallNs"], d["observedNs"],
                          d["walk"]["leadNs"]), ("process_begin", 0, MS, 120 * MS, 120 * MS, 2 * MS))
        comp = d["composition"]
        total = (sum(comp["busy"].values()) + sum(comp["inGatedCalls"].values())
                 + sum(comp["inUngatedBlockingCalls"].values()) + sum(comp["dependencyLatency"].values())
                 + comp["unrecordedNs"] + comp["deviceIdleNs"] + comp["threadIdleNs"])
        self.assertEqual(total, d["observedNs"])
        self.assertEqual(sum(rep.path_time_by_tier.values()), rep.total_path_ns)

    def test_without_markers_from_the_first_event(self):
        rep = cp.analyze(build(markers=False), sensitivity=0, alternatives=0)
        old = cp.analyze(build(), sensitivity=0, alternatives=0, window="first_event")
        for r in (rep, old):
            self.assertEqual((r.window_start, r.window_start_ns, r.window_lead_ns), ("first_event", MS, 0))
            self.assertEqual(r.wall_ns, 119 * MS)
            self.assertEqual(r.total_path_ns, 118 * MS)          # from the path's first event (rank 1's init)
            self.assertEqual(path_ns(r), r.total_path_ns)
            self.assertEqual(r.model.walk.lead, 0)
        self.assertEqual([r.spans[i].name for i in rep.path_span_indices],
                         [r.spans[i].name for i in old.path_span_indices])
        self.assertEqual(rep.modeled_path_ns, old.modeled_path_ns)

    def test_modeled_path_counts_the_time_since_process_begin_as_unrecorded(self):
        pb = cp.analyze(build(), sensitivity=0, alternatives=0)
        fe = cp.analyze(build(), sensitivity=0, alternatives=0, window="first_event")
        g = pb.model.graph
        leads = {pb.spans[v >> 1].name: ns for v, ns in g.lead.items()}
        self.assertEqual(leads, {"MPI_Init": MS, "MPI_Init_thread": round(1.5 * MS)})
        self.assertEqual(pb.modeled_path_ns - fe.modeled_path_ns, round(1.5 * MS))
        self.assertEqual(pb.modeled_unrecorded_idle_ns, fe.modeled_unrecorded_idle_ns)   # gaps as idle: no lead
        crit = pb.model.crit
        self.assertTrue(crit.on_all)                             # the cut test sees the weighted source edge

    def test_bad_window(self):
        with self.assertRaises(ValueError):
            cp.analyze(build(), window="program_start")


class TestLifecycleRendezvous(unittest.TestCase):
    def edges(self, rep):
        g = rep.model.graph
        return [(g.name(g.src[e] >> 1), g.name(g.dst[e] >> 1), cm.PROVS[g.prov[e]], cm.CONFS[g.conf[e]])
                for e in range(len(g)) if g.mode[e] == cm.DEP and cm.KINDS[g.kind[e]] == "arrival"
                and g.enabled[e]]

    def test_init_and_finalize_rendezvous(self):
        rep = cp.analyze(build(), sensitivity=0, alternatives=0)
        self.assertEqual(sorted(self.edges(rep)),
                         [("MPI_Finalize", "MPI_Finalize", "lifecycle_sync", "medium"),
                          ("MPI_Init_thread", "MPI_Init", "lifecycle_sync", "medium")])
        self.assertEqual(cm.edge_class("lifecycle_sync", "medium"), "inferred")
        self.assertIn("arrival", rep.path_dependency_by_kind)
        self.assertNotIn("dropped_contradicting_time", {k: v for k, v in rep.model.graph_evidence.items() if v})

    def test_a_finalize_that_does_not_synchronize_is_dropped(self):
        # rank 1 arrives at MPI_Finalize after rank 0 already left it
        rep = cp.analyze(build(fin1=(125.0, 130.0)), sensitivity=0, alternatives=0)
        self.assertEqual(rep.model.graph_evidence.get("dropped_contradicting_time"), {"arrival": 1})
        self.assertEqual([e for e in self.edges(rep) if e[0] == "MPI_Finalize"], [])

    def test_a_rank_missing_makes_it_low(self):
        # MPI_COMM_WORLD has 3 processes, the trace 2 (a lost record, one
        # node of a multi-node job): the last arriver may be the missing one
        t = Trace(TraceMetadata())
        for e in (lifecycle("MPI_Init", 1, 0, 1, 50), lifecycle("MPI_Init", 2, 1, 2, 50)):
            e.tags["csize"] = "3"
            t.add(e)
        rep = cp.analyze(t, sensitivity=0, alternatives=0)
        self.assertEqual([e[2:] for e in self.edges(rep)], [("lifecycle_sync", "low")])

    def test_blocking_without_the_rendezvous(self):
        from src.analysis.semantics import blocking_wait
        self.assertTrue(blocking_wait(lifecycle("MPI_Init", 1, 0, 1, 50)))
        self.assertTrue(blocking_wait(lifecycle("MPI_Finalize", 1, 0, 1, 50)))
        # a single process: no rendezvous partner, the call is its own
        t = Trace(TraceMetadata())
        for e in (marker(1, 0), lifecycle("MPI_Init", 1, 0, 1, 50), lifecycle("MPI_Finalize", 1, 0, 60, 70)):
            t.add(e)
        rep = cp.analyze(t, sensitivity=0, alternatives=0)
        self.assertEqual(rep.total_path_ns, 70 * MS)
        self.assertEqual(path_ns(rep), rep.total_path_ns)


class TestWindowCli(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        d = Path(cls.tmp.name)
        src = d / "t.json"
        write_json_export(build(), str(src))
        cls.out = {}
        for window in ("process-begin", "first-event"):
            js, ex = d / f"cp_{window}.json", d / f"export_{window}.json"
            r = subprocess.run([sys.executable, str(ROOT / "hprofiler"), "critical-path", str(src), "--window",
                                window, "--json", str(js), "--export", str(ex), "--sensitivity", "0",
                                "--alternatives", "0"], capture_output=True, text=True, timeout=180)
            cls.out[window] = (r, js, ex)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_both_windows(self):
        for window, (start, observed) in (("process-begin", ("process_begin", 120 * MS)),
                                          ("first-event", ("first_event", 118 * MS))):
            r, js, ex = self.out[window]
            self.assertEqual(r.returncode, 0, r.stderr[-800:])
            d = json.loads(js.read_text())
            self.assertEqual((d["windowStart"], d["observedNs"]), (start, observed))
            self.assertIn("Window", r.stdout)
            events = json.loads(ex.read_text())
            events = events["traceEvents"] if isinstance(events, dict) else events
            tagged = [float(e["args"]["path_ns"]) for e in events
                      if isinstance(e, dict) and str((e.get("args") or {}).get("on_critical_path")) == "1"]
            self.assertAlmostEqual(sum(tagged), d["observedNs"], delta=len(tagged))   # the harness's invariant
        self.assertIn("from process begin", self.out["process-begin"][0].stdout)
        self.assertIn("--window first-event", self.out["first-event"][0].stdout)


if __name__ == "__main__":
    unittest.main()
