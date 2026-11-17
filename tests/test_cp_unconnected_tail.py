"""
Where the observed critical path starts its walk back
(cp_model.last_events / connected_events): at the last event it can connect
to the program -- a host span's, or one an edge chain reaches from one --
never at a device record nothing recorded leads to. CUPTI reports every
completed event query (CUDA 12.8 / 13) and every event synchronization as
a synchronization record; one whose host call was not recorded sits on the
device lane with no edge, and a trace that ended on one walked back a
single record (143 ns of a 4.54 s Triton run, campaign smoke2).

  * a trace that ends on such records: the walk starts at the last host
    event; the records after it, how many and how much later, are in the
    report, the notes and the JSON (unconnectedTailNs / Records), and the
    path's spans still sum to the observed path;
  * every sync record without a host call is counted (unmatchedSyncRecords,
    a note), wherever it is;
  * with the host calls recorded (cuda_hook.c's cudaEventQuery, the CUPTI
    callback's cuEventQuery / cuEventSynchronize) nothing is left over;
  * a trace with no host span at all keeps walking from its last event.

Wire records in exactly the format the hooks emit, through the Runner's
parser (tests/test_gpu_activity.py's helpers).
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis import cp_model as cm
from src.analysis import criticalpath as cp
from tests.test_gpu_activity import build, cupti_kernel, cupti_sync, host

US = 1_000


def program(query_recorded: bool):
    """A launch, its kernel, an event record and an event sync; later a
    second launch -- the program's last host call, ending at 60.1 us --
    then two completed event queries (CUPTI sync records, corr 98 and 99),
    the second one the trace's last event (70.2 us)."""
    lines = [
        host("cudaLaunchKernel", 1 * US, 100, lid=1, corr=10, stream=7),
        cupti_kernel(10, 7, 2 * US, 50 * US),
        host("cudaEventRecord", 3 * US, 50, typ="event_record", op="event_record", lid=2, corr=11, stream=7,
             event=5),
        host("cudaEventSynchronize", 4 * US, 46 * US + 100, cat="sync", typ="sync", op="sync", lid=3, corr=12,
             sync="event", event=5),
        cupti_sync(12, "event", 4 * US + 10, 50 * US + 50),
        host("cudaLaunchKernel", 60 * US, 100, lid=4, corr=13, stream=7),
        cupti_kernel(13, 7, 60 * US + 50, 60 * US + 90),
        cupti_sync(98, "event", 55 * US, 55 * US + 200),
        cupti_sync(99, "event", 70 * US, 70 * US + 200),
    ]
    if query_recorded:
        lines += [
            host("cudaEventQuery", 55 * US - 50, 300, cat="sync", typ="sync", op="sync", lid=5, corr=98,
                 sync="event", event=5),
            host("cuEventQuery", 70 * US - 50, 300, cat="sync", typ="sync", op="sync", lid=6, corr=99,
                 sync="event", event=5, src="cupti_cb"),
        ]
    return build(lines)


class TestUnconnectedTail(unittest.TestCase):
    def test_the_walk_starts_at_the_last_event_it_can_connect(self):
        t = program(query_recorded=False)
        r = cp.analyze(t, sensitivity=0, alternatives=0)
        w = r.model.walk
        g = r.model.graph
        # the sink: the second launch's kernel end (60.09 us), not the
        # unmatched sync record at 70.2 us
        self.assertEqual(g.t[w.sink], 60 * US + 100)
        self.assertFalse(any(g.name(j) == "sync event" and g.t[2 * j + 1] > 60 * US for j in w.spans(g)))
        self.assertEqual((w.tail_records, w.tail_ns, w.tail_names), (1, 70 * US + 200 - (60 * US + 100),
                                                                        {"sync event": 1}))
        self.assertEqual((r.unconnected_tail_records, r.unconnected_tail_ns), (1, 10 * US + 100))
        # most of the run is on the observed path; the tail is what is not
        self.assertEqual(r.wall_ns, 70 * US + 200 - 1 * US)
        self.assertEqual(r.total_path_ns, r.wall_ns - r.unconnected_tail_ns)
        self.assertEqual(sum(c + gp for c, gp in zip(r.path_credited_ns, r.path_gap_ns)), r.total_path_ns)
        self.assertTrue(any("cannot connect" in n and "sync event x1" in n for n in r.notes), r.notes)
        d = cp.report_to_dict(r)
        self.assertEqual((d["unconnectedTailNs"], d["unconnectedTailRecords"]), (10 * US + 100, 1))
        self.assertEqual(d["walk"]["tailByName"], {"sync event": 1})

    def test_sync_records_without_a_host_call_are_counted(self):
        r = cp.analyze(program(query_recorded=False), sensitivity=0, alternatives=0)
        self.assertEqual(r.unmatched_sync_records, {"event": 2})
        self.assertEqual(cp.report_to_dict(r)["unmatchedSyncRecords"], {"event": 2})
        self.assertTrue(any("2 device synchronization record(s) have no recorded host call (sync=event 2)" in n
                            for n in r.notes), r.notes)

    def test_with_the_host_calls_recorded_nothing_is_left_over(self):
        t = program(query_recorded=True)
        r = cp.analyze(t, sensitivity=0, alternatives=0)
        g = r.model.graph
        self.assertEqual((r.unconnected_tail_records, r.unconnected_tail_ns), (0, 0))
        self.assertEqual(r.unmatched_sync_records, {})
        self.assertFalse(any("cannot connect" in n or "no recorded host call" in n for n in r.notes), r.notes)
        # the last host call (cuEventQuery, ending at 70.25 us) ends the path
        self.assertEqual(g.name(r.model.walk.sink >> 1), "cuEventQuery")
        self.assertEqual(r.total_path_ns, r.wall_ns)

    def test_a_trace_without_host_spans_walks_from_its_last_event(self):
        t = build([cupti_kernel(10, 7, 2 * US, 50 * US), cupti_kernel(11, 7, 51 * US, 60 * US),
                   cupti_sync(99, "event", 70 * US, 70 * US + 200)])
        g = cm.build_event_graph(t, cp.load_or_build_graph(t))
        self.assertTrue(all(cm.connected_events(g)[v] for v in g.order))
        w = cm.walk(g)
        self.assertEqual(g.t[w.sink], 70 * US + 200)
        self.assertEqual(w.tail_records, 0)


if __name__ == "__main__":
    unittest.main()
