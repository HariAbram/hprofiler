"""
The opt-in exact-match rule (HPROFILER_EXACT_P2P_HIGH=1, criticalpath.py
EXACT_P2P_ENV): an exact (non-wildcard) MPI match is "high" instead of
"medium" only when the capture of its channel is complete -- both
processes' hooks declare a complete point-to-point capture (counter
mpi_p2p_capture_complete = 1), the channel has as many sends as receives, no events were
lost, and the communicator had no cancelled send, matched probe or
persistent request and is registered. Every case through both edge builders;
off by default.
"""
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis import criticalpath as cp
from src.core.events import Category, CounterEvent, InstantEvent, SpanEvent
from src.core.trace import Trace, TraceMetadata


def _mpi(name, pid, start, dur, sid="", psid="", **tags):
    return SpanEvent(name=name, category=Category.MPI, start_ns=start, duration_ns=dur, pid=pid, tid=pid,
                     tags={k: str(v) for k, v in tags.items()}, span_id=sid, parent_span_id=psid)


def _capture(pid, complete=1):
    return CounterEvent("mpi_p2p_capture_complete", Category.MPI, 1, float(complete), "flag", pid)


def _channel(n_send=3, n_recv=3, commid=0, markers=True, extra=()):
    ev = []
    for k in range(n_send):
        ev.append(_mpi("MPI_Send", 10, 100 + 1000 * k, 10, type="send", rank=0, peer=1, wpeer=1, tag=5,
                       commid=commid))
    for k in range(n_recv):
        ev.append(_mpi("MPI_Recv", 11, 50 + 1000 * k, 200, type="recv", rank=1, peer=0, wpeer=0, tag=5,
                       commid=commid))
    if markers:
        ev += [_capture(10), _capture(11)]
    ev += list(extra)
    t = Trace(TraceMetadata())
    for e in ev:
        t.add(e)
    return t


def _tiers(trace):
    spans, preds = cp.build_dependency_graph(trace)
    mem = sorted(c for v, pl in preds.items() for u, k, c in pl if k == "p2p")
    g = cp.Graph(trace, trace.store.span_seqs(), cp.build_edges(trace))
    streamed = sorted(c for v, pl in g.preds_dict().items() for u, k, c in pl if k == "p2p")
    assert mem == streamed, (mem, streamed)
    return mem


class TestExactRule(unittest.TestCase):
    def _on(self):
        return mock.patch.dict(os.environ, {cp.EXACT_P2P_ENV: "1"})

    def test_off_by_default(self):
        with mock.patch.dict(os.environ, {cp.EXACT_P2P_ENV: ""}):
            self.assertEqual(_tiers(_channel()), ["medium"] * 3)
            self.assertEqual(cp.edges_version(), cp.EDGES_VERSION)

    def test_complete_channel_is_high(self):
        with self._on():
            t = _channel()
            self.assertEqual(_tiers(t), ["high"] * 3)
            self.assertEqual(cp.mpi_dependency_diagnostics(t)["exact_high"], 3)
            self.assertNotEqual(cp.edges_version(), cp.EDGES_VERSION)   # never mixed with flag-off edges

    def test_older_hooks_never_qualify(self):
        with self._on():
            self.assertEqual(_tiers(_channel(markers=False)), ["medium"] * 3)
            t = _channel(markers=False, extra=[_capture(10), _capture(11, complete=0)])
            self.assertEqual(_tiers(t), ["medium"] * 3)

    def test_unequal_counts_are_medium(self):
        with self._on():
            self.assertEqual(_tiers(_channel(n_send=4, n_recv=3)), ["medium"] * 3)

    def test_matched_probe_or_persistent_request_on_the_communicator(self):
        mprobe = InstantEvent("MPI_Mprobe", Category.MPI, 5, 11, 11, {"type": "mprobe", "commid": "0"})
        persistent = _mpi("MPI_Send_init", 10, 3, 1, type="send_init", rank=0, peer=1, tag=9, commid=0)
        with self._on():
            self.assertEqual(_tiers(_channel(extra=[mprobe])), ["medium"] * 3)
            self.assertEqual(_tiers(_channel(extra=[persistent])), ["medium"] * 3)
            # on another communicator: no effect
            other = InstantEvent("MPI_Mprobe", Category.MPI, 5, 11, 11, {"type": "mprobe", "commid": "7"})
            self.assertEqual(_tiers(_channel(extra=[other])), ["high"] * 3)

    def test_cancelled_send_on_the_communicator(self):
        cancelled = [_mpi("MPI_Isend", 10, 20, 5, sid="9", type="isend", rank=0, peer=1, wpeer=1, tag=6,
                          commid=0),
                     _mpi("MPI_Wait", 10, 30, 5, psid="9", type="wait", rank=0, cancelled=1)]
        with self._on():
            self.assertEqual(_tiers(_channel(extra=cancelled)), ["medium"] * 3)

    def test_unregistered_communicator(self):
        with self._on():
            self.assertEqual(_tiers(_channel(commid=-1)), ["medium"] * 3)

    def test_event_loss_still_wins(self):
        with self._on():
            t = _channel()
            t.metadata.capture_health = {"transport": {"10/mpi": {"dropped_full": 1}}}
            self.assertEqual(_tiers(t), ["low"] * 3)

    def test_wildcards_keep_high(self):
        with self._on():
            t = _channel(n_recv=0, extra=[
                _mpi("MPI_Recv", 11, 50 + 1000 * k, 200, type="recv", rank=1, peer=0, wpeer=0, tag=5,
                     commid=0, wildcard=1) for k in range(3)])
            self.assertEqual(_tiers(t), ["high"] * 3)


if __name__ == "__main__":
    unittest.main()
