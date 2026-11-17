"""
MPI sequence numbers and matched probes in the dependency graph
(src/analysis/criticalpath.py _P2PMatcher, _rendezvous_groups), on
synthetic traces in the wire format hooks/mpi_hook/mpi_hook.c emits:

  * sseq= (per sender, communicator and destination), rseq= (per receiver
    and communicator): a complete channel pairs as before; a lost record
    (a missing number) is placed where one channel must own it, so the
    pairs around it stay exact, and otherwise every later pairing of the
    channels it could belong to is "low" -- never silently shifted;
  * channel order from the numbers, wildcard receives among exact ones,
    several communicators, concurrent posts of two threads, an exact
    MPI_Irecv that only MPI_Test completed (it still consumes its send);
  * matched probes (MPI_Mprobe / MPI_Improbe -> MPI_Mrecv / MPI_Imrecv,
    mid=), including a receive of a message whose probe was not recorded;
  * cseq= / csize= on collectives: (commid, cseq) groups across collective
    types, a lost collective record, misaligned numbers;
  * the diagnostics and warnings that report all of it.
Every case is checked with both the in-memory and the streamed builder.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.events import Category, InstantEvent, SpanEvent
from src.core.trace import Trace, TraceMetadata
from src.analysis import criticalpath as cp


def _trace(events):
    t = Trace(TraceMetadata())
    for e in events:
        t.add(e)
    return t


def _mpi(name, pid, start, dur, sid="", psid="", tid=None, **tags):
    return SpanEvent(name=name, category=Category.MPI, start_ns=start, duration_ns=dur, pid=pid,
                     tid=pid if tid is None else tid, tags={k: str(v) for k, v in tags.items()},
                     span_id=sid, parent_span_id=psid)


def send(pid, start, rank, peer, tag, sseq, commid=0, dur=10, **extra):
    t = dict(type="send", rank=rank, peer=peer, wpeer=peer, tag=tag, commid=commid, **extra)
    if sseq is not None:
        t["sseq"] = sseq
    return _mpi("MPI_Send", pid, start, dur, **t)


def recv(pid, start, rank, peer, tag, rseq, commid=0, dur=50, tid=None, **extra):
    t = dict(type="recv", rank=rank, peer=peer, wpeer=peer, tag=tag, commid=commid, **extra)
    if rseq is not None:
        t["rseq"] = rseq
    return _mpi("MPI_Recv", pid, start, dur, tid=tid, **t)


def p2p_edges(trace):
    """[(src span, dst span, tier)] from both builders, which must agree."""
    spans, preds = cp.build_dependency_graph(trace)
    mem = sorted((u, v, c) for v, pl in preds.items() for u, k, c in pl if k == "p2p")
    g = cp.Graph(trace, trace.store.span_seqs(), cp.build_edges(trace))
    streamed = sorted((u, v, c) for v, pl in g.preds_dict().items() for u, k, c in pl if k == "p2p")
    assert mem == streamed, (mem, streamed)
    return [(spans[u], spans[v], c) for u, v, c in mem]


def pairs(trace):
    """{receive label: (send label, tier)}, labels from the msg= tag."""
    return {d.tags["msg"]: (s.tags["msg"], c) for s, d, c in p2p_edges(trace)}


def channel(n, lose_send=(), lose_recv=(), tag=7, commid=0, seq=True):
    """n messages rank 1 -> rank 0 on one channel; records dropped by index."""
    ev = []
    for k in range(n):
        if k not in lose_send:
            ev.append(send(11, 100 * k, 1, 0, tag, k + 1 if seq else None, commid, msg=f"s{k}"))
        if k not in lose_recv:
            ev.append(recv(10, 100 * k + 5, 0, 1, tag, k + 1 if seq else None, commid, msg=f"r{k}"))
    return ev


class TestCompleteAndLostRecords(unittest.TestCase):
    def test_a_complete_channel_pairs_kth_with_kth(self):
        got = pairs(_trace(channel(4)))
        self.assertEqual(got, {f"r{k}": (f"s{k}", "medium") for k in range(4)})

    def test_a_lost_send_record_is_placed_and_later_pairs_stay_exact(self):
        t = _trace(channel(5, lose_send=(1,)))
        got = pairs(t)
        # r1's send is the lost record: no edge; r2..r4 still pair with s2..s4
        self.assertEqual(got, {"r0": ("s0", "medium"), "r2": ("s2", "medium"), "r3": ("s3", "medium"),
                               "r4": ("s4", "medium")})
        diag = cp.mpi_dependency_diagnostics(t)
        self.assertEqual((diag["send_seq_gaps"], diag["seq_gaps_placed"], diag["send_record_lost"]), (1, 1, 1))
        self.assertNotIn("matched_after_gap", diag)

    def test_without_numbers_the_same_loss_shifts_every_later_pair(self):
        got = pairs(_trace(channel(5, lose_send=(1,), seq=False)))
        self.assertEqual(got["r1"][0], "s2")             # the old behaviour this replaces
        self.assertEqual(got["r3"][0], "s4")

    def test_a_lost_receive_record_is_placed(self):
        t = _trace(channel(5, lose_recv=(2,)))
        got = pairs(t)
        self.assertEqual(got, {f"r{k}": (f"s{k}", "medium") for k in (0, 1, 3, 4)})
        self.assertEqual(cp.mpi_dependency_diagnostics(t)["recv_seq_gaps"], 1)

    def test_lost_send_and_receive_on_one_channel_are_both_placed(self):
        got = pairs(_trace(channel(5, lose_send=(0,), lose_recv=(3,))))
        self.assertEqual(got, {"r1": ("s1", "medium"), "r2": ("s2", "medium"), "r4": ("s4", "medium")})

    def test_the_only_short_channel_of_a_pair_owns_its_gap(self):
        # two tags between the same ranks; the lost send (tag 8) is the only
        # channel with more receives than sends
        ev = []
        for k in range(3):
            ev.append(send(11, 200 * k, 1, 0, 7, 2 * k + 1, msg=f"a{k}"))
            if k != 1:
                ev.append(send(11, 200 * k + 100, 1, 0, 8, 2 * k + 2, msg=f"b{k}"))
            ev.append(recv(10, 200 * k + 5, 0, 1, 7, 2 * k + 1, msg=f"ra{k}"))
            ev.append(recv(10, 200 * k + 105, 0, 1, 8, 2 * k + 2, msg=f"rb{k}"))
        got = pairs(_trace(ev))
        self.assertEqual(got["rb2"], ("b2", "medium"))
        self.assertNotIn("rb1", got)
        self.assertEqual({got[f"ra{k}"][0] for k in range(3)}, {"a0", "a1", "a2"})

    def test_an_unplaceable_gap_makes_later_pairs_low_not_shifted_silently(self):
        # two lost sends of one pair, one per tag: both channels are one send
        # short, so which gap belongs to which channel is not decided
        ev, q = [], 0
        for k in range(4):
            for tag, lab in ((7, "a"), (8, "b")):
                q += 1
                if (lab, k) not in (("a", 1), ("b", 2)):
                    ev.append(send(11, 20 * q, 1, 0, tag, q, msg=f"{lab}{k}"))
                ev.append(recv(10, 20 * q + 5, 0, 1, tag, q, msg=f"r{lab}{k}"))
        t = _trace(ev)
        got = pairs(t)
        self.assertEqual(got["ra0"], ("a0", "medium"))     # before the gap: untouched
        self.assertEqual(got["rb0"], ("b0", "medium"))
        self.assertEqual(got["ra1"][1], "low")               # after it: may be shifted
        self.assertEqual(got["rb1"][1], "low")
        diag = cp.mpi_dependency_diagnostics(t)
        self.assertGreaterEqual(diag["matched_after_gap"], 2)
        self.assertTrue(any("may be shifted" in w for w in cp.dependency_warnings(diag)))

    def test_a_gap_on_one_communicator_leaves_another_alone(self):
        other = channel(4, commid=5)
        for e in other:
            e.tags["msg"] = "c5" + e.tags["msg"]
        got = pairs(_trace(channel(4, lose_send=(1,), commid=0) + other))
        self.assertEqual({k: v for k, v in got.items() if k.startswith("c5")},
                         {f"c5r{k}": (f"c5s{k}", "medium") for k in range(4)})
        self.assertEqual(got["r2"], ("s2", "medium"))


class TestOrderWildcardsAndThreads(unittest.TestCase):
    def test_numbers_order_the_channel_even_against_timestamps(self):
        # the sender's clock order disagrees with its post order: the numbers win
        ev = [send(11, 50, 1, 0, 7, 1, msg="s0"), send(11, 40, 1, 0, 7, 2, msg="s1"),
              recv(10, 60, 0, 1, 7, 1, msg="r0"), recv(10, 70, 0, 1, 7, 2, msg="r1")]
        self.assertEqual(pairs(_trace(ev)), {"r0": ("s0", "medium"), "r1": ("s1", "medium")})

    def test_a_wildcard_receive_between_exact_ones_takes_its_channel_slot(self):
        ev = [send(11, 0, 1, 0, 7, 1, msg="s0"), send(11, 10, 1, 0, 7, 2, msg="s1"),
              send(11, 20, 1, 0, 7, 3, msg="s2"),
              recv(10, 1, 0, 1, 7, 1, msg="r0"),
              _mpi("MPI_Irecv", 10, 2, 1, sid="9", type="irecv", rank=0, peer=-2, tag=-1, commid=0,
                   wildcard=1, rseq=2),
              recv(10, 30, 0, 1, 7, 3, msg="r2"),
              _mpi("MPI_Wait", 10, 40, 5, psid="9", type="wait", rank=0, rpeer=1, rtag=7, rwpeer=1,
                   msg="w1")]
        got = pairs(_trace(ev))
        self.assertEqual(got, {"r0": ("s0", "medium"), "w1": ("s1", "high"), "r2": ("s2", "medium")})

    def test_an_unresolved_wildcard_counts_as_a_receive_gap(self):
        ev = [send(11, 0, 1, 0, 7, 1, msg="s0"), send(11, 10, 1, 0, 7, 2, msg="s1"),
              send(11, 20, 1, 0, 7, 3, msg="s2"),
              recv(10, 1, 0, 1, 7, 1, msg="r0"),
              _mpi("MPI_Irecv", 10, 2, 1, sid="9", type="irecv", rank=0, peer=-2, tag=-1, commid=0,
                   wildcard=1, rseq=2),                     # no completion recorded
              recv(10, 30, 0, 1, 7, 3, msg="r2")]
        t = _trace(ev)
        self.assertEqual(pairs(t)["r2"], ("s2", "medium"))   # placed: the only channel into rank 0
        self.assertEqual(cp.mpi_dependency_diagnostics(t)["recv_seq_gaps"], 1)

    def test_an_exact_irecv_completed_by_test_only_still_consumes_its_send(self):
        ev = [send(11, 0, 1, 0, 7, None, msg="s0"), send(11, 10, 1, 0, 7, None, msg="s1"),
              _mpi("MPI_Irecv", 10, 1, 1, sid="9", type="irecv", rank=0, peer=1, wpeer=1, tag=7, commid=0),
              recv(10, 20, 0, 1, 7, None, msg="r1")]
        t = _trace(ev)
        t.add(InstantEvent(name="MPI_Test", category=Category.MPI, timestamp_ns=15, pid=10, tid=10,
                           tags={"type": "test", "flag": "1", "rank": "0", "psid": "9"}))
        self.assertEqual(pairs(t), {"r1": ("s1", "medium")})

    def test_overlapping_posts_of_two_threads_are_unordered(self):
        ev = [send(11, 0, 1, 0, 7, 1, msg="s0"), send(11, 10, 1, 0, 7, 2, msg="s1"),
              send(11, 20, 1, 0, 7, 3, msg="s2"),
              recv(10, 5, 0, 1, 7, 1, tid=101, dur=100, msg="r0"),
              recv(10, 6, 0, 1, 7, 2, tid=102, dur=100, msg="r1"),     # overlaps r0, other thread
              recv(10, 300, 0, 1, 7, 3, tid=101, msg="r2")]
        t = _trace(ev)
        got = pairs(t)
        self.assertEqual((got["r0"][1], got["r1"][1], got["r2"][1]), ("low", "low", "medium"))
        self.assertEqual(cp.mpi_dependency_diagnostics(t)["concurrent_posts"], 2)

    def test_sequential_posts_of_two_threads_keep_their_order(self):
        ev = [send(11, 0, 1, 0, 7, 1, msg="s0"), send(11, 10, 1, 0, 7, 2, msg="s1"),
              recv(10, 5, 0, 1, 7, 1, tid=101, dur=10, msg="r0"),
              recv(10, 50, 0, 1, 7, 2, tid=102, dur=10, msg="r1")]
        self.assertEqual(pairs(_trace(ev)), {"r0": ("s0", "medium"), "r1": ("s1", "medium")})


class TestMatchedProbes(unittest.TestCase):
    def probe_trace(self, *, unlinked=False):
        ev = [send(11, 0, 1, 0, 40, 1, msg="s0"), send(11, 10, 1, 0, 40, 2, msg="s1"),
              send(11, 20, 1, 0, 40, 3, msg="s2"),
              # MPI_Mprobe (blocking span, sid=mid) + MPI_Mrecv (psid=mid)
              _mpi("MPI_Mprobe", 10, 5, 20, sid="m1", type="mprobe", rank=0, peer=1, tag=-1, commid=0,
                   flag=1, wildcard=1, rpeer=1, rtag=40, rwpeer=1, rseq=1, msg="p0"),
              _mpi("MPI_Mrecv", 10, 30, 3, psid="m1", type="mrecv", rank=0, commid=0, rpeer=1, rtag=40,
                   rwpeer=1, rseq=1, via="mprobe", wildcard=1, msg="mr0"),
              # MPI_Improbe (an instant) + MPI_Mrecv carrying its channel
              _mpi("MPI_Mrecv", 10, 50, 3, psid="m2", type="mrecv", rank=0, commid=0, rpeer=1, rtag=40,
                   rwpeer=1, rseq=2, via="improbe", msg="mr1"),
              # MPI_Improbe + MPI_Imrecv + MPI_Wait
              _mpi("MPI_Imrecv", 10, 60, 1, sid="q3", psid="m3", type="imrecv", rank=0, commid=0, rpeer=1,
                   rtag=40, rwpeer=1, rseq=3, via="improbe", msg="ir2"),
              _mpi("MPI_Wait", 10, 70, 5, psid="q3", type="wait", rank=0, msg="w2")]
        if unlinked:
            ev += [send(11, 30, 1, 0, 40, 4, msg="s3"), send(11, 40, 1, 0, 40, 5, msg="s4"),
                   _mpi("MPI_Mrecv", 10, 80, 3, type="mrecv", rank=0, mid_unknown=1, msg="mx"),
                   recv(10, 90, 0, 1, 40, 4, msg="r4")]
        return _trace(ev)

    def test_each_probed_message_pairs_in_channel_order(self):
        got = pairs(self.probe_trace())
        # the blocking probe waited for s0; the others land on what completed them
        self.assertEqual(got, {"p0": ("s0", "high"), "mr1": ("s1", "medium"), "w2": ("s2", "medium")})

    def test_the_probe_links_to_its_receive(self):
        spans, preds = cp.build_dependency_graph(self.probe_trace())
        names = {i: s.tags.get("msg") for i, s in enumerate(spans)}
        links = {(names[u], names[v]) for v, pl in preds.items() for u, k, _c in pl if k == "explicit_span_id"}
        self.assertIn(("p0", "mr0"), links)

    def test_a_receive_of_an_unseen_probe_is_reported_and_its_channel_is_low(self):
        t = self.probe_trace(unlinked=True)
        got = pairs(t)
        self.assertEqual(got["r4"][1], "low")            # s3 went to the unrecorded probe: shifted?
        diag = cp.mpi_dependency_diagnostics(t)
        self.assertEqual(diag["matched_receive_unlinked"], 1)
        self.assertTrue(any("mid_unknown" in w for w in cp.dependency_warnings(diag)))


def coll(pid, start, rank, typ, cseq, commid=0, csize=3, dur=10, name=None, sid="", **extra):
    t = dict(type=typ, rank=rank, commid=commid, csize=csize, **extra)
    if cseq is not None:
        t["cseq"] = cseq
    return _mpi(name or f"MPI_{typ.capitalize()}", pid, start, dur, sid=sid, **t)


def arrival_groups(trace):
    spans, preds = cp.build_dependency_graph(trace)
    g = cp.Graph(trace, trace.store.span_seqs(), cp.build_edges(trace))
    mem = sorted((u, v, c) for v, pl in preds.items() for u, k, c in pl if k == "arrival")
    streamed = sorted((u, v, c) for v, pl in g.preds_dict().items() for u, k, c in pl if k == "arrival")
    assert mem == streamed, (mem, streamed)
    return [(spans[u].tags.get("msg"), spans[v].tags.get("msg"), c) for u, v, c in mem]


class TestCollectiveSequence(unittest.TestCase):
    def three_ranks(self, lose=None):
        """allreduce, barrier, allreduce on 3 ranks; rank 2 arrives last."""
        ev = []
        for rank, pid in ((0, 20), (1, 21), (2, 22)):
            for q, typ in ((1, "allreduce"), (2, "barrier"), (3, "allreduce")):
                if lose == (rank, q):
                    continue
                ev.append(coll(pid, 1000 * q + 10 * rank, rank, typ, q, msg=f"{typ}{q}.{rank}"))
        return _trace(ev)

    def test_each_numbered_collective_is_one_group(self):
        got = arrival_groups(self.three_ranks())
        self.assertIn(("barrier2.2", "barrier2.0", "high"), got)
        self.assertIn(("allreduce3.2", "allreduce3.1", "high"), got)
        self.assertEqual(len(got), 6)

    def test_a_lost_record_spoils_only_its_own_collective(self):
        t = self.three_ranks(lose=(2, 2))               # rank 2's barrier record lost
        got = arrival_groups(t)
        self.assertIn(("allreduce3.2", "allreduce3.0", "high"), got)   # still grouped exactly
        barrier = [c for s, d, c in got if s.startswith("barrier")]
        self.assertEqual(barrier, ["low"])               # two of three members: last arriver unknown
        diag = cp.mpi_dependency_diagnostics(t)
        self.assertEqual((diag["collective_seq_gaps"], diag["collectives_incomplete"]), (1, 1))

    def test_misaligned_numbers_fall_back_to_call_order(self):
        ev = [coll(20, 1000, 0, "allreduce", 1, msg="a.0"), coll(21, 1010, 1, "barrier", 1, msg="b.1")]
        diag = cp.mpi_dependency_diagnostics(_trace(ev))
        self.assertEqual(diag["collective_seq_misaligned"], 1)

    def test_a_persistent_collective_start_is_a_member_like_its_nonblocking_form(self):
        ev = []
        for rank, pid in ((0, 20), (1, 21)):
            ev.append(_mpi("MPI_Start", pid, 100 + 10 * rank, 2, sid=f"rq{rank}", type="start", rank=rank,
                           ptype="coll", ctype="allreduce", commid=0, csize=2, cseq=2, msg=f"st.{rank}"))
            ev.append(_mpi("MPI_Wait", pid, 200, 20, psid=f"rq{rank}", type="wait", rank=rank, msg=f"w.{rank}"))
        got = arrival_groups(_trace(ev))
        self.assertIn(("st.1", "w.0", "high"), got)


if __name__ == "__main__":
    unittest.main()
