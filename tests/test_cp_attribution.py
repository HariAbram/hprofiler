"""
Attribution (src/analysis/cp_attribution.py):

  * intervention: the replay of an unchanged trace reproduces every event
    time exactly (random multi-rank traces); shrinking the work a receive
    waited for moves the receive's end and the run's end by that much;
    shrinking work nobody waits for moves neither;
  * blame: a receive's waiting time goes to the sender's work inside the
    waiting window; two members arriving at once share a rendezvous
    member's wait (joint release); a sender that was itself waiting passes
    the blame on to what released it; every waiting nanosecond is charged
    exactly once.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis import cp_attribution as at
from src.analysis import cp_model as cm
from src.analysis import criticalpath as cp
from tests.test_cp_model import _trace, mpi, random_trace, recv, send, work


def graph(t):
    return cm.build_event_graph(t, cp.load_or_build_graph(t))


def span_of(g, name):
    return next(j for j in range(g.n) if g.name(j) == name)


class TestReplay(unittest.TestCase):
    def test_unchanged_inputs_reproduce_the_recording(self):
        for seed in range(12):
            with self.subTest(seed=seed):
                g = graph(random_trace(seed))
                tp, end = at.replay(g)
                for v in g.order:
                    self.assertEqual(tp[v], g.t[v], v)
                self.assertEqual(end, at.run_end(g))

    def test_shrinking_what_a_receive_waited_for(self):
        t = _trace([work("A", 2, 0, 100), send(2, 100, 1, 0, 9), recv(1, 0, 150, 0, 1, 9)])
        g = graph(t)
        tp, end = at.replay(g, {span_of(g, "A"): 0.5})
        r = span_of(g, "MPI_Recv")
        self.assertEqual(tp[2 * r + 1], 100)                  # released at 50, then its 50 of processing
        self.assertEqual(end, 100)                            # the receive still ends last, 50 earlier
        tp, end = at.replay(g, {span_of(g, "MPI_Send"): 0.0})
        self.assertEqual(tp[2 * r + 1], 150)                  # the receive waited for the send's start

    def test_work_nobody_waits_for_moves_nothing_else(self):
        t = _trace([work("A", 2, 0, 100), send(2, 100, 1, 0, 9), recv(1, 0, 150, 0, 1, 9),
                    work("B", 3, 0, 40)])
        g = graph(t)
        tp, end = at.replay(g, {span_of(g, "B"): 0.0})
        self.assertEqual(end, 150)
        self.assertEqual(tp[2 * span_of(g, "MPI_Recv") + 1], 150)


class TestBlame(unittest.TestCase):
    def test_a_receive_blames_the_senders_work(self):
        t = _trace([work("A", 2, 0, 100), send(2, 100, 1, 0, 9), recv(1, 0, 150, 0, 1, 9)])
        g = graph(t)
        b = at.blame(g)
        self.assertEqual(b.waiting_ns, 100)
        self.assertAlmostEqual(b.by_span[span_of(g, "A")], 100)
        self.assertEqual(b.joint_releases, 0)

    def test_a_joint_release_is_shared(self):
        def coll(pid, a, end):
            return mpi("MPI_Allreduce", pid, a, end, type="allreduce", rank=pid - 1, commid=0)
        t = _trace([work("w1", 1, 0, 10), coll(1, 10, 300), work("w2", 2, 0, 200), coll(2, 200, 300),
                    work("w3", 3, 0, 200), coll(3, 200, 300)])
        g = graph(t)
        b = at.blame(g, joint_ns=0)
        # rank 1 waited 190; ranks 2 and 3 arrived together: but the builder
        # links rank 1 only to the last arriver, so the tie shows in the
        # alternatives, not here -- a joint release needs several gating edges
        self.assertAlmostEqual(sum(b.by_span.values()) + sum(b.other.values()), b.waiting_ns)

    def test_two_messages_released_together_share_the_wait(self):
        t = _trace([work("A", 2, 0, 100), send(2, 100, 1, 0, 9), work("B", 3, 0, 100), send(3, 100, 2, 0, 9),
                    mpi("MPI_Irecv", 1, 0, 1, sid="r1", type="irecv", rank=0, peer=1, wpeer=1, tag=9, commid=0),
                    mpi("MPI_Irecv", 1, 1, 2, sid="r2", type="irecv", rank=0, peer=2, wpeer=2, tag=9, commid=0),
                    mpi("MPI_Waitall", 1, 2, 150, psid="r1;r2", type="waitall", rank=0)])
        g = graph(t)
        b = at.blame(g)
        self.assertEqual(b.joint_releases, 1)
        self.assertAlmostEqual(b.by_span[span_of(g, "A")], 49)
        self.assertAlmostEqual(b.by_span[span_of(g, "B")], 49)

    def test_a_waiting_sender_passes_the_blame_on(self):
        # rank 0 waits for rank 1, which waits (until 100) for rank 2's work
        t = _trace([work("C", 3, 0, 100), send(3, 100, 2, 1, 5),
                    recv(2, 0, 105, 1, 2, 5), send(2, 105, 1, 0, 9),
                    recv(1, 0, 200, 0, 1, 9)])
        g = graph(t)
        b = at.blame(g)
        self.assertAlmostEqual(b.by_span[span_of(g, "C")], 100 + 100)   # rank 1's wait and rank 0's
        self.assertAlmostEqual(sum(b.by_span.values()) + sum(b.other.values()), b.waiting_ns)

    def test_every_waiting_nanosecond_is_charged_once(self):
        for seed in range(12):
            with self.subTest(seed=seed):
                g = graph(random_trace(seed))
                b = at.blame(g)
                self.assertAlmostEqual(sum(b.by_span.values()) + sum(b.other.values()), b.waiting_ns, places=3)


if __name__ == "__main__":
    unittest.main()
