"""
The Timeline's analysis data (src/gui/timeline_analysis.py, Qt-free) carries
the evidence model's results: each dependency edge's status and number of
alternatives, the spans on the path in every / some alternative world, the
waiting blame and replayed counterfactuals per event, unknown-mode path
steps owned by their call, and the summary's clocks / alternatives /
evidence / blame / probability blocks.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.gui import timeline_analysis as ta
from tests import test_cp_alternatives as tca
from tests.test_cp_model import _trace, recv, send, work


def build(t):
    lanes = {}

    def lane_of(s):
        return lanes.setdefault((s.pid, s.tid), len(lanes))
    return ta.build(t, lane_of, lambda lane: {})


class TestAnalysisEvidence(unittest.TestCase):
    def test_edges_carry_their_status_and_alternatives(self):
        an = build(tca.TestPointToPointAlignment().trace())
        self.assertTrue(an.ok, an.error)
        p2p = [e for e in an.edges if e.kind == "p2p"]
        self.assertTrue(p2p)
        self.assertEqual({e.status for e in p2p}, {"ambiguous"})
        self.assertTrue(all(e.alternatives >= 1 for e in p2p))
        self.assertTrue(an.alternatives_evaluated)
        self.assertTrue(an.robust_every <= an.robust_some)
        sm = an.summary
        self.assertEqual(sm["alternatives"]["units"], {"p2p_alignment": 1})
        self.assertIn("ambiguous", sm["evidence"]["by_status"])
        self.assertEqual((sm["probabilities"]["status"], sm["probabilities"]["calibrated"]), ("unquantified", False))
        self.assertIn("clocks", sm)

    def test_blame_and_unknown_steps(self):
        t = _trace([work("A", 2, 0, 100), send(2, 100, 1, 0, 9), recv(1, 0, 150, 0, 1, 9),
                    recv(1, 150, 260, 0, 1, 11)])                   # the second receive is unmatched
        an = build(t)
        a = next(s for s in t.iter_spans() if s.name == "A")
        self.assertEqual(an.blame_of_eid.get(a.eid), 100)
        unknown = [st for st in an.steps if st.mode == "unknown"]
        self.assertTrue(unknown)
        second = max((s for s in t.iter_spans() if s.name == "MPI_Recv"), key=lambda s: s.start_ns)
        self.assertEqual({st.eid for st in unknown}, {second.eid})     # owned by the call
        self.assertEqual(an.summary["unmatchedByCategory"], {"mpi": 110})


if __name__ == "__main__":
    unittest.main()
