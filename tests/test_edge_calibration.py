"""
The calibration method (src/analysis/edge_calibration.py):

  * pool-adjacent-violators gives the monotone least-squares fit (pooling
    violators and tied scores); predictions step to the next lower fitted
    score;
  * expected calibration error, Brier score and log loss on known values;
  * a stratum with too few examples stays unquantified; a calibration
    survives a JSON round trip;
  * labelled examples come from an evidence set with alternatives;
  * without a calibration the path's probabilities are "unquantified".
"""
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis import cp_alternatives as alt
from src.analysis import cp_model as cm
from src.analysis import criticalpath as cp
from src.analysis import edge_calibration as ec
from tests import test_cp_alternatives as tca


class TestIsotonic(unittest.TestCase):
    def test_monotone_input_is_kept(self):
        xs, p = ec.pav([1, 2, 3], [0.0, 0.5, 1.0])
        self.assertEqual((xs, p), ([1, 2, 3], [0.0, 0.5, 1.0]))

    def test_violators_are_pooled(self):
        xs, p = ec.pav([1, 2, 3, 4], [0.0, 1.0, 0.0, 1.0])
        self.assertEqual(xs, [1, 2, 3, 4])
        self.assertEqual(p, [0.0, 0.5, 0.5, 1.0])
        self.assertEqual(p, sorted(p))

    def test_ties_are_pooled_first(self):
        xs, p = ec.pav([2, 1, 2, 2], [1.0, 0.0, 0.0, 1.0])
        self.assertEqual(xs, [1, 2])
        self.assertAlmostEqual(p[1], 2 / 3)

    def test_prediction_steps_down(self):
        cal = ec.Calibration(strata={"p2p/posting_order": {"x": [100.0, 300.0], "p": [0.2, 0.9], "n": 30}})
        self.assertEqual(cal.probability("p2p", "posting_order", 50), 0.2)     # below every fitted score
        self.assertEqual(cal.probability("p2p", "posting_order", 299), 0.2)
        self.assertEqual(cal.probability("p2p", "posting_order", 410), 0.9)
        self.assertIsNone(cal.probability("arrival", "call_order", 300))


class TestQuality(unittest.TestCase):
    def test_values(self):
        probs, labels = [0.9, 0.2, 0.8, 0.1], [True, False, False, False]
        self.assertAlmostEqual(ec.brier(probs, labels), (0.01 + 0.04 + 0.64 + 0.01) / 4)
        self.assertAlmostEqual(ec.log_loss([0.5, 0.5], [True, False]), math.log(2))
        # one bin per prediction: |0.9-1|, |0.2-0|, |0.8-0|, |0.1-0|
        self.assertAlmostEqual(ec.ece(probs, labels), (0.1 + 0.2 + 0.8 + 0.1) / 4)
        self.assertEqual(ec.ece([], []), 0.0)

    def test_evidence_score_orders_status_then_tier_then_class(self):
        s = ec.evidence_score
        self.assertGreater(s("observed", "low", "inferred"), s("inferred", "certain", "direct"))
        self.assertGreater(s("inferred", "high", "inferred"), s("inferred", "medium", "direct"))
        self.assertGreater(s("inferred", "high", "direct"), s("inferred", "high", "inferred"))
        self.assertGreater(s("unsupported", "low", "inferred"), s("contradicted", "certain", "direct"))


class TestCalibration(unittest.TestCase):
    def examples(self, n, stratum=("p2p", "posting_order")):
        out = []
        for i in range(n):
            score = 310 if i % 2 else 110
            out.append(ec.Labeled(*stratum, score, label=score == 310 or i % 4 == 0))
        return out

    def test_too_few_examples_stay_unquantified(self):
        cal = ec.fit(self.examples(10), dataset="synthetic", min_examples=20)
        self.assertEqual(cal.strata, {})
        self.assertEqual(ec.evaluate(cal, self.examples(4))["unquantified"], 4)

    def test_fit_round_trip_and_evaluation(self):
        cal = ec.fit(self.examples(40), dataset="synthetic", min_examples=20)
        self.assertEqual(set(cal.strata), {"p2p/posting_order"})
        self.assertEqual(cal.probability("p2p", "posting_order", 310), 1.0)
        self.assertAlmostEqual(cal.probability("p2p", "posting_order", 110), 0.5)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "cal.json")
            cal.save(path)
            back = ec.Calibration.load(path)
        self.assertEqual(back.to_dict(), cal.to_dict())
        q = ec.evaluate(cal, self.examples(40))
        self.assertEqual((q["quantified"], q["unquantified"]), (40, 0))
        self.assertAlmostEqual(q["ece"], 0.0)


class TestFromTraces(unittest.TestCase):
    def test_labelled_examples_from_alternatives(self):
        t = tca.TestPointToPointAlignment().trace()
        graph = cp.load_or_build_graph(t)
        res = cm.analyze_model(t, graph, sensitivity=0)
        a = alt.analyze_alternatives(t, graph, res, samples=0)
        spans = list(t.iter_spans())
        sends = [i for i, s in enumerate(spans) if s.name == "MPI_Send"]
        recvs = [i for i, s in enumerate(spans) if s.name == "MPI_Recv"]
        truth = {(sends[0], recvs[0], "p2p"), (sends[2], recvs[1], "p2p")}   # the second send was never received
        ex = [e for e in ec.labeled_examples(a.evidence, truth) if e.kind == "p2p"]
        self.assertEqual(len(ex), len([1 for r in a.evidence.cand.tolist() if r[2] == cp._KIND_CODE["p2p"]]))
        self.assertEqual(sum(e.label for e in ex), 2)

    def test_the_path_is_unquantified_without_a_calibration(self):
        t = tca.TestPointToPointAlignment().trace()
        rep = cp.analyze(t)
        self.assertEqual(rep.edge_probabilities, {"status": "unquantified"})


if __name__ == "__main__":
    unittest.main()
