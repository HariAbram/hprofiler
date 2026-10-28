"""
QML-visible behaviour of the critical-path evidence in the real Main.qml
(drivers run in their own processes: one QML engine / trace each --
tests/gui_evidence_driver.py, tests/gui_compare_joint_driver.py):

  * the bottom panel's Critical path tab: the four variants by their
    semantics, the modeled range with the variants at its ends, the
    span-level search value apart; alternative worlds "evaluated" with
    their worlds; edge statuses with counts and durations; clock domains
    with the applied shift; an undecidable step; sensitivity rows;
  * a large trace: "not evaluated" everywhere (state chip, edge details,
    event details, the worlds overlay marks nothing) until "Evaluate
    alternative worlds" is clicked -- busy, then evaluated, no freeze;
  * the details panel: an ambiguous edge's status, match group, other
    candidates and evidence tree; an unknown blocking call vs a gated one
    with its tail; the hover note;
  * calibrated and uncalibrated probabilities;
  * the Compare tab on a joint release: every origin, the off-path one
    included, its three attributions, the wait-blame row, and navigation
    to the Timeline;
  * no QML warning anywhere.
"""
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

try:
    import PySide6  # noqa: F401
    _PYSIDE6 = True
except ImportError:
    _PYSIDE6 = False

_HERE = Path(__file__).resolve().parent


def _run(script: str, *args: str) -> dict:
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    env.pop("HPROFILER_EDGE_CALIBRATION", None)
    p = subprocess.run([sys.executable, str(_HERE / script), *args], env=env, capture_output=True, text=True,
                       timeout=300)
    if p.returncode != 0:
        raise AssertionError(f"{script} {args} failed:\n{p.stdout[-3000:]}\n{p.stderr[-3000:]}")
    return json.loads(p.stdout.strip().splitlines()[-1])


@unittest.skipUnless(_PYSIDE6, "PySide6 not installed (optional gui extra)")
class TestEvidenceTimeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = _run("gui_evidence_driver.py", "default")

    def test_no_qml_warnings(self):
        self.assertTrue(self.r["ready"])
        self.assertEqual(self.r["warnings"], [])

    def test_variants_range_and_no_bound_labels(self):
        r = self.r
        self.assertIn("Modeled range: 1.35ms (direct-evidence edges only) to 2.40ms (gated calls' tails as "
                      "processing)", r["rangeText"])
        self.assertEqual(r["variantModeled"], {"default": "1.70ms", "unrecorded_as_work": "1.90ms · headline",
                                               "direct_evidence_only": "1.35ms · range min",
                                               "gated_tail_as_work": "2.40ms · range max"})
        cells = " ".join(r["cpNumbers"])
        self.assertNotIn("lower bound", cells)
        self.assertNotIn("upper bound", cells)
        self.assertNotIn("Span-level", cells)          # the search value has its own line
        self.assertTrue(r["legacyText"].startswith("Span-level search value"))
        self.assertIn("1.35ms – 2.40ms", r["cpNumbers"])

    def test_alternatives_evidence_clocks_sensitivity_sections(self):
        r = self.r
        self.assertEqual(r["altState"], "evaluated")
        self.assertEqual(r["worldRows"], 4)
        self.assertTrue(r["evidenceOpen"])
        for s in ("observed", "inferred", "ambiguous", "contradicted", "unsupported"):
            self.assertIn(s, r["statusCells"])
        i = r["statusCells"].index("ambiguous")
        self.assertEqual(r["statusCells"][i:i + 6], ["ambiguous", "1", "1", "600.0µs", "600.0µs", "—"])
        self.assertIn("opencl/0/9", r["clockCells"])
        self.assertIn("−400.0µs", r["clockCells"])
        self.assertEqual(r["undecidableRows"], 1)
        self.assertIn("modeled cut", r["sensitivityCells"])
        self.assertIn("replayed", r["sensitivityCells"])
        self.assertEqual(r["passedRows"], ["↪ cudaStreamSynchronize"])

    def test_ambiguous_edge_details(self):
        f = self.r["edgeFields"]
        self.assertTrue(f["status"].startswith("ambiguous"))
        self.assertEqual(f["group"], "p2p question: exactly one candidate is the answer (2 candidate(s))")
        self.assertEqual(f["alternatives"], "1 other candidate(s) for the same question")
        self.assertEqual(self.r["edgeCandidates"], 1)
        tree = self.r["edgeTreeLines"]
        self.assertTrue(any(t.startswith("✓ posting_order") for t in tree))
        self.assertTrue(any(t.startswith("✗ several_candidates") for t in tree))

    def test_unknown_blocking_vs_gated_waiting(self):
        r = self.r
        self.assertTrue(r["unknownFields"]["Blocking call"].startswith("unknown"))
        self.assertNotIn("Gated tail", r["unknownFields"])
        self.assertIn("unknown blocking", r["unknownHover"])
        self.assertTrue(r["gatedFields"]["Blocking call"].startswith("gated: released at +300.0µs"))
        self.assertEqual(r["gatedFields"]["Gated tail"],
                         "at most 100.0µs of its 200.0µs may have been processing (the time after its release)")
        self.assertIn("unknown", r["cpModes"])
        self.assertEqual(r["unknownVeils"], 1)

    def test_membership_marks_and_chip(self):
        r = self.r
        self.assertTrue(all(r["marks"][k] > 0 for k in ("variants", "near", "worlds")))
        self.assertEqual(r["chip"]["variants"], "strip: path in every variant · hatched: some only")
        self.assertIn("every evaluated world", r["chip"]["worlds"])

    def test_uncalibrated_probability(self):
        self.assertTrue(self.r["probText"].startswith("unquantified"))
        self.assertEqual(self.r["edgeFields"]["probability"], "unquantified (no calibration given)")


@unittest.skipUnless(_PYSIDE6, "PySide6 not installed (optional gui extra)")
class TestLargeTraceAlternatives(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = _run("gui_evidence_driver.py", "big")

    def test_not_evaluated_until_asked(self):
        r = self.r
        self.assertEqual(r["warnings"], [])
        self.assertEqual(r["altState"], "not evaluated")
        self.assertIn("exceed the automatic limit", r["altReason"])
        self.assertTrue(r["edgeFields"]["alternatives"].startswith("not evaluated"))
        self.assertIn("(alternatives not evaluated)", r["edgeFields"]["status"])
        self.assertTrue(r["beforeEvalOnPath"].startswith("— not evaluated"))
        self.assertEqual(r["marks"]["worlds"], 0)
        self.assertEqual(r["chipNotEvaluated"], "alternative worlds not evaluated: nothing marked")
        self.assertEqual(r["evaluateButton"], "Evaluate alternative worlds")

    def test_evaluating_on_request(self):
        r = self.r
        self.assertTrue(r["busySeen"])
        self.assertTrue(r["evalOk"])
        self.assertEqual(r["altStateAfter"], "evaluated")
        self.assertGreaterEqual(r["worldRowsAfter"], 2)
        self.assertIn("evaluated world", r["afterEvalOnPath"])
        self.assertTrue(r["jobMessage"].endswith("world(s) evaluated"))


@unittest.skipUnless(_PYSIDE6, "PySide6 not installed (optional gui extra)")
class TestCalibratedProbabilities(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = _run("gui_evidence_driver.py", "calibrated")

    def test_calibration_source_version_model_coverage(self):
        r = self.r
        self.assertEqual(r["warnings"], [])
        t = r["probText"]
        self.assertTrue(t.startswith("calibrated · source "))
        for s in ("dataset synthetic GUI test scenarios", "(current)", "isotonic regression", "1 strata",
                  "graph edges quantified", "expected wrong"):
            self.assertIn(s, t)
        self.assertEqual(r["edgeFields"]["probability"], "0.55, calibrated on synthetic GUI test scenarios")
        self.assertIn("[see Probabilities]", r["cpNumbers"])
        self.assertNotIn("[no calibration given]", r["cpNumbers"])


@unittest.skipUnless(_PYSIDE6, "PySide6 not installed (optional gui extra)")
class TestCompareJointRelease(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = _run("gui_compare_joint_driver.py")

    def test_no_qml_warnings(self):
        self.assertEqual(self.r["warnings"], [])

    def test_every_origin_of_the_joint_release_is_listed(self):
        self.assertTrue(self.r["jointPanelVisible"])
        self.assertEqual(self.r["jointRows"], ["compute (openmp, rank0, main)", "compute (openmp, rank2, main)"])

    def test_the_off_path_origin_shows_three_separate_attributions(self):
        r = self.r
        self.assertEqual(r["selectedAfterClick"], r["rank0Id"])
        self.assertEqual(r["rank0Attribution"], ["+0ns", "+25.00ms (share of 1 wait(s))", "+25.00ms"])
        self.assertTrue(r["rank0JointWaitsVisible"])
        self.assertEqual(r["rank0Rows"]["waiting it caused (wait blame)"], ["10.00ms", "35.00ms", "+25.00ms"])

    def test_navigation_to_the_contributing_event(self):
        r = self.r
        self.assertEqual(r["afterTimelineTab"], 1)
        self.assertEqual([r["focusRange"]["startNs"], r["focusRange"]["endNs"]], r["rank0Range"])
        self.assertEqual(r["selectedName"], "compute")


if __name__ == "__main__":
    unittest.main()
