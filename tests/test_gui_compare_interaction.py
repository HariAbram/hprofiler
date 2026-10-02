"""
Real-QML interaction test for the POPULATED Compare tab: candidate (opened
trace) vs baseline (--compare) on the MPI wait-propagation scenario, driven
by synthesized mouse input in a separate process (tests/gui_compare_driver.py
-- one QML engine / trace pairing per process): ranked contributors render,
clicking one opens its details (explanation, source snippet), "Show in
Timeline" switches to the Timeline zoomed to the contributor's range,
"Open in Source" selects it in the Source tab, phase selection filters the
contributors and "Show phase in Timeline" focuses the phase, changing the
thresholds recomputes the causal ranking, and nothing emits a QML warning.
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

_DRIVER = Path(__file__).resolve().parent / "gui_compare_driver.py"


@unittest.skipUnless(_PYSIDE6, "PySide6 not installed (optional gui extra)")
class TestCompareTabInteraction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
        p = subprocess.run([sys.executable, str(_DRIVER)], env=env, capture_output=True, text=True, timeout=180)
        if p.returncode != 0:
            raise AssertionError(f"driver failed:\n{p.stdout}\n{p.stderr}")
        cls.r = json.loads(p.stdout.strip().splitlines()[-1])

    def test_no_qml_warnings(self):
        self.assertEqual(self.r["warnings"], [])

    def test_ranked_contributors_render_with_origin_first(self):
        self.assertTrue(self.r["available"] and self.r["causal"])
        self.assertEqual(self.r["contributors"][0], "compute")
        self.assertEqual(self.r["listCount"], len(self.r["contributors"]))

    def test_clicking_a_contributor_shows_its_details_and_source(self):
        self.assertEqual(self.r["selectedId"], self.r["firstId"])
        self.assertEqual(self.r["detailLabel"], "compute")
        self.assertIn("measured", self.r["explanation"])
        self.assertGreater(self.r["snippetLines"], 0)

    def test_show_in_timeline_zooms_to_the_contributor_range(self):
        self.assertEqual(self.r["afterTimelineTab"], 1)
        start, end = self.r["rangeNs"]
        self.assertEqual((self.r["focusRange"]["startNs"], self.r["focusRange"]["endNs"]), (start, end))
        self.assertEqual(self.r["selectedName"], "compute")
        v0, v1 = self.r["timelineView"]
        self.assertLessEqual(v0, start)
        self.assertGreaterEqual(v1, end)
        self.assertLess(v1 - v0, 3 * (end - start))       # zoomed in, not the whole run

    def test_open_in_source_selects_it(self):
        self.assertEqual(self.r["afterSourceTab"], 6)
        self.assertEqual(self.r["sourceSelected"], "compute")

    def test_phase_navigation(self):
        self.assertEqual(self.r["phaseSelected"], 2)
        self.assertEqual(self.r["phaseContributors"], ["compute"])
        self.assertTrue(self.r["cpBefore"] and self.r["cpAfter"])
        self.assertEqual(self.r["afterPhaseTab"], 1)
        self.assertEqual((self.r["phaseFocus"]["startNs"], self.r["phaseFocus"]["endNs"]),
                         (self.r["phaseRange"]["startNs"], self.r["phaseRange"]["endNs"]))

    def test_thresholds_recompute_the_causal_ranking(self):
        self.assertEqual(self.r["contributorsAtHugeFloor"], 0)


if __name__ == "__main__":
    unittest.main()
