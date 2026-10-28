"""
The Compare tab's bridge (src/gui/comparison.py) on a joint release: ranks 0
and 2 both send to rank 1, which waits for both in one MPI_Waitall; the
candidate slows both senders by the same 10 ms per iteration (5
iterations). The critical path runs through one sender (rank 2: +50 ms
direct impact); rank 0 is off the path (direct impact 0) -- yet the wait
waited for it as much:

  * the wait's details list every origin of the joint release -- the chain
    origin and the joint one -- with each one's direct impact and its share
    of the wait (25 ms each);
  * an origin's details separate direct impact, jointly caused waiting and
    propagated wait blame (waitBlameNs: 10 -> 35 ms), and list the waits it
    released, with the co-origins;
  * the ranked lists mark joint releases and the waits an origin released;
  * every origin, chain step and released wait can be navigated to (its
    candidate timeline range).
"""
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtGui import QGuiApplication
    _QT = True
except ImportError:          # pragma: no cover
    _QT = False

import tests.compare_scenarios as S

MS = 1_000_000


@unittest.skipUnless(_QT, "PySide6 not installed")
class TestJointReleaseAttribution(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QGuiApplication.instance() or QGuiApplication([])
        from src.gui.comparison import ComparisonBridge
        from src.gui.theme import Theme
        cls.b = ComparisonBridge(S.mpi_waitall_two_senders(True), S.mpi_waitall_two_senders(False), Theme())
        rows = cls.b.contributors + cls.b.propagated + cls.b.offCriticalPath
        cls.wait = next(r for r in rows if r["label"] == "MPI_Waitall")
        cls.rank = {r["roles"].split(", ")[1]: r for r in rows if r["label"] == "compute"}

    def test_the_wait_lists_every_origin_with_its_own_values(self):
        d = self.b.contributorDetail(self.wait["id"])
        origins = d["jointOrigins"]
        self.assertEqual(len(origins), 2)
        by = {o["roles"].split(", ")[1]: o for o in origins}
        self.assertEqual(set(by), {"rank0", "rank2"})
        self.assertEqual({o["role"] for o in origins}, {"chain origin", "joint origin"})
        self.assertEqual(by["rank0"]["directNs"], 0)                 # off the critical path
        self.assertEqual(by["rank2"]["directNs"], 50 * MS)
        self.assertEqual({o["shareNs"] for o in origins}, {25 * MS})
        self.assertEqual({o["waitBlameDeltaNs"] for o in origins}, {25 * MS})
        self.assertTrue(all(o["hasTimeline"] and o["available"] for o in origins))

    def test_an_off_path_origin_separates_its_three_attributions(self):
        d = self.b.contributorDetail(self.rank["rank0"]["id"])
        a = d["attribution"]
        self.assertEqual((a["directNs"], a["jointWaitNs"], a["waitBlameDeltaNs"]), (0, 25 * MS, 25 * MS))
        self.assertEqual([w["label"] for w in d["jointWaits"]], ["MPI_Waitall"])
        self.assertEqual(d["jointWaits"][0]["coOrigins"], ["compute (openmp, rank2, main)"])
        row = next(r for r in d["rows"] if r["label"] == "waiting it caused (wait blame)")
        self.assertEqual((row["before"], row["after"], row["delta"], row["kind"]), ("10.00ms", "35.00ms", "+25.00ms", "derived"))
        on_path = self.b.contributorDetail(self.rank["rank2"]["id"])["attribution"]
        self.assertEqual((on_path["directNs"], on_path["jointWaitNs"]), (50 * MS, 25 * MS))

    def test_lists_mark_joint_releases(self):
        self.assertEqual(self.wait["jointOrigins"], 2)
        self.assertEqual((self.rank["rank0"]["jointWaits"], self.rank["rank2"]["jointWaits"]), (1, 1))
        self.assertIn(self.rank["rank0"]["id"], [r["id"] for r in self.b.offCriticalPath])

    def test_every_contributing_identity_is_navigable(self):
        d = self.b.contributorDetail(self.wait["id"])
        ids = [o["id"] for o in d["jointOrigins"]] + [c["identity"] for c in d["chain"]]
        for i in ids:
            t = self.b.identityTimeline(i)
            self.assertTrue(t["hasTimeline"], t)
            self.assertLess(t["timelineStartNs"], t["timelineEndNs"])
        self.assertTrue(all(c["selectable"] for c in d["chain"]))


if __name__ == "__main__":
    unittest.main()
