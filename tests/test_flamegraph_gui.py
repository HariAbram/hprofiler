"""
Tests for the standalone `hprofiler flamegraph --gui` popup: the
folded-stacks-to-QML-tree conversion (src/output/flamegraph.py's
build_qml_tree(), added alongside this feature) and
src/gui/flamegraph_bridge.py's FlameGraphBridge.

QML-level interaction (click to zoom, hover, Escape to reset, search
highlighting) was verified manually via QTest.mouseMove/mouseClick/
keyClick against a real rendered window during development (see
project_qml_gui memory) -- not re-encoded as a permanent test here,
matching the judgment call already made for the main dashboard's Roofline/
Profile screens (screenshot-verified, not permanently interaction-tested)
while tests/test_gui_timeline_hover.py stayed a permanent QML test
specifically because THAT bug (unqualified property access across a
MouseArea/parent boundary) was silent and easy to reintroduce by editing
the same pattern; FlameGraphWindow.qml's hover/click handlers use the
same `canvas.hitTest(...)`-via-id qualification throughout (not bare
`hitTest(...)`), so it isn't the exact same risk.

Skipped entirely if PySide6 isn't installed, matching this project's
other GUI tests.
"""
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtGui import QGuiApplication
    _PYSIDE6_AVAILABLE = True
except ImportError:
    _PYSIDE6_AVAILABLE = False

from src.output.flamegraph import build_qml_tree


class TestBuildQmlTree(unittest.TestCase):
    def test_simple_two_branch_stack(self):
        folded = "main;foo;bar 50\nmain;foo;baz 30\nmain;qux 20\n"
        tree = build_qml_tree(folded)
        self.assertEqual(tree["name"], "all")
        self.assertEqual(tree["value"], 100)
        main = tree["children"][0]
        self.assertEqual(main["name"], "main")
        self.assertEqual(main["value"], 100)
        names = {c["name"] for c in main["children"]}
        self.assertEqual(names, {"foo", "qux"})
        foo = next(c for c in main["children"] if c["name"] == "foo")
        self.assertEqual(foo["value"], 80)
        self.assertEqual({c["name"] for c in foo["children"]}, {"bar", "baz"})

    def test_children_sorted_by_value_descending(self):
        folded = "main;small 5\nmain;big 100\nmain;medium 40\n"
        tree = build_qml_tree(folded)
        main = tree["children"][0]
        values = [c["value"] for c in main["children"]]
        self.assertEqual(values, sorted(values, reverse=True))
        self.assertEqual(main["children"][0]["name"], "big")

    def test_empty_input_returns_none(self):
        self.assertIsNone(build_qml_tree(""))
        self.assertIsNone(build_qml_tree("   \n  \n"))

    def test_malformed_lines_are_skipped_not_fatal(self):
        folded = "main;foo 10\nnot a valid line at all\nmain;bar notanumber\nmain;baz 5\n"
        tree = build_qml_tree(folded)
        self.assertIsNotNone(tree)
        main = tree["children"][0]
        names = {c["name"] for c in main["children"]}
        self.assertEqual(names, {"foo", "baz"})

    def test_single_frame_stack(self):
        tree = build_qml_tree("only_frame 42\n")
        main_child = tree["children"][0]
        self.assertEqual(main_child["name"], "only_frame")
        self.assertEqual(main_child["value"], 42)
        self.assertEqual(main_child["children"], [])


@unittest.skipUnless(_PYSIDE6_AVAILABLE, "PySide6 not installed (optional gui extra)")
class TestFlameGraphBridge(unittest.TestCase):
    _app = None

    @classmethod
    def setUpClass(cls):
        cls._app = QGuiApplication.instance() or QGuiApplication([sys.argv[0]])

    def test_exposes_tree_title_and_total_samples(self):
        from src.gui.flamegraph_bridge import FlameGraphBridge
        tree = build_qml_tree("main;foo 10\nmain;bar 5\n")
        bridge = FlameGraphBridge(tree, "myprog — flame graph (fp)")
        self.assertEqual(bridge.title, "myprog — flame graph (fp)")
        self.assertEqual(bridge.totalSamples, 15)
        self.assertEqual(bridge.tree["name"], "all")

    def test_total_samples_zero_for_empty_tree(self):
        from src.gui.flamegraph_bridge import FlameGraphBridge
        bridge = FlameGraphBridge({"name": "all", "value": 0, "children": []}, "empty")
        self.assertEqual(bridge.totalSamples, 0)


if __name__ == "__main__":
    unittest.main()
