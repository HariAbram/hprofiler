"""
Tests for src/gui/shortcuts.py -- the one source of truth for keyboard
shortcuts (Help dialog + Main.qml's real Shortcut{} bindings). The data
table itself (SHORTCUTS/as_qml_rows/global_shortcuts) is pure Python;
ShortcutsBridge needs PySide6 (a QObject property), gated separately.
"""
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.gui.shortcuts import SHORTCUTS, ShortcutEntry, as_qml_rows, global_shortcuts

try:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QCoreApplication
    _PYSIDE6_AVAILABLE = True
except ImportError:
    _PYSIDE6_AVAILABLE = False

if _PYSIDE6_AVAILABLE:
    from src.gui.shortcuts import ShortcutsBridge


class TestShortcuts(unittest.TestCase):
    def test_every_entry_has_a_non_empty_sequence_and_label(self):
        for s in SHORTCUTS:
            self.assertTrue(s.sequence)
            self.assertTrue(s.label)

    def test_every_entry_has_a_known_scope(self):
        for s in SHORTCUTS:
            self.assertIn(s.scope, ("Global", "Timeline"))

    def test_no_duplicate_global_sequences(self):
        global_seqs = [s.sequence for s in SHORTCUTS if s.scope == "Global"]
        self.assertEqual(len(global_seqs), len(set(global_seqs)))

    def test_as_qml_rows_shape(self):
        rows = as_qml_rows()
        self.assertEqual(len(rows), len(SHORTCUTS))
        for row in rows:
            self.assertEqual(set(row.keys()), {"sequence", "label", "scope"})

    def test_global_shortcuts_filters_correctly(self):
        result = global_shortcuts()
        self.assertTrue(all(s.scope == "Global" for s in result))
        self.assertTrue(any(s.sequence == "Ctrl+K" for s in result))

    def test_shortcut_entry_is_frozen(self):
        s = ShortcutEntry("Ctrl+X", "Example", "Global")
        with self.assertRaises(Exception):
            s.sequence = "Ctrl+Y"


@unittest.skipUnless(_PYSIDE6_AVAILABLE, "PySide6 not installed (optional gui extra)")
class TestShortcutsBridge(unittest.TestCase):
    _app = None

    @classmethod
    def setUpClass(cls):
        cls._app = QCoreApplication.instance() or QCoreApplication([])

    def test_rows_matches_as_qml_rows(self):
        bridge = ShortcutsBridge()
        self.assertEqual(bridge.rows, as_qml_rows())

    def test_named_sequence_properties_match_the_table(self):
        bridge = ShortcutsBridge()
        self.assertEqual(bridge.commandPaletteSequence, "Ctrl+K")
        self.assertEqual(bridge.openProfileSequence, "Ctrl+O")
        self.assertEqual(bridge.shortcutsReferenceSequence, "F1")


if __name__ == "__main__":
    unittest.main()
