"""
No text in the Timeline's popups, bottom panel (Summary, Critical path,
Heatmap) or details panel is painted past the edge of its frame, with a
large system font in a small window as well as with the default font.

The popups' check boxes take the system font (larger than the theme's pixel
sizes on many desktops) and used to keep their one-line width, running past
the popup; summary rows elided their values. Real QML in a subprocess
(tests/gui_text_overflow_driver.py): the font must be set before the
process's one QML engine is created.
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

_DRIVER = Path(__file__).resolve().parent / "gui_text_overflow_driver.py"


def _run(font_pt: float, size: str) -> dict:
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    p = subprocess.run([sys.executable, str(_DRIVER), str(font_pt), size], env=env,
                       capture_output=True, text=True, timeout=240)
    if p.returncode != 0:
        raise AssertionError(f"driver failed:\n{p.stdout}\n{p.stderr}")
    return json.loads(p.stdout.strip().splitlines()[-1])


@unittest.skipUnless(_PYSIDE6, "PySide6 not installed (optional gui extra)")
class TestTextStaysInsideItsFrame(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.large = _run(13, "1100x760")
        cls.default = _run(0, "1280x800")

    def check(self, r):
        self.assertTrue(r["analysis"])
        self.assertGreaterEqual(r["summaryCards"], 5)     # the cards really rendered
        self.assertTrue(r["detailsPanel"])
        self.assertGreaterEqual(len(r["frames"]), 10)
        bad = {k: v for k, v in r["frames"].items() if v}
        self.assertEqual(bad, {}, f"{r['font']} at {r['size']}")
        self.assertEqual(r["warnings"], [])

    def test_large_font_small_window(self):
        self.check(self.large)

    def test_default_font(self):
        self.check(self.default)


if __name__ == "__main__":
    unittest.main()
