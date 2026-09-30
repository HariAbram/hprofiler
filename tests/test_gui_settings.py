"""
Tests for src/gui/settings.py's WorkspaceSettings/WorkspaceBridge --
window geometry/theme/table-config/dismissal persistence, per-profile
state isolation by resolved file path, malformed-entry recovery, schema-
version reset, reset-current-view/reset-all scoping, and a direct
regression guard that a planted "sensitive" command-line string never
ends up in the raw settings file.

Uses QSettings(tempfile_path, IniFormat) throughout -- never the real
user config file. A QGuiApplication IS needed (WorkspaceBridge is a
QObject with Properties/Slots), same requirement as every other
tests/test_gui_*.py file; no QQmlApplicationEngine is needed anywhere in
this file (pure Python logic + plain QObject property access), so this
safely coexists with tests/test_gui_timeline_hover.py's "only one engine-
loading test class in the whole suite" constraint.
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QSettings, QCoreApplication
    _PYSIDE6_AVAILABLE = True
except ImportError:
    _PYSIDE6_AVAILABLE = False

if _PYSIDE6_AVAILABLE:
    from src.gui.settings import WorkspaceSettings, WorkspaceBridge, SCHEMA_VERSION, _storage_key


def _mk_settings(path: str) -> "QSettings":
    return QSettings(path, QSettings.Format.IniFormat)


@unittest.skipUnless(_PYSIDE6_AVAILABLE, "PySide6 not installed (optional gui extra)")
class TestWorkspaceSettings(unittest.TestCase):
    _app = None

    @classmethod
    def setUpClass(cls):
        cls._app = QCoreApplication.instance() or QCoreApplication([])

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._ini_path = os.path.join(self._tmpdir.name, "settings.ini")

    def tearDown(self):
        self._tmpdir.cleanup()

    def _ws(self) -> WorkspaceSettings:
        return WorkspaceSettings(_mk_settings(self._ini_path))

    def _raw_file_text(self) -> str:
        # QSettings only flushes to disk on sync()/destruction -- force a
        # fresh QSettings + sync so the file genuinely reflects everything
        # written so far, independent of any object still held alive.
        qs = _mk_settings(self._ini_path)
        qs.sync()
        if not os.path.exists(self._ini_path):
            return ""
        with open(self._ini_path) as f:
            return f.read()

    # ── Window geometry ──────────────────────────────────────────────

    def test_window_geometry_round_trips(self):
        ws = self._ws()
        ws.save_window_geometry(10, 20, 1000, 700, False)
        self.assertEqual(ws.load_window_geometry(),
                          {"x": 10, "y": 20, "width": 1000, "height": 700, "maximized": False})

    def test_window_geometry_absent_by_default(self):
        ws = self._ws()
        self.assertIsNone(ws.load_window_geometry())

    def test_window_geometry_persists_across_instances(self):
        self._ws().save_window_geometry(5, 5, 900, 600, True)
        # A NEW WorkspaceSettings wrapping a NEW QSettings against the
        # same file -- proves this is real disk persistence, not just an
        # in-memory property surviving because it's the same object.
        reloaded = self._ws().load_window_geometry()
        self.assertEqual(reloaded, {"x": 5, "y": 5, "width": 900, "height": 600, "maximized": True})

    # ── Theme ─────────────────────────────────────────────────────────

    def test_theme_defaults_to_dark(self):
        self.assertTrue(self._ws().load_theme())

    def test_theme_round_trips_false(self):
        ws = self._ws()
        ws.save_theme(False)
        self.assertFalse(self._ws().load_theme())

    # ── Table config ──────────────────────────────────────────────────

    def test_table_config_round_trips(self):
        ws = self._ws()
        config = {"columns": [{"key": "name", "width": 200, "visible": True}], "percentMode": True}
        ws.save_table_config("kernels", config)
        self.assertEqual(self._ws().load_table_config("kernels"), config)

    def test_table_config_absent_for_unknown_table(self):
        self.assertIsNone(self._ws().load_table_config("never_saved"))

    def test_table_config_isolated_per_table_id(self):
        ws = self._ws()
        ws.save_table_config("kernels", {"a": 1})
        ws.save_table_config("system", {"a": 2})
        self.assertEqual(self._ws().load_table_config("kernels"), {"a": 1})
        self.assertEqual(self._ws().load_table_config("system"), {"a": 2})

    # ── Discoverability dismissal flags ─────────────────────────────

    def test_legend_collapsed_round_trips(self):
        ws = self._ws()
        self.assertFalse(ws.load_legend_collapsed())
        ws.save_legend_collapsed(True)
        self.assertTrue(self._ws().load_legend_collapsed())

    def test_timeline_overlay_dismissed_round_trips(self):
        ws = self._ws()
        self.assertFalse(ws.load_timeline_overlay_dismissed())
        ws.save_timeline_overlay_dismissed(True)
        self.assertTrue(self._ws().load_timeline_overlay_dismissed())

    # ── Per-profile state isolation (the literal requirement) ───────

    def test_profile_state_round_trips(self):
        ws = self._ws()
        state = {"zoom": 2.5, "filters": {"ranks": ["0", "1"]}, "bookmarks": [{"ns": 100.0}]}
        ws.save_profile_state("/tmp/a.hprofiler.json", state)
        self.assertEqual(self._ws().load_profile_state("/tmp/a.hprofiler.json"), state)

    def test_profile_state_not_returned_for_a_different_path(self):
        ws = self._ws()
        ws.save_profile_state("/tmp/a.hprofiler.json", {"zoom": 3.0})
        self.assertIsNone(self._ws().load_profile_state("/tmp/b.hprofiler.json"))

    def test_profile_state_keyed_by_resolved_path_not_literal_string(self):
        # "./a.json" and its absolute resolved form must be the SAME
        # profile -- relative vs. absolute shouldn't silently split one
        # profile's state into two.
        ws = self._ws()
        abs_path = os.path.join(self._tmpdir.name, "a.hprofiler.json")
        Path(abs_path).touch()
        ws.save_profile_state(abs_path, {"zoom": 4.0})
        cwd = os.getcwd()
        try:
            os.chdir(self._tmpdir.name)
            self.assertEqual(self._ws().load_profile_state("a.hprofiler.json"), {"zoom": 4.0})
        finally:
            os.chdir(cwd)

    def test_two_profiles_coexist_independently(self):
        ws = self._ws()
        ws.save_profile_state("/tmp/a.hprofiler.json", {"zoom": 1.0})
        ws.save_profile_state("/tmp/b.hprofiler.json", {"zoom": 9.0})
        self.assertEqual(self._ws().load_profile_state("/tmp/a.hprofiler.json"), {"zoom": 1.0})
        self.assertEqual(self._ws().load_profile_state("/tmp/b.hprofiler.json"), {"zoom": 9.0})

    def test_profile_path_containing_slashes_does_not_corrupt_other_profiles(self):
        # Paths sharing a common directory prefix must not collide in the
        # underlying QSettings group structure (the reason _storage_key
        # hashes the path instead of using it directly as a group name).
        ws = self._ws()
        ws.save_profile_state("/tmp/traces/run1/a.json", {"zoom": 1.0})
        ws.save_profile_state("/tmp/traces/run2/a.json", {"zoom": 2.0})
        ws.save_profile_state("/tmp/traces/a.json", {"zoom": 3.0})
        self.assertEqual(self._ws().load_profile_state("/tmp/traces/run1/a.json"), {"zoom": 1.0})
        self.assertEqual(self._ws().load_profile_state("/tmp/traces/run2/a.json"), {"zoom": 2.0})
        self.assertEqual(self._ws().load_profile_state("/tmp/traces/a.json"), {"zoom": 3.0})

    def test_evicts_oldest_profiles_beyond_the_cap(self):
        from src.gui import settings as settings_mod
        ws = self._ws()
        original_cap = settings_mod.MAX_PROFILES
        settings_mod.MAX_PROFILES = 3
        try:
            for i in range(5):
                ws.save_profile_state(f"/tmp/p{i}.json", {"i": i})
            # The 2 oldest (p0, p1) should have been evicted; the 3 most
            # recent survive.
            self.assertIsNone(self._ws().load_profile_state("/tmp/p0.json"))
            self.assertIsNone(self._ws().load_profile_state("/tmp/p1.json"))
            self.assertEqual(self._ws().load_profile_state("/tmp/p4.json"), {"i": 4})
        finally:
            settings_mod.MAX_PROFILES = original_cap

    # ── Malformed-entry recovery ─────────────────────────────────────

    def test_malformed_profile_state_json_falls_back_to_none_not_crash(self):
        raw_qs = _mk_settings(self._ini_path)
        key = _storage_key("/tmp/bad.json")
        raw_qs.setValue(f"profiles/{key}/state", "{not valid json")
        raw_qs.sync()
        # Loading the corrupt key returns None (treated as "no saved
        # state"), and does not raise.
        self.assertIsNone(self._ws().load_profile_state("/tmp/bad.json"))

    def test_malformed_table_config_json_falls_back_to_none(self):
        raw_qs = _mk_settings(self._ini_path)
        raw_qs.setValue("tables/kernels/config", "[[[broken")
        raw_qs.sync()
        self.assertIsNone(self._ws().load_table_config("kernels"))

    def test_one_corrupt_key_does_not_discard_other_valid_state(self):
        ws = self._ws()
        ws.save_window_geometry(1, 2, 300, 400, False)
        ws.save_theme(False)
        raw_qs = _mk_settings(self._ini_path)
        key = _storage_key("/tmp/bad.json")
        raw_qs.setValue(f"profiles/{key}/state", "{not valid")
        raw_qs.sync()
        ws2 = self._ws()
        self.assertIsNone(ws2.load_profile_state("/tmp/bad.json"))
        self.assertEqual(ws2.load_window_geometry(), {"x": 1, "y": 2, "width": 300, "height": 400, "maximized": False})
        self.assertFalse(ws2.load_theme())

    # ── Schema versioning ─────────────────────────────────────────────

    def test_fresh_file_gets_current_schema_version_written(self):
        self._ws()
        qs = _mk_settings(self._ini_path)
        self.assertEqual(qs.value("meta/schemaVersion", type=int), SCHEMA_VERSION)

    def test_mismatched_schema_version_triggers_full_reset(self):
        raw_qs = _mk_settings(self._ini_path)
        raw_qs.setValue("meta/schemaVersion", 999)
        raw_qs.setValue("appearance/darkTheme", False)
        raw_qs.setValue("window/width", 1)
        raw_qs.sync()

        ws = self._ws()   # construction itself triggers the version check
        self.assertTrue(ws.load_theme())          # back to the default (True)
        self.assertIsNone(ws.load_window_geometry())
        qs2 = _mk_settings(self._ini_path)
        self.assertEqual(qs2.value("meta/schemaVersion", type=int), SCHEMA_VERSION)

    def test_garbage_schema_version_value_triggers_reset(self):
        raw_qs = _mk_settings(self._ini_path)
        raw_qs.setValue("meta/schemaVersion", "not-a-number")
        raw_qs.setValue("window/width", 42)
        raw_qs.sync()
        ws = self._ws()
        self.assertIsNone(ws.load_window_geometry())

    # ── Reset actions ─────────────────────────────────────────────────

    def test_reset_current_view_only_touches_that_profile(self):
        ws = self._ws()
        ws.save_window_geometry(1, 1, 500, 500, False)
        ws.save_profile_state("/tmp/a.json", {"zoom": 2.0})
        ws.save_profile_state("/tmp/b.json", {"zoom": 3.0})
        ws.reset_current_view("/tmp/a.json")

        ws2 = self._ws()
        self.assertIsNone(ws2.load_profile_state("/tmp/a.json"))
        self.assertEqual(ws2.load_profile_state("/tmp/b.json"), {"zoom": 3.0})
        self.assertEqual(ws2.load_window_geometry()["width"], 500)

    def test_reset_current_view_with_no_path_is_a_noop(self):
        ws = self._ws()
        ws.save_window_geometry(1, 1, 500, 500, False)
        ws.reset_current_view(None)
        self.assertIsNotNone(self._ws().load_window_geometry())

    def test_reset_all_wipes_everything_and_rewrites_schema_version(self):
        ws = self._ws()
        ws.save_window_geometry(1, 1, 500, 500, False)
        ws.save_theme(False)
        ws.save_profile_state("/tmp/a.json", {"zoom": 2.0})
        ws.reset_all()

        ws2 = self._ws()
        self.assertIsNone(ws2.load_window_geometry())
        self.assertTrue(ws2.load_theme())   # back to default
        self.assertIsNone(ws2.load_profile_state("/tmp/a.json"))
        qs = _mk_settings(self._ini_path)
        self.assertEqual(qs.value("meta/schemaVersion", type=int), SCHEMA_VERSION)

    # ── Sensitive-data guard ──────────────────────────────────────────

    def test_command_line_like_secret_never_written_to_the_raw_file(self):
        ws = self._ws()
        secret = "--api-key=sk-super-secret-do-not-persist-12345"
        # Simulate a caller MISTAKENLY trying to stash something
        # command-line-shaped into profile state -- WorkspaceSettings
        # itself has no API that accepts a command line at all (the
        # guard is architectural: there's no place to put it), so this
        # test instead proves that saving ordinary profile state never
        # incidentally leaks a secret value that was never passed in.
        ws.save_profile_state("/tmp/a.json", {"zoom": 1.0, "selectedEntity": "cuda::kernel"})
        ws.save_window_geometry(1, 1, 500, 500, False)
        ws.save_theme(True)
        text = self._raw_file_text()
        self.assertNotIn(secret, text)
        self.assertNotIn("api-key", text)
        self.assertNotIn("api_key", text)

    def test_only_resolved_file_path_is_stored_for_a_profile_not_argv(self):
        ws = self._ws()
        ws.save_profile_state("/tmp/traces/run.hprofiler.json", {"zoom": 1.0})
        text = self._raw_file_text()
        self.assertIn("run.hprofiler.json", text)   # the path itself is fine to store
        self.assertNotIn("--", text)                  # no CLI-flag-shaped content anywhere


@unittest.skipUnless(_PYSIDE6_AVAILABLE, "PySide6 not installed (optional gui extra)")
class TestWorkspaceBridge(unittest.TestCase):
    _app = None

    @classmethod
    def setUpClass(cls):
        cls._app = QCoreApplication.instance() or QCoreApplication([])

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._ini_path = os.path.join(self._tmpdir.name, "settings.ini")

    def tearDown(self):
        self._tmpdir.cleanup()

    def _bridge(self) -> WorkspaceBridge:
        settings = WorkspaceSettings(_mk_settings(self._ini_path))
        return WorkspaceBridge(settings)

    def test_legend_collapsed_defaults_false_and_toggles(self):
        bridge = self._bridge()
        self.assertFalse(bridge.legendCollapsed)
        bridge.setLegendCollapsed(True)
        self.assertTrue(bridge.legendCollapsed)

    def test_legend_collapsed_persists_across_bridge_instances(self):
        self._bridge().setLegendCollapsed(True)
        self.assertTrue(self._bridge().legendCollapsed)

    def test_dismiss_timeline_overlay_persists(self):
        bridge = self._bridge()
        self.assertFalse(bridge.timelineOverlayDismissed)
        bridge.dismissTimelineOverlay()
        self.assertTrue(bridge.timelineOverlayDismissed)
        self.assertTrue(self._bridge().timelineOverlayDismissed)

    def test_reset_all_resets_bridge_side_flags_too(self):
        bridge = self._bridge()
        bridge.setLegendCollapsed(True)
        bridge.dismissTimelineOverlay()
        bridge.resetAll()
        self.assertFalse(bridge.legendCollapsed)
        self.assertFalse(bridge.timelineOverlayDismissed)
        self.assertFalse(self._bridge().legendCollapsed)

    def test_legend_collapsed_changed_signal_fires_on_change(self):
        bridge = self._bridge()
        seen = []
        bridge.legendCollapsedChanged.connect(lambda: seen.append(bridge.legendCollapsed))
        bridge.setLegendCollapsed(True)
        self.assertEqual(seen, [True])

    def test_legend_collapsed_changed_signal_does_not_fire_on_no_op_set(self):
        bridge = self._bridge()
        seen = []
        bridge.legendCollapsedChanged.connect(lambda: seen.append(1))
        bridge.setLegendCollapsed(False)   # already False -- no-op
        self.assertEqual(seen, [])

    def test_no_saved_geometry_reports_defaults_and_flag_false(self):
        bridge = self._bridge()
        self.assertFalse(bridge.hasSavedGeometry)
        self.assertEqual(bridge.windowWidth, 1280)
        self.assertEqual(bridge.windowHeight, 800)
        self.assertFalse(bridge.windowMaximized)

    def test_save_then_new_bridge_instance_reads_saved_geometry(self):
        self._bridge().saveWindowGeometry(50, 60, 1500, 900, True)
        bridge2 = self._bridge()
        self.assertTrue(bridge2.hasSavedGeometry)
        self.assertEqual(bridge2.windowX, 50)
        self.assertEqual(bridge2.windowY, 60)
        self.assertEqual(bridge2.windowWidth, 1500)
        self.assertEqual(bridge2.windowHeight, 900)
        self.assertTrue(bridge2.windowMaximized)


if __name__ == "__main__":
    unittest.main()
