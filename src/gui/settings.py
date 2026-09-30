"""
Workspace persistence (Phase 1 of the usability/persistence/loading
overhaul) -- QSettings(IniFormat)-backed, versioned, with GLOBAL settings
(profile-independent: window geometry, theme, per-table column config,
legend/first-use-overlay dismissal) kept separate from PER-PROFILE
settings (filters, timeline zoom/pan/grouping, selected entity,
bookmarks -- keyed by the trace file's own resolved absolute path).

Resolved via Qt's own org/app-name mechanism (app.py already calls
app.setApplicationName("hprofiler")/setOrganizationName("hprofiler")) to
~/.config/hprofiler/hprofiler.conf on Linux -- IniFormat, not
NativeFormat (registry/plist), deliberately: a plain text file is
human-diffable/greppable, which matters for "recover gracefully from
invalid settings" (a corrupt file is trivially inspectable) and for the
"copy diagnostics"/support story generally.

Deliberately does NOT persist AppInfo.commandLine/argv/env -- only the
trace file PATH is ever written. The profiled program's own command line
(already shown live in the window title/footer) can legitimately contain
secrets (API keys, credentials passed as args); nothing about restoring
"which profile was open" requires remembering how it was originally
launched, so that data never touches disk here.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QSettings, Property, Signal, Slot

SCHEMA_VERSION = 1

# Without a cap, a user who opens many different trace files over months
# would grow the settings file unboundedly -- keep only the most
# recently-opened profiles' per-profile state.
MAX_PROFILES = 50


def _resolved_path(trace_path: str) -> str:
    return str(Path(trace_path).resolve())


def _storage_key(trace_path: str) -> str:
    """A stable, QSettings-group-safe key for `trace_path` -- the raw
    resolved path can't be used directly as a group/key segment (it
    contains '/', which QSettings treats as ITS OWN group-nesting
    separator, so two different paths sharing a prefix would silently
    create overlapping/ambiguous groups). Hashing sidesteps that
    entirely; the raw path is still stored as a VALUE alongside the
    state (not as part of the key) purely for human-readability of the
    resulting .ini file and as a collision sanity-check, not because
    lookup depends on it -- lookup re-hashes the incoming path the same
    way every time, so it's always correct regardless."""
    return hashlib.sha256(_resolved_path(trace_path).encode("utf-8")).hexdigest()[:16]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class WorkspaceSettings:
    """Plain Python wrapper around QSettings -- no QObject/Qt signal
    machinery needed for the read/write logic itself, matching this
    codebase's established preference for plain, testable Python (see
    src/gui/tablemodel.py's TableConfig for the in-memory-only precedent
    this extends to disk). WorkspaceBridge below is the thin QObject
    Slot-bearing wrapper QML actually talks to."""

    def __init__(self, qsettings: QSettings | None = None):
        self._s = qsettings if qsettings is not None else QSettings()
        self._check_schema_version()

    def _check_schema_version(self) -> None:
        stored = self._s.value("meta/schemaVersion", None)
        if stored is None:
            # Fresh install/first run -- nothing to migrate, no reset needed.
            self._s.setValue("meta/schemaVersion", SCHEMA_VERSION)
            return
        try:
            stored_version = int(stored)
        except (TypeError, ValueError):
            stored_version = None
        if stored_version != SCHEMA_VERSION:
            # No migration table exists yet (this IS schema v1) -- any
            # mismatch, older or newer than this build knows about,
            # means "don't trust this file," not "guess how to upgrade
            # it." A future v2 would add `if stored_version == 1: ...`
            # upgrade steps here instead of an unconditional wipe.
            self._s.clear()
            self._s.setValue("meta/schemaVersion", SCHEMA_VERSION)

    # ── Global: window geometry ─────────────────────────────────────
    def save_window_geometry(self, x: int, y: int, w: int, h: int, maximized: bool) -> None:
        self._s.beginGroup("window")
        self._s.setValue("x", x)
        self._s.setValue("y", y)
        self._s.setValue("width", w)
        self._s.setValue("height", h)
        self._s.setValue("maximized", maximized)
        self._s.endGroup()

    def load_window_geometry(self) -> dict[str, Any] | None:
        self._s.beginGroup("window")
        try:
            if not self._s.contains("width"):
                return None
            return {
                "x": self._s.value("x", 0, type=int),
                "y": self._s.value("y", 0, type=int),
                "width": self._s.value("width", 1280, type=int),
                "height": self._s.value("height", 800, type=int),
                "maximized": self._s.value("maximized", False, type=bool),
            }
        except (TypeError, ValueError):
            return None
        finally:
            self._s.endGroup()

    # ── Global: theme ────────────────────────────────────────────────
    def save_theme(self, dark: bool) -> None:
        self._s.setValue("appearance/darkTheme", dark)

    def load_theme(self) -> bool:
        return bool(self._s.value("appearance/darkTheme", True, type=bool))

    # ── Global: per-table column config ─────────────────────────────
    def save_table_config(self, table_id: str, config: Any) -> None:
        self._s.setValue(f"tables/{table_id}/config", json.dumps(config))

    def load_table_config(self, table_id: str) -> Any | None:
        raw = self._s.value(f"tables/{table_id}/config", None)
        if not raw:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None

    # ── Global: discoverability dismissal / collapse flags ──────────
    def save_legend_collapsed(self, collapsed: bool) -> None:
        self._s.setValue("help/legendCollapsed", collapsed)

    def load_legend_collapsed(self) -> bool:
        return bool(self._s.value("help/legendCollapsed", False, type=bool))

    def save_timeline_overlay_dismissed(self, dismissed: bool) -> None:
        self._s.setValue("help/timelineOverlayDismissed", dismissed)

    def load_timeline_overlay_dismissed(self) -> bool:
        return bool(self._s.value("help/timelineOverlayDismissed", False, type=bool))

    # ── Per-profile state ────────────────────────────────────────────
    def save_profile_state(self, trace_path: str, state: dict[str, Any]) -> None:
        key = _storage_key(trace_path)
        self._s.beginGroup("profiles")
        self._s.beginGroup(key)
        self._s.setValue("path", _resolved_path(trace_path))
        self._s.setValue("state", json.dumps(state))
        self._s.setValue("lastOpenedIso", _now_iso())
        self._s.endGroup()
        self._s.endGroup()
        self._evict_old_profiles()

    def load_profile_state(self, trace_path: str) -> dict[str, Any] | None:
        key = _storage_key(trace_path)
        raw = self._s.value(f"profiles/{key}/state", None)
        if not raw:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            # One corrupted profile entry degrades to "no saved state for
            # this profile" (same as never having opened it before) --
            # never raises, never discards anything else in the file.
            return None

    def _evict_old_profiles(self) -> None:
        self._s.beginGroup("profiles")
        try:
            groups = self._s.childGroups()
            if len(groups) <= MAX_PROFILES:
                return
            dated = sorted(groups, key=lambda g: self._s.value(f"{g}/lastOpenedIso", ""))
            for stale in dated[: len(groups) - MAX_PROFILES]:
                self._s.remove(stale)
        finally:
            self._s.endGroup()

    # ── Lifecycle: reset actions ─────────────────────────────────────
    def reset_current_view(self, trace_path: str | None) -> None:
        """Only touches the given profile's own saved state -- window
        geometry, theme, table config, and every OTHER profile's state
        are untouched."""
        if not trace_path:
            return
        key = _storage_key(trace_path)
        self._s.remove(f"profiles/{key}")

    def reset_all(self) -> None:
        """Wipes every setting this app has ever saved -- window
        geometry, theme, table config, all profiles' state, dismissal
        flags. Destructive; callers should confirm with the user first
        (this method itself does not prompt)."""
        self._s.clear()
        self._s.setValue("meta/schemaVersion", SCHEMA_VERSION)

    def sync(self) -> None:
        self._s.sync()


class WorkspaceBridge(QObject):
    """Thin QObject wrapper exposing the subset of WorkspaceSettings QML
    actually needs to call directly (dismissal flags, reset actions,
    theme read/write) -- registered as the "Workspace" singleton.
    Structured per-profile/per-table state (filters, bookmarks, column
    config) is saved/restored by Python-side controller code at load/
    save time, not called piecemeal from QML."""

    themeChanged = Signal()
    legendCollapsedChanged = Signal()
    timelineOverlayDismissedChanged = Signal()

    def __init__(self, settings: WorkspaceSettings, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._settings = settings
        self._legend_collapsed = settings.load_legend_collapsed()
        self._timeline_overlay_dismissed = settings.load_timeline_overlay_dismissed()
        geo = settings.load_window_geometry()
        self._geometry = geo or {"x": 0, "y": 0, "width": 1280, "height": 800, "maximized": False}
        self._has_saved_geometry = geo is not None

    # ── Window geometry ──────────────────────────────────────────────
    # Read once at construction (constant=True -- Main.qml reads these
    # ONLY at startup to seed its own x/y/width/height/visibility;
    # nothing needs live change notification here since saves flow the
    # other direction, QML -> saveWindowGeometry -> disk, never back
    # into these properties during the same run).
    @Property(bool, constant=True)
    def hasSavedGeometry(self) -> bool:
        return self._has_saved_geometry

    @Property(int, constant=True)
    def windowX(self) -> int:
        return self._geometry["x"]

    @Property(int, constant=True)
    def windowY(self) -> int:
        return self._geometry["y"]

    @Property(int, constant=True)
    def windowWidth(self) -> int:
        return self._geometry["width"]

    @Property(int, constant=True)
    def windowHeight(self) -> int:
        return self._geometry["height"]

    @Property(bool, constant=True)
    def windowMaximized(self) -> bool:
        return self._geometry["maximized"]

    @Slot(int, int, int, int, bool)
    def saveWindowGeometry(self, x: int, y: int, w: int, h: int, maximized: bool) -> None:
        self._settings.save_window_geometry(x, y, w, h, maximized)

    @Property(bool, notify=legendCollapsedChanged)
    def legendCollapsed(self) -> bool:
        return self._legend_collapsed

    @Slot(bool)
    def setLegendCollapsed(self, collapsed: bool) -> None:
        if collapsed == self._legend_collapsed:
            return
        self._legend_collapsed = collapsed
        self._settings.save_legend_collapsed(collapsed)
        self.legendCollapsedChanged.emit()

    @Property(bool, notify=timelineOverlayDismissedChanged)
    def timelineOverlayDismissed(self) -> bool:
        return self._timeline_overlay_dismissed

    @Slot()
    def dismissTimelineOverlay(self) -> None:
        if self._timeline_overlay_dismissed:
            return
        self._timeline_overlay_dismissed = True
        self._settings.save_timeline_overlay_dismissed(True)
        self.timelineOverlayDismissedChanged.emit()

    @Slot(str)
    def resetCurrentView(self, trace_path: str) -> None:
        self._settings.reset_current_view(trace_path)

    @Slot()
    def resetAll(self) -> None:
        self._settings.reset_all()
        self._legend_collapsed = False
        self._timeline_overlay_dismissed = False
        self.legendCollapsedChanged.emit()
        self.timelineOverlayDismissedChanged.emit()
