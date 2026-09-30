"""
One source of truth for every keyboard shortcut this GUI defines (Phase
5 of the usability/persistence/loading overhaul) -- a plain data table,
not scattered `Keys.onPressed`/`Shortcut{}` literals with their own
one-off label strings. Consumed by:
  - ShortcutsDialog.qml (Help > Keyboard Shortcuts), which just renders
    this table -- can never drift from what the app actually binds,
    because nothing else hand-duplicates these strings.
  - Main.qml's real `Shortcut{}` items for the GLOBAL entries (sequence
    is read from here, not re-typed).
  - Tooltip hint suffixes wherever a control has a shortcut (e.g. a
    ToolTip.text like "Toggle theme (see Help > Keyboard Shortcuts)").

TimelineScreen.qml's own `Keys.onPressed` switch (Left/Right/Up/Down/
Plus/Minus/Home/End/0) is NOT re-implemented as `Shortcut{}` items here
-- it already works, is scoped to that one Item's focus, and QML
`Shortcut{}` (global, application-wide) would double-fire alongside it
if added. This table documents those entries for the reference dialog
without touching that working code path.
"""
from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QObject, Property


@dataclass(frozen=True)
class ShortcutEntry:
    sequence: str
    label: str
    scope: str   # "Global" | "Timeline"


SHORTCUTS: list[ShortcutEntry] = [
    # ── Global (real Shortcut{} bindings in Main.qml) ────────────────
    ShortcutEntry("Ctrl+K", "Open command palette", "Global"),
    ShortcutEntry("Ctrl+O", "Open a different profile", "Global"),
    ShortcutEntry("F1", "Show this keyboard shortcuts reference", "Global"),

    # ── Timeline (existing Keys.onPressed in TimelineScreen.qml) ─────
    ShortcutEntry("←  /  →", "Pan the view backward / forward", "Timeline"),
    ShortcutEntry("↑  /  ↓", "Scroll lanes up / down", "Timeline"),
    ShortcutEntry("+  /  -", "Zoom in / out (centered on the view)", "Timeline"),
    ShortcutEntry("Home  /  End", "Jump to the start / end of the trace", "Timeline"),
    ShortcutEntry("0", "Reset zoom and pan to the full trace", "Timeline"),
    ShortcutEntry("Shift + drag", "Select a time range", "Timeline"),
    ShortcutEntry("Double-click a span", "Zoom to that span", "Timeline"),
]


def as_qml_rows() -> list[dict[str, str]]:
    """Plain-dict shape for QML consumption (ShortcutsDialog.qml's
    ListView model) -- the same {sequence,label,scope} keys as
    ShortcutEntry's own fields, just JSON/QVariant-friendly."""
    return [{"sequence": s.sequence, "label": s.label, "scope": s.scope} for s in SHORTCUTS]


def global_shortcuts() -> list[ShortcutEntry]:
    return [s for s in SHORTCUTS if s.scope == "Global"]


class ShortcutsBridge(QObject):
    """Registered as the "Shortcuts" singleton -- backs
    ShortcutsDialog.qml's reference list. `rows` is `constant=True`: the
    shortcut table is fixed at build time, never changes at runtime."""

    @Property('QVariantList', constant=True)
    def rows(self) -> list[dict[str, str]]:
        return as_qml_rows()

    # Individually-named properties (not a label-keyed map -- awkward to
    # index from QML with spaces in the label text) for the 3 real
    # Shortcut{} bindings Main.qml needs -- all three still pull from
    # the SAME SHORTCUTS table above, so the reference dialog and the
    # real binding can never disagree about what key opens what.
    @Property(str, constant=True)
    def commandPaletteSequence(self) -> str:
        return next(s.sequence for s in SHORTCUTS if s.label == "Open command palette")

    @Property(str, constant=True)
    def openProfileSequence(self) -> str:
        return next(s.sequence for s in SHORTCUTS if s.label == "Open a different profile")

    @Property(str, constant=True)
    def shortcutsReferenceSequence(self) -> str:
        return next(s.sequence for s in SHORTCUTS if s.label == "Show this keyboard shortcuts reference")
