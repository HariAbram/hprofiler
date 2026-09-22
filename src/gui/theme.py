"""
Color palette shared by every QML screen, exposed to QML as a single
context property ("AppTheme") so every .qml file reads e.g.
`AppTheme.background` / `AppTheme.categoryColor("cuda")` instead of each
screen hardcoding its own hex values -- and so the dark/light toggle
(Phase 7) only has to change state in one place.

Category hues are chosen independently from the TUI's Rich color names
(src/ui/app.py's _CAT_RICH) since Rich color names ("dodger_blue2") and
QML/CSS hex colors are different systems -- but chosen to be the same
semantic hue per category, so a user moving between the TUI and this GUI
sees the same category mean the same color in both.
"""
from __future__ import annotations

from PySide6.QtCore import QObject, Property, Signal, Slot

# category -> (dark-theme hex, light-theme hex). Most categories use the
# same saturated hex in both themes (legible on either background); only
# ones that were originally very close to white/black needed a per-theme
# adjustment for contrast.
_CATEGORY_COLORS: dict[str, tuple[str, str]] = {
    "cpu":     ("#22d3ee", "#0891b2"),
    "cuda":    ("#f87171", "#dc2626"),
    "rocm":    ("#e879f9", "#c026d3"),
    "opencl":  ("#fbbf24", "#b45309"),
    "openmp":  ("#4ade80", "#15803d"),
    "mpi":     ("#60a5fa", "#2563eb"),
    "nccl":    ("#f472b6", "#db2777"),
    "memory":  ("#818cf8", "#4f46e5"),
    "sync":    ("#e5e7eb", "#374151"),
    "jit":     ("#c084fc", "#7e22ce"),
    "nvtx":    ("#fb923c", "#c2410c"),
    "other":   ("#9ca3af", "#6b7280"),
}

# severity family (from analysis/dashboard.py's diagnose()/top_findings())
# -> (dark hex, light hex).
_SEVERITY_COLORS: dict[str, tuple[str, str]] = {
    "red":    ("#f87171", "#dc2626"),
    "yellow": ("#fbbf24", "#b45309"),
    "green":  ("#4ade80", "#15803d"),
    "cyan":   ("#22d3ee", "#0891b2"),
}

_DARK = {
    "background":   "#0d1117",
    "surface":      "#11161d",
    "panelBorder":  "#30363d",
    "panelBorderFocus": "#22d3ee",
    "text":         "#e6edf3",
    "textMuted":    "#8b949e",
    "accent":       "#22d3ee",
}

_LIGHT = {
    "background":   "#ffffff",
    "surface":      "#f6f8fa",
    "panelBorder":  "#d0d7de",
    "panelBorderFocus": "#0969da",
    "text":         "#1f2328",
    "textMuted":    "#656d76",
    "accent":       "#0969da",
}


class Theme(QObject):
    """Registered as the "AppTheme" context property. `dark` toggles the
    whole palette; every plain-string property recomputes from `dark`
    when it changes (QML properties bound to these re-evaluate
    automatically via themeChanged)."""

    themeChanged = Signal()

    def __init__(self, dark: bool = True, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._dark = dark

    def _pal(self) -> dict[str, str]:
        return _DARK if self._dark else _LIGHT

    @Property(bool, notify=themeChanged)
    def dark(self) -> bool:
        return self._dark

    @dark.setter
    def dark(self, value: bool) -> None:
        if value != self._dark:
            self._dark = value
            self.themeChanged.emit()

    @Slot()
    def toggle(self) -> None:
        self.dark = not self._dark

    @Property(str, notify=themeChanged)
    def background(self) -> str:
        return self._pal()["background"]

    @Property(str, notify=themeChanged)
    def surface(self) -> str:
        return self._pal()["surface"]

    @Property(str, notify=themeChanged)
    def panelBorder(self) -> str:
        return self._pal()["panelBorder"]

    @Property(str, notify=themeChanged)
    def panelBorderFocus(self) -> str:
        return self._pal()["panelBorderFocus"]

    @Property(str, notify=themeChanged)
    def text(self) -> str:
        return self._pal()["text"]

    @Property(str, notify=themeChanged)
    def textMuted(self) -> str:
        return self._pal()["textMuted"]

    @Property(str, notify=themeChanged)
    def accent(self) -> str:
        return self._pal()["accent"]

    @Slot(str, result=str)
    def categoryColor(self, category: str) -> str:
        light_dark = _CATEGORY_COLORS.get(category, _CATEGORY_COLORS["other"])
        return light_dark[0] if self._dark else light_dark[1]

    @Slot(str, result=str)
    def severityColor(self, severity: str) -> str:
        light_dark = _SEVERITY_COLORS.get(severity, _SEVERITY_COLORS["cyan"])
        return light_dark[0] if self._dark else light_dark[1]
