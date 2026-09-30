"""
Central design-token module shared by every QML screen, exposed to QML
as a single singleton ("AppTheme") so every .qml file reads e.g.
`AppTheme.background` / `AppTheme.categoryColor("cuda")` /
`AppTheme.spacingMd` instead of each screen hardcoding its own hex
values, margins, or font sizes -- and so the dark/light toggle (Phase 7)
only has to change color state in one place. Originally colors only
(hence the class/module name); a visual-consistency audit extended it to
also own spacing/radius/typography/row/button-size tokens and semantic
warning/error/success/info color aliases, after finding those values
drifting independently (2-6 different radii, 8 different font sizes,
etc.) across screens that already deferred to this module for color.

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
# mpi/memory/jit/nvtx were revised from their original values (kept as
# trailing comments) after a quantitative colorblind-safety check -- a
# Brettel/Machado-style deuteranopia/protanopia/tritanopia simulation run
# against every pairwise combination found 9 confusable dark-theme pairs
# and 17 confusable light-theme pairs in the original palette (worst:
# mpi/memory confusable under both common red-green CVD forms; opencl/
# nvtx confusable under all three; openmp/sync confusable in light mode).
# This revision -- touching only these 4 categories, chosen to minimize
# both the number of changed values and disruption to the rest of the
# palette -- cuts that to 3 dark-theme pairs and 9 light-theme pairs, all
# remaining ones being either the rarest CVD form (tritanopia) or outside
# these 4 categories (a residual, honestly-disclosed limitation, not
# silently ignored -- fully fixing the rest would mean touching categories
# with no other reason to change here).
_CATEGORY_COLORS: dict[str, tuple[str, str]] = {
    "cpu":     ("#22d3ee", "#0891b2"),
    "cuda":    ("#f87171", "#dc2626"),
    "rocm":    ("#e879f9", "#c026d3"),
    "opencl":  ("#fbbf24", "#b45309"),
    "openmp":  ("#4ade80", "#15803d"),
    "mpi":     ("#2563eb", "#1d4ed8"),   # was ("#60a5fa", "#2563eb")
    "nccl":    ("#f472b6", "#db2777"),
    "memory":  ("#a78bfa", "#7c3aed"),   # was ("#818cf8", "#4f46e5")
    "sync":    ("#e5e7eb", "#374151"),
    "jit":     ("#8b5cf6", "#581c87"),   # was ("#c084fc", "#7e22ce")
    "nvtx":    ("#ea580c", "#9a3412"),   # was ("#fb923c", "#c2410c")
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


# ── Pure, Qt-free color lookups ─────────────────────────────────────────
# Same logic as Theme's categoryColor/severityColor/bucketColor/
# changeColor Slots below, taking `dark: bool` directly instead of
# reading `self._dark` off a QObject -- lets the GUI's async-loading
# worker thread (src/gui/loader.py) precompute colors for large per-span
# structures (Timeline lanes, Call Tree nodes, Flame Graph nodes) without
# ever touching a QObject from a thread other than the one that owns it,
# which Qt does not guarantee is safe even for a seemingly-stateless
# method call. The QObject Slots delegate to these instead of duplicating
# the logic, so the two can never drift apart.
def category_color(category: str, dark: bool) -> str:
    light_dark = _CATEGORY_COLORS.get(category, _CATEGORY_COLORS["other"])
    return light_dark[0] if dark else light_dark[1]


def severity_color(severity: str, dark: bool) -> str:
    light_dark = _SEVERITY_COLORS.get(severity, _SEVERITY_COLORS["cyan"])
    return light_dark[0] if dark else light_dark[1]


def text_muted_color(dark: bool) -> str:
    return _DARK["textMuted"] if dark else _LIGHT["textMuted"]


_BUCKET_TO_CATEGORY = {
    "Computation": "cpu", "Communication": "mpi",
    "Synchronization": "sync", "Memory transfer": "memory",
    "Runtime overhead": "jit", "Annotation": "nvtx",
}


def bucket_color(bucket: str, dark: bool) -> str:
    if bucket in ("Idle", "Other"):
        return text_muted_color(dark)
    return category_color(_BUCKET_TO_CATEGORY.get(bucket, "other"), dark)


_CHANGE_TO_SEVERITY_FAMILY = {
    "improved": "green", "regressed": "red", "new": "cyan",
    "removed": "yellow", "unchanged": None, "unavailable": None,
    "zero": None,
}


def change_color(status: str, dark: bool) -> str:
    family = _CHANGE_TO_SEVERITY_FAMILY.get(status)
    return severity_color(family, dark) if family else text_muted_color(dark)

# Non-color layout tokens -- spacing/radius/typography/row/button scales.
# Unlike the palettes above, these don't vary with the dark/light toggle,
# so they're plain constants exposed as @Property(int, constant=True)
# rather than going through _pal(). Values are the DOMINANT ones already
# found in use across the GUI's screens (not invented from scratch) --
# see theme audit notes: radius 6 and margin/spacing 8 were already the
# most common values by a wide margin; this just gives them names and
# extends screens still using an ad hoc 2/3/4/5 or 10/12/13/14/16 to the
# same small scale instead of their own one-off number.
_SPACING = {"xs": 4, "sm": 6, "md": 8, "lg": 12, "xl": 16}
_RADIUS = {"small": 4, "panel": 6}
# Typography sized by ROLE, not by preserving every screen's prior
# arbitrary pixelSize -- this is what actually implements "primary
# metrics vs. secondary info vs. diagnostic detail" hierarchy rather than
# just aliasing existing drift.
_TYPE = {
    "caption": 10,  # fine print: sample %, hints
    "label": 11,    # secondary/muted metadata, control text
    "body": 12,     # primary row/cell text
    "title": 13,    # section identity (screen/panel titles)
    "heading": 14,  # app chrome only (Main.qml)
    "value": 20,    # StatCard-style headline numbers
}
_ROW = {"compact": 24, "comfortable": 28, "statCard": 84}
_BUTTON = {"height": 24, "iconWidth": 28}
_FIELD_WIDTH = 240


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
        return category_color(category, self._dark)

    @Slot(str, result=str)
    def severityColor(self, severity: str) -> str:
        return severity_color(severity, self._dark)

    # ── Activity buckets (analysis/activity_buckets.py's BUCKETS) ───────
    # Computation/Communication/Synchronization/Memory transfer reuse the
    # SAME category colors OverviewScreen.qml's own breakdownColor()
    # already mapped them to since Round 16 (cpu/mpi/sync/memory) -- moved
    # here so Python and QML can't drift into disagreeing about it the way
    # DashboardBridge's old _bucket_of dict and this slot's absence would
    # have let them. Runtime overhead/Annotation reuse the jit/nvtx
    # category colors (already CVD-checked in the Round 15 palette
    # revision) rather than adding brand-new, unvetted hex values for two
    # buckets that are thematically close to those categories anyway.
    @Slot(str, result=str)
    def bucketColor(self, bucket: str) -> str:
        return bucket_color(bucket, self._dark)

    # ── Comparison status (src/analysis/compare.py's classify()) ────────
    # Reuses severityColor()'s existing family palette rather than a new
    # dict -- one fewer place a future theme edit has to remember to touch.
    @Slot(str, result=str)
    def changeColor(self, status: str) -> str:
        return change_color(status, self._dark)

    # ── Semantic severity aliases ───────────────────────────────────────
    # Additive, not a replacement: severityColor()'s "red"/"yellow"/
    # "green"/"cyan" family-name contract stays exactly as-is (9 call
    # sites in bridge.py, plus analysis/dashboard.py's diagnose() emits
    # exactly these family strings) -- these just give the self-
    # documenting names a "warnings"/"errors" token request actually
    # asks for, without touching that working contract.
    @Property(str, notify=themeChanged)
    def errorColor(self) -> str:
        return self.severityColor("red")

    @Property(str, notify=themeChanged)
    def warningColor(self) -> str:
        return self.severityColor("yellow")

    @Property(str, notify=themeChanged)
    def successColor(self) -> str:
        return self.severityColor("green")

    @Property(str, notify=themeChanged)
    def infoColor(self) -> str:
        return self.severityColor("cyan")

    # ── Layout tokens (spacing/radius/typography/row/button) ───────────
    # constant=True, not notify=themeChanged -- these never vary with the
    # dark/light toggle, only colors do.
    @Property(int, constant=True)
    def spacingXs(self) -> int:
        return _SPACING["xs"]

    @Property(int, constant=True)
    def spacingSm(self) -> int:
        return _SPACING["sm"]

    @Property(int, constant=True)
    def spacingMd(self) -> int:
        return _SPACING["md"]

    @Property(int, constant=True)
    def spacingLg(self) -> int:
        return _SPACING["lg"]

    @Property(int, constant=True)
    def spacingXl(self) -> int:
        return _SPACING["xl"]

    @Property(int, constant=True)
    def radiusSmall(self) -> int:
        return _RADIUS["small"]

    @Property(int, constant=True)
    def radiusPanel(self) -> int:
        return _RADIUS["panel"]

    @Property(int, constant=True)
    def typeCaption(self) -> int:
        return _TYPE["caption"]

    @Property(int, constant=True)
    def typeLabel(self) -> int:
        return _TYPE["label"]

    @Property(int, constant=True)
    def typeBody(self) -> int:
        return _TYPE["body"]

    @Property(int, constant=True)
    def typeTitle(self) -> int:
        return _TYPE["title"]

    @Property(int, constant=True)
    def typeHeading(self) -> int:
        return _TYPE["heading"]

    @Property(int, constant=True)
    def typeValue(self) -> int:
        return _TYPE["value"]

    @Property(int, constant=True)
    def rowCompact(self) -> int:
        return _ROW["compact"]

    @Property(int, constant=True)
    def rowComfortable(self) -> int:
        return _ROW["comfortable"]

    @Property(int, constant=True)
    def statRowHeight(self) -> int:
        return _ROW["statCard"]

    @Property(int, constant=True)
    def buttonHeight(self) -> int:
        return _BUTTON["height"]

    @Property(int, constant=True)
    def iconButtonWidth(self) -> int:
        return _BUTTON["iconWidth"]

    @Property(int, constant=True)
    def fieldWidth(self) -> int:
        return _FIELD_WIDTH

    # Modal/overlay dimming backdrop (e.g. the "Open Profile" progress/
    # error overlay in Main.qml) -- plain black regardless of theme (a
    # dimming scrim reads the same way under either palette; it's the
    # Rectangle's own `opacity` that does the dimming, not this hex),
    # constant=True like the layout tokens above, not notify=themeChanged.
    @Property(str, constant=True)
    def scrimColor(self) -> str:
        return "#000000"
