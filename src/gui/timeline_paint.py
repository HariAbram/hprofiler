"""
QPainter drawing for the Timeline: lanes, aggregate (header) rows, the
dependency / critical-path overlay, the minimap, the span-level score
track and exported figures -- ONE implementation, used by the on-screen
items (timeline_items.py), by figure export, and by the pixel tests (which
paint into a QImage without any QML engine).

Encodings (see the design notes in docs/VISUALIZATION_DESIGN.md):

  exact interval        filled rectangle, start..end, fractional edges kept
  sub-pixel events      per column: a bar whose HEIGHT is the busy fraction
                        of the column, plus a presence tick at the top when
                        the bar is too low to see -- never a full-height bar
  explicit waiting      hatched neutral fill + a thin stripe in the wait's
                        own colour (same wall time as a sync-lane span)
  inferred waiting      dotted veil over the part of a gated call before its
                        last dependency arrived (from the dependency graph)
  no data               cross-hatch: outside the lane's first..last event
  gap                   nothing drawn
  critical path         a band at the top of the lane: solid while the path
                        is busy in the lane, hatched while it waits
  dependency edge       line + arrowhead at the gated event, a family-shaped
                        marker at the source; colour = family, width and
                        opacity = tier, solid = direct, dashed = inferred,
                        dotted = low tier
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QImage, QPainter, QPen, QPolygonF

from . import theme as theme_mod
from . import timeline_lod as lod

MARGIN = 3            # top/bottom inset of a lane's bars, px
BAND = 3              # critical-path band height, px
TICK = 3              # presence-tick height, px

TIER_WIDTH = {"certain": 2.4, "high": 2.0, "medium": 1.6, "low": 1.2}
TIER_ALPHA = {"certain": 1.0, "high": 0.9, "medium": 0.75, "low": 0.6}


@dataclass
class Style:
    dark: bool
    background: QColor
    surface: QColor
    border: QColor
    text: QColor
    muted: QColor
    accent: QColor
    critical: QColor
    overlap: QColor
    idle: QColor

    @classmethod
    def for_theme(cls, dark: bool) -> "Style":
        pal = theme_mod._DARK if dark else theme_mod._LIGHT
        return cls(dark, QColor(pal["background"]), QColor(pal["surface"]), QColor(pal["panelBorder"]),
                   QColor(pal["text"]), QColor(pal["textMuted"]), QColor(pal["accent"]),
                   QColor(theme_mod.critical_color(dark)), QColor(theme_mod.overlap_color(dark)),
                   QColor(theme_mod.activity_color("idle", dark)))


_RGB_CACHE: dict[str, tuple[int, int, int]] = {}


def _rgb(hex_or_name: str) -> tuple[int, int, int]:
    v = _RGB_CACHE.get(hex_or_name)
    if v is None:
        c = QColor(hex_or_name)
        v = _RGB_CACHE[hex_or_name] = (c.red(), c.green(), c.blue())
    return v


def _with_alpha(c: QColor, a: float) -> QColor:
    out = QColor(c)
    out.setAlphaF(max(0.0, min(1.0, a)))
    return out


_TINT_CACHE: dict[tuple, QColor] = {}


def _tinted(color: str, a: float) -> QColor:
    """A shared QColor for a colour string at alpha `a` (callers only paint
    with it, never mutate it) -- a dense lane has thousands of rectangles
    in a handful of colours."""
    key = (color, a)
    c = _TINT_CACHE.get(key)
    if c is None:
        if len(_TINT_CACHE) > 4096:
            _TINT_CACHE.clear()
        c = _TINT_CACHE[key] = _with_alpha(QColor(color), a)
    return c


def tier_pen(color: QColor | str, tier: str, cls: str, *, scale: float = 1.0) -> QPen:
    c = QColor(color) if isinstance(color, str) else QColor(color)
    c.setAlphaF(TIER_ALPHA.get(tier, 0.6))
    pen = QPen(c, TIER_WIDTH.get(tier, 1.2) * scale)
    if cls == "inferred":
        pen.setStyle(Qt.PenStyle.DotLine if tier == "low" else Qt.PenStyle.DashLine)
    pen.setCapStyle(Qt.PenCapStyle.FlatCap)
    return pen


# ── density layer (numpy -> QImage: one draw call per lane) ───────────────

def density_image(busy: np.ndarray, wait: np.ndarray, count: np.ndarray, colors: list, h: int,
                  style: Style, *, fallback: str = "", alpha: float = 0.9, ticks: bool = True,
                  dim: np.ndarray | None = None) -> QImage | None:
    w = len(busy)
    if w == 0 or h <= 0:
        return None
    has = (busy > 0) | (wait > 0) | (count > 0)
    if not has.any():
        return None
    inner = max(1, h - 2 * MARGIN)
    bh = np.where(busy > 0, np.maximum(busy * inner, 1.0), 0.0)
    wh = np.where(wait > 0, np.maximum(wait * inner, 1.0), 0.0)
    wh = np.minimum(wh, inner - np.minimum(bh, inner))
    img = np.zeros((h, w, 4), dtype=np.uint8)
    ys = np.arange(h, dtype=np.float64)[:, None]
    bottom = h - MARGIN
    busy_mask = (ys >= bottom - bh[None, :]) & (ys < bottom)
    wait_mask = (ys >= bottom - bh[None, :] - wh[None, :]) & (ys < bottom - bh[None, :])
    xs = np.arange(w)[None, :]
    hatch = ((xs + ys.astype(np.int64)) % 4) == 0
    # per-column colour of the busy part
    rgb = np.zeros((w, 3), dtype=np.uint8)
    fb = _rgb(fallback) if fallback else (style.accent.red(), style.accent.green(), style.accent.blue())
    for c in np.nonzero(busy > 0)[0].tolist():
        rgb[c] = _rgb(colors[c]) if colors[c] else fb
    a_busy = int(alpha * 255)
    col_alpha = np.full(w, a_busy, dtype=np.uint8)
    if dim is not None:
        col_alpha = np.where(dim, int(0.22 * 255), a_busy).astype(np.uint8)
    img[..., :3] = np.where(busy_mask[..., None], rgb[None, :, :], img[..., :3])
    img[..., 3] = np.where(busy_mask, col_alpha[None, :], img[..., 3])
    m = style.muted
    wmask = wait_mask & hatch
    img[..., 0] = np.where(wmask, m.red(), img[..., 0])
    img[..., 1] = np.where(wmask, m.green(), img[..., 1])
    img[..., 2] = np.where(wmask, m.blue(), img[..., 2])
    img[..., 3] = np.where(wmask, 220, img[..., 3])
    wfill = wait_mask & ~hatch
    img[..., 0] = np.where(wfill, style.idle.red(), img[..., 0])
    img[..., 1] = np.where(wfill, style.idle.green(), img[..., 1])
    img[..., 2] = np.where(wfill, style.idle.blue(), img[..., 2])
    img[..., 3] = np.where(wfill, 70, img[..., 3])
    if ticks:
        # presence: events in the column, too short for a visible bar
        tick = (count > 0) & (bh < 2) & (wh < 2)
        if tick.any():
            tmask = (ys < TICK) & tick[None, :]
            t = style.text
            img[..., 0] = np.where(tmask, t.red(), img[..., 0])
            img[..., 1] = np.where(tmask, t.green(), img[..., 1])
            img[..., 2] = np.where(tmask, t.blue(), img[..., 2])
            img[..., 3] = np.where(tmask, 190, img[..., 3])
    buf = np.ascontiguousarray(img)
    q = QImage(buf.data, w, h, 4 * w, QImage.Format.Format_RGBA8888)
    return q.copy()        # detach from the numpy buffer


def _hatch_brush(color: QColor, pattern=Qt.BrushStyle.BDiagPattern) -> QBrush:
    b = QBrush(color, pattern)
    return b


@dataclass
class LaneOptions:
    selected_eid: int = -1
    matched: Callable[[int], bool] | None = None
    dim: Callable[[lod.Rect], bool] | None = None           # highlight: True = not highlighted
    cp_steps: list[tuple[float, float, str, str]] = None    # (x0, x1, mode, tier)
    inferred_waits: list[tuple[float, float]] = None
    path_eids: set | None = None
    show_unobserved: bool = True


def paint_lane(p: QPainter, lr: lod.LaneRender, w: float, h: float, style: Style,
               opts: LaneOptions | None = None) -> None:
    opts = opts or LaneOptions()
    p.save()
    p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
    top, inner = MARGIN, max(1.0, h - 2 * MARGIN)
    if opts.show_unobserved and lr.observed is not None:
        lo, hi = lr.observed
        brush = _hatch_brush(_with_alpha(style.muted, 0.35), Qt.BrushStyle.DiagCrossPattern)
        if lo > 0.5:
            p.fillRect(QRectF(0, top, min(lo, w), inner), brush)
        if hi < w - 0.5:
            p.fillRect(QRectF(max(hi, 0), top, w - max(hi, 0), inner), brush)
    dim_cols = None
    img = density_image(lr.busy, lr.wait, lr.count, lr.col_color, int(round(h)), style,
                        dim=dim_cols)
    if img is not None:
        p.drawImage(QRectF(0, 0, lr.width, int(round(h))), img)
    sel_rect = None
    wait_paint: dict[float, tuple] = {}
    for r in lr.rects:
        x0, x1 = r.x0, max(r.x1, r.x0 + lod.MIN_RECT_PX)
        rect = QRectF(x0, top, x1 - x0, inner)
        faded = opts.dim is not None and opts.dim(r)
        a = 0.22 if faded else 1.0
        if r.activity in lod.WAIT_LIKE:
            wp = wait_paint.get(a)
            if wp is None:
                wp = wait_paint[a] = (_with_alpha(style.idle, 0.28 * a), _hatch_brush(_with_alpha(style.muted, 0.85 * a)))
            p.fillRect(rect, wp[0])
            p.fillRect(rect, wp[1])
            stripe = r.stripe or r.color
            if stripe:
                p.fillRect(QRectF(x0, top + inner - 2, x1 - x0, 2), _tinted(stripe, 0.9 * a))
        elif r.activity == lod.ANNOT:
            p.setPen(QPen(_tinted(r.color, a), 1))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(rect.adjusted(0.5, 0.5, -0.5, -0.5))
        else:
            p.fillRect(rect, _tinted(r.color, a))
        if opts.matched is not None and r.eid >= 0 and opts.matched(r.eid):
            p.setPen(QPen(style.accent, 2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(QRectF(x0 + 0.5, 1.5, max(1.0, x1 - x0 - 1), h - 3))
        if r.eid >= 0 and r.eid == opts.selected_eid:
            sel_rect = QRectF(x0, 1, x1 - x0, h - 2)
    for x0, x1 in opts.inferred_waits or ():
        rect = QRectF(x0, top, max(1.0, x1 - x0), inner)
        p.fillRect(rect, _hatch_brush(_with_alpha(style.background, 0.75), Qt.BrushStyle.Dense6Pattern))
    if opts.cp_steps:
        wait_fill, wait_hatch = _with_alpha(style.critical, 0.35), _hatch_brush(style.critical)
        for x0, x1, mode, _tier in opts.cp_steps:
            rect = QRectF(x0, 0, max(1.0, x1 - x0), BAND)
            if mode == "busy":
                p.fillRect(rect, style.critical)
            elif mode in ("wait", "gap", "idle"):
                p.fillRect(rect, wait_fill)
                p.fillRect(rect, wait_hatch)
    if sel_rect is not None:
        p.setPen(QPen(style.accent, 2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(sel_rect)
    p.restore()


def paint_group(p: QPainter, busy: np.ndarray, wait: np.ndarray, colors: list, w: float, h: float,
                style: Style, *, cp_cols: np.ndarray | None = None) -> None:
    p.save()
    img = density_image(busy, wait, np.zeros(len(busy), dtype=np.int64), colors, int(round(h)), style,
                        alpha=0.8, ticks=False)
    if img is not None:
        p.drawImage(QRectF(0, 0, len(busy), int(round(h))), img)
    if cp_cols is not None and len(cp_cols):
        for c in np.nonzero(cp_cols > 0)[0].tolist():
            p.fillRect(QRectF(c, 0, 1, BAND), _with_alpha(style.critical, 0.35 + 0.65 * float(cp_cols[c])))
    p.restore()


# ── overlay: dependency edges and critical-path hops ────────────────────

def _marker(p: QPainter, family: str, pt: QPointF, size: float, color: QColor) -> None:
    p.setPen(QPen(color, 1))
    p.setBrush(color)
    s = size
    if family in ("mpi_collective", "openmp", "nccl"):
        p.drawPolygon(QPolygonF([QPointF(pt.x(), pt.y() - s), QPointF(pt.x() + s, pt.y()),
                                 QPointF(pt.x(), pt.y() + s), QPointF(pt.x() - s, pt.y())]))
    elif family == "submission":
        p.drawRect(QRectF(pt.x() - s * 0.8, pt.y() - s * 0.8, 1.6 * s, 1.6 * s))
    elif family == "gpu_sync":
        p.drawPolygon(QPolygonF([QPointF(pt.x(), pt.y() - s), QPointF(pt.x() + s, pt.y() + s),
                                 QPointF(pt.x() - s, pt.y() + s)]))
    else:
        p.drawEllipse(pt, s * 0.85, s * 0.85)


def _arrowhead(p: QPainter, a: QPointF, b: QPointF, color: QColor, size: float = 6.0) -> None:
    dx, dy = b.x() - a.x(), b.y() - a.y()
    n = (dx * dx + dy * dy) ** 0.5
    if n < 1e-6:
        dx, dy, n = 0.0, 1.0, 1.0
    ux, uy = dx / n, dy / n
    left = QPointF(b.x() - size * ux + size * 0.5 * uy, b.y() - size * uy - size * 0.5 * ux)
    right = QPointF(b.x() - size * ux - size * 0.5 * uy, b.y() - size * uy + size * 0.5 * ux)
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(color)
    p.drawPolygon(QPolygonF([b, left, right]))


@dataclass
class OverlayEdge:
    """One drawable dependency: pixel endpoints and its style keys."""
    id: int
    x0: float
    y0: float
    x1: float
    y1: float
    family: str
    tier: str
    cls: str
    on_path: bool
    anchored: tuple[bool, bool] = (True, True)     # endpoint on its own lane (False: on a collapsed header)


def paint_edges(p: QPainter, edges: Iterable[OverlayEdge], style: Style, *, selected: int = -1,
                path_mode: bool = False) -> None:
    p.save()
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    for e in edges:
        a, b = QPointF(e.x0, e.y0), QPointF(e.x1, e.y1)
        fam = "critical" if (path_mode and e.on_path) else e.family
        color = QColor(theme_mod.family_color(fam, style.dark))
        if e.id == selected:
            halo = QPen(_with_alpha(style.accent, 0.45), TIER_WIDTH.get(e.tier, 1.2) + 5)
            p.setPen(halo)
            p.drawLine(a, b)
        pen = tier_pen(color, e.tier, e.cls, scale=1.3 if (path_mode and e.on_path) else 1.0)
        p.setPen(pen)
        p.drawLine(a, b)
        _marker(p, e.family, a, 3.2 if e.anchored[0] else 2.2, _with_alpha(color, TIER_ALPHA.get(e.tier, 0.6)))
        _arrowhead(p, a, b, _with_alpha(color, TIER_ALPHA.get(e.tier, 0.6)))
        if e.id == selected:
            p.setPen(QPen(style.accent, 2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(a, 6, 6)
            p.drawEllipse(b, 6, 6)
    p.restore()


def paint_transition_markers(p: QPainter, points: Iterable[tuple[float, float]], style: Style) -> None:
    """A path switch (rank / thread / runtime / device / stream): a small
    chevron above the lane the path continues on."""
    p.save()
    p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    for x, y in points:
        p.setPen(QPen(style.background, 1))
        p.setBrush(style.critical)
        p.drawPolygon(QPolygonF([QPointF(x - 5, y - 8), QPointF(x + 5, y - 8), QPointF(x, y - 2)]))
    p.restore()


# ── span-level score track (legacy DP: what it counts, and twice) ────────

def paint_score_track(p: QPainter, legacy: dict, a: float, b: float, w: float, h: float, style: Style) -> None:
    p.save()
    if not legacy or not legacy.get("available"):
        p.restore()
        return
    k = w / max(b - a, 1e-9)
    ivs = [iv for iv in legacy["intervals"] if iv[1] > a and iv[0] < b]
    # stack intervals greedily into rows: overlapping counted intervals
    # end up on different rows, so the overlap is visible as stacking
    ends: list[float] = []
    rows: list[int] = []
    for s, e, _n, _k in sorted(ivs):
        for r, end in enumerate(ends):
            if end <= s:
                ends[r] = e
                rows.append(r)
                break
        else:
            ends.append(e)
            rows.append(len(ends) - 1)
    nrows = max(1, len(ends))
    rh = max(2.0, (h - 4) / nrows)
    for (s, e, _n, _k), r in zip(sorted(ivs), rows):
        x0, x1 = max(0.0, (s - a) * k), min(w, (e - a) * k)
        p.fillRect(QRectF(x0, 2 + r * rh, max(1.0, x1 - x0), max(1.0, rh - 1)), _with_alpha(style.muted, 0.8))
    for s, e, depth in legacy.get("overlaps", ()):
        if e <= a or s >= b:
            continue
        x0, x1 = max(0.0, (s - a) * k), min(w, (e - a) * k)
        rect = QRectF(x0, 0, max(1.0, x1 - x0), h)
        p.fillRect(rect, _with_alpha(style.overlap, 0.25))
        p.fillRect(rect, _hatch_brush(style.overlap, Qt.BrushStyle.FDiagPattern))
    p.restore()


# ── minimap (overview of the whole run + the visible window) ────────────

def paint_minimap(p: QPainter, busy: np.ndarray, wait: np.ndarray, cp: np.ndarray | None, t0: float, t1: float,
                  a: float, b: float, w: float, h: float, style: Style) -> None:
    p.save()
    p.fillRect(QRectF(0, 0, w, h), style.surface)
    n = len(busy)
    if n:
        img = density_image(busy, wait, np.zeros(n, dtype=np.int64), [None] * n, int(h), style,
                            fallback=style.accent.name(), alpha=0.7, ticks=False)
        if img is not None:
            p.drawImage(QRectF(0, 0, w, h), img)
        if cp is not None and len(cp):
            bw = w / len(cp)
            for c in np.nonzero(cp > 0)[0].tolist():
                p.fillRect(QRectF(c * bw, 0, max(1.0, bw), BAND), _with_alpha(style.critical, 0.4 + 0.6 * float(cp[c])))
    span = max(t1 - t0, 1e-9)
    x0, x1 = (a - t0) / span * w, (b - t0) / span * w
    win = QRectF(x0, 0.5, max(2.0, x1 - x0), h - 1)
    p.fillRect(win, _with_alpha(style.accent, 0.12))
    p.setPen(QPen(style.accent, 1.5))
    p.setBrush(Qt.BrushStyle.NoBrush)
    p.drawRect(win)
    p.restore()


# ── figure export (publication: axis, labels, legend) ───────────────────

def paint_ruler(p: QPainter, ticks: list[dict], a: float, b: float, x: float, w: float, h: float,
                style: Style, font: QFont) -> None:
    p.save()
    p.setFont(font)
    p.setPen(QPen(style.text, 1))
    p.drawLine(QPointF(x, h - 1), QPointF(x + w, h - 1))
    k = w / max(b - a, 1e-9)
    for t in ticks:
        tx = x + (t["ns"] - a) * k
        p.drawLine(QPointF(tx, h - 6), QPointF(tx, h - 1))
        p.drawText(QPointF(tx + 2, h - 8), t["label"])
    p.restore()


def legend_entries(dark: bool, *, color_mode: str, has_density: bool, has_unobserved: bool,
                   families: Iterable[str] = (), critical: bool = False) -> list[dict[str, Any]]:
    """What a figure (or the on-screen legend) explains: one entry per
    encoding actually drawn."""
    out: list[dict[str, Any]] = []
    if color_mode in ("bucket", "activity"):
        for act in ("work", "communication", "memory", "overhead", "jit"):
            out.append({"kind": "fill", "color": theme_mod.activity_color(act, dark),
                        "label": lod.ACTIVITY_LABELS[lod.ACTIVITY.index(act)]})
    out.append({"kind": "wait", "color": theme_mod.activity_color("wait", dark),
                "label": "Explicit waiting (hatched)"})
    if has_density:
        out.append({"kind": "density", "color": theme_mod.text_muted_color(dark),
                    "label": "Events shorter than a pixel: bar height = busy share of the column; tick = events present"})
    if has_unobserved:
        out.append({"kind": "unobserved", "color": theme_mod.text_muted_color(dark),
                    "label": "No data (before the lane's first / after its last event)"})
    if critical:
        out.append({"kind": "critical", "color": theme_mod.critical_color(dark),
                    "label": "Observed critical path (band: solid = busy, hatched = waiting)"})
    from .timeline_analysis import FAMILY_LABELS
    for fam in families:
        out.append({"kind": "edge", "color": theme_mod.family_color(fam, dark),
                    "label": FAMILY_LABELS.get(fam, fam), "family": fam})
    if families:
        out.append({"kind": "tiers", "color": theme_mod.text_muted_color(dark),
                    "label": "Edge width/opacity = evidence tier (certain > high > medium > low); "
                             "solid = direct, dashed = inferred, dotted = low"})
    return out
