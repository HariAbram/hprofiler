"""
QML items that paint the Timeline with QPainter (timeline_paint.py), so the
same code renders the screen, exported figures and the pixel tests. Each
item only holds view parameters; the data and the level-of-detail decision
come from its TimelineModel (`model`), which caches renders per viewport.

Registered as Hprofiler 1.0 types (register_types(), called by
TimelineModel's constructor so every engine that has a model has them):

    LanePainter        one lane, or a header row's aggregate strip
    OverlayPainter     dependency edges, critical-path hops, path switches
                       (the visible rows only; scrolls with them)
    MinimapPainter     whole-run overview with the visible window
    ScoreTrackPainter  what the span-level search counts, and twice
"""
from __future__ import annotations

from PySide6.QtCore import Property, QObject, Signal
from PySide6.QtGui import QPainter
from PySide6.QtQuick import QQuickPaintedItem


def _prop(typ, name: str, default, *, repaint: bool = True):
    attr = "_" + name
    sig_name = name + "Changed"

    def getter(self):
        return getattr(self, attr, default)

    def setter(self, value):
        if getattr(self, attr, default) == value:
            return
        setattr(self, attr, value)
        getattr(self, sig_name).emit()
        if repaint:
            self.update()
    return getter, setter, sig_name


class _Painted(QQuickPaintedItem):
    modelChanged = Signal()
    viewStartNsChanged = Signal()
    viewEndNsChanged = Signal()
    revisionChanged = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._model = None
        self._viewStartNs = 0.0
        self._viewEndNs = 1.0
        self._revision = 0
        self.setOpaquePainting(False)
        self.setAntialiasing(False)

    def _get_model(self):
        return self._model

    def _set_model(self, m) -> None:
        if m is self._model:
            return
        self._model = m
        self.modelChanged.emit()
        self.update()

    model = Property(QObject, _get_model, _set_model, notify=modelChanged)
    viewStartNs = Property(float, *_prop(float, "viewStartNs", 0.0)[:2], notify=viewStartNsChanged)
    viewEndNs = Property(float, *_prop(float, "viewEndNs", 1.0)[:2], notify=viewEndNsChanged)
    revision = Property(int, *_prop(int, "revision", 0)[:2], notify=revisionChanged)


class LanePainter(_Painted):
    laneIndexChanged = Signal()
    kindChanged = Signal()
    groupIdChanged = Signal()
    laneIndexesChanged = Signal()
    laneNameChanged = Signal()
    syncOverlayLaneIndexChanged = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._laneIndex = -1
        self._kind = "lane"
        self._groupId = ""
        self._laneIndexes = []
        self._laneName = ""
        self._syncOverlayLaneIndex = -1

    laneIndex = Property(int, *_prop(int, "laneIndex", -1)[:2], notify=laneIndexChanged)
    kind = Property(str, *_prop(str, "kind", "lane")[:2], notify=kindChanged)
    groupId = Property(str, *_prop(str, "groupId", "")[:2], notify=groupIdChanged)
    laneIndexes = Property('QVariantList', *_prop(list, "laneIndexes", [])[:2], notify=laneIndexesChanged)
    # read by tests and by the pairing legend: the lane's store name, and the
    # lane its explicit waits come from (-1 none)
    laneName = Property(str, *_prop(str, "laneName", "", repaint=False)[:2], notify=laneNameChanged)
    syncOverlayLaneIndex = Property(int, *_prop(int, "syncOverlayLaneIndex", -1, repaint=False)[:2],
                                    notify=syncOverlayLaneIndexChanged)

    def paint(self, painter: QPainter) -> None:
        m = self._model
        if m is None or self.width() <= 0 or self.height() <= 0 or self._viewEndNs <= self._viewStartNs:
            return
        try:
            if self._kind == "lane":
                m.paint_lane_item(painter, self._laneIndex, self._viewStartNs, self._viewEndNs,
                                  self.width(), self.height())
            else:
                m.paint_group_item(painter, self._groupId, list(self._laneIndexes or []), self._viewStartNs,
                                   self._viewEndNs, self.width(), self.height())
        except Exception as exc:        # a paint must never take the GUI down
            m.note_paint_error(exc)


class OverlayPainter(_Painted):
    scrollYChanged = Signal()
    rowHeightChanged = Signal()
    hoverLaneChanged = Signal()
    hoverSpanIdxChanged = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._scrollY = 0.0
        self._rowHeight = 26.0
        self._hoverLane = -1
        self._hoverSpanIdx = -1.0
        self.setAntialiasing(True)

    scrollY = Property(float, *_prop(float, "scrollY", 0.0)[:2], notify=scrollYChanged)
    rowHeight = Property(float, *_prop(float, "rowHeight", 26.0)[:2], notify=rowHeightChanged)
    hoverLane = Property(int, *_prop(int, "hoverLane", -1)[:2], notify=hoverLaneChanged)
    hoverSpanIdx = Property(float, *_prop(float, "hoverSpanIdx", -1.0)[:2], notify=hoverSpanIdxChanged)

    def paint(self, painter: QPainter) -> None:
        m = self._model
        if m is None or self.width() <= 0 or self._viewEndNs <= self._viewStartNs:
            return
        try:
            m.paint_overlay_item(painter, self.width(), self.height(), self._viewStartNs, self._viewEndNs,
                                 self._scrollY, self._rowHeight, self._hoverLane, self._hoverSpanIdx)
        except Exception as exc:
            m.note_paint_error(exc)


class MinimapPainter(_Painted):
    def paint(self, painter: QPainter) -> None:
        m = self._model
        if m is None or self.width() <= 0:
            return
        try:
            m.paint_minimap_item(painter, self.width(), self.height(), self._viewStartNs, self._viewEndNs)
        except Exception as exc:
            m.note_paint_error(exc)


class ScoreTrackPainter(_Painted):
    def paint(self, painter: QPainter) -> None:
        m = self._model
        if m is None or self.width() <= 0:
            return
        try:
            m.paint_score_item(painter, self.width(), self.height(), self._viewStartNs, self._viewEndNs)
        except Exception as exc:
            m.note_paint_error(exc)


_REGISTERED = False


def register_types() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    from PySide6.QtQml import qmlRegisterType
    for cls, name in ((LanePainter, "LanePainter"), (OverlayPainter, "OverlayPainter"),
                      (MinimapPainter, "MinimapPainter"), (ScoreTrackPainter, "ScoreTrackPainter")):
        qmlRegisterType(cls, "Hprofiler", 1, 0, name)
    _REGISTERED = True
