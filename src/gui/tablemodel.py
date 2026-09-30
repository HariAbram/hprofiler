"""
Reusable table infrastructure for the GUI's data tables (Kernels, Call Tree,
System devices/metrics, Overview findings, Compare) -- built on real Qt
model/view classes (QAbstractListModel + QSortFilterProxyModel), not the
per-screen hand-rolled JS .filter()/.sort() every table used before this
round (KernelsScreen.qml's old recompute()).

Deliberately NOT QAbstractTableModel + QML TableView/HorizontalHeaderView:
those are *styled* Qt Quick Controls types, and this project has already
been bitten twice by style/component resolution under the offscreen QPA
platform this test suite runs under (see tests/test_gui_timeline_hover.py's
module docstring). A QAbstractListModel (one row = one item, one role per
column) driving a plain ListView sidesteps that risk entirely while still
being genuinely "Qt's model/view architecture" -- QSortFilterProxyModel
re-sorts/re-filters incrementally (sort()/beginFilterChange()+endFilterChange(), never
beginResetModel() on the SOURCE model), which is what makes "avoid
unnecessary full-table rebuilding" literally true rather than aspirational.
The frozen identifier column reuses TimelineScreen.qml's existing fixed-
label-column pattern (a fixed-width Item + an x-offset scroll for the rest)
instead of a second synced view.

Column formatting (kind -> display string) is centralized in FormatBridge
below (registered as the "Format" singleton) so every table's cell
delegate calls the SAME small set of src/analysis/dashboard.py fmt_*
helpers, rather than each screen re-implementing number formatting in QML
JS (a real drift risk -- see that module's fmt_ns/fmt_pct/etc.).
"""
from __future__ import annotations

import csv
from typing import Any

from PySide6.QtCore import (
    QObject, Property, Signal, Slot, Qt, QAbstractListModel, QSortFilterProxyModel,
    QModelIndex,
)

from ..analysis import dashboard as dash
from . import clipboard
from .columns import ColumnSpec

# One Qt role per column (assigned by position in the ColumnSpec list, so
# it's stable for a given table's lifetime -- tables are built once per
# trace, like every other bridge in this codebase), plus one extra "row"
# role (immediately after the last column role) holding the whole source
# dict for delegates that want the full row (click-to-select, copy).


class DictListTableModel(QAbstractListModel):
    """One row = one dict (the same shape every bridge already builds via
    trace.aggregated_stats()/etc.) -- this class's only job is exposing
    that list of dicts to QML through real Qt roles instead of a single
    QVariantList property QML re-filters/re-sorts in JS."""

    def __init__(self, rows: list[dict[str, Any]], columns: list[ColumnSpec],
                 parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._columns = list(columns)
        self._role_for_key = {c.key: Qt.UserRole + 1 + i for i, c in enumerate(self._columns)}
        self._row_role = Qt.UserRole + 1 + len(self._columns)
        self._rows: list[dict[str, Any]] = list(rows)

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._rows)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:
        if not index.isValid() or not (0 <= index.row() < len(self._rows)):
            return None
        row = self._rows[index.row()]
        if role == self._row_role:
            return row
        for key, r in self._role_for_key.items():
            if r == role:
                return row.get(key)
        return None

    def roleNames(self) -> dict[int, bytes]:
        names = {r: key.encode() for key, r in self._role_for_key.items()}
        names[self._row_role] = b"row"
        return names

    def setRows(self, rows: list[dict[str, Any]]) -> None:
        """Replaces the source data -- a genuine reset, unlike sorting/
        filtering (which go through TableFilterProxy and never call this).
        Every bridge in this codebase computes its data once at
        construction and never calls this in normal operation; it exists
        for completeness/testability, not because tables currently reload."""
        self.beginResetModel()
        self._rows = list(rows)
        self.endResetModel()

    def rowDict(self, row: int) -> dict[str, Any]:
        return self._rows[row] if 0 <= row < len(self._rows) else {}

    def columnSpecs(self) -> list[ColumnSpec]:
        return self._columns

    def roleForKey(self, key: str) -> int:
        return self._role_for_key.get(key, -1)


class TableFilterProxy(QSortFilterProxyModel):
    """Sorting/filtering state for one table. All mutation goes through
    sort()/beginFilterChange()+endFilterChange() (incremental Qt operations) -- never touches
    the source DictListTableModel, so re-sorting or re-filtering a table
    never fires the source model's modelAboutToBeReset/modelReset signals
    (see tests/test_gui_bridge.py's TestDataTableModels for the QSignalSpy
    assertion that pins this down as a regression guard)."""

    sortChanged = Signal()
    filtersChanged = Signal()
    countChanged = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._sort_key = ""
        self._sort_descending = False
        self._text_filter = ""
        self._text_filter_keys: list[str] = []
        self._min_values: dict[str, float] = {}
        self._max_values: dict[str, float] = {}
        self._category_filter: list[str] = []
        self.setDynamicSortFilter(True)
        self.rowsInserted.connect(self.countChanged)
        self.rowsRemoved.connect(self.countChanged)
        self.modelReset.connect(self.countChanged)

    def _model(self) -> DictListTableModel | None:
        m = self.sourceModel()
        return m if isinstance(m, DictListTableModel) else None

    # ── sorting ──────────────────────────────────────────────────────────
    @Property(str, notify=sortChanged)
    def sortKey(self) -> str:
        return self._sort_key

    @Property(bool, notify=sortChanged)
    def sortDescending(self) -> bool:
        return self._sort_descending

    @Slot(str)
    def toggleSort(self, key: str) -> None:
        if self._sort_key == key:
            self._sort_descending = not self._sort_descending
        else:
            self._sort_key = key
            self._sort_descending = False
        self.sortChanged.emit()
        self.sort(0, Qt.DescendingOrder if self._sort_descending else Qt.AscendingOrder)

    def lessThan(self, left: QModelIndex, right: QModelIndex) -> bool:
        model = self._model()
        if model is None or not self._sort_key:
            return False
        role = model.roleForKey(self._sort_key)
        lv = model.data(left, role)
        rv = model.data(right, role)
        spec = next((c for c in model.columnSpecs() if c.key == self._sort_key), None)
        if spec is not None and spec.numeric:
            return float(lv or 0) < float(rv or 0)
        return str(lv or "").lower() < str(rv or "").lower()

    # ── filtering ────────────────────────────────────────────────────────
    @Property(str, notify=filtersChanged)
    def textFilter(self) -> str:
        return self._text_filter

    @textFilter.setter
    def textFilter(self, value: str) -> None:
        if value != self._text_filter:
            self.beginFilterChange()
            self._text_filter = value
            self.filtersChanged.emit()
            self.endFilterChange(QSortFilterProxyModel.Direction.Rows)

    @Property('QVariantList', notify=filtersChanged)
    def textFilterKeys(self) -> list[str]:
        return self._text_filter_keys

    @textFilterKeys.setter
    def textFilterKeys(self, keys: list) -> None:
        self.beginFilterChange()
        self._text_filter_keys = list(keys)
        self.filtersChanged.emit()
        self.endFilterChange(QSortFilterProxyModel.Direction.Rows)

    @Slot(str, float)
    def setMin(self, key: str, value: float) -> None:
        self.beginFilterChange()
        self._min_values[key] = value
        self.filtersChanged.emit()
        self.endFilterChange(QSortFilterProxyModel.Direction.Rows)

    @Slot(str, float)
    def setMax(self, key: str, value: float) -> None:
        self.beginFilterChange()
        self._max_values[key] = value
        self.filtersChanged.emit()
        self.endFilterChange(QSortFilterProxyModel.Direction.Rows)

    @Slot(str)
    def clearMin(self, key: str) -> None:
        self.beginFilterChange()
        self._min_values.pop(key, None)
        self.filtersChanged.emit()
        self.endFilterChange(QSortFilterProxyModel.Direction.Rows)

    @Slot(str)
    def clearMax(self, key: str) -> None:
        self.beginFilterChange()
        self._max_values.pop(key, None)
        self.filtersChanged.emit()
        self.endFilterChange(QSortFilterProxyModel.Direction.Rows)

    @Property('QVariantList', notify=filtersChanged)
    def categoryFilter(self) -> list[str]:
        return self._category_filter

    @Slot(str, bool)
    def setCategoryEnabled(self, category: str, on: bool) -> None:
        self.beginFilterChange()
        if on and category not in self._category_filter:
            self._category_filter.append(category)
        elif not on and category in self._category_filter:
            self._category_filter.remove(category)
        self.filtersChanged.emit()
        self.endFilterChange(QSortFilterProxyModel.Direction.Rows)

    @Slot()
    def clearFilters(self) -> None:
        self.beginFilterChange()
        self._text_filter = ""
        self._min_values.clear()
        self._max_values.clear()
        self._category_filter = []
        self.filtersChanged.emit()
        self.endFilterChange(QSortFilterProxyModel.Direction.Rows)

    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:
        model = self._model()
        if model is None:
            return True
        row = model.rowDict(source_row)

        if self._text_filter:
            needle = self._text_filter.lower()
            keys = self._text_filter_keys or [c.key for c in model.columnSpecs() if not c.numeric]
            if not any(needle in str(row.get(k, "")).lower() for k in keys):
                return False

        for key, lo in self._min_values.items():
            if float(row.get(key, 0) or 0) < lo:
                return False
        for key, hi in self._max_values.items():
            if float(row.get(key, 0) or 0) > hi:
                return False

        if self._category_filter and row.get("category") not in self._category_filter:
            return False

        return True

    @Property(int, notify=countChanged)
    def matchCount(self) -> int:
        return self.rowCount()

    @Property(int, constant=True)
    def sourceCount(self) -> int:
        model = self._model()
        return model.rowCount() if model else 0


class TableConfig(QObject):
    """Per-table session state that must survive tab switches: column
    widths/visibility/order and the absolute/percentage display toggle.
    Deliberately in-memory only (no QSettings/disk persistence) -- "session"
    here means "don't reset when switching tabs," not "remember across a
    relaunch"; every bridge that owns a TableConfig is a singleton that
    lives for the whole GUI process, which already satisfies that."""

    columnsChanged = Signal()
    modeChanged = Signal()

    def __init__(self, columns: list[ColumnSpec], parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._specs = list(columns)
        self._order = [c.key for c in self._specs]
        self._widths = {c.key: c.width for c in self._specs}
        self._visible = {c.key: c.visible for c in self._specs}
        self._percent_mode = False

    @Property('QVariantList', notify=columnsChanged)
    def columns(self) -> list[dict[str, Any]]:
        by_key = {c.key: c for c in self._specs}
        out = []
        for key in self._order:
            c = by_key[key]
            out.append({
                "key": c.key, "title": c.title, "kind": c.kind,
                "width": self._widths[key], "align": c.align, "numeric": c.numeric,
                "bar": c.bar, "barMaxKey": c.bar_max_key, "pctOf": c.pct_of,
                "definition": c.definition, "frozen": c.frozen,
                "visible": self._visible[key],
            })
        return out

    @Property(bool, notify=modeChanged)
    def percentMode(self) -> bool:
        return self._percent_mode

    @Slot()
    def togglePercentMode(self) -> None:
        self._percent_mode = not self._percent_mode
        self.modeChanged.emit()

    @Slot(str, float)
    def setColumnWidth(self, key: str, width: float) -> None:
        if key in self._widths and width > 0:
            self._widths[key] = width
            self.columnsChanged.emit()

    @Slot(str, bool)
    def setColumnVisible(self, key: str, visible: bool) -> None:
        if key in self._visible:
            self._visible[key] = visible
            self.columnsChanged.emit()

    @Slot(int, int)
    def moveColumn(self, from_index: int, to_index: int) -> None:
        if 0 <= from_index < len(self._order) and 0 <= to_index < len(self._order):
            key = self._order.pop(from_index)
            self._order.insert(to_index, key)
            self.columnsChanged.emit()

    @Slot()
    def resetLayout(self) -> None:
        self._order = [c.key for c in self._specs]
        self._widths = {c.key: c.width for c in self._specs}
        self._visible = {c.key: c.visible for c in self._specs}
        self.columnsChanged.emit()


class TableBundle(QObject):
    """What a bridge exposes as ONE property for a table -- `rows` (the
    proxy, bind a ListView's model directly to it), `filters` (the same
    proxy, for its sort/filter slots -- exposed under a second name for
    QML-side clarity, not a second object), and `config`. Also owns
    row-level actions (copy/export) since those need both the CURRENT
    (sorted+filtered) view order and the column visibility state."""

    def __init__(self, rows: list[dict[str, Any]], columns: list[ColumnSpec],
                 parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._model = DictListTableModel(rows, columns)
        self._proxy = TableFilterProxy()
        self._proxy.setSourceModel(self._model)
        self._config = TableConfig(columns)
        self._bar_maxima: dict[str, float] = {}
        for c in columns:
            if c.bar and c.numeric:
                values = [float(r.get(c.key, 0) or 0) for r in rows]
                self._bar_maxima[c.key] = max(values) if values else 0.0

    @Property(QObject, constant=True)
    def rows(self) -> QObject:
        return self._proxy

    @Property(QObject, constant=True)
    def filters(self) -> QObject:
        return self._proxy

    @Property(QObject, constant=True)
    def config(self) -> QObject:
        return self._config

    @Property('QVariantMap', constant=True)
    def barMaxima(self) -> dict[str, float]:
        return self._bar_maxima

    def setRows(self, rows: list[dict[str, Any]]) -> None:
        """Replaces the underlying data wholesale (a genuine model reset,
        via DictListTableModel.setRows() -- see its own docstring on why
        that's different from sort/filter). For bridges whose data CAN
        legitimately change after construction, unlike this codebase's
        usual "compute once, expose constant=True" bridges -- e.g.
        ComparisonBridge.setChangeThresholds() re-classifying every row
        against new noise-floor thresholds. Recomputes barMaxima too,
        since a genuinely different row set can have different maxima."""
        self._model.setRows(rows)
        for c in self._model.columnSpecs():
            if c.bar and c.numeric:
                values = [float(r.get(c.key, 0) or 0) for r in rows]
                self._bar_maxima[c.key] = max(values) if values else 0.0

    def _visible_columns(self) -> list[ColumnSpec]:
        visible_keys = {c["key"] for c in self._config.columns if c["visible"]}
        order = [c["key"] for c in self._config.columns]
        by_key = {c.key: c for c in self._model.columnSpecs()}
        return [by_key[k] for k in order if k in visible_keys]

    def _proxy_row_dict(self, proxy_row: int) -> dict[str, Any]:
        idx = self._proxy.index(proxy_row, 0)
        if not idx.isValid():
            return {}
        return self._model.rowDict(self._proxy.mapToSource(idx).row())

    @Slot(int, result=str)
    def rowAsText(self, proxy_row: int) -> str:
        row = self._proxy_row_dict(proxy_row)
        cols = self._visible_columns()
        return "\t".join(str(row.get(c.key, "")) for c in cols)

    @Slot(int)
    def copyRow(self, proxy_row: int) -> None:
        clipboard.copy_text(self.rowAsText(proxy_row))

    @Slot()
    def copyAll(self) -> None:
        lines = [self.rowAsText(r) for r in range(self._proxy.rowCount())]
        clipboard.copy_text("\n".join(lines))

    @Slot(str, result=bool)
    def exportCsv(self, path: str) -> bool:
        """Writes the CURRENTLY visible, sorted, filtered rows -- matches
        InspectorBridge.exportTo()'s plain open()+try/except OSError
        pattern, csv.writer instead of json.dump."""
        cols = self._visible_columns()
        try:
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([c.title for c in cols])
                for r in range(self._proxy.rowCount()):
                    row = self._proxy_row_dict(r)
                    writer.writerow([row.get(c.key, "") for c in cols])
            return True
        except OSError:
            return False


class FormatBridge(QObject):
    """Registered as the "Format" singleton -- every table cell formats a
    raw numeric value through this instead of re-implementing
    src/analysis/dashboard.py's fmt_* helpers in QML JS a second time."""

    @Slot(str, float, result=str)
    def formatNumber(self, kind: str, value: float) -> str:
        if kind == "time_ns":
            return dash.fmt_ns(value)
        if kind == "count":
            return dash.fmt_count(value)
        if kind == "pct":
            return dash.fmt_pct(value)
        if kind == "bytes":
            return dash.fmt_bytes(value)
        if kind == "bandwidth":
            return dash.fmt_bandwidth_gbs(value)
        if kind == "throughput":
            return dash.fmt_tf(value)
        return str(value)

    @Slot(float, result=str)
    def signedNs(self, value: float) -> str:
        return dash.fmt_signed_ns(value)

    @Slot(float, result=str)
    def signedPct(self, value: float) -> str:
        return dash.fmt_signed_pct(value)
