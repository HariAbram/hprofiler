"""
Column specifications for the GUI's data tables (src/gui/tablemodel.py's
DictListTableModel/TableConfig, rendered by components/DataTable.qml) --
one place naming every column a table can show, its formatting `kind`, and
(for numeric columns) its tooltip definition.

A row is a plain dict (the shape every bridge builds via
trace.aggregated_stats()/etc.) -- a ColumnSpec describes how to turn one
dict key into a header + a formatted, sortable, filterable cell.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ColumnSpec:
    key: str            # the row dict key this column reads
    title: str           # header text
    # text | time_ns | count | pct | bytes | bandwidth | throughput |
    # category | badge | status
    kind: str = "text"
    width: float = 90
    align: str = "right"           # "left" | "right"
    numeric: bool = False           # sortable/filterable as a number, not a string
    bar: str = ""                   # "" | "self" (bar width from this column's own
                                     #   own value) | another column's key to size the bar by
    bar_max_key: str = ""           # row-dict key holding this row's bar denominator,
                                     # e.g. "totalNs" for a "self time" bar scaled to the row's own total
    pct_of: str = ""                 # for kind="pct": which raw column it's a percentage of (tooltip text only)
    definition: str = ""             # full-value/definition tooltip text
    frozen: bool = False             # stays fixed (doesn't scroll horizontally) -- at most one per table
    visible: bool = True             # default visibility (user can still toggle via ColumnMenu)


_TIME_DEF = "Wall-clock duration. See the Inspector for measured vs. derived."
_COUNT_DEF = "Number of times this event occurred in the trace."
_SHARE_DEF = "Share of the trace's total measured time this row accounts for."

KERNEL_COLUMNS: list[ColumnSpec] = [
    ColumnSpec("name", "Function", kind="text", width=220, align="left", frozen=True,
               definition="The function/kernel name (truncated for JIT hash names -- see rawName for the full identifier)."),
    ColumnSpec("category", "Category", kind="category", width=90, align="left",
               definition="Which runtime/backend this event belongs to (cpu, cuda, mpi, ...)."),
    ColumnSpec("count", "Calls", kind="count", width=70, numeric=True, definition=_COUNT_DEF),
    ColumnSpec("totalNs", "Total", kind="time_ns", width=90, numeric=True, bar="self",
               definition="Sum of every call's duration -- NOT merged-interval, so concurrent calls on different threads can sum past the trace's wall time."),
    ColumnSpec("avgNs", "Avg", kind="time_ns", width=90, numeric=True, definition="Mean duration per call: Total / Calls."),
    ColumnSpec("minNs", "Min", kind="time_ns", width=90, numeric=True, definition=_TIME_DEF, visible=False),
    ColumnSpec("maxNs", "Max", kind="time_ns", width=90, numeric=True, definition=_TIME_DEF, visible=False),
    ColumnSpec("sharePct", "Share", kind="pct", width=70, numeric=True, bar="self",
               pct_of="totalNs", definition=_SHARE_DEF),
]

CALLTREE_COLUMNS: list[ColumnSpec] = [
    ColumnSpec("name", "Function", kind="text", width=220, align="left", frozen=True,
               definition="Call-tree node name; indentation shows the call path."),
    ColumnSpec("category", "Category", kind="category", width=90, align="left"),
    ColumnSpec("count", "Calls", kind="count", width=70, numeric=True, definition=_COUNT_DEF),
    ColumnSpec("totalNs", "Total (incl.)", kind="time_ns", width=100, numeric=True, bar="self",
               definition="Inclusive time -- this node plus everything it called."),
    ColumnSpec("selfNs", "Self", kind="time_ns", width=90, numeric=True,
               definition="Exclusive time -- only while this frame itself was the directly-measured leaf, excluding children."),
    ColumnSpec("avgNs", "Avg", kind="time_ns", width=90, numeric=True, definition="Total (incl.) / Calls."),
]

DEVICE_COLUMNS: list[ColumnSpec] = [
    ColumnSpec("name", "Device", kind="text", width=180, align="left", frozen=True),
    ColumnSpec("backend", "Backend", kind="category", width=80, align="left"),
    ColumnSpec("computeCap", "Compute", kind="text", width=70),
    ColumnSpec("smCount", "SMs", kind="count", width=60, numeric=True),
    # kind="text" (not a Format.formatNumber kind) but numeric=True --
    # units are in the title instead of per-cell, so the raw float still
    # sorts correctly as a number (TableFilterProxy.lessThan() keys off
    # `numeric`, independent of `kind`).
    ColumnSpec("clockGhz", "Clock (GHz)", kind="text", width=90, numeric=True),
    ColumnSpec("peaks", "Peak FLOPs", kind="text", width=160, align="left"),
    ColumnSpec("bandwidth", "Bandwidth", kind="text", width=100),
    ColumnSpec("vram", "VRAM", kind="text", width=70),
    ColumnSpec("ridgeHint", "Roofline", kind="text", width=100, align="left"),
]

# Field-shaped rows (label/value/kind/reason -- see inspector.py's
# convention) rendered as a 2-column table so an absent counter shows
# "unavailable + reason" instead of the row silently vanishing (the
# existing `visible: System.ipc > 0`-style gating this replaces).
# kind="text": DataTableCell.qml separately recognizes a "value" column on
# a Field-shaped row (row.kind present) and colors it by that Field kind
# regardless of the column's OWN kind -- see that file's isFieldValueCell.
SYSTEM_METRIC_COLUMNS: list[ColumnSpec] = [
    ColumnSpec("label", "Metric", kind="text", width=160, align="left", frozen=True),
    ColumnSpec("value", "Value", kind="text", width=220, align="left"),
]

FINDINGS_COLUMNS: list[ColumnSpec] = [
    ColumnSpec("label", "Finding", kind="text", width=220, align="left", frozen=True),
    ColumnSpec("value", "Detail", kind="text", width=160, align="left"),
]

COMPARE_COLUMNS: list[ColumnSpec] = [
    ColumnSpec("name", "Function", kind="text", width=200, align="left", frozen=True),
    ColumnSpec("category", "Category", kind="category", width=80, align="left"),
    ColumnSpec("status", "Status", kind="status", width=90, align="left",
               definition="improved/regressed require the change to clear BOTH a minimum percentage and a minimum absolute-time noise floor -- see the Compare tab's own note. unchanged means within that floor, not identical."),
    ColumnSpec("baselineNs", "Baseline", kind="time_ns", width=90, numeric=True),
    ColumnSpec("comparisonNs", "Comparison", kind="time_ns", width=90, numeric=True),
    ColumnSpec("deltaNs", "Δ Time", kind="time_ns", width=90, numeric=True,
               definition="Comparison minus baseline; negative is faster."),
    ColumnSpec("deltaPct", "Δ %", kind="pct", width=70, numeric=True),
    ColumnSpec("matchKind", "Matched by", kind="text", width=90, align="left", visible=False,
               definition="exact: identical (category,name). normalized: matched after stripping JIT-hash naming. unmatched: only present in one run."),
]

METRIC_DEFINITIONS: dict[str, str] = {
    "totalNs": KERNEL_COLUMNS[3].definition,
    "avgNs": KERNEL_COLUMNS[4].definition,
    "sharePct": _SHARE_DEF,
    "count": _COUNT_DEF,
    "selfNs": "Exclusive time -- only while this frame itself was the directly-measured leaf, excluding children.",
    "deltaNs": "Comparison minus baseline; negative is faster.",
}
