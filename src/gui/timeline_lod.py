"""
Level of detail for one Timeline lane: what is drawn as an exact interval
and what as a per-pixel-column density -- decided here, Qt-free, so the
rule that keeps the Timeline temporally honest is unit-testable.

The rule
--------
A span at least one pixel wide is an exact rectangle from its start to its
end (clipped to the view; fractional pixel edges are kept). A span
narrower than a pixel is never widened to one: the pixel columns it falls
in get its busy time as a FRACTION of the column -- painted as a bar whose
height is that fraction -- and an event count. A column in which only a
few nanoseconds are busy therefore shows a sliver plus a presence tick,
never a full-height block, at any zoom.

Before this module, two paths stretched short events into continuous bars:
spans narrower than a pixel were drawn one pixel wide at full height (a
lane with thousands of 10 ns events every few microseconds read as solidly
busy when zoomed out), and occupancy bins were painted with an opacity
floor of 25% for any non-zero value, so a column 0.1% busy looked a quarter
busy -- and both switched abruptly at a fixed span count, so the apparent
activity of a lane changed with the zoom.

Sources
-------
* exact: the window's spans are enumerated from the store (up to
  EXACT_LIMIT of them); rectangles and densities are exact.
* index: more spans than that in the window -- the store's precomputed
  multiresolution activity index gives busy time per column (union of the
  lane's spans) and, at its finest level, the number of spans starting per
  column. Busy time is assumed uniform inside one index bin, so a column
  narrower than a few index bins is approximate: the render says so
  (`approx`) and by how much (`index_bin_ns`).

Activity classes (fill colour depends on the colour mode; the class picks
the PATTERN, so waiting never reads as work in any mode):
    work, communication, memory, overhead, jit   solid
    wait (explicit synchronization), idle        hatched, neutral tone
    annotation                                   outline only
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np

from ..analysis import activity_buckets as ab

WORK, COMM, MEM, OVERHEAD, JIT, WAIT, IDLE, ANNOT, OTHER = range(9)
ACTIVITY = ("work", "communication", "memory", "overhead", "jit", "wait", "idle", "annotation", "other")
ACTIVITY_LABELS = ("Productive computation", "Communication", "Memory transfer", "Runtime overhead",
                   "JIT compilation", "Explicit waiting (synchronization)", "Off-CPU / idle (measured)",
                   "Annotation range", "Other")
WAIT_LIKE = frozenset({WAIT, IDLE})

_BUCKET_CLASS = {
    "Computation": WORK, "Communication": COMM, "Memory transfer": MEM, "Runtime overhead": OVERHEAD,
    "Synchronization": WAIT, "Idle": IDLE, "Annotation": ANNOT,
}

# Spans enumerated per lane and paint at most; above it the activity index.
EXACT_LIMIT = 6000
# Narrower than this many pixels: aggregated into its columns.
MIN_RECT_PX = 1.0


def activity_of(span) -> int:
    """The activity class of a span (activity_buckets' bucket, with JIT
    compilation split from runtime overhead)."""
    if span.category.value == "jit":
        return JIT
    return _BUCKET_CLASS.get(ab.bucket_of_span(span), OTHER)


@dataclass
class Rect:
    x0: float               # pixels from the lane's left edge, clipped to [0, width]
    x1: float
    color: str
    activity: int
    eid: int                # store event id (-1: a synthetic piece, e.g. a wait cut-out)
    start_ns: int
    end_ns: int
    name: str = ""
    stripe: str = ""        # wait pieces: the wait's own colour, drawn as a thin stripe
    inside_wait: bool = False


@dataclass
class LaneRender:
    width: int
    view_start: float
    view_end: float
    rects: list[Rect] = field(default_factory=list)
    busy: np.ndarray = None           # per column: fraction busy with sub-pixel work (0..1)
    wait: np.ndarray = None           # per column: fraction in sub-pixel waits / cut-out waits
    count: np.ndarray = None          # per column: events too short to draw (exact or estimated)
    col_color: list = None            # per column: colour of the sub-pixel time that dominates it
    source: str = "exact"             # exact | index
    approx: bool = False
    index_bin_ns: float = 0.0
    events_in_view: int = 0
    observed: tuple[float, float] | None = None   # pixel range between the lane's first and last event
    note: str = ""

    def __post_init__(self) -> None:
        w = self.width
        if self.busy is None:
            self.busy = np.zeros(w)
        if self.wait is None:
            self.wait = np.zeros(w)
        if self.count is None:
            self.count = np.zeros(w, dtype=np.int64)
        if self.col_color is None:
            self.col_color = [None] * w

    @property
    def scale(self) -> float:
        return self.width / max(self.view_end - self.view_start, 1e-9)

    def x_of(self, t: float) -> float:
        return (t - self.view_start) * self.scale

    def hit(self, x: float, slack: float = 1.5) -> Rect | None:
        """The rectangle under pixel x (the innermost: last drawn), else the
        nearest within `slack` pixels."""
        best = None
        for r in self.rects:
            if r.x0 <= x <= max(r.x1, r.x0 + 1):
                best = r
        if best is not None:
            return best
        near = [r for r in self.rects if r.x0 - slack <= x <= r.x1 + slack]
        return min(near, key=lambda r: min(abs(r.x0 - x), abs(r.x1 - x))) if near else None

    def column_info(self, x: float) -> dict | None:
        c = int(x)
        if c < 0 or c >= self.width:
            return None
        if self.busy[c] <= 0 and self.wait[c] <= 0 and self.count[c] <= 0:
            return None
        ns_per_px = (self.view_end - self.view_start) / self.width
        return {"column": c, "busy": float(self.busy[c]), "wait": float(self.wait[c]),
                "count": int(self.count[c]), "startNs": self.view_start + c * ns_per_px,
                "endNs": self.view_start + (c + 1) * ns_per_px, "source": self.source,
                "approx": self.approx}


def _coverage(lo: np.ndarray, hi: np.ndarray, width: int) -> np.ndarray:
    """Covered length per unit column [c, c+1) of the union of the pixel
    intervals [lo, hi) (sorted by lo), in 0..1."""
    if not len(lo):
        return np.zeros(width)
    # union in one pass (intervals are start-sorted)
    ulo, uhi = [], []
    cur_lo, cur_hi = float(lo[0]), float(hi[0])
    for a, b in zip(lo[1:].tolist(), hi[1:].tolist()):
        if a <= cur_hi:
            if b > cur_hi:
                cur_hi = b
        else:
            ulo.append(cur_lo)
            uhi.append(cur_hi)
            cur_lo, cur_hi = a, b
    ulo.append(cur_lo)
    uhi.append(cur_hi)
    ulo_a, uhi_a = np.asarray(ulo), np.asarray(uhi)
    lengths = uhi_a - ulo_a
    cum = np.concatenate(([0.0], np.cumsum(lengths)))
    edges = np.arange(width + 1, dtype=np.float64)
    idx = np.searchsorted(ulo_a, edges, side="right") - 1
    ii = np.clip(idx, 0, None)
    full = np.where(idx >= 0, cum[ii], 0.0)
    part = np.where(idx >= 0, np.clip(edges - ulo_a[ii], 0, lengths[ii]), 0.0)
    c = full + part
    return np.clip(np.diff(c), 0.0, 1.0)


def _subtract(a: float, b: float, cuts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """[a, b) minus the sorted, disjoint intervals `cuts`."""
    out = []
    cur = a
    for c0, c1 in cuts:
        if c1 <= cur:
            continue
        if c0 >= b:
            break
        if c0 > cur:
            out.append((cur, min(c0, b)))
        cur = max(cur, c1)
        if cur >= b:
            break
    if cur < b:
        out.append((cur, b))
    return out


def _intersect(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Intersection of two sorted, disjoint interval lists."""
    out = []
    i = j = 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if hi > lo:
            out.append((lo, hi))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def _union(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def render_spans(spans: Iterable, view_start: float, view_end: float, width: int, *,
                 color_of: Callable, waits: list | None = None, wait_color: str = "",
                 extent: tuple[int, int] | None = None, limit: int = EXACT_LIMIT,
                 activity: Callable | None = None) -> LaneRender | None:
    """The exact render of a lane's spans (start order) overlapping the view,
    or None when there are more than `limit` of them (the caller uses the
    activity index then).

    `waits`: explicit waits of the thread's paired synchronization lane
    (spans) -- where they fall inside this lane's own spans (a barrier wait
    inside a parallel region) they are cut out of them and drawn as
    waiting; a span lying entirely inside one wait (a task run during a
    barrier) stays as is. Waits outside this lane's spans are not drawn
    here: they are on the sync lane itself.

    `activity`: activity_of, or a memoized equivalent."""
    activity_of_ = activity or activity_of
    width = max(1, int(width))
    span_ns = max(view_end - view_start, 1e-9)
    k = width / span_ns
    lr = LaneRender(width, view_start, view_end)
    if extent is not None:
        lr.observed = (max(0.0, (extent[0] - view_start) * k), min(float(width), (extent[1] - view_start) * k))
    wait_iv = _union(((max(w.start_ns, view_start), min(w.end_ns, view_end)) for w in (waits or ())
                      if w.end_ns > view_start and w.start_ns < view_end))
    sub_lo: list[float] = []
    sub_hi: list[float] = []
    subw_lo: list[float] = []
    subw_hi: list[float] = []
    col_time: dict[int, dict[str, float]] = {}
    own: list[tuple[float, float]] = []
    n = 0
    for s in spans:
        e = s.start_ns + max(s.duration_ns, 0)
        if e <= view_start and s.duration_ns > 0 or s.start_ns >= view_end:
            continue
        n += 1
        if n > limit:
            return None
        act = activity_of_(s)
        if wait_iv and act not in WAIT_LIKE and act != ANNOT and e > s.start_ns:
            own.append((max(s.start_ns, view_start), min(e, view_end)))
        x0 = (s.start_ns - view_start) * k
        x1 = (e - view_start) * k
        if x1 - x0 >= MIN_RECT_PX:
            color = color_of(s)
            inside = any(a <= s.start_ns and e <= b for a, b in wait_iv) if wait_iv and act not in WAIT_LIKE else False
            if wait_iv and act not in WAIT_LIKE and act != ANNOT and not inside:
                for a, b in _subtract(max(s.start_ns, view_start), min(e, view_end), wait_iv):
                    lr.rects.append(Rect(max(0.0, (a - view_start) * k), min(float(width), (b - view_start) * k),
                                         color, act, s.eid, s.start_ns, e, s.name))
            else:
                lr.rects.append(Rect(max(0.0, x0), min(float(width), x1), color, act, s.eid, s.start_ns, e,
                                     s.name, inside_wait=inside))
            continue
        # narrower than a pixel: its columns get the time, not a stretched bar
        c = min(width - 1, max(0, int((x0 + x1) / 2)))
        lr.count[c] += 1
        if act in WAIT_LIKE:
            subw_lo.append(max(0.0, x0))
            subw_hi.append(min(float(width), max(x1, x0)))
        else:
            sub_lo.append(max(0.0, x0))
            sub_hi.append(min(float(width), max(x1, x0)))
            ct = col_time.setdefault(c, {})
            color = color_of(s)
            ct[color] = ct.get(color, 0.0) + (x1 - x0)
    lr.events_in_view = n
    # sub-pixel time already shown by an exact rectangle (a short call
    # nested in a long one) is not density too: each instant once
    shown = _union((r.x0, r.x1) for r in lr.rects)

    def density(lo: list, hi: list) -> np.ndarray:
        pieces = []
        for a, b in sorted(zip(lo, hi)):
            pieces.extend(_subtract(a, b, shown) if shown else [(a, b)])
        if not pieces:
            return np.zeros(width)
        return _coverage(np.asarray([p[0] for p in pieces]), np.asarray([p[1] for p in pieces]), width)
    if sub_lo:
        lr.busy = density(sub_lo, sub_hi)
    if subw_lo:
        lr.wait = density(subw_lo, subw_hi)
    for c, times in col_time.items():
        lr.col_color[c] = max(times.items(), key=lambda kv: kv[1])[0]
    # waits of the paired lane: drawn as waiting where this lane is inside
    # one of its own spans (exact pieces), whatever their width
    for a, b in _intersect(wait_iv, _union(own)):
        wx0, wx1 = (a - view_start) * k, (b - view_start) * k
        if wx1 - wx0 >= MIN_RECT_PX:
            lr.rects.insert(0, Rect(max(0.0, wx0), min(float(width), wx1), wait_color, WAIT, -1, int(a), int(b),
                                    "", stripe=wait_color))
        else:
            c = min(width - 1, max(0, int((wx0 + wx1) / 2)))
            lr.wait[c] = min(1.0, lr.wait[c] + (wx1 - wx0))
    # draw order: waits first, then work, then work inside waits on top
    lr.rects.sort(key=lambda r: (0 if r.activity in WAIT_LIKE else (2 if r.inside_wait else 1)))
    return lr


def render_occupancy(occupancy: np.ndarray, starts: np.ndarray | None, view_start: float, view_end: float,
                     width: int, color: str, *, wait_occupancy: np.ndarray | None = None,
                     index_bin_ns: float = 0.0, extent: tuple[int, int] | None = None,
                     events_in_view: int = 0) -> LaneRender:
    """The render of a lane from activity-index occupancy (busy fraction per
    column) and, when the index resolves it, spans starting per column.
    `wait_occupancy`: the paired synchronization lane's occupancy, moved
    from busy to waiting."""
    width = max(1, int(width))
    lr = LaneRender(width, view_start, view_end, source="index")
    occ = np.clip(np.asarray(occupancy, dtype=np.float64)[:width], 0.0, 1.0)
    if len(occ) < width:
        occ = np.concatenate((occ, np.zeros(width - len(occ))))
    if wait_occupancy is not None:
        w = np.clip(np.asarray(wait_occupancy, dtype=np.float64)[:width], 0.0, 1.0)
        if len(w) < width:
            w = np.concatenate((w, np.zeros(width - len(w))))
        lr.wait = np.minimum(w, occ)
        lr.busy = np.clip(occ - lr.wait, 0.0, 1.0)
    else:
        lr.busy = occ
    if starts is not None:
        st = np.asarray(starts, dtype=np.int64)[:width]
        lr.count[:len(st)] = st
    lr.col_color = [color if b > 0 else None for b in lr.busy.tolist()]
    col_ns = (view_end - view_start) / width
    lr.index_bin_ns = index_bin_ns
    # uniform-within-a-bin error: a column spanning fewer than ~8 index bins
    lr.approx = index_bin_ns > 0 and col_ns < 8 * index_bin_ns
    lr.events_in_view = events_in_view
    if extent is not None:
        k = width / max(view_end - view_start, 1e-9)
        lr.observed = (max(0.0, (extent[0] - view_start) * k), min(float(width), (extent[1] - view_start) * k))
    lr.note = "density from the activity index" + (" (approximate at this zoom)" if lr.approx else "")
    return lr


def starts_per_column(level: dict, view_start: float, view_end: float, width: int) -> np.ndarray | None:
    """Spans starting per column from the finest activity level, when its
    bins are no wider than a column (else None: counts not resolvable)."""
    starts = level.get("starts")
    if starts is None:
        return None
    w = level["bin_ns"]
    col = (view_end - view_start) / max(width, 1)
    if w > col:
        return None
    t0 = level["t0"]
    edges = np.clip(((np.linspace(view_start, view_end, width + 1) - t0) / w).astype(np.int64), 0, len(starts))
    cum = np.concatenate(([0], np.cumsum(starts)))
    return np.diff(cum[edges])


def apparent_occupancy(lr: LaneRender) -> np.ndarray:
    """What the render SHOWS per column as a fraction of the column: the
    covered fraction of exact rectangles plus the density bars (busy +
    wait), capped at 1 -- for tests: honest rendering keeps this within
    rounding of the true occupancy at every zoom."""
    w = lr.width
    shown = np.zeros(w)
    if lr.rects:
        lo = np.asarray([r.x0 for r in lr.rects])
        hi = np.asarray([r.x1 for r in lr.rects])
        order = np.argsort(lo, kind="stable")
        shown = _coverage(lo[order], hi[order], w)
    return np.clip(shown + lr.busy + lr.wait, 0.0, 1.0)


def column_fractions(lr: LaneRender) -> tuple[np.ndarray, np.ndarray]:
    """(busy, wait) share of each column in a render, by activity class:
    work-like rectangles and density are busy, wait-like ones waiting,
    annotations neither. Index renders carry no class split beyond the
    paired sync lane's waits."""
    w = lr.width
    busy = lr.busy.copy()
    wait = lr.wait.copy()
    work = [(r.x0, r.x1) for r in lr.rects if r.activity not in WAIT_LIKE and r.activity != ANNOT]
    waits = [(r.x0, r.x1) for r in lr.rects if r.activity in WAIT_LIKE]
    for ivs, acc in ((work, busy), (waits, wait)):
        if ivs:
            ivs.sort()
            acc += _coverage(np.asarray([i[0] for i in ivs]), np.asarray([i[1] for i in ivs]), w)
    busy = np.clip(busy, 0.0, 1.0)
    wait = np.clip(wait, 0.0, 1.0)
    # a wait cut out of a span is not also busy
    return np.clip(busy - wait, 0.0, 1.0), np.minimum(wait, 1.0)


def aggregate_columns(children: list[tuple[np.ndarray, np.ndarray, str | None]]) -> tuple[np.ndarray, np.ndarray]:
    """A thread / process / device row from its members: each member gives
    (busy, wait, dominant colour) per column already aggregated for its own
    level. Returns (busy, wait) as the MEAN over members -- the share of the
    members busy / waiting in that column (a thread's member lanes are
    combined by max first, see TimelineModel.groupRender)."""
    if not children:
        return np.zeros(0), np.zeros(0)
    busy = np.mean([c[0] for c in children], axis=0)
    wait = np.mean([c[1] for c in children], axis=0)
    return busy, wait
