"""
Store-independent pieces of the trace store: the lane rule, small integer
codes, filters, and the streaming algorithms every store shares (union
coverage, the multiresolution activity index, per-thread exclusive-time
attribution). Each store only has to provide ordered iterators and a few
lookups; everything computed from them lives here, so a memory-backed and a
disk-backed trace produce the same numbers by construction.
"""
from __future__ import annotations

import heapq
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator

import numpy as np

from ..events import Category, SpanEvent

# Integer codes persisted in disk stores. The mapping is ALSO written into
# each store's catalog (tables `categories` / `buckets`), and readers decode
# through that stored mapping, so appending a new Category or bucket later
# never reinterprets old stores.
CATEGORY_CODES: dict[str, int] = {c.value: i for i, c in enumerate(Category)}
CATEGORY_BY_CODE: list[str] = [c.value for c in Category]

# Activity buckets (src/analysis/activity_buckets.py), same order.
BUCKET_NAMES: tuple[str, ...] = (
    "Computation", "Communication", "Synchronization", "Memory transfer",
    "Runtime overhead", "Idle", "Annotation", "Other",
)
BUCKET_CODES: dict[str, int] = {b: i for i, b in enumerate(BUCKET_NAMES)}

# Span flag bits (column `flags`).
FLAG_DEVICE = 1        # activity_buckets.is_device_timed
FLAG_STACK = 2         # has stack frames
FLAG_GPU_MODEL = 4     # CUDA/ROCm host/device model span (rt= tag)
FLAG_SAMPLED = 8       # perf sample (category cpu)

# Event ids: (shard << 38) | (kind << 36) | row -- below 2**53, so they
# survive a round trip through a QML/JavaScript number.
KIND_SPAN, KIND_INSTANT, KIND_COUNTER = 0, 1, 2
_ROW_BITS, _KIND_BITS = 36, 2
_ROW_MASK = (1 << _ROW_BITS) - 1
MAX_SHARDS = 1 << 15


def make_eid(shard: int, kind: int, row: int) -> int:
    return (shard << (_ROW_BITS + _KIND_BITS)) | (kind << _ROW_BITS) | row


def split_eid(eid: int) -> tuple[int, int, int]:
    return eid >> (_ROW_BITS + _KIND_BITS), (eid >> _ROW_BITS) & 3, eid & _ROW_MASK


# Persisted dependency edges: (dst ordinal, src ordinal, kind code,
# confidence code), in insertion order.
EDGE_DTYPE = np.dtype([("dst", np.int64), ("src", np.int64), ("kind", np.int8), ("conf", np.int8)])

# Tags copied into indexed columns (the full tag dict is stored as well).
INDEXED_TAGS = ("stream", "corr", "lid", "type", "side")


# ── lanes ────────────────────────────────────────────────────────────────────

def lane_base(span: SpanEvent) -> tuple[str, str, str]:
    """(category, kind, ident) of the display lane a span belongs to --
    the rule documented on Trace.lanes(). `kind` is thread / stream /
    device / "" and ident a string ("" when the kind has none)."""
    cat = span.category.value
    tags = span.tags
    side = tags.get("side")
    if side == "cpu":
        return (cat, "thread", str(span.tid)) if span.tid else (cat, "", "")
    if "stream" in tags and (side == "gpu" or cat in ("cuda", "rocm")
                             or tags.get("type") == "memcpy_async"):
        return (cat, "stream", str(tags["stream"]))
    if side == "gpu" or span.name.endswith("_gpu"):
        return (cat, "device", "")
    if span.tid:
        return (cat, "thread", str(span.tid))
    return (cat, "", "")


def lane_name(base: tuple[str, str, str], pid: int, shared: bool) -> str:
    cat, kind, ident = base
    if kind in ("stream", "thread"):
        name = f"{cat}/{kind}-{ident}"
    elif kind == "device":
        name = f"{cat}/device"
    else:
        name = cat
    if shared:
        name += f"@{pid}"
    return name


@dataclass
class LaneInfo:
    """One display lane. Every lane belongs to exactly one process: a base
    key used by several processes gets an @PID suffix per process."""
    name: str
    cat: str
    kind: str
    ident: str
    pid: int
    count: int
    first_seq: int
    min_start: int
    max_end: int
    max_dur: int
    lane_key: int = field(default=-1, compare=False)   # store-internal key id


def assign_lane_names(groups: list[dict]) -> list[LaneInfo]:
    """groups: one dict per (pid, base) with count/first_seq/min_start/
    max_end/max_dur. Returns lanes ordered by first appearance."""
    pids_per_base: dict[tuple, set[int]] = {}
    for g in groups:
        pids_per_base.setdefault(g["base"], set()).add(g["pid"])
    lanes = []
    for g in sorted(groups, key=lambda g: g["first_seq"]):
        base = g["base"]
        lanes.append(LaneInfo(
            name=lane_name(base, g["pid"], len(pids_per_base[base]) > 1),
            cat=base[0], kind=base[1], ident=base[2], pid=g["pid"],
            count=g["count"], first_seq=g["first_seq"], min_start=g["min_start"],
            max_end=g["max_end"], max_dur=g["max_dur"], lane_key=g.get("lane_key", -1),
        ))
    return lanes


def span_bucket(span: SpanEvent) -> str:
    from ...analysis.activity_buckets import bucket_of_span
    return bucket_of_span(span)


def span_flags(span: SpanEvent) -> int:
    from ...analysis.activity_buckets import is_device_timed
    f = 0
    if is_device_timed(span):
        f |= FLAG_DEVICE
    if span.stack_frames:
        f |= FLAG_STACK
    if span.tags.get("rt") in ("cuda", "rocm"):
        f |= FLAG_GPU_MODEL
    if span.category.value == "cpu":
        f |= FLAG_SAMPLED
    return f


# ── filters ──────────────────────────────────────────────────────────────────

@dataclass
class SpanFilter:
    """Event-level filter for window/count queries. Empty = no constraint."""
    name_query: str = ""
    name_is_regex: bool = False
    min_dur_ns: int = 0
    buckets: frozenset[str] = frozenset()
    categories: frozenset[str] = frozenset()
    names: frozenset[str] = frozenset()     # exact names
    # Overlapping [range_start_ns, range_end_ns) (both set) -- "time range only".
    range_start_ns: int | None = None
    range_end_ns: int | None = None

    def is_empty(self) -> bool:
        return not (self.name_query or self.min_dur_ns > 0 or self.buckets
                    or self.categories or self.names or self.range_start_ns is not None)

    def name_matcher(self):
        """None when names are unconstrained, else predicate(name)."""
        if not self.name_query and not self.names:
            return None
        rx = None
        if self.name_query and self.name_is_regex:
            try:
                rx = re.compile(self.name_query, re.IGNORECASE)
            except re.error:
                rx = None
        q = self.name_query.lower()
        exact = self.names

        def match(name: str) -> bool:
            if exact and name not in exact:
                return False
            if rx is not None:
                return bool(rx.search(name))
            # plain substring -- also for a regex that doesn't compile
            # (matched literally, as the Timeline filter always did)
            return not q or q in name.lower()
        return match

    def matches(self, span: SpanEvent) -> bool:
        if self.min_dur_ns > 0 and span.duration_ns < self.min_dur_ns:
            return False
        if self.range_start_ns is not None and not (
                span.end_ns > self.range_start_ns and span.start_ns < self.range_end_ns):
            return False
        if self.categories and span.category.value not in self.categories:
            return False
        if self.buckets and span_bucket(span) not in self.buckets:
            return False
        m = self.name_matcher()
        return m is None or m(span.name)


# ── streaming interval algorithms ────────────────────────────────────────────

def union_length(intervals: Iterable[tuple[int, int]]) -> int:
    """Length of the union of intervals given in start order (streaming,
    O(1) state)."""
    total = 0
    cur_lo = cur_hi = None
    for lo, hi in intervals:
        if hi <= lo:
            continue
        if cur_hi is None or lo > cur_hi:
            if cur_hi is not None:
                total += cur_hi - cur_lo
            cur_lo, cur_hi = lo, hi
        elif hi > cur_hi:
            cur_hi = hi
    if cur_hi is not None:
        total += cur_hi - cur_lo
    return total


def union_intervals(intervals: Iterable[tuple[int, int]]) -> Iterator[tuple[int, int]]:
    """Disjoint union of intervals given in start order (streaming)."""
    cur_lo = cur_hi = None
    for lo, hi in intervals:
        if hi <= lo:
            continue
        if cur_hi is None or lo > cur_hi:
            if cur_hi is not None:
                yield cur_lo, cur_hi
            cur_lo, cur_hi = lo, hi
        elif hi > cur_hi:
            cur_hi = hi
    if cur_hi is not None:
        yield cur_lo, cur_hi


# ── multiresolution activity index ───────────────────────────────────────────

# Bins across the whole trace extent, finest first. Each level is the next
# finer one summed in groups of 4, so a view at any zoom reads at most a few
# thousand bins.
ACTIVITY_LEVELS = (65536, 16384, 4096, 1024, 256)


def _coverage_at(edges: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """C(e) = covered time of the disjoint, sorted intervals [lo, hi) that
    lies before each edge e."""
    lengths = hi - lo
    cum = np.concatenate(([0.0], np.cumsum(lengths)))
    idx = np.searchsorted(lo, edges, side="right") - 1
    full = cum[np.clip(idx, 0, None)]           # intervals strictly before idx
    full = np.where(idx >= 0, full, 0.0)
    part = np.where(idx >= 0, np.clip(edges - lo[np.clip(idx, 0, None)], 0, lengths[np.clip(idx, 0, None)]), 0.0)
    return full + part


class ActivityBuilder:
    """Builds one lane's occupancy bins from its spans streamed in start
    order: busy time per bin of the lane's UNION of spans (nested or
    overlapping spans count once), plus how many spans start in each bin.
    State is the bin arrays plus a bounded batch of union intervals."""

    def __init__(self, t0: int, t1: int, nbins: int = ACTIVITY_LEVELS[0]):
        self.t0 = int(t0)
        self.span_ns = max(int(t1) - int(t0), 1)
        self.nbins = nbins
        self.width = self.span_ns / nbins
        self.busy = np.zeros(nbins, dtype=np.float64)
        self.starts = np.zeros(nbins, dtype=np.int64)
        self._lo: list[float] = []
        self._hi: list[float] = []
        self._cur: list[float] | None = None
        self._start_buf: list[int] = []

    def add(self, start: int, end: int) -> None:
        self._start_buf.append(start)
        if len(self._start_buf) >= 65536:
            self._flush_starts()
        if end <= start:
            return
        lo, hi = float(start - self.t0), float(end - self.t0)
        cur = self._cur
        if cur is None or lo > cur[1]:
            if cur is not None:
                self._lo.append(cur[0])
                self._hi.append(cur[1])
                if len(self._lo) >= 65536:
                    self._flush_union()
            self._cur = [lo, hi]
        elif hi > cur[1]:
            cur[1] = hi

    def _flush_starts(self) -> None:
        if self._start_buf:
            rel = (np.asarray(self._start_buf, dtype=np.float64) - self.t0) / self.width
            b = np.clip(rel.astype(np.int64), 0, self.nbins - 1)
            self.starts += np.bincount(b, minlength=self.nbins)
            self._start_buf = []

    def _flush_union(self) -> None:
        if not self._lo:
            return
        lo = np.asarray(self._lo)
        hi = np.asarray(self._hi)
        self._lo, self._hi = [], []
        b0 = max(int(lo[0] // self.width), 0)
        b1 = min(int(np.ceil(hi[-1] / self.width)), self.nbins)
        if b1 <= b0:
            b1 = min(b0 + 1, self.nbins)
            b0 = b1 - 1
        edges = np.arange(b0, b1 + 1, dtype=np.float64) * self.width
        c = _coverage_at(edges, lo, hi)
        self.busy[b0:b1] += np.diff(c)

    def finish(self) -> list[dict]:
        if self._cur is not None:
            self._lo.append(self._cur[0])
            self._hi.append(self._cur[1])
            self._cur = None
        self._flush_union()
        self._flush_starts()
        levels = []
        busy = self.busy
        nb = self.nbins
        for n in ACTIVITY_LEVELS:
            if n > nb:
                continue
            factor = nb // n
            b = busy.reshape(n, factor).sum(axis=1) if factor > 1 else busy
            levels.append({"nbins": n, "bin_ns": self.span_ns / n, "busy": b.astype(np.float32)})
        return levels


def resample_occupancy(level: dict, t0: int, start: float, end: float, n_out: int) -> np.ndarray:
    """Occupancy fraction (0..1) of n_out equal bins across [start, end),
    from one precomputed level (busy ns per source bin; busy assumed uniform
    within a source bin)."""
    busy = np.asarray(level["busy"], dtype=np.float64)
    w = level["bin_ns"]
    cum = np.concatenate(([0.0], np.cumsum(busy)))
    edges = np.linspace(start - t0, end - t0, n_out + 1)
    pos = np.clip(edges / w, 0, len(busy))
    i = np.clip(pos.astype(np.int64), 0, len(busy) - 1)
    frac = pos - i
    c = cum[i] + frac * busy[i]
    c = np.where(pos >= len(busy), cum[-1], c)
    out_w = (end - start) / n_out
    return np.clip(np.diff(c) / out_w, 0.0, 1.0)


def occupancy_from_spans(spans: Iterable[tuple[int, int]], start: float, end: float,
                         n_out: int) -> np.ndarray:
    """Exact occupancy bins straight from spans (start order) -- for windows
    finer than the precomputed index, or with filters applied."""
    width = (end - start) / n_out
    lo_l, hi_l = [], []
    for a, b in union_intervals((max(a, start), min(b, end)) for a, b in spans if b > start and a < end):
        lo_l.append(a - start)
        hi_l.append(b - start)
    if not lo_l:
        return np.zeros(n_out)
    edges = np.arange(n_out + 1, dtype=np.float64) * width
    c = _coverage_at(edges, np.asarray(lo_l, dtype=np.float64), np.asarray(hi_l, dtype=np.float64))
    return np.clip(np.diff(c) / width, 0.0, 1.0)


# ── exclusive-time attribution (streaming, per thread) ───────────────────────

@dataclass
class ExclusiveAggregate:
    """Exclusive host time and device time grouped by (pid, tid, category,
    name, bucket, device) -- fine enough for every breakdown the UIs show
    (bucket, category, name, per-thread sync). Built per thread by
    exclusive_rows(); see activity_buckets.ExclusiveTime for the rule."""
    rows: dict[tuple[int, int, str, str, str, bool], int] = field(default_factory=dict)
    threads: set[tuple[int, int]] = field(default_factory=set)

    def add(self, key: tuple, ns: int) -> None:
        if ns > 0:
            self.rows[key] = self.rows.get(key, 0) + ns

    def totals(self, keyfn, *, include_device: bool = True) -> dict:
        out: dict = {}
        for (pid, tid, cat, name, bucket, device), ns in self.rows.items():
            if device and not include_device:
                continue
            k = keyfn(pid, tid, cat, name, bucket, device)
            out[k] = out.get(k, 0) + ns
        return out

    def bucket_totals(self) -> dict[str, int]:
        return self.totals(lambda pid, tid, cat, name, bucket, device: bucket)


def exclusive_thread_sweep(spans: Iterable[SpanEvent], agg: ExclusiveAggregate) -> None:
    """Attribute one (pid, tid)'s exclusive time. `spans` are that thread's
    host spans (not device-timed, not Annotation, duration > 0) in
    (start_ns, seq) order. Identical to activity_buckets.ExclusiveTime:
    every instant belongs to the active span with the latest start (ties:
    earliest end, then arrival order); perf samples (category cpu) keep
    their nominal weight only where no instrumented span covers their
    start. State: the currently open spans only."""
    pending_ends: list[tuple[int, int, int]] = []      # (end, seq, slot)
    active: list[tuple[int, int, int, int]] = []        # (-start, end, seq, slot)
    ended: set[int] = set()
    keys: dict[int, tuple] = {}
    prev_t: int | None = None
    max_end = None          # latest end of any instrumented span started so far
    pending_samples: list[SpanEvent] = []
    sample_t = None

    def owner_add(t: int) -> None:
        nonlocal prev_t
        if prev_t is not None and t > prev_t:
            while active and active[0][3] in ended:
                ended.discard(heapq.heappop(active)[3])
            if active:
                agg.add(keys[active[0][3]], t - prev_t)
        prev_t = t

    def drain_ends(before: tuple) -> None:
        # process ends whose (time, 0, seq) sorts before `before`
        while pending_ends and (pending_ends[0][0], 0, pending_ends[0][1]) < before:
            end, seq, slot = heapq.heappop(pending_ends)
            owner_add(end)
            ended.add(slot)

    def settle_samples() -> None:
        for smp in pending_samples:
            if max_end is not None and max_end > smp.start_ns:
                continue
            agg.add(_agg_key(smp, device=False), smp.duration_ns)
        pending_samples.clear()

    slot_counter = 0
    for s in spans:
        if s.category.value == "cpu":
            if sample_t is not None and s.start_ns != sample_t:
                settle_samples()
            sample_t = s.start_ns
            pending_samples.append(s)
            continue
        if pending_samples and s.start_ns > sample_t:
            settle_samples()
        start, end = s.start_ns, s.start_ns + s.duration_ns
        drain_ends((start, 1, s.seq))
        owner_add(start)
        slot = slot_counter
        slot_counter += 1
        keys[slot] = _agg_key(s, device=False)
        heapq.heappush(active, (-start, end, s.seq, slot))
        heapq.heappush(pending_ends, (end, s.seq, slot))
        if max_end is None or end > max_end:
            max_end = end
        # drop keys of spans no longer referenced (bounded state)
        if len(keys) > 4096 and len(keys) > 4 * (len(active) + len(pending_ends)):
            live = {a[3] for a in active} | {p[2] for p in pending_ends}
            keys = {k: v for k, v in keys.items() if k in live}
    while pending_ends:
        end, seq, slot = heapq.heappop(pending_ends)
        owner_add(end)
        ended.add(slot)
    settle_samples()


def _agg_key(s, *, device: bool) -> tuple:
    bucket = getattr(s, "bucket", None) or span_bucket(s)
    return (s.pid, s.tid, s.category.value, s.name, bucket, device)


class LiteSpan:
    """The fields exclusive_thread_sweep reads, without tags/stack -- what
    a store can produce from index columns alone during finalization."""
    __slots__ = ("start_ns", "duration_ns", "seq", "pid", "tid", "name", "category", "bucket")

    def __init__(self, start_ns, end_ns, seq, pid, tid, name, category, bucket):
        self.start_ns = start_ns
        self.duration_ns = end_ns - start_ns
        self.seq = seq
        self.pid = pid
        self.tid = tid
        self.name = name
        self.category = category
        self.bucket = bucket
