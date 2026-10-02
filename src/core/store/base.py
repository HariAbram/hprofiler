"""
TraceStore: where a Trace's events live.

Stores implement a small set of primitives -- append, ordered iteration,
lookup by id, in-place update/delete for post-processing, and metadata
blobs. Everything derived (display lanes, window queries, per-name
statistics, exclusive-time attribution, the multiresolution activity index,
persisted dependency edges) is implemented ONCE here on top of those
primitives; a store may override a derived method with a faster native
query (DiskTraceStore uses SQL), and the parity tests check that both give
identical results.

Iteration orders (the only orders any consumer relies on):
  "seq"    arrival order (global sequence number assigned at append)
  "start"  (start_ns, seq)
"""
from __future__ import annotations

import abc
import heapq
from typing import Any, Iterable, Iterator

import numpy as np

from ..events import AnyEvent, CounterEvent, InstantEvent, SpanEvent
from .common import (
    ACTIVITY_LEVELS, ActivityBuilder, ExclusiveAggregate, LaneInfo, SpanFilter,
    assign_lane_names, exclusive_thread_sweep, lane_base, occupancy_from_spans,
    resample_occupancy, span_bucket, _agg_key,
)


class TraceStore(abc.ABC):
    """See module docstring."""

    kind = "abstract"

    # ── writing ──────────────────────────────────────────────────────────
    @abc.abstractmethod
    def append(self, event: AnyEvent) -> int:
        """Store one event; assigns event.seq / event.eid. Returns eid."""

    def append_many(self, events: Iterable[AnyEvent]) -> None:
        for e in events:
            self.append(e)

    def flush(self) -> None:
        """Make every appended event visible to queries (no-op in memory)."""

    # ── post-processing mutation ─────────────────────────────────────────
    @abc.abstractmethod
    def update_span(self, span: SpanEvent) -> None:
        """Persist changes to a span previously returned by this store
        (matched by span.eid): tid, tags, stack frames, span ids,
        name/category, timing."""

    @abc.abstractmethod
    def delete_spans(self, eids: Iterable[int]) -> None:
        """Remove spans (post-processing de-duplication)."""

    # ── primitive reads ──────────────────────────────────────────────────
    @abc.abstractmethod
    def span_count(self) -> int: ...

    @abc.abstractmethod
    def instant_count(self) -> int: ...

    @abc.abstractmethod
    def counter_count(self) -> int: ...

    @abc.abstractmethod
    def iter_spans(self, *, order: str = "seq", pid: int | None = None, tid: int | None = None,
                   lane: str | None = None, categories: Iterable[str] | None = None,
                   window: tuple[int, int] | None = None, has_stack: bool | None = None,
                   gpu_model: bool | None = None, with_ids: bool | None = None,
                   filt: SpanFilter | None = None) -> Iterator[SpanEvent]:
        """Spans matching every given constraint. `window=(a, b)`: spans
        overlapping it (start <= b and end > a). `gpu_model`: CUDA/ROCm
        host/device-model spans (rt= tag) only / never. `with_ids`: spans
        carrying a span_id or parent_span_id only."""

    @abc.abstractmethod
    def iter_instants(self) -> Iterator[InstantEvent]: ...

    @abc.abstractmethod
    def iter_counters(self, *, order: str = "seq") -> Iterator[CounterEvent]:
        """order "seq", or "name" = (name, timestamp_ns, seq)."""

    @abc.abstractmethod
    def event_by_id(self, eid: int) -> AnyEvent | None: ...

    @abc.abstractmethod
    def span_by_seq(self, seq: int) -> SpanEvent | None: ...

    @abc.abstractmethod
    def span_seqs(self) -> np.ndarray:
        """Sequence numbers of all spans in seq order (the canonical node
        numbering for graph algorithms: ordinal = index into this)."""

    # ── metadata blobs ───────────────────────────────────────────────────
    @abc.abstractmethod
    def save_meta(self, key: str, value: Any) -> None: ...

    @abc.abstractmethod
    def load_meta(self, key: str, default: Any = None) -> Any: ...

    # ── derived: generic implementations ─────────────────────────────────
    def _cache(self) -> dict:
        """Per-store cache of derived results, cleared on any change."""
        c = getattr(self, "_derived", None)
        if c is None:
            c = self._derived = {}
        return c

    def invalidate(self) -> None:
        self._derived = {}

    def memo(self, key: Any, compute) -> Any:
        """compute() once per store content (cleared with the other derived
        results whenever events change)."""
        c = self._cache()
        k = ("memo", key)
        if k not in c:
            c[k] = compute()
        return c[k]

    # ── lightweight scans (no full SpanEvent per row) ─────────────────────
    def iter_intervals(self, *, pid: int | None = None, categories: Iterable[str] | None = None,
                       exclude_types: Iterable[str] | None = None,
                       timed_only: bool = True) -> Iterator[tuple[int, int]]:
        """(start_ns, end_ns) of the selected spans in start order --
        skipping spans whose `type` tag is in `exclude_types`, and
        zero-duration spans when `timed_only`."""
        excl = frozenset(exclude_types or ())
        for s in self.iter_spans(order="start", pid=pid, categories=categories):
            if timed_only and s.duration_ns <= 0:
                continue
            if excl and s.tags.get("type") in excl:
                continue
            yield s.start_ns, s.end_ns

    def iter_spans_light(self, *, pid: int | None = None, tid: int | None = None) -> Iterator[SpanEvent]:
        """iter_spans(pid=, tid=) in arrival order for readers of timing,
        name, category and span ids only: stores may leave tags and stack
        frames empty (and eid -1, so such a span can never be written back
        over the stored one)."""
        return self.iter_spans(pid=pid, tid=tid)

    def interval_arrays(self, *, categories: Iterable[str] | None = None,
                        chunk: int = 65536) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """(start_ns, end_ns) arrays of the selected spans in bounded
        chunks, in no particular order (for order-independent binning)."""
        starts: list[int] = []
        ends: list[int] = []
        for s in self.iter_spans(categories=categories):
            starts.append(s.start_ns)
            ends.append(s.end_ns)
            if len(starts) >= chunk:
                yield np.asarray(starts, dtype=np.int64), np.asarray(ends, dtype=np.int64)
                starts, ends = [], []
        if starts:
            yield np.asarray(starts, dtype=np.int64), np.asarray(ends, dtype=np.int64)

    def first_with_tag(self, key: str) -> dict[str, SpanEvent]:
        """{name: the first span (arrival order) of that name whose tag
        `key` is set (truthy)}."""
        c = self._cache()
        if ("first_with_tag", key) not in c:
            out: dict[str, SpanEvent] = {}
            for s in self.iter_spans():
                if s.name not in out and s.tags.get(key):
                    out[s.name] = s
            c[("first_with_tag", key)] = out
        return c[("first_with_tag", key)]

    def pids(self) -> list[int]:
        c = self._cache()
        if "pids" not in c:
            c["pids"] = sorted({s.pid for s in self.iter_spans()})
        return c["pids"]

    def threads(self) -> list[tuple[int, int]]:
        c = self._cache()
        if "threads" not in c:
            c["threads"] = sorted({(s.pid, s.tid) for s in self.iter_spans()})
        return c["threads"]

    def span_extent(self, *, timed_only: bool = True) -> tuple[int, int] | None:
        """(min start, max end) over spans (with duration > 0 by default)."""
        key = ("extent", timed_only)
        c = self._cache()
        if key not in c:
            lo = hi = None
            for s in self.iter_spans():
                if timed_only and s.duration_ns <= 0:
                    continue
                lo = s.start_ns if lo is None or s.start_ns < lo else lo
                hi = s.end_ns if hi is None or s.end_ns > hi else hi
            c[key] = None if lo is None else (lo, hi)
        return c[key]

    def event_extent(self) -> tuple[int, int] | None:
        """(min, max) over spans (incl. zero-duration), instants, counters."""
        c = self._cache()
        if "event_extent" not in c:
            lo = hi = None
            for t0, t1 in self._event_times():
                lo = t0 if lo is None or t0 < lo else lo
                hi = t1 if hi is None or t1 > hi else hi
            c["event_extent"] = None if lo is None else (lo, hi)
        return c["event_extent"]

    def _event_times(self) -> Iterator[tuple[int, int]]:
        for s in self.iter_spans():
            yield s.start_ns, s.end_ns
        for i in self.iter_instants():
            yield i.timestamp_ns, i.timestamp_ns
        for k in self.iter_counters():
            yield k.timestamp_ns, k.timestamp_ns

    def lane_infos(self) -> list[LaneInfo]:
        """Display lanes in order of first appearance (see Trace.lanes)."""
        c = self._cache()
        if "lanes" not in c:
            groups: dict[tuple, dict] = {}
            for s in self.iter_spans():
                key = (s.pid, lane_base(s))
                g = groups.get(key)
                if g is None:
                    groups[key] = {"pid": s.pid, "base": key[1], "count": 1, "first_seq": s.seq,
                                   "min_start": s.start_ns, "max_end": s.end_ns,
                                   "max_dur": s.duration_ns}
                else:
                    g["count"] += 1
                    g["min_start"] = min(g["min_start"], s.start_ns)
                    g["max_end"] = max(g["max_end"], s.end_ns)
                    g["max_dur"] = max(g["max_dur"], s.duration_ns)
            c["lanes"] = assign_lane_names(list(groups.values()))
        return c["lanes"]

    def lane(self, name: str) -> LaneInfo | None:
        c = self._cache()
        if "lane_by_name" not in c:
            c["lane_by_name"] = {ln.name: ln for ln in self.lane_infos()}
        return c["lane_by_name"].get(name)

    def window(self, lane: str, start: int, end: int, *, filt: SpanFilter | None = None,
               limit: int | None = None) -> list[SpanEvent]:
        """Spans of one lane overlapping [start, end] in start order."""
        out = []
        for s in self.iter_spans(order="start", lane=lane, window=(start, end), filt=filt):
            out.append(s)
            if limit is not None and len(out) >= limit:
                break
        return out

    def window_columns(self, lane: str, start: int, end: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """window() as columns -- (start_ns, end_ns, name) in start order --
        for renderers that only need geometry and a color key."""
        starts, ends, names = [], [], []
        for s in self.iter_spans(order="start", lane=lane, window=(start, end)):
            starts.append(s.start_ns)
            ends.append(s.end_ns)
            names.append(s.name)
        return np.asarray(starts, dtype=np.int64), np.asarray(ends, dtype=np.int64), names

    def span_by_span_id(self, span_id: str) -> SpanEvent | None:
        """The span carrying `span_id` (the latest one if several do)."""
        c = self._cache()
        if "by_span_id" not in c:
            c["by_span_id"] = {s.span_id: s.eid for s in self.iter_spans(with_ids=True) if s.span_id}
        eid = c["by_span_id"].get(span_id)
        return None if eid is None else self.event_by_id(eid)

    def count_window(self, lane: str, start: int, end: int, *, filt: SpanFilter | None = None,
                     cap: int | None = None) -> int:
        n = 0
        for _ in self.iter_spans(order="start", lane=lane, window=(start, end), filt=filt):
            n += 1
            if cap is not None and n > cap:
                break
        return n

    def lane_count(self, lane: str, filt: SpanFilter | None = None) -> int:
        info = self.lane(lane)
        if info is None:
            return 0
        if filt is None or filt.is_empty():
            return info.count
        return sum(1 for _ in self.iter_spans(lane=lane, filt=filt))

    def aggregate_stats(self) -> list[dict]:
        """Per (category, name): count / total / min / max / avg / pct of
        summed durations; sorted by total (ties: first appearance)."""
        c = self._cache()
        if "agg" not in c:
            totals: dict[tuple[str, str], dict] = {}
            for s in self.iter_spans():
                key = (s.category.value, s.name)
                r = totals.get(key)
                d = s.duration_ns
                if r is None:
                    totals[key] = {"name": s.name, "category": key[0], "count": 1, "total_ns": d,
                                   "min_ns": d, "max_ns": d}
                else:
                    r["count"] += 1
                    r["total_ns"] += d
                    if d < r["min_ns"]:
                        r["min_ns"] = d
                    if d > r["max_ns"]:
                        r["max_ns"] = d
            c["agg"] = finish_aggregate(list(totals.values()))
        return [dict(r) for r in c["agg"]]

    def exclusive_aggregate(self) -> ExclusiveAggregate:
        """activity_buckets.ExclusiveTime, computed per thread from
        time-ordered iterators and kept as an aggregate (see common.py)."""
        c = self._cache()
        if "excl" not in c:
            agg = self._compute_exclusive()
            agg.rows = dict(sorted(agg.rows.items()))   # same iteration order in every store
            c["excl"] = agg
        return c["excl"]

    def _compute_exclusive(self) -> ExclusiveAggregate:
        from .common import FLAG_DEVICE
        agg = ExclusiveAggregate()
        host_threads: set[tuple[int, int]] = set()
        for s in self.iter_spans():
            if s.duration_ns <= 0:
                continue
            b = span_bucket(s)
            if b == "Annotation":
                continue
            if _is_device(s):
                agg.add(_agg_key(s, device=True), s.duration_ns)
            else:
                host_threads.add((s.pid, s.tid))
        agg.threads = host_threads
        for pid, tid in sorted(host_threads):
            exclusive_thread_sweep(
                (s for s in self.iter_spans(order="start", pid=pid, tid=tid)
                 if s.duration_ns > 0 and not _is_device(s) and span_bucket(s) != "Annotation"),
                agg)
        return agg

    def activity(self, lane: str) -> list[dict] | None:
        """Multiresolution occupancy levels of one lane (see common.py),
        built lazily (and persisted by finalize() in disk stores)."""
        c = self._cache()
        key = ("activity", lane)
        if key not in c:
            c[key] = self._build_activity(lane)
        return c[key]

    def activity_grid(self) -> tuple[int, int] | None:
        """(t0, t1) the activity bins span: the timed-span extent."""
        return self.span_extent(timed_only=True) or self.span_extent(timed_only=False)

    def _build_activity(self, lane: str) -> list[dict] | None:
        grid = self.activity_grid()
        if grid is None or self.lane(lane) is None:
            return None
        b = ActivityBuilder(grid[0], grid[1])
        for s in self.iter_spans(order="start", lane=lane):
            b.add(s.start_ns, s.end_ns)
        levels = b.finish()
        for lv in levels:
            lv["t0"] = grid[0]
        levels[0]["starts"] = b.starts
        return levels

    def occupancy(self, lane: str, start: float, end: float, n_bins: int) -> np.ndarray | None:
        """Occupancy fraction per output bin from the activity index, or
        None when the window is finer than the finest precomputed level
        (caller then bins exact spans)."""
        if n_bins <= 0 or end <= start:
            return None
        levels = self.activity(lane)
        if not levels:
            return None
        out_w = (end - start) / n_bins
        usable = [lv for lv in levels if lv["bin_ns"] <= out_w]
        if not usable:
            return None
        lv = max(usable, key=lambda lv: lv["bin_ns"])   # coarsest that still resolves a bin
        return resample_occupancy(lv, lv["t0"], start, end, n_bins)

    def estimate_starts(self, lane: str, start: float, end: float) -> int | None:
        """Approximate number of spans starting in [start, end) from the
        finest activity level (exact to one bin at each edge)."""
        levels = self.activity(lane)
        if not levels or "starts" not in levels[0]:
            return None
        lv = levels[0]
        starts = lv["starts"]
        w = lv["bin_ns"]
        i0 = int(max(0, (start - lv["t0"]) // w))
        i1 = int(min(len(starts), (end - lv["t0"]) // w + 1))
        return int(starts[i0:i1].sum()) if i1 > i0 else 0

    def lanes_in_window(self, start: int, end: int, *, filt: SpanFilter | None = None,
                        limit_per_lane: int | None = None) -> dict[str, list[SpanEvent]]:
        out: dict[str, list[SpanEvent]] = {}
        for ln in self.lane_infos():
            if ln.max_end <= start or ln.min_start > end:
                continue
            spans = self.window(ln.name, start, end, filt=filt, limit=limit_per_lane)
            if spans:
                out[ln.name] = spans
        return out

    def events_in_window(self, start: int, end: int, *, lanes: Iterable[str] | None = None,
                         filt: SpanFilter | None = None) -> Iterator[SpanEvent]:
        """Spans of the given lanes (default: all) overlapping the window,
        merged across lanes in (start, seq) order."""
        names = list(lanes) if lanes is not None else [ln.name for ln in self.lane_infos()]
        iters = [self.iter_spans(order="start", lane=n, window=(start, end), filt=filt) for n in names]
        return heapq.merge(*iters, key=lambda s: (s.start_ns, s.seq))

    def occupancy_exact(self, lane: str, start: float, end: float, n_bins: int,
                        filt: SpanFilter | None = None) -> np.ndarray:
        return occupancy_from_spans(
            ((s.start_ns, s.end_ns) for s in self.iter_spans(order="start", lane=lane,
                                                            window=(int(start), int(end)), filt=filt)),
            start, end, n_bins)

    def finalize(self) -> None:
        """Build everything derived that a viewer needs (lanes, activity
        index, aggregates, exclusive time). Disk stores also build their
        indexes and persist the results."""
        self.flush()
        self.lane_infos()
        self.aggregate_stats()
        self.exclusive_aggregate()
        for ln in self.lane_infos():
            self.activity(ln.name)

    # ── dependency edges (critical path) ─────────────────────────────────
    def save_edges(self, version: str, edges: np.ndarray) -> None:
        """edges: structured array (dst, src, kind, conf) in insertion order."""
        self._cache()[("edges", version)] = edges

    def load_edges(self, version: str, kinds: Iterable[int] | None = None) -> np.ndarray | None:
        """Persisted edges of `version` (None if absent or stale), only
        those whose kind code is in `kinds` when given."""
        edges = self._cache().get(("edges", version))
        if edges is not None and kinds is not None:
            edges = edges[np.isin(edges["kind"], list(kinds))]
        return edges

    def close(self) -> None:
        pass


def _is_device(s: SpanEvent) -> bool:
    from ...analysis.activity_buckets import is_device_timed
    return is_device_timed(s)


def finish_aggregate(rows: list[dict]) -> list[dict]:
    for r in rows:
        r["avg_ns"] = r["total_ns"] / r["count"] if r["count"] else 0
    total_time = sum(r["total_ns"] for r in rows) or 1
    for r in rows:
        r["pct"] = 100.0 * r["total_ns"] / total_time
    return sorted(rows, key=lambda r: r["total_ns"], reverse=True)
