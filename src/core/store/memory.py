"""
MemoryTraceStore: every event as a Python object in arrival order -- the
behaviour Trace always had. For small traces and unit tests; the GUI/CLI
use DiskTraceStore for captures (see src/core/store/disk.py).
"""
from __future__ import annotations

import copy
import threading
from typing import Any, Iterable, Iterator

import numpy as np

from ..events import AnyEvent, CounterEvent, InstantEvent, SpanEvent
from .base import TraceStore
from .common import (
    KIND_COUNTER, KIND_INSTANT, KIND_SPAN, SpanFilter, lane_base, make_eid, split_eid,
)


class MemoryTraceStore(TraceStore):
    kind = "memory"

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._spans: list[SpanEvent] = []
        self._instants: list[InstantEvent] = []
        self._counters: list[CounterEvent] = []
        self._by_seq: dict[int, AnyEvent] = {}
        self._next_seq = 0
        self._meta: dict[str, Any] = {}
        self._derived = {}

    # ── writing ──────────────────────────────────────────────────────────
    def append(self, event: AnyEvent) -> int:
        if getattr(event, "eid", -1) != -1:
            # already held by another store: never share one object
            event = copy.copy(event)
            if isinstance(event, (SpanEvent, InstantEvent)):
                event.tags = dict(event.tags)
        with self._lock:
            seq = self._next_seq
            self._next_seq += 1
            event.seq = seq
            if isinstance(event, SpanEvent):
                event.eid = make_eid(0, KIND_SPAN, seq)
                self._spans.append(event)
            elif isinstance(event, InstantEvent):
                event.eid = make_eid(0, KIND_INSTANT, seq)
                self._instants.append(event)
            elif isinstance(event, CounterEvent):
                event.eid = make_eid(0, KIND_COUNTER, seq)
                self._counters.append(event)
            else:
                raise TypeError(f"not an event: {event!r}")
            self._by_seq[seq] = event
            self._derived = {}
            return event.eid

    def update_span(self, span: SpanEvent) -> None:
        with self._lock:
            cur = self._by_seq.get(span.seq)
            if cur is not None and cur is not span:
                idx = self._index_of(cur)
                self._spans[idx] = span
                self._by_seq[span.seq] = span
            self._derived = {}

    def _index_of(self, span: SpanEvent) -> int:
        for i, s in enumerate(self._spans):
            if s is span:
                return i
        raise KeyError(span.seq)

    def delete_spans(self, eids: Iterable[int]) -> None:
        seqs = {split_eid(e)[2] for e in eids}
        if not seqs:
            return
        with self._lock:
            self._spans = [s for s in self._spans if s.seq not in seqs]
            for q in seqs:
                self._by_seq.pop(q, None)
            self._derived = {}

    # ── reads ────────────────────────────────────────────────────────────
    def span_count(self) -> int:
        return len(self._spans)

    def instant_count(self) -> int:
        return len(self._instants)

    def counter_count(self) -> int:
        return len(self._counters)

    def spans_list(self) -> list[SpanEvent]:
        with self._lock:
            return list(self._spans)

    def _thread_index(self) -> dict[tuple[int, int], list[SpanEvent]]:
        c = self._cache()
        if "by_thread" not in c:
            idx: dict[tuple[int, int], list[SpanEvent]] = {}
            for s in self._spans:
                idx.setdefault((s.pid, s.tid), []).append(s)
            for lst in idx.values():
                lst.sort(key=lambda s: (s.start_ns, s.seq))
            c["by_thread"] = idx
        return c["by_thread"]

    def _lane_index(self) -> dict[str, list[SpanEvent]]:
        c = self._cache()
        if "by_lane" not in c:
            infos = self.lane_infos()
            by_key = {(ln.pid, (ln.cat, ln.kind, ln.ident)): ln.name for ln in infos}
            idx: dict[str, list[SpanEvent]] = {ln.name: [] for ln in infos}
            for s in self._spans:
                idx[by_key[(s.pid, lane_base(s))]].append(s)
            for lst in idx.values():
                lst.sort(key=lambda s: (s.start_ns, s.seq))
            c["by_lane"] = idx
            c["lane_starts"] = {n: np.fromiter((s.start_ns for s in lst), dtype=np.int64, count=len(lst))
                                for n, lst in idx.items()}
        return c["by_lane"]

    def iter_spans(self, *, order: str = "seq", pid: int | None = None, tid: int | None = None,
                   lane: str | None = None, categories: Iterable[str] | None = None,
                   window: tuple[int, int] | None = None, has_stack: bool | None = None,
                   gpu_model: bool | None = None, with_ids: bool | None = None,
                   filt: SpanFilter | None = None) -> Iterator[SpanEvent]:
        cats = frozenset(categories) if categories is not None else None
        if lane is not None:
            lst = self._lane_index().get(lane, [])
            if window is not None and lst:
                info = self.lane(lane)
                starts = self._cache()["lane_starts"][lane]
                lo = int(np.searchsorted(starts, window[0] - info.max_dur, side="left"))
                hi = int(np.searchsorted(starts, window[1], side="right"))
                lst = lst[lo:hi]
            if order == "seq":
                lst = sorted(lst, key=lambda s: s.seq)
        elif pid is not None and tid is not None:
            lst = self._thread_index().get((pid, tid), [])
            if order == "seq":
                lst = sorted(lst, key=lambda s: s.seq)
        else:
            with self._lock:
                lst = list(self._spans)
            if order == "start":
                lst.sort(key=lambda s: (s.start_ns, s.seq))
        fm = filt if filt is not None and not filt.is_empty() else None
        for s in lst:
            if pid is not None and s.pid != pid:
                continue
            if tid is not None and s.tid != tid:
                continue
            if cats is not None and s.category.value not in cats:
                continue
            if window is not None and not (s.start_ns <= window[1] and s.end_ns > window[0]):
                continue
            if has_stack is not None and bool(s.stack_frames) != has_stack:
                continue
            if gpu_model is not None and (s.tags.get("rt") in ("cuda", "rocm")) != gpu_model:
                continue
            if with_ids is not None and bool(s.span_id or s.parent_span_id) != with_ids:
                continue
            if fm is not None and not fm.matches(s):
                continue
            yield s

    def iter_instants(self) -> Iterator[InstantEvent]:
        with self._lock:
            return iter(list(self._instants))

    def iter_counters(self, *, order: str = "seq") -> Iterator[CounterEvent]:
        with self._lock:
            lst = list(self._counters)
        if order == "name":
            lst.sort(key=lambda c: (c.name, c.timestamp_ns, c.seq))
        return iter(lst)

    def event_by_id(self, eid: int) -> AnyEvent | None:
        _shard, kind, seq = split_eid(eid)
        ev = self._by_seq.get(seq)
        return ev if ev is not None and ev.eid == eid else None

    def span_by_seq(self, seq: int) -> SpanEvent | None:
        ev = self._by_seq.get(seq)
        return ev if isinstance(ev, SpanEvent) else None

    def span_seqs(self) -> np.ndarray:
        with self._lock:
            return np.fromiter((s.seq for s in self._spans), dtype=np.int64, count=len(self._spans))

    # ── metadata ─────────────────────────────────────────────────────────
    def save_meta(self, key: str, value: Any) -> None:
        self._meta[key] = value

    def load_meta(self, key: str, default: Any = None) -> Any:
        return self._meta.get(key, default)
