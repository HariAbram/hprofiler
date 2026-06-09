"""
Trace: container for all profiling events from one run.
Provides analysis helpers (totals, top-N, flame-graph data).
"""

from __future__ import annotations
import heapq
import time
from collections import defaultdict
from dataclasses import dataclass, field
from .events import SpanEvent, InstantEvent, CounterEvent, Category, AnyEvent


@dataclass
class TraceMetadata:
    command: str = ""
    args: list[str] = field(default_factory=list)
    start_time_ns: int = field(default_factory=lambda: time.monotonic_ns())
    end_time_ns: int = 0
    pid: int = 0
    backends_used: list[str] = field(default_factory=list)
    hostname: str = ""
    cwd: str = ""


class Trace:
    """Collects and queries profiling events."""

    def __init__(self, metadata: TraceMetadata | None = None) -> None:
        self.metadata = metadata or TraceMetadata()
        self._events:   list[AnyEvent]     = []
        self._spans:    list[SpanEvent]    = []
        self._instants: list[InstantEvent] = []
        self._counters: list[CounterEvent] = []
        # Fast-path flags for compose() checks
        self._has_stacks: bool = False
        self._has_cpu:    bool = False
        # Populated post-run by the disasm collector
        self._disasm: dict[str, "KernelDisasm"] = {}  # type: ignore[type-arg]
        # Populated post-run by device capability queries
        self._devices: list["DevicePeak"] = []  # type: ignore[type-arg]

    def add_disasm(self, kd: "KernelDisasm") -> None:  # type: ignore[type-arg]
        self._disasm[kd.name] = kd

    @property
    def disasm(self) -> dict[str, "KernelDisasm"]:  # type: ignore[type-arg]
        return dict(self._disasm)

    def set_devices(self, devices: list["DevicePeak"]) -> None:  # type: ignore[type-arg]
        self._devices = list(devices)

    @property
    def devices(self) -> list["DevicePeak"]:  # type: ignore[type-arg]
        return list(self._devices)

    def add(self, event: AnyEvent) -> None:
        self._events.append(event)
        if isinstance(event, SpanEvent):
            self._spans.append(event)
            if event.stack_frames:
                self._has_stacks = True
            if event.category == Category.CPU:
                self._has_cpu = True
        elif isinstance(event, InstantEvent):
            self._instants.append(event)
        elif isinstance(event, CounterEvent):
            self._counters.append(event)

    def add_many(self, events: list[AnyEvent]) -> None:
        for event in events:
            self.add(event)

    @property
    def spans(self) -> list[SpanEvent]:
        return self._spans

    @property
    def instants(self) -> list[InstantEvent]:
        return self._instants

    @property
    def counters(self) -> list[CounterEvent]:
        return self._counters

    @property
    def all_events(self) -> list[AnyEvent]:
        return list(self._events)

    @property
    def duration_ns(self) -> int:
        if not self._events:
            return 0
        end = self.metadata.end_time_ns or time.monotonic_ns()
        return end - self.metadata.start_time_ns

    def spans_by_category(self) -> dict[Category, list[SpanEvent]]:
        result: dict[Category, list[SpanEvent]] = defaultdict(list)
        for s in self._spans:
            result[s.category].append(s)
        return dict(result)

    def top_spans(self, n: int = 20) -> list[SpanEvent]:
        return heapq.nlargest(n, self._spans, key=lambda s: s.duration_ns)

    def aggregated_stats(self) -> list[dict]:
        """Group spans by name, return sorted by total time desc."""
        totals: dict[str, dict] = defaultdict(lambda: {
            "name": "", "category": "", "count": 0,
            "total_ns": 0, "min_ns": float("inf"), "max_ns": 0
        })
        for s in self._spans:
            key = f"{s.category.value}::{s.name}"
            r = totals[key]
            r["name"] = s.name
            r["category"] = s.category.value
            r["count"] += 1
            r["total_ns"] += s.duration_ns
            r["min_ns"] = min(r["min_ns"], s.duration_ns)
            r["max_ns"] = max(r["max_ns"], s.duration_ns)

        rows = list(totals.values())
        for r in rows:
            r["avg_ns"] = r["total_ns"] / r["count"] if r["count"] else 0
            if r["min_ns"] == float("inf"):
                r["min_ns"] = 0

        total_time = sum(r["total_ns"] for r in rows) or 1
        for r in rows:
            r["pct"] = 100.0 * r["total_ns"] / total_time

        return sorted(rows, key=lambda r: r["total_ns"], reverse=True)

    def lanes(self) -> dict[str, list[SpanEvent]]:
        """Return spans grouped into display lanes.

        CUDA spans with a 'stream' tag get a per-stream lane (cuda/stream-N)
        so the Timeline can show kernel/memcpy overlap across streams.
        All other spans keep the existing (category/thread-TID) grouping.
        """
        lanes: dict[str, list[SpanEvent]] = defaultdict(list)
        for s in self._spans:
            if s.category.value in ("cuda", "rocm") and "stream" in s.tags:
                key = f"{s.category.value}/stream-{s.tags['stream']}"
            elif s.tid:
                key = f"{s.category.value}/thread-{s.tid}"
            else:
                key = s.category.value
            lanes[key].append(s)
        return dict(lanes)
