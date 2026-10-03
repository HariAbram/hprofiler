"""
Trace: container for all profiling events from one run.
Provides analysis helpers (totals, top-N, flame-graph data).
"""

from __future__ import annotations
import heapq
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable, Iterator
from .events import SpanEvent, InstantEvent, CounterEvent, Category, AnyEvent

if TYPE_CHECKING:
    from .store import LaneInfo, TraceStore


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
    # Wall-clock ISO timestamp of when profiling started -- unlike
    # start_time_ns (monotonic, unrecoverable as a real timestamp after
    # the fact), this is for DISPLAY ("capture time" in the GUI's
    # Overview tab), not timing math. "" (not populated) for any trace
    # captured before this field existed, or built by hand in a test --
    # the GUI renders that as "unavailable", not a fake/guessed time.
    capture_time_iso: str = ""
    # CUDA/ROCm device-activity provenance, keyed "pid/rt" (see
    # src/core/gpu_activity.py): native tracer status, clock mapping,
    # dropped records, correlation/de-duplication counts.
    device_activity: dict = field(default_factory=dict)
    # Capture integrity (src/core/receiver.py): "state" (running ->
    # complete; "interrupted" when a store is reopened that never
    # completed), the hook transports' status per pid/hook (drops, lost,
    # oversize, missing final drain) and the receiver's counters (malformed /
    # unknown / partial records, processing errors). Empty for traces
    # captured before this existed. receiver.capture_warnings() renders it.
    capture_health: dict = field(default_factory=dict)


class Trace:
    """Collects and queries profiling events.

    Events live in a TraceStore (src/core/store): MemoryTraceStore by
    default (small traces, tests), DiskTraceStore for captures and large
    traces. The list properties (`spans`, `instants`, `counters`,
    `all_events`) and `lanes()` materialize every event and remain for
    compatibility and small traces; code that may see large traces uses the
    iterator/query methods (iter_spans, events_in_window, lanes_in_window,
    aggregate_stats, lane_infos, event_by_id) and the store's precomputed
    aggregations."""

    def __init__(self, metadata: TraceMetadata | None = None, store: "TraceStore | None" = None) -> None:
        from .store import MemoryTraceStore
        self.metadata = metadata or TraceMetadata()
        self.store = store if store is not None else MemoryTraceStore()
        self._lock = threading.Lock()
        # Fast-path flags for compose() checks
        self._has_stacks: bool = False
        self._has_cpu:    bool = False
        # Populated post-run by the disasm collector
        self._disasm: dict[str, "KernelDisasm"] = {}  # type: ignore[type-arg]
        self._disasm_version: int = 0  # incremented by add_disasm; lets poller detect annotation updates
        # Populated by cuda_hook CUPTI PC sampling records
        self._pc_samples: dict[str, list[tuple[int, int, int]]] = {}  # func → [(pc_offset, stall_reason, count)]
        # Populated post-run by device capability queries
        self._devices: list["DevicePeak"] = []  # type: ignore[type-arg]

    def add_disasm(self, kd: "KernelDisasm") -> None:  # type: ignore[type-arg]
        with self._lock:
            self._disasm[kd.name] = kd
            # Also store under 255-char prefix so spans recorded by older hook
            # builds (which truncated kname to 255 chars) still match.
            if len(kd.name) > 255:
                self._disasm.setdefault(kd.name[:255], kd)
            self._disasm_version += 1

    def add_pc_sample(self, func_name: str, pc_offset: int, stall_reason: int, count: int) -> None:
        with self._lock:
            if func_name not in self._pc_samples:
                self._pc_samples[func_name] = []
            self._pc_samples[func_name].append((pc_offset, stall_reason, count))

    @property
    def disasm(self) -> dict[str, "KernelDisasm"]:  # type: ignore[type-arg]
        with self._lock:
            return dict(self._disasm)

    def set_devices(self, devices: list["DevicePeak"]) -> None:  # type: ignore[type-arg]
        self._devices = list(devices)

    @property
    def devices(self) -> list["DevicePeak"]:  # type: ignore[type-arg]
        return list(self._devices)

    # ── writing ──────────────────────────────────────────────────────────
    def add(self, event: AnyEvent) -> None:
        self.store.append(event)
        if isinstance(event, SpanEvent):
            if event.stack_frames:
                self._has_stacks = True
            if event.category == Category.CPU:
                self._has_cpu = True

    def add_many(self, events: list[AnyEvent]) -> None:
        for event in events:
            self.add(event)

    def update_span(self, span: SpanEvent) -> None:
        """Persist in-place changes to a span obtained from this trace
        (required for disk-backed traces; harmless in memory)."""
        self.store.update_span(span)
        if span.stack_frames:
            self._has_stacks = True

    def update_spans(self, spans: "Iterable[SpanEvent]") -> None:
        spans = list(spans)
        if hasattr(self.store, "update_spans"):
            self.store.update_spans(spans)
        else:
            for s in spans:
                self.store.update_span(s)
        if any(s.stack_frames for s in spans):
            self._has_stacks = True

    def delete_spans(self, spans: "Iterable[SpanEvent]") -> None:
        self.store.delete_spans([s.eid for s in spans])

    def remove_spans(self, span_ids: set[int]) -> None:
        """Drop the spans whose id() is in span_ids (compatibility; prefer
        delete_spans)."""
        if span_ids:
            self.store.delete_spans([s.eid for s in self.iter_spans() if id(s) in span_ids])

    # ── list views (materialize; small traces / compatibility) ───────────
    @property
    def spans(self) -> list[SpanEvent]:
        spans_list = getattr(self.store, "spans_list", None)
        if spans_list is not None:
            return spans_list()
        return list(self.store.iter_spans())

    @property
    def instants(self) -> list[InstantEvent]:
        return list(self.store.iter_instants())

    @property
    def counters(self) -> list[CounterEvent]:
        return list(self.store.iter_counters())

    @property
    def all_events(self) -> list[AnyEvent]:
        return list(heapq.merge(self.store.iter_spans(), self.store.iter_instants(),
                                self.store.iter_counters(), key=lambda e: e.seq))

    # ── queries (bounded memory) ─────────────────────────────────────────
    def iter_spans(self, **constraints) -> "Iterator[SpanEvent]":
        """See TraceStore.iter_spans for the constraints (order, pid, tid,
        lane, categories, window, has_stack, gpu_model, filt)."""
        return self.store.iter_spans(**constraints)

    def iter_instants(self) -> "Iterator[InstantEvent]":
        return self.store.iter_instants()

    def iter_counters(self, order: str = "seq") -> "Iterator[CounterEvent]":
        return self.store.iter_counters(order=order)

    def span_count(self) -> int:
        return self.store.span_count()

    def event_by_id(self, eid: int) -> AnyEvent | None:
        return self.store.event_by_id(eid)

    def events_in_window(self, start_ns: int, end_ns: int, *, lanes=None, filt=None) -> "Iterator[SpanEvent]":
        """Spans overlapping [start_ns, end_ns] (of the given lanes), in
        (start, arrival) order."""
        return self.store.events_in_window(start_ns, end_ns, lanes=lanes, filt=filt)

    def lanes_in_window(self, start_ns: int, end_ns: int, *, filt=None,
                        limit_per_lane: int | None = None) -> dict[str, list[SpanEvent]]:
        return self.store.lanes_in_window(start_ns, end_ns, filt=filt, limit_per_lane=limit_per_lane)

    def lane_infos(self) -> "list[LaneInfo]":
        return self.store.lane_infos()

    def aggregate_stats(self) -> list[dict]:
        """Per (category, name) span statistics, computed by the store."""
        return self.store.aggregate_stats()

    def counter_values(self) -> dict[str, float]:
        """Last value of each counter by name (what the summaries show)."""
        out: dict[str, float] = {}
        for c in self.store.iter_counters():
            out[c.name] = c.value
        return out

    def finalize(self) -> None:
        """Build the store's indexes and derived tables, persist metadata."""
        self.save()
        self.store.finalize()

    def save(self) -> None:
        """Write metadata/disasm/devices into the store (disk stores keep
        them; a no-op in effect for memory stores)."""
        from . import trace_io
        trace_io.save_trace_meta(self)

    def close(self) -> None:
        self.store.close()

    @property
    def duration_ns(self) -> int:
        if not self.store.span_count() and not self.store.instant_count() \
                and not self.store.counter_count():
            return 0
        end = self.metadata.end_time_ns or time.monotonic_ns()
        return end - self.metadata.start_time_ns

    def spans_by_category(self) -> dict[Category, list[SpanEvent]]:
        result: dict[Category, list[SpanEvent]] = defaultdict(list)
        for s in self.iter_spans():
            result[s.category].append(s)
        return dict(result)

    def top_spans(self, n: int = 20) -> list[SpanEvent]:
        return heapq.nlargest(n, self.iter_spans(), key=lambda s: s.duration_ns)

    def aggregated_stats(self) -> list[dict]:
        """Group spans by name, return sorted by total time desc."""
        return self.store.aggregate_stats()

    def cct(self) -> "CCT":  # type: ignore[name-defined]
        """Build and return a Calling Context Tree from all stacked spans."""
        from ..analysis.cct import CCT
        return CCT.build(self)

    def lanes(self) -> dict[str, list[SpanEvent]]:
        """Return spans grouped into display lanes (parse names with
        parse_lane_name()).

        Host API calls (side=cpu) always stay on their thread's lane.
        Device spans with a 'stream' tag -- CUDA/ROCm device work (side=gpu,
        including copies/memsets in the memory category) and pre-split
        CUDA/ROCm spans -- get a per-stream lane (cuda/stream-N) so the
        Timeline shows kernel/memcpy overlap across streams. Device-timed
        spans with no stream (OpenCL side=gpu, *_gpu transfers) get a
        category/device lane: their tid is whichever driver callback thread
        fired, which would otherwise scatter them into (and overlap with)
        unrelated host-thread lanes. Everything else is category/thread-TID.

        A key shared by several processes is suffixed with @PID. Stream ids
        repeat across processes (the default stream is 0 in every rank, and
        same-binary processes hash identical stream pointers), and tids
        repeat across nodes after merge-nodes -- without the suffix, every
        rank's kernels were drawn in ONE lane. Single-process traces keep
        the unsuffixed names.
        """
        return {ln.name: list(self.store.iter_spans(lane=ln.name)) for ln in self.store.lane_infos()}


def parse_lane_name(name: str) -> tuple[str, str, str, int | None]:
    """Inverse of Trace.lanes()'s naming: (category, kind, ident, pid).
    kind is "thread", "stream", "device" or "" (bare category); ident is
    the tid/stream id as a string ("" when kind has none); pid is set only
    for @PID-disambiguated lanes."""
    pid: int | None = None
    body = name
    if "@" in body:
        body, _, p = body.rpartition("@")
        try:
            pid = int(p)
        except ValueError:
            body, pid = name, None
    cat, _, suffix = body.partition("/")
    for kind in ("thread", "stream"):
        if suffix.startswith(kind + "-"):
            return cat, kind, suffix[len(kind) + 1:], pid
    if suffix == "device":
        return cat, "device", "", pid
    return cat, "", suffix, pid
