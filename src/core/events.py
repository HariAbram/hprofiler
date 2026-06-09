"""
Core event data model. All profiling backends emit these types.
Timestamps are in nanoseconds (monotonic clock).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventKind(str, Enum):
    SPAN = "span"          # begin/end duration event
    INSTANT = "instant"    # point-in-time marker
    COUNTER = "counter"    # numeric gauge (e.g. memory usage)


class Category(str, Enum):
    CPU = "cpu"
    GPU_CUDA = "cuda"
    GPU_ROCM = "rocm"
    GPU_OPENCL = "opencl"
    OPENMP = "openmp"
    MPI = "mpi"
    NCCL = "nccl"
    MEMORY = "memory"
    SYNC = "sync"
    JIT = "jit"
    NVTX = "nvtx"
    OTHER = "other"


@dataclass
class SpanEvent:
    """A duration event: kernel launch, function call, memory transfer, etc."""
    name: str
    category: Category
    start_ns: int           # wall-clock ns since trace start
    duration_ns: int
    pid: int = 0
    tid: int = 0
    tags: dict[str, Any] = field(default_factory=dict)
    stack_frames: list[str] = field(default_factory=list)  # innermost→outermost (backtrace order)
    span_id: str = ""         # hook-assigned unique ID (uint64 decimal string from C)
    parent_span_id: str = ""  # parent's span_id; empty = root span

    @property
    def end_ns(self) -> int:
        return self.start_ns + self.duration_ns

    @property
    def start_us(self) -> float:
        return self.start_ns / 1_000

    @property
    def duration_us(self) -> float:
        return self.duration_ns / 1_000

    @property
    def duration_ms(self) -> float:
        return self.duration_ns / 1_000_000


@dataclass
class InstantEvent:
    """A point-in-time marker (e.g. JIT compilation completed)."""
    name: str
    category: Category
    timestamp_ns: int
    pid: int = 0
    tid: int = 0
    tags: dict[str, Any] = field(default_factory=dict)


@dataclass
class CounterEvent:
    """A numeric gauge sampled over time (e.g. GPU memory usage)."""
    name: str
    category: Category
    timestamp_ns: int
    value: float
    unit: str = ""
    pid: int = 0


from typing import Union
AnyEvent = Union[SpanEvent, InstantEvent, CounterEvent]
