"""
Calling Context Tree (CCT) for hprofiler.

Builds a tree where each path from root to leaf represents a unique call path.
Each node accumulates both exclusive (self) and inclusive (subtree) metrics:
count, total time, min/max per call.

The CCT serves two purposes:
  1. Aggregated view — instead of thousands of identical span records for
     a repeated kernel launch, each unique call path collapses to one node.
  2. Call-path hotspot attribution — identifies which C++ code paths are
     responsible for the most GPU time.

GPU starvation detection is also here: it computes the fraction of wall time
where the GPU was idle and the CPU was visibly busy (sync stalls, launch gaps).

Usage:
    from hprofiler.analysis.cct import CCT
    cct = CCT.build(trace)
    cct.print_summary(wall_ns=trace.duration_ns)
    stats = cct.gpu_starvation(trace)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.trace import Trace
    from ..core.events import SpanEvent


# ── CCT node ──────────────────────────────────────────────────────────────────

@dataclass
class CCTNode:
    """A single node in the calling context tree."""
    frame: str
    parent: "CCTNode | None" = field(default=None, repr=False)
    children: "dict[str, CCTNode]" = field(default_factory=dict)

    # Exclusive (self) metrics — charged only when this frame is the leaf
    self_count: int = 0
    self_ns: int = 0
    self_min_ns: int = 2**62
    self_max_ns: int = 0
    categories: set = field(default_factory=set)

    # Inclusive metrics — propagated up from all descendants
    incl_ns: int = 0
    incl_count: int = 0

    @property
    def display_name(self) -> str:
        """Symbol name only (strips |lib|offset if present)."""
        return self.frame.split("|")[0]

    @property
    def lib_path(self) -> str:
        """Library path from frame info, or empty string."""
        parts = self.frame.split("|")
        return parts[1] if len(parts) >= 2 else ""

    @property
    def lib_offset(self) -> str:
        """Hex offset string from frame info, or empty string."""
        parts = self.frame.split("|")
        return parts[2] if len(parts) >= 3 else ""

    @property
    def self_avg_ns(self) -> float:
        return self.self_ns / self.self_count if self.self_count else 0.0

    def get_or_add_child(self, frame: str) -> "CCTNode":
        if frame not in self.children:
            self.children[frame] = CCTNode(frame=frame, parent=self)
        return self.children[frame]

    def all_nodes(self):
        """DFS iterator over this node and all descendants."""
        yield self
        for child in self.children.values():
            yield from child.all_nodes()

    def call_path(self) -> list[str]:
        """Return display names from root to this node (excludes [root])."""
        path: list[str] = []
        n: CCTNode | None = self
        while n is not None and n.frame != "[root]":
            path.append(n.display_name)
            n = n.parent
        return list(reversed(path))


# ── CCT ───────────────────────────────────────────────────────────────────────

class CCT:
    """Calling Context Tree built from a profiling Trace."""

    def __init__(self) -> None:
        self.root = CCTNode(frame="[root]")

    # ── construction ─────────────────────────────────────────────────────────

    @classmethod
    def build(cls, trace: "Trace") -> "CCT":
        """
        Build a CCT from all spans that carry call-stack information.

        GPU spans (CUDA/ROCm/OpenCL) use the stack_frames attached by
        callstack.h.  CPU perf-sample spans use the 'stack' tag written by
        the perf-script parser in runner.py.
        """
        cct = cls()
        seen_perf: set[tuple] = set()   # dedup perf samples by (pid,tid,ts)

        for span in trace.spans:
            frames = _extract_frames(span, seen_perf)
            if not frames:
                continue

            # frames is innermost-first; build path as root→outermost→...→leaf
            path = list(reversed(frames)) + [span.name]

            node = cct.root
            for frame in path:
                node = node.get_or_add_child(frame)

            dur = span.duration_ns
            node.self_count += 1
            node.self_ns += dur
            if dur < node.self_min_ns:
                node.self_min_ns = dur
            if dur > node.self_max_ns:
                node.self_max_ns = dur
            node.categories.add(span.category.value)

            # Propagate inclusive time up to root
            n: CCTNode | None = node
            while n is not None:
                n.incl_ns += dur
                n.incl_count += 1
                n = n.parent

        return cct

    # ── queries ───────────────────────────────────────────────────────────────

    def top_self(self, n: int = 15, category: str | None = None) -> list[CCTNode]:
        """Top-N leaf nodes by exclusive time, optionally filtered by category."""
        nodes = [
            nd for nd in self.root.all_nodes()
            if nd.self_count > 0 and nd.frame != "[root]"
            and (category is None or category in nd.categories)
        ]
        return sorted(nodes, key=lambda nd: nd.self_ns, reverse=True)[:n]

    def top_incl(self, n: int = 15, category: str | None = None) -> list[CCTNode]:
        """Top-N nodes by inclusive time (hot call paths)."""
        nodes = [
            nd for nd in self.root.all_nodes()
            if nd.incl_count > 0 and nd.frame != "[root]"
            and (category is None or category in nd.categories)
        ]
        return sorted(nodes, key=lambda nd: nd.incl_ns, reverse=True)[:n]

    def total_exclusive_ns(self) -> int:
        return sum(
            nd.self_ns for nd in self.root.all_nodes()
            if nd.frame != "[root]"
        )

    # ── text output ───────────────────────────────────────────────────────────

    def print_summary(self, wall_ns: int = 0, top_n: int = 15) -> None:
        nodes = self.top_self(top_n)
        if not nodes:
            return
        total = self.total_exclusive_ns() or 1
        wall  = wall_ns or total

        print(f"\n  Call-path hotspots (CCT, top {min(top_n, len(nodes))}):")
        hdr = (f"  {'Function':<42} {'Cat':<8} {'Calls':>7}"
               f" {'Total':>10} {'Avg':>10} {'%wall':>6}")
        print(hdr)
        print(f"  {'-'*85}")
        for nd in nodes:
            path = nd.call_path()
            # Show at most two callers above the leaf
            if len(path) > 3:
                display = "…/" + "/".join(p.split("(")[0][:20] for p in path[-2:])
            else:
                display = "/".join(p.split("(")[0][:22] for p in path)
            display = display[:42]
            cat = ",".join(sorted(nd.categories))[:8]
            pct = 100.0 * nd.self_ns / wall
            print(
                f"  {display:<42} {cat:<8} {nd.self_count:>7}"
                f" {_fmt_ns(nd.self_ns):>10} {_fmt_ns(nd.self_avg_ns):>10}"
                f" {pct:>5.1f}%"
            )


# ── GPU starvation analysis ───────────────────────────────────────────────────

def gpu_starvation(trace: "Trace") -> dict:
    """
    Analyse GPU starvation from the span timeline.

    Returns a dict with keys:
      wall_ns          — total wall time
      gpu_active_ns    — time at least one GPU kernel was executing
      gpu_active_pct   — gpu_active_ns / wall_ns * 100
      sync_stall_ns    — time spent in GPU sync calls (CPU waiting for GPU)
      sync_stall_pct   — sync_stall_ns / wall_ns * 100
      launch_gap_ns    — wall time with no GPU kernels running (excl. sync)
      launch_gap_pct   — launch_gap_ns / wall_ns * 100
      sync_calls       — number of synchronisation calls captured
    """
    from ..core.events import Category

    wall_ns = trace.duration_ns or 1

    _GPU_CATS = {Category.GPU_CUDA, Category.GPU_ROCM, Category.GPU_OPENCL}
    _SYNC_NAMES = {
        "cudaDeviceSynchronize", "cudaStreamSynchronize",
        "cudaEventSynchronize",  "hipDeviceSynchronize",
        "hipStreamSynchronize",  "hipEventSynchronize",
        "clFinish", "clWaitForEvents",
    }

    kernel_intervals: list[tuple[int, int]] = []
    sync_ns = 0
    sync_count = 0

    for span in trace.spans:
        if span.category in _GPU_CATS:
            if span.tags.get("type") == "kernel" or span.tags.get("side") in ("gpu", None):
                kernel_intervals.append((span.start_ns, span.start_ns + span.duration_ns))
        if span.name in _SYNC_NAMES and span.duration_ns > 0:
            sync_ns += span.duration_ns
            sync_count += 1

    # Merge overlapping kernel intervals to get true GPU active time
    gpu_active_ns = _merge_intervals(kernel_intervals)

    launch_gap_ns = max(0, wall_ns - gpu_active_ns - sync_ns)

    return {
        "wall_ns":        wall_ns,
        "gpu_active_ns":  gpu_active_ns,
        "gpu_active_pct": 100.0 * gpu_active_ns / wall_ns,
        "sync_stall_ns":  sync_ns,
        "sync_stall_pct": 100.0 * sync_ns / wall_ns,
        "launch_gap_ns":  launch_gap_ns,
        "launch_gap_pct": 100.0 * launch_gap_ns / wall_ns,
        "sync_calls":     sync_count,
    }


# ── addr2line annotation for call-stack frames ────────────────────────────────

def annotate_stack_frames(trace: "Trace") -> int:
    """
    Resolve lib|offset frame info in span.stack_frames to file:line.

    For each frame of the form 'sym|/lib.so|0xoffset', calls addr2line and
    rewrites the frame to 'sym (file:line)|/lib.so|0xoffset'.

    Returns the number of frames resolved.
    """
    import shutil
    from ..analysis.addr2line import _find_symbolizer, _resolve_batch

    tool = _find_symbolizer()
    if tool is None:
        return 0

    # Collect (lib, offset) pairs across all frames in all stacked spans
    lib_addrs: dict[str, list[tuple[str, object, int]]] = {}
    # lib_addrs[lib] = [(hex_addr, span, frame_idx), ...]

    for span in trace.spans:
        for idx, frame in enumerate(span.stack_frames):
            parts = frame.split("|")
            if len(parts) != 3:
                continue
            _, lib, offset = parts
            if not lib or not offset.startswith("0x"):
                continue
            lib_addrs.setdefault(lib, []).append((offset, span, idx))

    resolved_count = 0
    for lib, entries in lib_addrs.items():
        from pathlib import Path
        if not Path(lib).exists():
            continue
        unique_addrs = list({e[0] for e in entries})
        resolved = _resolve_batch(tool, lib, unique_addrs)
        for hex_addr, span, idx in entries:
            if hex_addr in resolved:
                file_, line_ = resolved[hex_addr]
                parts = span.stack_frames[idx].split("|")
                sym = parts[0]
                short_file = file_.split("/")[-1] if "/" in file_ else file_
                parts[0] = f"{sym} ({short_file}:{line_})"
                span.stack_frames[idx] = "|".join(parts)
                resolved_count += 1

    return resolved_count


# ── helpers ───────────────────────────────────────────────────────────────────

def _extract_frames(span: "SpanEvent", seen_perf: set) -> list[str]:
    """
    Extract call-stack frames from a span.

    Returns frames in innermost-first order, or [] if no stack info.
    Deduplicates perf CPU samples using seen_perf (mutated in place).
    """
    from ..core.events import Category

    # GPU spans: frames from callstack.h attached by runner.py
    if span.stack_frames:
        return list(span.stack_frames)

    # CPU perf samples: reconstruct from the folded stack tag
    if span.category == Category.CPU and span.duration_ns == 0:
        key = (span.pid, span.tid, span.start_ns)
        if key in seen_perf:
            return []
        stack_str = span.tags.get("stack", "")
        if not stack_str:
            return []
        seen_perf.add(key)
        # stack tag is outermost-first; return innermost-first
        frames = [f for f in stack_str.split(";") if f and f != "[cpu]"]
        return list(reversed(frames))

    return []


def _merge_intervals(intervals: list[tuple[int, int]]) -> int:
    """Return the total length of a union of [start, end) intervals."""
    if not intervals:
        return 0
    sorted_ivs = sorted(intervals)
    total = 0
    cur_start, cur_end = sorted_ivs[0]
    for s, e in sorted_ivs[1:]:
        if s <= cur_end:
            cur_end = max(cur_end, e)
        else:
            total += cur_end - cur_start
            cur_start, cur_end = s, e
    total += cur_end - cur_start
    return total


def _fmt_ns(ns: float) -> str:
    if ns >= 1_000_000_000:
        return f"{ns/1_000_000_000:.3f}s"
    if ns >= 1_000_000:
        return f"{ns/1_000_000:.2f}ms"
    if ns >= 1_000:
        return f"{ns/1_000:.1f}µs"
    return f"{ns:.0f}ns"
