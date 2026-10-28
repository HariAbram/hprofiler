"""
CUDA / ROCm stream semantics beyond FIFO order (criticalpath._add_gpu_edges
uses these):

  * host functions (cudaLaunchHostFunc, cudaStreamAddCallback,
    cuLaunchHostFunc): stream-ordered work, run by the runtime on its own
    callback thread one at a time;
  * a synchronous copy that blocks the host (blocking=1, the hook's reading
    of CUDA's API synchronization behaviour) returns after its device copy;
  * the legacy default stream (stream 0) synchronizes with every blocking
    stream of its device: its work starts after their earlier work and
    their later work starts after it. A stream created non-blocking
    (cudaStreamNonBlocking / hipStreamNonBlocking: nonblocking=1 on its
    creation) is exempt. Streams whose creation the trace did not record are
    assumed blocking (cudaStreamCreate's default) at "medium"; programs
    compiled for the per-thread default stream break this assumption.
"""
from __future__ import annotations

from collections import defaultdict

STREAM_ORDERED_EXTRA_OPS = frozenset({"host_func"})


def stream_flags(spans, items) -> tuple[set[tuple], set[tuple]]:
    """((pid, rt, stream) created, (pid, rt, stream) created non-blocking)
    from the stream-creation calls."""
    created, nonblocking = set(), set()
    for _i, s in items(spans):
        t = s.tags
        if t.get("op") == "stream_create" and t.get("stream") not in (None, "0"):
            key = (s.pid, t.get("rt"), t["stream"])
            created.add(key)
            if t.get("nonblocking") == "1":
                nonblocking.add(key)
    return created, nonblocking


def host_function_order(spans, items) -> list[tuple[int, int]]:
    """(earlier, later) consecutive host-function executions of one callback
    thread on different streams (same-stream order is the stream's)."""
    by_thread: dict[tuple, list[int]] = defaultdict(list)
    for i, s in items(spans):
        if s.tags.get("side") == "gpu" and s.tags.get("op") == "host_func":
            by_thread[(s.pid, s.tid)].append(i)
    out = []
    for lst in by_thread.values():
        lst.sort(key=lambda i: spans[i].start_ns)
        for a, b in zip(lst, lst[1:]):
            if spans[a].tags.get("stream") != spans[b].tags.get("stream"):
                out.append((a, b))
    return out
