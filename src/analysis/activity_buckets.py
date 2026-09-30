"""
Maps each span to one of a small set of activity buckets -- Computation /
Communication / Synchronization / Memory transfer / Runtime overhead / Idle /
Annotation / Other -- for the GUI's Overview "Time breakdown" panel and
Timeline legend/grouping (both must agree, so both import this module rather
than each keeping their own mapping, which is how they drifted apart before).

The `tags["type"]` sub-tag (e.g. "type=kernel,grid=...", set by the C hooks
right next to the span's name/duration -- see hooks/*/*.c's `emit_span`
calls) is already this codebase's established compute-vs-overhead
discriminator: `tags.get("type") == "kernel"` is used identically in
bridge.py and output/summary.py to separate a GPU
kernel's actual execution from the API calls (alloc/free/launch) around it,
and analysis/pop_efficiency.py's `_DATA_MOVEMENT_TYPES` already excludes
memcpy/alloc/free from "useful compute" for the same reason. This module
generalizes that established pattern into a full activity taxonomy instead
of inventing a new one.

`type` is looked up FIRST (it's the precise, per-event signal); `category`
is the fallback for spans with no `type` tag at all (a plain CPU function
call, for instance, never carries one).

Annotation (nvtx_range/nvtx_mark/roctx_range/roctx_mark) is its OWN bucket,
excluded from `bucket_totals()`'s totals -- these are user-inserted markers
that overlap real work by design (an NVTX range typically spans several real
kernels/memcpys underneath it), so bucketing them as additional "work" would
double-count time already attributed to whatever they wrap.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.events import SpanEvent

BUCKETS = (
    "Computation", "Communication", "Synchronization", "Memory transfer",
    "Runtime overhead", "Idle", "Annotation", "Other",
)

# OMPT/GOMP parallel-region and work-sharing constructs measure the WHOLE
# region's wall time (including the real work threads do inside it), not
# just dispatch overhead -- same reasoning as CUDA/ROCm/OpenCL "kernel".
_COMPUTATION_TYPES = frozenset({
    "kernel", "jit_kernel", "work", "task", "single", "parallel",
    "parallel_region", "implicit_task", "offload",
})

# Collective/point-to-point payload transfer + MPI RMA data ops. "start"/
# "startall" (persistent-request issue) are grouped here, not with "wait*",
# since they represent initiating a transfer, not waiting on one.
_COMMUNICATION_TYPES = frozenset({
    "allgather", "allreduce", "alltoall", "broadcast", "reduce",
    "reduce_scatter", "send", "recv", "isend", "irecv", "send_init",
    "recv_init", "get", "put", "accumulate", "read", "write",
    "start", "startall",
})

# Genuine synchronization/blocking points -- the "wait*"/"test*" family,
# OpenMP critical sections, and MPI RMA window fence/lock/unlock (these
# establish memory consistency, they don't move payload bytes themselves).
_SYNCHRONIZATION_TYPES = frozenset({
    "barrier", "wait", "waitall", "waitany", "waitsome", "test", "testall",
    "testany", "testsome", "cancel", "critical", "sync",
    "win_fence", "win_flush", "win_flush_all",
    "win_lock", "win_lock_all", "win_unlock", "win_unlock_all",
})

# Host<->device (and SVM) byte-copy transfers -- pop_efficiency.py's
# _DATA_MOVEMENT_TYPES minus alloc/free, which belong in Runtime overhead
# below (allocation bookkeeping, not a data transfer).
_MEMORY_TRANSFER_TYPES = frozenset({
    "DtoH", "HtoD", "memcpy", "memcpy_async", "svm_memcpy",
})

# API/administrative bookkeeping around real work: memory management,
# CUDA-graph dispatch (the graph body's own kernels get their own "kernel"
# spans; this span is just the launch call), JIT compile/load (distinct
# from "jit_kernel", the JIT'd code actually running), and MPI
# communicator/group lifecycle.
_OVERHEAD_TYPES = frozenset({
    "alloc", "alloc_async", "alloc_managed", "alloc_pinned",
    "free", "free_async", "free_pinned",
    "graph_launch", "jit_compile", "jit_load",
    "comm_init", "comm_init_all", "comm_destroy", "group", "task_create",
})

_ANNOTATION_TYPES = frozenset({
    "nvtx_mark", "nvtx_range", "roctx_mark", "roctx_range",
})

# Fallback when a span carries no `type` tag at all (e.g. a plain CPU
# function-call span, or hooks/os_tracer's sched-trace spans, which have no
# type sub-tag by design).
_CATEGORY_FALLBACK = {
    "cpu": "Computation", "cuda": "Computation", "rocm": "Computation",
    "opencl": "Computation", "openmp": "Computation",
    "mpi": "Communication", "nccl": "Communication",
    "sync": "Synchronization",
    "memory": "Memory transfer",
    "jit": "Runtime overhead",
    "nvtx": "Annotation",
    # Off-CPU/scheduler time is, from the thread's own perspective, idle
    # time -- it wasn't running, regardless of why.
    "sched": "Idle",
}


def bucket_of(category: str, span_type: str) -> str:
    if span_type in _ANNOTATION_TYPES:
        return "Annotation"
    if span_type in _COMPUTATION_TYPES:
        return "Computation"
    if span_type in _COMMUNICATION_TYPES:
        return "Communication"
    if span_type in _SYNCHRONIZATION_TYPES:
        return "Synchronization"
    if span_type in _MEMORY_TRANSFER_TYPES:
        return "Memory transfer"
    if span_type in _OVERHEAD_TYPES:
        return "Runtime overhead"
    return _CATEGORY_FALLBACK.get(category, "Other")


def bucket_of_span(span: "SpanEvent") -> str:
    # OpenCL emits two spans per kernel: side=cpu is the host enqueue call
    # (API overhead), side=gpu the real device execution. Both carry
    # type=kernel, so without this the enqueue call counted as compute.
    if span.tags.get("side") == "cpu" and span.tags.get("type") == "kernel":
        return "Runtime overhead"
    return bucket_of(span.category.value, span.tags.get("type", ""))


# Categories whose spans measure device-side activity (kernels, device
# copies, collectives timed with device events). Their tid is the thread
# that LAUNCHED them, not a thread they ran on, so they must never be
# nested under (or subtracted from) that host thread's own spans.
_DEVICE_CATEGORIES = frozenset({"cuda", "rocm", "opencl", "nccl"})


def is_device_timed(span: "SpanEvent") -> bool:
    tags = span.tags
    side = tags.get("side")
    if side == "gpu":
        return True
    if side == "cpu":
        return False
    if span.name.endswith("_gpu") or tags.get("type") == "memcpy_async":
        return True
    return span.category.value in _DEVICE_CATEGORIES


def is_sampled(span: "SpanEvent") -> bool:
    """perf samples (the only producer of cpu-category spans): a nominal
    one-sampling-interval weight, an estimate rather than a measured
    interval."""
    return span.category.value == "cpu"


def exclusive_host_ns(spans: list) -> dict[int, int]:
    """id(span) -> nanoseconds that span exclusively owns on its host
    thread. Host spans nest (an OpenMP barrier inside a parallel region, an
    MPI_Wait inside an instrumented function...); summing their raw
    durations counts the inner time twice. Here every instant of a
    (pid, tid)'s timeline belongs to exactly one span: the innermost one
    covering it (latest start; ties -> earliest end). Per thread, the
    attributed total therefore equals the union of that thread's span
    coverage and can never exceed wall time.

    Device-timed spans (see is_device_timed), Annotation spans and
    zero-duration samples are not host work and are left out."""
    return ExclusiveTime(spans).owned


class ExclusiveTime:
    """Classification + exclusive attribution computed ONCE for a span
    list, so every consumer (time breakdown, wait %, diagnosis, findings)
    shares one O(n log n) pass instead of redoing it per metric."""

    def __init__(self, spans: list) -> None:
        import heapq

        self.spans = spans
        self.bucket = [bucket_of_span(s) for s in spans]
        self.device = [is_device_timed(s) for s in spans]
        self.threads: set[tuple[int, int]] = set()
        by_thread: dict[tuple[int, int], list[int]] = {}
        sampled: list[int] = []
        for i, s in enumerate(spans):
            if s.duration_ns <= 0 or self.device[i] or self.bucket[i] == "Annotation":
                continue
            key = (s.pid, s.tid)
            self.threads.add(key)
            if is_sampled(s):
                sampled.append(i)
            else:
                by_thread.setdefault(key, []).append(i)

        self.owned: dict[int, int] = {}
        owned = self.owned
        heappush, heappop = heapq.heappush, heapq.heappop
        for idxs in by_thread.values():
            boundaries = []   # (time, kind 0=end/1=start, span index)
            for i in idxs:
                s = spans[i]
                boundaries.append((s.start_ns, 1, i))
                boundaries.append((s.start_ns + s.duration_ns, 0, i))
            boundaries.sort()
            active: list[tuple[int, int, int]] = []   # (-start, end, index)
            ended: set[int] = set()
            prev_t = None
            for t, kind, i in boundaries:
                if prev_t is not None and t > prev_t:
                    while active and active[0][2] in ended:
                        heappop(active)
                    if active:
                        owner = id(spans[active[0][2]])
                        owned[owner] = owned.get(owner, 0) + (t - prev_t)
                prev_t = t
                if kind == 1:
                    s = spans[i]
                    heappush(active, (-s.start_ns, s.start_ns + s.duration_ns, i))
                else:
                    ended.add(i)

        # Sampled estimates (perf samples, nominal one-interval weight) never
        # nest with measured spans: a sample taken while its thread was
        # inside an instrumented call is already accounted for by that
        # call's measured duration, so it contributes nothing; samples
        # outside every instrumented span stand for uninstrumented CPU work
        # and keep their nominal weight.
        import bisect
        covered: dict[tuple[int, int], tuple[list[int], list[int]]] = {}
        for key, idxs in by_thread.items():
            ivs = sorted((spans[i].start_ns, spans[i].start_ns + spans[i].duration_ns) for i in idxs)
            starts: list[int] = []
            ends: list[int] = []
            for lo, hi in ivs:
                if ends and lo <= ends[-1]:
                    ends[-1] = max(ends[-1], hi)
                else:
                    starts.append(lo)
                    ends.append(hi)
            covered[key] = (starts, ends)
        for i in sampled:
            s = spans[i]
            starts, ends = covered.get((s.pid, s.tid), ([], []))
            k = bisect.bisect_right(starts, s.start_ns) - 1
            if k >= 0 and s.start_ns < ends[k]:
                continue
            owned[id(s)] = s.duration_ns

    def totals(self, key, *, include_device: bool = True) -> dict:
        """Attributed time per key(span): host spans contribute their
        exclusive time, device-timed spans their full duration (device
        activity doesn't nest under host threads). Annotation spans and
        zero-duration samples contribute nothing."""
        out: dict = {}
        owned = self.owned
        for i, s in enumerate(self.spans):
            if s.duration_ns <= 0 or self.bucket[i] == "Annotation":
                continue
            if self.device[i]:
                if not include_device:
                    continue
                ns = s.duration_ns
            else:
                ns = owned.get(id(s), 0)
            if ns > 0:
                k = key(i, s)
                out[k] = out.get(k, 0) + ns
        return out

    def bucket_totals(self) -> dict[str, int]:
        return self.totals(lambda i, s: self.bucket[i])


def exclusive_totals(spans: list, key, *, include_device: bool = True) -> dict:
    """ExclusiveTime(spans).totals with a plain key(span) function."""
    return ExclusiveTime(spans).totals(lambda i, s: key(s), include_device=include_device)


def bucket_totals(spans: list, *, idle_ns: int = 0, et: "ExclusiveTime | None" = None) -> dict[str, int]:
    """Attributed time per bucket (see exclusive_totals), Annotation
    excluded entirely (see module docstring). Previously a plain sum of
    durations: for the same OpenMP program, barrier time nested inside a
    per-thread region span (GNU libgomp hook) was counted as BOTH
    Computation and Synchronization, so the breakdown and the resulting
    diagnosis depended on which OpenMP runtime the binary linked against.
    Callers that want a derived Idle slice (e.g. GPU launch-gap time, which
    isn't a span) pass `idle_ns`."""
    totals = (et or ExclusiveTime(spans)).bucket_totals()
    if idle_ns > 0:
        totals["Idle"] = totals.get("Idle", 0) + idle_ns
    return totals
