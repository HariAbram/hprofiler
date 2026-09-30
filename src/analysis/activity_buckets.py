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
bridge.py, output/summary.py, and analysis/context.py to separate a GPU
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
    "parallel_region", "offload",
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
    return bucket_of(span.category.value, span.tags.get("type", ""))


def bucket_totals(spans: list, *, idle_ns: int = 0) -> dict[str, int]:
    """Summed duration_ns per bucket, Annotation spans excluded entirely
    (see module docstring) -- callers that want to also show a derived Idle
    slice (e.g. GPU launch-gap time, which isn't a span at all) pass
    `idle_ns` and it's added to the "Idle" bucket rather than needing a
    second merge step at every call site."""
    totals: dict[str, int] = {}
    for s in spans:
        if s.duration_ns <= 0:
            continue
        bucket = bucket_of_span(s)
        if bucket == "Annotation":
            continue
        totals[bucket] = totals.get(bucket, 0) + s.duration_ns
    if idle_ns > 0:
        totals["Idle"] = totals.get("Idle", 0) + idle_ns
    return totals
