"""
Runtime semantics: what a call means for the dependency graph.

blocking_wait(span): the call returns only once something outside its own
thread has happened -- a message arrived, every member reached a
collective or barrier, child tasks finished, a device synchronized. A
recorded dependency (an edge into its end) says what; without one, its
time is waiting for something the trace did not record, or processing:
the model cannot tell (cp_model's "unknown" mode), never plain work.

The per-runtime modules hold the runtimes' state machines (mpi, openmp,
gpu, opencl, nccl): which events of which objects order which others.
"""
from __future__ import annotations

# MPI calls that wait: receives (and the receive half of Sendrecv),
# completions, blocking probes, synchronous sends (they return once the
# receive has started), collectives and barriers.
MPI_BLOCKING_TYPES = frozenset({
    "recv", "sendrecv", "sendrecv_replace", "ssend", "wait", "waitall", "waitany", "waitsome",
    "probe", "mrecv",
    "allreduce", "bcast", "reduce", "alltoall", "allgather", "scatter", "gather", "barrier", "scan",
    "exscan", "allgatherv", "alltoallv", "gatherv", "scatterv", "reduce_scatter", "reduce_scatter_block",
    "win_fence", "win_flush", "win_flush_all", "win_unlock", "win_unlock_all",
})

# Host calls that synchronize with a device (CUDA, HIP, OpenCL)
DEVICE_SYNC_NAMES = frozenset({
    "cudaDeviceSynchronize", "cudaStreamSynchronize", "cudaEventSynchronize", "cuCtxSynchronize",
    "cuStreamSynchronize", "cuEventSynchronize",
    "hipDeviceSynchronize", "hipStreamSynchronize", "hipEventSynchronize",
    "clFinish", "clWaitForEvents",
})


def blocking_wait(span) -> bool:
    """Whether `span` is a call that waits for another lane (see module
    docstring). Device-timed spans are device work, never host waits."""
    tags = span.tags
    if tags.get("side") == "gpu":
        return False
    cat = span.category.value
    if cat == "mpi":
        return tags.get("type") in MPI_BLOCKING_TYPES
    if cat == "sync":
        name = span.name
        if name.startswith("omp_"):
            # barriers, taskwait/taskgroup, lock / critical / ordered waits
            return name.startswith("omp_barrier") or name in ("omp_taskwait", "omp_taskgroup", "omp_reduction") \
                or name.endswith("_wait")
        if name.startswith("GOMP_"):
            return True
        return name in DEVICE_SYNC_NAMES or tags.get("type") == "sync"
    return False
