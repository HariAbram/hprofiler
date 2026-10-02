"""
Synthetic before/after trace pairs for the comparison engine
(src/analysis/causal_compare.py): each builder returns (baseline,
candidate) Traces whose difference is one known, hand-checkable cause.
GPU scenarios use the exact record format the CUDA hook / CUPTI decoder
emit (tests/test_gpu_activity.py helpers) and go through the same
correlation step (gpu_activity.assemble) as a real run.
"""
from __future__ import annotations

from src.core.events import Category, SpanEvent
from src.core.trace import Trace, TraceMetadata

MS = 1_000_000
US = 1_000


def _span(t: Trace, name, cat, start, dur, pid=1, tid=1, **tags):
    t.add(SpanEvent(name, cat, int(start), int(dur), pid, tid, dict(tags)))


def same_name_different_paths(slow_y: bool, iters: int = 5) -> Trace:
    """`update` runs under solve_x and under solve_y every iteration; the
    candidate slows down only the solve_y one."""
    t = Trace(TraceMetadata(command="./solver"))
    c = 1 * MS
    for _ in range(iters):
        _span(t, "solve_x", Category.NVTX, c, 2.2 * MS, type="nvtx_range")
        _span(t, "update", Category.OPENMP, c + 0.1 * MS, 2 * MS, type="work", file="/src/solver.c", line=40)
        c += 2.3 * MS
        dy = (4 if slow_y else 2) * MS
        _span(t, "solve_y", Category.NVTX, c, dy + 0.2 * MS, type="nvtx_range")
        _span(t, "update", Category.OPENMP, c + 0.1 * MS, dy, type="work", file="/src/solver.c", line=40)
        c += dy + 0.3 * MS
    return t


def reordered(swap: bool, iters: int = 5) -> Trace:
    """Two independent calls per iteration; the candidate runs them in the
    opposite order (same durations)."""
    t = Trace(TraceMetadata(command="./app"))
    c = 1 * MS
    for _ in range(iters):
        _span(t, "step", Category.NVTX, c, 5.4 * MS, type="nvtx_range")
        order = [("assemble", 2 * MS), ("factor", 3 * MS)]
        if swap:
            order.reverse()
        x = c + 0.1 * MS
        for name, d in order:
            _span(t, name, Category.OPENMP, x, d, type="work")
            x += d + 0.1 * MS
        c += 5.5 * MS
    return t


def iterations(n: int) -> Trace:
    """n identical iterations of compute + barrier."""
    t = Trace(TraceMetadata(command="./loop"))
    c = 1 * MS
    _span(t, "setup", Category.OPENMP, c, 3 * MS, type="work")
    c += 3.1 * MS
    for _ in range(n):
        _span(t, "compute", Category.OPENMP, c, 2 * MS, type="work")
        _span(t, "omp_barrier", Category.SYNC, c + 2 * MS, 0.5 * MS, type="barrier")
        c += 2.6 * MS
    _span(t, "write_output", Category.OPENMP, c, 1 * MS, type="work")
    return t


def mpi_wait(slow_rank0: bool, iters: int = 5) -> Trace:
    """rank 0 computes then sends to rank 1, which computes then receives;
    the candidate's rank-0 compute is slower, so rank 1 waits longer in
    MPI_Recv without doing any more work itself."""
    t = Trace(TraceMetadata(command="mpirun -np 2 ./halo"))
    c = 1 * MS
    w0 = (20 if slow_rank0 else 10) * MS
    for i in range(iters):
        _span(t, "compute", Category.OPENMP, c, w0, pid=100, tid=100, type="work")
        _span(t, "MPI_Send", Category.MPI, c + w0, 0.1 * MS, pid=100, tid=100,
              type="send", rank="0", peer="1", tag="7", bytes="4096")
        _span(t, "compute", Category.OPENMP, c, 10 * MS, pid=200, tid=200, type="work")
        recv_end = c + w0 + 0.15 * MS
        _span(t, "MPI_Recv", Category.MPI, c + 10 * MS, recv_end - (c + 10 * MS), pid=200, tid=200,
              type="recv", rank="1", peer="0", tag="7", bytes="4096")
        c = recv_end + 0.2 * MS
    return t


def _gpu(lines) -> Trace:
    from tests.test_gpu_activity import build
    t = build(lines)
    t.metadata.command = "./cuda_app"
    return t


def stream_overlap(serialized: bool, iters: int = 5) -> Trace:
    """Kernels A (stream 11) and B (stream 22) overlap; in the candidate B
    is launched on A's stream, so it runs after A."""
    from tests.test_gpu_activity import host, cupti_kernel
    lines = []
    c = 1 * MS
    for i in range(iters):
        lid = 10 * i
        lines.append(host("cudaLaunchKernel", c, 20 * US, lid=lid + 1, corr=lid + 1, stream=11))
        lines.append(host("cudaLaunchKernel", c + 30 * US, 20 * US, lid=lid + 2, corr=lid + 2,
                          stream=11 if serialized else 22))
        a0, a1 = c + 60 * US, c + 60 * US + 4 * MS
        lines.append(cupti_kernel(lid + 1, 7, a0, a1, name="_Z1Av"))
        b0 = a1 if serialized else c + 70 * US
        b1 = b0 + 3 * MS
        lines.append(cupti_kernel(lid + 2, 7 if serialized else 8, b0, b1, name="_Z1Bv"))
        end = max(a1, b1)
        lines.append(host("cudaDeviceSynchronize", c + 60 * US, end + 10 * US - (c + 60 * US), cat="sync",
                          typ="sync", op="sync", lid=lid + 3, corr=lid + 3, sync="device"))
        c = end + 100 * US
    return _gpu(lines)


def becomes_critical(slow_host: bool, iters: int = 5) -> Trace:
    """Kernel L (10 ms, stream 11) hides kernel K (2 ms, stream 22) -- until
    the host work before K's launch grows, so K starts late, outlasts L and
    the final device sync waits for K."""
    from tests.test_gpu_activity import host, cupti_kernel
    lines = []
    c = 1 * MS
    for i in range(iters):
        lid = 10 * i
        lines.append(host("cudaLaunchKernel", c, 20 * US, lid=lid + 1, corr=lid + 1, stream=11))
        l0, l1 = c + 50 * US, c + 50 * US + 10 * MS
        lines.append(cupti_kernel(lid + 1, 7, l0, l1, name="_Z1Lv"))
        h = int((9.5 if slow_host else 1) * MS)
        lines.append(f"span:openmp:1:100:{c + 30 * US}:{int(h)}:prepare:type=work")
        launch = c + 30 * US + h + 10 * US
        lines.append(host("cudaLaunchKernel", launch, 20 * US, lid=lid + 2, corr=lid + 2, stream=22))
        k0 = launch + 40 * US
        k1 = k0 + 2 * MS
        lines.append(cupti_kernel(lid + 2, 8, k0, k1, name="_Z1Kv"))
        sync0 = launch + 30 * US
        end = max(l1, k1)
        lines.append(host("cudaDeviceSynchronize", sync0, end + 10 * US - sync0, cat="sync", typ="sync",
                          op="sync", lid=lid + 3, corr=lid + 3, sync="device"))
        c = end + 100 * US
    return _gpu(lines)
