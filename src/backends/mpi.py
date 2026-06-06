"""
MPI backend: profiles MPI point-to-point and collective calls using the
PMPI profiling interface.

The hook (hooks/mpi_hook/libhprofiler_mpi.so) provides MPI_* wrappers that
forward to PMPI_* while emitting spans for every MPI call.  Because the
PMPI interface is part of the MPI standard, no LD_PRELOAD trickery is
needed — the library is injected via LD_PRELOAD and the dynamic linker
resolves MPI_* symbols to our wrappers first.

Captured events (category "mpi"):
  MPI_Send / MPI_Recv / MPI_Isend / MPI_Irecv / MPI_Wait / MPI_Waitall
  MPI_Bcast / MPI_Reduce / MPI_Allreduce / MPI_Alltoall / MPI_Allgather
  MPI_Scatter / MPI_Gather / MPI_Barrier / MPI_Scan
  MPI_Put / MPI_Get (one-sided RMA)

Tags on spans:
  type=send|recv|allreduce|...   call type
  bytes=N                        message size in bytes
  rank=N                         this process's MPI rank
  peer=N                         partner rank (point-to-point only)
  tag=N                          MPI message tag

Requirements:
  The profiled program must be linked against an MPI library.
  The hook must be compiled with the same MPI installation.
  hprofiler build   compiles the hook (requires MPI headers on the build host).
"""

from __future__ import annotations
from pathlib import Path
from .base import Backend

_HOOK_LIB = Path(__file__).parent.parent.parent / "build" / "lib" / "libhprofiler_mpi.so"


class MPIBackend(Backend):
    name = "mpi"
    description = "MPI call tracing via PMPI wrapper (Send/Recv/Allreduce/Barrier/…)"

    def is_available(self) -> bool:
        return _HOOK_LIB.exists()

    def availability_note(self) -> str:
        if not _HOOK_LIB.exists():
            return "hook not built — run: hprofiler build  (requires MPI headers)"
        return ""

    def preload_libs(self) -> list[str]:
        return [str(_HOOK_LIB)] if _HOOK_LIB.exists() else []
