"""
NCCL backend: profiles NCCL multi-GPU collective operations via LD_PRELOAD.

The hook (hooks/nccl_hook/libhprofiler_nccl.so) intercepts NCCL calls and
records GPU-accurate spans using cudaEvent pairs.

Captured calls (category "nccl"):
  ncclAllReduce, ncclBroadcast, ncclReduce, ncclAllGather, ncclReduceScatter
  ncclSend, ncclRecv
  ncclGroupStart / ncclGroupEnd

Tags:
  type=allreduce|broadcast|...
  bytes=N          message volume in bytes
  stream=ID        CUDA stream index
  peer=N           remote rank (point-to-point only)

Requirements:
  NCCL library (libnccl.so) must be installed and visible to the runtime.
  hprofiler build   compiles the hook (no NCCL headers needed).
"""

from __future__ import annotations
from pathlib import Path
from .base import Backend

_HOOK_LIB = Path(__file__).parent.parent.parent / "build" / "lib" / "libhprofiler_nccl.so"
_NCCL_CANDIDATES = [
    Path("/usr/lib/x86_64-linux-gnu/libnccl.so.2"),
    Path("/usr/local/lib/libnccl.so.2"),
    Path("/usr/local/cuda/lib64/libnccl.so.2"),
    Path("/opt/nccl/lib/libnccl.so.2"),
]


def _nccl_available() -> bool:
    import shutil
    if shutil.which("nccl-version"):
        return True
    return any(p.exists() for p in _NCCL_CANDIDATES)


class NCCLBackend(Backend):
    name = "nccl"
    description = "NCCL collective tracing via LD_PRELOAD (AllReduce, Broadcast, Send/Recv, …)"

    def is_available(self) -> bool:
        return _HOOK_LIB.exists() and _nccl_available()

    def availability_note(self) -> str:
        if not _HOOK_LIB.exists():
            return "hook not built — run: hprofiler build"
        if not _nccl_available():
            return "libnccl not found — install NCCL from developer.nvidia.com/nccl"
        return ""

    def preload_libs(self) -> list[str]:
        return [str(_HOOK_LIB)] if _HOOK_LIB.exists() else []
