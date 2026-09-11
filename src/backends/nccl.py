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
import glob
import os
import subprocess
from pathlib import Path
from .base import Backend

_HOOK_LIB = Path(__file__).parent.parent.parent / "build" / "lib" / "libhprofiler_nccl.so"


def _nccl_available() -> bool:
    """True if libnccl is available on this system.

    Checks in priority order (mirrors cuda.py/opencl.py):
    1. ldconfig cache  — authoritative on most Linux distros
    2. Env-var paths   — $NCCL_ROOT / $NCCL_HOME / $CUDA_PATH (common on clusters)
    3. Glob patterns   — covers non-ldconfig setups (containers, sysroot installs)
       including x86_64, aarch64 (Jetson / Grace), and PowerPC system paths
    """
    try:
        out = subprocess.run(
            ["ldconfig", "-p"], capture_output=True, text=True, timeout=5
        ).stdout
        if "libnccl.so" in out:
            return True
    except Exception:
        pass

    candidates = [
        "/usr/lib/x86_64-linux-gnu/libnccl.so*",
        "/usr/lib/aarch64-linux-gnu/libnccl.so*",
        "/usr/lib/powerpc64le-linux-gnu/libnccl.so*",
        "/usr/lib64/libnccl.so*",
        "/usr/local/lib/libnccl.so*",
        "/usr/local/cuda/lib64/libnccl.so*",
        "/opt/nccl/lib/libnccl.so*",
    ]
    for env_var in ("NCCL_ROOT", "NCCL_HOME", "CUDA_PATH", "CUDA_HOME"):
        root = os.environ.get(env_var, "")
        if root:
            candidates += [
                f"{root}/lib64/libnccl.so*",
                f"{root}/lib/libnccl.so*",
            ]
    return any(glob.glob(p) for p in candidates)


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
