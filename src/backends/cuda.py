"""
CUDA backend: LD_PRELOAD hook that intercepts CUDA Runtime API calls.

The hook (hooks/cuda_hook/cuda_hook.c) wraps:
  cudaLaunchKernel, cudaMemcpy, cudaMemcpyAsync, cudaDeviceSynchronize,
  cudaMalloc, cudaFree, and cuLaunchKernel (driver API).

Each wrapped call emits a span record over the profiler socket.
CUDA events are used to measure actual GPU execution time asynchronously.

For JIT-compiled kernels (ACPP, nvcc --device-debug), the hook reads
the kernel name from the function pointer via cuFuncGetAttribute / dladdr.
"""

from __future__ import annotations
import glob
import os
import shutil
import subprocess
from pathlib import Path
from .base import Backend

_HOOK_LIB = Path(__file__).parent.parent.parent / "build" / "lib" / "libhprofiler_cuda.so"


def _libcuda_exists() -> bool:
    """True if libcuda (NVIDIA driver) is available on this system.

    Checks in priority order:
    1. ldconfig cache  — authoritative on most Linux distros
    2. Env-var paths   — $CUDA_PATH / $CUDA_HOME / $CUDA_ROOT (common on clusters)
    3. Glob patterns   — covers non-ldconfig setups (containers, sysroot installs)
       including x86_64, aarch64 (Jetson / Grace), and PowerPC system paths
    """
    try:
        out = subprocess.run(
            ["ldconfig", "-p"], capture_output=True, text=True, timeout=5
        ).stdout
        if "libcuda.so" in out:
            return True
    except Exception:
        pass

    candidates = [
        "/usr/lib/x86_64-linux-gnu/libcuda.so*",
        "/usr/lib/aarch64-linux-gnu/libcuda.so*",
        "/usr/lib/powerpc64le-linux-gnu/libcuda.so*",
        "/usr/lib64/libcuda.so*",
        "/usr/local/cuda/lib64/libcuda.so*",
        "/usr/local/cuda/lib/libcuda.so*",
    ]
    for env_var in ("CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"):
        root = os.environ.get(env_var, "")
        if root:
            candidates += [
                f"{root}/lib64/libcuda.so*",
                f"{root}/lib/libcuda.so*",
                f"{root}/targets/*/lib/libcuda.so*",
            ]
    return any(glob.glob(p) for p in candidates)


class CUDABackend(Backend):
    name = "cuda"
    description = "CUDA Runtime + Driver API tracing via LD_PRELOAD"

    def is_available(self) -> bool:
        if not _HOOK_LIB.exists():
            return False
        if _libcuda_exists():
            return True
        try:
            r = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
            return r.returncode == 0
        except Exception:
            return False

    def preload_libs(self) -> list[str]:
        return [str(_HOOK_LIB)] if _HOOK_LIB.exists() else []
