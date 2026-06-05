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
import shutil
import subprocess
from pathlib import Path
from .base import Backend

_HOOK_LIB = Path(__file__).parent.parent.parent / "build" / "lib" / "libhprofiler_cuda.so"


class CUDABackend(Backend):
    name = "cuda"
    description = "CUDA Runtime + Driver API tracing via LD_PRELOAD"

    def is_available(self) -> bool:
        if not _HOOK_LIB.exists():
            return False
        # Check for libcuda
        for lib in ["/usr/lib/x86_64-linux-gnu/libcuda.so.1",
                    "/usr/local/cuda/lib64/libcuda.so.1"]:
            if Path(lib).exists():
                return True
        try:
            r = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
            return r.returncode == 0
        except Exception:
            return False

    def preload_libs(self) -> list[str]:
        return [str(_HOOK_LIB)] if _HOOK_LIB.exists() else []
