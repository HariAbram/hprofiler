"""
OpenCL backend: LD_PRELOAD hook that enables command-queue profiling.

The hook wraps clCreateCommandQueue / clCreateCommandQueueWithProperties
to automatically set CL_QUEUE_PROFILING_ENABLE, then wraps:
  clEnqueueNDRangeKernel, clEnqueueTask,
  clEnqueueCopyBuffer, clEnqueueReadBuffer, clEnqueueWriteBuffer

After each enqueue, a completion callback reads CL_PROFILING_COMMAND_START
and CL_PROFILING_COMMAND_END from the cl_event and emits a span record.

Works for any OpenCL device (CPU or GPU), including ACPP's OpenCL backend.
"""

from __future__ import annotations
import glob
import os
import subprocess
from pathlib import Path
from .base import Backend

_HOOK_LIB = Path(__file__).parent.parent.parent / "build" / "lib" / "libhprofiler_opencl.so"


def _libopencl_exists() -> bool:
    """True if libOpenCL is available on this system.

    Checks in priority order:
    1. ldconfig cache  — authoritative on most Linux distros
    2. Env-var paths   — $OPENCL_ROOT / $CUDA_PATH / $ROCM_PATH (common on clusters)
    3. Glob patterns   — covers non-ldconfig setups (containers, sysroot installs)
       including x86_64, aarch64 (Jetson / Grace), and AMD ROCm paths
    """
    try:
        out = subprocess.run(
            ["ldconfig", "-p"], capture_output=True, text=True, timeout=5
        ).stdout
        if "libOpenCL.so" in out:
            return True
    except Exception:
        pass

    candidates = [
        "/usr/lib/x86_64-linux-gnu/libOpenCL.so*",
        "/usr/lib/aarch64-linux-gnu/libOpenCL.so*",
        "/usr/lib/powerpc64le-linux-gnu/libOpenCL.so*",
        "/usr/lib64/libOpenCL.so*",
        "/usr/lib/libOpenCL.so*",
        "/opt/rocm/lib/libOpenCL.so*",
        "/opt/rocm-*/lib/libOpenCL.so*",
        "/usr/local/cuda/lib64/libOpenCL.so*",
    ]
    for env_var in ("OPENCL_ROOT", "CUDA_PATH", "CUDA_HOME", "ROCM_PATH", "ROCM_HOME"):
        root = os.environ.get(env_var, "")
        if root:
            candidates += [
                f"{root}/lib64/libOpenCL.so*",
                f"{root}/lib/libOpenCL.so*",
            ]
    return any(glob.glob(p) for p in candidates)


class OpenCLBackend(Backend):
    name = "opencl"
    description = "OpenCL command-queue profiling via LD_PRELOAD"

    def is_available(self) -> bool:
        if not _HOOK_LIB.exists():
            return False
        if _libopencl_exists():
            return True
        try:
            r = subprocess.run(["clinfo", "--list"], capture_output=True, timeout=5)
            return r.returncode == 0
        except Exception:
            return False

    def preload_libs(self) -> list[str]:
        return [str(_HOOK_LIB)] if _HOOK_LIB.exists() else []
