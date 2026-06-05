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
import subprocess
from pathlib import Path
from .base import Backend

_HOOK_LIB = Path(__file__).parent.parent.parent / "build" / "lib" / "libhprofiler_opencl.so"


class OpenCLBackend(Backend):
    name = "opencl"
    description = "OpenCL command-queue profiling via LD_PRELOAD"

    def is_available(self) -> bool:
        if not _HOOK_LIB.exists():
            return False
        for lib in ["/usr/lib/x86_64-linux-gnu/libOpenCL.so.1",
                    "/usr/lib/libOpenCL.so.1",
                    "/opt/rocm/lib/libOpenCL.so.1"]:
            if Path(lib).exists():
                return True
        try:
            r = subprocess.run(["clinfo", "--list"], capture_output=True, timeout=5)
            return r.returncode == 0
        except Exception:
            return False

    def preload_libs(self) -> list[str]:
        return [str(_HOOK_LIB)] if _HOOK_LIB.exists() else []
