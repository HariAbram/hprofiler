"""
ROCm backend: uses roctracer (libroctracer64.so) via LD_PRELOAD hook,
or falls back to our own HIP API wrapper hook.

The hook intercepts:
  hipLaunchKernel, hipMemcpy, hipMemcpyAsync, hipMalloc, hipFree,
  hipDeviceSynchronize, hipStreamSynchronize.

For ACPP targeting ROCm, kernel names come from the __attribute__((used))
kernel wrapper symbols emitted by ACPP's SSCP compilation.
"""

from __future__ import annotations
import os
import subprocess
from pathlib import Path
from .base import Backend

_HOOK_LIB = Path(__file__).parent.parent.parent / "build" / "lib" / "libhprofiler_rocm.so"
_ROCM_PATH = Path("/opt/rocm")


class ROCmBackend(Backend):
    name = "rocm"
    description = "ROCm/HIP kernel tracing via roctracer + LD_PRELOAD"

    def is_available(self) -> bool:
        if not _HOOK_LIB.exists():
            return False
        return (_ROCM_PATH / "lib" / "libamdhip64.so").exists() or \
               (_ROCM_PATH / "lib" / "libhip_hcc.so").exists()

    def preload_libs(self) -> list[str]:
        libs: list[str] = []
        roctracer = _ROCM_PATH / "lib" / "libroctracer64.so"
        if roctracer.exists():
            libs.append(str(roctracer))
        if _HOOK_LIB.exists():
            libs.append(str(_HOOK_LIB))
        return libs

    def env_vars(self) -> dict[str, str]:
        return {
            "ROCTRACER_DOMAIN": "hip",
            "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES", "0"),
        }
