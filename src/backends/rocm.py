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
import re
import subprocess
from pathlib import Path
from .base import Backend

_HOOK_LIB = Path(__file__).parent.parent.parent / "build" / "lib" / "libhprofiler_rocm.so"


def _version_sort_key(path: str) -> tuple[int, ...]:
    """Numeric sort key from a trailing version, e.g. '/opt/rocm-10.0' -> (10, 0).

    Plain string sort would rank '9.0' above '10.0' (lexicographic '9' > '1'),
    picking the wrong "newest" install when both are present.
    """
    m = re.search(r"(\d+(?:\.\d+)*)$", path.rstrip("/"))
    if not m:
        return (0,)
    return tuple(int(x) for x in m.group(1).split("."))


def _find_rocm_root() -> Path:
    """Locate ROCm installation — checks env override, then common cluster paths."""
    env_override = os.environ.get("ROCM_PATH") or os.environ.get("ROCM_HOME")
    if env_override:
        return Path(env_override)
    candidates = [
        "/opt/rocm",
        "/usr/local/rocm",
    ]
    # Also find versioned dirs like /opt/rocm-5.7.0
    import glob
    for pattern in ["/opt/rocm-*", "/usr/local/rocm-*"]:
        hits = sorted(glob.glob(pattern), key=_version_sort_key, reverse=True)  # newest first
        candidates.extend(hits)
    for p in candidates:
        if Path(p).is_dir():
            return Path(p)
    return Path("/opt/rocm")  # fallback (may not exist)

_ROCM_PATH = _find_rocm_root()


def _hip_lib_exists() -> bool:
    """True if libamdhip64.so is findable — checks ROCm root and system lib paths."""
    for lib_dir in [
        _ROCM_PATH / "lib",
        _ROCM_PATH / "lib64",
        Path("/usr/lib/x86_64-linux-gnu"),
        Path("/usr/lib/aarch64-linux-gnu"),
        Path("/usr/lib64"),
        Path("/usr/lib"),
    ]:
        if (lib_dir / "libamdhip64.so").exists():
            return True
        # Versioned symlinks like libamdhip64.so.5
        import glob
        if glob.glob(str(lib_dir / "libamdhip64.so.*")):
            return True
        if (lib_dir / "libhip_hcc.so").exists():
            return True
    return False


class ROCmBackend(Backend):
    name = "rocm"
    description = "ROCm/HIP kernel tracing via LD_PRELOAD hook"

    def is_available(self) -> bool:
        if not _HOOK_LIB.exists():
            return False
        return _hip_lib_exists()

    def preload_libs(self) -> list[str]:
        if _HOOK_LIB.exists():
            return [str(_HOOK_LIB)]
        return []

    def env_vars(self) -> dict[str, str]:
        return {
            "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES", "0"),
        }
