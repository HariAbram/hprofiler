"""
OpenMP backend: OMPT (OpenMP Tools Interface, OpenMP 5.0+).

The tool library (hooks/ompt_tool/ompt_tool.c) registers callbacks for:
  - ompt_callback_parallel_begin/end   (parallel regions)
  - ompt_callback_task_create/schedule (task events)
  - ompt_callback_thread_begin/end     (thread lifecycle)
  - ompt_callback_work                 (loop/sections distribution)
  - ompt_callback_sync_region          (barriers, taskwait, etc.)

Loaded via OMP_TOOL_LIBRARIES.

IMPORTANT: requires LLVM libomp (clang's OpenMP runtime). GCC libgomp does
NOT implement the OMPT 5.0 ABI on Ubuntu 24.04 and will silently produce 0
events. The profiled binary must be compiled with clang++ and linked against
libomp.so (not libgomp.so). For ACPP/hipSYCL, use ACPP_VISIBILITY_MASK=omp
with a clang-based toolchain.
"""

from __future__ import annotations
import glob
from pathlib import Path
from .base import Backend

_TOOL_LIB = Path(__file__).parent.parent.parent / "build" / "lib" / "libhprofiler_ompt.so"


def _libomp_paths() -> list[str]:
    """Find installed LLVM libomp shared libraries."""
    patterns = [
        # Debian / Ubuntu
        "/lib/*/libomp.so*",
        "/usr/lib/*/libomp.so*",
        "/usr/lib/libomp.so*",
        "/usr/local/lib/*/libomp.so*",
        "/usr/local/lib/libomp.so*",
        # RHEL / Rocky / CentOS — lib64 flat
        "/usr/lib64/libomp.so*",
        "/usr/lib64/*/libomp.so*",
        # RHEL — LLVM versioned: /usr/lib64/llvm21/lib64/libomp.so
        "/usr/lib64/llvm*/lib64/libomp.so*",
        "/usr/lib64/llvm*/lib/libomp.so*",
        # Red Hat Developer Toolset
        "/opt/rh/llvm-toolset*/root/usr/lib64/libomp.so*",
        # Generic /opt LLVM installs
        "/opt/llvm*/lib/libomp.so*",
        "/opt/llvm*/lib64/libomp.so*",
    ]
    found = []
    for p in patterns:
        found.extend(glob.glob(p))
    return found


def _install_hint() -> str:
    """Return the right package-manager hint for the current distro."""
    import os
    os_release = ""
    try:
        os_release = open("/etc/os-release").read().lower()
    except OSError:
        pass
    if any(x in os_release for x in ("rhel", "rocky", "centos", "fedora", "almalinux")):
        return "dnf install llvm-toolset  (or: yum install llvm)"
    if "arch" in os_release:
        return "pacman -S openmp"
    if "suse" in os_release or "opensuse" in os_release:
        return "zypper install libomp-devel"
    # Default: Debian/Ubuntu
    return "apt install libomp-dev  (then compile binary with clang++)"


class OpenMPBackend(Backend):
    name = "openmp"
    description = "OpenMP parallel region / task tracing via OMPT (requires clang libomp, not GCC libgomp)"

    def is_available(self) -> bool:
        return _TOOL_LIB.exists()

    def libomp_available(self) -> bool:
        return bool(_libomp_paths())

    def availability_note(self) -> str:
        if not _TOOL_LIB.exists():
            return "hook not built — run: hprofiler build"
        if not self.libomp_available():
            return f"libomp not found — install: {_install_hint()}"
        return ""

    def env_vars(self) -> dict[str, str]:
        if not _TOOL_LIB.exists():
            return {}
        return {"OMP_TOOL_LIBRARIES": str(_TOOL_LIB)}
