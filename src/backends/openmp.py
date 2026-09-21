"""
OpenMP backend: two independent capture mechanisms, injected together so
whichever OpenMP runtime the profiled binary actually links against gets
covered without needing to know in advance which one that is.

1. OMPT (hooks/ompt_tool/ompt_tool.c), loaded via OMP_TOOL_LIBRARIES.
   Registers callbacks for parallel regions, tasks, thread lifecycle,
   work distribution, and sync regions (barriers, taskwait, ...).
   Requires LLVM libomp (clang's OpenMP runtime) -- GCC libgomp does not
   implement the OMPT 5.0 ABI in typical distro/vendor builds (confirmed
   empirically on this project's own dev machine: `nm -D libgomp.so.1 |
   grep ompt` finds no OMPT symbols exported at all). The profiled binary
   must be linked against libomp.so (not libgomp.so) for this path to
   produce any events.

2. Direct GOMP_* interception (hooks/gomp_hook/gomp_hook.c), LD_PRELOADed
   unconditionally. For binaries linked against GNU's libgomp instead --
   the common case for anything built with plain gcc/gfortran, including
   most HPC-cluster module-system toolchains (e.g. GROMACS on Dardel's
   cpeGNU environment). Works by intercepting libgomp's own public ABI
   directly (the same LD_PRELOAD interposition every other hook in this
   codebase uses), not a vendor tools callback API -- so it has no
   dependency on OMPT support existing at all.

Both are always injected together when this backend is active: each only
has any effect on a binary that actually imports the specific symbols it
intercepts, so on any given run at most one of the two produces events
(matching whichever runtime the binary is really linked against) and the
other is a harmless no-op -- there is no reliable way to know in advance
which one a given binary uses without inspecting it, so this backend
doesn't try to guess and instead covers both.
"""

from __future__ import annotations
import glob
import os
from pathlib import Path
from .base import Backend

_TOOL_LIB = Path(__file__).parent.parent.parent / "build" / "lib" / "libhprofiler_ompt.so"
_GOMP_LIB = Path(__file__).parent.parent.parent / "build" / "lib" / "libhprofiler_gomp.so"


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
        # ROCm's bundled LLVM (e.g. Cray/HPC clusters like Dardel, where the
        # system C compiler is the Cray `cc` wrapper and clang/libomp come
        # from the ROCm install instead of a system package):
        # /opt/rocm-6.3.3/llvm/lib/libomp.so (top-level `llvm` is usually a
        # symlink to lib/llvm, but check both forms in case it isn't).
        "/opt/rocm*/llvm/lib/libomp.so*",
        "/opt/rocm*/lib/llvm/lib/libomp.so*",
    ]
    for env_var in ("ROCM_PATH", "ROCM_HOME", "LLVM_HOME", "LLVM_ROOT", "LLVM_PATH"):
        root = os.environ.get(env_var, "")
        if root:
            patterns += [
                f"{root}/llvm/lib/libomp.so*",
                f"{root}/lib/llvm/lib/libomp.so*",
                f"{root}/lib/libomp.so*",
                f"{root}/lib64/libomp.so*",
            ]
    found = []
    for p in patterns:
        found.extend(glob.glob(p))
    return found


def _install_hint() -> str:
    """Return the right package-manager hint for the current distro."""
    os_release = ""
    try:
        os_release = open("/etc/os-release").read().lower()
    except OSError:
        pass
    # HPC/Cray login nodes rarely have root for a package-manager install —
    # if a ROCm module is loaded (e.g. Dardel/LUMI-style clusters) its
    # bundled LLVM already ships libomp.so, just not on the default search
    # path unless $ROCM_PATH is exported.
    hpc_hint = (
        " — or, on an HPC/Cray cluster, `module load rocm` (or similar) and "
        "`export ROCM_PATH=/opt/rocm-X.Y.Z` to use its bundled libomp"
    )
    if any(x in os_release for x in ("rhel", "rocky", "centos", "fedora", "almalinux")):
        return "dnf install llvm-toolset  (or: yum install llvm)" + hpc_hint
    if "arch" in os_release:
        return "pacman -S openmp" + hpc_hint
    if "suse" in os_release or "opensuse" in os_release:
        return "zypper install libomp-devel" + hpc_hint
    # Default: Debian/Ubuntu
    return "apt install libomp-dev  (then compile binary with clang++)" + hpc_hint


class OpenMPBackend(Backend):
    name = "openmp"
    description = "OpenMP parallel region / task tracing — OMPT (clang libomp) + direct GOMP_* interception (GCC libgomp)"

    def is_available(self) -> bool:
        # gomp_hook.c has no libomp/detection dependency at all (a plain
        # LD_PRELOAD symbol interposer against libgomp's own stable ABI) —
        # available whenever it's built, regardless of whether libomp is
        # findable on this system. OMPT remains additionally available
        # when its own hook is built, independent of the gomp path.
        return _TOOL_LIB.exists() or _GOMP_LIB.exists()

    def libomp_available(self) -> bool:
        return bool(_libomp_paths())

    def availability_note(self) -> str:
        if not _TOOL_LIB.exists() and not _GOMP_LIB.exists():
            return "hooks not built — run: hprofiler build"
        if _TOOL_LIB.exists() and not self.libomp_available():
            return (f"OMPT path (for clang/libomp binaries) needs libomp, not found "
                    f"— install: {_install_hint()} — GCC/libgomp binaries are still "
                    f"covered via direct GOMP_* interception either way")
        return ""

    def env_vars(self) -> dict[str, str]:
        if not _TOOL_LIB.exists():
            return {}
        return {"OMP_TOOL_LIBRARIES": str(_TOOL_LIB)}

    def preload_libs(self) -> list[str]:
        # LD_PRELOAD both hooks unconditionally (whichever are built) — see
        # module docstring for why injecting both, rather than trying to
        # detect which OpenMP runtime the target actually uses, is the
        # robust choice. For OMPT: LD_PRELOAD (in addition to
        # OMP_TOOL_LIBRARIES above) is what puts the dlopen() interposer
        # defined in ompt_tool.c into the global interposition chain —
        # OMP_TOOL_LIBRARIES alone loads it as a dlopen plugin into a
        # private namespace, which does not achieve that.
        libs = []
        if _TOOL_LIB.exists():
            libs.append(str(_TOOL_LIB))
        if _GOMP_LIB.exists():
            libs.append(str(_GOMP_LIB))
        return libs
