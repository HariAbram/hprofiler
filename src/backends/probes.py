"""
Cheap "is this backend usable here?" probes.

A library or tool being installed does not make a backend usable: a CUDA
driver/library mismatch makes every CUDA call fail (error 803/804), a desktop
libamdhip64 without an AMD GPU traces nothing, and likwid-perfctr without PMU
access refuses to start the program at all. Each probe does the smallest real
operation that fails in those cases -- in a subprocess, with a timeout, so a
hanging driver cannot hang hprofiler -- and returns (usable, reason). Results
are cached per process.
"""
from __future__ import annotations

import functools
import glob
import os
import shutil
import subprocess
import sys

PROBE_TIMEOUT_S = 15

_CUDA_PROBE = r"""
import ctypes, sys
try:
    lib = ctypes.CDLL("libcuda.so.1")
except OSError as e:
    print("nolib", e); sys.exit(0)
r = lib.cuInit(0)
if r != 0:
    name = ctypes.c_char_p()
    lib.cuGetErrorName(r, ctypes.byref(name))
    print("init", r, (name.value or b"").decode()); sys.exit(0)
n = ctypes.c_int(0)
r = lib.cuDeviceGetCount(ctypes.byref(n))
print("count", r, n.value)
"""

_OPENCL_PROBE = r"""
import ctypes, sys
try:
    lib = ctypes.CDLL("libOpenCL.so.1")
except OSError as e:
    print("nolib", e); sys.exit(0)
n = ctypes.c_uint(0)
r = lib.clGetPlatformIDs(0, None, ctypes.byref(n))
if r != 0 or n.value == 0:
    print("platforms", r, n.value); sys.exit(0)
plats = (ctypes.c_void_p * n.value)()
lib.clGetPlatformIDs(n.value, plats, None)
total = 0
for p in plats:
    d = ctypes.c_uint(0)
    if lib.clGetDeviceIDs(ctypes.c_void_p(p), ctypes.c_ulong(0xFFFFFFFF), 0, None, ctypes.byref(d)) == 0:
        total += d.value
print("devices", 0, total)
"""


def _run_probe(code: str) -> list[str]:
    try:
        r = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True,
                           timeout=PROBE_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return ["timeout"]
    except OSError as e:
        return ["error", str(e)]
    out = (r.stdout or "").strip().split()
    return out or ["error", (r.stderr or "").strip()[:200]]


@functools.lru_cache(maxsize=None)
def cuda() -> tuple[bool, str]:
    """cuInit + cuDeviceGetCount through the driver library."""
    out = _run_probe(_CUDA_PROBE)
    if out[0] == "nolib":
        return False, "libcuda.so.1 cannot be loaded (no NVIDIA driver)"
    if out[0] == "init":
        return False, f"cuInit failed: {out[1]} {out[2] if len(out) > 2 else ''}".strip() + \
            " (driver/library version mismatch or no device)"
    if out[0] == "count":
        if out[1] != "0":
            return False, f"cuDeviceGetCount failed: {out[1]}"
        n = int(out[2])
        return (n > 0, f"{n} CUDA device(s)" if n else "no CUDA device")
    return False, f"CUDA probe failed ({' '.join(out)[:120]})"


@functools.lru_cache(maxsize=None)
def rocm() -> tuple[bool, str]:
    """/dev/kfd accessible and at least one GPU agent in the KFD topology
    (no HIP initialization)."""
    if not os.path.exists("/dev/kfd"):
        return False, "no /dev/kfd (no AMD GPU kernel driver)"
    if not os.access("/dev/kfd", os.R_OK | os.W_OK):
        return False, "/dev/kfd not accessible (add the user to the render/video group)"
    gpus = 0
    for props in glob.glob("/sys/class/kfd/kfd/topology/nodes/*/properties"):
        try:
            for line in open(props):
                key, _, val = line.partition(" ")
                if key == "simd_count" and int(val or 0) > 0:
                    gpus += 1
                    break
        except (OSError, ValueError):
            continue
    return (gpus > 0, f"{gpus} AMD GPU agent(s)" if gpus else "no GPU agent in /sys/class/kfd topology")


@functools.lru_cache(maxsize=None)
def opencl() -> tuple[bool, str]:
    """At least one OpenCL device on any platform."""
    out = _run_probe(_OPENCL_PROBE)
    if out[0] == "nolib":
        return False, "libOpenCL.so.1 cannot be loaded"
    if out[0] == "devices":
        n = int(out[2])
        return (n > 0, f"{n} OpenCL device(s)" if n else "no OpenCL device")
    if out[0] == "platforms":
        return False, "no OpenCL platform (no ICD installed)"
    return False, f"OpenCL probe failed ({' '.join(out)[:120]})"


@functools.lru_cache(maxsize=None)
def likwid(group: str) -> tuple[bool, str]:
    """likwid-perfctr measuring `true` on core 0: fails without PMU access."""
    exe = shutil.which("likwid-perfctr")
    if not exe:
        return False, "likwid-perfctr not found"
    try:
        r = subprocess.run([exe, "-C", "0", "-g", group, "true"], capture_output=True, text=True,
                           timeout=PROBE_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return False, "likwid-perfctr access test timed out"
    except OSError as e:
        return False, f"likwid-perfctr cannot run: {e}"
    if r.returncode == 0:
        return True, f"PMU access ok (group {group})"
    msg = next((ln.strip() for ln in (r.stderr + r.stdout).splitlines() if ln.strip()), "")
    try:
        paranoid = open("/proc/sys/kernel/perf_event_paranoid").read().strip()
    except OSError:
        paranoid = "?"
    return False, f"no PMU access (perf_event_paranoid={paranoid}): {msg[:120]}"
