"""
Device capability queries for CUDA, ROCm, and CPU.

Uses the driver API (libcuda.so / libamdhip64.so) via ctypes — no CUPTI
or extra dependencies needed. Falls back to CLI tools (nvidia-smi, lscpu)
when the driver library is unavailable.
"""

from __future__ import annotations
import re
import subprocess
from dataclasses import dataclass, field
from typing import Optional


# ── CUDA cores per SM by compute capability ──────────────────────────────────
_CUDA_CORES_PER_SM: dict[tuple[int, int], int] = {
    (3, 0): 192, (3, 5): 192, (3, 7): 192,   # Kepler
    (5, 0): 128, (5, 2): 128,                  # Maxwell
    (6, 0):  64, (6, 1): 128, (6, 2): 128,    # Pascal
    (7, 0):  64, (7, 5):  64,                  # Volta / Turing
    (8, 0):  64, (8, 6): 128, (8, 7): 128,    # Ampere A100 / GA106+
    (8, 9): 128,                               # Ada Lovelace
    (9, 0): 128,                               # Hopper
}

def _cuda_cores_per_sm(major: int, minor: int) -> int:
    return _CUDA_CORES_PER_SM.get(
        (major, minor),
        _CUDA_CORES_PER_SM.get((major, 0), 64),
    )


# FP64 throughput as a fraction of FP32 peak, by compute capability.
# Consumer GPUs intentionally ship with crippled FP64 (1/32 or 1/64 of FP32).
# Datacenter / workstation SKUs (V100, A100, H100) retain full FP64.
# Where the same cc covers both (e.g. 7.5 Turing), use the conservative consumer ratio.
_CUDA_FP64_RATIO: dict[tuple[int, int], float] = {
    (7, 0): 1 / 2,    # Volta: V100/Titan V = fp32 / 2
    (7, 5): 1 / 32,   # Turing consumer (RTX 2000); Quadro is /16 but we can't distinguish
    (8, 0): 1 / 2,    # Ampere A100 = fp32 / 2
    (8, 6): 1 / 64,   # Ampere consumer (RTX 3000 / GA104)
    (8, 7): 1 / 64,   # Ampere Jetson / RTX high-end GA102
    (8, 9): 1 / 64,   # Ada Lovelace consumer (RTX 4000)
    (9, 0): 1 / 2,    # Hopper H100 / H200
}

def _cuda_fp64_ratio(major: int, minor: int) -> float:
    return _CUDA_FP64_RATIO.get(
        (major, minor),
        _CUDA_FP64_RATIO.get((major, 0), 1 / 32),   # safe default: consumer-class
    )


@dataclass
class DevicePeak:
    """Theoretical peak capabilities of a single compute device."""
    name:           str
    backend:        str     # "cuda", "rocm", "cpu"
    fp32_tflops:    float   # peak FP32 TFLOPs/s
    fp64_tflops:    float
    fp16_tflops:    float
    bandwidth_gbs:  float   # peak memory bandwidth GB/s
    sm_count:       int     # SMs (GPU) or logical cores (CPU)
    core_clock_ghz: float
    mem_clock_ghz:  float
    mem_bus_bits:   int
    vram_gb:        float
    compute_cap:    str     # e.g. "8.0" or "gfx908"
    tensor_tflops:  float = 0.0  # tensor core peak if available (FP16/BF16)

    @property
    def ridge_point(self) -> float:
        """Arithmetic intensity (FLOPs/byte) at the roofline knee."""
        bw = self.bandwidth_gbs * 1e9
        peak = self.fp32_tflops * 1e12
        return peak / bw if bw > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name, "backend": self.backend,
            "fp32_tflops": self.fp32_tflops, "fp64_tflops": self.fp64_tflops,
            "fp16_tflops": self.fp16_tflops, "bandwidth_gbs": self.bandwidth_gbs,
            "sm_count": self.sm_count, "core_clock_ghz": self.core_clock_ghz,
            "mem_clock_ghz": self.mem_clock_ghz, "mem_bus_bits": self.mem_bus_bits,
            "vram_gb": self.vram_gb, "compute_cap": self.compute_cap,
            "tensor_tflops": self.tensor_tflops,
        }

    @staticmethod
    def from_dict(d: dict) -> "DevicePeak":
        return DevicePeak(
            name=d.get("name", ""),
            backend=d.get("backend", ""),
            fp32_tflops=float(d.get("fp32_tflops", 0)),
            fp64_tflops=float(d.get("fp64_tflops", 0)),
            fp16_tflops=float(d.get("fp16_tflops", 0)),
            bandwidth_gbs=float(d.get("bandwidth_gbs", 0)),
            sm_count=int(d.get("sm_count", 0)),
            core_clock_ghz=float(d.get("core_clock_ghz", 0)),
            mem_clock_ghz=float(d.get("mem_clock_ghz", 0)),
            mem_bus_bits=int(d.get("mem_bus_bits", 0)),
            vram_gb=float(d.get("vram_gb", 0)),
            compute_cap=d.get("compute_cap", ""),
            tensor_tflops=float(d.get("tensor_tflops", 0)),
        )


def query_cuda_devices() -> list[DevicePeak]:
    """Query CUDA devices via the driver API using ctypes."""
    try:
        import ctypes

        for libname in ("libcuda.so.1", "libcuda.so"):
            try:
                cuda = ctypes.CDLL(libname)
                break
            except OSError:
                continue
        else:
            return []

        if cuda.cuInit(0) != 0:
            return []

        count = ctypes.c_int(0)
        if cuda.cuDeviceGetCount(ctypes.byref(count)) != 0 or count.value == 0:
            return []

        devices: list[DevicePeak] = []
        for i in range(count.value):
            dev = ctypes.c_int(i)

            name_buf = ctypes.create_string_buffer(256)
            cuda.cuDeviceGetName(name_buf, 256, dev)
            name = name_buf.value.decode("utf-8", errors="replace").strip()

            def _attr(attr_id: int) -> int:
                v = ctypes.c_int(0)
                cuda.cuDeviceGetAttribute(ctypes.byref(v), attr_id, dev)
                return v.value

            # Stable CU_DEVICE_ATTRIBUTE_* ids
            sm_count       = _attr(16)   # MULTIPROCESSOR_COUNT
            core_clock_khz = _attr(13)   # CLOCK_RATE (kHz)
            mem_clock_khz  = _attr(36)   # MEMORY_CLOCK_RATE (kHz)
            mem_bus_bits   = _attr(37)   # GLOBAL_MEMORY_BUS_WIDTH (bits)
            major          = _attr(75)   # COMPUTE_CAPABILITY_MAJOR
            minor          = _attr(76)   # COMPUTE_CAPABILITY_MINOR

            total_mem = ctypes.c_size_t(0)
            cuda.cuDeviceTotalMem(ctypes.byref(total_mem), dev)

            cores_sm       = _cuda_cores_per_sm(major, minor)
            core_clock_ghz = core_clock_khz / 1e6
            # Peak FP32: SMs × cores/SM × 2 (FMA = mul+add) × clock
            fp32_tflops   = sm_count * cores_sm * 2 * core_clock_ghz / 1000
            fp64_tflops   = fp32_tflops * _cuda_fp64_ratio(major, minor)
            fp16_tflops   = fp32_tflops * 2

            # Tensor cores: rough estimate from known GPU families
            tensor_tflops = 0.0
            if (major, minor) == (7, 0):   # Volta: 8 TC/SM, 64 FP16 FLOPs/TC/clock
                tensor_tflops = sm_count * 8 * 64 * core_clock_ghz / 1000
            elif major >= 8:               # Ampere+: ~4× FP32 peak in BF16
                tensor_tflops = fp32_tflops * 4

            mem_clock_ghz  = mem_clock_khz / 1e6
            bandwidth_gbs  = 2 * mem_clock_ghz * mem_bus_bits / 8  # DDR ×2

            devices.append(DevicePeak(
                name=name or f"CUDA device {i}",
                backend="cuda",
                fp32_tflops=fp32_tflops,
                fp64_tflops=fp64_tflops,
                fp16_tflops=fp16_tflops,
                bandwidth_gbs=bandwidth_gbs,
                sm_count=sm_count,
                core_clock_ghz=core_clock_ghz,
                mem_clock_ghz=mem_clock_ghz,
                mem_bus_bits=mem_bus_bits,
                vram_gb=total_mem.value / 1e9,
                compute_cap=f"{major}.{minor}",
                tensor_tflops=tensor_tflops,
            ))
        return devices
    except Exception:
        return []


def query_rocm_devices() -> list[DevicePeak]:
    """Query ROCm devices via the HIP runtime API using ctypes."""
    try:
        import ctypes

        for libname in ("libamdhip64.so", "libhip_hcc.so"):
            try:
                hip = ctypes.CDLL(libname)
                break
            except OSError:
                continue
        else:
            return []

        count = ctypes.c_int(0)
        if hip.hipGetDeviceCount(ctypes.byref(count)) != 0 or count.value == 0:
            return []

        devices: list[DevicePeak] = []
        for i in range(count.value):
            def _attr(attr_id: int) -> int:
                v = ctypes.c_int(0)
                hip.hipDeviceGetAttribute(ctypes.byref(v), attr_id, i)
                return v.value

            # hipDeviceAttribute_t stable ids
            sm_count       = _attr(17)   # hipDeviceAttributeMultiprocessorCount
            core_clock_khz = _attr(8)    # hipDeviceAttributeClockRate
            mem_clock_khz  = _attr(31)   # hipDeviceAttributeMemoryClockRate
            mem_bus_bits   = _attr(32)   # hipDeviceAttributeMemoryBusWidth
            gfx_major      = _attr(87)   # hipDeviceAttributeComputeCapabilityMajor
            gfx_minor      = _attr(88)   # hipDeviceAttributeComputeCapabilityMinor

            name_buf = ctypes.create_string_buffer(256)
            hip.hipDeviceGetName(name_buf, 256, i)
            name = name_buf.value.decode("utf-8", errors="replace").strip()

            total_mem = ctypes.c_size_t(0)
            hip.hipDeviceTotalMem(ctypes.byref(total_mem), i)

            # AMD GCN/RDNA: 64 shader processors per CU.
            # CDNA2 (MI200 series, gfx90a) has 2 shader engines per CU = 128 SPs/CU.
            gfx_str = f"gfx{gfx_major}{gfx_minor:02d}"
            shaders_per_cu = 128 if gfx_str in ("gfx90a", "gfx940", "gfx941", "gfx942") else 64
            core_clock_ghz = core_clock_khz / 1e6
            fp32_tflops    = sm_count * shaders_per_cu * 2 * core_clock_ghz / 1000
            # CDNA2 retains full FP64 (fp32/2); consumer RDNA is fp32/16
            fp64_ratio     = 0.5 if gfx_str in ("gfx90a", "gfx940", "gfx941", "gfx942") else 1 / 16
            fp64_tflops    = fp32_tflops * fp64_ratio
            fp16_tflops    = fp32_tflops * 2

            mem_clock_ghz  = mem_clock_khz / 1e6
            bandwidth_gbs  = 2 * mem_clock_ghz * mem_bus_bits / 8

            devices.append(DevicePeak(
                name=name or f"ROCm device {i}",
                backend="rocm",
                fp32_tflops=fp32_tflops,
                fp64_tflops=fp64_tflops,
                fp16_tflops=fp16_tflops,
                bandwidth_gbs=bandwidth_gbs,
                sm_count=sm_count,
                core_clock_ghz=core_clock_ghz,
                mem_clock_ghz=mem_clock_ghz,
                mem_bus_bits=mem_bus_bits,
                vram_gb=total_mem.value / 1e9,
                compute_cap=f"gfx{gfx_major}{gfx_minor:02d}",
            ))
        return devices
    except Exception:
        return []


def query_cpu_device() -> Optional[DevicePeak]:
    """Query CPU capabilities from lscpu and /proc/cpuinfo."""
    try:
        lscpu = subprocess.run(
            ["lscpu"], capture_output=True, text=True, timeout=5
        ).stdout

        def _field(pat: str) -> str:
            m = re.search(pat, lscpu, re.I | re.M)
            return m.group(1).strip() if m else ""

        name        = _field(r"Model name\s*:\s*(.+)")
        cores_str   = _field(r"^CPU\(s\)\s*:\s*(\d+)")
        max_mhz_str = _field(r"CPU max MHz\s*:\s*([\d.]+)")
        cur_mhz_str = _field(r"CPU MHz\s*:\s*([\d.]+)")

        cores      = int(cores_str)  if cores_str  else 1
        clock_ghz  = float(max_mhz_str or cur_mhz_str or "3000") / 1000

        # Vector width from /proc/cpuinfo flags
        cpuinfo = subprocess.run(
            ["grep", "-m1", "^flags", "/proc/cpuinfo"],
            capture_output=True, text=True, timeout=3,
        ).stdout.lower()

        if "avx512f" in cpuinfo:
            vec_fp32 = 16    # 512-bit / 32-bit
        elif "avx2" in cpuinfo or "avx" in cpuinfo:
            vec_fp32 = 8     # 256-bit / 32-bit
        elif "sse4" in cpuinfo:
            vec_fp32 = 4
        else:
            vec_fp32 = 1

        # Peak FP32: cores × clock_GHz × SIMD_width × 2 (FMA)
        fp32_tflops = cores * clock_ghz * vec_fp32 * 2 / 1000
        fp64_tflops = fp32_tflops / 2  # FP64 SIMD width is half

        # Memory bandwidth: try dmidecode, fall back to conservative estimate
        bw_gbs = 50.0
        try:
            dmi = subprocess.run(
                ["dmidecode", "-t", "memory"],
                capture_output=True, text=True, timeout=5,
            ).stdout
            # Find the fastest DIMM speed and total channel width
            speeds  = [int(m) for m in re.findall(r"Speed:\s*(\d+)\s*MT/s", dmi)]
            widths  = [int(m) for m in re.findall(r"Data Width:\s*(\d+)\s*bits", dmi)]
            slots   = len(re.findall(r"Size:\s*\d+\s*[MG]B", dmi))
            if speeds and widths:
                # Estimate: assume dual-channel (slots/2) at max speed
                channels = max(slots // 2, 1)
                bw_gbs = max(speeds) * max(widths) / 8 * channels / 1000
        except Exception:
            pass

        return DevicePeak(
            name=name or "CPU",
            backend="cpu",
            fp32_tflops=fp32_tflops,
            fp64_tflops=fp64_tflops,
            fp16_tflops=fp32_tflops,
            bandwidth_gbs=bw_gbs,
            sm_count=cores,
            core_clock_ghz=clock_ghz,
            mem_clock_ghz=0.0,
            mem_bus_bits=0,
            vram_gb=0.0,
            compute_cap="",
        )
    except Exception:
        return None


def query_devices(backends: list[str]) -> list[DevicePeak]:
    """Query all relevant device peaks for the given backend set."""
    devices: list[DevicePeak] = []
    if "cuda" in backends:
        devices.extend(query_cuda_devices())
    if "rocm" in backends:
        devices.extend(query_rocm_devices())
    if any(b in backends for b in ("cpu", "openmp", "opencl")):
        cpu = query_cpu_device()
        if cpu:
            devices.append(cpu)
    return devices
