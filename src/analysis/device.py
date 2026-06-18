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


# L2 cache bandwidth (GB/s) by CUDA compute capability.
# Values are from NVIDIA architecture whitepapers and benchmarked measurements.
# Used for multi-level roofline L2 ceiling.
_CUDA_L2_BW_GBS: dict[tuple[int, int], float] = {
    (6, 0): 2000.0,   # P100 (GP100, HBM2, 4 MB L2)
    (6, 1): 400.0,    # GTX 1080 (GP104, 2 MB L2)
    (7, 0): 3072.0,   # V100 (GV100, HBM2, 6 MB L2)
    (7, 5): 1700.0,   # RTX 2080 Ti (TU102, 6 MB L2)
    (8, 0): 6144.0,   # A100 (GA100, HBM2e, 40/80 MB L2)
    (8, 6): 1560.0,   # RTX 3090 (GA102, GDDR6X, 6 MB L2)
    (8, 9): 2400.0,   # RTX 4090 (AD102, GDDR6X, 72 MB L2)
    (9, 0): 12288.0,  # H100 (GH100, HBM3, 50 MB L2)
}


def _cuda_l2_bw_gbs(major: int, minor: int, dram_bw_gbs: float) -> float:
    v = _CUDA_L2_BW_GBS.get((major, minor),
        _CUDA_L2_BW_GBS.get((major, 0), 0.0))
    return v if v > 0 else dram_bw_gbs * 4   # fallback: 4× DRAM bandwidth


# L2 bandwidth (GB/s) by AMD GFX version.
_ROCM_L2_BW_GBS: dict[str, float] = {
    "gfx908": 1600.0,   # MI100 (CDNA1)
    "gfx90a": 6400.0,   # MI200/MI250 (CDNA2)
    "gfx940": 6400.0,   # MI300A (CDNA3)
    "gfx941": 6400.0,
    "gfx942": 6400.0,
    "gfx1030": 860.0,   # RX 6800 XT (RDNA2)
    "gfx1100": 1920.0,  # RX 7900 XTX (RDNA3)
    "gfx1101": 960.0,   # RX 7800 XT (RDNA3)
}


def _rocm_l2_bw_gbs(compute_cap: str, dram_bw_gbs: float) -> float:
    v = _ROCM_L2_BW_GBS.get(compute_cap.lower(), 0.0)
    return v if v > 0 else dram_bw_gbs * 4   # fallback: 4× DRAM bandwidth


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
    l2_bandwidth_gbs: float = 0.0  # L2 cache peak bandwidth (GPU); 0 = unknown
    l1_bandwidth_gbs: float = 0.0  # L1 cache peak bandwidth (GPU); 0 = unknown
    l3_bandwidth_gbs: float = 0.0  # L3/LLC peak bandwidth (CPU); 0 = unknown

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
            "l2_bandwidth_gbs": self.l2_bandwidth_gbs,
            "l1_bandwidth_gbs": self.l1_bandwidth_gbs,
            "l3_bandwidth_gbs": self.l3_bandwidth_gbs,
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
            l2_bandwidth_gbs=float(d.get("l2_bandwidth_gbs", 0)),
            l1_bandwidth_gbs=float(d.get("l1_bandwidth_gbs", 0)),
            l3_bandwidth_gbs=float(d.get("l3_bandwidth_gbs", 0)),
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
            l2_bw_gbs      = _cuda_l2_bw_gbs(major, minor, bandwidth_gbs)

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
                l2_bandwidth_gbs=l2_bw_gbs,
            ))
        return devices
    except Exception:
        return []


def _rocminfo_gpu_props() -> list[dict]:
    """
    Parse `rocminfo` output and return one dict per GPU agent with keys:
    'arch' (e.g. 'gfx908'), 'name' (marketing name), 'cu_count' (int).
    Returns [] if rocminfo is not available or fails.
    """
    try:
        out = subprocess.run(
            ["rocminfo"], capture_output=True, text=True, timeout=15
        ).stdout
    except Exception:
        return []

    agents: list[dict] = []
    current: dict = {}
    in_agent = False

    for line in out.splitlines():
        stripped = line.strip()
        # Agent separator: a row of asterisks
        if re.match(r'^\*{3,}\s*$', stripped):
            if current and current.get("Device Type", "").upper() == "GPU":
                agents.append(current)
            current = {}
            in_agent = True
            continue
        if not in_agent:
            continue
        if ":" in stripped:
            key, _, val = stripped.partition(":")
            current[key.strip()] = val.strip()

    if current and current.get("Device Type", "").upper() == "GPU":
        agents.append(current)

    result = []
    for a in agents:
        # 'Name' in rocminfo is the ISA arch name (e.g. 'gfx908')
        arch = a.get("Name", "")
        if not arch.startswith("gfx"):
            # fall back to ISA Info Name: 'amdgcn-amd-amdhsa--gfx908'
            m = re.search(r'gfx[0-9a-f]+', a.get("ISA 1", ""))
            arch = m.group(0) if m else ""
        cu_str = a.get("Compute Unit", "0")
        try:
            cu_count = int(cu_str.split("(")[0])
        except ValueError:
            cu_count = 0
        result.append({
            "arch":     arch,
            "name":     a.get("Marketing Name", a.get("Name", "")),
            "cu_count": cu_count,
        })
    return result


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

        # rocminfo gives reliable arch name (e.g. 'gfx908') and CU count.
        rocminfo = _rocminfo_gpu_props()

        devices: list[DevicePeak] = []
        for i in range(count.value):
            def _attr(attr_id: int, _i: int = i) -> int:
                v = ctypes.c_int(0)
                hip.hipDeviceGetAttribute(ctypes.byref(v), attr_id, _i)
                return v.value

            # Use CUDA-compatible attribute IDs — HIP maintains the same numbering:
            #   16 = MULTIPROCESSOR_COUNT  13 = CLOCK_RATE (kHz)
            #   36 = MEMORY_CLOCK_RATE     37 = GLOBAL_MEMORY_BUS_WIDTH
            # Probe compute capability: ROCm 6.x uses 87/88, ROCm 5.x / CUDA use 75/76.
            sm_count       = _attr(16)
            core_clock_khz = _attr(13)
            mem_clock_khz  = _attr(36)
            mem_bus_bits   = _attr(37)

            gfx_major = gfx_minor = 0
            for maj_id, min_id in ((87, 88), (75, 76)):
                maj = _attr(maj_id)
                if 1 <= maj <= 12:
                    gfx_major = maj
                    gfx_minor = _attr(min_id)
                    break

            name_buf = ctypes.create_string_buffer(256)
            hip.hipDeviceGetName(name_buf, 256, i)
            name = name_buf.value.decode("utf-8", errors="replace").strip()

            total_mem = ctypes.c_size_t(0)
            hip.hipDeviceTotalMem(ctypes.byref(total_mem), i)

            # Prefer rocminfo for arch name (includes stepping, e.g. 'gfx908' not 'gfx900')
            # and CU count (attribute 16 may still fail on some ROCm versions).
            ri = rocminfo[i] if i < len(rocminfo) else {}
            gfx_str = ri.get("arch", "") or f"gfx{gfx_major}{gfx_minor:02d}"
            if not gfx_str.startswith("gfx"):
                gfx_str = f"gfx{gfx_major}{gfx_minor:02d}"
            if ri.get("cu_count", 0) > 0 and sm_count == 0:
                sm_count = ri["cu_count"]
            if ri.get("name"):
                name = ri["name"]

            # Shader processors per CU:
            #   CDNA1 (gfx908 MI100): 64 SPs/CU
            #   CDNA2 (gfx90a MI200/MI250): 128 SPs/CU (2 matrix engines per CU)
            #   CDNA3 (gfx940/941/942 MI300): 128 SPs/CU
            _cdna2_plus = {"gfx90a", "gfx940", "gfx941", "gfx942"}
            shaders_per_cu = 128 if gfx_str in _cdna2_plus else 64
            core_clock_ghz = core_clock_khz / 1e6
            fp32_tflops    = sm_count * shaders_per_cu * 2 * core_clock_ghz / 1000

            # FP64 throughput ratios (relative to FP32 peak):
            #   CDNA1 gfx908  (MI100):  1:2  (fp64 = fp32 / 2)
            #   CDNA2 gfx90a  (MI200):  1:1  (fp64 = fp32)
            #   CDNA3 gfx940+ (MI300):  1:1
            #   RDNA consumer:          1:16
            _full_fp64  = {"gfx90a", "gfx940", "gfx941", "gfx942"}
            _half_fp64  = {"gfx908"}
            fp64_ratio  = (1.0 if gfx_str in _full_fp64 else
                           0.5 if gfx_str in _half_fp64 else
                           1 / 16)
            fp64_tflops = fp32_tflops * fp64_ratio
            fp16_tflops = fp32_tflops * 2

            mem_clock_ghz = mem_clock_khz / 1e6
            bandwidth_gbs = 2 * mem_clock_ghz * mem_bus_bits / 8
            l2_bw_gbs     = _rocm_l2_bw_gbs(gfx_str, bandwidth_gbs)

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
                compute_cap=gfx_str,
                l2_bandwidth_gbs=l2_bw_gbs,
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
        # FP16 native compute only with AVX-512BF16 or AVX-512FP16; zero otherwise
        has_fp16 = "avx512_bf16" in cpuinfo or "avx512fp16" in cpuinfo
        fp16_tflops = fp32_tflops * 2 if has_fp16 else 0.0

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

        # L3 bandwidth: typically 3–5× DRAM BW on modern x86.
        # This is a rough architecture estimate; actual value depends on core count
        # and cache topology. Shown as "~estimate" in the roofline tooltip.
        l3_bw_gbs = bw_gbs * 4.0

        return DevicePeak(
            name=name or "CPU",
            backend="cpu",
            fp32_tflops=fp32_tflops,
            fp64_tflops=fp64_tflops,
            fp16_tflops=fp16_tflops,
            bandwidth_gbs=bw_gbs,
            sm_count=cores,
            core_clock_ghz=clock_ghz,
            mem_clock_ghz=0.0,
            mem_bus_bits=0,
            vram_gb=0.0,
            compute_cap="",
            l3_bandwidth_gbs=l3_bw_gbs,
        )
    except Exception:
        return None


def query_devices(backends: list[str]) -> list[DevicePeak]:
    """Query all relevant device peaks for the given backend set.

    For the opencl backend we also probe the CUDA and ROCm drivers: an OpenCL
    program running on an NVIDIA or AMD GPU benefits from the same roofline /
    peak-info display as a native CUDA/ROCm run.  The device entries are tagged
    with their true backend ("cuda"/"rocm") so the UI can label them correctly.
    """
    devices: list[DevicePeak] = []
    if "cuda" in backends:
        devices.extend(query_cuda_devices())
    if "rocm" in backends:
        devices.extend(query_rocm_devices())

    if "opencl" in backends:
        # Opportunistically probe whichever GPU drivers are present.
        # Skip if already populated (e.g. user also requested cuda/rocm explicitly).
        if not any(d.backend == "cuda" for d in devices):
            devices.extend(query_cuda_devices())
        if not any(d.backend == "rocm" for d in devices):
            devices.extend(query_rocm_devices())

    if any(b in backends for b in ("cpu", "openmp", "opencl")):
        cpu = query_cpu_device()
        if cpu:
            devices.append(cpu)
    return devices
