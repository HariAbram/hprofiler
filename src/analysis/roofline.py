"""
Roofline model analysis.

Estimates per-kernel FLOPs and memory bytes from disasm instruction-type
counts × thread count (grid × block), then compares achieved rates against
device theoretical peaks.

The estimates are approximate — hardware counters (CUPTI, rocProfiler) would
be more accurate — but they reliably identify whether a kernel is memory-bound
or compute-bound and show how far it sits from the roofline ceiling.
"""

from __future__ import annotations
import math
import re
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from .device import DevicePeak
from ..disasm.classifier import InsnType

if TYPE_CHECKING:
    from ..core.trace import Trace
    from ..core.events import SpanEvent
    from ..disasm.extractor import KernelDisasm


# ── FLOPs per instruction ─────────────────────────────────────────────────────
# Values are estimates for the dominant data type (FP32).
# VECTOR: assumes 256-bit SIMD (8-wide FP32 FMA) for x86; 4-wide v_ for AMDGCN.
# COMPUTE: scalar FMA = 2 FLOPs (mul + add).

_FLOPS: dict[str, dict[InsnType, float]] = {
    "x86": {
        InsnType.VEC_SP:   16.0,  # YMM FP32 ×8 lanes; non-FMA → ×1, FMA caught by COMPUTE
        InsnType.VEC_DP:    8.0,  # YMM FP64 ×4 lanes
        InsnType.VEC_MEM:   0.0,  # pure load / store
        InsnType.VECTOR:   16.0,  # integer SIMD — same bucket as VEC_SP as fallback
        InsnType.COMPUTE:   2.0,  # scalar FMA / imul
        InsnType.SCALAR:    0.5,  # ~half of scalars are FP ops
        InsnType.MEMORY:    0.0,
        InsnType.CONTROL:   0.0,
        InsnType.SYNC:      0.0,
        InsnType.OTHER:     0.0,
    },
    "sass": {
        InsnType.COMPUTE:   2.0,  # FFMA (FP32 FMA) = mul + add
        InsnType.VECTOR:    2.0,  # HMMA / HFMA conservative
        InsnType.VEC_SP:    2.0,
        InsnType.VEC_DP:    2.0,
        InsnType.VEC_MEM:   0.0,
        InsnType.SCALAR:    1.0,  # FADD, FMUL, FCMP
        InsnType.MEMORY:    0.0,
        InsnType.CONTROL:   0.0,
        InsnType.SYNC:      0.0,
        InsnType.OTHER:     0.0,
    },
    "amdgcn": {
        InsnType.VEC_SP:    4.0,  # v_fma_f32 (SIMD4 lane × 2 ops)
        InsnType.VEC_DP:    2.0,  # v_fma_f64 (SIMD2 lane × 2 ops)
        InsnType.VEC_MEM:   0.0,
        InsnType.VECTOR:    4.0,  # integer lane ops — same bucket as VEC_SP
        InsnType.COMPUTE:   2.0,  # v_mac_f32 / v_mul_f32 pair
        InsnType.SCALAR:    1.0,  # s_mul_i32 etc.
        InsnType.MEMORY:    0.0,
        InsnType.CONTROL:   0.0,
        InsnType.SYNC:      0.0,
        InsnType.OTHER:     0.0,
    },
}
_FLOPS["ptx"]     = _FLOPS["sass"]
_FLOPS["aarch64"] = {                # NEON/SVE
    InsnType.VEC_SP:   8.0,   # 128-bit NEON FP32 ×4 lanes; FMA = ×2 ops
    InsnType.VEC_DP:   4.0,   # 128-bit NEON FP64 ×2 lanes
    InsnType.VEC_MEM:  0.0,
    InsnType.VECTOR:   8.0,   # integer SIMD — same bucket as VEC_SP
    InsnType.COMPUTE:  2.0,   # scalar FMA (fmadd)
    InsnType.SCALAR:   0.5,
    InsnType.MEMORY:   0.0,
    InsnType.CONTROL:  0.0,
    InsnType.SYNC:     0.0,
    InsnType.OTHER:    0.0,
}
_FLOPS["rv64"] = {               # RV64GV (RVV 1.0)
    InsnType.VEC_SP:   4.0,   # RVV FP32 — LMUL=1 baseline (4 FP32 per 128-bit)
    InsnType.VEC_DP:   2.0,   # RVV FP64 — LMUL=1 baseline
    InsnType.VEC_MEM:  0.0,
    InsnType.VECTOR:   4.0,   # integer RVV
    InsnType.COMPUTE:  2.0,   # FMA (fmadd.s, vfmadd.vv)
    InsnType.SCALAR:   0.5,
    InsnType.MEMORY:   0.0,
    InsnType.CONTROL:  0.0,
    InsnType.SYNC:     0.0,
    InsnType.OTHER:    0.0,
}

# Average bytes per memory instruction (varies by variant; this is a midpoint)
_MEM_BYTES: dict[str, float] = {
    "x86":    8.0,   # movss=4, movaps=16, vmovups ymm=32; midpoint
    "aarch64":8.0,
    "sass":   8.0,   # LDG.E=4, LDG.64=8, LDG.128=16; midpoint
    "ptx":    8.0,
    "amdgcn": 8.0,   # flat_load_dword=4, flat_load_dwordx4=16; midpoint
}


def _flops_table(arch: str) -> dict[InsnType, float]:
    a = arch.lower()
    if a in ("x86-64", "x86_64", "amd64", "x86", "cpu"):   return _FLOPS["x86"]
    if a in ("aarch64", "arm64", "armv8"):                  return _FLOPS["aarch64"]
    if a in ("sass", "cuda"):                                return _FLOPS["sass"]
    if a in ("ptx",):                                        return _FLOPS["ptx"]
    if a in ("amdgcn", "rocm", "hip", "gcn"):               return _FLOPS["amdgcn"]
    if a in ("rv64", "riscv64", "riscv", "rv32"):           return _FLOPS["rv64"]
    return _FLOPS["sass"]


def _mem_bytes(arch: str) -> float:
    a = arch.lower()
    if a in ("x86-64", "x86_64", "amd64", "x86", "cpu", "aarch64", "arm64"): return 8.0
    if a in ("sass", "cuda", "ptx"): return 8.0
    if a in ("amdgcn", "rocm", "hip", "gcn"): return 8.0
    if a in ("rv64", "riscv64", "riscv", "rv32"): return 4.0  # vle32 = 4 bytes/elem
    return 8.0


def _parse_dim3(tag: str) -> int:
    """Parse 'NxMxK' or 'N' into a volume integer."""
    parts = re.split(r"[x×]", tag.strip())
    v = 1
    for p in parts:
        try:
            v *= max(1, int(p))
        except ValueError:
            pass
    return max(v, 1)


def _thread_count(tags: dict) -> int:
    """
    Total parallel work items from span tags.

    GPU spans: grid=NxMxK × block=NxMxK
    CPU/OpenMP spans: count=N  (work-sharing loop iteration count)
    Falls back to 1 if neither is present.
    """
    grid_s  = tags.get("grid",  "")
    block_s = tags.get("block", "")
    if grid_s or block_s:
        return _parse_dim3(grid_s or "1") * _parse_dim3(block_s or "1")

    # OpenMP work loop: count=N gives the loop iteration count.
    # ACPP's OMP backend emits count=0 (OMPT hook doesn't receive the real
    # iteration count for ACPP-generated loops), so we only use it when > 0.
    count_s = tags.get("count", "")
    if count_s:
        try:
            c = int(count_s)
            if c > 0:
                return c
        except ValueError:
            pass
    return 1


# ── Per-kernel result ─────────────────────────────────────────────────────────

@dataclass
class KernelMetrics:
    kernel_name:   str
    arch:          str
    duration_ns:   int      # GPU-accurate if cudaEvent/hipEvent timing used
    threads:       int      # grid × block volume
    est_flops:     float    # FLOPs  (from hw counters when available, else disasm estimate)
    est_bytes:     float    # memory bytes (from hw counters when available, else estimate)
    arith_intensity: float  # FLOPs / byte  (∞ if no memory ops)
    achieved_tflops: float  # est_flops / duration in TFLOPs/s
    achieved_gbs:    float  # est_bytes / duration in GB/s
    flops_pct:     float    # achieved_tflops / device.fp32_tflops × 100
    bw_pct:        float    # achieved_gbs / device.bandwidth_gbs × 100
    bound:         str      # "compute", "memory", or "unknown"
    ridge:         float    # device ridge point (FLOPs/byte)
    data_source:   str = "disasm"   # "hardware_counters" | "disasm"
    fp64_fraction: float = 0.0   # fraction of FLOPs that are FP64 (0–1)
    fp16_fraction: float = 0.0   # fraction of FLOPs that are FP16 (0–1)
    fp64_tflops:   float = 0.0   # device FP64 peak (TFLOPs/s)
    fp16_tflops:   float = 0.0   # device FP16 peak (TFLOPs/s)
    # Multi-level roofline
    l2_ai:         float = 0.0   # FLOPs / L2_bytes  (GPU)
    l1_ai:         float = 0.0   # FLOPs / L1_bytes  (GPU)
    l3_ai:         float = 0.0   # FLOPs / L3_bytes  (CPU LLC)
    l2_bytes:      float = 0.0   # L2 traffic bytes (GPU)
    l1_bytes:      float = 0.0   # L1 traffic bytes (GPU)
    l3_bytes:      float = 0.0   # L3 traffic bytes (CPU)
    # SM occupancy and ILP
    occupancy_pct: float = 0.0   # SM occupancy % 0–100 (GPU, from ncu)
    ipc:           float = 0.0   # instructions per cycle (GPU, from ncu)


def metrics_from_counters(
    counters: "KernelCounters",   # type: ignore[name-defined]
    device:   DevicePeak,
    span:     Optional["SpanEvent"] = None,   # type: ignore[name-defined]
    arch:     str = "",
) -> Optional[KernelMetrics]:
    """
    Build KernelMetrics from hardware counter measurements.

    FP ops: fp32 + fp64 × 2 (FP64 has twice the register pressure / bandwidth
    cost, so weight it 2× for the compute-bound comparison against fp32_tflops).
    DRAM bytes: exact from IMC/arbiter when source is ncu/rocprof/likwid/perf_uncore;
                LLC-miss proxy otherwise (read-only, write traffic missing).
    """
    from .hwcounters import KernelCounters

    dur_ns = counters.duration_ns
    if dur_ns <= 0 and span is not None:
        dur_ns = span.duration_ns

    # ncu doesn't report per-kernel wall time in its CSV metrics mode.
    # Derive from sm__cycles_elapsed.max (preferred): the .max aggregation gives
    # the elapsed cycles of the busiest SM, which equals kernel wall time in
    # clock cycles regardless of how many SMs participated.
    # Fallback to .sum / sm_count only when .max wasn't collected (older ncu).
    if dur_ns <= 0 and counters.sm_cycles_max > 0:
        clock_hz = device.core_clock_ghz * 1e9 if device.core_clock_ghz > 0 else 1e9
        dur_ns = int(counters.sm_cycles_max / clock_hz * 1e9)
    elif dur_ns <= 0 and counters.sm_cycles_elapsed > 0:
        sm_count = device.sm_count or 1
        clock_hz = device.core_clock_ghz * 1e9 if device.core_clock_ghz > 0 else 1e9
        dur_ns = int(counters.sm_cycles_elapsed / (sm_count * clock_hz) * 1e9)

    if dur_ns <= 0:
        return None

    # Use all FP precisions with equal weight — standard "FP32-equivalent" view.
    # Do NOT double fp64_ops: that was non-standard and would misplace FP64
    # kernels against the FP32 ceiling. Instead, fp64_fraction is recorded so
    # the chart can annotate high-FP64 kernels and draw a separate FP64 ceiling.
    fp32 = counters.fp32_ops
    fp64 = counters.fp64_ops
    fp16 = counters.fp16_ops
    flops = fp32 + fp64 + fp16 * 0.5
    byt   = max(counters.dram_bytes, 0.0)   # guard against negative hw-counter glitches
    ai    = flops / byt if byt > 0.0 else float("inf")

    secs            = dur_ns / 1e9   # dur_ns > 0 guaranteed by early return above
    achieved_tflops = (flops / secs) / 1e12
    achieved_gbs    = (byt   / secs) / 1e9

    # Clamp percentages to [0, 999] — values >100% are valid (measurement error /
    # tensor-core overlap) but >1000% indicate a hw-counter miscalibration.
    fp32_peak = max(device.fp32_tflops,   1e-9)
    bw_peak   = max(device.bandwidth_gbs, 1e-9)
    flops_pct = min(100 * achieved_tflops / fp32_peak, 999.0)
    bw_pct    = min(100 * achieved_gbs    / bw_peak,   999.0)

    bound = "compute" if (byt == 0.0 or (ai != float("inf") and ai >= device.ridge_point)) else "memory"

    threads = _thread_count(span.tags) if span else 1

    # Encode counter source in data_source so the chart can annotate accuracy
    src = counters.source or "hardware_counters"
    if src == "perf_llcproxy":
        src = "hw_counters(LLC proxy — write traffic missing)"
    elif src in ("ncu", "rocprof", "likwid", "perf_uncore"):
        src = f"hw_counters({src})"

    # Precision fractions for chart annotation
    total_ops = max(fp32 + fp64 + fp16 * 0.5, 1e-30)   # never zero
    fp64_frac = fp64 / total_ops
    fp16_frac = (fp16 * 0.5) / total_ops

    # Multi-level arithmetic intensities
    l2_b = max(counters.l2_bytes, 0.0)
    l1_b = max(counters.l1_bytes, 0.0)
    l3_b = max(counters.l3_bytes, 0.0)
    l2_ai = flops / l2_b if l2_b > 0.0 else 0.0
    l1_ai = flops / l1_b if l1_b > 0.0 else 0.0
    l3_ai = flops / l3_b if l3_b > 0.0 else 0.0

    return KernelMetrics(
        kernel_name=counters.kernel_name,
        arch=arch,
        duration_ns=dur_ns,
        threads=threads,
        est_flops=flops,
        est_bytes=byt,
        arith_intensity=ai,
        achieved_tflops=achieved_tflops,
        achieved_gbs=achieved_gbs,
        flops_pct=flops_pct,
        bw_pct=bw_pct,
        bound=bound,
        ridge=device.ridge_point,
        data_source=src,
        fp64_fraction=fp64_frac,
        fp16_fraction=fp16_frac,
        fp64_tflops=device.fp64_tflops,
        fp16_tflops=device.fp16_tflops,
        l2_ai=l2_ai,
        l1_ai=l1_ai,
        l3_ai=l3_ai,
        l2_bytes=l2_b,
        l1_bytes=l1_b,
        l3_bytes=l3_b,
        occupancy_pct=counters.occupancy_pct,
        ipc=counters.ipc,
    )


def compute_kernel_metrics(
    span:   "SpanEvent",
    kd:     "KernelDisasm",
    device: DevicePeak,
) -> Optional[KernelMetrics]:
    """
    Estimate FLOPs, bandwidth and utilization for a single kernel span.
    Returns None when there is insufficient data (zero duration, no instructions).
    """
    if span.duration_ns <= 0 or not kd.lines:
        return None

    threads = _thread_count(span.tags)
    ftbl    = _flops_table(kd.arch)
    mbytes  = _mem_bytes(kd.arch)

    # Aggregate instruction counts by type
    counts: dict[InsnType, int] = {}
    for ln in kd.lines:
        counts[ln.itype] = counts.get(ln.itype, 0) + 1

    flops_per_thread  = sum(cnt * ftbl.get(itype, 0.0) for itype, cnt in counts.items())
    mem_insns_per_th  = counts.get(InsnType.MEMORY, 0)
    bytes_per_thread  = mem_insns_per_th * mbytes

    est_flops = flops_per_thread * threads
    est_bytes = bytes_per_thread * threads

    if est_bytes > 0:
        arith_intensity = est_flops / est_bytes
    else:
        arith_intensity = float("inf")

    secs = span.duration_ns / 1e9
    achieved_tflops = (est_flops / secs) / 1e12  if secs > 0 else 0.0
    achieved_gbs    = (est_bytes / secs) / 1e9   if secs > 0 else 0.0

    flops_pct = 100.0 * achieved_tflops / device.fp32_tflops  if device.fp32_tflops  > 0 else 0.0
    bw_pct    = 100.0 * achieved_gbs    / device.bandwidth_gbs if device.bandwidth_gbs > 0 else 0.0

    ridge = device.ridge_point
    if est_bytes == 0 or arith_intensity == float("inf"):
        bound = "compute"
    elif arith_intensity >= ridge:
        bound = "compute"
    else:
        bound = "memory"

    return KernelMetrics(
        kernel_name=span.name,
        arch=kd.arch,
        duration_ns=span.duration_ns,
        threads=threads,
        est_flops=est_flops,
        est_bytes=est_bytes,
        arith_intensity=arith_intensity,
        achieved_tflops=achieved_tflops,
        achieved_gbs=achieved_gbs,
        flops_pct=flops_pct,
        bw_pct=bw_pct,
        bound=bound,
        ridge=ridge,
    )


def analyze_trace(trace: "Trace") -> list[tuple[DevicePeak, KernelMetrics]]:
    """
    Return (device, metrics) pairs for every kernel that has both:
    - a profiled span with grid/block tags and non-zero duration
    - disassembly data

    One result per unique kernel name (uses the span with maximum duration as
    the most representative single invocation).
    """
    devices = getattr(trace, "devices", [])
    if not devices or not trace.disasm:
        return []

    disasm = trace.disasm

    # Build a secondary lookup: short/demangled name → disasm key.
    # ACPP stores disasm under _acpp_kernel_short() names (e.g. "test_Relax")
    # but span names are full mangled symbols (_Z18__acpp_sscp_kernel...).
    # Also index by any suffix after the last '::' for general C++ demangling.
    from ..disasm.extractor import _acpp_kernel_short
    _short_to_key: dict[str, str] = {}
    for key in disasm:
        _short_to_key[_acpp_kernel_short(key)] = key
        if "::" in key:
            _short_to_key[key.rsplit("::", 1)[-1]] = key

    def _disasm_key(span_name: str) -> str | None:
        if span_name in disasm:
            return span_name
        short = _acpp_kernel_short(span_name)
        if short in disasm:
            return short
        if short in _short_to_key:
            return _short_to_key[short]
        # fallback: check if any disasm key is a substring of the span name
        for key in disasm:
            if key in span_name or span_name in key:
                return key
        return None

    # Group spans by name, pick the one with max duration per name
    best_span: dict[str, "SpanEvent"] = {}
    for span in trace.spans:
        key = _disasm_key(span.name)
        if key is None:
            continue
        prev = best_span.get(key)
        if prev is None or span.duration_ns > prev.duration_ns:
            best_span[key] = span

    results: list[tuple[DevicePeak, KernelMetrics]] = []
    for name, span in best_span.items():
        kd  = disasm[name]
        cat = span.category.value

        # Match span category to device backend
        dev = next(
            (d for d in devices
             if d.backend == cat
             or (cat in ("openmp", "cpu", "opencl") and d.backend == "cpu")),
            devices[0],
        )

        m = compute_kernel_metrics(span, kd, dev)
        if m:
            results.append((dev, m))

    return sorted(results, key=lambda x: x[1].achieved_tflops, reverse=True)


# ── Roofline chart (returns Rich Text) ───────────────────────────────────────

def roofline_chart(
    device:  DevicePeak,
    metrics: list[KernelMetrics],
    width:   int = 62,
    height:  int = 10,
) -> "Text":  # type: ignore[name-defined]
    """
    Return a Rich Text roofline plot.

    X axis: arithmetic intensity (FLOPs/byte), log10 scale 0.01 → 10 000
    Y axis: TFLOPS/s, linear scale 0 → device peak FP32

    Color coding:
      cyan    memory-bandwidth ceiling (diagonal slope)
      yellow  compute ceiling (flat line) + ridge marker
      green   compute-bound kernels
      red     memory-bound kernels
    """
    from rich.text import Text

    peak_tflops = device.fp32_tflops
    peak_bw_gbs = device.bandwidth_gbs
    y_max       = peak_tflops if peak_tflops > 0 else 1.0

    X_LOG_MIN, X_LOG_MAX = -2.0, 4.0   # 0.01 … 10 000 FLOPs/byte
    YLABEL_W = 8   # chars reserved for y-axis label + "│"

    def to_col(intensity: float) -> int:
        if intensity <= 0:
            return 0
        lv = math.log10(max(intensity, 1e-9))
        return max(0, min(width - 1,
                          int((lv - X_LOG_MIN) / (X_LOG_MAX - X_LOG_MIN) * (width - 1))))

    def to_row(tflops: float) -> int:
        frac = min(tflops / y_max, 1.0) if y_max > 0 else 0.0
        return max(0, min(height - 1, int((1.0 - frac) * (height - 1))))

    # ── Build character + style grid ──────────────────────────────────────────
    # Each cell: (char, style_string)
    Cell = tuple[str, str]
    grid: list[list[Cell]] = [
        [(" ", "") for _ in range(width)] for _ in range(height)
    ]

    # Memory-bandwidth diagonal (cyan).
    # When the slope is steep (near the ridge) the diagonal can jump 2+ rows
    # between adjacent columns.  Fill vertical gaps so the line is continuous.
    prev_bw_row = height - 1
    for col in range(width):
        intensity  = 10 ** (X_LOG_MIN + col * (X_LOG_MAX - X_LOG_MIN) / (width - 1))
        mem_tflops = intensity * peak_bw_gbs / 1000
        if mem_tflops >= y_max:
            break
        row = to_row(mem_tflops)
        lo, hi = min(row, prev_bw_row), max(row, prev_bw_row)
        for r in range(lo, hi + 1):
            if 0 <= r < height:
                grid[r][col] = ("╱", "cyan")
        prev_bw_row = row

    # Compute ceiling (yellow horizontal)
    for col in range(width):
        intensity = 10 ** (X_LOG_MIN + col * (X_LOG_MAX - X_LOG_MIN) / (width - 1))
        if intensity * peak_bw_gbs / 1000 >= y_max:
            grid[0][col] = ("─", "yellow")

    # Ridge-point marker
    rc = to_col(device.ridge_point)
    if 0 <= rc < width:
        grid[0][rc] = ("▲", "bold yellow")

    # Kernel points — drawn last so they sit on top of the line
    LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    plotted: list[str] = []
    for idx, m in enumerate(metrics[:len(LETTERS)]):
        letter = LETTERS[idx]
        ai     = m.arith_intensity if m.arith_intensity < 1e9 else 9999.0
        col    = to_col(ai)
        row    = to_row(m.achieved_tflops)
        style  = "bold bright_green" if m.bound == "compute" else "bold bright_red"
        grid[row][col] = (letter, style)
        plotted.append(letter)

    # ── Render to Rich Text ───────────────────────────────────────────────────
    out = Text()

    for r in range(height):
        # Y-axis label
        frac  = 1.0 - r / max(height - 1, 1)
        y_val = frac * y_max
        lbl   = f"{y_val:6.2f} " if y_val < 10 else f"{y_val:6.1f} "
        out.append(lbl, style="dim")
        out.append("│", style="dim")

        # Row cells — merge runs of same style for compactness
        prev_style = None
        buf        = ""
        for char, style in grid[r]:
            if style != prev_style:
                if buf:
                    out.append(buf, style=prev_style or "")
                buf, prev_style = char, style
            else:
                buf += char
        if buf:
            out.append(buf, style=prev_style or "")
        out.append("\n")

    # X axis line
    out.append(" " * YLABEL_W + "└" + "─" * width + "\n", style="dim")

    # Tick labels
    ticks = [(-2, "0.01"), (-1, "0.1"), (0, "1"), (1, "10"),
             (2, "100"),   (3, "1K"),   (4, "10K")]
    ruler = [" "] * (YLABEL_W + width + 2)
    for exp, lbl in ticks:
        col = int((exp - X_LOG_MIN) / (X_LOG_MAX - X_LOG_MIN) * (width - 1))
        pos = YLABEL_W + col
        for i, ch in enumerate(lbl):
            if 0 <= pos + i < len(ruler):
                ruler[pos + i] = ch
    out.append("".join(ruler) + "\n", style="dim")
    out.append(
        " " * YLABEL_W
        + f"{'Arithmetic Intensity  (FLOPs / byte,  log scale)':^{width}}\n",
        style="dim italic",
    )

    # ── Legend ────────────────────────────────────────────────────────────────
    out.append("\n")
    for idx, m in enumerate(metrics[:len(LETTERS)]):
        letter = plotted[idx]
        ai_str = f"{m.arith_intensity:8.2f}" if m.arith_intensity < 1e9 else "       ∞"
        style  = "bright_green" if m.bound == "compute" else "bright_red"
        bound_icon = "⬆ compute" if m.bound == "compute" else "⬅ memory"
        out.append(f"  {letter} ", style=f"bold {style}")
        out.append(f"{m.kernel_name[:30]:<30}", style="bold white")
        out.append(f"  {ai_str} F/B", style="white")
        out.append(f"  FP32: {m.flops_pct:5.1f}%", style="yellow")
        out.append(f"  BW: {m.bw_pct:5.1f}%", style="cyan")
        out.append(f"  {bound_icon}\n", style=f"bold {style}")

    return out
