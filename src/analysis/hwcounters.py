"""
Hardware performance counter collection via platform CLI tools.

  CUDA:  ncu  (Nsight Compute ≥ 2021)      — exact per-kernel FLOPs + DRAM bytes
  ROCm:  rocprof                             — exact per-kernel FLOPs + DRAM bytes
  CPU:   LIKWID (preferred) → perf stat     — FLOPs + true DRAM bandwidth

Counter accuracy for roofline
──────────────────────────────
GPU:
  CUDA  — ncu Perfworks metrics count predicated-on SASS instructions, so FMA
          already counts as 2 FP ops.  DRAM bytes come from the DRAM arbiter
          (dram__bytes), not from LLC misses.

  ROCm  — rocprof SQ_INSTS_VALU_* count wave-level instruction issue.  DRAM
          traffic uses TCC_EA_{RD,WR}REQ_* with correct 32B/64B split.

CPU:
  LIKWID (first choice, if accessD daemon is running or msr-safe is loaded):
    • Programs Intel IMC / AMD UMC uncore PMU directly → exact DRAM bytes.
    • Uses FLOPS_DP / FLOPS_SP LIKWID groups → counts FP ops not instructions.
    • No runtime sudo; setup requires one-time sysadmin step.

  perf stat (fallback):
    • Tier-1: Intel FP events (fp_arith_inst_retired.*) + uncore_imc for DRAM.
              uncore_imc requires perf_event_paranoid ≤ 0.
    • Tier-2: Intel FP events + LLC miss proxy (DRAM write traffic missing).
    • Tier-3: AMD FP events (fp_ret_sse_avx_ops) + LLC miss proxy.
    • FMA already counts as 2 in FP_ARITH_INST_RETIRED on Intel.
"""

from __future__ import annotations
import csv
import io
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class KernelCounters:
    """Hardware counter measurements for a single kernel / whole program."""
    kernel_name: str
    fp32_ops:    float   # FP32 operations  (FMA = 2)
    fp64_ops:    float   # FP64 operations  (FMA = 2)
    fp16_ops:    float   # FP16/BF16 operations
    dram_bytes:  float   # DRAM read + write bytes (exact from IMC/arbiter)
    l2_bytes:    float   # L2 cache traffic bytes  (0 if unavailable)
    l1_bytes:    float   # L1 cache traffic bytes  (0 if unavailable)
    duration_ns: int     # kernel / program elapsed time
    sm_cycles_elapsed: float = 0.0  # sm__cycles_elapsed.sum (legacy, used only as last fallback)
    sm_cycles_max:     float = 0.0  # sm__cycles_elapsed.max = wall-clock cycles for duration
    source: str = ""     # "ncu" | "rocprof" | "likwid" | "perf_uncore" | "perf_llcproxy"
    l3_bytes:      float = 0.0   # LLC (L3) cache traffic bytes — CPU only (LLC-loads × 64)
    occupancy_pct: float = 0.0   # SM occupancy percentage 0–100 (GPU, from ncu)
    ipc:           float = 0.0   # instructions per clock cycle (GPU, from ncu)

    @property
    def arith_intensity(self) -> float:
        """Arithmetic intensity in FLOPs/byte against DRAM."""
        flops = self.fp32_ops + self.fp64_ops * 2   # weight FP64 heavier
        return flops / self.dram_bytes if self.dram_bytes > 0 else float("inf")

    @property
    def total_flops(self) -> float:
        return self.fp32_ops + self.fp64_ops + self.fp16_ops


class CounterPermissionError(Exception):
    """Raised when the profiling tool cannot access hardware counters."""


def _find_tool(*names: str) -> Optional[str]:
    # sudo strips PATH down to /usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin.
    # CUDA tools live under /usr/local/cuda/bin which is in the user's PATH but not sudo's.
    # Build a search path that includes common CUDA/ROCm install locations so the tool
    # is found regardless of whether we're running as root.
    extra_dirs = [
        "/usr/local/cuda/bin",
        "/usr/local/cuda-12/bin",
        "/usr/local/cuda-11/bin",
        "/opt/cuda/bin",
        "/opt/rocm/bin",
        "/opt/rocm/hip/bin",
    ]
    # Also honour the invoking user's PATH if sudo preserved SUDO_USER's environment
    import os
    user_path = os.environ.get("PATH", "")

    for name in names:
        # 1. Standard shutil.which (works when PATH is intact)
        path = shutil.which(name)
        if path:
            return path
        # 2. Absolute path given directly
        if Path(name).exists():
            return name
        # 3. Search extra CUDA/ROCm dirs
        for d in extra_dirs:
            candidate = Path(d) / name
            if candidate.exists():
                return str(candidate)
        # 4. Search every directory in the user's PATH string
        for d in user_path.split(os.pathsep):
            if d:
                candidate = Path(d) / name
                if candidate.exists():
                    return str(candidate)
    return None


# ── CUDA — ncu (Nsight Compute) ───────────────────────────────────────────────
#
# Metrics are split into two tiers:
#
#   CORE  — chip-independent Perfworks names present on ALL CUDA 11+ GPUs.
#           If even one metric in the list is unsupported, ncu produces an
#           empty CSV with no error message (silent failure).  Keep this set
#           minimal and verified.
#
#   EXT   — metrics available on Volta+ / newer ncu.  Tried first; if ncu
#           produces empty output the caller retries with CORE only.
#
# FMA counts as 2 in pred_on.sum metrics (per NVIDIA ISA reference).
# `sm__cycles_elapsed.sum` is used to derive duration when ncu does not
# provide explicit timing (see `_cycles_to_ns` in roofline.py).

_NCU_METRICS_CORE = [
    # FP32
    "sm__sass_thread_inst_executed_op_ffma_pred_on.sum",
    "sm__sass_thread_inst_executed_op_fadd_pred_on.sum",
    "sm__sass_thread_inst_executed_op_fmul_pred_on.sum",
    # FP64
    "sm__sass_thread_inst_executed_op_dfma_pred_on.sum",
    "sm__sass_thread_inst_executed_op_dadd_pred_on.sum",
    "sm__sass_thread_inst_executed_op_dmul_pred_on.sum",
    # FP16
    "sm__sass_thread_inst_executed_op_hfma_pred_on.sum",
    # DRAM — aggregate (always present)
    "dram__bytes.sum",
    # L2
    "l2tex__t_bytes.sum",
    # SM cycles for duration.
    # .max = wall-clock cycles (independent of how many SMs participated).
    # .sum is kept for legacy; equals max × participating_SM_count.
    "sm__cycles_elapsed.max",
    "sm__cycles_elapsed.sum",
]

_NCU_METRICS_EXT = _NCU_METRICS_CORE + [
    # Separate DRAM read / write (Volta+, newer ncu)
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    # L1/TEX (Volta+)
    "l1tex__t_bytes.sum",
    # SM active cycles for utilisation
    "sm__cycles_active.sum",
    # Tensor cores — Volta+ HMMA: each inst = 512 FP16 ops on Ampere 16×8×16
    "sm__inst_executed_pipe_tensor_op_hmma_pred_on.sum",
    # Occupancy — average active warps as % of peak (0–100)
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    # IPC — instructions executed per clock (divide by sm__cycles_elapsed.sum)
    "sm__inst_executed.sum",
]

_NCU_METRICS      = ",".join(_NCU_METRICS_EXT)
_NCU_METRICS_CORE_STR = ",".join(_NCU_METRICS_CORE)


def _ncu_run(ncu: str, metrics_str: str, command: list[str],
             log_path: str, env: dict | None) -> tuple[str, str]:
    """Run ncu once, return (csv_text, combined_output)."""
    result = subprocess.run(
        [ncu, "--csv", "--log-file", log_path,
         "--target-processes", "all",
         "--metrics", metrics_str,
         "--"] + command,
        capture_output=True, text=True,
        env=env, timeout=600,
    )
    combined = result.stdout + result.stderr
    csv_text  = Path(log_path).read_text(errors="replace")
    if not csv_text.strip():
        csv_text = result.stdout   # some ncu versions write CSV to stdout
    # ncu writes ==ERROR== ERR_NVGPUCTRPERM to the log file (not stderr).
    # Include it in combined so the permission check in collect_cuda catches it.
    combined = combined + csv_text
    return csv_text, combined


def collect_cuda(command: list[str],
                 env: dict | None = None) -> list[KernelCounters]:
    """
    Re-run *command* under ncu and return per-kernel hardware counters.

    Two-tier metric strategy:
      1. Extended set (Volta+ metrics, separate DRAM R/W, L1, tensor cores)
      2. Core set fallback (universally supported on all CUDA 11+ GPUs)

    ncu silently produces an empty CSV when any requested metric is
    unsupported — no error message, no partial output.  The retry with the
    core set handles GPUs where the extended metrics don't exist.

    Raises:
      FileNotFoundError       ncu not installed
      CounterPermissionError  GPU profiling access denied
      RuntimeError            ncu ran but produced no parseable output
    """
    ncu = _find_tool("ncu", "nv-nsight-cu-cli",
                     "/usr/local/cuda/bin/ncu",
                     "/opt/cuda/bin/ncu")
    if not ncu:
        raise FileNotFoundError(
            "ncu (Nsight Compute) not found.\n"
            "Install the CUDA toolkit:\n"
            "  sudo apt install nsight-compute\n"
            "or download from developer.nvidia.com/nsight-compute"
        )

    log_fd, log_path = tempfile.mkstemp(suffix=".ncu.csv")
    os.close(log_fd)
    try:
        # ── Pass 1: extended metrics ──────────────────────────────────────
        csv_text, combined = _ncu_run(ncu, _NCU_METRICS, command, log_path, env)

        # ── Permission check (before deciding whether to retry) ───────────
        _PERM_KW = ("ERR_NVGPUCTRPERM", "perf_event_paranoid",
                    "NVML_ERROR_INSUFFICIENT_PERMISSIONS")
        if any(kw in combined for kw in _PERM_KW):
            raise CounterPermissionError(
                "ncu cannot access GPU hardware counters.\n\n"
                "Fix (choose one):\n"
                "  1. Already running as root?  Try option 2.\n"
                "  2. Disable driver restriction (permanent, needs reboot):\n"
                "       sudo sh -c 'echo \"options nvidia "
                "NVreg_RestrictProfilingToAdminUsers=0\" "
                "> /etc/modprobe.d/nvprofiling.conf'\n"
                "       sudo update-initramfs -u && sudo reboot\n"
                "  3. Temporarily (lost on reboot):\n"
                "       sudo sh -c "
                "'echo 0 > /proc/sys/kernel/perf_event_paranoid'"
            )

        counters = _parse_ncu_csv(csv_text)

        # ── Pass 2: core metrics only (if extended set was rejected) ──────
        # ncu silently produces an empty CSV when a metric is unsupported.
        # Retry with the universally-supported core set.
        if not counters:
            # Truncate log file before reuse
            open(log_path, "w").close()
            csv_text2, combined2 = _ncu_run(
                ncu, _NCU_METRICS_CORE_STR, command, log_path, env
            )
            if any(kw in combined2 for kw in _PERM_KW):
                raise CounterPermissionError(
                    "ncu cannot access GPU hardware counters (permission "
                    "denied on core metric set).\n"
                    "Run as root or disable NVreg_RestrictProfilingToAdminUsers."
                )
            counters = _parse_ncu_csv(csv_text2)
            combined  = combined2   # use second-pass output for error message

        if not counters:
            # Show full ncu output so the user can see unsupported-metric
            # warnings or other diagnostics — not just the first 600 chars.
            ncu_diag = combined.strip()[-2000:]
            raise RuntimeError(
                "ncu ran the program but produced no counter data "
                "(both extended and core metric sets returned empty CSV).\n\n"
                "Possible causes:\n"
                "  • No CUDA kernels executed (check the program ran correctly)\n"
                "  • ncu version mismatch — try upgrading Nsight Compute\n"
                "  • Running inside a container without GPU counter access\n\n"
                f"ncu diagnostic output (last 2000 chars):\n{ncu_diag}"
            )

        return counters
    finally:
        try:
            os.unlink(log_path)
        except OSError:
            pass


_NCU_AVG_METRICS = {
    "sm__warps_active.avg.pct_of_peak_sustained_active",
}


def _parse_ncu_csv(csv_text: str) -> list[KernelCounters]:
    """Accumulate ncu --csv rows into per-kernel KernelCounters."""
    # ncu writes ==PROF== / ==ERROR== / ==WARNING== status lines to the log
    # file before the CSV header.  DictReader uses the first line as column
    # names, so these lines corrupt the header and every row is silently
    # skipped.  Strip them before parsing.
    clean_lines = [ln for ln in csv_text.splitlines() if not ln.startswith("==")]
    csv_text = "\n".join(clean_lines)

    accum: dict[str, dict] = {}
    # Track per-invocation count to correctly average metrics like occupancy.
    # We count invocations by the number of rows seen for sm__cycles_elapsed.max
    # (one row per kernel invocation).
    inv_count: dict[str, int] = {}
    try:
        reader = csv.DictReader(io.StringIO(csv_text))
        for row in reader:
            kname  = row.get("Kernel Name", "").strip()
            metric = row.get("Metric Name", "").strip()
            # ncu uses locale-aware formatting: "539 017" not "539017".
            # Strip both commas (en_US) and spaces (fr_FR / ncu default).
            val_s  = row.get("Metric Value", "0").replace(",", "").replace(" ", "").strip()
            if not kname or not metric:
                continue
            try:
                val = float(val_s)
            except ValueError:
                continue
            accum.setdefault(kname, {})[metric] = \
                accum.get(kname, {}).get(metric, 0.0) + val
            if metric == "sm__cycles_elapsed.max":
                inv_count[kname] = inv_count.get(kname, 0) + 1
    except Exception:
        return []

    results = []
    for kname, m in accum.items():
        n_inv = max(inv_count.get(kname, 1), 1)

        def g(k: str) -> float: return m.get(k, 0.0)

        fp32 = (g("sm__sass_thread_inst_executed_op_ffma_pred_on.sum") * 2
                + g("sm__sass_thread_inst_executed_op_fadd_pred_on.sum")
                + g("sm__sass_thread_inst_executed_op_fmul_pred_on.sum"))
        fp64 = (g("sm__sass_thread_inst_executed_op_dfma_pred_on.sum") * 2
                + g("sm__sass_thread_inst_executed_op_dadd_pred_on.sum")
                + g("sm__sass_thread_inst_executed_op_dmul_pred_on.sum"))
        fp16 =  g("sm__sass_thread_inst_executed_op_hfma_pred_on.sum") * 2

        # Tensor: each HMMA inst on Ampere = 256 FP16 FMAs = 512 FP16 ops
        tc_hmma = g("sm__inst_executed_pipe_tensor_op_hmma_pred_on.sum")
        fp16 += tc_hmma * 512

        dram_rd = g("dram__bytes_read.sum")
        dram_wr = g("dram__bytes_write.sum")
        # Fallback: older ncu builds expose only the aggregate
        if dram_rd == 0 and dram_wr == 0:
            dram_rd = g("dram__bytes.sum")

        # SM utilisation fraction (per-SM sum → normalise by elapsed)
        elapsed = g("sm__cycles_elapsed.sum")
        active  = g("sm__cycles_active.sum")
        util = active / elapsed if elapsed > 0 else 0.0

        # Occupancy is reported as percentage per invocation; average across invocations.
        occupancy = g("sm__warps_active.avg.pct_of_peak_sustained_active") / n_inv

        # IPC: instructions executed / elapsed cycles (summed; ratio is preserved).
        inst_exec = g("sm__inst_executed.sum")
        cycles_el = g("sm__cycles_elapsed.sum")
        ipc = inst_exec / cycles_el if cycles_el > 0 else 0.0

        results.append(KernelCounters(
            kernel_name=kname,
            fp32_ops=fp32, fp64_ops=fp64, fp16_ops=fp16,
            dram_bytes=dram_rd + dram_wr,
            l2_bytes=g("l2tex__t_bytes.sum"),
            l1_bytes=g("l1tex__t_bytes.sum"),
            duration_ns=0,           # filled by metrics_from_counters via sm_cycles
            sm_cycles_max=g("sm__cycles_elapsed.max"),
            sm_cycles_elapsed=g("sm__cycles_elapsed.sum"),
            source="ncu",
            occupancy_pct=occupancy,
            ipc=ipc,
        ))
    return results


# ── ROCm — rocprof ────────────────────────────────────────────────────────────
#
# TCC = Texture Cache Controller (L2 on AMD).
# TCC_EA_RDREQ_sum   = total read requests to the memory fabric from L2
# TCC_EA_RDREQ_32B_sum = those requests that carried only 32 B (vs 64 B default)
# → exact bytes = (TCC_EA_RDREQ_sum - TCC_EA_RDREQ_32B_sum) × 64
#               + TCC_EA_RDREQ_32B_sum × 32
# Similarly for writes.
#
# GFX10+ (RDNA/CDNA2+) renamed counters to GL2C_* for the L2 cache.

_ROCPROF_COUNTERS_GFX9 = [
    # FP operations — counted at wave issue (not FMA-folded)
    "SQ_INSTS_VALU_ADD_F32",
    "SQ_INSTS_VALU_MUL_F32",
    "SQ_INSTS_VALU_FMA_F32",       # each = 2 FP32 ops
    "SQ_INSTS_VALU_ADD_F64",
    "SQ_INSTS_VALU_FMA_F64",       # each = 2 FP64 ops
    "SQ_INSTS_VALU_ADD_F16",
    "SQ_INSTS_VALU_FMA_F16",       # each = 2 FP16 ops
    # DRAM traffic via L2→fabric (correct 32B/64B split)
    "TCC_EA_RDREQ_sum",            # total read requests
    "TCC_EA_RDREQ_32B_sum",        # of which are 32-byte
    "TCC_EA_WRREQ_sum",            # total write requests
    "TCC_EA_WRREQ_32B_sum",        # of which are 32-byte
    # L2 hits/misses (for L2 bandwidth in the hierarchy model)
    "TCC_HIT_sum",
    "TCC_MISS_sum",
]

# CDNA2 (MI200 series) / GFX9x — same names, plus matrix engine
_ROCPROF_COUNTERS_CDNA2 = _ROCPROF_COUNTERS_GFX9 + [
    "SQ_INSTS_VALU_MFMA_F32_16x16x4",   # Matrix FMA 16×16×4  FP32
    "SQ_INSTS_VALU_MFMA_F32_32x32x2",   # Matrix FMA 32×32×2  FP32
    "SQ_INSTS_VALU_MFMA_F64_16x16x4",   # Matrix FMA 16×16×4  FP64
]

# GFX10 / RDNA2+ rename the L2 → GL2C
_ROCPROF_COUNTERS_GFX10 = [
    "SQ_INSTS_VALU_ADD_F32",
    "SQ_INSTS_VALU_MUL_F32",
    "SQ_INSTS_VALU_FMA_F32",
    "SQ_INSTS_VALU_ADD_F64",
    "SQ_INSTS_VALU_FMA_F64",
    "GL2C_MC_RDREQ_sum",               # GFX10: L2→MC reads
    "GL2C_MC_WRREQ_sum",               # GFX10: L2→MC writes
    "GL2C_HIT_sum",
    "GL2C_MISS_sum",
]


def _rocprof_counter_set() -> list[str]:
    """Detect AMD GPU generation and return the right counter set."""
    try:
        r = subprocess.run(["rocm-smi", "--showproductname"],
                           capture_output=True, text=True, timeout=5)
        txt = r.stdout.lower()
        if any(x in txt for x in ("mi200", "mi250", "mi210", "cdna2", "gfx90a")):
            return _ROCPROF_COUNTERS_CDNA2
        if any(x in txt for x in ("rdna2", "rdna3", "gfx10", "gfx11", "rx 6", "rx 7")):
            return _ROCPROF_COUNTERS_GFX10
    except Exception:
        pass
    return _ROCPROF_COUNTERS_GFX9   # safe default for GFX8/GFX9/MI100


def collect_rocm(command: list[str],
                 env: dict | None = None) -> list[KernelCounters]:
    """Re-run *command* under rocprof and return per-kernel hardware counters."""
    rocprof = _find_tool("rocprof",
                         "/opt/rocm/bin/rocprof",
                         "/usr/bin/rocprof")
    if not rocprof:
        raise FileNotFoundError(
            "rocprof not found.\n"
            "Install ROCm: https://docs.amd.com\n"
            "Or: sudo apt install rocm-dev"
        )

    counter_set = _rocprof_counter_set()
    counter_fd, counter_path = tempfile.mkstemp(suffix=".txt")
    out_dir = tempfile.mkdtemp()
    try:
        with os.fdopen(counter_fd, "w") as f:
            f.write("pmc: " + " ".join(counter_set) + "\n")

        out_csv = os.path.join(out_dir, "results.csv")
        result = subprocess.run(
            [rocprof, "--stats", "-i", counter_path, "-o", out_csv, "--"] + command,
            capture_output=True, text=True,
            env=env, timeout=600,
        )
        combined = result.stdout + result.stderr

        if any(kw in combined for kw in ("Permission", "permission", "EPERM")):
            raise CounterPermissionError(
                "rocprof cannot access GPU performance counters.\n"
                "Fix: sudo usermod -a -G video $USER  (then re-login)\n"
                "Or run as root."
            )

        if not Path(out_csv).exists():
            raise RuntimeError(
                f"rocprof produced no output.\nrocprof stderr:\n{combined[:800]}"
            )

        counters = _parse_rocprof_csv(
            Path(out_csv).read_text(errors="replace"), counter_set
        )
        if not counters:
            raise RuntimeError("rocprof ran but counter CSV is empty.")
        return counters
    finally:
        try:
            os.unlink(counter_path)
            shutil.rmtree(out_dir, ignore_errors=True)
        except OSError:
            pass


def _rocprof_dram_bytes(row_g: "Callable[[str], float]",
                        counter_set: list[str]) -> float:
    """
    Compute exact DRAM bytes from TCC or GL2C request counters.

    TCC_EA_RDREQ_sum counts all read requests regardless of size.
    TCC_EA_RDREQ_32B_sum counts the subset that were 32-byte requests.
    All others are 64-byte.  Same logic applies to writes.
    """
    if "GL2C_MC_RDREQ_sum" in counter_set:
        # GFX10: each request = 64 B
        return (row_g("GL2C_MC_RDREQ_sum") + row_g("GL2C_MC_WRREQ_sum")) * 64

    rd_total = row_g("TCC_EA_RDREQ_sum")
    rd_32b   = row_g("TCC_EA_RDREQ_32B_sum")
    wr_total = row_g("TCC_EA_WRREQ_sum")
    wr_32b   = row_g("TCC_EA_WRREQ_32B_sum")

    rd_bytes = rd_32b * 32 + (rd_total - rd_32b) * 64
    wr_bytes = wr_32b * 32 + (wr_total - wr_32b) * 64
    return rd_bytes + wr_bytes


def _parse_rocprof_csv(csv_text: str,
                       counter_set: list[str]) -> list[KernelCounters]:
    results = []
    try:
        reader = csv.DictReader(io.StringIO(csv_text))
        for row in reader:
            kname = (row.get("Kernel-Name") or row.get("KernelName") or "").strip()
            if not kname:
                continue

            def g(k: str) -> float:
                try:
                    return float((row.get(k) or "0").replace(",", ""))
                except ValueError:
                    return 0.0

            fp32 = (g("SQ_INSTS_VALU_ADD_F32") + g("SQ_INSTS_VALU_MUL_F32")
                    + g("SQ_INSTS_VALU_FMA_F32") * 2)
            fp64 = g("SQ_INSTS_VALU_ADD_F64") + g("SQ_INSTS_VALU_FMA_F64") * 2
            fp16 = g("SQ_INSTS_VALU_ADD_F16") + g("SQ_INSTS_VALU_FMA_F16") * 2

            # MFMA (matrix engine on CDNA2): each instruction operates on a
            # 16×16×4 or 32×32×2 tile; count total element-level FP ops.
            fp32 += g("SQ_INSTS_VALU_MFMA_F32_16x16x4") * (16 * 16 * 4 * 2)
            fp32 += g("SQ_INSTS_VALU_MFMA_F32_32x32x2") * (32 * 32 * 2 * 2)
            fp64 += g("SQ_INSTS_VALU_MFMA_F64_16x16x4") * (16 * 16 * 4 * 2)

            dram = _rocprof_dram_bytes(g, counter_set)

            # L2 traffic from HIT+MISS × line size
            l2_key = "GL2C" if "GL2C_HIT_sum" in counter_set else "TCC"
            l2 = (g(f"{l2_key}_HIT_sum") + g(f"{l2_key}_MISS_sum")) * 64

            begin = g("BeginNs")
            end   = g("EndNs")

            results.append(KernelCounters(
                kernel_name=kname,
                fp32_ops=fp32, fp64_ops=fp64, fp16_ops=fp16,
                dram_bytes=dram, l2_bytes=l2, l1_bytes=0.0,
                duration_ns=int(end - begin) if end > begin else 0,
                source="rocprof",
            ))
    except Exception:
        pass
    return results


# ── CPU — LIKWID (preferred) ──────────────────────────────────────────────────
#
# LIKWID programs the Intel IMC / AMD UMC uncore PMUs via its access daemon
# (likwid-accessD, runs setuid root or as a systemd service).  Users need to
# be in the `likwid` group — no per-invocation sudo required.
#
# Groups used:
#   FLOPS_DP  — FP64 + FP32 SIMD instruction counts, derives Mflops/s
#   MEM_DP    — Intel IMC CAS counts → exact DRAM read + write bytes/s
#
# LIKWID -O outputs machine-readable tables:
#   TABLE,Group N Computation,<GROUP>
#   Metric,core 0,core 1,...,Sum
#   DP [MFLOPS/s],123.4,...,987.6
#   Memory BW [MBytes/s],...,12345.6
#   Runtime (RDTSC) [s],...,1.234

def _likwid_available() -> bool:
    if not shutil.which("likwid-perfctr"):
        return False
    # Quick probe: try to initialise with a no-op marker to check daemon access.
    try:
        r = subprocess.run(
            ["likwid-perfctr", "-g", "INSTR_RETIRED_ANY", "-C", "0", "--", "true"],
            capture_output=True, text=True, timeout=10,
        )
        # If the daemon is not running or the user is not in the likwid group,
        # likwid-perfctr exits non-zero with a "permission" or "accessD" message.
        if "permission" in r.stderr.lower() or "accessd" in r.stderr.lower():
            return False
        return True
    except Exception:
        return False


def _cpu_count() -> int:
    try:
        return os.cpu_count() or 1
    except Exception:
        return 1


def _p_core_cpus() -> list[int]:
    """
    Return logical CPU IDs that are P-cores (Performance cores) on hybrid Intel CPUs.

    Detects by comparing max frequency: P-cores run at a higher max frequency than
    E-cores.  Returns all CPUs if the topology cannot be determined.
    """
    try:
        freqs: dict[int, int] = {}
        import glob as _glob
        for path in _glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cpufreq/cpuinfo_max_freq"):
            cpu_id = int(path.split("/cpu")[2].split("/")[0])
            with open(path) as f:
                freqs[cpu_id] = int(f.read().strip())
        if not freqs:
            return list(range(_cpu_count()))
        max_freq = max(freqs.values())
        # P-cores run at max_freq; E-cores are typically 20–40% slower.
        # Use 95% of max_freq as threshold to be robust against turbo variance.
        threshold = max_freq * 0.95
        p_cores = sorted(cpu for cpu, freq in freqs.items() if freq >= threshold)
        return p_cores if p_cores else list(range(_cpu_count()))
    except Exception:
        return list(range(_cpu_count()))


def collect_cpu_likwid(command: list[str],
                       env: dict | None = None) -> Optional[KernelCounters]:
    """
    Use LIKWID to collect accurate FLOPs + DRAM bandwidth for the whole program.

    Runs two passes (FLOPS_DP group, then MEM_DP group) because most PMUs
    cannot multiplex FP retire events and uncore IMC events simultaneously.
    Elapsed time is taken from the FLOPS_DP pass.
    """
    n_cores = _cpu_count()
    cpu_mask = f"S0:0-{n_cores - 1}"

    results: dict[str, dict] = {}  # group_name → {metric: value}

    for group in ("FLOPS_DP", "MEM_DP"):
        stat_fd, stat_path = tempfile.mkstemp(suffix=f".likwid_{group}.txt")
        os.close(stat_fd)
        try:
            r = subprocess.run(
                ["likwid-perfctr", "-O",        # machine-readable tables
                 "-g", group,
                 "-C", cpu_mask,
                 "--"] + command,
                capture_output=True, text=True,
                env=env, timeout=600,
            )
            results[group] = _parse_likwid_table(r.stdout + r.stderr, group)
        finally:
            try:
                os.unlink(stat_path)
            except OSError:
                pass

    def g(grp: str, metric: str) -> float:
        return results.get(grp, {}).get(metric, 0.0)

    # Runtime from FLOPS_DP pass (seconds)
    runtime_s = g("FLOPS_DP", "Runtime (RDTSC) [s]")
    if runtime_s <= 0:
        return None

    # FP operations: Mflops/s × runtime × 1e6
    # LIKWID FLOPS_DP group reports DP [MFLOPS/s] and SP [MFLOPS/s] summed across cores.
    dp_mf = g("FLOPS_DP", "DP [MFLOPS/s]")
    sp_mf = g("FLOPS_DP", "SP [MFLOPS/s]")
    avx512_mf = g("FLOPS_DP", "AVX512 [MFLOPS/s]")   # present when AVX-512 active

    fp64_ops = dp_mf   * runtime_s * 1e6
    fp32_ops = sp_mf   * runtime_s * 1e6
    fp32_ops += avx512_mf * runtime_s * 1e6   # AVX-512 counted separately in some versions

    # DRAM bandwidth: MEM_DP group → Memory BW [MBytes/s] (read + write via IMC CAS)
    mem_bw_mb = g("MEM_DP", "Memory BW [MBytes/s]")
    # Some LIKWID versions report "Memory bandwidth [MBytes/s]"
    if mem_bw_mb == 0.0:
        mem_bw_mb = g("MEM_DP", "Memory bandwidth [MBytes/s]")
    # Fallback: "Memory data volume [GBytes]" directly
    mem_vol_gb = g("MEM_DP", "Memory data volume [GBytes]")
    if mem_vol_gb > 0:
        dram_bytes = mem_vol_gb * 1e9
    else:
        dram_bytes = mem_bw_mb * runtime_s * 1e6

    if fp64_ops == 0 and fp32_ops == 0 and dram_bytes == 0:
        return None

    return KernelCounters(
        kernel_name="(whole program)",
        fp32_ops=fp32_ops, fp64_ops=fp64_ops, fp16_ops=0.0,
        dram_bytes=dram_bytes, l2_bytes=0.0, l1_bytes=0.0,
        duration_ns=int(runtime_s * 1e9),
        source="likwid",
    )


def _parse_likwid_table(text: str, group: str) -> dict[str, float]:
    """
    Parse LIKWID -O output tables.

    -O format (LIKWID ≥ 5):
        TABLE,Group N Computation,<GROUP>
        Metric,core 0,...,Sum
        DP [MFLOPS/s],100,...,800.0
        ...

    We find the table for `group` and extract the Sum column for each metric.
    For the runtime we look at a single-value row (no Sum column).
    """
    result: dict[str, float] = {}
    in_table  = False
    sum_col   = -1

    for line in text.splitlines():
        line = line.rstrip()
        # Detect table header for the target group
        if line.startswith("TABLE,") and group in line:
            in_table  = True
            sum_col   = -1
            continue
        # Any other TABLE line ends this table
        if line.startswith("TABLE,") and in_table:
            in_table = False
            continue
        if not in_table:
            continue

        parts = [p.strip() for p in line.split(",")]
        if not parts:
            continue

        # "Metric,core 0,...,Sum" header row
        if parts[0] == "Metric":
            try:
                sum_col = parts.index("Sum")
            except ValueError:
                sum_col = len(parts) - 1   # last column
            continue

        # Data row: first column is metric name, sum_col is the value we want
        metric = parts[0]
        if not metric:
            continue

        # For single-value rows (e.g. "Runtime (RDTSC) [s],1.234")
        if sum_col < 0 or sum_col >= len(parts):
            if len(parts) >= 2:
                try:
                    result[metric] = float(parts[-1].replace(",", ""))
                except ValueError:
                    pass
            continue

        try:
            result[metric] = float(parts[sum_col].replace(",", ""))
        except ValueError:
            pass

    return result


# ── CPU — perf stat (fallback) ────────────────────────────────────────────────
#
# Three tiers:
#   1. Intel FP events + uncore_imc (exact DRAM) — needs paranoid ≤ 0
#   2. Intel FP events + LLC miss proxy (writes not counted)
#   3. AMD FP events  + LLC miss proxy

_PERF_EVENTS_INTEL_FP = [
    "fp_arith_inst_retired.scalar_single",       # FP32 scalar   (FMA = 2)
    "fp_arith_inst_retired.scalar_double",       # FP64 scalar   (FMA = 2)
    "fp_arith_inst_retired.128b_packed_single",  # SSE   FP32 ×4
    "fp_arith_inst_retired.128b_packed_double",  # SSE   FP64 ×2
    "fp_arith_inst_retired.256b_packed_single",  # AVX2  FP32 ×8  (FMA = ×16)
    "fp_arith_inst_retired.256b_packed_double",  # AVX2  FP64 ×4  (FMA = ×8)
    "fp_arith_inst_retired.512b_packed_single",  # AVX-512 FP32 ×16
    "fp_arith_inst_retired.512b_packed_double",  # AVX-512 FP64 ×8
]

# Intel IMC uncore — actual DRAM CAS commands.  Each = 1 cache line = 64 bytes.
# perf_event_paranoid ≤ 0 required.  Multiple IMC channels are summed automatically.
_PERF_EVENTS_INTEL_UNCORE = [
    "uncore_imc/cas_count_read/",    # DRAM read transactions
    "uncore_imc/cas_count_write/",   # DRAM write transactions
]

# AMD Zen 2/3/4 — fp_ret_sse_avx_ops already counts element-level ops (not instructions)
# so a vfmadd with 8 FP32 elements = 16 operations (8 mul + 8 add).
_PERF_EVENTS_AMD_FP = [
    "fp_ret_sse_avx_ops.all",              # total FP ops (add+mul+FMA×2, all widths)
    "fp_ret_sse_avx_ops.fp64_add_sub_ops", # FP64 add/sub only (to split from FP32)
    "fp_ret_sse_avx_ops.fp64_mult_ops",
    "fp_ret_sse_avx_ops.fp64_mult_add_ops",
]

# Generic fallback LLC events (available at paranoid ≤ 1)
_PERF_EVENTS_LLC = [
    "LLC-load-misses",
    "LLC-loads",
]


def _cpu_vendor() -> str:
    """Return 'intel', 'amd', or 'unknown'."""
    try:
        info = Path("/proc/cpuinfo").read_text()
        m = re.search(r"vendor_id\s*:\s*(\S+)", info)
        if m:
            v = m.group(1).lower()
            if "genuineintel" in v:
                return "intel"
            if "authenticamd" in v:
                return "amd"
    except Exception:
        pass
    return "unknown"


def _perf_run(events: list[str], command: list[str],
              env: dict | None,
              taskset_cpus: list[int] | None = None) -> tuple[str, int]:
    """Run perf stat with events, return (combined_output, returncode)."""
    prefix: list[str] = []
    if taskset_cpus and shutil.which("taskset"):
        prefix = ["taskset", "-c", ",".join(str(c) for c in taskset_cpus)]
    r = subprocess.run(
        prefix + ["perf", "stat", "-e", ",".join(events), "--"] + command,
        capture_output=True, text=True,
        env=env, timeout=600,
    )
    return r.stderr + r.stdout, r.returncode


def _drop_bad_event(events: list[str], text: str) -> list[str]:
    """Remove the first unrecognised event name from the list, if any."""
    m = (re.search(r"Unable to find event on a PMU of '([^']+)'", text)
         or re.search(r"event '([^']+)' is not supported", text))
    if m:
        bad = m.group(1).lower()
        return [e for e in events if e.lower().rstrip("/") != bad.rstrip("/")]
    return []   # could not identify → give up


def collect_cpu(command: list[str],
                env: dict | None = None) -> Optional[KernelCounters]:
    """
    Collect CPU FLOPs + DRAM bandwidth for the whole program.

    Strategy:
      1. LIKWID (most accurate — IMC uncore + PMU FP counts)
      2. Intel fp_arith_inst_retired + uncore_imc (exact DRAM)
      3. Intel fp_arith_inst_retired + LLC miss proxy
      4. AMD fp_ret_sse_avx_ops + LLC miss proxy
      5. LLC miss proxy only (partial — no FP count)
    """
    if not shutil.which("perf"):
        raise FileNotFoundError(
            "perf not found.\n"
            "Install: sudo apt install linux-tools-$(uname -r) linux-tools-common"
        )

    vendor = _cpu_vendor()

    # ── Tier 0: LIKWID ────────────────────────────────────────────────────
    if _likwid_available():
        result = collect_cpu_likwid(command, env)
        if result is not None:
            return result

    # ── Tier 1: Intel FP + uncore IMC ────────────────────────────────────
    if vendor in ("intel", "unknown"):
        fp_events = list(_PERF_EVENTS_INTEL_FP)
        # Try to add uncore IMC; some kernels expose it even at paranoid=1
        uncore_ok = _probe_uncore_events()
        events = fp_events + (_PERF_EVENTS_INTEL_UNCORE if uncore_ok else _PERF_EVENTS_LLC)

        combined, _ = _perf_run_with_retry(events, command, env)
        if combined:
            result = _parse_perf_stat(combined, uncore_ok)
            if result is not None:
                result.source = "perf_uncore" if uncore_ok else "perf_llcproxy"
                if result.fp32_ops == 0 and result.fp64_ops == 0:
                    # FP events returned 0 — likely hybrid CPU (E-cores don't have
                    # fp_arith_inst_retired).  Retry pinned to P-cores only.
                    p_cores = _p_core_cpus()
                    all_cores = list(range(_cpu_count()))
                    if p_cores != all_cores and shutil.which("taskset"):
                        combined2, _ = _perf_run_with_retry(events, command, env,
                                                            taskset_cpus=p_cores)
                        if combined2:
                            result2 = _parse_perf_stat(combined2, uncore_ok)
                            if result2 is not None and (result2.fp32_ops > 0 or result2.fp64_ops > 0):
                                result2.source = result.source
                                return result2
                    # All retries failed: FP truly not measurable
                    import sys
                    p_str = f"0-{p_cores[-1]}" if p_cores else "0"
                    print(
                        "[hprofiler] Warning: FP operation counters returned 0.\n"
                        "  Possible causes:\n"
                        f"    • Hybrid Intel CPU: program ran on E-cores (FP events require P-cores).\n"
                        f"      Fix: pin to P-cores manually:  taskset -c {p_str} ./your_program\n"
                        "           or install LIKWID for hybrid-aware FP counting.\n"
                        "    • Program runs too fast for perf multiplexing (< 200 ms).\n"
                        "    • Program uses no FP instructions.\n"
                        "  The roofline chart will show bandwidth measurement only.",
                        file=sys.stderr,
                    )
                return result

    # ── Tier 2: AMD FP + LLC ──────────────────────────────────────────────
    if vendor in ("amd", "unknown"):
        events = list(_PERF_EVENTS_AMD_FP) + list(_PERF_EVENTS_LLC)
        combined, _ = _perf_run_with_retry(events, command, env)
        if combined:
            result = _parse_perf_stat_amd(combined)
            if result is not None:
                result.source = "perf_llcproxy"
                return result

    # ── Tier 3: LLC proxy only ────────────────────────────────────────────
    combined, _ = _perf_run(_PERF_EVENTS_LLC, command, env)
    if "Permission denied" in combined or "perf_event_paranoid" in combined:
        raise CounterPermissionError(
            "perf cannot access hardware performance counters.\n\n"
            "Fix:\n"
            "  Temporarily:  "
            "sudo sh -c 'echo 0 > /proc/sys/kernel/perf_event_paranoid'\n"
            "  Permanently:  "
            "echo 'kernel.perf_event_paranoid=0' | "
            "sudo tee /etc/sysctl.d/99-perf.conf && sudo sysctl -p"
        )
    result = _parse_perf_stat(combined, uncore_events=False)
    if result is not None:
        result.source = "perf_llcproxy"
        return result

    raise RuntimeError(
        "perf stat produced no usable counter data.\n"
        f"Output:\n{combined[:600]}"
    )


def _probe_uncore_events() -> bool:
    """Return True if uncore_imc events are accessible (paranoid ≤ 0)."""
    try:
        r = subprocess.run(
            ["perf", "stat", "-e", "uncore_imc/cas_count_read/", "--", "true"],
            capture_output=True, text=True, timeout=10,
        )
        out = r.stderr + r.stdout
        return ("not supported" not in out.lower() and
                "permission" not in out.lower() and
                "Bad event" not in out)
    except Exception:
        return False


def _perf_run_with_retry(events: list[str], command: list[str],
                         env: dict | None,
                         taskset_cpus: list[int] | None = None) -> tuple[str, int]:
    """Run perf stat, dropping unrecognised events until it succeeds or list is empty."""
    while events:
        combined, rc = _perf_run(events, command, env, taskset_cpus)
        if "Permission denied" in combined or "perf_event_paranoid" in combined:
            raise CounterPermissionError(
                "perf cannot access hardware performance counters.\n"
                "  sudo sh -c 'echo 0 > /proc/sys/kernel/perf_event_paranoid'"
            )
        if "Bad event name" in combined or "event syntax error" in combined:
            new_events = _drop_bad_event(events, combined)
            if new_events and len(new_events) < len(events):
                events = new_events
                continue
            break
        if rc == 0 or "seconds time elapsed" in combined:
            return combined, rc
        break
    return "", 1


def _parse_perf_stat(text: str, uncore_events: bool = False) -> Optional[KernelCounters]:
    """
    Parse Intel perf stat output.

    FP_ARITH_INST_RETIRED already counts FLOPs (not instructions): on Intel,
    a 256-bit FMA adds 16 to the 256B_PACKED_SINGLE counter (8 SP elements × 2 FP ops).
    So the multipliers below are for SIMD lane width only, not ×2 for FMA.
    """
    ev: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or not line[0].isdigit():
            continue
        parts = re.split(r"\s{2,}", line, maxsplit=2)
        if len(parts) < 2:
            continue
        raw = re.sub(r"[^\d]", "", parts[0])
        try:
            val = float(raw)
        except ValueError:
            continue
        key = re.sub(r"^[\w-]+/", "", parts[1].rstrip("/")).lower().strip()
        key = re.sub(r":[ukpHG]+$", "", key)   # strip perf qualifiers (:u, :k, :p, etc.)
        if key:
            ev[key] = ev.get(key, 0.0) + val

    def g(k: str) -> float:
        return ev.get(k.lower(), 0.0)

    scalar_sp  = g("fp_arith_inst_retired.scalar_single")
    scalar_dp  = g("fp_arith_inst_retired.scalar_double")
    sse_sp     = g("fp_arith_inst_retired.128b_packed_single")   # ×4 elements
    sse_dp     = g("fp_arith_inst_retired.128b_packed_double")   # ×2 elements
    avx2_sp    = g("fp_arith_inst_retired.256b_packed_single")   # ×8
    avx2_dp    = g("fp_arith_inst_retired.256b_packed_double")   # ×4
    avx512_sp  = g("fp_arith_inst_retired.512b_packed_single")   # ×16
    avx512_dp  = g("fp_arith_inst_retired.512b_packed_double")   # ×8

    fp32_ops = scalar_sp + sse_sp * 4 + avx2_sp * 8 + avx512_sp * 16
    fp64_ops = scalar_dp + sse_dp * 2 + avx2_dp * 4 + avx512_dp * 8

    if uncore_events:
        rd = g("cas_count_read")   # uncore_imc/ prefix stripped by key normalisation
        wr = g("cas_count_write")
        dram_bytes = (rd + wr) * 64   # each CAS = 1 cache line = 64 bytes
    else:
        dram_bytes = g("llc-load-misses") * 64   # read-only proxy

    # L3 traffic: all LLC loads (hits + misses) × 64-byte cache line
    l3_bytes = g("llc-loads") * 64

    if fp32_ops == 0 and fp64_ops == 0 and dram_bytes == 0:
        return None

    duration_ns = 0
    m = re.search(r"([\d,.]+)\s+seconds\s+time\s+elapsed", text, re.I)
    if m:
        try:
            duration_ns = int(float(m.group(1).replace(",", ".")) * 1e9)
        except ValueError:
            pass

    return KernelCounters(
        kernel_name="(whole program)",
        fp32_ops=fp32_ops, fp64_ops=fp64_ops, fp16_ops=0.0,
        dram_bytes=dram_bytes, l2_bytes=0.0, l1_bytes=0.0,
        duration_ns=duration_ns,
        l3_bytes=l3_bytes,
        source="",   # caller sets this
    )


def _parse_perf_stat_amd(text: str) -> Optional[KernelCounters]:
    """
    Parse AMD perf stat output.

    fp_ret_sse_avx_ops.all counts element-level FP operations directly —
    a 256-bit FMA adds 16 to the counter (8 SP × 2 FP ops per FMA).
    The .fp64_* sub-events let us split FP32 from FP64.
    """
    ev: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or not line[0].isdigit():
            continue
        parts = re.split(r"\s{2,}", line, maxsplit=2)
        if len(parts) < 2:
            continue
        raw = re.sub(r"[^\d]", "", parts[0])
        try:
            val = float(raw)
        except ValueError:
            continue
        key = re.sub(r"^[\w-]+/", "", parts[1].rstrip("/")).lower().strip()
        key = re.sub(r":[ukpHG]+$", "", key)   # strip perf qualifiers (:u, :k, :p, etc.)
        if key:
            ev[key] = ev.get(key, 0.0) + val

    def g(k: str) -> float:
        return ev.get(k.lower(), 0.0)

    all_ops  = g("fp_ret_sse_avx_ops.all")
    fp64_add = g("fp_ret_sse_avx_ops.fp64_add_sub_ops")
    fp64_mul = g("fp_ret_sse_avx_ops.fp64_mult_ops")
    fp64_fma = g("fp_ret_sse_avx_ops.fp64_mult_add_ops") * 2  # FMA = 2 ops
    fp64_ops = fp64_add + fp64_mul + fp64_fma
    fp32_ops = max(0.0, all_ops - fp64_ops)   # remainder is FP32

    dram_bytes = g("llc-load-misses") * 64
    l3_bytes   = g("llc-loads") * 64

    if all_ops == 0 and dram_bytes == 0:
        return None

    duration_ns = 0
    m = re.search(r"([\d,.]+)\s+seconds\s+time\s+elapsed", text, re.I)
    if m:
        try:
            duration_ns = int(float(m.group(1).replace(",", ".")) * 1e9)
        except ValueError:
            pass

    return KernelCounters(
        kernel_name="(whole program)",
        fp32_ops=fp32_ops, fp64_ops=fp64_ops, fp16_ops=0.0,
        dram_bytes=dram_bytes, l2_bytes=0.0, l1_bytes=0.0,
        duration_ns=duration_ns,
        l3_bytes=l3_bytes,
        source="",
    )


# ── Dispatch ─────────────────────────────────────────────────────────────────

def collect(backend: str, command: list[str],
            env: dict | None = None) -> list[KernelCounters]:
    """
    Collect hardware counters for the given backend.
    Returns a list of KernelCounters (one per kernel for GPU, one total for CPU).

    Raises FileNotFoundError / CounterPermissionError with actionable messages.
    """
    if backend == "cuda":
        return collect_cuda(command, env)
    if backend == "rocm":
        return collect_rocm(command, env)
    if backend in ("cpu", "openmp", "opencl"):
        result = collect_cpu(command, env)
        return [result] if result else []
    raise ValueError(
        f"Hardware counter collection not supported for backend '{backend}'"
    )
