"""
Static assembly advisor for hprofiler.

Analyses KernelDisasm objects to surface missed optimisation opportunities in
x86-64, CUDA SASS, PTX, and AMDGCN (ROCm) assembly.  All checks are purely
static — no runtime data required — and are intentionally conservative so
every hint reflects something actionable in typical HPC C++ code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..disasm.extractor import KernelDisasm, DisasmLine


# ── Advice record ─────────────────────────────────────────────────────────────

@dataclass
class AsmAdvice:
    severity: str   # "crit" | "warn" | "info"
    category: str   # "vectorize" | "memory" | "compute" | "sync" | "register" | "branch"
    message:  str   # one-line summary
    detail:   str = ""  # longer explanation / fix

    @property
    def icon(self) -> str:
        return {"crit": "[!]", "warn": "[~]", "info": "[i]"}.get(self.severity, "[?]")

    @property
    def rich_color(self) -> str:
        return {"crit": "bold red", "warn": "yellow", "info": "cyan"}.get(
            self.severity, "white"
        )


# ── Dispatcher ────────────────────────────────────────────────────────────────

def advise(kd: "KernelDisasm") -> list[AsmAdvice]:
    """Return optimization hints for *kd*.  Empty list = nothing notable."""
    arch = kd.arch.lower()
    if arch in ("x86-64", "x86_64", "amd64", "x86", "cpu", "aarch64", "arm64"):
        return _advise_cpu(kd)
    if arch in ("sass", "cuda"):
        return _advise_sass(kd)
    if arch in ("ptx",):
        return _advise_ptx(kd)
    if arch in ("amdgcn", "rocm", "hip", "gcn"):
        return _advise_amdgcn(kd)
    return []


# ── x86-64 advisor ────────────────────────────────────────────────────────────

def _advise_cpu(kd: "KernelDisasm") -> list[AsmAdvice]:
    from ..disasm.classifier import InsnType

    lines = kd.lines
    n = len(lines)
    if n < 8:
        return []

    advice: list[AsmAdvice] = []

    # Count instruction type buckets
    cnt: dict = {t: 0 for t in InsnType}
    for ln in lines:
        cnt[ln.itype] = cnt.get(ln.itype, 0) + 1

    vec_total = (cnt[InsnType.VEC_SP] + cnt[InsnType.VEC_DP]
                 + cnt[InsnType.VEC_MEM] + cnt[InsnType.VECTOR])
    compute   = cnt[InsnType.COMPUTE]
    scalar    = cnt[InsnType.SCALAR]
    memory    = cnt[InsnType.MEMORY]
    control   = cnt[InsnType.CONTROL]

    vec_pct     = 100.0 * vec_total / n
    mem_pct     = 100.0 * memory    / n
    ctrl_pct    = 100.0 * control   / n
    compute_pct = 100.0 * compute   / n

    # R1 — Low vectorisation
    if vec_pct < 10.0 and n >= 20:
        advice.append(AsmAdvice(
            severity="warn", category="vectorize",
            message=f"Low vectorisation: only {vec_pct:.0f}% SIMD instructions",
            detail="Compile with -O3 -march=native to enable AVX2/AVX-512 "
                   "auto-vectorisation. Check vectorisation report with "
                   "-fopt-info-vec-missed. For manual control use "
                   "#pragma GCC ivdep or std::execution::par_unseq.",
        ))

    # R2 — High memory traffic
    if mem_pct > 35.0:
        advice.append(AsmAdvice(
            severity="warn", category="memory",
            message=f"Memory-heavy kernel: {mem_pct:.0f}% load/store instructions",
            detail="Consider AoS→SoA data layout, loop tiling/blocking to improve "
                   "cache reuse, or __builtin_prefetch / _mm_prefetch for "
                   "latency hiding on predictable access patterns.",
        ))

    # R3 — Register spills (stack accesses)
    spills = sum(1 for ln in lines if "[rsp" in ln.operands or "[rbp" in ln.operands)
    spill_pct = 100.0 * spills / n
    if spill_pct > 8.0:
        advice.append(AsmAdvice(
            severity="warn", category="register",
            message=f"Register pressure: {spills} stack spill/reload accesses ({spill_pct:.0f}%)",
            detail="Reduce live-variable count by splitting large loop bodies. "
                   "Use -fno-tree-sink or __attribute__((optimize(\"O1\"))) on "
                   "adjacent helper functions to free caller-saved registers.",
        ))

    # R4 — Scalar FP without FMA
    _SCALAR_FP = {"mulss", "addss", "subss", "mulsd", "addsd", "subsd",
                  "mulps", "addps", "subps", "mulpd", "addpd", "subpd"}
    scalar_fp = sum(1 for ln in lines if ln.mnemonic.lower() in _SCALAR_FP)
    if scalar_fp > 0 and compute_pct < 3.0 and n >= 15:
        advice.append(AsmAdvice(
            severity="info", category="compute",
            message=f"Scalar/legacy FP ({scalar_fp} insns) without FMA fusion",
            detail="Compile with -mfma to fuse mul+add into VFMADD231PS/SD. "
                   "FMA halves instruction count and rounding error for a*b+c "
                   "patterns. Or use std::fma() / _mm_fmadd_ps intrinsics.",
        ))

    # R5 — SSE (128-bit XMM) but not AVX (256-bit YMM/ZMM)
    has_ymm = any("ymm" in ln.operands.lower() or "zmm" in ln.operands.lower()
                  for ln in lines)
    has_xmm = any("xmm" in ln.operands.lower() for ln in lines)
    if has_xmm and not has_ymm and n >= 20:
        advice.append(AsmAdvice(
            severity="info", category="vectorize",
            message="128-bit SSE (XMM) used but not 256-bit AVX (YMM)",
            detail="Add -mavx2 or -march=native to double SIMD width. "
                   "Beware AVX↔SSE transition penalties if mixing code paths — "
                   "insert vzeroupper at domain boundaries.",
        ))

    # R6 — High branch density (check for calls too)
    if ctrl_pct > 15.0 and n >= 20:
        call_count = sum(1 for ln in lines if ln.mnemonic.lower().startswith("call"))
        if call_count > 3:
            advice.append(AsmAdvice(
                severity="info", category="branch",
                message=f"Frequent function calls in hot code ({call_count} CALL insns)",
                detail="Mark hot callees with __attribute__((always_inline)) or "
                       "raise -finline-limit=N. Profile with perf call-graph to "
                       "confirm call sites account for the samples.",
            ))
        else:
            advice.append(AsmAdvice(
                severity="info", category="branch",
                message=f"High branch density: {ctrl_pct:.0f}% control-flow instructions",
                detail="Consider #pragma GCC unroll N for small loops, "
                       "or predicated CMOV patterns instead of short branches. "
                       "Check for loop-carried dependencies preventing unrolling.",
            ))

    return advice


# ── CUDA SASS advisor ─────────────────────────────────────────────────────────

_S_GLOBAL  = re.compile(r'^(LDG|STG)\b',                        re.I)
_S_SHARED  = re.compile(r'^(LDS|STS|LDGSTS)\b',                 re.I)
_S_COMPUTE = re.compile(r'^(FFMA|DFMA|HMMA|BMMA|IMAD|HFMA|FMUL|FADD|DMUL|DADD)\b', re.I)
_S_BARRIER = re.compile(r'^BAR\b',                               re.I)
_S_ATOM    = re.compile(r'^(ATOM|RED)\b',                        re.I)
_S_ASYNC   = re.compile(r'^LDGSTS\b',                            re.I)
_S_HFMA    = re.compile(r'^HFMA\b',                              re.I)
_S_HMMA    = re.compile(r'^HMMA\b',                              re.I)


def _advise_sass(kd: "KernelDisasm") -> list[AsmAdvice]:
    lines = kd.lines
    n = len(lines)
    if n < 5:
        return []

    advice: list[AsmAdvice] = []

    global_mem = sum(1 for ln in lines if _S_GLOBAL.match(ln.mnemonic))
    shared_mem = sum(1 for ln in lines if _S_SHARED.match(ln.mnemonic))
    compute    = sum(1 for ln in lines if _S_COMPUTE.match(ln.mnemonic))
    barriers   = sum(1 for ln in lines if _S_BARRIER.match(ln.mnemonic))
    atomics    = sum(1 for ln in lines if _S_ATOM.match(ln.mnemonic))
    has_async  = any(_S_ASYNC.match(ln.mnemonic) for ln in lines)
    has_hfma   = any(_S_HFMA.match(ln.mnemonic) for ln in lines)
    has_hmma   = any(_S_HMMA.match(ln.mnemonic) for ln in lines)

    # Stall-weighted hotness: count insns with stall ≥ 10 (high-latency)
    high_stall = sum(1 for ln in lines if ln.stall_cycles >= 10)

    global_pct  = 100.0 * global_mem / n
    compute_pct = 100.0 * compute    / n
    barrier_pct = 100.0 * barriers   / n
    atomic_pct  = 100.0 * atomics    / n
    stall_pct   = 100.0 * high_stall / n

    # R1 — Global memory without shared memory tiling
    if global_pct > 20.0 and shared_mem < 3 and n >= 15:
        advice.append(AsmAdvice(
            severity="crit", category="memory",
            message=f"High global memory traffic ({global_pct:.0f}%) with no shared memory",
            detail="Tile the working set into shared memory (SHMEM). "
                   "LDG from global memory: 200–800 cycle latency per cache miss. "
                   "LDS from shared memory: ~20 cycles. "
                   "Pattern: load tile → __syncthreads() → compute from SHMEM.",
        ))

    # R2 — Low compute density (memory/control bound)
    if compute_pct < 15.0 and n >= 20:
        advice.append(AsmAdvice(
            severity="warn", category="compute",
            message=f"Low compute density: {compute_pct:.0f}% FMA/multiply-accumulate instructions",
            detail="Kernel may be memory- or control-bound. "
                   "Increase arithmetic intensity through register tiling, "
                   "loop unrolling (#pragma unroll), or reducing redundant "
                   "global loads by staging data in registers.",
        ))

    # R3 — High stall count (from SASS control word decode)
    if stall_pct > 20.0:
        advice.append(AsmAdvice(
            severity="warn", category="memory",
            message=f"{stall_pct:.0f}% of instructions have max-stall scheduling (cycles ≥ 10)",
            detail="High stall counts indicate the compiler expects latency bubbles. "
                   "Increase ILP by interleaving independent operations between "
                   "high-latency instructions (LDG, IMAD chains). "
                   "Occupancy tuning (more warps) hides latency via context-switching.",
        ))

    # R4 — Frequent block barriers
    if barrier_pct > 4.0:
        advice.append(AsmAdvice(
            severity="warn", category="sync",
            message=f"Frequent barriers: {barriers} BAR.SYNC ({barrier_pct:.0f}% of insns)",
            detail="Reduce synchronisation by merging work phases. "
                   "Use warp-level primitives (__shfl_sync, cooperative groups) "
                   "when only intra-warp communication is needed — they are "
                   "cheaper than full block barriers.",
        ))

    # R5 — Frequent global atomics
    if atomic_pct > 5.0:
        advice.append(AsmAdvice(
            severity="info", category="compute",
            message=f"Frequent atomics: {atomics} ATOM/RED instructions ({atomic_pct:.0f}%)",
            detail="Replace global atomics with per-block shared-memory reduction "
                   "followed by a single global atomic per block. "
                   "Reduces contention on the L2/DRAM atomic unit significantly.",
        ))

    # R6 — FP16 compute without tensor core instructions
    if has_hfma and not has_hmma:
        advice.append(AsmAdvice(
            severity="info", category="compute",
            message="FP16 arithmetic (HFMA) without tensor core instructions (HMMA)",
            detail="Use the wmma (warp matrix multiply) API or cuBLAS/cuDNN to "
                   "dispatch tensor core HMMA instructions. "
                   "Tensor core throughput is ~8× higher than scalar HFMA for "
                   "matrix multiply workloads (16×16×16 tiles per warp per cycle).",
        ))

    # R7 — No async prefetch for global loads
    if global_mem > 5 and not has_async and n >= 30:
        advice.append(AsmAdvice(
            severity="info", category="memory",
            message="No async copy (LDGSTS) — global loads are synchronous",
            detail="Use cuda::pipeline or __pipeline_memcpy_async to overlap "
                   "global memory loads with computation (requires sm_80+, Ampere). "
                   "Enables double-buffering: load next tile while computing current.",
        ))

    return advice


# ── PTX advisor ───────────────────────────────────────────────────────────────

def _advise_ptx(kd: "KernelDisasm") -> list[AsmAdvice]:
    lines = kd.lines
    n = len(lines)
    if n < 5:
        return []

    advice: list[AsmAdvice] = []

    global_ld  = sum(1 for ln in lines
                     if ln.mnemonic.lower().startswith("ld")
                     and ".global" in (ln.operands + ln.mnemonic).lower())
    global_st  = sum(1 for ln in lines
                     if ln.mnemonic.lower().startswith("st")
                     and ".global" in (ln.operands + ln.mnemonic).lower())
    shared_ops = sum(1 for ln in lines
                     if ".shared" in (ln.operands + ln.mnemonic).lower())
    atom_ops   = sum(1 for ln in lines if ln.mnemonic.lower() in ("atom", "red"))
    barriers   = sum(1 for ln in lines if ln.mnemonic.lower() == "bar")
    vec_ops    = sum(1 for ln in lines
                     if ".v2" in ln.mnemonic or ".v4" in ln.mnemonic
                     or ".v2" in ln.operands or ".v4" in ln.operands)

    global_total = global_ld + global_st
    global_pct   = 100.0 * global_total / n
    atom_pct     = 100.0 * atom_ops     / n
    barrier_pct  = 100.0 * barriers     / n
    vec_pct      = 100.0 * vec_ops      / n

    if global_pct > 20.0 and shared_ops < 3 and n >= 15:
        advice.append(AsmAdvice(
            severity="warn", category="memory",
            message=f"High global memory traffic ({global_pct:.0f}%) without .shared usage",
            detail="Move frequently reused data into .shared address space. "
                   "PTX .shared latency is ~5 ns vs ~200 ns for .global. "
                   "Note: PTX is a virtual ISA — the SASS backend may already "
                   "transform this; check SASS disassembly for the final code.",
        ))

    if atom_pct > 5.0:
        advice.append(AsmAdvice(
            severity="info", category="compute",
            message=f"Frequent atomic operations ({atom_ops}, {atom_pct:.0f}%)",
            detail="Replace global atomics with per-block .shared atomics "
                   "followed by a single global reduce per block. "
                   "Shared atomics serialise within the block, not across all blocks.",
        ))

    if barrier_pct > 3.0:
        advice.append(AsmAdvice(
            severity="info", category="sync",
            message=f"Frequent barriers ({barriers} bar.sync, {barrier_pct:.0f}%)",
            detail="Use warp shuffles (__shfl_sync) instead of block barriers "
                   "when only intra-warp communication is needed.",
        ))

    if vec_pct < 5.0 and global_total > 5 and n >= 20:
        advice.append(AsmAdvice(
            severity="info", category="memory",
            message="No vectorised memory access (.v2/.v4 loads) detected",
            detail="Use int2/int4/float2/float4 types for coalesced vector loads. "
                   "A single ld.global.v4.f32 fetches 128 bits — "
                   "4× fewer memory transactions vs four scalar loads.",
        ))

    return advice


# ── AMDGCN advisor ────────────────────────────────────────────────────────────

def _advise_amdgcn(kd: "KernelDisasm") -> list[AsmAdvice]:
    lines = kd.lines
    n = len(lines)
    if n < 5:
        return []

    advice: list[AsmAdvice] = []

    valu    = sum(1 for ln in lines if ln.mnemonic.lower().startswith("v_"))
    salu    = sum(1 for ln in lines if ln.mnemonic.lower().startswith("s_")
                                       and "waitcnt" not in ln.mnemonic.lower()
                                       and "barrier" not in ln.mnemonic.lower())
    lds     = sum(1 for ln in lines if ln.mnemonic.lower().startswith("ds_"))
    global_ = sum(1 for ln in lines
                  if any(ln.mnemonic.lower().startswith(p)
                         for p in ("buffer_", "global_", "flat_")))
    waitcnt = sum(1 for ln in lines if "waitcnt" in ln.mnemonic.lower())

    valu_pct   = 100.0 * valu    / n
    global_pct = 100.0 * global_ / n
    wait_pct   = 100.0 * waitcnt / n

    # R1 — Low VALU density (scalar/branch dominated)
    if valu_pct < 30.0 and n >= 20:
        advice.append(AsmAdvice(
            severity="warn", category="compute",
            message=f"Low VALU utilisation: {valu_pct:.0f}% VALU vs {100.0*salu/n:.0f}% SALU",
            detail="VALU executes across all 64 wavefront lanes in parallel; "
                   "SALU is scalar and sequential. "
                   "Minimise per-thread branching and move uniform computations "
                   "to SALU where possible (compiler usually handles this). "
                   "Divergent control flow forces VALU underutilisation.",
        ))

    # R2 — Global memory without LDS tiling
    if global_pct > 20.0 and lds < 5 and n >= 15:
        advice.append(AsmAdvice(
            severity="crit", category="memory",
            message=f"High global memory ({global_pct:.0f}%) without LDS (shared memory)",
            detail="Stage frequently reused data in LDS (Local Data Share). "
                   "LDS bandwidth is roughly 100× higher than global memory. "
                   "Use __local pointers in OpenCL or __shared__ in HIP to "
                   "tile the working set.",
        ))

    # R3 — Frequent s_waitcnt (memory latency stalls)
    if wait_pct > 5.0:
        advice.append(AsmAdvice(
            severity="warn", category="memory",
            message=f"Frequent s_waitcnt ({waitcnt} insns, {wait_pct:.0f}%) — latency stalls",
            detail="s_waitcnt serialises the wavefront waiting for memory. "
                   "Increase ILP by issuing multiple independent loads before "
                   "the first use. Alternatively, increase occupancy so the "
                   "hardware can context-switch to other wavefronts.",
        ))

    return advice
