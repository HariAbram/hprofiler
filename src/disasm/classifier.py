"""
Instruction-type classifier for x86-64, SASS (NVIDIA GPU),
AMDGCN (ROCm GPU), and PTX (CUDA IR).

Each classifier maps (mnemonic, operands) → InsnType.
Used to colour-code disassembly in the TUI and compute instruction-mix stats.
"""

from __future__ import annotations
import re
from enum import Enum


class InsnType(str, Enum):
    VEC_SP  = "vec_sp"   # FP32 / single-precision SIMD (vaddps, vmulss, …)
    VEC_DP  = "vec_dp"   # FP64 / double-precision SIMD (vaddpd, vmulsd, …)
    VEC_MEM = "vec_mem"  # SIMD load / store (vmovaps, vgatherdps, vbroadcastss, …)
    VECTOR  = "vector"   # integer / misc SIMD — catch-all (vpxor, vpandn, …)
    SCALAR  = "scalar"   # ordinary integer or scalar FP
    MEMORY  = "memory"   # scalar load / store / atomic
    CONTROL = "control"  # branch / call / return / exit
    SYNC    = "sync"     # barriers, fences, synchronisation
    COMPUTE = "compute"  # FMA / multiply-accumulate — heavy-compute ALU
    OTHER   = "other"


# Rich colour per instruction type (used in the TUI)
ITYPE_COLOR: dict[InsnType, str] = {
    InsnType.VEC_SP:  "bright_green",
    InsnType.VEC_DP:  "cyan",
    InsnType.VEC_MEM: "orange3",
    InsnType.VECTOR:  "green4",
    InsnType.SCALAR:  "steel_blue1",
    InsnType.MEMORY:  "yellow",
    InsnType.CONTROL: "magenta",
    InsnType.SYNC:    "red",
    InsnType.COMPUTE: "bright_blue",
    InsnType.OTHER:   "grey62",
}

# Short label for the side annotation column and the mix bar
ITYPE_LABEL: dict[InsnType, str] = {
    InsnType.VEC_SP:  "vsp",
    InsnType.VEC_DP:  "vdp",
    InsnType.VEC_MEM: "vld",
    InsnType.VECTOR:  "vec",
    InsnType.SCALAR:  "scl",
    InsnType.MEMORY:  "mem",
    InsnType.CONTROL: "ctl",
    InsnType.SYNC:    "syn",
    InsnType.COMPUTE: "fma",
    InsnType.OTHER:   "   ",
}

# ── x86 / x86-64 ─────────────────────────────────────────────────────────────

_X86_VEC_MNE  = re.compile(
    r'^v[a-z]|^p[a-z]{2,}(?:b|w|d|q)',     # VEX prefix or packed SSE
    re.I
)
_X86_VEC_OPS  = re.compile(r'\b(ymm|zmm|[yk][0-7])\b', re.I)
_X86_SSE_OPS  = re.compile(r'\bxmm[0-9]+\b', re.I)
_X86_MEM_OPS  = re.compile(r'\[')
_X86_MEM_MNE  = re.compile(
    r'^(mov[a-z]*|lea|push|pop|ld[a-z]*|st[a-z]*|prefetch[a-z]*|movnt[a-z]*)$',
    re.I
)
_X86_CTL      = re.compile(
    r'^(j[a-z]+|call[a-z]*|ret[a-z]*|loop[a-z]*|int|syscall|hlt|ud2)$',
    re.I
)
_X86_SYNC     = re.compile(r'^(mfence|sfence|lfence|pause|lock)$', re.I)
_X86_FMA      = re.compile(r'^(fma|vfma|imul[a-z]*|idiv[a-z]*|mulx|adox|adcx)', re.I)

# Vector sub-type detection for x86 -------------------------------------------
# Primary-memory mnemonics: vmov*, vbroadcast*, vgather*, vscatter*, vmaskmov*
_X86_VMEM_MNE = re.compile(
    r'^v(?:mov|broadcast|gather|scatter|maskmov|pmaskmov|expand|compress)',
    re.I
)
# Mnemonic suffix indicates precision (matched at end, optional trailing digits)
_X86_SP_MNE   = re.compile(r'(?:ps|ss)\d*$', re.I)   # packed/scalar single  (FP32)
_X86_DP_MNE   = re.compile(r'(?:pd|sd)\d*$', re.I)   # packed/scalar double  (FP64)


def _x86_vec_subtype(mnemonic: str, operands: str) -> InsnType:
    """Return the vector sub-type for an instruction already known to be SIMD."""
    m = mnemonic.strip().lower()
    # Pure memory ops: vmov*, vbroadcast*, vgather*, vscatter*, …
    if _X86_VMEM_MNE.match(m):
        return InsnType.VEC_MEM
    if _X86_SP_MNE.search(m):
        return InsnType.VEC_SP
    if _X86_DP_MNE.search(m):
        return InsnType.VEC_DP
    return InsnType.VECTOR   # integer SIMD (vpxor, vpand, vpcmpeq, …)


def classify_x86(mnemonic: str, operands: str = "") -> InsnType:
    m = mnemonic.strip()
    if _X86_SYNC.match(m):
        return InsnType.SYNC
    if _X86_CTL.match(m):
        return InsnType.CONTROL
    # FMA / heavy-compute checked before SIMD so vfmadd*ps stays in COMPUTE
    if _X86_FMA.match(m):
        return InsnType.COMPUTE
    # YMM / ZMM → SIMD sub-type
    if _X86_VEC_OPS.search(operands):
        return _x86_vec_subtype(m, operands)
    # VEX-encoded mnemonic → SIMD (but not the zero-upper helpers)
    if _X86_VEC_MNE.match(m) and m.lower() not in ("vzeroupper", "vzeroall"):
        return _x86_vec_subtype(m, operands)
    # XMM operands → SIMD
    if _X86_SSE_OPS.search(operands):
        return _x86_vec_subtype(m, operands)
    # Scalar loads/stores that touch memory
    if _X86_MEM_OPS.search(operands) and _X86_MEM_MNE.match(m):
        return InsnType.MEMORY
    return InsnType.SCALAR


# ── SASS (NVIDIA GPU PTX-compiled assembly) ───────────────────────────────────

_SASS_MEM  = re.compile(r'^(LDG|STG|LDS|STS|LDL|STL|LD|ST|RED|ATOM|LDC|LDGSTS)\b')
_SASS_CTL  = re.compile(r'^(BRA|CAL|RET|EXIT|BRX|JCAL|SYNC|SSY|BREAK|PRET|LONGJMP)\b')
_SASS_FMA  = re.compile(r'^(FFMA|DFMA|HFMA|FMUL|FADD|FDIV|DMUL|DADD|IMAD|IMUL|XMAD|HMMA|BMMA)\b')
_SASS_SYNC = re.compile(r'^(BAR|MEMBAR|CCTL|DEPBAR|SETLMEMBASE)\b')


def classify_sass(mnemonic: str, operands: str = "") -> InsnType:
    m = mnemonic.strip().upper()
    if _SASS_MEM.match(m):   return InsnType.MEMORY
    if _SASS_SYNC.match(m):  return InsnType.SYNC
    if _SASS_CTL.match(m):   return InsnType.CONTROL
    if _SASS_FMA.match(m):   return InsnType.COMPUTE
    return InsnType.SCALAR


# ── AMDGCN (ROCm / RDNA) ─────────────────────────────────────────────────────

def classify_amdgcn(mnemonic: str, operands: str = "") -> InsnType:
    m = mnemonic.strip().lower()
    if m.startswith("v_"):
        if "_f32" in m or "_f16" in m or "_bf16" in m or "_f8" in m:
            return InsnType.VEC_SP
        if "_f64" in m:
            return InsnType.VEC_DP
        return InsnType.VECTOR   # integer lane ops (v_add_u32, v_lshl, …)
    if m.startswith("s_"):
        if "branch" in m or "cbranch" in m or m == "s_endpgm":
            return InsnType.CONTROL
        if "waitcnt" in m or "barrier" in m:
            return InsnType.SYNC
        return InsnType.SCALAR
    if any(m.startswith(p) for p in ("ds_", "flat_", "global_", "buffer_", "scratch_")):
        return InsnType.MEMORY
    return InsnType.OTHER


# ── PTX (CUDA IR text) ────────────────────────────────────────────────────────

_PTX_VEC  = re.compile(r'\.(v2|v4)\b')
_PTX_MEM  = re.compile(r'^(ld|st|atom|red|prefetch|prefetchu|suld|sust)\b', re.I)
_PTX_CTL  = re.compile(r'^(bra|call|ret|exit|brx|setp|selp)\b', re.I)
_PTX_SYNC = re.compile(r'^(bar|membar|fence|atom)\b', re.I)
_PTX_FMA  = re.compile(r'^(fma|mad|mul|add|div)\b', re.I)


def classify_ptx(mnemonic: str, operands: str = "") -> InsnType:
    m = mnemonic.strip().lower()
    if _PTX_VEC.search(m):    return InsnType.VECTOR
    if _PTX_SYNC.match(m):    return InsnType.SYNC
    if _PTX_CTL.match(m):     return InsnType.CONTROL
    if _PTX_MEM.match(m):     return InsnType.MEMORY
    if _PTX_FMA.match(m):     return InsnType.COMPUTE
    return InsnType.SCALAR


# ── AArch64 ──────────────────────────────────────────────────────────────────

# SIMD/FP register reference: v0..v31, q0..q31, d0..d31, h/b regs
_A64_VEC_OPS = re.compile(r'\b[vqdhb]\d+\b', re.I)

# Only mnemonics that are EXCLUSIVELY SIMD (never have scalar-only forms).
_A64_VEC_MNE = re.compile(
    r'^([su]?qadd|[su]?qsub|[su]?hadd|[su]?hsub|[su]?abd|[su]?max|[su]?min|'
    r'[su]?addl[pv]?|[su]?addw|[su]?adalp|[su]?addv|'
    r'[su]?mlal[12]?|[su]?mlsl[12]?|'
    r'addp|addv|maxp|minp|[su]qrdmulh|[su]qrshl|[su]qrshrn[12]?|'
    r'rev(16|32|64)|ext|dup|ins|umov|smov|movi|mvni|'
    r'zip[12]|uzp[12]|trn[12]|tbl|tbx|'
    r'aese|aesd|aesimc|aesmc|sha1[chm]|sha1h|sha256[hsu]|'
    r'pmul|pmull[12]?|'
    r'ld[1-4][rq]?|st[1-4]|ld[1-4]r)$',
    re.I
)
_A64_MEM = re.compile(
    r'^(ldr[bhwx]?|str[bhwx]?|ldp|stp|ldur[bhwx]?|stur[bhwx]?|'
    r'ldnp|stnp|ldaxr[bhwx]?|stlxr[bhwx]?|ldxr[bhwx]?|stxr[bhwx]?|'
    r'ldar[bhwx]?|stlr[bhwx]?|cas[a-z]*|swp[a-z]*|ld[auc][a-z]*|'
    r'prfm|prfum)$',
    re.I
)
_A64_CTL = re.compile(
    r'^(b|bl|br|blr|ret|eret|'
    r'b\.(eq|ne|cs|cc|hs|lo|mi|pl|vs|vc|hi|ls|ge|lt|gt|le|al)|'
    r'cbz|cbnz|tbz|tbnz|svc|hvc|smc|brk|hlt|dcps[123])$',
    re.I
)
_A64_SYNC = re.compile(
    r'^(dmb|dsb|isb|sevl|wfe|wfi|yield|sev|clrex)$',
    re.I
)
_A64_FMA = re.compile(
    r'^(fmadd|fmsub|fnmadd|fnmsub|'
    r'madd|msub|smaddl|smsubl|umaddl|umsubl|mul|mneg|smull|smulh|umull|umulh)$',
    re.I
)

# AArch64 vector sub-type helpers ─────────────────────────────────────────────
# NEON ld1/st1..ld4/st4 (including replicate forms) and SVE ld1w/st1d/ldff1…
_A64_VEC_MEM_MNE = re.compile(
    r'^(?:ld[1-4]r?q?|st[1-4]|ld1[wdhb]|ldff1[wdhb]|ldnt1[wdhb]|'
    r'st1[wdhb]|stnt1[wdhb])$',
    re.I,
)
# Lane qualifiers: NEON v0.2d/.1d or SVE z0.d  →  FP64
_A64_LANE_DP = re.compile(r'\bv\d+\.(?:2d|1d)\b|\bz\d+\.d\b', re.I)
# Lane qualifiers: NEON v0.4s/.2s/.8h/.4h or SVE z0.s/.h  →  FP32 / FP16
_A64_LANE_SP = re.compile(
    r'\bv\d+\.(?:[24]s|[48]h)\b|\bz\d+\.(?:s|h)\b', re.I
)


def _a64_vec_subtype(mnemonic: str, operands: str) -> InsnType:
    """Return the SIMD sub-type for an AArch64 instruction known to be SIMD."""
    m = mnemonic.strip().lower()
    if _A64_VEC_MEM_MNE.match(m):      return InsnType.VEC_MEM
    if _A64_LANE_DP.search(operands):  return InsnType.VEC_DP
    if _A64_LANE_SP.search(operands):  return InsnType.VEC_SP
    return InsnType.VECTOR              # integer SIMD or unknown lane width


def classify_aarch64(mnemonic: str, operands: str = "") -> InsnType:
    m = mnemonic.strip().lower()
    if _A64_SYNC.match(m):    return InsnType.SYNC
    if _A64_CTL.match(m):     return InsnType.CONTROL
    # FMA before SIMD so scalar fmadd stays COMPUTE, not mistaken for VECTOR
    if _A64_FMA.match(m):     return InsnType.COMPUTE
    if _A64_VEC_OPS.search(operands): return _a64_vec_subtype(m, operands)
    if _A64_VEC_MNE.match(m): return _a64_vec_subtype(m, operands)
    if _A64_MEM.match(m):     return InsnType.MEMORY
    return InsnType.SCALAR


# ── RISC-V 64 (RV64GV) ───────────────────────────────────────────────────────

# Vector FMA: vfmadd, vfmsub, vfnmadd, vfnmsub, vfwmacc, vfnmacc, …
_RV_FMA_VEC  = re.compile(
    r'^vf(?:madd|msub|nmadd|nmsub|wmacc|nmacc|wnmacc|wnmsac)', re.I
)
# Vector memory: vle8/16/32/64, vse32, vlse, vluxei, vlm, vl1r–vl8r, …
_RV_VEC_MEM  = re.compile(
    r'^v(?:le\d|lse\d|luxei\d|loxei\d|lm\b|l[1-8]r|'
    r'se\d|sse\d|suxei\d|soxei\d|sm\b|s[1-8]r)',
    re.I,
)
# Scalar FP FMA: fmadd.s, fmsub.d, fnmadd.s, fnmsub.d
_RV_FMA_SCAL = re.compile(r'^fn?m(?:add|sub)\.[sd]', re.I)
# Scalar / FP memory loads & stores
_RV_MEM      = re.compile(
    r'^(?:l[bhwdtq]u?|s[bhwdtq]|flw|fld|flh|fsw|fsd|fsh|'
    r'lr\.[wd]|sc\.[wd]|amo\w+\.[wd]u?)$',
    re.I,
)
# Control flow
_RV_CTL      = re.compile(
    r'^(?:beq|bne|bltu?|bgeu?|jal[r]?|ret|ecall|ebreak|[msu]ret|wfi|'
    r'sfence\.\w+|hfence\.\w+|c\.j\w*)$',
    re.I,
)
# Fence / sync
_RV_SYNC     = re.compile(r'^fence(?:\.[it])?$', re.I)


def classify_rv64(mnemonic: str, operands: str = "") -> InsnType:
    m = mnemonic.strip().lower()
    if _RV_SYNC.match(m):      return InsnType.SYNC
    if _RV_CTL.match(m):       return InsnType.CONTROL
    # FMA before generic vector/FP so vfmadd → COMPUTE, not VEC_SP
    if _RV_FMA_VEC.match(m):   return InsnType.COMPUTE
    if _RV_FMA_SCAL.match(m):  return InsnType.COMPUTE
    if _RV_VEC_MEM.match(m):   return InsnType.VEC_MEM
    if m.startswith("vf"):     return InsnType.VEC_SP    # vfadd, vfmul, … (FP RVV)
    if m.startswith("v"):      return InsnType.VECTOR    # vadd, vmul, … (integer RVV)
    if _RV_MEM.match(m):       return InsnType.MEMORY
    if m.startswith("f"):      return InsnType.SCALAR    # fadd.s, fmul.d, fsgnj, …
    return InsnType.SCALAR


# ── Dispatcher ────────────────────────────────────────────────────────────────

def classify(arch: str, mnemonic: str, operands: str = "") -> InsnType:
    a = arch.lower()
    if a in ("x86", "x86-64", "x86_64", "amd64", "i386", "cpu"):
        return classify_x86(mnemonic, operands)
    if a in ("sass", "cuda"):
        return classify_sass(mnemonic, operands)
    if a in ("amdgcn", "rocm", "hip", "gcn"):
        return classify_amdgcn(mnemonic, operands)
    if a in ("ptx",):
        return classify_ptx(mnemonic, operands)
    if a in ("aarch64", "arm64", "armv8", "arm"):
        return classify_aarch64(mnemonic, operands)
    if a in ("rv64", "riscv64", "riscv", "rv32"):
        return classify_rv64(mnemonic, operands)
    return InsnType.OTHER
