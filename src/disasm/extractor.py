"""
Disassembly extraction for all supported backends.

Backend → tool mapping:
  cpu / opencl-cpu (AoT ELF)     objdump -d / llvm-objdump
  opencl-cpu (ACPP SSCP .jit.so) objdump -d  on the cached .so
  cuda (AoT)                     cuobjdump --dump-sass + --dump-ptx
  cuda (JIT cubin captured)      nvdisasm on a saved temp file
  rocm / hip (AoT)               llvm-objdump --triple=amdgcn-amd-amdhsa

JIT cubin path (CUDA):
  cuda_hook.c intercepts cuModuleLoadData / cuModuleLoadDataEx.
  The binary is saved to /tmp/hprofiler_cubin_<pid>_<n>.bin.
  collect_disasm() picks those up automatically.
"""

from __future__ import annotations
import re, os, shutil, subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .classifier import InsnType, classify, ITYPE_COLOR, ITYPE_LABEL


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class DisasmLine:
    addr:     int      = 0
    mnemonic: str      = ""
    operands: str      = ""
    itype:    InsnType = InsnType.OTHER
    comment:  str      = ""
    raw:      str      = ""   # full original text line

    @property
    def text(self) -> str:
        return f"{self.mnemonic}  {self.operands}".strip()


@dataclass
class KernelDisasm:
    name:   str
    arch:   str                       # 'x86-64' | 'sass' | 'ptx' | 'amdgcn'
    source: str                       # path to the file that was disassembled
    lines:  list[DisasmLine] = field(default_factory=list)

    def itype_counts(self) -> dict[InsnType, int]:
        counts: dict[InsnType, int] = {}
        for ln in self.lines:
            if ln.itype != InsnType.OTHER:
                counts[ln.itype] = counts.get(ln.itype, 0) + 1
        return counts

    def itype_pcts(self) -> dict[InsnType, float]:
        counts = self.itype_counts()
        total  = sum(counts.values()) or 1
        return {t: 100.0 * c / total for t, c in counts.items()}

    def total_insns(self) -> int:
        return sum(self.itype_counts().values())


# ── Shared helpers ────────────────────────────────────────────────────────────

def _run(cmd: list[str], timeout: int = 45) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout
    except Exception:
        return ""


_tool_cache: dict[tuple[str, ...], Optional[str]] = {}
_nm_cache: dict[str, dict[str, tuple[int, int]]] = {}


def _tool(*names: str) -> Optional[str]:
    """Return the first tool name that exists on PATH (cached)."""
    if names not in _tool_cache:
        _tool_cache[names] = next((n for n in names if shutil.which(n)), None)
    return _tool_cache[names]


def _nm_load(binary_path: str) -> dict[str, tuple[int, int]]:
    """Parse nm output for binary_path, caching by path."""
    if binary_path not in _nm_cache:
        text = _run(["nm", "-S", "--defined-only", binary_path])
        syms: dict[str, tuple[int, int]] = {}
        for line in text.splitlines():
            parts = line.split()
            if len(parts) == 4 and parts[2] in ("T", "t", "W", "w"):
                try:
                    syms[parts[3]] = (int(parts[0], 16), int(parts[1], 16))
                except ValueError:
                    pass
        _nm_cache[binary_path] = syms
    return _nm_cache[binary_path]


# ── x86-64 ELF (CPU + .jit.so) ───────────────────────────────────────────────

# objdump intel-syntax line:
#   addr:  mnemonic [operands]  [# comment]
# (--no-show-raw-insn strips the hex bytes)
_OBJ_LINE = re.compile(
    r'^\s*([0-9a-f]+):\s+'
    r'([a-z][a-z0-9]*(?:\.[a-z0-9]+)*)'   # mnemonic (possibly with suffixes)
    r'(?:\s+(.*?))?'                        # optional operands
    r'(?:\s*#\s*(.*))?$',                   # optional comment
    re.I,
)
_SYM_HDR = re.compile(r'^[0-9a-f]+ <(.+)>:')


def _parse_objdump_x86(text: str, symbol: Optional[str] = None,
                        arch: str = "x86-64") -> list[DisasmLine]:
    lines: list[DisasmLine] = []
    in_sym = (symbol is None)

    for raw in text.splitlines():
        m = _SYM_HDR.match(raw)
        if m:
            sym = m.group(1)
            if symbol is None:
                in_sym = True
            else:
                # Match short name anywhere in the mangled symbol
                in_sym = (symbol in sym or sym.endswith(symbol))
            continue
        if not in_sym:
            continue
        # Skip pure label / blank lines
        if not raw.strip() or raw.strip().endswith(":"):
            continue

        m = _OBJ_LINE.match(raw.rstrip())
        if not m:
            continue
        addr_s, mnem, ops, comment = m.groups()
        ops     = (ops or "").strip()
        comment = (comment or "").strip()

        # Strip trailing comment that crept into operands
        if "#" in ops and not comment:
            ops, _, comment = ops.partition("#")
            ops     = ops.strip()
            comment = comment.strip()

        itype = classify(arch, mnem, ops)
        lines.append(DisasmLine(
            addr=int(addr_s, 16),
            mnemonic=mnem, operands=ops, itype=itype,
            comment=comment, raw=raw,
        ))
    return lines


def _disasm_elf_capstone(path: str, sym_addr: int, sym_size: int,
                         sym_name: str, arch: str = "auto") -> Optional[KernelDisasm]:
    """
    Fast path: disassemble a single symbol using capstone + direct ELF read.
    Reads only the bytes for the target function — no subprocess needed.
    Falls back to None if capstone is unavailable or arch is unsupported.

    arch="auto" detects x86-64 vs AArch64 from the ELF e_machine field.
    Explicit values: "x86-64", "x86_64", "amd64", "x86", "aarch64", "arm64".
    """
    try:
        import capstone  # type: ignore[import]
    except ImportError:
        return None

    try:
        import struct
        with open(path, "rb") as f:
            elf = f.read()

        if len(elf) < 64:
            return None

        # Detect architecture from ELF e_machine (offset 18, 2 bytes LE).
        # EM_X86_64 = 62, EM_AARCH64 = 183, EM_RISCV = 243
        e_machine, = struct.unpack_from("<H", elf, 18)

        if arch == "auto":
            if e_machine == 62:    arch = "x86-64"
            elif e_machine == 183: arch = "aarch64"
            elif e_machine == 243: arch = "rv64"
            else:                  return None
        elif arch not in ("x86-64", "x86_64", "amd64", "x86",
                          "aarch64", "arm64", "armv8",
                          "rv64", "riscv64", "riscv"):
            return None

        if arch in ("aarch64", "arm64", "armv8"):
            cs_arch = capstone.CS_ARCH_ARM64
            cs_mode = capstone.CS_MODE_ARM
            classify_arch = "aarch64"
            out_arch = "aarch64"
        elif arch in ("rv64", "riscv64", "riscv"):
            # capstone ≥ 5.0 required for RISC-V
            if not hasattr(capstone, "CS_ARCH_RISCV"):
                return None
            cs_arch = capstone.CS_ARCH_RISCV
            cs_mode = capstone.CS_MODE_RISCV64
            classify_arch = "rv64"
            out_arch = "rv64"
        else:
            cs_arch = capstone.CS_ARCH_X86
            cs_mode = capstone.CS_MODE_64
            classify_arch = "x86-64"
            out_arch = "x86-64"

        # Walk ELF64 section headers to convert sym_addr → file offset.
        e_shoff,    = struct.unpack_from("<Q", elf, 40)
        e_shentsize,= struct.unpack_from("<H", elf, 58)
        e_shnum,    = struct.unpack_from("<H", elf, 60)

        file_offset = None
        for i in range(e_shnum):
            sh = e_shoff + i * e_shentsize
            sh_type,  = struct.unpack_from("<I", elf, sh + 4)
            sh_addr,  = struct.unpack_from("<Q", elf, sh + 16)
            sh_offset,= struct.unpack_from("<Q", elf, sh + 24)
            sh_size,  = struct.unpack_from("<Q", elf, sh + 32)
            if sh_type == 1 and sh_addr <= sym_addr < sh_addr + sh_size:
                file_offset = sh_offset + (sym_addr - sh_addr)
                break

        if file_offset is None:
            return None

        code = elf[file_offset: file_offset + sym_size]
        if not code:
            return None

        md = capstone.Cs(cs_arch, cs_mode)
        md.detail = False

        lines: list[DisasmLine] = []
        for insn in md.disasm(code, sym_addr):
            mnem  = insn.mnemonic
            ops   = insn.op_str
            itype = classify(classify_arch, mnem, ops)
            lines.append(DisasmLine(
                addr=insn.address, mnemonic=mnem, operands=ops,
                itype=itype, raw=f"{insn.address:#x}: {mnem}  {ops}",
            ))

        if not lines:
            return None
        return KernelDisasm(name=sym_name, arch=out_arch, source=path, lines=lines)

    except Exception:
        return None


def _elf_arch(path: str) -> str:
    """Return the classifier arch name for an ELF binary (reads e_machine)."""
    try:
        import struct
        with open(path, "rb") as f:
            header = f.read(20)
        e_machine, = struct.unpack_from("<H", header, 18)
        if e_machine == 62:  return "x86-64"
        if e_machine == 183: return "aarch64"
        if e_machine == 243: return "rv64"
    except Exception:
        pass
    return "x86-64"


def disasm_elf(path: str, symbol: Optional[str] = None,
               arch: str = "x86-64") -> Optional[KernelDisasm]:
    """Disassemble an ELF file (or a specific symbol) with objdump / llvm-objdump."""
    if not Path(path).exists():
        return None

    tool = _tool("llvm-objdump", "objdump")
    if not tool:
        return None

    # Build the command — try per-symbol flag first (faster for large binaries)
    base_cmd = [tool, "-d", "--no-show-raw-insn", "-M", "intel", "--wide"]
    if symbol:
        flag = "--disassemble-symbols" if "llvm" in tool else "--disassemble"
        text = _run(base_cmd + [f"{flag}={symbol}", path])
        if not text.strip():          # flag not supported or symbol not found
            text = _run(base_cmd + [path])
    else:
        text = _run(base_cmd + [path])

    inst_lines = _parse_objdump_x86(text, symbol, arch)
    if not inst_lines:
        return None
    return KernelDisasm(
        name=symbol or Path(path).stem,
        arch=arch, source=path, lines=inst_lines,
    )


# ── CUDA SASS (cuobjdump) ─────────────────────────────────────────────────────

_SASS_KERN = re.compile(r'^\s*Function\s*:\s*(\S+)')
_SASS_LINE = re.compile(
    r'/\*([0-9a-f]+)\*/\s+([A-Z][A-Z0-9_]+(?:\.[A-Z0-9_.]+)*)\s*(.*?)\s*;?$',
    re.I,
)


def _parse_sass(text: str) -> dict[str, list[DisasmLine]]:
    kernels: dict[str, list[DisasmLine]] = {}
    current: Optional[str] = None
    for raw in text.splitlines():
        m = _SASS_KERN.match(raw)
        if m:
            current = m.group(1)
            kernels.setdefault(current, [])
            continue
        if current is None:
            continue
        m = _SASS_LINE.search(raw)
        if m:
            addr_s, mnem, ops = m.groups()
            itype = classify("sass", mnem, ops)
            kernels[current].append(DisasmLine(
                addr=int(addr_s, 16),
                mnemonic=mnem, operands=ops.strip(),
                itype=itype, raw=raw,
            ))
    return kernels


def disasm_cuda_sass(binary_path: str) -> dict[str, KernelDisasm]:
    """Extract SASS from a CUDA binary using cuobjdump."""
    if not _tool("cuobjdump"):
        return {}
    text = _run(["cuobjdump", "--dump-sass", binary_path], timeout=60)
    return {
        name: KernelDisasm(name=name, arch="sass", source=binary_path, lines=lns)
        for name, lns in _parse_sass(text).items()
        if lns
    }


def disasm_cuda_cubin(cubin_path: str) -> dict[str, KernelDisasm]:
    """Disassemble a raw cubin/PTX file saved by the CUDA hook."""
    if not Path(cubin_path).exists():
        return {}

    # ACPP JIT passes PTX text (not a binary) to cuModuleLoadData.
    # Detect by checking the first two bytes for PTX signatures.
    with open(cubin_path, "rb") as _f:
        _magic = _f.read(2)
    if _magic in (b"//", b".v") or (_magic[:1] == b"." and _magic != b"\x7f"):
        text = Path(cubin_path).read_text(errors="replace")
        return _parse_ptx_text(text, cubin_path)

    # Try cuobjdump first (works on fatbinary too), then nvdisasm
    text = _run(["cuobjdump", "--dump-sass", cubin_path], timeout=30)
    if not text.strip():
        sm = os.environ.get("HPROFILER_CUDA_SM", "sm_80")
        text = _run(["nvdisasm", "-b", sm, cubin_path], timeout=30)
    if not text.strip():
        return {}
    kernels = _parse_sass(text)
    if not kernels:
        # nvdisasm output has a different format — fall back to labelling all lines
        lines = []
        for raw in text.splitlines():
            m = _SASS_LINE.search(raw)
            if m:
                addr_s, mnem, ops = m.groups()
                lines.append(DisasmLine(
                    addr=int(addr_s, 16), mnemonic=mnem,
                    operands=ops.strip(),
                    itype=classify("sass", mnem, ops), raw=raw,
                ))
        if lines:
            name = Path(cubin_path).stem
            return {name: KernelDisasm(name=name, arch="sass",
                                        source=cubin_path, lines=lines)}
    return {
        name: KernelDisasm(name=name, arch="sass", source=cubin_path, lines=lns)
        for name, lns in kernels.items()
    }


# ── PTX (CUDA intermediate representation) ───────────────────────────────────

_PTX_ENTRY = re.compile(r'\.(?:entry|func)\s+(\w+)')
# \s+ instead of \s{2,} — ACPP-generated PTX uses single-tab indentation.
# Operands are optional (bare "ret;" has no operands).
_PTX_INSN  = re.compile(r'^\s+([a-z][a-z0-9]*(?:\.[a-z0-9]+)*)(?:\s+(.*?))?\s*;', re.I)


def _parse_ptx_text(text: str, source: str) -> dict[str, KernelDisasm]:
    """Parse PTX source text and return {short_name: KernelDisasm}.

    Handles both cuobjdump-extracted PTX and raw PTX passed to
    cuModuleLoadData by ACPP JIT compilation.  Mangled ACPP symbols are
    shortened to a human-readable name via _acpp_kernel_short.
    """
    kernels: dict[str, list[DisasmLine]] = {}
    current: Optional[str] = None
    for raw in text.splitlines():
        m = _PTX_ENTRY.search(raw)
        if m:
            current = m.group(1)
            kernels.setdefault(current, [])
            continue
        if current:
            m = _PTX_INSN.match(raw)
            if m:
                mnem = m.group(1)
                ops  = (m.group(2) or "").strip()
                itype = classify("ptx", mnem, ops)
                kernels[current].append(DisasmLine(
                    mnemonic=mnem, operands=ops, itype=itype, raw=raw,
                ))
    result: dict[str, KernelDisasm] = {}
    for mangled, lns in kernels.items():
        if not lns:
            continue
        short = _acpp_kernel_short(mangled)
        result[short] = KernelDisasm(name=short, arch="ptx", source=source, lines=lns)
    return result


def disasm_cuda_ptx(binary_path: str) -> dict[str, KernelDisasm]:
    """Extract and annotate PTX from a compiled CUDA binary via cuobjdump."""
    if not _tool("cuobjdump"):
        return {}
    text = _run(["cuobjdump", "--dump-ptx", binary_path], timeout=60)
    return _parse_ptx_text(text, binary_path)


# ── ROCm / AMDGCN ────────────────────────────────────────────────────────────

_AMDGCN_SYM  = re.compile(r'^<(\w+(?:\.kd)?)>:')
_AMDGCN_LINE = re.compile(
    r'^\s*([0-9a-f]+):\s+(?:[0-9a-f]{8}\s+)?([a-z_][a-z0-9_]+)\s*(.*)',
    re.I,
)


def disasm_rocm_binary(binary_path: str) -> dict[str, KernelDisasm]:
    """Disassemble a ROCm/HIP binary with llvm-objdump."""
    tool = _tool("llvm-objdump")
    if not tool or not Path(binary_path).exists():
        return {}
    text = _run([
        tool, "-d", "--triple=amdgcn-amd-amdhsa",
        "--no-show-raw-insn", binary_path,
    ], timeout=60)

    kernels: dict[str, list[DisasmLine]] = {}
    current: Optional[str] = None
    for raw in text.splitlines():
        m = _AMDGCN_SYM.match(raw)
        if m:
            current = m.group(1)
            kernels.setdefault(current, [])
            continue
        if current:
            m = _AMDGCN_LINE.match(raw)
            if m:
                addr_s, mnem, ops = m.groups()
                try:
                    kernels[current].append(DisasmLine(
                        addr=int(addr_s, 16), mnemonic=mnem,
                        operands=ops.strip(),
                        itype=classify("amdgcn", mnem, ops), raw=raw,
                    ))
                except ValueError:
                    pass
    return {
        name: KernelDisasm(name=name, arch="amdgcn", source=binary_path, lines=lns)
        for name, lns in kernels.items()
        if lns
    }


# ── Address → symbol lookup (for OpenMP codeptr_ra) ──────────────────────────

def _symbol_at_addr(binary_path: str, addr: int) -> Optional[tuple[str, int, int]]:
    """
    Return (mangled_name, sym_addr, sym_size) for the function containing `addr`.
    Uses the cached nm symbol table.
    Falls back to the nearest preceding symbol if size is unavailable.
    """
    syms = _nm_load(binary_path)
    best: Optional[tuple[str, int, int]] = None
    best_addr = -1
    for sym_name, (sym_addr, sym_size) in syms.items():
        if sym_size > 0 and sym_addr <= addr < sym_addr + sym_size:
            return (sym_name, sym_addr, sym_size)
        if sym_addr <= addr and sym_addr > best_addr:
            best_addr = sym_addr
            best = (sym_name, sym_addr, sym_size)
    return best


# ── Find symbols in a .jit.so that match a short kernel name ─────────────────

def _find_jit_symbol(so_path: str, short_name: str = "") -> Optional[str]:
    """
    Return the full mangled symbol in `so_path` that matches `short_name`.
    If short_name is empty, returns the first exported function symbol.
    """
    text = _run(["nm", "--defined-only", so_path])
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[1] in ("T", "t", "W", "w"):
            sym = parts[2]
            if not short_name or short_name in sym:
                return sym
    return None


def _acpp_kernel_short(mangled: str) -> str:
    """
    Extract the user-visible kernel name from an ACPP SSCP mangled symbol.
    Pattern: ...ZZ<len><name>... where ZZ marks a local lambda-in-function.

    Multiple lambdas in the same source function get the same name.
    Disambiguate using the Itanium ABI lambda ordinal ($_ N) so that
    kernel_128_0 / kernel_128_1 / kernel_128_2 are distinct.
    """
    idx = mangled.find("ZZ")
    if idx < 0:
        return mangled
    p = mangled[idx + 2:]
    length, i = 0, 0
    while i < len(p) and p[i].isdigit():
        length = length * 10 + int(p[i])
        i += 1
    if length > 0 and i + length <= len(p):
        name = p[i:i + length]
        # Append lambda ordinal to disambiguate kernels from the same function
        m = re.search(r'\$_(\d+)', mangled)
        if m:
            name = f"{name}_{m.group(1)}"
        return name
    return mangled


# ── Top-level collection ──────────────────────────────────────────────────────

def collect_disasm(
    command:      list[str],
    backends:     list[str],
    jit_spans:    list[dict],           # [{name, so_path, mangled?}] from JIT load events
    omp_syms:     dict[str, tuple] | None = None,  # {span_name: ("sym",name)|("lib",(path,off))}
    profiled_pid: int = 0,              # PID of the profiled process; used to filter /tmp files
    cpu_names:    set[str] | None = None,  # CPU function names from perf sampling
) -> dict[str, KernelDisasm]:
    """
    Collect disassembly for all profiled kernels.

    Only the specific kernel/function is disassembled for each backend:
      CUDA AoT   — cuobjdump SASS/PTX per kernel
      CUDA JIT   — PTX text captured from cuModuleLoadData
      ROCm AoT   — llvm-objdump per kernel
      ROCm JIT   — binary captured from hipModuleLoadData
      OpenCL JIT — objdump on the .jit.so symbol
      OpenMP/CPU — objdump on the specific function containing codeptr_ra

    Returns {kernel_name: KernelDisasm}.
    """
    result: dict[str, KernelDisasm] = {}
    binary = command[0] if command else ""
    if omp_syms is None:
        omp_syms = {}
    if cpu_names is None:
        cpu_names = set()

    # ── CUDA AoT: SASS + PTX from the main binary ────────────────────────────
    if "cuda" in backends and binary and Path(binary).exists():
        sass = disasm_cuda_sass(binary)
        result.update(sass)
        # Fill in PTX for kernels we didn't get SASS for
        ptx = disasm_cuda_ptx(binary)
        for name, kd in ptx.items():
            result.setdefault(name, kd)

    # ── CUDA JIT cubins saved by the hook ────────────────────────────────────
    # cuda_hook.c saves cubins as /tmp/hprofiler_cubin_<pid>_<n>.bin.
    # Filter by the profiled process's PID so we never read stale cubin files
    # left over from previous runs (which would show the wrong code).
    if "cuda" in backends:
        import glob
        pid_pat = str(profiled_pid) if profiled_pid else "*"
        for cubin_path in glob.glob(f"/tmp/hprofiler_cubin_{pid_pat}_*.bin"):
            jit_kd = disasm_cuda_cubin(cubin_path)
            for name, kd in jit_kd.items():
                result.setdefault(name, kd)
            try:
                os.unlink(cubin_path)
            except OSError:
                pass

    # ── ROCm AoT + JIT binaries saved by the hook ────────────────────────────
    if "rocm" in backends:
        if binary and Path(binary).exists():
            rocm = disasm_rocm_binary(binary)
            result.update(rocm)
        import glob as _rocm_glob
        pid_pat = str(profiled_pid) if profiled_pid else "*"
        for rocm_path in _rocm_glob.glob(f"/tmp/hprofiler_rocm_{pid_pat}_*.bin"):
            with open(rocm_path, "rb") as _rf:
                _rm = _rf.read(4)
            if _rm[:2] in (b"//", b".v", b"; ") or _rm[:1] in (b".", b";"):
                text = Path(rocm_path).read_text(errors="replace")
                jit_kd = _parse_ptx_text(text, rocm_path)
            else:
                jit_kd = disasm_rocm_binary(rocm_path)
            for name, kd in jit_kd.items():
                result.setdefault(name, kd)
            try:
                os.unlink(rocm_path)
            except OSError:
                pass

    # ── OpenCL / CPU: disassemble ACPP SSCP .jit.so files ───────────────────
    # Also check /tmp/hprofiler_jit_*.so copies saved by the hook.
    # Filter by PID so stale copies from previous runs are ignored.
    import glob as _glob
    pid_pat = str(profiled_pid) if profiled_pid else "*"
    tmp_jit = {Path(p).name: p for p in _glob.glob(f"/tmp/hprofiler_jit_{pid_pat}_*.so")}

    for entry in jit_spans:
        so_path    = entry.get("so_path", "")
        short_name = entry.get("name", "")
        mangled    = entry.get("mangled", "")

        # Prefer the /tmp copy saved by the hook (original may be gone)
        if not Path(so_path).exists():
            # Try to match by the original basename in tmp copies
            so_path = ""  # will use tmp_jit below

        if not so_path and not tmp_jit:
            continue

        # Use each temp copy once per jit_span entry
        candidate = so_path or (list(tmp_jit.values())[0] if tmp_jit else "")
        if not candidate or not Path(candidate).exists():
            continue

        # Find the full mangled symbol.  Prefer an explicit mangled name;
        # fall back to nm-search on the saved copy.
        # If short_name looks like a filename (.so suffix), don't pass it
        # as a symbol search hint — search for any function instead.
        sym_to_extract = mangled or None
        if not sym_to_extract:
            search_hint = "" if short_name.endswith(".so") else short_name
            sym_to_extract = _find_jit_symbol(candidate, search_hint)

        # Derive a human-readable kernel name from the mangled symbol.
        # This replaces filename basenames with the actual kernel name.
        if sym_to_extract:
            extracted = _acpp_kernel_short(sym_to_extract)
            if not short_name or short_name.endswith(".so"):
                short_name = extracted

        kd = disasm_elf(candidate, symbol=sym_to_extract, arch="x86-64")
        if kd:
            kd.name = short_name or kd.name
            result[kd.name] = kd
            # Remove the used temp file from disk and from the dict so the
            # next jit_span entry doesn't reuse the same .so file.
            if candidate in tmp_jit.values():
                used_key = next(k for k, v in tmp_jit.items() if v == candidate)
                del tmp_jit[used_key]
                try:
                    os.unlink(candidate)
                except OSError:
                    pass

    # ── CPU / OpenMP: targeted disasm via codeptr resolution ─────────────────
    # The OMPT hook emits either:
    #   sym=<mangled>          — dladdr succeeded (symbol in main binary)
    #   lib=<path>,offset=0x<n> — dladdr failed, but /proc/self/maps found the
    #                             library and computed the static file offset
    # For both forms we resolve symbol name+size with nm, then disassemble with
    # capstone (reads only the function bytes) or fall back to objdump.
    #
    # sym_cache: (target_path, sym_name) → KernelDisasm
    # Prevents running objdump twice when multiple span names (e.g. omp_loop and
    # omp_barrier_implicit) fall at different offsets within the same function.
    if binary and Path(binary).exists() and omp_syms:
        seen_keys: set[str] = set()
        sym_cache: dict[tuple[str, str], Optional[KernelDisasm]] = {}
        for span_name, sym_info in omp_syms.items():
            if span_name in result:
                continue
            kind, payload = sym_info

            if kind == "sym":
                sym_name: str = payload
                if sym_name in seen_keys:
                    continue
                seen_keys.add(sym_name)
                target_path = binary
            elif kind == "lib":
                lib_path, static_off = payload
                key = f"{lib_path}:{static_off}"
                if key in seen_keys or not Path(lib_path).exists():
                    continue
                seen_keys.add(key)
                sym_name = ""
                for s_name, (s_addr, s_size) in _nm_load(lib_path).items():
                    if s_size > 0 and s_addr <= static_off < s_addr + s_size:
                        sym_name = s_name
                        break
                if not sym_name:
                    continue
                target_path = lib_path
            else:
                continue

            # Skip if a different span already produced disasm for this symbol.
            cache_key = (target_path, sym_name)
            if cache_key in sym_cache:
                continue   # duplicate function — suppress redundant entry

            # Look up sym address+size, then disassemble.
            sym_addr, sym_size = _nm_load(target_path).get(sym_name, (0, 0))

            kd = (
                _disasm_elf_capstone(target_path, sym_addr, sym_size, span_name)
                if sym_addr and sym_size
                else None
            )
            if not kd:
                kd = disasm_elf(target_path, symbol=sym_name,
                                arch=_elf_arch(target_path))
                if kd:
                    kd.name = span_name
            sym_cache[cache_key] = kd   # record result (None = not found)
            if kd and kd.lines:
                result[span_name] = kd

    # ── CPU (perf sampling): disasm top functions by name in main binary ──────
    # perf-sampled spans have function names but no sym=/lib= tags, so they
    # don't go through the OMPT path above.  Try to look up each unique name
    # in the main binary's symbol table and disassemble it.
    if cpu_names and binary and Path(binary).exists():
        nm_syms = _nm_load(binary)
        elf_arch = _elf_arch(binary)
        for fn_name in cpu_names:
            if fn_name in result:
                continue
            if fn_name not in nm_syms:
                continue
            sym_addr, sym_size = nm_syms[fn_name]
            if not sym_addr or not sym_size:
                continue
            kd = _disasm_elf_capstone(binary, sym_addr, sym_size, fn_name, elf_arch)
            if not kd:
                kd = disasm_elf(binary, symbol=fn_name, arch=elf_arch)
                if kd:
                    kd.name = fn_name
            if kd and kd.lines:
                result[fn_name] = kd

    return result
