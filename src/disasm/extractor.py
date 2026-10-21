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
import copy
import re, os, shutil, subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .classifier import InsnType, classify, ITYPE_COLOR, ITYPE_LABEL


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class DisasmLine:
    addr:         int      = 0
    mnemonic:     str      = ""
    operands:     str      = ""
    itype:        InsnType = InsnType.OTHER
    comment:      str      = ""
    raw:          str      = ""   # full original text line
    # Source location (populated by source_ann.annotate_disasm if debug info present)
    source_file:  str      = ""
    source_line:  int      = 0
    # Runtime annotation fields (populated post-run)
    sample_pct:   float    = 0.0  # % of samples landing on this instruction
    stall_cycles: int      = -1   # SASS: compiler-estimated stall count (0-15); -1 = unknown
    stall_reason: str      = ""   # CUPTI dominant stall reason string

    @property
    def text(self) -> str:
        return f"{self.mnemonic}  {self.operands}".strip()


@dataclass
class KernelDisasm:
    name:          str
    arch:          str                       # 'x86-64' | 'sass' | 'ptx' | 'amdgcn'
    source:        str                       # path to the file that was disassembled
    lines:         list[DisasmLine] = field(default_factory=list)
    ptxas_derived: bool = False              # True when SASS was compiled from PTX via ptxas
    mangled_name:  str  = ""                 # original mangled symbol (for pc_sample lookup)
    # Identity of the code blob this came from: module_id() of the saved
    # JIT/module image ("" for code taken from the program's own binary or
    # a call site). Disassembly is keyed by (name, module) -- see key.
    module:        str  = ""
    # Kernel spans of this code (set by the runner, attribute_launches()),
    # and how they were counted (launch_basis): "module" -- spans carrying
    # this module id, linked to this exact code; "name" -- spans of this
    # name that no other entry's module claims (code without a module id,
    # or launches without one); "unknown" (launches None) -- spans of this
    # name ran but cannot be split among several entries; "" -- not counted
    # (older traces).
    launches:      Optional[int] = 0
    launch_basis:  str  = ""

    @property
    def key(self) -> str:
        """Trace.disasm key: "name@module" for module code, else the name."""
        return f"{self.name}@{self.module}" if self.module else self.name

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

def module_id(data: bytes) -> str:
    """Module id of a saved code blob: 64-bit FNV-1a of its bytes as 16 hex
    digits -- the hooks' hp_fnv1a64 (hooks/common/hp_module.h), which put
    the same id on the module's jit span and on every launch from it."""
    h = 0xcbf29ce484222325
    for b in data:
        h = ((h ^ b) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


GPU_ARCHS = frozenset({"sass", "ptx", "amdgcn", "gcn", "rocm", "hip"})


def attribute_launches(entries, launches) -> None:
    """Set launches and launch_basis on every entry from the trace's kernel
    spans; `launches` maps (span name, mod= or "") to a span count.

      * code with a module id: the spans carrying that id ("module"); when
        none does, no span of that name carries any module id and this is
        the only entry of that name, those spans by name ("name");
      * code without a module id (the program's own binary, a static
        runtime, older hooks): the spans of its name that carry no module
        id, or one no entry of that name has ("name");
      * spans that ran but cannot be split among several entries of one
        name: launches None ("unknown") -- never a 0 for a kernel that ran.

    An entry is matched under its name and its mangled name."""
    entries = list(entries)

    def names_of(kd) -> set:
        return {n for n in (kd.name, kd.mangled_name) if n}

    by_name: dict[str, list] = {}
    mods: dict[str, set] = {}
    for kd in entries:
        for n in names_of(kd):
            by_name.setdefault(n, []).append(kd)
            if kd.module:
                mods.setdefault(n, set()).add(kd.module)
    unclaimed: dict[str, int] = {}
    with_mod: dict[str, int] = {}
    for (n, m), c in launches.items():
        if m:
            with_mod[n] = with_mod.get(n, 0) + c
        if not m or m not in mods.get(n, ()):
            unclaimed[n] = unclaimed.get(n, 0) + c
    for kd in entries:
        ns = names_of(kd)
        if kd.module:
            c = sum(launches.get((n, kd.module), 0) for n in ns)
            no_mod = sum(launches.get((n, ""), 0) for n in ns)
            if c or not no_mod:
                kd.launches, kd.launch_basis = c, "module"
            elif all(len(by_name[n]) == 1 and not with_mod.get(n) for n in ns):
                kd.launches, kd.launch_basis = no_mod, "name"
            else:
                kd.launches, kd.launch_basis = None, "unknown"
            continue
        u = sum(unclaimed.get(n, 0) for n in ns)
        plain = {id(k) for n in ns for k in by_name[n] if not k.module}
        if u and len(plain) > 1:
            kd.launches, kd.launch_basis = None, "unknown"
        elif u or kd.arch in GPU_ARCHS:
            kd.launches, kd.launch_basis = u, "name"
        else:                                   # CPU code: no kernel spans to count
            kd.launches, kd.launch_basis = 0, ""


def _file_module_id(path: str, known: "dict[str, str] | None" = None) -> str:
    """The mod= a hook reported for `path`, else computed from the file."""
    if known and known.get(path):
        return known[path]
    try:
        with open(path, "rb") as f:
            return module_id(f.read())
    except OSError:
        return ""

def _run(cmd: list[str], timeout: int = 45) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout
    except Exception:
        return ""


_tool_cache: dict[tuple[str, ...], Optional[str]] = {}
_nm_cache: dict[str, dict[str, tuple[int, int]]] = {}


def _tool(*names: str) -> Optional[str]:
    """Return the first tool name that exists on PATH or as an absolute path (cached)."""
    if names not in _tool_cache:
        found = None
        for n in names:
            if Path(n).is_absolute():
                if Path(n).exists():
                    found = n
                    break
            elif shutil.which(n):
                found = n
                break
        _tool_cache[names] = found
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
    r'/\*([0-9a-f]+)\*/\s+(?:@!?[A-Z]\w*,\s*)?'   # /*offset*/ + optional predicate
    r'([A-Z][A-Z0-9_]+(?:\.[A-Z0-9_.]+)*)\s*'      # mnemonic
    r'(.*?)\s*;',                                    # operands up to semicolon
    re.I,
)
# Bare control-word line (second line of each 128-bit SASS instruction on Volta+):
#   /* 0x000fc40000000f00 */
_SASS_CTRL = re.compile(r'^\s*/\*\s*(0x[0-9a-f]+)\s*\*/\s*$', re.I)


def _sass_stall(ctrl_hex: str) -> int:
    """Extract stall cycle count from a SASS control word (Volta/Turing/Ampere/Ada).

    cuobjdump emits each 128-bit instruction as two lines:
      line 1: /*offset*/  MNEMONIC operands;   /* instruction_word */
      line 2:                                  /* control_word     */

    In the control word, bits [11:8] encode the warp-scheduler stall count
    (0 = issue next cycle; 15 = maximum latency/stall).  High stall on a load
    instruction indicates latency-critical memory access.
    """
    try:
        ctrl = int(ctrl_hex, 16)
        return (ctrl >> 8) & 0xF
    except (ValueError, TypeError):
        return -1


def _parse_sass(text: str) -> dict[str, list[DisasmLine]]:
    kernels: dict[str, list[DisasmLine]] = {}
    current: Optional[str] = None
    last_line: Optional[DisasmLine] = None   # for attaching the control-word stall

    for raw in text.splitlines():
        m = _SASS_KERN.match(raw)
        if m:
            current = m.group(1)
            kernels.setdefault(current, [])
            last_line = None
            continue
        if current is None:
            continue

        # Instruction line
        m = _SASS_LINE.search(raw)
        if m:
            addr_s, mnem, ops = m.group(1), m.group(2), m.group(3)
            itype = classify("sass", mnem, ops)
            last_line = DisasmLine(
                addr=int(addr_s, 16),
                mnemonic=mnem, operands=ops.strip(),
                itype=itype, raw=raw,
            )
            kernels[current].append(last_line)
            continue

        # Control-word line (Volta+ two-line format): attach stall to previous insn
        m = _SASS_CTRL.match(raw)
        if m and last_line is not None:
            last_line.stall_cycles = _sass_stall(m.group(1))

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


def cuda_fatbins(binary_path: str) -> list[bytes]:
    """The fat binaries nvcc embedded in an executable's .nv_fatbin section,
    each its header plus payload -- exactly the bytes the CUDA hook hashes
    when the runtime registers it (__cudaRegisterFatBinary), so
    module_id() of one is the mod= on launches of its kernels. [] if the
    file has no such section."""
    import struct
    try:
        with open(binary_path, "rb") as f:
            elf = f.read()
    except OSError:
        return []
    if elf[:4] != b"\x7fELF" or len(elf) < 64 or elf[4] != 2:
        return []
    shoff, = struct.unpack_from("<Q", elf, 40)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", elf, 58)
    if not shoff or shentsize < 64 or shoff + shnum * shentsize > len(elf) or shstrndx >= shnum:
        return []

    def section(i):
        name, typ = struct.unpack_from("<II", elf, shoff + i * shentsize)
        off, size = struct.unpack_from("<QQ", elf, shoff + i * shentsize + 24)
        return name, typ, off, size

    _n, _t, str_off, str_size = section(shstrndx)
    for i in range(shnum):
        name, typ, off, size = section(i)
        end = elf.find(b"\0", str_off + name, str_off + str_size)
        if typ == 8 or elf[str_off + name:end] != b".nv_fatbin" or off + size > len(elf):
            continue
        blob, out, o = elf[off:off + size], [], 0
        while o + 16 <= len(blob):
            magic, hdr, payload = struct.unpack_from("<I2xHQ", blob, o)
            if magic != 0xBA55ED50 or hdr < 16 or o + hdr + payload > len(blob):
                o += 8                                   # padding between fat binaries
                continue
            out.append(blob[o:o + hdr + payload])
            o += hdr + payload
        return out
    return []


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

_PTX_ENTRY = re.compile(r'\.(?:entry|func)\s+([^\s(]+)')
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
        # Use the full mangled name as the key so it matches span names captured
        # by the CUDA hook (cuModuleGetFunction / cuLaunchKernel).  Shortened
        # names like "curvilinear4sg_ci_0" look nicer but break span↔disasm
        # matching when nvdisasm/cuobjdump aren't available and PTX is the only
        # disasm source.
        result[mangled] = KernelDisasm(
            name=mangled, arch="ptx", source=source,
            mangled_name=mangled, lines=lns,
        )
    return result


def disasm_cuda_ptx(binary_path: str) -> dict[str, KernelDisasm]:
    """Extract and annotate PTX from a compiled CUDA binary via cuobjdump."""
    if not _tool("cuobjdump"):
        return {}
    text = _run(["cuobjdump", "--dump-ptx", binary_path], timeout=60)
    return _parse_ptx_text(text, binary_path)


# ── ROCm / AMDGCN ────────────────────────────────────────────────────────────

_AMDGCN_SYM  = re.compile(r'^[0-9a-f]+\s+<([^>]+)>:')
# ROCm llvm-objdump format: "\t<mnemonic>[ <operands>]  // <HEX_ADDR>: <RAW_BYTES>"
# The address lives in the trailing C++ comment, not at the start of the line.
_AMDGCN_LINE = re.compile(r'^\t([a-z_][a-z0-9_]+)\s*(.*)', re.I)
_AMDGCN_ADDR = re.compile(r'//\s*([0-9a-f]+)\s*:', re.I)


def _rocm_binary_type(binary_path: str) -> str:
    """Return 'hsaco', 'bundle', or '' for unrecognised / host ELFs.

    'hsaco'  — ELF64 with e_machine == EM_AMDGPU (224)
    'bundle' — Clang offload bundle (__CLANG_OFFLOAD_BUNDLE__) containing AMDGCN
    ''       — x86-64, ARM, or any other non-AMDGCN binary
    """
    try:
        with open(binary_path, "rb") as f:
            hdr = f.read(24)
        if hdr[:24] == b"__CLANG_OFFLOAD_BUNDLE__":
            return "bundle"
        if len(hdr) < 4 or hdr[:4] != b"\x7fELF":
            return ""
        if len(hdr) >= 20 and int.from_bytes(hdr[18:20], "little") == 224:
            return "hsaco"
        return ""   # x86-64 or other non-AMDGCN ELF
    except OSError:
        return ""


def disasm_rocm_binary(binary_path: str) -> dict[str, KernelDisasm]:
    """Disassemble an AMDGCN HSACO ELF or clang offload bundle with llvm-objdump.

    Rejects x86-64 ELFs (the host fat binary) so that calling this on the
    main executable never returns x86-64 host symbols.
    For clang offload bundles --triple is added so llvm-objdump extracts the
    embedded AMDGCN code object.
    """
    btype = _rocm_binary_type(binary_path)
    if not btype:
        return {}
    # Search for an llvm-objdump that has AMDGCN target support.
    # Check common ROCm, AdaptiveCpp, and system LLVM install locations first,
    # then fall through to whatever is in PATH (including versioned names).
    _llvm_objdump_candidates = [
        "/opt/rocm/bin/llvm-objdump",
        "/opt/rocm/llvm/bin/llvm-objdump",
        "/opt/AdaptiveCpp/bin/llvm-objdump",
        "/opt/adaptivecpp/bin/llvm-objdump",
        "/usr/lib/llvm/bin/llvm-objdump",
    ]
    # Also search versioned installs (llvm-17, llvm-16, etc.)
    import glob as _g
    for _pat in ("/usr/lib/llvm-*/bin/llvm-objdump",
                 "/opt/rocm-*/bin/llvm-objdump",
                 "/opt/rocm-*/llvm/bin/llvm-objdump"):
        _llvm_objdump_candidates.extend(sorted(_g.glob(_pat), reverse=True))
    _llvm_objdump_candidates += ["llvm-objdump"]
    # Add versioned names found in PATH
    for _ver in range(20, 13, -1):
        _llvm_objdump_candidates.append(f"llvm-objdump-{_ver}")
    tool = _tool(*_llvm_objdump_candidates)
    if not tool or not Path(binary_path).exists():
        return {}
    cmd = [tool, "-d", "--no-show-raw-insn"]
    if btype == "bundle":
        cmd.append("--triple=amdgcn-amd-amdhsa")
    cmd.append(binary_path)
    text = _run(cmd, timeout=60)

    kernels: dict[str, list[DisasmLine]] = {}
    kd_names: set[str] = set()   # symbols that have a .kd descriptor = real kernel entries
    current: Optional[str] = None
    for raw in text.splitlines():
        m = _AMDGCN_SYM.match(raw)
        if m:
            sym = m.group(1)
            if sym.endswith(".kd"):
                kd_names.add(sym[:-3])
                current = None   # skip the descriptor data lines
            else:
                current = sym
                kernels.setdefault(current, [])
            continue
        if current:
            m = _AMDGCN_LINE.match(raw)
            if m:
                mnem, rest = m.groups()
                # Address is in the trailing comment "// XXXXXXXX: ..."
                addr_m = _AMDGCN_ADDR.search(rest)
                addr = int(addr_m.group(1), 16) if addr_m else 0
                # Strip the comment from operands
                ops = rest.split('//')[0].strip() if '//' in rest else rest.strip()
                try:
                    kernels[current].append(DisasmLine(
                        addr=addr, mnemonic=mnem, operands=ops,
                        itype=classify("amdgcn", mnem, ops), raw=raw,
                    ))
                except ValueError:
                    pass

    # Only include true kernel entry points (those with a .kd descriptor).
    # Device functions and ACPP runtime helpers do not have .kd counterparts.
    # Fall back to all non-empty symbols if the ELF has no .kd sections (older format).
    if kd_names:
        return {
            name: KernelDisasm(name=name, arch="amdgcn", source=binary_path, lines=lns)
            for name, lns in kernels.items()
            if name in kd_names and lns
        }
    return {
        name: KernelDisasm(name=name, arch="amdgcn", source=binary_path, lines=lns)
        for name, lns in kernels.items()
        if lns
    }


def _disasm_hip_aot_binary(binary_path: str) -> dict[str, KernelDisasm]:
    """Extract and disassemble AMDGCN kernels from a native HIP AoT binary.

    hipcc embeds the GPU HSACO as a clang offload bundle inside the host ELF.
    ROCm 6+ compresses the HSACO entries (LZ4/zlib) so llvm-objdump on the raw
    bundle fails.

    Strategy:
      1. Scan the host ELF for __CLANG_OFFLOAD_BUNDLE__ magic, extract bytes.
      2. Walk bundle entries: if an AMDGCN entry is an uncompressed HSACO ELF
         (\x7fELF magic), disassemble it directly.
      3. Otherwise write the extracted bundle to a temp file and let
         clang-offload-bundler unbundle it (handles decompression transparently).
    """
    import struct, tempfile as _tf, glob as _gb

    _BUNDLE_MAGIC = b"__CLANG_OFFLOAD_BUNDLE__"
    try:
        raw = Path(binary_path).read_bytes()
    except OSError:
        return {}

    idx = raw.find(_BUNDLE_MAGIC)
    if idx == -1:
        return {}

    bundle = raw[idx:]
    if len(bundle) < 32:
        return {}

    # Parse the bundle header to find the total byte extent and entries.
    num_objs, = struct.unpack_from("<Q", bundle, 24)
    entries: list[tuple[int, int, str]] = []  # (offset, size, triple)
    max_end = 32
    pos = 32
    for _ in range(min(int(num_objs), 1024)):
        if pos + 24 > len(bundle):
            break
        offset, bsz, triple_sz = struct.unpack_from("<QQQ", bundle, pos)
        triple_b = bundle[pos + 24: pos + 24 + int(triple_sz)]
        triple = triple_b.decode("utf-8", errors="replace")
        entries.append((int(offset), int(bsz), triple))
        sec_end = int(offset) + int(bsz)
        if 0 < sec_end < 512 * 1024 * 1024 and sec_end > max_end:
            max_end = sec_end
        pos += 24 + int(triple_sz)

    if max_end <= len(_BUNDLE_MAGIC) or not entries:
        return {}

    bundle_bytes = bundle[:max_end]
    result: dict[str, KernelDisasm] = {}

    # --- Pass 1: directly extract uncompressed HSACO ELF entries.
    for offset, bsz, triple in entries:
        if "amdgcn" not in triple:
            continue
        entry = bundle_bytes[offset: offset + bsz]
        if len(entry) < 4 or entry[:4] != b"\x7fELF":
            continue  # compressed or non-ELF; handled in pass 2
        fd, tmp = _tf.mkstemp(suffix=".hsaco", prefix="hprofiler_fat_")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(entry)
            for name, kd in disasm_rocm_binary(tmp).items():
                result.setdefault(name, kd)
        except Exception:
            pass
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass
        if result:
            return result

    # --- Pass 2: clang-offload-bundler on the extracted bundle (handles LZ4/zlib).
    _bundler_candidates = [
        "/opt/rocm/llvm/bin/clang-offload-bundler",
        "/opt/rocm/bin/clang-offload-bundler",
        "/opt/AdaptiveCpp/bin/clang-offload-bundler",
        "/opt/adaptivecpp/bin/clang-offload-bundler",
    ]
    for _pat in ("/opt/rocm-*/llvm/bin/clang-offload-bundler",
                 "/opt/rocm-*/bin/clang-offload-bundler"):
        _bundler_candidates.extend(sorted(_gb.glob(_pat), reverse=True))
    _bundler_candidates.append("clang-offload-bundler")
    bundler = _tool(*_bundler_candidates)
    if not bundler:
        return result

    fd_b, bundle_path = _tf.mkstemp(suffix=".bundle", prefix="hprofiler_fat_")
    try:
        with os.fdopen(fd_b, "wb") as f:
            f.write(bundle_bytes)

        r = subprocess.run(
            [bundler, "--list", "--type=o", f"--input={bundle_path}"],
            capture_output=True, text=True, timeout=15,
        )
        if r.returncode != 0:
            return result
        targets = [t.strip() for t in r.stdout.splitlines()
                   if t.strip() and "amdgcn" in t]

        for target in targets:
            fd_o, out_path = _tf.mkstemp(suffix=".hsaco", prefix="hprofiler_fat_")
            os.close(fd_o)
            try:
                subprocess.run(
                    [bundler, "--unbundle", "--type=o",
                     f"--input={bundle_path}", f"--output={out_path}",
                     f"--targets={target}"],
                    capture_output=True, timeout=30,
                )
                if Path(out_path).exists() and Path(out_path).stat().st_size > 0:
                    for name, kd in disasm_rocm_binary(out_path).items():
                        result.setdefault(name, kd)
            except Exception:
                pass
            finally:
                try:
                    os.unlink(out_path)
                except OSError:
                    pass
            if result:
                break
    except Exception:
        pass
    finally:
        try:
            os.unlink(bundle_path)
        except OSError:
            pass

    return result


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


def _hook_kernel_name(mangled: str) -> str:
    """The span name the OpenCL hook gives an ACPP SSCP host kernel
    (opencl_hook.c extract_kernel_name): <name> from the first
    "ZZ<len><name>", else the whole symbol."""
    idx = mangled.find("ZZ")
    if idx >= 0:
        p = mangled[idx + 2:]
        i = 0
        while i < len(p) and p[i].isdigit():
            i += 1
        if i and 0 < int(p[:i]) <= len(p) - i:
            return p[i:i + int(p[:i])]
    return mangled


def dedupe_unlinked(result: dict[str, KernelDisasm]) -> dict[str, KernelDisasm]:
    """Drop a name-keyed entry whose code also exists module-keyed (an AoT
    HIP kernel disassembled from the binary and from its registered fat
    binary): one variant, not two."""
    by_name: dict[str, list[KernelDisasm]] = {}
    for kd in result.values():
        if kd.module:
            by_name.setdefault(kd.name, []).append(kd)
    for key, kd in list(result.items()):
        if kd.module or kd.name not in by_name:
            continue
        text = [ln.text for ln in kd.lines]
        if any([ln.text for ln in m.lines] == text for m in by_name[kd.name]):
            del result[key]
    return result


# ── PTX → SASS via ptxas ─────────────────────────────────────────────────────

def ptxas_compile_to_sass(ptx_path: str, sm_arch: str) -> dict[str, KernelDisasm]:
    """Compile a PTX file to a SASS cubin using ptxas, then disassemble with cuobjdump.

    sm_arch: ptxas -arch argument, e.g. "sm_86".
    Returns {short_name: KernelDisasm(arch="sass", ptxas_derived=True)}.
    Returns {} if ptxas/cuobjdump is unavailable or compilation fails.

    NOTE: The SASS produced by ptxas may differ slightly from the SASS the CUDA
    driver generates at runtime.  PC offsets from CUPTI samples (runtime SASS)
    are therefore approximate matches against this offline-compiled SASS.
    """
    if not _tool("ptxas") or not _tool("cuobjdump"):
        return {}
    if not Path(ptx_path).exists():
        return {}

    import tempfile as _tf
    fd, cubin_path = _tf.mkstemp(suffix=".cubin", prefix="hprofiler_ptxas_")
    os.close(fd)
    try:
        r = subprocess.run(
            ["ptxas", f"-arch={sm_arch}", ptx_path, "-o", cubin_path],
            capture_output=True, text=True, timeout=120,
        )
        if r.returncode != 0:
            return {}
        if not Path(cubin_path).exists() or Path(cubin_path).stat().st_size == 0:
            return {}

        sass_raw = disasm_cuda_sass(cubin_path)
        result: dict[str, KernelDisasm] = {}
        for mangled, kd in sass_raw.items():
            kd.name          = mangled
            kd.mangled_name  = mangled
            kd.ptxas_derived = True
            result[mangled]  = kd
        return result
    except Exception:
        return {}
    finally:
        try:
            Path(cubin_path).unlink()
        except OSError:
            pass


# ── Top-level collection ──────────────────────────────────────────────────────

def _cuda_aot_code(binary: str, launched_by_module: dict) -> dict[str, KernelDisasm]:
    """An executable's own CUDA code: SASS, PTX where there is no SASS.

    With a shared cudart the hook gives each of its fat binaries a module id
    (its hash, see cuda_fatbins()) and puts it on the launches of its
    kernels; when any launch carries one, every fat binary is disassembled
    on its own and its code keyed by that id. Otherwise (a static cudart,
    older traces) the binary's code is keyed by name. `source` is the
    executable either way."""
    out: dict[str, KernelDisasm] = {}
    fatbins = cuda_fatbins(binary)
    if any(module_id(b) in launched_by_module for b in fatbins):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            for i, blob in enumerate(fatbins):
                mod = module_id(blob)
                path = os.path.join(tmp, f"fatbin{i}.bin")
                with open(path, "wb") as f:
                    f.write(blob)
                code = disasm_cuda_ptx(path)
                code.update(disasm_cuda_sass(path))          # SASS where there is some
                for kd in code.values():
                    kd.module = mod if mod in launched_by_module else ""
                    kd.source = binary
                    out.setdefault(kd.key, kd)
        return out
    out.update(disasm_cuda_sass(binary))
    for name, kd in disasm_cuda_ptx(binary).items():    # PTX for kernels without SASS
        out.setdefault(name, kd)
    return out


def collect_disasm(
    command:      list[str],
    backends:     list[str],
    jit_spans:    list[dict],           # [{name, so_path, mangled?}] from JIT load events
    omp_syms:     dict[str, tuple] | None = None,  # {span_name: ("sym",name)|("lib",(path,off))}
    profiled_pid: int = 0,              # PID of the profiled process; used to filter /tmp files
    cpu_names:    set[str] | None = None,  # CPU function names from perf sampling
    sm_version:   str = "",             # e.g. "sm_86" — enables ptxas PTX→SASS for JIT kernels
    real_binary:  str = "",             # actual profiled exe (SO_PEERCRED-resolved), see below
    module_ids:   dict[str, str] | None = None,       # saved path -> mod= its hook reported
    launched_by_module: dict[str, set[str]] | None = None,  # mod -> kernel span names launched from it
    real_binaries: list[str] | None = None,  # every exe to read AoT GPU code from, in order
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

    Code from a captured JIT/module blob carries that blob's module id
    (module_id(); the hooks put the same mod= on its launches) and is keyed
    "name@module", so two programs defining the same kernel name keep
    separate entries. Code from the program's own binary or a call site is
    keyed by name.

    Returns {KernelDisasm.key: KernelDisasm}.
    """
    module_ids = module_ids or {}
    launched_by_module = launched_by_module or {}
    result: dict[str, KernelDisasm] = {}
    binary = command[0] if command else ""
    # command[0] is wrong whenever the profiled command is launcher-wrapped
    # (`hprofiler run -- srun -n 4 gmx_mpi ...` -> "srun"). OpenMP/MPI
    # disasm resolves call sites via dladdr inside the profiled process
    # (sym=/symfile= tags), so it is launcher-agnostic; CUDA/ROCm AoT disasm
    # and the CPU perf-sampled-by-name path disassemble/nm the WHOLE binary
    # with no per-span tag. real_binary (SO_PEERCRED on the hook's socket
    # connection, runner.py's _peer_real_exe) is the actual profiled exe
    # however it was launched, so it takes priority for every use of
    # `binary` below.
    # real_binaries (runner.disasm_binaries): several connected executables,
    # the ones that launched GPU work first; their ahead-of-time GPU code is
    # merged (by key: name and module), the first one's taking precedence.
    aot = [b for b in (real_binaries or []) if b and Path(b).exists()]
    if real_binary and Path(real_binary).exists() and real_binary not in aot:
        aot.insert(0, real_binary)
    if aot:
        binary = aot[0]
    elif binary and Path(binary).exists():
        aot = [binary]
    if omp_syms is None:
        omp_syms = {}
    if cpu_names is None:
        cpu_names = set()

    # ── CUDA AoT: SASS + PTX from the program's executable(s) ───────────────
    for exe in (aot if "cuda" in backends else ()):
        for kd in _cuda_aot_code(exe, launched_by_module).values():
            result.setdefault(kd.key, kd)

    # ── CUDA JIT cubins saved by the hook ────────────────────────────────────
    # cuda_hook.c saves cubins as /tmp/hprofiler_cubin_<pid>_<n>.bin and names
    # each in its module-load span (path=, the keys of module_ids): exactly
    # this trace's cubins, whatever process saved them (a wrapper's child
    # has another pid). Older hooks name none: filter by the profiled
    # process's PID so we never read stale cubin files left over from
    # previous runs (which would show the wrong code).
    if "cuda" in backends:
        import glob
        cubin_paths = sorted(p for p in module_ids if "hprofiler_cubin_" in p and Path(p).exists())
        if not cubin_paths:
            pid_pat = str(profiled_pid) if profiled_pid else "*"
            cubin_paths = glob.glob(f"/tmp/hprofiler_cubin_{pid_pat}_*.bin")
        for cubin_path in cubin_paths:
            # Detect PTX content (ACPP JIT passes raw PTX text)
            try:
                with open(cubin_path, "rb") as _cf:
                    _magic = _cf.read(2)
                _is_ptx = _magic in (b"//", b".v") or (
                    _magic[:1] == b"." and _magic != b"\x7f"
                )
            except OSError:
                _is_ptx = False

            jit_kd: dict[str, KernelDisasm] = {}
            if _is_ptx and sm_version:
                # Try ptxas → SASS so CUPTI PC offsets can be matched
                jit_kd = ptxas_compile_to_sass(cubin_path, sm_version)

            if not jit_kd:
                # Fallback: PTX parse (or non-PTX cubin via cuobjdump/nvdisasm)
                jit_kd = disasm_cuda_cubin(cubin_path)

            mod = _file_module_id(cubin_path, module_ids)
            for kd in jit_kd.values():
                kd.module = mod
                result.setdefault(kd.key, kd)
            try:
                os.unlink(cubin_path)
            except OSError:
                pass

    # ── ROCm AoT + JIT binaries saved by the hook ────────────────────────────
    if "rocm" in backends:
        for exe in aot:
            rocm = disasm_rocm_binary(exe)
            if not rocm:
                # Main binary is a native HIP AoT binary with embedded fat bundle
                rocm = _disasm_hip_aot_binary(exe)
            for key, kd in rocm.items():
                result.setdefault(key, kd)

        # Collect paths from the jit_spans path= tag (set by hipModuleLoadData hook)
        rocm_jit_paths: set[str] = set()
        for span in jit_spans:
            p = span.get("path", "")
            if p and "hprofiler_rocm_" in p:
                rocm_jit_paths.add(p)

        # Also glob /tmp for any files not yet picked up via span tags
        import glob as _rocm_glob
        pid_pat = str(profiled_pid) if profiled_pid else "*"
        for p in _rocm_glob.glob(f"/tmp/hprofiler_rocm_{pid_pat}_*.bin"):
            rocm_jit_paths.add(p)

        for rocm_path in sorted(rocm_jit_paths):
            if not Path(rocm_path).exists():
                continue
            try:
                with open(rocm_path, "rb") as _rf:
                    _rm = _rf.read(4)
            except OSError:
                continue
            if _rm[:2] in (b"//", b".v", b"; ") or _rm[:1] in (b".", b";"):
                text = Path(rocm_path).read_text(errors="replace")
                jit_kd = _parse_ptx_text(text, rocm_path)
            else:
                jit_kd = disasm_rocm_binary(rocm_path)
            mod = _file_module_id(rocm_path, module_ids)
            for kd in jit_kd.values():
                kd.module = mod
                result.setdefault(kd.key, kd)
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
    # Also pick up CPU OCL binaries extracted via clGetProgramInfo.
    tmp_jit.update({Path(p).name: p for p in _glob.glob(f"/tmp/hprofiler_ocl_{pid_pat}_*.bin")})

    for entry in jit_spans:
        so_path    = entry.get("so_path", "")
        short_name = entry.get("name", "")
        mangled    = entry.get("mangled", "")
        if not so_path and entry.get("path"):     # ROCm entries: handled above
            continue

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
        mod = (entry.get("mod") if candidate == entry.get("so_path") else "") \
            or _file_module_id(candidate, module_ids)

        # Which symbols: an explicit mangled name; else every kernel that was
        # launched from this binary (mod= on its spans), named exactly as its
        # spans are; else (older hooks, nothing launched) the symbol matching
        # the span name, or the first function, named as before.
        targets: list[tuple[str, str]] = []          # (symbol, entry name)
        if mangled:
            targets = [(mangled, "")]
        elif launched_by_module.get(mod):
            launched = launched_by_module[mod]
            for sym in _nm_load(candidate):
                hit = sym if sym in launched else _hook_kernel_name(sym)
                if hit in launched:
                    targets.append((sym, hit))
        if not targets:
            search_hint = "" if short_name.endswith((".so", ".bin")) else short_name
            sym = _find_jit_symbol(candidate, search_hint)
            if sym:
                targets = [(sym, "")]

        made = False
        try:
            with open(candidate, "rb") as _f:
                _head = _f.read(2)
        except OSError:
            _head = b""
        if _head[:2] == b"//" or _head[:1] == b".":
            # PTX text (NVIDIA OpenCL program binaries): every .entry, whose
            # name is the kernel's own name, as its launches report it --
            # like the symbols above, only the ones launched from this
            # program when its launches carry mod=
            text = Path(candidate).read_text(errors="replace").rstrip("\x00")
            launched = launched_by_module.get(mod)
            for kd in _parse_ptx_text(text, candidate).values():
                if launched and kd.name not in launched:
                    continue
                kd.module = mod
                result[kd.key] = kd
                made = True
            targets = []
        for sym_to_extract, exact_name in sorted(targets):
            # A readable name from the mangled symbol replaces a file basename.
            name = exact_name
            if not name:
                name = short_name
                if not name or name.endswith((".so", ".bin")):
                    name = _acpp_kernel_short(sym_to_extract)
            kd = disasm_elf(candidate, symbol=sym_to_extract, arch="x86-64")
            if kd:
                kd.name = name or kd.name
                kd.module = mod
                if sym_to_extract != kd.name:
                    kd.mangled_name = kd.mangled_name or sym_to_extract
                result[kd.key] = kd
                made = True
        if made:
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
    if omp_syms:
        sym_cache: dict[tuple[str, str], Optional[KernelDisasm]] = {}
        lib_syms: dict[tuple[str, int], str] = {}
        for span_name, sym_info in omp_syms.items():
            if span_name in result:
                continue
            kind, payload = sym_info

            if kind == "sym":
                # payload is (sym_name, symfile) -- symfile is the ELF file
                # dladdr() found the symbol in (hooks/common/codeptr_resolve.h),
                # which differs from `binary` (command[0]) when the profiled
                # command is a launcher (`hprofiler run -- srun -n 4 gmx_mpi`
                # -> command[0] is `srun`). Falls back to `binary` for tags
                # from a hook build without symfile= (payload[1] is None).
                sym_name, symfile = payload
                target_path = symfile if symfile and Path(symfile).exists() else binary
                if not target_path or not Path(target_path).exists():
                    continue
            elif kind == "lib":
                lib_path, static_off = payload
                if not Path(lib_path).exists():
                    continue
                sym_name = lib_syms.get((lib_path, static_off), "")
                if not sym_name:
                    for s_name, (s_addr, s_size) in _nm_load(lib_path).items():
                        if s_size > 0 and s_addr <= static_off < s_addr + s_size:
                            sym_name = s_name
                            break
                    lib_syms[(lib_path, static_off)] = sym_name
                if not sym_name:
                    continue
                target_path = lib_path
            else:
                continue

            # Several event names issued from one function (MPI_Send and
            # MPI_Recv in the same routine, a barrier and a critical section
            # in one parallel region) share its disassembly: disassemble
            # once, then file the same listing under each name.
            cache_key = (target_path, sym_name)
            if cache_key in sym_cache:
                cached = sym_cache[cache_key]
                if cached and cached.lines:
                    alias = copy.copy(cached)
                    alias.name = span_name
                    result[span_name] = alias
                continue

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
            if kd:
                # kd.name is the hprofiler event label (e.g. "omp_barrier"),
                # not an ELF symbol; sym_name is the symbol actually
                # resolved at the call site and disassembled. Keep it on
                # mangled_name (as CUPTI lookups do) so annotate_with_perf()
                # filters `perf annotate` by a symbol that exists and UIs
                # can show which function the user is looking at.
                kd.mangled_name = sym_name
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

    return dedupe_unlinked(result)


# ── Post-run annotation: perf annotate (CPU) ──────────────────────────────────

# perf annotate --stdio output line:  "  12.34  :      40a100:   mov  rax, [rbx]"
_PERF_ANNOT = re.compile(r'^\s*([\d.]+)\s*:\s+([0-9a-f]+):\s', re.I)


def annotate_with_perf(kd: KernelDisasm, perf_data: str, binary: str = "") -> int:
    """Attach CPU sample percentages to DisasmLines by running perf annotate --stdio.

    Reads perf.data produced by 'perf record -e cycles:u', maps sample counts
    to instruction addresses using perf's own ASLR/PIE remapping, and sets
    DisasmLine.sample_pct.  Returns the number of lines annotated.
    """
    if not shutil.which("perf"):
        return 0
    if not Path(perf_data).exists():
        return 0

    addr_to_line: dict[int, DisasmLine] = {
        ln.addr: ln for ln in kd.lines if ln.addr
    }
    if not addr_to_line:
        return 0

    # Filter by the REAL resolved symbol (kd.mangled_name) when we have
    # one, not kd.name: for OMP/MPI-hook-resolved kernels, kd.name is an
    # hprofiler-invented event label (e.g. "omp_barrier") that no real
    # ELF symbol is ever named, so `perf annotate -s omp_barrier` matches
    # nothing and every line silently stays at sample_pct=0 regardless of
    # whether perf actually recorded samples in that function. kd.name IS
    # the real symbol for perf-sampled-by-name kernels (the other caller
    # of this function), where mangled_name is unset -- falls back to it.
    sym_filter = kd.mangled_name or kd.name

    # Try with --no-source first (skips source interleaving), fall back without it
    base = ["perf", "annotate", "--stdio", "-s", sym_filter, "-i", perf_data]
    if binary:
        base.append(binary)
    out = _run(base + ["--no-source", "-q"], timeout=30)
    if not out.strip():
        out = _run(base, timeout=30)
    if not out.strip():
        return 0

    annotated = 0
    for raw in out.splitlines():
        m = _PERF_ANNOT.match(raw)
        if not m:
            continue
        try:
            pct  = float(m.group(1))
            addr = int(m.group(2), 16)
        except ValueError:
            continue
        if pct > 0.0 and addr in addr_to_line:
            addr_to_line[addr].sample_pct = pct
            annotated += 1

    return annotated


# ── Post-run annotation: CUPTI PC sampling (GPU) ─────────────────────────────

# Stall reason index → short string (from CUpti_ActivityPCSamplingStallReason)
_CUPTI_STALL_NAMES: dict[int, str] = {
    1:  "none",
    2:  "inst_fetch",
    3:  "exec_dep",
    4:  "mem_dep",
    5:  "texture",
    6:  "sync",
    7:  "const_mem",
    8:  "pipe_busy",
    9:  "mem_throttle",
    10: "not_sel",
    11: "other",
    12: "sleeping",
}


def annotate_with_cupti(
    kd: KernelDisasm,
    samples: "list[tuple[int, int, int]]",
) -> int:
    """Attach CUPTI PC sample percentages to DisasmLines.

    samples: list of (pc_offset, stall_reason_int, count) tuples collected by
             the CUPTI activity hook in cuda_hook.c.  pc_offset is a byte offset
             from the start of the kernel function, matching DisasmLine.addr.

    Sets DisasmLine.sample_pct (% of all samples for this kernel) and
    DisasmLine.stall_reason (dominant stall reason string).
    Returns the number of lines annotated.
    """
    if not samples:
        return 0

    # Accumulate per-pc: {pc_offset: {stall_reason: count}}
    pc_counts: dict[int, int] = {}
    pc_stall:  dict[int, dict[int, int]] = {}
    for pc, stall, count in samples:
        pc_counts[pc] = pc_counts.get(pc, 0) + count
        pc_stall.setdefault(pc, {})[stall] = pc_stall[pc].get(stall, 0) + count

    total = sum(pc_counts.values()) or 1
    addr_to_line = {ln.addr: ln for ln in kd.lines if ln.addr is not None}

    annotated = 0
    for pc, count in pc_counts.items():
        if pc in addr_to_line:
            ln = addr_to_line[pc]
            ln.sample_pct = 100.0 * count / total
            dom = max(pc_stall[pc], key=pc_stall[pc].get)
            ln.stall_reason = _CUPTI_STALL_NAMES.get(dom, "")
            annotated += 1

    return annotated
