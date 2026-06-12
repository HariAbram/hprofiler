"""
Source-level annotation for disassembly: maps instruction addresses to
source file:line using addr2line (x86-64/AMDGCN), DWARF line tables
(SASS cubins), and PTX .loc directives.

After calling annotate_disasm(kd), DisasmLine.source_file and
DisasmLine.source_line are populated for any instruction whose address
could be resolved.  Instructions with no debug info are left at
("", 0) — the TUI skips them silently.
"""

from __future__ import annotations
import re
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .extractor import KernelDisasm, DisasmLine


# ── shared addr2line/llvm-symbolizer helpers ──────────────────────────────────

def _symbolizer() -> str | None:
    for t in ("llvm-symbolizer", "addr2line"):
        p = shutil.which(t)
        if p:
            return p
    return None


def _batch_resolve(tool: str, binary: str,
                   addrs: list[str]) -> dict[str, tuple[str, int]]:
    """
    Run addr2line / llvm-symbolizer on a batch of hex addresses.
    Returns {hex_addr: (file, line_number)}.
    """
    if not addrs:
        return {}
    is_llvm = "llvm" in Path(tool).name
    cmd = (
        [tool, "--exe", binary, "--functions", "--demangle"]
        if is_llvm
        else [tool, "-e", binary, "-f", "-C"]
    )
    try:
        proc = subprocess.run(
            cmd, input="\n".join(addrs) + "\n",
            capture_output=True, text=True, timeout=30,
        )
    except Exception:
        return {}

    result: dict[str, tuple[str, int]] = {}
    lines = proc.stdout.splitlines()

    if is_llvm:
        idx = 0
        for addr in addrs:
            if idx < len(lines): idx += 1          # skip function name
            loc = lines[idx].strip() if idx < len(lines) else ""; idx += 1
            while idx < len(lines) and not lines[idx].strip(): idx += 1
            if loc and "??" not in loc:
                parts = loc.rsplit(":", 1)
                if len(parts) == 2:
                    try:
                        result[addr] = (parts[0], int(parts[1]))
                    except ValueError:
                        pass
    else:
        for i in range(0, len(lines) - 1, 2):
            addr_idx = i // 2
            if addr_idx >= len(addrs):
                break
            loc = lines[i + 1].strip()
            if loc and "??" not in loc:
                parts = loc.rsplit(":", 1)
                if len(parts) == 2:
                    try:
                        result[addrs[addr_idx]] = (parts[0], int(parts[1]))
                    except ValueError:
                        pass
    return result


# ── x86-64 / AArch64 — addr2line on the ELF binary ───────────────────────────

def _annotate_x86_64(kd: "KernelDisasm") -> None:
    tool = _symbolizer()
    if not tool or not Path(kd.source).exists():
        return
    addrs = [hex(ln.addr) for ln in kd.lines if ln.addr > 0]
    if not addrs:
        return
    resolved = _batch_resolve(tool, kd.source, addrs)
    for ln in kd.lines:
        loc = resolved.get(hex(ln.addr))
        if loc:
            ln.source_file, ln.source_line = loc


# ── CUDA SASS — parse DWARF .debug_line from the cubin ───────────────────────

_DWARF_LOC = re.compile(
    r'^(.+?)\s+(\d+)\s+(0x[0-9a-f]+)',
    re.I,
)


def _read_dwarf_lines(cubin: str) -> dict[int, tuple[str, int]]:
    """
    Run `objdump --dwarf=decodedline <cubin>` and return {addr: (file, line)}.
    Falls back to llvm-dwarfdump if objdump is unavailable.
    """
    tool = shutil.which("objdump") or shutil.which("llvm-objdump")
    if not tool:
        return {}
    try:
        proc = subprocess.run(
            [tool, "--dwarf=decodedline", cubin],
            capture_output=True, text=True, timeout=30,
        )
    except Exception:
        return {}

    result: dict[int, tuple[str, int]] = {}
    current_file = ""
    for raw in proc.stdout.splitlines():
        # CU: line identifies the compilation unit's file
        cu = re.match(r'^CU:\s*(.+):', raw)
        if cu:
            current_file = cu.group(1).strip()
            continue
        m = _DWARF_LOC.match(raw.strip())
        if m:
            fname = m.group(1).strip()
            # prefer the explicit file name in the row; fall back to CU file
            if fname == "File name":     # header row — skip
                continue
            try:
                lineno = int(m.group(2))
                addr   = int(m.group(3), 16)
                result[addr] = (fname if fname else current_file, lineno)
            except ValueError:
                pass
    return result


def _annotate_sass(kd: "KernelDisasm") -> None:
    if not Path(kd.source).exists():
        return
    addr_map = _read_dwarf_lines(kd.source)
    if not addr_map:
        return
    for ln in kd.lines:
        loc = addr_map.get(ln.addr)
        if loc:
            ln.source_file, ln.source_line = loc


# ── AMDGCN — re-run llvm-objdump with --line-numbers ─────────────────────────

_LLVM_FILE = re.compile(r'^; (.+):(\d+)$')   # "; file.cpp:42"
_LLVM_ADDR = re.compile(r'^\s*([0-9a-f]+):', re.I)


def _annotate_amdgcn(kd: "KernelDisasm") -> None:
    tool = shutil.which("llvm-objdump")
    if not tool or not Path(kd.source).exists():
        return
    try:
        proc = subprocess.run(
            [tool, "-d", "--triple=amdgcn-amd-amdhsa",
             "--line-numbers", "--no-show-raw-insn", kd.source],
            capture_output=True, text=True, timeout=60,
        )
    except Exception:
        return

    addr_map: dict[int, tuple[str, int]] = {}
    cur_file, cur_line = "", 0
    for raw in proc.stdout.splitlines():
        mf = _LLVM_FILE.match(raw)
        if mf:
            cur_file = mf.group(1)
            try: cur_line = int(mf.group(2))
            except ValueError: pass
            continue
        ma = _LLVM_ADDR.match(raw)
        if ma and cur_file:
            try:
                addr_map[int(ma.group(1), 16)] = (cur_file, cur_line)
            except ValueError:
                pass

    for ln in kd.lines:
        loc = addr_map.get(ln.addr)
        if loc:
            ln.source_file, ln.source_line = loc


# ── PTX — parse embedded .file / .loc directives ─────────────────────────────

_PTX_FILE = re.compile(r'^\s*\.file\s+(\d+)\s+"([^"]+)"')
_PTX_LOC  = re.compile(r'^\s*\.loc\s+(\d+)\s+(\d+)')
_PTX_ADDR = re.compile(r'^\s*/\*\s*([0-9a-f]+)\s*\*/', re.I)  # /* offset */


def _annotate_ptx(kd: "KernelDisasm") -> None:
    """Annotate PTX disasm lines using .file / .loc directives in the PTX text."""
    if not Path(kd.source).exists():
        return
    try:
        text = Path(kd.source).read_text(errors="replace")
    except OSError:
        return

    file_map: dict[str, str] = {}   # file_idx → path
    cur_file, cur_line = "", 0
    addr_map: dict[int, tuple[str, int]] = {}

    for raw in text.splitlines():
        mf = _PTX_FILE.match(raw)
        if mf:
            file_map[mf.group(1)] = mf.group(2)
            continue
        ml = _PTX_LOC.match(raw)
        if ml:
            cur_file = file_map.get(ml.group(1), "")
            try: cur_line = int(ml.group(2))
            except ValueError: pass
            continue
        ma = _PTX_ADDR.match(raw)
        if ma and cur_file:
            try:
                addr_map[int(ma.group(1), 16)] = (cur_file, cur_line)
            except ValueError:
                pass

    for ln in kd.lines:
        loc = addr_map.get(ln.addr)
        if loc:
            ln.source_file, ln.source_line = loc


# ── public entry point ────────────────────────────────────────────────────────

def annotate_disasm(kd: "KernelDisasm") -> None:
    """
    Populate source_file / source_line on each DisasmLine in *kd*.
    No-ops silently if debug info is absent or tools are missing.
    """
    arch = kd.arch.lower()
    if arch in ("x86_64", "x86-64", "x86", "aarch64", "arm64"):
        _annotate_x86_64(kd)
    elif arch == "sass":
        _annotate_sass(kd)
    elif arch == "amdgcn":
        _annotate_amdgcn(kd)
    elif arch == "ptx":
        _annotate_ptx(kd)
