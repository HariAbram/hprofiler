"""
Source-level annotation: map span names / addresses to file:line using
addr2line (binutils) or llvm-symbolizer.

Usage:
    from .addr2line import annotate_trace
    annotate_trace(trace, binary="/path/to/executable")

After the call, CPU/OpenMP spans that resolve to a known symbol will have
two extra tags added:
    file   — source file path (possibly relative)
    line   — source line number as string

For GPU spans the cubin/ELF embedded DWARF is not yet queried — only the
host binary is searched.
"""

from __future__ import annotations
import re
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.trace import Trace

_SYMBOLIZER_CACHE: dict[tuple[str, str], tuple[str, str]] = {}


def _find_symbolizer() -> str | None:
    for tool in ("llvm-symbolizer", "addr2line"):
        p = shutil.which(tool)
        if p:
            return p
    return None


def _resolve_batch(tool: str, binary: str,
                   addresses: list[str]) -> dict[str, tuple[str, str]]:
    """
    Run addr2line / llvm-symbolizer on a batch of hex addresses.
    Returns {hex_addr: (file, line)} for resolved addresses.
    """
    if not addresses:
        return {}

    is_llvm = "llvm" in Path(tool).name
    cmd = (
        [tool, "--exe", binary, "--functions", "--demangle"]
        if is_llvm
        else [tool, "-e", binary, "-f", "-C"]
    )

    try:
        proc = subprocess.run(
            cmd, input="\n".join(addresses) + "\n",
            capture_output=True, text=True, timeout=30,
        )
    except Exception:
        return {}

    result: dict[str, tuple[str, str]] = {}
    lines = proc.stdout.splitlines()

    if is_llvm:
        # llvm-symbolizer: per-address block of "funcname\nfile:line\n\n"
        idx = 0
        for addr in addresses:
            # skip function name line
            if idx < len(lines):
                idx += 1
            if idx < len(lines):
                loc = lines[idx].strip()
                idx += 1
            else:
                continue
            # skip blank separator
            while idx < len(lines) and not lines[idx].strip():
                idx += 1
            if loc and loc != "??:0":
                parts = loc.rsplit(":", 1)
                if len(parts) == 2:
                    result[addr] = (parts[0], parts[1])
    else:
        # addr2line: alternating function / file:line pairs
        for i in range(0, len(lines) - 1, 2):
            if i // 2 >= len(addresses):
                break
            addr = addresses[i // 2]
            loc  = lines[i + 1].strip()
            if loc and loc != "??:0" and "??" not in loc:
                parts = loc.rsplit(":", 1)
                if len(parts) == 2:
                    result[addr] = (parts[0], parts[1])

    return result


def annotate_trace(trace: "Trace", binary: str | None = None) -> int:
    """
    Annotate CPU/OpenMP spans in *trace* with file:line source locations.

    Uses the binary from trace.metadata.command if *binary* is not supplied.
    Looks up addresses from span tags (``lib``/``offset`` tags set by the
    OMPT hook, or ``sym`` tags resolved via addr2line --demangle).

    Returns the number of spans annotated.
    """
    from ..core.events import Category

    tool = _find_symbolizer()
    if tool is None:
        return 0

    if binary is None:
        cmd = trace.metadata.command or ""
        if cmd:
            binary = str(Path(trace.metadata.cwd or "") / cmd)
    if not binary or not Path(binary).exists():
        return 0

    # Collect (lib_path, hex_offset) pairs that need resolution
    lookup: dict[str, list[tuple[str, object]]] = {}  # lib -> [(hex_addr, span)]

    for span in trace.spans:
        if span.category not in (Category.CPU, Category.OPENMP, Category.SYNC):
            continue
        if "file" in span.tags:
            continue   # already annotated

        lib  = span.tags.get("lib", binary)
        off_s = span.tags.get("offset", "")
        if off_s:
            try:
                hex_addr = hex(int(off_s, 16))
            except ValueError:
                continue
            lookup.setdefault(lib, []).append((hex_addr, span))

    annotated = 0
    for lib, pairs in lookup.items():
        if not Path(lib).exists():
            continue
        addrs = list({p[0] for p in pairs})

        resolved: dict[str, tuple[str, str]] = {}
        uncached: list[str] = []
        for addr in addrs:
            key = (lib, addr)
            if key in _SYMBOLIZER_CACHE:
                resolved[addr] = _SYMBOLIZER_CACHE[key]
            else:
                uncached.append(addr)
        if uncached:
            fresh = _resolve_batch(tool, lib, uncached)
            for addr, loc in fresh.items():
                _SYMBOLIZER_CACHE[(lib, addr)] = loc
                resolved[addr] = loc

        for hex_addr, span in pairs:
            if hex_addr in resolved:
                file_, line_ = resolved[hex_addr]
                span.tags["file"] = file_
                span.tags["line"] = line_
                annotated += 1

    return annotated
