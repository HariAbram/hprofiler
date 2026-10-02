"""
Opening and persisting traces independent of the storage backend.

  open_trace(path)         a `.hpstore` directory (DiskTraceStore) or a
                           Chrome/Perfetto JSON file -- small JSON loads into
                           memory; large JSON is imported once into a
                           DiskTraceStore cached next to it (or in a temp dir)
  create_disk_trace(path)  a new, empty disk-backed Trace (capture target)
  save_trace_meta(trace)   metadata, disassembly, device peaks, PC samples
                           and flags into the store (`meta` table)
"""
from __future__ import annotations

import dataclasses
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable

from .trace import Trace, TraceMetadata

# JSON files above this size are imported into a disk store instead of
# being held in memory (override: HPROFILER_JSON_MEMORY_LIMIT_MB).
DEFAULT_JSON_MEMORY_LIMIT = 256 * 1024 * 1024


# ── metadata / extras ────────────────────────────────────────────────────────

def metadata_to_json(meta: TraceMetadata) -> dict:
    return dataclasses.asdict(meta)


def metadata_from_json(d: dict) -> TraceMetadata:
    names = {f.name for f in dataclasses.fields(TraceMetadata)}
    return TraceMetadata(**{k: v for k, v in (d or {}).items() if k in names})


def disasm_to_json(kd) -> dict:
    return {
        "arch": kd.arch,
        "source": kd.source,
        "mangledName": kd.mangled_name,
        "ptxasDerived": kd.ptxas_derived,
        "lines": [
            {
                "addr": ln.addr, "mnemonic": ln.mnemonic, "operands": ln.operands,
                "itype": ln.itype.value, "comment": ln.comment, "raw": ln.raw,
                "sourceFile": ln.source_file, "sourceLine": ln.source_line,
                "samplePct": ln.sample_pct, "stallCycles": ln.stall_cycles,
                "stallReason": ln.stall_reason,
            }
            for ln in kd.lines
        ],
    }


def disasm_from_json(name: str, kd_raw: dict):
    from ..disasm.extractor import KernelDisasm, DisasmLine
    from ..disasm.classifier import InsnType
    lines = [
        DisasmLine(
            addr=ln.get("addr", 0), mnemonic=ln.get("mnemonic", ""), operands=ln.get("operands", ""),
            itype=InsnType(ln.get("itype", "other")), comment=ln.get("comment", ""),
            raw=ln.get("raw", ""), source_file=ln.get("sourceFile", ""),
            source_line=ln.get("sourceLine", 0), sample_pct=ln.get("samplePct", 0.0),
            stall_cycles=ln.get("stallCycles", -1), stall_reason=ln.get("stallReason", ""),
        )
        for ln in kd_raw.get("lines", [])
    ]
    return KernelDisasm(name=name, arch=kd_raw.get("arch", ""), source=kd_raw.get("source", ""),
                        mangled_name=kd_raw.get("mangledName", ""),
                        ptxas_derived=kd_raw.get("ptxasDerived", False), lines=lines)


def restore_disasm(trace: Trace, disasm_raw: dict | None) -> None:
    if not disasm_raw:
        return
    try:
        for name, kd_raw in disasm_raw.items():
            trace.add_disasm(disasm_from_json(name, kd_raw))
    except Exception:
        pass


def restore_devices(trace: Trace, devices_raw: list | None) -> None:
    if not devices_raw:
        return
    try:
        from ..analysis.device import DevicePeak
        trace.set_devices([DevicePeak.from_dict(x) for x in devices_raw])
    except Exception:
        pass


def save_trace_meta(trace: Trace) -> None:
    store = trace.store
    store.save_meta("metadata", metadata_to_json(trace.metadata))
    store.save_meta("flags", {"has_stacks": trace._has_stacks, "has_cpu": trace._has_cpu})
    store.save_meta("devices", [d.to_dict() for d in trace.devices])
    store.save_meta("disasm", {name: disasm_to_json(kd) for name, kd in trace.disasm.items()
                               if kd.name == name})
    store.save_meta("pc_samples", {k: [list(t) for t in v] for k, v in trace._pc_samples.items()})


def load_trace_meta(trace: Trace) -> None:
    store = trace.store
    trace.metadata = metadata_from_json(store.load_meta("metadata", {}))
    flags = store.load_meta("flags", {}) or {}
    trace._has_stacks = bool(flags.get("has_stacks"))
    trace._has_cpu = bool(flags.get("has_cpu"))
    restore_devices(trace, store.load_meta("devices"))
    restore_disasm(trace, store.load_meta("disasm"))
    for k, v in (store.load_meta("pc_samples", {}) or {}).items():
        trace._pc_samples[k] = [tuple(t) for t in v]


# ── opening / creating ───────────────────────────────────────────────────────

def is_store(path: str | os.PathLike) -> bool:
    from .store import is_store_path
    return is_store_path(path)


def open_store_trace(path: str | os.PathLike, *, finalize: bool = True) -> Trace:
    """Open a `.hpstore` directory. A store that was never finalized (an
    interrupted capture) is finalized now so viewers get indexes."""
    from .store import DiskTraceStore
    store = DiskTraceStore(path, create=False)
    trace = Trace(store=store)
    load_trace_meta(trace)
    if finalize and not store.is_finalized():
        store.finalize()
    return trace


def create_disk_trace(path: str | os.PathLike, metadata: TraceMetadata | None = None,
                      *, overwrite: bool = True, batch_size: int = 8192) -> Trace:
    from .store import DiskTraceStore
    p = Path(path)
    if p.exists() and overwrite:
        if is_store(p):
            shutil.rmtree(p)
        elif p.is_dir() and not any(p.iterdir()):
            p.rmdir()
    store = DiskTraceStore(p, create=True, batch_size=batch_size)
    trace = Trace(metadata, store=store)
    return trace


def json_memory_limit() -> int:
    mb = os.environ.get("HPROFILER_JSON_MEMORY_LIMIT_MB")
    try:
        return int(mb) * 1024 * 1024 if mb else DEFAULT_JSON_MEMORY_LIMIT
    except ValueError:
        return DEFAULT_JSON_MEMORY_LIMIT


def import_cache_path(json_path: Path) -> Path:
    return json_path.with_name(json_path.name + ".hpstore")


def _file_stamp(p: Path) -> dict:
    st = p.stat()
    return {"name": p.name, "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def write_json_export(trace: Trace, path: str | os.PathLike, **kw) -> None:
    """chrome_trace.write() plus a note in a disk store of which file it
    wrote, so open_trace() on that JSON later opens the store instead."""
    from ..output import chrome_trace
    chrome_trace.write(trace, path, **kw)
    if trace.store.kind == "disk":
        trace.store.save_meta("json_export", _file_stamp(Path(path)))
        trace.store.flush()


def _matching_store(store_path: Path, json_path: Path, key: str) -> Trace | None:
    """Open `store_path` if its meta[key] records exactly this JSON file
    (name, size, mtime) -- a stale, foreign or half-written store is never
    used in place of the JSON."""
    if not is_store(store_path):
        return None
    try:
        from .store import DiskTraceStore
        probe = DiskTraceStore(store_path, create=False)
        try:
            ok = probe.load_meta(key) == _file_stamp(json_path)
        finally:
            probe.close()
        return open_store_trace(store_path) if ok else None
    except Exception:
        return None


def open_trace(path: str | os.PathLike, *, collect_disasm: bool = False,
               progress_cb: Callable[[int, int], None] | None = None,
               cancel_check: Callable[[], bool] | None = None,
               disk: bool | None = None) -> Trace:
    """Open any trace hprofiler writes. A `.hpstore` directory opens
    directly; so does the JSON `hprofiler run` exported from one (the store
    next to it records the export). Otherwise `disk=None` decides by size:
    JSON above json_memory_limit() is imported into a DiskTraceStore
    (cached as `<file>.hpstore` next to it when writable, reused while it
    matches the JSON), so a huge trace is never held in memory."""
    from ..output import chrome_trace
    p = Path(path)
    if is_store(p):
        return open_store_trace(p)
    if p.suffix == ".json" and disk is not False:
        sibling = _matching_store(p.with_suffix(".hpstore"), p, "json_export")
        if sibling is not None:
            return sibling
    use_disk = disk if disk is not None else p.stat().st_size > json_memory_limit()
    if not use_disk:
        return chrome_trace.load_trace_from_json(p, collect_disasm=collect_disasm,
                                                 progress_cb=progress_cb, cancel_check=cancel_check)
    cache = import_cache_path(p)
    cached = _matching_store(cache, p, "imported_from")
    if cached is not None:
        return cached
    target = cache
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not os.access(target.parent, os.W_OK):
            raise PermissionError
    except (OSError, PermissionError):
        target = Path(tempfile.mkdtemp(prefix="hprofiler_import_")) / (p.name + ".hpstore")
    trace = create_disk_trace(target)
    try:
        chrome_trace.load_trace_from_json(p, collect_disasm=collect_disasm, progress_cb=progress_cb,
                                          cancel_check=cancel_check, into=trace)
        trace.finalize()
        trace.store.save_meta("imported_from", _file_stamp(p))    # written last: import complete
        trace.store.flush()
    except BaseException:
        trace.close()
        shutil.rmtree(target, ignore_errors=True)
        raise
    return trace
