"""
Chrome Trace Format (CTF) / Perfetto JSON output.

Compatible with:
  - chrome://tracing  (paste JSON)
  - Perfetto UI       (ui.perfetto.dev)
  - speedscope.app    (partial)

Spec: https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import IO

from ..core.trace import Trace
from ..core.events import SpanEvent, InstantEvent, CounterEvent


def _category_color(cat: str) -> str:
    colors = {
        "cpu": "good",
        "cuda": "terrible",
        "rocm": "bad",
        "opencl": "yellow",
        "openmp": "olive",
        "memory": "grey",
        "sync": "white",
        "jit": "purple",
        "nvtx": "thread_state_usermode",
    }
    return colors.get(cat, "generic_work")


def write(trace: Trace, out: Path | str | IO, pretty: bool = False) -> None:
    """Write trace as a Perfetto-compatible JSON file."""
    events = []
    meta = trace.metadata

    # Metadata process/thread names
    events.append({
        "ph": "M", "pid": meta.pid, "tid": 0,
        "name": "process_name",
        "args": {"name": meta.command or "profiled-process"},
    })

    # GPU categories whose spans cover async GPU execution time (from CPU launch
    # to GPU completion).  If emitted on the same tid as CPU spans they will
    # overlap with hipDeviceSynchronize / hipMalloc / etc. and Perfetto drops
    # them with SLICE_DROP_OVERLAPPING_COMPLETE_EVENT.
    # Fix: assign each (category, stream) pair its own virtual tid so Perfetto
    # renders them on a dedicated GPU row that never conflicts with CPU rows.
    _GPU_CATS = frozenset({"cuda", "rocm", "opencl"})
    _GPU_BASE_TID = 2_000_000_000  # far above any realistic OS tid
    _gpu_tid_map: dict[tuple[str, str], int] = {}

    def _gpu_tid(cat: str, stream: str) -> int:
        key = (cat, stream)
        if key not in _gpu_tid_map:
            _gpu_tid_map[key] = _GPU_BASE_TID + len(_gpu_tid_map)
        return _gpu_tid_map[key]

    for span in trace.spans:
        cat = span.category.value
        if cat in _GPU_CATS:
            stream = span.tags.get("stream", "")
            tid = _gpu_tid(cat, stream)
        else:
            tid = span.tid
        events.append({
            "ph": "X",
            "name": span.name,
            "cat": cat,
            "ts": span.start_us,
            "dur": span.duration_us,
            "pid": span.pid or meta.pid,
            "tid": tid,
            "cname": _category_color(cat),
            "args": {**span.tags, **({"_stack": span.stack_frames} if span.stack_frames else {})},
        })

    # Emit thread-name metadata for each virtual GPU track so Perfetto labels them.
    for (cat, stream), vtid in _gpu_tid_map.items():
        label = f"GPU/{cat}" + (f"/stream-{stream}" if stream else "")
        events.append({
            "ph": "M", "pid": meta.pid, "tid": vtid,
            "name": "thread_name", "args": {"name": label},
        })

    for inst in trace.instants:
        events.append({
            "ph": "i",
            "name": inst.name,
            "cat": inst.category.value,
            "ts": inst.timestamp_ns / 1_000,
            "pid": inst.pid or meta.pid,
            "tid": inst.tid,
            "s": "t",
        })

    counters_by_name: dict[str, list] = {}
    for ctr in trace.counters:
        counters_by_name.setdefault(ctr.name, []).append(ctr)

    for name, ctrs in counters_by_name.items():
        for ctr in sorted(ctrs, key=lambda c: c.timestamp_ns):
            events.append({
                "ph": "C",
                "name": name,
                "cat": ctr.category.value,
                "ts": ctr.timestamp_ns / 1_000,
                "pid": ctr.pid or meta.pid,
                "tid": 0,
                "args": {name: ctr.value},
            })

    payload = {
        "traceEvents": events,
        "displayTimeUnit": "ms",
        "metadata": {
            "command": meta.command,
            "args": meta.args,
            "backends": meta.backends_used,
            "hostname": meta.hostname,
            "cwd": meta.cwd,
            "duration_ms": trace.duration_ns / 1_000_000,
            "devices": [d.to_dict() for d in trace.devices],
        },
    }

    if trace.disasm:
        payload["disasm"] = {
            name: {
                "arch": kd.arch,
                "source": kd.source,
                "lines": [
                    {
                        "addr": ln.addr,
                        "mnemonic": ln.mnemonic,
                        "operands": ln.operands,
                        "itype": ln.itype.value,
                        "comment": ln.comment,
                        "raw": ln.raw,
                    }
                    for ln in kd.lines
                ],
            }
            for name, kd in trace.disasm.items()
        }

    indent = 2 if pretty else None
    if isinstance(out, (str, Path)):
        with open(out, "w") as f:
            json.dump(payload, f, indent=indent)
    else:
        json.dump(payload, out, indent=indent)
