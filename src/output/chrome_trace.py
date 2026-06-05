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

    for span in trace.spans:
        events.append({
            "ph": "X",
            "name": span.name,
            "cat": span.category.value,
            "ts": span.start_us,
            "dur": span.duration_us,
            "pid": span.pid or meta.pid,
            "tid": span.tid,
            "cname": _category_color(span.category.value),
            "args": span.tags,
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

    indent = 2 if pretty else None
    if isinstance(out, (str, Path)):
        with open(out, "w") as f:
            json.dump(payload, f, indent=indent)
    else:
        json.dump(payload, out, indent=indent)
