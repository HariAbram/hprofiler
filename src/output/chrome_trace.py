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
from typing import IO, Callable

from ..core.trace import Trace, TraceMetadata
from ..core.events import SpanEvent, InstantEvent, CounterEvent, Category


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
        args = {**span.tags, **({"_stack": span.stack_frames} if span.stack_frames else {})}
        if cat in _GPU_CATS:
            stream = span.tags.get("stream", "")
            tid = _gpu_tid(cat, stream)
            # The virtual tid only exists for Perfetto's track layout; the
            # real launching/callback thread is what every hprofiler
            # analysis keys on, so it must survive a save/reload.
            args["_tid"] = span.tid
        else:
            tid = span.tid
        # Same keys as the hook wire protocol (runner._parse_record pops
        # them into these fields) -- critical-path request linking, MPI
        # wildcard resolution and call-tree parent links all depend on them.
        if span.span_id:
            args["sid"] = span.span_id
        if span.parent_span_id:
            args["psid"] = span.parent_span_id
        events.append({
            "ph": "X",
            "name": span.name,
            "cat": cat,
            "ts": span.start_us,
            "dur": span.duration_us,
            "pid": span.pid or meta.pid,
            "tid": tid,
            "cname": _category_color(cat),
            "args": args,
        })

    # Emit thread-name metadata for each virtual GPU track so Perfetto labels them.
    for (cat, stream), vtid in _gpu_tid_map.items():
        label = f"GPU/{cat}" + (f"/stream-{stream}" if stream else "")
        events.append({
            "ph": "M", "pid": meta.pid, "tid": vtid,
            "name": "thread_name", "args": {"name": label},
        })

    for inst in trace.instants:
        ev = {
            "ph": "i",
            "name": inst.name,
            "cat": inst.category.value,
            "ts": inst.timestamp_ns / 1_000,
            "pid": inst.pid or meta.pid,
            "tid": inst.tid,
            "s": "t",
        }
        if inst.tags:
            ev["args"] = dict(inst.tags)
        events.append(ev)

    counters_by_name: dict[str, list] = {}
    counter_units: dict[str, str] = {}
    for ctr in trace.counters:
        counters_by_name.setdefault(ctr.name, []).append(ctr)
        if ctr.unit:
            counter_units[ctr.name] = ctr.unit

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
            "captureTime": meta.capture_time_iso,
            # CLOCK_MONOTONIC ns, same domain as every event timestamp.
            # Without these a reloaded trace's duration_ns was "time since
            # the file was loaded" (TraceMetadata's default start).
            "startTimeNs": meta.start_time_ns,
            "endTimeNs": meta.end_time_ns,
            "pid": meta.pid,
            # A counter event's own args are its plotted series in
            # Perfetto, so units live here instead of on each event.
            "counterUnits": counter_units,
        },
    }

    if trace.disasm:
        payload["disasm"] = {
            name: {
                "arch": kd.arch,
                "source": kd.source,
                "mangledName": kd.mangled_name,
                "ptxasDerived": kd.ptxas_derived,
                "lines": [
                    {
                        "addr": ln.addr,
                        "mnemonic": ln.mnemonic,
                        "operands": ln.operands,
                        "itype": ln.itype.value,
                        "comment": ln.comment,
                        "raw": ln.raw,
                        "sourceFile": ln.source_file,
                        "sourceLine": ln.source_line,
                        "samplePct": ln.sample_pct,
                        "stallCycles": ln.stall_cycles,
                        "stallReason": ln.stall_reason,
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


def _us_to_ns(us: float) -> int:
    """Rounded, not truncated: int(us * 1000) turned ~0.7% of realistic
    CLOCK_MONOTONIC timestamps into value-1 ns (float products land just
    under the integer), and could turn a 1 ns span into a 0 ns one."""
    return int(round(us * 1_000))


class LoadCancelled(Exception):
    """Raised by load_trace_from_json() when `cancel_check` reports a
    cancellation request mid-parse. Generic (not GUI-specific) so this
    module stays usable by the TUI/CLI/analysis code exactly as before --
    a caller that never passes `cancel_check` can never see this."""


def load_trace_from_json(
    path: str | Path,
    collect_disasm: bool = False,
    *,
    progress_cb: Callable[[int, int], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> Trace:
    """Reconstruct a Trace from a Chrome Trace JSON file written by write()
    above -- the read side of this module's format, moved here from
    src/ui/app.py (which re-exports this name for backward compatibility)
    so loading a trace doesn't require importing the whole Textual-based
    TUI module; every UI (TUI, GUI, `hprofiler compare`/`export`/etc.)
    needs this regardless of which viewer it ends up using, if any.

    `progress_cb`/`cancel_check` are optional and additive -- every
    existing caller passes neither and sees byte-identical behavior.
    Both exist for the GUI's async-loading worker (src/gui/loader.py),
    which runs this on a background thread and needs SOME way to report
    "N of M events processed" and check "has the user asked to cancel"
    during the one genuinely large loop below (the event-parsing loop --
    the dominant cost for a large trace, confirmed by direct profiling:
    a single blocking json.load() followed by one Python object per
    event). Deliberately plain callables, not Qt Signals -- this module
    has no Qt dependency and must not gain one just for this."""
    with open(path) as f:
        data = json.load(f)

    meta_raw = data.get("metadata", {})
    metadata = TraceMetadata(
        command=meta_raw.get("command", ""),
        args=meta_raw.get("args", []),
        backends_used=meta_raw.get("backends", []),
        hostname=meta_raw.get("hostname", ""),
        cwd=meta_raw.get("cwd", ""),
        # "" for any trace saved before this field existed -- the GUI
        # renders that as "unavailable", not a guessed/fake time.
        capture_time_iso=meta_raw.get("captureTime", ""),
        pid=meta_raw.get("pid", 0) or 0,
    )
    counter_units: dict[str, str] = meta_raw.get("counterUnits") or {}
    trace = Trace(metadata)

    events = data.get("traceEvents", [])
    total_events = len(events)
    # Every 5000 events, not every single one -- calling back into
    # Python (and, transitively, emitting a Qt signal across the thread
    # boundary) has real per-call overhead; this keeps that overhead
    # negligible relative to the parse itself while still giving
    # "measurable progress" many times over the course of a large trace.
    _PROGRESS_STRIDE = 5000

    for i, ev in enumerate(events):
        if progress_cb is not None and (i % _PROGRESS_STRIDE == 0 or i == total_events - 1):
            progress_cb(i + 1, total_events)
        if cancel_check is not None and i % _PROGRESS_STRIDE == 0 and cancel_check():
            raise LoadCancelled(f"Cancelled while parsing event {i}/{total_events}")

        ph      = ev.get("ph", "")
        cat_str = ev.get("cat", "other")
        try:
            cat = Category(cat_str)
        except ValueError:
            cat = Category.OTHER

        if ph == "X":
            args = ev.get("args", {})
            stack = args.pop("_stack", [])
            span_id = args.pop("sid", "")
            parent_span_id = args.pop("psid", "")
            tid = args.pop("_tid", ev.get("tid", 0))
            start_ns = _us_to_ns(ev.get("ts", 0))
            trace.add(SpanEvent(
                name=ev.get("name", ""),
                category=cat,
                start_ns=start_ns,
                # Derived from the rounded END, not rounded separately, so
                # start+dur always lands on the originally-written end.
                duration_ns=max(_us_to_ns(ev.get("ts", 0) + ev.get("dur", 0)) - start_ns, 0),
                pid=ev.get("pid", 0),
                tid=tid,
                tags=args,
                stack_frames=stack if isinstance(stack, list) else [],
                span_id=str(span_id),
                parent_span_id=str(parent_span_id),
            ))
        elif ph == "i":
            trace.add(InstantEvent(
                name=ev.get("name", ""),
                category=cat,
                timestamp_ns=_us_to_ns(ev.get("ts", 0)),
                pid=ev.get("pid", 0),
                tid=ev.get("tid", 0),
                tags=dict(ev.get("args") or {}),
            ))
        elif ph == "C":
            args = ev.get("args", {})
            name = ev.get("name", "counter")
            if not args:
                # No sampled value at all -- skip rather than invent a 0.
                continue
            val = list(args.values())[0]
            trace.add(CounterEvent(
                name=name,
                category=cat,
                timestamp_ns=_us_to_ns(ev.get("ts", 0)),
                value=float(val),
                unit=counter_units.get(name, ""),
                pid=ev.get("pid", 0),
            ))

    # Profiling window. Files written before startTimeNs/endTimeNs existed
    # fall back to the event extent -- never TraceMetadata's default
    # (monotonic time at LOAD), which made trace.duration_ns meaningless.
    start_raw = meta_raw.get("startTimeNs")
    end_raw = meta_raw.get("endTimeNs")
    if isinstance(start_raw, int) and isinstance(end_raw, int) and end_raw > start_raw > 0:
        metadata.start_time_ns = start_raw
        metadata.end_time_ns = end_raw
    else:
        starts = [s.start_ns for s in trace.spans] + [i.timestamp_ns for i in trace.instants] \
            + [c.timestamp_ns for c in trace.counters]
        ends = [s.end_ns for s in trace.spans] + [i.timestamp_ns for i in trace.instants] \
            + [c.timestamp_ns for c in trace.counters]
        metadata.start_time_ns = min(starts) if starts else 0
        metadata.end_time_ns = max(ends) if ends else 0

    # Restore device peaks saved at profile time
    for d in meta_raw.get("devices", []):
        try:
            from ..analysis.device import DevicePeak
            trace.set_devices([DevicePeak.from_dict(x) for x in meta_raw["devices"]])
            break
        except Exception:
            pass

    # Restore serialized disasm if present in the JSON.
    disasm_raw = data.get("disasm")
    if disasm_raw:
        try:
            from ..disasm.extractor import KernelDisasm, DisasmLine
            from ..disasm.classifier import InsnType
            for name, kd_raw in disasm_raw.items():
                lines = [
                    DisasmLine(
                        addr=ln.get("addr", 0),
                        mnemonic=ln.get("mnemonic", ""),
                        operands=ln.get("operands", ""),
                        itype=InsnType(ln.get("itype", "other")),
                        comment=ln.get("comment", ""),
                        raw=ln.get("raw", ""),
                        source_file=ln.get("sourceFile", ""),
                        source_line=ln.get("sourceLine", 0),
                        sample_pct=ln.get("samplePct", 0.0),
                        stall_cycles=ln.get("stallCycles", -1),
                        stall_reason=ln.get("stallReason", ""),
                    )
                    for ln in kd_raw.get("lines", [])
                ]
                trace.add_disasm(KernelDisasm(
                    name=name,
                    arch=kd_raw.get("arch", ""),
                    source=kd_raw.get("source", ""),
                    mangled_name=kd_raw.get("mangledName", ""),
                    ptxas_derived=kd_raw.get("ptxasDerived", False),
                    lines=lines,
                ))
        except Exception:
            pass

    if collect_disasm and not trace.disasm:
        import threading as _threading
        def _bg_disasm():
            try:
                from ..core.runner import _collect_disasm
                _collect_disasm(trace, [metadata.command] + metadata.args, metadata.backends_used)
            except Exception:
                pass
        _threading.Thread(target=_bg_disasm, daemon=True).start()

    return trace
