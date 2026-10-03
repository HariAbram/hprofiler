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


# First line of the streaming layout write() produces: one JSON value per
# line (metadata, then one trace event per line), so a reader can process a
# huge file line by line instead of json.load()-ing all of it. Still a plain
# JSON object -- Perfetto, chrome://tracing and json.load read it as is.
JSON_LAYOUT_LINE = '{"hprofilerJsonLayout": 1,'

_GPU_CATS = frozenset({"cuda", "rocm", "opencl"})
_GPU_BASE_TID = 2_000_000_000  # far above any realistic OS tid


def _iter_trace_events(trace: Trace, extra_args=None):
    """The Chrome trace events of a trace, streamed from its store."""
    meta = trace.metadata
    yield {
        "ph": "M", "pid": meta.pid, "tid": 0,
        "name": "process_name",
        "args": {"name": meta.command or "profiled-process"},
    }

    # GPU categories whose spans cover async GPU execution time (from CPU launch
    # to GPU completion).  If emitted on the same tid as CPU spans they will
    # overlap with hipDeviceSynchronize / hipMalloc / etc. and Perfetto drops
    # them with SLICE_DROP_OVERLAPPING_COMPLETE_EVENT.
    # Fix: assign each (category, stream) pair its own virtual tid so Perfetto
    # renders them on a dedicated GPU row that never conflicts with CPU rows.
    gpu_tid_map: dict[tuple[str, str], int] = {}

    def _gpu_tid(cat: str, stream: str) -> int:
        key = (cat, stream)
        if key not in gpu_tid_map:
            gpu_tid_map[key] = _GPU_BASE_TID + len(gpu_tid_map)
        return gpu_tid_map[key]

    for span in trace.iter_spans():
        cat = span.category.value
        args = {**span.tags, **({"_stack": span.stack_frames} if span.stack_frames else {})}
        side = span.tags.get("side")
        # Device rows: device-side spans (side=gpu, including CUDA/ROCm
        # copies/memsets in the memory category) and pre-split GPU spans.
        # Host API calls (side=cpu) stay on their real thread's row.
        if side == "gpu" or (cat in _GPU_CATS and side != "cpu"):
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
        if extra_args is not None:
            more = extra_args.get(span.eid)
            if more:
                args.update(more)
        yield {
            "ph": "X",
            "name": span.name,
            "cat": cat,
            "ts": span.start_us,
            "dur": span.duration_us,
            "pid": span.pid or meta.pid,
            "tid": tid,
            "cname": _category_color(cat),
            "args": args,
        }

    # Emit thread-name metadata for each virtual GPU track so Perfetto labels them.
    for (cat, stream), vtid in gpu_tid_map.items():
        label = f"GPU/{cat}" + (f"/stream-{stream}" if stream else "")
        yield {
            "ph": "M", "pid": meta.pid, "tid": vtid,
            "name": "thread_name", "args": {"name": label},
        }

    for inst in trace.iter_instants():
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
        yield ev

    for ctr in trace.iter_counters(order="name"):
        yield {
            "ph": "C",
            "name": ctr.name,
            "cat": ctr.category.value,
            "ts": ctr.timestamp_ns / 1_000,
            "pid": ctr.pid or meta.pid,
            "tid": 0,
            "args": {ctr.name: ctr.value},
        }


def _metadata_json(trace: Trace) -> dict:
    meta = trace.metadata
    counter_units: dict[str, str] = {}
    for ctr in trace.iter_counters(order="name"):
        if ctr.unit:
            counter_units[ctr.name] = ctr.unit
    return {
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
        # CUDA/ROCm native-tracer status, clock mapping and
        # correlation/de-duplication counts (src/core/gpu_activity.py).
        "deviceActivity": meta.device_activity,
        # Transport / receiver / capture-state integrity (src/core/receiver.py).
        "captureHealth": meta.capture_health,
    }


def write(trace: Trace, out: Path | str | IO, pretty: bool = False,
          extra_args: dict[int, dict] | None = None) -> None:
    """Write trace as a Perfetto-compatible JSON file, streamed from the
    trace's store one event per line (see JSON_LAYOUT_LINE) -- memory use
    does not grow with the number of events. `pretty` indents the metadata
    block only. `extra_args` ({span eid: {key: value}}) adds args to
    individual spans in the output without changing the trace (critical-path
    --export)."""
    if isinstance(out, (str, Path)):
        with open(out, "w") as f:
            _write_stream(trace, f, pretty, extra_args)
    else:
        _write_stream(trace, out, pretty, extra_args)


def _write_stream(trace: Trace, f: IO, pretty: bool, extra_args=None) -> None:
    from ..core.trace_io import disasm_to_json
    dumps = json.dumps
    f.write(JSON_LAYOUT_LINE + "\n")
    f.write('"metadata": ' + dumps(_metadata_json(trace), indent=2 if pretty else None) + ",\n")
    f.write('"displayTimeUnit": "ms",\n')
    f.write('"traceEvents": [')
    first = True
    for ev in _iter_trace_events(trace, extra_args):
        f.write(("\n" if first else ",\n") + dumps(ev))
        first = False
    f.write("\n]")
    disasm = trace.disasm
    if disasm:
        f.write(',\n"disasm": ' + dumps({name: disasm_to_json(kd) for name, kd in disasm.items()}))
    f.write("\n}\n")


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
    into: Trace | None = None,
) -> Trace:
    """Reconstruct a Trace from a Chrome Trace JSON file written by write()
    above (or by any earlier hprofiler version) -- the read side of this
    module's format.

    Files in the streaming layout (JSON_LAYOUT_LINE) are read line by line,
    so memory stays bounded when `into` is a disk-backed Trace (see
    core/trace_io.open_trace); older single-line files are json.load()ed.
    By default events go into a new in-memory Trace.

    `progress_cb(done, total)` / `cancel_check()` are optional and additive
    -- the GUI's async loader (src/gui/loader.py) uses them to report
    progress and to stop a load the user cancelled (LoadCancelled).
    Deliberately plain callables, not Qt Signals -- this module has no Qt
    dependency. In the streaming layout done/total are bytes read / file
    size; otherwise events processed / total events."""
    path = Path(path)
    trace = into if into is not None else Trace(TraceMetadata())
    with open(path) as f:
        first = f.readline()
        if first.strip() == JSON_LAYOUT_LINE:
            meta_raw, disasm_raw, issues = _load_streaming(f, trace, path.stat().st_size,
                                                           len(first), progress_cb, cancel_check)
        else:
            f.seek(0)
            data = json.load(f)
            meta_raw = data.get("metadata", {})
            disasm_raw = data.get("disasm")
            events = data.get("traceEvents", [])
            counter_units: dict[str, str] = meta_raw.get("counterUnits") or {}
            total_events = len(events)
            for i, ev in enumerate(events):
                if progress_cb is not None and (i % _PROGRESS_STRIDE == 0 or i == total_events - 1):
                    progress_cb(i + 1, total_events)
                if cancel_check is not None and i % _PROGRESS_STRIDE == 0 and cancel_check():
                    raise LoadCancelled(f"Cancelled while parsing event {i}/{total_events}")
                _add_event(trace, ev, counter_units)
            issues = {}
    _finish_load(trace, meta_raw, disasm_raw, collect_disasm)
    if issues:
        # Recorded with the trace (capture_warnings() renders it) and said now.
        trace.metadata.capture_health = {**trace.metadata.capture_health, "load": issues}
        import sys
        if issues.get("truncated"):
            print(f"[hprofiler][warn] {path}: the file is truncated (an interrupted export or copy) -- "
                  f"loaded the {issues.get('events', 0)} complete events before the cut; later events "
                  f"{'and the disassembly ' if issues.get('disasm_missing') else ''}are missing",
                  file=sys.stderr)
        if issues.get("bad_lines"):
            print(f"[hprofiler][warn] {path}: {issues['bad_lines']} unreadable event line(s) skipped",
                  file=sys.stderr)
    return trace


# Every 5000 events, not every single one -- calling back into Python (and,
# transitively, emitting a Qt signal across the thread boundary) has real
# per-call overhead; this keeps it negligible relative to the parse itself.
_PROGRESS_STRIDE = 5000


def _load_streaming(f, trace: Trace, total_bytes: int, done_bytes: int,
                    progress_cb, cancel_check) -> tuple[dict, dict | None, dict]:
    """Returns (metadata, disasm, issues). A file cut short (interrupted
    export, partial copy) loads every complete event before the cut and
    reports issues={"truncated": True, ...}; an unreadable line in the
    middle is skipped and counted (bad_lines), never fatal."""
    meta_raw: dict = {}
    disasm_raw = None
    counter_units: dict[str, str] = {}
    in_events = False
    events_closed = False
    ended = False
    n = 0
    bad_lines = 0
    meta_lines: list[str] = []
    for line in f:
        if not line.endswith("\n"):
            ended = line.strip() == "}"       # the writer ends with "}" (no newline) -- else cut
        done_bytes += len(line)
        if in_events:
            if line.startswith("]"):
                in_events = False
                events_closed = True
                continue
            body = line.rstrip("\n")
            if body.endswith(","):
                body = body[:-1]
            if not body:
                continue
            if n % _PROGRESS_STRIDE == 0:
                if progress_cb is not None:
                    progress_cb(done_bytes, total_bytes)
                if cancel_check is not None and cancel_check():
                    raise LoadCancelled(f"Cancelled after {n} events")
            try:
                ev = json.loads(body)
            except json.JSONDecodeError:
                bad_lines += 1            # a cut-off last line, or damage in the middle
                continue
            _add_event(trace, ev, counter_units)
            n += 1
            continue
        if line.startswith('"traceEvents": ['):
            in_events = True
            continue
        if line.startswith('"metadata": ') or meta_lines:
            meta_lines.append(line)
            text = "".join(meta_lines)
            try:
                meta_raw = json.loads(text[len('"metadata": '):].rstrip().rstrip(","))
            except json.JSONDecodeError:
                continue            # pretty-printed metadata spans lines
            meta_lines = []
            counter_units = meta_raw.get("counterUnits") or {}
            continue
        if line.startswith('"disasm": '):
            try:
                disasm_raw = json.loads(line[len('"disasm": '):].rstrip().rstrip(","))
            except json.JSONDecodeError:
                disasm_raw = None        # cut inside the disassembly section
                ended = False
                continue
        if line.strip() == "}":
            ended = True
    if progress_cb is not None:
        progress_cb(total_bytes, total_bytes)
    issues: dict = {}
    if not events_closed or not ended:
        issues = {"truncated": True, "events": n, "disasm_missing": events_closed and disasm_raw is None}
        if not meta_raw:
            issues["metadata_missing"] = True
        if bad_lines:
            bad_lines -= 1                  # the cut-off last event line is part of the truncation
    if bad_lines:
        issues["bad_lines"] = bad_lines
    return meta_raw, disasm_raw, issues


def _add_event(trace: Trace, ev: dict, counter_units: dict[str, str]) -> None:
    ph = ev.get("ph", "")
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
            return
        val = list(args.values())[0]
        trace.add(CounterEvent(
            name=name,
            category=cat,
            timestamp_ns=_us_to_ns(ev.get("ts", 0)),
            value=float(val),
            unit=counter_units.get(name, ""),
            pid=ev.get("pid", 0),
        ))


def _finish_load(trace: Trace, meta_raw: dict, disasm_raw: dict | None, collect_disasm: bool) -> None:
    from ..core.trace_io import restore_devices, restore_disasm
    trace.store.flush()
    metadata = trace.metadata
    metadata.command = meta_raw.get("command", "")
    metadata.args = meta_raw.get("args", [])
    metadata.backends_used = meta_raw.get("backends", [])
    metadata.hostname = meta_raw.get("hostname", "")
    metadata.cwd = meta_raw.get("cwd", "")
    # "" for any trace saved before this field existed -- the GUI
    # renders that as "unavailable", not a guessed/fake time.
    metadata.capture_time_iso = meta_raw.get("captureTime", "")
    metadata.pid = meta_raw.get("pid", 0) or 0
    metadata.device_activity = dict(meta_raw.get("deviceActivity") or {})
    metadata.capture_health = dict(meta_raw.get("captureHealth") or {})

    # Profiling window. Files written before startTimeNs/endTimeNs existed
    # fall back to the event extent -- never TraceMetadata's default
    # (monotonic time at LOAD), which made trace.duration_ns meaningless.
    start_raw = meta_raw.get("startTimeNs")
    end_raw = meta_raw.get("endTimeNs")
    if isinstance(start_raw, int) and isinstance(end_raw, int) and end_raw > start_raw > 0:
        metadata.start_time_ns = start_raw
        metadata.end_time_ns = end_raw
    else:
        ext = trace.store.event_extent()
        metadata.start_time_ns = ext[0] if ext else 0
        metadata.end_time_ns = ext[1] if ext else 0

    # Restore device peaks / disassembly saved at profile time
    restore_devices(trace, meta_raw.get("devices"))
    restore_disasm(trace, disasm_raw)
    trace.save()

    if collect_disasm and not trace.disasm:
        import threading as _threading
        def _bg_disasm():
            try:
                from ..core.runner import _collect_disasm
                _collect_disasm(trace, [metadata.command] + metadata.args, metadata.backends_used)
            except Exception:
                pass
        _threading.Thread(target=_bg_disasm, daemon=True).start()
