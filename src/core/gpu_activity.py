"""
CUDA/ROCm host submission vs. device execution.

Every CUDA/HIP call the hooks intercept that submits device work (kernel
launch, async copy, memset, graph launch) is reported as TWO kinds of span,
linked by a correlation id:

  host API span   side=cpu  timing=host     The intercepted call itself, on
                                            the calling thread, measured with
                                            CLOCK_MONOTONIC around the call.
  device span     side=gpu  timing=...      The work on the GPU:
      device       measured: start/end from CUPTI activity records (CUDA) or
                   ROCprofiler-SDK buffered tracing (ROCm), mapped onto
                   CLOCK_MONOTONIC (see the gpuact "clock" metadata).
      proxy_event  hook's own GPU event pair (no native tracer available):
                   the DURATION is device-measured, the START is the host
                   submission time -- an estimate, wrong under queue backlog.
      proxy_host   no device timing at all -- the host call's own interval.
      proxy_flush  submission-to-flush wall time; an upper bound.

Correlation tags (all string-valued, like every wire tag):
  lid    hook-assigned, unique per process and runtime. On host spans, proxy
         device spans, and ROCm native records (pushed as ROCprofiler's
         external correlation id around the real call).
  corr   native id: CUPTI correlationId (CUDA) / ROCprofiler internal id.
  corr2  CUDA only: the other CUPTI id. A runtime call and the driver call
         it makes get different ids, and kernel records may carry either;
         memcpy records carry the driver id (corr) and the runtime id
         (corr2). Matching therefore uses the union of both.
  rt     "cuda" | "rocm" -- needed because copies/syncs are category
         memory/sync, and both runtimes' lids start at 1.
Plus dev/ctx/stream (unified program stream)/nstream (vendor stream id)/
queue (HSA queue)/op/src on device spans.

`assemble()` runs once on a freshly collected trace (Runner.run): it
correlates, removes duplicate device spans (a proxy span whose launch the
native tracer also saw; exact duplicate native records), unifies native
stream ids with the program's stream handles, sets the device span's tid and
parent link to its host submission, and derives queueing tags. The result is
persisted in the trace JSON, so it never needs to run again on reload.

Old traces (written before this split) have neither `side` nor `rt` on
CUDA/ROCm spans: their kernel span is a hybrid (host start, event duration)
and is classified as proxy_event by timing_source(). Nothing here rewrites
such traces.
"""
from __future__ import annotations

import bisect
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from .events import SpanEvent
    from .trace import Trace

RUNTIMES = frozenset({"cuda", "rocm"})

TIMING_HOST = "host"
TIMING_DEVICE = "device"
PROXY_EVENT = "proxy_event"
PROXY_HOST = "proxy_host"
PROXY_FLUSH = "proxy_flush"
PROXY_TIMINGS = frozenset({PROXY_EVENT, PROXY_HOST, PROXY_FLUSH})

# Hook builds before the host/device split tagged fallbacks this way.
_LEGACY_TIMING = {"cpu": PROXY_HOST, "cpu_flush": PROXY_FLUSH}

# Device operations that occupy the GPU (copies/memsets included), as
# opposed to "sync" (CUPTI synchronization records -- a host-side wait).
DEVICE_WORK_OPS = frozenset({"kernel", "memcpy", "memset", "graph"})

# Host spans whose submission is expected to produce device work.
_SUBMITTING_OPS = DEVICE_WORK_OPS

# Clock-mapping slack between a host span (CLOCK_MONOTONIC) and a native
# device timestamp mapped onto it. With CUPTI's timestamp callback both are
# literally the same clock; with the offset method the error is the
# calibration bracket (~1 us). 50 us is far above either and far below any
# real launch latency, so it only absorbs mapping error, never real order.
CLOCK_TOLERANCE_NS = 50_000


# ── tag helpers ─────────────────────────────────────────────────────────────

def runtime_of(span: "SpanEvent") -> str | None:
    rt = span.tags.get("rt")
    if rt in RUNTIMES:
        return rt
    cat = span.category.value
    return cat if cat in RUNTIMES else None


def is_new_format(span: "SpanEvent") -> bool:
    """Emitted by a hook that splits host and device spans (has rt=)."""
    return span.tags.get("rt") in RUNTIMES


def timing_source(span: "SpanEvent") -> str:
    """How this span's interval was obtained: host | device | proxy_*.

    Works for every backend: OpenCL's side=gpu spans come from
    CL_PROFILING (device-measured), side=cpu ones are host calls."""
    tags = span.tags
    t = tags.get("timing")
    if t:
        return _LEGACY_TIMING.get(t, t)
    side = tags.get("side")
    if side == "gpu":
        return TIMING_DEVICE
    if side == "cpu":
        return TIMING_HOST
    typ = tags.get("type")
    if span.category.value in RUNTIMES and typ in ("kernel", "graph_launch"):
        # Pre-split CUDA/ROCm kernel span: launch-call start + event duration.
        return PROXY_EVENT
    if typ == "memcpy_async":
        return PROXY_EVENT
    return TIMING_HOST


def is_device_kernel(span: "SpanEvent") -> bool:
    """A kernel's execution on the device (native, proxy, or OpenCL's
    side=gpu span) -- never a host launch/enqueue call."""
    tags = span.tags
    if tags.get("type") != "kernel" or tags.get("side") == "cpu":
        return False
    return span.category.value in ("cuda", "rocm", "opencl")


def is_host_submission(span: "SpanEvent") -> bool:
    tags = span.tags
    return tags.get("side") == "cpu" and tags.get("rt") in RUNTIMES and \
        ("lid" in tags or "corr" in tags)


def is_device_span(span: "SpanEvent") -> bool:
    """New-format CUDA/ROCm device-side span (native or proxy)."""
    tags = span.tags
    return tags.get("side") == "gpu" and tags.get("rt") in RUNTIMES


def op_of(span: "SpanEvent") -> str:
    op = span.tags.get("op")
    if op:
        return op
    typ = span.tags.get("type", "")
    if typ == "kernel":
        return "kernel"
    if typ == "graph_launch":
        return "graph"
    if typ.startswith("memcpy") or typ in ("HtoD", "DtoH"):
        return "memcpy"
    if typ == "memset":
        return "memset"
    if typ == "sync":
        return "sync"
    return typ


def _int_tag(span: "SpanEvent", key: str) -> int | None:
    v = span.tags.get(key)
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


# ── GPU-active time ────────────────────────────────────────────────────────

@dataclass
class KernelActivity:
    """Kernel intervals for GPU-active metrics, from ONE timing source.

    source: "device" (measured device intervals), "proxy" (proxy_event
    spans: event-measured durations placed at submission time), or None
    when the trace has no usable kernel intervals. Proxy and device
    intervals are never combined: a proxy interval sits at the host
    submission time, so its union with real device intervals would count
    the same kernel at two different places on the timeline."""
    intervals: list[tuple[int, int]] = field(default_factory=list)
    source: str | None = None
    used: int = 0
    excluded: int = 0          # kernel spans left out (other timing source)
    excluded_reason: str = ""


def kernel_activity(spans: Iterable["SpanEvent"]) -> KernelActivity:
    kernels = [s for s in spans if is_device_kernel(s)]
    if not kernels:
        return KernelActivity()
    native = [s for s in kernels if timing_source(s) == TIMING_DEVICE]
    if native:
        chosen, source = native, "device"
        reason = "proxy-timed kernels not combined with device-measured ones"
    else:
        chosen = [s for s in kernels if timing_source(s) == PROXY_EVENT]
        source = "proxy" if chosen else None
        reason = "host-timed fallbacks carry no device interval"
    return KernelActivity(
        intervals=[(s.start_ns, s.end_ns) for s in chosen],
        source=source,
        used=len(chosen),
        excluded=len(kernels) - len(chosen),
        excluded_reason=reason if len(kernels) > len(chosen) else "",
    )


def merged_length(intervals: list[tuple[int, int]]) -> int:
    total = 0
    cur_lo = cur_hi = None
    for lo, hi in sorted(intervals):
        if cur_hi is None or lo > cur_hi:
            if cur_hi is not None:
                total += cur_hi - cur_lo
            cur_lo, cur_hi = lo, hi
        elif hi > cur_hi:
            cur_hi = hi
    if cur_hi is not None:
        total += cur_hi - cur_lo
    return total


SOURCE_LABELS = {
    "device": "device-measured (CUPTI / ROCprofiler-SDK)",
    "proxy": "host-side proxy (GPU event durations placed at submission time)",
}


# ── correlation ────────────────────────────────────────────────────────────

@dataclass
class Correlation:
    host_of: dict[int, int] = field(default_factory=dict)       # device idx -> host idx
    confidence: dict[int, str] = field(default_factory=dict)    # device idx -> certain/high
    devices_of: dict[int, list[int]] = field(default_factory=lambda: defaultdict(list))
    unmatched: list[int] = field(default_factory=list)          # had ids, no host span
    precedes_submission: list[int] = field(default_factory=list)  # only host started after it
    ambiguous: int = 0                                          # resolved among >1 host by time


def _host_keys(span: "SpanEvent") -> list[tuple[str, str]]:
    keys = []
    lid = span.tags.get("lid")
    if lid:
        keys.append(("lid", lid))
    for k in ("corr", "corr2"):
        v = span.tags.get(k)
        if v and v != "0":
            keys.append(("corr", v))
    return keys


def _items(spans):
    return spans.items() if isinstance(spans, dict) else enumerate(spans)


def correlate(spans) -> Correlation:
    """Link every new-format device span to the host submission that
    produced it.

    lid is preferred (exact, hook-assigned). Otherwise the union of
    candidates sharing corr/corr2. A correlation id can be reused (CUPTI's
    are 32-bit and wrap; the same id appears on several host spans after a
    wrap), so among candidates the latest host span that started no later
    than the device start wins -- a device operation cannot begin before it
    was submitted, and the most recent submission with that id is the one
    still outstanding. Only when every candidate starts after the device op
    is CLOCK_TOLERANCE_NS allowed (host/device clock-mapping error), and
    beyond it the record is reported as preceding its submission. Delivery
    order is irrelevant: this runs on the complete span list."""
    by_key: dict[tuple, list[int]] = defaultdict(list)
    for i, s in _items(spans):
        if not is_host_submission(s):
            continue
        rt = s.tags["rt"]
        for kind, val in _host_keys(s):
            by_key[(s.pid, rt, kind, val)].append(i)
    starts: dict[tuple, list[int]] = {}
    for key, idxs in by_key.items():
        idxs.sort(key=lambda i: spans[i].start_ns)
        starts[key] = [spans[i].start_ns for i in idxs]

    out = Correlation()
    for i, s in _items(spans):
        if not is_device_span(s):
            continue
        rt = s.tags["rt"]
        cands: list[int] = []
        lid = s.tags.get("lid")
        if lid:
            cands = list(by_key.get((s.pid, rt, "lid", lid), ()))
        if not cands:
            seen: set[int] = set()
            for k in ("corr", "corr2"):
                v = s.tags.get(k)
                if v and v != "0":
                    for h in by_key.get((s.pid, rt, "corr", v), ()):
                        if h not in seen:
                            seen.add(h)
                            cands.append(h)
            cands.sort(key=lambda h: spans[h].start_ns)
        if not cands:
            if lid or s.tags.get("corr"):
                out.unmatched.append(i)
            continue
        pick = None
        for h in reversed(cands):              # latest submission before it
            if spans[h].start_ns <= s.start_ns:
                pick = h
                break
        if pick is None and spans[cands[0]].start_ns <= s.start_ns + CLOCK_TOLERANCE_NS:
            pick = cands[0]                    # clock-mapping error only
        if pick is None:
            out.precedes_submission.append(i)
            continue
        out.host_of[i] = pick
        if len(cands) == 1:
            out.confidence[i] = "certain"
        else:
            out.confidence[i] = "high"
            out.ambiguous += 1
        out.devices_of[pick].append(i)
    return out


def submission_ns(spans: list["SpanEvent"], corr: Correlation, i: int) -> int:
    """When device span i was submitted: its host span's start, or (no host
    span) its own start -- the latest it can possibly have been submitted."""
    h = corr.host_of.get(i)
    return spans[h].start_ns if h is not None else spans[i].start_ns


# ── assembly (runs once on a freshly collected trace) ──────────────────────

def _summary_entry(summary: dict, pid: int, rt: str) -> dict:
    return summary.setdefault(f"{pid}/{rt}", {})


def assemble(trace: "Trace") -> dict:
    """Correlate, de-duplicate and annotate CUDA/ROCm spans in place.

    Returns (and stores in trace.metadata.device_activity) a per-"pid/rt"
    summary merged with whatever status the hooks reported over the wire
    (gpuact lines: tracer status, clock mapping, dropped records)."""
    meta_da = trace.metadata.device_activity
    if meta_da.get("_assembled"):
        return meta_da
    # Only host/device-model spans take part (rt= tag): the subset is read
    # from the store, never the whole trace.
    spans = list(trace.iter_spans(gpu_model=True))
    if not spans:
        meta_da["_assembled"] = True
        return meta_da
    changed: set[int] = set()

    corr = correlate(spans)
    drop: set[int] = set()

    # 1. Exact duplicate native records (a buffer delivered twice, or two
    #    tracer instances in one process): identical id, timing and op.
    seen: dict[tuple, int] = {}
    duplicates: dict[tuple[int, str], int] = defaultdict(int)
    for i, s in enumerate(spans):
        if not is_device_span(s) or timing_source(s) != TIMING_DEVICE:
            continue
        key = (s.pid, s.tags["rt"], op_of(s), s.tags.get("corr"), s.tags.get("lid"),
               s.start_ns, s.duration_ns, s.name, s.tags.get("nstream"), s.tags.get("queue"))
        if key in seen:
            drop.add(i)
            duplicates[(s.pid, s.tags["rt"])] += 1
        else:
            seen[key] = i

    # 2. A proxy device span whose submission the native tracer also saw is
    #    the same launch observed twice: keep the measured one.
    natively_seen_hosts = {h for d, h in corr.host_of.items()
                           if d not in drop and timing_source(spans[d]) == TIMING_DEVICE}
    deduped: dict[tuple[int, str], int] = defaultdict(int)
    for d, h in corr.host_of.items():
        if timing_source(spans[d]) in PROXY_TIMINGS and h in natively_seen_hosts:
            drop.add(d)
            deduped[(spans[d].pid, spans[d].tags["rt"])] += 1

    # 3. Stream unification. Native records name streams by the vendor's
    #    own id (CUPTI streamId / HSA queue); the host spans -- and every
    #    proxy span, and the NCCL hook -- use the program's stream handle
    #    (hashed). Learn vendor id -> handle from 1:1 submissions.
    learned: dict[tuple, set[str]] = defaultdict(set)
    live_devices = {h: [d for d in ds if d not in drop] for h, ds in corr.devices_of.items()}

    def vendor_key(s: "SpanEvent") -> tuple | None:
        if s.tags.get("nstream") is not None:
            return (s.pid, s.tags["rt"], s.tags.get("ctx") or s.tags.get("dev"), "s", s.tags["nstream"])
        if s.tags.get("queue") is not None:
            return (s.pid, s.tags["rt"], s.tags.get("dev"), "q", s.tags["queue"])
        return None

    for h, ds in live_devices.items():
        hstream = spans[h].tags.get("stream")
        if hstream is None or len(ds) != 1:
            continue
        vk = vendor_key(spans[ds[0]])
        if vk is not None and op_of(spans[ds[0]]) != "graph":
            learned[vk].add(hstream)

    for i, s in enumerate(spans):
        # Every native record (incl. CUPTI sync records, timing=host);
        # proxy spans already carry the program's stream handle.
        if i in drop or not is_device_span(s) or timing_source(s) in PROXY_TIMINGS:
            continue
        h = corr.host_of.get(i)
        hstream = spans[h].tags.get("stream") if h is not None else None
        before = s.tags.get("stream")
        if hstream is not None and len(live_devices.get(h, ())) == 1:
            s.tags["stream"] = hstream
        else:
            vk = vendor_key(s)
            mapped = learned.get(vk) if vk is not None else None
            if mapped and len(mapped) == 1:
                s.tags["stream"] = next(iter(mapped))
            elif s.tags.get("nstream") is not None:
                s.tags["stream"] = "n" + s.tags["nstream"]
            elif s.tags.get("queue") is not None:
                s.tags["stream"] = "q" + s.tags["queue"]
        if s.tags.get("stream") != before:
            changed.add(i)

    # 4. Launching thread, parent link and derived queueing tags.
    for d, h in corr.host_of.items():
        if d in drop:
            continue
        dev, host = spans[d], spans[h]
        if not dev.tid:
            dev.tid = host.tid
            changed.add(d)
        if host.span_id and not dev.parent_span_id:
            dev.parent_span_id = host.span_id
            changed.add(d)
        if timing_source(dev) != TIMING_DEVICE or op_of(dev) == "sync":
            continue
        changed.add(d)
        dev.tags["api_ns"] = str(host.duration_ns)
        dev.tags["launch_ns"] = str(dev.start_ns - host.start_ns)
        dev.tags["queue_ns"] = str(max(0, dev.start_ns - host.end_ns))
        submitted = _int_tag(dev, "submitted")
        if submitted and dev.start_ns >= submitted:
            dev.tags["submit_ns"] = str(dev.start_ns - submitted)

    if changed - drop:
        trace.update_spans(spans[i] for i in sorted(changed - drop))
    if drop:
        trace.delete_spans(spans[i] for i in sorted(drop))

    # 5. Summary.
    counts: dict[tuple[int, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for i, s in enumerate(spans):
        if i in drop or not is_new_format(s):
            continue
        c = counts[(s.pid, s.tags["rt"])]
        if is_device_span(s):
            ts = timing_source(s)
            if ts == TIMING_DEVICE:
                c["device_spans"] += 1
            elif ts in PROXY_TIMINGS:
                c["proxy_spans"] += 1
            else:                               # CUPTI sync records (host-timed)
                c["sync_records"] += 1
            if i in corr.host_of:
                c["correlated"] += 1
        elif is_host_submission(s):
            c["host_calls"] += 1
            if op_of(s) in _SUBMITTING_OPS and not live_devices.get(i) and "err" not in s.tags:
                c["_no_device"] += 1
    for i in corr.unmatched:
        counts[(spans[i].pid, spans[i].tags["rt"])]["unmatched_device"] += 1
    for i in corr.precedes_submission:
        counts[(spans[i].pid, spans[i].tags["rt"])]["precedes_submission"] += 1
    for key, n in duplicates.items():
        counts[key]["duplicate_records"] += n
    for key, n in deduped.items():
        counts[key]["deduplicated_proxy"] += n

    for (pid, rt), c in counts.items():
        # A submission without a device span only means something was lost
        # when a native tracer was producing them (proxy mode never times
        # a blocking copy on the device).
        no_device = c.pop("_no_device", 0)
        if no_device and c.get("device_spans"):
            c["host_without_device"] = no_device
        entry = _summary_entry(meta_da, pid, rt)
        for k, v in c.items():
            entry[k] = v
        if c.get("device_spans"):
            entry["timing"] = "device" if not c.get("proxy_spans") else "device+proxy"
        elif c.get("proxy_spans"):
            entry["timing"] = "proxy"
    meta_da["_assembled"] = True
    return meta_da


_COUNTER_FIELDS = frozenset({"dropped", "notime", "bad_records", "buffer_alloc_failed",
                             "internal_records"})


def record_status(trace: "Trace", pid: int, api: str, fields: dict[str, str]) -> None:
    """Merge one hook `gpuact:` status line into the trace metadata.
    Counters (dropped, notime, bad_records, ...) accumulate -- the hooks
    send per-buffer deltas; everything else is last-wins."""
    rt = {"cupti": "cuda", "rocprofiler": "rocm"}.get(api, api)
    entry = _summary_entry(trace.metadata.device_activity, pid, rt)
    entry["tracer"] = api
    for k, v in fields.items():
        if k in _COUNTER_FIELDS:
            try:
                entry[k] = int(entry.get(k, 0)) + int(v)
            except ValueError:
                pass
        else:
            entry[k] = v


def parse_status_line(line: str) -> tuple[int, str, dict[str, str]] | None:
    """gpuact:<pid>:<api>:<k=v,...>"""
    parts = line.strip().split(":", 3)
    if len(parts) != 4 or parts[0] != "gpuact":
        return None
    try:
        pid = int(parts[1])
    except ValueError:
        return None
    fields: dict[str, str] = {}
    for kv in parts[3].split(","):
        if "=" in kv:
            k, v = kv.split("=", 1)
            fields[k] = v
    return pid, parts[2], fields


def describe(device_activity: dict) -> list[str]:
    """Human-readable one-liners for CLI/report output."""
    lines = []
    for key in sorted(k for k in device_activity if not k.startswith("_")):
        e = device_activity[key]
        pid, _, rt = key.partition("/")
        timing = e.get("timing")
        status = e.get("status")
        if not timing and not status:
            continue
        src = {"device": "device-measured", "proxy": "host-side proxy",
               "device+proxy": "device-measured (+ proxy spans for unmatched launches)"}.get(timing or "", "no device spans")
        bits = [f"{rt.upper()} pid {pid}: {src}"]
        if status and status != "active":
            bits.append(f"native tracer {status}" + (f" ({e['reason']})" if e.get("reason") else ""))
        if e.get("clock"):
            bits.append(f"clock={e['clock']}")
        for k, label in (("dropped", "records dropped"), ("notime", "records without timestamps"),
                         ("bad_records", "implausible records skipped"),
                         ("unmatched_device", "uncorrelated device ops"),
                         ("host_without_device", "submissions with no device record"),
                         ("deduplicated_proxy", "proxy duplicates removed")):
            if e.get(k):
                bits.append(f"{e[k]} {label}")
        lines.append("; ".join(bits))
    return lines
