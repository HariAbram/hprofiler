"""
OpenTelemetry Protocol (OTLP) JSON exporter for hprofiler traces.

Converts a completed Trace to OTLP format and either:
  • Writes it to a .otlp.json file (replay with `curl` or `otelcol`)
  • POSTs it directly to any OTLP-compatible HTTP collector

No external dependencies — uses only the Python standard library.

Compatible collectors (default OTLP/HTTP port 4318):
  Grafana Alloy, otelcol, Jaeger ≥ 1.35, Grafana Tempo,
  DataDog Agent, Honeycomb, New Relic, and any other OTLP receiver.

Mapping:
  SpanEvent     → OTLP Span (all spans are root-level; no parent inference)
  InstantEvent  → zero-duration OTLP Span
  CounterEvent  → OTLP Gauge metric (sent to /v1/metrics when --otlp-endpoint)
  TraceMetadata → OTLP Resource attributes
  Category      → InstrumentationScope name  (e.g. "hprofiler.cuda")

Time alignment:
  hprofiler timestamps are monotonic-ns relative to trace start.
  At export time the difference (time.time_ns() - time.monotonic_ns()) converts
  them to Unix epoch nanoseconds; the error is ≤ clock drift since the run, i.e.
  sub-millisecond for immediately post-run exports.
"""

from __future__ import annotations

import hashlib
import json
import struct
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.trace import Trace

_VERSION = "0.1.0"
_SCOPE   = "hprofiler"


# ── OTLP attribute helpers ────────────────────────────────────────────────────

def _attr_str(key: str, val: str) -> dict:
    return {"key": key, "value": {"stringValue": val}}

def _attr_int(key: str, val: int) -> dict:
    return {"key": key, "value": {"intValue": str(val)}}

def _attr_dbl(key: str, val: float) -> dict:
    return {"key": key, "value": {"doubleValue": val}}

def _attr_bool(key: str, val: bool) -> dict:
    return {"key": key, "value": {"boolValue": val}}


def _coerce_tag(key: str, raw: object) -> dict:
    """Convert a hprofiler span tag to the most appropriate OTLP attribute."""
    if isinstance(raw, bool):
        return _attr_bool(key, raw)
    if isinstance(raw, int):
        return _attr_int(key, raw)
    if isinstance(raw, float):
        return _attr_dbl(key, raw)
    s = str(raw)
    # Try integer, then float, then fall back to string.
    try:
        return _attr_int(key, int(s))
    except ValueError:
        pass
    try:
        return _attr_dbl(key, float(s))
    except ValueError:
        pass
    return _attr_str(key, s)


# ── ID generation ─────────────────────────────────────────────────────────────

def _trace_id(meta_key: str) -> str:
    """16-byte (32 hex char) trace ID derived from run metadata."""
    return hashlib.sha256(meta_key.encode()).digest()[:16].hex()


def _span_id(index: int) -> str:
    """8-byte (16 hex char) span ID from sequential index."""
    return struct.pack(">Q", index & 0xFFFF_FFFF_FFFF_FFFF).hex()


def _span_id_from_str(sid: str, fallback_idx: int) -> str:
    """Convert a hook-emitted uint64 decimal string to a 16-char hex span ID."""
    if sid:
        try:
            return struct.pack(">Q", int(sid) & 0xFFFF_FFFF_FFFF_FFFF).hex()
        except (ValueError, struct.error):
            pass
    return _span_id(fallback_idx)


# ── Resource builder ──────────────────────────────────────────────────────────

def _resource(meta: "TraceMetadata") -> dict:  # type: ignore[name-defined]
    attrs = [
        _attr_str("service.name",    _SCOPE),
        _attr_str("service.version", _VERSION),
    ]
    if meta.hostname:
        attrs.append(_attr_str("host.name", meta.hostname))
    if meta.pid:
        attrs.append(_attr_int("process.pid", meta.pid))
    cmd = " ".join(filter(None, [meta.command] + (meta.args or [])))
    if cmd:
        attrs.append(_attr_str("process.command_line", cmd))
    if meta.cwd:
        attrs.append(_attr_str("process.cwd", meta.cwd))
    if meta.backends_used:
        attrs.append(_attr_str("hprofiler.backends", ",".join(meta.backends_used)))
    return {"attributes": attrs}


# ── Traces payload ────────────────────────────────────────────────────────────

def build_traces_payload(trace: "Trace") -> dict:
    """Return the OTLP JSON ``resourceSpans`` payload for *trace*."""
    from ..core.events import SpanEvent, InstantEvent

    meta  = trace.metadata
    # Epoch offset: converts monotonic ns → Unix ns.  Computed once per export.
    epoch_offset     = time.time_ns() - time.monotonic_ns()
    trace_epoch_ns   = meta.start_time_ns + epoch_offset

    run_key  = f"{meta.command}:{meta.start_time_ns}:{meta.hostname}"
    tid      = _trace_id(run_key)

    # Collect spans grouped by category (→ InstrumentationScope)
    by_cat: dict[str, list[dict]] = {}
    idx = 0

    # Two-pass: first assign span IDs so parentSpanId can reference them.
    # span_id_map: hook span_id string → OTLP 16-hex spanId
    span_id_map: dict[str, str] = {}
    all_evs: list[tuple] = []  # (ev, cat, start_ns, end_ns, name, attrs, otlp_sid)
    for ev in trace.all_events:
        if isinstance(ev, SpanEvent):
            cat      = ev.category.value
            start_ns = trace_epoch_ns + ev.start_ns
            end_ns   = start_ns + max(ev.duration_ns, 0)
            name     = ev.name or f"[{cat}]"
            attrs    = [_attr_str("hprofiler.category", cat)]
            if ev.pid:
                attrs.append(_attr_int("process.pid",  ev.pid))
            if ev.tid:
                attrs.append(_attr_int("thread.id",    ev.tid))
            for k, v in ev.tags.items():
                attrs.append(_coerce_tag(f"hprofiler.tag.{k}", v))
            otlp_sid = _span_id_from_str(ev.span_id, idx)
            if ev.span_id:
                span_id_map[ev.span_id] = otlp_sid

        elif isinstance(ev, InstantEvent):
            cat      = ev.category.value
            start_ns = trace_epoch_ns + ev.timestamp_ns
            end_ns   = start_ns
            name     = ev.name or f"[{cat}:instant]"
            attrs    = [_attr_str("hprofiler.category", cat)]
            if ev.pid:
                attrs.append(_attr_int("process.pid", ev.pid))
            if ev.tid:
                attrs.append(_attr_int("thread.id",   ev.tid))
            otlp_sid = _span_id(idx)

        else:
            continue  # CounterEvent → metrics payload, not spans

        all_evs.append((ev, cat, start_ns, end_ns, name, attrs, otlp_sid))
        idx += 1

    for ev, cat, start_ns, end_ns, name, attrs, otlp_sid in all_evs:
        span: dict = {
            "traceId":           tid,
            "spanId":            otlp_sid,
            "name":              name,
            "kind":              1,          # SPAN_KIND_INTERNAL
            "startTimeUnixNano": str(start_ns),
            "endTimeUnixNano":   str(end_ns),
            "attributes":        attrs,
            "status":            {"code": 1},  # STATUS_CODE_OK
        }
        # Attach parentSpanId when the hook emitted a psid= tag
        if isinstance(ev, SpanEvent) and ev.parent_span_id:
            parent_otlp_sid = span_id_map.get(ev.parent_span_id)
            if parent_otlp_sid:
                span["parentSpanId"] = parent_otlp_sid
        by_cat.setdefault(cat, []).append(span)

    scope_spans = [
        {
            "scope":  {"name": f"{_SCOPE}.{cat}", "version": _VERSION},
            "spans":  spans,
        }
        for cat, spans in by_cat.items()
    ]

    return {
        "resourceSpans": [
            {
                "resource":   _resource(meta),
                "scopeSpans": scope_spans,
            }
        ]
    }


# ── Metrics payload ───────────────────────────────────────────────────────────

def build_metrics_payload(trace: "Trace") -> dict:
    """Return the OTLP JSON ``resourceMetrics`` payload for *trace*."""
    if not trace.counters:
        return {"resourceMetrics": []}

    meta           = trace.metadata
    epoch_offset   = time.time_ns() - time.monotonic_ns()
    trace_epoch_ns = meta.start_time_ns + epoch_offset

    # Group data points by counter name
    by_name: dict[str, list[dict]] = {}
    for ctr in trace.counters:
        ts = trace_epoch_ns + ctr.timestamp_ns
        dp: dict = {
            "timeUnixNano": str(ts),
            "asDouble":     float(ctr.value),
        }
        if ctr.pid:
            dp["attributes"] = [_attr_int("process.pid", ctr.pid)]
        by_name.setdefault(ctr.name, []).append(dp)

    metrics = []
    for name, data_points in by_name.items():
        m: dict = {
            "name":  name,
            "gauge": {"dataPoints": data_points},
        }
        # Find unit from the first counter with that name
        for ctr in trace.counters:
            if ctr.name == name and ctr.unit:
                m["unit"] = ctr.unit
                break
        metrics.append(m)

    return {
        "resourceMetrics": [
            {
                "resource": _resource(meta),
                "scopeMetrics": [
                    {
                        "scope":   {"name": f"{_SCOPE}.metrics", "version": _VERSION},
                        "metrics": metrics,
                    }
                ],
            }
        ]
    }


# ── HTTP transport ────────────────────────────────────────────────────────────

def _post(url: str, payload: dict, timeout: int) -> None:
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status not in (200, 202):
                raise RuntimeError(f"HTTP {resp.status}")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to reach {url}: {exc.reason}") from exc
    except Exception as exc:
        raise RuntimeError(f"OTLP export to {url} failed: {exc}") from exc


# ── Public API ────────────────────────────────────────────────────────────────

def export(
    trace: "Trace",
    endpoint: str | None = None,
    output_file: str | None = None,
    timeout: int = 10,
) -> None:
    """
    Export *trace* in OTLP JSON format.

    Parameters
    ----------
    endpoint:    OTLP HTTP base URL, e.g. ``"http://localhost:4318"``.
                 Traces are POSTed to ``{endpoint}/v1/traces`` and metrics
                 (if present) to ``{endpoint}/v1/metrics``.
    output_file: Path to write the OTLP traces JSON.  Can be replayed later:
                 ``curl -X POST http://localhost:4318/v1/traces \\``
                 ``     -H 'Content-Type: application/json' -d @<file>``
    timeout:     HTTP request timeout in seconds (default 10).
    """
    traces_payload = build_traces_payload(trace)

    if output_file:
        Path(output_file).write_text(json.dumps(traces_payload))

    if endpoint:
        base = endpoint.rstrip("/")
        _post(f"{base}/v1/traces", traces_payload, timeout)

        metrics_payload = build_metrics_payload(trace)
        if metrics_payload.get("resourceMetrics"):
            _post(f"{base}/v1/metrics", metrics_payload, timeout)
