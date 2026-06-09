"""LLM tool definitions and implementations for the analysis agent."""

from __future__ import annotations
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.trace import Trace


def _ms(ns: int | float) -> float:
    return round(ns / 1_000_000, 3)


def _fmt_bytes(b: float) -> str:
    if b >= 1024 ** 3:
        return f"{b / 1024 ** 3:.2f} GB"
    if b >= 1024 ** 2:
        return f"{b / 1024 ** 2:.1f} MB"
    if b >= 1024:
        return f"{b / 1024:.0f} KB"
    return f"{b:.0f} B"


# ── Tool definitions (OpenAI function-calling format) ─────────────────────────

TOOL_DEFINITIONS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_hotspots",
            "description": (
                "Get the top N slowest spans from the profile. "
                "Optionally filter by backend category or minimum duration."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {
                        "type": "integer",
                        "description": "Number of results to return (default 10)",
                    },
                    "category": {
                        "type": "string",
                        "description": (
                            "Filter by category: cuda, rocm, opencl, openmp, "
                            "mpi, nccl, memory, sync, cpu, jit, nvtx"
                        ),
                    },
                    "min_duration_ms": {
                        "type": "number",
                        "description": "Only include spans longer than this (milliseconds)",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_kernel_details",
            "description": (
                "Get detailed statistics for a specific kernel or function by name. "
                "Supports partial/substring matching. "
                "Returns count, total time, p50/p90 latency, and representative tags."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Kernel or function name (or substring)",
                    },
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_memory_profile",
            "description": (
                "Detailed breakdown of memory operations: allocations, H2D/D2H/D2D transfers, "
                "bytes moved, and effective bandwidth per operation type."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_timeline_phases",
            "description": (
                "Divide the trace into equal time buckets and show per-category active % "
                "in each bucket. Useful for finding idle gaps, pipeline bubbles, "
                "and phase transitions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "bucket_ms": {
                        "type": "number",
                        "description": "Bucket width in ms (0 = auto: wall_time / 10)",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_sync_analysis",
            "description": (
                "Analyze synchronization overhead: which sync calls take the most time, "
                "total sync cost as % of wall time, and what each sync appears to be "
                "waiting for (via parent span links)."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_mpi_communication",
            "description": (
                "Summarize MPI communication: operations by type, bytes exchanged, "
                "blocking vs non-blocking breakdown, and send-wait request pairs."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_spans",
            "description": "Flexible span query with filtering, sorting, and tag inspection.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Filter by category",
                    },
                    "name_contains": {
                        "type": "string",
                        "description": "Filter by name substring (case-insensitive)",
                    },
                    "min_duration_ms": {
                        "type": "number",
                        "description": "Minimum span duration in milliseconds",
                    },
                    "max_duration_ms": {
                        "type": "number",
                        "description": "Maximum span duration in milliseconds",
                    },
                    "sort_by": {
                        "type": "string",
                        "enum": ["duration", "start_time", "name"],
                        "description": "Sort field (default: duration descending)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results to return (default 20)",
                    },
                    "include_tags": {
                        "type": "boolean",
                        "description": "Include span tags in output (default true)",
                    },
                },
                "required": [],
            },
        },
    },
]


# ── Dispatcher ────────────────────────────────────────────────────────────────

def execute_tool(name: str, arguments: dict, trace: "Trace") -> str:
    """Dispatch a tool call and return the result as a JSON string."""
    _handlers = {
        "get_hotspots":          _get_hotspots,
        "get_kernel_details":    _get_kernel_details,
        "get_memory_profile":    _get_memory_profile,
        "get_timeline_phases":   _get_timeline_phases,
        "get_sync_analysis":     _get_sync_analysis,
        "get_mpi_communication": _get_mpi_communication,
        "query_spans":           _query_spans,
    }
    fn = _handlers.get(name)
    if not fn:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        return fn(trace, **arguments)
    except TypeError as e:
        return json.dumps({"error": f"Bad arguments for {name}: {e}"})
    except Exception as e:
        return json.dumps({"error": str(e)})


# ── Tool implementations ───────────────────────────────────────────────────────

def _wall_ns(trace: "Trace") -> int:
    """Compute wall time from span range (more reliable than metadata when loading from JSON)."""
    if trace.spans:
        return max(max(s.end_ns for s in trace.spans) - min(s.start_ns for s in trace.spans), 1)
    return max(trace.duration_ns, 1)


def _get_hotspots(
    trace: "Trace",
    n: int = 10,
    category: str | None = None,
    min_duration_ms: float | None = None,
) -> str:
    from collections import defaultdict

    wall_ns = _wall_ns(trace)
    spans = trace.spans
    if category:
        spans = [s for s in spans if s.category.value == category]
    if min_duration_ms is not None:
        spans = [s for s in spans if s.duration_ns >= min_duration_ms * 1e6]

    # Aggregate by name+category
    agg: dict[tuple, dict] = defaultdict(
        lambda: {"count": 0, "total_ns": 0, "max_ns": 0, "tags": {}}
    )
    for s in spans:
        key = (s.name, s.category.value)
        e = agg[key]
        e["count"] += 1
        e["total_ns"] += s.duration_ns
        e["max_ns"] = max(e["max_ns"], s.duration_ns)
        if not e["tags"] and s.tags:
            e["tags"] = {k: v for k, v in s.tags.items() if k not in ("sid", "psid")}

    top = sorted(agg.items(), key=lambda kv: -kv[1]["total_ns"])[: int(n)]
    result = []
    for (sname, cat), e in top:
        entry: dict = {
            "name": sname,
            "category": cat,
            "count": e["count"],
            "total_ms": _ms(e["total_ns"]),
            "avg_ms": _ms(e["total_ns"] / e["count"]),
            "max_ms": _ms(e["max_ns"]),
            "pct_wall_time": round(100.0 * e["total_ns"] / wall_ns, 2),
        }
        if e["tags"]:
            entry["tags"] = e["tags"]
        result.append(entry)

    return json.dumps(result, indent=2, default=str)


def _get_kernel_details(trace: "Trace", name: str = "") -> str:
    matching = [s for s in trace.spans if name.lower() in s.name.lower()]
    if not matching:
        return json.dumps({"error": f"No spans matching '{name}'"})

    wall_ns = _wall_ns(trace)
    total_ns = sum(s.duration_ns for s in matching)
    durs = sorted(s.duration_ns for s in matching)
    n = len(durs)

    result: dict = {
        "query": name,
        "match_count": n,
        "total_ms": _ms(total_ns),
        "pct_wall_time": round(100.0 * total_ns / wall_ns, 2),
        "avg_ms": _ms(total_ns / n),
        "min_ms": _ms(durs[0]),
        "max_ms": _ms(durs[-1]),
        "p50_ms": _ms(durs[n // 2]),
        "p90_ms": _ms(durs[min(n - 1, int(n * 0.9))]),
        "categories": list(dict.fromkeys(s.category.value for s in matching)),
    }

    # Representative tags
    all_tags: dict = {}
    for s in matching[:5]:
        for k, v in s.tags.items():
            if k not in ("sid", "psid") and k not in all_tags:
                all_tags[k] = v
    if all_tags:
        result["representative_tags"] = all_tags

    # Parent span links
    sid_map = {s.span_id: s.name for s in trace.spans if s.span_id}
    parent_names = list(dict.fromkeys(
        sid_map[s.parent_span_id]
        for s in matching
        if s.parent_span_id and s.parent_span_id in sid_map
    ))
    if parent_names:
        result["launched_from"] = parent_names[:3]

    return json.dumps(result, indent=2, default=str)


def _get_memory_profile(trace: "Trace") -> str:
    mem_spans = [s for s in trace.spans if s.category.value == "memory"]
    if not mem_spans:
        return json.dumps({"note": "No memory operation spans in this trace."})

    by_type: dict[str, dict] = {}
    for s in mem_spans:
        op = s.tags.get("type", s.name)
        if op not in by_type:
            by_type[op] = {"count": 0, "total_ns": 0, "total_bytes": 0.0}
        e = by_type[op]
        e["count"] += 1
        e["total_ns"] += s.duration_ns
        e["total_bytes"] += float(s.tags.get("bytes", 0) or 0)

    result = []
    for op, e in sorted(by_type.items(), key=lambda kv: -kv[1]["total_ns"]):
        entry: dict = {
            "operation": op,
            "count": e["count"],
            "total_ms": _ms(e["total_ns"]),
            "avg_ms": _ms(e["total_ns"] / e["count"]),
        }
        if e["total_bytes"] > 0:
            entry["total_bytes"] = _fmt_bytes(e["total_bytes"])
            if e["total_ns"] > 0:
                bw = (e["total_bytes"] / 1e9) / (e["total_ns"] / 1e9)
                entry["effective_bandwidth_gbs"] = round(bw, 2)
        result.append(entry)

    return json.dumps(result, indent=2, default=str)


def _get_timeline_phases(trace: "Trace", bucket_ms: float = 0.0) -> str:
    wall_ns = _wall_ns(trace)
    bucket_ns = int(bucket_ms * 1e6) if bucket_ms > 0 else max(1, wall_ns // 10)
    n_buckets = max(1, (wall_ns + bucket_ns - 1) // bucket_ns)
    if n_buckets > 20:
        bucket_ns = (wall_ns + 19) // 20
        n_buckets = 20
    bucket_ms_actual = bucket_ns / 1e6

    buckets: list[dict[str, float]] = [{} for _ in range(n_buckets)]
    for s in trace.spans:
        if s.duration_ns <= 0:
            continue
        b_start = max(0, s.start_ns // bucket_ns)
        b_end = min(n_buckets - 1, s.end_ns // bucket_ns)
        for b in range(int(b_start), int(b_end) + 1):
            bns0 = b * bucket_ns
            bns1 = bns0 + bucket_ns
            overlap = min(s.end_ns, bns1) - max(s.start_ns, bns0)
            if overlap > 0:
                cat = s.category.value
                buckets[b][cat] = buckets[b].get(cat, 0.0) + overlap

    phases = []
    for i, b in enumerate(buckets):
        entry: dict = {
            "window": f"{i * bucket_ms_actual:.1f}–{(i + 1) * bucket_ms_actual:.1f}ms",
        }
        for cat, ns in sorted(b.items(), key=lambda kv: -kv[1]):
            pct = round(100.0 * ns / bucket_ns, 1)
            if pct >= 1.0:
                entry[cat] = f"{pct}%"
        if len(entry) == 1:
            entry["idle"] = "100%"
        phases.append(entry)

    return json.dumps(
        {"bucket_size_ms": round(bucket_ms_actual, 2), "phases": phases},
        indent=2,
    )


def _get_sync_analysis(trace: "Trace") -> str:
    sync_spans = [s for s in trace.spans if s.category.value == "sync"]
    if not sync_spans:
        return json.dumps({"note": "No synchronization spans found."})

    wall_ns = _wall_ns(trace)
    total_sync_ns = sum(s.duration_ns for s in sync_spans)
    sid_map = {s.span_id: s.name for s in trace.spans if s.span_id}

    by_name: dict[str, dict] = {}
    for s in sync_spans:
        if s.name not in by_name:
            by_name[s.name] = {"count": 0, "total_ns": 0, "parents": set()}
        e = by_name[s.name]
        e["count"] += 1
        e["total_ns"] += s.duration_ns
        if s.parent_span_id and s.parent_span_id in sid_map:
            e["parents"].add(sid_map[s.parent_span_id])

    breakdown = []
    for sname, e in sorted(by_name.items(), key=lambda kv: -kv[1]["total_ns"]):
        entry: dict = {
            "name": sname,
            "count": e["count"],
            "total_ms": _ms(e["total_ns"]),
            "pct_wall_time": round(100.0 * e["total_ns"] / wall_ns, 2),
            "pct_of_total_sync": round(100.0 * e["total_ns"] / total_sync_ns, 1),
        }
        if e["parents"]:
            entry["waiting_for"] = sorted(e["parents"])[:3]
        breakdown.append(entry)

    return json.dumps(
        {
            "total_sync_ms": _ms(total_sync_ns),
            "sync_pct_wall_time": round(100.0 * total_sync_ns / wall_ns, 2),
            "breakdown": breakdown,
        },
        indent=2,
        default=str,
    )


def _get_mpi_communication(trace: "Trace") -> str:
    mpi_spans = [s for s in trace.spans if s.category.value == "mpi"]
    if not mpi_spans:
        return json.dumps({"note": "No MPI spans found. Was --backend mpi used?"})

    wall_ns = _wall_ns(trace)
    by_op: dict[str, dict] = {}
    for s in mpi_spans:
        op = s.name
        if op not in by_op:
            by_op[op] = {"count": 0, "total_ns": 0, "total_bytes": 0.0, "linked": False}
        e = by_op[op]
        e["count"] += 1
        e["total_ns"] += s.duration_ns
        e["total_bytes"] += float(s.tags.get("bytes", 0) or 0)
        if s.parent_span_id:
            e["linked"] = True

    ops = []
    for op, e in sorted(by_op.items(), key=lambda kv: -kv[1]["total_ns"]):
        entry: dict = {
            "operation": op,
            "count": e["count"],
            "total_ms": _ms(e["total_ns"]),
            "pct_wall_time": round(100.0 * e["total_ns"] / wall_ns, 2),
        }
        if e["total_bytes"] > 0:
            entry["total_bytes"] = _fmt_bytes(e["total_bytes"])
        if e["linked"]:
            entry["has_send_wait_links"] = True
        ops.append(entry)

    isend_ids = {s.span_id for s in mpi_spans if "isend" in s.name.lower() and s.span_id}
    wait_parents = {s.parent_span_id for s in mpi_spans if s.parent_span_id}
    linked_pairs = len(isend_ids & wait_parents)

    return json.dumps(
        {
            "total_mpi_spans": len(mpi_spans),
            "linked_isend_wait_pairs": linked_pairs,
            "operations": ops,
        },
        indent=2,
        default=str,
    )


def _query_spans(
    trace: "Trace",
    category: str | None = None,
    name_contains: str | None = None,
    min_duration_ms: float | None = None,
    max_duration_ms: float | None = None,
    sort_by: str = "duration",
    limit: int = 20,
    include_tags: bool = True,
) -> str:
    wall_ns = _wall_ns(trace)
    spans = trace.spans

    if category:
        spans = [s for s in spans if s.category.value == category]
    if name_contains:
        spans = [s for s in spans if name_contains.lower() in s.name.lower()]
    if min_duration_ms is not None:
        spans = [s for s in spans if s.duration_ns >= min_duration_ms * 1e6]
    if max_duration_ms is not None:
        spans = [s for s in spans if s.duration_ns <= max_duration_ms * 1e6]

    if sort_by == "start_time":
        spans = sorted(spans, key=lambda s: s.start_ns)
    elif sort_by == "name":
        spans = sorted(spans, key=lambda s: s.name)
    else:
        spans = sorted(spans, key=lambda s: s.duration_ns, reverse=True)

    spans = spans[: int(limit)]
    sid_map = {s.span_id: s.name for s in trace.spans if s.span_id}

    result = []
    for s in spans:
        entry: dict = {
            "name": s.name,
            "category": s.category.value,
            "duration_ms": _ms(s.duration_ns),
            "pct_wall_time": round(100.0 * s.duration_ns / wall_ns, 2),
            "start_ms": _ms(s.start_ns),
        }
        if include_tags and s.tags:
            entry["tags"] = {k: v for k, v in s.tags.items() if k not in ("sid", "psid")}
        if s.parent_span_id:
            entry["parent"] = sid_map.get(s.parent_span_id, f"sid:{s.parent_span_id[:8]}")
        result.append(entry)

    return json.dumps(
        {"total_matching": len(result), "spans": result},
        indent=2,
        default=str,
    )
