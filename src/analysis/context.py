"""Convert a Trace into a compact, LLM-ready structured context dict."""

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


def build_profile_context(trace: "Trace") -> dict:
    """Return a structured dict capturing essential profile data for LLM analysis."""
    meta = trace.metadata
    # Use span range for wall time — more reliable than metadata when loading from JSON
    if trace.spans:
        span_start = min(s.start_ns for s in trace.spans)
        span_end = max(s.end_ns for s in trace.spans)
        wall_ns = max(span_end - span_start, 1)
    else:
        wall_ns = max(trace.duration_ns, 1)

    cmd = " ".join(filter(None, [meta.command] + (meta.args or [])))
    thread_count = len({s.tid for s in trace.spans if s.tid})
    run: dict = {
        "command": cmd or "(unknown)",
        "wall_time_ms": _ms(wall_ns),
        "backends": meta.backends_used or [],
        "hostname": meta.hostname or "unknown",
        "total_spans": len(trace.spans),
        "thread_count": thread_count or None,
    }
    if thread_count > 1:
        run["note"] = (
            "Span durations sum per-thread, so category totals can exceed 100% of wall time "
            "in multi-threaded profiling. Percentages reflect cumulative thread time, not elapsed time."
        )

    hardware = _build_hardware(trace)

    # Time breakdown by category
    by_cat: dict[str, int] = {}
    for s in trace.spans:
        by_cat[s.category.value] = by_cat.get(s.category.value, 0) + s.duration_ns

    breakdown: dict = {}
    for cat, ns in sorted(by_cat.items(), key=lambda kv: -kv[1]):
        pct = 100.0 * ns / wall_ns
        if pct >= 0.1:
            breakdown[cat] = {
                "total_ms": _ms(ns),
                "pct_wall_time": round(pct, 2),
                "span_count": sum(1 for s in trace.spans if s.category.value == cat),
            }

    gpu_util: dict[str, object] = {}
    gpu_mem_peak: dict[str, float] = {}
    for c in trace.counters:
        if c.name.startswith("gpu_utilization_pct"):
            lbl = c.name.replace("gpu_utilization_pct", "").strip("[]") or "0"
            key = f"gpu{lbl}_util_pct"
            gpu_util[key] = max(float(gpu_util.get(key, 0)), c.value)
        if c.name.startswith("gpu_mem_used_bytes"):
            lbl = c.name.replace("gpu_mem_used_bytes", "").strip("[]") or "0"
            gpu_mem_peak[lbl] = max(gpu_mem_peak.get(lbl, 0.0), c.value)

    for cat_val in ("cuda", "rocm"):
        kernel_ns = sum(
            s.duration_ns for s in trace.spans
            if s.category.value == cat_val and s.tags.get("type") == "kernel"
        )
        if kernel_ns:
            gpu_util[f"{cat_val}_kernel_active_pct"] = round(100.0 * kernel_ns / wall_ns, 2)

    if gpu_mem_peak:
        for lbl, b in gpu_mem_peak.items():
            gpu_util[f"gpu{lbl}_mem_used"] = _fmt_bytes(b)

    memory = _build_memory(trace)
    hotspots = _build_hotspots(trace, wall_ns, n=15)

    hw_counters: dict[str, float] = {}
    for c in trace.counters:
        if c.name in ("ipc", "cache_miss_pct", "branch_miss_pct",
                      "dram_bw_gbs", "l3_bw_gbs", "flops_sp", "flops_dp"):
            hw_counters[c.name] = round(c.value, 3)

    hierarchy = _build_hierarchy(trace, wall_ns)
    roofline = _build_roofline(trace)

    ctx: dict = {"run": run}
    if hardware:
        ctx["hardware"] = hardware
    ctx["time_breakdown_by_category"] = breakdown
    if gpu_util:
        ctx["gpu_activity"] = gpu_util
    if memory:
        ctx["memory"] = memory
    ctx["top_hotspots"] = hotspots
    if hw_counters:
        ctx["hardware_counters"] = hw_counters
    if hierarchy:
        ctx["span_hierarchy_sample"] = hierarchy
    if roofline:
        ctx["roofline_analysis"] = roofline

    return ctx


def _build_hardware(trace: "Trace") -> dict:
    devices = trace.devices
    if not devices:
        return {}
    result: dict = {}
    for d in devices:
        key = f"{d.backend}_device"
        entry: dict = {
            "name": d.name,
            "fp32_tflops": round(d.fp32_tflops, 2),
            "fp64_tflops": round(d.fp64_tflops, 2),
            "peak_mem_bw_gbs": round(d.bandwidth_gbs, 1),
        }
        if getattr(d, "fp16_tflops", 0):
            entry["fp16_tflops"] = round(d.fp16_tflops, 2)
        if getattr(d, "tensor_tflops", 0):
            entry["tensor_tflops"] = round(d.tensor_tflops, 2)
        if getattr(d, "vram_gb", 0):
            entry["vram_gb"] = round(d.vram_gb, 1)
        if getattr(d, "sm_count", 0):
            entry["sm_count"] = d.sm_count
        result[key] = entry
    return result


def _build_memory(trace: "Trace") -> dict:
    mem: dict = {}

    rss_bytes = 0.0
    for c in trace.counters:
        if c.name == "process_max_rss_bytes":
            rss_bytes = max(rss_bytes, c.value)
    if rss_bytes:
        mem["process_peak_rss"] = _fmt_bytes(rss_bytes)

    mem_spans = [s for s in trace.spans if s.category.value == "memory"]
    if mem_spans:
        total_bytes = sum(float(s.tags.get("bytes", 0) or 0) for s in mem_spans)
        total_ns = sum(s.duration_ns for s in mem_spans)
        entry: dict = {
            "transfer_count": len(mem_spans),
            "total_transfer_ms": _ms(total_ns),
        }
        if total_bytes > 0:
            entry["total_bytes_moved"] = _fmt_bytes(total_bytes)
            if total_ns > 0:
                bw = (total_bytes / 1e9) / (total_ns / 1e9)
                entry["effective_bandwidth_gbs"] = round(bw, 2)
        mem["transfers"] = entry

    return mem


def _build_hotspots(trace: "Trace", wall_ns: int, n: int = 15) -> list[dict]:
    stats = trace.aggregated_stats()
    result: list[dict] = []
    for row in stats[:n]:
        entry: dict = {
            "name": row["name"],
            "category": row["category"],
            "count": row["count"],
            "total_ms": round(row["total_ns"] / 1e6, 3),
            "avg_ms": round(row["avg_ns"] / 1e6, 3),
            "pct_wall_time": round(100.0 * row["total_ns"] / wall_ns, 2),
        }
        rep = next(
            (s for s in trace.spans if s.name == row["name"] and s.category.value == row["category"]),
            None,
        )
        if rep and rep.tags:
            key_tags = {
                k: v for k, v in rep.tags.items()
                if k in ("type", "grid", "block", "stream", "bytes", "dir", "count")
            }
            if key_tags:
                entry["tags"] = key_tags
        result.append(entry)
    return result


def _build_hierarchy(trace: "Trace", wall_ns: int) -> list[dict]:
    sid_to_span = {s.span_id: s for s in trace.spans if s.span_id}
    parent_to_children: dict[str, list] = {}
    for s in trace.spans:
        if s.parent_span_id and s.parent_span_id in sid_to_span:
            parent_to_children.setdefault(s.parent_span_id, []).append(s)

    if not parent_to_children:
        return []

    result: list[dict] = []
    for parent_sid, children in sorted(
        parent_to_children.items(),
        key=lambda kv: -sum(c.duration_ns for c in kv[1]),
    )[:10]:
        parent = sid_to_span[parent_sid]
        parent_ms = _ms(parent.duration_ns)
        children_ms = _ms(sum(c.duration_ns for c in children))
        result.append({
            "parent": f"{parent.name} ({parent.category.value}, {parent_ms}ms)",
            "children_count": len(children),
            "children_total_ms": children_ms,
            "children_pct_of_parent": round(
                100.0 * children_ms / parent_ms if parent_ms > 0 else 0, 1
            ),
            "child_names": list(dict.fromkeys(c.name for c in children))[:5],
        })
    return result


def _build_roofline(trace: "Trace") -> list[dict]:
    try:
        from ..analysis.roofline import analyze_trace
        results = analyze_trace(trace)
        if not results:
            return []
        out: list[dict] = []
        for _, m in results[:10]:
            bound = getattr(m, "bound", "unknown")
            arith = round(getattr(m, "arith_intensity", 0), 3)
            pct = round(
                (getattr(m, "bw_pct", 0) if bound == "memory" else getattr(m, "flops_pct", 0)),
                1,
            )
            out.append({
                "kernel": getattr(m, "kernel_name", "?")[:40],
                "duration_ms": _ms(getattr(m, "duration_ns", 0)),
                "bound": bound,
                "arithmetic_intensity_flop_per_byte": arith,
                "achieved_pct_of_peak": pct,
            })
        return out
    except Exception:
        return []


def context_to_str(ctx: dict) -> str:
    """Serialize context dict to compact JSON for prompt inclusion."""
    return json.dumps(ctx, indent=2, default=str)
