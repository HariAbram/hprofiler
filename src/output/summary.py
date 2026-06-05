"""Terminal-friendly text summary of a trace."""

from __future__ import annotations
from ..core.trace import Trace


def _fmt_ns(ns: float) -> str:
    if ns >= 1_000_000_000:
        return f"{ns/1_000_000_000:.3f}s"
    if ns >= 1_000_000:
        return f"{ns/1_000_000:.2f}ms"
    if ns >= 1_000:
        return f"{ns/1_000:.1f}µs"
    return f"{ns:.0f}ns"


def _fmt_bytes(b: float) -> str:
    if b >= 1024**3:
        return f"{b/1024**3:.2f} GB"
    if b >= 1024**2:
        return f"{b/1024**2:.1f} MB"
    if b >= 1024:
        return f"{b/1024:.0f} KB"
    return f"{b:.0f} B"


def print_summary(trace: Trace, top_n: int = 20) -> None:
    meta = trace.metadata
    print(f"\n{'='*72}")
    print(f"  Profiler Summary: {meta.command} {' '.join(meta.args)}")
    print(f"{'='*72}")
    print(f"  Total time  : {_fmt_ns(trace.duration_ns)}")
    print(f"  Backends    : {', '.join(meta.backends_used) or '(none)'}")
    print(f"  Total spans : {len(trace.spans)}")

    # ── CPU microarch + memory stats from counter events ───────────────────
    ctrs: dict[str, float] = {}
    for c in trace.counters:
        if c.name in ("ipc", "cache_miss_pct", "branch_miss_pct",
                      "process_max_rss_bytes"):
            ctrs[c.name] = c.value   # last sample wins

    # Peak GPU utilisation / memory from polling counters
    gpu_util_peak: dict[str, float] = {}
    gpu_mem_peak:  dict[str, float] = {}
    for c in trace.counters:
        if c.name.startswith("gpu_utilization_pct"):
            key = c.name
            gpu_util_peak[key] = max(gpu_util_peak.get(key, 0.0), c.value)
        if c.name.startswith("gpu_mem_used_bytes"):
            key = c.name
            gpu_mem_peak[key] = max(gpu_mem_peak.get(key, 0.0), c.value)

    # Print microarch stats if present
    has_arch = any(k in ctrs for k in ("ipc", "cache_miss_pct", "branch_miss_pct"))
    if has_arch:
        print(f"\n  CPU microarch:")
        if "ipc" in ctrs:
            print(f"    IPC                 : {ctrs['ipc']:.2f}")
        if "cache_miss_pct" in ctrs:
            print(f"    LLC cache miss rate : {ctrs['cache_miss_pct']:.2f}%")
        if "branch_miss_pct" in ctrs:
            print(f"    Branch miss rate    : {ctrs['branch_miss_pct']:.2f}%")

    if "process_max_rss_bytes" in ctrs:
        print(f"\n  Peak process RSS    : {_fmt_bytes(ctrs['process_max_rss_bytes'])}")

    # GPU kernel active % — computed from span durations (accurate even for
    # short-running workloads where nvidia-smi polling would read 0%).
    wall_ns = trace.duration_ns or 1
    for cat_val, label in (("cuda", "CUDA"), ("rocm", "ROCm")):
        from ..core.events import Category
        gpu_spans = [s for s in trace.spans
                     if s.category.value == cat_val
                     and s.tags.get("type") == "kernel"]
        if gpu_spans:
            kernel_ns = sum(s.duration_ns for s in gpu_spans)
            pct = 100.0 * kernel_ns / wall_ns
            print(f"\n  {label} kernel active      : "
                  f"{pct:.2f}% of wall time  "
                  f"({_fmt_ns(kernel_ns)} total kernel time, "
                  f"{len(gpu_spans)} launches)")

    if gpu_util_peak:
        print(f"\n  GPU utilisation (nvidia-smi peak, 100 ms poll):")
        for key, val in sorted(gpu_util_peak.items()):
            lbl = key.replace("gpu_utilization_pct", "").strip("[]") or "0"
            print(f"    GPU {lbl:<4}  compute  : {val:.0f}%  "
                  f"(0% expected if kernels are shorter than the poll interval)")
        for key, val in sorted(gpu_mem_peak.items()):
            lbl = key.replace("gpu_mem_used_bytes", "").strip("[]") or "0"
            print(f"    GPU {lbl:<4}  mem used : {_fmt_bytes(val)}")

    # ── Category breakdown ──────────────────────────────────────────────────
    by_cat = trace.spans_by_category()
    if by_cat:
        print(f"\n  Events by category:")
        for cat, spans in sorted(by_cat.items(),
                                  key=lambda kv: -sum(s.duration_ns for s in kv[1])):
            total = sum(s.duration_ns for s in spans)
            print(f"    {cat.value:<12} {len(spans):>6} events   {_fmt_ns(total):>12}")

    stats = trace.aggregated_stats()
    if stats:
        print(f"\n  Top {min(top_n, len(stats))} hotspots:")
        hdr = (f"  {'Function':<40} {'Cat':<8} {'Count':>6}"
               f" {'Total':>10} {'Avg':>10} {'%':>6}")
        print(hdr)
        print(f"  {'-'*80}")
        for row in stats[:top_n]:
            name = row["name"][:38]
            print(
                f"  {name:<40} {row['category']:<8} {row['count']:>6}"
                f" {_fmt_ns(row['total_ns']):>10} {_fmt_ns(row['avg_ns']):>10}"
                f" {row['pct']:>5.1f}%"
            )

    print(f"{'='*72}\n")
