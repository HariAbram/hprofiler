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
    print(f"  Total spans : {trace.span_count()}")

    # ── CPU microarch + memory stats from counter events ───────────────────
    ctrs: dict[str, float] = {}
    for c in trace.iter_counters():
        if c.name in ("ipc", "cache_miss_pct", "branch_miss_pct",
                      "process_max_rss_bytes"):
            ctrs[c.name] = c.value   # last sample wins

    # Peak GPU utilisation / memory from polling counters
    gpu_util_peak: dict[str, float] = {}
    gpu_mem_peak:  dict[str, float] = {}
    for c in trace.iter_counters():
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

    # GPU kernel active % — use merged intervals so concurrent streams don't
    # cause the percentage to exceed 100%.
    ext = trace.store.span_extent(timed_only=True)
    wall_ns = max(ext[1] - ext[0], 1) if ext else (trace.duration_ns or 1)

    # One timing source per number (core/gpu_activity.kernel_activity):
    # device-measured kernels when present, else event-timed proxies --
    # never a union of both, which would place the same kernel at its
    # submission time AND at its real execution time.
    from ..core import gpu_activity as _ga
    for cat_val, label in (("cuda", "CUDA"), ("rocm", "ROCm"), ("opencl", "OpenCL")):
        ka = _ga.kernel_activity(trace.iter_spans(categories=(cat_val,)))
        if not ka.used:
            continue
        active_ns = _ga.merged_length(ka.intervals)
        total_ns = sum(hi - lo for lo, hi in ka.intervals)
        pct = 100.0 * active_ns / wall_ns
        print(f"\n  {label + ' kernel active':<23}: "
              f"{pct:.2f}% of wall time  "
              f"({_fmt_ns(active_ns)} active, "
              f"{_fmt_ns(total_ns)} accumulated, "
              f"{ka.used} launches)")
        print(f"    timing source       : {_ga.SOURCE_LABELS.get(ka.source, ka.source)}")
        if ka.excluded:
            print(f"    excluded            : {ka.excluded} kernel span(s) -- {ka.excluded_reason}")
        if ka.source == "device" and cat_val in _ga.RUNTIMES:
            native = [s for s in trace.iter_spans(categories=(cat_val,))
                      if _ga.is_device_kernel(s) and "queue_ns" in s.tags]
            if native:
                def _p50(key: str) -> float:
                    vals = sorted(int(s.tags[key]) for s in native)
                    return vals[len(vals) // 2]
                print(f"    per launch (median) : host call {_fmt_ns(_p50('api_ns'))}, "
                      f"queued {_fmt_ns(_p50('queue_ns'))}, "
                      f"device {_fmt_ns(sorted(s.duration_ns for s in native)[len(native) // 2])}")

    for line in _ga.describe(meta.device_activity):
        print(f"  {line}")

    if gpu_util_peak:
        _amd_backends = {"rocm"}
        smi_tool = "rocm-smi" if any(b in _amd_backends for b in (meta.backends_used or [])) \
                   else "nvidia-smi"
        print(f"\n  GPU utilisation ({smi_tool} peak, 1 s poll):")
        for key, val in sorted(gpu_util_peak.items()):
            lbl = key.replace("gpu_utilization_pct", "").strip("[]") or "gpu0"
            print(f"    {lbl:<6}  compute  : {val:.0f}%  "
                  f"(0% expected if kernels are shorter than the poll interval)")
        for key, val in sorted(gpu_mem_peak.items()):
            lbl = key.replace("gpu_mem_used_bytes", "").strip("[]") or "gpu0"
            print(f"    {lbl:<6}  mem used : {_fmt_bytes(val)}")

    # ── Category breakdown ──────────────────────────────────────────────────
    by_cat: dict[str, list[int]] = {}
    for r in trace.aggregate_stats():           # store-side per-name totals
        v = by_cat.setdefault(r["category"], [0, 0])
        v[0] += r["count"]
        v[1] += r["total_ns"]
    if by_cat:
        print(f"\n  Events by category:")
        for cat, (n, total) in sorted(by_cat.items(), key=lambda kv: -kv[1][1]):
            print(f"    {cat:<12} {n:>6} events   {_fmt_ns(total):>12}")

    stats = trace.aggregated_stats()
    timed  = [r for r in stats if r["total_ns"] > 0]
    samples = [r for r in stats if r["total_ns"] == 0 and r["category"] == "cpu"]
    if timed:
        has_omp = any(r["category"] in ("openmp", "sync", "opencl") for r in timed)
        total_note = " (accumulated device/thread time, not wall time)" if has_omp else ""
        print(f"\n  Top {min(top_n, len(timed))} hotspots{total_note}:")
        hdr = (f"  {'Function':<40} {'Cat':<8} {'Count':>6}"
               f" {'Total':>10} {'Avg/call':>10} {'%':>6}")
        print(hdr)
        print(f"  {'-'*80}")
        for row in timed[:top_n]:
            name = row["name"][:38]
            print(
                f"  {name:<40} {row['category']:<8} {row['count']:>6}"
                f" {_fmt_ns(row['total_ns']):>10} {_fmt_ns(row['avg_ns']):>10}"
                f" {row['pct']:>5.1f}%"
            )
    if samples:
        top_s = sorted(samples, key=lambda r: -r["count"])[:10]
        print(f"\n  Top CPU sample functions (see the Flame Graph tab for the full view):")
        print(f"  {'Function':<50} {'Samples':>8}")
        print(f"  {'-'*60}")
        for row in top_s:
            print(f"  {row['name'][:48]:<50} {row['count']:>8}")

    # ── CCT call-path hotspots (only when HPROFILER_CALLSTACK captured stacks) ─
    if trace._has_stacks:
        try:
            cct = trace.cct()
            cct.print_summary(wall_ns=trace.duration_ns, top_n=top_n)
        except Exception:
            pass

    # ── GPU starvation analysis ────────────────────────────────────────────────
    _GPU_BACKENDS = {"cuda", "rocm", "opencl"}
    if any(b in (meta.backends_used or []) for b in _GPU_BACKENDS):
        try:
            from ..analysis.cct import gpu_starvation
            sv = gpu_starvation(trace)
            if sv["gpu_active_pct"] > 0 or sv["sync_stall_pct"] > 0:
                print(f"\n  GPU timeline analysis:")
                print(f"    GPU kernel active      : {sv['gpu_active_pct']:>6.1f}%"
                      f"  ({_fmt_ns(sv['gpu_active_ns'])})")
                print(f"    CPU sync stalls        : {sv['sync_stall_pct']:>6.1f}%"
                      f"  ({_fmt_ns(sv['sync_stall_ns'])}"
                      f", {sv['sync_calls']} sync calls)")
                print(f"    GPU idle (launch gaps) : {sv['launch_gap_pct']:>6.1f}%"
                      f"  ({_fmt_ns(sv['launch_gap_ns'])})")
                if sv['sync_stall_pct'] > 20:
                    print(f"    [!] High sync stall — consider async launches "
                          f"or batching kernel submissions")
                if sv['launch_gap_pct'] > 30:
                    print(f"    [!] High launch gap — GPU idle >30% of wall time; "
                          f"check CPU-side compute between launches")
        except Exception:
            pass

    print(f"{'='*72}\n")
