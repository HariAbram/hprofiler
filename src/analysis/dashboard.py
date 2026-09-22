"""
UI-agnostic "what's wrong with this trace" analysis and small formatting
helpers, shared by the TUI's Overview tab (src/ui/app.py) and the Qt/QML
GUI (src/gui/). This module presents analysis that already exists
elsewhere (analysis/cct.gpu_starvation, analysis/pop_efficiency.load_balance,
Trace.aggregated_stats) as diagnosis/findings text -- it does not compute
new metrics -- so no UI layer reimplements this logic, and the TUI, the
GUI, and `hprofiler summary`/`efficiency` can never disagree with each
other about what's wrong with a given trace.

Originally lived directly in src/ui/app.py (Rich/Textual-specific);
extracted here when the Qt/QML GUI was added so both UIs import the same
functions instead of one copying the other. Functions that return actual
rendered output (colors, markup) stay UI-specific and are NOT here --
this module returns plain data (strings, tuples, dataclasses) that each
UI formats in its own idiom.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.trace import Trace

_GPU_CATS = ("cuda", "rocm", "opencl")

_JIT_HASH_RE = re.compile(r'^(\d+)\.(\d+)\.jit\.so$')


# ── Formatting helpers ──────────────────────────────────────────────────────

def fmt_ns(ns: float) -> str:
    if ns >= 1_000_000_000:
        return f"{ns / 1e9:.3f}s"
    if ns >= 1_000_000:
        return f"{ns / 1e6:.2f}ms"
    if ns >= 1_000:
        return f"{ns / 1e3:.1f}µs"
    return f"{ns:.0f}ns"


def fmt_bytes(b: float) -> str:
    if b >= 1024**3:
        return f"{b/1024**3:.2f} GB"
    if b >= 1024**2:
        return f"{b/1024**2:.1f} MB"
    if b >= 1024:
        return f"{b/1024:.0f} KB"
    return f"{b:.0f} B"


def fmt_tf(tf: float) -> str:
    """Format TFLOPs value compactly."""
    if tf >= 1000:
        return f"{tf/1000:.1f} PF"
    if tf >= 1:
        return f"{tf:.1f} TF"
    if tf >= 0.001:
        return f"{tf*1000:.0f} GF"
    return f"{tf:.2g} TF"


def fmt_kernel_name(name: str) -> str:
    """Shorten ACPP SSCP hash-named JIT kernels to a readable form."""
    m = _JIT_HASH_RE.match(name)
    if m:
        return f"[jit:{m.group(1)[-6:]}…{m.group(2)[-4:]}]"
    return name


# ── Trace-derived timing helpers ────────────────────────────────────────────

def merged_ns(spans: list) -> int:
    """Merged-interval sum of span durations — prevents >100% from concurrent streams."""
    ivs = sorted((s.start_ns, s.start_ns + s.duration_ns)
                 for s in spans if s.duration_ns > 0)
    merged = cur_lo = cur_hi = 0
    for lo, hi in ivs:
        if lo > cur_hi:
            merged += cur_hi - cur_lo
            cur_lo, cur_hi = lo, hi
        else:
            cur_hi = max(cur_hi, hi)
    merged += cur_hi - cur_lo
    return merged


def trace_wall_ns(trace: Any) -> int:
    """Derive wall time from span timestamps (works for live runs and JSON-loaded traces)."""
    timed = [s for s in trace.spans if s.duration_ns > 0]
    if not timed:
        return trace.duration_ns or 1
    span_end   = max(s.start_ns + s.duration_ns for s in timed)
    span_start = min(s.start_ns for s in timed)
    return max(span_end - span_start, 1)


# ── Diagnosis / findings ─────────────────────────────────────────────────────

def bottleneck_analysis(trace: Trace, ctrs: dict[str, float]) -> list[str]:
    """
    Short "icon + one-liner" diagnostic tips. Each tip is prefixed with a
    2-char icon; callers split via `icon, body = tip[:2], tip[2:].strip()`.

    Icons are plain box-drawing/geometric-shape glyphs ("!", "▲", "◆" --
    matching top_findings' vocabulary), deliberately not emoji -- emoji
    glyph coverage over a bare SSH session to an HPC cluster is unreliable
    (missing glyphs silently fall back to whatever the local font
    substitutes, e.g. a stray unrelated letter) and many render as
    double-width, which would also throw off the fixed `tip[:2]` icon
    slice every caller relies on.
    """
    tips: list[str] = []
    meta = trace.metadata

    if any(b in _GPU_CATS for b in (meta.backends_used or [])):
        try:
            from .cct import gpu_starvation
            sv = gpu_starvation(trace)
            if sv["launch_gap_pct"] > 30:
                tips.append(
                    f"! Low GPU occupancy — idle {sv['launch_gap_pct']:.0f}% of wall "
                    f"time between kernel launches; check CPU-side work in between")
            if sv["sync_stall_pct"] > 20:
                tips.append(
                    f"▲ High GPU sync stall ({sv['sync_stall_pct']:.0f}%) — "
                    f"consider async launches or batching kernel submissions")
        except Exception:
            pass

    ipc = ctrs.get("ipc", 0.0)
    if 0 < ipc < 1.0:
        tips.append(f"▲ Low IPC ({ipc:.2f}) — likely stalled on memory or branch mispredicts")

    cache_miss = ctrs.get("cache_miss_pct", -1.0)
    if cache_miss >= 20:
        tips.append(f"▲ High LLC miss rate ({cache_miss:.0f}%) — working set may exceed cache")

    try:
        from .pop_efficiency import useful_time_by_pid, load_balance
        lb = load_balance(useful_time_by_pid(trace))
        if lb is not None and lb < 0.85:
            tips.append(
                f"▲ Load imbalance ({1/lb:.1f}×) — the busiest rank/thread does "
                f"{(1/lb - 1)*100:.0f}% more useful work than average; likely the "
                f"straggler others wait on")
    except Exception:
        pass

    return tips


def diagnose(trace: Trace) -> tuple[str, str]:
    """One-line overall diagnosis + a severity level ("red"/"yellow"/
    "cyan"/"green" -- UI-agnostic; each UI maps this to its own actual
    color), for the headline DIAGNOSIS stat. Uses the same thresholds as
    bottleneck_analysis/top_findings so all three never disagree."""
    meta  = trace.metadata
    backs = meta.backends_used or []

    if any(b in _GPU_CATS for b in backs):
        try:
            from .cct import gpu_starvation
            sv = gpu_starvation(trace)
            if sv["launch_gap_pct"] > 30:
                return "GPU starvation", "red"
            if sv["sync_stall_pct"] > 20:
                return "GPU sync-bound", "yellow"
            if sv["gpu_active_pct"] >= 70:
                return "GPU-bound", "green"
        except Exception:
            pass

    if "mpi" in backs:
        try:
            from .pop_efficiency import useful_time_by_pid, load_balance
            lb = load_balance(useful_time_by_pid(trace))
            if lb is not None and lb < 0.7:
                return "Load imbalance", "red"
            if lb is not None and lb < 0.85:
                return "Mild imbalance", "yellow"
        except Exception:
            pass

    stats = trace.aggregated_stats()
    if stats and stats[0]["pct"] > 40:
        return f"{stats[0]['category']}-bound", "cyan"

    return "Balanced", "green"


def top_findings(trace: Trace) -> list[tuple[str, str, str, str]]:
    """(icon, severity, title, metric) tuples for the "Top findings"
    panel, most-actionable first. `severity` is one of "red"/"yellow"/
    "cyan" (UI-agnostic; see diagnose()). Always tries one cheap,
    always-available fallback (the single hottest function's share of
    total time) so even a plain single-threaded CPU trace shows something
    instead of an empty panel."""
    out: list[tuple[str, str, str, str]] = []
    meta = trace.metadata

    if any(b in _GPU_CATS for b in (meta.backends_used or [])):
        try:
            from .cct import gpu_starvation
            sv = gpu_starvation(trace)
            if sv["launch_gap_pct"] > 30:
                out.append(("!", "red", "Low GPU occupancy",
                            f"{sv['gpu_active_pct']:.0f}% active"))
            if sv["sync_stall_pct"] > 20:
                out.append(("▲", "yellow", "High GPU sync stall",
                            f"{sv['sync_stall_pct']:.0f}%"))
        except Exception:
            pass

    try:
        from .pop_efficiency import useful_time_by_pid, load_balance
        lb = load_balance(useful_time_by_pid(trace))
        if lb is not None and lb < 0.85:
            out.append(("▲", "yellow", "Load imbalance", f"{1/lb:.1f}×"))
    except Exception:
        pass

    stats = trace.aggregated_stats()
    if stats and stats[0]["pct"] > 30:
        out.append((
            "◆", "cyan",
            f"{fmt_kernel_name(stats[0]['name'])[:28]} dominates",
            f"{stats[0]['pct']:.0f}% of total time",
        ))

    return out[:4]


# ── Source correlation ──────────────────────────────────────────────────────

@dataclass
class SourceContext:
    """Plain-data result of find_source_context -- each UI renders this
    in its own idiom (Rich Text for the TUI, a QML ListModel for the GUI)."""
    path: str                    # full path to the source file
    display_name: str            # path's basename, for a compact title
    hotspot_name: str            # the function/span name this context is for
    hot_line: int                # the 1-indexed line the hotspot maps to
    lines: list[tuple[int, str]]  # (line_no, text) for [hot_line-context, hot_line+context]


def find_source_context(trace: Trace, context: int = 3) -> SourceContext | None:
    """Source lines around the top hotspot's file/line tag (the same
    file=/line= tags the Kernels tab's/HotspotsWidget's inline location
    column already uses), or None if no hotspot carries one, the file
    doesn't exist on THIS machine, or the tagged line is out of range. A
    profile collected on a cluster and opened locally routinely hits the
    "file doesn't exist here" case — that's an expected, not exceptional,
    outcome, so callers should show an explanatory message, not blank."""
    for row in trace.aggregated_stats():
        file_tag = line_tag = None
        for s in trace.spans:
            if s.name == row["name"] and s.tags.get("file"):
                file_tag, line_tag = s.tags["file"], s.tags.get("line")
                break
        if not file_tag:
            continue
        try:
            line_no = int(line_tag)
        except (TypeError, ValueError):
            continue
        path = Path(file_tag)
        if not path.is_file():
            continue
        try:
            lines = path.read_text(errors="replace").splitlines()
        except OSError:
            continue
        if not (1 <= line_no <= len(lines)):
            continue

        lo = max(1, line_no - context)
        hi = min(len(lines), line_no + context)
        return SourceContext(
            path=str(path),
            display_name=path.name,
            hotspot_name=row["name"],
            hot_line=line_no,
            lines=[(ln, lines[ln - 1]) for ln in range(lo, hi + 1)],
        )
    return None
