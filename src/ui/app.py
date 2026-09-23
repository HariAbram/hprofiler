"""
Profiler TUI — card-based dashboard layout, inspired by Paraver and VTune.

Tabs (numbered 1..N in the tab strip; Call Tree/Roofline/Source only
appear when the trace actually has the data behind them):
  Overview   — dashboard: diagnosis, headline stats, condensed timeline
               preview, top findings, hot kernels, source correlation
  Timeline   — Paraver-style Gantt: one row per (category, thread), zoomable,
               hover a span to reveal MPI/NCCL communication connector lines
  Kernels    — VTune-style: function table with bar visualization, sortable
  Call Tree  — hierarchical call tree (only when stack traces were captured)
  Roofline   — log-log arithmetic-intensity/TFLOP-s scatter (only when
               hardware-counter or disassembly-estimated kernel metrics exist)
  Source     — annotated disassembly with instruction-mix breakdown
               (only when --disasm was passed or the trace already has it)
  System     — device specs, GPU utilisation, CPU microarch counters
  Profile    — per-backend activity, time breakdown, hotspots, insight tips

Keyboard shortcuts:
  1-7              jump directly to a tab
  Tab / Shift+Tab  cycle tabs
  ← →              scroll timeline
  + / -            zoom timeline
  r                reset timeline zoom
  j/k or ↑ ↓       navigate kernel/call-tree rows
  s                cycle sort column (kernels)
  /                focus name filter (kernels)
  ?                toggle help overlay
  q                quit
"""

from __future__ import annotations
import json
import math
import zlib
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from rich.text import Text
from rich.panel import Panel

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import ScrollableContainer, Horizontal, Vertical, Grid
from textual.events import MouseMove
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widget import Widget
from textual.widgets import (
    TabbedContent, TabPane,
    DataTable, Input, Static, RichLog, Tree,
)

from ..core.trace import Trace
from ..core.events import SpanEvent, Category
from ..analysis.dashboard import (
    fmt_ns as _fmt_ns,
    fmt_bytes as _fmt_bytes,
    fmt_tf as _fmt_tf,
    fmt_kernel_name as _fmt_kernel_name,
    merged_ns as _merged_ns,
    trace_wall_ns as _trace_wall_ns,
    bottleneck_analysis as _bottleneck_analysis,
    diagnose as _diagnose,
    top_findings as _top_findings,
    find_source_context,
    _GPU_CATS,
)
from .braille_canvas import BrailleCanvas

# ── Color palette ─────────────────────────────────────────────────────────────

_CAT_RICH: dict[str, str] = {
    "cpu":     "cyan",
    "cuda":    "red",
    "rocm":    "magenta",
    "opencl":  "yellow",
    "openmp":  "green",
    "mpi":     "dodger_blue2",
    "nccl":    "deep_pink3",
    "memory":  "blue",
    "sync":    "white",
    "jit":     "purple",
    "nvtx":    "orange3",
    "other":   "grey70",
}

def _cat_color(cat: str) -> str:
    return _CAT_RICH.get(cat, "grey70")

# Connector-line style per criticalpath.py edge confidence tier (see its
# module docstring's "Edge confidence" section) -- lets a communication
# line's own color signal how directly it's proven, the same distinction
# `hprofiler critical-path`'s "Path evidence strength" breakdown surfaces
# in the CLI, now visible directly in the live Timeline.
_CONNECTOR_STYLE: dict[str, str] = {
    "certain": "bold bright_white",
    "high":    "bold bright_cyan",
    "medium":  "grey58",
}

# Per-function color palette — 8 primary hues spaced 45° apart, interleaved
# so adjacent palette indices are ~180° apart in hue (maximum contrast).
# idx 0 (red) vs idx 1 (cyan), idx 2 (green) vs idx 3 (magenta), etc.
_SPAN_PALETTE = [
    "bright_red",      # 0°
    "bright_cyan",     # 180°
    "bright_green",    # 120°
    "bright_magenta",  # 300°
    "bright_yellow",   # 60°
    "deep_sky_blue1",  # 210°
    "orange3",         # 30°
    "medium_purple1",  # 270°
    "hot_pink",        # 330°
    "turquoise2",      # 175°
    "chartreuse3",     # 90°
    "gold1",           # 45°
    "cornflower_blue", # 230°
    "light_salmon1",   # 15°
    "spring_green2",   # 150°
    "plum2",           # 285°
]

def _span_color(name: str) -> str:
    """Stable per-function color derived from the function name."""
    return _SPAN_PALETTE[(hash(name) & 0x7FFFFFFF) % len(_SPAN_PALETTE)]

def _bar(frac: float, width: int, filled: str = "█", empty: str = "░") -> str:
    n = max(0, min(width, int(frac * width)))
    return filled * n + empty * (width - n)

def _grad_bar(frac: float, width: int) -> str:
    """Gradient block bar using sub-character resolution (▏▎▍▌▋▊▉█)."""
    blocks = " ▏▎▍▌▋▊▉█"
    total_eighths = int(frac * width * 8)
    full  = total_eighths // 8
    rem   = total_eighths %  8
    s = "█" * full
    if rem and full < width:
        s += blocks[rem]
        s += " " * (width - full - 1)
    else:
        s += " " * (width - full)
    return s[:width]

def _grade(pct: float) -> tuple[str, str]:
    """Return (letter_grade, color) for a percentage 0–100."""
    if pct >= 85: return "A+", "bright_green"
    if pct >= 70: return "A",  "green"
    if pct >= 55: return "B",  "yellow"
    if pct >= 35: return "C",  "orange3"
    if pct >= 15: return "D",  "red"
    return "F",  "bright_red"

def _sparkline(values: list[float], width: int = 12) -> str:
    """Compact sparkline using Braille dots."""
    if not values: return " " * width
    mn, mx = min(values), max(values)
    rng = mx - mn or 1
    spark_chars = "▁▂▃▄▅▆▇█"
    result = ""
    for v in values[-width:]:
        idx = int((v - mn) / rng * 7)
        result += spark_chars[idx]
    return result.ljust(width)



# ── Help overlay ──────────────────────────────────────────────────────────────

_HELP_TEXT = """\
[bold]Keyboard Shortcuts[/bold]

[bold]Global[/bold]
  [yellow]1-7[/yellow]              jump directly to a tab
  [yellow]Tab / Shift+Tab[/yellow]  switch tabs
  [yellow]q[/yellow]                quit
  [yellow]?[/yellow]                this help

[bold]Overview tab[/bold]
  Dashboard: diagnosis + headline stats, a condensed
  timeline preview, top findings, hot kernels, and
  source context for the single hottest function.

[bold]Timeline tab[/bold]
  [yellow]← →[/yellow]             scroll horizontally
  [yellow]↑ ↓[/yellow]             pan up / down (when lanes overflow)
  [yellow]+ -[/yellow]             zoom in / out
  [yellow]r[/yellow]               reset zoom, scroll & pan
  Hover a span for its MPI/NCCL connector lines
  (color: white/cyan/grey = certain/high/medium evidence)

[bold]Kernels tab[/bold]
  [yellow]j/k[/yellow] or [yellow]↑ ↓[/yellow]     navigate rows
  [yellow]s[/yellow]               cycle sort column
  [yellow]/[/yellow]               focus name filter

[bold]Roofline tab[/bold]  (shown when kernel metrics exist)
  Log-log scatter, one dot per kernel:
    [bright_cyan]cyan[/bright_cyan]    = compute-bound
    [bright_magenta]magenta[/bright_magenta] = memory-bound
  The diagonal-then-flat line is its roofline knee.
  For hardware-counter based metrics, run first:
    [yellow]hprofiler roofline --backend <backend> -- ./app[/yellow]
  (disassembly-only estimates are used otherwise)

[bold]Source tab[/bold]  (only shown when [yellow]--disasm[/yellow] is passed)
  [yellow]j/k[/yellow] or [yellow]↑ ↓[/yellow]     select kernel
  Left pane: kernel list with arch + timing
  Right pane: annotated assembly
  Bottom: instruction mix (vec/scl/mem/ctl)

  Colours:
    [bright_green]vec[/bright_green] SIMD/AVX/YMM/ZMM   [cyan]scl[/cyan] scalar ALU
    [yellow]mem[/yellow] load/store        [magenta]ctl[/magenta] branch/call
    [bright_blue]fma[/bright_blue] FMA/multiply-acc  [red]syn[/red] barrier/fence

[bold]Output format[/bold]
  Traces are Perfetto-compatible JSON.
  Open at ui.perfetto.dev
"""


class HelpScreen(ModalScreen):
    BINDINGS = [Binding("escape", "dismiss", "Close"),
                Binding("q",      "dismiss", "Close", show=False),
                Binding("?",      "dismiss", "Close", show=False)]

    DEFAULT_CSS = """
    HelpScreen { align: center middle; }
    #help-box {
        width: 66; height: auto;
        max-height: 90%;
        padding: 1 2;
        background: $surface;
        border: round $accent;
    }
    """

    def compose(self) -> ComposeResult:
        with ScrollableContainer(id="help-box"):
            yield Static(_HELP_TEXT)

    def action_dismiss(self) -> None:
        self.dismiss()


# ── System tab ────────────────────────────────────────────────────────────────

class SystemWidget(Static):
    """Minimal system / device information card."""

    def __init__(self, trace: Trace, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._trace = trace

    def render(self) -> Any:  # noqa: ANN401
        trace = self._trace
        meta  = trace.metadata
        L: list[str] = []

        def _kv(key: str, val: str, key_w: int = 14) -> None:
            L.append(f"  [dim]{key:<{key_w}}[/dim]  {val}")

        def _sep(title: str = "") -> None:
            if title:
                L.append(f"\n  [dim]── {title} {'─' * max(0, 44 - len(title))}[/dim]")
            else:
                L.append(f"  [dim]{'─' * 48}[/dim]")

        # ── Run info ──────────────────────────────────────────────────────
        cmd = f"{meta.command} {' '.join(meta.args[:4])}"
        if len(cmd) > 50: cmd = cmd[:47] + "…"
        _kv("Command", f"[cyan]{cmd}[/cyan]")
        _kv("Host",    f"[dim]{meta.hostname or '—'}[/dim]")
        # _trace_wall_ns, not trace.duration_ns -- the latter is meaningless
        # for a trace reconstructed by load_trace_from_json (see
        # tests/README.md's "A real bug this test suite caught"); every
        # other tab already derives wall time from the spans themselves,
        # so this was the one remaining place that could disagree with them.
        _kv("Duration", f"[yellow]{_fmt_ns(_trace_wall_ns(trace))}[/yellow]")
        _kv("Backend",  "  ".join(
            f"[{_cat_color(b)}]{b}[/{_cat_color(b)}]"
            for b in (meta.backends_used or ["none"])
        ))

        # ── Device specs ──────────────────────────────────────────────────
        for dev in trace.devices:
            bk_col = _cat_color(dev.backend)
            sm_label = "SMs" if dev.backend in ("cuda", "rocm") else "Cores"
            _sep(f"{dev.backend.upper()} — {dev.name}")
            if dev.compute_cap:
                _kv("Compute cap", f"[dim]{dev.compute_cap}[/dim]")
            _kv(sm_label, f"[dim]{dev.sm_count}[/dim]")
            if dev.core_clock_ghz:
                _kv("Clock",  f"[dim]{dev.core_clock_ghz:.2f} GHz[/dim]")
            L.append("")
            if dev.fp16_tflops:
                _kv("FP16 peak",   f"[magenta]{_fmt_tf(dev.fp16_tflops)}/s[/magenta]")
            if dev.fp32_tflops:
                _kv("FP32 peak",   f"[bright_green]{_fmt_tf(dev.fp32_tflops)}/s[/bright_green]")
            if dev.fp64_tflops:
                _kv("FP64 peak",   f"[red]{_fmt_tf(dev.fp64_tflops)}/s[/red]")
            if dev.tensor_tflops > 0:
                _kv("Tensor peak", f"[cyan]{_fmt_tf(dev.tensor_tflops)}/s[/cyan]")
            L.append("")
            if dev.bandwidth_gbs:
                _kv("Bandwidth", f"[bright_cyan]{dev.bandwidth_gbs:.0f} GB/s[/bright_cyan]")
            if dev.vram_gb > 0:
                _kv("VRAM", f"[blue]{dev.vram_gb:.1f} GB[/blue]")
            if dev.ridge_point > 0:
                hint = (
                    "memory-bound"    if dev.ridge_point < 5  else
                    "balanced"        if dev.ridge_point < 30 else
                    "compute-bound"
                )
                _kv("Ridge point",
                    f"[dim]{dev.ridge_point:.0f} FLOPs/byte  ·  {hint}[/dim]")

        # ── GPU utilisation (rocm-smi / nvidia-smi polling) ───────────────
        gpu_util_peak: dict[str, float] = {}
        gpu_mem_peak:  dict[str, float] = {}
        for c in trace.counters:
            if c.name.startswith("gpu_utilization_pct"):
                gpu_util_peak[c.name] = max(gpu_util_peak.get(c.name, 0.0), c.value)
            elif c.name.startswith("gpu_mem_used_bytes"):
                gpu_mem_peak[c.name] = max(gpu_mem_peak.get(c.name, 0.0), c.value)

        if gpu_util_peak or gpu_mem_peak:
            _sep("GPU UTILISATION (peak, 1 s poll)")
            all_gpu_ids = sorted(
                set(k.replace("gpu_utilization_pct", "").replace("gpu_mem_used_bytes", "").strip("[]")
                    for k in list(gpu_util_peak) + list(gpu_mem_peak))
            )
            for gid in all_gpu_ids:
                util_key = f"gpu_utilization_pct[{gid}]"
                mem_key  = f"gpu_mem_used_bytes[{gid}]"
                util_val = gpu_util_peak.get(util_key, -1.0)
                mem_val  = gpu_mem_peak.get(mem_key, -1.0)
                util_str = (
                    f"[{'bright_green' if util_val >= 80 else ('yellow' if util_val >= 20 else 'dim')}]"
                    f"compute {util_val:.0f}%"
                    f"[/{'bright_green' if util_val >= 80 else ('yellow' if util_val >= 20 else 'dim')}]"
                ) if util_val >= 0 else ""
                mem_str = f"[blue]{_fmt_bytes(mem_val)} VRAM[/blue]" if mem_val >= 0 else ""
                parts = "  ".join(p for p in (util_str, mem_str) if p)
                _kv(gid, parts)

        # ── CPU microarch ─────────────────────────────────────────────────
        ctrs: dict[str, float] = {c.name: c.value for c in trace.counters}
        ipc   = ctrs.get("ipc", 0.0)
        cmiss = ctrs.get("cache_miss_pct", -1.0)
        bmiss = ctrs.get("branch_miss_pct", -1.0)
        rss   = ctrs.get("process_max_rss_bytes", 0.0)
        if ipc > 0 or cmiss >= 0 or rss > 0:
            _sep("CPU")
            if ipc > 0:
                col = "bright_green" if ipc >= 2 else ("yellow" if ipc >= 1 else "red")
                _kv("IPC", f"[{col}]{ipc:.2f}[/{col}]")
            if cmiss >= 0:
                col = "green" if cmiss < 5 else ("yellow" if cmiss < 20 else "red")
                _kv("LLC miss rate", f"[{col}]{cmiss:.1f}%[/{col}]")
            if bmiss >= 0:
                col = "green" if bmiss < 1 else ("yellow" if bmiss < 5 else "red")
                _kv("Branch miss", f"[{col}]{bmiss:.1f}%[/{col}]")
            if rss > 0:
                L.append("")
                _kv("Peak RSS", f"[yellow]{_fmt_bytes(rss)}[/yellow]")

        L.append("")
        return "\n".join(L)


# ── Profile tab ───────────────────────────────────────────────────────────────

class ProfileWidget(Static):
    """Minimal profiling results: activity, backend breakdown, hotspots, insight."""

    def __init__(self, trace: Trace, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._trace = trace

    def render(self) -> Any:  # noqa: ANN401
        trace  = self._trace
        spans  = trace.spans
        wall_ns = _trace_wall_ns(trace)
        L: list[str] = []

        def _sep(title: str = "") -> None:
            if title:
                L.append(f"\n  [dim]── {title} {'─' * max(0, 44 - len(title))}[/dim]\n")
            else:
                L.append("")

        # ── GPU activity ──────────────────────────────────────────────────
        for cat_val, label in (("cuda", "CUDA"), ("rocm", "ROCm")):
            kspans = [s for s in spans
                      if s.category.value == cat_val and s.tags.get("type") == "kernel"]
            if not kspans:
                continue
            kern_ns  = _merged_ns(kspans)
            kern_acc = sum(s.duration_ns for s in kspans)
            sync_ns  = _merged_ns([s for s in spans if s.category.value == "sync"])
            pct      = 100.0 * kern_ns / wall_ns
            sync_pct = 100.0 * sync_ns / wall_ns
            avg_ns   = kern_acc / len(kspans)
            eff      = pct / (pct + sync_pct) * 100 if (pct + sync_pct) > 0 else 0
            color    = _cat_color(cat_val)
            grade, gc = _grade(pct)

            _sep(f"{label} ACTIVITY")
            L.append(
                f"  [bold]Kernel active[/bold]   "
                f"[yellow]{pct:.1f}%[/yellow] of wall time"
                f"  [{gc}]({grade})[/{gc}]"
            )
            L.append(
                f"  [dim]               "
                f"  {_fmt_ns(kern_ns)} active (merged)"
                f"  ·  {_fmt_ns(kern_acc)} accumulated[/dim]"
            )
            L.append(
                f"  [dim]Sync overhead    {sync_pct:.1f}% of wall time[/dim]"
            )
            L.append(
                f"  [dim]GPU efficiency   {eff:.0f}%  (kernel / (kernel + sync))[/dim]"
            )
            L.append("")
            L.append(
                f"  [dim]{len(kspans)} kernel launches"
                f"  ·  {_fmt_ns(avg_ns)} average"
                f"  ·  {_fmt_ns(kern_acc)} total[/dim]"
            )

        # ── Time breakdown ────────────────────────────────────────────────
        by_cat: dict[str, dict] = defaultdict(lambda: {"ns": 0, "n": 0})
        for s in spans:
            by_cat[s.category.value]["ns"] += s.duration_ns
            by_cat[s.category.value]["n"]  += 1
        grand = sum(v["ns"] for v in by_cat.values()) or 1

        _sep("TIME BREAKDOWN")
        BAR = 16
        for cat, info in sorted(by_cat.items(), key=lambda kv: -kv[1]["ns"]):
            frac  = info["ns"] / grand
            color = _cat_color(cat)
            bar   = _bar(frac, BAR)
            L.append(
                f"  [{color}]{cat:<8}[/{color}]"
                f"  [{color}]{bar}[/{color}]"
                f"  [yellow]{frac*100:5.1f}%[/yellow]"
                f"  [dim]{_fmt_ns(info['ns']):>9}[/dim]"
            )

        # ── Hotspots ──────────────────────────────────────────────────────
        # Single row per hotspot with fixed-width columns.
        # Name is not padded with markup (markup inflates Python length);
        # instead we pad a plain string first, then wrap in markup.
        stats = trace.aggregated_stats()
        if stats:
            _sep("HOTSPOTS")
            total_all = sum(r["total_ns"] for r in stats) or 1
            NAME_W = 36
            CAT_W  = 7
            # Column header — plain strings only, no markup
            hdr = (
                f"  {'#':>2}  "
                f"{'FUNCTION':<{NAME_W}}  "
                f"{'CATEGORY':<{CAT_W}}  "
                f"{'SHARE':>6}  "
                f"{'TOTAL':>9}  "
                f"{'AVG':>9}  "
                f"CALLS"
            )
            L.append(f"  [dim]{hdr.strip()}[/dim]")
            L.append(f"  [dim]{'─' * (len(hdr) - 2)}[/dim]")
            for i, row in enumerate(stats[:12]):
                color   = _cat_color(row["category"])
                avg_ns  = row["total_ns"] / max(row["count"], 1)
                pct     = row["total_ns"] / total_all * 100
                # Pad plain name first, then apply markup
                name_plain = row["name"][:NAME_W]
                name_pad   = f"{name_plain:<{NAME_W}}"
                cat_pad    = f"{row['category']:<{CAT_W}}"
                L.append(
                    f"  [dim]{i+1:>2}[/dim]  "
                    f"[bold {color}]{name_pad}[/bold {color}]  "
                    f"[{color}]{cat_pad}[/{color}]  "
                    f"[yellow]{pct:6.1f}%[/yellow]  "
                    f"[white]{_fmt_ns(row['total_ns']):>9}[/white]  "
                    f"[dim]{_fmt_ns(avg_ns):>9}[/dim]  "
                    f"[dim]{row['count']}[/dim]"
                )

        # ── Insight ───────────────────────────────────────────────────────
        # _bottleneck_analysis is defined further down in this module (see
        # "Dashboard analysis helpers") and shared with DashboardWidget's
        # "Top findings" panel -- no import needed, it's a module global by
        # the time any widget actually renders.
        try:
            ctrs_d = {c.name: c.value for c in trace.counters}
            ctr_sub: dict[str, float] = {
                k: ctrs_d[k]
                for k in ("ipc", "cache_miss_pct", "branch_miss_pct")
                if k in ctrs_d
            }
            tips = _bottleneck_analysis(trace, ctr_sub)
            if tips:
                _sep("INSIGHT")
                for tip in tips:
                    icon = tip[:2]
                    body = tip[2:].strip()
                    # Hard-wrap at 62 chars
                    words, lines_w, cur = body.split(), [], ""
                    for w in words:
                        if len(cur) + len(w) + 1 > 60:
                            lines_w.append(cur); cur = w
                        else:
                            cur = (cur + " " + w).strip()
                    if cur: lines_w.append(cur)
                    for j, wl in enumerate(lines_w):
                        pfx = f"  {icon} " if j == 0 else "      "
                        L.append(f"{pfx}[dim]{wl}[/dim]")
                    L.append("")
        except Exception:
            pass

        L.append("")
        return "\n".join(L)


# ── Dashboard analysis helpers ──────────────────────────────────────────────
#
# _bottleneck_analysis/_diagnose/_top_findings/_GPU_CATS live in
# analysis/dashboard.py now (shared with the Qt/QML GUI) -- imported at
# the top of this file under their original names for every existing
# call site here to keep working unchanged. _diagnose's color is now a
# UI-agnostic severity family ("red"/"yellow"/"green"/"cyan"); _TUI_SHADE
# maps "green" to the brighter "bright_green" this tab always used for
# its one happy-path case, so the Rich-rendered result is unchanged.
_TUI_SHADE = {"green": "bright_green"}


def _mini_row(spans: list, width: int, view_start: int, view_dur: float, color: str) -> Text:
    """Coarse fixed-width density row for the Dashboard's timeline preview:
    a simplified, loop-based cousin of TimelineWidget._density_row. Fine at
    preview width/span-count — the numpy-vectorised path in TimelineWidget
    exists specifically for that widget's full interactive zoom/pan over
    potentially far more spans."""
    row = Text()
    if view_dur <= 0 or width <= 0:
        return row
    cov = [0.0] * width
    scale = width / view_dur
    for s in spans:
        if s.duration_ns <= 0:
            continue
        x0 = max(0.0, min(float(width), (s.start_ns - view_start) * scale))
        x1 = max(0.0, min(float(width), (s.start_ns + s.duration_ns - view_start) * scale))
        if x1 <= x0:
            ix = min(int(x0), width - 1)
            if ix >= 0:
                cov[ix] = max(cov[ix], 0.2)
            continue
        i0, i1 = int(x0), min(int(x1), width - 1)
        for i in range(i0, i1 + 1):
            cov[i] = 1.0
    for c in cov:
        row.append("█" if c > 0.05 else " ", style=color if c > 0.05 else "")
    return row


def _source_snippet(trace: Trace, context: int = 3) -> Text | None:
    """Rich-formatted wrapper around analysis/dashboard.find_source_context
    (the actual file/line lookup logic, shared with the Qt/QML GUI, which
    renders the same SourceContext through its own QML component instead)."""
    ctx = find_source_context(trace, context=context)
    if ctx is None:
        return None
    gw = len(str(ctx.lines[-1][0])) if ctx.lines else 1
    out = Text()
    out.append(f"{ctx.display_name}  ·  {ctx.hotspot_name[:24]}\n", style="dim cyan")
    for ln, text in ctx.lines:
        hot = ln == ctx.hot_line
        out.append(f"{'▶' if hot else ' '}{ln:>{gw}} │ ", style="bold yellow" if hot else "dim")
        out.append(f"{text}\n", style="bold" if hot else "dim")
    return out


# ── Overview tab ─────────────────────────────────────────────────────────────

class DashboardWidget(Widget):
    """
    Landing-page dashboard: five headline stat cards, a condensed
    multi-category timeline preview, actionable findings, a hot-kernels
    table, and (when file/line tags plus the source file itself are
    available) correlated source context for the top hotspot.
    """

    DEFAULT_CSS = """
    DashboardWidget { height: 1fr; }
    #dash-stats { height: 5; margin: 0 0 1 0; }
    .stat-card {
        width: 1fr; height: 100%;
        border: round $primary;
        padding: 0 1;
        margin: 0 1 0 0;
    }
    .stat-card:last-of-type { margin-right: 0; }
    #dash-grid { height: 1fr; grid-size: 2 2; grid-gutter: 1; }
    .dash-panel { border: round $primary; padding: 0 1; height: 1fr; }
    """

    def __init__(self, trace: Trace, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.trace = trace

    def compose(self) -> ComposeResult:
        with Horizontal(id="dash-stats"):
            yield Static(id="stat-diag", classes="stat-card")
            yield Static(id="stat-wall", classes="stat-card")
            yield Static(id="stat-gpu",  classes="stat-card")
            yield Static(id="stat-wait", classes="stat-card")
            yield Static(id="stat-mem",  classes="stat-card")
        with Grid(id="dash-grid"):
            yield Static(id="dash-timeline", classes="dash-panel")
            yield Static(id="dash-findings", classes="dash-panel")
            yield DataTable(id="dash-kernels", cursor_type="row", classes="dash-panel")
            yield Static(id="dash-source", classes="dash-panel")

    def on_mount(self) -> None:
        self._populate()

    def _populate(self) -> None:  # noqa: C901 — one straight-line pass over 5+4 panels
        trace   = self.trace
        wall_ns = _trace_wall_ns(trace)
        meta    = trace.metadata

        diag_label, diag_severity = _diagnose(trace)
        diag_color = _TUI_SHADE.get(diag_severity, diag_severity)
        self.query_one("#stat-diag", Static).update(Text.from_markup(
            f"[dim]DIAGNOSIS[/dim]\n[bold {diag_color}]{diag_label}[/bold {diag_color}]"))
        self.query_one("#stat-wall", Static).update(Text.from_markup(
            f"[dim]WALL TIME[/dim]\n[bold]{_fmt_ns(wall_ns)}[/bold]"))

        gpu_pct = None
        if any(b in _GPU_CATS for b in (meta.backends_used or [])):
            try:
                from ..analysis.cct import gpu_starvation
                gpu_pct = gpu_starvation(trace)["gpu_active_pct"]
            except Exception:
                gpu_pct = None
        if gpu_pct is not None:
            _, gcol = _grade(gpu_pct)
            gpu_text = f"[dim]GPU ACTIVE[/dim]\n[bold {gcol}]{gpu_pct:.0f}%[/bold {gcol}]"
        else:
            gpu_text = "[dim]GPU ACTIVE[/dim]\n[dim]n/a[/dim]"
        self.query_one("#stat-gpu", Static).update(Text.from_markup(gpu_text))

        mpi_present = any(s.category.value == "mpi" for s in trace.spans)
        if mpi_present:
            wait_ns    = _merged_ns([s for s in trace.spans if s.category.value == "mpi"])
            wait_label = "MPI WAIT"
        else:
            wait_ns    = _merged_ns([s for s in trace.spans if s.category.value == "sync"])
            wait_label = "SYNC WAIT"
        wait_pct = 100.0 * wait_ns / wall_ns
        wcol = "red" if wait_pct >= 40 else ("yellow" if wait_pct >= 20 else "bright_green")
        self.query_one("#stat-wait", Static).update(Text.from_markup(
            f"[dim]{wait_label}[/dim]\n[bold {wcol}]{wait_pct:.0f}%[/bold {wcol}]"))

        ctrs = {c.name: c.value for c in trace.counters}
        rss  = ctrs.get("process_max_rss_bytes", 0.0)
        if rss > 0:
            mem_str = _fmt_bytes(rss)
        else:
            vram_total = sum(v for k, v in ctrs.items() if k.startswith("gpu_mem_used_bytes"))
            mem_str = _fmt_bytes(vram_total) if vram_total > 0 else "n/a"
        self.query_one("#stat-mem", Static).update(Text.from_markup(
            f"[dim]PEAK MEMORY[/dim]\n[bold]{mem_str}[/bold]"))

        # ── Execution timeline preview: top-3 categories by accumulated time ──
        tl_panel = self.query_one("#dash-timeline", Static)
        tl_panel.border_title = "Execution timeline"
        by_cat: dict[str, list] = defaultdict(list)
        for s in trace.spans:
            if s.duration_ns > 0:
                by_cat[s.category.value].append(s)
        top_cats = sorted(by_cat.items(), key=lambda kv: -sum(s.duration_ns for s in kv[1]))[:3]
        timed = [s for s in trace.spans if s.duration_ns > 0]
        view_start = min((s.start_ns for s in timed), default=0)
        view_end   = max((s.end_ns for s in timed), default=1)
        view_dur   = max(view_end - view_start, 1)
        width = max(10, self.size.width // 2 - 12) if self.size.width else 40

        body = Text()
        if not top_cats:
            body.append("No timed spans recorded.", style="dim")
        for cat, cspans in top_cats:
            color = _cat_color(cat)
            body.append(f"{cat:<8}", style=f"bold {color}")
            body.append_text(_mini_row(cspans, width, view_start, view_dur, color))
            body.append("\n")
        if top_cats:
            body.append("\n")
            body.append_text(Text.from_markup(
                "  ".join(f"[{_cat_color(c)}]■[/{_cat_color(c)}] {c}" for c, _ in top_cats)))
        tl_panel.update(body)

        # ── Top findings ──────────────────────────────────────────────────
        f_panel = self.query_one("#dash-findings", Static)
        f_panel.border_title = "Top findings"
        findings = _top_findings(trace)
        if findings:
            ftext = Text()
            for icon, color, title, metric in findings:
                ftext.append(f"{icon} ", style=f"bold {color}")
                ftext.append(title, style="bold")
                ftext.append(f"\n   {metric}\n", style=color)
            f_panel.update(ftext)
        else:
            f_panel.update(Text("No actionable findings — looks balanced.", style="dim"))

        # ── Hot kernels ───────────────────────────────────────────────────
        dt = self.query_one("#dash-kernels", DataTable)
        dt.border_title = "Hot kernels"
        dt.add_columns("Kernel", "Calls", "Total", "Share")
        stats     = trace.aggregated_stats()
        total_all = sum(r["total_ns"] for r in stats) or 1
        for row in stats[:8]:
            color = _cat_color(row["category"])
            dt.add_row(
                Text(_fmt_kernel_name(row["name"])[:30], style=f"bold {color}"),
                str(row["count"]),
                _fmt_ns(row["total_ns"]),
                f"{row['total_ns']/total_all*100:.1f}%",
            )

        # ── Source correlation ───────────────────────────────────────────
        s_panel = self.query_one("#dash-source", Static)
        s_panel.border_title = "Source correlation"
        snippet = _source_snippet(trace)
        s_panel.update(snippet if snippet is not None else Text(
            "No source correlation available — either the hottest function "
            "carries no file/line tag, or its source file isn't present on "
            "this machine (expected when viewing a trace collected "
            "elsewhere, e.g. on a cluster).",
            style="dim"))


# ── Timeline widget ───────────────────────────────────────────────────────────

class TimelineWidget(Widget):
    """
    Paraver-style Gantt timeline.

    Lanes are grouped by thread (sequential T1, T2, … not raw TIDs).
    All categories for the same thread appear together.
    A blank spacer row separates each lane for readability.
    ↑↓ scrolls vertically when there are more lanes than screen height.
    """

    DEFAULT_CSS = """
    TimelineWidget {
        height: 1fr;
        background: $surface;
        border: round $primary;
        padding: 0 1;
    }
    TimelineWidget:focus {
        border: round $accent;
    }
    """

    can_focus = True

    BINDINGS = [
        Binding("right", "scroll_right", "→"),
        Binding("left",  "scroll_left",  "←"),
        Binding("up",    "scroll_up",    "↑"),
        Binding("down",  "scroll_down",  "↓"),
        Binding("equal", "zoom_in",  "+"),
        Binding("plus",  "zoom_in",  "+", show=False),
        Binding("minus", "zoom_out", "-"),
        Binding("r",     "reset",    "Reset"),
    ]

    # Named view_x/view_y to avoid collision with Textual's built-in scroll_x/scroll_y
    # (Widget.scroll_x/y are managed by Textual's viewport system and get reset by layout).
    view_x: reactive[int]   = reactive(0)
    view_y: reactive[int]   = reactive(0)
    zoom:   reactive[float] = reactive(1.0)
    _hover: reactive[str]   = reactive("")   # hover info shown in status bar
    # id() of the span currently under the cursor, or 0 for none -- gates
    # which connector lines render() actually draws (see _connectors):
    # drawing every MPI/NCCL edge at once on a busy trace is a hairball,
    # so only the hovered span's own edges are shown, on demand.
    _hover_span_id: reactive[int] = reactive(0)

    # label column: "omp  T12  (120)" = up to 17 chars
    _LABEL_W = 17
    _COL_W   = 18

    _CAT_ABBREV: dict[str, str] = {
        "openmp": "omp",  "opencl": "ocl", "memory": "mem",
        "cuda":   "cuda", "rocm":   "rocm","cpu":    "cpu",
        "sync":   "sync", "jit":    "jit", "nvtx":   "nvtx", "other":  "?",
    }

    def __init__(self, trace: Trace, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.border_title = "Timeline view"
        self.trace        = trace
        self._lanes       = trace.lanes()
        self._lane_counts = {k: len(v) for k, v in self._lanes.items()}

        # Assign stable sequential numbers T1, T2, … to each unique TID
        all_tids = sorted({
            int(ln.split("/thread-")[1])
            for ln in self._lanes
            if "/thread-" in ln
        })
        self._tid_seq: dict[int, int] = {tid: i + 1 for i, tid in enumerate(all_tids)}

        # Assign stable sequential numbers S0, S1, … to each unique CUDA stream ID
        all_sids = sorted({
            int(ln.split("/stream-")[1])
            for ln in self._lanes
            if "/stream-" in ln
        })
        self._stream_seq: dict[int, int] = {sid: sid for sid in all_sids}

        # MPI lane -> that rank's own rank= tag value, when known -- "mpi
        # rank2" reads far more meaningfully than a generic "mpi T3"
        # sequential thread number, since rank is what a user actually
        # thinks in terms of. Every span on one OS thread within one MPI
        # process reports the same rank, so the first one found suffices.
        # Falls back to the generic thread-sequence label (via
        # _lane_label's existing logic) for any mpi lane where no span
        # happens to carry a rank= tag (e.g. collectives use rank= too, but
        # a lane with zero mpi spans somehow wouldn't be a /mpi lane at
        # all, so this is only a defensive fallback, not an expected case).
        self._lane_rank: dict[str, str] = {}
        for lane_name, lane_spans in self._lanes.items():
            if not lane_name.startswith("mpi/"):
                continue
            for s in lane_spans:
                rank = s.tags.get("rank")
                if rank is not None:
                    self._lane_rank[lane_name] = rank
                    break

        # Build span_id → name lookup for hover parent annotation.
        self._sid_name: dict[str, str] = {
            s.span_id: s.name
            for s in trace.spans
            if s.span_id
        }

        # Sort: group by thread first, then by category within the thread.
        # CUDA stream lanes are sorted together by stream ID.
        def _sort_key(name: str) -> tuple[int, str]:
            parts = name.split("/", 1)
            cat   = parts[0]
            if len(parts) > 1:
                suffix = parts[1]
                if suffix.startswith("thread-"):
                    try:
                        tid = int(suffix.removeprefix("thread-"))
                        return (self._tid_seq.get(tid, 9999), cat)
                    except ValueError:
                        pass
                elif suffix.startswith("stream-"):
                    try:
                        sid = int(suffix.removeprefix("stream-"))
                        return (10000 + sid, cat)
                    except ValueError:
                        pass
            return (0, cat)

        self._lane_names = sorted(self._lanes.keys(), key=_sort_key)

        all_spans = trace.spans
        # Anchor the visible window to timed spans only; CPU sample spans
        # (duration_ns=0) may use a different clock base (e.g. CLOCK_BOOTTIME
        # vs CLOCK_MONOTONIC on suspended machines) and would bloat the window.
        timed_spans = [s for s in all_spans if s.duration_ns > 0]
        anchor = timed_spans if timed_spans else all_spans
        if anchor:
            self._view_start = min(s.start_ns for s in anchor)
            self._view_end   = max(s.end_ns   for s in anchor)
        else:
            self._view_start = trace.metadata.start_time_ns
            self._view_end   = trace.metadata.end_time_ns or self._view_start + 1
        self._trace_dur = max(self._view_end - self._view_start, 1)

        # Per-function color map — stable hash-based assignment: the same
        # function name gets the same color across DIFFERENT traces/runs,
        # not just within one. The previous scheme assigned colors by
        # first-seen (encounter) order, which depends on arbitrary thread-
        # scheduling order and so gave the same function a different color
        # from one run to the next -- bad for building muscle memory
        # across repeated profiling sessions of the same program.
        #
        # A function's PREFERRED palette slot is crc32(name) % palette size
        # (crc32, not Python's builtin hash() -- the latter is randomly
        # salted per-process for strings by default, defeating the whole
        # point of a stable assignment). If two distinct names in THIS
        # trace prefer the same slot, the one that sorts later probes
        # forward (open addressing) to the next free slot -- so up to
        # len(_SPAN_PALETTE) distinct functions still always get visually
        # distinct colors within one trace, matching the old guarantee;
        # a name only shifts off its preferred slot when the palette is
        # genuinely crowded (more distinct functions than colors), and
        # which names collide (not WHETHER any do) is itself a
        # deterministic function of the name set, not of render order.
        distinct_names = sorted({s.name for s in (all_spans if all_spans else [])})
        assigned: dict[str, int] = {}
        taken: set[int] = set()
        for name in distinct_names:
            idx = zlib.crc32(name.encode("utf-8")) % len(_SPAN_PALETTE)
            while idx in taken and len(taken) < len(_SPAN_PALETTE):
                idx = (idx + 1) % len(_SPAN_PALETTE)
            assigned[name] = idx
            taken.add(idx)
        self._func_colors: dict[str, str] = {
            name: _SPAN_PALETTE[idx] for name, idx in assigned.items()
        }
        # Integer palette index per function — used by the numpy render path.
        self._func_color_idx: dict[str, int] = assigned

        # ── Spatial index + numpy column arrays ──────────────────────────────
        # Sort each lane once by start_ns; keep numpy arrays of start, end, and
        # palette-index so _density_row can run loop-free over visible spans.
        self._sorted_spans: dict[str, list] = {
            lane: sorted(spans_list, key=lambda s: s.start_ns)
            for lane, spans_list in self._lanes.items()
        }
        cidx_map = self._func_color_idx   # name → int palette index
        self._starts_arr: dict[str, np.ndarray] = {}
        self._ends_arr:   dict[str, np.ndarray] = {}
        self._cidx_arr:   dict[str, np.ndarray] = {}
        self._max_dur:    dict[str, int]         = {}
        for lane, slist in self._sorted_spans.items():
            if slist:
                self._starts_arr[lane] = np.array(
                    [s.start_ns for s in slist], dtype=np.int64)
                self._ends_arr[lane]   = np.array(
                    [s.end_ns   for s in slist], dtype=np.int64)
                self._cidx_arr[lane]   = np.array(
                    [cidx_map.get(s.name, 0) for s in slist], dtype=np.int32)
                self._max_dur[lane]    = int(
                    self._ends_arr[lane].max() - self._starts_arr[lane].min())
            else:
                self._starts_arr[lane] = np.empty(0, dtype=np.int64)
                self._ends_arr[lane]   = np.empty(0, dtype=np.int64)
                self._cidx_arr[lane]   = np.empty(0, dtype=np.int32)
                self._max_dur[lane]    = 0

        # ── Cross-rank communication connectors (MPI/NCCL) ────────────────
        # Reuses criticalpath.py's dependency-graph edges directly (resolved
        # wildcard matching, commid=-scoped rendezvous, confidence tiers --
        # see its module docstring) rather than re-deriving send/recv or
        # collective-participant matching here. Computed once at
        # construction, same as every other precomputed structure above --
        # TimelineWidget is built once per loaded/completed trace (see
        # ProfilerApp.compose), not live-refreshed as new events stream in.
        #
        # Drawing EVERY connector simultaneously on a busy trace produces a
        # hairball of overlapping lines -- render() only actually draws the
        # ones touching the currently-hovered span (see _hover_span_id),
        # so pred_span_id/succ_span_id are kept alongside the lane/time
        # data specifically to make that O(1)-per-connector filter possible
        # without re-walking the dependency graph on every mouse move.
        # (pred_lane, pred_mid_ns, succ_lane, succ_mid_ns, confidence, pred_span_id, succ_span_id)
        self._connectors: list[tuple[str, float, str, float, str, int, int]] = []
        try:
            from ..analysis import criticalpath as _cp
            cp_spans, cp_preds = _cp.build_dependency_graph(trace)
            span_lane: dict[int, str] = {
                id(s): lane for lane, spans_list in self._lanes.items() for s in spans_list
            }
            for succ_idx, edges in cp_preds.items():
                succ = cp_spans[succ_idx]
                if succ.category.value not in ("mpi", "nccl"):
                    continue
                succ_lane = span_lane.get(id(succ))
                if succ_lane is None:
                    continue
                succ_mid = (succ.start_ns + succ.end_ns) / 2.0
                for pred_idx, kind, confidence in edges:
                    if kind not in ("p2p", "arrival"):
                        continue
                    pred = cp_spans[pred_idx]
                    pred_lane = span_lane.get(id(pred))
                    # Same-lane edges need no cross-lane connector -- the
                    # spans are already visually adjacent in one row.
                    if pred_lane is None or pred_lane == succ_lane:
                        continue
                    pred_mid = (pred.start_ns + pred.end_ns) / 2.0
                    self._connectors.append(
                        (pred_lane, pred_mid, succ_lane, succ_mid, confidence, id(pred), id(succ)))
        except Exception:
            # Connector lines are a display enhancement layered on an
            # otherwise-independent, already-working Timeline -- a failure
            # here (e.g. an unusual trace shape criticalpath.py doesn't
            # handle) must not take down the whole tab. Falls back to no
            # connectors, same as before this feature existed.
            self._connectors = []

        # span id -> how many connectors touch it, precomputed once so the
        # hover-text "(N links)" hint (on_mouse_move) is an O(1) lookup
        # instead of a linear scan of self._connectors on every pixel of
        # mouse movement.
        self._connector_count: dict[int, int] = defaultdict(int)
        for _pl, _pn, _sl, _sn, _conf, pred_sid, succ_sid in self._connectors:
            self._connector_count[pred_sid] += 1
            self._connector_count[succ_sid] += 1

    def _lane_label(self, lane_name: str) -> str:
        parts = lane_name.split("/", 1)
        cat   = parts[0]
        abbr  = self._CAT_ABBREV.get(cat, cat[:4])
        count = self._lane_counts.get(lane_name, 0)

        if len(parts) > 1:
            suffix = parts[1]
            if suffix.startswith("thread-"):
                rank = self._lane_rank.get(lane_name)
                if rank is not None:
                    # "mpi rank2" reads far more meaningfully than a
                    # generic sequential "mpi T3" -- rank is what a user
                    # actually thinks in terms of for an MPI trace.
                    core = f"{abbr:<5} rank{rank}"
                else:
                    try:
                        tid = int(suffix.removeprefix("thread-"))
                        seq = self._tid_seq.get(tid, tid)
                        core = f"{abbr:<5} T{seq}"
                    except ValueError:
                        core = abbr
            elif suffix.startswith("stream-"):
                try:
                    sid = int(suffix.removeprefix("stream-"))
                    core = f"{abbr:<5} S{sid}"   # e.g. "cuda  S0", "cuda  S1"
                except ValueError:
                    core = abbr
            else:
                core = abbr
        else:
            core = abbr

        # Reserve 4 chars for "  (N)" count suffix; truncate core to fit _LABEL_W.
        count_str  = f"({count})"
        # core + "  " + count_str must fit in _LABEL_W
        max_core   = self._LABEL_W - 2 - len(count_str)
        core_part  = core[:max_core] if len(core) > max_core else f"{core:<{max_core}}"
        label      = f"{core_part}  {count_str}"
        return f"{label:<{self._LABEL_W}}"

    def _density_row(self, lane_name: str, width: int,
                     _unused: str) -> tuple[list[str], list[str], float]:
        """
        Return (per-column chars, per-column styles, visible-window
        utilisation %) -- raw, not yet RLE-encoded into a Text; pass to
        _row_from_columns (optionally after overlaying connector-line
        characters onto specific columns) to get the final Text row.

        Fully vectorised — no Python loop over spans:
          1. Spatial index  : np.searchsorted clips to only the visible spans O(log n)
          2. Numpy broadcast: pixel positions computed for all spans at once
          3. Diff + cumsum  : interior pixel activity accumulated without loops
          4. searchsorted   : dominant-function color assigned to pixels in O(width)

        Renders each pixel as:
          █  solid block colored by the function dominating that column
             blank (unstyled space) for idle (no span coverage) -- quieter
             than a visible dot, which read as noise on sparse traces
        """
        visible_ns = self._trace_dur / self.zoom
        offset_ns  = self._trace_dur * self.view_x / (width * self.zoom)
        vis_start  = self._view_start + offset_ns
        vis_end    = vis_start + visible_ns

        # ── 1. Spatial index ──────────────────────────────────────────────
        starts  = self._starts_arr[lane_name]
        max_dur = self._max_dur[lane_name]

        lo = int(np.searchsorted(starts, vis_start - max_dur, side="left"))
        hi = int(np.searchsorted(starts, vis_end,             side="right"))

        if lo >= hi:
            return [" "] * width, [""] * width, 0.0

        s_ns = self._starts_arr[lane_name][lo:hi].astype(np.float64)
        e_ns = self._ends_arr[lane_name][lo:hi].astype(np.float64)
        c_np = self._cidx_arr[lane_name][lo:hi]          # int32 palette indices

        # ── 2. Pixel positions — all spans at once ────────────────────────
        scale = width * self.zoom / self._trace_dur
        cx0 = np.clip((s_ns - self._view_start) * scale - self.view_x,
                      0.0, float(width)).astype(np.float32)
        cx1 = np.clip((e_ns - self._view_start) * scale - self.view_x,
                      0.0, float(width)).astype(np.float32)

        vis = cx1 > cx0 + 1e-6            # drop zero-width spans
        cx0 = cx0[vis]; cx1 = cx1[vis]; c_np = c_np[vis]
        if len(cx0) == 0:
            return [" "] * width, [""] * width, 0.0

        ix0 = cx0.astype(np.int32)
        ix1 = np.minimum(cx1.astype(np.int32), width - 1)

        # ── 3. Activity accumulation via scatter-add + diff/cumsum ────────
        activity = np.zeros(width, dtype=np.float32)

        # Left-boundary overlap for every span (= total overlap for single-pixel spans)
        left_ov = np.minimum(cx1, (ix0 + 1).astype(np.float32)) - cx0
        np.add.at(activity, ix0, left_ov)

        multi = ix1 > ix0
        if multi.any():
            ix0m, ix1m = ix0[multi], ix1[multi]
            # Right boundary
            np.add.at(activity, ix1m, cx1[multi] - ix1m.astype(np.float32))
            # Interior via diff array: cumsum adds 1.0 to pixels ix0m+1 .. ix1m-1
            diff = np.zeros(width + 1, dtype=np.float32)
            np.add.at(diff, ix0m + 1,  1.0)
            np.add.at(diff, ix1m,     -1.0)
            activity += np.cumsum(diff[:width])

        # KNOWN LIMITATION: this accumulates per-pixel coverage as a SUM of
        # each span's overlap fraction (scatter-add above), then clips to
        # [0,1] -- not a true interval union. When two+ spans overlap the
        # SAME sub-pixel time window on the same lane (e.g. OMPT's nested
        # parallel-region + work/barrier spans on one thread, which share
        # one `openmp/thread-TID` lane per Trace.lanes()) their fractions
        # sum first and are clipped after, so a genuinely-partially-covered
        # pixel can read as more covered than it truly is (only a sum that
        # reaches >=1.0 gets corrected by the clip; a sum that overlaps but
        # stays <1.0 does not). A fully correct fix needs real per-pixel
        # interval-union math, which is a materially bigger rewrite of this
        # vectorized routine (documented as a deliberately hand-tuned,
        # performance-critical path -- ~8ms at 250k spans) than the size of
        # this bug warrants; not attempted here to avoid risking a
        # correctness or performance regression in exchange for a display
        # metric that is only skewed under same-lane overlap, never crashes
        # or produces a wildly wrong value (bounded to [0,1] either way).
        np.clip(activity, 0.0, 1.0, out=activity)
        util_pct = float(activity.sum()) / width * 100.0

        # Minimum visibility: sub-pixel spans contribute << 0.05 to activity
        # and render as invisible dots.  Boost any pixel actually touched by a
        # span to just above the IDLE threshold (0.05) so it always draws as a
        # coloured block.  util_pct is computed before this expansion so it
        # reflects the real GPU utilisation, not the inflated render width.
        cov = np.zeros(width + 1, dtype=np.int32)
        np.add.at(cov, ix0, 1)
        np.add.at(cov, np.minimum(ix1 + 1, width), -1)
        activity = np.where(np.cumsum(cov[:width]) > 0,
                            np.maximum(activity, 0.06), activity)

        # ── 4. Dominant color per pixel via searchsorted — O(width) ──────
        # ix0 is monotonically non-decreasing (sorted spans → sorted start pixels).
        # For pixel p, the candidate span is the last one whose left edge ≤ p.
        pix      = np.arange(width, dtype=np.int32)
        sp       = np.searchsorted(ix0, pix, side="right") - 1   # shape (width,)
        sp_safe  = np.clip(sp, 0, len(ix0) - 1)
        covered  = (sp >= 0) & (ix1[sp_safe] >= pix)
        dom_idx  = np.where(covered, c_np[sp_safe].astype(np.int32), -1)

        # ── Per-column char/style arrays ───────────────────────────────────
        # Returned raw (not yet RLE-encoded into a Text) so a caller can
        # overlay connector-line characters (see _ConnectorOverlay /
        # TimelineWidget.render) onto specific columns before the final
        # Text is built via _row_from_columns -- splicing arbitrary
        # characters into an already-built Rich Text is awkward with its
        # API, but overwriting entries in a plain list is trivial. width is
        # terminal columns (rarely more than a few hundred), so this
        # per-column Python loop is negligible next to the numpy work above
        # over however many thousand spans are actually in view.
        IDLE = 0.05
        chars:  list[str] = [" "] * width
        styles: list[str] = [""] * width
        for i in range(width):
            if activity[i] > IDLE:
                ci = int(dom_idx[i])
                chars[i]  = "█"
                styles[i] = _SPAN_PALETTE[ci] if ci >= 0 else "white"

        return chars, styles, util_pct

    @staticmethod
    def _row_from_columns(chars: list[str], styles: list[str]) -> Text:
        """RLE-encodes parallel per-column char/style lists into a Rich
        Text, grouping consecutive same-style columns into one run --
        shared by lane data rows and (now overlay-able) spacer rows."""
        row = Text()
        width = len(chars)
        i = 0
        while i < width:
            j = i + 1
            while j < width and styles[j] == styles[i]:
                j += 1
            row.append("".join(chars[i:j]), style=styles[i])
            i = j
        return row

    @staticmethod
    def _apply_overlay(chars: list[str], styles: list[str],
                       canvas: BrailleCanvas, canvas_row: int, width: int) -> None:
        """Overwrites entries in chars/styles wherever the Braille canvas
        has a connector-line dot in this row, leaving every other column
        (the vast majority — connectors are sparse) completely untouched."""
        for col in range(width):
            cell = canvas.cell(col, canvas_row)
            if cell is not None:
                chars[col], styles[col] = cell

    def on_mouse_move(self, event: MouseMove) -> None:
        """Update hover info when the mouse moves over the timeline."""
        UTIL_W = 6
        width  = self.size.width - self._COL_W - UTIL_W
        height = self.size.height
        if width < 4 or height < 4:
            return

        x, y = event.x - self._COL_W - 1, event.y
        # Rows 0-1 = ruler; lanes start at row 2, each lane = 2 rows (data + spacer)
        lane_row = y - 2
        if lane_row < 0 or x < 0 or x >= width:
            self._hover = ""
            self._hover_span_id = 0
            return

        visible_ns = self._trace_dur / self.zoom
        offset_ns  = self._trace_dur * self.view_x / (width * self.zoom)
        ns_per_px  = visible_ns / max(width, 1)

        total_lanes  = len(self._lane_names)
        visible_rows = max(1, (height - 4) // 2)
        max_sy       = max(0, total_lanes - visible_rows)
        lo           = min(self.view_y, max_sy)
        hi           = min(total_lanes, lo + visible_rows)

        lane_idx = lane_row // 2   # integer division: data row or spacer row → same lane
        actual_lane = lo + lane_idx
        if actual_lane >= hi:
            self._hover = ""
            self._hover_span_id = 0
            return

        lane_name   = self._lane_names[actual_lane]
        sorted_spans = self._sorted_spans[lane_name]
        if not sorted_spans:
            self._hover = ""
            self._hover_span_id = 0
            return

        # Time position the mouse is pointing at (absolute trace time)
        cursor_abs = int(self._view_start + offset_ns + x * ns_per_px)

        # Spatial index: narrow to spans that could contain cursor_abs
        starts  = self._starts_arr[lane_name]
        max_dur = self._max_dur[lane_name]
        lo_s = int(np.searchsorted(starts, cursor_abs - max_dur, side="left"))
        hi_s = int(np.searchsorted(starts, cursor_abs,           side="right"))
        candidates = sorted_spans[lo_s:hi_s]

        # Only show hover when cursor is inside a span's actual time range
        containing = [s for s in candidates
                      if s.start_ns <= cursor_abs <= s.end_ns]
        if not containing:
            self._hover = ""
            self._hover_span_id = 0
            return

        # If multiple spans overlap here, prefer the shortest (most specific)
        span  = min(containing, key=lambda s: s.duration_ns)
        dur   = _fmt_ns(span.duration_ns)
        start = _fmt_ns(span.start_ns - self._view_start)
        cat   = lane_name.split("/")[0]
        hover = f"{span.name}  [{cat}]  @{start}  dur {dur}"
        if span.parent_span_id:
            parent_name = self._sid_name.get(span.parent_span_id, span.parent_span_id[:8])
            hover += f"  ↑{parent_name}"
        n_links = self._connector_count.get(id(span), 0)
        if n_links:
            hover += f"  ⇄{n_links}"
        self._hover = hover
        self._hover_span_id = id(span)

    def on_leave(self, _event: Any) -> None:
        self._hover = ""
        self._hover_span_id = 0

    def render(self) -> Any:  # noqa: ANN401
        UTIL_W = 6
        width  = self.size.width - self._COL_W - UTIL_W
        height = self.size.height
        if width < 4 or height < 4:
            return Text("(too small)")

        out = Text()

        visible_ns   = self._trace_dur / self.zoom
        offset_ns    = self._trace_dur * self.view_x / (width * self.zoom)
        ns_per_px    = visible_ns / max(width, 1)
        tick_step_px = max(12, width // 8)

        # ── Time ruler (2 rows) ──────────────────────────────────────────
        out.append(" " * self._COL_W, style="dim")
        for i in range(0, width, tick_step_px):
            ts_ns = offset_ns + i * ns_per_px
            cell  = f"{_fmt_ns(ts_ns):<{tick_step_px}}"[:tick_step_px]
            out.append(cell, style="bold white")
        out.append("\n")

        out.append(" " * self._COL_W, style="dim")
        for i in range(width):
            out.append("┬" if i % tick_step_px == 0 else "─", style="dim")
        out.append("\n")

        # ── Lanes — 2 rows each (data + blank spacer) ────────────────────
        # Overhead: 2 ruler + 1 footer = 3; each lane needs 2 rows
        overhead     = 3
        visible_rows = max(1, (height - overhead) // 2)
        total_lanes  = len(self._lane_names)

        max_sy = max(0, total_lanes - visible_rows)
        lo = min(self.view_y, max_sy)
        hi = min(total_lanes, lo + visible_rows)
        visible_lanes = self._lane_names[lo:hi]
        lanes_drawn   = len(visible_lanes)

        # ── Cross-rank communication connectors ───────────────────────────
        # A Braille sub-cell canvas (src/ui/braille_canvas.py) covering
        # exactly the visible lanes' 2 rows each x the content width.
        # Endpoints outside the visible time window are clipped to the
        # nearest edge (matching how _density_row already clips spans
        # themselves) rather than hidden — a connector with one end
        # scrolled off-screen still shows as a line running to that edge,
        # which is more informative than vanishing entirely; a connector
        # with BOTH ends off the same side is skipped since nothing about
        # it would be visible anyway.
        #
        # Only the HOVERED span's own connectors are drawn, not every one
        # at once — on a busy trace, rendering all of them simultaneously
        # is a hairball of overlapping lines that reads as noise rather
        # than information. Hovering a specific send/recv or collective
        # call reveals just what that call was waiting on, on demand.
        canvas: BrailleCanvas | None = None
        if self._connectors and lanes_drawn and self._hover_span_id:
            lane_row: dict[str, int] = {name: i for i, name in enumerate(visible_lanes)}
            canvas = BrailleCanvas(cols=width, rows=lanes_drawn * 2)
            scale = width * self.zoom / self._trace_dur
            for pred_lane, pred_ns, succ_lane, succ_ns, confidence, pred_sid, succ_sid in self._connectors:
                if self._hover_span_id not in (pred_sid, succ_sid):
                    continue
                pred_row = lane_row.get(pred_lane)
                succ_row = lane_row.get(succ_lane)
                if pred_row is None or succ_row is None:
                    continue
                pred_col = (pred_ns - self._view_start) * scale - self.view_x
                succ_col = (succ_ns - self._view_start) * scale - self.view_x
                if (pred_col < 0 and succ_col < 0) or (pred_col > width and succ_col > width):
                    continue
                pred_col = max(0.0, min(float(width), pred_col))
                succ_col = max(0.0, min(float(width), succ_col))
                style = _CONNECTOR_STYLE.get(confidence, "grey58")
                # Elbow routing (vertical - horizontal - vertical), not a
                # raw diagonal: a straight line from one lane's data row to
                # another's sweeps across every column AND every
                # intermediate row along the way, painting over whatever
                # span data happens to be there. Routing the long
                # horizontal traversal through the SOURCE lane's own
                # spacer row (never a data row) means only the final short
                # vertical drop/rise into the target actually crosses
                # other lanes' content — as a single thin vertical line at
                # one column, not a diagonal smear across the whole width.
                # (row*8 = that lane's starting dot-row, 2 char rows x 4
                # dots each; +2 = data row's own vertical center; +6 =
                # its adjacent spacer row's center.)
                pred_data_y   = pred_row * 8 + 2
                pred_spacer_y = pred_row * 8 + 6
                succ_data_y   = succ_row * 8 + 2
                x0, x1 = int(pred_col * 2), int(succ_col * 2)
                canvas.line(x0, pred_data_y, x0, pred_spacer_y, style=style)
                canvas.line(x0, pred_spacer_y, x1, pred_spacer_y, style=style)
                canvas.line(x1, pred_spacer_y, x1, succ_data_y, style=style)

        for row_idx, lane_name in enumerate(visible_lanes):
            cat   = lane_name.split("/")[0]
            color = _cat_color(cat)

            # Data row
            out.append(f"{self._lane_label(lane_name)} ", style=f"bold {color}")
            chars, styles, util_pct = self._density_row(lane_name, width, color)
            if canvas is not None:
                self._apply_overlay(chars, styles, canvas, row_idx * 2, width)
            out.append(self._row_from_columns(chars, styles))
            util_col = ("bright_green" if util_pct >= 50
                        else "yellow"  if util_pct >= 20
                        else "red")
            out.append(f" {util_pct:4.0f}%", style=f"dim {util_col}")
            out.append("\n")

            # Blank spacer row — gives visual breathing room between lanes,
            # and (when a connector's routed path passes through it) now
            # also carries the connector line's continuation between two
            # non-adjacent lanes.
            if canvas is not None:
                spacer_chars: list[str]  = [" "] * width
                spacer_styles: list[str] = [""] * width
                self._apply_overlay(spacer_chars, spacer_styles, canvas, row_idx * 2 + 1, width)
                out.append(" " * self._COL_W)
                out.append(self._row_from_columns(spacer_chars, spacer_styles))
            out.append("\n")

        # ── Pad to push footer to the bottom of the widget ───────────────
        used = 2 + lanes_drawn * 2        # ruler rows + lane rows
        pad  = max(0, height - used - 1)  # -1 for the footer line itself
        for _ in range(pad):
            out.append("\n")

        # ── Status footer — always at the bottom ─────────────────────────
        lane_info = (f"  lanes {lo+1}–{hi}/{total_lanes}"
                     if total_lanes > visible_rows else "")
        if self._hover:
            # Show hover info in place of key hints
            out.append(f" {self._hover}", style="bold white")
        else:
            out.append(
                f" zoom {self.zoom:.1f}×"
                f"  offset {_fmt_ns(offset_ns)}"
                f"  window {_fmt_ns(visible_ns)}"
                f"{lane_info}"
                f"    [←→] scroll  [↑↓] pan  [+/-] zoom  [r] reset",
                style="dim italic",
            )
        return out

    def action_scroll_right(self) -> None:
        canvas_w  = max(1, self.size.width - self._COL_W)
        step      = max(1, canvas_w // 8)
        # Maximum useful scroll: exactly enough to bring the last virtual pixel on-screen
        max_scroll = max(0, int(canvas_w * (self.zoom - 1)))
        self.view_x = min(self.view_x + step, max_scroll)

    def action_scroll_left(self) -> None:
        canvas_w = max(1, self.size.width - self._COL_W)
        step     = max(1, canvas_w // 8)
        self.view_x = max(0, self.view_x - step)

    def action_scroll_up(self) -> None:
        self.view_y = max(0, self.view_y - 1)

    def action_scroll_down(self) -> None:
        # Clamp here, not in render(), so we never mutate reactive state during a render
        canvas_h     = max(1, self.size.height)
        visible_rows = max(1, (canvas_h - 4) // 2)
        max_sy       = max(0, len(self._lane_names) - visible_rows)
        self.view_y = min(self.view_y + 1, max_sy)

    def action_zoom_in(self) -> None:
        self.zoom = min(self.zoom * 1.5, 128.0)

    def action_zoom_out(self) -> None:
        self.zoom = max(self.zoom / 1.5, 0.125)

    def action_reset(self) -> None:
        self.view_x = 0
        self.view_y = 0
        self.zoom     = 1.0
        self.refresh()


# ── Hotspots widget ───────────────────────────────────────────────────────────

_SORT_COLS   = ["total_ns", "avg_ns", "count", "min_ns", "max_ns"]
_SORT_LABELS = ["Total ▼",  "Avg ▼",  "Count ▼","Min ▼", "Max ▼"]


class HotspotsWidget(Widget):
    """
    VTune-style sortable function table with inline % bars.
    s → cycle sort column; / → focus filter input; j/k or ↑↓ → navigate rows.
    """

    DEFAULT_CSS = """
    HotspotsWidget { height: 1fr; layout: vertical; border: round $primary; padding: 0 1; }
    #hs-filter    { height: 3; dock: top; }
    #hs-table     { height: 1fr; }
    #hs-sort-hint { height: 1; dock: bottom; color: $text-muted; }
    """

    BINDINGS = [
        Binding("j", "cursor_down_row", "↓", show=False),
        Binding("k", "cursor_up_row",   "↑", show=False),
    ]

    sort_idx:    reactive[int] = reactive(0)
    filter_text: reactive[str] = reactive("")

    def __init__(self, trace: Trace, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.trace = trace
        self.border_title = "Kernels"

    def compose(self) -> ComposeResult:
        yield Input(placeholder="  / filter by name…", id="hs-filter")
        yield DataTable(id="hs-table", cursor_type="row")
        # NOTE: the literal brackets below must be escaped (\[) -- unescaped
        # "[s]"/"[/]" collide with Rich markup (strikethrough-open /
        # close-most-recent-tag), which silently struck through and
        # truncated this exact hint text before this fix.
        yield Static(
            "  [dim]\\[s] cycle sort   \\[/] filter   \\[j/k ↑↓] navigate[/dim]",
            id="hs-sort-hint",
        )

    def on_mount(self) -> None:
        self._rebuild()

    def action_cursor_down_row(self) -> None:
        self.query_one("#hs-table", DataTable).action_cursor_down()

    def action_cursor_up_row(self) -> None:
        self.query_one("#hs-table", DataTable).action_cursor_up()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "hs-filter":
            self.filter_text = event.value
            self._rebuild()

    def action_next_sort(self) -> None:
        self.sort_idx = (self.sort_idx + 1) % len(_SORT_COLS)
        self._rebuild()

    def _rebuild(self) -> None:
        dt: DataTable = self.query_one("#hs-table", DataTable)
        dt.clear(columns=True)

        sort_col   = _SORT_COLS[self.sort_idx]
        sort_label = _SORT_LABELS[self.sort_idx]

        dt.add_columns(
            "  Function", "Backend", "Count",
            sort_label, "Avg", "Min", "Max", "%", "Distribution (% of total)",
        )

        all_stats = sorted(self.trace.aggregated_stats(),
                           key=lambda r: r.get(sort_col, 0), reverse=True)
        # Hide zero-duration CPU samples unless the user is actively filtering
        flt = self.filter_text.lower()
        if flt:
            stats = all_stats
        else:
            stats = [r for r in all_stats if r["total_ns"] > 0 or r["category"] != "cpu"]
        total_all = sum(r["total_ns"] for r in stats) or 1
        bar_w     = 24

        # Build source-location lookup from annotated spans (file/line tags)
        src_loc: dict[str, str] = {}
        for s in self.trace.spans:
            if s.name not in src_loc and s.tags.get("file"):
                f = Path(s.tags["file"]).name
                src_loc[s.name] = f"{f}:{s.tags.get('line', '?')}"

        for row in stats:
            if flt and flt not in row["name"].lower():
                continue
            display_name = _fmt_kernel_name(row["name"])
            color = _cat_color(row["category"])
            frac  = row["total_ns"] / total_all
            bar   = _bar(frac, bar_w)

            name_text = Text()
            name_text.append(f"  {display_name[:46]}", style=f"bold {color}")
            loc = src_loc.get(row["name"])
            if loc:
                name_text.append(f"\n  {loc}", style="dim cyan")

            dt.add_row(
                name_text,
                Text(row["category"],          style=color),
                str(row["count"]),
                _fmt_ns(row["total_ns"]),
                _fmt_ns(row["avg_ns"]),
                _fmt_ns(row["min_ns"]),
                _fmt_ns(row["max_ns"]),
                f"{row['pct']:.1f}%",
                Text(bar, style=f"bold {color}"),
                key=row["name"],
            )


# ── Roofline widget ───────────────────────────────────────────────────────────

def _has_roofline_data(trace: Trace) -> bool:
    """Cheap existence check used by ProfilerApp.compose() to decide
    whether the Roofline tab is worth showing at all — mirrors the
    trace._has_stacks / trace.disasm checks already used for the Call
    Tree / Disasm tabs' own conditional visibility."""
    try:
        from ..analysis.roofline import analyze_trace
        pts = analyze_trace(trace)
        return any(m.arith_intensity > 0 and m.achieved_tflops > 0 for _, m in pts)
    except Exception:
        return False


class RooflineWidget(Widget):
    """
    Log-log roofline scatter — one dot per profiled GPU kernel — drawn with
    the same Braille sub-cell canvas Timeline uses for its communication
    connectors (src/ui/braille_canvas.py). Reuses analysis/roofline.py's
    existing per-kernel arithmetic-intensity / achieved-TFLOP/s estimates
    (hardware-counter based when available, disassembly-based estimate
    otherwise — see KernelMetrics.data_source) rather than computing
    anything new here; this widget's job is only to plot them.
    """

    DEFAULT_CSS = """
    RooflineWidget { height: 1fr; border: round $primary; padding: 0 1; }
    """

    _BOUND_COLOR = {"compute": "bright_cyan", "memory": "bright_magenta", "unknown": "grey58"}

    def __init__(self, trace: Trace, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.trace = trace
        self.border_title = "Roofline"
        try:
            from ..analysis.roofline import analyze_trace
            self._metrics = analyze_trace(trace)
        except Exception:
            self._metrics = []

    def render(self) -> Any:  # noqa: ANN401
        pts = [(d, m) for d, m in self._metrics
               if m.arith_intensity > 0 and m.achieved_tflops > 0]
        if not pts:
            return Text(
                "No roofline data for this trace — needs GPU hardware-counter\n"
                "or disassembly-estimated kernel metrics. Try:\n\n"
                "  hprofiler roofline --backend <backend> -- ./app",
                style="dim",
            )

        width  = max(20, self.size.width - 2) if self.size.width else 60
        height = max(8, self.size.height - 3) if self.size.height else 20

        ai_vals    = [m.arith_intensity  for _, m in pts]
        tflop_vals = [m.achieved_tflops  for _, m in pts]
        ridge_vals = [m.ridge            for _, m in pts if m.ridge > 0]
        peak_vals  = [d.fp32_tflops      for d, _ in pts if d.fp32_tflops > 0]

        lo_x = math.log10(max(min(ai_vals) * 0.5, 1e-3))
        hi_x = math.log10(max(max(ai_vals + ridge_vals) * 2.0, 10 ** (lo_x + 1)))
        lo_y = math.log10(max(min(tflop_vals) * 0.3, 1e-4))
        hi_y = math.log10(max(max(tflop_vals + peak_vals) * 1.3, 10 ** (lo_y + 1)))

        def to_dot(ai: float, tf: float) -> tuple[int, int]:
            fx = (math.log10(max(ai, 1e-6)) - lo_x) / (hi_x - lo_x)
            fy = (math.log10(max(tf, 1e-6)) - lo_y) / (hi_y - lo_y)
            fx = min(1.0, max(0.0, fx))
            fy = min(1.0, max(0.0, fy))
            return int(fx * (width * 2 - 1)), int((1.0 - fy) * (height * 4 - 1))

        canvas = BrailleCanvas(cols=width, rows=height)

        # Roofline knee per distinct device (bandwidth-bound diagonal up to
        # the ridge point, then a flat compute-bound ceiling), drawn first
        # so kernel dots always render on top of it, not underneath.
        seen: set[str] = set()
        for dev, m in pts:
            key = f"{dev.name}:{dev.fp32_tflops:.3f}"
            if key in seen or dev.fp32_tflops <= 0 or dev.bandwidth_gbs <= 0 or m.ridge <= 0:
                continue
            seen.add(key)
            ai_lo = 10 ** lo_x
            tf_lo = max(dev.bandwidth_gbs * ai_lo / 1000.0, 1e-6)
            p0 = to_dot(ai_lo, tf_lo)
            p1 = to_dot(m.ridge, dev.fp32_tflops)
            p2 = to_dot(10 ** hi_x, dev.fp32_tflops)
            canvas.line(p0[0], p0[1], p1[0], p1[1], style="dim white")
            canvas.line(p1[0], p1[1], p2[0], p2[1], style="dim white")

        for dev, m in pts:
            x, y = to_dot(m.arith_intensity, m.achieved_tflops)
            canvas.set_dot(x, y, style=f"bold {self._BOUND_COLOR.get(m.bound, 'grey58')}")

        body = Text()
        for cy in range(height):
            for cx in range(width):
                cell = canvas.cell(cx, cy)
                if cell is not None:
                    ch, style = cell
                    body.append(ch, style=style)
                else:
                    body.append(" ")
            body.append("\n")

        n_compute = sum(1 for _, m in pts if m.bound == "compute")
        n_memory  = sum(1 for _, m in pts if m.bound == "memory")
        n_unknown = len(pts) - n_compute - n_memory
        legend = (
            f"AI {10**lo_x:.2g}–{10**hi_x:.2g} FLOP/B"
            f"   TFLOP/s {10**lo_y:.2g}–{10**hi_y:.2g}"
            f"   [bold bright_cyan]●[/bold bright_cyan] compute {n_compute}"
            f"   [bold bright_magenta]●[/bold bright_magenta] memory {n_memory}"
        )
        if n_unknown:
            legend += f"   [bold grey58]●[/bold grey58] unknown {n_unknown}"
        body.append_text(Text.from_markup(legend))
        return body


# ── Flame graph widget ────────────────────────────────────────────────────────

class FlameGraphWidget(Static):
    """ASCII proportional flame graph of CPU samples."""

    def __init__(self, trace: Trace, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._trace = trace

    def render(self) -> Any:  # noqa: ANN401
        cpu_spans = [s for s in self._trace.spans if s.category == Category.CPU]
        if not cpu_spans:
            return Panel(
                "[dim]No CPU samples captured.\n\n"
                "Run with [bold]--backend cpu[/bold] to enable CPU profiling.[/dim]",
                title="[bold]CPU Flame Graph[/bold]",
                border_style="dim",
            )

        totals: dict[str, int] = defaultdict(int)
        for s in cpu_spans:
            totals[s.name] += max(s.duration_ns, 1)

        sorted_items = sorted(totals.items(), key=lambda kv: -kv[1])
        grand_total  = sum(totals.values()) or 1
        width        = 72

        lines: list[str] = []
        lines.append(
            f"[bold cyan]CPU Flame Graph[/bold cyan]  "
            f"[dim]samples: {len(cpu_spans)}   total: {_fmt_ns(grand_total)}[/dim]\n"
        )

        for name, dur in sorted_items[:30]:
            bar_w = max(1, int(dur / grand_total * width))
            pct   = 100.0 * dur / grand_total
            lines.append(
                f"[cyan]{'█' * bar_w:<{width}}[/cyan]  "
                f"[bold]{name[:36]:<36}[/bold]  "
                f"[yellow]{_fmt_ns(dur):>10}[/yellow]  "
                f"[dim]{pct:5.1f}%[/dim]"
            )

        lines.append(
            f"\n[dim]Showing top {min(30, len(sorted_items))} of "
            f"{len(sorted_items)} functions[/dim]"
        )
        return "\n".join(lines)


# ── Call tree widget ─────────────────────────────────────────────────────────

# Construction logic lives in analysis/call_tree.py (pure trace analysis, no
# Textual dependency) so the GUI can reuse it without importing this whole
# Textual-based module -- re-exported here under the original names so every
# existing call site/test in this file keeps working unchanged.
from ..analysis.call_tree import (  # noqa: E402
    _CTNode, _RawNode, _ct_build_raw, _ct_aggregate, _StackNode,
    _ct_build_from_stacks, _ct_build,
)


class CallTreeWidget(Widget):
    """
    Hierarchical call tree for all backends.
    Builds parent-child relationships from temporal containment of spans,
    then aggregates duplicate siblings by name for a compact view.
    Keys: ↑↓ navigate · Enter/Space expand · e expand all · u collapse all
    """

    DEFAULT_CSS = """
    CallTreeWidget { height: 1fr; layout: vertical; border: round $primary; padding: 0 1; }
    #ct-tree      { height: 1fr; }
    #ct-hint      { height: 1; dock: bottom; color: $text-muted; }
    """

    def __init__(self, trace: Trace, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.border_title = "Call Tree"
        self._trace = trace

    def compose(self) -> ComposeResult:
        yield Tree("Call Tree", id="ct-tree")
        # NOTE: literal brackets must be escaped (\[) -- see HotspotsWidget's
        # #hs-sort-hint for why an unescaped "[u]"/"[e]"/"[enter/space]"
        # collides with Rich markup (u = underline shorthand; anything else
        # bracketed is still consumed as an unrecognised style tag and
        # silently vanishes from the rendered text either way).
        yield Static(
            "  [dim]\\[↑↓] navigate  \\[enter/space] expand  \\[e] expand all  \\[u] collapse all[/dim]",
            id="ct-hint",
        )

    def on_mount(self) -> None:
        self._rebuild()

    def _rebuild(self) -> None:
        tree: Tree = self.query_one("#ct-tree", Tree)  # type: ignore[type-arg]
        tree.clear()
        wall_ns = max(_trace_wall_ns(self._trace), 1)
        spans = [s for s in self._trace.spans if s.duration_ns > 0]

        if not spans:
            tree.root.add_leaf("[dim]No duration spans captured.[/dim]")
            tree.root.expand()
            return

        roots = _ct_build(spans)
        for node in roots:
            self._add_node(tree.root, node, wall_ns)
        tree.root.expand()

    def _add_node(self, parent: Any, node: _CTNode, wall_ns: int) -> None:
        color = _cat_color(node.category)
        pct   = 100.0 * node.total_ns / wall_ns
        label = (
            f"[{color}]{node.name}[/{color}]"
            f"  [dim]{node.count}×[/dim]"
            f"  [yellow]{_fmt_ns(node.total_ns)}[/yellow]"
            f"  [dim]{pct:.1f}%[/dim]"
        )
        if node.count > 1:
            label += f"  [dim]avg {_fmt_ns(node.avg_ns)}[/dim]"
        if node.children:
            branch = parent.add(label, data=node)
            for child in node.children:
                self._add_node(branch, child, wall_ns)
        else:
            parent.add_leaf(label, data=node)

    def on_key(self, event: Any) -> None:
        tree: Tree = self.query_one("#ct-tree", Tree)  # type: ignore[type-arg]
        if event.key == "e":
            tree.root.expand_all()
            event.stop()
        elif event.key == "u":
            tree.root.collapse_all()
            tree.root.expand()
            event.stop()


# ── Disasm widget ─────────────────────────────────────────────────────────────

class DisasmWidget(Widget):
    """
    Split-pane disassembly viewer.

    Left  — kernel list (from profiled spans); ↑↓ to select.
    Right — annotated assembly coloured by instruction type.
    Bottom — instruction-mix bar (vector / scalar / memory / control).

    Supports all backends:
      cpu / opencl-cpu  x86-64 from .jit.so or the main binary
      cuda (AoT)        SASS from cuobjdump
      cuda (JIT)        cubin captured by the hook, disassembled with nvdisasm
      rocm              AMDGCN from llvm-objdump
    """

    DEFAULT_CSS = """
    DisasmWidget {
        height: 1fr;
        layout: vertical;
        border: round $primary;
        padding: 0 1;
    }
    #disasm-h {
        height: 1fr;
        layout: horizontal;
    }
    #disasm-kernels {
        width: 32;
        border-right: solid $primary;
    }
    #disasm-asm {
        width: 1fr;
    }
    #disasm-mix {
        height: 3;
        dock: bottom;
        padding: 0 1;
        background: $boost;
    }
    #disasm-hints {
        height: auto;
        max-height: 6;
        dock: bottom;
        padding: 0 1;
        background: $boost;
    }
    """

    can_focus = True

    BINDINGS = [
        Binding("up",   "prev_kernel", "↑", show=False),
        Binding("down", "next_kernel", "↓", show=False),
        Binding("k",    "prev_kernel", "↑", show=False),
        Binding("j",    "next_kernel", "↓", show=False),
    ]

    _sel: reactive[int] = reactive(0)

    def __init__(self, trace: Trace, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.border_title = "Source"
        self._trace    = trace
        # Build the ordered kernel list: profiled kernels first (by total time),
        # supplemented by kernels only present in disasm data.
        stats = trace.aggregated_stats()
        profiled_names = [r["name"] for r in stats]
        disasm_only    = [n for n in trace.disasm if n not in profiled_names]
        self._kernel_names: list[str] = profiled_names + disasm_only
        self._stats_by_name = {r["name"]: r for r in stats}
        # Cache pre-rendered Text objects per kernel so navigating back is instant.
        self._disasm_cache: dict[str, Any] = {}

    # ── Composition ──────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        with Horizontal(id="disasm-h"):
            yield DataTable(id="disasm-kernels", cursor_type="row", show_cursor=True)
            with ScrollableContainer(id="disasm-asm"):
                yield RichLog(id="disasm-log", highlight=False, markup=True,
                              wrap=False, auto_scroll=False)
        yield Static("", id="disasm-hints")
        yield Static("", id="disasm-mix")

    def on_mount(self) -> None:
        self._rebuild_kernel_list()
        self._show_disasm()
        self._show_mix()
        self._show_hints()
        self._last_disasm_version = -1  # track _disasm_version to catch annotation updates
        # Poll until background disasm collection + annotation finishes, then refresh.
        self.set_interval(0.5, self._poll_disasm_ready)

    # ── Kernel list (left pane) ───────────────────────────────────────────────

    def _rebuild_kernel_list(self) -> None:
        dt: DataTable = self.query_one("#disasm-kernels", DataTable)
        dt.clear(columns=True)
        dt.add_columns("Kernel", "Arch", "Total")
        disasm = self._trace.disasm

        for name in self._kernel_names:
            arch  = disasm[name].arch if name in disasm else "—"
            stat  = self._stats_by_name.get(name)
            total = _fmt_ns(stat["total_ns"]) if stat else "—"
            has_d = "✓" if name in disasm else " "
            color = _cat_color(stat["category"]) if stat else "grey70"
            disp = _fmt_kernel_name(name)
            dt.add_row(
                Text(f"{has_d} {disp[:22]}", style=f"bold {color}"),
                Text(arch, style="dim"),
                Text(total),
                key=name,
            )

    # ── Background disasm poller ──────────────────────────────────────────────

    def _poll_disasm_ready(self) -> None:
        """Refresh kernel list and disasm panes when background collection updates."""
        cur_version = self._trace._disasm_version
        if cur_version <= self._last_disasm_version:
            return
        # If only annotations changed (version bumped but set of names didn't grow),
        # invalidate the render cache so heat/stall columns are re-drawn.
        disasm = self._trace.disasm
        if cur_version != self._last_disasm_version:
            self._disasm_cache.clear()
        self._last_disasm_version = cur_version
        stats = self._trace.aggregated_stats()
        profiled_names = [r["name"] for r in stats]
        disasm_only = [n for n in disasm if n not in profiled_names]
        self._kernel_names = profiled_names + disasm_only
        self._stats_by_name = {r["name"]: r for r in stats}
        self._rebuild_kernel_list()
        self._show_disasm()
        self._show_mix()
        self._show_hints()

    # ── Disassembly pane (right) ──────────────────────────────────────────────

    def _show_disasm(self) -> None:
        from rich.text import Text
        from ..disasm.classifier import ITYPE_COLOR, ITYPE_LABEL

        log: RichLog = self.query_one("#disasm-log", RichLog)
        log.clear()

        if not self._kernel_names:
            log.write(Text("No kernels profiled.", style="dim"))
            return

        idx    = min(self._sel, len(self._kernel_names) - 1)
        name   = self._kernel_names[idx]
        disasm = self._trace.disasm

        if name not in disasm:
            msg = Text()
            msg.append(name, style="bold")

            # Distinguish two very different failure points that both land
            # here, so the message doesn't send someone chasing a missing
            # objdump/cuobjdump install when the real issue is that no
            # call-site symbol was ever captured for this event in the
            # first place (e.g. a trace captured before the hooks were
            # rebuilt with codeptr resolution, or a construct that
            # genuinely doesn't resolve one yet -- point-to-point MPI,
            # omp_critical_hold). A real user hit exactly this: the old,
            # always-the-same "install objdump" tip was actively
            # misleading when the actual fix was "rebuild and re-capture".
            tag_info: tuple[str, str] | None = None
            for s in self._trace.spans:
                if s.name != name:
                    continue
                if s.tags.get("sym"):
                    tag_info = ("sym", s.tags["sym"])
                    break
                if s.tags.get("lib"):
                    off = s.tags.get("offset", "?")
                    tag_info = ("lib", f"{s.tags['lib']},offset={off}")
                    break
                if s.tags.get("type") == "kernel":
                    tag_info = ("kernel", "")
                    break

            if tag_info is None:
                msg.append(
                    "\n\nNo disassembly available -- this event has no "
                    "resolved call-site symbol at all (no sym=/lib= tag on "
                    "any captured span), so there was nothing to "
                    "disassemble in the first place. This is NOT a missing "
                    "objdump/nm problem.\n\n"
                    "Most likely cause: this trace was captured with an "
                    "older build of the profiler hooks, from before they "
                    "resolved call sites for this construct. Rebuild the "
                    "hooks and capture a FRESH trace -- re-opening an old "
                    "trace file will never show disassembly here even "
                    "after rebuilding, since the tag is only written at "
                    "capture time.\n\n"
                    "If you already rebuilt and re-ran: point-to-point MPI "
                    "calls (Send/Recv/Isend/Irecv/Wait*) and OpenMP's "
                    "omp_critical_hold don't resolve a call site yet -- a "
                    "known, disclosed gap, not a bug.",
                    style="dim")
            elif tag_info[0] == "kernel":
                msg.append(
                    "\n\nNo disassembly available for this GPU kernel.\n\n"
                    "Tips:\n"
                    "  • For CUDA AoT: install cuobjdump (CUDA toolkit)\n"
                    "  • For ROCm: install llvm-objdump\n"
                    "  • JIT cubins are auto-captured when the CUDA hook is loaded",
                    style="dim")
            else:
                kind, detail = tag_info
                tag_str = f"sym={detail}" if kind == "sym" else detail
                msg.append(
                    f"\n\nCall site resolved ({tag_str}) but disassembly "
                    "still failed -- likely a missing tool, or the symbol/"
                    "library wasn't found where expected on this machine.\n\n"
                    "Tips:\n"
                    "  • objdump and/or nm must be installed\n"
                    "  • Check the binary/library path above still exists "
                    "and is readable from where the TUI is running",
                    style="dim")
            log.write(msg)
            return

        # Return the cached render if available.
        if name in self._disasm_cache:
            for obj in self._disasm_cache[name]:
                log.write(obj)
            return

        kd   = disasm[name]
        stat = self._stats_by_name.get(name)

        hdr = Text()
        hdr.append(name, style="bold")
        hdr.append(f"  arch: {kd.arch}   source: {kd.source}", style="dim")
        if stat:
            hdr.append(f"   {_fmt_ns(stat['total_ns'])}", style="yellow")
            hdr.append(f"  {stat['count']}×  {stat['pct']:.1f}%", style="dim")

        sep = Text("─" * 80, style="dim")

        ptxas_caveat: Text | None = None
        if getattr(kd, "ptxas_derived", False):
            ptxas_caveat = Text()
            ptxas_caveat.append("⚠ ", style="yellow")
            ptxas_caveat.append(
                "SASS compiled offline from PTX via ptxas — not the actual runtime-JIT SASS. "
                "PC heat offsets are approximate.",
                style="dim yellow",
            )

        if not kd.lines:
            body = Text("(no instructions decoded)", style="dim")
            cache = [hdr, sep]
            if ptxas_caveat:
                cache.append(ptxas_caveat)
            cache.append(body)
            self._disasm_cache[name] = cache
            for obj in self._disasm_cache[name]:
                log.write(obj)
            return

        _MAX_LINES = 500
        addr_w, mne_w, ops_w = 8, 12, 36

        # Determine whether to show heat / stall / source columns
        has_heat   = any(ln.sample_pct > 0.0 for ln in kd.lines)
        has_stall  = any(ln.stall_cycles >= 0 for ln in kd.lines)
        has_source = any(ln.source_file for ln in kd.lines)

        # Build a single Text object with style spans — avoids all markup
        # parsing overhead and is an order of magnitude faster than
        # per-line log.write() calls or markup string formatting.
        body = Text()
        prev_src: tuple[str, int] = ("", 0)
        for ln in kd.lines[:_MAX_LINES]:
            color = ITYPE_COLOR.get(ln.itype, "grey70")
            label = ITYPE_LABEL.get(ln.itype, "   ")

            # Source annotation: emit a dim comment when file:line changes
            if has_source and ln.source_file:
                cur_src = (ln.source_file, ln.source_line)
                if cur_src != prev_src:
                    prev_src = cur_src
                    indent = " " * (
                        (7 if has_heat else 0) +
                        (3 if has_stall else 0) +
                        addr_w + 2
                    )
                    short_file = Path(ln.source_file).name
                    body.append(
                        f"{indent}// {short_file}:{ln.source_line}\n",
                        style="dim cyan",
                    )

            addr_s = f"{ln.addr:0{addr_w}x}" if ln.addr else " " * addr_w
            mne_s  = f"{ln.mnemonic:<{mne_w}}"[:mne_w]
            ops_s  = f"{ln.operands:<{ops_w}}"[:ops_w]

            # Heat column
            if has_heat:
                p = ln.sample_pct
                if p >= 10.0:
                    heat_s = f"{p:5.1f}%"
                    heat_style = "bold red"
                elif p >= 5.0:
                    heat_s = f"{p:5.1f}%"
                    heat_style = "red"
                elif p >= 1.0:
                    heat_s = f"{p:5.1f}%"
                    heat_style = "yellow"
                elif p > 0.0:
                    heat_s = f"{p:5.1f}%"
                    heat_style = "white"
                else:
                    heat_s = "      "
                    heat_style = "dim"
                body.append(heat_s + " ", style=heat_style)

            # Stall column
            if has_stall:
                sc = ln.stall_cycles
                if sc >= 0:
                    stall_col = "red" if sc >= 10 else ("yellow" if sc >= 5 else "dim")
                    body.append(f"{sc:2d} ", style=stall_col)
                else:
                    body.append("   ", style="dim")

            body.append(addr_s + "  ", style="dim")
            body.append(mne_s + "  " + ops_s, style=color)
            body.append("  " + label, style=color)
            if ln.stall_reason:
                body.append(f"  [{ln.stall_reason}]", style="dim yellow")
            elif ln.comment:
                body.append(f"  ; {ln.comment}", style="dim")
            body.append("\n")

        if len(kd.lines) > _MAX_LINES:
            body.append(
                f"… {len(kd.lines) - _MAX_LINES} more instructions not shown",
                style="dim"
            )

        cache = [hdr, sep]
        if ptxas_caveat:
            cache.append(ptxas_caveat)
        cache.append(body)
        self._disasm_cache[name] = cache
        for obj in self._disasm_cache[name]:
            log.write(obj)

    # ── Instruction-mix bar (bottom) ─────────────────────────────────────────

    def _show_mix(self) -> None:
        from ..disasm.classifier import ITYPE_COLOR, ITYPE_LABEL, InsnType

        mix_bar: Static = self.query_one("#disasm-mix", Static)

        if not self._kernel_names:
            mix_bar.update("")
            return

        idx  = min(self._sel, len(self._kernel_names) - 1)
        name = self._kernel_names[idx]
        disasm = self._trace.disasm

        if name not in disasm or not disasm[name].lines:
            mix_bar.update("[dim]No instruction mix data[/dim]")
            return

        pcts    = disasm[name].itype_pcts()
        total   = disasm[name].total_insns()
        bar_w   = 10
        parts   = []
        order   = [InsnType.VEC_SP, InsnType.VEC_DP, InsnType.VEC_MEM, InsnType.VECTOR,
                   InsnType.COMPUTE, InsnType.MEMORY, InsnType.SCALAR,
                   InsnType.CONTROL, InsnType.SYNC]
        for itype in order:
            pct = pcts.get(itype, 0.0)
            if pct < 0.5:
                continue
            color = ITYPE_COLOR[itype]
            label = ITYPE_LABEL[itype]
            bar   = _bar(pct / 100, bar_w)
            parts.append(
                f"[{color}]{label}[/{color}] [{color}]{bar}[/{color}] "
                f"[dim]{pct:.0f}%[/dim]"
            )
        mix_bar.update(
            f"  [dim]insns: {total}[/dim]   " + "   ".join(parts)
            if parts else f"  [dim]insns: {total}[/dim]"
        )

    # ── Assembly optimization hints (bottom) ─────────────────────────────────

    def _show_hints(self) -> None:
        hints_bar: Static = self.query_one("#disasm-hints", Static)

        if not self._kernel_names:
            hints_bar.update("")
            return

        idx  = min(self._sel, len(self._kernel_names) - 1)
        name = self._kernel_names[idx]
        disasm = self._trace.disasm

        if name not in disasm or not disasm[name].lines:
            hints_bar.update("")
            return

        try:
            from ..analysis.asm_advisor import advise, AsmAdvice
            advices = advise(disasm[name])
        except Exception:
            hints_bar.update("")
            return

        if not advices:
            hints_bar.update("")
            return

        parts: list[str] = []
        _SEV_COLOR = {"crit": "bold red", "warn": "yellow", "info": "cyan"}
        for adv in advices[:4]:  # show at most 4 hints
            col = _SEV_COLOR.get(adv.severity, "white")
            msg = adv.message[:72]
            parts.append(
                f"[{col}]{adv.icon}[/{col}] "
                f"[dim]{adv.category:<8}[/dim] "
                f"[{col}]{msg}[/{col}]"
            )
        hints_bar.update("\n".join(parts))

    # ── Navigation ───────────────────────────────────────────────────────────

    def _move(self, delta: int) -> None:
        if not self._kernel_names:
            return
        self._sel = max(0, min(len(self._kernel_names) - 1, self._sel + delta))
        dt: DataTable = self.query_one("#disasm-kernels", DataTable)
        dt.move_cursor(row=self._sel)
        self._show_disasm()
        self._show_mix()
        self._show_hints()

    def action_prev_kernel(self) -> None:
        self._move(-1)

    def action_next_kernel(self) -> None:
        self._move(1)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        try:
            idx = list(self.query_one("#disasm-kernels", DataTable).rows.keys()).index(event.row_key)
            self._sel = idx
            self._show_disasm()
            self._show_mix()
            self._show_hints()
        except (ValueError, AttributeError):
            pass


# ── Main App ──────────────────────────────────────────────────────────────────

class TopBar(Horizontal):
    """
    Replaces Textual's default `Header()` (a generic app-title + clock bar
    that doesn't carry any profiler-specific context). Left: app name +
    command. Right: whatever run context actually applies to this trace
    (rank count, device, wall time) -- fields that don't apply (e.g. no
    MPI spans, no GPU device) are simply omitted rather than shown as a
    fake "n/a", since a single-process CPU-only trace has no "rank" concept
    to begin with.
    """

    DEFAULT_CSS = """
    TopBar { height: 1; background: $boost; }
    TopBar > #topbar-left  { width: 1fr;  content-align: left middle;  padding: 0 1; }
    TopBar > #topbar-right { width: auto; content-align: right middle; padding: 0 1; color: $text-muted; }
    """

    def __init__(self, trace: Trace, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._trace = trace

    def compose(self) -> ComposeResult:
        yield Static(self._left_text(), id="topbar-left")
        yield Static(self._right_text(), id="topbar-right")

    def _left_text(self) -> Text:
        meta = self._trace.metadata
        cmd = f"{meta.command} {' '.join(meta.args[:2])}".strip() or "(no command)"
        if len(cmd) > 50:
            cmd = cmd[:47] + "…"
        return Text.from_markup(f"[bold bright_white]hprofiler[/bold bright_white]  [dim]{cmd}[/dim]")

    def _right_text(self) -> Text:
        trace = self._trace
        parts: list[str] = []
        n_ranks = len({
            s.tags.get("rank") for s in trace.spans
            if s.category.value == "mpi" and s.tags.get("rank") is not None
        })
        if n_ranks > 1:
            parts.append(f"{n_ranks} ranks")
        if trace.devices:
            dev = trace.devices[0]
            suffix = f" ×{len(trace.devices)}" if len(trace.devices) > 1 else ""
            parts.append(f"{dev.name}{suffix}")
        parts.append(_fmt_ns(_trace_wall_ns(trace)))
        return Text("  ·  ".join(parts), style="dim")


class BottomBar(Static):
    """
    Replaces Textual's default `Footer()` (reverse-video key chips) with a
    plain "key  description" hint row. Text is swapped per active tab (see
    ProfilerApp.on_tabbed_content_tab_activated / _TAB_HINTS) so the hints
    shown are always ones that actually do something on the current tab.
    """

    DEFAULT_CSS = "BottomBar { height: 1; background: $boost; padding: 0 1; }"

    _GLOBAL_HINTS = [("?", "help"), ("1-7", "jump tab"), ("tab", "cycle"), ("q", "quit")]

    def show_hints(self, extra: list[tuple[str, str]]) -> None:
        text = Text()
        for i, (key, desc) in enumerate(list(extra) + self._GLOBAL_HINTS):
            if i:
                text.append("    ")
            text.append(key, style="bold bright_cyan")
            text.append(f" {desc}", style="dim")
        self.update(text)


# Per-tab key hints shown in BottomBar, prepended to BottomBar._GLOBAL_HINTS
# -- keyed by TabPane id, kept next to ProfilerApp.compose()'s tab wiring so
# the two stay easy to update together.
_TAB_HINTS: dict[str, list[tuple[str, str]]] = {
    "tab-overview": [],
    # Timeline shows its own live scroll/zoom/offset footer inside the
    # widget itself (with real-time state the static hint row below can't
    # carry) -- repeating the same key list here would just duplicate it.
    "tab-timeline": [],
    "tab-kernels":  [("j/k", "navigate"), ("s", "sort"), ("/", "filter")],
    "tab-calltree": [("↑↓", "navigate"), ("enter", "expand")],
    "tab-roofline": [],
    "tab-source":   [("j/k", "select kernel")],
    "tab-system":   [("↑↓", "scroll")],
    "tab-profile":  [("↑↓", "scroll")],
}


class ProfilerApp(App):
    """Multi-backend profiler TUI."""

    CSS = """
    Screen { background: $surface; }
    TabbedContent { height: 1fr; }
    TabPane { padding: 1 1; }
    #system-scroll, #profile-scroll {
        height: 1fr;
        border: round $primary;
        padding: 0 1;
    }
    """

    BINDINGS = [
        Binding("q",             "quit",         "Quit"),
        Binding("question_mark", "help",         "Help"),
        Binding("s",             "cycle_sort",   "Sort",   show=False),
        Binding("slash",         "focus_filter", "Filter", show=False),
        Binding("1", "goto_tab(1)", "1", show=False),
        Binding("2", "goto_tab(2)", "2", show=False),
        Binding("3", "goto_tab(3)", "3", show=False),
        Binding("4", "goto_tab(4)", "4", show=False),
        Binding("5", "goto_tab(5)", "5", show=False),
        Binding("6", "goto_tab(6)", "6", show=False),
        Binding("7", "goto_tab(7)", "7", show=False),
    ]

    TITLE = "hprofiler"

    def __init__(self, trace: Trace, collect_disasm: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.trace = trace
        self._collect_disasm = collect_disasm
        # Populated by compose(), in display order -- lets action_goto_tab
        # map digit keys to whichever tabs actually got composed (Call
        # Tree/Roofline/Source are conditional), instead of hardcoding ids
        # that could shift depending on what this trace contains.
        self._tab_ids: list[str] = []

    def compose(self) -> ComposeResult:
        yield TopBar(self.trace, id="top-bar")

        with TabbedContent(id="main-tabs"):
            with TabPane("1 Overview", id="tab-overview"):
                yield DashboardWidget(self.trace)
            self._tab_ids.append("tab-overview")

            with TabPane("2 Timeline", id="tab-timeline"):
                yield TimelineWidget(self.trace)
            self._tab_ids.append("tab-timeline")

            with TabPane("3 Kernels", id="tab-kernels"):
                yield HotspotsWidget(self.trace, id="hotspots")
            self._tab_ids.append("tab-kernels")

            n = 3
            if self.trace._has_stacks:
                n += 1
                with TabPane(f"{n} Call Tree", id="tab-calltree"):
                    yield CallTreeWidget(self.trace)
                self._tab_ids.append("tab-calltree")

            if _has_roofline_data(self.trace):
                n += 1
                with TabPane(f"{n} Roofline", id="tab-roofline"):
                    yield RooflineWidget(self.trace)
                self._tab_ids.append("tab-roofline")

            if self._collect_disasm or self.trace.disasm:
                n += 1
                with TabPane(f"{n} Source", id="tab-source"):
                    yield DisasmWidget(self.trace, id="disasm")
                self._tab_ids.append("tab-source")

            n += 1
            with TabPane(f"{n} System", id="tab-system"):
                with ScrollableContainer(id="system-scroll"):
                    yield SystemWidget(self.trace)
            self._tab_ids.append("tab-system")

            n += 1
            with TabPane(f"{n} Profile", id="tab-profile"):
                with ScrollableContainer(id="profile-scroll"):
                    yield ProfileWidget(self.trace)
            self._tab_ids.append("tab-profile")

        yield BottomBar(id="bottom-bar")

    def on_mount(self) -> None:
        try:
            self.query_one("#system-scroll").border_title = "System"
            self.query_one("#profile-scroll").border_title = "Profile"
        except Exception:
            pass
        self._update_hints("tab-overview")

    def on_tabbed_content_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        self._update_hints(event.pane.id)

    def _update_hints(self, tab_id: str | None) -> None:
        try:
            self.query_one("#bottom-bar", BottomBar).show_hints(_TAB_HINTS.get(tab_id or "", []))
        except Exception:
            pass

    def action_goto_tab(self, n: int) -> None:
        if 1 <= n <= len(self._tab_ids):
            self.query_one("#main-tabs", TabbedContent).active = self._tab_ids[n - 1]

    def action_help(self) -> None:
        self.push_screen(HelpScreen())

    def action_cycle_sort(self) -> None:
        try:
            self.query_one("#hotspots", HotspotsWidget).action_next_sort()
        except Exception:
            pass

    def action_focus_filter(self) -> None:
        try:
            self.query_one("#hs-filter", Input).focus()
        except Exception:
            pass


# ── Public API ────────────────────────────────────────────────────────────────

def launch_viewer(trace: Trace, collect_disasm: bool = False) -> None:
    ProfilerApp(trace, collect_disasm=collect_disasm).run()


# Moved to output/chrome_trace.py (the read side of the format write()
# there produces) so loading a trace doesn't require importing this whole
# Textual-based module -- re-exported here so existing call sites/tests
# using `from src.ui.app import load_trace_from_json` keep working.
from ..output.chrome_trace import load_trace_from_json  # noqa: E402
