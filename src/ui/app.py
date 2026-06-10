"""
Profiler TUI — redesigned, inspired by Paraver and VTune.

Tabs:
  Overview  — dashboard: duration, backend breakdown, top-10 hotspots
  Timeline  — Paraver-style Gantt: one row per (category, thread), zoomable
  Hotspots  — VTune-style: function table with bar visualization, sortable
  Flame     — ASCII flame graph of CPU samples

Keyboard shortcuts:
  Tab / Shift+Tab  cycle tabs
  ← →              scroll timeline
  + / -            zoom timeline
  r                reset timeline zoom
  ↑ ↓              navigate hotspot rows
  s                cycle sort column (hotspots)
  /                focus name filter (hotspots)
  ?                toggle help overlay
  q                quit
"""

from __future__ import annotations
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from rich.text import Text
from rich.panel import Panel

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import ScrollableContainer, Horizontal, Vertical
from textual.events import MouseMove
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widget import Widget
from textual.widgets import (
    Header, Footer, TabbedContent, TabPane,
    DataTable, Input, Static, RichLog, Tree,
)

from ..core.trace import Trace
from ..core.events import SpanEvent, Category

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

def _fmt_ns(ns: float) -> str:
    if ns >= 1_000_000_000:
        return f"{ns / 1e9:.3f}s"
    if ns >= 1_000_000:
        return f"{ns / 1e6:.2f}ms"
    if ns >= 1_000:
        return f"{ns / 1e3:.1f}µs"
    return f"{ns:.0f}ns"

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

def _fmt_bytes(b: float) -> str:
    if b >= 1024**3:
        return f"{b/1024**3:.2f} GB"
    if b >= 1024**2:
        return f"{b/1024**2:.1f} MB"
    if b >= 1024:
        return f"{b/1024:.0f} KB"
    return f"{b:.0f} B"

def _fmt_tf(tf: float) -> str:
    """Format TFLOPs value compactly."""
    if tf >= 1000:
        return f"{tf/1000:.1f} PF"
    if tf >= 1:
        return f"{tf:.1f} TF"
    if tf >= 0.001:
        return f"{tf*1000:.0f} GF"
    return f"{tf:.2g} TF"


# ── Help overlay ──────────────────────────────────────────────────────────────

_HELP_TEXT = """\
[bold]Keyboard Shortcuts[/bold]

[bold]Global[/bold]
  [yellow]Tab / Shift+Tab[/yellow]  switch tabs
  [yellow]q[/yellow]                quit
  [yellow]?[/yellow]                this help

[bold]Timeline tab[/bold]
  [yellow]← →[/yellow]             scroll horizontally
  [yellow]↑ ↓[/yellow]             pan up / down (when lanes overflow)
  [yellow]+ -[/yellow]             zoom in / out
  [yellow]r[/yellow]               reset zoom, scroll & pan

[bold]Hotspots tab[/bold]
  [yellow]↑ ↓[/yellow]             navigate rows
  [yellow]s[/yellow]               cycle sort column
  [yellow]/[/yellow]               focus name filter

[bold]Disasm tab[/bold]  (only shown when [yellow]--disasm[/yellow] is passed)
  [yellow]↑ ↓[/yellow]             select kernel
  Left pane: kernel list with arch + timing
  Right pane: annotated assembly
  Bottom: instruction mix (vec/scl/mem/ctl)

  Colours:
    [bright_green]vec[/bright_green] SIMD/AVX/YMM/ZMM   [cyan]scl[/cyan] scalar ALU
    [yellow]mem[/yellow] load/store        [magenta]ctl[/magenta] branch/call
    [bright_blue]fma[/bright_blue] FMA/multiply-acc  [red]syn[/red] barrier/fence

[bold]Roofline[/bold]
  Use [yellow]hprofiler roofline --backend <backend> -- ./app[/yellow] for hardware-counter
  roofline analysis (ncu for CUDA, perf stat for CPU/OpenMP).

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
        width: 58; height: auto;
        padding: 1 2;
        background: $surface;
        border: solid $accent;
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
        _kv("Duration", f"[yellow]{_fmt_ns(trace.duration_ns)}[/yellow]")
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
        wall_ns = trace.duration_ns or 1
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
            kern_ns  = sum(s.duration_ns for s in kspans)
            sync_ns  = sum(s.duration_ns for s in spans if s.category.value == "sync")
            pct      = 100.0 * kern_ns / wall_ns
            sync_pct = 100.0 * sync_ns / wall_ns
            avg_ns   = kern_ns / len(kspans)
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
                f"  [dim]Sync overhead    {sync_pct:.1f}% of wall time[/dim]"
            )
            L.append(
                f"  [dim]GPU efficiency   {eff:.0f}%  (kernel / (kernel + sync))[/dim]"
            )
            L.append("")
            L.append(
                f"  [dim]{len(kspans)} kernel launches"
                f"  ·  {_fmt_ns(avg_ns)} average"
                f"  ·  {_fmt_ns(kern_ns)} total[/dim]"
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
        try:
            from ..output.summary import _bottleneck_analysis
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


# ── Overview tab (legacy wrapper kept for direct write() callers) ──────────────

class OverviewWidget(Static):
    """Dashboard: key metrics, time-by-backend bars, top-10 hotspots."""

    def __init__(self, trace: Trace, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._trace = trace

    def render(self) -> Any:  # noqa: ANN401
        import time as _time
        trace = self._trace
        meta  = trace.metadata
        spans = trace.spans

        # Hard limit: every rendered line must stay ≤ this many visible chars.
        # Rich markup tags are invisible but Python f-string padding counts them,
        # so we keep content short and never use markup inside padded fields.
        _LIM = 72

        L: list[str] = []

        def _sec(title: str, icon: str = "◈") -> None:
            pad = _LIM - len(title) - 3
            L.append(
                f"[bold bright_cyan]{icon} {title}[/bold bright_cyan]"
                f"[dim] {'─' * pad}[/dim]"
            )

        # ── Header ────────────────────────────────────────────────────────
        ts      = _time.strftime("%Y-%m-%d %H:%M:%S")
        cmd_str = f"{meta.command} {' '.join(meta.args[:3])}"
        if len(cmd_str) > 46: cmd_str = cmd_str[:43] + "…"
        host    = (meta.hostname or "")[:20]
        backs_plain = "  ".join(meta.backends_used or ["none"])
        backs_rich  = "  ".join(
            f"[{_cat_color(b)}]{b}[/{_cat_color(b)}]"
            for b in (meta.backends_used or ["none"])
        )
        dur_str = _fmt_ns(trace.duration_ns)

        L.append(
            f"  [bold bright_white]◆ HPROFILER[/bold bright_white]"
            f"  [dim cyan]{ts}[/dim cyan]"
            f"  [dim]·  {host}[/dim]"
        )
        L.append(
            f"  [cyan]{cmd_str}[/cyan]"
        )
        L.append(
            f"  [bold yellow]{dur_str}[/bold yellow]"
            f"  {backs_rich}"
            f"  [dim]·  {len(spans)} spans"
            f"  ·  {len(trace.instants)} instants"
            f"  ·  {len(trace.counters)} counters[/dim]"
        )
        L.append(f"  [dim bright_cyan]{'─' * (_LIM - 2)}[/dim bright_cyan]")
        L.append("")

        # ── Collect counters ──────────────────────────────────────────────
        ctrs_last: dict[str, float] = {}
        gpu_util_peak: dict[str, float] = {}
        gpu_mem_peak:  dict[str, float] = {}
        gpu_util_series: dict[str, list] = defaultdict(list)
        for c in trace.counters:
            ctrs_last[c.name] = c.value
            if c.name.startswith("gpu_utilization_pct"):
                gpu_util_peak[c.name] = max(gpu_util_peak.get(c.name, 0.0), c.value)
                gpu_util_series[c.name].append(c.value)
            if c.name.startswith("gpu_mem_used_bytes"):
                gpu_mem_peak[c.name] = max(gpu_mem_peak.get(c.name, 0.0), c.value)

        wall_ns = trace.duration_ns or 1
        devices = trace.devices

        # ── DEVICES ───────────────────────────────────────────────────────
        if devices:
            _sec("DEVICES", "◈")
            for idx, dev in enumerate(devices):
                bk_col   = _cat_color(dev.backend)
                sm_lbl   = "SMs" if dev.backend in ("cuda", "rocm") else "cores"
                ridge_hint = (
                    "[red]mem-bound[/red]"    if dev.ridge_point < 5  else
                    "[yellow]balanced[/yellow]"  if dev.ridge_point < 30 else
                    "[green]compute-bound[/green]"
                ) if dev.ridge_point > 0 else ""

                # line 1: name + topology
                L.append(
                    f"  [{bk_col}]GPU {idx}[/{bk_col}]"
                    f"  [bold]{dev.name}[/bold]"
                    f"  [dim]cap {dev.compute_cap or '?'}"
                    f"  {dev.sm_count} {sm_lbl}[/dim]"
                )
                # line 2: FP peaks (only non-zero)
                peaks = "  ".join(filter(None, [
                    f"[bright_green]FP32 {_fmt_tf(dev.fp32_tflops)}[/bright_green]" if dev.fp32_tflops else "",
                    f"[red]FP64 {_fmt_tf(dev.fp64_tflops)}[/red]"                   if dev.fp64_tflops else "",
                    f"[magenta]FP16 {_fmt_tf(dev.fp16_tflops)}[/magenta]"            if dev.fp16_tflops else "",
                    f"[cyan]Tensor {_fmt_tf(dev.tensor_tflops)}[/cyan]"              if dev.tensor_tflops > 0 else "",
                ]))
                L.append(f"       {peaks}")
                # line 3: memory
                mem = "  ".join(filter(None, [
                    f"[bright_cyan]BW {dev.bandwidth_gbs:.0f} GB/s[/bright_cyan]" if dev.bandwidth_gbs else "",
                    f"[blue]VRAM {dev.vram_gb:.1f} GB[/blue]"                      if dev.vram_gb > 0  else "",
                    f"[dim]ridge {dev.ridge_point:.0f} F/B[/dim]"                  if dev.ridge_point > 0 else "",
                    ridge_hint,
                ]))
                L.append(f"       {mem}")
            L.append("")

        # ── PERFORMANCE HEALTH ────────────────────────────────────────────
        _sec("PERFORMANCE HEALTH", "◈")
        BAR = 22  # bar width kept short so stats fit on same line

        for cat_val, label in (("cuda", "CUDA"), ("rocm", "ROCm")):
            kspans = [s for s in spans
                      if s.category.value == cat_val and s.tags.get("type") == "kernel"]
            if not kspans:
                continue
            kern_ns    = sum(s.duration_ns for s in kspans)
            pct        = 100.0 * kern_ns / wall_ns
            color      = _cat_color(cat_val)
            grade, gc  = _grade(pct)
            bar        = _grad_bar(pct / 100, BAR)
            avg_ns     = kern_ns / len(kspans)
            sync_ns    = sum(s.duration_ns for s in spans if s.category.value == "sync")
            sync_pct   = 100.0 * sync_ns / wall_ns
            eff        = pct / (pct + sync_pct) * 100 if (pct + sync_pct) > 0 else 0
            # line 1: bar + pct + grade
            L.append(
                f"  [bold]{label} Active[/bold]"
                f"  [{color}]{bar}[/{color}]"
                f"  [yellow]{pct:.1f}%[/yellow]"
                f"  [{gc}]{grade}[/{gc}]"
            )
            # line 2: stats indented under the bar
            L.append(
                f"  [dim]  {len(kspans)}×"
                f"  total {_fmt_ns(kern_ns)}"
                f"  avg {_fmt_ns(avg_ns)}"
                f"  sync {sync_pct:.1f}%"
                f"  eff {eff:.0f}%[/dim]"
            )

        # CPU microarch — each metric on one concise line
        ipc        = ctrs_last.get("ipc",            0.0)
        cache_miss = ctrs_last.get("cache_miss_pct", -1.0)
        br_miss    = ctrs_last.get("branch_miss_pct",-1.0)

        if ipc > 0:
            ipc_col  = "bright_green" if ipc >= 2 else ("yellow" if ipc >= 1 else "red")
            ipc_hint = "excellent" if ipc >= 3 else ("good" if ipc >= 2 else
                       "ok" if ipc >= 1 else "stalled")
            L.append(
                f"  [bold]IPC[/bold]"
                f"  [{ipc_col}]{_grad_bar(min(ipc/4.0,1.0), BAR)}[/{ipc_col}]"
                f"  [{ipc_col}]{ipc:.2f}[/{ipc_col}]"
                f"  [dim]{ipc_hint}[/dim]"
            )
        if cache_miss >= 0:
            cm_col  = "green" if cache_miss < 5 else ("yellow" if cache_miss < 20 else "red")
            cm_hint = "hot" if cache_miss < 5 else ("warm" if cache_miss < 20 else "thrashing!")
            L.append(
                f"  [bold]LLC miss[/bold]"
                f"  [{cm_col}]{_grad_bar(min(cache_miss/50.0,1.0), BAR)}[/{cm_col}]"
                f"  [{cm_col}]{cache_miss:.1f}%[/{cm_col}]"
                f"  [dim]{cm_hint}[/dim]"
            )
        if br_miss >= 0:
            bm_col  = "green" if br_miss < 1 else ("yellow" if br_miss < 5 else "red")
            bm_hint = "predictable" if br_miss < 1 else ("ok" if br_miss < 5 else "poor")
            L.append(
                f"  [bold]Branch[/bold]"
                f"  [{bm_col}]{_grad_bar(min(br_miss/20.0,1.0), BAR)}[/{bm_col}]"
                f"  [{bm_col}]{br_miss:.1f}%[/{bm_col}]"
                f"  [dim]{bm_hint}[/dim]"
            )

        # Memory / GPU util — one compact line
        rss    = ctrs_last.get("process_max_rss_bytes", 0.0)
        leaked = ctrs_last.get("gpu_memory_leaked_bytes", 0.0)
        mem_parts: list[str] = []
        if rss > 0:
            mem_parts.append(f"[bold]RSS[/bold] [yellow]{_fmt_bytes(rss)}[/yellow]")
        for key, val in sorted(gpu_util_peak.items()):
            lbl = key.replace("gpu_utilization_pct", "").strip("[]") or "0"
            gu_bar = _grad_bar(val / 100, 10)
            mem_parts.append(
                f"[bold]GPU{lbl}[/bold] [cyan]{gu_bar}[/cyan] [yellow]{val:.0f}%[/yellow]"
            )
        for key, val in sorted(gpu_mem_peak.items()):
            lbl = key.replace("gpu_mem_used_bytes", "").strip("[]") or "0"
            vt  = next((d.vram_gb * 1024**3 for d in devices if d.backend in ("cuda","rocm")), 0)
            mem_parts.append(
                f"[bold]VRAM{lbl}[/bold] [blue]{_grad_bar(val/vt if vt else 0, 10)}[/blue]"
                f" [yellow]{_fmt_bytes(val)}[/yellow]"
            )
        if leaked > 0:
            mem_parts.append(f"[bold bright_red]LEAK {_fmt_bytes(leaked)}[/bold bright_red]")
        if mem_parts:
            L.append("  " + "  ·  ".join(mem_parts))

        L.append("")

        # ── TIME BY BACKEND ───────────────────────────────────────────────
        _sec("TIME BY BACKEND", "◈")
        by_cat: dict[str, dict] = defaultdict(
            lambda: {"total_ns": 0, "count": 0, "dur_list": []}
        )
        for s in spans:
            by_cat[s.category.value]["total_ns"] += s.duration_ns
            by_cat[s.category.value]["count"]    += 1
            by_cat[s.category.value]["dur_list"].append(s.duration_ns)

        grand_total = sum(v["total_ns"] for v in by_cat.values()) or 1

        for cat, info in sorted(by_cat.items(), key=lambda kv: -kv[1]["total_ns"]):
            frac   = info["total_ns"] / grand_total
            color  = _cat_color(cat)
            dl     = sorted(info["dur_list"])
            avg_ns = info["total_ns"] / max(info["count"], 1)
            p99_ns = dl[min(int(len(dl)*0.99), len(dl)-1)] if dl else 0
            spark  = _sparkline(info["dur_list"][-14:], 7)
            # line 1: category  bar  %  total  count  sparkline
            L.append(
                f"  [{color}]{cat:<8}[/{color}]"
                f"  [{color}]{_grad_bar(frac, 20)}[/{color}]"
                f"  [yellow]{frac*100:5.1f}%[/yellow]"
                f"  [white]{_fmt_ns(info['total_ns']):>9}[/white]"
                f"  [dim]{info['count']:>4}×[/dim]"
                f"  [dim cyan]{spark}[/dim cyan]"
            )
            # line 2: timing stats (indented to align under bar)
            L.append(
                f"  [dim]          "
                f"avg {_fmt_ns(avg_ns):<10}"
                f"  p99 {_fmt_ns(p99_ns)}[/dim]"
            )

        if not by_cat:
            L.append("  [dim]No spans recorded.[/dim]")
        L.append("")

        # ── TOP HOTSPOTS ──────────────────────────────────────────────────
        _sec("TOP HOTSPOTS", "◈")
        stats = trace.aggregated_stats()
        if not stats:
            L.append("  [dim]No spans recorded.[/dim]")
        else:
            total_all = sum(r["total_ns"] for r in stats) or 1
            L.append(f"  [dim]{'─' * 68}[/dim]")

            for i, row in enumerate(stats[:12]):
                frac    = row["total_ns"] / total_all
                color   = _cat_color(row["category"])
                bar     = _grad_bar(frac, 10)
                # Truncate name to 28 chars — no padding (avoids column calc with markup)
                name    = row["name"][:28]
                avg_ns  = row["total_ns"] / max(row["count"], 1)
                grade, gc = _grade(row["pct"])
                cat_str   = row["category"][:6]
                # Line 1: index  name  bar  grade  (narrow — always fits)
                L.append(
                    f"  [dim]{i+1:>2}[/dim]"
                    f"  [bold {color}]{name}[/bold {color}]"
                    f"  [{color}]{bar}[/{color}]"
                    f"[{gc}]{grade}[/{gc}]"
                )
                # Line 2: indented stats — category / share / total / avg / count
                L.append(
                    f"      [dim][{cat_str}]"
                    f"  {row['pct']:5.1f}%"
                    f"  {_fmt_ns(row['total_ns']):>9}"
                    f"  avg {_fmt_ns(avg_ns)}"
                    f"  {row['count']}×[/dim]"
                )
        L.append("")

        # ── BOTTLENECK ADVISOR ────────────────────────────────────────────
        try:
            from ..output.summary import _bottleneck_analysis
            ctr_sub: dict[str, float] = {}
            for k in ("ipc", "cache_miss_pct", "branch_miss_pct"):
                if k in ctrs_last: ctr_sub[k] = ctrs_last[k]
            tips = _bottleneck_analysis(trace, ctr_sub)
            if tips:
                _sec("BOTTLENECK ADVISOR", "⚡")
                for tip in tips:
                    icon = tip[:2]
                    body = tip[2:].strip()
                    words, lines_w, cur = body.split(), [], ""
                    for w in words:
                        if len(cur) + len(w) + 1 > 64:
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

        return "\n".join(L)


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
        border: solid $primary;
        padding: 0 1;
    }
    TimelineWidget:focus {
        border: solid $accent;
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
        if all_spans:
            self._view_start = min(s.start_ns for s in all_spans)
            self._view_end   = max(s.end_ns   for s in all_spans)
        else:
            self._view_start = trace.metadata.start_time_ns
            self._view_end   = trace.metadata.end_time_ns or self._view_start + 1
        self._trace_dur = max(self._view_end - self._view_start, 1)

        # Per-function color map — encounter-order assignment avoids hash collisions.
        seen: dict[str, int] = {}
        for s in (all_spans if all_spans else []):
            if s.name not in seen:
                seen[s.name] = len(seen)
        self._func_colors: dict[str, str] = {
            name: _SPAN_PALETTE[idx % len(_SPAN_PALETTE)]
            for name, idx in seen.items()
        }
        # Integer palette index per function — used by the numpy render path.
        self._func_color_idx: dict[str, int] = {
            name: idx % len(_SPAN_PALETTE)
            for name, idx in seen.items()
        }

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

    def _lane_label(self, lane_name: str) -> str:
        parts = lane_name.split("/", 1)
        cat   = parts[0]
        abbr  = self._CAT_ABBREV.get(cat, cat[:4])
        count = self._lane_counts.get(lane_name, 0)

        if len(parts) > 1:
            suffix = parts[1]
            if suffix.startswith("thread-"):
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
                     _unused: str) -> tuple[Text, float]:
        """
        Return (rendered Text row, visible-window utilisation %).

        Fully vectorised — no Python loop over spans:
          1. Spatial index  : np.searchsorted clips to only the visible spans O(log n)
          2. Numpy broadcast: pixel positions computed for all spans at once
          3. Diff + cumsum  : interior pixel activity accumulated without loops
          4. searchsorted   : dominant-function color assigned to pixels in O(width)

        Renders each pixel as:
          █  solid block colored by the function dominating that column
          ·  dim dot for idle (no span coverage)
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
            return Text("·" * width, style="color(237)"), 0.0

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
            return Text("·" * width, style="color(237)"), 0.0

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

        # ── Build Rich Text row (run-length encoded by color) ─────────────
        row  = Text()
        IDLE = 0.05
        act  = activity          # local alias for speed
        i    = 0
        while i < width:
            if act[i] <= IDLE:
                j = i + 1
                while j < width and act[j] <= IDLE:
                    j += 1
                row.append("·" * (j - i), style="color(237)")
                i = j
            else:
                ci = int(dom_idx[i])
                c  = _SPAN_PALETTE[ci] if ci >= 0 else "white"
                j  = i + 1
                while j < width and act[j] > IDLE and int(dom_idx[j]) == ci:
                    j += 1
                row.append("█" * (j - i), style=c)
                i = j

        return row, util_pct

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
            return

        lane_name   = self._lane_names[actual_lane]
        sorted_spans = self._sorted_spans[lane_name]
        if not sorted_spans:
            self._hover = ""
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
        self._hover = hover

    def on_leave(self, _event: Any) -> None:
        self._hover = ""

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

        lanes_drawn = 0
        for lane_name in self._lane_names[lo:hi]:
            cat   = lane_name.split("/")[0]
            color = _cat_color(cat)

            # Data row
            out.append(f"{self._lane_label(lane_name)} ", style=f"bold {color}")
            density_row, util_pct = self._density_row(lane_name, width, color)
            out.append(density_row)
            util_col = ("bright_green" if util_pct >= 50
                        else "yellow"  if util_pct >= 20
                        else "red")
            out.append(f" {util_pct:4.0f}%", style=f"dim {util_col}")
            out.append("\n")

            # Blank spacer row — gives visual breathing room between lanes
            out.append("\n")
            lanes_drawn += 1

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
    s → cycle sort column; / → focus filter input; ↑↓ → navigate rows.
    """

    DEFAULT_CSS = """
    HotspotsWidget { height: 1fr; layout: vertical; }
    #hs-filter    { height: 3; dock: top; }
    #hs-table     { height: 1fr; }
    #hs-sort-hint { height: 1; dock: bottom; color: $text-muted; }
    """

    sort_idx:    reactive[int] = reactive(0)
    filter_text: reactive[str] = reactive("")

    def __init__(self, trace: Trace, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.trace = trace

    def compose(self) -> ComposeResult:
        yield Input(placeholder="  / filter by name…", id="hs-filter")
        yield DataTable(id="hs-table", cursor_type="row")
        yield Static("  [dim][s] cycle sort  [/] filter  [↑↓] navigate[/dim]", id="hs-sort-hint")

    def on_mount(self) -> None:
        self._rebuild()

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

        stats     = sorted(self.trace.aggregated_stats(),
                           key=lambda r: r.get(sort_col, 0), reverse=True)
        flt       = self.filter_text.lower()
        total_all = sum(r["total_ns"] for r in stats) or 1
        bar_w     = 24

        for row in stats:
            if flt and flt not in row["name"].lower():
                continue
            color = _cat_color(row["category"])
            frac  = row["total_ns"] / total_all
            bar   = _bar(frac, bar_w)

            dt.add_row(
                Text(f"  {row['name'][:46]}", style=f"bold {color}"),
                Text(row["category"],         style=color),
                str(row["count"]),
                _fmt_ns(row["total_ns"]),
                _fmt_ns(row["avg_ns"]),
                _fmt_ns(row["min_ns"]),
                _fmt_ns(row["max_ns"]),
                f"{row['pct']:.1f}%",
                Text(bar, style=f"bold {color}"),
                key=row["name"],
            )


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

@dataclass
class _CTNode:
    """Aggregated call-tree node: N spans with the same name at the same tree level."""
    name: str
    category: str
    total_ns: int
    count: int
    self_ns: int
    children: list["_CTNode"] = field(default_factory=list)

    @property
    def avg_ns(self) -> int:
        return self.total_ns // max(self.count, 1)


@dataclass
class _RawNode:
    span: SpanEvent
    children: list["_RawNode"] = field(default_factory=list)


def _ct_build_raw(spans: list[SpanEvent]) -> list[_RawNode]:
    """Containment tree for one thread via stack algorithm (O(n log n)).

    When spans carry explicit span_id / parent_span_id links (emitted by hooks),
    those take precedence over temporal containment for spans that stay within
    the same input set.
    """
    sorted_spans = sorted(spans, key=lambda s: s.start_ns)
    # Map from span_id → node for explicit linking.
    nodes_by_sid: dict[str, _RawNode] = {
        s.span_id: _RawNode(span=s)
        for s in sorted_spans
        if s.span_id
    }
    # Remaining spans without a span_id get plain nodes.
    all_nodes: dict[int, _RawNode] = {}  # id(span) → node
    for s in sorted_spans:
        if s.span_id and s.span_id in nodes_by_sid:
            all_nodes[id(s)] = nodes_by_sid[s.span_id]
        else:
            all_nodes[id(s)] = _RawNode(span=s)

    explicitly_parented: set[int] = set()  # id(span) of spans handled via explicit link
    for s in sorted_spans:
        if s.parent_span_id and s.parent_span_id in nodes_by_sid:
            parent_node = nodes_by_sid[s.parent_span_id]
            child_node  = all_nodes[id(s)]
            if child_node is not parent_node:
                parent_node.children.append(child_node)
                explicitly_parented.add(id(s))

    # Temporal containment for spans not already parented explicitly.
    stack: list[_RawNode] = []
    roots: list[_RawNode] = []
    for s in sorted_spans:
        if id(s) in explicitly_parented:
            continue
        while stack and stack[-1].span.end_ns <= s.start_ns:
            stack.pop()
        node = all_nodes[id(s)]
        if stack and stack[-1].span.end_ns >= s.end_ns:
            stack[-1].children.append(node)
        else:
            roots.append(node)
        stack.append(node)
    return roots


def _ct_aggregate(raw_nodes: list[_RawNode]) -> list[_CTNode]:
    """Recursively aggregate siblings by (name, category)."""
    groups: dict[tuple[str, str], list[_RawNode]] = defaultdict(list)
    for rn in raw_nodes:
        groups[(rn.span.name, rn.span.category.value)].append(rn)

    result: list[_CTNode] = []
    for (name, cat), nodes in groups.items():
        total_ns = sum(n.span.duration_ns for n in nodes)
        all_children: list[_RawNode] = []
        for n in nodes:
            all_children.extend(n.children)
        children = _ct_aggregate(all_children)
        self_ns = max(0, total_ns - sum(c.total_ns for c in children))
        result.append(_CTNode(
            name=name, category=cat,
            total_ns=total_ns, count=len(nodes),
            self_ns=self_ns,
            children=sorted(children, key=lambda c: -c.total_ns),
        ))
    return sorted(result, key=lambda n: -n.total_ns)


@dataclass
class _StackNode:
    """Mutable trie node used while building the stack-based call tree."""
    name: str
    category: str
    total_ns: int = 0
    count: int = 0          # spans that end here (leaves)
    children: "dict[str, _StackNode]" = field(default_factory=dict)

    def to_ctnode(self) -> "_CTNode":
        children = sorted(
            (c.to_ctnode() for c in self.children.values()),
            key=lambda n: -n.total_ns,
        )
        self_ns = max(0, self.total_ns - sum(c.total_ns for c in children))
        return _CTNode(
            name=self.name, category=self.category,
            total_ns=self.total_ns, count=self.count,
            self_ns=self_ns, children=children,
        )


def _ct_build_from_stacks(spans: list[SpanEvent]) -> list[_CTNode]:
    """Build call tree from captured CPU stack frames (from-main view)."""
    roots: dict[str, _StackNode] = {}

    for span in spans:
        # Frames arrive innermost-first (backtrace order); reverse to root-first.
        path = list(reversed(span.stack_frames))
        if not path:
            continue

        level = roots
        nodes_on_path: list[_StackNode] = []
        for i, frame in enumerate(path):
            is_leaf = (i == len(path) - 1)
            cat = span.category.value if is_leaf else "other"
            if frame not in level:
                level[frame] = _StackNode(name=frame, category=cat)
            node = level[frame]
            node.total_ns += span.duration_ns
            if is_leaf:
                node.count += 1
                # Promote category to the actual span category on the leaf.
                node.category = span.category.value
            nodes_on_path.append(node)
            level = node.children

        # Add the span itself as the innermost leaf (the actual API call / kernel).
        leaf_key = f"__leaf__{span.name}"
        if leaf_key not in level:
            level[leaf_key] = _StackNode(name=span.name, category=span.category.value)
        leaf = level[leaf_key]
        leaf.total_ns += span.duration_ns
        leaf.count    += 1

    return sorted((n.to_ctnode() for n in roots.values()), key=lambda n: -n.total_ns)


def _ct_build(spans: list[SpanEvent]) -> list[_CTNode]:
    """Call tree: uses CPU stack frames when available, temporal containment otherwise.

    Explicit parent_span_id links (from hooks) are preferred over temporal
    containment within the same thread, and are shown as cross-thread connections
    in the tree when the parent and child live on different threads.
    """
    duration_spans = [s for s in spans if s.duration_ns > 0]
    if not duration_spans:
        return []

    stacked = [s for s in duration_spans if s.stack_frames]
    if stacked:
        return _ct_build_from_stacks(stacked)

    # Per-thread temporal containment (explicit same-thread links handled inside).
    by_thread: dict[tuple[int, int], list[SpanEvent]] = defaultdict(list)
    for s in duration_spans:
        by_thread[(s.pid, s.tid)].append(s)

    if len(by_thread) == 1:
        return _ct_aggregate(_ct_build_raw(next(iter(by_thread.values()))))

    # Build per-thread roots first.
    thread_roots: dict[tuple[int, int], list[_CTNode]] = {}
    for (pid, tid), thread_spans in sorted(by_thread.items()):
        thread_roots[(pid, tid)] = _ct_aggregate(_ct_build_raw(thread_spans))

    # Identify spans that are explicit children of spans in a *different* thread.
    # We use the span_id → (pid, tid) map to detect cross-thread links.
    sid_to_thread: dict[str, tuple[int, int]] = {
        s.span_id: (s.pid, s.tid)
        for s in duration_spans
        if s.span_id
    }
    cross_thread_children: dict[tuple[int, int], list[SpanEvent]] = defaultdict(list)
    orphan_threads: set[tuple[int, int]] = set()
    for s in duration_spans:
        if s.parent_span_id:
            parent_thread = sid_to_thread.get(s.parent_span_id)
            my_thread = (s.pid, s.tid)
            if parent_thread and parent_thread != my_thread:
                cross_thread_children[parent_thread].append(s)
                orphan_threads.add(my_thread)

    all_roots: list[_CTNode] = []
    for (pid, tid), children in sorted(thread_roots.items()):
        total_ns = sum(c.total_ns for c in children)
        # Attach cross-thread children (e.g. GPU kernels under their NVTX parent thread).
        extra = cross_thread_children.get((pid, tid), [])
        if extra:
            extra_nodes = _ct_aggregate(_ct_build_raw(extra))
            children = children + extra_nodes
            total_ns = sum(c.total_ns for c in children)
        all_roots.append(_CTNode(
            name=f"Thread {tid} (pid {pid})",
            category="other",
            total_ns=total_ns, count=1, self_ns=0,
            children=sorted(children, key=lambda n: -n.total_ns),
        ))
    return sorted(all_roots, key=lambda n: -n.total_ns)


class CallTreeWidget(Widget):
    """
    Hierarchical call tree for all backends.
    Builds parent-child relationships from temporal containment of spans,
    then aggregates duplicate siblings by name for a compact view.
    Keys: ↑↓ navigate · Enter/Space expand · e expand all · u collapse all
    """

    DEFAULT_CSS = """
    CallTreeWidget { height: 1fr; layout: vertical; }
    #ct-tree      { height: 1fr; }
    #ct-hint      { height: 1; dock: bottom; color: $text-muted; }
    """

    def __init__(self, trace: Trace, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._trace = trace

    def compose(self) -> ComposeResult:
        yield Tree("Call Tree", id="ct-tree")
        yield Static(
            "  [dim][↑↓] navigate  [enter/space] expand  [e] expand all  [u] collapse all[/dim]",
            id="ct-hint",
        )

    def on_mount(self) -> None:
        self._rebuild()

    def _rebuild(self) -> None:
        tree: Tree = self.query_one("#ct-tree", Tree)  # type: ignore[type-arg]
        tree.clear()
        wall_ns = max(self._trace.duration_ns, 1)
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
    """

    can_focus = True

    BINDINGS = [
        Binding("up",   "prev_kernel", "↑", show=False),
        Binding("down", "next_kernel", "↓", show=False),
    ]

    _sel: reactive[int] = reactive(0)

    def __init__(self, trace: Trace, **kwargs: Any) -> None:
        super().__init__(**kwargs)
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
        yield Static("", id="disasm-mix")

    def on_mount(self) -> None:
        self._rebuild_kernel_list()
        self._show_disasm()
        self._show_mix()
        self._last_disasm_count = 0
        # Poll until background disasm collection finishes, then refresh.
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
            dt.add_row(
                Text(f"{has_d} {name[:22]}", style=f"bold {color}"),
                Text(arch, style="dim"),
                Text(total),
                key=name,
            )

    # ── Background disasm poller ──────────────────────────────────────────────

    def _poll_disasm_ready(self) -> None:
        """Refresh kernel list and current disasm once background collection completes."""
        disasm = self._trace.disasm
        cur_count = len(disasm)
        # Rebuild whenever disasm grows — covers both new kernel names (JIT kernels
        # that weren't in the profiled list) and existing names that now have arch
        # data (e.g. OpenMP spans that were profiled but disasm arrived later).
        if cur_count <= self._last_disasm_count:
            return
        self._last_disasm_count = cur_count
        stats = self._trace.aggregated_stats()
        profiled_names = [r["name"] for r in stats]
        disasm_only = [n for n in disasm if n not in profiled_names]
        self._kernel_names = profiled_names + disasm_only
        self._stats_by_name = {r["name"]: r for r in stats}
        self._rebuild_kernel_list()
        self._show_disasm()
        self._show_mix()

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
            msg.append("\n\nNo disassembly available for this kernel.\n\n"
                       "Tips:\n"
                       "  • For CUDA AoT: install cuobjdump (CUDA toolkit)\n"
                       "  • For CPU/ACPP: objdump must be installed\n"
                       "  • For ROCm: install llvm-objdump\n"
                       "  • JIT cubins are auto-captured when the CUDA hook is loaded",
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

        if not kd.lines:
            body = Text("(no instructions decoded)", style="dim")
            self._disasm_cache[name] = [hdr, sep, body]
            for obj in self._disasm_cache[name]:
                log.write(obj)
            return

        _MAX_LINES = 500
        addr_w, mne_w, ops_w = 8, 12, 36

        # Build a single Text object with style spans — avoids all markup
        # parsing overhead and is an order of magnitude faster than
        # per-line log.write() calls or markup string formatting.
        body = Text()
        for ln in kd.lines[:_MAX_LINES]:
            color = ITYPE_COLOR.get(ln.itype, "grey70")
            label = ITYPE_LABEL.get(ln.itype, "   ")

            addr_s = f"{ln.addr:0{addr_w}x}" if ln.addr else " " * addr_w
            mne_s  = f"{ln.mnemonic:<{mne_w}}"[:mne_w]
            ops_s  = f"{ln.operands:<{ops_w}}"[:ops_w]

            body.append(addr_s + "  ", style="dim")
            body.append(mne_s + "  " + ops_s, style=color)
            body.append("  " + label, style=color)
            if ln.comment:
                body.append(f"  ; {ln.comment}", style="dim")
            body.append("\n")

        if len(kd.lines) > _MAX_LINES:
            body.append(
                f"… {len(kd.lines) - _MAX_LINES} more instructions not shown",
                style="dim"
            )

        self._disasm_cache[name] = [hdr, sep, body]
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

    # ── Navigation ───────────────────────────────────────────────────────────

    def _move(self, delta: int) -> None:
        if not self._kernel_names:
            return
        self._sel = max(0, min(len(self._kernel_names) - 1, self._sel + delta))
        dt: DataTable = self.query_one("#disasm-kernels", DataTable)
        dt.move_cursor(row=self._sel)
        self._show_disasm()
        self._show_mix()

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
        except (ValueError, AttributeError):
            pass


# ── Main App ──────────────────────────────────────────────────────────────────

class ProfilerApp(App):
    """Multi-backend profiler TUI."""

    CSS = """
    Screen { background: $surface; }
    TabbedContent { height: 1fr; }
    TabPane { padding: 1 2; }
    .status-bar {
        height: 3;
        background: $boost;
        padding: 0 2;
        content-align: left middle;
        color: $text;
    }
    HotspotsWidget   { height: 1fr; }
    DisasmWidget     { height: 1fr; }
    #system-scroll   { height: 1fr; }
    #profile-scroll  { height: 1fr; }
    #overview-scroll { height: 1fr; }
    """

    BINDINGS = [
        Binding("q",             "quit",         "Quit"),
        Binding("question_mark", "help",         "Help"),
        Binding("s",             "cycle_sort",   "Sort",   show=False),
        Binding("slash",         "focus_filter", "Filter", show=False),
    ]

    TITLE = "Profiler"

    def __init__(self, trace: Trace, collect_disasm: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.trace = trace
        self._collect_disasm = collect_disasm

    def compose(self) -> ComposeResult:
        meta  = self.trace.metadata
        cmd   = f"{meta.command} {' '.join(meta.args[:2])}"
        yield Header(name=f"Profiler — {cmd[:60]}")

        with TabbedContent():
            with TabPane("System", id="tab-system"):
                with ScrollableContainer(id="system-scroll"):
                    yield SystemWidget(self.trace)

            with TabPane("Profile", id="tab-profile"):
                with ScrollableContainer(id="profile-scroll"):
                    yield ProfileWidget(self.trace)

            with TabPane("Timeline  [T]", id="tab-timeline"):
                yield TimelineWidget(self.trace)

            with TabPane("Hotspots  [H]", id="tab-hotspots"):
                yield HotspotsWidget(self.trace, id="hotspots")

            if self.trace._has_stacks:
                with TabPane("Call Tree  [C]", id="tab-calltree"):
                    yield CallTreeWidget(self.trace)

            if self.trace._has_cpu:
                with TabPane("Flame  [F]", id="tab-flame"):
                    with ScrollableContainer():
                        yield FlameGraphWidget(self.trace)

            if self._collect_disasm:
                with TabPane("Disasm  [D]", id="tab-disasm"):
                    yield DisasmWidget(self.trace, id="disasm")

        yield Static(self._status_bar(), classes="status-bar")
        yield Footer()

    def _status_bar(self) -> str:
        meta  = self.trace.metadata
        dur   = _fmt_ns(self.trace.duration_ns)
        backs = "  ".join(
            f"[{_cat_color(b)}]■ {b}[/{_cat_color(b)}]"
            for b in meta.backends_used
        ) or "[dim]none[/dim]"
        return (
            f"  [bold]Duration:[/bold] [yellow]{dur}[/yellow]    "
            f"[bold]Spans:[/bold] {len(self.trace.spans)}    "
            f"{backs}    "
            f"[dim][?] help[/dim]"
        )

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


def load_trace_from_json(path: str | Path, collect_disasm: bool = False) -> Trace:
    """Reconstruct a Trace from a Chrome Trace JSON file."""
    from ..core.trace import Trace, TraceMetadata
    from ..core.events import SpanEvent, InstantEvent, CounterEvent, Category

    with open(path) as f:
        data = json.load(f)

    meta_raw = data.get("metadata", {})
    metadata = TraceMetadata(
        command=meta_raw.get("command", ""),
        args=meta_raw.get("args", []),
        backends_used=meta_raw.get("backends", []),
        hostname=meta_raw.get("hostname", ""),
        cwd=meta_raw.get("cwd", ""),
    )
    trace = Trace(metadata)

    for ev in data.get("traceEvents", []):
        ph      = ev.get("ph", "")
        cat_str = ev.get("cat", "other")
        try:
            cat = Category(cat_str)
        except ValueError:
            cat = Category.OTHER

        if ph == "X":
            args = ev.get("args", {})
            stack = args.pop("_stack", [])
            trace.add(SpanEvent(
                name=ev.get("name", ""),
                category=cat,
                start_ns=int(ev.get("ts", 0) * 1_000),
                duration_ns=int(ev.get("dur", 0) * 1_000),
                pid=ev.get("pid", 0),
                tid=ev.get("tid", 0),
                tags=args,
                stack_frames=stack if isinstance(stack, list) else [],
            ))
        elif ph == "i":
            trace.add(InstantEvent(
                name=ev.get("name", ""),
                category=cat,
                timestamp_ns=int(ev.get("ts", 0) * 1_000),
                pid=ev.get("pid", 0),
                tid=ev.get("tid", 0),
            ))
        elif ph == "C":
            args = ev.get("args", {})
            name = ev.get("name", "counter")
            val  = list(args.values())[0] if args else 0
            trace.add(CounterEvent(
                name=name,
                category=cat,
                timestamp_ns=int(ev.get("ts", 0) * 1_000),
                value=float(val),
                pid=ev.get("pid", 0),
            ))

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
                    )
                    for ln in kd_raw.get("lines", [])
                ]
                trace.add_disasm(KernelDisasm(
                    name=name,
                    arch=kd_raw.get("arch", ""),
                    source=kd_raw.get("source", ""),
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
