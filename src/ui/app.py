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
from pathlib import Path
from typing import Any

from rich.text import Text
from rich.panel import Panel

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import ScrollableContainer, Horizontal, Vertical
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widget import Widget
from textual.widgets import (
    Header, Footer, TabbedContent, TabPane,
    DataTable, Input, Static, RichLog,
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
    "memory":  "blue",
    "sync":    "white",
    "jit":     "purple",
    "nvtx":    "orange3",
    "other":   "grey70",
}

def _cat_color(cat: str) -> str:
    return _CAT_RICH.get(cat, "grey70")

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


# ── Overview tab ──────────────────────────────────────────────────────────────

class OverviewWidget(Static):
    """Dashboard: key metrics, time-by-backend bars, top-10 hotspots."""

    def __init__(self, trace: Trace, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._trace = trace

    def render(self) -> Any:  # noqa: ANN401
        trace = self._trace
        meta  = trace.metadata
        spans = trace.spans

        lines: list[str] = []

        # ── Run info ──────────────────────────────────────────────────────
        cmd_str = f"{meta.command} {' '.join(meta.args[:3])}"
        if len(cmd_str) > 56:
            cmd_str = cmd_str[:53] + "…"
        backs = ", ".join(meta.backends_used) or "none"
        dur   = _fmt_ns(trace.duration_ns)

        lines.append(
            f"[bold white]Command:[/bold white] [cyan]{cmd_str}[/cyan]   "
            f"[bold white]Host:[/bold white] [dim]{meta.hostname}[/dim]"
        )
        lines.append(
            f"[bold white]Duration:[/bold white] [yellow]{dur}[/yellow]   "
            f"[bold white]Backends:[/bold white] {backs}   "
            f"[bold white]Spans:[/bold white] {len(spans)}   "
            f"[bold white]Instants:[/bold white] {len(trace.instants)}   "
            f"[bold white]Counters:[/bold white] {len(trace.counters)}"
        )
        lines.append("")

        # ── Device characteristics ────────────────────────────────────────
        devices = trace.devices
        if devices:
            lines.append(
                "[bold]── Device Characteristics ──────────────────────────────────────[/bold]"
            )
            for dev in devices:
                bk_color = _cat_color(dev.backend)
                tag      = f"[{bk_color}][{dev.backend}][/{bk_color}]"
                cap      = f"  cap {dev.compute_cap}" if dev.compute_cap else ""
                sm_label = "SMs" if dev.backend in ("cuda", "rocm") else "cores"
                clk      = f"{dev.core_clock_ghz:.2f} GHz" if dev.core_clock_ghz else ""
                fp32     = f"FP32 {_fmt_tf(dev.fp32_tflops)}" if dev.fp32_tflops else ""
                fp64     = f"FP64 {_fmt_tf(dev.fp64_tflops)}" if dev.fp64_tflops else ""
                tc       = (f"TC {_fmt_tf(dev.tensor_tflops)}"
                            if dev.tensor_tflops > 0 else "")
                bw       = (f"BW {dev.bandwidth_gbs:.0f} GB/s"
                            if dev.bandwidth_gbs else "")
                vram     = (f"VRAM {dev.vram_gb:.1f} GB"
                            if dev.vram_gb > 0 else "")
                ridge    = (f"ridge {dev.ridge_point:.0f} FLOPs/B"
                            if dev.ridge_point > 0 else "")

                details = "  ".join(filter(None, [
                    f"{dev.sm_count} {sm_label}", clk, fp32, fp64, tc, bw, vram, ridge
                ]))
                lines.append(
                    f"  {tag} [bold]{dev.name}[/bold]{cap}\n"
                    f"       [dim]{details}[/dim]"
                )
            lines.append("")

        # ── Run health ────────────────────────────────────────────────────
        # Collect counter events into dicts for O(1) lookup.
        ctrs_last: dict[str, float] = {}
        gpu_util_peak: dict[str, float] = {}
        gpu_mem_peak:  dict[str, float] = {}
        for c in trace.counters:
            ctrs_last[c.name] = c.value
            if c.name.startswith("gpu_utilization_pct"):
                gpu_util_peak[c.name] = max(gpu_util_peak.get(c.name, 0.0), c.value)
            if c.name.startswith("gpu_mem_used_bytes"):
                gpu_mem_peak[c.name] = max(gpu_mem_peak.get(c.name, 0.0), c.value)

        wall_ns = trace.duration_ns or 1
        health_lines: list[str] = []

        # GPU kernel active % computed from span durations (exact, not polling).
        for cat_val, label in (("cuda", "CUDA"), ("rocm", "ROCm")):
            kernel_spans = [s for s in spans
                            if s.category.value == cat_val
                            and s.tags.get("type") == "kernel"]
            if kernel_spans:
                kern_ns = sum(s.duration_ns for s in kernel_spans)
                pct     = 100.0 * kern_ns / wall_ns
                color   = _cat_color(cat_val)
                bar     = _bar(pct / 100, 20)
                health_lines.append(
                    f"  [bold white]{label} kernel active[/bold white]  "
                    f"[{color}]{bar}[/{color}]  "
                    f"[yellow]{pct:.2f}%[/yellow] of wall time  "
                    f"[dim]({_fmt_ns(kern_ns)}, {len(kernel_spans)} launches)[/dim]"
                )

        rss = ctrs_last.get("process_max_rss_bytes", 0.0)
        if rss > 0:
            health_lines.append(
                f"  [bold white]Peak process RSS   [/bold white]  "
                f"[yellow]{_fmt_bytes(rss)}[/yellow]"
            )

        if gpu_util_peak:
            for key, val in sorted(gpu_util_peak.items()):
                lbl = key.replace("gpu_utilization_pct", "").strip("[]") or "0"
                health_lines.append(
                    f"  [bold white]GPU {lbl} compute (smi)[/bold white]   "
                    f"[yellow]{val:.0f}%[/yellow]"
                    + ("  [dim](0% if kernels < poll interval)[/dim]" if val == 0 else "")
                )
            for key, val in sorted(gpu_mem_peak.items()):
                lbl = key.replace("gpu_mem_used_bytes", "").strip("[]") or "0"
                health_lines.append(
                    f"  [bold white]GPU {lbl} mem used (smi)[/bold white]  "
                    f"[yellow]{_fmt_bytes(val)}[/yellow]"
                )

        if health_lines:
            lines.append(
                "[bold]── Run Health ───────────────────────────────────────────────────[/bold]"
            )
            lines.extend(health_lines)
            lines.append("")

        # ── CPU microarch ─────────────────────────────────────────────────
        ipc        = ctrs_last.get("ipc", 0.0)
        cache_miss = ctrs_last.get("cache_miss_pct", -1.0)
        br_miss    = ctrs_last.get("branch_miss_pct", -1.0)
        if ipc > 0 or cache_miss >= 0 or br_miss >= 0:
            lines.append(
                "[bold]── CPU Microarch (perf stat) ────────────────────────────────────[/bold]"
            )
            if ipc > 0:
                # Colour: green >2, yellow 1–2, red <1
                ipc_color = "green" if ipc >= 2.0 else ("yellow" if ipc >= 1.0 else "red")
                lines.append(
                    f"  [bold white]IPC                [/bold white]  "
                    f"[{ipc_color}]{ipc:.2f}[/{ipc_color}]"
                )
            if cache_miss >= 0:
                cm_color = "green" if cache_miss < 5 else ("yellow" if cache_miss < 20 else "red")
                lines.append(
                    f"  [bold white]LLC cache miss rate[/bold white]  "
                    f"[{cm_color}]{cache_miss:.2f}%[/{cm_color}]"
                )
            if br_miss >= 0:
                bm_color = "green" if br_miss < 1 else ("yellow" if br_miss < 5 else "red")
                lines.append(
                    f"  [bold white]Branch miss rate   [/bold white]  "
                    f"[{bm_color}]{br_miss:.2f}%[/{bm_color}]"
                )
            lines.append("")

        # ── Time by backend ───────────────────────────────────────────────
        lines.append(
            "[bold]── Time by Backend ─────────────────────────────────────────────[/bold]"
        )
        by_cat: dict[str, dict] = defaultdict(lambda: {"total_ns": 0, "count": 0})
        for s in spans:
            by_cat[s.category.value]["total_ns"] += s.duration_ns
            by_cat[s.category.value]["count"]    += 1

        grand_total = sum(v["total_ns"] for v in by_cat.values()) or 1
        bar_w = 32

        for cat, info in sorted(by_cat.items(), key=lambda kv: -kv[1]["total_ns"]):
            frac  = info["total_ns"] / grand_total
            bar   = _bar(frac, bar_w)
            color = _cat_color(cat)
            lines.append(
                f"  [{color}]{cat:<10}[/{color}] "
                f"[{color}]{bar}[/{color}] "
                f"[yellow]{frac * 100:5.1f}%[/yellow]  "
                f"[white]{_fmt_ns(info['total_ns']):>10}[/white]  "
                f"[dim]{info['count']:>5} events[/dim]"
            )

        if not by_cat:
            lines.append("  [dim]No spans recorded.[/dim]")
        lines.append("")

        # ── Top 10 hotspots ───────────────────────────────────────────────
        lines.append(
            "[bold]── Top 10 Hotspots ─────────────────────────────────────────────[/bold]"
        )
        stats = trace.aggregated_stats()
        if not stats:
            lines.append("  [dim]No spans recorded.[/dim]")
        else:
            total_all = sum(r["total_ns"] for r in stats) or 1
            bar_w2    = 22
            for i, row in enumerate(stats[:10]):
                frac  = row["total_ns"] / total_all
                color = _cat_color(row["category"])
                bar   = _bar(frac, bar_w2)
                name  = row["name"][:34]
                cat_tag = f"({row['category']:<6})"  # parens avoid Rich markup conflicts
                lines.append(
                    f"  [dim]{i + 1:>2}.[/dim] "
                    f"[{color}]{cat_tag}[/{color}] "
                    f"[bold]{name:<34}[/bold]  "
                    f"[{color}]{bar}[/{color}]  "
                    f"[yellow]{row['pct']:5.1f}%[/yellow]  "
                    f"[white]{_fmt_ns(row['total_ns']):>10}[/white]  "
                    f"[dim]{row['count']:>4}×[/dim]"
                )

        return "\n".join(lines)


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

        label = f"{core}  ({count})"
        return f"{label:<{self._LABEL_W}}"

    def render(self) -> Any:  # noqa: ANN401
        width  = self.size.width - self._COL_W
        height = self.size.height
        if width < 4 or height < 4:
            return Text("(too small)")

        out = Text()

        visible_ns     = self._trace_dur / self.zoom
        offset_ns      = self._trace_dur * self.view_x / (width * self.zoom)
        tick_step_px   = max(12, width // 8)
        tick_ns_per_px = visible_ns / width

        # ── Time ruler (2 lines) ────────────────────────────────────────
        out.append(" " * self._COL_W, style="dim")
        for i in range(0, width, tick_step_px):
            ts_ns = offset_ns + i * tick_ns_per_px
            cell  = f"{_fmt_ns(ts_ns):<{tick_step_px}}"[:tick_step_px]
            out.append(cell, style="bold white")
        out.append("\n")

        out.append(" " * self._COL_W, style="dim")
        for i in range(width):
            out.append("┬" if i % tick_step_px == 0 else "─", style="dim")
        out.append("\n")

        # ── Visible lanes ────────────────────────────────────────────────
        # Each lane = 1 data row + 1 blank spacer → 2 lines per lane.
        # Reserve 2 lines for ruler + 1 legend + 1 status = 4 overhead.
        overhead     = 4
        visible_rows = max(1, (height - overhead) // 2)
        total_lanes  = len(self._lane_names)

        # Clamp for display only — never mutate reactive inside render()
        max_sy = max(0, total_lanes - visible_rows)
        lo = min(self.view_y, max_sy)
        hi = min(total_lanes, lo + visible_rows)

        for lane_name in self._lane_names[lo:hi]:
            spans = self._lanes[lane_name]
            cat   = lane_name.split("/")[0]
            color = _cat_color(cat)

            # data row
            out.append(f"{self._lane_label(lane_name)} ", style=f"bold {color}")

            cells = bytearray(width)
            for span in spans:
                rel0 = span.start_ns - self._view_start
                rel1 = span.end_ns   - self._view_start
                x0   = int((rel0 / self._trace_dur) * width * self.zoom) - self.view_x
                x1   = max(x0 + 1,
                            int((rel1 / self._trace_dur) * width * self.zoom) - self.view_x)
                for x in range(max(0, x0), min(width, x1)):
                    cells[x] = 1

            i = 0
            while i < width:
                filled = cells[i]
                j = i + 1
                while j < width and cells[j] == filled:
                    j += 1
                ch    = "█" if filled else "·"
                style = f"bold {color}" if filled else "color(237)"
                out.append(ch * (j - i), style=style)
                i = j
            out.append("\n")

            # blank spacer row
            out.append("\n")

        # ── Color legend ─────────────────────────────────────────────────
        present = {ln.split("/")[0] for ln in self._lane_names}
        out.append(" " * self._COL_W, style="dim")
        for cat in ["cpu", "cuda", "rocm", "opencl", "openmp",
                    "memory", "sync", "jit", "nvtx", "other"]:
            if cat in present:
                color = _cat_color(cat)
                out.append(f"█ {cat}  ", style=f"bold {color}")
        out.append("\n")

        # ── Status footer ─────────────────────────────────────────────────
        lane_info = (f"threads {lo + 1}–{hi} of {total_lanes}   "
                     if total_lanes > visible_rows else "")
        out.append(
            f" zoom {self.zoom:.1f}×   offset {_fmt_ns(offset_ns)}   "
            f"window {_fmt_ns(visible_ns)}   {lane_info}"
            f"[←→] scroll  [↑↓] pan  [+/-] zoom  [r] reset",
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
    HotspotsWidget  { height: 1fr; }
    DisasmWidget    { height: 1fr; }
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
            with TabPane("Overview  [O]", id="tab-overview"):
                with ScrollableContainer(id="overview-scroll"):
                    yield OverviewWidget(self.trace)

            with TabPane("Timeline  [T]", id="tab-timeline"):
                yield TimelineWidget(self.trace)

            with TabPane("Hotspots  [H]", id="tab-hotspots"):
                yield HotspotsWidget(self.trace, id="hotspots")

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
            trace.add(SpanEvent(
                name=ev.get("name", ""),
                category=cat,
                start_ns=int(ev.get("ts", 0) * 1_000),
                duration_ns=int(ev.get("dur", 0) * 1_000),
                pid=ev.get("pid", 0),
                tid=ev.get("tid", 0),
                tags=ev.get("args", {}),
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

    if collect_disasm:
        import threading as _threading
        def _bg_disasm():
            try:
                from ..core.runner import _collect_disasm
                _collect_disasm(trace, [metadata.command] + metadata.args, metadata.backends_used)
            except Exception:
                pass
        _threading.Thread(target=_bg_disasm, daemon=True).start()

    return trace
