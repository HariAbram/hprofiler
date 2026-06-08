"""
Terminal roofline viewer.

Renders the Plotly roofline chart as a PNG via kaleido and displays it inline
using the Kitty graphics protocol (WezTerm / kitty / Ghostty / iTerm2).

Keyboard controls
─────────────────
  n / p        cycle through kernels and show crosshairs for the selected one
  Esc          deselect kernel (hide crosshairs)
  + / =        zoom in
  -            zoom out
  ← → ↑ ↓     pan
  r            reset zoom to auto-fit
  w            open HTML version in browser
  q            quit

Crosshairs are plotly shapes baked into the rendered PNG for pixel-perfect
rendering at any resolution, matching the HTML version's appearance.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..analysis.device import DevicePeak
    from ..analysis.roofline import KernelMetrics

# rows below the chart image: 2 info + 1 keys + 1 blank separator
_STATUS_H = 4


# ── Plotly compatibility fix ──────────────────────────────────────────────────

def _fix_deprecated_axis_props(layout: dict) -> None:
    """Upgrade 'titlefont' → 'title.font' so go.Figure() validation passes."""
    import re
    for key in list(layout.keys()):
        if not re.match(r"[xy]axis\d*$", key):
            continue
        ax = layout[key]
        if "titlefont" in ax:
            tf = ax.pop("titlefont")
            title = ax.get("title", {})
            if isinstance(title, str):
                title = {"text": title}
            title.setdefault("font", tf)
            ax["title"] = title


# ── Crosshair shapes / annotations ───────────────────────────────────────────
# For log-scale axes in plotly Python, shape/annotation coordinates are in
# log10 space (same as layout.xaxis.range / layout.yaxis.range).

def _crosshair_shapes(
    m: "KernelMetrics",
    device: "DevicePeak",
    x_range: list[float],
    y_range: list[float],
) -> list[dict]:
    """
    Three plotly line shapes that form the crosshair for kernel *m*:
      ╎  vertical dotted   point → x-axis
      ╌  horizontal dotted point → y-axis
      ╎  dashed yellow     point → FP32/BW ceiling  (the headroom gap)
    """
    ai     = m.arith_intensity
    tflops = m.achieved_tflops
    if ai <= 0 or tflops <= 0:
        return []

    log_ai   = math.log10(ai)
    log_tf   = math.log10(max(tflops, 1e-12))
    log_ybot = y_range[0]
    log_xlft = x_range[0]

    ceiling  = min(ai * device.bandwidth_gbs / 1000.0, device.fp32_tflops)
    log_ceil = math.log10(max(ceiling, 1e-12))

    shapes = [
        # Vertical dotted drop to x-axis
        {
            "type": "line",
            "x0": log_ai, "x1": log_ai, "y0": log_ybot, "y1": log_tf,
            "xref": "x", "yref": "y",
            "line": {"color": "rgba(255,255,255,0.38)", "dash": "dot", "width": 1.5},
        },
        # Horizontal dotted line to y-axis
        {
            "type": "line",
            "x0": log_xlft, "x1": log_ai, "y0": log_tf, "y1": log_tf,
            "xref": "x", "yref": "y",
            "line": {"color": "rgba(255,255,255,0.38)", "dash": "dot", "width": 1.5},
        },
    ]

    # Dashed yellow gap from point up to the ceiling
    if ceiling > tflops * 1.02:
        shapes.append({
            "type": "line",
            "x0": log_ai, "x1": log_ai, "y0": log_tf, "y1": log_ceil,
            "xref": "x", "yref": "y",
            "line": {"color": "rgba(250,204,21,0.75)", "dash": "dash", "width": 2},
        })

    return shapes


def _crosshair_annotation(m: "KernelMetrics", device: "DevicePeak") -> dict:
    """Headroom label at the ceiling point for the selected kernel."""
    ai       = m.arith_intensity
    tflops   = m.achieved_tflops
    ceiling  = min(ai * device.bandwidth_gbs / 1000.0, device.fp32_tflops)
    headroom = ceiling / max(tflops, 1e-12)

    return {
        "x":   math.log10(max(ai,      1e-9)),
        "y":   math.log10(max(ceiling, 1e-9)),
        "xref": "x", "yref": "y",
        "text": f"Ceiling: {ceiling:.4f} TFLOPs/s<br>{headroom:.2f}× headroom",
        "showarrow": True, "arrowhead": 2,
        "arrowcolor": "rgba(250,204,21,0.85)",
        "ax": 55, "ay": -40,
        "font":        {"color": "rgba(250,204,21,1.0)", "size": 11},
        "bgcolor":     "rgba(17,24,39,0.88)",
        "bordercolor": "rgba(250,204,21,0.45)",
        "borderwidth": 1,
        "align": "left",
    }


# ── PNG rendering ─────────────────────────────────────────────────────────────

def _render_png(
    device: "DevicePeak",
    metrics: list["KernelMetrics"],
    trace_name: str,
    x_range: list[float],
    y_range: list[float],
    w_px: int,
    h_px: int,
    selected_m: "KernelMetrics | None" = None,
) -> bytes:
    import plotly.graph_objects as go
    import plotly.io as pio
    from .roofline_html import _make_figure

    fig_dict = _make_figure(device, metrics, trace_name=trace_name)
    _fix_deprecated_axis_props(fig_dict["layout"])
    fig_dict["layout"]["width"]          = w_px
    fig_dict["layout"]["height"]         = h_px
    fig_dict["layout"]["xaxis"]["range"] = x_range
    fig_dict["layout"]["yaxis"]["range"] = y_range

    if selected_m is not None:
        shapes = _crosshair_shapes(selected_m, device, x_range, y_range)
        if shapes:
            fig_dict["layout"]["shapes"] = shapes
        ann = _crosshair_annotation(selected_m, device)
        existing = fig_dict["layout"].get("annotations", [])
        fig_dict["layout"]["annotations"] = existing + [ann]

    fig = go.Figure(data=fig_dict["data"], layout=fig_dict["layout"])
    return pio.to_image(fig, format="png", engine="kaleido")


# ── Kernel selection helpers ──────────────────────────────────────────────────

def _plottable(metrics: list["KernelMetrics"]) -> list["KernelMetrics"]:
    """Kernels that have a real position on the chart (measured FLOPs + DRAM)."""
    return [
        m for m in metrics
        if m.arith_intensity > 0
        and m.arith_intensity < 1e9
        and m.achieved_tflops > 0
        and m.est_flops > 0
    ]


def _info_lines(
    m: "KernelMetrics",
    device: "DevicePeak",
    idx: int,
    total: int,
) -> tuple[str, str]:
    """Two lines of kernel stats for the info panel."""
    ceiling  = min(m.arith_intensity * device.bandwidth_gbs / 1000.0, device.fp32_tflops)
    headroom = ceiling / max(m.achieved_tflops, 1e-12)
    dur_ms   = m.duration_ns / 1e6

    line1 = f"  [{idx + 1}/{total}]  {m.kernel_name}"
    line2 = (
        f"  AI: {m.arith_intensity:.2f} F/B"
        f"  │  Perf: {m.achieved_tflops:.4f} TFLOPs/s ({m.flops_pct:.1f}%)"
        f"  │  {m.bound}-bound"
        f"  │  {headroom:.2f}× headroom"
        f"  │  {dur_ms:.2f} ms"
    )
    return line1, line2


# ── Public entry point ────────────────────────────────────────────────────────

def show(
    device: "DevicePeak",
    metrics: list["KernelMetrics"],
    trace_name: str = "",
    html_path: str | None = None,
) -> None:
    """
    Launch the terminal roofline viewer.

    *html_path* — if given, pressing **w** opens that file in the browser and
                  it is also pre-written so the file already exists on disk.
    """
    try:
        import plotly.io  # noqa: F401
    except ImportError:
        print("plotly + kaleido required: pip install plotly kaleido")
        return

    from .term_image import (
        RawTerminal, display_image, get_protocol,
        terminal_size, terminal_pixel_size, with_spinner,
    )

    if get_protocol() == "none":
        if html_path:
            print(
                "[hprofiler] No inline-image terminal detected "
                "(set TERM_PROGRAM=WezTerm or use kitty/ghostty).\n"
                f"[hprofiler] Opening HTML roofline in browser: {html_path}"
            )
            _open_html(html_path)
        else:
            print(
                "[hprofiler] No inline-image terminal detected.\n"
                "  Run in kitty, WezTerm, or Ghostty, or use --html."
            )
        return

    from .roofline_html import _axis_ranges

    orig_xr, orig_yr = _axis_ranges(device, metrics)
    x_lo, x_hi = orig_xr
    y_lo, y_hi = orig_yr

    kernels = _plottable(metrics)
    sel_idx = -1   # -1 = no crosshair

    _KEYS = (
        "  n/p  kernel ✛    +/-  zoom    ← → ↑ ↓  pan    r  reset    Esc  deselect"
        + ("    w  HTML" if html_path else "")
        + "    q  quit"
    )

    dirty     = True   # image needs re-render
    info_only = False  # only the info panel needs refreshing

    with RawTerminal() as term:
        while True:
            cols, rows = terminal_size()
            img_rows  = max(rows - _STATUS_H, 5)
            w_px, full_h_px = terminal_pixel_size()
            h_px = max(int(full_h_px * img_rows / max(rows, 1)), 100)

            if dirty:
                _xr = [x_lo, x_hi]
                _yr = [y_lo, y_hi]
                _m  = kernels[sel_idx] if (kernels and 0 <= sel_idx < len(kernels)) else None

                term.clear()
                try:
                    png = with_spinner(
                        term,
                        img_rows + 1,
                        "Rendering roofline …",
                        lambda: _render_png(
                            device, metrics, trace_name,
                            _xr, _yr, w_px, h_px, selected_m=_m,
                        ),
                    )
                except Exception as exc:
                    term.clear()
                    term.goto(2)
                    term.writeln(f"  Render error: {exc}")
                    term.writeln("  Make sure kaleido is installed:  pip install kaleido")
                    term.writeln(
                        "  Press q to quit"
                        + (" or w to open HTML." if html_path else ".")
                    )
                    key = term.read_key()
                    if key in ("q", "ctrl+c"):
                        break
                    if key == "w" and html_path:
                        _open_html(html_path)
                    continue

                term.clear()
                term.goto(0)
                display_image(png, cols, img_rows)
                dirty     = False
                info_only = True  # always refresh panel after render

            if info_only:
                # ── info panel (2 lines below the image) ─────────────────────
                term.goto(img_rows + 1)
                if kernels and 0 <= sel_idx < len(kernels):
                    l1, l2 = _info_lines(kernels[sel_idx], device, sel_idx, len(kernels))
                    term.write("\x1b[2K" + l1)          # \x1b[2K clears the line
                    term.goto(img_rows + 2)
                    term.write("\x1b[2K" + l2, style="2")
                else:
                    hint = (
                        f"  {len(kernels)} kernel(s) plotted — press n to select and show crosshairs"
                        if kernels else "  No plottable kernels (FLOPs not measured)"
                    )
                    term.write("\x1b[2K" + hint, style="2")
                    term.goto(img_rows + 2)
                    term.write("\x1b[2K")

                term.goto(rows - 1)
                term.write("\x1b[2K" + _KEYS, style="2")
                info_only = False

            key = term.read_key()

            if key in ("q", "ctrl+c"):
                break

            elif key == "n":              # next kernel → show crosshair
                if kernels:
                    sel_idx = (sel_idx + 1) % len(kernels)
                    dirty = True

            elif key == "p":              # previous kernel
                if kernels:
                    sel_idx = (sel_idx - 1) % len(kernels)
                    dirty = True

            elif key == "esc":
                if sel_idx != -1:
                    sel_idx = -1
                    dirty = True

            elif key in ("+", "="):       # zoom in to 70 % of current range
                cx = (x_lo + x_hi) / 2;  cy = (y_lo + y_hi) / 2
                hw = (x_hi - x_lo) * 0.35;  hh = (y_hi - y_lo) * 0.35
                x_lo, x_hi = cx - hw, cx + hw
                y_lo, y_hi = cy - hh, cy + hh
                dirty = True

            elif key == "-":              # zoom out to 135 %
                cx = (x_lo + x_hi) / 2;  cy = (y_lo + y_hi) / 2
                hw = (x_hi - x_lo) * 0.675;  hh = (y_hi - y_lo) * 0.675
                x_lo, x_hi = cx - hw, cx + hw
                y_lo, y_hi = cy - hh, cy + hh
                dirty = True

            elif key == "left":
                d = (x_hi - x_lo) * 0.15
                x_lo -= d;  x_hi -= d;  dirty = True

            elif key == "right":
                d = (x_hi - x_lo) * 0.15
                x_lo += d;  x_hi += d;  dirty = True

            elif key == "up":
                d = (y_hi - y_lo) * 0.15
                y_lo += d;  y_hi += d;  dirty = True

            elif key == "down":
                d = (y_hi - y_lo) * 0.15
                y_lo -= d;  y_hi -= d;  dirty = True

            elif key == "r":
                x_lo, x_hi = orig_xr
                y_lo, y_hi = orig_yr
                dirty = True

            elif key == "w" and html_path:
                _open_html(html_path)


def _open_html(path: str) -> None:
    from .term_image import open_in_browser
    open_in_browser(path)
