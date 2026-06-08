"""
Interactive HTML roofline chart using Plotly.js (CDN, no local install needed).

Generates a self-contained HTML file that can be opened in any browser.
The chart uses log-log scale (standard for roofline models) with:
  - Memory bandwidth ceiling  (cyan diagonal)
  - Compute ceiling           (yellow horizontal)
  - Per-kernel scatter points (green=compute-bound, red=memory-bound)
  - Hover tooltips with full details
  - Ridge-point annotation
"""

from __future__ import annotations
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..analysis.device import DevicePeak
    from ..analysis.roofline import KernelMetrics


# Plotly.js pinned version (avoids surprise breakage on "latest")
_PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.27.0.min.js"


def _axis_id(idx: int) -> str:
    """Plotly axis suffix: '' for first subplot, '2' for second, etc."""
    return "" if idx == 0 else str(idx + 1)


_AXIS_STYLE = {
    "type":       "log",
    "gridcolor":  "#374151",
    "tickcolor":  "#6b7280",
    "tickfont":   {"color": "#9ca3af"},
    "titlefont":  {"color": "#d1d5db"},
    "showline": True, "linecolor": "#4b5563",
    "zeroline": False,
}


def _axis_ranges(
    device: "DevicePeak",
    metrics: list["KernelMetrics"],
) -> tuple[list[float], list[float]]:
    """
    Auto-fit log10 [lo, hi] ranges for the X (AI) and Y (TFLOPs/s) axes.

    Strategy:
      X — left edge is the minimum of (data_ai / 5) and (ridge / 50), so the
          chart always shows both the memory-slope context AND any memory-bound
          data points that sit left of the ridge.  Right edge is one decade
          beyond the rightmost data point or ridge.
      Y — from below the lowest TFLOPs value to above the FP32 peak.
      Kernels with unmeasured FLOPs (est_flops == 0, clamped to left edge) are
      excluded from AI range computation to avoid distorting the X range.
    """
    peak  = max(device.fp32_tflops, 1e-9)
    ridge = device.ridge_point if device.ridge_point > 0 else 1.0

    # Exclude FLOPs-unmeasured kernels (their AI is a clamped placeholder, not real)
    ai_vals = [m.arith_intensity for m in metrics
               if 0 < m.arith_intensity < 1e9 and m.est_flops > 0]
    tp_vals = [m.achieved_tflops for m in metrics if m.achieved_tflops > 0]

    min_ai = min(ai_vals) if ai_vals else ridge
    max_ai = max(ai_vals) if ai_vals else ridge

    # X range ─────────────────────────────────────────────────────────────────
    # Left: show left of data AND enough BW slope for visual context
    x_left = min(min_ai / 5, ridge / 50)
    x_lo   = math.log10(max(x_left, 1e-5))
    # Right: one decade beyond the rightmost point or ridge
    x_hi   = math.log10(max(max_ai, ridge) * 10)
    if x_hi - x_lo < 2:    # enforce minimum 2-decade span
        x_hi = x_lo + 2

    # Y range ─────────────────────────────────────────────────────────────────
    min_tp = min(tp_vals) if tp_vals else peak * 0.01
    y_lo = math.log10(max(min_tp / 5, 1e-9))
    y_hi = math.log10(peak * 3)
    if y_hi - y_lo < 2:
        y_lo = y_hi - 2

    return [x_lo, x_hi], [y_lo, y_hi]


def _make_traces(device: "DevicePeak",
                 metrics: list["KernelMetrics"],
                 ax: str = "",
                 legendgroup: str = "") -> tuple[list, list]:
    """
    Build Plotly trace dicts + annotation dicts for *device*/*metrics*.

    *ax*          — Plotly axis suffix ('' → xaxis/yaxis, '2' → xaxis2/yaxis2, …)
    *legendgroup* — group name so multiple subplots share legend colours
    Returns (traces, annotations).
    """
    peak  = device.fp32_tflops
    bw    = device.bandwidth_gbs
    ridge = device.ridge_point

    xref = f"x{ax}"
    yref = f"y{ax}"

    def _ridge(tflops: float) -> float:
        return tflops * 1e12 / (bw * 1e9) if bw > 0 else 0.0

    def _tr(name, xs, ys, color, dash=None, width=2.5, show=True, sym=None, hover=None,
            custom=None, text=None, sizes=None) -> dict:
        t: dict = {
            "type": "scatter", "x": xs, "y": ys,
            "name": name, "xaxis": xref, "yaxis": yref,
            "showlegend": show,
        }
        if legendgroup:
            t["legendgroup"] = legendgroup
        line: dict = {"color": color, "width": width}
        if dash:
            line["dash"] = dash
        if sym is None:
            t["mode"] = "lines"
            t["line"] = line
            t["hoverinfo"] = "skip"
        else:
            t["mode"] = "markers+text"
            t["marker"] = {"color": color, "size": sizes, "symbol": sym,
                           "line": {"color": "white", "width": 0.5}}
            t["text"] = text
            t["textposition"] = "top center"
            t["textfont"] = {"size": 10, "color": color}
            t["hovertemplate"] = "%{customdata}<extra></extra>"
            t["customdata"] = custom
        return t

    def _bw_ridge(bw_gbs: float) -> float:
        """AI (FLOPs/byte) where the BW slope at bw_gbs meets the FP32 compute ceiling."""
        return peak * 1e12 / (bw_gbs * 1e9) if bw_gbs > 0 else ridge

    # Read cache BW values upfront so we can compute ridge_start before the traces list.
    l2_bw = device.l2_bandwidth_gbs
    l3_bw = device.l3_bandwidth_gbs

    # The FP32 compute ceiling must start from the leftmost ridge — whichever memory
    # level has the highest bandwidth terminates its BW slope first (lowest AI), and
    # the flat compute ceiling must connect there.  Without this, a gap appears between
    # the top of the L3/L2 BW line and the start of the FP32 ceiling.
    ridge_start = ridge                                           # DRAM ridge by default
    if l2_bw > 0: ridge_start = min(ridge_start, _bw_ridge(l2_bw))
    if l3_bw > 0: ridge_start = min(ridge_start, _bw_ridge(l3_bw))

    traces = [
        _tr(f"DRAM BW  ({bw:.0f} GB/s)",
            [1e-3, ridge], [v * bw / 1000 for v in [1e-3, ridge]],
            "rgba(34,211,238,0.9)"),
        _tr(f"FP32 ceiling  ({peak:.2f} TFLOPs/s)",
            [max(ridge_start, 1e-3), 1e5], [peak, peak],
            "rgba(250,204,21,0.9)", dash="dash"),
    ]

    # L2 cache bandwidth ceiling (GPU: architecture estimate)
    if l2_bw > 0:
        r_l2 = _bw_ridge(l2_bw)
        traces.append(_tr(f"L2 BW  ({l2_bw:.0f} GB/s)",
                          [1e-3, max(r_l2, 1e-3)],
                          [v * l2_bw / 1000 for v in [1e-3, max(r_l2, 1e-3)]],
                          "rgba(56,189,248,0.65)", width=1.8))

    # L3 cache bandwidth ceiling (CPU only)
    if l3_bw > 0:
        r_l3 = _bw_ridge(l3_bw)
        traces.append(_tr(f"L3 BW  (~{l3_bw:.0f} GB/s, est.)",
                          [1e-3, max(r_l3, 1e-3)],
                          [v * l3_bw / 1000 for v in [1e-3, max(r_l3, 1e-3)]],
                          "rgba(56,189,248,0.5)", width=1.5, dash="dot"))

    if device.fp64_tflops > 0 and device.fp64_tflops < peak * 0.95:
        r64 = _ridge(device.fp64_tflops)
        traces.append(_tr(f"FP64 ceiling  ({device.fp64_tflops:.2f} TFLOPs/s)",
                          [max(r64, 1e-3), 1e5],
                          [device.fp64_tflops, device.fp64_tflops],
                          "rgba(248,113,113,0.85)", dash="dashdot", width=1.8))
        traces.append(_tr("", [1e-3, r64], [v * bw / 1000 for v in [1e-3, r64]],
                          "rgba(248,113,113,0.4)", width=1.2, show=False))

    if device.fp16_tflops > peak * 1.05:
        r16 = _ridge(device.fp16_tflops)
        traces.append(_tr(f"FP16 ceiling  ({device.fp16_tflops:.1f} TFLOPs/s)",
                          [max(r16, 1e-3), 1e5],
                          [device.fp16_tflops, device.fp16_tflops],
                          "rgba(167,139,250,0.85)", dash="dot", width=1.8))

    if device.tensor_tflops > device.fp16_tflops * 1.05:
        r_tc = _ridge(device.tensor_tflops)
        traces.append(_tr(f"Tensor ceiling  ({device.tensor_tflops:.0f} TFLOPs/s)",
                          [max(r_tc, 1e-3), 1e5],
                          [device.tensor_tflops, device.tensor_tflops],
                          "rgba(52,211,153,0.85)", dash="dot", width=1.8))

    _AXIS_MIN_X = 1e-3   # matches _AXIS_STYLE range [-3, 5]
    _AXIS_MIN_Y = 1e-3

    # Occupancy-limited compute ceiling: when average occupancy < 100%, the
    # achievable peak is reduced proportionally (wave-front latency hiding).
    # Only draw when counters report < 80% occupancy (above that, the effect
    # is usually negligible compared to other bottlenecks).
    all_occ = [m.occupancy_pct for m in metrics if m.occupancy_pct > 0]
    if all_occ:
        avg_occ = sum(all_occ) / len(all_occ)
        if avg_occ < 80.0:
            occ_peak = peak * (avg_occ / 100.0)
            r_occ    = _ridge(occ_peak) if occ_peak > 0 else ridge
            traces.append(_tr(
                f"Occupancy ceiling  ({avg_occ:.0f}% → {occ_peak:.3f} TFLOPs/s)",
                [max(r_occ, 1e-3), 1e5], [occ_peak, occ_peak],
                "rgba(251,191,36,0.7)", dash="longdash", width=1.5,
            ))

    for bound, color, symbol in [
        ("compute", "rgba(74,222,128,1.0)", "circle"),
        ("memory",  "rgba(248,113,113,1.0)", "diamond"),
    ]:
        pts = [(m.arith_intensity if m.arith_intensity < 1e9 else ridge * 100,
                min(m.achieved_tflops,
                    m.arith_intensity * bw / 1000 if bound == "memory" and m.arith_intensity > 0
                    else m.achieved_tflops),
                m)
               for m in metrics if m.bound == bound]
        if not pts:
            continue
        # Clamp both axes to the visible minimum so points with ai=0 or tflops=0
        # (e.g. FLOPs not measurable on hybrid-core CPUs) still appear on the chart.
        xs, ys, ms = zip(*[(max(p[0], _AXIS_MIN_X), max(p[1], _AXIS_MIN_Y), p[2]) for p in pts])
        hover = []
        for m in ms:
            flops_unmeasured = m.est_flops == 0.0
            bw_note = ("<br><i>⚠ BW counter includes L2 write-backs — may exceed peak</i>"
                       if m.bw_pct >= 90 else "")
            fp_note = ("<br><b>⚠ FLOPs not measured</b> — point placed at chart left edge.<br>"
                       "<i>On hybrid Intel CPUs, pin to P-cores: taskset -c 0-7 ./app<br>"
                       "Or use LIKWID for accurate hybrid-core FP counting.</i>"
                       if flops_unmeasured else "")
            ceil_t = min(m.arith_intensity * bw / 1000, peak) if m.arith_intensity > 0 else 0.0
            hr = ceil_t / m.achieved_tflops if m.achieved_tflops > 0 else float("inf")
            prec = (f"FP64: {m.fp64_fraction*100:.1f}%  FP16: {m.fp16_fraction*100:.1f}%<br>"
                    if m.fp64_fraction > 0.01 or m.fp16_fraction > 0.01 else "")
            src_line = (f"FLOPs: {m.est_flops/1e9:.3f} GFLOPs<br>DRAM: {m.est_bytes/1e9:.3f} GB<br>"
                        if "disasm" not in m.data_source
                        else f"Est FLOPs: {m.est_flops/1e9:.3f} GFLOPs <i>(disasm)</i><br>"
                             f"Est DRAM: {m.est_bytes/1e9:.3f} GB <i>(disasm)</i><br>")
            ai_str = "N/A (FLOPs not measured)" if flops_unmeasured else f"{m.arith_intensity:.4f} FLOPs/byte"

            # Multi-level AI lines
            cache_lines = ""
            if m.l2_ai > 0:
                l2_bw_gbs = device.l2_bandwidth_gbs
                l2_achieved_gbs = (m.l2_bytes / (m.duration_ns / 1e9)) / 1e9 if m.duration_ns > 0 else 0.0
                l2_bw_str = (f"  ({l2_achieved_gbs:.1f} GB/s, {100*l2_achieved_gbs/l2_bw_gbs:.1f}% peak)"
                             if l2_bw_gbs > 0 else "")
                cache_lines += f"L2 AI: {m.l2_ai:.4f} F/B  ({m.l2_bytes/1e9:.3f} GB){l2_bw_str}<br>"
            if m.l1_ai > 0:
                cache_lines += f"L1 AI: {m.l1_ai:.4f} F/B  ({m.l1_bytes/1e9:.3f} GB)<br>"
            if m.l3_ai > 0:
                l3_bw_gbs = device.l3_bandwidth_gbs
                l3_achieved_gbs = (m.l3_bytes / (m.duration_ns / 1e9)) / 1e9 if m.duration_ns > 0 else 0.0
                l3_bw_str = (f"  ({l3_achieved_gbs:.1f} GB/s, ~{100*l3_achieved_gbs/l3_bw_gbs:.1f}% est. peak)"
                             if l3_bw_gbs > 0 else "")
                cache_lines += f"L3 AI: {m.l3_ai:.4f} F/B  ({m.l3_bytes/1e9:.3f} GB){l3_bw_str}<br>"

            # Occupancy and IPC
            occ_line = ""
            if m.occupancy_pct > 0:
                occ_warn = "  ⚠ low" if m.occupancy_pct < 50 else ""
                occ_line += f"Occupancy: {m.occupancy_pct:.1f}%{occ_warn}<br>"
            if m.ipc > 0:
                occ_line += f"IPC: {m.ipc:.2f}<br>"

            hover.append(
                f"<b>{m.kernel_name}</b><br>Arch: {m.arch}<br>"
                f"Duration: {m.duration_ns/1e6:.3f} ms  Threads: {m.threads:,}<br>"
                f"AI (DRAM): {ai_str}<br>{cache_lines}{src_line}{prec}"
                f"DRAM BW: {m.achieved_gbs:.1f} GB/s ({m.bw_pct:.1f}% peak){bw_note}"
                + ("" if flops_unmeasured else
                   f"<br>Perf: {m.achieved_tflops:.4f} TFLOPs/s ({m.flops_pct:.1f}% peak)<br>"
                   f"<b>FP32 ceiling: {ceil_t:.4f} TFLOPs/s ({hr:.1f}× headroom)</b><br>"
                   f"<b>Bound: {m.bound}</b>  ridge: {m.ridge:.1f} F/B") +
                (f"<br>{occ_line}" if occ_line else "") +
                f"{fp_note}<br><i>Source: {m.data_source}</i>"
            )
        # Split into measured (show normally) and unmeasured-FLOPs (gray x-mark)
        measured   = [(x, y, m, h) for x, y, m, h in zip(xs, ys, ms, hover)
                      if m.est_flops > 0]
        unmeasured = [(x, y, m, h) for x, y, m, h in zip(xs, ys, ms, hover)
                      if m.est_flops == 0]
        if measured:
            mxs, mys, mms, mhov = zip(*measured)
            msz = [max(10, 8 + 4 * math.log10(max(m.duration_ns, 1) / 1e6)) for m in mms]
            traces.append(_tr(f"{bound}-bound", list(mxs), list(mys), color,
                              sym=symbol, sizes=msz,
                              text=[m.kernel_name[:20] for m in mms], custom=list(mhov)))
        if unmeasured:
            uxs, uys, ums, uhov = zip(*unmeasured)
            usz = [max(10, 8 + 4 * math.log10(max(m.duration_ns, 1) / 1e6)) for m in ums]
            traces.append(_tr("BW only (FLOPs unmeasured)", list(uxs), list(uys),
                              "rgba(180,180,180,0.85)",
                              sym="x", sizes=usz,
                              text=[m.kernel_name[:20] for m in ums], custom=list(uhov)))

    annotations = [{
        "x": math.log10(ridge), "y": math.log10(peak),
        "xref": xref, "yref": yref,
        "text": f"DRAM ridge<br>{ridge:.1f} F/B",
        "showarrow": True, "arrowhead": 2, "arrowcolor": "rgba(250,204,21,0.8)",
        "ax": 30, "ay": -30,
        "font": {"color": "rgba(250,204,21,0.9)", "size": 11},
    }]

    # Ridge annotations for L2 and L3 ceilings — mark where each slope terminates
    if l2_bw > 0:
        r_l2_ann = peak * 1e12 / (l2_bw * 1e9)
        annotations.append({
            "x": math.log10(r_l2_ann), "y": math.log10(peak),
            "xref": xref, "yref": yref,
            "text": f"L2 ridge<br>{r_l2_ann:.2f} F/B",
            "showarrow": True, "arrowhead": 2,
            "arrowcolor": "rgba(56,189,248,0.8)",
            "ax": -30, "ay": -30,
            "font": {"color": "rgba(56,189,248,0.9)", "size": 10},
        })
    if l3_bw > 0:
        r_l3_ann = peak * 1e12 / (l3_bw * 1e9)
        annotations.append({
            "x": math.log10(r_l3_ann), "y": math.log10(peak),
            "xref": xref, "yref": yref,
            "text": f"L3 ridge<br>{r_l3_ann:.2f} F/B",
            "showarrow": True, "arrowhead": 2,
            "arrowcolor": "rgba(56,189,248,0.6)",
            "ax": -30, "ay": 30,
            "font": {"color": "rgba(56,189,248,0.7)", "size": 10},
        })

    return traces, annotations


def _make_figure(device: "DevicePeak",
                 metrics: list["KernelMetrics"],
                 trace_name: str = "",
                 data_source: str = "disasm") -> dict:
    """Build a single-device Plotly figure dict."""
    peak  = device.fp32_tflops
    bw    = device.bandwidth_gbs

    traces, annotations = _make_traces(device, metrics)
    x_range, y_range = _axis_ranges(device, metrics)

    layout = {
        "title": {
            "text": (
                f"Roofline — {trace_name}<br>"
                f"<sub>{device.name}  ({device.backend})"
                f"  |  FP32: {peak:.2f} TFLOPs/s"
                + (f"  |  FP64: {device.fp64_tflops:.3f} TFLOPs/s"
                   if device.fp64_tflops > 0 else "")
                + f"  |  BW: {bw:.0f} GB/s</sub>"
            ),
            "font": {"color": "#f9fafb", "size": 16},
        },
        "paper_bgcolor": "#111827",
        "plot_bgcolor":  "#1f2937",
        "font":          {"color": "#d1d5db", "family": "monospace"},
        "xaxis": {"title": "Arithmetic Intensity (FLOPs / byte)",
                  **_AXIS_STYLE, "range": x_range},
        "yaxis": {"title": "Performance (TFLOPs/s)",
                  **_AXIS_STYLE, "range": y_range},
        "legend": {"bgcolor": "#1f2937", "bordercolor": "#374151",
                   "borderwidth": 1, "font": {"color": "#d1d5db"}},
        "annotations": annotations,
        "hovermode": "closest",
        "margin": {"l": 70, "r": 30, "t": 100, "b": 70},
    }
    return {"data": traces, "layout": layout}


def _make_combined_figure(device_metrics: list[tuple["DevicePeak", list["KernelMetrics"]]],
                          trace_name: str = "") -> dict:
    """
    Build a Plotly figure with one subplot per device, all in a single HTML.
    Devices are arranged in a single row of N columns.
    """
    n = len(device_metrics)
    if n == 1:
        dev, mets = device_metrics[0]
        return _make_figure(dev, mets, trace_name)

    # Column widths split equally with a small gap
    gap = 0.04
    col_w = (1.0 - gap * (n - 1)) / n

    all_traces: list[dict] = []
    all_annotations: list[dict] = []
    layout: dict = {
        "paper_bgcolor": "#111827",
        "plot_bgcolor":  "#1f2937",
        "font":          {"color": "#d1d5db", "family": "monospace"},
        "legend": {"bgcolor": "#1f2937", "bordercolor": "#374151",
                   "borderwidth": 1, "font": {"color": "#d1d5db"},
                   "tracegroupgap": 4},
        "hovermode": "closest",
        "margin": {"l": 70, "r": 30, "t": 100, "b": 70},
        "title": {
            "text": f"Roofline — {trace_name}",
            "font": {"color": "#f9fafb", "size": 16},
        },
    }

    for col, (dev, mets) in enumerate(device_metrics):
        ax = _axis_id(col)
        xref = f"x{ax}"
        yref = f"y{ax}"
        domain_x = [col * (col_w + gap), col * (col_w + gap) + col_w]
        domain_y = [0.0, 1.0]

        traces, annotations = _make_traces(dev, mets, ax=ax,
                                           legendgroup=f"gpu{col}")
        all_traces.extend(traces)
        all_annotations.extend(annotations)

        axis_key_x = f"xaxis{ax}"
        axis_key_y = f"yaxis{ax}"
        sub_title = (f"{dev.name}  ({dev.backend})<br>"
                     f"FP32: {dev.fp32_tflops:.2f} TFLOPs/s  "
                     f"BW: {dev.bandwidth_gbs:.0f} GB/s")
        x_range, y_range = _axis_ranges(dev, mets)
        layout[axis_key_x] = {
            "title": ("Arithmetic Intensity (FLOPs / byte)"
                      if col == n // 2 else ""),
            "domain": domain_x,
            "anchor": yref,
            **_AXIS_STYLE,
            "range": x_range,
        }
        layout[axis_key_y] = {
            "title": "Performance (TFLOPs/s)" if col == 0 else "",
            "domain": domain_y,
            "anchor": xref,
            **_AXIS_STYLE,
            "range": y_range,
        }
        # Per-subplot title via annotation at the top of each subplot
        all_annotations.append({
            "x": (domain_x[0] + domain_x[1]) / 2,
            "y": 1.02,
            "xref": "paper", "yref": "paper",
            "text": f"<b>GPU {col}</b>: {sub_title}",
            "showarrow": False,
            "font": {"color": "#9ca3af", "size": 10},
            "xanchor": "center",
        })

    layout["annotations"] = all_annotations
    return {"data": all_traces, "layout": layout}


def _html_page(fig: dict, title: str,
               device_info: list[tuple[float, float]]) -> str:
    """
    Render a complete HTML page for *fig*.
    *device_info* — list of (bandwidth_gbs, fp32_tflops) per subplot axis index.
    Axis key for index 0 = "x", index 1 = "x2", etc.
    """
    # Build JS maps from axis id → BW / PEAK
    bw_map   = {(f"x{_axis_id(i)}"): bw   for i, (bw, pk) in enumerate(device_info)}
    peak_map = {(f"x{_axis_id(i)}"): pk   for i, (bw, pk) in enumerate(device_info)}

    return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Roofline — {title}</title>
  <script src="{_PLOTLY_CDN}"></script>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: #111827; color: #f9fafb;
      font-family: ui-monospace, 'Cascadia Code', 'Fira Code', monospace;
      padding: 16px;
    }}
    #chart {{ width: 100%; height: calc(100vh - 32px); min-height: 500px; }}
    .no-script {{ text-align:center; padding:40px; color:#9ca3af; border:1px solid #374151; }}
  </style>
</head>
<body>
  <div id="chart">
    <noscript><div class="no-script">
      JavaScript is required to render the interactive chart.
    </div></noscript>
  </div>
  <script>
    var figData   = {json.dumps(fig["data"])};
    var figLayout = {json.dumps(fig["layout"])};
    var chart = document.getElementById("chart");

    Plotly.newPlot(chart, figData, figLayout, {{
      responsive: true, displaylogo: false,
      modeBarButtonsToRemove: ["lasso2d", "select2d"],
    }});

    // Per-subplot device limits keyed by Plotly axis id ("x", "x2", …)
    var BW_MAP   = {json.dumps(bw_map)};
    var PEAK_MAP = {json.dumps(peak_map)};
    // Fallback for single-plot (axis id "x")
    var BW   = {device_info[0][0]};
    var PEAK = {device_info[0][1]};
    // Ceiling/BW line traces — skip crosshair for these (they have no point data).
    // Match by prefix so e.g. "DRAM BW  (900 GB/s)" matches "DRAM BW".
    var SKIP_PREFIXES = ["DRAM BW", "BW ceiling", "FP32 ceiling", "FP64 ceiling",
                         "FP16 ceiling", "Tensor ceiling", "L2 BW", "L3 BW",
                         "Occupancy ceiling", ""];
    function _skipTrace(name) {{
      if (!name) return true;   // empty-string trace (unnamed ceiling segments)
      for (var i = 0; i < SKIP_PREFIXES.length; i++) {{
        var p = SKIP_PREFIXES[i];
        if (p && name.indexOf(p) === 0) return true;
      }}
      return false;
    }}

    // ── SVG overlay helpers ───────────────────────────────────────────────────
    // Draw crosshairs directly on the chart SVG so Plotly's hover state and
    // tooltip are never disturbed (Plotly.relayout() during hover clears the
    // tooltip and can trigger plotly_unhover in a loop).

    function _svgLine(x1, y1, x2, y2, stroke, dash) {{
      var el = document.createElementNS("http://www.w3.org/2000/svg", "line");
      el.setAttribute("x1", x1); el.setAttribute("y1", y1);
      el.setAttribute("x2", x2); el.setAttribute("y2", y2);
      el.setAttribute("stroke", stroke);
      el.setAttribute("stroke-width", "1.5");
      el.setAttribute("stroke-dasharray", dash);
      el.setAttribute("pointer-events", "none");
      return el;
    }}

    function _svgText(x, y, lines, fg, bg) {{
      var g = document.createElementNS("http://www.w3.org/2000/svg", "g");
      var pad = 5, lh = 14;
      var maxW = lines.reduce(function(m, l) {{ return Math.max(m, l.length * 6.5); }}, 0);
      var h = lines.length * lh + pad * 2;
      var rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      rect.setAttribute("x", x + 8); rect.setAttribute("y", y - lh);
      rect.setAttribute("width", maxW + pad * 2); rect.setAttribute("height", h);
      rect.setAttribute("fill", bg); rect.setAttribute("rx", "3");
      rect.setAttribute("stroke", fg); rect.setAttribute("stroke-width", "0.5");
      rect.setAttribute("pointer-events", "none");
      g.appendChild(rect);
      lines.forEach(function(line, i) {{
        var t = document.createElementNS("http://www.w3.org/2000/svg", "text");
        t.setAttribute("x", x + 8 + pad); t.setAttribute("y", y + i * lh);
        t.setAttribute("fill", fg);
        t.setAttribute("font-size", "11");
        t.setAttribute("font-family", "monospace");
        t.setAttribute("pointer-events", "none");
        t.textContent = line;
        g.appendChild(t);
      }});
      return g;
    }}

    function removeCrosshairs() {{
      var el = document.getElementById("_hprofiler_ch");
      if (el) el.remove();
    }}

    function drawCrosshairs(ai, perf, ceiling, xax, yax) {{
      removeCrosshairs();
      // For subplots, each axis has a domain property [frac_start, frac_end].
      // Convert data → pixel within subplot domain, then add the domain offset
      // relative to the full figure.
      var fl   = chart._fullLayout;
      var figW = fl.width, figH = fl.height;
      var ml = fl.margin.l, mr = fl.margin.r,
          mt = fl.margin.t, mb = fl.margin.b;
      var plotW = figW - ml - mr;
      var plotH = figH - mt - mb;

      var xDom = xax.domain;   // [left_frac, right_frac] in paper coords
      var yDom = yax.domain;   // [bottom_frac, top_frac] in paper coords

      // Subplot's pixel origin (top-left corner in SVG coords)
      var subL = ml + xDom[0] * plotW;
      var subT = mt + (1 - yDom[1]) * plotH;

      // c2p returns pixel offset from the subplot's left / top edge
      var xPx   = xax.c2p(ai,      false) + subL;
      var yPx   = yax.c2p(perf,    false) + subT;
      var yCeil = yax.c2p(ceiling, false) + subT;
      var yBot  = yax.c2p(Math.pow(10, yax.range[0]), false) + subT;
      var xLeft = subL;

      var headroom = ceiling / perf;

      var svg = chart.querySelector("svg.main-svg");
      var g   = document.createElementNS("http://www.w3.org/2000/svg", "g");
      g.id = "_hprofiler_ch";

      // Dotted drop to x-axis
      g.appendChild(_svgLine(xPx, yPx, xPx, yBot,  "rgba(255,255,255,0.4)", "4 4"));
      // Dotted drop to y-axis
      g.appendChild(_svgLine(xPx, yPx, xLeft, yPx, "rgba(255,255,255,0.4)", "4 4"));
      // Dashed gap up to ceiling (only when there is a visible gap)
      if (Math.abs(yCeil - yPx) > 2) {{
        g.appendChild(_svgLine(xPx, yPx, xPx, yCeil, "rgba(250,204,21,0.7)", "6 3"));
      }}
      // Ceiling label
      var label = [
        "Ceiling: " + ceiling.toFixed(4) + " TFLOPs/s",
        headroom.toFixed(2) + "\xd7 headroom",
      ];
      g.appendChild(_svgText(xPx, yCeil - 4, label, "rgba(250,204,21,1)", "rgba(17,24,39,0.88)"));

      svg.appendChild(g);
    }}

    chart.on("plotly_hover", function(ev) {{
      var pt = ev.points[0];
      if (_skipTrace(pt.data.name)) return;
      // Resolve per-subplot BW and PEAK using the trace's axis id
      var axId  = pt.data.xaxis || "x";
      var bw    = BW_MAP[axId]   || BW;
      var peak  = PEAK_MAP[axId] || PEAK;
      var ceiling = Math.min(pt.x * bw / 1000, peak);
      // Retrieve axis objects for subplot-aware pixel conversion
      var xax = chart._fullLayout[axId === "x" ? "xaxis" : "xaxis" + axId.slice(1)];
      var yax = chart._fullLayout[(pt.data.yaxis || "y") === "y" ? "yaxis"
                                  : "yaxis" + (pt.data.yaxis || "y").slice(1)];
      drawCrosshairs(pt.x, pt.y, ceiling, xax, yax);
    }});

    chart.on("plotly_unhover", function() {{
      removeCrosshairs();
    }});
  </script>
</body>
</html>
"""


def write(device: "DevicePeak",
          metrics: list["KernelMetrics"],
          out_path: str | Path,
          trace_name: str = "") -> None:
    """Write a single-device interactive HTML roofline chart to *out_path*."""
    fig = _make_figure(device, metrics, trace_name)
    html = _html_page(fig, trace_name, [(device.bandwidth_gbs, device.fp32_tflops)])
    with open(out_path, "w") as f:
        f.write(html)


def write_from_trace(trace, out_path: str | Path) -> int:
    """
    Analyse *trace* and write a single HTML roofline file.

    When multiple GPU devices are present all are shown as side-by-side
    subplots within the same HTML file — no separate files.
    Returns the number of kernels plotted.
    """
    from ..analysis.roofline import analyze_trace
    from pathlib import Path as _Path

    results = analyze_trace(trace)
    if not results or not trace.devices:
        return 0

    name = trace.metadata.command or str(_Path(str(out_path)).stem)

    # Group metrics by device index
    by_device: dict[int, list] = {}
    for dev, m in results:
        idx = trace.devices.index(dev) if dev in trace.devices else 0
        by_device.setdefault(idx, []).append(m)

    device_metrics = [(trace.devices[i], mets)
                      for i, mets in sorted(by_device.items())]

    if len(device_metrics) == 1:
        dev, mets = device_metrics[0]
        fig = _make_figure(dev, mets, name)
        device_info = [(dev.bandwidth_gbs, dev.fp32_tflops)]
    else:
        fig = _make_combined_figure(device_metrics, name)
        device_info = [(d.bandwidth_gbs, d.fp32_tflops) for d, _ in device_metrics]

    html = _html_page(fig, name, device_info)
    with open(out_path, "w") as f:
        f.write(html)
    return sum(len(m) for _, m in device_metrics)
