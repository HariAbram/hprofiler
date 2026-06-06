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


def _make_figure(device: "DevicePeak",
                 metrics: list["KernelMetrics"],
                 trace_name: str = "",
                 data_source: str = "disasm") -> dict:
    """Build a Plotly figure dict (serialisable to JSON)."""

    peak   = device.fp32_tflops
    bw     = device.bandwidth_gbs
    ridge  = device.ridge_point   # FLOPs/byte

    # ── Ceiling lines ──────────────────────────────────────────────────────────
    # Span from 0.001 to 100 000 FLOPs/byte (wider than the plot range so
    # the line reaches the edge of the axis after log-scale clipping).
    x_bw   = [1e-3, ridge]
    y_bw   = [v * bw / 1000 for v in x_bw]           # TFLOPs/s = F/B × GB/s / 1000

    x_comp = [ridge, 1e5]
    y_comp = [peak,  peak]

    traces = [
        # Memory bandwidth ceiling
        {
            "type": "scatter",
            "x": x_bw, "y": y_bw,
            "mode": "lines",
            "name": f"BW ceiling  ({bw:.0f} GB/s)",
            "line": {"color": "rgba(34,211,238,0.9)", "width": 2.5},
            "hoverinfo": "skip",
        },
        # Compute ceiling
        {
            "type": "scatter",
            "x": x_comp, "y": y_comp,
            "mode": "lines",
            "name": f"FP32 ceiling  ({peak:.1f} TFLOPs/s)",
            "line": {"color": "rgba(250,204,21,0.9)", "width": 2.5, "dash": "dash"},
            "hoverinfo": "skip",
        },
    ]

    # ── Kernel scatter split by bound type ────────────────────────────────────
    for bound, color, symbol in [
        ("compute", "rgba(74,222,128,1.0)", "circle"),
        ("memory",  "rgba(248,113,113,1.0)", "diamond"),
    ]:
        if not any(m.bound == bound for m in metrics):
            continue

        pts = []
        for m in metrics:
            if m.bound != bound:
                continue
            ai  = m.arith_intensity if m.arith_intensity < 1e9 else ridge * 100
            tfl = m.achieved_tflops
            # Memory-bound kernels cannot physically exceed the BW ceiling.
            # ncu's dram__bytes.sum includes L2 write-backs + HW prefetch, so
            # the apparent bandwidth can be 5-15% above the theoretical peak.
            # Cap the plotted Y at the BW ceiling so the point stays inside the
            # roofline envelope; the actual measured value is shown in the hover.
            if bound == "memory" and ai > 0:
                bw_ceiling_tfl = ai * bw / 1000
                tfl = min(tfl, bw_ceiling_tfl)
            pts.append((ai, max(tfl, 1e-9), m))

        xs, ys, ms = zip(*pts)
        # Marker size: proportional to log(duration_ns), min 10 px
        sizes = [max(10, 8 + 4 * math.log10(max(m.duration_ns, 1) / 1e6))
                 for m in ms]

        hover = []
        for m in ms:
            bw_note = ""
            if m.bw_pct >= 90:
                bw_note = (
                    "<br><i>⚠ Bandwidth-saturated — HW counters (dram__bytes.sum) "
                    "include L2 write-backs + prefetch, so measured BW can "
                    "slightly exceed theoretical peak.</i>"
                )
            ceiling_tflops = min(m.arith_intensity * bw / 1000, peak)
            headroom = ceiling_tflops / m.achieved_tflops if m.achieved_tflops > 0 else float("inf")
            hover.append(
                f"<b>{m.kernel_name}</b><br>"
                f"Arch: {m.arch}<br>"
                f"Duration: {m.duration_ns/1e6:.3f} ms<br>"
                f"Threads: {m.threads:,}<br>"
                f"Arith intensity: {m.arith_intensity:.4f} FLOPs/byte<br>"
                + (f"FLOPs: {m.est_flops/1e9:.3f} GFLOPs<br>"
                   f"DRAM: {m.est_bytes/1e9:.3f} GB<br>"
                   if m.data_source != "disasm"
                   else f"Est FLOPs: {m.est_flops/1e9:.3f} GFLOPs  <i>(disasm estimate)</i><br>"
                        f"Est DRAM: {m.est_bytes/1e9:.3f} GB  <i>(disasm estimate)</i><br>")
                + f"Achieved FP32: {m.achieved_tflops:.4f} TFLOPs/s  ({m.flops_pct:.1f}% peak)<br>"
                  f"Achieved BW:   {m.achieved_gbs:.1f} GB/s  ({m.bw_pct:.1f}% peak)"
                  f"{bw_note}<br>"
                  f"<b>Peak at this AI: {ceiling_tflops:.4f} TFLOPs/s</b>  "
                  f"({headroom:.1f}× headroom)<br>"
                  f"<b>Bound: {m.bound}</b>  |  ridge: {m.ridge:.1f} FLOPs/byte<br>"
                  f"<i>Source: {m.data_source}</i>"
            )

        traces.append({
            "type": "scatter",
            "x": list(xs), "y": list(ys),
            "mode": "markers+text",
            "name": f"{bound}-bound",
            "text": [m.kernel_name[:20] for m in ms],
            "textposition": "top center",
            "textfont": {"size": 10, "color": color},
            "hovertemplate": "%{customdata}<extra></extra>",
            "customdata": hover,
            "marker": {
                "color": color, "size": list(sizes),
                "symbol": symbol,
                "line": {"color": "white", "width": 0.5},
            },
        })

    # ── Ridge annotation ───────────────────────────────────────────────────────
    annotations = [
        {
            "x": math.log10(ridge), "y": math.log10(peak),
            "xref": "x", "yref": "y",
            "text": f"Ridge<br>{ridge:.1f} F/B",
            "showarrow": True, "arrowhead": 2, "arrowcolor": "rgba(250,204,21,0.8)",
            "ax": 30, "ay": -30,
            "font": {"color": "rgba(250,204,21,0.9)", "size": 11},
        },
    ]

    # ── Layout ─────────────────────────────────────────────────────────────────
    layout = {
        "title": {
            "text": (
                f"Roofline Model — {trace_name}<br>"
                f"<sub>{device.name}  ({device.backend} · {device.compute_cap})"
                f"  |  Peak FP32: {peak:.1f} TFLOPs/s"
                f"  |  Peak BW: {bw:.0f} GB/s"
                + (f"  |  Tensor: {device.tensor_tflops:.0f} TFLOPs/s"
                   if device.tensor_tflops > 0 else "")
                + "</sub>"
            ),
            "font": {"color": "#f9fafb", "size": 16},
        },
        "paper_bgcolor": "#111827",
        "plot_bgcolor":  "#1f2937",
        "font":          {"color": "#d1d5db", "family": "monospace"},
        "xaxis": {
            "title": "Arithmetic Intensity (FLOPs / byte)",
            "type":  "log",
            "range": [-3, 5],           # 0.001 → 100 000
            "gridcolor":     "#374151",
            "tickcolor":     "#6b7280",
            "tickfont":      {"color": "#9ca3af"},
            "titlefont":     {"color": "#d1d5db"},
            "showline": True, "linecolor": "#4b5563",
            "zeroline": False,
        },
        "yaxis": {
            "title": "Performance (TFLOPs/s)",
            "type":  "log",
            "gridcolor":     "#374151",
            "tickcolor":     "#6b7280",
            "tickfont":      {"color": "#9ca3af"},
            "titlefont":     {"color": "#d1d5db"},
            "showline": True, "linecolor": "#4b5563",
            "zeroline": False,
        },
        "legend": {
            "bgcolor": "#1f2937", "bordercolor": "#374151", "borderwidth": 1,
            "font": {"color": "#d1d5db"},
        },
        "annotations": annotations,
        "hovermode": "closest",
        "margin": {"l": 70, "r": 30, "t": 100, "b": 70},
    }

    return {"data": traces, "layout": layout}


def write(device: "DevicePeak",
          metrics: list["KernelMetrics"],
          out_path: str | Path,
          trace_name: str = "") -> None:
    """Write an interactive HTML roofline chart to *out_path*."""
    src = "disasm" if all(m.data_source == "disasm" for m in metrics) else "hardware_counters"
    fig = _make_figure(device, metrics, trace_name, data_source=src)

    html = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Roofline — {trace_name}</title>
  <script src="{_PLOTLY_CDN}"></script>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: #111827;
      color: #f9fafb;
      font-family: ui-monospace, 'Cascadia Code', 'Fira Code', monospace;
      padding: 16px;
    }}
    #chart {{ width: 100%; height: calc(100vh - 32px); min-height: 500px; }}
    .no-script {{
      text-align: center; padding: 40px;
      color: #9ca3af; border: 1px solid #374151;
    }}
  </style>
</head>
<body>
  <div id="chart">
    <noscript>
      <div class="no-script">
        JavaScript is required to render the interactive chart.<br>
        Open this file in a modern browser with JS enabled.
      </div>
    </noscript>
  </div>
  <script>
    var figData   = {json.dumps(fig["data"])};
    var figLayout = {json.dumps(fig["layout"])};
    var chart = document.getElementById("chart");

    Plotly.newPlot(chart, figData, figLayout, {{
      responsive: true,
      displaylogo: false,
      modeBarButtonsToRemove: ["lasso2d", "select2d"],
    }});

    var BW   = {device.bandwidth_gbs};
    var PEAK = {device.fp32_tflops};
    var SKIP = ["BW ceiling", "FP32 ceiling"];

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

    function drawCrosshairs(ai, perf, ceiling) {{
      removeCrosshairs();
      var xax = chart._fullLayout.xaxis;
      var yax = chart._fullLayout.yaxis;
      var ml  = chart._fullLayout.margin.l;
      var mt  = chart._fullLayout.margin.t;

      // c2p converts actual data value → pixel offset within the plot area
      var xPx   = xax.c2p(ai,      false) + ml;
      var yPx   = yax.c2p(perf,    false) + mt;
      var yCeil = yax.c2p(ceiling, false) + mt;
      // plot-area edges
      var yBot  = yax.c2p(Math.pow(10, yax.range[0]), false) + mt;
      var xLeft = xax.c2p(Math.pow(10, xax.range[0]), false) + ml;

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
      if (SKIP.indexOf(pt.data.name) !== -1) return;
      var ceiling = Math.min(pt.x * BW / 1000, PEAK);
      drawCrosshairs(pt.x, pt.y, ceiling);
    }});

    chart.on("plotly_unhover", function() {{
      removeCrosshairs();
    }});
  </script>
</body>
</html>
"""
    with open(out_path, "w") as f:
        f.write(html)


def write_from_trace(trace, out_path: str | Path) -> int:
    """
    Analyse *trace* and write an HTML roofline chart.
    Returns the number of kernels plotted (0 if nothing to plot).
    """
    from ..analysis.roofline import analyze_trace
    from pathlib import Path as _Path

    results = analyze_trace(trace)
    if not results or not trace.devices:
        return 0

    device  = trace.devices[0]
    metrics = [m for _, m in results]
    name    = trace.metadata.command or str(_Path(str(out_path)).stem)

    write(device, metrics, out_path, trace_name=name)
    return len(metrics)
