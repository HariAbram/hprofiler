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
    "type":  "log",
    "range": [-3, 5],
    "gridcolor":  "#374151",
    "tickcolor":  "#6b7280",
    "tickfont":   {"color": "#9ca3af"},
    "titlefont":  {"color": "#d1d5db"},
    "showline": True, "linecolor": "#4b5563",
    "zeroline": False,
}


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

    traces = [
        _tr(f"BW ceiling  ({bw:.0f} GB/s)",
            [1e-3, ridge], [v * bw / 1000 for v in [1e-3, ridge]],
            "rgba(34,211,238,0.9)"),
        _tr(f"FP32 ceiling  ({peak:.2f} TFLOPs/s)",
            [ridge, 1e5], [peak, peak],
            "rgba(250,204,21,0.9)", dash="dash"),
    ]

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
        xs, ys, ms = zip(*[(p[0], max(p[1], 1e-9), p[2]) for p in pts])
        sizes = [max(10, 8 + 4 * math.log10(max(m.duration_ns, 1) / 1e6)) for m in ms]
        hover = []
        for m in ms:
            bw_note = ("<br><i>⚠ BW counter includes L2 write-backs — may exceed peak</i>"
                       if m.bw_pct >= 90 else "")
            ceil_t = min(m.arith_intensity * bw / 1000, peak)
            hr = ceil_t / m.achieved_tflops if m.achieved_tflops > 0 else float("inf")
            prec = (f"FP64: {m.fp64_fraction*100:.1f}%  FP16: {m.fp16_fraction*100:.1f}%<br>"
                    if m.fp64_fraction > 0.01 or m.fp16_fraction > 0.01 else "")
            src_line = (f"FLOPs: {m.est_flops/1e9:.3f} GFLOPs<br>DRAM: {m.est_bytes/1e9:.3f} GB<br>"
                        if "disasm" not in m.data_source
                        else f"Est FLOPs: {m.est_flops/1e9:.3f} GFLOPs <i>(disasm)</i><br>"
                             f"Est DRAM: {m.est_bytes/1e9:.3f} GB <i>(disasm)</i><br>")
            hover.append(
                f"<b>{m.kernel_name}</b><br>Arch: {m.arch}<br>"
                f"Duration: {m.duration_ns/1e6:.3f} ms  Threads: {m.threads:,}<br>"
                f"AI: {m.arith_intensity:.4f} FLOPs/byte<br>{src_line}{prec}"
                f"Perf: {m.achieved_tflops:.4f} TFLOPs/s ({m.flops_pct:.1f}% peak)<br>"
                f"BW: {m.achieved_gbs:.1f} GB/s ({m.bw_pct:.1f}% peak){bw_note}<br>"
                f"<b>FP32 ceiling: {ceil_t:.4f} TFLOPs/s ({hr:.1f}× headroom)</b><br>"
                f"<b>Bound: {m.bound}</b>  ridge: {m.ridge:.1f} F/B<br>"
                f"<i>Source: {m.data_source}</i>"
            )
        traces.append(_tr(f"{bound}-bound", list(xs), list(ys), color,
                          sym=symbol, sizes=list(sizes),
                          text=[m.kernel_name[:20] for m in ms], custom=hover))

    annotations = [{
        "x": math.log10(ridge), "y": math.log10(peak),
        "xref": xref, "yref": yref,
        "text": f"Ridge<br>{ridge:.1f} F/B",
        "showarrow": True, "arrowhead": 2, "arrowcolor": "rgba(250,204,21,0.8)",
        "ax": 30, "ay": -30,
        "font": {"color": "rgba(250,204,21,0.9)", "size": 11},
    }]
    return traces, annotations


def _make_figure(device: "DevicePeak",
                 metrics: list["KernelMetrics"],
                 trace_name: str = "",
                 data_source: str = "disasm") -> dict:
    """Build a single-device Plotly figure dict."""
    peak  = device.fp32_tflops
    bw    = device.bandwidth_gbs

    traces, annotations = _make_traces(device, metrics)

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
        "xaxis": {"title": "Arithmetic Intensity (FLOPs / byte)", **_AXIS_STYLE},
        "yaxis": {"title": "Performance (TFLOPs/s)", **_AXIS_STYLE},
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
        layout[axis_key_x] = {
            "title": ("Arithmetic Intensity (FLOPs / byte)"
                      if col == n // 2 else ""),
            "domain": domain_x,
            "anchor": yref,
            **_AXIS_STYLE,
        }
        layout[axis_key_y] = {
            "title": "Performance (TFLOPs/s)" if col == 0 else "",
            "domain": domain_y,
            "anchor": xref,
            **_AXIS_STYLE,
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
    var SKIP = ["BW ceiling", "FP32 ceiling", "FP64 ceiling", "FP16 ceiling",
                "Tensor ceiling", ""];

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
      if (SKIP.indexOf(pt.data.name) !== -1) return;
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
