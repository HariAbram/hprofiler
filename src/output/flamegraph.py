"""
Flame graph SVG generator.

Generates an interactive SVG flame graph from folded-stacks format
(Brendan Gregg's stackcollapse-perf.pl format):

    func1;func2;func3 count
    func1;func4 count

Public API:
    generate_svg(folded_stacks, title, width) -> str (SVG text)
    collect_folded_stacks(command, env, callgraph, freq) -> str (folded stacks)
    save(folded_stacks, path, title) -> None
"""

from __future__ import annotations

import hashlib
import html
import os
import re
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path


# ── Colour palette ─────────────────────────────────────────────────────────────

def _color_for(name: str) -> str:
    h = int(hashlib.md5(name.encode()).hexdigest()[:8], 16)
    # Warm orange–red tones for compute frames
    r = 200 + (h & 0x37)          # 200–255
    g = 80  + ((h >> 6)  & 0x5F)  # 80–175
    b = 20  + ((h >> 14) & 0x3F)  # 20–83
    return f"rgb({r},{g},{b})"


# ── Call-tree builder ──────────────────────────────────────────────────────────

def _build_tree(folded: str) -> dict:
    root: dict = {"name": "all", "value": 0, "children": {}}
    for line in folded.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        stack, count_s = parts
        try:
            count = int(count_s)
        except ValueError:
            continue
        node = root
        node["value"] += count
        for frame in stack.split(";"):
            frame = frame.strip()
            if not frame:
                continue
            if frame not in node["children"]:
                node["children"][frame] = {"name": frame, "value": 0, "children": {}}
            node = node["children"][frame]
            node["value"] += count
    return root


def _layout(node: dict, x: float, width: float, level: int, frames: list) -> None:
    frames.append({"name": node["name"], "x": x, "width": width, "level": level,
                   "samples": node["value"]})
    total = node["value"]
    if total == 0:
        return
    cx = x
    for child in sorted(node["children"].values(), key=lambda n: n["value"], reverse=True):
        cw = width * child["value"] / total
        if cw >= 0.5:
            _layout(child, cx, cw, level + 1, frames)
        cx += cw


# ── SVG renderer ───────────────────────────────────────────────────────────────

_JS = """\
(function(){
var info = document.getElementById('fg-info');
var frames = document.querySelectorAll('.fg-frame');
var totalW = parseFloat(document.getElementById('fg-root').getAttribute('width'));
var zoomStack = [{x:0, w:totalW}];

function pct(f){ return (parseFloat(f.getAttribute('width'))/totalW*100).toFixed(2); }

function applyZoom(z){
  var scale = totalW / z.w;
  var shift = z.x;
  frames.forEach(function(f){
    var ox = parseFloat(f.getAttribute('data-ox'));
    var ow = parseFloat(f.getAttribute('data-ow'));
    var nx = (ox - shift) * scale;
    var nw = ow * scale;
    f.setAttribute('x', nx.toFixed(2));
    f.setAttribute('width', nw.toFixed(2));
    var t = f.querySelector('text');
    if(t){ t.setAttribute('x', (nx + nw/2).toFixed(2)); t.style.display = nw<16?'none':''; }
    f.style.display = (nx+nw<0 || nx>totalW) ? 'none' : '';
  });
}

frames.forEach(function(f){
  f.setAttribute('data-ox', f.getAttribute('x'));
  f.setAttribute('data-ow', f.getAttribute('width'));
  f.addEventListener('mouseover', function(){
    info.textContent = f.getAttribute('data-name') + '  (' + pct(f) + '% of samples)';
  });
  f.addEventListener('mouseout', function(){ info.textContent=''; });
  f.addEventListener('click', function(){
    var z = {x: parseFloat(f.getAttribute('data-ox')),
             w: parseFloat(f.getAttribute('data-ow'))};
    zoomStack.push(z);
    applyZoom(z);
  });
});

document.getElementById('fg-root').addEventListener('dblclick', function(){
  if(zoomStack.length > 1) zoomStack.pop();
  applyZoom(zoomStack[zoomStack.length-1]);
});
})();
"""


def generate_svg(folded_stacks: str, title: str = "Flame Graph",
                 width: int = 1200) -> str:
    """Return an SVG string for an interactive flame graph."""
    if not folded_stacks.strip():
        return ""

    tree = _build_tree(folded_stacks)
    if tree["value"] == 0:
        return ""

    frames: list[dict] = []
    _layout(tree, 0.0, float(width), 0, frames)
    if not frames:
        return ""

    frame_h   = 16
    pad_top   = 44
    pad_bot   = 24
    num_lvls  = max(f["level"] for f in frames) + 1
    height    = pad_top + num_lvls * frame_h + pad_bot
    total     = tree["value"]

    parts: list[str] = []

    for f in frames:
        y    = height - pad_bot - (f["level"] + 1) * frame_h
        w    = f["width"]
        x    = f["x"]
        cx   = x + w / 2
        cy   = y + frame_h - 4
        col  = _color_for(f["name"])
        ne   = html.escape(f["name"])

        max_ch  = max(int(w / 6.5), 0)
        raw_lbl = f["name"] if len(f["name"]) <= max_ch else f["name"][:max_ch - 1] + "…"
        lbl_e   = html.escape(raw_lbl)

        parts.append(
            f'<g class="fg-frame"'
            f' data-name="{ne}">'
            f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="{frame_h-1}"'
            f' fill="{col}" rx="1"/>'
        )
        if w >= 16:
            vis = "" if w >= 16 else ' style="display:none"'
            parts.append(
                f'<text x="{cx:.1f}" y="{cy}" text-anchor="middle"'
                f' font-size="12" fill="#111" pointer-events="none"{vis}>{lbl_e}</text>'
            )
        parts.append("</g>")

    title_e = html.escape(title)
    js_escaped = _JS.replace("]]>", "]]]]><![CDATA[>")

    return (
        f'<?xml version="1.0" standalone="no"?>\n'
        f'<svg id="fg-root" version="1.1"'
        f' width="{width}" height="{height}"'
        f' xmlns="http://www.w3.org/2000/svg">\n'
        f'<style>'
        f'.fg-frame{{cursor:pointer}}'
        f'.fg-frame rect:hover{{stroke:#333;stroke-width:1}}'
        f'#fg-info{{font:12px monospace;fill:#555}}'
        f'</style>\n'
        f'<rect width="100%" height="100%" fill="#f8f8f8"/>\n'
        f'<text x="{width//2}" y="22" text-anchor="middle"'
        f' font-size="16" font-weight="bold" fill="#222">{title_e}</text>\n'
        f'<text x="{width//2}" y="36" text-anchor="middle"'
        f' font-size="11" fill="#888">'
        f'{total:,} samples — click frame to zoom, double-click background to reset'
        f'</text>\n'
        + "".join(parts)
        + f'\n<rect x="0" y="{height-pad_bot}" width="{width}" height="{pad_bot}" fill="#e8e8e8"/>\n'
        f'<text id="fg-info" x="8" y="{height-8}"></text>\n'
        f'<script type="text/javascript"><![CDATA[\n{_JS}]]></script>\n'
        f'</svg>\n'
    )


# ── perf runner ────────────────────────────────────────────────────────────────

def _perf_script_to_folded(script_output: str) -> str:
    """Convert `perf script` output to folded-stacks format."""
    counts: dict[str, int] = defaultdict(int)
    _FRAME = re.compile(r'^\s+[\da-f]+\s+(\S.*?)(?:\s+\(([^)]+)\))?$')
    cur_comm = ""
    cur_stack: list[str] = []

    def flush():
        if cur_stack:
            folded = cur_comm + ";" + ";".join(reversed(cur_stack))
            counts[folded] += 1

    for line in script_output.splitlines():
        if not line.strip():
            flush()
            cur_stack = []
            cur_comm = ""
            continue
        if not line[0].isspace():
            flush()
            cur_stack = []
            parts = line.split()
            cur_comm = parts[0] if parts else "?"
            continue
        m = _FRAME.match(line)
        if m:
            sym = m.group(1).split("+")[0].strip()
            if sym and sym != "[unknown]":
                cur_stack.append(sym)

    flush()
    return "\n".join(f"{s} {c}" for s, c in counts.items())


def collect_folded_stacks(
    command: list[str],
    env: dict,
    callgraph: str = "fp",
    freq: int = 99,
) -> str:
    """Run command under perf record with call-graph, return folded stacks."""
    if not command:
        return ""

    perf_data = tempfile.mktemp(suffix=".perf.data", prefix="hprofiler_fg_")
    try:
        perf_cmd = [
            "perf", "record",
            f"-F{freq}", "-g", f"--call-graph={callgraph}",
            "-o", perf_data, "--",
        ] + command
        subprocess.run(perf_cmd, env=env, check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if not Path(perf_data).exists():
            return ""

        r = subprocess.run(
            ["perf", "script", "-i", perf_data],
            capture_output=True, text=True, timeout=120,
        )
        return _perf_script_to_folded(r.stdout)
    except FileNotFoundError:
        return ""
    except Exception:
        return ""
    finally:
        try:
            os.unlink(perf_data)
        except OSError:
            pass


def save(folded_stacks: str, path: str, title: str = "Flame Graph") -> None:
    """Write flame graph SVG to path."""
    svg = generate_svg(folded_stacks, title)
    if not svg:
        raise ValueError("No stacks to render — perf may not have collected any samples.")
    Path(path).write_text(svg, encoding="utf-8")
