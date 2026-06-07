"""
Flame graph HTML generator.

Generates a self-contained interactive HTML flame graph from folded-stacks
format (Brendan Gregg's stackcollapse-perf.pl format):

    func1;func2;func3 count
    func1;func4 count

Public API:
    generate_html(folded_stacks, title) -> str (HTML text)
    collect_folded_stacks(command, env, callgraph, freq) -> str (folded stacks)
    save(folded_stacks, path, title) -> None
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path


# ── Tree builder ───────────────────────────────────────────────────────────────

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


def _tree_to_json(node: dict) -> dict:
    return {
        "n": node["name"],
        "v": node["value"],
        "c": [
            _tree_to_json(c)
            for c in sorted(node["children"].values(),
                            key=lambda x: x["value"], reverse=True)
            if c["value"] > 0
        ],
    }


# ── JavaScript ─────────────────────────────────────────────────────────────────

_JS = r"""
(function () {
'use strict';

const FRAME_H = 20;   // px per stack level
const MIN_PX  = 1;    // skip frames narrower than this

let zoomStack = [DATA];
let searchRe  = null;

// ── colour ──────────────────────────────────────────────────────────────────
function colorFor(name) {
    let h = 0;
    for (let i = 0; i < name.length; i++)
        h = (Math.imul(h, 31) + name.charCodeAt(i)) | 0;
    h = h >>> 0;
    return [200 + (h & 0x37), 80 + ((h >> 6) & 0x5F), 20 + ((h >> 14) & 0x3F)];
}

// ── layout ───────────────────────────────────────────────────────────────────
// Flat list of {node, x, depth, w, y} — y filled in during render()
function buildLayout(root, totalW) {
    const out = [];
    function walk(node, x, w, depth) {
        if (w < MIN_PX) return;
        out.push({node, x, depth, w, y: 0});
        let cx = x;
        for (const child of node.c) {
            const cw = w * child.v / node.v;
            walk(child, cx, cw, depth + 1);
            cx += cw;
        }
    }
    walk(root, 0, totalW, 0);
    return out;
}

function treeDepth(node) {
    let m = 0;
    for (const c of node.c) m = Math.max(m, 1 + treeDepth(c));
    return m;
}

// ── render ───────────────────────────────────────────────────────────────────
let _frames = [];

function render() {
    const wrap   = document.getElementById('canvas-wrap');
    const canvas = document.getElementById('canvas');
    const ctx    = canvas.getContext('2d');
    const root   = zoomStack[zoomStack.length - 1];
    const W      = Math.max(wrap.clientWidth, 400);
    const depth  = treeDepth(root) + 1;
    const H      = depth * FRAME_H + 4;

    canvas.width  = W;
    canvas.height = H;
    ctx.clearRect(0, 0, W, H);
    ctx.font = '11px monospace';

    _frames = buildLayout(root, W);

    for (const f of _frames) {
        const {node, x, depth: d, w} = f;
        // Flame graph: root at bottom → y decreases as depth grows
        const y = H - (d + 1) * FRAME_H;
        f.y = y;

        let [r, g, b] = colorFor(node.n);
        if (searchRe) {
            if (searchRe.test(node.n)) {
                // highlight: bright gold
                r = 255; g = 210; b = 20;
            } else {
                // dim non-matching frames
                r = Math.round(r * 0.22);
                g = Math.round(g * 0.22);
                b = Math.round(b * 0.22);
            }
        }
        ctx.fillStyle = 'rgb(' + r + ',' + g + ',' + b + ')';
        ctx.fillRect(x, y, w - 0.5, FRAME_H - 1);

        if (w > 32) {
            ctx.fillStyle = '#111';
            const maxCh = Math.floor((w - 6) / 6.5);
            let lbl = node.n;
            if (lbl.length > maxCh) lbl = lbl.slice(0, maxCh - 1) + '…';
            ctx.fillText(lbl, x + 3, y + FRAME_H - 5);
        }
    }
}

// ── hit test ─────────────────────────────────────────────────────────────────
function hitTest(mx, my) {
    // Iterate in reverse so topmost painted frame wins
    for (let i = _frames.length - 1; i >= 0; i--) {
        const f = _frames[i];
        if (mx >= f.x && mx < f.x + f.w && my >= f.y && my < f.y + FRAME_H - 1)
            return f;
    }
    return null;
}

// ── helpers ──────────────────────────────────────────────────────────────────
function pctTotal(node)  { return (node.v / DATA.v * 100).toFixed(1); }
function pctView(node)   { return (node.v / zoomStack[zoomStack.length - 1].v * 100).toFixed(1); }
function fmtCount(node)  { return node.v.toLocaleString(); }

// ── event wiring ─────────────────────────────────────────────────────────────
const canvas  = document.getElementById('canvas');
const tooltip = document.getElementById('tooltip');
const info    = document.getElementById('info');

canvas.addEventListener('mousemove', function (e) {
    const r = canvas.getBoundingClientRect();
    const f = hitTest(e.clientX - r.left, e.clientY - r.top);
    if (f) {
        const n = f.node;
        info.textContent =
            n.n + '  —  ' + fmtCount(n) + ' samples  (' +
            pctTotal(n) + '% total, ' + pctView(n) + '% of view)';
        tooltip.textContent =
            n.n + '\n' +
            fmtCount(n) + ' samples\n' +
            pctTotal(n) + '% of total    ' + pctView(n) + '% of view';
        tooltip.style.display = 'block';
        // keep tooltip inside viewport
        const tx = Math.min(e.clientX + 14, window.innerWidth - tooltip.offsetWidth - 10);
        tooltip.style.left = tx + 'px';
        tooltip.style.top  = (e.clientY - 10) + 'px';
    } else {
        tooltip.style.display = 'none';
        info.textContent = DATA.v.toLocaleString() + ' samples total';
    }
});

canvas.addEventListener('mouseleave', function () {
    tooltip.style.display = 'none';
});

canvas.addEventListener('click', function (e) {
    const r = canvas.getBoundingClientRect();
    const f = hitTest(e.clientX - r.left, e.clientY - r.top);
    if (f && f.node !== zoomStack[zoomStack.length - 1]) {
        zoomStack.push(f.node);
        render();
    }
});

// Right-click = go up one zoom level
canvas.addEventListener('contextmenu', function (e) {
    e.preventDefault();
    if (zoomStack.length > 1) { zoomStack.pop(); render(); }
});

document.getElementById('btn-reset').addEventListener('click', function () {
    zoomStack = [DATA]; render();
});

document.getElementById('btn-up').addEventListener('click', function () {
    if (zoomStack.length > 1) { zoomStack.pop(); render(); }
});

document.getElementById('search').addEventListener('input', function (e) {
    const q = e.target.value.trim();
    try {
        searchRe = q ? new RegExp(q, 'i') : null;
        e.target.style.borderColor = '';
    } catch (_) {
        // invalid regex — keep previous
        e.target.style.borderColor = '#e94560';
        return;
    }
    render();
});

document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape') {
        zoomStack = [DATA]; render();
    } else if (e.key === 'Backspace' &&
               zoomStack.length > 1 &&
               document.activeElement !== document.getElementById('search')) {
        e.preventDefault();
        zoomStack.pop(); render();
    }
});

window.addEventListener('resize', render);

// ── keyboard hint ────────────────────────────────────────────────────────────
document.getElementById('hints').textContent =
    'Click → zoom in   ·   Right-click / ↑ Up → zoom out   ' +
    '·   Esc → reset   ·   Search supports regex';

render();

})();
"""

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>TITLE_PLACEHOLDER</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#1a1a2e;color:#eee;font:13px/1.4 monospace;
     display:flex;flex-direction:column;height:100vh;overflow:hidden}
#toolbar{display:flex;align-items:center;gap:8px;padding:7px 12px;
         background:#16213e;border-bottom:1px solid #0f3460;flex-shrink:0;
         flex-wrap:wrap}
h1{font-size:14px;font-weight:bold;color:#e94560;margin-right:auto;
   white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
#search{background:#0f3460;border:1px solid #4a90d9;border-radius:4px;
        color:#eee;padding:4px 9px;font:inherit;width:220px;transition:.15s}
#search:focus{outline:none;border-color:#fff}
.btn{background:#0f3460;border:1px solid #555;border-radius:4px;
     color:#eee;padding:4px 11px;cursor:pointer;font:inherit;white-space:nowrap}
.btn:hover{border-color:#e94560;color:#fff}
#info{font-size:11px;color:#aaa;min-width:0;overflow:hidden;
      text-overflow:ellipsis;white-space:nowrap;flex:1;text-align:right}
#hints{font-size:10px;color:#555;width:100%;padding-top:3px}
#canvas-wrap{flex:1;overflow:auto;background:#111;position:relative}
canvas{display:block;cursor:pointer}
#tooltip{position:fixed;background:rgba(0,0,0,.92);color:#eee;
         padding:7px 11px;border-radius:5px;font:12px monospace;
         pointer-events:none;display:none;max-width:540px;white-space:pre;
         border:1px solid #444;z-index:999;line-height:1.65;
         box-shadow:0 4px 16px rgba(0,0,0,.6)}
</style>
</head>
<body>
<div id="toolbar">
  <h1>TITLE_PLACEHOLDER</h1>
  <input id="search" type="text" placeholder="Search (regex)&#x2026;">
  <button class="btn" id="btn-up">&#x2191; Up</button>
  <button class="btn" id="btn-reset">&#x27F2; Reset</button>
  <span id="info">TOTAL_PLACEHOLDER samples</span>
  <div id="hints"></div>
</div>
<div id="canvas-wrap"><canvas id="canvas"></canvas></div>
<div id="tooltip"></div>
<script>
const DATA=DATA_JSON_PLACEHOLDER;
JS_PLACEHOLDER
</script>
</body>
</html>
"""


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_html(folded_stacks: str, title: str = "Flame Graph") -> str:
    """Return a self-contained interactive HTML flame graph."""
    import html as _html
    if not folded_stacks.strip():
        return ""
    tree = _build_tree(folded_stacks)
    if tree["value"] == 0:
        return ""
    data_json = json.dumps(_tree_to_json(tree), separators=(",", ":"))
    total = tree["value"]
    title_e = _html.escape(title)
    return (
        _HTML_TEMPLATE
        .replace("TITLE_PLACEHOLDER", title_e)
        .replace("TOTAL_PLACEHOLDER", f"{total:,}")
        .replace("DATA_JSON_PLACEHOLDER", data_json)
        .replace("JS_PLACEHOLDER", _JS)
    )


def save(folded_stacks: str, path: str, title: str = "Flame Graph") -> None:
    """Write an interactive flame graph to *path* (HTML)."""
    html = generate_html(folded_stacks, title)
    if not html:
        raise ValueError("No stacks to render — perf may not have collected any samples.")
    Path(path).write_text(html, encoding="utf-8")


# ── perf runner ────────────────────────────────────────────────────────────────

def _perf_script_to_folded(script_output: str) -> str:
    """Convert `perf script` output to folded-stacks format."""
    counts: dict[str, int] = defaultdict(int)
    _FRAME = re.compile(r'^\s+[\da-f]+\s+(\S.*?)(?:\s+\(([^)]+)\))?$')
    cur_comm = ""
    cur_stack: list[str] = []

    def flush() -> None:
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
