"""
Terminal flamegraph viewer.

Renders a Plotly icicle chart as a PNG via kaleido (small, crisp font) and
displays it inline using the Kitty graphics protocol.  Click any bar to zoom
into that frame; the chart is re-rendered at every zoom level.

Click-to-zoom works by computing the same proportional icicle layout that
Plotly uses (branchvalues="total"), mapping the click's terminal-cell
position to a chart-pixel position, then finding the matching bar.

Controls
────────
  click           zoom into that frame
  u / Esc         zoom out one level
  r               reset to full view
  /               search — highlight frames by name substring
  w               open HTML version in browser
  q               quit

Requires: plotly + kaleido  (pip install plotly kaleido)
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional

_STATUS_H  = 4    # rows at the bottom for status / keys
# Plotly chart margins (must match _build_icicle layout margin)
_M_LEFT    = 10
_M_TOP     = 50
_M_RIGHT   = 10
_M_BOTTOM  = 10

_PALETTE = [
    "#e25c00", "#ef6c00", "#e65100", "#bf360c",
    "#f4511e", "#ff6e40", "#ff9100", "#ffd740",
    "#e040fb", "#7c4dff", "#40c4ff", "#64ffda",
    "#69f0ae", "#b2ff59", "#ffff00", "#ff6d00",
]


# ── Call-tree data model ──────────────────────────────────────────────────────

@dataclass
class FrameNode:
    name: str
    count: int = 0
    self_count: int = 0
    children: dict[str, "FrameNode"] = field(default_factory=dict)


def _parse_folded(folded: str) -> FrameNode:
    root = FrameNode(name="all")
    for raw in folded.splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            stack, _, count_s = line.rpartition(" ")
            count = int(count_s)
        except ValueError:
            continue
        frames = stack.split(";") if stack else []
        node = root
        node.count += count
        for frame in frames:
            if frame not in node.children:
                node.children[frame] = FrameNode(name=frame)
            node = node.children[frame]
            node.count += count
        node.self_count += count
    return root


# ── Layout computation (mirrors Plotly's icicle branchvalues="total") ─────────

@dataclass
class _Bar:
    node: FrameNode
    depth: int
    col0: int      # 0 … cols-1 (inclusive)
    col1: int      # exclusive


def _compute_layout(
    node: FrameNode,
    depth: int,
    col0: int,
    col1: int,
    result: list[_Bar],
    max_depth: int = 256,
) -> None:
    if col1 <= col0 or depth > max_depth:
        return
    result.append(_Bar(node, depth, col0, col1))
    width    = col1 - col0
    children = sorted(node.children.values(), key=lambda n: -n.count)
    if not children or node.count == 0:
        return
    cursor = col0
    n = len(children)
    for i, child in enumerate(children):
        if i == n - 1:
            child_end = col1
        else:
            child_end = cursor + max(1, round(child.count * width / node.count))
            child_end = min(child_end, col1 - (n - i - 1))
        if child_end > cursor:
            _compute_layout(child, depth + 1, cursor, child_end, result, max_depth)
        cursor = child_end


# ── Plotly icicle figure ──────────────────────────────────────────────────────

def _frame_color(name: str) -> str:
    h = int(hashlib.md5(name.encode()).hexdigest()[:8], 16)
    return _PALETTE[h % len(_PALETTE)]


def _build_icicle(
    zoom_root: FrameNode,
    total_samples: int,
    search: str = "",
    max_nodes: int = 2000,
    min_pct: float = 0.02,
    title: str = "",
    w_px: int = 1200,
    h_px: int = 800,
) -> dict:
    threshold = total_samples * min_pct / 100
    search_lc = search.lower()

    ids:     list[str] = []
    labels:  list[str] = []
    parents: list[str] = []
    values:  list[int] = []
    colors:  list[str] = []
    hovers:  list[str] = []

    def _collect(node: FrameNode, parent_id: str, depth: int) -> None:
        if len(ids) >= max_nodes:
            return
        node_id  = f"{parent_id}/{node.name}" if parent_id else node.name
        pct      = 100 * node.count / max(total_samples, 1)
        self_pct = 100 * node.self_count / max(total_samples, 1)

        ids.append(node_id)
        labels.append(node.name)
        parents.append(parent_id)
        values.append(node.count)
        hovers.append(
            f"<b>{node.name}</b><br>"
            f"Total: {pct:.2f}%  ({node.count:,} samples)<br>"
            f"Self:  {self_pct:.2f}%  ({node.self_count:,} samples)"
        )

        if search_lc and search_lc in node.name.lower():
            colors.append("#fbbf24")
        elif depth == 0:
            colors.append("#1e293b")
        else:
            colors.append(_frame_color(node.name))

        for child in sorted(node.children.values(), key=lambda n: -n.count):
            if child.count >= threshold:
                _collect(child, node_id, depth + 1)

    _collect(zoom_root, "", 0)

    n_match  = sum(1 for lbl in labels if search_lc and search_lc in lbl.lower())
    srch_tag = (
        f'  <span style="color:#fbbf24">⚲ {search!r}  ({n_match} frames)</span>'
        if search else ""
    )

    return {
        "data": [{
            "type": "icicle",
            "ids":      ids,
            "labels":   labels,
            "parents":  parents,
            "values":   values,
            "branchvalues": "total",
            "hovertext":     hovers,
            "hovertemplate": "%{hovertext}<extra></extra>",
            "marker":   {"colors": colors, "showscale": False},
            "textfont": {"family": "monospace", "size": 11},
            "tiling":   {"orientation": "v", "pad": 0},
            "pathbar":  {"visible": False},
        }],
        "layout": {
            "paper_bgcolor": "#111827",
            "plot_bgcolor":  "#111827",
            "font":   {"color": "#d1d5db", "family": "monospace"},
            "title":  {
                "text": (title or "Flame Graph") + srch_tag,
                "font": {"color": "#f9fafb", "size": 14},
            },
            "margin": {"l": _M_LEFT, "r": _M_RIGHT, "t": _M_TOP, "b": _M_BOTTOM},
            "width":  w_px,
            "height": h_px,
        },
    }


def _render_png(
    zoom_root: FrameNode,
    total_samples: int,
    search: str,
    title: str,
    w_px: int,
    h_px: int,
) -> bytes:
    import plotly.graph_objects as go
    import plotly.io as pio
    fig_dict = _build_icicle(
        zoom_root, total_samples,
        search=search, title=title, w_px=w_px, h_px=h_px,
    )
    fig = go.Figure(data=fig_dict["data"], layout=fig_dict["layout"])
    return pio.to_image(fig, format="png", engine="kaleido")


# ── Click → bar mapping ───────────────────────────────────────────────────────

def _hit_bar(
    click_row: int,
    click_col: int,
    bars: list[_Bar],
    max_depth: int,
    cols: int,
    rows: int,
    img_rows: int,
    w_px: int,
    full_h_px: int,
) -> Optional[_Bar]:
    """
    Map a terminal mouse-click (row, col) to the _Bar the user clicked on.

    The plotly icicle chart occupies [_M_LEFT … w_px-_M_RIGHT] × [_M_TOP … h_px-_M_BOTTOM]
    in pixel space.  Depth levels are distributed uniformly within that vertical
    range.  Our _compute_layout uses [0, cols] as the width coordinate space so
    column fractions translate directly to the bar's col0/col1 range.
    """
    h_px = max(int(full_h_px * img_rows / max(rows, 1)), 1)
    cell_w = w_px / max(cols, 1)
    cell_h = full_h_px / max(rows, 1)

    # Terminal-cell centre → image pixel
    px = click_col * cell_w + cell_w * 0.5
    py = click_row * cell_h + cell_h * 0.5

    chart_w = w_px - _M_LEFT - _M_RIGHT
    chart_h = h_px - _M_TOP  - _M_BOTTOM

    rx = px - _M_LEFT
    ry = py - _M_TOP

    if rx < 0 or ry < 0 or rx > chart_w or ry > chart_h:
        return None

    num_levels = max_depth + 1
    level_h    = chart_h / max(num_levels, 1)
    depth      = int(ry / level_h)

    layout_col = int((rx / chart_w) * cols)

    for bar in bars:
        if bar.depth == depth and bar.col0 <= layout_col < bar.col1:
            return bar
    return None


# ── Public entry point ────────────────────────────────────────────────────────

def show(folded: str, title: str = "", html_path: Optional[str] = None) -> None:
    """
    Launch the terminal flamegraph viewer.

    *folded*    — folded-stack string (``frame1;frame2;leaf count`` per line)
    *title*     — chart title
    *html_path* — pre-written HTML path; pressing **w** opens it in the browser
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
                "[hprofiler] No inline-image terminal detected.\n"
                f"[hprofiler] Opening HTML flamegraph in browser: {html_path}"
            )
            _open_html(html_path)
        else:
            print(
                "[hprofiler] No inline-image terminal detected.\n"
                "  Run in kitty, WezTerm, or Ghostty, or use --html."
            )
        return

    root = _parse_folded(folded)
    if root.count == 0:
        print("No stack data to display.")
        return

    total_samples = root.count
    zoom_stack: list[FrameNode] = [root]

    search      = ""
    search_mode = False
    search_buf  = ""
    dirty       = True
    bars:      list[_Bar] = []
    max_depth: int        = 0

    _KEYS_NORMAL = (
        "  click  zoom in    u/Esc  zoom out    /  search    r  reset"
        + ("    w  HTML" if html_path else "")
        + "    q  quit"
    )
    _KEYS_SEARCH = "  Type to filter  Enter=apply  Esc=cancel  Backspace=erase"

    with RawTerminal(mouse=True) as term:
        while True:
            cols, rows = terminal_size()
            img_rows   = max(rows - _STATUS_H, 5)
            w_px, full_h_px = terminal_pixel_size()
            h_px = max(int(full_h_px * img_rows / max(rows, 1)), 100)

            if dirty:
                zoom_root = zoom_stack[-1]

                # Recompute layout (needed for hit testing)
                bars = []
                _compute_layout(zoom_root, 0, 0, cols, bars)
                max_depth = max((b.depth for b in bars), default=0)

                _search = search
                term.clear()
                try:
                    png = with_spinner(
                        term,
                        img_rows + 1,
                        "Rendering flame graph …",
                        lambda: _render_png(
                            zoom_root, total_samples, _search, title, w_px, h_px,
                        ),
                    )
                except Exception as exc:
                    term.clear()
                    term.goto(2)
                    term.writeln(f"  Render error: {exc}")
                    term.writeln("  Make sure kaleido is installed:  pip install kaleido")
                    term.writeln("  Press q to quit.")
                    key = term.read_key()
                    if key in ("q", "ctrl+c"):
                        break
                    continue

                term.clear()
                term.goto(0)
                display_image(png, cols, img_rows)
                dirty = False

            # ── Status bar ────────────────────────────────────────────────────
            zoom_root = zoom_stack[-1]
            pct = 100.0 * zoom_root.count / max(total_samples, 1)
            if len(zoom_stack) > 1:
                info = f"  zoom: {zoom_root.name!r}  ({zoom_root.count:,} samples, {pct:.1f}%)"
            else:
                info = f"  {zoom_root.count:,} total samples"

            term.goto(rows - _STATUS_H)
            term.write("\x1b[2K" + info, style="2")

            term.goto(rows - _STATUS_H + 1)
            if search_mode:
                term.write(f"\x1b[2K  Search: {search_buf}_", style="33")
                term.goto(rows - _STATUS_H + 2)
                term.write("\x1b[2K" + _KEYS_SEARCH, style="2")
            else:
                if search:
                    term.write(
                        f"\x1b[2K  Filter: {search!r}  (/ to change, r to clear)",
                        style="33",
                    )
                    term.goto(rows - _STATUS_H + 2)
                term.write("\x1b[2K" + _KEYS_NORMAL, style="2")

            key = term.read_key()

            # ── search mode ────────────────────────────────────────────────────
            if search_mode:
                if key == "enter":
                    search      = search_buf
                    search_mode = False
                    search_buf  = ""
                    dirty = True
                elif key == "esc":
                    search_mode = False
                    search_buf  = ""
                elif key in ("\x7f", "backspace"):
                    search_buf = search_buf[:-1]
                elif len(key) == 1 and key.isprintable():
                    search_buf += key
                continue   # redraw status only, don't touch the image

            # ── normal mode ────────────────────────────────────────────────────
            if key in ("q", "ctrl+c"):
                break

            elif key.startswith("click:"):
                _, row_s, col_s = key.split(":")
                bar = _hit_bar(
                    int(row_s), int(col_s),
                    bars, max_depth,
                    cols, rows, img_rows, w_px, full_h_px,
                )
                if bar is not None and bar.node.children:
                    zoom_stack.append(bar.node)
                    dirty = True

            elif key in ("u", "esc", "\x7f"):   # zoom out
                if len(zoom_stack) > 1:
                    zoom_stack.pop()
                    dirty = True

            elif key == "r":
                zoom_stack = [root]
                search     = ""
                dirty      = True

            elif key == "/":
                search_mode = True
                search_buf  = search

            elif key == "w" and html_path:
                _open_html(html_path)


def _open_html(path: str) -> None:
    from .term_image import open_in_browser
    open_in_browser(path)
