"""
Terminal flamegraph viewer — ANSI color renderer with mouse click-to-zoom.

Each terminal row = one call-stack depth level.
Each bar = one frame, width proportional to its sample count.

Controls
────────
  click         zoom into that frame (fills full width)
  u / Esc       zoom out one level
  r             reset to full view
  /             search — highlight frames by name substring
  w             open HTML version in browser
  q             quit

No browser or kaleido required — uses 24-bit ANSI colors directly.
"""
from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass, field
from typing import Optional

_STATUS_H = 3   # rows reserved at the bottom

# Warm palette (d3-flame-graph inspired)
_PALETTE = [
    "#e25c00", "#ef6c00", "#e65100", "#bf360c",
    "#f4511e", "#ff6e40", "#ff9100", "#ffd740",
    "#e040fb", "#7c4dff", "#40c4ff", "#64ffda",
    "#69f0ae", "#b2ff59", "#ffff00", "#ff6d00",
]
_ROOT_BG   = (30,  41,  59)   # dark navy  for the zoom-root bar
_SEARCH_BG = (217, 119,   6)  # amber      for search matches


# ── Call-tree data model ──────────────────────────────────────────────────────

@dataclass
class FrameNode:
    name: str
    count: int = 0
    self_count: int = 0
    children: dict[str, "FrameNode"] = field(default_factory=dict)


def _parse_folded(folded: str) -> FrameNode:
    """Parse ``frame1;frame2;leaf count`` lines into a FrameNode tree."""
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


# ── Layout computation ────────────────────────────────────────────────────────

@dataclass
class _Bar:
    node: FrameNode
    depth: int
    col0: int
    col1: int   # exclusive


def _compute_layout(
    node: FrameNode,
    depth: int,
    col0: int,
    col1: int,
    result: list[_Bar],
    max_depth: int = 256,
) -> None:
    """Recursively fill *result* with _Bar positions for the subtree of *node*."""
    if col1 <= col0 or depth > max_depth:
        return
    result.append(_Bar(node, depth, col0, col1))
    width = col1 - col0
    children = sorted(node.children.values(), key=lambda n: -n.count)
    if not children or node.count == 0:
        return
    # Distribute children proportionally within [col0, col1].
    # Children may not fill the full width when node has self-samples.
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


# ── ANSI rendering ────────────────────────────────────────────────────────────

def _frame_rgb(name: str) -> tuple[int, int, int]:
    h = int(hashlib.md5(name.encode()).hexdigest()[:8], 16)
    c = _PALETTE[h % len(_PALETTE)]
    return int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)


def _render_bar(bar: _Bar, search_lc: str, is_root_depth: bool) -> str:
    width = bar.col1 - bar.col0
    if width <= 0:
        return ""

    match = bool(search_lc and search_lc in bar.node.name.lower())

    if match:
        r, g, b = _SEARCH_BG
    elif is_root_depth:
        r, g, b = _ROOT_BG
    else:
        r, g, b = _frame_rgb(bar.node.name)

    # Choose foreground colour by perceived luminance
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    fg = "\x1b[38;2;20;20;20m" if lum > 140 else "\x1b[38;2;240;240;240m"
    bg = f"\x1b[48;2;{r};{g};{b}m"

    if width >= 3:
        label = bar.node.name
        if len(label) > width - 2:
            label = label[: width - 2]
        text = " " + label + " " * (width - len(label) - 1)
    else:
        text = " " * width

    return f"{bg}{fg}{text}\x1b[0m"


def _render_screen(
    term: object,   # RawTerminal — avoid circular import in type hint
    bars: list[_Bar],
    cols: int,
    rows: int,
    search: str,
    search_mode: bool,
    search_buf: str,
    zoom_stack: list[FrameNode],
    html_path: Optional[str],
    total_samples: int,
) -> None:
    img_rows   = rows - _STATUS_H
    search_lc  = search.lower()

    # Group by depth
    by_depth: dict[int, list[_Bar]] = {}
    for bar in bars:
        by_depth.setdefault(bar.depth, []).append(bar)

    # Clear and redraw
    sys.stdout.write("\x1b[2J\x1b[H")

    for row_idx in range(img_rows):
        sys.stdout.write(f"\x1b[{row_idx + 1};1H")
        row_bars = by_depth.get(row_idx, [])
        if not row_bars:
            sys.stdout.write(" " * cols)
        else:
            parts: list[str] = []
            pos = 0
            for bar in row_bars:
                if bar.col0 > pos:
                    parts.append(" " * (bar.col0 - pos))
                parts.append(_render_bar(bar, search_lc, is_root_depth=(bar.depth == 0)))
                pos = bar.col1
            if pos < cols:
                parts.append(" " * (cols - pos))
            sys.stdout.write("".join(parts))

    sys.stdout.flush()

    # ── Status bar ────────────────────────────────────────────────────────────
    zoom_root = zoom_stack[-1]
    pct = 100.0 * zoom_root.count / max(total_samples, 1)
    if len(zoom_stack) > 1:
        info = f"  zoom: {zoom_root.name!r}  ({zoom_root.count:,} samples, {pct:.1f}%)"
    else:
        info = f"  {zoom_root.count:,} total samples"

    term.goto(rows - _STATUS_H)
    term.write(info, style="2")

    term.goto(rows - _STATUS_H + 1)
    if search_mode:
        term.write(f"  Search: {search_buf}_", style="33")
    elif search:
        term.write(f"  Filter: {search!r}  (/ to change, r to clear)", style="33")
    else:
        keys = (
            "  click  zoom in    u/Esc  zoom out    /  search    r  reset"
            + ("    w  HTML" if html_path else "")
            + "    q  quit"
        )
        term.write(keys, style="2")


# ── Public entry point ────────────────────────────────────────────────────────

def show(folded: str, title: str = "", html_path: Optional[str] = None) -> None:
    """
    Launch the interactive terminal flamegraph viewer.

    *folded*    — folded-stack string (``frame1;frame2;leaf count`` per line)
    *title*     — descriptive title (shown in HTML if *html_path* is set)
    *html_path* — path to a pre-written HTML flamegraph; **w** opens it
    """
    from .term_image import RawTerminal, terminal_size

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
    bars: list[_Bar] = []

    with RawTerminal(mouse=True) as term:
        while True:
            cols, rows = terminal_size()
            img_rows = rows - _STATUS_H

            if dirty:
                bars = []
                _compute_layout(zoom_stack[-1], 0, 0, cols, bars)
                dirty = False

            _render_screen(
                term, bars, cols, rows,
                search, search_mode, search_buf,
                zoom_stack, html_path, total_samples,
            )

            key = term.read_key()

            # ── search mode ────────────────────────────────────────────────────
            if search_mode:
                if key == "enter":
                    search      = search_buf
                    search_mode = False
                    search_buf  = ""
                elif key == "esc":
                    search_mode = False
                    search_buf  = ""
                elif key in ("\x7f", "backspace"):
                    search_buf = search_buf[:-1]
                elif len(key) == 1 and key.isprintable():
                    search_buf += key
                # Refresh status line only — image unchanged
                continue

            # ── normal mode ────────────────────────────────────────────────────
            if key in ("q", "ctrl+c"):
                break

            elif key.startswith("click:"):
                _, row_s, col_s = key.split(":")
                click_row, click_col = int(row_s), int(col_s)
                if click_row < img_rows:
                    for bar in bars:
                        if bar.depth == click_row and bar.col0 <= click_col < bar.col1:
                            if bar.node.children:
                                zoom_stack.append(bar.node)
                                dirty = True
                            break

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
                search_buf  = search     # pre-fill with current filter

            elif key == "w" and html_path:
                _open_html(html_path)

            elif key in ("scroll_up",):
                pass    # reserved for future vertical pan

            elif key in ("scroll_down",):
                pass


def _open_html(path: str) -> None:
    import subprocess
    opener = "open" if sys.platform == "darwin" else "xdg-open"
    subprocess.Popen([opener, str(path)], stderr=subprocess.DEVNULL)
