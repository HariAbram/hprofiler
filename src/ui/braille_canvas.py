"""
Braille sub-cell canvas for drawing lines in a character-grid terminal UI.

Unicode Braille Patterns (U+2800-U+28FF) pack 2x4 dots per character cell,
giving ~8x the effective resolution of plain characters for line art --
the same technique terminal-plotting libraries like drawille/plotext use
to get "canvas-like" output. Pure Unicode text: no terminal graphics
protocol (Sixel, Kitty, iTerm2 inline images) required, so it renders
correctly over any SSH/tmux/screen session, including a bare HPC cluster
login-node terminal -- unlike those protocols, which need the terminal
emulator (and any multiplexer in between) to explicitly support them and
degrade to garbled escape-code text, not a graceful fallback, when they
don't.

Used by TimelineWidget (src/ui/app.py) to draw MPI/NCCL communication
connector lines across lanes -- see its _ConnectorLayer.
"""

from __future__ import annotations

# Dot-position (sub_x, sub_y) within a cell -> bit offset in the cell's
# 8-bit mask, per the standard Braille terminal-graphics dot layout (the
# same one drawille/plotext use):
#   (0,0)=bit0  (1,0)=bit3
#   (0,1)=bit1  (1,1)=bit4
#   (0,2)=bit2  (1,2)=bit5
#   (0,3)=bit6  (1,3)=bit7
_DOT_BIT = {
    (0, 0): 0, (0, 1): 1, (0, 2): 2, (0, 3): 6,
    (1, 0): 3, (1, 1): 4, (1, 2): 5, (1, 3): 7,
}
BRAILLE_BASE = 0x2800
DOTS_PER_CELL_X = 2
DOTS_PER_CELL_Y = 4


class BrailleCanvas:
    """A `cols` x `rows` character-cell canvas addressable at
    `2*cols` x `4*rows` dot resolution.

    Sparse by design (a dict keyed by cell, not a dense array) since a
    connector-line overlay only ever touches a small fraction of a
    Timeline's cells -- most of the canvas stays completely empty and
    should cost nothing.

    When dots of different styles land in the same cell, the most
    recently drawn one wins for that cell's overall style: a terminal
    character cell renders as one glyph with one style, so sub-cell color
    mixing isn't possible -- callers drawing multiple overlapping
    connectors in different colors should be aware the last line drawn
    "wins" visually at any shared cell, not blend.
    """

    def __init__(self, cols: int, rows: int):
        self.cols = cols
        self.rows = rows
        self._dots: dict[tuple[int, int], int] = {}    # (cell_x, cell_y) -> 8-bit dot mask
        self._style: dict[tuple[int, int], str] = {}    # (cell_x, cell_y) -> style string

    def set_dot(self, x: int, y: int, style: str = "") -> None:
        """x, y are DOT coordinates (0 <= x < 2*cols, 0 <= y < 4*rows).
        Out-of-range coordinates are silently clipped (ignored), matching
        how the rest of TimelineWidget's rendering clips off-screen
        content rather than erroring on it."""
        cell_x, sub_x = divmod(x, DOTS_PER_CELL_X)
        cell_y, sub_y = divmod(y, DOTS_PER_CELL_Y)
        if not (0 <= cell_x < self.cols and 0 <= cell_y < self.rows):
            return
        bit = _DOT_BIT[(sub_x, sub_y)]
        key = (cell_x, cell_y)
        self._dots[key] = self._dots.get(key, 0) | (1 << bit)
        if style:
            self._style[key] = style

    def line(self, x0: int, y0: int, x1: int, y1: int, style: str = "") -> None:
        """Bresenham's line algorithm in dot-coordinate space (integer-only,
        no floating point accumulation error over long lines)."""
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        dx = abs(x1 - x0)
        sx = 1 if x0 < x1 else -1
        dy = -abs(y1 - y0)
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        x, y = x0, y0
        while True:
            self.set_dot(x, y, style)
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x += sx
            if e2 <= dx:
                err += dx
                y += sy

    def cell(self, cell_x: int, cell_y: int) -> tuple[str, str] | None:
        """Returns (braille_char, style) for one cell, or None if no dot
        in it was ever set -- callers should leave whatever was already
        rendered there untouched in that case, not overwrite it with a
        blank Braille cell (U+2800, which is a real glyph -- "all dots
        off" -- not "no character", and would visibly blank out content)."""
        key = (cell_x, cell_y)
        mask = self._dots.get(key)
        if mask is None:
            return None
        return chr(BRAILLE_BASE + mask), self._style.get(key, "")

    def cells(self):
        """Iterates (cell_x, cell_y, char, style) for every non-empty cell."""
        for (cx, cy), mask in self._dots.items():
            yield cx, cy, chr(BRAILLE_BASE + mask), self._style.get((cx, cy), "")
