"""
Inline image display and raw terminal I/O for HPC-cluster-friendly TUI.

Supports:
  • Kitty graphics protocol  (kitty, WezTerm, Ghostty, …)
  • Sixel                    (xterm -ti vt340, mlterm, …)
  • iTerm2 inline images     (iTerm2, WezTerm)
  • Halfblock fallback       (any terminal — low quality)

Auto-detection tries each protocol in the order above.
"""
from __future__ import annotations

import base64
import os
import sys
import termios
import tty
from typing import Callable, Optional


# ── Terminal detection ────────────────────────────────────────────────────────

def _detect_protocol() -> str:
    """Return 'kitty', 'iterm2', 'sixel', or 'none'."""
    env = os.environ

    # Kitty
    if env.get("KITTY_WINDOW_ID") or "kitty" in env.get("TERM", "").lower():
        return "kitty"

    prog = env.get("TERM_PROGRAM", "")
    term = env.get("TERM", "")

    # WezTerm supports Kitty protocol
    if "WezTerm" in prog:
        return "kitty"

    # Ghostty supports Kitty protocol
    if "ghostty" in prog.lower() or "ghostty" in term.lower():
        return "kitty"

    # iTerm2
    if "iTerm" in prog or env.get("ITERM_SESSION_ID"):
        return "iterm2"

    # Sixel: xterm with vt340 or explicit sixel capability
    if "xterm" in term or "mlterm" in term:
        return "sixel"

    # mintty (Windows)
    if "mintty" in prog.lower():
        return "sixel"

    return "none"


_PROTOCOL: Optional[str] = None   # cached after first call

def get_protocol() -> str:
    global _PROTOCOL
    if _PROTOCOL is None:
        _PROTOCOL = _detect_protocol()
    return _PROTOCOL


def terminal_size() -> tuple[int, int]:
    """Return (cols, rows) of the current terminal."""
    try:
        sz = os.get_terminal_size()
        return sz.columns, sz.lines
    except OSError:
        return 80, 24


def terminal_pixel_size() -> tuple[int, int]:
    """
    Return the terminal window's physical pixel dimensions (width_px, height_px).

    Reads the ``ws_xpixel`` / ``ws_ypixel`` fields from the TIOCGWINSZ ioctl,
    which WezTerm, kitty, and most modern terminals populate correctly.  Falls
    back to a character-count estimate (10 × 20 px per cell) when the ioctl is
    unavailable or returns zero — the fallback is intentionally generous so that
    images are rendered at high enough resolution.
    """
    import fcntl
    import struct
    try:
        # struct winsize { ws_row, ws_col, ws_xpixel, ws_ypixel } — 4 × uint16
        buf = fcntl.ioctl(sys.stdout.fileno(), termios.TIOCGWINSZ, b"\x00" * 8)
        _, _, xpx, ypx = struct.unpack("HHHH", buf)
        if xpx > 0 and ypx > 0:
            return xpx, ypx
    except Exception:
        pass
    cols, rows = terminal_size()
    return cols * 10, rows * 20   # conservative HiDPI fallback


# ── Display functions ─────────────────────────────────────────────────────────

def _display_kitty(png: bytes, cols: int, rows: int) -> None:
    """Kitty graphics protocol: chunked base64-encoded PNG."""
    data = base64.standard_b64encode(png).decode()
    chunk = 4096
    chunks = [data[i : i + chunk] for i in range(0, len(data), chunk)]

    # 'a=T' = transmit+display, 'f=100' = PNG, 'q=2' = suppress OK
    # 'c,r' = size hint in terminal cells
    params = f"a=T,f=100,q=2,c={cols},r={rows}"
    if len(chunks) == 1:
        sys.stdout.write(f"\x1b_G{params};{chunks[0]}\x1b\\")
    else:
        sys.stdout.write(f"\x1b_G{params},m=1;{chunks[0]}\x1b\\")
        for part in chunks[1:-1]:
            sys.stdout.write(f"\x1b_Gm=1;{part}\x1b\\")
        sys.stdout.write(f"\x1b_Gm=0;{chunks[-1]}\x1b\\")
    sys.stdout.flush()


def _display_iterm2(png: bytes) -> None:
    """iTerm2 inline images protocol."""
    data = base64.standard_b64encode(png).decode()
    size = len(png)
    sys.stdout.write(
        f"\x1b]1337;File=inline=1;size={size};preserveAspectRatio=1:{data}\a"
    )
    sys.stdout.flush()


def _display_sixel(png: bytes) -> None:
    """
    Convert PNG to sixel using Pillow (if available), else skip.
    Sixel encoding: each column of 6 pixels → one sixel character (0x3F + bitmask).
    We use a fast 16-colour dither via Pillow.
    """
    try:
        from PIL import Image
        import io

        img = Image.open(io.BytesIO(png)).convert("RGB")
        w, h = img.size
        # Quantize to 16 colours for basic sixel
        img_q = img.quantize(colors=16, method=Image.Quantize.FASTOCTREE)
        pal = img_q.getpalette()  # [r,g,b, r,g,b, ...]

        # Sixel header
        out = ["\x1bPq"]

        # Colour registers
        for i in range(16):
            r, g, b = pal[i*3], pal[i*3+1], pal[i*3+2]
            # Convert to percentage (0-100)
            out.append(f"#{i};2;{r*100//255};{g*100//255};{b*100//255}")

        pixels = img_q.load()
        for band_top in range(0, h, 6):
            band_h = min(6, h - band_top)
            for ci in range(16):
                out.append(f"#{ci}")
                run_char = None
                run_len  = 0
                for x in range(w):
                    bits = 0
                    for dy in range(band_h):
                        if pixels[x, band_top + dy] == ci:
                            bits |= (1 << dy)
                    ch = chr(0x3F + bits)
                    if ch == run_char:
                        run_len += 1
                    else:
                        if run_char is not None:
                            if run_len > 3:
                                out.append(f"!{run_len}{run_char}")
                            else:
                                out.append(run_char * run_len)
                        run_char, run_len = ch, 1
                if run_char is not None:
                    if run_len > 3:
                        out.append(f"!{run_len}{run_char}")
                    else:
                        out.append(run_char * run_len)
                out.append("$")  # CR (return to column 0)
            out.append("-")  # LF (advance 6 rows)
        out.append("\x1b\\")  # ST
        sys.stdout.write("".join(out))
        sys.stdout.flush()
    except Exception:
        # Sixel not possible — show a placeholder
        sys.stdout.write("[image could not be displayed — install Pillow for sixel]\n")
        sys.stdout.flush()


def display_image(png: bytes, cols: int, rows: int) -> None:
    """Display *png* bytes inline in the terminal using the best available protocol."""
    proto = get_protocol()
    if proto == "kitty":
        _display_kitty(png, cols, rows)
    elif proto == "iterm2":
        _display_iterm2(png)
    elif proto == "sixel":
        _display_sixel(png)
    else:
        sys.stdout.write(
            "[No inline image support detected.\n"
            " Run in kitty, WezTerm, or xterm with sixel support.]\n"
        )
        sys.stdout.flush()


# ── Raw terminal I/O ──────────────────────────────────────────────────────────

class RawTerminal:
    """
    Context manager that switches the terminal to raw mode and uses the
    alternate screen buffer so the output is cleanly separated from the
    rest of the shell session.

    Parameters
    ----------
    mouse : bool
        When True, enable SGR mouse button reporting so that ``read_key()``
        returns ``"click:row:col"`` tokens for left-button clicks.

    Usage::

        with RawTerminal(mouse=True) as term:
            while True:
                key = term.read_key()
                if key.startswith("click:"):
                    _, row, col = key.split(":")
                    ...
                elif key == "q":
                    break
    """

    def __init__(self, mouse: bool = False) -> None:
        self._mouse = mouse

    def __enter__(self) -> "RawTerminal":
        self._fd = sys.stdin.fileno()
        self._old = termios.tcgetattr(self._fd)
        # Alternate screen + hide cursor
        seq = "\x1b[?1049h\x1b[?25l"
        if self._mouse:
            # Button-event mouse reporting + SGR encoding (handles cols > 223)
            seq += "\x1b[?1002h\x1b[?1006h"
        sys.stdout.write(seq)
        sys.stdout.flush()
        tty.setraw(self._fd)
        return self

    def __exit__(self, *_) -> None:
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)
        seq = ""
        if self._mouse:
            seq += "\x1b[?1002l\x1b[?1006l"
        seq += "\x1b[?25h\x1b[?1049l"
        sys.stdout.write(seq)
        sys.stdout.flush()

    def clear(self) -> None:
        sys.stdout.write("\x1b[2J\x1b[H")
        sys.stdout.flush()

    def goto(self, row: int, col: int = 0) -> None:
        """Move cursor to (row, col), 0-indexed."""
        sys.stdout.write(f"\x1b[{row + 1};{col + 1}H")
        sys.stdout.flush()

    def write(self, text: str, style: str = "") -> None:
        if style:
            sys.stdout.write(f"\x1b[{style}m{text}\x1b[0m")
        else:
            sys.stdout.write(text)
        sys.stdout.flush()

    def writeln(self, text: str, style: str = "") -> None:
        self.write(text + "\r\n", style)

    def read_key(self) -> str:
        """
        Block until input arrives and return a string token.

        Regular keys  →  the character itself ('q', '+', 'r', …)
        Special keys  →  'up', 'down', 'left', 'right', 'enter', 'esc', 'ctrl+c'
        Mouse clicks  →  'click:{row}:{col}'  (0-indexed; left button only)
        Mouse scroll  →  'scroll_up' / 'scroll_down'
        """
        import select

        ch = sys.stdin.read(1)
        if ch != "\x1b":
            if ch in ("\x03", "\x04"):
                return "ctrl+c"
            if ch in ("\r", "\n"):
                return "enter"
            return ch

        # Read the rest of the escape sequence with a 50 ms timeout so a bare
        # Esc (no following bytes) is returned quickly as "esc".
        seq: list[str] = []
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], 0.05)
            if not ready:
                break
            c = sys.stdin.read(1)
            seq.append(c)
            # CSI sequences terminate with a letter or '~'
            if c.isalpha() or c == "~":
                break

        s = "".join(seq)

        if s == "[A": return "up"
        if s == "[B": return "down"
        if s == "[C": return "right"
        if s == "[D": return "left"

        # SGR mouse: \x1b[<{btn};{col};{row}M (press) or m (release)
        if s.startswith("[<"):
            body = s[2:]
            press = body.endswith("M")
            body  = body[:-1]
            try:
                btn_s, col_s, row_s = body.split(";")
                btn, col, row = int(btn_s), int(col_s), int(row_s)
                if press:
                    if btn == 0:                     # left click
                        return f"click:{row - 1}:{col - 1}"
                    if btn == 64:                    # scroll wheel up
                        return "scroll_up"
                    if btn == 65:                    # scroll wheel down
                        return "scroll_down"
            except (ValueError, IndexError):
                pass
            return "mouse"

        return "esc"


# ── Spinner for slow renders ──────────────────────────────────────────────────

def with_spinner(term: RawTerminal, row: int, msg: str, fn: Callable) -> object:
    """Run *fn()* while displaying a spinner at *row*."""
    import threading

    result_box: list = [None]
    error_box:  list = [None]
    done = threading.Event()

    def _run():
        try:
            result_box[0] = fn()
        except Exception as e:
            error_box[0] = e
        finally:
            done.set()

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    i = 0
    while not done.wait(timeout=0.1):
        term.goto(row)
        term.write(f"  {frames[i % len(frames)]}  {msg}", style="2")
        i += 1

    if error_box[0] is not None:
        raise error_box[0]
    return result_box[0]
