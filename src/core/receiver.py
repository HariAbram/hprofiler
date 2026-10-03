"""
Collector side of the hook socket protocol (src/core/runner.py's
`hprofiler run`).

Two stages, so the profiled program is never slowed down by how fast
Python can parse:

  readers   one thread per hook connection: recv() and append the raw
            bytes to a spool file (and nothing else) -- C speed, so the
            hooks' drain threads (hooks/common/hp_transport.h) are never
            back-pressured by event parsing;
  parser    one thread: reads every spool as it grows, splits lines
            (UTF-8 decoded per complete line, never per chunk), parses
            records and appends events to the trace. Per-connection order
            is preserved, so a stk: record still follows its span:.

At the end of the run `finish()` waits for every connection to reach EOF
(bounded by `timeout_s`, default HPROFILER_RECEIVER_TIMEOUT_S or 120 s, for
connections a lingering child process keeps open) and for the parser to
ingest the whole backlog -- nothing is cut off by a fixed join timeout.

Nothing is dropped silently: malformed lines, unknown record kinds,
connections that end in the middle of a line, processing errors and
connections still open at the deadline are counted (with a few samples),
and the hooks' `xport:` transport status records are kept per
process/hook, including whether each one ended with its final drain. All of
it goes into trace.metadata.capture_health.
"""
from __future__ import annotations

import os
import socket
import struct
import threading
import time
from collections import Counter, deque
from typing import Callable

from . import gpu_activity

READ_CHUNK = 1 << 20
MAX_SAMPLES = 5
# Integer fields of an xport: status. Each status is cumulative for one
# process image + hook (image= is the hook's start time in that image, so
# the images before and after an exec() -- same pid -- are kept apart);
# the latest status of an image wins.
_XPORT_INT = ("final", "emitted", "sent", "bytes", "dropped_full", "dropped_lost", "oversize",
              "format_errors", "blocked_ns", "max_block_ns", "waits", "threads", "ring_kb",
              "reconnects", "send_errors", "sanitized")
XPORT_WIRE_VERSION = 1


def parse_xport(line: str) -> tuple[int, str, dict] | None:
    """xport:<version>:<pid>:<hook>:k=v,...  ->  (pid, hook, fields)."""
    parts = line.strip().split(":", 4)
    if len(parts) != 5 or parts[0] != "xport":
        return None
    try:
        version, pid = int(parts[1]), int(parts[2])
    except ValueError:
        return None
    fields: dict = {"version": version}
    for kv in parts[4].split(","):
        if "=" in kv:
            k, v = kv.split("=", 1)
            if k in _XPORT_INT:
                try:
                    fields[k] = int(v)
                except ValueError:
                    fields[k] = v
            else:
                fields[k] = v
    return pid, parts[3], fields


def _peer_pid(client: socket.socket) -> int:
    try:
        creds = client.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
        return struct.unpack("3i", creds)[0]
    except OSError:
        return 0


class _Conn:
    __slots__ = ("id", "sock", "path", "wfd", "rfd", "done", "bytes", "lines", "tail", "keys",
                 "final_keys", "mem", "spool_failed", "peer_pid", "closed_at_deadline")

    def __init__(self, cid: int, sock: socket.socket, path: str) -> None:
        self.id = cid
        self.sock = sock
        self.path = path
        self.wfd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        self.rfd = os.open(path, os.O_RDONLY)
        self.done = threading.Event()
        self.bytes = 0
        self.lines = 0
        self.tail = b""
        self.keys: set[tuple[int, str, str]] = set()   # (pid, hook, image) seen in xport records
        self.final_keys: set[tuple[int, str, str]] = set()
        self.mem: deque[bytes] = deque()                # spill when the spool cannot be written
        self.spool_failed = False
        self.peer_pid = _peer_pid(sock)
        self.closed_at_deadline = False


class Receiver:
    def __init__(self, server_sock: socket.socket, spool_dir: str, trace,
                 handle_line: Callable[[str], None], *, on_connect: Callable[[socket.socket], None] | None = None):
        self.server = server_sock
        self.spool_dir = spool_dir
        self.trace = trace
        self.handle_line = handle_line           # records other than xport:
        self.on_connect = on_connect
        self._conns: list[_Conn] = []
        self._conns_lock = threading.Lock()
        self._next_id = 0
        self._stop_accept = threading.Event()
        self._readers: list[threading.Thread] = []
        self._parser_stop = threading.Event()
        self._wake = threading.Event()
        # health
        self.malformed = 0
        self.malformed_samples: list[str] = []
        self.unknown: Counter = Counter()
        self.partial_tails = 0
        self.errors = 0
        self.error_samples: list[str] = []
        self.lines = 0
        self.bytes = 0
        self.connections = 0
        self.spool_errors = 0
        self.decode_replaced = 0
        self.transport: dict[tuple[int, str, str], dict] = {}
        self._received_by_key: Counter = Counter()
        self.ended_without_final: set[tuple[int, str, str]] = set()
        self.conns_without_status = 0
        self.open_at_deadline = 0

    # ── accepting / reading ─────────────────────────────────────────────
    def start(self) -> None:
        self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True,
                                               name="hprofiler-accept")
        self._accept_thread.start()
        self._parser = threading.Thread(target=self._parse_loop, daemon=True, name="hprofiler-parse")
        self._parser.start()

    def _accept_loop(self) -> None:
        while not self._stop_accept.is_set():
            try:
                client, _ = self.server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            self._register(client)

    def _register(self, client: socket.socket) -> None:
        client.setblocking(True)
        if self.on_connect is not None:
            try:
                self.on_connect(client)
            except Exception:
                pass
        with self._conns_lock:
            cid = self._next_id
            self._next_id += 1
            try:
                conn = _Conn(cid, client, os.path.join(self.spool_dir, f"conn{cid}.spool"))
            except OSError as exc:
                self._note_error(f"cannot create spool: {exc!r}")
                client.close()
                return
            self._conns.append(conn)
            self.connections += 1
            t = threading.Thread(target=self._read_loop, args=(conn,), daemon=True,
                                 name=f"hprofiler-conn{cid}")
            self._readers.append(t)
        t.start()

    def _accept_pending(self) -> None:
        """Accept connections still queued in the listen backlog -- a short-
        lived child that connected just before the program exited must not be
        lost because the accept loop had already stopped."""
        try:
            self.server.setblocking(False)
        except OSError:
            return
        while True:
            try:
                client, _ = self.server.accept()
            except (BlockingIOError, InterruptedError):
                break
            except OSError:
                break
            self._register(client)

    def _read_loop(self, conn: _Conn) -> None:
        try:
            while True:
                try:
                    data = conn.sock.recv(READ_CHUNK)
                except OSError:
                    break
                if not data:
                    break
                conn.bytes += len(data)
                if not conn.spool_failed:
                    try:
                        view = memoryview(data)
                        while view:
                            n = os.write(conn.wfd, view)
                            view = view[n:]
                    except OSError as exc:
                        conn.spool_failed = True
                        self.spool_errors += 1
                        self._note_error(f"spool write failed, buffering in memory: {exc!r}")
                        conn.mem.append(bytes(view))
                else:
                    conn.mem.append(data)
                self._wake.set()
        finally:
            try:
                conn.sock.close()
            except OSError:
                pass
            try:
                os.close(conn.wfd)
            except OSError:
                pass
            conn.done.set()
            self._wake.set()

    # ── parsing ─────────────────────────────────────────────────────────
    def _note_error(self, msg: str) -> None:
        self.errors += 1
        if len(self.error_samples) < MAX_SAMPLES:
            self.error_samples.append(msg[:300])

    def note_malformed(self, line: str) -> None:
        self.malformed += 1
        if len(self.malformed_samples) < MAX_SAMPLES:
            self.malformed_samples.append(line[:200])

    def note_unknown(self, kind: str) -> None:
        self.unknown[kind[:24]] += 1

    def _process(self, conn: _Conn, chunk: bytes) -> None:
        data = conn.tail + chunk if conn.tail else chunk
        end = data.rfind(b"\n")
        if end < 0:
            conn.tail = data
            return
        conn.tail = data[end + 1:]
        for raw in data[:end].split(b"\n"):
            if not raw:
                continue
            try:
                line = raw.decode("utf-8")
            except UnicodeDecodeError:
                line = raw.decode("utf-8", errors="replace")
                self.decode_replaced += 1
            conn.lines += 1
            self.lines += 1
            if line.startswith("xport:"):
                self._xport(conn, line)
                continue
            try:
                self.handle_line(line)
            except Exception as exc:     # never lose the rest of the connection
                self._note_error(f"{type(exc).__name__}: {exc} -- line: {line[:120]}")
        self.bytes += end + 1

    def _xport(self, conn: _Conn, line: str) -> None:
        conn.lines -= 1          # status records are not data
        self.lines -= 1
        parsed = parse_xport(line)
        if parsed is None:
            self.note_malformed(line)
            return
        pid, hook, fields = parsed
        key = (pid, hook, str(fields.get("image", "")))
        conn.keys.add(key)
        if fields.get("final") == 1:
            conn.final_keys.add(key)
        entry = self.transport.setdefault(key, {})
        entry.update(fields)

    def _finish_conn(self, conn: _Conn) -> None:
        if conn.tail:
            self.partial_tails += 1
            self.note_malformed(conn.tail.decode("utf-8", errors="replace"))
            conn.tail = b""
        if not conn.keys and conn.lines:
            self.conns_without_status += 1
        for key in conn.keys:
            self._received_by_key[key] += conn.lines
            if key not in conn.final_keys:
                self.ended_without_final.add(key)
            else:
                self.ended_without_final.discard(key)
        try:
            os.close(conn.rfd)
        except OSError:
            pass
        try:
            os.unlink(conn.path)
        except OSError:
            pass

    def _drain_conn(self, conn: _Conn) -> bool:
        """Process whatever this connection has spooled; True when it is
        complete (EOF reached and everything ingested)."""
        done = conn.done.is_set()          # read BEFORE draining: nothing is written after done
        while True:
            try:
                chunk = os.read(conn.rfd, READ_CHUNK)
            except OSError as exc:
                self._note_error(f"spool read failed: {exc!r}")
                chunk = b""
            if not chunk:
                break
            self._process(conn, chunk)
        while conn.mem:
            self._process(conn, conn.mem.popleft())
        if done:
            self._finish_conn(conn)
            return True
        return False

    def _parse_loop(self) -> None:
        while True:
            with self._conns_lock:
                active = list(self._conns)
            idle = True
            for conn in active:
                before = conn.lines
                if self._drain_conn(conn):
                    with self._conns_lock:
                        self._conns.remove(conn)
                if conn.lines != before:
                    idle = False
            if self._parser_stop.is_set():
                with self._conns_lock:
                    if not self._conns:
                        return
            if idle:
                self._wake.wait(0.005)
                self._wake.clear()

    # ── end of run ──────────────────────────────────────────────────────
    def backlog_bytes(self) -> int:
        with self._conns_lock:
            conns = list(self._conns)
        total = 0
        for c in conns:
            try:
                total += max(0, os.fstat(c.rfd).st_size - os.lseek(c.rfd, 0, os.SEEK_CUR))
            except OSError:
                pass
        return total

    def finish(self, timeout_s: float | None = None, progress: Callable[[str], None] | None = None) -> None:
        """Stop accepting, wait for every connection's EOF (at most
        timeout_s), then for the parser to ingest everything."""
        if timeout_s is None:
            try:
                timeout_s = float(os.environ.get("HPROFILER_RECEIVER_TIMEOUT_S", "120"))
            except ValueError:
                timeout_s = 120.0
        self._stop_accept.set()
        self._accept_thread.join(timeout=2)
        self._accept_pending()
        deadline = time.monotonic() + timeout_s
        announced = 0.0
        while time.monotonic() < deadline:
            with self._conns_lock:
                alive = [t for t in self._readers if t.is_alive()]
            if not alive:
                break
            alive[0].join(timeout=0.5)
            self._accept_pending()            # (an accept thread stuck in accept() may still add one)
            if progress and time.monotonic() - announced > 5:
                announced = time.monotonic()
                progress("waiting for hook connections to close...")
        with self._conns_lock:
            still_open = [c for c in self._conns if not c.done.is_set()]
        for c in still_open:
            c.closed_at_deadline = True
            self.open_at_deadline += 1
            try:
                c.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        with self._conns_lock:
            readers = list(self._readers)
        for t in readers:
            t.join(timeout=5)
        self._parser_stop.set()
        self._wake.set()
        announced = time.monotonic()
        while self._parser.is_alive():
            self._parser.join(timeout=1.0)
            if progress and self._parser.is_alive() and time.monotonic() - announced > 3:
                announced = time.monotonic()
                progress(f"ingesting buffered events: {self.backlog_bytes() / 1e6:.1f} MB left")

    def health(self) -> dict:
        """What to store as trace.metadata.capture_health["receiver"] /
        ["transport"] -- see summary/warnings in capture_warnings()."""
        transport = {}
        images = Counter((pid, hook) for pid, hook, _ in self.transport)
        seen: Counter = Counter()
        for k in sorted(self.transport, key=lambda k: (k[0], k[1], int(k[2]) if k[2].isdigit() else 0)):
            pid, hook, _ = k
            e = dict(self.transport[k])
            e["received"] = int(self._received_by_key.get(k, 0))
            e["ended_without_final"] = k in self.ended_without_final
            seen[(pid, hook)] += 1
            # "pid/hook", or "pid/hook#n" for the n-th image of a pid that exec()ed
            name = f"{pid}/{hook}" if images[(pid, hook)] == 1 else f"{pid}/{hook}#{seen[(pid, hook)]}"
            transport[name] = e
        return {
            "receiver": {
                "connections": self.connections, "lines": self.lines, "bytes": self.bytes,
                "malformed": self.malformed, "malformed_samples": self.malformed_samples,
                "unknown_kinds": dict(self.unknown), "partial_tails": self.partial_tails,
                "errors": self.errors, "error_samples": self.error_samples,
                "spool_errors": self.spool_errors, "decode_replaced": self.decode_replaced,
                "connections_without_status": self.conns_without_status,
                "open_at_deadline": self.open_at_deadline,
            },
            "transport": transport,
        }


def run_warnings(metadata) -> list[str]:
    """Everything that makes a saved trace incomplete or partly estimated:
    the capture itself (capture_warnings) plus degraded native GPU tracing
    (gpu_activity.degraded_warnings). Shown after a run, by `summary`, and
    on the GUI Overview."""
    return (capture_warnings(getattr(metadata, "capture_health", None) or {})
            + gpu_activity.degraded_warnings(getattr(metadata, "device_activity", None) or {}))


def capture_warnings(health: dict) -> list[str]:
    """Human-readable problems recorded in trace.metadata.capture_health
    (empty when the capture was clean). Shared by the run summary, the
    `summary` command, the TUI and the GUI."""
    out: list[str] = []
    if not health:
        return out
    state = health.get("state")
    if state == "interrupted":
        out.append("capture did not complete (the collector was interrupted): events after the "
                   "last stored batch and the run metadata may be missing")
    load = health.get("load") or {}
    if load.get("truncated"):
        missing = "later events" + (" and the disassembly" if load.get("disasm_missing") else "")
        out.append(f"the trace file is truncated (interrupted export or copy): {load.get('events', 0)} "
                   f"complete events loaded, {missing} are missing"
                   + ("; the run metadata is missing too" if load.get("metadata_missing") else ""))
    if load.get("bad_lines"):
        out.append(f"{load['bad_lines']} unreadable event line(s) skipped while loading the trace file")
    rec = health.get("receiver") or {}
    for key, label in (("malformed", "malformed record(s) skipped"),
                       ("partial_tails", "connection(s) ended in the middle of a record"),
                       ("errors", "record(s) failed to process"),
                       ("spool_errors", "spool write failure(s) (buffered in memory instead)"),
                       ("open_at_deadline", "hook connection(s) still open when the capture ended "
                                            "(a child process outlived the program?)")):
        if rec.get(key):
            out.append(f"{rec[key]} {label}")
    if rec.get("unknown_kinds"):
        kinds = ", ".join(f"{k} x{v}" for k, v in sorted(rec["unknown_kinds"].items()))
        out.append(f"unknown record kinds ignored: {kinds}")
    if rec.get("decode_replaced"):
        out.append(f"{rec['decode_replaced']} record(s) with invalid UTF-8 (replaced characters)")
    if rec.get("connections_without_status"):
        out.append(f"{rec['connections_without_status']} hook connection(s) ended without a transport "
                   "status (process crashed, was killed, or called _exit/exec): its last buffered "
                   "events may be missing")
    for key, t in sorted((health.get("transport") or {}).items()):
        pid, _, hook = key.partition("/")
        bits = []
        for k, label in (("dropped_full", "dropped (ring full after waiting)"),
                         ("dropped_lost", "lost (collector unreachable)"),
                         ("oversize", "rejected as oversize (> 64 KiB)"),
                         ("format_errors", "failed to format")):
            if t.get(k):
                bits.append(f"{t[k]} {label}")
        if t.get("ended_without_final"):
            bits.append("ended without its final drain (crash, kill, _exit or exec): buffered events "
                        "may be missing")
        elif t.get("final") == 1 and "sent" in t and t.get("received", 0) < t["sent"]:
            bits.append(f"{t['sent'] - t['received']} record(s) sent but not received")
        if t.get("blocked_ns", 0) > 100_000_000:
            bits.append(f"producers waited {t['blocked_ns'] / 1e9:.2f} s in total for ring space "
                        f"(max {t.get('max_block_ns', 0) / 1e6:.0f} ms) -- the collector fell behind")
        if bits:
            out.append(f"{hook} hook, pid {pid}: " + "; ".join(bits))
    return out
