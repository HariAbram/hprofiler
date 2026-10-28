"""
Process runner: starts the target with the selected backends' hooks injected
(LD_PRELOAD libraries, OMP_TOOL_LIBRARIES for the OMPT tool, perf and likwid
alongside or around the process), collects the hooks' records from the
HPROFILER_SOCKET Unix socket (src/core/receiver.py) into the trace store, and
post-processes the run (GPU correlation, perf samples, source lines,
counters, disassembly).
"""

from __future__ import annotations
import datetime
import os
import re
import resource
from collections import Counter, deque
import shutil
import sys
import socket
import subprocess
import threading
import time
import tempfile
import struct
import socket as _socket
from pathlib import Path
from typing import Callable, Optional

from .events import SpanEvent, InstantEvent, CounterEvent, Category, AnyEvent  # noqa: F401
from .trace import Trace, TraceMetadata
from .store import TraceStore
from . import gpu_activity

from .hookdir import hooks_dir
HOOKS_DIR = hooks_dir()

# Wire records (DOCUMENTATION.md, "Wire protocol"): span:, inst:, ctr:,
# stk:, gpuact:, pcsa:, xport:.
#
# Tags are "key=val[,key=val...]" without colons: if the text after the LAST
# colon matches, it is the tags segment and everything before it is the name
# (which may itself contain colons, e.g. "ns::kernel"); otherwise the whole
# remainder is the name.
_TAGS_RE = re.compile(r"^[^,=]+=[^,]*(,[^,=]+=[^,]*)*$")


def _split_name_tags(rest: str) -> tuple[str, str]:
    if ":" not in rest:
        return rest, ""
    name, maybe_tags = rest.rsplit(":", 1)
    if _TAGS_RE.match(maybe_tags):
        return name, maybe_tags
    return rest, ""


def _parse_record(line: str) -> AnyEvent | None:
    try:
        parts = line.strip().split(":", 6)
        kind = parts[0]
        if kind == "span" and len(parts) >= 7:
            _, cat, pid, tid, start_ns, dur_ns, rest = parts[0], parts[1], int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5]), parts[6]
            name, tags_str = _split_name_tags(rest)
            tags: dict = {}
            if tags_str:
                for kv in tags_str.split(","):
                    if "=" in kv:
                        k, v = kv.split("=", 1)
                        tags[k] = v
            span_id       = tags.pop("sid",  "")
            parent_span_id = tags.pop("psid", "")
            return SpanEvent(
                name=name,
                category=Category(cat) if cat in Category._value2member_map_ else Category.OTHER,
                start_ns=start_ns,
                duration_ns=dur_ns,
                pid=pid,
                tid=tid,
                tags=tags,
                span_id=span_id,
                parent_span_id=parent_span_id,
            )
        if kind == "span" and len(parts) == 6:
            _, cat, pid, tid, start_ns, dur_ns_name = parts
            dur_ns_name_parts = dur_ns_name.split(":", 1)
            dur_ns = int(dur_ns_name_parts[0])
            name = dur_ns_name_parts[1] if len(dur_ns_name_parts) > 1 else ""
            return SpanEvent(
                name=name, category=Category(cat) if cat in Category._value2member_map_ else Category.OTHER,
                start_ns=int(start_ns), duration_ns=dur_ns, pid=int(pid), tid=int(tid),
            )
        if kind == "inst" and len(parts) >= 6:
            cat, pid, tid, ts_ns = parts[1], int(parts[2]), int(parts[3]), int(parts[4])
            # inst: has no dur_ns field, so name[:tags] starts at parts[5];
            # split(":", 6) may have split inside it, so rejoin.
            rest = ":".join(parts[5:])
            name, tags_str = _split_name_tags(rest)
            tags: dict = {}
            if tags_str:
                for kv in tags_str.split(","):
                    if "=" in kv:
                        k, v = kv.split("=", 1)
                        tags[k] = v
            return InstantEvent(
                name=name,
                category=Category(cat) if cat in Category._value2member_map_ else Category.OTHER,
                timestamp_ns=ts_ns,
                pid=pid,
                tid=tid,
                tags=tags,
            )
        if kind == "ctr" and len(parts) >= 6:
            _, cat, pid, ts_ns, name, value = parts[:6]
            unit = parts[6] if len(parts) > 6 else ""
            return CounterEvent(
                name=name,
                category=Category(cat) if cat in Category._value2member_map_ else Category.OTHER,
                timestamp_ns=int(ts_ns),
                value=float(value),
                unit=unit,
                pid=int(pid),
            )
    except Exception:
        pass
    return None


# (pid, tid) -> recent spans, to attach a stk: record to its span:. A stk:
# record follows its span on the same connection, but every hook library has
# its own connection, so another hook's span from the same thread can arrive
# in between; match by start_ns within a short ring.
RECENT_SPANS_PER_THREAD = 8


def _remember_recent_span(
    recent: dict[tuple[int, int], "deque[SpanEvent]"], ev: SpanEvent,
    maxlen: int = RECENT_SPANS_PER_THREAD,
) -> None:
    key = (ev.pid, ev.tid)
    ring = recent.get(key)
    if ring is None:
        ring = deque(maxlen=maxlen)
        recent[key] = ring
    ring.append(ev)


def _find_recent_span(
    recent: dict[tuple[int, int], "deque[SpanEvent]"],
    pid: int, tid: int, start_ns: int,
) -> SpanEvent | None:
    ring = recent.get((pid, tid))
    if ring is None:
        return None
    for candidate in reversed(ring):
        if candidate.start_ns == start_ns:
            return candidate
    return None


def _static_cuda_binary(command: list[str]) -> str:
    """The program in `command` linked with the static CUDA runtime, or "".

    Checks the command and every executable ELF file named in its arguments:
    the command is often a wrapper (a script, env, srun, mpirun) whose own
    binary says nothing about the program it starts."""
    candidates = [shutil.which(command[0]) or command[0]]
    candidates += [a for a in command[1:] if os.path.isfile(a) and os.access(a, os.X_OK)]
    for target in candidates:
        try:
            with open(target, "rb") as f:
                if f.read(4) != b"\x7fELF":
                    continue
            r = subprocess.run(["ldd", target], capture_output=True, text=True, timeout=5)
            if "libcudart" in r.stdout + r.stderr or "libcuda.so" in r.stdout + r.stderr:
                continue                      # the shared runtime: the hook interposes it
            nm_r = subprocess.run(["nm", target], capture_output=True, text=True, timeout=10)
            if "T cudaLaunchKernel" in nm_r.stdout or "T cuLaunchKernel" in nm_r.stdout:
                return target
        except (OSError, subprocess.SubprocessError):
            continue
    return ""


def clock_domain() -> dict:
    """The clock this host's hooks timestamp with: CLOCK_MONOTONIC, which is
    one clock per kernel boot (boot_id) and time namespace. Two traces with
    the same boot_id and namespace offset are on the same clock."""
    def first_line(path: str) -> str:
        try:
            with open(path) as f:
                return f.readline().strip()
        except OSError:
            return ""
    timens = 0
    try:
        with open("/proc/self/timens_offsets") as f:      # Linux >= 5.6
            for line in f:
                parts = line.split()
                if len(parts) == 3 and parts[0] == "monotonic":
                    timens = int(parts[1]) * 1_000_000_000 + int(parts[2])
    except (OSError, ValueError):
        pass
    return {"clock": "CLOCK_MONOTONIC",
            "boot_id": first_line("/proc/sys/kernel/random/boot_id"),
            "clocksource": first_line("/sys/devices/system/clocksource/clocksource0/current_clocksource"),
            "timens_monotonic_ns": timens}


def _peer_process(client: "socket.socket") -> tuple[int, str]:
    """(pid, real executable) of the process on the other end of this
    connection (SO_PEERCRED, kernel-verified, then /proc/<pid>/exe). CUDA/ROCm
    AoT disassembly reads the whole binary, and command[0] is a launcher
    (srun, mpirun) when one wraps the program; hooks connect from inside the
    real process, so its credentials name the right binary. (0, "") on any
    failure (process gone, permissions, non-Linux)."""
    import struct
    try:
        creds = client.getsockopt(
            socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
        peer_pid, _uid, _gid = struct.unpack("3i", creds)
        return peer_pid, os.readlink(f"/proc/{peer_pid}/exe")
    except Exception:
        return 0, ""


def _peer_real_exe(client: "socket.socket") -> str:
    return _peer_process(client)[1]


def disasm_binaries(exes_by_pid: "dict[int, list[str]] | set[str] | None", gpu_pids: "set[int]") -> list[str]:
    """The executables whose ahead-of-time GPU code is disassembled (all of
    them, merged), in a fixed order: those of the processes that emitted
    kernel launches or device activity, else every connected executable --
    a wrapper or launcher (env, a shell, mpirun) connects too when the hooks
    are preloaded into it, and its binary has no GPU code. A process that
    exec()s (env) connects once per image under one pid: its last image,
    the program, comes first (it is also the binary for CPU code). Never in
    set order: string hashing differs per Python process. A plain set
    (older callers) has no pids: every executable, sorted."""
    if not exes_by_pid:
        return []
    if not isinstance(exes_by_pid, dict):
        return sorted(exes_by_pid)
    pids = sorted(p for p in exes_by_pid if p in gpu_pids) or sorted(exes_by_pid)
    last = sorted({exes_by_pid[p][-1] for p in pids if exes_by_pid[p]})
    rest = sorted({e for p in pids for e in exes_by_pid[p]} - set(last))
    return last + rest


# Linux sun_path holds 108 bytes including the terminating NUL.
_SUN_PATH_MAX = 107


class _CollectWait:
    """Stands in for the child process in node-collector mode: wait() returns
    on SIGINT/SIGTERM, or idle_exit_s seconds after the last hook
    connection closed (once at least one was seen)."""
    pid = 0
    returncode = None

    def __init__(self, receiver, idle_exit_s: float | None) -> None:
        self.receiver = receiver
        self.idle_exit_s = idle_exit_s

    def wait(self) -> None:
        import signal
        stop = threading.Event()
        old = {}
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGINT, signal.SIGTERM):
                old[sig] = signal.signal(sig, lambda *_a: stop.set())
        idle_since = None
        try:
            while not stop.is_set():
                if self.receiver.connections and self.receiver.open_connections() == 0:
                    idle_since = idle_since or time.monotonic()
                    if self.idle_exit_s is not None and time.monotonic() - idle_since >= self.idle_exit_s:
                        break
                else:
                    idle_since = None
                stop.wait(0.2)
        finally:
            for sig, handler in old.items():
                signal.signal(sig, handler)


def _open_collector_socket(address: str | None = None):
    """The collector's listening AF_UNIX socket.

    Returns (socket, HPROFILER_SOCKET value, private spool directory,
    socket path or None). The socket lives in a private directory under
    $TMPDIR. A long $TMPDIR (batch jobs: /scratch/<job>/...) would exceed
    the 107-byte socket path limit, so the socket then goes to a private
    directory under /tmp, and when even that is not possible to Linux's
    abstract namespace (HPROFILER_SOCKET="@<name>"; the hooks connect to it
    by that name, and the receiver accepts only this user's processes).
    The spool directory stays under $TMPDIR.

    `address` (node-collector mode): bind exactly there instead -- "@name"
    for an abstract socket (node-local, no file), else a path (an existing
    stale socket file is replaced)."""
    import secrets
    import socket as sock_mod
    spool_dir = tempfile.mkdtemp(prefix="hprofiler_")
    if address:
        s = sock_mod.socket(sock_mod.AF_UNIX, sock_mod.SOCK_STREAM)
        if address.startswith("@"):
            s.bind("\0" + address[1:])
            path = None
        else:
            path = os.path.abspath(address)
            if len(os.fsencode(path)) > _SUN_PATH_MAX:
                s.close()
                raise OSError(f"collector socket path longer than {_SUN_PATH_MAX} bytes: {path} "
                              f"(use an abstract socket, @name)")
            try:
                st = os.lstat(path)
                import stat as _stat
                if _stat.S_ISSOCK(st.st_mode):
                    os.unlink(path)          # a stale socket of an earlier collector
            except FileNotFoundError:
                pass
            s.bind(path)
        s.listen(1024)
        s.settimeout(0.5)
        return s, (address if path is None else path), spool_dir, path
    candidates = [spool_dir]
    if len(os.fsencode(os.path.join(spool_dir, "s.sock"))) > _SUN_PATH_MAX:
        try:
            candidates.append(tempfile.mkdtemp(prefix="hprofiler_", dir="/tmp"))
        except OSError:
            pass
    for d in candidates:
        path = os.path.join(d, "s.sock")
        if len(os.fsencode(path)) > _SUN_PATH_MAX:
            continue
        s = sock_mod.socket(sock_mod.AF_UNIX, sock_mod.SOCK_STREAM)
        try:
            s.bind(path)
        except OSError:
            s.close()
            continue
        if d != spool_dir:
            print(f"[hprofiler] TMPDIR is too long for a socket path; collector socket: {path}",
                  file=sys.stderr)
        for other in candidates:
            if other not in (d, spool_dir):
                try:
                    os.rmdir(other)
                except OSError:
                    pass
        s.listen(128)        # many ranks/processes may connect at once
        s.settimeout(0.5)
        return s, path, spool_dir, path
    for other in candidates[1:]:
        try:
            os.rmdir(other)
        except OSError:
            pass
    name = f"hprofiler-{os.getpid()}-{secrets.token_hex(8)}"
    s = sock_mod.socket(sock_mod.AF_UNIX, sock_mod.SOCK_STREAM)
    s.bind("\0" + name)
    print(f"[hprofiler] no usable socket path under TMPDIR or /tmp; using the abstract "
          f"socket @{name}", file=sys.stderr)
    s.listen(128)
    s.settimeout(0.5)
    return s, "@" + name, spool_dir, None


def hook_environment(backends: list[str], sock_address: str, command: list[str] | None = None,
                     env_extra: dict[str, str] | None = None, base_env: dict[str, str] | None = None) -> dict:
    """The environment a program needs for these backends' hooks to report
    to the collector at `sock_address` (HPROFILER_SOCKET): LD_PRELOAD,
    OMP_TOOL_LIBRARIES and each backend's variables. Used by `hprofiler run`
    and `hprofiler exec` (node-collector mode)."""
    env = dict(os.environ if base_env is None else base_env)
    env["HPROFILER_SOCKET"] = sock_address
    env.update(env_extra or {})
    # --python-jit: Python JIT compilers' (Triton's) own compile as jit spans,
    # from a sitecustomize that runs the program's own one first
    # (src/instrument/pysite/).
    if env.get("HPROFILER_PYTHON_JIT") == "1":
        site = str(Path(__file__).resolve().parent.parent / "instrument" / "pysite")
        env["PYTHONPATH"] = os.pathsep.join(p for p in (site, env.get("PYTHONPATH", "")) if p)

    # ── Inject hooks via backend interface ───────────────────────────────
    from ..backends import ALL_BACKENDS
    preload_libs: list[str] = []
    _hook_backends: list[str] = []   # backends that expect LD_PRELOAD
    _PRELOAD_BACKENDS = {"cuda", "opencl", "rocm", "nccl", "mpi"}
    for bname in backends:
        backend_cls = ALL_BACKENDS.get(bname)
        if backend_cls is None:
            continue
        try:
            b = backend_cls()
            added = False
            for lib_path in b.preload_libs():
                if lib_path and Path(lib_path).exists():
                    preload_libs.append(lib_path)
                    added = True
            if bname in _PRELOAD_BACKENDS and not added:
                _hook_backends.append(bname)
            for k, v in b.env_vars().items():
                # OMP_TOOL_LIBRARIES uses ":" separator
                if k == "OMP_TOOL_LIBRARIES":
                    existing = env.get(k, "")
                    env[k] = ":".join(filter(None, [existing, v]))
                else:
                    env.setdefault(k, v)
        except Exception:
            pass

    # Warn when a hook-based backend was requested but the .so is missing
    import sys as _sys
    for bname in _hook_backends:
        hook_path = HOOKS_DIR / f"libhprofiler_{bname}.so"
        if not hook_path.exists():
            print(
                f"[hprofiler][warn] {bname} hook not found at {hook_path}\n"
                f"  Run 'hprofiler build' in the hprofiler directory to compile it.\n"
                f"  On clusters: load the ROCm/CUDA module first, then build.",
                file=_sys.stderr,
            )

    # Warn when the target binary has a statically-linked CUDA runtime —
    # LD_PRELOAD hooks cannot intercept static symbols, so 0 events will
    # be captured even though the program runs normally.
    if "cuda" in backends and command:
        _target = _static_cuda_binary(command)
        if _target:
            # LD_PRELOAD cannot intercept a static runtime; start CUPTI
            # when the hook loads instead, so its callbacks report the
            # host calls and its activity records the device work
            # (hooks/cuda_hook/cupti_trace.c).
            env.setdefault("HPROFILER_CUPTI_EAGER", "1")
            print(
                f"[hprofiler][warn] '{_target}' appears to be linked with "
                f"the STATIC CUDA runtime (libcudart_static.a).\n"
                f"  LD_PRELOAD interception is not possible; CUDA calls and device "
                f"work will be recorded through CUPTI instead (needs libcupti -- "
                f"see the 'gpuact' status in the trace). Without CUPTI no cuda "
                f"events are captured: rebuild with the shared runtime "
                f"(nvcc -cudart shared).",
                file=_sys.stderr,
            )

    if preload_libs:
        existing = env.get("LD_PRELOAD", "")
        env["LD_PRELOAD"] = ":".join(filter(None, [existing] + preload_libs))

    return env


class Runner:
    def __init__(
        self,
        command: list[str],
        backends: list[str],
        env_extra: dict[str, str] | None = None,
        perf_freq: int = 99,
        on_event: Callable[[AnyEvent], None] | None = None,
        collect_disasm: bool = False,
        store: "TraceStore | None" = None,
        collect_address: str | None = None,
        idle_exit_s: float | None = None,
    ) -> None:
        self.command = command
        self.backends = backends
        self.env_extra = env_extra or {}
        self.perf_freq = perf_freq
        self.on_event = on_event
        self.collect_disasm = collect_disasm
        # Where received events go: a TraceStore (e.g. DiskTraceStore for a
        # production capture -- events are appended in bounded batches and
        # never all held in memory). None = an in-memory store.
        self.store = store
        self._trace: Trace | None = None
        self._disasm_thread: threading.Thread | None = None
        # The profiled command's exit status after run() (see TraceMetadata).
        self.exit_status: int | None = None
        # Node-collector mode (`hprofiler collect`): no command is started;
        # hooks of processes started elsewhere (`hprofiler exec`) report to
        # this socket address ("@name": abstract, else a path). The capture
        # ends on SIGINT/SIGTERM, or idle_exit_s seconds after the last hook
        # connection closed (at least one having been seen).
        self.collect_address = collect_address
        self.idle_exit_s = idle_exit_s

    def run(self) -> Trace:
        import platform

        server_sock, sock_address, _sock_dir, sock_path = _open_collector_socket(self.collect_address)
        _abstract_socket = sock_path is None

        env = None if self.collect_address is not None else \
            hook_environment(self.backends, sock_address, self.command, self.env_extra)

        import sys as _sys
        metadata = TraceMetadata(
            command=self.command[0] if self.command else "(node collector)",
            args=self.command[1:] if self.command else [f"socket={sock_address}"],
            start_time_ns=0,  # filled in after Popen below
            pid=os.getpid(),
            backends_used=list(self.backends),
            hostname=platform.node(),
            cwd=os.getcwd(),
            capture_time_iso=datetime.datetime.now().isoformat(timespec="seconds"),
            clock=clock_domain(),
        )
        trace = Trace(metadata, store=self.store)
        self._trace = trace

        # A disk-backed capture records its metadata and "running" state up
        # front: if the collector itself dies, reopening the store says the
        # capture is incomplete instead of looking like a finished run.
        metadata.capture_health = {"state": "running"}
        if trace.store.kind == "disk":
            try:
                trace.save()
            except Exception:
                pass

        _recent_spans: dict[tuple[int, int], deque[SpanEvent]] = {}
        # Real executables of the connected processes, by pid in connection
        # order (see _peer_process; disasm_binaries() picks among them).
        _real_binary_paths: dict[int, list[str]] = {}

        def on_connect(client: socket.socket) -> None:
            peer_pid, real_exe = _peer_process(client)
            if real_exe:
                seen = _real_binary_paths.setdefault(peer_pid, [])
                if real_exe in seen:
                    seen.remove(real_exe)
                seen.append(real_exe)

        _KNOWN = ("span", "inst", "ctr")

        def handle_line(line: str) -> None:
            """One record from a hook (called by the receiver's parser thread,
            in per-connection order). Unknown kinds and records that fail to
            parse are counted, never silently skipped."""
            kind = line.split(":", 1)[0]
            if kind == "stk":
                parts = line.strip().split(":", 4)
                if len(parts) != 5:
                    receiver.note_malformed(line)
                    return
                try:
                    pid, tid, start_ns = int(parts[1]), int(parts[2]), int(parts[3])
                except ValueError:
                    receiver.note_malformed(line)
                    return
                # Frames may be plain "sym" or "sym|/lib.so|0xoffset"
                frames = [f for f in parts[4].split(";") if f]
                span = _find_recent_span(_recent_spans, pid, tid, start_ns)
                if span is not None:
                    span.stack_frames = frames
                    trace.update_span(span)
            elif kind == "gpuact":
                parsed = gpu_activity.parse_status_line(line)
                if parsed is None:
                    receiver.note_malformed(line)
                else:
                    gpu_activity.record_status(trace, *parsed)
            elif kind == "pcsa":
                # pcsa:<pid>:<ts_ns>:<func_name>:<pc_offset_hex>:<stall_reason_int>:<count>
                parts = line.strip().split(":", 6)
                try:
                    if len(parts) != 7:
                        raise ValueError
                    trace.add_pc_sample(parts[3], int(parts[4], 16), int(parts[5]), int(parts[6]))
                except ValueError:
                    receiver.note_malformed(line)
            elif kind in _KNOWN:
                ev = _parse_record(line)
                if ev is None:
                    receiver.note_malformed(line)
                    return
                if isinstance(ev, SpanEvent):
                    _remember_recent_span(_recent_spans, ev)
                trace.add(ev)
                if self.on_event:
                    self.on_event(ev)
            else:
                receiver.note_unknown(kind)

        from .receiver import Receiver
        # An abstract socket has no file permissions: accept only this
        # user's processes (a private directory protects a path socket).
        receiver = Receiver(server_sock, _sock_dir, trace, handle_line, on_connect=on_connect,
                            require_uid=os.getuid() if _abstract_socket else None)
        receiver.start()
        # The parser thread is CPU-bound; a short switch interval lets the
        # socket readers (which only recv and spool) run often enough to
        # keep the hooks' rings drained. Restored after the capture.
        _switch_interval = sys.getswitchinterval()
        sys.setswitchinterval(0.0005)

        # ── LIKWID command wrapping ───────────────────────────────────────────
        # likwid-perfctr must be the outer process; env vars (incl. HPROFILER_SOCKET)
        # are inherited by the child so all socket-based hooks still work.
        _likwid_backend = None
        run_command = list(self.command)
        if "likwid" in self.backends and self.collect_address is None:
            from ..backends.likwid import LIKWIDBackend
            _likwid_backend = LIKWIDBackend()
            run_command = _likwid_backend.wrap_command(self.command)

        # ── Start the profiled process directly ───────────────────────────────
        # perf record and perf stat attach via -p PID so we can run both
        # simultaneously without nesting and still get the env vars right.
        # Read HPROFILER_CALLGRAPH from env_extra without mutating caller's dict.
        callgraph = self.env_extra.get("HPROFILER_CALLGRAPH")
        # close_fds=False: the program inherits file descriptors exactly as
        # it would without hprofiler -- an MPI launcher's PMI socket
        # (PMI_FD: MPICH/Intel MPI Hydra, Slurm --mpi=pmi2) must survive
        # when one `hprofiler run` wraps each rank. Nothing of hprofiler's
        # leaks: every descriptor it opens itself (collector socket and
        # connections, spool and store files, perf temp files) is
        # non-inheritable (Python's default, O_CLOEXEC/SOCK_CLOEXEC), which
        # tests/test_runner_process.py checks.
        if self.collect_address is not None:
            print(f"[hprofiler] collecting on {sock_address}: start programs with "
                  f"`hprofiler exec --socket {sock_address} --backend {','.join(self.backends)} -- ...`",
                  file=_sys.stderr)
            proc = _CollectWait(receiver, self.idle_exit_s)
        else:
            proc = subprocess.Popen(run_command, env=env, close_fds=False)
        metadata.start_time_ns = time.monotonic_ns()
        pid = proc.pid

        # ── Attach perf record for CPU sampling ───────────────────────────────
        perf_record_proc: subprocess.Popen | None = None
        perf_data: str | None = None
        if "cpu" in self.backends and pid and shutil.which("perf"):
            fd, perf_data = tempfile.mkstemp(suffix=".perf.data", prefix="hprofiler_")
            os.close(fd)
            perf_cmd = [
                "perf", "record",
                f"-F{self.perf_freq}", "-e", "cycles:u",
                "--clockid=monotonic",   # align with hook CLOCK_MONOTONIC
                "-p", str(pid), "-o", perf_data,
            ]
            if callgraph:
                perf_cmd.append(f"--call-graph={callgraph}")
            try:
                perf_record_proc = subprocess.Popen(
                    perf_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            except Exception:
                try:
                    os.unlink(perf_data)
                except OSError:
                    pass
                perf_record_proc = None
                perf_data = None

        # ── Attach perf stat for CPU microarch counters ───────────────────────
        # For every backend that runs on or dispatches from the CPU: IPC and
        # cache misses also explain CPU-side launch and data-preparation costs
        # in GPU/MPI programs.
        perf_stat_proc: subprocess.Popen | None = None
        perf_stat_file: str | None = None
        _cpu_backends = {"cpu", "openmp", "likwid", "opencl", "cuda", "rocm", "nccl", "mpi"}
        if pid and shutil.which("perf") and any(b in self.backends for b in _cpu_backends):
            fd, perf_stat_file = tempfile.mkstemp(suffix=".perf_stat.txt", prefix="hprofiler_")
            os.close(fd)
            _MICROARCH_EVENTS = (
                "cycles,instructions,cache-references,cache-misses,"
                "branches,branch-misses,task-clock"
            )
            try:
                perf_stat_proc = subprocess.Popen(
                    ["perf", "stat", "-p", str(pid),
                     "-e", _MICROARCH_EVENTS, "-o", perf_stat_file],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            except Exception:
                try:
                    os.unlink(perf_stat_file)
                except OSError:
                    pass
                perf_stat_proc = None
                perf_stat_file = None

        # ── GPU utilization polling ───────────────────────────────────────────
        gpu_stop = threading.Event()
        gpu_thread = threading.Thread(
            target=_gpu_poll,
            args=(trace, self.backends, gpu_stop),
            daemon=True,
        )
        gpu_thread.start()

        # ── Wait for the profiled process ─────────────────────────────────────
        rss_before = resource.getrusage(resource.RUSAGE_CHILDREN)
        proc.wait()
        rss_after  = resource.getrusage(resource.RUSAGE_CHILDREN)
        metadata.end_time_ns = time.monotonic_ns()
        # Return code, or -N when killed by signal N (subprocess convention).
        metadata.exit_status = proc.returncode
        self.exit_status = proc.returncode

        # ── Tear down background collectors ───────────────────────────────────
        gpu_stop.set()

        if perf_record_proc is not None:
            try:
                perf_record_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                perf_record_proc.kill()

        if perf_stat_proc is not None:
            try:
                perf_stat_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                perf_stat_proc.kill()

        gpu_thread.join(timeout=2)

        # Every connection's data is ingested before post-processing: wait
        # for EOF on all of them (bounded, reported) and for the parser to
        # work through the backlog the program left behind.
        import sys as _sys_rx
        _sys_rx.setswitchinterval(_switch_interval)
        receiver.finish(progress=lambda msg: print(f"[hprofiler] {msg}", file=_sys_rx.stderr))
        server_sock.close()
        metadata.capture_health = {"state": "complete", **receiver.health()}
        if sock_path is not None:
            try:
                os.unlink(sock_path)
            except OSError:
                pass
        try:
            os.rmdir(_sock_dir)
        except OSError:
            pass

        # ── CUDA/ROCm host<->device correlation and de-duplication ────────────
        # Native activity records arrive in buffers, out of order and often
        # after the host spans they belong to, so this runs once on the
        # complete event set.
        try:
            gpu_activity.assemble(trace)
        except Exception as exc:   # never lose a run to post-processing
            import sys as _sys_ga
            print(f"[hprofiler][warn] GPU activity correlation failed: {exc!r} -- "
                  "device spans are left uncorrelated", file=_sys_ga.stderr)

        # Known holes in the MPI dependency graph (receives without a
        # sender, Test*-only and orphan completions), counted once here and
        # stored with the trace (one pass over the MPI spans only).
        if "mpi" in self.backends:
            try:
                from ..analysis.criticalpath import mpi_dependency_diagnostics
                metadata.dependency_diagnostics = mpi_dependency_diagnostics(trace)
            except Exception as exc:
                print(f"[hprofiler][warn] MPI dependency diagnostics failed: {exc!r}",
                      file=_sys_rx.stderr)

        # Incomplete capture / degraded device tracing: say so now (the
        # same list is stored with the trace and shown by `summary`/GUI).
        from .receiver import run_warnings
        for _w in run_warnings(metadata):
            print(f"[hprofiler][warn] {_w}", file=_sys_rx.stderr)

        # ── Parse perf record output ──────────────────────────────────────────
        if perf_data and Path(perf_data).exists():
            _parse_perf_script(perf_data, trace, self.perf_freq)
            # Do NOT delete perf_data yet — _collect_disasm needs it for
            # perf annotate instruction-level heat.  It is deleted there.

        # Annotate CPU/OpenMP spans with source file:line from DWARF
        try:
            from ..analysis.addr2line import annotate_trace as _ann_trace
            _ann_trace(trace)
        except Exception:
            pass

        # ── Emit microarch counter events ─────────────────────────────────────
        if perf_stat_file:
            _collect_microarch_counters(perf_stat_file, trace, metadata.end_time_ns)

        # ── Emit max-RSS counter event ────────────────────────────────────────
        _collect_rss(rss_before, rss_after, trace, metadata.end_time_ns)

        # ── LIKWID counter post-processing ────────────────────────────────────
        if _likwid_backend is not None:
            _likwid_backend.post_process(trace)

        # ── Total zero-event sanity check ───────────────────────────────────────
        if self.backends and not trace.span_count() and not trace.store.instant_count():
            import sys as _sys1
            msg = _total_zero_event_warning(self.backends, self.command, metadata.exit_status)
            print(msg, file=_sys1.stderr)

        # ── OpenMP zero-event check ───────────────────────────────────────────
        # Both capture paths (OMPT for LLVM libomp, GOMP_* interception for GNU
        # libgomp) are injected; if neither produced events, give
        # troubleshooting steps for both.
        if "openmp" in self.backends:
            import sys as _sys2
            if next(iter(trace.iter_spans(categories=("openmp",))), None) is None:
                print(
                    "[hprofiler][warn] openmp backend active but 0 OpenMP events "
                    "were captured.\n"
                    "  Check which OpenMP runtime the binary actually links:\n"
                    "    ldd <binary> | grep -E 'omp|gomp'\n"
                    "  libomp.so  -> OMPT path: confirm $OMP_TOOL_LIBRARIES pointed at "
                    "a real libhprofiler_ompt.so (run 'hprofiler build' if missing).\n"
                    "  libgomp.so -> GOMP_* interception path: confirm "
                    "build/lib/libhprofiler_gomp.so exists and was actually LD_PRELOADed "
                    "(check $LD_PRELOAD in the run's environment) — this path only covers "
                    "GOMP_parallel/loop(non-static)/barrier/critical/single, not every "
                    "construct (see DOCUMENTATION.md's openmp backend section), so a "
                    "program using ONLY unintercepted constructs (e.g. tasks, sections, "
                    "target offload) can legitimately still show 0 events.\n"
                    "  neither   -> statically-linked OpenMP runtime, or a runtime other "
                    "than libomp/libgomp — LD_PRELOAD interception cannot intercept "
                    "compile-time-resolved symbols.",
                    file=_sys2.stderr,
                )

        # ── Query device theoretical peaks ────────────────────────────────────
        try:
            from ..analysis.device import query_devices
            devs = query_devices(self.backends)
            if devs:
                trace.set_devices(devs)
        except Exception:
            pass

        if self.collect_disasm:
            self._disasm_thread = threading.Thread(
                target=_collect_disasm,
                args=(trace, self.command, self.backends, pid, perf_data),
                kwargs={"real_binary_paths": _real_binary_paths},
                daemon=True,
            )
            self._disasm_thread.start()

        # ── Resolve stack frame addresses to file:line via addr2line ─────────
        # If HPROFILER_CALLSTACK was set and the hooks captured lib+offset info
        # in frames ("sym|/lib.so|0xoffset"), run addr2line to annotate them
        # with source locations.  This is best-effort and never blocks the run.
        if trace._has_stacks:
            try:
                from ..analysis.cct import annotate_stack_frames
                annotate_stack_frames(trace)
            except Exception:
                pass

        return trace


# ── GPU utilization polling ───────────────────────────────────────────────────

def _find_tool(*names: str) -> Optional[str]:
    """Find a tool by name in PATH, then in common install directories."""
    for name in names:
        v = shutil.which(name)
        if v:
            return v
    # Fall back to well-known install paths (common on HPC clusters where
    # ROCm/CUDA bin dirs may not be in the system PATH).
    for name in names:
        for prefix in ("/opt/rocm/bin", "/usr/local/rocm/bin",
                       "/usr/local/cuda/bin", "/usr/bin"):
            p = Path(prefix) / name
            if p.exists():
                return str(p)
    return None


def _gpu_poll(trace: Trace, backends: list[str], stop: threading.Event) -> None:
    """Background thread: poll GPU utilisation every second, emit CounterEvents."""
    _nvidia_backends = {"cuda", "opencl", "nccl"}
    _amd_backends    = {"rocm", "opencl"}
    if any(b in backends for b in _nvidia_backends):
        nvidia_smi = _find_tool("nvidia-smi")
        if nvidia_smi:
            _poll_nvidia_smi(trace, stop, nvidia_smi)
            return
    if any(b in backends for b in _amd_backends):
        rocm_smi = _find_tool("rocm-smi")
        if rocm_smi:
            _poll_rocm_smi(trace, stop, rocm_smi)


def _poll_nvidia_smi(trace: Trace, stop: threading.Event,
                     tool: str = "nvidia-smi") -> None:
    while not stop.is_set():
        try:
            r = subprocess.run(
                [tool,
                 "--query-gpu=index,utilization.gpu,utilization.memory,memory.used",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3,
            )
            ts = time.monotonic_ns()
            for line in r.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 4:
                    continue
                try:
                    idx      = int(parts[0])
                    util_gpu = float(parts[1])
                    util_mem = float(parts[2])
                    mem_mib  = float(parts[3])
                except ValueError:
                    continue
                suf = f"[gpu{idx}]"
                trace.add(CounterEvent(f"gpu_utilization_pct{suf}",
                                        Category.MEMORY, ts, util_gpu, "%"))
                trace.add(CounterEvent(f"gpu_mem_util_pct{suf}",
                                        Category.MEMORY, ts, util_mem, "%"))
                trace.add(CounterEvent(f"gpu_mem_used_bytes{suf}",
                                        Category.MEMORY, ts, mem_mib * 1024**2, "bytes"))
        except Exception:
            pass
        stop.wait(timeout=1.0)


def _poll_rocm_smi(trace: Trace, stop: threading.Event,
                   tool: str = "rocm-smi") -> None:
    while not stop.is_set():
        try:
            r_use = subprocess.run(
                [tool, "--showuse", "--csv"],
                capture_output=True, text=True, timeout=3,
            )
            r_mem = subprocess.run(
                [tool, "--showmeminfo", "vram", "--csv"],
                capture_output=True, text=True, timeout=3,
            )
            ts = time.monotonic_ns()

            # Parse GPU utilization — find "use"/"utiliz" column from header.
            # Strip comment lines (#) and banner separator lines (=, -) that
            # newer rocm-smi versions emit around the CSV data.
            use_lines = [l.strip() for l in r_use.stdout.strip().splitlines()
                         if l.strip()
                         and not l.startswith(("#", "=", "-"))
                         and "," in l]
            if len(use_lines) >= 2:
                hdr = [h.lower() for h in use_lines[0].split(",")]
                use_col = next(
                    (i for i, h in enumerate(hdr)
                     if ("use" in h or "utiliz" in h) and "mem" not in h and i > 0),
                    1,
                )
                for line in use_lines[1:]:
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) <= use_col:
                        continue
                    try:
                        idx      = int(parts[0].replace("card", ""))
                        util_pct = float(parts[use_col].rstrip("%"))
                        trace.add(CounterEvent(f"gpu_utilization_pct[gpu{idx}]",
                                               Category.MEMORY, ts, util_pct, "%"))
                    except (ValueError, IndexError):
                        pass

            # Parse VRAM — find "used" column from header and detect its unit
            mem_lines = [l.strip() for l in r_mem.stdout.strip().splitlines()
                         if l.strip()
                         and not l.startswith(("#", "=", "-"))
                         and "," in l]
            if len(mem_lines) >= 2:
                hdr = [h.lower() for h in mem_lines[0].split(",")]
                used_col = next(
                    (i for i, h in enumerate(hdr) if "used" in h and i > 0),
                    None,
                )
                if used_col is not None:
                    h = hdr[used_col]
                    if "(gb)" in h:
                        mult = 1024 ** 3
                    elif "(mb)" in h:
                        mult = 1024 ** 2
                    elif "(kb)" in h:
                        mult = 1024
                    else:
                        mult = 1  # assume bytes (rocm-smi typically uses bytes)
                    for line in mem_lines[1:]:
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) <= used_col:
                            continue
                        try:
                            idx        = int(parts[0].replace("card", ""))
                            used_bytes = float(parts[used_col]) * mult
                            trace.add(CounterEvent(f"gpu_mem_used_bytes[gpu{idx}]",
                                                   Category.MEMORY, ts, used_bytes, "bytes"))
                        except (ValueError, IndexError):
                            pass
        except Exception:
            pass
        stop.wait(timeout=1.0)


# ── CPU microarchitecture counters (perf stat) ────────────────────────────────

def _parse_perf_stat_microarch(text: str) -> dict[str, float]:
    """Parse `perf stat -o FILE` output → {metric: value}."""
    ev: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or not line[0].isdigit():
            continue
        parts = re.split(r"\s{2,}", line, maxsplit=2)
        if len(parts) < 2:
            continue
        raw = parts[0].replace(",", "")  # strip thousands separators only
        try:
            val = float(raw)
        except ValueError:
            continue
        # Strip perf event qualifiers (:u, :k, :p, :H) and PMU prefix (cpu/)
        key = re.sub(r"^[\w-]+/", "", parts[1].rstrip("/")).lower().strip()
        key = re.sub(r":[ukpHG]+$", "", key)
        if key:
            ev[key] = ev.get(key, 0.0) + val

    def g(k: str) -> float:
        return ev.get(k, 0.0)

    result: dict[str, float] = {}
    cyc   = g("cycles")
    ins   = g("instructions")
    cref  = g("cache-references")
    cmiss = g("cache-misses")
    br    = g("branches")
    bmiss = g("branch-misses")
    if cyc > 0 and ins > 0:
        result["ipc"] = ins / cyc
    if cref > 0:
        result["cache_miss_pct"] = 100.0 * cmiss / cref
    if br > 0:
        result["branch_miss_pct"] = 100.0 * bmiss / br
    return result


_LAUNCHER_NAMES = {"srun", "mpirun", "mpiexec", "aprun", "jsrun", "ibrun"}


def _total_zero_event_warning(backends: list[str], command: list[str],
                              exit_status: int | None = None) -> str:
    """Warning for a run that captured no events from any backend. When the
    program itself failed (non-zero exit status) that is the likely reason
    and the warning says so first. Otherwise the hooks never reached the
    collector, typically because a job launcher (srun, mpirun, aprun, ...)
    did not pass HPROFILER_SOCKET/LD_PRELOAD to the processes it started.
    With SLURM this depends on site defaults and --export and can be
    intermittent, so launcher-specific advice is given when the command is a
    known launcher."""
    if exit_status:
        how = (f"was killed by signal {-exit_status}" if exit_status < 0
               else f"exited with status {exit_status}")
        return (
            f"[hprofiler][warn] run captured ZERO events of any kind across all active "
            f"backends ({', '.join(backends)}), and the profiled command {how}: it most "
            f"likely failed before making any profiled call (check its own output). "
            f"If it ran normally without hprofiler, the hooks may not have reached the "
            f"collector instead (see DOCUMENTATION.md, \"Launchers and MPI jobs\")."
        )
    launcher = command[0] if command else ""
    launcher_hint = ""
    if launcher in _LAUNCHER_NAMES:
        launcher_hint = (
            f"\n  '{launcher}' is a job launcher -- it must propagate "
            f"HPROFILER_SOCKET and LD_PRELOAD to the process(es) it actually "
            f"spawns, not just see them itself. If this is intermittent (same "
            f"command works on a later run with no changes), check your site's "
            f"{launcher} environment-export defaults, e.g. try "
            f"`{launcher} --export=ALL ...` (SLURM) or the equivalent for your "
            f"launcher; this is a launcher/site config issue, not something "
            f"hprofiler's own retry logic can work around from inside the "
            f"already-spawned process.\n"
        )
    return (
        f"[hprofiler][warn] run completed but captured ZERO events of any "
        f"kind across all active backends ({', '.join(backends)}) -- "
        f"this points to the hooks never connecting to the collector socket "
        f"for this entire run, not any one backend's constructs simply not "
        f"being used." + launcher_hint
    )


def _collect_microarch_counters(stat_file: str, trace: Trace, ts_ns: int) -> None:
    try:
        text = Path(stat_file).read_text(errors="replace")
        for name, value in _parse_perf_stat_microarch(text).items():
            trace.add(CounterEvent(name, Category.CPU, ts_ns, value, ""))
    except Exception:
        pass
    finally:
        try:
            Path(stat_file).unlink()
        except OSError:
            pass


# ── Process max-RSS ───────────────────────────────────────────────────────────

def _collect_rss(
    before: "resource.struct_rusage",
    after: "resource.struct_rusage",
    trace: Trace,
    ts_ns: int,
) -> None:
    """Emit max RSS of the profiled process as a CounterEvent."""
    try:
        # ru_maxrss is in KB on Linux, bytes on macOS
        import sys
        rss = after.ru_maxrss
        if rss <= 0:
            return
        if sys.platform != "darwin":
            rss *= 1024   # KB → bytes on Linux
        trace.add(CounterEvent(
            "process_max_rss_bytes", Category.CPU, ts_ns, float(rss), "bytes",
        ))
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────

def _collect_disasm(
    trace: Trace,
    command: list[str],
    backends: list[str],
    profiled_pid: int = 0,
    perf_data: str | None = None,
    real_binary_paths: "dict[int, list[str]] | set[str] | None" = None,
) -> None:
    """Post-run: extract disassembly for the profiled kernels and attach it.

    ACPP SSCP .jit.so paths come from the OpenCL hook's jit_load spans.
    real_binary_paths: the connected processes' executables by pid
    (_peer_process), read by the whole-binary CUDA/ROCm AoT paths when
    command[0] is a launcher (disasm_binaries() chooses)."""
    # Resolve the binary path: it may be relative (e.g. './main').
    # Use the saved cwd from the trace metadata to make it absolute.
    cwd = trace.metadata.cwd or ""
    if command and cwd:
        import os as _os
        binary_rel = command[0]
        command = [str(_os.path.join(cwd, binary_rel)
                       if not _os.path.isabs(binary_rel) else binary_rel)] + command[1:]

    try:
        from ..disasm.extractor import attribute_launches, collect_disasm
    except ImportError:
        return

    # Build jit_spans, omp_syms, and cpu_names from trace events
    jit_spans: list[dict] = []
    # omp_syms:  {span_name: ("sym", mangled_name)}
    #         or {span_name: ("lib", (lib_path, static_offset))}
    omp_syms: dict[str, tuple] = {}
    # cpu_names: function names from perf-sampled CPU spans (no sym/lib tags)
    cpu_names: set[str] = set()
    # Module identity (hooks/common/hp_module.h): saved blob path -> mod=,
    # and how often each kernel name was launched from each module ("" for
    # launches without a module id).
    module_ids: dict[str, str] = {}
    launches: Counter = Counter()
    gpu_pids: set[int] = set()            # processes that launched or ran GPU work

    for span in trace.iter_spans():
        mod = span.tags.get("mod", "")
        if span.category.value in ("cuda", "rocm") or span.tags.get("rt") in ("cuda", "rocm"):
            gpu_pids.add(span.pid)
        if span.category == Category.JIT and mod and span.tags.get("path"):
            module_ids[span.tags["path"]] = mod
        elif span.category.value in ("cuda", "rocm", "opencl", "sycl") and (
                span.tags.get("side") == "gpu" or span.tags.get("type") == "jit_kernel"):
            launches[(span.name, mod)] += 1
        # OpenCL/ACPP SSCP JIT .so files
        if span.category == Category.JIT and span.tags.get("type") == "jit_load":
            so_path = span.tags.get("path", "")
            if so_path and (".jit.so" in so_path or "hprofiler_jit_" in so_path
                            or "hprofiler_ocl_" in so_path):
                jit_spans.append({
                    "name":    span.name,
                    "so_path": so_path,
                    "mangled": span.tags.get("mangled", ""),
                    "mod":     mod,
                })
        # ROCm JIT: hipModuleLoadData emits type=jit_compile with path= of saved ELF
        if span.category == Category.JIT and span.tags.get("type") == "jit_compile":
            jit_path = span.tags.get("path", "")
            if jit_path and "hprofiler_rocm_" in jit_path:
                jit_spans.append({
                    "name":    span.name,
                    "so_path": "",
                    "path":    jit_path,
                    "mod":     mod,
                })
        # OpenMP/MPI/CPU spans: the first resolved call site per span name --
        # sym=<mangled> (+ symfile=, the ELF dladdr found it in; the command
        # may be a launcher) or lib=<path>,offset=0x<off>. Without symfile=
        # (older hooks) collect_disasm() falls back to command[0].
        if span.category.value in ("openmp", "sync", "cpu", "mpi") and span.name not in omp_syms:
            sym = span.tags.get("sym", "")
            if sym:
                omp_syms[span.name] = ("sym", (sym, span.tags.get("symfile") or None))
                continue
            lib = span.tags.get("lib", "")
            off_s = span.tags.get("offset", "")
            if lib and off_s:
                try:
                    omp_syms[span.name] = ("lib", (lib, int(off_s, 16)))
                except ValueError:
                    pass
                continue
            # perf-sampled CPU span: no sym/lib tags — collect the name directly
            if span.category.value == "cpu" and span.name and span.name != "[cpu]":
                cpu_names.add(span.name)

    # Derive SM version string from device info for ptxas PTX→SASS compilation
    sm_version = ""
    for dev in trace.devices:
        cc = getattr(dev, "compute_cap", "")
        if cc and "." in cc:
            major, minor = cc.split(".", 1)
            sm_version = f"sm_{major}{minor}"
            break

    # Connected executables beat a launcher command[0]: those of the
    # processes that did GPU work (SPMD: every rank runs the same binary),
    # else all of them, in a fixed order.
    binaries = disasm_binaries(real_binary_paths, gpu_pids)

    launched_by_module: dict[str, set[str]] = {}
    for (name, mod) in launches:
        if mod:
            launched_by_module.setdefault(mod, set()).add(name)

    try:
        disasm_map = collect_disasm(command, backends, jit_spans, omp_syms, profiled_pid, cpu_names,
                                    sm_version=sm_version, real_binaries=binaries,
                                    module_ids=module_ids, launched_by_module=launched_by_module)
        attribute_launches(disasm_map.values(), launches)

        # Launches recorded as "<jit-kernel>" (the hook could resolve neither
        # a name nor a module) get no disassembly: attaching some other
        # kernel's code -- this used to alias the first GPU kernel found --
        # would show code that may never have run.

        # ── Instruction-level heat annotation ─────────────────────────────
        # CPU kernels: perf annotate instruction percentages
        _GPU_ARCHS = {"sass", "ptx", "amdgcn", "gcn", "rocm", "hip"}
        if perf_data and Path(perf_data).exists():
            try:
                from ..disasm.extractor import annotate_with_perf
                for kd in disasm_map.values():
                    if kd.arch.lower() not in _GPU_ARCHS:
                        annotate_with_perf(kd, perf_data)
            except Exception:
                pass
            finally:
                try:
                    Path(perf_data).unlink()
                except OSError:
                    pass

        # GPU kernels: CUPTI PC sampling (populated by pcsa: socket records)
        # Only SASS disasm has addresses that match CUPTI PC offsets.
        # ptxas_derived kernels also have SASS (compiled offline from PTX) so
        # they are included — CUPTI name lookup tries both short name and
        # the original mangled symbol stored in kd.mangled_name.
        _has_pc_samples = bool(trace._pc_samples)
        _ptx_only_warned = False
        try:
            from ..disasm.extractor import annotate_with_cupti
            import sys as _sys
            for kd in disasm_map.values():
                if kd.arch.lower() != "sass":
                    # Only warn for PTX kernels — if ptxas was available it would
                    # have converted them already; reaching here means ptxas failed
                    # or was not installed.
                    if _has_pc_samples and kd.arch.lower() == "ptx" and not _ptx_only_warned:
                        _ptx_only_warned = True
                        print(
                            "[hprofiler][warn] --gpu-pc-sampling: CUPTI samples SASS offsets but "
                            "this kernel is PTX-only (ptxas not found or compilation failed).\n"
                            "  Install ptxas (CUDA toolkit) so hprofiler can compile PTX → SASS "
                            "offline and match CUPTI samples to instructions.",
                            file=_sys.stderr,
                        )
                    continue
                # Try short name first, then the original mangled symbol (for ptxas-derived SASS)
                samples = trace._pc_samples.get(kd.name, [])
                if not samples and kd.mangled_name:
                    samples = trace._pc_samples.get(kd.mangled_name, [])
                if samples:
                    annotate_with_cupti(kd, samples)
        except Exception:
            pass

        try:
            from ..disasm.source_ann import annotate_disasm as _ann_src
            for kd in disasm_map.values():
                _ann_src(kd)
        except Exception:
            pass

        for kd in disasm_map.values():
            trace.add_disasm(kd)
    except Exception:
        pass   # disasm is best-effort; never crash the profiler run

    # Clean up any remaining /tmp/hprofiler_* scratch files for this PID.
    # The extractor removes files as it processes them, but files left over
    # from a crashed or interrupted run accumulate otherwise.
    if profiled_pid:
        import glob as _gl
        for _p in _gl.glob(f"/tmp/hprofiler_*_{profiled_pid}_*"):
            try:
                Path(_p).unlink()
            except OSError:
                pass


def _nm_symbols(so_path: str) -> dict[int, str]:
    """Return {addr: name} map from a shared object's symbol table."""
    try:
        r = subprocess.run(["nm", "-D", "--defined-only", so_path],
                           capture_output=True, text=True, timeout=10)
        syms: dict[int, str] = {}
        for line in r.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 3:
                try:
                    syms[int(parts[0], 16)] = parts[2]
                except ValueError:
                    pass
        return syms
    except Exception:
        return {}


_so_sym_cache: dict[str, dict[int, str]] = {}


def _resolve_jit_sym(addr: int, so_path: str) -> str:
    """Resolve an address inside a JIT .so to its nearest symbol."""
    if so_path not in _so_sym_cache:
        _so_sym_cache[so_path] = _nm_symbols(so_path)
    syms = _so_sym_cache[so_path]
    if not syms:
        return "[unknown]"
    # Find the largest symbol address <= addr
    candidates = [(a, n) for a, n in syms.items() if a <= addr]
    if not candidates:
        return "[unknown]"
    _, name = max(candidates, key=lambda kv: kv[0])
    # Demangle C++ names
    try:
        dm = subprocess.run(["c++filt", name], capture_output=True, text=True, timeout=2)
        if dm.returncode == 0:
            name = dm.stdout.strip()
    except Exception:
        pass
    return name


def _parse_perf_script(perf_data: str, trace: Trace, freq: int = 99) -> None:
    """Parse `perf script` output into SpanEvents.

    Flat format: one line per sample, a zero-duration span each. Stack format
    (--perf-callgraph): a header line followed by indented frames, samples
    separated by blank lines; each sample becomes ONE span named after its
    leaf frame, with the ancestors as stack_frames (innermost first) and a
    nominal duration of one sampling interval (1e9/freq ns) -- the weight
    perf report itself assumes.
    """
    try:
        result = subprocess.run(
            ["perf", "script", "-i", perf_data],
            capture_output=True, text=True, timeout=120,
        )
    except Exception:
        return

    lines = result.stdout.splitlines()

    if "data size field is 0" in result.stderr:
        import sys
        print("[hprofiler][cpu] perf data empty — try increasing --perf-freq "
              "or use --perf-callgraph=fp", file=sys.stderr)
        return

    import re
    # Sample header: "  comm  pid/tid  timestamp:  [period  event:  addr  sym  (dso)]"
    # The timestamp field always ends with ":"  and is a float like "12345.678901:"
    _HDR = re.compile(
        r'^\s*(\S+)\s+(\d+)(?:/(\d+))?\s+(\d+\.\d+):\s*'    # comm pid[/tid] ts:
        r'(?:\d+\s+\S+:\s+)?'                                  # optional period event:
        r'(?:[\da-f]+\s+)?'                                    # optional addr
        r'(\S.*?)?\s*(?:\(([^)]+)\))?$'                        # sym? (dso)?
    )
    # Stack frame: starts with spaces then a hex address
    _FRAME = re.compile(r'^\s+[\da-f]{4,}\s+(\S.*?)(?:\s+\(([^)]+)\))?$')

    def _sym_from_match(sym_raw: str | None, dso_raw: str | None,
                        addr_raw: str | None = None) -> str:
        if not sym_raw or sym_raw == "[unknown]":
            if dso_raw and dso_raw.endswith(".jit.so") and addr_raw:
                try:
                    addr = int(addr_raw, 16)
                    return _resolve_jit_sym(addr, dso_raw)
                except ValueError:
                    pass
            return ""
        return sym_raw.split("+")[0]

    # Unified parse: in stack format there are blank-line separators;
    # in flat format every line is a self-contained sample.
    # Detect: if the output has blank lines, it's stack mode.
    has_stacks = any(not ln.strip() for ln in lines)

    cur_pid = cur_tid = 0
    cur_ts: int = 0
    cur_top_sym = ""
    cur_stack: list[str] = []
    # Nominal per-sample weight (stack format only; flat samples keep 0).
    sample_weight_ns = max(1, round(1_000_000_000 / freq)) if freq > 0 else 1

    def _flush():
        if not cur_ts:
            return
        rel_ts = cur_ts
        if cur_stack:
            # cur_stack is innermost-first: [0] is the leaf, [1:] the
            # ancestors, already in stack_frames order.
            leaf = cur_stack[0]
            trace.add(SpanEvent(
                name=leaf, category=Category.CPU,
                start_ns=rel_ts, duration_ns=sample_weight_ns,
                pid=cur_pid, tid=cur_tid,
                stack_frames=cur_stack[1:],
            ))
        else:
            # Call-graph was requested but unwinding failed for this one
            # sample -- still a real sample, gets the same nominal
            # weight as every other one in this run so it doesn't bias
            # aggregate percentages downward.
            name = cur_top_sym or "[cpu]"
            if name:
                trace.add(SpanEvent(
                    name=name, category=Category.CPU,
                    start_ns=rel_ts, duration_ns=sample_weight_ns,
                    pid=cur_pid, tid=cur_tid,
                ))

    for raw_line in lines:
        if not raw_line.strip():
            if has_stacks:
                _flush()
                cur_stack = []
                cur_top_sym = ""
                cur_ts = 0
            continue

        # Try sample header
        m = _HDR.match(raw_line)
        if m:
            if has_stacks and cur_ts:
                _flush()
                cur_stack = []
                cur_top_sym = ""
                cur_ts = 0  # reset so back-to-back headers don't double-flush
            try:
                cur_pid = int(m.group(2))
                cur_tid = int(m.group(3)) if m.group(3) else cur_pid
                cur_ts  = int(float(m.group(4)) * 1_000_000_000)
            except ValueError:
                continue
            sym = _sym_from_match(m.group(5), m.group(6))
            cur_top_sym = sym
            if not has_stacks:
                # Flat format: emit immediately
                name = sym or "[cpu]"
                trace.add(SpanEvent(
                    name=name, category=Category.CPU,
                    start_ns=cur_ts,
                    duration_ns=0, pid=cur_pid, tid=cur_tid,
                ))
            continue

        # Try stack frame
        if has_stacks:
            m = _FRAME.match(raw_line)
            if m:
                sym = _sym_from_match(m.group(1), m.group(2))
                if sym:
                    cur_stack.append(sym)

    if has_stacks:
        _flush()
