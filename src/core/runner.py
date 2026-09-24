"""
Process runner: launches the target program with profiling hooks injected.

Environment setup per backend:
  - CUDA hook:    LD_PRELOAD += libhprofiler_cuda.so
  - OpenCL hook:  LD_PRELOAD += libhprofiler_opencl.so
  - OMPT tool:    OMP_TOOL_LIBRARIES = libhprofiler_ompt.so
  - ROCm:         LD_PRELOAD += libhprofiler_rocm.so  (or roctracer)
  - perf:         perf record run alongside the process

All hooks communicate back via a Unix socket written to HPROFILER_SOCKET.
The runner binds the socket, forks/execs the target, and reads events
until the process exits.
"""

from __future__ import annotations
import os
import re
import resource
from collections import deque
import shutil
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

HOOKS_DIR = Path(__file__).parent.parent.parent / "build" / "lib"

# Wire protocol from C hooks: newline-delimited ASCII records
# span:<category>:<pid>:<tid>:<start_ns>:<dur_ns>:<name>[:<tag=val>...]
# inst:<category>:<pid>:<tid>:<ts_ns>:<name>
# ctr:<category>:<pid>:<ts_ns>:<name>:<value>:<unit>

# Tags are always "key=val[,key=val...]" with no colons in them, so once a
# candidate tail (everything after the *last* colon) matches this, it's the
# tags segment and everything before it is the name — even if the name
# itself contains colons (e.g. demangled C++ "Namespace::kernel", or an
# NVTX label with a ':' in it). If it doesn't match, there's no tags segment
# and the whole remainder is the name.
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
            # inst has one fewer leading numeric field than span (no dur_ns), so
            # "name[:tags]" starts one position earlier (parts[5], not parts[6]).
            # The top-level split(":", 6) was sized for span's shape, so for inst
            # it may have already split *inside* a colon-containing name or tags
            # blob before we get here -- rejoin everything from parts[5] onward
            # so _split_name_tags sees the same undivided tail span: gets.
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


# Maps (pid, tid) -> a small ring of recent SpanEvents from that thread, used
# to attach a stk: record to the span: record it annotates.
#
# stk: records are sent immediately after their span: record on the same
# connection, but each LD_PRELOAD hook (cuda/rocm/opencl/ompt/nccl/mpi) opens
# its own socket connection with its own handler thread, so two different
# hooks emitting from the same OS tid in close succession can interleave
# across connections. A single last-write-wins slot could then have already
# been overwritten by the other hook's span by the time this stk: record
# arrives, even though the correct span is still very recent. Keep a short
# ring instead of one slot and match by start_ns anywhere in it.
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


def _peer_real_exe(client: "socket.socket") -> str:
    """Resolve the REAL executable path of whatever process is on the
    other end of this Unix-domain socket connection, via SO_PEERCRED --
    a kernel-verified credential (the accepting side cannot be lied to
    about it), not anything a hook has to self-report.

    Why this exists: CUDA/ROCm AoT disassembly (disasm_cuda_sass/
    disasm_rocm_binary) disassembles the WHOLE BINARY, keyed off
    `command[0]` -- unlike OpenMP/MPI's disasm, which resolves a specific
    call site via dladdr inside the profiled process (sym=/symfile=
    tags) and so already gets the right binary regardless of how the
    process was launched. When `command[0]` is a launcher
    (`hprofiler run -- srun -n 4 gmx_mpi ...`), it's `srun`, never the
    real GPU binary, and there's no per-span tag to fall back on for
    this whole-binary case. Every hook connects to HPROFILER_SOCKET from
    INSIDE the real profiled process (that's how LD_PRELOAD hooking
    works), so the peer credentials of that exact connection give the
    real PID for free -- /proc/<pid>/exe then resolves to the real
    binary, launcher or no launcher.

    Returns "" (not an exception) on any failure -- a process that's
    already exited by the time this runs, permission issues, or a non-
    Linux platform (SO_PEERCRED is Linux-specific) are all just "we
    don't know", the same as command[0] not existing today.
    """
    import struct
    try:
        creds = client.getsockopt(
            socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
        peer_pid, _uid, _gid = struct.unpack("3i", creds)
        return os.readlink(f"/proc/{peer_pid}/exe")
    except Exception:
        return ""


class Runner:
    def __init__(
        self,
        command: list[str],
        backends: list[str],
        env_extra: dict[str, str] | None = None,
        perf_freq: int = 99,
        on_event: Callable[[AnyEvent], None] | None = None,
        collect_disasm: bool = False,
    ) -> None:
        self.command = command
        self.backends = backends
        self.env_extra = env_extra or {}
        self.perf_freq = perf_freq
        self.on_event = on_event
        self.collect_disasm = collect_disasm
        self._trace: Trace | None = None
        self._disasm_thread: threading.Thread | None = None

    def run(self) -> Trace:
        import socket as sock_mod
        import platform

        _sock_dir = tempfile.mkdtemp(prefix="hprofiler_")
        sock_path = os.path.join(_sock_dir, "s.sock")

        server_sock = sock_mod.socket(sock_mod.AF_UNIX, sock_mod.SOCK_STREAM)
        server_sock.bind(sock_path)
        server_sock.listen(16)
        server_sock.settimeout(0.5)

        env = dict(os.environ)
        env["HPROFILER_SOCKET"] = sock_path
        env.update(self.env_extra)

        # ── Inject hooks via backend interface ───────────────────────────────
        from ..backends import ALL_BACKENDS
        preload_libs: list[str] = []
        _hook_backends: list[str] = []   # backends that expect LD_PRELOAD
        _PRELOAD_BACKENDS = {"cuda", "opencl", "rocm", "nccl", "mpi"}
        for bname in self.backends:
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
        if "cuda" in self.backends and self.command:
            _target = shutil.which(self.command[0]) or self.command[0]
            try:
                r = subprocess.run(
                    ["ldd", _target], capture_output=True, text=True, timeout=5
                )
                ldd_out = r.stdout + r.stderr
                has_cudart = "libcudart" in ldd_out or "libcuda" in ldd_out
                if not has_cudart and Path(_target).exists():
                    # Double-check: are CUDA symbols statically baked in?
                    nm_r = subprocess.run(
                        ["nm", _target], capture_output=True, text=True, timeout=10
                    )
                    if "T cudaLaunchKernel" in nm_r.stdout or "T cuLaunchKernel" in nm_r.stdout:
                        print(
                            f"[hprofiler][warn] '{self.command[0]}' appears to be linked with "
                            f"the STATIC CUDA runtime (libcudart_static.a).\n"
                            f"  Direct symbol interception via LD_PRELOAD is not possible, so "
                            f"CUDA Runtime/Driver API calls will NOT be captured for this run "
                            f"(0 cuda events expected). Rebuild with the shared runtime "
                            f"(nvcc -cudart shared) to enable CUDA profiling.",
                            file=_sys.stderr,
                        )
            except Exception:
                pass

        if preload_libs:
            existing = env.get("LD_PRELOAD", "")
            env["LD_PRELOAD"] = ":".join(filter(None, [existing] + preload_libs))

        metadata = TraceMetadata(
            command=self.command[0],
            args=self.command[1:],
            start_time_ns=0,  # filled in after Popen below
            pid=os.getpid(),
            backends_used=list(self.backends),
            hostname=platform.node(),
            cwd=os.getcwd(),
        )
        trace = Trace(metadata)
        self._trace = trace

        events_lock = threading.Lock()
        client_threads: list[threading.Thread] = []
        _recent_spans: dict[tuple[int, int], deque[SpanEvent]] = {}
        # Real exe path(s) of whatever process(es) actually connected to
        # the socket -- see _peer_real_exe's docstring. A set, not a
        # single value: an MPI job launches one hook connection per rank,
        # typically all the same binary (SPMD), but nothing here assumes
        # that -- collect_disasm() just needs ANY one real binary path,
        # strictly better than the launcher command[0] it'd use otherwise.
        _real_binary_paths: set[str] = set()

        def handle_client(client: sock_mod.socket) -> None:
            buf = ""
            real_exe = _peer_real_exe(client)
            if real_exe:
                with events_lock:
                    _real_binary_paths.add(real_exe)
            try:
                while True:
                    data = client.recv(4096)
                    if not data:
                        break
                    buf += data.decode("utf-8", errors="replace")
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        if line.startswith("stk:"):
                            try:
                                parts = line.strip().split(":", 4)
                                if len(parts) == 5:
                                    pid, tid, start_ns = int(parts[1]), int(parts[2]), int(parts[3])
                                    # Frames may be plain "sym" or "sym|/lib.so|0xoffset"
                                    frames = [f for f in parts[4].split(";") if f]
                                    with events_lock:
                                        span = _find_recent_span(_recent_spans, pid, tid, start_ns)
                                        if span is not None:
                                            span.stack_frames = frames
                                            trace._has_stacks = True
                            except Exception:
                                pass
                        elif line.startswith("pcsa:"):
                            # pcsa:<pid>:<ts_ns>:<func_name>:<pc_offset_hex>:<stall_reason_int>:<count>
                            try:
                                parts = line.strip().split(":", 6)
                                if len(parts) == 7:
                                    func_name = parts[3]
                                    pc_offset = int(parts[4], 16)
                                    stall_reason = int(parts[5])
                                    count = int(parts[6])
                                    with events_lock:
                                        trace.add_pc_sample(func_name, pc_offset, stall_reason, count)
                            except Exception:
                                pass
                        else:
                            ev = _parse_record(line)
                            if ev is not None:
                                with events_lock:
                                    if isinstance(ev, SpanEvent):
                                        _remember_recent_span(_recent_spans, ev)
                                    trace.add(ev)
                                if self.on_event:
                                    self.on_event(ev)
            except Exception:
                pass
            finally:
                client.close()

        def accept_loop(stop_event: threading.Event) -> None:
            while not stop_event.is_set():
                try:
                    client, _ = server_sock.accept()
                    t = threading.Thread(target=handle_client, args=(client,), daemon=True)
                    t.start()
                    client_threads.append(t)
                    # Prune finished threads to prevent unbounded growth for large MPI jobs
                    if len(client_threads) > 50:
                        client_threads[:] = [th for th in client_threads if th.is_alive()]
                except sock_mod.timeout:
                    continue
                except Exception:
                    break

        stop_accept = threading.Event()
        accept_thread = threading.Thread(target=accept_loop, args=(stop_accept,), daemon=True)
        accept_thread.start()

        # ── LIKWID command wrapping ───────────────────────────────────────────
        # likwid-perfctr must be the outer process; env vars (incl. HPROFILER_SOCKET)
        # are inherited by the child so all socket-based hooks still work.
        _likwid_backend = None
        run_command = list(self.command)
        if "likwid" in self.backends:
            from ..backends.likwid import LIKWIDBackend
            _likwid_backend = LIKWIDBackend()
            run_command = _likwid_backend.wrap_command(self.command)

        # ── Start the profiled process directly ───────────────────────────────
        # perf record and perf stat attach via -p PID so we can run both
        # simultaneously without nesting and still get the env vars right.
        # Read HPROFILER_CALLGRAPH from env_extra without mutating caller's dict.
        callgraph = self.env_extra.get("HPROFILER_CALLGRAPH")
        proc = subprocess.Popen(run_command, env=env)
        metadata.start_time_ns = time.monotonic_ns()
        pid = proc.pid

        # ── Attach perf record for CPU sampling ───────────────────────────────
        perf_record_proc: subprocess.Popen | None = None
        perf_data: str | None = None
        if "cpu" in self.backends and shutil.which("perf"):
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
        # Run for any backend where CPU-side behaviour is interesting:
        # cpu/openmp/likwid are obviously CPU-bound; opencl dispatches from the
        # CPU so IPC/cache-miss still give useful context.
        perf_stat_proc: subprocess.Popen | None = None
        perf_stat_file: str | None = None
        # Collect CPU microarch counters for any backend that dispatches from CPU.
        # GPU backends (cuda/rocm/nccl/mpi) are included because IPC and cache-
        # miss rates help diagnose CPU-side launch overhead and data-prep costs.
        _cpu_backends = {"cpu", "openmp", "likwid", "opencl", "cuda", "rocm", "nccl", "mpi"}
        if shutil.which("perf") and any(b in self.backends for b in _cpu_backends):
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

        stop_accept.set()
        accept_thread.join(timeout=2)
        for t in client_threads:
            t.join(timeout=1)
        server_sock.close()
        try:
            os.unlink(sock_path)
        except OSError:
            pass
        try:
            os.rmdir(_sock_dir)
        except OSError:
            pass

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
        if self.backends and not trace.spans and not trace.instants:
            import sys as _sys1
            msg = _total_zero_event_warning(self.backends, self.command)
            print(msg, file=_sys1.stderr)

        # ── OpenMP zero-event sanity check ────────────────────────────────────
        # Two independent capture paths are injected together (see
        # src/backends/openmp.py's module docstring): OMPT (needs LLVM
        # libomp) and direct GOMP_* interception (needs GNU libgomp, no
        # OMPT dependency). If BOTH produced zero events, something else is
        # wrong — warn with troubleshooting steps for both paths rather
        # than assuming which one should have worked.
        if "openmp" in self.backends:
            import sys as _sys2
            omp_spans = [s for s in trace.spans if s.category.value == "openmp"]
            if not omp_spans:
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


def _total_zero_event_warning(backends: list[str], command: list[str]) -> str:
    """
    Message for when a run completes but captures ZERO events of any
    kind across every active backend -- a much stronger signal than any
    one backend's own zero-event check (see the "openmp" one right after
    this is used in .run()): it means the hooks never connected to the
    collector socket for this entire run, not that one backend's
    particular constructs simply weren't exercised.

    A real user hit exactly this running via `srun` (SLURM): the run
    completed normally (GROMACS printed its full performance summary)
    but captured zero spans of any kind, then the IDENTICAL command
    captured 60381 events on the very next invocation with no code
    change in between. Every hook's ensure_connected() is retried on
    every single emit call, not just once at process startup, so a
    transient "listener wasn't ready yet" race would only ever lose the
    first few events -- not literally all of them across a multi-second
    run. A total loss for the whole run instead points to
    HPROFILER_SOCKET/LD_PRELOAD never having reached the profiled
    process's environment at all, which is exactly what happens when a
    job launcher (srun/mpirun/aprun/...) doesn't propagate the parent
    environment to the process(es) it actually spawns -- SLURM in
    particular can do this depending on site defaults / whether
    `--export` was set, and it can be intermittent (site-dependent
    scheduling/environment-cache behavior), matching the user's "works
    on the very next run" report -- not something hprofiler's own retry
    logic can work around from inside the already-spawned process.
    """
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
    real_binary_paths: "set[str] | None" = None,
) -> None:
    """
    Post-run: extract disassembly for all profiled kernels and attach to trace.

    JIT .so paths come from spans emitted by the OpenCL hook when it
    intercepts dlopen of ACPP SSCP .jit.so files (tag type=jit_load, path=...).

    real_binary_paths: real exe path(s) resolved via SO_PEERCRED on each
    hook's socket connection (see _peer_real_exe) -- the actual profiled
    binary, correct regardless of whether `command[0]` is a launcher
    (`srun`/`mpirun`). Passed through to collect_disasm() for the CUDA/
    ROCm AoT disasm paths, which (unlike OpenMP/MPI's dladdr-resolved
    sym=/symfile= tags) disassemble the whole binary keyed off
    command[0] and had no launcher-aware fallback at all until this.
    """
    # Resolve the binary path: it may be relative (e.g. './main').
    # Use the saved cwd from the trace metadata to make it absolute.
    cwd = trace.metadata.cwd or ""
    if command and cwd:
        import os as _os
        binary_rel = command[0]
        command = [str(_os.path.join(cwd, binary_rel)
                       if not _os.path.isabs(binary_rel) else binary_rel)] + command[1:]

    try:
        from ..disasm.extractor import collect_disasm
    except ImportError:
        return

    # Build jit_spans, omp_syms, and cpu_names from trace events
    jit_spans: list[dict] = []
    # omp_syms:  {span_name: ("sym", mangled_name)}
    #         or {span_name: ("lib", (lib_path, static_offset))}
    omp_syms: dict[str, tuple] = {}
    # cpu_names: function names from perf-sampled CPU spans (no sym/lib tags)
    cpu_names: set[str] = set()

    for span in trace.spans:
        # OpenCL/ACPP SSCP JIT .so files
        if span.category == Category.JIT and span.tags.get("type") == "jit_load":
            so_path = span.tags.get("path", "")
            if so_path and (".jit.so" in so_path or "hprofiler_jit_" in so_path
                            or "hprofiler_ocl_" in so_path):
                jit_spans.append({
                    "name":    span.name,
                    "so_path": so_path,
                    "mangled": span.tags.get("mangled", ""),
                })
        # ROCm JIT: hipModuleLoadData emits type=jit_compile with path= of saved ELF
        if span.category == Category.JIT and span.tags.get("type") == "jit_compile":
            jit_path = span.tags.get("path", "")
            if jit_path and "hprofiler_rocm_" in jit_path:
                jit_spans.append({
                    "name":    span.name,
                    "so_path": "",
                    "path":    jit_path,
                })
        # OpenMP/MPI/CPU: extract the first resolved codeptr info per span
        # name. Hook emits sym=<mangled> (dladdr success) or
        # lib=<path>,offset=0x<off> (fallback) -- "mpi" was missing here
        # even though mpi_hook.c's collectives + MPI_Barrier do emit these
        # tags, so an MPI span's own call-site info was silently never
        # looked at; the Source tab's kernel list includes every profiled
        # span name (not just GPU kernels), so MPI_Bcast/MPI_Allreduce/
        # MPI_Barrier showed up there with "No disassembly available"
        # unconditionally, not because objdump was missing.
        #
        # sym=<mangled> is paired with an optional symfile=<path> -- the
        # ELF file dladdr() actually found the symbol in. The profiled
        # command is routinely a launcher wrapping the real binary
        # (`hprofiler run -- srun -n 4 gmx_mpi ...`), so command[0] (what
        # collect_disasm() used to assume was always the right file to
        # `nm`/disassemble) is `srun`, not the profiled program -- a real
        # user's genuinely-resolved sym= still produced "No disassembly
        # available" for exactly this reason. symfile is None when the
        # hook build predates this fix; collect_disasm() falls back to
        # command[0] in that case, same as before.
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

    # Any ONE real binary path is strictly better than command[0] when
    # that's a launcher -- for the common SPMD case (all ranks running
    # the same binary) any one is correct; picking one is not an attempt
    # at "the right" rank, there generally isn't a wrong one here.
    real_binary = next(iter(real_binary_paths), "") if real_binary_paths else ""

    try:
        disasm_map = collect_disasm(command, backends, jit_spans, omp_syms, profiled_pid, cpu_names,
                                    sm_version=sm_version, real_binary=real_binary)

        import copy as _copy

        # CUDA/ROCm ACPP JIT: kernel launches are recorded as "<jit-kernel>"
        # because dladdr can't resolve JIT function pointers.  Map the first
        # captured JIT kernel to that name so clicking it shows something.
        _JIT_NAME = "<jit-kernel>"
        if (any(s.name == _JIT_NAME for s in trace.spans)
                and _JIT_NAME not in disasm_map):
            # Alias first available GPU kernel to <jit-kernel> so spans get
            # disassembly even when the hook couldn't resolve the name (e.g.
            # native HIP AoT where stubs aren't in .dynsym).
            for kd in disasm_map.values():
                if kd.arch in ("ptx", "sass", "amdgcn"):
                    alias = _copy.copy(kd)
                    alias.name = _JIT_NAME
                    disasm_map[_JIT_NAME] = alias
                    break

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
    """
    Parse perf script output into SpanEvents.

    Handles both formats:
      Flat:  comm pid ts: period event: addr sym (dso)
      Stack: same header followed by indented frame lines, blank-line separated.

    In stack mode (--perf-callgraph was passed), each SAMPLE becomes ONE
    SpanEvent -- name=the leaf/currently-executing frame, stack_frames=the
    ancestor chain (innermost-first, same convention hook-captured spans
    already use) -- not one span per frame per sample the way this used
    to work. That old shape (N spans per sample, duration_ns=0,
    stack_frames never set, the whole stack redundantly duplicated as a
    string in tags["stack"]) was invisible to analysis/call_tree.py's
    _ct_build (requires duration_ns>0 AND stack_frames truthy) and only
    ever got consumed by analysis/cct.py's own separate tag-string
    re-parser -- which already prefers span.stack_frames when present
    (cct.py's _extract_frames, checked first), so this needs no matching
    change there. `duration_ns` is a nominal per-sample weight
    (1e9/freq ns, i.e. "this sample represents one sampling interval"),
    the same assumption perf's own report/annotate percentages already
    make -- there's no real "duration" for a single sampled instant.
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
    # Nominal per-sample weight: each sample stands in for one sampling
    # interval's worth of wall time. Only used in stack mode -- the flat
    # (no --perf-callgraph) path below is intentionally left at
    # duration_ns=0, unchanged, out of scope for this fix.
    sample_weight_ns = max(1, round(1_000_000_000 / freq)) if freq > 0 else 1

    def _flush():
        if not cur_ts:
            return
        rel_ts = cur_ts
        if cur_stack:
            # cur_stack is innermost-first (backtrace order) exactly as
            # appended -- cur_stack[0] is the leaf/currently-executing
            # frame, cur_stack[1:] is its ancestor chain, already in the
            # same innermost-first order SpanEvent.stack_frames expects
            # (see core/events.py's field comment) -- no reversal needed.
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
