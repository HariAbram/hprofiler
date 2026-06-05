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
import shutil
import socket
import subprocess
import threading
import time
import tempfile
import struct
import socket as _socket
from pathlib import Path
from typing import Callable

from .events import SpanEvent, InstantEvent, CounterEvent, Category, AnyEvent  # noqa: F401
from .trace import Trace, TraceMetadata

HOOKS_DIR = Path(__file__).parent.parent.parent / "build" / "lib"

# Wire protocol from C hooks: newline-delimited ASCII records
# span:<category>:<pid>:<tid>:<start_ns>:<dur_ns>:<name>[:<tag=val>...]
# inst:<category>:<pid>:<tid>:<ts_ns>:<name>
# ctr:<category>:<pid>:<ts_ns>:<name>:<value>:<unit>


def _parse_record(line: str) -> AnyEvent | None:
    try:
        parts = line.strip().split(":", 6)
        kind = parts[0]
        if kind == "span" and len(parts) >= 7:
            _, cat, pid, tid, start_ns, dur_ns, rest = parts[0], parts[1], int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5]), parts[6]
            name_tags = rest.split(":", 1)
            name = name_tags[0]
            tags: dict = {}
            if len(name_tags) > 1:
                for kv in name_tags[1].split(","):
                    if "=" in kv:
                        k, v = kv.split("=", 1)
                        tags[k] = v
            return SpanEvent(
                name=name,
                category=Category(cat) if cat in Category._value2member_map_ else Category.OTHER,
                start_ns=start_ns,
                duration_ns=dur_ns,
                pid=pid,
                tid=tid,
                tags=tags,
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
            _, cat, pid, tid, ts_ns, name = parts[:6]
            return InstantEvent(
                name=name,
                category=Category(cat) if cat in Category._value2member_map_ else Category.OTHER,
                timestamp_ns=int(ts_ns),
                pid=int(pid),
                tid=int(tid),
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

    def run(self) -> Trace:
        import socket as sock_mod
        import platform

        sock_path = tempfile.mktemp(suffix=".sock", prefix="hprofiler_")

        server_sock = sock_mod.socket(sock_mod.AF_UNIX, sock_mod.SOCK_STREAM)
        server_sock.bind(sock_path)
        server_sock.listen(16)
        server_sock.settimeout(0.5)

        env = dict(os.environ)
        env["HPROFILER_SOCKET"] = sock_path
        env.update(self.env_extra)

        preload_libs: list[str] = []
        if "cuda" in self.backends:
            lib = HOOKS_DIR / "libhprofiler_cuda.so"
            if lib.exists():
                preload_libs.append(str(lib))
        if "opencl" in self.backends:
            lib = HOOKS_DIR / "libhprofiler_opencl.so"
            if lib.exists():
                preload_libs.append(str(lib))
        if "rocm" in self.backends:
            lib = HOOKS_DIR / "libhprofiler_rocm.so"
            if lib.exists():
                preload_libs.append(str(lib))

        if preload_libs:
            existing = env.get("LD_PRELOAD", "")
            env["LD_PRELOAD"] = ":".join(filter(None, [existing] + preload_libs))

        if "openmp" in self.backends:
            lib = HOOKS_DIR / "libhprofiler_ompt.so"
            if lib.exists():
                existing = env.get("OMP_TOOL_LIBRARIES", "")
                env["OMP_TOOL_LIBRARIES"] = ":".join(filter(None, [existing, str(lib)]))

        metadata = TraceMetadata(
            command=self.command[0],
            args=self.command[1:],
            start_time_ns=time.monotonic_ns(),
            pid=os.getpid(),
            backends_used=list(self.backends),
            hostname=platform.node(),
            cwd=os.getcwd(),
        )
        trace = Trace(metadata)
        self._trace = trace

        events_lock = threading.Lock()
        client_threads: list[threading.Thread] = []

        def handle_client(client: sock_mod.socket) -> None:
            buf = ""
            try:
                while True:
                    data = client.recv(4096)
                    if not data:
                        break
                    buf += data.decode("utf-8", errors="replace")
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        ev = _parse_record(line)
                        if ev is not None:
                            with events_lock:
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
                except sock_mod.timeout:
                    continue
                except Exception:
                    break

        stop_accept = threading.Event()
        accept_thread = threading.Thread(target=accept_loop, args=(stop_accept,), daemon=True)
        accept_thread.start()

        # ── Start the profiled process directly ───────────────────────────────
        # perf record and perf stat attach via -p PID so we can run both
        # simultaneously without nesting and still get the env vars right.
        callgraph = self.env_extra.pop("HPROFILER_CALLGRAPH", None)
        proc = subprocess.Popen(self.command, env=env)
        pid = proc.pid

        # ── Attach perf record for CPU sampling ───────────────────────────────
        perf_record_proc: subprocess.Popen | None = None
        perf_data: str | None = None
        if "cpu" in self.backends and shutil.which("perf"):
            perf_data = tempfile.mktemp(suffix=".perf.data", prefix="hprofiler_")
            perf_cmd = [
                "perf", "record",
                f"-F{self.perf_freq}", "-e", "cycles:u",
                "-p", str(pid), "-o", perf_data,
            ]
            if callgraph:
                perf_cmd.append(f"--call-graph={callgraph}")
            try:
                perf_record_proc = subprocess.Popen(
                    perf_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            except FileNotFoundError:
                perf_record_proc = None
                perf_data = None

        # ── Attach perf stat for CPU microarch counters ───────────────────────
        perf_stat_proc: subprocess.Popen | None = None
        perf_stat_file: str | None = None
        if shutil.which("perf"):
            perf_stat_file = tempfile.mktemp(suffix=".perf_stat.txt", prefix="hprofiler_")
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
            except FileNotFoundError:
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

        # ── Parse perf record output ──────────────────────────────────────────
        if perf_data and Path(perf_data).exists():
            _parse_perf_script(perf_data, trace, metadata.start_time_ns)
            try:
                os.unlink(perf_data)
            except OSError:
                pass

        # ── Emit microarch counter events ─────────────────────────────────────
        if perf_stat_file:
            _collect_microarch_counters(perf_stat_file, trace, metadata.end_time_ns)

        # ── Emit max-RSS counter event ────────────────────────────────────────
        _collect_rss(rss_before, rss_after, trace, metadata.end_time_ns)

        # ── Query device theoretical peaks ────────────────────────────────────
        try:
            from ..analysis.device import query_devices
            devs = query_devices(self.backends)
            if devs:
                trace.set_devices(devs)
        except Exception:
            pass

        if self.collect_disasm:
            import threading as _threading
            _threading.Thread(
                target=_collect_disasm,
                args=(trace, self.command, self.backends),
                daemon=True,
            ).start()
        return trace


# ── GPU utilization polling ───────────────────────────────────────────────────

def _gpu_poll(trace: Trace, backends: list[str], stop: threading.Event) -> None:
    """Background thread: poll GPU utilisation every second, emit CounterEvents."""
    if "cuda" in backends and shutil.which("nvidia-smi"):
        _poll_nvidia_smi(trace, stop)
    elif "rocm" in backends and shutil.which("rocm-smi"):
        _poll_rocm_smi(trace, stop)


def _poll_nvidia_smi(trace: Trace, stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            r = subprocess.run(
                ["nvidia-smi",
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
                suf = f"[gpu{idx}]" if idx else ""
                trace.add(CounterEvent(f"gpu_utilization_pct{suf}",
                                        Category.MEMORY, ts, util_gpu, "%"))
                trace.add(CounterEvent(f"gpu_mem_util_pct{suf}",
                                        Category.MEMORY, ts, util_mem, "%"))
                trace.add(CounterEvent(f"gpu_mem_used_bytes{suf}",
                                        Category.MEMORY, ts, mem_mib * 1024**2, "bytes"))
        except Exception:
            pass
        stop.wait(timeout=0.1)


def _poll_rocm_smi(trace: Trace, stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            r = subprocess.run(
                ["rocm-smi", "--showuse", "--showmeminfo", "vram", "--csv"],
                capture_output=True, text=True, timeout=3,
            )
            ts = time.monotonic_ns()
            for line in r.stdout.strip().splitlines():
                if line.startswith("#") or "," not in line:
                    continue
                parts = [p.strip() for p in line.split(",")]
                # rocm-smi CSV columns vary by version; look for numeric fields
                nums = []
                for p in parts:
                    try:
                        nums.append(float(p))
                    except ValueError:
                        pass
                if len(nums) >= 2:
                    trace.add(CounterEvent("gpu_utilization_pct",
                                            Category.MEMORY, ts, nums[0], "%"))
                    trace.add(CounterEvent("gpu_mem_used_bytes",
                                            Category.MEMORY, ts, nums[1] * 1024**2, "bytes"))
        except Exception:
            pass
        stop.wait(timeout=0.1)


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
        raw = re.sub(r"[^\d]", "", parts[0])
        try:
            val = float(raw)
        except ValueError:
            continue
        key = re.sub(r"^[\w-]+/", "", parts[1].rstrip("/")).lower().strip()
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

def _collect_disasm(trace: Trace, command: list[str], backends: list[str]) -> None:
    """
    Post-run: extract disassembly for all profiled kernels and attach to trace.

    JIT .so paths come from spans emitted by the OpenCL hook when it
    intercepts dlopen of ACPP SSCP .jit.so files (tag type=jit_load, path=...).
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

    # Build jit_spans and omp_codeptrs from trace events
    jit_spans: list[dict] = []
    # omp_syms:  {span_name: ("sym", mangled_name)}
    #         or {span_name: ("lib", (lib_path, static_offset))}
    omp_syms: dict[str, tuple] = {}

    for span in trace.spans:
        # OpenCL/ACPP SSCP JIT .so files
        if span.category == Category.JIT and span.tags.get("type") == "jit_load":
            so_path = span.tags.get("path", "")
            if so_path and (".jit.so" in so_path or "hprofiler_jit_" in so_path):
                jit_spans.append({
                    "name":    span.name,
                    "so_path": so_path,
                    "mangled": span.tags.get("mangled", ""),
                })
        # OpenMP/CPU: extract the first resolved codeptr info per span name.
        # Hook emits sym=<mangled> (dladdr success) or lib=<path>,offset=0x<off>
        if span.category.value in ("openmp", "sync", "cpu") and span.name not in omp_syms:
            sym = span.tags.get("sym", "")
            if sym:
                omp_syms[span.name] = ("sym", sym)
                continue
            lib = span.tags.get("lib", "")
            off_s = span.tags.get("offset", "")
            if lib and off_s:
                try:
                    omp_syms[span.name] = ("lib", (lib, int(off_s, 16)))
                except ValueError:
                    pass

    try:
        disasm_map = collect_disasm(command, backends, jit_spans, omp_syms)

        import copy as _copy

        # CUDA/ROCm ACPP JIT: kernel launches are recorded as "<jit-kernel>"
        # because dladdr can't resolve JIT function pointers.  Map the first
        # captured JIT kernel to that name so clicking it shows something.
        _JIT_NAME = "<jit-kernel>"
        if (any(s.name == _JIT_NAME for s in trace.spans)
                and _JIT_NAME not in disasm_map):
            for kd in disasm_map.values():
                if kd.arch in ("ptx", "sass", "amdgcn") and (
                    "hprofiler_cubin_" in kd.source
                    or "hprofiler_rocm_" in kd.source
                ):
                    alias = _copy.copy(kd)
                    alias.name = _JIT_NAME
                    disasm_map[_JIT_NAME] = alias
                    break

        for kd in disasm_map.values():
            trace.add_disasm(kd)
    except Exception:
        pass   # disasm is best-effort; never crash the profiler run


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


def _parse_perf_script(perf_data: str, trace: Trace, trace_start_ns: int) -> None:
    """
    Parse perf script output into SpanEvents.

    Handles both formats:
      Flat:  comm pid ts: period event: addr sym (dso)
      Stack: same header followed by indented frame lines, blank-line separated.
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

    def _flush():
        if not cur_ts:
            return
        name = cur_stack[0] if cur_stack else (cur_top_sym or "[cpu]")
        if name:
            trace.add(SpanEvent(
                name=name, category=Category.CPU,
                start_ns=max(0, cur_ts - trace_start_ns),
                duration_ns=0, pid=cur_pid, tid=cur_tid,
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
                    start_ns=max(0, cur_ts - trace_start_ns),
                    duration_ns=0, pid=cur_pid, tid=cur_tid,
                ))
            continue

        # Try stack frame
        if has_stacks:
            m = _FRAME.match(raw_line)
            if m:
                sym = _sym_from_match(m.group(1), m.group(2))
                if sym:
                    cur_stack.insert(0, sym)

    if has_stacks:
        _flush()
