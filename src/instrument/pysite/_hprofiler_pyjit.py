"""
Python JIT compilation as hprofiler jit spans (`hprofiler run --python-jit`;
loaded by sitecustomize.py next to this file).

Triton (3.x) compiles a kernel at its first launch -- Python -> TTIR ->
TTGIR -> LLIR -> PTX -> ptxas -> cubin -- in Python and in its own
libraries, inside no hooked call (only the final cuModuleLoadData is seen).
triton.knobs.compilation.listener is called at the end of every compile()
with the stage durations (CompileTimer: ir_initialization, the lowering
stages, store_results; microseconds) and whether the cache had it:

  span    jit "triton_jit_compile"  type=jit_compile,stage=frontend,cache=miss,cache_hit=0,
                                    hash=<kernel hash>,kernel=<name>,ir_init_ns=,<stage>_ns=...,
                                    store_ns=,stages=ttir/ttgir/...
  instant jit "triton_jit_cache"    type=jit_cache,cache=hit,cache_hit=1,hash=,kernel=,lookup_ns=

The listener runs right after compile() takes its end time, so the span ends
at the call (CLOCK_MONOTONIC -- the hooks' clock, time.monotonic_ns() on
Linux) and the stages are laid out back to back before it. A listener the
program set before this one is registered is chained (called after this
one, with the same arguments); one it sets later replaces this one.

Records go to the collector over HPROFILER_SOCKET as wire lines, like a
hook's, one connection per process (a forked child opens its own), ended
by a final transport status (hook "pyjit") so the collector knows nothing
was cut off. Nothing here may break the program: every failure disables it.
"""
import atexit
import os
import re
import socket
import sys
import threading
import time

_lock = threading.Lock()
_sock = None
_sock_pid = 0
_failed = False
_counts = {"emitted": 0, "sent": 0, "bytes": 0}
_image_ns = time.monotonic_ns()
_UNSAFE = re.compile(r"[^A-Za-z0-9_.+-]")   # tag values: never ':' ',' ';' '='


def _tag(v) -> str:
    return _UNSAFE.sub("_", str(v))[:256]


def _send(line: str) -> None:
    global _sock, _sock_pid, _failed, _image_ns
    with _lock:
        if _sock is not None and _sock_pid != os.getpid():     # forked: the parent's socket
            _sock, _failed, _image_ns = None, False, time.monotonic_ns()
            _counts.update(emitted=0, sent=0, bytes=0)
        if _failed:
            return
        _counts["emitted"] += 1
        data = line.encode("utf-8", "replace")
        try:
            if _sock is None:
                addr = os.environ["HPROFILER_SOCKET"]
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.connect("\0" + addr[1:] if addr.startswith("@") else addr)
                _sock, _sock_pid = s, os.getpid()
            _sock.sendall(data)
            _counts["sent"] += 1
            _counts["bytes"] += len(data)
        except (OSError, KeyError):
            _failed = True


@atexit.register
def _final_status() -> None:
    """The transport status a hook sends last (receiver.parse_xport)."""
    global _sock
    with _lock:
        if _sock is None or _sock_pid != os.getpid():
            return
        line = (f"xport:1:{os.getpid()}:pyjit:mode=sync,final=1,emitted={_counts['emitted']},"
                f"sent={_counts['sent']},bytes={_counts['bytes']},image={_image_ns}\n")
        try:
            _sock.sendall(line.encode())
            _sock.close()
        except OSError:
            pass
        _sock = None


def _record(src, metadata, times, cache_hit: bool, end_ns: int) -> None:
    us = 1000
    ir = int(getattr(times, "ir_initialization", 0) or 0) * us
    stages = [(str(n), int(d or 0) * us) for n, d in (getattr(times, "lowering_stages", None) or [])]
    store = int(getattr(times, "store_results", 0) or 0) * us
    total = ir + sum(d for _, d in stages) + store
    md = metadata or {}
    name = md.get("name") or getattr(src, "name", "") or getattr(getattr(src, "fn", None), "__name__", "")
    common = f"hash={_tag(md.get('hash', ''))},kernel={_tag(name)}"
    pid, tid = os.getpid(), threading.get_native_id()
    if cache_hit:
        _send(f"inst:jit:{pid}:{tid}:{end_ns}:triton_jit_cache:type=jit_cache,cache=hit,cache_hit=1,"
              f"{common},lookup_ns={total}\n")
        return
    parts = [f"type=jit_compile,stage=frontend,cache=miss,cache_hit=0,{common},ir_init_ns={ir}"]
    parts += [f"{_tag(n)}_ns={d}" for n, d in stages]
    parts.append(f"store_ns={store}")
    if stages:
        parts.append("stages=" + "/".join(_tag(n) for n, _ in stages))
    _send(f"span:jit:{pid}:{tid}:{end_ns - total}:{total}:triton_jit_compile:{','.join(parts)}\n")


def _chain(prev):
    def listener(*, src, metadata, metadata_group, times, cache_hit):
        end_ns = time.monotonic_ns()
        try:
            _record(src, metadata, times, cache_hit, end_ns)
        except Exception:
            pass
        if prev is not None:
            prev(src=src, metadata=metadata, metadata_group=metadata_group, times=times, cache_hit=cache_hit)
    listener._hprofiler_prev = prev
    return listener


def _hook_triton(triton_module) -> None:
    try:
        knobs = getattr(triton_module, "knobs", None)
        if knobs is None:
            import triton.knobs as knobs
        comp = knobs.compilation
        prev = getattr(comp, "listener", None)
        if not hasattr(prev, "_hprofiler_prev"):
            comp.listener = _chain(prev)
    except Exception as e:
        print(f"[hprofiler] python-jit: cannot instrument Triton ({e!r})", file=sys.stderr)


class _TritonFinder:
    """Runs _hook_triton() once `import triton` has executed the package."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "triton":
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None:
                break
        else:
            return None
        loader = spec.loader
        if loader is None or not hasattr(loader, "exec_module"):
            return spec
        real_exec = loader.exec_module

        def exec_module(module):
            real_exec(module)
            _hook_triton(module)
        loader.exec_module = exec_module
        try:
            sys.meta_path.remove(self)                # once
        except ValueError:
            pass
        return spec


def install() -> None:
    if "triton" in sys.modules:
        _hook_triton(sys.modules["triton"])
    else:
        sys.meta_path.insert(0, _TritonFinder())
