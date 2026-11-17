"""
AdaptiveCpp (SYCL) dependencies: which reach the trace, which do not.

AdaptiveCpp (25.10) has no tracing interface for its task graph: nothing
names a command's requirements to a tool (its instrumentation API gives
per-command timestamps, runtime_event_handlers only allocations; the DAG
is internal C++, no stable ABI). Its dependencies reach the trace only
where the runtime lowers them to calls a hook records -- checked on this
machine with tests/fixtures/sycl_deps.cpp:

  CUDA     cudaEventRecord + cudaStreamWaitEvent between its streams (an
           event wait between the device spans), stream order, or a host
           wait (cudaEventSynchronize) before the next submission
  OpenCL   one in-order queue per device: queue order (stronger than the
           SYCL dependencies), barriers with wait lists, clWaitForEvents;
           USM copies / fills / prefetches through the Intel USM extension
           (opencl_hook.c wraps the extension lookup)
  HIP      like CUDA (not run here: no AMD GPU)

Not recorded -- counted here, never guessed:
  * host-device kernels (category sycl, its OpenMP backend): the runtime
    orders them in its own worker threads, so a dependency on a command of
    another queue or device is a wait inside it -- the kernel calls have
    only their threads' order (sycl_host_kernel_calls);
  * a CUDA / HIP command depending on a host-device command: lowered to a
    stream host function that waits inside the runtime -- the host
    function is recorded, not what it waited for; its time counts as the
    stream's (acpp_host_functions: host functions in AdaptiveCpp processes,
    which may also be the program's own).
"""
from __future__ import annotations

_MARKERS = ("__acpp_", "hipsycl")


def _acpp_name(name: str) -> bool:
    return name.startswith("acpp_jit") or any(m in name for m in _MARKERS)


def dependency_diagnostics(trace) -> dict:
    calls, call_pids, acpp = 0, set(), set()
    for s in trace.iter_spans(categories=("sycl",)):
        calls += 1
        call_pids.add(s.pid)
    acpp |= call_pids
    host_funcs: dict[int, int] = {}
    for s in trace.iter_spans(categories=("jit", "cuda", "rocm", "opencl")):
        if _acpp_name(s.name):
            acpp.add(s.pid)
        if s.tags.get("op") == "host_func" and s.tags.get("side") == "gpu":
            host_funcs[s.pid] = host_funcs.get(s.pid, 0) + 1
    if not acpp:
        acpp = {i.pid for i in trace.iter_instants() if i.category.value == "jit" and _acpp_name(i.name)}
    out = {}
    if calls:
        out["sycl_host_kernel_calls"] = calls
        out["sycl_host_kernel_processes"] = len(call_pids)
    n = sum(c for pid, c in host_funcs.items() if pid in acpp)
    if n:
        out["acpp_host_functions"] = n
    return out


def dependency_warnings(diag: dict) -> list[str]:
    out = []
    if diag.get("sycl_host_kernel_calls"):
        out.append(f"{diag['sycl_host_kernel_calls']} host-device SYCL kernel call(s) in "
                   f"{diag.get('sycl_host_kernel_processes', 1)} process(es): AdaptiveCpp orders them in its own "
                   "worker threads and exposes no task graph, so a dependency on a command of another queue or "
                   "device is not recorded -- they have only their threads' order (no edge is guessed)")
    if diag.get("acpp_host_functions"):
        out.append(f"{diag['acpp_host_functions']} stream host function(s) in AdaptiveCpp process(es): "
                   "AdaptiveCpp lowers a device command's dependency on a host-device command to one, and the "
                   "wait inside it is not recorded -- its time counts as the stream's")
    return out
