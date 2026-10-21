"""
What a trace's jit spans cover: which compilation stage each one is
(stage= tag, set by the hooks), so a summary never presents a driver's
module-load time as the program's JIT cost.

  frontend  source -> IR/PTX/code object inside the program
            (nvrtcCompileProgram, hiprtcCompileProgram; with `--python-jit`
            Triton's Python -> ... -> cubin, "triton_jit_compile", per-stage
            tags, src/instrument/pysite/; its cache hits are
            "triton_jit_cache" instants)
  runtime   a language runtime's own JIT before any driver call: AdaptiveCpp
            SSCP stage 2 (LLVM IR -> SPIR-V/PTX/AMDGCN/host .jit.so), the
            span "acpp_jit_compile" between a persistent kernel-cache miss
            and its store (hooks/common/hp_acpp.h); warm runs record only
            "acpp_jit_cache" instants (cache=hit)
  source    an OpenCL build from OpenCL C: the driver runs front and back end
  backend   driver compilation or module load of IR/PTX/SPIR-V/code objects
            (cuModuleLoadData*, hipModuleLoadData*, clBuildProgram on IL or
            binaries, clLinkProgram)
  load      ready-made code (ACPP .jit.so dlopen, saved program binaries,
            HIP fat-binary registration): no compilation at all
  build     an OpenCL build in a trace from hooks that did not record whether
            it started from source or IL

Other runtimes that generate their own IR before calling the driver
(Numba, ...; Triton without --python-jit) do that in no hooked call, so it
is in no span.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.events import SpanEvent
    from ..core.trace import Trace

STAGES = {
    "frontend": "front-end compilation",
    "runtime":  "runtime JIT (AdaptiveCpp stage 2)",
    "source":   "driver build from source",
    "backend":  "back-end compile / module load",
    "load":     "loading ready-made code",
    "build":    "driver build (source or IL unknown)",
}
FRONTEND_STAGES = ("frontend", "runtime", "source")

_FRONTEND_APIS = ("nvrtcCompileProgram", "hiprtcCompileProgram")
_OPENCL_BUILDS = ("clBuildProgram", "clCompileProgram")


def jit_stage(span: "SpanEvent") -> str:
    """The STAGES key of one jit span (inferred for traces without stage=)."""
    st = span.tags.get("stage")
    if st in STAGES:
        return st
    if span.name in _FRONTEND_APIS:
        return "frontend"
    if span.name == "acpp_jit_compile":
        return "runtime"
    if span.tags.get("type") == "jit_load" or span.name == "__hipRegisterFatBinary":
        return "load"
    if span.name in _OPENCL_BUILDS:
        return "build"
    return "backend"


def jit_coverage(trace: "Trace") -> dict | None:
    """{"stages": {stage: {"count", "total_ns", "names"}}, "frontend_seen",
    "note"} for the trace's jit spans, None without any."""
    stages: dict[str, dict] = {}
    for s in trace.iter_spans(categories=("jit",)):
        e = stages.setdefault(jit_stage(s), {"count": 0, "total_ns": 0, "names": set()})
        e["count"] += 1
        e["total_ns"] += s.duration_ns
        e["names"].add(s.name)
    if not stages:
        return None
    frontend = any(st in stages for st in FRONTEND_STAGES)
    note = ""
    if not frontend and any(st in stages for st in ("backend", "load")):
        note = ("back-end compile / load spans only: no front-end or runtime compilation was "
                "observed. Either everything came from a persistent cache (a warm run), or a "
                "runtime generated its own IR, PTX or SPIR-V in no hooked call -- Python JITs "
                "such as Triton do (record Triton's with `hprofiler run --python-jit`) -- so "
                "these totals are not the program's JIT cost.")
    elif "build" in stages and not frontend:
        note = ("OpenCL builds recorded without stage= (older hooks): whether they compiled "
                "OpenCL C or only lowered IL (SPIR-V) is unknown.")
    return {"stages": stages, "frontend_seen": frontend, "note": note}
