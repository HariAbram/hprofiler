"""
POP-style parallel efficiency decomposition, computed inline from a live
hprofiler trace -- no offline network simulator (Dimemas) required.

Background: the POP (Performance Optimisation and Productivity) Centre of
Excellence standard metrics decompose total efficiency as

    Global Efficiency      = Parallel Efficiency x Computational Scaling
    Parallel Efficiency    = Load Balance x Communication Efficiency
    Communication Efficiency = Serialization Efficiency x Transfer Efficiency

Load Balance and Communication Efficiency are exact ratios of measured "useful
time" per rank and are computed here directly from the trace -- no
approximation needed. The Serialization/Transfer split is normally obtained
by replaying the trace over a simulated zero-contention network (Dimemas);
this module instead:

  - fits an empirical latency/bandwidth (alpha/beta) model directly from the
    trace's own population of (bytes, duration) pairs on MPI/NCCL spans, and
    reports how close actual communication time is to that fitted "ideal" as
    a `transfer_efficiency` PROXY -- this is an approximation, not a replay,
    and is reported as such (see EfficiencyReport.notes).
  - only computes a true `serialization_efficiency` when a CriticalPathReport
    (src/analysis/criticalpath.py) is supplied: the fraction of communication
    time that critical-path analysis shows was structurally unavoidable
    (blocked on a genuine cross-rank/cross-runtime dependency) vs. avoidable
    stalling. Without one, serialization_efficiency is left as None.

Computational Scaling additionally requires a second, lower-process-count
"baseline" trace -- inherent to the POP metric itself, not a limitation of
this module (POP's own methodology requires a reference case too).

GPU and NCCL layers are hprofiler-specific extensions beyond stock POP
(which only covers MPI+OpenMP): GPU efficiency reuses the existing
disassembly-based roofline analysis; NCCL efficiency uses NCCL's own
standard bus-bandwidth formula (the same metric nccl-tests reports).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.trace import Trace
    from ..core.events import SpanEvent
    from .criticalpath import CriticalPathReport

_USEFUL_CATS = frozenset({"cpu", "cuda", "rocm", "opencl", "openmp"})
_COMM_CATS = frozenset({"mpi", "nccl"})


# ── report ───────────────────────────────────────────────────────────────────

@dataclass
class EfficiencyReport:
    load_balance: float | None = None
    comm_efficiency: float | None = None
    parallel_efficiency: float | None = None
    transfer_efficiency: float | None = None          # proxy, duration-weighted across mpi/nccl -- see module docstring
    transfer_efficiency_by_category: dict[str, float] = field(default_factory=dict)  # per-category, before weighting
    serialization_efficiency: float | None = None      # only set if a critical path was supplied
    computational_scaling: float | None = None          # only set if a baseline trace was supplied
    global_efficiency: float | None = None
    gpu_efficiency: float | None = None                  # 0-1, duration-weighted mean of roofline flops_pct
    nccl_bus_bw_gbs: float | None = None                 # achieved, GB/s
    nccl_efficiency: float | None = None                 # only set if a peak bandwidth was supplied
    per_rank_useful_ns: dict[int, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


# ── interval merging (avoids double-counting overlapping GPU-stream spans) ────

def _merge_intervals(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    ivs = sorted(intervals)
    total = 0
    cur_lo, cur_hi = ivs[0]
    for lo, hi in ivs[1:]:
        if lo <= cur_hi:
            cur_hi = max(cur_hi, hi)
        else:
            total += cur_hi - cur_lo
            cur_lo, cur_hi = lo, hi
    total += cur_hi - cur_lo
    return total


def _wall_ns(trace: "Trace") -> int:
    # NOTE: deliberately NOT falling back to trace.duration_ns -- for a
    # trace reconstructed by load_trace_from_json, TraceMetadata.start_time_ns
    # defaults to the *load* time (dataclass field default_factory), not the
    # original run's start, making trace.duration_ns meaningless for a saved
    # trace (see the same fix/note in criticalpath.py's _wall_ns). Only
    # reachable here for a trace with no timed spans at all.
    timed = [s for s in trace.spans if s.duration_ns > 0]
    if not timed:
        return 1
    lo = min(s.start_ns for s in timed)
    hi = max(s.start_ns + s.duration_ns for s in timed)
    return max(hi - lo, 1)


# ── Load Balance / Communication Efficiency (exact) ────────────────────────────

_DATA_MOVEMENT_TYPES = frozenset({"memcpy", "memcpy_async", "alloc", "free", "HtoD", "DtoH"})


def useful_time_by_pid(trace: "Trace") -> dict[int, int]:
    """Merged (non-overlap-double-counted) 'useful compute' time per
    rank/process. Excludes memcpy/alloc/free spans even though they're
    tagged with a "compute" category (cuda/rocm) -- POP's definition of
    "useful computation" is specifically non-communication, non-data-movement
    time, so counting a large H2D/D2H transfer as "useful" would inflate
    Load Balance / Communication Efficiency for transfer-heavy GPU codes."""
    by_pid: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for s in trace.spans:
        if s.duration_ns <= 0 or s.category.value not in _USEFUL_CATS:
            continue
        if s.tags.get("type") in _DATA_MOVEMENT_TYPES:
            continue
        by_pid[s.pid].append((s.start_ns, s.start_ns + s.duration_ns))
    return {pid: _merge_intervals(ivs) for pid, ivs in by_pid.items()}


def load_balance(useful_by_pid: dict[int, int]) -> float | None:
    if not useful_by_pid:
        return None
    vals = list(useful_by_pid.values())
    mx = max(vals)
    if mx <= 0:
        return None
    return (sum(vals) / len(vals)) / mx


def comm_efficiency(useful_by_pid: dict[int, int], wall_ns: int) -> float | None:
    if not useful_by_pid or wall_ns <= 0:
        return None
    return min(1.0, max(useful_by_pid.values()) / wall_ns)


# ── Transfer efficiency proxy (self-calibrated alpha/beta fit) ────────────────

def _comm_spans_with_bytes(trace: "Trace") -> list["SpanEvent"]:
    """mpi/nccl spans that carry a bytes= tag -- suitable for the
    alpha/beta size-vs-duration fit, which needs a size per point."""
    out = []
    for s in trace.spans:
        if s.category.value in _COMM_CATS and s.duration_ns > 0 and "bytes" in s.tags:
            out.append(s)
    return out


def _all_comm_spans(trace: "Trace") -> list["SpanEvent"]:
    """Every mpi/nccl span, regardless of whether it carries a bytes= tag.
    Used as the "total communication time" denominator for serialization
    efficiency -- MPI_Barrier/MPI_Init/ncclCommInitAll etc. carry no bytes=
    tag (they don't transfer application data) but are still real
    communication/sync cost; excluding them would make Serialization
    Efficiency look artificially good whenever a lot of the actual overhead
    is in barriers rather than point-to-point/collective data transfer.

    EXCLUDES type=="group" (nccl_hook.c's "ncclGroup" span, emitted for
    ncclGroupStart/End): its [start,end] interval structurally CONTAINS
    every individual collective/P2P call made inside that group, each of
    which is ALSO in this list as its own span -- summing both (this
    function's only consumer, serialization_efficiency_from_path, does a
    naive sum(duration_ns), not an interval union) would double-count
    every grouped operation's duration in the "total communication time"
    denominator. No corresponding MPI construct needs the same exclusion
    (PMPI's profiling-interface convention doesn't produce a wrapper span
    like this for any MPI call)."""
    return [
        s for s in trace.spans
        if s.category.value in _COMM_CATS and s.duration_ns > 0
        and s.tags.get("type") != "group"
    ]


def fit_alpha_beta(spans: list["SpanEvent"]) -> tuple[float, float] | None:
    """Fit duration_ns ~= alpha + bytes/beta from a trace's own message
    population (a self-calibrating substitute for a Dimemas-style separate
    micro-benchmark run). Returns None if there's too little data or size
    variance to regress meaningfully."""
    import numpy as np

    pts = []
    for s in spans:
        try:
            b = float(s.tags["bytes"])
        except (KeyError, ValueError, TypeError):
            continue
        if b > 0:
            pts.append((b, float(s.duration_ns)))
    if len(pts) < 4:
        return None
    xs = np.array([p[0] for p in pts], dtype=float)
    ys = np.array([p[1] for p in pts], dtype=float)
    if float(np.ptp(xs)) == 0.0:
        return None
    slope, intercept = np.polyfit(xs, ys, 1)
    if slope <= 0:
        return None
    beta = 1.0 / slope           # ns per byte -> bytes per ns
    alpha = max(float(intercept), 0.0)
    return alpha, float(beta)


def transfer_efficiency_proxy(spans: list["SpanEvent"], alpha: float, beta: float) -> float:
    ideal_total = 0.0
    actual_total = 0.0
    for s in spans:
        try:
            b = float(s.tags.get("bytes", 0))
        except (ValueError, TypeError):
            b = 0.0
        ideal_total += alpha + (b / beta if beta > 0 else 0.0)
        actual_total += s.duration_ns
    if actual_total <= 0:
        return 1.0
    return min(1.0, ideal_total / actual_total)


def transfer_efficiency_by_category(
    trace: "Trace",
) -> tuple[dict[str, float], float | None, list[str]]:
    """Fit alpha/beta and score transfer efficiency SEPARATELY per
    communication category (mpi, nccl), then combine duration-weighted.

    MPI and NCCL have very different latency/bandwidth characteristics --
    MPI often crosses the node interconnet (network fabric), NCCL intra-node
    typically uses NVLink (10-100x higher bandwidth). Fitting one alpha/beta
    model across both mixed together would let whichever category has more
    data points or more size variance dominate the regression, producing a
    nonsensical "ideal" time for the other category (e.g. predicting MPI
    should be as fast as NVLink, or NCCL as slow as the network) -- fitting
    per category and only combining the final efficiency SCORES (not the
    raw data) avoids that.
    """
    notes: list[str] = []
    per_cat: dict[str, float] = {}
    weighted_sum, weight_total = 0.0, 0.0
    comm_spans = _comm_spans_with_bytes(trace)
    for cat in sorted(_COMM_CATS):
        cat_spans = [s for s in comm_spans if s.category.value == cat]
        fit = fit_alpha_beta(cat_spans)
        if fit is None:
            continue
        alpha, beta = fit
        eff = transfer_efficiency_proxy(cat_spans, alpha, beta)
        per_cat[cat] = eff
        w = sum(s.duration_ns for s in cat_spans)
        weighted_sum += w * eff
        weight_total += w

    if not per_cat:
        notes.append(
            "Not enough MPI/NCCL messages with varying byte sizes in this trace "
            "to fit a latency/bandwidth model -- transfer_efficiency omitted."
        )
        return per_cat, None, notes

    notes.append(
        "transfer_efficiency is a self-calibrated proxy (alpha/beta fit "
        "separately per communication category: " + ", ".join(sorted(per_cat)) +
        "), not a Dimemas-style network replay."
    )
    overall = weighted_sum / weight_total if weight_total > 0 else None
    return per_cat, overall, notes


# ── NCCL bus-bandwidth (standard nccl-tests metric) ────────────────────────────

def nccl_bus_bandwidth_gbs(trace: "Trace") -> float | None:
    """Achieved NCCL bus bandwidth using the standard ring-allreduce formula
    busBW = 2*(n-1)/n * bytes/time (same metric nccl-tests reports)."""
    samples = []
    for s in trace.spans:
        if s.category.value != "nccl" or s.duration_ns <= 0:
            continue
        if s.tags.get("type") != "allreduce":
            continue
        try:
            b = float(s.tags["bytes"])
            n = int(s.tags.get("nranks", -1))
        except (KeyError, ValueError, TypeError):
            continue
        if n < 2:
            continue
        secs = s.duration_ns / 1e9
        if secs <= 0:
            continue
        bus_bw = 2.0 * (n - 1) / n * b / secs / 1e9  # GB/s
        samples.append((s.duration_ns, bus_bw))
    if not samples:
        return None
    total_dur = sum(d for d, _ in samples) or 1
    return sum(d * bw for d, bw in samples) / total_dur


# ── GPU efficiency (reuses existing disassembly-based roofline) ───────────────

def duration_weighted_pct(pairs: list[tuple[int, float]]) -> float | None:
    """pairs: (duration_ns, pct in 0-100). Returns the duration-weighted mean
    as a 0-1 fraction, or None for empty input / zero total weight.

    Weighting must be by wall-clock duration, NOT by a rate like
    achieved_tflops -- weighting by a rate would let a short, high-throughput
    kernel dominate a long-running, lower-throughput one, backwards from what
    "duration-weighted" means. (An earlier version of gpu_efficiency did
    exactly that, plus used `x or 1.0` as a fallback, which treated a
    genuine 0.0 achieved_tflops kernel -- e.g. a pure memory-bound kernel --
    as if it had full weight, since 0.0 is falsy in Python.)
    """
    weighted, total = 0.0, 0.0
    for dur_ns, pct in pairs:
        w = max(dur_ns, 0)
        weighted += w * min(pct, 100.0)
        total += w
    if total <= 0:
        return None
    return (weighted / total) / 100.0


def gpu_efficiency(trace: "Trace") -> float | None:
    if not trace.disasm:
        return None
    try:
        from .roofline import analyze_trace
    except Exception:
        return None
    results = analyze_trace(trace)
    if not results:
        return None
    return duration_weighted_pct([(m.duration_ns, m.flops_pct) for _, m in results])


# ── Computational Scaling (approximate: IPC-ratio proxy, needs --baseline) ────

def computational_scaling(trace: "Trace", baseline: "Trace") -> float | None:
    def _mean_ipc(t: "Trace") -> float | None:
        vals = [c.value for c in t.counters if c.name == "ipc"]
        return sum(vals) / len(vals) if vals else None

    ipc_now = _mean_ipc(trace)
    ipc_base = _mean_ipc(baseline)
    if ipc_now is None or ipc_base is None or ipc_base <= 0:
        return None
    return min(1.0, ipc_now / ipc_base)


# ── top-level entry point ──────────────────────────────────────────────────────

def analyze(
    trace: "Trace",
    baseline: "Trace | None" = None,
    nccl_peak_bw_gbs: float | None = None,
    critical_path: "CriticalPathReport | None" = None,
) -> EfficiencyReport:
    report = EfficiencyReport()

    useful = useful_time_by_pid(trace)
    report.per_rank_useful_ns = useful
    wall = _wall_ns(trace)

    report.load_balance = load_balance(useful)
    report.comm_efficiency = comm_efficiency(useful, wall)
    if report.load_balance is not None and report.comm_efficiency is not None:
        report.parallel_efficiency = report.load_balance * report.comm_efficiency
    if len(useful) == 0:
        report.notes.append(
            "No spans with 'useful compute' categories (cpu/cuda/rocm/opencl/openmp) "
            "found in this trace -- Load Balance and Communication Efficiency omitted."
        )
    elif len(useful) == 1:
        report.notes.append(
            "Only one rank/process observed in this trace -- Load Balance and "
            "Communication Efficiency are not meaningful for a single-rank run."
        )

    per_cat_eff, overall_eff, transfer_notes = transfer_efficiency_by_category(trace)
    report.transfer_efficiency_by_category = per_cat_eff
    report.transfer_efficiency = overall_eff
    report.notes.extend(transfer_notes)

    if critical_path is not None:
        try:
            from .criticalpath import serialization_efficiency_from_path
            report.serialization_efficiency = serialization_efficiency_from_path(
                critical_path, _all_comm_spans(trace)
            )
        except Exception as e:
            report.notes.append(f"serialization_efficiency computation failed: {e}")
    else:
        report.notes.append(
            "serialization_efficiency requires a critical-path report "
            "(hprofiler critical-path) -- omitted."
        )

    if baseline is not None:
        report.computational_scaling = computational_scaling(trace, baseline)
        if report.computational_scaling is None:
            report.notes.append(
                "Baseline trace supplied but no 'ipc' counter samples found in "
                "one or both traces (requires the likwid or cpu backend) -- "
                "computational_scaling omitted."
            )
    else:
        report.notes.append(
            "computational_scaling requires a --baseline trace at a lower "
            "rank/thread count -- omitted."
        )

    if report.parallel_efficiency is not None:
        report.global_efficiency = report.parallel_efficiency * (report.computational_scaling or 1.0)

    report.gpu_efficiency = gpu_efficiency(trace)
    if report.gpu_efficiency is None and any(
        s.category.value in ("cuda", "rocm") for s in trace.spans
    ):
        report.notes.append(
            "gpu_efficiency requires disassembly data (record/view with --disasm) -- omitted."
        )

    report.nccl_bus_bw_gbs = nccl_bus_bandwidth_gbs(trace)
    if report.nccl_bus_bw_gbs is not None and nccl_peak_bw_gbs:
        report.nccl_efficiency = min(1.0, report.nccl_bus_bw_gbs / nccl_peak_bw_gbs)
    elif report.nccl_bus_bw_gbs is not None:
        report.notes.append(
            "nccl_bus_bw_gbs computed but no --interconnect-bw peak given -- "
            "nccl_efficiency (as a %) omitted."
        )

    return report
