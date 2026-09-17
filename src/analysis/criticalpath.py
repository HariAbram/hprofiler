"""
Live, N-way cross-runtime critical-path and blame attribution.

Generalizes CASITA (MPI+CUDA only) and HPCToolkit's blame-shifting (CPU+GPU
pairs only) to an arbitrary combination of hprofiler's backends (CUDA, ROCm,
OpenCL, OpenMP, MPI, NCCL) in one dependency graph built directly from a
single already-unified trace -- there is no separate per-runtime trace to
merge, because every hook in a run already reports to the same collector.

Scope (stated explicitly, matching CASITA/Score-P's own scoping): this models
*known structural synchronization semantics per programming model* --
same-stream/same-thread ordering, device syncs, MPI/NCCL point-to-point and
collective pairing (span_id/parent_span_id where available, SPMD call-order
matching otherwise), and OpenMP barrier rendezvous -- not arbitrary data-flow
dependencies. An edge means "the destination provably cannot proceed until
the source reaches the marked point", not "the destination reads data the
source wrote".

Single-node only: hprofiler's collector listens on an AF_UNIX socket, which
is only reachable by processes on the same node/filesystem, so today's traces
are inherently single-node regardless of clock synchronization -- there is no
multi-node case to guard against yet. If a future collector adds network
transport across nodes, this module would need an explicit per-node clock
offset correction step before it's safe to add cross-node edges (the same
class of problem Score-P/Vampir solve for multi-node traces); that is out of
scope here.

Collective/barrier pairing assumes SPMD-style loose synchronization (ranks
call the Nth collective of a given type in the same relative order, and one
round finishes before the next starts on most ranks) -- true for the common
GROMACS-style loop structure this was built for, but not guaranteed in
general.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.trace import Trace
    from ..core.events import SpanEvent

# Edge kinds and their "gate time" semantics:
#   "sequential" / "device_sync" / "explicit_span_id" -> predecessor must
#     have ENDED (gate = pred.end_ns): ordinary same-thread/same-stream/
#     Isend-Wait dependencies, where the successor genuinely cannot begin
#     until the predecessor call has returned.
#   "arrival" / "p2p" -> predecessor only has to have STARTED (gate =
#     pred.start_ns), evaluated against the successor's END, not its start:
#     used for anything where the successor's span covers a *wait*, so its
#     own start can legitimately precede the predecessor's start (e.g. an
#     MPI_Recv posted early that then blocks -- the "post an early receive"
#     pattern is common HPC practice specifically to overlap communication
#     setup with compute) and what actually explains its long duration is
#     when the predecessor showed up, not when it fully finished. Same
#     reasoning as collective/barrier rendezvous: you can't finish until
#     the other side has at least arrived, not until it's fully done.
_END_GATED = {"sequential", "device_sync", "explicit_span_id"}
_START_GATED = {"arrival", "p2p"}

_DEVICE_SYNC_NAMES = frozenset({"cudaDeviceSynchronize", "hipDeviceSynchronize", "cuCtxSynchronize"})
_GPU_CATS = frozenset({"cuda", "rocm"})
_MPI_COLLECTIVE_TYPES = frozenset({
    "allreduce", "bcast", "reduce", "alltoall", "allgather",
    "scatter", "gather", "barrier", "scan",
})
_NCCL_COLLECTIVE_TYPES = frozenset({
    "allreduce", "broadcast", "reduce", "allgather", "reduce_scatter",
})


@dataclass
class CriticalPathReport:
    path_span_indices: list[int]
    spans: list["SpanEvent"]              # same objects the indices refer into
    wall_ns: int
    time_on_path_by_category: dict[str, int] = field(default_factory=dict)
    wait_caused_by_category: dict[str, int] = field(default_factory=dict)
    total_path_ns: int = 0
    notes: list[str] = field(default_factory=list)

    def top_blame(self, n: int = 10) -> list[tuple[str, int]]:
        return sorted(self.wait_caused_by_category.items(), key=lambda kv: -kv[1])[:n]


# ── rendezvous clustering (shared by MPI/NCCL collectives and OMP barriers) ───

def _cluster_rendezvous(items: list[tuple[int, int, int]]) -> list[list[int]]:
    """items: (start_ns, end_ns, span_index). Group items into clusters whose
    intervals form a connected overlap chain -- each cluster is one
    "rendezvous" (one collective call / one barrier), returned as a list of
    span indices. Singleton clusters (no actual overlap with anything) are
    dropped -- nothing to pair them with."""
    if not items:
        return []
    ordered = sorted(items, key=lambda t: t[0])
    clusters: list[list[tuple[int, int, int]]] = []
    cur = [ordered[0]]
    cur_hi = ordered[0][1]
    for it in ordered[1:]:
        if it[0] <= cur_hi:
            cur.append(it)
            cur_hi = max(cur_hi, it[1])
        else:
            clusters.append(cur)
            cur = [it]
            cur_hi = it[1]
    clusters.append(cur)
    return [[t[2] for t in c] for c in clusters if len(c) > 1]


# ── edge builders ────────────────────────────────────────────────────────────

def _add_program_order_edges(preds: dict[int, list[tuple[int, str]]], spans: list["SpanEvent"]) -> None:
    by_thread: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, s in enumerate(spans):
        by_thread[(s.pid, s.tid)].append(i)
    for key, idxs in by_thread.items():
        idxs.sort(key=lambda i: spans[i].start_ns)
        for a, b in zip(idxs, idxs[1:]):
            preds[b].append((a, "sequential"))


def _add_stream_order_edges(preds: dict[int, list[tuple[int, str]]], spans: list["SpanEvent"]) -> None:
    by_stream: dict[tuple[int, str, str], list[int]] = defaultdict(list)
    for i, s in enumerate(spans):
        if s.category.value in _GPU_CATS or (s.category.value == "sync" and "stream" in s.tags):
            stream = s.tags.get("stream")
            if stream is not None:
                by_stream[(s.pid, s.category.value if s.category.value in _GPU_CATS else "gpu", stream)].append(i)
    # merge per-pid across the cuda/rocm/sync-with-stream split so a stream's
    # sync call chains after its own kernel/memcpy spans, not a separate list
    by_pid_stream: dict[tuple[int, str], list[int]] = defaultdict(list)
    for (pid, _cat, stream), idxs in by_stream.items():
        by_pid_stream[(pid, stream)].extend(idxs)
    for key, idxs in by_pid_stream.items():
        idxs = sorted(set(idxs), key=lambda i: spans[i].start_ns)
        for a, b in zip(idxs, idxs[1:]):
            preds[b].append((a, "sequential"))


def _add_device_sync_edges(preds: dict[int, list[tuple[int, str]]], spans: list["SpanEvent"]) -> None:
    by_pid_gpu: dict[int, list[int]] = defaultdict(list)
    by_pid_sync: dict[int, list[int]] = defaultdict(list)
    for i, s in enumerate(spans):
        if s.category.value in _GPU_CATS:
            by_pid_gpu[s.pid].append(i)
        elif s.category.value == "sync" and s.name in _DEVICE_SYNC_NAMES:
            by_pid_sync[s.pid].append(i)

    for pid, gpu_idxs in by_pid_gpu.items():
        gpu_idxs.sort(key=lambda i: spans[i].start_ns)
        sync_idxs = sorted(by_pid_sync.get(pid, []), key=lambda i: spans[i].start_ns)
        last_sync_end = -1
        gi = 0
        for si in sync_idxs:
            sync_start = spans[si].start_ns
            epoch: list[int] = []
            while gi < len(gpu_idxs) and spans[gpu_idxs[gi]].start_ns < sync_start:
                if spans[gpu_idxs[gi]].start_ns >= last_sync_end:
                    epoch.append(gpu_idxs[gi])
                gi += 1
            for g in epoch:
                preds[si].append((g, "device_sync"))
            last_sync_end = spans[si].end_ns


def _add_explicit_span_id_edges(preds: dict[int, list[tuple[int, str]]], spans: list["SpanEvent"]) -> None:
    """Any span whose parent_span_id matches another span's span_id gets a
    direct edge -- generalizes the existing Isend/Wait (sid=/psid=)
    correlation already used for CCT/call-tree arrows to critical-path use."""
    by_span_id: dict[str, int] = {}
    for i, s in enumerate(spans):
        if s.span_id:
            by_span_id[s.span_id] = i
    for i, s in enumerate(spans):
        if s.parent_span_id and s.parent_span_id in by_span_id:
            parent_idx = by_span_id[s.parent_span_id]
            if parent_idx != i:
                preds[i].append((parent_idx, "explicit_span_id"))


def _add_mpi_p2p_edges(preds: dict[int, list[tuple[int, str]]], spans: list["SpanEvent"]) -> None:
    """Pair the Nth send with the Nth recv for each (sender, receiver, tag)
    key, in each side's own chronological order -- NOT "the latest send
    that had already started by the time this recv started". MPI guarantees
    FIFO delivery for messages between the same ordered pair with the same
    tag, so Nth-with-Nth is always the semantically correct pairing
    regardless of which span happens to start first in wall-clock time; a
    "the send must already have started" precondition would silently miss
    the dependency entirely for the common "post an early receive, then
    block" pattern, where the recv's span starts well before the matching
    send's does.
    """
    sends: dict[tuple, list[int]] = defaultdict(list)
    recvs: dict[tuple, list[int]] = defaultdict(list)
    for i, s in enumerate(spans):
        if s.category.value != "mpi":
            continue
        t = s.tags.get("type", "")
        if t == "send" and "peer" in s.tags:
            key = (s.tags.get("rank"), s.tags.get("peer"), s.tags.get("tag"))
            sends[key].append(i)
        elif t == "recv" and "peer" in s.tags:
            # Recv's key is from the receiver's point of view; flip to match
            # the sender's (rank=sender, peer=receiver, tag) key.
            key = (s.tags.get("peer"), s.tags.get("rank"), s.tags.get("tag"))
            recvs[key].append(i)
    for key, recv_idxs in recvs.items():
        send_idxs = sorted(sends.get(key, []), key=lambda i: spans[i].start_ns)
        recv_idxs_sorted = sorted(recv_idxs, key=lambda i: spans[i].start_ns)
        for ri, si in zip(recv_idxs_sorted, send_idxs):
            preds[ri].append((si, "p2p"))


def _add_last_arriver_edges(
    preds: dict[int, list[tuple[int, str]]], spans: list["SpanEvent"], cluster: list[int]
) -> None:
    """Only the single latest-starting ("last arriver") participant in a
    rendezvous cluster is a valid 'arrival' predecessor for the others --
    everyone waits for that one specifically, not for each other pairwise.

    This is deliberately NOT a full mutual clique between all cluster
    members: a full clique would let the backward walk keep chaining
    through arrival edges after already reaching the last arriver (e.g.
    last-arriver -> 2nd-last-arriver -> 3rd-last-arriver -> ...), which
    doesn't correspond to anything real -- everyone else in the cluster
    already had arrived *before* the last one, so they can't be "blamed"
    for further delaying it. The last arriver's own predecessors (what
    happened on its thread/rank before it reached the rendezvous) come from
    ordinary structural edges (program order, stream order, ...), not from
    more arrival edges within this same cluster -- so the last arriver gets
    none here.
    """
    if len(cluster) < 2:
        return
    last = max(cluster, key=lambda i: spans[i].start_ns)
    for i in cluster:
        if i != last:
            preds[i].append((last, "arrival"))


def _add_rendezvous_edges(
    preds: dict[int, list[tuple[int, str]]],
    spans: list["SpanEvent"],
    category: str,
    collective_types: frozenset[str],
) -> None:
    by_type: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    for i, s in enumerate(spans):
        if s.category.value != category:
            continue
        t = s.tags.get("type", "")
        if t in collective_types and s.duration_ns > 0:
            by_type[t].append((s.start_ns, s.end_ns, i))

    for t, items in by_type.items():
        for cluster in _cluster_rendezvous(items):
            _add_last_arriver_edges(preds, spans, cluster)


def _add_omp_barrier_edges(preds: dict[int, list[tuple[int, str]]], spans: list["SpanEvent"]) -> None:
    by_pid_name: dict[tuple[int, str], list[tuple[int, int, int]]] = defaultdict(list)
    for i, s in enumerate(spans):
        if s.category.value == "sync" and s.name.startswith("omp_") and s.duration_ns > 0:
            by_pid_name[(s.pid, s.name)].append((s.start_ns, s.end_ns, i))
    for key, items in by_pid_name.items():
        for cluster in _cluster_rendezvous(items):
            _add_last_arriver_edges(preds, spans, cluster)


def build_dependency_graph(trace: "Trace") -> tuple[list["SpanEvent"], dict[int, list[tuple[int, str]]]]:
    spans = trace.spans
    preds: dict[int, list[tuple[int, str]]] = defaultdict(list)

    _add_program_order_edges(preds, spans)
    _add_stream_order_edges(preds, spans)
    _add_device_sync_edges(preds, spans)
    _add_explicit_span_id_edges(preds, spans)
    _add_mpi_p2p_edges(preds, spans)
    _add_rendezvous_edges(preds, spans, "mpi", _MPI_COLLECTIVE_TYPES)
    _add_rendezvous_edges(preds, spans, "nccl", _NCCL_COLLECTIVE_TYPES)
    _add_omp_barrier_edges(preds, spans)

    return spans, dict(preds)


# ── backward critical-path walk ────────────────────────────────────────────────

def compute_critical_path(spans: list["SpanEvent"], preds: dict[int, list[tuple[int, str]]]) -> list[int]:
    """Walk backward from the last-ending span, at each step picking the
    single predecessor that most tightly explains why the current node
    couldn't have progressed any earlier.

    Causality is enforced explicitly, and differently depending on edge
    kind, using the *current* node's own [start_ns, end_ns):
      - END_GATED (sequential/device_sync/p2p/explicit_span_id): a
        predecessor must have ENDED before the current node STARTED --
        it's asking "what determined when this node could begin".
      - START_GATED ("arrival", i.e. rendezvous clustering for OpenMP
        barriers and MPI/NCCL collectives): a predecessor only has to have
        STARTED sometime before the current node ENDED -- it's asking "who
        did this node have to wait for to *finish*", since a rendezvous
        span's own duration already includes however long it waited for
        the slowest arriver. Without this end-referenced check (as opposed
        to comparing against the current node's start), a same-cluster
        span that starts *after* the current node could be picked simply
        for having a numerically later gate time, walking the path
        backward in time -- which is exactly the failure mode this
        distinction prevents.
    """
    if not spans:
        return []
    end_idx = max(range(len(spans)), key=lambda i: spans[i].end_ns)
    path = [end_idx]
    visited = {end_idx}
    cur = end_idx
    while True:
        cur_start = spans[cur].start_ns
        cur_end = spans[cur].end_ns
        candidates = preds.get(cur, [])
        best_idx, best_gate = None, -1
        for p_idx, kind in candidates:
            if p_idx in visited:
                continue
            if kind in _END_GATED:
                gate, limit = spans[p_idx].end_ns, cur_start
            else:
                assert kind in _START_GATED
                gate, limit = spans[p_idx].start_ns, cur_end
            if gate <= limit and gate > best_gate:
                best_gate, best_idx = gate, p_idx
        if best_idx is None:
            break
        path.append(best_idx)
        visited.add(best_idx)
        cur = best_idx
    path.reverse()
    return path


def attribute_blame(spans: list["SpanEvent"], path: list[int], wall_ns: int) -> CriticalPathReport:
    time_on_path: dict[str, int] = defaultdict(int)
    wait_caused: dict[str, int] = defaultdict(int)
    total_path_ns = 0

    for i, idx in enumerate(path):
        s = spans[idx]
        time_on_path[s.category.value] += s.duration_ns
        total_path_ns += s.duration_ns
        if i > 0:
            prev = spans[path[i - 1]]
            gap = s.start_ns - prev.end_ns
            if gap > 0:
                wait_caused[prev.category.value] += gap
                total_path_ns += gap

    return CriticalPathReport(
        path_span_indices=path,
        spans=spans,
        wall_ns=wall_ns,
        time_on_path_by_category=dict(time_on_path),
        wait_caused_by_category=dict(wait_caused),
        total_path_ns=total_path_ns,
    )


def _wall_ns(spans: list["SpanEvent"]) -> int:
    # NOTE: deliberately NOT using trace.duration_ns here -- for a trace
    # reconstructed by load_trace_from_json, TraceMetadata.start_time_ns
    # defaults to the *load* time (dataclass field default_factory), not the
    # original run's start, making trace.duration_ns meaningless for a saved
    # trace. src/output/summary.py and src/analysis/cct.py already avoid
    # this the same way, by deriving wall time from the spans themselves.
    timed = [s for s in spans if s.duration_ns > 0]
    if not timed:
        return 1
    return max(1, max(s.end_ns for s in timed) - min(s.start_ns for s in timed))


def analyze(trace: "Trace") -> CriticalPathReport:
    spans, preds = build_dependency_graph(trace)
    path = compute_critical_path(spans, preds)
    wall_ns = _wall_ns(spans)
    report = attribute_blame(spans, path, wall_ns)
    n_pids = len({s.pid for s in spans})
    if n_pids == 0:
        report.notes.append("No spans in this trace -- nothing to analyze.")
    elif n_pids == 1:
        report.notes.append(
            "Only one process observed -- cross-rank/collective edges did not apply; "
            "this is an intra-process (thread/stream) critical path only."
        )
    if report.total_path_ns > wall_ns * 1.05:
        report.notes.append(
            "Time accounted for exceeds wall time: the path passes through "
            "'arrival'-gated rendezvous edges (OpenMP barriers / MPI-NCCL "
            "collectives), which by construction connect spans that overlap "
            "in real time across different threads/ranks -- each still "
            "contributes its own full duration to the category breakdown, "
            "so overlapping segments are counted once per thread/rank, not "
            "merged into one wall-clock interval."
        )
    return report


# ── bridge for pop_efficiency.py ───────────────────────────────────────────────

def _span_identity_key(s: "SpanEvent") -> tuple:
    """A value-based identity key for matching the "same" span across two
    independently-obtained span lists, e.g. two separate `trace.spans`
    property reads (which each return a new list but the same underlying
    SpanEvent objects -- fine for raw `id()` matching) or, more importantly,
    two Trace objects loaded from the same underlying trace file/data by
    different `load_trace_from_json` calls (DIFFERENT SpanEvent object
    instances entirely -- `id()` matching would then silently match nothing,
    making serialization_efficiency_from_path return a wrong ratio instead
    of raising -- exactly the kind of silent-wrong-result bug that matters
    here). Not perfectly collision-proof, but the odds of two genuinely
    different events sharing pid+tid+category+start+duration+name are
    negligible, and if they do collide they're interchangeable for this
    purpose anyway."""
    return (s.pid, s.tid, s.category.value, s.start_ns, s.duration_ns, s.name)


def serialization_efficiency_from_path(
    report: CriticalPathReport, comm_spans: list["SpanEvent"]
) -> float | None:
    """Fraction of total communication time that critical-path analysis shows
    was structurally unavoidable (the comm span is itself on the critical
    path) vs. total communication time observed in the trace. `comm_spans`
    should come from the same trace `report` was built from -- matched by a
    value-based identity key (see _span_identity_key), not raw object
    identity, so this is still correct even if `comm_spans` was obtained via
    a separate load of "the same" trace data."""
    if not comm_spans:
        return None
    comm_total_ns = sum(s.duration_ns for s in comm_spans)
    if comm_total_ns <= 0:
        return None
    on_path_keys = {_span_identity_key(report.spans[i]) for i in report.path_span_indices}
    comm_on_path_ns = sum(
        s.duration_ns for s in comm_spans if _span_identity_key(s) in on_path_keys
    )
    return min(1.0, comm_on_path_ns / comm_total_ns)
