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
collective pairing, and OpenMP barrier rendezvous -- not arbitrary data-flow
dependencies. An edge means "the destination provably cannot proceed until
the source reaches the marked point", not "the destination reads data the
source wrote".

Single-node only: hprofiler's collector listens on an AF_UNIX socket, which
is only reachable by processes on the same node/filesystem, so a *single*
trace is inherently single-node regardless of clock synchronization. A
multi-node run instead produces one trace file per node; `src/analysis/
multinode.py` merges several of these into one Trace (with an explicit
per-node clock-offset correction, error bounds included, not treated as
exact -- the same class of problem Score-P/Vampir solve for multi-node
traces) before handing it to this module, so `build_dependency_graph`/
`analyze` work unmodified either way -- they don't know or care whether the
Trace they were given came from one node or was merged from several; the
MPI `rank=`/`peer=`/`commid=` tags this module already matches on are
already globally unique across an entire job regardless of physical node
placement. See multinode.py and `hprofiler merge-nodes` for that step;
this module's own scope is unchanged by it.

Collective/barrier pairing assumes SPMD-style loose synchronization (ranks
call the Nth collective of a given type in the same relative order, and one
round finishes before the next starts on most ranks) -- true for the common
GROMACS-style loop structure this was built for, but not guaranteed in
general. Since hooks/mpi_hook/mpi_hook.c gained real communicator identity
(commid=), collectives are now clustered per-(type, commid) whenever
possible instead of per-type-only, which removes one source of false
pairing (two unrelated communicators doing the same collective type at
overlapping times); see EDGE_CONFIDENCE and _add_rendezvous_edges.

── Edge confidence classes ─────────────────────────────────────────────────
Every edge records not just *that* a dependency exists but how directly the
underlying data proves it -- this module's response to the critique that its
matching was "a heuristic" with no way to tell a hardware-enforced ordering
from a best-effort guess:
  "certain" -- enforced by the runtime/hardware itself (same-thread program
    order; CUDA/HIP stream & event semantics), or by an explicit id the hook
    itself assigned and later referenced (sid=/psid=, including through
    MPI_Waitall/Waitsome's multi-id psid= lists).
  "high"    -- MPI point-to-point or collective matching backed by resolved
    MPI_Status data (a wildcard MPI_ANY_SOURCE/ANY_TAG match resolved to its
    real peer/tag -- see mpi_hook.c's file header) and/or real communicator
    identity (commid= from MPI_Comm_dup/split/create, not the -1
    "unregistered" sentinel).
  "medium"  -- the same kind of matching without that extra evidence: exact
    (non-wildcard) tag matching by call order only, or rendezvous clustering
    when no commid was available to scope it.
CriticalPathReport surfaces a confidence breakdown for the reported path so
a caller can see how much of it rests on strong vs. weaker evidence, not
just accept a single number silently built on a mix of both.

── Formal critical path (DAG longest-path DP) ──────────────────────────────
compute_critical_path runs a dynamic program over the dependency DAG:
`accounted[v] = max over causally-valid (u,v) of accounted[u] + gap + dur(v)`,
computed in topological order, then reconstructed by tracing the argmax
choices back from the node achieving the global maximum. This is provably
optimal for "which chain of observed dependencies accounts for the most
wall-clock time" (longest path in a DAG is solvable exactly in O(V+E) time,
unlike general-graph longest path, which is NP-hard); a greedy backward
walk picking the locally-tightest predecessor at each step is not
guaranteed to reach the node the DP proves globally best when two
predecessors compete. The DAG is acyclic by construction (every edge builder
only points from an earlier-enabling event to a later-gated one), but the
DP still verifies this via topological sort and, if a cycle is ever
detected, falls back to that greedy walk (cycle-safe via its `visited` set)
with a report note -- defensive, not silently assumed.
"""

from __future__ import annotations

import os
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..core import gpu_activity as ga
from .activity_buckets import is_annotation
from .span_links import SpanIdResolver, id_family

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
#   "launch" -> host submission call -> the device work it submitted
#     (matched by correlation id, core/gpu_activity.correlate). The device
#     work may start before the call returns, so the gate is the call's
#     START; what the successor adds to the path is its end minus the
#     call's end (queueing delay + execution, or only the tail beyond the
#     call when they overlap).
#   "device_wait" -> device work -> the host synchronization call that
#     waited for it (stream/device/event sync). The call starts before the
#     work finishes -- that is the wait -- so the gate is the work's END
#     against the call's END, and the call adds only its tail after the
#     work completed.
_OVERLAP_KINDS = {"launch", "device_wait"}

# Per-edge-kind confidence is the *default*; p2p/arrival edges override this
# per-instance (see _add_mpi_p2p_edges / _add_rendezvous_edges) based on
# whether resolved-status / real commid= evidence backed that specific edge.
_DEFAULT_CONFIDENCE = {
    "sequential":       "certain",
    "device_sync":      "certain",
    "explicit_span_id": "certain",
    "launch":           "certain",
    "device_wait":      "certain",
    "p2p":              "medium",
    "arrival":          "medium",
}
_CONFIDENCE_RANK = {"certain": 3, "high": 2, "medium": 1, "low": 0}

_DEVICE_SYNC_NAMES = frozenset({"cudaDeviceSynchronize", "hipDeviceSynchronize", "cuCtxSynchronize"})
_GPU_CATS = frozenset({"cuda", "rocm"})
_MPI_COLLECTIVE_TYPES = frozenset({
    "allreduce", "bcast", "reduce", "alltoall", "allgather",
    "scatter", "gather", "barrier", "scan", "exscan",
    "allgatherv", "alltoallv", "gatherv", "scatterv", "reduce_scatter", "reduce_scatter_block",
})
_NCCL_COLLECTIVE_TYPES = frozenset({
    "allreduce", "broadcast", "reduce", "allgather", "reduce_scatter",
    "alltoall",
})
# MPI_Test/MPI_Testany/MPI_Testsome/MPI_Testall/MPI_Cancel are emitted as
# *instant* events (mpi_hook.c) since they're meant to be non-blocking
# polls, not durations -- trace.spans (what this whole module walks)
# excludes InstantEvents by construction, so they naturally don't appear as
# graph nodes and need no special exclusion here. One consequence, stated
# explicitly rather than left as a silent gap: a non-blocking receive
# completed exclusively via a Test*/poll loop (never Wait/Waitall/Waitany/
# Waitsome) gets no cross-rank p2p edge in this version -- see
# _index_mpi_completers and DOCUMENTATION.md's Known Limitations.
_MPI_WAIT_NAMES = frozenset({"MPI_Wait", "MPI_Waitall", "MPI_Waitany", "MPI_Waitsome"})


@dataclass
class CriticalPathReport:
    path_span_indices: list[int]
    spans: list["SpanEvent"]              # same objects the indices refer into
    wall_ns: int
    time_on_path_by_category: dict[str, int] = field(default_factory=dict)
    wait_caused_by_category: dict[str, int] = field(default_factory=dict)
    total_path_ns: int = 0
    notes: list[str] = field(default_factory=list)
    # Confidence tier of each *edge actually used* by the reported path --
    # one entry per hop, so len == max(0, len(path_span_indices) - 1). The
    # first span on the path has no incoming edge (it's the root of the
    # backward walk) and so contributes no entry here.
    path_edge_confidence: list[str] = field(default_factory=list)
    # Edge kind of each hop (same indexing as path_edge_confidence), and the
    # time each path span adds beyond its predecessor -- its full duration
    # for ordinary edges, only the non-overlapping tail for launch/
    # device_wait hops. Empty for reports built without kinds.
    path_edge_kinds: list[str] = field(default_factory=list)
    path_credited_ns: list[int] = field(default_factory=list)
    # Idle time before each path span (0 for the first): sum(path_gap_ns) +
    # sum(path_credited_ns) == total_path_ns. See attribute_blame.
    path_gap_ns: list[int] = field(default_factory=list)
    # The longest-path DP's own value for this path (accounted[] of its last
    # span): what the DP maximizes, >= total_path_ns -- start-gated hops
    # (rendezvous, p2p) let a successor overlap its predecessor in wall
    # time, and the DP counts both. 0 for reports built without edge kinds.
    dp_accounted_ns: int = 0
    # Path time (credited + idle before it) inside each annotation range
    # (NVTX/ROCTX, by name) open on a path span's thread when it started:
    # display only -- ranges are never path nodes and carry no path time of
    # their own (annotation_time_on_path). Nested ranges each count it.
    path_time_by_annotation: dict[str, int] = field(default_factory=dict)
    # Known holes in the graph (mpi_dependency_diagnostics): receives with
    # no point-to-point edge, Test*-only and orphan completions, ...
    diagnostics: dict = field(default_factory=dict)

    def top_blame(self, n: int = 10) -> list[tuple[str, int]]:
        return sorted(self.wait_caused_by_category.items(), key=lambda kv: -kv[1])[:n]

    def confidence_breakdown_ns(self) -> dict[str, int]:
        """Wall-clock ns on the path attributable to each edge confidence
        tier -- each span's duration is credited to the tier of the edge
        that explains *that span's own* inclusion on the path (the edge
        pointing INTO it from its chosen predecessor), so the totals sum to
        total_path_ns minus the root span's own duration (path[0] has no
        such edge -- nothing explains its inclusion, it's the start of the
        backward walk). path_edge_confidence[i] is the confidence of the
        edge from path[i] to path[i+1], so it's path[i+1] -- not path[i] --
        whose duration it's crediting. Lets a caller see how much of the
        reported path rests on strong vs. weaker evidence, instead of a
        single number silently mixing both."""
        out: dict[str, int] = defaultdict(int)
        credited = self.path_credited_ns
        for i, conf in enumerate(self.path_edge_confidence):
            span_idx = self.path_span_indices[i + 1]
            out[conf] += credited[i + 1] if len(credited) == len(self.path_span_indices) \
                else self.spans[span_idx].duration_ns
        return dict(out)


# ── small parsing helpers for semicolon/slash-delimited MPI tag values ────────

def _items(spans):
    """(index, span) pairs of a full span list, or of an ordinal-keyed
    subset dict (the scalable pipeline hands each builder only the spans
    it needs; keys are the span's ordinal in the whole trace)."""
    return spans.items() if isinstance(spans, dict) else enumerate(spans)


def _parse_multi(value: str) -> list[str]:
    return [v for v in value.split(";") if v]


def _parse_rmatches(value: str) -> dict[str, tuple[str, str]]:
    """rmatches=<req_id>/<peer>/<tag>;... (mpi_hook.c's MPI_Waitall/
    MPI_Waitsome) -> {req_id: (peer, tag)}."""
    out: dict[str, tuple[str, str]] = {}
    for entry in _parse_multi(value):
        parts = entry.split("/")
        if len(parts) == 3:
            out[parts[0]] = (parts[1], parts[2])
    return out


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
# preds[v] is a list of (predecessor_index, kind, confidence) triples.

def _add_program_order_edges(preds: dict[int, list[tuple[int, str, str]]], spans: list["SpanEvent"]) -> None:
    by_thread: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, s in _items(spans):
        # CUDA/ROCm device spans (side=gpu) ran on a stream, not on the
        # thread whose id they carry; launch edges connect them instead.
        # Annotation ranges (NVTX/ROCTX) are labels, not work: a closed
        # top-level range would become the predecessor of everything after
        # it, and its duration -- all of its children's -- would make the
        # chain of enclosing ranges the longest path.
        if ga.is_device_span(s) or is_annotation(s):
            continue
        by_thread[(s.pid, s.tid)].append(i)
    # Predecessor = the most recently COMPLETED span on the thread (latest
    # end <= b.start; ties -> the outermost). Linking by start time instead
    # would break the chain at every nested span: a barrier inside a
    # parallel region / implicit task has the ENCLOSING span as its
    # start-order predecessor, whose end gate it can never satisfy, so the
    # edge would be dropped and nothing after it would connect back (on a
    # continuously busy OpenMP program, only ~28% of wall time explained).
    import heapq
    for key, idxs in by_thread.items():
        idxs.sort(key=lambda i: spans[i].start_ns)
        pending: list[tuple[int, int, int]] = []   # (end, start, idx) not yet completed
        best: tuple[int, int, int] | None = None   # (end, -start, idx) latest completed
        for b in idxs:
            b_start = spans[b].start_ns
            while pending and pending[0][0] <= b_start:
                end, start, i = heapq.heappop(pending)
                cand = (end, -start, i)
                if best is None or cand > best:
                    best = cand
            if best is not None:
                preds[b].append((best[2], "sequential", "certain"))
            heapq.heappush(pending, (spans[b].end_ns, b_start, b))


def _add_stream_order_edges(preds: dict[int, list[tuple[int, str, str]]], spans: list["SpanEvent"]) -> None:
    # Pre-split traces only (no rt= tag); _add_gpu_edges handles the
    # host/device model.
    by_stream: dict[tuple[int, str, str], list[int]] = defaultdict(list)
    for i, s in _items(spans):
        if ga.is_new_format(s):
            continue
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
            preds[b].append((a, "sequential", "certain"))


def _add_device_sync_edges(preds: dict[int, list[tuple[int, str, str]]], spans: list["SpanEvent"]) -> None:
    by_pid_gpu: dict[int, list[int]] = defaultdict(list)
    by_pid_sync: dict[int, list[int]] = defaultdict(list)
    for i, s in _items(spans):
        if ga.is_new_format(s):   # pre-split traces only, see _add_gpu_edges
            continue
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
                preds[si].append((g, "device_sync", "certain"))
            last_sync_end = spans[si].end_ns


def _sync_kind(s: "SpanEvent") -> str | None:
    kind = s.tags.get("sync")
    if kind:
        return kind
    n = s.name
    if "DeviceSynchronize" in n or "CtxSynchronize" in n:
        return "device"
    if "StreamSynchronize" in n:
        return "stream"
    if "EventSynchronize" in n:
        return "event"
    return None


def _add_gpu_edges(preds: dict[int, list[tuple[int, str, str]]], spans: list["SpanEvent"],
                   lossy: "set[int] | str | None" = None) -> None:
    """CUDA/ROCm host/device model (spans carrying rt=, see
    core/gpu_activity.py):

      launch       host submission -> its device work, by correlation id
      sequential   device work -> next device work on the same stream, in
                   submission (= FIFO execution) order
      device_wait  device work -> the sync call that waited for it: the last
                   work submitted to the synced stream (stream sync), to each
                   stream (device/context sync), or to the event's stream
                   before the cudaEventRecord/hipEventRecord (event sync)
      sequential   cross-stream: last work before an event record -> first
                   work submitted to the waiting stream after
                   cudaStreamWaitEvent/hipStreamWaitEvent

    Event syncs whose record call was never seen fall back to the
    device-wide rule at "medium" confidence (it might have waited for less).

    Streams are keyed by device too: the default (NULL) stream exists once
    per GPU, so two devices' default-stream work is not one stream. The
    device is the host call's hdev= (the hooks' current device), else the
    native record's dev= (mapped to hdev= where correlated launches show
    the mapping); a device-wide sync waits for its own device's streams,
    and only at "medium" when its device is unknown and the process used
    several. Kernels of one graph launch are not chained to each other (the
    graph may run them concurrently and its edges are not recorded): the
    stream orders them after the previous work and before the next.

    In a process that lost CUDA/ROCm events (`lossy`, event_loss_pids) the
    event-record lookups (event sync, stream wait) are "low": a lost
    cudaEventRecord makes the latest record seen an older one.
    """
    if not any(ga.is_new_format(s) for _i, s in _items(spans)):
        return
    import bisect

    corr = ga.correlate(spans)
    for d, h in corr.host_of.items():
        preds[d].append((h, "launch", corr.confidence[d]))

    # native device id -> host-side device ordinal, learned from launches
    native_to_host: dict[tuple, set[str]] = defaultdict(set)
    for d, h in corr.host_of.items():
        nd, hd = spans[d].tags.get("dev"), spans[h].tags.get("hdev")
        if nd is not None and hd is not None:
            native_to_host[(spans[d].pid, spans[d].tags["rt"], str(nd))].add(str(hd))

    def device_of(i: int) -> str | None:
        s = spans[i]
        hd = s.tags.get("hdev")
        if hd is None and i in corr.host_of:
            hd = spans[corr.host_of[i]].tags.get("hdev")
        if hd is not None:
            return "h" + str(hd)
        nd = s.tags.get("dev")
        if nd is None:
            return None
        mapped = native_to_host.get((s.pid, s.tags["rt"], str(nd)))
        return "h" + next(iter(mapped)) if mapped and len(mapped) == 1 else "n" + str(nd)

    work = [i for i, s in _items(spans)
            if ga.is_device_span(s) and ga.op_of(s) in ga.DEVICE_WORK_OPS]
    sub = {i: ga.submission_ns(spans, corr, i) for i in work}
    by_stream: dict[tuple, list[int]] = defaultdict(list)
    streams_of: dict[tuple, list[tuple]] = defaultdict(list)
    for i in work:
        st = spans[i].tags.get("stream")
        if st is None:
            continue
        key = (spans[i].pid, spans[i].tags["rt"], device_of(i), st)
        if key not in by_stream:
            streams_of[key[:2]].append(key)
        by_stream[key].append(i)

    def graph_of(i: int):
        """The graph launch a device span belongs to (its host submission),
        or None."""
        h = corr.host_of.get(i)
        return h if h is not None and ga.op_of(spans[h]) == "graph" and ga.op_of(spans[i]) != "graph" \
            else None

    sub_sorted: dict[tuple, list[int]] = {}
    for key, lst in by_stream.items():
        lst.sort(key=lambda i: (sub[i], _effective_start_ns(spans[i])))
        sub_sorted[key] = [sub[i] for i in lst]
        # consecutive work, one graph launch's kernels forming one group
        groups: list[list[int]] = []
        for i in lst:
            g = graph_of(i)
            if groups and g is not None and graph_of(groups[-1][0]) == g:
                groups[-1].append(i)
            else:
                groups.append([i])
        for ga_, gb in zip(groups, groups[1:]):
            # the stream waits for the whole previous group: its last to finish
            a = max(ga_, key=lambda j: _effective_end_ns(spans[j])) if len(ga_) > 1 else ga_[0]
            for b in gb:
                preds[b].append((a, "sequential", "certain"))

    def stream_keys(pid: int, rt: str, dev: str | None, st) -> list[tuple[tuple, str]]:
        """The stream keys a host call naming stream `st` on device `dev`
        refers to, with the confidence that it is that one."""
        if st is None:
            return []
        if dev is not None and (pid, rt, dev, st) in by_stream:
            return [((pid, rt, dev, st), "certain")]
        cands = [k for k in streams_of.get((pid, rt), ()) if k[3] == st]
        return [(k, "certain" if len(cands) == 1 else "medium") for k in cands]

    def last_before(key: tuple, cutoff: int, end_limit: int) -> int | None:
        """Last work on `key` submitted strictly before `cutoff` that had
        finished by `end_limit` (what a wait started at cutoff could have
        waited for)."""
        lst = by_stream.get(key)
        if not lst:
            return None
        pos = bisect.bisect_left(sub_sorted[key], cutoff) - 1
        for _ in range(64):   # in-order streams: the first hit is the answer
            if pos < 0:
                return None
            if _effective_end_ns(spans[lst[pos]]) <= end_limit:
                return lst[pos]
            pos -= 1
        return None

    records: dict[tuple, list[int]] = defaultdict(list)
    for i, s in _items(spans):
        if s.tags.get("side") == "cpu" and ga.is_new_format(s) and ga.op_of(s) == "event_record" \
                and s.tags.get("event"):
            records[(s.pid, s.tags["rt"], s.tags["event"])].append(i)
    rec_starts: dict[tuple, list[int]] = {}
    for key, lst in records.items():
        lst.sort(key=lambda i: spans[i].start_ns)
        rec_starts[key] = [spans[i].start_ns for i in lst]

    def record_before(pid: int, rt: str, event: str | None, t: int) -> int | None:
        if not event:
            return None
        key = (pid, rt, event)
        pos = bisect.bisect_left(rec_starts.get(key, []), t) - 1
        return records[key][pos] if pos >= 0 else None

    def host_dev(j: int) -> str | None:
        hd = spans[j].tags.get("hdev")
        return None if hd is None else "h" + str(hd)

    def weaker(a: str, b: str) -> str:
        return a if _CONFIDENCE_RANK[a] <= _CONFIDENCE_RANK[b] else b

    for i, s in _items(spans):
        if s.tags.get("side") != "cpu" or not ga.is_new_format(s):
            continue
        op = ga.op_of(s)
        pid, rt = s.pid, s.tags["rt"]
        if op == "sync":
            kind = _sync_kind(s)
            end_limit = s.end_ns + ga.CLOCK_TOLERANCE_NS
            targets: list[tuple[int, str]] = []
            if kind == "stream":
                st = s.tags.get("stream")
                if st is None:   # e.g. a CUPTI-callback span: use its sync record
                    st = next((spans[d].tags.get("stream") for d in corr.devices_of.get(i, ())
                               if ga.op_of(spans[d]) == "sync" and spans[d].tags.get("stream")), None)
                for key, conf in stream_keys(pid, rt, host_dev(i), st):
                    j = last_before(key, s.start_ns, end_limit)
                    if j is not None:
                        targets.append((j, conf))
            elif kind == "event":
                rec = record_before(pid, rt, s.tags.get("event"), s.start_ns)
                if rec is not None:
                    for key, conf in stream_keys(pid, rt, host_dev(rec), spans[rec].tags.get("stream")):
                        j = last_before(key, spans[rec].start_ns, end_limit)
                        if j is not None:
                            targets.append((j, "low" if _lossy(lossy, pid) else conf))
                else:
                    for key in streams_of.get((pid, rt), ()):
                        j = last_before(key, s.start_ns, end_limit)
                        if j is not None:
                            targets.append((j, "medium"))
            elif kind == "device":
                keys = streams_of.get((pid, rt), ())
                devices = {k[2] for k in keys}
                dev = host_dev(i)
                for key in keys:
                    if len(devices) <= 1:
                        conf = "certain"              # one device: it is this one
                    elif dev is not None and key[2] == dev:
                        conf = "certain"
                    elif dev is not None and key[2] is not None and key[2].startswith("h"):
                        continue                      # another device's stream
                    else:
                        conf = "medium"               # which device was synced is unknown
                    j = last_before(key, s.start_ns, end_limit)
                    if j is not None:
                        targets.append((j, conf))
            for j, conf in targets:
                preds[i].append((j, "device_wait", conf))
        elif op == "stream_wait":
            rec = record_before(pid, rt, s.tags.get("event"), s.start_ns)
            waiting = s.tags.get("stream")
            if rec is None or waiting is None or spans[rec].tags.get("stream") is None:
                continue
            pkeys = stream_keys(pid, rt, host_dev(rec), spans[rec].tags["stream"])
            wkeys = stream_keys(pid, rt, host_dev(i), waiting)
            if len(pkeys) != 1 or len(wkeys) != 1:
                continue
            (pkey, pconf), (wkey, wconf) = pkeys[0], wkeys[0]
            producer = last_before(pkey, spans[rec].start_ns, 1 << 62)
            pos = bisect.bisect_left(sub_sorted.get(wkey, []), s.start_ns)
            if producer is not None and pos < len(by_stream.get(wkey, ())):
                consumer = by_stream[wkey][pos]
                if consumer != producer:
                    conf = "low" if _lossy(lossy, pid) else weaker(pconf, wconf)
                    preds[consumer].append((producer, "sequential", conf))


class _SpanIdIndex:
    """span_id -> ordinal. Hooks number requests and regions per process
    (MPI request ids and OpenMP parallel-region ids restart in every rank in
    traces from older hooks), so a parent id is looked up in the child's own
    process first (span_links.SpanIdResolver: when two hooks gave one id to
    two spans -- MPI requests and CUDA host calls did, before hp_ids.h --
    the child's own hook family decides, never "the last span wins"); an
    id from another process is used only when exactly one process carries
    it (hand-built traces)."""

    _AMBIGUOUS = object()

    def __init__(self) -> None:
        self.by_pid = SpanIdResolver()
        self.by_id: dict[str, object] = {}       # span_id -> (pid, ordinal) | _AMBIGUOUS

    def add(self, s: "SpanEvent", i: int) -> None:
        self.by_pid.add_span(s, i)
        entry = self.by_id.get(s.span_id)
        if entry is None or (entry is not self._AMBIGUOUS and entry[0] == s.pid):
            self.by_id[s.span_id] = (s.pid, i)
        else:
            self.by_id[s.span_id] = self._AMBIGUOUS

    def get(self, pid: int, span_id: str, start_ns: int, family: str) -> int | None:
        if (pid, span_id) in self.by_pid._by:
            return self.by_pid.resolve(pid, span_id, start_ns, family)
        entry = self.by_id.get(span_id)
        return None if entry is None or entry is self._AMBIGUOUS else entry[1]


def _add_explicit_span_id_edges(preds: dict[int, list[tuple[int, str, str]]], spans: list["SpanEvent"]) -> None:
    """Any span whose parent_span_id references another span's span_id gets
    a direct edge -- generalizes the existing Isend/Wait (sid=/psid=)
    correlation already used for CCT/call-tree arrows to critical-path use.

    parent_span_id can itself be a ';'-separated list of ids (mpi_hook.c's
    MPI_Waitall/MPI_Waitsome tag several requests' ids in one psid= when
    several complete in the same call) -- split and link each one
    individually. Before this, only a single exact-string match was tried,
    so a Waitall/Waitsome span's psid="4;6;5" never matched anything (no
    span has span_id "4;6;5" -- the individual ids "4", "6", "5" do) and
    those calls were silently never linked to the Isend/Irecv spans that
    created the requests they waited on.
    """
    ids = _SpanIdIndex()
    for i, s in _items(spans):
        # An annotation range is the psid= of the calls inside it (their
        # calling context), never a dependency.
        if s.span_id and not is_annotation(s):
            ids.add(s, i)
    for i, s in _items(spans):
        # A CUDA/ROCm device span's parent is its host submission, already
        # linked by a "launch" edge with the right (start-gated) semantics.
        if not s.parent_span_id or ga.is_device_span(s) or is_annotation(s):
            continue
        family = id_family(s)
        for req_id in _parse_multi(s.parent_span_id):
            parent_idx = ids.get(s.pid, req_id, s.start_ns, family)
            if parent_idx is not None and parent_idx != i:
                preds[i].append((parent_idx, "explicit_span_id", "certain"))


# ── MPI point-to-point matching ──────────────────────────────────────────────
#
# Keys. MPI pairs a message with a receive by (communicator, sender,
# receiver, tag) in posting order (non-overtaking). The hook reports rank=
# (the caller's MPI_COMM_WORLD rank), peer= (the other side as a rank of the
# call's communicator), wpeer= (that rank translated to MPI_COMM_WORLD) and
# commid= (a communicator id agreed by its members). Messages are matched on
#     send:    (rank,  wpeer, tag, commid)
#     receive: (wpeer, rank,  tag, commid)      -- wpeer = resolved source
# so sub-communicators whose ranks are renumbered pair correctly. Traces
# recorded before wpeer= existed fall back to the old key (rank, peer, tag)
# -- correct on MPI_COMM_WORLD only -- with a one-time warning and a
# `legacy_key` diagnostic.
#
# Cancelled requests (cancelled=1 on a single completion, cpsid= on a
# multi-completion) neither send nor receive: they are dropped before
# pairing, so they create no edge and do not shift later pairs.

_P2P_SEND_TYPES = frozenset({"send", "isend", "ssend", "bsend", "rsend", "issend", "ibsend", "irsend"})
_legacy_key_warned = False


def _warn_legacy_key_once() -> None:
    global _legacy_key_warned
    if _legacy_key_warned:
        return
    _legacy_key_warned = True
    import sys
    print("[hprofiler][warn] this trace has no wpeer= tags (recorded by an older MPI hook): "
          "point-to-point messages are matched on (sender world rank, receiver-side peer, tag) "
          "without the communicator, which is correct on MPI_COMM_WORLD only -- messages on "
          "sub-communicators with renumbered ranks are not paired", file=sys.stderr)


def _parse_rwmatches(value: str) -> dict[str, str]:
    """rwmatches=<req_id>/<world source>;... -> {req_id: world source}."""
    out: dict[str, str] = {}
    for entry in _parse_multi(value):
        parts = entry.split("/")
        if len(parts) == 2:
            out[parts[0]] = parts[1]
    return out


ALL_PROCESSES = "all"

# ── Exact Nth-with-Nth matching (opt-in, HPROFILER_EXACT_P2P_HIGH=1) ─────────
# MPI's non-overtaking rule pairs the k-th send with the k-th receive of a
# (sender, receiver, tag, communicator) channel exactly -- when the capture of
# that channel is complete. Under the flag an exact (non-wildcard) match is
# "high" instead of "medium" only when all of these hold:
#   * the trace is keyed by world ranks and communicator ids (wpeer=);
#   * neither process lost MPI events (event_loss_pids: those are "low");
#   * both processes' hooks declare a complete point-to-point capture
#     (counter mpi_p2p_capture_complete = 1: every send/receive call of the
#     runtime's MPI version is wrapped or marked -- traces of older hooks never
#     qualify);
#   * the channel has as many sends as receives;
#   * the communicator had no cancelled send, and in neither process any
#     matched probe (MPI_Mprobe/Improbe: their messages are not paired) or
#     persistent request (*_init: not paired), and it is a registered
#     communicator (commid != -1, which can mix several).
EXACT_P2P_ENV = "HPROFILER_EXACT_P2P_HIGH"
_PERSISTENT_TYPES = frozenset({"send_init", "bsend_init", "ssend_init", "rsend_init", "recv_init"})


def exact_p2p_enabled() -> bool:
    return os.environ.get(EXACT_P2P_ENV, "") not in ("", "0")


def _exact_capture_facts(trace: "Trace") -> dict:
    """{"complete": {pid}, "tainted": {(pid, commid)}}: the hooks' capture
    counters (mpi_p2p_capture_complete) and matched-probe instants;
    persistent requests are added by the matcher from the spans."""
    complete: set[int] = set()
    tainted: set[tuple[int, str]] = set()
    for c in trace.iter_counters():
        if c.name == "mpi_p2p_capture_complete" and c.value == 1:
            complete.add(c.pid)
    for i in trace.iter_instants():
        if i.category.value == "mpi" and i.name in ("MPI_Mprobe", "MPI_Improbe"):
            tainted.add((i.pid, str(i.tags.get("commid", "*"))))
    return {"complete": complete, "tainted": tainted}


def event_loss_pids(trace: "Trace", hooks: tuple[str, ...]) -> "set[int] | str":
    """Processes whose `hooks` lost events in this capture (dropped or lost
    records, oversize/unformattable records, a connection that ended
    without its final drain, records sent but not received), as a set of
    pids -- or ALL_PROCESSES when a loss cannot be tied to a process of the
    trace (unreadable records, a connection without a status, or a pid
    not in the trace, e.g. renumbered by merge-nodes).

    Order-based pairing (the k-th send with the k-th receive of a channel,
    an event sync with the latest record of its event) shifts after one
    lost call: those edges are downgraded to "low" in such processes."""
    health = getattr(trace.metadata, "capture_health", None) or {}
    rec = health.get("receiver") or {}
    if any(rec.get(k) for k in ("malformed", "errors", "partial_tails", "connections_without_status")):
        return ALL_PROCESSES
    lossy: set[int] = set()
    for key, t in (health.get("transport") or {}).items():
        pid_s, _, hook = key.partition("/")
        if hook.split("#")[0] not in hooks:
            continue
        lost = (any(t.get(k) for k in ("dropped_full", "dropped_lost", "oversize", "format_errors"))
                or t.get("ended_without_final")
                or (t.get("final") == 1 and "sent" in t and t.get("received", 0) < t["sent"]))
        if lost:
            try:
                lossy.add(int(pid_s))
            except ValueError:
                return ALL_PROCESSES
    if lossy and not lossy <= set(trace.store.pids()):
        return ALL_PROCESSES
    return lossy


def _lossy(lossy, *pids: int) -> bool:
    return bool(lossy) and (lossy == ALL_PROCESSES or any(p in lossy for p in pids))


class _P2PMatcher:
    """Collects MPI spans in ordinal order (add) and yields the point-to-point
    edges (edges) plus what could not be paired (diagnostics). Shared by
    build_dependency_graph and the streamed build_edges, so both produce the
    same edges in the same order. `lossy` (event_loss_pids): processes whose
    pairings are order-based guesses after lost events -- "low"."""

    def __init__(self, lossy: "set[int] | str | None" = None, exact: dict | None = None) -> None:
        self.lossy = lossy or set()
        self.exact = exact                 # _exact_capture_facts(), or None: rule off
        self.pid_of: dict[int, int] = {}
        # (ordinal, start, span_id, rank, peer, wpeer, tag, commid)
        self.sends: list[tuple] = []
        # (ordinal, start, conf, rank, peer, wpeer, tag, commid)  blocking receives
        self.blocking: list[tuple] = []
        # ((pid, span_id), ordinal, start, wildcard, rank, peer, wpeer, tag, commid)
        self.irecvs: list[tuple] = []
        # Request ids are unique per process only: everything is keyed (pid, id).
        # (pid, req_id) -> (completer ordinal, rpeer, rtag, rwpeer)
        self.completers: dict[tuple[int, str], tuple[int, str | None, str | None, str | None]] = {}
        self.cancelled: set[tuple[int, str]] = set()
        self.world_info = False
        self.diag: dict = {}

    def add(self, i: int, s: "SpanEvent") -> None:
        tags = s.tags
        t = tags.get("type", "")
        if self.lossy or self.exact is not None:
            self.pid_of[i] = s.pid
        if self.exact is not None and t in _PERSISTENT_TYPES:
            self.exact["tainted"].add((s.pid, str(tags.get("commid", "*"))))
        if "wpeer" in tags or "rwpeer" in tags or "rwmatches" in tags:
            self.world_info = True
        if t in _P2P_SEND_TYPES and "peer" in tags and tags.get("null") != "1":
            self.sends.append((i, s.start_ns, (s.pid, s.span_id) if s.span_id else None,
                               tags.get("rank"), tags.get("peer"),
                               tags.get("wpeer"), tags.get("tag"), tags.get("commid")))
        elif t == "recv" and "peer" in tags and tags.get("null") != "1":
            self.blocking.append((i, s.start_ns, "high" if tags.get("wildcard") == "1" else "medium",
                                  tags.get("rank"), tags.get("peer"), tags.get("wpeer"),
                                  tags.get("tag"), tags.get("commid")))
        elif t == "irecv" and s.span_id and tags.get("null") != "1":
            self.irecvs.append(((s.pid, s.span_id), i, s.start_ns, tags.get("wildcard") == "1",
                                tags.get("rank"), tags.get("peer") if "peer" in tags else None,
                                tags.get("wpeer"), tags.get("tag"), tags.get("commid")))
        if s.name in _MPI_WAIT_NAMES and s.parent_span_id:              # completions
            req_ids = _parse_multi(s.parent_span_id)
            pid = s.pid
            if "cpsid" in tags:
                self.cancelled.update((pid, r) for r in _parse_multi(tags["cpsid"]))
            elif tags.get("cancelled") == "1" and len(req_ids) == 1:
                self.cancelled.add((pid, req_ids[0]))
            rmatches = _parse_rmatches(tags["rmatches"]) if "rmatches" in tags else {}
            rw = _parse_rwmatches(tags["rwmatches"]) if "rwmatches" in tags else {}
            single_rpeer, single_rtag = tags.get("rpeer"), tags.get("rtag")
            single_rw = tags.get("rwpeer")
            for req_id in req_ids:
                if req_id in rmatches:
                    self.completers[(pid, req_id)] = (i, rmatches[req_id][0], rmatches[req_id][1],
                                                      rw.get(req_id))
                elif len(req_ids) == 1 and single_rpeer is not None:
                    self.completers[(pid, req_id)] = (i, single_rpeer, single_rtag, single_rw)
                else:
                    self.completers.setdefault((pid, req_id), (i, None, None, None))

    def _key(self, src_world, src_local, dst_world, tag, commid, world: bool):
        """(src, dst, tag[, commid]); None when a world rank is unknown."""
        if not world:
            return (src_world, src_local, tag) if src_local is not None else None
        return (src_world, dst_world, tag, commid) if src_world is not None and dst_world is not None \
            else None

    def edges(self) -> list[tuple[int, int, str]]:
        """[(dst ordinal, src ordinal, confidence)] in deterministic order."""
        world = self.world_info
        diag = self.diag
        diag["key"] = "world+commid" if world else "legacy"
        if not world and (self.sends or self.blocking or self.irecvs):
            _warn_legacy_key_once()
            diag["legacy_key"] = True

        def world_of(w, local, commid):
            # wpeer is missing on MPI_COMM_WORLD only in hand-built traces;
            # there the local rank is the world rank.
            return w if w is not None else (local if commid == "0" else None)

        sends: dict[tuple, list[tuple[int, int]]] = defaultdict(list)
        cancelled_sends = 0
        unkeyed = Counter()
        cancelled_comms: set[str] = set()
        for i, start, span_key, rank, peer, wpeer, tag, commid in self.sends:
            if span_key is not None and span_key in self.cancelled:
                cancelled_sends += 1
                cancelled_comms.add(str(commid))
                continue
            if world:
                key = self._key(rank, None, world_of(wpeer, peer, commid), tag, commid, True)
            else:
                key = (rank, peer, tag)
            if key is None:
                unkeyed[str(commid)] += 1
                continue
            sends[key].append((start, i))

        # (key, order_ns, target ordinal, confidence, commid)
        recv_events: list[tuple] = []
        for i, start, conf, rank, peer, wpeer, tag, commid in self.blocking:
            key = (world_of(wpeer, peer, commid), rank, tag, commid) if world else (peer, rank, tag)
            if world and key[0] is None:
                unkeyed[str(commid)] += 1
                continue
            recv_events.append((key, start, i, conf, commid))
        cancelled_recvs = 0
        unfinished = 0
        for span_key, _i, start, wildcard, rank, peer, wpeer, tag, commid in self.irecvs:
            if span_key in self.cancelled:
                cancelled_recvs += 1
                continue
            completer = self.completers.get(span_key)
            if completer is None:
                unfinished += 1          # completed by Test* only, or never
                continue
            target, rpeer, rtag, rwpeer = completer
            if wildcard:
                if rpeer is None:
                    unfinished += 1
                    continue
                src_local, src_tag, src_world = rpeer, rtag, rwpeer
                conf = "high"
            else:
                if peer is None:
                    continue
                src_local, src_tag, src_world = peer, tag, wpeer
                conf = "medium"
            if world:
                sw = world_of(src_world, src_local, commid)
                if sw is None:
                    unkeyed[str(commid)] += 1
                    continue
                key = (sw, rank, src_tag, commid)
            else:
                key = (src_local, rank, src_tag)
            recv_events.append((key, start, target, conf, commid))

        by_key: dict[tuple, list[tuple[int, int, str, str]]] = defaultdict(list)
        for key, order_ns, target, conf, commid in recv_events:
            by_key[key].append((order_ns, target, conf, commid))
        out: list[tuple[int, int, str]] = []
        unmatched = Counter(unkeyed)
        ex = self.exact
        for key, recv_list in by_key.items():
            recv_list.sort(key=lambda r: r[0])
            send_list = [i for _st, i in sorted(sends.get(key, []), key=lambda r: r[0])]
            channel_complete = ex is not None and world and len(send_list) == len(recv_list)
            for k, (_order, target, conf, commid) in enumerate(recv_list):
                if k < len(send_list):
                    src = send_list[k]
                    if channel_complete and conf == "medium":
                        c = str(commid)
                        sp, rp = self.pid_of.get(src, -1), self.pid_of.get(target, -1)
                        if (c not in cancelled_comms and c != "-1"
                                and sp in ex["complete"] and rp in ex["complete"]
                                and not {(sp, c), (rp, c), (sp, "*"), (rp, "*")} & ex["tainted"]):
                            conf = "high"
                            diag["exact_high"] = diag.get("exact_high", 0) + 1
                    if self.lossy and _lossy(self.lossy, self.pid_of.get(target, -1), self.pid_of.get(src, -1)):
                        conf = "low"
                        diag["matched_after_loss"] = diag.get("matched_after_loss", 0) + 1
                    out.append((target, src, conf))
                else:
                    unmatched[str(commid)] += 1
        diag["receives"] = len(recv_events) + sum(unkeyed.values())
        diag["matched"] = len(out)
        if unmatched:
            diag["unmatched_by_comm"] = dict(sorted(unmatched.items()))
        if unkeyed:
            diag["no_world_rank_by_comm"] = dict(sorted(unkeyed.items()))
        if cancelled_sends or cancelled_recvs:
            diag["cancelled"] = cancelled_sends + cancelled_recvs
        if unfinished:
            diag["irecv_without_wait"] = unfinished
        return out


_MPI_TEST_NAMES = frozenset({"MPI_Test", "MPI_Testany", "MPI_Testsome", "MPI_Testall"})


def mpi_dependency_diagnostics(trace: "Trace") -> dict:
    """Known holes in the MPI part of the dependency graph, counted (one
    pass over MPI spans and instants -- no full graph build):

      receives / matched      receive completions, and those that got a
                              point-to-point edge
      unmatched_by_comm       {commid: receive completions with no edge}
      no_world_rank_by_comm   {commid: messages without a world rank to key
                              on} (part of unmatched)
      legacy_key              the trace predates wpeer= (old matching key)
      test_only_completions   requests completed only via MPI_Test* (no
                              edge, by design: instants are not graph nodes)
      orphan_completions      Wait*/Waitall/... spans that name no request
                              the hook saw posted (e.g. a request from an
                              unhooked call) -- their waits have no
                              upstream edge
      cancelled               cancelled requests (excluded from matching)
      matched_after_loss      pairings in processes that lost MPI events
                              (event_loss_pids): order-based, possibly
                              shifted -- their edges are "low"
      exact_high              (HPROFILER_EXACT_P2P_HIGH only) exact matches
                              graded "high": their channel's capture is
                              complete (see EXACT_P2P_ENV)
    Empty for a trace without MPI."""
    m = _P2PMatcher(event_loss_pids(trace, ("mpi",)),
                    _exact_capture_facts(trace) if exact_p2p_enabled() else None)
    posted: set[tuple[int, str]] = set()             # (pid, request id): per process
    completed_by_wait: set[tuple[int, str]] = set()
    orphan = 0
    n = 0
    for s in trace.iter_spans(categories=("mpi",)):
        m.add(n, s)
        n += 1
        if s.span_id:
            posted.add((s.pid, s.span_id))
        if s.name in _MPI_WAIT_NAMES:
            ids = _parse_multi(s.parent_span_id) if s.parent_span_id else []
            if not ids:
                if s.tags.get("completed_index") != "none":
                    orphan += 1
            completed_by_wait.update((s.pid, r) for r in ids)
    if not n:
        return {}
    m.edges()
    out = dict(m.diag)
    test_ids: set[tuple[int, str]] = set()
    for ev in trace.iter_instants():
        if ev.name not in _MPI_TEST_NAMES:
            continue
        tags = ev.tags
        if ev.name in ("MPI_Test", "MPI_Testany", "MPI_Testall") and tags.get("flag") != "1":
            continue
        if tags.get("psid"):
            test_ids.update((ev.pid, r) for r in _parse_multi(str(tags["psid"])))
    test_only = len(test_ids - completed_by_wait)
    if test_only:
        out["test_only_completions"] = test_only
    unknown = len(completed_by_wait - posted)
    if orphan or unknown:
        out["orphan_completions"] = orphan + unknown
    if not (m.sends or m.blocking or m.irecvs or orphan or unknown or test_only):
        return {}
    return out


def dependency_warnings(diag: dict) -> list[str]:
    """Human-readable warnings for mpi_dependency_diagnostics() (shown by
    `critical-path`, after a run and by `summary`/GUI via run_warnings)."""
    out: list[str] = []
    if not diag:
        return out
    um = diag.get("unmatched_by_comm") or {}
    if um:
        per = ", ".join(f"{k}: {v}" for k, v in sorted(um.items(), key=lambda kv: -kv[1])[:6])
        out.append(f"{sum(um.values())} of {diag.get('receives', 0)} MPI receive completion(s) got no "
                   f"point-to-point edge (per communicator id: {per})"
                   + ("; no world rank to key on for "
                      f"{sum((diag.get('no_world_rank_by_comm') or {}).values())} of them"
                      if diag.get("no_world_rank_by_comm") else "")
                   + " -- the sender is missing (unhooked call, lost events) or not identifiable")
    if diag.get("legacy_key"):
        out.append("trace recorded without wpeer= tags (older MPI hook): messages were paired on "
                   "(sender world rank, receiver-side peer, tag) without the communicator, which is "
                   "correct on MPI_COMM_WORLD only")
    if diag.get("test_only_completions"):
        out.append(f"{diag['test_only_completions']} MPI request(s) completed only via MPI_Test* "
                   "polling: no dependency edge (by design)")
    if diag.get("orphan_completions"):
        out.append(f"{diag['orphan_completions']} MPI wait(s) completed request(s) the hook never saw "
                   "initiated (unhooked call, or posted before profiling): no upstream edge")
    if diag.get("matched_after_loss"):
        out.append(f"{diag['matched_after_loss']} MPI message pairing(s) involve a process that lost MPI "
                   "events: messages are paired in order per (sender, receiver, tag, communicator), so "
                   "one lost call shifts every later pair on its channel -- these edges are 'low' "
                   "confidence")
    return out


def _add_mpi_p2p_edges(preds: dict[int, list[tuple[int, str, str]]], spans: list["SpanEvent"],
                       lossy: "set[int] | str | None" = None, exact: dict | None = None) -> dict:
    """Point-to-point edges into each receive completion (a blocking
    MPI_Recv or the Wait*/Waitall/Waitany/Waitsome that completed an
    MPI_Irecv) from the send it received -- the Nth send to the Nth
    receive of each (sender, receiver, tag, communicator), in posting order
    (MPI's non-overtaking rule), NOT "the latest send that had already
    started": a receive posted early, then blocking, still pairs. The edge
    lands on the completer, not on MPI_Irecv (which returns at once).
    Confidence: "high" for a wildcard receive resolved from MPI_Status,
    "medium" for an exact receive, "low" in processes that lost MPI events
    (`lossy`, event_loss_pids). Returns the matcher's diagnostics."""
    m = _P2PMatcher(lossy, exact)
    for i, s in _items(spans):
        if s.category.value == "mpi":
            m.add(i, s)
    for target, src, conf in m.edges():
        preds[target].append((src, "p2p", conf))
    return m.diag


def _add_last_arriver_edges(
    preds: dict[int, list[tuple[int, str, str]]], spans: list["SpanEvent"], cluster: list[int], confidence: str
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
            preds[i].append((last, "arrival", confidence))


def _add_rendezvous_edges(
    preds: dict[int, list[tuple[int, str, str]]],
    spans: list["SpanEvent"],
    category: str,
    collective_types: frozenset[str],
) -> None:
    """Cluster collective calls into rendezvous groups per (type, commid)
    when a real communicator id is available (mpi_hook.c's MPI_Comm_dup/
    split/create bootstrap -- see its file header), falling back to
    per-type-only clustering when it
    isn't (commid absent, or the sentinel -1 "unregistered": MPI_COMM_SELF
    or a communicator created via an API mpi_hook.c doesn't intercept).
    The NCCL hook names a communicator the same way on every rank (commid=,
    from the ncclUniqueId) and numbers its collectives (seq=): NCCL requires
    all ranks to issue them in the same order, so (commid, seq) is one call
    and is grouped exactly, overlapping or not. Traces without commid= (older
    NCCL hook) use the per-type-only fallback. commid-scoped groups are
    "high" confidence (a cross-rank-agreed identity backs the grouping, not
    just overlapping timestamps of the same call type); unscoped ones are
    "medium".
    """
    by_type: dict[tuple, list[tuple[int, int, int]]] = defaultdict(list)
    for i, s in _items(spans):
        if s.category.value != category:
            continue
        key = _collective_group(s, collective_types)
        if key is not None:
            by_type[key].append((s.start_ns, s.end_ns, i))

    for key, items in by_type.items():
        for cluster in _rendezvous_clusters(key, items):
            _add_last_arriver_edges(preds, spans, cluster, key[-1])


def _omp_barrier_group(s: "SpanEvent") -> tuple | None:
    """(name, team) of an OpenMP barrier span, None for anything else.

    Only barriers are rendezvous (every thread of the team must arrive);
    omp_critical_wait, omp_taskwait, omp_taskgroup and omp_reduction are
    not: a thread waits there for a lock holder or its own child tasks, not
    for the last arriver. The team is the parallel region -- the OMPT tool's
    psid= (its parallel id), the GNU libgomp hook's team= -- so concurrent
    teams' barriers stay apart; without either (older traces) barriers of
    one name are grouped per process, as before."""
    if s.category.value != "sync" or not s.name.startswith("omp_barrier") or s.duration_ns <= 0:
        return None
    return (s.name, s.tags.get("team") or s.parent_span_id or None)


def _add_omp_barrier_edges(preds: dict[int, list[tuple[int, str, str]]], spans: list["SpanEvent"]) -> None:
    by_group: dict[tuple, list[tuple[int, int, int]]] = defaultdict(list)
    for i, s in _items(spans):
        g = _omp_barrier_group(s)
        if g is not None:
            by_group[(s.pid, *g)].append((s.start_ns, s.end_ns, i))
    for key, items in by_group.items():
        for cluster in _cluster_rendezvous(items):
            _add_last_arriver_edges(preds, spans, cluster, "certain")


def build_dependency_graph(trace: "Trace") -> tuple[list["SpanEvent"], dict[int, list[tuple[int, str, str]]]]:
    spans = trace.spans
    preds: dict[int, list[tuple[int, str, str]]] = defaultdict(list)

    _add_program_order_edges(preds, spans)
    _add_stream_order_edges(preds, spans)
    _add_device_sync_edges(preds, spans)
    _add_gpu_edges(preds, spans, event_loss_pids(trace, ("cuda", "rocm")))
    _add_explicit_span_id_edges(preds, spans)
    _add_mpi_p2p_edges(preds, spans, event_loss_pids(trace, ("mpi",)),
                       _exact_capture_facts(trace) if exact_p2p_enabled() else None)
    _add_rendezvous_edges(preds, spans, "mpi", _MPI_COLLECTIVE_TYPES)
    _add_rendezvous_edges(preds, spans, "nccl", _NCCL_COLLECTIVE_TYPES)
    _add_omp_barrier_edges(preds, spans)

    return spans, dict(preds)


# ── formal critical-path DP (DAG longest path) ─────────────────────────────

def _effective_start_ns(s: "SpanEvent") -> int:
    """The real execution-start wall-clock time for gate/gap computation --
    prefers the GPU-timeline exec-start (xs= tag, emitted by cuda_hook.c/
    rocm_hook.c via a reference-event calibration technique -- see their
    file comments; NOT independently verified against real kernel
    execution on this development machine, no working CUDA/ROCm GPU here,
    see DOCUMENTATION.md's Known Limitations) over the CPU-side launch-call
    time (start_ns) when available. Under stream queue backlog these can
    differ significantly: several kernels launched back-to-back all get
    CPU launch timestamps within microseconds of each other, but only the
    first can start executing immediately -- start_ns alone would report
    every one of them as if it began at launch-call time, understating how
    long a queued kernel actually waited before it could run. Falls back
    to start_ns for any span without an xs= tag (everything non-GPU, and
    any GPU span where calibration wasn't available) -- purely additive,
    changes nothing when xs= is absent."""
    xs = s.tags.get("xs")
    if xs is not None:
        try:
            return int(xs)
        except ValueError:
            pass
    return s.start_ns


def _effective_end_ns(s: "SpanEvent") -> int:
    # duration_ns (from cudaEventElapsedTime/hipEventElapsedTime) is already
    # GPU-measured elapsed time, accurate regardless of xs='s absolute
    # position -- so effective_start + duration is the corresponding real
    # execution-end, consistent with _effective_start_ns's reasoning.
    return _effective_start_ns(s) + s.duration_ns


def _edge_gap_and_gate(spans: list["SpanEvent"], u: int, v: int, kind: str) -> tuple[int, int, int]:
    """Returns (gate, limit, gap): gate<=limit is the causality condition;
    gap is the idle time between u
    and v this edge implies should be credited to the path (0 for
    START_GATED/rendezvous edges, since the wait is already inside v's own
    duration -- see the module docstring).

    Uses each span's *effective* start/end (see _effective_start_ns) rather
    than raw start_ns/end_ns directly, so GPU spans with a calibrated xs=
    exec-start get more accurate causal timing here specifically -- the
    numeric reasoning this DP's correctness actually depends on -- without
    touching edge *existence*/ordering elsewhere in this module (program-
    order and stream-order sorting, rendezvous clustering), which stay
    correct using plain start_ns regardless: FIFO submission order and FIFO
    execution order are the same order, so xs= cannot change *which* edges
    exist or *what order* spans are chained in, only how much idle time is
    credited between them.
    """
    if kind in _END_GATED:
        gate, limit = _effective_end_ns(spans[u]), _effective_start_ns(spans[v])
        gap = max(0, limit - gate)
        return gate, limit, gap
    if kind == "launch":
        # gap may be negative (work started before the call returned):
        # accounted[v] = accounted[u] + gap + dur(v) = accounted[u] + v.end - u.end.
        v_start = _effective_start_ns(spans[v])
        return spans[u].start_ns, v_start + ga.CLOCK_TOLERANCE_NS, v_start - spans[u].end_ns
    if kind == "device_wait":
        u_end = _effective_end_ns(spans[u])
        return u_end, spans[v].end_ns + ga.CLOCK_TOLERANCE_NS, spans[v].start_ns - u_end
    gate, limit = _effective_start_ns(spans[u]), _effective_end_ns(spans[v])
    return gate, limit, 0


def _topological_order(n: int, preds: dict[int, list[tuple[int, str, str]]]) -> list[int] | None:
    """Kahn's algorithm. Returns None if the graph isn't a DAG (shouldn't
    happen by construction -- every edge builder only ever points from an
    earlier-enabling event to a later-gated one -- but verified rather than
    assumed; see the module docstring)."""
    indeg = [0] * n
    succs: dict[int, list[int]] = defaultdict(list)
    for v, plist in preds.items():
        for (u, _kind, _conf) in plist:
            indeg[v] += 1
            succs[u].append(v)
    dq = deque(i for i in range(n) if indeg[i] == 0)
    order: list[int] = []
    while dq:
        u = dq.popleft()
        order.append(u)
        for v in succs.get(u, []):
            indeg[v] -= 1
            if indeg[v] == 0:
                dq.append(v)
    return order if len(order) == n else None


def _compute_critical_path_dp(
    spans: list["SpanEvent"], preds: dict[int, list[tuple[int, str, str]]]
) -> tuple[list[int], list[str]] | None:
    """Formal DAG longest-path dynamic program -- see module docstring.
    Returns (path, per-hop-confidence) or None if the graph has a cycle
    (caller falls back to the greedy walk)."""
    full = _compute_critical_path_dp_full(spans, preds)
    return None if full is None else (full[0], full[1])


def _compute_critical_path_dp_full(
    spans: list["SpanEvent"], preds: dict[int, list[tuple[int, str, str]]]
) -> tuple[list[int], list[str], list[str]] | None:
    """_compute_critical_path_dp plus the edge kind of each hop."""
    n = len(spans)
    order = _topological_order(n, preds)
    if order is None:
        return None

    accounted = [0] * n
    chosen: list[tuple[int, str, str] | None] = [None] * n
    gpu_model = [ga.is_new_format(s) for s in spans]
    annotation = [is_annotation(s) for s in spans]
    for v in order:
        if annotation[v]:          # not a node (no builder links one; edges persisted
            continue               # by older builders are ignored)
        dur_v = spans[v].duration_ns
        best = dur_v
        best_choice = None
        for (u, kind, conf) in preds.get(v, []):
            if annotation[u]:
                continue
            gate, limit, gap = _edge_gap_and_gate(spans, u, v, kind)
            if gate > limit:
                continue
            candidate = accounted[u] + gap + dur_v
            if candidate > best:
                best = candidate
                best_choice = (u, kind, conf)
            elif (candidate == best and best_choice is not None
                  and (gpu_model[v] or kind in _OVERLAP_KINDS or best_choice[1] in _OVERLAP_KINDS)
                  and _effective_end_ns(spans[u]) > _effective_end_ns(spans[best_choice[0]])):
                # Gap-filling edges make every chain between the same two
                # points account the same time, so ties are the norm in the
                # host/device model: a sync's full duration via program order
                # equals the work it waited for plus its tail; a kernel after
                # a cross-stream wait ties between its own stream's previous
                # work and the other stream's producer. The predecessor that
                # finished last is the one that actually gated v. (Applied
                # to CUDA/ROCm host/device spans only, so pre-split traces
                # keep their established path.)
                best_choice = (u, kind, conf)
        accounted[v] = best
        chosen[v] = best_choice

    end_idx = max((i for i in range(n) if not annotation[i]), key=lambda i: accounted[i], default=None)
    if end_idx is None:                           # annotations only
        return [], [], []
    path = [end_idx]
    edge_confidence: list[str] = []
    edge_kinds: list[str] = []
    cur = end_idx
    while chosen[cur] is not None:
        u, kind, conf = chosen[cur]
        edge_confidence.append(conf)
        edge_kinds.append(kind)
        path.append(u)
        cur = u
    path.reverse()
    edge_confidence.reverse()
    edge_kinds.reverse()
    return path, edge_confidence, edge_kinds


def _compute_critical_path_greedy(
    spans: list["SpanEvent"], preds: dict[int, list[tuple[int, str, str]]]
) -> tuple[list[int], list[str]]:
    """Fallback used only if the dependency graph is somehow not a DAG (see
    _topological_order) -- the original backward walk, robust to cycles by
    construction via its `visited` set, at the cost of only being a local
    greedy choice rather than the proven-optimal DP path."""
    path, conf, _kinds = _compute_critical_path_greedy_full(spans, preds)
    return path, conf


def _compute_critical_path_greedy_full(
    spans: list["SpanEvent"], preds: dict[int, list[tuple[int, str, str]]]
) -> tuple[list[int], list[str], list[str]]:
    nodes = [i for i in range(len(spans)) if not is_annotation(spans[i])]
    if not nodes:
        return [], [], []
    end_idx = max(nodes, key=lambda i: spans[i].end_ns)
    path = [end_idx]
    edge_confidence: list[str] = []
    edge_kinds: list[str] = []
    visited = {end_idx}
    cur = end_idx
    while True:
        candidates = preds.get(cur, [])
        best_idx, best_gate, best_conf, best_kind = None, -1, None, None
        for p_idx, kind, conf in candidates:
            if p_idx in visited or is_annotation(spans[p_idx]):
                continue
            gate, limit, _gap = _edge_gap_and_gate(spans, p_idx, cur, kind)
            if gate <= limit and gate > best_gate:
                best_gate, best_idx, best_conf, best_kind = gate, p_idx, conf, kind
        if best_idx is None:
            break
        path.append(best_idx)
        edge_confidence.append(best_conf)
        edge_kinds.append(best_kind)
        visited.add(best_idx)
        cur = best_idx
    path.reverse()
    edge_confidence.reverse()
    edge_kinds.reverse()
    return path, edge_confidence, edge_kinds


def compute_critical_path(
    spans: list["SpanEvent"], preds: dict[int, list[tuple[int, str, str]]]
) -> list[int]:
    """Public entry point -- see module docstring for the DP this runs.
    Kept as a thin wrapper returning just the path (matching the pre-DP
    signature every existing caller/test uses); call
    compute_critical_path_with_confidence for the per-hop confidence too."""
    path, _conf = compute_critical_path_with_confidence(spans, preds)
    return path


def compute_critical_path_with_confidence(
    spans: list["SpanEvent"], preds: dict[int, list[tuple[int, str, str]]]
) -> tuple[list[int], list[str]]:
    path, conf, _kinds = _compute_path_full(spans, preds)
    return path, conf


def _compute_path_full(
    spans: list["SpanEvent"], preds: dict[int, list[tuple[int, str, str]]]
) -> tuple[list[int], list[str], list[str]]:
    if not spans:
        return [], [], []
    result = _compute_critical_path_dp_full(spans, preds)
    if result is not None:
        return result
    return _compute_critical_path_greedy_full(spans, preds)


def attribute_blame(spans: list["SpanEvent"], path: list[int], wall_ns: int,
                    path_edge_confidence: list[str] | None = None,
                    path_edge_kinds: list[str] | None = None) -> CriticalPathReport:
    """The time the path covers, split into each span's own new time and the
    idle gaps before them.

    The path is walked keeping a frontier, the latest end reached so far:
    a span adds only the part of it beyond the frontier (own), and an idle
    gap only from the frontier to its start -- each wall-clock instant
    counts once, so total_path_ns <= the path's extent <= wall time. This
    is what makes overlap correct on every hop: a sync call overlapping the
    work it waited for, device work overlapping the call that launched it
    (launch/device_wait), and start-gated hops (rendezvous, p2p), whose
    successor may start before its predecessor -- or an earlier path span
    on another rank -- ended; measured from the immediate predecessor only,
    a zig-zag between two ranks counted the same wall time twice. Idle time
    is blamed on the predecessor's category (what the path waited for).

    dp_accounted_ns is the DP's value along this path (gap + duration per
    hop, as the DP adds them): what the longest-path search maximizes, an
    upper bound of total_path_ns that counts overlapping time on both sides
    of a start-gated hop. The report shows total_path_ns."""
    time_on_path: dict[str, int] = defaultdict(int)
    wait_caused: dict[str, int] = defaultdict(int)
    total_path_ns = 0
    kinds = list(path_edge_kinds or [])
    credited_ns: list[int] = []
    gap_ns: list[int] = []

    frontier = None
    for i, idx in enumerate(path):
        s = spans[idx]
        if frontier is None:
            gap, own = 0, s.duration_ns
            frontier = s.end_ns
        else:
            gap = max(0, s.start_ns - frontier)
            own = max(0, s.end_ns - max(s.start_ns, frontier))
            frontier = max(frontier, s.end_ns)
        if gap:
            wait_caused[spans[path[i - 1]].category.value] += gap
        time_on_path[s.category.value] += own
        total_path_ns += gap + own
        credited_ns.append(own)
        gap_ns.append(gap)

    dp_accounted = 0
    if path and len(kinds) == len(path) - 1:
        dp_accounted = spans[path[0]].duration_ns
        for i in range(1, len(path)):
            _gate, _limit, gap = _edge_gap_and_gate(spans, path[i - 1], path[i], kinds[i - 1])
            dp_accounted += gap + spans[path[i]].duration_ns

    return CriticalPathReport(
        path_span_indices=path,
        spans=spans,
        wall_ns=wall_ns,
        time_on_path_by_category=dict(time_on_path),
        wait_caused_by_category=dict(wait_caused),
        total_path_ns=total_path_ns,
        path_edge_confidence=list(path_edge_confidence or []),
        path_edge_kinds=kinds,
        path_credited_ns=credited_ns,
        path_gap_ns=gap_ns,
        dp_accounted_ns=dp_accounted,
    )


def annotation_time_on_path(trace: "Trace", report: CriticalPathReport) -> dict[str, int]:
    """{range name: path time inside it}: each path step's credited time
    and the idle gap before it go to every annotation range (NVTX/ROCTX)
    open on the step's thread when it started -- a device span's thread is
    the one that launched it. The ranges themselves are no path nodes
    (annotations contribute nothing); this only tells which program phases
    the path ran through."""
    path = report.path_span_indices
    if not path:
        return {}
    ranges: dict[tuple[int, int], list[tuple[int, int, str]]] = defaultdict(list)
    for a in trace.iter_spans(categories=("nvtx", "other")):
        if is_annotation(a) and a.duration_ns > 0:
            ranges[(a.pid, a.tid)].append((a.start_ns, a.end_ns, a.name))
    if not ranges:
        return {}
    steps: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    credited, gaps = report.path_credited_ns, report.path_gap_ns
    for i, idx in enumerate(path):
        s = report.spans[idx]
        if (s.pid, s.tid) in ranges:
            steps[(s.pid, s.tid)].append((s.start_ns, credited[i] + gaps[i]))
    out: dict[str, int] = defaultdict(int)
    for key, pts in steps.items():
        rs = sorted(ranges[key])
        pts.sort()
        open_: list[tuple[int, str]] = []       # (end, name) of ranges begun so far
        j = 0
        for t, ns in pts:
            while j < len(rs) and rs[j][0] <= t:
                open_.append((rs[j][1], rs[j][2]))
                j += 1
            open_ = [r for r in open_ if r[0] > t]
            for _end, name in open_:
                out[name] += ns
    return dict(out)


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
    """Critical path of a trace of any size: edges are built from
    time-ordered iterators and category subsets (never every span at once),
    persisted in the trace's store, and the DP runs over compact arrays --
    see "Scalable pipeline" below. Same result as running
    _compute_path_full() on build_dependency_graph()."""
    graph = load_or_build_graph(trace)
    path, path_edge_confidence, path_edge_kinds = graph.critical_path()
    spans = graph.spans
    ext = trace.store.span_extent(timed_only=True)
    wall_ns = max(1, ext[1] - ext[0]) if ext else 1
    report = attribute_blame(spans, path, wall_ns, path_edge_confidence, path_edge_kinds)
    report.path_time_by_annotation = annotation_time_on_path(trace, report)
    n_pids = len(trace.store.pids())
    if n_pids == 0:
        report.notes.append("No spans in this trace -- nothing to analyze.")
    elif n_pids == 1:
        report.notes.append(
            "Only one process observed -- cross-rank/collective edges did not apply; "
            "this is an intra-process (thread/stream) critical path only."
        )
    if report.dp_accounted_ns > report.total_path_ns * 1.01:
        report.notes.append(
            f"The longest-path search's own value for this path ({report.dp_accounted_ns / 1e6:.2f} ms) "
            f"exceeds the time it covers ({report.total_path_ns / 1e6:.2f} ms): it passes through "
            "start-gated hops (MPI/NCCL collectives, OpenMP barriers, point-to-point messages) whose "
            "successor overlaps its predecessor in wall time on another rank/thread, and the search "
            "counts that overlap on both sides. Totals and breakdowns count each instant once."
        )
    ka = ga.kernel_activity(trace.iter_spans(categories=("cuda", "rocm", "opencl")))
    if ka.source == "proxy":
        report.notes.append(
            "GPU device spans are host-side proxies (no CUPTI / ROCprofiler-SDK "
            "activity in this trace): their durations are GPU-event measured but "
            "their start times are submission times, so queueing delay is "
            "invisible and stream-order/launch edges can be dropped where queued "
            "work appears to overlap."
        )
    try:
        report.diagnostics = mpi_dependency_diagnostics(trace)
    except Exception:          # diagnostics must never cost the analysis
        report.diagnostics = {}
    report.notes.extend(dependency_warnings(report.diagnostics))
    breakdown = report.confidence_breakdown_ns()
    medium_or_below = sum(ns for tier, ns in breakdown.items() if _CONFIDENCE_RANK.get(tier, 0) <= 1)
    if report.total_path_ns > 0 and medium_or_below / report.total_path_ns > 0.3:
        report.notes.append(
            "Over 30% of the reported path's time rests on 'medium' confidence "
            "edges (call-order matching without resolved MPI status data or a "
            "real communicator id) -- see confidence_breakdown_ns() / "
            "DOCUMENTATION.md's edge confidence table before treating this "
            "path as precise."
        )
    return report


# ── Scalable pipeline: streamed edge building, persisted edges, array DP ───────
#
# Node ids are ordinals: the index of a span in arrival (seq) order, the
# same numbering build_dependency_graph() uses for its span list, so both
# pipelines number nodes identically. Builders write edges into an
# _EdgeSink (compact typed arrays) in the same order they appended to the
# dict-of-lists `preds`, which preserves every tie-break. The program-order
# builder -- the only one that touches every span -- streams each thread's
# spans in start order holding only the spans still open; every other
# builder gets just its category subset as an {ordinal: span} dict.

EDGE_KINDS = ("sequential", "device_sync", "explicit_span_id", "p2p", "arrival", "launch", "device_wait")
EDGE_CONFS = ("certain", "high", "medium", "low")
_KIND_CODE = {k: i for i, k in enumerate(EDGE_KINDS)}
_CONF_CODE = {c: i for i, c in enumerate(EDGE_CONFS)}
# Bumped whenever an edge builder changes what it emits: persisted edges
# with another version are rebuilt.
# cp-2: world-rank MPI keys, per-device streams, graphs, teams, loss, NCCL seq
# cp-3: annotation ranges (NVTX/ROCTX) are no graph nodes
# cp-4: a span id shared by two hooks' spans resolves by the child's hook family
EDGES_VERSION = "cp-4"


class _EdgeSink:
    """Drop-in for the `preds` dict the builders append to: preds[v].append(
    (u, kind, conf)) records the edge in compact arrays."""

    def __init__(self) -> None:
        import array
        self.dst = array.array("q")
        self.src = array.array("q")
        self.kind = array.array("b")
        self.conf = array.array("b")

    def __getitem__(self, v: int) -> "_EdgeAppender":
        return _EdgeAppender(self, v)

    def to_array(self):
        import numpy as np
        from ..core.store import EDGE_DTYPE
        out = np.empty(len(self.dst), dtype=EDGE_DTYPE)
        out["dst"] = np.frombuffer(self.dst, dtype=np.int64) if len(self.dst) else []
        out["src"] = np.frombuffer(self.src, dtype=np.int64) if len(self.src) else []
        out["kind"] = np.frombuffer(self.kind, dtype=np.int8) if len(self.kind) else []
        out["conf"] = np.frombuffer(self.conf, dtype=np.int8) if len(self.conf) else []
        return out


class _EdgeAppender:
    __slots__ = ("sink", "v")

    def __init__(self, sink: _EdgeSink, v: int) -> None:
        self.sink, self.v = sink, v

    def append(self, edge: tuple[int, str, str]) -> None:
        u, kind, conf = edge
        self.sink.dst.append(self.v)
        self.sink.src.append(u)
        self.sink.kind.append(_KIND_CODE[kind])
        self.sink.conf.append(_CONF_CODE[conf])


class SpanSequence:
    """Read-only sequence of a trace's spans by ordinal, fetched from the
    store on demand (CriticalPathReport.spans without materializing them)."""

    def __init__(self, store, seqs) -> None:
        self._store = store
        self._seqs = seqs
        self._cache: dict[int, "SpanEvent"] = {}

    def __len__(self) -> int:
        return len(self._seqs)

    def __getitem__(self, i: int) -> "SpanEvent":
        if i < 0:
            i += len(self._seqs)
        s = self._cache.get(i)
        if s is None:
            s = self._store.span_by_seq(int(self._seqs[i]))
            self._cache[i] = s
        return s

    def __iter__(self):
        for s in self._store.iter_spans():
            yield s


class Graph:
    """Dependency graph of a trace: ordinals, persisted edges, node arrays."""

    def __init__(self, trace: "Trace", seqs, edges) -> None:
        self.trace = trace
        self.seqs = seqs
        self.edges = edges
        self.spans = SpanSequence(trace.store, seqs)

    def ordinal_of(self, seq: int) -> int:
        import numpy as np
        return int(np.searchsorted(self.seqs, seq))

    def preds_dict(self) -> dict[int, list[tuple[int, str, str]]]:
        """The edges as build_dependency_graph()'s dict (small traces)."""
        preds: dict[int, list[tuple[int, str, str]]] = defaultdict(list)
        for d, s_, k, c in self.edges.tolist():
            preds[d].append((s_, EDGE_KINDS[k], EDGE_CONFS[c]))
        return dict(preds)

    def subset(self, **query) -> dict[int, "SpanEvent"]:
        return {self.ordinal_of(s.seq): s for s in self.trace.iter_spans(**query)}

    def endpoints(self, ordinals: set[int], **query) -> dict[int, "SpanEvent"]:
        """{ordinal: span} for just `ordinals` among the spans `query`
        selects (one streaming pass; only those spans are kept)."""
        out = {}
        for s in self.trace.iter_spans(**query):
            i = self.ordinal_of(s.seq)
            if i in ordinals:
                out[i] = s
        return out

    def critical_path(self) -> tuple[list[int], list[str], list[str]]:
        return _dp_arrays(self.trace, self.seqs, self.edges)


def edges_version() -> str:
    """Persisted edges' version: the exact-match rule changes them."""
    return EDGES_VERSION + ("+exact" if exact_p2p_enabled() else "")


def load_or_build_graph(trace: "Trace") -> Graph:
    """Persisted edges when the store has them for the current events and
    builder version, else build (streaming) and persist them."""
    import numpy as np
    store = trace.store
    seqs = store.span_seqs()
    version = edges_version()
    edges = store.load_edges(version)
    if edges is None:
        edges = build_edges(trace, seqs)
        store.save_edges(version, edges)
    return Graph(trace, seqs, edges)


def load_comm_graph(trace: "Trace", max_build_spans: int | None = None) -> "Graph | None":
    """The trace's communication edges (p2p + arrival) for timeline
    connectors: read straight from persisted edges when present (only
    those kinds are loaded), else built -- unless the trace has more than
    `max_build_spans` spans, then None (connectors are skipped rather than
    paying a full dependency-graph build just to open a view)."""
    import numpy as np
    store = trace.store
    kinds = (_KIND_CODE["p2p"], _KIND_CODE["arrival"])
    edges = store.load_edges(edges_version(), kinds=kinds)
    if edges is None:
        if max_build_spans is not None and store.span_count() > max_build_spans:
            return None
        full = load_or_build_graph(trace)
        edges = full.edges[np.isin(full.edges["kind"], kinds)]
        return Graph(trace, full.seqs, edges)
    if not len(edges):          # nothing to draw: skip the ordinal table
        return Graph(trace, np.empty(0, dtype=np.int64), edges)
    return Graph(trace, store.span_seqs(), edges)


def build_edges(trace: "Trace", seqs=None):
    """Every dependency edge of the trace (EDGE_DTYPE array, insertion
    order) -- same edges, same order as build_dependency_graph()."""
    import numpy as np
    if seqs is None:
        seqs = trace.store.span_seqs()

    def ordinal_of(seq: int) -> int:
        return int(np.searchsorted(seqs, seq))

    def legacy_gpu(pid: int) -> dict[int, "SpanEvent"]:
        # what _add_stream_order_edges/_add_device_sync_edges read, in
        # ordinal order: pre-split GPU spans and stream/device sync calls
        out = {ordinal_of(s.seq): s for s in trace.iter_spans(pid=pid, categories=_GPU_CATS)
               if not ga.is_new_format(s)}
        out.update((ordinal_of(s.seq), s) for s in trace.iter_spans(pid=pid, categories=("sync",))
                   if not ga.is_new_format(s) and ("stream" in s.tags or s.name in _DEVICE_SYNC_NAMES))
        return dict(sorted(out.items()))

    # Stream, device-sync, host/device and barrier edges never cross
    # processes, so those builders run one process at a time (peak memory =
    # the largest process's subset); each destination still receives its
    # edges in the same builder order as build_dependency_graph().
    sink = _EdgeSink()
    _add_program_order_edges_streaming(sink, trace, ordinal_of)
    pids = trace.store.pids()
    for pid in pids:
        gpu_legacy = legacy_gpu(pid)
        _add_stream_order_edges(sink, gpu_legacy)
        _add_device_sync_edges(sink, gpu_legacy)
        del gpu_legacy
    gpu_lossy = event_loss_pids(trace, ("cuda", "rocm"))
    for pid in pids:
        _add_gpu_edges(sink, {ordinal_of(s.seq): s for s in trace.iter_spans(pid=pid, gpu_model=True)},
                       gpu_lossy)
    _add_explicit_span_id_edges_compact(sink, trace, ordinal_of)
    _add_mpi_edges_compact(sink, trace, ordinal_of)
    nccl: dict[tuple, list[tuple[int, int, int]]] = defaultdict(list)
    for s in trace.iter_spans(categories=("nccl",)):
        _collect_collective(nccl, s, _NCCL_COLLECTIVE_TYPES, ordinal_of)
    _add_clustered_arrival_edges(sink, nccl)
    for pid in pids:
        _add_omp_barrier_edges_compact(sink, trace, pid, ordinal_of)
    return sink.to_array()


# The compact builders below are the span-subset builders above rewritten
# to stream spans once and keep only small tuples (ordinal, times, a few
# tag values) -- MPI calls, request ids and barriers can each be a large
# share of a trace. Each reproduces its original's iteration order, so
# every destination gets the same edges in the same order (checked against
# build_dependency_graph() by tests/test_trace_store_parity.py).

def _add_clustered_arrival_edges(preds, groups: dict[tuple, list[tuple[int, int, int]]]) -> None:
    """_cluster_rendezvous + _add_last_arriver_edges over {(group key...,
    confidence): [(start, end, ordinal), ...]} in insertion order; the last
    arriver is the first latest-starting member in cluster order, as in
    _add_last_arriver_edges."""
    for key, items in groups.items():
        confidence = key[-1]
        start_of = {i: st for st, _en, i in items}
        clusters = _rendezvous_clusters(key, items) if len(key) == 4 else _cluster_rendezvous(items)
        for cluster in clusters:
            last = max(cluster, key=start_of.__getitem__)
            for i in cluster:
                if i != last:
                    preds[i].append((last, "arrival", confidence))


def _collective_group(s: "SpanEvent", types: frozenset[str]) -> tuple | None:
    """(type, commid, seq, confidence) rendezvous group of a collective span
    (see _add_rendezvous_edges), None for anything else. seq is set only
    for NCCL calls with commid= and seq=: an exact group."""
    t = s.tags.get("type", "")
    if t not in types or s.duration_ns <= 0:
        return None
    commid = s.tags.get("commid")
    if commid is None or commid == "-1":
        return (t, None, None, "medium")
    seq = s.tags.get("seq") if s.category.value == "nccl" else None
    return (t, commid, seq, "high")


def _rendezvous_clusters(key: tuple, items: list[tuple[int, int, int]]) -> list[list[int]]:
    """An exact (commid, seq) group is one rendezvous; otherwise clusters of
    overlapping calls (_cluster_rendezvous)."""
    if key[2] is not None:
        return [[i for _st, _en, i in items]] if len(items) > 1 else []
    return _cluster_rendezvous(items)


def _collect_collective(groups, s: "SpanEvent", types: frozenset[str], ordinal_of) -> None:
    """One span's contribution to _add_rendezvous_edges' by_type grouping
    (key ends with the confidence that grouping implies)."""
    key = _collective_group(s, types)
    if key is not None:
        groups[key].append((s.start_ns, s.end_ns, ordinal_of(s.seq)))


def _add_explicit_span_id_edges_compact(preds, trace: "Trace", ordinal_of) -> None:
    """_add_explicit_span_id_edges: span ids -> ordinal (last wins, as
    there), then links in arrival order."""
    ids = _SpanIdIndex()
    children: list[tuple[int, int, str, int, str]] = []
    for s in trace.iter_spans(with_ids=True):
        if is_annotation(s):
            continue
        i = ordinal_of(s.seq)
        if s.span_id:
            ids.add(s, i)
        if s.parent_span_id and not ga.is_device_span(s):
            children.append((i, s.pid, s.parent_span_id, s.start_ns, id_family(s)))
    for i, pid, psid, start, family in children:
        for req_id in _parse_multi(psid):
            parent_idx = ids.get(pid, req_id, start, family)
            if parent_idx is not None and parent_idx != i:
                preds[i].append((parent_idx, "explicit_span_id", "certain"))


def _add_mpi_edges_compact(preds, trace: "Trace", ordinal_of) -> dict:
    """_add_mpi_p2p_edges then _add_rendezvous_edges(..., "mpi", ...) from
    one pass over the MPI spans. Returns the matcher's diagnostics."""
    m = _P2PMatcher(event_loss_pids(trace, ("mpi",)),
                    _exact_capture_facts(trace) if exact_p2p_enabled() else None)
    colls: dict[tuple, list[tuple[int, int, int]]] = defaultdict(list)
    for s in trace.iter_spans(categories=("mpi",)):
        i = ordinal_of(s.seq)
        m.add(i, s)
        _collect_collective(colls, s, _MPI_COLLECTIVE_TYPES, ordinal_of)
    for target, src, conf in m.edges():
        preds[target].append((src, "p2p", conf))
    _add_clustered_arrival_edges(preds, colls)
    return m.diag


def _add_omp_barrier_edges_compact(preds, trace: "Trace", pid: int, ordinal_of) -> None:
    """_add_omp_barrier_edges for one process."""
    by_name: dict[tuple, list[tuple[int, int, int]]] = defaultdict(list)
    for s in trace.iter_spans(pid=pid, categories=("sync",)):
        g = _omp_barrier_group(s)
        if g is not None:
            by_name[(*g, "certain")].append((s.start_ns, s.end_ns, ordinal_of(s.seq)))
    _add_clustered_arrival_edges(preds, by_name)


def _add_program_order_edges_streaming(preds, trace: "Trace", ordinal_of) -> None:
    """_add_program_order_edges, one thread at a time in (start, seq)
    order, holding only the spans still open on that thread."""
    import heapq
    for pid, tid in trace.store.threads():
        pending: list[tuple[int, int, int]] = []
        best: tuple[int, int, int] | None = None
        for s in trace.iter_spans(order="start", pid=pid, tid=tid):
            if ga.is_device_span(s) or is_annotation(s):
                continue
            b = ordinal_of(s.seq)
            b_start = s.start_ns
            while pending and pending[0][0] <= b_start:
                end, start, i = heapq.heappop(pending)
                cand = (end, -start, i)
                if best is None or cand > best:
                    best = cand
            if best is not None:
                preds[b].append((best[2], "sequential", "certain"))
            heapq.heappush(pending, (s.end_ns, b_start, b))


def _dp_arrays(trace: "Trace", seqs, edges) -> tuple[list[int], list[str], list[str]]:
    """_compute_critical_path_dp_full over compact per-node arrays (one
    streaming pass loads start/end/effective start and the GPU-model flag)
    and the edge array sorted stably by destination."""
    import array
    from collections import deque
    import numpy as np

    n = len(seqs)
    if n == 0:
        return [], [], []
    start = array.array("q", bytes(8 * n))
    end = array.array("q", bytes(8 * n))
    eff = array.array("q", bytes(8 * n))
    gpu = bytearray(n)
    annot = bytearray(n)
    i = 0
    for s in trace.iter_spans():           # seq order == ordinal order
        start[i] = s.start_ns
        end[i] = s.end_ns
        eff[i] = _effective_start_ns(s)
        gpu[i] = 1 if ga.is_new_format(s) else 0
        annot[i] = 1 if is_annotation(s) else 0
        i += 1

    # Typed arrays, not lists: per-node/per-edge Python ints would cost
    # ~36 bytes each instead of 8.
    def q(a) -> array.array:
        return array.array("q", np.ascontiguousarray(a, dtype=np.int64).tobytes())

    order = np.argsort(edges["dst"], kind="stable")
    dst = edges["dst"][order]
    src = q(edges["src"][order])
    kind = array.array("b", edges["kind"][order].astype(np.int8).tobytes())
    conf = array.array("b", edges["conf"][order].astype(np.int8).tobytes())
    indptr = q(np.searchsorted(dst, np.arange(n + 1)))
    del dst, order

    # Kahn's algorithm (any topological order gives the same DP values;
    # ties are decided per node by its predecessors' insertion order).
    indeg = q(np.bincount(edges["dst"], minlength=n)) if len(edges) else q(np.zeros(n))
    by_src = np.argsort(edges["src"], kind="stable")
    succ_dst = q(edges["dst"][by_src])
    succ_ptr = q(np.searchsorted(edges["src"][by_src], np.arange(n + 1)))
    del by_src
    dq = deque(v for v in range(n) if indeg[v] == 0)
    topo = array.array("q")
    while dq:
        u = dq.popleft()
        topo.append(u)
        for e in range(succ_ptr[u], succ_ptr[u + 1]):
            v = succ_dst[e]
            indeg[v] -= 1
            if indeg[v] == 0:
                dq.append(v)
    if len(topo) != n:
        # Cycle (never expected): fall back to the materialized greedy walk.
        spans = trace.spans
        return _compute_critical_path_greedy_full(spans, Graph(trace, seqs, edges).preds_dict())

    SEQ, DSYNC, EXPL, P2P, ARR, LAUNCH, DWAIT = range(7)
    overlap = (LAUNCH, DWAIT)
    tol = ga.CLOCK_TOLERANCE_NS
    acc = array.array("q", bytes(8 * n))
    chosen = array.array("q", [-1]) * n
    for v in topo:
        if annot[v]:                              # not a node (see the full DP)
            continue
        dur_v = end[v] - start[v]
        best = dur_v
        best_e = -1
        eff_v = eff[v]
        for e in range(indptr[v], indptr[v + 1]):
            u = src[e]
            if annot[u]:
                continue
            k = kind[e]
            eff_u = eff[u]
            eff_end_u = eff_u + (end[u] - start[u])
            if k <= EXPL:                         # end-gated
                gate, limit = eff_end_u, eff_v
                gap = limit - gate if limit > gate else 0
            elif k == LAUNCH:
                gate, limit = start[u], eff_v + tol
                gap = eff_v - end[u]
            elif k == DWAIT:
                gate, limit = eff_end_u, end[v] + tol
                gap = start[v] - eff_end_u
            else:                                 # p2p / arrival: start-gated
                gate, limit = eff_u, eff_v + dur_v
                gap = 0
            if gate > limit:
                continue
            cand = acc[u] + gap + dur_v
            if cand > best:
                best = cand
                best_e = e
            elif (cand == best and best_e >= 0
                  and (gpu[v] or k in overlap or kind[best_e] in overlap)):
                bu = src[best_e]
                if eff_end_u > eff[bu] + (end[bu] - start[bu]):
                    best_e = e
        acc[v] = best
        chosen[v] = best_e

    end_idx = max((v for v in range(n) if not annot[v]), key=acc.__getitem__, default=None)
    if end_idx is None:                           # annotations only
        return [], [], []
    path = [end_idx]
    confs: list[str] = []
    kinds: list[str] = []
    cur = end_idx
    while chosen[cur] >= 0:
        e = chosen[cur]
        confs.append(EDGE_CONFS[conf[e]])
        kinds.append(EDGE_KINDS[kind[e]])
        cur = src[e]
        path.append(cur)
    path.reverse()
    confs.reverse()
    kinds.reverse()
    return path, confs, kinds


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
