"""
Multi-node trace alignment and merging.

hprofiler's collector is a local AF_UNIX socket (see criticalpath.py's
module docstring for why that makes today's traces inherently single-node
regardless of clock synchronization) -- a multi-node job naturally produces
one independent trace file per node, each on that node's own
CLOCK_MONOTONIC timeline with its own arbitrary epoch. This module aligns
several such per-node traces onto one common timeline and combines them
into a single Trace, so criticalpath.py / pop_efficiency.py can operate
across node boundaries instead of just within one.

── Clock offset estimation ──────────────────────────────────────────────
estimate_clock_offset() implements Cristian's algorithm (the classic
round-trip clock-offset estimate, the same technique NTP itself is built
on): given four timestamps from a single round-trip ping/reply exchange
between a node and a reference node, it returns both a point estimate AND
an explicit error bound (half the round-trip time, the standard bound
under the assumption of symmetric network latency) -- never claiming more
precision than the measurement actually supports.

offset_from_counters() reads that estimate back out of a trace that was
captured with hooks/mpi_hook/mpi_hook.c's opt-in HPROFILER_CLOCK_SYNC
round-trip exchange (off by default; see that function's comment for the
full protocol and its own verification-status note -- this Python side is
independently, fully unit-tested against synthetic offset scenarios, but
the C-side round-trip capture itself has never executed against a real
multi-node job on this development machine, which cannot form a real
multi-rank MPI_COMM_WORLD at all).

── Merging ───────────────────────────────────────────────────────────────
merge_traces() combines N per-node traces into one, applying each node's
clock offset to its timestamps and remapping `pid` into a per-node-unique
namespace (pid 1234 on node A and pid 1234 on node B are unrelated
processes -- not remapping would let unrelated nodes' events get grouped
together by every (pid, tid)-keyed piece of existing analysis code). `tid`
is deliberately left unchanged: everywhere in this codebase that groups by
thread already groups by the (pid, tid) pair together, so a per-node-
unique pid alone is sufficient to keep those tuples unique post-merge.
MPI `rank=`/`peer=` tags are also left unchanged -- MPI_COMM_WORLD ranks
are already globally unique across an entire job regardless of which
physical node a rank happens to run on, so criticalpath.py's cross-rank
P2P/collective matching (built on those tags, not pid/tid) works
transparently across a merge with no further change needed; this is a
direct benefit of this redesign's Phase 1 MPI protocol work capturing real
ranks and communicator identity.

validate_causality() is a sanity check for the result: after merging, a
matched MPI send can never causally complete after its receive already
finished. A violation means either a bad offset estimate or a genuine
clock anomaly -- surfaced explicitly rather than silently trusted.

Selective aggregation: merge_traces() takes a plain list, so a caller
merges only the NodeTrace entries they want included (e.g. a subset of
ranks/nodes) simply by not including the others -- no separate filtering
API needed beyond ordinary list slicing/filtering before the call.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.trace import Trace

# mpi_hook.c's HPROFILER_CLOCK_SYNC counters carry these exact names.
_COUNTER_OFFSET = "clock_offset_vs_rank0_ns"
_COUNTER_ERROR_BOUND = "clock_offset_error_bound_ns"
_COUNTER_ROUND_TRIP = "clock_sync_round_trip_ns"

# pid namespace stride for merge_traces -- large enough that no real
# process's pid (max ~4*10^6 on a default-configured Linux system, per
# /proc/sys/kernel/pid_max) could collide with the next node's offset.
PID_NAMESPACE_STRIDE = 10_000_000


@dataclass
class ClockOffsetEstimate:
    offset_ns: int
    error_bound_ns: int
    round_trip_ns: int


def estimate_clock_offset(t1: int, t2: int, t3: int, t4: int) -> ClockOffsetEstimate:
    """Cristian's algorithm. t1 = local send time, t2 = remote receive
    time, t3 = remote reply time, t4 = local reply-received time -- all in
    their own clock's units (e.g. ns since CLOCK_MONOTONIC boot on their
    respective machine, not comparable to each other before this
    computation). Returns the estimated offset (remote clock minus local
    clock, at the same real instant -- add it to a local timestamp to
    express it in the remote clock's frame) and its error bound (half the
    round trip; assumes symmetric network latency, which is a real
    approximation, not an exact guarantee -- hence returning a bound, not
    pretending the point estimate alone is precise)."""
    round_trip = t4 - t1
    offset = (t2 + t3) // 2 - (t1 + round_trip // 2)
    return ClockOffsetEstimate(offset_ns=offset, error_bound_ns=round_trip // 2,
                               round_trip_ns=round_trip)


def offset_from_counters(trace: "Trace") -> ClockOffsetEstimate | None:
    """Reads back the HPROFILER_CLOCK_SYNC counters mpi_hook.c emits (see
    module docstring). Returns None if absent -- either HPROFILER_CLOCK_SYNC
    wasn't set for the run that produced this trace, or this trace IS the
    reference (rank 0), which never emits these (its offset is 0 by
    construction, not measured)."""
    offset = error_bound = round_trip = None
    for c in trace.counters:
        if c.name == _COUNTER_OFFSET:
            offset = int(c.value)
        elif c.name == _COUNTER_ERROR_BOUND:
            error_bound = int(c.value)
        elif c.name == _COUNTER_ROUND_TRIP:
            round_trip = int(c.value)
    if offset is None or error_bound is None:
        return None
    return ClockOffsetEstimate(offset_ns=offset, error_bound_ns=error_bound,
                               round_trip_ns=round_trip or 0)


@dataclass
class NodeTrace:
    trace: "Trace"
    offset_ns: int = 0
    error_bound_ns: int = 0
    label: str = ""   # e.g. hostname -- diagnostics only, not load-bearing


def merge_traces(nodes: list[NodeTrace]) -> tuple["Trace", list[str]]:
    """Combines several per-node traces into one Trace on a common
    timeline -- see module docstring for the offset/pid-remapping/rank-
    tag-preservation rules. Returns (merged_trace, warnings); a node with
    no distinguishing offset data merges at an uncorrected 0 offset and is
    called out explicitly in warnings (unless it's node 0, the assumed
    reference, for which offset=0 is the expected, correct value, not a
    missing measurement)."""
    from ..core.trace import Trace, TraceMetadata

    merged = Trace(TraceMetadata(command="(multi-node merge)"))
    warnings: list[str] = []

    for node_idx, node in enumerate(nodes):
        pid_offset = node_idx * PID_NAMESPACE_STRIDE
        if node_idx != 0 and node.offset_ns == 0 and node.error_bound_ns == 0:
            warnings.append(
                f"node {node_idx} ({node.label or 'unlabeled'}): no clock-offset "
                f"data provided -- merged at an uncorrected 0 offset; timestamps "
                f"are not meaningfully comparable to other nodes unless this node "
                f"genuinely is the reference clock"
            )
        for s in node.trace.spans:
            merged.add(replace(s, pid=s.pid + pid_offset,
                               start_ns=s.start_ns + node.offset_ns,
                               tags={**s.tags, "node": str(node_idx)}))
        for i in node.trace.instants:
            merged.add(replace(i, pid=i.pid + pid_offset,
                               timestamp_ns=i.timestamp_ns + node.offset_ns,
                               tags={**i.tags, "node": str(node_idx)}))
        for c in node.trace.counters:
            merged.add(replace(c, pid=c.pid + pid_offset,
                               timestamp_ns=c.timestamp_ns + node.offset_ns))

    return merged, warnings


def validate_causality(trace: "Trace") -> list[str]:
    """After merging, a matched MPI send can never causally complete after
    its receive already finished -- checks this holds for every p2p edge
    criticalpath.py's own matching would build (reusing
    build_dependency_graph so this validates exactly the pairing the
    critical-path/blame-attribution engine itself will use, not a separate
    approximation of it). A violation means either a bad clock-offset
    estimate or a genuine anomaly -- surfaced explicitly, not silently
    accepted into a critical-path report that would then misattribute
    blame across the false ordering."""
    from ..analysis import criticalpath as cp

    spans, preds = cp.build_dependency_graph(trace)
    problems: list[str] = []
    for v, edges in preds.items():
        for (u, kind, conf) in edges:
            if kind != "p2p":
                continue
            send, recv = spans[u], spans[v]
            if send.start_ns > recv.end_ns:
                problems.append(
                    f"causality violation ({conf} confidence match): send on "
                    f"pid={send.pid} starts at {send.start_ns}ns, AFTER its "
                    f"matched receive on pid={recv.pid} already ended at "
                    f"{recv.end_ns}ns -- clock offset estimate is likely wrong "
                    f"for one of these nodes"
                )
    return problems
