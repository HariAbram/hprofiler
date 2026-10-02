"""
Trace projection: a run-independent, structure-aware model of one trace,
built for comparing runs (src/analysis/causal_compare.py) but usable by any
analysis that needs "what ran where, in which context, and what depended on
what" without raw process/thread/stream ids or absolute timestamps.

A projection groups spans into **nodes**: one per (phase, normalized calling
context, role). The calling context is the chain of enclosing host spans on
the same thread (temporal containment, the rule the call tree uses); a
device span's context is the context of the host call that launched it
(CUDA/ROCm launch edges) or, without one, the host span open on its
launching thread when it started. Names are normalized (JIT hashes,
addresses, clone suffixes). Roles replace raw ids that differ between
runs:

    backend   category, split into host/device for GPU runtimes
    rank      MPI rank (rank= tag), else process order of first activity
    thread    main (first active thread of the process) / worker / device
    stream    stream order within the process (numeric ids sorted, handles
              by first use), "" for host work
    device    device ordinal tag
    comm      communicator, by order of first use within the process (commid=,
              agreed across ranks by the MPI hook); "unregistered" for -1

Each node carries measured counts and durations (inclusive, exclusive
"self", queue delay from native device timing, time overlapped with
concurrent work on other threads/streams) and graph-derived values from
the critical path (time credited on the path, idle time on the path
blamed on it). Node-level dependency edges aggregate the span-level edges
of src/analysis/criticalpath.py with their best confidence.

**Phases** (heuristic): the driver thread (main thread of rank 0, or of the
first process) is cut into a prologue, repeated iterations and an epilogue
when its top-level call sequence -- tokens augmented with the kinds of
cross-thread dependency edges (p2p, arrival, device_wait, device_sync)
under each call -- recurs; otherwise into segments of consecutive
same-kind top-level calls. When one call covers most of the driver's time
(a `solve()` wrapping the loop), detection descends into its children.
Every span of every process is assigned to the phase window containing its
start (device work: its launching call's phase).

Everything is computed from streaming store queries; memory is O(spans)
only for three int32 arrays (node, path, phase per span), plus the nodes,
top-level host calls of each process and the critical path.
"""
from __future__ import annotations

import bisect
import heapq
import itertools
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from . import activity_buckets as ab
from . import criticalpath as cp
from . import dashboard as dash

PROJECTION_VERSION = 1

# Phase detection limits: beyond these, consecutive iterations are grouped
# into blocks / consecutive segments merged, keeping alignment O(phases^2)
# affordable.
MAX_ITERATION_PHASES = 400
MAX_SEGMENT_PHASES = 200
MAX_PERIOD_TOKENS = 200_000      # driver calls examined for recurrence
DRIVER_DEPTH = 3                 # call depth recorded on the driver thread
MIN_PERIODIC_SCORE = 0.6
WAIT_BUCKETS = frozenset({"Synchronization", "Communication"})
_CROSS_KINDS = frozenset({"p2p", "arrival", "device_wait", "device_sync"})

_HEX_RE = re.compile(r"0x[0-9a-fA-F]{4,}")
_CLONE_RE = re.compile(r"\s*\[clone [^\]]*\]")


def normalize_name(name: str) -> str:
    """Run-independent spelling of a span name: JIT hash names shortened,
    code addresses and compiler clone suffixes removed."""
    n = dash.fmt_kernel_name(name)
    n = _CLONE_RE.sub("", n)
    n = _HEX_RE.sub("0x…", n)
    return n.strip()


@dataclass(frozen=True)
class NodeKey:
    """Identity of a node within its phase."""
    path: tuple[str, ...]          # "category:name" tokens, outermost first
    backend: str
    rank: str
    thread: str
    stream: str = ""
    device: str = ""
    comm: str = ""

    @property
    def leaf(self) -> str:
        return self.path[-1] if self.path else ""

    def roles(self) -> dict[str, str]:
        return {"backend": self.backend, "rank": self.rank, "thread": self.thread,
                "stream": self.stream, "device": self.device, "comm": self.comm}

    def role_text(self) -> str:
        parts = [self.backend, self.rank, self.thread]
        if self.stream:
            parts.append(f"stream {self.stream}")
        if self.device:
            parts.append(f"device {self.device}")
        if self.comm:
            parts.append(f"comm {self.comm}")
        return ", ".join(p for p in parts if p)

    def path_text(self, limit: int = 4) -> str:
        names = [t.split(":", 1)[-1] for t in self.path]
        if len(names) > limit:
            names = ["…"] + names[-limit:]
        return " > ".join(names)


class ProjectedNode:
    __slots__ = ("id", "phase", "key", "category", "raw_name", "bucket", "source",
                 "count", "total_ns", "self_ns", "min_ns", "max_ns", "first_start", "last_end",
                 "queue_ns", "queued", "overlap_ns", "cp_ns", "cp_count", "blame_ns",
                 "threads", "timing")
    # timing: "host", or the device timing source of the node's spans
    # ("device", "proxy_event", ... -- "mixed" if they differ)

    def __init__(self, nid: int, phase: int, key: NodeKey, category: str, raw_name: str,
                 bucket: str) -> None:
        self.id = nid
        self.phase = phase
        self.key = key
        self.category = category
        self.raw_name = raw_name
        self.bucket = bucket
        self.source = ""
        self.count = 0
        self.total_ns = 0
        self.self_ns = 0
        self.min_ns = 0
        self.max_ns = 0
        self.first_start = 0
        self.last_end = 0
        self.queue_ns = 0
        self.queued = 0          # spans carrying a measured queue delay
        self.overlap_ns = 0
        self.cp_ns = 0
        self.cp_count = 0
        self.blame_ns = 0
        self.threads: set = set()
        self.timing = ""

    @property
    def wait_ns(self) -> int:
        return self.self_ns if self.bucket in WAIT_BUCKETS else 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "phase": self.phase, "path": list(self.key.path), **self.key.roles(),
            "category": self.category, "rawName": self.raw_name, "bucket": self.bucket,
            "source": self.source, "count": self.count, "totalNs": self.total_ns,
            "selfNs": self.self_ns, "queueNs": self.queue_ns if self.queued else None,
            "overlapNs": self.overlap_ns, "criticalNs": self.cp_ns, "blameNs": self.blame_ns,
            "firstStartNs": self.first_start, "lastEndNs": self.last_end,
            "threads": len(self.threads),
        }


@dataclass
class Phase:
    index: int
    kind: str                     # prologue / iteration / epilogue / segment / whole
    label: str
    start_ns: int
    end_ns: int
    iterations: int = 0           # iterations grouped into this phase
    driver_tokens: frozenset = frozenset()
    node_paths: frozenset = frozenset()
    critical_ns: int = 0          # critical-path time (credited + idle) within it

    @property
    def duration_ns(self) -> int:
        return max(0, self.end_ns - self.start_ns)

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index, "kind": self.kind, "label": self.label,
                "startNs": self.start_ns, "endNs": self.end_ns, "durationNs": self.duration_ns,
                "iterations": self.iterations, "criticalNs": self.critical_ns}


@dataclass
class CPSegment:
    node: int                     # node id (-1: span with no node)
    credited_ns: int
    gap_before_ns: int            # idle time on the path before it (blamed on the previous node)
    start_ns: int
    end_ns: int
    edge_kind: str                # kind of the edge into it ("" for the first)
    confidence: str


@dataclass
class TraceProjection:
    command: str
    start_ns: int
    end_ns: int
    phases: list[Phase]
    phase_method: str             # iterations / segments / whole
    anchor: str                   # recurring driver call that starts an iteration
    driver: str                   # "rank0 main" ...
    nodes: list[ProjectedNode]
    edges: list[tuple[int, int, str, int, str]]   # (src node, dst node, kind, count, best confidence)
    critical_path: list[CPSegment]
    critical_total_ns: int
    span_count: int
    availability: dict[str, bool]
    notes: list[str] = field(default_factory=list)

    @property
    def wall_ns(self) -> int:
        return max(0, self.end_ns - self.start_ns)

    def phase_nodes(self, phase: int) -> list[ProjectedNode]:
        idx = self._by_phase()
        return idx.get(phase, [])

    def _by_phase(self) -> dict[int, list[ProjectedNode]]:
        if not hasattr(self, "_phase_cache"):
            d: dict[int, list[ProjectedNode]] = defaultdict(list)
            for n in self.nodes:
                d[n.phase].append(n)
            self._phase_cache = dict(d)
        return self._phase_cache

    def adjacency(self) -> tuple[dict[int, list], dict[int, list]]:
        """(in-edges, out-edges) per node: lists of (other node, kind, count, confidence)."""
        if not hasattr(self, "_adj_cache"):
            ins: dict[int, list] = defaultdict(list)
            outs: dict[int, list] = defaultdict(list)
            for s, d, kind, count, conf in self.edges:
                ins[d].append((s, kind, count, conf))
                outs[s].append((d, kind, count, conf))
            self._adj_cache = (dict(ins), dict(outs))
        return self._adj_cache

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": self.command, "startNs": self.start_ns, "endNs": self.end_ns,
            "wallNs": self.wall_ns, "phaseMethod": self.phase_method, "anchor": self.anchor,
            "driver": self.driver, "phases": [p.to_dict() for p in self.phases],
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [{"src": s, "dst": d, "kind": k, "count": c, "confidence": cf}
                      for s, d, k, c, cf in self.edges],
            "criticalTotalNs": self.critical_total_ns, "availability": self.availability,
            "notes": self.notes,
        }


def build_projection(trace) -> TraceProjection:
    """The projection of `trace`, computed once per trace content."""
    return trace.store.memo(("projection", PROJECTION_VERSION), lambda: _Builder(trace).build())


# ── builder ──────────────────────────────────────────────────────────────────

class _Rec:
    """A driver-thread call recorded for phase detection."""
    __slots__ = ("start", "end", "token", "depth", "children", "kinds")

    def __init__(self, start: int, end: int, token: str, depth: int) -> None:
        self.start, self.end, self.token, self.depth = start, end, token, depth
        self.children: list[int] = []
        self.kinds: set[str] = set()

    @property
    def aug(self) -> str:
        return self.token + ("|" + ",".join(sorted(self.kinds)) if self.kinds else "")


class _Frame:
    __slots__ = ("end", "path", "node", "dur", "child", "ratio")

    def __init__(self, end: int, path: int, node: ProjectedNode, dur: int, ratio: float) -> None:
        self.end, self.path, self.node, self.dur, self.child = end, path, node, dur, 0
        # share of this span's time credited on the critical path: its own
        # credit when it is a path span, else its enclosing span's share
        self.ratio = ratio

    def close(self) -> None:
        own = max(0, self.dur - self.child)
        self.node.self_ns += own
        if self.ratio:
            self.node.cp_ns += int(round(own * self.ratio))


def source_key(source: str) -> str:
    """Run-independent form of a source location: file basename and line
    (two checkouts of the same code live under different directories)."""
    if "/" in source and ":" in source:
        return Path(source.rsplit(":", 1)[0]).name + ":" + source.rsplit(":", 1)[1]
    return source


def _source_of(s) -> str:
    tags = s.tags
    f = tags.get("file")
    if f:
        return f"{f}:{tags.get('line', '?')}"
    sym = tags.get("sym")
    if sym:
        return str(sym)
    if s.stack_frames:
        return str(s.stack_frames[0]).split("|", 1)[0]
    return ""


def _stream_order_key(value: str):
    v = value[1:] if value[:1] in ("n", "q") and value[1:].isdigit() else value
    return (0, int(v)) if v.lstrip("-").isdigit() else (1, value)


class _Builder:
    def __init__(self, trace) -> None:
        self.trace = trace
        self.store = trace.store
        self.names: dict[str, str] = {}
        self.path_parent: list[int] = [-1]
        self.path_token: list[str] = [""]
        self.path_index: dict[tuple[int, str], int] = {}
        self.nodes: list[ProjectedNode] = []
        self.node_index: dict[tuple, ProjectedNode] = {}
        self.comms: dict[int, dict[str, str]] = {}

    # small helpers
    def token(self, s) -> str:
        n = self.names.get(s.name)
        if n is None:
            n = self.names[s.name] = normalize_name(s.name)
        return f"{s.category.value}:{n}"

    def intern(self, parent: int, token: str) -> int:
        k = (parent, token)
        p = self.path_index.get(k)
        if p is None:
            p = len(self.path_parent)
            self.path_parent.append(parent)
            self.path_token.append(token)
            self.path_index[k] = p
        return p

    def path_tuple(self, p: int) -> tuple[str, ...]:
        out = []
        while p > 0:
            out.append(self.path_token[p])
            p = self.path_parent[p]
        return tuple(reversed(out))

    def node(self, phase: int, path: int, backend: str, rank: str, thread: str, stream: str,
             device: str, comm: str, s) -> ProjectedNode:
        k = (phase, path, backend, rank, thread, stream, device, comm)
        n = self.node_index.get(k)
        if n is None:
            key = NodeKey(self.path_tuple(path), backend, rank, thread, stream, device, comm)
            n = ProjectedNode(len(self.nodes), phase, key, s.category.value, s.name,
                              ab.bucket_of_span(s))
            self.nodes.append(n)
            self.node_index[k] = n
        return n

    @staticmethod
    def add(n: ProjectedNode, s, tid: int) -> None:
        dur = s.duration_ns
        if n.count == 0:
            n.min_ns = n.max_ns = dur
            n.first_start = s.start_ns
            n.last_end = s.end_ns
        else:
            n.min_ns = min(n.min_ns, dur)
            n.max_ns = max(n.max_ns, dur)
            n.first_start = min(n.first_start, s.start_ns)
            n.last_end = max(n.last_end, s.end_ns)
        n.count += 1
        n.total_ns += dur
        n.threads.add(tid)
        if not n.source:
            n.source = _source_of(s)

    def phase_at(self, t: int) -> int:
        i = bisect.bisect_right(self.phase_starts, t) - 1
        return min(max(i, 0), len(self.phase_starts) - 1)

    # ── build ────────────────────────────────────────────────────────────
    def build(self) -> TraceProjection:
        store, trace = self.store, self.trace
        n_spans = store.span_count()
        ext = store.span_extent(timed_only=True) or store.span_extent(timed_only=False) or (0, 0)
        self.lo, self.hi = ext
        notes: list[str] = []
        if n_spans == 0:
            return TraceProjection(trace.metadata.command, self.lo, self.hi,
                                   [Phase(0, "whole", "whole run", self.lo, self.hi)], "whole", "",
                                   "", [], [], [], 0, 0,
                                   {"criticalPath": False, "edges": False, "queue": False,
                                    "source": False, "ranks": False, "phases": False},
                                   ["empty trace"])
        self.seqs = store.span_seqs()
        graph = cp.load_or_build_graph(trace)
        self.edges = graph.edges
        self.roles()
        self.detect_phases(notes)
        self.n = len(self.seqs)
        self.node_of = np.full(self.n, -1, dtype=np.int32)
        self.path_of = np.full(self.n, -1, dtype=np.int32)
        self.phase_of = np.full(self.n, -1, dtype=np.int32)
        self.prepare_critical_path(graph)
        cache = getattr(graph.spans, "_cache", None)
        if cache is not None:
            cache.clear()          # path spans fetched for the credit table are not needed again
        del graph
        self.host_pass()
        self.device_pass()
        edges = self.node_edges()
        segments, cp_total = self.finish_critical_path()
        phases = self.finish_phases()
        self.node_index = {}
        self.path_index = {}
        self.node_of = self.path_of = self.phase_of = None
        self.edges = None
        has_queue = any(n.queued for n in self.nodes)
        avail = {
            "criticalPath": bool(segments), "edges": bool(edges), "queue": has_queue,
            "source": any(n.source for n in self.nodes), "ranks": self.has_ranks,
            "phases": self.phase_method != "whole",
            "deviceTiming": any(n.timing == "device" for n in self.nodes),
        }
        if any(n.key.backend.endswith(".device") for n in self.nodes) and not has_queue:
            notes.append("device spans carry no measured queue delay (no CUPTI / ROCprofiler-SDK "
                         "timing): queueing cannot be measured")
        return TraceProjection(
            command=trace.metadata.command, start_ns=self.lo, end_ns=self.hi, phases=phases,
            phase_method=self.phase_method, anchor=self.anchor, driver=self.driver_label,
            nodes=self.nodes, edges=edges, critical_path=segments, critical_total_ns=cp_total,
            span_count=n_spans, availability=avail, notes=notes)

    def ordinal(self, seq: int) -> int:
        return int(np.searchsorted(self.seqs, seq))

    # ── roles ────────────────────────────────────────────────────────────
    def roles(self) -> None:
        store, trace = self.store, self.trace
        infos = store.lane_infos()
        pid_first: dict[int, int] = {}
        for ln in infos:
            pid_first[ln.pid] = min(pid_first.get(ln.pid, ln.min_start), ln.min_start)
        rank: dict[int, str] = {}
        for pid in store.pids():
            for s in itertools.islice(trace.iter_spans(pid=pid, categories=("mpi",)), 64):
                r = s.tags.get("rank")
                if r is not None:
                    rank[pid] = str(r)
                    break
        self.has_ranks = bool(rank)
        order = sorted(store.pids(), key=lambda p: (pid_first.get(p, 0), p))
        self.rank_role = {p: (f"rank{rank[p]}" if p in rank else f"proc{i}") for i, p in enumerate(order)}
        # driver process: lowest rank, else first active process
        if rank:
            self.driver_pid = min(rank, key=lambda p: (int(rank[p]) if str(rank[p]).lstrip("-").isdigit()
                                                      else 1 << 30, p))
        else:
            self.driver_pid = order[0] if order else 0
        # main thread per process: first thread with host activity
        first_host: dict[tuple[int, int], int] = {}
        for pid, tid in store.threads():
            for s in itertools.islice(trace.iter_spans(order="start", pid=pid, tid=tid), 64):
                if s.duration_ns > 0 and not ab.is_device_timed(s):
                    first_host[(pid, tid)] = s.start_ns
                    break
        self.main_tid: dict[int, int] = {}
        for (pid, tid), t in sorted(first_host.items(), key=lambda kv: (kv[1], kv[0])):
            self.main_tid.setdefault(pid, tid)
        # stream roles per process: numeric ids in order, handles by first use
        streams: dict[int, list[tuple[Any, str]]] = defaultdict(list)
        seen: set = set()
        for ln in sorted(infos, key=lambda ln: ln.min_start):
            if ln.kind == "stream" and (ln.pid, ln.ident) not in seen:
                seen.add((ln.pid, ln.ident))
                streams[ln.pid].append(ln.ident)
        self.stream_role: dict[tuple[int, str], str] = {}
        for pid, ids in streams.items():
            numeric = all(_stream_order_key(v)[0] == 0 for v in ids)
            ordered = sorted(ids, key=_stream_order_key) if numeric else ids
            for i, v in enumerate(ordered):
                self.stream_role[(pid, v)] = f"s{i}"

    def comm_role(self, pid: int, s) -> str:
        if s.category.value not in ("mpi", "nccl"):
            return ""
        raw = s.tags.get("commid")
        if raw is None:
            return ""
        raw = str(raw)
        if raw == "-1":
            return "unregistered"
        roles = self.comms.setdefault(pid, {})
        r = roles.get(raw)
        if r is None:
            r = roles[raw] = f"c{len(roles)}"
        return r

    def thread_role(self, pid: int, tid: int) -> str:
        return "main" if self.main_tid.get(pid) == tid else "worker"

    # ── phases ───────────────────────────────────────────────────────────
    def cross_kinds(self) -> dict[int, set[str]]:
        e = self.edges
        out: dict[int, set[str]] = defaultdict(set)
        if not len(e):
            return out
        for code, kind in enumerate(cp.EDGE_KINDS):
            if kind not in _CROSS_KINDS:
                continue
            sel = e[e["kind"] == code]
            for o in itertools.chain(sel["dst"].tolist(), sel["src"].tolist()):
                out[o].add(kind)
        return out

    def detect_phases(self, notes: list[str]) -> None:
        pid = self.driver_pid
        tid = self.main_tid.get(pid)
        self.driver_label = f"{self.rank_role.get(pid, '?')} main thread"
        self.anchor = ""
        recs: list[_Rec] = []
        if tid is not None:
            kinds = self.cross_kinds()
            stack: list[tuple[int, int, int]] = []     # (end, rec index or -1, depth)
            for s in self.trace.iter_spans(order="start", pid=pid, tid=tid):
                if s.duration_ns <= 0 or ab.is_device_timed(s):
                    continue
                while stack and stack[-1][0] <= s.start_ns:
                    stack.pop()
                if stack and stack[-1][0] >= s.end_ns:
                    depth, parent = stack[-1][2] + 1, stack[-1][1]
                else:
                    depth, parent = 0, -1
                ri = -1
                if depth <= DRIVER_DEPTH:
                    ri = len(recs)
                    recs.append(_Rec(s.start_ns, s.end_ns, self.token(s), depth))
                    if parent >= 0 and depth > 0:
                        recs[parent].children.append(ri)
                ks = kinds.get(self.ordinal(s.seq))
                if ks:
                    if ri >= 0:
                        recs[ri].kinds |= ks
                    for _e, r, _d in stack:
                        if r >= 0:
                            recs[r].kinds |= ks
                stack.append((s.end_ns, ri if ri >= 0 else (stack[-1][1] if stack else -1), depth))
        level = [i for i, r in enumerate(recs) if r.depth == 0]
        for _ in range(DRIVER_DEPTH):
            if not level:
                break
            covered = sum(recs[i].end - recs[i].start for i in level) or 1
            dom = max(level, key=lambda i: recs[i].end - recs[i].start)
            if (recs[dom].end - recs[dom].start) >= 0.8 * covered and len(recs[dom].children) >= 2:
                level = recs[dom].children
            else:
                break
        seq = [recs[i] for i in level]
        seq.sort(key=lambda r: r.start)
        self.make_phases(seq, notes)

    def make_phases(self, seq: list[_Rec], notes: list[str]) -> None:
        lo, hi = self.lo, self.hi
        phases: list[Phase] = []
        best = _find_period(seq) if len(seq) >= 2 else None
        if best is not None:
            pos, last_end_idx, anchor = best
            self.anchor = anchor.split("|", 1)[0]
            starts = [seq[p].start for p in pos]
            ends = starts[1:] + [seq[last_end_idx - 1].end]
            c = len(pos)
            block = -(-c // MAX_ITERATION_PHASES)
            if block > 1:
                notes.append(f"{c} iterations grouped into blocks of {block} for alignment")
            if starts[0] > lo:
                phases.append(Phase(0, "prologue", "prologue", lo, starts[0]))
            for b in range(0, c, block):
                k1 = min(c, b + block)
                tokens = frozenset(r.aug for r in seq if starts[b] <= r.start < ends[k1 - 1])
                label = f"iteration {b + 1}" if k1 - b == 1 else f"iterations {b + 1}–{k1}"
                phases.append(Phase(0, "iteration", label, starts[b], ends[k1 - 1], k1 - b, tokens))
            if ends[-1] < hi:
                tail = frozenset(r.aug for r in seq if r.start >= ends[-1])
                phases.append(Phase(0, "epilogue", "epilogue", ends[-1], hi, 0, tail))
            self.phase_method = "iterations"
        elif seq:
            groups: list[list[_Rec]] = []
            for r in seq:
                if groups and groups[-1][-1].aug == r.aug:
                    groups[-1].append(r)
                else:
                    groups.append([r])
            merge = -(-len(groups) // MAX_SEGMENT_PHASES)
            if merge > 1:
                notes.append(f"{len(groups)} top-level segments merged in groups of {merge}")
                groups = [list(itertools.chain.from_iterable(groups[i:i + merge]))
                          for i in range(0, len(groups), merge)]
            if groups[0][0].start > lo:
                phases.append(Phase(0, "prologue", "prologue", lo, groups[0][0].start))
            for gi, g in enumerate(groups):
                end = groups[gi + 1][0].start if gi + 1 < len(groups) else hi
                names = [t.split(":", 1)[-1] for t in dict.fromkeys(r.token for r in g)]
                label = f"segment {gi + 1}: {names[0]}" + (" …" if len(names) > 1 else "") + \
                        (f" ×{len(g)}" if len(g) > 1 and len(names) == 1 else "")
                phases.append(Phase(0, "segment", label, g[0].start, end, 0,
                                    frozenset(r.aug for r in g)))
            self.phase_method = "segments"
        if not phases:
            phases = [Phase(0, "whole", "whole run", lo, hi)]
            self.phase_method = "whole"
        for i, p in enumerate(phases):
            p.index = i
        self.phases = phases
        self.phase_starts = [p.start_ns for p in phases]

    # ── host spans ───────────────────────────────────────────────────────
    def host_pass(self) -> None:
        trace = self.trace
        node_of, path_of, phase_of = self.node_of, self.path_of, self.phase_of
        for pid, tid in self.store.threads():
            rank = self.rank_role.get(pid, "")
            thread = self.thread_role(pid, tid)
            stack: list[_Frame] = []
            for s in trace.iter_spans(order="start", pid=pid, tid=tid):
                dur = s.duration_ns
                if dur <= 0:
                    continue
                o = self.ordinal(s.seq)
                while stack and stack[-1].end <= s.start_ns:
                    stack.pop().close()
                if ab.is_device_timed(s):
                    path_of[o] = stack[-1].path if stack else 0     # launch context
                    continue
                parent = stack[-1] if stack and stack[-1].end >= s.end_ns else None
                path = self.intern(parent.path if parent else 0, self.token(s))
                ph = self.phase_at(s.start_ns)
                n = self.node(ph, path, s.category.value + (".host" if s.category.value in
                                                              ("cuda", "rocm", "opencl") else ""),
                              rank, thread, "", "", self.comm_role(pid, s), s)
                self.add(n, s, tid)
                n.timing = "host"
                node_of[o] = n.id
                path_of[o] = path
                phase_of[o] = ph
                if parent is not None:
                    parent.child += dur
                credit = self.path_credit.get(o)
                ratio = min(1.0, credit / dur) if credit is not None else (parent.ratio if parent else 0.0)
                stack.append(_Frame(s.end_ns, path, n, dur, ratio))
            for f in stack:
                f.close()

    # ── device spans, overlap ────────────────────────────────────────────
    def device_pass(self) -> None:
        """Device nodes, plus the overlap accounting: one start-ordered pass
        per process feeds a sweep per device (concurrency across streams)
        and one for the process's top-level host calls (concurrency across
        threads; per-thread stacks tell which calls are top-level)."""
        e = self.edges
        launch_code = cp.EDGE_KINDS.index("launch")
        sel = e[e["kind"] == launch_code] if len(e) else e
        order = np.argsort(sel["dst"], kind="stable") if len(sel) else np.empty(0, dtype=np.int64)
        l_dst = sel["dst"][order] if len(sel) else np.empty(0, dtype=np.int64)
        l_src = sel["src"][order] if len(sel) else np.empty(0, dtype=np.int64)
        node_of, path_of, phase_of = self.node_of, self.path_of, self.phase_of
        for pid in self.store.pids():
            rank = self.rank_role.get(pid, "")
            sweeps: dict[str, _Sweep] = {}
            host_sweep = _Sweep(self.nodes)
            open_ends: dict[int, list[int]] = defaultdict(list)    # tid -> ends of open host calls
            for s in self.trace.iter_spans(order="start", pid=pid):
                if s.duration_ns <= 0:
                    continue
                o = self.ordinal(s.seq)
                if not ab.is_device_timed(s):
                    ends = open_ends[s.tid]
                    while ends and ends[-1] <= s.start_ns:
                        ends.pop()
                    if not ends or ends[-1] < s.end_ns:      # not inside an open call: top-level
                        nid = int(node_of[o])
                        if nid >= 0:
                            host_sweep.push(s.start_ns, s.end_ns, nid, s.tid)
                    ends.append(s.end_ns)
                    continue
                host = -1
                if len(l_dst):
                    i = int(np.searchsorted(l_dst, o))
                    if i < len(l_dst) and l_dst[i] == o:
                        host = int(l_src[i])
                if host >= 0 and path_of[host] >= 0:
                    parent_path, ph = int(path_of[host]), int(phase_of[host])
                else:
                    parent_path = int(path_of[o]) if path_of[o] >= 0 else 0
                    ph = self.phase_at(s.start_ns)
                tags = s.tags
                stream_raw = tags.get("stream")
                stream = "" if stream_raw is None else self.stream_role.get((pid, str(stream_raw)),
                                                                            f"?{stream_raw}")
                dev = str(tags.get("dev", "")) if tags.get("dev") is not None else ""
                path = self.intern(parent_path, self.token(s))
                n = self.node(ph, path, s.category.value + ".device", rank, "device", stream, dev,
                              self.comm_role(pid, s), s)
                self.add(n, s, s.tid)
                n.self_ns += s.duration_ns
                tm = str(tags.get("timing") or "device")
                n.timing = tm if n.timing in ("", tm) else "mixed"
                q = tags.get("queue_ns")
                if q is not None:
                    try:
                        n.queue_ns += int(q)
                        n.queued += 1
                    except (TypeError, ValueError):
                        pass
                node_of[o] = n.id
                phase_of[o] = ph
                credit = self.path_credit.get(o)
                if credit is not None:
                    n.cp_ns += credit
                sw = sweeps.get(dev)
                if sw is None:
                    sw = sweeps[dev] = _Sweep(self.nodes)
                sw.push(s.start_ns, s.end_ns, n.id, stream or f"t{s.tid}")
            for sw in sweeps.values():
                sw.finish()
            host_sweep.finish()

    # ── node-level edges ─────────────────────────────────────────────────
    def node_edges(self) -> list[tuple[int, int, str, int, str]]:
        e = self.edges
        if not len(e):
            return []
        ns = self.node_of[e["src"]].astype(np.int64)
        nd = self.node_of[e["dst"]].astype(np.int64)
        keep = (ns >= 0) & (nd >= 0) & (ns != nd)
        if not keep.any():
            return []
        ns, nd = ns[keep], nd[keep]
        kind = e["kind"][keep].astype(np.int64)
        conf = e["conf"][keep].astype(np.int64)
        m = max(len(self.nodes), 1)
        code = (ns * m + nd) * 8 + kind
        uniq, inv, counts = np.unique(code, return_inverse=True, return_counts=True)
        best = np.full(len(uniq), 99, dtype=np.int64)
        np.minimum.at(best, inv, conf)
        out = []
        for c, n, b in zip(uniq.tolist(), counts.tolist(), best.tolist()):
            k = c % 8
            pair = c // 8
            out.append((pair // m, pair % m, cp.EDGE_KINDS[k], n, cp.EDGE_CONFS[b]))
        return out

    # ── critical path ────────────────────────────────────────────────────
    def prepare_critical_path(self, graph) -> None:
        """The path (ordinals, time credited to each span, idle gap before
        it), computed before the passes so that a path span's credit can be
        spread over its own and its descendants' exclusive time: a path
        through an enclosing call (an NVTX range, a parallel region) is a
        path through what ran inside it."""
        self.path_credit: dict[int, int] = {}
        self.path_rows: list[tuple] = []
        path, confs, kinds = graph.critical_path()
        if not path:
            return
        report = cp.attribute_blame(graph.spans, path, max(1, self.hi - self.lo), confs, kinds)
        credited = report.path_credited_ns
        spans = graph.spans
        prev = None
        for i, o in enumerate(path):
            s = spans[o]
            gap = max(0, s.start_ns - prev.end_ns) if prev is not None else 0
            own = credited[i] if len(credited) == len(path) else s.duration_ns
            self.path_credit[o] = own
            self.path_rows.append((o, own, gap, s.start_ns, s.end_ns,
                                   kinds[i - 1] if 0 < i <= len(kinds) else "",
                                   confs[i - 1] if 0 < i <= len(confs) else ""))
            prev = s

    def finish_critical_path(self) -> tuple[list[CPSegment], int]:
        """Segments with their nodes; idle time on the path is blamed on the
        node of the span before it (what the path was waiting for)."""
        segments: list[CPSegment] = []
        total = 0
        prev_node = -1
        for o, own, gap, start, end, kind, conf in self.path_rows:
            nid = int(self.node_of[o])
            ph = self.nodes[nid].phase if nid >= 0 else self.phase_at(start)
            if gap and prev_node >= 0:
                self.nodes[prev_node].blame_ns += gap
            if nid >= 0:
                self.nodes[nid].cp_count += 1
            self.phases[ph].critical_ns += own + gap
            total += own + gap
            segments.append(CPSegment(nid, own, gap, start, end, kind, conf))
            prev_node = nid
        self.path_rows = []
        return segments, total

    def finish_phases(self) -> list[Phase]:
        paths: dict[int, set] = defaultdict(set)
        for n in self.nodes:
            paths[n.phase].add(n.key.path)
        for p in self.phases:
            p.node_paths = frozenset(paths.get(p.index, ()))
        return self.phases


class _Sweep:
    """Online overlap accounting for intervals arriving in start order:
    each interval is credited with the time during which an interval on a
    different lane (stream / thread) was also active."""

    def __init__(self, nodes: list[ProjectedNode]) -> None:
        self.nodes = nodes
        self.active: list[tuple[int, int]] = []     # heap (end, id)
        self.info: dict[int, list] = {}             # id -> [node, lane, overlapped]
        self.lanes: Counter = Counter()
        self.t = None
        self.next_id = 0

    def _advance(self, to: int | None) -> None:
        while self.active and (to is None or self.active[0][0] <= to):
            end, i = heapq.heappop(self.active)
            self._credit(end)
            node, lane, ov = self.info.pop(i)
            self.lanes[lane] -= 1
            if not self.lanes[lane]:
                del self.lanes[lane]
            node.overlap_ns += ov
        if to is not None:
            self._credit(to)

    def _credit(self, t: int) -> None:
        if self.t is not None and t > self.t and len(self.lanes) >= 2:
            length = t - self.t
            for rec in self.info.values():
                rec[2] += length
        if self.t is None or t > self.t:
            self.t = t

    def push(self, start: int, end: int, nid: int, lane) -> None:
        self._advance(start)
        i = self.next_id
        self.next_id += 1
        self.info[i] = [self.nodes[nid], lane, 0]
        self.lanes[lane] += 1
        heapq.heappush(self.active, (end, i))

    def finish(self) -> None:
        self._advance(None)


def _jaccard_multiset(a: Counter, b: Counter) -> float:
    keys = set(a) | set(b)
    if not keys:
        return 1.0
    inter = sum(min(a[k], b[k]) for k in keys)
    union = sum(max(a[k], b[k]) for k in keys)
    return inter / union if union else 1.0


def _find_period(seq: list[_Rec]) -> tuple[list[int], int, str] | None:
    """(anchor positions, end index of the last iteration, anchor token) for
    the recurring call that best splits `seq` into similar iterations, or
    None. Each candidate anchor (a call that recurs) cuts the sequence at
    its occurrences; it scores (share of iterations whose call multiset is
    >= 80% similar to the most common one) x (share of the driver's time
    covered by the iterations); ties go to the anchor whose first
    occurrence comes earliest (smallest prologue)."""
    seq = seq[:MAX_PERIOD_TOKENS]
    toks = [r.aug for r in seq]
    n = len(toks)
    total = max(1, seq[-1].end - seq[0].start)
    best = None
    for anchor, c in Counter(toks).most_common(32):
        if c < 2:
            break
        pos = [i for i, t in enumerate(toks) if t == anchor]
        lengths = [pos[k + 1] - pos[k] for k in range(c - 1)]
        L = int(median(lengths))
        last_end = min(n, pos[-1] + L)
        segs = [Counter(toks[pos[k]:pos[k + 1]]) for k in range(c - 1)] + [Counter(toks[pos[-1]:last_end])]
        sigs = Counter(tuple(sorted(sg.items())) for sg in segs)
        modal = Counter(dict(sigs.most_common(1)[0][0]))
        regular = sum(1 for sg in segs if _jaccard_multiset(sg, modal) >= 0.8) / len(segs)
        covered = (seq[last_end - 1].end - seq[pos[0]].start) / total
        score = regular * covered
        rank = (round(score, 6), -pos[0])
        if best is None or rank > best[0]:
            best = (rank, pos, last_end, anchor)
    if best is None or best[0][0] < MIN_PERIODIC_SCORE:
        return None
    _rank, pos, last_end, anchor = best
    # Start iterations at their real beginning: the anchor is the most
    # regular recurring call, not necessarily the first call of an
    # iteration ([launch, launch, sync] cut at the sync). Move every cut
    # back over the calls that consistently precede each occurrence.
    gaps = [pos[0]] + [pos[k + 1] - pos[k] - 1 for k in range(len(pos) - 1)]
    k = 0
    while k < min(gaps) and len({tuple(toks[p - k - 1:p]) for p in pos}) == 1:
        k += 1
    if k:
        pos = [p - k for p in pos]
        period = int(median([pos[i + 1] - pos[i] for i in range(len(pos) - 1)]))
        last_end = min(n, pos[-1] + period)
    return pos, last_end, anchor
