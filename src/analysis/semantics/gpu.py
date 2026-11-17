"""
CUDA / ROCm stream semantics beyond FIFO order (criticalpath._add_gpu_edges
uses these):

  * host functions (cudaLaunchHostFunc, cudaStreamAddCallback,
    cuLaunchHostFunc): stream-ordered work, run by the runtime on its own
    callback thread one at a time;
  * a synchronous copy that blocks the host (blocking=1, the hook's reading
    of CUDA's API synchronization behaviour) returns after its device copy;
  * the legacy default stream (stream 0) synchronizes with every blocking
    stream of its device: its work starts after their earlier work and
    their later work starts after it. A stream created non-blocking
    (cudaStreamNonBlocking / hipStreamNonBlocking: nonblocking=1 on its
    creation) is exempt. A stream's flags come from its creation call, from
    CUPTI's stream-created callback or from a query when the hook first saw
    it in use (stream_info); a stream with none of these is assumed blocking
    (cudaStreamCreate's default) at "medium". Programs compiled for the
    per-thread default stream break this assumption; a stream handle reused
    after cudaStreamDestroy gets another stream= number (cuda_hook.c), so two
    lives of one address are two streams;
  * a CUDA graph launch's work: the exec's own edges (graph_exec /
    graph_nodes / graph_edges instants, cupti_trace.c), between the nodes
    that left device records, through those that did not (empty, host,
    event, child-graph markers): graph_dependencies().
"""
from __future__ import annotations

from collections import defaultdict

STREAM_ORDERED_EXTRA_OPS = frozenset({"host_func"})


def stream_flags(spans, items) -> tuple[set[tuple], set[tuple]]:
    """((pid, rt, stream) created, (pid, rt, stream) created non-blocking)
    from the stream-creation calls."""
    created, nonblocking = set(), set()
    for _i, s in items(spans):
        t = s.tags
        # stream_create: the creating call; stream_info: CUPTI's stream-created
        # callback or a one-time query of a stream first seen in use
        if t.get("op") in ("stream_create", "stream_info") and t.get("stream") not in (None, "0") \
                and (t.get("op") == "stream_create" or "nonblocking" in t):
            key = (s.pid, t.get("rt"), t["stream"])
            created.add(key)
            if t.get("nonblocking") == "1":
                nonblocking.add(key)
    return created, nonblocking


def host_function_order(spans, items) -> list[tuple[int, int]]:
    """(earlier, later) consecutive host-function executions of one callback
    thread on different streams (same-stream order is the stream's)."""
    by_thread: dict[tuple, list[int]] = defaultdict(list)
    for i, s in items(spans):
        if s.tags.get("side") == "gpu" and s.tags.get("op") == "host_func":
            by_thread[(s.pid, s.tid)].append(i)
    out = []
    for lst in by_thread.values():
        lst.sort(key=lambda i: spans[i].start_ns)
        for a, b in zip(lst, lst[1:]):
            if spans[a].tags.get("stream") != spans[b].tags.get("stream"):
                out.append((a, b))
    return out


# ── CUDA graphs ──────────────────────────────────────────────────────────────

GRAPH_STRUCTURE_TYPES = frozenset({"graph_exec", "graph_nodes", "graph_edges"})


def graph_structures(instants) -> dict[tuple[int, str], dict]:
    """(pid, gexec) -> {"edges": [(a, b)] full dependencies, "weak": n
    (programmatic / launch-order edges: no completion order, not used),
    "types": {node: CUgraphNodeType}, "usable": bool (lossy / truncated
    snapshots are not), "header": the graph_exec tags} from the
    graph_exec / graph_nodes / graph_edges instants."""
    out: dict[tuple[int, str], dict] = {}
    for ev in instants:
        t = ev.tags
        typ = t.get("type")
        if typ not in GRAPH_STRUCTURE_TYPES or not t.get("gexec"):
            continue
        g = out.setdefault((ev.pid, str(t["gexec"])), {"edges": [], "weak": 0, "types": {}, "header": None})
        if typ == "graph_exec":
            g["header"] = dict(t)
        elif typ == "graph_nodes":
            for e in str(t.get("nodes", "")).split(";"):
                node, _, ty = e.partition("/")
                if node:
                    g["types"][node] = int(ty) if ty.lstrip("-").isdigit() else -1
        else:
            for e in str(t.get("edges", "")).split(";"):
                if not e:
                    continue
                weak = e.endswith("p")
                a, _, b = e.rstrip("p").partition("-")
                if not a or not b:
                    continue
                if weak:
                    g["weak"] += 1
                else:
                    g["edges"].append((a, b))
    for g in out.values():
        h = g["header"] or {}
        g["usable"] = bool(h) and h.get("lossy") != "1" and h.get("truncated") != "1"
    return out


def graph_dependencies(members: dict[str, int], struct: dict, cache: dict) -> list[tuple[int, int]]:
    """[(source, destination)] span indexes for one graph launch: each
    recorded node (members: gnode -> span index) after its nearest recorded
    ancestors along the exec's full-dependency edges, passing through nodes
    without a record in this launch (empty, host, event and marker nodes,
    disabled nodes, and lost records -- through those the order still
    holds). `cache` keeps the node-level answer per (exec, recorded set)."""
    key = (id(struct), frozenset(members))
    pairs = cache.get(key)
    if pairs is None:
        preds_of: dict[str, list[str]] = struct.get("_preds")
        if preds_of is None:
            preds_of = defaultdict(list)
            for a, b in struct["edges"]:
                preds_of[b].append(a)
            struct["_preds"] = preds_of
        pairs = []
        for node in members:
            stack, seen = list(preds_of.get(node, ())), set()
            while stack:
                x = stack.pop()
                if x in seen:
                    continue
                seen.add(x)
                if x in members:
                    pairs.append((x, node))
                else:
                    stack.extend(preds_of.get(x, ()))
        cache[key] = pairs
    return [(members[a], members[b]) for a, b in pairs]


def dependency_diagnostics(trace) -> dict:
    """GPU dependence-capture holes (flat keys, gpu_*):

      gpu_graph_launches               CUDA graph launches recorded
      gpu_graph_launches_unstructured  of those, without the exec's structure
                                       (no gexec=: native tracing off; or no
                                       graph_exec record: the exec was made
                                       before tracing began) -- their nodes
                                       are ordered by the launch stream only
      gpu_graph_unusable               execs whose snapshot is lossy (edge data
                                       the driver would not give) or truncated
      gpu_graph_weak_edges             programmatic / launch-order edges (the
                                       downstream node may start early): not
                                       used as dependencies
      gpu_graph_unmapped_nodes         exec nodes without a clone pairing (no
                                       device record can name them)
      gpu_streams_queried              streams first seen in use, flags queried
      gpu_handle_table_full            stream / event handles the hook could not
                                       give a life (their reuse is not told apart)
    Empty when there is nothing to report."""
    structs = graph_structures(trace.iter_instants())
    out: dict = {}
    launches = unstructured = 0
    for s in trace.iter_spans(categories=("cuda",)):
        t = s.tags
        if t.get("op") == "graph" and t.get("side") == "cpu":
            launches += 1
            g = structs.get((s.pid, str(t.get("gexec")))) if t.get("gexec") else None
            if g is None or not g["usable"]:
                unstructured += 1
        elif t.get("op") == "stream_info" and t.get("src") == "queried":
            out["gpu_streams_queried"] = out.get("gpu_streams_queried", 0) + 1
    if launches:
        out["gpu_graph_launches"] = launches
        if unstructured:
            out["gpu_graph_launches_unstructured"] = unstructured
    unusable = sum(1 for g in structs.values() if g["header"] and not g["usable"])
    weak = sum(g["weak"] for g in structs.values())
    unmapped = sum(int((g["header"] or {}).get("unmapped", 0)) for g in structs.values())
    for k, v in (("gpu_graph_unusable", unusable), ("gpu_graph_weak_edges", weak),
                 ("gpu_graph_unmapped_nodes", unmapped)):
        if v:
            out[k] = v
    for c in trace.iter_counters():
        if c.name == "cuda_handle_table_full" and c.value:
            out["gpu_handle_table_full"] = out.get("gpu_handle_table_full", 0) + int(c.value)
    if set(out) <= {"gpu_graph_launches", "gpu_streams_queried"}:
        return {}
    return out


def dependency_warnings(diag: dict) -> list[str]:
    out = []
    if diag.get("gpu_graph_launches_unstructured"):
        out.append(f"{diag['gpu_graph_launches_unstructured']} of {diag.get('gpu_graph_launches', 0)} CUDA graph "
                   "launch(es) have no recorded graph structure (native tracing off, or the exec was instantiated "
                   "before tracing began): their nodes are ordered only by the launch stream, not by the graph's edges")
    if diag.get("gpu_graph_unusable"):
        out.append(f"{diag['gpu_graph_unusable']} CUDA graph exec(s) recorded an incomplete structure (lossy / "
                   "truncated): no edges are taken from them")
    if diag.get("gpu_graph_weak_edges"):
        out.append(f"{diag['gpu_graph_weak_edges']} CUDA graph edge(s) are programmatic / launch-order edges (the "
                   "downstream node may start before the upstream one ends): not used as completion dependencies")
    if diag.get("gpu_handle_table_full"):
        out.append(f"{diag['gpu_handle_table_full']} CUDA stream / event handle use(s) found the hook's handle table "
                   "full: a reused handle of those is not told apart from its earlier life")
    return out
