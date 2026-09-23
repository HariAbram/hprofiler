"""
Call GRAPH construction: unlike analysis/call_tree.py (a tree -- the same
function called from two different callers becomes two separate nodes,
by design, so total time down each branch is unambiguous), this merges
every occurrence of a function into ONE node regardless of who called
it, with caller->callee relationships represented as edges. Built for
the GUI Timeline's call-graph panel (a node-and-edge diagram of what's
currently visible), which is a genuinely different question from the
Call Tree tab's ("what's the time breakdown down each call path") --
this answers "which functions call which, overall", not "how expensive
is each specific call path".

Reuses the exact same span.stack_frames walk (reversed, root-first, own
name as guaranteed leaf) as call_tree.py's _ct_build_from_stacks, for
the same reason: one shared understanding of "what a span's call path
is", not two that could silently disagree.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..core.events import SpanEvent


@dataclass
class CallGraphNode:
    name: str
    category: str = "other"   # "other" for a pure ancestor frame never
                               # directly measured as a leaf (e.g. "main") --
                               # mirrors call_tree.py's _StackNode convention
    total_ns: int = 0         # self/leaf time only (spans where this frame
                               # IS the measured span, not an ancestor) --
                               # same semantics as Trace.aggregated_stats(),
                               # for consistency with the Kernels tab
    count: int = 0            # leaf-span count, same scope as total_ns


@dataclass
class CallGraphEdge:
    caller: str
    callee: str
    count: int = 0     # number of spans whose path had this caller->callee
                        # adjacency (i.e. the edge was "traversed")
    total_ns: int = 0  # sum of duration_ns for spans traversing this edge --
                        # an edge-thickness/color-weight signal, not a
                        # separate timing measurement of the edge itself


def build_call_graph(spans: list[SpanEvent]) -> tuple[list[CallGraphNode], list[CallGraphEdge]]:
    """Build a call graph from spans with captured stack frames (spans
    without any are silently skipped, same as call_tree.py's behavior --
    HPROFILER_CALLSTACK/--call-tree wasn't enabled for this run).
    Returns (nodes, edges), both sorted by total_ns/count descending so
    a caller capping the result to the top N can just slice."""
    node_stats: dict[str, CallGraphNode] = {}
    edge_stats: dict[tuple[str, str], CallGraphEdge] = {}

    def _node(name: str) -> CallGraphNode:
        if name not in node_stats:
            node_stats[name] = CallGraphNode(name=name)
        return node_stats[name]

    for span in spans:
        if span.duration_ns <= 0 or not span.stack_frames:
            continue
        path = list(reversed(span.stack_frames)) + [span.name]

        for frame in path:
            _node(frame)  # ensure every frame on the path has a node,
                           # even pure-ancestor frames with no leaf time

        leaf = _node(span.name)
        leaf.total_ns += span.duration_ns
        leaf.count += 1
        leaf.category = span.category.value

        for i in range(len(path) - 1):
            caller, callee = path[i], path[i + 1]
            key = (caller, callee)
            if key not in edge_stats:
                edge_stats[key] = CallGraphEdge(caller=caller, callee=callee)
            edge = edge_stats[key]
            edge.count += 1
            edge.total_ns += span.duration_ns

    nodes = sorted(node_stats.values(), key=lambda n: -n.total_ns)
    edges = sorted(edge_stats.values(), key=lambda e: -e.total_ns)
    return nodes, edges


def layout_call_graph(
    nodes: list[CallGraphNode], edges: list[CallGraphEdge], max_nodes: int = 30,
) -> dict:
    """Cap to the top `max_nodes` (by total_ns) and assign each a
    normalized [0,1] (x, y) position via a simple layered/BFS-depth
    layout -- x = layer (BFS distance from a root, root = a node with no
    incoming edge among the kept set), y = position within its layer.
    Cycles (recursion) don't break this: a node only ever gets a layer
    the FIRST time it's reached by the BFS, so a back-edge into an
    already-layered node is simply drawn as a (possibly leftward-
    pointing) edge, not specially handled -- deliberately simple, this
    is a compact overview panel, not a general-purpose graph-drawing
    engine.

    Returns {"nodes": [{name, category, totalNs, count, x, y,
                        layer, layerIndex}, ...],
             "edges": [{caller, callee, count, totalNs,
                        callerIdx, calleeIdx}, ...],
             "numLayers": N, "maxLayerSize": M, "truncated": K}
    keyed by index into the returned "nodes" list (QML-friendly: no
    string lookups needed at paint time). `x`/`y` are normalized [0,1]
    for a caller that just wants "roughly where" (e.g. a quick
    minimap); `layer`/`layerIndex` are the raw, un-normalized BFS
    layer and within-layer position, alongside `numLayers`/
    `maxLayerSize` -- together these let a renderer give every node a
    FIXED pixel size and spacing regardless of how many total
    layers/nodes there are (normalized x/y alone can't do this: cramming
    a layer of 10 nodes into the same fixed pixel height as a layer of
    2 is exactly what made nodes visually overlap in the GUI's call-graph
    panel before this field existed), sizing a scrollable canvas to fit
    instead of squeezing everything into one fixed viewport.
    """
    kept = nodes[:max_nodes]
    kept_names = {n.name for n in kept}
    name_to_idx = {n.name: i for i, n in enumerate(kept)}

    kept_edges = [e for e in edges if e.caller in kept_names and e.callee in kept_names
                  and e.caller != e.callee]

    incoming: dict[str, int] = {n.name: 0 for n in kept}
    adjacency: dict[str, list[str]] = {n.name: [] for n in kept}
    for e in kept_edges:
        incoming[e.callee] += 1
        adjacency[e.caller].append(e.callee)

    roots = [n.name for n in kept if incoming[n.name] == 0]
    if not roots and kept:
        # Pure cycle with nothing pointing "in" from outside the kept
        # set (unusual, but possible after capping to max_nodes) --
        # seed the BFS from the single hottest node instead of leaving
        # everything unlayered.
        roots = [kept[0].name]

    layer_of: dict[str, int] = {}
    frontier = list(roots)
    for name in frontier:
        layer_of[name] = 0
    while frontier:
        nxt: list[str] = []
        for name in frontier:
            for callee in adjacency.get(name, []):
                if callee not in layer_of:
                    layer_of[callee] = layer_of[name] + 1
                    nxt.append(callee)
        frontier = nxt
    # Any node never reached (disconnected from every root, e.g. isolated
    # by capping) still needs a position -- park it at layer 0.
    for n in kept:
        layer_of.setdefault(n.name, 0)

    by_layer: dict[int, list[CallGraphNode]] = {}
    for n in kept:
        by_layer.setdefault(layer_of[n.name], []).append(n)
    for layer_nodes in by_layer.values():
        layer_nodes.sort(key=lambda n: -n.total_ns)

    max_layer = max(layer_of.values()) if kept else 0
    max_layer_size = max((len(v) for v in by_layer.values()), default=0)
    out_nodes = [None] * len(kept)
    for layer, layer_nodes in by_layer.items():
        x = layer / max_layer if max_layer > 0 else 0.5
        n_in_layer = len(layer_nodes)
        for i, n in enumerate(layer_nodes):
            y = (i + 0.5) / n_in_layer
            out_nodes[name_to_idx[n.name]] = {
                "name": n.name, "category": n.category,
                "totalNs": n.total_ns, "count": n.count,
                "x": x, "y": y,
                "layer": layer, "layerIndex": i,
            }

    out_edges = [
        {
            "caller": e.caller, "callee": e.callee,
            "count": e.count, "totalNs": e.total_ns,
            "callerIdx": name_to_idx[e.caller], "calleeIdx": name_to_idx[e.callee],
        }
        for e in kept_edges
    ]
    return {
        "nodes": out_nodes, "edges": out_edges,
        "numLayers": max_layer + 1 if kept else 0,
        "maxLayerSize": max_layer_size,
        "truncated": len(nodes) - len(kept),
    }
