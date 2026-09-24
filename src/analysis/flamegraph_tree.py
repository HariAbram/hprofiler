"""
Flame-graph tree construction: reuses analysis/call_tree.py's existing
_ct_build() -- the same dispatcher already backing the Call Tree tab
(TUI's CallTreeWidget, GUI's CallTreeBridge) -- rather than a separate
tree-building implementation. _ct_build_from_stacks() (what _ct_build
delegates to whenever any span carries stack_frames) already produces a
category-aware, INCLUSIVE-time tree: a node's total_ns accumulates
through every descendant, exactly flame-graph width semantics (a frame's
width = its own time + everything it called). No separate aggregation
logic to keep in sync with the Call Tree tab's.

Works from whatever combination of spans currently carries stack_frames
-- CPU samples from a --perf-callgraph run, hook-captured GPU/MPI/OpenMP
API calls from --call-tree, or both together in one trace -- same
behavior as the Call Tree tab, deliberately not narrowed to CPU-only.
"""
from __future__ import annotations

from typing import Any

from ..core.events import SpanEvent
from .call_tree import _CTNode, _ct_build


def _ctnode_to_flame(node: _CTNode) -> dict[str, Any]:
    return {
        "name": node.name,
        "value": node.total_ns,
        "category": node.category,
        "children": [_ctnode_to_flame(c) for c in node.children],
    }


def build_flame_tree(spans: list[SpanEvent]) -> dict[str, Any]:
    """{name, value, category, children} tree for the Flame Graph tab
    (TUI and GUI). Multiple roots (common: several distinct top-level
    call trees, e.g. separate threads with no shared ancestor) are
    wrapped under one synthetic "all" root, same convention
    output/flamegraph.py's own _build_tree used. Empty input (no spans
    with stack_frames -- --perf-callgraph/--call-tree wasn't used for
    this run) returns a zero-value "all" node with no children, not an
    error -- same "no data" convention the Call Tree tab already uses."""
    roots = [_ctnode_to_flame(r) for r in _ct_build(spans)]
    return {
        "name": "all",
        "value": sum(r["value"] for r in roots),
        "category": "other",
        "children": roots,
    }
