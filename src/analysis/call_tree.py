"""
Call-tree construction: from captured CPU stack frames when available,
temporal containment otherwise. Shared by the TUI's CallTreeWidget
(src/ui/app.py) and the GUI's CallTreeBridge (src/gui/bridge.py) --
extracted here (rather than left in src/ui/app.py, where it originated)
so the GUI doesn't pay for importing the whole Textual-based TUI module
(textual+rich, ~180ms) just to reuse this pure trace-analysis logic. See
src/ui/app.py's re-export of these same names for why existing call
sites there needed no changes.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from ..core.events import SpanEvent


@dataclass
class _CTNode:
    """Aggregated call-tree node: N spans with the same name at the same tree level."""
    name: str
    category: str
    total_ns: int
    count: int
    self_ns: int
    children: list["_CTNode"] = field(default_factory=list)

    @property
    def avg_ns(self) -> int:
        return self.total_ns // max(self.count, 1)


@dataclass
class _RawNode:
    span: SpanEvent
    children: list["_RawNode"] = field(default_factory=list)


def _ct_build_raw(spans: list[SpanEvent]) -> list[_RawNode]:
    """Containment tree for one thread via stack algorithm (O(n log n)).

    When spans carry explicit span_id / parent_span_id links (emitted by hooks),
    those take precedence over temporal containment for spans that stay within
    the same input set.
    """
    sorted_spans = sorted(spans, key=lambda s: s.start_ns)
    # Map from span_id → node for explicit linking.
    nodes_by_sid: dict[str, _RawNode] = {
        s.span_id: _RawNode(span=s)
        for s in sorted_spans
        if s.span_id
    }
    # Remaining spans without a span_id get plain nodes.
    all_nodes: dict[int, _RawNode] = {}  # id(span) → node
    for s in sorted_spans:
        if s.span_id and s.span_id in nodes_by_sid:
            all_nodes[id(s)] = nodes_by_sid[s.span_id]
        else:
            all_nodes[id(s)] = _RawNode(span=s)

    explicitly_parented: set[int] = set()  # id(span) of spans handled via explicit link
    for s in sorted_spans:
        if s.parent_span_id and s.parent_span_id in nodes_by_sid:
            parent_node = nodes_by_sid[s.parent_span_id]
            child_node  = all_nodes[id(s)]
            if child_node is not parent_node:
                parent_node.children.append(child_node)
                explicitly_parented.add(id(s))

    # Temporal containment for spans not already parented explicitly.
    # Explicitly-parented spans are still pushed onto the stack so their
    # temporally-contained children nest under them correctly.
    stack: list[_RawNode] = []
    roots: list[_RawNode] = []
    for s in sorted_spans:
        while stack and stack[-1].span.end_ns <= s.start_ns:
            stack.pop()
        node = all_nodes[id(s)]
        if id(s) not in explicitly_parented:
            if stack and stack[-1].span.end_ns >= s.end_ns:
                stack[-1].children.append(node)
            else:
                roots.append(node)
        stack.append(node)
    return roots


def _ct_aggregate(raw_nodes: list[_RawNode]) -> list[_CTNode]:
    """Recursively aggregate siblings by (name, category)."""
    groups: dict[tuple[str, str], list[_RawNode]] = defaultdict(list)
    for rn in raw_nodes:
        groups[(rn.span.name, rn.span.category.value)].append(rn)

    result: list[_CTNode] = []
    for (name, cat), nodes in groups.items():
        total_ns = sum(n.span.duration_ns for n in nodes)
        all_children: list[_RawNode] = []
        for n in nodes:
            all_children.extend(n.children)
        children = _ct_aggregate(all_children)
        # self_ns = this node's cumulative duration minus its children's --
        # correct when children are strictly nested within their parent's
        # own interval (the normal call-stack case), but when a node's
        # invocations spawn CONCURRENT children (e.g. nested OpenMP
        # parallel regions/tasks that genuinely overlap in wall time), the
        # children's summed duration can exceed the parent's, and max(0, …)
        # silently floors self_ns to 0 rather than the small positive value
        # truer interval-aware accounting would show. Kept as a safe,
        # non-crashing floor rather than negative/nonsensical self time;
        # a fully correct fix would need to redefine "self time" under
        # concurrent (not just sequential) children, a bigger design
        # decision than this pass's bug fixes.
        self_ns = max(0, total_ns - sum(c.total_ns for c in children))
        result.append(_CTNode(
            name=name, category=cat,
            total_ns=total_ns, count=len(nodes),
            self_ns=self_ns,
            children=sorted(children, key=lambda c: -c.total_ns),
        ))
    return sorted(result, key=lambda n: -n.total_ns)


@dataclass
class _StackNode:
    """Mutable trie node used while building the stack-based call tree."""
    name: str
    category: str
    total_ns: int = 0
    count: int = 0          # spans that end here (leaves)
    children: "dict[str, _StackNode]" = field(default_factory=dict)

    def to_ctnode(self) -> "_CTNode":
        children = sorted(
            (c.to_ctnode() for c in self.children.values()),
            key=lambda n: -n.total_ns,
        )
        self_ns = max(0, self.total_ns - sum(c.total_ns for c in children))
        return _CTNode(
            name=self.name, category=self.category,
            total_ns=self.total_ns, count=self.count,
            self_ns=self_ns, children=children,
        )


def _ct_build_from_stacks(spans: list[SpanEvent]) -> list[_CTNode]:
    """Build call tree from captured CPU stack frames (from-main view).

    Walks each span's full root-first path -- its reversed stack_frames
    (the caller chain) PLUS its own name appended as the definite final
    leaf -- in one pass, with one dict key per level (the bare function
    name). This used to be two separate steps: a frame-only path walk
    keyed by bare name, then a second "add the span's own name as a
    leaf" step keyed by f"__leaf__{name}" -- a DIFFERENT key for what is
    often the SAME logical node. Any function that is both an
    intermediate ancestor frame for some spans and its own separately-
    measured leaf span for others (an extremely common pattern: a
    function that does direct work AND calls sub-functions, e.g. a
    driving loop that also does some work inline) never merged, and
    showed up as two same-named sibling nodes at the same tree level
    instead of one aggregated node -- contradicting _CTNode's own
    docstring ("N spans with the same name at the same tree level").
    Found via the Qt/QML GUI's Call Tree screen using a deliberately
    two-level-deep synthetic trace; the TUI's Call Tree tab shares this
    exact function, so it had the same bug whenever real profiled code
    hit this shape, not just the GUI.
    """
    roots: dict[str, _StackNode] = {}

    for span in spans:
        # Frames arrive innermost-first (backtrace order); reverse to
        # root-first, then the span's own name is the guaranteed final
        # leaf of its own path.
        full_path = list(reversed(span.stack_frames)) + [span.name]

        level = roots
        for i, frame in enumerate(full_path):
            is_leaf = (i == len(full_path) - 1)
            if frame not in level:
                level[frame] = _StackNode(
                    name=frame, category=span.category.value if is_leaf else "other")
            node = level[frame]
            node.total_ns += span.duration_ns
            if is_leaf:
                node.count += 1
                # Promote category in case this node was created earlier
                # only as an ancestor frame (category "other") by some
                # other span that passed through it.
                node.category = span.category.value
            level = node.children

    return sorted((n.to_ctnode() for n in roots.values()), key=lambda n: -n.total_ns)


def _ct_build(spans: list[SpanEvent]) -> list[_CTNode]:
    """Call tree: uses CPU stack frames when available, temporal containment otherwise.

    Explicit parent_span_id links (from hooks) are preferred over temporal
    containment within the same thread, and are shown as cross-thread connections
    in the tree when the parent and child live on different threads.
    """
    duration_spans = [s for s in spans if s.duration_ns > 0]
    if not duration_spans:
        return []

    stacked = [s for s in duration_spans if s.stack_frames]
    if stacked:
        return _ct_build_from_stacks(stacked)

    # Per-thread temporal containment (explicit same-thread links handled inside).
    by_thread: dict[tuple[int, int], list[SpanEvent]] = defaultdict(list)
    for s in duration_spans:
        by_thread[(s.pid, s.tid)].append(s)

    if len(by_thread) == 1:
        return _ct_aggregate(_ct_build_raw(next(iter(by_thread.values()))))

    # Build per-thread roots first.
    thread_roots: dict[tuple[int, int], list[_CTNode]] = {}
    for (pid, tid), thread_spans in sorted(by_thread.items()):
        thread_roots[(pid, tid)] = _ct_aggregate(_ct_build_raw(thread_spans))

    # Identify spans that are explicit children of spans in a *different* thread.
    # We use the span_id → (pid, tid) map to detect cross-thread links.
    sid_to_thread: dict[str, tuple[int, int]] = {
        s.span_id: (s.pid, s.tid)
        for s in duration_spans
        if s.span_id
    }
    cross_thread_children: dict[tuple[int, int], list[SpanEvent]] = defaultdict(list)
    orphan_threads: set[tuple[int, int]] = set()
    for s in duration_spans:
        if s.parent_span_id:
            parent_thread = sid_to_thread.get(s.parent_span_id)
            my_thread = (s.pid, s.tid)
            if parent_thread and parent_thread != my_thread:
                cross_thread_children[parent_thread].append(s)
                orphan_threads.add(my_thread)

    all_roots: list[_CTNode] = []
    for (pid, tid), children in sorted(thread_roots.items()):
        total_ns = sum(c.total_ns for c in children)
        # Attach cross-thread children (e.g. GPU kernels under their NVTX parent thread).
        extra = cross_thread_children.get((pid, tid), [])
        if extra:
            extra_nodes = _ct_aggregate(_ct_build_raw(extra))
            children = children + extra_nodes
            total_ns = sum(c.total_ns for c in children)
        all_roots.append(_CTNode(
            name=f"Thread {tid} (pid {pid})",
            category="other",
            total_ns=total_ns, count=1, self_ns=0,
            children=sorted(children, key=lambda n: -n.total_ns),
        ))
    return sorted(all_roots, key=lambda n: -n.total_ns)
