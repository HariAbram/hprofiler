"""
Hierarchical rows of the Timeline -- Qt-free.

    System
    └── Node                  (merged multi-node traces: the `node` tag)
        └── Process / MPI rank
            ├── Thread T<n>   host thread: one sub-row per runtime that
            │   ├── mpi       recorded calls on it (lanes are per category
            │   └── openmp    and thread in the trace store)
            └── Device <d>    device work: per device (`dev`/`hdev`), then
                └── stream    per queue / stream, then per runtime

A level is only shown when it carries information: one node is not a row,
a thread or stream with a single lane is one row ("T3 · openmp"), lanes
without a thread or stream (legacy bare-category lanes) sit directly under
their process. Every lane appears exactly once; header rows carry the lane
indexes of everything under them (their aggregate strip is drawn from
those).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Order of runtime sub-rows under one thread / stream.
_CATEGORY_ORDER = {c: i for i, c in enumerate(
    ("mpi", "nccl", "openmp", "sycl", "cuda", "rocm", "opencl", "memory", "jit", "cpu", "sync",
     "sched", "other", "nvtx"))}


@dataclass
class LaneMeta:
    index: int
    name: str
    category: str
    kind: str              # thread / stream / device / ""
    ident: str
    pid: int
    count: int
    tid: int | None = None
    rank: str | None = None
    node: str | None = None
    device: str | None = None
    thread_label: str = ""


@dataclass
class TreeNode:
    id: str
    kind: str              # system / node / process / thread / device / stream / lane
    label: str
    sort: tuple = ()
    lane: int = -1
    category: str = ""
    children: list["TreeNode"] = field(default_factory=list)

    def lanes(self) -> list[int]:
        if self.lane >= 0:
            return [self.lane]
        out: list[int] = []
        for c in self.children:
            out.extend(c.lanes())
        return out


def _cat_key(cat: str) -> int:
    return _CATEGORY_ORDER.get(cat, 50)


def build_tree(lanes: list[LaneMeta], *, system_label: str = "System") -> TreeNode:
    root = TreeNode("system", "system", system_label)
    nodes: dict[str, TreeNode] = {}
    procs: dict[int, TreeNode] = {}
    threads: dict[tuple[int, int], TreeNode] = {}
    devices: dict[tuple[int, str], TreeNode] = {}
    streams: dict[tuple[int, str, str], TreeNode] = {}
    multi_node = len({ln.node for ln in lanes}) > 1

    def process(ln: LaneMeta) -> TreeNode:
        p = procs.get(ln.pid)
        if p is None:
            label = f"rank {ln.rank} · pid {ln.pid}" if ln.rank is not None else f"pid {ln.pid}"
            rank_sort = int(ln.rank) if ln.rank is not None and str(ln.rank).lstrip("-").isdigit() else 1 << 30
            p = procs[ln.pid] = TreeNode(f"p{ln.pid}", "process", label, (rank_sort, ln.pid))
            parent = root
            if multi_node:
                key = ln.node if ln.node is not None else "?"
                parent = nodes.get(key)
                if parent is None:
                    parent = nodes[key] = TreeNode(f"n{key}", "node", f"node {key}", (str(key),))
                    root.children.append(parent)
            parent.children.append(p)
        return p

    for ln in lanes:
        p = process(ln)
        leaf = TreeNode(f"l{ln.index}", "lane", ln.category, (_cat_key(ln.category), ln.name), lane=ln.index,
                        category=ln.category)
        if ln.kind == "thread" and ln.tid is not None:
            t = threads.get((ln.pid, ln.tid))
            if t is None:
                t = threads[(ln.pid, ln.tid)] = TreeNode(
                    f"t{ln.pid}.{ln.tid}", "thread", ln.thread_label or f"tid {ln.tid}", (0, ln.tid))
                p.children.append(t)
            t.children.append(leaf)
        elif ln.kind in ("stream", "device"):
            dkey = ln.device if ln.device is not None else ""
            d = devices.get((ln.pid, dkey))
            if d is None:
                label = f"device {dkey}" if dkey else "device"
                d = devices[(ln.pid, dkey)] = TreeNode(f"d{ln.pid}.{dkey}", "device", label, (1, dkey))
                p.children.append(d)
            if ln.kind == "stream":
                skey = (ln.pid, dkey, ln.ident)
                s = streams.get(skey)
                if s is None:
                    try:
                        num = int(ln.ident)
                    except ValueError:
                        num = 1 << 30
                    s = streams[skey] = TreeNode(f"s{ln.pid}.{dkey}.{ln.ident}", "stream", f"stream {ln.ident}",
                                                 (num, ln.ident))
                    d.children.append(s)
                s.children.append(leaf)
            else:
                leaf.label = f"{ln.category} (no stream)"
                d.children.append(leaf)
        else:
            leaf.sort = (2,) + leaf.sort
            p.children.append(leaf)

    def finish(n: TreeNode) -> None:
        for c in n.children:
            finish(c)
        n.children.sort(key=lambda c: c.sort)
    finish(root)
    return root


def flatten(root: TreeNode, collapsed: dict[str, bool], *, lane_row, show_root: bool = True
            ) -> list[dict[str, Any]]:
    """Visible rows, depth-first. `lane_row(lane_index)` returns the lane's
    row dict or None (filtered out / hidden); headers whose subtree has no
    visible lane are dropped. A thread or stream with exactly one visible
    lane is shown as that lane's row, labelled with both."""
    rows: list[dict[str, Any]] = []

    def visit(n: TreeNode, depth: int, prefix: str) -> None:
        if n.lane >= 0:
            r = lane_row(n.lane)
            if r is None:
                return
            r = dict(r)
            r["level"] = depth
            r["hierLabel"] = f"{prefix} · {n.label}" if prefix else n.label
            rows.append(r)
            return
        visible = [c for c in n.children if any(lane_row(i) is not None for i in c.lanes())]
        if not visible:
            return
        if n.kind in ("thread", "stream") and len(visible) == 1 and visible[0].lane >= 0:
            visit(visible[0], depth, n.label)
            return
        is_root = n.kind == "system"
        if is_root and not show_root:
            for c in visible:
                visit(c, depth, "")
            return
        lanes = [i for c in visible for i in c.lanes() if lane_row(i) is not None]
        lane_rows = [lane_row(i) for i in lanes]
        is_collapsed = collapsed.get(n.id, False)
        rows.append({
            "kind": "group", "groupId": n.id, "nodeKind": n.kind, "label": n.label, "level": depth,
            "laneIndexes": lanes, "laneCount": len(lanes),
            "count": sum(r["count"] for r in lane_rows),
            "filteredCount": sum(r["filteredCount"] for r in lane_rows),
            "collapsed": is_collapsed,
        })
        if not is_collapsed:
            for c in visible:
                visit(c, depth + 1, "")

    visit(root, 0, "")
    return rows


def aggregate_levels(root: TreeNode) -> dict[str, dict]:
    """How each header aggregates its members (see TimelineModel.groupRender):
    a thread or stream combines its lanes by MAX (lanes of one thread nest:
    a barrier wait inside a parallel region), a process its threads by MEAN
    (host utilization), a device its streams by MEAN, a node or the system
    their processes by MEAN."""
    out: dict[str, dict] = {}

    def visit(n: TreeNode) -> None:
        if n.lane >= 0:
            return
        out[n.id] = {"kind": n.kind, "children": [(c.id if c.lane < 0 else None, c.lane) for c in n.children]}
        for c in n.children:
            visit(c)
    visit(root)
    return out
