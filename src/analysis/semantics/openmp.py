"""
OpenMP task and lock semantics (the OMPT tool's task ids, ompt_tool.c; the
GNU libgomp hook's lock= tags, gomp_hook.c):

  * taskwait: returns once every child task its task created before it
    has completed -- children: omp_task_create spans with parent= that
    task's id, created before the taskwait started;
  * taskgroup: returns once every task created inside it by its task, and
    every descendant of those, has completed;
  * task dependences (omp_task_dependence instants src= -> sink=): the sink
    starts after the source completed;
  * locks and critical sections: a contended acquisition (an omp_*_wait
    span) ends when the previous holder of that lock (wait_id= / lock=;
    GNU libgomp's unnamed critical sections share one lock) released it.

A task's completion is the end of its last execution span (omp_task with
id=; a task that was suspended has fragment=1 spans before its final one).
Edges: task completion -> join "task_join" (end to end), dependence source
-> sink "task_dep" (end to start), release -> acquisition "lock_order" (end
to end). The GNU hook records no task ids on taskwait: none there.
"""
from __future__ import annotations

from collections import defaultdict

# lock acquisition ends within this of the previous holder's release
# (one clock; only the hook's own call overhead lies between)
HANDOFF_TOLERANCE_NS = 50_000


def _end(s) -> int:
    return s.start_ns + max(0, s.duration_ns)


def add_task_edges(preds, spans, items, instants=()) -> None:
    """task_join / task_dep edges into `preds`; `instants`: the trace's
    instant events (omp_task_dependence)."""
    by_pid: dict[int, dict] = defaultdict(lambda: {"exec": defaultdict(list), "created": {}, "joins": []})
    for i, s in items(spans):
        cat = s.category.value
        t = s.tags
        if cat == "openmp" and s.name == "omp_task" and t.get("id"):
            by_pid[s.pid]["exec"][t["id"]].append(i)
        elif cat == "openmp" and t.get("type") == "task_create" and t.get("id") and "parent" in t:
            by_pid[s.pid]["created"][t["id"]] = (t["parent"], s.start_ns)
        elif cat == "sync" and s.name in ("omp_taskwait", "omp_taskgroup") and t.get("task"):
            by_pid[s.pid]["joins"].append(i)
    deps: dict[int, list[tuple[str, str]]] = defaultdict(list)
    for ins in instants:
        if ins.name == "omp_task_dependence" and ins.tags.get("src") and ins.tags.get("sink"):
            deps[ins.pid].append((ins.tags["src"], ins.tags["sink"]))
    for pid in sorted(set(by_pid) | set(deps)):
        p = by_pid[pid]
        completion = {tid: max(lst, key=lambda j: _end(spans[j])) for tid, lst in p["exec"].items()}
        first = {tid: min(lst, key=lambda j: spans[j].start_ns) for tid, lst in p["exec"].items()}
        children: dict[str, list[tuple[int, str]]] = defaultdict(list)
        for tid, (parent, created) in p["created"].items():
            children[parent].append((created, tid))
        for lst in children.values():
            lst.sort()
        for j in p["joins"]:
            s = spans[j]
            mine = children.get(s.tags["task"], [])
            if s.name == "omp_taskwait":
                waited = [tid for created, tid in mine if created < s.start_ns]
            else:                                  # taskgroup: created inside it, and their descendants
                waited = [tid for created, tid in mine if s.start_ns <= created <= _end(s)]
                todo, seen = list(waited), set(waited)
                while todo:
                    for _c, tid in children.get(todo.pop(), ()):
                        if tid not in seen:
                            seen.add(tid)
                            waited.append(tid)
                            todo.append(tid)
            for tid in waited:
                k = completion.get(tid)
                if k is not None and k != j:
                    preds[j].append((k, "task_join", "certain"))
        for src, sink in deps.get(pid, ()):
            a, b = completion.get(src), first.get(sink)
            if a is not None and b is not None and a != b:
                preds[b].append((a, "task_dep", "certain"))


def add_lock_edges(preds, spans, items) -> None:
    """lock_order edges: each contended acquisition from the release that
    let it through."""
    import bisect
    holds: dict[tuple, list[int]] = defaultdict(list)
    waits: dict[tuple, list[int]] = defaultdict(list)
    for i, s in items(spans):
        t = s.tags
        lock = t.get("wait_id") or t.get("lock")
        if lock and "mutex" in t:
            key = (s.pid, t["mutex"], lock)
        elif s.name in ("omp_critical_wait", "omp_critical_hold") and "named" not in t and "mutex" not in t:
            key = (s.pid, "critical", "unnamed")        # GNU libgomp: one lock for every unnamed critical
        else:
            continue
        if s.name.endswith("_hold") or t.get("type") == "hold":
            holds[key].append(i)
        elif s.name.endswith("_wait"):
            waits[key].append(i)
    for key in sorted(waits, key=str):
        hl = sorted(holds.get(key, ()), key=lambda j: _end(spans[j]))
        ends = [_end(spans[j]) for j in hl]
        for w in waits[key]:
            sw = spans[w]
            acquired = _end(sw)
            # the latest release no later than the acquisition, by another thread, during the wait
            pos = bisect.bisect_right(ends, acquired + HANDOFF_TOLERANCE_NS) - 1
            while pos >= 0 and spans[hl[pos]].tid == sw.tid:
                pos -= 1
            if pos >= 0 and ends[pos] >= sw.start_ns and acquired - ends[pos] <= HANDOFF_TOLERANCE_NS:
                preds[w].append((hl[pos], "lock_order", "high"))
