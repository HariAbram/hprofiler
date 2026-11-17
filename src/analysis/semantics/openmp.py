"""
OpenMP task and lock semantics (the OMPT tool's task ids, ompt_tool.c; the
GNU libgomp hook's lock= tags, gomp_hook.c):

  * taskwait: returns once every child task its task created before it
    has completed -- children: omp_task_create spans with parent= that
    task's id, created before the taskwait started;
  * taskgroup: returns once every task created inside it by its task, and
    every descendant of those, has completed;
  * task dependences (omp_task_dependence instants src= -> sink=): the sink
    starts after the source completed. The GNU libgomp hook records each
    task's depend clauses instead (deps= on omp_task_create, tokens for the
    storage locations, see gomp_hook.c): dependence_pairs() applies
    OpenMP's rules to the siblings in creation order -- reconstructed from
    the clauses, not reported by the runtime ("high", not "certain");
  * taskwait with depend clauses (GNU: deps= on omp_taskwait): waits for
    the earlier siblings its clauses conflict with;
  * locks and critical sections: a contended acquisition (an omp_*_wait
    span) ends when the previous holder of that lock (wait_id= / lock=;
    GNU libgomp's unnamed critical sections share one lock) released it.

A task's completion is the end of its last execution span (omp_task with
id=; a task that was suspended has fragment=1 spans before its final one).
Edges: task completion -> join "task_join" (end to end), dependence source
-> sink "task_dep" (end to start), release -> acquisition "lock_order" (end
to end). Traces of GNU hooks older than parent= / task= have no task joins.
"""
from __future__ import annotations

from collections import defaultdict

# lock acquisition ends within this of the previous holder's release
# (one clock; only the hook's own call overhead lies between)
HANDOFF_TOLERANCE_NS = 50_000


def _end(s) -> int:
    return s.start_ns + max(0, s.duration_ns)


def parse_deps(value: str) -> list[tuple[str, str]]:
    """deps=<kind><token>;... -> [(kind, token)]: kind i in, o out/inout, m
    mutexinoutset, s inoutset; token "*" is omp_all_memory."""
    out = []
    for e in str(value).split(";"):
        if len(e) >= 2 and e[0] in "ioms":
            out.append((e[0], e[1:]))
    return out


def _new_state(after: list) -> dict:
    # group: the latest writer (or set of inoutset tasks) of a location;
    # readers: the in-tasks since; set_preds: what that inoutset set waited for
    return {"group": list(after), "kind": "o", "readers": [], "set_preds": list(after)}


def _preds_for(state: dict, all_group: list, deps: list[tuple[str, str]]) -> list:
    """What a task (or a taskwait) with `deps` waits for, by OpenMP's rules,
    against the siblings generated before it."""
    if any(tok == "*" for _k, tok in deps):      # omp_all_memory: every earlier dependence
        out = list(all_group)
        for st in state.values():
            out.extend(st["readers"] or st["group"])
        return out
    out = []
    for k, tok in deps:
        st = state.get(tok) or _new_state(all_group)
        if k == "i":
            out.extend(st["group"])
        elif k == "s" and st["kind"] == "s" and not st["readers"] and st["group"]:
            out.extend(st["set_preds"])          # joins the inoutset set: unordered within it
        else:                                    # out / inout / mutexinoutset / a new inoutset set
            out.extend(st["readers"] or st["group"])
    return out


def dependence_pairs(tasks: list[tuple[str, list[tuple[str, str]]]]) -> tuple[list[tuple[str, str]], dict]:
    """[(source id, sink id)] among one parent's children `tasks` -- (task
    id, deps), in creation order -- by OpenMP's dependence rules: in waits
    for the latest out / inout / mutexinoutset (or inoutset set) on the
    location; out / inout / mutexinoutset for the in-tasks since, else that
    writer; inoutset tasks next to each other form one set, unordered among
    themselves; omp_all_memory for every earlier dependence, and every later
    one for it. mutexinoutset is ordered like out: GNU libgomp implements it
    so (mutual exclusion through order). Returns the pairs and the location
    state after the last task (for taskwaits, see taskwait_preds)."""
    state: dict[str, dict] = {}
    all_group: list = []
    pairs: list[tuple[str, str]] = []
    for tid, deps in tasks:
        preds = _preds_for(state, all_group, deps)
        for p in dict.fromkeys(preds):
            if p != tid:
                pairs.append((p, tid))
        if any(tok == "*" for _k, tok in deps):
            state.clear()
            all_group = [tid]
            continue
        for k, tok in deps:
            st = state.get(tok)
            if st is None:
                st = state[tok] = _new_state(all_group)
            if k == "i":
                st["readers"].append(tid)
            elif k == "s" and st["kind"] == "s" and not st["readers"] and st["group"]:
                st["group"].append(tid)
            elif k == "s":
                st.update(set_preds=list(st["readers"] or st["group"]), group=[tid], kind="s", readers=[])
            else:
                st.update(group=[tid], kind="o", readers=[], set_preds=[])
    return pairs, {"state": state, "all": all_group}


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
            by_pid[s.pid]["created"][t["id"]] = (t["parent"], s.start_ns, i,
                                                 parse_deps(t["deps"]) if t.get("deps") else None)
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
        ordered: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
        for tid, (parent, created, ordinal, _deps) in p["created"].items():
            children[parent].append((created, tid))
            ordered[parent].append((created, ordinal, tid))
        for lst in children.values():
            lst.sort()
        for lst in ordered.values():
            lst.sort()
        gnu: list[tuple[str, str]] = []           # (source, sink) from the GNU hook's deps=
        for parent, lst in ordered.items():
            if any(p["created"][tid][3] for _c, _o, tid in lst):
                pairs, _st = dependence_pairs([(tid, p["created"][tid][3] or []) for _c, _o, tid in lst])
                gnu.extend(pairs)
        for j in p["joins"]:
            s = spans[j]
            mine = children.get(s.tags["task"], [])
            if s.name == "omp_taskwait" and s.tags.get("depend") == "1" and not s.tags.get("deps"):
                continue                          # its clauses were not decoded: what it waited for is unknown
            if s.name == "omp_taskwait" and s.tags.get("deps"):
                # taskwait depend(...): the earlier siblings its clauses conflict with
                earlier = [(tid, p["created"][tid][3] or []) for c, _o, tid in ordered.get(s.tags["task"], [])
                           if c < s.start_ns]
                _pairs, st = dependence_pairs(earlier)
                for tid in dict.fromkeys(_preds_for(st["state"], st["all"], parse_deps(s.tags["deps"]))):
                    k = completion.get(tid)
                    if k is not None and k != j:
                        preds[j].append((k, "task_join", "high"))
                continue
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
        for src, sink in gnu:
            a, b = completion.get(src), first.get(sink)
            if a is not None and b is not None and a != b:
                preds[b].append((a, "task_dep", "high"))


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


def dependency_diagnostics(trace) -> dict:
    """Holes in the OpenMP dependence capture, counted (flat keys, omp_*):

      omp_gnu_dep_tasks          GNU tasks whose depend clauses were decoded
      omp_deps_unsupported       {reason: tasks / taskwaits} whose clauses
                                 were not decoded (abi: a libgomp ABI the hook
                                 does not know; counts / kind: an array it
                                 could not read safely) -- their dependences
                                 are unknown
      omp_deps_partial           tasks with entries left out (deps_omitted=,
                                 over the per-record limit, or dep_unknown=,
                                 the token table full): some dependences
                                 missing
      omp_gnu_legacy_tasks       GNU tasks without parent= (an older hook): no
                                 taskwait / taskgroup / dependence edges
      omp_lock_unknown           lock records without a token (lock=?)
      omp_gomp_abi_level         libgomp's ABI (x*10000+y*100+z of GOMP_x.y.z)
    Empty when there is nothing to report."""
    out: dict = {}
    unsupported: dict[str, int] = {}
    for s in trace.iter_spans(categories=("openmp", "sync")):
        t = s.tags
        if s.name == "omp_task_create" and t.get("type") == "task_create":
            if "team" in t and "parent" not in t:
                out["omp_gnu_legacy_tasks"] = out.get("omp_gnu_legacy_tasks", 0) + 1
            if t.get("deps") is not None and "ndeps" in t:
                out["omp_gnu_dep_tasks"] = out.get("omp_gnu_dep_tasks", 0) + 1
        if t.get("deps_unsupported"):
            unsupported[t["deps_unsupported"]] = unsupported.get(t["deps_unsupported"], 0) + 1
        if t.get("deps_omitted") or t.get("dep_unknown"):
            out["omp_deps_partial"] = out.get("omp_deps_partial", 0) + 1
        if t.get("lock") == "?":
            out["omp_lock_unknown"] = out.get("omp_lock_unknown", 0) + 1
    if unsupported:
        out["omp_deps_unsupported"] = dict(sorted(unsupported.items()))
    for c in trace.iter_counters():
        if c.name == "gomp_abi_level":
            out["omp_gomp_abi_level"] = int(c.value)
    if set(out) <= {"omp_gomp_abi_level", "omp_gnu_dep_tasks"}:
        return {}
    return out


def dependency_warnings(diag: dict) -> list[str]:
    out = []
    un = diag.get("omp_deps_unsupported") or {}
    if un:
        per = ", ".join(f"{k}: {v}" for k, v in un.items())
        out.append(f"{sum(un.values())} OpenMP task / taskwait depend clause list(s) were not decoded ({per}"
                   + (f"; libgomp ABI {diag['omp_gomp_abi_level']}" if diag.get("omp_gomp_abi_level") else "")
                   + "): their task dependences are unknown, not inferred")
    if diag.get("omp_deps_partial"):
        out.append(f"{diag['omp_deps_partial']} OpenMP task(s) recorded only part of their depend clauses "
                   "(deps_omitted= / dep_unknown=): some dependences between them are missing")
    if diag.get("omp_gnu_legacy_tasks"):
        out.append(f"{diag['omp_gnu_legacy_tasks']} GNU OpenMP task(s) were recorded without parent= (an older "
                   "hook): no taskwait, taskgroup or dependence edges for them")
    if diag.get("omp_lock_unknown"):
        out.append(f"{diag['omp_lock_unknown']} OpenMP lock record(s) have no lock token (lock=?, table full): "
                   "their hand-offs are not linked")
    return out
