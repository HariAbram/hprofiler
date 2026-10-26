"""
The Timeline's exact-value side: one event's details (timing, place,
source / stack / kernel / disassembly, critical-path membership, its
dependencies and their evidence, its comparison counterpart), one
dependency's evidence, "jump to", the coordinated summary views (every
value carries the highlight / focus that leads back to the events it was
computed from), the heatmap, and figure export.

Values are tagged like the Inspector's (src/gui/inspector.py): "measured"
(read off the trace), "derived" (computed from measured values or the
dependency graph), "estimated" or "unavailable" (with the reason).
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np

from ..analysis import cp_model as cm
from ..analysis import dashboard as dash
from ..core import gpu_activity as ga
from . import timeline_lod as lod

fmt = dash.fmt_ns


def _f(label: str, value: Any, kind: str = "measured", reason: str = "", action: dict | None = None
       ) -> dict[str, Any]:
    out = {"label": label, "value": "" if value is None else str(value), "kind": kind, "reason": reason}
    if action:
        out["action"] = action
    return out


def _unavail(label: str, reason: str) -> dict[str, Any]:
    return _f(label, "", "unavailable", reason)


# ── one event ─────────────────────────────────────────────────────────────

def event_details(m, lane: int, eid: int) -> dict[str, Any]:
    if lane < 0 or lane >= len(m._lane_names) or eid < 0:
        return {}
    s = m._store.event_by_id(eid)
    if s is None or not hasattr(s, "duration_ns"):
        return {}
    t0 = float(m._view_start)
    tags = s.tags
    ex = m._lane_extra[lane]
    act = lod.activity_of(s)
    sections: list[dict[str, Any]] = []

    ident = [_f("Name", s.name), _f("Category", s.category.value),
             _f("Runtime", tags.get("rt") or s.category.value),
             _f("Activity", lod.ACTIVITY_LABELS[act], "derived",
                "from the span's type tag (analysis/activity_buckets.py)")]
    if tags.get("type"):
        ident.append(_f("Type", tags["type"]))
    sections.append({"title": "Event", "fields": ident})

    timing = [_f("Start", "+" + fmt(s.start_ns - t0)), _f("End", "+" + fmt(s.end_ns - t0)),
              _f("Duration", fmt(s.duration_ns))]
    src = ga.timing_source(s)
    timing.append(_f("Timing source", src, "measured" if src in (ga.TIMING_HOST, ga.TIMING_DEVICE) else "estimated",
                     "" if src in (ga.TIMING_HOST, ga.TIMING_DEVICE)
                     else "host-side proxy: start is the submission time, queueing invisible"))
    if tags.get("queue_ns") is not None:
        try:
            timing.append(_f("Queued before execution", fmt(int(tags["queue_ns"]))))
        except (TypeError, ValueError):
            pass
    sections.append({"title": "Time", "fields": timing})

    place = []
    if ex.get("node") is not None:
        place.append(_f("Node", ex["node"]))
    place.append(_f("Process", f"pid {s.pid}"))
    rank = tags.get("rank", ex.get("rank"))
    place.append(_f("MPI rank", rank) if rank is not None else _unavail("MPI rank", "no rank= tag on this process"))
    if ex["kind"] == "thread":
        place.append(_f("Thread", ex["threadLabel"]))
    elif ex["kind"] in ("stream", "device"):
        place.append(_f("Device", ex["device"]) if ex.get("device") is not None
                     else _unavail("Device", "no hdev=/dev= tag"))
        place.append(_f("Stream / queue", ex["ident"]) if ex["kind"] == "stream"
                     else _unavail("Stream / queue", "device work without a stream tag"))
        place.append(_f("Launched by thread", f"tid {s.tid}"))
    place.append(_f("Lane", m._lane_meta[lane]["label"]))
    sections.append({"title": "Where", "fields": place})

    attr = []
    if tags.get("file"):
        loc = f"{tags['file']}:{tags.get('line', '?')}"
        attr.append(_f("Source", loc, action={"kind": "source", "category": s.category.value, "name": s.name}))
    elif tags.get("sym"):
        attr.append(_f("Symbol", tags["sym"] + (f" ({tags['lib']})" if tags.get("lib") else ""),
                       action={"kind": "source", "category": s.category.value, "name": s.name}))
    else:
        attr.append(_unavail("Source", "no file=/sym= tag on this event (call-site capture off or not supported)"))
    if s.stack_frames:
        frames = s.stack_frames[:8]
        attr.append(_f("Stack", " ← ".join(frames) + (" ← …" if len(s.stack_frames) > 8 else "")))
    else:
        attr.append(_unavail("Stack", "no call stack recorded (HPROFILER_CALLSTACK / --call-tree)"))
    if ga.is_device_kernel(s):
        dims = ", ".join(f"{k}={tags[k]}" for k in ("grid", "block", "regs", "smem") if k in tags)
        if dims:
            attr.append(_f("Kernel", dims))
        try:
            kd, how = m._trace.disasm_for(s.name, str(tags.get("mod", "")))
        except Exception:
            kd, how = None, ""
        attr.append(_f("Disassembly", f"available ({how})" if kd is not None else "",
                       "measured" if kd is not None else "unavailable",
                       "" if kd is not None else "not collected for this kernel (--disasm)",
                       action={"kind": "source", "category": s.category.value, "name": s.name}
                       if kd is not None else None))
    sections.append({"title": "Source, stack, kernel", "fields": attr})

    an = m._analysis
    cpf: list[dict[str, Any]] = []
    deps: list[dict[str, Any]] = []
    if an is None or not an.ok:
        reason = ("computing…" if m._analysis_state == "running"
                  else "the dependency graph is not computed yet: enable Critical path or Dependencies")
        cpf.append(_unavail("Critical path", reason))
    else:
        on = eid in an.path_eids
        cpf.append(_f("On the observed path", "yes" if on else "no", "derived",
                      "the chain of last-releasing dependencies (each instant counted once)"))
        if eid in an.credit_of_eid:
            cpf.append(_f("Its time on the path", fmt(an.credit_of_eid[eid]), "derived"))
        if eid in an.on_all:
            longest = "on every longest path"
        elif eid in an.on_some:
            longest = "on some longest paths (equal-length alternatives exist)"
        else:
            longest = "not on a longest path"
        cpf.append(_f("Modeled longest path", longest, "derived",
                      "required work only (waiting and dependency latency count zero); a model"))
        if eid in an.slack_of_eid:
            cpf.append(_f("Slack", fmt(an.slack_of_eid[eid]), "derived",
                          "how much one of its busy segments could grow before the modeled path does"))
        for r in an.sensitivity:
            if r.get("eid") == eid:
                cpf.append(_f("Counterfactual", f"busy time → 0 shortens the modeled path by "
                              f"{fmt(r['modeled_reduction_ns'])}", "estimated",
                              "a graph counterfactual, not a measured speedup"))
        for w in an.waits:
            if w.eid == eid:
                cause = an.edges[w.edge] if 0 <= w.edge < len(an.edges) else None
                txt = f"waited {fmt(w.t1 - w.t0)} before its last dependency"
                if cause is not None:
                    txt += f" ({cause.kind} from {m._lane_display(cause.src_lane)}, {cause.tier})"
                cpf.append(_f("Inferred waiting", txt, "derived",
                              "start of the call to its binding dependency's source event"))
                break
        for k in an.edges_of_eid.get(eid, []):
            e = an.edges[k]
            incoming = e.dst_eid == eid
            other_eid, other_lane = (e.src_eid, e.src_lane) if incoming else (e.dst_eid, e.dst_lane)
            other = m._store.event_by_id(other_eid) if other_eid >= 0 else None
            deps.append({
                "edge": k, "direction": "in" if incoming else "out", "kind": e.kind, "family": e.family,
                "runtime": e.runtime, "tier": e.tier, "provenance": e.provenance, "class": e.cls,
                "reason": cm.edge_reason(e.kind, e.provenance, e.tier), "onPath": e.on_path,
                "partner": other.name if other is not None else "?", "partnerLane": other_lane,
                "partnerSpanIdx": float(other_eid), "partnerPlace": m._lane_display(other_lane),
                "latencyNs": float(abs(e.t_dst - e.t_src)),
                "partnerStartNs": float(other.start_ns) if other is not None else 0.0,
            })
    sections.append({"title": "Critical path", "fields": cpf})

    if s.category.value == "mpi":
        mpi = [_f(k, tags[k]) for k in ("peer", "wpeer", "tag", "commid", "bytes", "rpeer", "rtag") if k in tags]
        partners = [d for d in deps if d["kind"] in ("p2p", "arrival")]
        for d in partners[:4]:
            mpi.append(_f("Matched " + ("sender" if d["direction"] == "in" else "receiver")
                          if d["kind"] == "p2p" else "Rendezvous (last arriver)",
                          f"{d['partner']} @ {d['partnerPlace']}", "derived" if d["class"] == "direct" else "estimated",
                          f"{d['provenance']} / {d['tier']}"))
        if mpi:
            sections.append({"title": "MPI", "fields": mpi})

    if m._comparison_hook is not None:
        try:
            cmpf = m._comparison_hook(lane, eid)
        except Exception as exc:
            cmpf = [_unavail("Comparison", f"{type(exc).__name__}: {exc}")]
        if cmpf:
            sections.append({"title": "Comparison", "fields": cmpf})

    other_tags = [_f(k, v) for k, v in sorted(tags.items())
                  if k not in ("rank", "peer", "wpeer", "tag", "commid", "bytes", "file", "line", "sym", "lib")]
    if other_tags:
        sections.append({"title": "Tags", "fields": other_tags[:40]})
    return {"title": s.name, "laneIndex": lane, "spanIdx": float(eid), "startNs": float(s.start_ns),
            "durNs": float(s.duration_ns), "category": s.category.value, "pid": s.pid, "tid": s.tid,
            "sections": sections, "dependencies": deps}


def edge_details(m, edge_id: int) -> dict[str, Any]:
    an = m._analysis
    if an is None or not an.ok or edge_id < 0 or edge_id >= len(an.edges):
        return {}
    e = an.edges[edge_id]
    src = m._store.event_by_id(e.src_eid) if e.src_eid >= 0 else None
    dst = m._store.event_by_id(e.dst_eid) if e.dst_eid >= 0 else None
    t0 = float(m._view_start)
    from .timeline_analysis import FAMILY_LABELS
    return {
        "id": edge_id, "kind": e.kind, "family": e.family, "familyLabel": FAMILY_LABELS.get(e.family, e.family),
        "runtime": e.runtime, "tier": e.tier, "provenance": e.provenance, "class": e.cls,
        "provenanceText": cm.PROVENANCE[e.provenance][1], "reason": cm.edge_reason(e.kind, e.provenance, e.tier),
        "onPath": e.on_path, "latencyNs": float(e.t_dst - e.t_src),
        "source": {"name": src.name if src else "?", "laneIndex": e.src_lane, "spanIdx": float(e.src_eid),
                   "place": m._lane_display(e.src_lane), "event": e.src_end, "atNs": float(e.t_src),
                   "at": "+" + fmt(e.t_src - t0)},
        "target": {"name": dst.name if dst else "?", "laneIndex": e.dst_lane, "spanIdx": float(e.dst_eid),
                   "place": m._lane_display(e.dst_lane), "event": e.dst_end, "atNs": float(e.t_dst),
                   "at": "+" + fmt(e.t_dst - t0)},
        "note": "evidence classes are not probabilities: nothing calibrates the tiers",
    }


# ── jump to ───────────────────────────────────────────────────────────────

_UNIT = {"ns": 1.0, "us": 1e3, "µs": 1e3, "ms": 1e6, "s": 1e9}
_TIME = r"([0-9]+(?:\.[0-9]*)?)\s*(ns|us|µs|ms|s)?"


def resolve_jump(m, text: str) -> dict[str, Any]:
    """"12.5ms" / "@3s" (offset from the trace start), "1ms-2ms" (a range),
    "rank 3", "tid 1234" / "T3", "stream 7", or an event name (its first
    occurrence; "next" cycles through search matches instead)."""
    t = (text or "").strip()
    if not t:
        return {"kind": "none", "message": "type a time, rank, thread, stream or event name"}
    t0 = float(m._view_start)
    mt = re.fullmatch(r"@?\s*" + _TIME + r"\s*(?:-|\.\.|to)\s*" + _TIME, t)
    if mt:
        a = float(mt.group(1)) * _UNIT[mt.group(2) or "ns"]
        b = float(mt.group(3)) * _UNIT[mt.group(4) or mt.group(2) or "ns"]
        if b > a:
            return {"kind": "range", "startNs": t0 + a, "endNs": t0 + b}
    mt = re.fullmatch(r"@?\s*" + _TIME, t)
    if mt:
        return {"kind": "time", "ns": t0 + float(mt.group(1)) * _UNIT[mt.group(2) or "ns"]}
    mt = re.fullmatch(r"(?:rank|r)\s*(-?\d+)", t, re.IGNORECASE)
    if mt:
        lanes = [i for i, ex in enumerate(m._lane_extra) if str(ex.get("rank")) == mt.group(1)]
        return _lane_jump(m, lanes, f"rank {mt.group(1)}")
    mt = re.fullmatch(r"(?:tid|thread)\s*(\d+)", t, re.IGNORECASE)
    if mt:
        lanes = [i for i, ex in enumerate(m._lane_extra) if ex.get("tid") == int(mt.group(1))]
        return _lane_jump(m, lanes, f"tid {mt.group(1)}")
    mt = re.fullmatch(r"T(\d+)", t)
    if mt:
        lanes = [i for i, ex in enumerate(m._lane_extra) if ex.get("threadLabel", "").startswith(f"T{mt.group(1)} ")]
        return _lane_jump(m, lanes, t)
    mt = re.fullmatch(r"(?:stream|s|queue)\s*(\S+)", t, re.IGNORECASE)
    if mt:
        lanes = [i for i, ex in enumerate(m._lane_extra) if ex.get("kind") == "stream" and ex.get("ident") == mt.group(1)]
        if lanes:
            return _lane_jump(m, lanes, f"stream {mt.group(1)}")
    names = sorted({r["name"] for r in m._store.aggregate_stats()})
    exact = [n for n in names if n == t]
    cands = exact or [n for n in names if t.lower() in n.lower()]
    if cands:
        found = m._occurrences(frozenset(cands[:20]), None, 1)
        if found:
            li, eid, start = found[0]
            s = m._store.event_by_id(eid)
            return {"kind": "event", "laneIndex": li, "spanIdx": float(eid), "startNs": float(start),
                    "durNs": float(s.duration_ns) if s is not None else 0.0, "name": s.name if s else cands[0],
                    "candidates": len(cands)}
    return {"kind": "none", "message": f"nothing matches \"{t}\""}


def _lane_jump(m, lanes: list[int], label: str) -> dict[str, Any]:
    if not lanes:
        return {"kind": "none", "message": f"no lane for {label}"}
    for li in lanes:
        # unfold collapsed headers above it so the lane is a row
        for node_id in _ancestors(m, li):
            if m._group_collapsed.get(node_id):
                m._group_collapsed[node_id] = False
    m._rebuild_rows()
    pos = min((m._row_pos[li] for li in lanes if li in m._row_pos), default=-1)
    return {"kind": "lane", "laneIndexes": lanes, "rowPos": pos, "label": label}


def _ancestors(m, lane: int) -> list[str]:
    out: list[str] = []

    def visit(n, path) -> bool:
        if n.lane == lane:
            out.extend(path)
            return True
        return any(visit(c, path + [n.id]) for c in n.children)
    visit(m._tree, [])
    return out


# ── coordinated summary views ────────────────────────────────────────────

_ACT_OF_BUCKET = {"Computation": "work", "Communication": "communication", "Memory transfer": "memory",
                  "Runtime overhead": "overhead", "Synchronization": "wait", "Idle": "idle",
                  "Annotation": "annotation"}


def summaries(m) -> dict[str, Any]:
    agg = m._store.exclusive_aggregate()
    host: dict[str, int] = {}
    dev: dict[str, int] = {}
    wait_by_cat: dict[str, int] = {}
    by_pid: dict[int, dict[str, int]] = {}
    for (pid, _tid, cat, _name, bucket, device), ns in agg.rows.items():
        act = "jit" if cat == "jit" else _ACT_OF_BUCKET.get(bucket, "other")
        if act == "annotation":
            continue
        (dev if device else host)[act] = (dev if device else host).get(act, 0) + ns
        if not device:
            d = by_pid.setdefault(pid, {})
            d[act] = d.get(act, 0) + ns
            if act == "wait":
                wait_by_cat[cat] = wait_by_cat.get(cat, 0) + ns

    def rows(d: dict[str, int], scope: str) -> list[dict[str, Any]]:
        tot = sum(d.values()) or 1
        out = []
        for act in lod.ACTIVITY:
            if act in d:
                label = lod.ACTIVITY_LABELS[lod.ACTIVITY.index(act)]
                out.append({"key": act, "label": label, "ns": float(d[act]), "share": d[act] / tot,
                            "colorKind": "activity", "hatched": act in ("wait", "idle"),
                            "highlight": {"activities": [act], "label": f"{label} ({scope})"}})
        return out

    procs = []
    for pid, d in sorted(by_pid.items(), key=lambda kv: -sum(kv[1].values()))[:32]:
        tot = sum(d.values()) or 1
        lanes = [i for i, ex in enumerate(m._lane_extra) if ex["pid"] == pid]
        rank = next((ex["rank"] for ex in m._lane_extra if ex["pid"] == pid and ex["rank"] is not None), None)
        procs.append({"label": f"rank {rank}" if rank is not None else f"pid {pid}", "pid": pid,
                      "parts": [{"key": a, "share": d[a] / tot, "hatched": a in ("wait", "idle")}
                                for a in lod.ACTIVITY if a in d],
                      "ns": float(sum(d.values())),
                      "highlight": {"lanes": lanes, "label": f"rank {rank}" if rank is not None else f"pid {pid}"}})
    wtot = sum(wait_by_cat.values()) or 1
    waits = [{"key": cat, "label": f"{cat} waiting", "ns": float(ns), "colorKind": "category", "share": ns / wtot,
              "highlight": {"activities": ["wait"], "categories": [cat], "label": f"{cat} waiting"}}
             for cat, ns in sorted(wait_by_cat.items(), key=lambda kv: -kv[1])]

    # union busy time per stream / thread lane, from the activity index
    lanes_busy = []
    for i, ln in enumerate(m._lane_names):
        if m._lane_extra[i]["category"] in m._NOT_WORK_CATEGORIES:
            continue
        lv = m._store.activity(ln)
        if not lv:
            continue
        busy = float(np.asarray(lv[0]["busy"], dtype=np.float64).sum())
        ex = m._lane_extra[i]
        lanes_busy.append({"laneIndex": i, "label": m._lane_display(i), "kind": ex["kind"], "ns": busy,
                           "category": ex["category"],
                           "highlight": {"lanes": [i], "label": m._lane_display(i)}})
    lanes_busy.sort(key=lambda r: -r["ns"])
    out: dict[str, Any] = {
        "host": rows(host, "host threads, exclusive time"), "device": rows(dev, "device work"),
        "processes": procs, "waits": waits,
        "streams": [r for r in lanes_busy if r["kind"] in ("stream", "device")][:16],
        "threads": [r for r in lanes_busy if r["kind"] == "thread"][:16],
        "analysis": m._analysis_state,
    }
    an = m._analysis
    if an is not None and an.ok:
        sm = an.summary
        tot = max(1, sm["observedNs"])
        out["tiers"] = [{"key": t, "ns": float(ns), "share": ns / tot} for t, ns in sm["tiers"].items() if ns]
        out["classes"] = [{"key": c, "ns": float(ns), "share": ns / tot} for c, ns in sm["classes"].items() if ns]
        out["graphTiers"] = [{"key": t, "count": int(n)} for t, n in sm["graph"]["byTier"].items()]
        out["graphClasses"] = [{"key": c, "count": int(n)} for c, n in sm["graph"]["byClass"].items()]
        out["coverage"] = _coverage_rows(sm)
        top = sorted(an.waits, key=lambda w: -(w.t1 - w.t0))[:12]
        tw = []
        for w in top:
            e = an.edges[w.edge] if 0 <= w.edge < len(an.edges) else None
            src = m._store.event_by_id(e.src_eid) if e is not None and e.src_eid >= 0 else None
            tw.append({"name": w.name, "place": m._lane_display(w.lane), "laneIndex": w.lane, "spanIdx": float(w.eid),
                       "waitNs": float(w.t1 - w.t0), "startNs": float(w.t0), "endNs": float(w.call_end),
                       "cause": (f"{e.kind} from {src.name if src else '?'} @ {m._lane_display(e.src_lane)}"
                                 if e is not None else "?"),
                       "causeLaneIndex": e.src_lane if e is not None else -1,
                       "causeSpanIdx": float(e.src_eid) if e is not None else -1.0,
                       "tier": e.tier if e is not None else "", "provenance": e.provenance if e is not None else "",
                       "edge": w.edge})
        out["topWaits"] = tw
        tr: dict[str, int] = {}
        for t in an.transitions:
            k = f"{m._lane_extra[t.from_lane]['category']} → {m._lane_extra[t.to_lane]['category']}" \
                if t.from_lane >= 0 and t.to_lane >= 0 else "?"
            tr[k] = tr.get(k, 0) + 1
        out["transitions"] = [{"key": k, "count": n} for k, n in sorted(tr.items(), key=lambda kv: -kv[1])]
    return out


def _coverage_rows(sm: dict) -> list[dict[str, Any]]:
    d = sm.get("diagnostics") or {}
    out = []
    if d:
        rec, mat = d.get("receives", 0), d.get("matched", 0)
        out.append({"label": "MPI receive completions with a sender", "value": f"{mat} / {rec}",
                    "share": (mat / rec) if rec else 1.0, "kind": "derived"})
        for k, v in sorted((d.get("unmatched_by_comm") or {}).items()):
            out.append({"label": f"no sender found (communicator {k})", "value": str(v), "share": None,
                        "kind": "derived"})
        for key, label in (("test_only_completions", "Test*-only completions (no edge by design)"),
                           ("orphan_completions", "waits on requests never seen posted"),
                           ("matched_after_loss", "paired after lost events ('low')")):
            if d.get(key):
                out.append({"label": label, "value": str(d[key]), "share": None, "kind": "derived"})
    g = sm.get("graph") or {}
    dropped = sum((g.get("dropped") or {}).values())
    if dropped:
        out.append({"label": "dependency edges dropped (contradict timestamps)", "value": str(dropped),
                    "share": None, "kind": "derived"})
    if g.get("rooted"):
        out.append({"label": "rooted-collective edges modeled all-to-all", "value": str(g["rooted"]),
                    "share": None, "kind": "derived"})
    return out


def heatmap(m, n_bins: int) -> dict[str, Any]:
    """Busy share per time bin for each process (its host threads' mean) and
    each device (its streams' mean): the run's shape at a glance; a cell
    leads to that time range and those lanes."""
    n = max(4, min(int(n_bins), 400))
    t0, t1 = float(m._view_start), float(m._view_end)
    rows = []

    def walk(node):
        if node.kind in ("process", "device"):
            lanes = [i for i in node.lanes() if i in m._visible_lanes or not m._visible_lanes]
            if lanes:
                busy, wait, _c, _cp = m.render_group(node.id, lanes, t0, t1, n)
                label = node.label if node.kind == "process" else f"  {node.label}"
                rows.append({"label": label, "groupId": node.id, "laneIndexes": lanes,
                             "busy": [round(float(v), 3) for v in busy], "wait": [round(float(v), 3) for v in wait]})
        for c in node.children:
            if c.lane < 0:
                walk(c)
    walk(m._tree)
    return {"rows": rows[:64], "t0": t0, "t1": t1, "bins": n, "truncated": len(rows) > 64}


# ── figure export ────────────────────────────────────────────────────────

def export_figure(m, path: str, a: float, b: float, width: int) -> None:
    from PySide6.QtCore import QPointF, QRectF, QSize, Qt
    from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
    from . import timeline_paint as tp
    style = m._style()
    label_w, row_h, top_h, pad = 190, 20, 56, 12
    rows = m._rows
    lane_w = max(200, int(width) - label_w - 2 * pad)
    families = sorted({an_e for an_e in (m._overlay.get("families") or [])})
    legend = tp.legend_entries(style.dark, color_mode=m._color_mode, has_density=True, has_unobserved=True,
                               families=families, critical=bool(m._overlay.get("criticalPath")))
    legend_h = 18 * len(legend) + 24
    H = top_h + row_h * len(rows) + legend_h + pad
    W = label_w + lane_w + 2 * pad
    svg = path.lower().endswith(".svg")
    if svg:
        from PySide6.QtSvg import QSvgGenerator
        dev = QSvgGenerator()
        dev.setFileName(path)
        dev.setSize(QSize(W, H))
        dev.setViewBox(QRectF(0, 0, W, H))
        dev.setTitle("hprofiler timeline")
    else:
        dev = QImage(W * 2, H * 2, QImage.Format.Format_ARGB32)
        dev.setDevicePixelRatio(2.0)
        dev.fill(style.background)
    p = QPainter(dev)
    try:
        if svg:
            p.fillRect(QRectF(0, 0, W, H), style.background)
        font = QFont()
        font.setPixelSize(11)
        bold = QFont(font)
        bold.setBold(True)
        p.setFont(bold)
        p.setPen(style.text)
        cmd = m._trace.metadata.command or "trace"
        t0 = float(m._view_start)
        p.drawText(QPointF(pad, 18), f"hprofiler timeline — {cmd}   [{fmt(a - t0)} … {fmt(b - t0)}]")
        ticks = m.timeTicks(a, b, 8)
        tp.paint_ruler(p, ticks, a, b, pad + label_w, lane_w, top_h - 4, style, font)
        p.setFont(font)
        for i, r in enumerate(rows):
            y = top_h + i * row_h
            indent = 10 * int(r.get("level", 0))
            p.setPen(style.text if r["kind"] == "lane" else style.muted)
            label = r.get("hierLabel") or r["label"]
            p.drawText(QRectF(pad + indent, y, label_w - indent - 4, row_h),
                       int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft), label)
            p.save()
            p.translate(pad + label_w, y)
            if r["kind"] == "lane":
                m.paint_lane_item(p, r["laneIndex"], a, b, lane_w, row_h)
            else:
                m.paint_group_item(p, r["groupId"], r["laneIndexes"], a, b, lane_w, row_h)
            p.restore()
            p.setPen(QPen(style.border, 0.5))
            p.drawLine(QPointF(pad + label_w, y + row_h), QPointF(pad + label_w + lane_w, y + row_h))
        p.save()
        p.translate(pad + label_w, top_h)
        p.setClipRect(QRectF(0, 0, lane_w, row_h * len(rows)))
        m.paint_overlay_item(p, lane_w, row_h * len(rows), a, b, 0.0, row_h, -1, -1.0)
        p.restore()
        ly = top_h + row_h * len(rows) + 16
        p.setPen(style.text)
        p.setFont(bold)
        p.drawText(QPointF(pad, ly), "Legend")
        p.setFont(font)
        for j, e in enumerate(legend):
            y = ly + 6 + 18 * j
            sw = QRectF(pad, y, 22, 12)
            c = QColor(e["color"])
            if e["kind"] == "fill":
                p.fillRect(sw, c)
            elif e["kind"] in ("wait", "unobserved"):
                pat = Qt.BrushStyle.BDiagPattern if e["kind"] == "wait" else Qt.BrushStyle.DiagCrossPattern
                p.fillRect(sw, tp._with_alpha(style.idle, 0.3))
                p.fillRect(sw, tp._hatch_brush(c, pat))
            elif e["kind"] == "density":
                p.fillRect(QRectF(pad, y + 8, 22, 4), c)
                p.fillRect(QRectF(pad + 10, y, 1, 3), style.text)
            elif e["kind"] == "critical":
                p.fillRect(QRectF(pad, y, 22, 3), c)
            else:
                p.setPen(tp.tier_pen(c, "certain", "direct"))
                p.drawLine(QPointF(pad, y + 6), QPointF(pad + 22, y + 6))
            p.setPen(style.text)
            p.drawText(QPointF(pad + 30, y + 10), e["label"])
    finally:
        p.end()
    if not svg:
        if not dev.save(path):
            raise OSError(f"could not write {path}")
