"""
Side-by-side timelines of two runs (the Compare tab) -- Qt-free.

Events are matched with hprofiler's own comparison (analysis/projection.py
+ causal_compare.py), never by timestamps: an event belongs to a projected
node; the node to a comparison identity (same call path, runtime and
rank / thread / stream roles in aligned phases); the counterpart is the
node of that identity in the SAME aligned phase pair of the other run,
and within it the event with the same occurrence index (k-th by start
time). Every match states that basis and its confidence (the identity's
match score times the phase alignment's confidence -- a heuristic).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..analysis import dashboard as dash

fmt = dash.fmt_ns


class EventMatcher:
    def __init__(self, detail: dict, trace_a, trace_b) -> None:
        """detail: what causal_compare.compare_projections(detail=...)
        filled in; trace_a = baseline, trace_b = candidate."""
        self.ok = bool(detail) and detail.get("comp") is not None
        self.detail = detail
        self.traces = {"baseline": trace_a, "candidate": trace_b}
        self.proj = {"baseline": detail.get("pa"), "candidate": detail.get("pb")}
        self.comp = detail.get("comp")
        self.rows = detail.get("rows") or {}
        self.method = detail.get("method", "")
        self.confidence = float(detail.get("confidence") or 0.0)
        self._starts: dict[str, np.ndarray] = {}
        self._durs: dict[str, np.ndarray] = {}

    # per side: start / duration of every span by ordinal (one pass, lazily)
    def _arrays(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        if side not in self._starts:
            p = self.proj[side]
            n = len(p.span_seqs)
            st = np.zeros(n, dtype=np.int64)
            du = np.zeros(n, dtype=np.int64)
            for i, s in enumerate(self.traces[side].iter_spans()):
                if i >= n:
                    break
                st[i], du[i] = s.start_ns, s.duration_ns
            self._starts[side], self._durs[side] = st, du
        return self._starts[side], self._durs[side]

    def _ordinal(self, side: str, span) -> int:
        p = self.proj[side]
        i = int(np.searchsorted(p.span_seqs, span.seq))
        return i if i < len(p.span_seqs) and p.span_seqs[i] == span.seq else -1

    def identity_of(self, side: str, span) -> Any:
        if not self.ok:
            return None
        o = self._ordinal(side, span)
        if o < 0:
            return None
        node = int(self.proj[side].span_node[o])
        if node < 0:
            return None
        return (self.comp.ident_b if side == "candidate" else self.comp.ident_a).get(node)

    def status_of(self, side: str, span) -> str:
        ident = self.identity_of(side, span)
        if ident is None:
            return "unmatched"
        row = self.rows.get(ident.id)
        if row is None:
            return "unmatched"
        st = row["status"]
        if st in ("unchanged", "zero", "unavailable") and row.get("impactStatus") in ("regressed", "improved"):
            return row["impactStatus"]
        return st

    def match(self, side: str, eid: int) -> dict[str, Any]:
        """The counterpart of one event in the other run, with the basis."""
        other_side = "baseline" if side == "candidate" else "candidate"
        if not self.ok:
            return {"status": "unavailable",
                    "basis": "no node-level comparison (alignment confidence too low: aggregate comparison only)"}
        tr = self.traces[side]
        span = tr.store.event_by_id(int(eid))
        if span is None or not hasattr(span, "duration_ns"):
            return {"status": "unavailable", "basis": "no such event"}
        o = self._ordinal(side, span)
        node = int(self.proj[side].span_node[o]) if o >= 0 else -1
        if node < 0:
            return {"status": "unmatched", "basis": "not a projected node (annotation, perf sample, or zero-length)"}
        ident = (self.comp.ident_b if side == "candidate" else self.comp.ident_a).get(node)
        if ident is None:
            return {"status": "unmatched", "basis": "its node has no comparison identity"}
        row = self.rows.get(ident.id, {})
        pair_idx, other_node = None, None
        for pair, (na, nb) in sorted(ident.pairs.items()):
            mine = nb if side == "candidate" else na
            if mine is not None and mine.id == node:
                pair_idx, other_node = pair, (na if side == "candidate" else nb)
                break
        starts, durs = self._arrays(side)
        p = self.proj[side]
        mine_ords = np.nonzero(np.asarray(p.span_node) == node)[0]
        order = mine_ords[np.argsort(starts[mine_ords], kind="stable")]
        hits = np.nonzero(order == o)[0]
        k = int(hits[0]) if len(hits) else -1
        out: dict[str, Any] = {
            "identity": ident.id, "label": row.get("label", ""), "roles": row.get("roles", ""),
            "status": self.status_of(side, span), "cause": row.get("cause", ""),
            "causeLabel": row.get("causeLabel", ""), "explanation": row.get("explanation", ""),
            "matchKind": (row.get("match") or {}).get("kind", ""), "matchScore": (row.get("match") or {}).get("score"),
            "confidence": row.get("confidence"), "pair": pair_idx, "method": self.method,
            "occurrence": k, "occurrences": [int(len(order)), 0],
            "durationNs": [None, None], "evidence": row.get("evidence", []),
            "chain": row.get("chain", []), "measured": row.get("measured", {}), "derived": row.get("derived", {}),
        }
        mine_dur = int(span.duration_ns)
        if other_node is None:
            out["counterpart"] = None
            out["basis"] = (f"identity #{ident.id} ({row.get('label', '')}, {row.get('roles', '')}) has no node in "
                            f"the {other_side}'s aligned phase: {'new' if side == 'candidate' else 'removed'} work")
            out["durationNs"] = [None, float(mine_dur)] if side == "candidate" else [float(mine_dur), None]
            return out
        ostarts, odurs = self._arrays(other_side)
        op = self.proj[other_side]
        their = np.nonzero(np.asarray(op.span_node) == other_node.id)[0]
        their = their[np.argsort(ostarts[their], kind="stable")]
        out["occurrences"][1] = int(len(their))
        basis = (f"identity #{ident.id}: {out['matchKind']} match (score {out['matchScore']}) of "
                 f"{row.get('label', '')} [{row.get('roles', '')}] in aligned phase pair {pair_idx} "
                 f"({self.method}, alignment confidence {self.confidence:.2f}); occurrence {k + 1} of "
                 f"{len(order)} here, {len(their)} there")
        if k < 0 or k >= len(their):
            out["counterpart"] = None
            out["basis"] = basis + f" -- the {other_side} has no occurrence {k + 1}"
            return out
        ord_other = int(their[k])
        other_seq = int(op.span_seqs[ord_other])
        cspan = self.traces[other_side].store.span_by_seq(other_seq)
        out["counterpart"] = {"eid": float(cspan.eid), "startNs": float(cspan.start_ns),
                              "durNs": float(cspan.duration_ns), "name": cspan.name, "seq": other_seq}
        other_dur = int(odurs[ord_other])
        a_dur, b_dur = (other_dur, mine_dur) if side == "candidate" else (mine_dur, other_dur)
        out["durationNs"] = [float(a_dur), float(b_dur)]
        out["deltaNs"] = float(b_dur - a_dur)
        out["relDelta"] = (b_dur - a_dur) / a_dur if a_dur > 0 else None
        out["basis"] = basis
        return out


def alignment_offset(mode: str, report: dict, *, phase: int = -1, identity_row: dict | None = None,
                     event_pair: tuple[float, float] | None = None, starts: tuple[float, float] = (0.0, 0.0)
                     ) -> dict[str, Any]:
    """The time shift that puts the baseline under the candidate:
    candidate_ns = baseline_ns + offset. Modes: start (both runs' first
    events), event (a matched event pair), phase (an aligned phase pair's
    starts), region (a contributor's aligned region). Never a fit of
    timestamps: each offset comes from one stated correspondence."""
    base_t0, cand_t0 = starts
    if mode == "event" and event_pair is not None:
        b, c = event_pair
        return {"offsetNs": c - b, "basis": "the selected event and its matched counterpart start together"}
    if mode == "phase" and phase >= 0:
        rows = report.get("phases") or []
        if 0 <= phase < len(rows) and rows[phase].get("baseline") and rows[phase].get("candidate"):
            b = rows[phase]["baseline"]["startNs"]
            c = rows[phase]["candidate"]["startNs"]
            return {"offsetNs": c - b, "basis": f"aligned phase pair {phase} ({rows[phase]['label']}) starts together"}
        return {"offsetNs": cand_t0 - base_t0, "basis": "that phase is not in both runs: aligned by start"}
    if mode == "region" and identity_row is not None:
        tl = identity_row.get("timeline") or {}
        if tl.get("baseline") and tl.get("candidate"):
            return {"offsetNs": tl["candidate"]["startNs"] - tl["baseline"]["startNs"],
                    "basis": f"the matched region of {identity_row.get('label', '')} starts together"}
    return {"offsetNs": cand_t0 - base_t0, "basis": "application start (each run's first event)"}


def delta_strip(report: dict) -> list[dict[str, Any]]:
    """Where the runs diverge: one segment per aligned phase pair, on the
    candidate's clock (removed phases on the baseline's), with its status."""
    out = []
    for p in report.get("phases") or []:
        c, b = p.get("candidate"), p.get("baseline")
        span = c or b
        if span is None:
            continue
        out.append({"index": p["index"], "label": p["label"], "pairStatus": p["status"],
                    "deltaStatus": p.get("deltaStatus", ""), "deltaNs": float(p.get("deltaNs", 0)),
                    "criticalDeltaNs": float(p.get("criticalDeltaNs", 0)),
                    "startNs": float(span["startNs"]), "endNs": float(span["endNs"]),
                    "side": "candidate" if c else "baseline",
                    "baselineStartNs": float(b["startNs"]) if b else None,
                    "baselineEndNs": float(b["endNs"]) if b else None})
    return out


def structure_diff(lane_extra_a: list[dict], lane_extra_b: list[dict]) -> list[dict[str, Any]]:
    """Ranks, threads, devices and streams present in one run only (by
    rank and per-rank counts -- pids and tids differ between runs)."""
    def summary(extra):
        ranks: dict[str, dict[str, set]] = {}
        for ex in extra:
            r = str(ex["rank"]) if ex["rank"] is not None else f"pid#{sorted({e['pid'] for e in extra}).index(ex['pid'])}"
            d = ranks.setdefault(r, {"threads": set(), "streams": set(), "devices": set(), "runtimes": set()})
            if ex["kind"] == "thread" and ex["tid"] is not None:
                d["threads"].add(ex["tid"])
            if ex["kind"] == "stream":
                d["streams"].add((ex["device"], ex["ident"]))
            if ex["device"] is not None:
                d["devices"].add(ex["device"])
            d["runtimes"].add(ex["category"])
        return ranks
    A, B = summary(lane_extra_a), summary(lane_extra_b)
    out = []
    for r in sorted(set(A) | set(B), key=lambda x: (len(x), x)):
        a, b = A.get(r), B.get(r)
        if a is None or b is None:
            out.append({"rank": r, "change": "only in the " + ("candidate" if a is None else "baseline"), "kind": "measured"})
            continue
        for key, label in (("threads", "threads"), ("streams", "streams / queues"), ("devices", "devices")):
            if len(a[key]) != len(b[key]):
                out.append({"rank": r, "change": f"{label}: {len(a[key])} → {len(b[key])}", "kind": "measured"})
        added, removed = b["runtimes"] - a["runtimes"], a["runtimes"] - b["runtimes"]
        if added or removed:
            out.append({"rank": r, "change": "runtimes " + ", ".join(
                [f"+{x}" for x in sorted(added)] + [f"-{x}" for x in sorted(removed)]), "kind": "measured"})
    return out


def regression_chain(row: dict) -> list[dict[str, Any]]:
    """The upstream chain of a changed wait as steps, the observed change
    first and every inferred link labelled as such:
        longer wait (measured) -> what it waited for (graph-derived,
        heuristic phase offsets) -> ... -> the origin (its own measured
        change)."""
    steps = [{"label": row.get("label", ""), "roles": row.get("roles", ""), "identity": row.get("id"),
              "text": f"{row.get('causeLabel') or row.get('cause') or 'changed'}: own time "
                      f"{dash.fmt_signed_ns(row.get('ownDeltaNs', 0))}", "kind": "measured"}]
    for c in row.get("chain") or []:
        own = c.get("ownDeltaNs")
        shift = c.get("endShiftNs")
        steps.append({"label": c["label"], "roles": c["roles"], "identity": c["identity"],
                      "text": ("ends " + dash.fmt_signed_ns(shift) + " later relative to its phase"
                               if shift is not None else "upstream dependency")
                              + (f"; own time {dash.fmt_signed_ns(own)}" if own is not None else ""),
                      "kind": "derived (dependency graph, heuristic phase offsets)"})
    if len(steps) > 1:
        steps[-1]["text"] += "  ← origin" if (row.get("chain") or [{}])[-1].get("status") in ("regressed", "improved") \
            else ""
    return steps
