"""
Matches and diffs two traces' aggregated stats for the GUI's Comparison
mode (Phase C) -- pure Python, no Qt, so the matching/classification logic
is independently unit-testable without PySide6, same discipline as every
other module in this package. Reused by src/gui/comparison.py's
ComparisonBridge, not duplicated there.

Matching key: (category, name), the SAME stable identifier Round 16
already established for cross-tab navigation (span_id is real but never
serialized to JSON -- see src/gui/nav.py's own docstring). Fallback tier
reuses the EXISTING dash.fmt_kernel_name() normalizer (already used to
shorten JIT hash-named kernels for display) applied to both sides before a
second matching pass -- not a new fuzzy-matching heuristic invented for
this feature.

No fabricated statistics: hprofiler captures single-run traces only, with
no repeated-trial/variance data anywhere, so classify() below uses a
fixed, DISCLOSED noise-floor threshold (both a minimum percentage AND a
minimum absolute time must be cleared before a delta counts as a real
improvement/regression) -- an explicit heuristic guard against reading
run-to-run noise as a meaningful change, not a statistical significance
test. This mirrors the existing measured/derived/estimated/unavailable
honesty contract (Round 16): a delta suppressed by the noise floor is
`kind: "derived"` with a `reason` explaining why, not silently hidden.
"""
from __future__ import annotations

from typing import Any

from . import activity_buckets
from . import dashboard as dash

# Both must be cleared for a delta to count as a real change -- see the
# module docstring for why neither threshold alone is sufficient.
NOISE_FLOOR_PCT = 5.0
NOISE_FLOOR_NS = 1_000_000.0  # 1ms

STATUS_IMPROVED = "improved"
STATUS_REGRESSED = "regressed"
STATUS_UNCHANGED = "unchanged"
STATUS_NEW = "new"
STATUS_REMOVED = "removed"
STATUS_UNAVAILABLE = "unavailable"
STATUS_ZERO = "zero"


def match_rows(rows_a: list[dict], rows_b: list[dict]) -> list[dict[str, Any]]:
    """Matches baseline (`rows_a`) and comparison (`rows_b`) rows --
    either shape works as long as each dict has "category"/"name" keys,
    which both Trace.aggregated_stats() rows and call-tree-path dicts do
    -- by (category, name). Returns one entry per row on EITHER side:
    {"category", "name", "base", "comp", "matchKind"}, matchKind one of
    "exact"/"normalized"/"baseline_only"/"comparison_only"; `base`/`comp`
    are the original row dicts, None on whichever side didn't match."""
    index_a = {(r["category"], r["name"]): r for r in rows_a}
    index_b = {(r["category"], r["name"]): r for r in rows_b}

    out: list[dict[str, Any]] = []
    used_b: set[tuple[str, str]] = set()

    for key, ra in index_a.items():
        if key in index_b:
            out.append({"category": key[0], "name": key[1], "base": ra, "comp": index_b[key], "matchKind": "exact"})
            used_b.add(key)
            continue
        norm_name = dash.fmt_kernel_name(key[1])
        match = None
        for kb, rb in index_b.items():
            if kb in used_b or kb[0] != key[0]:
                continue
            if dash.fmt_kernel_name(kb[1]) == norm_name:
                match = (kb, rb)
                break
        if match is not None:
            kb, rb = match
            out.append({"category": key[0], "name": key[1], "base": ra, "comp": rb, "matchKind": "normalized"})
            used_b.add(kb)
        else:
            out.append({"category": key[0], "name": key[1], "base": ra, "comp": None, "matchKind": "baseline_only"})

    for key, rb in index_b.items():
        if key in used_b:
            continue
        out.append({"category": key[0], "name": key[1], "base": None, "comp": rb, "matchKind": "comparison_only"})

    return out


def classify(base_ns: float | None, comp_ns: float | None, *,
             noise_pct: float = NOISE_FLOOR_PCT, noise_ns: float = NOISE_FLOOR_NS) -> tuple[str, dict[str, Any]]:
    """Returns (status, delta) where delta is a Field-shaped dict
    ({"value", "kind", "reason"}, this codebase's existing measured/
    derived/estimated/unavailable vocabulary) for delta_ns = comp - base.
    status is one of improved/regressed/unchanged/new/removed/
    unavailable/zero -- "zero" (both sides genuinely 0) is kept distinct
    from "unavailable" (no data on one or both sides), since they mean
    very different things to a user scanning a comparison table."""
    if base_ns is None and comp_ns is None:
        return STATUS_UNAVAILABLE, {"value": None, "kind": "unavailable", "reason": "no data on either side"}
    if base_ns is None:
        return STATUS_NEW, {"value": comp_ns, "kind": "measured", "reason": ""}
    if comp_ns is None:
        return STATUS_REMOVED, {"value": -base_ns, "kind": "measured", "reason": ""}

    delta_ns = comp_ns - base_ns
    if base_ns == 0 and comp_ns == 0:
        return STATUS_ZERO, {"value": 0.0, "kind": "measured", "reason": ""}
    if base_ns == 0:
        # No percentage is computable from a zero baseline -- any nonzero
        # comparison value is still a real, just unquantifiable-as-%, change.
        status = STATUS_REGRESSED if delta_ns > 0 else STATUS_IMPROVED
        return status, {"value": delta_ns, "kind": "measured", "reason": ""}

    pct = 100.0 * delta_ns / base_ns
    if abs(pct) < noise_pct or abs(delta_ns) < noise_ns:
        return STATUS_UNCHANGED, {
            "value": delta_ns, "kind": "derived",
            "reason": f"below the {noise_pct:.0f}%/{dash.fmt_ns(noise_ns)} noise-floor guard "
                      "-- a disclosed heuristic threshold, not a statistical significance test "
                      "(hprofiler has no repeated-trial variance data to test against)",
        }
    status = STATUS_REGRESSED if delta_ns > 0 else STATUS_IMPROVED
    return status, {"value": delta_ns, "kind": "measured", "reason": ""}


def compare_aggregates(trace_a: Any, trace_b: Any, *,
                        noise_pct: float = NOISE_FLOOR_PCT, noise_ns: float = NOISE_FLOOR_NS) -> list[dict[str, Any]]:
    """One row per (category,name) seen in EITHER trace, ranked by
    absolute impact (|deltaNs|) descending -- the Overview "largest
    improvements/regressions" ranking and the Compare tab's main table
    both consume this directly."""
    matched = match_rows(trace_a.aggregated_stats(), trace_b.aggregated_stats())
    out = []
    for m in matched:
        base, comp = m["base"], m["comp"]
        base_ns = base["total_ns"] if base else None
        comp_ns = comp["total_ns"] if comp else None
        status, delta = classify(base_ns, comp_ns, noise_pct=noise_pct, noise_ns=noise_ns)
        pct = None
        if base_ns and comp_ns is not None:
            pct = 100.0 * (comp_ns - base_ns) / base_ns
        out.append({
            "category": m["category"], "name": m["name"], "matchKind": m["matchKind"], "status": status,
            "baseNs": base_ns, "compNs": comp_ns,
            "deltaNs": delta["value"], "deltaKind": delta["kind"], "deltaReason": delta["reason"], "deltaPct": pct,
            "baseCount": base["count"] if base else None, "compCount": comp["count"] if comp else None,
        })
    out.sort(key=lambda r: abs(r["deltaNs"] or 0.0), reverse=True)
    return out


def compare_buckets(trace_a: Any, trace_b: Any, *,
                     noise_pct: float = NOISE_FLOOR_PCT, noise_ns: float = NOISE_FLOOR_NS) -> list[dict[str, Any]]:
    """One row per activity bucket (see activity_buckets.py) present in
    EITHER trace -- Compare tab's bucket-delta panel, the aggregate-level
    analog of compare_aggregates()'s per-function rows. Annotation is
    excluded, same reasoning as bucket_totals() itself (overlaps real
    work by design, would double-count)."""
    totals_a = activity_buckets.bucket_totals(trace_a.spans)
    totals_b = activity_buckets.bucket_totals(trace_b.spans)
    out = []
    for bucket in activity_buckets.BUCKETS:
        if bucket == "Annotation":
            continue
        a = totals_a.get(bucket, 0)
        b = totals_b.get(bucket, 0)
        if a == 0 and b == 0:
            continue
        status, delta = classify(float(a), float(b), noise_pct=noise_pct, noise_ns=noise_ns)
        out.append({"bucket": bucket, "baseNs": float(a), "compNs": float(b),
                     "deltaNs": delta["value"], "status": status})
    return out


def top_changes(compared_rows: list[dict], *, status: str, limit: int = 10) -> list[dict[str, Any]]:
    """The top `limit` rows from compare_aggregates()'s output with the
    given `status` ("improved" or "regressed", typically), ranked by
    absolute impact -- Overview's "largest improvements/regressions"."""
    matching = [r for r in compared_rows if r["status"] == status]
    return sorted(matching, key=lambda r: abs(r["deltaNs"] or 0.0), reverse=True)[:limit]


def normalized_coverage(trace: Any, n_buckets: int = 60) -> list[float]:
    """Activity coverage across `trace`'s OWN wall-clock duration,
    normalized to [0,1] of ITS OWN span -- not an absolute-time overlay.
    Two independently-captured runs have unrelated monotonic clock
    origins and generally different total durations, so lining them up
    on a shared absolute time axis would be actively misleading, not just
    inconvenient; each side's strip is meant to be read as "this run's
    own shape," compared side by side, with that normalization stated in
    the UI (see ComparisonBridge)."""
    spans = [s for s in trace.spans if s.duration_ns > 0]
    if not spans:
        return [0.0] * n_buckets
    start = min(s.start_ns for s in spans)
    end = max(s.start_ns + s.duration_ns for s in spans)
    dur = max(end - start, 1)
    return dash.bucket_coverage(spans, n_buckets, start, float(dur))


def report_dict(trace_a: Any, trace_b: Any, *,
                 noise_pct: float = NOISE_FLOOR_PCT, noise_ns: float = NOISE_FLOOR_NS,
                 top_n: int = 10) -> dict[str, Any]:
    """The full comparison as one JSON-serializable dict -- backs
    ComparisonBridge.exportReport()."""
    aggregates = compare_aggregates(trace_a, trace_b, noise_pct=noise_pct, noise_ns=noise_ns)
    buckets = compare_buckets(trace_a, trace_b, noise_pct=noise_pct, noise_ns=noise_ns)
    return {
        "noiseFloor": {"pct": noise_pct, "ns": noise_ns,
                       "note": "fixed disclosed heuristic threshold, not a statistical significance test"},
        "aggregates": aggregates,
        "buckets": buckets,
        "topImprovements": top_changes(aggregates, status=STATUS_IMPROVED, limit=top_n),
        "topRegressions": top_changes(aggregates, status=STATUS_REGRESSED, limit=top_n),
        "newCount": sum(1 for r in aggregates if r["status"] == STATUS_NEW),
        "removedCount": sum(1 for r in aggregates if r["status"] == STATUS_REMOVED),
    }
