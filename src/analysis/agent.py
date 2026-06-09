"""Agentic analysis loop for hprofiler traces."""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from ..core.trace import Trace
    from .llm.base import LLMProvider


_SYSTEM_PROMPT = """\
Always respond in English regardless of the language of the data or prior messages.

You are an expert HPC performance engineer specialising in heterogeneous computing \
(CUDA, OpenMP, MPI, ROCm, OpenCL, NCCL). You are analysing real profiling data \
captured by hprofiler.

Analysis methodology:
1. Start with time_breakdown_by_category to identify the dominant cost area.
2. Drill into top_hotspots and use tools to get deeper data on suspicious entries.
3. Classify each bottleneck: memory-bound, compute-bound, sync-stalled, \
   CPU-limited, launch-overhead, or communication-bound.
4. For GPU kernels: consider grid/block dimensions for occupancy hints, \
   and whether bandwidth is near the hardware peak from the roofline data.
5. For synchronisation: identify what is being waited for and whether async \
   overlap is possible.
6. For memory transfers: check direction (H2D/D2H/D2D), bandwidth vs peak, \
   and whether pinned memory or async copies are used.
7. For MPI: look for load imbalance (uneven wait times), small-message overhead, \
   and unnecessary synchronisation.
8. Formulate specific, actionable recommendations with realistic impact estimates.

Output format (use Markdown):

## Executive Summary
2–3 sentences: biggest bottleneck and its root cause.

## Top Bottlenecks (ranked by impact)
### 1. [Name] — [Root Cause Category]
- **Evidence:** specific timing and percentage
- **Root cause:** WHY this is slow
- **Fix:** specific code change or configuration
- **Estimated impact:** rough speedup or time saved

## Secondary Observations
(brief bullets)

## Optimization Roadmap
Priority-ordered list with [HIGH/MED/LOW] tags.

Rules:
- Reference actual function/kernel names, timings, and percentages.
- Avoid generic advice; every recommendation must be grounded in the data.
- If uncertain, say so and describe what additional profiling would confirm it.
"""


@dataclass
class AnalysisConfig:
    max_turns: int = 8
    max_tokens: int = 4096
    temperature: float = 0.2
    verbose: bool = False


@dataclass
class AnalysisReport:
    content: str
    turns_used: int
    model: str
    trace_summary: str = ""


ProgressFn = Callable[[str], None]


def analyze(
    trace: "Trace",
    provider: "LLMProvider",
    config: AnalysisConfig | None = None,
    on_progress: ProgressFn | None = None,
) -> AnalysisReport:
    """Run multi-turn agentic analysis on a Trace."""
    from .context import build_profile_context, context_to_str
    from .agent_tools import TOOL_DEFINITIONS, execute_tool

    cfg = config or AnalysisConfig()
    ctx = build_profile_context(trace)
    ctx_str = context_to_str(ctx)

    run_info = ctx.get("run", {})
    trace_summary = (
        f"{run_info.get('command', '?')} — "
        f"{run_info.get('wall_time_ms', 0)} ms wall time, "
        f"{run_info.get('total_spans', 0)} spans, "
        f"backends: {', '.join(run_info.get('backends', []))}"
    )

    initial = (
        "Please analyse the following profiling data and identify performance "
        "bottlenecks and optimisation opportunities. Use the available tools to "
        "drill down into areas that need deeper investigation.\n\n"
        f"## Profile Data\n```json\n{ctx_str}\n```"
    )

    messages: list[dict] = [{"role": "user", "content": initial}]
    last_content = ""
    turns = 0
    use_tools = True

    for turn in range(cfg.max_turns):
        turns = turn + 1
        if on_progress:
            on_progress(f"[turn {turns}] querying {provider.display_name} …")

        try:
            response = provider.chat(
                messages=messages,
                system=_SYSTEM_PROMPT,
                tools=TOOL_DEFINITIONS if use_tools else None,
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
            )
        except RuntimeError as exc:
            err = str(exc)
            # Some models/endpoints reject tool-use — degrade gracefully
            if use_tools and ("tool" in err.lower() or "function" in err.lower()):
                if on_progress:
                    on_progress("Model does not support tool use — falling back to single-shot")
                use_tools = False
                response = provider.chat(
                    messages=messages,
                    system=_SYSTEM_PROMPT,
                    tools=None,
                    max_tokens=cfg.max_tokens,
                    temperature=cfg.temperature,
                )
            else:
                raise

        if response.content:
            last_content = response.content

        if not response.tool_calls:
            break

        # Log tool calls
        if on_progress:
            for tc in response.tool_calls:
                args_preview = json.dumps(tc.arguments)[:80]
                on_progress(f"  ⚙  {tc.name}({args_preview})")
        elif cfg.verbose:
            for tc in response.tool_calls:
                print(f"  ⚙  {tc.name}({json.dumps(tc.arguments)[:80]})", flush=True)

        messages.append({
            "role": "assistant",
            "content": response.content or None,
            "tool_calls": [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                for tc in response.tool_calls
            ],
        })

        for tc in response.tool_calls:
            result = execute_tool(tc.name, tc.arguments, trace)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    return AnalysisReport(
        content=last_content,
        turns_used=turns,
        model=provider.display_name,
        trace_summary=trace_summary,
    )


def compare_traces(
    trace_a: "Trace",
    trace_b: "Trace",
    label_a: str = "before",
    label_b: str = "after",
    provider: "LLMProvider" | None = None,
    config: AnalysisConfig | None = None,
    on_progress: ProgressFn | None = None,
) -> AnalysisReport:
    """Compare two traces; report improvements, regressions, and probable causes."""
    from .context import build_profile_context, context_to_str

    if provider is None:
        raise ValueError("provider is required")

    cfg = config or AnalysisConfig()
    ctx_a = build_profile_context(trace_a)
    ctx_b = build_profile_context(trace_b)
    diff = _compute_diff(ctx_a, ctx_b)

    message = (
        f"Compare these two profiling runs ({label_a} vs {label_b}). "
        "Identify what changed, whether the changes are improvements or regressions, "
        "and what likely caused them.\n\n"
        f"## {label_a.capitalize()} (baseline)\n```json\n{context_to_str(ctx_a)}\n```\n\n"
        f"## {label_b.capitalize()} (modified)\n```json\n{context_to_str(ctx_b)}\n```\n\n"
        f"## Key Differences\n```json\n{json.dumps(diff, indent=2)}\n```"
    )

    compare_system = (
        _SYSTEM_PROMPT
        + "\n\nStructure your response as:\n"
        "## Summary of Changes\n"
        "## Improvements\n"
        "## Regressions\n"
        "## Unchanged / Inconclusive\n"
        "## Conclusion"
    )

    if on_progress:
        on_progress(f"Comparing traces with {provider.display_name} …")

    response = provider.chat(
        messages=[{"role": "user", "content": message}],
        system=compare_system,
        tools=None,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
    )

    ra = ctx_a.get("run", {})
    rb = ctx_b.get("run", {})
    summary = (
        f"{ra.get('command', label_a)} ({ra.get('wall_time_ms', 0)} ms) vs "
        f"{rb.get('command', label_b)} ({rb.get('wall_time_ms', 0)} ms)"
    )

    return AnalysisReport(
        content=response.content,
        turns_used=1,
        model=provider.display_name,
        trace_summary=summary,
    )


def _compute_diff(ctx_a: dict, ctx_b: dict) -> dict:
    diff: dict = {}

    wt_a = ctx_a.get("run", {}).get("wall_time_ms", 0)
    wt_b = ctx_b.get("run", {}).get("wall_time_ms", 0)
    if wt_a and wt_b:
        diff["wall_time_change_ms"] = round(wt_b - wt_a, 3)
        diff["wall_time_change_pct"] = round(100.0 * (wt_b - wt_a) / wt_a, 1)

    bd_a = ctx_a.get("time_breakdown_by_category", {})
    bd_b = ctx_b.get("time_breakdown_by_category", {})
    cat_changes: dict = {}
    for cat in set(bd_a) | set(bd_b):
        ms_a = bd_a.get(cat, {}).get("total_ms", 0)
        ms_b = bd_b.get(cat, {}).get("total_ms", 0)
        if ms_a or ms_b:
            cat_changes[cat] = {
                "before_ms": ms_a,
                "after_ms": ms_b,
                "delta_ms": round(ms_b - ms_a, 3),
            }
    if cat_changes:
        diff["category_changes"] = cat_changes

    hs_a = {h["name"]: h for h in ctx_a.get("top_hotspots", [])}
    hs_b = {h["name"]: h for h in ctx_b.get("top_hotspots", [])}
    changes = []
    for hname in list(hs_a.keys())[:8]:
        if hname in hs_b:
            a_ms = hs_a[hname].get("total_ms", 0)
            b_ms = hs_b[hname].get("total_ms", 0)
            if abs(b_ms - a_ms) > 0.1:
                changes.append({
                    "name": hname,
                    "before_ms": a_ms,
                    "after_ms": b_ms,
                    "delta_ms": round(b_ms - a_ms, 3),
                })
    if changes:
        diff["hotspot_changes"] = sorted(changes, key=lambda c: abs(c["delta_ms"]), reverse=True)

    return diff
