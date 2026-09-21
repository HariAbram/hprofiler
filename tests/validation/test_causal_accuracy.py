"""
Accuracy validation suite for the causal-attribution dependency graph
(src/analysis/criticalpath.py) -- distinct in purpose from tests/test_
criticalpath.py's unit tests and tests/integration/run_matrix.sh's crash-
safety matrix.

This is the direct response to the critique that "current integration
tests mostly verify 'does not crash'... not validation of measurement
correctness": rather than individual pass/fail assertions on one scenario
at a time, this builds a battery of synthetic scenarios with KNOWN ground-
truth edge sets (constructed by hand, so the correct answer is known
exactly, not estimated), runs the real dependency-graph builder against
each, and reports aggregate PRECISION and RECALL -- both overall and
broken down per confidence tier (certain/high/medium -- see criticalpath.
py's module docstring) -- across the whole battery. A silent regression
that starts producing a few spurious edges, or starts missing some it used
to find, shows up as a precision/recall drop here even if every individual
unit test elsewhere still happens to pass on its own narrower assertion.

Also includes a determinism ("perturbation") check: the same logical
scenario, with its spans inserted in several different orders, must
produce byte-identical results -- catching any latent dependence on dict/
set iteration order that a single fixed-order unit test could not surface.

Run directly (not part of `python3 -m unittest discover -s tests`, which
only looks in tests/ itself, not tests/validation/):
    python3 -m unittest discover -s tests/validation -p "test_*.py" -v
or:
    python3 tests/validation/test_causal_accuracy.py
"""
from __future__ import annotations

import random
import sys
import unittest
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.core.trace import Trace, TraceMetadata
from src.core.events import SpanEvent, Category
from src.analysis import criticalpath as cp


def _span(pid, tid, cat, start_ns, dur_ns, name, label=None, tags=None, span_id="", parent_span_id=""):
    # `label` is this test suite's own stable per-span identity, stored in
    # a tag the algorithm never looks at -- decoupled from `name` (which
    # the algorithm DOES inspect for OMP barrier recognition, and which
    # can legitimately be identical across several spans, e.g. every
    # thread's barrier call is literally named "omp_barrier_implicit") and
    # from `span_id` (used for real sid=/psid= matching). Defaults to
    # `name` for the common case where names already happen to be unique.
    tags = dict(tags or {})
    tags["_label"] = label if label is not None else name
    return SpanEvent(name=name, category=cat, start_ns=start_ns, duration_ns=dur_ns,
                     pid=pid, tid=tid, tags=tags,
                     span_id=span_id, parent_span_id=parent_span_id)


# An expected edge, identified by each span's test-only "_label" tag
# (unique within its scenario, stable under reordering) -- NOT span.name,
# which the algorithm itself may require to take a specific repeated value
# (e.g. OMP barrier recognition) independent of this suite's own
# bookkeeping.
ExpectedEdge = tuple[str, str, str, str]  # (pred_label, succ_label, kind, confidence)


@dataclass
class Scenario:
    name: str
    spans: list[SpanEvent]
    expected_edges: set[ExpectedEdge]
    # Names of spans that MUST NOT be a predecessor of the given span in
    # the produced graph, for scenarios specifically testing that a false
    # edge is correctly NOT created (e.g. two unrelated communicators).
    forbidden_edges: set[ExpectedEdge] = field(default_factory=set)


def _mk_trace(spans: list[SpanEvent]) -> Trace:
    t = Trace(TraceMetadata())
    for s in spans:
        t.add(s)
    return t


# ── scenario generators ─────────────────────────────────────────────────

def scenario_program_order_chain() -> Scenario:
    spans = [_span(1, 1, Category.CPU, i * 100, 50, f"po_{i}") for i in range(5)]
    expected = {(f"po_{i}", f"po_{i+1}", "sequential", "certain") for i in range(4)}
    return Scenario("program_order_chain", spans, expected)


def scenario_mpi_exact_match() -> Scenario:
    spans = []
    expected: set[ExpectedEdge] = set()
    for i in range(5):
        send = _span(0, 1, Category.MPI, i * 1000, 10, f"send_{i}",
                     tags={"type": "send", "rank": "0", "peer": "1", "tag": str(i)})
        recv = _span(1, 1, Category.MPI, i * 1000 + 5, 10, f"recv_{i}",
                     tags={"type": "recv", "rank": "1", "peer": "0", "tag": str(i)})
        spans += [send, recv]
        expected.add((f"send_{i}", f"recv_{i}", "p2p", "medium"))
    return Scenario("mpi_exact_match", spans, expected)


def scenario_mpi_wildcard_match() -> Scenario:
    spans = []
    expected: set[ExpectedEdge] = set()
    for i in range(5):
        send = _span(0, 1, Category.MPI, i * 1000, 10, f"wsend_{i}",
                     tags={"type": "send", "rank": "0", "peer": "1", "tag": str(i)})
        # mpi_hook.c resolves wildcard recvs' peer/tag directly on their own
        # span (see its file header) -- the recv's tags ARE the resolved
        # values already.
        recv = _span(1, 1, Category.MPI, i * 1000 + 5, 10, f"wrecv_{i}",
                     tags={"type": "recv", "rank": "1", "peer": "0", "tag": str(i), "wildcard": "1"})
        spans += [send, recv]
        expected.add((f"wsend_{i}", f"wrecv_{i}", "p2p", "high"))
    return Scenario("mpi_wildcard_match", spans, expected)


def scenario_async_mpi_wait() -> Scenario:
    # _index_mpi_completers (criticalpath.py) specifically matches on the
    # real MPI_Wait/Waitall/Waitany/Waitsome names -- a generically-named
    # "wait" span (as an earlier version of this scenario used) would
    # never be recognized as a completer at all, silently producing a
    # false negative that looked like an algorithm bug but was actually a
    # scenario-construction mistake. Use the real names throughout.
    isend = _span(0, 1, Category.MPI, 0, 10, "MPI_Isend", label="isend",
                  tags={"type": "isend", "rank": "0", "peer": "1", "tag": "9"})
    irecv = _span(1, 1, Category.MPI, 0, 3, "MPI_Irecv", label="irecv", span_id="req1",
                  tags={"type": "irecv", "rank": "1", "peer": "0", "tag": "9"})
    wait = _span(1, 1, Category.MPI, 3, 200, "MPI_Wait", label="wait", parent_span_id="req1")
    spans = [isend, irecv, wait]
    expected = {
        ("isend", "wait", "p2p", "medium"),
        ("irecv", "wait", "explicit_span_id", "certain"),
    }
    return Scenario("async_mpi_wait", spans, expected)


def scenario_commid_scoped_rendezvous_no_cross_talk() -> Scenario:
    """Two unrelated communicators, each with its own rendezvous, whose
    wall-clock ranges deliberately overlap -- the exact false-positive
    case commid-scoping exists to prevent (see criticalpath.py's
    _add_rendezvous_edges). Ground truth: each cluster's non-last-arriver
    depends on its OWN last arriver; NEVER on the other cluster's."""
    a1 = _span(1, 1, Category.MPI, 0, 100, "a1", tags={"type": "allreduce", "commid": "1"})
    a2 = _span(2, 1, Category.MPI, 10, 80, "a2", tags={"type": "allreduce", "commid": "1"})
    b1 = _span(3, 1, Category.MPI, 5, 90, "b1", tags={"type": "allreduce", "commid": "2"})
    b2 = _span(4, 1, Category.MPI, 50, 100, "b2", tags={"type": "allreduce", "commid": "2"})
    spans = [a1, a2, b1, b2]
    expected = {("a2", "a1", "arrival", "high"), ("b2", "b1", "arrival", "high")}
    forbidden = {
        ("b1", "a1", "arrival", "high"), ("b2", "a1", "arrival", "high"),
        ("a1", "b1", "arrival", "high"), ("a2", "b1", "arrival", "high"),
    }
    return Scenario("commid_scoped_rendezvous_no_cross_talk", spans, expected, forbidden)


def scenario_device_sync_chain() -> Scenario:
    k1 = _span(0, 1, Category.GPU_CUDA, 0, 50, "k1")
    k2 = _span(0, 1, Category.GPU_CUDA, 50, 50, "k2")
    sync = _span(0, 1, Category.SYNC, 100, 10, "cudaDeviceSynchronize")
    spans = [k1, k2, sync]
    expected = {
        ("k1", "cudaDeviceSynchronize", "device_sync", "certain"),
        ("k2", "cudaDeviceSynchronize", "device_sync", "certain"),
        ("k1", "k2", "sequential", "certain"),  # program order, same thread
    }
    return Scenario("device_sync_chain", spans, expected)


def scenario_omp_barrier() -> Scenario:
    # Both spans MUST literally be named "omp_barrier_implicit" for
    # _add_omp_barrier_edges to recognize and cluster them together --
    # label= gives this suite its own distinct per-span identity anyway.
    b1 = _span(1, 1, Category.SYNC, 0, 100, "omp_barrier_implicit", label="bar_t1")
    b2 = _span(1, 2, Category.SYNC, 20, 90, "omp_barrier_implicit", label="bar_t2")
    spans = [b1, b2]
    # bar_t2 starts later (20 > 0) -- it's the last arriver, so it's the
    # PREDECESSOR of bar_t1 (bar_t1 depends on/waits for it), not the other
    # way around -- see _add_last_arriver_edges.
    expected = {("bar_t2", "bar_t1", "arrival", "certain")}
    return Scenario("omp_barrier", spans, expected)


ALL_SCENARIOS = [
    scenario_program_order_chain,
    scenario_mpi_exact_match,
    scenario_mpi_wildcard_match,
    scenario_async_mpi_wait,
    scenario_commid_scoped_rendezvous_no_cross_talk,
    scenario_device_sync_chain,
    scenario_omp_barrier,
]


def _produced_edges(spans: list[SpanEvent], preds: dict) -> set[ExpectedEdge]:
    by_idx_label = {i: s.tags["_label"] for i, s in enumerate(spans)}
    out: set[ExpectedEdge] = set()
    for v, edges in preds.items():
        for (u, kind, conf) in edges:
            out.add((by_idx_label[u], by_idx_label[v], kind, conf))
    return out


@dataclass
class AccuracyReport:
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    per_tier: dict[str, dict[str, int]] = field(default_factory=lambda: {
        "certain": {"tp": 0, "fp": 0, "fn": 0},
        "high": {"tp": 0, "fp": 0, "fn": 0},
        "medium": {"tp": 0, "fp": 0, "fn": 0},
    })

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom else 1.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom else 1.0


def evaluate(scenarios: list[Scenario]) -> tuple[AccuracyReport, list[str]]:
    report = AccuracyReport()
    failures: list[str] = []
    for sc in scenarios:
        trace = _mk_trace(sc.spans)
        spans, preds = cp.build_dependency_graph(trace)
        produced = _produced_edges(spans, preds)

        tp = sc.expected_edges & produced
        fn = sc.expected_edges - produced
        fp_forbidden = sc.forbidden_edges & produced

        report.true_positives += len(tp)
        report.false_negatives += len(fn)
        report.false_positives += len(fp_forbidden)
        for (_p, _s, _k, conf) in tp:
            report.per_tier.setdefault(conf, {"tp": 0, "fp": 0, "fn": 0})["tp"] += 1
        for (_p, _s, _k, conf) in fn:
            report.per_tier.setdefault(conf, {"tp": 0, "fp": 0, "fn": 0})["fn"] += 1
        for (_p, _s, _k, conf) in fp_forbidden:
            report.per_tier.setdefault(conf, {"tp": 0, "fp": 0, "fn": 0})["fp"] += 1

        if fn:
            failures.append(f"[{sc.name}] missing expected edges: {fn}")
        if fp_forbidden:
            failures.append(f"[{sc.name}] produced forbidden edges: {fp_forbidden}")

    return report, failures


class TestCausalAccuracy(unittest.TestCase):
    def test_precision_and_recall_across_all_scenarios(self):
        report, failures = evaluate([s() for s in ALL_SCENARIOS])
        for f in failures:
            print(f"  {f}")
        print(f"\n  Overall: precision={report.precision:.3f} recall={report.recall:.3f} "
              f"(tp={report.true_positives} fp={report.false_positives} fn={report.false_negatives})")
        for tier, counts in sorted(report.per_tier.items()):
            denom_p = counts["tp"] + counts["fp"]
            denom_r = counts["tp"] + counts["fn"]
            p = counts["tp"] / denom_p if denom_p else float("nan")
            r = counts["tp"] / denom_r if denom_r else float("nan")
            print(f"    {tier:<10} precision={p:.3f} recall={r:.3f} "
                 f"(tp={counts['tp']} fp={counts['fp']} fn={counts['fn']})")

        self.assertEqual(failures, [])
        self.assertEqual(report.precision, 1.0)
        self.assertEqual(report.recall, 1.0)


class TestDeterminism(unittest.TestCase):
    """A silent dependence on dict/set iteration order would be a real
    correctness bug (the same underlying trace producing a different
    reported critical path from one run to the next) -- checked directly
    by re-running each scenario with its spans inserted in several
    different orders and confirming byte-identical results, not just
    re-running the same call twice (which would trivially be
    deterministic even with a latent order-dependence bug, since nothing
    about the input actually changes between identical calls)."""

    def test_edge_set_and_critical_path_stable_under_input_reordering(self):
        rng = random.Random(1234)
        for make_scenario in ALL_SCENARIOS:
            sc = make_scenario()
            baseline_trace = _mk_trace(sc.spans)
            baseline_spans, baseline_preds = cp.build_dependency_graph(baseline_trace)
            baseline_edges = _produced_edges(baseline_spans, baseline_preds)
            baseline_path, baseline_conf = cp.compute_critical_path_with_confidence(
                baseline_spans, baseline_preds)
            baseline_path_names = [baseline_spans[i].name for i in baseline_path]

            for trial in range(5):
                shuffled = list(sc.spans)
                rng.shuffle(shuffled)
                trace = _mk_trace(shuffled)
                spans, preds = cp.build_dependency_graph(trace)
                edges = _produced_edges(spans, preds)
                self.assertEqual(edges, baseline_edges,
                                 f"[{sc.name}] trial {trial}: edge set changed under reordering")
                path, conf = cp.compute_critical_path_with_confidence(spans, preds)
                path_names = [spans[i].name for i in path]
                self.assertEqual(path_names, baseline_path_names,
                                 f"[{sc.name}] trial {trial}: critical path changed under reordering")
                self.assertEqual(conf, baseline_conf,
                                 f"[{sc.name}] trial {trial}: path confidence changed under reordering")


if __name__ == "__main__":
    unittest.main()
