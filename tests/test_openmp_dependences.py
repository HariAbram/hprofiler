"""
OpenMP task dependences reconstructed from the GNU libgomp hook's deps=
tags (src/analysis/semantics/openmp.py dependence_pairs / add_task_edges):
the dependence rules per type (in, out/inout, mutexinoutset ordered like out
as libgomp does, inoutset sets, omp_all_memory), siblings only, taskwait
depend(...) waiting for exactly the conflicting earlier siblings, taskwait /
taskgroup membership through parent= / task=, and the diagnostics for
clauses the hook could not decode -- whose dependences stay unknown.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.events import Category, CounterEvent, SpanEvent
from src.core.trace import Trace, TraceMetadata
from src.analysis import criticalpath as cp
from src.analysis.semantics import openmp as sem


def pairs(*tasks):
    """tasks: (id, "deps") -> sorted pairs."""
    got, _ = sem.dependence_pairs([(tid, sem.parse_deps(d)) for tid, d in tasks])
    return sorted(got)


class TestRules(unittest.TestCase):
    def test_readers_wait_for_the_writer_and_the_next_writer_for_the_readers(self):
        self.assertEqual(pairs(("w", "o1"), ("r1", "i1"), ("r2", "i1"), ("w2", "o1")),
                         [("r1", "w2"), ("r2", "w2"), ("w", "r1"), ("w", "r2")])

    def test_writers_in_a_row(self):
        self.assertEqual(pairs(("a", "o1"), ("b", "o1"), ("c", "o1")), [("a", "b"), ("b", "c")])

    def test_different_locations_are_independent(self):
        self.assertEqual(pairs(("a", "o1"), ("b", "o2"), ("c", "i1;i2")), [("a", "c"), ("b", "c")])

    def test_mutexinoutset_is_ordered_like_out(self):
        self.assertEqual(pairs(("a", "m1"), ("b", "m1"), ("c", "i1")), [("a", "b"), ("b", "c")])

    def test_an_inoutset_set_is_unordered_within(self):
        self.assertEqual(pairs(("w", "o1"), ("s1", "s1"), ("s2", "s1"), ("r", "i1")),
                         [("s1", "r"), ("s2", "r"), ("w", "s1"), ("w", "s2")])

    def test_omp_all_memory_orders_everything_around_it(self):
        self.assertEqual(pairs(("a", "o1"), ("b", "i2"), ("all", "o*"), ("c", "i1"), ("d", "o3")),
                         [("a", "all"), ("all", "c"), ("all", "d"), ("b", "all")])

    def test_a_task_without_clauses_takes_part_in_nothing(self):
        self.assertEqual(pairs(("a", "o1"), ("x", ""), ("b", "i1")), [("a", "b")])

    def test_taskwait_depend_waits_for_the_conflicting_siblings_only(self):
        _p, st = sem.dependence_pairs([("w", [("o", "1")]), ("r", [("i", "1")]), ("u", [("o", "2")])])
        self.assertEqual(sorted(sem._preds_for(st["state"], st["all"], [("i", "1")])), ["w"])
        self.assertEqual(sorted(sem._preds_for(st["state"], st["all"], [("o", "1")])), ["r"])


def _span(name, cat, start, dur, tid=1, **tags):
    return SpanEvent(name=name, category=cat, start_ns=start, duration_ns=dur, pid=7, tid=tid,
                     tags={k: str(v) for k, v in tags.items()})


def gnu_trace(extra=()):
    """The implicit task 100 creates t1 out(1), t2 in(1), then a taskwait
    depend(in:1) and a plain taskwait; t3 is created inside a taskgroup and
    creates t4."""
    S, O = Category.SYNC, Category.OPENMP
    ev = [
        _span("omp_task_create", O, 10, 0, type="task_create", id="t1", team=1, parent="100", depend=1,
              deps="o1", ndeps=1),
        _span("omp_task_create", O, 20, 0, type="task_create", id="t2", team=1, parent="100", depend=1,
              deps="i1", ndeps=1),
        _span("omp_task", O, 30, 100, tid=2, type="task", id="t1", team=1),
        _span("omp_task", O, 140, 50, tid=3, type="task", id="t2", team=1),
        _span("omp_taskwait", S, 40, 120, type="sync", team=1, task="100", depend=1, deps="i1", ndeps=1),
        _span("omp_taskwait", S, 200, 10, type="sync", team=1, task="100"),
        _span("omp_task_create", O, 300, 0, type="task_create", id="t3", team=1, parent="100"),
        _span("omp_task", O, 310, 100, tid=2, type="task", id="t3", team=1),
        _span("omp_task_create", O, 320, 0, tid=2, type="task_create", id="t4", team=1, parent="t3"),
        _span("omp_task", O, 330, 200, tid=3, type="task", id="t4", team=1),
        _span("omp_taskgroup", S, 290, 300, type="sync", team=1, task="100"),
    ] + list(extra)
    t = Trace(TraceMetadata())
    for e in ev:
        t.add(e)
    return t


def edges(trace, kind):
    spans, preds = cp.build_dependency_graph(trace)
    g = cp.Graph(trace, trace.store.span_seqs(), cp.build_edges(trace))
    mem = sorted((u, v, c) for v, pl in preds.items() for u, k, c in pl if k == kind)
    streamed = sorted((u, v, c) for v, pl in g.preds_dict().items() for u, k, c in pl if k == kind)
    assert mem == streamed, (mem, streamed)

    def lab(s):
        return s.tags.get("id") or f"{s.name}@{s.start_ns}"
    return sorted((lab(spans[u]), lab(spans[v]), c) for u, v, c in mem)


class TestGraph(unittest.TestCase):
    def test_reconstructed_dependence_is_high_not_certain(self):
        self.assertEqual(edges(gnu_trace(), "task_dep"), [("t1", "t2", "high")])

    def test_joins(self):
        got = edges(gnu_trace(), "task_join")
        # taskwait depend(in:1): the writer t1 only; the plain taskwait: t1 and t2;
        # the taskgroup: t3 and its descendant t4
        self.assertIn(("t1", "omp_taskwait@40", "high"), got)
        self.assertNotIn(("t2", "omp_taskwait@40", "high"), got)
        self.assertIn(("t1", "omp_taskwait@200", "certain"), got)
        self.assertIn(("t2", "omp_taskwait@200", "certain"), got)
        self.assertIn(("t3", "omp_taskgroup@290", "certain"), got)
        self.assertIn(("t4", "omp_taskgroup@290", "certain"), got)

    def test_undecoded_clauses_give_no_edges_and_are_reported(self):
        S, O = Category.SYNC, Category.OPENMP
        t = gnu_trace([
            _span("omp_task_create", O, 400, 0, type="task_create", id="t5", team=1, parent="100", depend=1,
                  deps_unsupported="abi"),
            _span("omp_task", O, 410, 10, tid=2, type="task", id="t5", team=1),
            _span("omp_taskwait", S, 430, 5, type="sync", team=1, task="100", depend=1,
                  deps_unsupported="abi"),
        ])
        t.add(CounterEvent(name="gomp_abi_level", category=Category.OPENMP, timestamp_ns=1, value=60000,
                           pid=7, unit="version"))
        self.assertNotIn("t5", [d for _s, d, _c in edges(t, "task_dep")])
        self.assertFalse([e for e in edges(t, "task_join") if e[1] == "omp_taskwait@430"])
        diag = cp.dependency_diagnostics(t)
        self.assertEqual(diag["omp_deps_unsupported"], {"abi": 2})
        self.assertEqual(diag["omp_gomp_abi_level"], 60000)
        self.assertTrue(any("not decoded" in w and "unknown" in w for w in cp.dependency_warnings(diag)))

    def test_an_older_gnu_hook_is_reported(self):
        t = Trace(TraceMetadata())
        t.add(_span("omp_task_create", Category.OPENMP, 1, 0, type="task_create", id="x", team=1, depend=1))
        self.assertEqual(cp.dependency_diagnostics(t)["omp_gnu_legacy_tasks"], 1)


if __name__ == "__main__":
    unittest.main()
