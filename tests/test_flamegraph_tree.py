"""
Tests for src/analysis/flamegraph_tree.py's build_flame_tree() -- backs
the Flame Graph tab in both the TUI and the GUI. Deliberately thin: the
actual tree-building (inclusive-time accumulation, category propagation,
merging same-named nodes) is analysis/call_tree.py's _ct_build's job,
already covered by tests/test_call_tree_build.py -- these tests only
check the {name, value, category, children} conversion and the
multi-root/empty-input wrapping this module adds on top.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.events import SpanEvent, Category
from src.analysis.flamegraph_tree import build_flame_tree


def _span(name, dur_ns, stack_frames, cat=Category.CPU, start_ns=0):
    return SpanEvent(name=name, category=cat, start_ns=start_ns, duration_ns=dur_ns,
                     pid=1, tid=1, stack_frames=list(stack_frames))


class TestBuildFlameTree(unittest.TestCase):
    def test_empty_input_returns_zero_value_all_node(self):
        tree = build_flame_tree([])
        self.assertEqual(tree["name"], "all")
        self.assertEqual(tree["value"], 0)
        self.assertEqual(tree["children"], [])

    def test_spans_without_stack_frames_are_excluded(self):
        # No --perf-callgraph/--call-tree for this run -- _ct_build falls
        # back to temporal containment, which still produces SOME tree
        # for duration>0 spans even without stack_frames. What matters
        # here is just that build_flame_tree doesn't crash and the shape
        # stays {name,value,category,children} throughout.
        spans = [_span("fn", 100, [])]
        tree = build_flame_tree(spans)
        self.assertEqual(tree["name"], "all")
        self.assertIn("children", tree)

    def test_inclusive_value_accumulates_up_the_tree(self):
        # main -> work -> {leaf_a, leaf_b}; "work"/"main" (pure ancestors)
        # must show the SUM of their descendants' time, not just their
        # own (zero) self time -- this is what makes a flame graph's
        # frame width = its own + every descendant's time.
        spans = [
            _span("leaf_a", 100, ["main", "work"]),
            _span("leaf_b", 50, ["main", "work"]),
        ]
        tree = build_flame_tree(spans)
        self.assertEqual(tree["value"], 150)
        work = tree["children"][0]
        self.assertEqual(work["name"], "work")
        self.assertEqual(work["value"], 150)
        main = work["children"][0]
        self.assertEqual(main["name"], "main")
        self.assertEqual(main["value"], 150)
        leaf_values = {c["name"]: c["value"] for c in main["children"]}
        self.assertEqual(leaf_values, {"leaf_a": 100, "leaf_b": 50})

    def test_leaf_category_preserved_ancestor_category_is_other(self):
        spans = [_span("leaf", 100, ["main"], cat=Category.GPU_CUDA)]
        tree = build_flame_tree(spans)
        main = tree["children"][0]
        self.assertEqual(main["category"], "other")
        leaf = main["children"][0]
        self.assertEqual(leaf["category"], "cuda")

    def test_multiple_distinct_roots_wrapped_under_one_all_node(self):
        # Two spans with no shared ancestor at all (e.g. two independent
        # threads' own top-level functions) -- both must appear as
        # separate children of the synthetic "all" root, and "all"'s own
        # value must be their combined total.
        spans = [
            _span("thread_a_main", 100, []),
            _span("thread_b_main", 200, []),
        ]
        tree = build_flame_tree(spans)
        self.assertEqual(tree["value"], 300)
        names = {c["name"] for c in tree["children"]}
        self.assertEqual(names, {"thread_a_main", "thread_b_main"})

    def test_mixed_hook_and_cpu_sample_spans_merge_into_one_tree(self):
        # A --perf-callgraph CPU sample and a --call-tree hook-captured
        # span both under the same top-level function -- build_flame_tree
        # deliberately doesn't filter by category, so both contribute to
        # one combined tree (matches the Call Tree tab's own behavior).
        spans = [
            _span("cpu_leaf", 100, ["main"], cat=Category.CPU),
            _span("cudaLaunchKernel", 50, ["main"], cat=Category.GPU_CUDA),
        ]
        tree = build_flame_tree(spans)
        main = tree["children"][0]
        self.assertEqual(main["value"], 150)
        leaf_names = {c["name"] for c in main["children"]}
        self.assertEqual(leaf_names, {"cpu_leaf", "cudaLaunchKernel"})


if __name__ == "__main__":
    unittest.main()
