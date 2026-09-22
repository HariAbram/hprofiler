"""
Regression test for src/ui/app.py's _ct_build_from_stacks -- the from-
stacks call-tree builder shared by the TUI's Call Tree tab and the Qt/
QML GUI's Call Tree screen (src/gui/bridge.py's CallTreeBridge).

Found via the GUI's Call Tree screen with a deliberately two-level-deep
synthetic trace (a function that BOTH does its own measured work AND
calls sub-functions -- an extremely common, realistic shape, not an edge
case): a function appearing both as an intermediate ancestor frame (for
its children's spans) and as its own directly-measured leaf span used to
split into two same-named SIBLING nodes at the same tree level instead
of merging into one, because the two code paths that create nodes used
different dict keys for what should be the same logical node (bare name
during the frame-path walk vs f"__leaf__{name}" for the span's own
leaf-add step). The TUI's Call Tree tab shares this exact function, so
it had the identical bug for any real profiled program with this shape,
not something specific to the GUI.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.events import SpanEvent, Category
from src.ui.app import _ct_build_from_stacks, _ct_build


def _span(name, start_ns, dur_ns, stack_frames):
    """`stack_frames` is the CALLER chain only, innermost (nearest)
    caller first (real backtrace() order) -- must NOT include `name`
    itself, which is added as the definite leaf separately."""
    return SpanEvent(name=name, category=Category.CPU, start_ns=start_ns,
                     duration_ns=dur_ns, pid=1, tid=1, tags={}, stack_frames=stack_frames)


class TestCallTreeMerging(unittest.TestCase):
    def test_function_that_is_both_ancestor_and_leaf_merges_into_one_node(self):
        # simulate_step: measured directly (9ms) AND is the immediate
        # caller of compute_forces/update_positions -- must be ONE node
        # in main's children, not two same-named siblings.
        spans = [
            _span("simulate_step", 0, 9_000_000, ["main"]),
            _span("compute_forces", 0, 800_000, ["simulate_step", "main"]),
            _span("compute_forces", 1_000_000, 800_000, ["simulate_step", "main"]),
            _span("update_positions", 2_000_000, 600_000, ["simulate_step", "main"]),
        ]
        roots = _ct_build_from_stacks(spans)
        self.assertEqual(len(roots), 1)
        main = roots[0]
        self.assertEqual(main.name, "main")

        simulate_step_children = [c for c in main.children if c.name == "simulate_step"]
        self.assertEqual(len(simulate_step_children), 1,
                         f"expected exactly one merged 'simulate_step' node, "
                         f"found {len(simulate_step_children)}: "
                         f"{[c.name for c in main.children]}")

        merged = simulate_step_children[0]
        # count reflects only the direct span (1), not additionally
        # incremented just for being an ancestor of compute_forces/
        # update_positions' spans.
        self.assertEqual(merged.count, 1)
        child_names = sorted(c.name for c in merged.children)
        self.assertEqual(child_names, ["compute_forces", "update_positions"])

        compute_forces = next(c for c in merged.children if c.name == "compute_forces")
        self.assertEqual(compute_forces.count, 2)
        self.assertEqual(compute_forces.total_ns, 1_600_000)

    def test_ancestor_only_frame_gets_other_category_until_seen_as_a_leaf(self):
        # A frame that's NEVER itself a measured span (only ever an
        # ancestor) keeps category "other" and count 0 -- it contributes
        # inclusive time but was never the actual measured call.
        spans = [_span("leaf_fn", 0, 100, ["never_measured", "main"])]
        roots = _ct_build_from_stacks(spans)
        main = roots[0]
        never_measured = main.children[0]
        self.assertEqual(never_measured.name, "never_measured")
        self.assertEqual(never_measured.category, "other")
        self.assertEqual(never_measured.count, 0)
        self.assertEqual(never_measured.total_ns, 100)

    def test_single_frame_stack_produces_direct_child_of_root(self):
        spans = [_span("work", 0, 500, ["main"])]
        roots = _ct_build_from_stacks(spans)
        self.assertEqual(len(roots), 1)
        self.assertEqual(roots[0].name, "main")
        self.assertEqual(len(roots[0].children), 1)
        self.assertEqual(roots[0].children[0].name, "work")
        self.assertEqual(roots[0].children[0].count, 1)

    def test_ct_build_falls_back_to_temporal_containment_without_stacks(self):
        # Non-regression: spans with no stack_frames at all must still
        # produce a tree via the temporal-containment path, not crash or
        # silently return nothing.
        spans = [
            SpanEvent(name="outer", category=Category.CPU, start_ns=0, duration_ns=1000,
                     pid=1, tid=1, tags={}),
            SpanEvent(name="inner", category=Category.CPU, start_ns=100, duration_ns=200,
                     pid=1, tid=1, tags={}),
        ]
        roots = _ct_build(spans)
        self.assertTrue(roots)


if __name__ == "__main__":
    unittest.main()
