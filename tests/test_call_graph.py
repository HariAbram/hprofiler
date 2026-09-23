"""
Tests for src/analysis/call_graph.py: build_call_graph() (nodes/edges
from span stack_frames) and layout_call_graph() (caps + assigns each
node a normalized [0,1] position via BFS layering). Backs the GUI
Timeline's call-graph panel.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.events import SpanEvent, Category
from src.analysis.call_graph import build_call_graph, layout_call_graph


def _span(name, stack_frames, dur_ns=1000, cat=Category.CPU):
    return SpanEvent(name=name, category=cat, start_ns=0, duration_ns=dur_ns,
                     pid=1, tid=1, stack_frames=list(stack_frames))


class TestBuildCallGraph(unittest.TestCase):
    def test_simple_chain_produces_expected_nodes_and_edges(self):
        # stack_frames is innermost-first (backtrace order): "b" called
        # "a" (the span itself), so reversed -> ["main", "b"] + ["a"]
        spans = [_span("a", ["b", "main"], dur_ns=100)]
        nodes, edges = build_call_graph(spans)
        names = {n.name for n in nodes}
        self.assertEqual(names, {"main", "b", "a"})
        edge_pairs = {(e.caller, e.callee) for e in edges}
        self.assertEqual(edge_pairs, {("main", "b"), ("b", "a")})

    def test_leaf_node_gets_self_time_ancestor_nodes_do_not(self):
        spans = [_span("leaf_fn", ["mid_fn", "main"], dur_ns=500)]
        nodes, _ = build_call_graph(spans)
        by_name = {n.name: n for n in nodes}
        self.assertEqual(by_name["leaf_fn"].total_ns, 500)
        self.assertEqual(by_name["leaf_fn"].count, 1)
        # main/mid_fn are pure ancestors here -- no self time of their own
        self.assertEqual(by_name["main"].total_ns, 0)
        self.assertEqual(by_name["mid_fn"].total_ns, 0)

    def test_same_function_called_from_two_callers_merges_into_one_node(self):
        # This is THE key difference from call_tree.py's tree: "worker"
        # appears under two different callers, but is still ONE node.
        spans = [
            _span("worker", ["path_a", "main"], dur_ns=100),
            _span("worker", ["path_b", "main"], dur_ns=200),
        ]
        nodes, edges = build_call_graph(spans)
        worker_nodes = [n for n in nodes if n.name == "worker"]
        self.assertEqual(len(worker_nodes), 1)
        self.assertEqual(worker_nodes[0].total_ns, 300)
        self.assertEqual(worker_nodes[0].count, 2)
        edge_pairs = {(e.caller, e.callee) for e in edges}
        self.assertIn(("path_a", "worker"), edge_pairs)
        self.assertIn(("path_b", "worker"), edge_pairs)

    def test_edge_weights_aggregate_across_repeated_calls(self):
        spans = [_span("fn", ["caller"], dur_ns=100) for _ in range(5)]
        _, edges = build_call_graph(spans)
        edge = next(e for e in edges if e.caller == "caller" and e.callee == "fn")
        self.assertEqual(edge.count, 5)
        self.assertEqual(edge.total_ns, 500)

    def test_spans_without_stack_frames_are_skipped(self):
        spans = [_span("no_stack", [], dur_ns=100)]
        nodes, edges = build_call_graph(spans)
        self.assertEqual(nodes, [])
        self.assertEqual(edges, [])

    def test_zero_duration_spans_are_skipped(self):
        spans = [_span("instant", ["main"], dur_ns=0)]
        nodes, edges = build_call_graph(spans)
        self.assertEqual(nodes, [])

    def test_nodes_sorted_by_total_ns_descending(self):
        spans = [
            _span("hot", ["main"], dur_ns=1000),
            _span("cold", ["main"], dur_ns=10),
        ]
        nodes, _ = build_call_graph(spans)
        leaf_nodes = [n for n in nodes if n.name in ("hot", "cold")]
        self.assertEqual([n.name for n in leaf_nodes], ["hot", "cold"])


class TestLayoutCallGraph(unittest.TestCase):
    def test_simple_chain_gets_increasing_layers(self):
        spans = [_span("c", ["b", "a"], dur_ns=100)]
        nodes, edges = build_call_graph(spans)
        layout = layout_call_graph(nodes, edges)
        by_name = {n["name"]: n for n in layout["nodes"]}
        self.assertLess(by_name["a"]["x"], by_name["b"]["x"])
        self.assertLess(by_name["b"]["x"], by_name["c"]["x"])

    def test_all_x_y_within_unit_range(self):
        spans = [
            _span("leaf1", ["mid1", "root"], dur_ns=100),
            _span("leaf2", ["mid2", "root"], dur_ns=200),
            _span("leaf1", ["mid2", "root"], dur_ns=50),
        ]
        nodes, edges = build_call_graph(spans)
        layout = layout_call_graph(nodes, edges)
        for n in layout["nodes"]:
            self.assertGreaterEqual(n["x"], 0.0)
            self.assertLessEqual(n["x"], 1.0)
            self.assertGreaterEqual(n["y"], 0.0)
            self.assertLessEqual(n["y"], 1.0)

    def test_edges_reference_valid_node_indices(self):
        spans = [_span("c", ["b", "a"], dur_ns=100)]
        nodes, edges = build_call_graph(spans)
        layout = layout_call_graph(nodes, edges)
        n_nodes = len(layout["nodes"])
        for e in layout["edges"]:
            self.assertGreaterEqual(e["callerIdx"], 0)
            self.assertLess(e["callerIdx"], n_nodes)
            self.assertGreaterEqual(e["calleeIdx"], 0)
            self.assertLess(e["calleeIdx"], n_nodes)

    def test_caps_to_max_nodes_and_reports_truncated_count(self):
        spans = [_span(f"leaf{i}", [f"caller{i}"], dur_ns=1000 - i) for i in range(20)]
        nodes, edges = build_call_graph(spans)
        self.assertEqual(len(nodes), 40)  # 20 leaves + 20 distinct callers
        layout = layout_call_graph(nodes, edges, max_nodes=10)
        self.assertEqual(len(layout["nodes"]), 10)
        self.assertEqual(layout["truncated"], 30)

    def test_self_recursion_does_not_infinite_loop(self):
        # A function whose OWN name appears in its own stack_frames
        # (direct recursion) -- reversed(stack_frames)+[name] would put
        # "recur" adjacent to itself; build_call_graph's `caller !=
        # callee` filter in layout must keep this from being treated as
        # an edge that could stall BFS layering.
        spans = [_span("recurse", ["recurse", "main"], dur_ns=100)]
        nodes, edges = build_call_graph(spans)
        layout = layout_call_graph(nodes, edges)  # must return, not hang
        self.assertTrue(len(layout["nodes"]) > 0)

    def test_empty_input(self):
        layout = layout_call_graph([], [])
        self.assertEqual(layout["nodes"], [])
        self.assertEqual(layout["edges"], [])
        self.assertEqual(layout["truncated"], 0)
        self.assertEqual(layout["numLayers"], 0)
        self.assertEqual(layout["maxLayerSize"], 0)

    def test_nodes_carry_raw_layer_and_layer_index(self):
        # x/y are normalized [0,1] -- can't reconstruct "how many layers
        # total" or "how many nodes share this layer" from them alone.
        # layer/layerIndex are the raw ints a pixel-based QML layout
        # needs to give every node a fixed size/spacing regardless of
        # how many nodes/layers there are.
        spans = [
            _span("leaf1", ["mid1", "root"], dur_ns=100),
            _span("leaf2", ["mid2", "root"], dur_ns=200),
        ]
        nodes, edges = build_call_graph(spans)
        layout = layout_call_graph(nodes, edges)
        by_name = {n["name"]: n for n in layout["nodes"]}
        self.assertEqual(by_name["root"]["layer"], 0)
        self.assertEqual(by_name["mid1"]["layer"], 1)
        self.assertEqual(by_name["leaf1"]["layer"], 2)
        # mid1/mid2 are both in layer 1 -- distinct layerIndex values.
        self.assertNotEqual(by_name["mid1"]["layerIndex"], by_name["mid2"]["layerIndex"])

    def test_num_layers_and_max_layer_size_match_the_layout(self):
        spans = [
            _span("leaf1", ["mid1", "root"], dur_ns=100),
            _span("leaf2", ["mid2", "root"], dur_ns=200),
            _span("leaf3", ["mid2", "root"], dur_ns=50),
        ]
        nodes, edges = build_call_graph(spans)
        layout = layout_call_graph(nodes, edges)
        # root(0) -> mid1/mid2(1) -> leaf1/leaf2/leaf3(2): 3 layers.
        self.assertEqual(layout["numLayers"], 3)
        # Layer 2 (leaf1, leaf2, leaf3) is the widest, at 3 nodes.
        self.assertEqual(layout["maxLayerSize"], 3)


if __name__ == "__main__":
    unittest.main()
