"""
Tests for src/gui/models.py's TimelineModel -- the Timeline screen's data
layer (Phase 4). Skipped if PySide6 isn't installed. See
tests/test_gui_bridge.py's docstring for why these run headless/offscreen.
"""
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtGui import QGuiApplication
    _PYSIDE6_AVAILABLE = True
except ImportError:
    _PYSIDE6_AVAILABLE = False

from src.core.trace import Trace, TraceMetadata
from src.core.events import SpanEvent, Category


def _span(pid, tid, cat, start_ns, dur_ns, name, tags=None):
    return SpanEvent(name=name, category=cat, start_ns=start_ns, duration_ns=dur_ns,
                     pid=pid, tid=tid, tags=dict(tags or {}))


def _mk_trace(spans):
    t = Trace(TraceMetadata())
    for s in spans:
        t.add(s)
    return t


@unittest.skipUnless(_PYSIDE6_AVAILABLE, "PySide6 not installed")
class TestTimelineModel(unittest.TestCase):
    _app = None

    @classmethod
    def setUpClass(cls):
        cls._app = QGuiApplication.instance() or QGuiApplication([])

    def _model(self, trace):
        from src.gui.theme import Theme
        from src.gui.models import TimelineModel
        return TimelineModel(trace, Theme(dark=True))

    def test_lanes_labeled_by_mpi_rank_not_generic_thread_number(self):
        a = _span(100, 101, Category.MPI, 0, 10, "MPI_Send", tags={"rank": "5"})
        m = self._model(_mk_trace([a]))
        labels = [l["label"] for l in m.lanes]
        self.assertTrue(any("rank5" in l for l in labels))

    def test_non_mpi_lane_uses_sequential_thread_label(self):
        a = _span(100, 101, Category.CPU, 0, 10, "kernel")
        m = self._model(_mk_trace([a]))
        labels = [l["label"] for l in m.lanes]
        self.assertTrue(any("T1" in l for l in labels))

    def test_lane_colors_are_hex(self):
        a = _span(100, 101, Category.GPU_CUDA, 0, 10, "k")
        m = self._model(_mk_trace([a]))
        for lane in m.lanes:
            self.assertTrue(lane["color"].startswith("#"))

    def test_visible_spans_culls_to_viewport(self):
        # 100 spans spread across a wide range -- a narrow viewport must
        # only return the ones actually overlapping it.
        spans = [_span(1, 1, Category.CPU, i * 1_000_000, 1000, f"s{i}") for i in range(100)]
        m = self._model(_mk_trace(spans))
        lane_idx = 0
        all_visible = m.visibleSpans(lane_idx, 0, 100_000_000, 1000)
        self.assertEqual(len(all_visible), 100)
        narrow = m.visibleSpans(lane_idx, 0, 5_000_000, 1000)
        self.assertLess(len(narrow), 100)
        self.assertGreaterEqual(len(narrow), 5)

    def test_visible_spans_out_of_range_lane_index_returns_empty(self):
        m = self._model(_mk_trace([_span(1, 1, Category.CPU, 0, 10, "x")]))
        self.assertEqual(m.visibleSpans(99, 0, 1000, 100), [])
        self.assertEqual(m.visibleSpans(-1, 0, 1000, 100), [])

    def test_visible_spans_lo_bound_uses_per_span_duration_not_lane_span(self):
        # Regression test for a real bug: _max_dur used to be the lane's
        # FULL first-to-last time range (ends.max()-starts.min()), not the
        # longest INDIVIDUAL span's own duration -- so searchsorted's
        # look-back margin was wildly oversized for any lane whose spans
        # spread across most of the trace, pinning `lo` near index 0
        # regardless of how far into the trace the query window actually
        # was. One long early span (duration 1000) followed by a cluster
        # of short spans far later -- querying a window that starts well
        # after the early span's TRUE end (but still within the old,
        # bogus, whole-lane-range look-back) must not resurrect it.
        early = _span(1, 1, Category.OPENMP, 0, 1000, "early_long_span")
        late = [_span(1, 1, Category.OPENMP, 10_000_000 + i * 100, 50, f"late{i}")
                for i in range(5)]
        m = self._model(_mk_trace([early] + late))
        lane_idx = 0
        result = m.visibleSpans(lane_idx, 9_000_000, 9_500_000, 2000)
        self.assertEqual(result, [])
        names = {r["name"] for r in m.visibleSpans(lane_idx, 10_000_000, 10_001_000, 2000)}
        self.assertNotIn("early_long_span", names)
        self.assertTrue(any(n.startswith("late") for n in names))

    def test_visible_spans_preserves_idle_gap_between_two_clusters(self):
        # Two clusters of spans with a genuine, deliberate idle gap
        # between them (no span at all covers that time) -- querying
        # squarely inside the gap must return nothing, not spans smeared
        # in from either cluster.
        cluster_a = [_span(1, 1, Category.OPENMP, i * 1000, 500, f"a{i}") for i in range(20)]
        cluster_b = [_span(1, 1, Category.OPENMP, 100_000 + i * 1000, 500, f"b{i}") for i in range(20)]
        m = self._model(_mk_trace(cluster_a + cluster_b))
        lane_idx = 0
        gap = m.visibleSpans(lane_idx, 40_000, 60_000, 2000)
        self.assertEqual(gap, [])

    def test_visible_spans_max_dur_is_longest_single_span_not_lane_range(self):
        a = _span(1, 1, Category.CPU, 0, 500, "a")
        b = _span(1, 1, Category.CPU, 1_000_000, 30, "b")
        m = self._model(_mk_trace([a, b]))
        self.assertEqual(m._max_dur["cpu/thread-1"], 500)

    def test_visible_spans_respects_max_spans_cap(self):
        spans = [_span(1, 1, Category.CPU, i * 100, 50, f"s{i}") for i in range(500)]
        m = self._model(_mk_trace(spans))
        capped = m.visibleSpans(0, 0, 100_000, 50)
        self.assertLessEqual(len(capped), 60)  # cap=50 plus stepping slack

    def test_span_at_returns_full_detail(self):
        a = _span(1, 1, Category.MPI, 1000, 500, "MPI_Bcast", tags={"rank": "2"})
        m = self._model(_mk_trace([a]))
        detail = m.spanAt(0, 0)
        self.assertEqual(detail["name"], "MPI_Bcast")
        self.assertEqual(detail["category"], "mpi")
        self.assertEqual(detail["durNs"], 500.0)
        self.assertEqual(detail["tags"]["rank"], "2")

    def test_span_at_out_of_range_returns_empty_dict(self):
        m = self._model(_mk_trace([_span(1, 1, Category.CPU, 0, 10, "x")]))
        self.assertEqual(m.spanAt(0, 99), {})
        self.assertEqual(m.spanAt(99, 0), {})

    def test_cross_lane_connector_reshaped_with_lane_and_span_indices(self):
        # Same matched send/recv pattern as tests/test_timeline_connectors.py
        # (the TUI's equivalent test) -- proves this model consumes the
        # exact same criticalpath.py dependency graph, just reshaped for
        # Canvas rendering (lane/span indices) instead of Braille coords.
        send = _span(1, 101, Category.MPI, 1000, 50, "MPI_Send",
                     tags={"type": "send", "rank": "0", "peer": "1", "tag": "7"})
        recv = _span(1, 201, Category.MPI, 1100, 80, "MPI_Recv",
                     tags={"type": "recv", "rank": "1", "peer": "0", "tag": "7"})
        m = self._model(_mk_trace([send, recv]))
        self.assertEqual(len(m.connectors), 1)
        c = m.connectors[0]
        self.assertIn("predLane", c)
        self.assertIn("succLane", c)
        self.assertNotEqual(c["predLane"], c["succLane"])
        self.assertTrue(c["color"].startswith("#"))

    def test_minimal_single_span_trace_does_not_crash(self):
        self._model(_mk_trace([_span(1, 1, Category.CPU, 0, 10, "main")]))

    def test_call_graph_scoped_to_visible_window(self):
        # Two spans with captured stacks, one inside the queried window,
        # one entirely before it -- only the visible one should
        # contribute nodes/edges.
        visible = SpanEvent(name="in_view", category=Category.CPU,
                            start_ns=1000, duration_ns=100, pid=1, tid=1,
                            stack_frames=["caller_a"])
        offscreen = SpanEvent(name="out_of_view", category=Category.CPU,
                              start_ns=0, duration_ns=10, pid=1, tid=1,
                              stack_frames=["caller_b"])
        m = self._model(_mk_trace([visible, offscreen]))
        result = m.callGraph(900.0, 1200.0)
        names = {n["name"] for n in result["nodes"]}
        self.assertIn("in_view", names)
        self.assertIn("caller_a", names)
        self.assertNotIn("out_of_view", names)
        self.assertNotIn("caller_b", names)

    def test_call_graph_nodes_have_normalized_positions_and_colors(self):
        span = SpanEvent(name="leaf", category=Category.CPU,
                         start_ns=0, duration_ns=100, pid=1, tid=1,
                         stack_frames=["root"])
        m = self._model(_mk_trace([span]))
        result = m.callGraph(0.0, 1000.0)
        self.assertEqual(len(result["nodes"]), 2)
        for n in result["nodes"]:
            self.assertGreaterEqual(n["x"], 0.0)
            self.assertLessEqual(n["x"], 1.0)
            self.assertTrue(n["color"].startswith("#"))

    def test_call_graph_empty_when_no_spans_have_stack_frames(self):
        m = self._model(_mk_trace([_span(1, 1, Category.CPU, 0, 100, "fn")]))
        result = m.callGraph(0.0, 1000.0)
        self.assertEqual(result["nodes"], [])
        self.assertEqual(result["edges"], [])

    # ── rows (table/timeline-upgrade round: the visual row list that
    #    replaces `lanes` in TimelineScreen.qml's ListView) ──────────────

    def test_rows_is_1_to_1_with_lanes_before_any_filter_or_grouping(self):
        spans = [_span(1, i, Category.CPU, 0, 100, f"fn{i}") for i in range(1, 5)]
        m = self._model(_mk_trace(spans))
        self.assertEqual(len(m.rows), len(m.lanes))
        for i, (row, lane) in enumerate(zip(m.rows, m.lanes)):
            self.assertEqual(row["kind"], "lane")
            self.assertEqual(row["laneIndex"], i)
            self.assertEqual(row["name"], lane["name"])
            self.assertEqual(row["label"], lane["label"])
            self.assertEqual(row["color"], lane["color"])
            self.assertEqual(row["count"], lane["count"])
            self.assertEqual(row["filteredCount"], lane["count"])

    def test_rows_laneIndex_addresses_the_same_lane_visibleSpans_uses(self):
        # The whole point of carrying laneIndex on each row: visibleSpans/
        # spanAt/findByName must keep working unchanged when called with
        # a row's own laneIndex, regardless of anything rows does.
        a = _span(1, 1, Category.MPI, 0, 500, "MPI_Bcast", tags={"rank": "2"})
        m = self._model(_mk_trace([a]))
        row = m.rows[0]
        detail = m.spanAt(row["laneIndex"], 0)
        self.assertEqual(detail["name"], "MPI_Bcast")

    def test_many_lanes_scale_without_error(self):
        # 60 lanes -- far more than a typical viewport shows at once
        # (the row-virtualization case a plain Repeater-per-lane would
        # have instantiated eagerly and unconditionally before this
        # round). Just confirms the model scales cleanly; the QML-side
        # ListView delegate recycling itself is verified via the real
        # shared-engine interaction test in test_gui_timeline_hover.py
        # (Repeater/ListView-created delegates aren't reliably reachable
        # via findChildren() under PySide6, a limitation documented
        # elsewhere in this test suite already -- contentHeight vs.
        # viewport height is what's actually asserted there instead of a
        # direct per-delegate count).
        spans = [_span(1, tid, Category.CPU, 0, 100, f"fn{tid}") for tid in range(1, 61)]
        m = self._model(_mk_trace(spans))
        self.assertEqual(len(m.rows), 60)
        self.assertEqual(len({r["laneIndex"] for r in m.rows}), 60)

    # ── Filtering (Phase B2) ─────────────────────────────────────────

    def _filter_trace(self):
        # Two MPI ranks (each its own lane), one CUDA stream lane, and a
        # plain CPU thread lane -- enough dimensions to exercise every
        # filterDimensions entry except "device" (never available).
        r0 = _span(10, 10, Category.MPI, 0, 100, "MPI_Send", tags={"rank": "0"})
        r1 = _span(11, 11, Category.MPI, 0, 100, "MPI_Recv", tags={"rank": "1"})
        cuda = _span(10, 0, Category.GPU_CUDA, 0, 200, "matmul_kernel",
                     tags={"stream": "7", "type": "kernel"})
        cpu = _span(10, 99, Category.CPU, 0, 50, "cpu_fn")
        return _mk_trace([r0, r1, cuda, cpu])

    def test_filter_dimensions_report_available_values_and_reasons(self):
        m = self._model(self._filter_trace())
        dims = {d["key"]: d for d in m.filterDimensions}
        self.assertTrue(dims["rank"]["available"])
        self.assertEqual({v["value"] for v in dims["rank"]["values"]}, {"0", "1"})
        self.assertTrue(dims["stream"]["available"])
        self.assertTrue(dims["process"]["available"])
        self.assertTrue(dims["thread"]["available"])
        self.assertTrue(dims["runtime"]["available"])
        # device: never available in this trace format -- must say why,
        # not just report an empty list silently.
        self.assertFalse(dims["device"]["available"])
        self.assertTrue(dims["device"]["reason"])
        self.assertEqual(dims["device"]["values"], [])

    def test_filter_dimensions_unavailable_when_trace_has_no_such_data(self):
        # A trace with no MPI/stream tags at all: rank/stream must both
        # honestly report unavailable-with-reason, not an empty control.
        m = self._model(_mk_trace([_span(1, 1, Category.CPU, 0, 10, "f")]))
        dims = {d["key"]: d for d in m.filterDimensions}
        self.assertFalse(dims["rank"]["available"])
        self.assertTrue(dims["rank"]["reason"])
        self.assertFalse(dims["stream"]["available"])
        self.assertTrue(dims["stream"]["reason"])

    def test_apply_filters_by_rank_keeps_only_matching_lane(self):
        m = self._model(self._filter_trace())
        m.applyFilters({"ranks": ["0"]})
        names = {r["name"] for r in m.rows}
        self.assertEqual(len(m.rows), 1)
        self.assertIn("mpi/thread-10", names)

    def test_apply_filters_by_process_keeps_only_matching_lanes(self):
        m = self._model(self._filter_trace())
        m.applyFilters({"processes": [11]})
        self.assertEqual(len(m.rows), 1)
        self.assertEqual(m.rows[0]["name"], "mpi/thread-11")

    def test_apply_filters_by_runtime_keeps_only_matching_lanes(self):
        m = self._model(self._filter_trace())
        m.applyFilters({"runtimes": ["cuda"]})
        self.assertEqual(len(m.rows), 1)
        self.assertTrue(m.rows[0]["name"].startswith("cuda/"))

    def test_apply_filters_by_stream_keeps_only_matching_lane(self):
        m = self._model(self._filter_trace())
        m.applyFilters({"streams": ["7"]})
        self.assertEqual(len(m.rows), 1)
        self.assertEqual(m.rows[0]["name"], "cuda/stream-7")

    def test_apply_filters_name_query_reduces_filtered_count_not_row_count(self):
        # An event-level filter narrows filteredCount (what's drawn) but
        # does NOT remove the lane's row entirely -- only "active only"
        # does that (tested separately below).
        m = self._model(self._filter_trace())
        total_rows_before = len(m.rows)
        m.applyFilters({"nameQuery": "matmul"})
        self.assertEqual(len(m.rows), total_rows_before)
        by_name = {r["name"]: r["filteredCount"] for r in m.rows}
        self.assertEqual(by_name["cuda/stream-7"], 1)
        self.assertEqual(by_name["mpi/thread-10"], 0)

    def test_apply_filters_active_only_hides_lanes_with_zero_matches(self):
        m = self._model(self._filter_trace())
        m.applyFilters({"nameQuery": "matmul", "activeOnly": True})
        self.assertEqual(len(m.rows), 1)
        self.assertEqual(m.rows[0]["name"], "cuda/stream-7")

    def test_apply_filters_min_duration_filters_events(self):
        m = self._model(self._filter_trace())
        m.applyFilters({"minDurationNs": 150})
        by_name = {r["name"]: r["filteredCount"] for r in m.rows}
        self.assertEqual(by_name["cuda/stream-7"], 1)   # dur 200, passes
        self.assertEqual(by_name["mpi/thread-10"], 0)    # dur 100, fails

    def test_apply_filters_bucket_filters_by_activity_bucket(self):
        m = self._model(self._filter_trace())
        m.applyFilters({"buckets": ["Computation"]})
        by_name = {r["name"]: r["filteredCount"] for r in m.rows}
        # cuda kernel (type=kernel) is Computation; MPI send/recv with no
        # type tag falls back to category "mpi" -> Communication.
        self.assertEqual(by_name["cuda/stream-7"], 1)
        self.assertEqual(by_name["mpi/thread-10"], 0)

    def test_apply_filters_regex_name_query(self):
        m = self._model(self._filter_trace())
        m.applyFilters({"nameQuery": r"^matmul_\w+$", "nameIsRegex": True})
        by_name = {r["name"]: r["filteredCount"] for r in m.rows}
        self.assertEqual(by_name["cuda/stream-7"], 1)

    def test_apply_filters_invalid_regex_matches_nothing_not_crash(self):
        m = self._model(self._filter_trace())
        m.applyFilters({"nameQuery": "(unclosed", "nameIsRegex": True})
        for r in m.rows:
            self.assertEqual(r["filteredCount"], 0)

    def test_apply_filters_time_range_only(self):
        spans = [
            _span(1, 1, Category.CPU, 0, 100, "early"),
            _span(1, 1, Category.CPU, 10_000, 100, "late"),
        ]
        m = self._model(_mk_trace(spans))
        m.applyFilters({"timeRangeOnly": True, "rangeStartNs": 9_000, "rangeEndNs": 11_000})
        self.assertEqual(m.rows[0]["filteredCount"], 1)

    def test_clear_filters_restores_full_rows_and_counts(self):
        m = self._model(self._filter_trace())
        full_count = len(m.rows)
        m.applyFilters({"ranks": ["0"], "nameQuery": "matmul"})
        self.assertLess(len(m.rows), full_count)
        m.clearFilters()
        self.assertEqual(len(m.rows), full_count)
        for r in m.rows:
            self.assertEqual(r["filteredCount"], r["count"])
        self.assertEqual(m.activeFilters, {})

    def test_visible_spans_respects_active_event_filter(self):
        # visibleSpans() (the Canvas paint path) must also honor the
        # active event-level filter, not just rows' filteredCount summary
        # -- otherwise the row would claim 0 matches while the lane
        # still visually paints every span.
        m = self._model(self._filter_trace())
        cuda_lane_idx = next(i for i, l in enumerate(m.lanes) if l["name"] == "cuda/stream-7")
        m.applyFilters({"nameQuery": "does_not_exist"})
        self.assertEqual(m.visibleSpans(cuda_lane_idx, 0, 1_000_000, 100), [])
        m.clearFilters()
        self.assertEqual(len(m.visibleSpans(cuda_lane_idx, 0, 1_000_000, 100)), 1)

    def test_hidden_row_count_and_filtered_span_count_track_filters(self):
        m = self._model(self._filter_trace())
        self.assertEqual(m.hiddenRowCount, 0)
        m.applyFilters({"ranks": ["0"]})
        self.assertEqual(m.hiddenRowCount, len(m.lanes) - 1)
        self.assertEqual(m.filteredSpanCount, 1)

    def test_apply_filters_never_renumbers_lanes(self):
        # Regression guard: filtering must never change laneIndex
        # addressing -- spanAt/findByName/visibleSpans all key off the
        # ORIGINAL lane index, and Round 16's cross-tab nav depends on
        # that staying stable regardless of filter state.
        m = self._model(self._filter_trace())
        before = {l["name"]: i for i, l in enumerate(m.lanes)}
        m.applyFilters({"ranks": ["0"]})
        for r in m.rows:
            self.assertEqual(before[r["name"]], r["laneIndex"])

    # ── Grouping / collapse (Phase B3) ──────────────────────────────

    def test_set_grouping_by_runtime_creates_group_rows(self):
        m = self._model(self._filter_trace())
        m.setGrouping("runtime")
        self.assertEqual(m.grouping, "runtime")
        kinds = [r["kind"] for r in m.rows]
        self.assertIn("group", kinds)
        group_rows = [r for r in m.rows if r["kind"] == "group"]
        labels = {r["label"] for r in group_rows}
        self.assertEqual(labels, {"Runtime mpi", "Runtime cuda", "Runtime cpu"})
        # Every group is followed immediately by its own member lane rows
        # (not collapsed by default).
        lane_rows = [r for r in m.rows if r["kind"] == "lane"]
        self.assertEqual(len(lane_rows), len(m.lanes))

    def test_set_grouping_none_restores_flat_rows(self):
        m = self._model(self._filter_trace())
        m.setGrouping("runtime")
        m.setGrouping("none")
        self.assertEqual(m.grouping, "none")
        self.assertTrue(all(r["kind"] == "lane" for r in m.rows))
        self.assertEqual(len(m.rows), len(m.lanes))

    def test_set_grouping_unknown_key_falls_back_to_none(self):
        m = self._model(self._filter_trace())
        m.setGrouping("bogus")
        self.assertEqual(m.grouping, "none")

    def test_group_by_rank_labels_unavailable_lanes_together(self):
        # cuda/stream-7 and cpu/thread-99 have no MPI rank at all --
        # honestly grouped together as "(unavailable)", not silently
        # dropped or fabricated a rank.
        m = self._model(self._filter_trace())
        m.setGrouping("rank")
        group_rows = {r["label"]: r for r in m.rows if r["kind"] == "group"}
        self.assertIn("(unavailable)", group_rows)
        self.assertEqual(group_rows["(unavailable)"]["laneCount"], 2)
        self.assertIn("Rank 0", group_rows)
        self.assertIn("Rank 1", group_rows)

    def test_group_collapse_hides_member_lane_rows_but_keeps_group_row(self):
        m = self._model(self._filter_trace())
        m.setGrouping("runtime")
        group_id = next(r["groupId"] for r in m.rows if r["kind"] == "group" and r["label"] == "Runtime cuda")
        m.setGroupCollapsed(group_id, True)
        rows = m.rows
        group_row = next(r for r in rows if r["kind"] == "group" and r["groupId"] == group_id)
        self.assertTrue(group_row["collapsed"])
        # No lane row from the collapsed group appears anywhere in rows.
        cuda_lane_indexes = set(group_row["laneIndexes"])
        for r in rows:
            if r["kind"] == "lane":
                self.assertNotIn(r["laneIndex"], cuda_lane_indexes)
        # Other groups' lane rows are unaffected.
        self.assertTrue(any(r["kind"] == "lane" for r in rows))

    def test_group_row_aggregates_count_across_members(self):
        m = self._model(self._filter_trace())
        m.setGrouping("rank")
        group_row = next(r for r in m.rows if r["kind"] == "group" and r["label"] == "(unavailable)")
        self.assertEqual(group_row["count"], 2)   # cuda + cpu lanes, 1 span each
        self.assertEqual(group_row["filteredCount"], 2)

    def test_collapse_all_and_expand_all_groups(self):
        m = self._model(self._filter_trace())
        m.setGrouping("runtime")
        m.collapseAllGroups()
        self.assertTrue(all(r["kind"] == "group" for r in m.rows))
        m.expandAllGroups()
        self.assertTrue(any(r["kind"] == "lane" for r in m.rows))

    def test_group_coverage_length_and_bounds(self):
        m = self._model(self._filter_trace())
        m.setGrouping("runtime")
        group_row = next(r for r in m.rows if r["kind"] == "group" and r["label"] == "Runtime cuda")
        cov = m.groupCoverage(group_row["laneIndexes"], 0, 1_000_000, 10)
        self.assertEqual(len(cov), 10)
        for frac in cov:
            self.assertIn(frac, (0.0, 1.0))
        self.assertTrue(any(f == 1.0 for f in cov))   # the cuda span (dur 200) covers bucket 0

    def test_group_coverage_empty_lane_list_returns_all_zero(self):
        m = self._model(self._filter_trace())
        cov = m.groupCoverage([], 0, 1_000_000, 5)
        self.assertEqual(cov, [0.0] * 5)

    def test_group_coverage_invalid_bucket_count_returns_empty(self):
        m = self._model(self._filter_trace())
        self.assertEqual(m.groupCoverage([0], 0, 1_000_000, 0), [])

    # ── Hide / isolate / reorder (Phase B3) ─────────────────────────

    def test_hide_lane_removes_it_from_rows(self):
        m = self._model(self._filter_trace())
        target = m.lanes[0]["name"]
        m.hideLane(target)
        self.assertNotIn(target, [r["name"] for r in m.rows])
        self.assertIn(target, m.hiddenLanes)

    def test_show_lane_restores_a_hidden_lane(self):
        m = self._model(self._filter_trace())
        target = m.lanes[0]["name"]
        m.hideLane(target)
        m.showLane(target)
        self.assertIn(target, [r["name"] for r in m.rows])
        self.assertNotIn(target, m.hiddenLanes)

    def test_isolate_lane_shows_only_that_one_lane(self):
        m = self._model(self._filter_trace())
        target = m.lanes[0]["name"]
        m.isolateLane(target)
        self.assertEqual([r["name"] for r in m.rows], [target])
        self.assertEqual(m.isolatedLanes, [target])

    def test_isolate_lane_replaces_not_accumulates(self):
        m = self._model(self._filter_trace())
        m.isolateLane(m.lanes[0]["name"])
        m.isolateLane(m.lanes[1]["name"])
        self.assertEqual([r["name"] for r in m.rows], [m.lanes[1]["name"]])

    def test_show_all_lanes_clears_hide_and_isolate(self):
        m = self._model(self._filter_trace())
        full_count = len(m.rows)
        m.hideLane(m.lanes[0]["name"])
        m.isolateLane(m.lanes[1]["name"])
        m.showAllLanes()
        self.assertEqual(len(m.rows), full_count)
        self.assertEqual(m.hiddenLanes, [])
        self.assertEqual(m.isolatedLanes, [])

    def test_move_row_changes_order(self):
        m = self._model(self._filter_trace())
        names = [l["name"] for l in m.lanes]
        last = names[-1]
        m.moveRow(last, 0)
        self.assertEqual(m.rows[0]["name"], last)

    def test_set_row_order_appends_lanes_it_omits(self):
        m = self._model(self._filter_trace())
        names = [l["name"] for l in m.lanes]
        m.setRowOrder([names[-1]])
        result_names = [r["name"] for r in m.rows]
        self.assertEqual(result_names[0], names[-1])
        self.assertEqual(set(result_names), set(names))

    def test_row_index_for_lane_tracks_hide_and_grouping(self):
        m = self._model(self._filter_trace())
        target_idx = next(i for i, l in enumerate(m.lanes) if l["name"] == m.lanes[0]["name"])
        self.assertEqual(m.rowIndexForLane(target_idx), 0)
        m.hideLane(m.lanes[0]["name"])
        self.assertEqual(m.rowIndexForLane(target_idx), -1)
        m.showAllLanes()

    def test_row_index_for_lane_under_grouping_points_at_lane_row_not_group_row(self):
        m = self._model(self._filter_trace())
        m.setGrouping("runtime")
        cuda_idx = next(i for i, l in enumerate(m.lanes) if l["name"] == "cuda/stream-7")
        pos = m.rowIndexForLane(cuda_idx)
        self.assertGreaterEqual(pos, 0)
        self.assertEqual(m.rows[pos]["kind"], "lane")
        self.assertEqual(m.rows[pos]["laneIndex"], cuda_idx)

    # ── Color mode (Phase B4) ────────────────────────────────────────

    def test_color_mode_defaults_to_function(self):
        m = self._model(self._filter_trace())
        self.assertEqual(m.colorMode, "function")

    def test_color_mode_bucket_groups_same_bucket_spans_under_one_color(self):
        # Two different function NAMES that land in the same activity
        # bucket (both Computation: a cuda kernel and a plain cpu span
        # with no type tag) must share a color in "bucket" mode, even
        # though "function" mode would give them different ones.
        a = _span(1, 1, Category.GPU_CUDA, 0, 100, "kernel_a", tags={"type": "kernel"})
        b = _span(1, 2, Category.CPU, 0, 100, "cpu_fn_b")
        m = self._model(_mk_trace([a, b]))
        lane_a = next(i for i, l in enumerate(m.lanes) if l["name"] == "cuda/thread-1")
        lane_b = next(i for i, l in enumerate(m.lanes) if l["name"] == "cpu/thread-2")
        m.setColorMode("bucket")
        self.assertEqual(m.colorMode, "bucket")
        color_a = m.visibleSpans(lane_a, 0, 1000, 10)[0]["color"]
        color_b = m.visibleSpans(lane_b, 0, 1000, 10)[0]["color"]
        self.assertEqual(color_a, color_b)

    def test_color_mode_category_colors_by_span_category(self):
        m = self._model(self._filter_trace())
        m.setColorMode("category")
        cuda_lane = next(i for i, l in enumerate(m.lanes) if l["name"] == "cuda/stream-7")
        color = m.visibleSpans(cuda_lane, 0, 1_000_000, 10)[0]["color"]
        self.assertTrue(color.startswith("#"))

    def test_color_mode_invalid_falls_back_to_function(self):
        m = self._model(self._filter_trace())
        m.setColorMode("bogus")
        self.assertEqual(m.colorMode, "function")

    # ── Search (Phase B4) ────────────────────────────────────────────

    def test_search_returns_match_count_and_sets_matched_flag(self):
        m = self._model(self._filter_trace())
        cuda_lane = next(i for i, l in enumerate(m.lanes) if l["name"] == "cuda/stream-7")
        count = m.search("matmul", False)
        self.assertEqual(count, 1)
        spans = m.visibleSpans(cuda_lane, 0, 1_000_000, 10)
        self.assertTrue(spans[0]["matched"])

    def test_search_matched_flag_absent_when_not_searching(self):
        m = self._model(self._filter_trace())
        cuda_lane = next(i for i, l in enumerate(m.lanes) if l["name"] == "cuda/stream-7")
        spans = m.visibleSpans(cuda_lane, 0, 1_000_000, 10)
        self.assertNotIn("matched", spans[0])

    def test_search_empty_query_behaves_like_clear(self):
        m = self._model(self._filter_trace())
        m.search("matmul", False)
        count = m.search("", False)
        self.assertEqual(count, 0)
        self.assertEqual(m.searchMatchCount, 0)
        cuda_lane = next(i for i, l in enumerate(m.lanes) if l["name"] == "cuda/stream-7")
        self.assertNotIn("matched", m.visibleSpans(cuda_lane, 0, 1_000_000, 10)[0])

    def test_search_invalid_regex_returns_zero_not_crash(self):
        m = self._model(self._filter_trace())
        count = m.search("(unclosed", True)
        self.assertEqual(count, 0)

    def test_search_regex_matches(self):
        m = self._model(self._filter_trace())
        count = m.search(r"^matmul_\w+$", True)
        self.assertEqual(count, 1)

    def test_next_match_and_previous_match_cycle_with_wraparound(self):
        # Two mpi lanes named identically so search finds 2 occurrences
        # to cycle through.
        a = _span(1, 1, Category.MPI, 0, 100, "MPI_Bcast")
        b = _span(1, 2, Category.MPI, 500, 100, "MPI_Bcast")
        m = self._model(_mk_trace([a, b]))
        count = m.search("MPI_Bcast", False)
        self.assertEqual(count, 2)
        first = m.nextMatch()
        second = m.nextMatch()
        third = m.nextMatch()   # wraps back to the first
        self.assertEqual(third["laneIndex"], first["laneIndex"])
        self.assertEqual(third["spanIdx"], first["spanIdx"])
        self.assertNotEqual((first["laneIndex"], first["spanIdx"]),
                             (second["laneIndex"], second["spanIdx"]))
        # previousMatch from the wrapped-to-first position goes back to
        # the second (wrapping the other way).
        prev = m.previousMatch()
        self.assertEqual((prev["laneIndex"], prev["spanIdx"]),
                          (second["laneIndex"], second["spanIdx"]))

    def test_next_match_empty_when_no_matches(self):
        m = self._model(self._filter_trace())
        m.search("does_not_exist", False)
        self.assertEqual(m.nextMatch(), {})
        self.assertEqual(m.previousMatch(), {})

    def test_clear_search_removes_matched_flag(self):
        m = self._model(self._filter_trace())
        cuda_lane = next(i for i, l in enumerate(m.lanes) if l["name"] == "cuda/stream-7")
        m.search("matmul", False)
        m.clearSearch()
        self.assertEqual(m.searchMatchCount, 0)
        self.assertEqual(m.searchCursor, -1)
        self.assertNotIn("matched", m.visibleSpans(cuda_lane, 0, 1_000_000, 10)[0])

    def test_find_by_name_uses_index_and_matches_old_behavior(self):
        m = self._model(self._filter_trace())
        results = m.findByName("cuda", "matmul_kernel", 50)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["startNs"], 0.0)

    def test_find_by_name_no_match_returns_empty(self):
        m = self._model(self._filter_trace())
        self.assertEqual(m.findByName("cuda", "nonexistent", 50), [])

    def test_find_by_name_respects_max_results(self):
        spans = [_span(1, 1, Category.CPU, i * 1000, 100, "hot_fn") for i in range(10)]
        m = self._model(_mk_trace(spans))
        self.assertEqual(len(m.findByName("cpu", "hot_fn", 3)), 3)

    # ── Time ruler / bookmarks / named ranges (Phase B5) ────────────

    def test_time_ticks_returns_nice_round_numbers(self):
        m = self._model(self._filter_trace())
        ticks = m.timeTicks(0, 1_000_000_000, 8)   # 0..1s, ~8 ticks
        self.assertGreater(len(ticks), 0)
        for t in ticks:
            self.assertIn("ns", t)
            self.assertIn("label", t)
            self.assertGreaterEqual(t["ns"], 0)
            self.assertLessEqual(t["ns"], 1_000_000_000)

    def test_time_ticks_monotonic_and_evenly_spaced(self):
        m = self._model(self._filter_trace())
        ticks = m.timeTicks(0, 1_000_000_000, 10)
        values = [t["ns"] for t in ticks]
        self.assertEqual(values, sorted(values))
        if len(values) > 2:
            gaps = {round(b - a, 3) for a, b in zip(values, values[1:])}
            self.assertEqual(len(gaps), 1)   # all gaps identical -- evenly spaced

    def test_time_ticks_empty_for_degenerate_range(self):
        m = self._model(self._filter_trace())
        self.assertEqual(m.timeTicks(1000, 1000, 8), [])
        self.assertEqual(m.timeTicks(0, 1000, 0), [])

    def test_add_bookmark_returns_id_and_appears_sorted(self):
        m = self._model(self._filter_trace())
        id2 = m.addBookmark(500, "second")
        id1 = m.addBookmark(100, "first")
        self.assertNotEqual(id1, id2)
        self.assertEqual([b["name"] for b in m.bookmarks], ["first", "second"])

    def test_add_bookmark_default_name_uses_formatted_offset(self):
        m = self._model(self._filter_trace())
        m.addBookmark(100, "")
        self.assertTrue(m.bookmarks[0]["name"])   # non-empty, auto-generated

    def test_remove_bookmark_by_id(self):
        m = self._model(self._filter_trace())
        bid = m.addBookmark(100, "x")
        m.addBookmark(200, "y")
        m.removeBookmark(bid)
        self.assertEqual([b["name"] for b in m.bookmarks], ["y"])

    def test_add_named_range_normalizes_reversed_bounds(self):
        m = self._model(self._filter_trace())
        m.addNamedRange(500, 100, "r")
        self.assertEqual(m.namedRanges[0]["startNs"], 100.0)
        self.assertEqual(m.namedRanges[0]["endNs"], 500.0)

    def test_remove_named_range_by_id(self):
        m = self._model(self._filter_trace())
        rid = m.addNamedRange(0, 100, "a")
        m.addNamedRange(200, 300, "b")
        m.removeNamedRange(rid)
        self.assertEqual([r["name"] for r in m.namedRanges], ["b"])


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(_PYSIDE6_AVAILABLE, "PySide6 not installed")
class TestTimelineOperationsNeverMutateMeasurements(unittest.TestCase):
    """Zoom/filter/group/hide/isolate/reorder/search/colour/bookmark are
    view operations: the Trace's spans and every summary computed from
    them must be byte-identical before and after."""

    @classmethod
    def setUpClass(cls):
        cls._app = QGuiApplication.instance() or QGuiApplication([])

    def test_view_operations_leave_trace_and_summaries_untouched(self):
        import copy
        from src.core.trace import Trace, TraceMetadata
        from src.core.events import SpanEvent, Category
        from src.gui.theme import Theme
        from src.gui.models import TimelineModel
        from src.analysis import activity_buckets as ab
        t = Trace(TraceMetadata(command="a.out"))
        for i in range(200):
            cat = [Category.CPU, Category.MPI, Category.GPU_CUDA, Category.SYNC][i % 4]
            tags = {"type": "kernel", "stream": str(i % 2)} if cat is Category.GPU_CUDA else {"rank": str(i % 2)}
            t.add(SpanEvent(name=f"f{i % 7}", category=cat, start_ns=i * 1000, duration_ns=500 + i,
                            pid=1 + i % 2, tid=10 + i % 3, tags=tags))
        snapshot = lambda: [(s.name, s.category, s.start_ns, s.duration_ns, s.pid, s.tid,
                             dict(s.tags), s.span_id, s.parent_span_id) for s in t.spans]
        before = copy.deepcopy(snapshot())
        stats_before = t.aggregated_stats()
        buckets_before = ab.bucket_totals(t.spans)

        m = TimelineModel(t, Theme(dark=True))
        m.applyFilters({"runtimes": ["cpu", "mpi"], "minDurationNs": 550, "nameQuery": "f[0-3]",
                        "nameIsRegex": True, "activeOnly": True})
        for g in ("rank", "process", "runtime", "stream", "thread"):
            m.setGrouping(g)
            m.collapseAllGroups()
            m.expandAllGroups()
        m.setGrouping("none")
        lanes = [lane["name"] for lane in m.lanes]
        m.hideLane(lanes[0])
        m.isolateLane(lanes[-1])
        m.showAllLanes()
        m.moveRow(lanes[0], len(lanes) - 1)
        for mode in ("bucket", "category", "function"):
            m.setColorMode(mode)
        m.search("f2", False)
        m.nextMatch()
        m.previousMatch()
        m.addBookmark(5_000.0, "b")
        m.addNamedRange(1_000.0, 9_000.0, "r")
        for i in range(len(lanes)):
            m.visibleSpans(i, 0.0, 200_000.0, 800)
        m.clearFilters()
        m.clearSearch()

        self.assertEqual(snapshot(), before)
        self.assertEqual(t.aggregated_stats(), stats_before)
        self.assertEqual(ab.bucket_totals(t.spans), buckets_before)
