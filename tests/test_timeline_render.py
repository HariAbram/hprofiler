"""
The Timeline's rendering is temporally honest at every zoom (src/gui/
timeline_lod.py decides, timeline_paint.py draws), and its overlays,
summaries, navigation, export and comparison matching show the values the
analyses computed.

Regression: zooming made a thread look active where it had no event.
Two paths stretched short events into continuous bars -- spans narrower
than a pixel were drawn one pixel wide at full height, and occupancy bins
were painted with an opacity floor of 25% for any non-zero value -- and
both switched at a fixed span count, so a lane's apparent activity changed
with the zoom. Now (and checked here, at many zooms):

  * events appear only during their actual intervals; gaps stay empty;
  * the painted share of every column equals its true busy share (exact
    renders) -- zooming never changes apparent occupancy;
  * sub-pixel events become a bar whose height is the busy fraction plus a
    presence tick, never a full-height bar;
  * clipping, large absolute timestamps and nested / overlapping spans
    keep the geometry exact; cached renders are keyed by the viewport.

Pixel tests paint into a QImage directly (no QML engine).
"""
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
try:
    from PySide6.QtGui import QGuiApplication, QImage, QPainter, QColor
    _QT = True
except ImportError:          # pragma: no cover
    _QT = False

from src.core.events import Category, SpanEvent
from src.core.trace import Trace, TraceMetadata
from src.gui import timeline_lod as lod
from src.gui import timeline_hierarchy as th


def S(start, dur, name="work", cat=Category.OPENMP, pid=1, tid=1, eid=0, **tags):
    s = SpanEvent(name, cat, int(start), int(dur), pid, tid, tags={"type": "work", **tags})
    s.eid = eid
    return s


def true_occupancy(intervals, a, b, width):
    """Busy share per column of the union of [s, e) -- an independent,
    brute-force reference (boundaries of every interval and column)."""
    out = np.zeros(width)
    ivs = sorted((max(s, a), min(e, b)) for s, e in intervals if e > a and s < b)
    merged = []
    for s, e in ivs:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    col = (b - a) / width
    for s, e in merged:
        c0, c1 = int((s - a) // col), int(min(width - 1, (e - a) // col))
        for c in range(c0, c1 + 1):
            lo, hi = a + c * col, a + (c + 1) * col
            out[c] += max(0.0, min(e, hi) - max(s, lo)) / col
    return np.clip(out, 0, 1)


def render(spans, a, b, width, **kw):
    return lod.render_spans(sorted(spans, key=lambda s: s.start_ns), a, b, width,
                            color_of=lambda s: "#22d3ee", **kw)


class TestLevelOfDetail(unittest.TestCase):
    def test_exact_rectangles_are_the_event_intervals(self):
        spans = [S(100, 300, eid=1), S(500, 50, eid=2)]
        lr = render(spans, 0, 1000, 1000)
        self.assertEqual([(r.x0, r.x1, r.eid) for r in lr.rects], [(100.0, 400.0, 1), (500.0, 550.0, 2)])
        self.assertEqual(lr.busy.sum(), 0)

    def test_a_span_starting_left_of_the_view_is_clipped_not_shifted(self):
        lr = render([S(-500, 700, eid=7)], 0, 1000, 100)
        self.assertEqual((lr.rects[0].x0, lr.rects[0].x1), (0.0, 20.0))

    def test_sub_pixel_events_are_a_busy_fraction_never_a_full_bar(self):
        # 1000 events of 10 ns, one every 10 us: 0.1% busy, one per column
        spans = [S(i * 10_000, 10, eid=i) for i in range(1000)]
        lr = render(spans, 0, 10_000_000, 1000)
        self.assertEqual(lr.rects, [])
        np.testing.assert_allclose(lr.busy, 0.001, atol=1e-12)
        self.assertTrue((lr.count == 1).all())
        np.testing.assert_allclose(lod.apparent_occupancy(lr), 0.001, atol=1e-12)

    def test_apparent_occupancy_equals_true_occupancy_at_every_zoom(self):
        rng = random.Random(4)
        spans, ivs, t = [], [], 0
        for i in range(3000):
            t += rng.choice([5, 50, 500, 5000, 50_000])
            d = rng.choice([3, 40, 400, 4000, 200_000])
            spans.append(S(t, d, eid=i))
            ivs.append((t, t + d))
            t += rng.choice([0, d // 2, d, 2 * d])
        t_end = t
        for zoom in (1, 2, 5, 10, 50, 200, 1000, 10_000):
            vis = t_end / zoom
            for frac in (0.0, 0.37, 0.81):
                a = frac * (t_end - vis)
                b = a + vis
                for width in (800, 1333):
                    with self.subTest(zoom=zoom, frac=frac, width=width):
                        lr = render(spans, a, b, width, limit=10 ** 9)
                        got = lod.apparent_occupancy(lr)
                        want = true_occupancy(ivs, a, b, width)
                        np.testing.assert_allclose(got, want, atol=1e-9)

    def test_gaps_stay_empty_at_every_zoom(self):
        # two bursts of short events with a 1 ms gap between them
        spans = [S(i * 100, 20, eid=i) for i in range(5000)] + \
                [S(1_500_000 + i * 100, 20, eid=10_000 + i) for i in range(5000)]
        gap = (500_000, 1_500_000)
        for zoom in (1, 3, 10, 40):
            a, b = 0, 2_000_000 / zoom * 1.0 + (gap[0] if zoom > 1 else 0)
            b = max(b, a + 1)
            lr = render(spans, a, b, 900, limit=10 ** 9)
            k = 900 / (b - a)
            for c in range(900):
                lo, hi = a + c / k, a + (c + 1) / k
                if lo >= gap[0] and hi <= gap[1]:
                    self.assertEqual((lr.busy[c], lr.wait[c], lr.count[c]), (0, 0, 0), (zoom, c))
            for r in lr.rects:
                self.assertFalse(r.x0 >= (gap[0] - a) * k + 1 and r.x1 <= (gap[1] - a) * k - 1)

    def test_integrated_busy_time_of_an_interval_does_not_change_with_zoom(self):
        rng = random.Random(9)
        spans = []
        t = 0
        for i in range(4000):
            t += rng.randrange(1, 3000)
            spans.append(S(t, rng.randrange(1, 2000), eid=i))
        a0, b0 = 1_000_000, 3_000_000
        totals = []
        for width in (40, 200, 1000, 5000):
            lr = render(spans, a0, b0, width, limit=10 ** 9)
            totals.append(lod.apparent_occupancy(lr).sum() * (b0 - a0) / width)
        for v in totals[1:]:
            self.assertAlmostEqual(v, totals[0], delta=1e-6 * (b0 - a0))

    def test_large_absolute_timestamps_keep_nanosecond_geometry(self):
        base = 1_700_000_000_000_000        # ~20 days of CLOCK_MONOTONIC
        lr = render([S(base + 1234, 3, eid=1)], base + 1230, base + 1240, 1000)
        self.assertEqual(len(lr.rects), 1)
        self.assertAlmostEqual(lr.rects[0].x0, 400.0, places=6)
        self.assertAlmostEqual(lr.rects[0].x1, 700.0, places=6)

    def test_nested_and_overlapping_spans_never_exceed_full(self):
        spans = [S(0, 1000, eid=1)] + [S(i * 7, 3, eid=10 + i) for i in range(140)] + [S(500, 900, eid=2)]
        lr = render(spans, 0, 2000, 50)
        self.assertTrue((lod.apparent_occupancy(lr) <= 1.0 + 1e-12).all())

    def test_waits_are_cut_out_only_inside_the_lanes_own_spans(self):
        region = S(0, 1_000_000, "omp_parallel_region", eid=1)
        task = S(700_000, 100_000, "omp_task", eid=2)
        barrier = S(500_000, 500_000, "omp_barrier", cat=Category.SYNC, eid=3, type="barrier")
        later = S(2_000_000, 100_000, "omp_barrier", cat=Category.SYNC, eid=4, type="barrier")
        lr = render([region, task], 0, 2_500_000, 1000, waits=[barrier, later], wait_color="#e5e7eb")
        kinds = sorted((round(r.x0), round(r.x1), lod.ACTIVITY[r.activity]) for r in lr.rects)
        # region busy until the barrier, waiting after it, the task on top;
        # the later barrier is outside every region span: not drawn here
        self.assertEqual(kinds, [(0, 200, "work"), (200, 400, "wait"), (280, 320, "work")])

    def test_index_renders_have_no_opacity_floor_and_say_when_approximate(self):
        occ = np.full(100, 0.001)
        lr = lod.render_occupancy(occ, np.ones(100, dtype=np.int64), 0, 1e9, 100, "#22d3ee",
                                  index_bin_ns=1e9 / 65536)
        np.testing.assert_allclose(lod.apparent_occupancy(lr), 0.001)
        self.assertFalse(lr.approx)          # 10 ms columns over 15 us index bins
        lr = lod.render_occupancy(occ, None, 0, 1e6, 100, "#22d3ee", index_bin_ns=2000.0)
        self.assertTrue(lr.approx)           # 10 us columns over 2 us bins

    def test_hit_testing_exact_and_aggregated(self):
        spans = [S(100, 300, eid=1)] + [S(600 + i, 1, eid=100 + i) for i in range(5)]
        lr = render(spans, 0, 1000, 100)
        self.assertEqual(lr.hit(20).eid, 1)
        self.assertIsNone(lr.hit(80))
        info = lr.column_info(60)
        self.assertEqual(info["count"], 5)
        self.assertAlmostEqual(info["busy"], 0.5)
        self.assertIsNone(lr.column_info(90))


class TestHierarchy(unittest.TestCase):
    def metas(self):
        L = th.LaneMeta
        return [L(0, "mpi/thread-10", "mpi", "thread", "10", 1, 5, tid=10, rank="0", thread_label="T1 · tid 10"),
                L(1, "openmp/thread-10", "openmp", "thread", "10", 1, 9, tid=10, rank="0", thread_label="T1 · tid 10"),
                L(2, "openmp/thread-11", "openmp", "thread", "11", 1, 9, tid=11, rank="0", thread_label="T2 · tid 11"),
                L(3, "cuda/stream-7", "cuda", "stream", "7", 1, 3, rank="0", device="0"),
                L(4, "mpi/thread-20", "mpi", "thread", "20", 2, 4, tid=20, rank="1", thread_label="T3 · tid 20")]

    def lane_row(self, i):
        return {"kind": "lane", "laneIndex": i, "name": f"l{i}", "label": f"l{i}", "count": 1, "filteredCount": 1}

    def test_every_lane_once_under_process_thread_device_stream(self):
        rows = th.flatten(th.build_tree(self.metas()), {}, lane_row=self.lane_row)
        self.assertEqual(sorted(r["laneIndex"] for r in rows if r["kind"] == "lane"), [0, 1, 2, 3, 4])
        labels = [(r["level"], r.get("hierLabel") or r["label"]) for r in rows]
        self.assertEqual(labels, [
            (0, "System"), (1, "rank 0 · pid 1"), (2, "T1 · tid 10"), (3, "mpi"), (3, "openmp"),
            (2, "T2 · tid 11 · openmp"), (2, "device 0"), (3, "stream 7 · cuda"),
            (1, "rank 1 · pid 2"), (2, "T3 · tid 20 · mpi")])

    def test_collapsed_header_hides_members_and_carries_them(self):
        tree = th.build_tree(self.metas())
        rows = th.flatten(tree, {"p1": True}, lane_row=self.lane_row)
        p1 = next(r for r in rows if r.get("groupId") == "p1")
        self.assertTrue(p1["collapsed"])
        self.assertEqual(sorted(p1["laneIndexes"]), [0, 1, 2, 3])
        self.assertNotIn(0, [r.get("laneIndex") for r in rows])

    def test_filtered_lanes_drop_empty_headers(self):
        rows = th.flatten(th.build_tree(self.metas()), {}, lane_row=lambda i: self.lane_row(i) if i == 4 else None)
        self.assertEqual([r.get("groupId", r.get("laneIndex")) for r in rows], ["system", "p2", 4])

    def test_nodes_only_when_the_trace_has_several(self):
        ms = self.metas()
        for m in ms:
            m.node = "0" if m.pid == 1 else "1"
        rows = th.flatten(th.build_tree(ms), {}, lane_row=self.lane_row)
        self.assertEqual([r["label"] for r in rows if r.get("nodeKind") == "node"], ["node 0", "node 1"])


@unittest.skipUnless(_QT, "PySide6 not installed")
class TestPainting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QGuiApplication.instance() or QGuiApplication([])

    def paint(self, lr, w=400, h=26):
        from src.gui.timeline_paint import Style, paint_lane
        img = QImage(w, h, QImage.Format.Format_ARGB32)
        img.fill(QColor(0, 0, 0, 0))
        p = QPainter(img)
        paint_lane(p, lr, w, h, Style.for_theme(True))
        p.end()
        return img

    def painted(self, img, x0=0, x1=None):
        x1 = img.width() if x1 is None else x1
        return sum(1 for x in range(x0, x1) for y in range(img.height()) if img.pixelColor(x, y).alpha() > 0)

    def test_sparse_sub_pixel_events_paint_a_sliver_not_a_lane(self):
        spans = [S(i * 10_000, 10, eid=i) for i in range(400)]
        lr = render(spans, 0, 4_000_000, 400)
        lr.observed = None
        img = self.paint(lr)
        total = 400 * 26
        # one pixel of bar (the minimum visible) + the 3-pixel presence tick
        # per column: under 20% of the lane, where the old renderer filled
        # every column at full height (~77% of it, inside the margins)
        self.assertLess(self.painted(img) / total, 0.2)
        self.assertGreater(self.painted(img), 0)

    def test_an_exact_event_paints_exactly_its_columns(self):
        lr = render([S(100, 100, eid=1)], 0, 400, 400)
        lr.observed = None
        img = self.paint(lr)
        cols = [x for x in range(400) if img.pixelColor(x, 13).alpha() > 0]
        self.assertEqual((cols[0], cols[-1], len(cols)), (100, 199, 100))

    def test_a_wait_reads_neutral_and_work_in_its_colour(self):
        region = S(0, 400, "omp_parallel_region", eid=1)
        barrier = S(200, 200, "omp_barrier", cat=Category.SYNC, eid=2, type="barrier")
        lr = render([region], 0, 400, 400, waits=[barrier], wait_color="#e5e7eb")
        lr.observed = None
        img = self.paint(lr)
        busy = img.pixelColor(100, 13)
        self.assertEqual((busy.red(), busy.green(), busy.blue()), (0x22, 0xd3, 0xee))
        for x in range(205, 395, 7):
            c = img.pixelColor(x, 12)
            self.assertLessEqual(max(c.red(), c.green(), c.blue()) - min(c.red(), c.green(), c.blue()), 30)

    def test_no_data_is_cross_hatched_outside_the_lanes_events(self):
        lr = render([S(200, 50, eid=1)], 0, 400, 400, extent=(200, 250))
        img = self.paint(lr)
        self.assertGreater(self.painted(img, 0, 190), 0)          # hatch before the first event
        self.assertEqual(self.painted(img, 260, 390) > 0, True)   # and after the last
        lr = render([S(200, 50, eid=1)], 0, 400, 400, extent=(0, 400))
        img = self.paint(lr)
        self.assertEqual(self.painted(img, 260, 390), 0)          # an observed gap stays empty

    def test_evidence_tiers_and_classes_have_their_own_line_styles(self):
        from PySide6.QtCore import Qt
        from src.gui.timeline_paint import tier_pen
        widths = [tier_pen("#ffffff", t, "direct").widthF() for t in ("certain", "high", "medium", "low")]
        self.assertEqual(widths, sorted(widths, reverse=True))
        self.assertEqual(tier_pen("#ffffff", "certain", "direct").style(), Qt.PenStyle.SolidLine)
        self.assertEqual(tier_pen("#ffffff", "medium", "inferred").style(), Qt.PenStyle.DashLine)
        self.assertEqual(tier_pen("#ffffff", "low", "inferred").style(), Qt.PenStyle.DotLine)
        alphas = [tier_pen("#ffffff", t, "direct").color().alphaF() for t in ("certain", "high", "medium", "low")]
        self.assertEqual(alphas, sorted(alphas, reverse=True))


def comb_trace():
    from tests.test_criticalpath_comb import comb_cycle
    return comb_cycle(cycles=2)


@unittest.skipUnless(_QT, "PySide6 not installed")
class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QGuiApplication.instance() or QGuiApplication([])
        from src.gui.models import TimelineModel
        from src.gui.theme import Theme
        cls.trace = comb_trace()
        cls.m = TimelineModel(cls.trace, Theme(dark=True))
        cls.m.startAnalysis()
        assert cls.m.wait_for_analysis(120), cls.m.analysisMessage
        from src.analysis import criticalpath as cp
        cls.report = cp.analyze(cls.trace)

    def lane(self, cat, pid=1):
        return next(i for i, ex in enumerate(self.m._lane_extra) if ex["category"] == cat and ex["pid"] == pid
                    and ex["kind"] == "thread")

    def full(self):
        return self.m.viewStartNs, self.m.viewStartNs + self.m.traceDurationNs

    def test_renders_are_keyed_by_viewport_not_reused_after_zoom(self):
        a, b = self.full()
        li = self.lane("mpi")
        r1 = self.m.render_lane(li, a, b, 500)
        r2 = self.m.render_lane(li, a, a + (b - a) / 7, 500)
        r3 = self.m.render_lane(li, a, b, 500)
        self.assertIs(r1, r3)
        self.assertIsNot(r1, r2)
        self.assertEqual((r2.view_start, r2.view_end), (a, a + (b - a) / 7))

    def test_timeline_and_cli_values_agree(self):
        sm = self.m.criticalSummary
        r = self.report
        self.assertEqual((sm["wallNs"], sm["observedNs"], sm["modeledNs"], sm["pathSpans"]),
                         (r.wall_ns, r.total_path_ns, r.modeled_path_ns, len(r.path_span_indices)))
        self.assertEqual(sm["tiers"], {t: int(r.path_time_by_tier.get(t, 0)) for t in ("certain", "high", "medium", "low")})
        # the span-level DP value is the legacy attribute_blame one
        from src.analysis import criticalpath as cp
        g = cp.load_or_build_graph(self.trace)
        path, confs, kinds = g.critical_path()
        legacy = cp.attribute_blame(g.spans, path, r.wall_ns, confs, kinds)
        self.assertEqual(sm["legacyValueNs"], legacy.dp_accounted_ns)
        self.assertGreater(sm["legacyValueNs"], sm["legacyUnionNs"])        # it counts some time twice

    def test_critical_path_highlight_selects_the_paths_spans(self):
        an = self.m._analysis
        want = {self.report.spans[i].eid for i in self.report.path_span_indices}
        self.assertEqual(an.path_eids, want)
        owned = {self.report.spans[j].eid for j, ns in zip(self.report.path_span_indices, self.report.path_credited_ns) if ns}
        self.assertEqual(set(an.credit_of_eid), owned)
        # every busy/wait band piece lies inside its owner's interval
        for st in an.steps:
            if st.mode in ("busy", "wait"):
                s = self.trace.store.event_by_id(st.eid)
                self.assertTrue(s.start_ns <= st.t0 <= st.t1 <= s.end_ns + 1)

    def test_dependency_arrows_connect_the_right_events(self):
        an = self.m._analysis
        p2p = [e for e in an.edges if e.kind == "p2p"]
        self.assertTrue(p2p)
        for e in p2p:
            send = self.trace.store.event_by_id(e.src_eid)
            recv = self.trace.store.event_by_id(e.dst_eid)
            self.assertEqual((send.name, e.t_src, e.src_end), ("MPI_Isend", send.start_ns, "start"))
            self.assertEqual((recv.name, e.t_dst, e.dst_end), ("MPI_Waitall", recv.end_ns, "end"))
        # drawn where those events are
        a, b = self.full()
        self.m.setOverlay({"families": ["mpi_p2p"]})
        img = QImage(800, 600, QImage.Format.Format_ARGB32)
        p = QPainter(img)
        self.m.paint_overlay_item(p, 800, 600, a, b, 0.0, 26.0, -1, -1.0)
        p.end()
        k = 800 / (b - a)
        drawn = {g.id: g for g in self.m._overlay_geometry}
        for e in p2p:
            g = drawn[e.id]
            self.assertAlmostEqual(g.x0, (e.t_src - a) * k)
            self.assertAlmostEqual(g.x1, (e.t_dst - a) * k)
            self.assertAlmostEqual(g.y0, self.m._anchor_pos[e.src_lane] * 26 + 13)
            self.assertAlmostEqual(g.y1, self.m._anchor_pos[e.dst_lane] * 26 + 13)
            self.assertEqual(g.cls, e.cls)
        self.assertEqual(self.m.edgeAt(drawn[p2p[0].id].x0, drawn[p2p[0].id].y0), p2p[0].id)
        self.m.setOverlay({"families": []})

    def test_summary_selection_highlights_the_right_intervals(self):
        a, b = self.full()
        self.m.setHighlight({"activities": ["wait"], "categories": ["mpi"], "label": "mpi waiting"})
        mpi = self.lane("mpi")
        lr = self.m.render_lane(mpi, a, b, 3000)
        opts = self.m._lane_options(mpi, lr)
        for r in lr.rects:
            want_bright = r.activity in lod.WAIT_LIKE and r.eid >= 0
            self.assertEqual(not opts.dim(r), want_bright, r.name)
        sync = self.lane("sync")
        lr = self.m.render_lane(sync, a, b, 3000)
        self.assertTrue(all(self.m._lane_options(sync, lr).dim(r) for r in lr.rects))
        self.m.clearHighlight()
        self.assertIsNone(self.m._lane_options(mpi, lr).dim)

    def test_details_of_an_event_carry_its_evidence(self):
        an = self.m._analysis
        e = next(e for e in an.edges if e.kind == "p2p")
        d = self.m.eventDetails(e.dst_lane, float(e.dst_eid))
        titles = [s["title"] for s in d["sections"]]
        self.assertEqual(titles[:5], ["Event", "Time", "Where", "Source, stack, kernel", "Critical path"])
        dep = next(x for x in d["dependencies"] if x["edge"] == e.id)
        self.assertEqual((dep["direction"], dep["kind"], dep["provenance"], dep["tier"]),
                         ("in", "p2p", e.provenance, e.tier))
        self.assertEqual(dep["partner"], "MPI_Isend")
        ed = self.m.edgeDetails(e.id)
        self.assertIn("not probabilities", ed["note"])
        self.assertEqual(ed["source"]["spanIdx"], float(e.src_eid))

    def test_header_rows_follow_their_lanes(self):
        a, b = self.full()
        px = 400
        t_node = next(n for n in self.m._tree_nodes.values() if n.kind == "thread")
        lanes = t_node.lanes()
        busy, wait, _c, _cp = self.m.render_group(t_node.id, lanes, a, b, px)
        work = [lod.column_fractions(self.m.render_lane(li, a, b, px)) for li in lanes
                if self.m._lane_extra[li]["category"] not in self.m._NOT_WORK_CATEGORIES]
        want_wait = np.max([w for _b, w in work], axis=0)
        want_busy = np.clip(np.max([bb for bb, _w in work], axis=0) - want_wait, 0, 1)
        np.testing.assert_allclose(busy, want_busy)
        np.testing.assert_allclose(wait, np.minimum(want_wait, 1 - want_busy))

    def test_path_switch_navigation(self):
        a, _b = self.full()
        t = self.m.nextTransition(a, 1)
        self.assertEqual(t["index"], 0)
        t2 = self.m.nextTransition(t["ns"], 1)
        self.assertGreaterEqual(t2["ns"], t["ns"])
        back = self.m.nextTransition(t2["ns"], -1)
        self.assertLess(back["index"], t2["index"])

    def test_history_and_jumps(self):
        m = self.m
        m.pushView(0, 100)
        m.pushView(10, 20)
        m.pushView(10, 20)                      # collapsed
        m.pushView(30, 40)
        self.assertEqual(m.historyBack(), {"startNs": 10.0, "endNs": 20.0})
        self.assertTrue(m.canGoForward)
        self.assertEqual(m.historyForward(), {"startNs": 30.0, "endNs": 40.0})
        a, _b = self.full()
        self.assertEqual(m.resolveJump("2ms"), {"kind": "time", "ns": a + 2e6})
        self.assertEqual(m.resolveJump("1ms-3ms")["kind"], "range")
        self.assertEqual(m.resolveJump("rank 1")["kind"], "lane")
        ev = m.resolveJump("_Z6unpackv")
        self.assertEqual((ev["kind"], ev["name"]), ("event", "_Z6unpackv"))
        self.assertEqual(m.resolveJump("no-such-thing")["kind"], "none")

    def test_figure_export_png_and_svg(self):
        a, b = self.full()
        with tempfile.TemporaryDirectory() as d:
            for ext in ("png", "svg"):
                path = os.path.join(d, "f." + ext)
                self.assertEqual(self.m.exportFigure(path, a, b, 1200), "")
                self.assertGreater(os.path.getsize(path), 1000)
            self.assertIn("Explicit waiting", Path(os.path.join(d, "f.svg")).read_text())

    def test_summaries_lead_back_to_events(self):
        s = self.m.summaries()
        acts = {r["key"] for r in s["host"]}
        self.assertIn("wait", acts)
        for r in s["host"] + s["waits"] + s["processes"]:
            self.assertTrue(r["highlight"])
        tw = s["topWaits"][0]
        ev = self.trace.store.event_by_id(int(tw["spanIdx"]))
        self.assertTrue(ev.start_ns <= tw["startNs"] and tw["startNs"] + tw["waitNs"] <= ev.end_ns)
        h = self.m.heatmap(40)
        self.assertEqual(len(h["rows"][0]["busy"]), 40)


@unittest.skipUnless(_QT, "PySide6 not installed")
class TestLargeLaneZoom(unittest.TestCase):
    """The activity index path on a lane too dense to enumerate: no opacity
    floor, the same apparent occupancy as an exact render where both apply,
    and the approximation is flagged."""

    @classmethod
    def setUpClass(cls):
        cls.app = QGuiApplication.instance() or QGuiApplication([])
        from src.gui.models import TimelineModel
        from src.gui.theme import Theme
        t = Trace(TraceMetadata())
        rng = random.Random(1)
        x = 0
        cls.ivs = []
        for i in range(60_000):
            x += rng.choice([200, 1500, 20_000])
            d = rng.choice([30, 300])
            t.add(SpanEvent("tick", Category.OPENMP, x, d, 1, 1, tags={"type": "work"}))
            cls.ivs.append((x, x + d))
        t.add(SpanEvent("other", Category.MPI, 0, 10, 1, 2, tags={"type": "send"}))
        t.finalize()
        cls.m = TimelineModel(t, Theme(dark=True))
        cls.lane = next(i for i, ex in enumerate(cls.m._lane_extra) if ex["category"] == "openmp")

    def test_zoomed_out_uses_the_index_and_stays_honest(self):
        a, b = self.m.viewStartNs, self.m.viewStartNs + self.m.traceDurationNs
        lr = self.m.render_lane(self.lane, a, b, 1000)
        self.assertEqual(lr.source, "index")
        want = true_occupancy(self.ivs, a, b, 1000)
        got = lod.apparent_occupancy(lr)
        # uniform-within-a-bin: within one index bin's share per column
        bin_share = lr.index_bin_ns / ((b - a) / 1000)
        self.assertLess(np.abs(got - want).max(), 2 * bin_share + 1e-9)
        self.assertLess(got.mean(), 0.2)                     # sparse: not a solid bar
        self.assertAlmostEqual(got.sum(), want.sum(), delta=0.01 * want.sum())

    def test_zoomed_in_switches_to_exact_without_changing_occupancy(self):
        a = self.m.viewStartNs + 0.3 * self.m.traceDurationNs
        b = a + 0.002 * self.m.traceDurationNs
        lr = self.m.render_lane(self.lane, a, b, 800)
        self.assertEqual(lr.source, "exact")
        np.testing.assert_allclose(lod.apparent_occupancy(lr), true_occupancy(self.ivs, a, b, 800), atol=1e-9)


class TestDenseCriticalPathBand(unittest.TestCase):
    """Zoomed out, the critical-path band is drawn per column (busy or
    waiting, whichever holds more of the column's path time) instead of one
    rectangle per step; it marks exactly the columns the steps touch."""

    def test_columns_match_the_steps(self):
        from src.gui.models import TimelineModel
        rng = random.Random(5)
        width = 300
        x0, x1, mode = [], [], []
        x = 0.0
        while x < width - 1:
            d = rng.choice([0.0, 0.05, 0.3, 2.5])
            x0.append(x)
            x1.append(min(float(width), x + d))
            mode.append(rng.choice([0, 0, 1]))
            x += d + rng.choice([0.0, 0.0, 0.2, 7.0])           # gaps too
        x0a, x1a, ma = np.asarray(x0), np.asarray(x1), np.asarray(mode, dtype=np.int8)
        runs = TimelineModel._cp_columns(x0a, x1a, ma, width)
        state = np.full(width, -1)
        for c0, c1, m, _t in runs:
            state[int(c0):int(c1)] = 0 if m == "busy" else 1
        busy = np.zeros(width)
        wait = np.zeros(width)
        for a, b, m in zip(x0, x1, mode):
            c = int(a)
            while c < width and (c < b or c == int(a)):
                ov = max(min(b, c + 1) - max(a, c), 1e-6 if c == int(a) else 0.0)
                (busy if m == 0 else wait)[c] += ov
                c += 1
        want = np.where((busy > 0) & (busy >= wait), 0, np.where(wait > 0, 1, -1))
        np.testing.assert_array_equal(state, want)
        self.assertTrue((state == -1).any())                     # gaps stay empty
        # runs never overlap and cover only marked columns
        self.assertEqual(sum(int(c1 - c0) for c0, c1, *_ in runs), int((want >= 0).sum()))


@unittest.skipUnless(_QT, "PySide6 not installed")
class TestSpanPrefetch(unittest.TestCase):
    """The per-lane prefetch that serves pans and small zooms from memory
    returns exactly the spans (and order) of the store's own window query,
    on both store kinds, with and without an event filter."""

    @staticmethod
    def build(store=None):
        t = Trace(TraceMetadata(), store=store) if store is not None else Trace(TraceMetadata())
        rng = random.Random(7)
        x = 0
        for i in range(4000):
            x += rng.choice([0, 50, 400, 3000])
            d = rng.choice([0, 20, 300, 5000, 90_000])          # zero-length and long spans too
            t.add(SpanEvent("k%d" % (i % 5), Category.OPENMP, x, d, 1, 1, tags={"type": "work"}))
        t.finalize()
        return t

    def check(self, trace):
        from src.gui.models import TimelineModel
        from src.gui.theme import Theme
        from src.core.store.common import SpanFilter
        m = TimelineModel(trace, Theme(dark=True))
        lane = next(n for n in m._lane_names if "openmp" in n)
        st = trace.store
        rng = random.Random(3)
        a0, dur = int(m.viewStartNs), int(m.traceDurationNs)
        for filt in (None, SpanFilter(min_dur_ns=100)):
            A = a0 + dur // 3
            w = dur // 50
            for step in range(60):
                # pans (mostly cache hits), zooms, and jumps (misses)
                r = rng.random()
                if r < 0.6:
                    A += rng.randint(-w // 5, w // 5)
                elif r < 0.85:
                    w = max(1, int(w * rng.choice([0.8, 1.25])))
                else:
                    A = a0 + rng.randint(0, dur)
                B = A + w
                got = [x.eid for x in m._window_spans(lane, A, B, filt)[0]]
                want = [x.eid for x in st.iter_spans(order="start", lane=lane, window=(A, B), filt=filt)]
                self.assertEqual(got, want, (step, A, B))

    def test_memory_store(self):
        self.app = QGuiApplication.instance() or QGuiApplication([])
        self.check(self.build())

    def test_disk_store(self):
        from src.core.store import DiskTraceStore
        self.app = QGuiApplication.instance() or QGuiApplication([])
        with tempfile.TemporaryDirectory() as d:
            self.check(self.build(DiskTraceStore(os.path.join(d, "p.hpstore"))))


@unittest.skipUnless(_QT, "PySide6 not installed")
class TestComparisonMatching(unittest.TestCase):
    """Selecting an event in one run identifies its counterpart by the
    comparison's identity, aligned phase and occurrence -- not by time."""

    @classmethod
    def setUpClass(cls):
        cls.app = QGuiApplication.instance() or QGuiApplication([])
        from tests import compare_scenarios as Sc
        from src.gui.comparison import ComparisonBridge
        from src.gui.models import TimelineModel
        from src.gui.theme import Theme
        cls.base, cls.cand = Sc.mpi_wait(False), Sc.mpi_wait(True)
        theme = Theme(dark=True)
        cls.bridge = ComparisonBridge(cls.cand, cls.base, theme)
        cls.cand_model = TimelineModel(cls.cand, theme)
        cls.bridge.attach_candidate_timeline(cls.cand_model)

    def computes(self, trace, pid):
        return sorted((s for s in trace.iter_spans() if s.name == "compute" and s.pid == pid),
                      key=lambda s: s.start_ns)

    def test_kth_occurrence_matches_kth_occurrence(self):
        cand = self.computes(self.cand, 100)
        base = self.computes(self.base, 100)
        for k in (0, 2, 4):
            m = self.bridge.matchEvent("candidate", float(cand[k].eid))
            self.assertEqual(m["counterpart"]["eid"], float(base[k].eid), k)
            self.assertEqual(m["durationNs"], [float(base[k].duration_ns), float(cand[k].duration_ns)])
            self.assertEqual(m["deltaNs"], 10_000_000.0)
            # the scenario's phases are its iterations: the k-th compute is
            # the one occurrence of its node in aligned phase pair k
            self.assertIn("identity #", m["basis"])
            self.assertIn("aligned phase pair", m["basis"])
            self.assertIn("occurrence 1 of 1", m["basis"])
            self.assertEqual(m["status"], "regressed")
        # and back
        m = self.bridge.matchEvent("baseline", float(base[1].eid))
        self.assertEqual(m["counterpart"]["eid"], float(cand[1].eid))

    def test_counterpart_is_on_the_matching_lane_of_the_other_model(self):
        cand = self.computes(self.cand, 100)
        m = self.bridge.matchEvent("candidate", float(cand[0].eid))
        bm = self.bridge.baselineTimeline
        span = self.base.store.event_by_id(int(m["counterpart"]["eid"]))
        self.assertEqual(m["counterpart"]["laneIndex"], bm._lane_of_span(span))

    def test_trace_back_separates_observed_and_inferred(self):
        recv = next(s for s in self.cand.iter_spans() if s.name == "MPI_Recv")
        m = self.bridge.matchEvent("candidate", float(recv.eid))
        steps = m["steps"]
        self.assertEqual(steps[0]["kind"], "measured")
        self.assertTrue(all("derived" in s["kind"] for s in steps[1:]))
        self.assertEqual([s["label"] for s in steps[1:]], ["MPI_Send", "compute"])

    def test_event_details_gain_a_comparison_section(self):
        cand = self.computes(self.cand, 100)
        lane = self.cand_model._lane_of_span(cand[0])
        d = self.cand_model.eventDetails(lane, float(cand[0].eid))
        sec = next(s for s in d["sections"] if s["title"] == "Comparison")
        labels = [f["label"] for f in sec["fields"]]
        self.assertIn("Duration baseline → candidate", labels)
        self.assertIn("Basis", labels)

    def test_status_colouring_and_alignment(self):
        self.cand_model.setColorMode("comparison")
        self.assertEqual(self.cand_model.colorMode, "comparison")
        cand = self.computes(self.cand, 100)
        from src.gui import theme as theme_mod
        self.assertEqual(self.cand_model._span_color(cand[0]), theme_mod.change_color("regressed", True))
        self.cand_model.setColorMode("function")
        st = self.bridge.timelineStarts
        r = self.bridge.alignmentOffset("start", -1, -1, 0.0, 0.0)
        self.assertEqual(r["offsetNs"], st["candidate"] - st["baseline"])
        strip = self.bridge.deltaStrip
        self.assertTrue(strip)
        self.assertTrue(all("deltaStatus" in p for p in strip))


if __name__ == "__main__":
    unittest.main()
