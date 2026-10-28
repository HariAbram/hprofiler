"""
The Timeline presents the critical-path model's results as computed, and
never claims what was not evaluated (src/gui/timeline_analysis.py,
timeline_details.py, models.py, timeline_paint.py -- no QML engine):

  * alternative worlds: "not evaluated" on a trace over the automatic limit
    (no edge "unambiguous", no "every world", the worlds overlay marks
    nothing) until evaluated on request on a background thread; "none
    found"; "evaluation truncated"; ambiguous edges with their other
    candidates, evaluated worlds, the least robust one, its alternative
    edges;
  * the four variants by their semantics and the modeled range: the range's
    ends are the only minimum / maximum, and nothing is labelled a bound;
  * unknown blocking time has its own path-band code and paint, apart from
    gated waiting; per-event gated tails;
  * clocks: contradicted and dropped edges are listed and drawn on request,
    never as active dependencies; undecidable steps lead to their edges;
  * blame by category, joint releases, sensitivity / replay rows;
  * probabilities: unquantified without a calibration; with one, its
    source, version, model and coverage, per-edge probabilities;
  * stable / variant-only / near-critical / world membership marks.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtGui import QColor, QGuiApplication, QImage, QPainter
    _QT = True
except ImportError:          # pragma: no cover
    _QT = False

from src.analysis import cp_alternatives as alt
from src.analysis import cp_model as cm
from src.analysis import criticalpath as cp
from tests import evidence_scenarios as E
from tests.test_cp_model import _trace, recv, send, work

if _QT:
    from src.gui import theme as theme_mod
    from src.gui import timeline_analysis as ta
    from src.gui import timeline_paint as tp
    from src.gui.models import TimelineModel
    from src.gui.theme import Theme

QML = Path(__file__).resolve().parent.parent / "src" / "gui" / "qml" / "components"


def model(trace) -> "TimelineModel":
    m = TimelineModel(trace, Theme(dark=True))
    m.startAnalysis()
    assert m.wait_for_analysis(120), m.analysisMessage
    return m


def cp_fields(m, s) -> dict:
    d = m.eventDetails(m._lane_of_span(s), float(s.eid))
    sec = next(x for x in d["sections"] if x["title"] == "Critical path")
    return {f["label"]: f for f in sec["fields"]}


def example1():
    return _trace([work("work", 2, 0, 900), send(2, 900, 1, 0, 9, end=950), work("post", 2, 1100, 1400),
                   work("prep", 1, 0, 200), recv(1, 200, 1000, 0, 1, 9), work("solve", 1, 1000, 1600)])


@unittest.skipUnless(_QT, "PySide6 not installed")
class _Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._app = QGuiApplication.instance() or QGuiApplication([])


class TestAlternativesNotEvaluated(_Base):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._limit = cp.ALTERNATIVES_AUTO_NODE_LIMIT
        cp.ALTERNATIVES_AUTO_NODE_LIMIT = 50            # below the trace: skipped, as above 100k nodes
        try:
            cls.m = model(E.big_trace())
        finally:
            cp.ALTERNATIVES_AUTO_NODE_LIMIT = cls._limit
        cls.an = cls.m._analysis

    def test_state_is_not_evaluated_never_none(self):
        a = self.m.criticalSummary["alternatives"]
        self.assertEqual((a["state"], a["stateLabel"]), ("not_evaluated", "not evaluated"))
        self.assertIn("exceed the automatic limit", a["reason"])
        self.assertFalse(a["evaluated"])
        self.assertNotIn("every", a)
        for k in range(len(self.an.edges)):
            d = self.m.edgeDetails(k)
            self.assertTrue(d["alternativesText"].startswith("not evaluated"), d["alternativesText"])
            self.assertNotEqual(d["alternativesText"], "none")
            if d["status"] in ("observed", "inferred"):
                self.assertTrue(d["statusShown"].endswith("(alternatives not evaluated)"), d["statusShown"])

    def test_no_event_is_claimed_on_every_world(self):
        for s in self.m._trace.iter_spans():
            if s.eid not in self.an.path_eids:
                continue
            f = cp_fields(self.m, s)
            self.assertEqual(f["Across alternatives"]["kind"], "unavailable")
            self.assertIn("not evaluated", f["Across alternatives"]["reason"])
            self.assertFalse(any("every evaluated world" in x["value"] or "every alternative" in x["value"]
                                 for x in f.values()))
        self.m.setOverlay({"membership": "worlds"})
        self.assertEqual(self.m._membership_sets(), (None, set()))
        self.m.setOverlay({"membership": ""})

    def test_evaluating_on_request_runs_in_the_background_and_updates_everything(self):
        m = model(E.big_trace())
        m._analysis.report.alternatives = None
        ta.apply_alternatives(m._analysis, None, n_nodes=2 * m._analysis.report.model.graph.n)
        self.assertEqual(m._analysis.alternatives_state, "not_evaluated")
        m.evaluateAlternatives(2)
        self.assertTrue(m.alternativesJob["running"])          # returned at once: a worker thread
        self.assertTrue(m.wait_for_alternatives(120), m.alternativesJob)
        an = m._analysis
        self.assertIn(an.alternatives_state, ("evaluated", "truncated"))
        a = m.criticalSummary["alternatives"]
        self.assertTrue(a["evaluated"])
        self.assertGreaterEqual(len(a["worldRows"]), 2)
        self.assertIn("ambiguous", {e.status for e in an.edges})
        self.assertIn("world(s) evaluated", m.alternativesJob["message"])


class TestAlternativesStates(_Base):
    def test_none_found(self):
        m = model(_trace([work("a", 2, 0, 100), send(2, 100, 1, 0, 9), recv(1, 50, 400, 0, 1, 9)]))
        an = m._analysis
        self.assertEqual(an.alternatives_state, "none_found")
        self.assertEqual(m.criticalSummary["alternatives"]["stateLabel"], "none found")
        p2p = next(k for k, e in enumerate(an.edges) if e.kind == "p2p")
        self.assertEqual(m.edgeDetails(p2p)["alternativesText"], "none found")
        s = next(s for s in m._trace.iter_spans() if s.eid in an.path_eids)
        self.assertIn("none were found", cp_fields(m, s)["Across alternatives"]["value"])

    def test_truncated_and_cancelled(self):
        t = E.evidence_trace()
        graph = cp.load_or_build_graph(t)
        res = cm.analyze_model(t, graph, sensitivity=0, keep_skeleton=True)
        a = alt.analyze_alternatives(t, graph, res, samples=0, one_change=0)
        st, why = ta.alternatives_state(a, 2 * res.graph.n)
        self.assertEqual(st, "truncated")
        self.assertIn("1 choice units touch the recorded path; 0 got a one-change world", why)
        a = alt.analyze_alternatives(t, graph, res, samples=3, cancel=lambda: True)
        self.assertEqual(ta.alternatives_state(a, 1)[0], "truncated")
        self.assertIn("stopped early", ta.alternatives_state(a, 1)[1])

    def test_ambiguous_edges_worlds_and_alternative_edges(self):
        m = model(E.evidence_trace())
        an = m._analysis
        self.assertEqual(an.alternatives_state, "evaluated")
        amb = next(k for k, e in enumerate(an.edges) if e.status == "ambiguous")
        d = m.edgeDetails(amb)
        self.assertEqual(d["group"]["cardinality"], "exactly_one")
        self.assertEqual([c["status"] for c in d["candidates"]], ["unsupported"])
        self.assertEqual(d["candidates"][0]["source"]["name"], "MPI_Send")
        self.assertGreaterEqual(d["candidates"][0]["source"]["laneIndex"], 0)
        self.assertTrue(any(t["kind"] == "several_candidates" and t.get("against") for t in d["evidenceTree"]))
        self.assertTrue(any(r.startswith("posting_order:") and "tag=9" in r for r in d["records"]))
        rows = m.criticalSummary["alternatives"]["worldRows"]
        worst = m.criticalSummary["alternatives"]["worstWorld"]
        self.assertEqual(rows[worst]["robustShare"], min(r["robustShare"] for r in rows))
        alt_ids = an.worlds[worst]["altEdges"]
        self.assertTrue(alt_ids)
        self.assertEqual(an.extra_edges[alt_ids[0] - len(an.edges)].role, "alternative")
        a, b = m.viewStartNs, m.viewStartNs + m.traceDurationNs
        self.assertFalse(set(alt_ids) & set(m._visible_extra_ids(a, b)))       # only for the selected world
        m.setOverlay({"membership": "world", "world": worst})
        self.assertTrue(set(alt_ids) <= set(m._visible_extra_ids(a, b)))
        self.assertEqual(m.edgeDetails(alt_ids[0])["role"], "alternative")
        m.setOverlay({"membership": "", "world": -1})


class TestVariantsAndRange(_Base):
    def test_variants_by_semantics_and_the_range_ends(self):
        m = model(example1())
        sm = m.criticalSummary
        rows = {r["name"]: r for r in sm["variantsTable"]}
        self.assertEqual(set(rows), set(cm.VARIANTS))
        self.assertEqual({k: r["modeledNs"] for k, r in rows.items()},
                         {"default": 1500, "unrecorded_as_work": 1500, "direct_evidence_only": 1250,
                          "gated_tail_as_work": 1600})
        self.assertEqual((rows["direct_evidence_only"]["edges"], rows["gated_tail_as_work"]["tails"],
                          rows["unrecorded_as_work"]["unrecorded"]), ("direct evidence only", "processing", "work"))
        r = sm["modeledRange"]
        self.assertEqual((r["minNs"], r["maxNs"]), (1250, 1600))
        self.assertEqual(r["minVariants"], ["direct-evidence edges only"])
        self.assertEqual([k for k, v in rows.items() if v["isMin"]], ["direct_evidence_only"])
        self.assertEqual([k for k, v in rows.items() if v["isMax"]], ["gated_tail_as_work"])
        for v in rows.values():
            self.assertTrue(r["minNs"] <= v["modeledNs"] <= r["maxNs"])

    def test_no_value_is_labelled_a_bound(self):
        for f in ("TimelineCriticalPathTab.qml", "TimelineBottomPanel.qml"):
            text = (QML / f).read_text()
            self.assertNotIn("lower bound", text, f)
            self.assertNotIn("upper bound", text, f)

    def test_stable_variant_only_and_near_critical_sets(self):
        m = model(E.evidence_trace())
        an = m._analysis
        sets = list(an.variant_eids.values())
        self.assertEqual(an.stable_eids, set.intersection(*sets))
        self.assertEqual(an.variant_only_eids, set.union(*sets) - an.stable_eids)
        self.assertTrue(an.stable_eids and an.variant_only_eids and an.near_eids)
        sm = m.criticalSummary
        self.assertEqual((sm["stableCount"], sm["variantOnlyCount"]), (len(an.stable_eids), len(an.variant_only_eids)))


class TestBlockingTime(_Base):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.m = model(E.evidence_trace())
        cls.an = cls.m._analysis

    def test_unknown_time_has_its_own_band_code(self):
        arr = self.m._cp_arrays()
        unknown_steps = [k for k, s in enumerate(self.an.steps) if s.mode == "unknown"]
        self.assertTrue(unknown_steps)
        self.assertEqual({int(arr["mode"][k]) for k in unknown_steps}, {3})
        s = next(s for s in self.m._trace.iter_spans() if s.eid in self.an.unknown_eids and s.eid in self.an.path_eids)
        li = self.m._lane_of_span(s)
        self.m.setOverlay({"criticalPath": True})
        lr = self.m.render_lane(li, self.m.viewStartNs, self.m.viewStartNs + self.m.traceDurationNs, 600)
        self.assertIn("unknown", {x[2] for x in self.m._lane_options(li, lr).cp_steps})
        self.assertEqual(self.m._cp_columns(*[__import__("numpy").array(v) for v in ([0.0], [50.0], [3])], 100)[0][2],
                         "unknown")

    def test_unknown_is_painted_unlike_gated_waiting(self):
        style = tp.Style.for_theme(True)
        lr = __import__("src.gui.timeline_lod", fromlist=["render_spans"]).render_spans([], 0, 100, 100,
                                                                                         color_of=lambda s: "")

        def pixel(opts, x, y):
            img = QImage(100, 20, QImage.Format.Format_ARGB32)
            img.fill(QColor(style.background))
            p = QPainter(img)
            tp.paint_lane(p, lr, 100, 20, style, opts)
            p.end()
            return img.pixelColor(x, y).name()
        band_unknown = pixel(tp.LaneOptions(cp_steps=[(0, 60, "unknown", "")]), 30, 1)
        band_wait = pixel(tp.LaneOptions(cp_steps=[(0, 60, "wait", "")]), 30, 1)
        self.assertNotEqual(band_unknown, band_wait)
        veil_unknown = pixel(tp.LaneOptions(unknown_calls=[(0, 60)]), 30, tp.MARGIN)
        veil_gated = pixel(tp.LaneOptions(inferred_waits=[(0, 60)]), 30, tp.MARGIN)
        self.assertNotEqual(veil_unknown, veil_gated)
        self.assertEqual(veil_unknown, QColor(theme_mod.evidence_color("unknown", True)).name())
        kinds = [e["kind"] for e in tp.legend_entries(True, color_mode="function", has_density=False,
                                                      has_unobserved=False, critical=True)]
        self.assertIn("unknown", kinds)

    def test_hover_and_details_name_the_mode(self):
        spans = list(self.m._trace.iter_spans())
        unk = next(s for s in spans if s.eid in self.an.unknown_eids)
        gated = next(s for s in spans if s.eid in self.an.gated_of_eid and s.name == "MPI_Recv")
        li = self.m._lane_of_span(unk)
        self.assertIn("unknown blocking", self.m.spanAt(li, float(unk.eid))["analysisNote"])
        self.assertTrue(cp_fields(self.m, unk)["Blocking call"]["value"].startswith("unknown"))
        self.assertTrue(cp_fields(self.m, gated)["Blocking call"]["value"].startswith("gated"))
        self.assertNotIn("Gated tail", cp_fields(self.m, unk))
        self.m.setOverlay({"unknownBlocking": True})
        lr = self.m.render_lane(li, self.m.viewStartNs, self.m.viewStartNs + self.m.traceDurationNs, 600)
        self.assertEqual(len(self.m._lane_options(li, lr).unknown_calls), 1)
        self.m.setOverlay({"unknownBlocking": False})

    def test_per_event_gated_tail(self):
        m = model(example1())
        r = next(s for s in m._trace.iter_spans() if s.name == "MPI_Recv")
        f = cp_fields(m, r)
        self.assertEqual(f["Gated tail"]["value"],
                         "at most 100ns of its 800ns may have been processing (the time after its release)")
        self.assertIn("released at +900ns", f["Blocking call"]["value"])
        g = m._analysis.report.model.graph
        j = next(i for i, s in enumerate(m._trace.iter_spans()) if s.name == "MPI_Recv")
        self.assertEqual(m._analysis.gated_of_eid[r.eid]["tailNs"], cm._tail_after_release(g, j))


class TestClocks(_Base):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.m = model(E.evidence_trace())
        cls.an = cls.m._analysis

    def test_contradicted_edges_are_listed_never_active(self):
        c = self.m.criticalSummary["clock"]
        self.assertEqual((c["disabledByClock"], c["droppedByTime"]), (1, 1))
        disabled = [x for x in self.an.extra_edges if x.role == "disabled"]
        self.assertEqual(sorted(x.reason for x in disabled), ["clock_bound", "time_contradiction"])
        st = {r["status"]: r for r in self.m.criticalSummary["statusSummary"]}
        self.assertEqual((st["contradicted"]["edges"], st["contradicted"]["disabled"]), (0, 2))
        self.assertFalse(any(e.status == "contradicted" for e in self.an.edges))
        a, b = self.m.viewStartNs, self.m.viewStartNs + self.m.traceDurationNs
        ids = [len(self.an.edges) + i for i, x in enumerate(self.an.extra_edges) if x.role == "disabled"]
        self.assertFalse(set(ids) & set(self.m._visible_extra_ids(a, b)))
        self.m.setOverlay({"disabledEdges": True})
        self.assertTrue(set(ids) <= set(self.m._visible_extra_ids(a, b)))
        self.m.setOverlay({"disabledEdges": False})
        d = self.m.edgeDetails(ids[0])
        self.assertEqual((d["role"], d["status"]), ("disabled", "contradicted"))
        self.assertIn("not an active dependency", d["reason"])

    def test_domains_and_undecidable_steps(self):
        c = self.m.criticalSummary["clock"]
        shifted = [d for d in c["domains"] if d["shifted"]]
        self.assertEqual([(d["domain"], d["shift"]) for d in shifted], [("opencl/0/9", "−400.0µs")])
        steps = self.m.criticalSummary["undecidableSteps"]
        self.assertEqual(len(steps), 1)
        u = steps[0]
        self.assertEqual((u["chosenKind"], u["alternativeKind"], u["marginNs"]), ("device_wait", "device_wait", 10_000))
        for k in (u["chosenEdge"], u["alternativeEdge"]):
            self.assertEqual(self.m.edgeDetails(k)["kind"], "device_wait")
        self.assertNotEqual(u["chosenDomain"], u["altDomain"])


class TestBlameAndSensitivity(_Base):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.m = model(E.evidence_trace())
        cls.an = cls.m._analysis

    def test_blame_by_category_joint_releases(self):
        b = self.m.criticalSummary["blame"]
        self.assertTrue(b["available"])
        self.assertAlmostEqual(sum(r["ns"] for r in b["byCategory"]), b["waitingNs"], delta=10)
        buckets = {r["key"]: r["ns"] for r in b["byCategory"] if r["bucket"]}
        self.assertEqual(buckets, {"(unrecorded)": 5000.0})        # the gap between the sync and the send
        joint = b["jointWaits"]
        wa = next(w for w in joint if w["name"] == "MPI_Waitall")
        self.assertEqual([r["name"] for r in wa["releasers"]], ["MPI_Send", "MPI_Send"])
        producer = next(s for s in self.m._trace.iter_spans() if s.name == "produce")
        sends = [s for s in self.m._trace.iter_spans() if s.name == "MPI_Send" and s.pid in (3, 4)]
        self.assertTrue(all(self.an.joint_of_eid.get(s.eid) for s in sends))
        self.assertIn("Waiting it caused", cp_fields(self.m, producer))

    def test_waiting_passed_on_through_a_waiting_call(self):
        rows = self.m.criticalSummary["blame"]["passedOn"]
        self.assertEqual([(r["name"], r["ns"]) for r in rows], [("cudaStreamSynchronize", 270_000)])
        sync = next(s for s in self.m._trace.iter_spans() if s.name == "cudaStreamSynchronize" and s.pid == 11)
        self.assertEqual(cp_fields(self.m, sync)["Waiting passed on"]["value"], "270.0µs")

    def test_joint_releases_are_marked_on_the_timeline(self):
        from src.gui import timeline_paint
        seen = []
        orig = timeline_paint.paint_joint_markers
        timeline_paint.paint_joint_markers = lambda p, pts, style: seen.extend(pts)
        try:
            self.m.setOverlay({"criticalPath": True})
            img = QImage(900, 600, QImage.Format.Format_ARGB32)
            p = QPainter(img)
            a, b = self.m.viewStartNs, self.m.viewStartNs + self.m.traceDurationNs
            self.m.paint_overlay_item(p, 900, 600, a, b, 0.0, 20, -1, -1.0)
            p.end()
        finally:
            timeline_paint.paint_joint_markers = orig
        joint = [w for w in self.an.blame_waits if w["joint"]]
        self.assertEqual(len(seen), len([w for w in joint if self.m._anchor_pos.get(w["laneIndex"]) is not None]))
        self.assertTrue(seen)
        style = tp.Style.for_theme(True)
        img = QImage(40, 40, QImage.Format.Format_ARGB32)
        img.fill(QColor(style.background))
        p = QPainter(img)
        tp.paint_joint_markers(p, [(20, 20)], style)
        p.end()
        self.assertEqual(img.pixelColor(20, 20).name(), QColor(style.background).name())   # hollow
        self.assertNotEqual(img.pixelColor(20, 15).name(), QColor(style.background).name())

    def test_sensitivity_rows_and_replay_scope(self):
        rows = self.m.criticalSummary["sensitivityRows"]
        res = self.an.report.model
        self.assertEqual([(r["modeledNs"], r["replayNs"]) for r in rows],
                         [(x["modeled_reduction_ns"], x["replay_reduction_ns"]) for x in res.sensitivity])
        spans = [r for r in rows if r["scope"] == "span"]
        self.assertTrue(spans and all(r["laneIndex"] >= 0 for r in spans))
        other = next(s for s in self.m._trace.iter_spans()
                     if s.eid in self.an.path_eids and s.eid not in self.an.sensitivity_eids)
        f = cp_fields(self.m, other)
        self.assertEqual(f["Replay"]["kind"], "unavailable")
        self.assertIn("only for the sensitivity candidates", f["Replay"]["reason"])


class TestProbabilities(_Base):
    def test_unquantified_without_a_calibration(self):
        os.environ.pop("HPROFILER_EDGE_CALIBRATION", None)
        m = model(E.evidence_trace())
        p = m.criticalSummary["probabilities"]
        self.assertEqual((p["status"], p["calibrated"]), ("unquantified", False))
        p2p = next(k for k, e in enumerate(m._analysis.edges) if e.kind == "p2p")
        self.assertEqual(m.edgeDetails(p2p)["probability"], "unquantified (no calibration given)")

    def test_calibrated_source_version_model_coverage_and_per_edge(self):
        with tempfile.TemporaryDirectory() as d:
            path = E.calibration_file(os.path.join(d, "cal.json"))
            os.environ["HPROFILER_EDGE_CALIBRATION"] = path
            try:
                m = model(E.evidence_trace())
            finally:
                os.environ.pop("HPROFILER_EDGE_CALIBRATION", None)
        p = m.criticalSummary["probabilities"]
        self.assertTrue(p["calibrated"])
        self.assertEqual((p["source"], p["dataset"]), (path, "synthetic GUI test scenarios"))
        self.assertTrue(p["versionMatches"])
        self.assertIn("isotonic", p["model"])
        self.assertEqual(p["strata"], 1)
        self.assertGreaterEqual(p["edgesQuantified"], 2)
        self.assertGreaterEqual(p["quantifiedSteps"], 1)
        an = m._analysis
        amb = next(k for k, e in enumerate(an.edges) if e.status == "ambiguous")
        inf = next(k for k, e in enumerate(an.edges) if e.kind == "p2p" and e.status == "inferred")
        self.assertEqual(m.edgeDetails(amb)["probability"], "0.55, calibrated on synthetic GUI test scenarios")
        self.assertEqual(m.edgeDetails(inf)["probability"], "0.90, calibrated on synthetic GUI test scenarios")
        other = next(k for k, e in enumerate(an.edges) if e.kind == "launch")
        self.assertIn("has no stratum", m.edgeDetails(other)["probability"])


class TestMembershipMarks(_Base):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.m = model(E.evidence_trace())
        cls.an = cls.m._analysis

    def marks(self):
        out = {}
        for li in range(len(self.m._lane_names)):
            lr = self.m.render_lane(li, self.m.viewStartNs, self.m.viewStartNs + self.m.traceDurationNs, 800)
            for x0, x1, key in self.m._lane_options(li, lr).marks or ():
                out.setdefault(key, 0)
                out[key] += 1
        return out

    def test_each_mode_marks_its_sets(self):
        an = self.an
        self.m.setOverlay({"membership": "variants"})
        self.assertEqual(self.m._membership_sets(), (an.stable_eids | an.variant_only_eids, an.variant_only_eids))
        self.assertEqual(set(self.marks()), {"robust", "partial"})
        self.m.setOverlay({"membership": "near"})
        self.assertEqual(self.m._membership_sets()[0], an.near_eids)
        self.assertEqual(set(self.marks()), {"robust"})
        self.m.setOverlay({"membership": "variant", "variant": "direct_evidence_only"})
        self.assertEqual(self.m._membership_sets(), (an.variant_eids["direct_evidence_only"], set()))
        self.assertEqual(set(self.marks()), {"robust"})
        self.m.setOverlay({"membership": "worlds"})
        keep, partial = self.m._membership_sets()
        self.assertEqual((keep, partial), (an.robust_some, an.robust_some - an.robust_every))
        self.assertEqual(set(self.marks()), {"robust", "partial"})
        self.m.setOverlay({"membership": ""})
        self.assertEqual(self.marks(), {})

    def test_marks_paint_their_colours(self):
        style = tp.Style.for_theme(True)
        lr = __import__("src.gui.timeline_lod", fromlist=["render_spans"]).render_spans([], 0, 100, 100,
                                                                                         color_of=lambda s: "")
        img = QImage(100, 20, QImage.Format.Format_ARGB32)
        img.fill(QColor(style.background))
        p = QPainter(img)
        tp.paint_lane(p, lr, 100, 20, style, tp.LaneOptions(marks=[(10, 40, "robust"), (50, 90, "partial")]))
        p.end()
        self.assertEqual(img.pixelColor(20, 19).name(), QColor(theme_mod.evidence_color("robust", True)).name())
        self.assertNotEqual(img.pixelColor(60, 19).name(), img.pixelColor(20, 19).name())
        self.assertEqual(img.pixelColor(20, 5).name(), QColor(style.background).name())     # only the strip


if __name__ == "__main__":
    unittest.main()
