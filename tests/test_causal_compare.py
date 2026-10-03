"""
Structure- and causality-aware comparison (src/analysis/projection.py,
src/analysis/causal_compare.py), on synthetic before/after pairs whose
single difference is known (tests/compare_scenarios.py):

  * identical function names under different call paths
  * reordered independent work (must not be reported)
  * inserted iterations (alignment gap, "more invocations")
  * changed stream overlap ("lost overlap", stream / dependency evidence)
  * MPI wait propagation (the wait is traced to the slower sender's work)
  * a non-critical kernel becoming critical ("moved onto the critical path")

plus robustness to different raw pids / tids / stream handles /
communicator ids / timestamps, phase detection, matching a renamed
function through its source location, the confidence-driven fallback to
the (category, name) aggregate comparison, memory/disk store parity and
the JSON / text outputs.
"""
from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.analysis import causal_compare as cc
from src.analysis import compare as cmp
from src.analysis.projection import build_projection, normalize_name
from src.core.events import Category, SpanEvent
from src.core.store import DiskTraceStore
from src.core.trace import Trace, TraceMetadata
from tests import compare_scenarios as S

MS = S.MS


def find(rows, label, **roles):
    out = [r for r in rows if r["label"] == label and all(r["roleMap"].get(k) == v for k, v in roles.items())]
    return out


def relabeled(trace: Trace, *, shift: int = 7_000_000_000, pid_off: int = 5000, tid_off: int = 300) -> Trace:
    """The same run as seen by another execution: other pids/tids, stream
    handles, communicator ids and clock origin."""
    t = Trace(TraceMetadata(command=trace.metadata.command))
    for s in trace.iter_spans():
        tags = dict(s.tags)
        if "stream" in tags:
            tags["stream"] = f"0x7f{int(str(tags['stream']).lstrip('n') or 0) + 0xa000:x}"
        if tags.get("commid") not in (None, "-1"):
            tags["commid"] = str(int(tags["commid"]) + 40)
        t.add(SpanEvent(s.name, s.category, s.start_ns + shift, s.duration_ns, s.pid + pid_off,
                        s.tid + tid_off if s.tid else 0, tags, list(s.stack_frames),
                        f"x{s.span_id}" if s.span_id else "", f"x{s.parent_span_id}" if s.parent_span_id else ""))
    return t


class TestCallPaths(unittest.TestCase):
    def test_same_name_under_different_paths_is_attributed_to_the_right_path(self):
        r = cc.compare_traces(S.same_name_different_paths(False), S.same_name_different_paths(True))
        self.assertEqual(r["alignment"]["method"], "phase-aligned")
        self.assertGreater(r["alignment"]["confidence"], 0.9)
        updates = [x for x in r["contributors"] if x["label"] == "update"]
        self.assertEqual(len(updates), 1)
        u = updates[0]
        self.assertEqual(u["fullPath"], ["nvtx:solve_y", "openmp:update"])
        self.assertEqual(u["cause"], "increased_work")
        self.assertAlmostEqual(u["ownDeltaNs"], 10 * MS, delta=MS / 100)
        self.assertEqual(u["measured"]["count"], [5, 5])
        # solve_x > update did not change and is nowhere among the changes
        for lst in ("contributors", "offCriticalPath", "improvements"):
            self.assertFalse([x for x in r[lst] if x["fullPath"] == ["nvtx:solve_x", "openmp:update"]])
        # the aggregate (category, name) view only sees one merged `update`
        agg = [x for x in r["aggregate"]["aggregates"] if x["name"] == "update"]
        self.assertEqual(len(agg), 1)

    def test_renamed_function_matches_through_source_location(self):
        def trace(name):
            t = Trace(TraceMetadata(command="./r"))
            c = MS
            for _ in range(4):
                t.add(SpanEvent("step", Category.NVTX, c, 3 * MS, 1, 1, {"type": "nvtx_range"}))
                t.add(SpanEvent(name, Category.OPENMP, c + 100_000, 2 * MS, 1, 1,
                                {"type": "work", "file": "/s/k.c", "line": "12"}))
                c += int(3.5 * MS)
            return t
        r = cc.compare_traces(trace("kernel_v1"), trace("kernel_v2"))
        self.assertFalse(r["newWork"])
        self.assertFalse(r["removedWork"])
        pa = build_projection(trace("kernel_v1"))
        pb = build_projection(trace("kernel_v2"))
        pairs = cc.match_nodes(pa, pa.phase_nodes(1), pb, pb.phase_nodes(1))
        renamed = [(a, b, s) for a, b, s in pairs if a and b and a.key.leaf != b.key.leaf]
        self.assertEqual(len(renamed), 1)
        self.assertGreaterEqual(renamed[0][2], cc.MIN_NODE_SCORE)


class TestReorderAndRelabel(unittest.TestCase):
    def test_reordered_independent_work_is_not_a_regression(self):
        r = cc.compare_traces(S.reordered(False), S.reordered(True))
        self.assertEqual(r["alignment"]["method"], "phase-aligned")
        self.assertGreater(r["alignment"]["confidence"], 0.9)
        self.assertEqual(r["contributors"], [])
        self.assertEqual(r["offCriticalPath"], [])
        self.assertEqual(r["newWork"], [])
        self.assertEqual(r["removedWork"], [])
        self.assertEqual(r["wallTime"]["status"], cmp.STATUS_UNCHANGED)

    def test_other_pids_tids_streams_comm_ids_and_clock_match_exactly(self):
        for build in (lambda: S.mpi_wait(False), lambda: S.stream_overlap(False), lambda: S.iterations(6)):
            base = build()
            r = cc.compare_traces(base, relabeled(build()))
            self.assertEqual(r["alignment"]["method"], "phase-aligned")
            self.assertAlmostEqual(r["alignment"]["confidence"], 1.0, places=3)
            for lst in ("contributors", "offCriticalPath", "newWork", "removedWork", "improvements"):
                self.assertEqual(r[lst], [], lst)


class TestIterations(unittest.TestCase):
    def test_phases_detected_prologue_iterations_epilogue(self):
        p = build_projection(S.iterations(10))
        self.assertEqual(p.phase_method, "iterations")
        self.assertEqual([ph.kind for ph in p.phases], ["prologue"] + ["iteration"] * 10 + ["epilogue"])
        self.assertTrue(p.anchor.endswith("compute"))

    def test_iteration_start_is_rotated_to_the_first_call(self):
        # [launch, launch, sync] per iteration: the sync is the most regular
        # anchor, but iterations must start at the first launch
        p = build_projection(S.stream_overlap(False))
        self.assertEqual([ph.kind for ph in p.phases], ["iteration"] * 5)

    def test_inserted_iteration_is_a_gap_and_more_invocations(self):
        r = cc.compare_traces(S.iterations(10), S.iterations(11))
        al = r["alignment"]
        self.assertEqual(al["method"], "phase-aligned")
        statuses = [p["status"] for p in al["pairs"]]
        self.assertEqual(statuses.count("inserted"), 1)
        self.assertEqual(statuses.count("matched"), 12)
        self.assertGreater(al["confidence"], 0.9)
        self.assertTrue(any("inserted" in n for n in al["notes"]))
        compute = find(r["contributors"], "compute")
        self.assertEqual(len(compute), 1)
        self.assertEqual(compute[0]["cause"], "more_invocations")
        self.assertEqual(compute[0]["measured"]["count"], [10, 11])

    def test_removed_iterations_and_no_false_positives_elsewhere(self):
        r = cc.compare_traces(S.iterations(12), S.iterations(10))
        statuses = [p["status"] for p in r["alignment"]["pairs"]]
        self.assertEqual(statuses.count("removed"), 2)
        self.assertEqual(r["contributors"], [])
        imp = find(r["improvements"], "compute")
        self.assertEqual(imp[0]["cause"], "more_invocations")
        self.assertEqual(imp[0]["causeLabel"], "fewer invocations")


class TestOverlapAndCriticalPath(unittest.TestCase):
    def test_changed_stream_overlap_is_lost_overlap(self):
        r = cc.compare_traces(S.stream_overlap(False), S.stream_overlap(True))
        self.assertEqual(r["wallTime"]["status"], cmp.STATUS_REGRESSED)
        top = r["contributors"][0]
        self.assertEqual(top["label"], "_Z1Bv")
        self.assertEqual(top["cause"], "lost_overlap")
        self.assertEqual(top["match"]["kind"], "structural")      # stream role changed s1 -> s0
        types = {e["type"] for e in top["evidence"]}
        self.assertIn("edge_added", types)                          # stream order A -> B
        self.assertIn("stream_changed", types)
        self.assertAlmostEqual(top["impactNs"], 15 * MS, delta=MS / 10)
        oa, ob = top["measured"]["overlapNs"]
        self.assertGreater(oa, 14 * MS)
        self.assertEqual(ob, 0)
        # the device sync waits longer only because of B: a propagated wait
        sync = find(r["propagated"], "cudaDeviceSynchronize")
        self.assertEqual(len(sync), 1)
        self.assertEqual(sync[0]["chain"][-1]["label"], "_Z1Bv")
        # A did not change at all
        self.assertFalse(find(r["contributors"] + r["offCriticalPath"], "_Z1Av"))

    def test_non_critical_kernel_becoming_critical(self):
        r = cc.compare_traces(S.becomes_critical(False), S.becomes_critical(True))
        k = find(r["contributors"] + r["propagated"], "_Z1Kv")
        self.assertTrue(k)
        k = k[0]
        self.assertEqual(k["cause"], "moved_onto_critical_path")
        self.assertEqual(k["derived"]["onCriticalPath"], [False, True])
        self.assertEqual(k["status"], cmp.STATUS_UNCHANGED)          # its own time did not change
        self.assertEqual(k["chain"][-1]["label"], "prepare")
        self.assertEqual(k["chain"][-1]["cause"], "increased_work")
        prepare = find(r["contributors"], "prepare")[0]
        self.assertEqual(prepare["cause"], "increased_work")
        self.assertEqual(r["contributors"][0]["label"], "prepare")
        # L, which bounded the iteration in the baseline, left the critical path
        lk = find(r["improvements"], "_Z1Lv")[0]
        self.assertEqual(lk["causeLabel"], "moved off the critical path")
        self.assertLess(lk["impactNs"], 0)

    def test_mpi_wait_propagates_to_the_sender_compute(self):
        r = cc.compare_traces(S.mpi_wait(False), S.mpi_wait(True))
        top = r["contributors"][0]
        self.assertEqual((top["label"], top["roleMap"]["rank"]), ("compute", "rank0"))
        self.assertEqual(top["cause"], "increased_work")
        self.assertAlmostEqual(top["impactNs"], 50 * MS, delta=MS / 10)   # capped at its own +50 ms
        recv = find(r["propagated"], "MPI_Recv", rank="rank1")
        self.assertEqual(len(recv), 1)
        recv = recv[0]
        self.assertEqual(recv["cause"], "communication")
        self.assertEqual(recv["status"], cmp.STATUS_REGRESSED)
        self.assertEqual([c["label"] for c in recv["chain"]], ["MPI_Send", "compute"])
        self.assertEqual(recv["chain"][-1]["roles"].split(", ")[1], "rank0")
        self.assertEqual(recv["propagatedFrom"], top["id"])
        # rank 1's own compute did not change
        self.assertFalse(find(r["contributors"] + r["offCriticalPath"], "compute", rank="rank1"))

    def test_critical_path_views_and_decomposition(self):
        r = cc.compare_traces(S.becomes_critical(False), S.becomes_critical(True))
        cpv = r["criticalPath"]
        before = {c["label"] for c in cpv["baseline"]["composition"]}
        after = {c["label"] for c in cpv["candidate"]["composition"]}
        self.assertIn("_Z1Lv", before)
        self.assertIn("_Z1Kv", after)
        self.assertNotIn("_Z1Kv", before)
        self.assertTrue(cpv["candidate"]["segments"])
        d = r["criticalPathChange"]
        self.assertEqual(d["deltaNs"], d["contributorsNs"] + d["offPathNs"] + d["otherNs"])
        pair = r["phases"][1]
        self.assertTrue(pair["topContributors"])


class TestFallbackAndOutputs(unittest.TestCase):
    def test_unrelated_runs_fall_back_to_the_aggregate_comparison(self):
        r = cc.compare_traces(S.iterations(10), S.same_name_different_paths(False))
        al = r["alignment"]
        self.assertEqual(al["method"], "aggregate")
        self.assertLess(al["confidence"], cc.DEFAULT_MIN_CONFIDENCE)
        self.assertIn("aggregate", al["fallbackReason"])
        self.assertEqual(r["contributors"], [])
        self.assertTrue(r["aggregate"]["aggregates"])
        self.assertIn("aggregate", cc.render_text(r))

    def test_whole_run_when_no_phases(self):
        def one(d):
            t = Trace(TraceMetadata(command="./once"))
            t.add(SpanEvent("main_work", Category.OPENMP, MS, d, 1, 1, {"type": "work"}))
            t.add(SpanEvent("io", Category.OPENMP, MS + d + 10, 2 * MS, 1, 1, {"type": "work"}))
            return t
        r = cc.compare_traces(one(10 * MS), one(20 * MS))
        self.assertIn(r["alignment"]["method"], ("phase-aligned", "whole-run"))
        self.assertEqual(r["contributors"][0]["label"], "main_work")

    def test_noise_floor_is_disclosed_and_respected(self):
        r = cc.compare_traces(S.same_name_different_paths(False), S.same_name_different_paths(True),
                              noise_pct=500.0)
        self.assertEqual(r["contributors"], [])
        self.assertIn("not a statistical significance test", r["noiseFloor"]["note"])
        self.assertTrue(any(u["conclusion"] == "statistical significance" for u in r["unavailable"]))

    def test_unavailable_conclusions_are_reported(self):
        r = cc.compare_traces(S.reordered(False), S.reordered(True))
        concl = {u["conclusion"] for u in r["unavailable"]}
        self.assertIn("source locations", concl)
        r = cc.compare_traces(S.stream_overlap(False), S.stream_overlap(True))
        self.assertNotIn("queueing", {u["conclusion"] for u in r["unavailable"]})   # CUPTI timing present

    def test_report_is_json_serializable_and_text_renders(self):
        r = cc.compare_traces(S.mpi_wait(False), S.mpi_wait(True))
        s = json.dumps(r)
        back = json.loads(s)
        self.assertEqual(back["schema"], cc.COMPARE_SCHEMA)
        for key in ("baseline", "candidate", "wallTime", "alignment", "contributors", "propagated",
                    "phases", "criticalPath", "unavailable", "aggregate", "noiseFloor"):
            self.assertIn(key, back)
        txt = cc.render_text(r)
        self.assertIn("Ranked causal contributors", txt)
        self.assertIn("compute", txt)
        self.assertIn("MPI_Recv", txt)

    def test_values_are_labeled_by_how_they_were_obtained(self):
        r = cc.compare_traces(S.mpi_wait(False), S.mpi_wait(True))
        self.assertEqual(r["wallTime"]["kind"], "measured")
        self.assertEqual(r["alignment"]["kind"], "heuristic")
        self.assertEqual(r["criticalPath"]["baseline"]["kind"], "derived")
        self.assertEqual(r["contributors"][0]["impactKind"], "derived")
        self.assertIn("measured", r["contributors"][0]["explanation"])


class TestStoresAndNames(unittest.TestCase):
    def test_disk_and_memory_stores_give_the_same_comparison(self):
        tmp = tempfile.mkdtemp(prefix="hprofiler_cmp_")
        try:
            def to_disk(t, name):
                d = Trace(copy.deepcopy(t.metadata), store=DiskTraceStore(os.path.join(tmp, name)))
                for s in t.iter_spans():
                    d.add(copy.deepcopy(s))
                d.finalize()
                return d
            a, b = S.mpi_wait(False), S.mpi_wait(True)
            rm = cc.compare_traces(a, b)
            da, db = to_disk(a, "a.hpstore"), to_disk(b, "b.hpstore")
            rd = cc.compare_traces(da, db)
            strip = lambda r: {k: v for k, v in r.items() if k != "aggregate"}  # noqa: E731
            self.assertEqual(json.dumps(strip(rm), sort_keys=True), json.dumps(strip(rd), sort_keys=True))
            da.close()
            db.close()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_name_normalization(self):
        self.assertEqual(normalize_name("foo [clone .constprop.0]"), "foo")
        self.assertEqual(normalize_name("cb@0x7f12aa001234"), "cb@0x…")


def _scenarios() -> dict:
    return {"paths": (S.same_name_different_paths(False), S.same_name_different_paths(True)),
            "mpi": (S.mpi_wait(False), S.mpi_wait(True)),
            "overlap": (S.stream_overlap(False), S.stream_overlap(True)),
            "critical": (S.becomes_critical(False), S.becomes_critical(True)),
            "insert": (S.iterations(10), S.iterations(11)),
            "reorder": (S.reordered(False), S.reordered(True))}


def _canon(report: dict) -> str:
    return json.dumps({k: v for k, v in report.items() if k != "aggregate"}, sort_keys=True)


_HASH_SCRIPT = r"""
import hashlib, json, sys
sys.path.insert(0, sys.argv[1])
from tests.test_causal_compare import _scenarios, _canon
from src.analysis import causal_compare as cc
print(json.dumps({n: hashlib.sha1(_canon(cc.compare_traces(a, b)).encode()).hexdigest()
                  for n, (a, b) in _scenarios().items()}, sort_keys=True))
"""


class TestInvariance(unittest.TestCase):
    """The comparison must not depend on dict/set iteration order (hash
    seed), on how the traces were loaded (in memory, JSON re-import, disk
    import), or on whether dependency edges were already persisted."""

    def test_independent_of_hash_seed(self):
        outs = set()
        for seed in ("0", "1", "4242", "987654"):
            p = subprocess.run([sys.executable, "-c", _HASH_SCRIPT, str(REPO)], capture_output=True, text=True,
                               timeout=300, env={**os.environ, "PYTHONHASHSEED": seed}, cwd=str(REPO))
            self.assertEqual(p.returncode, 0, p.stderr[-2000:])
            outs.add(p.stdout.strip())
        self.assertEqual(len(outs), 1, outs)

    def test_independent_of_loading_mode(self):
        from src.core import trace_io
        from src.output import chrome_trace
        tmp = tempfile.mkdtemp(prefix="hprofiler_cmp_load_")
        try:
            for name, (a, b) in _scenarios().items():
                ref = _canon(cc.compare_traces(a, b))
                paths = []
                for side, t in (("a", a), ("b", b)):
                    pth = os.path.join(tmp, f"{name}_{side}.json")
                    chrome_trace.write(t, pth)
                    paths.append(pth)
                mem = [chrome_trace.load_trace_from_json(x) for x in paths]
                self.assertEqual(_canon(cc.compare_traces(*mem)), ref, f"{name}: JSON reload")
                disk = [trace_io.open_trace(x, disk=True) for x in paths]
                self.assertEqual(_canon(cc.compare_traces(*disk)), ref, f"{name}: disk import")
                for t in disk:
                    t.close()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_independent_of_persisted_edges(self):
        from src.analysis import criticalpath as cp
        tmp = tempfile.mkdtemp(prefix="hprofiler_cmp_edges_")
        try:
            def to_disk(t, name):
                d = Trace(copy.deepcopy(t.metadata), store=DiskTraceStore(os.path.join(tmp, name)))
                for sp in t.iter_spans():
                    d.add(copy.deepcopy(sp))
                d.finalize()
                return d
            for name, (a, b) in _scenarios().items():
                da, db = to_disk(a, f"{name}_a.hpstore"), to_disk(b, f"{name}_b.hpstore")
                self.assertIsNone(da.store.load_edges(cp.EDGES_VERSION))
                first = _canon(cc.compare_traces(da, db))          # builds + persists edges
                self.assertIsNotNone(da.store.load_edges(cp.EDGES_VERSION), name)
                second = _canon(cc.compare_traces(da, db))         # reads persisted edges
                self.assertEqual(first, second, name)
                self.assertEqual(first, _canon(cc.compare_traces(a, b)), f"{name}: memory store")
                da.close()
                db.close()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_swapping_baseline_and_candidate_mirrors_the_attribution(self):
        # A slower rank-0 compute makes rank 1 wait in MPI_Recv. Forward the
        # wait's critical-path increase is credited to the compute; reversed,
        # the shrunken wait must likewise be credited to the faster compute
        # (origins are traced for improvements as well as regressions).
        a, b = S.mpi_wait(False), S.mpi_wait(True)
        fwd, rev = cc.compare_traces(a, b), cc.compare_traces(b, a)
        self.assertEqual(fwd["wallTime"]["deltaNs"], -rev["wallTime"]["deltaNs"])
        self.assertEqual(fwd["criticalPathChange"]["deltaNs"], -rev["criticalPathChange"]["deltaNs"])
        imp = lambda rows: {r["label"]: (r["impactNs"], r["propagatedFrom"] is not None) for r in rows}  # noqa: E731
        f, r = imp(fwd["contributors"]), imp(rev["improvements"])
        self.assertEqual(set(f), set(r))
        for label in f:
            self.assertEqual(f[label][0], -r[label][0], label)
            self.assertEqual(f[label][1], r[label][1], label)
        self.assertEqual(f["compute"], (50_000_000, False))
        recv = next(x for x in rev["improvements"] if x["label"] == "MPI_Recv")
        self.assertIn("origin: compute less work (-50.00ms", recv["explanation"])
        for name, (x, y) in _scenarios().items():
            p, q = cc.compare_traces(x, y), cc.compare_traces(y, x)
            self.assertEqual(p["wallTime"]["deltaNs"], -q["wallTime"]["deltaNs"], name)


class TestCli(unittest.TestCase):
    def test_compare_command_text_and_json(self):
        from src.output import chrome_trace
        tmp = tempfile.mkdtemp(prefix="hprofiler_cmpcli_")
        try:
            a, b = os.path.join(tmp, "before.json"), os.path.join(tmp, "after.json")
            chrome_trace.write(S.mpi_wait(False), a)
            chrome_trace.write(S.mpi_wait(True), b)
            exe = [sys.executable, str(Path(__file__).resolve().parent.parent / "hprofiler"), "compare"]
            p = subprocess.run(exe + [a, b], capture_output=True, text=True, timeout=120)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertIn("Ranked causal contributors", p.stdout)
            out = os.path.join(tmp, "r.json")
            p = subprocess.run(exe + [a, b, "--format", "json", "-o", out], capture_output=True, text=True,
                               timeout=120)
            self.assertEqual(p.returncode, 0, p.stderr)
            rep = json.loads(Path(out).read_text())
            self.assertEqual(rep["contributors"][0]["label"], "compute")
            p = subprocess.run(exe + [a, b, "--format", "json"], capture_output=True, text=True, timeout=120)
            self.assertEqual(json.loads(p.stdout)["schema"], cc.COMPARE_SCHEMA)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
