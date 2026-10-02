"""
Multi-million-event stress test for the disk-backed TraceStore.

Two subprocesses (so peak RSS is measured in a clean interpreter):

  capture   -- append N synthetic spans (+ counters and instants) through
               Trace.add() into a DiskTraceStore exactly like the runner
               does, then finalize() (indexes, lanes, aggregates, exclusive
               time, multiresolution activity index);
  explore   -- reopen the store cold, build the GUI TimelineModel when
               PySide6 is available, issue a mix of window queries from
               fully zoomed-out to a few microseconds, then the TUI
               timeline's lane windows, aggregate queries, the overview
               numbers and the call tree.

Asserts that peak RSS growth stays bounded and does NOT scale with N (a
small run is the baseline), that window queries stay interactive, and that
the answers are exact (span counts, per-name totals, window contents
checked against the deterministic generator). Default N is 2,000,000;
override with HPROFILER_STRESS_EVENTS (e.g. 5000000). Takes ~1 min at 2M.

    python3 -m unittest tests.integration.test_store_stress
"""
from __future__ import annotations

import json
import os
import resource
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
N_EVENTS = int(os.environ.get("HPROFILER_STRESS_EVENTS", "2000000"))
N_BASELINE = 200_000

# Generous bounds, so the test stays stable on a loaded machine. Holding
# 2M SpanEvent objects in Python alone costs well over 1 GB.
MAX_RSS_GROWTH_MB = 300          # absolute, any phase
MAX_RSS_SCALING_MB = 120         # N-event run minus the baseline run
MAX_WINDOW_MEDIAN_MS = 25
MAX_WINDOW_P95_MS = 150

PIDS, TIDS, NAMES = 4, 8, 200
CATS = ("openmp", "sync", "mpi", "cpu")


def generate(n: int):
    """Deterministic span stream: (i, name, cat, start, dur, pid, tid)."""
    clock: dict[tuple[int, int], int] = {}
    for i in range(n):
        pid, tid = 100 + i % PIDS, (i // PIDS) % TIDS
        c = clock.get((pid, tid), 1_000_000_000)
        d = 1_000 + (i * 7919) % 50_000
        if i % 97 == 0:
            d = 0                                     # sample-like markers
        yield i, f"fn{(i * 31) % NAMES}", CATS[i % 4], c, d, pid, tid
        clock[(pid, tid)] = c + max(d, 1) + (i * 104729) % 20_000


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def child_capture(path: str, n: int) -> dict:
    sys.path.insert(0, str(ROOT))
    from src.core.events import Category, CounterEvent, InstantEvent, SpanEvent
    from src.core.trace import TraceMetadata
    from src.core.trace_io import create_disk_trace
    base = _rss_mb()
    t = create_disk_trace(path, TraceMetadata(command="stress", start_time_ns=1_000_000_000))
    cats = {c: Category(c) for c in CATS}
    t0 = time.perf_counter()
    for i, name, cat, start, dur, pid, tid in generate(n):
        t.add(SpanEvent(name, cats[cat], start, dur, pid, tid, {"type": "work", "i": str(i % 1000)}))
        if i % 50_000 == 0:
            t.add(CounterEvent("ipc", Category.CPU, start, 1.5, "", pid))
            t.add(InstantEvent("mark", Category.CPU, start, pid, tid))
    t.store.flush()
    capture_s = time.perf_counter() - t0
    capture_rss = _rss_mb() - base
    t0 = time.perf_counter()
    t.finalize()
    finalize_s = time.perf_counter() - t0
    out = {"capture_s": capture_s, "capture_rss": capture_rss, "finalize_s": finalize_s,
           "finalize_rss": _rss_mb() - base, "spans": t.span_count(),
           "bytes": sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())}
    t.close()
    return out


def child_explore(path: str, n: int) -> dict:
    sys.path.insert(0, str(ROOT))
    from src.core.trace_io import open_trace
    base = _rss_mb()
    t0 = time.perf_counter()
    t = open_trace(path)
    st = t.store
    out: dict = {"open_s": time.perf_counter() - t0, "finalized": st.is_finalized(),
                 "spans": t.span_count()}

    # exactness: per-name totals and one thread's windows vs the generator
    agg = {r["name"]: (r["count"], r["total_ns"]) for r in t.aggregate_stats()}
    exp: dict[str, list[int]] = {}
    pick = (101, 3)
    probe_rows = []
    for i, name, _cat, start, dur, pid, tid in generate(n):
        v = exp.setdefault(name, [0, 0])
        v[0] += 1
        v[1] += dur
        if (pid, tid) == pick:
            probe_rows.append((start, dur))
    out["agg_exact"] = agg == {k: tuple(v) for k, v in exp.items()}
    lo, hi = probe_rows[0][0], probe_rows[-1][0]
    ok = True
    for k in range(5):
        a = lo + (hi - lo) * k // 5
        b = a + (hi - lo) // (10 ** (k + 1))
        got = [(s.start_ns, s.duration_ns) for s in t.iter_spans(order="start", pid=pick[0], tid=pick[1],
                                                                  window=(a, b))]
        want = [(s, d) for s, d in probe_rows if s <= b and s + d > a]
        ok &= got == want
    out["window_exact"] = ok
    del probe_rows, exp

    # interactive window queries, the way the Timeline issues them
    model = None
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtGui import QGuiApplication
        from src.gui.models import TimelineModel
        from src.gui.theme import Theme
        app = QGuiApplication.instance() or QGuiApplication([sys.argv[0]])
        q0 = time.perf_counter()
        model = TimelineModel(t, Theme(dark=True))
        out["model_s"] = time.perf_counter() - q0
    except ImportError:
        pass
    lanes = st.lane_infos()
    ext = st.span_extent()
    total = ext[1] - ext[0]
    times, modes = [], {"spans": 0, "bins": 0}
    for k in range(120):
        li = k % len(lanes)
        w = total / (10 ** (k % 7))
        a = ext[0] + (total - w) * ((k * 37) % 100) / 100
        q0 = time.perf_counter()
        if model is not None:
            view = model.laneView(li, a, a + w, 1600, 2000)
            modes[view["mode"]] += 1
        else:
            name = lanes[li].name
            if st.count_window(name, int(a), int(a + w), cap=2000) > 2000:
                st.occupancy(name, a, a + w, 1600)
                modes["bins"] += 1
            else:
                st.window(name, int(a), int(a + w))
                modes["spans"] += 1
        times.append((time.perf_counter() - q0) * 1000)
    times.sort()
    out.update(window_median_ms=statistics.median(times), window_p95_ms=times[int(0.95 * len(times))],
               window_max_ms=times[-1], modes=modes, gui=model is not None)
    q0 = time.perf_counter()
    st.exclusive_aggregate().bucket_totals()
    t.aggregate_stats()
    out["aggregates_ms"] = (time.perf_counter() - q0) * 1000

    # The TUI timeline (hprofiler view) on the same store: lane setup plus
    # a fully zoomed-out and a zoomed-in window per lane.
    try:
        from src.ui.app import TimelineWidget
    except ImportError:
        TimelineWidget = None
    if TimelineWidget is not None:
        q0 = time.perf_counter()
        w = TimelineWidget(t)
        out["tui_init_ms"] = (time.perf_counter() - q0) * 1000
        tui_modes = {"spans": 0, "bins": 0}
        q0 = time.perf_counter()
        for name in w._lane_names:
            for a, b in ((ext[0], ext[1]), (ext[0] + total // 2, ext[0] + total // 2 + total // 5000)):
                tui_modes[w._lane_columns(name, a, b, 160)[0]] += 1
        out["tui_windows_ms"] = (time.perf_counter() - q0) * 1000 / (2 * len(w._lane_names))
        out["tui_modes"] = tui_modes

    # Whole-trace views the GUI/TUI open with: overview numbers and the
    # call tree (one thread's spans in memory at a time).
    from src.gui.bridge import compute_dashboard_data
    from src.analysis.call_tree import build_call_tree
    q0 = time.perf_counter()
    compute_dashboard_data(t, True)
    out["overview_s"] = time.perf_counter() - q0
    q0 = time.perf_counter()
    out["call_tree_roots"] = len(build_call_tree(t))
    out["call_tree_s"] = time.perf_counter() - q0
    out["explore_rss"] = _rss_mb() - base
    t.close()
    return out


def run_child(phase: str, path: str, n: int) -> dict:
    proc = subprocess.run([sys.executable, __file__, "--child", phase, path, str(n)],
                          capture_output=True, text=True, cwd=str(ROOT), timeout=3600)
    if proc.returncode != 0:
        raise AssertionError(f"{phase} child failed:\n{proc.stdout}\n{proc.stderr}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


class TestStoreStress(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="hprofiler_stress_")
        cls.results = {}
        for n in (N_BASELINE, N_EVENTS):
            path = os.path.join(cls.tmp, f"s{n}.hpstore")
            cls.results[n] = {"capture": run_child("capture", path, n), "explore": run_child("explore", path, n)}
            shutil.rmtree(path, ignore_errors=True)
        r = cls.results[N_EVENTS]
        sys.stderr.write(f"\n[store stress] {N_EVENTS:,} spans: capture {r['capture']['capture_s']:.1f}s "
                         f"(+{r['capture']['capture_rss']:.0f} MB), finalize {r['capture']['finalize_s']:.1f}s "
                         f"(+{r['capture']['finalize_rss']:.0f} MB), store {r['capture']['bytes'] / 1e6:.0f} MB; "
                         f"reopen {r['explore']['open_s'] * 1000:.0f} ms, windows median "
                         f"{r['explore']['window_median_ms']:.1f} ms / p95 {r['explore']['window_p95_ms']:.1f} ms "
                         f"/ max {r['explore']['window_max_ms']:.1f} ms {r['explore']['modes']}; "
                         f"TUI timeline init {r['explore'].get('tui_init_ms', 0):.0f} ms, "
                         f"{r['explore'].get('tui_windows_ms', 0):.1f} ms/window; overview "
                         f"{r['explore']['overview_s']:.1f}s, call tree {r['explore']['call_tree_s']:.1f}s; "
                         f"explore +{r['explore']['explore_rss']:.0f} MB\n")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_all_events_stored_and_reopened_finalized(self):
        for n, r in self.results.items():
            self.assertEqual(r["capture"]["spans"], n)
            self.assertEqual(r["explore"]["spans"], n)
            self.assertTrue(r["explore"]["finalized"])

    def test_answers_are_exact(self):
        for r in self.results.values():
            self.assertTrue(r["explore"]["agg_exact"])
            self.assertTrue(r["explore"]["window_exact"])

    def test_peak_memory_is_bounded_and_does_not_scale_with_events(self):
        big, small = self.results[N_EVENTS], self.results[N_BASELINE]
        for phase, key in (("capture", "capture_rss"), ("capture", "finalize_rss"), ("explore", "explore_rss")):
            self.assertLess(big[phase][key], MAX_RSS_GROWTH_MB, key)
            self.assertLess(big[phase][key] - small[phase][key], MAX_RSS_SCALING_MB, key)

    def test_window_queries_are_responsive(self):
        r = self.results[N_EVENTS]["explore"]
        self.assertLess(r["window_median_ms"], MAX_WINDOW_MEDIAN_MS)
        self.assertLess(r["window_p95_ms"], MAX_WINDOW_P95_MS)
        self.assertGreater(r["modes"]["bins"], 0)       # zoomed out -> occupancy bins
        self.assertGreater(r["modes"]["spans"], 0)      # zoomed in -> exact spans

    def test_tui_timeline_opens_fast_and_switches_between_bins_and_spans(self):
        r = self.results[N_EVENTS]["explore"]
        if "tui_init_ms" not in r:
            self.skipTest("textual not installed")
        self.assertLess(r["tui_init_ms"], 5_000)
        self.assertLess(r["tui_windows_ms"], MAX_WINDOW_P95_MS)
        self.assertGreater(r["tui_modes"]["bins"], 0)
        self.assertGreater(r["tui_modes"]["spans"], 0)

    def test_call_tree_covers_every_thread(self):
        self.assertEqual(self.results[N_EVENTS]["explore"]["call_tree_roots"], PIDS * TIDS)


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--child":
        fn = {"capture": child_capture, "explore": child_explore}[sys.argv[2]]
        print(json.dumps(fn(sys.argv[3], int(sys.argv[4]))))
    else:
        unittest.main()
