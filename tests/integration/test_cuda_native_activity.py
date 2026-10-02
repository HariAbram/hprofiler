"""
Native CUDA device-activity tracing on a real GPU (CUPTI), end to end:
hook -> socket -> Runner -> gpu_activity.assemble -> analysis, for
tests/fixtures/cuda_streams.cu (two streams, async copies/memsets, a
cross-stream event wait, event/stream/device syncs) in all three
HPROFILER_DEVICE_ACTIVITY modes.

These are FUNCTIONAL checks -- correlation, de-duplication, stream
attribution, ordering invariants the hardware guarantees, critical-path
edges -- not timing-accuracy measurements against a vendor profiler.
Skips when nvcc, a working CUDA device or CUPTI is unavailable.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from src.core import gpu_activity as ga
from src.core.runner import Runner
from src.analysis import criticalpath as cp
from src.output import chrome_trace

ROUNDS = 3


def _build_and_check(tmp: str, cudart: str = "shared") -> str | None:
    if not shutil.which("nvcc") or not (REPO / "build" / "lib" / "libhprofiler_cuda.so").exists():
        return None
    exe = os.path.join(tmp, f"cuda_streams_{cudart}")
    try:
        if subprocess.run(["nvcc", "-O2", "-cudart", cudart, "-o", exe,
                           str(REPO / "tests" / "fixtures" / "cuda_streams.cu")],
                          capture_output=True, timeout=300).returncode != 0:
            return None
        if subprocess.run([exe], capture_output=True, timeout=120).returncode != 0:
            return None          # no usable device
    except (OSError, subprocess.TimeoutExpired):
        return None
    return exe


class _CudaRuns(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="hprofiler_cuda_native_")
        cls.exe = _build_and_check(cls.tmp)
        cls.traces = {}
        if cls.exe:
            for mode in ("auto", "off", "both"):
                cls.traces[mode] = Runner(command=[cls.exe], backends=["cuda"],
                                          env_extra={"HPROFILER_DEVICE_ACTIVITY": mode}).run()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def entry(self, mode):
        da = self.traces[mode].metadata.device_activity
        keys = [k for k in da if k.endswith("/cuda")]
        self.assertEqual(len(keys), 1, da)
        return da[keys[0]]

    def setUp(self):
        if not self.exe:
            self.skipTest("nvcc / CUDA device unavailable")
        if self.entry("auto").get("status") != "active":
            self.skipTest(f"CUPTI not active: {self.entry('auto')}")


class TestAutoMode(_CudaRuns):
    def test_every_submission_has_its_measured_device_work(self):
        t = self.traces["auto"]
        e = self.entry("auto")
        self.assertEqual(e["timing"], "device")
        self.assertEqual(e.get("unmatched_device", 0), 0)
        self.assertEqual(e.get("host_without_device", 0), 0)
        self.assertNotIn("proxy_spans", e)
        work = [s for s in t.spans if ga.is_device_span(s) and ga.op_of(s) in ga.DEVICE_WORK_OPS]
        # per round: 3 kernels, 2 async copies, 1 memset; plus the final blocking copy
        self.assertEqual(len(work), ROUNDS * 6 + 1)
        self.assertTrue(all(ga.timing_source(s) == "device" for s in work))
        corr = ga.correlate(t.spans)
        spans = t.spans
        for i, s in enumerate(spans):
            if ga.is_device_span(s) and ga.op_of(s) in ga.DEVICE_WORK_OPS:
                h = corr.host_of[i]
                self.assertLessEqual(spans[h].start_ns, s.start_ns + ga.CLOCK_TOLERANCE_NS)
                self.assertEqual(s.parent_span_id, spans[h].span_id)
                self.assertGreaterEqual(int(s.tags["queue_ns"]), 0)

    def test_streams_are_program_streams_and_execute_in_order(self):
        t = self.traces["auto"]
        hosts = {s.tags["stream"] for s in t.spans
                 if ga.is_host_submission(s) and s.name in ("cudaMemcpyAsync", "cudaMemsetAsync")}
        self.assertEqual(len(hosts), 2)
        by_stream = defaultdict(list)
        for s in t.spans:
            if ga.is_device_span(s) and ga.op_of(s) in ga.DEVICE_WORK_OPS and s.name != "memcpy DtoH" or \
                    (s.name == "memcpy DtoH" and s.tags.get("async") == "1"):
                by_stream[s.tags["stream"]].append(s)
        self.assertEqual(set(by_stream), hosts)
        for lst in by_stream.values():
            lst.sort(key=lambda s: s.start_ns)
            for a, b in zip(lst, lst[1:]):
                self.assertLessEqual(a.end_ns, b.start_ns, (a.name, b.name))

    def test_cross_stream_wait_is_respected_and_becomes_an_edge(self):
        t = self.traces["auto"]
        s1 = next(s.tags["stream"] for s in t.spans if s.name == "memcpy HtoD")
        s2 = next(s.tags["stream"] for s in t.spans if s.name == "memset")
        kernels = sorted((s for s in t.spans if ga.is_device_kernel(s)), key=lambda s: s.start_ns)
        producers = [k for k in kernels if k.tags["stream"] == s1]
        on_s2 = [k for k in kernels if k.tags["stream"] == s2]
        consumers = on_s2[1::2]
        self.assertEqual((len(producers), len(consumers)), (ROUNDS, ROUNDS))
        for p, c in zip(producers, consumers):
            self.assertGreaterEqual(c.start_ns, p.end_ns)
        spans, preds = cp.build_dependency_graph(t)
        index = {id(s): i for i, s in enumerate(spans)}
        for p, c in zip(producers, consumers):
            self.assertIn((index[id(p)], "sequential", "certain"), preds[index[id(c)]])
        for sync in (s for s in spans if s.name == "cudaEventSynchronize"):
            waited = {spans[u].tags.get("stream") for u, k, _ in preds[index[id(sync)]] if k == "device_wait"}
            self.assertEqual(waited, {s1})

    def test_gpu_active_and_critical_path(self):
        t = self.traces["auto"]
        ka = ga.kernel_activity(t.spans)
        self.assertEqual((ka.source, ka.used, ka.excluded), ("device", ROUNDS * 3, 0))
        report = cp.analyze(t)
        self.assertGreater(len(report.path_span_indices), 1)
        self.assertLessEqual(report.total_path_ns, report.wall_ns * 1.001)
        self.assertIn("launch", report.path_edge_kinds)

    def test_round_trip(self):
        t = self.traces["auto"]
        path = os.path.join(self.tmp, "auto.json")
        chrome_trace.write(t, path)
        r = chrome_trace.load_trace_from_json(path)
        key = lambda s: (s.name, s.start_ns, s.parent_span_id, s.tags.get("stream"))
        self.assertEqual(sorted(map(key, r.spans)), sorted(map(key, t.spans)))
        self.assertEqual(r.metadata.device_activity, t.metadata.device_activity)


class TestProxyAndBothModes(_CudaRuns):
    def test_off_mode_is_labelled_proxy(self):
        t = self.traces["off"]
        e = self.entry("off")
        self.assertEqual((e["status"], e["timing"]), ("disabled", "proxy"))
        dev = [s for s in t.spans if ga.is_device_span(s)]
        self.assertEqual(len(dev), ROUNDS * 6)     # the blocking copy has no proxy span
        self.assertTrue(all(ga.timing_source(s) in ga.PROXY_TIMINGS for s in dev))
        self.assertEqual(ga.kernel_activity(t.spans).source, "proxy")
        self.assertNotIn("host_without_device", e)

    def test_both_mode_keeps_exactly_the_measured_span(self):
        t = self.traces["both"]
        e = self.entry("both")
        self.assertEqual(e["deduplicated_proxy"], ROUNDS * 6)
        self.assertEqual(e.get("unmatched_device", 0), 0)
        self.assertGreater(e.get("internal_records", 0), 0)   # hook's own event syncs, dropped
        dev = [s for s in t.spans if ga.is_device_span(s) and ga.op_of(s) in ga.DEVICE_WORK_OPS]
        self.assertEqual(len(dev), ROUNDS * 6 + 1)
        self.assertTrue(all(ga.timing_source(s) == "device" for s in dev))


class TestStaticRuntime(_CudaRuns):
    """A statically linked CUDA runtime cannot be intercepted with
    LD_PRELOAD. The Runner detects it and starts CUPTI when the hook loads
    (HPROFILER_CUPTI_EAGER), so callbacks report the host calls."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.static_exe = _build_and_check(cls.tmp, "static") if cls.exe else None
        cls.static = Runner(command=[cls.static_exe], backends=["cuda"]).run() if cls.static_exe else None

    def test_static_program_is_fully_correlated(self):
        if self.static is None:
            self.skipTest("could not build/run a static-runtime fixture")
        t = self.static
        e = next(v for k, v in t.metadata.device_activity.items() if k.endswith("/cuda"))
        self.assertEqual((e["status"], e["timing"]), ("active", "device"))
        self.assertEqual(e.get("unmatched_device", 0), 0)
        hosts = [s for s in t.spans if ga.is_host_submission(s)]
        self.assertTrue(hosts and all(s.tags["src"] == "cupti_cb" for s in hosts))
        work = [s for s in t.spans if ga.is_device_span(s) and ga.op_of(s) in ga.DEVICE_WORK_OPS]
        self.assertEqual(len(work), ROUNDS * 6 + 1)
        self.assertTrue(all(s.parent_span_id for s in work))
        report = cp.analyze(t)
        self.assertLessEqual(report.total_path_ns, report.wall_ns * 1.001)


if __name__ == "__main__":
    unittest.main()
