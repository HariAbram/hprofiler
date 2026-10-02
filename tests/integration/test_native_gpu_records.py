"""
Record-level tests for the native GPU tracers' decoders, compiled against
the REAL vendor headers: tests/native/cupti_records_test.c (CUPTI structs)
and tests/native/rocprof_records_test.c (ROCprofiler-SDK structs) feed
synthetic activity records through hooks/cuda_hook/cupti_trace.c and
hooks/rocm_hook/rocprof_trace.c exactly as the buffer callbacks do, and
print the wire lines produced. This test parses those lines with the
Runner's own parser and feeds them through src/core/gpu_activity.py.

No GPU is needed. Skips when a compiler or the headers are missing.
Header locations: HPROFILER_CUPTI_INCLUDE / HPROFILER_ROCPROFILER_SDK_INCLUDE,
else the build's CMake cache, else the usual install paths.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from src.core import gpu_activity as ga
from src.core.runner import _parse_record
from src.core.trace import Trace


def _cmake_cache(var: str) -> str | None:
    cache = REPO / "build" / "CMakeCache.txt"
    if not cache.exists():
        return None
    for line in cache.read_text(errors="replace").splitlines():
        if line.startswith(var + ":"):
            value = line.split("=", 1)[1]
            return value if value and not value.endswith("NOTFOUND") else None
    return None


def _find_include(env: str, cache_var: str, probe: str, defaults: list[str]) -> str | None:
    for cand in [os.environ.get(env), _cmake_cache(cache_var), *defaults]:
        if cand and (Path(cand) / probe).exists():
            return cand
    return None


def _compile_and_run(sources: list[str], flag_sets: list[list[str]], tmp: str) -> str | None:
    if not shutil.which("gcc"):
        return None
    exe = os.path.join(tmp, "harness")
    for flags in flag_sets:
        r = subprocess.run(["gcc", "-std=gnu11", "-O1", *flags, *sources, "-ldl", "-lpthread", "-o", exe],
                           capture_output=True, text=True)
        if r.returncode == 0:
            out = subprocess.run([exe], capture_output=True, text=True, timeout=60)
            return out.stdout if out.returncode == 0 else None
    return None


def _parse(output: str) -> tuple[list, list]:
    spans, status = [], []
    for line in output.splitlines():
        if line.startswith("gpuact:"):
            status.append(ga.parse_status_line(line))
        elif line.startswith("span:"):
            ev = _parse_record(line)
            assert ev is not None, line
            spans.append(ev)
    return spans, status


class TestCuptiDecoder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = None
        inc = _find_include("HPROFILER_CUPTI_INCLUDE", "CUPTI_INCLUDE_DIR", "cupti.h",
                            ["/usr/local/cuda/include", "/usr/local/cuda/extras/CUPTI/include", "/usr/include"])
        if inc is None:
            return
        with tempfile.TemporaryDirectory() as tmp:
            base = ["-DHP_HAVE_CUPTI", "-DHP_CUPTI_UNIT_TEST", f"-I{inc}", f"-I{REPO}/hooks/cuda_hook"]
            cls.out = _compile_and_run(
                [str(REPO / "tests/native/cupti_records_test.c"), str(REPO / "hooks/cuda_hook/cupti_trace.c")],
                [base + ["-DHP_CUPTI_MEMCPY6", "-DHP_CUPTI_SYNC2"], base], tmp)
        if cls.out is not None:
            cls.spans, cls.status = _parse(cls.out)
            cls.by_name = {}
            for s in cls.spans:
                cls.by_name.setdefault(s.name, []).append(s)

    def setUp(self):
        if self.out is None:
            self.skipTest("gcc or CUPTI headers unavailable")

    def test_concurrent_kernels(self):
        big, small = self.by_name["_Z4axpyPf"][0], self.by_name["_Z4scalePf"][0]
        self.assertEqual((big.start_ns, big.duration_ns), (1_000_000, 500_000))
        self.assertEqual((small.start_ns, small.duration_ns), (1_200_000, 700_000))
        self.assertEqual(big.tags["nstream"], "7")
        self.assertEqual(small.tags["nstream"], "8")
        for k in (big, small):
            self.assertEqual((k.tags["side"], k.tags["timing"], k.tags["src"], k.tags["rt"]),
                             ("gpu", "device", "cupti", "cuda"))
            self.assertEqual(k.tags["grid"], "64x1x1")
        self.assertEqual(big.tags["corr"], "101")

    def test_latency_timestamps_only_when_consistent(self):
        k103, k104 = self.by_name["_Z4axpyPf"][1:3]
        self.assertEqual((k103.tags["queued"], k103.tags["submitted"]), ("1900000", "1950000"))
        self.assertNotIn("submitted", k104.tags)
        self.assertEqual(self.by_name["_Z9graphnodev"][0].tags["graph"], "77")

    def test_async_copy_carries_driver_and_runtime_ids(self):
        m = self.by_name["memcpy HtoD"][0]
        self.assertEqual((m.tags["corr"], m.tags["corr2"]), ("202", "201"))
        self.assertEqual((m.tags["bytes"], m.tags["dir"], m.tags["async"]), ("1048576", "HtoD", "1"))
        self.assertEqual(m.category.value, "memory")
        self.assertEqual(self.by_name["memset"][0].tags["bytes"], "4096")
        p2p = self.by_name["memcpy PtoP"][0]
        self.assertEqual((p2p.tags["src_dev"], p2p.tags["dst_dev"]), ("0", "1"))

    def test_sync_records(self):
        stream_sync = self.by_name["sync stream"][0]
        ctx_sync = self.by_name["sync context"][0]
        self.assertEqual((stream_sync.tags["timing"], stream_sync.tags["nstream"]), ("host", "7"))
        self.assertNotIn("nstream", ctx_sync.tags)   # CUPTI_SYNCHRONIZATION_INVALID_VALUE

    def test_incomplete_corrupt_and_dropped_records(self):
        self.assertNotIn("_Z10unfinishedv", self.by_name)
        self.assertNotIn("_Z7corruptv", self.by_name)
        trace = Trace()
        for pid, api, fields in self.status:
            ga.record_status(trace, pid, api, fields)
        e = trace.metadata.device_activity["4242/cuda"]
        self.assertEqual((e["dropped"], e["notime"], e["bad_records"], e["internal_records"]),
                         (42, 1, 1, 1))
        self.assertNotIn("sync event", self.by_name)   # the hook's own event sync
        self.assertIn("pcrecord:30", self.out)   # routed to PC sampling

    def test_offset_clock_mapping(self):
        self.assertEqual(self.by_name["_Z7shiftedv"][0].start_ns, 9_000_000 - 1_000)

    def test_callback_correlation_capture(self):
        captured = [l for l in self.out.splitlines() if l.startswith("captured:")]
        self.assertEqual(captured, ["captured:600:601", "captured:610:0"])

    def test_callback_spans_for_unintercepted_submissions(self):
        cb = [s for s in self.spans if s.tags.get("src") == "cupti_cb"]
        self.assertEqual([(s.name, s.tags["type"], s.tags["corr"]) for s in cb], [
            ("cudaLaunchKernel", "launch", "700"),     # nested cuLaunchKernel (701) not repeated
            ("cudaMemcpyAsync", "launch", "710"),
            ("cudaMemcpy", "memcpy", "711"),           # blocking: the call is the transfer
            ("cuLaunchKernel", "launch", "740"),
            ("cudaStreamSynchronize", "sync", "750"),   # static-runtime mode only
            ("cudaStreamWaitEvent", "stream_wait", "751"),
        ])
        self.assertEqual(cb[4].tags["sync"], "stream")
        self.assertTrue(all(s.tags["side"] == "cpu" and s.tags["timing"] == "host" for s in cb))

    def test_end_to_end_correlation_with_hook_host_spans(self):
        """Decoder output + the host spans cuda_hook.c would have emitted."""
        trace = Trace()
        hosts = [
            "span:cuda:4242:7:999000:20000:cudaLaunchKernel:type=launch,op=kernel,stream=11,"
            "side=cpu,rt=cuda,timing=host,lid=1,sid=1,corr=101",
            "span:cuda:4242:7:1100000:20000:cudaLaunchKernel:type=launch,op=kernel,stream=22,"
            "side=cpu,rt=cuda,timing=host,lid=2,sid=2,corr=102",
            "span:memory:4242:7:2900000:30000:cudaMemcpyAsync:type=launch,op=memcpy,stream=33,"
            "side=cpu,rt=cuda,timing=host,lid=3,sid=3,corr=201,corr2=202",
        ]
        for ev in [_parse_record(h) for h in hosts] + self.spans:
            trace.add(ev)
        ga.assemble(trace)
        names = {s.name: s for s in trace.spans}
        self.assertEqual(names["_Z4scalePf"].tags["stream"], "22")
        self.assertEqual(names["memcpy HtoD"].tags["stream"], "33")
        self.assertEqual(names["memcpy HtoD"].tags["queue_ns"], str(3_000_000 - 2_930_000))
        # kernels on CUPTI stream 7 after the first launch inherit stream 11
        self.assertEqual(self.by_name["_Z4axpyPf"][0].tags["stream"], "11")
        ka = ga.kernel_activity(trace.spans)
        self.assertEqual(ka.source, "device")


class TestRocprofilerDecoder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out = None
        roots = [f"{os.environ.get('ROCM_PATH', '/opt/rocm')}/include", "/opt/rocm/include"]
        inc = _find_include("HPROFILER_ROCPROFILER_SDK_INCLUDE", "ROCPROFILER_SDK_INCLUDE_DIR",
                            "rocprofiler-sdk/rocprofiler.h", roots)
        if inc is None:
            return
        with tempfile.TemporaryDirectory() as tmp:
            cls.out = _compile_and_run(
                [str(REPO / "tests/native/rocprof_records_test.c"), str(REPO / "hooks/rocm_hook/rocprof_trace.c")],
                [["-D__HIP_PLATFORM_AMD__", "-DHP_HAVE_ROCPROFILER_SDK", "-DHP_ROCPROF_UNIT_TEST",
                  f"-I{inc}", f"-I{REPO}/hooks/rocm_hook"]], tmp)
        if cls.out is not None:
            cls.spans, cls.status = _parse(cls.out)
            cls.by_name = {s.name: s for s in cls.spans}

    def setUp(self):
        if self.out is None:
            self.skipTest("gcc or ROCprofiler-SDK headers unavailable "
                          "(set HPROFILER_ROCPROFILER_SDK_INCLUDE=<rocm>/include)")

    def test_dispatches_with_names_queues_and_external_ids(self):
        a, b = self.by_name["_Z4axpyPf"], self.by_name["_Z5scalePf"]   # ".kd" stripped
        self.assertEqual((a.start_ns, a.duration_ns, a.tid), (1_000_500, 500_000, 900))
        self.assertEqual((a.tags["lid"], a.tags["corr"], a.tags["queue"]), ("1", "20", str(0xA0)))
        self.assertEqual(b.tags["queue"], str(0xB0))
        self.assertEqual(a.tags["grid"], "64x1x1")       # work-items / workgroup
        self.assertEqual(a.tags["block"], "256x1x1")
        unnamed = self.by_name["kernel_99"]
        self.assertNotIn("lid", unnamed.tags)

    def test_memory_copy(self):
        m = self.by_name["memcpy HtoD"]
        self.assertEqual((m.tags["bytes"], m.tags["lid"], m.tags["src_dev"], m.tags["dst_dev"]),
                         ("1048576", "3", "10", "11"))

    def test_truncated_unfinished_and_dropped(self):
        self.assertEqual(len(self.spans), 4)
        trace = Trace()
        for pid, api, fields in self.status:
            ga.record_status(trace, pid, api, fields)
        e = trace.metadata.device_activity["4242/rocm"]
        self.assertEqual((e["dropped"], e["notime"], e["bad_records"]), (15, 1, 1))

    def test_end_to_end_external_correlation(self):
        trace = Trace()
        for h in ["span:rocm:4242:900:990000:5000:hipLaunchKernel:type=launch,op=kernel,stream=33,"
                  "side=cpu,rt=rocm,timing=host,lid=1,sid=1",
                  "span:rocm:4242:900:1100000:5000:hipLaunchKernel:type=launch,op=kernel,stream=44,"
                  "side=cpu,rt=rocm,timing=host,lid=2,sid=2"]:
            trace.add(_parse_record(h))
        for s in self.spans:
            trace.add(s)
        ga.assemble(trace)
        names = {s.name: s for s in trace.spans}
        self.assertEqual(names["_Z4axpyPf"].tags["stream"], "33")
        self.assertEqual(names["_Z5scalePf"].tags["stream"], "44")
        self.assertEqual(names["kernel_99"].tags["stream"], "33")   # queue 0xA0 learned


if __name__ == "__main__":
    unittest.main()
