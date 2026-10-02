"""
Host-submission / device-execution model (src/core/gpu_activity.py) and its
consumers, driven by synthetic native records in exactly the wire format
hooks/cuda_hook/cupti_trace.c and hooks/rocm_hook/rocprof_trace.c emit
(tests/integration/test_native_gpu_records.py checks those decoders against
real CUPTI / ROCprofiler-SDK structs). Every trace goes through the same
parser the Runner uses (runner._parse_record + gpuact status lines).

Covers: concurrent streams, asynchronous copies, missing correlations,
reused correlation ids, dropped records (buffer overflow), out-of-order
delivery, proxy/native de-duplication, never mixing proxy and device
intervals for GPU-active time, JSON round trip, lanes, activity buckets and
the critical-path edges (launch, stream order, stream/device/event sync,
cross-stream waits).
"""
from __future__ import annotations

import io
import os
import random
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core import gpu_activity as ga
from src.core.runner import _parse_record
from src.core.trace import Trace
from src.core.events import SpanEvent
from src.analysis import activity_buckets as ab
from src.analysis import criticalpath as cp
from src.analysis.cct import gpu_starvation
from src.output import chrome_trace

PID = 1
TID = 100


# ── synthetic wire records ────────────────────────────────────────────────────

def _line(cat, start, dur, name, tags, pid=PID, tid=TID):
    return f"span:{cat}:{pid}:{tid}:{start}:{dur}:{name}:" + ",".join(f"{k}={v}" for k, v in tags.items())


def host(api, start, dur, *, cat="cuda", typ="launch", op="kernel", lid, corr=None, corr2=None,
         stream=None, rt="cuda", pid=PID, tid=TID, **extra):
    """A hook host API span (cuda_hook.c / rocm_hook.c sub_host)."""
    tags = {"type": typ, "op": op}
    if stream is not None:
        tags["stream"] = stream
    tags.update(extra)
    tags.update({"side": "cpu", "rt": rt, "timing": "host", "lid": lid,
                 "sid": (pid << 32) | lid})
    if corr is not None:
        tags["corr"] = corr
    if corr2 is not None:
        tags["corr2"] = corr2
    return _line(cat, start, dur, api, tags, pid, tid)


def cupti_kernel(corr, nstream, start, end, name="_Z4axpyPf", pid=PID, ctx=1, **extra):
    """cupti_trace.c emit_kernel()."""
    tags = {"type": "kernel", "side": "gpu", "rt": "cuda", "op": "kernel", "timing": "device",
            "src": "cupti", "corr": corr, "dev": 0, "ctx": ctx, "nstream": nstream,
            "grid": "64x1x1", "block": "256x1x1", **extra}
    return _line("cuda", start, end - start, name, tags, pid, 0)


def cupti_memcpy(corr, corr2, nstream, start, end, dir_="HtoD", nbytes=1 << 20, pid=PID):
    """cupti_trace.c emit_memcpy(): corr = driver id, corr2 = runtime id."""
    tags = {"type": "memcpy", "side": "gpu", "rt": "cuda", "op": "memcpy", "timing": "device",
            "src": "cupti", "dir": dir_, "bytes": nbytes, "corr": corr, "dev": 0, "ctx": 1,
            "nstream": nstream, "corr2": corr2, "async": 1}
    return _line("memory", start, end - start, f"memcpy {dir_}", tags, pid, 0)


def cupti_sync(corr, kind, start, end, nstream=None, pid=PID):
    tags = {"type": "sync_wait", "side": "gpu", "rt": "cuda", "op": "sync", "timing": "host",
            "src": "cupti", "sync": kind, "corr": corr, "ctx": 1}
    if nstream is not None:
        tags["nstream"] = nstream
    return _line("sync", start, end - start, f"sync {kind}", tags, pid, 0)


def proxy_kernel(lid, stream, start, dur, name="_Z4axpyPf", timing="proxy_event", corr=None,
                 pid=PID, rt="cuda"):
    """cuda_hook.c sub_proxy() + pk_flush()."""
    tags = {"type": "kernel", "op": "kernel", "stream": stream, "grid": "64x1x1",
            "block": "256x1x1", "side": "gpu", "rt": rt, "lid": lid}
    if corr is not None:
        tags["corr"] = corr
    tags["timing"] = timing
    return _line(rt, start, dur, name, tags, pid, TID)


def roc_dispatch(corr, lid, queue, start, end, name="_Z4axpyPf", pid=PID, tid=TID):
    """rocprof_trace.c emit_dispatch()."""
    tags = {"type": "kernel", "side": "gpu", "rt": "rocm", "op": "kernel", "timing": "device",
            "src": "rocprofiler", "corr": corr, "dev": 11, "queue": queue, "dispatch": corr + 1000,
            "grid": "64x1x1", "block": "256x1x1"}
    if lid is not None:
        tags["lid"] = lid
    return _line("rocm", start, end - start, name, tags, pid, tid)


def build(lines, assemble=True):
    trace = Trace()
    for ln in lines:
        if ln.startswith("gpuact:"):
            ga.record_status(trace, *ga.parse_status_line(ln))
            continue
        ev = _parse_record(ln)
        assert ev is not None, ln
        trace.add(ev)
    if assemble:
        ga.assemble(trace)
    return trace


def by_name(trace, name):
    return [s for s in trace.spans if s.name == name]


def one(trace, name):
    found = by_name(trace, name)
    assert len(found) == 1, (name, found)
    return found[0]


# ── scenario: two streams running concurrently ───────────────────────────────

def concurrent_streams():
    return [
        host("cudaLaunchKernel", 1000, 50, lid=1, corr=101, corr2=601, stream=11),
        host("cudaLaunchKernel", 1100, 40, lid=2, corr=102, stream=22),
        cupti_kernel(101, 7, 1200, 5200, name="_Z4bigv"),
        cupti_kernel(102, 8, 1300, 3300, name="_Z5smallv"),
        host("cudaDeviceSynchronize", 1200, 4100, cat="sync", typ="sync", op="sync", lid=3, corr=103,
             sync="device"),
    ]


class TestTimingSource(unittest.TestCase):
    def test_classification(self):
        t = build(concurrent_streams())
        self.assertEqual(ga.timing_source(one(t, "_Z4bigv")), "device")
        self.assertEqual(ga.timing_source(by_name(t, "cudaLaunchKernel")[0]), "host")

    def test_legacy_spans(self):
        old = SpanEvent("k", ga_cat("cuda"), 0, 10, tags={"type": "kernel", "stream": "1"})
        old_cpu = SpanEvent("k", ga_cat("cuda"), 0, 10, tags={"type": "kernel", "timing": "cpu"})
        old_flush = SpanEvent("k", ga_cat("cuda"), 0, 10, tags={"type": "kernel", "timing": "cpu_flush"})
        self.assertEqual(ga.timing_source(old), "proxy_event")
        self.assertEqual(ga.timing_source(old_cpu), "proxy_host")
        self.assertEqual(ga.timing_source(old_flush), "proxy_flush")
        ocl_dev = SpanEvent("k", ga_cat("opencl"), 0, 10, tags={"type": "kernel", "side": "gpu"})
        ocl_host = SpanEvent("k", ga_cat("opencl"), 0, 10, tags={"type": "kernel", "side": "cpu"})
        self.assertEqual(ga.timing_source(ocl_dev), "device")
        self.assertTrue(ga.is_device_kernel(ocl_dev))
        self.assertFalse(ga.is_device_kernel(ocl_host))


def ga_cat(value):
    from src.core.events import Category
    return Category(value)


class TestConcurrentStreams(unittest.TestCase):
    def setUp(self):
        self.t = build(concurrent_streams())
        self.big, self.small = one(self.t, "_Z4bigv"), one(self.t, "_Z5smallv")
        self.h1, self.h2 = sorted(by_name(self.t, "cudaLaunchKernel"), key=lambda s: s.start_ns)

    def test_device_spans_take_the_program_stream_and_launching_thread(self):
        self.assertEqual(self.big.tags["stream"], "11")
        self.assertEqual(self.small.tags["stream"], "22")
        self.assertEqual(self.big.tags["nstream"], "7")
        self.assertEqual(self.big.tid, TID)
        self.assertEqual(self.big.parent_span_id, self.h1.span_id)

    def test_host_duration_queueing_and_execution_are_separate(self):
        self.assertEqual(self.big.tags["api_ns"], "50")
        self.assertEqual(self.big.tags["launch_ns"], "200")
        self.assertEqual(self.big.tags["queue_ns"], "150")      # 1200 - (1000 + 50)
        self.assertEqual(self.big.duration_ns, 4000)
        self.assertEqual(self.small.tags["queue_ns"], "160")

    def test_gpu_active_is_the_union_of_device_intervals(self):
        ka = ga.kernel_activity(self.t.spans)
        self.assertEqual(ka.source, "device")
        self.assertEqual(ga.merged_length(ka.intervals), 4000)
        sv = gpu_starvation(self.t)
        self.assertEqual(sv["gpu_active_ns"], 4000)
        self.assertEqual(sv["gpu_active_source"], "device")

    def test_lanes(self):
        lanes = self.t.lanes()
        self.assertIn(self.h1, lanes["cuda/thread-100"])
        self.assertIn(self.big, lanes["cuda/stream-11"])
        self.assertIn(self.small, lanes["cuda/stream-22"])

    def test_host_calls_are_runtime_overhead_and_device_time_is_not_host_time(self):
        self.assertEqual(ab.bucket_of_span(self.h1), "Runtime overhead")
        self.assertTrue(ab.is_device_timed(self.big))
        self.assertFalse(ab.is_device_timed(self.h1))
        owned = ab.exclusive_host_ns(self.t.spans)
        self.assertNotIn(id(self.big), owned)
        self.assertEqual(owned[id(self.h1)], 50)

    def test_summary(self):
        e = self.t.metadata.device_activity[f"{PID}/cuda"]
        self.assertEqual(e["device_spans"], 2)
        self.assertEqual(e["correlated"], 2)
        self.assertEqual(e["timing"], "device")
        self.assertNotIn("unmatched_device", e)

    def test_cupti_sync_records_are_not_proxy_spans(self):
        t = build(concurrent_streams() + [cupti_sync(103, "context", 1210, 5290)])
        e = t.metadata.device_activity[f"{PID}/cuda"]
        self.assertEqual((e["timing"], e["sync_records"]), ("device", 1))
        self.assertNotIn("proxy_spans", e)


class TestAsyncCopies(unittest.TestCase):
    def test_runtime_and_driver_ids_both_match(self):
        # The hook captured the runtime id (corr) and the nested driver id
        # (corr2); CUPTI's memcpy record has them the other way round.
        t = build([
            host("cudaMemcpyAsync", 1000, 30, cat="memory", op="memcpy", lid=1, corr=201, corr2=202,
                 stream=33, dir="HtoD", bytes=1 << 20),
            cupti_memcpy(202, 201, 8, 1500, 2500),
        ])
        copy = one(t, "memcpy HtoD")
        self.assertEqual(copy.tags["stream"], "33")
        self.assertEqual(copy.tags["queue_ns"], "470")
        self.assertTrue(ab.is_device_timed(copy))
        self.assertEqual(t.metadata.device_activity[f"{PID}/cuda"]["correlated"], 1)

    def test_copy_overlapping_a_kernel_on_another_stream(self):
        t = build(concurrent_streams() + [
            host("cudaMemcpyAsync", 1150, 20, cat="memory", op="memcpy", lid=4, corr=104, stream=33),
            cupti_memcpy(204, 104, 9, 1400, 2400),
        ])
        copy = one(t, "memcpy HtoD")
        self.assertEqual(copy.tags["stream"], "33")
        # copies don't count as kernel activity
        self.assertEqual(ga.merged_length(ga.kernel_activity(t.spans).intervals), 4000)


class TestMissingCorrelation(unittest.TestCase):
    def test_device_record_without_host_span_is_kept_and_counted(self):
        t = build(concurrent_streams() + [cupti_kernel(999, 7, 6000, 7000, name="_Z6orphanv")])
        orphan = one(t, "_Z6orphanv")
        e = t.metadata.device_activity[f"{PID}/cuda"]
        self.assertEqual(e["unmatched_device"], 1)
        # its stream id is still unified: nstream 7 was learned from launch #1
        self.assertEqual(orphan.tags["stream"], "11")
        self.assertEqual(orphan.tid, 0)
        self.assertNotIn("queue_ns", orphan.tags)
        self.assertEqual(ga.kernel_activity(t.spans).used, 3)

    def test_unknown_stream_keeps_a_vendor_lane(self):
        t = build([cupti_kernel(999, 42, 6000, 7000, name="_Z6orphanv")])
        self.assertEqual(one(t, "_Z6orphanv").tags["stream"], "n42")
        self.assertIn("cuda/stream-n42", t.lanes())

    def test_submission_whose_record_never_arrived(self):
        t = build(concurrent_streams() + [
            host("cudaLaunchKernel", 2000, 10, lid=9, corr=109, stream=11),
            host("cudaLaunchKernel", 2100, 10, lid=10, corr=110, stream=11, err=98),  # failed call
        ])
        self.assertEqual(t.metadata.device_activity[f"{PID}/cuda"]["host_without_device"], 1)


class TestReusedCorrelationIds(unittest.TestCase):
    def test_each_record_goes_to_the_latest_submission_before_it(self):
        # A wrapped 32-bit CUPTI id: the same corr=7 on two submissions only
        # 10 us apart (closer than the clock tolerance -- the harder case).
        t = build([
            host("cudaLaunchKernel", 1_000_000, 1_000, lid=1, corr=7, stream=1),
            host("cudaLaunchKernel", 1_010_000, 1_000, lid=2, corr=7, stream=2),
            cupti_kernel(7, 5, 1_005_000, 1_006_000, name="_Z5firstv"),
            cupti_kernel(7, 6, 1_020_000, 1_030_000, name="_Z6secondv"),
            cupti_kernel(7, 5, 800_000, 900_000, name="_Z5earlyv"),   # 100 us before any submission
        ])
        corr = ga.correlate(t.spans)
        first, second, early = (t.spans.index(one(t, n)) for n in ("_Z5firstv", "_Z6secondv", "_Z5earlyv"))
        self.assertEqual(t.spans[corr.host_of[first]].start_ns, 1_000_000)
        self.assertEqual(t.spans[corr.host_of[second]].start_ns, 1_010_000)
        self.assertEqual(corr.confidence[first], "high")
        self.assertIn(early, corr.precedes_submission)
        self.assertEqual(one(t, "_Z5firstv").tags["stream"], "1")
        self.assertEqual(one(t, "_Z6secondv").tags["stream"], "2")
        self.assertEqual(t.metadata.device_activity[f"{PID}/cuda"]["precedes_submission"], 1)

    def test_clock_mapping_error_is_tolerated(self):
        # Device start mapped 5 us before its (only) submission: offset-
        # calibration error, not a different launch.
        t = build([
            host("cudaLaunchKernel", 1_000_000, 1_000, lid=1, corr=9, stream=1),
            cupti_kernel(9, 5, 995_000, 1_100_000),
        ])
        e = t.metadata.device_activity[f"{PID}/cuda"]
        self.assertEqual(e["correlated"], 1)
        self.assertNotIn("precedes_submission", e)

    def test_graph_launch_one_id_many_kernels(self):
        t = build([
            host("cudaGraphLaunch", 100, 10, op="graph", lid=1, corr=50, stream=3),
            cupti_kernel(50, 12, 200, 300, name="_Z5nodeAv", graph=4),
            cupti_kernel(50, 13, 210, 320, name="_Z5nodeBv", graph=4),
        ])
        a, b = one(t, "_Z5nodeAv"), one(t, "_Z5nodeBv")
        h = one(t, "cudaGraphLaunch")
        self.assertEqual(a.parent_span_id, h.span_id)
        self.assertEqual(b.parent_span_id, h.span_id)
        # graph nodes run on internal streams: not forced onto the launch stream
        self.assertEqual(a.tags["stream"], "n12")
        self.assertEqual(b.tags["stream"], "n13")


class TestBufferOverflow(unittest.TestCase):
    def test_dropped_records_accumulate_and_are_reported(self):
        lines = concurrent_streams() + [
            "gpuact:1:cupti:status=active,api_version=28,headers_version=28,clock=monotonic_callback,"
            "correlation=callback,latency=1",
            "gpuact:1:cupti:dropped=37,notime=1",
            "gpuact:1:cupti:dropped=5",
            host("cudaLaunchKernel", 6000, 10, lid=9, corr=109, stream=11),   # its record was dropped
        ]
        t = build(lines)
        e = t.metadata.device_activity[f"{PID}/cuda"]
        self.assertEqual(e["dropped"], 42)
        self.assertEqual(e["notime"], 1)
        self.assertEqual(e["status"], "active")
        self.assertEqual(e["clock"], "monotonic_callback")
        self.assertEqual(e["host_without_device"], 1)
        text = "\n".join(ga.describe(t.metadata.device_activity))
        self.assertIn("42 records dropped", text)
        self.assertIn("1 submissions with no device record", text)

    def test_unavailable_tracer_status(self):
        t = build([
            "gpuact:1:cupti:status=unavailable,reason=libcupti_v18_older_than_headers_v28",
            host("cudaLaunchKernel", 100, 10, lid=1, stream=1),
            proxy_kernel(1, 1, 100, 500),
        ])
        e = t.metadata.device_activity[f"{PID}/cuda"]
        self.assertEqual(e["timing"], "proxy")
        self.assertIn("libcupti_v18_older_than_headers_v28", "\n".join(ga.describe(t.metadata.device_activity)))


class TestOutOfOrderDelivery(unittest.TestCase):
    def test_result_is_independent_of_arrival_order(self):
        lines = concurrent_streams() + [
            host("cudaMemcpyAsync", 1150, 20, cat="memory", op="memcpy", lid=4, corr=104, stream=33),
            cupti_memcpy(204, 104, 9, 1400, 2400),
            cupti_kernel(999, 7, 6000, 7000, name="_Z6orphanv"),
        ]

        def snapshot(order):
            t = build(order)
            return sorted((s.name, s.start_ns, s.tid, s.parent_span_id,
                           tuple(sorted(s.tags.items()))) for s in t.spans), \
                {k: v for k, v in t.metadata.device_activity.items() if k != "_assembled"}

        reference = snapshot(lines)
        # device records first (buffer flushed before the host spans arrived)
        devices_first = [l for l in lines if "side=gpu" in l] + [l for l in lines if "side=gpu" not in l]
        self.assertEqual(snapshot(devices_first), reference)
        rng = random.Random(7)
        for _ in range(5):
            shuffled = lines[:]
            rng.shuffle(shuffled)
            self.assertEqual(snapshot(shuffled), reference)


class TestDeduplication(unittest.TestCase):
    def test_proxy_span_for_a_natively_seen_launch_is_removed(self):
        # HPROFILER_DEVICE_ACTIVITY=both: hook proxy + CUPTI record, same launch.
        t = build([
            host("cudaLaunchKernel", 1000, 50, lid=1, corr=101, stream=11),
            proxy_kernel(1, 11, 1000, 3900, corr=101),
            cupti_kernel(101, 7, 1200, 5200),
        ])
        kernels = [s for s in t.spans if ga.is_device_kernel(s)]
        self.assertEqual(len(kernels), 1)
        self.assertEqual(ga.timing_source(kernels[0]), "device")
        self.assertEqual(t.metadata.device_activity[f"{PID}/cuda"]["deduplicated_proxy"], 1)

    def test_proxy_without_native_counterpart_stays(self):
        t = build([
            host("cudaLaunchKernel", 1000, 50, lid=1, corr=101, stream=11),
            host("cudaLaunchKernel", 2000, 50, lid=2, corr=102, stream=11),
            proxy_kernel(1, 11, 1000, 3900, corr=101),
            proxy_kernel(2, 11, 2000, 900, corr=102),
            cupti_kernel(101, 7, 1200, 5200),        # record for launch 2 lost
        ])
        self.assertEqual(sum(ga.is_device_kernel(s) for s in t.spans), 2)
        e = t.metadata.device_activity[f"{PID}/cuda"]
        self.assertEqual(e["timing"], "device+proxy")
        # ... but GPU-active never mixes the two sources
        ka = ga.kernel_activity(t.spans)
        self.assertEqual((ka.source, ka.used, ka.excluded), ("device", 1, 1))

    def test_exact_duplicate_native_records(self):
        rec = cupti_kernel(101, 7, 1200, 5200)
        t = build([host("cudaLaunchKernel", 1000, 50, lid=1, corr=101, stream=11), rec, rec])
        self.assertEqual(sum(ga.is_device_kernel(s) for s in t.spans), 1)
        self.assertEqual(t.metadata.device_activity[f"{PID}/cuda"]["duplicate_records"], 1)


class TestNeverMixProxyAndDevice(unittest.TestCase):
    def test_native_process_and_proxy_process(self):
        t = build([
            host("cudaLaunchKernel", 900, 10, lid=1, corr=1, stream=1, pid=1),
            cupti_kernel(1, 7, 1000, 2000, pid=1),
            host("cudaLaunchKernel", 1400, 10, lid=1, stream=1, pid=2),
            proxy_kernel(1, 1, 1400, 2100, pid=2),
        ])
        ka = ga.kernel_activity(t.spans)
        self.assertEqual((ka.source, ka.used, ka.excluded), ("device", 1, 1))
        self.assertEqual(ga.merged_length(ka.intervals), 1000)
        self.assertEqual(gpu_starvation(t)["gpu_active_ns"], 1000)

    def test_blocking_copy_in_proxy_mode_is_not_a_lost_record(self):
        t = build([
            host("cudaLaunchKernel", 0, 10, lid=1, stream=1),
            proxy_kernel(1, 1, 0, 1000),
            host("cudaMemcpy", 2000, 500, cat="memory", typ="memcpy", op="memcpy", lid=2),
        ])
        self.assertNotIn("host_without_device", t.metadata.device_activity[f"{PID}/cuda"])

    def test_proxy_only_trace_excludes_host_timed_fallbacks(self):
        t = build([
            host("cudaLaunchKernel", 0, 10, lid=1, stream=1),
            proxy_kernel(1, 1, 0, 1000),
            host("cudaLaunchKernel", 2000, 10, lid=2, stream=1),
            proxy_kernel(2, 1, 2000, 10, timing="proxy_host"),
        ])
        ka = ga.kernel_activity(t.spans)
        self.assertEqual((ka.source, ka.used, ka.excluded), ("proxy", 1, 1))
        sv = gpu_starvation(t)
        self.assertEqual(sv["gpu_active_source"], "proxy")
        self.assertEqual(sv["gpu_active_ns"], 1000)

    def test_summary_prints_the_source(self):
        from src.output.summary import print_summary
        buf = io.StringIO()
        with redirect_stdout(buf):
            print_summary(build(concurrent_streams()))
        out = buf.getvalue()
        self.assertIn("CUDA kernel active", out)
        self.assertIn("device-measured", out)
        self.assertIn("queued", out)


class TestRocm(unittest.TestCase):
    def test_external_correlation_and_queue_mapping(self):
        t = build([
            host("hipLaunchKernel", 1000, 40, cat="rocm", lid=1, stream=33, rt="rocm"),
            host("hipLaunchKernel", 1100, 40, cat="rocm", lid=2, stream=44, rt="rocm"),
            roc_dispatch(20, 1, 160, 1200, 2200, name="_Z1av"),
            roc_dispatch(21, 2, 176, 1250, 2000, name="_Z1bv"),
            roc_dispatch(22, None, 160, 2300, 2400, name="_Z1cv"),   # no external id
        ])
        a, b, c = one(t, "_Z1av"), one(t, "_Z1bv"), one(t, "_Z1cv")
        self.assertEqual((a.tags["stream"], b.tags["stream"]), ("33", "44"))
        self.assertEqual(a.tags["queue_ns"], "160")
        self.assertEqual(c.tags["stream"], "33")      # learned from queue 160
        e = t.metadata.device_activity[f"{PID}/rocm"]
        self.assertEqual((e["correlated"], e["unmatched_device"]), (2, 1))

    def test_hip_streams_sharing_one_queue_are_not_merged(self):
        t = build([
            host("hipLaunchKernel", 1000, 40, cat="rocm", lid=1, stream=33, rt="rocm"),
            host("hipLaunchKernel", 1100, 40, cat="rocm", lid=2, stream=44, rt="rocm"),
            roc_dispatch(20, 1, 160, 1200, 2200, name="_Z1av"),
            roc_dispatch(21, 2, 160, 2250, 3000, name="_Z1bv"),
            roc_dispatch(22, None, 160, 3100, 3200, name="_Z1cv"),
        ])
        self.assertEqual(one(t, "_Z1av").tags["stream"], "33")
        self.assertEqual(one(t, "_Z1bv").tags["stream"], "44")
        self.assertEqual(one(t, "_Z1cv").tags["stream"], "q160")   # ambiguous: vendor lane

    def test_cuda_and_rocm_lids_do_not_collide(self):
        t = build([
            host("cudaLaunchKernel", 1000, 10, lid=1, stream=1),
            proxy_kernel(1, 1, 1000, 500, name="_Z4cudav"),
            host("hipLaunchKernel", 1000, 10, cat="rocm", lid=1, stream=2, rt="rocm"),
            roc_dispatch(20, 1, 160, 1100, 1300, name="_Z3hipv"),
        ])
        corr = ga.correlate(t.spans)
        hip = t.spans.index(one(t, "_Z3hipv"))
        self.assertEqual(t.spans[corr.host_of[hip]].name, "hipLaunchKernel")


class TestRoundTrip(unittest.TestCase):
    def test_json_preserves_correlation_and_provenance(self):
        lines = concurrent_streams() + ["gpuact:1:cupti:status=active,clock=monotonic_callback",
                                        "gpuact:1:cupti:dropped=3"]
        t = build(lines)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.json")
            chrome_trace.write(t, path)
            r = chrome_trace.load_trace_from_json(path)
        big = one(r, "_Z4bigv")
        h1 = min(by_name(r, "cudaLaunchKernel"), key=lambda s: s.start_ns)
        self.assertEqual(big.parent_span_id, h1.span_id)
        self.assertEqual(big.tid, TID)
        self.assertEqual(big.tags["queue_ns"], "150")
        self.assertEqual(h1.tags["side"], "cpu")
        self.assertEqual(r.metadata.device_activity[f"{PID}/cuda"]["dropped"], 3)
        self.assertTrue(r.metadata.device_activity["_assembled"])
        before = [(s.name, s.start_ns) for s in r.spans]
        ga.assemble(r)          # already assembled: no-op
        self.assertEqual([(s.name, s.start_ns) for s in r.spans], before)
        # host spans stay on their real thread in the Chrome trace
        self.assertEqual(h1.tid, TID)

    def test_legacy_trace_untouched(self):
        t = Trace()
        t.add(SpanEvent("k", ga_cat("cuda"), 0, 100, pid=1, tid=5, tags={"type": "kernel", "stream": "1"}))
        ga.assemble(t)
        self.assertEqual(t.spans[0].tid, 5)
        self.assertEqual(ga.kernel_activity(t.spans).source, "proxy")


# ── critical path ──────────────────────────────────────────────────────────

def path_of(trace):
    spans, preds = cp.build_dependency_graph(trace)
    path, conf, kinds = cp._compute_path_full(spans, preds)
    return [spans[i].name for i in path], kinds, spans, preds, path


class TestCriticalPath(unittest.TestCase):
    def test_device_sync_waits_for_the_work_that_finished_last(self):
        t = build(concurrent_streams())
        names, kinds, *_ = path_of(t)
        self.assertEqual(names, ["cudaLaunchKernel", "_Z4bigv", "cudaDeviceSynchronize"])
        self.assertEqual(kinds, ["launch", "device_wait"])
        report = cp.analyze(t)
        # every instant of the run counted once: no double counting of the
        # sync call over the kernel it waited for
        self.assertEqual(report.total_path_ns, 4300)
        self.assertEqual(report.wall_ns, 4300)
        self.assertEqual(report.time_on_path_by_category["cuda"], 50 + 4000)
        self.assertEqual(report.time_on_path_by_category["sync"], 100)
        self.assertEqual(report.wait_caused_by_category["cuda"], 150)   # queueing delay

    def test_launch_edge_tolerates_work_starting_before_the_call_returns(self):
        t = build([
            host("cudaLaunchKernel", 1000, 500, lid=1, corr=1, stream=1),
            cupti_kernel(1, 7, 1200, 3000),
            host("cudaStreamSynchronize", 1600, 1500, cat="sync", typ="sync", op="sync", lid=2,
                 sync="stream", stream=1),
        ])
        names, kinds, *_ = path_of(t)
        self.assertEqual(names, ["cudaLaunchKernel", "_Z4axpyPf", "cudaStreamSynchronize"])
        report = cp.analyze(t)
        self.assertEqual(report.total_path_ns, 3100 - 1000)

    def test_stream_order_event_sync_and_cross_stream_wait(self):
        t = build([
            host("cudaLaunchKernel", 1000, 10, lid=1, corr=201, stream=11),
            host("cudaEventRecord", 1020, 5, typ="event_record", op="event_record", lid=2, event=55, stream=11),
            host("cudaStreamWaitEvent", 1030, 5, typ="stream_wait", op="stream_wait", lid=3, event=55, stream=22),
            host("cudaLaunchKernel", 1040, 10, lid=4, corr=204, stream=22),
            host("cudaEventSynchronize", 1060, 2050, cat="sync", typ="sync", op="sync", lid=5,
                 sync="event", event=55),
            host("cudaStreamSynchronize", 3120, 1040, cat="sync", typ="sync", op="sync", lid=6,
                 sync="stream", stream=22),
            cupti_kernel(201, 7, 1100, 3100, name="_Z8producerv"),
            cupti_kernel(204, 8, 3150, 4150, name="_Z8consumerv"),
        ])
        names, kinds, spans, preds, path = path_of(t)
        self.assertEqual(names, ["cudaLaunchKernel", "_Z8producerv", "_Z8consumerv", "cudaStreamSynchronize"])
        self.assertEqual(kinds, ["launch", "sequential", "device_wait"])
        idx = {s.name: i for i, s in enumerate(spans) if s.name != "cudaLaunchKernel"}
        ev_sync_preds = {(spans[u].name, k, c) for u, k, c in preds[idx["cudaEventSynchronize"]]}
        self.assertIn(("_Z8producerv", "device_wait", "certain"), ev_sync_preds)
        self.assertNotIn("_Z8consumerv", {n for n, _, _ in ev_sync_preds})
        consumer_preds = {(spans[u].name, k) for u, k, _ in preds[idx["_Z8consumerv"]]}
        self.assertIn(("_Z8producerv", "sequential"), consumer_preds)

    def test_cross_stream_producer_gates_the_consumer_not_its_own_stream(self):
        # s2's independent kernel finishes early; the consumer on s2 really
        # waits for s1's producer (cudaStreamWaitEvent). Both predecessors
        # account the same time; the one that finished last gated it.
        t = build([
            host("cudaLaunchKernel", 1000, 10, lid=1, corr=1, stream=11),
            host("cudaLaunchKernel", 1020, 10, lid=2, corr=2, stream=22),
            host("cudaEventRecord", 1040, 5, typ="event_record", op="event_record", lid=3, event=5, stream=11),
            host("cudaStreamWaitEvent", 1050, 5, typ="stream_wait", op="stream_wait", lid=4, event=5, stream=22),
            host("cudaLaunchKernel", 1060, 10, lid=5, corr=5, stream=22),
            cupti_kernel(1, 7, 1100, 9000, name="_Z8producerv"),
            cupti_kernel(2, 8, 1150, 2000, name="_Z11independentv"),
            cupti_kernel(5, 8, 9050, 9500, name="_Z8consumerv"),
        ])
        names, kinds, *_ = path_of(t)
        self.assertEqual(names[-2:], ["_Z8producerv", "_Z8consumerv"])
        self.assertEqual(cp.analyze(t).wait_caused_by_category.get("cuda"), 50 + 90)

    def test_stream_sync_ignores_other_streams(self):
        t = build(concurrent_streams()[:4] + [
            host("cudaStreamSynchronize", 1200, 2200, cat="sync", typ="sync", op="sync", lid=3,
                 sync="stream", stream=22),
        ])
        spans, preds = cp.build_dependency_graph(t)
        si = next(i for i, s in enumerate(spans) if s.name == "cudaStreamSynchronize")
        waited = {spans[u].name for u, k, _ in preds[si] if k == "device_wait"}
        self.assertEqual(waited, {"_Z5smallv"})

    def test_event_sync_without_its_record_falls_back_at_medium_confidence(self):
        t = build(concurrent_streams()[:4] + [
            host("cudaEventSynchronize", 1200, 4100, cat="sync", typ="sync", op="sync", lid=3,
                 sync="event", event=77),
        ])
        spans, preds = cp.build_dependency_graph(t)
        si = next(i for i, s in enumerate(spans) if s.name == "cudaEventSynchronize")
        self.assertEqual({(spans[u].name, c) for u, k, c in preds[si] if k == "device_wait"},
                         {("_Z4bigv", "medium"), ("_Z5smallv", "medium")})

    def test_device_spans_are_not_in_host_program_order(self):
        t = build(concurrent_streams())
        spans, preds = cp.build_dependency_graph(t)
        big = next(i for i, s in enumerate(spans) if s.name == "_Z4bigv")
        self.assertEqual({k for _u, k, _c in preds[big]}, {"launch"})

    def test_proxy_trace_gets_a_note(self):
        t = build([
            host("cudaLaunchKernel", 0, 10, lid=1, stream=1),
            proxy_kernel(1, 1, 0, 1000),
        ])
        self.assertTrue(any("host-side proxies" in n for n in cp.analyze(t).notes))


if __name__ == "__main__":
    unittest.main()
