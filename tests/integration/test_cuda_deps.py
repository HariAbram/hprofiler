"""
Blocking copies, stream flags and host functions through the CUDA hook
(tests/fixtures/cuda_deps.cu on a real CUDA device), with native device
activity (CUPTI) and with the event-pair proxy (HPROFILER_DEVICE_ACTIVITY=
off):

  * a synchronous cudaMemcpy says whether it blocked the host, per CUDA's
    documented API synchronization behaviour: blocking=1 to host memory and
    from pinned host memory, staged from pageable host memory to the
    device, 0 device to device; cudaMemcpyDefault resolved from the
    pointers; cudaMemset on device memory carries no blocking tag;
  * stream creation names the stream and its flags (nonblocking=1,
    priority=);
  * a host function or stream callback is an enqueue call (op=host_func,
    side=cpu) plus, when it runs, a stream-ordered span on the same stream
    (side=gpu, exec=host, timing=host) linked by lid=, as long as the
    function took.

Skips without nvcc, a CUDA device or a hook library that predates these
tags.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from src.core.hookdir import hooks_dir  # noqa: E402  (HPROFILER_HOOKS_DIR, else build/lib)


class TestCUDADependencyTags(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not shutil.which("nvcc") or not (hooks_dir() / "libhprofiler_cuda.so").exists():
            raise unittest.SkipTest("needs nvcc and the built CUDA hook")
        cls.tmp = tempfile.mkdtemp()
        exe = os.path.join(cls.tmp, "cuda_deps")
        r = subprocess.run(["nvcc", "-cudart", "shared", "-o", exe, str(ROOT / "tests/fixtures/cuda_deps.cu")],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise unittest.SkipTest(f"cannot build the fixture: {r.stderr[:300]}")
        probe = subprocess.run([exe], capture_output=True, text=True, timeout=120)
        if probe.returncode != 0 or "host_fn=1 callback=1" not in probe.stdout:
            raise unittest.SkipTest(f"no usable CUDA device ({probe.stdout.strip()})")
        from src.core.runner import Runner
        cls.runs = {}
        for mode in ("auto", "off"):
            with mock.patch.dict(os.environ, {"HPROFILER_DEVICE_ACTIVITY": mode}):
                cls.runs[mode] = list(Runner([exe], backends=["cuda"]).run().iter_spans())
        if not any(s.tags.get("op") == "stream_create" for s in cls.runs["auto"]):
            raise unittest.SkipTest("the hook library predates stream_create -- rebuild build/ or point "
                                    "HPROFILER_HOOKS_DIR at a fresh build")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(getattr(cls, "tmp", ""), ignore_errors=True)


    def test_synchronous_copies_say_whether_they_blocked(self):
        for mode, spans in self.runs.items():
            with self.subTest(mode=mode):
                copies = [(s.tags["dir"], s.tags.get("blocking")) for s in sorted(spans, key=lambda s: s.start_ns)
                          if s.name == "cudaMemcpy"]
                self.assertEqual(copies, [("HtoD", "staged"), ("HtoD", "1"), ("DtoH", "1"), ("DtoD", "0"),
                                          ("Default", "1")])
                memset = next(s for s in spans if s.name == "cudaMemset")
                self.assertNotIn("blocking", memset.tags)

    def test_stream_creation_records_flags(self):
        for mode, spans in self.runs.items():
            with self.subTest(mode=mode):
                created = {s.name: s.tags for s in spans if s.tags.get("op") == "stream_create"}
                self.assertEqual(set(created), {"cudaStreamCreate", "cudaStreamCreateWithFlags",
                                                "cudaStreamCreateWithPriority"})
                self.assertNotIn("nonblocking", created["cudaStreamCreate"])
                self.assertEqual(created["cudaStreamCreateWithFlags"].get("nonblocking"), "1")
                self.assertIn("priority", created["cudaStreamCreateWithPriority"])
                self.assertEqual(len({t["stream"] for t in created.values()}), 3)

    def test_the_graph_follows_the_stream_semantics(self):
        from src.analysis import criticalpath as cp
        from src.core.trace import Trace
        for mode, spans in self.runs.items():
            with self.subTest(mode=mode):
                t = Trace()
                for s in spans:
                    t.add(s)
                sp, preds = cp.build_dependency_graph(t)
                e = [(sp[u], sp[v], k, c) for v, pl in preds.items() for u, k, c in pl]
                nonblocking = next(s.tags["stream"] for s in spans if s.name == "cudaStreamCreateWithFlags")
                # the copies that block the host wait for their device copy
                waits = [b.tags.get("blocking") for a, b, k, _c in e if k == "device_wait" and b.name == "cudaMemcpy"]
                # (the proxy times no synchronous copy: only CUPTI records it on the device)
                self.assertEqual(sorted(waits), ["1", "1", "1"] if mode == "auto" else [])
                # the legacy default stream orders the blocking streams' first kernels, not the non-blocking one's
                legacy = {b.tags["stream"] for a, b, k, c in e
                          if k == "sequential" and a.tags.get("side") == b.tags.get("side") == "gpu"
                          and a.tags.get("stream") == "0" and b.tags.get("stream") not in (None, "0")}
                self.assertEqual(len(legacy), 2)
                self.assertNotIn(nonblocking, legacy)
                # kernel -> host function -> kernel on the non-blocking stream
                hf = [(a.tags.get("op"), b.tags.get("op")) for a, b, k, _c in e if k == "sequential"
                      and nonblocking == a.tags.get("stream") == b.tags.get("stream")]
                self.assertIn(("kernel", "host_func"), hf)
                self.assertIn(("host_func", "kernel"), hf)

    def test_host_functions_are_stream_ordered_spans(self):
        for mode, spans in self.runs.items():
            with self.subTest(mode=mode):
                enq = {s.tags["lid"]: s for s in spans if s.tags.get("op") == "host_func" and s.tags.get("side") == "cpu"}
                ran = [s for s in spans if s.tags.get("type") == "host_func"]
                self.assertEqual(sorted(e.name for e in enq.values()), ["cudaLaunchHostFunc", "cudaStreamAddCallback"])
                self.assertEqual(sorted(r.tags["lid"] for r in ran), sorted(enq))
                for r in ran:
                    e = enq[r.tags["lid"]]
                    self.assertEqual((r.tags["side"], r.tags["exec"], r.tags["timing"]), ("gpu", "host", "host"))
                    self.assertEqual(r.tags["stream"], e.tags["stream"])
                    self.assertGreaterEqual(r.start_ns, e.start_ns)
                # the functions sleep 2 ms and 1 ms
                durs = {enq[r.tags["lid"]].name: r.duration_ns for r in ran}
                self.assertGreater(durs["cudaLaunchHostFunc"], 1_800_000)
                self.assertGreater(durs["cudaStreamAddCallback"], 800_000)


if __name__ == "__main__":
    unittest.main()
