"""
OpenCL queue, event and wait-list ids through the real OpenCL hook
(tests/fixtures/ocl_deps.c, on any OpenCL device): every enqueue names its
queue (queue=, ooo=1 on an out-of-order one), the event it returned
(ev=) and its wait list (wait=, nwait=, wait_unknown= for an event the hook
cannot name -- here a user event); a blocking transfer says so
(blocking=1); clFinish names its queue and clWaitForEvents its events; a
non-blocking transfer's device span links to its enqueue (psid= -> sid=).
Skips without gcc, OpenCL headers or a device, or with a hook library that
predates the ids (HPROFILER_HOOKS_DIR points at another build).
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


class TestOpenCLDependencyIds(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.core.hookdir import hooks_dir
        cls.tmp = tempfile.mkdtemp()
        exe = os.path.join(cls.tmp, "ocl_deps")
        if not shutil.which("gcc") or not (hooks_dir() / "libhprofiler_opencl.so").exists():
            raise unittest.SkipTest("needs gcc and the built OpenCL hook")
        r = subprocess.run(["gcc", "-O1", str(ROOT / "tests/fixtures/ocl_deps.c"), "-o", exe, "-lOpenCL"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise unittest.SkipTest(f"cannot build the fixture: {r.stderr[:200]}")
        probe = subprocess.run([exe], capture_output=True, text=True, timeout=120)
        if probe.returncode != 0:
            raise unittest.SkipTest(f"no usable OpenCL device ({probe.stdout.strip()})")
        cls.ooo = "ooo=1" in probe.stdout
        from src.core.runner import Runner
        trace = Runner([exe], backends=["opencl"]).run()
        cls.spans = sorted((s for s in trace.iter_spans() if s.category.value in ("opencl", "memory", "sync")),
                           key=lambda s: s.start_ns)
        if not any("queue" in s.tags for s in cls.spans):
            raise unittest.SkipTest("the hook library predates queue= -- rebuild build/ or point "
                                    "HPROFILER_HOOKS_DIR at a fresh build")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def host(self, name, nth=0):
        return [s for s in self.spans if s.name == name and s.tags.get("side") != "gpu"
                and not s.name.endswith("_gpu")][nth]

    def test_queues(self):
        k1, k2 = self.host("k1"), self.host("k2")
        self.assertNotEqual(k1.tags["queue"], k2.tags["queue"])
        self.assertNotIn("ooo", k1.tags)
        self.assertEqual(k2.tags.get("ooo") == "1", self.ooo)
        finishes = [s.tags["queue"] for s in self.spans if s.name == "clFinish"]
        self.assertEqual(finishes, [k1.tags["queue"], k2.tags["queue"]])

    def test_wait_lists_name_the_events_waited_for(self):
        k1, k2 = self.host("k1"), self.host("k2")
        read = self.host("clEnqueueReadBuffer")
        marker = self.host("clEnqueueMarkerWithWaitList")
        barrier = self.host("clEnqueueBarrierWithWaitList")
        self.assertEqual(k2.tags["wait"], k1.tags["ev"])
        self.assertEqual(read.tags["wait"], k2.tags["ev"])
        self.assertEqual(barrier.tags["wait"], read.tags["ev"])
        self.assertEqual(barrier.tags["queue"], k1.tags["queue"])
        self.assertEqual(marker.tags["queue"], k2.tags["queue"])
        waits = next(s for s in self.spans if s.name == "clWaitForEvents")
        self.assertEqual(waits.tags["wait"].split(";"), [read.tags["ev"], marker.tags["ev"]])
        self.assertEqual(len({k1.tags["ev"], k2.tags["ev"], read.tags["ev"], marker.tags["ev"]}), 4)

    def test_an_unnamed_event_is_counted_not_guessed(self):
        gated = self.host("k1", 1)                  # waits for a user event
        self.assertEqual((gated.tags.get("nwait"), gated.tags.get("wait_unknown")), ("1", "1"))
        self.assertNotIn("wait", gated.tags)
        self.assertNotIn("ev", gated.tags)          # the caller asked for no event

    def test_blocking_transfers(self):
        self.assertEqual(self.host("clEnqueueWriteBuffer").tags.get("blocking"), "1")
        self.assertNotIn("blocking", self.host("clEnqueueReadBuffer").tags)
        # only the non-blocking read has a device span
        self.assertEqual([s.name for s in self.spans if s.name.endswith("Buffer_gpu")], ["clReadBuffer_gpu"])

    def test_the_graph_follows_queue_semantics(self):
        from src.analysis import criticalpath as cp
        from src.core.trace import Trace
        t = Trace()
        for s in self.spans:
            t.add(s)
        sp, preds = cp.build_dependency_graph(t)

        def side(s):
            return "dev" if s.tags.get("side") == "gpu" else "host"
        e = {(k, sp[u].name, side(sp[u]), sp[v].name, side(sp[v])) for v, pl in preds.items()
             for u, k, _c in pl if k in ("sequential", "device_wait") and side(sp[u]) == "dev"}
        self.assertIn(("sequential", "k1", "dev", "k2", "dev"), e)                     # wait list across queues
        self.assertIn(("sequential", "k2", "dev", "clReadBuffer_gpu", "dev"), e)
        self.assertIn(("device_wait", "clReadBuffer_gpu", "dev", "clWaitForEvents", "host"), e)
        self.assertIn(("device_wait", "k1", "dev", "clEnqueueWriteBuffer", "host"), e)  # blocking, in-order queue
        self.assertIn(("device_wait", "k1", "dev", "clFinish", "host"), e)
        self.assertIn(("device_wait", "clReadBuffer_gpu", "dev", "clFinish", "host"), e)
        # the barrier made the user-event kernel follow the read
        self.assertIn(("sequential", "clReadBuffer_gpu", "dev", "k1", "dev"), e)

    def test_device_spans_link_to_their_enqueue(self):
        by_sid = {s.span_id: s for s in self.spans if s.span_id}
        dev = [s for s in self.spans if s.tags.get("side") == "gpu" or s.name.endswith("_gpu")]
        self.assertEqual(len(dev), 4)               # k1, k2, the read, the gated k1
        for d in dev:
            h = by_sid[d.parent_span_id]
            self.assertEqual(h.tags["queue"], d.tags["queue"], d.name)
            self.assertEqual(h.tags.get("ev"), d.tags.get("ev"), d.name)


if __name__ == "__main__":
    unittest.main()
