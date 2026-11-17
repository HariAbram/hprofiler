"""
OpenCL event capture through the real OpenCL hook on every platform's first
device (tests/fixtures/ocl_events.c; here an NVIDIA GPU and an Intel CPU):

  * a 40-event wait list is recorded whole (wait= every id, nwait=40);
  * a kernel gated on a user event waits for the clSetUserEventStatus call
    that completed it (an explicit edge, host call -> kernel);
  * an event released and a new one at the same address: the waiter names
    the new event's id, never the released one's;
  * copy / rectangle read / map / unmap / migrate commands are recorded and
    their wait lists chain them; so are the Intel USM fill and copy, which
    are reached through clGetExtensionFunctionAddressForPlatform (where a
    platform has the extension: here the Intel CPU runtime);
  * each device's clock is mapped on its own: clGetDeviceAndHostTimer where
    the device has it (method host_timer, clock_err_ns= on its spans, a
    bounded clock domain), else the first-callback offset (callback_end,
    unbounded, reported);
  * with a tiny event table (HPROFILER_OCL_EVENT_CAP), events beyond it get
    no id -- counted, and their waiters list them as wait_unknown -- and no
    wait list ever names an id the hook did not hand out for that event.

Skips without gcc, OpenCL headers, a device or a hook that predates dev=.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


def _run(exe, args=(), env=None):
    from src.core.runner import Runner
    rf, wf = os.pipe()
    saved = os.dup(1)
    os.dup2(wf, 1)
    try:
        with mock.patch.dict(os.environ, env or {}):
            trace = Runner([exe, *args], backends=["opencl"]).run()
    finally:
        os.dup2(saved, 1)
        os.close(wf)
    out = os.read(rf, 65536).decode()
    os.close(rf)
    return trace, out


class TestOpenCLEvents(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.core.hookdir import hooks_dir
        cls.tmp = tempfile.mkdtemp()
        cls.exe = os.path.join(cls.tmp, "ocl_events")
        if not shutil.which("gcc") or not (hooks_dir() / "libhprofiler_opencl.so").exists():
            raise unittest.SkipTest("needs gcc and the built OpenCL hook")
        r = subprocess.run(["gcc", "-O1", str(ROOT / "tests/fixtures/ocl_events.c"), "-o", cls.exe, "-lOpenCL"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise unittest.SkipTest(f"cannot build the fixture: {r.stderr[:200]}")
        base = subprocess.run([cls.exe], capture_output=True, text=True, timeout=120)
        if base.returncode != 0:
            raise unittest.SkipTest(f"no usable OpenCL device ({base.stdout.strip()[:120]})")
        cls.trace, cls.out = _run(cls.exe)
        cls.spans = sorted(cls.trace.iter_spans(), key=lambda s: s.start_ns)
        if not any("dev" in s.tags for s in cls.spans if s.tags.get("side") == "gpu"):
            raise unittest.SkipTest("the OpenCL hook predates dev= -- point HPROFILER_HOOKS_DIR at a fresh build")
        from src.analysis import criticalpath as cp
        cls.cp = cp
        cls.gspans, cls.preds = cp.build_dependency_graph(cls.trace)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(getattr(cls, "tmp", ""), ignore_errors=True)

    def host(self, name):
        return [s for s in self.spans if s.name == name and s.tags.get("side") != "gpu" and not s.name.endswith("_gpu")]

    def by_queue(self, name):
        out = defaultdict(list)
        for s in self.host(name):
            out[s.tags.get("queue")].append(s)
        return out

    def test_a_long_wait_list_is_recorded_whole(self):
        fills = self.by_queue("clEnqueueFillBuffer")
        longs = [s for s in self.host("inc") if s.tags.get("nwait") == "40"]
        self.assertTrue(longs)
        for k in longs:
            want = [f.tags["ev"] for f in fills[k.tags["queue"]][:40]]
            self.assertEqual(k.tags["wait"].split(";"), want)
            self.assertNotIn("wait_unknown", k.tags)

    def test_a_user_event_gates_its_waiter_on_the_host_call(self):
        idx = {id(s): i for i, s in enumerate(self.gspans)}
        for setter in self.host("clSetUserEventStatus"):
            ev = setter.tags["ev"]
            gated = next(s for s in self.host("inc") if s.tags.get("wait") == ev)
            dev = next(i for i, s in enumerate(self.gspans) if s.tags.get("side") == "gpu"
                       and s.parent_span_id == gated.span_id)
            kinds = {(self.gspans[u].name, k) for u, k, _c in self.preds.get(dev, [])}
            self.assertIn(("clSetUserEventStatus", "explicit_span_id"), kinds)
            self.assertIn(idx[id(setter)], {u for u, _k, _c in self.preds.get(dev, [])})

    def test_a_reused_event_address_never_inherits_the_old_id(self):
        reused = [ln for ln in self.out.splitlines() if ln.startswith("dev=") and "reused=1" in ln.split()]
        if not reused:
            self.skipTest("no driver reused an event address in this run")
        for q, barriers in self.by_queue("clEnqueueBarrierWithWaitList").items():
            barrier = barriers[0]
            markers = [m for m in self.by_queue("clEnqueueMarkerWithWaitList")[q] if m.start_ns < barrier.start_ns]
            # the last marker made the event the barrier waits for; the first one's was released
            self.assertEqual(barrier.tags["wait"], markers[-1].tags["ev"])
            self.assertNotEqual(barrier.tags["wait"], markers[0].tags["ev"])

    def test_more_commands_and_their_chain(self):
        for name in ("clEnqueueCopyBuffer", "clEnqueueReadBufferRect", "clEnqueueMapBuffer",
                     "clEnqueueUnmapMemObject", "clEnqueueMigrateMemObjects"):
            self.assertEqual(len(self.host(name)), len({s.tags["queue"] for s in self.host("inc")}), name)
        for q, copies in self.by_queue("clEnqueueCopyBuffer").items():
            rect = self.by_queue("clEnqueueReadBufferRect")[q][0]
            mp = self.by_queue("clEnqueueMapBuffer")[q][0]
            um = self.by_queue("clEnqueueUnmapMemObject")[q][0]
            self.assertEqual(rect.tags["wait"], copies[0].tags["ev"])
            self.assertEqual(mp.tags["wait"], rect.tags["ev"])
            self.assertEqual(um.tags["wait"], mp.tags["ev"])
            self.assertEqual(rect.tags["bytes"], "32")         # a buffer rectangle's width is in bytes

    def test_intel_usm_extension_commands(self):
        if "usm=1" not in self.out:
            self.skipTest("no platform with cl_intel_unified_shared_memory")
        for q, copies in self.by_queue("clEnqueueMemcpyINTEL").items():
            fill = self.by_queue("clEnqueueMemFillINTEL")[q][0]
            copy = copies[0]
            self.assertEqual((fill.tags["type"], copy.tags["type"]), ("usm_fill", "usm_memcpy"))
            self.assertEqual(copy.tags["wait"], fill.tags["ev"])
            marker = [m for m in self.by_queue("clEnqueueMarkerWithWaitList")[q] if m.start_ns > copy.start_ns][0]
            self.assertEqual(marker.tags["wait"], copy.tags["ev"])
            dev = [s for s in self.spans if s.parent_span_id == copy.span_id and s.tags.get("side") == "gpu"]
            self.assertEqual([s.name for s in dev], ["clMemcpyINTEL_gpu"])

    def test_each_device_clock_on_its_own(self):
        from src.analysis import cp_clock
        clocks = {i.tags["dev"]: i.tags for i in self.trace.iter_instants() if i.name == "ocl_clock"}
        devspans = [s for s in self.spans if s.tags.get("side") == "gpu" and s.tags.get("dev") is not None]
        for s in devspans:
            c = clocks[s.tags["dev"]]
            self.assertEqual(s.tags["clock"], c["method"])
            key, bound = cp_clock.domain_of(s, {})
            if c["method"] == "host_timer":
                self.assertGreater(int(s.tags["clock_err_ns"]), 0)
                self.assertEqual(key[0], "opencl_dev")
                self.assertLess(bound, 1e9)
            else:
                self.assertNotIn("clock_err_ns", s.tags)
                self.assertEqual(bound, float("inf"))
        if len(clocks) > 1:
            self.assertEqual(len({cp_clock.domain_of(s, {})[0] for s in devspans}), len(clocks))
        diag = self.cp.dependency_diagnostics(self.trace)
        self.assertEqual(diag.get("ocl_clock_unbounded", 0),
                         sum(1 for c in clocks.values() if c["method"] == "callback_end"))

    def test_table_pressure_names_nothing_falsely(self):
        trace, _out = _run(self.exe, ["press"], {"HPROFILER_OCL_EVENT_CAP": "8"})
        full = sum(c.value for c in trace.iter_counters() if c.name == "ocl_event_table_full")
        self.assertGreater(full, 0)
        spans = sorted(trace.iter_spans(), key=lambda s: s.start_ns)
        for k in (s for s in spans if s.name == "inc" and s.tags.get("side") == "cpu"):
            q = k.tags["queue"]
            last10 = [m for m in spans if m.name == "clEnqueueMarkerWithWaitList" and m.tags.get("queue") == q][-10:]
            named = [w for w in k.tags.get("wait", "").split(";") if w]
            # every named id is one of the ten events it waits for; the rest are unknown
            self.assertTrue(set(named) <= {m.tags.get("ev") for m in last10}, (named, [m.tags.get("ev") for m in last10]))
            self.assertEqual(len(named) + int(k.tags.get("wait_unknown", 0)), 10)
        diag = self.cp.dependency_diagnostics(trace)
        self.assertGreater(diag.get("ocl_event_table_full", 0), 0)


if __name__ == "__main__":
    unittest.main()
