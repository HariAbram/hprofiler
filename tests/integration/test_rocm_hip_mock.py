"""
The ROCm hook's HIP wrappers against a stand-in HIP runtime
(tests/fixtures/hip_mock.c built as libamdhip64.so, first on
LD_LIBRARY_PATH; the program is tests/fixtures/hip_streams.c). No AMD GPU
is needed -- and none of this checks device execution or ROCprofiler-SDK:
device spans are the hook's hipEvent-pair proxies on the mock's clock.

  * stream creation records flags and priority; a stream destroyed and
    re-created at the same address gets another stream= number (two
    lives), the destroy names the life that ended; an event likewise;
  * host functions (hipLaunchHostFunc, hipStreamAddCallback) run on the
    runtime's one callback thread in submission order: each enqueue span
    links to its execution span (lid=), and the dependency graph orders
    consecutive ones across streams;
  * cooperative and Ext launches, and the 2D / peer / device-to-device /
    async copies, are recorded like their plain forms.

Skips without gcc or a hook that predates stream_create on HIP.
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


class TestRocmHipMock(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.core.hookdir import hooks_dir
        if not shutil.which("gcc") or not (hooks_dir() / "libhprofiler_rocm.so").exists():
            raise unittest.SkipTest("needs gcc and the built ROCm hook")
        cls.tmp = tempfile.mkdtemp()
        lib = os.path.join(cls.tmp, "libamdhip64.so")
        exe = os.path.join(cls.tmp, "hip_streams")
        for cmd in (["gcc", "-O1", "-shared", "-fPIC", str(ROOT / "tests/fixtures/hip_mock.c"), "-o", lib,
                     "-lpthread"],
                    ["gcc", "-O1", str(ROOT / "tests/fixtures/hip_streams.c"), "-o", exe, f"-L{cls.tmp}",
                     "-lamdhip64"]):
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                raise unittest.SkipTest(f"cannot build: {r.stderr[:200]}")
        env = {"LD_LIBRARY_PATH": cls.tmp + ":" + os.environ.get("LD_LIBRARY_PATH", "")}
        cls.base = subprocess.run([exe], capture_output=True, text=True, env={**os.environ, **env}).stdout
        from src.core.runner import Runner
        from src.backends import rocm
        if not rocm.ROCmBackend().is_available():
            raise unittest.SkipTest("the rocm backend wants a libamdhip64 in the system paths")
        rf, wf = os.pipe()
        saved = os.dup(1)
        os.dup2(wf, 1)
        try:
            with mock.patch.dict(os.environ, env):
                cls.trace = Runner([exe], backends=["rocm"]).run()
        finally:
            os.dup2(saved, 1)
            os.close(wf)
        cls.out = os.read(rf, 65536).decode()
        os.close(rf)
        cls.spans = sorted(cls.trace.iter_spans(), key=lambda s: s.start_ns)
        if not any(s.tags.get("op") == "stream_create" for s in cls.spans):
            raise unittest.SkipTest("the ROCm hook predates stream_create -- point HPROFILER_HOOKS_DIR at a fresh build")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(getattr(cls, "tmp", ""), ignore_errors=True)

    def named(self, name):
        return [s for s in self.spans if s.name == name]

    def test_output_unchanged(self):
        self.assertIn("order=f1,cb2,f3,", self.out)
        self.assertEqual(self.out.strip(), self.base.strip())

    def test_stream_flags_and_lives(self):
        p = self.named("hipStreamCreateWithPriority")[0]
        self.assertEqual((p.tags["nonblocking"], p.tags["priority"]), ("1", "-1"))
        self.assertIn("reused_stream=1", self.out)
        first, second = [s.tags["stream"] for s in self.named("hipStreamCreate")]
        destroy = self.named("hipStreamDestroy")[0]
        self.assertEqual(destroy.tags["stream"], first)            # the life that ended
        self.assertNotEqual(first, second)
        last = self.named("hipLaunchKernel")[-1]
        self.assertEqual(last.tags["stream"], second)
        lanes = {ln.name for ln in self.trace.store.lane_infos()}
        self.assertIn(f"rocm/stream-{second}", lanes)

    def test_event_lives(self):
        self.assertIn("reused_event=1", self.out)
        ext = self.named("hipExtLaunchKernel")[0]
        start, stop = ext.tags["start_event"], ext.tags["stop_event"]
        recs = [s.tags["event"] for s in self.named("hipEventRecord")]
        self.assertEqual(recs[0], start)                           # the same event, still alive
        self.assertNotEqual(recs[1], stop)                         # a new event at the old address

    def test_host_functions_in_order_on_one_thread(self):
        enq = self.named("hipLaunchHostFunc") + self.named("hipStreamAddCallback")
        runs = sorted((s for s in self.spans if s.tags.get("op") == "host_func" and s.tags.get("side") == "gpu"),
                      key=lambda s: s.start_ns)
        self.assertEqual(len(runs), 3)
        self.assertEqual(len({s.tid for s in runs}), 1)
        by_lid = {s.tags["lid"]: s for s in enq}
        self.assertEqual([by_lid[s.tags["lid"]].name for s in runs],
                         ["hipLaunchHostFunc", "hipStreamAddCallback", "hipLaunchHostFunc"])
        from src.analysis import criticalpath as cp
        spans, preds = cp.build_dependency_graph(self.trace)
        idx = {(s.start_ns, s.tid, s.name): i for i, s in enumerate(spans)}
        r = [idx[(s.start_ns, s.tid, s.name)] for s in runs]
        # each later run waits for the one before it (one callback thread)
        for a, b in zip(r, r[1:]):
            self.assertIn(a, {u for u, _k, _c in preds.get(b, [])})

    def test_more_launches_and_copies(self):
        coop = self.named("hipLaunchCooperativeKernel")[0]
        self.assertEqual(coop.tags["cooperative"], "1")
        want = {"hipMemcpy2D": ("memcpy", "HtoD", "64"), "hipMemcpy2DAsync": ("launch", "HtoD", "64"),
                "hipMemcpyPeer": ("memcpy", "PtoP", "64"), "hipMemcpyPeerAsync": ("launch", "PtoP", "64"),
                "hipMemcpyDtoD": ("memcpy", "DtoD", "32"), "hipMemcpyDtoDAsync": ("launch", "DtoD", "32"),
                "hipMemcpyHtoDAsync": ("launch", "HtoD", "16"), "hipMemcpyDtoHAsync": ("launch", "DtoH", "8")}
        for name, (typ, d, nb) in want.items():
            s = self.named(name)[0]
            self.assertEqual((s.tags["type"], s.tags["dir"], s.tags["bytes"]), (typ, d, nb), name)
            if typ == "launch":                                    # a device span for the async forms
                self.assertTrue(any(x.tags.get("lid") == s.tags["lid"] and x.tags.get("side") == "gpu"
                                    for x in self.spans), name)


if __name__ == "__main__":
    unittest.main()
