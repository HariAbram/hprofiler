"""
AdaptiveCpp dependencies through the real hooks (tests/fixtures/sycl_deps.cpp,
built with acpp for the generic SSCP target; src/analysis/semantics/sycl.py):

  cuda   lowered to cudaEventRecord + cudaStreamWaitEvent: each dependent
         kernel's device span waits for the kernel it depends on, across
         the two streams;
  ocl    one in-order queue per device: the kernels chain in queue order;
         its USM copies (clEnqueueMemcpyINTEL, an extension function the
         hook wraps) are recorded, and every wait list names its events;
  omp    host-device kernels: no edge between kernels of different queues
         -- the runtime resolves those dependencies itself -- and the trace
         says so (sycl_host_kernel_calls);
  mixed  a CUDA copy depending on a host-device kernel: a stream host
         function waits for it inside the runtime -- recorded, but nothing
         links it to the kernel, and the trace says so
         (acpp_host_functions).

HIP is not run (no AMD GPU). Skips without acpp (ACPP=, else
/opt/AdaptiveCpp/bin/acpp or acpp on PATH), the built hooks, or a device.
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


def _acpp():
    for c in (os.environ.get("ACPP"), "/opt/AdaptiveCpp/bin/acpp", shutil.which("acpp")):
        if c and os.access(c, os.X_OK):
            return c
    return None


class TestSyclDeps(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.core.hookdir import hooks_dir
        acpp = _acpp()
        if not acpp:
            raise unittest.SkipTest("needs AdaptiveCpp's acpp")
        if not all((hooks_dir() / f"libhprofiler_{h}.so").exists() for h in ("cuda", "opencl")):
            raise unittest.SkipTest("needs the built CUDA and OpenCL hooks")
        cls.tmp = tempfile.mkdtemp()
        cls.exe = os.path.join(cls.tmp, "sycl_deps")
        r = subprocess.run([acpp, "-O2", str(ROOT / "tests/fixtures/sycl_deps.cpp"), "-o", cls.exe],
                           capture_output=True, text=True, timeout=900)
        if r.returncode != 0:
            raise unittest.SkipTest(f"cannot build the fixture: {r.stderr[-300:]}")
        cls.runs = {}

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(getattr(cls, "tmp", ""), ignore_errors=True)

    def run_mode(self, mode, backends):
        if mode in self.runs:
            return self.runs[mode]
        base = subprocess.run([self.exe, mode], capture_output=True, text=True, timeout=300)
        if base.returncode != 0 or base.stdout.startswith("nodev"):
            self.skipTest(f"no device for {mode}: {base.stdout.strip()[:80]}")
        from src.core.runner import Runner
        from src.analysis import criticalpath as cp
        rf, wf = os.pipe()
        saved = os.dup(1)
        os.dup2(wf, 1)
        try:
            trace = Runner([self.exe, mode], backends=backends).run()
        finally:
            os.dup2(saved, 1)
            os.close(wf)
        out = os.read(rf, 65536).decode()
        os.close(rf)
        self.assertIn(f"ok {mode}", out)
        spans, preds = cp.build_dependency_graph(trace)
        self.runs[mode] = (trace, spans, preds, cp.dependency_diagnostics(trace))
        return self.runs[mode]

    @staticmethod
    def kernels(spans, cat):
        return sorted((i for i, s in enumerate(spans) if s.category.value == cat and s.tags.get("side") == "gpu"
                       and s.tags.get("type") == "kernel"), key=lambda i: spans[i].start_ns)

    def test_cuda_event_waits_between_streams(self):
        _t, spans, preds, diag = self.run_mode("cuda", ["cuda"])
        k = self.kernels(spans, "cuda")
        self.assertEqual(len(k), 4)
        streams = [spans[i].tags["stream"] for i in k]
        self.assertEqual((streams[0], streams[1]), (streams[2], streams[3]))       # kA, kC on q1; kB, kD on q2
        self.assertNotEqual(streams[0], streams[1])
        for src, dst in ((k[0], k[1]), (k[2], k[3])):                              # kA -> kB, kC -> kD
            self.assertIn(src, {u for u, _k, _c in preds.get(dst, [])})
        self.assertTrue(any(s.name == "cudaStreamWaitEvent" for s in spans))
        self.assertNotIn("sycl_host_kernel_calls", diag)
        self.assertNotIn("acpp_host_functions", diag)

    def test_opencl_queue_order_and_usm_copies(self):
        _t, spans, preds, _diag = self.run_mode("ocl", ["opencl"])
        k = self.kernels(spans, "opencl")
        self.assertEqual(len(k), 4)
        self.assertEqual(len({spans[i].tags["queue"] for i in k}), 1)               # one in-order queue
        for a, b in zip(k, k[1:]):
            self.assertIn(a, {u for u, _k, _c in preds.get(b, [])})
        cpy = [s for s in spans if s.name == "clEnqueueMemcpyINTEL"]
        self.assertTrue(cpy)
        self.assertTrue(all(s.tags.get("ev") for s in cpy))
        self.assertFalse([s.name for s in spans if s.tags.get("wait_unknown")])

    def test_host_device_dependencies_are_reported_missing(self):
        _t, spans, preds, diag = self.run_mode("omp", ["opencl"])
        host = [i for i, s in enumerate(spans) if s.category.value == "sycl"]
        self.assertTrue(host)
        hs = set(host)
        for v in host:                                                             # only their threads' order
            for u, kind, _c in preds.get(v, []):
                if u in hs:
                    self.assertEqual((kind, spans[u].tid), ("sequential", spans[v].tid))
        self.assertEqual(diag["sycl_host_kernel_calls"], len(host))
        from src.analysis import criticalpath as cp
        self.assertTrue(any("host-device SYCL kernel call" in w for w in cp.dependency_warnings(diag)))

    def test_a_dependency_across_backends_waits_unrecorded(self):
        _t, spans, preds, diag = self.run_mode("mixed", ["cuda", "opencl"])
        hf = [i for i, s in enumerate(spans) if s.tags.get("op") == "host_func" and s.tags.get("side") == "gpu"]
        self.assertTrue(hf)
        host = {i for i, s in enumerate(spans) if s.category.value == "sycl"}
        self.assertTrue(host)
        for v in hf:                                                               # nothing from the host kernel
            self.assertFalse({u for u, _k, _c in preds.get(v, [])} & host)
        # the copy after it on the stream waits for it
        h2d = [i for i, s in enumerate(spans) if s.name == "memcpy HtoD" and s.tags.get("side") == "gpu"]
        self.assertTrue(any(v in {u for u, _k, _c in preds.get(c, [])} for v in hf for c in h2d))
        self.assertGreaterEqual(diag.get("acpp_host_functions", 0), 1)
        self.assertGreater(diag.get("sycl_host_kernel_calls", 0), 0)


if __name__ == "__main__":
    unittest.main()
