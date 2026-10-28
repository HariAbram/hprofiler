"""
OpenCL kernel names through the real OpenCL hook (tests/fixtures/ocl_names.c
on any OpenCL device, e.g. the Intel CPU runtime): a kernel whose name is
longer than the hook's former 256-byte buffer must be reported under its
own full name on both the host enqueue span and the device span -- never as
the previously launched kernel. Skips without gcc, OpenCL headers or a
device.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from src.core.hookdir import hooks_dir  # noqa: E402  (HPROFILER_HOOKS_DIR, else build/lib)

LONG = "long_kernel_" + "abcdefghij" * 32


class TestOpenCLKernelNames(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.exe = os.path.join(cls.tmp, "ocl_names")
        if not shutil.which("gcc") or not (hooks_dir() / "libhprofiler_opencl.so").exists():
            raise unittest.SkipTest("needs gcc and the built OpenCL hook")
        r = subprocess.run(["gcc", "-O1", str(ROOT / "tests/fixtures/ocl_names.c"), "-o", cls.exe, "-lOpenCL"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise unittest.SkipTest(f"cannot build the fixture: {r.stderr[:200]}")
        probe = subprocess.run([cls.exe], capture_output=True, text=True, timeout=120)
        if probe.returncode != 0:
            raise unittest.SkipTest(f"no usable OpenCL device ({probe.stdout.strip()})")
        from src.core.runner import Runner
        cls.trace = Runner([cls.exe], backends=["opencl"]).run()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _counts(self, side):
        return Counter(s.name for s in self.trace.iter_spans(categories=("opencl",))
                       if s.tags.get("type") == "kernel" and s.tags.get("side") == side)

    def test_long_kernel_name_on_device_spans(self):
        self.assertEqual(self._counts("gpu"), Counter({"short_k": 3, LONG: 2}))

    def test_long_kernel_name_on_host_spans(self):
        self.assertEqual(self._counts("cpu"), Counter({"short_k": 3, LONG: 2}))


if __name__ == "__main__":
    unittest.main()
