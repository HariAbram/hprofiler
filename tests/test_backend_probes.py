"""
Backend usability (src/backends/probes.py, Backend.usability, `hprofiler
backends` and `--backend auto`): installed is not usable --

  * CUDA with a driver/library mismatch (cuInit fails, e.g. 803/804) or no
    device; ROCm without /dev/kfd or a GPU agent; LIKWID without PMU access;
    OpenCL without a device; NCCL without a usable CUDA GPU;
  * auto-detection picks usable backends only and says which installed ones
    it skipped; an explicitly requested unusable backend is used, with a
    warning.

The probes' own subprocess and file access are mocked: the hardware states
are not reproducible on demand.
"""
import importlib.machinery
import importlib.util
import io
import subprocess
import sys
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.backends import probes


def _cli():
    loader = importlib.machinery.SourceFileLoader("hprofiler_cli", str(ROOT / "hprofiler"))
    spec = importlib.util.spec_from_loader("hprofiler_cli", loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


class TestProbes(unittest.TestCase):
    def setUp(self):
        for f in (probes.cuda, probes.rocm, probes.opencl, probes.likwid):
            f.cache_clear()

    def test_cuda_driver_mismatch_is_not_usable(self):
        with mock.patch.object(probes, "_run_probe", return_value="init 804 CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE".split()):
            ok, why = probes.cuda()
        self.assertFalse(ok)
        self.assertIn("804", why)

    def test_cuda_states(self):
        with mock.patch.object(probes, "_run_probe", return_value=["count", "0", "0"]):
            self.assertEqual(probes.cuda(), (False, "no CUDA device"))
        probes.cuda.cache_clear()
        with mock.patch.object(probes, "_run_probe", return_value=["count", "0", "2"]):
            self.assertEqual(probes.cuda(), (True, "2 CUDA device(s)"))
        probes.cuda.cache_clear()
        with mock.patch.object(probes, "_run_probe", return_value=["timeout"]):
            self.assertFalse(probes.cuda()[0])

    def test_rocm_without_kfd_or_gpu_agent(self):
        with mock.patch("os.path.exists", return_value=False):
            ok, why = probes.rocm()
        self.assertFalse(ok)
        self.assertIn("/dev/kfd", why)
        probes.rocm.cache_clear()
        with mock.patch("os.path.exists", return_value=True), mock.patch("os.access", return_value=True), \
                mock.patch("glob.glob", return_value=[]):
            self.assertEqual(probes.rocm(), (False, "no GPU agent in /sys/class/kfd topology"))

    def test_likwid_without_pmu_access(self):
        done = subprocess.CompletedProcess([], 1, stdout="", stderr="Cannot use performance monitoring\n")
        with mock.patch("shutil.which", return_value="/usr/bin/likwid-perfctr"), \
                mock.patch("subprocess.run", return_value=done):
            ok, why = probes.likwid("FLOPS_DP")
        self.assertFalse(ok)
        self.assertIn("no PMU access", why)

    def test_opencl_without_device(self):
        with mock.patch.object(probes, "_run_probe", return_value=["devices", "0", "0"]):
            self.assertEqual(probes.opencl(), (False, "no OpenCL device"))


class TestAutoDetection(unittest.TestCase):
    def _with_usability(self, states):
        from src.backends import ALL_BACKENDS
        patches = [mock.patch.object(cls, "usability", lambda self, _s=states.get(name, ("missing", "")): _s)
                   for name, cls in ALL_BACKENDS.items()]
        patches += [mock.patch.object(cls, "is_available", lambda self, _s=states.get(name, ("missing", "")):
                                      _s[0] != "missing") for name, cls in ALL_BACKENDS.items()]
        return patches

    def test_auto_picks_usable_only_and_names_the_skipped(self):
        cli = _cli()
        patches = self._with_usability({"openmp": ("usable", ""), "cuda": ("installed", "cuInit failed: 804"),
                                        "likwid": ("installed", "no PMU access"), "mpi": ("usable", "")})
        for p in patches:
            p.start()
        try:
            err = io.StringIO()
            with redirect_stderr(err):
                chosen = cli._resolve_backends("auto")
        finally:
            for p in patches:
                p.stop()
        self.assertEqual(sorted(chosen), ["mpi", "openmp"])
        self.assertIn("cuda (cuInit failed: 804)", err.getvalue())
        self.assertIn("likwid (no PMU access)", err.getvalue())

    def test_explicit_unusable_backend_is_used_with_a_warning(self):
        cli = _cli()
        patches = self._with_usability({"cuda": ("installed", "cuInit failed: 804")})
        for p in patches:
            p.start()
        try:
            err = io.StringIO()
            with redirect_stderr(err):
                chosen = cli._resolve_backends("cuda")
        finally:
            for p in patches:
                p.stop()
        self.assertEqual(chosen, ["cuda"])
        self.assertIn("installed but not usable: cuInit failed: 804", err.getvalue())

    def test_backends_command_shows_the_reason(self):
        r = subprocess.run([sys.executable, str(ROOT / "hprofiler"), "backends"], capture_output=True,
                           text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("auto-detection uses the usable ones", r.stdout)
        self.assertRegex(r.stdout, r"\b(usable|installed|missing)\b")


if __name__ == "__main__":
    unittest.main()
