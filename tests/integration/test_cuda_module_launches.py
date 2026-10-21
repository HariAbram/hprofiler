"""
Disassembly launch counts on a real GPU: every kernel span carries its
module's id and every disassembly entry counts the spans of its code.

  * Triton's launch pattern (tests/fixtures/cuda_launch_ex.c): PTX loaded
    with cuModuleLoadData, launched 3x through cuLaunchKernelEx from
    dlopen+dlsym (no LD_PRELOAD wrapper sees it; CUPTI's callback does, and
    the function handle in its parameters names the module) and 1x through
    cuLaunchKernel;
  * an ahead-of-time kernel with a shared cudart (tests/fixtures/
    cuda_mini.cu): the hook records the runtime's fat-binary registration,
    so the launch carries the fat binary's id -- the one the extractor
    computes from the executable -- and its code is keyed by it.

`hprofiler run --backend cuda --disasm`, seconds each. Skips without a
compiler, the CUDA hook, CUDA headers or a usable device.
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
HP = [sys.executable, str(ROOT / "hprofiler")]
CUDA_INC = next((d for d in (os.environ.get("CUDA_HOME", "") + "/include", "/usr/local/cuda/include",
                             "/usr/include") if os.path.exists(os.path.join(d, "cuda.h"))), None)


def _profile(tmp, exe, name):
    out = os.path.join(tmp, f"{name}.json")
    r = subprocess.run(HP + ["run", "-b", "cuda", "--no-ui", "--no-summary", "--disasm", "-o", out, "--", exe],
                       capture_output=True, text=True, timeout=180)
    if r.returncode != 0:
        raise AssertionError(r.stdout[-2000:] + r.stderr[-2000:])
    from src.core.trace_io import open_trace
    return open_trace(out)


def _kernels(trace):
    return [s for s in trace.iter_spans(categories=("cuda",)) if s.tags.get("side") == "gpu"]


class TestModuleLaunches(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="hp_cuda_mod_")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        if not (ROOT / "build/lib/libhprofiler_cuda.so").exists():
            self.skipTest("CUDA hook not built")

    def _build(self, cmd, exe):
        if subprocess.run(cmd, capture_output=True, timeout=300).returncode != 0:
            self.skipTest(f"cannot build {exe}")
        if subprocess.run([exe], capture_output=True, timeout=60).returncode != 0:
            self.skipTest("no usable CUDA device")

    def test_dlsym_launch_ex_carries_the_module(self):
        if not shutil.which("gcc") or not CUDA_INC:
            self.skipTest("needs gcc and cuda.h")
        exe = os.path.join(self.tmp, "cuda_launch_ex")
        self._build(["gcc", f"-I{CUDA_INC}", str(ROOT / "tests/fixtures/cuda_launch_ex.c"), "-o", exe,
                     "-lcuda", "-ldl"], exe)
        t = _profile(self.tmp, exe, "launch_ex")
        da = next(v for k, v in t.metadata.device_activity.items() if k.endswith("/cuda"))
        if da.get("status") != "active":
            self.skipTest(f"CUPTI not active: {da}")
        kernels = _kernels(t)
        self.assertEqual(len(kernels), 4)
        mods = {s.tags.get("mod") for s in kernels}
        self.assertEqual(len(mods), 1, mods)
        mod = mods.pop()
        self.assertTrue(mod)
        ex = [s for s in t.iter_spans(categories=("cuda",)) if s.name == "cuLaunchKernelEx"]
        self.assertEqual([(s.tags.get("src"), s.tags.get("mod")) for s in ex], [("cupti_cb", mod)] * 3)
        kd = t.disasm[f"scale_kernel@{mod}"]
        self.assertEqual((kd.launches, kd.launch_basis), (4, "module"))

    def test_aot_kernel_with_shared_cudart_is_keyed_by_its_fat_binary(self):
        if not shutil.which("nvcc"):
            self.skipTest("needs nvcc")
        exe = os.path.join(self.tmp, "cuda_mini_shared")
        self._build(["nvcc", "-cudart", "shared", "-o", exe, str(ROOT / "tests/fixtures/cuda_mini.cu")], exe)
        t = _profile(self.tmp, exe, "aot")
        from src.disasm.extractor import cuda_fatbins, module_id
        ids = [module_id(b) for b in cuda_fatbins(exe)]
        kernels = _kernels(t)
        self.assertEqual(len(kernels), 1)
        mod = kernels[0].tags.get("mod")
        self.assertIn(mod, ids)                    # the hook's id = the extractor's, from the file
        launch = [s for s in t.iter_spans(categories=("cuda",)) if s.name == "cudaLaunchKernel"]
        self.assertEqual([s.tags.get("mod") for s in launch], [mod])
        kd = t.disasm[f"{kernels[0].name}@{mod}"]
        self.assertEqual((kd.launches, kd.launch_basis, kd.arch), (1, "module", "sass"))
        self.assertEqual(kd.source, exe)

    def test_static_cudart_behind_a_wrapper_is_disassembled_every_time(self):
        # The wrappers connect too (the hooks are preloaded into them; a
        # static cudart makes the hook start CUPTI at load): a forking shell
        # script under its own pid (the harness's exitwrap.sh), and env,
        # which exec()s the program under the same pid. The executable that
        # ran the GPU work is chosen, not whichever a set yields first.
        if not shutil.which("nvcc"):
            self.skipTest("needs nvcc")
        exe = os.path.join(self.tmp, "cuda_mini_static")
        self._build(["nvcc", "-cudart", "static", "-o", exe, str(ROOT / "tests/fixtures/cuda_mini.cu")], exe)
        wrap = os.path.join(self.tmp, "wrap.sh")
        Path(wrap).write_text('#!/bin/sh\n"$@"\nexit $?\n')
        os.chmod(wrap, 0o755)
        seen = set()
        for run, wrapper in enumerate([[wrap]] * 3 + [["env", "X=1"]] * 2):
            out = os.path.join(self.tmp, f"static{run}.json")
            r = subprocess.run(HP + ["run", "-b", "cuda", "--no-ui", "--no-summary", "--disasm", "-o", out,
                                     "--"] + wrapper + [exe], capture_output=True, text=True, timeout=180)
            self.assertEqual(r.returncode, 0, r.stderr[-1500:])
            from src.core.trace_io import open_trace
            t = open_trace(out)
            self.assertEqual(len(_kernels(t)), 1)
            seen.add(tuple(sorted((k, kd.source, kd.launches, kd.launch_basis, len(kd.lines))
                                  for k, kd in t.disasm.items() if k == kd.key)))
        self.assertEqual(len(seen), 1, seen)
        entries = seen.pop()
        self.assertTrue(entries)
        self.assertEqual({e[1] for e in entries}, {exe})
        self.assertEqual([e[2:4] for e in entries if e[2]], [(1, "name")])   # by name: static runtime


if __name__ == "__main__":
    unittest.main()
