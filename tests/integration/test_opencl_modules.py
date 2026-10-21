"""
Disassembly keyed by module through the real OpenCL hook
(tests/fixtures/ocl_variants.c on any OpenCL device whose binaries are ELF,
e.g. the Intel CPU runtime): two programs that both define `variant_k` keep
two entries, each launch carries the module id (mod=) of the binary it ran,
and every kernel of a multi-kernel program is disassembled. Also checks the
stage= label on the builds. Skips without gcc, OpenCL headers or a device.
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


class TestOpenCLModules(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.exe = os.path.join(cls.tmp, "ocl_variants")
        if not shutil.which("gcc") or not (ROOT / "build/lib/libhprofiler_opencl.so").exists():
            raise unittest.SkipTest("needs gcc and the built OpenCL hook")
        r = subprocess.run(["gcc", "-O1", str(ROOT / "tests/fixtures/ocl_variants.c"), "-o", cls.exe,
                            "-lOpenCL"], capture_output=True, text=True)
        if r.returncode != 0:
            raise unittest.SkipTest(f"cannot build the fixture: {r.stderr[:200]}")
        probe = subprocess.run([cls.exe], capture_output=True, text=True, timeout=120)
        if probe.returncode != 0:
            raise unittest.SkipTest(f"no usable OpenCL device ({probe.stdout.strip()})")
        from src.core.runner import Runner
        runner = Runner([cls.exe], backends=["opencl"], collect_disasm=True)
        cls.trace = runner.run()
        if runner._disasm_thread is not None:
            runner._disasm_thread.join(timeout=300)
        cls.kernels = [s for s in cls.trace.iter_spans(categories=("opencl",))
                       if s.tags.get("type") == "kernel"]
        cls.loads = [s for s in cls.trace.iter_spans(categories=("jit",))
                     if s.tags.get("type") == "jit_load"]
        if not cls.loads:
            raise unittest.SkipTest("this OpenCL device returns no ELF binaries to disassemble")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _mods(self, name, side):
        return Counter(s.tags.get("mod") for s in self.kernels
                       if s.name == name and s.tags.get("side") == side)

    def test_every_launch_names_its_module(self):
        load_mods = {s.tags.get("mod") for s in self.loads}
        self.assertEqual(len(load_mods), 3)
        for side in ("cpu", "gpu"):
            mods = self._mods("variant_k", side)
            self.assertEqual(sorted(mods.values()), [2, 3], side)
            self.assertTrue(set(mods) <= load_mods, side)
            self.assertEqual(set(self._mods("pair_one", side)), set(self._mods("pair_two", side)))

    def test_variants_keep_separate_disassembly(self):
        variants = [kd for k, kd in self.trace.disasm.items() if kd.name == "variant_k" and k == kd.key]
        self.assertEqual(len(variants), 2, sorted(self.trace.disasm))
        self.assertEqual(sorted(kd.launches for kd in variants), [2, 3])
        self.assertNotEqual([ln.text for ln in variants[0].lines], [ln.text for ln in variants[1].lines])
        gpu_mods = self._mods("variant_k", "gpu")
        for kd in variants:
            self.assertEqual(kd.launches, gpu_mods[kd.module])
            found, how = self.trace.disasm_for("variant_k", kd.module)
            self.assertIs(found, kd)
            self.assertEqual(how, "module")
        _kd, how = self.trace.disasm_for("variant_k")
        self.assertEqual(how, "name-ambiguous")          # a name-only lookup is flagged

    def test_every_kernel_of_a_program_is_disassembled(self):
        for name in ("pair_one", "pair_two"):
            kd, how = self.trace.disasm_for(name)
            self.assertIsNotNone(kd, name)
            self.assertEqual(how, "module")
            self.assertEqual(kd.launches, 1)

    def test_builds_from_source_are_labelled(self):
        builds = [s for s in self.trace.iter_spans(categories=("jit",)) if s.name == "clBuildProgram"]
        self.assertEqual(len(builds), 3)
        self.assertEqual({s.tags.get("stage") for s in builds}, {"source"})

    def test_round_trip_keeps_module_keys(self):
        from src.core.trace_io import open_trace, write_json_export
        out = Path(self.tmp) / "json-only"
        out.mkdir(exist_ok=True)
        p = out / "t.json"
        write_json_export(self.trace, str(p))
        back = open_trace(str(p))
        self.assertEqual(sorted(k for k, kd in back.disasm.items() if k == kd.key),
                         sorted(k for k, kd in self.trace.disasm.items() if k == kd.key))
        kd, how = back.disasm_for("variant_k", next(iter(self._mods("variant_k", "gpu"))))
        self.assertEqual(how, "module")


if __name__ == "__main__":
    unittest.main()
