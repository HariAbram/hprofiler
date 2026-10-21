"""
Disassembly keyed by module identity, and the JIT stage labels.

  * the module id: the hooks' hp_fnv1a64 (tests/native/hp_module_test.c,
    also run under ASan+UBSan) equals extractor.module_id() on the same
    file; the C kernel table keeps names whole and forgets an unloaded
    module's functions -- the CUDA and ROCm hooks name launches from it, and
    neither runtime is available here, so this is their only executed test;
  * Trace.disasm_for(): exact module matches, own-binary code by symbol,
    and name-only matches reported as such ("name", "name-ambiguous");
    Trace.disasm_listing() lists each variant;
  * serialization: name@module keys, module and launch count survive JSON
    and the store; traces written before (name keys, no module) still load;
  * the runner: launches per (name, module) counted, saved paths' module
    ids and launched names handed to the extractor, and no "<jit-kernel>"
    alias attaching some other kernel's code;
  * the extractor on two real ELF objects that both define `variant_k`
    (what the OpenCL hook saves): two entries, each with its own code;
  * gpu_activity.assemble() copies mod= from the host launch to the
    native device span (synthetic CUPTI records);
  * analysis/jit_coverage.py: the stage of every hook's jit spans, and the
    "back-end / load only" note.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.core.events import Category, SpanEvent
from src.core.trace import Trace, TraceMetadata
from src.disasm.extractor import DisasmLine, KernelDisasm, collect_disasm, module_id

NATIVE = ROOT / "tests" / "native"


def _kd(name, module="", launches=0, text=("add r1, r2",), arch="x86-64"):
    lines = [DisasmLine(addr=i, mnemonic=t.split()[0], operands=" ".join(t.split()[1:]))
             for i, t in enumerate(text)]
    return KernelDisasm(name=name, arch=arch, source="/tmp/x", lines=lines, module=module,
                        launches=launches)


def _span(name, cat, start=0, dur=10, **tags):
    return SpanEvent(name=name, category=Category(cat), start_ns=start, duration_ns=dur, pid=1, tid=1,
                     tags={k: str(v) for k, v in tags.items()})


class TestNativeModuleTable(unittest.TestCase):
    def _build(self, d, flags=()):
        exe = os.path.join(d, "hp_module_test")
        r = subprocess.run(["gcc", "-O1", "-g", "-Wall", "-Wextra", *flags, "-pthread",
                            str(NATIVE / "hp_module_test.c"), "-o", exe], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        self.assertNotIn("warning", r.stderr)
        return exe

    @unittest.skipUnless(shutil.which("gcc"), "needs gcc")
    def test_table_and_hash(self):
        with tempfile.TemporaryDirectory() as d:
            exe = self._build(d)
            p = subprocess.run([exe], capture_output=True, text=True, timeout=120)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            self.assertIn("RESULT fails=0", p.stdout)
            blob = Path(d) / "blob.bin"
            blob.write_bytes(bytes(range(256)) * 37 + b"\x7fELF tail")
            out = subprocess.run([exe, str(blob)], capture_output=True, text=True).stdout
            self.assertEqual(out.strip(), "MOD " + module_id(blob.read_bytes()))

    @unittest.skipUnless(shutil.which("gcc"), "needs gcc")
    def test_elf_image_size_of_real_files(self):
        # what the CUDA/ROCm hooks save of an image passed without a length:
        # the whole file -- a cubin's program headers follow its section
        # headers (the hooks used to stop there, and cuobjdump rejected it)
        with tempfile.TemporaryDirectory() as d:
            exe = self._build(d)
            files = [Path(exe)]
            if shutil.which("ptxas"):
                ptx = Path(d) / "k.ptx"
                ptx.write_text(".version 7.0\n.target sm_75\n.address_size 64\n"
                               ".visible .entry k(.param .u64 p)\n{\n\tret;\n}\n")
                cubin = Path(d) / "k.cubin"
                if subprocess.run(["ptxas", "-arch=sm_75", str(ptx), "-o", str(cubin)],
                                  capture_output=True).returncode == 0:
                    files.append(cubin)
            if shutil.which("nvcc"):            # a fat binary: header + payload
                fat = Path(d) / "k.fatbin"
                if subprocess.run(["nvcc", "-fatbin", "-arch=sm_75", "-o", str(fat),
                                   str(ROOT / "tests/fixtures/cuda_mini.cu")], capture_output=True).returncode == 0:
                    files.append(fat)
            for f in files:
                out = subprocess.run([exe, str(f)], capture_output=True, text=True).stdout
                self.assertIn(f"SIZE {f.stat().st_size}\n", out, f)

    @unittest.skipUnless(shutil.which("gcc"), "needs gcc")
    def test_table_under_asan(self):
        with tempfile.TemporaryDirectory() as d:
            try:
                exe = self._build(d, ("-fsanitize=address,undefined", "-fno-sanitize-recover=all"))
            except AssertionError:
                self.skipTest("ASan not available")
            p = subprocess.run([exe], capture_output=True, text=True, timeout=300,
                               env=dict(os.environ, ASAN_OPTIONS="detect_leaks=0"))
            if "ASan runtime does not come first" in p.stderr or "Shadow memory range" in p.stderr:
                self.skipTest("ASan cannot run here")
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr[-3000:])
            self.assertIn("RESULT fails=0", p.stdout)

    def test_python_hash_reference_values(self):
        self.assertEqual(module_id(b""), "cbf29ce484222325")
        self.assertEqual(module_id(b"a"), "af63dc4c8601ec8c")
        self.assertEqual(module_id(b"foobar"), "85944171f73967e8")


class TestLookup(unittest.TestCase):
    def _trace(self, *kds):
        t = Trace(TraceMetadata())
        for kd in kds:
            t.add_disasm(kd)
        return t

    def test_variants_are_separate_entries(self):
        a, b = _kd("k", "aa", 2, ("add r1, r2",)), _kd("k", "bb", 3, ("mul r1, r2",))
        t = self._trace(a, b)
        self.assertEqual(sorted(t.disasm), ["k@aa", "k@bb"])
        self.assertEqual(t.disasm_for("k", "aa"), (a, "module"))
        self.assertEqual(t.disasm_for("k", "bb"), (b, "module"))
        self.assertEqual(t.disasm_for("k@aa"), (a, "module"))
        # no module on the lookup: most-launched variant, flagged
        self.assertEqual(t.disasm_for("k"), (b, "name-ambiguous"))
        self.assertEqual(t.disasm_listing(["k", "other"]), ["k", "k@bb", "k@aa", "other"])

    def test_one_module_with_linked_launches_is_exact(self):
        a = _kd("k", "aa", 5)
        t = self._trace(a)
        self.assertEqual(t.disasm_for("k"), (a, "module"))
        self.assertEqual(t.disasm_listing(["k"]), ["k"])

    def test_unlinked_launches_are_a_name_match(self):
        a = _kd("k", "aa", 0)
        self.assertEqual(self._trace(a).disasm_for("k"), (a, "name"))

    def test_launch_module_without_its_code_falls_back_to_name(self):
        a = _kd("k", "aa", 4)
        self.assertEqual(self._trace(a).disasm_for("k", "cc"), (a, "name"))
        self.assertEqual(self._trace().disasm_for("k", "cc"), (None, "none"))

    def test_own_binary_code_is_matched_by_symbol(self):
        cpu = _kd("omp_barrier")
        t = self._trace(cpu)
        self.assertEqual(t.disasm_for("omp_barrier"), (cpu, "symbol"))
        self.assertTrue(Trace.DISASM_FOUND_HOW["name"])

    def test_long_names_keep_the_old_prefix_alias(self):
        long = "n" * 300
        kd = _kd(long)
        t = self._trace(kd)
        self.assertIs(t.disasm_for(long[:255])[0], kd)
        self.assertEqual(t.disasm_listing([long]), [long])     # alias not listed twice


class TestSerialization(unittest.TestCase):
    def _trace(self):
        t = Trace(TraceMetadata(command="x"))
        t.add(_span("k", "opencl", side="gpu", mod="aa"))
        t.add_disasm(_kd("k", "aa", 2, ("add r1, r2",)))
        t.add_disasm(_kd("k", "bb", 3, ("mul r1, r2",)))
        t.add_disasm(_kd("cpu_fn"))
        return t

    def _check(self, back):
        keys = sorted(k for k, kd in back.disasm.items() if k == kd.key)
        self.assertEqual(keys, ["cpu_fn", "k@aa", "k@bb"])
        kd, how = back.disasm_for("k", "bb")
        self.assertEqual((kd.module, kd.launches, kd.lines[0].mnemonic, how), ("bb", 3, "mul", "module"))

    def test_json_round_trip(self):
        from src.core.trace_io import open_trace, write_json_export
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.json"
            write_json_export(self._trace(), str(p))
            self._check(open_trace(str(p)))

    def test_store_round_trip(self):
        from src.core.trace_io import create_disk_trace, open_trace
        with tempfile.TemporaryDirectory() as d:
            cap = create_disk_trace(str(Path(d) / "t.hpstore"))
            t = Trace(TraceMetadata(command="x"), store=cap.store)
            src = self._trace()
            for s in src.iter_spans():
                t.add(s)
            for kd in src.disasm.values():
                t.add_disasm(kd)
            t.finalize()
            t.save()
            self._check(open_trace(str(Path(d) / "t.hpstore")))

    def test_old_name_keyed_entries_load(self):
        from src.core.trace_io import restore_disasm
        t = Trace(TraceMetadata())
        restore_disasm(t, {"benchmark_func": {"arch": "x86-64", "source": "/tmp/a.bin",
                                              "lines": [{"addr": 0, "mnemonic": "ret"}]}})
        kd, how = t.disasm_for("benchmark_func")
        self.assertEqual((kd.name, kd.module, how), ("benchmark_func", "", "symbol"))


class TestRunnerCollect(unittest.TestCase):
    def _run(self, spans, returned):
        from src.core import runner
        t = Trace(TraceMetadata(command="./app"))
        for s in spans:
            t.add(s)
        seen = {}

        def fake(command, backends, jit_spans, *a, **kw):
            seen.update(kw, jit_spans=jit_spans)
            return dict(returned)
        with mock.patch("src.disasm.extractor.collect_disasm", fake):
            runner._collect_disasm(t, ["./app"], ["cuda", "opencl"])
        return t, seen

    def test_launches_counted_and_module_ids_passed(self):
        kd = _kd("variant_k", "aa", arch="sass")
        t, seen = self._run([
            _span("ocl_kernel.bin", "jit", type="jit_load", stage="load", path="/tmp/hprofiler_ocl_1_0.bin",
                  mod="aa"),
            _span("cuModuleLoadData", "jit", type="jit_compile", stage="backend",
                  path="/tmp/hprofiler_cubin_1_0.bin", mod="cc"),
            *[_span("variant_k", "opencl", start=i, side="gpu", mod="aa") for i in range(3)],
            _span("variant_k", "opencl", side="cpu", mod="aa"),      # host span: not a launch count
            _span("variant_k", "opencl", side="gpu", mod="bb"),
        ], {"variant_k@aa": kd})
        self.assertEqual(seen["module_ids"], {"/tmp/hprofiler_ocl_1_0.bin": "aa",
                                              "/tmp/hprofiler_cubin_1_0.bin": "cc"})
        self.assertEqual(seen["launched_by_module"], {"aa": {"variant_k"}, "bb": {"variant_k"}})
        self.assertEqual(seen["jit_spans"][0]["mod"], "aa")
        self.assertEqual(t.disasm["variant_k@aa"].launches, 3)

    def test_no_jit_kernel_alias(self):
        gpu = _kd("_Z6kernelPf", arch="sass")
        t, _ = self._run([_span("<jit-kernel>", "cuda", side="gpu")], {"_Z6kernelPf": gpu})
        self.assertNotIn("<jit-kernel>", t.disasm)
        self.assertEqual(t.disasm_for("<jit-kernel>"), (None, "none"))

    def test_aot_kernels_without_module_ids_are_counted_by_name(self):
        # BabelStream CUDA (AoT, shared cudart): launches carry no mod=; the
        # binary's code has no module -- 20 device spans, not 0
        copy, nstream = _kd("_Z11copy_kernelIdEvPKT_PS0_", arch="sass"), _kd("_Z14nstreamv", arch="sass")
        t, _ = self._run([_span("_Z11copy_kernelIdEvPKT_PS0_", "cuda", start=i, side="gpu") for i in range(20)]
                         + [_span("cudaLaunchKernel", "cuda", side="cpu")],
                         {copy.key: copy, nstream.key: nstream})
        got = {k: (kd.launches, kd.launch_basis) for k, kd in t.disasm.items()}
        self.assertEqual(got, {"_Z11copy_kernelIdEvPKT_PS0_": (20, "name"), "_Z14nstreamv": (0, "name")})


class TestDisasmBinaries(unittest.TestCase):
    """Which executables the ahead-of-time GPU code is read from
    (runner.disasm_binaries): a static-cudart run behind a wrapper has two
    connected processes, only one of which ran GPU work."""

    def test_the_gpu_process_wins_whatever_the_hash_seed(self):
        code = ("import sys; sys.path.insert(0, %r)\n"
                "from src.core.runner import disasm_binaries\n"
                "exes = {981128: ['/bin/bash'], 981141: ['/opt/app/cuda-static'], 981150: ['/bin/sh'],\n"
                "        981160: ['/usr/bin/env', '/opt/app/cuda-static']}\n"
                "flat = {e for v in exes.values() for e in v}\n"
                "print(disasm_binaries(exes, {981141}), disasm_binaries(exes, {981160}),\n"
                "      disasm_binaries({981128: ['/bin/bash'], 981150: ['/bin/sh']}, set()),\n"
                "      disasm_binaries(flat, set()))\n") % str(ROOT)
        outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                               env=dict(os.environ, PYTHONHASHSEED=str(seed))).stdout
                for seed in range(21)}
        self.assertEqual(len(outs), 1, outs)
        self.assertEqual(outs.pop().strip(),
                         # the GPU process; env exec()ing the program: the program first;
                         # no GPU process: all; older callers (a set): all, sorted
                         "['/opt/app/cuda-static'] ['/opt/app/cuda-static', '/usr/bin/env'] "
                         "['/bin/bash', '/bin/sh'] ['/bin/bash', '/bin/sh', '/opt/app/cuda-static', '/usr/bin/env']")

    def test_runner_passes_the_gpu_processes_executables(self):
        from src.core import runner
        t = Trace(TraceMetadata(command="env"))
        t.add(SpanEvent(name="_Z1kv", category=Category("cuda"), start_ns=0, duration_ns=5, pid=981141,
                        tid=1, tags={"side": "gpu", "rt": "cuda"}))
        seen = {}
        with mock.patch("src.disasm.extractor.collect_disasm",
                        lambda *a, **kw: seen.update(kw) or {}):
            runner._collect_disasm(t, ["env", "X=1", "./app"], ["cuda"],
                                   real_binary_paths={981128: ["/bin/bash"], 981141: ["/opt/app/cuda-static"]})
        self.assertEqual(seen["real_binaries"], ["/opt/app/cuda-static"])

    def test_code_of_several_executables_is_merged_by_name_and_module(self):
        from src.disasm import extractor
        with tempfile.TemporaryDirectory() as d:
            a, b = os.path.join(d, "a"), os.path.join(d, "b")
            for p in (a, b):
                Path(p).write_bytes(b"\x7fELF")
            code = {a: [_kd("k", arch="sass"), _kd("only_a", arch="sass")],
                    b: [_kd("k", arch="sass"), _kd("k", "mm", text=("mul r1, r2",), arch="sass"),
                        _kd("only_b", arch="sass")]}
            for exe, kds in code.items():
                for kd in kds:
                    kd.source = exe
            with mock.patch.object(extractor, "_cuda_aot_code",
                                   lambda exe, _l: {kd.key: kd for kd in code[exe]}):
                res = collect_disasm(["/usr/bin/env"], ["cuda"], [], profiled_pid=999999999,
                                     real_binaries=[a, b])
        self.assertEqual({k: kd.source for k, kd in res.items()},
                         {"k": a, "only_a": a, "k@mm": b, "only_b": b})


class TestLaunchAttribution(unittest.TestCase):
    """extractor.attribute_launches: every kernel that ran gets its count."""

    def _attr(self, entries, launches):
        from src.disasm.extractor import attribute_launches
        attribute_launches(entries, launches)
        return [(kd.key, kd.launches, kd.launch_basis) for kd in entries]

    def test_module_less_entries_by_name(self):
        self.assertEqual(self._attr([_kd("a", arch="sass"), _kd("b", arch="sass")],
                                    {("a", ""): 7, ("b", ""): 501}),
                         [("a", 7, "name"), ("b", 501, "name")])

    def test_module_entries_unchanged(self):
        self.assertEqual(self._attr([_kd("k", "aa", arch="ptx"), _kd("n", "aa", arch="ptx")],
                                    {("k", "aa"): 20}),
                         [("k@aa", 20, "module"), ("n@aa", 0, "module")])

    def test_two_modules_defining_one_name(self):
        entries = [_kd("k", "aa", arch="sass"), _kd("k", "bb", arch="sass")]
        self.assertEqual(self._attr(entries, {("k", "aa"): 2, ("k", "bb"): 3}),
                         [("k@aa", 2, "module"), ("k@bb", 3, "module")])
        # launches without a module id cannot be split between them: unknown, not 0
        self.assertEqual(self._attr(entries, {("k", ""): 5}),
                         [("k@aa", None, "unknown"), ("k@bb", None, "unknown")])

    def test_one_module_whose_launches_carry_no_id(self):
        # older traces of a cuLaunchKernelEx program (Triton): one module per name
        self.assertEqual(self._attr([_kd("add_kernel", "cc", arch="sass")], {("add_kernel", ""): 4}),
                         [("add_kernel@cc", 4, "name")])

    def test_binary_code_and_a_module_with_one_name(self):
        # spans with the module's id are its own; the rest are the binary's
        self.assertEqual(self._attr([_kd("k", arch="sass"), _kd("k", "aa", arch="sass")],
                                    {("k", "aa"): 2, ("k", ""): 3, ("k", "zz"): 1}),
                         [("k", 4, "name"), ("k@aa", 2, "module")])

    def test_mangled_name_and_cpu_code(self):
        k = _kd("vadd", arch="sass")
        k.mangled_name = "_Z4vaddPf"
        self.assertEqual(self._attr([k, _kd("main")], {("_Z4vaddPf", ""): 3}),
                         [("vadd", 3, "name"), ("main", 0, "")])

    def test_unknown_count_round_trips_as_null(self):
        from src.core.trace_io import disasm_from_json, disasm_to_json
        kd = _kd("k", "aa", arch="sass")
        kd.launches, kd.launch_basis = None, "unknown"
        raw = disasm_to_json(kd)
        self.assertIsNone(raw["launches"])
        back = disasm_from_json(kd.key, json.loads(json.dumps(raw)))
        self.assertEqual((back.launches, back.launch_basis), (None, "unknown"))
        old = disasm_from_json("k", {"arch": "sass", "launches": 3})       # before launchBasis
        self.assertEqual((old.launches, old.launch_basis), (3, ""))

    def test_name_counted_module_code_is_not_reported_exact(self):
        t = Trace(TraceMetadata())
        kd = _kd("add_kernel", "cc", arch="sass")
        kd.launches, kd.launch_basis = 4, "name"
        t.add_disasm(kd)
        self.assertEqual(t.disasm_for("add_kernel")[1], "name")
        kd.launch_basis = "module"
        self.assertEqual(t.disasm_for("add_kernel")[1], "module")


@unittest.skipUnless(shutil.which("gcc") and (shutil.which("objdump") or shutil.which("llvm-objdump")),
                     "needs gcc and objdump")
class TestExtractorVariants(unittest.TestCase):
    def test_two_binaries_defining_the_same_kernel(self):
        with tempfile.TemporaryDirectory() as d:
            paths = []
            for i, body in enumerate(("return x + 1;", "int s = 0; for (int k = 0; k < x; k++) s += k * x; return s;")):
                c = Path(d) / f"v{i}.c"
                c.write_text(f"int variant_k(int x) {{ {body} }}\nint helper_{i}(int x) {{ return x - {i}; }}\n")
                o = Path(d) / f"v{i}.bin"
                subprocess.run(["gcc", "-O1", "-c", str(c), "-o", str(o)], check=True)
                paths.append(str(o))
            mods = [module_id(Path(p).read_bytes()) for p in paths]
            jit = [{"name": "ocl_kernel.bin", "so_path": p, "mangled": "", "mod": m} for p, m in zip(paths, mods)]
            res = collect_disasm(["/nonexistent"], ["opencl"], jit, profiled_pid=999999999,
                                 launched_by_module={m: {"variant_k"} for m in mods})
            self.assertEqual(sorted(res), sorted(f"variant_k@{m}" for m in mods))
            a, b = (res[f"variant_k@{m}"] for m in mods)
            self.assertNotEqual([ln.text for ln in a.lines], [ln.text for ln in b.lines])
            self.assertEqual({a.module, b.module}, set(mods))
            # without the hook's mod= the id is computed from the file
            jit[0]["mod"] = ""
            res = collect_disasm(["/nonexistent"], ["opencl"], jit[:1], profiled_pid=999999999,
                                 launched_by_module={mods[0]: {"variant_k"}})
            self.assertEqual(list(res), [f"variant_k@{mods[0]}"])


class TestExtractorPtxBinaries(unittest.TestCase):
    def test_nvidia_opencl_ptx_binary(self):
        # NVIDIA's OpenCL returns PTX text as the program binary
        ptx = (".version 8.0\n.target sm_75\n.address_size 64\n"
               ".visible .entry benchmark_func(\n.param .u64 p\n)\n{\n"
               "\tld.param.u64 %rd1, [p];\n\tadd.s32 %r1, %r2, %r3;\n\tret;\n}\n")
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "prog.bin"
            p.write_bytes(ptx.encode() + b"\x00")
            mod = module_id(p.read_bytes())
            res = collect_disasm(["/nonexistent"], ["opencl"],
                                 [{"name": "ocl_kernel.bin", "so_path": str(p), "mangled": "", "mod": mod}],
                                 profiled_pid=999999999, launched_by_module={mod: {"benchmark_func"}})
            self.assertEqual(list(res), [f"benchmark_func@{mod}"])
            kd = res[f"benchmark_func@{mod}"]
            self.assertEqual(kd.arch, "ptx")
            self.assertIn("add.s32", [ln.mnemonic for ln in kd.lines])

    def test_only_the_launched_kernels_of_a_program(self):
        # BabelStream's OpenCL program also holds nstream, never launched by default
        ptx = ".version 8.0\n.target sm_75\n.address_size 64\n" + "".join(
            f".visible .entry {k}(\n.param .u64 p\n)\n{{\n\tadd.s32 %r1, %r2, %r3;\n\tret;\n}}\n"
            for k in ("copy", "nstream"))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "prog.bin"
            p.write_bytes(ptx.encode())
            mod = module_id(p.read_bytes())
            spans = [{"name": "ocl_kernel.bin", "so_path": str(p), "mangled": "", "mod": mod}]
            res = collect_disasm(["/nonexistent"], ["opencl"], spans, profiled_pid=999999999,
                                 launched_by_module={mod: {"copy"}})
            self.assertEqual(list(res), [f"copy@{mod}"])
            # older hooks (launches without mod=): every kernel, as before
            res = collect_disasm(["/nonexistent"], ["opencl"], spans, profiled_pid=999999999)
            self.assertEqual(sorted(res), [f"copy@{mod}", f"nstream@{mod}"])

    def test_cuda_cubin_saved_by_a_wrapped_process(self):
        # profiled through a wrapper script: the profiled pid is the wrapper's,
        # the cubin was saved by its child -- the module-load span names it
        ptx = ".version 8.0\n.target sm_75\n.address_size 64\n" \
              ".visible .entry vadd(\n.param .u64 p\n)\n{\n\tadd.s32 %r1, %r2, %r3;\n\tret;\n}\n"
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "hprofiler_cubin_424242_0.bin"
            p.write_text(ptx)
            mod = module_id(p.read_bytes())
            res = collect_disasm(["/nonexistent"], ["cuda"], [], profiled_pid=999999999,
                                 module_ids={str(p): mod})
            self.assertEqual(list(res), [f"vadd@{mod}"])
            self.assertFalse(p.exists())              # consumed, as the pid-matched ones are


class TestAssembleCopiesModule(unittest.TestCase):
    def test_native_device_span_gets_the_launch_module(self):
        from tests.test_gpu_activity import build, cupti_kernel, host
        trace = build([host("cuLaunchKernel", 1000, 50, lid=1, corr=7, stream=3, mod="0123456789abcdef"),
                       cupti_kernel(7, 13, 1100, 1300, name="_Z1kv")])
        dev = [s for s in trace.iter_spans(categories=("cuda",)) if s.tags.get("side") == "gpu"]
        self.assertEqual([s.tags.get("mod") for s in dev], ["0123456789abcdef"])


class TestJitStages(unittest.TestCase):
    def test_stage_of_every_hook_span(self):
        from src.analysis.jit_coverage import jit_stage
        cases = [
            (_span("nvrtcCompileProgram", "jit", type="jit_compile", stage="frontend"), "frontend"),
            (_span("hiprtcCompileProgram", "jit", type="jit_compile"), "frontend"),   # inferred
            (_span("cuModuleLoadDataEx", "jit", type="jit_compile", stage="backend"), "backend"),
            (_span("cuModuleLoadData", "jit", type="jit_compile"), "backend"),
            (_span("clBuildProgram", "jit", status="ok", stage="source"), "source"),
            (_span("clBuildProgram", "jit", status="ok"), "build"),                    # older hook
            (_span("x.jit.so", "jit", type="jit_load", path="/tmp/a"), "load"),
            (_span("__hipRegisterFatBinary", "jit", type="jit_compile"), "load"),
        ]
        for span, want in cases:
            self.assertEqual(jit_stage(span), want, span.name)

    def test_backend_only_note(self):
        from src.analysis.jit_coverage import jit_coverage
        t = Trace(TraceMetadata())
        t.add(_span("clBuildProgram", "jit", dur=500, status="ok", stage="backend"))
        t.add(_span("ocl_kernel.bin", "jit", dur=0, type="jit_load", stage="load"))
        jc = jit_coverage(t)
        self.assertFalse(jc["frontend_seen"])
        self.assertIn("back-end compile / load spans only", jc["note"])
        self.assertEqual(jc["stages"]["backend"]["total_ns"], 500)
        t.add(_span("nvrtcCompileProgram", "jit", dur=900, stage="frontend"))
        self.assertEqual(jit_coverage(t)["note"], "")
        self.assertIsNone(jit_coverage(Trace(TraceMetadata())))

    def test_summary_prints_the_stages(self):
        import io
        from contextlib import redirect_stdout
        from src.output.summary import print_summary
        t = Trace(TraceMetadata(command="./app"))
        t.add(_span("clBuildProgram", "jit", dur=500, status="ok", stage="backend"))
        buf = io.StringIO()
        with redirect_stdout(buf):
            print_summary(t)
        out = buf.getvalue()
        self.assertIn("JIT spans by stage", out)
        self.assertIn("back-end compile / module load", out)
        self.assertIn("no front-end or runtime compilation was observed", out)


if __name__ == "__main__":
    unittest.main()
