"""
`hprofiler run --python-jit` (src/instrument/pysite/): a Python JIT
compiler's own compilation as jit spans. A stub `triton` package (in a temp
dir) whose compile() calls triton.knobs.compilation.listener the way Triton
3.6 does -- stage durations in microseconds, cache_hit -- run through the
real Runner:

  * a compile: one span triton_jit_compile, stage=frontend, cache_hit=0, the
    hash and kernel name, per-stage *_ns tags, its duration their sum, and it
    ends when the listener was called (CLOCK_MONOTONIC, the trace clock);
  * a cache hit: an instant triton_jit_cache (cache_hit=1), no span;
  * the program's own sitecustomize still runs, and a listener it set
    before is still called (chained);
  * without the flag nothing is added (PYTHONPATH untouched).
"""
import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

STUB = '''
import os
class CompileTimes:
    def __init__(self, ir_initialization, lowering_stages, store_results):
        self.ir_initialization = ir_initialization
        self.lowering_stages = lowering_stages
        self.store_results = store_results
class _Compilation:
    listener = None
class _Knobs:
    def __init__(self):
        self.compilation = _Compilation()
knobs = _Knobs()
calls = []
def _user_listener(*, src, metadata, metadata_group, times, cache_hit):
    calls.append(cache_hit)
knobs.compilation.listener = _user_listener          # set by "the program" at import
class _Src:
    name = "add_kernel"
def compile(hash, cache_hit, ir, stages, store):
    l = knobs.compilation.listener
    if l:
        l(src=_Src(), metadata={"hash": hash, "name": "add_kernel"}, metadata_group={},
          times=CompileTimes(ir, stages, store), cache_hit=cache_hit)
'''

PROG = '''
import time
import triton
before = time.monotonic_ns()
triton.compile("9f3c0de1", False, 1500, [("ttir", 2000), ("ttgir", 3000), ("llir", 1000),
                                         ("ptx", 4000), ("cubin", 9000)], 500)
after = time.monotonic_ns()
triton.compile("9f3c0de1", True, 120, [], 0)
print("WINDOW", before, after)
print("USER_LISTENER", triton.calls)
'''


def _run(python_jit):
    from src.core.runner import Runner
    tmp = tempfile.mkdtemp()
    stub, user = Path(tmp, "stub"), Path(tmp, "usersite")
    (stub / "triton").mkdir(parents=True)
    (stub / "triton" / "__init__.py").write_text(textwrap.dedent(STUB))
    user.mkdir()
    (user / "sitecustomize.py").write_text("print('USER_SITECUSTOMIZE_RAN', flush=True)\n")
    prog = Path(tmp, "prog.py")
    prog.write_text(textwrap.dedent(PROG))
    out = Path(tmp, "out.txt")
    env = {"PYTHONPATH": f"{user}{os.pathsep}{stub}"}
    if python_jit:
        env["HPROFILER_PYTHON_JIT"] = "1"
    with open(out, "w") as f:
        saved = os.dup(1)
        os.dup2(f.fileno(), 1)
        try:
            trace = Runner([sys.executable, str(prog)], backends=["openmp"], env_extra=env).run()
        finally:
            os.dup2(saved, 1)
            os.close(saved)
    return trace, out.read_text()


class TestPythonJit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trace, cls.out = _run(True)

    def test_compile_span(self):
        spans = [s for s in self.trace.iter_spans(categories=("jit",))]
        self.assertEqual([s.name for s in spans], ["triton_jit_compile"], self.out)
        s = spans[0]
        self.assertEqual({k: s.tags[k] for k in ("type", "stage", "cache", "cache_hit", "hash", "kernel")},
                         {"type": "jit_compile", "stage": "frontend", "cache": "miss", "cache_hit": "0",
                          "hash": "9f3c0de1", "kernel": "add_kernel"})
        self.assertEqual({k: int(s.tags[k]) for k in ("ir_init_ns", "ttir_ns", "ttgir_ns", "llir_ns",
                                                     "ptx_ns", "cubin_ns", "store_ns")},
                         {"ir_init_ns": 1_500_000, "ttir_ns": 2_000_000, "ttgir_ns": 3_000_000,
                          "llir_ns": 1_000_000, "ptx_ns": 4_000_000, "cubin_ns": 9_000_000, "store_ns": 500_000})
        self.assertEqual(s.tags["stages"], "ttir/ttgir/llir/ptx/cubin")
        self.assertEqual(s.duration_ns, 21_000_000)
        # it ends when the listener ran: inside the program's own clock window
        before, after = (int(x) for x in next(l for l in self.out.splitlines() if l.startswith("WINDOW")).split()[1:])
        self.assertTrue(before <= s.start_ns + s.duration_ns <= after, (before, s.end_ns, after))

    def test_cache_hit_is_an_instant(self):
        hits = [i for i in self.trace.iter_instants() if i.name == "triton_jit_cache"]
        self.assertEqual(len(hits), 1)
        self.assertEqual((hits[0].tags["cache_hit"], hits[0].tags["cache"], hits[0].tags["lookup_ns"]),
                         ("1", "hit", "120000"))

    def test_the_programs_sitecustomize_and_listener_still_run(self):
        self.assertIn("USER_SITECUSTOMIZE_RAN", self.out)
        self.assertIn("USER_LISTENER [False, True]", self.out)

    def test_the_connection_ends_with_a_final_transport_status(self):
        transport = self.trace.metadata.capture_health.get("transport", {})
        pyjit = {k: v for k, v in transport.items() if k.endswith("/pyjit")}
        self.assertEqual([v.get("final") for v in pyjit.values()], [1], transport)
        self.assertEqual(next(iter(pyjit.values()))["sent"], 2)

    def test_jit_coverage_sees_the_front_end(self):
        from src.analysis.jit_coverage import jit_coverage
        jc = jit_coverage(self.trace)
        self.assertTrue(jc["frontend_seen"])
        self.assertEqual(jc["stages"]["frontend"]["count"], 1)

    def test_off_by_default(self):
        trace, out = _run(False)
        self.assertEqual(list(trace.iter_spans(categories=("jit",))), [])
        self.assertIn("USER_SITECUSTOMIZE_RAN", out)
        self.assertIn("USER_LISTENER [False, True]", out)
        from src.core.runner import hook_environment
        env = hook_environment([], "@x", base_env={"PYTHONPATH": "/a"})
        self.assertEqual(env["PYTHONPATH"], "/a")
        env = hook_environment([], "@x", env_extra={"HPROFILER_PYTHON_JIT": "1"}, base_env={"PYTHONPATH": "/a"})
        self.assertEqual(env["PYTHONPATH"], f"{ROOT / 'src/instrument/pysite'}{os.pathsep}/a")


if __name__ == "__main__":
    unittest.main()
