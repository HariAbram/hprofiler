"""
GNU libgomp task dependences, parents, joins and locks through the real GOMP
hook (tests/fixtures/gomp_deps.c, whose header lists the expected
dependences): the dependence pairs the analysis reconstructs from deps=
equal the program's, for every dependence type GCC passes (in, out, inout,
mutexinoutset, inoutset, depobj, omp_all_memory); taskwait depend(...)
waits for the conflicting sibling only; the taskgroup waits for its task
and that task's child; tasks executed on other threads than their creator
still name their parent; a lock initialised twice at one address gets two
tokens; no address appears in the trace. Run again with the ABI level
forced past the checked one (HPROFILER_GOMP_ABI_LEVEL), the new-format
clause lists are deps_unsupported=abi, their dependences are absent, not
guessed, and the diagnostics say so.

Skips without gcc/libgomp or with a hook that predates parent=.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

EXPECTED = {(0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (6, 8), (7, 8), (9, 10),
            (3, 11), (5, 11), (8, 11), (10, 11), (11, 12)}
NEW_FORMAT = {4, 5, 6, 7, 9}          # mutexinoutset / inoutset / depobj: depend[0] == 0


def _profile(exe, env=None):
    from src.core.runner import Runner
    with mock.patch.dict(os.environ, env or {}):
        rf, wf = os.pipe()
        saved = os.dup(1)
        os.dup2(wf, 1)
        try:
            trace = Runner([exe], backends=["openmp"]).run()
        finally:
            os.dup2(saved, 1)
            os.close(wf)
        out = os.read(rf, 65536).decode()
        os.close(rf)
    return trace, out


def _tasks(trace):
    """The single thread's tasks in creation order, and every span."""
    spans = list(trace.iter_spans())
    created = sorted((s for s in spans if s.name == "omp_task_create"), key=lambda s: s.start_ns)
    first_parent = created[0].tags["parent"]
    top = [s for s in created if s.tags.get("parent") == first_parent]
    return top, created, spans


class TestGompDependences(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.core.hookdir import hooks_dir
        if not shutil.which("gcc") or not (hooks_dir() / "libhprofiler_gomp.so").exists():
            raise unittest.SkipTest("needs gcc and the built GOMP hook")
        cls.tmp = tempfile.mkdtemp()
        cls.exe = os.path.join(cls.tmp, "gomp_deps")
        r = subprocess.run(["gcc", "-O1", "-fopenmp", str(ROOT / "tests/fixtures/gomp_deps.c"), "-o", cls.exe],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise unittest.SkipTest(f"cannot build the fixture (needs GCC >= 12): {r.stderr[:300]}")
        cls.base = subprocess.run([cls.exe], capture_output=True, text=True, timeout=60).stdout
        cls.trace, cls.out = _profile(cls.exe)
        cls.top, cls.created, cls.spans = _tasks(cls.trace)
        if "parent" not in cls.created[0].tags:
            raise unittest.SkipTest("the GOMP hook predates parent= -- point HPROFILER_HOOKS_DIR at a fresh build")
        from src.analysis import criticalpath as cp
        cls.cp = cp
        cls.gspans, cls.preds = cp.build_dependency_graph(cls.trace)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(getattr(cls, "tmp", ""), ignore_errors=True)

    def index(self):
        return {s.tags["id"]: k for k, s in enumerate(self.top)}

    def edge_pairs(self, preds, spans, kind):
        return [(spans[u], spans[v], c) for v, pl in preds.items() for u, k, c in pl if k == kind]

    def test_output_unchanged(self):
        self.assertEqual(self.out.strip(), self.base.strip())

    def test_every_dependence_type_is_reconstructed_exactly(self):
        self.assertEqual(len(self.top), 14)
        idx = self.index()
        got = {(idx[a.tags["id"]], idx[b.tags["id"]])
               for a, b, c in self.edge_pairs(self.preds, self.gspans, "task_dep")
               if a.tags.get("id") in idx and b.tags.get("id") in idx}
        self.assertEqual(got, EXPECTED)
        kinds = {self.top[k].tags["deps"][0] for k in (0, 1, 4, 6)}
        self.assertEqual(kinds, {"o", "i", "m", "s"})
        self.assertIn("o*", self.top[11].tags["deps"])

    def test_taskwait_depend_and_the_taskgroup(self):
        idx = self.index()
        joins = self.edge_pairs(self.preds, self.gspans, "task_join")
        tw_dep = [(a, c) for a, b, c in joins if b.name == "omp_taskwait" and b.tags.get("depend") == "1"]
        self.assertEqual([(idx[a.tags["id"]], c) for a, c in tw_dep], [(11, "high")])
        tg = {a.tags["id"] for a, b, _c in joins if b.name == "omp_taskgroup"}
        t13 = self.top[13].tags["id"]
        child = next(s for s in self.created if s.tags.get("parent") == t13)
        self.assertEqual(tg, {t13, child.tags["id"]})
        plain = {idx[a.tags["id"]] for a, b, _c in joins
                 if b.name == "omp_taskwait" and b.tags.get("depend") != "1" and a.tags.get("id") in idx}
        self.assertEqual(plain, set(range(14)))

    def test_tasks_run_elsewhere_still_name_their_parent(self):
        creator = self.top[0].tid
        runs = [s for s in self.spans if s.name == "omp_task"]
        self.assertTrue(any(s.tid != creator for s in runs), "every task ran on its creator's thread")
        t13 = self.top[13].tags["id"]
        child = next(s for s in self.created if s.tags.get("parent") == t13)
        ran13 = next(s for s in runs if s.tags["id"] == t13)
        self.assertEqual(child.tid, ran13.tid)          # created by T13, on the thread running T13

    def test_lock_tokens_and_no_addresses(self):
        holds = [s for s in self.spans if s.name == "omp_lock_hold"]
        self.assertEqual(len({s.tags["lock"] for s in holds}), 2)
        for s in self.spans:
            for k, v in s.tags.items():
                if k not in ("sym", "symfile", "lib", "offset", "codeptr", "stack"):
                    self.assertIsNone(re.search(r"0x[0-9a-f]{6,}", str(v)), (s.name, k, v))
        abi = [c.value for c in self.trace.iter_counters() if c.name == "gomp_abi_level"]
        self.assertTrue(abi and abi[0] >= 50000, abi)

    def test_an_unknown_abi_leaves_new_format_clauses_undecoded(self):
        trace, _out = _profile(self.exe, {"HPROFILER_GOMP_ABI_LEVEL": "60000"})
        top, _created, _spans = _tasks(trace)
        for k, s in enumerate(top):
            if k in NEW_FORMAT:
                self.assertEqual(s.tags.get("deps_unsupported"), "abi", k)
                self.assertNotIn("deps", s.tags)
            elif k < 13:
                self.assertIn("deps", s.tags, k)
        spans, preds = self.cp.build_dependency_graph(trace)
        idx = {s.tags["id"]: k for k, s in enumerate(top)}
        got = {(idx[a.tags["id"]], idx[b.tags["id"]])
               for a, b, c in self.edge_pairs(preds, spans, "task_dep")
               if a.tags.get("id") in idx and b.tags.get("id") in idx}
        self.assertTrue(got <= EXPECTED, got - EXPECTED)        # nothing invented
        self.assertFalse({p for p in got if p[0] in NEW_FORMAT or p[1] in NEW_FORMAT})
        diag = self.cp.dependency_diagnostics(trace)
        self.assertEqual(diag["omp_deps_unsupported"], {"abi": len(NEW_FORMAT)})


if __name__ == "__main__":
    unittest.main()
