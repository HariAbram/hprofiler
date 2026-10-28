"""
OpenMP tasks, locks, critical sections and ordered regions through the real
hooks (tests/fixtures/omp_sync_counts.c), for both runtimes:

  * GNU libgomp (gcc): gomp_hook.c -- GOMP_task through its trampoline
    (omp_task_create / omp_task), GOMP_taskwait, GOMP_taskgroup_start/end,
    the omp_*_lock API; critical sections as before;
  * LLVM libomp (clang): ompt_tool.c's mutex callbacks (critical, locks,
    ordered), lock init/destroy and nest-lock instants, task dependences.

The program's output must equal an unprofiled run (the GNU trampoline copies
every task's arguments), and every construct must be counted exactly. Skips
when the compiler or the hook is missing.
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
SRC = ROOT / "tests/fixtures/omp_sync_counts.c"
T = 4


def _build_and_run(cc, hook, src=SRC):
    from src.core.hookdir import hooks_dir         # HPROFILER_HOOKS_DIR, else build/lib
    if not shutil.which(cc) or not (hooks_dir() / hook).exists():
        raise unittest.SkipTest(f"needs {cc} and {hooks_dir() / hook}")
    tmp = tempfile.mkdtemp()
    exe = os.path.join(tmp, "omp_sync")
    r = subprocess.run([cc, "-fopenmp", "-O1", str(src), "-o", exe], capture_output=True, text=True)
    if r.returncode != 0:
        shutil.rmtree(tmp, ignore_errors=True)
        raise unittest.SkipTest(f"cannot build with {cc}: {r.stderr[:200]}")
    env = dict(os.environ, OMP_NUM_THREADS=str(T))
    base = subprocess.run([exe], capture_output=True, text=True, timeout=120, env=env).stdout
    from src.core.runner import Runner
    out = os.path.join(tmp, "prof.out")
    with open(out, "w") as f:
        saved = os.dup(1)
        os.dup2(f.fileno(), 1)
        try:
            trace = Runner([exe], backends=["openmp"]).run()
        finally:
            os.dup2(saved, 1)
            os.close(saved)
    prof = Path(out).read_text()
    shutil.rmtree(tmp, ignore_errors=True)
    names = Counter(s.name for s in trace.iter_spans())
    names.update(i.name for i in trace.iter_instants())
    return base, prof, names, trace


def _skip_unless_recorded(test, spans, tag, what):
    """The hook library in use predates `tag` (build/ is rebuilt by hand):
    nothing to check against it."""
    if not any(tag in s.tags for s in spans):
        test.skipTest(f"{what}: the hook library predates {tag}= -- rebuild build/ "
                      f"or point HPROFILER_HOOKS_DIR at a fresh build")


def _lines(text):
    return sorted(ln for ln in text.splitlines() if ln[:2] in ("R ", "C ", "O ", "T "))


def _task_counts(trace):
    """(created, executed = omp_task spans that are not suspension
    fragments, fragments, spans with id=0, executed spans not complete)"""
    tasks = [s for s in trace.iter_spans() if s.name == "omp_task"]
    done = [s for s in tasks if s.tags.get("fragment") != "1"]
    return (sum(1 for s in trace.iter_spans() if s.name == "omp_task_create"), len(done),
            len(tasks) - len(done), sum(1 for s in tasks if s.tags.get("id") == "0"),
            sum(1 for s in done if s.tags.get("status", "1") != "1"))


class TestGnuLibgomp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base, cls.prof, cls.n, cls.trace = _build_and_run("gcc", "libhprofiler_gomp.so")

    def test_output_unchanged(self):
        self.assertEqual(_lines(self.prof), _lines(self.base))
        self.assertIn("C counter=28 crit=40", self.prof)

    def test_tasks(self):
        self.assertEqual(self.n["omp_task_create"], 27)
        self.assertEqual(self.n["omp_task"], 27)
        self.assertEqual(self.n["omp_taskwait"], 2)
        self.assertEqual(self.n["omp_taskgroup"], 1)
        undeferred = [s for s in self.trace.iter_spans() if s.name == "omp_task_create"
                      and s.tags.get("undeferred") == "1"]
        self.assertEqual(len(undeferred), 1)

    def test_locks_and_critical(self):
        # waits: 5 set per thread + the outer nest acquisition (the inner one
        # is owned); holds: those + the successful test
        self.assertEqual(self.n["omp_lock_wait"], T * 6)
        self.assertEqual(self.n["omp_lock_hold"], T * 7)
        self.assertEqual(self.n["omp_critical_wait"], T * 10)
        self.assertEqual(self.n["omp_critical_hold"], T * 10)


class TestLlvmLibompOmpt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base, cls.prof, cls.n, cls.trace = _build_and_run("clang", "libhprofiler_ompt.so")
        if not cls.n.get("parallel_region"):
            raise unittest.SkipTest("the clang build did not use LLVM libomp with OMPT")

    def test_output_unchanged(self):
        self.assertEqual(_lines(self.prof), _lines(self.base))

    def test_critical(self):
        self.assertEqual(self.n["omp_critical_wait"], T * 10)
        self.assertEqual(self.n["omp_critical_hold"], T * 10)

    def test_locks(self):
        self.assertEqual(self.n["omp_lock_wait"], T * 6)      # 5 set + outer nest; tests have no wait
        self.assertEqual(self.n["omp_lock_hold"], T * 7)      # + the successful test
        self.assertEqual(self.n["omp_nest_lock"], T * 2)      # inner acquire + release by the owner
        self.assertEqual(self.n["omp_lock_init"], 2)
        self.assertEqual(self.n["omp_lock_destroy"], 2)

    def test_ordered(self):
        self.assertEqual(self.n["omp_ordered"], 16)
        self.assertEqual(self.n["omp_ordered_wait"], 16)

    def test_tasks(self):
        # explicit tasks only: no implicit-task fragment recorded as omp_task
        created, executed, _fragments, id0, not_complete = _task_counts(self.trace)
        self.assertEqual((created, executed, id0, not_complete), (27, 27, 0, 0))

    def test_sync_region_kinds(self):
        # named by ompt_sync_region_t, the kind itself in sk= (the hook used
        # to name kinds 5-10 one slot late: taskwait came out as omp_sync,
        # taskgroup as omp_taskwait, a loop's implicit barrier as omp_reduction)
        sync = [s for s in self.trace.iter_spans() if s.category.value == "sync"
                and s.tags.get("type") == "sync" and "mutex" not in s.tags]       # sync regions
        _skip_unless_recorded(self, sync, "sk", "sync-region kinds")
        self.assertEqual(self.n["omp_taskwait"], 2)
        self.assertEqual(self.n["omp_taskgroup"], 1)
        self.assertEqual(self.n["omp_sync"], 0)
        self.assertEqual(self.n["omp_reduction"], 0)
        self.assertTrue(all(s.tags.get("sk") for s in sync))
        # taskwait / taskgroup: the task whose descendants they wait for;
        # barriers: their team (the parallel region)
        self.assertTrue(all(s.tags.get("task") for s in sync if s.name in ("omp_taskwait", "omp_taskgroup")))
        self.assertTrue(all(s.parent_span_id for s in sync if s.name.startswith("omp_barrier")))

    def test_task_parents(self):
        creates = [s for s in self.trace.iter_spans() if s.name == "omp_task_create"]
        _skip_unless_recorded(self, creates, "parent", "task parents")
        self.assertTrue(all(s.tags.get("parent") for s in creates))
        self.assertEqual(sum(1 for s in creates if s.tags.get("undeferred") == "1"), 1)

    def test_task_graph(self):
        # taskwait after 16 tasks, a taskgroup of 8, a final taskwait for all
        # 27 children of the single thread's task; one recorded dependence
        creates = [s for s in self.trace.iter_spans() if s.name == "omp_task_create"]
        _skip_unless_recorded(self, creates, "parent", "task graph")
        from collections import Counter as C
        from src.analysis import criticalpath as cp
        _spans, preds = cp.build_dependency_graph(self.trace)
        kinds = C(k for pl in preds.values() for _u, k, _c in pl)
        self.assertEqual(kinds["task_join"], 16 + 8 + 27)
        self.assertEqual(kinds["task_dep"], 1)

    def test_task_dependences(self):
        self.assertEqual(self.n["omp_task_dependences"], 2)
        self.assertEqual(self.n["omp_task_dependence"], 1)
        deps = [i for i in self.trace.iter_instants() if i.name == "omp_task_dependence"]
        creates = {s.tags.get("id") for s in self.trace.iter_spans() if s.name == "omp_task_create"}
        self.assertTrue({deps[0].tags["src"], deps[0].tags["sink"]} <= creates)


class TestTaskCounts(unittest.TestCase):
    """tests/fixtures/omp_task_counts.c: 27 tasks -- leaf tasks, a taskwait,
    a nested task its parent waits for, a task calling taskyield. Every task
    created is executed exactly once as a non-fragment omp_task span (LLVM:
    status complete; suspension fragments carry fragment=1 and the task's
    id); GNU libgomp's trampoline gives one span per task, as before."""
    SRC = ROOT / "tests/fixtures/omp_task_counts.c"

    def _check(self, cc, hook):
        base, prof, n, trace = _build_and_run(cc, hook, self.SRC)
        if cc == "clang" and not n.get("parallel_region"):
            self.skipTest("the clang build did not use LLVM libomp with OMPT")
        self.assertIn("T tasks_done=27", prof)
        created, executed, fragments, id0, not_complete = _task_counts(trace)
        self.assertEqual((created, executed, id0, not_complete), (27, 27, 0, 0))
        ids = {s.tags.get("id") for s in trace.iter_spans() if s.name == "omp_task_create"}
        tasks = [s for s in trace.iter_spans() if s.name == "omp_task"]
        self.assertEqual({s.tags.get("id") for s in tasks if s.tags.get("fragment") != "1"}, ids)
        self.assertTrue({s.tags.get("id") for s in tasks} <= ids)       # fragments: of a created task
        self.assertEqual(n["omp_task"], executed + fragments)
        return n, fragments

    def test_llvm_libomp(self):
        self._check("clang", "libhprofiler_ompt.so")

    def test_gnu_libgomp_unchanged(self):
        n, fragments = self._check("gcc", "libhprofiler_gomp.so")
        self.assertEqual(fragments, 0)


if __name__ == "__main__":
    unittest.main()
