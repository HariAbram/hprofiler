"""
Span ids shared by two hooks' spans (src/analysis/span_links.py): traces
recorded before hooks/common/hp_ids.h gave every hook its own id space carry
one id on, e.g., an MPI request and a CUDA host call ((pid << 32) | n in
both hooks). Found when `hprofiler gui` on an LLNL Comb (MPI + CUDA) trace
died with "maximum recursion depth exceeded" building the flame graph: the
call tree shared one node per id, which closed a cycle.

  * the resolver picks the child's own hook family, else an enclosing
    annotation range, else the latest-started same-family span before it;
  * the call tree of a thread whose ids collide is a forest (every span
    once, no cycle), with each explicit link to the right span -- both
    builders (list and store);
  * the critical path's explicit edges link an MPI completion to its MPI
    request, not to the CUDA call with the same id -- both builders;
  * the hooks' id spaces (hp_ids.h) differ for the same pid and counter.
"""
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analysis import call_tree as ct
from src.analysis import criticalpath as cp
from src.analysis.span_links import SpanIdResolver, id_family
from src.core.events import Category, SpanEvent
from src.core.trace import Trace, TraceMetadata

X = str((1711339 << 32) | 1)        # the MPI hook's first request and the CUDA hook's first call


def _span(name, cat, start, end, sid="", psid="", tid=1, **tags):
    return SpanEvent(name=name, category=Category(cat), start_ns=start, duration_ns=end - start, pid=1,
                     tid=tid, tags={k: str(v) for k, v in tags.items()}, span_id=sid, parent_span_id=psid)


def colliding():
    """One thread: a CUDA launch (sid X), its kernel (psid X, timed on the
    launching thread), and an MPI receive with the same id X inside the
    kernel's interval, completed by an MPI_Wait (psid X). With one node per
    id the receive's node was both the kernel's parent (explicit) and its
    child (temporal): a cycle."""
    return [_span("cudaLaunchKernel", "cuda", 0, 10, sid=X, side="cpu", rt="cuda"),
            _span("_Z4packv", "cuda", 150, 300, psid=X, side="gpu", rt="cuda"),
            _span("MPI_Irecv", "mpi", 200, 210, sid=X, type="irecv"),
            _span("MPI_Wait", "mpi", 220, 260, psid=X, type="wait")]


def _walk(nodes):
    seen, stack = [], list(nodes)
    while stack:
        n = stack.pop()
        seen.append(n)
        stack.extend(n.children)
    return seen


class TestResolver(unittest.TestCase):
    def test_family_then_time(self):
        r = SpanIdResolver()
        spans = colliding()
        for i, s in enumerate(spans):
            r.add_span(s, i)
        self.assertEqual(r.ambiguous_ids(), 1)
        self.assertEqual(r.parent_of(spans[1]), 0)          # kernel -> its CUDA launch
        self.assertEqual(r.parent_of(spans[3]), 2)          # Wait -> its MPI request
        self.assertEqual([id_family(s) for s in spans], ["gpu", "gpu", "mpi", "mpi"])

    def test_families_without_tags(self):
        bar = SpanEvent("omp_barrier", Category.SYNC, 0, 1, 1, 1)
        sync = SpanEvent("cudaStreamSynchronize", Category.SYNC, 0, 1, 1, 1)
        rng = SpanEvent("phase", Category.NVTX, 0, 1, 1, 1)
        self.assertEqual([id_family(s) for s in (bar, sync, rng)], ["openmp", "gpu", "annotation"])

    def test_no_guess_across_families(self):
        r = SpanIdResolver()
        r.add(1, "7", 0, "gpu", "a")
        r.add(1, "7", 5, "openmp", "b")
        self.assertIsNone(r.resolve(1, "7", 10, "mpi"))
        self.assertEqual(r.resolve(1, "7", 10, "gpu"), "a")
        r.add(1, "8", 0, "annotation", "range")
        r.add(1, "8", 3, "mpi", "req")
        self.assertEqual(r.resolve(1, "8", 10, "gpu"), "range")    # a host call's enclosing range


class TestCallTreeWithCollidingIds(unittest.TestCase):
    def test_the_raw_tree_is_a_forest_with_the_right_links(self):
        spans = colliding()
        roots = ct._ct_build_raw(spans)
        nodes = _walk(roots)
        self.assertEqual(sorted(n.span.name for n in nodes), sorted(s.name for s in spans))
        parent = {id(c.span): n.span.name for n in nodes for c in n.children}
        by = {s.name: s for s in spans}
        self.assertEqual(parent[id(by["_Z4packv"])], "cudaLaunchKernel")
        self.assertEqual(parent[id(by["MPI_Wait"])], "MPI_Irecv")

    def test_both_builders_finish(self):
        t = Trace(TraceMetadata())
        for s in colliding():
            t.add(s)
        for roots in (ct._ct_build(colliding()), ct.build_call_tree(t)):
            names = [n.name for n in _walk(roots)]
            self.assertEqual(sorted(names), ["MPI_Irecv", "MPI_Wait", "_Z4packv", "cudaLaunchKernel"])

    def test_many_collisions_stay_shallow(self):
        # the Comb pattern repeated: one shared id per iteration
        spans = []
        for k in range(1500):
            o, sid = k * 1000, str((1 << 32) | k)
            spans += [_span("cudaLaunchKernel", "cuda", o, o + 10, sid=sid, side="cpu", rt="cuda"),
                      _span("_Z4packv", "cuda", o + 150, o + 300, psid=sid, side="gpu", rt="cuda"),
                      _span("MPI_Irecv", "mpi", o + 200, o + 210, sid=sid, type="irecv"),
                      _span("MPI_Wait", "mpi", o + 220, o + 260, psid=sid, type="wait")]
        t = Trace(TraceMetadata())
        for s in spans:
            t.add(s)
        roots = ct.build_call_tree(t)          # used to recurse until RecursionError
        self.assertEqual(sum(n.count for n in _walk(roots)), 6000)

    def test_flame_graph_of_such_a_trace(self):
        from src.analysis.flamegraph_tree import build_flame_tree_for
        t = Trace(TraceMetadata())
        for s in colliding():
            t.add(s)
        self.assertIsNotNone(build_flame_tree_for(t))


class TestCriticalPathExplicitEdges(unittest.TestCase):
    def test_completion_links_to_its_request_in_both_builders(self):
        # the CUDA call carrying the same id comes later (last-wins picked it)
        spans = [_span("MPI_Irecv", "mpi", 0, 10, sid=X, type="irecv"),
                 _span("cudaLaunchKernel", "cuda", 20, 30, sid=X, side="cpu", rt="cuda"),
                 _span("MPI_Wait", "mpi", 40, 90, psid=X, type="wait")]
        t = Trace(TraceMetadata())
        for s in spans:
            t.add(s)
        full, preds = cp.build_dependency_graph(t)
        exp = [(full[u].name, full[v].name) for v, pl in preds.items() for u, k, _c in pl if k == "explicit_span_id"]
        self.assertEqual(exp, [("MPI_Irecv", "MPI_Wait")])
        edges = cp.build_edges(t)
        streamed = [(full[int(s_)].name, full[int(d)].name) for d, s_, k in
                    zip(edges["dst"], edges["src"], edges["kind"]) if k == cp._KIND_CODE["explicit_span_id"]]
        self.assertEqual(streamed, exp)


@unittest.skipUnless(shutil.which("gcc"), "needs gcc")
class TestHookIdSpaces(unittest.TestCase):
    def test_hooks_never_share_an_id(self):
        src = ('#include <stdio.h>\n#include "hp_ids.h"\nint main(void) {\n'
               '  unsigned tags[] = {HP_ID_CUDA, HP_ID_ROCM, HP_ID_MPI, HP_ID_OMPT, HP_ID_NVTX, HP_ID_OPENCL};\n'
               '  for (int i = 0; i < 6; i++) printf("%llu\\n", (unsigned long long)hp_span_id(tags[i], 1711339u, 1u));\n'
               '  printf("%llu\\n", (unsigned long long)hp_span_id(HP_ID_MPI, 4194303u, 0xffffffffu));\n'
               '  return 0;\n}\n')
        with tempfile.TemporaryDirectory() as d:
            c = Path(d) / "ids.c"
            c.write_text(src)
            exe = Path(d) / "ids"
            subprocess.run(["gcc", "-Wall", "-Werror", f"-I{ROOT / 'hooks/common'}", str(c), "-o", str(exe)],
                           check=True, capture_output=True)
            ids = [int(x) for x in subprocess.run([str(exe)], capture_output=True, text=True).stdout.split()]
        self.assertEqual(len(set(ids[:6])), 6)
        self.assertNotIn(int(X), ids)                       # the old layout
        self.assertLess(ids[-1], 1 << 63)                   # fits a signed 64-bit integer
        self.assertEqual((ids[-1] >> 32) & 0xffffff, 4194303)


if __name__ == "__main__":
    unittest.main()
