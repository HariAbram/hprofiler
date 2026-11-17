"""
AdaptiveCpp dependencies the trace cannot show (src/analysis/semantics/sycl.py),
on synthetic spans: host-device kernel calls (category sycl) are counted,
and stream host functions only in processes that ran AdaptiveCpp (a sycl
span, an acpp_jit span or instant, an __acpp_ / hipsycl kernel name) --
never another program's. Neither gets an edge. Real runs:
tests/integration/test_sycl_deps.py.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis import criticalpath as cp
from src.core.events import Category, InstantEvent, SpanEvent
from src.core.trace import Trace, TraceMetadata


def S(name, cat, start, end, pid=1, tid=1, **tags):
    return SpanEvent(name, Category(cat), start, end - start, pid, tid, tags={k: str(v) for k, v in tags.items()})


def host_func(pid, start=100):
    return S("host_func", "cuda", start, start + 50, pid=pid, tid=9, type="host_func", side="gpu", op="host_func",
             stream=7)


def trace(events):
    t = Trace(TraceMetadata())
    for e in events:
        t.add(e)
    return t


class TestSyclDiagnostics(unittest.TestCase):
    def test_host_device_kernels_are_counted(self):
        t = trace([S("main", "sycl", 0, 10, tid=2, type="jit_kernel", side="cpu"),
                   S("main", "sycl", 20, 30, tid=3, type="jit_kernel", side="cpu"),
                   S("main", "sycl", 0, 10, pid=2, tid=4, type="jit_kernel", side="cpu")])
        d = cp.dependency_diagnostics(t)
        self.assertEqual((d["sycl_host_kernel_calls"], d["sycl_host_kernel_processes"]), (3, 2))
        w = cp.dependency_warnings(d)
        self.assertTrue(any(x.startswith("3 host-device SYCL kernel call(s) in 2 process(es)") for x in w), w)
        spans, preds = cp.build_dependency_graph(t)
        self.assertFalse([k for pl in preds.values() for _u, k, _c in pl if k != "sequential"])

    def test_host_functions_count_in_acpp_processes_only(self):
        marks = {
            "jit span": S("acpp_jit_compile", "jit", 0, 5, pid=1, type="jit_compile"),
            "kernel name": S("_Z18__acpp_sscp_kernelIZ4mainE2kA", "cuda", 0, 5, pid=1, side="gpu", type="kernel"),
            "jit instant": InstantEvent("acpp_jit_cache", Category.JIT, 3, 1, 1, tags={"type": "jit_cache"}),
        }
        for what, mark in marks.items():
            with self.subTest(what):
                d = cp.dependency_diagnostics(trace([mark, host_func(1), host_func(2)]))
                self.assertEqual(d.get("acpp_host_functions"), 1)              # pid 2 is no AdaptiveCpp process
                self.assertTrue(any("stream host function" in x for x in cp.dependency_warnings(d)))

    def test_nothing_without_adaptivecpp(self):
        d = cp.dependency_diagnostics(trace([host_func(1), S("cudaLaunchKernel", "cuda", 0, 5)]))
        self.assertFalse([k for k in d if "sycl" in k or "acpp" in k])


if __name__ == "__main__":
    unittest.main()
