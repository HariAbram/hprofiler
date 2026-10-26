"""
SYCL kernels run on the host are category "sycl", never "opencl".

AdaptiveCpp's generic compiler JIT-compiles each kernel into a host library
(.jit.so); on its OpenMP backend every work-group is one call of it. The
OpenCL hook times those calls (it intercepts the dlopen/dlsym), and used to
record them under its own category: a mixbench sycl-omp run showed four
"opencl" lanes of 19,010 spans each although no OpenCL ran.

  * the hook emits "sycl" (tests/native/opencl_trampoline_test.c);
  * a span from an older hook or JSON export (opencl + type=jit_kernel) is
    relabeled by Trace.add -- real OpenCL work stays "opencl";
  * an existing store is relabeled once when opened (seq and event ids kept,
    lanes and aggregates rebuilt);
  * the category is last in the enum, so stored codes of the others hold.
"""
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core import trace_io
from src.core.events import Category, SpanEvent
from src.core.runner import _parse_record
from src.core.store import DiskTraceStore
from src.core.trace import Trace, TraceMetadata

OLD_HOOK_KERNEL = "span:opencl:100:101:1000:500:runbench_warmup:type=jit_kernel,side=cpu,mod=8d6b22cd635f14f6"
NEW_HOOK_KERNEL = "span:sycl:100:101:2000:500:runbench_warmup:type=jit_kernel,side=cpu,mod=8d6b22cd635f14f6"
REAL_OPENCL = "span:opencl:100:101:3000:500:vecadd:type=kernel,side=cpu"


class TestIngestion(unittest.TestCase):
    def test_old_and_new_hook_records_are_sycl_real_opencl_stays(self):
        t = Trace(TraceMetadata())
        for line in (OLD_HOOK_KERNEL, NEW_HOOK_KERNEL, REAL_OPENCL):
            t.add(_parse_record(line))
        cats = [(s.name, s.category) for s in t.iter_spans()]
        self.assertEqual(cats, [("runbench_warmup", Category.SYCL), ("runbench_warmup", Category.SYCL),
                                ("vecadd", Category.GPU_OPENCL)])
        lanes = {ln.name for ln in t.lane_infos()}
        self.assertIn("sycl/thread-101", lanes)
        self.assertIn("opencl/thread-101", lanes)          # only the real OpenCL call
        self.assertEqual(sum(1 for _ in t.iter_spans(categories=("opencl",))), 1)

    def test_an_older_json_export_imports_as_sycl(self):
        events = [{"ph": "X", "cat": "opencl", "name": "k", "ts": 1.0, "dur": 2.0, "pid": 1, "tid": 2,
                   "args": {"type": "jit_kernel", "side": "cpu"}},
                  {"ph": "X", "cat": "opencl", "name": "clEnqueueNDRangeKernel", "ts": 5.0, "dur": 1.0,
                   "pid": 1, "tid": 2, "args": {"type": "kernel", "side": "cpu"}}]
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "old.json")
            Path(p).write_text(json.dumps({"traceEvents": events}))
            from src.output.chrome_trace import load_trace_from_json
            t = load_trace_from_json(p)
            self.assertEqual({s.name: s.category for s in t.iter_spans()},
                             {"k": Category.SYCL, "clEnqueueNDRangeKernel": Category.GPU_OPENCL})

    def test_category_is_appended_so_older_codes_hold(self):
        from src.core.store.common import CATEGORY_CODES
        self.assertIs(list(Category)[-1], Category.SYCL)
        self.assertEqual(CATEGORY_CODES["opencl"], 3)
        self.assertEqual(CATEGORY_CODES["other"], 12)


class TestExistingStores(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp, "old.hpstore")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _old_store(self):
        """A store as an older hprofiler wrote it: kernel calls stored as
        opencl (appended straight to the store, past Trace.add), and no
        category revision."""
        store = DiskTraceStore(self.path)
        t = Trace(TraceMetadata(), store=store)
        for i in range(4):
            store.append(SpanEvent("omp_loop_static", Category.OPENMP, i * 100, 90, 1, 2 + i % 2))
            store.append(SpanEvent("runbench_warmup", Category.GPU_OPENCL, i * 100 + 10, 50, 1, 2 + i % 2,
                                   tags={"type": "jit_kernel", "side": "cpu", "mod": "aa"}))
        store.append(SpanEvent("vecadd", Category.GPU_OPENCL, 1000, 5, 1, 2, tags={"type": "kernel", "side": "cpu"}))
        store._set_info("category_revision", 0)
        t.finalize()
        before = {s.seq: (s.eid, s.name, s.category) for s in t.iter_spans()}
        t.close()
        return before

    def test_opening_relabels_once_and_keeps_ids(self):
        before = self._old_store()
        t = trace_io.open_trace(self.path)
        after = {s.seq: (s.eid, s.name, s.category) for s in t.iter_spans()}
        self.assertEqual(set(after), set(before))
        for seq, (eid, name, cat) in after.items():
            self.assertEqual(eid, before[seq][0])                        # event ids kept
            want = Category.SYCL if name == "runbench_warmup" else before[seq][2]
            self.assertEqual(cat, want, name)
        lanes = {ln.name: ln.count for ln in t.lane_infos()}
        self.assertEqual(lanes.get("sycl/thread-2"), 2)
        self.assertEqual(lanes.get("sycl/thread-3"), 2)
        self.assertEqual(lanes.get("opencl/thread-2"), 1)                # vecadd only
        self.assertNotIn("opencl/thread-3", lanes)
        self.assertEqual(sum(1 for _ in t.iter_spans(categories=("sycl",))), 4)
        self.assertEqual({r["category"] for r in t.aggregate_stats() if r["name"] == "runbench_warmup"}, {"sycl"})
        version = t.store._content_version
        t.close()
        t = trace_io.open_trace(self.path)                                 # second open: nothing to do
        self.assertEqual(t.store._content_version, version)
        self.assertEqual(t.store._info_int("category_revision", 0), 1)
        t.close()

    def test_new_stores_need_no_relabel(self):
        t = trace_io.create_disk_trace(self.path, TraceMetadata())
        self.assertEqual(t.store._info_int("category_revision", 0), 1)
        t.close()


if __name__ == "__main__":
    unittest.main()
