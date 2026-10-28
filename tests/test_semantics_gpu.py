"""
CUDA stream semantics in the dependency graph (criticalpath._add_gpu_edges,
semantics/gpu.py), on synthetic wire records:

  * the legacy default stream: a synchronous copy (stream 0) runs after the
    earlier work of a blocking stream and before its later work -- "high"
    where the stream's creation was recorded, "medium" where it is assumed
    blocking; a stream created non-blocking is exempt; older hooks' copies
    without stream= count as stream 0;
  * a blocking copy's host call returns after its device copy;
  * host functions are stream-ordered, and one callback thread runs them in
    turn;
  * both pipelines build the same edges.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis import criticalpath as cp
from tests.test_gpu_activity import build, cupti_kernel, cupti_memcpy, host


def edges(t, kinds=("sequential", "device_wait")):
    spans, preds = cp.build_dependency_graph(t)
    full = [(v, u, cp._KIND_CODE[k], cp._CONF_CODE[c]) for v, pl in preds.items() for u, k, c in pl]
    assert sorted(cp.build_edges(t).tolist()) == sorted(full), "pipelines differ"
    return {(spans[u].name, spans[v].name, k, c) for v, pl in preds.items() for u, k, c in pl
            if k in kinds and spans[u].tags.get("side") == "gpu"}


def scenario(created=True, copy_stream=True):
    lines = [host("cudaLaunchKernel", 1000, 50, lid=3, corr=201, stream=11),
             host("cudaLaunchKernel", 1060, 50, lid=4, corr=202, stream=22),
             cupti_kernel(201, 7, 1200, 2000, name="k_a"),
             cupti_kernel(202, 8, 1200, 3000, name="k_b"),
             host("cudaMemcpy", 1100, 1500, cat="memory", typ="memcpy", op="memcpy", lid=5, corr=203, dir="DtoH",
                  blocking=1, **({"stream": 0} if copy_stream else {})),
             cupti_memcpy(203, 0, 1, 2100, 2500, dir_="DtoH"),
             host("cudaLaunchKernel", 2700, 50, lid=6, corr=204, stream=11),
             cupti_kernel(204, 7, 2800, 3500, name="k_c")]
    if created:
        lines = [host("cudaStreamCreate", 100, 10, typ="stream_create", op="stream_create", lid=1, stream=11),
                 host("cudaStreamCreateWithFlags", 120, 10, typ="stream_create", op="stream_create", lid=2,
                      stream=22, nonblocking=1)] + lines
    return build(lines)


class TestLegacyDefaultStream(unittest.TestCase):
    def test_a_blocking_stream_orders_around_the_copy(self):
        e = edges(scenario())
        self.assertIn(("k_a", "memcpy DtoH", "sequential", "high"), e)
        self.assertIn(("memcpy DtoH", "k_c", "sequential", "high"), e)
        self.assertFalse({x for x in e if "k_b" in x[:2]})           # the non-blocking stream is exempt

    def test_streams_without_recorded_flags_are_assumed_blocking(self):
        e = edges(scenario(created=False))
        self.assertIn(("k_a", "memcpy DtoH", "sequential", "medium"), e)
        self.assertIn(("k_b", "memcpy DtoH", "sequential", "medium"), e)

    def test_an_older_hooks_copy_without_a_stream_is_on_stream_zero(self):
        e = edges(scenario(copy_stream=False))
        self.assertIn(("k_a", "memcpy DtoH", "sequential", "high"), e)


class TestBlockingCopy(unittest.TestCase):
    def test_the_host_call_waits_for_its_copy(self):
        self.assertIn(("memcpy DtoH", "cudaMemcpy", "device_wait", "certain"), edges(scenario()))


class TestHostFunctions(unittest.TestCase):
    def test_stream_order_and_one_callback_thread(self):
        def hf(lid, stream, start, end, tid=900):
            from tests.test_gpu_activity import _line
            # as cuda_hook.c's host_fn_ran(): hdev= of the enqueueing thread
            return _line("cuda", start, end - start, "host_func",
                         {"type": "host_func", "op": "host_func", "stream": stream, "side": "gpu", "rt": "cuda",
                          "timing": "host", "exec": "host", "lid": lid, "hdev": 0}, tid=tid)
        t = build([host("cudaLaunchKernel", 1000, 50, lid=1, corr=301, stream=11, hdev=0),
                   cupti_kernel(301, 7, 1100, 2000, name="k1"),
                   host("cudaLaunchHostFunc", 1060, 20, lid=2, op="host_func", stream=11, hdev=0),
                   hf(2, 11, 2010, 3000),
                   host("cudaLaunchHostFunc", 1090, 20, lid=3, op="host_func", stream=22, hdev=0),
                   hf(3, 22, 3010, 3500),
                   host("cudaLaunchKernel", 1120, 50, lid=4, corr=302, stream=11, hdev=0),
                   cupti_kernel(302, 7, 3020, 3600, name="k2")])
        e = edges(t, kinds=("sequential",))
        self.assertIn(("k1", "host_func", "sequential", "certain"), e)      # stream 11: kernel, function, kernel
        self.assertIn(("host_func", "k2", "sequential", "certain"), e)
        self.assertIn(("host_func", "host_func", "sequential", "high"), e)  # one callback thread


if __name__ == "__main__":
    unittest.main()
