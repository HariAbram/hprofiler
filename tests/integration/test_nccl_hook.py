"""
The NCCL hook against a stand-in libnccl (tests/fixtures/nccl_mock.cu: real
kernels on the caller's streams, grouped operations launched at
ncclGroupEnd as NCCL documents) driven by tests/fixtures/nccl_prog.c -- no
NCCL or second GPU needed, a CUDA device and nvcc are:

  * pseq= numbers the sends to / receives from each peer (NCCL pairs them
    in that order), across groups;
  * operations inside ncclGroupStart/End are host calls (timing=deferred,
    side=cpu, group=), the group's work one event-pair span per stream it used
    (type=group_exec), and the outermost group one type=group span;
  * HPROFILER_NCCL_ASYNC=1: every call is a host span (side=cpu, sid=) plus
    a device span resolved later (side=gpu, psid=, timing=async, ref_err=),
    none lost, device spans of one stream in FIFO order, none placed before
    its host call by more than the reference's uncertainty.

What it cannot show: real NCCL's internal behaviour (its kernels, proxy
threads, inter-GPU transfers). Skips without nvcc, a CUDA device or a hook
library that predates these tags.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from src.core.hookdir import hooks_dir  # noqa: E402  (HPROFILER_HOOKS_DIR, else build/lib)


def _profile(exe, env_extra):
    from src.core.runner import Runner
    with mock.patch.dict(os.environ, env_extra):
        trace = Runner([exe], backends=["nccl"]).run()
    return [s for s in trace.iter_spans(categories=("nccl",))], list(trace.iter_instants())


class TestNCCLHookWithMock(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not shutil.which("nvcc") or not (hooks_dir() / "libhprofiler_nccl.so").exists():
            raise unittest.SkipTest("needs nvcc and the built NCCL hook")
        cls.tmp = tempfile.mkdtemp()
        lib = os.path.join(cls.tmp, "libnccl_mock.so")
        exe = os.path.join(cls.tmp, "nccl_prog")
        cuda = Path(shutil.which("nvcc")).resolve().parent.parent
        for cmd in (["nvcc", "-shared", "-Xcompiler", "-fPIC", "-cudart", "shared", "-o", lib,
                     str(ROOT / "tests/fixtures/nccl_mock.cu")],
                    ["gcc", "-O1", str(ROOT / "tests/fixtures/nccl_prog.c"), "-o", exe,
                     f"-I{cuda}/include", f"-L{cls.tmp}", "-lnccl_mock", f"-L{cuda}/lib64", "-lcudart",
                     f"-Wl,-rpath,{cls.tmp}"]):
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                raise unittest.SkipTest(f"cannot build the fixture: {r.stderr[:300]}")
        probe = subprocess.run([exe], capture_output=True, text=True, timeout=120)
        if probe.returncode != 0:
            raise unittest.SkipTest(f"no usable CUDA device ({probe.stdout.strip()})")
        cls.sync, _ = _profile(exe, {"HPROFILER_NCCL_ASYNC": "0"})
        cls.asyn, cls.async_instants = _profile(exe, {"HPROFILER_NCCL_ASYNC": "1"})
        if not any("pseq" in s.tags for s in cls.sync):
            raise unittest.SkipTest("the hook library predates pseq= -- rebuild build/ or point "
                                    "HPROFILER_HOOKS_DIR at a fresh build")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(getattr(cls, "tmp", ""), ignore_errors=True)

    def test_p2p_sequence_numbers(self):
        seqs = defaultdict(list)
        for s in sorted(self.sync, key=lambda s: s.start_ns):
            if s.tags.get("type") in ("send", "recv"):
                seqs[(s.tags["type"], s.tags["peer"])].append(int(s.tags["pseq"]))
        self.assertEqual(dict(seqs), {("send", "1"): [1, 2, 3, 4], ("recv", "3"): [1, 2], ("send", "2"): [1]})

    def test_grouped_operations_are_deferred_host_calls(self):
        grouped = [s for s in self.sync if s.tags.get("timing") == "deferred"]
        self.assertTrue(all(s.tags.get("side") == "cpu" for s in grouped))
        self.assertEqual(sorted((s.tags["group"], s.name) for s in grouped),
                         [("1", "ncclRecv"), ("1", "ncclSend"), ("2", "ncclAllReduce"), ("2", "ncclSend")])
        outside = [s for s in self.sync if s.tags.get("type") in ("send", "recv", "allreduce")
                   and "group" not in s.tags]
        self.assertEqual(len(outside), 5)
        for s in outside:                       # event-pair timed: the ~200 us kernel
            self.assertNotIn("timing", s.tags)
            self.assertGreater(s.duration_ns, 100_000, s.name)

    def test_group_work_is_timed_per_stream(self):
        execs = [s for s in self.sync if s.tags.get("type") == "group_exec"]
        self.assertEqual(sorted((s.tags["group"], s.tags["ops"]) for s in execs),
                         [("1", "2"), ("1", "2"), ("2", "2"), ("2", "2")])
        for g in ("1", "2"):
            self.assertEqual(len({s.tags["stream"] for s in execs if s.tags["group"] == g}), 2)
        self.assertTrue(all(s.duration_ns > 100_000 for s in execs))
        groups = [s for s in self.sync if s.tags.get("type") == "group"]
        self.assertEqual(sorted(s.tags["group"] for s in groups), ["1", "2"])    # outermost only

    def test_async_device_spans_link_to_their_calls(self):
        host = {s.span_id: s for s in self.asyn if s.tags.get("side") == "cpu" and s.span_id}
        dev = [s for s in self.asyn if s.tags.get("side") == "gpu"]
        self.assertEqual(len(host), 9)          # 5 ops outside groups + 4 group_exec pairs
        self.assertEqual(sorted(d.parent_span_id for d in dev), sorted(host))
        self.assertFalse([i for i in self.async_instants if i.name == "nccl_async_lost"])
        for d in dev:
            self.assertEqual(d.tags["timing"], "async")
            h = host[d.parent_span_id]
            self.assertEqual((h.name, h.tags.get("stream")), (d.name, d.tags.get("stream")))
            self.assertGreaterEqual(d.start_ns + int(d.tags["ref_err"]), h.start_ns, d.name)
            self.assertGreater(d.duration_ns, 100_000, d.name)

    def test_async_device_spans_of_a_stream_do_not_overlap(self):
        by_stream = defaultdict(list)
        for d in self.asyn:
            if d.tags.get("side") == "gpu":
                by_stream[d.tags["stream"]].append(d)
        for spans in by_stream.values():
            spans.sort(key=lambda s: s.start_ns)
            for a, b in zip(spans, spans[1:]):
                self.assertLessEqual(a.start_ns + a.duration_ns, b.start_ns + 2_000, (a.name, b.name))


if __name__ == "__main__":
    unittest.main()
