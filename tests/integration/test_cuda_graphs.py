"""
CUDA graph structure, graph versions and stream / event lives through the
real CUDA hook and CUPTI on a GPU (tests/fixtures/cuda_graphs.cu, whose
header lists the graphs):

  * each exec's structure is recorded once (graph_exec / graph_nodes /
    graph_edges instants), on the exec's own node ids -- the ones its
    device records name (gnode=) -- with every node's type, a child graph's
    markers too;
  * the dependency graph follows the exec's edges: the fork / join of G1,
    the empty join node passed through, the disabled node passed through in
    the last launch, the edges around a child graph (whose own nodes CUPTI
    reports without a clone pairing: unmapped, ordered only by the launch
    stream -- reported, not guessed), and the captured fork / join of G3;
  * launches name their exec and its version (gexec= / gver=: 1, 1, 2 after
    cudaGraphExecUpdate, 3 after cudaGraphNodeSetEnabled);
  * a host function captured into a graph runs once per launch: the hook
    passes it through unwrapped (captured=1) -- it used to wrap it in a
    one-shot context that the second launch called after it was freed;
  * every stream's creation is reported once (stream_info) with its flags,
    priority and CUPTI stream id; a stream destroyed and re-created at the
    same address gets another stream= number, so its two lives are two
    lanes; an event likewise when the driver reuses its address.

Skips without nvcc, a usable GPU, CUPTI or a hook that predates gnode=.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

NVCC = shutil.which("nvcc") or ("/usr/local/cuda/bin/nvcc" if Path("/usr/local/cuda/bin/nvcc").exists() else None)


def _name(s):
    """kA from the mangled _Z2kAPi."""
    n = s.name
    if n.startswith("_Z") and n.endswith("Pi"):
        k = n[2:-2].lstrip("0123456789")
        return k
    return n


class TestCudaGraphs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.core.hookdir import hooks_dir
        if not NVCC or not (hooks_dir() / "libhprofiler_cuda.so").exists():
            raise unittest.SkipTest("needs nvcc and the built CUDA hook")
        cls.tmp = tempfile.mkdtemp()
        cls.exe = os.path.join(cls.tmp, "cuda_graphs")
        r = subprocess.run([NVCC, "-O1", "-Wno-deprecated-gpu-targets", "-cudart", "shared",
                            str(ROOT / "tests/fixtures/cuda_graphs.cu"), "-o", cls.exe],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise unittest.SkipTest(f"cannot build the fixture: {r.stderr[:300]}")
        base = subprocess.run([cls.exe], capture_output=True, text=True, timeout=120)
        if base.returncode != 0 or "update=" not in base.stdout:
            raise unittest.SkipTest(f"no usable GPU ({base.stdout.strip()[:120]})")
        cls.base = base.stdout
        from src.core.runner import Runner
        rf, wf = os.pipe()
        saved = os.dup(1)
        os.dup2(wf, 1)
        try:
            with mock.patch.dict(os.environ, {}):
                cls.trace = Runner([cls.exe], backends=["cuda"]).run()
        finally:
            os.dup2(saved, 1)
            os.close(wf)
        cls.out = os.read(rf, 65536).decode()
        os.close(rf)
        cls.spans = sorted(cls.trace.iter_spans(), key=lambda s: s.start_ns)
        if not any("gnode" in s.tags for s in cls.spans):
            raise unittest.SkipTest("no gnode= in the trace: CUPTI inactive, or the hook predates it")
        from src.analysis import criticalpath as cp
        cls.cp = cp
        cls.gspans, cls.preds = cp.build_dependency_graph(cls.trace)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(getattr(cls, "tmp", ""), ignore_errors=True)

    def launches(self):
        return [s for s in self.spans if s.tags.get("op") == "graph" and s.tags.get("side") == "cpu"]

    def graph_edges_of(self, launch):
        """{(src kernel, dst kernel)} graph_dep edges among one launch's work."""
        corr = launch.tags.get("corr")
        mine = {i for i, s in enumerate(self.gspans) if s.tags.get("gnode") and s.tags.get("corr") == corr}
        return {(_name(self.gspans[u]), _name(self.gspans[v]))
                for v, pl in self.preds.items() if v in mine
                for u, k, _c in pl if k == "graph_dep" and u in mine}

    def test_output_unchanged(self):
        self.assertEqual(self.out.split("update=")[1], self.base.split("update=")[1])

    def test_each_exec_structure_once_on_its_own_node_ids(self):
        execs = [i for i in self.trace.iter_instants() if i.tags.get("type") == "graph_exec"]
        self.assertEqual(len(execs), 4)                      # four instantiations, launched 9 times
        edges = {}
        for i in self.trace.iter_instants():
            if i.tags.get("type") == "graph_edges":
                for e in i.tags["edges"].split(";"):
                    a, b = e.rstrip("p").split("-")
                    edges.setdefault(i.tags["gexec"], set()).update((a, b))
        for s in self.spans:
            if s.tags.get("gnode") and s.tags.get("graph") in edges and _name(s) not in ("kX", "kY"):
                self.assertIn(s.tags["gnode"], edges[s.tags["graph"]], _name(s))
        # the node lists parse (a ':' in a tag value would fold the tags into the name)
        from src.analysis.semantics import gpu as sem_gpu
        self.assertFalse([i.name for i in self.trace.iter_instants() if i.name.startswith("cudaGraphNodes:")])
        structs = sem_gpu.graph_structures(self.trace.iter_instants())
        for (_pid, gexec), g in structs.items():
            self.assertTrue(g["types"], gexec)
            self.assertNotIn(-1, g["types"].values())
            self.assertTrue(set(g["types"]) >= {n for e in g["edges"] for n in e}, gexec)
        self.assertTrue(any(t in (100, 101) for g in structs.values() for t in g["types"].values()))   # child markers

    def test_a_captured_host_function_runs_on_every_launch(self):
        self.assertIn("host_runs=3", self.out)
        self.assertIn("host_runs=3", self.base)
        enq = [s for s in self.spans if s.name == "cudaLaunchHostFunc"]
        self.assertEqual([s.tags.get("captured") for s in enq], ["1"])
        self.assertFalse([s for s in self.spans if s.tags.get("op") == "host_func" and s.tags.get("side") == "gpu"])
        g4 = self.launches()[6:]
        self.assertEqual(len(g4), 3)
        kernels = sorted(_name(s) for s in self.spans if s.tags.get("side") == "gpu" and _name(s) in ("kM", "kN"))
        self.assertEqual(kernels, ["kM"] * 3 + ["kN"] * 3)

    def test_versions(self):
        g1 = [s.tags.get("gver") for s in self.launches()][:4]
        self.assertEqual(g1, ["1", "1", "2", "3"])
        updates = [i.tags["api"] for i in self.trace.iter_instants() if i.tags.get("type") == "graph_exec_update"]
        self.assertEqual(updates, ["cuGraphExecUpdate", "cuGraphNodeSetEnabled"])

    def test_fork_join_and_the_empty_join(self):
        want = {("kA", "kB"), ("kA", "kC"), ("kB", "kD"), ("kC", "kD")}
        for launch in self.launches()[:3]:
            self.assertEqual(self.graph_edges_of(launch), want)

    def test_a_disabled_node_is_passed_through(self):
        self.assertEqual(self.graph_edges_of(self.launches()[3]), {("kA", "kC"), ("kA", "kD"), ("kC", "kD")})

    def test_around_a_child_graph(self):
        got = self.graph_edges_of(self.launches()[4])
        self.assertIn(("kP", "kQ"), got)                     # through the child's markers
        self.assertFalse({e for e in got if "kX" in e or "kY" in e})   # unmapped: not placed
        diag = self.cp.dependency_diagnostics(self.trace)
        self.assertEqual(diag.get("gpu_graph_unmapped_nodes"), 2)
        self.assertNotIn("gpu_graph_launches_unstructured", diag)

    def test_captured_fork_join(self):
        self.assertEqual(self.graph_edges_of(self.launches()[5]),
                         {("kF", "kG"), ("kF", "kH"), ("kG", "kJ"), ("kH", "kJ")})

    def test_a_graph_runs_as_one_group_on_its_launch_stream(self):
        # kB ran on an internal stream (nstream differs) but is ordered on the launch stream
        launch = self.launches()[0]
        kB = next(i for i, s in enumerate(self.gspans) if _name(s) == "kB" and s.tags.get("corr") == launch.tags["corr"])
        kinds = {k for _u, k, _c in self.preds.get(kB, [])}
        self.assertIn("graph_dep", kinds)
        self.assertIn("launch", kinds)

    def test_stream_info_once_per_stream_with_flags(self):
        infos = [s for s in self.spans if s.tags.get("op") == "stream_info"]
        created = [s for s in self.spans if s.tags.get("op") == "stream_create"]
        self.assertEqual(len(infos), len(created))
        self.assertEqual(len({s.tags["nstream"] for s in infos}), len(infos))
        prio = next(s for s in infos if s.tags.get("priority") not in (None, "0"))
        self.assertEqual(prio.tags["nonblocking"], "1")
        self.assertTrue(self.base.strip().endswith(f"prio={prio.tags['priority']}"))

    def test_a_reused_stream_handle_is_a_new_stream(self):
        self.assertIn("stream_reused=1", self.base)
        l1 = next(s for s in self.spans if _name(s) == "kL1" and s.tags.get("side") == "gpu")
        l2 = next(s for s in self.spans if _name(s) == "kL2" and s.tags.get("side") == "gpu")
        self.assertNotEqual(l1.tags["stream"], l2.tags["stream"])
        destroy = next(s for s in self.spans if s.tags.get("op") == "stream_destroy")
        self.assertEqual(destroy.tags["stream"], l1.tags["stream"])     # names the life that ended
        lanes = {ln.name for ln in self.trace.store.lane_infos()}
        self.assertIn(f"cuda/stream-{l1.tags['stream']}", lanes)
        self.assertIn(f"cuda/stream-{l2.tags['stream']}", lanes)

    def test_a_reused_event_handle_is_a_new_event(self):
        records = [s.tags["event"] for s in self.spans if s.tags.get("op") == "event_record"][-2:]
        if "event_reused=1" not in self.base:
            self.skipTest("the driver did not reuse an event address in this run")
        self.assertNotEqual(records[0], records[1])


if __name__ == "__main__":
    unittest.main()
