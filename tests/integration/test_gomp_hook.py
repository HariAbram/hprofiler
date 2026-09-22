"""
Integration test for hooks/gomp_hook/gomp_hook.c -- the direct GOMP_*
interception path added for binaries linked against GNU's libgomp (which
has no OMPT support in typical builds; hooks/ompt_tool/ompt_tool.c's OMPT
path produces zero events for such binaries -- see both files' header
comments, src/backends/openmp.py's module docstring, and
project_causal_attribution_redesign memory for the real-world trigger: a
user profiling GROMACS on the Dardel HPC cluster, where `ldd gmx_mpi`
confirmed libgomp.so.1, not libomp).

Builds the real hook, compiles tests/fixtures/gomp_mini.c with plain gcc
(confirmed via `ldd` to link libgomp, not libomp -- the whole point),
LD_PRELOADs the hook, captures the actual wire-protocol bytes over a real
AF_UNIX socket, and asserts on resolved per-thread/per-construct
correctness -- not just "some events showed up" (tests/integration/
run_matrix.sh's gomp section already covers that crash-safety /
nonzero-event level; this is the measurement-correctness level, same
split as test_mpi_protocol.py for the MPI hook).

Skips (does not fail) if gcc/gcc-openmp is unavailable, consistent with
the rest of tests/integration/.
"""
from __future__ import annotations
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from src.core.runner import _parse_record
from src.core.events import SpanEvent, InstantEvent


class FakeListener:
    """Same minimal AF_UNIX capture harness as test_mpi_protocol.py."""

    def __init__(self):
        self.dir = tempfile.mkdtemp(prefix="hprofiler_gomp_test_")
        self.path = os.path.join(self.dir, "sock")
        self.srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.srv.bind(self.path)
        self.srv.listen(8)
        self.srv.settimeout(0.2)
        self.lines: list[str] = []
        self._lock = threading.Lock()
        self._stop = False
        self._readers: list[threading.Thread] = []
        self._acceptor = threading.Thread(target=self._accept_loop, daemon=True)
        self._acceptor.start()

    def _accept_loop(self):
        while not self._stop:
            try:
                conn, _ = self.srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            t = threading.Thread(target=self._read_loop, args=(conn,), daemon=True)
            t.start()
            self._readers.append(t)

    def _read_loop(self, conn: socket.socket):
        buf = b""
        conn.settimeout(5.0)
        try:
            while True:
                data = conn.recv(4096)
                if not data:
                    break
                buf += data
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    with self._lock:
                        self.lines.append(line.decode("utf-8", "replace"))
        except OSError:
            pass
        finally:
            conn.close()

    def drain(self, timeout=2.0):
        for t in list(self._readers):
            t.join(timeout=timeout)

    def stop(self):
        self._stop = True
        try:
            self.srv.close()
        except OSError:
            pass
        shutil.rmtree(self.dir, ignore_errors=True)


def _gcc_openmp_available() -> bool:
    if shutil.which("gcc") is None:
        return False
    probe = subprocess.run(
        ["gcc", "-fopenmp", "-x", "c", "-o", os.devnull, "-"],
        input="int main(void){return 0;}", capture_output=True, text=True,
    )
    return probe.returncode == 0


@unittest.skipUnless(_gcc_openmp_available(), "gcc -fopenmp not available")
class TestGompHook(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="hprofiler_gomp_build_")
        cls.hook_so = os.path.join(cls.tmp, "libhprofiler_gomp_test.so")
        cls.fixture_bin = os.path.join(cls.tmp, "gomp_mini")

        build_hook = subprocess.run(
            ["gcc", "-shared", "-fPIC", "-O2", "-o", cls.hook_so,
             str(REPO / "hooks" / "gomp_hook" / "gomp_hook.c"), "-ldl", "-lpthread"],
            capture_output=True, text=True,
        )
        if build_hook.returncode != 0:
            raise unittest.SkipTest(f"failed to build gomp_hook.c: {build_hook.stderr[-2000:]}")

        build_fixture = subprocess.run(
            ["gcc", "-O0", "-fopenmp", "-o", cls.fixture_bin,
             str(REPO / "tests" / "fixtures" / "gomp_mini.c")],
            capture_output=True, text=True,
        )
        if build_fixture.returncode != 0:
            raise unittest.SkipTest(f"failed to build gomp_mini.c: {build_fixture.stderr[-2000:]}")

        ldd = subprocess.run(["ldd", cls.fixture_bin], capture_output=True, text=True)
        if "libgomp" not in ldd.stdout:
            raise unittest.SkipTest(
                f"gomp_mini did not link libgomp as expected -- ldd output: {ldd.stdout}")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _run_and_capture(self):
        listener = FakeListener()
        try:
            env = dict(os.environ)
            env["HPROFILER_SOCKET"] = listener.path
            env["LD_PRELOAD"] = self.hook_so
            proc = subprocess.run([self.fixture_bin], env=env,
                                  capture_output=True, text=True, timeout=30)
            self.assertEqual(proc.returncode, 0,
                             f"fixture exited non-zero: stdout={proc.stdout!r} stderr={proc.stderr!r}")
            self.assertIn("sum=6", proc.stdout,
                          "interception must not change program behavior/correctness")
            listener.drain(timeout=2.0)
            events = [e for e in (_parse_record(l) for l in listener.lines) if e is not None]
            self.assertTrue(events, f"no events captured; raw lines={listener.lines!r}")
            return events
        finally:
            listener.stop()

    def test_parallel_region_gives_one_span_per_thread(self):
        events = self._run_and_capture()
        regions = [e for e in events if isinstance(e, SpanEvent) and e.name == "omp_parallel_region"]
        self.assertEqual(len(regions), 4, "num_threads(4) must give exactly 4 per-thread spans")
        tids = {e.tid for e in regions}
        self.assertEqual(len(tids), 4, "each parallel-region span must be on a distinct real thread id")
        for e in regions:
            self.assertGreater(e.duration_ns, 0)

    def test_explicit_and_implicit_barriers_captured(self):
        events = self._run_and_capture()
        barriers = [e for e in events if isinstance(e, SpanEvent) and e.name == "omp_barrier"]
        # 4 threads x (1 implicit end-of-for barrier + 1 explicit "#pragma
        # omp barrier") = 8 -- verified stable across repeated runs during
        # development after adding the destructor flush (see gomp_hook.c's
        # destructor comment); a flaky count here would mean that
        # regressed.
        self.assertEqual(len(barriers), 8)
        for e in barriers:
            self.assertEqual(e.category.value, "sync")

    def test_critical_sections_wait_and_hold_both_captured(self):
        events = self._run_and_capture()
        waits = [e for e in events if isinstance(e, SpanEvent) and e.name == "omp_critical_wait"]
        holds = [e for e in events if isinstance(e, SpanEvent) and e.name == "omp_critical_hold"]
        # One anonymous + one named critical section x 4 threads each.
        self.assertEqual(len(waits), 8)
        self.assertEqual(len(holds), 8)
        named_waits = [e for e in waits if e.tags.get("named") == "1"]
        self.assertEqual(len(named_waits), 4)
        # Hold spans include real work (usleep(1000) or usleep(500) in the
        # fixture) -- must reflect genuine elapsed time, not a zero stub.
        for e in holds:
            self.assertGreater(e.duration_ns, 100_000)  # > 100us, well under the 500-1000us usleep

    def test_single_region_ran_on_exactly_one_thread(self):
        events = self._run_and_capture()
        singles = [e for e in events if isinstance(e, InstantEvent) and e.name == "omp_single"]
        self.assertEqual(len(singles), 1, "exactly one thread executes a #pragma omp single region")

    def test_spans_carry_a_resolved_codeptr_tag_for_disasm(self):
        # Regression test: gomp_hook.c used to capture codeptr_ra (in
        # GOMP_parallel's closure) but never resolve or emit it, so every
        # span from this hook had no sym=/lib= tag at all -- the Source
        # tab's "No disassembly available" was unconditional for GNU-
        # libgomp binaries, not a missing-objdump problem. At least one of
        # sym=/lib= must be present so src/core/runner.py's
        # _collect_disasm() has something to disassemble (the user's own
        # call site -- there's no ELF symbol literally named
        # "omp_parallel_region" for objdump to find on its own).
        events = self._run_and_capture()
        for name in ("omp_parallel_region", "omp_barrier", "omp_critical_wait"):
            spans = [e for e in events if isinstance(e, SpanEvent) and e.name == name]
            self.assertTrue(spans, f"no {name} spans captured")
            resolved = [e for e in spans if e.tags.get("sym") or e.tags.get("lib")]
            self.assertTrue(
                resolved,
                f"none of the {len(spans)} {name!r} spans carry a sym=/lib= tag; "
                f"sample tags={spans[0].tags!r}",
            )


if __name__ == "__main__":
    unittest.main()
