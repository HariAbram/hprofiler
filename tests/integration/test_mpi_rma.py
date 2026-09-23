"""
Integration test for hooks/mpi_hook/mpi_hook.c's one-sided (RMA)
synchronization wrappers -- MPI_Win_fence, MPI_Win_flush, MPI_Win_flush_all,
MPI_Win_lock, MPI_Win_lock_all, MPI_Win_unlock, MPI_Win_unlock_all. Added
alongside MPI_Put/Get/Accumulate (which already existed) after an audit
found the standard permits an MPI implementation to defer RMA completion
past a Put/Get/Accumulate call's own return -- real completion is only
guaranteed after one of these synchronization calls, none of which were
previously intercepted at all, so a deferred-completion transport would
have that transfer's real cost attributed to nothing in the trace.

Builds the real libhprofiler_mpi.so, LD_PRELOADs it into
tests/fixtures/mpi_win_self.c, and captures the actual wire lines the hook
emits over a real AF_UNIX socket -- same harness as test_mpi_protocol.py
(FakeListener duplicated here rather than imported, matching that file's
own self-contained convention). Uses MPI_COMM_SELF for the same reason
test_mpi_protocol.py's mpi_proto_self.c does: this dev machine's
MPICH/Hydra cannot form a real multi-rank MPI_COMM_WORLD.

Skips (does not fail) if mpicc or a working MPI environment is
unavailable, consistent with the rest of tests/integration/.
"""
from __future__ import annotations
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from src.core.runner import _parse_record
from src.core.events import SpanEvent


class FakeListener:
    """Minimal stand-in for Runner's socket server -- see
    test_mpi_protocol.py's identical class for the full rationale."""

    def __init__(self):
        self.dir = tempfile.mkdtemp(prefix="hprofiler_mpi_rma_test_")
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
        deadline = time.time() + timeout
        for t in list(self._readers):
            remaining = max(0.0, deadline - time.time())
            t.join(timeout=remaining)

    def stop(self):
        self._stop = True
        try:
            self.srv.close()
        except OSError:
            pass
        shutil.rmtree(self.dir, ignore_errors=True)


def _mpicc_available() -> bool:
    return shutil.which("mpicc") is not None and shutil.which("mpirun") is not None


@unittest.skipUnless(_mpicc_available(), "mpicc/mpirun not available")
class TestMpiRmaSync(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="hprofiler_mpi_rma_build_")
        cls.hook_so = os.path.join(cls.tmp, "libhprofiler_mpi_test.so")
        cls.fixture_bin = os.path.join(cls.tmp, "mpi_win_self")

        build_hook = subprocess.run(
            ["mpicc", "-shared", "-fPIC", "-O2", "-o", cls.hook_so,
             str(REPO / "hooks" / "mpi_hook" / "mpi_hook.c"), "-ldl", "-lpthread"],
            capture_output=True, text=True,
        )
        if build_hook.returncode != 0:
            raise unittest.SkipTest(f"failed to build mpi_hook.c: {build_hook.stderr[-2000:]}")

        build_fixture = subprocess.run(
            ["mpicc", "-O2", "-o", cls.fixture_bin,
             str(REPO / "tests" / "fixtures" / "mpi_win_self.c")],
            capture_output=True, text=True,
        )
        if build_fixture.returncode != 0:
            raise unittest.SkipTest(f"failed to build mpi_win_self.c: {build_fixture.stderr[-2000:]}")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _run_and_capture(self) -> list:
        listener = FakeListener()
        try:
            env = dict(os.environ)
            env["HPROFILER_SOCKET"] = listener.path
            env["LD_PRELOAD"] = self.hook_so
            proc = subprocess.run([self.fixture_bin], env=env,
                                  capture_output=True, text=True, timeout=30)
            self.assertEqual(proc.returncode, 0,
                             f"fixture exited non-zero: stdout={proc.stdout!r} stderr={proc.stderr!r}")
            listener.drain(timeout=2.0)
            events = [e for e in (_parse_record(l) for l in listener.lines) if e is not None]
            self.assertTrue(events, f"no events captured; raw lines={listener.lines!r} "
                                    f"stderr={proc.stderr!r}")
            return events
        finally:
            listener.stop()

    def _spans_named(self, events, name):
        return [e for e in events if isinstance(e, SpanEvent) and e.name == name]

    def test_win_fence_emitted_as_mpi_category(self):
        events = self._run_and_capture()
        fences = self._spans_named(events, "MPI_Win_fence")
        # The fixture calls fence 3 times (before Put, before Get, after Get).
        self.assertEqual(len(fences), 3)
        for f in fences:
            self.assertEqual(f.category.value, "mpi")
            self.assertEqual(f.tags.get("type"), "win_fence")

    def test_win_lock_unlock_pair_emitted(self):
        events = self._run_and_capture()
        locks = self._spans_named(events, "MPI_Win_lock")
        unlocks = self._spans_named(events, "MPI_Win_unlock")
        self.assertEqual(len(locks), 1)
        self.assertEqual(len(unlocks), 1)
        self.assertEqual(locks[0].tags.get("peer"), "0")
        self.assertEqual(unlocks[0].tags.get("peer"), "0")

    def test_win_flush_emitted_between_lock_and_unlock(self):
        events = self._run_and_capture()
        flushes = self._spans_named(events, "MPI_Win_flush")
        self.assertEqual(len(flushes), 1)
        lock = self._spans_named(events, "MPI_Win_lock")[0]
        unlock = self._spans_named(events, "MPI_Win_unlock")[0]
        self.assertLessEqual(lock.start_ns, flushes[0].start_ns)
        self.assertLessEqual(flushes[0].start_ns, unlock.start_ns)

    def test_win_lock_all_flush_all_unlock_all_emitted(self):
        events = self._run_and_capture()
        self.assertEqual(len(self._spans_named(events, "MPI_Win_lock_all")), 1)
        self.assertEqual(len(self._spans_named(events, "MPI_Win_flush_all")), 1)
        self.assertEqual(len(self._spans_named(events, "MPI_Win_unlock_all")), 1)

    def test_put_get_accumulate_still_emitted_alongside_new_sync_calls(self):
        # Regression guard: adding the new Win_* wrappers must not have
        # disturbed the pre-existing Put/Get/Accumulate wrappers.
        events = self._run_and_capture()
        self.assertEqual(len(self._spans_named(events, "MPI_Put")), 1)
        self.assertEqual(len(self._spans_named(events, "MPI_Get")), 1)
        self.assertEqual(len(self._spans_named(events, "MPI_Accumulate")), 1)


if __name__ == "__main__":
    unittest.main()
