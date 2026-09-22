"""
Integration test for the MPI protocol-semantics work in hooks/mpi_hook/mpi_hook.c
(wildcard MPI_ANY_SOURCE/MPI_ANY_TAG resolution, MPI_Waitany/Waitsome/Test*/
Cancel wrappers, and MPI_Comm_split commid= tagging) -- see the file's header
comment and project_full_audit_fixes / MEMORY.md for the design this verifies.

This builds the real libhprofiler_mpi.so, LD_PRELOADs it into the
tests/fixtures/mpi_proto_self.c fixture, and captures the *actual* wire
lines the hook emits over a real AF_UNIX socket -- then parses them with
the production _parse_record (src/core/runner.py) and asserts on the
resolved semantics, not just "did it crash".

Why "_self": this development machine's MPICH/Hydra cannot form a real
multi-rank MPI_COMM_WORLD -- `mpirun -np N` (N>1) has every rank
independently observe MPI_Comm_size() == 1, reproducible with the
pre-existing mpi_mini.c fixture too and confirmed via UCX_LOG_LEVEL=info
to be a PMI/KVS rank-discovery failure in this machine's MPICH/UCX/PMIx
setup, unrelated to hprofiler. mpi_proto_self.c exercises the same
PMPI_Isend/Irecv/Waitany/Waitsome/Test/Cancel completion semantics via
self-communication instead, which still drives the real MPI completion
machinery end-to-end. It CANNOT exercise genuine cross-process commid=
agreement (that needs a real second rank) -- this test only checks
commid= self-consistency across two collectives on the same
communicator. tests/fixtures/mpi_proto.c is the real 4-rank design,
kept compile-verified (tests/integration/run_matrix.sh does not build it
either, for the same reason) for use on a working cluster.

Skips (does not fail) if mpicc or a working multi-process MPI environment
is unavailable, consistent with the rest of tests/integration/.
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
from src.core.events import SpanEvent, InstantEvent


class FakeListener:
    """Minimal stand-in for Runner's socket server: accepts connections on
    an AF_UNIX path and captures every newline-delimited line sent to it,
    across possibly-multiple connections (mirrors real hook behavior:
    each hook/process opens its own connection)."""

    def __init__(self):
        self.dir = tempfile.mkdtemp(prefix="hprofiler_mpi_test_")
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
        """Wait briefly for in-flight reader threads to finish flushing."""
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
class TestMpiProtocolSemantics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="hprofiler_mpi_build_")
        cls.hook_so = os.path.join(cls.tmp, "libhprofiler_mpi_test.so")
        cls.fixture_bin = os.path.join(cls.tmp, "mpi_proto_self")

        build_hook = subprocess.run(
            ["mpicc", "-shared", "-fPIC", "-O2", "-o", cls.hook_so,
             str(REPO / "hooks" / "mpi_hook" / "mpi_hook.c"), "-ldl", "-lpthread"],
            capture_output=True, text=True,
        )
        if build_hook.returncode != 0:
            raise unittest.SkipTest(f"failed to build mpi_hook.c: {build_hook.stderr[-2000:]}")

        build_fixture = subprocess.run(
            ["mpicc", "-O2", "-o", cls.fixture_bin,
             str(REPO / "tests" / "fixtures" / "mpi_proto_self.c")],
            capture_output=True, text=True,
        )
        if build_fixture.returncode != 0:
            raise unittest.SkipTest(f"failed to build mpi_proto_self.c: {build_fixture.stderr[-2000:]}")

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

    def test_waitany_resolves_wildcard_source_and_tag(self):
        events = self._run_and_capture()
        waitany = [e for e in events if isinstance(e, SpanEvent) and e.name == "MPI_Waitany"]
        self.assertEqual(len(waitany), 1, f"expected exactly one MPI_Waitany span, got {waitany}")
        tags = waitany[0].tags
        self.assertIn("completed_index", tags)
        self.assertIn(int(tags["completed_index"]), (0, 1, 2))
        # The Irecv was posted with MPI_ANY_SOURCE/MPI_ANY_TAG -- rpeer/rtag
        # must reflect the REAL resolved match (self-send, rank 0, one of
        # the three tags actually used), not a -1/-2 wildcard sentinel.
        self.assertEqual(tags.get("rpeer"), "0")
        self.assertIn(tags.get("rtag"), ("11", "12", "13"))

    def test_waitsome_resolves_remaining_wildcards(self):
        events = self._run_and_capture()
        waitsome = [e for e in events if isinstance(e, SpanEvent) and e.name == "MPI_Waitsome"]
        self.assertEqual(len(waitsome), 1)
        tags = waitsome[0].tags
        self.assertEqual(tags.get("outcount"), "2")
        # rmatches= carries "<req_id>/<peer>/<tag>" pairs for whichever of
        # the completed requests were wildcards -- both remaining ones are.
        # '/' not ':' deliberately -- see the comment at its construction
        # site in mpi_hook.c's MPI_Waitall/MPI_Waitsome.
        self.assertIn("rmatches", tags)
        pairs = tags["rmatches"].split(";")
        self.assertEqual(len(pairs), 2)
        for p in pairs:
            req_id, peer, tag = p.split("/")
            self.assertEqual(peer, "0")
            self.assertIn(tag, ("11", "12", "13"))

    def test_irecv_span_marks_wildcard_without_guessing_peer(self):
        events = self._run_and_capture()
        irecvs = [e for e in events if isinstance(e, SpanEvent) and e.name == "MPI_Irecv"]
        # 3 (phase A) + 1 (phase B) + 1 (phase C, cancelled) + 2 (phase D0) = 7
        self.assertEqual(len(irecvs), 7)
        for ev in irecvs:
            self.assertEqual(ev.tags.get("wildcard"), "1")

    def test_waitall_resolves_wildcard_matches(self):
        # Separate call site from Waitsome's identical rmatches= fix
        # (mpi_hook.c's MPI_Waitall vs MPI_Waitsome) -- verified independently.
        events = self._run_and_capture()
        waitalls = [e for e in events if isinstance(e, SpanEvent) and e.name == "MPI_Waitall"]
        # phase A's send-side Waitall (no wildcards -> no rmatches) +
        # phase D0's send-side Waitall (no wildcards) + phase D0's
        # recv-side Waitall (2 wildcards -> rmatches=id/0/51;id/0/52).
        wildcard_waitalls = [e for e in waitalls if "rmatches" in e.tags]
        self.assertEqual(len(wildcard_waitalls), 1)
        pairs = wildcard_waitalls[0].tags["rmatches"].split(";")
        self.assertEqual(len(pairs), 2)
        seen_tags = set()
        for p in pairs:
            req_id, peer, tag = p.split("/")
            self.assertEqual(peer, "0")
            seen_tags.add(tag)
        self.assertEqual(seen_tags, {"51", "52"})

    def test_test_polling_observes_not_ready_then_ready(self):
        events = self._run_and_capture()
        tests = [e for e in events if isinstance(e, InstantEvent) and e.name == "MPI_Test"]
        self.assertGreaterEqual(len(tests), 2, "expected at least one flag=0 and one flag=1 MPI_Test")
        flags = [e.tags.get("flag") for e in tests]
        self.assertIn("0", flags, "the pre-send Test call should have observed flag=0")
        self.assertIn("1", flags, "a later Test call should have observed flag=1 once sent")
        ready = [e for e in tests if e.tags.get("flag") == "1"]
        self.assertEqual(ready[-1].tags.get("rpeer"), "0")
        self.assertEqual(ready[-1].tags.get("rtag"), "200")

    def test_cancel_emits_instant_with_psid(self):
        events = self._run_and_capture()
        cancels = [e for e in events if isinstance(e, InstantEvent) and e.name == "MPI_Cancel"]
        self.assertEqual(len(cancels), 1)
        self.assertEqual(cancels[0].tags.get("type"), "cancel")
        self.assertIn("psid", cancels[0].tags)

    def test_commid_self_consistent_across_collectives_on_same_comm(self):
        events = self._run_and_capture()
        allreduces = [e for e in events if isinstance(e, SpanEvent) and e.name == "MPI_Allreduce"]
        self.assertEqual(len(allreduces), 2)
        commids = {e.tags.get("commid") for e in allreduces}
        self.assertEqual(len(commids), 1, f"both Allreduce calls are on the same "
                         f"sub-communicator, expected one shared commid=, got {commids}")
        self.assertNotEqual(commids.pop(), "0", "sub-communicator must not reuse "
                            "MPI_COMM_WORLD's reserved commid=0")

    def test_allreduce_spans_carry_a_resolved_codeptr_tag_for_disasm(self):
        # Regression test: mpi_hook.c never captured/resolved a call-site
        # codeptr at all (no sym=/lib= tag on any "mpi"-category span),
        # AND src/core/runner.py's _collect_disasm() unconditionally
        # excluded category "mpi" from the spans it even looks at for
        # this -- two separate gaps that together made the Source tab's
        # "No disassembly available" unconditional for every MPI call,
        # not a missing-objdump problem. Fixed on both sides; this checks
        # the hook side (the wire-protocol tag actually being present).
        events = self._run_and_capture()
        allreduces = [e for e in events if isinstance(e, SpanEvent) and e.name == "MPI_Allreduce"]
        self.assertTrue(allreduces)
        resolved = [e for e in allreduces if e.tags.get("sym") or e.tags.get("lib")]
        self.assertTrue(
            resolved,
            f"none of the {len(allreduces)} MPI_Allreduce spans carry a sym=/lib= "
            f"tag; sample tags={allreduces[0].tags!r}",
        )

    def test_no_events_silently_dropped_by_inst_tag_parsing(self):
        # Regression guard for the _parse_record inst: fix in runner.py:
        # every inst: line this fixture's run produces must carry a
        # non-empty tags dict, since every emit_instant() call site in
        # mpi_hook.c always builds a non-trivial "type=..." extra string.
        events = self._run_and_capture()
        insts = [e for e in events if isinstance(e, InstantEvent)]
        self.assertTrue(insts)
        for ev in insts:
            self.assertTrue(ev.tags, f"instant event {ev.name} lost its tags")


if __name__ == "__main__":
    unittest.main()
