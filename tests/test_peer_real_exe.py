"""
Tests for src/core/runner.py's _peer_real_exe(): resolves the REAL
executable path of whatever process is on the other end of a hook's
Unix-domain socket connection, via SO_PEERCRED (a kernel-verified
credential -- not anything a hook has to self-report).

Why this exists: CUDA/ROCm AoT disassembly (disasm_cuda_sass/
disasm_rocm_binary) disassembles the WHOLE BINARY, keyed off
command[0] -- unlike OpenMP/MPI's disasm, which resolves a specific
call site via dladdr from inside the profiled process (sym=/symfile=
tags) and so is already launcher-agnostic. When command[0] is a
launcher (`hprofiler run -- srun -n 4 gmx_mpi ...`), it's "srun", never
the real GPU binary, and there was no fallback at all for this
whole-binary case. Every hook connects to HPROFILER_SOCKET from INSIDE
the real profiled process (how LD_PRELOAD hooking works), so the peer
credentials of that connection give the real PID for free.

socket.socketpair() gives a real, connected pair of Unix-domain sockets
within this SAME test process -- both ends' SO_PEERCRED naturally
resolve to this test process's own PID, no launcher/subprocess
simulation needed to test the mechanism itself for real.
"""
import os
import socket
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.runner import _peer_real_exe


class TestPeerRealExe(unittest.TestCase):
    def test_resolves_this_process_own_exe_via_a_real_socketpair(self):
        a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            real_exe = _peer_real_exe(a)
        finally:
            a.close()
            b.close()
        self.assertEqual(real_exe, os.readlink("/proc/self/exe"))

    def test_returns_empty_string_for_a_closed_socket(self):
        a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        b.close()
        a.close()
        # SO_PEERCRED (or the /proc/<pid>/exe readlink) on an already-
        # closed socket must fail safely, not raise -- a process that
        # connected then exited before disasm collection runs is exactly
        # the case this needs to tolerate, not a rare edge case.
        self.assertEqual(_peer_real_exe(a), "")

    def test_returns_empty_string_for_a_non_socket_object(self):
        class _NotASocket:
            def getsockopt(self, *a, **k):
                raise OSError("not a real socket")
        self.assertEqual(_peer_real_exe(_NotASocket()), "")


if __name__ == "__main__":
    unittest.main()
