"""
How `Runner` starts and finishes the profiled program:

  * file descriptors are inherited exactly as without hprofiler (an MPI
    launcher's PMI socket, PMI_FD, must survive), and none of hprofiler's
    own descriptors leak into the program;
  * the program's exit status is recorded (metadata, JSON round trip) and
    explains a zero-event run;
  * the collector socket falls back to a short directory, then to the
    abstract namespace, when $TMPDIR is too long for an AF_UNIX path -- and
    the hooks still connect (checked with the GNU libgomp hook).
"""
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.core import runner as runner_mod
from src.core.runner import Runner
from src.core.trace_io import create_disk_trace, open_trace

_CHILD = textwrap.dedent("""
    import os, sys
    fd = int(os.environ["HB_TEST_FD"])
    os.write(fd, b"inherited-fd-works")
    os.close(fd)
    fds = sorted(int(x) for x in os.listdir("/proc/self/fd"))
    print(" ".join(map(str, fds)))
    sys.exit(int(os.environ.get("HB_EXIT", "0")))
""")


def _fd_targets(fds_line: str, pid_hint=None) -> list[int]:
    return [int(x) for x in fds_line.split()]


class TestFdInheritance(unittest.TestCase):
    def _run(self, store=None, exit_code=0):
        r, w = os.pipe()
        os.set_inheritable(w, True)
        env = {"HB_TEST_FD": str(w), "HB_EXIT": str(exit_code)}
        out = tempfile.NamedTemporaryFile(mode="w+", suffix=".out", delete=False)
        try:
            cmd = [sys.executable, "-c", _CHILD]
            with mock.patch.dict(os.environ, env):
                # stdout of the child goes to a file we can read afterwards
                saved = os.dup(1)
                os.dup2(out.fileno(), 1)
                try:
                    runner = Runner(cmd, backends=[], store=store)
                    trace = runner.run()
                finally:
                    os.dup2(saved, 1)
                    os.close(saved)
            os.close(w)
            data = os.read(r, 100)
            os.close(r)
            out.seek(0)
            fds_line = out.read().strip().splitlines()[-1]
            return runner, trace, data, fds_line
        finally:
            out.close()
            os.unlink(out.name)

    def _unprofiled_fds(self):
        r, w = os.pipe()
        os.set_inheritable(w, True)
        env = dict(os.environ, HB_TEST_FD=str(w))
        with tempfile.TemporaryFile(mode="w+") as out:
            subprocess.run([sys.executable, "-c", _CHILD], env=env, stdout=out, close_fds=False,
                           check=True)
            out.seek(0)
            line = out.read().strip()
        os.close(w)
        os.close(r)
        return line

    def test_inherited_fd_is_usable_in_the_program(self):
        _runner, _trace, data, _fds = self._run()
        self.assertEqual(data, b"inherited-fd-works")

    def test_program_sees_the_same_descriptors_as_without_hprofiler(self):
        tmp = tempfile.mkdtemp()
        try:
            capture = create_disk_trace(str(Path(tmp) / "t.hpstore"))
            _runner, _trace, _data, fds = self._run(store=capture.store)
            # the descriptor numbers can differ (the parent's pipe ends are
            # numbered by allocation order); the COUNT must not grow
            self.assertEqual(len(fds.split()), len(self._unprofiled_fds().split()),
                             f"profiled: {fds!r} unprofiled: {self._unprofiled_fds()!r}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestExitStatus(unittest.TestCase):
    def test_exit_status_recorded_and_round_tripped(self):
        tmp = tempfile.mkdtemp()
        try:
            capture = create_disk_trace(str(Path(tmp) / "t.hpstore"))
            runner = Runner([sys.executable, "-c", "import sys; sys.exit(3)"], backends=[],
                            store=capture.store)
            trace = runner.run()
            self.assertEqual(runner.exit_status, 3)
            self.assertEqual(trace.metadata.exit_status, 3)
            trace.finalize()
            from src.core.trace_io import write_json_export
            only_json = Path(tmp) / "json-only"
            only_json.mkdir()
            jp = only_json / "t.json"            # no sibling store: the JSON path is read
            write_json_export(trace, str(jp))
            self.assertEqual(open_trace(str(jp)).metadata.exit_status, 3)
            trace.save()
            self.assertEqual(open_trace(str(Path(tmp) / "t.hpstore")).metadata.exit_status, 3)
            from src.core.receiver import run_warnings
            self.assertTrue(any("exited with status 3" in w for w in run_warnings(trace.metadata)))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_signal_is_negative(self):
        runner = Runner([sys.executable, "-c", "import os, signal; os.kill(os.getpid(), signal.SIGTERM)"],
                        backends=[])
        runner.run()
        self.assertEqual(runner.exit_status, -15)

    def test_zero_event_warning_blames_a_failed_program(self):
        msg = runner_mod._total_zero_event_warning(["mpi"], ["./app"], 2)
        self.assertIn("exited with status 2", msg)
        self.assertIn("failed before making any profiled call", msg)
        msg = runner_mod._total_zero_event_warning(["mpi"], ["./app"], 0)
        self.assertIn("never connecting", msg)

    def test_cli_exit_status_flag(self):
        tmp = tempfile.mkdtemp()
        try:
            base = [sys.executable, str(ROOT / "hprofiler"), "run", "--backend", "openmp", "--no-ui",
                    "--no-summary", "--no-json", "-o", str(Path(tmp) / "a.hpstore")]
            prog = ["--", sys.executable, "-c", "import sys; sys.exit(5)"]
            r = subprocess.run(base + prog, capture_output=True, text=True, timeout=120)
            self.assertEqual(r.returncode, 0, r.stderr)          # default: unchanged behaviour
            self.assertIn("exited with status 5", r.stderr)
            base[-1] = str(Path(tmp) / "b.hpstore")
            r = subprocess.run(base + ["--exit-status"] + prog,
                               capture_output=True, text=True, timeout=120)
            self.assertEqual(r.returncode, 5, r.stderr)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestStaticCudaBehindWrappers(unittest.TestCase):
    @unittest.skipUnless(shutil.which("gcc") and shutil.which("nm") and shutil.which("ldd"), "needs gcc/nm/ldd")
    def test_program_named_in_a_wrappers_arguments_is_checked(self):
        # a binary that defines cudaLaunchKernel itself (as one linked with
        # libcudart_static.a does), started through a wrapper script
        tmp = tempfile.mkdtemp()
        try:
            src = Path(tmp) / "fake.c"
            src.write_text("int cudaLaunchKernel(void) { return 0; }\nint main(void) { return cudaLaunchKernel(); }\n")
            exe = Path(tmp) / "fake_static_cuda"
            subprocess.run(["gcc", "-O0", str(src), "-o", str(exe)], check=True)
            wrapper = Path(tmp) / "wrap.sh"
            wrapper.write_text('#!/bin/sh\nexec "$@"\n')
            wrapper.chmod(0o755)
            self.assertEqual(runner_mod._static_cuda_binary([str(exe)]), str(exe))
            self.assertEqual(runner_mod._static_cuda_binary([str(wrapper), "/tmp/x.exit", str(exe), "-n", "3"]),
                             str(exe))
            self.assertEqual(runner_mod._static_cuda_binary([str(wrapper), sys.executable, "-c", "pass"]), "")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestCollectorSocketPath(unittest.TestCase):
    def test_long_tmpdir_falls_back_to_a_short_directory(self):
        long_dir = tempfile.mkdtemp(prefix="x" * 60)
        deep = Path(long_dir) / ("y" * 60)
        deep.mkdir()
        try:
            with mock.patch.dict(os.environ, {"TMPDIR": str(deep)}), \
                    mock.patch.object(tempfile, "tempdir", None):
                sock, address, spool, path = runner_mod._open_collector_socket()
            try:
                self.assertIsNotNone(path)
                self.assertLessEqual(len(os.fsencode(path)), runner_mod._SUN_PATH_MAX)
                self.assertTrue(path.startswith("/tmp/"))
                self.assertTrue(spool.startswith(str(deep)))
            finally:
                sock.close()
                shutil.rmtree(os.path.dirname(path), ignore_errors=True)
                shutil.rmtree(spool, ignore_errors=True)
        finally:
            shutil.rmtree(long_dir, ignore_errors=True)

    def test_abstract_socket_when_no_path_fits(self):
        with mock.patch.object(runner_mod, "_SUN_PATH_MAX", 1):
            sock, address, spool, path = runner_mod._open_collector_socket()
        try:
            self.assertIsNone(path)
            self.assertTrue(address.startswith("@"))
            import socket
            c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            c.connect("\0" + address[1:])
            c.close()
        finally:
            sock.close()
            shutil.rmtree(spool, ignore_errors=True)

    @unittest.skipUnless(shutil.which("gcc") and (ROOT / "build/lib/libhprofiler_gomp.so").exists(),
                         "needs gcc and the built GNU libgomp hook")
    def test_hooks_connect_to_an_abstract_socket(self):
        tmp = tempfile.mkdtemp()
        try:
            exe = Path(tmp) / "gomp_mini"
            subprocess.run(["gcc", "-fopenmp", "-O1", str(ROOT / "tests/fixtures/gomp_mini.c"), "-o", str(exe)],
                           check=True, capture_output=True)
            with mock.patch.object(runner_mod, "_SUN_PATH_MAX", 1):
                trace = Runner([str(exe)], backends=["openmp"]).run()
            self.assertGreater(trace.span_count(), 0)
            self.assertEqual(trace.metadata.capture_health["receiver"].get("rejected_peers", 0), 0)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
