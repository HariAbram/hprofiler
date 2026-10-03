"""
Native tests for the shared hook transport (hooks/common/hp_transport.h) and
the OpenCL hook's JIT trampolines, built from tests/native/*.c and run one
scenario per process (the transport reads its configuration once):

  plain     every scenario, with exact accounting asserted
            (received + dropped == emitted, per-thread order, final status)
  ASan+UBSan  the same scenarios -- memory errors / undefined behaviour
  TSan      the concurrent scenarios -- data races in the lock-free rings,
            the drain thread and shutdown. Needs `setarch -R` on kernels
            whose mmap layout TSan rejects; skipped (not failed) when TSan
            cannot start here at all.

See tests/native/transport_test.c for what each scenario does.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NATIVE = REPO / "tests" / "native"
SCENARIOS = ("order", "long", "newline", "overflow", "wait", "threads", "fork", "nocollector", "exec")


def _gcc_ok(*flags: str) -> bool:
    if not shutil.which("gcc"):
        return False
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "t.c")
        Path(src).write_text("#include <pthread.h>\nint main(void){return 0;}\n")
        r = subprocess.run(["gcc", *flags, "-pthread", src, "-o", os.path.join(d, "t")],
                           capture_output=True)
        if r.returncode:
            return False
        return subprocess.run(_runner(flags) + [os.path.join(d, "t")], capture_output=True).returncode == 0


def _runner(flags) -> list[str]:
    if "-fsanitize=thread" in flags and shutil.which("setarch"):
        return ["setarch", os.uname().machine, "-R"]
    return []


def _parse(stdout: str) -> dict:
    for line in stdout.splitlines():
        if line.startswith("RESULT "):
            out = {}
            for kv in line.split()[1:]:
                k, _, v = kv.partition("=")
                out[k] = int(v) if v.lstrip("-").isdigit() else v
            return out
    raise AssertionError(f"no RESULT line in output: {stdout!r}")


class _TransportScenarios:
    """Mixin: one build of transport_test.c, asserted per scenario."""
    FLAGS: tuple[str, ...] = ()
    ENV: dict[str, str] = {}
    RUN: tuple[str, ...] = SCENARIOS
    TIMEOUT = 120

    @classmethod
    def setUpClass(cls):
        if not _gcc_ok(*cls.FLAGS):
            raise unittest.SkipTest(f"gcc {' '.join(cls.FLAGS) or '(plain)'} not usable here")
        cls.tmp = tempfile.mkdtemp(prefix="hp_transport_")
        cls.exe = os.path.join(cls.tmp, "transport_test")
        b = subprocess.run(["gcc", "-O1", "-g", "-pthread", "-Wall", "-Wextra", "-Werror", *cls.FLAGS,
                            str(NATIVE / "transport_test.c"), "-o", cls.exe, "-ldl"],
                           capture_output=True, text=True)
        if b.returncode:
            raise AssertionError(f"build failed:\n{b.stderr}")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def run_scenario(self, name: str) -> dict:
        if name not in self.RUN:
            self.skipTest(f"{name} not run under {self.FLAGS}")
        env = {**os.environ, **self.ENV}
        env.pop("HPROFILER_SOCKET", None)
        p = subprocess.run(_runner(self.FLAGS) + [self.exe, name], capture_output=True, text=True,
                           timeout=self.TIMEOUT, env=env)
        for marker in ("ThreadSanitizer", "AddressSanitizer", "runtime error:", "LeakSanitizer"):
            self.assertNotIn(marker, p.stderr, f"{name}: sanitizer report:\n{p.stderr[-4000:]}")
        self.assertEqual(p.returncode, 0, f"{name}: exit {p.returncode}\n{p.stderr[-2000:]}")
        return _parse(p.stdout)

    def assert_clean(self, r: dict) -> None:
        self.assertEqual(r["bad_order"], 0, r)
        self.assertEqual(r["dropped_lost"], 0, r)
        self.assertEqual(r["final"] >= 1, True, r)

    def test_order(self):
        r = self.run_scenario("order")
        self.assert_clean(r)
        self.assertEqual((r["lines"], r["gaps"], r["emitted"], r["sent"], r["dropped_full"]),
                         (160000, 0, 160000, 160000, 0), r)

    def test_long_records_intact_oversize_counted(self):
        r = self.run_scenario("long")
        self.assertEqual((r["long_ok"], r["long_bad"], r["oversize"], r["emitted"], r["sent"]),
                         (2, 0, 1, 3, 2), r)

    def test_embedded_newlines_sanitized(self):
        r = self.run_scenario("newline")
        self.assertEqual((r["lines"], r["newline_ok"], r["sanitized"]), (1, 1, 1), r)

    def test_overflow_drops_counted_exactly(self):
        r = self.run_scenario("overflow")
        self.assert_clean(r)
        self.assertGreater(r["dropped_full"], 0, r)
        self.assertEqual(r["lines"] + r["dropped_full"], r["emitted"], r)
        self.assertEqual(r["emitted"], 80001, r)
        self.assertEqual(r["waits"], 0, "HPROFILER_RING_WAIT_MS=0 must never wait")

    def test_bounded_wait_loses_nothing(self):
        r = self.run_scenario("wait")
        self.assert_clean(r)
        self.assertEqual((r["lines"], r["dropped_full"], r["gaps"]), (80001, 0, 0), r)
        self.assertGreater(r["waits"], 0, r)

    def test_short_lived_threads_drained(self):
        r = self.run_scenario("threads")
        self.assert_clean(r)
        self.assertEqual((r["lines"], r["gaps"], r["emitted"]), (30000, 0, 30000), r)

    def test_fork_child_gets_its_own_transport(self):
        r = self.run_scenario("fork")
        self.assert_clean(r)
        self.assertEqual((r["lines"], r["pids"], r["final"], r["images"], r["gaps"]), (3000, 2, 2, 2, 0), r)

    def test_no_collector_never_hangs(self):
        if "nocollector" not in self.RUN:
            self.skipTest("not run here")
        env = {**os.environ, **self.ENV}
        p = subprocess.run(_runner(self.FLAGS) + [self.exe, "nocollector"], capture_output=True,
                           text=True, timeout=60, env=env)
        self.assertEqual(p.returncode, 0, p.stderr[-2000:])
        self.assertIn("returned=1", p.stdout)

    def test_exec_drains_before_image_replaced(self):
        r = self.run_scenario("exec")
        self.assert_clean(r)
        # 500 records from the pre-exec image + 1 from the new image, both
        # with a final status; same pid, told apart by image=
        self.assertEqual((r["lines"], r["pids"], r["images"], r["final"]), (501, 1, 2, 2), r)


class TestTransportPlain(_TransportScenarios, unittest.TestCase):
    pass


class TestTransportASanUBSan(_TransportScenarios, unittest.TestCase):
    FLAGS = ("-fsanitize=address,undefined", "-fno-omit-frame-pointer", "-fno-sanitize-recover=all")
    ENV = {"ASAN_OPTIONS": "detect_leaks=0"}     # the drain thread is detached at exit by design
    TIMEOUT = 300


class TestTransportTSan(_TransportScenarios, unittest.TestCase):
    FLAGS = ("-fsanitize=thread",)
    # fork/exec in a TSan process: the child must not be killed for being
    # forked while threads exist
    ENV = {"TSAN_OPTIONS": "die_after_fork=0 halt_on_error=1"}
    TIMEOUT = 600


class TestOpenclTrampolines(unittest.TestCase):
    def test_every_trampoline_dispatches_its_own_slot(self):
        if not shutil.which("gcc"):
            self.skipTest("gcc not available")
        with tempfile.TemporaryDirectory() as d:
            exe = os.path.join(d, "tramp")
            b = subprocess.run(["gcc", "-O2", "-w", "-pthread", str(NATIVE / "opencl_trampoline_test.c"),
                                "-o", exe, "-ldl"], capture_output=True, text=True)
            self.assertEqual(b.returncode, 0, b.stderr[-2000:])
            env = {k: v for k, v in os.environ.items() if k != "HPROFILER_SOCKET"}
            p = subprocess.run([exe], capture_output=True, text=True, timeout=60, env=env)
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            self.assertIn("RESULT trampolines=256 wrong=0", p.stdout)


if __name__ == "__main__":
    unittest.main()
