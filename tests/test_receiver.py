"""
Collector-side protocol handling (src/core/receiver.py) over real AF_UNIX
connections: nothing a hook sends is dropped without being counted, one bad
record never costs the rest of a connection, lines are decoded per complete
line (a UTF-8 character split across two sends survives), transport status
records are kept per process image, and the warnings derived from all of it
(plus degraded GPU tracing) are what `hprofiler run`/`summary`/the GUI show.
"""
from __future__ import annotations

import os
import socket
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.receiver import Receiver, capture_warnings, parse_xport, run_warnings
from src.core.trace import TraceMetadata
from src.core import gpu_activity as ga


class _Harness:
    def __init__(self, handle=None, accept_timeout=None):
        self.dir = tempfile.mkdtemp(prefix="hp_rx_")
        self.path = os.path.join(self.dir, "s")
        self.srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.srv.bind(self.path)
        self.srv.listen(16)
        if accept_timeout is not None:
            self.srv.settimeout(accept_timeout)       # as Runner does
        self.lines: list[str] = []
        self.rx = Receiver(self.srv, self.dir, None, handle or self.lines.append)
        self.rx.start()

    def connect(self) -> socket.socket:
        c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        c.connect(self.path)
        return c

    def finish(self, timeout_s: float = 10.0) -> dict:
        self.rx.finish(timeout_s=timeout_s)
        self.srv.close()
        return self.rx.health()


def _xport(pid=100, hook="gomp", image=1, **kw) -> bytes:
    fields = {"mode": "ring", "final": 1, "emitted": 0, "sent": 0, "dropped_full": 0, "dropped_lost": 0,
              "oversize": 0, "format_errors": 0, "blocked_ns": 0, "image": image}
    fields.update(kw)
    return ("xport:1:%d:%s:%s\n" % (pid, hook, ",".join(f"{k}={v}" for k, v in fields.items()))).encode()


class TestReceiver(unittest.TestCase):
    def test_every_complete_line_delivered_in_order(self):
        h = _Harness()
        c = h.connect()
        for i in range(2000):
            c.sendall(f"span:openmp:1:1:{i}:1:x{i}\n".encode())
        c.sendall(_xport(emitted=2000, sent=2000))
        c.close()
        health = h.finish()
        self.assertEqual(h.lines, [f"span:openmp:1:1:{i}:1:x{i}" for i in range(2000)])
        t = health["transport"]["100/gomp"]
        self.assertEqual((t["received"], t["final"], t["ended_without_final"]), (2000, 1, False))
        self.assertEqual(capture_warnings({"state": "complete", **health}), [])

    def test_utf8_character_split_across_sends(self):
        h = _Harness()
        c = h.connect()
        rec = "span:cuda:1:1:0:1:kernel_é中:type=kernel\n".encode()
        cut = rec.index(b"\xe4") + 1                 # in the middle of a 3-byte character
        c.sendall(rec[:cut])
        time.sleep(0.05)
        c.sendall(rec[cut:])
        c.close()
        health = h.finish()
        self.assertEqual(h.lines, ["span:cuda:1:1:0:1:kernel_é中:type=kernel"])
        self.assertEqual(health["receiver"]["decode_replaced"], 0)

    def test_invalid_utf8_counted_not_dropped(self):
        h = _Harness()
        c = h.connect()
        c.sendall(b"span:cpu:1:1:0:1:bad\xff\xfe\n")
        c.close()
        health = h.finish()
        self.assertEqual(len(h.lines), 1)
        self.assertEqual(health["receiver"]["decode_replaced"], 1)

    def test_processing_error_does_not_end_the_connection(self):
        got = []

        def handle(line: str):
            if "boom" in line:
                raise ValueError("bad record")
            got.append(line)
        h = _Harness(handle)
        c = h.connect()
        c.sendall(b"span:cpu:1:1:0:1:a\nspan:cpu:1:1:0:1:boom\nspan:cpu:1:1:0:1:b\n")
        c.close()
        health = h.finish()
        self.assertEqual(got, ["span:cpu:1:1:0:1:a", "span:cpu:1:1:0:1:b"])
        self.assertEqual(health["receiver"]["errors"], 1)
        self.assertIn("ValueError", health["receiver"]["error_samples"][0])
        self.assertTrue(any("failed to process" in w for w in capture_warnings(health)))

    def test_partial_last_line_counted(self):
        h = _Harness()
        c = h.connect()
        c.sendall(b"span:cpu:1:1:0:1:whole\nspan:cpu:1:1:0:1:cut-off-wi")
        c.close()
        health = h.finish()
        self.assertEqual(h.lines, ["span:cpu:1:1:0:1:whole"])
        self.assertEqual(health["receiver"]["partial_tails"], 1)
        self.assertEqual(health["receiver"]["malformed"], 1)
        self.assertTrue(any("in the middle of a record" in w for w in capture_warnings(health)))

    def test_malformed_and_unknown_counted_by_caller_hooks(self):
        def handle(line: str):
            if line.startswith("zzz:"):
                h.rx.note_unknown("zzz")
            else:
                h.rx.note_malformed(line)
        h = _Harness(handle)
        c = h.connect()
        c.sendall(b"zzz:1\nzzz:2\nspan:not-a-number\n")
        c.close()
        health = h.finish()
        self.assertEqual(health["receiver"]["unknown_kinds"], {"zzz": 2})
        self.assertEqual(health["receiver"]["malformed"], 1)
        self.assertEqual(health["receiver"]["malformed_samples"], ["span:not-a-number"])
        ws = capture_warnings(health)
        self.assertTrue(any("zzz x2" in w for w in ws), ws)
        self.assertTrue(any("malformed" in w for w in ws), ws)

    def test_connection_without_final_status_is_flagged(self):
        h = _Harness()
        c = h.connect()
        c.sendall(b"span:cpu:1:1:0:1:a\n" + _xport(final=0, emitted=5, sent=1))
        c.close()                                        # crashed before the final drain
        health = h.finish()
        t = health["transport"]["100/gomp"]
        self.assertTrue(t["ended_without_final"])
        self.assertTrue(any("ended without its final drain" in w for w in capture_warnings(health)))

    def test_hook_side_losses_reported(self):
        h = _Harness()
        c = h.connect()
        c.sendall(_xport(emitted=10, sent=7, dropped_full=2, oversize=1, blocked_ns=250_000_000,
                         max_block_ns=200_000_000))
        c.close()
        ws = capture_warnings(h.finish())
        self.assertEqual(len(ws), 1, ws)
        for s in ("2 dropped", "1 rejected as oversize", "waited 0.25 s"):
            self.assertIn(s, ws[0])

    def test_exec_images_of_one_pid_kept_apart(self):
        h = _Harness()
        a = h.connect()
        a.sendall(b"span:cpu:100:1:0:1:a\n" * 3 + _xport(image=111, emitted=3, sent=3, dropped_full=4))
        a.close()
        b = h.connect()
        b.sendall(b"span:cpu:100:1:0:1:b\n" + _xport(image=222, emitted=1, sent=1))
        b.close()
        health = h.finish()
        tr = health["transport"]
        self.assertEqual(sorted(tr), ["100/gomp#1", "100/gomp#2"])
        self.assertEqual((tr["100/gomp#1"]["received"], tr["100/gomp#1"]["dropped_full"]), (3, 4))
        self.assertEqual((tr["100/gomp#2"]["received"], tr["100/gomp#2"]["dropped_full"]), (1, 0))
        # the pre-exec image's drops are not hidden by the later image's status
        self.assertTrue(any("4 dropped" in w for w in capture_warnings(health)))

    def test_old_status_without_image_still_parsed(self):
        pid, hook, f = parse_xport("xport:1:7:mpi:mode=sync,final=1,emitted=3,sent=3")
        self.assertEqual((pid, hook, f["final"], f["emitted"], f.get("image")), (7, "mpi", 1, 3, None))
        self.assertIsNone(parse_xport("xport:1:notapid:mpi:final=1"))

    def test_connection_left_in_listen_backlog_is_still_read(self):
        # Runner's server socket polls accept() with a timeout; a child that
        # connects after the accept loop stopped (just before the program
        # exited) must still be read from the backlog, not lost.
        h = _Harness(accept_timeout=0.2)
        h.rx._stop_accept.set()
        h.rx._accept_thread.join(timeout=5)
        self.assertFalse(h.rx._accept_thread.is_alive())
        c = h.connect()
        c.sendall(b"span:cpu:5:5:0:1:late\n" + _xport(pid=5, emitted=1, sent=1))
        c.close()
        health = h.finish()
        self.assertEqual(h.lines, ["span:cpu:5:5:0:1:late"])
        self.assertEqual(health["transport"]["5/gomp"]["received"], 1)

    def test_finish_returns_at_deadline_for_lingering_connection(self):
        h = _Harness()
        c = h.connect()                       # a child that never closes its socket
        c.sendall(b"span:cpu:1:1:0:1:a\n")
        t0 = time.monotonic()
        health = h.finish(timeout_s=0.5)
        self.assertLess(time.monotonic() - t0, 5)
        self.assertEqual(h.lines, ["span:cpu:1:1:0:1:a"])
        self.assertEqual(health["receiver"]["open_at_deadline"], 1)
        c.close()


class TestRunWarnings(unittest.TestCase):
    def _meta(self, device_activity=None, capture_health=None) -> TraceMetadata:
        m = TraceMetadata()
        m.device_activity = device_activity or {}
        m.capture_health = capture_health or {}
        return m

    def test_interrupted_capture(self):
        ws = run_warnings(self._meta(capture_health={"state": "interrupted"}))
        self.assertTrue(ws and "did not complete" in ws[0])

    def test_gpu_missing_final_flush_only_when_promised(self):
        active = {"tracer": "cupti", "status": "active", "clock": "monotonic_callback"}
        promised = {"9/cuda": {**active, "final_marker": "1"}}
        flushed = {"9/cuda": {**active, "final_marker": "1", "final_flush": "1"}}
        old_hook = {"9/cuda": dict(active)}
        self.assertTrue(any("final flush" in w for w in run_warnings(self._meta(promised))))
        self.assertEqual(run_warnings(self._meta(flushed)), [])
        self.assertEqual(run_warnings(self._meta(old_hook)), [], "old traces must not be flagged")
        self.assertTrue(any("no final flush" in d for d in ga.describe(promised)))

    def test_gpu_degraded_modes(self):
        da = {"9/cuda": {"tracer": "cupti", "status": "unavailable", "reason": "libcupti_not_found"},
              "9/rocm": {"tracer": "rocprofiler", "status": "active", "dropped": 12, "clock": "unmapped",
                         "final_marker": "1", "final_flush": "1"},
              "10/cuda": {"tracer": "cupti", "status": "disabled", "reason": "env_off"}}
        ws = run_warnings(self._meta(da))
        self.assertTrue(any("CUDA pid 9" in w and "proxies" in w for w in ws), ws)
        self.assertTrue(any("ROCM pid 9" in w and "dropped 12" in w for w in ws), ws)
        self.assertTrue(any("could not be mapped" in w for w in ws), ws)
        self.assertFalse(any("pid 10" in w for w in ws), "an explicit opt-out is not a warning")

    def test_record_status_merges_final_flush(self):
        from src.core.trace import Trace
        t = Trace()
        ga.record_status(t, 5, "cupti", {"status": "active", "final_marker": "1"})
        self.assertTrue(ga.missing_final_flush(t.metadata.device_activity["5/cuda"]))
        ga.record_status(t, 5, "cupti", {"final_flush": "1"})
        self.assertFalse(ga.missing_final_flush(t.metadata.device_activity["5/cuda"]))


if __name__ == "__main__":
    unittest.main()
