"""
Damaged, incomplete and interrupted traces are reported, never silently
treated as complete and never surfaced as a raw traceback:

  * .hpstore with a missing / foreign / empty shard or a corrupt catalog
    -> StoreError naming the problem (an empty shard just gets its tables);
  * a store whose capture never completed (capture_health state "running")
    reopens as "interrupted", with a warning;
  * streamed Chrome JSON cut short (interrupted export or copy) loads every
    complete event before the cut and records the truncation; an
    unreadable line in the middle is skipped and counted;
  * the CLI turns any of these into one line and exit status 2;
  * the GUI loader classifies store damage as invalid input.
"""
from __future__ import annotations

import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.core import trace_io
from src.core.events import Category, SpanEvent
from src.core.receiver import capture_warnings
from src.core.store import StoreError
from src.core.trace import TraceMetadata
from src.output import chrome_trace

T0 = 1_000_000_000


def _make_store(path: Path, n: int = 200, pids=(11, 12), state: str = "complete"):
    meta = TraceMetadata()
    meta.command = "./app"
    meta.capture_health = {"state": state}
    trace = trace_io.create_disk_trace(path, meta)
    for i in range(n):
        trace.add(SpanEvent(f"k{i % 7}", Category.OPENMP, T0 + i * 1000, 500, pids[i % len(pids)], 1,
                            {"type": "parallel"}))
    if state == "complete":
        trace.finalize()
    trace.save()
    trace.close()


class TestDamagedStores(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="hp_storeerr_"))
        self.store = self.tmp / "t.hpstore"
        _make_store(self.store)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _open(self):
        with redirect_stderr(io.StringIO()):
            return trace_io.open_trace(self.store)

    def test_intact_store_opens(self):
        t = self._open()
        self.assertEqual(t.span_count(), 200)
        t.close()

    def test_missing_shard(self):
        shard = sorted((self.store / "shards").iterdir())[0]
        shard.unlink()
        with self.assertRaises(StoreError) as cm:
            self._open()
        self.assertIn("missing", str(cm.exception))
        self.assertIn(shard.name, str(cm.exception))
        self.assertFalse(shard.exists(), "opening must not recreate the missing shard")

    def test_foreign_shard(self):
        shard = sorted((self.store / "shards").iterdir())[0]
        shard.write_bytes(b"this is not a database at all" * 10)
        with self.assertRaises(StoreError) as cm:
            self._open()
        self.assertIn("not an SQLite file", str(cm.exception))

    def test_corrupt_catalog(self):
        (self.store / "catalog.sqlite").write_bytes(b"garbage" * 100)
        for side in ("catalog.sqlite-wal", "catalog.sqlite-shm"):
            (self.store / side).unlink(missing_ok=True)
        with self.assertRaises(StoreError) as cm:
            self._open()
        self.assertIn("catalog", str(cm.exception))

    def test_empty_shard_gets_tables(self):
        # capture killed right after creating a shard
        store2 = self.tmp / "e.hpstore"
        _make_store(store2, n=10, pids=(21,), state="running")
        shard = sorted((store2 / "shards").iterdir())[0]
        for side in list((store2 / "shards").glob(shard.name + "-*")):
            side.unlink()
        shard.write_bytes(b"")
        with redirect_stderr(io.StringIO()):
            t = trace_io.open_trace(store2)
        self.assertEqual(t.span_count(), 0)
        self.assertEqual(t.metadata.capture_health.get("state"), "interrupted")
        t.close()


class TestInterruptedCapture(unittest.TestCase):
    def test_running_store_reopens_as_interrupted(self):
        tmp = Path(tempfile.mkdtemp(prefix="hp_interrupt_"))
        try:
            store = tmp / "i.hpstore"
            _make_store(store, state="running")
            err = io.StringIO()
            with redirect_stderr(err):
                t = trace_io.open_trace(store)
            self.assertEqual(t.metadata.capture_health["state"], "interrupted")
            self.assertIn("did not complete", err.getvalue())
            self.assertTrue(any("did not complete" in w for w in capture_warnings(t.metadata.capture_health)))
            self.assertEqual(t.span_count(), 200, "events stored before the interruption are kept")
            t.close()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestTruncatedJson(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="hp_trunc_"))
        trace = trace_io.create_disk_trace(cls.tmp / "src.hpstore", TraceMetadata(command="./app"))
        for i in range(500):
            trace.add(SpanEvent(f"k{i % 5}", Category.CPU, T0 + i * 1000, 400, 1, 1 + i % 3, {}))
        trace.finalize()
        from src.disasm.extractor import KernelDisasm
        trace.add_disasm(KernelDisasm(name="k0", arch="x86-64", source="/bin/true"))
        cls.full = cls.tmp / "full.json"
        chrome_trace.write(trace, cls.full)
        trace.close()
        cls.data = cls.full.read_bytes()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _load(self, data: bytes, name: str):
        p = self.tmp / name
        p.write_bytes(data)
        err = io.StringIO()
        with redirect_stderr(err):
            t = chrome_trace.load_trace_from_json(p)
        return t, err.getvalue()

    def test_complete_file_has_no_issues(self):
        t, err = self._load(self.data, "ok.json")
        self.assertEqual(t.span_count(), 500)
        self.assertNotIn("load", t.metadata.capture_health)
        self.assertEqual(err, "")

    def test_cut_inside_events(self):
        events_at = self.data.index(b'"traceEvents": [')
        cut = events_at + (len(self.data) - events_at) // 2
        t, err = self._load(self.data[:cut], "half.json")
        load = t.metadata.capture_health["load"]
        self.assertTrue(load["truncated"])
        self.assertGreaterEqual(load["events"], t.span_count())   # + process/thread name records
        self.assertGreater(t.span_count(), 100)
        self.assertLess(t.span_count(), 500)
        self.assertNotIn("bad_lines", load, "the cut-off last line is the truncation, not a bad line")
        self.assertEqual(t.metadata.command, "./app", "metadata precedes the events and survives")
        self.assertIn("truncated", err)
        self.assertTrue(any("truncated" in w for w in capture_warnings(t.metadata.capture_health)))

    def test_cut_inside_disassembly(self):
        cut = self.data.index(b'"disasm": ') + 20
        t, _ = self._load(self.data[:cut], "nodis.json")
        load = t.metadata.capture_health["load"]
        self.assertEqual((load["truncated"], load["disasm_missing"], t.span_count()), (True, True, 500))

    def test_cut_before_events(self):
        cut = self.data.index(b'"traceEvents": [') - 5
        t, _ = self._load(self.data[:cut], "nometa.json")
        self.assertTrue(t.metadata.capture_health["load"]["truncated"])
        self.assertEqual(t.span_count(), 0)

    def test_bad_line_in_the_middle_skipped_and_counted(self):
        lines = self.data.split(b"\n")
        idx = next(i for i, l in enumerate(lines) if l.startswith(b'{"ph": "X"')) + 10
        lines[idx] = b'{"name": "broken", "ph": "X", ,,,},'
        t, err = self._load(b"\n".join(lines), "bad.json")
        load = t.metadata.capture_health["load"]
        self.assertEqual(load, {"bad_lines": 1})
        self.assertEqual(t.span_count(), 499)
        self.assertIn("unreadable", err)

    def test_legacy_single_object_json_still_raises_json_error(self):
        p = self.tmp / "legacy.json"
        p.write_text('{"traceEvents": [{"ph": "X", "name": "a"')
        with self.assertRaises(json.JSONDecodeError):
            chrome_trace.load_trace_from_json(p)


class TestCliAndGui(unittest.TestCase):
    def test_cli_reports_damage_in_one_line(self):
        tmp = Path(tempfile.mkdtemp(prefix="hp_cli_err_"))
        try:
            store = tmp / "t.hpstore"
            _make_store(store)
            sorted((store / "shards").iterdir())[0].unlink()
            p = subprocess.run([sys.executable, str(REPO / "hprofiler"), "summary", str(store)],
                               capture_output=True, text=True, timeout=120)
            self.assertEqual(p.returncode, 2, p.stderr)
            self.assertNotIn("Traceback", p.stderr)
            self.assertIn("[hprofiler] error:", p.stderr)
            self.assertIn("missing", p.stderr)
            legacy = tmp / "bad.json"
            legacy.write_text('{"traceEvents": [')
            p = subprocess.run([sys.executable, str(REPO / "hprofiler"), "summary", str(legacy)],
                               capture_output=True, text=True, timeout=120)
            self.assertEqual(p.returncode, 2, p.stderr)
            self.assertIn("not valid JSON", p.stderr)
            self.assertNotIn("Traceback", p.stderr)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_gui_classifies_store_damage(self):
        try:
            from src.gui.errors import ErrorKind, classify_load_exception
        except ImportError as exc:      # PySide6 missing
            self.skipTest(str(exc))
        err = classify_load_exception(StoreError("/x/t.hpstore: the store is incomplete -- 1 shard file(s) missing"),
                                      file="/x/t.hpstore", stage="opening")
        self.assertEqual(err.kind, ErrorKind.INVALID_INPUT)
        self.assertIn("incomplete", err.message)


if __name__ == "__main__":
    unittest.main()
