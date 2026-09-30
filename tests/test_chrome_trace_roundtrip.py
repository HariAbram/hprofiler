"""
Regression test for src/output/chrome_trace.py: write()/load_trace_from_json()
silently dropped most of KernelDisasm/DisasmLine's fields on a JSON
round-trip -- mangled_name, ptxas_derived, source_file, source_line,
sample_pct, stall_cycles, stall_reason were all computed correctly at
profile time but never serialized, so re-opening a SAVED trace
(`hprofiler gui saved.hprofiler.json`, or the TUI's `hprofiler view`)
always showed them at their dataclass defaults (empty/0/-1) regardless
of what was actually collected. Found while investigating a real user
report of the GUI's Source tab showing no per-instruction statistics and
no indication of what function was being disassembled -- this is one of
two root causes (the other: annotate_with_perf's symbol filter, see
test_disasm_categories.py).

Only matters for a trace that's SAVED and RE-OPENED later, not one
viewed immediately after profiling in the same process (where the UI
reads the live Trace object directly, never through this JSON path) --
but re-opening a saved trace to look at collected disassembly is a
completely ordinary, expected workflow.
"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.trace import Trace, TraceMetadata
from src.core.events import SpanEvent, Category
from src.disasm.extractor import KernelDisasm, DisasmLine
from src.disasm.classifier import InsnType
from src.output import chrome_trace


class TestDisasmJsonRoundtrip(unittest.TestCase):
    def _roundtrip(self, kd: KernelDisasm) -> KernelDisasm:
        trace = Trace(TraceMetadata(command="a.out", args=[]))
        trace.add(SpanEvent(name=kd.name, category=Category.CPU,
                            start_ns=0, duration_ns=100, pid=1, tid=1))
        trace.add_disasm(kd)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            chrome_trace.write(trace, path)
            loaded = chrome_trace.load_trace_from_json(path)
        finally:
            Path(path).unlink(missing_ok=True)
        return loaded.disasm[kd.name]

    def test_mangled_name_survives_roundtrip(self):
        kd = KernelDisasm(name="omp_barrier", arch="x86-64", source="/bin/a.out",
                          mangled_name="_ZN3gmx19ThreadedForceBufferIA4_fEC2Eibi",
                          lines=[DisasmLine(addr=1, mnemonic="nop", operands="")])
        self.assertEqual(self._roundtrip(kd).mangled_name,
                         "_ZN3gmx19ThreadedForceBufferIA4_fEC2Eibi")

    def test_ptxas_derived_survives_roundtrip(self):
        kd = KernelDisasm(name="k", arch="sass", source="/tmp/x.cubin", ptxas_derived=True,
                          lines=[DisasmLine(addr=1, mnemonic="nop", operands="")])
        self.assertTrue(self._roundtrip(kd).ptxas_derived)

    def test_disasm_line_annotations_survive_roundtrip(self):
        kd = KernelDisasm(
            name="hot_fn", arch="x86-64", source="/bin/a.out",
            lines=[
                DisasmLine(addr=0x10, mnemonic="mov", operands="rax, rbx",
                          itype=InsnType.SCALAR, source_file="foo.cpp", source_line=42,
                          sample_pct=12.5, stall_cycles=3, stall_reason="LG Throttle"),
            ],
        )
        ln = self._roundtrip(kd).lines[0]
        self.assertEqual(ln.source_file, "foo.cpp")
        self.assertEqual(ln.source_line, 42)
        self.assertEqual(ln.sample_pct, 12.5)
        self.assertEqual(ln.stall_cycles, 3)
        self.assertEqual(ln.stall_reason, "LG Throttle")

    def test_defaults_are_sane_when_fields_absent(self):
        # A trace file written by an OLDER hprofiler build (before this
        # fix) won't have these keys in its JSON at all -- must not crash,
        # and stall_cycles=-1 (not 0) means "unknown", matching
        # DisasmLine's own dataclass default.
        kd = KernelDisasm(name="k", arch="x86-64", source="/bin/a.out",
                          lines=[DisasmLine(addr=1, mnemonic="nop", operands="")])
        loaded = self._roundtrip(kd)
        self.assertEqual(loaded.mangled_name, "")
        self.assertFalse(loaded.ptxas_derived)
        ln = loaded.lines[0]
        self.assertEqual(ln.source_file, "")
        self.assertEqual(ln.sample_pct, 0.0)
        self.assertEqual(ln.stall_cycles, -1)


class TestCaptureTimeRoundtrip(unittest.TestCase):
    """TraceMetadata.capture_time_iso -- new, additive field for the GUI
    Overview tab's "Captured" summary field (see src/gui/bridge.py's
    DashboardBridge.captureTime). Populated in runner.py at profile time,
    serialized as chrome_trace.py's "captureTime" metadata key. Checks
    the round-trip explicitly since every OTHER metadata field it sits
    next to (command/args/hostname/backends_used) already round-trips
    implicitly through the other tests in this file -- this is the one
    new key that could silently regress without its own coverage."""

    def _roundtrip(self, meta: TraceMetadata) -> TraceMetadata:
        trace = Trace(meta)
        trace.add(SpanEvent(name="fn", category=Category.CPU, start_ns=0, duration_ns=100, pid=1, tid=1))
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            chrome_trace.write(trace, path)
            loaded = chrome_trace.load_trace_from_json(path)
        finally:
            Path(path).unlink(missing_ok=True)
        return loaded.metadata

    def test_capture_time_survives_roundtrip(self):
        meta = TraceMetadata(command="a.out", args=[], capture_time_iso="2026-09-25T10:00:00")
        self.assertEqual(self._roundtrip(meta).capture_time_iso, "2026-09-25T10:00:00")

    def test_defaults_to_empty_when_absent_from_json(self):
        # A trace saved by a build before this field existed has no
        # "captureTime" key at all -- must default to "", which the GUI
        # renders as "unavailable", not a crash or a fabricated time.
        meta = TraceMetadata(command="a.out", args=[])  # capture_time_iso="" by default
        self.assertEqual(self._roundtrip(meta).capture_time_iso, "")


class TestLoadTraceProgressAndCancellation(unittest.TestCase):
    """progress_cb/cancel_check (GUI async-loading worker support, see
    src/gui/loader.py) -- optional and additive, every pre-existing
    caller (TUI/CLI/analysis code, every OTHER test in this file) passes
    neither and is completely unaffected; verified separately below."""

    def _write_trace_with_n_events(self, n: int) -> str:
        trace = Trace(TraceMetadata(command="a.out", args=[]))
        for i in range(n):
            trace.add(SpanEvent(name=f"fn{i}", category=Category.CPU,
                                start_ns=i * 1000, duration_ns=100, pid=1, tid=1))
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        chrome_trace.write(trace, path)
        return path

    def test_no_callbacks_behaves_exactly_as_before(self):
        path = self._write_trace_with_n_events(10)
        try:
            trace = chrome_trace.load_trace_from_json(path)
            self.assertEqual(len(trace.spans), 10)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_progress_cb_called_with_final_total(self):
        path = self._write_trace_with_n_events(10)
        calls = []
        try:
            chrome_trace.load_trace_from_json(path, progress_cb=lambda i, t: calls.append((i, t)))
        finally:
            Path(path).unlink(missing_ok=True)
        self.assertTrue(calls)
        # Every call reports the same total (the event count is known
        # up front, right after json.load()).
        totals = {t for _, t in calls}
        self.assertEqual(len(totals), 1)
        # The LAST progress call must report the final index -- "reached
        # 100%", not silently stopping short.
        self.assertEqual(calls[-1][0], calls[-1][1])

    def test_progress_cb_reports_increasing_progress_on_a_larger_trace(self):
        # Large enough to cross the internal reporting stride more than
        # once, so this proves genuinely incremental progress, not just
        # a single "done" call.
        path = self._write_trace_with_n_events(12_000)
        calls = []
        try:
            chrome_trace.load_trace_from_json(path, progress_cb=lambda i, t: calls.append((i, t)))
        finally:
            Path(path).unlink(missing_ok=True)
        self.assertGreater(len(calls), 2)
        indices = [i for i, _ in calls]
        self.assertEqual(indices, sorted(indices))
        # The final call's index must equal ITS OWN reported total (not a
        # hardcoded event count -- write() also emits one leading
        # "process_name" metadata event alongside the N spans, so the
        # true JSON event count is N+1, which is exactly what's being
        # measured here and is correct, not an off-by-one).
        self.assertEqual(indices[-1], calls[-1][1])

    def test_cancel_check_true_raises_load_cancelled_before_any_work(self):
        path = self._write_trace_with_n_events(10)
        try:
            with self.assertRaises(chrome_trace.LoadCancelled):
                chrome_trace.load_trace_from_json(path, cancel_check=lambda: True)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_cancel_check_false_loads_normally(self):
        path = self._write_trace_with_n_events(10)
        try:
            trace = chrome_trace.load_trace_from_json(path, cancel_check=lambda: False)
            self.assertEqual(len(trace.spans), 10)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_cancel_check_only_consulted_not_called_every_single_event(self):
        # cancel_check is checked on the same stride as progress_cb, not
        # once per event -- a real (if cheap) per-call cost shouldn't be
        # paid tens of thousands of times for a large trace.
        path = self._write_trace_with_n_events(12_000)
        call_count = [0]
        def counting_check():
            call_count[0] += 1
            return False
        try:
            chrome_trace.load_trace_from_json(path, cancel_check=counting_check)
        finally:
            Path(path).unlink(missing_ok=True)
        self.assertLess(call_count[0], 12_000)

    def test_load_cancelled_is_a_distinct_exception_type(self):
        # The GUI worker needs to tell "the user cancelled" apart from
        # "loading genuinely failed" -- LoadCancelled must not be
        # confusable with a generic Exception a classifier would treat
        # as a real failure.
        self.assertTrue(issubclass(chrome_trace.LoadCancelled, Exception))


if __name__ == "__main__":
    unittest.main()


class TestLosslessRoundTrip(unittest.TestCase):
    """Everything `hprofiler run` keeps in memory must survive the save/
    reload every other entry point (GUI, `view`, `summary`,
    `critical-path`, `merge-nodes`) goes through. Before this, span ids,
    parent links, GPU spans' real tid, the profiling window, instant tags
    and counter units were all dropped -- measured on a real MPI run:
    the critical-path graph lost every communication edge on reload."""

    def _roundtrip(self, trace: Trace) -> Trace:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            chrome_trace.write(trace, path)
            return chrome_trace.load_trace_from_json(path)
        finally:
            Path(path).unlink(missing_ok=True)

    def _mpi_trace(self) -> Trace:
        base = 432_000_123_456_789
        t = Trace(TraceMetadata(command="a.out", args=[], pid=7,
                                start_time_ns=base - 1000, end_time_ns=base + 10_000_000))
        mk = lambda name, typ, off, dur, **kw: SpanEvent(
            name=name, category=Category.MPI, start_ns=base + off, duration_ns=dur,
            pid=7, tid=7, tags={"type": typ, "rank": "0", **kw.pop("tags", {})}, **kw)
        t.add(mk("MPI_Irecv", "irecv", 0, 500, span_id="1", tags={"peer": "0", "tag": "5", "bytes": "8"}))
        t.add(mk("MPI_Isend", "isend", 1000, 500, span_id="2", tags={"peer": "0", "tag": "5", "bytes": "8"}))
        t.add(mk("MPI_Waitall", "waitall", 2000, 3000, parent_span_id="1;2"))
        t.add(mk("MPI_Irecv", "irecv", 6000, 500, span_id="3", tags={"peer": "-2", "tag": "-1", "wildcard": "1"}))
        t.add(mk("MPI_Wait", "wait", 7000, 800, parent_span_id="3", tags={"rpeer": "0", "rtag": "100"}))
        return t

    def test_span_ids_and_parent_links_survive(self):
        orig = self._mpi_trace()
        loaded = self._roundtrip(orig)
        self.assertEqual([(s.span_id, s.parent_span_id) for s in orig.spans],
                         [(s.span_id, s.parent_span_id) for s in loaded.spans])
        # and they are not left behind as ordinary tags
        for s in loaded.spans:
            self.assertNotIn("sid", s.tags)
            self.assertNotIn("psid", s.tags)

    def test_critical_path_graph_identical_after_reload(self):
        from src.analysis import criticalpath as cp
        orig = self._mpi_trace()
        loaded = self._roundtrip(orig)

        def edges(t):
            spans, preds = cp.build_dependency_graph(t)
            key = lambda i: (spans[i].name, spans[i].start_ns)
            return sorted((key(u), key(v), kind, conf)
                          for v, es in preds.items() for (u, kind, conf) in es)

        e_orig, e_loaded = edges(orig), edges(loaded)
        self.assertTrue(any(k == "explicit_span_id" for _, _, k, _ in e_orig))
        self.assertEqual(e_orig, e_loaded)

    def test_gpu_span_real_tid_survives_but_perfetto_track_is_virtual(self):
        t = Trace(TraceMetadata(command="a.out", args=[]))
        t.add(SpanEvent(name="k", category=Category.GPU_CUDA, start_ns=1_000, duration_ns=50,
                        pid=3, tid=4242, tags={"type": "kernel", "stream": "0"}))
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            chrome_trace.write(t, path)
            import json
            raw = [e for e in json.loads(Path(path).read_text())["traceEvents"] if e.get("ph") == "X"]
            self.assertGreaterEqual(raw[0]["tid"], 2_000_000_000)
            loaded = chrome_trace.load_trace_from_json(path)
        finally:
            Path(path).unlink(missing_ok=True)
        self.assertEqual(loaded.spans[0].tid, 4242)
        self.assertNotIn("_tid", loaded.spans[0].tags)

    def test_profiling_window_and_pid_survive(self):
        orig = self._mpi_trace()
        loaded = self._roundtrip(orig)
        self.assertEqual(loaded.metadata.start_time_ns, orig.metadata.start_time_ns)
        self.assertEqual(loaded.metadata.end_time_ns, orig.metadata.end_time_ns)
        self.assertEqual(loaded.duration_ns, orig.duration_ns)
        self.assertEqual(loaded.metadata.pid, 7)

    def test_old_file_without_window_uses_event_extent_not_load_time(self):
        import json, time
        orig = self._mpi_trace()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name
        try:
            chrome_trace.write(orig, path)
            data = json.loads(Path(path).read_text())
            for k in ("startTimeNs", "endTimeNs"):
                del data["metadata"][k]
            Path(path).write_text(json.dumps(data))
            loaded = chrome_trace.load_trace_from_json(path)
        finally:
            Path(path).unlink(missing_ok=True)
        first = min(s.start_ns for s in orig.spans)
        last = max(s.end_ns for s in orig.spans)
        d1 = loaded.duration_ns
        time.sleep(0.01)
        self.assertEqual(d1, last - first)
        self.assertEqual(loaded.duration_ns, d1)  # must not grow with wall-clock time

    def test_instant_tags_and_counter_units_survive(self):
        from src.core.events import InstantEvent, CounterEvent
        t = Trace(TraceMetadata(command="a.out", args=[]))
        t.add(InstantEvent(name="m", category=Category.NVTX, timestamp_ns=10, pid=1, tid=2, tags={"x": "1"}))
        t.add(CounterEvent(name="mem", category=Category.MEMORY, timestamp_ns=20, value=1.5, unit="MB", pid=1))
        loaded = self._roundtrip(t)
        self.assertEqual(loaded.instants[0].tags, {"x": "1"})
        self.assertEqual(loaded.counters[0].unit, "MB")
        self.assertEqual(loaded.counters[0].value, 1.5)

    def test_counter_without_value_is_skipped_not_zero(self):
        import json
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump({"traceEvents": [{"ph": "C", "name": "gpu_util", "cat": "other", "ts": 1.0, "pid": 1}]}, f)
            path = f.name
        try:
            loaded = chrome_trace.load_trace_from_json(path)
        finally:
            Path(path).unlink(missing_ok=True)
        self.assertEqual(loaded.counters, [])

    def test_timestamps_exact_at_realistic_magnitudes(self):
        import random
        rng = random.Random(7)
        t = Trace(TraceMetadata(command="a.out", args=[]))
        for i in range(3000):
            t.add(SpanEvent(name=f"f{i}", category=Category.CPU, start_ns=rng.randint(10**12, 10**15),
                            duration_ns=rng.choice([1, 2, 7, 999, 123_457, 10**9 + 1]), pid=1, tid=1))
        loaded = self._roundtrip(t)
        for a, b in zip(t.spans, loaded.spans):
            self.assertEqual((a.start_ns, a.duration_ns), (b.start_ns, b.duration_ns))
