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


if __name__ == "__main__":
    unittest.main()
