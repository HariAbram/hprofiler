"""
Regression test for DisasmWidget._show_disasm's "No disassembly
available" message (src/ui/app.py) -- it used to show the exact same
generic "install objdump/cuobjdump/llvm-objdump" tips regardless of WHY
disassembly was missing. A real user hit this after a fix that made
gomp_hook.c/mpi_hook.c resolve call-site sym=/lib= tags (see
project_disasm_codeptr_fix memory): they were still viewing a trace
captured BEFORE rebuilding the hooks (or with a construct that genuinely
doesn't resolve a tag yet), so every span had no sym=/lib= tag at all --
but the old message told them to go install objdump, which was never the
actual problem and sent them chasing the wrong fix.

The message now distinguishes three cases by checking the actual spans'
tags: (1) no sym=/lib= tag anywhere for this name -- nothing was ever
resolved, most likely a stale trace or an unrebuilt hook, NOT a missing-
tool problem; (2) a sym=/lib= tag IS present but disasm still failed --
now the objdump/nm tips are actually relevant; (3) a GPU kernel with no
disasm -- the original CUDA/ROCm-specific tips, unchanged.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from textual.app import App, ComposeResult
from textual.widgets import RichLog

from src.core.trace import Trace, TraceMetadata
from src.core.events import SpanEvent, Category
from src.ui.app import DisasmWidget


def _span(name, cat, tags=None):
    return SpanEvent(name=name, category=cat, start_ns=0, duration_ns=1000,
                     pid=1, tid=1, tags=dict(tags or {}))


def _mk_trace(spans):
    t = Trace(TraceMetadata(command="./app"))
    for s in spans:
        t.add(s)
    return t


class _HostApp(App):
    def __init__(self, trace, **kwargs):
        super().__init__(**kwargs)
        self._trace = trace

    def compose(self) -> ComposeResult:
        yield DisasmWidget(self._trace, id="disasm")


def _rendered_text(log: RichLog) -> str:
    return "\n".join(getattr(strip, "text", str(strip)) for strip in log.lines)


class TestDisasmMissingMessage(unittest.IsolatedAsyncioTestCase):
    async def test_no_codeptr_tag_blames_stale_trace_not_missing_tools(self):
        trace = _mk_trace([_span("omp_barrier", Category.SYNC, {"type": "sync"})])
        app = _HostApp(trace)
        async with app.run_test(size=(120, 40)):
            text = _rendered_text(app.query_one("#disasm-log", RichLog))
            self.assertIn("no resolved call-site symbol", text)
            self.assertIn("NOT a missing objdump", text)
            self.assertIn("Rebuild the hooks", text)
            self.assertNotIn("install cuobjdump", text)

    async def test_sym_tag_present_shows_tool_tips(self):
        trace = _mk_trace([_span("omp_parallel_region", Category.OPENMP,
                                 {"type": "parallel_region", "sym": "main"})])
        app = _HostApp(trace)
        async with app.run_test(size=(120, 40)):
            text = _rendered_text(app.query_one("#disasm-log", RichLog))
            self.assertIn("sym=main", text)
            self.assertIn("objdump and/or nm must be installed", text)
            self.assertNotIn("Rebuild the hooks", text)

    async def test_lib_offset_tag_present_shows_tool_tips(self):
        trace = _mk_trace([_span("omp_barrier", Category.SYNC,
                                 {"type": "sync", "lib": "/opt/app/app", "offset": "0x1a2b"})])
        app = _HostApp(trace)
        async with app.run_test(size=(120, 40)):
            text = _rendered_text(app.query_one("#disasm-log", RichLog))
            self.assertIn("/opt/app/app", text)
            self.assertIn("offset=0x1a2b", text)
            self.assertIn("objdump and/or nm must be installed", text)

    async def test_gpu_kernel_keeps_original_cuda_rocm_tips(self):
        trace = _mk_trace([_span("my_kernel", Category.GPU_CUDA, {"type": "kernel"})])
        app = _HostApp(trace)
        async with app.run_test(size=(120, 40)):
            text = _rendered_text(app.query_one("#disasm-log", RichLog))
            self.assertIn("install cuobjdump", text)
            self.assertIn("llvm-objdump", text)
            self.assertNotIn("Rebuild the hooks", text)


if __name__ == "__main__":
    unittest.main()
