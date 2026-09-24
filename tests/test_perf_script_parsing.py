"""
Regression tests for src/core/runner.py's _parse_perf_script() -- no
tests existed for this function at all before (a real pre-existing gap).
Covers the rewrite that makes --perf-callgraph-enabled CPU sampling
produce ONE SpanEvent per sample (name=leaf frame, stack_frames=ancestor
chain, duration_ns=a nominal per-sample weight) instead of the old shape
(one span per stack FRAME per sample, duration_ns=0, stack_frames never
set, the whole stack redundantly duplicated as a string in
tags["stack"]) -- the old shape was structurally invisible to
analysis/call_tree.py's _ct_build (Call Tree tab / Flame Graph tab),
which requires duration_ns>0 AND stack_frames truthy.

Can't run real `perf record`/`perf script` in this sandbox (same
documented perf_event_paranoid limitation as the rest of this project's
perf-dependent work) -- mocks subprocess.run to return synthetic `perf
script` text in the exact format the real tool produces, verified
against the _HDR/_FRAME regexes' own documented format comments.
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.trace import Trace, TraceMetadata
from src.core.runner import _parse_perf_script


def _fake_perf_script_result(stdout: str, stderr: str = ""):
    r = MagicMock()
    r.stdout = stdout
    r.stderr = stderr
    return r


# Two samples, each a 3-deep stack (innermost frame first, as perf script
# itself prints it): sample 1 lands in "leaf_a" called by "mid" called by
# "main"; sample 2 lands in "leaf_b" via the same "mid"/"main" chain.
_STACKED_OUTPUT = """myprog 100/200 1000.000000:     100 cycles:u:
\t7f0001 leaf_a (/bin/myprog)
\t7f0002 mid (/bin/myprog)
\t7f0003 main (/bin/myprog)

myprog 100/200 1000.000100:     100 cycles:u:
\t7f0004 leaf_b (/bin/myprog)
\t7f0002 mid (/bin/myprog)
\t7f0003 main (/bin/myprog)

"""

# Single sample, no --perf-callgraph -- no indented frame lines, no blank
# lines at all (has_stacks is False).
_FLAT_OUTPUT = (
    "myprog 100/200 1000.000000:     100 cycles:u: 7f0001 leaf_a (/bin/myprog)\n"
)


class TestParsePerfScriptStacked(unittest.TestCase):
    def _parse(self, freq=1000):
        trace = Trace(TraceMetadata())
        with patch("src.core.runner.subprocess.run",
                   return_value=_fake_perf_script_result(_STACKED_OUTPUT)):
            _parse_perf_script("/fake/perf.data", trace, freq)
        return trace

    def test_one_span_per_sample_not_per_frame(self):
        trace = self._parse()
        # 2 samples -> 2 spans, NOT 2 samples * 3 frames = 6.
        self.assertEqual(len(trace.spans), 2)

    def test_span_name_is_the_leaf_frame(self):
        trace = self._parse()
        names = {s.name for s in trace.spans}
        self.assertEqual(names, {"leaf_a", "leaf_b"})

    def test_stack_frames_are_ancestors_innermost_first(self):
        trace = self._parse()
        leaf_a = next(s for s in trace.spans if s.name == "leaf_a")
        # perf script prints innermost-first; "mid" (nearer the leaf)
        # must come before "main" (the outer/root frame).
        self.assertEqual(leaf_a.stack_frames, ["mid", "main"])

    def test_no_stack_tag_left_on_the_span(self):
        trace = self._parse()
        for s in trace.spans:
            self.assertNotIn("stack", s.tags)
            self.assertNotIn("depth", s.tags)

    def test_duration_is_nominal_per_sample_weight(self):
        trace = self._parse(freq=1000)  # 1000 Hz -> 1e6 ns/sample
        for s in trace.spans:
            self.assertEqual(s.duration_ns, 1_000_000)

    def test_higher_freq_gives_smaller_per_sample_weight(self):
        trace = self._parse(freq=1_000_000)  # 1 MHz -> 1000 ns/sample
        for s in trace.spans:
            self.assertEqual(s.duration_ns, 1000)

    def test_pid_and_tid_parsed_correctly(self):
        trace = self._parse()
        for s in trace.spans:
            self.assertEqual(s.pid, 100)
            self.assertEqual(s.tid, 200)

    def test_resulting_spans_feed_a_real_flame_tree(self):
        # End-to-end: the rewritten spans must actually work with the
        # Flame Graph tab's tree builder, not just look superficially
        # right in isolation.
        from src.analysis.flamegraph_tree import build_flame_tree
        trace = self._parse()
        tree = build_flame_tree(trace.spans)
        self.assertEqual(tree["value"], 2_000_000)  # 2 samples * 1e6 ns
        main = tree["children"][0]
        self.assertEqual(main["name"], "main")
        mid = main["children"][0]
        self.assertEqual(mid["name"], "mid")
        leaf_names = {c["name"] for c in mid["children"]}
        self.assertEqual(leaf_names, {"leaf_a", "leaf_b"})


class TestParsePerfScriptFlatUnchanged(unittest.TestCase):
    """The no---perf-callgraph path is explicitly out of scope for the
    rewrite -- confirms it still behaves exactly as before."""

    def test_flat_samples_still_have_zero_duration_and_no_stack_frames(self):
        trace = Trace(TraceMetadata())
        with patch("src.core.runner.subprocess.run",
                   return_value=_fake_perf_script_result(_FLAT_OUTPUT)):
            _parse_perf_script("/fake/perf.data", trace, 1000)
        self.assertEqual(len(trace.spans), 1)
        s = trace.spans[0]
        self.assertEqual(s.name, "leaf_a")
        self.assertEqual(s.duration_ns, 0)
        self.assertEqual(s.stack_frames, [])


if __name__ == "__main__":
    unittest.main()
