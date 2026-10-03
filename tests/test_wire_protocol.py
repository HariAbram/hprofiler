"""
Regression tests for _parse_record's handling of inst: records (src/core/runner.py).

The inst: branch must keep the trailing tags segment (MPI_Test/
MPI_Testany/MPI_Testsome/MPI_Testall/MPI_Cancel instants carry
psid=/rpeer=/rtag=/flag= there, see mpi_hook.c) and must split inst's
"name[:tags]" tail at its own position (one later than span's shape), so
a colon-containing name is not truncated.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.runner import _parse_record
from src.core.events import SpanEvent, InstantEvent, Category


class TestInstantTagParsing(unittest.TestCase):
    def test_inst_with_tags(self):
        line = "inst:mpi:1234:5678:99999:MPI_Test:type=test,flag=1,rank=0,psid=5\n"
        ev = _parse_record(line)
        self.assertIsInstance(ev, InstantEvent)
        self.assertEqual(ev.name, "MPI_Test")
        self.assertEqual(ev.category, Category.MPI)
        self.assertEqual(ev.timestamp_ns, 99999)
        self.assertEqual(ev.pid, 1234)
        self.assertEqual(ev.tid, 5678)
        self.assertEqual(ev.tags, {"type": "test", "flag": "1", "rank": "0", "psid": "5"})

    def test_inst_without_tags_backward_compatible(self):
        line = "inst:cuda:100:200:12345:jit_compile_done\n"
        ev = _parse_record(line)
        self.assertIsInstance(ev, InstantEvent)
        self.assertEqual(ev.name, "jit_compile_done")
        self.assertEqual(ev.tags, {})

    def test_inst_name_with_colon_not_truncated(self):
        # Demangled C++ names can contain "::" -- must not be chopped by the
        # shared top-level split(":", 6), same guarantee span: already has.
        line = "inst:mpi:1:2:3:Namespace::Method:rank=0\n"
        ev = _parse_record(line)
        self.assertIsInstance(ev, InstantEvent)
        self.assertEqual(ev.name, "Namespace::Method")
        self.assertEqual(ev.tags, {"rank": "0"})

    def test_inst_cancel_event_with_psid(self):
        line = "inst:mpi:42:42:1000:MPI_Cancel:type=cancel,rank=1,psid=7\n"
        ev = _parse_record(line)
        self.assertEqual(ev.name, "MPI_Cancel")
        self.assertEqual(ev.tags["psid"], "7")
        self.assertEqual(ev.tags["type"], "cancel")

    def test_span_tags_still_parse_unaffected(self):
        # Guard that the span: sibling branch parses tags too.
        line = "span:mpi:1:2:0:500:MPI_Waitany:type=waitany,rank=0,completed_index=2,psid=9\n"
        ev = _parse_record(line)
        self.assertIsInstance(ev, SpanEvent)
        self.assertEqual(ev.name, "MPI_Waitany")
        self.assertEqual(ev.parent_span_id, "9")
        self.assertEqual(ev.tags["completed_index"], "2")


if __name__ == "__main__":
    unittest.main()
