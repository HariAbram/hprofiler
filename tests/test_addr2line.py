"""
Regression test for a self-audit bug: llvm-symbolizer's --inlining default
(on) means a single input address can produce MULTIPLE (function, file:line)
pairs before the blank separator, not just one. The previous parser only
consumed the first pair per address, then assumed the very next line was
already the blank separator -- for an inlined address it wasn't (it was the
next inlined frame's pair), so it read those leftover lines as if they
belonged to the NEXT address, misaligning every subsequent address's result.
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis import addr2line


class TestLLVMSymbolizerBatchParsing(unittest.TestCase):
    def test_inlined_address_does_not_misalign_subsequent_addresses(self):
        # addr2 resolves through inlining: TWO (func, file:line) pairs
        # before its blank separator, unlike addr1 and addr3.
        fake_stdout = (
            "funcA\n"
            "file1.cpp:10\n"
            "\n"
            "inner_func\n"
            "file2.cpp:20\n"
            "outer_func\n"
            "file2.cpp:5\n"
            "\n"
            "funcC\n"
            "file3.cpp:30\n"
            "\n"
        )
        addresses = ["0x1", "0x2", "0x3"]

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=fake_stdout, returncode=0)
            result = addr2line._resolve_batch("llvm-symbolizer", "/fake/binary", addresses)

        self.assertEqual(result["0x1"], ("file1.cpp", "10"))
        # innermost (first) frame of the inlined chain, not the outer one
        self.assertEqual(result["0x2"], ("file2.cpp", "20"))
        # the bug: without the fix, addr3 would incorrectly get
        # ("file2.cpp", "5") -- addr2's OUTER frame -- instead of its own
        # real location.
        self.assertEqual(result["0x3"], ("file3.cpp", "30"))

    def test_no_inlining_unaffected(self):
        fake_stdout = (
            "funcA\nfile1.cpp:10\n\n"
            "funcB\nfile2.cpp:20\n\n"
            "funcC\nfile3.cpp:30\n\n"
        )
        addresses = ["0x1", "0x2", "0x3"]
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=fake_stdout, returncode=0)
            result = addr2line._resolve_batch("llvm-symbolizer", "/fake/binary", addresses)
        self.assertEqual(result["0x1"], ("file1.cpp", "10"))
        self.assertEqual(result["0x2"], ("file2.cpp", "20"))
        self.assertEqual(result["0x3"], ("file3.cpp", "30"))

    def test_unresolvable_address_skipped_without_misaligning(self):
        fake_stdout = (
            "??\n??:0\n\n"
            "funcB\nfile2.cpp:20\n\n"
        )
        addresses = ["0x1", "0x2"]
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=fake_stdout, returncode=0)
            result = addr2line._resolve_batch("llvm-symbolizer", "/fake/binary", addresses)
        self.assertNotIn("0x1", result)
        self.assertEqual(result["0x2"], ("file2.cpp", "20"))


if __name__ == "__main__":
    unittest.main()
