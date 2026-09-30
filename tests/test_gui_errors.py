"""
Tests for src/gui/errors.py -- HprofilerLoadError construction/wrapping
and classify_load_exception()'s dispatch from raw Python exceptions to
the closed ErrorKind vocabulary. Pure Python, no Qt/PySide6 dependency at
all (errors.py imports nothing from PySide6), so this file needs no
QGuiApplication and no offscreen QPA setup -- runs on any Python install.
"""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.gui.errors import HprofilerLoadError, ErrorKind, classify_load_exception


class TestHprofilerLoadError(unittest.TestCase):
    def test_default_message_filled_in_per_kind(self):
        e = HprofilerLoadError(kind=ErrorKind.PERMISSION_DENIED)
        self.assertTrue(e.message)
        self.assertNotEqual(e.message, "")

    def test_explicit_message_overrides_default(self):
        e = HprofilerLoadError(kind=ErrorKind.INVALID_INPUT, message="custom")
        self.assertEqual(e.message, "custom")

    def test_str_returns_message_not_repr(self):
        e = HprofilerLoadError(kind=ErrorKind.INTERNAL_ERROR, message="oops")
        self.assertEqual(str(e), "oops")

    def test_is_raisable_and_catchable_as_an_exception(self):
        with self.assertRaises(HprofilerLoadError) as ctx:
            raise HprofilerLoadError(kind=ErrorKind.UNSUPPORTED_DATA, message="bad shape")
        self.assertEqual(ctx.exception.kind, ErrorKind.UNSUPPORTED_DATA)
        self.assertEqual(ctx.exception.message, "bad shape")

    def test_is_catchable_as_a_plain_exception(self):
        with self.assertRaises(Exception):
            raise HprofilerLoadError(kind=ErrorKind.INTERNAL_ERROR)

    def test_wrap_unexpected_captures_traceback_text(self):
        try:
            raise RuntimeError("boom")
        except RuntimeError as exc:
            wrapped = HprofilerLoadError.wrap_unexpected(exc, stage="computing_dashboard")
        self.assertEqual(wrapped.kind, ErrorKind.INTERNAL_ERROR)
        self.assertIn("boom", wrapped.detail)
        self.assertIn("RuntimeError", wrapped.traceback_text)
        self.assertIn("boom", wrapped.traceback_text)
        self.assertEqual(wrapped.stage, "computing_dashboard")

    def test_to_dict_shape(self):
        e = HprofilerLoadError(kind=ErrorKind.MISSING_DEPENDENCY, detail="d", file="f.json", stage="s")
        d = e.to_dict()
        self.assertEqual(d["kind"], "missing_dependency")
        self.assertEqual(d["detail"], "d")
        self.assertEqual(d["file"], "f.json")
        self.assertEqual(d["stage"], "s")
        self.assertIn("message", d)
        self.assertIn("tracebackText", d)

    def test_to_json_round_trips_through_json_loads(self):
        e = HprofilerLoadError(kind=ErrorKind.INVALID_INPUT, detail="d")
        parsed = json.loads(e.to_json())
        self.assertEqual(parsed["kind"], "invalid_input")

    def test_every_error_kind_has_a_default_message(self):
        for kind in ErrorKind:
            e = HprofilerLoadError(kind=kind)
            self.assertTrue(e.message, f"{kind} has no default message")


class TestClassifyLoadException(unittest.TestCase):
    def test_file_not_found_is_invalid_input(self):
        e = classify_load_exception(FileNotFoundError("nope"), file="x.json")
        self.assertEqual(e.kind, ErrorKind.INVALID_INPUT)
        self.assertEqual(e.file, "x.json")

    def test_permission_error_is_permission_denied(self):
        e = classify_load_exception(PermissionError("nope"), file="x.json")
        self.assertEqual(e.kind, ErrorKind.PERMISSION_DENIED)

    def test_is_a_directory_is_invalid_input(self):
        e = classify_load_exception(IsADirectoryError("dir"), file="somedir")
        self.assertEqual(e.kind, ErrorKind.INVALID_INPUT)

    def test_json_decode_error_is_invalid_input(self):
        try:
            json.loads("{not valid")
        except json.JSONDecodeError as exc:
            e = classify_load_exception(exc, file="x.json")
        self.assertEqual(e.kind, ErrorKind.INVALID_INPUT)

    def test_import_error_is_missing_dependency(self):
        e = classify_load_exception(ImportError("no module named foo"))
        self.assertEqual(e.kind, ErrorKind.MISSING_DEPENDENCY)

    def test_attribute_error_is_unsupported_data(self):
        # This is exactly the exception class load_trace_from_json's
        # event-parsing loop raises for a well-formed-JSON-but-foreign-
        # schema file (a non-dict event, e.g.) -- see chrome_trace.py.
        e = classify_load_exception(AttributeError("'list' object has no attribute 'get'"))
        self.assertEqual(e.kind, ErrorKind.UNSUPPORTED_DATA)

    def test_type_error_is_unsupported_data(self):
        e = classify_load_exception(TypeError("unsupported operand"))
        self.assertEqual(e.kind, ErrorKind.UNSUPPORTED_DATA)

    def test_value_error_is_unsupported_data(self):
        e = classify_load_exception(ValueError("bad value"))
        self.assertEqual(e.kind, ErrorKind.UNSUPPORTED_DATA)

    def test_key_error_is_unsupported_data(self):
        e = classify_load_exception(KeyError("missing"))
        self.assertEqual(e.kind, ErrorKind.UNSUPPORTED_DATA)

    def test_unrecognized_exception_is_internal_error(self):
        e = classify_load_exception(RuntimeError("mystery"))
        self.assertEqual(e.kind, ErrorKind.INTERNAL_ERROR)
        self.assertTrue(e.traceback_text)

    def test_already_classified_error_passes_through_unchanged(self):
        original = HprofilerLoadError(kind=ErrorKind.PERMISSION_DENIED, message="already classified")
        result = classify_load_exception(original)
        self.assertIs(result, original)

    def test_stage_is_preserved_through_classification(self):
        e = classify_load_exception(FileNotFoundError("nope"), file="x.json", stage="reading_file")
        self.assertEqual(e.stage, "reading_file")


if __name__ == "__main__":
    unittest.main()
