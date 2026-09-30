"""
Error classification for GUI profile loading (Phase 2 of the usability/
persistence/loading overhaul). Pure Python, no Qt dependency -- produced
on the loading worker thread (src/gui/loader.py), consumed by
ErrorState.qml (via a QVariantMap-shaped dict) and the log (see
logging_setup.py). Wraps the specific, currently-uncaught exception
sites in src/output/chrome_trace.py's load_trace_from_json() and any
bridge-construction failure, replacing "raw traceback crashes the
subprocess" with a concise, classified, user-facing message plus the
full original detail kept available (not discarded) for the expandable
technical-details panel.

Kind vocabulary is intentionally small and closed (an Enum, not free-form
strings) so ErrorState.qml can switch on it for icon/color choice without
guessing at string values, mirroring this project's established closed-
vocabulary conventions elsewhere (e.g. Field's measured/derived/
estimated/unavailable kind).
"""
from __future__ import annotations

import json
import traceback
from dataclasses import dataclass
from enum import Enum
from typing import Any


class ErrorKind(str, Enum):
    INVALID_INPUT = "invalid_input"           # malformed JSON, missing required fields
    UNSUPPORTED_DATA = "unsupported_data"      # foreign/future trace schema, unrecognized shape
    MISSING_DEPENDENCY = "missing_dependency"  # an optional tool/library this feature needs isn't present
    PERMISSION_DENIED = "permission_denied"    # couldn't read the trace file
    METRIC_UNAVAILABLE = "metric_unavailable"  # one sub-computation failed; the rest of the load can proceed
    INTERNAL_ERROR = "internal_error"          # anything else -- caught and wrapped, never raised raw


# One human-readable, concise (not alarming, not vague) default message
# per kind -- callers may still pass their own more specific `message`,
# this is only the fallback.
_DEFAULT_MESSAGES: dict[ErrorKind, str] = {
    ErrorKind.INVALID_INPUT: "This file isn't a valid trace.",
    ErrorKind.UNSUPPORTED_DATA: "This file doesn't look like a trace hprofiler produced.",
    ErrorKind.MISSING_DEPENDENCY: "A required tool or library isn't available.",
    ErrorKind.PERMISSION_DENIED: "This file couldn't be read.",
    ErrorKind.METRIC_UNAVAILABLE: "This metric isn't available for this trace.",
    ErrorKind.INTERNAL_ERROR: "Something went wrong while loading this profile.",
}


@dataclass
class HprofilerLoadError(Exception):
    """A classified, user-facing load failure. Deliberately a plain
    dataclass (not just a string) so `message` (short, shown by default)
    stays separate from `detail`/`traceback_text` (the original
    exception's own text, shown only when the user expands "technical
    details") -- concise-by-default, nothing discarded.

    `stage`/`file` identify WHERE this happened (a LoadStage value and,
    where meaningful, the specific file involved -- the trace file
    itself, or a source file a metric tried and failed to read) --
    exposed in the technical-details panel alongside the traceback, per
    the "relevant file and failed stage" requirement."""

    kind: ErrorKind
    message: str = ""
    detail: str = ""
    file: str = ""
    stage: str = ""
    traceback_text: str = ""

    def __post_init__(self) -> None:
        if not self.message:
            self.message = _DEFAULT_MESSAGES[self.kind]

    def __str__(self) -> str:
        return self.message

    @classmethod
    def wrap_unexpected(cls, exc: BaseException, *, stage: str = "", file: str = "") -> "HprofilerLoadError":
        """The catch-all: anything not already recognized as a specific
        kind becomes INTERNAL_ERROR, with the real traceback preserved
        (never discarded, never shown by default)."""
        return cls(
            kind=ErrorKind.INTERNAL_ERROR,
            detail=str(exc),
            file=file,
            stage=stage,
            traceback_text="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        )

    def to_dict(self) -> dict[str, Any]:
        """QML-consumable shape -- what ErrorState.qml/the log actually
        read. `kind` is the Enum's own string value (already a plain
        str via `str, Enum`), not a Python-only object."""
        return {
            "kind": self.kind.value,
            "message": self.message,
            "detail": self.detail,
            "file": self.file,
            "stage": self.stage,
            "tracebackText": self.traceback_text,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def classify_load_exception(exc: BaseException, *, file: str = "", stage: str = "") -> HprofilerLoadError:
    """Maps a raw exception raised while loading a trace file to a
    classified HprofilerLoadError. This is the ONE place that decides
    "what kind of failure was this" for the loading path -- see
    src/gui/loader.py for where it's called (wrapping
    load_trace_from_json() and bridge construction)."""
    if isinstance(exc, HprofilerLoadError):
        return exc
    if isinstance(exc, PermissionError):
        return HprofilerLoadError(
            kind=ErrorKind.PERMISSION_DENIED,
            message=f"Permission denied reading {file or 'this file'}.",
            detail=str(exc), file=file, stage=stage,
        )
    if isinstance(exc, FileNotFoundError):
        return HprofilerLoadError(
            kind=ErrorKind.INVALID_INPUT,
            message=f"{file or 'This file'} doesn't exist.",
            detail=str(exc), file=file, stage=stage,
        )
    if isinstance(exc, IsADirectoryError):
        return HprofilerLoadError(
            kind=ErrorKind.INVALID_INPUT,
            message=f"{file or 'This path'} is a directory, not a trace file.",
            detail=str(exc), file=file, stage=stage,
        )
    if isinstance(exc, json.JSONDecodeError):
        return HprofilerLoadError(
            kind=ErrorKind.INVALID_INPUT,
            message="This file isn't valid JSON.",
            detail=str(exc), file=file, stage=stage,
        )
    if isinstance(exc, ImportError):
        return HprofilerLoadError(
            kind=ErrorKind.MISSING_DEPENDENCY,
            message=f"A required dependency is missing: {exc.name or str(exc)}.",
            detail=str(exc), file=file, stage=stage,
        )
    if isinstance(exc, (AttributeError, TypeError, ValueError, KeyError)):
        # These are exactly the exception types load_trace_from_json's
        # event-parsing loop raises for a well-formed-JSON-but-foreign-
        # schema file (see chrome_trace.py) -- a real JSON file, just not
        # one shaped like a trace hprofiler produced.
        return HprofilerLoadError(
            kind=ErrorKind.UNSUPPORTED_DATA,
            detail=str(exc), file=file, stage=stage,
        )
    return HprofilerLoadError.wrap_unexpected(exc, stage=stage, file=file)
