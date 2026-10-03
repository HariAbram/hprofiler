"""
Logging setup for the GUI. Backs the Help menu's "Open Log File" action and
gives every HprofilerLoadError (src/gui/errors.py) a durable record beyond
what was visible in the GUI at the time.

Uses QStandardPaths.AppDataLocation (~/.local/share/hprofiler on Linux),
SEPARATE from QSettings' config-file location (~/.config/hprofiler, see
settings.py) -- platform convention (config vs. data/log directories), and
"open the log" points at exactly one file.
"""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOGGER_NAME = "hprofiler.gui"
_LOG_FILENAME = "hprofiler-gui.log"

_configured = False
_log_path: Path | None = None


def setup_logging(log_dir: Path | str | None = None) -> Path:
    """Configures the `hprofiler.gui` logger with a rotating file
    handler and returns the log file's path. Safe to call more than
    once (e.g. once per profile-switch subprocess) -- only the first
    call actually attaches a handler; subsequent calls just return the
    already-established path, so log lines from repeated calls in the
    same process never duplicate.

    `log_dir` is overridable (tests pass a temp directory) -- defaults
    to QStandardPaths.AppDataLocation, which requires a QCoreApplication/
    QGuiApplication to already exist (it reads the app/org name set on
    it) -- callers must construct the QGuiApplication before calling
    this with no explicit `log_dir`."""
    global _configured, _log_path

    if log_dir is None:
        from PySide6.QtCore import QStandardPaths
        resolved_dir = Path(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation))
    else:
        resolved_dir = Path(log_dir)

    resolved_dir.mkdir(parents=True, exist_ok=True)
    log_path = resolved_dir / _LOG_FILENAME

    if _configured:
        return _log_path if _log_path is not None else log_path

    handler = RotatingFileHandler(log_path, maxBytes=2_000_000, backupCount=2, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))

    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    _configured = True
    _log_path = log_path
    logger.info("hprofiler GUI logging started")
    return log_path


def get_logger() -> logging.Logger:
    """The logger any GUI code should use -- `setup_logging()` should
    have already been called once at startup; if it hasn't, this still
    returns a usable (unconfigured -- no handler, so nothing is written
    anywhere) logger rather than raising, matching Python logging's own
    "safe to log before configuration, it just goes nowhere" convention."""
    return logging.getLogger(_LOGGER_NAME)


def current_log_path() -> Path | None:
    """The active log file's path, or None if setup_logging() hasn't
    been called yet in this process."""
    return _log_path


def log_error(error: "object", *, extra_context: str = "") -> None:
    """Logs an HprofilerLoadError (or anything with a to_dict()/message/
    tracebackText shape) at ERROR level with its full detail -- the one
    call site every load-failure path should go through, so "open the
    log" is reliably useful rather than empty for anything but crashes."""
    logger = get_logger()
    message = getattr(error, "message", str(error))
    stage = getattr(error, "stage", "")
    traceback_text = getattr(error, "traceback_text", "")
    prefix = f"[{stage}] " if stage else ""
    suffix = f" ({extra_context})" if extra_context else ""
    logger.error("%s%s%s", prefix, message, suffix)
    if traceback_text:
        logger.error(traceback_text)
