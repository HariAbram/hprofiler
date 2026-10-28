"""
Where the hook libraries (libhprofiler_*.so) are loaded from: build/lib,
or $HPROFILER_HOOKS_DIR when set -- e.g. a separate build of the hooks for
this host, while build/ belongs to another environment (a container).
"""
from __future__ import annotations

import os
from pathlib import Path

DEFAULT_HOOKS_DIR = Path(__file__).resolve().parent.parent.parent / "build" / "lib"


def hooks_dir() -> Path:
    override = os.environ.get("HPROFILER_HOOKS_DIR")
    return Path(override) if override else DEFAULT_HOOKS_DIR
