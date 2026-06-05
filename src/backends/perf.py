"""
CPU backend: Linux perf + /proc-based sampling fallback.
Uses DWARF call-graph unwinding for JIT-friendly stack capture (ACPP, etc).
"""

from __future__ import annotations
import shutil
import subprocess
from .base import Backend


class PerfBackend(Backend):
    name = "cpu"
    description = "CPU sampling via Linux perf (DWARF call-graph, JIT-aware)"

    def is_available(self) -> bool:
        if not shutil.which("perf"):
            return False
        try:
            r = subprocess.run(["perf", "stat", "true"],
                               capture_output=True, timeout=5)
            return r.returncode == 0
        except Exception:
            return False
