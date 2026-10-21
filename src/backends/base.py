"""Base class for profiling backends."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.trace import Trace


class Backend(ABC):
    name: str = ""
    description: str = ""

    @abstractmethod
    def is_available(self) -> bool:
        """True if this backend is installed: its hook or tool and the
        library it needs are present (see usability() for whether it can
        actually work here)."""

    def usability(self) -> tuple[str, str]:
        """("usable" | "installed" | "missing", reason): installed but not
        usable when a probe (src/backends/probes.py) shows the hardware or
        access it needs is absent. Auto-detection picks usable backends only."""
        return ("usable", "") if self.is_available() else ("missing", "")

    def env_vars(self) -> dict[str, str]:
        """Additional environment variables to inject into the profiled process."""
        return {}

    def preload_libs(self) -> list[str]:
        """Paths to shared libraries to inject via LD_PRELOAD."""
        return []

    def wrap_command(self, command: list[str]) -> list[str] | None:
        """Return a wrapped command (e.g. prepend a tool), or None for no wrapping."""
        return None

    def post_process(self, trace: "Trace") -> None:
        """Called after the profiled process exits; can add events to the trace."""
