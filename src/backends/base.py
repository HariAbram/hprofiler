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
        """Return True if this backend can be used on the current system."""

    def env_vars(self) -> dict[str, str]:
        """Additional environment variables to inject into the profiled process."""
        return {}

    def preload_libs(self) -> list[str]:
        """Paths to shared libraries to inject via LD_PRELOAD."""
        return []

    def post_process(self, trace: "Trace") -> None:
        """Called after the profiled process exits; can add events to the trace."""
