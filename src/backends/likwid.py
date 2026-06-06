"""
LIKWID backend: hardware PMU counter collection via likwid-perfctr.

Wraps the target command with likwid-perfctr so hardware counters are
collected over the full run. Results are parsed from the output file and
emitted as CounterEvents in the trace.

Configuration (environment variables):
  HPROFILER_LIKWID_GROUP   Counter group (default: FLOPS_DP)
  HPROFILER_LIKWID_CORES   CPU core range to monitor (default: all, e.g. "0-7")

Common groups:
  FLOPS_DP    Double-precision FLOPs + MFLOP/s
  FLOPS_SP    Single-precision FLOPs + MFLOP/s
  MEM         DRAM bandwidth (read/write GB/s)
  L2          L2 cache hit/miss rates
  L3          L3 cache hit/miss rates
  BRANCH      Branch prediction miss rate
  CLOCK       CPI / clock frequency only (always works, no special permissions)

Requirements:
  likwid-perfctr must be in PATH.
  PMU access: either run as root, or:
    sudo sysctl -w kernel.perf_event_paranoid=-1
    sudo sysctl -w kernel.nmi_watchdog=0
"""

from __future__ import annotations
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING
from .base import Backend

if TYPE_CHECKING:
    from ..core.trace import Trace

_DEFAULT_GROUP = "FLOPS_DP"


def _default_cores() -> str:
    n = os.cpu_count() or 1
    return f"0-{n - 1}" if n > 1 else "0"


class LIKWIDBackend(Backend):
    name = "likwid"
    description = "Hardware PMU counters via likwid-perfctr (FLOPS, bandwidth, cache, CPI)"

    def __init__(self) -> None:
        self.group = os.environ.get("HPROFILER_LIKWID_GROUP", _DEFAULT_GROUP)
        self.cores = os.environ.get("HPROFILER_LIKWID_CORES", _default_cores())
        self._output_file: str | None = None

    def is_available(self) -> bool:
        return shutil.which("likwid-perfctr") is not None

    def availability_note(self) -> str:
        if not self.is_available():
            return "install likwid: apt install likwid  (or build from github.com/RRZE-HPC/likwid)"
        return ""

    def wrap_command(self, command: list[str]) -> list[str]:
        self._output_file = tempfile.mktemp(suffix=".txt", prefix="hprofiler_likwid_")
        return [
            "likwid-perfctr",
            "-C", self.cores,
            "-g", self.group,
            "-o", self._output_file,
            "--",
        ] + command

    def post_process(self, trace: "Trace") -> None:
        if not self._output_file:
            return
        path = Path(self._output_file)
        if not path.exists():
            return
        try:
            text = path.read_text(errors="replace")
            _parse_and_emit(text, trace)
        except Exception:
            pass
        finally:
            try:
                path.unlink()
            except OSError:
                pass


# ── Output parser ─────────────────────────────────────────────────────────────

def _parse_metric_tables(text: str) -> list[tuple[str, list[float], str]]:
    """
    Parse all metric tables from likwid-perfctr output.
    Returns [(metric_name, [per_core_values], unit), ...].
    """
    results: list[tuple[str, list[float], str]] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Metric table header has "Metric" and at least one "Core N" column
        if (line.startswith("|")
                and "Metric" in line
                and re.search(r"Core\s+\d+", line)):
            i += 1
            # Skip separator line (+---+---+)
            if i < len(lines) and lines[i].strip().startswith("+"):
                i += 1
            # Parse data rows until next separator or non-pipe line
            while i < len(lines):
                row = lines[i].strip()
                if not row.startswith("|"):
                    break
                if row.startswith("+"):
                    i += 1
                    continue
                parts = [p.strip() for p in row.strip("|").split("|")]
                if len(parts) >= 2:
                    raw_name = parts[0].strip()
                    unit_m = re.search(r'\[([^\]]+)\]', raw_name)
                    unit = unit_m.group(1) if unit_m else ""
                    name = re.sub(r'\s*\[[^\]]+\]', '', raw_name).strip()
                    values: list[float] = []
                    for cell in parts[1:]:
                        try:
                            values.append(float(cell.replace(",", ".")))
                        except ValueError:
                            pass
                    if name and values:
                        results.append((name, values, unit))
                i += 1
        else:
            i += 1
    return results


def _normalize(name: str) -> str:
    key = name.lower()
    key = re.sub(r'[^a-z0-9]+', '_', key)
    return key.strip('_')


def _parse_and_emit(text: str, trace: "Trace") -> None:
    from ..core.events import CounterEvent, Category

    # Detect access errors before wasting time on parsing
    if ("Error" in text or "Cannot" in text) and len(text) < 300:
        import sys
        print(f"[hprofiler][likwid] counter collection issue:\n{text.strip()}",
              file=sys.stderr)
        return

    ts = trace.metadata.end_time_ns or time.monotonic_ns()
    metrics = _parse_metric_tables(text)

    for name, values, unit in metrics:
        key = _normalize(name)
        is_rate = any(x in unit for x in ("MFLOP/s", "MByte/s", "GB/s", "MHz"))

        # Per-core values
        for idx, val in enumerate(values):
            trace.add(CounterEvent(
                name=f"likwid.core{idx}.{key}",
                category=Category.CPU,
                timestamp_ns=ts,
                value=val,
                unit=unit,
                pid=0,
            ))

        # Aggregate: sum for rates/throughput, average for latency/CPI/clock
        if len(values) > 1:
            agg_val = sum(values) if is_rate else sum(values) / len(values)
            agg_label = "total" if is_rate else "avg"
            trace.add(CounterEvent(
                name=f"likwid.{agg_label}.{key}",
                category=Category.CPU,
                timestamp_ns=ts,
                value=agg_val,
                unit=unit,
                pid=0,
            ))
