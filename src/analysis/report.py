"""Rich terminal output and file writing for analysis reports."""

from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import AnalysisReport


def print_report(report: "AnalysisReport") -> None:
    """Render the analysis report to the terminal using Rich."""
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.text import Text

    console = Console()
    console.print()

    header = Text()
    header.append("hprofiler AI Performance Analysis\n", style="bold cyan")
    header.append(f"Model: {report.model}", style="dim")
    if report.turns_used > 1:
        header.append(f"  ·  {report.turns_used} turns", style="dim")
    if report.trace_summary:
        header.append(f"\n{report.trace_summary}", style="dim")
    console.print(Panel(header, border_style="cyan"))

    console.print(Markdown(report.content))
    console.print()


def save_report(report: "AnalysisReport", path: str) -> None:
    """Save the analysis report as a Markdown file."""
    content = (
        "# hprofiler Performance Analysis Report\n\n"
        f"**Model:** {report.model}  \n"
        f"**Trace:** {report.trace_summary}  \n"
        f"**Analysis turns:** {report.turns_used}\n\n"
        "---\n\n"
        f"{report.content}\n"
    )
    Path(path).write_text(content, encoding="utf-8")
    print(f"[hprofiler] Analysis report saved to: {path}")
