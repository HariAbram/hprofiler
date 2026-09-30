"""
Shared clipboard write, extracted from InspectorBridge.copyToClipboard so
the new tables' "copy row"/"copy all" actions (src/gui/tablemodel.py) reuse
one implementation instead of each bridge re-wrapping
QGuiApplication.clipboard() independently.
"""
from __future__ import annotations

from PySide6.QtGui import QGuiApplication


def copy_text(text: str) -> None:
    clipboard = QGuiApplication.clipboard()
    if clipboard is not None:
        clipboard.setText(text)
