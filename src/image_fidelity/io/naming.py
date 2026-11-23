"""Filesystem-safe names for result files and directories."""

from __future__ import annotations

import re
from typing import Any


def safe_name(value: Any, default: str = "item") -> str:
    """Replace characters outside ``[A-Za-z0-9_.-]`` with underscores."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._") or default
