"""Strict-JSON conversion helpers for experiment evidence."""

from __future__ import annotations

import math
from typing import Any


def json_safe(value: Any) -> Any:
    """Return a JSON-safe copy, replacing non-finite floats with null."""

    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value
