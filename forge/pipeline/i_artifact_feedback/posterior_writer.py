"""Append-only writer for measured validation, training, and evaluation evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from forge.utils.json_safety import json_safe


class PosteriorWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(json_safe(record), ensure_ascii=False, allow_nan=False, default=str) + "\n")
