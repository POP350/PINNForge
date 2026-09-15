"""Append-only adaptive-control audit logging."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any


class AdaptiveAuditLogger:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self.records: list[dict[str, Any]] = []

    def bind(self, path: str | Path | None) -> None:
        self.path = Path(path) if path is not None else None

    def emit(self, record: dict[str, Any]) -> dict[str, Any]:
        item = deepcopy(record)
        self.records.append(item)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
        return item

    def state_dict(self) -> dict[str, Any]:
        return {"records": deepcopy(self.records)}

    def load_state_dict(self, state: dict[str, Any] | None) -> None:
        self.records = list((state or {}).get("records") or [])

