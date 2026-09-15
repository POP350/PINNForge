"""Local environment loading for developer-only secrets.

This intentionally supports a tiny `.env.local` subset instead of adding a
runtime dependency. Values are loaded only when the variable is not already set.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_local_env(path: str | Path | None = None) -> dict[str, str]:
    env_path = Path(path) if path is not None else _project_root() / ".env.local"
    loaded: dict[str, str] = {}
    if not env_path.exists():
        return loaded
    for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = _unquote(value.strip())
        if not key or key in os.environ:
            continue
        os.environ[key] = value
        loaded[key] = value
    return loaded


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value
