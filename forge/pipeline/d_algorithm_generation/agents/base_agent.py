"""Shared interface for the minimal Phase 1 multi-agent pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class AgentResult:
    success: bool
    data: Dict[str, Any]
    message: str = ""


class BaseAgent:
    name: str = "BaseAgent"

    def run(self, payload: Dict[str, Any]) -> AgentResult:
        raise NotImplementedError
