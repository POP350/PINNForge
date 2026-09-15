"""Shared types for the generic physics engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PhysicsBatch:
    domain_samples: Any
    constraint_samples: dict[str, Any]
    observation_samples: Any | None = None
