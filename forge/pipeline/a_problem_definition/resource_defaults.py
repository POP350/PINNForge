"""Unified problem-independent defaults for formal PDE experiments."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


UNIFIED_RESOURCE_DEFAULTS: dict[str, Any] = {
    "maximum_sampling_points": None,
    "evaluation_size": 2_500,
    "residual_evaluation_size": 2_048,
    "low_fidelity_iterations": 1_000,
    "final_iterations": 10_000,
}


def resolve_resource_defaults() -> dict[str, Any]:
    """Return a fresh copy of the one common resource configuration."""

    return deepcopy(UNIFIED_RESOURCE_DEFAULTS)


__all__ = ["UNIFIED_RESOURCE_DEFAULTS", "resolve_resource_defaults"]
