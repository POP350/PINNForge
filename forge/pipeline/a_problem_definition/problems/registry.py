"""Registry preloaded from PINNacle and open to external problem providers."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from forge.pipeline.c_knowledge_retrieval.pinnacle_registry import PINNACLE_PDE_PROFILES
from forge.pipeline.a_problem_definition.problems.base import PhysicsProblem
from forge.pipeline.a_problem_definition.problems.adapters.pinnacle_adapter import make_pinnacle_problem_class
from forge.pipeline.a_problem_definition.problems.external import AllenCahn1D, DarcyFlow2D, KovasznayFlow2D, ShallowWater2D

PINNACLE_PROBLEMS: dict[str, type[PhysicsProblem]] = {
    problem_id: make_pinnacle_problem_class(problem_id)
    for problem_id in PINNACLE_PDE_PROFILES
}
PROBLEMS: dict[str, type[PhysicsProblem]] = dict(PINNACLE_PROBLEMS)
PROBLEMS[KovasznayFlow2D.problem_id] = KovasznayFlow2D
PROBLEMS[AllenCahn1D.problem_id] = AllenCahn1D
PROBLEMS[DarcyFlow2D.problem_id] = DarcyFlow2D
PROBLEMS[ShallowWater2D.problem_id] = ShallowWater2D
_PROBLEM_CONFIGURATIONS: dict[str, dict[str, Any]] = {}

_PROBLEM_ALIASES: dict[str, str] = {key.lower(): key for key in PROBLEMS}
for canonical, profile in PINNACLE_PDE_PROFILES.items():
    for alias in [profile["pinnacle_name"], *profile["aliases"]]:
        _PROBLEM_ALIASES.setdefault(str(alias).lower(), canonical)


def register_problem(problem_id: str, problem_cls: type[PhysicsProblem]) -> None:
    PROBLEMS[problem_id] = problem_cls
    _PROBLEM_ALIASES[str(problem_id).lower()] = problem_id


def get_problem(problem_id: str) -> PhysicsProblem:
    resolved = problem_id if problem_id in PROBLEMS else _PROBLEM_ALIASES.get(str(problem_id).lower())
    if resolved not in PROBLEMS:
        known = ", ".join(sorted(PROBLEMS))
        raise KeyError(f"Unknown problem '{problem_id}'. Known: {known}")
    configuration = deepcopy(_PROBLEM_CONFIGURATIONS.get(resolved) or {})
    return PROBLEMS[resolved](configuration) if configuration else PROBLEMS[resolved]()


def configure_problem(problem_id: str, configuration: Mapping[str, Any] | None) -> None:
    """Configure any registered provider without adding controller branches."""

    resolved = problem_id if problem_id in PROBLEMS else _PROBLEM_ALIASES.get(str(problem_id).lower())
    if resolved not in PROBLEMS:
        raise KeyError(f"Unknown problem '{problem_id}'")
    values = deepcopy(dict(configuration or {}))
    if values:
        _PROBLEM_CONFIGURATIONS[resolved] = values
    else:
        _PROBLEM_CONFIGURATIONS.pop(resolved, None)


def clear_problem_configuration(problem_id: str) -> None:
    resolved = problem_id if problem_id in PROBLEMS else _PROBLEM_ALIASES.get(str(problem_id).lower())
    if resolved is not None:
        _PROBLEM_CONFIGURATIONS.pop(resolved, None)


def list_problems() -> list[str]:
    return sorted(PROBLEMS)


__all__ = [
    "get_problem",
    "list_problems",
    "register_problem",
    "configure_problem",
    "clear_problem_configuration",
    "PINNACLE_PROBLEMS",
]
