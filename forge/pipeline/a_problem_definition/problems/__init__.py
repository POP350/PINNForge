"""Generic PhysicsProblem layer."""

from .base import PDEProblem, PhysicsProblem
from .registry import get_problem, list_problems, register_problem
from .schemas import (
    ConstraintSpec,
    DomainSpec,
    GoverningLawSpec,
    ObservationSpec,
    PhysicalParameterSpec,
    PhysicsProblemSpec,
)

__all__ = [
    "PDEProblem",
    "PhysicsProblem",
    "PhysicsProblemSpec",
    "DomainSpec",
    "GoverningLawSpec",
    "ConstraintSpec",
    "ObservationSpec",
    "PhysicalParameterSpec",
    "get_problem",
    "list_problems",
    "register_problem",
]
