"""Schemas for PDE-oriented physics-informed learning problems."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FrozenProblemModel(BaseModel):
    model_config = ConfigDict(frozen=True)


TaskType = Literal["forward", "inverse", "parameter_identification", "data_assimilation", "surrogate_modeling"]
LawType = Literal["pde", "ode", "dae", "algebraic", "integral", "constitutive", "conservation", "empirical"]
ConstraintType = Literal[
    "boundary",
    "initial",
    "periodic",
    "interface",
    "conservation",
    "positivity",
    "inequality",
    "symmetry",
    "monotonicity",
    "constitutive",
    "parameter_bound",
    "algebraic_consistency",
]


class DomainSpec(FrozenProblemModel):
    variables: list[str] = Field(default_factory=list)
    bounds: dict[str, tuple[float, float]] = Field(default_factory=dict)
    geometry: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class GoverningLawSpec(FrozenProblemModel):
    law_id: str
    name: str
    law_type: LawType | str
    variables: list[str]
    parameters: list[str] = Field(default_factory=list)
    expression: str | None = None
    differential_order: int | None = None
    required: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConstraintSpec(FrozenProblemModel):
    constraint_id: str = Field(min_length=1)
    name: str
    constraint_type: ConstraintType | str
    target: str | None = None
    expression: str | None = None
    enforcement_options: list[str] = Field(default_factory=lambda: ["soft"])
    required: bool = True
    derivative_variable: str | None = None
    derivative_order: int | None = Field(default=None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ObservationSpec(FrozenProblemModel):
    available: bool = False
    variables: list[str] = Field(default_factory=list)
    noise_level: float | None = None
    sparse: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class PhysicalParameterSpec(FrozenProblemModel):
    name: str
    trainable: bool
    initial_value: float | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None
    unit: str | None = None


class PhysicsProblemSpec(FrozenProblemModel):
    schema_version: str = "1.0"
    problem_id: str
    name: str
    law_types: list[str]
    task_type: TaskType | str
    input_variables: list[str]
    output_variables: list[str]
    domain: DomainSpec
    governing_laws: list[GoverningLawSpec]
    constraints: list[ConstraintSpec]
    observations: ObservationSpec | None = None
    unknown_parameters: list[PhysicalParameterSpec] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_constraint_ids(self) -> "PhysicsProblemSpec":
        constraint_ids = [item.constraint_id.strip() for item in self.constraints]
        if any(not constraint_id for constraint_id in constraint_ids):
            raise ValueError("constraint_id must not be blank")
        if len(constraint_ids) != len(set(constraint_ids)):
            raise ValueError("constraint_id values must be unique within a problem")
        return self


def migrate_problem_spec(payload: dict | PhysicsProblemSpec) -> PhysicsProblemSpec:
    """Read legacy or current problem specs into the frozen v1.0 schema."""
    if isinstance(payload, PhysicsProblemSpec):
        return payload
    migrated = dict(payload)
    migrated.setdefault("schema_version", "1.0")
    migrated.setdefault("observations", None)
    migrated.setdefault("unknown_parameters", [])
    migrated.setdefault("metadata", {})
    return PhysicsProblemSpec(**migrated)
