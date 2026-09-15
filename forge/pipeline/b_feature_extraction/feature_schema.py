"""Pydantic schema for benchmark problem features."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ProblemFeature(BaseModel):
    schema_version: str = "1.0"
    benchmark_id: str
    pde_type: str
    equation_order: int = Field(ge=1)
    spatial_dimension: int = Field(ge=1)
    time_dependent: bool
    has_initial_condition: bool = False
    boundary_type: list[str] = Field(default_factory=list)
    has_shock: bool = False
    has_multiscale: bool = False
    has_conservation_law: bool = False
    has_periodicity: bool = False
    variable_has_nonnegative_physics: bool = False
    forward_or_inverse: Literal["forward", "inverse"] = "forward"
    noise_level: float = Field(default=0.0, ge=0.0)
    geometry: str = "interval"

    def as_dict(self) -> dict:
        return self.model_dump()


class PhysicsProblemFeature(BaseModel):
    schema_version: str = "1.0"
    problem_id: str
    law_types: list[str]
    task_type: str
    time_dependent: bool
    differential_order: int | None = None
    spatial_dimension: int | None = None
    state_dimension: int | None = None
    is_stiff: bool = False
    is_multiscale: bool = False
    has_discontinuity: bool = False
    has_shock: bool = False
    has_conservation_law: bool = False
    has_constitutive_relation: bool = False
    has_algebraic_constraints: bool = False
    has_integral_constraints: bool = False
    has_inequality_constraints: bool = False
    has_symmetry: bool = False
    has_unknown_parameters: bool = False
    data_available: bool = False
    data_sparsity: str | None = None
    noise_level: float | None = None
    geometry_complexity: str | None = None
    constraint_types: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def as_search_dict(self) -> dict[str, Any]:
        """Map generic features to the keys consumed by search and retrieval."""
        profile = self.metadata.get("pinnacle_profile") or {}
        if not isinstance(profile, dict):
            profile = {}
        challenge_tags = list(
            dict.fromkeys(
                [
                    *(self.metadata.get("challenge_tags") or []),
                    *(profile.get("challenge_tags") or []),
                ]
            )
        )
        pde_family = (
            self.metadata.get("pde_family")
            or self.metadata.get("pde_type")
            or profile.get("family")
        )
        return {
            "benchmark_id": self.problem_id,
            "problem_id": self.problem_id,
            "pde_type": self.metadata.get("pde_type") or ",".join(self.law_types),
            "pde_family": pde_family,
            "equation_order": self.differential_order or 1,
            "spatial_dimension": self.spatial_dimension or 1,
            "state_dimension": self.state_dimension or 1,
            "time_dependent": self.time_dependent,
            "has_initial_condition": "initial" in self.constraint_types,
            "boundary_type": ["Dirichlet"] if "boundary" in self.constraint_types else [],
            "has_shock": self.has_shock,
            "has_multiscale": self.is_multiscale,
            "has_conservation_law": self.has_conservation_law,
            "has_periodicity": bool(
                self.metadata.get("has_periodicity", "periodic" in self.constraint_types)
            ),
            "is_stiff": self.is_stiff,
            "is_multiscale": self.is_multiscale,
            "has_discontinuity": self.has_discontinuity,
            "has_constitutive_relation": self.has_constitutive_relation,
            "has_algebraic_constraints": self.has_algebraic_constraints,
            "has_integral_constraints": self.has_integral_constraints,
            "has_inequality_constraints": self.has_inequality_constraints,
            "has_symmetry": self.has_symmetry,
            "variable_has_nonnegative_physics": "positivity" in self.constraint_types,
            "forward_or_inverse": "inverse" if self.task_type in {"inverse", "parameter_identification"} else "forward",
            "task_type": self.task_type,
            "noise_level": self.noise_level or 0.0,
            "geometry": self.metadata.get("geometry", self.geometry_complexity or "generic"),
            "law_types": self.law_types,
            "constraint_types": self.constraint_types,
            "data_available": self.data_available,
            "has_unknown_parameters": self.has_unknown_parameters,
            "has_complex_geometry": bool(self.metadata.get("has_complex_geometry", False)),
            "has_interface": bool(self.metadata.get("has_interface", False)),
            "has_piecewise_coefficient": bool(self.metadata.get("has_piecewise_coefficient", False)),
            "has_varying_coefficient": bool(self.metadata.get("has_varying_coefficient", False)),
            "has_oscillatory_solution": bool(self.metadata.get("has_oscillatory_solution", False)),
            "has_chaotic_dynamics": bool(self.metadata.get("has_chaotic_dynamics", False)),
            "has_long_time_horizon": bool(self.metadata.get("has_long_time_horizon", False)),
            "has_incompressibility": bool(self.metadata.get("has_incompressibility", False)),
            "has_coupled_fields": bool(self.metadata.get("has_coupled_fields", False)),
            "has_pressure_gauge": bool(self.metadata.get("has_pressure_gauge", False)),
            "has_variable_coefficients": bool(
                self.metadata.get(
                    "has_variable_coefficients",
                    self.metadata.get("has_varying_coefficient", False),
                )
            ),
            "has_conservation_form": bool(
                self.metadata.get("has_conservation_form", self.has_conservation_law)
            ),
            "has_positivity_requirement": bool(
                self.metadata.get(
                    "has_positivity_requirement",
                    "positivity" in self.constraint_types,
                )
            ),
            "has_observation_data": bool(
                self.metadata.get("has_observation_data", self.data_available)
            ),
            "equation_type": self.metadata.get("equation_type"),
            "num_outputs": int(self.metadata.get("num_outputs", self.state_dimension or 1)),
            "highest_derivative_order": int(
                self.metadata.get("highest_derivative_order", self.differential_order or 1)
            ),
            "has_noisy_observations": bool(self.metadata.get("has_noisy_observations", False)),
            "is_elliptic": bool(self.metadata.get("is_elliptic", False)),
            "elliptic": bool(self.metadata.get("elliptic", False)),
            "diffusion_dominated": bool(self.metadata.get("diffusion_dominated", False)),
            "convection_dominated": bool(self.metadata.get("convection_dominated", False)),
            "has_wave_propagation": bool(self.metadata.get("has_wave_propagation", False)),
            "challenge_tags": challenge_tags,
            "unknown_parameters": self.metadata.get("unknown_parameters", []),
            "pde_equation": self.metadata.get("pde_equation"),
            "pde_equation_id": self.metadata.get("pde_equation_id"),
            "equation_id": self.metadata.get("pde_equation_id"),
            "has_manufactured_source": self.metadata.get("has_manufactured_source"),
            "reference_solution_type": self.metadata.get("reference_solution_type"),
            "direct_literature_comparison": self.metadata.get("direct_literature_comparison"),
            "pinnacle_profile": self.metadata.get("pinnacle_profile"),
        }
