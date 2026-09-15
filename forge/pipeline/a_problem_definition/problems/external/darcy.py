"""Manufactured variable-coefficient Darcy flow problem."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from forge.pipeline.a_problem_definition.problems.capabilities import (
    ConstraintBatch,
    SinusoidalCoefficientField,
    coordinate_derivative,
)
from forge.pipeline.a_problem_definition.problems.schemas import (
    ConstraintSpec,
    DomainSpec,
    GoverningLawSpec,
    ObservationSpec,
    PhysicsProblemSpec,
)

from .base import ExternalPDEProblem


class DarcyFlow2D(ExternalPDEProblem):
    problem_id = "darcy_flow_2d"
    name = "Darcy Flow 2D"
    evaluation_shape = (48, 48)

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        values = dict(config or {})
        self.a0 = float(values.get("a0", 2.0))
        self.a1 = float(values.get("a1", 0.5))
        self.kx = float(values.get("kx", 2.0))
        self.ky = float(values.get("ky", 3.0))
        self.coefficient = SinusoidalCoefficientField(
            self.a0, self.a1, (self.kx, self.ky)
        )
        super().__init__(values)

    def get_spec(self) -> PhysicsProblemSpec:
        return PhysicsProblemSpec(
            schema_version="1.0",
            problem_id=self.problem_id,
            name=self.name,
            law_types=["pde"],
            task_type="forward",
            input_variables=["x", "y"],
            output_variables=["u"],
            domain=DomainSpec(
                variables=["x", "y"],
                bounds={"x": (0.0, 1.0), "y": (0.0, 1.0)},
                geometry="unit_square",
                metadata={"variable_roles": {"x": "spatial", "y": "spatial"}},
            ),
            governing_laws=[
                GoverningLawSpec(
                    law_id="darcy_conservation",
                    name="Variable-coefficient Darcy equation",
                    law_type="pde",
                    variables=["x", "y", "u"],
                    parameters=["a0", "a1", "kx", "ky"],
                    expression="-d_x(a*u_x)-d_y(a*u_y)-f=0",
                    differential_order=2,
                    metadata={
                        "required_derivatives": ["u_x", "u_y", "(a*u_x)_x", "(a*u_y)_y"],
                        "coefficient_capability": "spatial_coefficient_field",
                    },
                )
            ],
            constraints=[
                ConstraintSpec(
                    constraint_id="boundary",
                    name="Homogeneous Dirichlet boundary",
                    constraint_type="boundary",
                    target="boundary",
                    expression="u=0 on boundary",
                )
            ],
            observations=ObservationSpec(available=False),
            metadata={
                "problem_version": "1.0",
                "source": "manufactured solution",
                "spatial_dimension": 2,
                "time_dependent": False,
                "pde_type": "variable_coefficient_elliptic",
                "pde_family": "darcy",
                "equation_type": "elliptic",
                "num_outputs": 1,
                "highest_derivative_order": 2,
                "is_elliptic": True,
                "elliptic": True,
                "has_variable_coefficients": True,
                "has_varying_coefficient": True,
                "has_heterogeneous_medium": True,
                "has_complex_geometry": False,
                "has_observation_data": False,
                "challenge_tags": [
                    "elliptic",
                    "variable coefficient",
                    "heterogeneous medium",
                    "second-order derivatives",
                    "scalar output",
                ],
                "parameters": {"a0": self.a0, "a1": self.a1, "kx": self.kx, "ky": self.ky},
                "coefficient_field": "a0+a1*sin(kx*pi*x)*sin(ky*pi*y)",
                "reference_solution_type": "manufactured analytic",
                "reference_solution": "sin(pi*x)*sin(pi*y)",
                "evaluation_grid": list(self.evaluation_shape),
                "metric_providers": [
                    "per_output_error",
                    "residual_components",
                    "constraint_components",
                ],
                "case_study": False,
                "stability_guidance": {
                    "preferred_constraint_enforcement": {
                        "method": "problem_hard",
                        "transform_id": "darcy_zero_dirichlet",
                        "reason": "Remove boundary/PDE gradient competition for the homogeneous Dirichlet problem.",
                    },
                    "network": {
                        "preferred_activations": ["tanh", "silu"],
                        "avoid_for_default_search": ["relu", "high_frequency_sine"],
                    },
                    "optimization": {
                        "adam_learning_rate_max": 0.001,
                        "scheduler_horizon": "match_adam_phase_iterations",
                        "lbfgs_policy": "use a short terminal phase or omit it when rollback evidence shows degradation",
                    },
                    "loss_weighting": {
                        "preferred": ["fixed", "normalized", "inverse_magnitude"],
                        "lra_max_weight_recommended": 50.0,
                    },
                },
            },
        )

    def reference_solution(self, samples: torch.Tensor) -> torch.Tensor:
        x, y = samples[:, 0:1], samples[:, 1:2]
        return torch.sin(torch.pi * x) * torch.sin(torch.pi * y)

    def constraint_enforcement_capabilities(self) -> Mapping[str, Any]:
        return {
            "soft_penalty": {"transform_ids": ["none"]},
            "problem_hard": {
                "transform_ids": ["darcy_zero_dirichlet"],
                "parameters": {
                    "darcy_zero_dirichlet": {"envelope_scale": 16.0}
                },
            },
        }

    def build_constraint_output_transform(
        self, transform_id: str, parameters: Mapping[str, Any] | None = None
    ):
        if transform_id != "darcy_zero_dirichlet":
            return super().build_constraint_output_transform(transform_id, parameters)
        values = dict(parameters or {})
        unsupported = set(values) - {"envelope_scale"}
        if unsupported:
            raise ValueError(
                "darcy_zero_dirichlet only accepts envelope_scale; unsupported="
                f"{sorted(unsupported)}"
            )
        envelope_scale = float(values.get("envelope_scale", 16.0))
        if not envelope_scale > 0.0:
            raise ValueError("darcy_zero_dirichlet envelope_scale must be positive")

        def transform(
            inputs: torch.Tensor, raw_outputs: torch.Tensor
        ) -> torch.Tensor:
            x, y = inputs[:, 0:1], inputs[:, 1:2]
            envelope = envelope_scale * x * (1.0 - x) * y * (1.0 - y)
            return envelope * raw_outputs

        return transform

    def source(self, samples: torch.Tensor) -> torch.Tensor:
        x, y = samples[:, 0:1], samples[:, 1:2]
        u = self.reference_solution(samples)
        u_x = torch.pi * torch.cos(torch.pi * x) * torch.sin(torch.pi * y)
        u_y = torch.pi * torch.sin(torch.pi * x) * torch.cos(torch.pi * y)
        laplacian_u = -2.0 * torch.pi**2 * u
        a = self.coefficient(samples)
        a_x = (
            self.a1
            * self.kx
            * torch.pi
            * torch.cos(self.kx * torch.pi * x)
            * torch.sin(self.ky * torch.pi * y)
        )
        a_y = (
            self.a1
            * self.ky
            * torch.pi
            * torch.sin(self.kx * torch.pi * x)
            * torch.cos(self.ky * torch.pi * y)
        )
        return -(a_x * u_x + a_y * u_y + a * laplacian_u)

    def sample_constraints(
        self,
        n: int,
        *,
        category: str | None = None,
        seed: int | None = None,
        device: str | None = None,
    ) -> dict[str, ConstraintBatch]:
        if not self.matching_category(category, ("boundary",)):
            return {}
        points = self.sample_box_boundary(n, seed=seed, device=device or "cpu")
        return {
            "boundary": ConstraintBatch(
                "boundary", "boundary", points, target=torch.zeros((int(n), 1), device=points.device)
            )
        }

    def compute_governing_residuals(
        self,
        model: Any,
        samples: torch.Tensor,
        *,
        create_graph: bool = True,
    ) -> dict[str, torch.Tensor]:
        u = model(samples)[:, :1]
        u_x = coordinate_derivative(u, samples, 0, create_graph=True)
        u_y = coordinate_derivative(u, samples, 1, create_graph=True)
        coefficient = self.coefficient(samples)
        flux_x = coordinate_derivative(
            coefficient * u_x, samples, 0, create_graph=create_graph
        )
        flux_y = coordinate_derivative(
            coefficient * u_y, samples, 1, create_graph=create_graph
        )
        return {"darcy_conservation": -flux_x - flux_y - self.source(samples)}

    def compute_constraint_residuals(
        self,
        model: Any,
        samples: Mapping[str, Any],
        *,
        create_graph: bool = True,
    ) -> dict[str, torch.Tensor]:
        if "boundary" not in samples:
            return {}
        batch = samples["boundary"]
        points = self.constraint_points(batch)
        target = batch.target_values() if isinstance(batch, ConstraintBatch) else torch.zeros_like(points[:, :1])
        return {"boundary": model(points)[:, :1] - target.to(points)}
