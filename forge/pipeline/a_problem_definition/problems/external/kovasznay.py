"""Kovasznay's exact steady incompressible-flow problem."""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch

from forge.pipeline.a_problem_definition.problems.capabilities import (
    ConstraintBatch,
    coordinate_derivative,
    pressure_gauge_constraint,
)
from forge.pipeline.a_problem_definition.problems.schemas import (
    ConstraintSpec,
    DomainSpec,
    GoverningLawSpec,
    ObservationSpec,
    PhysicsProblemSpec,
)

from .base import ExternalPDEProblem


class KovasznayFlow2D(ExternalPDEProblem):
    problem_id = "kovasznay_flow_2d"
    name = "Kovasznay Flow 2D"
    evaluation_shape = (40, 40)

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        values = dict(config or {})
        self.reynolds = float(values.get("reynolds", 40.0))
        self.viscosity = float(values.get("viscosity", 1.0 / self.reynolds))
        self.lambda_value = self.reynolds / 2.0 - math.sqrt(
            self.reynolds**2 / 4.0 + 4.0 * math.pi**2
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
            output_variables=["u", "v", "p"],
            domain=DomainSpec(
                variables=["x", "y"],
                bounds={"x": (-0.5, 1.0), "y": (-0.5, 1.5)},
                geometry="rectangle",
                metadata={"variable_roles": {"x": "spatial", "y": "spatial"}},
            ),
            governing_laws=[
                GoverningLawSpec(
                    law_id="continuity",
                    name="Incompressible continuity",
                    law_type="pde",
                    variables=["x", "y", "u", "v"],
                    expression="u_x + v_y = 0",
                    differential_order=1,
                    metadata={"required_derivatives": ["u_x", "v_y"]},
                ),
                GoverningLawSpec(
                    law_id="x_momentum",
                    name="Steady x momentum",
                    law_type="pde",
                    variables=["x", "y", "u", "v", "p"],
                    parameters=["viscosity"],
                    expression="u*u_x + v*u_y + p_x - nu*(u_xx + u_yy) = 0",
                    differential_order=2,
                    metadata={"required_derivatives": ["u_x", "u_y", "u_xx", "u_yy", "p_x"]},
                ),
                GoverningLawSpec(
                    law_id="y_momentum",
                    name="Steady y momentum",
                    law_type="pde",
                    variables=["x", "y", "u", "v", "p"],
                    parameters=["viscosity"],
                    expression="u*v_x + v*v_y + p_y - nu*(v_xx + v_yy) = 0",
                    differential_order=2,
                    metadata={"required_derivatives": ["v_x", "v_y", "v_xx", "v_yy", "p_y"]},
                ),
            ],
            constraints=[
                ConstraintSpec(
                    constraint_id="boundary",
                    name="Analytic velocity and pressure boundary",
                    constraint_type="boundary",
                    target="boundary",
                    expression="(u,v,p)=(u_exact,v_exact,p_exact) on boundary",
                ),
                ConstraintSpec(
                    constraint_id="pressure_gauge",
                    name="Pressure anchor",
                    constraint_type="algebraic_consistency",
                    target="pressure_gauge",
                    expression="p(0,0)=0",
                    metadata={"gauge_mode": "anchor", "pressure_index": 2},
                ),
            ],
            observations=ObservationSpec(available=False),
            metadata={
                "problem_version": "1.0",
                "source": "analytic Kovasznay solution",
                "spatial_dimension": 2,
                "time_dependent": False,
                "pde_type": "steady_incompressible_flow",
                "pde_family": "navier_stokes",
                "equation_type": "steady_incompressible_flow",
                "num_outputs": 3,
                "highest_derivative_order": 2,
                "has_incompressibility": True,
                "has_coupled_fields": True,
                "has_pressure_gauge": True,
                "has_complex_geometry": False,
                "has_observation_data": False,
                "challenge_tags": [
                    "steady incompressible flow",
                    "multi-output",
                    "pressure gauge",
                    "coupled residuals",
                    "second-order velocity derivatives",
                ],
                "reference_solution_type": "analytic",
                "reference_solution": "standard Kovasznay solution",
                "evaluation_grid": list(self.evaluation_shape),
                "metric_providers": [
                    "per_output_error",
                    "residual_components",
                    "constraint_components",
                    "velocity_pressure_flow",
                ],
                "metric_provider_options": {
                    "velocity_pressure_flow": {
                        "velocity_indices": [0, 1],
                        "pressure_index": 2,
                    }
                },
                "case_study": False,
            },
        )

    def reference_solution(self, samples: torch.Tensor) -> torch.Tensor:
        x = samples[:, 0:1]
        y = samples[:, 1:2]
        exponential = torch.exp(self.lambda_value * x)
        u = 1.0 - exponential * torch.cos(2.0 * torch.pi * y)
        v = (
            self.lambda_value
            / (2.0 * torch.pi)
            * exponential
            * torch.sin(2.0 * torch.pi * y)
        )
        p = 0.5 * (1.0 - torch.exp(2.0 * self.lambda_value * x))
        return torch.cat((u, v, p), dim=1)

    def sample_constraints(
        self,
        n: int,
        *,
        category: str | None = None,
        seed: int | None = None,
        device: str | None = None,
    ) -> dict[str, ConstraintBatch]:
        result: dict[str, ConstraintBatch] = {}
        target_device = device or "cpu"
        if self.matching_category(category, ("boundary",)):
            points = self.sample_box_boundary(n, seed=seed, device=target_device)
            result["boundary"] = ConstraintBatch(
                "boundary", "boundary", points, target=self.reference_solution
            )
        if self.matching_category(category, ("pressure_gauge", "algebraic_consistency", "custom_constraint_domain")):
            point = torch.tensor([[0.0, 0.0]], device=target_device)
            result["pressure_gauge"] = ConstraintBatch(
                "pressure_gauge", "pressure_gauge", point,
                metadata={"pressure_index": 2, "mode": "anchor", "target": 0.0},
            )
        return result

    def compute_governing_residuals(
        self,
        model: Any,
        samples: torch.Tensor,
        *,
        create_graph: bool = True,
    ) -> dict[str, torch.Tensor]:
        prediction = model(samples)
        u, v, p = (prediction[:, index : index + 1] for index in range(3))
        u_x = coordinate_derivative(u, samples, 0, create_graph=True)
        u_y = coordinate_derivative(u, samples, 1, create_graph=True)
        v_x = coordinate_derivative(v, samples, 0, create_graph=True)
        v_y = coordinate_derivative(v, samples, 1, create_graph=True)
        p_x = coordinate_derivative(p, samples, 0, create_graph=create_graph)
        p_y = coordinate_derivative(p, samples, 1, create_graph=create_graph)
        u_xx = coordinate_derivative(u_x, samples, 0, create_graph=create_graph)
        u_yy = coordinate_derivative(u_y, samples, 1, create_graph=create_graph)
        v_xx = coordinate_derivative(v_x, samples, 0, create_graph=create_graph)
        v_yy = coordinate_derivative(v_y, samples, 1, create_graph=create_graph)
        return {
            "continuity": u_x + v_y,
            "x_momentum": u * u_x + v * u_y + p_x - self.viscosity * (u_xx + u_yy),
            "y_momentum": u * v_x + v * v_y + p_y - self.viscosity * (v_xx + v_yy),
        }

    def compute_constraint_residuals(
        self,
        model: Any,
        samples: Mapping[str, Any],
        *,
        create_graph: bool = True,
    ) -> dict[str, torch.Tensor]:
        residuals: dict[str, torch.Tensor] = {}
        if "boundary" in samples:
            batch = samples["boundary"]
            prediction = model(self.constraint_points(batch))
            target = batch.target_values() if isinstance(batch, ConstraintBatch) else None
            if target is None:
                target = self.reference_solution(self.constraint_points(batch))
            residuals["boundary"] = prediction - target.to(prediction)
        if "pressure_gauge" in samples:
            points = self.constraint_points(samples["pressure_gauge"])
            residuals["pressure_gauge"] = pressure_gauge_constraint(
                model(points), pressure_index=2, target=0.0, mode="anchor"
            )
        return residuals
