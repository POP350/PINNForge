"""Configurable one-dimensional Allen--Cahn interface problem."""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch

from forge.pipeline.a_problem_definition.problems.capabilities import (
    ConstraintBatch,
    coordinate_derivative,
    periodic_constraint_residual,
)
from forge.pipeline.a_problem_definition.problems.schemas import (
    ConstraintSpec,
    DomainSpec,
    GoverningLawSpec,
    ObservationSpec,
    PhysicsProblemSpec,
)

from .base import ExternalPDEProblem


class AllenCahn1D(ExternalPDEProblem):
    problem_id = "allen_cahn_1d"
    name = "Allen-Cahn 1D"
    evaluation_shape = (128, 16)

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        values = dict(config or {})
        self.epsilon = float(values.get("epsilon", 0.1))
        self.x_bounds = tuple(values.get("x_bounds", (-1.0, 1.0)))
        self.t_bounds = tuple(values.get("t_bounds", (0.0, 0.25)))
        self.boundary_type = str(values.get("boundary_type", "dirichlet")).lower()
        if self.epsilon <= 0.0:
            raise ValueError("Allen-Cahn epsilon must be positive")
        if self.boundary_type not in {"dirichlet", "periodic"}:
            raise ValueError("Allen-Cahn boundary_type must be dirichlet or periodic")
        super().__init__(values)

    def get_spec(self) -> PhysicsProblemSpec:
        boundary_constraints = (
            [
                ConstraintSpec(
                    constraint_id="boundary",
                    name="Dirichlet boundary",
                    constraint_type="boundary",
                    target="boundary",
                    expression="u=u_exact on x=x_min,x_max",
                )
            ]
            if self.boundary_type == "dirichlet"
            else [
                ConstraintSpec(
                    constraint_id="periodic_value",
                    name="Periodic value",
                    constraint_type="periodic",
                    target="periodic",
                    expression="u(x_min,t)=u(x_max,t)",
                ),
                ConstraintSpec(
                    constraint_id="periodic_derivative",
                    name="Periodic first derivative",
                    constraint_type="periodic",
                    target="periodic",
                    expression="u_x(x_min,t)=u_x(x_max,t)",
                    derivative_variable="x",
                    derivative_order=1,
                ),
            ]
        )
        return PhysicsProblemSpec(
            schema_version="1.0",
            problem_id=self.problem_id,
            name=self.name,
            law_types=["pde"],
            task_type="forward",
            input_variables=["x", "t"],
            output_variables=["u"],
            domain=DomainSpec(
                variables=["x", "t"],
                bounds={"x": self.x_bounds, "t": self.t_bounds},
                geometry="space_time_interval",
                metadata={
                    "variable_roles": {
                        "x": {"role": "spatial"},
                        "t": {"role": "temporal", "causal_ordered": True},
                    }
                },
            ),
            governing_laws=[
                GoverningLawSpec(
                    law_id="allen_cahn",
                    name="Allen-Cahn equation",
                    law_type="pde",
                    variables=["x", "t", "u"],
                    parameters=["epsilon"],
                    expression="u_t - epsilon^2*u_xx + u^3 - u = 0",
                    differential_order=2,
                    metadata={"required_derivatives": ["u_t", "u_xx"]},
                )
            ],
            constraints=[
                ConstraintSpec(
                    constraint_id="initial",
                    name="Initial interface",
                    constraint_type="initial",
                    target="initial",
                    expression="u(x,0)=tanh(x/(sqrt(2)*epsilon))",
                ),
                *boundary_constraints,
            ],
            observations=ObservationSpec(available=False),
            metadata={
                "problem_version": "1.0",
                "source": "stationary manufactured interface solution",
                "spatial_dimension": 1,
                "time_dependent": True,
                "pde_type": "nonlinear_parabolic",
                "pde_family": "reaction_diffusion",
                "equation_type": "nonlinear_parabolic",
                "num_outputs": 1,
                "highest_derivative_order": 2,
                "is_stiff": True,
                "has_interface": True,
                "has_periodicity": self.boundary_type == "periodic",
                "has_long_time_horizon": False,
                "has_long_time_sensitivity": True,
                "has_complex_geometry": False,
                "has_observation_data": False,
                "challenge_tags": [
                    "nonlinear parabolic",
                    "stiff reaction",
                    "interface dynamics",
                    "second-order derivative",
                    "long-time sensitivity",
                ],
                "parameters": {"epsilon": self.epsilon},
                "boundary_mode": self.boundary_type,
                "reference_solution_type": "analytic stationary interface",
                "reference_solution": "tanh(x/(sqrt(2)*epsilon))",
                "evaluation_grid": list(self.evaluation_shape),
                "metric_providers": [
                    "per_output_error",
                    "residual_components",
                    "constraint_components",
                    "interface_region_error",
                ],
                "metric_provider_options": {
                    "interface_region_error": {"threshold": 0.5}
                },
                "case_study": False,
            },
        )

    def reference_solution(self, samples: torch.Tensor) -> torch.Tensor:
        return torch.tanh(samples[:, 0:1] / (math.sqrt(2.0) * self.epsilon))

    def sample_constraints(
        self,
        n: int,
        *,
        category: str | None = None,
        seed: int | None = None,
        device: str | None = None,
    ) -> dict[str, ConstraintBatch]:
        target_device = device or "cpu"
        generator = self._generator(seed)
        result: dict[str, ConstraintBatch] = {}
        if self.matching_category(category, ("initial", "initial_surface")):
            x = self.x_bounds[0] + torch.rand((int(n), 1), generator=generator) * (
                self.x_bounds[1] - self.x_bounds[0]
            )
            t = torch.full_like(x, self.t_bounds[0])
            points = torch.cat((x, t), dim=1).to(target_device)
            result["initial"] = ConstraintBatch(
                "initial", "initial", points, target=self.reference_solution
            )
        if self.matching_category(category, ("boundary", "periodic")):
            t = self.t_bounds[0] + torch.rand((int(n), 1), generator=generator) * (
                self.t_bounds[1] - self.t_bounds[0]
            )
            sides = torch.randint(0, 2, (int(n), 1), generator=generator)
            x = torch.where(
                sides == 0,
                torch.full_like(t, self.x_bounds[0]),
                torch.full_like(t, self.x_bounds[1]),
            )
            points = torch.cat((x, t), dim=1).to(target_device)
            if self.boundary_type == "dirichlet":
                result["boundary"] = ConstraintBatch(
                    "boundary", "boundary", points, target=self.reference_solution
                )
            else:
                left = torch.cat((torch.full_like(t, self.x_bounds[0]), t), dim=1).to(target_device)
                right = torch.cat((torch.full_like(t, self.x_bounds[1]), t), dim=1).to(target_device)
                result["periodic_value"] = ConstraintBatch(
                    "periodic_value", "periodic", left, paired_points=right
                )
                result["periodic_derivative"] = ConstraintBatch(
                    "periodic_derivative", "periodic", left, paired_points=right,
                    metadata={"derivative_axis": 0},
                )
        return result

    def compute_governing_residuals(
        self,
        model: Any,
        samples: torch.Tensor,
        *,
        create_graph: bool = True,
    ) -> dict[str, torch.Tensor]:
        u = model(samples)[:, :1]
        u_t = coordinate_derivative(u, samples, 1, create_graph=create_graph)
        u_x = coordinate_derivative(u, samples, 0, create_graph=True)
        u_xx = coordinate_derivative(u_x, samples, 0, create_graph=create_graph)
        return {"allen_cahn": u_t - self.epsilon**2 * u_xx + u**3 - u}

    def compute_constraint_residuals(
        self,
        model: Any,
        samples: Mapping[str, Any],
        *,
        create_graph: bool = True,
    ) -> dict[str, torch.Tensor]:
        residuals: dict[str, torch.Tensor] = {}
        for name in ("initial", "boundary"):
            if name not in samples:
                continue
            batch = samples[name]
            points = self.constraint_points(batch)
            target = batch.target_values() if isinstance(batch, ConstraintBatch) else self.reference_solution(points)
            residuals[name] = model(points)[:, :1] - target.to(points)
        if "periodic_value" in samples:
            residuals["periodic_value"] = periodic_constraint_residual(
                model, samples["periodic_value"], output_indices=(0,), create_graph=create_graph
            )
        if "periodic_derivative" in samples:
            residuals["periodic_derivative"] = periodic_constraint_residual(
                model,
                samples["periodic_derivative"],
                output_indices=(0,),
                derivative_axis=0,
                create_graph=create_graph,
            )
        return residuals
