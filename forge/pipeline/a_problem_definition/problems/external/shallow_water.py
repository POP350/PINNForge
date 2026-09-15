"""Smooth short-time periodic two-dimensional shallow-water system."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from forge.pipeline.a_problem_definition.problems.capabilities import (
    ConstraintBatch,
    conservative_system_residual,
    periodic_constraint_residual,
    positive_output_transform,
)
from forge.pipeline.a_problem_definition.problems.schemas import (
    ConstraintSpec,
    DomainSpec,
    GoverningLawSpec,
    ObservationSpec,
    PhysicsProblemSpec,
)

from .base import ExternalPDEProblem


class ShallowWater2D(ExternalPDEProblem):
    problem_id = "shallow_water_2d"
    name = "Shallow Water 2D"
    evaluation_shape = (20, 20, 8)

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        values = dict(config or {})
        self.gravity = float(values.get("gravity", 9.81))
        self.minimum_depth = float(values.get("minimum_depth", 1.0e-3))
        self.initial_state = tuple(values.get("initial_state", (1.0, 0.1, -0.05)))
        self.t_bounds = tuple(values.get("t_bounds", (0.0, 0.1)))
        self.periodic_derivatives = bool(values.get("periodic_derivatives", False))
        if self.minimum_depth <= 0.0 or self.initial_state[0] <= self.minimum_depth:
            raise ValueError("shallow-water depth must remain above minimum_depth")
        super().__init__(values)

    def get_spec(self) -> PhysicsProblemSpec:
        periodic = []
        for axis in ("x", "y"):
            periodic.append(
                ConstraintSpec(
                    constraint_id=f"periodic_{axis}_value",
                    name=f"Periodic values across {axis}",
                    constraint_type="periodic",
                    target="periodic",
                    expression=f"(h,qx,qy) periodic in {axis}",
                )
            )
            if self.periodic_derivatives:
                periodic.append(
                    ConstraintSpec(
                        constraint_id=f"periodic_{axis}_derivative",
                        name=f"Periodic first derivatives across {axis}",
                        constraint_type="periodic",
                        target="periodic",
                        derivative_variable=axis,
                        derivative_order=1,
                    )
                )
        equations = [
            ("mass", "h_t + (qx)_x + (qy)_y = 0"),
            ("x_momentum", "qx_t + (qx^2/h + g*h^2/2)_x + (qx*qy/h)_y = 0"),
            ("y_momentum", "qy_t + (qx*qy/h)_x + (qy^2/h + g*h^2/2)_y = 0"),
        ]
        return PhysicsProblemSpec(
            schema_version="1.0",
            problem_id=self.problem_id,
            name=self.name,
            law_types=["pde", "conservation"],
            task_type="forward",
            input_variables=["x", "y", "t"],
            output_variables=["h", "q_x", "q_y"],
            domain=DomainSpec(
                variables=["x", "y", "t"],
                bounds={"x": (0.0, 1.0), "y": (0.0, 1.0), "t": self.t_bounds},
                geometry="periodic_space_time_box",
                metadata={
                    "variable_roles": {
                        "x": {"role": "spatial"},
                        "y": {"role": "spatial"},
                        "t": {"role": "temporal", "causal_ordered": True},
                    }
                },
            ),
            governing_laws=[
                GoverningLawSpec(
                    law_id=law_id,
                    name=law_id.replace("_", " ").title(),
                    law_type="conservation",
                    variables=["x", "y", "t", "h", "q_x", "q_y"],
                    parameters=["gravity", "minimum_depth"],
                    expression=expression,
                    differential_order=1,
                    metadata={"residual_capability": "conservative_system_residual"},
                )
                for law_id, expression in equations
            ],
            constraints=[
                ConstraintSpec(
                    constraint_id="initial",
                    name="Smooth positive initial state",
                    constraint_type="initial",
                    target="initial",
                    expression="(h,qx,qy)=(h0,qx0,qy0) at t=0",
                ),
                *periodic,
                ConstraintSpec(
                    constraint_id="positivity",
                    name="Positive water depth",
                    constraint_type="positivity",
                    target="positivity",
                    expression="h>=h_min",
                    enforcement_options=["soft", "problem_hard"],
                    metadata={"output_index": 0, "minimum": self.minimum_depth},
                ),
            ],
            observations=ObservationSpec(available=False),
            metadata={
                "problem_version": "1.0",
                "source": "smooth uniform-flow analytic baseline",
                "spatial_dimension": 2,
                "time_dependent": True,
                "pde_type": "nonlinear_hyperbolic_system",
                "pde_family": "shallow_water",
                "equation_type": "nonlinear_hyperbolic_system",
                "num_outputs": 3,
                "highest_derivative_order": 1,
                "has_conservation_law": True,
                "has_conservation_form": True,
                "has_periodicity": True,
                "has_coupled_fields": True,
                "has_positivity_requirement": True,
                "has_complex_geometry": False,
                "has_observation_data": False,
                "challenge_tags": [
                    "nonlinear hyperbolic conservation law",
                    "transient",
                    "multi-output",
                    "coupled system",
                    "positivity requirement",
                    "periodic boundary",
                ],
                "parameters": {
                    "gravity": self.gravity,
                    "minimum_depth": self.minimum_depth,
                    "initial_state": list(self.initial_state),
                },
                "reference_solution_type": "analytic uniform flow",
                "reference_solution": "constant positive conservative state",
                "evaluation_grid": list(self.evaluation_shape),
                "metric_providers": [
                    "per_output_error",
                    "residual_components",
                    "constraint_components",
                    "mass_conservation",
                ],
                "metric_provider_options": {
                    "constraint_components": {"conservation_components": ["positivity"]},
                    "mass_conservation": {"component_index": 0, "time_axis": 2},
                },
                "case_study": False,
            },
        )

    def reference_solution(self, samples: torch.Tensor) -> torch.Tensor:
        state = torch.tensor(self.initial_state, dtype=samples.dtype, device=samples.device)
        return state.reshape(1, 3).repeat(samples.shape[0], 1)

    def constraint_enforcement_capabilities(self) -> Mapping[str, Any]:
        return {
            "soft_penalty": {"transform_ids": ["none"]},
            "problem_hard": {
                "transform_ids": ["positive_channels"],
                "parameters": {
                    "positive_channels": {
                        "indices": [0],
                        "minimum": self.minimum_depth,
                        "beta": 1.0,
                    }
                },
            },
        }

    def build_constraint_output_transform(
        self, transform_id: str, parameters: Mapping[str, Any] | None = None
    ):
        if transform_id != "positive_channels":
            return super().build_constraint_output_transform(transform_id, parameters)
        values = dict(parameters or {})
        indices = tuple(values.get("indices", (0,)))
        minimum = float(values.get("minimum", self.minimum_depth))
        beta = float(values.get("beta", 1.0))

        def transform(_: torch.Tensor, raw_outputs: torch.Tensor) -> torch.Tensor:
            return positive_output_transform(
                raw_outputs, indices=indices, minimum=minimum, beta=beta
            )

        return transform

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
            xy = torch.rand((int(n), 2), generator=generator)
            points = torch.cat((xy, torch.full((int(n), 1), self.t_bounds[0])), dim=1).to(target_device)
            result["initial"] = ConstraintBatch(
                "initial", "initial", points, target=self.reference_solution
            )
        if self.matching_category(category, ("periodic", "boundary")):
            t = self.t_bounds[0] + torch.rand((int(n), 1), generator=generator) * (
                self.t_bounds[1] - self.t_bounds[0]
            )
            other = torch.rand((int(n), 1), generator=generator)
            for axis, axis_index in (("x", 0), ("y", 1)):
                if axis_index == 0:
                    left = torch.cat((torch.zeros_like(other), other, t), dim=1)
                    right = torch.cat((torch.ones_like(other), other, t), dim=1)
                else:
                    left = torch.cat((other, torch.zeros_like(other), t), dim=1)
                    right = torch.cat((other, torch.ones_like(other), t), dim=1)
                result[f"periodic_{axis}_value"] = ConstraintBatch(
                    f"periodic_{axis}_value", "periodic", left.to(target_device), paired_points=right.to(target_device)
                )
                if self.periodic_derivatives:
                    result[f"periodic_{axis}_derivative"] = ConstraintBatch(
                        f"periodic_{axis}_derivative", "periodic", left.to(target_device),
                        paired_points=right.to(target_device), metadata={"derivative_axis": axis_index}
                    )
        if self.matching_category(category, ("positivity", "custom_constraint_domain")):
            points = self.sample_domain(n, seed=seed, device=target_device)
            result["positivity"] = ConstraintBatch("positivity", "positivity", points)
        return result

    def compute_governing_residuals(
        self,
        model: Any,
        samples: torch.Tensor,
        *,
        create_graph: bool = True,
    ) -> dict[str, torch.Tensor]:
        state = model(samples)[:, :3]
        h, q_x, q_y = (state[:, index : index + 1] for index in range(3))
        h_safe = torch.clamp(h, min=self.minimum_depth)
        fluxes = (
            (q_x, q_y),
            (q_x.square() / h_safe + 0.5 * self.gravity * h_safe.square(), q_x * q_y / h_safe),
            (q_x * q_y / h_safe, q_y.square() / h_safe + 0.5 * self.gravity * h_safe.square()),
        )
        residuals = conservative_system_residual(
            state,
            samples,
            time_axis=2,
            spatial_axes=(0, 1),
            fluxes=fluxes,
            create_graph=create_graph,
        )
        return dict(zip(("mass", "x_momentum", "y_momentum"), residuals))

    def compute_constraint_residuals(
        self,
        model: Any,
        samples: Mapping[str, Any],
        *,
        create_graph: bool = True,
    ) -> dict[str, torch.Tensor]:
        residuals: dict[str, torch.Tensor] = {}
        if "initial" in samples:
            batch = samples["initial"]
            points = self.constraint_points(batch)
            target = batch.target_values() if isinstance(batch, ConstraintBatch) else self.reference_solution(points)
            residuals["initial"] = model(points)[:, :3] - target.to(points)
        for axis, axis_index in (("x", 0), ("y", 1)):
            value_name = f"periodic_{axis}_value"
            if value_name in samples:
                residuals[value_name] = periodic_constraint_residual(
                    model, samples[value_name], output_indices=(0, 1, 2), create_graph=create_graph
                )
            derivative_name = f"periodic_{axis}_derivative"
            if derivative_name in samples:
                residuals[derivative_name] = periodic_constraint_residual(
                    model, samples[derivative_name], output_indices=(0, 1, 2),
                    derivative_axis=axis_index, create_graph=create_graph
                )
        if "positivity" in samples:
            points = self.constraint_points(samples["positivity"])
            depth = model(points)[:, :1]
            residuals["positivity"] = torch.relu(self.minimum_depth - depth)
        return residuals
