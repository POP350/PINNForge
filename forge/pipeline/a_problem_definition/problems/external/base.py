"""Shared implementation utilities for external coordinate-PINN problems."""

from __future__ import annotations

from itertools import product
from typing import Any, Mapping, Sequence

import torch

from forge.pipeline.a_problem_definition.problems.base import PhysicsProblem
from forge.pipeline.a_problem_definition.problems.capabilities import BoxGeometry


class ExternalPDEProblem(PhysicsProblem):
    """Box-domain problem base with seeded sampling and a fixed evaluation grid."""

    evaluation_shape: tuple[int, ...] = ()

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        spec = self.get_spec()
        self._variables = list(spec.input_variables)
        self._bounds = tuple(tuple(spec.domain.bounds[name]) for name in self._variables)
        self.geometry = BoxGeometry(self._bounds)

    @staticmethod
    def _generator(seed: int | None) -> torch.Generator:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(0 if seed is None else int(seed))
        return generator

    def sample_domain(
        self,
        n: int,
        *,
        seed: int | None = None,
        device: str | None = None,
    ) -> torch.Tensor:
        return self.geometry.sample_interior(
            int(n), generator=self._generator(seed), device=device or "cpu"
        )

    def sample_box_boundary(
        self,
        n: int,
        *,
        seed: int | None = None,
        device: str | None = None,
    ) -> torch.Tensor:
        return self.geometry.sample_boundary(
            int(n), generator=self._generator(seed), device=device or "cpu"
        )

    def evaluation_samples(self, *, device: str | None = None) -> torch.Tensor:
        shape = self.evaluation_shape or tuple(32 for _ in self._variables)
        axes = [
            torch.linspace(lower, upper, int(count))
            for (lower, upper), count in zip(self._bounds, shape)
        ]
        return torch.cartesian_prod(*axes).reshape(-1, len(axes)).to(device or "cpu")

    def residual_evaluation_samples(self, *, device: str | None = None) -> torch.Tensor:
        return self.evaluation_samples(device=device)

    def residual_surface_metadata(self) -> Mapping[str, Any]:
        spec = self.get_spec()
        spatial = self._variables[: int(spec.metadata.get("spatial_dimension") or 0)]
        temporal = [name for name in self._variables if name in {"t", "time"}]
        return {
            "coordinate_bounds": dict(spec.domain.bounds),
            "boundary_axes": spatial,
            "initial_axes": temporal,
        }

    def get_problem_features(self) -> dict[str, Any]:
        return dict(self.get_spec().metadata)

    @staticmethod
    def constraint_points(value: Any) -> torch.Tensor:
        points = getattr(value, "points", value)
        if not isinstance(points, torch.Tensor):
            raise TypeError("constraint samples must expose a torch Tensor named 'points'")
        return points

    @staticmethod
    def matching_category(category: str | None, accepted: Sequence[str]) -> bool:
        return category is None or str(category) in set(accepted)
