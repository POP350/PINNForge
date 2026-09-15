"""Reusable, PDE-agnostic capabilities for externally supplied problems."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import torch


TensorFunction = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class ConstraintBatch:
    """Coordinates and optional targets for one registered constraint."""

    name: str
    category: str
    points: torch.Tensor
    target: torch.Tensor | TensorFunction | None = None
    paired_points: torch.Tensor | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def target_values(self) -> torch.Tensor | None:
        if callable(self.target):
            return self.target(self.points)
        return self.target


def coordinate_derivative(
    values: torch.Tensor,
    coordinates: torch.Tensor,
    axis: int,
    *,
    order: int = 1,
    create_graph: bool = True,
) -> torch.Tensor:
    """Differentiate one tensor with respect to one coordinate repeatedly."""

    if order < 1:
        raise ValueError("derivative order must be positive")
    if values.ndim == 2 and values.shape[1] > 1:
        return torch.cat(
            [
                coordinate_derivative(
                    values[:, index : index + 1],
                    coordinates,
                    axis,
                    order=order,
                    create_graph=create_graph,
                )
                for index in range(values.shape[1])
            ],
            dim=1,
        )
    result = values
    for derivative_order in range(order):
        if not result.requires_grad:
            return torch.zeros(
                (coordinates.shape[0], result.shape[1] if result.ndim > 1 else 1),
                dtype=coordinates.dtype,
                device=coordinates.device,
            )
        gradient = torch.autograd.grad(
            result,
            coordinates,
            grad_outputs=torch.ones_like(result),
            create_graph=create_graph or derivative_order < order - 1,
            retain_graph=True,
            allow_unused=True,
        )[0]
        if gradient is None:
            return torch.zeros(
                (coordinates.shape[0], result.shape[1] if result.ndim > 1 else 1),
                dtype=coordinates.dtype,
                device=coordinates.device,
            )
        result = gradient[:, axis : axis + 1]
    return result


def conservative_system_residual(
    state: torch.Tensor,
    coordinates: torch.Tensor,
    *,
    time_axis: int,
    fluxes: Sequence[Sequence[torch.Tensor]],
    spatial_axes: Sequence[int],
    source: torch.Tensor | None = None,
    create_graph: bool = True,
) -> tuple[torch.Tensor, ...]:
    """Return ``state_t + div(flux) - source`` for every state component."""

    if state.ndim != 2:
        raise ValueError("state must have shape [n_points, n_components]")
    if len(fluxes) != state.shape[1]:
        raise ValueError("one flux vector is required per state component")
    residuals = []
    for component, component_fluxes in enumerate(fluxes):
        if len(component_fluxes) != len(spatial_axes):
            raise ValueError("every flux vector must match spatial_axes")
        residual = coordinate_derivative(
            state[:, component : component + 1],
            coordinates,
            time_axis,
            create_graph=create_graph,
        )
        for flux, axis in zip(component_fluxes, spatial_axes):
            residual = residual + coordinate_derivative(
                flux, coordinates, axis, create_graph=create_graph
            )
        if source is not None:
            residual = residual - source[:, component : component + 1]
        residuals.append(residual)
    return tuple(residuals)


@dataclass(frozen=True)
class SinusoidalCoefficientField:
    """Deterministic positive spatial coefficient field."""

    base: float
    amplitude: float
    frequencies: tuple[float, ...]

    def __call__(self, coordinates: torch.Tensor) -> torch.Tensor:
        if coordinates.shape[1] < len(self.frequencies):
            raise ValueError("coefficient field received too few coordinates")
        value = torch.ones_like(coordinates[:, :1])
        for axis, frequency in enumerate(self.frequencies):
            value = value * torch.sin(torch.pi * float(frequency) * coordinates[:, axis : axis + 1])
        result = float(self.base) + float(self.amplitude) * value
        if self.base <= abs(self.amplitude):
            raise ValueError("coefficient field requires base > abs(amplitude)")
        return result


@dataclass(frozen=True)
class BoxGeometry:
    """Axis-aligned geometry with deterministic seeded samplers."""

    bounds: tuple[tuple[float, float], ...]

    def sample_interior(
        self, n: int, *, generator: torch.Generator, device: str
    ) -> torch.Tensor:
        unit = torch.rand((int(n), len(self.bounds)), generator=generator)
        lower = torch.tensor([item[0] for item in self.bounds], dtype=unit.dtype)
        upper = torch.tensor([item[1] for item in self.bounds], dtype=unit.dtype)
        return (lower + unit * (upper - lower)).to(device)

    def sample_boundary(
        self, n: int, *, generator: torch.Generator, device: str
    ) -> torch.Tensor:
        points = self.sample_interior(n, generator=generator, device="cpu")
        axes = torch.randint(0, len(self.bounds), (int(n),), generator=generator)
        sides = torch.randint(0, 2, (int(n),), generator=generator)
        for index in range(int(n)):
            axis = int(axes[index])
            points[index, axis] = self.bounds[axis][int(sides[index])]
        return points.to(device)

    def contains(self, points: torch.Tensor) -> torch.Tensor:
        mask = torch.ones(points.shape[0], dtype=torch.bool, device=points.device)
        for axis, (lower, upper) in enumerate(self.bounds):
            mask &= (points[:, axis] >= lower) & (points[:, axis] <= upper)
        return mask


@dataclass(frozen=True)
class CircleObstacleGeometry:
    """Circular excluded region used by generic masked-domain sampling."""

    center: tuple[float, float]
    radius: float

    def contains(self, points: torch.Tensor) -> torch.Tensor:
        center = torch.tensor(self.center, dtype=points.dtype, device=points.device)
        return torch.linalg.vector_norm(points[:, :2] - center, dim=1) <= float(self.radius)

    def sample_boundary(
        self, n: int, *, generator: torch.Generator, device: str
    ) -> torch.Tensor:
        angles = 2.0 * torch.pi * torch.rand((int(n), 1), generator=generator)
        center = torch.tensor(self.center, dtype=angles.dtype)
        points = center + float(self.radius) * torch.cat((torch.cos(angles), torch.sin(angles)), dim=1)
        return points.to(device)


class MaskedDomainSampler:
    """Rejection sampler for a base geometry with reusable exclusion masks."""

    def __init__(
        self,
        base_geometry: BoxGeometry,
        *,
        exclusions: Sequence[Callable[[torch.Tensor], torch.Tensor]] = (),
        maximum_attempt_multiplier: int = 100,
    ) -> None:
        self.base_geometry = base_geometry
        self.exclusions = tuple(exclusions)
        self.maximum_attempt_multiplier = int(maximum_attempt_multiplier)

    def sample(self, n: int, *, seed: int | None = None, device: str = "cpu") -> torch.Tensor:
        requested = int(n)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(0 if seed is None else int(seed))
        accepted: list[torch.Tensor] = []
        count = 0
        attempts = 0
        while count < requested and attempts < self.maximum_attempt_multiplier:
            candidates = self.base_geometry.sample_interior(
                max(requested - count, 16), generator=generator, device="cpu"
            )
            keep = torch.ones(candidates.shape[0], dtype=torch.bool)
            for exclusion in self.exclusions:
                keep &= ~exclusion(candidates)
            selected = candidates[keep]
            if selected.numel():
                accepted.append(selected)
                count += int(selected.shape[0])
            attempts += 1
        if count < requested:
            raise RuntimeError("masked domain sampler could not satisfy the requested point count")
        return torch.cat(accepted, dim=0)[:requested].to(device)


def periodic_constraint_residual(
    model: torch.nn.Module,
    batch: ConstraintBatch,
    *,
    output_indices: Sequence[int] | None = None,
    derivative_axis: int | None = None,
    create_graph: bool = True,
) -> torch.Tensor:
    """Evaluate value or first-derivative periodicity on paired points."""

    if batch.paired_points is None:
        raise ValueError("periodic constraints require paired_points")
    left = batch.points.detach().clone().requires_grad_(derivative_axis is not None)
    right = batch.paired_points.detach().clone().requires_grad_(derivative_axis is not None)
    left_values = model(left)
    right_values = model(right)
    indices = list(output_indices) if output_indices is not None else list(range(left_values.shape[1]))
    left_values = left_values[:, indices]
    right_values = right_values[:, indices]
    if derivative_axis is not None:
        left_values = coordinate_derivative(
            left_values, left, derivative_axis, create_graph=create_graph
        )
        right_values = coordinate_derivative(
            right_values, right, derivative_axis, create_graph=create_graph
        )
    return left_values - right_values


def pressure_gauge_constraint(
    prediction: torch.Tensor,
    *,
    pressure_index: int,
    target: float = 0.0,
    mode: str = "anchor",
) -> torch.Tensor:
    """Return an anchor or mean pressure-gauge residual."""

    pressure = prediction[:, pressure_index : pressure_index + 1]
    if mode == "anchor":
        return pressure - float(target)
    if mode == "mean":
        return pressure.mean().reshape(1, 1) - float(target)
    raise ValueError("pressure gauge mode must be 'anchor' or 'mean'")


def observation_constraint(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    output_indices: Sequence[int] | None = None,
) -> torch.Tensor:
    indices = list(output_indices) if output_indices is not None else list(range(prediction.shape[1]))
    return prediction[:, indices] - target.to(device=prediction.device, dtype=prediction.dtype)


def positive_output_transform(
    raw_outputs: torch.Tensor,
    *,
    indices: Sequence[int],
    minimum: float = 1.0e-4,
    beta: float = 1.0,
) -> torch.Tensor:
    """Apply a softplus lower bound to selected output channels."""

    transformed = raw_outputs.clone()
    selected = list(indices)
    transformed[:, selected] = float(minimum) + torch.nn.functional.softplus(
        raw_outputs[:, selected], beta=float(beta)
    )
    return transformed


__all__ = [
    "BoxGeometry",
    "CircleObstacleGeometry",
    "ConstraintBatch",
    "MaskedDomainSampler",
    "SinusoidalCoefficientField",
    "conservative_system_residual",
    "coordinate_derivative",
    "observation_constraint",
    "periodic_constraint_residual",
    "positive_output_transform",
    "pressure_gauge_constraint",
]
