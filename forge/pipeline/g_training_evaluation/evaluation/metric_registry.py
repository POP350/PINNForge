"""Registered, problem-declared metric providers for the generic evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch

from forge.pipeline.a_problem_definition.problems.capabilities import coordinate_derivative


@dataclass(frozen=True)
class MetricContext:
    problem: Any
    model: torch.nn.Module
    evaluation_samples: torch.Tensor
    prediction: torch.Tensor
    reference: torch.Tensor | None
    residual_samples: torch.Tensor
    governing_residuals: Mapping[str, torch.Tensor]
    constraint_samples: Mapping[str, Any]
    constraint_residuals: Mapping[str, torch.Tensor]
    training_time: float | None
    device: str


MetricProvider = Callable[[MetricContext, Mapping[str, Any]], Mapping[str, Mapping[str, Any]]]
_METRIC_PROVIDERS: dict[str, MetricProvider] = {}


def register_metric_provider(provider_id: str, provider: MetricProvider, *, replace: bool = False) -> None:
    key = str(provider_id).strip()
    if not key:
        raise ValueError("metric provider_id must not be empty")
    if key in _METRIC_PROVIDERS and not replace:
        raise KeyError(f"metric provider already registered: {key}")
    _METRIC_PROVIDERS[key] = provider


def evaluate_metric_providers(context: MetricContext) -> dict[str, dict[str, Any]]:
    metadata = dict(context.problem.get_spec().metadata or {})
    provider_ids = list(metadata.get("metric_providers") or [])
    options = dict(metadata.get("metric_provider_options") or {})
    groups: dict[str, dict[str, Any]] = {
        "solution_metrics": {},
        "physics_metrics": {},
        "constraint_metrics": {},
        "conservation_metrics": {},
        "resource_metrics": {},
    }
    for provider_id in provider_ids:
        if provider_id not in _METRIC_PROVIDERS:
            raise KeyError(f"unknown metric provider: {provider_id}")
        provided = _METRIC_PROVIDERS[provider_id](context, dict(options.get(provider_id) or {}))
        for group, values in provided.items():
            if group not in groups:
                raise ValueError(f"metric provider {provider_id!r} returned unknown group {group!r}")
            groups[group].update(dict(values))
    return groups


def _per_output_error(context: MetricContext, _: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    if context.reference is None:
        return {"solution_metrics": {}}
    names = list(context.problem.get_spec().output_variables)
    values: dict[str, Any] = {}
    for index, name in enumerate(names):
        error = context.prediction[:, index] - context.reference[:, index]
        reference = context.reference[:, index]
        values[f"{name}_mse"] = float(error.square().mean().detach().cpu())
        values[f"{name}_relative_l2"] = float(
            torch.linalg.vector_norm(error).detach().cpu()
            / (torch.linalg.vector_norm(reference).detach().cpu() + 1.0e-12)
        )
    normalized = [values[f"{name}_relative_l2"] for name in names]
    values["normalized_relative_l2"] = sum(normalized) / len(normalized) if normalized else None
    values["maximum_absolute_error"] = float(
        (context.prediction - context.reference).abs().max().detach().cpu()
    )
    return {"solution_metrics": values}


def _residual_components(context: MetricContext, _: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    return {
        "physics_metrics": {
            f"{name}_mse": float(value.detach().square().mean().cpu())
            for name, value in context.governing_residuals.items()
        }
    }


def _constraint_components(context: MetricContext, options: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    conservation = set(options.get("conservation_components") or [])
    constraint_values: dict[str, Any] = {}
    conservation_values: dict[str, Any] = {}
    for name, value in context.constraint_residuals.items():
        target = conservation_values if name in conservation else constraint_values
        target[f"{name}_mse"] = float(value.detach().square().mean().cpu())
    return {
        "constraint_metrics": constraint_values,
        "conservation_metrics": conservation_values,
    }


def _velocity_pressure_flow(context: MetricContext, options: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    if context.reference is None:
        return {"solution_metrics": {}}
    velocity_indices = list(options.get("velocity_indices") or [0, 1])
    pressure_index = int(options.get("pressure_index", 2))
    predicted_velocity = context.prediction[:, velocity_indices]
    reference_velocity = context.reference[:, velocity_indices]
    velocity_error = predicted_velocity - reference_velocity
    predicted_pressure = context.prediction[:, pressure_index]
    reference_pressure = context.reference[:, pressure_index]
    if bool(options.get("align_pressure_gauge", False)):
        predicted_pressure = predicted_pressure - (predicted_pressure - reference_pressure).mean()
    pressure_error = predicted_pressure - reference_pressure
    return {
        "solution_metrics": {
            "velocity_combined_relative_l2": float(
                torch.linalg.vector_norm(velocity_error).detach().cpu()
                / (torch.linalg.vector_norm(reference_velocity).detach().cpu() + 1.0e-12)
            ),
            "pressure_gauge_aligned_relative_l2": float(
                torch.linalg.vector_norm(pressure_error).detach().cpu()
                / (torch.linalg.vector_norm(reference_pressure).detach().cpu() + 1.0e-12)
            ),
        }
    }


def _interface_region_error(context: MetricContext, options: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    if context.reference is None:
        return {"solution_metrics": {"interface_region_mse": None, "interface_region_count": 0}}
    threshold = float(options.get("threshold", 0.5))
    component = int(options.get("component_index", 0))
    reference = context.reference[:, component]
    mask = reference.abs() < threshold
    if not bool(mask.any()):
        return {"solution_metrics": {"interface_region_mse": None, "interface_region_count": 0}}
    error = context.prediction[mask, component] - reference[mask]
    return {
        "solution_metrics": {
            "interface_region_mse": float(error.square().mean().detach().cpu()),
            "interface_region_count": int(mask.sum().detach().cpu()),
            "interface_region_threshold": threshold,
        }
    }


def _mass_conservation(context: MetricContext, options: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    component = int(options.get("component_index", 0))
    time_axis = int(options.get("time_axis", -1))
    times = context.evaluation_samples[:, time_axis]
    unique_times, inverse = torch.unique(times, sorted=True, return_inverse=True)
    means = torch.stack(
        [context.prediction[inverse == index, component].mean() for index in range(unique_times.numel())]
    )
    initial = means[0]
    drift = (means - initial).abs()
    return {
        "conservation_metrics": {
            "total_mass_drift": float(drift.max().detach().cpu()),
            "relative_mass_drift": float(
                (drift.max() / (initial.abs() + 1.0e-12)).detach().cpu()
            ),
        }
    }


def _vorticity_error(context: MetricContext, options: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    reference_getter = getattr(context.problem, "reference_vorticity", None)
    if not callable(reference_getter):
        return {"solution_metrics": {"vorticity_mse": None}}
    inputs = context.evaluation_samples.detach().clone().to(context.device).requires_grad_(True)
    prediction = context.model(inputs)
    u_index = int(options.get("u_index", 0))
    v_index = int(options.get("v_index", 1))
    x_axis = int(options.get("x_axis", 0))
    y_axis = int(options.get("y_axis", 1))
    predicted_vorticity = coordinate_derivative(
        prediction[:, v_index : v_index + 1], inputs, x_axis, create_graph=False
    ) - coordinate_derivative(
        prediction[:, u_index : u_index + 1], inputs, y_axis, create_graph=False
    )
    reference = reference_getter(inputs).to(predicted_vorticity)
    return {
        "solution_metrics": {
            "vorticity_mse": float(
                (predicted_vorticity - reference).square().mean().detach().cpu()
            )
        }
    }


register_metric_provider("per_output_error", _per_output_error)
register_metric_provider("residual_components", _residual_components)
register_metric_provider("constraint_components", _constraint_components)
register_metric_provider("velocity_pressure_flow", _velocity_pressure_flow)
register_metric_provider("interface_region_error", _interface_region_error)
register_metric_provider("mass_conservation", _mass_conservation)
register_metric_provider("vorticity_error", _vorticity_error)


__all__ = [
    "MetricContext",
    "evaluate_metric_providers",
    "register_metric_provider",
]
