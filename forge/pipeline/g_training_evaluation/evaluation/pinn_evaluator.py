"""Generic evaluator for PhysicsProblem-backed learning systems."""

from __future__ import annotations

import math
import time
from typing import Any

import torch

from forge.pipeline.a_problem_definition.problems.base import PhysicsProblem
from forge.pipeline.g_training_evaluation.evaluation.metric_registry import (
    MetricContext,
    evaluate_metric_providers,
)
from forge.utils.reproducibility import default_device


PRIMARY_RANKING_METRIC = "mse"
DEFAULT_TOP_RESIDUAL_FRACTION = 0.05
MINIMUM_TOP_RESIDUAL_POINTS = 16
MAXIMUM_TOP_RESIDUAL_POINTS = 512
DEFAULT_RESIDUAL_AXIS_BINS = 8
DEFAULT_SURFACE_TOLERANCE_RATIO = 0.02
MAXIMUM_RESIDUAL_REGIONS = 3


class PINNEvaluator:
    def __init__(self, benchmark: object, device: str | None = None) -> None:
        if not isinstance(benchmark, PhysicsProblem):
            raise TypeError("PINNEvaluator requires a registered PhysicsProblem instance")
        self.problem = benchmark
        self.device = default_device(device)

    def evaluate(self, model: torch.nn.Module, training_time: float | None = None, convergence_epoch: int | None = None) -> dict[str, Any]:
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
        start = time.perf_counter()
        model.eval()
        samples = self.problem.evaluation_samples(device=self.device)
        with torch.no_grad():
            prediction = model(samples)
            reference = self.problem.reference_solution(samples)
            prediction_metrics = _prediction_metrics(prediction, reference)
            prediction_metrics.update(
                _fourier_metrics(self.problem, samples, prediction, reference)
            )
            wave_solution_metrics = _wave_solution_metrics(
                self.problem, samples, prediction, reference
            )
            prediction_metrics["temporal_l2re"] = (
                wave_solution_metrics.get("temporal_l2re")
                if wave_solution_metrics
                else _temporal_l2re(self.problem, samples, prediction, reference)
            )
            shock_metrics = _shock_region_metrics(self.problem, model, self.device)
        wave_initial_metrics = _wave_initial_state_metrics(
            self.problem, model, self.device
        )
        inference_time = time.perf_counter() - start

        residual_samples = self.problem.residual_evaluation_samples(device=self.device)
        residual_inputs = residual_samples.detach().clone().requires_grad_(True)
        governing = self.problem.compute_governing_residuals(
            model, residual_inputs, create_graph=False
        )
        governing_errors = {
            f"governing_residual_error/{name}": _rmse(residual)
            for name, residual in governing.items()
        }
        governing_residual_distribution = {
            name: _absolute_residual_distribution(residual)
            for name, residual in governing.items()
        }
        residual_spatial_summary = _residual_spatial_summary(
            samples=residual_samples,
            residuals=governing,
            coordinate_names=list(self.problem.get_spec().input_variables),
            surface_metadata=self.problem.residual_surface_metadata(),
        )
        governing_residual_error = _mean_or_none(governing_errors.values())

        constraint_samples = self.problem.sample_constraints(128, seed=0, device=self.device)
        constraint_residuals = self.problem.compute_constraint_residuals(model, constraint_samples, create_graph=False)
        constraint_errors = {
            f"constraint_violation/{name}": _rmse(residual)
            for name, residual in constraint_residuals.items()
        }
        constraint_violation = _mean_or_none(constraint_errors.values())
        observation_metrics = _observation_metrics(self.problem, model, self.device)
        parameter_metrics = _parameter_metrics(self.problem, model)
        metric_groups = evaluate_metric_providers(
            MetricContext(
                problem=self.problem,
                model=model,
                evaluation_samples=samples,
                prediction=prediction,
                reference=reference,
                residual_samples=residual_inputs,
                governing_residuals=governing,
                constraint_samples=constraint_samples,
                constraint_residuals=constraint_residuals,
                training_time=training_time,
                device=self.device,
            )
        )
        metric_groups["resource_metrics"].update(
            {
                "wall_clock_time": training_time,
                "peak_gpu_memory": (
                    int(torch.cuda.max_memory_allocated(self.device))
                    if self.device.startswith("cuda") and torch.cuda.is_available()
                    else 0
                ),
            }
        )

        metrics = {
            **prediction_metrics,
            **wave_solution_metrics,
            **wave_initial_metrics,
            **shock_metrics,
            **governing_errors,
            **constraint_errors,
            **observation_metrics,
            **parameter_metrics,
            "governing_residual_error": governing_residual_error,
            "constraint_violation": constraint_violation,
            "conservation_error": constraint_errors.get("constraint_violation/conservation"),
            "positivity_violation": constraint_errors.get("constraint_violation/positivity"),
            "algebraic_consistency_error": constraint_errors.get("constraint_violation/algebraic_consistency"),
            "observation_error": observation_metrics.get("observation_error"),
            "parameter_error": parameter_metrics.get("parameter_error"),
            "validation_error": None,
            "pde_residual_error": governing_residual_error,
            "governing_residual_distribution": governing_residual_distribution,
            "residual_distribution": governing_residual_distribution,
            "residual_spatial_summary": residual_spatial_summary,
            "boundary_error": constraint_errors.get("constraint_violation/boundary"),
            "initial_error": constraint_errors.get("constraint_violation/initial"),
            "inference_time": inference_time,
            "training_time": training_time,
            "convergence_epoch": convergence_epoch,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "peak_memory": (
                int(torch.cuda.max_memory_allocated(self.device))
                if self.device.startswith("cuda") and torch.cuda.is_available()
                else 0
            ),
            "metric_groups": metric_groups,
            "final_metrics": {
                "primary_metric": {
                    "name": "relative_l2",
                    "value": prediction_metrics.get("relative_l2_error"),
                },
                **metric_groups,
                "case_study": bool(self.problem.get_spec().metadata.get("case_study", False)),
            },
        }
        metrics["nan_detected"] = any(
            isinstance(value, float) and (value != value or value in {float("inf"), float("-inf")})
            for value in metrics.values()
        )
        metrics.update(self.solution_collapse_diagnostic(metrics))
        metrics["score_breakdown"] = self.score_breakdown(metrics)
        metrics["score"] = metrics["score_breakdown"]["total_score"]
        return metrics
    def evaluate_mse(self, model: torch.nn.Module) -> float | None:
        """Evaluate only reference MSE for inexpensive training-curve samples."""

        was_training = model.training
        model.eval()
        try:
            samples = self.problem.evaluation_samples(device=self.device)
            with torch.no_grad():
                reference = self.problem.reference_solution(samples)
                if reference is None:
                    return None
                prediction = model(samples)
                return float(torch.mean((prediction - reference) ** 2).detach().cpu())
        finally:
            model.train(was_training)

    @staticmethod
    def solution_collapse_diagnostic(metrics: dict[str, Any]) -> dict[str, Any]:
        """Identify a nearly constant prediction that only appears physics-valid."""

        prediction_std = metrics.get("prediction_std")
        reference_std = metrics.get("reference_std")
        relative_l2 = metrics.get("relative_l2_error")
        if prediction_std is None or reference_std is None or relative_l2 is None:
            return {
                "solution_collapse_detected": False,
                "solution_collapse_type": None,
                "prediction_reference_std_ratio": None,
            }
        reference_scale = float(reference_std)
        if reference_scale <= 0.0:
            return {
                "solution_collapse_detected": False,
                "solution_collapse_type": None,
                "prediction_reference_std_ratio": None,
            }
        variance_ratio = float(prediction_std) / max(reference_scale, 1e-12)
        detected = variance_ratio < 0.01 and float(relative_l2) > 0.5
        prediction_mean = metrics.get("prediction_mean")
        zero_like = (
            prediction_mean is not None
            and abs(float(prediction_mean)) < 0.05 * max(reference_scale, 1e-12)
        )
        return {
            "solution_collapse_detected": detected,
            "solution_collapse_type": (
                "zero_solution" if detected and zero_like else "constant_solution" if detected else None
            ),
            "prediction_reference_std_ratio": variance_ratio,
        }

    @staticmethod
    def score_breakdown(metrics: dict[str, Any]) -> dict[str, float]:
        rel = metrics.get("relative_l2_error")
        governing = metrics.get("governing_residual_error")
        convergence_epoch = metrics.get("convergence_epoch")
        training_time = metrics.get("training_time") or 0.0
        parameter_count = metrics.get("parameter_count") or 0
        accuracy_score = 1.0 / (1.0 + float(rel if rel is not None else 1.0))
        physics_score = 1.0 / (1.0 + float(governing if governing is not None else 10.0))
        convergence_score = 1.0 / (1.0 + float(convergence_epoch if convergence_epoch is not None else 10000.0) / 10000.0)
        cost_penalty = min(float(training_time), 3600.0) / 3600.0
        complexity_penalty = min(float(parameter_count), 1_000_000.0) / 1_000_000.0
        collapse_penalty = 0.0
        collapse = PINNEvaluator.solution_collapse_diagnostic(metrics)
        variance_ratio = collapse.get("prediction_reference_std_ratio")
        if variance_ratio is not None:
            if collapse["solution_collapse_detected"]:
                collapse_penalty = 0.50
            elif variance_ratio < 0.25 and float(rel if rel is not None else 0.0) > 0.2:
                collapse_penalty = 0.15 * (1.0 - variance_ratio / 0.25)
        total_score = (
            0.50 * accuracy_score
            + 0.30 * physics_score
            + 0.10 * convergence_score
            - 0.05 * cost_penalty
            - 0.05 * complexity_penalty
            - collapse_penalty
        )
        return {
            "accuracy_score": accuracy_score,
            "physics_score": physics_score,
            "convergence_score": convergence_score,
            "cost_penalty": cost_penalty,
            "complexity_penalty": complexity_penalty,
            "collapse_penalty": collapse_penalty,
            "total_score": total_score,
        }

    @classmethod
    def score(cls, metrics: dict[str, Any]) -> float:
        return cls.score_breakdown(metrics)["total_score"]


def _absolute_residual_distribution(residual: torch.Tensor) -> dict[str, float | int | None]:
    """Return compact absolute-residual quantiles for feedback-guided sampling."""

    values = residual.detach().abs().reshape(-1)
    finite_mask = torch.isfinite(values)
    finite = values[finite_mask]
    sample_count = int(values.numel())
    finite_count = int(finite.numel())
    summary: dict[str, float | int | None] = {
        "sample_count": sample_count,
        "finite_count": finite_count,
        "non_finite_count": sample_count - finite_count,
        "mean_absolute": None,
        "median_absolute": None,
        "p90_absolute": None,
        "p95_absolute": None,
        "p99_absolute": None,
        "maximum_absolute": None,
    }
    if finite_count == 0:
        return summary
    quantiles = torch.quantile(
        finite,
        torch.tensor(
            [0.50, 0.90, 0.95, 0.99],
            device=finite.device,
            dtype=finite.dtype,
        ),
    )
    summary.update(
        {
            "mean_absolute": float(finite.mean().cpu()),
            "median_absolute": float(quantiles[0].cpu()),
            "p90_absolute": float(quantiles[1].cpu()),
            "p95_absolute": float(quantiles[2].cpu()),
            "p99_absolute": float(quantiles[3].cpu()),
            "maximum_absolute": float(finite.max().cpu()),
        }
    )
    return summary


def _residual_spatial_summary(
    *,
    samples: torch.Tensor,
    residuals: dict[str, torch.Tensor] | Any,
    coordinate_names: list[str],
    surface_metadata: dict[str, Any] | Any | None,
    top_residual_fraction: float = DEFAULT_TOP_RESIDUAL_FRACTION,
    minimum_top_points: int = MINIMUM_TOP_RESIDUAL_POINTS,
    maximum_top_points: int = MAXIMUM_TOP_RESIDUAL_POINTS,
    number_of_axis_bins: int = DEFAULT_RESIDUAL_AXIS_BINS,
    surface_tolerance_ratio: float = DEFAULT_SURFACE_TOLERANCE_RATIO,
    maximum_regions: int = MAXIMUM_RESIDUAL_REGIONS,
) -> dict[str, Any]:
    """Compress fixed-grid residual locations into deterministic diagnostics.

    ``localized_error`` is true when at least half of the selected top-residual
    points occupy no more than one quarter of occupied axis-bin cells. A
    constant residual field is explicitly non-localized because top-k tie
    ordering carries no spatial information.
    """

    if not isinstance(samples, torch.Tensor) or samples.ndim != 2:
        return {
            "available": False,
            "reason": "residual_evaluation_samples_are_not_a_2d_tensor",
            "equations": {},
        }
    if len(coordinate_names) != int(samples.shape[1]):
        return {
            "available": False,
            "reason": "coordinate_name_count_mismatch",
            "equations": {},
        }
    if not isinstance(residuals, dict):
        residuals = dict(residuals or {})
    fraction = min(1.0, max(0.0, float(top_residual_fraction)))
    minimum = max(1, int(minimum_top_points))
    maximum = max(minimum, int(maximum_top_points))
    bins = max(1, int(number_of_axis_bins))
    region_cap = max(1, int(maximum_regions))
    metadata = dict(surface_metadata or {}) if isinstance(surface_metadata, dict) else {}
    equations: dict[str, Any] = {}
    for equation_id, residual in residuals.items():
        equations[str(equation_id)] = _equation_residual_spatial_summary(
            samples=samples,
            residual=residual,
            coordinate_names=coordinate_names,
            surface_metadata=metadata,
            top_residual_fraction=fraction,
            minimum_top_points=minimum,
            maximum_top_points=maximum,
            number_of_axis_bins=bins,
            surface_tolerance_ratio=float(surface_tolerance_ratio),
            maximum_regions=region_cap,
        )
    return {
        "available": bool(equations),
        "reason": None if equations else "no_governing_residual_equations",
        "coordinate_names": list(coordinate_names),
        "sample_basis": "fixed_residual_evaluator_points",
        "top_residual_fraction": fraction,
        "minimum_top_points": minimum,
        "maximum_top_points": maximum,
        "number_of_axis_bins": bins,
        "surface_tolerance_ratio": float(surface_tolerance_ratio),
        "maximum_regions": region_cap,
        "equations": equations,
    }


def _equation_residual_spatial_summary(
    *,
    samples: torch.Tensor,
    residual: torch.Tensor,
    coordinate_names: list[str],
    surface_metadata: dict[str, Any],
    top_residual_fraction: float,
    minimum_top_points: int,
    maximum_top_points: int,
    number_of_axis_bins: int,
    surface_tolerance_ratio: float,
    maximum_regions: int,
) -> dict[str, Any]:
    if not isinstance(residual, torch.Tensor) or residual.shape[0] != samples.shape[0]:
        return {
            "available": False,
            "reason": "residual_point_count_mismatch",
            "boundary_concentration": None,
            "initial_surface_concentration": None,
            "localized_error": None,
        }
    absolute = residual.detach().abs().reshape(int(samples.shape[0]), -1).mean(dim=1)
    coordinates = samples.detach()
    valid_mask = torch.isfinite(absolute) & torch.isfinite(coordinates).all(dim=1)
    valid_indices = torch.nonzero(valid_mask, as_tuple=False).reshape(-1)
    if int(valid_indices.numel()) == 0:
        return {
            "available": False,
            "reason": "no_finite_residual_coordinates",
            "boundary_concentration": None,
            "initial_surface_concentration": None,
            "localized_error": None,
        }
    valid_coordinates = coordinates[valid_indices]
    valid_residual = absolute[valid_indices]
    valid_count = int(valid_residual.numel())
    requested_top = int(math.ceil(valid_count * top_residual_fraction))
    top_count = min(valid_count, maximum_top_points, max(minimum_top_points, requested_top))
    top_local_indices = torch.topk(
        valid_residual,
        k=top_count,
        largest=True,
        sorted=True,
    ).indices
    top_coordinates = valid_coordinates[top_local_indices]
    top_residual = valid_residual[top_local_indices]
    axis_edges, axis_bin_indices, occupied_cells = _axis_bin_assignment(
        valid_coordinates,
        number_of_axis_bins,
    )
    top_axis_bin_indices = axis_bin_indices[top_local_indices]
    axis_statistics = _axis_bin_statistics(
        valid_coordinates=valid_coordinates,
        valid_residual=valid_residual,
        top_axis_bin_indices=top_axis_bin_indices,
        coordinate_names=coordinate_names,
        axis_edges=axis_edges,
        axis_bin_indices=axis_bin_indices,
    )
    localized_error = _localized_residual_flag(
        valid_residual=valid_residual,
        top_axis_bin_indices=top_axis_bin_indices,
        occupied_cells=occupied_cells,
    )
    regions = _top_residual_regions(
        top_coordinates=top_coordinates,
        top_residual=top_residual,
        top_axis_bin_indices=top_axis_bin_indices,
        coordinate_names=coordinate_names,
        axis_edges=axis_edges,
        maximum_regions=maximum_regions,
    )
    boundary_concentration = _surface_concentration(
        valid_coordinates=valid_coordinates,
        top_coordinates=top_coordinates,
        coordinate_names=coordinate_names,
        metadata=surface_metadata,
        surface_kind="boundary",
        tolerance_ratio=surface_tolerance_ratio,
    )
    initial_concentration = _surface_concentration(
        valid_coordinates=valid_coordinates,
        top_coordinates=top_coordinates,
        coordinate_names=coordinate_names,
        metadata=surface_metadata,
        surface_kind="initial",
        tolerance_ratio=surface_tolerance_ratio,
    )
    return {
        "available": True,
        "reason": None,
        "top_residual_fraction": float(top_count / valid_count),
        "requested_top_residual_fraction": top_residual_fraction,
        "top_point_count": top_count,
        "valid_point_count": valid_count,
        "non_finite_point_count": int(samples.shape[0]) - valid_count,
        "global_mean_abs_residual": float(valid_residual.mean().cpu()),
        "top_mean_abs_residual": float(top_residual.mean().cpu()),
        "top_regions": regions,
        "axis_bin_statistics": axis_statistics,
        "boundary_concentration": boundary_concentration,
        "initial_surface_concentration": initial_concentration,
        "localized_error": localized_error,
        "localization_rule": {
            "top_point_coverage": 0.50,
            "maximum_occupied_cell_fraction": 0.25,
            "constant_residual_is_not_localized": True,
        },
        "sample_basis": "fixed_residual_evaluator_points",
    }


def _axis_bin_assignment(
    coordinates: torch.Tensor,
    number_of_bins: int,
) -> tuple[list[torch.Tensor], torch.Tensor, set[tuple[int, ...]]]:
    edges: list[torch.Tensor] = []
    indices: list[torch.Tensor] = []
    for axis in range(int(coordinates.shape[1])):
        values = coordinates[:, axis]
        lower = values.min()
        upper = values.max()
        if float((upper - lower).abs().cpu()) <= 0.0:
            axis_edges = torch.stack([lower, upper])
            axis_index = torch.zeros_like(values, dtype=torch.long)
        else:
            axis_edges = torch.linspace(
                lower,
                upper,
                number_of_bins + 1,
                device=values.device,
                dtype=values.dtype,
            )
            axis_index = torch.bucketize(
                values.contiguous(),
                axis_edges[1:-1],
                right=False,
            )
        edges.append(axis_edges)
        indices.append(axis_index)
    matrix = torch.stack(indices, dim=1)
    occupied = {
        tuple(int(value) for value in row)
        for row in matrix.detach().cpu().tolist()
    }
    return edges, matrix, occupied


def _axis_bin_statistics(
    *,
    valid_coordinates: torch.Tensor,
    valid_residual: torch.Tensor,
    top_axis_bin_indices: torch.Tensor,
    coordinate_names: list[str],
    axis_edges: list[torch.Tensor],
    axis_bin_indices: torch.Tensor,
) -> dict[str, list[dict[str, Any]]]:
    top_count = max(1, int(top_axis_bin_indices.shape[0]))
    result: dict[str, list[dict[str, Any]]] = {}
    for axis, name in enumerate(coordinate_names):
        items: list[dict[str, Any]] = []
        edges = axis_edges[axis]
        bin_count = max(1, int(edges.numel()) - 1)
        for bin_index in range(bin_count):
            mask = axis_bin_indices[:, axis] == bin_index
            values = valid_residual[mask]
            if int(values.numel()) == 0:
                continue
            top_in_bin = int(
                (top_axis_bin_indices[:, axis] == bin_index).sum().item()
            )
            items.append(
                {
                    "lower": float(edges[bin_index].cpu()),
                    "upper": float(edges[bin_index + 1].cpu()),
                    "sample_count": int(values.numel()),
                    "mean_abs_residual": float(values.mean().cpu()),
                    "p95_abs_residual": float(
                        torch.quantile(values, 0.95).cpu()
                    ),
                    "top_point_fraction": float(top_in_bin / top_count),
                    "sample_basis": "fixed_residual_evaluator_points",
                }
            )
        result[str(name)] = items
    return result


def _localized_residual_flag(
    *,
    valid_residual: torch.Tensor,
    top_axis_bin_indices: torch.Tensor,
    occupied_cells: set[tuple[int, ...]],
) -> bool:
    if int(valid_residual.numel()) < 2:
        return False
    spread = float((valid_residual.max() - valid_residual.min()).cpu())
    scale = max(float(valid_residual.abs().max().cpu()), 1.0)
    if spread <= 1.0e-12 * scale:
        return False
    cell_counts: dict[tuple[int, ...], int] = {}
    for row in top_axis_bin_indices.detach().cpu().tolist():
        cell = tuple(int(value) for value in row)
        cell_counts[cell] = cell_counts.get(cell, 0) + 1
    target = math.ceil(0.50 * int(top_axis_bin_indices.shape[0]))
    cumulative = 0
    cells_needed = 0
    for count in sorted(cell_counts.values(), reverse=True):
        cumulative += count
        cells_needed += 1
        if cumulative >= target:
            break
    allowed_cells = max(1, math.floor(0.25 * max(1, len(occupied_cells))))
    return cells_needed <= allowed_cells


def _top_residual_regions(
    *,
    top_coordinates: torch.Tensor,
    top_residual: torch.Tensor,
    top_axis_bin_indices: torch.Tensor,
    coordinate_names: list[str],
    axis_edges: list[torch.Tensor],
    maximum_regions: int,
) -> list[dict[str, Any]]:
    groups: dict[tuple[int, ...], list[int]] = {}
    for index, row in enumerate(top_axis_bin_indices.detach().cpu().tolist()):
        cell = tuple(int(value) for value in row)
        groups.setdefault(cell, []).append(index)
    ordered = sorted(
        groups.items(),
        key=lambda item: (
            -len(item[1]),
            -float(top_residual[item[1]].mean().cpu()),
            item[0],
        ),
    )
    top_count = max(1, int(top_residual.numel()))
    regions: list[dict[str, Any]] = []
    for cell, member_indices in ordered[:maximum_regions]:
        values = top_residual[member_indices]
        bounds = {
            str(name): [
                float(axis_edges[axis][cell[axis]].cpu()),
                float(axis_edges[axis][cell[axis] + 1].cpu()),
            ]
            for axis, name in enumerate(coordinate_names)
        }
        regions.append(
            {
                "coordinate_bounds": bounds,
                "point_count": len(member_indices),
                "point_fraction": float(len(member_indices) / top_count),
                "mean_abs_residual": float(values.mean().cpu()),
                "max_abs_residual": float(values.max().cpu()),
                "compression": "occupied_axis_bin_cell",
            }
        )
    return regions


def _surface_concentration(
    *,
    valid_coordinates: torch.Tensor,
    top_coordinates: torch.Tensor,
    coordinate_names: list[str],
    metadata: dict[str, Any],
    surface_kind: str,
    tolerance_ratio: float,
) -> bool | None:
    bounds = metadata.get("coordinate_bounds")
    if not isinstance(bounds, dict):
        return None
    if surface_kind == "boundary":
        if not metadata.get("boundary_available"):
            return None
        axes = metadata.get("spatial_boundary_axes")
        if not isinstance(axes, list) or not axes:
            return None
        axis_names = [str(name) for name in axes]
        near = _near_axis_bounds(
            valid_coordinates, coordinate_names, bounds, axis_names, tolerance_ratio
        )
        top_near = _near_axis_bounds(
            top_coordinates, coordinate_names, bounds, axis_names, tolerance_ratio
        )
    elif surface_kind == "initial":
        if not metadata.get("initial_surface_available"):
            return None
        time_variable = metadata.get("time_variable")
        if not isinstance(time_variable, str) or time_variable not in coordinate_names:
            return None
        if time_variable not in bounds:
            return None
        axis = coordinate_names.index(time_variable)
        lower, upper = (float(value) for value in bounds[time_variable])
        tolerance = max(0.0, tolerance_ratio) * abs(upper - lower)
        near = valid_coordinates[:, axis] <= lower + tolerance
        top_near = top_coordinates[:, axis] <= lower + tolerance
    else:
        raise ValueError(f"Unknown residual surface kind: {surface_kind}")
    global_fraction = float(near.float().mean().cpu()) if near.numel() else 0.0
    top_fraction = float(top_near.float().mean().cpu()) if top_near.numel() else 0.0
    return bool(
        top_fraction >= 0.50
        and top_fraction >= max(2.0 * global_fraction, global_fraction + 0.10)
    )


def _near_axis_bounds(
    coordinates: torch.Tensor,
    coordinate_names: list[str],
    bounds: dict[str, Any],
    axis_names: list[str],
    tolerance_ratio: float,
) -> torch.Tensor:
    near = torch.zeros(
        int(coordinates.shape[0]),
        dtype=torch.bool,
        device=coordinates.device,
    )
    for name in axis_names:
        if name not in coordinate_names or name not in bounds:
            continue
        axis = coordinate_names.index(name)
        lower, upper = (float(value) for value in bounds[name])
        tolerance = max(0.0, tolerance_ratio) * abs(upper - lower)
        near = near | (coordinates[:, axis] <= lower + tolerance)
        near = near | (coordinates[:, axis] >= upper - tolerance)
    return near


def _prediction_metrics(prediction: torch.Tensor, reference: torch.Tensor | None) -> dict[str, float | None]:
    if reference is None:
        return {
            "mae": None,
            "mse": None,
            "merr": None,
            "mxe": None,
            "l1re": None,
            "l2re": None,
            "crmse": None,
            "relative_l1_error": None,
            "relative_l2_error": None,
            "relative_linf_error": None,
            "prediction_mean": torch.mean(prediction).item(),
            "prediction_std": torch.std(prediction, unbiased=False).item(),
            "reference_std": None,
        }
    error = prediction - reference
    absolute_error = torch.abs(error)
    mae = torch.mean(absolute_error).item()
    mse = torch.mean(error**2).item()
    merr = torch.max(absolute_error).item()
    l1re = _pinnacle_ratio(
        torch.sum(absolute_error).item(), torch.sum(torch.abs(reference)).item()
    )
    l2re = _pinnacle_ratio(torch.linalg.norm(error).item(), torch.linalg.norm(reference).item())
    crmse = torch.abs(torch.mean(error)).item()
    return {
        "mae": mae,
        "mse": mse,
        "merr": merr,
        "mxe": merr,
        "l1re": l1re,
        "l2re": l2re,
        "crmse": crmse,
        "relative_l1_error": l1re,
        "relative_l2_error": l2re,
        "relative_linf_error": _pinnacle_ratio(merr, torch.max(torch.abs(reference)).item()),
        "prediction_mean": torch.mean(prediction).item(),
        "prediction_std": torch.std(prediction, unbiased=False).item(),
        "reference_std": torch.std(reference, unbiased=False).item(),
    }


def _fourier_metrics(
    problem: PhysicsProblem,
    samples: torch.Tensor,
    prediction: torch.Tensor,
    reference: torch.Tensor | None,
) -> dict[str, float | None]:
    metrics = {
        "fmse_low": None,
        "fmse_mid": None,
        "fmse_high": None,
        "frmse_low": None,
        "frmse_mid": None,
        "frmse_high": None,
    }
    if reference is None:
        return metrics
    calculator = getattr(problem, "pinnacle_fourier_metrics", None)
    if not callable(calculator):
        return metrics
    values = calculator(samples, prediction, reference) or {}
    for band in ("low", "mid", "high"):
        value = values.get(band)
        if value is None:
            continue
        numeric = float(value)
        metrics[f"fmse_{band}"] = numeric
        # PINNacle's released artifacts use the historical `frmse_*` spelling.
        metrics[f"frmse_{band}"] = numeric
    return metrics


def _temporal_l2re(
    problem: PhysicsProblem,
    samples: torch.Tensor,
    prediction: torch.Tensor,
    reference: torch.Tensor | None,
) -> list[dict[str, float]] | None:
    if reference is None:
        return None
    spec = problem.get_spec()
    input_variables = list(getattr(spec, "input_variables", []) or [])
    if "t" not in input_variables:
        return None
    time_values = samples[:, input_variables.index("t")]
    unique_times, inverse = torch.unique(time_values, sorted=True, return_inverse=True)
    # Random continuous samples do not define meaningful temporal slices.
    if unique_times.numel() < 2 or unique_times.numel() > samples.shape[0] // 2:
        return None
    result: list[dict[str, float]] = []
    for index, time_value in enumerate(unique_times):
        mask = inverse == index
        denominator = torch.linalg.norm(reference[mask]).item()
        l2re = _pinnacle_ratio(
            torch.linalg.norm(prediction[mask] - reference[mask]).item(), denominator
        )
        result.append({"time": float(time_value.detach().cpu()), "l2re": l2re})
    return result


def _wave_solution_metrics(
    problem: PhysicsProblem,
    samples: torch.Tensor,
    prediction: torch.Tensor,
    reference: torch.Tensor | None,
) -> dict[str, Any]:
    problem_id = str(getattr(problem.get_spec(), "problem_id", ""))
    if problem_id != "wave_1d":
        return {}
    defaults: dict[str, Any] = {
        "prediction_amplitude": torch.std(
            prediction, unbiased=False
        ).item(),
        "reference_amplitude": None,
        "amplitude_ratio": None,
        "amplitude_absolute_error": None,
        "temporal_mse": None,
        "temporal_reference_norm": None,
        "temporal_prediction_norm": None,
        "temporal_amplitude_ratio": None,
        "temporal_l2re": None,
        "temporal_phase_error": None,
        "mean_phase_error": None,
        "maximum_phase_error": None,
        "phase_error_valid": False,
        "phase_error_method": "dominant_spatial_mode_hilbert_phase",
    }
    if reference is None:
        return defaults
    prediction_amplitude = torch.std(prediction, unbiased=False).item()
    reference_amplitude = torch.std(reference, unbiased=False).item()
    defaults.update(
        {
            "prediction_amplitude": prediction_amplitude,
            "reference_amplitude": reference_amplitude,
            "amplitude_ratio": (
                prediction_amplitude / reference_amplitude
                if reference_amplitude > 0.0
                else None
            ),
            "amplitude_absolute_error": abs(
                prediction_amplitude - reference_amplitude
            ),
        }
    )
    spec = problem.get_spec()
    variables = list(spec.input_variables)
    if "t" not in variables:
        return defaults
    time_index = variables.index("t")
    time_values = samples[:, time_index]
    unique_times, inverse = torch.unique(time_values, sorted=True, return_inverse=True)
    if unique_times.numel() < 2 or unique_times.numel() > samples.shape[0] // 2:
        return defaults

    temporal_mse: list[dict[str, Any]] = []
    temporal_reference_norm: list[dict[str, Any]] = []
    temporal_prediction_norm: list[dict[str, Any]] = []
    temporal_amplitude_ratio: list[dict[str, Any]] = []
    temporal_l2re: list[dict[str, Any]] = []
    prediction_rows: list[torch.Tensor] = []
    reference_rows: list[torch.Tensor] = []
    consistent_width: int | None = None
    for index, time_value in enumerate(unique_times):
        mask = inverse == index
        slice_prediction = prediction[mask].reshape(-1)
        slice_reference = reference[mask].reshape(-1)
        if "x" in variables:
            x_values = samples[mask, variables.index("x")]
            order = torch.argsort(x_values)
            slice_prediction = slice_prediction[order]
            slice_reference = slice_reference[order]
        error = slice_prediction - slice_reference
        reference_norm = torch.linalg.vector_norm(slice_reference).item()
        prediction_norm = torch.linalg.vector_norm(slice_prediction).item()
        reference_std = torch.std(slice_reference, unbiased=False).item()
        prediction_std = torch.std(slice_prediction, unbiased=False).item()
        time_number = float(time_value.detach().cpu())
        temporal_mse.append(
            {"time": time_number, "mse": torch.mean(error.square()).item()}
        )
        temporal_reference_norm.append(
            {"time": time_number, "reference_norm": reference_norm}
        )
        temporal_prediction_norm.append(
            {"time": time_number, "prediction_norm": prediction_norm}
        )
        temporal_amplitude_ratio.append(
            {
                "time": time_number,
                "amplitude_ratio": (
                    prediction_std / reference_std
                    if reference_std > 0.0
                    else None
                ),
            }
        )
        temporal_l2re.append(
            {
                "time": time_number,
                "l2re": (
                    torch.linalg.vector_norm(error).item() / reference_norm
                    if reference_norm > 0.0
                    else None
                ),
            }
        )
        width = int(slice_reference.numel())
        if consistent_width is None:
            consistent_width = width
        if width == consistent_width:
            prediction_rows.append(slice_prediction)
            reference_rows.append(slice_reference)

    defaults.update(
        {
            "temporal_mse": temporal_mse,
            "temporal_reference_norm": temporal_reference_norm,
            "temporal_prediction_norm": temporal_prediction_norm,
            "temporal_amplitude_ratio": temporal_amplitude_ratio,
            "temporal_l2re": temporal_l2re,
        }
    )
    if (
        len(reference_rows) != int(unique_times.numel())
        or consistent_width is None
        or consistent_width < 2
        or len(reference_rows) < 4
    ):
        return defaults

    reference_matrix = torch.stack(reference_rows)
    prediction_matrix = torch.stack(prediction_rows)
    try:
        _, _, vh = torch.linalg.svd(reference_matrix, full_matrices=False)
    except RuntimeError:
        return defaults
    dominant_spatial_mode = vh[0]
    reference_signal = reference_matrix @ dominant_spatial_mode
    prediction_signal = prediction_matrix @ dominant_spatial_mode
    reference_analytic = _analytic_signal(reference_signal)
    prediction_analytic = _analytic_signal(prediction_signal)
    reference_envelope = reference_analytic.abs()
    prediction_envelope = prediction_analytic.abs()
    reference_threshold = max(
        float(reference_envelope.max().item()) * 1.0e-6,
        torch.finfo(reference_envelope.dtype).eps,
    )
    prediction_threshold = max(
        float(reference_envelope.max().item()) * 1.0e-8,
        torch.finfo(prediction_envelope.dtype).eps,
    )
    valid = (reference_envelope > reference_threshold) & (
        prediction_envelope > prediction_threshold
    )
    if not bool(valid.any()):
        return defaults
    wrapped = torch.angle(
        torch.exp(
            1j
            * (
                torch.angle(prediction_analytic)
                - torch.angle(reference_analytic)
            )
        )
    ).abs()
    phase_entries = [
        {
            "time": float(unique_times[index].detach().cpu()),
            "phase_error": (
                float(wrapped[index].detach().cpu()) if bool(valid[index]) else None
            ),
            "valid": bool(valid[index]),
        }
        for index in range(int(unique_times.numel()))
    ]
    valid_errors = wrapped[valid]
    defaults.update(
        {
            "temporal_phase_error": phase_entries,
            "mean_phase_error": float(valid_errors.mean().detach().cpu()),
            "maximum_phase_error": float(valid_errors.max().detach().cpu()),
            "phase_error_valid": True,
        }
    )
    return defaults


def _analytic_signal(signal: torch.Tensor) -> torch.Tensor:
    """Return a Hilbert analytic signal using a differentiability-free FFT."""

    count = int(signal.numel())
    spectrum = torch.fft.fft(signal)
    multiplier = torch.zeros(
        count, dtype=signal.dtype, device=signal.device
    )
    multiplier[0] = 1.0
    if count % 2 == 0:
        multiplier[count // 2] = 1.0
        multiplier[1 : count // 2] = 2.0
    else:
        multiplier[1 : (count + 1) // 2] = 2.0
    return torch.fft.ifft(spectrum * multiplier)


def _wave_initial_state_metrics(
    problem: PhysicsProblem,
    model: torch.nn.Module,
    device: str,
) -> dict[str, Any]:
    defaults = {
        "initial_displacement_mse": None,
        "initial_displacement_rmse": None,
        "initial_displacement_relative_l2": None,
        "initial_velocity_mse": None,
        "initial_velocity_rmse": None,
        "initial_velocity_relative_l2": None,
        "initial_velocity_relative_l2_denominator_zero": False,
    }
    if str(getattr(problem.get_spec(), "problem_id", "")) != "wave_1d":
        return defaults
    sampler = getattr(problem, "initial_state_evaluation_samples", None)
    displacement_target = getattr(problem, "initial_displacement", None)
    velocity_target = getattr(problem, "initial_velocity", None)
    if not all(callable(value) for value in (sampler, displacement_target, velocity_target)):
        return defaults
    inputs = sampler(device=device).detach().clone().requires_grad_(True)
    prediction = model(inputs)
    displacement_reference = displacement_target(inputs).to(
        device=prediction.device, dtype=prediction.dtype
    )
    displacement_error = prediction - displacement_reference
    gradient = torch.autograd.grad(
        outputs=prediction,
        inputs=inputs,
        grad_outputs=torch.ones_like(prediction),
        create_graph=False,
        retain_graph=False,
        allow_unused=False,
    )[0]
    time_index = list(problem.get_spec().input_variables).index("t")
    velocity_prediction = gradient[:, time_index : time_index + 1]
    velocity_reference = velocity_target(inputs).to(
        device=prediction.device, dtype=prediction.dtype
    )
    velocity_error = velocity_prediction - velocity_reference
    displacement_denominator = torch.linalg.vector_norm(
        displacement_reference
    ).item()
    velocity_denominator = torch.linalg.vector_norm(velocity_reference).item()
    displacement_mse = torch.mean(displacement_error.square()).item()
    velocity_mse = torch.mean(velocity_error.square()).item()
    return {
        "initial_displacement_mse": displacement_mse,
        "initial_displacement_rmse": math.sqrt(displacement_mse),
        "initial_displacement_relative_l2": (
            torch.linalg.vector_norm(displacement_error).item()
            / displacement_denominator
            if displacement_denominator > 0.0
            else None
        ),
        "initial_velocity_mse": velocity_mse,
        "initial_velocity_rmse": math.sqrt(velocity_mse),
        "initial_velocity_relative_l2": (
            torch.linalg.vector_norm(velocity_error).item()
            / velocity_denominator
            if velocity_denominator > 0.0
            else None
        ),
        "initial_velocity_relative_l2_denominator_zero": (
            velocity_denominator == 0.0
        ),
    }


def _pinnacle_ratio(numerator: float, denominator: float) -> float:
    if denominator != 0.0:
        return numerator / denominator
    return float("nan") if numerator == 0.0 else float("inf")


def _shock_region_metrics(problem: PhysicsProblem, model: torch.nn.Module, device: str) -> dict[str, Any]:
    features = problem.get_problem_features() if hasattr(problem, "get_problem_features") else {}
    if not bool(features.get("has_shock")):
        return {
            "shock_region_mse": None,
            "shock_region_relative_l2_error": None,
            "shock_region_relative_linf_error": None,
            "shock_region_sample_count": 0,
            "shock_region_definition": None,
        }
    sampler = getattr(problem, "shock_region_evaluation_samples", None)
    samples = sampler(device=device) if callable(sampler) else None
    if samples is None:
        return {
            "shock_region_mse": None,
            "shock_region_relative_l2_error": None,
            "shock_region_relative_linf_error": None,
            "shock_region_sample_count": 0,
            "shock_region_definition": None,
        }
    reference = problem.reference_solution(samples)
    if reference is None:
        return {
            "shock_region_mse": None,
            "shock_region_relative_l2_error": None,
            "shock_region_relative_linf_error": None,
            "shock_region_sample_count": int(samples.shape[0]) if hasattr(samples, "shape") else None,
            "shock_region_definition": "benchmark_defined_no_reference",
        }
    prediction = model(samples)
    error = prediction - reference
    definition_getter = getattr(problem, "shock_region_definition", None)
    definition = definition_getter() if callable(definition_getter) else {
        "name": "benchmark_defined_shock_region"
    }
    definition = {**(definition or {}), "evaluation_point_count": int(samples.shape[0])}
    return {
        "shock_region_mse": torch.mean(error**2).item(),
        "shock_region_relative_l2_error": torch.linalg.norm(error).item() / (torch.linalg.norm(reference).item() + 1e-12),
        "shock_region_relative_linf_error": torch.max(torch.abs(error)).item() / (torch.max(torch.abs(reference)).item() + 1e-12),
        "shock_region_sample_count": int(samples.shape[0]),
        "shock_region_definition": definition,
    }


def _rmse(value: torch.Tensor) -> float:
    return torch.mean(value.detach() ** 2).sqrt().item()


def _mean_or_none(values) -> float | None:
    values = [float(value) for value in values if value is not None]
    return sum(values) / len(values) if values else None


def _observation_metrics(problem: PhysicsProblem, model: torch.nn.Module, device: str) -> dict[str, Any]:
    if not hasattr(problem, "evaluation_observations"):
        return {"observation_error": None, "observation_mse": None}
    observations = problem.evaluation_observations(device=device)
    if not observations:
        return {"observation_error": None, "observation_mse": None}
    metrics: dict[str, Any] = {}
    errors = []
    with torch.no_grad():
        for name, (inputs, target) in observations.items():
            inputs = inputs.to(device)
            target = target.to(device)
            residual = model(inputs) - target
            rmse = _rmse(residual)
            mse = torch.mean(residual.detach() ** 2).item()
            metrics[f"observation_error/{name}"] = rmse
            metrics[f"observation_mse/{name}"] = mse
            errors.append(rmse)
    metrics["observation_error"] = _mean_or_none(errors)
    metrics["observation_mse"] = _mean_or_none(metrics[key] for key in metrics if key.startswith("observation_mse/"))
    return metrics


def _parameter_metrics(problem: PhysicsProblem, model: torch.nn.Module) -> dict[str, Any]:
    reference = problem.reference_parameters()
    model_parameters = getattr(model, "physical_parameters", None)
    if not reference or not model_parameters:
        return {"parameter_error": None, "parameter_relative_error": None, "identified_parameters": {}}
    metrics: dict[str, Any] = {}
    absolute_errors = []
    relative_errors = []
    identified = {}
    for name, parameter in model_parameters.items():
        value = float(parameter.detach().cpu())
        identified[name] = value
        if name not in reference:
            continue
        true_value = float(reference[name])
        absolute_error = abs(value - true_value)
        relative_error = absolute_error / (abs(true_value) + 1e-12)
        metrics[f"identified_parameter/{name}"] = value
        metrics[f"parameter_error/{name}"] = absolute_error
        metrics[f"parameter_relative_error/{name}"] = relative_error
        absolute_errors.append(absolute_error)
        relative_errors.append(relative_error)
    metrics["identified_parameters"] = identified
    metrics["parameter_error"] = _mean_or_none(absolute_errors)
    metrics["parameter_relative_error"] = _mean_or_none(relative_errors)
    return metrics
