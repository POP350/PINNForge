"""Compact, backward-compatible candidate artifact serialization.

The search controller keeps its complete in-memory records for ranking,
reflection, and resume state.  This module only changes the per-candidate
machine-readable artifacts written below ``runtime_candidates``.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

from forge.utils.json_safety import json_safe


OUTPUT_DETAILS = frozenset({"metrics", "compact", "diagnostic"})
MAXIMUM_REGULAR_HISTORY_RECORDS = 100

TEMPORAL_FIELDS = (
    "temporal_l2re",
    "temporal_mse",
    "temporal_reference_norm",
    "temporal_prediction_norm",
    "temporal_amplitude_ratio",
    "temporal_phase_error",
)

CANONICAL_METRIC_FIELDS = (
    "mse",
    "proxy_mse",
    "selected_model_mse",
    "last_checkpoint_mse",
    "mae",
    "merr",
    "l1re",
    "l2re",
    "pde_residual",
    "boundary_error",
    "initial_error",
    "constraint_violation",
    "initial_displacement_mse",
    "initial_velocity_mse",
    "amplitude_ratio",
    "mean_phase_error",
    "maximum_phase_error",
    "prediction_mean",
    "prediction_std",
    "reference_std",
    "training_time",
    "parameter_count",
    "peak_gpu_memory",
    "actual_peak_sampling_points",
    "convergence_status",
    "nan_detected",
    "solution_valid",
    "solution_collapse_detected",
)

_METRIC_FALLBACKS = {
    "l1re": ("relative_l1", "relative_l1_error"),
    "l2re": ("relative_l2", "relative_l2_error"),
    "pde_residual": ("pde_residual_error", "governing_residual_error"),
    "merr": ("mxe",),
}

_CANDIDATE_FIELDS = (
    "problem_id",
    "spec_id",
    "generation",
    "generation_index",
    "candidate_slot",
    "candidate_role",
    "parent_spec_ids",
    "seed",
    "training_seed",
    "evaluation_seed",
    "timestamp",
    "status",
    "training_success",
    "failure_reason",
    "formal_status",
    "generation_rank",
    "proxy_rank",
    "fidelity",
    "evaluation_stage",
    "generation_mode",
    "source_low_fidelity_candidate_id",
    "source_generation_id",
    "source_low_fidelity_proxy_mse",
    "low_fidelity_rank",
    "high_fidelity_rank",
    "rank_change",
    "proposal_strategy",
    "changed_modules",
    "budget",
    "constraint_breakdown",
    "solution_valid",
    "solution_collapse_detected",
    "top3_eligible",
    "design_hypothesis",
    "variation_reason",
    "proposal_metadata",
    "generation_policy_audit",
    "population_role",
    "optimization_budget_audit",
    "metric_contract_hash",
    "evaluation_grid_hash",
    "runtime_policy_hash",
    "optimizer_phase_signature",
    "cross_fidelity_audit",
    "high_fidelity_model_selection_audit",
    "adaptive_policy_spec",
    "adaptive_policy_warnings",
    "low_fidelity_proxy_mse",
    "low_fidelity_proxy_complete",
    "low_fidelity_proxy_exact",
    "low_fidelity_proxy_status",
    "low_fidelity_proxy_observations",
)

_CANDIDATE_INDEX_FIELDS = (
    "problem_id",
    "spec_id",
    "generation",
    "candidate_slot",
    "candidate_role",
    "fidelity",
    "status",
    "training_success",
    "failure_reason",
    "generation_rank",
    "proxy_rank",
    "low_fidelity_rank",
    "high_fidelity_rank",
    "proxy_mse",
    "selected_model_mse",
    "last_checkpoint_mse",
    "low_fidelity_proxy_mse",
)


def normalize_output_detail(value: Any) -> str:
    """Return a validated output detail level."""

    detail = str(value or "metrics").strip().casefold()
    if detail not in OUTPUT_DETAILS:
        choices = ", ".join(sorted(OUTPUT_DETAILS))
        raise ValueError(f"output_detail must be one of: {choices}")
    return detail


def resolve_history_interval(
    total_iterations: int,
    *,
    output_detail: str = "compact",
    configured_interval: int | None = None,
) -> int:
    """Resolve history cadence without changing optimizer iteration counts."""

    detail = normalize_output_detail(output_detail)
    configured = max(1, int(configured_interval or 1))
    if detail == "diagnostic":
        return configured
    compact_interval = max(
        1,
        math.ceil(max(0, int(total_iterations)) / MAXIMUM_REGULAR_HISTORY_RECORDS),
    )
    return max(configured, compact_interval)


def compact_training_history(
    history: Iterable[dict[str, Any]],
    *,
    maximum_regular_records: int = MAXIMUM_REGULAR_HISTORY_RECORDS,
) -> list[dict[str, Any]]:
    """Downsample ordinary points while retaining phase and exceptional events."""

    source = [dict(item) for item in history if isinstance(item, dict)]
    if not source:
        return []
    source.sort(key=lambda item: int(item.get("iteration") or 0))

    phase_first: dict[str, int] = {}
    phase_last: dict[str, int] = {}
    dynamic_update_indices: set[int] = set()
    previous_effective_weights: Any = None
    for index, item in enumerate(source):
        phase = str(item.get("phase") or "")
        phase_first.setdefault(phase, index)
        phase_last[phase] = index
        dynamic = item.get("dynamic_weighting")
        effective = item.get("effective_loss_weights")
        explicitly_updated = isinstance(dynamic, dict) and dynamic.get("updated") is True
        weights_changed = bool(effective) and (
            previous_effective_weights is not None
            and effective != previous_effective_weights
        )
        if explicitly_updated or weights_changed:
            dynamic_update_indices.add(index)
        if effective:
            previous_effective_weights = deepcopy(effective)

    forced = {0, len(source) - 1, *phase_first.values(), *phase_last.values()}
    for index, item in enumerate(source):
        if _is_exceptional_history_record(item) or index in dynamic_update_indices:
            forced.add(index)

    regular = [index for index in range(len(source)) if index not in forced]
    regular_limit = max(0, int(maximum_regular_records))
    if len(regular) > regular_limit:
        stride = math.ceil(len(regular) / regular_limit) if regular_limit else len(regular) + 1
        regular = regular[::stride][:regular_limit]
    selected = sorted(forced | set(regular))

    compacted: list[dict[str, Any]] = []
    for index in selected:
        compacted.append(
            _compact_history_entry(
                source[index],
                retain_dynamic=index in dynamic_update_indices,
            )
        )
    return compacted


def history_with_required_events(
    report: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Add an audit-only non-finite event when training stopped off cadence."""

    source = dict(report or {})
    history = [
        dict(item)
        for item in source.get("history") or []
        if isinstance(item, dict)
    ]
    has_non_finite = any(
        str(item.get("nan_inf_status") or "finite").casefold() != "finite"
        for item in history
    )
    if bool(source.get("nan_detected")) and not has_non_finite:
        history.append(
            {
                "iteration": int(source.get("actual_iterations") or 0) + 1,
                "phase": (
                    str(history[-1].get("phase"))
                    if history
                    else "training_failure"
                ),
                "elapsed_seconds": source.get("training_time"),
                "total_loss": None,
                "raw_losses": {},
                "nan_inf_status": "non_finite",
                "event_type": str(
                    source.get("failure_reason") or "non_finite_training_event"
                ),
            }
        )
    return history


def compact_training_report(
    report: dict[str, Any] | None,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return the candidate-record training summary without embedded history."""

    source = dict(report or {})
    keys = (
        "success",
        "actual_iterations",
        "phase_iterations",
        "training_time",
        "final_loss",
        "nan_detected",
        "stopped_early",
        "failure_reason",
        "peak_gpu_memory",
        "actual_peak_sampling_points",
        "adaptive_control_summary",
        "adaptive_compute_budget_summary",
        "runtime_scaled_policy_summary",
        "best_reference_mse_checkpoint",
        "reference_mse_model_selection_enabled",
        "best_training_loss_checkpoint",
        "final_model_policy",
        "selected_checkpoint_source",
        "selected_checkpoint_step",
        "selected_checkpoint_metric",
        "pre_lbfgs_physics_checkpoint",
        "best_lbfgs_physics_checkpoint",
    )
    compact = {key: json_safe(source.get(key)) for key in keys if key in source}
    static_weights = _static_loss_weights(source.get("history") or history)
    if static_weights:
        compact["static_loss_weights"] = static_weights
    compact["history_file"] = "training_history.jsonl"
    compact["history_record_count"] = len(history)
    return compact


def compact_final_metrics(
    metrics: dict[str, Any] | None,
    training_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Canonicalize scalar metrics and replace long diagnostics by summaries."""

    source = dict(metrics or {})
    report = dict(training_report or {})
    grouped = source.get("final_metrics")
    required_groups = {
        "primary_metric",
        "solution_metrics",
        "physics_metrics",
        "constraint_metrics",
        "conservation_metrics",
        "resource_metrics",
        "case_study",
    }
    if isinstance(grouped, dict) and required_groups.issubset(grouped):
        result = {key: json_safe(grouped[key]) for key in required_groups}
        for field in (
            "mse",
            "proxy_mse",
            "selected_model_mse",
            "last_checkpoint_mse",
        ):
            if source.get(field) is not None:
                result[field] = json_safe(source[field])
        resources = dict(result.get("resource_metrics") or {})
        resources.setdefault("wall_clock_time", report.get("training_time"))
        resources.setdefault("peak_gpu_memory", report.get("peak_gpu_memory", 0))
        result["resource_metrics"] = resources
        return result
    compact: dict[str, Any] = {}
    for field in CANONICAL_METRIC_FIELDS:
        value = source.get(field)
        if value is None:
            for alias in _METRIC_FALLBACKS.get(field, ()):
                if source.get(alias) is not None:
                    value = source.get(alias)
                    break
        if value is None and field in report:
            value = report.get(field)
        if value is not None or field in source or field in report:
            compact[field] = json_safe(value)

    if "peak_gpu_memory" not in compact and "peak_memory" in source:
        compact["peak_gpu_memory"] = json_safe(source.get("peak_memory"))
    temporal = temporal_summary(source)
    if temporal:
        compact["temporal_summary"] = temporal
    residual = compact_residual_summary(source)
    if residual:
        compact["residual_distribution"] = residual
    return compact


def temporal_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    """Summarize full temporal arrays using the values used by evaluation."""

    result: dict[str, Any] = {}
    definitions = {
        "l2re": ("temporal_l2re", "l2re"),
        "mse": ("temporal_mse", "mse"),
        "amplitude_ratio": ("temporal_amplitude_ratio", "amplitude_ratio"),
        "phase_error": ("temporal_phase_error", "phase_error"),
    }
    for output_name, (field, value_key) in definitions.items():
        points = _finite_temporal_points(metrics.get(field), value_key)
        if not points:
            continue
        values = [value for _, value in points]
        if output_name == "l2re":
            result[output_name] = {
                "mean": sum(values) / len(values),
                "median": _quantile(values, 0.50),
                "p90": _quantile(values, 0.90),
                "maximum": max(values),
                "worst_time": max(points, key=lambda item: item[1])[0],
            }
        elif output_name in {"mse", "phase_error"}:
            result[output_name] = {
                "mean": sum(values) / len(values),
                "maximum": max(values),
                "worst_time": max(points, key=lambda item: item[1])[0],
            }
        else:
            worst = max(points, key=lambda item: abs(item[1] - 1.0))
            result[output_name] = {
                "mean": sum(values) / len(values),
                "minimum": min(values),
                "maximum": max(values),
                "worst_time": worst[0],
            }
    return result


def temporal_diagnostics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Return only the complete temporal arrays for diagnostic mode."""

    return {
        field: json_safe(metrics.get(field))
        for field in TEMPORAL_FIELDS
        if metrics.get(field) is not None
    }


def residual_diagnostics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Return full spatial partitions for diagnostic mode."""

    payload: dict[str, Any] = {}
    if metrics.get("residual_spatial_summary") is not None:
        payload["residual_spatial_summary"] = json_safe(
            metrics["residual_spatial_summary"]
        )
    return payload


def compact_residual_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    """Merge distribution scalars with at most three localized regions."""

    distributions = dict(
        metrics.get("residual_distribution")
        or metrics.get("governing_residual_distribution")
        or {}
    )
    spatial = dict(metrics.get("residual_spatial_summary") or {})
    equations = dict(spatial.get("equations") or {})
    compact: dict[str, Any] = {}
    allowed = (
        "sample_count",
        "mean_absolute",
        "median_absolute",
        "p90_absolute",
        "p95_absolute",
        "p99_absolute",
        "maximum_absolute",
    )
    for equation_id in sorted(set(distributions) | set(equations)):
        distribution = dict(distributions.get(equation_id) or {})
        equation = dict(equations.get(equation_id) or {})
        item = {
            key: json_safe(distribution.get(key))
            for key in allowed
            if key in distribution
        }
        if "localized_error" in equation:
            item["localized_error"] = json_safe(equation.get("localized_error"))
        if equation.get("top_regions"):
            item["top_regions"] = json_safe(list(equation["top_regions"])[:3])
        if item:
            compact[str(equation_id)] = item
    return compact


def compact_candidate_record(
    record: dict[str, Any],
    *,
    metrics: dict[str, Any],
    training_report: dict[str, Any],
    artifacts: dict[str, Any],
) -> dict[str, Any]:
    """Build the non-duplicating public candidate record."""

    compact = {
        key: json_safe(record.get(key))
        for key in _CANDIDATE_FIELDS
        if key in record
    }
    compact["training_report"] = json_safe(training_report)
    compact["artifacts"] = json_safe(artifacts)
    return compact


def compact_candidate_index(record: dict[str, Any]) -> dict[str, Any]:
    """Return the lightweight root index entry for one candidate.

    The candidate directory is the sole detailed source.  This index is only
    for discovery, status inspection, and ranking without recursively scanning
    every JSON document.
    """

    candidate_id = str(record.get("spec_id") or "unknown_candidate")
    compact = {
        key: json_safe(record.get(key))
        for key in _CANDIDATE_INDEX_FIELDS
        if key in record
    }
    compact["candidate_record"] = (
        f"runtime_candidates/{candidate_id}/candidate_record.json"
    )
    return compact


def load_runtime_candidate_records(output_dir: str | Path) -> list[dict[str, Any]]:
    """Hydrate resume records from their unique per-candidate artifacts.

    New runs keep only candidate IDs in ``search_state.json``.  This loader also
    accepts older candidate records that embedded metrics or history.
    """

    root = Path(output_dir) / "runtime_candidates"
    if not root.is_dir():
        return []
    records: list[dict[str, Any]] = []
    for candidate_dir in sorted(item for item in root.iterdir() if item.is_dir()):
        record_path = candidate_dir / "candidate_record.json"
        if not record_path.is_file():
            continue
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(record, dict):
            continue
        artifacts = dict(record.get("artifacts") or {})

        def read_artifact(name: str, fallback_name: str) -> Any:
            reference = artifacts.get(name)
            relative = reference.get("path") if isinstance(reference, dict) else None
            path = candidate_dir / str(relative or fallback_name)
            if not path.is_file():
                return None
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None

        metrics = record.get("metrics")
        if not isinstance(metrics, dict):
            metrics = read_artifact("final_metrics", "final_metrics.json") or {}
        runtime_spec = read_artifact(
            "runtime_spec", "algorithm_spec_normalized.json"
        )
        if runtime_spec is None:
            runtime_spec = read_artifact(
                "normalized_spec", "algorithm_spec_normalized.json"
            ) or {}
        search_spec = read_artifact("search_spec", "algorithm_spec_search.json")
        if search_spec is None:
            search_spec = runtime_spec
        raw_spec = read_artifact("raw_spec", "algorithm_spec_raw.json")
        if raw_spec is None:
            raw_spec = search_spec
        validation = read_artifact("validation_report", "validation_report.json")
        training_report = dict(record.get("training_report") or {})
        if "history" not in training_report:
            training_report["history"] = load_candidate_training_history(
                record, base_dir=candidate_dir
            )
        hydrated = dict(record)
        hydrated.update(
            {
                "metrics": metrics,
                "generation_metrics": metrics,
                "training_report": training_report,
                "generation_training_report": training_report,
                "normalized_algorithm_spec": runtime_spec,
                "algorithm_spec": runtime_spec,
                "search_algorithm_spec": search_spec,
                "algorithm_spec_raw": raw_spec,
                "validation_report": validation or {},
            }
        )
        # Ranking and checkpoint scalars have a single durable source in
        # final_metrics.json.  Recreate the legacy top-level view in memory so
        # resume callers stay independent of the artifact schema version.
        for field in (
            "proxy_mse",
            "selected_model_mse",
            "last_checkpoint_mse",
        ):
            if hydrated.get(field) is None and metrics.get(field) is not None:
                hydrated[field] = metrics[field]
        records.append(hydrated)
    return records


def artifact_reference(path: Path, *, include_hash: bool = True) -> dict[str, str]:
    """Return a candidate-local path and optional SHA-256."""

    reference = {"path": path.name}
    if include_hash:
        reference["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return reference


def write_json(path: Path, payload: Any) -> None:
    """Write compact machine-readable JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            json_safe(payload),
            handle,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            default=str,
        )


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    """Write one compact JSON object per line."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in records:
            handle.write(
                json.dumps(
                    json_safe(item),
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                    default=str,
                )
                + "\n"
            )


def load_candidate_training_history(
    candidate: dict[str, Any] | str | Path,
    *,
    base_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Load new external history first, with legacy embedded-history fallback."""

    if isinstance(candidate, (str, Path)):
        candidate_path = Path(candidate)
        record = json.loads(candidate_path.read_text(encoding="utf-8"))
        root = candidate_path.parent
    else:
        record = dict(candidate)
        root = Path(base_dir) if base_dir is not None else Path.cwd()
    report = dict(record.get("training_report") or {})
    history_file = report.get("history_file")
    if history_file:
        path = root / str(history_file)
        if path.exists():
            return [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
    return [
        dict(item)
        for item in report.get("history") or []
        if isinstance(item, dict)
    ]


def _compact_history_entry(
    item: dict[str, Any],
    *,
    retain_dynamic: bool,
) -> dict[str, Any]:
    raw_losses = item.get("raw_losses") or item.get("raw_loss_components") or {}
    compact = {
        "iteration": item.get("iteration"),
        "phase": item.get("phase"),
        "elapsed_seconds": item.get("elapsed_seconds"),
        "total_loss": item.get("total_loss"),
        "raw_losses": json_safe(raw_losses),
        "learning_rate": item.get("learning_rate"),
        "gradient_norm": item.get("gradient_norm"),
        "mean_residual": item.get("mean_residual"),
        "maximum_residual": item.get("maximum_residual"),
        "sampling_point_count": item.get("sampling_point_count"),
        "nan_inf_status": item.get("nan_inf_status", "finite"),
    }
    compact = {key: json_safe(value) for key, value in compact.items() if value is not None}
    conditional = (
        "adaptive_refinement_event",
        "early_stopping",
        "checkpoint_iteration",
        "event_type",
        "gradient_statistics",
        "reference_mse",
        "reference_mse_probe_error",
    )
    for key in conditional:
        value = item.get(key)
        if value not in (None, False, "", {}, []):
            compact[key] = json_safe(value)
    stage = item.get("training_stage")
    if isinstance(stage, dict):
        filtered_stage = {
            key: value
            for key, value in stage.items()
            if value not in (None, False, "", {}, [])
            and not (key == "time_window_fraction" and value in (1, 1.0))
        }
        if filtered_stage:
            compact["training_stage"] = json_safe(filtered_stage)
    elif stage not in (None, False, "", {}, []):
        compact["training_stage"] = json_safe(stage)
    fraction = item.get("time_window_fraction")
    if fraction not in (None, 1, 1.0):
        compact["time_window_fraction"] = json_safe(fraction)
    if retain_dynamic:
        if item.get("effective_loss_weights"):
            compact["effective_loss_weights"] = json_safe(
                item["effective_loss_weights"]
            )
        if item.get("dynamic_weighting"):
            compact["dynamic_weighting"] = json_safe(item["dynamic_weighting"])
    return compact


def _is_exceptional_history_record(item: dict[str, Any]) -> bool:
    status = str(item.get("nan_inf_status") or "finite").casefold()
    return (
        status != "finite"
        or bool(item.get("adaptive_refinement_event"))
        or bool(item.get("early_stopping"))
        or bool(item.get("checkpoint_iteration"))
        or bool(item.get("event_type"))
        or bool(item.get("reference_mse_probe_error"))
    )


def _static_loss_weights(history: Iterable[dict[str, Any]]) -> dict[str, Any]:
    for item in history:
        if not isinstance(item, dict):
            continue
        if item.get("dynamic_weighting"):
            return {}
        weights = item.get("effective_loss_weights")
        if weights:
            return json_safe(dict(weights))
    return {}


def _finite_temporal_points(payload: Any, value_key: str) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for item in payload or []:
        if not isinstance(item, dict):
            continue
        time_value = item.get("time")
        value = item.get(value_key)
        if (
            isinstance(time_value, (int, float))
            and not isinstance(time_value, bool)
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(time_value))
            and math.isfinite(float(value))
        ):
            points.append((float(time_value), float(value)))
    return points


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction
