"""Search-time contracts for adaptive Trainer policies.

The three public layers in this module are deliberately separate:

* an AlgorithmSpec describes an inheritable design;
* a RuntimeScaledTrainingPolicy is a fidelity-specific execution view;
* training summaries describe observed behaviour and are never inherited.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any, Mapping


_OPTIMIZER_STEP_FIELDS: tuple[tuple[str, ...], ...] = (
    ("loss", "weighting_strategy", "parameters", "diagnostic_interval"),
    ("loss", "weighting_strategy", "parameters", "cooldown_iterations"),
    ("training", "extra_parameters", "adaptive_control", "lbfgs_stall_recovery", "minimum_completed_iterations_before_detection"),
    ("training", "extra_parameters", "adaptive_control", "lbfgs_stall_recovery", "patience"),
    ("training", "extra_parameters", "adaptive_control", "lbfgs_stall_recovery", "maximum_recovery_iterations"),
    ("training", "extra_parameters", "adaptive_control", "lbfgs_stall_recovery", "minimum_recovery_iterations"),
    ("training", "extra_parameters", "adaptive_control", "sampling_control", "diagnostic_interval"),
    ("training", "extra_parameters", "adaptive_control", "sampling_control", "minimum_iterations_before_control"),
    ("training", "extra_parameters", "adaptive_control", "sampling_control", "cooldown_iterations"),
    ("training", "extra_parameters", "adaptive_control", "sampling_control", "observation_window_iterations"),
    ("training", "extra_parameters", "adaptive_control", "curriculum_control", "minimum_iterations_per_level"),
    ("training", "extra_parameters", "adaptive_control", "curriculum_control", "maximum_iterations_per_level"),
    ("training", "extra_parameters", "adaptive_control", "curriculum_control", "validation_interval"),
    ("training", "extra_parameters", "adaptive_control", "curriculum_control", "cooldown_iterations"),
    ("training", "extra_parameters", "adaptive_control", "curriculum_control", "minimum_remaining_budget_after_advance"),
    ("training", "extra_parameters", "adaptive_control", "curriculum_control", "observation_window_iterations"),
    ("training", "extra_parameters", "physics_validation", "evaluation_interval"),
)

_ZERO_ALLOWED_FIELDS = {
    "cooldown_iterations",
    "minimum_iterations_per_level",
    "minimum_iterations_before_control",
    "minimum_remaining_budget_after_advance",
    "minimum_recovery_iterations",
}

_SAMPLING_BUDGET_FIELDS: tuple[tuple[str, ...], ...] = (
    ("training", "extra_parameters", "physics_validation", "points_per_component"),
    ("training", "extra_parameters", "physics_validation", "maximum_total_points"),
    ("training", "extra_parameters", "adaptive_control", "sampling_control", "maximum_candidate_points_evaluated"),
)

_UNCHANGED_ADAPTIVE_FIELDS: tuple[tuple[str, ...], ...] = (
    ("loss", "weighting_strategy", "name"),
    ("loss", "weighting_strategy", "parameters", "ema_beta"),
    ("loss", "weighting_strategy", "parameters", "minimum_weight"),
    ("loss", "weighting_strategy", "parameters", "maximum_weight"),
    ("loss", "weighting_strategy", "parameters", "minimum_update_ratio"),
    ("loss", "weighting_strategy", "parameters", "maximum_update_ratio"),
    ("loss", "weighting_strategy", "parameters", "conflict_threshold"),
    ("training", "extra_parameters", "adaptive_control", "sampling_control", "replacement_fraction"),
    ("training", "extra_parameters", "adaptive_control", "sampling_control", "hard_point_retention_fraction"),
    ("training", "extra_parameters", "adaptive_control", "sampling_control", "random_exploration_fraction"),
    ("training", "extra_parameters", "adaptive_control", "sampling_control", "candidate_multiplier"),
    ("training", "extra_parameters", "adaptive_control", "curriculum_control", "minimum_relative_improvement"),
    ("training", "extra_parameters", "adaptive_control", "curriculum_control", "normal_advance_condition"),
    ("training", "extra_parameters", "adaptive_control", "curriculum_control", "forced_advance_policy"),
    ("training", "extra_parameters", "adaptive_control", "curriculum_control", "post_advance_controller_ema_decay"),
    ("training", "extra_parameters", "best_checkpoint", "final_model_policy"),
)


def _path_name(path: tuple[str, ...]) -> str:
    if path[:3] == ("training", "extra_parameters", "adaptive_control"):
        return ".".join(path[3:])
    if path[:2] == ("training", "extra_parameters"):
        return ".".join(path[2:])
    return ".".join(path)


def _get(mapping: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = mapping
    for part in path:
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _set(mapping: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    current = mapping
    for part in path[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    current[path[-1]] = value


def adaptive_policy_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Return the deterministic, inheritable adaptive design only."""

    loss = dict(spec.get("loss") or {})
    training = dict(spec.get("training") or {})
    extra = dict(training.get("extra_parameters") or {})
    result = {
        "loss_weighting": deepcopy(loss.get("weighting_strategy") or {"name": "fixed", "parameters": {}}),
        "adaptive_control": deepcopy(extra.get("adaptive_control") or {"enabled": False}),
        "physics_validation": deepcopy(extra.get("physics_validation") or {"enabled": False, "strategy": "disabled"}),
        "best_checkpoint": deepcopy(extra.get("best_checkpoint") or {"enabled": False, "final_model_policy": "last"}),
    }
    return _sorted(result)


@dataclass(frozen=True)
class RuntimeScaledTrainingPolicy:
    runtime_spec: dict[str, Any]
    audit: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"runtime_spec": deepcopy(self.runtime_spec), "audit": deepcopy(self.audit)}


class AdaptivePolicyBudgetScaler:
    """Create a fidelity execution view without mutating the normalized design."""

    def scale(
        self,
        normalized_spec: Mapping[str, Any],
        *,
        source_fidelity: str,
        target_fidelity: str,
        source_optimizer_step_budget: int,
        target_optimizer_step_budget: int,
        source_sampling_budget: int | None = None,
        target_sampling_budget: int | None = None,
        source_validation_budget: int | None = None,
        target_validation_budget: int | None = None,
        experiment_contract: Mapping[str, Any] | None = None,
        runtime_base_spec: Mapping[str, Any] | None = None,
    ) -> RuntimeScaledTrainingPolicy:
        source = deepcopy(dict(normalized_spec))
        runtime = deepcopy(dict(runtime_base_spec or normalized_spec))
        source_steps = max(1, int(source_optimizer_step_budget))
        target_steps = max(1, int(target_optimizer_step_budget))
        optimizer_ratio = target_steps / source_steps
        source_sampling = max(1, int(source_sampling_budget or _sampling_point_budget(source) or 1))
        target_sampling = max(1, int(target_sampling_budget or source_sampling))
        sampling_ratio = target_sampling / source_sampling
        source_validation = max(1, int(source_validation_budget or source_sampling))
        target_validation = max(1, int(target_validation_budget or target_sampling))
        validation_ratio = target_validation / source_validation
        scaled: dict[str, Any] = {}
        unchanged: dict[str, Any] = {}
        clamped: dict[str, Any] = {}

        for path in _OPTIMIZER_STEP_FIELDS:
            original = _get(source, path)
            if isinstance(original, bool) or not isinstance(original, int):
                continue
            minimum = 0 if path[-1] in _ZERO_ALLOWED_FIELDS else 1
            raw = int(round(original * optimizer_ratio))
            value = max(minimum, min(target_steps, raw))
            _set(runtime, path, value)
            item = {"original": original, "scaled": value, "basis": "optimizer_steps"}
            scaled[_path_name(path)] = item
            if value != raw:
                clamped[_path_name(path)] = {**item, "unclamped": raw, "reason": "target_optimizer_budget"}

        for path in _SAMPLING_BUDGET_FIELDS:
            original = _get(source, path)
            if isinstance(original, bool) or not isinstance(original, int):
                continue
            basis = "validation_points" if "physics_validation" in path else "sampling_points"
            ratio = validation_ratio if basis == "validation_points" else sampling_ratio
            maximum = target_validation if basis == "validation_points" else target_sampling
            raw = int(round(original * ratio))
            value = max(1, min(maximum, raw))
            _set(runtime, path, value)
            item = {"original": original, "scaled": value, "basis": basis}
            scaled[_path_name(path)] = item
            if value != raw:
                clamped[_path_name(path)] = {**item, "unclamped": raw, "reason": f"target_{basis}_budget"}

        # Preserve controller invariants after independent integer rounding.
        curriculum = _get(runtime, ("training", "extra_parameters", "adaptive_control", "curriculum_control"))
        if (
            isinstance(curriculum, dict)
            and curriculum.get("enabled") is True
            and curriculum.get("strategy") != "none"
        ):
            minimum = int(curriculum.get("minimum_iterations_per_level", 0))
            maximum = int(curriculum.get("maximum_iterations_per_level", 0))
            if maximum <= minimum:
                fixed = min(target_steps, minimum + 1)
                curriculum["maximum_iterations_per_level"] = fixed
                clamped["curriculum_control.maximum_iterations_per_level"] = {
                    "original": maximum, "scaled": fixed, "reason": "must_exceed_minimum_level_residence"
                }
            interval = max(1, int(curriculum.get("validation_interval", 1)))
            checks = max(1, int(curriculum.get("minimum_validation_checks", 1)))
            required_window = interval * checks
            if int(curriculum.get("observation_window_iterations", 0)) < required_window:
                fixed = min(target_steps, required_window)
                curriculum["observation_window_iterations"] = fixed
                clamped["curriculum_control.observation_window_iterations"] = {
                    "scaled": fixed, "reason": "minimum_validation_checks"
                }

        for path in _UNCHANGED_ADAPTIVE_FIELDS:
            value = _get(source, path)
            if value is not None:
                unchanged[_path_name(path)] = {
                    "value": deepcopy(value),
                    "reason": "dimensionless_ratio_or_policy_identity",
                }

        audit = {
            "source_fidelity": str(source_fidelity),
            "target_fidelity": str(target_fidelity),
            "source_optimizer_step_budget": source_steps,
            "target_optimizer_step_budget": target_steps,
            "optimizer_budget_ratio": optimizer_ratio,
            "sampling_budget_ratio": sampling_ratio,
            "validation_budget_ratio": validation_ratio,
            "scaled_fields": _sorted(scaled),
            "unchanged_fields": _sorted(unchanged),
            "clamped_fields": _sorted(clamped),
            "original_spec_unchanged": source == dict(normalized_spec),
            "experiment_contract_version": (experiment_contract or {}).get("version")
            or (experiment_contract or {}).get("experiment_contract_version"),
        }
        return RuntimeScaledTrainingPolicy(_sorted(runtime), _sorted(audit))


def build_trainer_capability_summary(
    problem: Any,
    builder_capabilities: Mapping[str, Any],
    *,
    registry: Any | None = None,
) -> dict[str, Any]:
    """Build a compact, reference-free capability preflight for LLM design."""

    if registry is not None:
        capability = registry.capabilities.to_dict()
        loss_count = len(registry.loss_components)
        weightable = sum(bool(item.dynamically_weightable) for item in registry.loss_components)
        sampling_count = len(registry.sampling_components)
        replaceable = sum(bool(item.replacement_supported) for item in registry.sampling_components)
        curriculum_count = len(registry.curricula)
        recoverable = sum(bool(item.recoverable) for item in registry.optimization_stages)
        source = "built_training_component_registry"
    else:
        problem_spec = problem.get_spec()
        constraints = [item for item in problem_spec.constraints if not bool((item.metadata or {}).get("aggregate"))]
        loss_count = max(1, len(problem_spec.governing_laws)) + len(constraints)
        weightable = loss_count
        sampling_roles = {str(item.constraint_type) for item in constraints}
        sampling_count = 1 + len(sampling_roles)
        replaceable = 1 if problem_spec.input_variables else 0
        declared_roles = dict(problem_spec.domain.metadata.get("variable_roles") or {})
        temporal = any(
            (value == "temporal") or (isinstance(value, Mapping) and value.get("role") == "temporal")
            for value in declared_roles.values()
        ) or bool(problem_spec.metadata.get("time_dependent", False))
        runtime_factory = callable(getattr(problem, "build_curriculum_runtimes", None))
        curriculum_count = int(runtime_factory or temporal)
        optimizers = set(builder_capabilities.get("optimizers") or ())
        recoverable = int("lbfgs" in optimizers and bool(optimizers - {"lbfgs"}))
        probes = bool(loss_count and sampling_count)
        capability = {
            "supports_dynamic_loss_weighting": loss_count >= 2,
            "supports_stage_reallocation": bool(recoverable),
            "supports_resampling": bool(replaceable),
            "supports_curriculum": bool(curriculum_count),
            "supports_curriculum_advance": bool(curriculum_count),
            "supports_curriculum_rollback": bool(curriculum_count),
            "supports_forced_curriculum_advance": bool(curriculum_count),
            "supports_validation_probes": probes,
            "supports_component_gradient_diagnostics": loss_count >= 2,
        }
        source = "lightweight_builder_preflight"

    eligible = {
        "loss_weighting": ["fixed", "lra"] + (["gradient_balance"] if capability.get("supports_dynamic_loss_weighting") else []),
        "sampling_control": ["none"] + (["fixed_budget_subset_replacement"] if capability.get("supports_resampling") else []),
        "curriculum_control": ["none"] + (["residual_gated_level_advance"] if capability.get("supports_curriculum_advance") else []),
        "lbfgs_recovery": ["disabled"] + (["terminate_candidate", "reallocate_to_adam"] if capability.get("supports_stage_reallocation") else []),
        "physics_validation": ["disabled"] + (["fixed_component_probe"] if capability.get("supports_validation_probes") else []),
        "final_model_policy": ["last", "best_train_loss"],
    }
    summary = {
        "schema_version": "1.0",
        "source": source,
        "trainer_capabilities": {
            "supports_dynamic_loss_weighting": bool(capability.get("supports_dynamic_loss_weighting")),
            "supports_lbfgs_stall_recovery": bool(capability.get("supports_stage_reallocation")),
            "supports_sampling_replacement": bool(capability.get("supports_resampling")),
            "supports_curriculum_advance": bool(capability.get("supports_curriculum_advance")),
            "supports_physics_validation": bool(capability.get("supports_validation_probes")),
        },
        "component_summary": {
            "loss_component_count": int(loss_count),
            "dynamically_weightable_component_count": int(weightable),
            "sampling_component_count": int(sampling_count),
            "replaceable_sampling_component_count": int(replaceable),
            "curriculum_runtime_count": int(curriculum_count),
            "recoverable_stage_count": int(recoverable),
        },
        "eligible_adaptive_policies": eligible,
        "forbidden_information": ["coordinates", "reference_metrics", "evaluator_outputs", "runtime_objects"],
    }
    return _sorted(summary)


def _sampling_point_budget(spec: Mapping[str, Any]) -> int:
    sampling = dict(spec.get("sampling") or {})
    total = 0
    for name in ("interior", "boundary", "initial"):
        value = ((sampling.get(name) or {}).get("parameters") or {}).get("n_points", 0)
        if isinstance(value, int) and not isinstance(value, bool):
            total += max(0, value)
    return total


def _sorted(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _sorted(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, list):
        return [_sorted(item) for item in value]
    if isinstance(value, tuple):
        return [_sorted(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Adaptive policy data must be finite")
    return deepcopy(value)


__all__ = [
    "AdaptivePolicyBudgetScaler",
    "RuntimeScaledTrainingPolicy",
    "adaptive_policy_spec",
    "build_trainer_capability_summary",
]
