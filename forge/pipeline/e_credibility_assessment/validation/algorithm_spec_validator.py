"""Deterministic static validation for the Open AlgorithmSpec."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from copy import deepcopy
import math
from typing import Any

from forge.pipeline.d_algorithm_generation.specs import REQUIRED_TOP_LEVEL_FIELDS
from forge.pipeline.a_problem_definition.problems.sampling_budget import (
    SamplingContractError,
    resolve_sampling_budget_plan,
)


ARCHITECTURE_ALIASES = {"fnn": "mlp"}
ACTIVATION_ALIASES = {"swish": "silu"}
INITIALIZER_ALIASES = {
    "he_normal": "kaiming_normal",
    "he_uniform": "kaiming_uniform",
}
WEIGHTING_ALIASES = {"none": "fixed"}
OPTIMIZER_ALIASES = {"l_bfgs": "lbfgs"}
GOVERNING_LOSS_TERM_NAMES = frozenset(
    {"pde_residual", "governing_residual", "pde_component"}
)


@dataclass(frozen=True)
class ValidationIssue:
    path: str
    code: str
    message: str


@dataclass
class ValidationReport:
    valid: bool
    errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)
    raw_spec: dict[str, Any] | None = None
    normalized_spec: dict[str, Any] | None = None
    budget_usage: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AlgorithmSpecValidator:
    """Owns raw structure checks, conservative normalization, and constraints."""

    def validate(self, spec: dict, problem: Any, experiment_config: dict, builder_capabilities: dict) -> ValidationReport:
        return validate_algorithm_spec(spec, problem, experiment_config, builder_capabilities)


def normalize_algorithm_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Normalize representation only; never tune or repair algorithm values."""

    normalized = deepcopy(spec)
    network = normalized["network"]
    architecture = _normalized_name(network["architecture"])
    network["architecture"] = ARCHITECTURE_ALIASES.get(architecture, architecture)
    network.setdefault("extra_parameters", {})
    for field_name in ("activation", "output_activation", "initialization", "input_transform"):
        component = network[field_name]
        component["name"] = _normalized_name(component["name"])
        component.setdefault("parameters", {})
    network["activation"]["name"] = ACTIVATION_ALIASES.get(network["activation"]["name"], network["activation"]["name"])
    network["initialization"]["name"] = INITIALIZER_ALIASES.get(
        network["initialization"]["name"], network["initialization"]["name"]
    )
    residual = network["residual_connections"]
    residual.setdefault("parameters", {})
    if not residual["enabled"]:
        residual["parameters"] = {}

    sampling = normalized["sampling"]
    for section in ("interior", "boundary", "initial"):
        sampling[section]["strategy"] = _normalized_name(sampling[section]["strategy"])
        sampling[section].setdefault("parameters", {})
    adaptive = sampling["adaptive_refinement"]
    adaptive["strategy"] = _normalized_name(adaptive["strategy"])
    adaptive.setdefault("parameters", {})
    if not adaptive["enabled"]:
        adaptive["parameters"] = {}

    enforcement = normalized.setdefault(
        "constraint_enforcement",
        {"method": "soft_penalty", "transform_id": "none", "parameters": {}},
    )
    enforcement["method"] = _normalized_name(enforcement["method"])
    enforcement["transform_id"] = _normalized_name(enforcement["transform_id"])
    enforcement.setdefault("parameters", {})

    loss = normalized["loss"]
    for term in loss["terms"]:
        term["name"] = _normalized_name(term["name"])
        term["loss_function"] = _normalized_name(term["loss_function"])
        term.setdefault("parameters", {})
    for field_name in ("weighting_strategy", "aggregation"):
        loss[field_name]["name"] = _normalized_name(loss[field_name]["name"])
        loss[field_name].setdefault("parameters", {})
    loss["weighting_strategy"]["name"] = WEIGHTING_ALIASES.get(
        loss["weighting_strategy"]["name"], loss["weighting_strategy"]["name"]
    )
    weighting_parameters = loss["weighting_strategy"]["parameters"]
    if loss["weighting_strategy"]["name"] == "gradient_balance":
        if "ema_decay" in weighting_parameters:
            weighting_parameters.setdefault("ema_beta", weighting_parameters.pop("ema_decay"))
        if "minimum_gradient_cosine" in weighting_parameters:
            weighting_parameters.setdefault(
                "conflict_threshold", weighting_parameters.pop("minimum_gradient_cosine")
            )
        if "maximum_single_update_ratio" in weighting_parameters:
            ratio = weighting_parameters.pop("maximum_single_update_ratio")
            weighting_parameters.setdefault("maximum_update_ratio", ratio)
            if isinstance(ratio, (int, float)) and not isinstance(ratio, bool) and ratio > 0:
                weighting_parameters.setdefault("minimum_update_ratio", 1.0 / float(ratio))

    for phase in normalized["optimization"]["phases"]:
        optimizer = _normalized_name(phase["optimizer"])
        phase["optimizer"] = OPTIMIZER_ALIASES.get(optimizer, optimizer)
        phase.setdefault("parameters", {})
        phase["scheduler"]["name"] = _normalized_name(phase["scheduler"]["name"])
        phase["scheduler"].setdefault("parameters", {})

    training = normalized["training"]
    training["batch_mode"] = _normalized_name(training["batch_mode"])
    training.setdefault("extra_parameters", {})
    if "adaptive_control" in training["extra_parameters"]:
        adaptive_control = training["extra_parameters"]["adaptive_control"]
        if isinstance(adaptive_control, dict):
            for section_name in ("lbfgs_stall_recovery", "sampling_control", "curriculum_control"):
                raw_section = adaptive_control.get(section_name)
                if isinstance(raw_section, str):
                    name = _normalized_name(raw_section)
                    if section_name == "lbfgs_stall_recovery":
                        adaptive_control[section_name] = {
                            "enabled": name != "disabled", "action": name
                        }
                    else:
                        adaptive_control[section_name] = {
                            "enabled": name != "none", "strategy": name
                        }
            had_stall_config = "lbfgs_stall_recovery" in adaptive_control
            adaptive_control.setdefault("enabled", False)
            adaptive_control.setdefault("freeze_lbfgs_objective", True)
            stall = adaptive_control.setdefault("lbfgs_stall_recovery", {})
            if isinstance(stall, dict):
                explicitly_selected_stall_policy = any(
                    name in stall
                    for name in ("enabled", "action", "policy", "recovery_policy")
                )
                if "policy" in stall:
                    stall.setdefault("action", stall.pop("policy"))
                if "recovery_policy" in stall:
                    stall.setdefault("action", stall.pop("recovery_policy"))
                if "minimum_iterations_before_detection" in stall:
                    stall.setdefault(
                        "minimum_completed_iterations_before_detection",
                        stall.pop("minimum_iterations_before_detection"),
                    )
                if "recovery_learning_rate_scale" in stall:
                    stall.setdefault(
                        "adam_learning_rate_scale",
                        stall.pop("recovery_learning_rate_scale"),
                    )
                stall_defaults = {
                    "enabled": True,
                    "patience": 20,
                    "minimum_relative_improvement": 1e-4,
                    "minimum_step_size": 1e-8,
                    "minimum_parameter_update_norm": 1e-10,
                    "maximum_repeated_line_search_failures": 10,
                    "minimum_completed_iterations_before_detection": 50,
                    "action": "reallocate_to_adam",
                    "adam_learning_rate_scale": 0.1,
                    "maximum_recovery_iterations": 1000,
                    "minimum_recovery_iterations": 100,
                    "reuse_remaining_budget": True,
                    "return_to_lbfgs": False,
                }
                for name, value in stall_defaults.items():
                    stall.setdefault(name, value)
                has_lbfgs_stage = any(
                    _normalized_name(phase.get("optimizer")) in {"lbfgs", "l_bfgs"}
                    and int(phase.get("iterations") or 0) > 0
                    for phase in normalized["optimization"]["phases"]
                )
                legacy_dormant_stall = (
                    had_stall_config
                    and not explicitly_selected_stall_policy
                    and not has_lbfgs_stage
                )
                if legacy_dormant_stall:
                    # Pre-policy-space specs sometimes carried recovery tuning
                    # beside Adam-only schedules. Preserve their accepted data,
                    # but make the unavailable controller explicitly dormant.
                    stall["enabled"] = False
                    stall["action"] = "disabled"
                elif not stall.get("enabled", False) or stall.get("action") == "disabled":
                    adaptive_control["lbfgs_stall_recovery"] = {
                        "action": "disabled", "enabled": False
                    }
                    stall = adaptive_control["lbfgs_stall_recovery"]
            sampling_control = adaptive_control.get("sampling_control")
            if isinstance(sampling_control, dict):
                sampling_defaults = {
                    "enabled": False,
                    "strategy": "fixed_budget_subset_replacement",
                    "diagnostic_interval": 200,
                    "minimum_iterations_before_control": 500,
                    "patience": 3,
                    "cooldown_iterations": 500,
                    "maximum_updates": 10,
                    "p95_to_median_threshold": 10.0,
                    "replacement_fraction": 0.2,
                    "hard_point_retention_fraction": 0.2,
                    "random_exploration_fraction": 0.1,
                    "candidate_multiplier": 4,
                    "maximum_point_age": 20,
                    "maximum_consecutive_retention": 5,
                    "minimum_point_age": 2,
                    "observation_window_iterations": 300,
                    "minimum_validation_stability_checks": 2,
                    "validation_degradation_ratio": 1.5,
                    "associated_component_degradation_ratio": 2.0,
                    "maximum_candidate_points_evaluated": 100000,
                }
                for name, value in sampling_defaults.items():
                    sampling_control.setdefault(name, value)
                if not sampling_control.get("enabled", False) or sampling_control.get("strategy") == "none":
                    adaptive_control["sampling_control"] = {
                        "enabled": False, "strategy": "none"
                    }
                    sampling_control = adaptive_control["sampling_control"]
            curriculum_control = adaptive_control.get("curriculum_control")
            if isinstance(curriculum_control, dict):
                curriculum_defaults = {
                    "enabled": False,
                    "strategy": "residual_gated_level_advance",
                    "minimum_iterations_per_level": 500,
                    "maximum_iterations_per_level": 3000,
                    "validation_interval": 100,
                    "patience": 3,
                    "cooldown_iterations": 300,
                    "minimum_relative_improvement": 0.001,
                    "maximum_validation_score": None,
                    "maximum_advances": 20,
                    "normal_advance_condition": "physics_validation_stable",
                    "forced_advance_policy": "advance_one_level",
                    "minimum_remaining_budget_after_advance": 500,
                    "observation_window_iterations": 500,
                    "forced_observation_window_multiplier": 2.0,
                    "minimum_validation_checks": 3,
                    "aggregate_degradation_ratio": 1.5,
                    "associated_component_degradation_ratio": 2.0,
                    "critical_component_degradation_ratio": 1.5,
                    "rollback_on_non_finite": True,
                    "post_advance_controller_ema_decay": 0.5,
                }
                for name, value in curriculum_defaults.items():
                    curriculum_control.setdefault(name, value)
                if not curriculum_control.get("enabled", False) or curriculum_control.get("strategy") == "none":
                    adaptive_control["curriculum_control"] = {
                        "enabled": False, "strategy": "none"
                    }
                    curriculum_control = adaptive_control["curriculum_control"]
            bounded_control_enabled = bool(
                isinstance(sampling_control, dict)
                and sampling_control.get("enabled")
            ) or bool(
                isinstance(curriculum_control, dict)
                and curriculum_control.get("enabled")
            )
            if bounded_control_enabled and not had_stall_config and not adaptive_control.get("enabled"):
                adaptive_control["lbfgs_stall_recovery"] = {
                    "action": "disabled",
                    "enabled": False,
                }
            rollback = adaptive_control.setdefault("rollback", {})
            if isinstance(rollback, dict):
                rollback.setdefault("enabled", True)
                rollback.setdefault("loss_degradation_ratio", 2.0)
                rollback.setdefault("patience", 3)
    extra = training["extra_parameters"]
    if isinstance(extra.get("component_control"), dict):
        extra["component_control"].setdefault("enabled", False)
    if isinstance(extra.get("controller_coordinator"), dict):
        coordinator = extra["controller_coordinator"]
        coordinator.setdefault("enabled", True)
        coordinator.setdefault("maximum_non_safety_actions_per_cycle", 1)
        coordinator.setdefault("observation_window_iterations", 100)
        coordinator.setdefault("global_cooldown_iterations", 100)
        coordinator.setdefault("priority", [
            "rollback", "terminate_candidate", "stage_recovery", "loss_weight",
            "learning_rate", "sampling", "curriculum",
        ])
    if isinstance(extra.get("physics_validation"), str):
        strategy = _normalized_name(extra["physics_validation"])
        extra["physics_validation"] = {
            "enabled": strategy != "disabled", "strategy": strategy
        }
    if isinstance(extra.get("physics_validation"), dict):
        physics = extra["physics_validation"]
        defaults = {
            "enabled": False, "strategy": "fixed_component_probe", "evaluation_interval": 100,
            "normalization": "initial_probe_value",
            "category_aggregation": "median_worst_blend", "median_weight": 0.7,
            "worst_weight": 0.3, "minimum_improvement": 5e-3,
            "epsilon": 1e-12, "seed": 1729, "points_per_component": 128,
            "maximum_total_points": 4096, "category_weights": {},
        }
        for name, value in defaults.items():
            physics.setdefault(name, value)
        if not physics.get("enabled", False) or physics.get("strategy") == "disabled":
            extra["physics_validation"] = {
                "enabled": False, "strategy": "disabled"
            }
    if isinstance(extra.get("best_checkpoint"), str):
        policy = _normalized_name(extra["best_checkpoint"])
        extra["best_checkpoint"] = {
            "enabled": policy != "last",
            "final_model_policy": policy,
        }
    if isinstance(extra.get("best_checkpoint"), dict):
        best = extra["best_checkpoint"]
        defaults = {"enabled": False, "final_model_policy": "last"}
        for name, value in defaults.items():
            best.setdefault(name, value)
        if best.get("final_model_policy") == "best_train_loss":
            best.setdefault("selection_metric", "training_loss")
            best.setdefault("evaluation_interval", 100)
        if not best.get("enabled", False) or best.get("final_model_policy") == "last":
            extra["best_checkpoint"] = {
                "enabled": False, "final_model_policy": "last"
            }
    time_strategy = training.setdefault(
        "time_strategy", {"name": "none", "parameters": {}}
    )
    time_strategy["name"] = _normalized_name(time_strategy.get("name") or "none")
    time_strategy.setdefault("parameters", {})
    pretraining = training.setdefault(
        "initial_state_pretraining", {"enabled": False, "parameters": {}}
    )
    pretraining.setdefault("enabled", False)
    pretraining.setdefault("parameters", {})
    if not pretraining["enabled"]:
        pretraining["parameters"] = {}
    for field_name in ("gradient_clipping", "early_stopping"):
        training[field_name].setdefault("parameters", {})
        if not training[field_name]["enabled"]:
            training[field_name]["parameters"] = {}
    return _sort_dicts(_normalize_nested(normalized))


def validate_algorithm_spec(
    spec: dict,
    problem: Any,
    experiment_config: dict,
    builder_capabilities: dict,
) -> ValidationReport:
    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []
    if not isinstance(spec, dict):
        return ValidationReport(valid=False, errors=[ValidationIssue("$", "INVALID_TYPE", "AlgorithmSpec must be an object")])

    for field_name in REQUIRED_TOP_LEVEL_FIELDS:
        if field_name not in spec:
            errors.append(ValidationIssue(field_name, "MISSING_FIELD", f"Missing required field '{field_name}'"))
    if errors:
        return ValidationReport(valid=False, errors=errors, raw_spec=deepcopy(spec))

    _validate_structure(spec, errors)
    if errors:
        return ValidationReport(valid=False, errors=errors, raw_spec=deepcopy(spec))

    try:
        normalized = normalize_algorithm_spec(spec)
    except Exception as exc:
        return ValidationReport(
            valid=False,
            errors=[ValidationIssue("$", "NORMALIZATION_FAILED", str(exc))],
        )

    _validate_finite_numbers(normalized, "$", errors)
    _validate_positive_values(normalized, errors)
    _validate_components(normalized, builder_capabilities, errors)
    _validate_registered_option_schemas(normalized, builder_capabilities, errors)
    _validate_fine_tuning_decision(normalized, errors)
    _validate_adaptive_control(normalized, errors, problem)
    _validate_adaptive_policy_compatibility(
        normalized, builder_capabilities, errors
    )
    _validate_problem_combinations(
        normalized, problem, builder_capabilities, errors
    )
    budget_usage = _validate_budget(normalized, problem, experiment_config, errors)
    return ValidationReport(
        valid=not errors,
        errors=errors,
        warnings=warnings,
        raw_spec=deepcopy(spec),
        normalized_spec=normalized,
        budget_usage=budget_usage,
    )


def _issue(errors: list[ValidationIssue], path: str, code: str, message: str) -> None:
    errors.append(ValidationIssue(path, code, message))


def _validate_governing_loss_component_uniqueness(
    loss_terms: list[dict[str, Any]],
    governing_names: tuple[str, ...],
    errors: list[ValidationIssue],
) -> None:
    """Reject declared PDE terms that expand to the same runtime component."""

    if not governing_names:
        return
    governing_name_set = set(governing_names)
    component_sources: dict[str, list[int]] = {}
    for index, term in enumerate(loss_terms):
        source_name = str(term.get("name") or "")
        if source_name not in GOVERNING_LOSS_TERM_NAMES:
            continue
        parameters = dict(term.get("parameters") or {})
        selected_name = parameters.get("component_name")
        selected_index = parameters.get("component_index")
        if selected_name is not None:
            selected = str(selected_name)
            if selected not in governing_name_set:
                _issue(
                    errors,
                    f"loss.terms[{index}].parameters.component_name",
                    "GOVERNING_COMPONENT_UNAVAILABLE",
                    f"Problem has no governing residual component '{selected}'",
                )
                continue
            residual_names = (selected,)
        elif selected_index is not None:
            try:
                component_index = int(selected_index)
            except (TypeError, ValueError):
                # The registered option schema reports the invalid type/value.
                continue
            if component_index < 0 or component_index >= len(governing_names):
                _issue(
                    errors,
                    f"loss.terms[{index}].parameters.component_index",
                    "GOVERNING_COMPONENT_INDEX_OUT_OF_RANGE",
                    f"Governing residual component index {component_index} is out of range",
                )
                continue
            residual_names = (governing_names[component_index],)
        elif source_name == "pde_component":
            # The existing PDE_COMPONENT_SELECTOR_REQUIRED validation owns this
            # malformed declaration; it contributes no resolvable component.
            continue
        else:
            residual_names = governing_names

        for residual_name in residual_names:
            component_id = f"pde/{residual_name}"
            component_sources.setdefault(component_id, []).append(index)

    for component_id, source_indices in sorted(component_sources.items()):
        if len(source_indices) <= 1:
            continue
        _issue(
            errors,
            "loss.terms",
            "DUPLICATE_EXECUTABLE_LOSS_COMPONENT",
            f"Executable loss component '{component_id}' is produced by loss terms "
            f"at indices {source_indices}; aggregate and component-specific PDE "
            "terms must not overlap",
        )


def _validate_fine_tuning_decision(
    spec: dict[str, Any], errors: list[ValidationIssue]
) -> None:
    """Keep the audit declaration synchronized with the executable final phase.

    Legacy specs may omit the declaration.  Newly generated specs receive it
    from the complete design contract and must provide all three fields.
    """

    metadata = spec.get("generation_metadata") or {}
    fields = {
        "fine_tuning_enabled",
        "fine_tuning_optimizer",
        "fine_tuning_decision_basis",
    }
    present = fields.intersection(metadata)
    if not present:
        return
    missing = sorted(fields - set(metadata))
    for name in missing:
        _issue(
            errors,
            f"generation_metadata.{name}",
            "INCOMPLETE_FINE_TUNING_DECISION",
            "A fine-tuning declaration must include enabled, optimizer, and decision basis fields",
        )
    if missing:
        return

    enabled = metadata.get("fine_tuning_enabled")
    optimizer = metadata.get("fine_tuning_optimizer")
    basis = metadata.get("fine_tuning_decision_basis")
    if not isinstance(enabled, bool):
        _issue(
            errors,
            "generation_metadata.fine_tuning_enabled",
            "INVALID_FINE_TUNING_ENABLED",
            "fine_tuning_enabled must be a boolean",
        )
        return
    if not isinstance(basis, list) or not basis or not all(
        isinstance(item, str) and item.strip() for item in basis
    ):
        _issue(
            errors,
            "generation_metadata.fine_tuning_decision_basis",
            "FINE_TUNING_DECISION_BASIS_REQUIRED",
            "State at least one non-empty reason for enabling or disabling fine-tuning",
        )

    phases = (spec.get("optimization") or {}).get("phases") or []
    if not enabled:
        if optimizer not in (None, ""):
            _issue(
                errors,
                "generation_metadata.fine_tuning_optimizer",
                "DISABLED_FINE_TUNING_HAS_OPTIMIZER",
                "fine_tuning_optimizer must be null when fine-tuning is disabled",
            )
        return

    if len(phases) < 2:
        _issue(
            errors,
            "optimization.phases",
            "FINE_TUNING_PHASE_REQUIRED",
            "Enabled fine-tuning requires a dedicated final phase after the main optimization phase",
        )
        return
    if not isinstance(optimizer, str) or not optimizer.strip():
        _issue(
            errors,
            "generation_metadata.fine_tuning_optimizer",
            "FINE_TUNING_OPTIMIZER_REQUIRED",
            "Enabled fine-tuning must name its final-phase optimizer",
        )
        return
    final_optimizer = _normalized_name(phases[-1].get("optimizer"))
    declared_optimizer = OPTIMIZER_ALIASES.get(
        _normalized_name(optimizer), _normalized_name(optimizer)
    )
    if declared_optimizer != final_optimizer:
        _issue(
            errors,
            "generation_metadata.fine_tuning_optimizer",
            "FINE_TUNING_OPTIMIZER_MISMATCH",
            "fine_tuning_optimizer must equal optimization.phases[-1].optimizer",
        )


def _validate_adaptive_control(
    spec: dict[str, Any], errors: list[ValidationIssue], problem: Any
) -> None:
    """Validate the finite-action controller contract without tuning values."""

    extra = dict((spec.get("training") or {}).get("extra_parameters") or {})
    unknown_extra = sorted(
        set(extra)
        - {
            "adaptive_control", "component_control", "controller_coordinator",
            "physics_validation", "best_checkpoint",
        }
    )
    if unknown_extra:
        _issue(
            errors,
            "training.extra_parameters",
            "UNSUPPORTED_ADAPTIVE_CONTROL_FIELD",
            f"Unsupported training extra parameters: {unknown_extra}",
        )
    _validate_component_control_extensions(extra, errors, problem)
    control = extra.get("adaptive_control")
    if control is None:
        if (
            ((spec.get("loss") or {}).get("weighting_strategy") or {}).get("name")
            == "gradient_balance"
            and not bool((extra.get("component_control") or {}).get("enabled"))
        ):
            _issue(
                errors,
                "loss.weighting_strategy.name",
                "ADAPTIVE_CONTROL_REQUIRED",
                "gradient_balance requires training.extra_parameters.adaptive_control.enabled=true",
            )
        return
    if not isinstance(control, dict):
        _issue(errors, "training.extra_parameters.adaptive_control", "INVALID_TYPE", "adaptive_control must be an object")
        return
    allowed_control = {
        "enabled", "freeze_lbfgs_objective", "lbfgs_stall_recovery",
        "rollback", "sampling_control", "curriculum_control",
    }
    unknown = sorted(set(control) - allowed_control)
    if unknown:
        _issue(errors, "training.extra_parameters.adaptive_control", "UNSUPPORTED_ADAPTIVE_CONTROL_FIELD", f"Unsupported adaptive-control fields: {unknown}")

    def contains_reference_metric(value: Any) -> bool:
        forbidden = ("reference", "evaluator", "l1re", "l2re", "amplitude_ratio", "phase_error", "spectrum_error")
        if isinstance(value, dict):
            return any(
                any(token in str(key).casefold() for token in forbidden)
                or contains_reference_metric(item)
                for key, item in value.items()
            )
        if isinstance(value, (list, tuple)):
            return any(contains_reference_metric(item) for item in value)
        return False

    if contains_reference_metric(control):
        _issue(errors, "training.extra_parameters.adaptive_control", "REFERENCE_METRIC_FORBIDDEN", "Trainer controllers may not consume reference/evaluator metrics")
    if control.get("freeze_lbfgs_objective") is not True:
        _issue(errors, "training.extra_parameters.adaptive_control.freeze_lbfgs_objective", "LBFGS_OBJECTIVE_MUST_BE_FROZEN", "L-BFGS objective freezing cannot be disabled")

    _validate_sampling_control(spec, extra, control, errors)
    _validate_curriculum_control(spec, extra, control, errors)

    stall = control.get("lbfgs_stall_recovery") or {}
    allowed_stall = {
        "enabled", "patience", "minimum_relative_improvement", "minimum_step_size",
        "minimum_parameter_update_norm", "maximum_repeated_line_search_failures",
        "minimum_completed_iterations_before_detection", "action",
        "adam_learning_rate_scale", "maximum_recovery_iterations",
        "minimum_recovery_iterations", "reuse_remaining_budget", "return_to_lbfgs",
    }
    if not isinstance(stall, dict):
        _issue(errors, "training.extra_parameters.adaptive_control.lbfgs_stall_recovery", "INVALID_TYPE", "lbfgs_stall_recovery must be an object")
    else:
        stall_unknown = sorted(set(stall) - allowed_stall)
        if stall_unknown:
            _issue(errors, "training.extra_parameters.adaptive_control.lbfgs_stall_recovery", "UNSUPPORTED_ADAPTIVE_CONTROL_FIELD", f"Unsupported L-BFGS recovery fields: {stall_unknown}")
        action = stall.get("action", "reallocate_to_adam")
        allowed_actions = {"terminate_candidate", "reallocate_to_adam"}
        if not stall.get("enabled", False):
            allowed_actions.add("disabled")
        if action not in allowed_actions:
            _issue(errors, "training.extra_parameters.adaptive_control.lbfgs_stall_recovery.action", "INVALID_CONTROL_ACTION", "action must be disabled, terminate_candidate, or reallocate_to_adam")
        if not stall.get("enabled", False):
            action = "disabled"
        if float(stall.get("adam_learning_rate_scale", 0.1)) <= 0.0:
            _issue(errors, "training.extra_parameters.adaptive_control.lbfgs_stall_recovery.adam_learning_rate_scale", "INVALID_RECOVERY_LEARNING_RATE", "Adam recovery learning-rate scale must be positive")
        total_iterations = sum(int(phase.get("iterations") or 0) for phase in spec["optimization"]["phases"])
        maximum_recovery = int(stall.get("maximum_recovery_iterations", 0 if action == "disabled" else 1000))
        minimum_recovery = int(stall.get("minimum_recovery_iterations", 0 if action == "disabled" else 100))
        if maximum_recovery < 0 or maximum_recovery > total_iterations:
            _issue(errors, "training.extra_parameters.adaptive_control.lbfgs_stall_recovery.maximum_recovery_iterations", "RECOVERY_BUDGET_EXCEEDED", "Recovery budget must be non-negative and cannot exceed the total optimization budget")
        if minimum_recovery < 0 or minimum_recovery > maximum_recovery:
            _issue(errors, "training.extra_parameters.adaptive_control.lbfgs_stall_recovery.minimum_recovery_iterations", "INVALID_RECOVERY_BUDGET", "minimum_recovery_iterations must be between zero and maximum_recovery_iterations")
        for field_name in (() if action == "disabled" else ("patience", "maximum_repeated_line_search_failures", "minimum_completed_iterations_before_detection")):
            if int(stall.get(field_name, 1)) <= 0:
                _issue(errors, f"training.extra_parameters.adaptive_control.lbfgs_stall_recovery.{field_name}", "INVALID_CONTROL_PARAMETER", f"{field_name} must be positive")
        if action != "disabled" and stall.get("return_to_lbfgs") is not False:
            _issue(errors, "training.extra_parameters.adaptive_control.lbfgs_stall_recovery.return_to_lbfgs", "LBFGS_RETURN_FORBIDDEN", "Recovery may not return to the stalled L-BFGS phase")

    rollback = control.get("rollback") or {}
    allowed_rollback = {"enabled", "loss_degradation_ratio", "patience"}
    if not isinstance(rollback, dict):
        _issue(errors, "training.extra_parameters.adaptive_control.rollback", "INVALID_TYPE", "rollback must be an object")
    else:
        rollback_unknown = sorted(set(rollback) - allowed_rollback)
        if rollback_unknown:
            _issue(errors, "training.extra_parameters.adaptive_control.rollback", "UNSUPPORTED_ADAPTIVE_CONTROL_FIELD", f"Unsupported rollback fields: {rollback_unknown}")
        if float(rollback.get("loss_degradation_ratio", 2.0)) <= 1.0:
            _issue(errors, "training.extra_parameters.adaptive_control.rollback.loss_degradation_ratio", "INVALID_ROLLBACK_PARAMETER", "loss_degradation_ratio must be greater than one")
        if int(rollback.get("patience", 3)) <= 0:
            _issue(errors, "training.extra_parameters.adaptive_control.rollback.patience", "INVALID_ROLLBACK_PARAMETER", "rollback patience must be positive")

    weighting = (spec.get("loss") or {}).get("weighting_strategy") or {}
    if weighting.get("name") != "gradient_balance":
        return
    if (
        control.get("enabled") is not True
        and not bool((extra.get("component_control") or {}).get("enabled"))
    ):
        _issue(errors, "training.extra_parameters.adaptive_control.enabled", "ADAPTIVE_CONTROL_REQUIRED", "gradient_balance requires adaptive control to be enabled")
    parameters = dict(weighting.get("parameters") or {})
    checks = (
        (0.0 < float(parameters.get("minimum_update_ratio", 0.5)) <= 1.0, "minimum_update_ratio", "must be in (0, 1]"),
        (float(parameters.get("maximum_update_ratio", 2.0)) >= 1.0, "maximum_update_ratio", "must be at least one"),
        (float(parameters.get("minimum_update_ratio", 0.5)) <= float(parameters.get("maximum_update_ratio", 2.0)), "minimum_update_ratio", "cannot exceed maximum_update_ratio"),
        (float(parameters.get("minimum_weight", 0.01)) < float(parameters.get("maximum_weight", 20.0)), "minimum_weight", "must be smaller than maximum_weight"),
        (0.0 <= float(parameters.get("ema_beta", 0.95)) < 1.0, "ema_beta", "must be in [0, 1)"),
        (int(parameters.get("diagnostic_interval", 50)) > 0, "diagnostic_interval", "must be positive"),
        (int(parameters.get("patience", 3)) > 0, "patience", "must be positive"),
        (int(parameters.get("cooldown_iterations", 200)) >= 0, "cooldown_iterations", "must be non-negative"),
        (-1.0 <= float(parameters.get("conflict_threshold", -0.1)) <= 1.0, "conflict_threshold", "must be in [-1, 1]"),
    )
    for valid, field_name, message in checks:
        if not valid:
            _issue(errors, f"loss.weighting_strategy.parameters.{field_name}", "INVALID_CONTROL_PARAMETER", message)

    if float(parameters.get("minimum_update_ratio", 0.5)) < 0.5:
        _issue(errors, "loss.weighting_strategy.parameters.minimum_update_ratio", "HARD_SAFETY_BOUND", "Per-action loss-weight reduction cannot exceed 2x")
    if float(parameters.get("maximum_update_ratio", 2.0)) > 2.0:
        _issue(errors, "loss.weighting_strategy.parameters.maximum_update_ratio", "HARD_SAFETY_BOUND", "Per-action loss-weight increase cannot exceed 2x")


def _validate_adaptive_policy_compatibility(
    spec: dict[str, Any],
    builder_capabilities: dict[str, Any],
    errors: list[ValidationIssue],
) -> None:
    """Central compatibility matrix for search-visible adaptive policies."""

    extra = dict((spec.get("training") or {}).get("extra_parameters") or {})
    control = dict(extra.get("adaptive_control") or {})
    sampling_control = dict(control.get("sampling_control") or {})
    curriculum_control = dict(control.get("curriculum_control") or {})
    stall = dict(control.get("lbfgs_stall_recovery") or {})
    physics = dict(extra.get("physics_validation") or {})
    best = dict(extra.get("best_checkpoint") or {})
    weighting = dict((spec.get("loss") or {}).get("weighting_strategy") or {})
    capability_snapshot = dict(
        builder_capabilities.get("trainer_capabilities_snapshot") or {}
    )
    trainer_caps = dict(capability_snapshot.get("trainer_capabilities") or {})

    if weighting.get("name") == "lra" and bool(
        (extra.get("component_control") or {}).get("enabled")
    ):
        _issue(
            errors,
            "loss.weighting_strategy.name",
            "POLICY_CONFLICT",
            "LRA and GradientBalance cannot both modify registered loss-component weights",
        )

    first_order = any(
        str(phase.get("optimizer") or "").casefold() != "lbfgs"
        and int(phase.get("iterations") or 0) > 0
        for phase in (spec.get("optimization") or {}).get("phases") or []
    )
    if weighting.get("name") == "gradient_balance" and not first_order:
        _issue(errors, "loss.weighting_strategy.name", "DYNAMIC_STAGE_REQUIRED", "GradientBalance requires a first-order dynamic-objective stage")
    if sampling_control.get("enabled") and not first_order:
        _issue(errors, "training.extra_parameters.adaptive_control.sampling_control", "DYNAMIC_STAGE_REQUIRED", "Sampling replacement requires a first-order dynamic-objective stage")
    if curriculum_control.get("enabled") and not first_order:
        _issue(errors, "training.extra_parameters.adaptive_control.curriculum_control", "DYNAMIC_STAGE_REQUIRED", "Curriculum advancement requires a first-order dynamic-objective stage")

    if sampling_control.get("enabled") and trainer_caps and not trainer_caps.get(
        "supports_sampling_replacement", False
    ):
        _issue(errors, "training.extra_parameters.adaptive_control.sampling_control", "CAPABILITY_NOT_SUPPORTED", "Current problem preflight does not expose a replaceable fixed-budget sampler")
    if curriculum_control.get("enabled") and trainer_caps and not trainer_caps.get(
        "supports_curriculum_advance", False
    ):
        _issue(errors, "training.extra_parameters.adaptive_control.curriculum_control", "CAPABILITY_NOT_SUPPORTED", "Current problem has no eligible registered Curriculum Runtime")
    if physics.get("enabled") and trainer_caps and not trainer_caps.get(
        "supports_physics_validation", False
    ):
        _issue(errors, "training.extra_parameters.physics_validation", "CAPABILITY_NOT_SUPPORTED", "Current problem cannot build a fixed component Physics Probe")

    if stall.get("enabled") and stall.get("action") == "reallocate_to_adam":
        if not first_order:
            _issue(errors, "training.extra_parameters.adaptive_control.lbfgs_stall_recovery", "RECOVERY_STAGE_REQUIRED", "reallocate_to_adam requires an existing first-order optimizer family")
        if not any(str(phase.get("optimizer") or "").casefold() == "lbfgs" for phase in spec["optimization"]["phases"]):
            _issue(errors, "training.extra_parameters.adaptive_control.lbfgs_stall_recovery", "RECOVERY_POLICY_CONFLICT", "L-BFGS recovery cannot be enabled without an L-BFGS stage")

    if control and control.get("freeze_lbfgs_objective") is not True:
        _issue(errors, "training.extra_parameters.adaptive_control.freeze_lbfgs_objective", "HARD_SAFETY_BOUND", "The second-order objective freeze is immutable")

    _validate_exposed_adaptive_schemas(
        spec, builder_capabilities, errors
    )


def _validate_exposed_adaptive_schemas(
    spec: dict[str, Any],
    builder_capabilities: dict[str, Any],
    errors: list[ValidationIssue],
) -> None:
    registry = builder_capabilities.get("algorithm_spec_option_registry") or {}
    fields = registry.get("fields") or {}
    extra = spec["training"].get("extra_parameters") or {}
    control = extra.get("adaptive_control") or {}
    selections = (
        (
            "training.extra_parameters.adaptive_control.lbfgs_stall_recovery.action",
            (control.get("lbfgs_stall_recovery") or {}).get("action", "disabled"),
            control.get("lbfgs_stall_recovery") or {},
        ),
        (
            "training.extra_parameters.adaptive_control.sampling_control.strategy",
            (control.get("sampling_control") or {}).get("strategy", "none"),
            control.get("sampling_control") or {},
        ),
        (
            "training.extra_parameters.adaptive_control.curriculum_control.strategy",
            (control.get("curriculum_control") or {}).get("strategy", "none"),
            control.get("curriculum_control") or {},
        ),
        (
            "training.extra_parameters.physics_validation.strategy",
            (extra.get("physics_validation") or {}).get("strategy", "disabled"),
            extra.get("physics_validation") or {},
        ),
        (
            "training.extra_parameters.best_checkpoint.final_model_policy",
            (extra.get("best_checkpoint") or {}).get("final_model_policy", "last"),
            extra.get("best_checkpoint") or {},
        ),
    )
    ignored = {"enabled", "strategy", "action", "selection_metric"}
    for field_path, option_name, values in selections:
        option = (((fields.get(field_path) or {}).get("options") or {}).get(option_name))
        if not isinstance(option, dict):
            _issue(errors, field_path, "UNREGISTERED_ADAPTIVE_POLICY", f"Adaptive policy '{option_name}' is not registered")
            continue
        properties = ((option.get("parameter_schema") or {}).get("properties") or {})
        for name, schema in properties.items():
            if name in values and name not in ignored:
                _validate_schema_value(values[name], schema, f"{field_path.rsplit('.', 1)[0]}.{name}", errors)


def _validate_sampling_control(
    spec: dict[str, Any],
    extra: dict[str, Any],
    control: dict[str, Any],
    errors: list[ValidationIssue],
) -> None:
    sampling = control.get("sampling_control")
    if sampling is None:
        return
    path = "training.extra_parameters.adaptive_control.sampling_control"
    if not isinstance(sampling, dict):
        _issue(errors, path, "INVALID_TYPE", "sampling_control must be an object")
        return
    allowed = {
        "enabled", "strategy", "diagnostic_interval",
        "minimum_iterations_before_control", "patience", "cooldown_iterations",
        "maximum_updates", "p95_to_median_threshold", "replacement_fraction",
        "hard_point_retention_fraction", "random_exploration_fraction",
        "candidate_multiplier", "maximum_point_age",
        "maximum_consecutive_retention", "minimum_point_age",
        "observation_window_iterations", "minimum_validation_stability_checks",
        "validation_degradation_ratio", "associated_component_degradation_ratio",
        "maximum_candidate_points_evaluated",
    }
    unknown = sorted(set(sampling) - allowed)
    if unknown:
        _issue(errors, path, "UNSUPPORTED_SAMPLING_CONTROL_FIELD", f"Unsupported fields: {unknown}")
    strategy = sampling.get("strategy", "fixed_budget_subset_replacement")
    if strategy not in {"none", "fixed_budget_subset_replacement"}:
        _issue(errors, f"{path}.strategy", "UNSUPPORTED_SAMPLING_CONTROL_STRATEGY", "Only fixed_budget_subset_replacement is registered")
    if not sampling.get("enabled", False) or strategy == "none":
        if sampling.get("enabled", False) and strategy == "none":
            _issue(errors, path, "POLICY_STATE_CONFLICT", "sampling strategy 'none' cannot be enabled")
        return
    replacement = float(sampling.get("replacement_fraction", 0.2))
    retention = float(sampling.get("hard_point_retention_fraction", 0.2))
    exploration = float(sampling.get("random_exploration_fraction", 0.1))
    if not 0.0 < replacement < 1.0:
        _issue(errors, f"{path}.replacement_fraction", "INVALID_SAMPLING_CONTROL_RATIO", "replacement_fraction must be in (0, 1)")
    if not 0.0 <= retention <= 1.0:
        _issue(errors, f"{path}.hard_point_retention_fraction", "INVALID_SAMPLING_CONTROL_RATIO", "hard retention must be in [0, 1]")
    if not 0.0 <= exploration <= 1.0:
        _issue(errors, f"{path}.random_exploration_fraction", "INVALID_SAMPLING_CONTROL_RATIO", "exploration must be in [0, 1]")
    if replacement + retention > 1.0 + 1e-12:
        _issue(errors, path, "OVERLAPPING_SAMPLING_CONTROL_RATIOS", "replacement and hard-retention fractions must not overlap")
    positive = (
        "diagnostic_interval", "patience", "maximum_point_age",
        "observation_window_iterations", "minimum_validation_stability_checks",
        "maximum_candidate_points_evaluated",
    )
    for name in positive:
        if int(sampling.get(name, 1)) <= 0:
            _issue(errors, f"{path}.{name}", "INVALID_SAMPLING_CONTROL_PARAMETER", f"{name} must be positive")
    for name in ("cooldown_iterations", "maximum_updates", "minimum_iterations_before_control", "minimum_point_age", "maximum_consecutive_retention"):
        if int(sampling.get(name, 0)) < 0:
            _issue(errors, f"{path}.{name}", "INVALID_SAMPLING_CONTROL_PARAMETER", f"{name} must be non-negative")
    if int(sampling.get("candidate_multiplier", 4)) < 1:
        _issue(errors, f"{path}.candidate_multiplier", "INVALID_SAMPLING_CONTROL_PARAMETER", "candidate_multiplier must be at least one")
    for name in ("validation_degradation_ratio", "associated_component_degradation_ratio"):
        if float(sampling.get(name, 1.5)) <= 1.0:
            _issue(errors, f"{path}.{name}", "INVALID_SAMPLING_CONTROL_PARAMETER", f"{name} must be greater than one")
    if sampling.get("enabled"):
        physics = extra.get("physics_validation") or {}
        if not isinstance(physics, dict) or physics.get("enabled") is not True:
            _issue(errors, path, "PHYSICS_VALIDATION_REQUIRED", "Sampling control requires physics_validation.enabled=true")
        adaptive = (spec.get("sampling") or {}).get("adaptive_refinement") or {}
        if adaptive.get("enabled"):
            _issue(errors, path, "SAMPLING_CONTROL_CONFLICT", "Fixed-budget replacement cannot run with adaptive point addition")


def _validate_curriculum_control(
    spec: dict[str, Any],
    extra: dict[str, Any],
    control: dict[str, Any],
    errors: list[ValidationIssue],
) -> None:
    curriculum = control.get("curriculum_control")
    if curriculum is None:
        return
    path = "training.extra_parameters.adaptive_control.curriculum_control"
    if not isinstance(curriculum, dict):
        _issue(errors, path, "INVALID_TYPE", "curriculum_control must be an object")
        return
    allowed = {
        "enabled", "strategy", "minimum_iterations_per_level",
        "maximum_iterations_per_level", "validation_interval", "patience",
        "cooldown_iterations", "minimum_relative_improvement",
        "maximum_validation_score", "maximum_advances",
        "normal_advance_condition", "forced_advance_policy",
        "minimum_remaining_budget_after_advance",
        "observation_window_iterations", "forced_observation_window_multiplier",
        "minimum_validation_checks", "aggregate_degradation_ratio",
        "associated_component_degradation_ratio",
        "critical_component_degradation_ratio", "rollback_on_non_finite",
        "post_advance_controller_ema_decay",
    }
    unknown = sorted(set(curriculum) - allowed)
    if unknown:
        _issue(
            errors, path, "UNSUPPORTED_CURRICULUM_CONTROL_FIELD",
            f"Unsupported fields: {unknown}",
        )
    strategy = curriculum.get("strategy", "residual_gated_level_advance")
    if strategy not in {"none", "residual_gated_level_advance"}:
        _issue(
            errors, f"{path}.strategy", "UNSUPPORTED_CURRICULUM_CONTROL_STRATEGY",
            "Only residual_gated_level_advance is registered",
        )
    if not curriculum.get("enabled", False) or strategy == "none":
        if curriculum.get("enabled", False) and strategy == "none":
            _issue(errors, path, "POLICY_STATE_CONFLICT", "curriculum strategy 'none' cannot be enabled")
        return
    minimum = int(curriculum.get("minimum_iterations_per_level", 500))
    maximum = int(curriculum.get("maximum_iterations_per_level", 3000))
    if minimum < 0:
        _issue(errors, f"{path}.minimum_iterations_per_level", "INVALID_CURRICULUM_CONTROL_PARAMETER", "minimum iterations must be non-negative")
    if maximum <= minimum:
        _issue(errors, f"{path}.maximum_iterations_per_level", "INVALID_CURRICULUM_CONTROL_PARAMETER", "maximum iterations must exceed minimum iterations")
    for name in (
        "validation_interval", "patience", "observation_window_iterations",
        "minimum_validation_checks",
    ):
        if int(curriculum.get(name, 1)) <= 0:
            _issue(errors, f"{path}.{name}", "INVALID_CURRICULUM_CONTROL_PARAMETER", f"{name} must be positive")
    if int(curriculum.get("observation_window_iterations", 500)) < (
        int(curriculum.get("validation_interval", 100))
        * int(curriculum.get("minimum_validation_checks", 3))
    ):
        _issue(
            errors,
            f"{path}.observation_window_iterations",
            "INVALID_CURRICULUM_OBSERVATION_BUDGET",
            "observation window cannot contain the required validation checks",
        )
    for name in (
        "cooldown_iterations", "maximum_advances",
        "minimum_remaining_budget_after_advance",
    ):
        if int(curriculum.get(name, 0)) < 0:
            _issue(errors, f"{path}.{name}", "INVALID_CURRICULUM_CONTROL_PARAMETER", f"{name} must be non-negative")
    if float(curriculum.get("minimum_relative_improvement", 0.001)) < 0.0:
        _issue(errors, f"{path}.minimum_relative_improvement", "INVALID_CURRICULUM_CONTROL_PARAMETER", "minimum relative improvement must be non-negative")
    maximum_score = curriculum.get("maximum_validation_score")
    if maximum_score is not None and float(maximum_score) <= 0.0:
        _issue(errors, f"{path}.maximum_validation_score", "INVALID_CURRICULUM_CONTROL_PARAMETER", "maximum validation score must be positive or null")
    if float(curriculum.get("forced_observation_window_multiplier", 2.0)) < 1.0:
        _issue(errors, f"{path}.forced_observation_window_multiplier", "INVALID_CURRICULUM_CONTROL_PARAMETER", "forced observation multiplier must be at least one")
    for name in (
        "aggregate_degradation_ratio", "associated_component_degradation_ratio",
        "critical_component_degradation_ratio",
    ):
        if float(curriculum.get(name, 1.5)) <= 1.0:
            _issue(errors, f"{path}.{name}", "INVALID_CURRICULUM_CONTROL_PARAMETER", f"{name} must be greater than one")
    decay = float(curriculum.get("post_advance_controller_ema_decay", 0.5))
    if not 0.0 <= decay <= 1.0:
        _issue(errors, f"{path}.post_advance_controller_ema_decay", "INVALID_CURRICULUM_CONTROL_PARAMETER", "EMA decay must be in [0, 1]")
    normal = str(curriculum.get("normal_advance_condition", "physics_validation_stable"))
    if normal not in {
        "physics_validation_stable", "absolute_threshold", "relative_plateau",
        "relative_improvement", "hybrid",
    }:
        _issue(errors, f"{path}.normal_advance_condition", "INVALID_CURRICULUM_ADVANCE_CONDITION", "normal advance condition is not registered")
    forced = str(curriculum.get("forced_advance_policy", "advance_one_level"))
    if forced not in {"advance_one_level", "terminate_candidate", "hold_until_budget_end"}:
        _issue(errors, f"{path}.forced_advance_policy", "INVALID_CURRICULUM_FORCED_POLICY", "forced policy is not registered")
    if not isinstance(curriculum.get("rollback_on_non_finite", True), bool):
        _issue(errors, f"{path}.rollback_on_non_finite", "INVALID_TYPE", "rollback_on_non_finite must be boolean")
    total_iterations = sum(
        int(phase.get("iterations") or 0)
        for phase in (spec.get("optimization") or {}).get("phases") or []
    )
    if int(curriculum.get("minimum_remaining_budget_after_advance", 0)) > total_iterations:
        _issue(errors, f"{path}.minimum_remaining_budget_after_advance", "CURRICULUM_BUDGET_EXCEEDED", "remaining-budget requirement exceeds the optimization budget")
    if curriculum.get("enabled"):
        physics = extra.get("physics_validation") or {}
        if not isinstance(physics, dict) or physics.get("enabled") is not True:
            _issue(errors, path, "PHYSICS_VALIDATION_REQUIRED", "Curriculum control requires physics_validation.enabled=true")
        rollback = control.get("rollback") or {}
        if not isinstance(rollback, dict) or rollback.get("enabled") is not True:
            _issue(errors, path, "CURRICULUM_ROLLBACK_REQUIRED", "Curriculum control requires rollback.enabled=true")
        if control.get("freeze_lbfgs_objective") is not True:
            _issue(errors, path, "LBFGS_OBJECTIVE_MUST_BE_FROZEN", "Curriculum state must be frozen in second-order stages")
        if not any(
            str(phase.get("optimizer", "")).casefold() != "lbfgs"
            for phase in (spec.get("optimization") or {}).get("phases") or []
        ):
            _issue(errors, path, "FIRST_ORDER_CURRICULUM_STAGE_REQUIRED", "Curriculum control requires a first-order optimization stage")
        time_strategy = (spec.get("training") or {}).get("time_strategy") or {}
        if time_strategy.get("name") == "time_window_curriculum":
            levels = tuple((time_strategy.get("parameters") or {}).get("window_fractions") or ())
            if len(levels) < 2:
                _issue(errors, "training.time_strategy.parameters.window_fractions", "INSUFFICIENT_CURRICULUM_LEVELS", "A curriculum must declare at least two levels")


def _validate_component_control_extensions(
    extra: dict[str, Any], errors: list[ValidationIssue], problem: Any
) -> None:
    """Validate the finite, reference-free generic control protocol."""

    base = "training.extra_parameters"
    control = extra.get("component_control")
    if control is not None:
        if not isinstance(control, dict):
            _issue(errors, f"{base}.component_control", "INVALID_TYPE", "component_control must be an object")
        else:
            unknown = sorted(set(control) - {"enabled"})
            if unknown:
                _issue(errors, f"{base}.component_control", "UNSUPPORTED_COMPONENT_CONTROL_FIELD", f"Unsupported fields: {unknown}")
            if not isinstance(control.get("enabled", False), bool):
                _issue(errors, f"{base}.component_control.enabled", "INVALID_TYPE", "enabled must be boolean")

    coordinator = extra.get("controller_coordinator")
    if coordinator is not None:
        path = f"{base}.controller_coordinator"
        allowed = {"enabled", "maximum_non_safety_actions_per_cycle", "observation_window_iterations", "global_cooldown_iterations", "priority"}
        if not isinstance(coordinator, dict):
            _issue(errors, path, "INVALID_TYPE", "controller_coordinator must be an object")
        else:
            unknown = sorted(set(coordinator) - allowed)
            if unknown:
                _issue(errors, path, "UNSUPPORTED_COORDINATOR_FIELD", f"Unsupported fields: {unknown}")
            if int(coordinator.get("maximum_non_safety_actions_per_cycle", 1)) != 1:
                _issue(errors, f"{path}.maximum_non_safety_actions_per_cycle", "UNSAFE_ACTION_CONCURRENCY", "Exactly one non-safety action per cycle is required")
            for name in ("observation_window_iterations", "global_cooldown_iterations"):
                if int(coordinator.get(name, 100)) < 0:
                    _issue(errors, f"{path}.{name}", "INVALID_CONTROL_PARAMETER", f"{name} must be non-negative")
            expected = {"rollback", "terminate_candidate", "stage_recovery", "loss_weight", "learning_rate", "sampling", "curriculum"}
            priority = coordinator.get("priority") or []
            if not isinstance(priority, list) or len(priority) != len(expected) or set(priority) != expected:
                _issue(errors, f"{path}.priority", "INVALID_ACTION_PRIORITY", "priority must contain each registered action category exactly once")

    score_names_provider = getattr(
        problem, "physics_validation_score_names", None
    )
    problem_validation_scores = (
        {
            str(name)
            for name in (score_names_provider() or ())
        }
        if callable(score_names_provider)
        else set()
    )
    allowed_validation_scores = {
        "physics_validation_score",
        *problem_validation_scores,
    }
    physics = extra.get("physics_validation")
    if physics is not None:
        path = f"{base}.physics_validation"
        allowed = {
            "enabled", "strategy", "evaluation_interval", "normalization",
            "category_aggregation", "median_weight", "worst_weight",
            "minimum_improvement", "epsilon", "seed",
            "points_per_component", "maximum_total_points",
            "category_weights", "minimum_component_scale",
            "region_normalization", "second_order_degradation_ratio",
            "second_order_guard_metric",
        }
        if not isinstance(physics, dict):
            _issue(errors, path, "INVALID_TYPE", "physics_validation must be an object")
        else:
            unknown = sorted(set(physics) - allowed)
            if unknown:
                _issue(errors, path, "UNSUPPORTED_PHYSICS_VALIDATION_FIELD", f"Unsupported fields: {unknown}")
            strategy = physics.get("strategy", "fixed_component_probe")
            if strategy not in {"disabled", "fixed_component_probe"}:
                _issue(errors, f"{path}.strategy", "UNSUPPORTED_PHYSICS_VALIDATION_STRATEGY", "strategy must be disabled or fixed_component_probe")
            if not physics.get("enabled", False):
                if strategy != "disabled":
                    _issue(errors, path, "POLICY_STATE_CONFLICT", "disabled physics validation must use strategy='disabled'")
                physics = None
            if physics is None:
                pass
            elif physics.get("normalization", "initial_probe_value") != "initial_probe_value":
                _issue(errors, f"{path}.normalization", "INVALID_PROBE_NORMALIZATION", "Only initial_probe_value is supported")
            if physics is not None and physics.get("category_aggregation", "median_worst_blend") != "median_worst_blend":
                _issue(errors, f"{path}.category_aggregation", "INVALID_PROBE_AGGREGATION", "Only median_worst_blend is supported")
            if physics is not None and int(physics.get("evaluation_interval", 100)) <= 0:
                _issue(errors, f"{path}.evaluation_interval", "INVALID_PROBE_INTERVAL", "evaluation_interval must be positive")
            median = float((physics or {}).get("median_weight", 0.7))
            worst = float((physics or {}).get("worst_weight", 0.3))
            if median < 0.0 or worst < 0.0 or not math.isclose(median + worst, 1.0, rel_tol=0.0, abs_tol=1e-12):
                _issue(errors, path, "INVALID_PROBE_AGGREGATION_WEIGHTS", "median_weight and worst_weight must be non-negative and sum to one")
            for name in (() if physics is None else ("epsilon", "points_per_component", "maximum_total_points")):
                if float(physics.get(name, 1)) <= 0.0:
                    _issue(errors, f"{path}.{name}", "INVALID_PROBE_BUDGET", f"{name} must be positive")
            if (
                physics is not None
                and float(physics.get("minimum_component_scale", 0.0))
                < 0.0
            ):
                _issue(
                    errors,
                    f"{path}.minimum_component_scale",
                    "INVALID_PROBE_SCALE",
                    "minimum_component_scale must be non-negative",
                )
            if (
                physics is not None
                and physics.get(
                    "region_normalization", "per_region_initial"
                )
                not in {"per_region_initial", "shared_pde_rmse"}
            ):
                _issue(
                    errors,
                    f"{path}.region_normalization",
                    "INVALID_REGION_NORMALIZATION",
                    "region_normalization must be per_region_initial or shared_pde_rmse",
                )
            if (
                physics is not None
                and float(
                    physics.get("second_order_degradation_ratio", 1.0)
                )
                < 1.0
            ):
                _issue(
                    errors,
                    f"{path}.second_order_degradation_ratio",
                    "INVALID_SECOND_ORDER_GUARD",
                    "second_order_degradation_ratio must be at least one",
                )
            if (
                physics is not None
                and physics.get(
                    "second_order_guard_metric",
                    "physics_validation_score",
                )
                not in allowed_validation_scores
            ):
                _issue(
                    errors,
                    f"{path}.second_order_guard_metric",
                    "INVALID_SECOND_ORDER_GUARD_METRIC",
                    "second-order guard metric is not provided by this problem",
                )
            weights = (physics or {}).get("category_weights") or {}
            try:
                valid_weights = isinstance(weights, dict) and all(float(value) > 0.0 for value in weights.values())
            except (TypeError, ValueError):
                valid_weights = False
            if physics is not None and not valid_weights:
                _issue(errors, f"{path}.category_weights", "INVALID_CATEGORY_WEIGHTS", "category weights must be a positive static mapping")

    best = extra.get("best_checkpoint")
    if best is not None:
        path = f"{base}.best_checkpoint"
        allowed = {
            "enabled", "selection_metric", "final_model_policy",
            "evaluation_interval",
        }
        if not isinstance(best, dict):
            _issue(errors, path, "INVALID_TYPE", "best_checkpoint must be an object")
        else:
            unknown = sorted(set(best) - allowed)
            if unknown:
                _issue(errors, path, "UNSUPPORTED_BEST_CHECKPOINT_FIELD", f"Unsupported fields: {unknown}")
            selection_metric = best.get("selection_metric", "training_loss")
            if selection_metric != "training_loss":
                _issue(errors, f"{path}.selection_metric", "INVALID_BEST_SELECTION_METRIC", "best_checkpoint only supports selection_metric=training_loss")
            final_model_policy = best.get("final_model_policy", "last")
            if final_model_policy not in {"last", "best_train_loss"}:
                _issue(errors, f"{path}.final_model_policy", "INVALID_FINAL_MODEL_POLICY", "final_model_policy must be last or best_train_loss")
            if final_model_policy == "best_train_loss" and selection_metric != "training_loss":
                _issue(errors, f"{path}.selection_metric", "INVALID_BEST_SELECTION_METRIC", "best_train_loss requires selection_metric=training_loss")
            if int(best.get("evaluation_interval", 100)) < 1:
                _issue(errors, f"{path}.evaluation_interval", "INVALID_BEST_CHECKPOINT_PARAMETER", "evaluation_interval must be positive")

    forbidden = ("reference", "evaluator", "l1re", "l2re", "amplitude", "phase", "spectrum_error", "network_layers", "activation", "output_transform", "add_loss", "remove_loss", "pde_residual")

    def contains_forbidden(value: Any) -> bool:
        if isinstance(value, dict):
            return any(
                any(token in str(key).casefold() for token in forbidden)
                or contains_forbidden(item)
                for key, item in value.items()
            )
        if isinstance(value, (list, tuple)):
            return any(contains_forbidden(item) for item in value)
        return False

    for name in ("component_control", "controller_coordinator", "physics_validation", "best_checkpoint"):
        if contains_forbidden(extra.get(name)):
            _issue(errors, f"{base}.{name}", "FORBIDDEN_CONTROL_INFORMATION", "Control-plane configuration may not contain reference metrics or structural mutations")


def _validate_structure(spec: dict[str, Any], errors: list[ValidationIssue]) -> None:
    if not isinstance(spec.get("spec_version"), str):
        _issue(errors, "spec_version", "INVALID_TYPE", "spec_version must be a string")
    for section in ("network", "sampling", "loss", "optimization", "training", "generation_metadata"):
        if not isinstance(spec.get(section), dict):
            _issue(errors, section, "INVALID_TYPE", f"{section} must be an object")
    if errors:
        return

    enforcement = spec.get("constraint_enforcement")
    if enforcement is not None:
        if not isinstance(enforcement, dict):
            _issue(
                errors,
                "constraint_enforcement",
                "INVALID_TYPE",
                "constraint_enforcement must be an object",
            )
        else:
            if not isinstance(enforcement.get("method"), str):
                _issue(
                    errors,
                    "constraint_enforcement.method",
                    "MISSING_OR_INVALID_FIELD",
                    "constraint enforcement method must be a string",
                )
            if not isinstance(enforcement.get("transform_id"), str):
                _issue(
                    errors,
                    "constraint_enforcement.transform_id",
                    "MISSING_OR_INVALID_FIELD",
                    "constraint transform_id must be a string",
                )
            if not isinstance(enforcement.get("parameters", {}), dict):
                _issue(
                    errors,
                    "constraint_enforcement.parameters",
                    "INVALID_TYPE",
                    "constraint enforcement parameters must be an object",
                )
    if errors:
        return

    network = spec["network"]
    if "architecture_parameters" in network:
        _issue(
            errors,
            "network.architecture_parameters",
            "LEGACY_FIELD_UNSUPPORTED",
            "Use network.extra_parameters for architecture settings",
        )
    required_network_fields = {"architecture", "hidden_layers", "activation", "output_activation", "initialization", "input_transform", "residual_connections"}
    for key in sorted(required_network_fields - set(network)):
        _issue(errors, f"network.{key}", "MISSING_FIELD", f"Missing required network field '{key}'")
    hidden = network.get("hidden_layers")
    if not isinstance(hidden, list) or not hidden or not all(isinstance(value, int) and not isinstance(value, bool) for value in hidden):
        _issue(errors, "network.hidden_layers", "INVALID_HIDDEN_LAYERS", "hidden_layers must be a non-empty integer list")
    if not isinstance(network.get("architecture"), str):
        _issue(errors, "network.architecture", "INVALID_TYPE", "architecture must be a string")
    for key in ("activation", "output_activation", "initialization", "input_transform", "residual_connections"):
        if key in network and not isinstance(network[key], dict):
            _issue(errors, f"network.{key}", "INVALID_TYPE", f"network.{key} must be an object")
    for key in ("activation", "output_activation", "initialization", "input_transform"):
        if isinstance(network.get(key), dict) and not isinstance(network[key].get("name"), str):
            _issue(errors, f"network.{key}.name", "MISSING_OR_INVALID_FIELD", "component name must be a string")
    if isinstance(network.get("residual_connections"), dict) and not isinstance(network["residual_connections"].get("enabled"), bool):
        _issue(errors, "network.residual_connections.enabled", "MISSING_OR_INVALID_FIELD", "enabled must be a boolean")
    if errors:
        return

    sampling = spec["sampling"]
    for key in ("interior", "boundary", "initial", "adaptive_refinement"):
        if key not in sampling:
            _issue(errors, f"sampling.{key}", "MISSING_FIELD", f"Missing required sampling field '{key}'")
        elif not isinstance(sampling[key], dict):
            _issue(errors, f"sampling.{key}", "INVALID_TYPE", f"sampling.{key} must be an object")
    for key in ("interior", "boundary", "initial"):
        if isinstance(sampling.get(key), dict) and not isinstance(sampling[key].get("strategy"), str):
            _issue(errors, f"sampling.{key}.strategy", "MISSING_OR_INVALID_FIELD", "strategy must be a string")
    adaptive = sampling.get("adaptive_refinement")
    if isinstance(adaptive, dict):
        if not isinstance(adaptive.get("enabled"), bool):
            _issue(errors, "sampling.adaptive_refinement.enabled", "MISSING_OR_INVALID_FIELD", "enabled must be a boolean")
        if not isinstance(adaptive.get("strategy"), str):
            _issue(errors, "sampling.adaptive_refinement.strategy", "MISSING_OR_INVALID_FIELD", "strategy must be a string")
    if errors:
        return

    loss = spec["loss"]
    for key in ("terms", "weighting_strategy", "aggregation"):
        if key not in loss:
            _issue(errors, f"loss.{key}", "MISSING_FIELD", f"Missing required loss field '{key}'")
    terms = loss.get("terms")
    if not isinstance(terms, list) or not terms:
        _issue(errors, "loss.terms", "INVALID_LOSS_TERMS", "loss.terms must be a non-empty list")
    elif not all(isinstance(term, dict) for term in terms):
        _issue(errors, "loss.terms", "INVALID_TYPE", "every loss term must be an object")
    else:
        for index, term in enumerate(terms):
            if "loss_function_parameters" in term:
                _issue(
                    errors,
                    f"loss.terms[{index}].loss_function_parameters",
                    "LEGACY_FIELD_UNSUPPORTED",
                    "Use loss.terms[].parameters for loss-function settings",
                )
            if not isinstance(term.get("name"), str):
                _issue(errors, f"loss.terms[{index}].name", "MISSING_OR_INVALID_FIELD", "loss term name must be a string")
            if not isinstance(term.get("loss_function"), str):
                _issue(errors, f"loss.terms[{index}].loss_function", "MISSING_OR_INVALID_FIELD", "loss_function must be a string")
            if not isinstance(term.get("weight"), (int, float)) or isinstance(term.get("weight"), bool):
                _issue(errors, f"loss.terms[{index}].weight", "MISSING_OR_INVALID_FIELD", "weight must be numeric")
    for key in ("weighting_strategy", "aggregation"):
        if key in loss and not isinstance(loss[key], dict):
            _issue(errors, f"loss.{key}", "INVALID_TYPE", f"loss.{key} must be an object")

    phases = spec["optimization"].get("phases")
    if not isinstance(phases, list) or not phases:
        _issue(errors, "optimization.phases", "INVALID_PHASES", "optimization.phases must be a non-empty list")
    elif not all(isinstance(phase, dict) for phase in phases):
        _issue(errors, "optimization.phases", "INVALID_TYPE", "every optimization phase must be an object")
    elif any(not isinstance(phase.get("scheduler"), dict) for phase in phases):
        _issue(errors, "optimization.phases[].scheduler", "INVALID_TYPE", "every scheduler must be an object")
    else:
        for index, phase in enumerate(phases):
            if "optimizer_parameters" in phase:
                _issue(
                    errors,
                    f"optimization.phases[{index}].optimizer_parameters",
                    "LEGACY_FIELD_UNSUPPORTED",
                    "Use optimization.phases[].parameters for optimizer settings",
                )
            if not isinstance(phase.get("optimizer"), str):
                _issue(errors, f"optimization.phases[{index}].optimizer", "MISSING_OR_INVALID_FIELD", "optimizer must be a string")
            if not isinstance(phase.get("iterations"), int) or isinstance(phase.get("iterations"), bool):
                _issue(errors, f"optimization.phases[{index}].iterations", "MISSING_OR_INVALID_FIELD", "iterations must be an integer")
            if not isinstance(phase["scheduler"].get("name"), str):
                _issue(errors, f"optimization.phases[{index}].scheduler.name", "MISSING_OR_INVALID_FIELD", "scheduler name must be a string")

    training = spec["training"]
    for key in ("batch_mode", "batch_size", "gradient_clipping", "early_stopping"):
        if key not in training:
            _issue(errors, f"training.{key}", "MISSING_FIELD", f"Missing required training field '{key}'")
    if "batch_mode" in training and not isinstance(training["batch_mode"], str):
        _issue(errors, "training.batch_mode", "INVALID_TYPE", "batch_mode must be a string")
    if training.get("batch_mode") == "full_batch" and training.get("batch_size") is not None:
        _issue(
            errors,
            "training.batch_size",
            "FULL_BATCH_REQUIRES_NULL_BATCH_SIZE",
            "batch_size must be null when batch_mode is full_batch",
        )
    for key in ("gradient_clipping", "early_stopping"):
        if key in training and not isinstance(training[key], dict):
            _issue(errors, f"training.{key}", "INVALID_TYPE", f"training.{key} must be an object")
        elif isinstance(training.get(key), dict) and not isinstance(training[key].get("enabled"), bool):
            _issue(errors, f"training.{key}.enabled", "MISSING_OR_INVALID_FIELD", "enabled must be a boolean")
    for key in ("time_strategy", "initial_state_pretraining"):
        if key in training and not isinstance(training[key], dict):
            _issue(errors, f"training.{key}", "INVALID_TYPE", f"training.{key} must be an object")
    if isinstance(training.get("time_strategy"), dict) and not isinstance(
        training["time_strategy"].get("name"), str
    ):
        _issue(
            errors,
            "training.time_strategy.name",
            "MISSING_OR_INVALID_FIELD",
            "time strategy name must be a string",
        )
    if isinstance(training.get("initial_state_pretraining"), dict) and not isinstance(
        training["initial_state_pretraining"].get("enabled"), bool
    ):
        _issue(
            errors,
            "training.initial_state_pretraining.enabled",
            "MISSING_OR_INVALID_FIELD",
            "enabled must be a boolean",
        )
    if errors:
        return
    _validate_parameter_objects(spec, "$", errors)
    for path, value in _walk_named_fields(spec, "enabled"):
        if not isinstance(value, bool):
            _issue(errors, path, "INVALID_ENABLED", "enabled must be a boolean")


def _validate_parameter_objects(value: Any, path: str, errors: list[ValidationIssue]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            child_path = f"{path}.{key}" if path != "$" else key
            if key in {"parameters", "extra_parameters"} and not isinstance(item, dict):
                _issue(errors, child_path, "INVALID_PARAMETERS", f"{key} must be an object")
            _validate_parameter_objects(item, child_path, errors)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_parameter_objects(item, f"{path}[{index}]", errors)


def _validate_finite_numbers(value: Any, path: str, errors: list[ValidationIssue]) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        _issue(errors, path, "NON_FINITE_NUMBER", "Floating-point values must be finite")
    elif isinstance(value, dict):
        for key, item in value.items():
            _validate_finite_numbers(item, f"{path}.{key}", errors)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_finite_numbers(item, f"{path}[{index}]", errors)


def _validate_positive_values(spec: dict[str, Any], errors: list[ValidationIssue]) -> None:
    for index, width in enumerate(spec["network"].get("hidden_layers") or []):
        if width <= 0:
            _issue(errors, f"network.hidden_layers[{index}]", "INVALID_WIDTH", "network width must be positive")
    for index, phase in enumerate(spec["optimization"].get("phases") or []):
        iterations = phase.get("iterations")
        if not isinstance(iterations, int) or isinstance(iterations, bool) or iterations < 0:
            _issue(errors, f"optimization.phases[{index}].iterations", "INVALID_ITERATIONS", "iterations must be a non-negative integer")
        learning_rate = (phase.get("parameters") or {}).get("learning_rate")
        if learning_rate is not None and (not isinstance(learning_rate, (int, float)) or isinstance(learning_rate, bool) or learning_rate <= 0):
            _issue(errors, f"optimization.phases[{index}].parameters.learning_rate", "INVALID_LEARNING_RATE", "learning_rate must be positive")
    for section in ("interior", "boundary", "initial"):
        n_points = ((spec["sampling"].get(section) or {}).get("parameters") or {}).get("n_points", 0)
        if not isinstance(n_points, int) or isinstance(n_points, bool) or n_points < 0:
            _issue(errors, f"sampling.{section}.parameters.n_points", "INVALID_SAMPLE_COUNT", "n_points must be a non-negative integer")
    for index, term in enumerate(spec["loss"].get("terms") or []):
        weight = term.get("weight")
        if (
            not isinstance(weight, (int, float))
            or isinstance(weight, bool)
            or not math.isfinite(float(weight))
            or float(weight) <= 0.0
        ):
            _issue(
                errors,
                f"loss.terms[{index}].weight",
                "INVALID_LOSS_WEIGHT",
                "loss weight must be a finite positive number",
            )


def _validate_components(spec: dict[str, Any], capabilities: dict[str, Any], errors: list[ValidationIssue]) -> None:
    checks = [
        ("network.architecture", spec["network"].get("architecture"), "networks"),
        ("network.activation.name", spec["network"]["activation"].get("name"), "activations"),
        ("network.output_activation.name", spec["network"]["output_activation"].get("name"), "activations"),
        ("network.initialization.name", spec["network"]["initialization"].get("name"), "initializers"),
        ("network.input_transform.name", spec["network"]["input_transform"].get("name"), "input_transforms"),
        (
            "constraint_enforcement.method",
            spec["constraint_enforcement"].get("method"),
            "constraint_enforcement_methods",
        ),
        ("loss.weighting_strategy.name", spec["loss"]["weighting_strategy"].get("name"), "weighting_strategies"),
        ("loss.aggregation.name", spec["loss"]["aggregation"].get("name"), "aggregations"),
    ]
    checks.append(("sampling.interior.strategy", spec["sampling"]["interior"].get("strategy"), "interior_samplers"))
    for section in ("boundary", "initial"):
        checks.append((f"sampling.{section}.strategy", spec["sampling"][section].get("strategy"), "constraint_samplers"))
    adaptive = spec["sampling"]["adaptive_refinement"]
    if adaptive.get("enabled"):
        checks.append(("sampling.adaptive_refinement.strategy", adaptive.get("strategy"), "adaptive_samplers"))
    for index, term in enumerate(spec["loss"].get("terms") or []):
        checks.append((f"loss.terms[{index}].loss_function", term.get("loss_function"), "loss_functions"))
    for index, phase in enumerate(spec["optimization"].get("phases") or []):
        checks.append((f"optimization.phases[{index}].optimizer", phase.get("optimizer"), "optimizers"))
        checks.append((f"optimization.phases[{index}].scheduler.name", phase["scheduler"].get("name"), "schedulers"))
    for path, value, category in checks:
        if value not in set(capabilities.get(category) or []):
            _issue(errors, path, "UNSUPPORTED_COMPONENT", f"No registered builder exists for '{value}'")


def _validate_registered_option_schemas(
    spec: dict[str, Any],
    capabilities: dict[str, Any],
    errors: list[ValidationIssue],
) -> None:
    registry = capabilities.get("algorithm_spec_option_registry") or {}
    fields = registry.get("fields") or {}
    selections: list[tuple[str, str, dict[str, Any], str]] = [
        ("network.architecture", spec["network"]["architecture"], spec["network"].get("extra_parameters") or {}, "network.extra_parameters"),
        ("network.activation.name", spec["network"]["activation"]["name"], spec["network"]["activation"].get("parameters") or {}, "network.activation.parameters"),
        ("network.output_activation.name", spec["network"]["output_activation"]["name"], spec["network"]["output_activation"].get("parameters") or {}, "network.output_activation.parameters"),
        ("network.initialization.name", spec["network"]["initialization"]["name"], spec["network"]["initialization"].get("parameters") or {}, "network.initialization.parameters"),
        ("network.input_transform.name", spec["network"]["input_transform"]["name"], spec["network"]["input_transform"].get("parameters") or {}, "network.input_transform.parameters"),
        (
            "constraint_enforcement.method",
            spec["constraint_enforcement"]["method"],
            spec["constraint_enforcement"].get("parameters") or {},
            "constraint_enforcement.parameters",
        ),
    ]
    for section in ("interior", "boundary", "initial"):
        sampler = spec["sampling"][section]
        selections.append(
            (f"sampling.{section}.strategy", sampler["strategy"], sampler.get("parameters") or {}, f"sampling.{section}.parameters")
        )
    adaptive = spec["sampling"]["adaptive_refinement"]
    if adaptive.get("enabled"):
        selections.append(
            (
                "sampling.adaptive_refinement.strategy",
                adaptive["strategy"],
                adaptive.get("parameters") or {},
                "sampling.adaptive_refinement.parameters",
            )
        )
    time_strategy = spec["training"]["time_strategy"]
    selections.append(
        (
            "training.time_strategy.name",
            time_strategy["name"],
            time_strategy.get("parameters") or {},
            "training.time_strategy.parameters",
        )
    )
    for index, term in enumerate(spec["loss"].get("terms") or []):
        selections.append(
            ("loss.terms[].loss_function", term["loss_function"], term.get("parameters") or {}, f"loss.terms[{index}].parameters")
        )
    for field_name in ("weighting_strategy", "aggregation"):
        component = spec["loss"][field_name]
        selections.append(
            (f"loss.{field_name}.name", component["name"], component.get("parameters") or {}, f"loss.{field_name}.parameters")
        )
    for index, phase in enumerate(spec["optimization"].get("phases") or []):
        selections.append(
            ("optimization.phases[].optimizer", phase["optimizer"], phase.get("parameters") or {}, f"optimization.phases[{index}].parameters")
        )
        selections.append(
            (
                "optimization.phases[].scheduler.name",
                phase["scheduler"]["name"],
                phase["scheduler"].get("parameters") or {},
                f"optimization.phases[{index}].scheduler.parameters",
            )
        )

    for field_path, option_name, parameters, parameter_path in selections:
        field = fields.get(field_path) or {}
        options = field.get("options") or {}
        option = options.get(option_name)
        if not isinstance(option, dict):
            if options:
                _issue(
                    errors,
                    field_path,
                    "UNREGISTERED_OPTION",
                    f"Option '{option_name}' is not registered for {field_path}",
                )
            continue
        schema = option.get("parameter_schema") or {}
        properties = schema.get("properties") or {}
        if schema.get("additionalProperties") is False:
            for name in sorted(set(parameters) - set(properties)):
                _issue(
                    errors,
                    f"{parameter_path}.{name}",
                    "UNREGISTERED_OPTION_PARAMETER",
                    f"Parameter '{name}' is not legal for {field_path}='{option_name}'",
                )
        for name, value in parameters.items():
            parameter_schema = properties.get(name)
            if isinstance(parameter_schema, dict):
                _validate_schema_value(value, parameter_schema, f"{parameter_path}.{name}", errors)
        _validate_parameter_compatibility(parameters, option.get("compatibility") or [], parameter_path, errors)

    direct_values: list[tuple[str, Any, str]] = [
        ("network.hidden_layers", spec["network"]["hidden_layers"], "network.hidden_layers"),
        ("network.residual_connections.enabled", spec["network"]["residual_connections"]["enabled"], "network.residual_connections.enabled"),
        ("network.residual_connections.parameters", spec["network"]["residual_connections"].get("parameters") or {}, "network.residual_connections.parameters"),
        (
            "constraint_enforcement.transform_id",
            spec["constraint_enforcement"]["transform_id"],
            "constraint_enforcement.transform_id",
        ),
        (
            "constraint_enforcement.parameters",
            spec["constraint_enforcement"].get("parameters") or {},
            "constraint_enforcement.parameters",
        ),
        ("training.batch_mode", spec["training"]["batch_mode"], "training.batch_mode"),
        ("training.batch_size", spec["training"]["batch_size"], "training.batch_size"),
        ("training.gradient_clipping.enabled", spec["training"]["gradient_clipping"]["enabled"], "training.gradient_clipping.enabled"),
        ("training.gradient_clipping.parameters", spec["training"]["gradient_clipping"].get("parameters") or {}, "training.gradient_clipping.parameters"),
        ("training.early_stopping.enabled", spec["training"]["early_stopping"]["enabled"], "training.early_stopping.enabled"),
        ("training.early_stopping.parameters", spec["training"]["early_stopping"].get("parameters") or {}, "training.early_stopping.parameters"),
        (
            "training.initial_state_pretraining.enabled",
            spec["training"]["initial_state_pretraining"]["enabled"],
            "training.initial_state_pretraining.enabled",
        ),
        (
            "training.initial_state_pretraining.parameters",
            spec["training"]["initial_state_pretraining"].get("parameters") or {},
            "training.initial_state_pretraining.parameters",
        ),
    ]
    for index, phase in enumerate(spec["optimization"].get("phases") or []):
        direct_values.append(
            ("optimization.phases[].iterations", phase["iterations"], f"optimization.phases[{index}].iterations")
        )
    for index, term in enumerate(spec["loss"].get("terms") or []):
        direct_values.append(
            (
                "loss.terms[].weight",
                term["weight"],
                f"loss.terms[{index}].weight",
            )
        )
    for field_path, value, value_path in direct_values:
        value_schema = (fields.get(field_path) or {}).get("value_schema")
        if isinstance(value_schema, dict):
            _validate_schema_value(value, value_schema, value_path, errors)


def _validate_schema_value(
    value: Any,
    schema: dict[str, Any],
    path: str,
    errors: list[ValidationIssue],
) -> None:
    expected = schema.get("type")
    valid_type = {
        "boolean": isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "string": isinstance(value, str),
        "array": isinstance(value, (list, tuple)),
        "object": isinstance(value, dict),
        "null": value is None,
    }.get(expected, True)
    if not valid_type:
        _issue(errors, path, "OPTION_PARAMETER_TYPE", f"Expected {expected}, got {type(value).__name__}")
        return
    if "enum" in schema and value not in schema["enum"]:
        _issue(errors, path, "OPTION_PARAMETER_ENUM", f"Value must be one of {schema['enum']}")
    if expected in {"integer", "number"}:
        numeric = float(value)
        if "minimum" in schema and numeric < float(schema["minimum"]):
            _issue(errors, path, "OPTION_PARAMETER_RANGE", f"Value must be >= {schema['minimum']}")
        if "maximum" in schema and numeric > float(schema["maximum"]):
            _issue(errors, path, "OPTION_PARAMETER_RANGE", f"Value must be <= {schema['maximum']}")
        if "exclusiveMinimum" in schema and numeric <= float(schema["exclusiveMinimum"]):
            _issue(errors, path, "OPTION_PARAMETER_RANGE", f"Value must be > {schema['exclusiveMinimum']}")
        if "exclusiveMaximum" in schema and numeric >= float(schema["exclusiveMaximum"]):
            _issue(errors, path, "OPTION_PARAMETER_RANGE", f"Value must be < {schema['exclusiveMaximum']}")
    if expected == "array":
        if "minItems" in schema and len(value) < int(schema["minItems"]):
            _issue(errors, path, "OPTION_PARAMETER_LENGTH", f"Array needs at least {schema['minItems']} items")
        if "maxItems" in schema and len(value) > int(schema["maxItems"]):
            _issue(errors, path, "OPTION_PARAMETER_LENGTH", f"Array allows at most {schema['maxItems']} items")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                _validate_schema_value(item, item_schema, f"{path}[{index}]", errors)
    if expected == "object":
        properties = schema.get("properties") or {}
        if schema.get("additionalProperties") is False:
            for name in sorted(set(value) - set(properties)):
                _issue(errors, f"{path}.{name}", "UNREGISTERED_OPTION_PARAMETER", f"Parameter '{name}' is not legal")
        for name, item in value.items():
            item_schema = properties.get(name)
            if isinstance(item_schema, dict):
                _validate_schema_value(item, item_schema, f"{path}.{name}", errors)


def _validate_parameter_compatibility(
    parameters: dict[str, Any],
    rules: list[dict[str, Any]],
    path: str,
    errors: list[ValidationIssue],
) -> None:
    for rule in rules:
        if rule.get("rule") != "parameter_order":
            continue
        lower_name = str(rule.get("lower"))
        upper_name = str(rule.get("upper"))
        lower = parameters.get(lower_name)
        upper = parameters.get(upper_name)
        if lower is not None and upper is not None and float(lower) >= float(upper):
            _issue(
                errors,
                path,
                "OPTION_PARAMETER_COMPATIBILITY",
                f"'{lower_name}' must be smaller than '{upper_name}'",
            )


def _validate_problem_combinations(
    spec: dict[str, Any],
    problem: Any,
    builder_capabilities: dict[str, Any],
    errors: list[ValidationIssue],
) -> None:
    problem_spec = problem.get_spec()
    enforcement = spec["constraint_enforcement"]
    method = enforcement["method"]
    transform_id = enforcement["transform_id"]
    parameters = enforcement.get("parameters") or {}
    capabilities_fn = getattr(problem, "constraint_enforcement_capabilities", None)
    capabilities = capabilities_fn() if callable(capabilities_fn) else {
        "soft_penalty": {"transform_ids": ["none"]},
        "problem_hard": {"transform_ids": []},
    }
    method_capability = capabilities.get(method) if isinstance(capabilities, dict) else None
    if not isinstance(method_capability, dict):
        _issue(
            errors,
            "constraint_enforcement.method",
            "CONSTRAINT_ENFORCEMENT_UNAVAILABLE",
            f"Problem '{problem_spec.problem_id}' does not support method '{method}'",
        )
    else:
        transform_ids = set(method_capability.get("transform_ids") or [])
        if transform_id not in transform_ids:
            _issue(
                errors,
                "constraint_enforcement.transform_id",
                "HARD_TRANSFORM_UNAVAILABLE",
                f"Transform '{transform_id}' is not registered for problem '{problem_spec.problem_id}'",
            )
    if parameters:
        _issue(
            errors,
            "constraint_enforcement.parameters",
            "UNREGISTERED_OPTION_PARAMETER",
            "Registered constraint transforms currently accept no parameters",
        )
    constraints_by_id = {
        str(constraint.constraint_id): constraint
        for constraint in problem_spec.constraints
    }
    constraint_ids = set(constraints_by_id)
    aggregate_constraint_ids = {
        constraint_id
        for constraint_id, constraint in constraints_by_id.items()
        if bool((constraint.metadata or {}).get("aggregate"))
    }
    concrete_constraint_ids = constraint_ids - aggregate_constraint_ids
    required_constraint_ids = {
        constraint_id
        for constraint_id in concrete_constraint_ids
        if bool(constraints_by_id[constraint_id].required)
    }
    loss_terms = spec["loss"].get("terms") or []
    term_name_list = [str(term.get("name")) for term in loss_terms]
    term_names = set(term_name_list)
    supported_core_terms = set(
        (builder_capabilities.get("loss_term_parameters") or {}).keys()
    )
    allowed_term_names = supported_core_terms | concrete_constraint_ids
    for index, term_name in enumerate(term_name_list):
        if term_name not in allowed_term_names:
            _issue(
                errors,
                f"loss.terms[{index}].name",
                "unsupported_loss_term",
                f"Unsupported loss term '{term_name}' for problem "
                f"'{problem_spec.problem_id}'",
            )
        if term_name_list.count(term_name) > 1 and term_name_list.index(term_name) == index:
            _issue(
                errors,
                "loss.terms",
                "duplicate_loss_term",
                f"Loss term '{term_name}' must appear exactly once",
            )

    direct_constraint_terms = term_names.intersection(concrete_constraint_ids)
    legacy_constraint_terms = term_names.intersection(
        {"boundary_condition", "initial_condition"}
    )
    independent_constraint_mode = bool(direct_constraint_terms)
    if independent_constraint_mode and legacy_constraint_terms:
        _issue(
            errors,
            "loss.terms",
            "legacy_independent_loss_mixing",
            "Legacy aggregate constraint terms cannot be combined with direct "
            "PDE constraint_id loss terms",
        )
    if independent_constraint_mode:
        required_independent_terms = {"pde_residual"} | required_constraint_ids
        for missing_name in sorted(required_independent_terms - term_names):
            _issue(
                errors,
                "loss.terms",
                "required_constraint_loss_missing",
                f"Independent constraint mode requires loss term '{missing_name}'",
            )
        if spec["loss"]["aggregation"].get("name") != "weighted_sum":
            _issue(
                errors,
                "loss.aggregation.name",
                "independent_constraints_require_weighted_sum",
                "Independent constraint losses require weighted_sum aggregation",
            )

    for index, term in enumerate(loss_terms):
        constraint = constraints_by_id.get(str(term.get("name")))
        if constraint is None or str(term.get("name")) not in concrete_constraint_ids:
            continue
        term_parameters = term.get("parameters") or {}
        expected_variable = (
            constraint.derivative_variable
            or (constraint.metadata or {}).get("derivative_variable")
        )
        expected_order = (
            constraint.derivative_order
            if constraint.derivative_order is not None
            else (constraint.metadata or {}).get("derivative_order")
        )
        if (
            "derivative_variable" in term_parameters
            and expected_variable is not None
            and term_parameters["derivative_variable"] != expected_variable
        ):
            _issue(
                errors,
                f"loss.terms[{index}].parameters.derivative_variable",
                "constraint_derivative_declaration_mismatch",
                f"Expected derivative_variable='{expected_variable}'",
            )
        if (
            "derivative_order" in term_parameters
            and expected_order is not None
            and int(term_parameters["derivative_order"]) != int(expected_order)
        ):
            _issue(
                errors,
                f"loss.terms[{index}].parameters.derivative_order",
                "constraint_derivative_declaration_mismatch",
                f"Expected derivative_order={int(expected_order)}",
            )

    is_wave_1d = str(problem_spec.problem_id) == "wave_1d"
    wave_split_terms = {
        "spatial_boundary",
        "initial_displacement",
        "initial_velocity",
    }
    for unavailable_name in sorted(
        wave_split_terms.intersection(term_names) - concrete_constraint_ids
    ):
        _issue(
            errors,
            "loss.terms",
            f"{unavailable_name.upper()}_UNAVAILABLE",
            f"Problem '{problem_spec.problem_id}' has no "
            f"'{unavailable_name}' constraint",
        )
    legacy_wave_aggregate = (
        is_wave_1d
        and "boundary_condition" in term_names
        and not direct_constraint_terms
    )
    wave_has_spatial_boundary = bool(
        {"spatial_boundary", "boundary_condition"}.intersection(term_names)
    )
    governing_terms = set(GOVERNING_LOSS_TERM_NAMES)
    _validate_governing_loss_component_uniqueness(
        loss_terms,
        tuple(str(law.law_id) for law in problem_spec.governing_laws),
        errors,
    )
    if not term_names.intersection(governing_terms):
        _issue(
            errors,
            "loss.terms",
            "GOVERNING_LOSS_REQUIRED",
            "At least one positive-weight governing PDE loss term is required",
        )
    interior_points = int(
        ((spec["sampling"].get("interior") or {}).get("parameters") or {}).get(
            "n_points", 0
        )
        or 0
    )
    if interior_points <= 0:
        _issue(
            errors,
            "sampling.interior.parameters.n_points",
            "INTERIOR_SAMPLES_REQUIRED",
            "A governing PDE loss requires positive interior sample coverage",
        )
    if method == "soft_penalty":
        for category, loss_name in (
            ("initial", "initial_condition"),
            ("boundary", "boundary_condition"),
        ):
            if category not in constraint_ids:
                continue
            if is_wave_1d and category == "initial":
                continue
            category_constraint_ids = {
                constraint_id
                for constraint_id in concrete_constraint_ids
                if str(constraints_by_id[constraint_id].target or "") == category
            }
            category_uses_direct_constraints = bool(
                category_constraint_ids.intersection(direct_constraint_terms)
            )
            if (
                is_wave_1d
                and category == "boundary"
                and (
                    "spatial_boundary" in term_names
                    or legacy_wave_aggregate
                )
            ):
                loss_name = (
                    "spatial_boundary"
                    if "spatial_boundary" in term_names
                    else "boundary_condition"
                )
            if loss_name not in term_names and not category_uses_direct_constraints:
                _issue(
                    errors,
                    "loss.terms",
                    f"{category.upper()}_LOSS_REQUIRED_FOR_SOFT_PENALTY",
                    f"soft_penalty requires a positive-weight {loss_name} term",
                )
            sample_points = int(
                ((spec["sampling"].get(category) or {}).get("parameters") or {}).get(
                    "n_points", 0
                )
                or 0
            )
            if sample_points <= 0:
                _issue(
                    errors,
                    f"sampling.{category}.parameters.n_points",
                    f"{category.upper()}_SAMPLES_REQUIRED_FOR_SOFT_PENALTY",
                    f"soft_penalty requires positive {category} sample coverage",
                )
    if "initial_condition" in term_names and "initial" not in constraint_ids:
        _issue(errors, "loss.terms", "INITIAL_CONDITION_UNAVAILABLE", "Problem has no initial condition")
    if "boundary_condition" in term_names and "boundary" not in constraint_ids:
        _issue(errors, "loss.terms", "BOUNDARY_CONDITION_UNAVAILABLE", "Problem has no boundary condition")
    if "spatial_boundary" in term_names and "spatial_boundary" not in constraint_ids:
        _issue(
            errors,
            "loss.terms",
            "SPATIAL_BOUNDARY_UNAVAILABLE",
            "Problem has no spatial_boundary constraint",
        )
    if is_wave_1d:
        wave_independent_mode = bool(term_names.intersection(wave_split_terms))
        initial_points = int(
            ((spec["sampling"].get("initial") or {}).get("parameters") or {}).get(
                "n_points", 0
            )
            or 0
        )
        if initial_points <= 0 and wave_independent_mode:
            _issue(
                errors,
                "sampling.initial.parameters.n_points",
                "wave_initial_sampling_zero",
                "wave_1d requires positive initial-state sampling",
            )
        if "initial_displacement" not in term_names and wave_independent_mode:
            _issue(
                errors,
                "loss.terms",
                "wave_initial_displacement_missing",
                "wave_1d requires the initial_displacement loss",
            )
        if "initial_velocity" not in term_names and wave_independent_mode:
            _issue(
                errors,
                "loss.terms",
                "wave_initial_velocity_missing",
                "wave_1d requires the initial_velocity loss",
            )
        for term in loss_terms:
            if term.get("name") != "initial_velocity":
                continue
            term_parameters = term.get("parameters") or {}
            if (
                term_parameters.get("derivative_variable") != "t"
                or int(term_parameters.get("derivative_order") or 0) != 1
            ):
                _issue(
                    errors,
                    "loss.terms",
                    "wave_initial_velocity_derivative_invalid",
                    "initial_velocity must use the first derivative with respect to t",
                )
        if not wave_has_spatial_boundary and wave_independent_mode:
            _issue(
                errors,
                "loss.terms",
                "wave_spatial_boundary_missing",
                "wave_1d requires the spatial_boundary loss",
            )
        if (
            wave_independent_mode
            and (
                "initial_displacement" not in term_names
                or "initial_velocity" not in term_names
                or not wave_has_spatial_boundary
            )
        ):
            _issue(
                errors,
                "loss.terms",
                "wave_split_constraints_incomplete",
                "Wave split constraints must include spatial_boundary, "
                "initial_displacement, and initial_velocity exactly once",
            )
        architecture = str(spec["network"].get("architecture") or "")
        initializer = str(spec["network"]["initialization"].get("name") or "")
        if architecture == "siren_mlp" and initializer != "siren":
            _issue(
                errors,
                "network.initialization.name",
                "wave_siren_initializer_required",
                "siren_mlp requires the layer-aware siren initializer",
            )
        if initializer == "siren" and architecture != "siren_mlp":
            _issue(
                errors,
                "network.initialization.name",
                "wave_siren_initializer_incompatible",
                "The siren initializer is only valid with siren_mlp",
            )
        time_strategy = spec["training"]["time_strategy"]
        if time_strategy.get("name") == "time_window_curriculum":
            time_parameters = time_strategy.get("parameters") or {}
            fractions = [float(value) for value in time_parameters.get("window_fractions") or []]
            iterations = [int(value) for value in time_parameters.get("iterations_per_window") or []]
            if (
                not fractions
                or len(fractions) != len(iterations)
                or any(right <= left for left, right in zip(fractions, fractions[1:]))
                or fractions[-1] != 1.0
            ):
                _issue(
                    errors,
                    "training.time_strategy.parameters",
                    "wave_time_window_curriculum_invalid",
                    "Wave time windows must be strictly increasing, match iteration counts, and end at 1.0",
                )
        pretraining = spec["training"]["initial_state_pretraining"]
        if pretraining.get("enabled") and int(
            (pretraining.get("parameters") or {}).get("iterations") or 0
        ) <= 0:
            _issue(
                errors,
                "training.initial_state_pretraining.parameters.iterations",
                "wave_initial_pretraining_iterations_invalid",
                "Enabled initial-state pretraining requires positive iterations",
            )
    for index, term in enumerate(loss_terms):
        name = term.get("name")
        parameters = term.get("parameters") or {}
        if name == "pde_component" and not any(
            key in parameters for key in ("component_name", "component_index")
        ):
            _issue(
                errors,
                f"loss.terms[{index}].parameters",
                "PDE_COMPONENT_SELECTOR_REQUIRED",
                "pde_component requires component_name or component_index",
            )
        if name == "constraint_component":
            component_name = str(parameters.get("component_name") or "")
            if not component_name:
                _issue(
                    errors,
                    f"loss.terms[{index}].parameters.component_name",
                    "CONSTRAINT_COMPONENT_NAME_REQUIRED",
                    "constraint_component requires component_name",
                )
            elif component_name not in constraint_ids:
                _issue(
                    errors,
                    f"loss.terms[{index}].parameters.component_name",
                    "CONSTRAINT_COMPONENT_UNAVAILABLE",
                    f"Problem has no constraint component '{component_name}'",
                )
    phases = spec["optimization"].get("phases") or []
    weighting_name = str(spec["loss"]["weighting_strategy"].get("name") or "")
    if weighting_name == "ntk":
        if spec["loss"]["aggregation"].get("name") != "weighted_sum":
            _issue(
                errors,
                "loss.aggregation.name",
                "NTK_REQUIRES_WEIGHTED_SUM",
                "PINNacle NTK requires weighted_sum aggregation",
            )
        if legacy_wave_aggregate:
            _issue(
                errors,
                "loss.terms",
                "NTK_REQUIRES_SPLIT_WAVE_CONSTRAINTS",
                "Wave NTK requires the three stable split constraint terms",
            )
        ntk_parameters = spec["loss"]["weighting_strategy"].get("parameters") or {}
        minimum = float(ntk_parameters.get("min_weight", 1.0e-6))
        maximum = float(ntk_parameters.get("max_weight", 1.0e6))
        if minimum >= maximum:
            _issue(
                errors,
                "loss.weighting_strategy.parameters",
                "NTK_WEIGHT_BOUNDS_INVALID",
                "min_weight must be smaller than max_weight",
            )
    multiadam_indices = [index for index, phase in enumerate(phases) if phase.get("optimizer") == "multiadam"]
    if multiadam_indices:
        if len(spec["loss"].get("terms") or []) < 2:
            _issue(errors, "loss.terms", "MULTIADAM_REQUIRES_MULTIPLE_LOSSES", "MultiAdam requires at least two loss terms")
        if spec["training"]["gradient_clipping"].get("enabled"):
            _issue(errors, "training.gradient_clipping", "MULTIADAM_CLIPPING_UNSUPPORTED", "MultiAdam currently does not support gradient clipping")
        if weighting_name != "fixed":
            _issue(
                errors,
                "loss.weighting_strategy.name",
                "MULTIADAM_DYNAMIC_WEIGHTING_CONFLICT",
                "MultiAdam cannot be combined with LRA, NTK, or another dynamic weighting strategy",
            )
        if any(phase.get("optimizer") == "lbfgs" for phase in phases):
            _issue(
                errors,
                "optimization.phases",
                "MULTIADAM_LBFGS_CONFLICT",
                "The first MultiAdam reproduction cannot include an L-BFGS phase",
            )
        for index in multiadam_indices:
            parameters = phases[index].get("parameters") or {}
            group_weights = parameters.get("group_weights")
            grouping = parameters.get("grouping")
            expected_count = 2 if grouping is not None else len(loss_terms)
            if group_weights is not None and len(group_weights) != expected_count:
                _issue(
                    errors,
                    f"optimization.phases[{index}].parameters.group_weights",
                    "MULTIADAM_WEIGHT_COUNT_MISMATCH",
                    f"MultiAdam group_weights must contain {expected_count} values",
                )
            if grouping == "dirichlet_vs_non_dirichlet":
                if not is_wave_1d or not wave_split_terms.issubset(term_names):
                    _issue(
                        errors,
                        f"optimization.phases[{index}].parameters.grouping",
                        "MULTIADAM_DIRICHLET_GROUPING_UNAVAILABLE",
                        "dirichlet_vs_non_dirichlet requires identifiable Wave1D split constraints",
                    )
            if grouping is not None and bool(parameters.get("normalize_updates", False)):
                _issue(
                    errors,
                    f"optimization.phases[{index}].parameters.normalize_updates",
                    "MULTIADAM_GROUP_NORMALIZATION_UNSUPPORTED",
                    "PINNacle-compatible grouped MultiAdam does not L2-normalize group updates",
                )
    lbfgs_indices = [index for index, phase in enumerate(phases) if phase.get("optimizer") == "lbfgs"]
    if lbfgs_indices:
        if spec["training"].get("batch_mode") != "full_batch":
            _issue(errors, "training.batch_mode", "LBFGS_REQUIRES_FULL_BATCH", "L-BFGS requires full_batch training")
        if lbfgs_indices[-1] != len(phases) - 1 or len(lbfgs_indices) > 1:
            _issue(errors, "optimization.phases", "LBFGS_MUST_BE_LAST", "The current trainer supports one final L-BFGS phase")
    adaptive = spec["sampling"]["adaptive_refinement"]
    if adaptive.get("enabled") and not callable(getattr(problem, "compute_governing_residuals", None)):
        _issue(errors, "sampling.adaptive_refinement", "RAR_REQUIRES_RESIDUAL", "Adaptive refinement requires a residual evaluator")


def _validate_budget(spec: dict[str, Any], problem: Any, config: dict[str, Any], errors: list[ValidationIssue]) -> dict[str, int]:
    total_iterations = sum(int(phase.get("iterations") or 0) for phase in spec["optimization"].get("phases") or [])
    try:
        sampling_points = resolve_sampling_budget_plan(
            problem, spec
        ).total_training_points
    except SamplingContractError as exc:
        # Keep the report structurally complete while rejecting a contract
        # that cannot produce the runtime batch promised by AlgorithmSpec.
        sampling_points = sum(
            int(
                ((spec["sampling"].get(section) or {}).get("parameters") or {}).get(
                    "n_points"
                )
                or 0
            )
            for section in ("interior", "boundary", "initial")
        )
        _issue(
            errors,
            "sampling",
            "INVALID_SAMPLING_CONTRACT",
            str(exc),
        )
    problem_spec = problem.get_spec()
    input_dim = len(problem_spec.input_variables)
    output_dim = len(problem_spec.output_variables)
    network = spec["network"]
    architecture = network.get("architecture")
    extra = network.get("extra_parameters") or {}
    effective_input = input_dim
    extra_trainable = 0
    transform = network.get("input_transform") or {}
    transform_name = transform.get("name")
    transform_parameters = transform.get("parameters") or {}
    if transform_name in {
        "fourier_features",
        "periodic_positional_encoding",
    }:
        count = int(transform_parameters.get("frequency_count", 8))
        effective_input = input_dim * count * 2 + (
            input_dim if transform_parameters.get("include_raw_input", False) else 0
        )
        if transform_parameters.get("trainable_frequencies", False):
            extra_trainable += count
    elif transform_name == "multiscale_fourier_features":
        count = int(transform_parameters.get("frequency_count", 8))
        bands = len(
            transform_parameters.get("frequency_scales") or [1.0, 2.0, 4.0]
        )
        effective_input = input_dim * count * bands * 2 + (
            input_dim if transform_parameters.get("include_raw_input", False) else 0
        )
        if transform_parameters.get("trainable_frequencies", False):
            extra_trainable += count * bands
    if architecture == "fourier_mlp":
        frequencies = int(
            extra.get("frequency_count", extra.get("num_frequencies", 8))
        )
        effective_input = effective_input * frequencies * 2 + (
            effective_input if extra.get("include_raw_input", False) else 0
        )
        if extra.get("trainable_frequencies", extra.get("trainable", False)):
            extra_trainable += frequencies
    elif architecture == "multiscale_fourier_mlp":
        frequencies = int(extra.get("frequency_count", 8))
        bands = len(extra.get("frequency_scales") or [1.0, 2.0, 4.0])
        effective_input = effective_input * frequencies * bands * 2 + (
            effective_input if extra.get("include_raw_input", False) else 0
        )
        if extra.get("trainable_frequencies", False):
            extra_trainable += frequencies * bands
    elif architecture == "multiscale_mlp":
        effective_input = input_dim * len(extra.get("scales") or [1.0, 2.0, 4.0])
    hidden_layers = list(network.get("hidden_layers", []))
    widths = [effective_input, *hidden_layers, output_dim]
    if architecture in {"modified_mlp", "gated_mlp"}:
        model_parameters = sum(
            2 * (effective_input + 1) * right + (left + 1) * right
            for left, right in zip([effective_input, *hidden_layers[:-1]], hidden_layers)
        )
        model_parameters += (hidden_layers[-1] + 1) * output_dim
    elif architecture == "factorized_mlp":
        configured_rank = max(1, int(extra.get("rank", 16)))
        model_parameters = 0
        for left, right in zip([effective_input, *hidden_layers[:-1]], hidden_layers):
            rank = min(configured_rank, left, right)
            model_parameters += left * rank + (rank + 1) * right
        model_parameters += (hidden_layers[-1] + 1) * output_dim
    elif architecture in {"laaf", "laaf_mlp", "gaaf", "gaaf_mlp"}:
        model_parameters = sum((left + 1) * right for left, right in zip(widths, widths[1:]))
        model_parameters += sum(hidden_layers) if architecture in {"laaf", "laaf_mlp"} else len(hidden_layers)
    elif architecture in {"parallel_fnn", "pfnn"}:
        branch_widths = [effective_input, *hidden_layers, 1]
        model_parameters = output_dim * sum(
            (left + 1) * right for left, right in zip(branch_widths, branch_widths[1:])
        )
    else:
        model_parameters = sum((left + 1) * right for left, right in zip(widths, widths[1:]))
    residual_enabled = architecture in {"resnet", "residual_mlp"} or bool(network["residual_connections"].get("enabled"))
    if residual_enabled:
        hidden_widths = [effective_input, *hidden_layers]
        residual_multiplier = output_dim if architecture in {"parallel_fnn", "pfnn"} else 1
        model_parameters += residual_multiplier * sum(
            left * right for left, right in zip(hidden_widths, hidden_widths[1:]) if left != right
        )
    model_parameters += extra_trainable
    usage = {
        "total_iterations": total_iterations,
        "sampling_points": sampling_points,
        "estimated_model_parameters": model_parameters,
    }
    configured_sampling_limit = config.get("maximum_sampling_points")
    limits = {
        "total_iterations": int(config.get("maximum_total_iterations", 2**63 - 1)),
        "sampling_points": (
            None if configured_sampling_limit is None else int(configured_sampling_limit)
        ),
        "estimated_model_parameters": int(config.get("maximum_model_parameters", 2**63 - 1)),
    }
    paths = {
        "total_iterations": "optimization.phases",
        "sampling_points": "sampling",
        "estimated_model_parameters": "network.hidden_layers",
    }
    for key, value in usage.items():
        if limits[key] is not None and value > limits[key]:
            _issue(errors, paths[key], "BUDGET_EXCEEDED", f"{key}={value} exceeds external limit {limits[key]}")
    required_iterations = config.get("required_total_iterations")
    if required_iterations is not None and total_iterations != int(required_iterations):
        _issue(
            errors,
            "optimization.phases",
            "REQUIRED_ITERATION_BUDGET_NOT_MET",
            f"total_iterations={total_iterations} must equal the required experiment budget {int(required_iterations)}",
        )
    return usage


def _walk_named_fields(value: Any, field_name: str, path: str = "$"):
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{path}.{key}" if path != "$" else key
            if key == field_name:
                yield child, item
            yield from _walk_named_fields(item, field_name, child)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk_named_fields(item, field_name, f"{path}[{index}]")


def _normalized_name(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _normalize_nested(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalize_nested(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_nested(item) for item in value]
    return value


def _sort_dicts(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sort_dicts(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_sort_dicts(item) for item in value]
    return value
