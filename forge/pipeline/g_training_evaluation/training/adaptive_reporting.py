"""Compact, deterministic reporting for adaptive Trainer execution."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import math
from statistics import median
from typing import Any, Mapping


@dataclass
class AdaptiveComputeBudgetLedger:
    optimizer_steps: int = 0
    autograd_backward_calls: int = 0
    component_gradient_diagnostic_calls: int = 0
    component_gradient_diagnostic_count: int = 0
    sampling_training_points_scored: int = 0
    sampling_candidate_points_scored: int = 0
    sampling_updates: int = 0
    physics_probe_evaluations: int = 0
    physics_probe_points_evaluated: int = 0
    curriculum_diagnostic_checks: int = 0
    curriculum_runtime_refreshes: int = 0
    lbfgs_outer_calls: int = 0
    lbfgs_closure_evaluations: int = 0
    recovery_adam_steps: int = 0
    rollback_count: int = 0
    checkpoint_count: int = 0
    adaptive_control_wall_time: float = 0.0
    total_training_wall_time: float = 0.0

    def state_dict(self) -> dict[str, Any]:
        return asdict(self)

    def load_state_dict(
        self, state: Mapping[str, Any] | None, *, preserve_consumed: bool = False
    ) -> None:
        source = dict(state or {})
        for name in self.__dataclass_fields__:
            if name not in source:
                continue
            incoming = source[name]
            current = getattr(self, name)
            if preserve_consumed:
                incoming = max(current, incoming)
            setattr(self, name, type(current)(incoming))

    def reconcile(
        self,
        *,
        actual_iterations: int,
        phase_iterations: Mapping[str, int],
        optimizer_audit: Mapping[str, Any],
        component_gradient_audits: list[dict[str, Any]],
        sampling_summary: Mapping[str, Any],
        curriculum_summary: Mapping[str, Any],
        physics_history: list[dict[str, Any]],
        probe_component_budgets: Mapping[str, int],
        adaptive_summary: Mapping[str, Any],
        adaptive_history: list[dict[str, Any]],
        training_wall_time: float,
    ) -> dict[str, Any]:
        lbfgs_audits = [
            value for value in optimizer_audit.values()
            if isinstance(value, Mapping) and "closure_evaluations" in value
        ]
        closures = sum(int(item.get("closure_evaluations") or 0) for item in lbfgs_audits)
        outer = sum(int(item.get("outer_step_calls") or 0) for item in lbfgs_audits)
        self.optimizer_steps = max(self.optimizer_steps, int(actual_iterations))
        self.lbfgs_closure_evaluations = max(self.lbfgs_closure_evaluations, closures)
        self.lbfgs_outer_calls = max(self.lbfgs_outer_calls, outer)
        first_order_steps = max(0, int(actual_iterations) - sum(
            int(item.get("completed_internal_iterations") or 0) for item in lbfgs_audits
        ))
        self.autograd_backward_calls = max(
            self.autograd_backward_calls, first_order_steps + closures
        )
        self.component_gradient_diagnostic_calls = max(
            self.component_gradient_diagnostic_calls, len(component_gradient_audits)
        )
        gradient_count = sum(
            len(dict(item.get("component_grad_norms") or {}))
            for item in component_gradient_audits
        )
        self.component_gradient_diagnostic_count = max(
            self.component_gradient_diagnostic_count, gradient_count
        )
        self.sampling_training_points_scored = max(
            self.sampling_training_points_scored,
            int(sampling_summary.get("sampling_points_diagnosed") or 0),
        )
        self.sampling_candidate_points_scored = max(
            self.sampling_candidate_points_scored,
            int(sampling_summary.get("candidate_points_evaluated") or 0),
        )
        self.sampling_updates = max(
            self.sampling_updates,
            int(sampling_summary.get("sampling_updates_executed") or 0),
        )
        self.curriculum_diagnostic_checks = max(
            self.curriculum_diagnostic_checks,
            int(curriculum_summary.get("curriculum_diagnostic_checks") or 0),
        )
        self.curriculum_runtime_refreshes = max(
            self.curriculum_runtime_refreshes,
            int(curriculum_summary.get("curriculum_advances_executed") or 0),
        )
        self.physics_probe_evaluations = max(
            self.physics_probe_evaluations, len(physics_history)
        )
        points_per_probe = sum(max(0, int(value)) for value in probe_component_budgets.values())
        self.physics_probe_points_evaluated = max(
            self.physics_probe_points_evaluated,
            len(physics_history) * points_per_probe,
        )
        self.recovery_adam_steps = max(
            self.recovery_adam_steps,
            int(adaptive_summary.get("recovery_adam_steps") or adaptive_summary.get("recovered_iteration_budget") or 0),
        )
        self.rollback_count = max(
            self.rollback_count,
            int(adaptive_summary.get("rollbacks") or 0),
        )
        checkpoint_events = sum(
            "checkpoint" in str(item.get("event_type") or "")
            or bool(item.get("checkpoint_id"))
            for item in adaptive_history
        )
        self.checkpoint_count = max(self.checkpoint_count, int(checkpoint_events))
        self.adaptive_control_wall_time = max(
            self.adaptive_control_wall_time,
            float(sampling_summary.get("sampling_compute_time") or 0.0)
            + float(curriculum_summary.get("curriculum_runtime_refresh_cost") or 0.0)
            + float(curriculum_summary.get("curriculum_validation_cost") or 0.0),
        )
        self.total_training_wall_time = max(
            self.total_training_wall_time, float(training_wall_time)
        )
        return self.state_dict()


def build_adaptive_control_summary(
    *,
    spec: Mapping[str, Any],
    runtime_loss_weights: Mapping[str, float],
    component_gradient_audits: list[dict[str, Any]],
    adaptive_history: list[dict[str, Any]],
    adaptive_summary: Mapping[str, Any],
    sampling_summary: Mapping[str, Any],
    curriculum_summary: Mapping[str, Any],
    physics_history: list[dict[str, Any]],
    best_training_loss: Mapping[str, Any] | None,
    final_model_policy: str,
    compute_budget: Mapping[str, Any],
    maximum_components: int = 12,
) -> dict[str, Any]:
    """Build the prompt-safe outcome summary; never include reference metrics."""

    extra = dict((spec.get("training") or {}).get("extra_parameters") or {})
    control = dict(extra.get("adaptive_control") or {})
    weighting = dict((spec.get("loss") or {}).get("weighting_strategy") or {})
    events = [dict(item) for item in adaptive_history if isinstance(item, Mapping)]

    def event_count(controller: str, *tokens: str) -> int:
        return sum(
            str(item.get("controller") or "") == controller
            and any(token in str(item.get("event_type") or "") for token in tokens)
            for item in events
        )

    weights = _bounded_mapping(runtime_loss_weights, maximum_components)
    gradient_ratios = sorted(
        float(item["max_to_min_nonzero_component_ratio"])
        for item in component_gradient_audits
        if isinstance(item.get("max_to_min_nonzero_component_ratio"), (int, float))
        and math.isfinite(float(item["max_to_min_nonzero_component_ratio"]))
    )
    conflict_count = sum(bool(item.get("gradient_conflict_detected")) for item in component_gradient_audits)
    curriculum = dict(control.get("curriculum_control") or {})
    iterations_per_level = dict(curriculum_summary.get("iterations_per_level") or {})
    level_states = []
    for curriculum_id, levels in sorted(iterations_per_level.items()):
        level_states.append({"curriculum_id": str(curriculum_id), "iterations": _bounded_mapping(levels, maximum_components) if isinstance(levels, Mapping) else levels})
    physics_enabled = bool((extra.get("physics_validation") or {}).get("enabled"))
    final_score = physics_history[-1].get("aggregate_score") if physics_history else None
    best = dict(best_training_loss or {})
    total_wall = float(compute_budget.get("total_training_wall_time") or 0.0)
    adaptive_wall = float(compute_budget.get("adaptive_control_wall_time") or 0.0)
    executed_advances = int(curriculum_summary.get("curriculum_advances_executed") or 0)
    forced_advances = int(curriculum_summary.get("curriculum_advances_forced") or 0)
    return _sorted({
        "schema_version": "1.0",
        "loss_weighting": {
            "strategy": weighting.get("name", "fixed"),
            "weight_updates_proposed": event_count("gradient_balance", "proposed"),
            "weight_updates_executed": int(adaptive_summary.get("gradient_balance_actions") or 0),
            "weight_updates_rejected": event_count("gradient_balance", "rejected"),
            "weight_update_rollbacks": event_count("checkpoint_rollback", "rollback"),
            "final_component_weights": weights,
            "minimum_component_weight": min(weights.values()) if weights else None,
            "maximum_component_weight": max(weights.values()) if weights else None,
            "weight_saturation_fraction": _saturation_fraction(weights, weighting.get("parameters") or {}),
            "gradient_conflict_count": conflict_count,
            "median_gradient_imbalance": median(gradient_ratios) if gradient_ratios else None,
        },
        "lbfgs_recovery": {
            "stall_detected": bool(adaptive_summary.get("lbfgs_stall_detected")),
            "stall_iteration": adaptive_summary.get("lbfgs_stall_iteration"),
            "stall_reason": adaptive_summary.get("stall_reason"),
            "planned_lbfgs_steps": int(adaptive_summary.get("planned_lbfgs_iterations") or 0),
            "executed_lbfgs_steps": int(adaptive_summary.get("executed_lbfgs_iterations") or 0),
            "closure_evaluations": int(compute_budget.get("lbfgs_closure_evaluations") or 0),
            "recovered_budget": int(adaptive_summary.get("recovered_iteration_budget") or 0),
            "recovery_adam_steps": int(compute_budget.get("recovery_adam_steps") or 0),
            "unused_budget": int(adaptive_summary.get("unused_iteration_budget") or 0),
            "final_recovery_status": adaptive_summary.get("recovery_action") or "not_triggered",
        },
        "sampling_control": {
            "strategy": (control.get("sampling_control") or {}).get("strategy", "none"),
            "diagnostic_checks": int(sampling_summary.get("sampling_diagnostic_evaluations") or 0),
            "points_scored": int(sampling_summary.get("sampling_points_diagnosed") or 0),
            "candidate_points_scored": int(sampling_summary.get("candidate_points_evaluated") or 0),
            "updates_proposed": event_count("fixed_budget_sampling", "proposed"),
            "updates_executed": int(sampling_summary.get("sampling_updates_executed") or 0),
            "updates_rejected": int(sampling_summary.get("sampling_updates_rejected") or 0),
            "updates_rolled_back": int(sampling_summary.get("sampling_updates_rolled_back") or 0),
            "points_replaced": int(sampling_summary.get("sampling_points_replaced") or 0),
            "successful_observation_count": event_count("fixed_budget_sampling", "accepted", "succeeded"),
            "failed_observation_count": event_count("fixed_budget_sampling", "rollback", "failed"),
        },
        "curriculum_control": {
            "strategy": curriculum.get("strategy", "none"),
            "initial_level": 0 if curriculum.get("enabled") else None,
            "final_level": max((len(value) - 1 for value in iterations_per_level.values() if isinstance(value, Mapping)), default=None),
            "levels_completed": executed_advances - int(curriculum_summary.get("curriculum_advances_rolled_back") or 0),
            "normal_advances": executed_advances - forced_advances,
            "forced_advances": forced_advances,
            "rejected_advances": int(curriculum_summary.get("curriculum_advances_rejected") or 0),
            "rolled_back_advances": int(curriculum_summary.get("curriculum_advances_rolled_back") or 0),
            "iterations_per_level": level_states,
            "unused_levels": _bounded_mapping(curriculum_summary.get("unused_levels") or {}, maximum_components),
        },
        "physics_validation_and_model_selection": {
            "physics_probe_enabled": physics_enabled,
            "best_training_loss_iteration": best.get("iteration"),
            "best_training_loss_stage": best.get("stage") or best.get("stage_id"),
            "best_training_loss": best.get("training_loss"),
            "final_physics_score": final_score,
            "final_model_policy": str(final_model_policy),
            "last_vs_best_selected": (
                "best"
                if str(final_model_policy)
                in {"best_train_loss", "best_reference_mse"}
                else "last"
            ),
        },
        "cost": {
            "adaptive_control_wall_time": adaptive_wall,
            "total_training_wall_time": total_wall,
            "adaptive_overhead_fraction": adaptive_wall / total_wall if total_wall > 0.0 else 0.0,
            "gradient_diagnostic_calls": int(compute_budget.get("component_gradient_diagnostic_calls") or 0),
            "sampling_points_scored": int(compute_budget.get("sampling_training_points_scored") or 0) + int(compute_budget.get("sampling_candidate_points_scored") or 0),
            "validation_points_evaluated": int(compute_budget.get("physics_probe_points_evaluated") or 0),
            "closure_evaluations": int(compute_budget.get("lbfgs_closure_evaluations") or 0),
        },
    })


def _bounded_mapping(value: Mapping[str, Any], limit: int) -> dict[str, Any]:
    items = sorted(((str(key), deepcopy(item)) for key, item in dict(value or {}).items()), key=lambda pair: pair[0])
    return dict(items[: max(1, int(limit))])


def _saturation_fraction(weights: Mapping[str, float], parameters: Mapping[str, Any]) -> float:
    if not weights:
        return 0.0
    lower = float(parameters.get("minimum_weight", -math.inf))
    upper = float(parameters.get("maximum_weight", math.inf))
    saturated = sum(math.isclose(float(value), lower) or math.isclose(float(value), upper) for value in weights.values())
    return saturated / len(weights)


def _sorted(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _sorted(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, list):
        return [_sorted(item) for item in value]
    return deepcopy(value)


__all__ = ["AdaptiveComputeBudgetLedger", "build_adaptive_control_summary"]
