"""Finite, PDE-independent controller for advancing one registered level."""

from __future__ import annotations

import math
from typing import Any

from .state import AdaptiveAction, AdvanceCurriculumStateAction, TrainerObservableState


DEFAULTS: dict[str, Any] = {
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


class CurriculumAdvanceController:
    name = "curriculum_advance"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = {**DEFAULTS, **dict(config or {})}
        self.patience_counts: dict[str, int] = {}
        self.last_validation_scores: dict[str, float] = {}
        self.last_component_scores: dict[str, dict[str, float]] = {}
        self.last_action_iterations: dict[str, int] = {}
        self.rollback_cooldown_until: dict[str, int] = {}
        self.advance_count = 0
        self.forced_advance_count = 0
        self.rollback_count = 0
        self.last_observation: dict[str, Any] = {}
        self.pending_action: AdaptiveAction | None = None

    def diagnostic_due(self, state: TrainerObservableState) -> tuple[bool, str]:
        stage = state.stage_descriptor
        if stage is None or not bool(stage.first_order) or not bool(
            stage.dynamic_objective_allowed
        ):
            return False, "stage_frozen_or_not_first_order"
        if state.iteration <= 0 or state.iteration % int(
            self.config["validation_interval"]
        ) != 0:
            return False, "validation_interval_not_reached"
        if state.pending_rollback:
            return False, "pending_rollback"
        if state.coordinator_observation_active:
            return False, "global_observation_window"
        if self.advance_count >= int(self.config["maximum_advances"]):
            return False, "maximum_advances_reached"
        if state.validation_state is None:
            return False, "physics_validation_unavailable"
        return True, "diagnostic_due"

    def observe(self, state: TrainerObservableState) -> dict[str, Any]:
        self.pending_action = None
        due, due_reason = self.diagnostic_due(state)
        if not due:
            self.last_observation = {"eligible": False, "reason": due_reason}
            return dict(self.last_observation)

        validation = state.validation_state
        aggregate = float(getattr(validation, "aggregate_score", float("inf")))
        components = {
            str(name): float(value)
            for name, value in dict(
                getattr(validation, "normalized_component_scores", {}) or {}
            ).items()
        }
        remaining = int(
            state.curriculum_budget_state.get(
                "remaining_iteration_budget", state.remaining_iteration_budget
            )
        )
        minimum_remaining = int(
            self.config["minimum_remaining_budget_after_advance"]
        )
        candidates: list[tuple[bool, str, dict[str, Any], dict[str, Any]]] = []
        reasons: dict[str, str] = {}
        for curriculum_id, raw_snapshot in sorted(state.curriculum_states.items()):
            snapshot = dict(raw_snapshot)
            current = int(snapshot["current_level_index"])
            final = int(snapshot["final_level_index"])
            residence = int(snapshot.get("iterations_in_current_level", 0))
            if current >= final:
                reasons[curriculum_id] = "final_level_reached"
                continue
            if state.iteration < int(
                self.rollback_cooldown_until.get(curriculum_id, 0)
            ):
                reasons[curriculum_id] = "rollback_cooldown"
                continue
            last_action = self.last_action_iterations.get(curriculum_id)
            if last_action is not None and state.iteration - last_action < int(
                self.config["cooldown_iterations"]
            ):
                reasons[curriculum_id] = "controller_cooldown"
                continue
            diagnostic = dict(state.curriculum_diagnostics.get(curriculum_id) or {})
            forced = residence >= int(self.config["maximum_iterations_per_level"])
            policy = str(self.config["forced_advance_policy"])
            if forced and policy == "terminate_candidate":
                self.last_observation = {
                    "eligible": True,
                    "reason": "maximum_residence_terminate_candidate",
                    "curriculum_id": curriculum_id,
                    "iterations_in_level": residence,
                    "trigger_category": "maximum_residence_termination",
                }
                self.pending_action = AdaptiveAction(
                    action_type="terminate_candidate",
                    controller=self.name,
                    target_component_ids=(curriculum_id,),
                    after={
                        "recovery_adam_iterations": 0,
                        "termination_reason": "curriculum_maximum_residence",
                    },
                    reason=dict(self.last_observation),
                )
                return dict(self.last_observation)
            if forced and policy == "hold_until_budget_end":
                reasons[curriculum_id] = f"maximum_residence_{policy}"
                continue
            if remaining < minimum_remaining:
                reasons[curriculum_id] = "remaining_budget_insufficient"
                continue
            if residence < int(self.config["minimum_iterations_per_level"]):
                reasons[curriculum_id] = "minimum_residence_not_reached"
                continue

            associated_ids = tuple(diagnostic.get("associated_component_ids") or ())
            critical_ids = tuple(diagnostic.get("critical_component_ids") or ())
            stable, statistics = self._validation_stable(
                curriculum_id=curriculum_id,
                aggregate=aggregate,
                component_scores=components,
                associated_component_ids=associated_ids,
                critical_component_ids=critical_ids,
            )
            if stable:
                self.patience_counts[curriculum_id] = (
                    self.patience_counts.get(curriculum_id, 0) + 1
                )
            else:
                self.patience_counts[curriculum_id] = 0
            normal = self.patience_counts[curriculum_id] >= int(
                self.config["patience"]
            )
            if forced:
                if not bool(diagnostic.get("force_advance_supported", False)):
                    reasons[curriculum_id] = "forced_advance_not_supported"
                    continue
            elif not normal:
                reasons[curriculum_id] = "physics_validation_not_stable"
                continue
            statistics.update(
                {
                    "trigger_category": (
                        "maximum_residence_forced" if forced else "physics_validation_stable"
                    ),
                    "iterations_in_level": residence,
                    "validation_checks": self.patience_counts[curriculum_id],
                    "aggregate_score": aggregate,
                }
            )
            candidates.append((forced, curriculum_id, snapshot, {
                **diagnostic, "trigger_statistics": statistics,
            }))

        if not candidates:
            self.last_observation = {
                "eligible": True,
                "reason": "no_curriculum_ready",
                "curriculum_reasons": reasons,
                "patience_counts": dict(self.patience_counts),
            }
            return dict(self.last_observation)

        forced, curriculum_id, snapshot, diagnostic = sorted(
            candidates, key=lambda item: (not item[0], item[1])
        )[0]
        current = int(snapshot["current_level_index"])
        target = current + 1
        level_ids = tuple(diagnostic["level_ids"])
        statistics = dict(diagnostic["trigger_statistics"])
        observation_budget = int(self.config["observation_window_iterations"])
        if forced:
            observation_budget = max(
                observation_budget,
                int(math.ceil(
                    observation_budget
                    * float(self.config["forced_observation_window_multiplier"])
                )),
            )
        self.last_observation = {
            "eligible": True,
            "reason": statistics["trigger_category"],
            "curriculum_id": curriculum_id,
            "forced": forced,
            **statistics,
        }
        self.pending_action = AdvanceCurriculumStateAction(
            action_type="advance_curriculum_state",
            controller=self.name,
            curriculum_id=curriculum_id,
            from_level_index=current,
            from_level_id=str(snapshot["current_level_id"]),
            to_level_index=target,
            to_level_id=str(level_ids[target]),
            forced=forced,
            trigger_statistics=statistics,
            runtime_state_version=int(snapshot["runtime_state_version"]),
            runtime_state_digest=str(snapshot["state_digest"]),
            associated_component_ids=tuple(
                diagnostic.get("associated_component_ids") or ()
            ),
            associated_sampler_ids=tuple(
                diagnostic.get("associated_sampler_ids") or ()
            ),
            requested_compute_budget=observation_budget,
            requested_iteration_budget=minimum_remaining,
            before={
                "curriculum_id": curriculum_id,
                "level_index": current,
                "level_id": str(snapshot["current_level_id"]),
                "runtime_state_version": int(snapshot["runtime_state_version"]),
                "runtime_state_digest": str(snapshot["state_digest"]),
            },
            after={
                "curriculum_id": curriculum_id,
                "from_level_index": current,
                "from_level_id": str(snapshot["current_level_id"]),
                "to_level_index": target,
                "to_level_id": str(level_ids[target]),
                "forced": bool(forced),
                "runtime_state_version": int(snapshot["runtime_state_version"]),
                "runtime_state_digest": str(snapshot["state_digest"]),
            },
            reason=dict(self.last_observation),
            limits={
                "single_step_only": True,
                "monotonic": True,
                "minimum_remaining_budget_after_advance": minimum_remaining,
                "observation_window_iterations": observation_budget,
            },
        )
        return dict(self.last_observation)

    def _validation_stable(
        self,
        *,
        curriculum_id: str,
        aggregate: float,
        component_scores: dict[str, float],
        associated_component_ids: tuple[str, ...],
        critical_component_ids: tuple[str, ...],
    ) -> tuple[bool, dict[str, Any]]:
        previous = self.last_validation_scores.get(curriculum_id)
        previous_components = self.last_component_scores.get(curriculum_id, {})
        finite = math.isfinite(aggregate) and all(
            math.isfinite(value) for value in component_scores.values()
        )
        relative_improvement = None
        relative_change = None
        if previous is not None and math.isfinite(previous):
            denominator = max(abs(previous), 1e-12)
            relative_improvement = (previous - aggregate) / denominator
            relative_change = abs(aggregate - previous) / denominator
        critical_degraded = any(
            name in component_scores
            and name in previous_components
            and component_scores[name]
            > previous_components[name]
            * float(self.config["critical_component_degradation_ratio"])
            for name in critical_component_ids
        )
        associated_degraded = any(
            name in component_scores
            and name in previous_components
            and component_scores[name]
            > previous_components[name]
            * float(self.config["associated_component_degradation_ratio"])
            for name in associated_component_ids
        )
        threshold = self.config.get("maximum_validation_score")
        absolute_ok = threshold is not None and aggregate <= float(threshold)
        minimum = float(self.config["minimum_relative_improvement"])
        plateau = relative_change is not None and relative_change <= minimum
        improving = relative_improvement is not None and relative_improvement >= minimum
        mode = str(self.config["normal_advance_condition"])
        if mode == "absolute_threshold":
            condition = absolute_ok
        elif mode == "relative_plateau":
            condition = plateau
        elif mode == "relative_improvement":
            condition = improving
        elif mode in {"hybrid", "physics_validation_stable"}:
            condition = (absolute_ok if threshold is not None else True) and (
                plateau or improving
            )
        else:
            condition = False
        stable = finite and condition and not critical_degraded and not associated_degraded
        self.last_validation_scores[curriculum_id] = aggregate
        self.last_component_scores[curriculum_id] = dict(component_scores)
        return stable, {
            "relative_improvement": relative_improvement,
            "relative_change": relative_change,
            "critical_component_degraded": critical_degraded,
            "associated_component_degraded": associated_degraded,
        }

    def propose_action(self) -> AdaptiveAction | None:
        return self.pending_action

    def mark_action_applied(self, iteration: int, *, forced: bool = False) -> None:
        action = self.pending_action
        if action is not None:
            self.last_action_iterations[action.curriculum_id] = int(iteration)
            self.patience_counts[action.curriculum_id] = 0
            self.last_validation_scores.pop(action.curriculum_id, None)
            self.last_component_scores.pop(action.curriculum_id, None)
        self.advance_count += 1
        if forced:
            self.forced_advance_count += 1
        self.pending_action = None

    def mark_action_rolled_back(self, curriculum_id: str, iteration: int) -> None:
        self.rollback_count += 1
        self.rollback_cooldown_until[str(curriculum_id)] = (
            int(iteration) + int(self.config["cooldown_iterations"])
        )
        self.patience_counts[str(curriculum_id)] = 0
        self.pending_action = None

    def state_dict(self) -> dict[str, Any]:
        return {
            "patience_counts": dict(self.patience_counts),
            "last_validation_scores": dict(self.last_validation_scores),
            "last_component_scores": {
                name: dict(value) for name, value in self.last_component_scores.items()
            },
            "last_action_iterations": dict(self.last_action_iterations),
            "rollback_cooldown_until": dict(self.rollback_cooldown_until),
            "advance_count": int(self.advance_count),
            "forced_advance_count": int(self.forced_advance_count),
            "rollback_count": int(self.rollback_count),
            "last_observation": dict(self.last_observation),
        }

    def load_state_dict(self, state: dict[str, Any] | None) -> None:
        state = dict(state or {})
        self.patience_counts = {
            str(name): int(value)
            for name, value in (state.get("patience_counts") or {}).items()
        }
        self.last_validation_scores = {
            str(name): float(value)
            for name, value in (state.get("last_validation_scores") or {}).items()
        }
        self.last_component_scores = {
            str(name): {str(key): float(value) for key, value in values.items()}
            for name, values in (state.get("last_component_scores") or {}).items()
        }
        self.last_action_iterations = {
            str(name): int(value)
            for name, value in (state.get("last_action_iterations") or {}).items()
        }
        self.rollback_cooldown_until = {
            str(name): int(value)
            for name, value in (state.get("rollback_cooldown_until") or {}).items()
        }
        self.advance_count = int(state.get("advance_count") or 0)
        self.forced_advance_count = int(state.get("forced_advance_count") or 0)
        self.rollback_count = int(state.get("rollback_count") or 0)
        self.last_observation = dict(state.get("last_observation") or {})
        self.pending_action = None
