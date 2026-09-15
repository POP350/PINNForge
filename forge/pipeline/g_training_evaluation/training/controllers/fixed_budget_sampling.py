"""Bounded controller for PDE-independent fixed-budget point replacement."""

from __future__ import annotations

from typing import Any

from .state import ReplaceSamplingSubsetAction, TrainerObservableState


DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "strategy": "fixed_budget_subset_replacement",
    "diagnostic_interval": 200,
    "minimum_iterations_before_control": 500,
    "patience": 3,
    "cooldown_iterations": 500,
    "maximum_updates": 10,
    "p95_to_median_threshold": 10.0,
    "replacement_fraction": 0.20,
    "hard_point_retention_fraction": 0.20,
    "random_exploration_fraction": 0.10,
    "candidate_multiplier": 4,
    "maximum_point_age": 20,
    "maximum_consecutive_retention": 5,
    "minimum_point_age": 2,
    "observation_window_iterations": 300,
    "minimum_validation_stability_checks": 2,
    "validation_degradation_ratio": 1.5,
    "associated_component_degradation_ratio": 2.0,
    "maximum_candidate_points_evaluated": 100000,
    "epsilon": 1e-12,
}


class FixedBudgetSamplingController:
    name = "fixed_budget_sampling"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = {**DEFAULTS, **dict(config or {})}
        self.patience_counts: dict[str, int] = {}
        self.last_action_iteration: int | None = None
        self.update_count = 0
        self.last_observation: dict[str, Any] = {}
        self.pending_action: ReplaceSamplingSubsetAction | None = None
        self.rollback_cooldown_until = 0

    def diagnostic_due(self, state: TrainerObservableState) -> tuple[bool, str]:
        stage = state.stage_descriptor
        if stage is None or not bool(stage.first_order) or not bool(stage.dynamic_objective_allowed):
            return False, "stage_frozen_or_not_first_order"
        if state.iteration < int(self.config["minimum_iterations_before_control"]):
            return False, "minimum_iterations_not_reached"
        if state.iteration % int(self.config["diagnostic_interval"]) != 0:
            return False, "diagnostic_interval_not_reached"
        if state.pending_rollback:
            return False, "pending_rollback"
        if state.coordinator_observation_active:
            return False, "global_observation_window"
        if state.iteration < self.rollback_cooldown_until:
            return False, "rollback_cooldown"
        if (
            self.last_action_iteration is not None
            and state.iteration - self.last_action_iteration
            < int(self.config["cooldown_iterations"])
        ):
            return False, "controller_cooldown"
        if self.update_count >= int(self.config["maximum_updates"]):
            return False, "maximum_updates_reached"
        remaining = int(state.sampling_budget_state.get("remaining_candidate_points", 0))
        if remaining <= 0:
            return False, "candidate_budget_exhausted"
        return True, "diagnostic_due"

    def observe(self, state: TrainerObservableState) -> dict[str, Any]:
        self.pending_action = None
        due, reason = self.diagnostic_due(state)
        if not due:
            self.last_observation = {"eligible": False, "reason": reason}
            return dict(self.last_observation)
        candidates: list[tuple[float, str, dict[str, Any]]] = []
        for sampler_id, diagnostic in sorted(state.sampling_diagnostics.items()):
            stats = dict(diagnostic.get("statistics") or {})
            ratio = float(stats.get("p95_to_median_ratio", 0.0))
            if ratio > float(self.config["p95_to_median_threshold"]):
                self.patience_counts[sampler_id] = self.patience_counts.get(sampler_id, 0) + 1
            else:
                self.patience_counts[sampler_id] = 0
            if self.patience_counts[sampler_id] >= int(self.config["patience"]):
                candidates.append((ratio, str(sampler_id), dict(diagnostic)))
        if not candidates:
            self.last_observation = {
                "eligible": True,
                "reason": "sampling_imbalance_patience_not_reached",
                "patience_counts": dict(self.patience_counts),
            }
            return dict(self.last_observation)
        _, sampler_id, diagnostic = sorted(candidates, key=lambda item: (-item[0], item[1]))[0]
        point_count = int(diagnostic["point_count"])
        replacement_count = max(1, int(point_count * float(self.config["replacement_fraction"])))
        candidate_count = max(
            replacement_count,
            replacement_count * int(self.config["candidate_multiplier"]),
        )
        remaining = int(state.sampling_budget_state.get("remaining_candidate_points", 0))
        if candidate_count > remaining:
            self.last_observation = {
                "eligible": False,
                "reason": "candidate_budget_insufficient",
                "candidate_count": candidate_count,
                "remaining_candidate_points": remaining,
            }
            return dict(self.last_observation)
        stats = dict(diagnostic.get("statistics") or {})
        component_ids = tuple(sorted(diagnostic.get("scoring_component_ids") or ()))
        version = int(diagnostic.get("sampler_state_version", 0))
        self.last_observation = {
            "eligible": True,
            "reason": "persistent_sampling_distribution_imbalance",
            "sampler_id": sampler_id,
            "statistics": stats,
            "patience_count": self.patience_counts[sampler_id],
        }
        self.pending_action = ReplaceSamplingSubsetAction(
            action_type="replace_sampling_subset",
            controller=self.name,
            sampler_id=sampler_id,
            replacement_fraction=float(self.config["replacement_fraction"]),
            hard_point_retention_fraction=float(self.config["hard_point_retention_fraction"]),
            exploration_fraction=float(self.config["random_exploration_fraction"]),
            scoring_component_ids=component_ids,
            score_statistics=stats,
            candidate_count=candidate_count,
            requested_compute_budget=candidate_count,
            sampler_state_version=version,
            before={"point_count": point_count, "sampler_state_version": version},
            after={
                "sampler_id": sampler_id,
                "replacement_fraction": float(self.config["replacement_fraction"]),
                "hard_point_retention_fraction": float(self.config["hard_point_retention_fraction"]),
                "exploration_fraction": float(self.config["random_exploration_fraction"]),
                "candidate_count": candidate_count,
                "sampler_state_version": version,
            },
            reason=dict(self.last_observation),
            limits={
                "maximum_point_age": int(self.config["maximum_point_age"]),
                "maximum_consecutive_retention": int(self.config["maximum_consecutive_retention"]),
                "minimum_point_age": int(self.config["minimum_point_age"]),
                "maximum_candidate_points_evaluated": int(self.config["maximum_candidate_points_evaluated"]),
            },
        )
        return dict(self.last_observation)

    def propose_action(self) -> ReplaceSamplingSubsetAction | None:
        return self.pending_action

    def mark_action_applied(self, iteration: int) -> None:
        self.last_action_iteration = int(iteration)
        self.update_count += 1
        self.pending_action = None

    def mark_action_rolled_back(self, iteration: int) -> None:
        self.rollback_cooldown_until = int(iteration) + int(self.config["cooldown_iterations"])
        self.pending_action = None

    def on_curriculum_advanced(
        self, iteration: int, associated_sampler_ids: tuple[str, ...]
    ) -> None:
        for sampler_id in associated_sampler_ids:
            self.patience_counts.pop(str(sampler_id), None)
        self.last_observation = {}
        self.rollback_cooldown_until = max(
            int(self.rollback_cooldown_until),
            int(iteration) + int(self.config["cooldown_iterations"]),
        )
        self.pending_action = None

    def state_dict(self) -> dict[str, Any]:
        return {
            "patience_counts": dict(self.patience_counts),
            "last_action_iteration": self.last_action_iteration,
            "update_count": int(self.update_count),
            "last_observation": dict(self.last_observation),
            "rollback_cooldown_until": int(self.rollback_cooldown_until),
        }

    def load_state_dict(self, state: dict[str, Any] | None) -> None:
        state = dict(state or {})
        self.patience_counts = {str(k): int(v) for k, v in (state.get("patience_counts") or {}).items()}
        last = state.get("last_action_iteration")
        self.last_action_iteration = int(last) if last is not None else None
        self.update_count = int(state.get("update_count") or 0)
        self.last_observation = dict(state.get("last_observation") or {})
        self.rollback_cooldown_until = int(state.get("rollback_cooldown_until") or 0)
        self.pending_action = None
