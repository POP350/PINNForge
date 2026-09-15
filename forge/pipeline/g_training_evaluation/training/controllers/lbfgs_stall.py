"""Reproducible compound L-BFGS stall detection."""

from __future__ import annotations

import math
from statistics import median
from typing import Any

from .state import AdaptiveAction, TrainerObservableState


DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "patience": 20,
    "minimum_relative_improvement": 1.0e-4,
    "minimum_step_size": 1.0e-8,
    "minimum_parameter_update_norm": 1.0e-10,
    "maximum_repeated_line_search_failures": 10,
    "minimum_completed_iterations_before_detection": 50,
    "action": "reallocate_to_adam",
    "adam_learning_rate_scale": 0.1,
    "maximum_recovery_iterations": 1000,
    "minimum_recovery_iterations": 100,
    "reuse_remaining_budget": True,
    "return_to_lbfgs": False,
}


class LBFGSStallController:
    name = "lbfgs_stall"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = {**DEFAULTS, **dict(config or {})}
        self.observations: list[dict[str, Any]] = []
        self.triggered = False
        self.trigger_iteration: int | None = None
        self.pending_action: AdaptiveAction | None = None
        self.previous_loss: float | None = None

    def observe(self, state: TrainerObservableState) -> dict[str, Any]:
        if self.triggered:
            self.pending_action = None
            return {"eligible": False, "reason": "already_triggered"}
        second_order = (
            bool(state.stage_descriptor.second_order)
            if state.stage_descriptor is not None
            else str(state.optimizer_name).lower() == "lbfgs"
        )
        if not second_order:
            self.pending_action = None
            return {"eligible": False, "reason": "not_lbfgs"}

        loss = state.current_loss
        relative_improvement = None
        if (
            loss is not None
            and self.previous_loss is not None
            and math.isfinite(float(loss))
            and math.isfinite(float(self.previous_loss))
        ):
            relative_improvement = (
                float(self.previous_loss) - float(loss)
            ) / max(abs(float(self.previous_loss)), 1.0e-30)
        if loss is not None and math.isfinite(float(loss)):
            self.previous_loss = float(loss)

        item = {
            "relative_improvement": relative_improvement,
            "step_size": state.step_size,
            "parameter_update_norm": state.parameter_update_norm,
            "line_search_failed": bool(state.line_search_failed),
            "closure_evaluations_this_step": int(
                state.closure_evaluations_this_step
            ),
        }
        self.observations.append(item)
        patience = int(self.config["patience"])
        if len(self.observations) > patience:
            self.observations = self.observations[-patience:]

        diagnostic = {
            "eligible": False,
            "completed_stage_iterations": int(state.completed_stage_iterations),
            "window_size": len(self.observations),
        }
        if state.completed_stage_iterations < int(
            self.config["minimum_completed_iterations_before_detection"]
        ):
            diagnostic["reason"] = "minimum_iterations_not_reached"
            return diagnostic
        if len(self.observations) < patience:
            diagnostic["reason"] = "patience_window_not_full"
            return diagnostic

        improvements = [
            float(value["relative_improvement"])
            for value in self.observations
            if value["relative_improvement"] is not None
            and math.isfinite(float(value["relative_improvement"]))
        ]
        step_sizes = [
            abs(float(value["step_size"]))
            for value in self.observations
            if value["step_size"] is not None
            and math.isfinite(float(value["step_size"]))
        ]
        update_norms = [
            abs(float(value["parameter_update_norm"]))
            for value in self.observations
            if value["parameter_update_norm"] is not None
            and math.isfinite(float(value["parameter_update_norm"]))
        ]
        failures = sum(bool(value["line_search_failed"]) for value in self.observations)
        closure_growth_without_improvement = (
            sum(int(value["closure_evaluations_this_step"]) for value in self.observations)
            > patience
            and bool(improvements)
            and median(improvements) < float(
                self.config["minimum_relative_improvement"]
            )
        )
        low_improvement = (
            bool(improvements)
            and median(improvements)
            < float(self.config["minimum_relative_improvement"])
        )
        supporting = {
            "low_step_size": bool(step_sizes)
            and median(step_sizes) < float(self.config["minimum_step_size"]),
            "low_parameter_update_norm": bool(update_norms)
            and median(update_norms)
            < float(self.config["minimum_parameter_update_norm"]),
            "repeated_line_search_failures": failures
            >= int(self.config["maximum_repeated_line_search_failures"]),
            "closure_growth_without_improvement": closure_growth_without_improvement,
        }
        stalled = low_improvement and any(supporting.values())
        diagnostic.update(
            {
                "eligible": True,
                "median_relative_improvement": median(improvements)
                if improvements
                else None,
                "median_step_size": median(step_sizes) if step_sizes else None,
                "median_parameter_update_norm": median(update_norms)
                if update_norms
                else None,
                "repeated_line_search_failures": failures,
                "supporting_conditions": supporting,
                "stalled": stalled,
            }
        )
        self.pending_action = None
        if stalled:
            configured_action = str(self.config["action"])
            remaining = max(0, int(state.remaining_iteration_budget))
            maximum = int(self.config["maximum_recovery_iterations"])
            minimum = int(self.config["minimum_recovery_iterations"])
            recovery = min(remaining, maximum) if configured_action == "reallocate_to_adam" else 0
            if recovery < minimum:
                action_name = "terminate_candidate"
                recovery = 0
            else:
                action_name = (
                    (
                        "start_recovery_stage"
                        if state.stage_descriptor is not None
                        else "reallocate_to_adam"
                    )
                    if configured_action == "reallocate_to_adam"
                    else "terminate_candidate"
                )
            self.pending_action = AdaptiveAction(
                action_type=action_name,
                controller=self.name,
                before={"remaining_lbfgs_iterations": remaining},
                after={"recovery_adam_iterations": recovery},
                reason=diagnostic,
                limits={
                    "maximum_recovery_iterations": maximum,
                    "minimum_recovery_iterations": minimum,
                    "reuse_remaining_budget": bool(
                        self.config["reuse_remaining_budget"]
                    ),
                    "return_to_lbfgs": False,
                },
                requested_iteration_budget=recovery,
            )
        return diagnostic

    def should_act(self) -> bool:
        return self.pending_action is not None

    def propose_action(self) -> AdaptiveAction | None:
        return self.pending_action

    def mark_action_applied(self, iteration: int) -> None:
        self.triggered = True
        self.trigger_iteration = int(iteration)
        self.pending_action = None

    def state_dict(self) -> dict[str, Any]:
        return {
            "observations": list(self.observations),
            "triggered": bool(self.triggered),
            "trigger_iteration": self.trigger_iteration,
            "previous_loss": self.previous_loss,
        }

    def load_state_dict(self, state: dict[str, Any] | None) -> None:
        state = dict(state or {})
        self.observations = list(state.get("observations") or [])
        self.triggered = bool(state.get("triggered", False))
        trigger = state.get("trigger_iteration")
        self.trigger_iteration = int(trigger) if trigger is not None else None
        previous = state.get("previous_loss")
        self.previous_loss = float(previous) if previous is not None else None
        self.pending_action = None
