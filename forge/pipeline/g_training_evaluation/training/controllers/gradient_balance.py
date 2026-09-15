"""Bounded gradient-norm loss balancing for stochastic-gradient phases."""

from __future__ import annotations

import math
from statistics import median
from typing import Any

from .state import AdaptiveAction, TrainerObservableState


DEFAULTS: dict[str, Any] = {
    "ema_beta": 0.95,
    "diagnostic_interval": 50,
    "patience": 3,
    "cooldown_iterations": 200,
    "imbalance_threshold": 100.0,
    "conflict_threshold": -0.1,
    "minimum_weight": 0.01,
    "maximum_weight": 20.0,
    "minimum_update_ratio": 0.5,
    "maximum_update_ratio": 2.0,
    "alpha": 0.5,
    "epsilon": 1.0e-12,
}


class GradientBalanceController:
    name = "gradient_balance"
    allowed_optimizers = frozenset({"adam", "adamw", "radam", "nadam"})

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = {**DEFAULTS, **dict(config or {})}
        self.ema_gradient_norms: dict[str, float] = {}
        self.imbalance_patience_count = 0
        self.last_action_iteration: int | None = None
        self.last_observation: dict[str, Any] = {}
        self.pending_action: AdaptiveAction | None = None

    def observe(self, state: TrainerObservableState) -> dict[str, Any]:
        optimizer = str(state.optimizer_name).lower()
        first_order = (
            bool(state.stage_descriptor.first_order)
            if state.stage_descriptor is not None
            else optimizer in self.allowed_optimizers
        )
        if not first_order or (
            state.stage_descriptor is not None
            and not bool(state.stage_descriptor.dynamic_objective_allowed)
        ):
            self.pending_action = None
            self.last_observation = {
                "eligible": False,
                "reason": "optimizer_not_allowed",
                "optimizer": optimizer,
            }
            return dict(self.last_observation)

        interval = int(self.config["diagnostic_interval"])
        if state.iteration <= 0 or state.iteration % interval != 0:
            self.pending_action = None
            self.last_observation = {
                "eligible": False,
                "reason": "diagnostic_interval_not_reached",
            }
            return dict(self.last_observation)

        eligible = set(state.eligible_component_ids or state.component_gradient_norms)
        finite_norms = {
            str(name): float(value)
            for name, value in sorted(state.component_gradient_norms.items())
            if name in eligible
            if math.isfinite(float(value)) and float(value) >= 0.0
        }
        if len(finite_norms) < 2:
            self.pending_action = None
            self.imbalance_patience_count = 0
            self.last_observation = {
                "eligible": False,
                "reason": "fewer_than_two_finite_component_gradients",
            }
            return dict(self.last_observation)

        beta = float(self.config["ema_beta"])
        for name, value in finite_norms.items():
            if name not in self.ema_gradient_norms:
                self.ema_gradient_norms[name] = value
            else:
                self.ema_gradient_norms[name] = (
                    beta * self.ema_gradient_norms[name] + (1.0 - beta) * value
                )

        epsilon = float(self.config["epsilon"])
        active_ema = {
            name: max(float(self.ema_gradient_norms[name]), 0.0)
            for name in finite_norms
        }
        largest = max(active_ema.values())
        smallest = min(active_ema.values())
        ratio = largest / max(smallest, epsilon)
        minimum_cosine = state.minimum_gradient_cosine
        conflict = (
            minimum_cosine is not None
            and math.isfinite(float(minimum_cosine))
            and float(minimum_cosine) < float(self.config["conflict_threshold"])
        )
        in_cooldown = (
            self.last_action_iteration is not None
            and state.iteration - self.last_action_iteration
            < int(self.config["cooldown_iterations"])
        )
        imbalanced = ratio > float(self.config["imbalance_threshold"])
        self.imbalance_patience_count = (
            self.imbalance_patience_count + 1 if imbalanced else 0
        )
        self.last_observation = {
            "eligible": True,
            "gradient_imbalance_ratio": ratio,
            "minimum_gradient_cosine": minimum_cosine,
            "gradient_conflict": conflict,
            "patience_count": self.imbalance_patience_count,
            "in_cooldown": in_cooldown,
            "ema_gradient_norms": dict(active_ema),
        }
        self.pending_action = None
        if (
            not imbalanced
            or conflict
            or in_cooldown
            or self.imbalance_patience_count < int(self.config["patience"])
        ):
            return dict(self.last_observation)

        old_weights = {
            str(name): float(value)
            for name, value in sorted(state.loss_weights.items())
            if name in active_ema and math.isfinite(float(value)) and float(value) > 0.0
        }
        if len(old_weights) < 2:
            self.last_observation["eligible"] = False
            self.last_observation["reason"] = "loss_weights_unavailable"
            return dict(self.last_observation)

        target_norm = float(median(active_ema[name] for name in old_weights))
        alpha = float(self.config["alpha"])
        min_ratio = float(self.config["minimum_update_ratio"])
        max_ratio = float(self.config["maximum_update_ratio"])
        min_weight = float(self.config["minimum_weight"])
        max_weight = float(self.config["maximum_weight"])
        new_weights: dict[str, float] = {}
        ratio_limited = False
        bound_limited = False
        for name, old in old_weights.items():
            raw_ratio = (target_norm / max(active_ema[name], epsilon)) ** alpha
            limited_ratio = min(max(raw_ratio, min_ratio), max_ratio)
            ratio_limited = ratio_limited or limited_ratio != raw_ratio
            raw_weight = old * limited_ratio
            new_weight = min(max(raw_weight, min_weight), max_weight)
            bound_limited = bound_limited or new_weight != raw_weight
            if not math.isfinite(new_weight) or new_weight <= 0.0:
                new_weight = old
                bound_limited = True
            new_weights[name] = new_weight

        self.pending_action = AdaptiveAction(
            action_type="update_loss_weights",
            controller=self.name,
            before=old_weights,
            after=new_weights,
            reason=dict(self.last_observation),
            limits={
                "ratio_limited": ratio_limited,
                "global_bound_limited": bound_limited,
                "minimum_update_ratio": min_ratio,
                "maximum_update_ratio": max_ratio,
                "minimum_weight": min_weight,
                "maximum_weight": max_weight,
            },
            target_component_ids=tuple(sorted(new_weights)),
        )
        return dict(self.last_observation)

    def should_act(self) -> bool:
        return self.pending_action is not None

    def propose_action(self) -> AdaptiveAction | None:
        return self.pending_action

    def mark_action_applied(self, iteration: int) -> None:
        self.last_action_iteration = int(iteration)
        self.imbalance_patience_count = 0
        self.pending_action = None

    def on_curriculum_advanced(self, iteration: int, ema_decay: float) -> None:
        decay = min(1.0, max(0.0, float(ema_decay)))
        self.ema_gradient_norms = {
            name: float(value) * decay
            for name, value in self.ema_gradient_norms.items()
        }
        self.imbalance_patience_count = 0
        self.last_action_iteration = int(iteration)
        self.pending_action = None

    def state_dict(self) -> dict[str, Any]:
        return {
            "ema_gradient_norms": dict(self.ema_gradient_norms),
            "imbalance_patience_count": int(self.imbalance_patience_count),
            "last_action_iteration": self.last_action_iteration,
            "last_observation": dict(self.last_observation),
        }

    def load_state_dict(self, state: dict[str, Any] | None) -> None:
        state = dict(state or {})
        self.ema_gradient_norms = {
            str(name): float(value)
            for name, value in (state.get("ema_gradient_norms") or {}).items()
        }
        self.imbalance_patience_count = int(
            state.get("imbalance_patience_count") or 0
        )
        last = state.get("last_action_iteration")
        self.last_action_iteration = int(last) if last is not None else None
        self.last_observation = dict(state.get("last_observation") or {})
        self.pending_action = None
