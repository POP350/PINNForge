"""Deterministic arbitration and observation windows for controllers."""

from __future__ import annotations

from typing import Any, Iterable

from .state import AdaptiveAction, TrainerObservableState


DEFAULT_PRIORITY = (
    "rollback",
    "terminate_candidate",
    "stage_recovery",
    "loss_weight",
    "learning_rate",
    "sampling",
    "curriculum",
)
SAFETY_ACTIONS = frozenset({"rollback", "terminate_candidate"})
ACTION_PRIORITY_CATEGORY = {
    "rollback": "rollback",
    "terminate_candidate": "terminate_candidate",
    "start_recovery_stage": "stage_recovery",
    "shorten_optimization_stage": "stage_recovery",
    "update_loss_weights": "loss_weight",
    "update_learning_rate": "learning_rate",
    "replace_sampling_subset": "sampling",
    "update_sampling_allocation": "sampling",
    "advance_curriculum_state": "curriculum",
}


class ControllerCoordinator:
    def __init__(
        self,
        controllers: Iterable[Any] = (),
        config: dict[str, Any] | None = None,
    ) -> None:
        config = dict(config or {})
        self.controllers = tuple(controllers)
        self.maximum_non_safety_actions_per_cycle = int(
            config.get("maximum_non_safety_actions_per_cycle", 1)
        )
        self.observation_window_iterations = int(
            config.get("observation_window_iterations", 100)
        )
        self.global_cooldown_iterations = int(
            config.get("global_cooldown_iterations", 100)
        )
        if self.maximum_non_safety_actions_per_cycle != 1:
            raise ValueError("Exactly one non-safety action per cycle is required")
        if self.observation_window_iterations < 0 or self.global_cooldown_iterations < 0:
            raise ValueError("Coordinator windows must be non-negative")
        configured = tuple(config.get("priority") or DEFAULT_PRIORITY)
        if set(configured) != set(DEFAULT_PRIORITY) or len(configured) != len(DEFAULT_PRIORITY):
            raise ValueError("Controller priority must contain every registered priority exactly once")
        self.priority = configured
        self.last_non_safety_action_iteration: int | None = None
        self.observation_window_until: int | None = None
        self.failed_signatures: dict[str, int] = {}
        self.last_diagnostics: dict[str, dict[str, Any]] = {}

    def observe(self, state: TrainerObservableState) -> dict[str, dict[str, Any]]:
        self.last_diagnostics = {
            str(getattr(controller, "name", type(controller).__name__)): dict(
                controller.observe(state)
            )
            for controller in self.controllers
        }
        return dict(self.last_diagnostics)

    def observe_controller(
        self, controller: Any, state: TrainerObservableState
    ) -> dict[str, Any]:
        name = str(getattr(controller, "name", type(controller).__name__))
        diagnostic = dict(controller.observe(state))
        self.last_diagnostics[name] = diagnostic
        return diagnostic

    def observation_window_active(self, iteration: int) -> bool:
        default_until = (
            self.last_non_safety_action_iteration + self.observation_window_iterations
            if self.last_non_safety_action_iteration is not None
            else None
        )
        until = max(
            value for value in (default_until, self.observation_window_until)
            if value is not None
        ) if any(
            value is not None for value in (default_until, self.observation_window_until)
        ) else None
        return bool(until is not None and int(iteration) < int(until))

    def extend_observation_window(self, until_iteration: int) -> None:
        self.observation_window_until = max(
            int(until_iteration), int(self.observation_window_until or 0)
        )

    def collect_proposals(self) -> tuple[AdaptiveAction, ...]:
        proposals = []
        for controller in self.controllers:
            action = controller.propose_action()
            if action is not None:
                proposals.append(action)
        return tuple(proposals)

    def select_action(
        self,
        proposals: Iterable[AdaptiveAction],
        *,
        iteration: int,
    ) -> tuple[AdaptiveAction | None, dict[str, str]]:
        rejected: dict[str, str] = {}
        eligible: list[AdaptiveAction] = []
        observing = self.observation_window_active(iteration)
        for action in proposals:
            signature = self.action_signature(action)
            failed_at = self.failed_signatures.get(signature)
            if (
                failed_at is not None
                and int(iteration) - failed_at < self.global_cooldown_iterations
            ):
                rejected[action.action_id] = "failed_action_global_cooldown"
                continue
            if observing and action.action_type not in SAFETY_ACTIONS:
                rejected[action.action_id] = "global_observation_window"
                continue
            eligible.append(action)
        if not eligible:
            return None, rejected
        order = {name: index for index, name in enumerate(self.priority)}
        eligible.sort(
            key=lambda action: (
                order[ACTION_PRIORITY_CATEGORY[action.action_type]],
                action.action_id,
            )
        )
        selected = eligible[0]
        for action in eligible[1:]:
            rejected[action.action_id] = "lower_priority_same_cycle"
        return selected, rejected

    def record_outcome(
        self,
        action: AdaptiveAction,
        *,
        iteration: int,
        outcome: str,
    ) -> None:
        if outcome == "applied" and action.action_type not in SAFETY_ACTIONS:
            self.last_non_safety_action_iteration = int(iteration)
            self.extend_observation_window(
                int(iteration) + self.observation_window_iterations
            )
        if outcome in {"failed", "rolled_back"}:
            self.failed_signatures[self.action_signature(action)] = int(iteration)

    @staticmethod
    def action_signature(action: AdaptiveAction) -> str:
        curriculum_transition = (
            str(getattr(action, "curriculum_id", "")),
            int(getattr(action, "from_level_index", -1)),
            int(getattr(action, "to_level_index", -1)),
            bool(getattr(action, "forced", False)),
            str((getattr(action, "trigger_statistics", {}) or {}).get(
                "trigger_category", ""
            )),
        ) if action.action_type == "advance_curriculum_state" else None
        return repr(
            (
                action.controller_id,
                action.action_type,
                action.target_component_ids,
                curriculum_transition,
                tuple(sorted((str(name), repr(value)) for name, value in action.after.items())),
                tuple(sorted((str(name), repr(value)) for name, value in action.reason.items())),
            )
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "last_non_safety_action_iteration": self.last_non_safety_action_iteration,
            "observation_window_until": self.observation_window_until,
            "failed_signatures": dict(self.failed_signatures),
            "last_diagnostics": dict(self.last_diagnostics),
        }

    def load_state_dict(self, state: dict[str, Any] | None) -> None:
        state = dict(state or {})
        last = state.get("last_non_safety_action_iteration")
        self.last_non_safety_action_iteration = int(last) if last is not None else None
        until = state.get("observation_window_until")
        self.observation_window_until = int(until) if until is not None else None
        self.failed_signatures = {
            str(name): int(value)
            for name, value in (state.get("failed_signatures") or {}).items()
        }
        self.last_diagnostics = dict(state.get("last_diagnostics") or {})
