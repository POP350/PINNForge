"""Central validation boundary for every trainer-side adaptive action."""

from __future__ import annotations

import math
from typing import Any

from .state import AdaptiveAction, TrainerObservableState
from ..registry import TrainingComponentRegistry


REGISTERED_ACTION_TYPES = frozenset(
    {
        "update_loss_weights",
        "update_learning_rate",
        "shorten_optimization_stage",
        "start_recovery_stage",
        "terminate_candidate",
        "rollback",
        "replace_sampling_subset",
        "update_sampling_allocation",
        "advance_curriculum_state",
    }
)
STRUCTURAL_FIELDS = frozenset(
    {
        "network",
        "architecture",
        "hidden_layers",
        "activation",
        "initialization",
        "input_transform",
        "output_transform",
        "constraint_enforcement",
        "residual_definition",
        "loss_terms",
        "loss_function",
        "sampling_algorithm",
    }
)
REFERENCE_TOKENS = (
    "reference",
    "evaluator",
    "l1re",
    "l2re",
    "relative_error",
    "phase_error",
    "amplitude_ratio",
)
ALLOWED_PAYLOAD_FIELDS = {
    "update_learning_rate": frozenset({"learning_rate", "learning_rates"}),
    "shorten_optimization_stage": frozenset({"remaining_iterations", "new_iteration_limit"}),
    "start_recovery_stage": frozenset({"recovery_adam_iterations", "learning_rate_scale"}),
    "terminate_candidate": frozenset({"recovery_adam_iterations", "termination_reason"}),
    "rollback": frozenset(),
    "replace_sampling_subset": frozenset({
        "sampler_id", "replacement_fraction", "hard_point_retention_fraction",
        "exploration_fraction", "candidate_count", "sampler_state_version",
    }),
    "update_sampling_allocation": frozenset({"point_budget", "sampler_id"}),
    "advance_curriculum_state": frozenset({
        "curriculum_id", "from_level_index", "from_level_id",
        "to_level_index", "to_level_id", "forced",
        "runtime_state_version", "runtime_state_digest",
    }),
}


class ActionValidationError(ValueError):
    pass


class ActionBoundaryValidator:
    def validate(
        self,
        *,
        action: AdaptiveAction,
        registry: TrainingComponentRegistry,
        stage: Any,
        observable_state: TrainerObservableState,
    ) -> AdaptiveAction:
        if action.action_type not in REGISTERED_ACTION_TYPES:
            raise ActionValidationError(f"Action type is not registered: {action.action_type}")
        if action.action_type not in registry.capabilities.supported_action_types:
            raise ActionValidationError(
                f"Action type is not supported by the current training capabilities: {action.action_type}"
            )
        if self._contains_reference_metric(action):
            raise ActionValidationError("Adaptive actions may not reference evaluator metrics")
        forbidden = self._find_structural_fields(action.after)
        if forbidden:
            raise ActionValidationError(
                f"Adaptive action attempted to modify structural fields: {sorted(forbidden)}"
            )
        all_component_ids = set(registry.component_ids) | {
            item.sampler_id for item in registry.sampling_components
        } | set(registry.curriculum_ids)
        unknown_targets = set(action.target_component_ids) - all_component_ids
        if unknown_targets:
            raise ActionValidationError(
                f"Action targets unknown registered components: {sorted(unknown_targets)}"
            )
        allowed_payload = ALLOWED_PAYLOAD_FIELDS.get(action.action_type)
        if allowed_payload is not None:
            unexpected_payload = set(action.after) - allowed_payload
            if unexpected_payload:
                raise ActionValidationError(
                    f"Action contains fields outside its finite protocol: {sorted(unexpected_payload)}"
                )
        if action.requested_iteration_budget < 0:
            raise ActionValidationError("Requested iteration budget must be non-negative")
        remaining = int(observable_state.remaining_iteration_budget)
        if action.requested_iteration_budget > remaining:
            raise ActionValidationError("Action exceeds the remaining iteration budget")
        if action.action_type in {
            "update_loss_weights",
            "update_learning_rate",
            "replace_sampling_subset",
            "update_sampling_allocation",
            "advance_curriculum_state",
        } and not bool(stage.dynamic_objective_allowed):
            raise ActionValidationError("The current stage has a frozen objective")
        if action.action_type == "update_loss_weights":
            self._validate_weight_update(action, registry, stage, observable_state)
        if action.action_type == "replace_sampling_subset":
            self._validate_sampling_replacement(action, registry, stage, observable_state)
        if action.action_type == "advance_curriculum_state":
            self._validate_curriculum_advance(action, registry, stage, observable_state)
        return action

    @staticmethod
    def _validate_curriculum_advance(
        action: AdaptiveAction,
        registry: TrainingComponentRegistry,
        stage: Any,
        state: TrainerObservableState,
    ) -> None:
        if not registry.capabilities.supports_curriculum_advance:
            raise ActionValidationError("Trainer does not support curriculum advance")
        if not bool(stage.first_order) or not bool(stage.dynamic_objective_allowed):
            raise ActionValidationError("Curriculum advance requires a mutable first-order stage")
        curriculum_id = str(
            action.after.get("curriculum_id")
            or getattr(action, "curriculum_id", "")
        )
        descriptors = {item.curriculum_id: item for item in registry.curricula}
        if curriculum_id not in descriptors:
            raise ActionValidationError(f"Curriculum action targets unknown runtime: {curriculum_id}")
        descriptor = descriptors[curriculum_id]
        if not (
            descriptor.mutable_during_first_order_stage
            and descriptor.fixed_during_second_order_stage
            and descriptor.validation_supported
            and descriptor.state_snapshot_supported
            and descriptor.monotonic
            and descriptor.single_step_advance_only
        ):
            raise ActionValidationError("Curriculum runtime capability declaration is incomplete")
        if state.pending_rollback:
            raise ActionValidationError("A rollback is already pending")
        if state.coordinator_observation_active:
            raise ActionValidationError("Curriculum advance is blocked by the observation window")
        snapshot = dict(state.curriculum_states.get(curriculum_id) or {})
        if not snapshot:
            raise ActionValidationError("Current curriculum runtime state is unavailable")
        current_index = int(snapshot.get("current_level_index", -1))
        current_id = str(snapshot.get("current_level_id", ""))
        from_index = int(action.after.get(
            "from_level_index", getattr(action, "from_level_index", -1)
        ))
        from_id = str(action.after.get(
            "from_level_id", getattr(action, "from_level_id", "")
        ))
        target_index = int(action.after.get(
            "to_level_index", getattr(action, "to_level_index", -1)
        ))
        target_id = str(action.after.get(
            "to_level_id", getattr(action, "to_level_id", "")
        ))
        if from_index != current_index or from_id != current_id:
            raise ActionValidationError("Curriculum action source level is stale")
        if target_index != current_index + 1:
            raise ActionValidationError("Curriculum actions may advance exactly one level")
        if target_index > int(descriptor.final_level_index):
            raise ActionValidationError("Curriculum target exceeds the final level")
        if target_index >= len(descriptor.level_ids) or descriptor.level_ids[target_index] != target_id:
            raise ActionValidationError("Curriculum target level is not registered")
        action_version = int(action.after.get(
            "runtime_state_version", getattr(action, "runtime_state_version", -1)
        ))
        if action_version != int(snapshot.get("runtime_state_version", -2)):
            raise ActionValidationError("Curriculum runtime state version changed after proposal")
        action_digest = str(action.after.get(
            "runtime_state_digest", getattr(action, "runtime_state_digest", "")
        ))
        if not action_digest or action_digest != str(snapshot.get("state_digest", "")):
            raise ActionValidationError("Curriculum runtime state digest changed after proposal")
        forced = bool(action.after.get("forced", getattr(action, "forced", False)))
        if forced and not descriptor.force_advance_supported:
            raise ActionValidationError("Curriculum runtime does not support forced advance")
        component_ids = set(getattr(action, "associated_component_ids", ()))
        sampler_ids = set(getattr(action, "associated_sampler_ids", ()))
        if component_ids != set(descriptor.associated_loss_component_ids):
            raise ActionValidationError("Curriculum action loss associations do not match Registry")
        if sampler_ids != set(descriptor.associated_sampler_ids):
            raise ActionValidationError("Curriculum action sampler associations do not match Registry")
        if not component_ids.issubset(registry.component_ids):
            raise ActionValidationError("Curriculum action references unregistered loss components")
        registered_samplers = {item.sampler_id for item in registry.sampling_components}
        if not sampler_ids.issubset(registered_samplers):
            raise ActionValidationError("Curriculum action references unregistered samplers")
        minimum_remaining = int(
            action.limits.get("minimum_remaining_budget_after_advance", 0)
        )
        remaining = int(
            state.curriculum_budget_state.get(
                "remaining_iteration_budget", state.remaining_iteration_budget
            )
        )
        requested_compute = int(getattr(action, "requested_compute_budget", 0))
        if remaining < minimum_remaining or requested_compute > remaining:
            raise ActionValidationError("Curriculum action exceeds the remaining training budget")
        if action.requested_iteration_budget > remaining:
            raise ActionValidationError("Curriculum observation budget is unavailable")

        forbidden_key_tokens = (
            "coordinate", "raw_points", "point_tensor", "replacement_points",
            "pde_parameter", "coefficient", "problem_definition", "callable",
            "optimizer_family", "network_parameter", "add_loss", "remove_loss",
            "sampler_strategy",
        )

        def unsafe(value: Any) -> bool:
            if callable(value):
                return True
            if isinstance(value, dict):
                return any(
                    any(token in str(key).casefold() for token in forbidden_key_tokens)
                    or unsafe(item)
                    for key, item in value.items()
                )
            if isinstance(value, (tuple, list, set)):
                return any(unsafe(item) for item in value)
            return not isinstance(value, (str, int, float, bool, type(None)))

        if any(unsafe(value) for value in (action.before, action.after, action.reason, action.limits)):
            raise ActionValidationError("Curriculum actions may not carry executable or structural payloads")

    @staticmethod
    def _validate_sampling_replacement(
        action: AdaptiveAction,
        registry: TrainingComponentRegistry,
        stage: Any,
        state: TrainerObservableState,
    ) -> None:
        if not bool(stage.first_order) or not bool(stage.dynamic_objective_allowed):
            raise ActionValidationError("Sampling replacement requires a mutable first-order stage")
        sampler_id = str(action.after.get("sampler_id") or getattr(action, "sampler_id", ""))
        descriptors = {item.sampler_id: item for item in registry.sampling_components}
        if sampler_id not in descriptors:
            raise ActionValidationError(f"Sampling action targets unknown sampler: {sampler_id}")
        descriptor = descriptors[sampler_id]
        if not descriptor.replacement_supported:
            raise ActionValidationError("Sampler does not support subset replacement")
        if not descriptor.mutable_during_first_order_stage:
            raise ActionValidationError("Sampler is not mutable in first-order stages")
        if state.pending_rollback:
            raise ActionValidationError("A rollback is already pending")
        if state.coordinator_observation_active:
            raise ActionValidationError("Sampling replacement is blocked by the observation window")
        replacement = float(action.after.get("replacement_fraction", -1.0))
        retention = float(action.after.get("hard_point_retention_fraction", -1.0))
        exploration = float(action.after.get("exploration_fraction", -1.0))
        if not 0.0 < replacement < 1.0:
            raise ActionValidationError("replacement_fraction must be in (0, 1)")
        if not 0.0 <= retention <= 1.0 or not 0.0 <= exploration <= 1.0:
            raise ActionValidationError("retention and exploration fractions must be in [0, 1]")
        if replacement + retention > 1.0 + 1e-12:
            raise ActionValidationError("replacement and hard-retention fractions overlap")
        component_state = dict(state.sampling_component_states.get(sampler_id) or {})
        point_count = int(component_state.get("point_count", descriptor.point_budget or 0))
        if point_count < int(descriptor.minimum_point_count):
            raise ActionValidationError("Sampler point count is below its minimum")
        if descriptor.maximum_point_count is not None and point_count > int(descriptor.maximum_point_count):
            raise ActionValidationError("Sampler point count is above its maximum")
        expected_version = int(component_state.get("sampler_state_version", 0))
        action_version = int(action.after.get("sampler_state_version", -1))
        if action_version != expected_version:
            raise ActionValidationError("Sampler state version changed after action proposal")
        replacement_count = max(1, int(point_count * replacement))
        candidate_count = int(action.after.get("candidate_count", 0))
        if candidate_count < replacement_count:
            raise ActionValidationError("Candidate count cannot cover the requested replacements")
        requested_compute = int(getattr(action, "requested_compute_budget", candidate_count))
        remaining_compute = int(state.sampling_budget_state.get("remaining_candidate_points", 0))
        if requested_compute < candidate_count or requested_compute > remaining_compute:
            raise ActionValidationError("Sampling action exceeds the candidate scoring budget")
        scoring_ids = set(getattr(action, "scoring_component_ids", ()))
        if not scoring_ids or not scoring_ids.issubset(set(descriptor.associated_loss_component_ids)):
            raise ActionValidationError("Sampling scores must use associated registered loss components")

        coordinate_tokens = ("coordinate", "coordinates", "replacement_points", "point_tensor", "raw_points")

        def carries_coordinates(value: Any) -> bool:
            if isinstance(value, dict):
                return any(
                    any(token in str(key).casefold() for token in coordinate_tokens)
                    or carries_coordinates(item)
                    for key, item in value.items()
                )
            if isinstance(value, (tuple, list, set)):
                return any(carries_coordinates(item) for item in value)
            return False

        if any(carries_coordinates(value) for value in (action.before, action.after, action.reason)):
            raise ActionValidationError("Sampling actions may not carry coordinates")

    @staticmethod
    def _validate_weight_update(
        action: AdaptiveAction,
        registry: TrainingComponentRegistry,
        stage: Any,
        state: TrainerObservableState,
    ) -> None:
        if not action.after:
            raise ActionValidationError("A loss-weight action must contain updates")
        minimum = float(action.limits.get("minimum_weight", 0.0))
        maximum = float(action.limits.get("maximum_weight", float("inf")))
        minimum_ratio = float(action.limits.get("minimum_update_ratio", 0.0))
        maximum_ratio = float(action.limits.get("maximum_update_ratio", float("inf")))
        if not (0.0 < minimum < maximum):
            raise ActionValidationError("Invalid global loss-weight bounds")
        if not (0.0 < minimum_ratio <= 1.0 <= maximum_ratio):
            raise ActionValidationError("Invalid per-action loss-weight ratio bounds")
        eligible = set(registry.active_weightable_component_ids(stage.stage_id))
        for component_id, value in action.after.items():
            if component_id not in eligible:
                raise ActionValidationError(
                    f"Loss component is not dynamically weightable in this stage: {component_id}"
                )
            old = state.loss_weights.get(component_id)
            if old is None or not math.isfinite(float(old)) or float(old) <= 0.0:
                raise ActionValidationError(f"Current weight is unavailable for {component_id}")
            new = float(value)
            ratio = new / float(old)
            if not math.isfinite(new) or not minimum <= new <= maximum:
                raise ActionValidationError(f"Weight violates global bounds for {component_id}")
            if not minimum_ratio - 1e-12 <= ratio <= maximum_ratio + 1e-12:
                raise ActionValidationError(f"Weight violates ratio bounds for {component_id}")

    @classmethod
    def _contains_reference_metric(cls, action: AdaptiveAction) -> bool:
        def visit(value: Any) -> bool:
            if isinstance(value, dict):
                return any(
                    any(token in str(key).casefold() for token in REFERENCE_TOKENS)
                    or visit(item)
                    for key, item in value.items()
                )
            if isinstance(value, (tuple, list, set)):
                return any(visit(item) for item in value)
            return False

        return visit(action.before) or visit(action.after) or visit(action.reason)

    @classmethod
    def _find_structural_fields(cls, value: Any) -> set[str]:
        found: set[str] = set()
        if isinstance(value, dict):
            for key, item in value.items():
                normalized = str(key).casefold()
                if normalized in STRUCTURAL_FIELDS:
                    found.add(normalized)
                found.update(cls._find_structural_fields(item))
        elif isinstance(value, (tuple, list)):
            for item in value:
                found.update(cls._find_structural_fields(item))
        return found
