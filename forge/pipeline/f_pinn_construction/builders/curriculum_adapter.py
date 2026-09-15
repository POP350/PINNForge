"""Builder-side compatibility adapters for predeclared ordered curricula.

This module is intentionally outside the generic Trainer control plane: it
translates one registered axis representation into concrete sampler mutations.
"""

from __future__ import annotations

from copy import deepcopy
import math
import time
from typing import Any, Mapping

from forge.pipeline.g_training_evaluation.training.curriculum_runtime import (
    CurriculumStateSnapshot,
    CurriculumTransitionPreview,
    CurriculumTransitionResult,
    deterministic_state_digest,
)
from forge.pipeline.g_training_evaluation.training.registry import (
    CurriculumDescriptor,
    OptimizationStageDescriptor,
)


class OrderedSamplerWindowCurriculumRuntime:
    """Adapt predeclared ordered sampler-window fractions to the runtime API."""

    def __init__(
        self,
        *,
        sampler: Any,
        level_values: tuple[float, ...],
        associated_loss_component_ids: tuple[str, ...],
        associated_sampler_ids: tuple[str, ...],
        associated_variable_ids: tuple[str, ...],
        critical_component_ids: tuple[str, ...],
        structural_component_ids: tuple[str, ...],
        structural_sampler_ids: tuple[str, ...],
        curriculum_id: str = "registered_axis_curriculum",
    ) -> None:
        values = tuple(float(value) for value in level_values)
        if len(values) < 2 or any(
            not math.isfinite(value) or not 0.0 < value <= 1.0
            for value in values
        ):
            raise ValueError("Ordered sampler-window levels must contain finite fractions in (0, 1]")
        if any(right <= left for left, right in zip(values, values[1:])):
            raise ValueError("Ordered sampler-window levels must be strictly increasing")
        if not math.isclose(values[-1], 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("The final sampler-window level must cover the declared full scope")
        self.sampler = sampler
        self.level_values = values
        self._structural_component_ids = tuple(sorted(structural_component_ids))
        self._structural_sampler_ids = tuple(sorted(structural_sampler_ids))
        self._descriptor = CurriculumDescriptor(
            curriculum_id=curriculum_id,
            strategy_name="ordered_sampler_window",
            axis_role="temporal",
            level_ids=tuple(f"level_{index}" for index in range(len(values))),
            initial_level_index=0,
            final_level_index=len(values) - 1,
            monotonic=True,
            single_step_advance_only=True,
            rollback_supported=True,
            force_advance_supported=True,
            associated_loss_component_ids=associated_loss_component_ids,
            associated_sampler_ids=associated_sampler_ids,
            associated_variable_ids=associated_variable_ids,
            mutable_during_first_order_stage=True,
            fixed_during_second_order_stage=True,
            validation_supported=True,
            state_snapshot_supported=True,
            critical_component_ids=critical_component_ids,
            metadata={
                "probe_comparability_mode": "global_fixed_probe",
                "cross_level_best_checkpoint_supported": True,
                "post_advance_sampler_metadata_policy": "rebuild_affected",
            },
        )
        self.current_level_index = 0
        self.total_advances = 0
        self.forced_advances = 0
        self.rollback_count = 0
        self.last_advance_iteration: int | None = None
        self.last_rollback_iteration: int | None = None
        self.runtime_state_version = 0
        self._level_entry_iteration = 0
        self._last_observed_iteration = 0
        self.sampler.set_time_window_fraction(self.level_values[0])

    def get_descriptor(self) -> CurriculumDescriptor:
        return self._descriptor

    def _identity_state(self) -> dict[str, Any]:
        return {
            "curriculum_id": self._descriptor.curriculum_id,
            "strategy_name": self._descriptor.strategy_name,
            "axis_role": self._descriptor.axis_role,
            "level_ids": self._descriptor.level_ids,
            "level_values": self.level_values,
            "current_level_index": int(self.current_level_index),
            "total_advances": int(self.total_advances),
            "forced_advances": int(self.forced_advances),
            "rollback_count": int(self.rollback_count),
            "last_advance_iteration": self.last_advance_iteration,
            "last_rollback_iteration": self.last_rollback_iteration,
            "runtime_state_version": int(self.runtime_state_version),
            "level_entry_iteration": int(self._level_entry_iteration),
            "last_observed_iteration": int(self._last_observed_iteration),
        }

    def get_current_state(self) -> CurriculumStateSnapshot:
        identity = self._identity_state()
        return CurriculumStateSnapshot(
            curriculum_id=self._descriptor.curriculum_id,
            strategy_name=self._descriptor.strategy_name,
            axis_role=self._descriptor.axis_role,
            current_level_index=self.current_level_index,
            current_level_id=self._descriptor.level_ids[self.current_level_index],
            final_level_index=self._descriptor.final_level_index,
            iterations_in_current_level=max(
                0, self._last_observed_iteration - self._level_entry_iteration
            ),
            total_advances=self.total_advances,
            forced_advances=self.forced_advances,
            rollback_count=self.rollback_count,
            last_advance_iteration=self.last_advance_iteration,
            last_rollback_iteration=self.last_rollback_iteration,
            runtime_state_version=self.runtime_state_version,
            state_digest=deterministic_state_digest(identity),
            metadata={
                "is_final_level": self.current_level_index
                >= self._descriptor.final_level_index,
                "probe_comparability_mode": "global_fixed_probe",
            },
        )

    def can_advance(
        self,
        target_level_index: int,
        stage: OptimizationStageDescriptor,
    ) -> bool:
        return bool(
            stage.first_order
            and stage.dynamic_objective_allowed
            and int(target_level_index) == self.current_level_index + 1
            and int(target_level_index) <= self._descriptor.final_level_index
        )

    def preview_advance(self, target_level_index: int) -> CurriculumTransitionPreview:
        target = int(target_level_index)
        if target != self.current_level_index + 1 or target > self._descriptor.final_level_index:
            raise ValueError("Curriculum preview target must be the adjacent registered level")
        return CurriculumTransitionPreview(
            curriculum_id=self._descriptor.curriculum_id,
            from_level_index=self.current_level_index,
            to_level_index=target,
            associated_loss_component_ids=self._descriptor.associated_loss_component_ids,
            associated_sampler_ids=self._descriptor.associated_sampler_ids,
            associated_variable_ids=self._descriptor.associated_variable_ids,
            structural_component_ids=self._structural_component_ids,
            structural_sampler_ids=self._structural_sampler_ids,
            fixed_point_budget=True,
            metadata={"target_level_id": self._descriptor.level_ids[target]},
        )

    def apply_advance(
        self,
        target_level_index: int,
        *,
        iteration: int,
        forced: bool,
    ) -> CurriculumTransitionResult:
        preview = self.preview_advance(target_level_index)
        started = time.perf_counter()
        target = int(target_level_index)
        self.sampler.set_time_window_fraction(self.level_values[target])
        refresh = getattr(self.sampler, "refresh_curriculum_samplers", None)
        if not callable(refresh):
            raise ValueError("Sampler does not implement curriculum refresh")
        refresh(self._descriptor.associated_sampler_ids)
        previous = self.current_level_index
        self.current_level_index = target
        self.total_advances += 1
        self.forced_advances += int(bool(forced))
        self.last_advance_iteration = int(iteration)
        self._level_entry_iteration = int(iteration)
        self._last_observed_iteration = int(iteration)
        self.runtime_state_version += 1
        state = self.get_current_state()
        return CurriculumTransitionResult(
            curriculum_id=self._descriptor.curriculum_id,
            from_level_index=previous,
            to_level_index=target,
            runtime_state_version=state.runtime_state_version,
            state_digest=state.state_digest,
            refreshed_sampler_ids=preview.associated_sampler_ids,
            refresh_cost=time.perf_counter() - started,
            fixed_point_budget_preserved=True,
            metadata={"level_id": state.current_level_id},
        )

    def note_training_iteration(self, iteration: int) -> None:
        self._last_observed_iteration = max(
            self._last_observed_iteration, int(iteration)
        )

    def record_rollback(self, iteration: int) -> None:
        self.rollback_count += 1
        self.last_rollback_iteration = int(iteration)
        self._last_observed_iteration = max(
            self._last_observed_iteration, int(iteration)
        )
        self.runtime_state_version += 1

    def validate_runtime_state(self) -> bool:
        return bool(
            0 <= self.current_level_index <= self._descriptor.final_level_index
            and math.isclose(
                float(getattr(self.sampler, "time_window_fraction", float("nan"))),
                self.level_values[self.current_level_index],
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )

    def snapshot_state(self) -> Mapping[str, Any]:
        state = deepcopy(self._identity_state())
        state["state_digest"] = deterministic_state_digest(state)
        return state

    def restore_state(self, state: Mapping[str, Any]) -> None:
        value = dict(state or {})
        expected = value.pop("state_digest", None)
        if expected is not None and deterministic_state_digest(value) != str(expected):
            raise ValueError("Curriculum runtime checkpoint digest is invalid")
        if tuple(float(item) for item in value.get("level_values") or ()) != self.level_values:
            raise ValueError("Curriculum runtime level definitions changed")
        if tuple(value.get("level_ids") or ()) != self._descriptor.level_ids:
            raise ValueError("Curriculum runtime level IDs changed")
        if str(value.get("curriculum_id")) != self._descriptor.curriculum_id:
            raise ValueError("Curriculum runtime ID changed")
        self.current_level_index = int(value["current_level_index"])
        self.total_advances = int(value.get("total_advances") or 0)
        self.forced_advances = int(value.get("forced_advances") or 0)
        self.rollback_count = int(value.get("rollback_count") or 0)
        self.last_advance_iteration = value.get("last_advance_iteration")
        self.last_rollback_iteration = value.get("last_rollback_iteration")
        self.runtime_state_version = int(value.get("runtime_state_version") or 0)
        self._level_entry_iteration = int(value.get("level_entry_iteration") or 0)
        self._last_observed_iteration = int(value.get("last_observed_iteration") or 0)
        self.sampler.set_time_window_fraction(
            self.level_values[self.current_level_index]
        )
        if not self.validate_runtime_state():
            raise ValueError("Restored curriculum runtime state is inconsistent")
