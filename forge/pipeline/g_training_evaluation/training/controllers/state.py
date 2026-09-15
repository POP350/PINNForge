"""Strict controller input and action types.

The observable state deliberately has no evaluator/reference-solution fields.
Controllers only receive quantities produced by the optimization loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any, Mapping


@dataclass(frozen=True)
class TrainerObservableState:
    iteration: int
    stage: str
    optimizer_name: str
    current_loss: float | None = None
    recent_losses: tuple[float, ...] = ()
    loss_components: Mapping[str, float] = field(default_factory=dict)
    loss_weights: Mapping[str, float] = field(default_factory=dict)
    component_gradient_norms: Mapping[str, float] = field(default_factory=dict)
    minimum_gradient_cosine: float | None = None
    learning_rates: tuple[float, ...] = ()
    parameter_update_norm: float | None = None
    gradient_norm: float | None = None
    lbfgs_outer_step: int | None = None
    closure_evaluations: int = 0
    closure_evaluations_this_step: int = 0
    step_size: float | None = None
    line_search_failed: bool = False
    completed_stage_iterations: int = 0
    remaining_iteration_budget: int = 0
    sampling_state: Mapping[str, Any] = field(default_factory=dict)
    time_window_fraction: float = 1.0
    numerical_finite: bool = True
    stage_descriptor: Any | None = None
    capability_snapshot: Any | None = None
    eligible_component_ids: tuple[str, ...] = ()
    component_gradient_ema: Mapping[str, float] = field(default_factory=dict)
    gradient_conflicts: tuple[Any, ...] = ()
    optimizer_diagnostics: Mapping[str, Any] = field(default_factory=dict)
    budget_state: Mapping[str, Any] = field(default_factory=dict)
    validation_state: Any | None = None
    sampling_component_states: Mapping[str, Any] = field(default_factory=dict)
    sampling_diagnostics: Mapping[str, Any] = field(default_factory=dict)
    sampling_budget_state: Mapping[str, Any] = field(default_factory=dict)
    curriculum_states: Mapping[str, Any] = field(default_factory=dict)
    curriculum_diagnostics: Mapping[str, Any] = field(default_factory=dict)
    curriculum_budget_state: Mapping[str, Any] = field(default_factory=dict)
    pending_rollback: bool = False
    coordinator_observation_active: bool = False

    @property
    def stage_id(self) -> str:
        return self.stage

    @property
    def component_losses(self) -> Mapping[str, float]:
        return self.loss_components

    @property
    def effective_component_weights(self) -> Mapping[str, float]:
        return self.loss_weights


@dataclass(frozen=True)
class AdaptiveAction:
    action_type: str
    controller: str
    before: Mapping[str, Any] = field(default_factory=dict)
    after: Mapping[str, Any] = field(default_factory=dict)
    reason: Mapping[str, Any] = field(default_factory=dict)
    limits: Mapping[str, Any] = field(default_factory=dict)
    action_id: str = ""
    target_component_ids: tuple[str, ...] = ()
    requested_iteration_budget: int = 0

    def __post_init__(self) -> None:
        targets = tuple(
            sorted(
                str(item)
                for item in (
                    self.target_component_ids
                    or (
                        tuple(str(name) for name in self.after)
                        if self.action_type == "update_loss_weights"
                        else ()
                    )
                )
            )
        )
        object.__setattr__(self, "target_component_ids", targets)
        if not self.action_id:
            payload = repr(
                (
                    self.controller,
                    self.action_type,
                    targets,
                    tuple(sorted((str(name), repr(value)) for name, value in self.after.items())),
                )
            ).encode("utf-8")
            object.__setattr__(self, "action_id", hashlib.sha256(payload).hexdigest()[:16])

    @property
    def controller_id(self) -> str:
        return self.controller


@dataclass(frozen=True)
class ReplaceSamplingSubsetAction(AdaptiveAction):
    sampler_id: str = ""
    replacement_fraction: float = 0.0
    hard_point_retention_fraction: float = 0.0
    exploration_fraction: float = 0.0
    scoring_component_ids: tuple[str, ...] = ()
    score_statistics: Mapping[str, float] = field(default_factory=dict)
    candidate_count: int = 0
    requested_compute_budget: int = 0
    sampler_state_version: int = 0

    def __post_init__(self) -> None:
        if self.action_type != "replace_sampling_subset":
            raise ValueError("ReplaceSamplingSubsetAction requires replace_sampling_subset")
        object.__setattr__(self, "sampler_id", str(self.sampler_id))
        object.__setattr__(
            self, "scoring_component_ids", tuple(sorted(map(str, self.scoring_component_ids)))
        )
        object.__setattr__(self, "target_component_ids", (str(self.sampler_id),))
        super().__post_init__()


@dataclass(frozen=True)
class AdvanceCurriculumStateAction(AdaptiveAction):
    curriculum_id: str = ""
    from_level_index: int = 0
    from_level_id: str = ""
    to_level_index: int = 0
    to_level_id: str = ""
    forced: bool = False
    trigger_statistics: Mapping[str, Any] = field(default_factory=dict)
    runtime_state_version: int = 0
    runtime_state_digest: str = ""
    associated_component_ids: tuple[str, ...] = ()
    associated_sampler_ids: tuple[str, ...] = ()
    requested_compute_budget: int = 0

    def __post_init__(self) -> None:
        if self.action_type != "advance_curriculum_state":
            raise ValueError(
                "AdvanceCurriculumStateAction requires advance_curriculum_state"
            )
        object.__setattr__(self, "curriculum_id", str(self.curriculum_id))
        object.__setattr__(self, "from_level_id", str(self.from_level_id))
        object.__setattr__(self, "to_level_id", str(self.to_level_id))
        object.__setattr__(
            self, "associated_component_ids",
            tuple(sorted(str(item) for item in self.associated_component_ids)),
        )
        object.__setattr__(
            self, "associated_sampler_ids",
            tuple(sorted(str(item) for item in self.associated_sampler_ids)),
        )
        object.__setattr__(
            self,
            "target_component_ids",
            tuple(sorted({
                self.curriculum_id,
                *self.associated_component_ids,
                *self.associated_sampler_ids,
            })),
        )
        super().__post_init__()
