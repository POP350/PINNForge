"""PDE-independent ordered curriculum runtime contracts.

The control plane only sees registered level identities and immutable state
snapshots.  Concrete runtimes own the meaning of a level and every mutation
needed to enter it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping, Protocol, runtime_checkable

from .registry import CurriculumDescriptor, OptimizationStageDescriptor


def _immutable_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(value or {}))


def _canonical_value(value: Any) -> Any:
    """Return a strict JSON value without object-address fallbacks."""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Curriculum digest values must be finite")
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    raise TypeError(
        f"Curriculum digest does not support {type(value).__name__}; "
        "runtime identity state must be plain serializable data"
    )


def deterministic_state_digest(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        _canonical_value(value),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class CurriculumStateSnapshot:
    curriculum_id: str
    strategy_name: str
    axis_role: str
    current_level_index: int
    current_level_id: str
    final_level_index: int
    iterations_in_current_level: int = 0
    total_advances: int = 0
    forced_advances: int = 0
    rollback_count: int = 0
    last_advance_iteration: int | None = None
    last_rollback_iteration: int | None = None
    runtime_state_version: int = 0
    state_digest: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "curriculum_id", str(self.curriculum_id))
        object.__setattr__(self, "strategy_name", str(self.strategy_name))
        object.__setattr__(self, "axis_role", str(self.axis_role))
        object.__setattr__(self, "current_level_id", str(self.current_level_id))
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))
        if not self.curriculum_id or not self.current_level_id:
            raise ValueError("Curriculum state IDs must not be empty")
        if not 0 <= int(self.current_level_index) <= int(self.final_level_index):
            raise ValueError("Curriculum state level indices are inconsistent")
        for name in (
            "iterations_in_current_level", "total_advances", "forced_advances",
            "rollback_count", "runtime_state_version",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"Curriculum state {name} must be non-negative")
        if not self.state_digest:
            raise ValueError("Curriculum state digest must not be empty")

    @property
    def is_final_level(self) -> bool:
        return int(self.current_level_index) >= int(self.final_level_index)

    def to_dict(self) -> dict[str, Any]:
        return {
            "curriculum_id": self.curriculum_id,
            "strategy_name": self.strategy_name,
            "axis_role": self.axis_role,
            "current_level_index": int(self.current_level_index),
            "current_level_id": self.current_level_id,
            "final_level_index": int(self.final_level_index),
            "iterations_in_current_level": int(self.iterations_in_current_level),
            "total_advances": int(self.total_advances),
            "forced_advances": int(self.forced_advances),
            "rollback_count": int(self.rollback_count),
            "last_advance_iteration": self.last_advance_iteration,
            "last_rollback_iteration": self.last_rollback_iteration,
            "runtime_state_version": int(self.runtime_state_version),
            "state_digest": self.state_digest,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class CurriculumTransitionPreview:
    curriculum_id: str
    from_level_index: int
    to_level_index: int
    associated_loss_component_ids: tuple[str, ...]
    associated_sampler_ids: tuple[str, ...]
    associated_variable_ids: tuple[str, ...]
    structural_component_ids: tuple[str, ...]
    structural_sampler_ids: tuple[str, ...]
    fixed_point_budget: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "associated_loss_component_ids", "associated_sampler_ids",
            "associated_variable_ids", "structural_component_ids",
            "structural_sampler_ids",
        ):
            object.__setattr__(
                self, name, tuple(sorted(str(item) for item in getattr(self, name)))
            )
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))


@dataclass(frozen=True)
class CurriculumTransitionResult:
    curriculum_id: str
    from_level_index: int
    to_level_index: int
    runtime_state_version: int
    state_digest: str
    refreshed_sampler_ids: tuple[str, ...] = ()
    refresh_cost: float = 0.0
    fixed_point_budget_preserved: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "refreshed_sampler_ids",
            tuple(sorted(str(item) for item in self.refreshed_sampler_ids)),
        )
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))
        if float(self.refresh_cost) < 0.0:
            raise ValueError("Curriculum refresh cost must be non-negative")


@runtime_checkable
class CurriculumRuntimeProtocol(Protocol):
    def get_descriptor(self) -> CurriculumDescriptor:
        ...

    def get_current_state(self) -> CurriculumStateSnapshot:
        ...

    def can_advance(
        self,
        target_level_index: int,
        stage: OptimizationStageDescriptor,
    ) -> bool:
        ...

    def preview_advance(self, target_level_index: int) -> CurriculumTransitionPreview:
        ...

    def apply_advance(
        self,
        target_level_index: int,
        *,
        iteration: int,
        forced: bool,
    ) -> CurriculumTransitionResult:
        ...

    def note_training_iteration(self, iteration: int) -> None:
        ...

    def record_rollback(self, iteration: int) -> None:
        ...

    def validate_runtime_state(self) -> bool:
        ...

    def snapshot_state(self) -> Mapping[str, Any]:
        ...

    def restore_state(self, state: Mapping[str, Any]) -> None:
        ...

