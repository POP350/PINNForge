"""PDE-independent runtime contract for fixed-budget sampling mutation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import torch


@dataclass(frozen=True)
class SamplingBatch:
    sampler_id: str
    points: Any = field(repr=False)
    point_metadata: tuple[Mapping[str, Any], ...] = ()
    state_version: int = 0

    @property
    def point_count(self) -> int:
        shape = getattr(self.points, "shape", ())
        return int(shape[0]) if shape else 0


@dataclass(frozen=True)
class PointScoreResult:
    sampler_id: str
    scores: torch.Tensor = field(repr=False)
    component_scores: Mapping[str, torch.Tensor] = field(default_factory=dict, repr=False)
    statistics: Mapping[str, float] = field(default_factory=dict)
    scoring_component_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class SamplingMutationResult:
    sampler_id: str
    point_count_before: int
    point_count_after: int
    points_replaced: int
    exploration_points: int
    state_version_before: int
    state_version_after: int


@runtime_checkable
class SamplingRuntimeProtocol(Protocol):
    def get_training_points(self, sampler_id: str) -> SamplingBatch: ...

    def score_points(
        self,
        model: torch.nn.Module,
        component_ids: tuple[str, ...],
        points: SamplingBatch,
        context: Any,
    ) -> PointScoreResult: ...

    def generate_candidate_points(
        self, sampler_id: str, count: int, rng_state: Any | None = None
    ) -> SamplingBatch: ...

    def replace_subset(
        self,
        sampler_id: str,
        retained_indices: Sequence[int],
        replacement_points: SamplingBatch,
        **context: Any,
    ) -> SamplingMutationResult: ...

    def snapshot_state(self) -> Mapping[str, Any]: ...

    def restore_state(self, state: Mapping[str, Any]) -> None: ...
