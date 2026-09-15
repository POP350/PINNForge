"""Structured, JSON-safe records for run-scoped posterior memory."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from forge.utils.json_safety import json_safe


@dataclass
class TrainingDynamicsSummary:
    early_stage: dict[str, Any] = field(default_factory=dict)
    middle_stage: dict[str, Any] = field(default_factory=dict)
    late_stage: dict[str, Any] = field(default_factory=dict)
    optimizer_switch_stage: dict[str, Any] = field(default_factory=dict)
    best_iteration: int | None = None
    stagnation_detected: bool = False
    stagnation_start_iteration: int | None = None
    gradient_statistics: dict[str, Any] = field(default_factory=dict)
    reference_mse_trace_available: bool = False

    def to_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))


@dataclass
class FailureEvidence:
    label: str
    observed_evidence: list[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))


@dataclass
class CandidatePosterior:
    pde_id: str
    run_id: str
    generation: int
    candidate_id: str
    seed: int
    rank: int | None
    algorithm_spec: dict[str, Any]
    training_budget: dict[str, Any]
    final_metrics: dict[str, Any]
    training_dynamics: dict[str, Any]
    failure_flags: list[str]
    failure_evidence: list[dict[str, Any]]
    lineage: dict[str, Any]
    observed_facts: list[str]
    inferred_causes: list[dict[str, Any]]
    recommended_adjustments: list[dict[str, Any]]
    training_success: bool
    status: str

    def to_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))


@dataclass
class GenerationPosterior:
    pde_id: str
    run_id: str
    generation: int
    ranking_metric: str
    ranked_candidates: list[dict[str, Any]] = field(default_factory=list)
    successful_patterns: list[dict[str, Any]] = field(default_factory=list)
    failure_patterns: list[dict[str, Any]] = field(default_factory=list)
    training_observations: list[str] = field(default_factory=list)
    top3_comparison: list[dict[str, Any]] = field(default_factory=list)
    recommended_next_actions: list[dict[str, Any]] = field(default_factory=list)
    unresolved_hypotheses: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))


@dataclass
class ActivePosteriorRule:
    rule_id: str
    rule: str
    rule_type: str
    support_count: int
    contradiction_count: int
    confidence: float
    scope: dict[str, str]
    guidance_type: str = "soft"
    can_be_overridden: bool = True
    evidence_generations: list[int] = field(default_factory=list)
    component: str | None = None
    principle_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))


@dataclass
class PosteriorPromptContext:
    previous_top3: list[dict[str, Any]] = field(default_factory=list)
    latest_generation_posterior: dict[str, Any] = field(default_factory=dict)
    previous_generation_candidates: list[dict[str, Any]] = field(default_factory=list)
    previous_generation_failures: list[dict[str, Any]] = field(default_factory=list)
    active_run_posterior_rules: list[dict[str, Any]] = field(default_factory=list)
    recommended_next_actions: list[dict[str, Any]] = field(default_factory=list)
    unresolved_hypotheses: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))
