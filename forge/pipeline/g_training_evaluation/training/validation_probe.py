"""Fixed, component-driven physics validation without evaluator data."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import random
from statistics import median
from typing import Any, Mapping

import numpy as np
import torch

from .registry import TrainingComponentRegistry


@dataclass(frozen=True)
class ValidationProbeBudgetPolicy:
    points_per_component: int = 32
    maximum_total_points: int = 512


@dataclass(frozen=True)
class ValidationProbeSet:
    probe_set_id: str
    seed: int
    component_budgets: Mapping[str, int]
    batch: Any = field(repr=False, compare=False)


@dataclass(frozen=True)
class PhysicsValidationResult:
    aggregate_score: float
    category_scores: Mapping[str, float]
    component_scores: Mapping[str, float]
    normalized_component_scores: Mapping[str, float]
    worst_component_id: str | None
    worst_component_score: float | None
    median_component_score: float | None
    p90_component_score: float | None
    normalized_region_scores: Mapping[int, float] = field(default_factory=dict)
    mean_region_score: float | None = None
    worst_region_id: int | None = None
    worst_region_score: float | None = None
    auxiliary_scores: Mapping[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "aggregate_score": self.aggregate_score,
            "category_scores": dict(self.category_scores),
            "component_scores": dict(self.component_scores),
            "normalized_component_scores": dict(self.normalized_component_scores),
            "worst_component_id": self.worst_component_id,
            "worst_component_score": self.worst_component_score,
            "median_component_score": self.median_component_score,
            "p90_component_score": self.p90_component_score,
            "normalized_region_scores": {
                str(name): float(value)
                for name, value in self.normalized_region_scores.items()
            },
            "mean_region_score": self.mean_region_score,
            "worst_region_id": self.worst_region_id,
            "worst_region_score": self.worst_region_score,
            "auxiliary_scores": {
                str(name): float(value)
                for name, value in self.auxiliary_scores.items()
            },
        }


class ValidationProbeManager:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = dict(config or {})
        self.enabled = bool(config.get("enabled", False))
        self.evaluation_interval = max(1, int(config.get("evaluation_interval", 100)))
        self.normalization = str(config.get("normalization", "initial_probe_value"))
        self.category_aggregation = str(
            config.get("category_aggregation", "median_worst_blend")
        )
        self.median_weight = float(config.get("median_weight", 0.7))
        self.worst_weight = float(config.get("worst_weight", 0.3))
        self.epsilon = float(config.get("epsilon", 1e-12))
        self.minimum_component_scale = max(
            self.epsilon,
            float(config.get("minimum_component_scale", self.epsilon)),
        )
        self.region_normalization = str(
            config.get("region_normalization", "per_region_initial")
        )
        self.second_order_degradation_ratio = max(
            1.0,
            float(config.get("second_order_degradation_ratio", 1.0)),
        )
        self.second_order_guard_metric = str(
            config.get(
                "second_order_guard_metric",
                "physics_validation_score",
            )
        )
        self.seed = int(config.get("seed", 1729))
        self.category_weights = {
            str(name): float(value)
            for name, value in (config.get("category_weights") or {}).items()
        }
        self.policy = ValidationProbeBudgetPolicy(
            points_per_component=max(1, int(config.get("points_per_component", 32))),
            maximum_total_points=max(1, int(config.get("maximum_total_points", 512))),
        )
        self.probe_set: ValidationProbeSet | None = None
        self.initial_scales: dict[str, float] = {}
        self.initial_region_scales: dict[int, float] = {}
        self.last_result: PhysicsValidationResult | None = None
        self.history: list[dict[str, Any]] = []
        self._expected_probe_set_id: str | None = None

    def build_probes(
        self,
        registry: TrainingComponentRegistry,
        sampler: Any,
        budget_policy: ValidationProbeBudgetPolicy | None = None,
        seed: int | None = None,
    ) -> ValidationProbeSet | None:
        if not self.enabled or not registry.capabilities.supports_validation_probes:
            return None
        policy = budget_policy or self.policy
        supported = [
            item
            for item in registry.sampling_components
            if item.validation_probe_supported
        ]
        total_point_multiplier = sum(
            max(1, int(item.metadata.get("point_multiplier", 1)))
            for item in supported
        )
        per_component = min(
            policy.points_per_component,
            max(1, policy.maximum_total_points // max(1, total_point_multiplier)),
        )
        component_budgets = {
            item.sampler_id: min(
                per_component,
                int(item.point_budget) if item.point_budget is not None and item.point_budget > 0 else per_component,
            )
            for item in supported
        }
        overrides: dict[str, int] = {}
        for item in supported:
            overrides[str(item.sampler_id)] = component_budgets[
                item.sampler_id
            ]
            if item.domain_role == "interior":
                overrides["interior"] = component_budgets[item.sampler_id]
        probe_seed = self.seed if seed is None else int(seed)
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        try:
            random.seed(probe_seed)
            np.random.seed(probe_seed % (2**32 - 1))
            torch.manual_seed(probe_seed)
            probe_sampler = sampler.probe_clone()
            batch = probe_sampler.sample(overrides)
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "schema": registry.schema_version,
                    "seed": probe_seed,
                    "budgets": component_budgets,
                    "components": sorted(registry.component_ids),
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:20]
        self.probe_set = ValidationProbeSet(
            probe_set_id=fingerprint,
            seed=probe_seed,
            component_budgets=component_budgets,
            batch=batch,
        )
        if (
            self._expected_probe_set_id is not None
            and self._expected_probe_set_id != self.probe_set.probe_set_id
        ):
            raise ValueError("Validation probe set does not match the checkpoint")
        return self.probe_set

    def evaluate(
        self,
        model: torch.nn.Module,
        registry: TrainingComponentRegistry,
        loss_runtime: Any,
        probe_set: ValidationProbeSet | None = None,
    ) -> PhysicsValidationResult | None:
        selected = probe_set or self.probe_set
        if not self.enabled or selected is None:
            return None
        loss_state = loss_runtime.state_dict()
        was_training = bool(model.training)
        capture_residual_statistics = bool(
            getattr(loss_runtime, "capture_residual_statistics", False)
        )
        component_ids = {
            item.component_id
            for item in registry.loss_components
            if item.validation_enabled
        }
        try:
            model.eval()
            loss_runtime.set_active_term_names(component_ids)
            loss_runtime.freeze_dynamic_weighting()
            loss_runtime.capture_residual_statistics = True
            with torch.enable_grad():
                _, components = loss_runtime(model, selected.batch)
            residual_statistics = dict(
                getattr(loss_runtime, "last_residual_statistics", {}) or {}
            )
            raw = {
                component_id: float(components[component_id])
                for component_id in sorted(component_ids)
                if component_id in components
                and math.isfinite(float(components[component_id]))
            }
            auxiliary_scores: dict[str, float] = {}
            score_provider = getattr(
                loss_runtime.problem,
                "physics_validation_scores",
                None,
            )
            if callable(score_provider):
                provided = score_provider(model, selected.batch) or {}
                auxiliary_scores = {
                    str(name): float(value)
                    for name, value in provided.items()
                    if math.isfinite(float(value))
                }
        finally:
            model.train(was_training)
            loss_runtime.capture_residual_statistics = (
                capture_residual_statistics
            )
            loss_runtime.load_state_dict(loss_state)
            loss_runtime.last_term_values = []
            loss_runtime.last_term_names = []
            clear = getattr(loss_runtime.problem, "clear_autodiff_cache", None)
            if callable(clear):
                clear()
        if not self.initial_scales:
            self.initial_scales = {
                name: max(
                    abs(value),
                    self.minimum_component_scale,
                )
                for name, value in raw.items()
            }
        normalized = {
            name: value / max(self.initial_scales.get(name, abs(value)), self.epsilon)
            for name, value in raw.items()
        }
        raw_region_scores = {
            int(item["region_id"]): float(
                item["training_residual_rmse"]
            )
            for item in residual_statistics.get("region_statistics") or []
            if isinstance(item, Mapping)
            and item.get("region_id") is not None
            and item.get("training_residual_rmse") is not None
            and math.isfinite(float(item["training_residual_rmse"]))
        }
        if (
            raw_region_scores
            and self.region_normalization != "shared_pde_rmse"
            and not self.initial_region_scales
        ):
            self.initial_region_scales = {
                region_id: max(abs(value), self.epsilon)
                for region_id, value in raw_region_scores.items()
            }
        if self.region_normalization == "shared_pde_rmse":
            governing_scales = [
                self.initial_scales[descriptor.component_id]
                for descriptor in registry.loss_components
                if descriptor.category == "equation_residual"
                and descriptor.component_id in self.initial_scales
            ]
            shared_region_scale = math.sqrt(
                max(
                    sum(governing_scales) / len(governing_scales)
                    if governing_scales
                    else self.minimum_component_scale,
                    self.epsilon,
                )
            )
            normalized_region_scores = {
                region_id: value / shared_region_scale
                for region_id, value in raw_region_scores.items()
            }
        else:
            normalized_region_scores = {
                region_id: value
                / max(
                    self.initial_region_scales.get(
                        region_id, abs(value)
                    ),
                    self.epsilon,
                )
                for region_id, value in raw_region_scores.items()
            }
        result = self._aggregate(
            registry,
            raw,
            normalized,
            normalized_region_scores=normalized_region_scores,
            auxiliary_scores=auxiliary_scores,
        )
        self.last_result = result
        return result

    def next_due_iteration(self, current_iteration: int) -> int:
        last_iteration = (
            int(self.history[-1].get("iteration", 0))
            if self.history
            else 0
        )
        return max(
            int(current_iteration),
            last_iteration + int(self.evaluation_interval),
        )

    def is_due(self, current_iteration: int) -> bool:
        last_iteration = (
            int(self.history[-1].get("iteration", 0))
            if self.history
            else 0
        )
        return (
            int(current_iteration) - last_iteration
            >= int(self.evaluation_interval)
        )

    def _aggregate(
        self,
        registry: TrainingComponentRegistry,
        raw: dict[str, float],
        normalized: dict[str, float],
        *,
        normalized_region_scores: dict[int, float] | None = None,
        auxiliary_scores: dict[str, float] | None = None,
    ) -> PhysicsValidationResult:
        by_category: dict[str, list[float]] = {}
        for descriptor in registry.loss_components:
            if descriptor.component_id in normalized:
                by_category.setdefault(descriptor.category, []).append(
                    normalized[descriptor.component_id]
                )
        regional = dict(normalized_region_scores or {})
        if regional:
            pde_category = next(
                (
                    descriptor.category
                    for descriptor in registry.loss_components
                    if descriptor.category == "equation_residual"
                ),
                "equation_residual",
            )
            by_category.setdefault(pde_category, []).extend(
                regional.values()
            )
        category_scores: dict[str, float] = {}
        for category, values in sorted(by_category.items()):
            category_scores[category] = (
                self.median_weight * median(values)
                + self.worst_weight * max(values)
            )
        weights = {
            category: self.category_weights.get(category, 1.0)
            for category in category_scores
        }
        weight_total = sum(weights.values())
        aggregate = (
            sum(category_scores[name] * weights[name] for name in category_scores)
            / max(weight_total, self.epsilon)
            if category_scores
            else float("inf")
        )
        ordered = sorted(normalized.values())
        p90_index = max(0, math.ceil(0.9 * len(ordered)) - 1) if ordered else 0
        worst_id = max(normalized, key=normalized.get) if normalized else None
        worst_region_id = (
            max(regional, key=regional.get) if regional else None
        )
        return PhysicsValidationResult(
            aggregate_score=float(aggregate),
            category_scores=category_scores,
            component_scores=raw,
            normalized_component_scores=normalized,
            worst_component_id=worst_id,
            worst_component_score=normalized.get(worst_id) if worst_id else None,
            median_component_score=median(ordered) if ordered else None,
            p90_component_score=ordered[p90_index] if ordered else None,
            normalized_region_scores=regional,
            mean_region_score=(
                sum(regional.values()) / len(regional)
                if regional
                else None
            ),
            worst_region_id=worst_region_id,
            worst_region_score=(
                regional.get(worst_region_id)
                if worst_region_id is not None
                else None
            ),
            auxiliary_scores=dict(auxiliary_scores or {}),
        )

    def score_for_metric(
        self,
        result: PhysicsValidationResult | None,
        metric_name: str,
    ) -> float | None:
        """Return a finite, label-free score exposed by a fixed probe."""

        if result is None:
            return None
        if metric_name == "physics_validation_score":
            value = float(result.aggregate_score)
        else:
            raw = result.auxiliary_scores.get(str(metric_name))
            if raw is None:
                return None
            value = float(raw)
        return value if math.isfinite(value) else None

    def second_order_guard_score(
        self,
        result: PhysicsValidationResult | None,
    ) -> float | None:
        return self.score_for_metric(result, self.second_order_guard_metric)

    def record(self, iteration: int, stage_id: str, result: PhysicsValidationResult) -> None:
        self.history.append(
            {"iteration": int(iteration), "stage_id": str(stage_id), **result.to_dict()}
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "probe_set_id": self.probe_set.probe_set_id if self.probe_set else None,
            "probe_seed": self.probe_set.seed if self.probe_set else self.seed,
            "component_budgets": dict(self.probe_set.component_budgets) if self.probe_set else {},
            "initial_scales": dict(self.initial_scales),
            "initial_region_scales": {
                str(name): float(value)
                for name, value in self.initial_region_scales.items()
            },
            "last_result": self.last_result.to_dict() if self.last_result else None,
            "history": list(self.history),
        }

    def load_state_dict(self, state: dict[str, Any] | None) -> None:
        state = dict(state or {})
        expected = state.get("probe_set_id")
        self._expected_probe_set_id = str(expected) if expected else None
        if expected and self.probe_set and expected != self.probe_set.probe_set_id:
            raise ValueError("Validation probe set does not match the checkpoint")
        self.initial_scales = {
            str(name): float(value)
            for name, value in (state.get("initial_scales") or {}).items()
        }
        self.initial_region_scales = {
            int(name): float(value)
            for name, value in (
                state.get("initial_region_scales") or {}
            ).items()
        }
        result = state.get("last_result")
        if result:
            payload = dict(result)
            payload["normalized_region_scores"] = {
                int(name): float(value)
                for name, value in (
                    payload.get("normalized_region_scores") or {}
                ).items()
            }
            payload["auxiliary_scores"] = {
                str(name): float(value)
                for name, value in (
                    payload.get("auxiliary_scores") or {}
                ).items()
            }
            self.last_result = PhysicsValidationResult(**payload)
        else:
            self.last_result = None
        self.history = list(state.get("history") or [])
