"""Deterministic, evidence-only generation posterior construction."""

from __future__ import annotations

from collections import Counter
from typing import Any

from forge.pipeline.posterior_memory.models import GenerationPosterior


def build_generation_posterior(
    candidates: list[dict[str, Any]],
    *,
    pde_id: str,
    run_id: str,
    generation: int,
    ranking_metric: str = "global_reference_mse",
) -> GenerationPosterior:
    ranked = sorted(
        candidates,
        key=lambda item: (
            item.get("rank") is None,
            int(item.get("rank") or 10**9),
            _metric(item),
        ),
    )
    successful = [item for item in ranked if item.get("training_success")]
    failed = [item for item in ranked if not item.get("training_success") or item.get("failure_flags")]
    top3 = [item for item in successful if item.get("rank") in {1, 2, 3}]
    successful_patterns = _successful_patterns(top3)
    failure_patterns = _failure_patterns(failed)
    comparisons = _top3_comparisons(top3)
    observations = _training_observations(ranked, top3)
    recommendations = _deduplicated_recommendations(ranked)
    unresolved = _unresolved_hypotheses(top3, comparisons)
    return GenerationPosterior(
        pde_id=pde_id,
        run_id=run_id,
        generation=generation,
        ranking_metric=ranking_metric,
        ranked_candidates=[
            {
                "candidate_id": item.get("candidate_id"),
                "rank": item.get("rank"),
                "global_reference_mse": _metric(item),
                "training_success": bool(item.get("training_success")),
                "failure_flags": list(item.get("failure_flags") or []),
            }
            for item in ranked
        ],
        successful_patterns=successful_patterns,
        failure_patterns=failure_patterns,
        training_observations=observations,
        top3_comparison=comparisons,
        recommended_next_actions=recommendations,
        unresolved_hypotheses=unresolved,
    )


def _successful_patterns(top3: list[dict[str, Any]]) -> list[dict[str, Any]]:
    components = Counter(
        component for item in top3 for component in _executable_components(item.get("algorithm_spec") or {})
    )
    principles = Counter(
        principle
        for item in top3
        for principle in _applied_principles(item.get("algorithm_spec") or {})
    )
    patterns: list[dict[str, Any]] = []
    for component, count in components.most_common():
        if count >= 2:
            patterns.append(
                {
                    "pattern_type": "shared_component",
                    "component": component,
                    "support_count": count,
                    "observation": f"{component} appeared in {count} formally ranked Top-3 candidates.",
                    "causal_claim": False,
                    "guidance_type": "soft",
                    "confidence": min(0.90, 0.50 + 0.12 * count),
                    "can_be_overridden": True,
                }
            )
    for principle, count in principles.most_common():
        if count >= 2:
            patterns.append(
                {
                    "pattern_type": "applied_principle",
                    "principle_id": principle,
                    "support_count": count,
                    "observation": f"Principle {principle} was applied by {count} formally ranked Top-3 candidates.",
                    "causal_claim": False,
                    "guidance_type": "soft",
                    "confidence": min(0.90, 0.50 + 0.12 * count),
                    "can_be_overridden": True,
                }
            )
    return patterns[:20]


def _failure_patterns(failed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(flag for item in failed for flag in item.get("failure_flags") or [])
    return [
        {
            "pattern_type": "failure_flag",
            "failure_flag": flag,
            "support_count": count,
            "observation": f"{flag} occurred in {count} candidate(s) in this generation.",
            "guidance_type": "soft",
            "confidence": min(0.98, 0.55 + 0.12 * count),
            "can_be_overridden": True,
        }
        for flag, count in counts.most_common()
    ]


def _top3_comparisons(top3: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not top3:
        return []
    best = top3[0]
    result = []
    for candidate in top3[1:]:
        changed = _changed_modules(
            best.get("algorithm_spec") or {}, candidate.get("algorithm_spec") or {}
        )
        result.append(
            {
                "rank1_candidate_id": best.get("candidate_id"),
                "compared_candidate_id": candidate.get("candidate_id"),
                "rank1_mse": _metric(best),
                "compared_mse": _metric(candidate),
                "different_modules": changed,
                "attribution": (
                    "Multiple executable modules differ; the performance difference cannot be uniquely attributed to one field."
                    if len(changed) > 1
                    else "One top-level executable module differs, but this comparison alone is still not causal evidence."
                ),
            }
        )
    return result


def _training_observations(
    ranked: list[dict[str, Any]], top3: list[dict[str, Any]]
) -> list[str]:
    observations: list[str] = []
    if top3:
        observations.append(
            f"Rank-1 {top3[0].get('candidate_id')} achieved global reference MSE {_metric(top3[0]):.8g}."
        )
    plateau = sum(
        bool((item.get("training_dynamics") or {}).get("stagnation_detected")) for item in ranked
    )
    if plateau:
        observations.append(f"Training stagnation was detected in {plateau} candidate(s).")
    failures = Counter(flag for item in ranked for flag in item.get("failure_flags") or [])
    if failures:
        flag, count = failures.most_common(1)[0]
        observations.append(f"The most frequent measured failure was {flag} ({count} candidate(s)).")
    return observations


def _deduplicated_recommendations(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for item in candidates:
        for recommendation in item.get("recommended_adjustments") or []:
            if not isinstance(recommendation, dict):
                continue
            action = str(recommendation.get("action") or "").strip()
            if action:
                selected.setdefault(action, dict(recommendation))
    return list(selected.values())[:12]


def _unresolved_hypotheses(
    top3: list[dict[str, Any]], comparisons: list[dict[str, Any]]
) -> list[str]:
    unresolved = []
    if comparisons and any(len(item.get("different_modules") or []) > 1 for item in comparisons):
        unresolved.append(
            "Top-ranked candidates differ in multiple executable modules, so individual component effects remain unresolved."
        )
    if len(top3) < 3:
        unresolved.append("Fewer than three successful formal candidates limit within-generation comparison.")
    return unresolved


def _executable_components(spec: dict[str, Any]) -> list[str]:
    network = spec.get("network") or {}
    sampling = spec.get("sampling") or {}
    loss = spec.get("loss") or {}
    optimization = spec.get("optimization") or {}
    enforcement = spec.get("constraint_enforcement") or {}
    result = [
        f"network.architecture={network.get('architecture')}",
        f"network.activation={((network.get('activation') or {}).get('name'))}",
        f"sampling.interior={((sampling.get('interior') or {}).get('strategy'))}",
        f"loss.weighting={((loss.get('weighting_strategy') or {}).get('name'))}",
        f"constraint_enforcement={enforcement.get('method')}",
    ]
    optimizers = [str(item.get("optimizer")) for item in optimization.get("phases") or []]
    if optimizers:
        result.append(f"optimization.sequence={'->'.join(optimizers)}")
    return [item for item in result if not item.endswith("=None")]


def _applied_principles(spec: dict[str, Any]) -> list[str]:
    metadata = spec.get("generation_metadata") or {}
    return [
        str(item)
        for key in (
            "macro_physics_principles_applied",
            "architecture_design_principles_applied",
        )
        for item in metadata.get(key) or []
        if str(item).strip()
    ]


def _changed_modules(left: dict[str, Any], right: dict[str, Any]) -> list[str]:
    modules = (
        "network",
        "sampling",
        "constraint_enforcement",
        "loss",
        "optimization",
        "training",
    )
    return [name for name in modules if left.get(name) != right.get(name)]


def _metric(candidate: dict[str, Any]) -> float:
    value = (candidate.get("final_metrics") or {}).get("global_reference_mse")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("inf")

