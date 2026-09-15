"""Build evidence-grounded candidate posterior summaries from formal records."""

from __future__ import annotations

from collections import defaultdict
import math
from statistics import median
from typing import Any

from forge.pipeline.posterior_memory.models import (
    CandidatePosterior,
    FailureEvidence,
    TrainingDynamicsSummary,
)
from forge.utils.json_safety import json_safe


SUPPORTED_FAILURE_FLAGS = {
    "zero_solution_collapse",
    "constant_solution_collapse",
    "initial_condition_forgetting",
    "boundary_violation",
    "pde_residual_stagnation",
    "gradient_explosion",
    "nan_or_inf",
    "optimizer_divergence",
    "over_smoothing",
    "shock_region_failure",
    "excessive_runtime",
    "memory_overflow",
    "high_seed_variance",
    "low_train_loss_high_reference_error",
}


def build_candidate_posterior(
    record: dict[str, Any], *, pde_id: str, run_id: str
) -> CandidatePosterior:
    training_report = dict(record.get("training_report") or {})
    history = [item for item in training_report.get("history") or [] if isinstance(item, dict)]
    dynamics = summarize_training_dynamics(history)
    metrics = dict(record.get("generation_metrics") or record.get("metrics") or {})
    final_metrics = _final_metrics(metrics, training_report)
    evidence = detect_failure_evidence(record, dynamics.to_dict(), final_metrics)
    failure_flags = [item.label for item in evidence]
    observed = _observed_facts(record, dynamics.to_dict(), final_metrics, evidence)
    inferred, recommendations = _inferences_and_recommendations(evidence, dynamics.to_dict())
    spec = record.get("normalized_algorithm_spec") or record.get("algorithm_spec") or {}
    budget = record.get("training_budget") or {}
    optimization_phases = (spec.get("optimization") or {}).get("phases") or []
    actual_spec_iterations = sum(
        int(phase.get("iterations") or 0)
        for phase in optimization_phases
        if isinstance(phase, dict)
    )
    iterations = (
        budget.get("generation_budget_iterations")
        or record.get("generation_budget_iterations")
        or (record.get("budget") or {}).get("maximum_total_iterations")
        or actual_spec_iterations
    )
    return CandidatePosterior(
        pde_id=pde_id,
        run_id=run_id,
        generation=int(record.get("generation_index", record.get("generation", 0)) or 0),
        candidate_id=str(record.get("spec_id") or record.get("candidate_id") or "unknown"),
        seed=int(record.get("training_seed", record.get("seed", 0)) or 0),
        rank=_optional_int(record.get("generation_rank")),
        algorithm_spec=json_safe(spec),
        training_budget={
            "iterations": _optional_int(iterations),
            "generation_recommendation": _optional_int(
                record.get("generation_iteration_recommendation")
                or (record.get("budget") or {}).get("recommended_total_iterations")
            ),
        },
        final_metrics=final_metrics,
        training_dynamics=dynamics.to_dict(),
        failure_flags=failure_flags,
        failure_evidence=[item.to_dict() for item in evidence],
        lineage={
            "parent_ids": list(record.get("parent_spec_ids") or []),
            "population_role": record.get("population_role"),
            "candidate_role": record.get("candidate_role"),
            "source_candidate_id": record.get("source_candidate_id"),
            "fresh_training": bool(record.get("fresh_training", True)),
        },
        observed_facts=observed,
        inferred_causes=inferred,
        recommended_adjustments=recommendations,
        training_success=bool(record.get("training_success")),
        status=str(record.get("status") or "unknown"),
    )


def summarize_training_dynamics(history: list[dict[str, Any]]) -> TrainingDynamicsSummary:
    ordered = sorted(history, key=lambda item: int(item.get("iteration") or 0))
    if not ordered:
        return TrainingDynamicsSummary()
    partitions = _partition_stages(ordered)
    stage_summaries = {
        name: _stage_summary(items) for name, items in partitions.items()
    }
    reference_points = [
        item for item in ordered if _finite(item.get("reference_mse")) is not None
    ]
    target_series = reference_points or [
        item for item in ordered if _finite(item.get("total_loss")) is not None
    ]
    target_key = "reference_mse" if reference_points else "total_loss"
    best = min(target_series, key=lambda item: float(item[target_key])) if target_series else None
    switch = _optimizer_switch_summary(ordered)
    gradients = [
        value for item in ordered if (value := _finite(item.get("gradient_norm"))) is not None
    ]
    late = stage_summaries["late_stage"]
    stagnant = late.get("total_loss_trend") == "plateau" or late.get("pde_loss_trend") == "plateau"
    return TrainingDynamicsSummary(
        early_stage=stage_summaries["early_stage"],
        middle_stage=stage_summaries["middle_stage"],
        late_stage=late,
        optimizer_switch_stage=switch,
        best_iteration=_optional_int(best.get("iteration")) if best else None,
        stagnation_detected=stagnant,
        stagnation_start_iteration=(
            _optional_int(partitions["late_stage"][0].get("iteration"))
            if stagnant and partitions["late_stage"]
            else None
        ),
        gradient_statistics={
            "maximum": max(gradients) if gradients else None,
            "median": median(gradients) if gradients else None,
            "last": gradients[-1] if gradients else None,
        },
        reference_mse_trace_available=bool(reference_points),
    )


def detect_failure_evidence(
    record: dict[str, Any], dynamics: dict[str, Any], final_metrics: dict[str, Any]
) -> list[FailureEvidence]:
    found: dict[str, FailureEvidence] = {}

    def add(label: str, evidence: str, confidence: float) -> None:
        item = found.setdefault(label, FailureEvidence(label=label, confidence=confidence))
        item.confidence = max(item.confidence, confidence)
        if evidence not in item.observed_evidence:
            item.observed_evidence.append(evidence)

    failure_text = str(record.get("failure_reason") or "").lower()
    if bool(final_metrics.get("nan_detected")) or "non_finite" in failure_text or "nan" in failure_text:
        add("nan_or_inf", "Training or evaluation reported a non-finite value.", 0.99)
    if "out of memory" in failure_text or "cuda oom" in failure_text:
        add("memory_overflow", "The candidate failed with an out-of-memory error.", 0.99)

    gradients = dynamics.get("gradient_statistics") or {}
    maximum_gradient = _finite(gradients.get("maximum"))
    median_gradient = _finite(gradients.get("median"))
    if maximum_gradient is not None and maximum_gradient > max(1_000.0, 100.0 * (median_gradient or 1.0)):
        add("gradient_explosion", f"Maximum gradient norm reached {maximum_gradient:.6g}.", 0.92)

    late = dynamics.get("late_stage") or {}
    if late.get("pde_loss_trend") in {"plateau", "rebound"}:
        add("pde_residual_stagnation", f"Late PDE-loss trend was {late.get('pde_loss_trend')}.", 0.78)
    switch_effect = (dynamics.get("optimizer_switch_stage") or {}).get("effect")
    if switch_effect == "negative" or "optimizer" in failure_text and "diverg" in failure_text:
        add("optimizer_divergence", "The optimizer switch or optimizer phase increased the tracked objective.", 0.82)

    early = dynamics.get("early_stage") or {}
    initial_late_ratio = _finite(late.get("initial_loss_end_to_start_ratio"))
    reference_late_ratio = _finite(late.get("reference_mse_end_to_start_ratio"))
    if initial_late_ratio is not None and initial_late_ratio > 1.25 and (
        reference_late_ratio is None or reference_late_ratio > 1.10
    ):
        add("initial_condition_forgetting", f"Late initial loss increased by a factor of {initial_late_ratio:.3g}.", 0.82 if reference_late_ratio else 0.68)

    mse = _finite(final_metrics.get("global_reference_mse"))
    rel = _finite(final_metrics.get("relative_l2_error"))
    pred_std = _finite(final_metrics.get("prediction_std"))
    pred_mean = _finite(final_metrics.get("prediction_mean"))
    ref_std = _finite(final_metrics.get("reference_std"))
    if pred_std is not None and ref_std and pred_std / max(ref_std, 1e-12) < 0.01 and (rel or 0.0) > 0.5:
        if pred_mean is not None and abs(pred_mean) < 0.05 * max(ref_std, 1e-12):
            add("zero_solution_collapse", "Prediction variance and mean were near zero while relative error remained high.", 0.96)
        else:
            add("constant_solution_collapse", "Prediction variance was near zero while relative error remained high.", 0.94)
    elif pred_std is not None and ref_std and pred_std / max(ref_std, 1e-12) < 0.25 and (rel or 0.0) > 0.2:
        add("over_smoothing", "Prediction variance was much smaller than reference variance.", 0.83)

    boundary = _finite(final_metrics.get("boundary_rmse"))
    if boundary is not None and boundary > 0.10:
        add("boundary_violation", f"Final boundary error was {boundary:.6g}.", 0.80)
    shock = _finite(final_metrics.get("shock_region_mse"))
    if shock is not None and mse is not None and shock > max(0.05, 2.0 * mse):
        add("shock_region_failure", f"Shock-region MSE {shock:.6g} exceeded global MSE {mse:.6g}.", 0.86)

    final_loss = _finite(final_metrics.get("final_train_loss"))
    if final_loss is not None and mse is not None and final_loss < 1e-4 and mse > 0.05:
        add("low_train_loss_high_reference_error", f"Final train loss was {final_loss:.6g}, but reference MSE was {mse:.6g}.", 0.93)
    runtime = _finite(final_metrics.get("runtime_seconds"))
    iterations = _optional_int((record.get("training_budget") or {}).get("generation_budget_iterations"))
    iterations = iterations or _optional_int(record.get("generation_budget_iterations"))
    if runtime is not None and (runtime > 3600.0 or iterations and runtime / iterations > 2.0):
        add("excessive_runtime", f"Runtime was {runtime:.3f} seconds.", 0.75)
    seed_variance = _finite((record.get("metrics") or {}).get("seed_variance"))
    if seed_variance is not None and seed_variance > 0.25:
        add("high_seed_variance", f"Reported seed variance was {seed_variance:.6g}.", 0.90)
    return [found[key] for key in sorted(found)]


def _partition_stages(history: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    count = len(history)
    first = max(1, count // 3)
    second = max(first + 1, 2 * count // 3)
    return {
        "early_stage": history[:first],
        "middle_stage": history[first:second],
        "late_stage": history[second:],
    }


def _stage_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in ("total_loss", "pde_loss", "boundary_loss", "initial_loss", "reference_mse"):
        values = [value for item in items if (value := _finite(item.get(key))) is not None]
        result[f"{key}_trend"] = _trend(values)
        result[f"{key}_first"] = values[0] if values else None
        result[f"{key}_last"] = values[-1] if values else None
        result[f"{key}_end_to_start_ratio"] = (
            values[-1] / max(abs(values[0]), 1e-12) if len(values) >= 2 else None
        )
    if items:
        result["iteration_range"] = [items[0].get("iteration"), items[-1].get("iteration")]
    return result


def _trend(values: list[float]) -> str:
    if len(values) < 2:
        return "unavailable"
    width = max(1, len(values) // 4)
    start = sum(values[:width]) / width
    end = sum(values[-width:]) / width
    change = (end - start) / max(abs(start), 1e-12)
    if change <= -0.50:
        return "rapid_decrease"
    if change <= -0.10:
        return "decrease"
    if change >= 0.10:
        return "rebound"
    return "plateau"


def _optimizer_switch_summary(history: list[dict[str, Any]]) -> dict[str, Any]:
    phases: list[str] = []
    by_phase: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in history:
        phase = str(item.get("phase") or "unknown")
        if phase not in phases:
            phases.append(phase)
        by_phase[phase].append(item)
    if len(phases) < 2:
        return {"detected": False, "effect": "unavailable", "phases": phases}
    before, after = by_phase[phases[-2]], by_phase[phases[-1]]
    key = "reference_mse" if any(_finite(item.get("reference_mse")) is not None for item in before + after) else "total_loss"
    before_values = [value for item in before if (value := _finite(item.get(key))) is not None]
    after_values = [value for item in after if (value := _finite(item.get(key))) is not None]
    if not before_values or not after_values:
        return {"detected": True, "effect": "unavailable", "phases": phases, "metric": key}
    ratio = after_values[-1] / max(abs(before_values[-1]), 1e-12)
    effect = "positive" if ratio < 0.90 else "negative" if ratio > 1.10 else "neutral"
    return {
        "detected": True,
        "from_phase": phases[-2],
        "to_phase": phases[-1],
        "metric": key,
        "before": before_values[-1],
        "after": after_values[-1],
        "end_to_pre_switch_ratio": ratio,
        "effect": effect,
    }


def _final_metrics(metrics: dict[str, Any], training_report: dict[str, Any]) -> dict[str, Any]:
    return json_safe(
        {
            "global_reference_mse": metrics.get("mse"),
            "relative_l2_error": metrics.get("relative_l2_error", metrics.get("l2re")),
            "pde_residual_rmse": metrics.get("pde_residual_error", metrics.get("pde_residual")),
            "boundary_rmse": metrics.get("boundary_error"),
            "initial_rmse": metrics.get("initial_error"),
            "shock_region_mse": metrics.get("shock_region_mse", metrics.get("shock_region_error")),
            "prediction_mean": metrics.get("prediction_mean"),
            "prediction_std": metrics.get("prediction_std"),
            "reference_std": metrics.get("reference_std"),
            "runtime_seconds": metrics.get("training_time", training_report.get("training_time")),
            "final_train_loss": training_report.get("final_loss"),
            "nan_detected": bool(metrics.get("nan_detected") or training_report.get("nan_detected")),
            "peak_gpu_memory": metrics.get("peak_gpu_memory", training_report.get("peak_gpu_memory")),
        }
    )


def _observed_facts(
    record: dict[str, Any], dynamics: dict[str, Any], metrics: dict[str, Any], evidence: list[FailureEvidence]
) -> list[str]:
    facts = [
        f"Candidate status was {record.get('status') or 'unknown'}; training_success={bool(record.get('training_success'))}."
    ]
    mse = _finite(metrics.get("global_reference_mse"))
    if mse is not None:
        facts.append(f"Final global reference MSE was {mse:.8g}.")
    for stage_name in ("early_stage", "middle_stage", "late_stage"):
        stage = dynamics.get(stage_name) or {}
        facts.append(
            f"{stage_name} PDE-loss trend was {stage.get('pde_loss_trend', 'unavailable')}; "
            f"initial-loss trend was {stage.get('initial_loss_trend', 'unavailable')}."
        )
    switch = dynamics.get("optimizer_switch_stage") or {}
    if switch.get("detected"):
        facts.append(f"Optimizer switch effect on {switch.get('metric')} was {switch.get('effect')}.")
    for item in evidence:
        facts.extend(item.observed_evidence)
    return list(dict.fromkeys(facts))[:20]


def _inferences_and_recommendations(
    evidence: list[FailureEvidence], dynamics: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    action_map = {
        "initial_condition_forgetting": "increase protection of the initial condition during late optimization",
        "boundary_violation": "strengthen boundary enforcement or rebalance the boundary term",
        "pde_residual_stagnation": "change residual sampling, representation, or the late optimizer schedule",
        "gradient_explosion": "reduce the unstable learning rate or add evidence-justified gradient protection",
        "nan_or_inf": "avoid the measured unstable component combination and use a safer optimization schedule",
        "optimizer_divergence": "revise the optimizer transition and its learning-rate or chunk parameters",
        "zero_solution_collapse": "increase IC/BC anchoring and avoid the measured collapse-prone initialization",
        "constant_solution_collapse": "increase representational variation and constraint anchoring",
        "over_smoothing": "increase high-frequency or localized representation capacity",
        "shock_region_failure": "increase evidence-driven resolution in the shock region",
        "excessive_runtime": "reduce avoidable sampling or architecture cost while retaining physical coverage",
        "memory_overflow": "reduce peak model or sampling memory demand",
        "high_seed_variance": "prefer more stable components and retain uncertainty in the next design",
        "low_train_loss_high_reference_error": "do not optimize training loss alone; repair the observed generalization failure",
    }
    inferred = [
        {"claim": f"possible {item.label.replace('_', ' ')}", "confidence": item.confidence}
        for item in evidence
    ]
    recommendations = [
        {
            "action": action_map[item.label],
            "reason": item.observed_evidence[0] if item.observed_evidence else item.label,
            "guidance_type": "soft",
            "can_be_overridden": True,
        }
        for item in evidence
        if item.label in action_map
    ]
    if not recommendations and (dynamics.get("optimizer_switch_stage") or {}).get("effect") == "positive":
        recommendations.append(
            {
                "action": "prefer retaining the measured beneficial optimizer transition unless another bottleneck justifies a change",
                "reason": "the optimizer-switch metric improved after the transition",
                "guidance_type": "soft",
                "can_be_overridden": True,
            }
        )
    return inferred, recommendations


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
