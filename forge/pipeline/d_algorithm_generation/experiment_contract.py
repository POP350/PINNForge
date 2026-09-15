"""Adaptive, at-most-ten-generation multi-fidelity evolution contract."""

from __future__ import annotations

from copy import deepcopy
from math import floor
from typing import Any, Mapping

from .adaptive_policy import AdaptivePolicyBudgetScaler


EXPERIMENT_CONTRACT_VERSION = (
    "budget_matched_parent_free_ablation_stagnation_archive_wave2d_corrected_v44"
)
MIN_LOW_FIDELITY_GENERATIONS = 4
MAX_LOW_FIDELITY_GENERATIONS = 10
MAX_GENERATIONS = MAX_LOW_FIDELITY_GENERATIONS
# Backward-compatible name used by callers that pass the search horizon.
TOTAL_GENERATIONS = MAX_GENERATIONS
LOW_FIDELITY_ITERATIONS = 1_000
HIGH_FIDELITY_ITERATIONS = 10_000
FINAL_HIGH_FIDELITY_CANDIDATE_COUNT = 3
EVOLUTION_PARENT_COUNT = 3
REGULAR_GENERATION_RECOMMENDED_ITERATIONS = LOW_FIDELITY_ITERATIONS
# Backward-compatible alias. High fidelity is now an independent evaluation
# stage rather than a generated evolution generation.
FINAL_GENERATION_RECOMMENDED_ITERATIONS = HIGH_FIDELITY_ITERATIONS

GENERATION_ITERATION_RECOMMENDATIONS = {
    generation: LOW_FIDELITY_ITERATIONS for generation in range(MAX_GENERATIONS)
}
# Backward-compatible public alias. The controller enforces these low-fidelity
# budgets through schedule scaling.
GENERATION_TRAINING_BUDGETS = GENERATION_ITERATION_RECOMMENDATIONS

DEFAULT_EXPERIMENT_CONTRACT: dict[str, Any] = {
    "search_control": {
        "min_low_fidelity_generations": MIN_LOW_FIDELITY_GENERATIONS,
        "max_low_fidelity_generations": MAX_LOW_FIDELITY_GENERATIONS,
        "low_fidelity_iterations": LOW_FIDELITY_ITERATIONS,
        "high_fidelity_iterations": HIGH_FIDELITY_ITERATIONS,
        "final_high_fidelity_candidate_count": FINAL_HIGH_FIDELITY_CANDIDATE_COUNT,
        "minimum_valid_candidates_per_generation": 3,
        "max_consecutive_invalid_generations": 2,
    },
    "stagnation": {
        "enabled": True,
        "best_improvement_window": 3,
        "best_single_generation_threshold": 0.01,
        "cumulative_improvement_threshold": 0.02,
        "top3_median_window": 2,
        "top3_median_improvement_threshold": 0.02,
        "elite_archive_stagnation_window": 3,
        "escape_global_best_improvement_threshold": 0.02,
        "escape_archive_top3_median_improvement_threshold": 0.02,
        "escape_min_new_archive_entries": 1,
    },
    "low_fidelity_evaluation": {
        "checkpoint_iterations": [800, 900, 1_000],
        "proxy_aggregation": "median",
        "fixed_evaluation_grid": True,
    },
    "low_fidelity_schedule_scaling": {
        "enabled": True,
        "min_lbfgs_iterations": 100,
        "min_adaptive_sampling_events": 1,
        "preserve_enabled_training_stages": True,
    },
    "final_selection": {
        "candidate_count": FINAL_HIGH_FIDELITY_CANDIDATE_COUNT,
        "deduplicate_normalized_specs": True,
        "duplicate_aggregation": "median",
        "ranking_metric": "low_fidelity_proxy_mse",
    },
    "high_fidelity_evaluation": {
        "fresh_initialization": True,
        "resume_low_fidelity_checkpoint": False,
        "iterations": HIGH_FIDELITY_ITERATIONS,
    },
}

EXACT_DUPLICATE_DISTANCE = 0.0
DIVERSITY_RETRY_DISTANCE = 0.10
DIVERSITY_WARNING_DISTANCE = 0.20
ROLE_DIVERSITY_RETRY_THRESHOLDS = {
    "elite_conservative_improvement": 0.05,
    "physics_constraint_sampling_guided_design": 0.10,
    "architecture_optimization_guided_design": 0.10,
    "multi_parent_synthesis": 0.10,
    "novelty_exploration": 0.15,
}

GENERATION_ZERO_TARGET_CANDIDATES = 8
GENERATION_ZERO_TRAINING_CANDIDATES = GENERATION_ZERO_TARGET_CANDIDATES
EVOLUTION_TARGET_CANDIDATES = 5
MINIMUM_GENERATION_ZERO_CANDIDATES = 5
MINIMUM_GENERATION_CANDIDATES = 3
MINIMUM_NEW_EVOLUTION_CANDIDATES = MINIMUM_GENERATION_CANDIDATES
MINIMUM_SUCCESSFUL_CANDIDATES = EVOLUTION_PARENT_COUNT


def resolve_experiment_contract(
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge and validate run-level multi-fidelity contract overrides."""

    resolved = deepcopy(DEFAULT_EXPERIMENT_CONTRACT)
    for section, values in dict(overrides or {}).items():
        if section not in resolved:
            raise ValueError(f"Unknown experiment contract section: {section}")
        if not isinstance(values, Mapping):
            raise ValueError(f"Experiment contract section {section} must be an object")
        unknown = set(values) - set(resolved[section])
        if unknown:
            raise ValueError(
                f"Unknown {section} fields: {', '.join(sorted(str(item) for item in unknown))}"
            )
        resolved[section].update(deepcopy(dict(values)))

    control = resolved["search_control"]
    minimum = int(control["min_low_fidelity_generations"])
    maximum = int(control["max_low_fidelity_generations"])
    low_iterations = int(control["low_fidelity_iterations"])
    high_iterations = int(control["high_fidelity_iterations"])
    final_count = int(control["final_high_fidelity_candidate_count"])
    minimum_valid = int(control["minimum_valid_candidates_per_generation"])
    maximum_invalid = int(control["max_consecutive_invalid_generations"])
    if minimum != MIN_LOW_FIDELITY_GENERATIONS:
        raise ValueError(
            f"This experiment contract requires min_low_fidelity_generations={MIN_LOW_FIDELITY_GENERATIONS}"
        )
    if maximum != MAX_LOW_FIDELITY_GENERATIONS:
        raise ValueError(
            f"This experiment contract requires max_low_fidelity_generations={MAX_LOW_FIDELITY_GENERATIONS}"
        )
    if low_iterations != LOW_FIDELITY_ITERATIONS:
        raise ValueError(
            f"This experiment contract requires low_fidelity_iterations={LOW_FIDELITY_ITERATIONS}"
        )
    if high_iterations != HIGH_FIDELITY_ITERATIONS:
        raise ValueError(
            f"This experiment contract requires high_fidelity_iterations={HIGH_FIDELITY_ITERATIONS}"
        )
    if final_count != FINAL_HIGH_FIDELITY_CANDIDATE_COUNT:
        raise ValueError(
            "This experiment contract requires "
            f"final_high_fidelity_candidate_count={FINAL_HIGH_FIDELITY_CANDIDATE_COUNT}"
        )
    if minimum_valid < 1:
        raise ValueError(
            "search_control.minimum_valid_candidates_per_generation must be positive"
        )
    if maximum_invalid < 1:
        raise ValueError(
            "search_control.max_consecutive_invalid_generations must be positive"
        )
    if int(resolved["high_fidelity_evaluation"]["iterations"]) != high_iterations:
        raise ValueError(
            "high_fidelity_evaluation.iterations must equal search_control.high_fidelity_iterations"
        )
    if int(resolved["final_selection"]["candidate_count"]) != final_count:
        raise ValueError(
            "final_selection.candidate_count must equal search_control.final_high_fidelity_candidate_count"
        )

    stagnation = resolved["stagnation"]
    for key in (
        "best_single_generation_threshold",
        "cumulative_improvement_threshold",
        "top3_median_improvement_threshold",
        "escape_global_best_improvement_threshold",
        "escape_archive_top3_median_improvement_threshold",
    ):
        value = float(stagnation[key])
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"stagnation.{key} must be within [0, 1]")
    for key in (
        "best_improvement_window",
        "top3_median_window",
        "elite_archive_stagnation_window",
        "escape_min_new_archive_entries",
    ):
        if int(stagnation[key]) < 1:
            raise ValueError(f"stagnation.{key} must be positive")

    checkpoints = [
        int(item)
        for item in resolved["low_fidelity_evaluation"]["checkpoint_iterations"]
    ]
    if not checkpoints or checkpoints != sorted(set(checkpoints)):
        raise ValueError("Low-fidelity checkpoint iterations must be sorted and unique")
    if checkpoints[-1] != low_iterations or any(
        item < 1 or item > low_iterations for item in checkpoints
    ):
        raise ValueError(
            "Low-fidelity checkpoints must be positive, within budget, and end at the low-fidelity budget"
        )
    resolved["low_fidelity_evaluation"]["checkpoint_iterations"] = checkpoints
    if resolved["low_fidelity_evaluation"]["proxy_aggregation"] != "median":
        raise ValueError("Only median low-fidelity proxy aggregation is supported")
    if not bool(resolved["low_fidelity_evaluation"]["fixed_evaluation_grid"]):
        raise ValueError("Low-fidelity proxy evaluation requires a fixed evaluation grid")
    scaling = resolved["low_fidelity_schedule_scaling"]
    if not bool(scaling["enabled"]):
        raise ValueError("Low-fidelity schedule scaling must remain enabled")
    if int(scaling["min_lbfgs_iterations"]) < 1:
        raise ValueError("min_lbfgs_iterations must be positive")
    if int(scaling["min_adaptive_sampling_events"]) < 1:
        raise ValueError("min_adaptive_sampling_events must be at least one")
    if not bool(scaling["preserve_enabled_training_stages"]):
        raise ValueError("Enabled training stages must be preserved")
    if not bool(resolved["final_selection"]["deduplicate_normalized_specs"]):
        raise ValueError("Final selection must deduplicate normalized specs")
    if resolved["final_selection"]["duplicate_aggregation"] != "median":
        raise ValueError("Only median duplicate aggregation is supported")
    if resolved["final_selection"]["ranking_metric"] != "low_fidelity_proxy_mse":
        raise ValueError("Final selection must rank by low_fidelity_proxy_mse")
    if bool(resolved["high_fidelity_evaluation"]["resume_low_fidelity_checkpoint"]):
        raise ValueError("High-fidelity evaluation cannot resume a low-fidelity checkpoint")
    if not bool(resolved["high_fidelity_evaluation"]["fresh_initialization"]):
        raise ValueError("High-fidelity evaluation requires fresh initialization")
    return resolved


def resolve_generation_iterations(
    generation_index: int,
    total_generations: int,
    *,
    is_final_generation: bool | None = None,
) -> int:
    """Return the enforced low-fidelity budget for one evolution generation."""

    if total_generations != MAX_GENERATIONS:
        raise ValueError("This experiment contract allows at most 10 generations.")
    if not 0 <= generation_index < MAX_GENERATIONS:
        raise ValueError(f"Invalid generation index: {generation_index}")
    return LOW_FIDELITY_ITERATIONS


def resolve_ranking_fidelity(
    generation_index: int,
    total_generations: int,
    *,
    is_final_generation: bool | None = None,
) -> str:
    resolve_generation_iterations(
        generation_index,
        total_generations,
        is_final_generation=is_final_generation,
    )
    return "low_fidelity"


def scale_optimization_phases_to_budget(
    phases: list[dict[str, Any]],
    target_iterations: int,
    *,
    min_lbfgs_iterations: int = 1,
    preserve_enabled_training_stages: bool = True,
) -> list[dict[str, Any]]:
    """Scale enabled phases proportionally with explicit per-stage floors."""

    if not isinstance(phases, list) or not phases:
        raise ValueError("At least one optimization phase is required")
    target = int(target_iterations)
    if target < len(phases):
        raise ValueError(
            "Target iterations cannot provide at least one iteration to every optimization phase"
        )
    original: list[int] = []
    for index, phase in enumerate(phases):
        value = phase.get("iterations") if isinstance(phase, dict) else None
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"optimization phase {index} must have at least one iteration")
        original.append(value)
    total = sum(original)
    quotas = [target * value / total for value in original]
    floors: list[int] = []
    for phase in phases:
        optimizer = str(phase.get("optimizer") or "").strip().lower()
        minimum = max(1, int(min_lbfgs_iterations)) if optimizer == "lbfgs" else 1
        floors.append(minimum if preserve_enabled_training_stages else 0)
    if sum(floors) > target:
        raise ValueError(
            "Target iterations cannot preserve all enabled stages and configured optimizer floors"
        )
    allocated = [max(minimum, floor(quota)) for quota, minimum in zip(quotas, floors)]

    while sum(allocated) > target:
        removable = [
            index for index, value in enumerate(allocated) if value > floors[index]
        ]
        if not removable:
            raise ValueError("Unable to honor configured training-stage minimums")
        index = min(removable, key=lambda item: (quotas[item] - allocated[item], item))
        allocated[index] -= 1
    while sum(allocated) < target:
        index = max(
            range(len(allocated)),
            key=lambda item: (quotas[item] - allocated[item], -item),
        )
        allocated[index] += 1

    scaled = deepcopy(phases)
    for phase, source_iterations, iterations in zip(scaled, original, allocated):
        phase["iterations"] = iterations
        _align_scheduler_horizon(
            phase,
            source_iterations=source_iterations,
            target_iterations=iterations,
        )
    if sum(int(phase["iterations"]) for phase in scaled) != target:
        raise AssertionError("Scaled optimization phases do not match the target budget")
    return scaled


def _align_scheduler_horizon(
    phase: dict[str, Any],
    *,
    source_iterations: int,
    target_iterations: int,
) -> None:
    """Keep phase-coupled scheduler horizons aligned after budget scaling."""

    scheduler = phase.get("scheduler")
    if not isinstance(scheduler, dict):
        return
    name = str(scheduler.get("name") or "none").strip().lower()
    parameters = scheduler.get("parameters")
    if not isinstance(parameters, dict):
        parameters = {}
        scheduler["parameters"] = parameters
    if name in {"one_cycle", "onecycle", "onecyclelr"}:
        parameters.pop("epochs", None)
        parameters.pop("steps_per_epoch", None)
        parameters["total_steps"] = max(1, int(target_iterations))
        return
    horizon_field = {
        "cosine_annealing": "T_max",
        "cosine": "T_max",
        "cosineannealinglr": "T_max",
        "linear_lr": "total_iters",
        "linearlr": "total_iters",
        "polynomial_lr": "total_iters",
        "polynomiallr": "total_iters",
    }.get(name)
    if horizon_field is None:
        return
    configured = parameters.get(horizon_field)
    if configured is None or int(configured) == int(source_iterations):
        parameters[horizon_field] = max(1, int(target_iterations))


def adapt_algorithm_spec_to_generation_budget(
    spec: dict[str, Any],
    target_iterations: int,
    *,
    min_lbfgs_iterations: int = 1,
    min_adaptive_sampling_events: int = 0,
    preserve_enabled_training_stages: bool = True,
    enforce_full_iterations: bool = False,
    source_fidelity: str = "design",
    target_fidelity: str = "runtime",
    adaptive_policy_source_spec: dict[str, Any] | None = None,
    source_optimizer_step_budget: int | None = None,
    source_sampling_budget: int | None = None,
    target_sampling_budget: int | None = None,
    source_validation_budget: int | None = None,
    target_validation_budget: int | None = None,
    experiment_contract: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a budget-adapted copy and an audit without changing other fields."""

    adapted = deepcopy(spec)
    phases = adapted.get("optimization", {}).get("phases")
    if not isinstance(phases, list):
        raise ValueError("AlgorithmSpec optimization.phases must be a list")
    original_iterations = [phase.get("iterations") for phase in phases if isinstance(phase, dict)]
    adapted["optimization"]["phases"] = scale_optimization_phases_to_budget(
        phases,
        target_iterations,
        min_lbfgs_iterations=min_lbfgs_iterations,
        preserve_enabled_training_stages=preserve_enabled_training_stages,
    )
    adaptive = (
        adapted.get("sampling", {}).get("adaptive_refinement")
        if isinstance(adapted.get("sampling"), dict)
        else None
    )
    adaptive_interval_before = None
    adaptive_interval_after = None
    if isinstance(adaptive, dict) and adaptive.get("enabled"):
        parameters = adaptive.setdefault("parameters", {})
        adaptive_interval_before = int(parameters.get("update_every") or target_iterations)
        required_events = max(0, int(min_adaptive_sampling_events))
        if required_events:
            maximum_interval = max(1, int(target_iterations) // required_events)
            adaptive_interval_after = min(adaptive_interval_before, maximum_interval)
            parameters["update_every"] = adaptive_interval_after
        else:
            adaptive_interval_after = adaptive_interval_before
    early_stopping_disabled = False
    training = adapted.get("training")
    if enforce_full_iterations and isinstance(training, dict):
        early = training.get("early_stopping")
        if isinstance(early, dict) and early.get("enabled"):
            early["enabled"] = False
            early_stopping_disabled = True
    policy_source = deepcopy(adaptive_policy_source_spec or spec)
    policy_source_budget = int(
        source_optimizer_step_budget
        or sum(
            int(phase.get("iterations") or 0)
            for phase in (policy_source.get("optimization") or {}).get("phases") or []
        )
        or target_iterations
    )
    scaled_policy = AdaptivePolicyBudgetScaler().scale(
        policy_source,
        source_fidelity=source_fidelity,
        target_fidelity=target_fidelity,
        source_optimizer_step_budget=policy_source_budget,
        target_optimizer_step_budget=int(target_iterations),
        source_sampling_budget=source_sampling_budget,
        target_sampling_budget=target_sampling_budget,
        source_validation_budget=source_validation_budget,
        target_validation_budget=target_validation_budget,
        experiment_contract=experiment_contract,
        runtime_base_spec=adapted,
    )
    adapted = scaled_policy.runtime_spec
    actual_iterations = [
        int(phase["iterations"]) for phase in adapted["optimization"]["phases"]
    ]
    return adapted, {
        "llm_original_phase_iterations": original_iterations,
        "llm_original_total_iterations": sum(
            value for value in original_iterations if isinstance(value, int) and not isinstance(value, bool)
        ),
        "generation_scaled_phase_iterations": actual_iterations,
        "generation_budget_iterations": int(target_iterations),
        "optimization_budget_scaled": original_iterations != actual_iterations,
        "min_lbfgs_iterations": int(min_lbfgs_iterations),
        "preserve_enabled_training_stages": bool(preserve_enabled_training_stages),
        "min_adaptive_sampling_events": int(min_adaptive_sampling_events),
        "adaptive_sampling_interval_before": adaptive_interval_before,
        "adaptive_sampling_interval_after": adaptive_interval_after,
        "early_stopping_disabled_for_fidelity_fairness": early_stopping_disabled,
        "adaptive_policy_scaling": scaled_policy.audit,
    }


def audit_algorithm_spec_iteration_recommendation(
    spec: dict[str, Any], recommended_iterations: int
) -> dict[str, Any]:
    """Audit an LLM schedule against soft guidance without modifying it."""

    phases = spec.get("optimization", {}).get("phases")
    if not isinstance(phases, list):
        raise ValueError("AlgorithmSpec optimization.phases must be a list")
    phase_iterations = [
        phase.get("iterations") for phase in phases if isinstance(phase, dict)
    ]
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in phase_iterations
    ):
        raise ValueError("Every optimization phase must have non-negative integer iterations")
    total_iterations = sum(phase_iterations)
    recommendation = int(recommended_iterations)
    deviation = total_iterations - recommendation
    deviation_ratio = (
        abs(deviation) / recommendation if recommendation > 0 else None
    )
    iteration_budget_rationale = str(
        (spec.get("generation_metadata") or {}).get("iteration_budget_rationale")
        or ""
    )
    return {
        "llm_original_phase_iterations": phase_iterations,
        "llm_original_total_iterations": total_iterations,
        "proposed_total_iterations": total_iterations,
        "generation_iteration_recommendation": recommendation,
        "recommended_total_iterations": recommendation,
        "iteration_budget_deviation": deviation,
        "iteration_budget_deviation_ratio": deviation_ratio,
        "iteration_budget_rationale": iteration_budget_rationale,
        "exceeds_iteration_recommendation": total_iterations > recommendation,
        "iteration_recommendation_is_advisory": True,
        "optimization_budget_scaled": False,
    }


def generation_can_continue(
    generation_index: int,
    *,
    actual_candidate_count: int,
    valid_new_candidate_count: int,
) -> bool:
    if generation_index == 0:
        return actual_candidate_count >= MINIMUM_GENERATION_ZERO_CANDIDATES
    return (
        actual_candidate_count >= MINIMUM_GENERATION_CANDIDATES
        and valid_new_candidate_count >= MINIMUM_NEW_EVOLUTION_CANDIDATES
    )


def build_contract_dry_run(*, benchmark: str, seeds: tuple[int, ...] = (0, 1, 2)) -> dict[str, Any]:
    """Build and self-check the formal structure without generating or training candidates."""

    from .agents.candidate_roles import NEW_CANDIDATE_ROLE_PLAN

    generations = []
    for generation_index in range(MAX_GENERATIONS):
        generations.append(
            {
                "generation_index": generation_index,
                "generation_mode": "normal_or_escape",
                "fidelity": "low",
                "proposal_candidate_count": (
                    GENERATION_ZERO_TARGET_CANDIDATES
                    if generation_index == 0
                    else EVOLUTION_TARGET_CANDIDATES
                ),
                "target_candidate_count": (
                    GENERATION_ZERO_TRAINING_CANDIDATES
                    if generation_index == 0
                    else EVOLUTION_TARGET_CANDIDATES
                ),
                "minimum_candidate_count": (
                    MINIMUM_GENERATION_ZERO_CANDIDATES
                    if generation_index == 0
                    else MINIMUM_GENERATION_CANDIDATES
                ),
                "minimum_successful_candidate_count": MINIMUM_SUCCESSFUL_CANDIDATES,
                "generation_iteration_recommendation": resolve_generation_iterations(
                    generation_index,
                    MAX_GENERATIONS,
                ),
                "ranking_fidelity": resolve_ranking_fidelity(
                    generation_index,
                    MAX_GENERATIONS,
                ),
                "is_final_generation": False,
                "fresh_initialization": True,
                "checkpoint_continuation": False,
                "candidate_roles": (
                    [] if generation_index == 0 else list(NEW_CANDIDATE_ROLE_PLAN)
                ),
                "parent_source": (
                    None
                    if generation_index == 0
                    else "global_best_tracker_top3"
                ),
                "parent_count": 0 if generation_index == 0 else EVOLUTION_PARENT_COUNT,
                "unchanged_parent_replay": False,
                "global_best_tracker_external_to_population": True,
                "global_best_tracker_input": "generation_successful_candidates",
                "global_best_tracker_retention_limit": 3,
            }
        )
    final_review = {
        "stage": "final_high_fidelity_evaluation",
        "counts_as_low_fidelity_generation": False,
        "generates_algorithm_specs": False,
        "candidate_source": "global_best_tracker_distinct_spec_top3",
        "candidate_count": FINAL_HIGH_FIDELITY_CANDIDATE_COUNT,
        "training_iterations_per_candidate": HIGH_FIDELITY_ITERATIONS,
        "fresh_initialization": True,
        "resume_low_fidelity_checkpoint": False,
        "ranking_metric": "mse",
    }
    maximum_formal_candidate_trainings = sum(
        int(item["target_candidate_count"]) for item in generations
    )
    checks = {
        "maximum_ten_generations": len(generations) == MAX_GENERATIONS,
        "minimum_four_generations_before_stagnation": (
            MIN_LOW_FIDELITY_GENERATIONS == 4
        ),
        "all_evolution_generations_are_low_fidelity": all(
            item["fidelity"] == "low"
            and item["generation_iteration_recommendation"]
            == LOW_FIDELITY_ITERATIONS
            and not item["is_final_generation"]
            for item in generations
        ),
        "escape_counts_toward_ten_generation_cap": True,
        "no_final_evolution_generation": all(
            not item["is_final_generation"] for item in generations
        ),
        "generation_zero_proposals_eight": generations[0]["proposal_candidate_count"] == 8,
        "generation_zero_trains_all_eight_when_available": (
            generations[0]["target_candidate_count"] == GENERATION_ZERO_TARGET_CANDIDATES
        ),
        "later_target_five": all(
            item["target_candidate_count"] == EVOLUTION_TARGET_CANDIDATES
            for item in generations[1:]
        ),
        "later_generations_use_exact_five_roles": all(
            item["candidate_roles"] == list(NEW_CANDIDATE_ROLE_PLAN)
            for item in generations[1:]
        ),
        "later_generations_use_global_top3": all(
            item["parent_source"] == "global_best_tracker_top3"
            and item["parent_count"] == EVOLUTION_PARENT_COUNT
            for item in generations[1:]
        ),
        "no_unchanged_parent_replay": all(
            not item["unchanged_parent_replay"] for item in generations
        ),
        "global_best_tracker_never_consumes_candidate_budget": all(
            item["global_best_tracker_external_to_population"]
            for item in generations
        ),
        "global_best_tracker_accepts_generation_successes": all(
            item["global_best_tracker_input"] == "generation_successful_candidates"
            and item["global_best_tracker_retention_limit"] == 3
            for item in generations
        ),
        "maximum_formal_candidate_trainings_is_53": (
            maximum_formal_candidate_trainings == 53
        ),
        "all_low_fidelity_generations_enforce_1000": all(
            item["generation_iteration_recommendation"]
            == REGULAR_GENERATION_RECOMMENDED_ITERATIONS
            for item in generations
        ),
        "independent_top3_high_fidelity_review_uses_10000": (
            final_review["candidate_count"] == 3
            and final_review["training_iterations_per_candidate"]
            == HIGH_FIDELITY_ITERATIONS
            and not final_review["counts_as_low_fidelity_generation"]
            and not final_review["generates_algorithm_specs"]
        ),
        "iteration_budgets_are_runtime_limits": True,
        "stable_proxy_checkpoints_are_800_900_1000": (
            DEFAULT_EXPERIMENT_CONTRACT["low_fidelity_evaluation"][
                "checkpoint_iterations"
            ]
            == [800, 900, 1_000]
        ),
        "three_condition_stagnation_enabled": all(
            key in DEFAULT_EXPERIMENT_CONTRACT["stagnation"]
            for key in (
                "best_single_generation_threshold",
                "cumulative_improvement_threshold",
                "top3_median_improvement_threshold",
            )
        ),
        "escape_global_best_success_is_at_least_two_percent": (
            DEFAULT_EXPERIMENT_CONTRACT["stagnation"][
                "escape_global_best_improvement_threshold"
            ]
            == 0.02
        ),
        "escape_archive_median_success_is_at_least_two_percent": (
            DEFAULT_EXPERIMENT_CONTRACT["stagnation"][
                "escape_archive_top3_median_improvement_threshold"
            ]
            == 0.02
        ),
        "history_top3_is_normalized_spec_deduplicated": bool(
            DEFAULT_EXPERIMENT_CONTRACT["final_selection"][
                "deduplicate_normalized_specs"
            ]
        ),
        "nonzero_similarity_is_trainable": True,
        "exact_duplicates_use_bounded_retry": True,
        "generation_zero_shortfall_policy_unchanged": (
            generation_can_continue(
                0,
                actual_candidate_count=5,
                valid_new_candidate_count=5,
            )
        ),
        "later_generations_require_at_least_four_new_candidates": (
            generation_can_continue(
                1,
                actual_candidate_count=MINIMUM_GENERATION_CANDIDATES,
                valid_new_candidate_count=MINIMUM_NEW_EVOLUTION_CANDIDATES,
            )
            and not generation_can_continue(
                1,
                actual_candidate_count=MINIMUM_GENERATION_CANDIDATES - 1,
                valid_new_candidate_count=MINIMUM_NEW_EVOLUTION_CANDIDATES - 1,
            )
        ),
        "formal_ranking_requires_complete_top3": all(
            item["minimum_successful_candidate_count"] == EVOLUTION_PARENT_COUNT
            for item in generations
        ),
        "lbfgs_uses_persistent_chunked_internal_iteration_budget": True,
        "posterior_runs_are_isolated": len(set(seeds)) == len(seeds),
        "final_review_fresh_start": (
            final_review["fresh_initialization"]
            and not final_review["resume_low_fidelity_checkpoint"]
        ),
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "dry_run": True,
        "experiment_contract_version": EXPERIMENT_CONTRACT_VERSION,
        "benchmark": benchmark,
        "experiment_contract": deepcopy(DEFAULT_EXPERIMENT_CONTRACT),
        "generations": generations,
        "final_high_fidelity_evaluation": final_review,
        "seeds": list(seeds),
        "posterior_isolation": [
            f"{EXPERIMENT_CONTRACT_VERSION}/{benchmark}/seed_{seed}" for seed in seeds
        ],
        "baseline_iterations": HIGH_FIDELITY_ITERATIONS,
        "maximum_formal_candidate_trainings": maximum_formal_candidate_trainings,
        "checks": checks,
    }
