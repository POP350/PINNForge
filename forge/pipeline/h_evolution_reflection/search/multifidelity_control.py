"""Pure control algorithms for stagnation-aware multi-fidelity search."""

from __future__ import annotations

from copy import deepcopy
import math
from statistics import median
from typing import Any, Literal

from forge.pipeline.h_evolution_reflection.search.global_best_tracker import (
    GlobalBestTracker,
)

SearchPhase = Literal[
    "normal_search",
    "escape_scheduled",
    "escape_running",
    "finalist_selection",
    "high_fidelity_running",
    "completed",
]

_ALLOWED_PHASE_TRANSITIONS: dict[str, set[str]] = {
    "normal_search": {
        "normal_search",
        "escape_scheduled",
        "finalist_selection",
    },
    "escape_scheduled": {"escape_running"},
    "escape_running": {"normal_search", "finalist_selection"},
    "finalist_selection": {"high_fidelity_running"},
    "high_fidelity_running": {"completed"},
    "completed": set(),
}


def finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def nonnegative_relative_improvement(previous: Any, current: Any) -> float:
    """Return a finite non-negative relative improvement with safe zero handling."""

    left = finite_number(previous)
    right = finite_number(current)
    if left is None or right is None or left == 0.0:
        return 0.0
    return max(0.0, (left - right) / abs(left))


def initial_search_state() -> dict[str, Any]:
    return {
        "search_phase": "normal_search",
        "next_generation_mode": "normal",
        "low_fidelity_generation_count": 0,
        "current_generation_id": None,
        "current_generation_mode": "normal",
        "global_best_history": [],
        "top3_median_history": [],
        "best_improvement_history": [],
        "top3_improvement_history": [],
        "valid_candidate_count_history": [],
        "generation_best_history": [],
        "stagnation_window_baseline_index": 0,
        "stagnation_window_reset_generation": None,
        "generations_since_last_escape": 0,
        "escape_pending": False,
        "escape_active": False,
        "pre_escape_global_best_mse": None,
        "escape_source_generation_id": None,
        "escape_trigger_reasons": [],
        "escape_success_reasons": [],
        "global_top3_candidate_ids": [],
        "global_top3_spec_keys": [],
        "global_top3_source_generations": [],
        "global_top3_median_mse": None,
        "global_top3_changed": False,
        "new_global_top3_entry_count": 0,
        "archive_top3_median_improvement": 0.0,
        "generations_since_global_top3_update": 0,
        "global_best_stagnation": False,
        "global_best_exact_plateau": False,
        "population_stagnation": False,
        "elite_archive_stagnation": False,
        "consecutive_invalid_generation_count": 0,
        "final_high_fidelity_pending": False,
        "final_high_fidelity_candidate_ids": [],
        "completed_final_high_fidelity_candidate_ids": [],
        "termination_reason": None,
    }


def validate_search_state(state: dict[str, Any]) -> None:
    """Reject illegal flag combinations before persistence or execution."""

    phase = str(state.get("search_phase") or "")
    if phase not in _ALLOWED_PHASE_TRANSITIONS:
        raise ValueError(f"Unknown search phase: {phase!r}")
    pending = bool(state.get("escape_pending"))
    active = bool(state.get("escape_active"))
    if pending and active:
        raise ValueError("escape_pending and escape_active cannot both be true")
    if pending != (phase == "escape_scheduled"):
        raise ValueError(
            "escape_pending must be true exactly in escape_scheduled phase"
        )
    if active != (phase == "escape_running"):
        raise ValueError(
            "escape_active must be true exactly in escape_running phase"
        )
    if phase in {"finalist_selection", "high_fidelity_running", "completed"}:
        if pending or active:
            raise ValueError("Final or completed phases cannot schedule/run escape")
    if state.get("termination_reason") and phase not in {
        "finalist_selection",
        "high_fidelity_running",
        "completed",
    }:
        raise ValueError(
            "A terminated low-fidelity search must be in a final phase"
        )


def transition_search_phase(
    state: dict[str, Any], target_phase: SearchPhase
) -> dict[str, Any]:
    """Apply one centrally defined state-machine transition."""

    validate_search_state(state)
    source = str(state["search_phase"])
    if target_phase not in _ALLOWED_PHASE_TRANSITIONS[source]:
        raise ValueError(
            f"Illegal search phase transition: {source} -> {target_phase}"
        )
    updated = deepcopy(state)
    updated["search_phase"] = target_phase
    updated["escape_pending"] = target_phase == "escape_scheduled"
    updated["escape_active"] = target_phase == "escape_running"
    if target_phase == "escape_scheduled":
        updated["next_generation_mode"] = "escape"
    elif target_phase == "normal_search":
        updated["next_generation_mode"] = "normal"
    elif target_phase in {
        "finalist_selection",
        "high_fidelity_running",
        "completed",
    }:
        updated["next_generation_mode"] = None
    updated["final_high_fidelity_pending"] = (
        target_phase == "high_fidelity_running"
    )
    validate_search_state(updated)
    return updated


def prepare_next_generation(
    state: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    """Consume a scheduled escape atomically, including after resume."""

    validate_search_state(state)
    phase = str(state["search_phase"])
    if phase == "escape_scheduled":
        updated = transition_search_phase(state, "escape_running")
        mode = "escape"
    elif phase == "normal_search":
        updated = transition_search_phase(state, "normal_search")
        mode = "normal"
    else:
        raise RuntimeError(
            f"Cannot create a low-fidelity generation in phase {phase}"
        )
    updated["current_generation_mode"] = mode
    return updated, mode


def summarize_global_top3_archive(
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
) -> dict[str, Any]:
    """Compare bounded tracker snapshots using normalized Spec identity."""

    previous = dict(before or {})
    current = dict(after or {})
    previous_keys = [
        str(item) for item in previous.get("global_top3_spec_keys") or []
    ]
    current_keys = [
        str(item) for item in current.get("global_top3_spec_keys") or []
    ]
    new_keys = [item for item in current_keys if item not in previous_keys]
    previous_median = finite_number(previous.get("global_top3_median_mse"))
    current_median = finite_number(current.get("global_top3_median_mse"))
    return {
        "global_top3_candidate_ids": list(
            current.get("global_top3_candidate_ids") or []
        ),
        "global_top3_spec_keys": current_keys,
        "global_top3_source_generations": list(
            current.get("global_top3_source_generations") or []
        ),
        "global_top3_median_mse": current_median,
        "new_global_top3_entry_count": len(new_keys),
        "new_global_top3_spec_keys": new_keys,
        "global_top3_changed": previous_keys != current_keys,
        "archive_top3_median_improvement": nonnegative_relative_improvement(
            previous_median, current_median
        ),
    }


def summarize_low_fidelity_generation(
    records: list[dict[str, Any]],
    *,
    previous_state: dict[str, Any],
    generation_id: int,
    generation_mode: str,
    contract: dict[str, Any],
    archive_update: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute stable generation metrics and the complete stagnation audit."""

    values: list[float] = []
    for record in records:
        value = finite_number(record.get("low_fidelity_proxy_mse"))
        if record.get("training_success") and value is not None:
            values.append(value)
    values.sort()
    generation_best = values[0] if values else None
    top_values = values[:3]
    generation_top3_median = median(top_values) if top_values else None

    state = deepcopy(previous_state or initial_search_state())
    prior_global = (
        state["global_best_history"][-1]
        if state.get("global_best_history")
        else None
    )
    prior_top3 = (
        state["top3_median_history"][-1]
        if state.get("top3_median_history")
        else None
    )
    if prior_global is None:
        global_best = generation_best
    elif generation_best is None:
        global_best = prior_global
    else:
        global_best = min(float(prior_global), float(generation_best))
    best_improvement = nonnegative_relative_improvement(prior_global, global_best)
    top3_improvement = nonnegative_relative_improvement(
        prior_top3, generation_top3_median
    )

    state.setdefault("generation_best_history", []).append(generation_best)
    state.setdefault("global_best_history", []).append(global_best)
    state.setdefault("top3_median_history", []).append(generation_top3_median)
    state.setdefault("best_improvement_history", []).append(best_improvement)
    state.setdefault("top3_improvement_history", []).append(top3_improvement)
    state.setdefault("valid_candidate_count_history", []).append(len(values))
    minimum_valid = int(
        contract["search_control"][
            "minimum_valid_candidates_per_generation"
        ]
    )
    if len(values) < minimum_valid:
        state["consecutive_invalid_generation_count"] = int(
            state.get("consecutive_invalid_generation_count") or 0
        ) + 1
    else:
        state["consecutive_invalid_generation_count"] = 0
    archive = dict(archive_update or {})
    new_archive_entries = int(
        archive.get("new_global_top3_entry_count") or 0
    )
    for key in (
        "global_top3_candidate_ids",
        "global_top3_spec_keys",
        "global_top3_source_generations",
    ):
        if key in archive:
            state[key] = list(archive.get(key) or [])
    for key in (
        "global_top3_median_mse",
        "new_global_top3_entry_count",
        "global_top3_changed",
        "archive_top3_median_improvement",
    ):
        if key in archive:
            state[key] = archive.get(key)
    if archive_update is not None:
        if (
            state.get("low_fidelity_generation_count", 0) == 0
            or new_archive_entries > 0
        ):
            state["generations_since_global_top3_update"] = 0
        else:
            state["generations_since_global_top3_update"] = int(
                state.get("generations_since_global_top3_update") or 0
            ) + 1
    state["low_fidelity_generation_count"] = int(
        state.get("low_fidelity_generation_count") or 0
    ) + 1
    state["current_generation_id"] = int(generation_id)
    state["current_generation_mode"] = str(generation_mode)
    reset_generation = state.get("stagnation_window_reset_generation")
    if reset_generation is not None:
        state["generations_since_last_escape"] = max(
            0, int(generation_id) - int(reset_generation)
        )
    if state["low_fidelity_generation_count"] == 1:
        state["stagnation_window_baseline_index"] = 0

    stagnation_check = evaluate_stagnation(state, contract=contract)
    for key in (
        "global_best_stagnation",
        "global_best_exact_plateau",
        "population_stagnation",
        "elite_archive_stagnation",
    ):
        state[key] = bool(stagnation_check.get(key))
    summary = {
        "generation_id": int(generation_id),
        "generation_mode": str(generation_mode),
        "valid_candidate_count": len(values),
        "generation_best_mse": generation_best,
        "generation_top3_median_mse": generation_top3_median,
        "global_best_mse": global_best,
        "global_best_improvement": best_improvement,
        "top3_median_improvement": top3_improvement,
        "minimum_valid_candidates_per_generation": minimum_valid,
        "consecutive_invalid_generation_count": state[
            "consecutive_invalid_generation_count"
        ],
        **{
            key: deepcopy(state.get(key))
            for key in (
                "global_top3_changed",
                "new_global_top3_entry_count",
                "global_top3_candidate_ids",
                "global_top3_spec_keys",
                "global_top3_source_generations",
                "global_top3_median_mse",
                "archive_top3_median_improvement",
                "generations_since_global_top3_update",
            )
        },
        "stagnation_check": stagnation_check,
        **{
            key: bool(stagnation_check.get(key))
            for key in (
                "global_best_stagnation",
                "global_best_exact_plateau",
                "population_stagnation",
                "elite_archive_stagnation",
            )
        },
        "escape_triggered": False,
        "escape_success": None,
        "escape_success_reasons": [],
    }
    return state, summary


def evaluate_stagnation(
    state: dict[str, Any], *, contract: dict[str, Any]
) -> dict[str, Any]:
    config = contract["stagnation"]
    control = contract["search_control"]
    best_window = int(config["best_improvement_window"])
    top_window = int(config["top3_median_window"])
    count = int(state.get("low_fidelity_generation_count") or 0)
    baseline = int(state.get("stagnation_window_baseline_index") or 0)
    improvements_available = max(0, len(state.get("best_improvement_history") or []) - 1 - baseline)
    enough_generations = count >= int(control["min_low_fidelity_generations"])
    enough_best_window = improvements_available >= best_window
    top_improvements_available = max(
        0,
        len(state.get("top3_improvement_history") or []) - 1 - baseline,
    )
    enough_top3_window = top_improvements_available >= top_window

    best_values = list(state.get("best_improvement_history") or [])[
        baseline + 1 :
    ][-best_window:]
    top_values = list(state.get("top3_improvement_history") or [])[
        baseline + 1 :
    ][-top_window:]
    valid_counts = list(state.get("valid_candidate_count_history") or [])
    required_valid_window = max(best_window + 1, top_window + 1)
    window_counts = valid_counts[baseline:][-required_valid_window:]
    valid_top3_window = (
        len(window_counts) >= required_valid_window
        and all(int(value) >= 3 for value in window_counts)
    )
    three_small = (
        len(best_values) == best_window
        and all(
            finite_number(value) is not None
            and float(value) < float(config["best_single_generation_threshold"])
            for value in best_values
        )
    )
    two_small_top3 = (
        len(top_values) == top_window
        and all(
            finite_number(value) is not None
            and float(value) < float(config["top3_median_improvement_threshold"])
            for value in top_values
        )
    )
    no_global_best_progress = (
        len(best_values) == best_window
        and all(
            finite_number(value) is not None
            and float(value) <= 1.0e-12
            for value in best_values
        )
    )

    global_history = list(state.get("global_best_history") or [])
    current_global = global_history[-1] if global_history else None
    before_index = len(global_history) - best_window - 1
    global_before = global_history[before_index] if before_index >= baseline else None
    cumulative = nonnegative_relative_improvement(global_before, current_global)
    small_cumulative = (
        global_before is not None
        and cumulative < float(config["cumulative_improvement_threshold"])
    )
    enabled = bool(config.get("enabled", True))
    global_best_stagnation = bool(
        enabled
        and enough_generations
        and enough_best_window
        and three_small
        and small_cumulative
    )
    global_best_exact_plateau = bool(
        enabled
        and enough_generations
        and enough_best_window
        and no_global_best_progress
    )
    population_stagnation = bool(
        enabled
        and enough_generations
        and enough_top3_window
        and valid_top3_window
        and two_small_top3
    )
    elite_archive_stagnation = bool(
        enabled
        and enough_generations
        and int(state.get("generations_since_global_top3_update") or 0)
        >= int(config["elite_archive_stagnation_window"])
    )
    detected = bool(
        global_best_stagnation
        or global_best_exact_plateau
        or elite_archive_stagnation
        or population_stagnation
    )
    return {
        "enabled": enabled,
        "eligible": bool(
            enabled
            and enough_generations
            and (
                enough_best_window
                or enough_top3_window
                or elite_archive_stagnation
            )
        ),
        "enough_generations": enough_generations,
        "enough_complete_window": enough_best_window,
        "enough_best_window": enough_best_window,
        "enough_top3_window": enough_top3_window,
        "valid_top3_window": valid_top3_window,
        "best_improvements": best_values,
        "best_single_generation_threshold": float(
            config["best_single_generation_threshold"]
        ),
        "three_small_best_improvements": three_small,
        "global_best_before_window": global_before,
        "current_global_best": current_global,
        "cumulative_improvement": cumulative,
        "cumulative_improvement_threshold": float(
            config["cumulative_improvement_threshold"]
        ),
        "small_cumulative_improvement": small_cumulative,
        "top3_median_improvements": top_values,
        "top3_median_improvement_threshold": float(
            config["top3_median_improvement_threshold"]
        ),
        "two_small_top3_improvements": two_small_top3,
        "no_global_best_progress": no_global_best_progress,
        "global_best_stagnation": global_best_stagnation,
        "global_best_exact_plateau": global_best_exact_plateau,
        "population_stagnation": population_stagnation,
        "elite_archive_stagnation": elite_archive_stagnation,
        "elite_archive_stagnation_window": int(
            config["elite_archive_stagnation_window"]
        ),
        "generations_since_global_top3_update": int(
            state.get("generations_since_global_top3_update") or 0
        ),
        "population_stagnation_confirmation": population_stagnation,
        "stagnation_detected": detected,
    }


def evaluate_escape(
    *,
    pre_escape_global_best_mse: Any,
    escape_generation_best_mse: Any,
    new_global_top3_entry_count: int = 0,
    archive_top3_median_improvement: Any = 0.0,
    contract: dict[str, Any],
) -> dict[str, Any]:
    improvement = nonnegative_relative_improvement(
        pre_escape_global_best_mse, escape_generation_best_mse
    )
    config = contract["stagnation"]
    global_threshold = float(
        config["escape_global_best_improvement_threshold"]
    )
    archive_threshold = float(
        config["escape_archive_top3_median_improvement_threshold"]
    )
    minimum_entries = int(config["escape_min_new_archive_entries"])
    archive_improvement = finite_number(
        archive_top3_median_improvement
    ) or 0.0
    checks = {
        "global_best_improved": (
            improvement + 1.0e-12 >= global_threshold
        ),
        "new_global_top3_entries": (
            int(new_global_top3_entry_count) >= minimum_entries
        ),
        "archive_top3_median_improved": (
            archive_improvement + 1.0e-12 >= archive_threshold
        ),
    }
    reasons = [key for key, passed in checks.items() if passed]
    success = bool(reasons)
    return {
        "pre_escape_global_best_mse": finite_number(pre_escape_global_best_mse),
        "escape_generation_best_mse": finite_number(escape_generation_best_mse),
        "escape_improvement": improvement,
        "escape_global_best_improvement_threshold": global_threshold,
        "new_global_top3_entry_count": int(new_global_top3_entry_count),
        "escape_min_new_archive_entries": minimum_entries,
        "archive_top3_median_improvement": archive_improvement,
        "escape_archive_top3_median_improvement_threshold": archive_threshold,
        "comparison": "greater_than_or_equal",
        "escape_success_checks": checks,
        "escape_success_reasons": reasons,
        "escape_success": bool(success),
    }


def reset_stagnation_window_after_escape(
    state: dict[str, Any], *, generation_id: int
) -> dict[str, Any]:
    updated = deepcopy(state)
    history = list(updated.get("global_best_history") or [])
    updated["stagnation_window_baseline_index"] = max(0, len(history) - 1)
    updated["stagnation_window_reset_generation"] = int(generation_id)
    updated["generations_since_last_escape"] = 0
    updated["pre_escape_global_best_mse"] = None
    updated["escape_source_generation_id"] = None
    updated["escape_trigger_reasons"] = []
    updated["escape_success_reasons"] = []
    updated["generations_since_global_top3_update"] = 0
    for key in (
        "global_best_stagnation",
        "global_best_exact_plateau",
        "population_stagnation",
        "elite_archive_stagnation",
    ):
        updated[key] = False
    updated = transition_search_phase(updated, "normal_search")
    return updated


def advance_search_after_generation(
    state: dict[str, Any],
    generation_summary: dict[str, Any],
    *,
    generation_id: int,
    generation_mode: str,
    contract: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], str | None]:
    """Apply escape scheduling/success and the absolute low-fidelity cap."""

    if generation_mode not in {"normal", "escape"}:
        raise ValueError("generation_mode must be normal or escape")
    updated = deepcopy(state)
    summary = deepcopy(generation_summary)
    maximum = int(
        contract["search_control"]["max_low_fidelity_generations"]
    )
    termination_reason: str | None = None
    valid_count = int(summary.get("valid_candidate_count") or 0)
    minimum_valid = int(
        contract["search_control"][
            "minimum_valid_candidates_per_generation"
        ]
    )
    maximum_invalid = int(
        contract["search_control"]["max_consecutive_invalid_generations"]
    )
    repeated_failure = bool(
        summary.get("force_repeated_generation_failure")
        or (
            valid_count < minimum_valid
            and int(
                updated.get("consecutive_invalid_generation_count") or 0
            )
            >= maximum_invalid
        )
    )
    if generation_mode == "escape":
        if str(updated.get("search_phase")) != "escape_running":
            raise ValueError("Escape evaluation requires escape_running phase")
        escape_audit = evaluate_escape(
            pre_escape_global_best_mse=updated.get(
                "pre_escape_global_best_mse"
            ),
            escape_generation_best_mse=summary.get("generation_best_mse"),
            new_global_top3_entry_count=int(
                summary.get("new_global_top3_entry_count") or 0
            ),
            archive_top3_median_improvement=summary.get(
                "archive_top3_median_improvement"
            ),
            contract=contract,
        )
        summary.update(escape_audit)
        summary["escape_triggered"] = True
        updated["escape_success_reasons"] = list(
            escape_audit["escape_success_reasons"]
        )
        if repeated_failure:
            termination_reason = "repeated_generation_failure"
        elif int(updated["low_fidelity_generation_count"]) >= maximum:
            termination_reason = (
                "max_generations_reached_after_successful_escape"
                if escape_audit["escape_success"]
                else "escape_failed"
            )
        elif escape_audit["escape_success"]:
            updated = reset_stagnation_window_after_escape(
                updated, generation_id=generation_id
            )
        else:
            trigger_reasons = set(updated.get("escape_trigger_reasons") or [])
            termination_reason = (
                "elite_archive_stagnation_escape_failed"
                if "elite_archive_stagnation" in trigger_reasons
                else "escape_failed"
            )
    elif int(updated["low_fidelity_generation_count"]) >= maximum:
        termination_reason = "max_low_fidelity_generations_reached"
    elif repeated_failure:
        termination_reason = "repeated_generation_failure"
    elif bool(
        (summary.get("stagnation_check") or {}).get("stagnation_detected")
    ):
        check = summary.get("stagnation_check") or {}
        reasons = [
            name
            for name in (
                "global_best_stagnation",
                "global_best_exact_plateau",
                "elite_archive_stagnation",
                "population_stagnation",
            )
            if bool(check.get(name))
        ]
        updated = transition_search_phase(updated, "escape_scheduled")
        updated["pre_escape_global_best_mse"] = summary.get(
            "global_best_mse"
        )
        updated["escape_source_generation_id"] = int(generation_id)
        updated["escape_trigger_reasons"] = reasons
        summary["escape_triggered"] = True
    if termination_reason is not None:
        updated = transition_search_phase(updated, "finalist_selection")
        updated["termination_reason"] = termination_reason
    summary["next_generation_mode"] = updated.get("next_generation_mode")
    summary["search_phase"] = updated.get("search_phase")
    summary["termination_reason"] = termination_reason
    validate_search_state(updated)
    return updated, summary, termination_reason


def select_distinct_history_top_candidates(
    records: list[dict[str, Any]],
    *,
    candidate_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select finalists from the global archive across all generations."""

    tracker = build_global_best_tracker(
        records,
        candidate_count=int(candidate_count),
    )
    selected, audit = tracker.top_k(candidate_count)
    audit["selection_source"] = "global_successful_candidate_archive_rebuild"
    return selected, audit


def build_global_best_tracker(
    records: list[dict[str, Any]],
    *,
    candidate_count: int,
) -> GlobalBestTracker:
    """Replay every successful candidate into the bounded global Top-K archive."""

    tracker = GlobalBestTracker(default_top_k=int(candidate_count))
    by_generation: dict[int, list[dict[str, Any]]] = {}
    for item in records:
        if not item.get("training_success"):
            continue
        generation = int(item.get("generation_index", item.get("generation")) or 0)
        by_generation.setdefault(generation, []).append(item)
    for generation in sorted(by_generation):
        tracker.update_many(by_generation[generation])
    return tracker


def build_formal_generation_top3_tracker(
    records: list[dict[str, Any]],
    *,
    candidate_count: int,
) -> GlobalBestTracker:
    """Backward-compatible alias for the global archive rebuild."""

    return build_global_best_tracker(records, candidate_count=candidate_count)


def formal_generation_top3_records(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Backward-compatible view of per-generation formal Top-3 records."""

    return [
        item
        for item in records
        if int(item.get("generation_rank") or 0) in {1, 2, 3}
    ]


def replay_low_fidelity_control(
    records: list[dict[str, Any]],
    *,
    contract: dict[str, Any],
) -> dict[str, Any]:
    """Replay recorded generations through the current controller semantics."""

    low_records = [
        item
        for item in records
        if str(item.get("fidelity") or "low").casefold() != "high"
    ]
    by_generation: dict[int, list[dict[str, Any]]] = {}
    for item in low_records:
        generation = int(
            item.get("generation_index", item.get("generation")) or 0
        )
        by_generation.setdefault(generation, []).append(item)

    candidate_count = int(contract["final_selection"]["candidate_count"])
    tracker = GlobalBestTracker(default_top_k=candidate_count)
    state = initial_search_state()
    events: list[dict[str, Any]] = []
    for generation in sorted(by_generation):
        if state["search_phase"] == "finalist_selection":
            break
        state, mode = prepare_next_generation(state)
        before = tracker.audit_snapshot()
        tracker.update_many(
            [
                item
                for item in by_generation[generation]
                if item.get("training_success")
            ]
        )
        archive_update = summarize_global_top3_archive(
            before, tracker.audit_snapshot()
        )
        state, summary = summarize_low_fidelity_generation(
            by_generation[generation],
            previous_state=state,
            generation_id=generation,
            generation_mode=mode,
            contract=contract,
            archive_update=archive_update,
        )
        state, summary, reason = advance_search_after_generation(
            state,
            summary,
            generation_id=generation,
            generation_mode=mode,
            contract=contract,
        )
        events.append(
            {
                "generation_id": generation,
                "generation_mode": mode,
                "search_phase_after_generation": state["search_phase"],
                "next_generation_mode": state.get("next_generation_mode"),
                "stagnation_reasons": [
                    key
                    for key in (
                        "global_best_stagnation",
                        "global_best_exact_plateau",
                        "elite_archive_stagnation",
                        "population_stagnation",
                    )
                    if summary.get(key)
                ],
                "global_top3_source_generations": list(
                    summary.get("global_top3_source_generations") or []
                ),
                "new_global_top3_entry_count": int(
                    summary.get("new_global_top3_entry_count") or 0
                ),
                "escape_success": summary.get("escape_success"),
                "escape_success_reasons": list(
                    summary.get("escape_success_reasons") or []
                ),
                "termination_reason": reason,
            }
        )
        if reason is not None:
            break
    return {
        "events": events,
        "final_state": state,
        "processed_generation_count": len(events),
        "available_generation_count": len(by_generation),
        "unused_recorded_generation_ids": sorted(by_generation)[len(events) :],
        "global_best_tracker": tracker.audit_snapshot(),
    }
