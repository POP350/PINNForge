"""Auditable paper-aligned ablation modes for PINNForge experiments."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


FULL_ABLATION_MODE = "full"
NO_KNOWLEDGE_ABLATION_MODE = "no_knowledge"
NO_EXECUTION_FEEDBACK_ABLATION_MODE = "no_execution_feedback"
NO_EVOLUTIONARY_SEARCH_ABLATION_MODE = "no_evolutionary_search"

# Compatibility name used by older callers and artifacts. Its value is
# canonical so every new experiment contract uses the terminology in the paper.
STANDARD_ABLATION_MODE = FULL_ABLATION_MODE
ABLATION_MODES = (
    FULL_ABLATION_MODE,
    NO_KNOWLEDGE_ABLATION_MODE,
    NO_EXECUTION_FEEDBACK_ABLATION_MODE,
    NO_EVOLUTIONARY_SEARCH_ABLATION_MODE,
)

_ALIASES = {
    "none": FULL_ABLATION_MODE,
    "standard": FULL_ABLATION_MODE,
    "without_knowledge": NO_KNOWLEDGE_ABLATION_MODE,
    "without_knowledge_base": NO_KNOWLEDGE_ABLATION_MODE,
    "no_kb": NO_KNOWLEDGE_ABLATION_MODE,
    "without_execution_feedback": NO_EXECUTION_FEEDBACK_ABLATION_MODE,
    "no_feedback": NO_EXECUTION_FEEDBACK_ABLATION_MODE,
    "without_evolutionary_search": NO_EVOLUTIONARY_SEARCH_ABLATION_MODE,
    "no_evolution": NO_EVOLUTIONARY_SEARCH_ABLATION_MODE,
}


def normalize_ablation_mode(value: Any) -> str:
    """Return one canonical mode and reject silent experimental drift."""

    normalized = str(value or FULL_ABLATION_MODE).strip().casefold().replace("-", "_")
    normalized = _ALIASES.get(normalized, normalized)
    if normalized not in ABLATION_MODES:
        raise ValueError(
            "ablation_mode must be one of: " + ", ".join(ABLATION_MODES)
        )
    return normalized


def ablation_contract(mode: Any) -> dict[str, Any]:
    """Describe exactly what changes relative to the standard experiment."""

    resolved = normalize_ablation_mode(mode)
    contracts: dict[str, dict[str, Any]] = {
        FULL_ABLATION_MODE: {
            "description": "Full multi-fidelity evolution with prior and run-scoped posterior knowledge.",
            "search_strategy": "adaptive_multifidelity_evolution",
            "prior_knowledge_enabled": True,
            "posterior_knowledge_enabled": True,
            "generation_feedback_enabled": True,
            "execution_feedback_enabled": True,
            "multi_generation_search_enabled": True,
            "multi_generation_evolution_enabled": True,
            "multi_fidelity_search_enabled": True,
            "evolutionary_inheritance_enabled": True,
            "parent_algorithm_specs_exposed_to_llm": True,
            "global_archive_enabled": True,
            "final_evaluation": "global_low_fidelity_top3_x_10000_fresh_start",
            "direct_candidate_count": None,
            "training_iterations_per_direct_candidate": None,
        },
        NO_KNOWLEDGE_ABLATION_MODE: {
            "description": "The same multi-fidelity evolution with all prior and posterior knowledge-base retrieval and injection disabled.",
            "search_strategy": "adaptive_multifidelity_evolution",
            "prior_knowledge_enabled": False,
            "posterior_knowledge_enabled": False,
            # Ranked parents and measured generation feedback are intrinsic search
            # state, not retrieved prior/posterior knowledge-base entries.
            "generation_feedback_enabled": True,
            "execution_feedback_enabled": True,
            "multi_generation_search_enabled": True,
            "multi_generation_evolution_enabled": True,
            "multi_fidelity_search_enabled": True,
            "evolutionary_inheritance_enabled": True,
            "parent_algorithm_specs_exposed_to_llm": True,
            "global_archive_enabled": True,
            "final_evaluation": "global_low_fidelity_top3_x_10000_fresh_start",
            "direct_candidate_count": None,
            "training_iterations_per_direct_candidate": None,
        },
        NO_EXECUTION_FEEDBACK_ABLATION_MODE: {
            "description": "Keep knowledge guidance and the complete multi-fidelity evolutionary controller, but hide current execution outcomes from all later LLM calls.",
            "search_strategy": "adaptive_multifidelity_evolution",
            "prior_knowledge_enabled": True,
            # The knowledge subsystem remains configured, but current-run
            # posterior writes/injection are disabled below because they are
            # execution-derived feedback.
            "posterior_knowledge_enabled": True,
            "generation_feedback_enabled": False,
            "execution_feedback_enabled": False,
            "multi_generation_search_enabled": True,
            "multi_generation_evolution_enabled": True,
            "multi_fidelity_search_enabled": True,
            "evolutionary_inheritance_enabled": True,
            "parent_algorithm_specs_exposed_to_llm": True,
            "global_archive_enabled": True,
            "final_evaluation": "global_low_fidelity_top3_x_10000_fresh_start",
            "direct_candidate_count": None,
            "training_iterations_per_direct_candidate": None,
        },
        NO_EVOLUTIONARY_SEARCH_ABLATION_MODE: {
            "description": "Budget-matched independent LLM search with the same knowledge, execution feedback, posterior memory, low-fidelity candidate count, global archive, and high-fidelity Top-3 protocol, but no parent AlgorithmSpec inheritance or recombination.",
            "search_strategy": "budget_matched_independent_llm_search",
            "prior_knowledge_enabled": True,
            "posterior_knowledge_enabled": True,
            "generation_feedback_enabled": True,
            "execution_feedback_enabled": True,
            "multi_generation_search_enabled": True,
            "multi_generation_evolution_enabled": False,
            "multi_fidelity_search_enabled": True,
            "evolutionary_inheritance_enabled": False,
            "parent_algorithm_specs_exposed_to_llm": False,
            "global_archive_enabled": True,
            "final_evaluation": "global_low_fidelity_top3_x_10000_fresh_start",
            "direct_candidate_count": None,
            "training_iterations_per_direct_candidate": None,
        },
    }
    return {"mode": resolved, **deepcopy(contracts[resolved])}


def _explicit_execution_feedback(config: dict[str, Any] | None) -> bool | None:
    """Read the legacy nested feedback override when it is explicitly present."""

    raw = (config or {}).get("ablation") or {}
    if not isinstance(raw, dict):
        raise TypeError("ablation must be an object")
    if "execution_feedback" not in raw:
        return None
    value = raw["execution_feedback"]
    if not isinstance(value, bool):
        raise TypeError("ablation.execution_feedback must be a boolean")
    return value


def resolve_execution_feedback(config: dict[str, Any] | None) -> bool:
    """Return whether execution-derived evidence may be exposed to the LLM."""

    return bool(resolved_ablation_contract(config)["execution_feedback_enabled"])


def resolved_ablation_contract(config: dict[str, Any] | None) -> dict[str, Any]:
    """Return the persisted contract including the execution-feedback flag."""

    config = config or {}
    mode = normalize_ablation_mode(config.get("ablation_mode"))
    explicit_execution_feedback = _explicit_execution_feedback(config)

    # Backward compatibility: the former interface represented this paper
    # ablation as `ablation_mode=standard` plus a separate false flag.
    if explicit_execution_feedback is False and mode == FULL_ABLATION_MODE:
        mode = NO_EXECUTION_FEEDBACK_ABLATION_MODE

    contract = ablation_contract(mode)
    if (
        explicit_execution_feedback is not None
        and explicit_execution_feedback
        != bool(contract["execution_feedback_enabled"])
    ):
        raise ValueError(
            "ablation.execution_feedback conflicts with ablation_mode="
            f"{mode}; select one paper-aligned ablation mode instead"
        )

    execution_feedback = bool(contract["execution_feedback_enabled"])
    contract["reflection_to_llm_enabled"] = execution_feedback
    contract["current_run_posterior_updates_enabled"] = bool(
        execution_feedback and contract["posterior_knowledge_enabled"]
    )
    contract["true_fitness_ranking_enabled"] = True
    contract["true_fitness_parent_selection_enabled"] = bool(
        contract["evolutionary_inheritance_enabled"]
    )
    return contract


__all__ = [
    "ABLATION_MODES",
    "FULL_ABLATION_MODE",
    "NO_EXECUTION_FEEDBACK_ABLATION_MODE",
    "NO_EVOLUTIONARY_SEARCH_ABLATION_MODE",
    "NO_KNOWLEDGE_ABLATION_MODE",
    "STANDARD_ABLATION_MODE",
    "ablation_contract",
    "normalize_ablation_mode",
    "resolve_execution_feedback",
    "resolved_ablation_contract",
]
