"""Build the bounded next-generation context from one run-scoped memory."""

from __future__ import annotations

from typing import Any

from forge.pipeline.posterior_memory.compression import compress_prompt_context
from forge.pipeline.posterior_memory.models import PosteriorPromptContext


def build_posterior_prompt_context(
    *,
    candidate_records: list[dict[str, Any]],
    generation_summaries: list[dict[str, Any]],
    active_rules: list[dict[str, Any]],
    max_tokens: int,
) -> dict[str, Any]:
    if not generation_summaries:
        return PosteriorPromptContext(
            metadata={"empty": True, "token_estimate": 0, "max_tokens": max_tokens}
        ).to_dict()
    latest = generation_summaries[-1]
    generation = int(latest.get("generation") or 0)
    previous = [item for item in candidate_records if int(item.get("generation") or 0) == generation]
    top3 = sorted(
        [item for item in previous if item.get("rank") in {1, 2, 3}],
        key=lambda item: int(item.get("rank") or 99),
    )
    failures = [item for item in previous if item.get("failure_flags") or not item.get("training_success")]
    context = PosteriorPromptContext(
        previous_top3=top3,
        latest_generation_posterior=latest,
        previous_generation_candidates=[_candidate_summary(item) for item in previous],
        previous_generation_failures=[_failure_summary(item) for item in failures],
        active_run_posterior_rules=active_rules,
        recommended_next_actions=list(latest.get("recommended_next_actions") or []),
        unresolved_hypotheses=list(latest.get("unresolved_hypotheses") or []),
        metadata={"empty": False, "source_generation": generation},
    ).to_dict()
    return compress_prompt_context(context, max_tokens)


def _candidate_summary(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": item.get("candidate_id"),
        "rank": item.get("rank"),
        "final_metrics": item.get("final_metrics") or {},
        "training_dynamics": item.get("training_dynamics") or {},
        "failure_flags": item.get("failure_flags") or [],
        "observed_facts": item.get("observed_facts") or [],
        "inferred_causes": item.get("inferred_causes") or [],
        "recommended_adjustments": item.get("recommended_adjustments") or [],
    }


def _failure_summary(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": item.get("candidate_id"),
        "status": item.get("status"),
        "failure_flags": item.get("failure_flags") or [],
        "failure_evidence": item.get("failure_evidence") or [],
        "observed_facts": item.get("observed_facts") or [],
        "recommended_adjustments": item.get("recommended_adjustments") or [],
    }
