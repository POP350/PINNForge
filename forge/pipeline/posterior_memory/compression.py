"""Deterministic prompt-budget compression for posterior summaries."""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any


def estimate_context_tokens(value: Any) -> int:
    """Use a conservative UTF-8/JSON approximation without a tokenizer dependency."""

    serialized = json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
    return max(1, (len(serialized) + 3) // 4)


def compress_prompt_context(context: dict[str, Any], max_tokens: int) -> dict[str, Any]:
    """Keep newest formal evidence first and remove only lower-priority detail."""

    budget = max(256, int(max_tokens))
    compressed = deepcopy(context)
    original_tokens = estimate_context_tokens(compressed)
    changed = False

    rules = list(compressed.get("active_run_posterior_rules") or [])
    rules.sort(
        key=lambda item: (
            item.get("rule_type") == "failure",
            float(item.get("confidence") or 0.0),
            int(item.get("support_count") or 0),
        ),
        reverse=True,
    )
    compressed["active_run_posterior_rules"] = rules[:24]

    if estimate_context_tokens(compressed) > budget:
        compressed["active_run_posterior_rules"] = rules[:12]
        changed = True
    if estimate_context_tokens(compressed) > budget:
        compressed["previous_generation_failures"] = list(
            compressed.get("previous_generation_failures") or []
        )[:4]
        changed = True
    if estimate_context_tokens(compressed) > budget:
        compressed["previous_generation_candidates"] = [
            _compact_candidate(item, keep_spec=False)
            for item in (compressed.get("previous_generation_candidates") or [])
        ]
        changed = True
    if estimate_context_tokens(compressed) > budget:
        compressed["active_run_posterior_rules"] = rules[:6]
        latest = dict(compressed.get("latest_generation_posterior") or {})
        latest["training_observations"] = list(latest.get("training_observations") or [])[:8]
        latest["successful_patterns"] = list(latest.get("successful_patterns") or [])[:8]
        latest["failure_patterns"] = list(latest.get("failure_patterns") or [])[:8]
        compressed["latest_generation_posterior"] = latest
        changed = True
    if estimate_context_tokens(compressed) > budget:
        # Full executable Top-3 specs remain, but verbose generation prose is compacted.
        compressed["previous_top3"] = [
            _compact_candidate(item, keep_spec=True) for item in compressed.get("previous_top3") or []
        ]
        changed = True

    final_tokens = estimate_context_tokens(compressed)
    metadata = dict(compressed.get("metadata") or {})
    metadata.update(
        {
            "token_estimate": final_tokens,
            "max_tokens": budget,
            "compressed": changed,
            "original_token_estimate": original_tokens,
            "budget_overflow": final_tokens > budget,
        }
    )
    compressed["metadata"] = metadata
    return compressed


def _compact_candidate(candidate: dict[str, Any], *, keep_spec: bool) -> dict[str, Any]:
    result = {
        "candidate_id": candidate.get("candidate_id"),
        "rank": candidate.get("rank"),
        "final_metrics": candidate.get("final_metrics") or {},
        "training_dynamics": candidate.get("training_dynamics") or {},
        "failure_flags": candidate.get("failure_flags") or [],
        "observed_facts": list(candidate.get("observed_facts") or [])[:6],
        "recommended_adjustments": list(candidate.get("recommended_adjustments") or [])[:4],
    }
    if keep_spec:
        spec = deepcopy(candidate.get("algorithm_spec") or {})
        metadata = dict(spec.get("generation_metadata") or {})
        for key in (
            "expected_advantages",
            "possible_risks",
            "principle_tradeoffs",
            "role_requirements_applied",
        ):
            if key in metadata:
                metadata[key] = list(metadata.get(key) or [])[:3]
        spec["generation_metadata"] = metadata
        result["algorithm_spec"] = spec
    return result
