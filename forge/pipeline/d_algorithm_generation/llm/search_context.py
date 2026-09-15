"""Soft context assembly for direct AlgorithmSpec search."""

from __future__ import annotations

from typing import Any

from forge.pipeline.c_knowledge_retrieval.knowledge_base.kb_manager import KnowledgeBaseManager
from forge.pipeline.c_knowledge_retrieval.knowledge_base.posterior_updater import PosteriorUpdater


def retrieve_prior_context(problem_features: dict, max_items: int = 12) -> list[dict[str, Any]]:
    """Return relevant prior snippets as non-binding guidance.

    This function intentionally does not define a closed search space. The LLM
    may use or ignore these snippets when generating the complete AlgorithmSpec.
    """
    prior = KnowledgeBaseManager().retrieve_prior(problem_features)
    items: list[dict[str, Any]] = []
    for rule in prior.get("matched_rules", []):
        items.append(
            {
                "source": rule.get("source"),
                "rule_id": rule.get("rule_id"),
                "stable_id": rule.get("stable_id"),
                "reason": rule.get("reason"),
                "then": rule.get("then") or rule.get("required_components") or {},
                "implementation_status": rule.get("implementation_status"),
                "evidence_sources": rule.get("evidence_sources", []),
                "binding": False,
            }
        )
    if len(items) < max_items:
        for group_name in ["pde_pinn_priors", "pinn_variants", "physics_rules", "scenario_rules"]:
            for item in prior.get(group_name, []):
                rule_id = item.get("rule_id") or item.get("name") or item.get("id")
                if rule_id and all(existing.get("rule_id") != rule_id for existing in items):
                    items.append(
                        {
                            "source": group_name,
                            "rule_id": rule_id,
                            "reason": item.get("reason") or item.get("limitations"),
                            "then": item.get("then") or item.get("required_components") or {},
                            "implementation_status": item.get("implementation_status"),
                            "evidence_sources": item.get("evidence_sources", []),
                            "binding": False,
                        }
                    )
                if len(items) >= max_items:
                    break
            if len(items) >= max_items:
                break
    return items[:max_items]


def retrieve_posterior_context(benchmark: str, problem_features: dict, max_items: int = 8) -> list[dict[str, Any]]:
    """Return posterior records as non-binding guidance."""
    posterior = PosteriorUpdater().retrieve_relevant_posterior(problem_features)
    items: list[dict[str, Any]] = []
    failure_patterns = posterior.get("failure_patterns", [])
    high_priority_failures = [
        pattern
        for pattern in failure_patterns
        if pattern.get("severity") in {"high", "critical"} or pattern.get("priority") == "high"
    ]
    for pattern in high_priority_failures[:max_items]:
        if pattern.get("benchmark_id") in {None, benchmark}:
            items.append({"source": "posterior_empirical_failure", "binding": False, **pattern})
    for pattern in posterior.get("design_principle_patterns", []):
        if len(items) >= max_items:
            break
        if pattern.get("benchmark_id") in {None, benchmark, "unknown"}:
            items.append(
                {
                    "source": "posterior_design_principle",
                    "binding": False,
                    "evidence_scope": "measured_local_experiments",
                    **pattern,
                }
            )
    for pattern in posterior.get("success_patterns", [])[: max(0, max_items - len(items))]:
        if pattern.get("benchmark_id") in {None, benchmark}:
            items.append({"source": "posterior_success", "binding": False, **pattern})
    for pattern in failure_patterns:
        if pattern in high_priority_failures:
            continue
        if len(items) >= max_items:
            break
        if pattern.get("benchmark_id") in {None, benchmark}:
            items.append({"source": "posterior_failure", "binding": False, **pattern})
    for record in posterior.get("recent_records", [])[: max(0, max_items - len(items))]:
        items.append(
            {
                "source": "posterior_recent_record",
                "benchmark_id": record.get("benchmark_id"),
                "algorithm_id": record.get("algorithm_id"),
                "metrics": record.get("metrics", {}),
                "binding": False,
            }
        )
    return items[:max_items]


def build_open_search_context(
    benchmark: str,
    problem_features: dict,
    current_population: list[dict],
    prior_context: list[dict],
    posterior_context: list[dict],
    registry_summary: dict,
    budget: dict,
    trainer_capabilities_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the context passed to an LLM or rule-based proxy."""
    equation_context = problem_features.get("pinnacle_profile") or problem_features.get("pde_equation")
    return {
        "benchmark": benchmark,
        "problem_features": problem_features,
        "pde_equation_context": equation_context,
        "current_population": current_population,
        "prior_context": prior_context,
        "posterior_context": posterior_context,
        "optional_runtime_context": registry_summary,
        "budget": budget,
        "trainer_capabilities_snapshot": trainer_capabilities_snapshot or {},
        "prior_knowledge_is_soft_guidance": True,
        "posterior_knowledge_is_soft_guidance": True,
        "knowledge_is_binding": False,
        "closed_world_prior_filter_used": False,
        "llm_output_is_used_directly": True,
        "intermediate_validation": False,
        "automatic_repair": False,
    }
