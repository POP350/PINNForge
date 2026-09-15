"""Generation reflection grounded in measured candidate records."""

from __future__ import annotations

import json
from typing import Any

from forge.pipeline.d_algorithm_generation.agents.base_agent import AgentResult, BaseAgent
from forge.pipeline.d_algorithm_generation.llm.json_extractor import extract_json_object
from forge.pipeline.g_training_evaluation.evaluation import PRIMARY_RANKING_METRIC


class ReflectionAgent(BaseAgent):
    name = "ReflectionAgent"

    def run(self, payload: dict[str, Any]) -> AgentResult:
        records = list(payload.get("records") or [])
        successful = [record for record in records if record.get("training_success")]
        failed = [record for record in records if not record.get("training_success")]
        deterministic = {
            "contract_version": "1.0",
            "source_stage": "H",
            "target_stage": "D",
            "generation": payload.get("generation"),
            "measured_success_count": len(successful),
            "measured_failure_count": len(failed),
            "best_spec_id": min(
                successful,
                key=lambda item: float(
                    (item.get("metrics") or {}).get(PRIMARY_RANKING_METRIC, float("inf"))
                ),
            ).get("spec_id") if successful else None,
            "successful_spec_ids": [item.get("spec_id") for item in successful if item.get("spec_id")],
            "failed_spec_ids": [item.get("spec_id") for item in failed if item.get("spec_id")],
            "failure_reasons": [item.get("failure_reason") for item in failed[:8]],
            "next_directive": "Preserve measured strengths and repair executable failures using a complete AlgorithmSpec.",
        }
        provider = payload.get("provider")
        if provider is None:
            return AgentResult(True, {"reflection": deterministic})
        messages = [
            {
                "role": "system",
                "content": (
                    "Reflect only on measured PINN experiment evidence. Do not claim success from LLM self-assessment. "
                    "Return one JSON object with observed_strengths, failure_patterns, recommended_actions, and next_directive. "
                    "Recommendations must name concrete AlgorithmSpec modules or parameters and must not alter the PDE, budget, or seed. "
                    "Use AdaptiveControlSummary and compute/scaling summaries to distinguish an algorithm-structure failure from "
                    "frequent controller rollback, L-BFGS stall, unfinished curriculum, or excessive adaptive overhead. "
                    "Recommend only the next complete AlgorithmSpec design. Never inherit final dynamic weights, sampler points, "
                    "curriculum state, optimizer state, EMA state, or model checkpoints."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "measured_summary": deterministic,
                        "records": [_reflection_record_view(item) for item in records[:12]],
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        response = provider.complete_text(messages, temperature=0.1, max_tokens=2048, task="reflection")
        parsed, _ = extract_json_object(response.text)
        if response.success and isinstance(parsed, dict):
            reflection = dict(parsed)
            for key in (
                "contract_version",
                "source_stage",
                "target_stage",
                "generation",
                "measured_success_count",
                "measured_failure_count",
                "best_spec_id",
                "successful_spec_ids",
                "failed_spec_ids",
                "failure_reasons",
            ):
                reflection[key] = deterministic[key]
            reflection.setdefault("next_directive", deterministic["next_directive"])
        else:
            reflection = deterministic
        return AgentResult(bool(response.success and parsed), {"reflection": reflection, "raw_response": response.text, "response": response})


def _reflection_record_view(record: dict[str, Any]) -> dict[str, Any]:
    """Bound the LLM payload while retaining measured design/control evidence."""

    return {
        "spec_id": record.get("spec_id"),
        "training_success": bool(record.get("training_success")),
        "failure_reason": record.get("failure_reason"),
        "algorithm_spec": record.get("search_algorithm_spec")
        or record.get("normalized_algorithm_spec")
        or {},
        "formal_metrics": record.get("generation_metrics")
        or record.get("metrics")
        or {},
        "training_summary": {
            key: (record.get("training_report") or {}).get(key)
            for key in (
                "actual_iterations", "final_loss", "nan_detected",
                "stopped_early", "failure_reason",
            )
        },
        "adaptive_control_summary": record.get("adaptive_control_summary")
        or (record.get("training_report") or {}).get("adaptive_control_summary")
        or {},
        "runtime_scaled_policy_summary": record.get(
            "runtime_scaled_policy_summary"
        ) or {},
        "adaptive_compute_budget_summary": record.get(
            "adaptive_compute_budget_summary"
        ) or {},
        "adaptive_policy_warnings": list(
            record.get("adaptive_policy_warnings") or []
        )[:8],
    }
