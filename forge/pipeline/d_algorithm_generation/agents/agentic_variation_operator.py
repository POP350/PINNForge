"""Unified LLM-driven generation of initial and evolutionary AlgorithmSpecs."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from typing import Any

from forge.pipeline.d_algorithm_generation.agents.base_agent import AgentResult, BaseAgent
from forge.pipeline.d_algorithm_generation.llm.json_extractor import extract_json_object
from forge.pipeline.d_algorithm_generation.initialization_diversity import (
    executable_algorithm_spec,
)
from forge.pipeline.e_credibility_assessment.validation import validate_algorithm_spec
from forge.utils.json_safety import json_safe


ALLOWED_GENERATION_MODES = {
    "initialization",
    "mutation",
    "crossover",
    "hybrid",
    "restart",
    "elite_preservation",
}
FORBIDDEN_RESPONSE_KEYS = {
    "operations",
    "patch",
    "patch_operations",
    "mutation_steps",
    "crossover_points",
    "fields_to_swap",
    "parameter_delta",
    "inheritance_operations",
}
CORE_COMPONENT_FIELDS = ("network", "activation", "sampler", "loss", "optimizer", "constraint")


class AgenticVariationOperator(BaseAgent):
    """Ask the LLM for complete candidates and filter them without editing them."""

    name = "AgenticVariationOperator"

    def run(self, payload: dict[str, Any]) -> AgentResult:
        generation = int(payload.get("generation") or 0)
        if generation == 0:
            data = self._generate_initial_population(payload)
        else:
            data = self._generate_evolutionary_offspring(payload)
        return AgentResult(True, data, f"Generated {len(data['children'])} validated AlgorithmSpecs.")

    def _generate_initial_population(self, payload: dict[str, Any]) -> dict[str, Any]:
        target_size = max(1, int(payload.get("target_population_size") or 5))
        proposal_count = max(target_size, int(payload.get("proposal_count") or 8))
        raw = self._request(payload, mode="initialization", requested_count=proposal_count)
        parsed, response_rejections = _validate_response(
            raw["parsed"], mode="initialization", max_children=proposal_count, allowed_parent_ids=set()
        )
        validated, validation_rejections = self._normalize_and_validate(
            parsed.get("children") or [], payload, mode="initialization"
        )
        accepted, diversity_rejections = select_initial_population(validated, target_size)
        rejected = [*response_rejections, *validation_rejections, *diversity_rejections]
        traces = [raw["trace"]]
        refill_rounds = 0
        max_refills = max(0, int(payload.get("max_initial_refill_rounds", 2)))

        while len(accepted) < target_size and refill_rounds < max_refills:
            refill_rounds += 1
            refill_payload = {
                **payload,
                "accepted_candidates": [_child_prompt_view(item) for item in accepted],
                "rejection_summary": rejected[-20:],
            }
            refill = self._request(
                refill_payload,
                mode="initialization_refill",
                requested_count=target_size - len(accepted),
            )
            traces.append(refill["trace"])
            parsed_refill, response_refill_rejections = _validate_response(
                refill["parsed"],
                mode="initialization",
                max_children=target_size - len(accepted),
                allowed_parent_ids=set(),
            )
            validated_refill, refill_validation_rejections = self._normalize_and_validate(
                parsed_refill.get("children") or [], payload, mode="initialization"
            )
            accepted, refill_diversity_rejections = select_initial_population(
                [*accepted, *validated_refill], target_size
            )
            rejected.extend(response_refill_rejections)
            rejected.extend(refill_validation_rejections)
            rejected.extend(refill_diversity_rejections)

        if len(accepted) != target_size:
            rejection_counts = Counter(str(item.get("reason") or "unknown") for item in rejected)
            validation_codes = Counter(
                str(error.get("code") or "unknown")
                for item in rejected
                for error in (item.get("validation_errors") or [])
                if isinstance(error, dict)
            )
            trace_summary = [
                {
                    "mode": trace.get("mode"),
                    "success": trace.get("success"),
                    "parse_errors": trace.get("parse_errors") or [],
                    "provider_error": trace.get("error"),
                    "raw_characters": len(str(trace.get("raw_response") or "")),
                }
                for trace in traces
            ]
            raise RuntimeError(
                "Unable to construct the required valid and diverse initial AlgorithmSpecs: "
                f"accepted={len(accepted)}/{target_size}, refill_rounds={refill_rounds}, "
                f"rejection_counts={dict(rejection_counts)}, validation_codes={dict(validation_codes)}, "
                f"traces={json.dumps(trace_summary, ensure_ascii=False)}"
            )
        return {
            "children": accepted,
            "rejected_children": rejected,
            "feedback_analysis": "No parent feedback in generation 0.",
            "strategy_summary": str(parsed.get("strategy_summary") or ""),
            "generation_mode": "initialization",
            "proposal_count": proposal_count,
            "accepted_count": len(accepted),
            "refill_rounds": refill_rounds,
            "llm_calls": len(traces),
            "traces": traces,
        }

    def _generate_evolutionary_offspring(self, payload: dict[str, Any]) -> dict[str, Any]:
        parents = list(payload.get("parent_candidates") or [])
        if not parents:
            raise ValueError("Evolutionary offspring generation requires parent candidates")
        target_count = max(1, int(payload.get("target_new_children") or payload.get("max_children") or 4))
        max_children = max(target_count, int(payload.get("max_children") or target_count))
        raw = self._request(payload, mode="evolution", requested_count=target_count)
        parent_ids = {str(item.get("candidate_id")) for item in parents if item.get("candidate_id")}
        parsed, response_rejections = _validate_response(
            raw["parsed"], mode="evolution", max_children=max_children, allowed_parent_ids=parent_ids
        )
        children, validation_rejections = self._normalize_and_validate(
            parsed.get("children") or [], payload, mode="evolution"
        )
        reference_specs = [item.get("algorithm_spec") for item in parents if isinstance(item.get("algorithm_spec"), dict)]
        accepted, duplicate_rejections = _remove_exact_duplicates(children, reference_specs=reference_specs)
        rejected = [*response_rejections, *validation_rejections, *duplicate_rejections]
        traces = [raw["trace"]]
        refill_rounds = 0
        max_refills = max(0, int(payload.get("max_evolution_refill_rounds", 2)))
        while len(accepted) < target_count and refill_rounds < max_refills:
            refill_rounds += 1
            refill_payload = {
                **payload,
                "accepted_new_children": [_evolution_child_prompt_view(item) for item in accepted],
                "rejection_summary": rejected[-20:],
            }
            refill = self._request(
                refill_payload,
                mode="evolution_refill",
                requested_count=target_count - len(accepted),
            )
            traces.append(refill["trace"])
            parsed_refill, refill_response_rejections = _validate_response(
                refill["parsed"],
                mode="evolution",
                max_children=target_count - len(accepted),
                allowed_parent_ids=parent_ids,
            )
            validated_refill, refill_validation_rejections = self._normalize_and_validate(
                parsed_refill.get("children") or [], payload, mode="evolution"
            )
            accepted, refill_duplicate_rejections = _remove_exact_duplicates(
                [*accepted, *validated_refill], reference_specs=reference_specs
            )
            rejected.extend(refill_response_rejections)
            rejected.extend(refill_validation_rejections)
            rejected.extend(refill_duplicate_rejections)
        if len(accepted) != target_count:
            raise RuntimeError("Unable to generate four valid and unique evolutionary candidates")
        return {
            "children": accepted,
            "rejected_children": rejected,
            "feedback_analysis": str(parsed.get("feedback_analysis") or ""),
            "strategy_summary": str(parsed.get("strategy_summary") or ""),
            "generation_mode": "evolution",
            "target_new_children": target_count,
            "accepted_count": len(accepted),
            "refill_rounds": refill_rounds,
            "llm_calls": len(traces),
            "traces": traces,
        }

    def _request(self, payload: dict[str, Any], *, mode: str, requested_count: int) -> dict[str, Any]:
        provider = payload["provider"]
        messages = _variation_messages(payload, mode=mode, requested_count=requested_count)
        response = provider.complete_text(
            messages,
            temperature=float(payload.get("temperature", 0.2)),
            max_tokens=int(payload.get("max_tokens", 4096)),
            task=f"agentic_variation_{mode}",
        )
        parsed, parse_errors = extract_json_object(response.text)
        parsed = parsed if isinstance(parsed, dict) else {}
        return {
            "parsed": parsed,
            "trace": {
                "mode": mode,
                "success": bool(response.success and parsed),
                "raw_response": response.text,
                "parse_errors": parse_errors,
                "token_usage": response.token_usage,
                "error": response.error,
            },
        }

    @staticmethod
    def _normalize_and_validate(
        children: list[dict[str, Any]], payload: dict[str, Any], *, mode: str
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        for index, child in enumerate(children):
            spec = child.get("algorithm_spec")
            report = validate_algorithm_spec(
                spec,
                payload["problem"],
                payload["training_budget"],
                payload["builder_capabilities"],
            )
            if not report.valid:
                rejected.append(
                    {
                        "candidate_label": _candidate_label(child, index),
                        "reason": "algorithm_spec_validation_failed",
                        "validation_errors": [
                            {"path": item.path, "code": item.code, "message": item.message}
                            for item in report.errors
                        ],
                    }
                )
                continue
            normalized_child = deepcopy(child)
            normalized_child["algorithm_spec"] = report.normalized_spec
            normalized_child["candidate_id"] = _candidate_label(child, index)
            normalized_child["parent_ids"] = [str(item) for item in child.get("parent_ids") or []]
            normalized_child["generation_mode"] = (
                "initialization" if mode == "initialization" else str(child.get("generation_mode"))
            )
            accepted.append(normalized_child)
        return accepted, rejected


def _variation_messages(payload: dict[str, Any], *, mode: str, requested_count: int) -> list[dict[str, str]]:
    common = {
        "generation": int(payload.get("generation") or 0),
        "problem_features": json_safe(payload.get("problem_features") or {}),
        "pde_equation_spec": json_safe(payload.get("pde_equation_spec") or {}),
        "filtered_design_space": json_safe(payload.get("filtered_design_space") or {}),
        "prior_knowledge": json_safe(payload.get("prior_knowledge") or []),
        "posterior_knowledge": json_safe(payload.get("posterior_knowledge") or []),
        "population_summary": json_safe(payload.get("population_summary") or []),
        "stagnation_report": json_safe(payload.get("stagnation_report") or {}),
        "stage": json_safe(payload.get("stage") or {}),
        "training_budget": json_safe(payload.get("training_budget") or {}),
        "trainer_capabilities_snapshot": json_safe(
            payload.get("trainer_capabilities_snapshot") or {}
        ),
        "adaptive_feedback": json_safe(
            payload.get("adaptive_feedback") or {}
        ),
        "requested_count": requested_count,
    }
    if mode == "initialization":
        system = (
            "You are the AgenticVariationOperator in initialization mode. There are no parents. "
            f"Directly design {requested_count} complete, independently valid AlgorithmSpec candidates. "
            "Return JSON only with strategy_summary and children. Every child must contain candidate_label, "
            "generation_mode='initialization', parent_ids=[], design_intent, difference_summary, and a complete "
            "algorithm_spec. Candidates must be materially different: every pair should differ in at least two "
            "core dimensions among architecture/input representation, network structure, activation, sampling, "
            "loss/weighting, optimization schedule, constraints, or adaptive training. Do not output Python, patches, "
            "operations, mutation steps, crossover points, or incomplete specs. Stay inside filtered_design_space and budget."
        )
    elif mode == "initialization_refill":
        common["accepted_candidates"] = json_safe(payload.get("accepted_candidates") or [])
        common["rejection_summary"] = json_safe(payload.get("rejection_summary") or [])
        system = (
            "You are refilling generation-0 initialization. Directly return exactly the requested number of new, "
            "complete AlgorithmSpecs that explore design regions not covered by accepted_candidates and avoid every "
            "reported rejection reason. Each child requires generation_mode='initialization', parent_ids=[], "
            "design_intent, difference_summary, and algorithm_spec. Do not modify existing candidates and do not output "
            "patches, operations, mutation steps, crossover points, or Python. Return JSON only with strategy_summary and children."
        )
    elif mode == "evolution_refill":
        common["parent_candidates"] = json_safe(payload.get("parent_candidates") or [])
        common["accepted_new_children"] = json_safe(payload.get("accepted_new_children") or [])
        common["rejection_summary"] = json_safe(payload.get("rejection_summary") or [])
        system = (
            "Refill the missing evolutionary children using only the supplied controller-selected global low-fidelity "
            "distinct Top-3 (P1/P2/P3). "
            "Return exactly the requested number of new complete AlgorithmSpecs. They must not exactly duplicate any "
            "parent or accepted_new_child, parent_ids must come only from P1/P2/P3, and every child must include "
            "generation_mode, variation_reason, heritage_explanation, expected_improvements, and algorithm_spec. "
            "Return JSON only. Never output patches, operations, mutation steps, crossover points, or Python."
        )
    else:
        common["parent_candidates"] = json_safe(payload.get("parent_candidates") or [])
        common["max_children"] = requested_count
        system = (
            "You are the AgenticVariationOperator for PINN evolution. You receive only the controller-selected global "
            "low-fidelity distinct Top-3 (P1, P2, and P3). Every requested slot must be a new candidate, so directly "
            f"generate exactly {requested_count} new complete AlgorithmSpecs and do not copy P1, P2, or P3 unchanged. "
            "Analyze their measured formal metrics, compact training summaries, and lineage. Autonomously choose one to three "
            "parents; decide whether each child is best described as mutation, crossover, hybrid, or restart; decide "
            "what to preserve and redesign; and directly output complete independently valid child AlgorithmSpecs. "
            "Return JSON only with feedback_analysis, strategy_summary, and children. Every child must contain parent_ids "
            "drawn only from the supplied candidate IDs, generation_mode, variation_reason, heritage_explanation, "
            "expected_improvements, and algorithm_spec. generation_mode is explanatory metadata only. Never output Python, "
            "patches, operations, mutation steps, crossover points, inheritance instructions, or parameter deltas for code to apply."
            " When stagnation_report contains measured directives, record its source_generation in "
            "algorithm_spec.generation_metadata.feedback_source_generation and list applied directives in "
            "algorithm_spec.generation_metadata.reflection_directives_applied."
        )
    system += (
        " Select adaptive policies only from trainer_capabilities_snapshot.eligible_adaptive_policies; enabling every policy is not required. "
        "Use measured adaptive_feedback to distinguish structural error, controller rollback, optimizer stall, unfinished curriculum, and overhead. "
        "Output the next complete AlgorithmSpec, never a controller patch. Inherit policy design and initial hyperparameters only; "
        "never inherit final dynamic weights, sampling points, curriculum level, optimizer state, EMA state, or checkpoints. "
        "Do not use evaluator/reference metrics as Trainer control conditions or bypass fixed budgets."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(common, ensure_ascii=False)}]


def _validate_response(
    parsed: dict[str, Any], *, mode: str, max_children: int, allowed_parent_ids: set[str]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rejected: list[dict[str, Any]] = []
    if _contains_forbidden_key(parsed):
        return {**parsed, "children": []}, [{"reason": "forbidden_executable_operations"}]
    raw_children = parsed.get("children")
    if not isinstance(raw_children, list):
        return {**parsed, "children": []}, [{"reason": "children_must_be_array"}]
    valid: list[dict[str, Any]] = []
    for index, child in enumerate(raw_children[:max_children]):
        label = _candidate_label(child if isinstance(child, dict) else {}, index)
        reason = _child_response_error(child, mode=mode, allowed_parent_ids=allowed_parent_ids)
        if reason:
            rejected.append({"candidate_label": label, "reason": reason})
        else:
            valid.append(child)
    if len(raw_children) > max_children:
        rejected.extend(
            {"candidate_label": _candidate_label(item if isinstance(item, dict) else {}, index), "reason": "too_many_children"}
            for index, item in enumerate(raw_children[max_children:], start=max_children)
        )
    return {**parsed, "children": valid}, rejected


def _child_response_error(child: Any, *, mode: str, allowed_parent_ids: set[str]) -> str | None:
    if not isinstance(child, dict):
        return "child_must_be_object"
    if _contains_forbidden_key(child):
        return "forbidden_executable_operations"
    if not isinstance(child.get("algorithm_spec"), dict):
        return "algorithm_spec_must_be_object"
    generation_mode = str(child.get("generation_mode") or "")
    parent_ids = child.get("parent_ids")
    if not isinstance(parent_ids, list):
        return "parent_ids_must_be_array"
    if mode == "initialization":
        if generation_mode != "initialization":
            return "initial_generation_mode_required"
        if parent_ids:
            return "initial_parent_ids_must_be_empty"
        if not str(child.get("design_intent") or "").strip():
            return "design_intent_required"
        if not str(child.get("difference_summary") or "").strip():
            return "difference_summary_required"
        return None
    if generation_mode not in ALLOWED_GENERATION_MODES - {"initialization"}:
        return "invalid_generation_mode"
    if not parent_ids:
        return "evolution_parent_ids_required"
    if any(str(parent_id) not in allowed_parent_ids for parent_id in parent_ids):
        return "unknown_parent_id"
    return None


def canonicalize_algorithm_spec(spec: dict[str, Any]) -> str:
    return json.dumps(
        executable_algorithm_spec(spec),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def select_initial_population(
    children: list[dict[str, Any]], target_size: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, child in enumerate(children):
        canonical = canonicalize_algorithm_spec(child["algorithm_spec"])
        if canonical in seen:
            rejected.append({"candidate_label": _candidate_label(child, index), "reason": "exact_duplicate"})
            continue
        accepted.append(child)
        seen.add(canonical)
        if len(accepted) == target_size:
            break
    return accepted, rejected


def build_initial_design_signature(spec: dict[str, Any]) -> dict[str, Any]:
    network = spec.get("network") or {}
    sampling = spec.get("sampling") or {}
    loss = spec.get("loss") or {}
    phases = (spec.get("optimization") or {}).get("phases") or []
    layers = [int(item) for item in network.get("hidden_layers") or [] if isinstance(item, int)]
    adaptive = sampling.get("adaptive_refinement") or {}
    return {
        "network": (
            network.get("architecture"),
            (network.get("input_transform") or {}).get("name"),
            bool((network.get("residual_connections") or {}).get("enabled")),
        ),
        "activation": (network.get("activation") or {}).get("name"),
        "sampler": (
            (sampling.get("interior") or {}).get("strategy"),
            adaptive.get("strategy") if adaptive.get("enabled") else "none",
        ),
        "loss": (
            tuple((item.get("name"), item.get("loss_function")) for item in loss.get("terms") or []),
            (loss.get("weighting_strategy") or {}).get("name"),
            (loss.get("aggregation") or {}).get("name"),
        ),
        "optimizer": tuple(
            (item.get("optimizer"), (item.get("scheduler") or {}).get("name")) for item in phases
        ),
        "constraint": tuple(item.get("name") for item in loss.get("terms") or []),
        "depth": len(layers),
        "width": max(layers) if layers else 0,
    }


def is_initial_candidate_diverse(candidate_spec: dict[str, Any], accepted_specs: list[dict[str, Any]]) -> bool:
    candidate = build_initial_design_signature(candidate_spec)
    for accepted_spec in accepted_specs:
        accepted = build_initial_design_signature(accepted_spec)
        component_differences = sum(candidate.get(field) != accepted.get(field) for field in CORE_COMPONENT_FIELDS)
        depth_difference = abs(int(candidate.get("depth") or 0) - int(accepted.get("depth") or 0))
        widths = (int(candidate.get("width") or 0), int(accepted.get("width") or 0))
        width_ratio = max(widths) / max(1, min(widths)) if max(widths) else 1.0
        architecture_difference = depth_difference >= 2 or width_ratio >= 1.5
        if component_differences >= 2:
            continue
        if component_differences >= 1 and architecture_difference:
            continue
        return False
    return True


def _remove_exact_duplicates(
    children: list[dict[str, Any]], reference_specs: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: set[str] = {
        canonicalize_algorithm_spec(spec) for spec in (reference_specs or []) if isinstance(spec, dict)
    }
    for index, child in enumerate(children):
        canonical = canonicalize_algorithm_spec(child["algorithm_spec"])
        if canonical in seen:
            rejected.append({"candidate_label": _candidate_label(child, index), "reason": "exact_duplicate"})
            continue
        seen.add(canonical)
        accepted.append(child)
    return accepted, rejected


def _contains_forbidden_key(value: Any) -> bool:
    if isinstance(value, dict):
        return any(key in FORBIDDEN_RESPONSE_KEYS or _contains_forbidden_key(item) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_forbidden_key(item) for item in value)
    return False


def _candidate_label(child: dict[str, Any], index: int) -> str:
    return str(child.get("candidate_id") or child.get("candidate_label") or f"candidate_{index + 1:02d}")


def _child_prompt_view(child: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": child.get("candidate_id"),
        "design_intent": child.get("design_intent"),
        "difference_summary": child.get("difference_summary"),
        "algorithm_spec": child.get("algorithm_spec"),
    }


def _evolution_child_prompt_view(child: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": child.get("candidate_id"),
        "parent_ids": child.get("parent_ids") or [],
        "generation_mode": child.get("generation_mode"),
        "variation_reason": child.get("variation_reason") or "",
        "algorithm_spec": child.get("algorithm_spec"),
    }


__all__ = [
    "AgenticVariationOperator",
    "ALLOWED_GENERATION_MODES",
    "build_initial_design_signature",
    "canonicalize_algorithm_spec",
    "is_initial_candidate_diverse",
    "select_initial_population",
]
