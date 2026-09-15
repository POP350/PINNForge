"""LLM agent that emits complete Open AlgorithmSpec candidates."""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any

from forge.pipeline.c_knowledge_retrieval.knowledge_base.kb_manager import (
    DEFAULT_TOP_K_ARCHITECTURE_PRINCIPLES,
    DEFAULT_TOP_K_FAILURE_PATTERNS,
    DEFAULT_TOP_K_MACRO_PRINCIPLES,
    DEFAULT_TOP_K_POSTERIOR_EVIDENCE,
    MAX_COMPACT_LIST_ITEMS,
    KnowledgeBaseManager,
    _compact_value,
    _normalize_tag,
    _safe_float,
    _validated_top_k,
    matches_required_conditions,
    rank_design_principles,
)
from forge.pipeline.d_algorithm_generation.agents.base_agent import AgentResult, BaseAgent
from forge.pipeline.d_algorithm_generation.agents.candidate_roles import (
    ALLOWED_VARIATION_STRATEGIES,
    resolve_candidate_role,
)
from forge.pipeline.d_algorithm_generation.llm.json_extractor import (
    extract_json_object_with_audit,
)
from forge.pipeline.d_algorithm_generation.specs import (
    REQUIRED_TOP_LEVEL_FIELDS,
    algorithm_spec_template,
    llm_algorithm_spec_option_registry,
)
from forge.pipeline.posterior_memory.compression import estimate_context_tokens


DEFAULT_DESIGN_MAX_TOKENS = 80_000
MIN_DESIGN_MAX_TOKENS = 1024
MAX_PARENT_CONTEXT_TOKENS = 20_000
MAX_VALIDATION_REPAIR_TOKENS = 8_192
INDEPENDENT_PARENT_REFERENCE_FIELDS = {
    "algorithm_spec",
    "normalized_algorithm_spec",
    "search_algorithm_spec",
    "parent_ids",
    "parent_spec_ids",
    "candidate_id",
    "spec_id",
}
INDEPENDENT_LINEAGE_FIELDS = {
    *INDEPENDENT_PARENT_REFERENCE_FIELDS,
    "variation_strategy",
    "inherited_strengths",
}


def _strip_independent_lineage_fields(value: Any) -> Any:
    """Remove executable lineage while retaining aggregate measured evidence."""

    if isinstance(value, dict):
        return {
            str(key): _strip_independent_lineage_fields(item)
            for key, item in value.items()
            if str(key) not in INDEPENDENT_LINEAGE_FIELDS
        }
    if isinstance(value, list):
        return [_strip_independent_lineage_fields(item) for item in value]
    return value


class DesignAgent(BaseAgent):
    name = "DesignAgent"

    def run(self, payload: dict[str, Any]) -> AgentResult:
        provider = payload["provider"]
        max_tokens = _resolve_max_tokens(payload)
        role = resolve_candidate_role(payload)
        resolved_context = _resolved_design_context(payload)
        principles = resolved_context["design_principles"]
        posterior_principle_evidence = resolved_context["posterior_principle_evidence"]
        posterior_failure_patterns = resolved_context["posterior_failure_patterns"]
        if not bool(payload.get("execution_feedback", True)):
            posterior_principle_evidence = []
            posterior_failure_patterns = []
        elif not bool(payload.get("evolutionary_inheritance_enabled", True)):
            posterior_principle_evidence = _strip_independent_lineage_fields(
                posterior_principle_evidence
            )
            posterior_failure_patterns = _strip_independent_lineage_fields(
                posterior_failure_patterns
            )
        validation_repair = payload.get("validation_repair")
        if isinstance(validation_repair, dict):
            max_tokens = min(max_tokens, MAX_VALIDATION_REPAIR_TOKENS)
            messages = _validation_repair_messages(payload, validation_repair)
            request_mode = "validation_repair"
        else:
            messages = _design_messages(
                payload,
                principles,
                posterior_principle_evidence,
                posterior_failure_patterns,
            )
            request_mode = "full_design"
        retrieval_metadata = _principle_retrieval_metadata(
            payload,
            principles,
            posterior_principle_evidence,
            posterior_failure_patterns,
        )
        execution_feedback_audit = _assert_execution_feedback_boundary(
            messages, payload
        )
        evolutionary_inheritance_audit = (
            _assert_evolutionary_inheritance_boundary(messages, payload)
        )
        prompt_statistics = _prompt_statistics(messages, payload, principles, posterior_principle_evidence)
        prompt_statistics["execution_feedback_audit"] = execution_feedback_audit
        prompt_statistics["evolutionary_inheritance_audit"] = (
            evolutionary_inheritance_audit
        )
        response = provider.complete_text(
            messages,
            temperature=float(payload.get("temperature", 0.3)),
            max_tokens=max_tokens,
            task=str(payload.get("proposal_type") or payload.get("candidate_role") or "proposal"),
        )
        parsed, errors, extraction_repairs = extract_json_object_with_audit(
            response.text,
            preferred_keys=("proposals", "algorithm_spec", "spec_version", "schema_version"),
        )
        proposals, contract_errors = _extract_single_proposal(
            parsed,
            payload,
            initial_repairs=extraction_repairs,
        )
        errors = [*errors, *contract_errors]
        if not response.success:
            errors.append(
                "Provider request failed: "
                + str(response.error or "provider returned success=false")
            )
        output_token_limit_reached = (
            str(response.finish_reason or "").casefold() == "length"
        )
        # Some OpenAI-compatible gateways report ``length`` even when the JSON
        # object ended cleanly (for example, after hidden reasoning tokens use
        # the remaining allowance).  Reject only an actually incomplete
        # contract, not a complete proposal with a conservative finish reason.
        response_truncated = bool(
            output_token_limit_reached
            and not (isinstance(parsed, dict) and len(proposals) == 1 and not contract_errors)
        )
        if response_truncated:
            errors.append("Provider stopped at the output token limit; the response is truncated.")
        if role and len(proposals) == 1:
            _apply_role_metadata(proposals[0]["algorithm_spec"], role)
        transport_repairs = (
            list(proposals[0].get("transport_repairs") or [])
            if len(proposals) == 1
            else []
        )
        parse_success = bool(parsed is not None and len(proposals) == 1 and not errors)
        success = bool(response.success and parse_success and not response_truncated)
        return AgentResult(
            success,
            {
                "proposals": proposals,
                "raw_response": response.text,
                "parse_errors": errors,
                "response": response,
                "messages": messages,
                "design_principles": principles,
                "principle_retrieval_metadata": retrieval_metadata,
                "prompt_statistics": prompt_statistics,
                "principle_reference_warnings": _principle_reference_warnings(
                    [item.get("algorithm_spec") or {} for item in proposals],
                    principles,
                    posterior_principle_evidence,
                ),
                "requested_max_tokens": max_tokens,
                "request_mode": request_mode,
                "provider": response.provider,
                "provider_success": bool(response.success),
                "provider_error": response.error,
                "model": response.model,
                "model_version": response.model_version,
                "latency_seconds": response.latency_seconds,
                "finish_reason": response.finish_reason,
                "output_token_limit_reached": output_token_limit_reached,
                "response_truncated": response_truncated,
                "parse_success": parse_success,
                "proposal_count": len(proposals),
                "candidate_id_match": bool(
                    len(proposals) == 1
                    and str(proposals[0].get("candidate_id") or "")
                    == str(payload.get("candidate_id") or "")
                ),
                "transport_repairs": transport_repairs,
                "token_usage": dict(response.token_usage or {}),
            },
            (
                "Complete Open AlgorithmSpec generated."
                if success
                else (
                    f"DesignAgent provider request failed: {response.error}"
                    if not response.success and response.error
                    else "DesignAgent requires exactly one complete, non-truncated proposal per LLM call."
                )
            ),
        )


def resolve_shared_design_context(payload: dict[str, Any]) -> dict[str, Any]:
    """Retrieve and compact design knowledge once for reuse by sibling candidates."""

    principles = _resolve_design_principles(payload)
    return {
        "design_principles": principles,
        "posterior_principle_evidence": resolve_posterior_principle_evidence(payload, principles),
        "posterior_failure_patterns": resolve_posterior_failure_patterns(payload),
    }


def _resolved_design_context(payload: dict[str, Any]) -> dict[str, Any]:
    supplied = payload.get("resolved_design_context")
    if supplied is None:
        return resolve_shared_design_context(payload)
    if not isinstance(supplied, dict):
        raise ValueError("resolved_design_context must be an object")
    required = {
        "design_principles",
        "posterior_principle_evidence",
        "posterior_failure_patterns",
    }
    missing = sorted(required - set(supplied))
    if missing:
        raise ValueError(f"resolved_design_context is missing: {', '.join(missing)}")
    return deepcopy(supplied)


def _resolve_max_tokens(payload: dict[str, Any]) -> int:
    try:
        max_tokens = int(payload.get("max_tokens", DEFAULT_DESIGN_MAX_TOKENS))
    except (TypeError, ValueError) as exc:
        raise ValueError("DesignAgent max_tokens must be an integer") from exc
    if max_tokens < MIN_DESIGN_MAX_TOKENS:
        raise ValueError(
            "DesignAgent max_tokens must be at least "
            f"{MIN_DESIGN_MAX_TOKENS}, got {max_tokens}."
        )
    return max_tokens


def _extract_single_proposal(
    parsed: dict[str, Any] | None,
    payload: dict[str, Any],
    *,
    initial_repairs: list[dict[str, str]] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Apply the one-call/one-candidate transport contract with audited repairs.

    Envelope and identifier mistakes cannot change the executable algorithm, so
    unambiguous instances are repaired and recorded.  Multiple proposals and
    incomplete AlgorithmSpecs remain hard failures because selecting or filling
    those would require semantic judgment.
    """

    if not isinstance(parsed, dict):
        return [], []
    expected_candidate_id = payload.get("candidate_id")
    envelope_repairs: list[dict[str, str]] = deepcopy(initial_repairs or [])

    def record(path: str, action: str) -> None:
        envelope_repairs.append({"path": path, "action": action})

    raw_proposals: Any
    if "proposals" in parsed:
        raw_proposals = parsed.get("proposals")
    elif isinstance(parsed.get("proposal"), dict):
        raw_proposals = [parsed["proposal"]]
        record("proposal", "renamed the singular proposal envelope to proposals")
    elif all(key in parsed for key in REQUIRED_TOP_LEVEL_FIELDS):
        raw_proposals = [
            {
                "candidate_id": expected_candidate_id or "llm_open_spec",
                "algorithm_spec": parsed,
            }
        ]
        record("$", "wrapped a direct AlgorithmSpec in the required proposals envelope")
    elif isinstance(parsed.get("algorithm_spec"), dict):
        raw_proposals = [parsed]
        record("$", "wrapped a direct proposal object in the required proposals list")
    elif isinstance(parsed.get("spec"), dict):
        raw_proposals = [{**parsed, "algorithm_spec": parsed["spec"]}]
        record("spec", "renamed the proposal spec alias to algorithm_spec")
    elif _looks_like_executable_spec(parsed):
        raw_proposals = [
            {
                "candidate_id": expected_candidate_id or "llm_open_spec",
                "algorithm_spec": _spec_fields_from_mapping(parsed),
            }
        ]
        record(
            "$",
            "wrapped a direct AlgorithmSpec missing only non-executable transport fields",
        )
    else:
        return [], ["Response must contain a proposals list with exactly one item."]
    if isinstance(raw_proposals, dict):
        raw_proposals = [raw_proposals]
        record("proposals", "wrapped the single proposal object in a list")
    if not isinstance(raw_proposals, list):
        return [], ["The proposals field must be a list."]
    if len(raw_proposals) != 1:
        selected = _select_unambiguous_proposal(
            raw_proposals,
            expected_candidate_id=expected_candidate_id,
        )
        if selected is None:
            observed = [dict(item) for item in raw_proposals if isinstance(item, dict)]
            return observed, [f"Expected exactly one proposal, received {len(raw_proposals)}."]
        record(
            "proposals",
            "selected the unique proposal matching the supplied candidate_id from "
            f"{len(raw_proposals)} returned items",
        )
        raw_proposals = [selected]
    item = raw_proposals[0]
    if not isinstance(item, dict):
        return [], ["The single proposal must be an object."]
    original_spec = item.get("algorithm_spec")
    if not isinstance(original_spec, dict):
        for alias in ("spec", "normalized_algorithm_spec"):
            if isinstance(item.get(alias), dict):
                original_spec = item[alias]
                record(
                    f"proposals[0].{alias}",
                    f"renamed {alias} to algorithm_spec",
                )
                break
    if not isinstance(original_spec, dict) and _looks_like_executable_spec(item):
        original_spec = _spec_fields_from_mapping(item)
        record(
            "proposals[0]",
            "moved direct AlgorithmSpec fields into algorithm_spec",
        )
    if not isinstance(original_spec, dict):
        return [], ["The single proposal must contain an algorithm_spec object."]
    spec, spec_repairs = _canonicalize_llm_transport(original_spec)
    spec, problem_repairs = _canonicalize_problem_components(
        spec,
        payload.get("problem_spec") or {},
    )
    spec_repairs.extend(problem_repairs)
    template = algorithm_spec_template()
    if "spec_version" not in spec:
        spec["spec_version"] = template["spec_version"]
        spec_repairs.append(
            {
                "path": "spec_version",
                "action": "inserted the current non-executable schema version",
            }
        )
    metadata = spec.get("generation_metadata")
    if metadata is None:
        metadata = {}
        spec["generation_metadata"] = metadata
        spec_repairs.append(
            {
                "path": "generation_metadata",
                "action": "inserted missing non-executable generation metadata",
            }
        )
    if isinstance(metadata, dict):
        supplied_metadata = item.get("generation_metadata")
        if isinstance(supplied_metadata, dict):
            for key, value in supplied_metadata.items():
                metadata.setdefault(key, deepcopy(value))
            spec_repairs.append(
                {
                    "path": "generation_metadata",
                    "action": "moved proposal-level generation metadata into algorithm_spec",
                }
            )
        defaults = template["generation_metadata"]
        inserted = [key for key in defaults if key not in metadata]
        for key in inserted:
            metadata[key] = deepcopy(defaults[key])
        if inserted:
            spec_repairs.append(
                {
                    "path": "generation_metadata",
                    "action": "filled omitted audit-only metadata fields from schema defaults",
                }
            )
    missing = [field for field in REQUIRED_TOP_LEVEL_FIELDS if field not in spec]
    if missing:
        return [], [f"AlgorithmSpec is missing required top-level fields: {', '.join(missing)}."]
    candidate_id = item.get("candidate_id")
    if expected_candidate_id is not None and str(candidate_id or "") != str(expected_candidate_id):
        action = (
            "inserted the supplied candidate_id"
            if candidate_id in (None, "")
            else "replaced a mismatched candidate_id with the supplied candidate_id"
        )
        candidate_id = expected_candidate_id
        record("proposals[0].candidate_id", action)
    elif expected_candidate_id is None and not candidate_id:
        candidate_id = "llm_open_spec"
        record("proposals[0].candidate_id", "inserted the legacy default candidate_id")
    transport_repairs = [*envelope_repairs, *spec_repairs]
    return [
        {
            **item,
            "candidate_id": candidate_id,
            "algorithm_spec": spec,
            "algorithm_spec_raw": deepcopy(original_spec),
            "transport_repairs": transport_repairs,
        }
    ], []


def _looks_like_executable_spec(value: Any) -> bool:
    """Recognize a direct spec without guessing any executable section."""

    return isinstance(value, dict) and all(
        isinstance(value.get(section), dict)
        for section in ("network", "sampling", "loss", "optimization", "training")
    )


def _spec_fields_from_mapping(value: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        *REQUIRED_TOP_LEVEL_FIELDS,
        "constraint_enforcement",
        "schema_version",
    }
    return {
        key: deepcopy(item)
        for key, item in value.items()
        if key in allowed
    }


def _proposal_has_spec(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    return (
        isinstance(value.get("algorithm_spec"), dict)
        or isinstance(value.get("spec"), dict)
        or isinstance(value.get("normalized_algorithm_spec"), dict)
        or _looks_like_executable_spec(value)
    )


def _select_unambiguous_proposal(
    proposals: list[Any], *, expected_candidate_id: Any
) -> dict[str, Any] | None:
    """Select only when transport intent is provable without semantic ranking."""

    usable = [item for item in proposals if _proposal_has_spec(item)]
    if expected_candidate_id is not None:
        matching = [
            item
            for item in usable
            if str(item.get("candidate_id") or "") == str(expected_candidate_id)
        ]
        if len(matching) == 1:
            return matching[0]
    return usable[0] if len(usable) == 1 else None


def _canonicalize_problem_components(
    spec: dict[str, Any], problem_spec: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Resolve generic component aliases only when the registration is unique."""

    canonical = deepcopy(spec)
    repairs: list[dict[str, str]] = []
    constraints = [
        item
        for item in problem_spec.get("constraints") or []
        if isinstance(item, dict) and item.get("constraint_id")
    ]
    laws = [
        item
        for item in problem_spec.get("governing_laws") or []
        if isinstance(item, dict) and item.get("law_id")
    ]
    if not constraints and "constraints" not in problem_spec:
        return canonical, repairs

    by_type: dict[str, list[str]] = {}
    for constraint in constraints:
        by_type.setdefault(
            str(constraint.get("constraint_type") or "").casefold(), []
        ).append(str(constraint["constraint_id"]))
    boundary_ids = [
        *by_type.get("boundary", []),
        *by_type.get("periodic", []),
        *by_type.get("interface", []),
    ]
    initial_ids = list(by_type.get("initial", []))
    terms = ((canonical.get("loss") or {}).get("terms") or [])
    retained: list[Any] = []
    for index, term in enumerate(terms):
        if not isinstance(term, dict):
            retained.append(term)
            continue
        name = str(term.get("name") or "")
        replacement: str | None = None
        if name == "boundary_condition":
            if len(boundary_ids) == 1:
                replacement = boundary_ids[0]
            elif not boundary_ids:
                repairs.append(
                    {
                        "path": f"loss.terms[{index}]",
                        "action": "removed unavailable generic boundary_condition term",
                    }
                )
                continue
        elif name == "initial_condition":
            if len(initial_ids) == 1:
                replacement = initial_ids[0]
            elif not initial_ids:
                repairs.append(
                    {
                        "path": f"loss.terms[{index}]",
                        "action": "removed unavailable generic initial_condition term",
                    }
                )
                continue
        elif name == "constraint_component":
            component = (term.get("parameters") or {}).get("component_name")
            if not component and len(constraints) == 1:
                replacement = str(constraints[0]["constraint_id"])
        elif name == "pde_component":
            component = (term.get("parameters") or {}).get("component_name")
            if not component and len(laws) == 1:
                law_id = str(laws[0]["law_id"])
                term.setdefault("parameters", {})["component_name"] = law_id
                repairs.append(
                    {
                        "path": f"loss.terms[{index}].parameters.component_name",
                        "action": f"selected the only registered governing law ID {law_id}",
                    }
                )
        if replacement is not None:
            term["name"] = replacement
            parameters = dict(term.get("parameters") or {})
            parameters.pop("component_name", None)
            term["parameters"] = parameters
            repairs.append(
                {
                    "path": f"loss.terms[{index}].name",
                    "action": f"resolved generic component to registered ID {replacement}",
                }
            )
        retained.append(term)
    if isinstance(canonical.get("loss"), dict):
        canonical["loss"]["terms"] = retained
    return canonical, repairs


def _canonicalize_llm_transport(
    spec: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Canonicalize unambiguous legacy JSON shapes without tuning the algorithm.

    Every applied rewrite is recorded. Contradictory or ambiguous values are
    deliberately left untouched so deterministic validation can reject them.
    """

    canonical = deepcopy(spec)
    repairs: list[dict[str, str]] = []

    def record(path: str, action: str) -> None:
        repairs.append({"path": path, "action": action})

    if "spec_version" not in canonical and isinstance(canonical.get("schema_version"), str):
        canonical["spec_version"] = canonical.pop("schema_version")
        record("spec_version", "renamed schema_version to spec_version")
    elif (
        isinstance(canonical.get("spec_version"), str)
        and canonical.get("schema_version") == canonical.get("spec_version")
    ):
        canonical.pop("schema_version")
        record("schema_version", "removed redundant schema_version matching spec_version")

    network = canonical.get("network")
    if isinstance(network, dict):
        architecture = network.get("architecture")
        if isinstance(architecture, dict) and isinstance(architecture.get("name"), str):
            parameters = architecture.get("parameters")
            if _merge_legacy_parameters(network, "extra_parameters", parameters):
                network["architecture"] = architecture["name"]
                record(
                    "network.architecture",
                    "flattened {name, parameters} into architecture and extra_parameters",
                )
        if "architecture_parameters" in network and _rename_legacy_parameters(
            network, "architecture_parameters", "extra_parameters"
        ):
            record(
                "network.architecture_parameters",
                "renamed architecture_parameters to extra_parameters",
            )
        if "extra_parameters" not in network:
            network["extra_parameters"] = {}
            record("network.extra_parameters", "inserted an empty parameter object")
        for component_name in (
            "activation",
            "output_activation",
            "initialization",
            "input_transform",
            "residual_connections",
        ):
            component = network.get(component_name)
            if isinstance(component, dict) and "parameters" not in component:
                component["parameters"] = {}
                record(
                    f"network.{component_name}.parameters",
                    "inserted an empty parameter object",
                )

    sampling = canonical.get("sampling")
    if isinstance(sampling, dict):
        for sampler_name in ("interior", "boundary", "initial", "adaptive_refinement"):
            sampler = sampling.get(sampler_name)
            if isinstance(sampler, dict) and "parameters" not in sampler:
                sampler["parameters"] = {}
                record(
                    f"sampling.{sampler_name}.parameters",
                    "inserted an empty parameter object",
                )

    enforcement = canonical.get("constraint_enforcement")
    if isinstance(enforcement, dict) and "parameters" not in enforcement:
        enforcement["parameters"] = {}
        record(
            "constraint_enforcement.parameters",
            "inserted an empty parameter object",
        )

    loss = canonical.get("loss")
    if isinstance(loss, dict) and isinstance(loss.get("terms"), list):
        for index, term in enumerate(loss["terms"]):
            if not isinstance(term, dict):
                continue
            parameters = term.get("parameters")
            if (
                term.get("name") == "constraint_component"
                and isinstance(parameters, dict)
                and isinstance(parameters.get("component_name"), str)
                and parameters["component_name"].strip()
            ):
                component_name = parameters.pop("component_name").strip()
                term["name"] = component_name
                record(
                    f"loss.terms[{index}]",
                    "canonicalized constraint_component selector to its concrete constraint_id loss term",
                )
            if "loss_function_parameters" in term and _rename_legacy_parameters(
                term, "loss_function_parameters", "parameters"
            ):
                record(
                    f"loss.terms[{index}].loss_function_parameters",
                    "renamed loss_function_parameters to parameters",
                )
            if "parameters" not in term:
                term["parameters"] = {}
                record(
                    f"loss.terms[{index}].parameters",
                    "inserted an empty parameter object",
                )
        for component_name in ("weighting_strategy", "aggregation"):
            component = loss.get(component_name)
            if isinstance(component, dict) and "parameters" not in component:
                component["parameters"] = {}
                record(
                    f"loss.{component_name}.parameters",
                    "inserted an empty parameter object",
                )

    optimization = canonical.get("optimization")
    if isinstance(optimization, dict) and isinstance(optimization.get("phases"), list):
        for index, phase in enumerate(optimization["phases"]):
            if not isinstance(phase, dict):
                continue
            optimizer = phase.get("optimizer")
            if isinstance(optimizer, dict) and isinstance(optimizer.get("name"), str):
                if _merge_legacy_parameters(phase, "parameters", optimizer.get("parameters")):
                    phase["optimizer"] = optimizer["name"]
                    record(
                        f"optimization.phases[{index}].optimizer",
                        "flattened {name, parameters} into optimizer and parameters",
                    )
            if "optimizer_parameters" in phase and _rename_legacy_parameters(
                phase, "optimizer_parameters", "parameters"
            ):
                record(
                    f"optimization.phases[{index}].optimizer_parameters",
                    "renamed optimizer_parameters to parameters",
                )
            if "parameters" not in phase:
                phase["parameters"] = {}
                record(
                    f"optimization.phases[{index}].parameters",
                    "inserted an empty optimizer parameter object",
                )
            scheduler = phase.get("scheduler")
            if scheduler is None:
                phase["scheduler"] = {"name": "none", "parameters": {}}
                record(
                    f"optimization.phases[{index}].scheduler",
                    "inserted the no-op scheduler",
                )
            elif isinstance(scheduler, dict) and "parameters" not in scheduler:
                scheduler["parameters"] = {}
                record(
                    f"optimization.phases[{index}].scheduler.parameters",
                    "inserted an empty parameter object",
                )

    training = canonical.get("training")
    if isinstance(training, dict):
        if str(training.get("batch_mode") or "").casefold() == "full_batch":
            if "batch_size" not in training:
                training["batch_size"] = None
                record("training.batch_size", "inserted required null full-batch sentinel")
            elif isinstance(training.get("batch_size"), (int, float)) and not isinstance(
                training.get("batch_size"), bool
            ):
                training["batch_size"] = None
                record(
                    "training.batch_size",
                    "discarded incompatible numeric batch size for the full-batch-only trainer",
                )
        for component_name in (
            "gradient_clipping",
            "early_stopping",
            "time_strategy",
            "initial_state_pretraining",
        ):
            component = training.get(component_name)
            if isinstance(component, dict) and "parameters" not in component:
                component["parameters"] = {}
                record(
                    f"training.{component_name}.parameters",
                    "inserted an empty parameter object",
                )
        extra = training.get("extra_parameters")
        if isinstance(extra, dict):
            for section_name in ("physics_validation", "best_checkpoint"):
                section = extra.get(section_name)
                if isinstance(section, dict) and _flatten_nested_parameters(section):
                    record(
                        f"training.extra_parameters.{section_name}.parameters",
                        "flattened policy parameters beside the policy selector",
                    )
            adaptive_control = extra.get("adaptive_control")
            if isinstance(adaptive_control, dict):
                for section_name in (
                    "lbfgs_stall_recovery",
                    "sampling_control",
                    "curriculum_control",
                    "rollback",
                ):
                    section = adaptive_control.get(section_name)
                    if isinstance(section, dict) and _flatten_nested_parameters(section):
                        record(
                            "training.extra_parameters.adaptive_control."
                            f"{section_name}.parameters",
                            "flattened controller parameters beside the controller selector",
                        )

    return canonical, repairs


def _flatten_nested_parameters(container: dict[str, Any]) -> bool:
    """Flatten an unambiguous policy ``parameters`` object in place.

    Adaptive-policy schemas store their option parameters beside ``strategy`` or
    ``action``. LLMs commonly apply the regular ``{name, parameters}`` component
    shape here instead. The rewrite is representation-only when it does not
    overwrite or contradict an explicitly supplied flat value.
    """

    parameters = container.get("parameters")
    if not isinstance(parameters, dict):
        return False
    if any(name in container and container[name] != value for name, value in parameters.items()):
        return False
    for name, value in parameters.items():
        container.setdefault(name, value)
    container.pop("parameters")
    return True


def _merge_legacy_parameters(
    container: dict[str, Any], target_name: str, legacy_value: Any
) -> bool:
    """Merge a legacy parameter object only when no value would be overwritten."""

    if legacy_value is None:
        legacy_value = {}
    if not isinstance(legacy_value, dict):
        return False
    target = container.get(target_name)
    if target is None:
        container[target_name] = deepcopy(legacy_value)
        return True
    if not isinstance(target, dict):
        return False
    conflicts = {
        key for key, value in legacy_value.items() if key in target and target[key] != value
    }
    if conflicts:
        return False
    target.update(deepcopy(legacy_value))
    return True


def _rename_legacy_parameters(
    container: dict[str, Any], legacy_name: str, target_name: str
) -> bool:
    legacy_value = container.get(legacy_name)
    if not _merge_legacy_parameters(container, target_name, legacy_value):
        return False
    container.pop(legacy_name, None)
    return True


def _apply_role_metadata(spec: dict[str, Any], role: dict[str, Any]) -> None:
    metadata = spec.get("generation_metadata")
    if not isinstance(metadata, dict):
        return
    metadata.update(
        {
            "candidate_role": role["name"],
            "role_design_objective": role["objective"],
            "role_requirements_applied": list(role["requirements"]),
            "role_input_focus": list(role.get("input_focus") or []),
            "role_modifiable_modules": list(role.get("modifiable_modules") or []),
        }
    )
    if role.get("primary_axis"):
        metadata.update(
            {
                "primary_diversity_axis": role["primary_axis"],
                "diversity_hypothesis": metadata.get("diversity_hypothesis")
                or f"Materially differentiate the executable design along {role['primary_axis']}.",
                "major_executable_differences_expected": list(role.get("major_modules") or []),
            }
        )


def _design_messages(
    payload: dict[str, Any],
    principles: dict[str, Any] | None = None,
    posterior_principle_evidence: list[dict[str, Any]] | None = None,
    posterior_failure_patterns: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    execution_feedback = bool(payload.get("execution_feedback", True))
    evolutionary_inheritance = bool(
        payload.get("evolutionary_inheritance_enabled", True)
    )
    option_registry = payload.get("option_registry") or llm_algorithm_spec_option_registry()
    role = resolve_candidate_role(payload)
    principles = principles if principles is not None else _resolve_design_principles(payload)
    posterior_principle_evidence = (
        posterior_principle_evidence
        if posterior_principle_evidence is not None
        else resolve_posterior_principle_evidence(payload, principles)
    )
    posterior_failure_patterns = (
        posterior_failure_patterns
        if posterior_failure_patterns is not None
        else resolve_posterior_failure_patterns(payload)
    )
    role_name = str(role["name"]) if role else str(payload.get("candidate_role") or "")
    role_parent_context = (
        _role_specific_parent_context(
            payload.get("parent_specs") or [],
            role_name,
            execution_feedback=execution_feedback,
        )
        if evolutionary_inheritance
        else []
    )
    run_posterior_context = _compact_run_posterior_context(
        (payload.get("run_posterior_context") or {}) if execution_feedback else {},
        parent_candidate_ids={
            str(item.get("candidate_id"))
            for item in role_parent_context
            if item.get("candidate_id")
        },
        include_parent_algorithm_specs=evolutionary_inheritance,
    )
    design_template = algorithm_spec_template()
    # The runtime template remains a useful Adam -> L-BFGS integration fixture,
    # but presenting that particular pairing as the LLM's structural example
    # biases every proposal toward an unnecessary L-BFGS phase.  Design starts
    # from one main phase; a final fine-tuning phase must be chosen explicitly.
    design_template["optimization"]["phases"] = [
        deepcopy(design_template["optimization"]["phases"][0])
    ]
    design_template["generation_metadata"].update(
        {
            "fine_tuning_enabled": False,
            "fine_tuning_optimizer": None,
            "fine_tuning_decision_basis": [
                "No dedicated final refinement phase is enabled in this structural example."
            ],
        }
    )
    design_template = _problem_aware_design_template(
        design_template,
        payload.get("problem_spec") or {},
    )
    context = {
        "problem_features": payload.get("problem_features") or {},
        "problem_spec": payload.get("problem_spec") or {},
        "pinnacle_profile": payload.get("pinnacle_profile") or {},
        "pinnacle_algorithm_parameter_space": payload.get("pinnacle_algorithm_parameter_space") or {},
        "relevant_prior_knowledge": _compact_prior_knowledge(payload.get("prior_knowledge") or []),
        "macro_physics_principles": principles["macro_physics_principles"],
        "architecture_design_principles": principles["architecture_design_principles"],
        "posterior_principle_evidence": posterior_principle_evidence,
        "posterior_failure_patterns": posterior_failure_patterns,
        "latest_generation_posterior": run_posterior_context.get("latest_generation_posterior") or {},
        "previous_generation_candidates": run_posterior_context.get("previous_generation_candidates") or [],
        "previous_generation_failures": run_posterior_context.get("previous_generation_failures") or [],
        "active_run_posterior_rules": run_posterior_context.get("active_run_posterior_rules") or [],
        "posterior_recommended_next_actions": run_posterior_context.get("recommended_next_actions") or [],
        "posterior_unresolved_hypotheses": run_posterior_context.get("unresolved_hypotheses") or [],
        "posterior_context_metadata": run_posterior_context.get("metadata") or {},
        "generation_feedback": _compact_generation_feedback(
            payload.get("generation_feedback") or {},
            include_candidate_ids=evolutionary_inheritance,
        ),
        "search_stage": payload.get("search_stage") or {},
        "experiment_budget": payload.get("experiment_budget") or {},
        "trainer_capabilities_snapshot": payload.get(
            "trainer_capabilities_snapshot"
        ) or {},
        "algorithm_spec_contract": {
            "required_top_level_fields": list(REQUIRED_TOP_LEVEL_FIELDS),
            "complete_template": design_template,
            "fine_tuning_contract": {
                "default": "disabled",
                "representation": (
                    "When enabled, append one dedicated final optimization phase and identify it "
                    "in generation_metadata; when disabled, do not append a phase merely to follow "
                    "a conventional Adam-to-L-BFGS recipe."
                ),
                "allowed_optimizers": sorted(
                    (option_registry.get("fields") or {})
                    .get("optimization.phases[].optimizer", {})
                    .get("options", {})
                ),
                "decision_inputs": [
                    "PDE conditioning and smoothness",
                    "derivative and loss stability",
                    "low-fidelity or parent convergence evidence",
                    "memory and iteration budget",
                    "optimizer compatibility constraints",
                ],
                "lbfgs_special_case": (
                    "L-BFGS is one optional final optimizer with full-batch, final-position, "
                    "objective-freeze, and stall-recovery constraints; it is not the definition "
                    "of fine-tuning."
                ),
            },
            "strict_rules": [
                "Use spec_version, never schema_version, inside algorithm_spec.",
                "Use string-valued architecture and optimizer fields; put their settings in parameters.",
                "When training.batch_mode is full_batch, training.batch_size must be JSON null.",
            ],
            "preflight_checklist": [
                "Return exactly one proposal with the supplied candidate_id.",
                "Include every field shown in complete_template; do not return a partial patch.",
                "Use only option_registry names and parameter keys.",
                "Use no legacy schema_version, architecture_parameters, optimizer_parameters, or loss_function_parameters fields.",
                "Use finite numeric values and satisfy all hard sampling, model, and iteration constraints.",
                "For the full-batch-only trainer, set batch_mode to full_batch and batch_size to JSON null.",
                "Select adaptive policies only from trainer_capabilities_snapshot.eligible_adaptive_policies.",
                "Decide whether fine-tuning is justified; keep its metadata consistent with the final phase.",
                "Return a complete AlgorithmSpec, never a runtime controller action or state patch.",
            ],
        },
        "option_registry": option_registry,
        "parent_specs": role_parent_context,
    }
    if not execution_feedback:
        for key in (
            "posterior_principle_evidence",
            "posterior_failure_patterns",
            "latest_generation_posterior",
            "previous_generation_candidates",
            "previous_generation_failures",
            "active_run_posterior_rules",
            "posterior_recommended_next_actions",
            "posterior_unresolved_hypotheses",
            "posterior_context_metadata",
            "generation_feedback",
        ):
            context.pop(key, None)
    # Top-3 executable parents have one canonical prompt location.  The
    # run-scoped memory may retain its own archived copy, but injecting it here
    # as well would duplicate the largest part of every evolution prompt.
    if not role_parent_context:
        context["previous_top3"] = run_posterior_context.get("previous_top3") or []
    if execution_feedback and role_name == "architecture_optimization_guided_design":
        context["optimization_role_evidence"] = {
            "parents": [
                {
                    "candidate_id": parent.get("candidate_id"),
                    "rank": parent.get("rank"),
                    "evidence_location": "parent_specs[].optimization_feedback",
                }
                for parent in role_parent_context
            ],
            "instructions": [
                "Separate objective-definition problems, loss or gradient balancing problems, and optimizer or schedule problems.",
                "Distinguish loss-value imbalance, gradient-norm imbalance, and gradient-direction conflict.",
                "Do not add or remove physical loss terms unless the registered role boundary permits it.",
                "Prefer dynamic weighting, optimizer, scheduler, or training-stage changes when measured evidence supports an optimization failure.",
            ],
        }
    elif execution_feedback and role_name == "physics_constraint_sampling_guided_design":
        context["constraint_sampling_role_evidence"] = {
            "parents": [
                {
                    "candidate_id": parent.get("candidate_id"),
                    "rank": parent.get("rank"),
                    "evidence_location": "parent_specs[].physical_supervision_summary",
                }
                for parent in role_parent_context
            ],
            "instructions": [
                "Residual regions are diagnostic summaries from fixed evaluator points.",
                "Use concentration, residual quantiles, and constraint errors together before choosing localized refinement or sampling-budget redistribution.",
                "Do not infer that one high maximum residual alone proves adaptive sampling is necessary.",
                "Preserve global domain coverage and true boundary and initial-condition values.",
            ],
        }
    adaptive_role_focus = {
        "elite_conservative_improvement": [
            "Preserve adaptive policies with successful observations and reduce parameters that caused repeated rollback.",
            "Do not add several new control mechanisms in one conservative proposal.",
        ],
        "physics_constraint_sampling_guided_design": [
            "Inspect persistent worst registered physics components before changing weighting, sampling, or curriculum timing.",
            "Distinguish premature curriculum advance from insufficient physical supervision.",
        ],
        "architecture_optimization_guided_design": [
            "Do not attribute all final error to Trainer controllers; stable controllers with high error can indicate representation or capacity limits.",
            "Coordinate any representation change with gradient diagnostics, optimizer stages, and learning-rate scheduling.",
        ],
        "multi_parent_synthesis": [
            "Compare which adaptive policies repeatedly succeeded, rolled back, or incurred less overhead across the Top-3 before inheritance.",
        ],
        "novelty_exploration": [
            "A different eligible adaptive policy is allowed, but it must obey the capability summary and compatibility matrix.",
        ],
    }
    if execution_feedback and role_name in adaptive_role_focus:
        context["adaptive_policy_role_focus"] = adaptive_role_focus[role_name]
    if not payload.get("initialization_mode") and evolutionary_inheritance:
        context["top3_evolution_contract"] = {
            "source": (
                "global_best_tracker_top3"
                if execution_feedback
                else "global_best_tracker_selection_with_quality_and_order_hidden"
            ),
            "parent_count": 3,
            "all_roles_receive_full_top3": True,
            "rank1_primary_for_role": (
                "elite_conservative_improvement" if execution_feedback else None
            ),
            "python_field_splicing_forbidden": True,
            "all_children_must_be_new_after_validator_normalization": True,
            "allowed_variation_strategies": list(ALLOWED_VARIATION_STRATEGIES),
        }
    elif not payload.get("initialization_mode"):
        context["independent_search_contract"] = {
            "parent_algorithm_specs_available": False,
            "parent_ids_available": False,
            "parent_selection_enabled": False,
            "mutation_enabled": False,
            "multi_parent_synthesis_enabled": False,
            "execution_feedback_summary_enabled": execution_feedback,
            "run_scoped_posterior_enabled": bool(
                context.get("latest_generation_posterior")
                or context.get("active_run_posterior_rules")
            ),
            "require_fresh_complete_algorithm_spec": True,
        }
    problem_id = str(
        (payload.get("problem_features") or {}).get("problem_id")
        or (payload.get("problem_spec") or {}).get("problem_id")
        or ""
    ).casefold()
    if problem_id == "burgers_1d":
        context["anti_trivial_solution_contract"] = {
            "risk": (
                "u approximately 0 can satisfy the Burgers PDE residual and homogeneous boundary "
                "condition while violating the nonzero initial condition"
            ),
            "failure_signature": (
                "relative_l2 near 1, initial RMSE near sqrt(1/2), low prediction variance, "
                "and deceptively small PDE/boundary residuals"
            ),
            "required_for_soft_penalty": [
                "include positive-weight pde_residual, boundary_condition, and initial_condition terms",
                "use positive interior, boundary, and initial sample counts",
                "choose constraint weights or adaptive weighting that protect the initial condition",
            ],
            "registered_hard_alternative": {
                "method": "problem_hard",
                "transform_id": "burgers_1d_initial_dirichlet_exact",
                "effect": "enforces u(x,0)=-sin(pi*x) and u(-1,t)=u(1,t)=0 by construction",
            },
            "selection_rule": (
                "A low PDE residual is not evidence of success when the initial condition is forgotten "
                "or prediction variance collapses."
            ),
        }
    if problem_id == "wave_1d":
        context["wave_loss_balance_contract"] = {
            "required_stable_loss_terms": [
                "pde_residual",
                "spatial_boundary",
                "initial_displacement",
                "initial_velocity",
            ],
            "independent_static_weight_capability": (
                "Required Wave1D loss terms: pde_residual, "
                "spatial_boundary, initial_displacement, initial_velocity. "
                "Each required term must appear exactly once. Each term has "
                "an independent positive static weight. The weight is static "
                "during training unless an explicitly selected "
                "dynamic-weighting strategy is enabled."
            ),
            "independent_weight_search_range": [0.01, 100.0],
            "balancing_methods": {
                "dynamic_weighting": ["none", "lra", "ntk"],
                "optimizer": "multiadam",
                "multiadam_grouping": "dirichlet_vs_non_dirichlet",
            },
            "first_exploration_exclusions": [
                "Do not combine NTK and MultiAdam.",
                "Do not combine LRA and MultiAdam.",
                "Do not append L-BFGS to a MultiAdam phase.",
            ],
            "diagnostic_priority": (
                "When amplitude collapse or initial displacement error is high, "
                "first adjust loss/constraint balancing rather than immediately "
                "replacing the network."
            ),
            "current_reliable_parent": {
                "name": "R1 independent static weights",
                "weights": {
                    "pde_residual": 1.0,
                    "spatial_boundary": 40.0,
                    "initial_displacement": 12.0,
                    "initial_velocity": 8.0,
                },
                "evidence": (
                    "Wave1D is highly sensitive to the spatial-boundary "
                    "weight; larger is not automatically better."
                ),
            },
            "candidate_scope": {
                "static_weight_candidate": (
                    "Change only the four required loss weights; do not "
                    "change network, sampling, optimizer, or learning rate."
                ),
                "multiadam_candidate": (
                    "Change only learning rate, betas, group weights, and "
                    "the four required static loss weights."
                ),
            },
            "validity_rule": (
                "A low PDE residual alone is not a valid solution. Check "
                "amplitude ratio, phase error, initial displacement, initial "
                "velocity, and solution validity together."
            ),
            "ntk_status": (
                "NTK is experimental. It is successful only when the strict "
                "residual-sum implementation passes numerical parity and "
                "does not exhibit extreme weight oscillation."
            ),
            "neutrality": (
                "NTK and MultiAdam are alternatives to test; neither is "
                "presumed superior."
            ),
        }
    if payload.get("initial_population_role_map"):
        context["initial_population_role_map"] = deepcopy(payload["initial_population_role_map"])
    candidate_task = {
        "candidate_task": {
            "candidate_id": payload.get("candidate_id"),
            "candidate_slot": payload.get("candidate_slot"),
            "candidate_role": role["name"] if role else payload.get("candidate_role"),
            "role_objective": role["objective"] if role else payload.get("candidate_role_objective"),
            "role_requirements": role["requirements"] if role else [],
            "role_input_focus": role.get("input_focus") if role else [],
            "role_modifiable_modules": role.get("modifiable_modules") if role else [],
            "candidate_diversity_axis": (
                role.get("primary_axis") if role else payload.get("candidate_diversity_axis")
            ),
            "proposal_type": payload.get("proposal_type") or "proposal",
            "generation_mode": deepcopy(
                payload.get("generation_mode")
                or {
                    "state": "initialization"
                    if payload.get("initialization_mode")
                    else "normal",
                    "stagnation_detected": False,
                    "trigger_reason": None,
                    "behavior_overrides": {},
                }
            ),
            "generation_policy_audit": deepcopy(
                payload.get("generation_policy_audit") or {}
            ) if execution_feedback else {},
            "stagnation_requirements": (
                list(role.get("stagnation_requirements") or [])
                if role
                and bool(
                    (payload.get("generation_mode") or {}).get(
                        "stagnation_detected"
                    )
                )
                else []
            ),
        }
    }
    if not execution_feedback:
        candidate_task["candidate_task"].pop("generation_policy_audit", None)
    if not payload.get("initialization_mode"):
        if evolutionary_inheritance:
            candidate_task["candidate_task"]["proposal_metadata_contract"] = {
                "placement": "sibling of algorithm_spec, never inside executable AlgorithmSpec",
                "required_fields": [
                    "role",
                    "parent_ids",
                    "variation_strategy",
                    "inherited_strengths",
                    "major_changes",
                    "design_rationale",
                    "validation",
                ],
                "allowed_variation_strategies": list(ALLOWED_VARIATION_STRATEGIES),
                "parent_ids_must_come_from": [
                    item.get("candidate_id") for item in (payload.get("parent_specs") or [])[:3]
                ],
                "role_specific_lineage": (
                    "Rank-1 must be the primary/included parent; ranks 2 and 3 are supporting context."
                    if execution_feedback and role and role["name"] == "elite_conservative_improvement"
                    else (
                        "Use at least two unordered parents."
                        if role and role["name"] == "multi_parent_synthesis"
                        else "Inspect all three unordered parents; declare the parent IDs actually used."
                    )
                ),
            }
        else:
            candidate_task["candidate_task"]["proposal_metadata_contract"] = {
                "placement": "sibling of algorithm_spec, never inside executable AlgorithmSpec",
                "required_fields": [
                    "role",
                    "major_changes",
                    "design_rationale",
                    "validation",
                ],
                "forbidden_fields": [
                    "parent_ids",
                    "variation_strategy",
                    "inherited_strengths",
                ],
                "lineage": "parent_free_independent_proposal",
            }
    if role and role.get("principle_focus"):
        candidate_task["candidate_task"]["role_principle_context"] = {
            group: deepcopy(principles.get(group) or [])
            for group in role["principle_focus"]
        }
    if payload.get("duplicate_retry"):
        candidate_task["duplicate_retry"] = deepcopy(payload["duplicate_retry"])
    if payload.get("diversity_retry"):
        candidate_task["diversity_retry"] = deepcopy(payload["diversity_retry"])
    if payload.get("generation_retry"):
        candidate_task["generation_retry"] = deepcopy(payload["generation_retry"])
    if payload.get("availability_fill"):
        candidate_task["availability_fill"] = deepcopy(payload["availability_fill"])
    recommended_iterations = (payload.get("experiment_budget") or {}).get(
        "recommended_total_iterations"
    )
    experiment_budget = payload.get("experiment_budget") or {}
    maximum_generations = int(experiment_budget.get("maximum_generations") or 10)
    maximum_sampling_points = (payload.get("experiment_budget") or {}).get("maximum_sampling_points")
    maximum_model_parameters = int((payload.get("experiment_budget") or {}).get("maximum_model_parameters", 0))
    iteration_recommendation = (
        "HARD LOW-FIDELITY EXECUTION BUDGET: this candidate will execute exactly "
        f"{int(recommended_iterations)} optimizer iterations after the trusted controller proportionally "
        "normalizes every enabled optimization phase. Return a coherent positive multi-stage schedule; enabled "
        "stages, including L-BFGS, must remain meaningful after scaling. "
        if recommended_iterations is not None
        else ""
    )
    search_horizon_guidance = (
        f"STAGNATION-AWARE SEARCH HORIZON: the controller executes at least 4 and at most "
        f"{maximum_generations} low-fidelity generations. It may schedule one explicit escape generation only "
        "after all configured best-MSE, cumulative-improvement, and Top-3-median stagnation conditions hold. "
        "The independent final high-fidelity review generates no new AlgorithmSpec and is not an evolution generation. "
    ) if execution_feedback else (
        f"SEARCH HORIZON: the controller executes at least 4 and at most {maximum_generations} "
        "low-fidelity generations. Controller decisions and execution outcomes are intentionally hidden. "
        "The independent final high-fidelity review generates no new AlgorithmSpec. "
    )
    sampling_budget_requirement = (
        "NO GLOBAL SAMPLING-POINT LIMIT: there is no external cap on "
        "interior.n_points + boundary.n_points + initial.n_points. Choose sample counts from the PDE physics, "
        "numerical resolution needs, and the executable sampling strategy. "
        if maximum_sampling_points is None
        else (
            "HARD SAMPLING BUDGET: interior.n_points + boundary.n_points + initial.n_points "
            f"MUST be <= {int(maximum_sampling_points)}; this is one shared total, not a separate limit for each sampler. "
        )
    )
    generation_zero_contract = (
        "GENERATION-0 DIVERSITY CONTRACT: You are designing exactly one member of an eight-candidate initial population. "
        "Your candidate has a distinct assigned design role and primary diversity axis. Maximize meaningful executable "
        "diversity from the other role objectives while remaining physically justified, trainable, registry-valid, and "
        "within all hard budgets. Distinctness must come from normalized executable AlgorithmSpec fields, not candidate "
        "IDs, prose, metadata, aliases, or tiny numeric perturbations. Do not randomly combine components merely to appear "
        "different. Every major difference must have a physical, architectural, numerical, or optimization rationale. "
        "The role map describes objectives only; it contains no sibling AlgorithmSpecs. Return exactly one complete proposal. "
        "Record candidate_role, role_design_objective, primary_diversity_axis, diversity_hypothesis, and "
        "major_executable_differences_expected in generation_metadata. "
        if payload.get("initialization_mode")
        else ""
    )
    availability_contract = (
        "AVAILABILITY FILL CONTRACT: Candidate availability is more important than maximum diversity. "
        "Return one complete, valid, trainable AlgorithmSpec for the assigned slot. Meaningful similarity is acceptable, "
        "but it must differ from every accepted current-generation normalized executable AlgorithmSpec in at least one "
        "executable decision. Preserve the assigned candidate ID and role. "
        if payload.get("availability_fill")
        else ""
    )
    evolution_contract = (
        "GLOBAL TOP-3 FIVE-ROLE EVOLUTION CONTRACT: You receive the controller-selected global low-fidelity "
        "distinct Top-3 accumulated through the completed generations, "
        "including each parent's complete normalized AlgorithmSpec, metrics, training_summary, success_factors, "
        "and failure_factors. Inspect all three before designing. You autonomously choose parent usage and one "
        "allowed variation_strategy; do not ask Python to splice fields or apply a hard-coded genetic operator. "
        "The result must be a genuinely new normalized executable AlgorithmSpec: it must not equal any global Top-3 "
        "parent or any accepted sibling. Metadata-only changes, identifiers, prose, aliases, and hash restoration "
        "do not count as novelty. The elite_conservative_improvement role must include global Rank-1 as its primary parent "
        "and make a conservative but executable change. The multi_parent_synthesis role must use at least two parents. "
        "Use candidate_task.role_input_focus as the role's primary evidence and keep primary executable changes within "
        "candidate_task.role_modifiable_modules when that list is non-empty; any supporting cross-module change must be "
        "necessary, compatible, and explicitly justified. "
        "Treat candidate_task.generation_mode, candidate_task.generation_policy_audit, and "
        "candidate_task.stagnation_requirements as the exact audited policy for this request. Meet the required major-module "
        "change count, parent-diversity, failed-mechanism-avoidance, and replay restrictions recorded there. "
        "Return proposal metadata as fields beside algorithm_spec: role, parent_ids, variation_strategy, "
        "inherited_strengths, major_changes, design_rationale, and validation. This metadata is audit-only and is "
        "never passed to the Trusted Builder. "
        if not payload.get("initialization_mode")
        else ""
    )
    if not payload.get("initialization_mode") and not evolutionary_inheritance:
        evolution_contract = (
            "BUDGET-MATCHED INDEPENDENT SEARCH CONTRACT: This request receives the same PDE, prior knowledge, "
            "aggregate execution feedback, run-scoped posterior memory, 1,000-step low-fidelity budget, and search "
            "stage used by the full method, but it receives no parent AlgorithmSpec or parent ID. Generate a fresh "
            "complete AlgorithmSpec from the assigned independent role. Use measured summaries to diagnose what kinds "
            "of physical, constraint, representation, or optimization problems remain, but do not reconstruct, mutate, "
            "inherit, splice, or recombine any previous candidate. Do not request hidden parent details. Return audit "
            "metadata containing role, major_changes, design_rationale, and validation only."
        )
    elif not payload.get("initialization_mode") and not execution_feedback:
        evolution_contract = (
            "UNORDERED-PARENT FIVE-ROLE EVOLUTION CONTRACT: You receive exactly three parent AlgorithmSpecs "
            "selected by the controller, but their quality, measurements, ranks, selection reason, and original "
            "order are hidden. Treat all list positions as semantically equal. Inspect all three declared designs, "
            "choose parent usage and one allowed variation_strategy, and do not ask Python to splice fields. "
            "The result must be a genuinely new normalized executable AlgorithmSpec rather than a metadata-only "
            "change. The multi_parent_synthesis role must use at least two parents. Justify every decision only "
            "from the PDE contract, supplied prior knowledge, numerical principles, and AlgorithmSpec structure. "
            "Do not infer or claim that a parent performed well, poorly, converged, failed, or ranked above another. "
            "Return proposal metadata beside algorithm_spec for audit; it is never passed to the Trusted Builder. "
        )
    if payload.get("initialization_mode"):
        output_structure = (
            "{\"proposals\":[{\"candidate_id\":\"<supplied candidate_id>\","
            "\"algorithm_spec\":{...}}]}. "
        )
    elif evolutionary_inheritance:
        output_structure = (
            "{\"proposals\":[{\"candidate_id\":\"<supplied candidate_id>\",\"role\":\"<supplied role>\","
            "\"parent_ids\":[...],\"variation_strategy\":\"<allowed value>\",\"inherited_strengths\":[...],"
            "\"major_changes\":[...],\"design_rationale\":\"...\",\"validation\":{...},"
            "\"algorithm_spec\":{...}}]}. "
        )
    else:
        output_structure = (
            "{\"proposals\":[{\"candidate_id\":\"<supplied candidate_id>\",\"role\":\"<supplied role>\","
            "\"major_changes\":[...],\"design_rationale\":\"...\",\"validation\":{...},"
            "\"algorithm_spec\":{...}}]}. "
        )
    system = (
        "You are an open-ended PINN algorithm designer. Generate complete AlgorithmSpec JSON objects. "
        f"{generation_zero_contract}"
        f"{availability_contract}"
        f"{evolution_contract}"
        "Compose each AlgorithmSpec dynamically from independent field choices; never select from or enumerate a catalog of complete algorithms. "
        "The authoritative legal choices, per-option parameter schemas, defaults, ranges, aliases, and compatibility rules are in "
        "option_registry, which is a compact view exported by option_registry.py. Use those field options; do not generate Python source components. "
        "Do not output Python code. Do not modify the PDE equation, initial/boundary conditions, evaluation metrics, budget, or random seed. "
        "Parameter values are continuous or schema-constrained choices, not values from a predefined full-algorithm grid. "
        "Activation, output activation, initialization, input transform, weighting, aggregation, and scheduler "
        "components use {name, parameters}; samplers use {strategy, parameters}. "
        "STRUCTURAL CONTRACT: copy the shape and field names from algorithm_spec_contract.complete_template. "
        "Inside algorithm_spec use spec_version, never schema_version. network.architecture and every "
        "optimization phase optimizer are strings; their settings belong in extra_parameters and parameters, "
        "respectively. Do not emit legacy architecture_parameters, optimizer_parameters, or "
        "loss_function_parameters fields. The current trainer is full-batch only, so training.batch_mode MUST "
        "be 'full_batch' and training.batch_size MUST be JSON null, never 0 or a numeric batch size. "
        f"All widths must be positive integers; iterations and sample counts non-negative integers; continuous values finite. "
        f"{search_horizon_guidance}"
        f"{iteration_recommendation}"
        f"{sampling_budget_requirement}"
        f"HARD MODEL BUDGET: the estimated trainable parameter count MUST be <= {maximum_model_parameters}. "
        "For option entries, parameters lists the only legal parameter keys and their executable type/range/default rules. "
        "Alias entries must follow their canonical option. Value-schema entries constrain fields that are not named options. "
        "Activation, initializer, sampler, loss, weighting, aggregation, optimizer, and scheduler parameters must satisfy option_registry. "
        "RAR, RAD, and RAR-D belong in sampling.adaptive_refinement.strategy, while the base interior sampler remains an independent choice. "
        "For fourier_mlp use num_frequencies and sigma, never fourier_mapping_size or fourier_mapping_scale. "
        "All component names and parameter keys must match option_registry exactly. If no supported parameter is needed, use an empty parameters object. "
        "For constraint_component, component_name is mandatory and must be an exact constraint_id from problem_spec.constraints; emit one selector term for each required constraint and do not reuse a component_name. "
        "When problem_spec.metadata.required_network_architecture is present, network.architecture must equal it exactly because the executable problem contract enforces that architecture. "
        "When direct constraint IDs or constraint_component selectors are used as independent loss terms, use weighted_sum aggregation. "
        "At runtime an aggregate pde_residual is automatically expanded into one independently registered pde/<governing_law_id> loss per governing equation. "
        "Every expanded executable loss component ID must be unique: never combine an aggregate pde_residual or governing_residual term with a pde_component selector that targets any governing law already covered by that aggregate, and never repeat a component selector. "
        "With fixed weighting, the runtime first computes mean(weight_i * loss_i) separately for the governing and constraint groups, then applies the selected aggregation across the two group values; component count therefore does not increase a group's importance. "
        "Adaptive-policy option parameters are FLAT fields beside enabled plus action, strategy, or final_model_policy; never put a parameters object inside adaptive_control sections, physics_validation, or best_checkpoint. "
        "Selecting loss.weighting_strategy.name='gradient_balance' requires training.extra_parameters.adaptive_control.enabled=true. "
        "When pinnacle_profile is present, use its dimensions, equation count, constraint types, challenge tags, required_registry_parameters, and pinnacle_algorithm_parameter_space to select compatible components and numerical parameters; never change its physical parameters. "
        "CONSTRAINT-ENFORCEMENT FAIRNESS: constraint_enforcement is a public AlgorithmSpec search field, not a hidden repair. Choose only a method/transform_id pair listed in pinnacle_profile.constraint_enforcement_options. soft_penalty requires transform_id='none'; problem_hard requires an explicitly registered problem transform. All candidates have equal access to the same list, its computation counts toward the same budget, and any claimed improvement must identify the enforcement choice rather than attributing it only to network architecture. "
        "Posterior failure patterns are measured warnings. Do not reproduce an empirically failed parameter combination unchanged. "
        "OPTIONAL FINE-TUNING CONTRACT: fine-tuning is a role assigned to at most one dedicated final optimization phase, not an optimizer family. "
        "First decide from the PDE, conditioning, stability, budget, and measured parent/low-fidelity evidence whether a dedicated refinement phase is justified. "
        "If it is not justified, set generation_metadata.fine_tuning_enabled=false and fine_tuning_optimizer=null and do not append a conventional refinement phase automatically. "
        "If it is justified, append the dedicated phase, set fine_tuning_enabled=true, set fine_tuning_optimizer exactly equal to that final phase optimizer, and give a non-empty fine_tuning_decision_basis. "
        "Any optimizer registered in option_registry may serve this role when compatible; examples include a lower-learning-rate Adam/AdamW/RAdam/SGD/RMSprop phase as well as L-BFGS. "
        "Do not treat L-BFGS as mandatory. If L-BFGS is selected, obey its existing final-position, full-batch, frozen-objective, line-search, and recovery constraints. "
        "NONTRIVIAL-SOLUTION CONTRACT: when anti_trivial_solution_contract is present, treat it as a hard design requirement. "
        "For soft-penalty training, retain every required positive-weight physics and constraint term and positive sample coverage. "
        "Do not interpret a small PDE residual as success when the nonzero initial condition is violated or the prediction variance collapses. "
        "The registered problem-hard transform is a legal high-priority alternative, but do not invent an unregistered transform. "
        "PRINCIPLE-GUIDED DESIGN: first reason from macro_physics_principles about the immutable problem contract, conservation/causality, constraints, and relevant scales; then reason from architecture_design_principles about derivative compatibility, representation, capacity, conditioning, and cross-module compatibility. "
        "The supplied macro_physics_principles, architecture_design_principles, and posterior_principle_evidence have already been retrieved and ranked for the current problem. "
        "Do not assume omitted principles are invalid. Use only principles that are compatible with the immutable PDE contract and option_registry. "
        "Posterior evidence is local experimental evidence, not a universal law. When evidence conflicts, state the uncertainty and preserve exploration. "
        "RUN-SCOPED POSTERIOR CONTRACT: latest_generation_posterior, previous_top3, previous_generation_candidates, "
        "previous_generation_failures, and active_run_posterior_rules contain measured evidence only from the current PDE and current independent run. "
        "Treat every rule as soft, overridable guidance. Separate observed facts from inferred causes, do not make single-component causal claims when multiple modules differ, "
        "and state which measured training problem the proposal addresses. Generation 0 has an empty run-scoped posterior. "
        "For every proposal, generation_metadata must include macro_physics_principles_applied and architecture_design_principles_applied as lists of supplied principle_id values, plus principle_tradeoffs as a concise list explaining how physical fidelity, trainability, capacity, and cost were balanced. "
        "When candidate_task contains role_principle_context, treat those supplied principles as mandatory role-specific reasoning input: apply every relevant principle, translate it into concrete executable decisions, and record the applied principle IDs and tradeoffs in generation_metadata. "
        "generation_metadata may include posterior_evidence_applied, but every referenced evidence or principle ID must occur in the supplied context. "
        "If you propose a nearby combination, state the concrete stability mitigation and remaining NaN risk in generation_metadata.possible_risks. "
        "H->D FEEDBACK CONTRACT: when generation_feedback is non-empty, it is authoritative measured feedback from the immediately preceding generation. "
        "Read its reflection, directives, successful_candidates, and failed_candidates for every proposal type, including proposals without parents. "
        "Preserve measured strengths, do not repeat a measured failed AlgorithmSpec unchanged, and address at least one applicable directive. "
        "Record generation_feedback.source_generation in generation_metadata.feedback_source_generation and list the actions actually applied in "
        "generation_metadata.reflection_directives_applied. Feedback never overrides the PDE definition, legal component schemas, or hard budgets. "
        "generation_metadata must state the hypothesis, changed modules, expected advantages, and risks. "
        "ADAPTIVE-POLICY CONTRACT: use trainer_capabilities_snapshot to select only runtime-eligible policies. "
        "You need not enable every controller and should avoid stacking several expensive controllers without measured justification. "
        "For every enabled policy, state the measured training problem it addresses and choose only bounded registered parameters. "
        "Never name concrete runtime component IDs, use evaluator/reference metrics as Trainer conditions, bypass fixed sampling budgets, "
        "disable the L-BFGS objective freeze, or output a controller action. Adaptive-policy diversity is useful but not mandatory for every candidate. "
        "Treat adaptive_control_summary, runtime_scaled_policy_summary, and adaptive_compute_budget_summary as observed parent outcomes only: "
        "inherit policy design and initial hyperparameters, never final weights, points, curriculum level, optimizer state, EMA, or checkpoints. "
        "Generate exactly one complete AlgorithmSpec candidate for the supplied candidate_role. "
        "Return JSON only in this exact outer structure: "
        f"{output_structure}"
        "The proposals list MUST contain exactly one item. Do not return alternatives, multiple designs, commentary, "
        "Markdown, Python code, or an incomplete patch. The returned candidate_id MUST exactly match the supplied candidate_id. "
        "OUTPUT-SIZE CONTRACT: emit compact JSON and never repeat option_registry, parents, prompt context, the template, "
        "or explanatory analysis. Keep each audit prose list to at most three concise items and each prose item below "
        "160 characters. Spend output tokens on the one complete AlgorithmSpec, not on duplicated evidence.")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
        {"role": "user", "content": json.dumps(candidate_task, ensure_ascii=False)},
    ]


def _validation_repair_messages(
    payload: dict[str, Any], repair: dict[str, Any]
) -> list[dict[str, str]]:
    """Build a small correction request without replaying the full design context."""

    candidate_id = str(payload.get("candidate_id") or "")
    role_name = str(payload.get("candidate_role") or "")
    parent_ids = [
        str(item.get("candidate_id"))
        for item in payload.get("parent_specs") or []
        if isinstance(item, dict) and item.get("candidate_id")
    ]
    system = (
        "You repair one complete Open AlgorithmSpec that failed deterministic validation. "
        "Return JSON only. Preserve every valid executable decision and change only fields "
        "needed to resolve the supplied errors. Do not repeat parents, registries, templates, "
        "diagnostic histories, or explanatory prose. Return exactly one proposal as "
        '{"proposals":[{"candidate_id":"...","algorithm_spec":{...}}]}. '
        "The repaired algorithm_spec must remain complete, finite, trainable, and within the "
        "supplied hard budget."
    )
    context = {
        "candidate_id": candidate_id,
        "candidate_role": role_name,
        "allowed_parent_ids": parent_ids,
        "failure_reason": repair.get("reason"),
        "validation_errors": list(repair.get("validation_errors") or [])[:8],
        "failure_details": _bounded_prompt_value(
            repair.get("details") or [],
            max_dict_items=8,
            max_list_items=5,
            max_string_characters=240,
        ),
        "invalid_algorithm_spec": deepcopy(
            repair.get("rejected_algorithm_spec") or {}
        ),
        "experiment_budget": {
            key: value
            for key, value in (payload.get("experiment_budget") or {}).items()
            if key
            in {
                "required_total_iterations",
                "maximum_total_iterations",
                "maximum_sampling_points",
                "maximum_model_parameters",
                "constraint_resample_interval",
                "fidelity",
            }
        },
        "instruction": (
            "Correct the reported validation errors and return the entire repaired spec. "
            "Keep candidate_id unchanged."
        ),
    }
    task = {
        "candidate_id": candidate_id,
        "candidate_role": role_name,
        "request_mode": "validation_repair",
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
        {"role": "user", "content": json.dumps(task, ensure_ascii=False)},
    ]


def _problem_aware_design_template(
    template: dict[str, Any], problem_spec: dict[str, Any]
) -> dict[str, Any]:
    """Apply problem-owned architecture and constraint examples."""

    tailored = deepcopy(template)
    metadata = dict(problem_spec.get("metadata") or {})
    required_architecture = metadata.get("required_network_architecture")
    if required_architecture:
        tailored["network"]["architecture"] = str(required_architecture)
        tailored["network"]["extra_parameters"] = {}
    if "constraints" not in problem_spec:
        return tailored
    constraints = [
        item
        for item in problem_spec.get("constraints") or []
        if isinstance(item, dict)
        and item.get("constraint_id")
        and item.get("required", True)
    ]
    terms = [
        {
            "name": "pde_residual",
            "loss_function": "mse",
            "weight": 1.0,
            "parameters": {},
        }
    ]
    for constraint in constraints:
        terms.append(
            {
                "name": str(constraint["constraint_id"]),
                "loss_function": "mse",
                "weight": 10.0,
                "parameters": {},
            }
        )
    tailored["loss"]["terms"] = terms
    return tailored


def _role_specific_parent_context(
    parents: Any,
    role_name: str,
    *,
    execution_feedback: bool = True,
) -> list[dict[str, Any]]:
    """Keep only executable Top-3 specs and minimal role evidence under 20k tokens."""

    selected: list[dict[str, Any]] = []
    for raw_parent in list(parents or [])[:3]:
        if not isinstance(raw_parent, dict):
            continue
        if not execution_feedback:
            selected.append(
                {
                    "candidate_id": raw_parent.get("candidate_id"),
                    "generation": raw_parent.get("generation"),
                    "algorithm_spec": _compact_parent_algorithm_spec(
                        raw_parent.get("algorithm_spec") or {},
                        execution_feedback=False,
                    ),
                }
            )
            continue
        physical = raw_parent.get("physical_supervision_summary")
        optimization = raw_parent.get("optimization_feedback")
        parent = {
            "candidate_id": raw_parent.get("candidate_id"),
            "generation": raw_parent.get("generation"),
            "rank": raw_parent.get("rank"),
            "algorithm_spec": _compact_parent_algorithm_spec(
                raw_parent.get("algorithm_spec") or {}
            ),
            "fitness": raw_parent.get("fitness"),
            # metrics/formal_metrics are aliases in formal parent views.  Keep
            # one bounded copy instead of serializing both.
            "metrics": _compact_parent_metrics(
                raw_parent.get("formal_metrics") or raw_parent.get("metrics") or {}
            ),
        }
        if role_name == "physics_constraint_sampling_guided_design":
            parent["physical_supervision_summary"] = (
                _compact_physical_supervision_summary(physical)
            )
        elif role_name == "architecture_optimization_guided_design":
            parent["optimization_feedback"] = _compact_optimization_feedback(
                optimization
            )
        selected.append(parent)
    return _fit_parent_context_budget(selected)


def _compact_parent_algorithm_spec(
    value: Any, *, execution_feedback: bool = True
) -> dict[str, Any]:
    """Keep complete executable fields while bounding non-executable prose."""

    if not isinstance(value, dict):
        return {}
    spec = deepcopy(value)
    metadata = dict(spec.get("generation_metadata") or {})
    metadata_keys = [
        "candidate_role",
        "design_hypothesis",
        "diversity_hypothesis",
        "changed_modules",
        "expected_advantages",
        "possible_risks",
        "fine_tuning_enabled",
        "fine_tuning_optimizer",
    ]
    if execution_feedback:
        metadata_keys.extend(
            ["feedback_source_generation", "reflection_directives_applied"]
        )
    compact_metadata = {
        key: _bounded_prompt_value(
            metadata[key],
            max_dict_items=8,
            max_list_items=3,
            max_string_characters=240,
            remaining_depth=5,
        )
        for key in metadata_keys
        if key in metadata
    }
    spec["generation_metadata"] = compact_metadata
    return spec


def _compact_run_posterior_context(
    value: Any,
    *,
    parent_candidate_ids: set[str],
    include_parent_algorithm_specs: bool = True,
) -> dict[str, Any]:
    """Remove Gen1+ evidence duplicated by canonical Top-3 parent context."""

    if not isinstance(value, dict):
        return {}

    def strip_hereditary_fields(item: Any) -> Any:
        """Keep outcome summaries while removing every reconstructable lineage field."""

        if include_parent_algorithm_specs:
            return item
        return _strip_independent_lineage_fields(item)

    def candidate_view(item: Any) -> dict[str, Any]:
        if not isinstance(item, dict):
            return {}
        candidate_id = item.get("candidate_id") or item.get("spec_id")
        metrics = item.get("final_metrics") or item.get("metrics") or {}
        view = {
            "rank": item.get("rank"),
            "final_metrics": _compact_parent_metrics(metrics),
            "failure_flags": list(item.get("failure_flags") or [])[:4],
            "observed_facts": _bounded_prompt_value(
                list(item.get("observed_facts") or [])[:4],
                max_string_characters=240,
            ),
            "recommended_adjustments": _bounded_prompt_value(
                list(item.get("recommended_adjustments") or [])[:3],
                max_string_characters=240,
            ),
        }
        if include_parent_algorithm_specs:
            view["candidate_id"] = candidate_id
        return view

    previous_candidates = []
    for item in value.get("previous_generation_candidates") or []:
        compact = candidate_view(item)
        if str(compact.get("candidate_id") or "") in parent_candidate_ids:
            # The complete parent and its formal metrics already have one
            # canonical prompt location in parent_specs.
            continue
        if compact:
            previous_candidates.append(compact)
        if len(previous_candidates) >= 4:
            break

    previous_failures = [
        candidate_view(item)
        for item in (value.get("previous_generation_failures") or [])[:4]
        if isinstance(item, dict)
    ]
    latest = strip_hereditary_fields(
        _bounded_prompt_value(
            value.get("latest_generation_posterior") or {},
            max_dict_items=12,
            max_list_items=5,
            max_string_characters=240,
            remaining_depth=5,
        )
    )
    rules = [
        strip_hereditary_fields(
            _bounded_prompt_value(
                item,
                max_dict_items=10,
                max_list_items=4,
                max_string_characters=240,
                remaining_depth=4,
            )
        )
        for item in (value.get("active_run_posterior_rules") or [])[:8]
    ]
    compact_context = {
        "latest_generation_posterior": latest,
        "previous_generation_candidates": previous_candidates,
        "previous_generation_failures": previous_failures,
        "active_run_posterior_rules": rules,
        "recommended_next_actions": strip_hereditary_fields(
            _bounded_prompt_value(
                list(value.get("recommended_next_actions") or [])[:5]
            )
        ),
        "unresolved_hypotheses": strip_hereditary_fields(
            _bounded_prompt_value(
                list(value.get("unresolved_hypotheses") or [])[:5]
            )
        ),
        "metadata": {
            key: (value.get("metadata") or {}).get(key)
            for key in (
                "token_estimate",
                "original_token_estimate",
                "compressed",
                "budget_overflow",
            )
            if (value.get("metadata") or {}).get(key) is not None
        },
    }
    if not parent_candidate_ids:
        compact_context["previous_top3"] = [
            {
                **candidate_view(item),
                **(
                    {
                        "algorithm_spec": _compact_parent_algorithm_spec(
                            item.get("algorithm_spec") or {}
                        )
                    }
                    if include_parent_algorithm_specs
                    else {}
                ),
            }
            for item in (value.get("previous_top3") or [])[:3]
            if isinstance(item, dict)
        ]
    return compact_context


def _bounded_prompt_value(
    value: Any,
    *,
    max_dict_items: int = 16,
    max_list_items: int = 5,
    max_string_characters: int = 512,
    remaining_depth: int = 8,
) -> Any:
    """Deterministically bound diagnostic prose without changing executable specs."""

    if remaining_depth <= 0:
        return "<depth-limited>"
    if isinstance(value, str):
        if len(value) <= max_string_characters:
            return value
        return value[: max_string_characters - 3] + "..."
    if isinstance(value, dict):
        return {
            str(key): _bounded_prompt_value(
                item,
                max_dict_items=max_dict_items,
                max_list_items=max_list_items,
                max_string_characters=max_string_characters,
                remaining_depth=remaining_depth - 1,
            )
            for key, item in list(value.items())[:max_dict_items]
        }
    if isinstance(value, (list, tuple)):
        return [
            _bounded_prompt_value(
                item,
                max_dict_items=max_dict_items,
                max_list_items=max_list_items,
                max_string_characters=max_string_characters,
                remaining_depth=remaining_depth - 1,
            )
            for item in list(value)[:max_list_items]
        ]
    return value


def _compact_parent_metrics(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    priority = (
        "mse",
        "proxy_mse",
        "selected_model_mse",
        "last_checkpoint_mse",
        "primary_metric",
        "solution_metrics",
        "pde_residual_error",
        "boundary_error",
        "initial_error",
        "constraint_violation",
    )
    compact = {
        key: _bounded_prompt_value(value[key], max_dict_items=12, max_list_items=4)
        for key in priority
        if key in value
    }
    if compact:
        return compact
    return _bounded_prompt_value(value, max_dict_items=8, max_list_items=3)


def _fit_parent_context_budget(parents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if estimate_context_tokens(parents) <= MAX_PARENT_CONTEXT_TOKENS:
        return parents

    minimal = []
    for parent in parents:
        minimal.append(
            {
                key: deepcopy(parent[key])
                for key in (
                    "candidate_id",
                    "generation",
                    "rank",
                    "algorithm_spec",
                    "fitness",
                    "metrics",
                )
                if key in parent
            }
        )
    estimated_tokens = estimate_context_tokens(minimal)
    if estimated_tokens > MAX_PARENT_CONTEXT_TOKENS:
        raise ValueError(
            "The four executable parent specs exceed the hard prompt budget after "
            f"diagnostic compaction: estimated_tokens={estimated_tokens}, "
            f"maximum={MAX_PARENT_CONTEXT_TOKENS}."
        )
    return minimal


def _compact_optimization_feedback(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return _unavailable_diagnostic("legacy_candidate_record")
    gradient = value.get("gradient_summary") or {}
    return {
        "available": bool(value.get("available")),
        "reason": value.get("reason"),
        "plateau_detected": value.get("plateau_detected"),
        "loss_trajectory_summary": deepcopy(
            value.get("loss_trajectory_summary") or {}
        ),
        "gradient_summary": {
            "available": bool(gradient.get("available")),
            "reason": gradient.get("reason"),
            "dominant_component": gradient.get("dominant_component"),
            "gradient_imbalance_detected": gradient.get(
                "gradient_imbalance_detected"
            ),
            "gradient_conflict_detected": gradient.get(
                "gradient_conflict_detected"
            ),
        },
        "lbfgs_transition_degradation": value.get(
            "lbfgs_transition_degradation"
        ),
    }


def _compact_physical_supervision_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return _unavailable_diagnostic("legacy_candidate_record")
    spatial = value.get("residual_spatial_summary") or {}
    equations = spatial.get("equations") or {}
    return {
        "available": bool(value.get("available", True)),
        "reason": value.get("reason"),
        "pde_residual_error": value.get("pde_residual_error"),
        "boundary_error": value.get("boundary_error"),
        "initial_error": value.get("initial_error"),
        "constraint_violation": value.get("constraint_violation"),
        "residual_spatial_summary": {
            "available": bool(spatial.get("available")),
            "reason": spatial.get("reason"),
            "equation_count": len(equations) if isinstance(equations, dict) else 0,
            "localized_equation_count": sum(
                1
                for item in (
                    equations.values() if isinstance(equations, dict) else []
                )
                if isinstance(item, dict) and item.get("localized_error") is True
            ),
        },
    }


def _unavailable_diagnostic(reason: str) -> dict[str, Any]:
    return {"available": False, "reason": reason}


def _resolve_design_principles(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("knowledge_base_enabled") is False:
        return {
            "macro_physics_principles": [],
            "architecture_design_principles": [],
            "retrieval_metadata": {
                "knowledge_base_enabled": False,
                "macro_candidates": 0,
                "macro_returned": 0,
                "architecture_candidates": 0,
                "architecture_returned": 0,
            },
        }
    problem_features = payload.get("problem_features") or {}
    problem_spec = dict(payload.get("problem_spec") or {})
    if payload.get("pinnacle_profile"):
        problem_spec.setdefault("pinnacle_profile", payload["pinnacle_profile"])
    generation_feedback = payload.get("generation_feedback") or {}
    retrieval_feedback = dict(generation_feedback)
    retrieval_feedback["parent_candidates"] = [
        *(payload.get("parent_specs") or []),
        *(payload.get("population_summary") or []),
    ]
    search_stage = payload.get("search_stage") or {}
    top_k_macro = _validated_top_k(
        payload.get("top_k_macro_principles", DEFAULT_TOP_K_MACRO_PRINCIPLES),
        "top_k_macro_principles",
    )
    top_k_architecture = _validated_top_k(
        payload.get("top_k_architecture_principles", DEFAULT_TOP_K_ARCHITECTURE_PRINCIPLES),
        "top_k_architecture_principles",
    )
    manager = KnowledgeBaseManager()
    posterior_evidence = _raw_posterior_principle_evidence(payload)
    retrieved = manager.retrieve_design_principles(
        problem_features=problem_features,
        problem_spec=problem_spec,
        generation_feedback=retrieval_feedback,
        search_stage=search_stage,
        top_k_macro=top_k_macro,
        top_k_architecture=top_k_architecture,
        posterior_evidence=posterior_evidence,
    )
    metadata = dict(retrieved.get("retrieval_metadata") or {})

    macro = list(retrieved.get("macro_physics_principles") or [])
    if "macro_physics_principles" in payload:
        macro, count = rank_design_principles(
            payload.get("macro_physics_principles") or [],
            category="macro_physics",
            problem_features=problem_features,
            problem_spec=problem_spec,
            generation_feedback=retrieval_feedback,
            search_stage=search_stage,
            top_k=top_k_macro,
            posterior_evidence=posterior_evidence,
        )
        metadata.update({"macro_candidates": count, "macro_returned": len(macro)})

    architecture = list(retrieved.get("architecture_design_principles") or [])
    if "architecture_design_principles" in payload:
        architecture, count = rank_design_principles(
            payload.get("architecture_design_principles") or [],
            category="architecture_design",
            problem_features=problem_features,
            problem_spec=problem_spec,
            generation_feedback=retrieval_feedback,
            search_stage=search_stage,
            top_k=top_k_architecture,
            posterior_evidence=posterior_evidence,
        )
        metadata.update({"architecture_candidates": count, "architecture_returned": len(architecture)})
    return {
        "macro_physics_principles": macro,
        "architecture_design_principles": architecture,
        "retrieval_metadata": metadata,
    }


def resolve_posterior_principle_evidence(
    payload: dict[str, Any], principles: dict[str, Any]
) -> list[dict[str, Any]]:
    """Rank local positive and negative evidence without duplicating prompt injection."""
    if not bool(payload.get("execution_feedback", True)):
        return []
    top_k = _validated_top_k(
        payload.get("top_k_posterior_evidence", DEFAULT_TOP_K_POSTERIOR_EVIDENCE),
        "top_k_posterior_evidence",
    )
    candidates = _deduplicate_records(_raw_posterior_principle_evidence(payload))
    supplied_ids = {
        str(item.get("principle_id"))
        for group in ("macro_physics_principles", "architecture_design_principles")
        for item in principles.get(group) or []
        if item.get("principle_id")
    }
    problem_features = payload.get("problem_features") or {}
    benchmark = (
        problem_features.get("benchmark_id")
        or problem_features.get("problem_id")
        or (payload.get("problem_spec") or {}).get("problem_id")
    )
    ranked: list[dict[str, Any]] = []
    for item in candidates:
        scored = dict(item)
        score = 0.0
        if str(item.get("principle_id")) in supplied_ids:
            score += 3.0
        scope = item.get("problem_scope") or {}
        if not isinstance(scope, dict):
            scope = {}
        evidence_benchmark = item.get("benchmark_id") or scope.get("benchmark_id")
        if benchmark is not None and _normalized_equal(evidence_benchmark, benchmark):
            score += 3.0
        feature_scope = {
            key: value for key, value in scope.items() if key not in {"benchmark_id", "problem_id"}
        }
        if feature_scope and matches_required_conditions(feature_scope, problem_features):
            score += 2.0
        confidence = _safe_float(item.get("confidence"))
        evidence_count = _non_negative_int(item.get("evidence_count", item.get("application_count", 0)))
        seed_evidence = item.get("seed_evidence") or {}
        seed_total = _non_negative_int(
            seed_evidence.get("total", item.get("seed_total", 0))
            if isinstance(seed_evidence, dict)
            else item.get("seed_total", 0)
        )
        score += confidence * 2.0
        score += min(evidence_count, 5) * 0.2
        score += min(seed_total, 5) * 0.2
        scored["relevance_score"] = score
        ranked.append(scored)
    _mark_conflicting_evidence(ranked)
    ranked.sort(
        key=lambda item: (
            -_safe_float(item.get("relevance_score")),
            -_safe_float(item.get("confidence")),
            str(item.get("principle_id") or ""),
            _evidence_identity(item),
        )
    )
    return [_compact_posterior_evidence(item) for item in ranked[:top_k]]


def resolve_posterior_failure_patterns(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if not bool(payload.get("execution_feedback", True)):
        return []
    """Return a deterministic compact Top-K of empirical failure warnings."""
    top_k = _validated_top_k(
        payload.get("top_k_failure_patterns", DEFAULT_TOP_K_FAILURE_PATTERNS),
        "top_k_failure_patterns",
    )
    if "posterior_failure_patterns" in payload:
        raw = payload.get("posterior_failure_patterns") or []
    else:
        raw = [
            item
            for item in payload.get("posterior_knowledge") or []
            if isinstance(item, dict)
            and item.get("source") != "posterior_design_principle"
            and _looks_like_failure_pattern(item)
        ]
    candidates = _deduplicate_records(raw)
    problem_features = payload.get("problem_features") or {}
    benchmark = problem_features.get("benchmark_id") or problem_features.get("problem_id")
    ranked: list[dict[str, Any]] = []
    for item in candidates:
        score = _safe_float(item.get("confidence"))
        scope = item.get("problem_scope") or {}
        item_benchmark = item.get("benchmark_id") or (scope.get("benchmark_id") if isinstance(scope, dict) else None)
        if benchmark is not None and _normalized_equal(item_benchmark, benchmark):
            score += 3.0
        if isinstance(scope, dict):
            features = {key: value for key, value in scope.items() if key != "benchmark_id"}
            if features and matches_required_conditions(features, problem_features):
                score += 2.0
        if str(item.get("severity") or "").casefold() in {"high", "critical"}:
            score += 1.0
        scored = dict(item)
        scored["_relevance_score"] = score
        ranked.append(scored)
    ranked.sort(
        key=lambda item: (
            -_safe_float(item.get("_relevance_score")),
            -_safe_float(item.get("confidence")),
            str(item.get("spec_id") or item.get("failure_id") or ""),
            json.dumps(item, ensure_ascii=False, sort_keys=True, default=str),
        )
    )
    return [_compact_failure_pattern(item) for item in ranked[:top_k]]


def _raw_posterior_principle_evidence(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if "posterior_principle_evidence" in payload:
        raw = payload.get("posterior_principle_evidence") or []
    else:
        raw = [
            item
            for item in payload.get("posterior_knowledge") or []
            if isinstance(item, dict) and item.get("source") == "posterior_design_principle"
        ]
    return [dict(item) for item in raw if isinstance(item, dict)]


def _deduplicate_records(records: Any) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for item in records or []:
        if not isinstance(item, dict):
            continue
        identity = _evidence_identity(item)
        current = selected.get(identity)
        if current is None or (
            _safe_float(item.get("confidence")),
            len(json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)),
        ) > (
            _safe_float(current.get("confidence")),
            len(json.dumps(current, ensure_ascii=False, sort_keys=True, default=str)),
        ):
            selected[identity] = dict(item)
    return list(selected.values())


def _evidence_identity(item: dict[str, Any]) -> str:
    explicit = item.get("evidence_id") or item.get("failure_id")
    if explicit:
        return str(explicit)
    canonical = {
        key: value
        for key, value in item.items()
        if key not in {"relevance_score", "_relevance_score"}
    }
    return json.dumps(canonical, ensure_ascii=False, sort_keys=True, default=str)


def _mark_conflicting_evidence(items: list[dict[str, Any]]) -> None:
    by_principle: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        if item.get("principle_id"):
            by_principle.setdefault(str(item["principle_id"]), []).append(item)
    positive = {"supported", "partially_supported", "positive", "success"}
    negative = {"not_supported", "contradicted", "negative", "failed"}
    for evidence in by_principle.values():
        statuses = {str(item.get("evidence_status") or "").casefold() for item in evidence}
        if statuses.intersection(positive) and statuses.intersection(negative):
            for item in evidence:
                item["original_evidence_status"] = item.get("evidence_status")
                item["evidence_status"] = "conflicting"


def _compact_posterior_evidence(item: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "evidence_id",
        "source",
        "principle_id",
        "benchmark_id",
        "problem_scope",
        "tested_action",
        "observed_effect",
        "seed_evidence",
        "evidence_status",
        "original_evidence_status",
        "confidence",
        "evidence_count",
        "application_count",
        "training_success_rate",
        "failure_reason",
        "warning",
        "relevance_score",
    )
    return {key: _compact_value(item[key]) for key in fields if key in item}


def _compact_failure_pattern(item: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "spec_id",
        "problem_scope",
        "failed_combination",
        "failure_reason",
        "error_type",
        "recommended_mitigation",
        "confidence",
    )
    compact = {key: _compact_value(item[key]) for key in fields if key in item}
    if "problem_scope" not in compact and item.get("benchmark_id") is not None:
        compact["problem_scope"] = {"benchmark_id": _compact_value(item["benchmark_id"])}
    return compact


def _looks_like_failure_pattern(item: dict[str, Any]) -> bool:
    source = str(item.get("source") or "").casefold()
    return (
        "failure" in source
        or any(key in item for key in ("failed_combination", "failure_reason", "error_type"))
    )


def _compact_prior_knowledge(items: Any) -> list[dict[str, Any]]:
    fields = (
        "source",
        "rule_id",
        "stable_id",
        "reason",
        "then",
        "implementation_status",
        "evidence_sources",
        "binding",
    )
    compact: list[dict[str, Any]] = []
    for item in items or []:
        if isinstance(item, dict):
            compact.append({key: _compact_value(item[key]) for key in fields if key in item})
        if len(compact) >= 12:
            break
    return compact


def _compact_generation_feedback(
    feedback: dict[str, Any], *, include_candidate_ids: bool = True
) -> dict[str, Any]:
    if not feedback:
        return {}

    def _successful_view(item: dict[str, Any]) -> dict[str, Any]:
        metadata = ((item.get("algorithm_spec") or {}).get("generation_metadata") or {})
        view = {
            "strengths": _compact_value(list(item.get("strengths") or metadata.get("expected_advantages") or [])[:3]),
            "weaknesses": _compact_value(list(item.get("weaknesses") or metadata.get("possible_risks") or [])[:3]),
            "key_metrics": _compact_parent_metrics(item.get("key_metrics") or item.get("metrics") or {}),
            "adaptive_signals": _compact_adaptive_signals(item),
        }
        if include_candidate_ids:
            view["spec_id"] = item.get("spec_id") or item.get("candidate_id")
        return view

    def _failed_view(item: dict[str, Any]) -> dict[str, Any]:
        view = {
            "failure_reason": _compact_value(item.get("failure_reason")),
            "key_metrics": _compact_parent_metrics(item.get("key_metrics") or item.get("metrics") or {}),
            "adaptive_signals": _compact_adaptive_signals(item),
        }
        if include_candidate_ids:
            view["spec_id"] = item.get("spec_id") or item.get("candidate_id")
        return view

    compact = {
        "source_generation": feedback.get("source_generation"),
        "stagnated": bool(feedback.get("stagnated")),
        "stagnation": _compact_value(feedback.get("stagnation") or {}),
        "next_generation_mode": feedback.get("next_generation_mode"),
        "reflection": _compact_value(_drop_heavy_feedback_fields(feedback.get("reflection") or {})),
        "directives": _compact_value(list(feedback.get("directives") or [])[:MAX_COMPACT_LIST_ITEMS]),
        "successful_candidates": [
            _successful_view(item)
            for item in (feedback.get("successful_candidates") or [])[:MAX_COMPACT_LIST_ITEMS]
            if isinstance(item, dict)
        ],
        "failed_candidates": [
            _failed_view(item)
            for item in (feedback.get("failed_candidates") or [])[:MAX_COMPACT_LIST_ITEMS]
            if isinstance(item, dict)
        ],
    }
    return (
        compact
        if include_candidate_ids
        else _strip_independent_lineage_fields(compact)
    )


def _compact_adaptive_signals(item: dict[str, Any]) -> dict[str, Any]:
    control = item.get("adaptive_control_summary") or {}
    compute = item.get("adaptive_compute_budget_summary") or {}
    scaling = item.get("runtime_scaled_policy_summary") or {}
    sampling = control.get("sampling_control") or {}
    weighting = control.get("loss_weighting") or {}
    lbfgs = control.get("lbfgs_recovery") or {}
    signals = {
        "sampling_updates": sampling.get("updates_executed"),
        "sampling_points_scored": sampling.get("points_scored"),
        "gradient_conflict_count": weighting.get("gradient_conflict_count"),
        "weight_updates": weighting.get("weight_updates_executed"),
        "lbfgs_stall_detected": lbfgs.get("stall_detected"),
        "optimizer_steps": compute.get("optimizer_steps"),
        "optimizer_budget_ratio": scaling.get("optimizer_budget_ratio"),
        "warnings": [
            {
                key: warning.get(key)
                for key in ("path", "code", "message")
                if warning.get(key) is not None
            }
            for warning in list(item.get("adaptive_policy_warnings") or [])[:3]
            if isinstance(warning, dict)
        ],
    }
    return {key: value for key, value in signals.items() if value not in (None, [], {})}


def _drop_heavy_feedback_fields(value: Any) -> Any:
    forbidden = {
        "algorithm_spec",
        "normalized_algorithm_spec",
        "population",
        "population_summary",
        "parent_specs",
    }
    if isinstance(value, dict):
        return {
            str(key): _drop_heavy_feedback_fields(item)
            for key, item in value.items()
            if str(key) not in forbidden
        }
    if isinstance(value, list):
        return [_drop_heavy_feedback_fields(item) for item in value[:MAX_COMPACT_LIST_ITEMS]]
    return value


def _principle_retrieval_metadata(
    payload: dict[str, Any],
    principles: dict[str, Any],
    posterior_evidence: list[dict[str, Any]],
    failure_patterns: list[dict[str, Any]],
) -> dict[str, Any]:
    metadata = principles.get("retrieval_metadata") or {}
    return {
        "macro_candidate_count": int(metadata.get("macro_candidates") or 0),
        "macro_returned_count": len(principles.get("macro_physics_principles") or []),
        "architecture_candidate_count": int(metadata.get("architecture_candidates") or 0),
        "architecture_returned_count": len(principles.get("architecture_design_principles") or []),
        "posterior_candidate_count": len(_deduplicate_records(_raw_posterior_principle_evidence(payload))),
        "posterior_returned_count": len(posterior_evidence),
        "failure_candidate_count": len(_raw_failure_patterns(payload)),
        "failure_returned_count": len(failure_patterns),
        "top_k_macro": _validated_top_k(
            payload.get("top_k_macro_principles", DEFAULT_TOP_K_MACRO_PRINCIPLES), "top_k_macro_principles"
        ),
        "top_k_architecture": _validated_top_k(
            payload.get("top_k_architecture_principles", DEFAULT_TOP_K_ARCHITECTURE_PRINCIPLES),
            "top_k_architecture_principles",
        ),
        "top_k_posterior": _validated_top_k(
            payload.get("top_k_posterior_evidence", DEFAULT_TOP_K_POSTERIOR_EVIDENCE),
            "top_k_posterior_evidence",
        ),
        "top_k_failure_patterns": _validated_top_k(
            payload.get("top_k_failure_patterns", DEFAULT_TOP_K_FAILURE_PATTERNS),
            "top_k_failure_patterns",
        ),
    }


def _raw_failure_patterns(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if "posterior_failure_patterns" in payload:
        raw = payload.get("posterior_failure_patterns") or []
    else:
        raw = [
            item
            for item in payload.get("posterior_knowledge") or []
            if isinstance(item, dict)
            and item.get("source") != "posterior_design_principle"
            and _looks_like_failure_pattern(item)
        ]
    return _deduplicate_records(raw)


def _prompt_statistics(
    messages: list[dict[str, str]],
    payload: dict[str, Any],
    principles: dict[str, Any],
    posterior_evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    system_characters = len(messages[0]["content"]) if messages else 0
    user_characters = len(messages[1]["content"]) if len(messages) > 1 else 0
    candidate_task_characters = len(messages[2]["content"]) if len(messages) > 2 else 0
    serialized_characters = sum(len(str(message.get("content") or "")) for message in messages)
    context_block_characters: dict[str, int] = {}
    if len(messages) > 1:
        try:
            serialized_context = json.loads(messages[1]["content"])
        except (TypeError, ValueError, json.JSONDecodeError):
            serialized_context = None
        if isinstance(serialized_context, dict):
            context_block_characters = {
                str(key): len(
                    json.dumps(
                        value,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        default=str,
                    )
                )
                for key, value in serialized_context.items()
            }
    raw_duplicate_context = {
        "posterior_knowledge": payload.get("posterior_knowledge") or [],
        "posterior_principle_evidence": payload.get("posterior_principle_evidence") or [],
        "population_summary": payload.get("population_summary") or [],
        "generation_feedback": payload.get("generation_feedback") or {},
    }
    compact_replacement = {
        "posterior_principle_evidence": posterior_evidence,
        "generation_feedback": _compact_generation_feedback(
            payload.get("generation_feedback") or {},
            include_candidate_ids=bool(
                payload.get("evolutionary_inheritance_enabled", True)
            ),
        ),
    }
    removed_characters = max(
        0,
        len(json.dumps(raw_duplicate_context, ensure_ascii=False, default=str))
        - len(json.dumps(compact_replacement, ensure_ascii=False, default=str)),
    )
    return {
        "system_prompt_characters": system_characters,
        "user_context_characters": user_characters,
        "candidate_task_characters": candidate_task_characters,
        "estimated_uncompressed_user_context_characters": user_characters + removed_characters,
        "estimated_context_character_reduction": removed_characters,
        "estimated_input_tokens": max(1, serialized_characters // 4),
        "context_block_characters": context_block_characters,
        "macro_principle_count": len(principles.get("macro_physics_principles") or []),
        "architecture_principle_count": len(principles.get("architecture_design_principles") or []),
        "posterior_evidence_count": len(posterior_evidence),
    }


def _assert_execution_feedback_boundary(
    messages: list[dict[str, str]], payload: dict[str, Any]
) -> dict[str, Any]:
    """Fail closed if a feedback-free request still carries execution state.

    This is a final assertion at the existing send point, not a sanitizer: the
    controller and prompt assembly must already have constructed the permitted
    context.
    """

    if bool(payload.get("execution_feedback", True)):
        return {"enabled": True, "passed": True, "mode": "full_feedback"}
    if len(messages) < 3:
        raise AssertionError("feedback-free DesignAgent request is incomplete")
    try:
        context = json.loads(messages[1]["content"])
        task = json.loads(messages[2]["content"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise AssertionError(
            "feedback-free DesignAgent request must contain auditable JSON context"
        ) from exc
    forbidden_context_keys = {
        "posterior_principle_evidence",
        "posterior_failure_patterns",
        "latest_generation_posterior",
        "previous_generation_candidates",
        "previous_generation_failures",
        "active_run_posterior_rules",
        "posterior_recommended_next_actions",
        "posterior_unresolved_hypotheses",
        "posterior_context_metadata",
        "generation_feedback",
        "population_summary",
        "optimization_role_evidence",
        "constraint_sampling_role_evidence",
    }
    present = sorted(forbidden_context_keys.intersection(context))
    if present:
        raise AssertionError(
            "execution-derived context reached DesignAgent: " + ", ".join(present)
        )
    allowed_parent_keys = {"candidate_id", "generation", "algorithm_spec"}
    for parent in context.get("parent_specs") or []:
        extra = set(parent).difference(allowed_parent_keys)
        if extra:
            raise AssertionError(
                "feedback-free parent contains forbidden fields: "
                + ", ".join(sorted(extra))
            )
        metadata = dict((parent.get("algorithm_spec") or {}).get("generation_metadata") or {})
        forbidden_metadata = {
            "feedback_source_generation",
            "reflection_directives_applied",
            "posterior_evidence_applied",
            "generation_policy_audit",
        }
        leaked_metadata = sorted(forbidden_metadata.intersection(metadata))
        if leaked_metadata:
            raise AssertionError(
                "feedback-derived AlgorithmSpec metadata reached DesignAgent: "
                + ", ".join(leaked_metadata)
            )
    if "generation_policy_audit" in (task.get("candidate_task") or {}):
        raise AssertionError("controller policy audit reached feedback-free DesignAgent")

    # Unique textual canaries make regression tests and real failure messages
    # auditable without rejecting legitimate static words such as `mse` that
    # occur in the PDE loss/option registry itself.
    forbidden_sources = [
        payload.get("generation_feedback"),
        payload.get("run_posterior_context"),
        payload.get("posterior_principle_evidence"),
        payload.get("posterior_failure_patterns"),
        payload.get("population_summary"),
    ]
    for parent in payload.get("parent_specs") or []:
        if isinstance(parent, dict):
            forbidden_sources.extend(
                parent.get(key)
                for key in (
                    "rank",
                    "fitness",
                    "metrics",
                    "formal_metrics",
                    "training_summary",
                    "physical_supervision_summary",
                    "optimization_feedback",
                    "success_factors",
                    "failure_factors",
                )
            )

    def unique_strings(value: Any) -> list[str]:
        if isinstance(value, str):
            return [value] if len(value) >= 8 else []
        if isinstance(value, dict):
            return [item for child in value.values() for item in unique_strings(child)]
        if isinstance(value, (list, tuple)):
            return [item for child in value for item in unique_strings(child)]
        return []

    serialized = "\n".join(str(item.get("content") or "") for item in messages)
    leaked_canaries = sorted(
        {
            value
            for source in forbidden_sources
            for value in unique_strings(source)
            if value in serialized
        }
    )
    if leaked_canaries:
        raise AssertionError(
            "execution-feedback canary reached DesignAgent prompt: "
            + repr(leaked_canaries[:3])
        )
    return {
        "enabled": False,
        "passed": True,
        "mode": "algorithm_specs_only",
        "parent_count": len(context.get("parent_specs") or []),
        "parent_allowed_fields": sorted(allowed_parent_keys),
    }


def _assert_evolutionary_inheritance_boundary(
    messages: list[dict[str, str]], payload: dict[str, Any]
) -> dict[str, Any]:
    """Fail closed when independent search receives hereditary information."""

    if bool(payload.get("initialization_mode")):
        return {"enabled": False, "passed": True, "mode": "initialization"}
    if bool(payload.get("evolutionary_inheritance_enabled", True)):
        return {"enabled": True, "passed": True, "mode": "evolutionary_inheritance"}
    if payload.get("parent_specs"):
        raise AssertionError(
            "parent AlgorithmSpecs reached a budget-matched independent request"
        )
    if len(messages) < 3:
        raise AssertionError("independent DesignAgent request is incomplete")
    try:
        context = json.loads(messages[1]["content"])
        task = json.loads(messages[2]["content"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise AssertionError(
            "independent DesignAgent request must contain auditable JSON context"
        ) from exc

    if context.get("parent_specs"):
        raise AssertionError("parent_specs is non-empty in independent search context")
    if "top3_evolution_contract" in context:
        raise AssertionError("Top-3 evolution contract reached independent search")
    independent_contract = context.get("independent_search_contract") or {}
    if independent_contract.get("parent_algorithm_specs_available") is not False:
        raise AssertionError("independent search contract does not forbid parent specs")

    forbidden_keys = INDEPENDENT_PARENT_REFERENCE_FIELDS

    def find_forbidden(value: Any, path: str) -> list[str]:
        found: list[str] = []
        if isinstance(value, dict):
            for key, item in value.items():
                child_path = f"{path}.{key}" if path else str(key)
                if str(key) in forbidden_keys:
                    found.append(child_path)
                found.extend(find_forbidden(item, child_path))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                found.extend(find_forbidden(item, f"{path}[{index}]"))
        return found

    summary_blocks = {
        key: context.get(key)
        for key in (
            "generation_feedback",
            "posterior_principle_evidence",
            "posterior_failure_patterns",
            "latest_generation_posterior",
            "previous_top3",
            "previous_generation_candidates",
            "previous_generation_failures",
            "active_run_posterior_rules",
            "posterior_recommended_next_actions",
            "posterior_unresolved_hypotheses",
        )
        if key in context
    }
    leaked_paths = find_forbidden(summary_blocks, "context")
    if leaked_paths:
        raise AssertionError(
            "hereditary AlgorithmSpec information reached independent search: "
            + ", ".join(leaked_paths[:8])
        )

    metadata_contract = (
        (task.get("candidate_task") or {}).get("proposal_metadata_contract") or {}
    )
    required = set(metadata_contract.get("required_fields") or [])
    if required.intersection({"parent_ids", "variation_strategy", "inherited_strengths"}):
        raise AssertionError(
            "independent proposal metadata still requires evolutionary lineage"
        )
    return {
        "enabled": False,
        "passed": True,
        "mode": "budget_matched_independent_search",
        "parent_count": 0,
        "parent_algorithm_specs_exposed": False,
    }


def _principle_reference_warnings(
    specs: list[dict[str, Any]],
    principles: dict[str, Any],
    posterior_evidence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    allowed_principles = {
        str(item.get("principle_id"))
        for group in ("macro_physics_principles", "architecture_design_principles")
        for item in principles.get(group) or []
        if item.get("principle_id")
    }
    allowed_evidence = {
        str(value)
        for item in posterior_evidence
        for value in (item.get("evidence_id"), item.get("principle_id"))
        if value
    }
    warnings: list[dict[str, Any]] = []
    for index, spec in enumerate(specs):
        metadata = spec.get("generation_metadata") or {}
        for field in ("macro_physics_principles_applied", "architecture_design_principles_applied"):
            unknown = [str(item) for item in metadata.get(field) or [] if str(item) not in allowed_principles]
            if unknown:
                warnings.append({"proposal_index": index, "field": field, "unknown_ids": unknown})
        unknown_evidence = [
            str(item)
            for item in metadata.get("posterior_evidence_applied") or []
            if str(item) not in allowed_evidence
        ]
        if unknown_evidence:
            warnings.append(
                {"proposal_index": index, "field": "posterior_evidence_applied", "unknown_ids": unknown_evidence}
            )
    return warnings


def _normalized_equal(left: Any, right: Any) -> bool:
    return left is not None and right is not None and _normalize_tag(left) == _normalize_tag(right)


def _non_negative_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0
