"""Load prior and posterior knowledge files."""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
from typing import Any, Iterable

import yaml


DEFAULT_TOP_K_MACRO_PRINCIPLES = 8
DEFAULT_TOP_K_ARCHITECTURE_PRINCIPLES = 10
DEFAULT_TOP_K_POSTERIOR_EVIDENCE = 5
DEFAULT_TOP_K_FAILURE_PATTERNS = 4

GENERAL_PRINCIPLE_BASE_SCORE = 0.5
MAX_COMPACT_STRING_LENGTH = 240
MAX_COMPACT_LIST_ITEMS = 5

RELEVANCE_WEIGHTS = {
    "required_conditions": 3.0,
    "pde_family": 3.0,
    "problem_feature": 2.0,
    "challenge_tag": 2.0,
    "parent_weakness": 2.0,
    "reflection_directive": 2.0,
    "search_stage": 1.0,
    "positive_posterior_evidence": 1.5,
    "negative_posterior_warning": 1.0,
}

_MACRO_COMPACT_FIELDS = (
    "principle_id",
    "category",
    "title",
    "required_conditions",
    "relevance_tags",
    "implication",
    "design_requirements",
    "risks",
    "relevance_score",
)
_ARCHITECTURE_COMPACT_FIELDS = (
    "principle_id",
    "category",
    "title",
    "required_conditions",
    "relevance_tags",
    "target_modules",
    "recommendations",
    "risks",
    "tradeoffs",
    "relevance_score",
)


class _PrincipleRetrievalResult(dict):
    """Dict-compatible result whose values() preserves the legacy list-only view."""

    def values(self):  # pragma: no cover - retained for callers of the old API
        return (
            self.get("macro_physics_principles", []),
            self.get("architecture_design_principles", []),
        )


class _CompactPrinciple(dict):
    """Keep the compact wire format while supporting the legacy binding lookup."""

    def __getitem__(self, key):
        if key == "binding" and key not in self:
            return False
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key == "binding" and key not in self:
            return False
        return super().get(key, default)


class KnowledgeBaseManager:
    def __init__(self, root_dir: str | Path | None = None) -> None:
        self.root_dir = Path(root_dir) if root_dir else Path(__file__).resolve().parents[1] / "knowledge_base"
        self.prior_dir = self.root_dir / "prior"
        self.posterior_dir = self.root_dir / "posterior"

    def load_prior(self) -> dict:
        return {
            "pinn_variants": self._load_json(self.prior_dir / "pinn_variants.json"),
            "physics_rules": self._load_json(self.prior_dir / "physics_rules.json"),
            "scenario_rules": self._load_json(self.prior_dir / "scenario_rules.json"),
            "pde_pinn_priors": self._load_json(self.prior_dir / "pde_pinn_priors.json"),
            "design_principles": self._load_json(self.prior_dir / "design_principles.json"),
            "structured_pde_pinn_priors": self._load_json(self.prior_dir / "structured_pde_pinn_priors.json"),
            "literature_sources": self._load_json(self.prior_dir / "pinn_literature_sources.json"),
            "problem_families": self._load_problem_family_rules(),
        }

    def load_posterior(self) -> dict:
        empirical_failures = self._load_json(self.posterior_dir / "empirical_failure_patterns.json")
        generated_failures = self._load_json(self.posterior_dir / "failure_patterns.json")
        split_failures = self._load_jsonl(self.root_dir / "posterior_failure_patterns.jsonl")
        split_evidence = self._load_jsonl(self.root_dir / "posterior_principle_evidence.jsonl")
        return {
            "success_patterns": self._load_json(self.posterior_dir / "success_patterns.json"),
            "failure_patterns": [*empirical_failures, *generated_failures, *split_failures],
            "design_principle_patterns": [
                *self._load_json(self.posterior_dir / "design_principle_patterns.json"),
                *split_evidence,
            ],
            "experiment_records_path": str(self.posterior_dir / "experiment_records.jsonl"),
        }

    def retrieve_prior(self, problem_features: dict) -> dict:
        """Retrieve prior rules using generic PhysicsProblemFeature-style fields."""
        prior = self.load_prior()
        context = _problem_context(problem_features)
        matched_rules = []
        for group_name, rules in prior.items():
            for rule in _retrievable_rules(group_name, rules):
                if _rule_matches(rule, context):
                    matched = dict(rule)
                    matched["source"] = group_name
                    matched["rule_id"] = matched.get("rule_id", matched.get("id", _legacy_rule_id(matched, group_name)))
                    matched["stable_id"] = _stable_rule_id(matched, group_name)
                    matched.setdefault("implementation_status", "implemented")
                    matched_rules.append(matched)
        matched_rules.sort(key=lambda rule: _rule_specificity(rule), reverse=True)
        return {**prior, "matched_rules": matched_rules}

    def retrieve_design_principles(
        self,
        problem_features: dict,
        *,
        problem_spec: dict | None = None,
        generation_feedback: dict | None = None,
        search_stage: dict | None = None,
        top_k_macro: int = DEFAULT_TOP_K_MACRO_PRINCIPLES,
        top_k_architecture: int = DEFAULT_TOP_K_ARCHITECTURE_PRINCIPLES,
        posterior_evidence: list[dict[str, Any]] | None = None,
    ) -> dict:
        """Filter, score, deduplicate, rank, and compact relevant principles."""
        audit = self.audit_design_principles(
            problem_features,
            problem_spec=problem_spec,
            generation_feedback=generation_feedback,
            search_stage=search_stage,
            top_k_macro=top_k_macro,
            top_k_architecture=top_k_architecture,
            posterior_evidence=posterior_evidence,
        )
        macro_audit = audit["macro_physics_principles"]
        architecture_audit = audit["architecture_design_principles"]
        macro = [compact_design_principle(item) for item in macro_audit["top_k_principles"]]
        architecture = [compact_design_principle(item) for item in architecture_audit["top_k_principles"]]
        return _PrincipleRetrievalResult(
            {
                "macro_physics_principles": macro,
                "architecture_design_principles": architecture,
                "retrieval_metadata": {
                    "macro_candidates": len(macro_audit["matched_ranked_principles"]),
                    "architecture_candidates": len(architecture_audit["matched_ranked_principles"]),
                    "macro_returned": len(macro),
                    "architecture_returned": len(architecture),
                },
            }
        )

    def audit_design_principles(
        self,
        problem_features: dict,
        *,
        problem_spec: dict | None = None,
        generation_feedback: dict | None = None,
        search_stage: dict | None = None,
        top_k_macro: int = DEFAULT_TOP_K_MACRO_PRINCIPLES,
        top_k_architecture: int = DEFAULT_TOP_K_ARCHITECTURE_PRINCIPLES,
        posterior_evidence: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Expose complete rankings, score explanations, and typed exclusions."""
        top_k_macro = _validated_top_k(top_k_macro, "top_k_macro")
        top_k_architecture = _validated_top_k(top_k_architecture, "top_k_architecture")
        candidates = self._load_design_principle_candidates(include_invalid=True)
        posterior_evidence = (
            self.load_posterior().get("design_principle_patterns") or []
            if posterior_evidence is None
            else list(posterior_evidence)
        )
        macro = audit_design_principle_candidates(
            candidates["macro_physics_principles"],
            category="macro_physics",
            problem_features=problem_features,
            problem_spec=problem_spec or {},
            generation_feedback=generation_feedback or {},
            search_stage=search_stage or {},
            top_k=top_k_macro,
            posterior_evidence=posterior_evidence,
        )
        architecture = audit_design_principle_candidates(
            candidates["architecture_design_principles"],
            category="architecture_design",
            problem_features=problem_features,
            problem_spec=problem_spec or {},
            generation_feedback=generation_feedback or {},
            search_stage=search_stage or {},
            top_k=top_k_architecture,
            posterior_evidence=posterior_evidence,
        )
        return {
            "macro_physics_principles": macro,
            "architecture_design_principles": architecture,
            "matched_ranked_principles": [
                *macro["matched_ranked_principles"],
                *architecture["matched_ranked_principles"],
            ],
            "excluded_principles": [
                *macro["excluded_principles"],
                *architecture["excluded_principles"],
            ],
            "top_k_principles": [
                *macro["top_k_principles"],
                *architecture["top_k_principles"],
            ],
        }

    def _load_design_principle_candidates(
        self, *, include_invalid: bool = False
    ) -> dict[str, list[Any]]:
        """Read both the legacy mixed catalog and the split forward-compatible files."""
        legacy = self.load_prior().get("design_principles") or {}
        macro = list(legacy.get("macro_physics_principles") or [])
        architecture = list(legacy.get("architecture_design_principles") or [])
        macro.extend(
            _principle_file_items(
                self._load_json(self.root_dir / "macro_physics_principles.json"),
                "macro_physics_principles",
            )
        )
        architecture.extend(
            _principle_file_items(
                self._load_json(self.root_dir / "architecture_design_principles.json"),
                "architecture_design_principles",
            )
        )
        return {
            "macro_physics_principles": macro if include_invalid else [item for item in macro if isinstance(item, dict)],
            "architecture_design_principles": architecture if include_invalid else [item for item in architecture if isinstance(item, dict)],
        }

    @staticmethod
    def _load_json(path: Path):
        if not path.exists():
            return []
        text = path.read_text(encoding="utf-8-sig").strip()
        if not text:
            return []
        return json.loads(text)

    @staticmethod
    def _load_jsonl(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        items: list[dict[str, Any]] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            items.append(value)
        return items

    def _load_problem_family_rules(self) -> list[dict]:
        rules: list[dict] = []
        family_dir = self.root_dir / "problem_families"
        if not family_dir.exists():
            return rules
        for path in sorted(family_dir.glob("*.yaml")):
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if isinstance(payload, dict):
                for rule in payload.get("rules", []):
                    item = dict(rule)
                    item["family_file"] = path.name
                    rules.append(item)
        return rules


def matches_required_conditions(required_conditions: dict, problem_features: dict) -> bool:
    """Return whether every explicitly required feature is present and compatible."""
    if not required_conditions:
        return True
    if not isinstance(required_conditions, dict) or not isinstance(problem_features, dict):
        return False
    missing = object()
    for key, expected in required_conditions.items():
        actual = _condition_feature_value(problem_features, key, missing)
        if actual is missing:
            return False
        if isinstance(expected, bool):
            if not isinstance(actual, bool) or actual is not expected:
                return False
        elif isinstance(expected, str):
            if not isinstance(actual, str) or _normalize_tag(actual) != _normalize_tag(expected):
                return False
        elif isinstance(expected, list):
            if isinstance(actual, (list, tuple, set)):
                if not any(_condition_scalar_matches(allowed, value) for allowed in expected for value in actual):
                    return False
            elif not any(_condition_scalar_matches(allowed, actual) for allowed in expected):
                return False
        elif type(actual) is not type(expected) or actual != expected:
            return False
    return True


def score_design_principle(
    principle: dict,
    *,
    problem_features: dict,
    problem_spec: dict,
    generation_feedback: dict,
    search_stage: dict,
) -> float:
    """Compute a deterministic relevance score from problem and search context."""
    explanation = explain_design_principle_score(
        principle,
        problem_features=problem_features,
        problem_spec=problem_spec,
        generation_feedback=generation_feedback,
        search_stage=search_stage,
    )
    return float(explanation["relevance_score"]) if explanation["eligible"] else float("-inf")


def explain_design_principle_score(
    principle: dict,
    *,
    problem_features: dict,
    problem_spec: dict,
    generation_feedback: dict,
    search_stage: dict,
) -> dict[str, Any]:
    """Explain every additive score component and the context that matched it."""
    required = principle.get("required_conditions") or principle.get("applies_when") or {}
    condition_audit = _required_condition_audit(required, problem_features)
    empty_breakdown = {
        "general_base": 0.0,
        "required_conditions": 0.0,
        "pde_family": 0.0,
        "problem_features": 0.0,
        "challenge_tags": 0.0,
        "parent_weaknesses": 0.0,
        "reflection_directives": 0.0,
        "search_stage": 0.0,
        "positive_evidence": 0.0,
        "negative_warning": 0.0,
    }
    if condition_audit["missing_required_features"] or condition_audit["mismatched_conditions"]:
        return {
            "eligible": False,
            "relevance_score": float("-inf"),
            "score_breakdown": empty_breakdown,
            "matched_tags": [],
            "matched_features": [],
            "matched_weaknesses": [],
            **condition_audit,
        }

    breakdown = dict(empty_breakdown)
    if required:
        breakdown["required_conditions"] = RELEVANCE_WEIGHTS["required_conditions"]
    else:
        breakdown["general_base"] = GENERAL_PRINCIPLE_BASE_SCORE
    relevance_tags = {_normalize_tag(item) for item in principle.get("relevance_tags") or [] if item}
    pde_tags = _pde_family_tags(problem_features, problem_spec)
    declared_families = principle.get("pde_families") or principle.get("pde_family") or []
    if isinstance(declared_families, str):
        declared_families = [declared_families]
    family_tags = {_normalize_tag(item) for item in declared_families if item}
    matched_families = relevance_tags.intersection(pde_tags).intersection(family_tags)
    if pde_tags.intersection(family_tags):
        breakdown["pde_family"] = RELEVANCE_WEIGHTS["pde_family"]
        matched_families = pde_tags.intersection(family_tags)

    problem_tags = _problem_feature_tags(problem_features, problem_spec)
    challenge_tags = _challenge_tags(problem_features, problem_spec)
    weakness_tags = _feedback_tags(generation_feedback, weakness_only=True)
    directive_tags = _feedback_tags(generation_feedback, directives_only=True)
    stage_tags = _flatten_tags(search_stage)
    matched_features = relevance_tags.intersection(problem_tags)
    matched_challenges = relevance_tags.intersection(challenge_tags)
    matched_weaknesses = relevance_tags.intersection(weakness_tags)
    matched_directives = relevance_tags.intersection(directive_tags)
    matched_stage = relevance_tags.intersection(stage_tags)
    breakdown["problem_features"] = len(matched_features) * RELEVANCE_WEIGHTS["problem_feature"]
    breakdown["challenge_tags"] = len(matched_challenges) * RELEVANCE_WEIGHTS["challenge_tag"]
    if str(principle.get("category") or "") == "architecture_design":
        breakdown["parent_weaknesses"] = len(matched_weaknesses) * RELEVANCE_WEIGHTS["parent_weakness"]
        breakdown["reflection_directives"] = len(matched_directives) * RELEVANCE_WEIGHTS["reflection_directive"]
        if matched_stage:
            breakdown["search_stage"] = RELEVANCE_WEIGHTS["search_stage"]
    if principle.get("_positive_posterior_evidence"):
        breakdown["positive_evidence"] = RELEVANCE_WEIGHTS["positive_posterior_evidence"]
    if principle.get("_negative_posterior_warning"):
        breakdown["negative_warning"] = RELEVANCE_WEIGHTS["negative_posterior_warning"]
    matched_tags = set().union(
        matched_families,
        matched_features,
        matched_challenges,
        matched_weaknesses if str(principle.get("category") or "") == "architecture_design" else set(),
        matched_directives if str(principle.get("category") or "") == "architecture_design" else set(),
        matched_stage if str(principle.get("category") or "") == "architecture_design" else set(),
    )
    return {
        "eligible": True,
        "relevance_score": float(sum(breakdown.values())),
        "score_breakdown": breakdown,
        "matched_tags": sorted(matched_tags),
        "matched_features": sorted(matched_features),
        "matched_weaknesses": sorted(matched_weaknesses),
        "matched_directives": sorted(matched_directives),
        "matched_search_stage": sorted(matched_stage),
        **condition_audit,
    }


def compact_design_principle(principle: dict) -> dict:
    """Return the bounded prompt-facing projection for a ranked principle."""
    category = str(principle.get("category") or principle.get("principle_type") or "")
    fields = _MACRO_COMPACT_FIELDS if "macro" in category else _ARCHITECTURE_COMPACT_FIELDS
    compact = _CompactPrinciple()
    for field in fields:
        if field in principle:
            compact[field] = _compact_value(principle[field])
    if not compact.get("principle_id"):
        raise ValueError("design principle is missing principle_id")
    return compact


def audit_design_principle_candidates(
    principles: Iterable[Any],
    *,
    category: str,
    problem_features: dict,
    problem_spec: dict,
    generation_feedback: dict,
    search_stage: dict,
    top_k: int,
    posterior_evidence: Iterable[dict[str, Any]] = (),
) -> dict[str, list[dict[str, Any]]]:
    """Audit a principle group without hiding filtered or truncated candidates."""
    top_k = _validated_top_k(top_k, "top_k")
    excluded: list[dict[str, Any]] = []
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(principles):
        schema_errors = _principle_schema_errors(raw, category)
        if schema_errors:
            principle_id = raw.get("principle_id") if isinstance(raw, dict) else None
            excluded.append(
                {
                    "principle_id": str(principle_id or f"<invalid:{index}>"),
                    "category": category,
                    "excluded_reason": "invalid_principle_schema",
                    "schema_errors": schema_errors,
                    "missing_required_features": [],
                    "mismatched_conditions": {},
                }
            )
            continue
        normalized.append(_normalize_principle(raw, category))

    selected: dict[str, dict[str, Any]] = {}
    duplicate_groups: dict[str, list[dict[str, Any]]] = {}
    for principle in normalized:
        principle_id = str(principle["principle_id"])
        current = selected.get(principle_id)
        if current is None:
            selected[principle_id] = principle
            continue
        duplicate_groups.setdefault(principle_id, []).append(principle)
        if _principle_quality_key(principle) > _principle_quality_key(current):
            duplicate_groups[principle_id].append(current)
            selected[principle_id] = principle
    for principle_id, duplicates in sorted(duplicate_groups.items()):
        kept = selected[principle_id]
        for duplicate in duplicates:
            if duplicate is kept:
                continue
            excluded.append(
                {
                    "principle_id": principle_id,
                    "category": category,
                    "excluded_reason": "duplicate_principle_id",
                    "kept_confidence": _safe_float(kept.get("confidence")),
                    "discarded_confidence": _safe_float(duplicate.get("confidence")),
                    "missing_required_features": [],
                    "mismatched_conditions": {},
                }
            )

    evidence_by_id = _posterior_evidence_by_principle(posterior_evidence, problem_features)
    ranked: list[dict[str, Any]] = []
    for principle_id, principle in selected.items():
        evidence_flags = evidence_by_id.get(principle_id, set())
        principle["_positive_posterior_evidence"] = "positive" in evidence_flags
        principle["_negative_posterior_warning"] = "negative" in evidence_flags
        explanation = explain_design_principle_score(
            principle,
            problem_features=problem_features,
            problem_spec=problem_spec,
            generation_feedback=generation_feedback,
            search_stage=search_stage,
        )
        if not explanation["eligible"]:
            missing = explanation["missing_required_features"]
            excluded.append(
                {
                    "principle_id": principle_id,
                    "category": category,
                    "excluded_reason": (
                        "missing_required_feature" if missing else "required_condition_mismatch"
                    ),
                    "missing_required_features": missing,
                    "mismatched_conditions": explanation["mismatched_conditions"],
                }
            )
            continue
        principle["relevance_score"] = explanation["relevance_score"]
        ranked.append(_audit_principle_view(principle, explanation))
    ranked.sort(
        key=lambda item: (
            -float(item.get("relevance_score") or 0.0),
            -_safe_float(item.get("confidence")),
            str(item.get("principle_id") or ""),
        )
    )
    for item in ranked[top_k:]:
        excluded.append(
            {
                "principle_id": item["principle_id"],
                "category": category,
                "excluded_reason": "top_k_truncation",
                "relevance_score": item["relevance_score"],
                "missing_required_features": [],
                "mismatched_conditions": {},
            }
        )
    excluded.sort(key=lambda item: (str(item.get("principle_id") or ""), item["excluded_reason"]))
    return {
        "matched_ranked_principles": ranked,
        "excluded_principles": excluded,
        "top_k_principles": ranked[:top_k],
    }


def _audit_principle_view(principle: dict[str, Any], explanation: dict[str, Any]) -> dict[str, Any]:
    fields = set(_MACRO_COMPACT_FIELDS).union(_ARCHITECTURE_COMPACT_FIELDS).union(
        {"confidence", "pde_families", "version"}
    )
    view = {key: _compact_value(value) for key, value in principle.items() if key in fields}
    view.update(
        {
            "score_breakdown": explanation["score_breakdown"],
            "matched_tags": explanation["matched_tags"],
            "matched_features": explanation["matched_features"],
            "matched_weaknesses": explanation["matched_weaknesses"],
            "matched_directives": explanation["matched_directives"],
            "matched_search_stage": explanation["matched_search_stage"],
        }
    )
    return view


def _principle_schema_errors(principle: Any, expected_category: str) -> list[str]:
    if not isinstance(principle, dict):
        return ["principle must be an object"]
    errors: list[str] = []
    if not isinstance(principle.get("principle_id"), str) or not principle.get("principle_id", "").strip():
        errors.append("principle_id must be a non-empty string")
    conditions = principle.get("required_conditions", principle.get("applies_when", {}))
    if conditions is not None and not isinstance(conditions, dict):
        errors.append("required_conditions must be an object")
    tags = principle.get("relevance_tags")
    if tags is not None and not isinstance(tags, list):
        errors.append("relevance_tags must be a list")
    category = principle.get("category")
    if category is not None and category != expected_category:
        errors.append(f"category must be {expected_category}")
    return errors


def rank_design_principles(
    principles: Iterable[dict[str, Any]],
    *,
    category: str,
    problem_features: dict,
    problem_spec: dict,
    generation_feedback: dict,
    search_stage: dict,
    top_k: int,
    posterior_evidence: Iterable[dict[str, Any]] = (),
) -> tuple[list[dict[str, Any]], int]:
    """Apply the common retrieval pipeline to catalog or explicit payload principles."""
    audit = audit_design_principle_candidates(
        principles,
        category=category,
        problem_features=problem_features,
        problem_spec=problem_spec,
        generation_feedback=generation_feedback,
        search_stage=search_stage,
        top_k=top_k,
        posterior_evidence=posterior_evidence,
    )
    ranked = audit["matched_ranked_principles"]
    return [compact_design_principle(item) for item in audit["top_k_principles"]], len(ranked)


def _normalize_principle(principle: dict[str, Any], category: str) -> dict[str, Any]:
    item = dict(principle)
    item["category"] = category
    item["required_conditions"] = dict(
        item.get("required_conditions") or item.get("applies_when") or item.get("conditions") or {}
    )
    if not item.get("title"):
        item["title"] = str(item.get("principle") or item.get("principle_id") or "")
    if category == "macro_physics":
        item["implication"] = item.get("implication") or item.get("principle") or ""
        item["design_requirements"] = list(
            item.get("design_requirements") or item.get("design_implications") or []
        )
    else:
        item["recommendations"] = list(
            item.get("recommendations") or item.get("design_implications") or []
        )
    if not item.get("relevance_tags"):
        item["relevance_tags"] = _derived_principle_tags(item)
    else:
        item["relevance_tags"] = list(dict.fromkeys(_normalize_tag(value) for value in item["relevance_tags"] if value))
    item.setdefault("risks", [])
    item.setdefault("confidence", 0.0)
    return item


def _deduplicate_principles(principles: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prefer confidence, then version recency, then information completeness."""
    selected: dict[str, dict[str, Any]] = {}
    for principle in principles:
        principle_id = str(principle.get("principle_id") or "")
        current = selected.get(principle_id)
        if current is None or _principle_quality_key(principle) > _principle_quality_key(current):
            selected[principle_id] = principle
    return list(selected.values())


def _principle_quality_key(principle: dict[str, Any]) -> tuple[float, tuple[int, ...], int, str]:
    confidence = _safe_float(principle.get("confidence"))
    version = tuple(int(value) for value in re.findall(r"\d+", str(principle.get("version") or "0"))) or (0,)
    completeness = sum(
        1 for key, value in principle.items() if not key.startswith("_") and value not in (None, "", [], {})
    )
    canonical = json.dumps(principle, ensure_ascii=False, sort_keys=True, default=str)
    return confidence, version, completeness, canonical


def _posterior_evidence_by_principle(
    evidence: Iterable[dict[str, Any]], problem_features: dict
) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    benchmark = problem_features.get("benchmark_id") or problem_features.get("problem_id")
    for item in evidence:
        if not isinstance(item, dict) or not item.get("principle_id"):
            continue
        scope = item.get("problem_scope") or {}
        evidence_benchmark = item.get("benchmark_id") or (scope.get("benchmark_id") if isinstance(scope, dict) else None)
        if evidence_benchmark not in (None, "unknown", benchmark):
            continue
        status = str(item.get("evidence_status") or "").casefold()
        success_rate = _safe_float(item.get("training_success_rate"), default=0.5)
        flags = result.setdefault(str(item["principle_id"]), set())
        if status in {"supported", "partially_supported", "positive", "success"} or success_rate > 0.5:
            flags.add("positive")
        if status in {"not_supported", "contradicted", "negative", "failed", "conflicting"} or success_rate < 0.5:
            flags.add("negative")
    return result


def _principle_file_items(payload: Any, group_name: str) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        values = payload.get(group_name) or payload.get("principles") or []
        return values if isinstance(values, list) else []
    return []


def _validated_top_k(value: Any, name: str) -> int:
    try:
        top_k = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if top_k < 0:
        raise ValueError(f"{name} must be non-negative")
    return top_k


def _condition_scalar_matches(expected: Any, actual: Any) -> bool:
    if isinstance(expected, bool):
        return isinstance(actual, bool) and actual is expected
    if isinstance(expected, str):
        return isinstance(actual, str) and _normalize_tag(expected) == _normalize_tag(actual)
    return type(expected) is type(actual) and expected == actual


def _required_condition_audit(required_conditions: Any, problem_features: dict) -> dict[str, Any]:
    if not isinstance(required_conditions, dict):
        return {
            "missing_required_features": [],
            "mismatched_conditions": {"required_conditions": {"expected": "object", "actual": type(required_conditions).__name__}},
        }
    missing = object()
    missing_features: list[str] = []
    mismatched: dict[str, dict[str, Any]] = {}
    for key, expected in required_conditions.items():
        actual = _condition_feature_value(problem_features, key, missing)
        if actual is missing:
            missing_features.append(str(key))
            continue
        if isinstance(expected, list):
            values = actual if isinstance(actual, (list, tuple, set)) else [actual]
            matched = any(
                _condition_scalar_matches(allowed, value)
                for allowed in expected
                for value in values
            )
        else:
            matched = _condition_scalar_matches(expected, actual)
        if not matched:
            mismatched[str(key)] = {"expected": expected, "actual": actual}
    return {
        "missing_required_features": sorted(missing_features),
        "mismatched_conditions": mismatched,
    }


def _condition_feature_value(problem_features: dict, key: str, missing: Any) -> Any:
    aliases = {
        "pde_family": ("pde_type",),
        "pde_type": ("pde_family",),
        "has_multiscale": ("multiscale", "is_multiscale"),
        "is_multiscale": ("multiscale", "has_multiscale"),
        "is_stiff": ("stiff",),
        "stiff": ("is_stiff",),
        "oscillatory": ("has_oscillatory_solution",),
        "has_oscillatory_solution": ("oscillatory",),
        "elliptic": ("is_elliptic",),
        "is_elliptic": ("elliptic",),
        "periodic": ("has_periodicity",),
        "has_periodicity": ("periodic",),
        "positivity": ("has_positivity_requirement", "has_inequality_constraints"),
        "has_positivity_requirement": ("has_inequality_constraints", "positivity"),
        "has_variable_coefficients": ("has_varying_coefficient",),
        "has_varying_coefficient": ("has_variable_coefficients",),
        "incompressible": ("has_incompressibility",),
        "has_incompressibility": ("incompressible",),
        "long_time": ("has_long_time_horizon",),
        "has_long_time_horizon": ("long_time",),
        "has_localized_error": ("localized_error",),
        "has_boundary_condition": ("boundary_type_not_empty",),
    }
    if key in problem_features:
        return problem_features[key]
    for alias in aliases.get(key, ()):
        if alias in problem_features:
            return problem_features[alias]
    profile = problem_features.get("pinnacle_profile") or {}
    if not isinstance(profile, dict):
        profile = {}
    family = (
        problem_features.get("pde_family")
        or problem_features.get("pde_type")
        or profile.get("family")
    )
    if key in {"pde_family", "pde_type"} and family:
        return family
    tags = _flatten_tags(
        [problem_features.get("challenge_tags") or [], profile.get("challenge_tags") or []]
    )
    tag_conditions = {
        "elliptic": {"elliptic"},
        "is_elliptic": {"elliptic"},
        "is_stiff": {"stiff"},
        "stiff": {"stiff"},
        "has_complex_geometry": {"complex_geometry"},
        "has_interface": {"interface"},
        "has_piecewise_coefficient": {"piecewise_coefficient"},
        "has_varying_coefficient": {
            "varying_coefficient",
            "piecewise_coefficient",
            "heterogeneous",
        },
        "has_oscillatory_solution": {"oscillatory"},
        "oscillatory": {"oscillatory"},
        "has_chaotic_dynamics": {"chaotic"},
        "chaotic": {"chaotic"},
        "has_long_time_horizon": {"long_time"},
        "long_time": {"long_time"},
        "has_incompressibility": {"incompressible"},
        "incompressible": {"incompressible"},
        "has_noisy_observations": {"noisy_observations"},
        "has_periodicity": {"periodic", "periodic_boundary"},
        "periodic": {"periodic", "periodic_boundary"},
        "has_positivity_requirement": {"positivity", "positivity_requirement"},
        "positivity": {"positivity", "positivity_requirement"},
        "has_conservation_form": {"conservation", "conservation_law"},
        "has_variable_coefficients": {"variable_coefficient", "varying_coefficient"},
        "high_dimensional": {"high_dimensional"},
        "has_high_order_derivatives": {"high_order"},
        "nonlinear": {"nonlinear"},
    }
    condition_tags = tag_conditions.get(key, set())
    if condition_tags.intersection(tags):
        return True
    if key == "has_coupled_fields" and profile.get("output_dimension") is not None:
        return int(profile["output_dimension"]) > 1
    if key == "has_unknown_parameters" and profile.get("task_type"):
        return profile["task_type"] == "inverse"
    if key == "high_dimensional":
        dimension = problem_features.get("spatial_dimension")
        if dimension is not None:
            return int(dimension) >= 4
    if key == "has_high_order_derivatives":
        order = problem_features.get("highest_derivative_order")
        if order is None:
            order = problem_features.get("equation_order")
        if order is not None:
            return int(order) >= 3
    family_conditions = {
        "convection_dominated": {"burgers"},
        "diffusion_dominated": {"heat"},
        "has_wave_propagation": {"wave"},
        "has_wave_transport": {"advection"},
    }
    if key in family_conditions and family:
        return _normalize_tag(family) in family_conditions[key]
    return missing


def _normalize_tag(value: Any) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value).strip().casefold()).strip("_")
    aliases = {
        "has_shock": "shock",
        "has_steep_gradient": "steep_gradient",
        "is_multiscale": "multiscale",
        "has_multiscale": "multiscale",
        "diffusion": "diffusion_dominated",
        "ns": "navier_stokes",
        "nse": "navier_stokes",
        "navier_stokes_equations": "navier_stokes",
        "compressible_euler": "euler",
        "euler_equations": "euler",
        "electromagnetic": "maxwell",
        "maxwell_equations": "maxwell",
        "nls": "nonlinear_schrodinger",
        "nonlinear_schrodinger_equation": "nonlinear_schrodinger",
        "hj": "hamilton_jacobi",
        "hamilton_jacobi_equation": "hamilton_jacobi",
        "fractional_laplacian": "fractional_diffusion",
        "fsi": "fluid_structure",
        "magnetohydrodynamics": "mhd",
        "surface_laplacian": "laplace_beltrami",
    }
    return aliases.get(normalized, normalized)


def _derived_principle_tags(principle: dict[str, Any]) -> list[str]:
    tags: list[str] = []
    for key, value in (principle.get("required_conditions") or {}).items():
        if value is True:
            tags.append(_normalize_tag(key))
        elif isinstance(value, str):
            tags.append(_normalize_tag(value))
        elif isinstance(value, list):
            tags.extend(_normalize_tag(item) for item in value)
    text = " ".join(
        str(principle.get(key) or "") for key in ("principle_id", "title", "principle", "implication")
    )
    known_tags = (
        "shock",
        "steep_gradient",
        "localized_error",
        "multiscale",
        "periodic",
        "causality",
        "initial",
        "boundary",
        "conservation",
        "diffusion_dominated",
        "stiff",
        "inverse",
        "budget",
        "derivative",
        "sampling",
    )
    normalized_text = _normalize_tag(text)
    tags.extend(tag for tag in known_tags if tag in normalized_text)
    return list(dict.fromkeys(tag for tag in tags if tag))


def _problem_feature_tags(problem_features: dict, problem_spec: dict) -> set[str]:
    tags: set[str] = {"governing_equation", "constraint_consistency"}
    for key, value in problem_features.items():
        normalized_key = _normalize_tag(key)
        if value is True:
            tags.add(normalized_key)
        elif isinstance(value, str) and value:
            tags.update(_text_tags(value))
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                tags.update(_text_tags(item))
    tags.update(_flatten_tags(problem_spec))
    if problem_features.get("has_initial_condition") or problem_features.get("time_dependent"):
        tags.add("initial")
    if (
        problem_features.get("has_boundary_condition")
        or problem_features.get("boundary_type")
        or (problem_spec.get("constraints") if isinstance(problem_spec, dict) else None)
    ):
        tags.add("boundary")
    if problem_features.get("has_shock"):
        tags.update({"steep_gradient", "localized_error"})
    return tags


def _pde_family_tags(problem_features: dict, problem_spec: dict) -> set[str]:
    values = [
        problem_features.get("pde_family"),
        problem_features.get("pde_type"),
        problem_features.get("benchmark_id"),
        problem_features.get("problem_id"),
        problem_spec.get("pde_family"),
        problem_spec.get("problem_id"),
    ]
    tags: set[str] = set()
    for value in values:
        tags.update(_text_tags(value))
    return tags


def _challenge_tags(problem_features: dict, problem_spec: dict) -> set[str]:
    profile = (
        problem_features.get("pinnacle_profile")
        or problem_spec.get("pinnacle_profile")
        or {}
    )
    profile_tags = (profile.get("challenge_tags") or []) if isinstance(profile, dict) else []
    return _flatten_tags([problem_features.get("challenge_tags") or [], profile_tags])


def _feedback_tags(
    generation_feedback: dict, *, weakness_only: bool = False, directives_only: bool = False
) -> set[str]:
    values: list[Any] = []
    if directives_only:
        values.extend(generation_feedback.get("directives") or [])
        values.append(generation_feedback.get("reflection") or {})
    elif weakness_only:
        for group in ("successful_candidates", "failed_candidates", "parent_candidates"):
            for candidate in generation_feedback.get(group) or []:
                if isinstance(candidate, dict):
                    values.extend(candidate.get("weaknesses") or [])
                    values.append(candidate.get("main_weakness"))
                    values.append(candidate.get("failure_reason"))
                    training_summary = candidate.get("training_summary") or {}
                    if isinstance(training_summary, dict):
                        values.append(training_summary.get("main_weakness"))
                    metadata = ((candidate.get("algorithm_spec") or {}).get("generation_metadata") or {})
                    if isinstance(metadata, dict):
                        values.extend(metadata.get("possible_risks") or [])
    else:
        values.append(generation_feedback)
    return _flatten_tags(values)


def _flatten_tags(value: Any) -> set[str]:
    tags: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if item is True:
                tags.add(_normalize_tag(key))
            tags.update(_flatten_tags(item))
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            tags.update(_flatten_tags(item))
    elif value not in (None, ""):
        tags.update(_text_tags(value))
    return tags


def _text_tags(value: Any) -> set[str]:
    normalized = _normalize_tag(value)
    if not normalized:
        return set()
    parts = {part for part in normalized.split("_") if part}
    return {normalized, *parts}


def _compact_value(value: Any) -> Any:
    if isinstance(value, str):
        return value if len(value) <= MAX_COMPACT_STRING_LENGTH else value[: MAX_COMPACT_STRING_LENGTH - 1] + "…"
    if isinstance(value, list):
        return [_compact_value(item) for item in value[:MAX_COMPACT_LIST_ITEMS]]
    if isinstance(value, tuple):
        return [_compact_value(item) for item in value[:MAX_COMPACT_LIST_ITEMS]]
    if isinstance(value, dict):
        return {str(key): _compact_value(item) for key, item in value.items()}
    return value


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _rule_matches(rule: dict, context: dict) -> bool:
    applies = rule.get("applies_when") or rule.get("conditions") or rule.get("if") or {}
    if not applies and rule.get("recommended_for"):
        return _recommended_for_matches(rule.get("recommended_for", []), context)
    law_types = set(applies.get("law_types", []))
    if law_types and not law_types.intersection(context["law_types"]):
        return False
    constraint_types = set(applies.get("constraint_types", []))
    if constraint_types and not constraint_types.intersection(context["constraint_types"]):
        return False
    for key in [
        "benchmark_id",
        "problem_id",
        "equation_id",
        "pde_type",
        "spatial_dimension",
        "task_type",
        "has_shock",
        "has_multiscale",
        "has_conservation_law",
        "has_symmetry",
        "has_periodicity",
        "has_discontinuity",
        "is_stiff",
        "has_constitutive_relation",
        "has_algebraic_constraints",
        "has_integral_constraints",
        "has_inequality_constraints",
        "boundary_type_not_empty",
        "time_dependent",
        "has_unknown_parameters",
        "data_available",
        "has_manufactured_source",
        "reference_solution_type",
        "direct_literature_comparison",
    ]:
        if key in applies and not _value_matches(applies[key], context.get(key)):
            return False
    return True


def _problem_context(problem_features: dict) -> dict:
    return {
        "benchmark_id": problem_features.get("benchmark_id"),
        "problem_id": problem_features.get("problem_id"),
        "equation_id": problem_features.get("equation_id"),
        "pde_type": problem_features.get("pde_type"),
        "spatial_dimension": problem_features.get("spatial_dimension"),
        "law_types": set(problem_features.get("law_types", [])),
        "constraint_types": set(problem_features.get("constraint_types", [])),
        "task_type": problem_features.get("task_type") or problem_features.get("forward_or_inverse"),
        "has_shock": problem_features.get("has_shock"),
        "has_multiscale": problem_features.get("has_multiscale") or problem_features.get("is_multiscale"),
        "has_conservation_law": problem_features.get("has_conservation_law"),
        "has_symmetry": problem_features.get("has_symmetry"),
        "has_periodicity": problem_features.get("has_periodicity"),
        "has_discontinuity": problem_features.get("has_discontinuity"),
        "is_stiff": problem_features.get("is_stiff"),
        "has_constitutive_relation": problem_features.get("has_constitutive_relation"),
        "has_algebraic_constraints": problem_features.get("has_algebraic_constraints"),
        "has_integral_constraints": problem_features.get("has_integral_constraints"),
        "has_inequality_constraints": problem_features.get("has_inequality_constraints"),
        "boundary_type_not_empty": bool(problem_features.get("boundary_type")),
        "time_dependent": problem_features.get("time_dependent"),
        "has_unknown_parameters": problem_features.get("has_unknown_parameters"),
        "data_available": problem_features.get("data_available"),
        "has_manufactured_source": problem_features.get("has_manufactured_source"),
        "reference_solution_type": problem_features.get("reference_solution_type"),
        "direct_literature_comparison": problem_features.get("direct_literature_comparison"),
    }


def _retrievable_rules(group_name: str, payload) -> list[dict]:
    """Expose both legacy rule lists and structured benchmark families."""
    if isinstance(payload, list):
        return payload
    if group_name != "structured_pde_pinn_priors" or not isinstance(payload, dict):
        return []
    rules: list[dict] = []
    for family in payload.get("families", []):
        if not isinstance(family, dict):
            continue
        item = dict(family)
        item.setdefault("rule_id", item.get("family_id"))
        item.setdefault("stable_id", str(item.get("family_id", "structured_family")).upper())
        if "applies_when" not in item and item.get("benchmark_ids"):
            item["applies_when"] = {"benchmark_id": item["benchmark_ids"]}
        item.setdefault(
            "then",
            {
                key: item[key]
                for key in [
                    "trusted_equation",
                    "benchmark_contract",
                    "reference_metrics",
                    "recommended_evaluator_metrics",
                    "training_protocol",
                    "executable_prior_components",
                    "candidate_only_components",
                    "engineering_notes",
                ]
                if key in item
            },
        )
        rules.append(item)
    return rules


def _rule_specificity(rule: dict) -> int:
    """Rank exact benchmark contracts ahead of broad heuristic guidance."""
    applies = rule.get("applies_when") or rule.get("conditions") or rule.get("if") or {}
    weights = {
        "benchmark_id": 100,
        "equation_id": 80,
        "has_manufactured_source": 60,
        "direct_literature_comparison": 40,
        "reference_solution_type": 30,
        "pde_type": 10,
        "spatial_dimension": 5,
    }
    return sum(weight for key, weight in weights.items() if key in applies)


def _value_matches(expected, actual) -> bool:
    if isinstance(expected, list):
        return actual in expected
    return expected == actual


def _recommended_for_matches(recommended_for: list[str], context: dict) -> bool:
    tags = set(recommended_for)
    if "baseline" in tags or "smooth_pde" in tags:
        return True
    if "forward_problem" in tags and context.get("task_type") == "forward":
        return True
    if "inverse_problem" in tags and context.get("task_type") in {"inverse", "parameter_identification"}:
        return True
    if tags.intersection({"shock", "steep_gradient", "localized_error", "boundary_layer"}) and (
        context.get("has_shock") or context.get("has_discontinuity")
    ):
        return True
    if tags.intersection({"high_frequency", "multiscale"}) and context.get("has_multiscale"):
        return True
    return False


def _legacy_rule_id(rule: dict, group_name: str) -> str:
    if group_name == "pinn_variants" and rule.get("name"):
        return str(rule["name"])
    return group_name


def _stable_rule_id(rule: dict, group_name: str) -> str:
    explicit = rule.get("stable_id")
    if explicit:
        return str(explicit)
    legacy = rule.get("rule_id") or rule.get("id") or rule.get("name") or group_name
    mapping = {
        "PINN": "PINN_VARIANT_BASE_001",
        "gPINN": "PINN_VARIANT_GPINN_002",
        "RAR-PINN": "PINN_VARIANT_RAR_003",
        "Fourier_PINN": "PINN_VARIANT_FOURIER_004",
        "time_dependent_requires_initial_loss": "PHYSICS_RULE_INITIAL_001",
        "boundary_requires_boundary_loss": "PHYSICS_RULE_BOUNDARY_002",
        "conservation_prefers_constraint": "PHYSICS_PRIOR_CONSERVATION_003",
        "shock_prefers_gpinn_rar": "RULE_SHOCK_RAR_002",
        "multiscale_prefers_fourier": "RULE_MULTISCALE_FOURIER_003",
        "periodic_prefers_periodic_constraint": "RULE_PERIODIC_CONSTRAINT_004",
        "inverse_problem_parameter_identification": "RULE_INVERSE_PARAMETER_ID_001",
    }
    return mapping.get(str(legacy), str(legacy))
