"""Executable-field similarity checks used by evolutionary generations."""

from __future__ import annotations

from copy import deepcopy
import json
import math
from statistics import mean
from typing import Any

from forge.pipeline.e_credibility_assessment.validation.algorithm_spec_validator import (
    normalize_algorithm_spec,
)
from forge.pipeline.d_algorithm_generation.specs.option_registry import (
    algorithm_spec_option_registry,
)
from forge.pipeline.d_algorithm_generation.experiment_contract import (
    DIVERSITY_RETRY_DISTANCE,
    DIVERSITY_WARNING_DISTANCE,
    EXACT_DUPLICATE_DISTANCE,
    ROLE_DIVERSITY_RETRY_THRESHOLDS,
)


MODULE_DISTANCE_WEIGHTS = {
    "network": 0.25,
    "sampling": 0.20,
    "loss": 0.15,
    "constraints": 0.15,
    "optimization": 0.15,
    "training": 0.10,
}
NEGLIGIBLE_EXECUTABLE_CHANGE_DISTANCE = 1.0e-6

_AUDIT_FIELDS = {
    "generation_metadata",
    "candidate_id",
    "spec_id",
    "algorithm_id",
    "validation_status",
    "metrics",
    "audit",
    "timestamp",
}


def executable_algorithm_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Normalize aliases and remove identifiers and audit-only fields."""

    normalized = normalize_algorithm_spec(deepcopy(spec))
    _normalize_registry_aliases(normalized)
    return _strip_audit_fields(normalized)


def compute_algorithm_spec_distance(spec_a: dict, spec_b: dict) -> dict[str, Any]:
    """Return weighted module distances in [0, 1] for executable AlgorithmSpec fields."""

    left = executable_algorithm_spec(spec_a)
    right = executable_algorithm_spec(spec_b)
    left_modules = _distance_modules(left)
    right_modules = _distance_modules(right)
    module_distances = {
        module: round(_value_distance(left_modules[module], right_modules[module], module), 6)
        for module in MODULE_DISTANCE_WEIGHTS
    }
    total = sum(module_distances[module] * weight for module, weight in MODULE_DISTANCE_WEIGHTS.items())
    return {
        "total_distance": round(_clip(total), 6),
        "module_distances": module_distances,
    }


def role_diversity_retry_threshold(role_name: str | None) -> float:
    return float(
        ROLE_DIVERSITY_RETRY_THRESHOLDS.get(
            str(role_name or ""), DIVERSITY_RETRY_DISTANCE
        )
    )


def audit_candidate_similarity(
    candidate: dict[str, Any],
    references: list[dict[str, Any]],
    *,
    role_name: str | None = None,
) -> dict[str, Any]:
    """Audit the nearest executable design without using similarity as a rank signal."""

    if not references:
        return {
            "nearest_candidate_id": None,
            "nearest_candidate_role": None,
            "nearest_candidate_distance": None,
            "normalized_executable_distance": None,
            "overlapping_modules": [],
            "different_modules": [],
            "exact_duplicate": False,
            "exact_current_generation_duplicate": False,
            "exact_historical_duplicate": False,
            "negligible_executable_change": False,
            "similarity_status": "no_reference",
            "requires_diversity_retry": False,
            "accepted_despite_similarity": False,
            "local_design_variant": False,
        }
    comparisons: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for reference in references:
        spec = reference.get("algorithm_spec")
        if isinstance(spec, dict):
            comparisons.append(
                (reference, compute_algorithm_spec_distance(candidate["algorithm_spec"], spec))
            )
    if not comparisons:
        return audit_candidate_similarity(candidate, [], role_name=role_name)
    reference, distance = min(
        comparisons,
        key=lambda item: (
            float(item[1]["total_distance"]),
            0 if item[0].get("current_generation") else 1,
            str(item[0].get("candidate_id") or ""),
        ),
    )
    value = float(distance["total_distance"])
    module_distances = dict(distance.get("module_distances") or {})
    exact = value == EXACT_DUPLICATE_DISTANCE
    exact_current = exact and bool(reference.get("current_generation"))
    exact_historical = exact and not bool(reference.get("current_generation"))
    negligible_change = not exact and value <= NEGLIGIBLE_EXECUTABLE_CHANGE_DISTANCE
    retry_threshold = role_diversity_retry_threshold(role_name)
    requires_retry = exact or negligible_change or (not exact and value < retry_threshold)
    if exact_current:
        status = "exact_duplicate"
    elif exact_historical:
        status = "historical_exact_match_forbidden"
    elif negligible_change:
        status = "negligible_executable_change_forbidden"
    elif value < retry_threshold:
        status = "high_similarity_retry_recommended"
    elif value < DIVERSITY_WARNING_DISTANCE:
        status = "similarity_warning"
    else:
        status = "sufficient_difference"
    different_modules = [module for module, item in module_distances.items() if float(item) > 0.0]
    return {
        "nearest_candidate_id": reference.get("candidate_id"),
        "nearest_candidate_role": reference.get("candidate_role"),
        "nearest_candidate_distance": value,
        "normalized_executable_distance": value,
        "overlapping_modules": [
            module for module, item in module_distances.items() if float(item) == 0.0
        ],
        "different_modules": different_modules,
        "exact_duplicate": exact,
        "exact_current_generation_duplicate": exact_current,
        "exact_historical_duplicate": exact_historical,
        "negligible_executable_change": negligible_change,
        "similarity_status": status,
        "requires_diversity_retry": requires_retry,
        "accepted_despite_similarity": not exact and not negligible_change and value < DIVERSITY_WARNING_DISTANCE,
        "local_design_variant": not exact and 0 < len(different_modules) <= 2,
        "diversity_retry_threshold": retry_threshold,
    }




def _distance_modules(spec: dict[str, Any]) -> dict[str, Any]:
    network = deepcopy(spec.get("network") or {})
    sampling = deepcopy(spec.get("sampling") or {})
    loss = deepcopy(spec.get("loss") or {})
    constraint_terms = [
        item for item in loss.get("terms") or [] if item.get("name") != "pde_residual"
    ]
    constraints = {
        "enforcement": deepcopy(spec.get("constraint_enforcement") or {}),
        "input_transform": network.get("input_transform"),
        "boundary_sampling": sampling.get("boundary"),
        "initial_sampling": sampling.get("initial"),
        "constraint_terms": constraint_terms,
    }
    network.pop("input_transform", None)
    return {
        "network": network,
        "sampling": {
            "interior": sampling.get("interior"),
            "adaptive_refinement": sampling.get("adaptive_refinement"),
        },
        "loss": loss,
        "constraints": constraints,
        "optimization": spec.get("optimization") or {},
        "training": spec.get("training") or {},
    }


def _value_distance(left: Any, right: Any, path: str) -> float:
    if left == right:
        return 0.0
    if left is None or right is None:
        return 1.0
    if isinstance(left, bool) or isinstance(right, bool):
        return 0.0 if left == right else 1.0
    if _is_number(left) and _is_number(right):
        return _numeric_distance(float(left), float(right), path)
    if isinstance(left, str) and isinstance(right, str):
        return 0.0 if left == right else 1.0
    if isinstance(left, dict) and isinstance(right, dict):
        keys = sorted(set(left) | set(right))
        if not keys:
            return 0.0
        values = [
            _value_distance(left.get(key), right.get(key), f"{path}.{key}") for key in keys
        ]
        return _aggregate(values)
    if isinstance(left, list) and isinstance(right, list):
        if path.endswith("hidden_layers"):
            return _hidden_layer_distance(left, right)
        if "loss.terms" in path or path.endswith("constraint_terms"):
            return _named_item_list_distance(left, right, path)
        length_distance = abs(len(left) - len(right)) / max(1, len(left), len(right))
        paired = [
            _value_distance(left[index], right[index], f"{path}[{index}]")
            for index in range(min(len(left), len(right)))
        ]
        content_distance = _aggregate(paired) if paired else (1.0 if left or right else 0.0)
        return _clip(0.4 * length_distance + 0.6 * content_distance)
    return 1.0


def _numeric_distance(left: float, right: float, path: str) -> float:
    if left == right:
        return 0.0
    lower_path = path.casefold()
    if "learning_rate" in lower_path or lower_path.endswith(".weight"):
        if left > 0 and right > 0:
            return _clip(abs(math.log10(left) - math.log10(right)) / 4.0)
    scale = max(abs(left), abs(right), 1.0e-12)
    relative = abs(left - right) / scale
    if any(token in lower_path for token in ("iterations", "n_points", "batch_size")):
        return _clip(relative)
    return _clip(relative)


def _hidden_layer_distance(left: list[Any], right: list[Any]) -> float:
    left_widths = [float(item) for item in left if _is_number(item)]
    right_widths = [float(item) for item in right if _is_number(item)]
    depth = abs(len(left_widths) - len(right_widths)) / max(1, len(left_widths), len(right_widths))
    paired = []
    for left_width, right_width in zip(left_widths, right_widths):
        if left_width <= 0 or right_width <= 0:
            paired.append(1.0)
        else:
            paired.append(_clip(abs(math.log2(left_width / right_width)) / 2.0))
    width = mean(paired) if paired else (1.0 if left_widths or right_widths else 0.0)
    return _clip(0.45 * depth + 0.55 * width)


def _named_item_list_distance(left: list[Any], right: list[Any], path: str) -> float:
    if not all(isinstance(item, dict) and item.get("name") for item in [*left, *right]):
        length_distance = abs(len(left) - len(right)) / max(1, len(left), len(right))
        paired = [
            _value_distance(left[index], right[index], f"{path}[{index}]")
            for index in range(min(len(left), len(right)))
        ]
        content_distance = _aggregate(paired) if paired else (1.0 if left or right else 0.0)
        return _clip(0.4 * length_distance + 0.6 * content_distance)
    left_by_name = {str(item["name"]): item for item in left}
    right_by_name = {str(item["name"]): item for item in right}
    names = sorted(set(left_by_name) | set(right_by_name))
    values = [
        _value_distance(left_by_name.get(name), right_by_name.get(name), f"{path}.{name}")
        for name in names
    ]
    return _aggregate(values)


def _aggregate(values: list[float]) -> float:
    if not values:
        return 0.0
    return _clip(0.6 * max(values) + 0.4 * mean(values))


def _strip_audit_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _strip_audit_fields(item)
            for key, item in value.items()
            if str(key) not in _AUDIT_FIELDS
        }
    if isinstance(value, list):
        return [_strip_audit_fields(item) for item in value]
    return value


def _normalize_registry_aliases(spec: dict[str, Any]) -> None:
    registry = algorithm_spec_option_registry().get("fields") or {}
    locations = (
        ("network.architecture", [spec.get("network") or {}], "architecture"),
        ("network.activation.name", [(spec.get("network") or {}).get("activation") or {}], "name"),
        ("network.output_activation.name", [(spec.get("network") or {}).get("output_activation") or {}], "name"),
        ("network.initialization.name", [(spec.get("network") or {}).get("initialization") or {}], "name"),
        ("network.input_transform.name", [(spec.get("network") or {}).get("input_transform") or {}], "name"),
        ("constraint_enforcement.method", [spec.get("constraint_enforcement") or {}], "method"),
        ("sampling.interior.strategy", [(spec.get("sampling") or {}).get("interior") or {}], "strategy"),
        ("sampling.boundary.strategy", [(spec.get("sampling") or {}).get("boundary") or {}], "strategy"),
        ("sampling.initial.strategy", [(spec.get("sampling") or {}).get("initial") or {}], "strategy"),
        ("sampling.adaptive_refinement.strategy", [(spec.get("sampling") or {}).get("adaptive_refinement") or {}], "strategy"),
        ("loss.terms[].loss_function", list((spec.get("loss") or {}).get("terms") or []), "loss_function"),
        ("loss.weighting_strategy.name", [(spec.get("loss") or {}).get("weighting_strategy") or {}], "name"),
        ("loss.aggregation.name", [(spec.get("loss") or {}).get("aggregation") or {}], "name"),
        ("optimization.phases[].optimizer", list((spec.get("optimization") or {}).get("phases") or []), "optimizer"),
        (
            "optimization.phases[].scheduler.name",
            [phase.get("scheduler") or {} for phase in (spec.get("optimization") or {}).get("phases") or []],
            "name",
        ),
    )
    for registry_path, objects, key in locations:
        options = (registry.get(registry_path) or {}).get("options") or {}
        aliases = {
            str(name).casefold(): str(config["alias_of"]).casefold()
            for name, config in options.items()
            if isinstance(config, dict) and config.get("alias_of")
        }
        for item in objects:
            if isinstance(item, dict) and item.get(key) is not None:
                value = str(item[key]).casefold()
                item[key] = aliases.get(value, value)


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _clip(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


__all__ = [
    "MODULE_DISTANCE_WEIGHTS",
    "compute_algorithm_spec_distance",
    "executable_algorithm_spec",
]
