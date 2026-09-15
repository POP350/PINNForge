"""PDE-aware sampling-budget planning shared by validation and runtime.

AlgorithmSpec declares three *pools* (interior, boundary, and initial).  A
problem may expose several distinct coordinate samplers inside one pool.  This
module allocates each pool once across the active samplers instead of granting
the full pool to every loss component.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping


EQUATION_ALIASES = frozenset(
    {"pde_residual", "governing_residual", "pde_component"}
)
INITIAL_ALIASES = frozenset(
    {"initial_condition", "initial_displacement", "initial_velocity"}
)
BOUNDARY_ALIASES = frozenset({"boundary_condition", "spatial_boundary"})


class SamplingContractError(ValueError):
    """Raised when a PDE sampling contract cannot satisfy an AlgorithmSpec."""


@dataclass(frozen=True)
class SamplingSourcePlan:
    """One unique coordinate source in the effective training batch."""

    sampler_id: str
    domain_role: str
    constraint_category: str | None
    budget_pool: str
    associated_loss_component_ids: tuple[str, ...]
    point_budget: int
    point_multiplier: int
    allocation_weight: float
    minimum_points: int
    fixed_points: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sampler_id": self.sampler_id,
            "domain_role": self.domain_role,
            "constraint_category": self.constraint_category,
            "budget_pool": self.budget_pool,
            "associated_loss_component_ids": list(
                self.associated_loss_component_ids
            ),
            "point_budget": self.point_budget,
            "point_multiplier": self.point_multiplier,
            "effective_point_count": self.point_budget * self.point_multiplier,
            "allocation_weight": self.allocation_weight,
            "minimum_points": self.minimum_points,
            "fixed_points": self.fixed_points,
        }


@dataclass(frozen=True)
class SamplingBudgetPlan:
    """Deterministic allocation of AlgorithmSpec pools to active samplers."""

    sources: tuple[SamplingSourcePlan, ...]
    declared_pools: Mapping[str, int]
    allocated_pools: Mapping[str, int]

    @property
    def total_training_points(self) -> int:
        return sum(
            item.point_budget * item.point_multiplier for item in self.sources
        )

    def source(self, sampler_id: str) -> SamplingSourcePlan | None:
        return next(
            (item for item in self.sources if item.sampler_id == sampler_id),
            None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "allocation_policy": "shared_role_budget",
            "declared_pools": dict(self.declared_pools),
            "allocated_pools": dict(self.allocated_pools),
            "total_training_points": self.total_training_points,
            "sources": [item.to_dict() for item in self.sources],
        }


def constraint_category(constraint_type: str) -> str:
    value = str(constraint_type)
    if value == "observation":
        return "observation"
    if value in {"boundary", "periodic"}:
        return "boundary_constraint"
    if value == "initial":
        return "initial_constraint"
    if value == "interface":
        return "interface_constraint"
    if value == "conservation":
        return "conservation_constraint"
    if value == "integral":
        return "integral_constraint"
    return "auxiliary_constraint"


def constraint_domain_role(
    constraint_type: str, metadata: Mapping[str, Any] | None = None
) -> str:
    metadata = dict(metadata or {})
    if metadata.get("pinnacle_bc_type") == "pointset":
        return "observation"
    value = str(constraint_type)
    if value == "observation":
        return "observation"
    if value in {"boundary", "periodic"}:
        return "boundary"
    if value == "initial":
        return "initial_surface"
    if value == "interface":
        return "interface"
    if value == "integral":
        return "integral_domain"
    if value == "parameter_bound":
        return "parameter_domain"
    return "custom_constraint_domain"


def default_budget_pool(domain_role: str) -> str:
    if domain_role == "interior":
        return "interior"
    if domain_role == "initial_surface":
        return "initial"
    return "boundary"


def constraint_sampling_identity(constraint: Any) -> tuple[str, str, str]:
    """Return domain role, provider category, and unique coordinate-source ID."""

    metadata = dict(getattr(constraint, "metadata", {}) or {})
    role = constraint_domain_role(str(constraint.constraint_type), metadata)
    runtime_category = str(
        metadata.get("sampling_category")
        or constraint.target
        or constraint.constraint_type
    )
    coordinate_group = str(
        metadata.get("shared_coordinate_group") or runtime_category
    )
    return role, runtime_category, f"sample:{role}:{coordinate_group}"


def _constraint_point_multiplier(constraint: Any) -> int:
    metadata = dict(getattr(constraint, "metadata", {}) or {})
    value = int(metadata.get("sampling_point_multiplier", 1))
    if value < 1:
        raise SamplingContractError(
            "sampling_point_multiplier must be a positive integer"
        )
    return value


def _pool_count(spec: Mapping[str, Any], pool: str) -> int:
    return int(
        ((((spec.get("sampling") or {}).get(pool) or {}).get("parameters") or {}).get(
            "n_points"
        ))
        or 0
    )


def _problem_contract(problem: Any) -> dict[str, Any]:
    provider = getattr(problem, "sampling_contract", None)
    raw = provider() if callable(provider) else None
    if raw is None:
        raw = (problem.get_spec().metadata or {}).get("sampling_contract") or {}
    if not isinstance(raw, Mapping):
        raise SamplingContractError("sampling_contract must be a mapping")
    contract = dict(raw)
    policy = str(contract.get("allocation_policy", "shared_role_budget"))
    if policy != "shared_role_budget":
        raise SamplingContractError(
            f"Unsupported sampling allocation policy {policy!r}"
        )
    rules = contract.get("source_rules", contract.get("components", {})) or {}
    if not isinstance(rules, Mapping):
        raise SamplingContractError(
            "sampling_contract.source_rules must be a mapping"
        )
    contract["source_rules"] = dict(rules)
    return contract


def _merge_rule(
    rules: Mapping[str, Any], keys: tuple[str, ...]
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for key in keys:
        raw = rules.get(key)
        if raw is None:
            continue
        if not isinstance(raw, Mapping):
            raise SamplingContractError(
                f"Sampling source rule {key!r} must be a mapping"
            )
        merged.update(raw)
    return merged


def _allocate_pool(
    pool: str,
    total: int,
    definitions: list[dict[str, Any]],
) -> dict[str, int]:
    if total < 0:
        raise SamplingContractError(
            f"Sampling pool {pool!r} cannot have a negative point count"
        )
    if not definitions:
        return {}

    allocations: dict[str, int] = {}
    flexible: list[dict[str, Any]] = []
    committed = 0
    for item in definitions:
        sampler_id = str(item["sampler_id"])
        fixed = item["fixed_points"]
        minimum = int(item["minimum_points"])
        if minimum < 0:
            raise SamplingContractError(
                f"minimum_points for {sampler_id!r} must be non-negative"
            )
        if fixed is not None:
            fixed = int(fixed)
            if fixed < minimum:
                raise SamplingContractError(
                    f"fixed_points for {sampler_id!r} is below minimum_points"
                )
            allocations[sampler_id] = fixed
            committed += fixed
        else:
            allocations[sampler_id] = minimum
            committed += minimum
            flexible.append(item)

    if committed > total:
        raise SamplingContractError(
            f"Sampling pool {pool!r} requires at least {committed} points, "
            f"but AlgorithmSpec declares {total}"
        )
    remaining = total - committed
    if remaining and not flexible:
        raise SamplingContractError(
            f"Sampling pool {pool!r} has {remaining} unallocatable points because "
            "all active sources use fixed_points"
        )
    if not remaining:
        return allocations

    weights = [float(item["allocation_weight"]) for item in flexible]
    if any(not math.isfinite(value) or value < 0 for value in weights):
        raise SamplingContractError(
            f"Sampling pool {pool!r} has an invalid allocation weight"
        )
    if sum(weights) <= 0:
        weights = [1.0] * len(flexible)
    weight_sum = sum(weights)
    shares = [remaining * weight / weight_sum for weight in weights]
    floors = [math.floor(value) for value in shares]
    for item, value in zip(flexible, floors):
        allocations[str(item["sampler_id"])] += int(value)
    remainder = remaining - sum(floors)
    order = sorted(
        range(len(flexible)),
        key=lambda index: (
            -(shares[index] - floors[index]),
            str(flexible[index]["sampler_id"]),
        ),
    )
    for index in order[:remainder]:
        allocations[str(flexible[index]["sampler_id"])] += 1
    return allocations


def resolve_sampling_budget_plan(
    problem: Any, spec: Mapping[str, Any]
) -> SamplingBudgetPlan:
    """Resolve active unique samplers and allocate each declared pool once."""

    problem_spec = problem.get_spec()
    constraints = {
        str(item.constraint_id): item for item in problem_spec.constraints
    }
    grouped: dict[str, dict[str, Any]] = {
        "sample:interior": {
            "sampler_id": "sample:interior",
            "domain_role": "interior",
            "constraint_category": None,
            "point_multiplier": 1,
            "component_ids": [],
            "rule_keys": ["sample:interior", "interior"],
        }
    }

    for term in ((spec.get("loss") or {}).get("terms") or []):
        component_id = str(term.get("name") or "")
        parameters = dict(term.get("parameters") or {})
        source_id = str(parameters.get("component_name") or component_id)
        constraint = constraints.get(source_id) or constraints.get(component_id)
        legacy_wave_aggregate = (
            str(problem_spec.problem_id) == "wave_1d"
            and component_id == "boundary_condition"
        )
        expanded_constraints: list[Any] = []
        if constraint is None and component_id in INITIAL_ALIASES:
            expanded_constraints = [
                item
                for item in problem_spec.constraints
                if bool(item.required)
                and not bool((item.metadata or {}).get("aggregate"))
                and constraint_domain_role(
                    str(item.constraint_type), item.metadata
                )
                == "initial_surface"
            ]
        elif (
            constraint is None
            and component_id in BOUNDARY_ALIASES
            and not legacy_wave_aggregate
        ):
            expanded_constraints = [
                item
                for item in problem_spec.constraints
                if bool(item.required)
                and not bool((item.metadata or {}).get("aggregate"))
                and constraint_domain_role(
                    str(item.constraint_type), item.metadata
                )
                == "boundary"
            ]
        if expanded_constraints:
            for concrete in expanded_constraints:
                role, runtime_category, sampler_id = (
                    constraint_sampling_identity(concrete)
                )
                item = grouped.setdefault(
                    sampler_id,
                    {
                        "sampler_id": sampler_id,
                        "domain_role": role,
                        "constraint_category": runtime_category,
                        "point_multiplier": _constraint_point_multiplier(
                            concrete
                        ),
                        "component_ids": [],
                        "rule_keys": [],
                    },
                )
                if component_id not in item["component_ids"]:
                    item["component_ids"].append(component_id)
                for key in (
                    str(concrete.constraint_id),
                    component_id,
                    sampler_id,
                ):
                    if key not in item["rule_keys"]:
                        item["rule_keys"].append(key)
            continue
        if constraint is not None:
            role, runtime_category, sampler_id = constraint_sampling_identity(
                constraint
            )
            category: str | None = runtime_category
            point_multiplier = _constraint_point_multiplier(constraint)
            rule_keys = [
                str(constraint.constraint_id),
                component_id,
                source_id,
                sampler_id,
            ]
        elif component_id in EQUATION_ALIASES or component_id == "gradient_residual":
            sampler_id = "sample:interior"
            role = "interior"
            category = None
            point_multiplier = 1
            rule_keys = [component_id, sampler_id, "interior"]
        elif component_id in INITIAL_ALIASES:
            sampler_id = "sample:initial_surface:initial"
            role = "initial_surface"
            category = "initial"
            point_multiplier = 1
            rule_keys = [component_id, sampler_id, "initial"]
        elif component_id in BOUNDARY_ALIASES:
            sampler_id = "sample:boundary:boundary"
            role = "boundary"
            category = "boundary"
            point_multiplier = 1
            rule_keys = [component_id, sampler_id, "boundary"]
        elif component_id == "conservation":
            sampler_id = "sample:custom_constraint_domain:conservation"
            role = "custom_constraint_domain"
            category = "conservation"
            point_multiplier = 1
            rule_keys = [component_id, sampler_id]
        else:
            sampler_id = "sample:custom_constraint_domain:boundary"
            role = "custom_constraint_domain"
            category = "boundary"
            point_multiplier = 1
            rule_keys = [component_id, sampler_id]

        item = grouped.setdefault(
            sampler_id,
            {
                "sampler_id": sampler_id,
                "domain_role": role,
                "constraint_category": category,
                "point_multiplier": point_multiplier,
                "component_ids": [],
                "rule_keys": [],
            },
        )
        if component_id not in item["component_ids"]:
            item["component_ids"].append(component_id)
        for key in rule_keys:
            if key and key not in item["rule_keys"]:
                item["rule_keys"].append(key)

    contract = _problem_contract(problem)
    rules = contract["source_rules"]
    definitions: list[dict[str, Any]] = []
    for sampler_id in sorted(grouped):
        raw = grouped[sampler_id]
        rule = _merge_rule(rules, tuple(raw["rule_keys"]))
        pool = str(rule.get("budget_pool", default_budget_pool(raw["domain_role"])))
        if pool not in {"interior", "boundary", "initial"}:
            raise SamplingContractError(
                f"Sampling source {sampler_id!r} uses unknown budget_pool {pool!r}"
            )
        fixed = rule.get("fixed_points")
        minimum_default = 0 if fixed == 0 else 1
        point_multiplier = int(
            rule.get("point_multiplier", raw.get("point_multiplier", 1))
        )
        if point_multiplier < 1:
            raise SamplingContractError(
                f"point_multiplier for {sampler_id!r} must be positive"
            )
        definitions.append(
            {
                **raw,
                "budget_pool": pool,
                "fixed_points": None if fixed is None else int(fixed),
                "minimum_points": int(
                    rule.get("minimum_points", minimum_default)
                ),
                "allocation_weight": float(
                    rule.get("allocation_weight", rule.get("weight", 1.0))
                ),
                "point_multiplier": point_multiplier,
            }
        )

    declared = {
        pool: _pool_count(spec, pool)
        for pool in ("interior", "boundary", "initial")
    }
    allocations: dict[str, int] = {}
    allocated_pools: dict[str, int] = {}
    for pool in ("interior", "boundary", "initial"):
        pool_definitions = [
            item for item in definitions if item["budget_pool"] == pool
        ]
        pool_allocations = _allocate_pool(
            pool, declared[pool], pool_definitions
        )
        allocations.update(pool_allocations)
        allocated_pools[pool] = sum(pool_allocations.values())

    sources = tuple(
        SamplingSourcePlan(
            sampler_id=str(item["sampler_id"]),
            domain_role=str(item["domain_role"]),
            constraint_category=(
                None
                if item["constraint_category"] is None
                else str(item["constraint_category"])
            ),
            budget_pool=str(item["budget_pool"]),
            associated_loss_component_ids=tuple(item["component_ids"]),
            point_budget=int(allocations[item["sampler_id"]]),
            point_multiplier=int(item["point_multiplier"]),
            allocation_weight=float(item["allocation_weight"]),
            minimum_points=int(item["minimum_points"]),
            fixed_points=item["fixed_points"],
        )
        for item in definitions
    )
    return SamplingBudgetPlan(
        sources=sources,
        declared_pools=declared,
        allocated_pools=allocated_pools,
    )
