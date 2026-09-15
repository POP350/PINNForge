"""Feature extraction from benchmark objects or benchmark IDs."""

from __future__ import annotations

from forge.pipeline.a_problem_definition.problems.schemas import PhysicsProblemSpec

from .feature_schema import PhysicsProblemFeature, ProblemFeature


def extract_problem_features(benchmark: object) -> dict:
    if hasattr(benchmark, "get_spec"):
        return extract_physics_problem_feature(benchmark.get_spec()).as_search_dict()
    if hasattr(benchmark, "get_problem_features"):
        return benchmark.get_problem_features()
    if isinstance(benchmark, dict):
        return ProblemFeature(**benchmark).model_dump()
    raise TypeError("benchmark must expose get_problem_features() or be a feature dict")


def extract_physics_problem_feature(spec: PhysicsProblemSpec | dict) -> PhysicsProblemFeature:
    spec = spec if isinstance(spec, PhysicsProblemSpec) else PhysicsProblemSpec(**spec)
    law_types = list(dict.fromkeys(spec.law_types or [law.law_type for law in spec.governing_laws]))
    constraint_types = list(dict.fromkeys(constraint.constraint_type for constraint in spec.constraints))
    metadata = dict(spec.metadata)
    metadata["unknown_parameters"] = [
        parameter.model_dump() if hasattr(parameter, "model_dump") else dict(parameter)
        for parameter in spec.unknown_parameters
    ]
    differential_orders = [
        law.differential_order for law in spec.governing_laws
        if law.differential_order is not None
    ]
    return PhysicsProblemFeature(
        problem_id=spec.problem_id,
        law_types=law_types,
        task_type=spec.task_type,
        time_dependent=metadata.get("time_dependent", "t" in spec.input_variables or "time" in spec.input_variables),
        differential_order=max(differential_orders) if differential_orders else None,
        spatial_dimension=metadata.get("spatial_dimension"),
        state_dimension=len(spec.output_variables),
        is_stiff=bool(metadata.get("is_stiff", False)),
        is_multiscale=bool(metadata.get("has_multiscale", metadata.get("is_multiscale", False))),
        has_discontinuity=bool(metadata.get("has_discontinuity", False)),
        has_shock=bool(metadata.get("has_shock", False)),
        has_conservation_law=("conservation" in law_types) or bool(metadata.get("has_conservation_law", False)),
        has_constitutive_relation=("constitutive" in law_types)
        or bool(metadata.get("has_constitutive_relation", False)),
        has_algebraic_constraints=("algebraic_consistency" in constraint_types)
        or ("dae" in law_types)
        or bool(metadata.get("has_algebraic_constraints", False)),
        has_integral_constraints=("integral" in law_types)
        or bool(metadata.get("has_integral_constraints", False)),
        has_inequality_constraints=any(
            item in constraint_types for item in ["inequality", "positivity", "parameter_bound"]
        )
        or bool(metadata.get("has_inequality_constraints", False)),
        has_symmetry=("symmetry" in constraint_types) or bool(metadata.get("has_symmetry", False)),
        has_unknown_parameters=bool(spec.unknown_parameters)
        or bool(metadata.get("has_unknown_parameters", False)),
        data_available=bool(spec.observations and spec.observations.available),
        data_sparsity="sparse" if spec.observations and spec.observations.sparse else None,
        noise_level=spec.observations.noise_level if spec.observations else metadata.get("noise_level"),
        geometry_complexity=metadata.get("geometry"),
        constraint_types=constraint_types,
        metadata=metadata,
    )
