"""Design objectives for the five independent Top-3-driven evolution calls."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


ALLOWED_VARIATION_STRATEGIES = (
    "single_parent_mutation",
    "multi_parent_crossover",
    "crossover_then_mutation",
    "multi_parent_recomposition",
    "large_scale_mutation",
    "novel_recomposition",
)


CANDIDATE_ROLE_DEFINITIONS: dict[str, dict[str, Any]] = {
    "elite_conservative_improvement": {
        "default_temperature": 0.20,
        "stagnation_temperature_delta": 0.0,
        "stagnation_minimum_major_module_changes": 1,
        "stagnation_requirements": (
            "Remain conservative but apply at least one evidence-backed executable correction.",
            "Do not replay Rank-1 exactly.",
        ),
        "objective": (
            "Use Rank-1 of the controller-selected global low-fidelity distinct Top-3 as the primary parent, preserve its "
            "measured strengths, and conservatively repair its most important measured weakness."
        ),
        "requirements": (
            "Global Rank-1 must be included in parent_ids and treated as the primary parent; global ranks 2 and 3 are supporting evidence.",
            "Preserve evidence-supported Rank-1 components and target its largest measured training bottleneck.",
            "Concentrate on one principal problem and prefer one to three low-risk executable changes.",
            "Do not perform a large structural reconstruction or enable several high-risk mechanisms together.",
            "State inherited strengths, changed modules, design rationale, expected improvement, and remaining risks.",
        ),
    },
    "physics_constraint_sampling_guided_design": {
        "default_temperature": 0.36,
        "stagnation_temperature_delta": 0.08,
        "stagnation_minimum_major_module_changes": 1,
        "stagnation_requirements": (
            "Prioritize unresolved PDE, initial, boundary, conservation, interface, pressure-anchor, or causal failures.",
            "Use residual spatial evidence and per-equation residuals directly.",
            "When a sampling pattern repeatedly fails, change sampler family, refinement schedule, or point-budget allocation rather than uniformly increasing all point counts.",
        ),
        "primary_axis": "PDE physics, constraint enforcement, and residual-aware sampling",
        "major_modules": ("sampling", "constraints", "loss"),
        "principle_focus": ("macro_physics_principles",),
        "input_focus": (
            "problem_features",
            "problem_spec.constraints",
            "pinnacle_profile.constraint_enforcement_options",
            "parent_specs.formal_metrics",
            "parent_specs.physical_supervision_summary",
            "latest_generation_posterior",
        ),
        "modifiable_modules": ("sampling", "constraint_enforcement", "loss"),
        "objective": (
            "Jointly design constraint enforcement and training-point allocation from the PDE type, "
            "derivative order, conservation or causal structure, residual distribution, and measured "
            "boundary and initial-condition errors."
        ),
        "requirements": (
            "Inspect all three controller-selected global low-fidelity parents before selecting lineage and a variation strategy.",
            "Translate immutable PDE properties and relevant macro-physics principles into executable supervision decisions.",
            "Inspect per-equation residuals, residual quantiles and spatial concentration instead of relying only on global MSE.",
            "Determine whether error comes from inadequate constraints, competing constraints, poor local coverage, or imbalanced interior, boundary, and initial allocations.",
            "Make the primary executable changes in sampling, registered constraint enforcement, or boundary/initial loss treatment.",
            "Preserve the immutable PDE and mathematical boundary and initial conditions; modify only how they are sampled or enforced.",
            "Maintain global coverage and IC/BC anchors when introducing residual-adaptive, boundary-focused, or localized sampling.",
            "Do not evade an algorithm-design problem by increasing all sample counts without a bounded physical justification.",
        ),
    },
    "architecture_optimization_guided_design": {
        "default_temperature": 0.42,
        "stagnation_temperature_delta": 0.08,
        "stagnation_minimum_major_module_changes": 1,
        "stagnation_requirements": (
            "Use frequency, smoothness, localization, capacity, and component-gradient evidence to coordinate representation and optimization changes.",
            "Do not use only a width or learning-rate perturbation as the escape mechanism.",
            "Do not claim an optimization failure without measured evidence.",
        ),
        "primary_axis": "co-designed representation, gradient dynamics, and optimization process",
        "major_modules": ("network", "optimization", "loss", "training"),
        "principle_focus": ("architecture_design_principles",),
        "input_focus": (
            "generation_feedback",
            "parent_specs.algorithm_spec.network",
            "parent_specs.training_summary",
            "parent_specs.optimization_feedback",
            "latest_generation_posterior.training_dynamics",
            "active_run_posterior_rules",
        ),
        "modifiable_modules": ("network", "optimization", "loss", "training"),
        "objective": (
            "Jointly redesign network representation and optimization using solution frequency, "
            "smoothness, derivative order, capacity, gradient flow, loss dynamics, and measured convergence evidence."
        ),
        "requirements": (
            "Inspect all three controller-selected global low-fidelity parents before selecting lineage and a variation strategy.",
            "Assess capacity, representation bias, activation and input-transform suitability, and derivative stability.",
            "Inspect loss trajectories, reference-MSE probes, gradient norms, gradient conflicts, plateaus, instability, and optimizer audits.",
            "Coordinate network type, depth, width, activation, residual or frequency features with optimizer stages, schedulers, and loss weighting.",
            "Use measured evidence to enable or disable a final fine-tuning phase; do not append L-BFGS by convention.",
            "Form one coherent representation-gradient-optimization design; do not simultaneously expand the network, samples, and training length without evidence.",
            "Create a meaningful executable difference from every global low-fidelity Top-3 parent.",
        ),
    },
    "multi_parent_synthesis": {
        "default_temperature": 0.47,
        "stagnation_temperature_delta": 0.10,
        "stagnation_minimum_major_module_changes": 2,
        "stagnation_requirements": (
            "Use at least two complementary global low-fidelity Top-3 parent sources.",
            "Change at least two major executable modules and identify each inherited mechanism.",
            "Avoid repeatedly failed mechanisms unless the proposal explicitly repairs their failure condition.",
        ),
        "objective": (
            "Synthesize compatible measured strengths from at least two controller-selected global "
            "low-fidelity distinct Top-3 parents into one coherent new AlgorithmSpec."
        ),
        "requirements": (
            "Use at least two Top-3 candidate IDs in parent_ids.",
            "Let the LLM choose crossover, crossover-then-mutation, or recomposition; Python must not splice fields.",
            "Identify which measured mechanism came from each selected parent.",
            "Identify conflicting parent modules and explicitly resolve their compatibility.",
            "Explain why the unified design can outperform each selected parent rather than merely listing inherited components.",
            "The normalized executable result must differ from every parent and every sibling.",
        ),
    },
    "novelty_exploration": {
        "default_temperature": 0.58,
        "stagnation_temperature_delta": 0.15,
        "stagnation_minimum_major_module_changes": 2,
        "stagnation_requirements": (
            "Change at least two compatible major executable modules.",
            "Avoid repeatedly failed mechanisms and explain the stability mitigation for the escape hypothesis.",
        ),
        "primary_axis": "a physically justified cross-module hypothesis not covered by the other four roles",
        "major_modules": (
            "network",
            "sampling",
            "constraints",
            "loss",
            "optimization",
            "training",
        ),
        "objective": (
            "Use the full controller-selected global low-fidelity distinct Top-3 as evidence while producing a physically valid, "
            "meaningfully novel cross-module design that avoids parent algorithms and known failed combinations."
        ),
        "requirements": (
            "Inspect all three controller-selected global low-fidelity parents before selecting lineage and a variation strategy.",
            "Maintain a meaningful normalized difference from every global low-fidelity Top-3 parent and the other role objectives.",
            "Target a concrete measured or inferred failure mechanism and do not ignore confirmed run-scoped failure patterns.",
            "Propose a physically justified cross-module hypothesis not covered by the other four roles.",
            "Change at least two mutually compatible major executable modules.",
            "Do not randomly stack complex components or modify the fixed PDE and boundary or initial conditions.",
            "State the novelty source, expected benefit, numerical risk, and corresponding stability mitigation.",
        ),
    },
}


INDEPENDENT_PROPOSAL_ROLE_DEFINITIONS: dict[str, dict[str, Any]] = {
    "robust_independent_design": {
        "default_temperature": 0.20,
        "stagnation_temperature_delta": 0.08,
        "stagnation_minimum_major_module_changes": 1,
        "primary_axis": "robust parent-free complete algorithm design",
        "major_modules": ("network", "constraints", "optimization"),
        "input_focus": (
            "problem_features",
            "macro_physics_principles",
            "architecture_design_principles",
            "generation_feedback",
            "active_run_posterior_rules",
        ),
        "objective": (
            "Propose a robust complete AlgorithmSpec from the PDE, knowledge, aggregate execution feedback, "
            "and run-scoped posterior summaries without viewing or inheriting any previous AlgorithmSpec."
        ),
        "requirements": (
            "Start from a fresh design hypothesis rather than reconstructing a previous candidate.",
            "Use execution summaries to avoid repeated failure modes while keeping the design moderate-risk and trainable.",
            "Do not request, infer, name, copy, mutate, or recombine a parent AlgorithmSpec.",
        ),
        "stagnation_requirements": (
            "Change the independent design hypothesis when the global archive has stagnated.",
        ),
    },
    "physics_constraint_sampling_design": {
        "default_temperature": 0.36,
        "stagnation_temperature_delta": 0.08,
        "stagnation_minimum_major_module_changes": 1,
        "primary_axis": "parent-free physics, constraint, and sampling design",
        "major_modules": ("sampling", "constraints", "loss"),
        "principle_focus": ("macro_physics_principles",),
        "input_focus": (
            "problem_features",
            "problem_spec.constraints",
            "generation_feedback",
            "latest_generation_posterior",
            "posterior_failure_patterns",
        ),
        "modifiable_modules": ("sampling", "constraint_enforcement", "loss"),
        "objective": (
            "Create a fresh complete AlgorithmSpec centered on physics supervision, constraint enforcement, "
            "and sampling using aggregate residual and constraint feedback but no previous AlgorithmSpec."
        ),
        "requirements": (
            "Translate residual, boundary, initial-condition, conservation, and localization summaries into a new supervision design.",
            "Preserve the immutable PDE and mathematical conditions.",
            "Do not select a lineage or reproduce sampling, loss, or constraint fields from a parent specification.",
        ),
        "stagnation_requirements": (
            "Use unresolved physical failures to form a different parent-free supervision hypothesis.",
        ),
    },
    "architecture_optimization_design": {
        "default_temperature": 0.42,
        "stagnation_temperature_delta": 0.08,
        "stagnation_minimum_major_module_changes": 1,
        "primary_axis": "parent-free representation and optimization co-design",
        "major_modules": ("network", "optimization", "loss", "training"),
        "principle_focus": ("architecture_design_principles",),
        "input_focus": (
            "problem_features",
            "generation_feedback",
            "latest_generation_posterior.training_dynamics",
            "active_run_posterior_rules",
        ),
        "modifiable_modules": ("network", "optimization", "loss", "training"),
        "objective": (
            "Create a fresh representation--optimization design from PDE structure and aggregate training diagnostics "
            "without viewing or modifying any previous AlgorithmSpec."
        ),
        "requirements": (
            "Use loss, gradient, plateau, and stability summaries only as population-level diagnostic evidence.",
            "Coordinate architecture, activation, optimizer, scheduler, and loss weighting coherently.",
            "Do not perform conservative mutation or use any previous network or optimizer configuration as a base.",
        ),
        "stagnation_requirements": (
            "Form a materially different independent representation--optimization hypothesis under stagnation.",
        ),
    },
    "cross_module_independent_design": {
        "default_temperature": 0.47,
        "stagnation_temperature_delta": 0.10,
        "stagnation_minimum_major_module_changes": 2,
        "primary_axis": "parent-free cross-module coordination",
        "major_modules": ("network", "sampling", "loss", "optimization"),
        "input_focus": (
            "problem_features",
            "generation_feedback",
            "posterior_recommended_next_actions",
            "posterior_unresolved_hypotheses",
        ),
        "objective": (
            "Produce a coherent fresh cross-module AlgorithmSpec from knowledge and search summaries, "
            "without multi-parent synthesis or access to any previous executable specification."
        ),
        "requirements": (
            "Coordinate at least two compatible executable modules around one causal hypothesis.",
            "Use population-level successes and failures as guidance, not as sources of fields to inherit.",
            "Do not splice, recombine, or cite mechanisms from identifiable parent AlgorithmSpecs.",
        ),
        "stagnation_requirements": (
            "Change at least two compatible modules around a new independent hypothesis.",
        ),
    },
    "novelty_independent_design": {
        "default_temperature": 0.58,
        "stagnation_temperature_delta": 0.15,
        "stagnation_minimum_major_module_changes": 2,
        "primary_axis": "novel parent-free complete algorithm design",
        "major_modules": (
            "network",
            "sampling",
            "constraints",
            "loss",
            "optimization",
            "training",
        ),
        "input_focus": (
            "problem_features",
            "macro_physics_principles",
            "architecture_design_principles",
            "posterior_unresolved_hypotheses",
        ),
        "objective": (
            "Propose a physically justified novel complete AlgorithmSpec from scratch using knowledge and aggregate "
            "search evidence, with no parent specification, mutation, or recombination."
        ),
        "requirements": (
            "Introduce a meaningful parent-free cross-module hypothesis not covered by the other independent roles.",
            "State the expected benefit, numerical risk, and mitigation.",
            "Do not randomly stack components or reconstruct a previous high-performing design.",
        ),
        "stagnation_requirements": (
            "Use at least two compatible module changes to escape the current search pattern.",
        ),
    },
}

CANDIDATE_ROLE_DEFINITIONS.update(INDEPENDENT_PROPOSAL_ROLE_DEFINITIONS)

INDEPENDENT_PROPOSAL_ROLE_PLAN = (
    "robust_independent_design",
    "physics_constraint_sampling_design",
    "architecture_optimization_design",
    "cross_module_independent_design",
    "novelty_independent_design",
)


# These objectives preserve the same five independent design axes while
# removing every request to inspect measured outcomes, ranks, diagnostics, or
# stagnation evidence.  They are selected at the existing role-resolution
# point; no separate prompt-sanitization layer is introduced.
FEEDBACK_FREE_CANDIDATE_ROLE_OVERRIDES: dict[str, dict[str, Any]] = {
    "elite_conservative_improvement": {
        "objective": (
            "Choose one of the three unordered parent AlgorithmSpecs as a structural anchor and "
            "produce a conservative, executable refinement based only on its declared design."
        ),
        "requirements": (
            "Use at least one parent ID without assuming that list position indicates quality.",
            "Preserve a coherent subset of the chosen parent's declared components and make one to three low-risk executable changes.",
            "Justify changes from the PDE specification, prior knowledge, and AlgorithmSpec structure only.",
            "Do not claim that any parent component succeeded, failed, converged, or ranked better.",
        ),
    },
    "physics_constraint_sampling_guided_design": {
        "objective": (
            "Design constraint enforcement and training-point allocation from immutable PDE "
            "properties and the unordered parent AlgorithmSpecs, using no runtime residual observations."
        ),
        "input_focus": (
            "problem_features",
            "problem_spec.constraints",
            "pinnacle_profile.constraint_enforcement_options",
            "parent_specs.algorithm_spec",
        ),
        "requirements": (
            "Inspect all three unordered parent AlgorithmSpecs before selecting lineage.",
            "Translate PDE type, derivative order, domain geometry, and boundary or initial conditions into executable supervision decisions.",
            "Make the primary changes in sampling, registered constraint enforcement, or boundary/initial loss treatment.",
            "Preserve the immutable PDE and mathematical conditions; modify only how they are sampled or enforced.",
            "Do not assert residual concentration, constraint outcomes, or parent performance.",
        ),
    },
    "architecture_optimization_guided_design": {
        "objective": (
            "Co-design representation and optimization from PDE features, numerical principles, "
            "and the unordered parent AlgorithmSpecs without training-dynamics evidence."
        ),
        "input_focus": (
            "problem_features",
            "parent_specs.algorithm_spec.network",
            "parent_specs.algorithm_spec.optimization",
            "parent_specs.algorithm_spec.loss",
            "parent_specs.algorithm_spec.training",
        ),
        "requirements": (
            "Inspect all three unordered parent AlgorithmSpecs before selecting lineage.",
            "Reason about capacity, representation bias, derivative stability, and optimizer compatibility from declared structure only.",
            "Coordinate network, activation or input transforms with optimizer stages and loss weighting.",
            "Do not claim a plateau, gradient conflict, convergence result, or empirical benefit.",
            "Create a meaningful executable difference from every parent.",
        ),
    },
    "multi_parent_synthesis": {
        "objective": (
            "Synthesize compatible declared mechanisms from at least two unordered parent "
            "AlgorithmSpecs into one coherent new design."
        ),
        "requirements": (
            "Use at least two parent IDs without treating their order as a quality signal.",
            "Let the LLM choose crossover, crossover-then-mutation, or recomposition; Python must not splice fields.",
            "Identify the declared mechanism inherited from each selected parent and resolve module conflicts.",
            "Justify the synthesis structurally and physically without performance comparisons.",
            "The normalized executable result must differ from every parent and sibling.",
        ),
    },
    "novelty_exploration": {
        "objective": (
            "Use the three unordered parent AlgorithmSpecs as structural references while proposing "
            "a physically valid, meaningfully novel cross-module design."
        ),
        "requirements": (
            "Inspect all three unordered parent AlgorithmSpecs before selecting lineage.",
            "Maintain a meaningful normalized difference from every parent and other role objective.",
            "Propose a physically justified cross-module hypothesis based on PDE and design structure only.",
            "Change at least two mutually compatible major executable modules.",
            "State numerical risks and mitigations without claiming observed failures or improvements.",
        ),
    },
}


NEW_CANDIDATE_ROLE_PLAN = (
    "elite_conservative_improvement",
    "physics_constraint_sampling_guided_design",
    "architecture_optimization_guided_design",
    "multi_parent_synthesis",
    "novelty_exploration",
)


def candidate_role_definition(
    role_name: str, *, execution_feedback: bool = True
) -> dict[str, Any] | None:
    """Return the existing role contract or its feedback-free wording."""

    configured = CANDIDATE_ROLE_DEFINITIONS.get(str(role_name))
    if configured is None:
        return None
    resolved = deepcopy(configured)
    if not execution_feedback:
        resolved.update(deepcopy(FEEDBACK_FREE_CANDIDATE_ROLE_OVERRIDES[str(role_name)]))
        resolved["stagnation_requirements"] = ()
    return resolved


def resolve_candidate_role(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Resolve a role without assigning any fixed algorithm component to it."""

    role_name = payload.get("candidate_role")
    if role_name is None:
        return None
    role_name = str(role_name)
    configured = candidate_role_definition(
        role_name,
        execution_feedback=bool(payload.get("execution_feedback", True)),
    )
    if configured is None:
        from .initialization_candidate_roles import GENERATION_ZERO_ROLE_DEFINITIONS

        configured = GENERATION_ZERO_ROLE_DEFINITIONS.get(role_name)
    if configured is None:
        raise ValueError(f"Unknown candidate_role: {role_name}")
    objective = str(payload.get("candidate_role_objective") or configured["objective"])
    requirements = [str(item) for item in configured["requirements"]]
    return {
        "name": role_name,
        "objective": objective,
        "requirements": requirements,
        "default_temperature": float(configured["default_temperature"]),
        "primary_axis": configured.get("primary_axis"),
        "major_modules": list(configured.get("major_modules") or []),
        "principle_focus": list(configured.get("principle_focus") or []),
        "input_focus": list(configured.get("input_focus") or []),
        "modifiable_modules": list(configured.get("modifiable_modules") or []),
        "stagnation_temperature_delta": float(
            configured.get("stagnation_temperature_delta") or 0.0
        ),
        "stagnation_minimum_major_module_changes": int(
            configured.get("stagnation_minimum_major_module_changes") or 1
        ),
        "stagnation_requirements": list(
            configured.get("stagnation_requirements") or []
        ),
    }
