"""Complementary design objectives for the eight Generation-0 proposals."""

from __future__ import annotations

from typing import Any


GENERATION_ZERO_ROLE_DEFINITIONS: dict[str, dict[str, Any]] = {
    "robust_reference_anchor": {
        "default_temperature": 0.20,
        "primary_axis": "simple robust reference",
        "major_modules": ("network", "constraints"),
        "objective": (
            "Create a clear, moderate-complexity, low-risk reference design that treats the "
            "PDE, initial conditions, and boundary conditions reliably without copying the default template."
        ),
        "requirements": (
            "Prefer reliable, interpretable, physically justified choices.",
            "Avoid enabling several high-risk mechanisms simultaneously.",
            "Use problem-specific parameters rather than mechanically copying defaults.",
        ),
    },
    "constraint_causality_specialist": {
        "default_temperature": 0.35,
        "primary_axis": "constraints and temporal causality",
        "major_modules": ("constraints", "training"),
        "principle_focus": ("macro_physics_principles",),
        "objective": (
            "Use the supplied macro-physics principles to focus on initial and boundary "
            "conditions, temporal causality, and propagation of constraint errors through training."
        ),
        "requirements": (
            "Inspect pinnacle_profile.constraint_enforcement_options; when a registered problem_hard option exists, use one for this role and state a direct soft-versus-hard comparison hypothesis.",
            "Translate every applicable supplied macro-physics principle into explicit IC/BC, "
            "constraint-consistency, conservation, or temporal-causality decisions.",
            "Design IC/BC treatment and causal training behavior explicitly.",
            "Make a material executable difference in constraints or training stages.",
            "Keep representation and sampling conservative unless supporting the constraint design.",
        ),
    },
    "localized_sampling_specialist": {
        "default_temperature": 0.40,
        "primary_axis": "sampling and local refinement",
        "major_modules": ("sampling", "constraints"),
        "objective": (
            "Allocate sampling resources for shocks, steep gradients, and localized residuals "
            "while preserving global coverage and IC/BC anchors."
        ),
        "requirements": (
            "Prioritize interior sampling and adaptive refinement choices.",
            "Avoid collapsing all samples into one localized region.",
            "Create a material sampling difference without prescribing a fixed sampler.",
        ),
    },
    "representation_specialist": {
        "default_temperature": 0.45,
        "primary_axis": "network representation",
        "major_modules": ("network", "sampling"),
        "objective": (
            "Explore representation capacity, spectral bias, multiscale structure, and shock "
            "representation through coherent executable network choices."
        ),
        "requirements": (
            "Prioritize architecture, activation, topology, and legal feature encoding.",
            "Relate the representation to the current PDE rather than requiring a named architecture.",
            "Coordinate sampling and optimization with the selected representation.",
        ),
    },
    "residual_loss_specialist": {
        "default_temperature": 0.45,
        "primary_axis": "residual and loss construction",
        "major_modules": ("loss", "constraints"),
        "objective": (
            "Design the governing residual, constraint losses, weighting, aggregation, and any "
            "legal derivative residuals as a stable, cost-aware loss system."
        ),
        "requirements": (
            "Prioritize loss structure and balance among PDE, IC, and BC terms.",
            "Account for higher-order automatic-differentiation stability and cost.",
            "Use only loss components justified by supplied principles and the registry.",
        ),
    },
    "optimization_conditioning_specialist": {
        "default_temperature": 0.40,
        "primary_axis": "optimization and conditioning",
        "major_modules": ("optimization", "training"),
        "objective": (
            "Design the complete optimization phases, scheduling, gradient conditioning, and "
            "training stability protocol for the full iteration budget."
        ),
        "requirements": (
            "Create a material optimizer, scheduler, or training-schedule difference.",
            "Decide explicitly whether a dedicated final fine-tuning phase is justified; do not equate fine-tuning with L-BFGS or enable it by convention.",
            "Address gradient competition between loss terms.",
            "Do not add unjustified network complexity.",
        ),
    },
    "capacity_topology_explorer": {
        "default_temperature": 0.50,
        "primary_axis": "capacity and topology",
        "major_modules": ("network", "optimization"),
        "objective": (
            "Explore a materially different allocation of model capacity, depth, width, and "
            "topology under derivative-computation and parameter budgets."
        ),
        "requirements": (
            "Do more than perturb hidden widths by a few units.",
            "Balance representation benefit against parameter and derivative cost.",
            "Explain why the capacity profile suits the PDE.",
        ),
    },
    "cross_module_novel_synthesis": {
        "default_temperature": 0.60,
        "primary_axis": "novel coherent synthesis",
        "major_modules": ("network", "sampling", "loss", "optimization"),
        "objective": (
            "Propose a physically justified, coherent cross-module hypothesis not already "
            "covered by the other seven role objectives."
        ),
        "requirements": (
            "Derive novelty from multiple compatible executable modules.",
            "Do not randomly stack complex components or copy a fixed common combination.",
            "State the principal benefit, risk, and stability mitigation.",
        ),
    },
}


GENERATION_ZERO_ROLE_PLAN = (
    "robust_reference_anchor",
    "constraint_causality_specialist",
    "localized_sampling_specialist",
    "representation_specialist",
    "residual_loss_specialist",
    "optimization_conditioning_specialist",
    "capacity_topology_explorer",
    "cross_module_novel_synthesis",
)


def generation_zero_role_map() -> list[dict[str, str]]:
    return [
        {
            "role": role_name,
            "primary_axis": str(GENERATION_ZERO_ROLE_DEFINITIONS[role_name]["primary_axis"]),
        }
        for role_name in GENERATION_ZERO_ROLE_PLAN
    ]
