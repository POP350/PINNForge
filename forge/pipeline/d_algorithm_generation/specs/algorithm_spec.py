"""The single AlgorithmSpec transport contract used by the default pipeline.

The top-level structure is fixed. Independent legal field choices and their
parameter spaces live in :mod:`forge.pipeline.d_algorithm_generation.specs.option_registry`; complete
algorithm combinations are composed dynamically and are never pre-enumerated.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


REQUIRED_TOP_LEVEL_FIELDS = (
    "spec_version",
    "network",
    "sampling",
    "loss",
    "optimization",
    "training",
    "generation_metadata",
)


def algorithm_spec_template() -> dict[str, Any]:
    """Return a fresh complete example/template, never a shared mutable value."""

    return deepcopy(
        {
            "spec_version": "1.1",
            "network": {
                "architecture": "mlp",
                "hidden_layers": [128, 128, 128, 128],
                "activation": {"name": "tanh", "parameters": {}},
                "output_activation": {"name": "identity", "parameters": {}},
                "initialization": {"name": "xavier_normal", "parameters": {}},
                "input_transform": {"name": "none", "parameters": {}},
                "residual_connections": {"enabled": False, "parameters": {}},
                "extra_parameters": {},
            },
            "sampling": {
                "interior": {"strategy": "uniform", "parameters": {"n_points": 5000}},
                "boundary": {"strategy": "uniform", "parameters": {"n_points": 256}},
                "initial": {"strategy": "uniform", "parameters": {"n_points": 256}},
                "adaptive_refinement": {
                    "enabled": False,
                    "strategy": "residual_adaptive",
                    "parameters": {},
                },
            },
            "constraint_enforcement": {
                "method": "soft_penalty",
                "transform_id": "none",
                "parameters": {},
            },
            "loss": {
                "terms": [
                    {
                        "name": "pde_residual",
                        "loss_function": "mse",
                        "weight": 1.0,
                        "parameters": {},
                    },
                    {
                        "name": "boundary_condition",
                        "loss_function": "mse",
                        "weight": 10.0,
                        "parameters": {},
                    },
                    {
                        "name": "initial_condition",
                        "loss_function": "mse",
                        "weight": 10.0,
                        "parameters": {},
                    },
                ],
                "weighting_strategy": {"name": "fixed", "parameters": {}},
                "aggregation": {"name": "weighted_sum", "parameters": {}},
            },
            "optimization": {
                "phases": [
                    {
                        "optimizer": "adam",
                        "iterations": 3000,
                        "parameters": {"learning_rate": 0.001},
                        "scheduler": {"name": "none", "parameters": {}},
                    },
                    {
                        "optimizer": "lbfgs",
                        "iterations": 500,
                        "parameters": {
                            "learning_rate": 1.0,
                            "history_size": 100,
                            "line_search_fn": "strong_wolfe",
                        },
                        "scheduler": {"name": "none", "parameters": {}},
                    },
                ]
            },
            "training": {
                "batch_mode": "full_batch",
                "batch_size": None,
                "gradient_clipping": {"enabled": False, "parameters": {}},
                "early_stopping": {"enabled": False, "parameters": {}},
                "time_strategy": {"name": "none", "parameters": {}},
                "initial_state_pretraining": {
                    "enabled": False,
                    "parameters": {},
                },
                "extra_parameters": {
                    "adaptive_control": {
                        "enabled": False,
                        "freeze_lbfgs_objective": True,
                        "lbfgs_stall_recovery": {"enabled": False, "action": "disabled"},
                        "sampling_control": {"enabled": False, "strategy": "none"},
                        "curriculum_control": {"enabled": False, "strategy": "none"},
                        "rollback": {"enabled": True},
                    },
                    "physics_validation": {"enabled": False, "strategy": "disabled"},
                    "best_checkpoint": {"enabled": False, "final_model_policy": "last"},
                },
            },
            "generation_metadata": {
                "proposal_strategy": "knowledge_guided",
                "design_hypothesis": "",
                "parent_spec_ids": [],
                "changed_modules": [],
                "expected_advantages": [],
                "possible_risks": [],
                "macro_physics_principles_applied": [],
                "architecture_design_principles_applied": [],
                "posterior_evidence_applied": [],
                "principle_tradeoffs": [],
                "candidate_role": None,
                "role_design_objective": "",
                "role_requirements_applied": [],
                "role_input_focus": [],
                "role_modifiable_modules": [],
                "primary_diversity_axis": "",
                "diversity_hypothesis": "",
                "major_executable_differences_expected": [],
                "iteration_budget_rationale": "",
            },
        }
    )
