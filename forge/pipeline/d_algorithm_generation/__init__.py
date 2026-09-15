"""Stage D: generate complete AlgorithmSpec candidates with an LLM."""

from .agents.design_agent import DesignAgent
from .specs import algorithm_spec_template
from .adaptive_policy import (
    AdaptivePolicyBudgetScaler,
    RuntimeScaledTrainingPolicy,
    adaptive_policy_spec,
    build_trainer_capability_summary,
)

__all__ = [
    "DesignAgent",
    "algorithm_spec_template",
    "AdaptivePolicyBudgetScaler",
    "RuntimeScaledTrainingPolicy",
    "adaptive_policy_spec",
    "build_trainer_capability_summary",
]
