"""Agents used by the Open AlgorithmSpec search loop."""

from forge.pipeline.d_algorithm_generation.agents.base_agent import AgentResult, BaseAgent
from .agentic_variation_operator import AgenticVariationOperator
from .candidate_roles import CANDIDATE_ROLE_DEFINITIONS, NEW_CANDIDATE_ROLE_PLAN
from .design_agent import DesignAgent
from .initialization_candidate_roles import (
    GENERATION_ZERO_ROLE_DEFINITIONS,
    GENERATION_ZERO_ROLE_PLAN,
)

__all__ = [
    "AgentResult",
    "AgenticVariationOperator",
    "BaseAgent",
    "CANDIDATE_ROLE_DEFINITIONS",
    "DesignAgent",
    "GENERATION_ZERO_ROLE_DEFINITIONS",
    "GENERATION_ZERO_ROLE_PLAN",
    "NEW_CANDIDATE_ROLE_PLAN",
]
