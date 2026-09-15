"""LLM-designed PINNs organized as an explicit A-to-I closed loop."""

from .pipeline.a_problem_definition import get_problem, list_problems
from .pipeline.h_evolution_reflection import run_open_algorithm_spec_search

__version__ = "0.1.0"

__all__ = [
    "get_problem",
    "list_problems",
    "run_open_algorithm_spec_search",
]
