"""Public entrypoint for the Open AlgorithmSpec PINN search."""

from __future__ import annotations

from typing import Any

from forge.pipeline.h_evolution_reflection import run_open_algorithm_spec_search


def run_closed_loop(config: dict[str, Any]) -> dict[str, Any]:
    return run_open_algorithm_spec_search(dict(config))


__all__ = ["run_closed_loop"]
