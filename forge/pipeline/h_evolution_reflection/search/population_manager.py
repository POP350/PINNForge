"""Deterministic ranking, retention, diversity, and parent eligibility."""

from __future__ import annotations

from typing import Any

from forge.pipeline.g_training_evaluation.evaluation import PRIMARY_RANKING_METRIC


class PopulationManager:
    def __init__(
        self,
        maximum_size: int = 10,
        elite_count: int = 3,
        objective: str = PRIMARY_RANKING_METRIC,
    ) -> None:
        self.maximum_size = max(1, int(maximum_size))
        self.elite_count = max(1, int(elite_count))
        self.objective = objective
        self.population: list[dict[str, Any]] = []

    def update(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ranked = sorted([item for item in [*self.population, *records] if item.get("spec_id")], key=self._key)
        elites = ranked[: self.elite_count]
        diverse = []
        seen_architectures = {self._architecture(item) for item in elites}
        for item in ranked[self.elite_count :]:
            if self._architecture(item) not in seen_architectures:
                diverse.append(item)
                seen_architectures.add(self._architecture(item))
            if len(elites) + len(diverse) >= self.maximum_size:
                break
        for item in ranked:
            if item not in elites and item not in diverse and len(elites) + len(diverse) < self.maximum_size:
                diverse.append(item)
        self.population = [*elites, *diverse][: self.maximum_size]
        return list(self.population)

    def eligible_parents(self) -> list[dict[str, Any]]:
        """Return measured candidates without deciding how the LLM will use them."""

        return [item for item in self.population if item.get("training_success")]

    def summary(self) -> list[dict[str, Any]]:
        summaries = []
        for item in self.population:
            metrics = dict(item.get("metrics") or {})
            candidate_id = str(item.get("spec_id") or "unknown_candidate")
            summaries.append(
                {
                "spec_id": item.get("spec_id"),
                "metrics": {
                    key: metrics.get(key)
                    for key in (
                        "mse",
                        "mae",
                        "l1re",
                        "l2re",
                        "pde_residual",
                        "boundary_error",
                        "initial_error",
                    )
                    if metrics.get(key) is not None
                },
                "proxy_mse": item.get("low_fidelity_proxy_mse"),
                "architecture": self._architecture(item),
                "constraint_enforcement": dict(
                    (item.get("normalized_algorithm_spec") or {}).get(
                        "constraint_enforcement"
                    )
                    or {}
                ),
                "training_success": item.get("training_success"),
                "generation_mode": item.get("generation_mode") or item.get("proposal_strategy"),
                "parent_ids": item.get("parent_spec_ids") or [],
                "candidate_record": (
                    f"runtime_candidates/{candidate_id}/candidate_record.json"
                ),
            }
            )
        return summaries

    def _key(self, record: dict[str, Any]) -> tuple[int, float]:
        value = (
            record.get("low_fidelity_proxy_mse")
            if self.objective == "mse"
            and "low_fidelity_proxy_mse" in record
            else (record.get("metrics") or {}).get(self.objective)
        )
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = float("inf")
        return (0 if record.get("training_success") else 1, numeric)

    @staticmethod
    def _architecture(record: dict[str, Any]) -> str:
        return str(((record.get("normalized_algorithm_spec") or {}).get("network") or {}).get("architecture") or "unknown")
