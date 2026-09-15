"""Strictly isolated in-memory posterior state for one PDE and one run."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from forge.pipeline.posterior_memory.models import ActivePosteriorRule
from forge.pipeline.posterior_memory.prompt_context_builder import (
    build_posterior_prompt_context,
)
from forge.utils.json_safety import json_safe


@dataclass
class RunScopedPosteriorMemory:
    pde_id: str
    run_id: str
    posterior_context_max_tokens: int = 20_000
    candidate_records: list[dict[str, Any]] = field(default_factory=list)
    generation_summaries: list[dict[str, Any]] = field(default_factory=list)
    active_rules: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.pde_id = _required_scope_value(self.pde_id, "pde_id")
        self.run_id = _required_scope_value(self.run_id, "run_id")
        self.posterior_context_max_tokens = max(256, int(self.posterior_context_max_tokens))
        if not self.is_empty():
            raise ValueError("A new run-scoped posterior memory must start empty.")

    def start_run(self, *, pde_id: str | None = None, run_id: str | None = None) -> None:
        self._validate_scope(pde_id or self.pde_id, run_id or self.run_id)
        if not self.is_empty():
            raise RuntimeError("Cannot start a run with non-empty posterior memory.")

    def is_empty(self) -> bool:
        return not self.candidate_records and not self.generation_summaries and not self.active_rules

    def add_candidate_posterior(
        self, *, pde_id: str, run_id: str, posterior: dict[str, Any] | Any
    ) -> None:
        self._validate_scope(pde_id, run_id)
        payload = posterior.to_dict() if hasattr(posterior, "to_dict") else dict(posterior)
        self._validate_payload_scope(payload)
        self.candidate_records.append(deepcopy(json_safe(payload)))

    def add_generation_posterior(
        self, *, pde_id: str, run_id: str, posterior: dict[str, Any] | Any
    ) -> None:
        self._validate_scope(pde_id, run_id)
        payload = posterior.to_dict() if hasattr(posterior, "to_dict") else dict(posterior)
        self._validate_payload_scope(payload)
        self.generation_summaries.append(deepcopy(json_safe(payload)))
        self._update_active_rules(payload)

    def get_latest_generation_posterior(self) -> dict[str, Any] | None:
        return deepcopy(self.generation_summaries[-1]) if self.generation_summaries else None

    def get_active_rules(self) -> list[dict[str, Any]]:
        return deepcopy(self.active_rules)

    def build_prompt_context(self, *, max_tokens: int | None = None) -> dict[str, Any]:
        return build_posterior_prompt_context(
            candidate_records=self.candidate_records,
            generation_summaries=self.generation_summaries,
            active_rules=self.active_rules,
            max_tokens=max_tokens or self.posterior_context_max_tokens,
        )

    def principle_evidence(self) -> list[dict[str, Any]]:
        return [
            {
                "source": "posterior_design_principle",
                "principle_id": rule.get("principle_id"),
                "evidence_status": "supported",
                "confidence": rule.get("confidence"),
                "evidence_count": rule.get("support_count"),
                "guidance_type": "soft",
                "can_be_overridden": True,
                "scope": deepcopy(rule.get("scope") or {}),
            }
            for rule in self.active_rules
            if rule.get("principle_id") and rule.get("rule_type") == "success"
        ]

    def failure_guidance(self) -> list[dict[str, Any]]:
        return [
            {
                "source": "run_scoped_posterior_failure",
                "failure_flag": rule.get("component"),
                "warning": rule.get("rule"),
                "confidence": rule.get("confidence"),
                "support_count": rule.get("support_count"),
                "guidance_type": "soft",
                "can_be_overridden": True,
                "scope": deepcopy(rule.get("scope") or {}),
            }
            for rule in self.active_rules
            if rule.get("rule_type") == "failure"
        ]

    def snapshot_for_archive(
        self, *, output_detail: str = "compact"
    ) -> dict[str, Any]:
        """Archive posterior conclusions without copying every candidate.

        Candidate posteriors are working memory used while a run is active.
        Their specs and measurements already have a canonical home under
        ``runtime_candidates``.  Compact archives therefore retain only the
        generation conclusions, active rules, and a count/reference.  The
        explicit diagnostic mode remains available for forensic runs.
        """

        detail = str(output_detail or "compact").strip().casefold()
        if detail not in {"compact", "diagnostic"}:
            raise ValueError("output_detail must be compact or diagnostic")
        payload: dict[str, Any] = {
            "schema_version": "2.0",
            "memory_type": "RunScopedPosteriorMemory",
            "scope": {"pde_id": self.pde_id, "run_id": self.run_id},
            "candidate_count": len(self.candidate_records),
            "candidate_index": "candidate_records.jsonl",
            "generation_summaries": self.generation_summaries,
            "active_rules": self.active_rules,
        }
        if detail == "diagnostic":
            payload["candidate_records"] = self.candidate_records
        return deepcopy(json_safe(payload))

    def clear(self) -> None:
        self.candidate_records.clear()
        self.generation_summaries.clear()
        self.active_rules.clear()

    def _validate_scope(self, pde_id: str, run_id: str) -> None:
        if str(pde_id) != self.pde_id:
            raise ValueError("Cross-PDE posterior write is forbidden.")
        if str(run_id) != self.run_id:
            raise ValueError("Cross-run posterior write is forbidden.")

    def _validate_payload_scope(self, payload: dict[str, Any]) -> None:
        self._validate_scope(str(payload.get("pde_id")), str(payload.get("run_id")))

    def _update_active_rules(self, generation: dict[str, Any]) -> None:
        generation_index = int(generation.get("generation") or 0)
        for pattern in generation.get("successful_patterns") or []:
            component = pattern.get("component") or pattern.get("principle_id")
            if not component:
                continue
            self._merge_rule(
                rule_id=f"success:{component}",
                rule=f"Prefer retaining {component} when it remains compatible with the current measured bottleneck.",
                rule_type="success",
                support=int(pattern.get("support_count") or 1),
                generation=generation_index,
                component=str(component),
                principle_id=(str(pattern.get("principle_id")) if pattern.get("principle_id") else None),
            )
        for pattern in generation.get("failure_patterns") or []:
            flag = pattern.get("failure_flag")
            if not flag:
                continue
            self._merge_rule(
                rule_id=f"failure:{flag}",
                rule=f"Do not repeat a configuration exhibiting {flag} unchanged; target its measured cause.",
                rule_type="failure",
                support=int(pattern.get("support_count") or 1),
                generation=generation_index,
                component=str(flag),
                principle_id=None,
            )

    def _merge_rule(
        self,
        *,
        rule_id: str,
        rule: str,
        rule_type: str,
        support: int,
        generation: int,
        component: str,
        principle_id: str | None,
    ) -> None:
        existing = next((item for item in self.active_rules if item.get("rule_id") == rule_id), None)
        if existing is None:
            item = ActivePosteriorRule(
                rule_id=rule_id,
                rule=rule,
                rule_type=rule_type,
                support_count=max(1, support),
                contradiction_count=0,
                confidence=_rule_confidence(max(1, support), 0),
                scope={"pde_id": self.pde_id, "run_id": self.run_id},
                evidence_generations=[generation],
                component=component,
                principle_id=principle_id,
            ).to_dict()
            self.active_rules.append(item)
            return
        existing["support_count"] = int(existing.get("support_count") or 0) + max(1, support)
        generations = list(existing.get("evidence_generations") or [])
        if generation not in generations:
            generations.append(generation)
        existing["evidence_generations"] = generations
        existing["confidence"] = _rule_confidence(
            int(existing["support_count"]), int(existing.get("contradiction_count") or 0)
        )


@dataclass
class RunScopedPosteriorSession:
    pde_id: str
    run_id: str
    posterior_context_max_tokens: int = 20_000
    memory: RunScopedPosteriorMemory | None = field(default=None, init=False)

    def __enter__(self) -> RunScopedPosteriorMemory:
        self.memory = RunScopedPosteriorMemory(
            pde_id=self.pde_id,
            run_id=self.run_id,
            posterior_context_max_tokens=self.posterior_context_max_tokens,
        )
        self.memory.start_run()
        return self.memory

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if self.memory is not None:
            self.memory.clear()
        return False


def _rule_confidence(support: int, contradiction: int) -> float:
    return round(min(0.99, max(0.0, (support + 1.0) / (support + contradiction + 2.0))), 6)


def _required_scope_value(value: Any, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{name} is required")
    return text
