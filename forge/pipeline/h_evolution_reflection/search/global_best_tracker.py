"""Bounded persistent Top-3 leaderboard of normalized AlgorithmSpecs."""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Iterable

from forge.pipeline.d_algorithm_generation.initialization_diversity import (
    executable_algorithm_spec,
)
from forge.utils.json_safety import json_safe


class GlobalBestTracker:
    """Maintain only the median-aggregated global distinct-Spec Top-K."""

    FORMAT_VERSION = "3.0"

    def __init__(
        self,
        *,
        ranking_metric: str = "low_fidelity_proxy_mse",
        duplicate_aggregation: str = "median",
        default_top_k: int = 3,
    ) -> None:
        if ranking_metric != "low_fidelity_proxy_mse":
            raise ValueError(
                "GlobalBestTracker requires low_fidelity_proxy_mse ranking"
            )
        if duplicate_aggregation != "median":
            raise ValueError("GlobalBestTracker only supports median aggregation")
        self.ranking_metric = ranking_metric
        self.duplicate_aggregation = duplicate_aggregation
        self.default_top_k = max(1, int(default_top_k))
        self._groups: list[dict[str, Any]] = []
        self._excluded_record_count = 0

    def update(self, record: dict[str, Any]) -> bool:
        """Update one record and immediately restore the bounded Top-K invariant."""

        accepted = self._update_record(record)
        self._prune_to_top_k()
        return accepted

    def _update_record(self, record: dict[str, Any]) -> bool:
        candidate_id = _candidate_id(record)
        self._remove_candidate(candidate_id)
        value = _proxy_value(record)
        spec = record.get("search_algorithm_spec") or record.get(
            "normalized_algorithm_spec"
        )
        validation = record.get("validation_report") or {}
        if (
            not record.get("training_success")
            or value is None
            or not isinstance(spec, dict)
            or validation.get("valid") is False
        ):
            self._excluded_record_count += 1
            return False

        semantic_spec = executable_algorithm_spec(spec)
        group = next(
            (
                item
                for item in self._groups
                if item.get("normalized_executable_spec") == semantic_spec
            ),
            None,
        )
        if group is None:
            group = {
                "normalized_executable_spec": deepcopy(semantic_spec),
                "member_records": [],
            }
            self._groups.append(group)
        group["member_records"].append(_tracker_record_view(record, value))
        self._refresh_group(group)
        return True

    def update_many(self, records: Iterable[dict[str, Any]]) -> dict[str, int]:
        accepted = 0
        rejected = 0
        for record in records:
            if self._update_record(record):
                accepted += 1
            else:
                rejected += 1
        pruned = self._prune_to_top_k()
        return {
            "accepted": accepted,
            "rejected": rejected,
            "pruned_distinct_specs": pruned,
            "retained_distinct_specs": len(self._groups),
        }

    def top_k(
        self, count: int | None = None
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        requested = self.default_top_k if count is None else max(0, int(count))
        ranked = self._ranked_groups()
        selected: list[dict[str, Any]] = []
        for rank, group in enumerate(ranked[:requested], start=1):
            item = deepcopy(group["representative_record"])
            item["low_fidelity_rank"] = rank
            item["low_fidelity_duplicate_aggregation"] = (
                self.duplicate_aggregation
            )
            item["low_fidelity_aggregate_proxy_mse"] = group[
                "aggregate_low_fidelity_proxy_mse"
            ]
            item["low_fidelity_duplicate_count"] = group["duplicate_count"]
            item["low_fidelity_duplicate_member_ids"] = list(
                group["member_candidate_ids"]
            )
            selected.append(item)
        audit = {
            "selection_source": "global_best_tracker",
            "tracker_format_version": self.FORMAT_VERSION,
            "requested_candidate_count": requested,
            "selected_candidate_count": len(selected),
            "distinct_normalized_spec_count": len(ranked),
            "retention_limit": self.default_top_k,
            "bounded_top_k_only": True,
            "deduplicate_normalized_specs": True,
            "duplicate_aggregation": self.duplicate_aggregation,
            "ranking_metric": self.ranking_metric,
            "selected_candidate_ids": [
                item.get("spec_id") for item in selected
            ],
            "excluded_record_count": self._excluded_record_count,
            "insufficient_distinct_candidates": len(selected) < requested,
        }
        return selected, audit

    def leaderboard(self) -> list[dict[str, Any]]:
        return [
            {
                "historical_rank": rank,
                "representative_candidate_id": group[
                    "representative_candidate_id"
                ],
                "aggregate_low_fidelity_proxy_mse": group[
                    "aggregate_low_fidelity_proxy_mse"
                ],
                "duplicate_count": group["duplicate_count"],
                "member_candidate_ids": list(group["member_candidate_ids"]),
                "source_generation_ids": list(group["source_generation_ids"]),
                "representative_source_generation": group[
                    "representative_record"
                ].get("generation_index"),
                "normalized_spec_key": normalized_spec_key(
                    group["normalized_executable_spec"]
                ),
                "normalized_executable_spec": deepcopy(
                    group["normalized_executable_spec"]
                ),
            }
            for rank, group in enumerate(self._ranked_groups(), start=1)
        ]

    def snapshot(self) -> dict[str, Any]:
        top3, _ = self.top_k(self.default_top_k)
        return json_safe(
            {
                "format_version": self.FORMAT_VERSION,
                "ranking_metric": self.ranking_metric,
                "duplicate_aggregation": self.duplicate_aggregation,
                "default_top_k": self.default_top_k,
                "distinct_spec_groups": deepcopy(self._groups),
                "excluded_record_count": self._excluded_record_count,
                "retention_policy": "bounded_global_distinct_top_k_only",
                "leaderboard": self.leaderboard(),
                "global_distinct_top3": top3,
            }
        )

    def audit_snapshot(self) -> dict[str, Any]:
        top3, _ = self.top_k(self.default_top_k)
        leaderboard = self.leaderboard()
        return {
            "format_version": self.FORMAT_VERSION,
            "distinct_normalized_spec_count": len(self._groups),
            "retention_limit": self.default_top_k,
            "bounded_top_k_only": True,
            "excluded_record_count": self._excluded_record_count,
            "global_distinct_top3": [
                {
                    "low_fidelity_rank": item["low_fidelity_rank"],
                    "representative_candidate_id": item.get("spec_id"),
                    "source_generation": item.get("generation_index"),
                    "normalized_spec_key": normalized_spec_key(
                        item.get("search_algorithm_spec")
                        or item.get("normalized_algorithm_spec")
                        or {}
                    ),
                    "aggregate_low_fidelity_proxy_mse": item[
                        "low_fidelity_aggregate_proxy_mse"
                    ],
                    "duplicate_count": item[
                        "low_fidelity_duplicate_count"
                    ],
                }
                for item in top3
            ],
            "global_top3_candidate_ids": [
                item["representative_candidate_id"] for item in leaderboard
            ],
            "global_top3_spec_keys": [
                item["normalized_spec_key"] for item in leaderboard
            ],
            "global_top3_source_generations": [
                item["representative_source_generation"]
                for item in leaderboard
            ],
            "global_top3_median_mse": (
                float(
                    median(
                        item["aggregate_low_fidelity_proxy_mse"]
                        for item in leaderboard
                    )
                )
                if leaderboard
                else None
            ),
        }

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        # leaderboard and global_distinct_top3 are deterministic views of the
        # retained groups.  Persist the groups once and rebuild those views on
        # load instead of writing three copies of the same Top-K records.
        payload = {
            "format_version": self.FORMAT_VERSION,
            "ranking_metric": self.ranking_metric,
            "duplicate_aggregation": self.duplicate_aggregation,
            "default_top_k": self.default_top_k,
            "distinct_spec_groups": deepcopy(self._groups),
            "excluded_record_count": self._excluded_record_count,
            "retention_policy": "bounded_global_distinct_top_k_only",
        }
        target.write_text(
            json.dumps(json_safe(payload), separators=(",", ":"), ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "GlobalBestTracker":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(
                "GlobalBestTracker file must contain a JSON object"
            )
        if str(payload.get("format_version")) != cls.FORMAT_VERSION:
            raise ValueError("Unsupported GlobalBestTracker format version")
        tracker = cls(
            ranking_metric=str(
                payload.get("ranking_metric") or "low_fidelity_proxy_mse"
            ),
            duplicate_aggregation=str(
                payload.get("duplicate_aggregation") or "median"
            ),
            default_top_k=int(payload.get("default_top_k") or 3),
        )
        groups = payload.get("distinct_spec_groups") or []
        if not isinstance(groups, list):
            raise ValueError("distinct_spec_groups must be an array")
        tracker._groups = deepcopy(groups)
        tracker._excluded_record_count = int(
            payload.get("excluded_record_count") or 0
        )
        for group in tracker._groups:
            tracker._refresh_group(group)
        tracker._prune_to_top_k()
        return tracker

    @classmethod
    def from_records(
        cls,
        records: Iterable[dict[str, Any]],
        *,
        default_top_k: int = 3,
    ) -> "GlobalBestTracker":
        tracker = cls(default_top_k=default_top_k)
        tracker.update_many(records)
        return tracker

    def _ranked_groups(self) -> list[dict[str, Any]]:
        return sorted(
            self._groups,
            key=lambda item: (
                float(item["aggregate_low_fidelity_proxy_mse"]),
                str(item["representative_candidate_id"]),
            ),
        )

    def _prune_to_top_k(self) -> int:
        ranked = self._ranked_groups()
        pruned = max(0, len(ranked) - self.default_top_k)
        self._groups = ranked[: self.default_top_k]
        return pruned

    def _refresh_group(self, group: dict[str, Any]) -> None:
        members = list(group.get("member_records") or [])
        if not members:
            raise ValueError("GlobalBestTracker group cannot be empty")
        values = [float(item[self.ranking_metric]) for item in members]
        aggregate = float(median(values))
        representative = min(
            members,
            key=lambda item: (
                abs(float(item[self.ranking_metric]) - aggregate),
                str(item.get("spec_id") or ""),
            ),
        )
        group["member_records"] = members
        group["proxy_mse_values"] = values
        group["aggregate_low_fidelity_proxy_mse"] = aggregate
        group["duplicate_count"] = len(members)
        group["member_candidate_ids"] = [
            item.get("spec_id") for item in members
        ]
        group["source_generation_ids"] = [
            item.get("generation_index") for item in members
        ]
        group["representative_candidate_id"] = representative.get("spec_id")
        group["representative_record"] = deepcopy(representative)

    def _remove_candidate(self, candidate_id: str) -> None:
        remaining_groups: list[dict[str, Any]] = []
        for group in self._groups:
            members = [
                item
                for item in (group.get("member_records") or [])
                if str(item.get("spec_id")) != candidate_id
            ]
            if not members:
                continue
            group["member_records"] = members
            self._refresh_group(group)
            remaining_groups.append(group)
        self._groups = remaining_groups


def _candidate_id(record: dict[str, Any]) -> str:
    candidate_id = str(
        record.get("spec_id") or record.get("candidate_id") or ""
    )
    if not candidate_id:
        raise ValueError("GlobalBestTracker record requires a candidate ID")
    return candidate_id


def _proxy_value(record: dict[str, Any]) -> float | None:
    raw = record.get("low_fidelity_proxy_mse")
    if raw is None:
        raw = (record.get("metrics") or {}).get("mse")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _tracker_record_view(
    record: dict[str, Any], proxy_value: float
) -> dict[str, Any]:
    return json_safe(
        {
            "problem_id": record.get("problem_id"),
            "spec_id": _candidate_id(record),
            "generation": record.get("generation"),
            "generation_index": record.get(
                "generation_index", record.get("generation")
            ),
            "generation_rank": record.get("generation_rank"),
            "candidate_role": record.get("candidate_role"),
            "generation_mode": record.get("generation_mode"),
            "training_success": True,
            "validation_report": {
                "valid": bool(
                    (record.get("validation_report") or {}).get("valid", True)
                )
            },
            "search_algorithm_spec": deepcopy(
                record.get("search_algorithm_spec")
                or record.get("normalized_algorithm_spec")
                or {}
            ),
            "normalized_algorithm_spec": deepcopy(
                record.get("normalized_algorithm_spec") or {}
            ),
            "low_fidelity_proxy_mse": proxy_value,
            # Only the ranking scalar is needed after tracker admission.  Full
            # metrics remain in runtime_candidates/<id>/final_metrics.json.
            "metrics": {"mse": proxy_value},
            "training_seed": record.get("training_seed"),
            "evaluation_seed": record.get("evaluation_seed"),
        }
    )


def normalized_spec_key(spec: dict[str, Any]) -> str:
    """Return the canonical key used for distinct executable Spec identity."""

    return json.dumps(
        json_safe(executable_algorithm_spec(spec)),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


__all__ = ["GlobalBestTracker", "normalized_spec_key"]
