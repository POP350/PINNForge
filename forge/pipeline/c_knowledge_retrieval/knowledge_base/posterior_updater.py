"""Posterior knowledge feedback from completed experiments."""

from __future__ import annotations

from collections import Counter
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
from typing import Any


class PosteriorUpdater:
    def __init__(self, posterior_dir: str | Path | None = None) -> None:
        self.posterior_dir = Path(posterior_dir) if posterior_dir else Path(__file__).resolve().parent / "posterior"
        self.posterior_dir.mkdir(parents=True, exist_ok=True)
        self.records_path = self.posterior_dir / "experiment_records.jsonl"
        self.success_path = self.posterior_dir / "success_patterns.json"
        self.failure_path = self.posterior_dir / "failure_patterns.json"
        self.empirical_failure_path = self.posterior_dir / "empirical_failure_patterns.json"
        self.design_principle_path = self.posterior_dir / "design_principle_patterns.json"

    def update_from_experiment(
        self,
        problem_features: dict,
        algorithm_spec: dict,
        validation_report: dict,
        metrics: dict,
        summary: dict,
    ) -> None:
        generation_metadata = algorithm_spec.get("generation_metadata") or {}
        macro_principles = _principle_ids(
            summary.get("macro_physics_principles_applied")
            or generation_metadata.get("macro_physics_principles_applied")
            or []
        )
        architecture_principles = _principle_ids(
            summary.get("architecture_design_principles_applied")
            or generation_metadata.get("architecture_design_principles_applied")
            or []
        )
        record = {
            "schema_version": "1.1",
            "experiment_id": self._next_experiment_id(),
            "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
            "benchmark_id": problem_features.get("benchmark_id"),
            "problem_features": problem_features,
            "algorithm_id": algorithm_spec.get("algorithm_id"),
            "algorithm_name": algorithm_spec.get("name"),
            "algorithm_spec": algorithm_spec,
            "validation_report": validation_report,
            "metrics": metrics,
            "training_success": _experiment_succeeded(metrics, summary),
            "success_factors": summary.get("success_factors", []),
            "failure_factors": summary.get("failure_factors", []),
            "new_rule_candidates": summary.get("new_rule_candidates", []),
            "macro_physics_principles_applied": macro_principles,
            "architecture_design_principles_applied": architecture_principles,
            "principle_tradeoffs": _string_list(
                summary.get("principle_tradeoffs")
                or generation_metadata.get("principle_tradeoffs")
                or []
            ),
            "principle_assessment": summary.get("principle_assessment") or {},
            "evaluator_evidence": summary.get("evaluator_evidence", {}),
            "posterior_record_type": "direct_runner_evaluator_evidence",
        }
        with self.records_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.update_success_patterns()
        self.update_failure_patterns()
        self.update_design_principle_patterns()

    def update_success_patterns(self) -> None:
        records = self._read_records()
        patterns = []
        for benchmark_id, benchmark_records in self._records_by_benchmark(records).items():
            baseline = self._baseline_mean(benchmark_records)
            for key, values in self._component_scores(benchmark_records).items():
                mean_score = sum(values) / len(values)
                if mean_score > baseline:
                    component_type, component = key.split(":", 1)
                    patterns.append({
                        "benchmark_id": benchmark_id,
                        "component_type": component_type,
                        "component": component,
                        "mean_score": mean_score,
                        "baseline_mean": baseline,
                        "delta": mean_score - baseline,
                        "count": len(values),
                    })
        patterns.sort(key=lambda item: (item["delta"], item["count"]), reverse=True)
        self._write_json(self.success_path, patterns)

    def update_failure_patterns(self) -> None:
        records = self._read_records()
        patterns = []
        for benchmark_id, benchmark_records in self._records_by_benchmark(records).items():
            baseline = self._baseline_mean(benchmark_records)
            component_scores = self._component_scores(benchmark_records, include_failed=True)
            for key, values in component_scores.items():
                numeric_values = [value for value in values if value is not None]
                mean_score = sum(numeric_values) / len(numeric_values) if numeric_values else 0.0
                nan_count = sum(1 for value in values if value is None)
                if mean_score < baseline or nan_count:
                    component_type, component = key.split(":", 1)
                    patterns.append({
                        "benchmark_id": benchmark_id,
                        "component_type": component_type,
                        "component": component,
                        "mean_score": mean_score,
                        "baseline_mean": baseline,
                        "delta": mean_score - baseline,
                        "count": len(values),
                        "nan_or_failed_count": nan_count,
                    })
        patterns.sort(key=lambda item: (item["delta"], -item["nan_or_failed_count"]))
        self._write_json(self.failure_path, patterns)

    def update_design_principle_patterns(self) -> None:
        """Aggregate measured applications of explicit design principles."""
        grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
        for record in self._read_records():
            benchmark_id = str(record.get("benchmark_id") or "unknown")
            succeeded = bool(record.get("training_success"))
            for principle_type, field_name in (
                ("macro_physics", "macro_physics_principles_applied"),
                ("architecture_design", "architecture_design_principles_applied"),
            ):
                for principle_id in _principle_ids(record.get(field_name) or []):
                    key = (benchmark_id, principle_type, principle_id)
                    pattern = grouped.setdefault(
                        key,
                        {
                            "benchmark_id": benchmark_id,
                            "principle_type": principle_type,
                            "principle_id": principle_id,
                            "application_count": 0,
                            "successful_training_count": 0,
                            "failed_training_count": 0,
                            "evidence_experiment_ids": [],
                            "recent_tradeoffs": [],
                        },
                    )
                    pattern["application_count"] += 1
                    outcome_key = "successful_training_count" if succeeded else "failed_training_count"
                    pattern[outcome_key] += 1
                    if record.get("experiment_id"):
                        pattern["evidence_experiment_ids"].append(record["experiment_id"])
                    for tradeoff in record.get("principle_tradeoffs") or []:
                        text = str(tradeoff).strip()
                        if text and text not in pattern["recent_tradeoffs"]:
                            pattern["recent_tradeoffs"].append(text)
        patterns = []
        for pattern in grouped.values():
            total = int(pattern["application_count"])
            pattern["training_success_rate"] = (
                float(pattern["successful_training_count"]) / total if total else 0.0
            )
            pattern["evidence_experiment_ids"] = pattern["evidence_experiment_ids"][-20:]
            pattern["recent_tradeoffs"] = pattern["recent_tradeoffs"][-8:]
            patterns.append(pattern)
        patterns.sort(
            key=lambda item: (item["application_count"], item["training_success_rate"]),
            reverse=True,
        )
        self._write_json(self.design_principle_path, patterns)

    def retrieve_relevant_posterior(self, problem_features: dict) -> dict:
        benchmark_id = problem_features.get("benchmark_id")
        records = [
            record
            for record in self._read_records()
            if record.get("benchmark_id") == benchmark_id
        ]
        empirical_failures = [
            item for item in self._read_json(self.empirical_failure_path)
            if item.get("benchmark_id") in {None, benchmark_id}
        ]
        generated_failures = [
            item for item in self._read_json(self.failure_path)
            if item.get("benchmark_id") in {None, benchmark_id}
        ]
        return {
            "recent_records": records[-10:],
            "success_patterns": [
                item for item in self._read_json(self.success_path)
                if item.get("benchmark_id") in {None, benchmark_id}
            ],
            "failure_patterns": [*empirical_failures, *generated_failures],
            "design_principle_patterns": [
                item
                for item in self._read_json(self.design_principle_path)
                if item.get("benchmark_id") in {None, benchmark_id, "unknown"}
            ],
        }

    def _next_experiment_id(self) -> str:
        return f"exp_{len(self._read_records()) + 1:06d}"

    def _read_records(self) -> list[dict[str, Any]]:
        if not self.records_path.exists():
            return []
        records = []
        for line in self.records_path.read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                records.append(json.loads(line))
        return records

    @staticmethod
    def _records_by_benchmark(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            grouped[str(record.get("benchmark_id", "unknown"))].append(record)
        return grouped

    def _baseline_mean(self, records: list[dict[str, Any]]) -> float:
        baseline_scores = [
            float(record.get("metrics", {}).get("score"))
            for record in records
            if record.get("algorithm_name") == "PINN_MLP_Tanh"
            and self._is_pure_baseline(record.get("algorithm_spec", {}))
            and isinstance(record.get("metrics", {}).get("score"), int | float)
        ]
        if not baseline_scores:
            baseline_scores = [
                float(record.get("metrics", {}).get("score"))
                for record in records
                if isinstance(record.get("metrics", {}).get("score"), int | float)
            ]
        return sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0.0

    @staticmethod
    def _is_pure_baseline(spec: dict[str, Any]) -> bool:
        loss_names = {term.get("name") for term in spec.get("loss", {}).get("terms", [])}
        return (
            spec.get("network", {}).get("type") == "mlp"
            and spec.get("sampling", {}).get("method") != "rar"
            and "gradient_residual" not in loss_names
        )

    def _component_scores(self, records: list[dict[str, Any]], include_failed: bool = False) -> dict[str, list[float | None]]:
        scores: dict[str, list[float | None]] = defaultdict(list)
        for record in records:
            spec = record.get("algorithm_spec", {})
            metrics = record.get("metrics", {})
            score = metrics.get("score")
            nan_detected = bool(metrics.get("nan_detected"))
            valid_score = float(score) if isinstance(score, int | float) and not nan_detected else None
            if valid_score is None and not include_failed:
                continue
            for key in self._component_keys(spec):
                scores[key].append(valid_score)
        return scores

    @staticmethod
    def _component_keys(spec: dict[str, Any]) -> list[str]:
        keys = []
        network = spec.get("network", {})
        sampling = spec.get("sampling", {})
        optimizer = spec.get("optimizer", {})
        if network.get("type"):
            keys.append(f"network:{network['type']}")
        for term in spec.get("loss", {}).get("terms", []):
            if term.get("name"):
                keys.append(f"loss:{term['name']}")
        if sampling.get("method"):
            keys.append(f"sampling:{sampling['method']}")
        if optimizer.get("strategy"):
            keys.append(f"optimizer:{optimizer['strategy']}")
        for constraint in spec.get("physics_constraints", {}).get("constraints", []):
            keys.append(f"constraint:{constraint}")
        if spec.get("name"):
            keys.append(f"algorithm:{spec['name']}")
        return keys

    @staticmethod
    def _read_json(path: Path):
        if not path.exists() or not path.read_text(encoding="utf-8-sig").strip():
            return []
        return json.loads(path.read_text(encoding="utf-8-sig"))

    @staticmethod
    def _write_json(path: Path, payload) -> None:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _principle_ids(values: Any) -> list[str]:
    result: list[str] = []
    items = values if isinstance(values, list) else [values]
    for item in items:
        if isinstance(item, dict):
            value = item.get("principle_id") or item.get("id")
        else:
            value = item
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def _string_list(values: Any) -> list[str]:
    result: list[str] = []
    items = values if isinstance(values, list) else [values]
    for item in items:
        text = str(item or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def _experiment_succeeded(metrics: dict, summary: dict) -> bool:
    if "training_success" in summary:
        return bool(summary.get("training_success"))
    if metrics.get("nan_detected"):
        return False
    for key in ("l2re", "relative_l2", "mse", "score"):
        value = metrics.get(key)
        if isinstance(value, int | float):
            return value == value and value not in {float("inf"), -float("inf")}
    return not bool(summary.get("failure_factors"))
