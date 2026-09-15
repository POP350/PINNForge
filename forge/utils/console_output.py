"""Bounded console output for long-running formal searches."""

from __future__ import annotations

import json
from typing import Any, Mapping

from forge.utils.json_safety import json_safe


CONSOLE_LOG_LEVELS = ("quiet", "normal", "verbose")
_LEVEL_ORDER = {name: index for index, name in enumerate(CONSOLE_LOG_LEVELS)}


def normalize_console_log_level(value: Any) -> str:
    level = str(value or "normal").strip().lower()
    if level not in _LEVEL_ORDER:
        raise ValueError(
            "console_log_level must be one of: "
            + ", ".join(CONSOLE_LOG_LEVELS)
        )
    return level


def console_enabled(config: Mapping[str, Any], minimum: str = "normal") -> bool:
    current = normalize_console_log_level(config.get("console_log_level"))
    required = normalize_console_log_level(minimum)
    return _LEVEL_ORDER[current] >= _LEVEL_ORDER[required]


def compact_run_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    """Return a stable summary without embedding candidate or registry payloads."""

    def mapping(value: Any) -> Mapping[str, Any]:
        return value if isinstance(value, Mapping) else {}

    config = mapping(result.get("config"))
    archive = mapping(result.get("experiment_archive"))
    final_rank_1 = mapping(result.get("final_rank_1"))
    metrics = mapping(final_rank_1.get("metrics"))
    summary = {
        "status": result.get("status"),
        "search_status": result.get("search_status"),
        "problem_id": result.get("problem_id") or config.get("benchmark"),
        "experiment_contract_version": result.get("experiment_contract_version"),
        "actual_generations_completed": result.get("actual_generations_completed"),
        "termination_reason": result.get("termination_reason"),
        "final_rank_1_spec_id": (
            final_rank_1.get("spec_id") or final_rank_1.get("candidate_id")
        ),
        "final_rank_1_mse": metrics.get("mse"),
        "final_rank_1_relative_l2_error": (
            metrics.get("l2re")
            if metrics.get("l2re") is not None
            else metrics.get("relative_l2_error")
        ),
        "output_dir": config.get("output_dir"),
        "run_result_path": archive.get("run_result_path"),
    }
    return {key: value for key, value in summary.items() if value is not None}


def emit_run_result(result: Mapping[str, Any], *, level: str = "normal") -> None:
    resolved = normalize_console_log_level(level)
    if resolved == "quiet":
        return
    payload: Any = result if resolved == "verbose" else compact_run_summary(result)
    print(
        json.dumps(
            json_safe(payload),
            ensure_ascii=False,
            indent=2 if resolved == "verbose" else None,
            separators=None if resolved == "verbose" else (",", ":"),
            default=str,
        ),
        flush=True,
    )


__all__ = [
    "CONSOLE_LOG_LEVELS",
    "compact_run_summary",
    "console_enabled",
    "emit_run_result",
    "normalize_console_log_level",
]
