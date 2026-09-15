"""Unified CLI entry point for all registered PDE benchmarks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from forge.experiments.closed_loop import run_closed_loop
from forge.pipeline.a_problem_definition.problems.registry import (
    configure_problem,
    get_problem,
    list_problems,
)
from forge.pipeline.a_problem_definition.resource_defaults import (
    resolve_resource_defaults,
)
from forge.pipeline.b_feature_extraction.feature_extractor import extract_problem_features
from forge.pipeline.h_evolution_reflection.search.candidate_artifacts import (
    artifact_reference,
    compact_candidate_record,
    compact_final_metrics,
    compact_training_history,
    compact_training_report,
    history_with_required_events,
    write_json as write_candidate_json,
    write_jsonl as write_candidate_jsonl,
)
from forge.utils.json_safety import json_safe
from forge.utils.console_output import (
    CONSOLE_LOG_LEVELS,
    emit_run_result,
    normalize_console_log_level,
)
from forge.ablation import (
    ABLATION_MODES,
    STANDARD_ABLATION_MODE,
    normalize_ablation_mode,
    resolved_ablation_contract,
)


REQUIRED_ARTIFACTS = (
    "experiment_contract.json",
    "problem_definition.json",
    "pde_features.json",
    "population.json",
    "candidate_record.json",
    "algorithm_spec_raw.json",
    "algorithm_spec_normalized.json",
    "validation_report.json",
    "training_history.jsonl",
    "final_metrics.json",
    "resource_usage.json",
)
_UNSET = object()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run any registered problem through the common Open AlgorithmSpec search."
    )
    parser.add_argument("--problem", required=True, choices=list_problems())
    parser.add_argument(
        "--experiment-config",
        type=Path,
        help="Optional experiment JSON.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--independent-run-id", default=None)
    parser.add_argument(
        "--problem-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Repeatable problem-specific override. VALUE accepts JSON scalars, arrays, "
            "and objects; dotted keys create nested mappings."
        ),
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--llm-provider", default=None)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-api-base", default=None)
    parser.add_argument(
        "--ablation",
        "--ablation-mode",
        dest="ablation_mode",
        type=normalize_ablation_mode,
        choices=ABLATION_MODES,
        default=None,
        help=(
            "Select the paper experiment: full, no_knowledge, "
            "no_execution_feedback, or no_evolutionary_search. "
            "--ablation-mode is retained as a compatibility alias."
        ),
    )
    parser.add_argument(
        "--execution-feedback",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Deprecated compatibility switch. Prefer "
            "--ablation no_execution_feedback."
        ),
    )
    parser.add_argument("--llm-max-tokens", type=_parse_positive_integer, default=None)
    parser.add_argument(
        "--llm-max-concurrency",
        type=int,
        choices=(1, 2, 3),
        default=None,
        help="Maximum concurrent remote LLM design requests; GPU training remains serial.",
    )
    parser.add_argument("--llm-timeout-seconds", type=_parse_positive_integer, default=None)
    parser.add_argument(
        "--posterior-context-max-tokens", type=_parse_positive_integer, default=None
    )
    parser.add_argument(
        "--maximum-sampling-points",
        type=_parse_optional_positive_integer,
        default=_UNSET,
        help="Positive limit or unlimited/none for no limit.",
    )
    parser.add_argument(
        "--maximum-model-parameters", type=_parse_positive_integer, default=None
    )
    parser.add_argument(
        "--evaluation-size",
        type=_parse_evaluation_size,
        default=_UNSET,
        help="Positive evaluation cap or full/all/none for the full reference grid.",
    )
    parser.add_argument(
        "--residual-evaluation-size", type=_parse_positive_integer, default=None
    )
    parser.add_argument(
        "--constraint-resample-interval", type=_parse_nonnegative_integer, default=None
    )
    parser.add_argument(
        "--console-log-level",
        choices=CONSOLE_LOG_LEVELS,
        default=None,
        help=(
            "quiet suppresses console status, normal emits bounded summaries, "
            "and verbose prints the complete result JSON."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a compatible multi-generation search from output-dir.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Load an optional JSON and apply explicit CLI overrides."""

    config = (
        load_experiment_config(args.experiment_config)
        if args.experiment_config is not None
        else {}
    )
    config = dict(config)
    if "resource_profile" in config:
        raise ValueError(
            "resource_profile has been removed; configure concrete values "
            "under search instead"
        )
    configured_problem = config.get("problem_id")
    if configured_problem is not None and str(configured_problem) != args.problem:
        raise ValueError(
            f"config problem_id {configured_problem!r} does not match --problem {args.problem!r}"
        )
    config["problem_id"] = args.problem

    if args.seed is not None:
        config["seed"] = args.seed
    if args.independent_run_id is not None:
        config["independent_run_id"] = args.independent_run_id

    problem_config = dict(config.get("problem_config") or {})
    for expression in args.problem_option:
        key, value = _parse_problem_option(expression)
        _set_nested_option(problem_config, key, value)
    config["problem_config"] = problem_config

    search = dict(config.get("search") or {})
    for key in (
        "device",
        "llm_provider",
        "llm_model",
        "llm_api_base",
        "ablation_mode",
        "llm_max_tokens",
        "llm_max_concurrency",
        "llm_timeout_seconds",
        "posterior_context_max_tokens",
        "maximum_sampling_points",
        "maximum_model_parameters",
        "evaluation_size",
        "residual_evaluation_size",
        "constraint_resample_interval",
        "console_log_level",
    ):
        value = getattr(args, key)
        if value is _UNSET:
            continue
        if value is not None or key in {"maximum_sampling_points", "evaluation_size"}:
            search[key] = value
    if args.execution_feedback is not None:
        ablation = dict(search.get("ablation") or {})
        ablation["execution_feedback"] = args.execution_feedback
        search["ablation"] = ablation
    if args.resume:
        search["resume"] = True
    config["search"] = search
    return config


def _parse_problem_option(expression: str) -> tuple[str, Any]:
    key, separator, raw_value = str(expression).partition("=")
    key = key.strip()
    if not separator or not key or any(not part for part in key.split(".")):
        raise ValueError("--problem-option must use a non-empty KEY=VALUE expression")
    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError:
        value = raw_value
    return key, value


def _set_nested_option(target: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cursor = target
    for part in parts[:-1]:
        existing = cursor.get(part)
        if existing is None:
            nested: dict[str, Any] = {}
            cursor[part] = nested
            cursor = nested
        elif isinstance(existing, dict):
            cursor = existing
        else:
            raise ValueError(
                f"cannot set nested problem option {dotted_key!r}: {part!r} is not an object"
            )
    cursor[parts[-1]] = value


def _parse_positive_integer(value: str) -> int:
    try:
        resolved = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a positive integer") from exc
    if resolved < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return resolved


def _parse_nonnegative_integer(value: str) -> int:
    try:
        resolved = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a non-negative integer") from exc
    if resolved < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return resolved


def _parse_optional_positive_integer(value: str) -> int | None:
    normalized = str(value).strip().lower()
    if normalized in {"none", "unlimited", "no-limit", "nolimit"}:
        return None
    return _parse_positive_integer(normalized)


def _parse_evaluation_size(value: str) -> int | None:
    normalized = str(value).strip().lower()
    if normalized in {"none", "full", "all"}:
        return None
    return _parse_positive_integer(normalized)


def load_experiment_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"experiment config not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("experiment config must contain a JSON object")
    return payload


def prepare_experiment(
    problem_id: str,
    config: dict[str, Any],
    *,
    output_dir: Path | None = None,
    write_artifacts: bool = True,
) -> tuple[Any, Path, dict[str, Any]]:
    configured_id = config.get("problem_id")
    if configured_id is not None and str(configured_id) != problem_id:
        raise ValueError(f"config problem_id {configured_id!r} does not match --problem {problem_id!r}")
    if "resource_profile" in config:
        raise ValueError(
            "resource_profile has been removed; configure concrete values "
            "under search instead"
        )
    configure_problem(problem_id, dict(config.get("problem_config") or {}))
    problem = get_problem(problem_id)
    resources = resolve_resource_defaults()
    search = dict(config.get("search") or {})
    ablation = resolved_ablation_contract(search)
    ablation_mode = str(ablation["mode"])
    default_target = Path("outputs") / problem_id
    if ablation_mode != STANDARD_ABLATION_MODE:
        default_target = (
            default_target
            / "ablations"
            / ablation_mode
            / f"seed_{int(config.get('seed', 0))}"
        )
    target = Path(output_dir or config.get("output_dir") or default_target)
    if write_artifacts:
        target.mkdir(parents=True, exist_ok=True)
        _write_problem_artifacts(target, problem, resources)
    return problem, target, resources


def _write_problem_artifacts(target: Path, problem: Any, resources: dict[str, Any]) -> None:
    spec = problem.get_spec()
    features = extract_problem_features(problem)
    contract = _external_problem_contract(problem)
    documents: dict[str, Any] = {
        "experiment_contract.json": contract,
        "problem_definition.json": spec.model_dump(mode="json"),
        "pde_features.json": features,
        "population.json": {"status": "not_started", "candidates": []},
        "candidate_record.json": {"status": "not_started", "problem_id": spec.problem_id},
        "algorithm_spec_raw.json": {},
        "algorithm_spec_normalized.json": {},
        "validation_report.json": {"valid": True, "stage": "problem_registration"},
        "final_metrics.json": _empty_final_metrics(problem),
        "resource_usage.json": {"status": "not_started", **resources},
    }
    for name, payload in documents.items():
        (target / name).write_text(
            json.dumps(json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8"
        )
    (target / "training_history.jsonl").write_text("", encoding="utf-8")


def _external_problem_contract(problem: Any) -> dict[str, Any]:
    spec = problem.get_spec()
    return {
        "contract_version": "external-pde-1.0",
        "problem_id": spec.problem_id,
        "problem_version": spec.metadata.get("problem_version"),
        "case_study": bool(spec.metadata.get("case_study", False)),
        "benchmark_protocol": spec.metadata.get("benchmark_protocol", "external-pde-coordinate-pinn"),
        "not_official_operator_learning_protocol": bool(
            spec.metadata.get("not_official_operator_learning_protocol", False)
        ),
    }


def _empty_final_metrics(problem: Any) -> dict[str, Any]:
    return {
        "primary_metric": {"name": "relative_l2", "value": None},
        "solution_metrics": {},
        "physics_metrics": {},
        "constraint_metrics": {},
        "conservation_metrics": {},
        "resource_metrics": {},
        "case_study": bool(problem.get_spec().metadata.get("case_study", False)),
    }


def _finalize_formal_artifacts(
    target: Path,
    problem: Any,
    resources: dict[str, Any],
    result: dict[str, Any],
) -> None:
    """Materialize root-level external-PDE artifacts after search succeeds.

    The generic search controller owns directory admission and its experiment
    contract.  This wrapper only augments completed output; it never places
    files in the directory before the controller's safety check.
    """

    target.mkdir(parents=True, exist_ok=True)
    spec = problem.get_spec()
    contract_path = target / "experiment_contract.json"
    contract: dict[str, Any] = {}
    if contract_path.is_file():
        loaded = json.loads(contract_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            contract = loaded
    contract["external_problem"] = _external_problem_contract(problem)
    _write_json_document(contract_path, contract)
    _write_json_document(target / "problem_definition.json", spec.model_dump(mode="json"))
    _write_json_document(target / "pde_features.json", extract_problem_features(problem))

    population_path = target / "population.json"
    if not population_path.exists():
        _write_json_document(population_path, result.get("population") or {})

    best = result.get("best_candidate")
    if isinstance(best, dict):
        training_report = dict(best.get("training_report") or {})
        history = compact_training_history(history_with_required_events(training_report))
        metrics = compact_final_metrics(best.get("metrics") or {}, training_report)
        write_candidate_json(target / "algorithm_spec_raw.json", best.get("algorithm_spec_raw") or {})
        write_candidate_json(
            target / "algorithm_spec_normalized.json",
            best.get("normalized_algorithm_spec") or best.get("search_algorithm_spec") or {},
        )
        write_candidate_json(target / "validation_report.json", best.get("validation_report") or {})
        write_candidate_jsonl(target / "training_history.jsonl", history)
        write_candidate_json(target / "final_metrics.json", metrics)
        artifacts = {
            "raw_spec": artifact_reference(target / "algorithm_spec_raw.json"),
            "normalized_spec": artifact_reference(target / "algorithm_spec_normalized.json"),
            "validation_report": artifact_reference(target / "validation_report.json"),
            "training_history": artifact_reference(target / "training_history.jsonl"),
            "final_metrics": artifact_reference(target / "final_metrics.json"),
        }
        candidate_record = compact_candidate_record(
            best,
            metrics=metrics,
            training_report=compact_training_report(training_report, history),
            artifacts=artifacts,
        )
        write_candidate_json(target / "candidate_record.json", candidate_record)
    else:
        _write_json_document(
            target / "candidate_record.json",
            {"status": "no_successful_candidate", "problem_id": spec.problem_id},
        )
        for name in ("algorithm_spec_raw.json", "algorithm_spec_normalized.json"):
            _write_json_document(target / name, {})
        _write_json_document(
            target / "validation_report.json",
            {"valid": False, "stage": "completed_search", "reason": "no_successful_candidate"},
        )
        (target / "training_history.jsonl").write_text("", encoding="utf-8")
        _write_json_document(target / "final_metrics.json", _empty_final_metrics(problem))

    search_audit = dict(result.get("search_audit") or {})
    _write_json_document(
        target / "resource_usage.json",
        {
            "status": "completed",
            **resources,
            "actual_candidate_trainings": search_audit.get("actual_candidate_trainings"),
            "total_candidate_training_seconds": search_audit.get("total_candidate_training_seconds"),
            "total_gpu_time_seconds": search_audit.get("total_gpu_time_seconds"),
        },
    )


def _write_json_document(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def runtime_config(problem_id: str, config: dict[str, Any], target: Path, resources: dict[str, Any]) -> dict[str, Any]:
    search = dict(config.get("search") or {})
    ablation = resolved_ablation_contract(search)
    provider = search.get("llm_provider") or os.environ.get("LLM_PROVIDER")
    model = search.get("llm_model") or os.environ.get("LLM_MODEL")
    if not provider or not model:
        raise ValueError("formal search requires search.llm_provider and search.llm_model (or matching environment variables)")
    return {
        "benchmark": problem_id,
        "ablation_mode": str(ablation["mode"]),
        "ablation": dict(search.get("ablation") or {}),
        "output_dir": target,
        "resume": bool(search.get("resume", False)),
        "real_llm_provider": str(provider),
        "real_llm_model": str(model),
        "llm_api_base": search.get("llm_api_base"),
        "llm_max_tokens": int(search.get("llm_max_tokens", 80_000)),
        "llm_timeout_seconds": int(search.get("llm_timeout_seconds", 60)),
        "maximum_sampling_points": search.get(
            "maximum_sampling_points", resources["maximum_sampling_points"]
        ),
        "maximum_model_parameters": int(search.get("maximum_model_parameters", 2_000_000)),
        "evaluation_size": search.get("evaluation_size", resources["evaluation_size"]),
        "residual_evaluation_size": int(
            search.get("residual_evaluation_size", resources["residual_evaluation_size"])
        ),
        "constraint_resample_interval": int(search.get("constraint_resample_interval", 50)),
        "component_gradient_diagnostics_enabled": True,
        "component_gradient_imbalance_threshold": 100.0,
        "component_gradient_conflict_threshold": -0.1,
        "maximum_gradient_diagnostic_components": 12,
        "maximum_gradient_conflicting_pairs": 5,
        "seed": int(config.get("seed", 0)),
        "independent_run_id": config.get("independent_run_id"),
        "posterior_context_max_tokens": int(search.get("posterior_context_max_tokens", 20_000)),
        "device": str(search.get("device", "cpu")),
        "llm_max_concurrency": max(
            1, min(3, int(search.get("llm_max_concurrency", 3)))
        ),
        "console_log_level": normalize_console_log_level(
            search.get("console_log_level")
        ),
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = config_from_args(args)
    problem, target, resources = prepare_experiment(
        args.problem,
        config,
        output_dir=args.output_dir,
        write_artifacts=args.dry_run,
    )
    if args.dry_run:
        print(json.dumps({
            "status": "passed",
            "problem_id": problem.problem_id,
            "output_dir": str(target),
            "ablation": resolved_ablation_contract(config.get("search") or {}),
            "artifacts": list(REQUIRED_ARTIFACTS),
        }, ensure_ascii=False, indent=2))
        return
    resolved_runtime_config = runtime_config(args.problem, config, target, resources)
    result = run_closed_loop(resolved_runtime_config)
    _finalize_formal_artifacts(target, problem, resources, result)
    emit_run_result(
        result,
        level=resolved_runtime_config["console_log_level"],
    )


if __name__ == "__main__":
    main()
