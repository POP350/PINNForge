"""Run the Open AlgorithmSpec PINN search."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from forge.experiments.closed_loop import run_closed_loop
from forge import get_problem, list_problems
from forge.pipeline.d_algorithm_generation.experiment_contract import (
    EXPERIMENT_CONTRACT_VERSION,
    build_contract_dry_run,
)
from forge.utils.console_output import (
    CONSOLE_LOG_LEVELS,
    emit_run_result,
)
from forge.ablation import (
    ABLATION_MODES,
    STANDARD_ABLATION_MODE,
    normalize_ablation_mode,
    resolved_ablation_contract,
)


def main() -> None:
    args = parse_args()
    if args.list_problems:
        print(json.dumps(list_problems(), indent=2, ensure_ascii=False))
        return
    if args.dry_run:
        report = build_contract_dry_run(benchmark=args.benchmark)
        ablation_config: dict[str, Any] = {
            "ablation_mode": args.ablation_mode,
        }
        if args.execution_feedback is not None:
            ablation_config["ablation"] = {
                "execution_feedback": bool(args.execution_feedback)
            }
        report["ablation"] = resolved_ablation_contract(ablation_config)
        report["problem_validation"] = validate_problem(args.benchmark)
        output_dir = _resolve_output_dir(args)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "contract_dry_run.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return
    result = run_closed_loop(config_from_args(args))
    emit_run_result(_jsonable(result), level=args.console_log_level)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate, validate, build, train, and evolve complete Open AlgorithmSpec objects."
    )
    parser.add_argument("--list-problems", action="store_true", help="List registered problem IDs and exit.")
    parser.add_argument("--benchmark", default="burgers_1d")
    parser.add_argument(
        "--ablation",
        "--ablation-mode",
        dest="ablation_mode",
        type=normalize_ablation_mode,
        choices=ABLATION_MODES,
        default=STANDARD_ABLATION_MODE,
        help=(
            "full; no_knowledge; no_execution_feedback; or "
            "no_evolutionary_search. --ablation-mode is a compatibility alias."
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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the adaptive, at-most-ten-generation structure without LLM calls or training.",
    )
    parser.add_argument("--llm-provider", default=os.environ.get("LLM_PROVIDER"))
    parser.add_argument("--llm-model", default=os.environ.get("LLM_MODEL"))
    parser.add_argument("--llm-api-base", default=None)
    parser.add_argument("--llm-max-tokens", type=int, default=80_000)
    parser.add_argument(
        "--llm-max-concurrency",
        type=int,
        choices=(1, 2, 3),
        default=3,
        help=(
            "Maximum concurrent remote LLM design requests. With 1, the next request "
            "overlaps the previous candidate's single-GPU training."
        ),
    )
    parser.add_argument(
        "--llm-retry-low-diversity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Retry valid evolutionary candidates that are too similar to a sibling or parent. "
            "Use --no-llm-retry-low-diversity to keep them with an audit warning; exact and "
            "negligible executable duplicates are still retried."
        ),
    )
    parser.add_argument("--llm-timeout-seconds", type=int, default=60)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--output-detail",
        choices=("metrics", "compact", "diagnostic"),
        default=None,
        help=(
            "metrics (default) keeps only each generation's best MSE and relative L2 after completion; "
            "compact writes bounded candidate histories and summarized diagnostics; "
            "diagnostic also preserves complete per-step and spatial/time-series files."
        ),
    )
    parser.add_argument(
        "--console-log-level",
        choices=CONSOLE_LOG_LEVELS,
        default="normal",
        help=(
            "quiet suppresses console status, normal emits bounded summaries, "
            "and verbose prints the complete result JSON."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume low-fidelity/escape/final-review state from an existing "
            "output directory created by the same experiment contract."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--independent-run-id",
        default=None,
        help="Optional audit ID for this isolated run; a unique ID is generated when omitted.",
    )
    parser.add_argument(
        "--posterior-context-max-tokens",
        type=_parse_positive_integer,
        default=20_000,
        help="Approximate token budget for run-scoped posterior context injected into Gen1+ design calls.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--maximum-sampling-points",
        type=_parse_optional_positive_integer,
        default=None,
        help="Optional global sampling-point limit; omit it or use none/unlimited for no limit.",
    )
    parser.add_argument("--maximum-model-parameters", type=int, default=2_000_000)
    parser.add_argument(
        "--evaluation-size",
        type=_parse_evaluation_size,
        default=None,
        help="Positive test-set cap, or none/full/all for PINNacle's full reference grid; analytic-only PDEs use a deterministic 2,500/20,000-point grid.",
    )
    parser.add_argument(
        "--residual-evaluation-size",
        type=_parse_positive_integer,
        default=8192,
        help="Requested size of the independent deterministic interior grid used for PDE-residual metrics.",
    )
    parser.add_argument(
        "--constraint-resample-interval",
        type=_parse_nonnegative_integer,
        default=50,
        help="Reuse boundary/initial batches for this many training steps; use 0 to keep them fixed.",
    )
    parser.add_argument(
        "--component-gradient-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Collect trusted per-loss-component gradient audits at the first, middle, and final checkpoints.",
    )
    parser.add_argument(
        "--component-gradient-imbalance-threshold",
        type=float,
        default=100.0,
        help="Flag component-gradient imbalance above this largest/smallest nonzero norm ratio.",
    )
    parser.add_argument(
        "--component-gradient-conflict-threshold",
        type=float,
        default=-0.1,
        help="Flag a component-gradient pair when cosine similarity is below this value.",
    )
    parser.add_argument(
        "--maximum-gradient-diagnostic-components",
        type=_parse_positive_integer,
        default=12,
        help="Maximum registered major physical loss components in one gradient audit.",
    )
    parser.add_argument(
        "--maximum-gradient-conflicting-pairs",
        type=_parse_positive_integer,
        default=5,
        help="Maximum conflicting component pairs retained in one gradient audit.",
    )
    return parser.parse_args()


def _parse_evaluation_size(value: str) -> int | None:
    normalized = str(value).strip().lower()
    if normalized in {"none", "full", "all"}:
        return None
    try:
        resolved = int(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "evaluation size must be a positive integer or one of: none, full, all"
        ) from exc
    if resolved < 1:
        raise argparse.ArgumentTypeError("evaluation size must be a positive integer")
    return resolved


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


def config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = _resolve_output_dir(args)
    if not args.llm_provider or not args.llm_model:
        raise ValueError("A production LLM provider and model must be configured")
    provider = str(args.llm_provider)
    model = str(args.llm_model)
    return {
        "benchmark": args.benchmark,
        "ablation_mode": normalize_ablation_mode(args.ablation_mode),
        **(
            {"ablation": {"execution_feedback": bool(args.execution_feedback)}}
            if args.execution_feedback is not None
            else {}
        ),
        "output_dir": output_dir,
        **(
            {"output_detail": args.output_detail}
            if getattr(args, "output_detail", None) is not None
            else {}
        ),
        **({"resume": True} if getattr(args, "resume", False) else {}),
        "real_llm_provider": provider,
        "real_llm_model": model,
        "llm_api_base": args.llm_api_base,
        "llm_max_tokens": args.llm_max_tokens,
        "llm_max_concurrency": args.llm_max_concurrency,
        "llm_retry_low_diversity": args.llm_retry_low_diversity,
        "llm_timeout_seconds": args.llm_timeout_seconds,
        "maximum_sampling_points": args.maximum_sampling_points,
        "maximum_model_parameters": args.maximum_model_parameters,
        "evaluation_size": args.evaluation_size,
        "residual_evaluation_size": args.residual_evaluation_size,
        "constraint_resample_interval": args.constraint_resample_interval,
        "component_gradient_diagnostics_enabled": (
            args.component_gradient_diagnostics
        ),
        "component_gradient_imbalance_threshold": (
            args.component_gradient_imbalance_threshold
        ),
        "component_gradient_conflict_threshold": (
            args.component_gradient_conflict_threshold
        ),
        "maximum_gradient_diagnostic_components": (
            args.maximum_gradient_diagnostic_components
        ),
        "maximum_gradient_conflicting_pairs": (
            args.maximum_gradient_conflicting_pairs
        ),
        "seed": args.seed,
        "independent_run_id": args.independent_run_id,
        "posterior_context_max_tokens": args.posterior_context_max_tokens,
        "device": args.device,
        "console_log_level": args.console_log_level,
    }


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    config: dict[str, Any] = {
        "ablation_mode": getattr(args, "ablation_mode", None),
    }
    execution_feedback = getattr(args, "execution_feedback", None)
    if execution_feedback is not None:
        config["ablation"] = {"execution_feedback": bool(execution_feedback)}
    mode = str(resolved_ablation_contract(config)["mode"])
    mode_path = [] if mode == STANDARD_ABLATION_MODE else ["ablations", mode]
    return Path(
        args.output_dir
        or Path("outputs")
        / EXPERIMENT_CONTRACT_VERSION
        / args.benchmark
        / Path(*mode_path)
        / f"seed_{args.seed}"
    )


def validate_problem(problem_id: str) -> dict[str, Any]:
    """Validate one registered PDE through the common problem interface."""

    problem_spec = get_problem(problem_id).get_spec().model_dump()
    return {
        "registered": True,
        "problem_id": problem_spec.get("problem_id", problem_id),
        "input_dimension": len(problem_spec.get("input_variables") or []),
        "output_dimension": len(problem_spec.get("output_variables") or []),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


if __name__ == "__main__":
    main()
