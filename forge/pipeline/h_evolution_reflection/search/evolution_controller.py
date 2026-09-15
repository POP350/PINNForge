"""Single default closed loop for complete Open AlgorithmSpec candidates."""

from __future__ import annotations

from copy import deepcopy
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime
import gc
import hashlib
import json
import math
from pathlib import Path
import shutil
from statistics import median
from threading import BoundedSemaphore
import time
from typing import Any, Callable
from uuid import uuid4

from forge.pipeline.d_algorithm_generation.agents.candidate_roles import (
    ALLOWED_VARIATION_STRATEGIES,
    CANDIDATE_ROLE_DEFINITIONS,
    INDEPENDENT_PROPOSAL_ROLE_PLAN,
    NEW_CANDIDATE_ROLE_PLAN,
    candidate_role_definition,
)
from forge.pipeline.d_algorithm_generation.agents.initialization_candidate_roles import (
    GENERATION_ZERO_ROLE_DEFINITIONS,
    GENERATION_ZERO_ROLE_PLAN,
    generation_zero_role_map,
)
from forge.pipeline.d_algorithm_generation.agents.design_agent import (
    DEFAULT_DESIGN_MAX_TOKENS,
    DesignAgent,
    resolve_shared_design_context,
)
from forge.pipeline.d_algorithm_generation.specs import llm_algorithm_spec_option_registry
from forge.pipeline.d_algorithm_generation.adaptive_policy import (
    adaptive_policy_spec,
    build_trainer_capability_summary,
)
from forge.pipeline.d_algorithm_generation.initialization_diversity import (
    audit_candidate_similarity,
    executable_algorithm_spec,
)
from forge.pipeline.d_algorithm_generation.experiment_contract import (
    DIVERSITY_RETRY_DISTANCE,
    DIVERSITY_WARNING_DISTANCE,
    EXACT_DUPLICATE_DISTANCE,
    EVOLUTION_PARENT_COUNT,
    EVOLUTION_TARGET_CANDIDATES,
    DEFAULT_EXPERIMENT_CONTRACT,
    EXPERIMENT_CONTRACT_VERSION,
    GENERATION_ITERATION_RECOMMENDATIONS,
    GENERATION_ZERO_TRAINING_CANDIDATES,
    GENERATION_ZERO_TARGET_CANDIDATES,
    MINIMUM_GENERATION_CANDIDATES,
    MINIMUM_GENERATION_ZERO_CANDIDATES,
    MINIMUM_NEW_EVOLUTION_CANDIDATES,
    MINIMUM_SUCCESSFUL_CANDIDATES,
    MAX_GENERATIONS,
    HIGH_FIDELITY_ITERATIONS,
    LOW_FIDELITY_ITERATIONS,
    REGULAR_GENERATION_RECOMMENDED_ITERATIONS,
    ROLE_DIVERSITY_RETRY_THRESHOLDS,
    adapt_algorithm_spec_to_generation_budget,
    audit_algorithm_spec_iteration_recommendation,
    generation_can_continue,
    resolve_experiment_contract,
    resolve_generation_iterations,
    resolve_ranking_fidelity,
)
from forge.pipeline.h_evolution_reflection.reflection_agent import ReflectionAgent
from forge.pipeline.f_pinn_construction.builders import BuildError, PINNBuilder, builder_capabilities
from forge.pipeline.g_training_evaluation.evaluation import (
    PRIMARY_RANKING_METRIC,
    OpenSpecEvaluator,
)
from forge.pipeline.i_artifact_feedback import PosteriorWriter
from forge.pipeline.posterior_memory import (
    RunScopedPosteriorMemory,
    build_candidate_posterior,
    build_generation_posterior,
)
from forge.pipeline.c_knowledge_retrieval import get_pinnacle_profile
from forge.pipeline.d_algorithm_generation.llm.provider import build_provider, redact_secret
from forge.pipeline.d_algorithm_generation.llm.search_context import retrieve_prior_context
from forge.pipeline.b_feature_extraction.feature_extractor import extract_problem_features
from forge.pipeline.a_problem_definition.problems import get_problem
from forge.pipeline.h_evolution_reflection.search.population_manager import PopulationManager
from forge.pipeline.h_evolution_reflection.search.global_best_tracker import (
    GlobalBestTracker,
)
from forge.pipeline.h_evolution_reflection.search.multifidelity_control import (
    advance_search_after_generation,
    build_global_best_tracker,
    initial_search_state,
    prepare_next_generation,
    select_distinct_history_top_candidates,
    summarize_global_top3_archive,
    summarize_low_fidelity_generation,
    transition_search_phase,
    validate_search_state,
)
from forge.pipeline.h_evolution_reflection.search.candidate_artifacts import (
    artifact_reference,
    compact_candidate_index,
    compact_candidate_record,
    compact_final_metrics,
    compact_training_history,
    compact_training_report,
    history_with_required_events,
    load_runtime_candidate_records,
    normalize_output_detail,
    residual_diagnostics,
    resolve_history_interval,
    temporal_diagnostics,
    write_json as write_candidate_json,
    write_jsonl as write_candidate_jsonl,
)
from forge.pipeline.g_training_evaluation.training import OpenSpecTrainer
from forge.utils.reproducibility import set_global_seed
from forge.pipeline.e_credibility_assessment.validation import validate_algorithm_spec
from forge.utils.json_safety import json_safe
from forge.utils.console_output import console_enabled
from forge.ablation import (
    NO_EVOLUTIONARY_SEARCH_ABLATION_MODE,
    STANDARD_ABLATION_MODE,
    resolved_ablation_contract,
)


MAXIMUM_ROLE_TEMPERATURE = 0.85
DEFAULT_LLM_MAX_CONCURRENCY = 3
MAX_GENERATION_ZERO_REFILL_ROUNDS = 8
MAX_TRAINING_RECOVERY_REFILL_ROUNDS = 3
MAX_CANDIDATE_SHORTFALL_RECOVERY_CANDIDATES = 8
# Escape is a higher-exploration policy applied to the same five evolution
# roles. Keeping a single role contract prevents the escape branch from
# drifting away from EVOLUTION_TARGET_CANDIDATES.
ESCAPE_CANDIDATE_ROLE_PLAN = NEW_CANDIDATE_ROLE_PLAN


class _SingleWorkerCandidatePipeline:
    """Queue candidate work on one worker while the producer keeps running."""

    def __init__(self, consumer: Callable[[dict[str, Any]], None]) -> None:
        self._consumer = consumer
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="single-gpu-training",
        )
        self._worker_slot = BoundedSemaphore(1)
        self._futures: list[Future] = []
        self._closed = False

    def submit(self, child: dict[str, Any]) -> None:
        if self._closed:
            raise RuntimeError("Candidate training pipeline is already closed")
        # Apply backpressure instead of accumulating every generated candidate:
        # one PINN trains while exactly the next LLM request is allowed to run.
        self._worker_slot.acquire()
        try:
            future = self._executor.submit(self._consume_and_release, child)
        except BaseException:
            self._worker_slot.release()
            raise
        self._futures.append(future)

    def _consume_and_release(self, child: dict[str, Any]) -> None:
        try:
            self._consumer(child)
        finally:
            self._worker_slot.release()

    def close(
        self,
        *,
        cancel_pending: bool = False,
        propagate_errors: bool = True,
    ) -> None:
        if self._closed:
            return
        self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=cancel_pending)
        if not propagate_errors:
            return
        for future in self._futures:
            if not future.cancelled():
                future.result()


def _problem_owned_search_profile(problem: Any) -> dict[str, Any]:
    """Expose external-problem constraint and stability options to the designer."""

    spec = problem.get_spec()
    metadata = dict(spec.metadata or {})
    enforcement = dict(problem.constraint_enforcement_capabilities() or {})
    options: list[dict[str, str]] = []
    hard_transforms: list[str] = []
    for method, config in enforcement.items():
        transform_ids = list(dict(config or {}).get("transform_ids") or [])
        for transform_id in transform_ids:
            options.append(
                {"method": str(method), "transform_id": str(transform_id)}
            )
            if str(method) == "problem_hard":
                hard_transforms.append(str(transform_id))
    return {
        "pinnacle_name": spec.name,
        "aliases": [],
        "core_benchmark": False,
        "task_type": spec.task_type,
        "family": metadata.get("pde_family") or metadata.get("equation_type"),
        "spatial_dimension": metadata.get("spatial_dimension"),
        "input_dimension": len(spec.input_variables),
        "output_dimension": len(spec.output_variables),
        "time_dependent": bool(metadata.get("time_dependent")),
        "differential_order": metadata.get("highest_derivative_order"),
        "governing_equations": len(spec.governing_laws),
        "geometry": spec.domain.geometry,
        "constraint_types": [item.constraint_type for item in spec.constraints],
        "constraint_enforcement_options": options,
        "hard_constraint_transforms": hard_transforms,
        "challenge_tags": list(metadata.get("challenge_tags") or []),
        "physical_parameters": dict(metadata.get("parameters") or {}),
        "stability_guidance": deepcopy(metadata.get("stability_guidance") or {}),
        "pinnacle_algorithm_parameter_space": {},
    }


def run_open_algorithm_spec_search(config: dict[str, Any]) -> dict[str, Any]:
    """Run one isolated experiment and always destroy its live posterior state."""

    config = dict(config)
    resolved_ablation = resolved_ablation_contract(config)
    mode = str(resolved_ablation["mode"])
    config["ablation_mode"] = mode
    config["ablation_contract"] = resolved_ablation
    output_dir = Path(config.get("output_dir") or "outputs/open_algorithm_spec_search")
    problem_id = str(config.get("benchmark") or "burgers_1d")
    seed = int(config.get("seed") or 0)
    run_id = str(
        config.get("independent_run_id")
        or f"seed_{seed}_{uuid4().hex[:12]}"
    )
    memory = RunScopedPosteriorMemory(
        pde_id=problem_id,
        run_id=run_id,
        posterior_context_max_tokens=int(config.get("posterior_context_max_tokens") or 20_000),
    )
    memory.start_run()
    output_detail = normalize_output_detail(config.get("output_detail"))
    _console_status(
        config,
        f"[Search] started pde={problem_id} run={run_id}",
        minimum="verbose",
    )
    try:
        result = _run_open_algorithm_spec_search_impl(config, memory)
        snapshot = memory.snapshot_for_archive(
            output_detail=("diagnostic" if output_detail == "diagnostic" else "compact")
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        result["posterior_snapshot"] = snapshot
        if (
            output_detail == "metrics"
            and result.get("final_rank_1")
            and mode == STANDARD_ABLATION_MODE
            and bool(config["ablation_contract"]["execution_feedback_enabled"])
        ):
            metrics_path = output_dir / "generation_best_metrics.json"
            _write_json(metrics_path, _generation_best_metrics_payload(result))
            _prune_completed_metrics_output(output_dir, keep=metrics_path)
            result["experiment_archive"] = {
                "run_result_path": str(metrics_path),
                "automatic_context_reuse": False,
            }
        else:
            snapshot_path = output_dir / "posterior_snapshot.json"
            _write_json(snapshot_path, snapshot)
            summary_name = "open_search_summary.json"
            result["experiment_archive"] = {
                "run_result_path": str(output_dir / summary_name),
                "posterior_snapshot_path": str(snapshot_path),
                "automatic_context_reuse": False,
            }
            _write_json(
                output_dir / summary_name,
                _compact_search_result(result),
            )
            _console_status(
                config, "[Search] posterior snapshot archived", minimum="verbose"
            )
        return result
    finally:
        memory.clear()
        _console_status(
            config, "[Search] posterior memory cleared", minimum="verbose"
        )




def _run_open_algorithm_spec_search_impl(
    config: dict[str, Any], posterior_memory: RunScopedPosteriorMemory
) -> dict[str, Any]:
    config = dict(config)
    ablation = resolved_ablation_contract(config)
    knowledge_enabled = bool(ablation["prior_knowledge_enabled"])
    posterior_enabled = bool(ablation["posterior_knowledge_enabled"])
    execution_feedback_enabled = bool(ablation["execution_feedback_enabled"])
    evolutionary_inheritance_enabled = bool(
        ablation["evolutionary_inheritance_enabled"]
    )
    current_run_posterior_enabled = bool(
        posterior_enabled and execution_feedback_enabled
    )
    output_dir = Path(config.get("output_dir") or "outputs/open_algorithm_spec_search")
    output_detail = normalize_output_detail(config.get("output_detail"))
    experiment_contract = resolve_experiment_contract(
        config.get("experiment_contract")
    )
    search_control = experiment_contract["search_control"]
    stagnation_config = experiment_contract["stagnation"]
    low_fidelity_config = experiment_contract["low_fidelity_evaluation"]
    scaling_config = experiment_contract["low_fidelity_schedule_scaling"]
    generations = int(search_control["max_low_fidelity_generations"])
    execution_generations = generations
    if config.get("_test_mode"):
        execution_generations = int(config.get("_test_execution_generations") or generations)
        if not 1 <= execution_generations <= generations:
            raise ValueError("_test_execution_generations must be within the maximum ten-generation range")
    resume_requested = bool(config.get("resume"))
    _ensure_experiment_contract(
        output_dir,
        resume=resume_requested,
        experiment_contract=experiment_contract,
        ablation_settings=ablation,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_dir / "experiment_contract.json",
        {
            "experiment_contract_version": EXPERIMENT_CONTRACT_VERSION,
            **deepcopy(experiment_contract),
            "maximum_generations": generations,
            "generation_count_is_dynamic": True,
            "generation_iteration_recommendations": GENERATION_ITERATION_RECOMMENDATIONS,
            "regular_generation_recommended_iterations": REGULAR_GENERATION_RECOMMENDED_ITERATIONS,
            "high_fidelity_iterations": HIGH_FIDELITY_ITERATIONS,
            "iteration_recommendations_are_advisory": False,
            "final_high_fidelity_is_independent_stage": True,
            "generation_zero_target_candidates": GENERATION_ZERO_TARGET_CANDIDATES,
            "generation_zero_training_candidates": GENERATION_ZERO_TRAINING_CANDIDATES,
            "generation_zero_minimum_candidates": MINIMUM_GENERATION_ZERO_CANDIDATES,
            "evolution_target_candidates": EVOLUTION_TARGET_CANDIDATES,
            "evolution_minimum_candidates": MINIMUM_GENERATION_CANDIDATES,
            "minimum_new_evolution_candidates": MINIMUM_NEW_EVOLUTION_CANDIDATES,
            "minimum_successful_candidates_to_continue": MINIMUM_SUCCESSFUL_CANDIDATES,
            "role_diversity_retry_thresholds": ROLE_DIVERSITY_RETRY_THRESHOLDS,
            "exact_duplicate_distance": EXACT_DUPLICATE_DISTANCE,
            "diversity_retry_distance": DIVERSITY_RETRY_DISTANCE,
            "diversity_warning_distance": DIVERSITY_WARNING_DISTANCE,
            "screening_enabled": False,
            "promotion_enabled": False,
            "checkpoint_continuation_enabled": False,
            "resume_search_state_enabled": True,
            "lbfgs_execution_policy": "persistent_chunked_internal_iterations_v1",
            "output_detail": output_detail,
            "lbfgs_phase_iterations_count": "pytorch_internal_n_iter",
            "lbfgs_fixed_full_batch_within_phase": True,
            "constraint_resample_interval": max(
                0, int(config.get("constraint_resample_interval", 50))
            ),
            "ablation": deepcopy(ablation),
        },
    )
    problem_id = str(config.get("benchmark") or "burgers_1d")
    seed = int(config.get("seed") or 0)
    set_global_seed(seed)
    problem = get_problem(problem_id)
    if "evaluation_size" in config and hasattr(problem, "evaluation_size"):
        # Assignment is intentional even for None: adapters interpret it as an
        # explicit request for the complete reference evaluation grid.
        problem.evaluation_size = config["evaluation_size"]
    if "residual_evaluation_size" in config and hasattr(
        problem, "residual_evaluation_size"
    ):
        problem.residual_evaluation_size = config["residual_evaluation_size"]
    features = extract_problem_features(problem)
    pinnacle_profile = get_pinnacle_profile(problem_id) or _problem_owned_search_profile(
        problem
    )
    problem_spec = problem.get_spec().model_dump()
    budget = _experiment_budget(
        config,
        generation_index=0,
        total_generations=generations,
    )
    capabilities = builder_capabilities()
    trainer_capabilities_snapshot = build_trainer_capability_summary(
        problem, capabilities
    )
    capabilities["trainer_capabilities_snapshot"] = deepcopy(
        trainer_capabilities_snapshot
    )
    llm_option_registry = llm_algorithm_spec_option_registry()
    provider_name = str(config.get("real_llm_provider") or "")
    model_name = str(config.get("real_llm_model") or "")
    if not provider_name or not model_name:
        raise ValueError("A production LLM provider and model are required")
    provider = build_provider(
        provider=provider_name,
        model=model_name,
        api_key=config.get("llm_api_key"),
        base_url=config.get("llm_api_base"),
        timeout_seconds=int(config.get("llm_timeout_seconds") or 120),
        enable_thinking=config.get("llm_enable_thinking"),
    )
    if knowledge_enabled:
        prior = (
            deepcopy(config["frozen_prior_context"])
            if "frozen_prior_context" in config
            else retrieve_prior_context(
                features, max_items=int(config.get("llm_prior_context_items") or 8)
            )
        )
    else:
        if config.get("frozen_prior_context"):
            raise ValueError(
                "frozen_prior_context cannot be supplied in the no_knowledge ablation"
            )
        prior = []
    if config.get("frozen_posterior_context"):
        raise ValueError(
            "Persistent or externally frozen posterior context is forbidden in an independent run."
        )
    # The formal search never reads the legacy global posterior store.  All
    # measured guidance below is built from this invocation's memory instance.
    posterior: list[dict[str, Any]] = []
    population = PopulationManager(
        maximum_size=EVOLUTION_TARGET_CANDIDATES,
        elite_count=EVOLUTION_PARENT_COUNT,
        objective=PRIMARY_RANKING_METRIC,
    )
    global_best_tracker_path = output_dir / "global_best_tracker.json"
    global_best_tracker = (
        GlobalBestTracker.load(global_best_tracker_path)
        if global_best_tracker_path.exists()
        else GlobalBestTracker(
            default_top_k=int(
                experiment_contract["final_selection"]["candidate_count"]
            )
        )
    )
    record_writer = PosteriorWriter(output_dir / "candidate_records.jsonl")
    design_agent, reflection_agent = DesignAgent(), ReflectionAgent()
    builder = PINNBuilder()
    all_records: list[dict[str, Any]] = []
    generation_results: list[dict[str, Any]] = []
    reflections: list[dict[str, Any]] = []
    llm_trace: list[dict[str, Any]] = []
    total_calls = 0
    generation_feedback: dict[str, Any] = {}
    posterior_limit = int(config.get("llm_posterior_context_items") or 6)
    global_parent_top3: list[dict[str, Any]] = []
    formal_parent_count = EVOLUTION_PARENT_COUNT
    search_state = initial_search_state()
    start_generation = 0
    state_path = output_dir / "search_state.json"
    if resume_requested:
        restored = _load_search_resume_state(output_dir)
        search_state = restored["search_state"]
        generation_results = restored["generation_results"]
        all_records = restored["all_records"]
        reflections = restored["reflections"]
        generation_feedback = restored["generation_feedback"]
        global_parent_top3 = restored["global_top3"]
        start_generation = int(search_state.get("low_fidelity_generation_count") or 0)
        global_best_tracker = build_global_best_tracker(
            all_records,
            candidate_count=int(
                experiment_contract["final_selection"]["candidate_count"]
            ),
        )
        global_best_tracker.save(global_best_tracker_path)
        global_parent_top3, _ = global_best_tracker.top_k(formal_parent_count)
        global_parent_top3 = _hydrate_global_tracker_selection(
            global_parent_top3, all_records
        )
        if current_run_posterior_enabled:
            for record in all_records:
                candidate_posterior = build_candidate_posterior(
                    record,
                    pde_id=problem_id,
                    run_id=posterior_memory.run_id,
                )
                posterior_memory.add_candidate_posterior(
                    pde_id=problem_id,
                    run_id=posterior_memory.run_id,
                    posterior=candidate_posterior,
                )

    low_fidelity_generation_ids = (
        range(start_generation, execution_generations)
        if search_state.get("search_phase")
        in {"normal_search", "escape_scheduled"}
        else ()
    )
    for generation in low_fidelity_generation_ids:
        search_state, generation_mode_name = prepare_next_generation(
            search_state
        )
        generation_recommended_iterations = resolve_generation_iterations(
            generation,
            generations,
        )
        ranking_fidelity = resolve_ranking_fidelity(
            generation,
            generations,
        )
        budget = _experiment_budget(
            config,
            generation_index=generation,
            total_generations=generations,
        )
        generation_recommended_iterations = int(budget["recommended_total_iterations"])
        generation_started_at = datetime.now().isoformat(timespec="seconds")
        generation_timer = time.perf_counter()
        global_best_tracker_before = global_best_tracker.audit_snapshot()
        # Keep execution outcomes available to the Python controller for true
        # ranking, stopping, and escape decisions.  Only the LLM-facing copy is
        # blanked by the ablation.
        controller_feedback = dict(generation_feedback)
        input_feedback = (
            dict(controller_feedback) if execution_feedback_enabled else {}
        )
        generation_records: list[dict[str, Any]] = []
        target_population_size = (
            GENERATION_ZERO_TRAINING_CANDIDATES
            if generation == 0
            else EVOLUTION_TARGET_CANDIDATES
        )
        stage = _search_stage(
            generation,
            generations,
            generation_mode_name == "escape",
        )
        generation_mode = _resolve_generation_mode(
            controller_feedback,
            mode=generation_mode_name,
            stagnation_config=stagnation_config,
        )
        llm_stage = (
            stage
            if execution_feedback_enabled
            else _search_stage(generation, generations, False)
        )
        llm_generation_mode = (
            generation_mode
            if execution_feedback_enabled
            else _resolve_generation_mode(
                {}, mode="normal", stagnation_config=stagnation_config
            )
        )
        run_posterior_context = (
            posterior_memory.build_prompt_context()
            if current_run_posterior_enabled
            else {}
        )
        posterior_principle_evidence = (
            posterior_memory.principle_evidence()
            if current_run_posterior_enabled
            else []
        )
        posterior_failure_patterns = (
            posterior_memory.failure_guidance()
            if current_run_posterior_enabled
            else []
        )
        if generation > 0 and current_run_posterior_enabled:
            token_estimate = int(
                (run_posterior_context.get("metadata") or {}).get("token_estimate") or 0
            )
            _console_status(
                config,
                f"[Search] posterior context tokens={token_estimate}",
                minimum="verbose",
            )
        if generation == 0:
            parent_candidates: list[dict[str, Any]] = []
            target_new_children = len(GENERATION_ZERO_ROLE_PLAN)
        else:
            if len(global_parent_top3) != formal_parent_count:
                raise RuntimeError(
                    "Multi-generation search requires exactly three distinct global "
                    "low-fidelity archive entries before the next generation"
                )
            if evolutionary_inheritance_enabled:
                parent_candidates = [
                    _formal_parent_candidate_view(
                        item,
                        population.objective,
                        execution_feedback=execution_feedback_enabled,
                    )
                    for item in global_parent_top3
                ]
                if not execution_feedback_enabled:
                    parent_candidates = _deterministically_shuffle_parent_views(
                        parent_candidates,
                        seed=seed,
                        generation=generation,
                    )
            else:
                # The archive remains active for stopping and final Top-3
                # promotion, but no selected AlgorithmSpec becomes an LLM
                # parent or enters an independent proposal prompt.
                parent_candidates = []
            target_new_children = EVOLUTION_TARGET_CANDIDATES
            if (
                target_population_size != EVOLUTION_TARGET_CANDIDATES
                or target_new_children != len(NEW_CANDIDATE_ROLE_PLAN)
            ):
                raise ValueError(
                    "Evolution generations require exactly one new candidate per configured role"
                )
        previous_parent_ids = [str(item.get("candidate_id")) for item in parent_candidates]
        variation_budget = {
            **budget,
            "recommended_iterations": budget["recommended_total_iterations"],
            "iteration_recommendation_is_advisory": False,
            "generation_kind": "low_fidelity",
            "fidelity": "low",
            "generation_mode": (
                generation_mode_name if execution_feedback_enabled else "normal"
            ),
            "maximum_generations": generations,
            "generation_count_is_dynamic": True,
            "regular_generation_recommended_iterations": REGULAR_GENERATION_RECOMMENDED_ITERATIONS,
            "high_fidelity_iterations": HIGH_FIDELITY_ITERATIONS,
        }
        llm_max_concurrency = _configured_llm_max_concurrency(config)
        streamed_candidate_ids: set[str] = set()
        pipeline_anchor_reserved = False
        streaming_training_context = {
            "generation": generation,
            "generation_seed": seed,
            "run_seed": seed,
            "generation_mode_name": generation_mode_name,
            "variation_generation_mode": (
                "initialization"
                if generation == 0
                else (
                    "evolution"
                    if evolutionary_inheritance_enabled
                    else "independent_search"
                )
            ),
            "problem": problem,
            "problem_id": problem_id,
            "budget": budget,
            "capabilities": capabilities,
            "builder": builder,
            "device": str(config.get("device") or "cpu"),
            "output_dir": output_dir,
            "output_detail": output_detail,
            "input_feedback": input_feedback,
            "training_history_interval": config.get(
                "training_history_interval"
            ),
            "low_fidelity_config": low_fidelity_config,
            "generation_recommended_iterations": generation_recommended_iterations,
            "ranking_fidelity": ranking_fidelity,
            "trainer_capabilities_snapshot": trainer_capabilities_snapshot,
            "previous_parent_ids": previous_parent_ids,
            "search_state": search_state,
            "evaluation_seed": int(config.get("evaluation_seed") or 0),
        }

        def record_candidate_training(child: dict[str, Any]) -> None:
            child.setdefault("population_role", "llm_generated")
            child.setdefault("fresh_training", True)
            record = _train_low_fidelity_child(
                child,
                child_index=int(child.get("candidate_slot") or 0),
                context=streaming_training_context,
            )
            record["llm_generation_overlapped"] = True
            record["trained_before_next_llm_request"] = False
            generation_records.append(record)
            streamed_candidate_ids.add(str(child.get("candidate_id") or ""))
            _progress(
                config,
                {
                    "event": "candidate_training_complete",
                    "generation": generation,
                    "spec_id": record.get("spec_id"),
                    "status": record.get("status"),
                    "training_success": record.get("training_success"),
                    "failure_reason": record.get("failure_reason"),
                    "overlapped_with_llm_generation": True,
                    "trained_before_next_llm_request": False,
                },
            )

        single_request_training_pipeline = (
            _SingleWorkerCandidatePipeline(record_candidate_training)
            if llm_max_concurrency == 1
            else None
        )

        def train_candidate_while_llm_runs(child: dict[str, Any]) -> None:
            nonlocal pipeline_anchor_reserved
            # With concurrent requests, keep one accepted candidate for the
            # ordinary controller loop while the remaining candidates overlap
            # LLM generation. With one request at a time, enqueue training on a
            # single GPU worker and immediately allow the next LLM request.
            if llm_max_concurrency > 1 and not pipeline_anchor_reserved:
                pipeline_anchor_reserved = True
                return
            if single_request_training_pipeline is not None:
                single_request_training_pipeline.submit(child)
                return
            record_candidate_training(child)

        def generate_candidates_with_training_pipeline(
            producer: Callable[[], dict[str, Any]],
        ) -> dict[str, Any]:
            try:
                result = producer()
            except BaseException:
                if single_request_training_pipeline is not None:
                    single_request_training_pipeline.close(
                        cancel_pending=True,
                        propagate_errors=False,
                    )
                raise
            if single_request_training_pipeline is not None:
                single_request_training_pipeline.close()
            return result

        candidate_ready_callback = train_candidate_while_llm_runs
        if generation == 0:
            shared_design_payload = {
                "provider": provider,
                "problem": problem,
                "generation": generation,
                "problem_features": features,
                "problem_spec": problem_spec,
                "pinnacle_profile": pinnacle_profile,
                "pinnacle_algorithm_parameter_space": (
                    pinnacle_profile.get("pinnacle_algorithm_parameter_space", {})
                    if isinstance(pinnacle_profile, dict)
                    else {}
                ),
                "option_registry": llm_option_registry,
                "prior_knowledge": prior,
                "posterior_knowledge": posterior,
                "posterior_principle_evidence": posterior_principle_evidence,
                "posterior_failure_patterns": posterior_failure_patterns,
                "run_posterior_context": run_posterior_context,
                "parent_specs": [],
                "population_summary": [],
                "generation_feedback": {},
                "search_stage": {"name": "initialization"},
                "generation_mode": {
                    "state": "initialization",
                    "stagnation_detected": False,
                    "trigger_reason": None,
                    "improvement_threshold": stagnation_config[
                        "best_single_generation_threshold"
                    ],
                    "lookback_generations": stagnation_config[
                        "best_improvement_window"
                    ],
                    "behavior_overrides": {},
                },
                "experiment_budget": variation_budget,
                "builder_capabilities": capabilities,
                "trainer_capabilities_snapshot": trainer_capabilities_snapshot,
                "proposal_type": "initialization_design",
                "initialization_mode": True,
                "initial_population_role_map": generation_zero_role_map(),
                "max_tokens": _configured_design_max_tokens(config),
                "retry_low_diversity": bool(
                    config.get("llm_retry_low_diversity", True)
                ),
                "knowledge_base_enabled": knowledge_enabled,
                "execution_feedback": execution_feedback_enabled,
                "evolutionary_inheritance_enabled": evolutionary_inheritance_enabled,
            }
            shared_design_payload["resolved_design_context"] = resolve_shared_design_context(
                shared_design_payload
            )
            variation_result = generate_candidates_with_training_pipeline(
                lambda: _generate_initialization_role_candidates(
                    design_agent=design_agent,
                    shared_payload=shared_design_payload,
                    problem=problem,
                    budget=budget,
                    capabilities=capabilities,
                    minimum_accepted_candidates=MINIMUM_GENERATION_ZERO_CANDIDATES,
                    maximum_concurrency=llm_max_concurrency,
                    on_candidate_ready=candidate_ready_callback,
                )
            )
            _write_generation_zero_candidate_artifacts(
                output_dir / "generation_0", variation_result
            )
        else:
            shared_design_payload = {
                "provider": provider,
                "problem": problem,
                "generation": generation,
                "problem_features": features,
                "problem_spec": problem_spec,
                "pinnacle_profile": pinnacle_profile,
                "pinnacle_algorithm_parameter_space": (
                    pinnacle_profile.get("pinnacle_algorithm_parameter_space", {})
                    if isinstance(pinnacle_profile, dict)
                    else {}
                ),
                "option_registry": llm_option_registry,
                "prior_knowledge": prior,
                "posterior_knowledge": posterior,
                "posterior_principle_evidence": posterior_principle_evidence,
                "posterior_failure_patterns": posterior_failure_patterns,
                "run_posterior_context": run_posterior_context,
                "parent_specs": parent_candidates,
                "population_summary": [
                    {"candidate_id": item.get("candidate_id"), "rank": item.get("rank")}
                    for item in parent_candidates
                ] if execution_feedback_enabled else [],
                "generation_feedback": input_feedback,
                "search_stage": llm_stage,
                "generation_mode": llm_generation_mode,
                "_sampling_generation_mode": generation_mode,
                "experiment_budget": variation_budget,
                "builder_capabilities": capabilities,
                "trainer_capabilities_snapshot": trainer_capabilities_snapshot,
                "proposal_type": (
                    "evolutionary_design"
                    if evolutionary_inheritance_enabled
                    else "budget_matched_independent_design"
                ),
                "max_tokens": _configured_design_max_tokens(config),
                "retry_low_diversity": bool(
                    config.get("llm_retry_low_diversity", True)
                ),
                "knowledge_base_enabled": knowledge_enabled,
                "execution_feedback": execution_feedback_enabled,
                "evolutionary_inheritance_enabled": evolutionary_inheritance_enabled,
            }
            shared_design_payload["resolved_design_context"] = resolve_shared_design_context(
                shared_design_payload
            )
            variation_result = generate_candidates_with_training_pipeline(
                lambda: _generate_role_candidates(
                    design_agent=design_agent,
                    shared_payload=shared_design_payload,
                    generation=generation,
                    parent_candidates=parent_candidates,
                    problem=problem,
                    budget=budget,
                    capabilities=capabilities,
                    role_plan=(
                        (
                            ESCAPE_CANDIDATE_ROLE_PLAN
                            if generation_mode_name == "escape"
                            else NEW_CANDIDATE_ROLE_PLAN
                        )
                        if evolutionary_inheritance_enabled
                        else INDEPENDENT_PROPOSAL_ROLE_PLAN
                    ),
                    maximum_concurrency=llm_max_concurrency,
                    on_candidate_ready=candidate_ready_callback,
                )
            )
        generated_children = list(variation_result.get("children") or [])
        current_children = [
            {**child, "population_role": "llm_generated", "fresh_training": True}
            for child in generated_children
        ]
        valid_new_candidate_count = len(generated_children) if generation > 0 else len(current_children)
        pretraining_recovery_count = 0
        while not generation_can_continue(
            generation,
            actual_candidate_count=len(current_children),
            valid_new_candidate_count=valid_new_candidate_count,
        ):
            if (
                pretraining_recovery_count
                >= MAX_CANDIDATE_SHORTFALL_RECOVERY_CANDIDATES
            ):
                raise RuntimeError(
                    "Top-3 hard stop after bounded pretraining candidate recovery was exhausted: "
                    f"generation={generation}, accepted={len(current_children)}, "
                    f"required={MINIMUM_GENERATION_CANDIDATES}, "
                    f"recovery_candidates={pretraining_recovery_count}"
                )
            pretraining_recovery_count += 1
            recovery_result = _generate_training_recovery_candidate(
                design_agent=design_agent,
                shared_payload=shared_design_payload,
                generation=generation,
                recovery_index=pretraining_recovery_count,
                parent_candidates=parent_candidates,
                current_candidates=current_children,
                generation_records=[],
                problem=problem,
                budget=budget,
                capabilities=capabilities,
                objective=population.objective,
            )
            recovery_child = {
                **recovery_result["child"],
                "population_role": "llm_generated_recovery",
                "fresh_training": True,
            }
            current_children.append(recovery_child)
            generated_children.append(recovery_child)
            _merge_variation_result(variation_result, recovery_result)
            valid_new_candidate_count = (
                len(generated_children) if generation > 0 else len(current_children)
            )
        total_calls += int(variation_result.get("llm_calls") or 0)
        _append_llm_traces(
            llm_trace,
            variation_result.get("traces") or [],
            generation=generation,
            generation_mode=str(variation_result.get("generation_mode") or "evolution"),
            feedback_source_generation=input_feedback.get("source_generation"),
            provider_name=provider_name,
            model_name=model_name,
        )
        # Persist provider evidence before long-running candidate training.
        _write_json(
            output_dir / "llm_trace.json",
            _compact_llm_trace(llm_trace, output_detail=output_detail),
        )
        # Paired-seed architecture validation: every candidate in one run is
        # rebuilt after resetting the exact same run-level seed.
        generation_seed = seed
        child_index = 0
        training_children = [
            child
            for child in current_children
            if str(child.get("candidate_id") or "")
            not in streamed_candidate_ids
        ]
        training_recovery_count = pretraining_recovery_count
        training_recovery_exhausted = False
        required_successful_candidates = MINIMUM_SUCCESSFUL_CANDIDATES
        maximum_training_candidates = (
            GENERATION_ZERO_TRAINING_CANDIDATES
            if generation == 0
            else EVOLUTION_TARGET_CANDIDATES
        )
        if len(generation_records) + len(training_children) > maximum_training_candidates:
            raise AssertionError(
                "generated low-fidelity candidates exceed the experiment budget: "
                f"generation={generation}, generated="
                f"{len(generation_records) + len(training_children)}, "
                f"cap={maximum_training_candidates}"
            )

        def enqueue_training_recovery_candidate(recovery_index: int) -> None:
            nonlocal total_calls
            scheduled_or_completed = (
                len(generation_records)
                + len(training_children)
                - child_index
            )
            if scheduled_or_completed >= maximum_training_candidates:
                raise RuntimeError(
                    "candidate recovery would exceed the budget-matched "
                    f"generation training cap: generation={generation}, "
                    f"cap={maximum_training_candidates}"
                )
            successful_candidates = _successful_candidate_count(
                generation_records, population.objective
            )
            _progress(
                config,
                {
                    "event": "candidate_shortfall_llm_refill_started",
                    "generation": generation,
                    "successful_candidates": successful_candidates,
                    "required_candidates": required_successful_candidates,
                    "recovery_index": recovery_index,
                },
            )
            recovery_result = _generate_training_recovery_candidate(
                design_agent=design_agent,
                shared_payload=shared_design_payload,
                generation=generation,
                recovery_index=recovery_index,
                parent_candidates=parent_candidates,
                current_candidates=current_children,
                generation_records=generation_records,
                problem=problem,
                budget=budget,
                capabilities=capabilities,
                objective=population.objective,
            )
            recovery_child = {
                **recovery_result["child"],
                "population_role": "llm_generated_recovery",
                "fresh_training": True,
            }
            current_children.append(recovery_child)
            training_children.append(recovery_child)
            generated_children.append(recovery_child)
            _merge_variation_result(variation_result, recovery_result)
            recovery_calls = int(recovery_result.get("llm_calls") or 0)
            total_calls += recovery_calls
            _append_llm_traces(
                llm_trace,
                recovery_result.get("traces") or [],
                generation=generation,
                generation_mode="candidate_shortfall_recovery",
                feedback_source_generation=generation,
                provider_name=provider_name,
                model_name=model_name,
            )
            _write_json(
                output_dir / "llm_trace.json",
                _compact_llm_trace(llm_trace, output_detail=output_detail),
            )
            _progress(
                config,
                {
                    "event": "candidate_shortfall_llm_refill_generated",
                    "generation": generation,
                    "candidate_id": recovery_child.get("candidate_id"),
                    "recovery_index": recovery_index,
                    "llm_calls": recovery_calls,
                },
            )

        while True:
            if child_index >= len(training_children):
                if (
                    _successful_candidate_count(
                        generation_records, population.objective
                    )
                    >= required_successful_candidates
                ):
                    break
                if (
                    training_recovery_count
                    >= MAX_CANDIDATE_SHORTFALL_RECOVERY_CANDIDATES
                ):
                    training_recovery_exhausted = True
                    break
                scheduled_or_completed = (
                    len(generation_records)
                    + len(training_children)
                    - child_index
                )
                if scheduled_or_completed >= maximum_training_candidates:
                    training_recovery_exhausted = True
                    break
                training_recovery_count += 1
                enqueue_training_recovery_candidate(training_recovery_count)
                continue
            child = training_children[child_index]
            set_global_seed(generation_seed)
            generation_mode = str(child.get("generation_mode") or variation_result.get("generation_mode") or "evolution")
            candidate_id = _generation_candidate_id(generation, child_index, child)
            record = _evaluate_candidate(
                candidate_id=candidate_id,
                spec=child.get("algorithm_spec"),
                generation=generation,
                operation=generation_mode,
                problem=problem,
                problem_id=problem_id,
                budget=budget,
                capabilities=capabilities,
                builder=builder,
                device=str(config.get("device") or "cpu"),
                seed=generation_seed,
                train_candidate=True,
                checkpoint_root=(
                    output_dir / "checkpoints" / candidate_id
                    if output_detail == "diagnostic"
                    else None
                ),
                generation_feedback=input_feedback,
                training_iteration_limit=None,
                resume_checkpoint=None,
                history_interval=resolve_history_interval(
                    int(budget["required_total_iterations"]),
                    output_detail=output_detail,
                    configured_interval=config.get("training_history_interval"),
                ),
                reference_mse_interval=max(1, generation_recommended_iterations // 10),
                reference_mse_checkpoints=list(
                    low_fidelity_config["checkpoint_iterations"]
                ),
                design_algorithm_spec=deepcopy(
                    child.get("search_algorithm_spec")
                    or child.get("algorithm_spec")
                    or {}
                ),
                runtime_scaling_audit=deepcopy(
                    (child.get("optimization_budget_audit") or {}).get(
                        "adaptive_policy_scaling"
                    ) or {}
                ),
                trainer_capabilities_snapshot=trainer_capabilities_snapshot,
            )
            record = _apply_low_fidelity_proxy(
                record,
                checkpoint_iterations=list(
                    low_fidelity_config["checkpoint_iterations"]
                ),
            )
            record.update(
                {
                    "generation_index": generation,
                    "generation_iteration_recommendation": generation_recommended_iterations,
                    "iteration_recommendation_is_advisory": False,
                    "generation_kind": "low_fidelity",
                    "ranking_fidelity": ranking_fidelity,
                    "experiment_contract_version": EXPERIMENT_CONTRACT_VERSION,
                    "fidelity": "low",
                    "generation_mode": generation_mode_name,
                    "proposal_generation_mode": generation_mode,
                    "training_iterations": int(
                        budget["required_total_iterations"]
                    ),
                    "fresh_initialization": True,
                    "resumed_from_checkpoint": False,
                    "escape_parent_context": (
                        list(previous_parent_ids)
                        if generation_mode_name == "escape"
                        else []
                    ),
                    "escape_rationale": (
                        str(
                            (
                                child.get("proposal_metadata") or {}
                            ).get("design_rationale")
                            or child.get("design_rationale")
                            or ""
                        )
                        if generation_mode_name == "escape"
                        else ""
                    ),
                    "pre_escape_global_best_mse": (
                        search_state.get("pre_escape_global_best_mse")
                        if generation_mode_name == "escape"
                        else None
                    ),
                    "parent_spec_ids": list(child.get("parent_ids") or []),
                    "design_intent": child.get("design_intent") or "",
                    "difference_summary": child.get("difference_summary") or "",
                    "variation_reason": child.get("variation_reason") or "",
                    "heritage_explanation": child.get("heritage_explanation") or {},
                    "expected_improvements": child.get("expected_improvements") or [],
                    "candidate_slot": child.get("candidate_slot"),
                    "candidate_role": child.get("candidate_role"),
                    "proposal_metadata": deepcopy(child.get("proposal_metadata") or {}),
                    "generation_policy_audit": deepcopy(
                        child.get("generation_policy_audit") or {}
                    ),
                    "variation_strategy": child.get("variation_strategy"),
                    "inherited_strengths": list(child.get("inherited_strengths") or []),
                    "major_changes": list(child.get("major_changes") or []),
                    "design_rationale": child.get("design_rationale") or "",
                    "proposal_validation": deepcopy(child.get("proposal_validation") or {}),
                    "role_design_objective": child.get("role_design_objective") or "",
                    "primary_diversity_axis": child.get("primary_diversity_axis"),
                    "nearest_candidate_id": child.get("nearest_candidate_id"),
                    "nearest_candidate_role": child.get("nearest_candidate_role"),
                    "nearest_candidate_distance": child.get("nearest_candidate_distance"),
                    "normalized_executable_distance": child.get("normalized_executable_distance"),
                    "overlapping_modules": child.get("overlapping_modules") or [],
                    "different_modules": child.get("different_modules") or [],
                    "similarity_status": child.get("similarity_status") or "not_audited",
                    "diversity_retry_count": int(child.get("diversity_retry_count") or 0),
                    "exact_duplicate": bool(child.get("exact_duplicate")),
                    "exact_duplicate_after_retry": bool(child.get("exact_duplicate_after_retry")),
                    "accepted_despite_similarity": bool(child.get("accepted_despite_similarity")),
                    "local_design_variant": bool(child.get("local_design_variant")),
                    "diversity_retry_performed": bool(child.get("diversity_retry_performed")),
                    "diversity_warning": bool(child.get("diversity_warning")),
                    "population_role": child.get("population_role") or "llm_generated",
                    "source_candidate_id": child.get("source_candidate_id"),
                    "fresh_training": bool(child.get("fresh_training", True)),
                    "historical_checkpoint_reused": False,
                    "historical_metrics_reused": False,
                    "algorithm_spec": record.get("normalized_algorithm_spec") or {},
                    "search_algorithm_spec": deepcopy(
                        child.get("search_algorithm_spec")
                        or record.get("normalized_algorithm_spec")
                        or {}
                    ),
                    "algorithm_spec_raw": deepcopy(
                        child.get("algorithm_spec_raw")
                        if child.get("algorithm_spec_raw") is not None
                        else record.get("algorithm_spec_raw")
                    ),
                    "training_seed": generation_seed,
                    "run_seed": seed,
                    "model_initialization_seed": seed,
                    "sampling_seed": seed,
                    "evaluation_seed": int(config.get("evaluation_seed") or 0),
                    "llm_seed": None,
                    "llm_seed_supported": False,
                    "training_budget": {
                        "generation_iteration_recommendation": generation_recommended_iterations,
                        "iteration_recommendation_is_advisory": False,
                        "generation_kind": "low_fidelity",
                        "ranking_fidelity": ranking_fidelity,
                    },
                    "optimization_budget_audit": child.get("optimization_budget_audit") or {},
                    "formal_status": "completed" if record.get("training_success") else "failed",
                    "generation_metrics": json_safe(record.get("metrics") or {}),
                "generation_checkpoint": str(output_dir / "checkpoints" / candidate_id / "final_checkpoint.pt")
                    if record.get("training_success") and output_detail == "diagnostic"
                    else None,
                    "generation_rank": None,
                }
            )
            if generation == 0:
                for field in (
                    "nearest_candidate_id",
                    "nearest_candidate_role",
                    "nearest_candidate_distance",
                    "normalized_executable_distance",
                    "overlapping_modules",
                    "different_modules",
                    "similarity_status",
                    "diversity_retry_count",
                    "exact_duplicate",
                    "exact_duplicate_after_retry",
                    "accepted_despite_similarity",
                    "local_design_variant",
                    "diversity_retry_performed",
                    "diversity_warning",
                ):
                    record.pop(field, None)
            _write_candidate_runtime_snapshot(
                output_dir,
                record,
                output_detail=output_detail,
            )
            generation_records.append(record)
            _progress(config, {"event": "candidate_training_complete", "generation": generation, "spec_id": record.get("spec_id"), "status": record.get("status"), "training_success": record.get("training_success"), "failure_reason": record.get("failure_reason")})
            child_index += 1
        minimum_generated_candidates = (
            MINIMUM_GENERATION_ZERO_CANDIDATES
            if generation == 0
            else MINIMUM_GENERATION_CANDIDATES
        )
        successful_candidate_shortfall = max(
            0,
            MINIMUM_SUCCESSFUL_CANDIDATES
            - _successful_candidate_count(generation_records, population.objective),
        )
        generation_records = _rank_generation_formal_candidates(
            generation_records,
            objective=population.objective,
            count=formal_parent_count,
            ranking_fidelity=ranking_fidelity,
        )
        # Rewrite only the per-candidate public record after formal ranks exist.
        # Search and ranking continue to use the complete in-memory records.
        for ranked_record in generation_records:
            _write_candidate_runtime_snapshot(
                output_dir,
                ranked_record,
                output_detail=output_detail,
            )
        generation_top3 = sorted(
            [
                item
                for item in generation_records
                if item.get("generation_rank") in {1, 2, 3}
            ],
            key=lambda item: int(item.get("generation_rank") or 99),
        )
        # The persistent archive is the source of evolutionary parents. Feed
        # every successful candidate so a generation-rank-4 candidate can
        # still displace an older global elite.
        generation_tracker_records = [
            item
            for item in generation_records
            if item.get("training_success")
            and math.isfinite(_metric_value(item, population.objective))
        ]
        global_best_tracker_update = global_best_tracker.update_many(
            generation_tracker_records
        )
        global_best_tracker.save(global_best_tracker_path)
        global_best_tracker_after = global_best_tracker.audit_snapshot()
        global_parent_top3, _ = global_best_tracker.top_k(formal_parent_count)
        global_parent_top3 = _hydrate_global_tracker_selection(
            global_parent_top3, [*all_records, *generation_records]
        )
        global_ranked_top3 = [
            {
                "candidate_id": item.get("spec_id"),
                "global_rank": rank,
                "source_generation": item.get(
                    "generation_index", item.get("generation")
                ),
                "generation_rank": item.get("generation_rank"),
                "generation_metrics": item.get("generation_metrics") or item.get("metrics") or {},
                "population_role": item.get("population_role"),
            }
            for rank, item in enumerate(global_parent_top3, start=1)
        ]
        global_top3_archive_update = summarize_global_top3_archive(
            global_best_tracker_before,
            global_best_tracker_after,
        )
        candidate_posteriors: list[dict[str, Any]] = []
        generation_posterior_payload: dict[str, Any] = {}
        if current_run_posterior_enabled:
            for record in generation_records:
                candidate_posterior = build_candidate_posterior(
                    record,
                    pde_id=problem_id,
                    run_id=posterior_memory.run_id,
                )
                posterior_memory.add_candidate_posterior(
                    pde_id=problem_id,
                    run_id=posterior_memory.run_id,
                    posterior=candidate_posterior,
                )
                candidate_posteriors.append(candidate_posterior.to_dict())
            generation_posterior = build_generation_posterior(
                candidate_posteriors,
                pde_id=problem_id,
                run_id=posterior_memory.run_id,
                generation=generation,
                ranking_metric="global_reference_mse",
            )
            posterior_memory.add_generation_posterior(
                pde_id=problem_id,
                run_id=posterior_memory.run_id,
                posterior=generation_posterior,
            )
            generation_posterior_payload = generation_posterior.to_dict()
        generation_rank1 = (
            generation_top3[0] if generation_top3 else {}
        )
        generation_rank1_metrics = dict(
            generation_rank1.get("generation_metrics")
            or generation_rank1.get("metrics")
            or {}
        )
        generation_rank1_l2re = generation_rank1_metrics.get("l2re")
        if generation_rank1_l2re is None:
            generation_rank1_l2re = generation_rank1_metrics.get(
                "relative_l2_error"
            )
        _console_status(
            config,
            "[Search] "
            + json.dumps(
                {
                    "generation": generation,
                    "best_mse": _finite_metric(
                        generation_rank1_metrics.get("mse")
                    ),
                    "best_relative_l2_error": _finite_metric(
                        generation_rank1_l2re
                    ),
                },
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            ),
        )
        for record in generation_records:
            all_records.append(record)
            _progress(config, {"event": "candidate_complete", "generation": generation, "spec_id": record.get("spec_id"), "status": record.get("status"), "training_success": record.get("training_success"), "failure_reason": record.get("failure_reason")})
        population.population = []
        population.update(generation_records)
        if execution_feedback_enabled:
            reflection_result = reflection_agent.run(
                {
                    "provider": provider,
                    "generation": generation,
                    "records": generation_records,
                }
            )
            reflection = dict(reflection_result.data.get("reflection") or {})
            reflections.append(reflection)
        else:
            reflection = {}
        for record in generation_records:
            record["reflection"] = reflection
            record_writer.append(compact_candidate_index(record))
        search_state, generation_summary = summarize_low_fidelity_generation(
            generation_records,
            previous_state=search_state,
            generation_id=generation,
            generation_mode=generation_mode_name,
            contract=experiment_contract,
            archive_update=global_top3_archive_update,
        )
        generation_summary["force_repeated_generation_failure"] = bool(
            training_recovery_exhausted
            or len(generation_top3) < formal_parent_count
        )
        (
            search_state,
            generation_summary,
            low_fidelity_termination_reason,
        ) = advance_search_after_generation(
            search_state,
            generation_summary,
            generation_id=generation,
            generation_mode=generation_mode_name,
            contract=experiment_contract,
        )
        generation_feedback = _build_generation_feedback(
            generation=generation,
            reflection=reflection,
            records=generation_records,
            population_summary=population.summary(),
            objective=PRIMARY_RANKING_METRIC,
            stagnation_state=generation_summary.get("stagnation_check") or {},
        )
        generation_feedback["stagnated"] = bool(
            (generation_summary.get("stagnation_check") or {}).get(
                "stagnation_detected"
            )
        )
        generation_feedback["next_generation_mode"] = (
            search_state.get("next_generation_mode")
        )
        if current_run_posterior_enabled:
            posterior = _refresh_posterior_context(
                posterior, generation_feedback, posterior_limit
            )
        independent_boundary_audits = [
            dict(
                ((trace.get("prompt_statistics") or {}).get(
                    "evolutionary_inheritance_audit"
                ) or {})
            )
            for trace in (variation_result.get("traces") or [])
            if (
                ((trace.get("prompt_statistics") or {}).get(
                    "evolutionary_inheritance_audit"
                ) or {}).get("mode")
                == "budget_matched_independent_search"
            )
        ]
        generation_result = {
            "generation": generation,
            "generation_index": generation,
            **generation_summary,
            "generation_iteration_recommendation": generation_recommended_iterations,
            "iteration_recommendation_is_advisory": False,
            "generation_kind": "low_fidelity",
            "ranking_fidelity": ranking_fidelity,
            "experiment_contract_version": EXPERIMENT_CONTRACT_VERSION,
            "search_stage": stage,
            "generation_mode": generation_mode_name,
            "generation_policy": generation_mode,
            "records": generation_records,
            "num_candidates": len(generation_records),
            "num_training_successes": sum(bool(item.get("training_success")) for item in generation_records),
            "population": population.summary(),
            "reflection": reflection,
            "input_feedback": input_feedback,
            "feedback_for_next_generation": generation_feedback,
            "variation": {
                "generation_mode": variation_result.get("generation_mode"),
                "feedback_analysis": variation_result.get("feedback_analysis"),
                "strategy_summary": variation_result.get("strategy_summary"),
                "rejected_children": variation_result.get("rejected_children") or [],
                "proposal_count": variation_result.get("proposal_count"),
                "accepted_count": variation_result.get("accepted_count"),
                "refill_rounds": variation_result.get("refill_rounds", 0),
                "training_recovery_count": variation_result.get(
                    "training_recovery_count", 0
                ),
                "requested_llm_children_count": target_new_children,
                "accepted_llm_children_count": len(generated_children),
                "selected_for_training_count": len(current_children),
                "selected_for_training_ids": [
                    item.get("candidate_id") for item in current_children
                ],
                "candidate_audit": variation_result.get("candidate_audit") or [],
                **(
                    {"diversity_retries": variation_result.get("diversity_retries") or []}
                    if generation > 0
                    else {}
                ),
                "resolved_design_context": variation_result.get("resolved_design_context") or {},
            },
            "generation_training": {
                "mode": "enforced_low_fidelity_budget",
                "generation_iteration_recommendation": generation_recommended_iterations,
                "iteration_recommendation_is_advisory": False,
                "training_iterations": int(budget["required_total_iterations"]),
                "ranking_fidelity": ranking_fidelity,
                "attempted_candidate_ids": [item.get("spec_id") for item in generation_records],
                "successful_candidate_ids": [
                    item.get("spec_id")
                    for item in generation_records
                    if item.get("training_success") and math.isfinite(_metric_value(item, population.objective))
                ],
                "failed_candidate_ids": [
                    item.get("spec_id")
                    for item in generation_records
                    if not item.get("training_success") or not math.isfinite(_metric_value(item, population.objective))
                ],
                "llm_recovery_candidate_count": training_recovery_count,
                "generation_rankings": [
                    {"candidate_id": item.get("spec_id"), "generation_rank": item.get("generation_rank")}
                    for item in generation_top3
                ],
            },
            "generation_audit": {
                "ablation": ablation["mode"],
                "execution_performed": True,
                "fitness_used_for_selection": True,
                "execution_metrics_exposed_to_llm": execution_feedback_enabled,
                "five_candidate_roles_enabled": bool(generation > 0),
                "fitness_based_parent_selection": evolutionary_inheritance_enabled,
                "evolutionary_inheritance_enabled": evolutionary_inheritance_enabled,
                "parent_selection_for_generation_enabled": bool(
                    generation > 0 and evolutionary_inheritance_enabled
                ),
                "parent_algorithm_specs_exposed_to_llm": bool(
                    generation > 0
                    and evolutionary_inheritance_enabled
                    and parent_candidates
                ),
                "independent_proposal_roles_enabled": bool(
                    generation > 0 and not evolutionary_inheritance_enabled
                ),
                "independent_prompt_boundary_enforced": bool(
                    generation > 0 and not evolutionary_inheritance_enabled
                ),
                "independent_prompt_boundary_audit_count": len(
                    independent_boundary_audits
                ),
                "independent_prompt_boundary_passed": bool(
                    independent_boundary_audits
                    and all(
                        bool(item.get("passed"))
                        for item in independent_boundary_audits
                    )
                ) if generation > 0 and not evolutionary_inheritance_enabled else None,
                "execution_feedback_enabled": execution_feedback_enabled,
                "candidate_execution_performed": True,
                "true_fitness_ranking_retained": True,
                "true_fitness_parent_selection_retained": evolutionary_inheritance_enabled,
                "execution_outcomes_exposed_to_llm": execution_feedback_enabled,
                "reflection_exposed_to_llm": execution_feedback_enabled,
                "current_run_posterior_updated": current_run_posterior_enabled,
                "parent_order_rank_signal_hidden": bool(
                    generation > 0 and not execution_feedback_enabled
                ),
                "llm_role_count": len(NEW_CANDIDATE_ROLE_PLAN) if generation > 0 else len(GENERATION_ZERO_ROLE_PLAN),
                "maximum_generations": generations,
                "generation_count_is_dynamic": True,
                "low_fidelity_generation_count": search_state[
                    "low_fidelity_generation_count"
                ],
                "generation_mode": generation_mode_name,
                "generation_summary": json_safe(generation_summary),
                "search_state": json_safe(search_state),
                "parent_top3_ids": previous_parent_ids,
                "previous_parent_ids": previous_parent_ids,
                "generated_candidate_metadata": [
                    deepcopy(item.get("proposal_metadata") or {})
                    for item in current_children
                ],
                "generation_top3_ids": [
                    item.get("spec_id") for item in generation_top3
                ],
                "global_top3_ids": [
                    item.get("spec_id") for item in global_parent_top3
                ],
                "global_best_tracker_before": global_best_tracker_before,
                "global_best_tracker_after": global_best_tracker_after,
                "global_best_tracker_update": global_best_tracker_update,
                "global_best_tracker_input_scope": "generation_successful_candidates",
                "global_best_tracker_input_ids": [
                    item.get("spec_id") for item in generation_tracker_records
                ],
                "global_best_tracker_is_population_member": False,
                "current_population_ids": [item.get("spec_id") for item in generation_records],
                "training_seed": generation_seed,
                "next_generation_parent_ids": [
                    item.get("spec_id") for item in global_parent_top3
                ],
                "historical_checkpoint_reused": False,
                "historical_metrics_reused": False,
                "screening_enabled": False,
                "promotion_enabled": False,
                "checkpoint_continuation_enabled": False,
                "llm_call_count": int(variation_result.get("llm_calls") or 0),
                "llm_max_concurrency": llm_max_concurrency,
                "llm_training_pipeline_enabled": True,
                "llm_training_pipeline_mode": (
                    "single_llm_request_gpu_overlap"
                    if llm_max_concurrency == 1
                    else "parallel_generation_overlap"
                ),
                "llm_training_overlapped_candidate_count": len(
                    streamed_candidate_ids
                ),
                "llm_sequentially_interleaved_candidate_count": 0,
                "single_gpu_training_concurrency": 1,
                "llm_generation_quality": _summarize_llm_generation_quality(
                    variation_result.get("traces") or []
                ),
                "initial_llm_call_count": (
                    len(NEW_CANDIDATE_ROLE_PLAN)
                    if generation > 0
                    else len(GENERATION_ZERO_ROLE_PLAN)
                ),
                "role_plan": list(variation_result.get("role_plan") or []),
                "generated_valid_candidate_count": len(generated_children),
                "selected_training_candidate_count": len(current_children),
                "successful_new_candidates": (
                    len(generated_children) if generation > 0 else len(current_children)
                ),
                "duplicate_count": int(variation_result.get("duplicate_count") or 0),
                "retry_count": int(variation_result.get("retry_count") or 0),
                "target_candidate_count": target_population_size,
                "minimum_candidate_count": minimum_generated_candidates,
                "minimum_generated_candidate_count": minimum_generated_candidates,
                "minimum_successful_candidate_count": MINIMUM_SUCCESSFUL_CANDIDATES,
                "actual_candidate_count": len(current_children),
                "candidate_shortfall": successful_candidate_shortfall,
                "successful_candidate_shortfall": successful_candidate_shortfall,
                "exact_duplicate_count": int(variation_result.get("duplicate_count") or 0),
                "generation_continued_with_shortfall": successful_candidate_shortfall > 0,
            },
            "candidate_count": len(generation_records),
            "generation_budget_attempted": len(generation_records),
            "generation_budget_succeeded": sum(
                bool(item.get("training_success")) and math.isfinite(_metric_value(item, population.objective))
                for item in generation_records
            ),
            "generation_ranked_top3_ids": [
                item.get("spec_id") for item in generation_top3
            ],
            "generation_ranked_top3": [
                {
                    "candidate_id": item.get("spec_id"),
                    "generation_rank": item.get("generation_rank"),
                    "generation_metrics": item.get("generation_metrics") or {},
                    "population_role": item.get("population_role"),
                }
                for item in generation_top3
            ],
            # Public per-generation TOP-3 is the cumulative global archive.
            "ranked_top3": global_ranked_top3,
            "global_top3": global_ranked_top3,
            "ranked_top3_ids": [
                item.get("spec_id") for item in global_parent_top3
            ],
            "generation_started_at": generation_started_at,
            "generation_ended_at": datetime.now().isoformat(timespec="seconds"),
            "generation_runtime_seconds": time.perf_counter() - generation_timer,
            "generation_posterior": generation_posterior_payload,
            "candidate_training_time_seconds": sum(
                float((item.get("training_report") or {}).get("training_time") or 0.0)
                for item in generation_records
            ),
        }
        top3_key = "final_top_3" if ranking_fidelity == "full_budget" else "proxy_top_3"
        rank1_key = "final_rank_1" if ranking_fidelity == "full_budget" else "proxy_rank_1"
        generation_result[top3_key] = generation_result["global_top3"]
        generation_result[rank1_key] = (
            generation_result["global_top3"][0]
            if generation_result["global_top3"]
            else None
        )
        generation_results.append(generation_result)
        _write_json(
            output_dir / "generation_results.json",
            [_compact_generation_result(item) for item in generation_results],
        )
        _write_json(output_dir / "population.json", population.summary())
        _write_json(
            output_dir / "llm_trace.json",
            _compact_llm_trace(llm_trace, output_detail=output_detail),
        )
        _persist_search_resume_state(
            state_path=state_path,
            search_state=search_state,
            generation_results=generation_results,
            all_records=all_records,
            reflections=reflections,
            generation_feedback=generation_feedback,
            global_top3=global_parent_top3,
        )
        if low_fidelity_termination_reason is not None:
            break

    successful = [record for record in all_records if record.get("training_success")]
    test_execution_limited = bool(
        config.get("_test_mode") and execution_generations < generations
    )
    if not test_execution_limited and not search_state.get("termination_reason"):
        search_state = transition_search_phase(
            search_state, "finalist_selection"
        )
        search_state["termination_reason"] = "max_low_fidelity_generations_reached"
    search_termination_reason = (
        "test_execution_limit_reached"
        if test_execution_limited and not search_state.get("termination_reason")
        else str(search_state["termination_reason"])
    )

    final_selection: list[dict[str, Any]] = []
    final_selection_audit: dict[str, Any] = {}
    final_high_fidelity_records: list[dict[str, Any]] = []
    final_evaluation_path = (
        output_dir / "final_high_fidelity_evaluation.json"
    )
    if not test_execution_limited:
        selection_path = output_dir / "final_high_fidelity_selection.json"
        requested_final_count = int(
            experiment_contract["final_selection"]["candidate_count"]
        )
        if search_state.get("search_phase") == "normal_search":
            search_state = transition_search_phase(
                search_state, "finalist_selection"
            )
        if search_state.get("final_high_fidelity_candidate_ids") and selection_path.exists():
            selection_payload = json.loads(selection_path.read_text(encoding="utf-8"))
            final_selection = _hydrate_persisted_final_selection(
                selection_payload,
                all_records,
            )
            final_selection_audit = dict(selection_payload.get("selection_audit") or {})
            final_selection_audit[
                "selection_source"
            ] = "persisted_global_best_tracker_selection"
        else:
            final_selection, final_selection_audit = global_best_tracker.top_k(
                requested_final_count
            )
            search_state["final_high_fidelity_candidate_ids"] = [
                str(item.get("spec_id")) for item in final_selection
            ]
        rebuilt_selection, rebuilt_audit = (
            select_distinct_history_top_candidates(
                all_records,
                candidate_count=requested_final_count,
            )
        )
        tracker_consistency = {
            "passed": (
                _selection_signature(final_selection)
                == _selection_signature(rebuilt_selection)
            ),
            "tracker_selection_signature": _selection_signature(
                final_selection
            ),
            "global_top3_rebuild_signature": _selection_signature(
                rebuilt_selection
            ),
            "rebuild_audit": rebuilt_audit,
            "purpose": "bounded_global_top3_end_of_search_consistency_audit",
        }
        final_selection_audit["consistency_audit"] = tracker_consistency
        if not tracker_consistency["passed"]:
            raise RuntimeError(
                "GlobalBestTracker does not match the global successful-candidate rebuild"
            )
        _write_json(
            selection_path,
            {
                "stage": "final_high_fidelity_evaluation",
                "selected_candidate_ids": [
                    item.get("spec_id") for item in final_selection
                ],
                "selected_candidates": [
                    _compact_low_fidelity_selection_record(item)
                    for item in final_selection
                ],
                "selection_audit": _compact_selection_audit(
                    final_selection_audit
                ),
                "global_best_tracker": "global_best_tracker.json",
            },
        )
        if search_state.get("search_phase") == "finalist_selection":
            search_state = transition_search_phase(
                search_state, "high_fidelity_running"
            )
        _persist_search_resume_state(
            state_path=state_path,
            search_state=search_state,
            generation_results=generation_results,
            all_records=all_records,
            reflections=reflections,
            generation_feedback=generation_feedback,
            global_top3=global_parent_top3,
        )
        if final_evaluation_path.exists():
            final_high_fidelity_records = _hydrate_persisted_high_fidelity_records(
                json.loads(
                    final_evaluation_path.read_text(encoding="utf-8")
                ),
                load_runtime_candidate_records(output_dir),
            )
        completed_source_ids = {
            str(item.get("source_low_fidelity_candidate_id"))
            for item in final_high_fidelity_records
        }
        high_iterations = int(
            experiment_contract["search_control"]["high_fidelity_iterations"]
        )
        for selected in final_selection:
            source_id = str(selected.get("spec_id"))
            if source_id in completed_source_ids:
                continue
            high_spec, high_scaling_audit = (
                adapt_algorithm_spec_to_generation_budget(
                    deepcopy(
                        selected.get("search_algorithm_spec")
                        or selected.get("normalized_algorithm_spec")
                        or {}
                    ),
                    high_iterations,
                    min_lbfgs_iterations=int(
                        scaling_config["min_lbfgs_iterations"]
                    ),
                    min_adaptive_sampling_events=int(
                        scaling_config["min_adaptive_sampling_events"]
                    ),
                    preserve_enabled_training_stages=bool(
                        scaling_config["preserve_enabled_training_stages"]
                    ),
                    enforce_full_iterations=True,
                    source_fidelity="low",
                    target_fidelity="high",
                    adaptive_policy_source_spec=deepcopy(
                        selected.get("normalized_algorithm_spec")
                        or selected.get("search_algorithm_spec")
                        or {}
                    ),
                    source_optimizer_step_budget=int(
                        (selected.get("budget") or {}).get(
                            "required_total_iterations", LOW_FIDELITY_ITERATIONS
                        )
                    ),
                    source_sampling_budget=(selected.get("budget") or {}).get(
                        "maximum_sampling_points"
                    ),
                    target_sampling_budget=_high_fidelity_budget(
                        config, experiment_contract=experiment_contract
                    ).get("maximum_sampling_points"),
                    experiment_contract=experiment_contract,
                )
            )
            high_spec = _enforce_high_fidelity_validation_and_checkpoint(
                high_spec,
                high_iterations=high_iterations,
            )
            high_scaling_audit = deepcopy(high_scaling_audit)
            high_scaling_audit["model_selection_safety_policy"] = {
                "reference_mse_is_diagnostic_only": True,
                "physics_validation_forced": True,
                "second_order_physics_guard_forced": True,
                "second_order_degradation_ratio": 1.1,
                "pinnacle_best_train_loss_forced": True,
                "final_model_policy": "best_train_loss",
            }
            high_budget = _high_fidelity_budget(
                config,
                experiment_contract=experiment_contract,
            )
            low_rank = int(selected["low_fidelity_rank"])
            high_seed = int(selected.get("training_seed", seed))
            set_global_seed(high_seed)
            final_candidate_id = f"hf_rank{low_rank}_{source_id}"
            high_record = _evaluate_candidate(
                candidate_id=final_candidate_id,
                spec=high_spec,
                generation=int(selected.get("generation_index") or 0),
                operation="final_high_fidelity_evaluation",
                problem=problem,
                problem_id=problem_id,
                budget=high_budget,
                capabilities=capabilities,
                builder=builder,
                device=str(config.get("device") or "cpu"),
                seed=high_seed,
                train_candidate=True,
                checkpoint_root=(
                    output_dir
                    / "final_high_fidelity_evaluation"
                    / final_candidate_id
                    if output_detail == "diagnostic"
                    else None
                ),
                generation_feedback=None,
                training_iteration_limit=None,
                resume_checkpoint=None,
                history_interval=resolve_history_interval(
                    high_iterations,
                    output_detail=output_detail,
                    configured_interval=config.get("training_history_interval"),
                ),
                reference_mse_interval=max(1, high_iterations // 10),
                reference_mse_checkpoints=_fidelity_checkpoint_iterations(
                    high_spec,
                    high_iterations,
                ),
                design_algorithm_spec=deepcopy(
                    selected.get("search_algorithm_spec")
                    or selected.get("normalized_algorithm_spec")
                    or {}
                ),
                runtime_scaling_audit=deepcopy(
                    high_scaling_audit.get("adaptive_policy_scaling") or {}
                ),
                trainer_capabilities_snapshot=trainer_capabilities_snapshot,
            )
            high_training_report = dict(
                high_record.get("training_report") or {}
            )
            high_selected_mse = _finite_metric(
                (high_record.get("metrics") or {}).get("mse")
            )
            high_last_mse = _finite_metric(
                high_training_report.get("last_checkpoint_mse")
            )
            high_model_selection_audit = {
                "policy": high_training_report.get("final_model_policy"),
                "reference_mse_model_selection_enabled": bool(
                    high_training_report.get(
                        "reference_mse_model_selection_enabled"
                    )
                ),
                "best_reference_mse_checkpoint": None,
                "best_training_loss_checkpoint": deepcopy(
                    high_training_report.get("best_training_loss_checkpoint")
                ),
                "selected_checkpoint_source": high_training_report.get(
                    "selected_checkpoint_source"
                ),
                "selected_checkpoint_step": high_training_report.get(
                    "selected_checkpoint_step"
                ),
                "selected_checkpoint_metric": deepcopy(
                    high_training_report.get("selected_checkpoint_metric")
                ),
                "selected_model_mse": high_selected_mse,
                "last_checkpoint_mse": high_last_mse,
                "last_step_degradation_prevented": bool(
                    high_selected_mse is not None
                    and high_last_mse is not None
                    and high_last_mse > high_selected_mse
                ),
                "last_to_selected_mse_ratio": (
                    high_last_mse / high_selected_mse
                    if high_last_mse is not None
                    and high_selected_mse is not None
                    and high_selected_mse > 0.0
                    else None
                ),
            }
            high_record.update(
                {
                    "fidelity": "high",
                    "evaluation_stage": "final_high_fidelity_evaluation",
                    "source_low_fidelity_candidate_id": source_id,
                    "source_generation_id": int(
                        selected.get("generation_index") or 0
                    ),
                    "training_iterations": high_iterations,
                    "fresh_initialization": True,
                    "resumed_from_checkpoint": False,
                    "low_fidelity_checkpoint_reused": False,
                    "optimizer_state_reused": False,
                    "sampler_state_reused": False,
                    "low_fidelity_rank": low_rank,
                    "high_fidelity_rank": None,
                    "rank_change": None,
                    "source_low_fidelity_proxy_mse": selected.get(
                        "low_fidelity_aggregate_proxy_mse"
                    ),
                    "training_seed": high_seed,
                    "model_initialization_seed": high_seed,
                    "sampling_seed": high_seed,
                    "evaluation_seed": int(
                        config.get("evaluation_seed") or 0
                    ),
                    "search_algorithm_spec": deepcopy(
                        selected.get("search_algorithm_spec")
                        or selected.get("normalized_algorithm_spec")
                        or {}
                    ),
                    "optimization_budget_audit": high_scaling_audit,
                    "high_fidelity_model_selection_audit": (
                        high_model_selection_audit
                    ),
                    "experiment_contract_version": EXPERIMENT_CONTRACT_VERSION,
                }
            )
            high_record["cross_fidelity_audit"] = (
                _build_cross_fidelity_audit(
                    selected,
                    high_record,
                    high_training_report,
                )
            )
            _write_candidate_runtime_snapshot(
                output_dir,
                high_record,
                output_detail=output_detail,
            )
            final_high_fidelity_records.append(high_record)
            search_state[
                "completed_final_high_fidelity_candidate_ids"
            ] = [
                str(item.get("source_low_fidelity_candidate_id"))
                for item in final_high_fidelity_records
            ]
            _write_json(
                final_evaluation_path,
                {
                    "stage": "final_high_fidelity_evaluation",
                    "status": "in_progress",
                    "selection_audit": _compact_selection_audit(
                        final_selection_audit
                    ),
                    "records": [
                        _compact_high_fidelity_record(item)
                        for item in final_high_fidelity_records
                    ],
                },
            )
            _persist_search_resume_state(
                state_path=state_path,
                search_state=search_state,
                generation_results=generation_results,
                all_records=all_records,
                reflections=reflections,
                generation_feedback=generation_feedback,
                global_top3=global_parent_top3,
            )
        ranked_high = sorted(
            [
                item
                for item in final_high_fidelity_records
                if item.get("training_success")
                and math.isfinite(
                    _metric_value(item, PRIMARY_RANKING_METRIC)
                )
            ],
            key=lambda item: _metric_value(
                item, PRIMARY_RANKING_METRIC
            ),
        )
        high_rank_by_id = {
            str(item.get("source_low_fidelity_candidate_id")): rank
            for rank, item in enumerate(ranked_high, start=1)
        }
        updated_high_records: list[dict[str, Any]] = []
        for record in final_high_fidelity_records:
            item = dict(record)
            high_rank = high_rank_by_id.get(
                str(item.get("source_low_fidelity_candidate_id"))
            )
            item["high_fidelity_rank"] = high_rank
            item["rank_change"] = (
                int(item["low_fidelity_rank"]) - int(high_rank)
                if high_rank is not None
                else None
            )
            updated_high_records.append(item)
        final_high_fidelity_records = updated_high_records
        for high_record in final_high_fidelity_records:
            _write_candidate_runtime_snapshot(
                output_dir,
                high_record,
                output_detail=output_detail,
            )
        ranked_high = sorted(
            [
                item
                for item in final_high_fidelity_records
                if item.get("high_fidelity_rank") is not None
            ],
            key=lambda item: int(item["high_fidelity_rank"]),
        )
        if search_state.get("search_phase") == "high_fidelity_running":
            search_state = transition_search_phase(
                search_state, "completed"
            )
        _write_json(
            final_evaluation_path,
            {
                "stage": "final_high_fidelity_evaluation",
                "status": "completed",
                "candidate_source": "global_low_fidelity_top3",
                "ranking_scope": "global_high_fidelity_top3",
                "termination_reason": search_termination_reason,
                "selection_audit": _compact_selection_audit(
                    final_selection_audit
                ),
                "final_rank_1": (
                    _compact_high_fidelity_record(ranked_high[0])
                    if ranked_high
                    else None
                ),
                "final_top_3": [
                    _compact_high_fidelity_record(item)
                    for item in ranked_high[:3]
                ],
                "records": [
                    _compact_high_fidelity_record(item)
                    for item in final_high_fidelity_records
                ],
            },
        )
        _persist_search_resume_state(
            state_path=state_path,
            search_state=search_state,
            generation_results=generation_results,
            all_records=all_records,
            reflections=reflections,
            generation_feedback=generation_feedback,
            global_top3=global_parent_top3,
        )
    else:
        ranked_high = []

    result = {
        "status": "completed",
        "search_status": "completed",
        "pipeline": "open_algorithm_spec",
        "problem_id": problem_id,
        "ablation": deepcopy(ablation),
        "config": config,
        "experiment_contract_version": EXPERIMENT_CONTRACT_VERSION,
        "generation_iteration_recommendations": dict(GENERATION_ITERATION_RECOMMENDATIONS),
        "regular_generation_recommended_iterations": REGULAR_GENERATION_RECOMMENDED_ITERATIONS,
        "high_fidelity_iterations": int(
            search_control["high_fidelity_iterations"]
        ),
        "iteration_recommendations_are_advisory": False,
        "maximum_generations": generations,
        "actual_generations_completed": len(generation_results),
        "low_fidelity_generation_count": search_state[
            "low_fidelity_generation_count"
        ],
        "search_termination_reason": search_termination_reason,
        "termination_reason": search_termination_reason,
        "builder_capabilities": capabilities,
        "pinnacle_profile": pinnacle_profile,
        "generation_results": generation_results,
        "population": population.summary(),
        "reflections": reflections,
        "final_low_fidelity_feedback": generation_feedback,
        "records": all_records,
        "final_high_fidelity_evaluation": {
            "candidate_source": "global_low_fidelity_top3",
            "ranking_scope": "global_high_fidelity_top3",
            "selection_audit": final_selection_audit,
            "records": final_high_fidelity_records,
        },
        "final_rank_1": ranked_high[0] if ranked_high else None,
        "final_top_3": ranked_high[:3],
        "global_best_tracker": global_best_tracker.snapshot(),
        "best_candidate": ranked_high[0] if ranked_high else None,
        "evaluation_policy": {
            "primary_metric": PRIMARY_RANKING_METRIC,
            "pinnacle_metrics": [
                "mae", "mse", "merr", "l1re", "l2re", "crmse",
                "frmse_low", "frmse_mid", "frmse_high", "temporal_l2re",
            ],
            "mode": (
                "adaptive_up_to_ten_generation_multifidelity_evolution"
                if evolutionary_inheritance_enabled
                else "adaptive_up_to_ten_generation_multifidelity_independent_search"
            ),
            "population_size": population.maximum_size,
            "generation_zero_proposal_count": len(GENERATION_ZERO_ROLE_PLAN),
            "generation_iteration_recommendations": dict(GENERATION_ITERATION_RECOMMENDATIONS),
            "regular_generation_recommended_iterations": REGULAR_GENERATION_RECOMMENDED_ITERATIONS,
            "high_fidelity_iterations": int(
                search_control["high_fidelity_iterations"]
            ),
            "iteration_recommendations_are_advisory": False,
            "maximum_generations": generations,
            "actual_generations_completed": len(generation_results),
            "search_termination_reason": search_termination_reason,
            "formal_parent_count": formal_parent_count,
            "screening_enabled": False,
            "promotion_enabled": False,
            "checkpoint_continuation_enabled": False,
            "run_posterior_isolated": True,
            "old_posterior_reused": False,
            "posterior_scope": {
                "pde_id": posterior_memory.pde_id,
                "run_id": posterior_memory.run_id,
            },
            "baseline_iterations": HIGH_FIDELITY_ITERATIONS,
            "formal_baseline_comparison_stage": "final_high_fidelity_evaluation",
            "ablation_mode": ablation["mode"],
            "prior_knowledge_enabled": knowledge_enabled,
            "posterior_knowledge_enabled": posterior_enabled,
            "execution_feedback_enabled": execution_feedback_enabled,
            "current_run_posterior_updates_enabled": current_run_posterior_enabled,
            "evolutionary_inheritance_enabled": evolutionary_inheritance_enabled,
            "parent_algorithm_specs_exposed_to_llm": bool(
                evolutionary_inheritance_enabled
            ),
            "global_archive_enabled": True,
            "independent_proposal_role_plan": (
                list(INDEPENDENT_PROPOSAL_ROLE_PLAN)
                if not evolutionary_inheritance_enabled
                else []
            ),
        },
        "search_audit": {
            "passed": bool(successful),
            "num_llm_calls": total_calls,
            "num_candidates": len(all_records),
            "num_training_successes": len(successful),
            "num_feedback_expected_candidates": sum(bool(item.get("feedback_required")) for item in all_records),
            "num_feedback_acknowledged_candidates": sum(bool(item.get("feedback_acknowledged")) for item in all_records),
            "maximum_candidate_trainings_per_run": (
                GENERATION_ZERO_TRAINING_CANDIDATES
                + (MAX_GENERATIONS - 1) * EVOLUTION_TARGET_CANDIDATES
            ),
            "maximum_optimizer_step_budget": (
                (
                    GENERATION_ZERO_TRAINING_CANDIDATES
                    + (MAX_GENERATIONS - 1) * EVOLUTION_TARGET_CANDIDATES
                )
                * LOW_FIDELITY_ITERATIONS
                + 3 * HIGH_FIDELITY_ITERATIONS
            ),
            "budget_matched_to_full": bool(
                ablation["mode"] == NO_EVOLUTIONARY_SEARCH_ABLATION_MODE
            ),
            "actual_candidate_trainings": len(all_records),
            "ablation_contract": deepcopy(ablation),
            "prior_context_item_count": len(prior),
            "posterior_memory_entry_count": (
                len(posterior_memory.candidate_records)
                + len(posterior_memory.generation_summaries)
                if current_run_posterior_enabled
                else 0
            ),
            "total_candidate_training_seconds": sum(
                float((item.get("training_report") or {}).get("training_time") or 0.0)
                for item in all_records
            ),
            "total_gpu_time_seconds": (
                sum(
                    float((item.get("training_report") or {}).get("training_time") or 0.0)
                    for item in all_records
                )
                if str(config.get("device") or "cpu").startswith("cuda")
                else None
            ),
        },
    }
    _write_json(
        output_dir / "open_search_summary.json",
        _compact_search_result(result),
    )
    return result


def _append_llm_traces(
    destination: list[dict[str, Any]],
    traces: list[dict[str, Any]],
    *,
    generation: int,
    generation_mode: str,
    feedback_source_generation: Any,
    provider_name: str,
    model_name: str,
) -> None:
    for trace in traces:
        trace_usage = trace.get("token_usage") or {}
        destination.append(
            {
                "generation": generation,
                "generation_mode": generation_mode,
                "feedback_source_generation": feedback_source_generation,
                "success": trace.get("success"),
                "raw_response": redact_secret(str(trace.get("raw_response") or "")),
                "parse_errors": trace.get("parse_errors") or [],
                "token_usage": trace_usage,
                "prompt_tokens": int(
                    trace_usage.get("prompt_tokens") or trace.get("prompt_tokens") or 0
                ),
                "completion_tokens": int(
                    trace_usage.get("completion_tokens")
                    or trace.get("completion_tokens")
                    or 0
                ),
                "total_tokens": int(trace_usage.get("total_tokens") or 0),
                "provider": trace.get("provider") or provider_name,
                "provider_success": trace.get("provider_success"),
                "provider_error": redact_secret(str(trace.get("provider_error") or "")),
                "model": trace.get("model") or model_name,
                "model_version": trace.get("model_version"),
                "latency_seconds": trace.get("latency_seconds"),
                "proposal_count": trace.get("proposal_count"),
                "candidate_id_match": trace.get("candidate_id_match"),
                "candidate_id": trace.get("candidate_id"),
                "candidate_slot": trace.get("candidate_slot"),
                "candidate_role": trace.get("candidate_role"),
                "attempt": trace.get("attempt"),
                "retry_count": max(0, int(trace.get("attempt") or 1) - 1),
                "temperature": trace.get("temperature"),
                "requested_max_tokens": trace.get("requested_max_tokens"),
                "finish_reason": trace.get("finish_reason"),
                "response_truncated": bool(trace.get("response_truncated")),
                "parse_success": bool(trace.get("parse_success")),
                "validation_success": bool(trace.get("validation_success")),
                "validation_errors": trace.get("validation_errors") or [],
                "transport_repairs": trace.get("transport_repairs") or [],
                "prompt_statistics": deepcopy(
                    trace.get("prompt_statistics") or {}
                ),
                "error": trace.get("error"),
            }
        )


def _summarize_llm_generation_quality(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Return rejection and transport-repair metrics for one generation."""

    total_calls = len(traces)
    accepted_calls = sum(bool(trace.get("success")) for trace in traces)
    rejected_calls = total_calls - accepted_calls
    parse_failures = sum(not bool(trace.get("parse_success")) for trace in traces)
    provider_failures = sum(trace.get("provider_success") is False for trace in traces)
    validation_failures = sum(
        bool(trace.get("parse_success")) and not bool(trace.get("validation_success"))
        for trace in traces
    )
    duplicate_rejections = sum(
        "duplicate" in str(trace.get("error") or "").casefold()
        for trace in traces
    )
    repair_counts: dict[str, int] = {}
    repaired_calls = 0
    for trace in traces:
        repairs = list(trace.get("transport_repairs") or [])
        if repairs:
            repaired_calls += 1
        for repair in repairs:
            path = str((repair or {}).get("path") or "unknown")
            repair_counts[path] = repair_counts.get(path, 0) + 1
    return {
        "total_calls": total_calls,
        "accepted_calls": accepted_calls,
        "rejected_calls": rejected_calls,
        "rejection_rate": rejected_calls / total_calls if total_calls else 0.0,
        "parse_failures": parse_failures,
        "provider_failures": provider_failures,
        "validation_failures": validation_failures,
        "duplicate_rejections": duplicate_rejections,
        "transport_repaired_calls": repaired_calls,
        "transport_repair_count": sum(repair_counts.values()),
        "transport_repair_paths": repair_counts,
    }


def _successful_candidate_count(records: list[dict[str, Any]], objective: str) -> int:
    return sum(
        bool(item.get("training_success"))
        and math.isfinite(_metric_value(item, objective))
        for item in records
    )


def _compact_failure_summaries(
    failures: list[dict[str, Any]], *, limit: int = 8
) -> list[dict[str, Any]]:
    """Summarize retries without repeating complete rejected specifications."""

    summaries: list[dict[str, Any]] = []
    for failure in failures[-max(1, int(limit)) :]:
        if not isinstance(failure, dict):
            summaries.append({"reason": str(failure)[:240]})
            continue
        item = {
            key: failure.get(key)
            for key in (
                "candidate_id",
                "candidate_role",
                "generation",
                "reason",
            )
            if failure.get(key) is not None
        }
        details = failure.get("details")
        if details:
            values = details if isinstance(details, list) else [details]
            item["details"] = [str(value)[:240] for value in values[:3]]
        validation_errors = failure.get("validation_errors")
        if isinstance(validation_errors, list) and validation_errors:
            item["validation_errors"] = [
                {
                    key: str(error.get(key))[:240]
                    for key in ("path", "code", "message")
                    if isinstance(error, dict) and error.get(key) is not None
                }
                for error in validation_errors[:3]
            ]
            item["validation_error_count"] = len(validation_errors)
        attempt_failures = failure.get("attempt_failures")
        if isinstance(attempt_failures, list):
            item["attempt_failure_count"] = len(attempt_failures)
            item["attempt_failure_reasons"] = [
                str(value.get("reason") or "unknown")[:120]
                for value in attempt_failures[-3:]
                if isinstance(value, dict)
            ]
        summaries.append(item or {"reason": "unknown_failure"})
    return summaries


def _generate_slot_batch(
    *,
    requests: list[dict[str, Any]],
    design_agent: DesignAgent,
    problem: Any,
    budget: dict[str, int],
    capabilities: dict[str, Any],
    historical_references: list[dict[str, Any]],
    maximum_concurrency: int,
    initialization_mode: bool,
    on_candidate_ready: Callable[[dict[str, Any]], None] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Generate independent role slots concurrently and consume in slot order."""

    concurrency = max(1, min(DEFAULT_LLM_MAX_CONCURRENCY, int(maximum_concurrency)))
    request_semaphore = BoundedSemaphore(concurrency)
    children: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []

    def run(request: dict[str, Any], references: list[dict[str, Any]]) -> dict[str, Any]:
        with request_semaphore:
            return _generate_candidate_slot(
                design_agent=design_agent,
                base_payload=request["base_payload"],
                role_config=request["role_config"],
                problem=problem,
                budget=budget,
                capabilities=capabilities,
                current_references=references,
                historical_references=historical_references,
                maximum_attempts=int(request["maximum_attempts"]),
            )

    if concurrency == 1:
        for request in requests:
            try:
                slot = run(request, children)
            except Exception as exc:
                slot = _failed_generation_slot(request, exc)
            results.append(slot)
            child = slot.get("child")
            if child is not None:
                children.append(child)
                if on_candidate_ready is not None:
                    on_candidate_ready(child)
        return results, children

    # Submit in deterministic slot order. Consuming futures in the same order
    # keeps candidate acceptance and single-GPU training reproducible, while
    # the remaining network requests continue in background threads.
    with ThreadPoolExecutor(
        max_workers=concurrency,
        thread_name_prefix="llm-candidate",
    ) as executor:
        futures: list[Future] = [
            executor.submit(run, request, []) for request in requests
        ]
        for request, future in zip(requests, futures):
            try:
                slot = future.result()
                slot = _reconcile_parallel_slot(
                    slot=slot,
                    request=request,
                    accepted_children=children,
                    historical_references=historical_references,
                    design_agent=design_agent,
                    problem=problem,
                    budget=budget,
                    capabilities=capabilities,
                    initialization_mode=initialization_mode,
                    request_semaphore=request_semaphore,
                )
            except Exception as exc:
                slot = _failed_generation_slot(request, exc)
            results.append(slot)
            child = slot.get("child")
            if child is not None:
                children.append(child)
                if on_candidate_ready is not None:
                    on_candidate_ready(child)
    return results, children


def _failed_generation_slot(
    request: dict[str, Any], exc: Exception
) -> dict[str, Any]:
    """Convert one unexpected role failure into bounded refill evidence."""

    payload = dict(request.get("base_payload") or {})
    error = f"{type(exc).__name__}: {exc}"
    failure = {
        "candidate_id": payload.get("candidate_id"),
        "candidate_slot": payload.get("candidate_slot"),
        "candidate_role": payload.get("candidate_role"),
        "generation": payload.get("generation"),
        "reason": "candidate_slot_generation_exception",
        "details": [error],
    }
    trace = {
        "generation": payload.get("generation"),
        "candidate_id": payload.get("candidate_id"),
        "candidate_slot": payload.get("candidate_slot"),
        "candidate_role": payload.get("candidate_role"),
        "success": False,
        "parse_success": False,
        "validation_success": False,
        "parse_errors": [],
        "error": error,
    }
    return _slot_result(None, [trace], [], [failure], 0, 0)


def _reconcile_parallel_slot(
    *,
    slot: dict[str, Any],
    request: dict[str, Any],
    accepted_children: list[dict[str, Any]],
    historical_references: list[dict[str, Any]],
    design_agent: DesignAgent,
    problem: Any,
    budget: dict[str, int],
    capabilities: dict[str, Any],
    initialization_mode: bool,
    request_semaphore: BoundedSemaphore,
) -> dict[str, Any]:
    """Restore sibling novelty checks omitted from concurrent first attempts."""

    child = slot.get("child")
    if child is None or not accepted_children:
        return slot
    if initialization_mode:
        duplicate = _find_exact_generation_zero_duplicate(child, accepted_children)
        if duplicate is None:
            return slot
        retry_context = {
            "reason": "exact_normalized_duplicate",
            "duplicate_candidate_id": duplicate.get("candidate_id"),
            "instruction": (
                "Keep the assigned role and immutable contracts, but change at least "
                "one normalized executable AlgorithmSpec decision."
            ),
        }
    else:
        similarity = audit_candidate_similarity(
            child,
            [
                {**item, "current_generation": True}
                for item in accepted_children
            ],
            role_name=str(
                request["base_payload"].get("candidate_role") or ""
            ),
        )
        if not (
            similarity.get("exact_duplicate")
            or similarity.get("negligible_executable_change")
            or similarity.get("requires_diversity_retry")
        ):
            child.update(similarity)
            return slot
        if (
            similarity.get("requires_diversity_retry")
            and not similarity.get("exact_duplicate")
            and not similarity.get("negligible_executable_change")
            and not bool(
                request["base_payload"].get("retry_low_diversity", True)
            )
        ):
            child.update(similarity)
            child["accepted_despite_similarity"] = True
            child["diversity_warning"] = True
            return slot
        retry_context = {
            "reason": (
                "exact_normalized_duplicate"
                if similarity.get("exact_duplicate")
                else "low_executable_distance"
            ),
            "nearest_candidate_id": similarity.get("nearest_candidate_id"),
            "nearest_candidate_role": similarity.get("nearest_candidate_role"),
            "current_pairwise_distance": similarity.get("nearest_candidate_distance"),
            "overlapping_modules": similarity.get("overlapping_modules") or [],
            "different_modules": similarity.get("different_modules") or [],
            "instruction": (
                "Keep the assigned role and immutable contracts, but make one meaningful "
                "normalized executable change relative to accepted siblings."
            ),
        }

    repair_payload = {
        **request["base_payload"],
        "diversity_retry": retry_context,
    }
    if retry_context["reason"] == "exact_normalized_duplicate":
        repair_payload["duplicate_retry"] = retry_context
    with request_semaphore:
        repaired = _generate_candidate_slot(
            design_agent=design_agent,
            base_payload=repair_payload,
            role_config=request["role_config"],
            problem=problem,
            budget=budget,
            capabilities=capabilities,
            current_references=accepted_children,
            historical_references=historical_references,
            maximum_attempts=2,
        )
    combined = {
        "child": repaired.get("child") or child,
        "traces": [*(slot.get("traces") or []), *(repaired.get("traces") or [])],
        "candidate_audit": [
            *(slot.get("candidate_audit") or []),
            *(repaired.get("candidate_audit") or []),
        ],
        "failures": [
            *(slot.get("failures") or []),
            {
                "candidate_id": request["base_payload"].get("candidate_id"),
                "candidate_role": request["base_payload"].get("candidate_role"),
                "reason": "parallel_sibling_diversity_retry",
                **retry_context,
            },
            *(repaired.get("failures") or []),
        ],
        "duplicate_count": int(slot.get("duplicate_count") or 0)
        + int(repaired.get("duplicate_count") or 0)
        + int(retry_context["reason"] == "exact_normalized_duplicate"),
        "retry_count": int(slot.get("retry_count") or 0)
        + int(repaired.get("retry_count") or 0)
        + 1,
    }
    if repaired.get("child") is None:
        combined["child"]["accepted_despite_similarity"] = True
    return combined


def _generate_initialization_role_candidates(
    *,
    design_agent: DesignAgent,
    shared_payload: dict[str, Any],
    problem: Any,
    budget: dict[str, int],
    capabilities: dict[str, Any],
    minimum_accepted_candidates: int = MINIMUM_GENERATION_ZERO_CANDIDATES,
    maximum_concurrency: int = 1,
    on_candidate_ready: Callable[[dict[str, Any]], None] | None = None,
    role_plan: tuple[str, ...] = GENERATION_ZERO_ROLE_PLAN,
    allow_partial_on_exhaustion: bool = False,
) -> dict[str, Any]:
    """Generate initial role candidates and refill until the target is met.

    When ``allow_partial_on_exhaustion`` is enabled, a non-empty validated pool is
    returned after the bounded refill budget is exhausted.  The default remains
    strict so the normal evolutionary search keeps its existing minimum.
    """

    if minimum_accepted_candidates <= 0:
        raise ValueError("Generation-0 minimum accepted candidate count must be positive")
    if not role_plan:
        raise ValueError("Generation-0 role plan must not be empty")
    unknown_roles = sorted(set(role_plan) - set(GENERATION_ZERO_ROLE_DEFINITIONS))
    if unknown_roles:
        raise ValueError("Unknown Generation-0 roles: " + ", ".join(unknown_roles))
    if len(set(role_plan)) != len(role_plan):
        raise ValueError("Generation-0 role plan must contain unique roles")
    if minimum_accepted_candidates > len(role_plan):
        raise ValueError(
            "Generation-0 minimum accepted candidate count cannot exceed the role count"
        )

    requests: list[dict[str, Any]] = []
    for candidate_slot, role_name in enumerate(role_plan):
        role_config = GENERATION_ZERO_ROLE_DEFINITIONS[role_name]
        candidate_id = f"generation_0_candidate_{candidate_slot}"
        requests.append(
            {
                "base_payload": {
                    **shared_payload,
                    "initialization_mode": True,
                    "candidate_id": candidate_id,
                    "candidate_slot": candidate_slot,
                    "candidate_role": role_name,
                    "candidate_role_objective": role_config["objective"],
                    "candidate_diversity_axis": role_config["primary_axis"],
                    "temperature": float(role_config["default_temperature"]),
                    "expected_proposal_count": 1,
                },
                "role_config": role_config,
                "maximum_attempts": 2,
            }
        )
    slot_results, children = _generate_slot_batch(
        requests=requests,
        design_agent=design_agent,
        problem=problem,
        budget=budget,
        capabilities=capabilities,
        historical_references=[],
        maximum_concurrency=maximum_concurrency,
        initialization_mode=True,
        on_candidate_ready=on_candidate_ready,
    )
    traces: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    duplicate_count = 0
    retry_count = 0
    for request, slot in zip(requests, slot_results):
        candidate_id = request["base_payload"]["candidate_id"]
        traces.extend(slot["traces"])
        audits.extend(slot["candidate_audit"])
        if slot["child"] is None:
            failures.append(
                slot["failures"][-1]
                if slot["failures"]
                else {"candidate_id": candidate_id, "reason": "slot_failure"}
            )
        duplicate_count += int(slot["duplicate_count"])
        retry_count += int(slot["retry_count"])
    refill_round = 0
    refill_exhausted = False
    while len(children) < minimum_accepted_candidates:
        refill_round += 1
        if refill_round > MAX_GENERATION_ZERO_REFILL_ROUNDS:
            if allow_partial_on_exhaustion and children:
                refill_exhausted = True
                refill_round -= 1
                break
            raise RuntimeError(
                "Generation 0 exhausted its bounded LLM refill budget: "
                f"accepted={len(children)}/{minimum_accepted_candidates}, "
                f"refill_rounds={refill_round - 1}, "
                f"failures={json.dumps(_compact_failure_summaries(failures), ensure_ascii=False)}"
            )
        missing_slots = [
            (candidate_slot, role_name)
            for candidate_slot, role_name in enumerate(role_plan)
            if not any(
                int(item.get("candidate_slot") or 0) == candidate_slot
                for item in children
            )
        ]
        candidate_slot, role_name = missing_slots[(refill_round - 1) % len(missing_slots)]
        role_config = GENERATION_ZERO_ROLE_DEFINITIONS[role_name]
        candidate_id = f"generation_0_candidate_{candidate_slot}"
        slot = _generate_candidate_slot(
            design_agent=design_agent,
            base_payload={
                **shared_payload,
                "initialization_mode": True,
                "candidate_id": candidate_id,
                "candidate_slot": candidate_slot,
                "candidate_role": role_name,
                "candidate_role_objective": role_config["objective"],
                "candidate_diversity_axis": role_config["primary_axis"],
                "temperature": float(role_config["default_temperature"]),
                "expected_proposal_count": 1,
                "candidate_refill_round": refill_round,
                "generation_retry": {
                    "reason": "candidate_pool_shortfall",
                    "generation": 0,
                    "accepted_candidates": len(children),
                    "accepted_candidate_ids": [
                        item.get("candidate_id") for item in children
                    ],
                    "target_candidates": minimum_accepted_candidates,
                    "missing_candidates": minimum_accepted_candidates
                    - len(children),
                    "previous_failures": _compact_failure_summaries(failures),
                    "instruction": (
                        "This generation still has a missing role slot. Use the supplied failed-attempt "
                        "information to generate a corrected complete AlgorithmSpec for this exact slot."
                    ),
                },
            },
            role_config=role_config,
            problem=problem,
            budget=budget,
            capabilities=capabilities,
            current_references=children,
            historical_references=[],
        )
        traces.extend(slot["traces"])
        audits.extend(slot["candidate_audit"])
        failures.extend(slot["failures"])
        duplicate_count += int(slot["duplicate_count"])
        retry_count += int(slot["retry_count"])
        if slot["child"] is not None:
            children.append(slot["child"])
            if on_candidate_ready is not None:
                on_candidate_ready(slot["child"])
    return {
        "children": children,
        "rejected_children": failures,
        "feedback_analysis": "Generation 0 has no parent feedback.",
        "strategy_summary": (
            f"{len(role_plan)} role slots with one bounded corrective retry per failed or "
            "duplicate response; LLM refill starts only below the configured minimum."
        ),
        "generation_mode": "initialization",
        "proposal_count": len(role_plan),
        "accepted_count": len(children),
        "target_accepted_count": minimum_accepted_candidates,
        "partial_candidate_pool": len(children) < minimum_accepted_candidates,
        "refill_exhausted": refill_exhausted,
        "refill_rounds": refill_round,
        "llm_calls": len(traces),
        "traces": traces,
        "candidate_audit": audits,
        "duplicate_count": duplicate_count,
        "retry_count": retry_count,
        "role_plan": list(role_plan),
        "resolved_design_context": deepcopy(shared_payload.get("resolved_design_context") or {}),
    }




def _generate_candidate_slot(
    *,
    design_agent: DesignAgent,
    base_payload: dict[str, Any],
    role_config: dict[str, Any],
    problem: Any,
    budget: dict[str, int],
    capabilities: dict[str, Any],
    current_references: list[dict[str, Any]],
    historical_references: list[dict[str, Any]],
    maximum_attempts: int = 3,
) -> dict[str, Any]:
    """Fill one role slot, optionally allowing directed retry and availability calls."""

    traces: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    duplicate_count = 0
    retry_count = 0
    fallback: dict[str, Any] | None = None
    retry_context: dict[str, Any] | None = None
    exact_duplicate_after_retry = False
    if maximum_attempts < 1:
        raise ValueError("maximum_attempts must be positive")
    for attempt in range(1, maximum_attempts + 1):
        payload = dict(base_payload)
        if attempt == 2:
            retry_count += 1
            previous_failure = failures[-1] if failures else {}
            if (
                isinstance(previous_failure, dict)
                and previous_failure.get("reason")
                in {
                    "source_spec_validation_failed",
                    "validation_failed",
                    "schedule_scaling_failed",
                }
                and isinstance(
                    previous_failure.get("rejected_algorithm_spec"), dict
                )
            ):
                payload["validation_repair"] = {
                    key: deepcopy(previous_failure.get(key))
                    for key in (
                        "reason",
                        "validation_errors",
                        "details",
                        "rejected_algorithm_spec",
                    )
                    if previous_failure.get(key) is not None
                }
            elif retry_context is not None:
                payload["diversity_retry"] = retry_context
                if retry_context.get("reason") == "exact_normalized_duplicate":
                    payload["duplicate_retry"] = retry_context
            else:
                payload["generation_retry"] = {
                    **deepcopy(payload.get("generation_retry") or {}),
                    "previous_attempt_failure": json_safe(
                        failures[-1] if failures else {}
                    ),
                    "instruction": (
                        "Retry the same assigned role with one complete, valid, trainable AlgorithmSpec. "
                        "Use the candidate-shortfall context and previous-attempt failure supplied here."
                    ),
                }
        elif attempt == 3:
            retry_count += 1
            required_changes = max(
                1,
                int(
                    (
                        base_payload.get("generation_policy_audit") or {}
                    ).get("required_major_module_changes")
                    or 1
                ),
            )
            payload["availability_fill"] = {
                "instruction": (
                    "Generate one complete, valid and trainable AlgorithmSpec for the assigned candidate slot. "
                    "Candidate availability is now more important than maximum diversity. The proposal must not be "
                    "exactly identical to an already accepted current-generation AlgorithmSpec after normalization. "
                    "Meaningful similarity is acceptable. Change at least "
                    f"{required_changes} major executable module(s) with a clear rationale."
                )
            }
        child, audit, trace, failure = _run_initialization_design_call(
            design_agent=design_agent,
            payload=payload,
            role_config=role_config,
            problem=problem,
            budget=budget,
            capabilities=capabilities,
            attempt=attempt,
        )
        audits.append(audit)
        traces.append(trace)
        if child is None:
            failures.append(failure or {"reason": "unknown_generation_failure"})
            if attempt == 2 and fallback is not None:
                fallback["diversity_retry_count"] = retry_count
                fallback["accepted_despite_similarity"] = True
                return _slot_result(fallback, traces, audits, failures, duplicate_count, retry_count)
            continue
        if bool(base_payload.get("initialization_mode")):
            duplicate = _find_exact_generation_zero_duplicate(child, current_references)
            if duplicate is not None:
                duplicate_count += 1
                failure = {
                    "candidate_id": base_payload.get("candidate_id"),
                    "candidate_role": base_payload.get("candidate_role"),
                    "reason": "exact_normalized_duplicate",
                    "duplicate_candidate_id": duplicate.get("candidate_id"),
                }
                failures.append(failure)
                retry_context = {
                    "reason": "exact_normalized_duplicate",
                    "duplicate_candidate_id": duplicate.get("candidate_id"),
                    "instruction": (
                        "Keep the assigned role and immutable contracts, but change at least one "
                        "normalized executable AlgorithmSpec decision."
                    ),
                }
                continue
            return _slot_result(
                child, traces, audits, failures, duplicate_count, retry_count
            )
        references = [
            *[
                {**item, "current_generation": False}
                for item in historical_references
            ],
            *[
                {**item, "current_generation": True}
                for item in current_references
            ],
        ]
        similarity = audit_candidate_similarity(
            child, references, role_name=str(base_payload.get("candidate_role") or "")
        )
        child.update(similarity)
        child["diversity_retry_count"] = retry_count
        audit.update(similarity)
        policy = dict(base_payload.get("generation_policy_audit") or {})
        required_major_changes = int(
            policy.get("required_major_module_changes") or 0
        )
        actual_major_changes = list(similarity.get("different_modules") or [])
        if (
            policy.get("generation_mode") == "escape"
            and len(actual_major_changes) < required_major_changes
        ):
            fallback = None
            failure = {
                "candidate_id": base_payload.get("candidate_id"),
                "candidate_role": base_payload.get("candidate_role"),
                "reason": "stagnation_minimum_major_module_changes_not_met",
                "required_major_module_changes": required_major_changes,
                "actual_different_modules": actual_major_changes,
            }
            failures.append(failure)
            retry_context = {
                **failure,
                "instruction": (
                    "Keep the assigned role and trusted contracts, but change at least "
                    f"{required_major_changes} major executable modules relative to the nearest "
                    "Top-3 or accepted sibling. Metadata-only changes do not count."
                ),
            }
            continue
        if similarity["exact_duplicate"] or similarity.get("negligible_executable_change"):
            duplicate_count += 1
            exact_duplicate_after_retry = exact_duplicate_after_retry or attempt >= 2
            duplicate_scope = (
                "current_generation"
                if similarity.get("exact_current_generation_duplicate")
                else "previous_top3"
            )
            failure = {
                "candidate_id": base_payload.get("candidate_id"),
                "candidate_role": base_payload.get("candidate_role"),
                "reason": (
                    "exact_duplicate_after_retry" if attempt >= 2 else "exact_normalized_duplicate"
                ),
                "nearest_candidate_id": similarity.get("nearest_candidate_id"),
                "duplicate_scope": duplicate_scope,
                "normalized_executable_distance": similarity.get(
                    "normalized_executable_distance"
                ),
            }
            failures.append(failure)
            if fallback is not None:
                fallback["diversity_retry_count"] = retry_count
                fallback["accepted_despite_similarity"] = True
                return _slot_result(fallback, traces, audits, failures, duplicate_count, retry_count)
            retry_context = {
                "reason": "exact_normalized_duplicate",
                "duplicate_scope": duplicate_scope,
                "nearest_candidate_id": similarity.get("nearest_candidate_id"),
                "nearest_candidate_role": similarity.get("nearest_candidate_role"),
                "current_pairwise_distance": similarity.get("nearest_candidate_distance"),
                "overlapping_modules": similarity.get("overlapping_modules") or [],
                "different_modules": similarity.get("different_modules") or [],
                "instruction": (
                    "Keep the assigned role and immutable contracts, but make a non-negligible normalized executable change."
                ),
            }
            continue
        if (
            similarity["requires_diversity_retry"]
            and attempt == 1
            and maximum_attempts > 1
            and bool(base_payload.get("retry_low_diversity", True))
        ):
            fallback = child
            retry_context = {
                "reason": "low_executable_distance",
                "nearest_candidate_id": similarity.get("nearest_candidate_id"),
                "nearest_candidate_role": similarity.get("nearest_candidate_role"),
                "current_pairwise_distance": similarity.get("nearest_candidate_distance"),
                "overlapping_modules": similarity.get("overlapping_modules") or [],
                "different_modules": similarity.get("different_modules") or [],
                "instruction": (
                    "Keep the assigned role and physical objective. Make one meaningful executable change; metadata-only changes do not count."
                ),
            }
            continue
        if similarity["requires_diversity_retry"] and (
            maximum_attempts == 1
            or not bool(base_payload.get("retry_low_diversity", True))
        ):
            child["accepted_despite_similarity"] = True
            child["diversity_warning"] = True
        if exact_duplicate_after_retry:
            child["exact_duplicate_after_retry"] = True
        if (
            str(base_payload.get("candidate_role")) == "novelty_exploration"
            and bool(child.get("accepted_despite_similarity"))
        ):
            child["novelty_target_not_fully_met"] = True
        return _slot_result(child, traces, audits, failures, duplicate_count, retry_count)
    return _slot_result(None, traces, audits, failures, duplicate_count, retry_count)


def _find_exact_generation_zero_duplicate(
    candidate: dict[str, Any], references: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Return an identical normalized executable spec without similarity scoring."""

    candidate_spec = executable_algorithm_spec(candidate.get("algorithm_spec") or {})
    for reference in references:
        if candidate_spec == executable_algorithm_spec(reference.get("algorithm_spec") or {}):
            return reference
    return None


def _slot_result(
    child: dict[str, Any] | None,
    traces: list[dict[str, Any]],
    audits: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    duplicate_count: int,
    retry_count: int,
) -> dict[str, Any]:
    return {
        "child": child,
        "traces": traces,
        "candidate_audit": audits,
        "failures": failures,
        "duplicate_count": duplicate_count,
        "retry_count": retry_count,
    }


def _run_initialization_design_call(
    *,
    design_agent: DesignAgent,
    payload: dict[str, Any],
    role_config: dict[str, Any],
    problem: Any,
    budget: dict[str, int],
    capabilities: dict[str, Any],
    attempt: int,
) -> tuple[dict[str, Any] | None, dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    payload = dict(payload)
    if "generation_policy_audit" not in payload:
        payload["generation_policy_audit"] = _build_generation_policy_audit(
            role_name=str(payload["candidate_role"]),
            role_config=role_config,
            generation_mode=dict(payload.get("generation_mode") or {}),
            parent_candidates=list(payload.get("parent_specs") or []),
        )
    if not bool(payload.get("evolutionary_inheritance_enabled", True)):
        payload["generation_policy_audit"].update(
            {
                "parent_candidate_ids": [],
                "parent_selection_enabled": False,
                "parent_algorithm_specs_exposed_to_llm": False,
                "forbid_exact_parent_replay": False,
                "increase_parent_diversity": False,
                "evolutionary_inheritance_enabled": False,
            }
        )
    candidate_id = str(payload["candidate_id"])
    role_name = str(payload["candidate_role"])
    try:
        result = design_agent.run(payload)
    except Exception as exc:
        result = None
        data: dict[str, Any] = {}
        error = f"{type(exc).__name__}: {exc}"
    else:
        data = dict(result.data or {})
        error = None if result.success else result.message
    usage = dict(data.get("token_usage") or {})
    audit = {
        "generation": int(payload.get("generation") or 0),
        "candidate_id": candidate_id,
        "candidate_slot": int(payload["candidate_slot"]),
        "candidate_role": role_name,
        "primary_diversity_axis": role_config.get("primary_axis"),
        "temperature": float(
            payload.get("temperature", role_config["default_temperature"])
        ),
        "generation_policy_audit": deepcopy(
            payload.get("generation_policy_audit") or {}
        ),
        "requested_max_tokens": data.get("requested_max_tokens", payload.get("max_tokens")),
        "request_mode": data.get("request_mode", "full_design"),
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "finish_reason": data.get("finish_reason"),
        "response_truncated": bool(data.get("response_truncated")),
        "parse_success": bool(data.get("parse_success", bool(result and result.success))),
        "validation_success": False,
        "attempt": attempt,
    }
    trace = {
        **audit,
        "success": False,
        "raw_response": data.get("raw_response") or "",
        "parse_errors": list(data.get("parse_errors") or []),
        "token_usage": usage,
        "provider": data.get("provider"),
        "provider_success": data.get("provider_success"),
        "provider_error": data.get("provider_error"),
        "model": data.get("model"),
        "model_version": data.get("model_version"),
        "latency_seconds": data.get("latency_seconds"),
        "proposal_count": data.get("proposal_count", len(data.get("proposals") or [])),
        "candidate_id_match": data.get("candidate_id_match"),
        "transport_repairs": list(data.get("transport_repairs") or []),
        "prompt_statistics": deepcopy(data.get("prompt_statistics") or {}),
        "error": error,
    }
    proposals = list(data.get("proposals") or [])
    if result is None or not result.success or len(proposals) != 1:
        if trace["response_truncated"]:
            reason = "truncated_design_response"
            instruction = (
                "Regenerate the same candidate as compact JSON. Do not repeat "
                "parents, registry entries, the template, or explanatory prose."
            )
        elif data.get("provider_success") is False:
            reason = "design_provider_failure"
            instruction = "Retry the same assigned candidate after the provider failure."
        else:
            reason = "proposal_transport_contract_failed"
            instruction = (
                "Return one proposal for the supplied candidate_id, with one complete "
                "algorithm_spec; keep audit prose concise."
            )
        failure = {
            "candidate_id": candidate_id,
            "candidate_role": role_name,
            "reason": reason,
            "details": trace["parse_errors"] or [error],
            "retry_instruction": instruction,
        }
        return None, audit, trace, failure
    spec = proposals[0].get("algorithm_spec") if isinstance(proposals[0], dict) else None
    proposal_raw_spec = proposals[0].get("algorithm_spec_raw") if isinstance(proposals[0], dict) else None
    raw_spec = deepcopy(proposal_raw_spec if isinstance(proposal_raw_spec, dict) else spec)
    try:
        proposal_metadata = _extract_proposal_metadata(
            proposals[0], payload=payload, role_name=role_name
        )
    except ValueError as exc:
        failure = {
            "candidate_id": candidate_id,
            "candidate_role": role_name,
            "reason": "proposal_metadata_validation_failed",
            "details": str(exc),
        }
        audit["proposal_metadata_error"] = str(exc)
        trace["error"] = str(exc)
        return None, audit, trace, failure
    audit["proposal_metadata"] = deepcopy(proposal_metadata)
    trace["proposal_metadata"] = deepcopy(proposal_metadata)
    try:
        recommendation = budget.get("recommended_total_iterations")
        if recommendation is None:
            recommendation = budget.get("required_total_iterations")
        if recommendation is None:
            recommendation = budget.get("maximum_total_iterations")
        optimization_budget_audit = audit_algorithm_spec_iteration_recommendation(
            spec, int(recommendation or 0)
        )
    except (AttributeError, TypeError, ValueError, KeyError) as exc:
        failure = {
            "candidate_id": candidate_id,
            "candidate_role": role_name,
            "reason": "iteration_recommendation_audit_failed",
            "details": str(exc),
        }
        trace["error"] = str(exc)
        return None, audit, trace, failure
    if isinstance(spec, dict) and isinstance(spec.get("generation_metadata"), dict):
        spec["generation_metadata"].update(
            {
                "candidate_role": role_name,
                "role_design_objective": role_config["objective"],
                "role_requirements_applied": list(role_config["requirements"]),
                "primary_diversity_axis": role_config.get("primary_axis"),
                "diversity_hypothesis": spec["generation_metadata"].get("diversity_hypothesis")
                or f"Create meaningful executable diversity along {role_config.get('primary_axis') or role_name}.",
                "major_executable_differences_expected": list(role_config.get("major_modules") or []),
                "generation_policy_audit": deepcopy(
                    payload.get("generation_policy_audit") or {}
                ),
            }
        )
    # Normalize the LLM-authored search spec before applying the run fidelity
    # budget.  Keeping this normalized, unscaled form is essential: the final
    # high-fidelity review must reconstruct a fresh 10,000-iteration schedule
    # from the authored phase proportions, not expand the already compressed
    # 1,000-iteration schedule.
    source_validation_budget = dict(budget)
    source_validation_budget.pop("required_total_iterations", None)
    source_validation_budget["maximum_total_iterations"] = 2**63 - 1
    source_validation = validate_algorithm_spec(
        spec, problem, source_validation_budget, capabilities
    )
    if not source_validation.valid:
        serialized_validation_errors = [
            asdict(issue) for issue in source_validation.errors
        ]
        failure = {
            "candidate_id": candidate_id,
            "candidate_role": role_name,
            "reason": "source_spec_validation_failed",
            "validation_errors": serialized_validation_errors,
            "rejected_algorithm_spec": deepcopy(spec),
        }
        audit["validation_errors"] = serialized_validation_errors
        trace["validation_errors"] = serialized_validation_errors
        trace["error"] = "Generated source AlgorithmSpec failed deterministic validation."
        return None, audit, trace, failure
    search_algorithm_spec = deepcopy(source_validation.normalized_spec)
    scaling = dict(budget.get("schedule_scaling") or {})
    try:
        spec, scaling_audit = adapt_algorithm_spec_to_generation_budget(
            search_algorithm_spec,
            int(budget["required_total_iterations"]),
            min_lbfgs_iterations=int(scaling.get("min_lbfgs_iterations", 1)),
            min_adaptive_sampling_events=int(
                scaling.get("min_adaptive_sampling_events", 0)
            ),
            preserve_enabled_training_stages=bool(
                scaling.get("preserve_enabled_training_stages", True)
            ),
            enforce_full_iterations=True,
            source_fidelity="design",
            target_fidelity=str(budget.get("fidelity") or "low"),
            adaptive_policy_source_spec=search_algorithm_spec,
            source_optimizer_step_budget=sum(
                int(phase.get("iterations") or 0)
                for phase in search_algorithm_spec["optimization"]["phases"]
            ),
            target_sampling_budget=budget.get("maximum_sampling_points"),
        )
    except (AttributeError, TypeError, ValueError, KeyError) as exc:
        failure = {
            "candidate_id": candidate_id,
            "candidate_role": role_name,
            "reason": "schedule_scaling_failed",
            "details": str(exc),
            "rejected_algorithm_spec": deepcopy(search_algorithm_spec),
        }
        trace["error"] = str(exc)
        return None, audit, trace, failure
    optimization_budget_audit = {
        **optimization_budget_audit,
        **scaling_audit,
        "source_spec_normalized_before_scaling": True,
    }
    validation = validate_algorithm_spec(spec, problem, budget, capabilities)
    audit["validation_success"] = bool(validation.valid)
    trace["validation_success"] = bool(validation.valid)
    if not validation.valid:
        serialized_validation_errors = [asdict(issue) for issue in validation.errors]
        failure = {
            "candidate_id": candidate_id,
            "candidate_role": role_name,
            "reason": "validation_failed",
            "validation_errors": serialized_validation_errors,
            "rejected_algorithm_spec": deepcopy(spec),
        }
        audit["validation_errors"] = serialized_validation_errors
        trace["validation_errors"] = serialized_validation_errors
        trace["error"] = "Generated AlgorithmSpec failed deterministic validation."
        return None, audit, trace, failure
    normalized = validation.normalized_spec
    proposal_metadata["validation"] = {
        **deepcopy(proposal_metadata.get("validation") or {}),
        "trusted_validator_passed": True,
        "normalized_before_newness_check": True,
        "warning_count": len(validation.warnings),
    }
    trace["success"] = True
    trace["error"] = None
    child = {
        "candidate_id": candidate_id,
        "candidate_label": candidate_id,
        "candidate_slot": int(payload["candidate_slot"]),
        "candidate_role": role_name,
        "role_design_objective": role_config["objective"],
        "primary_diversity_axis": role_config.get("primary_axis"),
        "algorithm_spec": normalized,
        "search_algorithm_spec": search_algorithm_spec,
        "algorithm_spec_raw": raw_spec,
        "population_role": "llm_generated",
        "generation_mode": (
            "initialization"
            if payload.get("initialization_mode")
            else (
                "role_guided_design"
                if bool(payload.get("evolutionary_inheritance_enabled", True))
                else "independent_proposal"
            )
        ),
        "parent_ids": list(proposal_metadata.get("parent_ids") or []),
        "proposal_metadata": deepcopy(proposal_metadata),
        "variation_strategy": proposal_metadata.get("variation_strategy"),
        "inherited_strengths": deepcopy(proposal_metadata.get("inherited_strengths") or []),
        "major_changes": deepcopy(proposal_metadata.get("major_changes") or []),
        "design_rationale": proposal_metadata.get("design_rationale") or "",
        "proposal_validation": deepcopy(proposal_metadata.get("validation") or {}),
        "generation_policy_audit": deepcopy(
            payload.get("generation_policy_audit") or {}
        ),
        "fresh_training": True,
        "requires_fresh_training": True,
        "design_intent": role_config["objective"],
        "difference_summary": f"Primary executable diversity axis: {role_config.get('primary_axis') or role_name}.",
        "variation_reason": f"Independent {role_name} design call.",
        "heritage_explanation": {
            "overall": "Parent-free initialization."
            if payload.get("initialization_mode")
            else (
                "Designed from the shared controller-selected global low-fidelity distinct Top-3 context."
                if bool(payload.get("evolutionary_inheritance_enabled", True))
                else "Parent-free independent proposal conditioned only on knowledge, execution summaries, and run-scoped posterior memory."
            )
        },
        "expected_improvements": list(normalized.get("generation_metadata", {}).get("expected_advantages") or []),
        "validation_warning_count": len(validation.warnings),
        "optimization_budget_audit": optimization_budget_audit,
    }
    return child, audit, trace, None


def _extract_proposal_metadata(
    proposal: dict[str, Any], *, payload: dict[str, Any], role_name: str
) -> dict[str, Any]:
    """Canonicalize audit-only lineage metadata without changing execution.

    These fields never reach the Trusted Builder.  Missing envelope metadata
    can therefore be reconstructed from the assigned role, global low-fidelity Top-3 order,
    and AlgorithmSpec generation metadata.  Executable novelty is still
    checked later from normalized specs, so this cannot admit a replay.
    """

    if payload.get("initialization_mode"):
        return {
            "role": role_name,
            "parent_ids": [],
            "variation_strategy": None,
            "inherited_strengths": [],
            "major_changes": [],
            "design_rationale": "Parent-free generation-0 initialization.",
            "validation": {},
        }
    if not bool(payload.get("evolutionary_inheritance_enabled", True)):
        forbidden = {
            "parent_ids": proposal.get("parent_ids"),
            "variation_strategy": proposal.get("variation_strategy"),
            "inherited_strengths": proposal.get("inherited_strengths"),
        }
        populated = {
            key: value
            for key, value in forbidden.items()
            if value not in (None, "", [], (), {})
        }
        if populated:
            raise ValueError(
                "independent proposal contains forbidden evolutionary lineage metadata: "
                + ", ".join(sorted(populated))
            )
        spec_metadata = dict(
            ((proposal.get("algorithm_spec") or {}).get("generation_metadata") or {})
        )
        changes = proposal.get("major_changes")
        if not isinstance(changes, list) or not changes:
            changes = list(spec_metadata.get("changed_modules") or []) or [
                "Fresh complete AlgorithmSpec generated without a parent specification."
            ]
        rationale = proposal.get("design_rationale")
        if not isinstance(rationale, str) or not rationale.strip():
            rationale = str(
                spec_metadata.get("design_hypothesis")
                or payload.get("candidate_role_objective")
                or f"Parent-free {role_name} proposal."
            )
        self_validation = proposal.get("validation")
        if not isinstance(self_validation, dict):
            self_validation = {}
        return {
            "role": role_name,
            "parent_ids": [],
            "variation_strategy": None,
            "inherited_strengths": [],
            "major_changes": changes,
            "design_rationale": rationale,
            "validation": self_validation,
            "metadata_repairs": [],
            "lineage": "parent_free_independent_proposal",
        }
    repairs: list[dict[str, str]] = []

    def repair(path: str, action: str) -> None:
        repairs.append({"path": path, "action": action})

    if str(proposal.get("role") or "") != role_name:
        repair("role", "set audit role to the assigned candidate_role")
    allowed_parent_ids = [
        str(item.get("candidate_id"))
        for item in payload.get("parent_specs") or []
        if item.get("candidate_id")
    ]
    spec_metadata = dict(
        ((proposal.get("algorithm_spec") or {}).get("generation_metadata") or {})
    )
    raw_parent_ids = proposal.get("parent_ids")
    if not isinstance(raw_parent_ids, list) or not raw_parent_ids:
        raw_parent_ids = spec_metadata.get("parent_spec_ids") or []
        repair(
            "parent_ids",
            "reconstructed lineage from AlgorithmSpec metadata and global low-fidelity Top-3 order",
        )
    parent_ids = [
        item
        for item in dict.fromkeys(str(item) for item in raw_parent_ids)
        if item in allowed_parent_ids
    ]
    if role_name == "elite_conservative_improvement":
        if not allowed_parent_ids:
            raise ValueError("elite_conservative_improvement requires global low-fidelity Rank-1")
        parent_ids = [allowed_parent_ids[0], *[
            item for item in parent_ids if item != allowed_parent_ids[0]
        ]]
    elif role_name == "multi_parent_synthesis":
        for candidate_id in allowed_parent_ids:
            if candidate_id not in parent_ids:
                parent_ids.append(candidate_id)
            if len(parent_ids) >= 2:
                break
    elif not parent_ids and allowed_parent_ids:
        parent_ids = [allowed_parent_ids[0]]
    if not parent_ids:
        raise ValueError("evolution proposal could not resolve a global low-fidelity Top-3 parent")
    strategy = str(proposal.get("variation_strategy") or "")
    if strategy not in ALLOWED_VARIATION_STRATEGIES:
        strategy = (
            "multi_parent_recomposition"
            if role_name == "multi_parent_synthesis"
            else "novel_recomposition"
            if role_name == "novelty_exploration"
            else "single_parent_mutation"
        )
        repair(
            "variation_strategy",
            "inserted a role-compatible audit strategy",
        )
    inherited = proposal.get("inherited_strengths")
    changes = proposal.get("major_changes")
    rationale = proposal.get("design_rationale")
    self_validation = proposal.get("validation")
    if not isinstance(inherited, list):
        inherited = []
        repair("inherited_strengths", "inserted an empty audit list")
    if not isinstance(changes, list) or not changes:
        changes = list(spec_metadata.get("changed_modules") or []) or [
            "Executable differences are verified after deterministic normalization."
        ]
        repair("major_changes", "derived audit changes from generation metadata")
    if not isinstance(rationale, str) or not rationale.strip():
        rationale = str(
            spec_metadata.get("design_hypothesis")
            or payload.get("candidate_role_objective")
            or f"Role-guided {role_name} proposal."
        )
        repair("design_rationale", "derived rationale from assigned role metadata")
    if not isinstance(self_validation, dict):
        self_validation = {}
        repair("validation", "inserted an empty LLM self-check object")
    return {
        "role": role_name,
        "parent_ids": parent_ids,
        "variation_strategy": strategy,
        "inherited_strengths": deepcopy(inherited),
        "major_changes": deepcopy(changes),
        "design_rationale": rationale.strip(),
        "validation": deepcopy(self_validation),
        "metadata_repairs": repairs,
    }


def _write_generation_zero_candidate_artifacts(
    generation_dir: Path, variation_result: dict[str, Any]
) -> None:
    """Persist the generated role candidates without diversity scoring."""

    generation_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        generation_dir / "initialization_role_plan.json",
        {
            "role_plan": list(GENERATION_ZERO_ROLE_PLAN),
            "role_definitions": GENERATION_ZERO_ROLE_DEFINITIONS,
        },
    )
    _write_json(
        generation_dir / "initialization_candidates.json",
        [
            {
                key: item.get(key)
                for key in (
                    "candidate_id",
                    "candidate_slot",
                    "candidate_role",
                    "primary_diversity_axis",
                    "algorithm_spec",
                )
            }
            for item in variation_result.get("children") or []
        ],
    )


def _configured_design_max_tokens(config: dict[str, Any]) -> int:
    if config.get("llm_max_tokens") is not None:
        return int(config["llm_max_tokens"])
    return DEFAULT_DESIGN_MAX_TOKENS


def _configured_llm_max_concurrency(config: dict[str, Any]) -> int:
    return max(
        1,
        min(
            DEFAULT_LLM_MAX_CONCURRENCY,
            int(config.get("llm_max_concurrency") or DEFAULT_LLM_MAX_CONCURRENCY),
        ),
    )


def _generate_role_candidates(
    *,
    design_agent: DesignAgent,
    shared_payload: dict[str, Any],
    generation: int,
    parent_candidates: list[dict[str, Any]],
    problem: Any,
    budget: dict[str, int],
    capabilities: dict[str, Any],
    role_plan: tuple[str, ...] | list[str] | None = None,
    maximum_concurrency: int = 1,
    on_candidate_ready: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Generate one complete candidate per evolutionary or independent role."""

    inheritance_enabled = bool(
        shared_payload.get("evolutionary_inheritance_enabled", True)
    )
    if inheritance_enabled and len(parent_candidates) != EVOLUTION_PARENT_COUNT:
        raise ValueError(
            "Role-driven evolution requires exactly three controller-selected global low-fidelity parent candidates"
        )
    if not inheritance_enabled and parent_candidates:
        raise ValueError(
            "Budget-matched independent search must not receive parent candidates"
        )
    historical_references = (
        [
            {
                "candidate_id": item.get("candidate_id"),
                "candidate_role": (
                    f"previous_rank_{item.get('rank')}"
                    if bool(shared_payload.get("execution_feedback", True))
                    else "unordered_previous_parent"
                ),
                "algorithm_spec": item.get("algorithm_spec"),
            }
            for item in parent_candidates[:EVOLUTION_PARENT_COUNT]
        ]
        if inheritance_enabled
        else []
    )
    configured_role_plan = tuple(role_plan or NEW_CANDIDATE_ROLE_PLAN)
    if len(configured_role_plan) != EVOLUTION_TARGET_CANDIDATES:
        raise ValueError("Evolution and escape generations require five configured roles")
    requests: list[dict[str, Any]] = []
    for candidate_slot, role_name in enumerate(configured_role_plan, start=1):
        role_config = candidate_role_definition(
            role_name,
            execution_feedback=bool(
                shared_payload.get("execution_feedback", True)
            ),
        ) or GENERATION_ZERO_ROLE_DEFINITIONS.get(role_name)
        if role_config is None:
            raise ValueError(f"Unknown configured candidate role: {role_name}")
        generation_policy_audit = _build_generation_policy_audit(
            role_name=role_name,
            role_config=role_config,
            generation_mode=dict(shared_payload.get("generation_mode") or {}),
            parent_candidates=parent_candidates,
        )
        sampling_policy_audit = _build_generation_policy_audit(
            role_name=role_name,
            role_config=(CANDIDATE_ROLE_DEFINITIONS.get(role_name) or role_config),
            generation_mode=dict(
                shared_payload.get("_sampling_generation_mode")
                or shared_payload.get("generation_mode")
                or {}
            ),
            parent_candidates=parent_candidates,
        )
        if not inheritance_enabled:
            for audit_payload in (generation_policy_audit, sampling_policy_audit):
                audit_payload.update(
                    {
                        "parent_candidate_ids": [],
                        "parent_selection_enabled": False,
                        "parent_algorithm_specs_exposed_to_llm": False,
                        "forbid_exact_parent_replay": False,
                        "increase_parent_diversity": False,
                        "evolutionary_inheritance_enabled": False,
                    }
                )
        candidate_id = f"generation_{generation}_candidate_{candidate_slot}"
        requests.append(
            {
                "base_payload": {
                    **shared_payload,
                    "parent_specs": parent_candidates,
                    "candidate_id": candidate_id,
                    "candidate_slot": candidate_slot,
                    "candidate_role": role_name,
                    "candidate_role_objective": role_config["objective"],
                    "temperature": float(
                        sampling_policy_audit["effective_temperature"]
                    ),
                    "generation_policy_audit": generation_policy_audit,
                },
                "role_config": role_config,
                "maximum_attempts": 3,
            }
        )
    slot_results, children = _generate_slot_batch(
        requests=requests,
        design_agent=design_agent,
        problem=problem,
        budget=budget,
        capabilities=capabilities,
        historical_references=historical_references,
        maximum_concurrency=maximum_concurrency,
        initialization_mode=False,
        on_candidate_ready=on_candidate_ready,
    )
    traces: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    duplicate_count = 0
    retry_count = 0
    for request, slot in zip(requests, slot_results):
        candidate_id = request["base_payload"]["candidate_id"]
        role_name = request["base_payload"]["candidate_role"]
        traces.extend(slot["traces"])
        audits.extend(slot["candidate_audit"])
        failures.extend(slot["failures"])
        duplicate_count += int(slot["duplicate_count"])
        retry_count += int(slot["retry_count"])
        if slot["child"] is None:
            failures.append(
                {
                    "candidate_id": candidate_id,
                    "candidate_role": role_name,
                    "reason": "role_exhausted_three_attempts",
                    "generation": generation,
                    "attempt_failures": json_safe(slot["failures"]),
                }
            )
            continue
    return {
        "children": children,
        "rejected_children": failures,
        "feedback_analysis": (
            (
                "All roles used the same global low-fidelity Top-3 parent context."
                if bool(shared_payload.get("execution_feedback", True))
                else "Execution outcomes were withheld; all roles used the same unordered parent AlgorithmSpecs."
            )
            if inheritance_enabled
            else (
                "All roles used knowledge, aggregate execution feedback, and run-scoped posterior summaries; "
                "no parent AlgorithmSpec or parent ID was supplied."
            )
        ),
        "strategy_summary": (
            f"Attempt all {EVOLUTION_TARGET_CANDIDATES} "
            + ("Top-3-driven evolutionary" if inheritance_enabled else "parent-free independent proposal")
            + " roles; each role "
            f"has at most three attempts. Return all accepted candidates to the controller, "
            f"which performs bounded LLM refill until at least {MINIMUM_GENERATION_CANDIDATES} "
            "valid candidates exist."
        ),
        "generation_mode": (
            "evolution" if inheritance_enabled else "independent_search"
        ),
        "proposal_count": len(configured_role_plan),
        "accepted_count": len(children),
        "refill_rounds": 0,
        "llm_calls": len(traces),
        "traces": traces,
        "candidate_audit": audits,
        "duplicate_count": duplicate_count,
        "retry_count": retry_count,
        "role_plan": list(configured_role_plan),
        "evolutionary_inheritance_enabled": inheritance_enabled,
        "parent_algorithm_specs_exposed_to_llm": bool(
            inheritance_enabled and parent_candidates
        ),
        "resolved_design_context": deepcopy(shared_payload.get("resolved_design_context") or {}),
    }


def _generate_training_recovery_candidate(
    *,
    design_agent: DesignAgent,
    shared_payload: dict[str, Any],
    generation: int,
    recovery_index: int,
    parent_candidates: list[dict[str, Any]],
    current_candidates: list[dict[str, Any]],
    generation_records: list[dict[str, Any]],
    problem: Any,
    budget: dict[str, int],
    capabilities: dict[str, Any],
    objective: str,
) -> dict[str, Any]:
    """Ask the LLM for a replacement until one valid candidate is available."""

    inheritance_enabled = bool(
        shared_payload.get("evolutionary_inheritance_enabled", True)
    )
    if generation == 0:
        role_plan = GENERATION_ZERO_ROLE_PLAN
        role_definitions = GENERATION_ZERO_ROLE_DEFINITIONS
        candidate_slot = len(GENERATION_ZERO_ROLE_PLAN) + recovery_index
    else:
        generation_mode = dict(shared_payload.get("generation_mode") or {})
        role_plan = (
            (
                ESCAPE_CANDIDATE_ROLE_PLAN
                if str(generation_mode.get("state") or "").lower() == "escape"
                else NEW_CANDIDATE_ROLE_PLAN
            )
            if inheritance_enabled
            else INDEPENDENT_PROPOSAL_ROLE_PLAN
        )
        role_definitions = CANDIDATE_ROLE_DEFINITIONS
        candidate_slot = EVOLUTION_TARGET_CANDIDATES + recovery_index
    role_name = role_plan[(recovery_index - 1) % len(role_plan)]
    role_config = (
        candidate_role_definition(
            role_name,
            execution_feedback=bool(
                shared_payload.get("execution_feedback", True)
            ),
        )
        if generation > 0
        else role_definitions[role_name]
    )
    if role_config is None:
        raise ValueError(f"Unknown configured candidate role: {role_name}")
    execution_feedback = bool(shared_payload.get("execution_feedback", True))
    candidate_id = (
        f"generation_{generation}_recovery_candidate_{recovery_index}"
        if execution_feedback
        else f"generation_{generation}_candidate_{candidate_slot}"
    )
    if generation_records:
        required_candidates = MINIMUM_SUCCESSFUL_CANDIDATES
        shortfall_reason = "successful_candidate_shortfall"
    else:
        required_candidates = (
            MINIMUM_GENERATION_ZERO_CANDIDATES
            if generation == 0
            else MINIMUM_GENERATION_CANDIDATES
        )
        shortfall_reason = "candidate_pool_shortfall"
    successful = [
        item
        for item in generation_records
        if item.get("training_success") and math.isfinite(_metric_value(item, objective))
    ]
    failed = [item for item in generation_records if item not in successful]
    recovery_feedback = {
        "contract_version": "1.0",
        "source_stage": "training_evaluation",
        "target_stage": "algorithm_generation",
        "source_generation": generation,
        "target_generation": generation,
        "stagnated": False,
        "training_shortfall": bool(generation_records)
        and len(successful) < required_candidates,
        "reflection": {
            "summary": "The current generation has too few successfully trained and evaluated candidates."
        },
        "directives": [
            "Repair the observed build, training, evaluation, or metric failures.",
            "Do not reproduce a failed executable AlgorithmSpec unchanged.",
            "Prefer a conservative trainable design when the failure evidence is inconclusive.",
        ],
        "successful_candidates": [
            _candidate_feedback_view(item)
            for item in successful[:EVOLUTION_PARENT_COUNT]
        ],
        "failed_candidates": [_candidate_feedback_view(item) for item in failed[-8:]],
        "population": [],
        "requirements": [
            "Generate a replacement in the same generation.",
            "Address the supplied failure reasons in executable fields.",
        ],
    }
    if not bool(shared_payload.get("execution_feedback", True)):
        recovery_feedback = {}
    historical_references = (
        [
            {
                "candidate_id": item.get("candidate_id"),
                "candidate_role": (
                    f"previous_rank_{item.get('rank')}"
                    if bool(shared_payload.get("execution_feedback", True))
                    else "unordered_previous_parent"
                ),
                "algorithm_spec": item.get("algorithm_spec"),
            }
            for item in parent_candidates[:EVOLUTION_PARENT_COUNT]
        ]
        if inheritance_enabled
        else []
    )
    traces: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    duplicate_count = 0
    retry_count = 0
    refill_round = 0
    while True:
        refill_round += 1
        if refill_round > MAX_TRAINING_RECOVERY_REFILL_ROUNDS:
            raise RuntimeError(
                "Training-recovery generation exhausted its bounded LLM refill budget: "
                f"generation={generation}, recovery_index={recovery_index}, "
                f"refill_rounds={refill_round - 1}, "
                f"failures={json.dumps(_compact_failure_summaries(failures), ensure_ascii=False)}"
            )
        slot = _generate_candidate_slot(
            design_agent=design_agent,
            base_payload={
                **shared_payload,
                "parent_specs": parent_candidates,
                "candidate_id": candidate_id,
                "candidate_slot": candidate_slot,
                "candidate_role": role_name,
                "candidate_role_objective": role_config["objective"],
                "candidate_diversity_axis": role_config.get("primary_axis"),
                "temperature": float(role_config["default_temperature"]),
                "proposal_type": (
                    "candidate_shortfall_recovery"
                    if execution_feedback
                    else (
                        "initialization_design"
                        if generation == 0
                        else "evolutionary_design"
                    )
                ),
                "generation_feedback": recovery_feedback,
                "candidate_refill_round": refill_round,
                "generation_retry": {
                    "reason": shortfall_reason,
                    "successful_candidates": len(successful),
                    "required_candidates": required_candidates,
                    "failed_candidate_ids": [item.get("spec_id") for item in failed[-8:]],
                    "instruction": (
                        "Generate a new complete candidate because too few candidates survived training and evaluation. "
                        "Use the supplied failure feedback to avoid the observed failure modes."
                    ),
                } if bool(shared_payload.get("execution_feedback", True)) else {},
            },
            role_config=role_config,
            problem=problem,
            budget=budget,
            capabilities=capabilities,
            current_references=current_candidates,
            historical_references=historical_references,
        )
        traces.extend(slot["traces"])
        audits.extend(slot["candidate_audit"])
        failures.extend(slot["failures"])
        duplicate_count += int(slot["duplicate_count"])
        retry_count += int(slot["retry_count"])
        child = slot["child"]
        if child is None:
            continue
        child["training_recovery_index"] = recovery_index
        child["candidate_refill_round"] = refill_round
        return {
            "child": child,
            "rejected_children": failures,
            "llm_calls": len(traces),
            "traces": traces,
            "candidate_audit": audits,
            "duplicate_count": duplicate_count,
            "retry_count": retry_count,
            "refill_rounds": refill_round,
        }


def _merge_variation_result(
    variation_result: dict[str, Any], recovery_result: dict[str, Any]
) -> None:
    """Merge recovery-call accounting into the generation variation audit."""

    for key in ("traces", "candidate_audit", "rejected_children"):
        variation_result.setdefault(key, []).extend(recovery_result.get(key) or [])
    for key in ("llm_calls", "duplicate_count", "retry_count", "refill_rounds"):
        variation_result[key] = int(variation_result.get(key) or 0) + int(
            recovery_result.get(key) or 0
        )
    variation_result["accepted_count"] = int(
        variation_result.get("accepted_count") or 0
    ) + 1
    variation_result["training_recovery_count"] = int(
        variation_result.get("training_recovery_count") or 0
    ) + 1


def _evaluate_candidate_impl(
    *, candidate_id: str, spec: dict[str, Any], generation: int, operation: str,
    problem: Any, problem_id: str, budget: dict[str, int], capabilities: dict[str, Any],
    builder: PINNBuilder, device: str, seed: int, train_candidate: bool,
    checkpoint_root: Path | None, generation_feedback: dict[str, Any] | None = None,
    training_iteration_limit: int | None = None, resume_checkpoint: Path | None = None,
    history_interval: int = 1, reference_mse_interval: int = 0,
    reference_mse_checkpoints: list[int] | tuple[int, ...] | None = None,
    design_algorithm_spec: dict[str, Any] | None = None,
    runtime_scaling_audit: dict[str, Any] | None = None,
    trainer_capabilities_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    augment = getattr(problem, "augment_training_algorithm_spec", None)
    if callable(augment):
        spec = augment(spec)
    candidate_timer = time.perf_counter()
    feedback_source_generation = (generation_feedback or {}).get("source_generation")
    base = {
        "problem_id": problem_id,
        "spec_id": candidate_id,
        "generation": generation,
        "proposal_strategy": operation,
        "budget": budget,
        "seed": seed,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "metrics": {},
        "training_success": False,
        "failure_reason": None,
        "reflection": {},
        "feedback_required": feedback_source_generation is not None,
        "feedback_source_generation": feedback_source_generation,
        "feedback_acknowledged": False,
        "feedback_directives_applied": [],
        "algorithm_spec_raw": deepcopy(spec) if isinstance(spec, dict) else spec,
        "training_started_at": datetime.now().isoformat(timespec="seconds"),
        "trainer_capabilities_snapshot": deepcopy(
            trainer_capabilities_snapshot
            or capabilities.get("trainer_capabilities_snapshot")
            or {}
        ),
        "runtime_scaled_policy_summary": deepcopy(
            runtime_scaling_audit or {}
        ),
        "adaptive_policy_spec": adaptive_policy_spec(
            design_algorithm_spec or spec or {}
        ),
        "adaptive_policy_warnings": [],
        "runtime_physics_requirements": json_safe(
            (
                problem.runtime_physics_requirements()
                if callable(
                    getattr(problem, "runtime_physics_requirements", None)
                )
                else {}
            )
        ),
    }
    validation = validate_algorithm_spec(spec, problem, budget, capabilities)
    if not validation.valid:
        return {
            **base,
            "status": "validation_failed",
            "normalized_algorithm_spec": validation.normalized_spec,
            "validation_report": validation.to_dict(),
            "failure_reason": [asdict(issue) for issue in validation.errors],
        }

    normalized = validation.normalized_spec
    base.update(_candidate_contract_audit(problem, normalized))
    design_spec = deepcopy(design_algorithm_spec or normalized)
    base["adaptive_policy_spec"] = adaptive_policy_spec(design_spec)
    base["adaptive_policy_warnings"] = [
        asdict(issue) for issue in validation.warnings
    ]
    for field_path, clamp in sorted(
        dict((runtime_scaling_audit or {}).get("clamped_fields") or {}).items()
    ):
        base["adaptive_policy_warnings"].append(
            {
                "path": str(field_path),
                "code": "RUNTIME_POLICY_CLAMPED",
                "message": str((clamp or {}).get("reason") or "fidelity budget bound"),
            }
        )
    try:
        built = builder.build(
            normalized,
            problem,
            {
                "device": device,
                "maximum_sampling_points": budget.get("maximum_sampling_points"),
                "constraint_resample_interval": budget.get(
                    "constraint_resample_interval", 50
                ),
            },
        )
    except BuildError as exc:
        return {
            **base,
            "status": "build_failed",
            "normalized_algorithm_spec": normalized,
            "validation_report": validation.to_dict(),
            "failure_reason": [exc.report.to_dict()],
        }

    metadata = normalized["generation_metadata"]
    applied_directives = metadata.get("reflection_directives_applied") or []
    feedback_acknowledged = (
        feedback_source_generation is not None
        and metadata.get("feedback_source_generation") == feedback_source_generation
        and isinstance(applied_directives, list)
        and bool(applied_directives)
    )
    base.update(
        {
            "normalized_algorithm_spec": normalized,
            "parent_spec_ids": metadata.get("parent_spec_ids") or [],
            "validation_report": validation.to_dict(),
            "feedback_acknowledged": feedback_acknowledged,
            "feedback_directives_applied": applied_directives if isinstance(applied_directives, list) else [],
        }
    )
    if not train_candidate:
        return {**base, "status": "training_skipped_budget", "failure_reason": "generation_training_limit_reached"}
    candidate_evaluator = OpenSpecEvaluator(problem, device)
    reference_probe = getattr(candidate_evaluator, "evaluate_mse", None)
    trainer = OpenSpecTrainer(
        built,
        checkpoint_dir=checkpoint_root,
        history_interval=history_interval,
        reference_mse_probe=(lambda: reference_probe(built.model)) if callable(reference_probe) else None,
        reference_mse_interval=reference_mse_interval if callable(reference_probe) else 0,
        reference_mse_checkpoints=(
            reference_mse_checkpoints if callable(reference_probe) else None
        ),
        # Reference MSE is an observational benchmark diagnostic only. Model
        # selection remains label-free and uses summed training loss.
        select_best_reference_mse_checkpoint=False,
        component_gradient_diagnostics_enabled=bool(
            budget.get("component_gradient_diagnostics_enabled", True)
        ),
        gradient_imbalance_threshold=float(
            budget.get("component_gradient_imbalance_threshold", 100.0)
        ),
        gradient_conflict_threshold=float(
            budget.get("component_gradient_conflict_threshold", -0.1)
        ),
        maximum_gradient_components=int(
            budget.get("maximum_gradient_diagnostic_components", 12)
        ),
        maximum_conflicting_pairs=int(
            budget.get("maximum_gradient_conflicting_pairs", 5)
        ),
        runtime_scaled_policy_summary=deepcopy(
            runtime_scaling_audit or {}
        ),
    )
    if resume_checkpoint is not None:
        trainer.load_checkpoint(resume_checkpoint)
    training = trainer.train() if training_iteration_limit is None else trainer.train(maximum_iterations=training_iteration_limit)
    base["training_report"] = training.to_dict()
    base["adaptive_control_summary"] = deepcopy(
        training.adaptive_control_summary
    )
    base["adaptive_compute_budget_summary"] = deepcopy(
        training.adaptive_compute_budget_summary
    )
    if not training.success:
        return {**base, "status": "training_failed", "failure_reason": training.failure_reason}
    try:
        metrics = candidate_evaluator.evaluate(built.model, training)
    except Exception as exc:
        return {
            **base,
            "status": "evaluation_failed",
            "metrics": {},
            "training_success": False,
            "failure_reason": f"{type(exc).__name__}: {exc}",
        }
    wave_candidate_summary: dict[str, Any] = {}
    if problem_id == "wave_1d":
        sampling_snapshot = training.final_sampling_snapshot or {}
        wave_candidate_summary = {
            "initial_condition_points": int(
                sampling_snapshot.get("initial_condition_points") or 0
            ),
            "constraint_breakdown": json_safe(
                sampling_snapshot.get("constraint_breakdown") or []
            ),
            "initial_displacement_mse": metrics.get(
                "initial_displacement_mse"
            ),
            "initial_velocity_mse": metrics.get("initial_velocity_mse"),
            "amplitude_ratio": metrics.get("amplitude_ratio"),
            "mean_phase_error": metrics.get("mean_phase_error"),
        }
    if metrics.get("solution_collapse_detected"):
        return {
            **base,
            **wave_candidate_summary,
            "status": "solution_collapse",
            "metrics": metrics,
            "training_success": False,
            "failure_reason": {
                "reason": "solution_collapse_detected",
                "collapse_type": metrics.get("solution_collapse_type"),
                "prediction_reference_std_ratio": metrics.get(
                    "prediction_reference_std_ratio"
                ),
                "relative_l2_error": metrics.get("relative_l2_error"),
            },
            "training_ended_at": datetime.now().isoformat(timespec="seconds"),
            "total_wall_clock_runtime": time.perf_counter() - candidate_timer,
        }
    return {
        **base,
        **wave_candidate_summary,
        "status": "trained",
        "metrics": metrics,
        "training_success": True,
        "failure_reason": None,
        "training_ended_at": datetime.now().isoformat(timespec="seconds"),
        "total_wall_clock_runtime": time.perf_counter() - candidate_timer,
    }


def _evaluate_candidate(
    *, candidate_id: str, spec: dict[str, Any], generation: int, operation: str,
    problem: Any, problem_id: str, budget: dict[str, int], capabilities: dict[str, Any],
    builder: PINNBuilder, device: str, seed: int, train_candidate: bool,
    checkpoint_root: Path | None, generation_feedback: dict[str, Any] | None = None,
    training_iteration_limit: int | None = None, resume_checkpoint: Path | None = None,
    history_interval: int = 1, reference_mse_interval: int = 0,
    reference_mse_checkpoints: list[int] | tuple[int, ...] | None = None,
    design_algorithm_spec: dict[str, Any] | None = None,
    runtime_scaling_audit: dict[str, Any] | None = None,
    trainer_capabilities_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate one candidate and always release process-local accelerator state."""

    candidate_timer = time.perf_counter()
    started_at = datetime.now().isoformat(timespec="seconds")
    try:
        try:
            return _evaluate_candidate_impl(
                candidate_id=candidate_id,
                spec=spec,
                generation=generation,
                operation=operation,
                problem=problem,
                problem_id=problem_id,
                budget=budget,
                capabilities=capabilities,
                builder=builder,
                device=device,
                seed=seed,
                train_candidate=train_candidate,
                checkpoint_root=checkpoint_root,
                generation_feedback=generation_feedback,
                training_iteration_limit=training_iteration_limit,
                resume_checkpoint=resume_checkpoint,
                history_interval=history_interval,
                reference_mse_interval=reference_mse_interval,
                reference_mse_checkpoints=reference_mse_checkpoints,
                design_algorithm_spec=design_algorithm_spec,
                runtime_scaling_audit=runtime_scaling_audit,
                trainer_capabilities_snapshot=trainer_capabilities_snapshot,
            )
        except Exception as exc:
            return {
                "problem_id": problem_id,
                "spec_id": candidate_id,
                "generation": generation,
                "proposal_strategy": operation,
                "budget": deepcopy(budget),
                "seed": seed,
                "timestamp": started_at,
                "status": "candidate_execution_failed",
                "metrics": {},
                "training_success": False,
                "failure_reason": f"{type(exc).__name__}: {exc}",
                "algorithm_spec_raw": deepcopy(spec),
                "normalized_algorithm_spec": deepcopy(spec),
                "training_started_at": started_at,
                "training_ended_at": datetime.now().isoformat(timespec="seconds"),
                "total_wall_clock_runtime": time.perf_counter() - candidate_timer,
            }
    finally:
        _release_candidate_resources(problem, device)


def _release_candidate_resources(problem: Any, device: str) -> None:
    """Best-effort cleanup between independent candidate trainings."""

    clear = getattr(problem, "clear_autodiff_cache", None)
    if callable(clear):
        try:
            clear()
        except Exception:
            # Cleanup is advisory and must not replace the candidate result.
            pass
    gc.collect()
    if not str(device).startswith("cuda"):
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        # Cleanup must not replace the candidate's real result with a secondary
        # CUDA teardown error, especially after an out-of-memory failure.
        pass


def _train_low_fidelity_child(
    child: dict[str, Any],
    *,
    child_index: int,
    context: dict[str, Any],
) -> dict[str, Any]:
    """Train one candidate on the caller thread; never overlaps GPU work."""

    generation = int(context["generation"])
    generation_seed = int(context["generation_seed"])
    set_global_seed(generation_seed)
    proposal_generation_mode = str(
        child.get("generation_mode")
        or context.get("variation_generation_mode")
        or "evolution"
    )
    candidate_id = _generation_candidate_id(generation, child_index, child)
    output_dir = Path(context["output_dir"])
    output_detail = str(context["output_detail"])
    budget = context["budget"]
    low_fidelity_config = context["low_fidelity_config"]
    generation_recommended_iterations = int(
        context["generation_recommended_iterations"]
    )
    record = _evaluate_candidate(
        candidate_id=candidate_id,
        spec=child.get("algorithm_spec"),
        generation=generation,
        operation=proposal_generation_mode,
        problem=context["problem"],
        problem_id=str(context["problem_id"]),
        budget=budget,
        capabilities=context["capabilities"],
        builder=context["builder"],
        device=str(context["device"]),
        seed=generation_seed,
        train_candidate=True,
        checkpoint_root=(
            output_dir / "checkpoints" / candidate_id
            if output_detail == "diagnostic"
            else None
        ),
        generation_feedback=context["input_feedback"],
        training_iteration_limit=None,
        resume_checkpoint=None,
        history_interval=resolve_history_interval(
            int(budget["required_total_iterations"]),
            output_detail=output_detail,
            configured_interval=context.get("training_history_interval"),
        ),
        reference_mse_interval=max(
            1, generation_recommended_iterations // 10
        ),
        reference_mse_checkpoints=list(
            low_fidelity_config["checkpoint_iterations"]
        ),
        design_algorithm_spec=deepcopy(
            child.get("search_algorithm_spec")
            or child.get("algorithm_spec")
            or {}
        ),
        runtime_scaling_audit=deepcopy(
            (child.get("optimization_budget_audit") or {}).get(
                "adaptive_policy_scaling"
            )
            or {}
        ),
        trainer_capabilities_snapshot=context["trainer_capabilities_snapshot"],
    )
    record = _apply_low_fidelity_proxy(
        record,
        checkpoint_iterations=list(
            low_fidelity_config["checkpoint_iterations"]
        ),
    )
    generation_mode_name = str(context["generation_mode_name"])
    previous_parent_ids = list(context["previous_parent_ids"])
    search_state = context["search_state"]
    seed = int(context["run_seed"])
    ranking_fidelity = str(context["ranking_fidelity"])
    record.update(
        {
            "generation_index": generation,
            "generation_iteration_recommendation": generation_recommended_iterations,
            "iteration_recommendation_is_advisory": False,
            "generation_kind": "low_fidelity",
            "ranking_fidelity": ranking_fidelity,
            "experiment_contract_version": EXPERIMENT_CONTRACT_VERSION,
            "fidelity": "low",
            "generation_mode": generation_mode_name,
            "proposal_generation_mode": proposal_generation_mode,
            "training_iterations": int(budget["required_total_iterations"]),
            "fresh_initialization": True,
            "resumed_from_checkpoint": False,
            "escape_parent_context": (
                previous_parent_ids if generation_mode_name == "escape" else []
            ),
            "escape_rationale": (
                str(
                    (child.get("proposal_metadata") or {}).get(
                        "design_rationale"
                    )
                    or child.get("design_rationale")
                    or ""
                )
                if generation_mode_name == "escape"
                else ""
            ),
            "pre_escape_global_best_mse": (
                search_state.get("pre_escape_global_best_mse")
                if generation_mode_name == "escape"
                else None
            ),
            "parent_spec_ids": list(child.get("parent_ids") or []),
            "design_intent": child.get("design_intent") or "",
            "difference_summary": child.get("difference_summary") or "",
            "variation_reason": child.get("variation_reason") or "",
            "heritage_explanation": child.get("heritage_explanation") or {},
            "expected_improvements": child.get("expected_improvements") or [],
            "candidate_slot": child.get("candidate_slot"),
            "candidate_role": child.get("candidate_role"),
            "proposal_metadata": deepcopy(child.get("proposal_metadata") or {}),
            "generation_policy_audit": deepcopy(
                child.get("generation_policy_audit") or {}
            ),
            "variation_strategy": child.get("variation_strategy"),
            "inherited_strengths": list(child.get("inherited_strengths") or []),
            "major_changes": list(child.get("major_changes") or []),
            "design_rationale": child.get("design_rationale") or "",
            "proposal_validation": deepcopy(
                child.get("proposal_validation") or {}
            ),
            "role_design_objective": child.get("role_design_objective") or "",
            "primary_diversity_axis": child.get("primary_diversity_axis"),
            "nearest_candidate_id": child.get("nearest_candidate_id"),
            "nearest_candidate_role": child.get("nearest_candidate_role"),
            "nearest_candidate_distance": child.get("nearest_candidate_distance"),
            "normalized_executable_distance": child.get(
                "normalized_executable_distance"
            ),
            "overlapping_modules": child.get("overlapping_modules") or [],
            "different_modules": child.get("different_modules") or [],
            "similarity_status": child.get("similarity_status") or "not_audited",
            "diversity_retry_count": int(
                child.get("diversity_retry_count") or 0
            ),
            "exact_duplicate": bool(child.get("exact_duplicate")),
            "exact_duplicate_after_retry": bool(
                child.get("exact_duplicate_after_retry")
            ),
            "accepted_despite_similarity": bool(
                child.get("accepted_despite_similarity")
            ),
            "local_design_variant": bool(child.get("local_design_variant")),
            "diversity_retry_performed": bool(
                child.get("diversity_retry_performed")
            ),
            "diversity_warning": bool(child.get("diversity_warning")),
            "population_role": child.get("population_role") or "llm_generated",
            "source_candidate_id": child.get("source_candidate_id"),
            "fresh_training": bool(child.get("fresh_training", True)),
            "historical_checkpoint_reused": False,
            "historical_metrics_reused": False,
            "algorithm_spec": record.get("normalized_algorithm_spec") or {},
            "search_algorithm_spec": deepcopy(
                child.get("search_algorithm_spec")
                or record.get("normalized_algorithm_spec")
                or {}
            ),
            "algorithm_spec_raw": deepcopy(
                child.get("algorithm_spec_raw")
                if child.get("algorithm_spec_raw") is not None
                else record.get("algorithm_spec_raw")
            ),
            "training_seed": generation_seed,
            "run_seed": seed,
            "model_initialization_seed": seed,
            "sampling_seed": seed,
            "evaluation_seed": int(context.get("evaluation_seed") or 0),
            "llm_seed": None,
            "llm_seed_supported": False,
            "training_budget": {
                "generation_iteration_recommendation": generation_recommended_iterations,
                "iteration_recommendation_is_advisory": False,
                "generation_kind": "low_fidelity",
                "ranking_fidelity": ranking_fidelity,
            },
            "optimization_budget_audit": child.get(
                "optimization_budget_audit"
            )
            or {},
            "formal_status": (
                "completed" if record.get("training_success") else "failed"
            ),
            "generation_metrics": json_safe(record.get("metrics") or {}),
            "generation_checkpoint": (
                str(
                    output_dir
                    / "checkpoints"
                    / candidate_id
                    / "final_checkpoint.pt"
                )
                if record.get("training_success")
                and output_detail == "diagnostic"
                else None
            ),
            "generation_rank": None,
        }
    )
    if generation == 0:
        for field in (
            "nearest_candidate_id",
            "nearest_candidate_role",
            "nearest_candidate_distance",
            "normalized_executable_distance",
            "overlapping_modules",
            "different_modules",
            "similarity_status",
            "diversity_retry_count",
            "exact_duplicate",
            "exact_duplicate_after_retry",
            "accepted_despite_similarity",
            "local_design_variant",
            "diversity_retry_performed",
            "diversity_warning",
        ):
            record.pop(field, None)
    _write_candidate_runtime_snapshot(
        output_dir,
        record,
        output_detail=output_detail,
    )
    return record


def _experiment_budget(
    config: dict[str, Any],
    *,
    generation_index: int,
    total_generations: int,
) -> dict[str, Any]:
    contract = resolve_experiment_contract(config.get("experiment_contract"))
    scaling = contract["low_fidelity_schedule_scaling"]
    configured_iterations = int(
        contract["search_control"]["low_fidelity_iterations"]
    )
    if config.get("_test_mode") and config.get("_test_generation_iterations") is not None:
        configured_iterations = max(1, int(config["_test_generation_iterations"]))
    configured_sampling_limit = config.get("maximum_sampling_points")
    budget = {
        "recommended_total_iterations": max(0, int(configured_iterations)),
        "required_total_iterations": max(0, int(configured_iterations)),
        "maximum_total_iterations": max(0, int(configured_iterations)),
        "iteration_recommendation_is_advisory": False,
        "fidelity": "low",
        "schedule_scaling": deepcopy(scaling),
        "maximum_sampling_points": (
            None
            if configured_sampling_limit is None
            else max(0, int(configured_sampling_limit))
        ),
        "maximum_model_parameters": max(1, int(config.get("maximum_model_parameters", 2_000_000))),
        "constraint_resample_interval": max(
            0, int(config.get("constraint_resample_interval", 50))
        ),
        "component_gradient_diagnostics_enabled": bool(
            config.get("component_gradient_diagnostics_enabled", True)
        ),
        "component_gradient_imbalance_threshold": max(
            1.0,
            float(config.get("component_gradient_imbalance_threshold", 100.0)),
        ),
        "component_gradient_conflict_threshold": max(
            -1.0,
            min(
                1.0,
                float(config.get("component_gradient_conflict_threshold", -0.1)),
            ),
        ),
        "maximum_gradient_diagnostic_components": max(
            2,
            int(config.get("maximum_gradient_diagnostic_components", 12)),
        ),
        "maximum_gradient_conflicting_pairs": max(
            1,
            int(config.get("maximum_gradient_conflicting_pairs", 5)),
        ),
    }
    return budget


def _high_fidelity_budget(
    config: dict[str, Any], *, experiment_contract: dict[str, Any]
) -> dict[str, Any]:
    """Build the exact independent final-review budget."""

    budget = _experiment_budget(
        config,
        generation_index=0,
        total_generations=int(
            experiment_contract["search_control"][
                "max_low_fidelity_generations"
            ]
        ),
    )
    iterations = int(
        experiment_contract["search_control"]["high_fidelity_iterations"]
    )
    budget.update(
        {
            "recommended_total_iterations": iterations,
            "required_total_iterations": iterations,
            "maximum_total_iterations": iterations,
            "iteration_recommendation_is_advisory": False,
            "fidelity": "high",
            "evaluation_stage": "final_high_fidelity_evaluation",
        }
    )
    return budget


def _stable_contract_hash(value: Any) -> str:
    payload = json.dumps(
        json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _candidate_contract_audit(
    problem: Any,
    normalized_spec: dict[str, Any],
) -> dict[str, Any]:
    cached_grid_hash = getattr(
        problem, "_llm4pinn_evaluation_grid_hash", None
    )
    if not cached_grid_hash:
        try:
            points = problem.evaluation_samples(device="cpu")
            digest = hashlib.sha256()
            digest.update(str(tuple(points.shape)).encode("ascii"))
            digest.update(str(points.dtype).encode("ascii"))
            digest.update(
                points.detach().cpu().contiguous().numpy().tobytes()
            )
            cached_grid_hash = digest.hexdigest()
            setattr(
                problem,
                "_llm4pinn_evaluation_grid_hash",
                cached_grid_hash,
            )
        except Exception:
            cached_grid_hash = None
    problem_spec = problem.get_spec()
    metric_contract = {
        "problem_id": str(problem_spec.problem_id),
        "output_variables": list(problem_spec.output_variables),
        "evaluation_scope": getattr(problem, "evaluation_scope", None),
        "evaluation_size": getattr(problem, "evaluation_size", None),
        "evaluation_grid_hash": cached_grid_hash,
        "primary_metric": PRIMARY_RANKING_METRIC,
        "metric_reduction": "pointwise_global_mean",
    }
    runtime_policy = {
        "network": normalized_spec.get("network"),
        "sampling": normalized_spec.get("sampling"),
        "loss": normalized_spec.get("loss"),
        "optimization": normalized_spec.get("optimization"),
        "training": normalized_spec.get("training"),
        "constraint_enforcement": normalized_spec.get(
            "constraint_enforcement"
        ),
    }
    optimizer_signature = [
        {
            "optimizer": phase.get("optimizer"),
            "iterations": int(phase.get("iterations") or 0),
            "scheduler": (phase.get("scheduler") or {}).get("name"),
        }
        for phase in (
            normalized_spec.get("optimization", {}).get("phases") or []
        )
    ]
    return {
        "metric_contract_hash": _stable_contract_hash(metric_contract),
        "evaluation_grid_hash": cached_grid_hash,
        "metric_contract": metric_contract,
        "runtime_policy_hash": _stable_contract_hash(runtime_policy),
        "optimizer_phase_signature": optimizer_signature,
    }


def _enforce_high_fidelity_validation_and_checkpoint(
    spec: dict[str, Any],
    *,
    high_iterations: int,
) -> dict[str, Any]:
    result = deepcopy(spec)
    training = result.setdefault("training", {})
    extra = training.setdefault("extra_parameters", {})
    interval = max(1, min(500, int(high_iterations) // 20))
    physics = dict(extra.get("physics_validation") or {})
    physics.update(
        {
            "enabled": True,
            "strategy": "fixed_component_probe",
            "evaluation_interval": interval,
            "normalization": "initial_probe_value",
            "category_aggregation": "median_worst_blend",
            "median_weight": 0.7,
            "worst_weight": 0.3,
            "points_per_component": max(
                128, int(physics.get("points_per_component") or 0)
            ),
            "maximum_total_points": max(
                4096, int(physics.get("maximum_total_points") or 0)
            ),
            # Guard second-order optimization with the fixed, label-free
            # component probe. The trainer restores the best guarded model
            # and reallocates the remaining L-BFGS budget to Adam.
            "second_order_degradation_ratio": 1.1,
            "second_order_guard_metric": "physics_validation_score",
        }
    )
    physics.pop("reference_mse", None)
    extra["physics_validation"] = physics
    best = dict(extra.get("best_checkpoint") or {})
    for obsolete_field in (
        "minimum_improvement",
        "critical_component_degradation_ratio",
        "critical_component_score_floor",
    ):
        best.pop(obsolete_field, None)
    best.update(
        {
            "enabled": True,
            "selection_metric": "training_loss",
            "final_model_policy": "best_train_loss",
            "evaluation_interval": 100,
        }
    )
    extra["best_checkpoint"] = best
    return result


def _fidelity_checkpoint_iterations(
    spec: dict[str, Any],
    total_iterations: int,
) -> list[int]:
    checkpoints = {
        min(int(total_iterations), value)
        for value in (LOW_FIDELITY_ITERATIONS, int(total_iterations))
        if value > 0
    }
    cursor = 0
    for phase in spec.get("optimization", {}).get("phases") or []:
        cursor += int(phase.get("iterations") or 0)
        if 0 < cursor <= int(total_iterations):
            checkpoints.add(cursor)
    return sorted(checkpoints)


def _history_reference_mse_values(
    training_report: dict[str, Any],
    *,
    lbfgs: bool,
) -> list[float]:
    values: list[float] = []
    for item in training_report.get("history") or []:
        if not isinstance(item, dict):
            continue
        phase_is_lbfgs = (
            "lbfgs" in str(item.get("phase") or "").casefold()
        )
        value = _finite_metric(item.get("reference_mse"))
        if phase_is_lbfgs is lbfgs and value is not None:
            values.append(value)
    return values


def _build_cross_fidelity_audit(
    low_record: dict[str, Any],
    high_record: dict[str, Any],
    high_training_report: dict[str, Any],
) -> dict[str, Any]:
    checkpoints = dict(
        high_training_report.get("reference_mse_checkpoints") or {}
    )
    low_budget_step = int(LOW_FIDELITY_ITERATIONS)
    high_at_low_budget = _finite_metric(checkpoints.get(str(low_budget_step)))
    low_proxy = _finite_metric(
        low_record.get(
            "low_fidelity_aggregate_proxy_mse",
            low_record.get("low_fidelity_proxy_mse"),
        )
    )
    pre_lbfgs = _history_reference_mse_values(
        high_training_report, lbfgs=False
    )
    during_lbfgs = _history_reference_mse_values(
        high_training_report, lbfgs=True
    )
    high_last = _finite_metric(
        high_training_report.get("last_checkpoint_mse")
    )
    selected_mse = _finite_metric(
        (high_record.get("metrics") or {}).get("mse")
    )
    same_metric_contract = bool(
        low_record.get("metric_contract_hash")
        and low_record.get("metric_contract_hash")
        == high_record.get("metric_contract_hash")
    )
    same_seed = (
        int(low_record.get("training_seed", low_record.get("seed", -1)))
        == int(high_record.get("training_seed", high_record.get("seed", -2)))
    )
    same_runtime_policy = bool(
        low_record.get("runtime_policy_hash")
        and low_record.get("runtime_policy_hash")
        == high_record.get("runtime_policy_hash")
    )
    budget_derivation_audited = bool(
        high_record.get("optimization_budget_audit")
    )
    degradation_stage = None
    if during_lbfgs and pre_lbfgs and min(during_lbfgs) > min(pre_lbfgs):
        degradation_stage = "lbfgs"
    elif (
        high_last is not None
        and high_at_low_budget is not None
        and high_last > high_at_low_budget
    ):
        degradation_stage = f"post_step_{low_budget_step}"
    elif (
        low_proxy is not None
        and high_at_low_budget is not None
        and high_at_low_budget > low_proxy
    ):
        degradation_stage = (
            "fresh_start_or_runtime_policy_gap"
            if not same_runtime_policy
            else "paired_prefix_divergence"
        )
    return {
        "low_fidelity_selected_mse": low_proxy,
        "low_fidelity_budget_iterations": low_budget_step,
        "high_fidelity_mse_at_low_fidelity_budget": high_at_low_budget,
        "high_fidelity_best_mse_before_lbfgs": (
            min(pre_lbfgs) if pre_lbfgs else None
        ),
        "high_fidelity_best_mse_during_lbfgs": (
            min(during_lbfgs) if during_lbfgs else None
        ),
        "high_fidelity_last_mse": high_last,
        "high_fidelity_selected_mse": selected_mse,
        "low_high_fidelity_gap_at_low_fidelity_budget": (
            high_at_low_budget - low_proxy
            if high_at_low_budget is not None and low_proxy is not None
            else None
        ),
        "high_fidelity_degradation_ratio": (
            high_last / min(pre_lbfgs + during_lbfgs)
            if high_last is not None
            and pre_lbfgs + during_lbfgs
            and min(pre_lbfgs + during_lbfgs) > 0.0
            else None
        ),
        "metric_contract_hash_match": same_metric_contract,
        "training_seed_match": same_seed,
        "runtime_policy_hash_match": same_runtime_policy,
        "budget_scaled_policy_derivation_audited": (
            budget_derivation_audited
        ),
        "strict_prefix_parity_eligible": bool(
            same_metric_contract and same_seed and same_runtime_policy
        ),
        "paired_budget_comparison_eligible": bool(
            same_metric_contract
            and same_seed
            and (same_runtime_policy or budget_derivation_audited)
        ),
        "degradation_stage": degradation_stage,
        "reference_mse_used_for_training_control": False,
    }


def _rank_generation_formal_candidates(
    records: list[dict[str, Any]],
    *,
    objective: str,
    count: int,
    ranking_fidelity: str | None = None,
) -> list[dict[str, Any]]:
    ranked = sorted(
        [
            item
            for item in records
            if item.get("training_success") and math.isfinite(_metric_value(item, objective))
        ],
        key=lambda item: _metric_value(item, objective),
    )
    generation_index = int(records[0].get("generation", 1)) if records else 1
    required_candidates = MINIMUM_SUCCESSFUL_CANDIDATES
    if len(ranked) < required_candidates:
        raise RuntimeError(
            "Generation produced fewer successful candidates than required: "
            f"successful={len(ranked)}, required={required_candidates}"
        )
    rank_count = min(count, len(ranked))
    ranks = {item.get("spec_id"): rank for rank, item in enumerate(ranked[:rank_count], start=1)}
    updated: list[dict[str, Any]] = []
    for record in records:
        item = dict(record)
        item["generation_rank"] = ranks.get(item.get("spec_id"))
        fidelity = ranking_fidelity or str(item.get("ranking_fidelity") or "full_budget")
        rank_name = "final_rank" if fidelity == "full_budget" else "proxy_rank"
        item[rank_name] = item["generation_rank"]
        item["generation_metrics"] = json_safe(item.get("metrics") or {})
        item["generation_training_report"] = json_safe(item.get("training_report") or {})
        if item.get("training_success") and math.isfinite(_metric_value(item, objective)):
            item["formal_status"] = "completed"
        elif item.get("training_success"):
            item["formal_status"] = "invalid_metrics"
        else:
            item["formal_status"] = "failed"
        updated.append(item)
    return updated


def _resolve_generation_mode(
    generation_feedback: dict[str, Any],
    *,
    mode: str = "normal",
    stagnation_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve prompt policy from the explicit normal/escape state machine."""

    if mode not in {"normal", "escape"}:
        raise ValueError("generation mode must be normal or escape")
    config = dict(
        stagnation_config
        or DEFAULT_EXPERIMENT_CONTRACT["stagnation"]
    )
    if mode == "normal":
        return {
            "state": "normal",
            "stagnation_detected": False,
            "trigger_reason": None,
            "improvement_threshold": float(
                config["best_single_generation_threshold"]
            ),
            "lookback_generations": int(config["best_improvement_window"]),
            "behavior_overrides": {},
        }
    return {
        "state": "escape",
        "stagnation_detected": True,
        "trigger_reason": "three_condition_stagnation_detected",
        "improvement_threshold": float(
            config["best_single_generation_threshold"]
        ),
        "lookback_generations": int(config["best_improvement_window"]),
        "behavior_overrides": {
            "novelty_temperature_delta": 0.15,
            "cross_module_temperature_delta": 0.10,
            "minimum_major_module_changes": 2,
            "forbid_exact_parent_replay": True,
            "require_failed_mechanism_avoidance": True,
            "increase_parent_diversity": True,
        },
        "escape_requirements": [
            "Search stagnation has been detected.",
            "Do not copy or reserve a slot for pure Rank-1 replay.",
            "Do not make only an immaterial numeric-field tweak.",
            "Change at least one major executable module.",
            "Prefer two causally related major module changes.",
            "Explain how the design attempts to leave the current local region.",
            "Return one complete AlgorithmSpec.",
            "Pass the normal registry, validator, and experiment-budget checks.",
        ],
        "recent_failure_evidence": json_safe(
            (generation_feedback or {}).get("failed_candidates") or []
        ),
        "recent_residual_evidence": json_safe(
            (generation_feedback or {}).get("physical_supervision_summary") or {}
        ),
        "recent_similarity_evidence": json_safe(
            (generation_feedback or {}).get("candidate_similarity_summary") or {}
        ),
    }


def _build_generation_policy_audit(
    *,
    role_name: str,
    role_config: dict[str, Any],
    generation_mode: dict[str, Any],
    parent_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    state = str(generation_mode.get("state") or "normal")
    stagnated = state == "escape"
    base_temperature = float(role_config.get("default_temperature") or 0.0)
    delta = (
        max(
            0.15,
            float(role_config.get("stagnation_temperature_delta") or 0.0),
        )
        if stagnated
        else 0.0
    )
    effective_temperature = min(
        MAXIMUM_ROLE_TEMPERATURE,
        base_temperature + delta,
    )
    required_changes = (
        max(
            2,
            int(
                role_config.get("stagnation_minimum_major_module_changes")
                or 1
            ),
        )
        if stagnated
        else 1
    )
    return {
        "role": role_name,
        "generation_mode": state,
        "stagnation_detected": stagnated,
        "base_temperature": base_temperature,
        "temperature_delta": delta,
        "effective_temperature": effective_temperature,
        "maximum_role_temperature": MAXIMUM_ROLE_TEMPERATURE,
        "temperature_reason": "escape" if stagnated else "role_default",
        "parent_candidate_ids": [
            str(item.get("candidate_id"))
            for item in parent_candidates
            if item.get("candidate_id")
        ],
        "required_major_module_changes": required_changes,
        "forbid_exact_parent_replay": True,
        "require_failed_mechanism_avoidance": stagnated,
        "increase_parent_diversity": bool(
            stagnated
            and role_name
            in {"multi_parent_synthesis", "novelty_exploration"}
        ),
        "role_input_evidence": _role_input_evidence(
            role_name,
            parent_candidates,
        ),
        "role_modifiable_modules": list(
            role_config.get("modifiable_modules")
            or role_config.get("major_modules")
            or []
        ),
        "stagnation_requirements": (
            list(generation_mode.get("escape_requirements") or [])
            if stagnated
            else []
        ),
    }


def _role_input_evidence(
    role_name: str,
    parent_candidates: list[dict[str, Any]],
) -> list[str]:
    evidence: list[str] = []
    if role_name == "architecture_optimization_guided_design":
        for parent in parent_candidates:
            feedback = parent.get("optimization_feedback") or {}
            loss_summary = feedback.get("loss_trajectory_summary") or {}
            gradient = feedback.get("gradient_summary") or {}
            if any(value == "plateaued" for value in loss_summary.values()):
                evidence.append("persistent_loss_plateau")
            if gradient.get("gradient_imbalance_detected"):
                evidence.append("component_gradient_imbalance")
            if gradient.get("gradient_conflict_detected"):
                evidence.append("component_gradient_conflict")
            if feedback.get("optimizer_audit"):
                evidence.append("optimizer_audit")
            if feedback.get("mse_probe_history"):
                evidence.append("mse_probe_history")
            if feedback.get("lbfgs_transition_degradation"):
                evidence.append("lbfgs_degradation")
    elif role_name == "physics_constraint_sampling_guided_design":
        for parent in parent_candidates:
            supervision = parent.get("physical_supervision_summary") or {}
            spatial = supervision.get("residual_spatial_summary") or {}
            equations = (spatial.get("equations") or {}).values()
            if any(item.get("localized_error") for item in equations):
                evidence.append("localized_residual_region")
            if any(item.get("boundary_concentration") for item in equations):
                evidence.append("boundary_concentration")
            if any(
                item.get("initial_surface_concentration") for item in equations
            ):
                evidence.append("initial_surface_concentration")
            if supervision.get("residual_distribution"):
                evidence.append("residual_long_tail")
            if supervision.get("final_sampling_snapshot"):
                evidence.append("sampling_snapshot")
    return list(dict.fromkeys(evidence))


def _search_stage(
    generation: int,
    total: int,
    stagnated: bool,
) -> dict[str, Any]:
    if stagnated:
        return {"name": "stagnation", "elite": 0.25, "knowledge": 0.25, "novelty": 0.25, "random": 0.25}
    ratio = generation / max(1, total - 1)
    if ratio < 1 / 3:
        return {"name": "early", "elite": 0.0, "knowledge": 0.35, "novelty": 0.30, "random": 0.35}
    if ratio < 2 / 3:
        return {"name": "middle", "elite": 0.30, "knowledge": 0.30, "novelty": 0.20, "random": 0.20}
    return {"name": "late", "elite": 0.70, "knowledge": 0.15, "novelty": 0.10, "random": 0.05}


def _build_generation_feedback(
    *,
    generation: int,
    reflection: dict[str, Any],
    records: list[dict[str, Any]],
    population_summary: list[dict[str, Any]],
    objective: str,
    stagnation_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    successful = sorted(
        [item for item in records if item.get("training_success")],
        key=lambda item: _metric_value(item, objective),
    )
    failed = [item for item in records if not item.get("training_success")]
    directives = _reflection_directives(reflection)
    if not directives:
        directives = ["Preserve measured strengths and repair executable or numerical failures."]
    stagnation = dict(stagnation_state or {})
    return {
        "contract_version": "1.0",
        "source_stage": "H",
        "target_stage": "D",
        "source_generation": generation,
        "target_generation": generation + 1,
        "stagnated": bool(stagnation.get("stagnation_detected")),
        "stagnation": json_safe(stagnation),
        "reflection": json_safe(reflection),
        "directives": directives,
        "successful_candidates": [
            _candidate_feedback_view(item)
            for item in successful[:EVOLUTION_PARENT_COUNT]
        ],
        "failed_candidates": [_candidate_feedback_view(item) for item in failed[:8]],
        "physical_supervision_summary": [
            {
                "spec_id": item.get("spec_id"),
                "summary": json_safe(
                    item.get("physical_supervision_summary") or {}
                ),
            }
            for item in successful[:EVOLUTION_PARENT_COUNT]
        ],
        "candidate_similarity_summary": [
            {
                "spec_id": item.get("spec_id"),
                "nearest_candidate_id": item.get("nearest_candidate_id"),
                "normalized_executable_distance": item.get(
                    "normalized_executable_distance"
                ),
                "overlapping_modules": json_safe(
                    item.get("overlapping_modules") or []
                ),
                "different_modules": json_safe(
                    item.get("different_modules") or []
                ),
            }
            for item in records
        ],
        "population": json_safe(population_summary),
        "requirements": [
            "Use this feedback for unified agentic variation.",
            "Do not repeat a measured failed AlgorithmSpec unchanged.",
            "Record the source generation and applied directives in generation_metadata.",
        ],
    }


def _reflection_directives(reflection: dict[str, Any]) -> list[str]:
    directives: list[str] = []
    for key in ("next_directive", "recommended_actions", "recommendations", "next_actions", "directives"):
        value = reflection.get(key)
        values = value if isinstance(value, list) else [value]
        for item in values:
            if isinstance(item, dict):
                text_value = json.dumps(json_safe(item), ensure_ascii=False, sort_keys=True)
            elif item is None:
                continue
            else:
                text_value = str(item).strip()
            if text_value and text_value not in directives:
                directives.append(text_value)
    return directives[:8]


def _candidate_feedback_view(record: dict[str, Any]) -> dict[str, Any]:
    spec = record.get("search_algorithm_spec") or record.get("normalized_algorithm_spec") or {}
    return {
        "spec_id": record.get("spec_id"),
        "status": record.get("status"),
        "proposal_strategy": record.get("proposal_strategy"),
        "metrics": json_safe(record.get("metrics") or {}),
        "failure_reason": json_safe(record.get("failure_reason")),
        "adaptive_control_summary": json_safe(
            record.get("adaptive_control_summary")
            or (record.get("training_report") or {}).get("adaptive_control_summary")
            or {}
        ),
        "runtime_scaled_policy_summary": json_safe(
            record.get("runtime_scaled_policy_summary") or {}
        ),
        "adaptive_compute_budget_summary": json_safe(
            record.get("adaptive_compute_budget_summary") or {}
        ),
        "adaptive_policy_warnings": json_safe(
            list(record.get("adaptive_policy_warnings") or [])[:8]
        ),
        "algorithm_spec": json_safe(
            {
                key: spec.get(key)
                for key in (
                    "network",
                    "sampling",
                    "constraint_enforcement",
                    "loss",
                    "optimization",
                    "training",
                    "generation_metadata",
                )
                if key in spec
            }
        ),
    }


def _formal_parent_candidate_view(
    record: dict[str, Any],
    objective: str,
    *,
    execution_feedback: bool = True,
) -> dict[str, Any]:
    if not execution_feedback:
        spec = deepcopy(
            record.get("search_algorithm_spec")
            or record.get("normalized_algorithm_spec")
            or record.get("algorithm_spec")
            or {}
        )
        metadata = dict(spec.get("generation_metadata") or {})
        for key in (
            "feedback_source_generation",
            "reflection_directives_applied",
            "posterior_evidence_applied",
            "generation_policy_audit",
            "inherited_strengths",
            "expected_improvements",
        ):
            metadata.pop(key, None)
        if metadata:
            spec["generation_metadata"] = metadata
        else:
            spec.pop("generation_metadata", None)
        return {
            "candidate_id": record.get("spec_id") or record.get("candidate_id"),
            "generation": record.get("generation_index", record.get("generation")),
            "algorithm_spec": json_safe(spec),
        }
    training_report = record.get("training_report") or {}
    history = training_report.get("history") or []
    losses = [
        float(item["total_loss"])
        for item in history
        if isinstance(item, dict) and isinstance(item.get("total_loss"), (int, float))
    ]
    late_stage_plateau = False
    if len(losses) >= 10:
        window = max(2, len(losses) // 10)
        earlier = sum(losses[-2 * window : -window]) / window
        latest = sum(losses[-window:]) / window
        late_stage_plateau = abs(earlier - latest) / max(abs(earlier), 1.0e-12) < 1.0e-3
    failure_text = str(record.get("failure_reason") or "").lower()
    formal_metrics = record.get("generation_metrics") or record.get("metrics") or {}
    loss_dynamics = {
        key: _numeric_history_summary(history, key)
        for key in (
            "total_loss",
            "pde_loss",
            "boundary_loss",
            "initial_loss",
            "reference_mse",
        )
    }
    loss_dynamics = {
        key: value for key, value in loss_dynamics.items() if value["num_points"] > 0
    }
    gradient_dynamics = _numeric_history_summary(history, "gradient_norm")
    optimization_feedback = _build_optimization_feedback(
        history=history,
        training_report=training_report,
        loss_dynamics=loss_dynamics,
        late_stage_plateau=late_stage_plateau,
    )
    success_factors: list[str] = []
    failure_factors: list[str] = []
    if record.get("training_success"):
        success_factors.append("Training and formal evaluation completed successfully.")
    if losses and losses[-1] < losses[0]:
        success_factors.append("Recorded total loss decreased over training.")
    if math.isfinite(_metric_value(record, objective)):
        success_factors.append(
            f"Finite formal {objective}={_metric_value(record, objective):.8g}."
        )
    if late_stage_plateau:
        failure_factors.append("Late-stage loss history indicates a plateau.")
    if bool(training_report.get("nan_detected")):
        failure_factors.append("Training reported NaN or numerical instability.")
    if record.get("failure_reason"):
        failure_factors.append(str(record["failure_reason"]))
    reflection = record.get("reflection") or {}
    for item in reflection.get("success_factors") or []:
        if str(item) not in success_factors:
            success_factors.append(str(item))
    for item in reflection.get("failure_factors") or []:
        if str(item) not in failure_factors:
            failure_factors.append(str(item))
    return {
        "candidate_id": record.get("spec_id"),
        "generation": record.get("generation_index", record.get("generation")),
        "rank": record.get("low_fidelity_rank") or record.get("generation_rank"),
        "algorithm_spec": json_safe(
            record.get("search_algorithm_spec")
            or record.get("normalized_algorithm_spec")
            or {}
        ),
        "fitness": _metric_value(record, objective),
        "metrics": json_safe(formal_metrics),
        "formal_metrics": json_safe(formal_metrics),
        "training_summary": {
            "converged": bool(record.get("training_success")) and not bool(training_report.get("nan_detected")),
            "late_stage_plateau": late_stage_plateau,
            "gradient_instability": bool(training_report.get("nan_detected")) or "gradient" in failure_text,
            "main_strength": success_factors[0] if success_factors else "No confirmed strength recorded.",
            "main_weakness": failure_factors[0] if failure_factors else "No confirmed failure recorded.",
            "loss_history_summary": {
                "num_points": len(losses),
                "first_loss": losses[0] if losses else None,
                "last_loss": losses[-1] if losses else None,
                "minimum_loss": min(losses) if losses else None,
            },
            "loss_dynamics": loss_dynamics,
            "gradient_dynamics": gradient_dynamics,
            "optimizer_audit": json_safe(training_report.get("optimizer_audit") or {}),
            "actual_iterations": training_report.get("actual_iterations"),
            "stopped_early": bool(training_report.get("stopped_early")),
        },
        "physical_supervision_summary": {
            "pde_residual_error": formal_metrics.get("pde_residual_error"),
            "residual_distribution": json_safe(
                formal_metrics.get("residual_distribution")
                or formal_metrics.get("governing_residual_distribution")
                or {}
            ),
            "residual_spatial_summary": json_safe(
                formal_metrics.get("residual_spatial_summary")
                or {
                    "available": False,
                    "reason": "legacy_candidate_record",
                    "equations": {},
                }
            ),
            "boundary_error": formal_metrics.get("boundary_error"),
            "initial_error": formal_metrics.get("initial_error"),
            "constraint_violation": formal_metrics.get("constraint_violation"),
            "final_sampling_snapshot": json_safe(
                training_report.get("final_sampling_snapshot") or {}
            ),
        },
        "optimization_feedback": optimization_feedback,
        "adaptive_control_summary": json_safe(
            record.get("adaptive_control_summary")
            or training_report.get("adaptive_control_summary")
            or {}
        ),
        "runtime_scaled_policy_summary": json_safe(
            record.get("runtime_scaled_policy_summary") or {}
        ),
        "adaptive_compute_budget_summary": json_safe(
            record.get("adaptive_compute_budget_summary")
            or training_report.get("adaptive_compute_budget_summary")
            or {}
        ),
        "adaptive_policy_warnings": json_safe(
            list(record.get("adaptive_policy_warnings") or [])[:8]
        ),
        "success_factors": success_factors,
        "failure_factors": failure_factors,
        "lineage_summary": {
            "parent_ids": record.get("parent_spec_ids") or [],
            "generation_mode": record.get("generation_mode") or record.get("proposal_strategy"),
        },
    }


def _deterministically_shuffle_parent_views(
    parents: list[dict[str, Any]], *, seed: int, generation: int
) -> list[dict[str, Any]]:
    """Hide rank position while keeping runs reproducible."""

    return sorted(
        [deepcopy(item) for item in parents],
        key=lambda item: hashlib.sha256(
            f"execution-feedback-ablation:{seed}:{generation}:{item.get('candidate_id')}".encode(
                "utf-8"
            )
        ).hexdigest(),
    )


def _numeric_history_summary(
    history: list[dict[str, Any]], key: str
) -> dict[str, float | int | None]:
    values = [
        float(item[key])
        for item in history
        if isinstance(item, dict)
        and isinstance(item.get(key), (int, float))
        and not isinstance(item.get(key), bool)
        and math.isfinite(float(item[key]))
    ]
    return {
        "num_points": len(values),
        "first": values[0] if values else None,
        "last": values[-1] if values else None,
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
        "end_to_start_ratio": (
            values[-1] / values[0]
            if values and abs(values[0]) > 0.0
            else None
        ),
    }


def _build_optimization_feedback(
    *,
    history: list[dict[str, Any]],
    training_report: dict[str, Any],
    loss_dynamics: dict[str, dict[str, Any]],
    late_stage_plateau: bool,
) -> dict[str, Any]:
    loss_trajectory_summary = {
        key.removesuffix("_loss"): _trajectory_status(value)
        for key, value in loss_dynamics.items()
        if key != "reference_mse"
    }
    latest_values = {
        key.removesuffix("_loss"): float(value["last"])
        for key, value in loss_dynamics.items()
        if key != "reference_mse"
        and isinstance(value.get("last"), (int, float))
        and float(value["last"]) > 0.0
    }
    loss_value_ratio = (
        max(latest_values.values()) / min(latest_values.values())
        if len(latest_values) >= 2
        else None
    )
    audits = list(training_report.get("component_gradient_audits") or [])
    available_audits = [item for item in audits if item.get("available")]
    latest_gradient = available_audits[-1] if available_audits else {}
    conflict_counts: dict[tuple[str, str], int] = {}
    for audit in available_audits:
        pairwise = audit.get("pairwise_cosine_summary") or {}
        for pair in pairwise.get("conflicting_pairs") or []:
            names = tuple(
                sorted(
                    (
                        str(pair.get("component_a")),
                        str(pair.get("component_b")),
                    )
                )
            )
            conflict_counts[names] = conflict_counts.get(names, 0) + 1
    persistent_pairs = [
        list(pair)
        for pair, count in sorted(conflict_counts.items())
        if count >= 2
    ]
    mse_probe_history = [
        {
            "iteration": item.get("iteration"),
            "phase": item.get("phase"),
            "reference_mse": item.get("reference_mse"),
        }
        for item in history
        if isinstance(item, dict)
        and isinstance(item.get("reference_mse"), (int, float))
        and math.isfinite(float(item["reference_mse"]))
    ]
    lbfgs_values = [
        float(item["reference_mse"])
        for item in mse_probe_history
        if "lbfgs" in str(item.get("phase") or "").casefold()
    ]
    pre_lbfgs_values = [
        float(item["reference_mse"])
        for item in mse_probe_history
        if "lbfgs" not in str(item.get("phase") or "").casefold()
    ]
    lbfgs_degradation = bool(
        lbfgs_values
        and pre_lbfgs_values
        and lbfgs_values[-1] > pre_lbfgs_values[-1]
    )
    return {
        "available": bool(history or audits or training_report.get("optimizer_audit")),
        "reason": (
            None
            if history or audits or training_report.get("optimizer_audit")
            else "legacy_candidate_record"
        ),
        "loss_trajectory_summary": loss_trajectory_summary,
        "loss_value_imbalance": {
            "available": loss_value_ratio is not None,
            "max_to_min_final_loss_ratio": loss_value_ratio,
            "largest_final_component": (
                max(latest_values, key=latest_values.get)
                if latest_values
                else None
            ),
            "smallest_nonzero_final_component": (
                min(latest_values, key=latest_values.get)
                if latest_values
                else None
            ),
        },
        "gradient_summary": {
            "available": bool(available_audits),
            "reason": (
                None if available_audits else "component_gradient_diagnostics_unavailable"
            ),
            "total_grad_norm": latest_gradient.get("total_grad_norm"),
            "component_grad_norms": json_safe(
                latest_gradient.get("component_grad_norms") or {}
            ),
            "component_to_total_ratios": json_safe(
                latest_gradient.get("component_to_total_ratios") or {}
            ),
            "dominant_component": latest_gradient.get("dominant_component"),
            "weakest_nonzero_component": latest_gradient.get(
                "weakest_nonzero_component"
            ),
            "gradient_imbalance_detected": (
                any(
                    item.get("gradient_imbalance_detected") is True
                    for item in available_audits
                )
                if available_audits
                else None
            ),
            "gradient_conflict_detected": (
                any(
                    item.get("gradient_conflict_detected") is True
                    for item in available_audits
                )
                if available_audits
                else None
            ),
            "pairwise_cosine_summary": json_safe(
                latest_gradient.get("pairwise_cosine_summary")
                or {
                    "available": False,
                    "reason": "component_gradient_diagnostics_unavailable",
                }
            ),
            "persistent_conflict_pairs": persistent_pairs,
            "audit_count": len(audits),
        },
        "optimizer_audit": json_safe(training_report.get("optimizer_audit") or {}),
        "learning_rate_and_scheduler_history": [
            {
                "iteration": item.get("iteration"),
                "phase": item.get("phase"),
                "learning_rate": item.get("learning_rate"),
            }
            for item in history
            if isinstance(item, dict) and item.get("learning_rate") is not None
        ],
        "plateau_detected": bool(late_stage_plateau),
        "mse_probe_history": mse_probe_history,
        "lbfgs_transition_degradation": lbfgs_degradation,
        "diagnostic_distinctions": {
            "loss_value_imbalance": "compares observed loss magnitudes",
            "gradient_norm_imbalance": "compares component gradient norms",
            "gradient_direction_conflict": "uses pairwise gradient cosine similarity",
        },
    }


def _trajectory_status(summary: dict[str, Any]) -> str:
    first = summary.get("first")
    last = summary.get("last")
    if not isinstance(first, (int, float)) or not isinstance(last, (int, float)):
        return "unavailable"
    scale = max(abs(float(first)), 1.0e-12)
    relative_change = (float(last) - float(first)) / scale
    if abs(relative_change) <= 0.05:
        return "plateaued"
    if relative_change <= -0.20:
        return "decreasing"
    if relative_change >= 0.20:
        return "increasing"
    return "mixed_or_slow_change"


def _ensure_experiment_contract(
    output_dir: Path,
    *,
    resume: bool = False,
    experiment_contract: dict[str, Any] | None = None,
    ablation_settings: dict[str, Any] | None = None,
) -> None:
    if not output_dir.exists():
        if resume:
            raise RuntimeError(
                "Cannot resume: the experiment output directory does not exist"
            )
        return
    if resume:
        state_path = output_dir / "search_state.json"
        contract_path = output_dir / "experiment_contract.json"
        if not state_path.exists() or not contract_path.exists():
            raise RuntimeError(
                "Cannot resume without search_state.json and experiment_contract.json"
            )
        existing_contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if (
            existing_contract.get("experiment_contract_version")
            != EXPERIMENT_CONTRACT_VERSION
        ):
            raise RuntimeError(
                "Cannot resume an output directory created by a different experiment contract"
            )
        if experiment_contract is not None:
            for section, expected in experiment_contract.items():
                if existing_contract.get(section) != expected:
                    raise RuntimeError(
                        f"Cannot resume because experiment contract section {section} changed"
                    )
        if (
            ablation_settings is not None
            and existing_contract.get("ablation") != ablation_settings
        ):
            raise RuntimeError(
                "Cannot resume because the ablation contract changed"
            )
        return
    existing_names = {item.name for item in output_dir.iterdir()}
    permitted_dry_run_files = {"contract_dry_run.json"}
    if existing_names - permitted_dry_run_files:
        raise RuntimeError(
            "Refusing to continue an existing experiment output directory. "
            f"Use a new {EXPERIMENT_CONTRACT_VERSION} output path."
        )


def _persist_search_resume_state(
    *,
    state_path: Path,
    search_state: dict[str, Any],
    generation_results: list[dict[str, Any]],
    all_records: list[dict[str, Any]],
    reflections: list[dict[str, Any]],
    generation_feedback: dict[str, Any],
    global_top3: list[dict[str, Any]],
) -> None:
    validate_search_state(search_state)
    candidate_ids = [
        str(item.get("spec_id"))
        for item in all_records
        if item.get("spec_id") is not None
    ]
    previous_top3_references = [
        {
            "spec_id": item.get("spec_id"),
            "generation_rank": item.get("generation_rank"),
        }
        for item in global_top3
    ]
    _write_json(
        state_path,
        {
            "experiment_contract_version": EXPERIMENT_CONTRACT_VERSION,
            "search_state": search_state,
            "candidate_ids": candidate_ids,
            "generation_results_file": "generation_results.json",
            "generation_feedback": generation_feedback,
            "global_top3": previous_top3_references,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        },
    )


def _load_search_resume_state(output_dir: Path) -> dict[str, Any]:
    state_path = output_dir / "search_state.json"
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    if payload.get("experiment_contract_version") != EXPERIMENT_CONTRACT_VERSION:
        raise RuntimeError("Resume-state experiment contract version mismatch")
    state = payload.get("search_state")
    if not isinstance(state, dict):
        raise RuntimeError("Resume state is missing search_state")
    required_fields = set(initial_search_state())
    missing = sorted(required_fields - set(state))
    if missing:
        raise RuntimeError(
            "Resume state is missing required fields: " + ", ".join(missing)
        )
    validate_search_state(state)
    # Schema-v16 and older embedded every full candidate several times.  New
    # state files contain only IDs and hydrate the unique runtime artifacts.
    embedded_records = payload.get("all_records")
    if isinstance(embedded_records, list):
        all_records = list(embedded_records)
    else:
        requested_ids = {
            str(item) for item in payload.get("candidate_ids") or []
        }
        all_records = [
            item
            for item in load_runtime_candidate_records(output_dir)
            if not requested_ids or str(item.get("spec_id")) in requested_ids
        ]
        all_records = [
            item
            for item in all_records
            if str(item.get("fidelity") or "low").casefold() != "high"
        ]

    embedded_generations = payload.get("generation_results")
    if isinstance(embedded_generations, list):
        generation_results = list(embedded_generations)
    else:
        generation_path = output_dir / str(
            payload.get("generation_results_file") or "generation_results.json"
        )
        if generation_path.is_file():
            loaded_generations = json.loads(
                generation_path.read_text(encoding="utf-8")
            )
            generation_results = (
                list(loaded_generations)
                if isinstance(loaded_generations, list)
                else []
            )
        else:
            generation_results = []

    stored_top3 = list(payload.get("global_top3") or payload.get("previous_generation_top3") or [])
    records_by_id = {
        str(item.get("spec_id")): item for item in all_records
    }
    global_parent_top3 = [
        records_by_id.get(str(item.get("spec_id")), item)
        for item in stored_top3
        if isinstance(item, dict)
    ]
    embedded_reflections = payload.get("reflections")
    reflections = (
        list(embedded_reflections)
        if isinstance(embedded_reflections, list)
        else [
            dict(item.get("reflection") or {})
            for item in generation_results
            if isinstance(item, dict) and item.get("reflection")
        ]
    )
    return {
        "search_state": state,
        "generation_results": generation_results,
        "all_records": all_records,
        "reflections": reflections,
        "generation_feedback": dict(payload.get("generation_feedback") or {}),
        "global_top3": global_parent_top3,
    }


def _compact_generation_feedback(feedback: dict[str, Any]) -> dict[str, Any]:
    if not feedback:
        return {}
    return {
        "source_generation": feedback.get("source_generation"),
        "directives": json_safe(feedback.get("directives") or []),
        "reflection": json_safe(feedback.get("reflection") or {}),
        "stagnated": bool(feedback.get("stagnated")),
        "next_generation_mode": feedback.get("next_generation_mode"),
    }


def _generation_candidate_id(generation: int, index: int, child: dict[str, Any]) -> str:
    label = str(child.get("candidate_id") or child.get("candidate_label") or f"candidate_{index + 1:02d}")
    safe_label = "".join(character if character.isalnum() or character in {"_", "-"} else "_" for character in label)
    return f"g{generation}_{safe_label}"


def _metric_value(record: dict[str, Any], objective: str) -> float:
    if objective == "mse" and "low_fidelity_proxy_mse" in record:
        try:
            return float(record.get("low_fidelity_proxy_mse"))
        except (TypeError, ValueError):
            return float("inf")
    try:
        return float((record.get("metrics") or {}).get(objective, float("inf")))
    except (TypeError, ValueError):
        return float("inf")


def _selection_signature(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "normalized_executable_spec": executable_algorithm_spec(
                item.get("search_algorithm_spec")
                or item.get("normalized_algorithm_spec")
                or {}
            ),
            "aggregate_low_fidelity_proxy_mse": float(
                item.get("low_fidelity_aggregate_proxy_mse")
            ),
        }
        for item in records
    ]


def _hydrate_global_tracker_selection(
    selected: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Rejoin compact global-tracker entries with full runtime records."""

    records_by_id = {
        str(item.get("spec_id")): item
        for item in records
        if item.get("spec_id") is not None
    }
    hydrated: list[dict[str, Any]] = []
    for item in selected:
        source = records_by_id.get(str(item.get("spec_id")))
        if source is None:
            hydrated.append(item)
            continue
        merged = deepcopy(source)
        for key in (
            "low_fidelity_rank",
            "low_fidelity_aggregate_proxy_mse",
            "low_fidelity_duplicate_count",
            "low_fidelity_duplicate_member_ids",
        ):
            if key in item:
                merged[key] = deepcopy(item[key])
        hydrated.append(merged)
    return hydrated


def _apply_low_fidelity_proxy(
    record: dict[str, Any], *, checkpoint_iterations: list[int]
) -> dict[str, Any]:
    """Resolve a robust checkpoint median without changing evaluated metrics."""

    updated = dict(record)
    report = dict(updated.get("training_report") or {})
    metrics = dict(updated.get("metrics") or {})
    selected_model_mse = _finite_metric(metrics.get("mse"))
    observations = _reference_mse_observations(report)
    targets = sorted({int(item) for item in checkpoint_iterations})
    selected = _closest_reference_mse_observations(observations, targets)
    values = [value for _, value in selected]
    proxy = float(median(values)) if values else None
    checkpoint_mse: dict[str, float | None] = {}
    checkpoint_sources: dict[str, int | None] = {}
    proxy_observations: list[dict[str, Any]] = []
    for index, target in enumerate(targets):
        source_iteration, value = selected[index] if index < len(selected) else (None, None)
        checkpoint_mse[str(target)] = value
        checkpoint_sources[str(target)] = source_iteration
        if source_iteration is not None and value is not None:
            proxy_observations.append(
                {
                    "target_iteration": target,
                    "source_iteration": source_iteration,
                    "mse": value,
                    "exact": source_iteration == target,
                }
            )
    complete = len(selected) == len(targets) and bool(targets)
    exact = complete and all(
        item["exact"] for item in proxy_observations
    )
    last_checkpoint_mse = _finite_metric(report.get("last_checkpoint_mse"))
    last_checkpoint_iteration = report.get("last_checkpoint_iteration")
    if last_checkpoint_mse is None and observations:
        last_checkpoint_iteration, last_checkpoint_mse = max(
            observations.items(), key=lambda item: item[0]
        )
    updated["metrics"] = metrics
    # Explicit names prevent the rank proxy, selected artifact, and last
    # optimizer state from being interpreted as the same model measurement.
    updated["proxy_mse"] = proxy
    updated["selected_model_mse"] = selected_model_mse
    updated["last_checkpoint_mse"] = last_checkpoint_mse
    updated["last_checkpoint_iteration"] = last_checkpoint_iteration
    updated["low_fidelity_checkpoint_mse"] = checkpoint_mse
    updated["low_fidelity_checkpoint_sources"] = checkpoint_sources
    updated["low_fidelity_proxy_observations"] = proxy_observations
    updated["low_fidelity_proxy_mse"] = proxy
    updated["low_fidelity_proxy_complete"] = complete
    updated["low_fidelity_proxy_exact"] = exact
    updated["low_fidelity_proxy_status"] = (
        "exact"
        if exact
        else "nearest"
        if complete
        else "degraded"
        if proxy is not None
        else "unavailable"
    )
    return updated


def _finite_metric(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _reference_mse_observations(report: dict[str, Any]) -> dict[int, float]:
    """Collect every finite fixed-grid MSE observation keyed by actual step."""

    observations: dict[int, float] = {}
    for raw_iteration, raw_value in dict(
        report.get("reference_mse_checkpoints") or {}
    ).items():
        try:
            iteration = int(raw_iteration)
        except (TypeError, ValueError):
            continue
        value = _finite_metric(raw_value)
        if iteration >= 0 and value is not None:
            observations[iteration] = value
    for entry in report.get("history") or []:
        if not isinstance(entry, dict):
            continue
        try:
            iteration = int(entry.get("iteration"))
        except (TypeError, ValueError):
            continue
        value = _finite_metric(entry.get("reference_mse"))
        if iteration >= 0 and value is not None:
            observations.setdefault(iteration, value)
    try:
        last_iteration = int(report.get("last_checkpoint_iteration"))
    except (TypeError, ValueError):
        last_iteration = -1
    last_value = _finite_metric(report.get("last_checkpoint_mse"))
    if last_iteration >= 0 and last_value is not None:
        observations[last_iteration] = last_value
    return observations


def _closest_reference_mse_observations(
    observations: dict[int, float], targets: list[int]
) -> list[tuple[int, float]]:
    """Choose an ordered, unique set minimizing distance to requested steps."""

    available = sorted(observations.items())
    count = min(len(available), len(targets))
    if count == 0:
        return []
    if count < len(targets):
        # With fewer than the requested number, favor late observations because
        # they best represent the completed low-fidelity budget.
        return available[-count:]
    # Dynamic programming avoids enumerating O(n choose k) history subsets in
    # diagnostic mode, where every optimizer step may have an observation.
    # Each state is (total_distance, negative_iteration_sum, selected_indices).
    states: list[tuple[int, int, tuple[int, ...]] | None] = [
        (0, 0, ()),
        *([None] * len(targets)),
    ]
    for index, (iteration, _) in enumerate(available):
        next_states = list(states)
        upper = min(len(targets) - 1, index)
        for matched in range(upper, -1, -1):
            state = states[matched]
            if state is None:
                continue
            candidate = (
                state[0] + abs(int(iteration) - int(targets[matched])),
                state[1] - int(iteration),
                (*state[2], index),
            )
            current = next_states[matched + 1]
            if current is None or candidate[:2] < current[:2]:
                next_states[matched + 1] = candidate
        states = next_states
    selected_state = states[len(targets)]
    if selected_state is None:
        return []
    return [available[index] for index in selected_state[2]]


def _refresh_posterior_context(
    posterior: list[dict[str, Any]], generation_feedback: dict[str, Any], max_items: int
) -> list[dict[str, Any]]:
    if max_items <= 0:
        return []
    generation_entry = {
        "source": "current_run_generation_reflection",
        "binding": False,
        "source_generation": generation_feedback.get("source_generation"),
        "reflection": generation_feedback.get("reflection") or {},
        "directives": generation_feedback.get("directives") or [],
        "successful_candidates": generation_feedback.get("successful_candidates") or [],
        "failed_candidates": generation_feedback.get("failed_candidates") or [],
    }
    older = [
        item
        for item in posterior
        if not (
            item.get("source") == "current_run_generation_reflection"
            and item.get("source_generation") == generation_entry["source_generation"]
        )
    ]
    return [generation_entry, *older][:max_items]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, ensure_ascii=False, allow_nan=False, default=str), encoding="utf-8")


def _generation_best_metrics_payload(result: dict[str, Any]) -> dict[str, Any]:
    """Return only Rank-1 solution metrics for user-facing completed output."""

    def metrics_from(record: Any) -> dict[str, Any]:
        if not isinstance(record, dict):
            return {}
        return dict(
            record.get("generation_metrics")
            or record.get("metrics")
            or record.get("ranking_metrics")
            or {}
        )

    def metric_pair(record: Any) -> dict[str, float | None]:
        metrics = metrics_from(record)
        l2re = metrics.get("l2re")
        if l2re is None:
            l2re = metrics.get("relative_l2_error")
        return {
            "mse": _finite_metric(metrics.get("mse")),
            "relative_l2_error": _finite_metric(l2re),
        }

    generations = []
    for generation in result.get("generation_results") or []:
        if not isinstance(generation, dict):
            continue
        rank1 = generation.get("final_rank_1") or generation.get("proxy_rank_1")
        if not isinstance(rank1, dict):
            ranked = generation.get("ranked_top3") or []
            rank1 = ranked[0] if ranked and isinstance(ranked[0], dict) else {}
        generation_number = generation.get("generation")
        if generation_number is None:
            generation_number = generation.get("generation_index")
        generations.append(
            {
                "generation": int(generation_number or 0),
                **metric_pair(rank1),
            }
        )
    return {
        "problem_id": result.get("problem_id"),
        "generations": generations,
        "final_high_fidelity": metric_pair(result.get("final_rank_1")),
    }


def _prune_completed_metrics_output(output_dir: Path, *, keep: Path) -> None:
    """Remove known search artifacts after a successful metrics-only run."""

    target = output_dir.resolve()
    retained = keep.resolve()
    if target == Path(target.anchor) or retained.parent != target or not retained.is_file():
        raise RuntimeError("Refusing to prune an unsafe metrics output directory")
    generated_names = {
        "candidate_records.jsonl",
        "checkpoints",
        "experiment_contract.json",
        "final_high_fidelity_evaluation.json",
        "final_high_fidelity_selection.json",
        "generation_0",
        "generation_results.json",
        "global_best_tracker.json",
        "llm_trace.json",
        "open_search_summary.json",
        "population.json",
        "posterior_snapshot.json",
        "runtime_candidates",
        "search_state.json",
    }
    for child in target.iterdir():
        resolved = child.resolve()
        if resolved == retained or child.name not in generated_names:
            continue
        if child.is_symlink() or child.is_file():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)


def _compact_llm_trace(
    traces: list[dict[str, Any]], *, output_detail: str
) -> list[dict[str, Any]]:
    """Drop duplicated LLM response bodies from normal experiment output."""

    if normalize_output_detail(output_detail) == "diagnostic":
        return json_safe(traces)
    omitted = {
        "raw_response",
        "proposal_metadata",
        "generation_policy_audit",
        "validation_errors",
    }
    return [
        {
            key: json_safe(value)
            for key, value in item.items()
            if key not in omitted
        }
        for item in traces
        if isinstance(item, dict)
    ]


def _compact_generation_result(result: dict[str, Any]) -> dict[str, Any]:
    """Serialize one generation without embedding candidate payloads again."""

    compact = deepcopy(result)
    for key in (
        "records",
        "population",
        "input_feedback",
        "proxy_top_3",
        "final_top_3",
        "proxy_rank_1",
        "final_rank_1",
        "ranked_top3",
        "generation_ranked_top3",
        "global_top3",
        "generation_posterior",
    ):
        compact.pop(key, None)
    variation = dict(compact.get("variation") or {})
    variation.pop("resolved_design_context", None)
    variation.pop("candidate_audit", None)
    variation.pop("rejected_children", None)
    variation.pop("diversity_retries", None)
    compact["variation"] = variation
    audit = dict(compact.get("generation_audit") or {})
    for key in (
        "generation_summary",
        "search_state",
        "generated_candidate_metadata",
        "global_best_tracker_before",
        "global_best_tracker_after",
        "global_best_tracker_update",
    ):
        audit.pop(key, None)
    compact["generation_audit"] = audit
    compact["candidate_records"] = [
        {
            "spec_id": item.get("spec_id"),
            "generation_rank": item.get("generation_rank"),
            "training_success": bool(item.get("training_success")),
            "status": item.get("status"),
            "proxy_mse": item.get("proxy_mse"),
            "selected_model_mse": item.get("selected_model_mse"),
            "candidate_record": (
                f"runtime_candidates/{item.get('spec_id')}/candidate_record.json"
            ),
        }
        for item in result.get("records") or []
        if isinstance(item, dict)
    ]
    return json_safe(compact)


def _compact_high_fidelity_record(record: dict[str, Any]) -> dict[str, Any]:
    """Keep final ranking evidence and point to the unique candidate record."""

    candidate_id = str(record.get("spec_id") or "unknown_candidate")
    training_report = dict(record.get("training_report") or {})
    metrics = compact_final_metrics(record.get("metrics") or {}, training_report)
    ranking_metrics = {
        key: metrics.get(key)
        for key in (
            "mse",
            "proxy_mse",
            "selected_model_mse",
            "last_checkpoint_mse",
            "l2re",
        )
        if metrics.get(key) is not None
    }
    return json_safe(
        {
            "spec_id": candidate_id,
            "source_low_fidelity_candidate_id": record.get(
                "source_low_fidelity_candidate_id"
            ),
            "source_generation_id": record.get("source_generation_id"),
            "status": record.get("status"),
            "training_success": bool(record.get("training_success")),
            "failure_reason": record.get("failure_reason"),
            "low_fidelity_rank": record.get("low_fidelity_rank"),
            "high_fidelity_rank": record.get("high_fidelity_rank"),
            "rank_change": record.get("rank_change"),
            "source_low_fidelity_proxy_mse": record.get(
                "source_low_fidelity_proxy_mse"
            ),
            "high_fidelity_model_selection_audit": record.get(
                "high_fidelity_model_selection_audit"
            ),
            "cross_fidelity_audit": record.get(
                "cross_fidelity_audit"
            ),
            "ranking_metrics": ranking_metrics,
            "training_summary": {
                key: json_safe(training_report.get(key))
                for key in (
                    "success",
                    "actual_iterations",
                    "phase_iterations",
                    "training_time",
                    "final_loss",
                    "nan_detected",
                    "failure_reason",
                    "last_checkpoint_mse",
                    "best_reference_mse_checkpoint",
                    "reference_mse_model_selection_enabled",
                    "best_training_loss_checkpoint",
                    "final_model_policy",
                    "selected_checkpoint_source",
                    "selected_checkpoint_step",
                    "selected_checkpoint_metric",
                    "pre_lbfgs_physics_checkpoint",
                    "best_lbfgs_physics_checkpoint",
                )
                if key in training_report
            },
            "candidate_record": (
                f"runtime_candidates/{candidate_id}/candidate_record.json"
            ),
        }
    )


def _compact_low_fidelity_selection_record(
    record: dict[str, Any],
) -> dict[str, Any]:
    """Persist final-selection evidence without another AlgorithmSpec copy."""

    candidate_id = str(record.get("spec_id") or "unknown_candidate")
    return json_safe(
        {
            "spec_id": candidate_id,
            "low_fidelity_rank": record.get("low_fidelity_rank"),
            "aggregate_proxy_mse": record.get(
                "low_fidelity_aggregate_proxy_mse"
            ),
            "duplicate_count": record.get("low_fidelity_duplicate_count"),
            "duplicate_member_ids": record.get(
                "low_fidelity_duplicate_member_ids"
            ) or [],
            "candidate_record": (
                f"runtime_candidates/{candidate_id}/candidate_record.json"
            ),
        }
    )


def _hydrate_persisted_final_selection(
    payload: dict[str, Any],
    all_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Rejoin compact persisted ranks with canonical runtime candidates."""

    stored = [
        dict(item)
        for item in payload.get("selected_candidates") or []
        if isinstance(item, dict)
    ]
    selected_ids = list(payload.get("selected_candidate_ids") or [])
    if not selected_ids:
        selected_ids = [item.get("spec_id") for item in stored]
    stored_by_id = {
        str(item.get("spec_id")): item
        for item in stored
        if item.get("spec_id") is not None
    }
    records_by_id = {
        str(item.get("spec_id")): item
        for item in all_records
        if item.get("spec_id") is not None
    }
    hydrated: list[dict[str, Any]] = []
    for rank, candidate_id in enumerate(selected_ids, start=1):
        key = str(candidate_id)
        persisted = stored_by_id.get(key, {})
        source = records_by_id.get(key)
        # Legacy selection files embedded the complete record and remain
        # readable even when their candidate runtime directory is absent.
        if source is None and (
            persisted.get("search_algorithm_spec")
            or persisted.get("normalized_algorithm_spec")
        ):
            source = persisted
        if source is None:
            raise RuntimeError(
                f"Persisted final-selection candidate {key!r} is unavailable"
            )
        item = deepcopy(source)
        item["low_fidelity_rank"] = int(
            persisted.get("low_fidelity_rank") or rank
        )
        aggregate = persisted.get("aggregate_proxy_mse")
        if aggregate is None:
            aggregate = persisted.get("low_fidelity_aggregate_proxy_mse")
        if aggregate is None:
            aggregate = item.get("low_fidelity_proxy_mse")
        item["low_fidelity_aggregate_proxy_mse"] = aggregate
        item["low_fidelity_duplicate_count"] = int(
            persisted.get("duplicate_count")
            or persisted.get("low_fidelity_duplicate_count")
            or 1
        )
        item["low_fidelity_duplicate_member_ids"] = list(
            persisted.get("duplicate_member_ids")
            or persisted.get("low_fidelity_duplicate_member_ids")
            or [key]
        )
        hydrated.append(item)
    return hydrated


def _hydrate_persisted_high_fidelity_records(
    payload: dict[str, Any],
    runtime_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Restore resumable high-fidelity records from compact ranking entries."""

    stored = [
        dict(item)
        for item in payload.get("records") or []
        if isinstance(item, dict)
    ]
    runtime_by_id = {
        str(item.get("spec_id")): item
        for item in runtime_records
        if item.get("spec_id") is not None
    }
    hydrated: list[dict[str, Any]] = []
    for persisted in stored:
        candidate_id = str(persisted.get("spec_id") or "")
        source = runtime_by_id.get(candidate_id)
        # Schema-v16 files embedded enough metrics to remain self-contained.
        if source is None and isinstance(persisted.get("metrics"), dict):
            source = persisted
        if source is None:
            raise RuntimeError(
                f"Persisted high-fidelity candidate {candidate_id!r} is unavailable"
            )
        item = deepcopy(source)
        for field in (
            "source_low_fidelity_candidate_id",
            "source_generation_id",
            "status",
            "training_success",
            "failure_reason",
            "low_fidelity_rank",
            "high_fidelity_rank",
            "rank_change",
            "source_low_fidelity_proxy_mse",
        ):
            if field in persisted:
                item[field] = persisted[field]
        hydrated.append(item)
    return hydrated


def _compact_selection_audit(audit: dict[str, Any]) -> dict[str, Any]:
    """Remove normalized-spec copies while preserving selection correctness."""

    source = dict(audit or {})
    consistency = dict(source.get("consistency_audit") or {})
    compact_consistency = {
        "passed": consistency.get("passed"),
        "purpose": consistency.get("purpose"),
    }
    selected = source.get("selected_candidate_ids")
    if selected is None:
        selected = source.get("selected_spec_ids")
    return json_safe(
        {
            "selection_source": source.get("selection_source"),
            "candidate_count": source.get("candidate_count"),
            "selected_candidate_ids": selected,
            "deduplicate_normalized_specs": source.get(
                "deduplicate_normalized_specs"
            ),
            "duplicate_aggregation": source.get("duplicate_aggregation"),
            "ranking_metric": source.get("ranking_metric"),
            "consistency_audit": compact_consistency,
        }
    )


def _compact_search_result(result: dict[str, Any]) -> dict[str, Any]:
    """Return the root run summary without copying detailed artifact payloads."""

    ranked = list(result.get("final_top_3") or [])
    compact_ranked = [
        _compact_high_fidelity_record(item)
        for item in ranked
        if isinstance(item, dict)
    ]
    summary_keys = (
        "status",
        "search_status",
        "pipeline",
        "problem_id",
        "ablation",
        "experiment_contract_version",
        "generation_iteration_recommendations",
        "regular_generation_recommended_iterations",
        "high_fidelity_iterations",
        "iteration_recommendations_are_advisory",
        "maximum_generations",
        "actual_generations_completed",
        "low_fidelity_generation_count",
        "search_termination_reason",
        "termination_reason",
        "evaluation_policy",
        "search_audit",
        "experiment_archive",
    )
    compact = {
        key: json_safe(result.get(key))
        for key in summary_keys
        if key in result
    }
    config = dict(result.get("config") or {})
    compact["config"] = {
        key: json_safe(config.get(key))
        for key in (
            "benchmark",
            "seed",
            "independent_run_id",
            "device",
            "real_llm_provider",
            "real_llm_model",
            "llm_api_base",
            "llm_max_tokens",
            "llm_max_concurrency",
            "llm_timeout_seconds",
            "output_detail",
            "maximum_sampling_points",
            "maximum_model_parameters",
            "evaluation_size",
            "residual_evaluation_size",
        )
        if key in config
    }
    compact.update(
        {
            "generation_results": "generation_results.json",
            "candidate_index": "candidate_records.jsonl",
            "population": "population.json",
            "global_best_tracker": "global_best_tracker.json",
            "posterior_snapshot": "posterior_snapshot.json",
            "final_high_fidelity_selection": (
                "final_high_fidelity_selection.json"
            ),
            "final_high_fidelity_evaluation": (
                "final_high_fidelity_evaluation.json"
            ),
            "final_rank_1": compact_ranked[0] if compact_ranked else None,
            "final_top_3": compact_ranked,
            "best_candidate_record": (
                compact_ranked[0].get("candidate_record")
                if compact_ranked
                else None
            ),
        }
    )
    return json_safe(compact)


def _write_candidate_runtime_snapshot(
    output_dir: Path,
    record: dict[str, Any],
    *,
    output_detail: str = "compact",
) -> None:
    """Persist one candidate without duplicating its large in-memory objects."""

    detail = normalize_output_detail(output_detail)
    candidate_id = str(record.get("spec_id") or "unknown_candidate")
    target = output_dir / "runtime_candidates" / candidate_id
    raw_spec_path = target / "algorithm_spec_raw.json"
    search_spec_path = target / "algorithm_spec_search.json"
    normalized_spec_path = target / "algorithm_spec_normalized.json"
    validation_path = target / "validation_report.json"
    metrics_path = target / "final_metrics.json"
    history_path = target / "training_history.jsonl"

    if detail == "diagnostic":
        write_candidate_json(raw_spec_path, record.get("algorithm_spec_raw") or {})
    write_candidate_json(
        search_spec_path,
        record.get("search_algorithm_spec")
        or record.get("normalized_algorithm_spec")
        or {},
    )
    write_candidate_json(
        normalized_spec_path,
        record.get("normalized_algorithm_spec") or {},
    )
    search_reference = artifact_reference(search_spec_path)
    normalized_reference = artifact_reference(normalized_spec_path)

    validation = dict(record.get("validation_report") or {})
    compact_validation = {
        "valid": bool(validation.get("valid")),
        "errors": json_safe(validation.get("errors") or []),
        "warnings": json_safe(validation.get("warnings") or []),
        "budget_usage": json_safe(validation.get("budget_usage") or {}),
        "normalized_spec": normalized_reference,
    }
    if detail == "diagnostic":
        compact_validation["raw_spec"] = artifact_reference(raw_spec_path)
    write_candidate_json(validation_path, compact_validation)

    full_training_report = dict(record.get("training_report") or {})
    full_history = history_with_required_events(full_training_report)
    artifact_history = (
        full_history
        if detail == "diagnostic"
        else compact_training_history(full_history)
    )
    write_candidate_jsonl(history_path, artifact_history)

    full_metrics = dict(record.get("metrics") or {})
    for field in (
        "proxy_mse",
        "selected_model_mse",
        "last_checkpoint_mse",
    ):
        if record.get(field) is not None:
            full_metrics.setdefault(field, record.get(field))
    full_metrics.setdefault(
        "nan_detected",
        bool(full_training_report.get("nan_detected")),
    )
    full_metrics.setdefault("solution_collapse_detected", False)
    full_metrics.setdefault(
        "solution_valid",
        bool(record.get("training_success"))
        and not bool(full_metrics.get("nan_detected"))
        and not bool(full_metrics.get("solution_collapse_detected")),
    )
    artifact_metrics = compact_final_metrics(full_metrics, full_training_report)
    write_candidate_json(metrics_path, artifact_metrics)

    artifacts = {
        "search_spec": search_reference,
        "runtime_spec": normalized_reference,
        "normalized_spec": normalized_reference,
        "validation_report": artifact_reference(validation_path),
        "training_history": artifact_reference(history_path),
        "final_metrics": artifact_reference(metrics_path),
    }
    if detail == "diagnostic":
        artifacts["raw_spec"] = artifact_reference(raw_spec_path)
    if detail == "diagnostic":
        temporal_payload = temporal_diagnostics(full_metrics)
        if temporal_payload:
            temporal_path = target / "temporal_diagnostics.json"
            write_candidate_json(temporal_path, temporal_payload)
            artifacts["temporal_diagnostics"] = artifact_reference(temporal_path)
        residual_payload = residual_diagnostics(full_metrics)
        if residual_payload:
            residual_path = target / "residual_diagnostics.json"
            write_candidate_json(residual_path, residual_payload)
            artifacts["residual_diagnostics"] = artifact_reference(residual_path)

    training_summary = compact_training_report(
        full_training_report,
        artifact_history,
    )
    public_record = dict(record)
    public_record.setdefault(
        "solution_valid",
        full_metrics.get("solution_valid"),
    )
    public_record.setdefault(
        "solution_collapse_detected",
        full_metrics.get("solution_collapse_detected"),
    )
    candidate_record = compact_candidate_record(
        public_record,
        metrics=artifact_metrics,
        training_report=training_summary,
        artifacts=artifacts,
    )
    write_candidate_json(target / "candidate_record.json", candidate_record)


def _progress(config: dict[str, Any], payload: dict[str, Any]) -> None:
    if config.get("progress_stdout") and console_enabled(config, "verbose"):
        print(json.dumps(payload, ensure_ascii=False, default=str), flush=True)


def _console_status(
    config: dict[str, Any], message: str, *, minimum: str = "normal"
) -> None:
    if console_enabled(config, minimum):
        print(str(message), flush=True)
