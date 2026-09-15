"""Formal phase-based trainer for BuiltPINN objects."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from copy import deepcopy
import math
from pathlib import Path
import random
import time
from typing import Any, Callable
from types import SimpleNamespace
import hashlib
import json

import numpy as np
import torch

from forge.pipeline.f_pinn_construction.builders import BuiltPINN
from forge.pipeline.f_pinn_construction.builders.pinn_builder import (
    OptimizationPhase,
)
from .controllers import (
    AdaptiveAction,
    AdvanceCurriculumStateAction,
    ReplaceSamplingSubsetAction,
    TrainerObservableState,
)
from .adaptive_reporting import (
    AdaptiveComputeBudgetLedger,
    build_adaptive_control_summary,
)


class _StopLBFGS(RuntimeError):
    """Internal control flow used to stop a numerically invalid L-BFGS step."""


@dataclass
class TrainingReport:
    success: bool
    actual_iterations: int
    phase_iterations: dict[str, int]
    training_time: float
    final_loss: float | None
    nan_detected: bool
    stopped_early: bool
    history: list[dict[str, Any]] = field(default_factory=list)
    failure_reason: str | None = None
    peak_gpu_memory: int = 0
    adaptive_refinement_events: list[dict[str, Any]] = field(default_factory=list)
    actual_peak_sampling_points: int = 0
    final_sampling_snapshot: dict[str, Any] = field(default_factory=dict)
    optimizer_audit: dict[str, Any] = field(default_factory=dict)
    time_strategy_audit: dict[str, Any] = field(default_factory=dict)
    component_gradient_audits: list[dict[str, Any]] = field(default_factory=list)
    reference_mse_checkpoints: dict[str, float | None] = field(default_factory=dict)
    last_checkpoint_iteration: int | None = None
    last_checkpoint_mse: float | None = None
    best_reference_mse_checkpoint: dict[str, Any] | None = None
    reference_mse_model_selection_enabled: bool = False
    adaptive_control_history: list[dict[str, Any]] = field(default_factory=list)
    adaptive_control_summary: dict[str, Any] = field(default_factory=dict)
    physics_validation_history: list[dict[str, Any]] = field(default_factory=list)
    best_training_loss_checkpoint: dict[str, Any] | None = None
    trainer_capabilities: dict[str, Any] = field(default_factory=dict)
    registry_schema_version: str | None = None
    final_model_policy: str = "last"
    sampling_control_summary: dict[str, Any] = field(default_factory=dict)
    curriculum_control_summary: dict[str, Any] = field(default_factory=dict)
    adaptive_compute_budget_summary: dict[str, Any] = field(default_factory=dict)
    runtime_scaled_policy_summary: dict[str, Any] = field(default_factory=dict)
    selected_checkpoint_source: str = "last"
    selected_checkpoint_step: int | None = None
    selected_checkpoint_metric: dict[str, Any] | None = None
    pre_lbfgs_physics_checkpoint: dict[str, Any] | None = None
    best_lbfgs_physics_checkpoint: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _minimum_physics_probe_history_entry(
    history: list[dict[str, Any]],
    *,
    lbfgs: bool,
) -> dict[str, Any] | None:
    eligible = [
        dict(item)
        for item in history
        if isinstance(item, dict)
        and ("lbfgs" in str(item.get("stage_id") or "").casefold())
        is lbfgs
        and isinstance(item.get("aggregate_score"), (int, float))
        and math.isfinite(float(item["aggregate_score"]))
    ]
    if not eligible:
        return None
    return min(eligible, key=lambda item: float(item["aggregate_score"]))


def _has_first_order_recovery_phase(phases: list[Any]) -> bool:
    """Return whether a completed schedule contains a usable Adam-family phase.

    Non-finite L-BFGS recovery is a numerical safety boundary, so it must not
    depend on the optional stagnation controller being enabled.
    """

    return any(
        str(getattr(phase, "optimizer_name", "")).casefold()
        in {"adam", "adamw", "radam", "nadam"}
        for phase in phases
    )


def collect_component_gradient_diagnostics(
    *,
    total_loss: torch.Tensor,
    component_losses: dict[str, torch.Tensor],
    parameters: list[torch.nn.Parameter],
    audit_iteration: int,
    gradient_imbalance_threshold: float = 100.0,
    gradient_conflict_threshold: float = -0.1,
    maximum_components: int = 12,
    maximum_conflicting_pairs: int = 5,
) -> dict[str, Any]:
    """Measure registered loss gradients without touching ``parameter.grad``."""

    trainable = [parameter for parameter in parameters if parameter.requires_grad]
    base = {
        "available": False,
        "reason": None,
        "audit_iteration": int(audit_iteration),
        "total_grad_norm": None,
        "component_grad_norms": {},
        "component_to_total_ratios": {},
        "unavailable_components": {},
        "max_to_min_nonzero_component_ratio": None,
        "dominant_component": None,
        "weakest_nonzero_component": None,
        "gradient_imbalance_threshold": float(gradient_imbalance_threshold),
        "gradient_imbalance_detected": None,
        "gradient_conflict_threshold": float(gradient_conflict_threshold),
        "gradient_conflict_detected": None,
        "pairwise_cosine_summary": {
            "available": False,
            "reason": "fewer_than_two_valid_nonzero_component_gradients",
            "minimum": None,
            "median": None,
            "conflicting_pairs": [],
            "evaluated_pair_count": 0,
        },
    }
    if not trainable:
        base["reason"] = "no_trainable_parameters"
        return base
    selected = sorted(
        component_losses.items(), key=lambda item: str(item[0])
    )[: max(2, int(maximum_components))]
    if not selected:
        base["reason"] = "no_registered_major_loss_components"
        return base
    total_vector, total_reason = _safe_autograd_vector(total_loss, trainable)
    if total_vector is not None:
        base["total_grad_norm"] = float(torch.linalg.vector_norm(total_vector).cpu())
    elif total_reason:
        base["total_gradient_unavailable_reason"] = total_reason
    vectors: dict[str, torch.Tensor] = {}
    norms: dict[str, float] = {}
    unavailable: dict[str, str] = {}
    for name, component_loss in selected:
        vector, reason = _safe_autograd_vector(component_loss, trainable)
        if vector is None:
            unavailable[str(name)] = reason or "component_gradient_unavailable"
            continue
        norm = float(torch.linalg.vector_norm(vector).cpu())
        vectors[str(name)] = vector
        norms[str(name)] = norm
    base["component_grad_norms"] = norms
    base["unavailable_components"] = unavailable
    total_norm = base.get("total_grad_norm")
    if isinstance(total_norm, float) and total_norm > 0.0:
        base["component_to_total_ratios"] = {
            name: value / total_norm for name, value in norms.items()
        }
    nonzero = {name: value for name, value in norms.items() if value > 0.0}
    if nonzero:
        base["dominant_component"] = max(nonzero, key=nonzero.get)
        base["weakest_nonzero_component"] = min(nonzero, key=nonzero.get)
    if len(nonzero) >= 2:
        ratio = max(nonzero.values()) / min(nonzero.values())
        base["max_to_min_nonzero_component_ratio"] = ratio
        base["gradient_imbalance_detected"] = bool(
            ratio > float(gradient_imbalance_threshold)
        )
    else:
        base["gradient_imbalance_unavailable_reason"] = (
            "fewer_than_two_valid_nonzero_component_gradients"
        )
    pairwise: list[dict[str, Any]] = []
    names = list(nonzero)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            left = vectors[left_name]
            right = vectors[right_name]
            denominator = nonzero[left_name] * nonzero[right_name]
            cosine = float(torch.dot(left, right).cpu()) / denominator
            cosine = max(-1.0, min(1.0, cosine))
            pairwise.append(
                {
                    "component_a": left_name,
                    "component_b": right_name,
                    "cosine_similarity": cosine,
                }
            )
    if pairwise:
        cosine_values = sorted(
            float(item["cosine_similarity"]) for item in pairwise
        )
        midpoint = len(cosine_values) // 2
        median = (
            cosine_values[midpoint]
            if len(cosine_values) % 2
            else 0.5
            * (cosine_values[midpoint - 1] + cosine_values[midpoint])
        )
        conflicting = sorted(
            [
                item
                for item in pairwise
                if float(item["cosine_similarity"])
                < float(gradient_conflict_threshold)
            ],
            key=lambda item: float(item["cosine_similarity"]),
        )[: max(1, int(maximum_conflicting_pairs))]
        base["pairwise_cosine_summary"] = {
            "available": True,
            "reason": None,
            "minimum": cosine_values[0],
            "median": median,
            "conflicting_pairs": conflicting,
            "evaluated_pair_count": len(pairwise),
        }
        base["gradient_conflict_detected"] = bool(conflicting)
    base["available"] = bool(norms)
    base["reason"] = None if norms else "no_component_gradients_available"
    return base


def _safe_autograd_vector(
    loss: torch.Tensor,
    parameters: list[torch.nn.Parameter],
) -> tuple[torch.Tensor | None, str | None]:
    if not isinstance(loss, torch.Tensor) or not loss.requires_grad:
        return None, "loss_does_not_require_grad"
    try:
        gradients = torch.autograd.grad(
            loss,
            parameters,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
    except (RuntimeError, ValueError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if not any(gradient is not None for gradient in gradients):
        return None, "all_parameter_gradients_unused"
    chunks: list[torch.Tensor] = []
    for parameter, gradient in zip(parameters, gradients):
        if gradient is None:
            chunks.append(
                torch.zeros(
                    parameter.numel(),
                    dtype=parameter.dtype,
                    device="cpu",
                )
            )
            continue
        detached = gradient.detach().reshape(-1)
        if not torch.isfinite(detached).all():
            return None, "non_finite_component_gradient"
        chunks.append(detached.to(device="cpu"))
    return torch.cat(chunks), None


class OpenSpecTrainer:
    def __init__(
        self,
        built: BuiltPINN,
        *,
        checkpoint_dir: str | Path | None = None,
        history_interval: int = 1,
        reference_mse_probe: Callable[[], float | None] | None = None,
        reference_mse_interval: int = 0,
        reference_mse_checkpoints: list[int] | tuple[int, ...] | None = None,
        select_best_reference_mse_checkpoint: bool = False,
        component_gradient_diagnostics_enabled: bool = True,
        gradient_imbalance_threshold: float = 100.0,
        gradient_conflict_threshold: float = -0.1,
        maximum_gradient_components: int = 12,
        maximum_conflicting_pairs: int = 5,
        runtime_scaled_policy_summary: dict[str, Any] | None = None,
    ) -> None:
        self.built = built
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.history_interval = max(1, int(history_interval))
        self.history: list[dict[str, Any]] = []
        self._history_phase_names_recorded: set[str] = set()
        self._history_boundary_iterations: set[int] = set()
        self.completed_phase_iterations: dict[str, int] = {}
        self._train_started = 0.0
        self.runtime_scaled_policy_summary = deepcopy(
            runtime_scaled_policy_summary or {}
        )
        self.adaptive_compute_budget_ledger = AdaptiveComputeBudgetLedger()
        self._actual_iterations = 0
        self.adaptive_refinement_events: list[dict[str, Any]] = []
        self.optimizer_audit: dict[str, Any] = {}
        self.reference_mse_probe = reference_mse_probe
        self.reference_mse_interval = max(0, int(reference_mse_interval))
        self.reference_mse_checkpoint_iterations = tuple(
            sorted(
                {
                    int(item)
                    for item in (reference_mse_checkpoints or [])
                    if int(item) > 0
                }
            )
        )
        self.reference_mse_checkpoints: dict[str, float | None] = {}
        self.last_checkpoint_iteration: int | None = None
        self.last_checkpoint_mse: float | None = None
        self.select_best_reference_mse_checkpoint = bool(
            select_best_reference_mse_checkpoint
            and reference_mse_probe is not None
        )
        self.best_reference_mse_checkpoint: dict[str, Any] | None = None
        self._best_reference_mse_model_state: dict[str, torch.Tensor] | None = None
        self._selected_final_model_policy: str | None = None
        self._reference_probe_phases: set[str] = set()
        self.time_strategy_audit: dict[str, Any] = {}
        self.component_gradient_diagnostics_enabled = bool(
            component_gradient_diagnostics_enabled
        )
        self.gradient_imbalance_threshold = max(
            1.0, float(gradient_imbalance_threshold)
        )
        self.gradient_conflict_threshold = float(gradient_conflict_threshold)
        self.maximum_gradient_components = max(2, int(maximum_gradient_components))
        self.maximum_conflicting_pairs = max(1, int(maximum_conflicting_pairs))
        self.component_gradient_audits: list[dict[str, Any]] = []
        self._gradient_audit_targets: list[int] = []
        self._gradient_audit_targets_completed: set[int] = set()
        self._last_training_batch: Any | None = None
        self._last_training_phase: Any | None = None
        self.gradient_balance_controller = getattr(
            built, "gradient_balance_controller", None
        )
        self.lbfgs_stall_controller = getattr(built, "lbfgs_stall_controller", None)
        self.rollback_manager = getattr(built, "rollback_manager", None)
        self.adaptive_audit_logger = getattr(built, "adaptive_audit_logger", None)
        if self.adaptive_audit_logger is not None:
            audit_path = (
                self.checkpoint_dir / "adaptive_control_history.jsonl"
                if self.checkpoint_dir is not None
                else None
            )
            self.adaptive_audit_logger.bind(audit_path)
        self.adaptive_control_summary: dict[str, Any] = {
            "lbfgs_stall_detected": False,
            "lbfgs_stall_iteration": None,
            "planned_lbfgs_iterations": 0,
            "executed_lbfgs_iterations": 0,
            "recovered_iteration_budget": 0,
            "unused_iteration_budget": 0,
            "recovery_action": None,
            "gradient_balance_actions": 0,
            "rollbacks": 0,
        }
        self._pending_recovery: dict[str, Any] | None = None
        self.registry = getattr(built, "training_component_registry", None)
        self.action_boundary_validator = getattr(
            built, "action_boundary_validator", None
        )
        self.controller_coordinator = getattr(
            built, "controller_coordinator", None
        )
        self.validation_probe_manager = getattr(
            built, "validation_probe_manager", None
        )
        self.best_training_loss_checkpoint_manager = getattr(
            built, "best_training_loss_checkpoint_manager", None
        )
        self.fixed_budget_sampling_controller = getattr(
            built, "fixed_budget_sampling_controller", None
        )
        self.curriculum_runtimes: dict[str, Any] = dict(
            getattr(built, "curriculum_runtimes", {}) or {}
        )
        self._initial_curriculum_runtime_states = {
            curriculum_id: deepcopy(runtime.snapshot_state())
            for curriculum_id, runtime in self.curriculum_runtimes.items()
        }
        self.curriculum_advance_controller = getattr(
            built, "curriculum_advance_controller", None
        )
        self.sampling_control_summary: dict[str, Any] = {
            "sampling_diagnostic_evaluations": 0,
            "sampling_points_diagnosed": 0,
            "candidate_points_evaluated": 0,
            "sampling_updates_executed": 0,
            "sampling_updates_rejected": 0,
            "sampling_updates_rolled_back": 0,
            "sampling_points_replaced": 0,
            "sampling_compute_time": 0.0,
        }
        self._pending_sampling_observation: dict[str, Any] | None = None
        self._sampling_score_cache: dict[str, Any] = {}
        self.curriculum_control_summary: dict[str, Any] = {
            "curriculum_diagnostic_checks": 0,
            "curriculum_advances_proposed": 0,
            "curriculum_advances_executed": 0,
            "curriculum_advances_forced": 0,
            "curriculum_advances_rejected": 0,
            "curriculum_advances_rolled_back": 0,
            "curriculum_runtime_refresh_cost": 0.0,
            "curriculum_validation_cost": 0.0,
            "iterations_per_level": {},
            "unused_levels": {},
        }
        self._pending_curriculum_observation: dict[str, Any] | None = None
        self._curriculum_resume_disabled = False
        self._curriculum_termination_requested = False
        self._algorithm_spec_id = hashlib.sha256(
            json.dumps(
                getattr(self.built, "normalized_spec", {}),
                sort_keys=True,
                ensure_ascii=False,
                default=str,
            ).encode("utf-8")
        ).hexdigest()[:20]

    def train(self, maximum_iterations: int | None = None) -> TrainingReport:
        started = time.perf_counter()
        self._train_started = started
        self._actual_iterations = sum(self.completed_phase_iterations.values())
        device = next(self.built.model.parameters()).device
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        remaining = None if maximum_iterations is None else max(0, int(maximum_iterations))
        available_iterations = sum(
            max(
                0,
                int(phase.iterations)
                - int(self.completed_phase_iterations.get(phase.name, 0)),
            )
            for phase in self.built.optimization_phases
        )
        planned_iterations = (
            available_iterations
            if remaining is None
            else min(available_iterations, remaining)
        )
        self._configure_gradient_audit_targets(
            start_iteration=self._actual_iterations,
            planned_iterations=planned_iterations,
        )
        self._configure_history_boundary_iterations(
            start_iteration=self._actual_iterations,
            maximum_iterations=planned_iterations,
        )
        phase_counts: dict[str, int] = {}
        nan_detected = False
        stopped_early = False
        failure_reason = None
        final_loss: float | None = None
        training = self.built.normalized_spec["training"]
        clipping = training["gradient_clipping"]
        clipping_parameters = clipping.get("parameters") or {}
        clipping_norm = float(clipping_parameters.get("max_norm", 1.0)) if clipping.get("enabled") else None
        clipping_norm_type = float(clipping_parameters.get("norm_type", 2.0))
        early = training["early_stopping"]
        early_parameters = early.get("parameters") or {}
        threshold = float(early_parameters.get("threshold", -float("inf"))) if early.get("enabled") else -float("inf")
        patience = max(1, int(early_parameters.get("patience", 1))) if early.get("enabled") else 1
        below_threshold = 0
        adaptive_parameters = self.built.normalized_spec["sampling"]["adaptive_refinement"].get("parameters") or {}
        adaptive_interval = max(1, int(adaptive_parameters.get("update_every", 100)))
        non_lbfgs_completed = sum(
            int(self.completed_phase_iterations.get(phase.name, 0))
            for phase in self.built.optimization_phases
            if self._stage_descriptor(phase).first_order
        )
        self._initialize_physics_validation()

        try:
            for phase in self.built.optimization_phases:
                completed_before = int(self.completed_phase_iterations.get(phase.name, 0))
                available = max(0, int(phase.iterations) - completed_before)
                phase_budget = available if remaining is None else min(available, remaining)
                phase_counts[phase.name] = 0
                if phase_budget <= 0:
                    continue
                stage_descriptor = self._stage_descriptor(phase)
                if stage_descriptor.second_order:
                    self._maybe_run_physics_validation(
                        phase=SimpleNamespace(name="pre_second_order"),
                        force=True,
                        training_loss=final_loss,
                    )
                    if (
                        self.lbfgs_stall_controller is not None
                        and self.lbfgs_stall_controller.triggered
                    ):
                        pending = dict(self._pending_recovery or {})
                        recovery_budget = min(
                            int(pending.get("iterations") or 0), phase_budget
                        )
                        if recovery_budget > 0:
                            (
                                final_loss,
                                recovered,
                                recovery_failure,
                            ) = self._run_recovery_adam(
                                recovery_budget,
                                clipping_norm,
                                clipping_norm_type,
                                adaptive_interval,
                            )
                            phase_counts["adaptive_recovery_adam"] = recovered
                            self.completed_phase_iterations[
                                "adaptive_recovery_adam"
                            ] = int(
                                self.completed_phase_iterations.get(
                                    "adaptive_recovery_adam", 0
                                )
                            ) + recovered
                            if recovery_failure is not None:
                                failure_reason = recovery_failure
                            if remaining is not None:
                                remaining = max(0, remaining - recovered)
                        self._pending_recovery = None
                        self._emit_adaptive_audit(
                            iteration=self._actual_iterations,
                            stage="lbfgs",
                            controller="lbfgs_stall",
                            event_type="stalled_phase_skipped_on_resume",
                            trigger={"already_triggered": True},
                        )
                        continue
                    self._configure_training_stage(
                        phase=phase,
                        non_lbfgs_iteration=non_lbfgs_completed,
                    )
                    previous_training_mode = bool(self.built.model.training)
                    self.built.model.eval()
                    try:
                        final_loss, iterations, lbfgs_failure, lbfgs_audit = self._run_lbfgs(
                            phase, phase_budget, clipping_norm, clipping_norm_type
                        )
                    finally:
                        self.built.model.train(previous_training_mode)
                        unfreeze = getattr(
                            self.built.loss_function,
                            "unfreeze_dynamic_weighting",
                            None,
                        )
                        if callable(unfreeze):
                            unfreeze()
                    phase_counts[phase.name] = iterations
                    self.optimizer_audit[phase.name] = lbfgs_audit
                    self.adaptive_control_summary.update(
                        {
                            "planned_lbfgs_iterations": int(phase_budget),
                            "executed_lbfgs_iterations": int(iterations),
                            "lbfgs_stall_detected": bool(
                                lbfgs_audit.get("controlled_stall_detected", False)
                            ),
                            "lbfgs_stall_iteration": lbfgs_audit.get(
                                "stall_iteration"
                            ),
                            "recovery_action": lbfgs_audit.get("recovery_action"),
                        }
                    )
                    if lbfgs_failure is not None:
                        nan_detected = lbfgs_failure.startswith("non_finite")
                        failure_reason = lbfgs_failure
                    recovery_budget = int(
                        lbfgs_audit.get("recovery_adam_iterations") or 0
                    )
                    self._pending_recovery = (
                        {
                            "phase": phase.name,
                            "iterations": recovery_budget,
                            "action": lbfgs_audit.get("recovery_action"),
                        }
                        if lbfgs_audit.get("controlled_stall_detected")
                        else None
                    )
                    self.adaptive_control_summary["unused_iteration_budget"] = max(
                        0, phase_budget - iterations - recovery_budget
                    )
                    if self._pending_recovery is not None and self.checkpoint_dir is not None:
                        self.completed_phase_iterations[phase.name] = (
                            completed_before + iterations
                        )
                        self.save_checkpoint(
                            self.checkpoint_dir
                            / "adaptive_control"
                            / "recovery_resume_checkpoint.pt"
                        )
                    if recovery_budget > 0 and failure_reason is None:
                        (
                            final_loss,
                            recovered,
                            recovery_failure,
                        ) = self._run_recovery_adam(
                            recovery_budget,
                            clipping_norm,
                            clipping_norm_type,
                            adaptive_interval,
                        )
                        phase_counts["adaptive_recovery_adam"] = recovered
                        self.completed_phase_iterations[
                            "adaptive_recovery_adam"
                        ] = int(
                            self.completed_phase_iterations.get(
                                "adaptive_recovery_adam", 0
                            )
                        ) + recovered
                        non_lbfgs_completed += recovered
                        self.adaptive_control_summary[
                            "recovered_iteration_budget"
                        ] = recovered
                        self.adaptive_control_summary["unused_iteration_budget"] = max(
                            0, phase_budget - iterations - recovered
                        )
                        if recovery_failure is not None:
                            failure_reason = recovery_failure
                            nan_detected = recovery_failure.startswith("non_finite")
                    self._pending_recovery = None
                    if remaining is not None:
                        remaining = max(
                            0,
                            remaining
                            - iterations
                            - int(phase_counts.get("adaptive_recovery_adam", 0)),
                        )
                else:
                    for _ in range(phase_budget):
                        stage = self._configure_training_stage(
                            phase=phase,
                            non_lbfgs_iteration=non_lbfgs_completed + 1,
                        )
                        batch = self.built.sampler.sample()
                        self._last_training_batch = batch
                        self._last_training_phase = phase
                        phase.optimizer.zero_grad(set_to_none=True)
                        self.built.loss_function.capture_residual_statistics = self._should_record_history(
                            self._actual_iterations + 1,
                            phase.name,
                        )
                        loss, components = self.built.loss_function(self.built.model, batch)
                        if not torch.isfinite(loss):
                            nan_detected = True
                            failure_reason = "non_finite_loss"
                            self._release_iteration_graphs()
                            break
                        gradient_statistics = None
                        projected_iteration = self._actual_iterations + 1
                        controller_diagnostic_due = bool(
                            self.gradient_balance_controller is not None
                            and stage_descriptor.first_order
                            and stage_descriptor.dynamic_objective_allowed
                            and projected_iteration
                            % int(
                                self.gradient_balance_controller.config[
                                    "diagnostic_interval"
                                ]
                            )
                            == 0
                        )
                        if (
                            controller_diagnostic_due
                            or (
                                self._should_record_history(
                                    projected_iteration,
                                    phase.name,
                                )
                                and self._gradient_diagnostic_due(projected_iteration)
                            )
                        ):
                            gradient_statistics = (
                                self._collect_component_gradient_diagnostic(
                                    total_loss=loss,
                                    iteration=projected_iteration,
                                    phase=phase,
                                    checkpoint_kind=self._gradient_checkpoint_kind(
                                        projected_iteration
                                    ),
                                )
                            )
                        if phase.optimizer_name == "multiadam":
                            multiadam_losses = (
                                self.built.loss_function.last_term_values
                            )
                            if getattr(phase.optimizer, "grouping", None) is not None:
                                effective_weights = (
                                    self.built.loss_function.last_effective_loss_weights
                                )
                                multiadam_losses = [
                                    value
                                    * float(effective_weights.get(name, 1.0))
                                    for name, value in zip(
                                        self.built.loss_function.last_term_names,
                                        self.built.loss_function.last_term_values,
                                    )
                                ]
                            phase.optimizer.step_losses(
                                multiadam_losses,
                                self.built.loss_function.last_term_names,
                            )
                            self.optimizer_audit[phase.name] = {
                                "optimizer": "multiadam",
                                "grouping": getattr(
                                    phase.optimizer, "grouping", None
                                ),
                                "loss_groups": dict(
                                    getattr(
                                        phase.optimizer,
                                        "last_loss_groups",
                                        {},
                                    )
                                ),
                            }
                            if not self._parameters_finite():
                                nan_detected = True
                                failure_reason = "non_finite_parameter"
                                self._release_iteration_graphs()
                                break
                        else:
                            loss.backward()
                            if clipping_norm is not None:
                                torch.nn.utils.clip_grad_norm_(self.built.model.parameters(), clipping_norm, norm_type=clipping_norm_type)
                            if not self._gradients_finite():
                                nan_detected = True
                                failure_reason = "non_finite_gradient"
                                self._release_iteration_graphs()
                                break
                            gradient_norm = self._gradient_norm()
                            phase.optimizer.step()
                        if phase.optimizer_name == "multiadam":
                            gradient_norm = None
                        final_loss = float(loss.detach().cpu())
                        phase_counts[phase.name] += 1
                        self._actual_iterations += 1
                        non_lbfgs_completed += 1
                        self._note_curriculum_training_iteration()
                        if self.gradient_balance_controller is not None:
                            self._observe_gradient_balance(
                                phase=phase,
                                loss=final_loss,
                                components=components,
                                gradient_norm=gradient_norm,
                                diagnostic=gradient_statistics,
                            )
                        if (
                            self.rollback_manager is not None
                            and self._pending_curriculum_observation is None
                            and self.rollback_manager.observe_loss(final_loss)
                        ):
                            self._rollback_active_action(
                                phase=phase,
                                reason="post_action_loss_degradation_or_non_finite",
                            )
                        self._maybe_run_physics_validation(
                            phase=phase,
                            force=False,
                            training_loss=final_loss,
                        )
                        self._observe_sampling_control(phase=phase)
                        self._observe_curriculum_control(phase=phase)
                        if self._curriculum_termination_requested:
                            stopped_early = True
                            failure_reason = "curriculum_forced_termination"
                            self._release_iteration_graphs()
                            break
                        self._record_reference_mse_checkpoint(
                            self._actual_iterations
                        )
                        adaptive_event = self._actual_iterations % adaptive_interval == 0
                        dynamic_update = bool(
                            (
                                getattr(
                                    self.built.loss_function,
                                    "last_dynamic_weighting",
                                    None,
                                )
                                or {}
                            ).get("updated")
                        )
                        should_record = self._should_record_history(
                            self._actual_iterations,
                            phase.name,
                            event=adaptive_event or dynamic_update,
                        )
                        if should_record:
                            self.history.append(
                                self._history_entry(
                                    iteration=self._actual_iterations,
                                    phase=phase,
                                    total_loss=final_loss,
                                    components=components,
                                    batch=batch,
                                    gradient_norm=gradient_norm,
                                    gradient_statistics=gradient_statistics,
                                    adaptive_refinement_event=adaptive_event,
                                    training_stage=stage,
                                )
                            )
                            self._history_phase_names_recorded.add(str(phase.name))
                        self._scheduler_step(phase.scheduler, final_loss)
                        if adaptive_event:
                            self.built.sampler.adaptive_update(self.built.model)
                            sampler_events = getattr(
                                self.built.sampler, "adaptive_refinement_events", []
                            )
                            sampler_event = dict(sampler_events[-1]) if sampler_events else {}
                            self.adaptive_refinement_events.append(
                                {
                                    **sampler_event,
                                    "iteration": self._actual_iterations,
                                    "phase": phase.name,
                                    "elapsed_seconds": time.perf_counter() - self._train_started,
                                }
                            )
                            if (
                                self.history
                                and self.history[-1].get("iteration")
                                == self._actual_iterations
                            ):
                                self.history[-1][
                                    "adaptive_refinement_event"
                                ] = dict(self.adaptive_refinement_events[-1])
                        below_threshold = below_threshold + 1 if final_loss <= threshold else 0
                        if below_threshold >= patience:
                            stopped_early = True
                            if not should_record:
                                self.history.append(
                                    self._history_entry(
                                        iteration=self._actual_iterations,
                                        phase=phase,
                                        total_loss=final_loss,
                                        components=components,
                                        batch=batch,
                                        gradient_norm=gradient_norm,
                                        gradient_statistics=gradient_statistics,
                                        adaptive_refinement_event=adaptive_event,
                                        training_stage=stage,
                                    )
                                )
                                self._history_phase_names_recorded.add(
                                    str(phase.name)
                                )
                            self.history[-1]["early_stopping"] = True
                            self._release_iteration_graphs()
                            break
                        self._release_iteration_graphs()
                    if remaining is not None:
                        remaining = max(0, remaining - phase_counts[phase.name])
                self.completed_phase_iterations[phase.name] = completed_before + phase_counts[phase.name]
                if nan_detected or stopped_early or remaining == 0:
                    break
            if failure_reason is None and not nan_detected:
                self._collect_final_component_gradient_diagnostic()
                self._maybe_run_physics_validation(
                    phase=self._last_training_phase,
                    force=True,
                    training_loss=final_loss,
                )
                # Capture the model produced by the final optimizer step before
                # best-physics model selection can restore an earlier state.
                self._record_last_reference_mse_checkpoint()
                if self.checkpoint_dir is not None:
                    terminal_path = (
                        self.checkpoint_dir / "last_optimizer_checkpoint.pt"
                    )
                    terminal_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "model_state_dict": {
                                name: value.detach().clone().cpu()
                                for name, value in self.built.model.state_dict().items()
                            },
                            "iteration": int(self._actual_iterations),
                            "physics_validation": (
                                self.validation_probe_manager.last_result.to_dict()
                                if self.validation_probe_manager is not None
                                and self.validation_probe_manager.last_result is not None
                                else None
                            ),
                            "reference_metrics_used_for_selection": False,
                        },
                        terminal_path,
                    )
                if self._restore_best_reference_mse_model():
                    self._selected_final_model_policy = "best_reference_mse"
                elif (
                    self.best_training_loss_checkpoint_manager is not None
                    and self.best_training_loss_checkpoint_manager.final_model_policy
                    == "best_train_loss"
                    and self._pending_curriculum_observation is None
                ):
                    if self.best_training_loss_checkpoint_manager.restore_best_model(
                        self.built.model
                    ):
                        self._selected_final_model_policy = (
                            self.best_training_loss_checkpoint_manager.final_model_policy
                        )
            if self.checkpoint_dir is not None:
                self.save_checkpoint(self.checkpoint_dir / "final_checkpoint.pt")
                if self.history:
                    self.history[-1]["checkpoint_iteration"] = True
        except Exception as exc:
            failure_reason = f"{type(exc).__name__}: {exc}"

        actual = sum(phase_counts.values())
        training_wall_time = time.perf_counter() - started
        peak_gpu_memory = (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda" and torch.cuda.is_available()
            else 0
        )
        sampler = getattr(self.built, "sampler", None)
        adaptive_history = list(
            getattr(self.adaptive_audit_logger, "records", []) or []
        )
        physics_history = list(
            getattr(self.validation_probe_manager, "history", []) or []
        )
        best_training_loss_metadata = deepcopy(
            getattr(
                self.best_training_loss_checkpoint_manager,
                "best_metadata",
                None,
            )
        )
        pre_lbfgs_physics_checkpoint = _minimum_physics_probe_history_entry(
            physics_history, lbfgs=False
        )
        best_lbfgs_physics_checkpoint = _minimum_physics_probe_history_entry(
            physics_history, lbfgs=True
        )
        curriculum_summary = deepcopy(
            self._finalize_curriculum_control_summary()
        )
        probe_budgets = {}
        if (
            self.validation_probe_manager is not None
            and self.validation_probe_manager.probe_set is not None
        ):
            probe_budgets = dict(
                self.validation_probe_manager.probe_set.component_budgets
            )
        compute_budget = self.adaptive_compute_budget_ledger.reconcile(
            actual_iterations=self._actual_iterations,
            phase_iterations=phase_counts,
            optimizer_audit=self.optimizer_audit,
            component_gradient_audits=self.component_gradient_audits,
            sampling_summary=self.sampling_control_summary,
            curriculum_summary=curriculum_summary,
            physics_history=physics_history,
            probe_component_budgets=probe_budgets,
            adaptive_summary=self.adaptive_control_summary,
            adaptive_history=adaptive_history,
            training_wall_time=training_wall_time,
        )
        runtime_weights = {}
        get_weights = getattr(
            self.built.loss_function, "get_runtime_loss_weights", None
        )
        if callable(get_weights):
            runtime_weights = dict(get_weights())
        final_model_policy = str(
            self._selected_final_model_policy
            or (
                self.best_training_loss_checkpoint_manager.final_model_policy
                if self.best_training_loss_checkpoint_manager is not None
                else "last"
            )
        )
        compact_control_summary = build_adaptive_control_summary(
            spec=self.built.normalized_spec,
            runtime_loss_weights=runtime_weights,
            component_gradient_audits=self.component_gradient_audits,
            adaptive_history=adaptive_history,
            adaptive_summary=self.adaptive_control_summary,
            sampling_summary=self.sampling_control_summary,
            curriculum_summary=curriculum_summary,
            physics_history=physics_history,
            best_training_loss=deepcopy(
                getattr(
                    self.best_training_loss_checkpoint_manager,
                    "best_metadata",
                    None,
                )
            ),
            final_model_policy=final_model_policy,
            compute_budget=compute_budget,
        )
        report_adaptive_summary = {
            **deepcopy(self.adaptive_control_summary),
            **compact_control_summary,
        }
        return TrainingReport(
            success=failure_reason is None and not nan_detected,
            actual_iterations=actual,
            phase_iterations=phase_counts,
            training_time=training_wall_time,
            final_loss=final_loss,
            nan_detected=nan_detected,
            stopped_early=stopped_early,
            history=self.history,
            failure_reason=failure_reason,
            peak_gpu_memory=peak_gpu_memory,
            adaptive_refinement_events=self.adaptive_refinement_events,
            actual_peak_sampling_points=int(
                getattr(sampler, "peak_total_sampling_points", 0)
            ),
            final_sampling_snapshot=dict(
                getattr(sampler, "last_sampling_snapshot", {}) or {}
            ),
            optimizer_audit=dict(self.optimizer_audit),
            time_strategy_audit=dict(self.time_strategy_audit),
            component_gradient_audits=list(self.component_gradient_audits),
            reference_mse_checkpoints=dict(self.reference_mse_checkpoints),
            last_checkpoint_iteration=self.last_checkpoint_iteration,
            last_checkpoint_mse=self.last_checkpoint_mse,
            best_reference_mse_checkpoint=deepcopy(
                self.best_reference_mse_checkpoint
            ),
            reference_mse_model_selection_enabled=bool(
                self.select_best_reference_mse_checkpoint
            ),
            adaptive_control_history=adaptive_history,
            adaptive_control_summary=report_adaptive_summary,
            physics_validation_history=physics_history,
            best_training_loss_checkpoint=deepcopy(
                best_training_loss_metadata
                if final_model_policy == "best_train_loss"
                else None
            ),
            trainer_capabilities=(
                self.registry.capabilities.to_dict()
                if self.registry is not None
                else {}
            ),
            registry_schema_version=(
                self.registry.schema_version if self.registry is not None else None
            ),
            final_model_policy=final_model_policy,
            sampling_control_summary=deepcopy(self.sampling_control_summary),
            curriculum_control_summary=curriculum_summary,
            adaptive_compute_budget_summary=compute_budget,
            runtime_scaled_policy_summary=deepcopy(
                self.runtime_scaled_policy_summary
            ),
            selected_checkpoint_source=final_model_policy,
            selected_checkpoint_step=(
                int(best_training_loss_metadata["iteration"])
                if final_model_policy == "best_train_loss"
                and best_training_loss_metadata is not None
                else int(self.best_reference_mse_checkpoint["iteration"])
                if final_model_policy == "best_reference_mse"
                and self.best_reference_mse_checkpoint is not None
                else self.last_checkpoint_iteration
            ),
            selected_checkpoint_metric=(
                {
                    "name": str(
                            best_training_loss_metadata.get("selection_metric", "training_loss")
                    ),
                    "value": float(
                        best_training_loss_metadata["selection_score"]
                    ),
                }
                if final_model_policy == "best_train_loss"
                and best_training_loss_metadata is not None
                else {
                    "name": "reference_mse",
                    "value": float(
                        self.best_reference_mse_checkpoint["mse"]
                    ),
                }
                if final_model_policy == "best_reference_mse"
                and self.best_reference_mse_checkpoint is not None
                else {
                    "name": "training_loss",
                    "value": final_loss,
                }
            ),
            pre_lbfgs_physics_checkpoint=pre_lbfgs_physics_checkpoint,
            best_lbfgs_physics_checkpoint=best_lbfgs_physics_checkpoint,
        )

    def _emit_adaptive_audit(
        self,
        *,
        iteration: int,
        stage: str,
        controller: str,
        event_type: str,
        trigger: dict[str, Any] | None = None,
        before: dict[str, Any] | None = None,
        after: dict[str, Any] | None = None,
        limits: dict[str, Any] | None = None,
        checkpoint_id: str | None = None,
        rollback_status: str = "not_required",
    ) -> None:
        if self.adaptive_audit_logger is None:
            return
        self.adaptive_audit_logger.emit(
            {
                "iteration": int(iteration),
                "stage": str(stage),
                "controller": str(controller),
                "event_type": str(event_type),
                "trigger": deepcopy(trigger or {}),
                "before": deepcopy(before or {}),
                "after": deepcopy(after or {}),
                "limits": deepcopy(limits or {}),
                "checkpoint_id": checkpoint_id,
                "rollback_status": rollback_status,
            }
        )

    def _observe_gradient_balance(
        self,
        *,
        phase: Any,
        loss: float,
        components: dict[str, float],
        gradient_norm: float | None,
        diagnostic: dict[str, Any] | None,
    ) -> None:
        controller = self.gradient_balance_controller
        if controller is None:
            return
        diagnostic = dict(diagnostic or {})
        pairwise = dict(diagnostic.get("pairwise_cosine_summary") or {})
        stage_descriptor = self._stage_descriptor(phase)
        eligible_component_ids = (
            self.registry.active_weightable_component_ids(phase.name)
            if self.registry is not None
            else tuple(sorted(diagnostic.get("component_grad_norms") or {}))
        )
        state = TrainerObservableState(
            iteration=int(self._actual_iterations),
            stage=str(phase.name),
            optimizer_name=str(phase.optimizer_name),
            current_loss=float(loss),
            loss_components=dict(components),
            loss_weights=dict(
                getattr(
                    self.built.loss_function,
                    "get_runtime_loss_weights",
                    lambda: {},
                )()
            ),
            component_gradient_norms=dict(
                diagnostic.get("component_grad_norms") or {}
            ),
            minimum_gradient_cosine=pairwise.get("minimum"),
            learning_rates=tuple(
                float(group.get("lr", 0.0))
                for group in phase.optimizer.param_groups
            ),
            gradient_norm=gradient_norm,
            completed_stage_iterations=int(
                self.completed_phase_iterations.get(phase.name, 0)
            ),
            sampling_state=dict(
                getattr(self.built.sampler, "last_sampling_snapshot", {}) or {}
            ),
            time_window_fraction=float(
                getattr(self.built.sampler, "time_window_fraction", 1.0)
            ),
            numerical_finite=math.isfinite(float(loss)),
            stage_descriptor=stage_descriptor,
            capability_snapshot=(
                self.registry.capabilities if self.registry is not None else None
            ),
            eligible_component_ids=eligible_component_ids,
            component_gradient_ema=dict(controller.ema_gradient_norms),
            optimizer_diagnostics=dict(diagnostic),
            budget_state={
                "completed_stage_iterations": int(
                    self.completed_phase_iterations.get(phase.name, 0)
                ),
                "remaining_stage_iterations": max(
                    0,
                    int(getattr(phase, "iterations", 0))
                    - int(self.completed_phase_iterations.get(phase.name, 0)),
                ),
            },
            validation_state=getattr(
                self.validation_probe_manager, "last_result", None
            ),
        )
        if self.controller_coordinator is not None:
            observation = self.controller_coordinator.observe_controller(
                controller, state
            )
        else:
            observation = controller.observe(state)
        if observation.get("reason") != "diagnostic_interval_not_reached":
            event_type = "diagnostic"
            if observation.get("gradient_conflict"):
                event_type = "action_rejected_gradient_conflict"
            elif observation.get("in_cooldown"):
                event_type = "action_rejected_cooldown"
            self._emit_adaptive_audit(
                iteration=self._actual_iterations,
                stage=str(phase.name),
                controller=controller.name,
                event_type=event_type,
                trigger=observation,
            )
        proposals = (
            self.controller_coordinator.collect_proposals()
            if self.controller_coordinator is not None
            else tuple(
                item
                for item in (controller.propose_action(),)
                if item is not None
            )
        )
        if self.controller_coordinator is not None:
            action, rejected = self.controller_coordinator.select_action(
                proposals, iteration=self._actual_iterations
            )
            for action_id, reason in rejected.items():
                self._emit_adaptive_audit(
                    iteration=self._actual_iterations,
                    stage=str(phase.name),
                    controller="controller_coordinator",
                    event_type="action_rejected_by_coordinator",
                    trigger={"action_id": action_id, "reason": reason},
                )
        else:
            action = proposals[0] if proposals else None
        if action is None:
            return
        action_signature = self._action_signature(action)
        if (
            self.rollback_manager is not None
            and action_signature
            in self.rollback_manager.failed_action_signatures
        ):
            controller.mark_action_applied(self._actual_iterations)
            self._emit_adaptive_audit(
                iteration=self._actual_iterations,
                stage=str(phase.name),
                controller=action.controller,
                event_type="action_rejected_previous_rollback",
                trigger=dict(action.reason),
                before=dict(action.before),
                after=dict(action.after),
                limits=dict(action.limits),
            )
            return
        checkpoint_id = self._save_pre_action_checkpoint(action, loss)
        try:
            self.apply_validated_action(action, observable_state=state)
            if action.action_type == "terminate_candidate":
                controller.pending_action = None
                if self.controller_coordinator is not None:
                    self.controller_coordinator.record_outcome(
                        action,
                        iteration=self._actual_iterations,
                        outcome="applied",
                    )
                self._emit_adaptive_audit(
                    iteration=self._actual_iterations,
                    stage=str(phase.name),
                    controller=controller.name,
                    event_type="action_applied",
                    trigger=dict(action.reason),
                    after=dict(action.after),
                )
        except Exception as exc:
            self._emit_adaptive_audit(
                iteration=self._actual_iterations,
                stage=str(phase.name),
                controller=action.controller,
                event_type="action_failed",
                trigger={**dict(action.reason), "error": f"{type(exc).__name__}: {exc}"},
                before=dict(action.before),
                after=dict(action.after),
                limits=dict(action.limits),
                checkpoint_id=checkpoint_id,
            )
            if self.rollback_manager is not None:
                self.rollback_manager.clear()
            if self.controller_coordinator is not None:
                self.controller_coordinator.record_outcome(
                    action,
                    iteration=self._actual_iterations,
                    outcome="failed",
                )
            return
        controller.mark_action_applied(self._actual_iterations)
        self.adaptive_control_summary["gradient_balance_actions"] = int(
            self.adaptive_control_summary.get("gradient_balance_actions", 0)
        ) + 1
        if self.controller_coordinator is not None:
            self.controller_coordinator.record_outcome(
                action,
                iteration=self._actual_iterations,
                outcome="applied",
            )
        self._emit_adaptive_audit(
            iteration=self._actual_iterations,
            stage=str(phase.name),
            controller=action.controller,
            event_type="action_applied",
            trigger=dict(action.reason),
            before=dict(action.before),
            after=dict(action.after),
            limits=dict(action.limits),
            checkpoint_id=checkpoint_id,
        )

    def apply_validated_action(
        self,
        action: AdaptiveAction,
        *,
        observable_state: TrainerObservableState | None = None,
    ) -> Any:
        """Enforce the finite action space at the only mutation boundary."""

        if self._last_training_phase is None:
            raise ValueError("No active optimization stage")
        stage_descriptor = self._stage_descriptor(self._last_training_phase)
        if observable_state is None:
            observable_state = TrainerObservableState(
                iteration=self._actual_iterations,
                stage=stage_descriptor.stage_id,
                optimizer_name=str(self._last_training_phase.optimizer_name),
                loss_weights=self.built.loss_function.get_runtime_loss_weights(),
                remaining_iteration_budget=max(
                    0,
                    int(getattr(self._last_training_phase, "iterations", 0))
                    - int(
                        self.completed_phase_iterations.get(
                            self._last_training_phase.name, 0
                        )
                    ),
                ),
                stage_descriptor=stage_descriptor,
                capability_snapshot=(
                    self.registry.capabilities if self.registry is not None else None
                ),
                eligible_component_ids=(
                    self.registry.active_weightable_component_ids(
                        stage_descriptor.stage_id
                    )
                    if self.registry is not None
                    else tuple(self.built.loss_function.get_runtime_loss_weights())
                ),
            )
        if self.action_boundary_validator is not None and self.registry is not None:
            self.action_boundary_validator.validate(
                action=action,
                registry=self.registry,
                stage=stage_descriptor,
                observable_state=observable_state,
            )
        if action.action_type == "replace_sampling_subset":
            if not isinstance(action, ReplaceSamplingSubsetAction):
                raise ValueError("Sampling replacement requires its structured action type")
            return self._execute_sampling_action(action)
        if action.action_type == "advance_curriculum_state":
            if not isinstance(action, AdvanceCurriculumStateAction):
                raise ValueError("Curriculum advance requires its structured action type")
            return self._execute_curriculum_advance_action(
                action, observable_state
            )
        if action.action_type == "terminate_candidate":
            self._curriculum_termination_requested = True
            return None
        if action.action_type != "update_loss_weights":
            raise ValueError(f"No executor is enabled for action: {action.action_type}")
        old_all = self.built.loss_function.get_runtime_loss_weights()
        updated = dict(old_all)
        minimum = float(action.limits["minimum_weight"])
        maximum = float(action.limits["maximum_weight"])
        minimum_ratio = float(action.limits["minimum_update_ratio"])
        maximum_ratio = float(action.limits["maximum_update_ratio"])
        if set(action.after) - set(old_all):
            raise ValueError("An adaptive action cannot introduce loss terms")
        for name, value in action.after.items():
            before = float(old_all[name])
            after = float(value)
            ratio = after / before
            if not math.isfinite(after) or not minimum <= after <= maximum:
                raise ValueError("Adaptive loss weight violates its global bounds")
            if not minimum_ratio - 1e-12 <= ratio <= maximum_ratio + 1e-12:
                raise ValueError("Adaptive loss weight violates its per-action ratio bound")
            updated[name] = after
        self.built.loss_function.set_runtime_loss_weights(updated)
        return None

    def _runtime_state_payload(
        self, *, current_loss: float | None = None
    ) -> dict[str, Any]:
        device = next(self.built.model.parameters()).device
        return {
            "model_state_dict": deepcopy(self.built.model.state_dict()),
            "optimizer_states": [
                deepcopy(phase.optimizer.state_dict())
                for phase in self.built.optimization_phases
            ],
            "scheduler_states": [
                deepcopy(phase.scheduler.state_dict()) if phase.scheduler else None
                for phase in self.built.optimization_phases
            ],
            "active_optimizer_state": (
                deepcopy(self._last_training_phase.optimizer.state_dict())
                if self._last_training_phase is not None
                else None
            ),
            "active_phase_name": (
                str(self._last_training_phase.name)
                if self._last_training_phase is not None
                else None
            ),
            "history": deepcopy(self.history),
            "completed_phase_iterations": dict(self.completed_phase_iterations),
            "actual_iterations": int(self._actual_iterations),
            "adaptive_refinement_events": deepcopy(self.adaptive_refinement_events),
            "optimizer_audit": deepcopy(self.optimizer_audit),
            "component_gradient_audits": deepcopy(self.component_gradient_audits),
            "loss_state": self.built.loss_function.state_dict(),
            "sampler_state": self.built.sampler.state_dict(),
            "gradient_balance_controller_state": (
                self.gradient_balance_controller.state_dict()
                if self.gradient_balance_controller is not None
                else None
            ),
            "lbfgs_stall_controller_state": (
                self.lbfgs_stall_controller.state_dict()
                if self.lbfgs_stall_controller is not None
                else None
            ),
            "rollback_manager_state": (
                self.rollback_manager.state_dict()
                if self.rollback_manager is not None
                else None
            ),
            "adaptive_control_summary": deepcopy(self.adaptive_control_summary),
            "adaptive_compute_budget_ledger_state": (
                self.adaptive_compute_budget_ledger.state_dict()
            ),
            "runtime_scaled_policy_summary": deepcopy(
                self.runtime_scaled_policy_summary
            ),
            "pending_recovery": deepcopy(self._pending_recovery),
            "training_component_registry": (
                self.registry.to_dict() if self.registry is not None else None
            ),
            "controller_coordinator_state": (
                self.controller_coordinator.state_dict()
                if self.controller_coordinator is not None
                else None
            ),
            "validation_probe_manager_state": (
                self.validation_probe_manager.state_dict()
                if self.validation_probe_manager is not None
                else None
            ),
            "best_training_loss_checkpoint_manager_state": (
                self.best_training_loss_checkpoint_manager.state_dict()
                if self.best_training_loss_checkpoint_manager is not None
                else None
            ),
            "best_reference_mse_checkpoint": deepcopy(
                self.best_reference_mse_checkpoint
            ),
            "best_reference_mse_model_state": deepcopy(
                self._best_reference_mse_model_state
            ),
            "selected_final_model_policy": self._selected_final_model_policy,
            "fixed_budget_sampling_controller_state": (
                self.fixed_budget_sampling_controller.state_dict()
                if self.fixed_budget_sampling_controller is not None
                else None
            ),
            "pending_sampling_observation": deepcopy(
                self._pending_sampling_observation
            ),
            "sampling_control_summary": deepcopy(
                self.sampling_control_summary
            ),
            "curriculum_runtime_states": {
                curriculum_id: deepcopy(runtime.snapshot_state())
                for curriculum_id, runtime in self.curriculum_runtimes.items()
            },
            "curriculum_runtime_state_headers": {
                curriculum_id: runtime.get_current_state().to_dict()
                for curriculum_id, runtime in self.curriculum_runtimes.items()
            },
            "curriculum_advance_controller_state": (
                self.curriculum_advance_controller.state_dict()
                if self.curriculum_advance_controller is not None
                else None
            ),
            "pending_curriculum_observation": deepcopy(
                self._pending_curriculum_observation
            ),
            "curriculum_control_summary": deepcopy(
                self.curriculum_control_summary
            ),
            "curriculum_resume_disabled": bool(
                self._curriculum_resume_disabled
            ),
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
            "cuda_random_states": (
                torch.cuda.get_rng_state_all()
                if device.type == "cuda" and torch.cuda.is_available()
                else None
            ),
            "current_loss": current_loss,
        }

    def _restore_runtime_state(
        self, payload: dict[str, Any], *, preserve_consumed_budget: bool = False
    ) -> None:
        consumed = int(self._actual_iterations)
        completed = dict(self.completed_phase_iterations)
        consumed_compute = self.adaptive_compute_budget_ledger.state_dict()
        self.built.model.load_state_dict(payload["model_state_dict"])
        for phase, state in zip(
            self.built.optimization_phases, payload.get("optimizer_states") or []
        ):
            if state is not None:
                self._load_optimizer_state_if_compatible(phase.optimizer, state)
        for phase, state in zip(
            self.built.optimization_phases, payload.get("scheduler_states") or []
        ):
            if phase.scheduler is not None and state is not None:
                phase.scheduler.load_state_dict(state)
        active_state = payload.get("active_optimizer_state")
        active_phase_name = payload.get("active_phase_name")
        self._restore_matching_active_optimizer_state(
            self._last_training_phase,
            active_state,
            active_phase_name,
        )
        self.history = list(payload.get("history") or [])
        self.adaptive_refinement_events = list(
            payload.get("adaptive_refinement_events") or []
        )
        self.optimizer_audit = dict(payload.get("optimizer_audit") or {})
        self.component_gradient_audits = list(
            payload.get("component_gradient_audits") or []
        )
        self.completed_phase_iterations = {
            str(name): int(value)
            for name, value in (
                payload.get("completed_phase_iterations") or {}
            ).items()
        }
        self._actual_iterations = int(payload.get("actual_iterations") or 0)
        self.built.loss_function.load_state_dict(payload.get("loss_state"))
        self.built.sampler.load_state_dict(payload.get("sampler_state"))
        if self.gradient_balance_controller is not None:
            self.gradient_balance_controller.load_state_dict(
                payload.get("gradient_balance_controller_state")
            )
        if self.lbfgs_stall_controller is not None:
            self.lbfgs_stall_controller.load_state_dict(
                payload.get("lbfgs_stall_controller_state")
            )
        if self.rollback_manager is not None:
            self.rollback_manager.load_state_dict(
                payload.get("rollback_manager_state")
            )
        self.adaptive_control_summary = dict(
            payload.get("adaptive_control_summary")
            or self.adaptive_control_summary
        )
        self.adaptive_compute_budget_ledger.load_state_dict(
            payload.get("adaptive_compute_budget_ledger_state"),
            preserve_consumed=preserve_consumed_budget,
        )
        if preserve_consumed_budget:
            for name, value in consumed_compute.items():
                current = getattr(self.adaptive_compute_budget_ledger, name)
                setattr(
                    self.adaptive_compute_budget_ledger,
                    name,
                    type(current)(max(current, value)),
                )
        stored_runtime_policy = payload.get("runtime_scaled_policy_summary")
        if stored_runtime_policy is not None:
            if (
                self.runtime_scaled_policy_summary
                and dict(stored_runtime_policy) != self.runtime_scaled_policy_summary
            ):
                raise ValueError("Checkpoint runtime-scaled policy is incompatible")
            self.runtime_scaled_policy_summary = deepcopy(stored_runtime_policy)
        self._pending_recovery = deepcopy(payload.get("pending_recovery"))
        stored_registry = payload.get("training_component_registry")
        if stored_registry is not None and self.registry is not None:
            current = self.registry.to_dict()
            legacy_registry = int(payload.get("checkpoint_schema_version") or 0) < 5
            if (
                (
                    not legacy_registry
                    and stored_registry.get("schema_version") != current.get("schema_version")
                )
                or {
                    item["component_id"]
                    for item in stored_registry.get("loss_components") or []
                }
                != self.registry.component_ids
                or {
                    item["sampler_id"]
                    for item in stored_registry.get("sampling_components") or []
                }
                != {
                    item.sampler_id for item in self.registry.sampling_components
                }
                or (
                    not legacy_registry
                    and {
                        item["curriculum_id"]
                        for item in stored_registry.get("curricula") or []
                    }
                    != self.registry.curriculum_ids
                )
            ):
                raise ValueError("Checkpoint training-component registry is incompatible")
        if self.controller_coordinator is not None:
            self.controller_coordinator.load_state_dict(
                payload.get("controller_coordinator_state")
            )
        if self.validation_probe_manager is not None:
            self.validation_probe_manager.load_state_dict(
                payload.get("validation_probe_manager_state")
            )
        if self.best_training_loss_checkpoint_manager is not None:
            self.best_training_loss_checkpoint_manager.load_state_dict(
                payload.get("best_training_loss_checkpoint_manager_state")
            )
        stored_best_reference = payload.get("best_reference_mse_checkpoint")
        if stored_best_reference is not None:
            self.best_reference_mse_checkpoint = deepcopy(
                stored_best_reference
            )
            self._best_reference_mse_model_state = deepcopy(
                payload.get("best_reference_mse_model_state")
            )
        self._selected_final_model_policy = payload.get(
            "selected_final_model_policy"
        )
        if self.fixed_budget_sampling_controller is not None:
            self.fixed_budget_sampling_controller.load_state_dict(
                payload.get("fixed_budget_sampling_controller_state")
            )
        self._pending_sampling_observation = deepcopy(
            payload.get("pending_sampling_observation")
        )
        self.sampling_control_summary = dict(
            payload.get("sampling_control_summary")
            or self.sampling_control_summary
        )
        runtime_states = payload.get("curriculum_runtime_states")
        runtime_headers = payload.get("curriculum_runtime_state_headers") or {}
        if runtime_states is None:
            if self.curriculum_advance_controller is not None:
                self._curriculum_resume_disabled = True
                for curriculum_id, runtime in self.curriculum_runtimes.items():
                    runtime.restore_state(
                        self._initial_curriculum_runtime_states[curriculum_id]
                    )
                    descriptor = self.registry.curriculum(curriculum_id)
                    refresh = getattr(
                        self.built.sampler, "refresh_curriculum_samplers", None
                    )
                    if callable(refresh):
                        refresh(descriptor.associated_sampler_ids)
        else:
            if set(runtime_states) != set(self.curriculum_runtimes):
                raise ValueError("Checkpoint Curriculum Runtime IDs are incompatible")
            for curriculum_id, runtime in self.curriculum_runtimes.items():
                runtime.restore_state(runtime_states[curriculum_id])
                restored = runtime.get_current_state()
                expected = dict(runtime_headers.get(curriculum_id) or {})
                if expected and (
                    restored.current_level_index
                    != int(expected.get("current_level_index", -1))
                    or restored.runtime_state_version
                    != int(expected.get("runtime_state_version", -1))
                    or restored.state_digest != str(expected.get("state_digest", ""))
                ):
                    raise ValueError(
                        f"Checkpoint Curriculum Runtime state is inconsistent: {curriculum_id}"
                    )
            self._curriculum_resume_disabled = bool(
                payload.get("curriculum_resume_disabled", False)
            )
        if self.curriculum_advance_controller is not None:
            self.curriculum_advance_controller.load_state_dict(
                payload.get("curriculum_advance_controller_state")
            )
        self._pending_curriculum_observation = deepcopy(
            payload.get("pending_curriculum_observation")
        )
        self.curriculum_control_summary = dict(
            payload.get("curriculum_control_summary")
            or self.curriculum_control_summary
        )
        if payload.get("python_random_state") is not None:
            random.setstate(payload["python_random_state"])
        if payload.get("numpy_random_state") is not None:
            np.random.set_state(payload["numpy_random_state"])
        if payload.get("torch_random_state") is not None:
            torch.set_rng_state(payload["torch_random_state"])
        if payload.get("cuda_random_states") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(payload["cuda_random_states"])
        if preserve_consumed_budget:
            self._actual_iterations = consumed
            self.completed_phase_iterations = completed

    def _save_pre_action_checkpoint(
        self, action: AdaptiveAction, loss: float | None
    ) -> str | None:
        if self.rollback_manager is None:
            return None
        prefix = (
            "pre_sampling_action"
            if action.action_type == "replace_sampling_subset"
            else "pre_curriculum_advance"
            if action.action_type == "advance_curriculum_state"
            else "pre_action"
        )
        checkpoint_id = f"{prefix}_iter_{self._actual_iterations}_{action.controller}"
        self.adaptive_compute_budget_ledger.checkpoint_count += 1
        payload = self._runtime_state_payload(current_loss=loss)
        payload["action_signature"] = self._action_signature(action)
        payload["active_action_type"] = str(action.action_type)
        payload["active_action_metadata"] = {
            "curriculum_id": str(getattr(action, "curriculum_id", "")),
            "from_level_index": int(getattr(action, "from_level_index", -1)),
            "to_level_index": int(getattr(action, "to_level_index", -1)),
            "forced": bool(getattr(action, "forced", False)),
        }
        payload["coordinator_action_signature"] = (
            self.controller_coordinator.action_signature(action)
            if self.controller_coordinator is not None
            else None
        )
        self.rollback_manager.save(
            checkpoint_id,
            payload,
            self.checkpoint_dir,
        )
        return checkpoint_id

    @staticmethod
    def _optimizer_state_is_compatible(
        optimizer: Any,
        state: dict[str, Any] | None,
    ) -> bool:
        """Reject optimizer states that PyTorch would load but cannot execute.

        ``Optimizer.load_state_dict`` validates parameter-group cardinality, but
        it does not validate that the state belongs to the same optimizer type.
        Loading an L-BFGS group into Adam, for example, succeeds and replaces
        Adam's defaults; the next step then fails with ``KeyError: 'betas'``.
        """

        if not isinstance(state, dict):
            return False
        saved_groups = state.get("param_groups")
        current_groups = optimizer.state_dict().get("param_groups")
        if not isinstance(saved_groups, list) or not isinstance(current_groups, list):
            return False
        if len(saved_groups) != len(current_groups):
            return False
        required_group_keys = {"params", *map(str, optimizer.defaults)}
        for saved, current in zip(saved_groups, current_groups):
            if not isinstance(saved, dict) or not isinstance(current, dict):
                return False
            if not required_group_keys.issubset(saved):
                return False
            saved_params = saved.get("params")
            current_params = current.get("params")
            if not isinstance(saved_params, list) or not isinstance(current_params, list):
                return False
            if len(saved_params) != len(current_params):
                return False
        return True

    @staticmethod
    def _load_optimizer_state_if_compatible(
        optimizer: Any,
        state: dict[str, Any] | None,
    ) -> bool:
        """Atomically restore a compatible optimizer state."""

        if not OpenSpecTrainer._optimizer_state_is_compatible(optimizer, state):
            return False
        original_state = deepcopy(optimizer.state_dict())
        try:
            optimizer.load_state_dict(state)
        except (TypeError, ValueError, KeyError, RuntimeError):
            optimizer.load_state_dict(original_state)
            return False
        if not OpenSpecTrainer._optimizer_state_is_compatible(
            optimizer, optimizer.state_dict()
        ):
            optimizer.load_state_dict(original_state)
            return False
        return True

    @staticmethod
    def _restore_matching_active_optimizer_state(
        phase: Any | None,
        state: dict[str, Any] | None,
        saved_phase_name: str | None,
    ) -> bool:
        """Restore the duplicate active state only into the phase that saved it.

        Loading an Adam state into the current L-BFGS phase can succeed in
        PyTorch while silently replacing the L-BFGS parameter-group defaults.
        Its next step then fails with ``KeyError: 'tolerance_grad'``.
        """

        if phase is None or state is None or saved_phase_name is None:
            return False
        if str(getattr(phase, "name", "")) != str(saved_phase_name):
            return False
        return OpenSpecTrainer._load_optimizer_state_if_compatible(
            phase.optimizer, state
        )

    @staticmethod
    def _action_signature(action: AdaptiveAction) -> str:
        return repr(
            (
                str(action.controller),
                str(action.action_type),
                tuple(
                    sorted(
                        (str(name), repr(value))
                        for name, value in action.after.items()
                    )
                ),
            )
        )

    def _rollback_active_action(self, *, phase: Any, reason: str) -> None:
        manager = self.rollback_manager
        if manager is None or manager.active_checkpoint_id is None:
            return
        checkpoint_id = manager.active_checkpoint_id
        payload = manager.load_payload(next(self.built.model.parameters()).device)
        if payload is None:
            return
        active_action_type = str(payload.get("active_action_type") or "")
        active_action_metadata = dict(payload.get("active_action_metadata") or {})
        consumed_sampling = deepcopy(self.sampling_control_summary)
        sampling_updates = (
            int(self.fixed_budget_sampling_controller.update_count)
            if self.fixed_budget_sampling_controller is not None else 0
        )
        consumed_curriculum = deepcopy(self.curriculum_control_summary)
        curriculum_counts = (
            (
                int(self.curriculum_advance_controller.advance_count),
                int(self.curriculum_advance_controller.forced_advance_count),
                int(self.curriculum_advance_controller.rollback_count),
            )
            if self.curriculum_advance_controller is not None
            else (0, 0, 0)
        )
        best_state = (
            self.best_training_loss_checkpoint_manager.state_dict()
            if self.best_training_loss_checkpoint_manager is not None else None
        )
        self._restore_runtime_state(payload, preserve_consumed_budget=True)
        for name in (
            "sampling_diagnostic_evaluations", "sampling_points_diagnosed",
            "candidate_points_evaluated", "sampling_updates_executed",
            "sampling_updates_rejected", "sampling_points_replaced",
        ):
            self.sampling_control_summary[name] = max(
                int(self.sampling_control_summary.get(name, 0)),
                int(consumed_sampling.get(name, 0)),
            )
        self.sampling_control_summary["sampling_compute_time"] = max(
            float(self.sampling_control_summary.get("sampling_compute_time", 0.0)),
            float(consumed_sampling.get("sampling_compute_time", 0.0)),
        )
        if best_state is not None and self.best_training_loss_checkpoint_manager is not None:
            self.best_training_loss_checkpoint_manager.load_state_dict(best_state)
        signature = str(payload.get("action_signature") or checkpoint_id)
        manager.mark_rolled_back(signature)
        if self.controller_coordinator is not None:
            coordinator_signature = payload.get("coordinator_action_signature")
            if coordinator_signature:
                self.controller_coordinator.failed_signatures[
                    str(coordinator_signature)
                ] = int(self._actual_iterations)
        if self.gradient_balance_controller is not None:
            self.gradient_balance_controller.mark_action_applied(
                self._actual_iterations
                + int(
                    self.gradient_balance_controller.config[
                        "cooldown_iterations"
                    ]
                )
            )
        if self.fixed_budget_sampling_controller is not None:
            self.fixed_budget_sampling_controller.update_count = max(
                int(self.fixed_budget_sampling_controller.update_count),
                sampling_updates,
            )
            self.fixed_budget_sampling_controller.mark_action_rolled_back(
                self._actual_iterations
            )
            if active_action_type == "replace_sampling_subset":
                self.sampling_control_summary["sampling_updates_rolled_back"] = int(
                    consumed_sampling.get("sampling_updates_rolled_back", 0)
                ) + 1
        if (
            active_action_type == "advance_curriculum_state"
            and self.curriculum_advance_controller is not None
        ):
            controller = self.curriculum_advance_controller
            controller.advance_count = max(
                int(controller.advance_count), curriculum_counts[0]
            )
            controller.forced_advance_count = max(
                int(controller.forced_advance_count), curriculum_counts[1]
            )
            controller.rollback_count = max(
                int(controller.rollback_count), curriculum_counts[2]
            )
            curriculum_id = str(active_action_metadata.get("curriculum_id") or "")
            controller.mark_action_rolled_back(
                curriculum_id, self._actual_iterations
            )
            runtime = self.curriculum_runtimes.get(curriculum_id)
            if runtime is not None:
                runtime.record_rollback(self._actual_iterations)
            integer_fields = (
                "curriculum_diagnostic_checks", "curriculum_advances_proposed",
                "curriculum_advances_executed", "curriculum_advances_forced",
                "curriculum_advances_rejected",
            )
            for name in integer_fields:
                self.curriculum_control_summary[name] = max(
                    int(self.curriculum_control_summary.get(name, 0)),
                    int(consumed_curriculum.get(name, 0)),
                )
            for name in (
                "curriculum_runtime_refresh_cost", "curriculum_validation_cost",
            ):
                self.curriculum_control_summary[name] = max(
                    float(self.curriculum_control_summary.get(name, 0.0)),
                    float(consumed_curriculum.get(name, 0.0)),
                )
            self.curriculum_control_summary["curriculum_advances_rolled_back"] = int(
                consumed_curriculum.get("curriculum_advances_rolled_back", 0)
            ) + 1
            self.curriculum_control_summary["iterations_per_level"] = deepcopy(
                consumed_curriculum.get("iterations_per_level") or {}
            )
        self._pending_sampling_observation = None
        self._pending_curriculum_observation = None
        self.adaptive_control_summary["rollbacks"] = int(
            self.adaptive_control_summary.get("rollbacks", 0)
        ) + 1
        self._emit_adaptive_audit(
            iteration=self._actual_iterations,
            stage=str(phase.name),
            controller="checkpoint_rollback",
            event_type="rollback",
            trigger={"reason": reason},
            checkpoint_id=checkpoint_id,
            rollback_status="completed",
        )

    def _parameter_update_norm(
        self, before: list[torch.Tensor]
    ) -> float | None:
        squared = 0.0
        found = False
        for old, parameter in zip(before, self.built.model.parameters()):
            difference = parameter.detach() - old.to(parameter.device)
            value = float(torch.linalg.vector_norm(difference).cpu())
            squared += value * value
            found = True
        return squared**0.5 if found else None

    def _run_recovery_adam(
        self,
        budget: int,
        clipping_norm: float | None,
        clipping_norm_type: float,
        adaptive_interval: int,
    ) -> tuple[float | None, int, str | None]:
        controller = self.lbfgs_stall_controller
        scale = float(controller.config.get("adam_learning_rate_scale", 0.1))
        source_lr = 1e-3
        for configured in reversed(self.built.optimization_phases):
            if configured.optimizer_name in {"adam", "adamw", "radam", "nadam"}:
                source_lr = float(configured.optimizer.param_groups[0].get("lr", source_lr))
                break
        optimizer = torch.optim.Adam(
            self.built.model.parameters(), lr=max(1e-12, source_lr * scale)
        )
        phase = OptimizationPhase(
            name="adaptive_recovery_adam",
            optimizer_name="adam",
            iterations=int(budget),
            optimizer=optimizer,
            scheduler=None,
            parameters={"learning_rate": max(1e-12, source_lr * scale)},
        )
        self._last_training_phase = phase
        self._emit_adaptive_audit(
            iteration=self._actual_iterations,
            stage=phase.name,
            controller="lbfgs_stall",
            event_type="recovery_adam_started",
            trigger={"source_learning_rate": source_lr, "scale": scale},
            after={"iterations": int(budget), "learning_rate": source_lr * scale},
        )
        latest: float | None = None
        executed = 0
        failure: str | None = None
        for _ in range(int(budget)):
            stage = self._configure_training_stage(
                phase=phase, non_lbfgs_iteration=self._actual_iterations + 1
            )
            batch = self.built.sampler.sample()
            self._last_training_batch = batch
            optimizer.zero_grad(set_to_none=True)
            loss, components = self.built.loss_function(self.built.model, batch)
            if not torch.isfinite(loss):
                failure = "non_finite_recovery_loss"
                if self.rollback_manager is not None:
                    self._rollback_active_action(phase=phase, reason=failure)
                self._release_iteration_graphs()
                break
            projected = self._actual_iterations + 1
            due = bool(
                self.gradient_balance_controller is not None
                and projected
                % int(self.gradient_balance_controller.config["diagnostic_interval"])
                == 0
            )
            diagnostic = (
                self._collect_component_gradient_diagnostic(
                    total_loss=loss,
                    iteration=projected,
                    phase=phase,
                    checkpoint_kind="adaptive_control",
                )
                if due
                else None
            )
            loss.backward()
            if clipping_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.built.model.parameters(),
                    clipping_norm,
                    norm_type=clipping_norm_type,
                )
            if not self._gradients_finite():
                failure = "non_finite_recovery_gradient"
                if self.rollback_manager is not None:
                    self._rollback_active_action(phase=phase, reason=failure)
                self._release_iteration_graphs()
                break
            gradient_norm = self._gradient_norm()
            optimizer.step()
            if not self._parameters_finite():
                failure = "non_finite_recovery_parameter"
                if self.rollback_manager is not None:
                    self._rollback_active_action(phase=phase, reason=failure)
                self._release_iteration_graphs()
                break
            latest = float(loss.detach().cpu())
            executed += 1
            self._actual_iterations += 1
            # Recovery steps consume the same global optimizer-step budget as
            # the original phases and must satisfy the same fixed MSE probes.
            self._record_reference_mse_checkpoint(self._actual_iterations)
            self._observe_gradient_balance(
                phase=phase,
                loss=latest,
                components=components,
                gradient_norm=gradient_norm,
                diagnostic=diagnostic,
            )
            if self.rollback_manager is not None and self.rollback_manager.observe_loss(latest):
                rollback_baseline = getattr(
                    self.rollback_manager, "baseline_loss", None
                )
                self._rollback_active_action(
                    phase=phase, reason="recovery_loss_degradation"
                )
                if rollback_baseline is not None and math.isfinite(
                    float(rollback_baseline)
                ):
                    latest = float(rollback_baseline)
                # A rollback is the successful safety response to a harmful
                # adaptive action, not a numerical training failure.  Continue
                # the remaining recovery budget from the restored checkpoint
                # with fresh, more conservative Adam state.
                optimizer.state.clear()
                for group in optimizer.param_groups:
                    group["lr"] = max(1e-12, float(group["lr"]) * 0.5)
                self._emit_adaptive_audit(
                    iteration=self._actual_iterations,
                    stage=phase.name,
                    controller="checkpoint_rollback",
                    event_type="recovery_rollback_continued",
                    trigger={"reason": "recovery_loss_degradation"},
                    after={
                        "remaining_iterations": max(0, int(budget) - executed),
                        "learning_rates": [
                            float(group["lr"]) for group in optimizer.param_groups
                        ],
                    },
                    rollback_status="completed",
                )
                self._release_iteration_graphs()
                continue
            self._maybe_run_physics_validation(
                phase=phase,
                force=False,
                training_loss=latest,
            )
            adaptive_event = self._actual_iterations % adaptive_interval == 0
            if adaptive_event:
                self.built.sampler.adaptive_update(self.built.model)
            if self._should_record_history(
                self._actual_iterations, phase.name, event=adaptive_event
            ):
                self.history.append(
                    self._history_entry(
                        iteration=self._actual_iterations,
                        phase=phase,
                        total_loss=latest,
                        components=components,
                        batch=batch,
                        gradient_norm=gradient_norm,
                        gradient_statistics=diagnostic,
                        adaptive_refinement_event=adaptive_event,
                        training_stage=stage,
                    )
                )
            self._release_iteration_graphs()
        self._emit_adaptive_audit(
            iteration=self._actual_iterations,
            stage=phase.name,
            controller="lbfgs_stall",
            event_type="recovery_adam_finished",
            trigger={"failure": failure},
            after={"executed_iterations": executed, "final_loss": latest},
        )
        return latest, executed, failure

    def _stage_descriptor(self, phase: Any) -> Any:
        if self.registry is not None:
            try:
                return self.registry.stage(str(phase.name))
            except KeyError:
                pass
        second_order = isinstance(
            getattr(phase, "optimizer", None), torch.optim.LBFGS
        ) or str(getattr(phase, "optimizer_name", "")).casefold() in {
            "lbfgs", "l_bfgs"
        }
        return SimpleNamespace(
            stage_id=str(phase.name),
            optimizer_family=(
                "second_order_line_search"
                if second_order
                else "first_order_gradient"
            ),
            first_order=not second_order,
            second_order=second_order,
            dynamic_objective_allowed=not second_order,
            recoverable=second_order,
        )

    def _initialize_physics_validation(self) -> None:
        manager = self.validation_probe_manager
        if manager is None or self.registry is None:
            return
        if manager.probe_set is None:
            manager.build_probes(self.registry, self.built.sampler)
        if manager.probe_set is None or manager.initial_scales:
            return
        result = manager.evaluate(
            self.built.model,
            self.registry,
            self.built.loss_function,
        )
        if result is not None:
            manager.record(0, "training_start", result)

    def _maybe_run_physics_validation(
        self,
        *,
        phase: Any | None,
        force: bool,
        training_loss: float | None = None,
    ) -> None:
        manager = self.validation_probe_manager
        checkpoint_manager = self.best_training_loss_checkpoint_manager
        stage_id = str(getattr(phase, "name", "training_end"))
        training_loss_updated = bool(
            checkpoint_manager is not None
            and checkpoint_manager.final_model_policy == "best_train_loss"
            and training_loss is not None
            and (
                force
                or self._actual_iterations % checkpoint_manager.evaluation_interval == 0
            )
            and checkpoint_manager.consider(
                model=self.built.model,
                training_loss=training_loss,
                iteration=self._actual_iterations,
                stage_id=stage_id,
                algorithm_spec_id=self._algorithm_spec_id,
                directory=self.checkpoint_dir,
            )
        )
        if manager is None or self.registry is None or manager.probe_set is None:
            return
        if not force:
            if not manager.is_due(self._actual_iterations):
                return
        if (
            manager.history
            and int(manager.history[-1].get("iteration", -1))
            == int(self._actual_iterations)
        ):
            return
        result = manager.evaluate(
            self.built.model,
            self.registry,
            self.built.loss_function,
        )
        if result is None:
            return
        manager.record(self._actual_iterations, stage_id, result)
        self._assess_pending_sampling_action(result, phase)
        curriculum_pending_before_validation = (
            self._pending_curriculum_observation is not None
        )
        curriculum_started = time.perf_counter()
        curriculum_accepted = self._assess_pending_curriculum_action(
            result, phase
        )
        if curriculum_pending_before_validation:
            self.curriculum_control_summary["curriculum_validation_cost"] += (
                time.perf_counter() - curriculum_started
            )
        self._emit_adaptive_audit(
            iteration=self._actual_iterations,
            stage=stage_id,
            controller="physics_validation",
            event_type="physics_validation",
            trigger=result.to_dict(),
            after={"best_checkpoint_updated": training_loss_updated},
        )

    def _sampling_observable_state(
        self,
        *,
        phase: Any,
        diagnostics: dict[str, Any] | None = None,
        component_states: dict[str, Any] | None = None,
    ) -> TrainerObservableState:
        controller = self.fixed_budget_sampling_controller
        maximum = int(
            getattr(controller, "config", {}).get(
                "maximum_candidate_points_evaluated", 0
            )
        )
        consumed = int(
            self.sampling_control_summary.get("candidate_points_evaluated", 0)
        )
        coordinator_active = bool(
            self.controller_coordinator is not None
            and self.controller_coordinator.observation_window_active(
                self._actual_iterations
            )
        )
        return TrainerObservableState(
            iteration=int(self._actual_iterations),
            stage=str(phase.name),
            optimizer_name=str(getattr(phase, "optimizer_name", "")),
            stage_descriptor=self._stage_descriptor(phase),
            capability_snapshot=(
                self.registry.capabilities if self.registry is not None else None
            ),
            validation_state=getattr(
                self.validation_probe_manager, "last_result", None
            ),
            sampling_component_states=dict(component_states or {}),
            sampling_diagnostics=dict(diagnostics or {}),
            sampling_budget_state={
                "maximum_candidate_points": maximum,
                "consumed_candidate_points": consumed,
                "remaining_candidate_points": max(0, maximum - consumed),
            },
            pending_rollback=bool(
                self.rollback_manager is not None
                and self.rollback_manager.active_checkpoint_id is not None
            ),
            coordinator_observation_active=coordinator_active,
        )

    def _observe_sampling_control(self, *, phase: Any) -> None:
        controller = self.fixed_budget_sampling_controller
        sampler = getattr(self.built, "sampler", None)
        if controller is None or self.registry is None or sampler is None:
            return
        preliminary = self._sampling_observable_state(phase=phase)
        due, reason = controller.diagnostic_due(preliminary)
        if not due:
            if reason not in {"diagnostic_interval_not_reached", "minimum_iterations_not_reached"}:
                self._emit_adaptive_audit(
                    iteration=self._actual_iterations,
                    stage=str(phase.name),
                    controller=controller.name,
                    event_type="sampling_diagnostic_skipped",
                    trigger={"reason": reason},
                )
            return
        started = time.perf_counter()
        diagnostics: dict[str, Any] = {}
        component_states: dict[str, Any] = {}
        self._sampling_score_cache = {}
        for descriptor in self.registry.sampling_components:
            if not descriptor.replacement_supported:
                continue
            try:
                points = sampler.get_training_points(descriptor.sampler_id)
                context = self._sampling_observable_state(phase=phase)
                scored = sampler.score_points(
                    self.built.model,
                    descriptor.associated_loss_component_ids,
                    points,
                    context,
                )
            except Exception as exc:
                self.sampling_control_summary["sampling_updates_rejected"] += 1
                self._emit_adaptive_audit(
                    iteration=self._actual_iterations,
                    stage=str(phase.name),
                    controller=controller.name,
                    event_type="sampling_diagnostic_failed",
                    trigger={
                        "sampler_id": descriptor.sampler_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
                continue
            component_states[descriptor.sampler_id] = {
                "point_count": points.point_count,
                "sampler_state_version": points.state_version,
            }
            diagnostics[descriptor.sampler_id] = {
                "point_count": points.point_count,
                "sampler_state_version": points.state_version,
                "statistics": dict(scored.statistics),
                "scoring_component_ids": tuple(scored.scoring_component_ids),
            }
            self._sampling_score_cache[descriptor.sampler_id] = (points, scored)
            self.sampling_control_summary["sampling_diagnostic_evaluations"] += 1
            self.sampling_control_summary["sampling_points_diagnosed"] += points.point_count
        self.sampling_control_summary["sampling_compute_time"] += (
            time.perf_counter() - started
        )
        state = self._sampling_observable_state(
            phase=phase,
            diagnostics=diagnostics,
            component_states=component_states,
        )
        observation = (
            self.controller_coordinator.observe_controller(controller, state)
            if self.controller_coordinator is not None
            else controller.observe(state)
        )
        self._emit_adaptive_audit(
            iteration=self._actual_iterations,
            stage=str(phase.name),
            controller=controller.name,
            event_type="sampling_diagnostic",
            trigger=observation,
        )
        action = controller.propose_action()
        if action is None:
            return
        if self.controller_coordinator is not None:
            selected, rejected = self.controller_coordinator.select_action(
                (action,), iteration=self._actual_iterations
            )
            if selected is None:
                self.sampling_control_summary["sampling_updates_rejected"] += 1
                self._emit_adaptive_audit(
                    iteration=self._actual_iterations,
                    stage=str(phase.name),
                    controller="controller_coordinator",
                    event_type="sampling_action_rejected",
                    trigger={"rejections": rejected},
                )
                return
            action = selected
        checkpoint_id = self._save_pre_action_checkpoint(action, None)
        try:
            mutation = self.apply_validated_action(action, observable_state=state)
        except Exception as exc:
            self.sampling_control_summary["sampling_updates_rejected"] += 1
            if self.controller_coordinator is not None:
                self.controller_coordinator.record_outcome(
                    action, iteration=self._actual_iterations, outcome="failed"
                )
            self._emit_adaptive_audit(
                iteration=self._actual_iterations,
                stage=str(phase.name),
                controller=controller.name,
                event_type="sampling_action_failed",
                trigger={"error": f"{type(exc).__name__}: {exc}"},
                before=dict(action.before),
                after=dict(action.after),
                checkpoint_id=checkpoint_id,
            )
            if (
                self.rollback_manager is not None
                and self.rollback_manager.active_checkpoint_id is not None
            ):
                self._rollback_active_action(
                    phase=phase, reason="sampling_action_execution_failed"
                )
            return
        controller.mark_action_applied(self._actual_iterations)
        if self.controller_coordinator is not None:
            self.controller_coordinator.record_outcome(
                action, iteration=self._actual_iterations, outcome="applied"
            )
            self.controller_coordinator.extend_observation_window(
                self._actual_iterations
                + int(controller.config["observation_window_iterations"])
            )
        self.sampling_control_summary["sampling_updates_executed"] += 1
        self.sampling_control_summary["sampling_points_replaced"] += int(
            mutation.points_replaced
        )
        baseline = getattr(self.validation_probe_manager, "last_result", None)
        self._pending_sampling_observation = {
            "action_signature": (
                self.controller_coordinator.action_signature(action)
                if self.controller_coordinator is not None
                else self._action_signature(action)
            ),
            "sampler_id": action.sampler_id,
            "associated_component_ids": list(action.scoring_component_ids),
            "start_iteration": int(self._actual_iterations),
            "validation_checks": 0,
            "degradation_checks": 0,
            "baseline": baseline.to_dict() if baseline is not None else None,
            "checkpoint_id": (
                self.rollback_manager.active_checkpoint_id
                if self.rollback_manager is not None else None
            ),
        }
        self._emit_adaptive_audit(
            iteration=self._actual_iterations,
            stage=str(phase.name),
            controller=controller.name,
            event_type="action_applied",
            trigger=dict(action.reason),
            before=dict(action.before),
            after={
                **dict(action.after),
                "point_count": mutation.point_count_after,
                "points_replaced": mutation.points_replaced,
                "sampler_state_version": mutation.state_version_after,
            },
            limits=dict(action.limits),
            checkpoint_id=checkpoint_id,
        )

    def _execute_sampling_action(
        self, action: ReplaceSamplingSubsetAction
    ) -> Any:
        compute_started = time.perf_counter()
        sampler = self.built.sampler
        cached = self._sampling_score_cache.get(action.sampler_id)
        if cached is None:
            raise RuntimeError("Sampling action score cache is unavailable or stale")
        current, current_scores = cached
        if int(current.state_version) != int(action.sampler_state_version):
            raise RuntimeError("Sampler state changed before action execution")
        count = current.point_count
        replacement_count = max(1, int(count * action.replacement_fraction))
        hard_count = min(
            count - replacement_count,
            int(count * action.hard_point_retention_fraction),
        )
        scores = current_scores.scores.detach().cpu()
        ranked_high = torch.argsort(scores, descending=True).tolist()
        metadata = list(current.point_metadata)
        maximum_age = int(action.limits.get("maximum_point_age", 20))
        maximum_retention = int(
            action.limits.get("maximum_consecutive_retention", 5)
        )
        minimum_age = int(action.limits.get("minimum_point_age", 0))
        hard = [
            index for index in ranked_high
            if int(metadata[index].get("age", 0)) <= maximum_age
            and int(metadata[index].get("consecutive_retention_count", 0))
            <= maximum_retention
        ][:hard_count]
        protected = set(hard)
        ranked_low = torch.argsort(scores, descending=False).tolist()
        forced_release = [
            index for index in range(count)
            if int(metadata[index].get("age", 0)) > maximum_age
            or int(metadata[index].get("consecutive_retention_count", 0))
            > maximum_retention
        ]
        forced_release.sort(
            key=lambda index: (
                -int(metadata[index].get("age", 0)),
                -int(metadata[index].get("consecutive_retention_count", 0)),
                index,
            )
        )
        replace_indices = forced_release[:replacement_count]
        replace_indices.extend(
            index for index in ranked_low
            if index not in protected
            and index not in replace_indices
            and int(metadata[index].get("age", 0)) >= minimum_age
        )
        replace_indices = replace_indices[:replacement_count]
        if len(replace_indices) < replacement_count:
            replace_indices.extend(
                index for index in ranked_low
                if index not in protected and index not in replace_indices
            )
        if len(replace_indices) < replacement_count:
            replace_indices.extend(
                index for index in ranked_low if index not in replace_indices
            )
        replace_indices = replace_indices[:replacement_count]
        retained_indices = [
            index for index in range(count) if index not in set(replace_indices)
        ]
        candidates = sampler.generate_candidate_points(
            action.sampler_id, action.candidate_count
        )
        self.sampling_control_summary["candidate_points_evaluated"] += int(
            action.candidate_count
        )
        candidate_scores = sampler.score_points(
            self.built.model,
            action.scoring_component_ids,
            candidates,
            None,
        )
        exploration_count = min(
            replacement_count,
            math.ceil(replacement_count * action.exploration_fraction),
        )
        biased_count = replacement_count - exploration_count
        candidate_order = torch.argsort(
            candidate_scores.scores, descending=True
        ).tolist()
        selected = candidate_order[:biased_count]
        remaining = [index for index in range(candidates.point_count) if index not in set(selected)]
        if exploration_count:
            permutation = torch.randperm(len(remaining)).tolist()
            selected.extend(remaining[index] for index in permutation[:exploration_count])
        from .sampling_runtime import SamplingBatch

        replacement = SamplingBatch(
            sampler_id=action.sampler_id,
            points=candidates.points[selected].detach().clone(),
            point_metadata=tuple(candidates.point_metadata[index] for index in selected),
            state_version=candidates.state_version,
        )
        mutation = sampler.replace_subset(
            action.sampler_id,
            retained_indices,
            replacement,
            hard_retained_indices=hard,
            exploration_count=exploration_count,
            iteration=self._actual_iterations,
            current_scores=scores.tolist(),
        )
        self.sampling_control_summary["sampling_compute_time"] += (
            time.perf_counter() - compute_started
        )
        return mutation

    def _assess_pending_sampling_action(self, result: Any, phase: Any | None) -> None:
        pending = self._pending_sampling_observation
        controller = self.fixed_budget_sampling_controller
        if pending is None or controller is None:
            return
        baseline = pending.get("baseline")
        if not baseline:
            return
        pending["validation_checks"] = int(pending.get("validation_checks", 0)) + 1
        aggregate_bad = float(result.aggregate_score) > float(
            baseline["aggregate_score"]
        ) * float(controller.config["validation_degradation_ratio"])
        baseline_components = dict(baseline.get("normalized_component_scores") or {})
        component_bad = any(
            component_id in result.normalized_component_scores
            and component_id in baseline_components
            and float(result.normalized_component_scores[component_id])
            > float(baseline_components[component_id])
            * float(controller.config["associated_component_degradation_ratio"])
            for component_id in pending.get("associated_component_ids") or []
        )
        associated_ids = set(pending.get("associated_component_ids") or [])
        associated_categories = {
            descriptor.category
            for descriptor in self.registry.loss_components
            if descriptor.component_id in associated_ids
        } if self.registry is not None else set()
        baseline_categories = dict(baseline.get("category_scores") or {})
        category_bad = any(
            category in result.category_scores
            and category in baseline_categories
            and float(result.category_scores[category])
            > float(baseline_categories[category])
            * float(controller.config["associated_component_degradation_ratio"])
            for category in associated_categories
        )
        critical_ids = {
            descriptor.component_id
            for descriptor in self.registry.loss_components
            if bool(descriptor.metadata.get("critical_for_valid_solution", False))
        } if self.registry is not None else set()
        critical_bad = any(
            component_id in result.normalized_component_scores
            and component_id in baseline_components
            and float(result.normalized_component_scores[component_id])
            > float(baseline_components[component_id])
            * float(controller.config["associated_component_degradation_ratio"])
            for component_id in critical_ids
        )
        pending["degradation_checks"] = (
            int(pending.get("degradation_checks", 0)) + 1
            if aggregate_bad or component_bad or category_bad or critical_bad else 0
        )
        minimum_checks = int(controller.config["minimum_validation_stability_checks"])
        if int(pending["degradation_checks"]) >= minimum_checks:
            self._rollback_active_action(
                phase=phase or self._last_training_phase,
                reason="sampling_physics_validation_degradation",
            )
            return
        observation_complete = (
            self._actual_iterations - int(pending["start_iteration"])
            >= int(controller.config["observation_window_iterations"])
            and int(pending["validation_checks"]) >= minimum_checks
        )
        if observation_complete:
            if self.rollback_manager is not None:
                self.rollback_manager.clear()
            self._pending_sampling_observation = None

    def _note_curriculum_training_iteration(self) -> None:
        if self._curriculum_resume_disabled:
            return
        for runtime in self.curriculum_runtimes.values():
            runtime.note_training_iteration(self._actual_iterations)

    def _curriculum_observable_state(self, *, phase: Any) -> TrainerObservableState:
        snapshots: dict[str, Any] = {}
        diagnostics: dict[str, Any] = {}
        for curriculum_id, runtime in sorted(self.curriculum_runtimes.items()):
            state = runtime.get_current_state()
            descriptor = self.registry.curriculum(curriculum_id)
            snapshots[curriculum_id] = state.to_dict()
            diagnostics[curriculum_id] = {
                "level_ids": tuple(descriptor.level_ids),
                "axis_role": descriptor.axis_role,
                "associated_component_ids": tuple(
                    descriptor.associated_loss_component_ids
                ),
                "associated_sampler_ids": tuple(descriptor.associated_sampler_ids),
                "critical_component_ids": tuple(descriptor.critical_component_ids),
                "force_advance_supported": bool(
                    descriptor.force_advance_supported
                ),
                "probe_comparability_mode": descriptor.metadata.get(
                    "probe_comparability_mode", "global_fixed_probe"
                ),
            }
        total_budget = sum(
            int(item.iterations) for item in self.built.optimization_phases
        )
        remaining = max(0, total_budget - int(self._actual_iterations))
        coordinator_active = bool(
            self.controller_coordinator is not None
            and self.controller_coordinator.observation_window_active(
                self._actual_iterations
            )
        )
        return TrainerObservableState(
            iteration=int(self._actual_iterations),
            stage=str(phase.name),
            optimizer_name=str(getattr(phase, "optimizer_name", "")),
            remaining_iteration_budget=remaining,
            stage_descriptor=self._stage_descriptor(phase),
            capability_snapshot=(
                self.registry.capabilities if self.registry is not None else None
            ),
            validation_state=getattr(
                self.validation_probe_manager, "last_result", None
            ),
            curriculum_states=snapshots,
            curriculum_diagnostics=diagnostics,
            curriculum_budget_state={
                "total_iteration_budget": total_budget,
                "consumed_iteration_budget": int(self._actual_iterations),
                "remaining_iteration_budget": remaining,
            },
            pending_rollback=bool(
                self.rollback_manager is not None
                and self.rollback_manager.active_checkpoint_id is not None
            ),
            coordinator_observation_active=coordinator_active,
        )

    def _observe_curriculum_control(self, *, phase: Any) -> None:
        controller = self.curriculum_advance_controller
        if (
            controller is None
            or self.registry is None
            or not self.curriculum_runtimes
            or self._curriculum_resume_disabled
        ):
            return
        preliminary = self._curriculum_observable_state(phase=phase)
        due, reason = controller.diagnostic_due(preliminary)
        if not due:
            if reason not in {"validation_interval_not_reached"}:
                self._emit_adaptive_audit(
                    iteration=self._actual_iterations,
                    stage=str(phase.name),
                    controller=controller.name,
                    event_type="curriculum_diagnostic_skipped",
                    trigger={"reason": reason},
                )
            return
        manager = self.validation_probe_manager
        if (
            manager is not None
            and (
                not manager.history
                or int(manager.history[-1].get("iteration", -1))
                != self._actual_iterations
            )
        ):
            started = time.perf_counter()
            self._maybe_run_physics_validation(phase=phase, force=True)
            self.curriculum_control_summary["curriculum_validation_cost"] += (
                time.perf_counter() - started
            )
        state = self._curriculum_observable_state(phase=phase)
        self.curriculum_control_summary["curriculum_diagnostic_checks"] += 1
        observation = (
            self.controller_coordinator.observe_controller(controller, state)
            if self.controller_coordinator is not None
            else controller.observe(state)
        )
        self._emit_adaptive_audit(
            iteration=self._actual_iterations,
            stage=str(phase.name),
            controller=controller.name,
            event_type="curriculum_diagnostic",
            trigger=dict(observation),
        )
        action = controller.propose_action()
        if action is None:
            return
        self.curriculum_control_summary["curriculum_advances_proposed"] += 1
        selected = action
        rejected: dict[str, str] = {}
        if self.controller_coordinator is not None:
            selected, rejected = self.controller_coordinator.select_action(
                (action,), iteration=self._actual_iterations
            )
        if selected is None:
            self.curriculum_control_summary["curriculum_advances_rejected"] += 1
            self._emit_adaptive_audit(
                iteration=self._actual_iterations,
                stage=str(phase.name),
                controller=controller.name,
                event_type="curriculum_action_rejected",
                trigger={
                    "reason": rejected.get(action.action_id, "coordinator_rejected")
                },
                before=dict(action.before),
                after=dict(action.after),
            )
            return
        try:
            self.apply_validated_action(action, observable_state=state)
        except Exception as exc:
            self.curriculum_control_summary["curriculum_advances_rejected"] += 1
            self._emit_adaptive_audit(
                iteration=self._actual_iterations,
                stage=str(phase.name),
                controller=controller.name,
                event_type="curriculum_action_failed",
                trigger={"error": f"{type(exc).__name__}: {exc}"},
                before=dict(action.before),
                after=dict(action.after),
            )
            if self.controller_coordinator is not None:
                self.controller_coordinator.record_outcome(
                    action, iteration=self._actual_iterations, outcome="failed"
                )

    def _execute_curriculum_advance_action(
        self,
        action: AdvanceCurriculumStateAction,
        observable_state: TrainerObservableState,
    ) -> Any:
        runtime = self.curriculum_runtimes.get(action.curriculum_id)
        if runtime is None:
            raise ValueError("Curriculum Runtime is not registered")
        before_state = runtime.get_current_state()
        if (
            before_state.runtime_state_version != action.runtime_state_version
            or before_state.state_digest != action.runtime_state_digest
        ):
            raise ValueError("Curriculum Runtime changed after boundary validation")
        stage = self._stage_descriptor(self._last_training_phase)
        if not runtime.can_advance(action.to_level_index, stage):
            raise ValueError("Curriculum Runtime rejected the adjacent target level")
        baseline_result = getattr(
            self.validation_probe_manager, "last_result", None
        )
        if baseline_result is None:
            raise ValueError("Curriculum advance requires a fixed physics baseline")
        checkpoint_id = self._save_pre_action_checkpoint(
            action,
            float(self.history[-1]["total_loss"])
            if self.history and self.history[-1].get("total_loss") is not None
            else None,
        )
        self._emit_adaptive_audit(
            iteration=self._actual_iterations,
            stage=str(self._last_training_phase.name),
            controller=action.controller,
            event_type="curriculum_action_started",
            trigger=dict(action.trigger_statistics),
            before=dict(action.before),
            after=dict(action.after),
            checkpoint_id=checkpoint_id,
        )
        registry_components = tuple(sorted(self.registry.component_ids))
        registry_samplers = tuple(sorted(
            item.sampler_id for item in self.registry.sampling_components
        ))
        model_structure = tuple(
            (name, tuple(value.shape), str(value.dtype))
            for name, value in self.built.model.state_dict().items()
        )
        spec_id = self._algorithm_spec_id
        unaffected_sampler_ids = set(registry_samplers) - set(
            action.associated_sampler_ids
        )
        sampler_versions_before = {
            sampler_id: int(value)
            for sampler_id, value in dict(
                getattr(self.built.sampler, "_sampling_state_versions", {}) or {}
            ).items()
            if sampler_id in unaffected_sampler_ids
        }
        descriptor = self.registry.curriculum(action.curriculum_id)
        try:
            preview = runtime.preview_advance(action.to_level_index)
            if (
                preview.curriculum_id != action.curriculum_id
                or preview.from_level_index != action.from_level_index
                or preview.to_level_index != action.to_level_index
                or tuple(preview.structural_component_ids) != registry_components
                or tuple(preview.structural_sampler_ids) != registry_samplers
                or tuple(preview.associated_loss_component_ids)
                != tuple(descriptor.associated_loss_component_ids)
                or tuple(preview.associated_sampler_ids)
                != tuple(descriptor.associated_sampler_ids)
                or not preview.fixed_point_budget
            ):
                raise ValueError("Curriculum preview violates structural invariants")
            result = runtime.apply_advance(
                action.to_level_index,
                iteration=self._actual_iterations,
                forced=action.forced,
            )
            after_state = runtime.get_current_state()
            if (
                result.curriculum_id != action.curriculum_id
                or result.from_level_index != action.from_level_index
                or result.to_level_index != action.to_level_index
                or after_state.current_level_index != action.to_level_index
                or after_state.current_level_id != action.to_level_id
                or result.runtime_state_version != after_state.runtime_state_version
                or result.state_digest != after_state.state_digest
                or not result.fixed_point_budget_preserved
                or not runtime.validate_runtime_state()
            ):
                raise ValueError("Curriculum Runtime returned an inconsistent transition")
            if (
                tuple(sorted(self.registry.component_ids)) != registry_components
                or tuple(sorted(
                    item.sampler_id for item in self.registry.sampling_components
                )) != registry_samplers
                or tuple(
                    (name, tuple(value.shape), str(value.dtype))
                    for name, value in self.built.model.state_dict().items()
                ) != model_structure
                or hashlib.sha256(
                    json.dumps(
                        self.built.normalized_spec,
                        sort_keys=True,
                        ensure_ascii=False,
                        default=str,
                    ).encode("utf-8")
                ).hexdigest()[:20] != spec_id
                or {
                    sampler_id: int(value)
                    for sampler_id, value in dict(
                        getattr(
                            self.built.sampler, "_sampling_state_versions", {}
                        )
                        or {}
                    ).items()
                    if sampler_id in unaffected_sampler_ids
                }
                != sampler_versions_before
            ):
                raise ValueError("Curriculum advance changed structural training state")
        except Exception:
            if self.rollback_manager is not None and self.rollback_manager.active_checkpoint_id:
                self._rollback_active_action(
                    phase=self._last_training_phase,
                    reason="curriculum_runtime_transition_failed",
                )
            raise

        controller = self.curriculum_advance_controller
        controller.mark_action_applied(
            self._actual_iterations, forced=action.forced
        )
        if self.gradient_balance_controller is not None:
            self.gradient_balance_controller.on_curriculum_advanced(
                self._actual_iterations,
                float(controller.config["post_advance_controller_ema_decay"]),
            )
        if self.fixed_budget_sampling_controller is not None:
            self.fixed_budget_sampling_controller.on_curriculum_advanced(
                self._actual_iterations, action.associated_sampler_ids
            )
        if self.controller_coordinator is not None:
            self.controller_coordinator.record_outcome(
                action, iteration=self._actual_iterations, outcome="applied"
            )
            self.controller_coordinator.extend_observation_window(
                self._actual_iterations
                + int(action.limits["observation_window_iterations"])
            )
        self.curriculum_control_summary["curriculum_advances_executed"] += 1
        self.curriculum_control_summary["curriculum_advances_forced"] += int(
            action.forced
        )
        self.curriculum_control_summary["curriculum_runtime_refresh_cost"] += float(
            result.refresh_cost
        )
        per_level = self.curriculum_control_summary.setdefault(
            "iterations_per_level", {}
        ).setdefault(action.curriculum_id, {})
        per_level[action.from_level_id] = int(
            before_state.iterations_in_current_level
        )
        baseline = baseline_result.to_dict()
        self._pending_curriculum_observation = {
            "action": action,
            "curriculum_id": action.curriculum_id,
            "start_iteration": int(self._actual_iterations),
            "target_level_index": int(action.to_level_index),
            "target_runtime_state_version": int(result.runtime_state_version),
            "baseline": baseline,
            "validation_checks": 0,
            "degradation_checks": 0,
            "observation_window_iterations": int(
                action.limits["observation_window_iterations"]
            ),
            "checkpoint_id": checkpoint_id,
            "forced": bool(action.forced),
        }
        self._emit_adaptive_audit(
            iteration=self._actual_iterations,
            stage=str(self._last_training_phase.name),
            controller=action.controller,
            event_type="action_applied",
            trigger=dict(action.trigger_statistics),
            before={
                "curriculum_id": action.curriculum_id,
                "level": {
                    "index": action.from_level_index,
                    "id": action.from_level_id,
                },
                "runtime_state": {
                    "version": action.runtime_state_version,
                    "digest": action.runtime_state_digest,
                },
            },
            after={
                "level": {
                    "index": action.to_level_index,
                    "id": action.to_level_id,
                },
                "runtime_state": {
                    "version": result.runtime_state_version,
                    "digest": result.state_digest,
                },
                "forced": action.forced,
                "associated_component_ids": list(action.associated_component_ids),
                "associated_sampler_ids": list(action.associated_sampler_ids),
            },
            limits=dict(action.limits),
            checkpoint_id=checkpoint_id,
        )
        return result

    def _assess_pending_curriculum_action(
        self, result: Any, phase: Any | None
    ) -> bool:
        pending = self._pending_curriculum_observation
        controller = self.curriculum_advance_controller
        if pending is None or controller is None:
            return False
        runtime = self.curriculum_runtimes.get(pending["curriculum_id"])
        runtime_state = runtime.get_current_state() if runtime is not None else None
        integrity_bad = bool(
            runtime is None
            or runtime_state.current_level_index != pending["target_level_index"]
            or runtime_state.runtime_state_version
            != pending["target_runtime_state_version"]
            or not runtime.validate_runtime_state()
        )
        baseline = dict(pending["baseline"])
        pending["validation_checks"] = int(pending["validation_checks"]) + 1
        finite = math.isfinite(float(result.aggregate_score)) and all(
            math.isfinite(float(value))
            for value in result.normalized_component_scores.values()
        )
        aggregate_bad = bool(
            finite
            and float(result.aggregate_score)
            > float(baseline["aggregate_score"])
            * float(controller.config["aggregate_degradation_ratio"])
        )
        action = pending["action"]
        baseline_components = dict(
            baseline.get("normalized_component_scores") or {}
        )
        associated_bad = any(
            component_id in result.normalized_component_scores
            and component_id in baseline_components
            and float(result.normalized_component_scores[component_id])
            > float(baseline_components[component_id])
            * float(controller.config["associated_component_degradation_ratio"])
            for component_id in action.associated_component_ids
        )
        descriptor = self.registry.curriculum(action.curriculum_id)
        critical_bad = any(
            component_id in result.normalized_component_scores
            and component_id in baseline_components
            and float(result.normalized_component_scores[component_id])
            > float(baseline_components[component_id])
            * float(controller.config["critical_component_degradation_ratio"])
            for component_id in descriptor.critical_component_ids
        )
        non_finite_bad = bool(
            not finite and controller.config["rollback_on_non_finite"]
        )
        degraded = aggregate_bad or associated_bad or critical_bad or non_finite_bad
        pending["degradation_checks"] = (
            int(pending["degradation_checks"]) + 1 if degraded else 0
        )
        minimum_checks = int(controller.config["minimum_validation_checks"])
        if integrity_bad or int(pending["degradation_checks"]) >= minimum_checks:
            self._rollback_active_action(
                phase=phase or self._last_training_phase,
                reason=(
                    "curriculum_runtime_state_inconsistent"
                    if integrity_bad
                    else "curriculum_physics_validation_degradation"
                ),
            )
            return False
        complete = bool(
            self._actual_iterations - int(pending["start_iteration"])
            >= int(pending["observation_window_iterations"])
            and int(pending["validation_checks"]) >= minimum_checks
        )
        if not complete:
            return False
        if self.rollback_manager is not None:
            self.rollback_manager.clear()
        self._emit_adaptive_audit(
            iteration=self._actual_iterations,
            stage=str(getattr(phase, "name", "training")),
            controller="curriculum_advance",
            event_type="curriculum_observation_succeeded",
            trigger={
                "curriculum_id": pending["curriculum_id"],
                "validation_checks": pending["validation_checks"],
            },
            checkpoint_id=pending.get("checkpoint_id"),
        )
        self._pending_curriculum_observation = None
        return True

    def _finalize_curriculum_control_summary(self) -> dict[str, Any]:
        summary = deepcopy(self.curriculum_control_summary)
        per_level = summary.setdefault("iterations_per_level", {})
        unused: dict[str, list[str]] = {}
        for curriculum_id, runtime in sorted(self.curriculum_runtimes.items()):
            state = runtime.get_current_state()
            descriptor = self.registry.curriculum(curriculum_id)
            per_level.setdefault(curriculum_id, {})[
                state.current_level_id
            ] = int(state.iterations_in_current_level)
            unused[curriculum_id] = list(
                descriptor.level_ids[state.current_level_index + 1 :]
            )
        summary["unused_levels"] = unused
        return summary

    def _configure_training_stage(
        self,
        *,
        phase: Any,
        non_lbfgs_iteration: int,
    ) -> dict[str, Any]:
        """Apply component-category pretraining and role-based curriculum."""

        training = self.built.normalized_spec.get("training") or {}
        sampler = getattr(self.built, "sampler", None)
        loss_function = self.built.loss_function
        descriptor = self._stage_descriptor(phase)
        runtime_curriculum_active = bool(
            self.curriculum_runtimes
            and self.curriculum_advance_controller is not None
            and not self._curriculum_resume_disabled
        )
        if not descriptor.dynamic_objective_allowed:
            if (
                not runtime_curriculum_active
                and callable(getattr(sampler, "set_time_window_fraction", None))
            ):
                sampler.set_time_window_fraction(1.0)
            if callable(getattr(loss_function, "set_active_term_names", None)):
                loss_function.set_active_term_names(None)
            stage = {
                "name": (
                    "frozen_curriculum_state"
                    if runtime_curriculum_active
                    else "frozen_objective_full_domain"
                ),
                "time_window_fraction": float(
                    getattr(sampler, "time_window_fraction", 1.0)
                ),
                "initial_state_only": False,
            }
        else:
            pretraining = training.get("initial_state_pretraining") or {}
            pretraining_iterations = (
                int((pretraining.get("parameters") or {}).get("iterations") or 0)
                if pretraining.get("enabled")
                else 0
            )
            initial_only = non_lbfgs_iteration <= pretraining_iterations
            if callable(getattr(loss_function, "set_active_term_names", None)):
                pretraining_component_ids = {
                    component.component_id
                    for component in (
                        self.registry.loss_components
                        if self.registry is not None
                        else ()
                    )
                    if component.category == "initial_constraint"
                }
                loss_function.set_active_term_names(
                    pretraining_component_ids
                    if initial_only and pretraining_component_ids
                    else None
                )
            strategy = training.get("time_strategy") or {
                "name": "none",
                "parameters": {},
            }
            fraction = float(getattr(sampler, "time_window_fraction", 1.0))
            if not runtime_curriculum_active:
                fraction = 1.0
            if (
                not runtime_curriculum_active
                and strategy.get("name") == "time_window_curriculum"
            ):
                parameters = strategy.get("parameters") or {}
                fractions = [
                    float(value)
                    for value in parameters.get("window_fractions") or [1.0]
                ]
                iterations = [
                    int(value)
                    for value in parameters.get("iterations_per_window")
                    or [10**18]
                ]
                cursor = 0
                fraction = fractions[-1]
                for candidate_fraction, count in zip(fractions, iterations):
                    cursor += count
                    if non_lbfgs_iteration <= cursor:
                        fraction = candidate_fraction
                        break
            if (
                not runtime_curriculum_active
                and callable(getattr(sampler, "set_time_window_fraction", None))
            ):
                sampler.set_time_window_fraction(fraction)
            stage = {
                "name": (
                    "initial_state_pretraining"
                    if initial_only
                    else "runtime_curriculum"
                    if runtime_curriculum_active
                    else str(strategy.get("name") or "full_domain")
                ),
                "time_window_fraction": float(fraction),
                "initial_state_only": bool(initial_only),
            }
        self.time_strategy_audit = {
            "configured_time_strategy": dict(
                training.get("time_strategy") or {"name": "none", "parameters": {}}
            ),
            "configured_initial_state_pretraining": dict(
                training.get("initial_state_pretraining")
                or {"enabled": False, "parameters": {}}
            ),
            "window_history": list(
                getattr(sampler, "time_window_history", []) or []
            ),
            "latest_stage": dict(stage),
            "frozen_objective_full_domain": not descriptor.dynamic_objective_allowed,
            # Retained for schema/report compatibility with Trainer Controller
            # v1; the decision itself is now stage-capability driven.
            "lbfgs_full_domain": bool(
                descriptor.second_order and not descriptor.dynamic_objective_allowed
            ),
        }
        return stage

    def _run_lbfgs(
        self,
        phase: Any,
        budget: int,
        clipping_norm: float | None,
        clipping_norm_type: float,
    ) -> tuple[float | None, int, str | None, dict[str, Any]]:
        """Run L-BFGS in persistent chunks until its internal-iteration budget is met.

        ``phase.iterations`` denotes accepted PyTorch L-BFGS internal iterations,
        matching Adam's optimizer-iteration accounting.  Closure evaluations are
        audited separately because Strong-Wolfe can evaluate several trial points
        per internal iteration.  One full-batch sample is frozen for the complete
        phase so every line-search closure evaluates the same objective.
        """

        completed_iterations = 0
        closure_evaluations = 0
        accepted_loss_evaluations = 0
        internal_iterations = 0
        outer_calls = 0
        stalled_calls = 0
        consecutive_stalls = 0
        state_restarts = 0
        latest: float | None = None
        failure_reason: str | None = None
        termination_reason = "budget_fulfilled"
        converged_early = False
        controlled_stall = False
        recovery_adam_iterations = 0
        recovery_action: str | None = None
        stall_iteration: int | None = None
        physics_guard_triggered = False
        physics_guard_baseline_score: float | None = None
        physics_guard_rejected_score: float | None = None
        physics_guard_metric = str(
            getattr(
                self.validation_probe_manager,
                "second_order_guard_metric",
                "physics_validation_score",
            )
        )
        physics_guard_ratio = float(
            getattr(
                self.validation_probe_manager,
                "second_order_degradation_ratio",
                1.0,
            )
        )
        physics_guard_baseline = getattr(
            self.validation_probe_manager, "last_result", None
        )
        guard_score = getattr(
            self.validation_probe_manager,
            "second_order_guard_score",
            None,
        )
        if physics_guard_baseline is not None:
            physics_guard_baseline_score = (
                guard_score(physics_guard_baseline)
                if callable(guard_score)
                else float(physics_guard_baseline.aggregate_score)
            )
        physics_guard_model_state = (
            [
                parameter.detach().clone()
                for parameter in self.built.model.parameters()
            ]
            if physics_guard_ratio > 1.0
            and physics_guard_baseline_score is not None
            else None
        )
        best_train_checkpoint_manager = (
            self.best_training_loss_checkpoint_manager
        )
        physics_guard_best_train_checkpoint_state = (
            best_train_checkpoint_manager.state_dict()
            if physics_guard_model_state is not None
            and best_train_checkpoint_manager is not None
            else None
        )
        parameters = dict(getattr(phase, "parameters", {}) or {})
        first_group = phase.optimizer.param_groups[0]
        configured_max_iter = max(
            1, int(parameters.get("max_iter") or first_group.get("max_iter") or 20)
        )
        configured_max_eval = max(
            1, int(parameters.get("max_eval") or first_group.get("max_eval") or 25)
        )
        tolerance_grad = float(
            parameters.get("tolerance_grad")
            or first_group.get("tolerance_grad")
            or 1e-7
        )
        original_learning_rates = [float(group.get("lr", 1.0)) for group in phase.optimizer.param_groups]
        line_search_fn = first_group.get("line_search_fn")
        sampler = getattr(self.built, "sampler", None)
        phase_batch = sampler.sample() if sampler is not None else None
        frozen_loss = {}
        freeze = getattr(self.built.loss_function, "freeze_dynamic_weighting", None)
        if callable(freeze):
            frozen_loss = dict(freeze())
        registered_sampling_state = (
            sampler.snapshot_registered_components(self.registry, phase_batch)
            if sampler is not None
            and self.registry is not None
            and phase_batch is not None
            else {}
        )
        frozen_objective = {
            **frozen_loss,
            "frozen_sampling_components": registered_sampling_state,
            "frozen_sampling_ratios": dict(
                getattr(sampler, "last_sampling_snapshot", {}) or {}
            ),
            "frozen_time_window": float(
                getattr(sampler, "time_window_fraction", 1.0)
            ),
            "frozen_constraint_state": deepcopy(
                getattr(sampler, "_constraint_cache", {})
            ),
        }
        self._emit_adaptive_audit(
            iteration=self._actual_iterations,
            stage="lbfgs",
            controller="trainer_boundary",
            event_type="objective_frozen",
            trigger={"reason": "lbfgs_phase_entry"},
            after={
                "loss_weights": frozen_objective.get("frozen_loss_weights", {}),
                "loss_terms": frozen_objective.get("frozen_loss_terms", []),
                "time_window": frozen_objective["frozen_time_window"],
                "sampling_snapshot": frozen_objective["frozen_sampling_ratios"],
                "registered_sampling_component_ids": sorted(
                    registered_sampling_state
                ),
            },
        )
        self._last_training_batch = phase_batch
        self._last_training_phase = phase
        last_maximum_gradient: float | None = None
        last_step_size: float | None = None
        max_state_restarts = max(
            8,
            int(
                getattr(self.lbfgs_stall_controller, "config", {}).get(
                    "patience", 0
                )
            )
            + 2,
        )
        restart_after_stalls = 2
        max_outer_calls = max(16, int(budget) * 2 + 16)
        first_order_recovery_available = _has_first_order_recovery_phase(
            self.built.optimization_phases
        )

        def reallocate_non_finite_step(
            reason: str,
            parameter_values: list[torch.Tensor],
            loss_state: dict[str, Any],
            *,
            rejected_iterations: int = 0,
        ) -> bool:
            nonlocal completed_iterations, internal_iterations
            nonlocal failure_reason, termination_reason, controlled_stall
            nonlocal recovery_adam_iterations, recovery_action, stall_iteration
            if not first_order_recovery_available:
                return False
            if rejected_iterations > 0:
                completed_iterations = max(
                    0, completed_iterations - int(rejected_iterations)
                )
                internal_iterations = max(
                    0, internal_iterations - int(rejected_iterations)
                )
            with torch.no_grad():
                for parameter, previous in zip(
                    self.built.model.parameters(), parameter_values
                ):
                    parameter.copy_(previous.to(parameter.device))
            self._restore_lbfgs_loss_state(loss_state)
            self._reset_lbfgs_state(phase.optimizer)
            failure_reason = None
            controlled_stall = True
            recovery_action = "reallocate_to_adam_after_non_finite_lbfgs"
            recovery_adam_iterations = max(
                0, int(budget) - int(completed_iterations)
            )
            stall_iteration = int(self._actual_iterations)
            termination_reason = f"{reason}_reallocated_to_adam"
            self._emit_adaptive_audit(
                iteration=self._actual_iterations,
                stage="lbfgs",
                controller="lbfgs_non_finite_guard",
                event_type="budget_reallocated",
                trigger={"reason": reason},
                before={
                    "completed_lbfgs_iterations": int(completed_iterations),
                    "rejected_lbfgs_iterations": int(rejected_iterations),
                },
                after={
                    "recovery_adam_iterations": int(
                        recovery_adam_iterations
                    ),
                    "parameters_restored": True,
                },
            )
            return True

        while completed_iterations < budget and outer_calls < max_outer_calls:
            remaining_iterations = budget - completed_iterations
            call_max_iter = min(configured_max_iter, remaining_iterations)
            validation_manager = self.validation_probe_manager
            if (
                validation_manager is not None
                and validation_manager.probe_set is not None
                and validation_manager.history
            ):
                next_validation = validation_manager.next_due_iteration(
                    self._actual_iterations
                )
                if next_validation > self._actual_iterations:
                    call_max_iter = min(
                        call_max_iter,
                        max(1, next_validation - self._actual_iterations),
                    )
            next_checkpoint = next(
                (
                    checkpoint
                    for checkpoint in self.reference_mse_checkpoint_iterations
                    if checkpoint > self._actual_iterations
                ),
                None,
            )
            if next_checkpoint is not None:
                call_max_iter = min(
                    call_max_iter,
                    max(1, next_checkpoint - self._actual_iterations),
                )
            call_max_eval = max(
                1,
                min(
                    configured_max_eval,
                    math.ceil(
                        configured_max_eval * call_max_iter / configured_max_iter
                    ),
                ),
            )
            for group in phase.optimizer.param_groups:
                group["max_iter"] = call_max_iter
                group["max_eval"] = call_max_eval

            state_before = self._lbfgs_optimizer_state(phase.optimizer)
            n_iter_before = int(state_before.get("n_iter", 0)) if state_before else 0
            closure_calls_this_step = 0
            last_maximum_gradient = None
            loss_state_before = self._snapshot_lbfgs_loss_state()
            parameter_values_before = [
                parameter.detach().clone()
                for parameter in self.built.model.parameters()
            ]

            def closure():
                nonlocal closure_evaluations, closure_calls_this_step
                nonlocal failure_reason, last_maximum_gradient
                phase.optimizer.zero_grad(set_to_none=True)
                self._restore_lbfgs_loss_state(loss_state_before)
                self.built.loss_function.capture_residual_statistics = False
                loss, _ = self.built.loss_function(self.built.model, phase_batch)
                if not torch.isfinite(loss):
                    failure_reason = "non_finite_loss"
                    self._release_iteration_graphs()
                    raise _StopLBFGS(failure_reason)
                loss.backward()
                if clipping_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.built.model.parameters(),
                        clipping_norm,
                        norm_type=clipping_norm_type,
                    )
                if not self._gradients_finite():
                    failure_reason = "non_finite_gradient"
                    self._release_iteration_graphs()
                    raise _StopLBFGS(failure_reason)
                last_maximum_gradient = self._maximum_absolute_gradient()
                closure_evaluations += 1
                closure_calls_this_step += 1
                self._release_iteration_graphs()
                return loss

            outer_calls += 1
            try:
                phase.optimizer.step(closure)
            except _StopLBFGS:
                reason = str(failure_reason or "non_finite_lbfgs_step")
                reallocate_non_finite_step(
                    reason,
                    parameter_values_before,
                    loss_state_before,
                )
                break

            if not self._parameters_finite():
                failure_reason = "non_finite_parameter"
                reallocate_non_finite_step(
                    failure_reason,
                    parameter_values_before,
                    loss_state_before,
                )
                break

            state_after = self._lbfgs_optimizer_state(phase.optimizer)
            if state_after is None:
                # Minimal optimizer doubles used by unit tests have no PyTorch
                # state.  A successful step is one optimizer iteration there.
                iteration_delta = 1 if closure_calls_this_step else 0
                last_step_size = None
            else:
                n_iter_after = int(state_after.get("n_iter", n_iter_before))
                iteration_delta = max(0, n_iter_after - n_iter_before)
                raw_step_size = state_after.get("t")
                try:
                    last_step_size = (
                        float(raw_step_size.detach().cpu())
                        if isinstance(raw_step_size, torch.Tensor)
                        else float(raw_step_size)
                        if raw_step_size is not None
                        else None
                    )
                except (TypeError, ValueError):
                    last_step_size = None

            iteration_delta = min(iteration_delta, remaining_iterations)
            internal_iterations += iteration_delta
            completed_iterations += iteration_delta

            projected_iteration = self._actual_iterations + iteration_delta
            should_record = iteration_delta > 0 and self._should_record_history(
                projected_iteration,
                phase.name,
            )
            self._restore_lbfgs_loss_state(loss_state_before)
            self.built.loss_function.capture_residual_statistics = should_record
            phase.optimizer.zero_grad(set_to_none=True)
            accepted_loss, accepted_components = self.built.loss_function(
                self.built.model, phase_batch
            )
            accepted_loss_evaluations += 1
            if not torch.isfinite(accepted_loss):
                failure_reason = "non_finite_accepted_loss"
                reallocate_non_finite_step(
                    failure_reason,
                    parameter_values_before,
                    loss_state_before,
                    rejected_iterations=iteration_delta,
                )
                self._release_iteration_graphs()
                break
            latest = float(accepted_loss.detach().cpu())
            parameter_update_norm = self._parameter_update_norm(
                parameter_values_before
            )
            if iteration_delta > 0:
                self._actual_iterations += iteration_delta
                self._record_reference_mse_checkpoint(self._actual_iterations)
                if should_record:
                    gradient_statistics = None
                    if self._gradient_diagnostic_due(self._actual_iterations):
                        gradient_statistics = (
                            self._collect_component_gradient_diagnostic(
                                total_loss=accepted_loss,
                                iteration=self._actual_iterations,
                                phase=phase,
                                checkpoint_kind=self._gradient_checkpoint_kind(
                                    self._actual_iterations
                                ),
                            )
                        )
                    self.history.append(
                        self._history_entry(
                            iteration=self._actual_iterations,
                            phase=phase,
                            total_loss=latest,
                            components=accepted_components,
                            batch=phase_batch,
                            gradient_norm=None,
                            gradient_statistics=gradient_statistics,
                            adaptive_refinement_event=False,
                        )
                    )
                    self._history_phase_names_recorded.add(str(phase.name))
            self._release_iteration_graphs()
            if iteration_delta > 0:
                self._maybe_run_physics_validation(
                    phase=phase,
                    # A short L-BFGS phase may end before the next periodic
                    # probe. Always evaluate its terminal state so the
                    # label-free degradation guard cannot be skipped.
                    force=completed_iterations >= budget,
                    training_loss=latest,
                )
                latest_validation = getattr(
                    self.validation_probe_manager, "last_result", None
                )
                validation_history = list(
                    getattr(
                        self.validation_probe_manager, "history", []
                    )
                    or []
                )
                validation_is_current = bool(
                    validation_history
                    and int(
                        validation_history[-1].get("iteration", -1)
                    )
                    == int(self._actual_iterations)
                )
                current_guard_score = (
                    guard_score(latest_validation)
                    if callable(guard_score)
                    else float(latest_validation.aggregate_score)
                    if latest_validation is not None
                    else None
                )
                degradation_limit = (
                    float(physics_guard_baseline_score)
                    + (physics_guard_ratio - 1.0)
                    * max(abs(float(physics_guard_baseline_score)), 1.0e-12)
                    if physics_guard_baseline_score is not None
                    else None
                )
                if (
                    physics_guard_model_state is not None
                    and current_guard_score is not None
                    and validation_is_current
                    and degradation_limit is not None
                    and float(current_guard_score) > degradation_limit
                ):
                    physics_guard_triggered = True
                    physics_guard_rejected_score = float(current_guard_score)
                    with torch.no_grad():
                        for parameter, previous in zip(
                            self.built.model.parameters(),
                            physics_guard_model_state,
                        ):
                            parameter.copy_(
                                previous.to(parameter.device)
                            )
                    if (
                        best_train_checkpoint_manager is not None
                        and physics_guard_best_train_checkpoint_state
                        is not None
                    ):
                        # The rejected L-BFGS state may have a lower summed
                        # training loss and therefore may already be the
                        # best-train-loss checkpoint. Restore the checkpoint
                        # manager alongside the guarded model so final model
                        # selection cannot reload the rejected weights.
                        best_train_checkpoint_manager.load_state_dict(
                            physics_guard_best_train_checkpoint_state
                        )
                        if (
                            best_train_checkpoint_manager.best_score
                            is not None
                            and math.isfinite(
                                float(
                                    best_train_checkpoint_manager.best_score
                                )
                            )
                        ):
                            # Keep the returned/final training loss aligned
                            # with the restored model. Otherwise the terminal
                            # bookkeeping pass could pair the rejected
                            # L-BFGS loss with the guarded weights and create
                            # a misleading new best checkpoint.
                            latest = float(
                                best_train_checkpoint_manager.best_score
                            )
                    self._reset_lbfgs_state(phase.optimizer)
                    controlled_stall = True
                    if first_order_recovery_available:
                        recovery_action = (
                            "reallocate_to_adam_after_physics_degradation"
                        )
                        recovery_adam_iterations = max(
                            0, int(budget) - int(completed_iterations)
                        )
                    else:
                        recovery_action = (
                            "terminate_lbfgs_after_physics_degradation"
                        )
                        recovery_adam_iterations = 0
                    stall_iteration = int(self._actual_iterations)
                    termination_reason = (
                        "physics_validation_degradation_reallocated_to_adam"
                        if first_order_recovery_available
                        else "physics_validation_degradation_terminated_lbfgs"
                    )
                    self._emit_adaptive_audit(
                        iteration=self._actual_iterations,
                        stage="lbfgs",
                        controller="lbfgs_physics_guard",
                        event_type=(
                            "budget_reallocated"
                            if first_order_recovery_available
                            else "stage_terminated"
                        ),
                        trigger={
                            "metric": physics_guard_metric,
                            "baseline_score": (
                                physics_guard_baseline_score
                            ),
                            "rejected_score": (
                                physics_guard_rejected_score
                            ),
                            "maximum_ratio": physics_guard_ratio,
                        },
                        after={
                            "model_restored_to": "pre_second_order",
                            "recovery_adam_iterations": int(
                                recovery_adam_iterations
                            ),
                        },
                    )
                    break
                if (
                    physics_guard_model_state is not None
                    and current_guard_score is not None
                    and validation_is_current
                    and physics_guard_baseline_score is not None
                    and float(current_guard_score)
                    < float(physics_guard_baseline_score)
                ):
                    physics_guard_baseline_score = float(current_guard_score)
                    physics_guard_model_state = [
                        parameter.detach().clone()
                        for parameter in self.built.model.parameters()
                    ]
                    if best_train_checkpoint_manager is not None:
                        physics_guard_best_train_checkpoint_state = (
                            best_train_checkpoint_manager.state_dict()
                        )

            if self.lbfgs_stall_controller is not None:
                state = TrainerObservableState(
                    iteration=int(self._actual_iterations),
                    stage=str(phase.name),
                    optimizer_name=str(phase.optimizer_name),
                    current_loss=latest,
                    loss_components=dict(accepted_components),
                    loss_weights=dict(
                        getattr(
                            self.built.loss_function,
                            "last_effective_loss_weights",
                            {},
                        )
                    ),
                    learning_rates=tuple(
                        float(group.get("lr", 0.0))
                        for group in phase.optimizer.param_groups
                    ),
                    parameter_update_norm=parameter_update_norm,
                    gradient_norm=last_maximum_gradient,
                    lbfgs_outer_step=outer_calls,
                    closure_evaluations=closure_evaluations,
                    closure_evaluations_this_step=closure_calls_this_step,
                    step_size=last_step_size,
                    line_search_failed=iteration_delta == 0,
                    completed_stage_iterations=completed_iterations,
                    remaining_iteration_budget=max(
                        0, budget - completed_iterations
                    ),
                    sampling_state=dict(
                        getattr(sampler, "last_sampling_snapshot", {}) or {}
                    ),
                    time_window_fraction=float(
                        getattr(sampler, "time_window_fraction", 1.0)
                    ),
                    numerical_finite=True,
                    stage_descriptor=self._stage_descriptor(phase),
                    capability_snapshot=(
                        self.registry.capabilities
                        if self.registry is not None
                        else None
                    ),
                    eligible_component_ids=(),
                    optimizer_diagnostics={
                        "outer_step": outer_calls,
                        "closure_evaluations": closure_evaluations,
                        "step_size": last_step_size,
                        "parameter_update_norm": parameter_update_norm,
                    },
                    budget_state={
                        "completed_stage_iterations": completed_iterations,
                        "remaining_stage_iterations": max(
                            0, budget - completed_iterations
                        ),
                    },
                    validation_state=getattr(
                        self.validation_probe_manager, "last_result", None
                    ),
                )
                if self.controller_coordinator is not None:
                    diagnostic = self.controller_coordinator.observe_controller(
                        self.lbfgs_stall_controller, state
                    )
                else:
                    diagnostic = self.lbfgs_stall_controller.observe(state)
                self._emit_adaptive_audit(
                    iteration=self._actual_iterations,
                    stage="lbfgs",
                    controller="lbfgs_stall",
                    event_type="diagnostic",
                    trigger=diagnostic,
                )
                proposals = (
                    self.controller_coordinator.collect_proposals()
                    if self.controller_coordinator is not None
                    else tuple(
                        item
                        for item in (
                            self.lbfgs_stall_controller.propose_action(),
                        )
                        if item is not None
                    )
                )
                if self.controller_coordinator is not None:
                    action, rejected = self.controller_coordinator.select_action(
                        proposals, iteration=self._actual_iterations
                    )
                    for action_id, reason in rejected.items():
                        self._emit_adaptive_audit(
                            iteration=self._actual_iterations,
                            stage=str(phase.name),
                            controller="controller_coordinator",
                            event_type="action_rejected_by_coordinator",
                            trigger={"action_id": action_id, "reason": reason},
                        )
                else:
                    action = proposals[0] if proposals else None
                if action is not None:
                    if self.action_boundary_validator is not None and self.registry is not None:
                        self.action_boundary_validator.validate(
                            action=action,
                            registry=self.registry,
                            stage=self._stage_descriptor(phase),
                            observable_state=state,
                        )
                    checkpoint_id = self._save_pre_action_checkpoint(
                        action, latest
                    )
                    recovery_action = action.action_type
                    recovery_adam_iterations = int(
                        action.after.get("recovery_adam_iterations", 0)
                    )
                    controlled_stall = True
                    stall_iteration = int(self._actual_iterations)
                    termination_reason = "controlled_lbfgs_stall"
                    self.lbfgs_stall_controller.mark_action_applied(
                        self._actual_iterations
                    )
                    if self.controller_coordinator is not None:
                        self.controller_coordinator.record_outcome(
                            action,
                            iteration=self._actual_iterations,
                            outcome="applied",
                        )
                    self._emit_adaptive_audit(
                        iteration=self._actual_iterations,
                        stage="lbfgs",
                        controller=action.controller,
                        event_type="lbfgs_stall",
                        trigger=dict(action.reason),
                        before=dict(action.before),
                        after=dict(action.after),
                        limits=dict(action.limits),
                        checkpoint_id=checkpoint_id,
                    )
                    self._emit_adaptive_audit(
                        iteration=self._actual_iterations,
                        stage="lbfgs",
                        controller=action.controller,
                        event_type="budget_reallocated",
                        trigger={"action": recovery_action},
                        before={"remaining_lbfgs_iterations": budget - completed_iterations},
                        after={"recovery_adam_iterations": recovery_adam_iterations},
                        limits=dict(action.limits),
                        checkpoint_id=checkpoint_id,
                    )
                    break

            genuinely_converged = (
                last_maximum_gradient is not None
                and math.isfinite(last_maximum_gradient)
                and last_maximum_gradient <= tolerance_grad
            )
            zero_step = last_step_size is not None and abs(last_step_size) == 0.0
            stalled = iteration_delta == 0 or (zero_step and not genuinely_converged)
            if stalled:
                stalled_calls += 1
                consecutive_stalls += 1
            else:
                consecutive_stalls = 0

            if genuinely_converged and iteration_delta == 0:
                converged_early = True
                termination_reason = "gradient_tolerance_reached"
                break

            if consecutive_stalls >= restart_after_stalls:
                if state_restarts >= max_state_restarts:
                    if iteration_delta == 0:
                        failure_reason = (
                            "lbfgs_iteration_budget_unreachable: "
                            f"completed={completed_iterations}, requested={budget}"
                        )
                        termination_reason = "stall_recovery_exhausted"
                        break
                else:
                    state_restarts += 1
                    self._reset_lbfgs_state(phase.optimizer)
                    for index, group in enumerate(phase.optimizer.param_groups):
                        group["lr"] = max(
                            original_learning_rates[index] * (0.5**state_restarts),
                            1e-8,
                        )
                    consecutive_stalls = 0

        if completed_iterations < budget and failure_reason is None and not converged_early and not controlled_stall:
            failure_reason = (
                "lbfgs_iteration_budget_unreachable: "
                f"completed={completed_iterations}, requested={budget}"
            )
            termination_reason = "outer_call_limit_reached"
        elif failure_reason is not None and termination_reason == "budget_fulfilled":
            termination_reason = failure_reason

        for group in phase.optimizer.param_groups:
            group["max_iter"] = configured_max_iter
            group["max_eval"] = configured_max_eval

        audit = {
            "policy": "persistent_chunked_internal_iterations_v1",
            "configured_phase_iterations": int(budget),
            "completed_internal_iterations": int(completed_iterations),
            "outer_step_calls": int(outer_calls),
            "closure_evaluations": int(closure_evaluations),
            "accepted_loss_evaluations": int(accepted_loss_evaluations),
            "configured_max_iter_per_call": int(configured_max_iter),
            "configured_max_eval_per_call": int(configured_max_eval),
            "line_search_fn": line_search_fn,
            "fixed_full_batch_within_phase": True,
            "fixed_dynamic_loss_state_within_outer_step": True,
            "fixed_dynamic_loss_state_within_phase": True,
            "frozen_objective": {
                "loss_weights": frozen_objective.get("frozen_loss_weights", {}),
                "loss_terms": frozen_objective.get("frozen_loss_terms", []),
                "time_window": frozen_objective.get("frozen_time_window"),
                "sampling_snapshot": frozen_objective.get("frozen_sampling_ratios", {}),
                "registered_sampling_component_ids": sorted(
                    frozen_objective.get("frozen_sampling_components", {})
                ),
            },
            "stalled_calls": int(stalled_calls),
            "state_restarts": int(state_restarts),
            "last_step_size": last_step_size,
            "last_maximum_absolute_gradient": last_maximum_gradient,
            "converged_early": bool(converged_early),
            "budget_fulfilled": completed_iterations >= budget,
            "termination_reason": termination_reason,
            "controlled_stall_detected": controlled_stall,
            "stall_iteration": stall_iteration,
            "recovery_action": recovery_action,
            "recovery_adam_iterations": recovery_adam_iterations,
            "physics_guard_triggered": physics_guard_triggered,
            "physics_guard_metric": physics_guard_metric,
            "physics_guard_baseline_score": (
                physics_guard_baseline_score
            ),
            "physics_guard_rejected_score": (
                physics_guard_rejected_score
            ),
            "physics_guard_maximum_ratio": physics_guard_ratio,
            "skipped_stalled_iterations": max(
                0, int(budget) - int(completed_iterations)
            ),
            "final_learning_rates": [
                float(group.get("lr", 0.0)) for group in phase.optimizer.param_groups
            ],
        }
        return latest, completed_iterations, failure_reason, audit

    @staticmethod
    def _lbfgs_optimizer_state(optimizer: Any) -> dict[str, Any] | None:
        state = getattr(optimizer, "state", None)
        groups = getattr(optimizer, "param_groups", None) or []
        if not state or not groups:
            return None
        parameters = groups[0].get("params") or []
        if parameters and parameters[0] in state and isinstance(state[parameters[0]], dict):
            return state[parameters[0]]
        for value in state.values():
            if isinstance(value, dict) and ("n_iter" in value or "func_evals" in value):
                return value
        return None

    @staticmethod
    def _reset_lbfgs_state(optimizer: Any) -> None:
        state = getattr(optimizer, "state", None)
        if hasattr(state, "clear"):
            state.clear()

    def _snapshot_lbfgs_loss_state(self) -> dict[str, torch.Tensor | None]:
        loss_function = self.built.loss_function
        snapshot: dict[str, torch.Tensor | None] = {}
        for name in (
            "_adaptive_weights",
            "_ema_magnitudes",
            "_ntk_weights",
        ):
            value = getattr(loss_function, name, None)
            snapshot[name] = value.detach().clone() if isinstance(value, torch.Tensor) else None
        return snapshot

    def _restore_lbfgs_loss_state(
        self, snapshot: dict[str, torch.Tensor | None]
    ) -> None:
        loss_function = self.built.loss_function
        for name, value in snapshot.items():
            setattr(loss_function, name, value.detach().clone() if value is not None else None)

    def _configure_history_boundary_iterations(
        self,
        *,
        start_iteration: int,
        maximum_iterations: int,
    ) -> None:
        """Mark the first and final optimizer iterations in this train call."""

        cursor = int(start_iteration)
        remaining = max(0, int(maximum_iterations))
        boundaries: set[int] = set()
        for phase in self.built.optimization_phases:
            completed = int(self.completed_phase_iterations.get(phase.name, 0))
            available = max(0, int(phase.iterations) - completed)
            count = min(available, remaining)
            if count <= 0:
                continue
            boundaries.add(cursor + 1)
            boundaries.add(cursor + count)
            cursor += count
            remaining -= count
            if remaining <= 0:
                break
        self._history_boundary_iterations = boundaries

    def _should_record_history(
        self,
        iteration: int,
        phase_name: str | None = None,
        *,
        event: bool = False,
    ) -> bool:
        return (
            bool(event)
            or iteration == 1
            or iteration in self._history_boundary_iterations
            or (
                phase_name is not None
                and str(phase_name) not in self._history_phase_names_recorded
            )
            or iteration % self.history_interval == 0
        )

    def _release_iteration_graphs(self) -> None:
        """Drop graph references that must not survive an optimizer step."""

        loss_function = self.built.loss_function
        if hasattr(loss_function, "last_term_values"):
            loss_function.last_term_values = []
        if hasattr(loss_function, "last_term_names"):
            loss_function.last_term_names = []
        problem = getattr(loss_function, "problem", None)
        clear = getattr(problem, "clear_autodiff_cache", None)
        if callable(clear):
            clear()

    def _history_entry(
        self,
        *,
        iteration: int,
        phase: Any,
        total_loss: float,
        components: dict[str, float],
        batch: Any | None,
        gradient_norm: float | None,
        gradient_statistics: dict[str, Any] | None,
        adaptive_refinement_event: bool,
        training_stage: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        residual_statistics = dict(
            getattr(self.built.loss_function, "last_residual_statistics", {}) or {}
        )
        entry = {
            "iteration": int(iteration),
            "phase": phase.name,
            "elapsed_seconds": time.perf_counter() - self._train_started,
            "total_loss": total_loss,
            "component_losses": {
                str(name): float(value)
                for name, value in components.items()
                if name != "total_loss"
            },
            "component_category_losses": self._component_category_losses(
                components
            ),
            "learning_rate": float(phase.optimizer.param_groups[0].get("lr", 0.0)),
            "sampling_point_count": (
                self._sampling_point_count(batch)
                if batch is not None
                else self._configured_sampling_point_count()
            ),
            "maximum_residual": residual_statistics.get("maximum_residual"),
            "mean_residual": residual_statistics.get("mean_residual"),
            "raw_maximum_residual": residual_statistics.get(
                "raw_maximum_residual"
            ),
            "raw_mean_residual": residual_statistics.get(
                "raw_mean_residual"
            ),
            "residual_region_statistics": residual_statistics.get(
                "region_statistics"
            ),
            "mean_region_training_residual_rmse": residual_statistics.get(
                "mean_region_training_residual_rmse"
            ),
            "worst_region_training_residual_rmse": residual_statistics.get(
                "worst_region_training_residual_rmse"
            ),
            "mean_region_raw_residual_rmse": residual_statistics.get(
                "mean_region_raw_residual_rmse"
            ),
            "worst_region_raw_residual_rmse": residual_statistics.get(
                "worst_region_raw_residual_rmse"
            ),
            "gradient_norm": gradient_norm,
            "gradient_statistics": (
                dict(gradient_statistics) if gradient_statistics else None
            ),
            "nan_inf_status": "finite",
            "gpu_memory_allocated": (
                int(torch.cuda.memory_allocated(next(self.built.model.parameters()).device))
                if next(self.built.model.parameters()).device.type == "cuda" and torch.cuda.is_available()
                else 0
            ),
            "adaptive_refinement_event": bool(adaptive_refinement_event),
            "raw_loss_components": dict(components),
            "raw_losses": dict(
                getattr(
                    self.built.loss_function,
                    "last_raw_losses",
                    {},
                )
            ),
            "dynamic_weighting": (
                dict(self.built.loss_function.last_dynamic_weighting)
                if getattr(
                    self.built.loss_function,
                    "last_dynamic_weighting",
                    None,
                )
                else None
            ),
            "effective_loss_weights": dict(
                getattr(
                    self.built.loss_function,
                    "last_effective_loss_weights",
                    {},
                )
            ),
            "weighted_losses": dict(
                getattr(
                    self.built.loss_function,
                    "last_weighted_losses",
                    {},
                )
            ),
            "training_stage": dict(training_stage or {}),
            "time_window_fraction": float(
                (training_stage or {}).get("time_window_fraction", 1.0)
            ),
        }
        if self._should_probe_reference_mse(iteration, str(phase.name)):
            try:
                value = (
                    self.reference_mse_probe()
                    if self.reference_mse_probe
                    else None
                )
                entry["reference_mse"] = value
                self._consider_reference_mse_checkpoint(
                    value,
                    iteration=iteration,
                    phase_name=str(phase.name),
                )
            except Exception as exc:
                # Posterior observation must never change the optimizer outcome.
                entry["reference_mse"] = None
                entry["reference_mse_probe_error"] = f"{type(exc).__name__}: {exc}"
        return entry

    def _should_probe_reference_mse(self, iteration: int, phase_name: str) -> bool:
        if self.reference_mse_probe is None or self.reference_mse_interval <= 0:
            return False
        first_in_phase = phase_name not in self._reference_probe_phases
        if first_in_phase:
            self._reference_probe_phases.add(phase_name)
        return first_in_phase or int(iteration) % self.reference_mse_interval == 0

    def _record_reference_mse_checkpoint(self, iteration: int) -> None:
        checkpoint = int(iteration)
        if (
            self.reference_mse_probe is None
            or checkpoint not in self.reference_mse_checkpoint_iterations
            or str(checkpoint) in self.reference_mse_checkpoints
        ):
            return
        try:
            value = self.reference_mse_probe()
            number = float(value) if value is not None else None
            self.reference_mse_checkpoints[str(checkpoint)] = (
                number if number is not None and math.isfinite(number) else None
            )
            self._consider_reference_mse_checkpoint(
                number,
                iteration=checkpoint,
                phase_name=str(
                    getattr(self._last_training_phase, "name", "unknown")
                ),
            )
        except Exception:
            # Fixed-grid proxy observation must not alter optimizer control flow.
            self.reference_mse_checkpoints[str(checkpoint)] = None

    def _record_last_reference_mse_checkpoint(self) -> None:
        """Record the last optimizer-state MSE before model-selection restore."""

        iteration = int(self._actual_iterations)
        self.last_checkpoint_iteration = iteration if iteration > 0 else None
        if self.reference_mse_probe is None or iteration <= 0:
            self.last_checkpoint_mse = None
            return
        existing = self.reference_mse_checkpoints.get(str(iteration))
        try:
            number = float(existing)
        except (TypeError, ValueError):
            number = float("nan")
        if math.isfinite(number):
            self.last_checkpoint_mse = number
            self._consider_reference_mse_checkpoint(
                number,
                iteration=iteration,
                phase_name=str(
                    getattr(self._last_training_phase, "name", "unknown")
                ),
            )
            return
        try:
            value = self.reference_mse_probe()
            number = float(value) if value is not None else float("nan")
            self.last_checkpoint_mse = number if math.isfinite(number) else None
            self._consider_reference_mse_checkpoint(
                self.last_checkpoint_mse,
                iteration=iteration,
                phase_name=str(
                    getattr(self._last_training_phase, "name", "unknown")
                ),
            )
        except Exception:
            # Reference evaluation is observational and must not fail training.
            self.last_checkpoint_mse = None

    def _consider_reference_mse_checkpoint(
        self,
        value: Any,
        *,
        iteration: int,
        phase_name: str,
    ) -> bool:
        """Keep the best objective checkpoint only for explicit final review."""

        if not self.select_best_reference_mse_checkpoint:
            return False
        try:
            mse = float(value)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(mse):
            return False
        current = self.best_reference_mse_checkpoint
        if current is not None and mse >= float(current["mse"]):
            return False
        self._best_reference_mse_model_state = {
            name: tensor.detach().clone().cpu()
            for name, tensor in self.built.model.state_dict().items()
        }
        self.best_reference_mse_checkpoint = {
            "iteration": int(iteration),
            "phase": str(phase_name),
            "mse": mse,
            "selection_scope": "final_high_fidelity_only",
        }
        return True

    def _restore_best_reference_mse_model(self) -> bool:
        if (
            not self.select_best_reference_mse_checkpoint
            or self._best_reference_mse_model_state is None
        ):
            return False
        device = next(self.built.model.parameters()).device
        self.built.model.load_state_dict(
            {
                name: tensor.to(device)
                for name, tensor in self._best_reference_mse_model_state.items()
            }
        )
        return True

    @staticmethod
    def _sampling_point_count(batch: Any | None) -> int | None:
        if batch is None:
            return None
        total = int(getattr(batch.domain_samples, "shape", [0])[0])
        for value in (getattr(batch, "constraint_samples", {}) or {}).values():
            points = getattr(value, "points", value)
            if hasattr(points, "shape") and len(points.shape) > 0:
                total += int(points.shape[0])
        return total

    def _configured_sampling_point_count(self) -> int:
        if self.registry is not None:
            return sum(
                int(item.point_budget or 0)
                for item in self.registry.sampling_components
            )
        sampling = self.built.normalized_spec.get("sampling") or {}
        return sum(
            int(((section or {}).get("parameters") or {}).get("n_points") or 0)
            for section in sampling.values()
            if isinstance(section, dict)
        )

    def _component_category_losses(
        self, components: dict[str, float]
    ) -> dict[str, dict[str, float]]:
        if self.registry is None:
            return {"unclassified": dict(components)}
        result: dict[str, dict[str, float]] = {}
        for descriptor in self.registry.loss_components:
            if descriptor.component_id in components:
                result.setdefault(descriptor.category, {})[
                    descriptor.component_id
                ] = float(components[descriptor.component_id])
        return result

    def _gradient_norm(self) -> float | None:
        squared = 0.0
        found = False
        for parameter in self.built.model.parameters():
            if parameter.grad is None:
                continue
            found = True
            norm = float(torch.linalg.vector_norm(parameter.grad.detach()).cpu())
            squared += norm * norm
        return squared**0.5 if found else None

    def _configure_gradient_audit_targets(
        self,
        *,
        start_iteration: int,
        planned_iterations: int,
    ) -> None:
        if not self.component_gradient_diagnostics_enabled or planned_iterations <= 0:
            self._gradient_audit_targets = []
            return
        first = int(start_iteration) + 1
        middle = int(start_iteration) + max(1, math.ceil(planned_iterations / 2))
        last = int(start_iteration) + int(planned_iterations)
        self._gradient_audit_targets = sorted({first, middle, last})

    def _gradient_diagnostic_due(self, iteration: int) -> bool:
        if not self.component_gradient_diagnostics_enabled:
            return False
        return any(
            target <= int(iteration)
            and target not in self._gradient_audit_targets_completed
            for target in self._gradient_audit_targets
        )

    def _gradient_checkpoint_kind(self, iteration: int) -> str:
        pending = [
            target
            for target in self._gradient_audit_targets
            if target <= int(iteration)
            and target not in self._gradient_audit_targets_completed
        ]
        if not pending:
            return "unscheduled"
        target = max(pending)
        if target == self._gradient_audit_targets[0]:
            return "first"
        if target == self._gradient_audit_targets[-1]:
            return "last"
        return "middle"

    def _collect_component_gradient_diagnostic(
        self,
        *,
        total_loss: torch.Tensor,
        iteration: int,
        phase: Any,
        checkpoint_kind: str,
    ) -> dict[str, Any]:
        names = list(
            getattr(self.built.loss_function, "last_term_names", []) or []
        )
        values = list(
            getattr(self.built.loss_function, "last_term_values", []) or []
        )
        weights = dict(
            getattr(
                self.built.loss_function,
                "last_effective_loss_weights",
                {},
            )
            or {}
        )
        eligible = (
            {
                item.component_id
                for item in self.registry.loss_components
                if item.trainable
                and (not item.active_stages or str(phase.name) in item.active_stages)
            }
            if self.registry is not None
            else set(str(name) for name in names)
        )
        components = {
            str(name): value * float(weights.get(str(name), 1.0))
            for name, value in sorted(
                zip(names, values), key=lambda item: str(item[0])
            )[: self.maximum_gradient_components]
            if str(name) in eligible
        }
        diagnostic = collect_component_gradient_diagnostics(
            total_loss=total_loss,
            component_losses=components,
            parameters=list(self.built.model.parameters()),
            audit_iteration=int(iteration),
            gradient_imbalance_threshold=self.gradient_imbalance_threshold,
            gradient_conflict_threshold=self.gradient_conflict_threshold,
            maximum_components=self.maximum_gradient_components,
            maximum_conflicting_pairs=self.maximum_conflicting_pairs,
        )
        diagnostic.update(
            {
                "checkpoint_kind": checkpoint_kind,
                "phase": str(phase.name),
                "optimizer": str(getattr(phase, "optimizer_name", phase.name)),
                "learning_rates": [
                    float(group.get("lr", 0.0))
                    for group in phase.optimizer.param_groups
                ],
                "scheduler": (
                    type(phase.scheduler).__name__
                    if phase.scheduler is not None
                    else None
                ),
                "scheduler_last_epoch": (
                    int(getattr(phase.scheduler, "last_epoch"))
                    if phase.scheduler is not None
                    and getattr(phase.scheduler, "last_epoch", None) is not None
                    else None
                ),
                "component_weight_basis": "effective_registered_loss_weights",
            }
        )
        crossed = [
            target
            for target in self._gradient_audit_targets
            if target <= int(iteration)
            and target not in self._gradient_audit_targets_completed
        ]
        diagnostic["satisfied_audit_targets"] = crossed
        self._gradient_audit_targets_completed.update(crossed)
        self.component_gradient_audits.append(diagnostic)
        return diagnostic

    def _collect_final_component_gradient_diagnostic(self) -> None:
        if (
            not self.component_gradient_diagnostics_enabled
            or self._actual_iterations <= 0
            or self._last_training_batch is None
            or self._last_training_phase is None
        ):
            return
        if (
            self.component_gradient_audits
            and int(
                self.component_gradient_audits[-1].get("audit_iteration") or -1
            )
            == self._actual_iterations
        ):
            return
        loss_state = self._snapshot_lbfgs_loss_state()
        try:
            loss, _ = self.built.loss_function(
                self.built.model,
                self._last_training_batch,
            )
            if torch.isfinite(loss):
                self._collect_component_gradient_diagnostic(
                    total_loss=loss,
                    iteration=self._actual_iterations,
                    phase=self._last_training_phase,
                    checkpoint_kind="last_before_evaluation",
                )
        finally:
            self._restore_lbfgs_loss_state(loss_state)
            self._release_iteration_graphs()

    def _maximum_absolute_gradient(self) -> float | None:
        maximum: float | None = None
        for parameter in self.built.model.parameters():
            if parameter.grad is None:
                continue
            value = float(parameter.grad.detach().abs().max().cpu())
            maximum = value if maximum is None else max(maximum, value)
        return maximum

    def _gradients_finite(self) -> bool:
        return all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in self.built.model.parameters()
        )

    def _parameters_finite(self) -> bool:
        return all(torch.isfinite(parameter).all() for parameter in self.built.model.parameters())

    @staticmethod
    def _scheduler_step(scheduler: object | None, loss: float) -> None:
        if scheduler is None:
            return
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(loss)
        elif isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            # Old checkpoints and pre-v17 budget adaptation can carry a
            # shorter scheduler horizon than the optimizer phase. Never let
            # that metadata mismatch abort an otherwise valid candidate.
            if int(scheduler.last_epoch) >= int(scheduler.total_steps):
                return
            scheduler.step()
        else:
            scheduler.step()

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.adaptive_compute_budget_ledger.checkpoint_count += 1
        self.adaptive_compute_budget_ledger.optimizer_steps = max(
            self.adaptive_compute_budget_ledger.optimizer_steps,
            int(self._actual_iterations),
        )
        payload = self._runtime_state_payload(
            current_loss=(
                float(self.history[-1]["total_loss"])
                if self.history and self.history[-1].get("total_loss") is not None
                else None
            )
        )
        payload.update(
            {
                "checkpoint_schema_version": 5,
                "normalized_spec": self.built.normalized_spec,
                "time_strategy_audit": self.time_strategy_audit,
                "reference_mse_checkpoints": self.reference_mse_checkpoints,
                "last_checkpoint_iteration": self.last_checkpoint_iteration,
                "last_checkpoint_mse": self.last_checkpoint_mse,
                "adaptive_audit_logger_state": (
                    self.adaptive_audit_logger.state_dict()
                    if self.adaptive_audit_logger is not None
                    else None
                ),
            }
        )
        torch.save(payload, path)

    def load_checkpoint(self, path: str | Path) -> None:
        """Restore model, optimizer, scheduler, history, and phase progress."""

        device = next(self.built.model.parameters()).device
        checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
        if int(checkpoint.get("checkpoint_schema_version") or 0) >= 2:
            self._restore_runtime_state(checkpoint)
            self.time_strategy_audit = dict(
                checkpoint.get("time_strategy_audit") or {}
            )
            self.reference_mse_checkpoints = dict(
                checkpoint.get("reference_mse_checkpoints") or {}
            )
            self.last_checkpoint_iteration = checkpoint.get(
                "last_checkpoint_iteration"
            )
            self.last_checkpoint_mse = checkpoint.get("last_checkpoint_mse")
            if self.adaptive_audit_logger is not None:
                self.adaptive_audit_logger.load_state_dict(
                    checkpoint.get("adaptive_audit_logger_state")
                )
            if self.fixed_budget_sampling_controller is not None:
                versions = dict(
                    getattr(self.built.sampler, "_sampling_state_versions", {})
                    or {}
                )
                self._emit_adaptive_audit(
                    iteration=self._actual_iterations,
                    stage=str(
                        checkpoint.get("active_phase_name") or "resume"
                    ),
                    controller="fixed_budget_sampling",
                    event_type="sampling_resume_restored",
                    trigger={
                        "checkpoint_schema_version": int(
                            checkpoint.get("checkpoint_schema_version") or 0
                        ),
                        "sampler_state_versions": versions,
                        "pending_observation": bool(
                            self._pending_sampling_observation
                        ),
                    },
                )
            if self.curriculum_advance_controller is not None:
                self._emit_adaptive_audit(
                    iteration=self._actual_iterations,
                    stage=str(checkpoint.get("active_phase_name") or "resume"),
                    controller="curriculum_advance",
                    event_type="curriculum_resume_restored",
                    trigger={
                        "checkpoint_schema_version": int(
                            checkpoint.get("checkpoint_schema_version") or 0
                        ),
                        "resume_disabled": bool(self._curriculum_resume_disabled),
                        "pending_observation": bool(
                            self._pending_curriculum_observation
                        ),
                        "runtime_states": {
                            curriculum_id: runtime.get_current_state().to_dict()
                            for curriculum_id, runtime in self.curriculum_runtimes.items()
                        },
                    },
                )
            return
        self.built.model.load_state_dict(checkpoint["model_state_dict"])
        for phase, state in zip(self.built.optimization_phases, checkpoint.get("optimizer_states") or []):
            if state is not None:
                self._load_optimizer_state_if_compatible(phase.optimizer, state)
        for phase, state in zip(self.built.optimization_phases, checkpoint.get("scheduler_states") or []):
            if phase.scheduler is not None and state is not None:
                phase.scheduler.load_state_dict(state)
        self.history = list(checkpoint.get("history") or [])
        self.adaptive_refinement_events = list(checkpoint.get("adaptive_refinement_events") or [])
        self.optimizer_audit = dict(checkpoint.get("optimizer_audit") or {})
        self.component_gradient_audits = list(
            checkpoint.get("component_gradient_audits") or []
        )
        # Safe defaults retain compatibility with schema-v1 checkpoints.
        if checkpoint.get("loss_state"):
            self.built.loss_function.load_state_dict(checkpoint["loss_state"])
        if checkpoint.get("sampler_state"):
            self.built.sampler.load_state_dict(checkpoint["sampler_state"])
        stored = checkpoint.get("completed_phase_iterations") or {}
        if stored:
            self.completed_phase_iterations = {str(name): int(count) for name, count in stored.items()}
        else:
            counts: dict[str, int] = {}
            for item in self.history:
                phase_name = str(item.get("phase") or "")
                if phase_name:
                    counts[phase_name] = counts.get(phase_name, 0) + 1
            self.completed_phase_iterations = counts
