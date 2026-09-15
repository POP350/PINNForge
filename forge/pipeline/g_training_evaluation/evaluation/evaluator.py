"""Problem-owned numerical evaluation for trained Open AlgorithmSpec models."""

from __future__ import annotations

import math
from typing import Any

from forge.pipeline.g_training_evaluation.evaluation.pinn_evaluator import PINNEvaluator


class OpenSpecEvaluator:
    def __init__(self, problem: Any, device: str) -> None:
        self.evaluator = PINNEvaluator(problem, device=device)

    def evaluate(self, model, training_report: Any | None = None) -> dict[str, Any]:
        report = training_report.to_dict() if hasattr(training_report, "to_dict") else dict(training_report or {})
        metrics = self.evaluator.evaluate(
            model,
            training_time=float(report.get("training_time") or 0.0),
            convergence_epoch=None,
        )
        result = {
            "mae": metrics.get("mae"),
            "merr": metrics.get("merr"),
            "mxe": metrics.get("mxe"),
            "l1re": metrics.get("l1re"),
            "l2re": metrics.get("l2re"),
            "crmse": metrics.get("crmse"),
            "fmse_low": metrics.get("fmse_low"),
            "fmse_mid": metrics.get("fmse_mid"),
            "fmse_high": metrics.get("fmse_high"),
            "frmse_low": metrics.get("frmse_low"),
            "frmse_mid": metrics.get("frmse_mid"),
            "frmse_high": metrics.get("frmse_high"),
            "temporal_l2re": metrics.get("temporal_l2re"),
            "temporal_mse": metrics.get("temporal_mse"),
            "temporal_reference_norm": metrics.get("temporal_reference_norm"),
            "temporal_prediction_norm": metrics.get("temporal_prediction_norm"),
            "temporal_amplitude_ratio": metrics.get("temporal_amplitude_ratio"),
            "initial_displacement_mse": metrics.get("initial_displacement_mse"),
            "initial_displacement_rmse": metrics.get("initial_displacement_rmse"),
            "initial_displacement_relative_l2": metrics.get(
                "initial_displacement_relative_l2"
            ),
            "initial_velocity_mse": metrics.get("initial_velocity_mse"),
            "initial_velocity_rmse": metrics.get("initial_velocity_rmse"),
            "initial_velocity_relative_l2": metrics.get(
                "initial_velocity_relative_l2"
            ),
            "initial_velocity_relative_l2_denominator_zero": metrics.get(
                "initial_velocity_relative_l2_denominator_zero"
            ),
            "prediction_amplitude": metrics.get("prediction_amplitude"),
            "reference_amplitude": metrics.get("reference_amplitude"),
            "amplitude_ratio": metrics.get("amplitude_ratio"),
            "amplitude_absolute_error": metrics.get("amplitude_absolute_error"),
            "temporal_phase_error": metrics.get("temporal_phase_error"),
            "mean_phase_error": metrics.get("mean_phase_error"),
            "maximum_phase_error": metrics.get("maximum_phase_error"),
            "phase_error_valid": metrics.get("phase_error_valid"),
            "relative_l1": metrics.get("relative_l1_error"),
            "relative_l1_error": metrics.get("relative_l1_error"),
            "relative_l2": metrics.get("relative_l2_error"),
            "relative_l2_error": metrics.get("relative_l2_error"),
            "mse": metrics.get("mse"),
            "prediction_mean": metrics.get("prediction_mean"),
            "prediction_std": metrics.get("prediction_std"),
            "reference_std": metrics.get("reference_std"),
            "pde_residual": metrics.get("pde_residual_error"),
            "pde_residual_error": metrics.get("pde_residual_error"),
            "governing_residual_distribution": metrics.get(
                "governing_residual_distribution"
            )
            or {},
            "residual_distribution": metrics.get("residual_distribution") or {},
            "residual_spatial_summary": metrics.get("residual_spatial_summary")
            or {
                "available": False,
                "reason": "legacy_or_unavailable_residual_spatial_summary",
                "equations": {},
            },
            "boundary_error": metrics.get("boundary_error"),
            "initial_error": metrics.get("initial_error"),
            "constraint_violation": metrics.get("constraint_violation"),
            "shock_region_error": metrics.get("shock_region_mse"),
            "shock_region_mse": metrics.get("shock_region_mse"),
            "shock_region_relative_l2_error": metrics.get("shock_region_relative_l2_error"),
            "shock_region_relative_linf_error": metrics.get("shock_region_relative_linf_error"),
            "shock_region_sample_count": metrics.get("shock_region_sample_count"),
            "shock_region_definition": metrics.get("shock_region_definition"),
            "training_time": metrics.get("training_time"),
            "parameter_count": metrics.get("parameter_count"),
            "peak_gpu_memory": max(
                int(metrics.get("peak_memory") or 0), int(report.get("peak_gpu_memory") or 0)
            ),
            "actual_peak_sampling_points": int(
                report.get("actual_peak_sampling_points") or 0
            ),
            "final_sampling_snapshot": report.get("final_sampling_snapshot") or {},
            "convergence_status": (
                "stopped_early" if report.get("stopped_early") else
                ("completed" if report.get("success") and not metrics.get("nan_detected") else "failed")
            ),
            "nan_detected": metrics.get("nan_detected"),
            "metric_groups": metrics.get("metric_groups") or {},
            "final_metrics": metrics.get("final_metrics") or {},
        }
        invalid = [
            key
            for key, value in result.items()
            if isinstance(value, float) and not math.isfinite(value)
        ]
        if metrics.get("nan_detected") or invalid:
            fields = ", ".join(sorted(invalid)) or "underlying evaluator metrics"
            raise ValueError(f"non_finite_evaluation_metrics: {fields}")
        return result

    def evaluate_mse(self, model) -> float | None:
        """Observe reference MSE without contributing it to backpropagation."""

        return self.evaluator.evaluate_mse(model)
