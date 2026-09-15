"""Formal training for normalized Open AlgorithmSpec objects."""

from .trainer import OpenSpecTrainer, TrainingReport
from .adaptive_reporting import (
    AdaptiveComputeBudgetLedger,
    build_adaptive_control_summary,
)
from .registry import (
    CurriculumDescriptor,
    LossComponentDescriptor,
    OptimizationStageDescriptor,
    SamplingComponentDescriptor,
    TrainerCapabilities,
    TrainingComponentRegistry,
    VariableRoleDescriptor,
    derive_trainer_capabilities,
)
from .curriculum_runtime import (
    CurriculumRuntimeProtocol,
    CurriculumStateSnapshot,
    CurriculumTransitionPreview,
    CurriculumTransitionResult,
    deterministic_state_digest,
)
from .validation_probe import (
    PhysicsValidationResult,
    ValidationProbeBudgetPolicy,
    ValidationProbeManager,
    ValidationProbeSet,
)
from .best_train_loss_checkpoint import BestTrainLossCheckpointManager
from .sampling_runtime import (
    PointScoreResult,
    SamplingBatch,
    SamplingMutationResult,
    SamplingRuntimeProtocol,
)

__all__ = [
    "AdaptiveComputeBudgetLedger",
    "BestTrainLossCheckpointManager",
    "CurriculumDescriptor",
    "CurriculumRuntimeProtocol",
    "CurriculumStateSnapshot",
    "CurriculumTransitionPreview",
    "CurriculumTransitionResult",
    "LossComponentDescriptor",
    "OpenSpecTrainer",
    "OptimizationStageDescriptor",
    "PhysicsValidationResult",
    "PointScoreResult",
    "SamplingBatch",
    "SamplingMutationResult",
    "SamplingRuntimeProtocol",
    "SamplingComponentDescriptor",
    "TrainerCapabilities",
    "TrainingComponentRegistry",
    "TrainingReport",
    "ValidationProbeBudgetPolicy",
    "ValidationProbeManager",
    "ValidationProbeSet",
    "VariableRoleDescriptor",
    "derive_trainer_capabilities",
    "deterministic_state_digest",
    "build_adaptive_control_summary",
]
