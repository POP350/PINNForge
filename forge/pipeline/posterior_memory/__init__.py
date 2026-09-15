"""Run-local posterior memory, deliberately separate from persistent priors."""

from .candidate_posterior_builder import (
    SUPPORTED_FAILURE_FLAGS,
    build_candidate_posterior,
    detect_failure_evidence,
    summarize_training_dynamics,
)
from .compression import compress_prompt_context, estimate_context_tokens
from .generation_reflection import build_generation_posterior
from .models import (
    ActivePosteriorRule,
    CandidatePosterior,
    FailureEvidence,
    GenerationPosterior,
    PosteriorPromptContext,
    TrainingDynamicsSummary,
)
from .run_scoped_memory import RunScopedPosteriorMemory, RunScopedPosteriorSession

__all__ = [
    "ActivePosteriorRule",
    "CandidatePosterior",
    "FailureEvidence",
    "GenerationPosterior",
    "PosteriorPromptContext",
    "RunScopedPosteriorMemory",
    "RunScopedPosteriorSession",
    "SUPPORTED_FAILURE_FLAGS",
    "TrainingDynamicsSummary",
    "build_candidate_posterior",
    "build_generation_posterior",
    "compress_prompt_context",
    "detect_failure_evidence",
    "estimate_context_tokens",
    "summarize_training_dynamics",
]

