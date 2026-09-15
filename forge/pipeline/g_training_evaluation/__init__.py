"""Stage G: train PINNs and compute numerical metrics."""

from .evaluation import OpenSpecEvaluator
from .training import OpenSpecTrainer, TrainingReport

__all__ = ["OpenSpecEvaluator", "OpenSpecTrainer", "TrainingReport"]

