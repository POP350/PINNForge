"""Model-only checkpoint selected by minimum summed training loss."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Any

import torch


class BestTrainLossCheckpointManager:
    """Track the PINNacle/DeepXDE best step without reference metrics."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = dict(config or {})
        self.enabled = bool(config.get("enabled", False))
        self.final_model_policy = str(config.get("final_model_policy", "last"))
        self.evaluation_interval = max(1, int(config.get("evaluation_interval", 100)))
        self.best_score: float | None = None
        self.best_model_state: dict[str, torch.Tensor] | None = None
        self.best_metadata: dict[str, Any] | None = None

    def consider(
        self,
        *,
        model: torch.nn.Module,
        training_loss: float,
        iteration: int,
        stage_id: str,
        algorithm_spec_id: str,
        directory: str | Path | None = None,
    ) -> bool:
        """Update on every strict decrease, matching DeepXDE ``update_best``."""

        selection_score = float(training_loss)
        if (
            not self.enabled
            or self.final_model_policy != "best_train_loss"
            or not math.isfinite(selection_score)
            or (self.best_score is not None and selection_score >= self.best_score)
        ):
            return False
        self.best_score = selection_score
        self.best_model_state = {
            name: value.detach().clone().cpu()
            for name, value in model.state_dict().items()
        }
        self.best_metadata = {
            "iteration": int(iteration),
            "stage": str(stage_id),
            "selection_metric": "training_loss",
            "selection_score": selection_score,
            "training_loss": selection_score,
            "algorithm_spec_id": str(algorithm_spec_id),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if directory is not None:
            path = Path(directory) / "best_train_loss_checkpoint.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": self.best_model_state,
                    "metadata": self.best_metadata,
                },
                path,
            )
        return True

    def restore_best_model(self, model: torch.nn.Module) -> bool:
        if self.best_model_state is None:
            return False
        device = next(model.parameters()).device
        model.load_state_dict(
            {name: value.to(device) for name, value in self.best_model_state.items()}
        )
        return True

    def state_dict(self) -> dict[str, Any]:
        return {
            "best_score": self.best_score,
            "best_model_state": deepcopy(self.best_model_state),
            "best_metadata": deepcopy(self.best_metadata),
        }

    def load_state_dict(self, state: dict[str, Any] | None) -> None:
        state = dict(state or {})
        score = state.get("best_score")
        self.best_score = float(score) if score is not None else None
        self.best_model_state = deepcopy(state.get("best_model_state"))
        self.best_metadata = deepcopy(state.get("best_metadata"))
