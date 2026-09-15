"""Checkpoint bookkeeping and post-action degradation detection."""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from typing import Any

import torch


class CheckpointRollbackManager:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = dict(config or {})
        self.enabled = bool(config.get("enabled", True))
        self.loss_degradation_ratio = float(config.get("loss_degradation_ratio", 2.0))
        self.patience = max(1, int(config.get("patience", 3)))
        self.active_checkpoint_id: str | None = None
        self.active_checkpoint_path: str | None = None
        self.baseline_loss: float | None = None
        self.degradation_count = 0
        self.failed_action_signatures: list[str] = []

    def save(self, checkpoint_id: str, payload: dict[str, Any], directory: str | Path | None) -> str | None:
        if not self.enabled:
            return None
        self.active_checkpoint_id = str(checkpoint_id)
        self.baseline_loss = (
            float(payload["current_loss"])
            if payload.get("current_loss") is not None
            else None
        )
        self.degradation_count = 0
        if directory is None:
            self.active_checkpoint_path = None
            payload["_in_memory_checkpoint"] = True
            self._memory_payload = deepcopy(payload)
            return None
        path = Path(directory) / "adaptive_control" / f"{checkpoint_id}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        self.active_checkpoint_path = str(path)
        self._memory_payload = None
        return str(path)

    def observe_loss(self, loss: float | None) -> bool:
        if not self.enabled or self.active_checkpoint_id is None:
            return False
        if loss is None or not math.isfinite(float(loss)):
            return True
        if self.baseline_loss is None or not math.isfinite(self.baseline_loss):
            return False
        degraded = float(loss) > self.baseline_loss * self.loss_degradation_ratio
        self.degradation_count = self.degradation_count + 1 if degraded else 0
        return self.degradation_count >= self.patience

    def load_payload(self, device: torch.device) -> dict[str, Any] | None:
        if self.active_checkpoint_path is not None:
            return torch.load(
                Path(self.active_checkpoint_path),
                map_location=device,
                weights_only=False,
            )
        payload = getattr(self, "_memory_payload", None)
        return deepcopy(payload) if payload is not None else None

    def mark_rolled_back(self, action_signature: str) -> None:
        self.failed_action_signatures.append(str(action_signature))
        self.active_checkpoint_id = None
        self.active_checkpoint_path = None
        self.baseline_loss = None
        self.degradation_count = 0
        self._memory_payload = None

    def clear(self) -> None:
        self.active_checkpoint_id = None
        self.active_checkpoint_path = None
        self.baseline_loss = None
        self.degradation_count = 0
        self._memory_payload = None

    def state_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "loss_degradation_ratio": self.loss_degradation_ratio,
            "patience": self.patience,
            "active_checkpoint_id": self.active_checkpoint_id,
            "active_checkpoint_path": self.active_checkpoint_path,
            "baseline_loss": self.baseline_loss,
            "degradation_count": self.degradation_count,
            "failed_action_signatures": list(self.failed_action_signatures),
            "memory_payload": (
                deepcopy(getattr(self, "_memory_payload", None))
                if self.active_checkpoint_path is None
                else None
            ),
        }

    def load_state_dict(self, state: dict[str, Any] | None) -> None:
        state = dict(state or {})
        self.active_checkpoint_id = state.get("active_checkpoint_id")
        self.active_checkpoint_path = state.get("active_checkpoint_path")
        baseline = state.get("baseline_loss")
        self.baseline_loss = float(baseline) if baseline is not None else None
        self.degradation_count = int(state.get("degradation_count") or 0)
        self.failed_action_signatures = list(
            state.get("failed_action_signatures") or []
        )
        self._memory_payload = deepcopy(state.get("memory_payload"))
