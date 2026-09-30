"""Reproducibility helpers for experiment runs."""

from __future__ import annotations

from datetime import datetime
import os
import platform
import random
import subprocess
import uuid

import numpy as np
import torch


def default_device(preferred: str | None = None) -> str:
    """Return the preferred device, defaulting to CUDA when available."""

    if preferred:
        return preferred
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_global_seed(seed: int) -> None:
    """Seed all RNGs and select deterministic or throughput-oriented kernels.

    Deterministic execution remains the default.  Long server sweeps may set
    ``PINNFORGE_PERFORMANCE_MODE=1`` to allow faster CUDA kernels and TF32
    matrix multiplications while retaining seeded RNG streams.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    performance_mode = os.environ.get(
        "PINNFORGE_PERFORMANCE_MODE", ""
    ).strip().lower() in {"1", "true", "yes", "on"}
    if performance_mode:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)
        if hasattr(torch.backends, "cuda"):
            torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        return

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def seed_worker(worker_id: int) -> None:
    """DataLoader worker seeding hook."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed + worker_id)
    random.seed(worker_seed + worker_id)


def get_environment_info(device: str | None = None) -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": device,
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "git_commit": get_git_commit(),
    }


def get_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip()
    except Exception:
        return None


def create_experiment_id(prefix: str = "exp") -> str:
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    return f"{prefix}_{timestamp}_{uuid.uuid4().hex[:8]}"
