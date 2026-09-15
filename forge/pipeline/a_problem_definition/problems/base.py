"""Extensible physics-problem interface used by the PINN search pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Mapping


class PhysicsProblem(ABC):
    """Interface implemented by PINNacle problems and future problem providers."""

    problem_id: str
    name: str

    @abstractmethod
    def get_spec(self):
        """Return the structured PhysicsProblemSpec."""

    @abstractmethod
    def sample_domain(
        self,
        n: int,
        *,
        seed: int | None = None,
        device: str | None = None,
    ) -> Any:
        """Sample points from the state, temporal, spatial, or parameter domain."""

    @abstractmethod
    def sample_constraints(
        self,
        n: int,
        *,
        category: str | None = None,
        seed: int | None = None,
        device: str | None = None,
    ) -> Mapping[str, Any]:
        """Sample points required by physical constraints.

        ``category`` lets callers request only one aggregate constraint class,
        such as ``"boundary"`` or ``"initial"``. Providers that expose
        categories should avoid constructing unrelated constraint batches.
        """

    @abstractmethod
    def compute_governing_residuals(
        self,
        model: Any,
        samples: Any,
        *,
        create_graph: bool = True,
    ) -> Mapping[str, Any]:
        """Evaluate all governing-law residuals."""

    @abstractmethod
    def compute_constraint_residuals(
        self,
        model: Any,
        samples: Mapping[str, Any],
        *,
        create_graph: bool = True,
    ) -> Mapping[str, Any]:
        """Evaluate all physical-constraint residuals."""

    def get_observations(self) -> Any | None:
        """Return observations for inverse or data-assisted problems."""
        return None

    def sampling_contract(self) -> Mapping[str, Any]:
        """Return PDE-owned rules for allocating common sampling pools.

        The default contract shares each AlgorithmSpec role budget across the
        active unique coordinate sources.  Problems only need to override this
        through ``metadata.sampling_contract`` when a source has special
        semantics, such as a one-point pressure gauge.
        """

        return dict(
            (self.get_spec().metadata or {}).get("sampling_contract") or {}
        )

    def reference_solution(self, samples: Any) -> Any | None:
        """Return an exact/reference solution when available."""
        return None

    def reference_parameters(self) -> Mapping[str, float] | None:
        """Return true physical parameters when available."""
        return None

    def constraint_enforcement_capabilities(self) -> Mapping[str, Any]:
        """Return problem-owned constraint methods available to every candidate.

        Soft penalties are universally available. Providers may publish one or
        more differentiable hard output transforms without making them a hidden
        candidate-specific advantage.
        """
        return {
            "soft_penalty": {"transform_ids": ["none"]},
            "problem_hard": {"transform_ids": []},
        }

    def build_constraint_output_transform(
        self,
        transform_id: str,
        parameters: Mapping[str, Any] | None = None,
    ) -> Callable[[Any, Any], Any]:
        """Build a registered differentiable hard transform for this problem."""
        raise ValueError(
            f"Problem '{self.problem_id}' does not provide hard transform '{transform_id}'"
        )

    def evaluation_samples(
        self,
        *,
        device: str | None = None,
    ) -> Any:
        """Return a deterministic evaluation grid independent of training samples."""
        raise NotImplementedError

    def residual_evaluation_samples(
        self,
        *,
        device: str | None = None,
    ) -> Any:
        """Return a residual grid; defaults to the prediction evaluation grid."""
        return self.evaluation_samples(device=device)

    def residual_surface_metadata(self) -> Mapping[str, Any] | None:
        """Describe trusted coordinate surfaces available to residual diagnostics.

        Providers should return ``None`` when their geometry cannot reliably
        expose boundary or initial surfaces. Evaluators must not infer surfaces
        from coordinate names alone.
        """

        return None


PDEProblem = PhysicsProblem
