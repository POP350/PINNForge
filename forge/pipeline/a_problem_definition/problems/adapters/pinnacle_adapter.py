"""Executable bridge to the PDE definitions bundled with PINNacle.

The vendored PINNacle implementation owns the trusted equations, geometries,
boundary conditions, and reference data.  This module only adapts those
objects to :class:`PhysicsProblem`; it does not reimplement the PDEs.
"""

from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
import importlib
import inspect
from itertools import product
import os
from pathlib import Path
import sys
import threading
import types
from typing import Any, Callable, Mapping
import warnings

import numpy as np
import torch

from forge.pipeline.c_knowledge_retrieval.pinnacle_registry import get_pinnacle_profile
from forge.pipeline.a_problem_definition.problems.base import PhysicsProblem
from forge.pipeline.a_problem_definition.problems.schemas import (
    ConstraintSpec,
    DomainSpec,
    GoverningLawSpec,
    ObservationSpec,
    PhysicsProblemSpec,
)


_WORKSPACE_ROOT = Path(__file__).resolve().parents[5]
PINNACLE_ROOT = _WORKSPACE_ROOT / "third_party" / "pinnacle"
_LOAD_LOCK = threading.RLock()
DEFAULT_RESIDUAL_EVALUATION_SIZE = 8192

_PINNACLE_CLASS_MODULES: dict[str, str] = {
    "Burgers1D": "src.pde.burgers",
    "Burgers2D": "src.pde.burgers",
    "GrayScottEquation": "src.pde.chaotic",
    "KuramotoSivashinskyEquation": "src.pde.chaotic",
    "Heat2D_VaryingCoef": "src.pde.heat",
    "Heat2D_Multiscale": "src.pde.heat",
    "Heat2D_ComplexGeometry": "src.pde.heat",
    "Heat2D_LongTime": "src.pde.heat",
    "HeatND": "src.pde.heat",
    "PoissonInv": "src.pde.inverse",
    "HeatInv": "src.pde.inverse",
    "NS2D_LidDriven": "src.pde.ns",
    "NS2D_BackStep": "src.pde.ns",
    "NS2D_LongTime": "src.pde.ns",
    "Poisson2D_Classic": "src.pde.poisson",
    "PoissonBoltzmann2D": "src.pde.poisson",
    "Poisson3D_ComplexGeometry": "src.pde.poisson",
    "Poisson2D_ManyArea": "src.pde.poisson",
    "PoissonND": "src.pde.poisson",
    "Wave1D": "src.pde.wave",
    "Wave2D_Heterogeneous": "src.pde.wave",
    "Wave2D_LongTime": "src.pde.wave",
}


@dataclass(frozen=True)
class PINNacleConstraintBatch:
    """Points and metadata required to evaluate one DeepXDE constraint."""

    name: str
    category: str
    bc: Any
    points: torch.Tensor
    numpy_points: np.ndarray
    pointset_values: torch.Tensor | None = None
    pointset_component: int | list[int] | None = None
    location: str | None = None
    derivative_variable: str | None = None
    derivative_order: int = 0
    paired_points: torch.Tensor | None = None
    normal_axes: torch.Tensor | None = None
    coefficient_left: torch.Tensor | None = None
    coefficient_right: torch.Tensor | None = None


def _burgers_1d_initial_dirichlet_exact(
    inputs: torch.Tensor, raw_outputs: torch.Tensor
) -> torch.Tensor:
    """Exact IC/BC ansatz for PINNacle's canonical Burgers1D contract."""
    x = inputs[:, 0:1]
    t = inputs[:, 1:2]
    initial = -torch.sin(torch.pi * x)
    return initial + t * (1.0 - x.square()) * raw_outputs


_HARD_CONSTRAINT_TRANSFORMS: dict[
    str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
] = {
    "burgers_1d_initial_dirichlet_exact": _burgers_1d_initial_dirichlet_exact,
}


def load_pinnacle_class(class_name: str) -> type:
    """Load one vendored PINNacle class without importing its training CLI."""

    if class_name not in _PINNACLE_CLASS_MODULES:
        known = ", ".join(sorted(_PINNACLE_CLASS_MODULES))
        raise KeyError(f"Unknown PINNacle PDE class '{class_name}'. Known: {known}")
    if not PINNACLE_ROOT.is_dir():
        raise FileNotFoundError(f"Vendored PINNacle source not found: {PINNACLE_ROOT}")

    with _LOAD_LOCK:
        _prepare_pinnacle_imports()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"torch\.set_default_tensor_type\(\) is deprecated.*",
                category=UserWarning,
            )
            module = importlib.import_module(_PINNACLE_CLASS_MODULES[class_name])
        _restore_torch_defaults()
        _patch_reference_readers()
        return getattr(module, class_name)


def instantiate_pinnacle_problem(class_name: str) -> Any:
    """Instantiate a PINNacle PDE while its relative data paths are valid."""

    problem_class = load_pinnacle_class(class_name)
    with _LOAD_LOCK, _working_directory(PINNACLE_ROOT):
        return problem_class()


class PINNacleProblemAdapter(PhysicsProblem):
    """Expose a PINNacle PDE through the project's generic problem API."""

    def __init__(
        self,
        problem_id: str,
        *,
        pinnacle_problem: Any | None = None,
        evaluation_size: int | None = None,
        residual_evaluation_size: int = DEFAULT_RESIDUAL_EVALUATION_SIZE,
    ) -> None:
        profile = get_pinnacle_profile(problem_id)
        if profile is None:
            raise KeyError(f"'{problem_id}' is not present in the PINNacle registry")
        self.problem_id = _canonical_problem_id(problem_id)
        self.profile = profile
        self.name = profile["pinnacle_name"]
        self.pinnacle_problem = pinnacle_problem or instantiate_pinnacle_problem(self.name)
        _disable_dynamic_batch_range_caches(self.pinnacle_problem)
        self.geometry = getattr(self.pinnacle_problem, "geomtime", None) or getattr(self.pinnacle_problem, "geom", None)
        if self.geometry is None:
            raise ValueError(f"PINNacle problem '{self.name}' did not define a geometry")
        # Normalize the public ``None = full grid`` request at the adapter
        # boundary.  Keeping a literal None in the runtime object is fragile:
        # downstream sampling code naturally compares integer point counts and
        # an optional cap, which previously produced ``int < None`` failures.
        self._evaluation_size = 0
        self._evaluation_scope = ""
        self.evaluation_size = evaluation_size
        self._residual_evaluation_size = 0
        self._residual_evaluation_inputs: np.ndarray | None = None
        self.residual_evaluation_size = residual_evaluation_size
        self._reference_tree: Any | None = None
        self._reference_inputs: np.ndarray | None = None
        self._reference_outputs: np.ndarray | None = None
        self._constraint_descriptors = self._describe_constraints()
        self._initial_target_evaluators = self._build_initial_target_evaluators()

    def get_spec(self) -> PhysicsProblemSpec:
        input_variables = _input_variable_names(self.profile, int(self.pinnacle_problem.input_dim))
        output_variables = _output_variable_names(self.pinnacle_problem)
        bounds = _bounds_from_bbox(getattr(self.pinnacle_problem, "bbox", None), input_variables)
        governing_config = list(getattr(self.pinnacle_problem, "loss_config", []))[: int(self.pinnacle_problem.num_pde)]
        governing_laws = [
            GoverningLawSpec(
                law_id=_safe_name(config.get("name") or f"equation_{index}"),
                name=str(config.get("name") or f"Equation {index}"),
                law_type="pde",
                variables=input_variables + output_variables,
                parameters=sorted(self.profile.get("physical_parameters", {})),
                differential_order=int(self.profile["differential_order"]),
                metadata={"pinnacle_equation_index": index},
            )
            for index, config in enumerate(governing_config)
        ]
        category_counts = {
            category: sum(
                descriptor["category"] == category
                for descriptor in self._constraint_descriptors
            )
            for category in {item["category"] for item in self._constraint_descriptors}
        }
        constraints = [
            ConstraintSpec(
                constraint_id=descriptor["name"],
                name=descriptor["display_name"],
                constraint_type=descriptor["schema_type"],
                target=descriptor["category"],
                required=True,
                derivative_variable=descriptor.get("derivative_variable"),
                derivative_order=(
                    int(descriptor["derivative_order"])
                    if descriptor.get("derivative_order")
                    else None
                ),
                metadata={
                    "pinnacle_bc_index": descriptor["index"],
                    "pinnacle_bc_type": descriptor["bc_type"],
                    # Concrete PINNacle BCs usually draw independent
                    # coordinates. Wave displacement and velocity are the
                    # deliberate exception and share one initial-time batch.
                    "sampling_category": (
                        "initial"
                        if self.problem_id == "wave_1d"
                        and descriptor["category"] == "initial"
                        else descriptor["category"]
                        if self.problem_id != "wave_1d"
                        and category_counts[descriptor["category"]] == 1
                        else descriptor["name"]
                    ),
                    "shared_coordinate_group": (
                        "wave_initial"
                        if self.problem_id == "wave_1d"
                        and descriptor["category"] == "initial"
                        else descriptor["category"]
                        if self.problem_id != "wave_1d"
                        and category_counts[descriptor["category"]] == 1
                        else descriptor["name"]
                    ),
                    "sampling_point_multiplier": (
                        2 if descriptor["bc_type"] == "periodic" else 1
                    ),
                    "location": descriptor.get("location"),
                    "derivative_variable": descriptor.get("derivative_variable"),
                    "derivative_order": int(descriptor.get("derivative_order") or 0),
                },
            )
            for descriptor in self._constraint_descriptors
        ]
        categories = {descriptor["category"] for descriptor in self._constraint_descriptors}
        aggregate_constraints = [
            ConstraintSpec(
                constraint_id=category,
                name=f"Aggregate {category} constraints",
                constraint_type=category,
                target=category,
                required=False,
                metadata={"aggregate": True},
            )
            for category in ("boundary", "initial")
            if category in categories
        ]
        constraints = aggregate_constraints + constraints
        has_pointset = any(item["bc_type"] == "pointset" for item in self._constraint_descriptors)
        return PhysicsProblemSpec(
            problem_id=self.problem_id,
            name=self.name,
            law_types=["pde"],
            task_type=self.profile["task_type"],
            input_variables=input_variables,
            output_variables=output_variables,
            domain=DomainSpec(
                variables=input_variables,
                bounds=bounds,
                geometry=self.profile["geometry"],
                metadata={"source": "PINNacle", "pinnacle_class": self.name},
            ),
            governing_laws=governing_laws,
            constraints=constraints,
            observations=ObservationSpec(
                available=has_pointset,
                variables=output_variables,
                noise_level=self.profile.get("physical_parameters", {}).get("noise_std"),
                sparse=has_pointset,
                metadata={"source": "PINNacle PointSetBC"} if has_pointset else {},
            ),
            metadata={
                **self.get_problem_features(),
                **_problem_specific_training_and_metric_metadata(
                    self.problem_id
                ),
                "pinnacle_profile": self.profile,
                "pinnacle_source_root": str(PINNACLE_ROOT),
                "pinnacle_class": self.name,
                "num_pde": int(self.pinnacle_problem.num_pde),
                "num_boundary": len(self._constraint_descriptors),
                "sampling_contract": {
                    "version": "1.0",
                    "allocation_policy": "shared_role_budget",
                    "source_rules": (
                        {
                            # The legacy aggregate evaluates one spatial and
                            # one shared initial coordinate set per request.
                            "boundary_condition": {"point_multiplier": 2}
                        }
                        if self.problem_id == "wave_1d"
                        else {}
                    ),
                },
            },
        )

    def get_problem_features(self) -> dict[str, Any]:
        tags = set(self.profile.get("challenge_tags", []))
        constraint_types = list(self.profile.get("constraint_types", []))
        family = str(self.profile["family"])
        return {
            "benchmark_id": self.problem_id,
            "problem_id": self.problem_id,
            "pde_type": family,
            "pde_family": family,
            "equation_order": self.profile["differential_order"],
            "spatial_dimension": self.profile["spatial_dimension"],
            "state_dimension": self.profile["output_dimension"],
            "time_dependent": self.profile["time_dependent"],
            "has_initial_condition": "initial" in constraint_types or any(
                item["category"] == "initial" for item in self._constraint_descriptors
            ),
            "boundary_type": constraint_types,
            "has_shock": "shock" in tags,
            "has_multiscale": "multiscale" in tags,
            "has_conservation_law": "conservation" in tags,
            "has_periodicity": "periodic" in tags,
            "is_stiff": "stiff" in tags,
            "is_multiscale": "multiscale" in tags,
            "has_complex_geometry": "complex_geometry" in tags,
            "has_interface": "interface" in tags,
            "has_piecewise_coefficient": "piecewise_coefficient" in tags,
            "has_varying_coefficient": bool(
                tags.intersection({"varying_coefficient", "piecewise_coefficient", "heterogeneous"})
            ),
            "has_oscillatory_solution": "oscillatory" in tags,
            "has_chaotic_dynamics": "chaotic" in tags,
            "has_long_time_horizon": "long_time" in tags,
            "has_incompressibility": "incompressible" in tags,
            "has_coupled_fields": self.profile["output_dimension"] > 1,
            "has_noisy_observations": "noisy_observations" in tags,
            "has_unknown_parameters": self.profile["task_type"] == "inverse",
            "is_elliptic": "elliptic" in tags,
            "elliptic": "elliptic" in tags,
            "diffusion_dominated": family == "heat",
            "convection_dominated": family == "burgers",
            "has_wave_propagation": family == "wave",
            "forward_or_inverse": self.profile["task_type"],
            "task_type": self.profile["task_type"],
            "noise_level": self.profile.get("physical_parameters", {}).get("noise_std", 0.0),
            "geometry": self.profile["geometry"],
            "law_types": ["pde"],
            "constraint_types": sorted({item["category"] for item in self._constraint_descriptors}),
            "data_available": getattr(self.pinnacle_problem, "ref_data", None) is not None,
            "challenge_tags": sorted(tags),
        }

    def training_governing_residuals(
        self,
        samples: torch.Tensor,
        residuals: Mapping[str, torch.Tensor],
    ) -> Mapping[str, torch.Tensor]:
        """Scale optimization residuals without changing the governing PDE."""

        if self.problem_id != "poisson_2d_many_area":
            return residuals
        _, coefficient = self._many_area_region_ids_and_coefficients(samples)
        scale = coefficient.abs().clamp_min(1.0e-6)
        return {
            str(name): value / scale.to(device=value.device, dtype=value.dtype)
            for name, value in residuals.items()
        }

    def training_residual_region_ids(
        self, samples: torch.Tensor
    ) -> torch.Tensor | None:
        """Return stable 0-based ManyArea cell IDs for loss and sampling."""

        if self.problem_id != "poisson_2d_many_area":
            return None
        region_ids, _ = self._many_area_region_ids_and_coefficients(samples)
        return region_ids

    def training_residual_region_metadata(self) -> Mapping[str, Any] | None:
        if self.problem_id != "poisson_2d_many_area":
            return None
        split = tuple(
            int(value)
            for value in getattr(self.pinnacle_problem, "split", (5, 5))
        )
        coefficients = torch.as_tensor(
            np.asarray(self.pinnacle_problem.a_cof), dtype=torch.float64
        )
        return {
            "strategy": "piecewise_coefficient_normalized_equal_region",
            "split": list(split),
            "region_count": int(split[0] * split[1]),
            "coefficient_min": float(coefficients.min()),
            "coefficient_max": float(coefficients.max()),
            "coefficient_contrast": float(
                coefficients.max() / coefficients.min().clamp_min(1.0e-12)
            ),
            "training_residual": "raw_residual_divided_by_local_a",
            "raw_physics_residual_preserved_for_evaluation": True,
        }

    def balance_training_domain_samples(
        self, samples: torch.Tensor
    ) -> torch.Tensor:
        """Keep the fixed ManyArea interior budget evenly split over 25 cells."""

        if self.problem_id != "poisson_2d_many_area" or samples.shape[0] == 0:
            return samples
        region_ids, _ = self._many_area_region_ids_and_coefficients(samples)
        split = tuple(
            int(value)
            for value in getattr(self.pinnacle_problem, "split", (5, 5))
        )
        bbox = tuple(float(value) for value in self.pinnacle_problem.bbox)
        region_count = int(split[0] * split[1])
        total = int(samples.shape[0])
        base_quota, remainder = divmod(total, region_count)
        cell_width = (bbox[1] - bbox[0]) / split[0]
        cell_height = (bbox[3] - bbox[2]) / split[1]
        chunks: list[torch.Tensor] = []
        for region in range(region_count):
            quota = base_quota + (1 if region < remainder else 0)
            existing = samples[region_ids == region][:quota]
            shortage = quota - int(existing.shape[0])
            if shortage > 0:
                cell_x = region // split[1]
                cell_y = region % split[1]
                unit = torch.rand(
                    (shortage, samples.shape[1]),
                    dtype=samples.dtype,
                    device=samples.device,
                )
                generated = unit.clone()
                generated[:, 0] = (
                    bbox[0] + cell_x * cell_width + unit[:, 0] * cell_width
                )
                generated[:, 1] = (
                    bbox[2] + cell_y * cell_height + unit[:, 1] * cell_height
                )
                existing = torch.cat((existing, generated), dim=0)
            chunks.append(existing)
        balanced = torch.cat(chunks, dim=0)
        order = torch.randperm(
            balanced.shape[0], device=balanced.device
        )
        return balanced[order]

    def _many_area_region_ids_and_coefficients(
        self, samples: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        split = tuple(
            int(value)
            for value in getattr(self.pinnacle_problem, "split", (5, 5))
        )
        bbox = tuple(float(value) for value in self.pinnacle_problem.bbox)
        coordinates = samples.detach()
        lower = coordinates.new_tensor((bbox[0], bbox[2]))
        block_size = coordinates.new_tensor(
            (
                (bbox[1] - bbox[0] + 2.0e-5) / split[0],
                (bbox[3] - bbox[2] + 2.0e-5) / split[1],
            )
        )
        domain = torch.floor(
            (coordinates[:, :2] - lower + 1.0e-5) / block_size
        ).to(dtype=torch.long)
        domain_x = domain[:, 0].clamp(0, split[0] - 1)
        domain_y = domain[:, 1].clamp(0, split[1] - 1)
        region_ids = domain_x * split[1] + domain_y
        table = torch.as_tensor(
            np.asarray(self.pinnacle_problem.a_cof),
            dtype=coordinates.dtype,
            device=coordinates.device,
        )
        coefficient = table[domain_x, domain_y].reshape(-1, 1)
        return region_ids, coefficient

    def _many_area_coefficient_and_forcing(
        self, samples: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the known heterogeneous operator fields on-device."""

        problem = self.pinnacle_problem
        split = tuple(int(value) for value in getattr(problem, "split", (5, 5)))
        bbox = tuple(float(value) for value in problem.bbox)
        frequency_count = int(getattr(problem, "freq", 2))
        coordinates = samples.detach()
        lower = coordinates.new_tensor((bbox[0], bbox[2]))
        block_size = coordinates.new_tensor(
            (
                (bbox[1] - bbox[0] + 2.0e-5) / split[0],
                (bbox[3] - bbox[2] + 2.0e-5) / split[1],
            )
        )
        reduced = coordinates[:, :2] - lower + 1.0e-5
        domain = torch.floor(reduced / block_size).to(dtype=torch.long)
        domain_x = domain[:, 0].clamp(0, split[0] - 1)
        domain_y = domain[:, 1].clamp(0, split[1] - 1)
        local = reduced - domain.to(dtype=coordinates.dtype) * block_size
        coefficient_table = torch.as_tensor(
            np.asarray(problem.a_cof),
            device=coordinates.device,
            dtype=coordinates.dtype,
        )
        forcing_table = torch.as_tensor(
            np.asarray(problem.f_cof),
            device=coordinates.device,
            dtype=coordinates.dtype,
        )
        frequencies = torch.arange(
            frequency_count,
            device=coordinates.device,
            dtype=coordinates.dtype,
        )
        sine_x = torch.sin(
            torch.pi * local[:, 0:1] / block_size[0] * frequencies
        )
        sine_y = torch.sin(
            torch.pi * local[:, 1:2] / block_size[1] * frequencies
        )
        selected_forcing = forcing_table[domain_x, domain_y]
        forcing = selected_forcing[:, 0, 0] + torch.sum(
            selected_forcing
            * sine_x.unsqueeze(2)
            * sine_y.unsqueeze(1),
            dim=(1, 2),
        )
        coefficient = coefficient_table[domain_x, domain_y]
        return coefficient.reshape(-1), forcing.reshape(-1)

    def physics_validation_scores(
        self, model: Any, batch: Any
    ) -> Mapping[str, float]:
        """Return problem-derived, label-free validation scores when available.

        For ManyArea, the symmetric elliptic operator and its Robin condition
        define a variational energy. It is less sensitive to high-frequency
        strong-residual noise than pointwise residual-only model selection.
        """

        if self.problem_id != "poisson_2d_many_area":
            return {}
        domain = batch.domain_samples.detach().clone().requires_grad_(True)
        values = model(domain)[:, 0]
        gradient = torch.autograd.grad(
            outputs=values,
            inputs=domain,
            grad_outputs=torch.ones_like(values),
            create_graph=False,
            retain_graph=False,
            allow_unused=False,
        )[0]
        coefficient, forcing = self._many_area_coefficient_and_forcing(domain)
        bbox = tuple(float(value) for value in self.pinnacle_problem.bbox)
        area = (bbox[1] - bbox[0]) * (bbox[3] - bbox[2])
        interior_energy = float(area) * torch.mean(
            0.5 * coefficient * gradient[:, :2].square().sum(dim=1)
            - forcing * values
        )
        boundary_chunks = [
            item.points
            for name, item in batch.constraint_samples.items()
            if name != "interface_flux_continuity"
            and str(getattr(item, "category", "")) == "boundary"
        ]
        boundary_energy = domain.new_zeros(())
        if boundary_chunks:
            boundary = torch.cat(boundary_chunks, dim=0).detach()
            boundary_values = model(boundary)[:, 0]
            boundary_coefficient, _ = self._many_area_coefficient_and_forcing(
                boundary
            )
            perimeter = 2.0 * (
                (bbox[1] - bbox[0]) + (bbox[3] - bbox[2])
            )
            boundary_energy = float(perimeter) * torch.mean(
                0.5 * boundary_coefficient * boundary_values.square()
            )
        energy = interior_energy + boundary_energy
        return {"variational_energy": float(energy.detach().cpu())}

    def physics_validation_score_names(self) -> tuple[str, ...]:
        if self.problem_id == "poisson_2d_many_area":
            return ("variational_energy",)
        return ()

    def constraint_enforcement_capabilities(self) -> Mapping[str, Any]:
        transform_ids = list(self.profile.get("hard_constraint_transforms") or [])
        return {
            "soft_penalty": {"transform_ids": ["none"]},
            "problem_hard": {"transform_ids": transform_ids},
        }

    def residual_surface_metadata(self) -> Mapping[str, Any] | None:
        """Expose only surfaces that the PINNacle adapter can identify safely."""

        spec = self.get_spec()
        coordinate_names = list(spec.input_variables)
        spatial_dimension = int(self.profile.get("spatial_dimension") or 0)
        spatial_axes = coordinate_names[:spatial_dimension]
        constraint_types = {
            str(item.constraint_type) for item in spec.constraints
        }
        geometry = str(self.profile.get("geometry") or "")
        axis_aligned_geometries = {
            "interval",
            "interval_x_time",
            "periodic_interval_x_time",
            "rectangle_xt",
            "rectangle_xy_time",
            "box_xyt",
            "unit_square",
            "five_dimensional_hypercube",
        }
        time_variable = (
            "t"
            if bool(self.profile.get("time_dependent"))
            and "t" in coordinate_names
            else None
        )
        return {
            "source": "pinnacle_adapter",
            "coordinate_names": coordinate_names,
            "coordinate_bounds": {
                name: list(spec.domain.bounds[name])
                for name in coordinate_names
                if name in spec.domain.bounds
            },
            "spatial_boundary_axes": (
                spatial_axes
                if "boundary" in constraint_types
                and geometry in axis_aligned_geometries
                else None
            ),
            "boundary_available": bool(
                "boundary" in constraint_types
                and geometry in axis_aligned_geometries
            ),
            "time_variable": time_variable,
            "initial_surface_available": bool(
                time_variable is not None and "initial" in constraint_types
            ),
            "geometry": geometry,
        }

    def build_constraint_output_transform(
        self,
        transform_id: str,
        parameters: Mapping[str, Any] | None = None,
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        parameters = dict(parameters or {})
        if parameters:
            raise ValueError(
                f"Hard transform '{transform_id}' accepts no parameters; received {sorted(parameters)}"
            )
        available = set(self.profile.get("hard_constraint_transforms") or [])
        if transform_id not in available:
            raise ValueError(
                f"Hard transform '{transform_id}' is not registered for problem '{self.problem_id}'"
            )
        transform = _HARD_CONSTRAINT_TRANSFORMS.get(transform_id)
        if transform is None:
            raise ValueError(f"Hard transform '{transform_id}' has no executable implementation")
        return transform

    def sample_domain(
        self,
        n: int,
        *,
        seed: int | None = None,
        device: str | None = None,
    ) -> torch.Tensor:
        with _numpy_seed(seed):
            points = self.geometry.random_points(max(1, int(n)), random="pseudo")
        return _as_tensor(points, device=device)

    def sample_constraints(
        self,
        n: int,
        *,
        category: str | None = None,
        seed: int | None = None,
        device: str | None = None,
    ) -> Mapping[str, PINNacleConstraintBatch]:
        count = max(1, int(n))
        requested_category = None if category is None else str(category).strip().lower()
        known_categories = {
            descriptor["category"] for descriptor in self._constraint_descriptors
        }
        known_sources = {
            descriptor["name"] for descriptor in self._constraint_descriptors
        }
        aggregate_categories = {"boundary", "initial"}
        if requested_category is not None and requested_category not in (
            aggregate_categories | known_sources
        ):
            raise ValueError(
                f"Unknown constraint category '{category}' for {self.problem_id}; "
                "expected an aggregate category or concrete constraint_id"
            )
        if (
            requested_category in aggregate_categories
            and requested_category not in known_categories
        ):
            return {}
        batches: dict[str, PINNacleConstraintBatch] = {}
        shared_initial_points: np.ndarray | None = None
        with _numpy_seed(seed):
            for descriptor in self._constraint_descriptors:
                if (
                    requested_category is not None
                    and (
                        descriptor["category"] != requested_category
                        if requested_category in known_categories
                        else descriptor["name"] != requested_category
                    )
                ):
                    continue
                bc = descriptor["bc"]
                if descriptor["bc_type"] == "interface_flux":
                    batches[descriptor["name"]] = (
                        self._sample_many_area_interface_flux(
                            count,
                            device=device,
                        )
                    )
                    continue
                if descriptor["bc_type"] == "pointset":
                    batches[descriptor["name"]] = self._sample_pointset(
                        descriptor, count, device=device
                    )
                    continue
                if (
                    self.problem_id in {"wave_1d", "wave_2d_long_time"}
                    and descriptor["category"] == "initial"
                    and shared_initial_points is not None
                ):
                    points = shared_initial_points.copy()
                else:
                    points = self._sample_bc_points(bc, descriptor["category"], count)
                    if (
                        self.problem_id in {"wave_1d", "wave_2d_long_time"}
                        and descriptor["category"] == "initial"
                    ):
                        shared_initial_points = np.asarray(points, dtype=np.float32).copy()
                pointset_values = None
                pointset_component = None
                if descriptor["category"] == "initial":
                    if descriptor["name"] == "initial_velocity" and self.problem_id in {
                        "wave_1d",
                        "wave_2d_long_time",
                    }:
                        pointset_values = torch.zeros(
                            (len(points), 1),
                            dtype=torch.float32,
                            device=device or "cpu",
                        )
                    elif self.problem_id == "wave_1d":
                        target = (
                            self.initial_displacement(points)
                        )
                        pointset_values = _as_tensor(target, device=device)
                    else:
                        pointset_values = self._initial_target_values(
                            descriptor,
                            points,
                            device=device,
                        )
                    pointset_component = getattr(bc, "component", None)
                batches[descriptor["name"]] = PINNacleConstraintBatch(
                    name=descriptor["name"],
                    category=descriptor["category"],
                    bc=bc,
                    points=_as_tensor(points, device=device),
                    numpy_points=np.asarray(points, dtype=np.float32),
                    pointset_values=pointset_values,
                    pointset_component=pointset_component,
                    location=descriptor.get("location"),
                    derivative_variable=descriptor.get("derivative_variable"),
                    derivative_order=int(descriptor.get("derivative_order") or 0),
                )
        return batches

    def augment_training_algorithm_spec(
        self, spec: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Inject physics constraints required by the executable benchmark."""

        result = deepcopy(dict(spec))
        if self.problem_id != "poisson_2d_many_area":
            return result
        loss = result.setdefault("loss", {})
        terms = loss.setdefault("terms", [])
        boundary_weights = [
            float(term.get("weight", 1.0))
            for term in terms
            if isinstance(term, Mapping)
            and str(term.get("name")) != "pde_residual"
            and float(term.get("weight", 1.0)) > 0.0
        ]
        interface_weight = (
            float(np.median(boundary_weights))
            if boundary_weights
            else 1.0
        )
        if not any(
            str(term.get("name")) == "interface_flux_continuity"
            for term in terms
            if isinstance(term, Mapping)
        ):
            terms.append(
                {
                    "name": "interface_flux_continuity",
                    "loss_function": "mse",
                    "weight": interface_weight,
                    "parameters": {},
                }
            )
        training = result.setdefault("training", {})
        extra = training.setdefault("extra_parameters", {})
        physics = extra.get("physics_validation")
        if isinstance(physics, dict) and physics.get("enabled"):
            physics.setdefault("minimum_component_scale", 1.0)
            physics.setdefault(
                "region_normalization", "shared_pde_rmse"
            )
            physics.setdefault(
                "second_order_degradation_ratio", 1.1
            )
            physics["points_per_component"] = max(
                1024,
                int(physics.get("points_per_component") or 0),
            )
            physics["maximum_total_points"] = max(
                4096,
                int(physics.get("maximum_total_points") or 0),
            )
            physics["second_order_guard_metric"] = "variational_energy"
        return result

    def runtime_physics_requirements(self) -> Mapping[str, Any]:
        if self.problem_id != "poisson_2d_many_area":
            return {}
        return {
            "required_loss_terms": ["interface_flux_continuity"],
            "interface_law": "continuous_a_times_normal_derivative",
            "interface_segment_count": 40,
            "training_only_reference_values_used": False,
            "checkpoint_selection_metric": "variational_energy",
            "second_order_guard_metric": "variational_energy",
            "source_evidence": (
                "COMSOL field derivative audit and piecewise-coefficient "
                "elliptic interface law"
            ),
        }

    def constraint_breakdown(
        self,
        samples: Mapping[str, PINNacleConstraintBatch] | None = None,
    ) -> list[dict[str, Any]]:
        """Return stable, auditable metadata for every concrete constraint."""

        sample_counts = {
            name: int(batch.points.shape[0])
            for name, batch in (samples or {}).items()
        }
        return [
            {
                "constraint_id": str(descriptor["name"]),
                "constraint_type": str(descriptor["schema_type"]),
                "location": descriptor.get("location"),
                "sample_count": int(sample_counts.get(str(descriptor["name"]), 0)),
                "derivative_variable": descriptor.get("derivative_variable"),
                "derivative_order": int(descriptor.get("derivative_order") or 0),
            }
            for descriptor in self._constraint_descriptors
        ]

    @property
    def time_index(self) -> int | None:
        variables = list(self.get_spec().input_variables)
        return variables.index("t") if "t" in variables else None

    def initial_displacement(self, coordinates: Any) -> Any:
        """Trusted registered wave displacement target at the initial time."""

        if self.problem_id not in {"wave_1d", "wave_2d_long_time"}:
            raise AttributeError(
                "initial_displacement is only defined for registered wave initial states"
            )
        is_tensor = isinstance(coordinates, torch.Tensor)
        array = (
            coordinates.detach().cpu().numpy()
            if is_tensor
            else np.asarray(coordinates, dtype=np.float32)
        )
        if array.ndim != 2:
            raise ValueError("Initial coordinates must be a rank-2 array")
        input_dimension = len(self.get_spec().input_variables)
        if array.shape[1] == input_dimension - 1:
            time_lower = float(self.get_spec().domain.bounds["t"][0])
            array = np.column_stack(
                [array, np.full(array.shape[0], time_lower, dtype=np.float32)]
            )
        elif array.shape[1] != input_dimension:
            raise ValueError(
                f"Expected {input_dimension - 1} spatial columns or "
                f"{input_dimension} space-time columns"
            )
        values = np.asarray(self.pinnacle_problem.ref_sol(array), dtype=np.float32)
        if is_tensor:
            return torch.as_tensor(
                values, dtype=coordinates.dtype, device=coordinates.device
            )
        return values

    def initial_velocity(self, coordinates: Any) -> Any:
        """Trusted registered zero initial velocity target."""

        if self.problem_id not in {"wave_1d", "wave_2d_long_time"}:
            raise AttributeError(
                "initial_velocity is only defined for registered wave initial states"
            )
        if isinstance(coordinates, torch.Tensor):
            return torch.zeros(
                (coordinates.shape[0], 1),
                dtype=coordinates.dtype,
                device=coordinates.device,
            )
        array = np.asarray(coordinates)
        return np.zeros((array.shape[0], 1), dtype=np.float32)

    def initial_state_evaluation_samples(
        self,
        n: int = 2048,
        *,
        device: str | None = None,
    ) -> torch.Tensor:
        """Return a deterministic Wave1D initial-state grid."""

        if self.problem_id != "wave_1d":
            raise AttributeError("Initial-state evaluation is only defined for wave_1d")
        spec = self.get_spec()
        x_lower, x_upper = spec.domain.bounds["x"]
        t_lower = spec.domain.bounds["t"][0]
        x = torch.linspace(float(x_lower), float(x_upper), max(2, int(n)))
        t = torch.full_like(x, float(t_lower))
        return torch.stack((x, t), dim=1).to(device or "cpu")

    def compute_governing_residuals(
        self,
        model: Any,
        samples: torch.Tensor,
        *,
        create_graph: bool = True,
    ) -> Mapping[str, torch.Tensor]:
        inputs = samples if samples.requires_grad else samples.detach().clone().requires_grad_(True)
        outputs = model(inputs)
        values = self.pinnacle_problem.pde(inputs, outputs)
        if not isinstance(values, (list, tuple)):
            values = [values]
        configs = list(getattr(self.pinnacle_problem, "loss_config", []))[: len(values)]
        residuals: dict[str, torch.Tensor] = {}
        for index, value in enumerate(values):
            name = _safe_name(configs[index].get("name") if index < len(configs) else f"equation_{index}")
            tensor = value if value.ndim > 1 else value.reshape(-1, 1)
            residuals[name] = tensor if create_graph else tensor.detach()
        return residuals

    def compute_constraint_residuals(
        self,
        model: Any,
        samples: Mapping[str, PINNacleConstraintBatch],
        *,
        create_graph: bool = True,
    ) -> Mapping[str, torch.Tensor]:
        residuals: dict[str, torch.Tensor] = {}
        grouped: dict[str, list[torch.Tensor]] = {"boundary": [], "initial": []}
        for name, batch in samples.items():
            inputs = batch.points.detach().clone().requires_grad_(True)
            outputs = model(inputs)
            if name == "interface_flux_continuity":
                paired = batch.paired_points
                axes = batch.normal_axes
                coefficient_left = batch.coefficient_left
                coefficient_right = batch.coefficient_right
                if (
                    paired is None
                    or axes is None
                    or coefficient_left is None
                    or coefficient_right is None
                ):
                    raise RuntimeError(
                        "ManyArea interface samples are missing paired "
                        "coordinates or coefficient metadata"
                    )
                right_inputs = (
                    paired.detach().clone().requires_grad_(True)
                )
                right_outputs = model(right_inputs)
                left_gradient = torch.autograd.grad(
                    outputs=outputs,
                    inputs=inputs,
                    grad_outputs=torch.ones_like(outputs),
                    create_graph=create_graph,
                    retain_graph=create_graph,
                    allow_unused=False,
                )[0]
                right_gradient = torch.autograd.grad(
                    outputs=right_outputs,
                    inputs=right_inputs,
                    grad_outputs=torch.ones_like(right_outputs),
                    create_graph=create_graph,
                    retain_graph=create_graph,
                    allow_unused=False,
                )[0]
                gather_axes = axes.to(
                    device=outputs.device, dtype=torch.long
                ).reshape(-1, 1)
                left_normal = left_gradient.gather(1, gather_axes)
                right_normal = right_gradient.gather(1, gather_axes)
                left_a = coefficient_left.to(
                    device=outputs.device, dtype=outputs.dtype
                )
                right_a = coefficient_right.to(
                    device=outputs.device, dtype=outputs.dtype
                )
                flux_jump = left_a * left_normal - right_a * right_normal
                # The symmetric scale preserves the zero-jump law while
                # preventing high-contrast interfaces from dominating solely
                # because their coefficients are large.
                residual = flux_jump / (
                    0.5 * (left_a.abs() + right_a.abs())
                ).clamp_min(1.0e-6)
            elif (
                self.problem_id in {"wave_1d", "wave_2d_long_time"}
                and name == "initial_velocity"
            ):
                component = batch.pointset_component
                prediction = (
                    outputs[:, component : component + 1]
                    if isinstance(component, int)
                    else outputs
                )
                gradient = torch.autograd.grad(
                    outputs=prediction,
                    inputs=inputs,
                    grad_outputs=torch.ones_like(prediction),
                    create_graph=create_graph,
                    retain_graph=create_graph,
                    allow_unused=False,
                )[0]
                time_index = self.time_index
                if time_index is None:
                    raise RuntimeError(
                        f"{self.problem_id} input variables do not contain time"
                    )
                target = (
                    batch.pointset_values.to(
                        device=outputs.device, dtype=outputs.dtype
                    )
                    if batch.pointset_values is not None
                    else torch.zeros_like(gradient[:, time_index : time_index + 1])
                )
                residual = gradient[:, time_index : time_index + 1] - target
            elif batch.pointset_values is not None:
                component = batch.pointset_component
                prediction = outputs[:, component : component + 1] if isinstance(component, int) else outputs[:, component]
                residual = prediction - batch.pointset_values.to(device=outputs.device, dtype=outputs.dtype)
            else:
                # DeepXDE wraps NumPy-valued boundary functions with
                # ``torch.as_tensor`` without an explicit device.  The adapter
                # deliberately restores PyTorch's process-wide default device
                # to CPU after importing PINNacle, so those target tensors
                # would otherwise remain on CPU during CUDA training.  Scope
                # tensor factories to the active model device while evaluating
                # the boundary condition, then restore the previous default.
                with _torch_factory_device(inputs.device):
                    residual = batch.bc.error(
                        batch.numpy_points,
                        inputs,
                        outputs,
                        0,
                        int(inputs.shape[0]),
                    )
            if residual.ndim == 1:
                residual = residual.reshape(-1, 1)
            if not create_graph:
                residual = residual.detach()
            residuals[name] = residual
            grouped.setdefault(batch.category, []).append(residual)
        for category, values in grouped.items():
            if values:
                residuals[category] = torch.cat([value.reshape(-1, 1) for value in values], dim=0)
        return residuals

    def get_observations(self) -> Any | None:
        pointsets = [item for item in self._constraint_descriptors if item["bc_type"] == "pointset"]
        if not pointsets:
            return None
        observations = {}
        for descriptor in pointsets:
            bc = descriptor["bc"]
            observations[descriptor["name"]] = (
                torch.as_tensor(np.asarray(bc.points), dtype=torch.float32),
                torch.as_tensor(bc.values, dtype=torch.float32),
            )
        return observations

    def evaluation_observations(self, *, device: str | None = None) -> Mapping[str, tuple[torch.Tensor, torch.Tensor]]:
        observations = self.get_observations() or {}
        return {
            name: (inputs.to(device or "cpu"), targets.to(device or "cpu"))
            for name, (inputs, targets) in observations.items()
        }

    def reference_solution(self, samples: torch.Tensor) -> torch.Tensor | None:
        reference_function = getattr(self.pinnacle_problem, "ref_sol", None)
        if callable(reference_function):
            values = reference_function(samples.detach().cpu().numpy())
            return _as_tensor(values, device=str(samples.device), dtype=samples.dtype)
        if getattr(self.pinnacle_problem, "ref_data", None) is None:
            return None
        self._prepare_reference_index()
        assert self._reference_tree is not None and self._reference_outputs is not None
        _, indices = self._reference_tree.query(samples.detach().cpu().numpy(), k=1)
        return _as_tensor(self._reference_outputs[np.asarray(indices)], device=str(samples.device), dtype=samples.dtype)

    def evaluation_samples(self, *, device: str | None = None) -> torch.Tensor:
        ref_data = getattr(self.pinnacle_problem, "ref_data", None)
        if ref_data is not None:
            data = np.asarray(ref_data, dtype=np.float32)
            data = data[~np.isnan(data).any(axis=1)]
            inputs = data[:, : int(self.pinnacle_problem.input_dim)]
            if self.evaluation_size < inputs.shape[0]:
                inputs = inputs[_evenly_spaced_indices(inputs.shape[0], self.evaluation_size)]
            return _as_tensor(inputs, device=device)
        sample_points = self.evaluation_size
        points = _exact_geometry_random_points(
            self.geometry, sample_points, seed=0
        )
        return _as_tensor(points, device=device)

    @property
    def evaluation_size(self) -> int:
        """Return the resolved, always-positive evaluation point count."""

        return self._evaluation_size

    @evaluation_size.setter
    def evaluation_size(self, value: int | None) -> None:
        """Resolve ``None`` to the full reference grid before evaluation."""

        if value is None:
            ref_data = getattr(self.pinnacle_problem, "ref_data", None)
            if ref_data is not None:
                data = np.asarray(ref_data)
                valid_rows = int(np.count_nonzero(~np.isnan(data).any(axis=1)))
                if valid_rows < 1:
                    raise ValueError("PINNacle reference data contains no valid evaluation rows")
                self._evaluation_size = valid_rows
                self._evaluation_scope = "full_reference_grid"
                return
            # A continuous analytic domain has no finite "complete" grid.
            # Use the documented deterministic PINNacle-compatible grid.
            self._evaluation_size = (
                2500 if int(self.pinnacle_problem.input_dim) == 2 else 20_000
            )
            self._evaluation_scope = "canonical_analytic_grid"
            return
        resolved = int(value)
        if resolved < 1:
            raise ValueError("evaluation_size must be a positive integer or None")
        self._evaluation_size = resolved
        self._evaluation_scope = "capped_grid"

    @property
    def evaluation_scope(self) -> str:
        """Describe how the effective evaluation grid size was selected."""

        return self._evaluation_scope

    def pinnacle_fourier_metrics(
        self,
        samples: torch.Tensor,
        prediction: torch.Tensor,
        reference: torch.Tensor,
        *,
        low_cutoff: int = 5,
        high_cutoff: int = 13,
    ) -> dict[str, float]:
        """Reproduce PINNacle's released low/mid/high Fourier error calculation."""

        input_dim = int(self.pinnacle_problem.input_dim)
        spatial_geometry = getattr(self.pinnacle_problem, "geom", None)
        geometry_types = {base.__name__ for base in type(spatial_geometry).__mro__}
        if input_dim > 3 or not geometry_types.intersection({"Interval", "Hypercube"}):
            return {}
        bbox = list(getattr(self.pinnacle_problem, "bbox", None) or [])
        if len(bbox) < 2 * input_dim:
            return {}

        coordinates = samples.detach().cpu().numpy().astype(np.float64, copy=False)
        residual = (prediction - reference).detach().cpu().numpy().astype(np.float64, copy=False)
        finite = np.isfinite(coordinates).all(axis=1) & np.isfinite(residual).all(axis=1)
        coordinates, residual = coordinates[finite], residual[finite]
        if coordinates.shape[0] <= input_dim:
            return {}

        points_per_unit = 3.0e4
        lengths = []
        for index in range(input_dim):
            length = float(bbox[2 * index + 1]) - float(bbox[2 * index])
            if length <= 0:
                return {}
            lengths.append(length)
            points_per_unit /= length
        points_per_unit **= 1.0 / input_dim
        axes = [
            np.linspace(
                float(bbox[2 * index]),
                float(bbox[2 * index + 1]),
                int(np.ceil(lengths[index] * points_per_unit)) + 1,
                endpoint=False,
            )[1:]
            for index in range(input_dim)
        ]
        sample_grid = np.stack(np.meshgrid(*axes), axis=-1)
        flat_grid = sample_grid.reshape((-1, input_dim))

        if input_dim == 1:
            order = np.argsort(coordinates[:, 0])
            interpolated = np.stack(
                [np.interp(flat_grid[:, 0], coordinates[order, 0], residual[order, output]) for output in range(residual.shape[1])],
                axis=1,
            )
        else:
            from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
            from scipy.spatial import Delaunay

            triangulation = Delaunay(coordinates)
            interpolated = LinearNDInterpolator(triangulation, residual)(flat_grid)
            nearest = NearestNDInterpolator(coordinates, residual)(flat_grid)
            interpolated[np.isnan(interpolated)] = nearest[np.isnan(interpolated)]

        residual_grid = interpolated.reshape((*sample_grid.shape[:-1], residual.shape[1]))
        spectrum = np.fft.rfftn(residual_grid, axes=tuple(range(residual_grid.ndim - 1)))
        spectral_error = np.mean(np.abs(spectrum) ** 2 / residual_grid.size, axis=-1)
        low, mid, high = _frequency_band_means(spectral_error, low_cutoff, high_cutoff)
        return {"low": low, "mid": mid, "high": high}

    def shock_region_definition(self) -> dict[str, Any] | None:
        """Return the frozen Burgers-1D shock evaluation contract.

        This definition is deliberately expressed in physical coordinates and
        applied to the ordinary fixed evaluation grid.  Candidate samplers
        therefore cannot change the points used for the shock metric.
        """

        if self.problem_id != "burgers_1d":
            return None
        return {
            "name": "burgers_1d_late_time_central_window",
            "spatial_variable": "x",
            "time_variable": "t",
            "spatial_interval": [-0.5, 0.5],
            "time_interval": [0.75, 1.0],
            "selection_formula": "-0.5 <= x <= 0.5 and 0.75 <= t <= 1.0",
            "error_formulas": {
                "mse": "mean((u_pred - u_ref)^2)",
                "relative_l2": "||u_pred-u_ref||_2 / (||u_ref||_2 + 1e-12)",
                "relative_linf": "max|u_pred-u_ref| / (max|u_ref| + 1e-12)",
            },
            "source_grid": "fixed benchmark evaluation grid",
        }

    def shock_region_evaluation_samples(self, *, device: str | None = None) -> torch.Tensor | None:
        definition = self.shock_region_definition()
        if definition is None:
            return None
        samples = self.evaluation_samples(device=device)
        # Burgers1D has the registered input ordering [x, t].
        mask = (
            (samples[:, 0] >= float(definition["spatial_interval"][0]))
            & (samples[:, 0] <= float(definition["spatial_interval"][1]))
            & (samples[:, 1] >= float(definition["time_interval"][0]))
            & (samples[:, 1] <= float(definition["time_interval"][1]))
        )
        selected = samples[mask]
        if selected.shape[0] == 0:
            raise RuntimeError("Frozen Burgers-1D shock region contains no evaluation points")
        return selected

    def residual_evaluation_samples(self, *, device: str | None = None) -> torch.Tensor:
        """Return a fixed interior grid independent of the solution test set.

        PINNacle evaluates solution errors on all available reference rows but
        uses a separate finite geometry grid for PDE test loss.  Reproducing
        that separation prevents high-order derivatives from being evaluated
        over reference datasets containing hundreds of thousands of points.
        The CPU coordinates are cached so every candidate sees identical
        residual-evaluation points.
        """

        if self._residual_evaluation_inputs is None:
            points = _exact_geometry_random_points(
                self.geometry,
                self.residual_evaluation_size,
                seed=0,
            )
            points = np.asarray(points, dtype=np.float32)
            if points.ndim != 2 or points.shape[0] < 1:
                raise ValueError("Residual evaluation geometry returned no valid points")
            finite = points[~np.isnan(points).any(axis=1)]
            if finite.shape[0] < 1:
                raise ValueError("Residual evaluation geometry returned only NaN points")
            self._residual_evaluation_inputs = finite
        return _as_tensor(self._residual_evaluation_inputs, device=device)

    @property
    def residual_evaluation_size(self) -> int:
        """Return the requested size of the independent PDE-residual grid."""

        return self._residual_evaluation_size

    @residual_evaluation_size.setter
    def residual_evaluation_size(self, value: int) -> None:
        resolved = int(value)
        if resolved < 1:
            raise ValueError("residual_evaluation_size must be a positive integer")
        self._residual_evaluation_size = resolved
        self._residual_evaluation_inputs = None

    def clear_autodiff_cache(self) -> None:
        """Release DeepXDE's process-wide derivative graph cache.

        PINNacle equations evaluate derivatives through DeepXDE, whose
        Jacobian and Hessian helpers cache entries by the identity of each
        iteration's input/output tensors.  Those identities are never reused
        by the OpenSpec sampler, so leaving the cache populated retains one
        complete autograd graph per training step and grows CUDA memory
        linearly until the process runs out of memory.
        """

        gradients = sys.modules.get("deepxde.gradients")
        clear = getattr(gradients, "clear", None)
        if callable(clear):
            clear()

    def _build_initial_target_evaluators(
        self,
    ) -> dict[str, Callable[[np.ndarray], np.ndarray]]:
        """Cache fast target evaluators for expensive tabulated initial data."""

        if self.problem_id != "burgers_2d":
            return {}
        source_tables = list(getattr(self.pinnacle_problem, "ics", None) or [])
        if not source_tables:
            return {}

        from scipy.interpolate import RegularGridInterpolator

        evaluators: dict[str, Callable[[np.ndarray], np.ndarray]] = {}
        for descriptor in self._constraint_descriptors:
            if descriptor["category"] != "initial":
                continue
            component = getattr(descriptor["bc"], "component", None)
            if not isinstance(component, int) or not 0 <= component < len(source_tables):
                continue
            table = np.asarray(source_tables[component])
            if table.ndim != 2 or table.shape[1] < 3:
                continue
            coordinates = np.asarray(table[:, :2], dtype=np.float64)
            x_axis = np.unique(coordinates[:, 0])
            y_axis = np.unique(coordinates[:, 1])
            if len(x_axis) * len(y_axis) != len(coordinates):
                continue
            x_indices = np.searchsorted(x_axis, coordinates[:, 0])
            y_indices = np.searchsorted(y_axis, coordinates[:, 1])
            occupied = np.zeros((len(x_axis), len(y_axis)), dtype=bool)
            occupied[x_indices, y_indices] = True
            if not bool(occupied.all()):
                continue
            values = np.empty((len(x_axis), len(y_axis)), dtype=np.float64)
            values[x_indices, y_indices] = np.asarray(table[:, 2], dtype=np.float64)
            interpolator = RegularGridInterpolator(
                (x_axis, y_axis),
                values,
                method="linear",
                bounds_error=False,
                fill_value=None,
            )

            def evaluate(
                points: np.ndarray,
                *,
                _interpolator: Any = interpolator,
            ) -> np.ndarray:
                result = _interpolator(np.asarray(points, dtype=np.float64)[:, :2])
                return np.asarray(result, dtype=np.float32).reshape(-1, 1)

            evaluators[descriptor["name"]] = evaluate
        return evaluators

    def _initial_target_values(
        self,
        descriptor: Mapping[str, Any],
        points: np.ndarray,
        *,
        device: str | None,
    ) -> torch.Tensor | None:
        evaluator = self._initial_target_evaluators.get(str(descriptor["name"]))
        if evaluator is not None:
            return _as_tensor(evaluator(points), device=device)

        function = getattr(descriptor["bc"], "func", None)
        if not callable(function):
            return None
        with _torch_factory_device(device or "cpu"):
            values = function(points, 0, int(len(points)), None)
        tensor = _as_tensor(values, device=device)
        return tensor.reshape(-1, 1) if tensor.ndim == 1 else tensor

    def _describe_constraints(self) -> list[dict[str, Any]]:
        bcs = list(getattr(self.pinnacle_problem, "bcs", None) or [])
        configs = list(getattr(self.pinnacle_problem, "loss_config", []))[int(self.pinnacle_problem.num_pde) :]
        descriptors = []
        used: set[str] = set()
        for index, bc in enumerate(bcs):
            bc_type = _bc_type(bc)
            category = "initial" if bc_type == "initial" else "boundary"
            raw_name = configs[index].get("name") if index < len(configs) else f"{bc_type}_{index}"
            location = "initial_time" if category == "initial" else "spatial_boundary"
            derivative_variable = None
            derivative_order = 0
            if self.problem_id == "wave_1d":
                if index == 0:
                    raw_name = "initial_velocity"
                    category = "initial"
                    location = "t=t0"
                    derivative_variable = "t"
                    derivative_order = 1
                elif index == 1:
                    raw_name = "initial_displacement"
                    category = "initial"
                    location = "t=t0"
                elif index == 2:
                    raw_name = "spatial_boundary"
                    category = "boundary"
                    location = "x=x_min or x=x_max"
            elif self.problem_id == "wave_2d_long_time":
                if index == 0:
                    raw_name = "initial_displacement"
                    category = "initial"
                    location = "t=t0"
                elif index == 1:
                    raw_name = "initial_velocity"
                    category = "initial"
                    location = "t=t0"
                    derivative_variable = "t"
                    derivative_order = 1
                elif index == 2:
                    raw_name = "spatial_boundary"
                    category = "boundary"
                    location = "x or y spatial boundary"
            name = _unique_name(_safe_name(raw_name), used)
            descriptors.append(
                {
                    "index": index,
                    "name": name,
                    "display_name": str(raw_name),
                    "category": category,
                    "schema_type": "initial" if category == "initial" else ("periodic" if bc_type == "periodic" else "boundary"),
                    "bc_type": bc_type,
                    "bc": bc,
                    "location": location,
                    "derivative_variable": derivative_variable,
                    "derivative_order": derivative_order,
                }
            )
        if self.problem_id == "poisson_2d_many_area":
            descriptors.append(
                {
                    "index": len(descriptors),
                    "name": "interface_flux_continuity",
                    "display_name": "Internal interface flux continuity",
                    "category": "boundary",
                    "schema_type": "boundary",
                    "bc_type": "interface_flux",
                    "bc": None,
                    "location": "40 internal material interface segments",
                    "derivative_variable": "interface_normal",
                    "derivative_order": 1,
                }
            )
        return descriptors

    def _sample_many_area_interface_flux(
        self,
        count: int,
        *,
        device: str | None,
    ) -> PINNacleConstraintBatch:
        split = tuple(
            int(value)
            for value in getattr(self.pinnacle_problem, "split", (5, 5))
        )
        bbox = tuple(float(value) for value in self.pinnacle_problem.bbox)
        table = np.asarray(self.pinnacle_problem.a_cof, dtype=np.float32)
        x_edges = np.linspace(bbox[0], bbox[1], split[0] + 1)
        y_edges = np.linspace(bbox[2], bbox[3], split[1] + 1)
        segments: list[tuple[int, int, int]] = []
        segments.extend(
            (0, interface_x, cell_y)
            for interface_x in range(1, split[0])
            for cell_y in range(split[1])
        )
        segments.extend(
            (1, cell_x, interface_y)
            for interface_y in range(1, split[1])
            for cell_x in range(split[0])
        )
        total = max(1, int(count))
        choices = np.arange(total, dtype=np.int64) % len(segments)
        np.random.shuffle(choices)
        left = np.empty((total, 2), dtype=np.float32)
        right = np.empty((total, 2), dtype=np.float32)
        axes = np.empty(total, dtype=np.int64)
        coefficient_left = np.empty((total, 1), dtype=np.float32)
        coefficient_right = np.empty((total, 1), dtype=np.float32)
        cell_width = (bbox[1] - bbox[0]) / split[0]
        cell_height = (bbox[3] - bbox[2]) / split[1]
        offset_x = 0.02 * cell_width
        offset_y = 0.02 * cell_height
        for row, segment_index in enumerate(choices):
            axis, first, second = segments[int(segment_index)]
            if axis == 0:
                interface_x, cell_y = first, second
                margin = 0.02 * cell_height
                coordinate = np.random.uniform(
                    y_edges[cell_y] + margin,
                    y_edges[cell_y + 1] - margin,
                )
                left[row] = (x_edges[interface_x] - offset_x, coordinate)
                right[row] = (x_edges[interface_x] + offset_x, coordinate)
                coefficient_left[row, 0] = table[
                    interface_x - 1, cell_y
                ]
                coefficient_right[row, 0] = table[interface_x, cell_y]
            else:
                cell_x, interface_y = first, second
                margin = 0.02 * cell_width
                coordinate = np.random.uniform(
                    x_edges[cell_x] + margin,
                    x_edges[cell_x + 1] - margin,
                )
                left[row] = (coordinate, y_edges[interface_y] - offset_y)
                right[row] = (coordinate, y_edges[interface_y] + offset_y)
                coefficient_left[row, 0] = table[
                    cell_x, interface_y - 1
                ]
                coefficient_right[row, 0] = table[cell_x, interface_y]
            axes[row] = axis
        return PINNacleConstraintBatch(
            name="interface_flux_continuity",
            category="boundary",
            bc=None,
            points=_as_tensor(left, device=device),
            numpy_points=left,
            paired_points=_as_tensor(right, device=device),
            normal_axes=torch.as_tensor(
                axes, dtype=torch.long, device=device or "cpu"
            ),
            coefficient_left=_as_tensor(
                coefficient_left, device=device
            ),
            coefficient_right=_as_tensor(
                coefficient_right, device=device
            ),
            location="40 internal material interface segments",
            derivative_variable="interface_normal",
            derivative_order=1,
        )

    def _sample_bc_points(self, bc: Any, category: str, count: int) -> np.ndarray:
        candidate_count = max(256, count * 8)
        bc_type = _bc_type(bc)
        normal_dependent = bc_type in {"neumann", "robin"}
        for _ in range(4):
            if category == "initial" and hasattr(
                self.geometry, "random_initial_points"
            ):
                candidates = self.geometry.random_initial_points(
                    candidate_count, random="pseudo"
                )
            else:
                try:
                    candidates = self.geometry.random_boundary_points(
                        candidate_count, random="pseudo"
                    )
                except ValueError as exc:
                    # DeepXDE's GeometryXTime assumes that the wrapped spatial
                    # geometry returns exactly ``n`` boundary points.  Its
                    # Rectangle implementation samples ``n + 2`` points and
                    # removes near-corner values, which can occasionally leave
                    # ``n - 1`` rows.  GeometryXTime then tries to hstack those
                    # rows with exactly ``n`` time values and raises.  Build the
                    # same spatial-boundary x time sample explicitly with a
                    # small overdraw and a deterministic trim.
                    candidates = _geometry_time_boundary_points(
                        self.geometry,
                        candidate_count,
                        random="pseudo",
                        original_error=exc,
                    )
            points = np.asarray(bc.collocation_points(candidates), dtype=np.float32)
            if normal_dependent:
                points = _exclude_ambiguous_boundary_normal_points(points, bc)
            if len(points) > 0:
                if bc_type == "periodic":
                    pair_count = len(points) // 2
                    keep = min(pair_count, count)
                    return np.concatenate(
                        (points[:keep], points[pair_count : pair_count + keep]),
                        axis=0,
                    )
                if not normal_dependent or len(points) >= count:
                    return points[:count]
            candidate_count *= 4
        raise RuntimeError(
            f"PINNacle constraint '{type(bc).__name__}' could not produce {count} "
            f"unambiguous collocation points for {self.problem_id}"
        )

    def _sample_pointset(
        self,
        descriptor: Mapping[str, Any],
        count: int,
        *,
        device: str | None,
    ) -> PINNacleConstraintBatch:
        bc = descriptor["bc"]
        points = np.asarray(bc.points, dtype=np.float32)
        total = len(points)
        indices = np.random.choice(total, size=min(count, total), replace=False)
        selected_points = points[indices]
        values = torch.as_tensor(bc.values, dtype=torch.float32)[indices]
        return PINNacleConstraintBatch(
            name=descriptor["name"],
            category=descriptor["category"],
            bc=bc,
            points=_as_tensor(selected_points, device=device),
            numpy_points=selected_points,
            pointset_values=values.to(device or "cpu"),
            pointset_component=bc.component,
        )

    def _prepare_reference_index(self) -> None:
        if self._reference_tree is not None:
            return
        from scipy.spatial import cKDTree

        data = np.asarray(self.pinnacle_problem.ref_data, dtype=np.float32)
        input_dim = int(self.pinnacle_problem.input_dim)
        output_dim = int(self.pinnacle_problem.output_dim)
        self._reference_inputs = data[:, :input_dim]
        self._reference_outputs = data[:, input_dim : input_dim + output_dim]
        self._reference_tree = cKDTree(self._reference_inputs)


def _problem_specific_training_and_metric_metadata(
    problem_id: str,
) -> dict[str, Any]:
    """Return benchmark safeguards that belong to the executable contract."""

    if problem_id != "navier_stokes_2d_C":
        return {}
    return {
        "metric_providers": [
            "per_output_error",
            "residual_components",
            "constraint_components",
        ],
        "has_pressure_gauge": True,
        "pressure_gauge_contract": "pointwise_dirichlet_p_at_origin",
    }


def make_pinnacle_problem_class(problem_id: str) -> type[PINNacleProblemAdapter]:
    """Create a zero-argument registry class for one canonical problem ID."""

    canonical = _canonical_problem_id(problem_id)
    profile = get_pinnacle_profile(canonical)
    if profile is None:
        raise KeyError(f"Unknown PINNacle problem '{problem_id}'")

    class RegisteredPINNacleProblem(PINNacleProblemAdapter):
        def __init__(self) -> None:
            super().__init__(canonical)

    RegisteredPINNacleProblem.__name__ = f"{profile['pinnacle_name']}Problem"
    RegisteredPINNacleProblem.__qualname__ = RegisteredPINNacleProblem.__name__
    RegisteredPINNacleProblem.__module__ = __name__
    return RegisteredPINNacleProblem


def _prepare_pinnacle_imports() -> None:
    os.environ.setdefault("DDEBACKEND", "pytorch")
    root_text = str(PINNACLE_ROOT)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    _install_skopt_stub()
    for package, path in {
        "src": PINNACLE_ROOT / "src",
        "src.pde": PINNACLE_ROOT / "src" / "pde",
        "src.model": PINNACLE_ROOT / "src" / "model",
        "src.utils": PINNACLE_ROOT / "src" / "utils",
    }.items():
        if package not in sys.modules:
            module = types.ModuleType(package)
            module.__path__ = [str(path)]
            module.__package__ = package
            sys.modules[package] = module


def _install_skopt_stub() -> None:
    try:
        importlib.import_module("skopt")
        return
    except ModuleNotFoundError:
        pass
    module = types.ModuleType("skopt")
    module.sampler = types.SimpleNamespace()
    sys.modules["skopt"] = module


def _restore_torch_defaults() -> None:
    """Undo DeepXDE's process-wide CUDA default-tensor side effect."""

    torch.set_default_dtype(torch.float32)
    if hasattr(torch, "set_default_device"):
        torch.set_default_device("cpu")


@contextmanager
def _torch_factory_device(device: torch.device | str):
    """Create implicit DeepXDE tensors on the active model device."""

    if not hasattr(torch, "set_default_device"):
        yield
        return
    previous = torch.get_default_device() if hasattr(torch, "get_default_device") else "cpu"
    torch.set_default_device(device)
    try:
        yield
    finally:
        torch.set_default_device(previous)


def _patch_reference_readers() -> None:
    """Make PINNacle's text reference loader independent of Windows locale."""

    baseclass = importlib.import_module("src.pde.baseclass")
    if getattr(baseclass.BasePDE, "_llm4pinn_utf8_patch", False):
        return

    def trans_time_data_to_dataset(instance: Any, datapath: str) -> None:
        data = instance.ref_data
        slice_count = (data.shape[1] - instance.input_dim + 1) // instance.output_dim
        if slice_count * instance.output_dim != data.shape[1] - instance.input_dim + 1:
            raise ValueError("Data shape is not multiple of pde.output_dim")

        def extract_time(value: str) -> float | None:
            index = value.find("t=")
            if index == -1:
                return None
            return float(value[index + 2 :].split(" ")[0])

        times = None
        with open(datapath, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("%") and line.count("@") == slice_count * instance.output_dim:
                    times = [extract_time(value) for value in line.split("@")[1:]]
        if times is None or None in times:
            raise ValueError("Reference Data not in Comsol format or does not contain time info")
        time_values = np.asarray(times[:: instance.output_dim])
        time_grid, x0 = np.meshgrid(time_values, data[:, 0])
        columns = [x0.reshape(-1)]
        for index in range(1, instance.input_dim - 1):
            columns.append(np.stack([data[:, index] for _ in range(slice_count)]).T.reshape(-1))
        columns.append(time_grid.reshape(-1))
        for index in range(instance.output_dim):
            columns.append(data[:, instance.input_dim - 1 + index :: instance.output_dim].reshape(-1))
        instance.ref_data = np.stack(columns).T.astype(np.float32)

    def load_ref_data(
        instance: Any,
        datapath: str,
        transform_fn: Any | None = None,
        t_transpose: bool = False,
    ) -> None:
        instance.ref_data = np.loadtxt(datapath, comments="%", encoding="utf-8").astype(np.float32)
        if t_transpose:
            trans_time_data_to_dataset(instance, datapath)
        if transform_fn is not None:
            instance.ref_data = transform_fn(instance.ref_data)

    baseclass.BasePDE.trans_time_data_to_dataset = trans_time_data_to_dataset
    baseclass.BasePDE.load_ref_data = load_ref_data
    baseclass.BasePDE._llm4pinn_utf8_patch = True


def _canonical_problem_id(problem_id: str) -> str:
    from forge.pipeline.c_knowledge_retrieval.pinnacle_registry import PINNACLE_PDE_PROFILES

    requested = str(problem_id).lower()
    for canonical, profile in PINNACLE_PDE_PROFILES.items():
        names = [canonical, profile["pinnacle_name"], *profile["aliases"]]
        if requested in {str(name).lower() for name in names}:
            return canonical
    raise KeyError(f"Unknown PINNacle problem '{problem_id}'")


def _input_variable_names(profile: Mapping[str, Any], input_dim: int) -> list[str]:
    spatial = int(profile["spatial_dimension"])
    base = ["x", "y", "z"] + [f"x{index}" for index in range(4, max(spatial, 3) + 1)]
    variables = base[:spatial]
    if input_dim > spatial:
        variables.extend(["t"] + [f"q{index}" for index in range(input_dim - spatial - 1)])
    return variables[:input_dim]


def _output_variable_names(pinnacle_problem: Any) -> list[str]:
    config = getattr(pinnacle_problem, "output_config", None) or []
    names = [str(item.get("name") or f"u{index + 1}") for index, item in enumerate(config)]
    generated_names = all(name == f"y_{index + 1}" for index, name in enumerate(names))
    if len(names) == int(pinnacle_problem.output_dim) and not generated_names:
        return names
    return ["u"] if int(pinnacle_problem.output_dim) == 1 else [f"u{index + 1}" for index in range(int(pinnacle_problem.output_dim))]


def _bounds_from_bbox(bbox: Any, variables: list[str]) -> dict[str, tuple[float, float]]:
    if bbox is None or len(bbox) < 2 * len(variables):
        return {}
    return {
        variable: (float(bbox[2 * index]), float(bbox[2 * index + 1]))
        for index, variable in enumerate(variables)
    }


def _frequency_band_means(
    spectral_error: np.ndarray,
    low_cutoff: int,
    high_cutoff: int,
) -> tuple[float, float, float]:
    """Match PINNacle TesterCallback.frmse_calc frequency-band aggregation."""

    if spectral_error.ndim == 1:
        return (
            float(spectral_error[:low_cutoff].mean()),
            float(spectral_error[low_cutoff:high_cutoff].mean()),
            float(spectral_error[high_cutoff:].mean()),
        )

    totals = [0.0, 0.0, 0.0]
    counts = [0, 0, 0]
    for indices in product(*[range((size + 1) // 2) for size in spectral_error.shape[:-1]]):
        frequency_squared = sum(index**2 for index in indices)
        low_end = min(
            int(np.sqrt(max(0, low_cutoff**2 - frequency_squared))),
            spectral_error.shape[-1],
        )
        high_end = min(
            int(np.sqrt(max(0, high_cutoff**2 - frequency_squared))),
            spectral_error.shape[-1],
        )
        slices = (
            spectral_error[(*indices, slice(None, low_end))],
            spectral_error[(*indices, slice(low_end, high_end))],
            spectral_error[(*indices, slice(high_end, None))],
        )
        for band, values in enumerate(slices):
            totals[band] += float(values.sum())
            counts[band] += int(values.size)
    return tuple(total / count if count else float("nan") for total, count in zip(totals, counts))


def _geometry_time_boundary_points(
    geometry: Any,
    count: int,
    *,
    random: str,
    original_error: ValueError,
) -> np.ndarray:
    """Robustly sample a spatial boundary crossed with a time interval."""

    spatial_geometry = getattr(geometry, "geometry", None)
    time_domain = getattr(geometry, "timedomain", None)
    if spatial_geometry is None or time_domain is None:
        raise original_error

    target = max(1, int(count))
    overdraw = max(32, target // 1000)
    for _ in range(4):
        spatial = np.asarray(
            spatial_geometry.random_boundary_points(target + overdraw, random=random),
            dtype=np.float32,
        )
        if spatial.shape[0] >= target:
            time = np.asarray(
                time_domain.random_points(spatial.shape[0], random=random),
                dtype=np.float32,
            )
            if time.shape[0] >= target:
                time = np.random.permutation(time)
                return np.hstack((spatial[:target], time[:target]))
        overdraw *= 2
    raise original_error


def _bc_type(bc: Any) -> str:
    name = type(bc).__name__.lower()
    if name == "ic":
        return "initial"
    for candidate in [
        "pointset",
        "periodic",
        "dirichlet",
        "neumann",
        "robin",
        "operator",
    ]:
        if candidate in name:
            return candidate
    return name


def _exclude_ambiguous_boundary_normal_points(
    points: np.ndarray,
    bc: Any,
) -> np.ndarray:
    """Remove box corners/edges whose outward normal is not unique.

    DeepXDE's Hypercube/Rectangle normal implementation warns whenever a point
    touches more than one coordinate face and substitutes an averaged normal.
    That convention is unsuitable for Neumann and Robin constraints, where the
    mathematical boundary normal must identify one face.  OpenSpec samples its
    own constraint batches, so filtering here is the equivalent of DeepXDE's
    ``PDE(..., exclusions=...)`` without changing the requested batch size.
    """

    values = np.asarray(points)
    if values.ndim != 2 or values.shape[0] == 0:
        return values
    geometry = getattr(bc, "geom", None)
    ambiguous = np.zeros(values.shape[0], dtype=bool)
    for xmin, xmax in _axis_aligned_geometry_bounds(geometry, values.dtype):
        if values.shape[1] < xmin.size:
            continue
        coordinates = values[:, : xmin.size]
        face_directions = -np.isclose(coordinates, xmin).astype(np.int8)
        face_directions += np.isclose(coordinates, xmax).astype(np.int8)
        ambiguous |= np.count_nonzero(face_directions, axis=1) > 1
    return values[~ambiguous]


def _axis_aligned_geometry_bounds(
    geometry: Any,
    dtype: np.dtype,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Collect box bounds recursively through GeometryXTime and CSG nodes."""

    bounds: list[tuple[np.ndarray, np.ndarray]] = []
    pending = [geometry]
    visited: set[int] = set()
    while pending:
        current = pending.pop()
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))
        xmin = np.asarray(getattr(current, "xmin", []), dtype=dtype).reshape(-1)
        xmax = np.asarray(getattr(current, "xmax", []), dtype=dtype).reshape(-1)
        if xmin.size and xmin.shape == xmax.shape:
            bounds.append((xmin, xmax))
        for attribute in ("geometry", "geom1", "geom2"):
            child = getattr(current, attribute, None)
            if child is not None and child is not current:
                pending.append(child)
    return bounds


def _exact_geometry_random_points(
    geometry: Any,
    count: int,
    *,
    seed: int,
) -> np.ndarray:
    """Return an exact deterministic interior sample without grid overdraw warnings."""

    requested = max(1, int(count))
    sampler = getattr(geometry, "random_points", None)
    if not callable(sampler):
        raise TypeError(f"{type(geometry).__name__} does not provide random_points")
    with _numpy_seed(seed):
        points = np.asarray(sampler(requested, random="pseudo"), dtype=np.float32)
    if points.ndim != 2 or points.shape[0] < requested:
        raise ValueError(
            f"Geometry returned {points.shape[0] if points.ndim else 0} points; "
            f"{requested} required"
        )
    return points[:requested]


def _safe_name(value: Any) -> str:
    text = str(value or "constraint").strip().lower()
    safe = "".join(character if character.isalnum() else "_" for character in text)
    return "_".join(part for part in safe.split("_") if part) or "constraint"


def _unique_name(name: str, used: set[str]) -> str:
    candidate = name
    index = 2
    while candidate in used:
        candidate = f"{name}_{index}"
        index += 1
    used.add(candidate)
    return candidate


def _evenly_spaced_indices(total: int, limit: int) -> np.ndarray:
    if total <= limit:
        return np.arange(total)
    return np.linspace(0, total - 1, num=limit, dtype=np.int64)


def _as_tensor(
    values: Any,
    *,
    device: str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values.to(device=device or values.device, dtype=dtype)
    return torch.as_tensor(np.asarray(values), dtype=dtype, device=device or "cpu")


@contextmanager
def _working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


@contextmanager
def _numpy_seed(seed: int | None):
    if seed is None:
        yield
        return
    state = np.random.get_state()
    np.random.seed(int(seed))
    try:
        yield
    finally:
        np.random.set_state(state)


def _disable_dynamic_batch_range_caches(pinnacle_problem: Any) -> None:
    """Disable DeepXDE's identity cache for resampled OpenSpec constraints.

    DeepXDE's PyTorch IC/BC wrappers cache NumPy target values by ``id(X)``.
    That is valid for DeepXDE's own fixed training array, but OpenSpec creates
    a new constraint array on every sampling step.  Once an old array is
    released, CPython may reuse its identity for different coordinates and
    DeepXDE then returns stale target values.  Hard constraints can therefore
    acquire a spurious O(1) IC loss even though the transformed prediction is
    exact.  Replace only the recognized range wrappers with equivalent
    no-cache calls; native Robin/periodic/point-set functions are untouched.
    """

    for bc in list(getattr(pinnacle_problem, "bcs", None) or []):
        for attribute in ("func", "boundary_normal"):
            cached = getattr(bc, attribute, None)
            original = getattr(cached, "__wrapped__", None)
            if not callable(cached) or not callable(original):
                continue
            try:
                parameters = inspect.signature(original).parameters.values()
                positional_count = sum(
                    parameter.kind
                    in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
                    for parameter in parameters
                )
            except (TypeError, ValueError):
                positional_count = 1

            @wraps(original)
            def no_cache(
                X: np.ndarray,
                beg: int,
                end: int,
                aux_var: Any,
                *,
                _function: Callable[..., Any] = original,
                _positional_count: int = positional_count,
            ) -> Any:
                points = X[beg:end]
                if _positional_count >= 2:
                    auxiliary = None if aux_var is None else aux_var[beg:end]
                    return _function(points, auxiliary)
                return _function(points)

            setattr(no_cache, "_openspec_dynamic_batch_no_cache", True)
            setattr(bc, attribute, no_cache)


__all__ = [
    "PINNACLE_ROOT",
    "PINNacleConstraintBatch",
    "PINNacleProblemAdapter",
    "instantiate_pinnacle_problem",
    "load_pinnacle_class",
    "make_pinnacle_problem_class",
]
