"""Immutable PDE-agnostic training-component registry and capabilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from types import MappingProxyType
from typing import Any, Mapping


LOSS_COMPONENT_CATEGORIES = frozenset(
    {
        "equation_residual",
        "boundary_constraint",
        "initial_constraint",
        "interface_constraint",
        "observation",
        "integral_constraint",
        "conservation_constraint",
        "regularization",
        "auxiliary_constraint",
    }
)
SAMPLING_DOMAIN_ROLES = frozenset(
    {
        "interior",
        "boundary",
        "initial_surface",
        "interface",
        "observation",
        "integral_domain",
        "parameter_domain",
        "custom_constraint_domain",
    }
)
VARIABLE_ROLES = frozenset(
    {"spatial", "temporal", "parameter", "stochastic", "auxiliary"}
)
CURRICULUM_AXIS_ROLES = frozenset(
    {
        "temporal", "parameter", "spatial_domain", "geometry", "difficulty",
        "frequency", "scale", "fidelity", "custom",
    }
)


def _metadata(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(value or {}))


@dataclass(frozen=True)
class LossComponentDescriptor:
    component_id: str
    category: str
    sampler_id: str | None
    trainable: bool = True
    dynamically_weightable: bool = True
    validation_enabled: bool = True
    active_stages: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "component_id", str(self.component_id))
        object.__setattr__(self, "category", str(self.category))
        object.__setattr__(self, "active_stages", tuple(self.active_stages))
        object.__setattr__(self, "metadata", _metadata(self.metadata))
        if not self.component_id:
            raise ValueError("Loss component_id must not be empty")
        if self.category not in LOSS_COMPONENT_CATEGORIES:
            raise ValueError(f"Unsupported loss component category: {self.category}")


@dataclass(frozen=True)
class SamplingComponentDescriptor:
    sampler_id: str
    domain_role: str
    associated_loss_component_ids: tuple[str, ...] = ()
    point_budget: int | None = None
    mutable_during_first_order_stage: bool = False
    fixed_during_second_order_stage: bool = True
    replacement_supported: bool = False
    pointwise_scoring_supported: bool = False
    candidate_generation_supported: bool = False
    state_snapshot_supported: bool = False
    minimum_point_count: int = 1
    maximum_point_count: int | None = None
    validation_probe_supported: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sampler_id", str(self.sampler_id))
        object.__setattr__(
            self,
            "associated_loss_component_ids",
            tuple(sorted(str(item) for item in self.associated_loss_component_ids)),
        )
        object.__setattr__(self, "metadata", _metadata(self.metadata))
        if not self.sampler_id:
            raise ValueError("Sampling sampler_id must not be empty")
        if self.domain_role not in SAMPLING_DOMAIN_ROLES:
            raise ValueError(f"Unsupported sampling domain role: {self.domain_role}")
        if self.point_budget is not None and int(self.point_budget) < 0:
            raise ValueError("Sampling point_budget must be non-negative")
        if int(self.minimum_point_count) < 0:
            raise ValueError("Sampling minimum_point_count must be non-negative")
        if (
            self.maximum_point_count is not None
            and int(self.maximum_point_count) < int(self.minimum_point_count)
        ):
            raise ValueError("Sampling maximum_point_count must not be below minimum_point_count")
        complete_replacement = (
            self.pointwise_scoring_supported
            and self.candidate_generation_supported
            and self.state_snapshot_supported
            and self.mutable_during_first_order_stage
        )
        if self.replacement_supported and not complete_replacement:
            raise ValueError(
                "Sampling replacement requires scoring, candidate generation, state snapshot, and first-order mutability"
            )


@dataclass(frozen=True)
class VariableRoleDescriptor:
    variable_id: str
    role: str
    lower_bound: float | None = None
    upper_bound: float | None = None
    causal_ordered: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "variable_id", str(self.variable_id))
        object.__setattr__(self, "metadata", _metadata(self.metadata))
        if not self.variable_id:
            raise ValueError("Variable variable_id must not be empty")
        if self.role not in VARIABLE_ROLES:
            raise ValueError(f"Unsupported variable role: {self.role}")
        if (
            self.lower_bound is not None
            and self.upper_bound is not None
            and float(self.lower_bound) >= float(self.upper_bound)
        ):
            raise ValueError("Variable lower_bound must be below upper_bound")


@dataclass(frozen=True)
class OptimizationStageDescriptor:
    stage_id: str
    optimizer_family: str
    order: int
    first_order: bool
    second_order: bool
    dynamic_objective_allowed: bool
    recoverable: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage_id", str(self.stage_id))
        object.__setattr__(self, "optimizer_family", str(self.optimizer_family))
        object.__setattr__(self, "metadata", _metadata(self.metadata))
        if not self.stage_id:
            raise ValueError("Optimization stage_id must not be empty")
        if self.first_order == self.second_order:
            raise ValueError("An optimization stage must be exactly first- or second-order")
        if self.second_order and self.dynamic_objective_allowed:
            raise ValueError("Second-order stages must freeze their dynamic objective")


@dataclass(frozen=True)
class CurriculumDescriptor:
    curriculum_id: str
    strategy_name: str
    axis_role: str
    level_ids: tuple[str, ...]
    initial_level_index: int
    final_level_index: int
    monotonic: bool = True
    single_step_advance_only: bool = True
    rollback_supported: bool = True
    force_advance_supported: bool = False
    associated_loss_component_ids: tuple[str, ...] = ()
    associated_sampler_ids: tuple[str, ...] = ()
    associated_variable_ids: tuple[str, ...] = ()
    mutable_during_first_order_stage: bool = True
    fixed_during_second_order_stage: bool = True
    validation_supported: bool = True
    state_snapshot_supported: bool = True
    critical_component_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "curriculum_id", str(self.curriculum_id))
        object.__setattr__(self, "strategy_name", str(self.strategy_name))
        object.__setattr__(self, "axis_role", str(self.axis_role))
        for name in (
            "level_ids", "associated_loss_component_ids", "associated_sampler_ids",
            "associated_variable_ids", "critical_component_ids",
        ):
            values = tuple(str(item) for item in getattr(self, name))
            if name != "level_ids":
                values = tuple(sorted(values))
            object.__setattr__(self, name, values)
        object.__setattr__(self, "metadata", _metadata(self.metadata))
        if not self.curriculum_id or not self.strategy_name:
            raise ValueError("Curriculum IDs and strategy names must not be empty")
        if self.axis_role not in CURRICULUM_AXIS_ROLES:
            raise ValueError(f"Unsupported curriculum axis role: {self.axis_role}")
        if len(self.level_ids) < 2:
            raise ValueError("Curriculum must declare at least two levels")
        if len(set(self.level_ids)) != len(self.level_ids):
            raise ValueError("Duplicate curriculum level IDs are forbidden")
        if any(not item for item in self.level_ids):
            raise ValueError("Curriculum level IDs must not be empty")
        if not 0 <= int(self.initial_level_index) < len(self.level_ids):
            raise ValueError("Curriculum initial level index is out of range")
        if not 0 <= int(self.final_level_index) < len(self.level_ids):
            raise ValueError("Curriculum final level index is out of range")
        if int(self.initial_level_index) >= int(self.final_level_index):
            raise ValueError("Curriculum final level must follow its initial level")
        if not self.monotonic or not self.single_step_advance_only:
            raise ValueError("The current curriculum protocol requires monotonic single-step advance")
        if self.rollback_supported and not self.state_snapshot_supported:
            raise ValueError("Curriculum rollback requires runtime state snapshots")
        if not set(self.critical_component_ids).issubset(
            set(self.associated_loss_component_ids)
        ):
            raise ValueError("Critical curriculum components must be associated losses")


@dataclass(frozen=True)
class TrainerCapabilities:
    supports_dynamic_loss_weighting: bool
    supports_resampling: bool
    supports_curriculum: bool
    supports_temporal_curriculum: bool
    supports_curriculum_advance: bool
    supports_curriculum_rollback: bool
    supports_forced_curriculum_advance: bool
    supports_stage_reallocation: bool
    supports_validation_probes: bool
    supports_component_gradient_diagnostics: bool
    supported_action_types: frozenset[str]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["supported_action_types"] = sorted(self.supported_action_types)
        return value


@dataclass(frozen=True)
class TrainingComponentRegistry:
    loss_components: tuple[LossComponentDescriptor, ...]
    sampling_components: tuple[SamplingComponentDescriptor, ...]
    variable_roles: tuple[VariableRoleDescriptor, ...]
    optimization_stages: tuple[OptimizationStageDescriptor, ...]
    capabilities: TrainerCapabilities
    curricula: tuple[CurriculumDescriptor, ...] = ()
    schema_version: str = "2.0"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "loss_components", tuple(sorted(self.loss_components, key=lambda item: item.component_id))
        )
        object.__setattr__(
            self, "sampling_components", tuple(sorted(self.sampling_components, key=lambda item: item.sampler_id))
        )
        object.__setattr__(
            self, "variable_roles", tuple(sorted(self.variable_roles, key=lambda item: item.variable_id))
        )
        object.__setattr__(
            self, "optimization_stages", tuple(sorted(self.optimization_stages, key=lambda item: item.order))
        )
        object.__setattr__(
            self, "curricula", tuple(sorted(self.curricula, key=lambda item: item.curriculum_id))
        )
        self._validate_unique("loss component", [item.component_id for item in self.loss_components])
        self._validate_unique("sampling component", [item.sampler_id for item in self.sampling_components])
        self._validate_unique("variable role", [item.variable_id for item in self.variable_roles])
        self._validate_unique("optimization stage", [item.stage_id for item in self.optimization_stages])
        self._validate_unique("curriculum", [item.curriculum_id for item in self.curricula])
        sampler_ids = {item.sampler_id for item in self.sampling_components}
        loss_ids = {item.component_id for item in self.loss_components}
        for component in self.loss_components:
            if component.sampler_id is not None and component.sampler_id not in sampler_ids:
                raise ValueError(
                    f"Loss component {component.component_id!r} references unknown sampler {component.sampler_id!r}"
                )
        for component in self.sampling_components:
            unknown = set(component.associated_loss_component_ids) - loss_ids
            if unknown:
                raise ValueError(
                    f"Sampling component {component.sampler_id!r} references unknown losses {sorted(unknown)}"
                )
        variable_ids = {item.variable_id for item in self.variable_roles}
        for curriculum in self.curricula:
            unknown_losses = set(curriculum.associated_loss_component_ids) - loss_ids
            unknown_samplers = set(curriculum.associated_sampler_ids) - sampler_ids
            unknown_variables = set(curriculum.associated_variable_ids) - variable_ids
            if unknown_losses:
                raise ValueError(
                    f"Curriculum {curriculum.curriculum_id!r} references unknown losses {sorted(unknown_losses)}"
                )
            if unknown_samplers:
                raise ValueError(
                    f"Curriculum {curriculum.curriculum_id!r} references unknown samplers {sorted(unknown_samplers)}"
                )
            if unknown_variables:
                raise ValueError(
                    f"Curriculum {curriculum.curriculum_id!r} references unknown variables {sorted(unknown_variables)}"
                )

    @staticmethod
    def _validate_unique(kind: str, values: list[str]) -> None:
        duplicates = sorted({value for value in values if values.count(value) > 1})
        if duplicates:
            raise ValueError(f"Duplicate {kind} IDs: {duplicates}")

    def stage(self, stage_id: str) -> OptimizationStageDescriptor:
        for item in self.optimization_stages:
            if item.stage_id == stage_id:
                return item
        raise KeyError(f"Unknown optimization stage: {stage_id}")

    def loss_component(self, component_id: str) -> LossComponentDescriptor:
        for item in self.loss_components:
            if item.component_id == component_id:
                return item
        raise KeyError(f"Unknown loss component: {component_id}")

    def active_weightable_component_ids(self, stage_id: str) -> tuple[str, ...]:
        return tuple(
            item.component_id
            for item in self.loss_components
            if item.trainable
            and item.dynamically_weightable
            and (not item.active_stages or stage_id in item.active_stages)
        )

    def to_dict(self) -> dict[str, Any]:
        def descriptor(item: Any) -> dict[str, Any]:
            # ``dataclasses.asdict`` deep-copies values and cannot copy an
            # immutable MappingProxyType.  Serialize field-by-field so the
            # public checkpoint/audit representation remains plain data.
            return {
                descriptor_field.name: (
                    dict(value)
                    if descriptor_field.name == "metadata"
                    else list(value)
                    if isinstance(value, tuple)
                    else value
                )
                for descriptor_field in fields(item)
                for value in (getattr(item, descriptor_field.name),)
            }

        return {
            "schema_version": self.schema_version,
            "loss_components": [descriptor(item) for item in self.loss_components],
            "sampling_components": [descriptor(item) for item in self.sampling_components],
            "variable_roles": [descriptor(item) for item in self.variable_roles],
            "optimization_stages": [descriptor(item) for item in self.optimization_stages],
            "curricula": [descriptor(item) for item in self.curricula],
            "capabilities": self.capabilities.to_dict(),
        }

    @property
    def component_ids(self) -> frozenset[str]:
        return frozenset(item.component_id for item in self.loss_components)

    @property
    def curriculum_ids(self) -> frozenset[str]:
        return frozenset(item.curriculum_id for item in self.curricula)

    def curriculum(self, curriculum_id: str) -> CurriculumDescriptor:
        for item in self.curricula:
            if item.curriculum_id == curriculum_id:
                return item
        raise KeyError(f"Unknown curriculum: {curriculum_id}")


def derive_trainer_capabilities(
    *,
    loss_components: tuple[LossComponentDescriptor, ...],
    sampling_components: tuple[SamplingComponentDescriptor, ...],
    variable_roles: tuple[VariableRoleDescriptor, ...],
    optimization_stages: tuple[OptimizationStageDescriptor, ...],
    curricula: tuple[CurriculumDescriptor, ...] = (),
) -> TrainerCapabilities:
    dynamic = any(item.trainable and item.dynamically_weightable for item in loss_components)
    resampling = any(item.replacement_supported for item in sampling_components)
    temporal = any(item.role == "temporal" and item.causal_ordered for item in variable_roles)
    curriculum_ready = tuple(
        item for item in curricula
        if item.state_snapshot_supported and item.validation_supported
    )
    curriculum_advance = any(
        item.mutable_during_first_order_stage
        and item.monotonic
        and item.single_step_advance_only
        and item.final_level_index > item.initial_level_index
        for item in curriculum_ready
    ) and any(item.first_order and item.dynamic_objective_allowed for item in optimization_stages)
    stage_reallocation = any(item.recoverable for item in optimization_stages)
    probes = any(item.validation_probe_supported for item in sampling_components) and any(
        item.validation_enabled for item in loss_components
    )
    actions = {"terminate_candidate", "rollback"}
    if dynamic:
        actions.add("update_loss_weights")
    if any(item.first_order for item in optimization_stages):
        actions.add("update_learning_rate")
    if stage_reallocation:
        actions.update({"shorten_optimization_stage", "start_recovery_stage"})
    if resampling:
        actions.add("replace_sampling_subset")
    if curriculum_advance:
        actions.add("advance_curriculum_state")
    return TrainerCapabilities(
        supports_dynamic_loss_weighting=dynamic,
        supports_resampling=resampling,
        supports_curriculum=bool(curriculum_ready),
        supports_temporal_curriculum=bool(
            temporal or any(item.axis_role == "temporal" for item in curricula)
        ),
        supports_curriculum_advance=curriculum_advance,
        supports_curriculum_rollback=any(
            item.rollback_supported and item.state_snapshot_supported
            for item in curriculum_ready
        ),
        supports_forced_curriculum_advance=any(
            item.force_advance_supported for item in curriculum_ready
        ),
        supports_stage_reallocation=stage_reallocation,
        supports_validation_probes=probes,
        supports_component_gradient_diagnostics=len(loss_components) >= 2,
        supported_action_types=frozenset(actions),
    )
