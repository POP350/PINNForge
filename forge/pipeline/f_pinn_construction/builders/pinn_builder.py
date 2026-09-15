"""All-in-one Open AlgorithmSpec numeric search, validation, and PyTorch construction.

This module combines:
1. maximal numeric option-registry expansion;
2. AlgorithmSpec normalization and static validation;
3. deterministic Trusted Builder construction;
4. a one-call validate-and-build facade.

It keeps the existing project-owned PDE problem and PhysicsBatch interfaces.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
import inspect
import math
from typing import Any, Callable

import torch
from torch import nn


from forge.pipeline.f_pinn_construction.physics_types import PhysicsBatch
from forge.pipeline.a_problem_definition.problems.sampling_budget import (
    BOUNDARY_ALIASES,
    EQUATION_ALIASES,
    INITIAL_ALIASES,
    SamplingContractError,
    constraint_category as sampling_constraint_category,
    constraint_sampling_identity,
    resolve_sampling_budget_plan,
)
from forge.pipeline.d_algorithm_generation.specs import REQUIRED_TOP_LEVEL_FIELDS
from forge.pipeline.d_algorithm_generation.specs.option_registry import (
    PINNSAGENT_DEPTHS,
    PINNSAGENT_LEARNING_RATES,
    PINNSAGENT_POINT_COUNTS,
    PINNSAGENT_WIDTHS,
    algorithm_spec_option_registry as _base_algorithm_spec_option_registry,
    option_parameter_names,
)

# ---------------------------------------------------------------------------
# Numeric option-registry expansion
# ---------------------------------------------------------------------------
def _search(scale: str, values: list[Any] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"scale": scale}
    if values is not None:
        result["recommended_values"] = values
    return result


def _number(
    *,
    minimum: float | None = None,
    exclusive_minimum: float | None = None,
    maximum: float | None = None,
    exclusive_maximum: float | None = None,
    default: float | None = None,
    scale: str = "linear",
    recommended: list[float] | None = None,
) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "number", "x-search": _search(scale, recommended)}
    if minimum is not None:
        schema["minimum"] = minimum
    if exclusive_minimum is not None:
        schema["exclusiveMinimum"] = exclusive_minimum
    if maximum is not None:
        schema["maximum"] = maximum
    if exclusive_maximum is not None:
        schema["exclusiveMaximum"] = exclusive_maximum
    if default is not None:
        schema["default"] = default
    return schema


def _integer(
    *,
    minimum: int | None = None,
    maximum: int | None = None,
    default: int | None = None,
    scale: str = "linear",
    recommended: list[int] | None = None,
) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "integer", "x-search": _search(scale, recommended)}
    if minimum is not None:
        schema["minimum"] = minimum
    if maximum is not None:
        schema["maximum"] = maximum
    if default is not None:
        schema["default"] = default
    return schema


def _boolean(default: bool | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "boolean"}
    if default is not None:
        schema["default"] = default
    return schema


def _string(*values: str, default: str | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "string"}
    if values:
        schema["enum"] = list(values)
    if default is not None:
        schema["default"] = default
    return schema


def _array(
    item_schema: dict[str, Any],
    *,
    min_items: int = 1,
    max_items: int | None = None,
    default: list[Any] | None = None,
    recommended_lengths: list[int] | None = None,
) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "array",
        "minItems": min_items,
        "items": item_schema,
    }
    if max_items is not None:
        schema["maxItems"] = max_items
    if default is not None:
        schema["default"] = default
    if recommended_lengths is not None:
        schema["x-search"] = {"recommended_lengths": list(recommended_lengths)}
    return schema


def _pair(item_schema: dict[str, Any], default: list[Any] | None = None) -> dict[str, Any]:
    return _array(item_schema, min_items=2, max_items=2, default=default)


def _triple(item_schema: dict[str, Any], default: list[Any] | None = None) -> dict[str, Any]:
    return _array(item_schema, min_items=3, max_items=3, default=default)


def _object(properties: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "additionalProperties": False}


POSITIVE_LOG = dict(exclusive_minimum=0.0, scale="log")
PROBABILITY = dict(minimum=0.0, maximum=1.0, scale="linear")
STRICT_PROBABILITY = dict(minimum=0.0, exclusive_maximum=1.0, scale="linear")


# Common schemas.
LEARNING_RATE = _number(
    exclusive_minimum=0.0,
    maximum=10.0,
    default=1e-3,
    scale="log",
    recommended=sorted(set(PINNSAGENT_LEARNING_RATES + [3e-5, 3e-4, 3e-3, 1.0])),
)
EPSILON = _number(
    exclusive_minimum=0.0,
    maximum=1.0,
    default=1e-8,
    scale="log",
    recommended=[1e-12, 1e-10, 1e-8, 1e-6, 1e-4],
)
WEIGHT_DECAY = _number(
    minimum=0.0,
    maximum=10.0,
    default=0.0,
    scale="log_zero",
    recommended=[0.0, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1],
)
BETA_PAIR = _pair(
    _number(minimum=0.0, exclusive_maximum=1.0),
    default=[0.9, 0.999],
)


COMMON_NETWORK_PARAMETERS: dict[str, Any] = {
    "dropout_rate": _number(
        minimum=0.0, exclusive_maximum=1.0, default=0.0,
        recommended=[0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5],
    ),
    "input_scale": _number(
        minimum=-1e6, maximum=1e6, default=1.0,
        recommended=[0.01, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0],
    ),
    "input_shift": _number(minimum=-1e6, maximum=1e6, default=0.0),
    "output_scale": _number(
        minimum=-1e12, maximum=1e12, default=1.0,
        recommended=[0.01, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0],
    ),
    "output_shift": _number(minimum=-1e12, maximum=1e12, default=0.0),
}


ARCHITECTURE_PARAMETERS: dict[str, dict[str, Any]] = {
    "mlp": {"bias": _boolean(True)},
    "resnet": {"bias": _boolean(True)},
    "residual_mlp": {"bias": _boolean(True)},
    "fourier_mlp": {
        "bias": _boolean(True),
        "num_frequencies": _integer(
            minimum=1, maximum=4096, default=8, scale="log2",
            recommended=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
        ),
        "sigma": _number(
            exclusive_minimum=0.0, maximum=1e4, default=1.0, scale="log",
            recommended=[0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0],
        ),
        "include_raw_input": _boolean(False),
        "trainable": _boolean(False),
        "frequency_mode": _string("linear", "log", "random_uniform", "random_normal", default="linear"),
        "min_frequency": _number(exclusive_minimum=0.0, maximum=1e4, default=1.0, scale="log"),
        "max_frequency": _number(exclusive_minimum=0.0, maximum=1e6, default=8.0, scale="log"),
        "seed": _integer(minimum=0, maximum=2**31 - 1, default=0),
        "use_pi": _boolean(True),
        "frequency_count": _integer(minimum=1, maximum=4096, default=8),
        "frequency_scale": _number(exclusive_minimum=0.0, maximum=1e4, default=1.0),
        "trainable_frequencies": _boolean(False),
    },
    "multiscale_fourier_mlp": {
        "bias": _boolean(True),
        "frequency_count": _integer(minimum=1, maximum=4096, default=8),
        "frequency_scale": _number(exclusive_minimum=0.0, maximum=1e4, default=1.0),
        "frequency_scales": _array(
            _number(exclusive_minimum=0.0, maximum=1e4),
            min_items=1,
            max_items=64,
            default=[1.0, 2.0, 4.0],
        ),
        "trainable_frequencies": _boolean(False),
        "include_raw_input": _boolean(False),
    },
    "siren_mlp": {
        "bias": _boolean(True),
        "omega_0": _number(exclusive_minimum=0.0, maximum=1e4, default=30.0),
        "hidden_omega_0": _number(exclusive_minimum=0.0, maximum=1e4, default=30.0),
        "initialization_scale": _number(exclusive_minimum=0.0, maximum=100.0, default=1.0),
    },
    "multiscale_mlp": {
        "bias": _boolean(True),
        "scales": _array(
            _number(exclusive_minimum=0.0, maximum=1e6, scale="log"),
            min_items=1,
            max_items=64,
            default=[1.0, 2.0, 4.0],
        ),
    },
    "modified_mlp": {
        "bias": _boolean(True),
        "gate_temperature": _number(
            exclusive_minimum=0.0, maximum=1e3, default=1.0, scale="log",
            recommended=[0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
        ),
    },
    "gated_mlp": {
        "bias": _boolean(True),
        "gate_temperature": _number(exclusive_minimum=0.0, maximum=1e3, default=1.0, scale="log"),
    },
    "factorized_mlp": {
        "bias": _boolean(True),
        "rank": _integer(
            minimum=1, maximum=4096, default=16, scale="log2",
            recommended=[1, 2, 4, 8, 16, 32, 64, 128, 256],
        ),
    },
    "laaf": {
        "bias": _boolean(True),
        "activation_scale": _number(exclusive_minimum=0.0, maximum=1e4, default=10.0, scale="log"),
        "initial_slope": _number(exclusive_minimum=0.0, maximum=1e3, default=1.0, scale="log"),
    },
    "laaf_mlp": {
        "bias": _boolean(True),
        "activation_scale": _number(exclusive_minimum=0.0, maximum=1e4, default=10.0, scale="log"),
        "initial_slope": _number(exclusive_minimum=0.0, maximum=1e3, default=1.0, scale="log"),
    },
    "gaaf": {
        "bias": _boolean(True),
        "activation_scale": _number(exclusive_minimum=0.0, maximum=1e4, default=10.0, scale="log"),
        "initial_slope": _number(exclusive_minimum=0.0, maximum=1e3, default=1.0, scale="log"),
    },
    "gaaf_mlp": {
        "bias": _boolean(True),
        "activation_scale": _number(exclusive_minimum=0.0, maximum=1e4, default=10.0, scale="log"),
        "initial_slope": _number(exclusive_minimum=0.0, maximum=1e3, default=1.0, scale="log"),
    },
    "parallel_fnn": {"bias": _boolean(True)},
    "pfnn": {"bias": _boolean(True)},
}


for _architecture_parameters in ARCHITECTURE_PARAMETERS.values():
    for _name, _schema in COMMON_NETWORK_PARAMETERS.items():
        _architecture_parameters.setdefault(_name, deepcopy(_schema))


ACTIVATION_PARAMETERS: dict[str, dict[str, Any]] = {
    "identity": {},
    "tanh": {},
    "silu": {"inplace": _boolean(False)},
    "swish": {"inplace": _boolean(False)},
    "gelu": {},
    "relu": {"inplace": _boolean(False)},
    "leaky_relu": {
        "negative_slope": _number(minimum=0.0, maximum=10.0, default=0.01, scale="log_zero",
                                  recommended=[0.0, 1e-4, 1e-3, 0.01, 0.05, 0.1, 0.2, 0.5]),
        "inplace": _boolean(False),
    },
    "elu": {
        "alpha": _number(exclusive_minimum=0.0, maximum=100.0, default=1.0, scale="log"),
        "inplace": _boolean(False),
    },
    "softplus": {
        "beta": _number(exclusive_minimum=0.0, maximum=1e4, default=1.0, scale="log"),
        "threshold": _number(exclusive_minimum=0.0, maximum=1e5, default=20.0, scale="log"),
    },
    "sigmoid": {},
    "mish": {"inplace": _boolean(False)},
    "softsign": {},
    "tanhshrink": {},
    "hardtanh": {
        "min_val": _number(minimum=-1e6, maximum=1e6, default=-1.0),
        "max_val": _number(minimum=-1e6, maximum=1e6, default=1.0),
        "inplace": _boolean(False),
    },
    "celu": {
        "alpha": _number(exclusive_minimum=0.0, maximum=100.0, default=1.0, scale="log"),
        "inplace": _boolean(False),
    },
    "selu": {"inplace": _boolean(False)},
    # A shared activation instance is reused across layers, so num_parameters=1
    # is the universally shape-safe PReLU setting.
    "prelu": {
        "num_parameters": _integer(minimum=1, maximum=1, default=1),
        "init": _number(minimum=-100.0, maximum=100.0, default=0.25),
    },
    "relu6": {"inplace": _boolean(False)},
    "hardswish": {"inplace": _boolean(False)},
    "hard_swish": {"inplace": _boolean(False)},
    "hardsigmoid": {"inplace": _boolean(False)},
    "hard_sigmoid": {"inplace": _boolean(False)},
    "log_sigmoid": {},
    "softshrink": {"lambd": _number(minimum=0.0, maximum=1e4, default=0.5, scale="log_zero")},
    "hardshrink": {"lambd": _number(minimum=0.0, maximum=1e4, default=0.5, scale="log_zero")},
    "threshold": {
        "threshold": _number(minimum=-1e6, maximum=1e6, default=0.0),
        "value": _number(minimum=-1e6, maximum=1e6, default=0.0),
        "inplace": _boolean(False),
    },
    "rrelu": {
        "lower": _number(minimum=0.0, maximum=10.0, default=0.125),
        "upper": _number(minimum=0.0, maximum=10.0, default=1.0 / 3.0),
        "inplace": _boolean(False),
    },
    "randomized_leaky_relu": {
        "lower": _number(minimum=0.0, maximum=10.0, default=0.125),
        "upper": _number(minimum=0.0, maximum=10.0, default=1.0 / 3.0),
        "inplace": _boolean(False),
    },
    "sine": {
        "omega": _number(exclusive_minimum=0.0, maximum=1e5, default=1.0, scale="log",
                         recommended=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 50.0, 100.0]),
    },
    "sin": {"omega": _number(exclusive_minimum=0.0, maximum=1e5, default=1.0, scale="log")},
    "cosine": {
        "omega": _number(exclusive_minimum=0.0, maximum=1e5, default=1.0, scale="log"),
        "phase": _number(minimum=-1e4, maximum=1e4, default=0.0),
    },
    "cos": {
        "omega": _number(exclusive_minimum=0.0, maximum=1e5, default=1.0, scale="log"),
        "phase": _number(minimum=-1e4, maximum=1e4, default=0.0),
    },
    "gaussian": {
        "beta": _number(exclusive_minimum=0.0, maximum=1e5, default=1.0, scale="log"),
    },
    "laaf": {
        "initial_slope": _number(exclusive_minimum=0.0, maximum=1e3, default=1.0, scale="log"),
        "activation_scale": _number(exclusive_minimum=0.0, maximum=1e4, default=10.0, scale="log"),
    },
    "scaled_tanh": {
        "scale": _number(exclusive_minimum=0.0, maximum=1e5, default=1.0, scale="log"),
        "trainable": _boolean(False),
    },
    "snake": {
        "alpha": _number(exclusive_minimum=0.0, maximum=1e5, default=1.0, scale="log"),
        "trainable": _boolean(True),
        "epsilon": EPSILON,
    },
    "stan": {"initial_beta": _number(minimum=-1e3, maximum=1e3, default=1.0)},
}


INITIALIZER_PARAMETERS: dict[str, dict[str, Any]] = {
    "xavier_normal": {"gain": _number(exclusive_minimum=0.0, maximum=1e3, default=1.0, scale="log")},
    "xavier_uniform": {"gain": _number(exclusive_minimum=0.0, maximum=1e3, default=1.0, scale="log")},
    "glorot_normal": {"gain": _number(exclusive_minimum=0.0, maximum=1e3, default=1.0, scale="log")},
    "glorot_uniform": {"gain": _number(exclusive_minimum=0.0, maximum=1e3, default=1.0, scale="log")},
    "kaiming_normal": {
        "a": _number(minimum=0.0, maximum=100.0, default=0.0, scale="log_zero"),
        "mode": _string("fan_in", "fan_out", default="fan_in"),
        "nonlinearity": _string("linear", "sigmoid", "tanh", "relu", "leaky_relu", "selu", default="leaky_relu"),
    },
    "kaiming_uniform": {
        "a": _number(minimum=0.0, maximum=100.0, default=0.0, scale="log_zero"),
        "mode": _string("fan_in", "fan_out", default="fan_in"),
        "nonlinearity": _string("linear", "sigmoid", "tanh", "relu", "leaky_relu", "selu", default="leaky_relu"),
    },
    "orthogonal": {"gain": _number(exclusive_minimum=0.0, maximum=1e3, default=1.0, scale="log")},
    "normal": {
        "mean": _number(minimum=-1e3, maximum=1e3, default=0.0),
        "std": _number(exclusive_minimum=0.0, maximum=1e3, default=1.0, scale="log"),
    },
    "uniform": {
        "a": _number(minimum=-1e6, maximum=1e6, default=0.0),
        "b": _number(minimum=-1e6, maximum=1e6, default=1.0),
    },
    "trunc_normal": {
        "mean": _number(minimum=-1e3, maximum=1e3, default=0.0),
        "std": _number(exclusive_minimum=0.0, maximum=1e3, default=1.0, scale="log"),
        "a": _number(minimum=-1e6, maximum=1e6, default=-2.0),
        "b": _number(minimum=-1e6, maximum=1e6, default=2.0),
    },
    "sparse": {
        "sparsity": _number(minimum=0.0, maximum=1.0, default=0.1),
        "std": _number(exclusive_minimum=0.0, maximum=1e3, default=0.01, scale="log"),
    },
    "zeros": {},
    "ones": {},
    "constant": {"val": _number(minimum=-1e6, maximum=1e6, default=0.0)},
    "eye": {},
    "siren": {
        "initialization_scale": _number(
            exclusive_minimum=0.0, maximum=100.0, default=1.0
        )
    },
}

FOURIER_REPRESENTATION_PARAMETERS: dict[str, dict[str, Any]] = {
    "fourier_features": {
        "frequency_count": _integer(minimum=1, maximum=4096, default=8),
        "frequency_scale": _number(exclusive_minimum=0.0, maximum=1e4, default=1.0),
        "trainable_frequencies": _boolean(False),
        "include_raw_input": _boolean(False),
    },
    "multiscale_fourier_features": {
        "frequency_count": _integer(minimum=1, maximum=4096, default=8),
        "frequency_scale": _number(exclusive_minimum=0.0, maximum=1e4, default=1.0),
        "frequency_scales": _array(
            _number(exclusive_minimum=0.0, maximum=1e4),
            min_items=1,
            max_items=64,
            default=[1.0, 2.0, 4.0],
        ),
        "trainable_frequencies": _boolean(False),
        "include_raw_input": _boolean(False),
    },
    "periodic_positional_encoding": {
        "frequency_count": _integer(minimum=1, maximum=4096, default=8),
        "frequency_scale": _number(exclusive_minimum=0.0, maximum=1e4, default=1.0),
        "trainable_frequencies": _boolean(False),
        "include_raw_input": _boolean(True),
    },
}


SAMPLER_PARAMETERS: dict[str, dict[str, Any]] = {
    "uniform": {"n_points": _integer(minimum=0, default=5000, scale="log", recommended=PINNSAGENT_POINT_COUNTS)},
    "latin_hypercube": {
        "n_points": _integer(minimum=0, default=5000, scale="log", recommended=PINNSAGENT_POINT_COUNTS),
        "centered": _boolean(False),
    },
    "sobol": {
        "n_points": _integer(minimum=0, default=5000, scale="log", recommended=PINNSAGENT_POINT_COUNTS),
        "scramble": _boolean(True),
        "seed": _integer(minimum=0, maximum=2**31 - 1, default=0),
        "skip": _integer(minimum=0, maximum=1_000_000_000, default=0, scale="log_zero"),
    },
    "halton": {
        "n_points": _integer(minimum=0, default=5000, scale="log", recommended=PINNSAGENT_POINT_COUNTS),
        "start_index": _integer(minimum=1, maximum=1_000_000_000, default=1, scale="log"),
    },
    "hammersley": {
        "n_points": _integer(minimum=0, default=5000, scale="log", recommended=PINNSAGENT_POINT_COUNTS),
        "start_index": _integer(minimum=1, maximum=1_000_000_000, default=1, scale="log"),
    },
    "grid": {
        "n_points": _integer(minimum=0, default=5000, scale="log", recommended=PINNSAGENT_POINT_COUNTS),
        "points_per_axis": _integer(minimum=2, maximum=1_000_000, default=64, scale="log2"),
        "shuffle": _boolean(False),
    },
}

CONSTRAINT_SAMPLER_PARAMETERS = {
    "uniform": {"n_points": _integer(minimum=0, default=256, scale="log", recommended=PINNSAGENT_POINT_COUNTS)}
}

ADAPTIVE_PARAMETERS = {
    "candidate_points": _integer(
        minimum=1, default=1024, scale="log",
        recommended=[256, 512, 1024, 2048, 4096, 8192, 16384, 65536],
    ),
    "add_points": _integer(
        minimum=1, default=64, scale="log",
        recommended=[16, 32, 64, 128, 256, 512, 1024, 4096],
    ),
    "update_every": _integer(
        minimum=1, maximum=10_000_000, default=500, scale="log",
        recommended=[10, 25, 50, 100, 250, 500, 1000, 2500],
    ),
    "max_points": _integer(minimum=0, default=0, scale="log_zero"),
    "residual_exponent": _number(exclusive_minimum=0.0, maximum=100.0, default=2.0, scale="log"),
    "distribution_offset": _number(minimum=0.0, maximum=1e6, default=1.0, scale="log_zero"),
    "residual_component": _string(),
    "gradient_input_indices": _array(_integer(minimum=0, maximum=1024), min_items=1, max_items=1024),
    "residual_weight": _number(minimum=0.0, maximum=100.0, default=1.0, scale="log_zero"),
    "gradient_weight": _number(minimum=0.0, maximum=100.0, default=1.0, scale="log_zero"),
    "hybrid_mode": _string("product", "weighted_sum", default="product"),
    "replace_existing": _boolean(False),
}


GENERIC_LOSS_TERM_PARAMETERS: dict[str, Any] = {
    "component_index": _integer(minimum=0, maximum=1_000_000, default=0),
    "component_name": _string(),
    "input_indices": _array(_integer(minimum=0, maximum=1024), min_items=1, max_items=1024),
    "derivative_variable": _string(),
    "derivative_order": _integer(minimum=0, maximum=16, default=0),
}


LOSS_FUNCTION_PARAMETERS: dict[str, dict[str, Any]] = {
    "mse": {},
    "mae": {},
    "rmse": {"epsilon": EPSILON},
    "huber": {"delta": _number(exclusive_minimum=0.0, maximum=1e6, default=1.0, scale="log")},
    "smooth_l1": {"beta": _number(exclusive_minimum=0.0, maximum=1e6, default=1.0, scale="log")},
    "log_cosh": {"scale": _number(exclusive_minimum=0.0, maximum=1e6, default=1.0, scale="log")},
    "cauchy": {"scale": _number(exclusive_minimum=0.0, maximum=1e6, default=1.0, scale="log")},
    "charbonnier": {
        "epsilon": _number(exclusive_minimum=0.0, maximum=1e3, default=1e-3, scale="log"),
        "alpha": _number(exclusive_minimum=0.0, maximum=10.0, default=0.5, scale="log"),
    },
    "pseudo_huber": {"delta": _number(exclusive_minimum=0.0, maximum=1e6, default=1.0, scale="log")},
    "lp": {
        "p": _number(exclusive_minimum=0.0, maximum=100.0, default=2.0, scale="log"),
        "epsilon": EPSILON,
        "root": _boolean(False),
    },
    "tukey": {"c": _number(exclusive_minimum=0.0, maximum=1e6, default=4.685, scale="log")},
    "geman_mcclure": {"scale": _number(exclusive_minimum=0.0, maximum=1e6, default=1.0, scale="log")},
    "log_l2": {"epsilon": EPSILON},
    "quantile": {"quantile": _number(exclusive_minimum=0.0, exclusive_maximum=1.0, default=0.5)},
}


# The current AlgorithmSpec stores loss-function and loss-term parameters in the
# same object.  Expose residual-component selectors for every reducer; the
# Builder still enforces which selectors are legal for each loss term name.
for _loss_parameters in LOSS_FUNCTION_PARAMETERS.values():
    for _name, _schema in GENERIC_LOSS_TERM_PARAMETERS.items():
        _loss_parameters.setdefault(_name, deepcopy(_schema))


WEIGHTING_PARAMETERS: dict[str, dict[str, Any]] = {
    "fixed": {},
    "normalized": {"epsilon": EPSILON},
    "softmax": {
        "temperature": _number(exclusive_minimum=0.0, maximum=1e6, default=1.0, scale="log")
    },
    "inverse_magnitude": {
        "epsilon": EPSILON,
        "power": _number(exclusive_minimum=0.0, maximum=100.0, default=1.0, scale="log"),
        "normalize": _boolean(True),
    },
    "lra": {
        "alpha": _number(minimum=0.0, maximum=1.0, default=0.1),
        "epsilon": EPSILON,
        "reference_index": _integer(minimum=0, maximum=1_000_000, default=0),
        "min_weight": _number(exclusive_minimum=0.0, maximum=1e12, default=0.01, scale="log"),
        "max_weight": _number(exclusive_minimum=0.0, maximum=1e18, default=20.0, scale="log"),
        "minimum_update_ratio": _number(exclusive_minimum=0.0, maximum=1.0, default=0.5),
        "maximum_update_ratio": _number(minimum=1.0, maximum=100.0, default=2.0),
        "cooldown_iterations": _integer(minimum=0, maximum=10_000_000, default=0),
    },
    "gradient_balance": {
        "ema_beta": _number(minimum=0.0, exclusive_maximum=1.0, default=0.95),
        "diagnostic_interval": _integer(minimum=1, maximum=1_000_000, default=50),
        "patience": _integer(minimum=1, maximum=1_000_000, default=3),
        "cooldown_iterations": _integer(minimum=0, maximum=10_000_000, default=200),
        "imbalance_threshold": _number(exclusive_minimum=0.0, maximum=1e18, default=100.0, scale="log"),
        "conflict_threshold": _number(minimum=-1.0, maximum=1.0, default=-0.1),
        "minimum_weight": _number(exclusive_minimum=0.0, maximum=1e12, default=0.01, scale="log"),
        "maximum_weight": _number(exclusive_minimum=0.0, maximum=1e12, default=20.0, scale="log"),
        "minimum_update_ratio": _number(exclusive_minimum=0.0, maximum=1.0, default=0.5),
        "maximum_update_ratio": _number(minimum=1.0, maximum=100.0, default=2.0),
        "alpha": _number(minimum=0.0, maximum=1.0, default=0.5),
        "epsilon": EPSILON,
    },
    "ntk": {
        "epsilon": _number(exclusive_minimum=0.0, maximum=1.0, default=1e-12, scale="log"),
        "min_weight": _number(exclusive_minimum=0.0, maximum=1e12, default=1e-6, scale="log"),
        "max_weight": _number(exclusive_minimum=0.0, maximum=1e18, default=1e6, scale="log"),
        "update_every": _integer(minimum=1, maximum=1_000_000, default=1),
    },
    "ntk_trace": {
        "alpha": _number(minimum=0.0, maximum=1.0, default=0.1),
        "epsilon": EPSILON,
        "normalize": _boolean(True),
    },
    "ema_inverse_magnitude": {
        "alpha": _number(minimum=0.0, maximum=1.0, default=0.1),
        "epsilon": EPSILON,
        "power": _number(exclusive_minimum=0.0, maximum=100.0, default=1.0, scale="log"),
        "normalize": _boolean(True),
    },
    "loss_softmax": {
        "temperature": _number(exclusive_minimum=0.0, maximum=1e6, default=1.0, scale="log"),
        "normalize": _boolean(True),
    },
}


AGGREGATION_PARAMETERS: dict[str, dict[str, Any]] = {
    "weighted_sum": {},
    "mean": {},
    "max": {},
    "logsumexp": {"temperature": _number(exclusive_minimum=0.0, maximum=1e6, default=1.0, scale="log")},
    "p_norm": {
        "p": _number(exclusive_minimum=0.0, maximum=100.0, default=2.0, scale="log"),
        "epsilon": EPSILON,
    },
    "root_mean_square": {"epsilon": EPSILON},
}


OPTIMIZER_PARAMETERS: dict[str, dict[str, Any]] = {
    "adam": {
        "learning_rate": LEARNING_RATE, "betas": BETA_PAIR, "eps": EPSILON,
        "weight_decay": WEIGHT_DECAY, "amsgrad": _boolean(False), "maximize": _boolean(False),
        "foreach": _boolean(), "capturable": _boolean(False), "differentiable": _boolean(False),
        "fused": _boolean(), "decoupled_weight_decay": _boolean(False),
    },
    "adamw": {
        "learning_rate": LEARNING_RATE, "betas": BETA_PAIR, "eps": EPSILON,
        "weight_decay": _number(minimum=0.0, maximum=10.0, default=0.01, scale="log_zero"),
        "amsgrad": _boolean(False), "maximize": _boolean(False), "foreach": _boolean(),
        "capturable": _boolean(False), "differentiable": _boolean(False), "fused": _boolean(),
    },
    "sgd": {
        "learning_rate": LEARNING_RATE,
        "momentum": _number(minimum=0.0, exclusive_maximum=1.0, default=0.0,
                            recommended=[0.0, 0.5, 0.8, 0.9, 0.95, 0.99]),
        "dampening": _number(minimum=0.0, exclusive_maximum=1.0, default=0.0),
        "weight_decay": WEIGHT_DECAY, "nesterov": _boolean(False), "maximize": _boolean(False),
        "foreach": _boolean(), "differentiable": _boolean(False), "fused": _boolean(),
    },
    "rmsprop": {
        "learning_rate": LEARNING_RATE,
        "alpha": _number(minimum=0.0, exclusive_maximum=1.0, default=0.99),
        "eps": EPSILON, "weight_decay": WEIGHT_DECAY,
        "momentum": _number(minimum=0.0, exclusive_maximum=1.0, default=0.0),
        "centered": _boolean(False), "capturable": _boolean(False), "foreach": _boolean(),
        "maximize": _boolean(False), "differentiable": _boolean(False),
    },
    "radam": {
        "learning_rate": LEARNING_RATE, "betas": BETA_PAIR, "eps": EPSILON,
        "weight_decay": WEIGHT_DECAY, "decoupled_weight_decay": _boolean(False),
        "foreach": _boolean(), "maximize": _boolean(False), "capturable": _boolean(False),
        "differentiable": _boolean(False),
    },
    "nadam": {
        "learning_rate": LEARNING_RATE, "betas": BETA_PAIR, "eps": EPSILON,
        "weight_decay": WEIGHT_DECAY,
        "momentum_decay": _number(minimum=0.0, maximum=1.0, default=0.004, scale="log_zero"),
        "decoupled_weight_decay": _boolean(False), "foreach": _boolean(), "maximize": _boolean(False),
        "capturable": _boolean(False), "differentiable": _boolean(False),
    },
    "adagrad": {
        "learning_rate": LEARNING_RATE,
        "lr_decay": _number(minimum=0.0, maximum=1e6, default=0.0, scale="log_zero"),
        "weight_decay": WEIGHT_DECAY,
        "initial_accumulator_value": _number(minimum=0.0, maximum=1e6, default=0.0, scale="log_zero"),
        "eps": EPSILON, "foreach": _boolean(), "maximize": _boolean(False),
        "differentiable": _boolean(False), "fused": _boolean(),
    },
    "adadelta": {
        "learning_rate": _number(exclusive_minimum=0.0, maximum=100.0, default=1.0, scale="log"),
        "rho": _number(minimum=0.0, exclusive_maximum=1.0, default=0.9),
        "eps": EPSILON, "weight_decay": WEIGHT_DECAY, "foreach": _boolean(),
        "capturable": _boolean(False), "maximize": _boolean(False), "differentiable": _boolean(False),
    },
    "adamax": {
        "learning_rate": LEARNING_RATE, "betas": BETA_PAIR, "eps": EPSILON,
        "weight_decay": WEIGHT_DECAY, "foreach": _boolean(), "maximize": _boolean(False),
        "differentiable": _boolean(False), "capturable": _boolean(False),
    },
    "asgd": {
        "learning_rate": LEARNING_RATE,
        "lambd": _number(minimum=0.0, maximum=1e6, default=1e-4, scale="log_zero"),
        "alpha": _number(minimum=0.0, maximum=10.0, default=0.75),
        "t0": _number(minimum=0.0, maximum=1e15, default=1e6, scale="log_zero"),
        "weight_decay": WEIGHT_DECAY, "foreach": _boolean(), "maximize": _boolean(False),
        "differentiable": _boolean(False), "capturable": _boolean(False),
    },
    "rprop": {
        "learning_rate": LEARNING_RATE,
        "etas": _pair(_number(exclusive_minimum=0.0, maximum=10.0), default=[0.5, 1.2]),
        "step_sizes": _pair(_number(exclusive_minimum=0.0, maximum=1e12, scale="log"), default=[1e-6, 50.0]),
        "capturable": _boolean(False), "foreach": _boolean(), "maximize": _boolean(False),
        "differentiable": _boolean(False),
    },
    "multiadam": {
        "learning_rate": LEARNING_RATE, "betas": BETA_PAIR, "eps": EPSILON,
        "weight_decay": WEIGHT_DECAY,
        "group_weights": _array(_number(minimum=-1e6, maximum=1e6), min_items=2, max_items=10_000),
        "normalize_updates": _boolean(False),
        "grouping": _string(
            "pde_vs_constraints",
            "dirichlet_vs_non_dirichlet",
            default="pde_vs_constraints",
        ),
    },
    "lbfgs": {
        "learning_rate": _number(
            exclusive_minimum=0.0,
            maximum=100.0,
            default=1.0,
            scale="log",
            recommended=sorted(set(PINNSAGENT_LEARNING_RATES + [1.0])),
        ),
        "max_iter": _integer(minimum=1, maximum=10_000_000, default=20, scale="log"),
        "max_eval": _integer(minimum=1, maximum=100_000_000, default=25, scale="log"),
        "tolerance_grad": _number(exclusive_minimum=0.0, maximum=1e3, default=1e-7, scale="log"),
        "tolerance_change": _number(exclusive_minimum=0.0, maximum=1e3, default=1e-9, scale="log"),
        "history_size": _integer(minimum=1, maximum=100_000, default=100, scale="log"),
        "line_search_fn": _string("strong_wolfe", default="strong_wolfe"),
    },
    # Registered conditionally when the installed PyTorch provides it.
    "adafactor": {
        "learning_rate": _number(exclusive_minimum=0.0, maximum=100.0, default=0.01, scale="log"),
        "beta2_decay": _number(minimum=-100.0, maximum=0.0, default=-0.8),
        "eps": _pair(_number(minimum=0.0, maximum=1.0, scale="log_zero"), default=[1e-30, 1e-3]),
        "d": _number(exclusive_minimum=0.0, maximum=1e6, default=1.0, scale="log"),
        "weight_decay": WEIGHT_DECAY, "foreach": _boolean(), "maximize": _boolean(False),
    },
}


SCHEDULER_PARAMETERS: dict[str, dict[str, Any]] = {
    "none": {},
    "step_lr": {
        "step_size": _integer(minimum=1, maximum=100_000_000, default=1000, scale="log"),
        "gamma": _number(exclusive_minimum=0.0, maximum=10.0, default=0.1, scale="log"),
        "last_epoch": _integer(minimum=-1, maximum=100_000_000, default=-1),
    },
    "cosine_annealing": {
        "T_max": _integer(minimum=1, maximum=100_000_000, default=1000, scale="log"),
        "eta_min": _number(minimum=0.0, maximum=100.0, default=0.0, scale="log_zero"),
        "last_epoch": _integer(minimum=-1, maximum=100_000_000, default=-1),
    },
    "cosine_warm_restarts": {
        "T_0": _integer(minimum=1, maximum=100_000_000, default=1000, scale="log"),
        "T_mult": _integer(minimum=1, maximum=1_000_000, default=1, scale="log"),
        "eta_min": _number(minimum=0.0, maximum=100.0, default=0.0, scale="log_zero"),
        "last_epoch": _integer(minimum=-1, maximum=100_000_000, default=-1),
    },
    "exponential_lr": {
        "gamma": _number(exclusive_minimum=0.0, maximum=10.0, default=0.99, scale="log"),
        "last_epoch": _integer(minimum=-1, maximum=100_000_000, default=-1),
    },
    "reduce_on_plateau": {
        "mode": _string("min", "max", default="min"),
        "factor": _number(exclusive_minimum=0.0, exclusive_maximum=1.0, default=0.1, scale="log"),
        "patience": _integer(minimum=0, maximum=10_000_000, default=10, scale="log_zero"),
        "threshold": _number(minimum=0.0, maximum=1e6, default=1e-4, scale="log_zero"),
        "threshold_mode": _string("rel", "abs", default="rel"),
        "cooldown": _integer(minimum=0, maximum=10_000_000, default=0, scale="log_zero"),
        "min_lr": _number(minimum=0.0, maximum=100.0, default=0.0, scale="log_zero"),
        "eps": EPSILON,
    },
    "one_cycle": {
        "max_lr": _number(exclusive_minimum=0.0, maximum=100.0, default=1e-3, scale="log"),
        "total_steps": _integer(minimum=1, maximum=100_000_000, default=1000, scale="log"),
        "epochs": _integer(minimum=1, maximum=10_000_000, scale="log"),
        "steps_per_epoch": _integer(minimum=1, maximum=100_000_000, scale="log"),
        "pct_start": _number(exclusive_minimum=0.0, exclusive_maximum=1.0, default=0.3),
        "anneal_strategy": _string("cos", "linear", default="cos"),
        "cycle_momentum": _boolean(True),
        "base_momentum": _number(minimum=0.0, exclusive_maximum=1.0, default=0.85),
        "max_momentum": _number(minimum=0.0, exclusive_maximum=1.0, default=0.95),
        "div_factor": _number(exclusive_minimum=0.0, maximum=1e12, default=25.0, scale="log"),
        "final_div_factor": _number(exclusive_minimum=0.0, maximum=1e18, default=1e4, scale="log"),
        "three_phase": _boolean(False),
        "last_epoch": _integer(minimum=-1, maximum=100_000_000, default=-1),
    },
    "linear_lr": {
        "start_factor": _number(exclusive_minimum=0.0, maximum=1.0, default=1.0 / 3.0),
        "end_factor": _number(exclusive_minimum=0.0, maximum=1.0, default=1.0),
        "total_iters": _integer(minimum=1, maximum=100_000_000, default=5, scale="log"),
        "last_epoch": _integer(minimum=-1, maximum=100_000_000, default=-1),
    },
    "polynomial_lr": {
        "total_iters": _integer(minimum=1, maximum=100_000_000, default=5, scale="log"),
        "power": _number(exclusive_minimum=0.0, maximum=100.0, default=1.0, scale="log"),
        "last_epoch": _integer(minimum=-1, maximum=100_000_000, default=-1),
    },
    "constant_lr": {
        "factor": _number(exclusive_minimum=0.0, maximum=1.0, default=1.0 / 3.0),
        "total_iters": _integer(minimum=1, maximum=100_000_000, default=5, scale="log"),
        "last_epoch": _integer(minimum=-1, maximum=100_000_000, default=-1),
    },
    "multi_step_lr": {
        "milestones": _array(_integer(minimum=1, maximum=100_000_000), min_items=1, max_items=100_000),
        "gamma": _number(exclusive_minimum=0.0, maximum=10.0, default=0.1, scale="log"),
        "last_epoch": _integer(minimum=-1, maximum=100_000_000, default=-1),
    },
    "cyclic_lr": {
        "base_lr": _number(exclusive_minimum=0.0, maximum=100.0, default=1e-4, scale="log"),
        "max_lr": _number(exclusive_minimum=0.0, maximum=100.0, default=1e-3, scale="log"),
        "step_size_up": _integer(minimum=1, maximum=100_000_000, default=2000, scale="log"),
        "step_size_down": _integer(minimum=1, maximum=100_000_000, default=2000, scale="log"),
        "mode": _string("triangular", "triangular2", "exp_range", default="triangular"),
        "gamma": _number(exclusive_minimum=0.0, maximum=10.0, default=1.0, scale="log"),
        "scale_mode": _string("cycle", "iterations", default="cycle"),
        "cycle_momentum": _boolean(True),
        "base_momentum": _number(minimum=0.0, exclusive_maximum=1.0, default=0.8),
        "max_momentum": _number(minimum=0.0, exclusive_maximum=1.0, default=0.9),
        "last_epoch": _integer(minimum=-1, maximum=100_000_000, default=-1),
    },
}

SCHEDULER_ALIASES = {
    "steplr": "step_lr",
    "cosineannealinglr": "cosine_annealing",
    "cosine_annealing_warm_restarts": "cosine_warm_restarts",
    "cosineannealingwarmrestarts": "cosine_warm_restarts",
    "exponentiallr": "exponential_lr",
    "reducelronplateau": "reduce_on_plateau",
    "onecyclelr": "one_cycle",
}


COMPATIBILITY_RULES: dict[tuple[str, str], list[dict[str, Any]]] = {
    ("network.architecture", "fourier_mlp"): [
        {"rule": "parameter_order", "lower": "min_frequency", "upper": "max_frequency"},
    ],
    ("network.activation.name", "hardtanh"): [
        {"rule": "parameter_order", "lower": "min_val", "upper": "max_val"},
    ],
    ("network.output_activation.name", "hardtanh"): [
        {"rule": "parameter_order", "lower": "min_val", "upper": "max_val"},
    ],
    ("network.activation.name", "rrelu"): [
        {"rule": "parameter_order", "lower": "lower", "upper": "upper"},
    ],
    ("network.activation.name", "randomized_leaky_relu"): [
        {"rule": "parameter_order", "lower": "lower", "upper": "upper"},
    ],
    ("network.output_activation.name", "rrelu"): [
        {"rule": "parameter_order", "lower": "lower", "upper": "upper"},
    ],
    ("network.initialization.name", "uniform"): [
        {"rule": "parameter_order", "lower": "a", "upper": "b"},
    ],
    ("network.initialization.name", "trunc_normal"): [
        {"rule": "parameter_order", "lower": "a", "upper": "b"},
    ],
    ("loss.weighting_strategy.name", "lra"): [
        {"rule": "parameter_order", "lower": "min_weight", "upper": "max_weight"},
    ],
    ("loss.weighting_strategy.name", "ntk"): [
        {"rule": "parameter_order", "lower": "min_weight", "upper": "max_weight"},
    ],
    ("optimization.phases[].optimizer", "rprop"): [
        {"rule": "array_element_order", "parameter": "step_sizes", "lower_index": 0, "upper_index": 1},
        {"rule": "array_straddles", "parameter": "etas", "pivot": 1.0},
    ],
    ("optimization.phases[].scheduler.name", "one_cycle"): [
        {"rule": "parameter_order", "lower": "base_momentum", "upper": "max_momentum"},
        {"rule": "paired_parameters", "first": "epochs", "second": "steps_per_epoch"},
    ],
    ("optimization.phases[].scheduler.name", "cyclic_lr"): [
        {"rule": "parameter_order", "lower": "base_lr", "upper": "max_lr"},
        {"rule": "parameter_order", "lower": "base_momentum", "upper": "max_momentum"},
    ],
}


OPTION_PATCHES: dict[str, dict[str, dict[str, Any]]] = {
    "network.architecture": ARCHITECTURE_PARAMETERS,
    "network.activation.name": ACTIVATION_PARAMETERS,
    "network.output_activation.name": deepcopy(ACTIVATION_PARAMETERS),
    "network.initialization.name": INITIALIZER_PARAMETERS,
    "network.input_transform.name": {
        "none": {},
        "normalize_bounds": {},
        "unit_bounds": {},
        **FOURIER_REPRESENTATION_PARAMETERS,
    },
    "sampling.interior.strategy": SAMPLER_PARAMETERS,
    "sampling.boundary.strategy": CONSTRAINT_SAMPLER_PARAMETERS,
    "sampling.initial.strategy": CONSTRAINT_SAMPLER_PARAMETERS,
    "sampling.adaptive_refinement.strategy": {
        name: deepcopy(ADAPTIVE_PARAMETERS)
        for name in ("rar", "rad", "rar_d", "residual_adaptive", "gradient_adaptive", "hybrid_adaptive")
    },
    "loss.terms[].loss_function": LOSS_FUNCTION_PARAMETERS,
    "loss.weighting_strategy.name": WEIGHTING_PARAMETERS,
    "loss.aggregation.name": AGGREGATION_PARAMETERS,
    "optimization.phases[].optimizer": OPTIMIZER_PARAMETERS,
    "optimization.phases[].scheduler.name": SCHEDULER_PARAMETERS,
}

for alias, canonical in SCHEDULER_ALIASES.items():
    OPTION_PATCHES["optimization.phases[].scheduler.name"][alias] = deepcopy(SCHEDULER_PARAMETERS[canonical])


DIRECT_VALUE_SCHEMAS: dict[str, dict[str, Any]] = {
    "loss.terms[].weight": _number(
        minimum=0.0, maximum=1e18, default=1.0, scale="log_zero",
        recommended=[0.0, 1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0, 1e4, 1e6],
    ),
    "network.hidden_layers": _array(
        _integer(minimum=1, maximum=65_536, scale="log2",
                 recommended=PINNSAGENT_WIDTHS + [512, 1024, 2048]),
        min_items=1,
        max_items=64,
        default=[128, 128, 128, 128],
        recommended_lengths=PINNSAGENT_DEPTHS,
    ),
    "network.residual_connections.parameters": _object({
        "scale": _number(minimum=-100.0, maximum=100.0, default=1.0,
                         recommended=[-1.0, -0.5, 0.1, 0.5, 1.0, 2.0]),
    }),
    "optimization.phases[].iterations": _integer(
        minimum=0, maximum=100_000_000, default=3000, scale="log",
        recommended=[0, 10, 50, 100, 300, 500, 1000, 3000, 5000, 10000, 20000, 50000],
    ),
    "training.gradient_clipping.parameters": _object({
        "max_norm": _number(exclusive_minimum=0.0, maximum=1e12, default=1.0, scale="log",
                            recommended=[0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0]),
        "norm_type": _number(exclusive_minimum=0.0, maximum=1e6, default=2.0,
                             recommended=[1.0, 2.0]),
    }),
    "training.early_stopping.parameters": _object({
        "threshold": _number(minimum=0.0, maximum=1e12, default=1e-8, scale="log_zero",
                             recommended=[0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4]),
        "patience": _integer(minimum=0, maximum=100_000_000, default=500, scale="log_zero",
                             recommended=[0, 10, 50, 100, 500, 1000, 5000]),
    }),
}


def apply_numeric_max_expansion(registry: dict[str, Any]) -> dict[str, Any]:
    """Merge exhaustive numeric schemas and LLM search metadata."""

    fields = registry.setdefault("fields", {})
    for field_path, options_patch in deepcopy(OPTION_PATCHES).items():
        field = fields.setdefault(field_path, {})
        options = field.setdefault("options", {})
        for option_name, properties in options_patch.items():
            option = options.setdefault(option_name, {})
            schema = option.setdefault("parameter_schema", _object({}))
            schema["type"] = "object"
            schema["additionalProperties"] = False
            schema.setdefault("properties", {}).update(properties)
            compatibility = COMPATIBILITY_RULES.get((field_path, option_name))
            if compatibility:
                option["compatibility"] = deepcopy(compatibility)

    for field_path, schema in deepcopy(DIRECT_VALUE_SCHEMAS).items():
        fields.setdefault(field_path, {})["value_schema"] = schema

    registry["numeric_search_space_version"] = "2.0-max"
    registry["numeric_search_policy"] = {
        "continuous_ranges_are_allowed": True,
        "prefer_log_scale_for_positive_magnitude_parameters": True,
        "recommended_values_are_seeds_not_enumerated_limits": True,
        "builder_and_validator_remain_authoritative": True,
    }
    return registry


# Backward-compatible name used by the prior integration snippet.
def apply_parameter_expansion(registry: dict[str, Any]) -> dict[str, Any]:
    return apply_numeric_max_expansion(registry)

# ---------------------------------------------------------------------------
# Expanded registry wrapper
# ---------------------------------------------------------------------------
def algorithm_spec_option_registry() -> dict[str, Any]:
    """Return a fresh base registry with the maximal numeric expansion applied."""

    base_registry = deepcopy(_base_algorithm_spec_option_registry())
    return apply_numeric_max_expansion(base_registry)


def expanded_algorithm_spec_option_registry() -> dict[str, Any]:
    """Explicit alias for callers that want the expanded registry."""

    return algorithm_spec_option_registry()

# ---------------------------------------------------------------------------
# AlgorithmSpec normalization and validation
# ---------------------------------------------------------------------------
ARCHITECTURE_ALIASES = {
    "fnn": "mlp",
    "fouriermlp": "fourier_mlp",
    "multiscalemlp": "multiscale_mlp",
    "modifiedmlp": "modified_mlp",
    "gatedmlp": "gated_mlp",
    "factorizedmlp": "factorized_mlp",
    "residualmlp": "residual_mlp",
    "parallelfnn": "parallel_fnn",
}
ACTIVATION_ALIASES = {
    "swish": "silu",
    "sin": "sine",
    "cos": "cosine",
    "scaledtanh": "scaled_tanh",
    "hard_swish": "hardswish",
    "hard_sigmoid": "hardsigmoid",
    "logsigmoid": "log_sigmoid",
}
INITIALIZER_ALIASES = {
    "xaviernormal": "xavier_normal",
    "xavieruniform": "xavier_uniform",
    "glorotnormal": "glorot_normal",
    "glorotuniform": "glorot_uniform",
    "he_normal": "kaiming_normal",
    "henormal": "kaiming_normal",
    "he_uniform": "kaiming_uniform",
    "heuniform": "kaiming_uniform",
    "kaimingnormal": "kaiming_normal",
    "kaiminguniform": "kaiming_uniform",
    "truncnormal": "trunc_normal",
}
SAMPLER_ALIASES = {
    "lhs": "latin_hypercube",
    "latin_hypercube_sampling": "latin_hypercube",
    "residualadaptive": "residual_adaptive",
    "gradientadaptive": "gradient_adaptive",
    "hybridadaptive": "hybrid_adaptive",
}
LOSS_ALIASES = {
    "smoothl1": "smooth_l1",
    "logcosh": "log_cosh",
    "pseudohuber": "pseudo_huber",
    "gemanmcclure": "geman_mcclure",
    "logl2": "log_l2",
}
WEIGHTING_ALIASES = {
    "none": "fixed",
    "inversemagnitude": "inverse_magnitude",
    "emainversemagnitude": "ema_inverse_magnitude",
    "losssoftmax": "loss_softmax",
    "ntktrace": "ntk_trace",
}
AGGREGATION_ALIASES = {
    "weightedsum": "weighted_sum",
    "log_sum_exp": "logsumexp",
    "pnorm": "p_norm",
    "rms": "root_mean_square",
}
OPTIMIZER_ALIASES = {"l_bfgs": "lbfgs", "l_bfgs_optimizer": "lbfgs"}
SCHEDULER_ALIASES = {
    "step": "step_lr",
    "cosine": "cosine_annealing",
    "cosine_warm_restart": "cosine_warm_restarts",
    "cosinewarmrestart": "cosine_warm_restarts",
    "cosinewarmrestarts": "cosine_warm_restarts",
    "cosine_annealing_warm_restart": "cosine_warm_restarts",
    "exponential": "exponential_lr",
    "plateau": "reduce_on_plateau",
    "onecycle": "one_cycle",
}


@dataclass(frozen=True)
class ValidationIssue:
    path: str
    code: str
    message: str


@dataclass
class ValidationReport:
    valid: bool
    errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)
    raw_spec: dict[str, Any] | None = None
    normalized_spec: dict[str, Any] | None = None
    budget_usage: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AlgorithmSpecValidator:
    """Owns raw structure checks, conservative normalization, and constraints."""

    def validate(self, spec: dict, problem: Any, experiment_config: dict, builder_capabilities: dict) -> ValidationReport:
        return validate_algorithm_spec(spec, problem, experiment_config, builder_capabilities)


def normalize_algorithm_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Normalize representation only; never tune or repair algorithm values."""

    normalized = deepcopy(spec)
    network = normalized["network"]
    architecture = _normalized_name(network["architecture"])
    network["architecture"] = ARCHITECTURE_ALIASES.get(architecture, architecture)
    network.setdefault("extra_parameters", {})
    for field_name in ("activation", "output_activation", "initialization", "input_transform"):
        component = network[field_name]
        component["name"] = _normalized_name(component["name"])
        component.setdefault("parameters", {})
    network["activation"]["name"] = ACTIVATION_ALIASES.get(network["activation"]["name"], network["activation"]["name"])
    network["output_activation"]["name"] = ACTIVATION_ALIASES.get(
        network["output_activation"]["name"], network["output_activation"]["name"]
    )
    network["initialization"]["name"] = INITIALIZER_ALIASES.get(
        network["initialization"]["name"], network["initialization"]["name"]
    )
    residual = network["residual_connections"]
    residual.setdefault("parameters", {})
    if not residual["enabled"]:
        residual["parameters"] = {}

    sampling = normalized["sampling"]
    for section in ("interior", "boundary", "initial"):
        strategy = _normalized_name(sampling[section]["strategy"])
        sampling[section]["strategy"] = SAMPLER_ALIASES.get(strategy, strategy)
        sampling[section].setdefault("parameters", {})
    adaptive = sampling["adaptive_refinement"]
    adaptive_strategy = _normalized_name(adaptive["strategy"])
    adaptive["strategy"] = SAMPLER_ALIASES.get(adaptive_strategy, adaptive_strategy)
    adaptive.setdefault("parameters", {})
    if not adaptive["enabled"]:
        adaptive["parameters"] = {}

    enforcement = normalized.setdefault(
        "constraint_enforcement",
        {"method": "soft_penalty", "transform_id": "none", "parameters": {}},
    )
    enforcement["method"] = _normalized_name(enforcement["method"])
    enforcement["transform_id"] = _normalized_name(enforcement["transform_id"])
    enforcement.setdefault("parameters", {})

    loss = normalized["loss"]
    for term in loss["terms"]:
        term["name"] = _normalized_name(term["name"])
        loss_name = _normalized_name(term["loss_function"])
        term["loss_function"] = LOSS_ALIASES.get(loss_name, loss_name)
        term.setdefault("parameters", {})
    weighting_name = _normalized_name(loss["weighting_strategy"]["name"])
    loss["weighting_strategy"]["name"] = WEIGHTING_ALIASES.get(weighting_name, weighting_name)
    loss["weighting_strategy"].setdefault("parameters", {})
    aggregation_name = _normalized_name(loss["aggregation"]["name"])
    loss["aggregation"]["name"] = AGGREGATION_ALIASES.get(aggregation_name, aggregation_name)
    loss["aggregation"].setdefault("parameters", {})

    for phase in normalized["optimization"]["phases"]:
        optimizer = _normalized_name(phase["optimizer"])
        phase["optimizer"] = OPTIMIZER_ALIASES.get(optimizer, optimizer)
        phase.setdefault("parameters", {})
        scheduler_name = _normalized_name(phase["scheduler"]["name"])
        phase["scheduler"]["name"] = SCHEDULER_ALIASES.get(scheduler_name, scheduler_name)
        phase["scheduler"].setdefault("parameters", {})

    training = normalized["training"]
    training["batch_mode"] = _normalized_name(training["batch_mode"])
    training.setdefault("extra_parameters", {})
    for field_name in ("gradient_clipping", "early_stopping"):
        training[field_name].setdefault("parameters", {})
        if not training[field_name]["enabled"]:
            training[field_name]["parameters"] = {}
    return _sort_dicts(_normalize_nested(normalized))


def validate_algorithm_spec(
    spec: dict,
    problem: Any,
    experiment_config: dict,
    builder_capabilities: dict,
) -> ValidationReport:
    # Keep the public credibility validator authoritative.  This local entry
    # point remains for backward compatibility with callers that historically
    # imported validation from the all-in-one builder module.
    from forge.pipeline.e_credibility_assessment.validation.algorithm_spec_validator import (
        validate_algorithm_spec as public_validate_algorithm_spec,
    )

    return public_validate_algorithm_spec(
        spec, problem, experiment_config, builder_capabilities
    )

    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []
    if not isinstance(spec, dict):
        return ValidationReport(valid=False, errors=[ValidationIssue("$", "INVALID_TYPE", "AlgorithmSpec must be an object")])

    for field_name in REQUIRED_TOP_LEVEL_FIELDS:
        if field_name not in spec:
            errors.append(ValidationIssue(field_name, "MISSING_FIELD", f"Missing required field '{field_name}'"))
    if errors:
        return ValidationReport(valid=False, errors=errors, raw_spec=deepcopy(spec))

    _validate_structure(spec, errors)
    if errors:
        return ValidationReport(valid=False, errors=errors, raw_spec=deepcopy(spec))

    try:
        normalized = normalize_algorithm_spec(spec)
    except Exception as exc:
        return ValidationReport(
            valid=False,
            errors=[ValidationIssue("$", "NORMALIZATION_FAILED", str(exc))],
        )

    _validate_finite_numbers(normalized, "$", errors)
    _validate_positive_values(normalized, errors)
    _validate_components(normalized, builder_capabilities, errors)
    _validate_registered_option_schemas(normalized, builder_capabilities, errors)
    _validate_problem_combinations(normalized, problem, errors)
    budget_usage = _validate_budget(normalized, problem, experiment_config, errors)
    return ValidationReport(
        valid=not errors,
        errors=errors,
        warnings=warnings,
        raw_spec=deepcopy(spec),
        normalized_spec=normalized,
        budget_usage=budget_usage,
    )


def _issue(errors: list[ValidationIssue], path: str, code: str, message: str) -> None:
    errors.append(ValidationIssue(path, code, message))


def _validate_structure(spec: dict[str, Any], errors: list[ValidationIssue]) -> None:
    if not isinstance(spec.get("spec_version"), str):
        _issue(errors, "spec_version", "INVALID_TYPE", "spec_version must be a string")
    for section in ("network", "sampling", "loss", "optimization", "training", "generation_metadata"):
        if not isinstance(spec.get(section), dict):
            _issue(errors, section, "INVALID_TYPE", f"{section} must be an object")
    if errors:
        return

    enforcement = spec.get("constraint_enforcement")
    if enforcement is not None:
        if not isinstance(enforcement, dict):
            _issue(errors, "constraint_enforcement", "INVALID_TYPE", "constraint_enforcement must be an object")
        else:
            if not isinstance(enforcement.get("method"), str):
                _issue(errors, "constraint_enforcement.method", "MISSING_OR_INVALID_FIELD", "constraint enforcement method must be a string")
            if not isinstance(enforcement.get("transform_id"), str):
                _issue(errors, "constraint_enforcement.transform_id", "MISSING_OR_INVALID_FIELD", "constraint transform_id must be a string")
            if not isinstance(enforcement.get("parameters", {}), dict):
                _issue(errors, "constraint_enforcement.parameters", "INVALID_TYPE", "constraint enforcement parameters must be an object")
    if errors:
        return

    network = spec["network"]
    if "architecture_parameters" in network:
        _issue(
            errors,
            "network.architecture_parameters",
            "LEGACY_FIELD_UNSUPPORTED",
            "Use network.extra_parameters for architecture settings",
        )
    required_network_fields = {"architecture", "hidden_layers", "activation", "output_activation", "initialization", "input_transform", "residual_connections"}
    for key in sorted(required_network_fields - set(network)):
        _issue(errors, f"network.{key}", "MISSING_FIELD", f"Missing required network field '{key}'")
    hidden = network.get("hidden_layers")
    if not isinstance(hidden, list) or not hidden or not all(isinstance(value, int) and not isinstance(value, bool) for value in hidden):
        _issue(errors, "network.hidden_layers", "INVALID_HIDDEN_LAYERS", "hidden_layers must be a non-empty integer list")
    if not isinstance(network.get("architecture"), str):
        _issue(errors, "network.architecture", "INVALID_TYPE", "architecture must be a string")
    for key in ("activation", "output_activation", "initialization", "input_transform", "residual_connections"):
        if key in network and not isinstance(network[key], dict):
            _issue(errors, f"network.{key}", "INVALID_TYPE", f"network.{key} must be an object")
    for key in ("activation", "output_activation", "initialization", "input_transform"):
        if isinstance(network.get(key), dict) and not isinstance(network[key].get("name"), str):
            _issue(errors, f"network.{key}.name", "MISSING_OR_INVALID_FIELD", "component name must be a string")
    if isinstance(network.get("residual_connections"), dict) and not isinstance(network["residual_connections"].get("enabled"), bool):
        _issue(errors, "network.residual_connections.enabled", "MISSING_OR_INVALID_FIELD", "enabled must be a boolean")
    if errors:
        return

    sampling = spec["sampling"]
    for key in ("interior", "boundary", "initial", "adaptive_refinement"):
        if key not in sampling:
            _issue(errors, f"sampling.{key}", "MISSING_FIELD", f"Missing required sampling field '{key}'")
        elif not isinstance(sampling[key], dict):
            _issue(errors, f"sampling.{key}", "INVALID_TYPE", f"sampling.{key} must be an object")
    for key in ("interior", "boundary", "initial"):
        if isinstance(sampling.get(key), dict) and not isinstance(sampling[key].get("strategy"), str):
            _issue(errors, f"sampling.{key}.strategy", "MISSING_OR_INVALID_FIELD", "strategy must be a string")
    adaptive = sampling.get("adaptive_refinement")
    if isinstance(adaptive, dict):
        if not isinstance(adaptive.get("enabled"), bool):
            _issue(errors, "sampling.adaptive_refinement.enabled", "MISSING_OR_INVALID_FIELD", "enabled must be a boolean")
        if not isinstance(adaptive.get("strategy"), str):
            _issue(errors, "sampling.adaptive_refinement.strategy", "MISSING_OR_INVALID_FIELD", "strategy must be a string")
    if errors:
        return

    loss = spec["loss"]
    for key in ("terms", "weighting_strategy", "aggregation"):
        if key not in loss:
            _issue(errors, f"loss.{key}", "MISSING_FIELD", f"Missing required loss field '{key}'")
    terms = loss.get("terms")
    if not isinstance(terms, list) or not terms:
        _issue(errors, "loss.terms", "INVALID_LOSS_TERMS", "loss.terms must be a non-empty list")
    elif not all(isinstance(term, dict) for term in terms):
        _issue(errors, "loss.terms", "INVALID_TYPE", "every loss term must be an object")
    else:
        for index, term in enumerate(terms):
            if "loss_function_parameters" in term:
                _issue(
                    errors,
                    f"loss.terms[{index}].loss_function_parameters",
                    "LEGACY_FIELD_UNSUPPORTED",
                    "Use loss.terms[].parameters for loss-function settings",
                )
            if not isinstance(term.get("name"), str):
                _issue(errors, f"loss.terms[{index}].name", "MISSING_OR_INVALID_FIELD", "loss term name must be a string")
            if not isinstance(term.get("loss_function"), str):
                _issue(errors, f"loss.terms[{index}].loss_function", "MISSING_OR_INVALID_FIELD", "loss_function must be a string")
            if not isinstance(term.get("weight"), (int, float)) or isinstance(term.get("weight"), bool):
                _issue(errors, f"loss.terms[{index}].weight", "MISSING_OR_INVALID_FIELD", "weight must be numeric")
    for key in ("weighting_strategy", "aggregation"):
        if key in loss and not isinstance(loss[key], dict):
            _issue(errors, f"loss.{key}", "INVALID_TYPE", f"loss.{key} must be an object")

    phases = spec["optimization"].get("phases")
    if not isinstance(phases, list) or not phases:
        _issue(errors, "optimization.phases", "INVALID_PHASES", "optimization.phases must be a non-empty list")
    elif not all(isinstance(phase, dict) for phase in phases):
        _issue(errors, "optimization.phases", "INVALID_TYPE", "every optimization phase must be an object")
    elif any(not isinstance(phase.get("scheduler"), dict) for phase in phases):
        _issue(errors, "optimization.phases[].scheduler", "INVALID_TYPE", "every scheduler must be an object")
    else:
        for index, phase in enumerate(phases):
            if "optimizer_parameters" in phase:
                _issue(
                    errors,
                    f"optimization.phases[{index}].optimizer_parameters",
                    "LEGACY_FIELD_UNSUPPORTED",
                    "Use optimization.phases[].parameters for optimizer settings",
                )
            if not isinstance(phase.get("optimizer"), str):
                _issue(errors, f"optimization.phases[{index}].optimizer", "MISSING_OR_INVALID_FIELD", "optimizer must be a string")
            if not isinstance(phase.get("iterations"), int) or isinstance(phase.get("iterations"), bool):
                _issue(errors, f"optimization.phases[{index}].iterations", "MISSING_OR_INVALID_FIELD", "iterations must be an integer")
            if not isinstance(phase["scheduler"].get("name"), str):
                _issue(errors, f"optimization.phases[{index}].scheduler.name", "MISSING_OR_INVALID_FIELD", "scheduler name must be a string")

    training = spec["training"]
    for key in ("batch_mode", "batch_size", "gradient_clipping", "early_stopping"):
        if key not in training:
            _issue(errors, f"training.{key}", "MISSING_FIELD", f"Missing required training field '{key}'")
    if "batch_mode" in training and not isinstance(training["batch_mode"], str):
        _issue(errors, "training.batch_mode", "INVALID_TYPE", "batch_mode must be a string")
    if training.get("batch_mode") == "full_batch" and training.get("batch_size") is not None:
        _issue(
            errors,
            "training.batch_size",
            "FULL_BATCH_REQUIRES_NULL_BATCH_SIZE",
            "batch_size must be null when batch_mode is full_batch",
        )
    for key in ("gradient_clipping", "early_stopping"):
        if key in training and not isinstance(training[key], dict):
            _issue(errors, f"training.{key}", "INVALID_TYPE", f"training.{key} must be an object")
        elif isinstance(training.get(key), dict) and not isinstance(training[key].get("enabled"), bool):
            _issue(errors, f"training.{key}.enabled", "MISSING_OR_INVALID_FIELD", "enabled must be a boolean")
    if errors:
        return
    _validate_parameter_objects(spec, "$", errors)
    for path, value in _walk_named_fields(spec, "enabled"):
        if not isinstance(value, bool):
            _issue(errors, path, "INVALID_ENABLED", "enabled must be a boolean")


def _validate_parameter_objects(value: Any, path: str, errors: list[ValidationIssue]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            child_path = f"{path}.{key}" if path != "$" else key
            if key in {"parameters", "extra_parameters"} and not isinstance(item, dict):
                _issue(errors, child_path, "INVALID_PARAMETERS", f"{key} must be an object")
            _validate_parameter_objects(item, child_path, errors)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_parameter_objects(item, f"{path}[{index}]", errors)


def _validate_finite_numbers(value: Any, path: str, errors: list[ValidationIssue]) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        _issue(errors, path, "NON_FINITE_NUMBER", "Floating-point values must be finite")
    elif isinstance(value, dict):
        for key, item in value.items():
            _validate_finite_numbers(item, f"{path}.{key}", errors)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_finite_numbers(item, f"{path}[{index}]", errors)


def _validate_positive_values(spec: dict[str, Any], errors: list[ValidationIssue]) -> None:
    for index, width in enumerate(spec["network"].get("hidden_layers") or []):
        if width <= 0:
            _issue(errors, f"network.hidden_layers[{index}]", "INVALID_WIDTH", "network width must be positive")
    for index, phase in enumerate(spec["optimization"].get("phases") or []):
        iterations = phase.get("iterations")
        if not isinstance(iterations, int) or isinstance(iterations, bool) or iterations < 0:
            _issue(errors, f"optimization.phases[{index}].iterations", "INVALID_ITERATIONS", "iterations must be a non-negative integer")
        learning_rate = (phase.get("parameters") or {}).get("learning_rate")
        if learning_rate is not None and (not isinstance(learning_rate, (int, float)) or isinstance(learning_rate, bool) or learning_rate <= 0):
            _issue(errors, f"optimization.phases[{index}].parameters.learning_rate", "INVALID_LEARNING_RATE", "learning_rate must be positive")
    for section in ("interior", "boundary", "initial"):
        n_points = ((spec["sampling"].get(section) or {}).get("parameters") or {}).get("n_points", 0)
        if not isinstance(n_points, int) or isinstance(n_points, bool) or n_points < 0:
            _issue(errors, f"sampling.{section}.parameters.n_points", "INVALID_SAMPLE_COUNT", "n_points must be a non-negative integer")
    for index, term in enumerate(spec["loss"].get("terms") or []):
        weight = term.get("weight")
        if (
            not isinstance(weight, (int, float))
            or isinstance(weight, bool)
            or not math.isfinite(float(weight))
            or float(weight) <= 0.0
        ):
            _issue(
                errors,
                f"loss.terms[{index}].weight",
                "INVALID_LOSS_WEIGHT",
                "loss weight must be a finite positive number",
            )


def _validate_components(spec: dict[str, Any], capabilities: dict[str, Any], errors: list[ValidationIssue]) -> None:
    checks = [
        ("network.architecture", spec["network"].get("architecture"), "networks"),
        ("network.activation.name", spec["network"]["activation"].get("name"), "activations"),
        ("network.output_activation.name", spec["network"]["output_activation"].get("name"), "activations"),
        ("network.initialization.name", spec["network"]["initialization"].get("name"), "initializers"),
        ("network.input_transform.name", spec["network"]["input_transform"].get("name"), "input_transforms"),
        ("constraint_enforcement.method", spec["constraint_enforcement"].get("method"), "constraint_enforcement_methods"),
        ("loss.weighting_strategy.name", spec["loss"]["weighting_strategy"].get("name"), "weighting_strategies"),
        ("loss.aggregation.name", spec["loss"]["aggregation"].get("name"), "aggregations"),
    ]
    checks.append(("sampling.interior.strategy", spec["sampling"]["interior"].get("strategy"), "interior_samplers"))
    for section in ("boundary", "initial"):
        checks.append((f"sampling.{section}.strategy", spec["sampling"][section].get("strategy"), "constraint_samplers"))
    adaptive = spec["sampling"]["adaptive_refinement"]
    if adaptive.get("enabled"):
        checks.append(("sampling.adaptive_refinement.strategy", adaptive.get("strategy"), "adaptive_samplers"))
    for index, term in enumerate(spec["loss"].get("terms") or []):
        checks.append((f"loss.terms[{index}].loss_function", term.get("loss_function"), "loss_functions"))
    for index, phase in enumerate(spec["optimization"].get("phases") or []):
        checks.append((f"optimization.phases[{index}].optimizer", phase.get("optimizer"), "optimizers"))
        checks.append((f"optimization.phases[{index}].scheduler.name", phase["scheduler"].get("name"), "schedulers"))
    for path, value, category in checks:
        if value not in set(capabilities.get(category) or []):
            _issue(errors, path, "UNSUPPORTED_COMPONENT", f"No registered builder exists for '{value}'")


def _validate_registered_option_schemas(
    spec: dict[str, Any],
    capabilities: dict[str, Any],
    errors: list[ValidationIssue],
) -> None:
    registry = capabilities.get("algorithm_spec_option_registry") or {}
    fields = registry.get("fields") or {}
    selections: list[tuple[str, str, dict[str, Any], str]] = [
        ("network.architecture", spec["network"]["architecture"], spec["network"].get("extra_parameters") or {}, "network.extra_parameters"),
        ("network.activation.name", spec["network"]["activation"]["name"], spec["network"]["activation"].get("parameters") or {}, "network.activation.parameters"),
        ("network.output_activation.name", spec["network"]["output_activation"]["name"], spec["network"]["output_activation"].get("parameters") or {}, "network.output_activation.parameters"),
        ("network.initialization.name", spec["network"]["initialization"]["name"], spec["network"]["initialization"].get("parameters") or {}, "network.initialization.parameters"),
        ("network.input_transform.name", spec["network"]["input_transform"]["name"], spec["network"]["input_transform"].get("parameters") or {}, "network.input_transform.parameters"),
        ("constraint_enforcement.method", spec["constraint_enforcement"]["method"], spec["constraint_enforcement"].get("parameters") or {}, "constraint_enforcement.parameters"),
    ]
    for section in ("interior", "boundary", "initial"):
        sampler = spec["sampling"][section]
        selections.append(
            (f"sampling.{section}.strategy", sampler["strategy"], sampler.get("parameters") or {}, f"sampling.{section}.parameters")
        )
    adaptive = spec["sampling"]["adaptive_refinement"]
    if adaptive.get("enabled"):
        selections.append(
            (
                "sampling.adaptive_refinement.strategy",
                adaptive["strategy"],
                adaptive.get("parameters") or {},
                "sampling.adaptive_refinement.parameters",
            )
        )
    time_strategy = spec["training"]["time_strategy"]
    selections.append(
        (
            "training.time_strategy.name",
            time_strategy["name"],
            time_strategy.get("parameters") or {},
            "training.time_strategy.parameters",
        )
    )
    for index, term in enumerate(spec["loss"].get("terms") or []):
        selections.append(
            ("loss.terms[].loss_function", term["loss_function"], term.get("parameters") or {}, f"loss.terms[{index}].parameters")
        )
    for field_name in ("weighting_strategy", "aggregation"):
        component = spec["loss"][field_name]
        selections.append(
            (f"loss.{field_name}.name", component["name"], component.get("parameters") or {}, f"loss.{field_name}.parameters")
        )
    for index, phase in enumerate(spec["optimization"].get("phases") or []):
        selections.append(
            ("optimization.phases[].optimizer", phase["optimizer"], phase.get("parameters") or {}, f"optimization.phases[{index}].parameters")
        )
        selections.append(
            (
                "optimization.phases[].scheduler.name",
                phase["scheduler"]["name"],
                phase["scheduler"].get("parameters") or {},
                f"optimization.phases[{index}].scheduler.parameters",
            )
        )

    for field_path, option_name, parameters, parameter_path in selections:
        field = fields.get(field_path) or {}
        options = field.get("options") or {}
        option = options.get(option_name)
        if not isinstance(option, dict):
            if options:
                _issue(
                    errors,
                    field_path,
                    "UNREGISTERED_OPTION",
                    f"Option '{option_name}' is not registered for {field_path}",
                )
            continue
        schema = option.get("parameter_schema") or {}
        properties = schema.get("properties") or {}
        if schema.get("additionalProperties") is False:
            for name in sorted(set(parameters) - set(properties)):
                _issue(
                    errors,
                    f"{parameter_path}.{name}",
                    "UNREGISTERED_OPTION_PARAMETER",
                    f"Parameter '{name}' is not legal for {field_path}='{option_name}'",
                )
        for name, value in parameters.items():
            parameter_schema = properties.get(name)
            if isinstance(parameter_schema, dict):
                _validate_schema_value(value, parameter_schema, f"{parameter_path}.{name}", errors)
        _validate_parameter_compatibility(parameters, option.get("compatibility") or [], parameter_path, errors)

    direct_values: list[tuple[str, Any, str]] = [
        ("network.hidden_layers", spec["network"]["hidden_layers"], "network.hidden_layers"),
        ("network.residual_connections.enabled", spec["network"]["residual_connections"]["enabled"], "network.residual_connections.enabled"),
        ("network.residual_connections.parameters", spec["network"]["residual_connections"].get("parameters") or {}, "network.residual_connections.parameters"),
        ("constraint_enforcement.transform_id", spec["constraint_enforcement"]["transform_id"], "constraint_enforcement.transform_id"),
        ("constraint_enforcement.parameters", spec["constraint_enforcement"].get("parameters") or {}, "constraint_enforcement.parameters"),
        ("training.batch_mode", spec["training"]["batch_mode"], "training.batch_mode"),
        ("training.batch_size", spec["training"]["batch_size"], "training.batch_size"),
        ("training.gradient_clipping.enabled", spec["training"]["gradient_clipping"]["enabled"], "training.gradient_clipping.enabled"),
        ("training.gradient_clipping.parameters", spec["training"]["gradient_clipping"].get("parameters") or {}, "training.gradient_clipping.parameters"),
        ("training.early_stopping.enabled", spec["training"]["early_stopping"]["enabled"], "training.early_stopping.enabled"),
        ("training.early_stopping.parameters", spec["training"]["early_stopping"].get("parameters") or {}, "training.early_stopping.parameters"),
    ]
    for index, term in enumerate(spec["loss"].get("terms") or []):
        direct_values.append(
            ("loss.terms[].weight", term["weight"], f"loss.terms[{index}].weight")
        )
    for index, phase in enumerate(spec["optimization"].get("phases") or []):
        direct_values.append(
            ("optimization.phases[].iterations", phase["iterations"], f"optimization.phases[{index}].iterations")
        )
    for field_path, value, value_path in direct_values:
        value_schema = (fields.get(field_path) or {}).get("value_schema")
        if isinstance(value_schema, dict):
            _validate_schema_value(value, value_schema, value_path, errors)


def _validate_schema_value(
    value: Any,
    schema: dict[str, Any],
    path: str,
    errors: list[ValidationIssue],
) -> None:
    expected = schema.get("type")
    valid_type = {
        "boolean": isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "string": isinstance(value, str),
        "array": isinstance(value, (list, tuple)),
        "object": isinstance(value, dict),
        "null": value is None,
    }.get(expected, True)
    if not valid_type:
        _issue(errors, path, "OPTION_PARAMETER_TYPE", f"Expected {expected}, got {type(value).__name__}")
        return
    if "enum" in schema and value not in schema["enum"]:
        _issue(errors, path, "OPTION_PARAMETER_ENUM", f"Value must be one of {schema['enum']}")
    if expected in {"integer", "number"}:
        numeric = float(value)
        if "minimum" in schema and numeric < float(schema["minimum"]):
            _issue(errors, path, "OPTION_PARAMETER_RANGE", f"Value must be >= {schema['minimum']}")
        if "maximum" in schema and numeric > float(schema["maximum"]):
            _issue(errors, path, "OPTION_PARAMETER_RANGE", f"Value must be <= {schema['maximum']}")
        if "exclusiveMinimum" in schema and numeric <= float(schema["exclusiveMinimum"]):
            _issue(errors, path, "OPTION_PARAMETER_RANGE", f"Value must be > {schema['exclusiveMinimum']}")
        if "exclusiveMaximum" in schema and numeric >= float(schema["exclusiveMaximum"]):
            _issue(errors, path, "OPTION_PARAMETER_RANGE", f"Value must be < {schema['exclusiveMaximum']}")
    if expected == "array":
        if "minItems" in schema and len(value) < int(schema["minItems"]):
            _issue(errors, path, "OPTION_PARAMETER_LENGTH", f"Array needs at least {schema['minItems']} items")
        if "maxItems" in schema and len(value) > int(schema["maxItems"]):
            _issue(errors, path, "OPTION_PARAMETER_LENGTH", f"Array allows at most {schema['maxItems']} items")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                _validate_schema_value(item, item_schema, f"{path}[{index}]", errors)
    if expected == "object":
        properties = schema.get("properties") or {}
        if schema.get("additionalProperties") is False:
            for name in sorted(set(value) - set(properties)):
                _issue(errors, f"{path}.{name}", "UNREGISTERED_OPTION_PARAMETER", f"Parameter '{name}' is not legal")
        for name, item in value.items():
            item_schema = properties.get(name)
            if isinstance(item_schema, dict):
                _validate_schema_value(item, item_schema, f"{path}.{name}", errors)


def _validate_parameter_compatibility(
    parameters: dict[str, Any],
    rules: list[dict[str, Any]],
    path: str,
    errors: list[ValidationIssue],
) -> None:
    for rule in rules:
        rule_name = rule.get("rule")
        if rule_name == "parameter_order":
            lower_name = str(rule.get("lower"))
            upper_name = str(rule.get("upper"))
            lower = parameters.get(lower_name)
            upper = parameters.get(upper_name)
            if lower is not None and upper is not None and float(lower) >= float(upper):
                _issue(
                    errors,
                    path,
                    "OPTION_PARAMETER_COMPATIBILITY",
                    f"'{lower_name}' must be smaller than '{upper_name}'",
                )
        elif rule_name == "array_element_order":
            parameter_name = str(rule.get("parameter"))
            values = parameters.get(parameter_name)
            lower_index = int(rule.get("lower_index", 0))
            upper_index = int(rule.get("upper_index", 1))
            if isinstance(values, (list, tuple)) and len(values) > max(lower_index, upper_index):
                if float(values[lower_index]) >= float(values[upper_index]):
                    _issue(
                        errors,
                        f"{path}.{parameter_name}",
                        "OPTION_PARAMETER_COMPATIBILITY",
                        f"Element {lower_index} must be smaller than element {upper_index}",
                    )
        elif rule_name == "array_straddles":
            parameter_name = str(rule.get("parameter"))
            pivot = float(rule.get("pivot", 0.0))
            values = parameters.get(parameter_name)
            if isinstance(values, (list, tuple)) and len(values) >= 2:
                if not float(values[0]) < pivot < float(values[1]):
                    _issue(
                        errors,
                        f"{path}.{parameter_name}",
                        "OPTION_PARAMETER_COMPATIBILITY",
                        f"The pair must straddle {pivot}: first < {pivot} < second",
                    )
        elif rule_name == "paired_parameters":
            first = str(rule.get("first"))
            second = str(rule.get("second"))
            if (first in parameters) != (second in parameters):
                _issue(
                    errors,
                    path,
                    "OPTION_PARAMETER_COMPATIBILITY",
                    f"'{first}' and '{second}' must be provided together",
                )
        elif rule_name == "mutually_exclusive":
            names = [str(name) for name in rule.get("parameters") or []]
            present = [name for name in names if name in parameters]
            if len(present) > 1:
                _issue(
                    errors,
                    path,
                    "OPTION_PARAMETER_COMPATIBILITY",
                    f"Parameters are mutually exclusive: {present}",
                )


def _validate_problem_combinations(spec: dict[str, Any], problem: Any, errors: list[ValidationIssue]) -> None:
    problem_spec = problem.get_spec()
    enforcement = spec["constraint_enforcement"]
    method = enforcement["method"]
    transform_id = enforcement["transform_id"]
    parameters = enforcement.get("parameters") or {}
    capabilities_fn = getattr(problem, "constraint_enforcement_capabilities", None)
    capabilities = capabilities_fn() if callable(capabilities_fn) else {
        "soft_penalty": {"transform_ids": ["none"]},
        "problem_hard": {"transform_ids": []},
    }
    method_capability = capabilities.get(method) if isinstance(capabilities, dict) else None
    if not isinstance(method_capability, dict):
        _issue(errors, "constraint_enforcement.method", "CONSTRAINT_ENFORCEMENT_UNAVAILABLE", f"Problem '{problem_spec.problem_id}' does not support method '{method}'")
    elif transform_id not in set(method_capability.get("transform_ids") or []):
        _issue(errors, "constraint_enforcement.transform_id", "HARD_TRANSFORM_UNAVAILABLE", f"Transform '{transform_id}' is not registered for problem '{problem_spec.problem_id}'")
    if parameters:
        _issue(errors, "constraint_enforcement.parameters", "UNREGISTERED_OPTION_PARAMETER", "Registered constraint transforms currently accept no parameters")
    constraint_ids = {constraint.constraint_id for constraint in problem_spec.constraints}
    loss_terms = spec["loss"].get("terms") or []
    term_names = {term.get("name") for term in loss_terms}
    governing_terms = {"pde_residual", "governing_residual", "pde_component"}
    if not term_names.intersection(governing_terms):
        _issue(
            errors,
            "loss.terms",
            "GOVERNING_LOSS_REQUIRED",
            "At least one positive-weight governing PDE loss term is required",
        )
    interior_points = int(
        ((spec["sampling"].get("interior") or {}).get("parameters") or {}).get(
            "n_points", 0
        )
        or 0
    )
    if interior_points <= 0:
        _issue(
            errors,
            "sampling.interior.parameters.n_points",
            "INTERIOR_SAMPLES_REQUIRED",
            "A governing PDE loss requires positive interior sample coverage",
        )
    if method == "soft_penalty":
        for category, loss_name in (
            ("initial", "initial_condition"),
            ("boundary", "boundary_condition"),
        ):
            if category not in constraint_ids:
                continue
            if loss_name not in term_names:
                _issue(
                    errors,
                    "loss.terms",
                    f"{category.upper()}_LOSS_REQUIRED_FOR_SOFT_PENALTY",
                    f"soft_penalty requires a positive-weight {loss_name} term",
                )
            sample_points = int(
                ((spec["sampling"].get(category) or {}).get("parameters") or {}).get(
                    "n_points", 0
                )
                or 0
            )
            if sample_points <= 0:
                _issue(
                    errors,
                    f"sampling.{category}.parameters.n_points",
                    f"{category.upper()}_SAMPLES_REQUIRED_FOR_SOFT_PENALTY",
                    f"soft_penalty requires positive {category} sample coverage",
                )
    if "initial_condition" in term_names and "initial" not in constraint_ids:
        _issue(errors, "loss.terms", "INITIAL_CONDITION_UNAVAILABLE", "Problem has no initial condition")
    if "boundary_condition" in term_names and "boundary" not in constraint_ids:
        _issue(errors, "loss.terms", "BOUNDARY_CONDITION_UNAVAILABLE", "Problem has no boundary condition")
    for index, term in enumerate(loss_terms):
        name = term.get("name")
        parameters = term.get("parameters") or {}
        if name == "pde_component" and not any(
            key in parameters for key in ("component_name", "component_index")
        ):
            _issue(
                errors,
                f"loss.terms[{index}].parameters",
                "PDE_COMPONENT_SELECTOR_REQUIRED",
                "pde_component requires component_name or component_index",
            )
        if name == "constraint_component":
            component_name = str(parameters.get("component_name") or "")
            if not component_name:
                _issue(
                    errors,
                    f"loss.terms[{index}].parameters.component_name",
                    "CONSTRAINT_COMPONENT_NAME_REQUIRED",
                    "constraint_component requires component_name",
                )
            elif component_name not in constraint_ids:
                _issue(
                    errors,
                    f"loss.terms[{index}].parameters.component_name",
                    "CONSTRAINT_COMPONENT_UNAVAILABLE",
                    f"Problem has no constraint component '{component_name}'",
                )
    phases = spec["optimization"].get("phases") or []
    multiadam_indices = [index for index, phase in enumerate(phases) if phase.get("optimizer") == "multiadam"]
    if multiadam_indices:
        if len(spec["loss"].get("terms") or []) < 2:
            _issue(errors, "loss.terms", "MULTIADAM_REQUIRES_MULTIPLE_LOSSES", "MultiAdam requires at least two loss terms")
        if spec["training"]["gradient_clipping"].get("enabled"):
            _issue(errors, "training.gradient_clipping", "MULTIADAM_CLIPPING_UNSUPPORTED", "MultiAdam currently does not support gradient clipping")
        for index in multiadam_indices:
            group_weights = (phases[index].get("parameters") or {}).get("group_weights")
            if group_weights is not None and len(group_weights) != len(spec["loss"].get("terms") or []):
                _issue(
                    errors,
                    f"optimization.phases[{index}].parameters.group_weights",
                    "MULTIADAM_WEIGHT_COUNT_MISMATCH",
                    "MultiAdam group_weights must match the number of loss terms",
                )
    lbfgs_indices = [index for index, phase in enumerate(phases) if phase.get("optimizer") == "lbfgs"]
    if lbfgs_indices:
        if spec["training"].get("batch_mode") != "full_batch":
            _issue(errors, "training.batch_mode", "LBFGS_REQUIRES_FULL_BATCH", "L-BFGS requires full_batch training")
        if lbfgs_indices[-1] != len(phases) - 1 or len(lbfgs_indices) > 1:
            _issue(errors, "optimization.phases", "LBFGS_MUST_BE_LAST", "The current trainer supports one final L-BFGS phase")
    adaptive = spec["sampling"]["adaptive_refinement"]
    if adaptive.get("enabled") and not callable(getattr(problem, "compute_governing_residuals", None)):
        _issue(errors, "sampling.adaptive_refinement", "RAR_REQUIRES_RESIDUAL", "Adaptive refinement requires a residual evaluator")


def _validate_budget(spec: dict[str, Any], problem: Any, config: dict[str, Any], errors: list[ValidationIssue]) -> dict[str, int]:
    total_iterations = sum(int(phase.get("iterations") or 0) for phase in spec["optimization"].get("phases") or [])
    sampling_points = sum(
        int(((spec["sampling"].get(section) or {}).get("parameters") or {}).get("n_points") or 0)
        for section in ("interior", "boundary", "initial")
    )
    problem_spec = problem.get_spec()
    input_dim = len(problem_spec.input_variables)
    output_dim = len(problem_spec.output_variables)
    network = spec["network"]
    architecture = network.get("architecture")
    extra = network.get("extra_parameters") or {}
    effective_input = input_dim
    extra_trainable = 0
    if architecture == "fourier_mlp":
        frequencies = int(extra.get("num_frequencies", 8))
        effective_input = input_dim * frequencies * 2 + (input_dim if extra.get("include_raw_input", False) else 0)
        extra_trainable = frequencies if extra.get("trainable", False) else 0
    elif architecture == "multiscale_mlp":
        effective_input = input_dim * len(extra.get("scales") or [1.0, 2.0, 4.0])
    hidden_layers = list(network.get("hidden_layers", []))
    widths = [effective_input, *hidden_layers, output_dim]
    if architecture in {"modified_mlp", "gated_mlp"}:
        model_parameters = sum(
            2 * (input_dim + 1) * right + (left + 1) * right
            for left, right in zip([input_dim, *hidden_layers[:-1]], hidden_layers)
        )
        model_parameters += (hidden_layers[-1] + 1) * output_dim
    elif architecture == "factorized_mlp":
        configured_rank = max(1, int(extra.get("rank", 16)))
        model_parameters = 0
        for left, right in zip([input_dim, *hidden_layers[:-1]], hidden_layers):
            rank = min(configured_rank, left, right)
            model_parameters += left * rank + (rank + 1) * right
        model_parameters += (hidden_layers[-1] + 1) * output_dim
    elif architecture in {"laaf", "laaf_mlp", "gaaf", "gaaf_mlp"}:
        model_parameters = sum((left + 1) * right for left, right in zip(widths, widths[1:]))
        model_parameters += sum(hidden_layers) if architecture in {"laaf", "laaf_mlp"} else len(hidden_layers)
    elif architecture in {"parallel_fnn", "pfnn"}:
        branch_widths = [input_dim, *hidden_layers, 1]
        model_parameters = output_dim * sum(
            (left + 1) * right for left, right in zip(branch_widths, branch_widths[1:])
        )
    else:
        model_parameters = sum((left + 1) * right for left, right in zip(widths, widths[1:]))
    residual_enabled = architecture in {"resnet", "residual_mlp"} or bool(network["residual_connections"].get("enabled"))
    if residual_enabled:
        hidden_widths = [input_dim, *hidden_layers]
        residual_multiplier = output_dim if architecture in {"parallel_fnn", "pfnn"} else 1
        model_parameters += residual_multiplier * sum(
            left * right for left, right in zip(hidden_widths, hidden_widths[1:]) if left != right
        )
    model_parameters += extra_trainable
    usage = {
        "total_iterations": total_iterations,
        "sampling_points": sampling_points,
        "estimated_model_parameters": model_parameters,
    }
    configured_sampling_limit = config.get("maximum_sampling_points")
    limits = {
        "total_iterations": int(config.get("maximum_total_iterations", 2**63 - 1)),
        "sampling_points": (
            None if configured_sampling_limit is None else int(configured_sampling_limit)
        ),
        "estimated_model_parameters": int(config.get("maximum_model_parameters", 2**63 - 1)),
    }
    paths = {
        "total_iterations": "optimization.phases",
        "sampling_points": "sampling",
        "estimated_model_parameters": "network.hidden_layers",
    }
    for key, value in usage.items():
        if limits[key] is not None and value > limits[key]:
            _issue(errors, paths[key], "BUDGET_EXCEEDED", f"{key}={value} exceeds external limit {limits[key]}")
    required_iterations = config.get("required_total_iterations")
    if required_iterations is not None and total_iterations != int(required_iterations):
        _issue(
            errors,
            "optimization.phases",
            "REQUIRED_ITERATION_BUDGET_NOT_MET",
            f"total_iterations={total_iterations} must equal the required experiment budget {int(required_iterations)}",
        )
    return usage


def _walk_named_fields(value: Any, field_name: str, path: str = "$"):
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{path}.{key}" if path != "$" else key
            if key == field_name:
                yield child, item
            yield from _walk_named_fields(item, field_name, child)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk_named_fields(item, field_name, f"{path}[{index}]")


def _normalized_name(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _normalize_nested(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalize_nested(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_nested(item) for item in value]
    return value


def _sort_dicts(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sort_dicts(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_sort_dicts(item) for item in value]
    return value

# ---------------------------------------------------------------------------
# Deterministic PyTorch Trusted Builder
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BuildReport:
    status: str
    error_type: str
    path: str
    value: Any
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BuildError(ValueError):
    def __init__(self, report: BuildReport) -> None:
        super().__init__(report.message)
        self.report = report


@dataclass
class OptimizationPhase:
    name: str
    optimizer_name: str
    iterations: int
    optimizer: torch.optim.Optimizer
    scheduler: object | None
    parameters: dict[str, Any]


@dataclass
class BuiltPINN:
    model: nn.Module
    sampler: "OpenSpecSampler"
    loss_function: "OpenSpecLoss"
    optimization_phases: list[OptimizationPhase]
    normalized_spec: dict[str, Any]
    gradient_balance_controller: Any | None = None
    lbfgs_stall_controller: Any | None = None
    rollback_manager: Any | None = None
    adaptive_audit_logger: Any | None = None
    training_component_registry: Any | None = None
    action_boundary_validator: Any | None = None
    controller_coordinator: Any | None = None
    validation_probe_manager: Any | None = None
    best_training_loss_checkpoint_manager: Any | None = None
    fixed_budget_sampling_controller: Any | None = None
    curriculum_runtimes: dict[str, Any] = field(default_factory=dict)
    curriculum_advance_controller: Any | None = None


def _component_name(value: Any) -> str:
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


class SineActivation(nn.Module):
    def __init__(self, omega: float = 1.0) -> None:
        super().__init__()
        self.omega = float(omega)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega * inputs)


class GaussianActivation(nn.Module):
    def __init__(self, beta: float = 1.0) -> None:
        super().__init__()
        self.beta = float(beta)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.beta * inputs.square())


class LAAFActivation(nn.Module):
    """Trainable adaptive-slope tanh usable through the activation field."""

    def __init__(self, initial_slope: float = 1.0, activation_scale: float = 10.0) -> None:
        super().__init__()
        self.slope = nn.Parameter(torch.tensor(float(initial_slope)))
        self.activation_scale = float(activation_scale)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.activation_scale * self.slope * inputs)


class CosineActivation(nn.Module):
    """Frequency- and phase-controlled cosine activation."""

    def __init__(self, omega: float = 1.0, phase: float = 0.0) -> None:
        super().__init__()
        self.omega = float(omega)
        self.phase = float(phase)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.cos(self.omega * inputs + self.phase)


class ScaledTanhActivation(nn.Module):
    """Tanh with a fixed or trainable global input scale."""

    def __init__(self, scale: float = 1.0, trainable: bool = False) -> None:
        super().__init__()
        value = torch.tensor(float(scale))
        if trainable:
            self.scale = nn.Parameter(value)
        else:
            self.register_buffer("scale", value)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.scale * inputs)


class SnakeActivation(nn.Module):
    """Periodic Snake activation useful for oscillatory PDE solutions."""

    def __init__(self, alpha: float = 1.0, trainable: bool = True, epsilon: float = 1e-8) -> None:
        super().__init__()
        if alpha <= 0.0 or epsilon <= 0.0:
            raise ValueError("Snake alpha and epsilon must be positive")
        value = torch.tensor(float(alpha))
        if trainable:
            self.alpha = nn.Parameter(value)
        else:
            self.register_buffer("alpha", value)
        self.epsilon = float(epsilon)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        denominator = self.alpha.abs().clamp_min(self.epsilon)
        return inputs + torch.sin(self.alpha * inputs).square() / denominator


class StanActivation(nn.Module):
    """Self-scalable tanh activation with a trainable global beta."""

    def __init__(self, initial_beta: float = 1.0) -> None:
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(float(initial_beta)))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        tanh_value = torch.tanh(inputs)
        return tanh_value + self.beta * inputs * tanh_value


ACTIVATIONS: dict[str, type[nn.Module]] = {
    "identity": nn.Identity,
    "tanh": nn.Tanh,
    "silu": nn.SiLU,
    "swish": nn.SiLU,
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
    "softplus": nn.Softplus,
    "sigmoid": nn.Sigmoid,
    "mish": nn.Mish,
    "softsign": nn.Softsign,
    "tanhshrink": nn.Tanhshrink,
    "hardtanh": nn.Hardtanh,
    "celu": nn.CELU,
    "selu": nn.SELU,
    "prelu": nn.PReLU,
    "relu6": nn.ReLU6,
    "hardswish": nn.Hardswish,
    "hard_swish": nn.Hardswish,
    "hardsigmoid": nn.Hardsigmoid,
    "hard_sigmoid": nn.Hardsigmoid,
    "log_sigmoid": nn.LogSigmoid,
    "softshrink": nn.Softshrink,
    "hardshrink": nn.Hardshrink,
    "threshold": nn.Threshold,
    "rrelu": nn.RReLU,
    "randomized_leaky_relu": nn.RReLU,
    "sine": SineActivation,
    "sin": SineActivation,
    "cosine": CosineActivation,
    "cos": CosineActivation,
    "gaussian": GaussianActivation,
    "laaf": LAAFActivation,
    "scaled_tanh": ScaledTanhActivation,
    "snake": SnakeActivation,
    "stan": StanActivation,
}


def _siren_initializer_marker(
    tensor: torch.Tensor, initialization_scale: float = 1.0
) -> torch.Tensor:
    """Registry marker; SirenPINN applies layer-aware initialization itself."""

    del initialization_scale
    return tensor


INITIALIZERS: dict[str, Callable[..., Any]] = {
    "xavier_normal": nn.init.xavier_normal_,
    "xavier_uniform": nn.init.xavier_uniform_,
    "glorot_normal": nn.init.xavier_normal_,
    "glorot_uniform": nn.init.xavier_uniform_,
    "kaiming_normal": nn.init.kaiming_normal_,
    "kaiming_uniform": nn.init.kaiming_uniform_,
    "orthogonal": nn.init.orthogonal_,
    "normal": nn.init.normal_,
    "uniform": nn.init.uniform_,
    "trunc_normal": nn.init.trunc_normal_,
    "sparse": nn.init.sparse_,
    "zeros": nn.init.zeros_,
    "ones": nn.init.ones_,
    "constant": nn.init.constant_,
    "eye": nn.init.eye_,
    "siren": _siren_initializer_marker,
}

COMPONENT_PARAMETER_DEFAULTS: dict[str, dict[str, Any]] = {
    "threshold": {"threshold": 0.0, "value": 0.0},
}

INITIALIZER_PARAMETER_DEFAULTS: dict[str, dict[str, Any]] = {
    "sparse": {"sparsity": 0.1},
    "constant": {"val": 0.0},
}


class MultiAdamOptimizer(torch.optim.Optimizer):
    """Loss-group Adam used by PINNacle for multi-objective PINN training."""

    def __init__(
        self,
        params: Any,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.99, 0.99),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        group_weights: list[float] | None = None,
        normalize_updates: bool | None = None,
        grouping: str | None = None,
    ) -> None:
        if lr < 0.0 or eps <= 0.0 or weight_decay < 0.0:
            raise ValueError("Invalid MultiAdam learning rate, epsilon, or weight decay")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid MultiAdam beta parameters")
        if grouping not in {
            None,
            "pde_vs_constraints",
            "dirichlet_vs_non_dirichlet",
        }:
            raise ValueError(f"Unknown MultiAdam loss grouping: {grouping!r}")
        # Preserve the pre-existing per-loss optimizer when no grouping is
        # supplied.  Explicit grouping selects the PINNacle-compatible path,
        # which does not L2-normalize each group's Adam update.
        resolved_normalize = (
            grouping is None if normalize_updates is None else bool(normalize_updates)
        )
        defaults = {
            "lr": float(lr),
            "betas": tuple(float(value) for value in betas),
            "eps": float(eps),
            "weight_decay": float(weight_decay),
            "normalize_updates": resolved_normalize,
        }
        super().__init__(params, defaults)
        self.configured_group_weights = list(group_weights) if group_weights is not None else None
        self.grouping = grouping
        self.last_loss_groups: dict[str, list[str]] = {}

    @torch.no_grad()
    def step_losses(
        self,
        losses: list[torch.Tensor],
        loss_names: list[str] | None = None,
    ) -> None:
        if not losses:
            raise ValueError("MultiAdam requires at least one loss tensor")
        parameters = [
            parameter
            for group in self.param_groups
            for parameter in group["params"]
            if parameter.requires_grad
        ]
        gradients_by_loss: list[tuple[torch.Tensor | None, ...]] = []
        with torch.enable_grad():
            grouped_losses, group_names = self._group_losses(
                losses, loss_names
            )
            for index, loss in enumerate(grouped_losses):
                gradients_by_loss.append(
                    torch.autograd.grad(
                        loss,
                        parameters,
                        retain_graph=index < len(grouped_losses) - 1,
                        create_graph=False,
                        allow_unused=True,
                    )
                )
        if self.configured_group_weights is None:
            weights = torch.full(
                (len(grouped_losses),),
                1.0 / len(grouped_losses),
                dtype=parameters[0].dtype,
                device=parameters[0].device,
            )
        else:
            if len(self.configured_group_weights) != len(grouped_losses):
                raise ValueError("MultiAdam group_weights length must match the number of loss terms")
            weights = torch.tensor(
                self.configured_group_weights,
                dtype=parameters[0].dtype,
                device=parameters[0].device,
            )
            if self.grouping is None:
                # Compatibility for historical per-loss MultiAdam candidates.
                weights = weights / torch.clamp(weights.abs().sum(), min=1e-12)

        parameter_index = 0
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if not parameter.requires_grad:
                    continue
                gradients = [items[parameter_index] for items in gradients_by_loss]
                parameter_index += 1
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = [
                        torch.zeros_like(parameter) for _ in grouped_losses
                    ]
                    state["exp_avg_sq"] = [
                        torch.zeros_like(parameter) for _ in grouped_losses
                    ]
                elif len(state["exp_avg"]) != len(grouped_losses):
                    raise RuntimeError(
                        "MultiAdam loss-group count changed during training"
                    )
                state["step"] += 1
                step = state["step"]
                updates: list[torch.Tensor] = []
                for loss_index, gradient in enumerate(gradients):
                    gradient = torch.zeros_like(parameter) if gradient is None else gradient
                    if group["weight_decay"]:
                        gradient = gradient.add(parameter, alpha=group["weight_decay"])
                    exp_avg = state["exp_avg"][loss_index]
                    exp_avg_sq = state["exp_avg_sq"][loss_index]
                    exp_avg.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
                    corrected_avg = exp_avg / (1.0 - beta1**step)
                    corrected_sq = exp_avg_sq / (1.0 - beta2**step)
                    updates.append(corrected_avg / (corrected_sq.sqrt() + group["eps"]))
                stacked = torch.stack(updates)
                if group["normalize_updates"]:
                    norms = stacked.flatten(start_dim=1).norm(dim=1).clamp_min(1e-12)
                    stacked = stacked / norms.reshape((-1,) + (1,) * parameter.ndim)
                combined = torch.sum(
                    weights.reshape((-1,) + (1,) * parameter.ndim) * stacked,
                    dim=0,
                )
                parameter.add_(combined, alpha=-group["lr"])
        self.last_loss_groups = group_names

    def _group_losses(
        self,
        losses: list[torch.Tensor],
        loss_names: list[str] | None,
    ) -> tuple[list[torch.Tensor], dict[str, list[str]]]:
        if self.grouping is None:
            names = list(loss_names or [f"loss_{index}" for index in range(len(losses))])
            return list(losses), {
                f"loss_{index}": [name] for index, name in enumerate(names)
            }
        if loss_names is None or len(loss_names) != len(losses):
            raise ValueError(
                "Explicit MultiAdam grouping requires one stable name per loss"
            )
        if self.grouping == "pde_vs_constraints":
            memberships = {
                "pde": {
                    name for name in loss_names if _is_governing_component_id(name)
                },
                "constraints": set(loss_names),
            }
            memberships["constraints"] -= memberships["pde"]
        else:
            memberships = {
                "dirichlet": {
                    "initial_displacement",
                    "spatial_boundary",
                },
                "non_dirichlet": {
                    "initial_velocity",
                }
                | {
                    name
                    for name in loss_names
                    if _is_governing_component_id(name)
                },
            }
        unknown = set(loss_names) - set().union(*memberships.values())
        if unknown:
            raise ValueError(
                f"MultiAdam grouping {self.grouping!r} does not classify {sorted(unknown)}"
            )
        grouped: list[torch.Tensor] = []
        audit: dict[str, list[str]] = {}
        for group_name, members in memberships.items():
            selected = [
                loss
                for name, loss in zip(loss_names, losses)
                if name in members
            ]
            selected_names = [
                name for name in loss_names if name in members
            ]
            if not selected:
                raise ValueError(
                    f"MultiAdam group {group_name!r} has no active loss"
                )
            grouped.append(torch.stack(selected).sum())
            audit[group_name] = selected_names
        return grouped, audit

    def step(self, closure: Callable[..., Any] | None = None):
        raise RuntimeError("MultiAdamOptimizer requires step_losses(loss_tensors) through OpenSpecTrainer")


OPTIMIZERS: dict[str, type[torch.optim.Optimizer]] = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
    "rmsprop": torch.optim.RMSprop,
    "radam": torch.optim.RAdam,
    "nadam": torch.optim.NAdam,
    "adagrad": torch.optim.Adagrad,
    "adadelta": torch.optim.Adadelta,
    "adamax": torch.optim.Adamax,
    "asgd": torch.optim.ASGD,
    "rprop": torch.optim.Rprop,
    "multiadam": MultiAdamOptimizer,
    "lbfgs": torch.optim.LBFGS,
}

# Newer PyTorch releases expose additional dense optimizers.  Register them
# conditionally so the same source remains importable on older installations.
if hasattr(torch.optim, "Adafactor"):
    OPTIMIZERS["adafactor"] = torch.optim.Adafactor

SCHEDULERS: dict[str, type | None] = {
    "none": None,
    "step_lr": torch.optim.lr_scheduler.StepLR,
    "steplr": torch.optim.lr_scheduler.StepLR,
    "cosine_annealing": torch.optim.lr_scheduler.CosineAnnealingLR,
    "cosineannealinglr": torch.optim.lr_scheduler.CosineAnnealingLR,
    "cosine_warm_restarts": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    "cosine_annealing_warm_restarts": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    "cosineannealingwarmrestarts": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    "exponential_lr": torch.optim.lr_scheduler.ExponentialLR,
    "exponentiallr": torch.optim.lr_scheduler.ExponentialLR,
    "reduce_on_plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    "reducelronplateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    "one_cycle": torch.optim.lr_scheduler.OneCycleLR,
    "onecyclelr": torch.optim.lr_scheduler.OneCycleLR,
    "linear_lr": torch.optim.lr_scheduler.LinearLR,
    "polynomial_lr": torch.optim.lr_scheduler.PolynomialLR,
    "constant_lr": torch.optim.lr_scheduler.ConstantLR,
    "multi_step_lr": torch.optim.lr_scheduler.MultiStepLR,
    "cyclic_lr": torch.optim.lr_scheduler.CyclicLR,
}


def _mse_loss(value: torch.Tensor, _: dict[str, Any]) -> torch.Tensor:
    return torch.mean(value.square())


def _mae_loss(value: torch.Tensor, _: dict[str, Any]) -> torch.Tensor:
    return torch.mean(value.abs())


def _rmse_loss(value: torch.Tensor, parameters: dict[str, Any]) -> torch.Tensor:
    epsilon = float(parameters.get("epsilon", 1e-12))
    return torch.sqrt(torch.mean(value.square()) + epsilon)


def _huber_loss(value: torch.Tensor, parameters: dict[str, Any]) -> torch.Tensor:
    delta = float(parameters.get("delta", 1.0))
    return torch.nn.functional.huber_loss(value, torch.zeros_like(value), delta=delta)


def _smooth_l1_loss(value: torch.Tensor, parameters: dict[str, Any]) -> torch.Tensor:
    beta = float(parameters.get("beta", 1.0))
    return torch.nn.functional.smooth_l1_loss(value, torch.zeros_like(value), beta=beta)


def _log_cosh_loss(value: torch.Tensor, parameters: dict[str, Any]) -> torch.Tensor:
    scale = float(parameters.get("scale", 1.0))
    scaled = value / scale
    return torch.mean(scaled + torch.nn.functional.softplus(-2.0 * scaled) - math.log(2.0)) * scale


def _cauchy_loss(value: torch.Tensor, parameters: dict[str, Any]) -> torch.Tensor:
    scale = float(parameters.get("scale", 1.0))
    return torch.mean(torch.log1p((value / scale).square())) * (scale * scale)


def _charbonnier_loss(value: torch.Tensor, parameters: dict[str, Any]) -> torch.Tensor:
    epsilon = float(parameters.get("epsilon", 1e-3))
    alpha = float(parameters.get("alpha", 0.5))
    return torch.mean((value.square() + epsilon * epsilon).pow(alpha))


def _pseudo_huber_loss(value: torch.Tensor, parameters: dict[str, Any]) -> torch.Tensor:
    delta = float(parameters.get("delta", 1.0))
    return torch.mean(delta * delta * (torch.sqrt(1.0 + (value / delta).square()) - 1.0))


def _lp_loss(value: torch.Tensor, parameters: dict[str, Any]) -> torch.Tensor:
    p = float(parameters.get("p", 2.0))
    epsilon = float(parameters.get("epsilon", 1e-12))
    root = bool(parameters.get("root", False))
    moment = torch.mean((value.abs() + epsilon).pow(p))
    return moment.pow(1.0 / p) if root else moment


def _tukey_loss(value: torch.Tensor, parameters: dict[str, Any]) -> torch.Tensor:
    c = float(parameters.get("c", 4.685))
    scaled = value / c
    inside = scaled.abs() <= 1.0
    bounded = (c * c / 6.0) * (1.0 - (1.0 - scaled.square()).pow(3))
    saturated = torch.full_like(value, c * c / 6.0)
    return torch.mean(torch.where(inside, bounded, saturated))


def _geman_mcclure_loss(value: torch.Tensor, parameters: dict[str, Any]) -> torch.Tensor:
    scale = float(parameters.get("scale", 1.0))
    squared = value.square()
    return torch.mean(squared / (squared + scale * scale)) * (scale * scale)


def _log_l2_loss(value: torch.Tensor, parameters: dict[str, Any]) -> torch.Tensor:
    epsilon = float(parameters.get("epsilon", 1e-12))
    return torch.log1p(torch.mean(value.square()) + epsilon)


def _quantile_loss(value: torch.Tensor, parameters: dict[str, Any]) -> torch.Tensor:
    quantile = float(parameters.get("quantile", 0.5))
    return torch.mean(torch.maximum(quantile * value, (quantile - 1.0) * value))


LOSS_FUNCTIONS: dict[str, Callable[[torch.Tensor, dict[str, Any]], torch.Tensor]] = {
    "mse": _mse_loss,
    "mae": _mae_loss,
    "rmse": _rmse_loss,
    "huber": _huber_loss,
    "smooth_l1": _smooth_l1_loss,
    "log_cosh": _log_cosh_loss,
    "cauchy": _cauchy_loss,
    "charbonnier": _charbonnier_loss,
    "pseudo_huber": _pseudo_huber_loss,
    "lp": _lp_loss,
    "tukey": _tukey_loss,
    "geman_mcclure": _geman_mcclure_loss,
    "log_l2": _log_l2_loss,
    "quantile": _quantile_loss,
}

LOSS_PARAMETER_CONTRACTS: dict[str, set[str]] = {
    "mse": set(),
    "mae": set(),
    "rmse": {"epsilon"},
    "huber": {"delta"},
    "smooth_l1": {"beta"},
    "log_cosh": {"scale"},
    "cauchy": {"scale"},
    "charbonnier": {"epsilon", "alpha"},
    "pseudo_huber": {"delta"},
    "lp": {"p", "epsilon", "root"},
    "tukey": {"c"},
    "geman_mcclure": {"scale"},
    "log_l2": {"epsilon"},
    "quantile": {"quantile"},
}

WEIGHTING_PARAMETER_CONTRACTS: dict[str, set[str]] = {
    "fixed": set(),
    "normalized": {"epsilon"},
    "softmax": {"temperature"},
    "inverse_magnitude": {"epsilon", "power", "normalize"},
    "lra": {
        "alpha", "epsilon", "reference_index", "min_weight", "max_weight",
        "minimum_update_ratio", "maximum_update_ratio", "cooldown_iterations",
    },
    "gradient_balance": {
        "ema_beta", "diagnostic_interval", "patience", "cooldown_iterations",
        "imbalance_threshold", "conflict_threshold", "minimum_weight",
        "maximum_weight", "minimum_update_ratio", "maximum_update_ratio",
        "alpha", "epsilon",
    },
    "ntk": {"epsilon", "min_weight", "max_weight", "update_every"},
    "ntk_trace": {"alpha", "epsilon", "normalize"},
    "ema_inverse_magnitude": {"alpha", "epsilon", "power", "normalize"},
    "loss_softmax": {"temperature", "normalize"},
}

AGGREGATION_PARAMETER_CONTRACTS: dict[str, set[str]] = {
    "weighted_sum": set(),
    "mean": set(),
    "max": set(),
    "logsumexp": {"temperature"},
    "p_norm": {"p", "epsilon"},
    "root_mean_square": {"epsilon"},
}

LOSS_TERM_PARAMETER_CONTRACTS: dict[str, set[str]] = {
    "pde_residual": {"component_index", "component_name"},
    "governing_residual": {"component_index", "component_name"},
    "pde_component": {"component_index", "component_name"},
    "boundary_condition": set(),
    "spatial_boundary": set(),
    "initial_condition": set(),
    "initial_displacement": set(),
    "initial_velocity": {"derivative_variable", "derivative_order"},
    "conservation": set(),
    "constraint_component": {"component_name"},
    "gradient_residual": {"component_index", "component_name", "input_indices"},
}


class ActivationWithDropout(nn.Module):
    """Apply a shared activation followed by dropout at every hidden layer."""

    def __init__(self, activation: nn.Module, dropout_rate: float = 0.0) -> None:
        super().__init__()
        rate = float(dropout_rate)
        if not 0.0 <= rate < 1.0:
            raise ValueError("dropout_rate must be in [0, 1)")
        self.activation = activation
        self.dropout = nn.Dropout(rate) if rate > 0.0 else nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.activation(inputs))


class AffineInputTransform(nn.Module):
    """Apply a scalar affine map after the problem-owned input transform."""

    def __init__(self, base: nn.Module, scale: float = 1.0, shift: float = 0.0) -> None:
        super().__init__()
        self.base = base
        if hasattr(base, "output_dim"):
            self.output_dim = int(getattr(base, "output_dim"))
        self.register_buffer("scale", torch.tensor(float(scale)))
        self.register_buffer("shift", torch.tensor(float(shift)))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.base(inputs) * self.scale + self.shift


class OutputAffineWrapper(nn.Module):
    """Apply a scalar affine map to the final network prediction."""

    def __init__(self, model: nn.Module, scale: float = 1.0, shift: float = 0.0) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("scale", torch.tensor(float(scale)))
        self.register_buffer("shift", torch.tensor(float(shift)))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs) * self.scale + self.shift


class ConstraintOutputWrapper(nn.Module):
    """Apply a registered problem-owned hard transform to network outputs."""

    def __init__(
        self,
        model: nn.Module,
        transform: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        *,
        transform_id: str,
    ) -> None:
        super().__init__()
        self.model = model
        self.transform = transform
        self.transform_id = str(transform_id)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.transform(inputs, self.model(inputs))


class BoundsTransform(nn.Module):
    def __init__(self, lower: torch.Tensor, upper: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("lower", lower)
        self.register_buffer("span", torch.clamp(upper - lower, min=1e-12))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return 2.0 * (inputs - self.lower) / self.span - 1.0


class UnitBoundsTransform(BoundsTransform):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return (inputs - self.lower) / self.span


class FlexiblePINN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: list[int],
        activation: nn.Module,
        output_activation: nn.Module,
        input_transform: nn.Module,
        *,
        residual_enabled: bool = False,
        residual_scale: float = 1.0,
        bias: bool = True,
        feature_transform: nn.Module | None = None,
        feature_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.input_transform = input_transform
        self.feature_transform = feature_transform or nn.Identity()
        self.activation = activation
        self.output_activation = output_activation
        self.residual_enabled = bool(residual_enabled)
        self.residual_scale = float(residual_scale)
        effective_input = int(feature_dim or input_dim)
        widths = [effective_input, *hidden_layers]
        self.hidden = nn.ModuleList(
            nn.Linear(left, right, bias=bias) for left, right in zip(widths, widths[1:])
        )
        self.projections = nn.ModuleList(
            (
                nn.Identity()
                if not self.residual_enabled or left == right
                else nn.Linear(left, right, bias=False)
            )
            for left, right in zip(widths, widths[1:])
        )
        self.output = nn.Linear(hidden_layers[-1], output_dim, bias=bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        h = self.feature_transform(self.input_transform(inputs))
        for linear, projection in zip(self.hidden, self.projections):
            updated = self.activation(linear(h))
            h = updated + self.residual_scale * projection(h) if self.residual_enabled else updated
        return self.output_activation(self.output(h))


class ModifiedPINN(nn.Module):
    """Gated PINN MLP with input-conditioned encoder branches."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: list[int],
        activation: nn.Module,
        output_activation: nn.Module,
        input_transform: nn.Module,
        *,
        residual_enabled: bool = False,
        residual_scale: float = 1.0,
        bias: bool = True,
        gate_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.input_transform = input_transform
        self.activation = activation
        self.output_activation = output_activation
        self.residual_enabled = bool(residual_enabled)
        self.residual_scale = float(residual_scale)
        self.gate_temperature = float(gate_temperature)
        widths = [input_dim, *hidden_layers]
        self.u_encoders = nn.ModuleList(nn.Linear(input_dim, width, bias=bias) for width in hidden_layers)
        self.v_encoders = nn.ModuleList(nn.Linear(input_dim, width, bias=bias) for width in hidden_layers)
        self.gates = nn.ModuleList(nn.Linear(left, right, bias=bias) for left, right in zip(widths, widths[1:]))
        self.projections = nn.ModuleList(
            nn.Identity() if not self.residual_enabled or left == right else nn.Linear(left, right, bias=False)
            for left, right in zip(widths, widths[1:])
        )
        self.output = nn.Linear(hidden_layers[-1], output_dim, bias=bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        transformed = self.input_transform(inputs)
        h = transformed
        temperature = max(abs(self.gate_temperature), 1e-8)
        for u_encoder, v_encoder, gate, projection in zip(
            self.u_encoders, self.v_encoders, self.gates, self.projections
        ):
            u = self.activation(u_encoder(transformed))
            v = self.activation(v_encoder(transformed))
            z = torch.sigmoid(gate(h) / temperature)
            updated = (1.0 - z) * u + z * v
            h = updated + self.residual_scale * projection(h) if self.residual_enabled else updated
        return self.output_activation(self.output(h))


class FactorizedPINN(nn.Module):
    """Low-rank factorized MLP for parameter-efficient PINN candidates."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: list[int],
        activation: nn.Module,
        output_activation: nn.Module,
        input_transform: nn.Module,
        *,
        residual_enabled: bool = False,
        residual_scale: float = 1.0,
        bias: bool = True,
        rank: int = 16,
    ) -> None:
        super().__init__()
        self.input_transform = input_transform
        self.activation = activation
        self.output_activation = output_activation
        self.residual_enabled = bool(residual_enabled)
        self.residual_scale = float(residual_scale)
        widths = [input_dim, *hidden_layers]
        self.factors_in = nn.ModuleList()
        self.factors_out = nn.ModuleList()
        self.projections = nn.ModuleList()
        for left, right in zip(widths, widths[1:]):
            effective_rank = max(1, min(int(rank), left, right))
            self.factors_in.append(nn.Linear(left, effective_rank, bias=False))
            self.factors_out.append(nn.Linear(effective_rank, right, bias=bias))
            self.projections.append(
                nn.Identity() if not self.residual_enabled or left == right else nn.Linear(left, right, bias=False)
            )
        self.output = nn.Linear(hidden_layers[-1], output_dim, bias=bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        h = self.input_transform(inputs)
        for factor_in, factor_out, projection in zip(self.factors_in, self.factors_out, self.projections):
            updated = self.activation(factor_out(factor_in(h)))
            h = updated + self.residual_scale * projection(h) if self.residual_enabled else updated
        return self.output_activation(self.output(h))


class AdaptiveActivationPINN(nn.Module):
    """PINNacle-style locally or globally adaptive activation network."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: list[int],
        activation: nn.Module,
        output_activation: nn.Module,
        input_transform: nn.Module,
        *,
        mode: str,
        residual_enabled: bool = False,
        residual_scale: float = 1.0,
        bias: bool = True,
        activation_scale: float = 10.0,
        initial_slope: float = 1.0,
    ) -> None:
        super().__init__()
        if mode not in {"local", "global"}:
            raise ValueError(f"Unsupported adaptive activation mode: {mode}")
        self.input_transform = input_transform
        self.activation = activation
        self.output_activation = output_activation
        self.mode = mode
        self.residual_enabled = bool(residual_enabled)
        self.residual_scale = float(residual_scale)
        self.activation_scale = float(activation_scale)
        widths = [input_dim, *hidden_layers]
        self.hidden = nn.ModuleList(nn.Linear(left, right, bias=bias) for left, right in zip(widths, widths[1:]))
        self.slopes = nn.ParameterList(
            nn.Parameter(torch.full((width if mode == "local" else 1,), float(initial_slope)))
            for width in hidden_layers
        )
        self.projections = nn.ModuleList(
            nn.Identity() if not self.residual_enabled or left == right else nn.Linear(left, right, bias=False)
            for left, right in zip(widths, widths[1:])
        )
        self.output = nn.Linear(hidden_layers[-1], output_dim, bias=bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        h = self.input_transform(inputs)
        for linear, slope, projection in zip(self.hidden, self.slopes, self.projections):
            updated = self.activation(self.activation_scale * slope * linear(h))
            h = updated + self.residual_scale * projection(h) if self.residual_enabled else updated
        return self.output_activation(self.output(h))


class ParallelPINN(nn.Module):
    """Independent output branches for coupled and inverse PDE systems."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: list[int],
        activation: nn.Module,
        output_activation: nn.Module,
        input_transform: nn.Module,
        *,
        residual_enabled: bool = False,
        residual_scale: float = 1.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            FlexiblePINN(
                input_dim=input_dim,
                output_dim=1,
                hidden_layers=hidden_layers,
                activation=deepcopy(activation),
                output_activation=deepcopy(output_activation),
                input_transform=deepcopy(input_transform),
                residual_enabled=residual_enabled,
                residual_scale=residual_scale,
                bias=bias,
            )
            for _ in range(output_dim)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.cat([branch(inputs) for branch in self.branches], dim=1)


class FourierFeatures(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_frequencies: int = 8,
        sigma: float = 1.0,
        include_raw_input: bool = False,
        trainable: bool = False,
        frequency_mode: str = "linear",
        min_frequency: float = 1.0,
        max_frequency: float | None = None,
        seed: int | None = None,
        use_pi: bool = True,
    ) -> None:
        super().__init__()
        num_frequencies = int(num_frequencies)
        if num_frequencies <= 0:
            raise ValueError("num_frequencies must be positive")
        sigma = float(sigma)
        if sigma <= 0.0:
            raise ValueError("sigma must be positive")
        minimum = float(min_frequency)
        maximum = float(max_frequency) if max_frequency is not None else float(num_frequencies)
        mode = _component_name(frequency_mode)
        if mode in {"linear", "log", "random_uniform"} and maximum <= minimum:
            raise ValueError("max_frequency must exceed min_frequency")
        generator = torch.Generator(device="cpu")
        if seed is not None:
            generator.manual_seed(int(seed))
        if mode == "linear":
            frequencies = torch.linspace(minimum, maximum, num_frequencies)
        elif mode == "log":
            if minimum <= 0.0 or maximum <= 0.0:
                raise ValueError("log Fourier frequencies require positive bounds")
            frequencies = torch.logspace(math.log10(minimum), math.log10(maximum), num_frequencies)
        elif mode == "random_uniform":
            if maximum <= minimum:
                raise ValueError("max_frequency must exceed min_frequency")
            frequencies = minimum + (maximum - minimum) * torch.rand(num_frequencies, generator=generator)
        elif mode == "random_normal":
            frequencies = torch.randn(num_frequencies, generator=generator).abs().clamp_min(1e-6)
        else:
            raise ValueError(f"Unsupported Fourier frequency_mode: {frequency_mode!r}")
        frequencies = frequencies * sigma
        if trainable:
            self.frequencies = nn.Parameter(frequencies)
        else:
            self.register_buffer("frequencies", frequencies)
        self.include_raw_input = bool(include_raw_input)
        self.angle_scale = math.pi if use_pi else 1.0
        self.output_dim = input_dim * num_frequencies * 2 + (input_dim if include_raw_input else 0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        angles = self.angle_scale * inputs.unsqueeze(-1) * self.frequencies
        encoded = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1).flatten(start_dim=1)
        return torch.cat([inputs, encoded], dim=1) if self.include_raw_input else encoded


class MultiScaleFourierFeatures(nn.Module):
    """Deterministic Fourier bands with optional trainable frequencies."""

    def __init__(
        self,
        input_dim: int,
        frequency_count: int = 8,
        frequency_scale: float = 1.0,
        frequency_scales: list[float] | tuple[float, ...] = (1.0, 2.0, 4.0),
        trainable_frequencies: bool = False,
        include_raw_input: bool = False,
    ) -> None:
        super().__init__()
        count = int(frequency_count)
        scale = float(frequency_scale)
        bands = [float(value) for value in frequency_scales]
        if count < 1 or scale <= 0.0 or not bands or any(value <= 0.0 for value in bands):
            raise ValueError("Fourier counts and scales must be positive")
        base = torch.arange(1, count + 1, dtype=torch.float32)
        frequencies = torch.cat([base * scale * band for band in bands])
        if trainable_frequencies:
            self.frequencies = nn.Parameter(frequencies)
        else:
            self.register_buffer("frequencies", frequencies)
        self.include_raw_input = bool(include_raw_input)
        self.output_dim = (
            input_dim * int(frequencies.numel()) * 2
            + (input_dim if self.include_raw_input else 0)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        angles = torch.pi * inputs.unsqueeze(-1) * self.frequencies
        encoded = torch.cat(
            [torch.sin(angles), torch.cos(angles)], dim=-1
        ).flatten(start_dim=1)
        return (
            torch.cat([inputs, encoded], dim=1)
            if self.include_raw_input
            else encoded
        )


class SirenPINN(nn.Module):
    """SIREN with distinct first/hidden frequencies and canonical initialization."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: list[int],
        output_activation: nn.Module,
        input_transform: nn.Module,
        *,
        bias: bool = True,
        omega_0: float = 30.0,
        hidden_omega_0: float = 30.0,
        initialization_scale: float = 1.0,
        **_: Any,
    ) -> None:
        super().__init__()
        self.input_transform = input_transform
        self.output_activation = output_activation
        self.omega_0 = float(omega_0)
        self.hidden_omega_0 = float(hidden_omega_0)
        self.initialization_scale = float(initialization_scale)
        widths = [int(input_dim), *[int(width) for width in hidden_layers]]
        self.hidden = nn.ModuleList(
            nn.Linear(left, right, bias=bias)
            for left, right in zip(widths, widths[1:])
        )
        self.output = nn.Linear(hidden_layers[-1], output_dim, bias=bias)
        self.reset_siren_parameters()

    def reset_siren_parameters(self, initialization_scale: float | None = None) -> None:
        scale = (
            self.initialization_scale
            if initialization_scale is None
            else float(initialization_scale)
        )
        with torch.no_grad():
            first = self.hidden[0]
            first_bound = scale / max(1, first.in_features)
            first.weight.uniform_(-first_bound, first_bound)
            if first.bias is not None:
                first.bias.zero_()
            for layer in self.hidden[1:]:
                bound = (
                    scale
                    * math.sqrt(6.0 / max(1, layer.in_features))
                    / max(abs(self.hidden_omega_0), 1.0e-12)
                )
                layer.weight.uniform_(-bound, bound)
                if layer.bias is not None:
                    layer.bias.zero_()
            output_bound = (
                scale
                * math.sqrt(6.0 / max(1, self.output.in_features))
                / max(abs(self.hidden_omega_0), 1.0e-12)
            )
            self.output.weight.uniform_(-output_bound, output_bound)
            if self.output.bias is not None:
                self.output.bias.zero_()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        h = self.input_transform(inputs)
        for index, layer in enumerate(self.hidden):
            omega = self.omega_0 if index == 0 else self.hidden_omega_0
            h = torch.sin(omega * layer(h))
        return self.output_activation(self.output(h))


def _build_mlp_network(**kwargs: Any) -> nn.Module:
    return FlexiblePINN(**kwargs)


def _build_resnet_network(**kwargs: Any) -> nn.Module:
    kwargs["residual_enabled"] = True
    return FlexiblePINN(**kwargs)


def _build_modified_network(*, extra_parameters: dict[str, Any], **kwargs: Any) -> nn.Module:
    parameters = _checked_kwargs(
        ModifiedPINN.__init__,
        extra_parameters,
        "network.extra_parameters",
        skip={
            "self", "input_dim", "output_dim", "hidden_layers", "activation", "output_activation",
            "input_transform", "residual_enabled", "residual_scale", "bias",
        },
    )
    return ModifiedPINN(**kwargs, **parameters)


def _build_factorized_network(*, extra_parameters: dict[str, Any], **kwargs: Any) -> nn.Module:
    parameters = _checked_kwargs(
        FactorizedPINN.__init__,
        extra_parameters,
        "network.extra_parameters",
        skip={
            "self", "input_dim", "output_dim", "hidden_layers", "activation", "output_activation",
            "input_transform", "residual_enabled", "residual_scale", "bias",
        },
    )
    return FactorizedPINN(**kwargs, **parameters)


def _build_adaptive_activation_network(*, extra_parameters: dict[str, Any], mode: str, **kwargs: Any) -> nn.Module:
    parameters = _checked_kwargs(
        AdaptiveActivationPINN.__init__,
        extra_parameters,
        "network.extra_parameters",
        skip={
            "self", "input_dim", "output_dim", "hidden_layers", "activation", "output_activation",
            "input_transform", "mode", "residual_enabled", "residual_scale", "bias",
        },
    )
    return AdaptiveActivationPINN(**kwargs, mode=mode, **parameters)


def _build_laaf_network(*, extra_parameters: dict[str, Any], **kwargs: Any) -> nn.Module:
    return _build_adaptive_activation_network(extra_parameters=extra_parameters, mode="local", **kwargs)


def _build_gaaf_network(*, extra_parameters: dict[str, Any], **kwargs: Any) -> nn.Module:
    return _build_adaptive_activation_network(extra_parameters=extra_parameters, mode="global", **kwargs)


def _build_parallel_network(**kwargs: Any) -> nn.Module:
    return ParallelPINN(**kwargs)


def _build_fourier_network(*, extra_parameters: dict[str, Any], input_dim: int, **kwargs: Any) -> nn.Module:
    extra_parameters = dict(extra_parameters)
    aliases = {
        "frequency_count": "num_frequencies",
        "frequency_scale": "sigma",
        "trainable_frequencies": "trainable",
    }
    for alias, canonical in aliases.items():
        if alias in extra_parameters:
            if canonical in extra_parameters:
                _raise(
                    "invalid_parameter",
                    "network.extra_parameters",
                    [alias, canonical],
                    f"Use either '{alias}' or '{canonical}', not both",
                )
            extra_parameters[canonical] = extra_parameters.pop(alias)
    params = _checked_kwargs(
        FourierFeatures.__init__,
        extra_parameters,
        "network.extra_parameters",
        skip={"self", "input_dim"},
    )
    feature_transform = FourierFeatures(input_dim=input_dim, **params)
    return FlexiblePINN(
        input_dim=input_dim,
        feature_transform=feature_transform,
        feature_dim=feature_transform.output_dim,
        **kwargs,
    )


def _build_multiscale_fourier_network(
    *, extra_parameters: dict[str, Any], input_dim: int, **kwargs: Any
) -> nn.Module:
    params = _checked_kwargs(
        MultiScaleFourierFeatures.__init__,
        extra_parameters,
        "network.extra_parameters",
        skip={"self", "input_dim"},
    )
    feature_transform = MultiScaleFourierFeatures(input_dim=input_dim, **params)
    return FlexiblePINN(
        input_dim=input_dim,
        feature_transform=feature_transform,
        feature_dim=feature_transform.output_dim,
        **kwargs,
    )


def _build_siren_network(
    *, extra_parameters: dict[str, Any], **kwargs: Any
) -> nn.Module:
    params = _checked_kwargs(
        SirenPINN.__init__,
        extra_parameters,
        "network.extra_parameters",
        skip={
            "self",
            "input_dim",
            "output_dim",
            "hidden_layers",
            "activation",
            "output_activation",
            "input_transform",
            "residual_enabled",
            "residual_scale",
            "bias",
        },
    )
    kwargs.pop("activation", None)
    kwargs.pop("residual_enabled", None)
    kwargs.pop("residual_scale", None)
    return SirenPINN(**kwargs, **params)


def _build_multiscale_network(*, extra_parameters: dict[str, Any], input_dim: int, **kwargs: Any) -> nn.Module:
    scales = list(extra_parameters.get("scales") or [1.0, 2.0, 4.0])
    unknown = set(extra_parameters) - {"scales"}
    if unknown:
        _raise("unsupported_parameter", "network.extra_parameters", sorted(unknown), f"Unsupported multiscale_mlp parameters: {sorted(unknown)}")

    class MultiScale(nn.Module):
        def __init__(self, values: list[float]) -> None:
            super().__init__()
            self.register_buffer("scales", torch.tensor(values, dtype=torch.float32))

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return (inputs.unsqueeze(-1) * self.scales).flatten(start_dim=1)

    transform = MultiScale([float(value) for value in scales])
    return FlexiblePINN(
        input_dim=input_dim,
        feature_transform=transform,
        feature_dim=input_dim * len(scales),
        **kwargs,
    )


NETWORK_BUILDERS: dict[str, Callable[..., nn.Module]] = {
    "mlp": _build_mlp_network,
    "resnet": _build_resnet_network,
    "residual_mlp": _build_resnet_network,
    "fourier_mlp": _build_fourier_network,
    "multiscale_fourier_mlp": _build_multiscale_fourier_network,
    "siren_mlp": _build_siren_network,
    "multiscale_mlp": _build_multiscale_network,
    "modified_mlp": _build_modified_network,
    "gated_mlp": _build_modified_network,
    "factorized_mlp": _build_factorized_network,
    "laaf": _build_laaf_network,
    "laaf_mlp": _build_laaf_network,
    "gaaf": _build_gaaf_network,
    "gaaf_mlp": _build_gaaf_network,
    "parallel_fnn": _build_parallel_network,
    "pfnn": _build_parallel_network,
}

NETWORK_EXTRA_PARAMETER_CONTRACTS: dict[str, list[str]] = {
    "mlp": ["bias"],
    "resnet": ["bias"],
    "residual_mlp": ["bias"],
    "fourier_mlp": [
        "bias", "include_raw_input", "num_frequencies", "sigma", "trainable",
        "frequency_mode", "min_frequency", "max_frequency", "seed", "use_pi",
        "frequency_count", "frequency_scale", "trainable_frequencies",
    ],
    "multiscale_fourier_mlp": [
        "bias", "frequency_count", "frequency_scale", "frequency_scales",
        "trainable_frequencies", "include_raw_input",
    ],
    "siren_mlp": [
        "bias", "omega_0", "hidden_omega_0", "initialization_scale",
    ],
    "multiscale_mlp": ["bias", "scales"],
    "modified_mlp": ["bias", "gate_temperature"],
    "gated_mlp": ["bias", "gate_temperature"],
    "factorized_mlp": ["bias", "rank"],
    "laaf": ["bias", "activation_scale", "initial_slope"],
    "laaf_mlp": ["bias", "activation_scale", "initial_slope"],
    "gaaf": ["bias", "activation_scale", "initial_slope"],
    "gaaf_mlp": ["bias", "activation_scale", "initial_slope"],
    "parallel_fnn": ["bias"],
    "pfnn": ["bias"],
}

INPUT_TRANSFORMS = {
    "none",
    "normalize_bounds",
    "unit_bounds",
    "fourier_features",
    "multiscale_fourier_features",
    "periodic_positional_encoding",
}
SAMPLERS = {"uniform", "latin_hypercube", "sobol", "halton", "hammersley", "grid"}
CUSTOM_INTERIOR_SAMPLERS: dict[str, Callable[["OpenSpecSampler", int], torch.Tensor]] = {}
CONSTRAINT_SAMPLERS = {"uniform"}
ADAPTIVE_SAMPLERS = {
    "rar",
    "rad",
    "rar_d",
    "residual_adaptive",
    "gradient_adaptive",
    "hybrid_adaptive",
}
WEIGHTING_STRATEGIES = set(WEIGHTING_PARAMETER_CONTRACTS)
AGGREGATIONS = set(AGGREGATION_PARAMETER_CONTRACTS)
def register_runtime_network(
    name: str,
    builder: Callable[..., nn.Module],
    *,
    extra_parameters: list[str] | None = None,
    replace: bool = False,
) -> str:
    component_name = _runtime_registration_name(name)
    _register_runtime_value(NETWORK_BUILDERS, component_name, builder, replace=replace)
    NETWORK_EXTRA_PARAMETER_CONTRACTS[component_name] = sorted(set(extra_parameters or []))
    return component_name


def register_runtime_activation(name: str, activation: type[nn.Module], *, replace: bool = False) -> str:
    component_name = _runtime_registration_name(name)
    _register_runtime_value(ACTIVATIONS, component_name, activation, replace=replace)
    return component_name


def register_runtime_initializer(name: str, initializer: Callable[..., Any], *, replace: bool = False) -> str:
    component_name = _runtime_registration_name(name)
    _register_runtime_value(INITIALIZERS, component_name, initializer, replace=replace)
    return component_name


def register_runtime_loss(
    name: str,
    loss_function: Callable[[torch.Tensor, dict[str, Any]], torch.Tensor],
    *,
    parameters: set[str] | None = None,
    replace: bool = False,
) -> str:
    component_name = _runtime_registration_name(name)
    _register_runtime_value(LOSS_FUNCTIONS, component_name, loss_function, replace=replace)
    LOSS_PARAMETER_CONTRACTS[component_name] = set(parameters or set())
    return component_name


def register_runtime_sampler(
    name: str,
    sampler: Callable[["OpenSpecSampler", int], torch.Tensor],
    *,
    replace: bool = False,
) -> str:
    component_name = _runtime_registration_name(name)
    _register_runtime_value(CUSTOM_INTERIOR_SAMPLERS, component_name, sampler, replace=replace)
    SAMPLERS.add(component_name)
    return component_name


def _register_runtime_value(registry: dict[str, Any], name: str, value: Any, *, replace: bool) -> None:
    if name in registry and not replace:
        raise ValueError(f"Runtime component '{name}' is already registered")
    registry[name] = value


def _runtime_registration_name(name: str) -> str:
    normalized = _component_name(name)
    if not normalized or not normalized.replace("_", "").isalnum():
        raise ValueError(f"Invalid runtime component name: {name!r}")
    return normalized



def _numeric_parameter_search_space(option_registry: dict[str, Any]) -> dict[str, Any]:
    """Return only numeric schemas so prompts can expose compact search knobs."""

    numeric: dict[str, Any] = {}

    def contains_numeric(schema: Any) -> bool:
        if not isinstance(schema, dict):
            return False
        if schema.get("type") in {"number", "integer"}:
            return True
        if schema.get("type") == "array":
            return contains_numeric(schema.get("items"))
        return any(contains_numeric(value) for value in (schema.get("properties") or {}).values())

    fields = option_registry.get("fields") or {}
    for field_path, field in fields.items():
        field_numeric: dict[str, Any] = {}
        value_schema = field.get("value_schema")
        if contains_numeric(value_schema):
            field_numeric["value_schema"] = deepcopy(value_schema)
        for option_name, option in (field.get("options") or {}).items():
            parameter_schema = option.get("parameter_schema")
            if contains_numeric(parameter_schema):
                field_numeric.setdefault("options", {})[option_name] = deepcopy(parameter_schema)
        if field_numeric:
            numeric[field_path] = field_numeric
    return numeric


def builder_capabilities() -> dict[str, Any]:
    option_registry = algorithm_spec_option_registry()
    return {
        "algorithm_spec_option_registry": option_registry,
        "numeric_parameter_search_space": _numeric_parameter_search_space(option_registry),
        "networks": sorted(NETWORK_BUILDERS),
        "network_extra_parameters": option_parameter_names(option_registry, "network.architecture"),
        "network_residual_connection_parameters": ["scale"],
        "activations": sorted(ACTIVATIONS),
        "activation_parameters": option_parameter_names(option_registry, "network.activation.name"),
        "initializers": sorted(INITIALIZERS),
        "initializer_parameters": option_parameter_names(option_registry, "network.initialization.name"),
        "initializer_required_parameters": {
            name: _required_parameter_contract(target, skip={"tensor"})
            for name, target in INITIALIZERS.items()
        },
        "input_transforms": sorted(INPUT_TRANSFORMS),
        "input_transform_parameters": option_parameter_names(
            option_registry, "network.input_transform.name"
        ),
        "constraint_enforcement_methods": ["soft_penalty", "problem_hard"],
        "samplers": sorted(SAMPLERS),
        "interior_samplers": sorted(SAMPLERS),
        "constraint_samplers": sorted(CONSTRAINT_SAMPLERS),
        "sampler_parameters": option_parameter_names(option_registry, "sampling.interior.strategy"),
        "adaptive_samplers": sorted(ADAPTIVE_SAMPLERS),
        "adaptive_sampler_parameters": option_parameter_names(
            option_registry, "sampling.adaptive_refinement.strategy"
        ),
        "loss_functions": sorted(LOSS_FUNCTIONS),
        "loss_function_parameters": {name: sorted(values) for name, values in LOSS_PARAMETER_CONTRACTS.items()},
        "loss_term_parameters": {name: sorted(values) for name, values in LOSS_TERM_PARAMETER_CONTRACTS.items()},
        "weighting_strategies": sorted(WEIGHTING_STRATEGIES),
        "weighting_strategy_parameters": {name: sorted(values) for name, values in WEIGHTING_PARAMETER_CONTRACTS.items()},
        "loss_runtime_contract": {
            "governing_component_ids": "pde/<governing_law_id>",
            "aggregate_governing_terms_expand_independently": True,
            "fixed_weighting_group_reduction": "mean(weight_i * loss_i)",
            "fixed_weighting_groups": ["governing", "constraints"],
        },
        "aggregations": sorted(AGGREGATIONS),
        "aggregation_parameters": {name: sorted(values) for name, values in AGGREGATION_PARAMETER_CONTRACTS.items()},
        "optimizers": sorted(OPTIMIZERS),
        "optimizer_parameters": option_parameter_names(
            option_registry, "optimization.phases[].optimizer"
        ),
        "optimizer_constraints": {
            "multiadam": {
                "minimum_loss_terms": 2,
                "gradient_clipping_supported": False,
                "groupings": [
                    "pde_vs_constraints",
                    "dirichlet_vs_non_dirichlet",
                ],
                "description": (
                    "Explicit grouping maintains separate Adam moments per "
                    "loss group; omitted grouping preserves legacy per-loss behavior."
                ),
            }
        },
        "schedulers": sorted(SCHEDULERS),
        "scheduler_parameters": option_parameter_names(
            option_registry, "optimization.phases[].scheduler.name"
        ),
        "training": {
            "batch_modes": ["full_batch"],
            "full_batch_size": None,
            "gradient_clipping_parameters": ["max_norm", "norm_type"],
            "early_stopping_parameters": ["patience", "threshold"],
            "extra_parameters": [
                "adaptive_control", "component_control", "controller_coordinator",
                "physics_validation", "best_checkpoint",
            ],
            "adaptive_control": {
                "enabled": "boolean",
                "freeze_lbfgs_objective": {"const": True},
                "actions": [
                    "update_loss_weights",
                    "terminate_candidate",
                    "reallocate_to_adam",
                ],
                "reference_metrics_allowed": False,
                "lbfgs_stall_recovery": sorted(
                    {
                        "enabled", "patience", "minimum_relative_improvement",
                        "minimum_step_size", "minimum_parameter_update_norm",
                        "maximum_repeated_line_search_failures",
                        "minimum_completed_iterations_before_detection", "action",
                        "adam_learning_rate_scale", "maximum_recovery_iterations",
                        "minimum_recovery_iterations", "reuse_remaining_budget",
                        "return_to_lbfgs",
                    }
                ),
                "rollback": ["enabled", "loss_degradation_ratio", "patience"],
                "sampling_control": {
                    "strategies": ["fixed_budget_subset_replacement"],
                    "action": "replace_sampling_subset",
                    "fixed_total_point_budget": True,
                    "reference_metrics_allowed": False,
                },
                "curriculum_control": {
                    "strategies": ["residual_gated_level_advance"],
                    "action": "advance_curriculum_state",
                    "single_step_only": True,
                    "monotonic": True,
                    "axis_roles": [
                        "temporal", "parameter", "spatial_domain", "geometry",
                        "difficulty", "frequency", "scale", "fidelity", "custom",
                    ],
                    "probe_comparability": [
                        "global_fixed_probe", "active_scope_fixed_probe",
                    ],
                    "reference_metrics_allowed": False,
                },
            },
            "component_control": {"enabled": "boolean"},
            "controller_coordinator": {
                "maximum_non_safety_actions_per_cycle": {"const": 1},
                "priority_categories": [
                    "rollback", "terminate_candidate", "stage_recovery",
                    "loss_weight", "learning_rate", "sampling", "curriculum",
                ],
            },
            "physics_validation": {
                "reference_metrics_allowed": False,
                "normalization": ["initial_probe_value"],
                "category_aggregation": ["median_worst_blend"],
                "region_normalization": [
                    "per_region_initial", "shared_pde_rmse",
                ],
                "second_order_guard": "label_free_pre_stage_ratio",
            },
            "best_checkpoint": {
                "selection_metric": ["training_loss"],
                "final_model_policy": ["last", "best_train_loss"],
            },
        },
        "problem_owned_constraint_types": [
            "dirichlet", "initial", "neumann", "operator", "periodic", "pointset", "robin"
        ],
    }


class OpenSpecSampler:
    def __init__(
        self,
        problem: Any,
        spec: dict[str, Any],
        device: str,
        *,
        maximum_sampling_points: int | None = None,
        constraint_resample_interval: int = 50,
        legacy_wave_aggregate: bool = False,
    ) -> None:
        self.problem = problem
        self.spec = deepcopy(spec)
        self.device = str(device)
        self.adaptive_points: list[torch.Tensor] = []
        self._sobol_engine: torch.quasirandom.SobolEngine | None = None
        self._sequence_offsets = {"halton": 1, "hammersley": 1}
        self.maximum_sampling_points = (
            int(maximum_sampling_points) if maximum_sampling_points is not None else None
        )
        self.constraint_resample_interval = max(
            0, int(constraint_resample_interval)
        )
        self.legacy_wave_aggregate = bool(legacy_wave_aggregate)
        self._constraint_cache: dict[tuple[str, int], dict[str, Any]] = {}
        self._sample_call_count = 0
        self.peak_total_sampling_points = 0
        self.last_sampling_snapshot: dict[str, Any] = {}
        self.adaptive_refinement_events: list[dict[str, Any]] = []
        self.time_window_fraction = 1.0
        self.time_window_history: list[dict[str, Any]] = []
        self.component_registry: Any | None = None
        self.sampling_budget_plan: Any | None = None
        self._temporal_variable_index: int | None = None
        self._loss_runtime: Any | None = None
        self._controlled_sampling_points: dict[str, torch.Tensor] = {}
        self._latest_sampling_points: dict[str, torch.Tensor] = {}
        self._sampling_point_metadata: dict[str, list[dict[str, Any]]] = {}
        self._sampling_state_versions: dict[str, int] = {}
        self._sampling_control_generator = torch.Generator(device="cpu")
        self._sampling_control_generator.manual_seed(104729)
        self._next_sampling_point_id = 0

    def set_component_registry(self, registry: Any) -> None:
        self.component_registry = registry
        input_variables = list(self.problem.get_spec().input_variables)
        temporal = [
            item
            for item in registry.variable_roles
            if item.role == "temporal" and item.causal_ordered
        ]
        self._temporal_variable_index = (
            input_variables.index(temporal[0].variable_id)
            if temporal and temporal[0].variable_id in input_variables
            else None
        )

    def set_sampling_budget_plan(self, plan: Any) -> None:
        self.sampling_budget_plan = plan

    def set_loss_runtime(self, loss_runtime: Any) -> None:
        self._loss_runtime = loss_runtime

    def probe_clone(self) -> "OpenSpecSampler":
        clone = OpenSpecSampler(
            self.problem,
            self.spec,
            self.device,
            maximum_sampling_points=self.maximum_sampling_points,
            constraint_resample_interval=0,
            legacy_wave_aggregate=self.legacy_wave_aggregate,
        )
        if self.component_registry is not None:
            clone.set_component_registry(self.component_registry)
        if self.sampling_budget_plan is not None:
            clone.set_sampling_budget_plan(self.sampling_budget_plan)
        return clone

    def snapshot_registered_components(
        self, registry: Any, batch: PhysicsBatch
    ) -> dict[str, Any]:
        """Snapshot coordinates by registered sampler ID, not fixed field names."""

        result: dict[str, Any] = {}
        constraints = dict(getattr(batch, "constraint_samples", {}) or {})
        for descriptor in registry.sampling_components:
            if descriptor.domain_role == "interior":
                result[descriptor.sampler_id] = batch.domain_samples.detach().clone().cpu()
                continue
            category = descriptor.metadata.get("constraint_category")
            matching: dict[str, torch.Tensor] = {}
            for name, value in constraints.items():
                points = getattr(value, "points", value)
                if not isinstance(points, torch.Tensor):
                    continue
                value_category = getattr(value, "category", None)
                if category is None or value_category == category or str(name) in descriptor.associated_loss_component_ids:
                    matching[str(name)] = points.detach().clone().cpu()
            result[descriptor.sampler_id] = matching
        return result

    def set_time_window_fraction(self, fraction: float) -> None:
        value = float(fraction)
        if not 0.0 < value <= 1.0:
            raise ValueError("time window fraction must be in (0, 1]")
        if not self.time_window_history or self.time_window_history[-1]["fraction"] != value:
            self.time_window_history.append(
                {
                    "sample_call": int(self._sample_call_count),
                    "fraction": value,
                }
            )
        self.time_window_fraction = value

    def refresh_curriculum_samplers(
        self, associated_sampler_ids: tuple[str, ...]
    ) -> None:
        """Invalidate only sampler state declared as affected by a runtime."""

        affected = {str(item) for item in associated_sampler_ids}
        descriptors = {
            item.sampler_id: item
            for item in getattr(self.component_registry, "sampling_components", ())
        }
        unknown = affected - set(descriptors)
        if unknown:
            raise ValueError(
                f"Curriculum refresh references unknown samplers: {sorted(unknown)}"
            )
        clear_constraints = False
        for sampler_id in sorted(affected):
            descriptor = descriptors[sampler_id]
            if descriptor.domain_role == "interior":
                self._controlled_sampling_points.pop(sampler_id, None)
                self._latest_sampling_points.pop(sampler_id, None)
                self._sampling_point_metadata.pop(sampler_id, None)
                self._sampling_state_versions[sampler_id] = int(
                    self._sampling_state_versions.get(sampler_id, 0)
                ) + 1
            else:
                clear_constraints = True
        if clear_constraints:
            self._constraint_cache = {}
            self._constraint_cache_call = -1

    def state_dict(self) -> dict[str, Any]:
        """Return every mutable sampler field required by resume/rollback."""

        return {
            "adaptive_points": [point.detach().clone().cpu() for point in self.adaptive_points],
            "sequence_offsets": dict(self._sequence_offsets),
            "sobol_num_generated": (
                int(self._sobol_engine.num_generated)
                if self._sobol_engine is not None
                else None
            ),
            "constraint_cache": self._serializable_constraint_cache(),
            "sample_call_count": int(self._sample_call_count),
            "peak_total_sampling_points": int(self.peak_total_sampling_points),
            "last_sampling_snapshot": deepcopy(self.last_sampling_snapshot),
            "adaptive_refinement_events": deepcopy(self.adaptive_refinement_events),
            "time_window_fraction": float(self.time_window_fraction),
            "time_window_history": deepcopy(self.time_window_history),
            "controlled_sampling_points": {
                str(name): value.detach().clone().cpu()
                for name, value in self._controlled_sampling_points.items()
            },
            "latest_sampling_points": {
                str(name): value.detach().clone().cpu()
                for name, value in self._latest_sampling_points.items()
            },
            "sampling_point_metadata": deepcopy(self._sampling_point_metadata),
            "sampling_state_versions": dict(self._sampling_state_versions),
            "sampling_control_rng_state": self._sampling_control_generator.get_state(),
            "next_sampling_point_id": int(self._next_sampling_point_id),
        }

    def load_state_dict(self, state: dict[str, Any] | None) -> None:
        state = dict(state or {})
        self.adaptive_points = [
            point.detach().clone().to(self.device)
            for point in state.get("adaptive_points") or []
        ]
        self._sequence_offsets = {
            str(name): int(value)
            for name, value in (
                state.get("sequence_offsets") or {"halton": 1, "hammersley": 1}
            ).items()
        }
        self._sobol_engine = None
        self._restored_sobol_num_generated = state.get("sobol_num_generated")
        self._constraint_cache = self._restore_constraint_cache(
            state.get("constraint_cache") or {}
        )
        self._sample_call_count = int(state.get("sample_call_count") or 0)
        self.peak_total_sampling_points = int(
            state.get("peak_total_sampling_points") or 0
        )
        self.last_sampling_snapshot = deepcopy(
            state.get("last_sampling_snapshot") or {}
        )
        self.adaptive_refinement_events = deepcopy(
            state.get("adaptive_refinement_events") or []
        )
        self.time_window_fraction = float(state.get("time_window_fraction") or 1.0)
        self.time_window_history = deepcopy(state.get("time_window_history") or [])
        self._controlled_sampling_points = {
            str(name): value.detach().clone().to(self.device)
            for name, value in (state.get("controlled_sampling_points") or {}).items()
        }
        self._latest_sampling_points = {
            str(name): value.detach().clone().to(self.device)
            for name, value in (state.get("latest_sampling_points") or {}).items()
        }
        self._sampling_point_metadata = deepcopy(
            state.get("sampling_point_metadata") or {}
        )
        self._sampling_state_versions = {
            str(name): int(value)
            for name, value in (state.get("sampling_state_versions") or {}).items()
        }
        rng_state = state.get("sampling_control_rng_state")
        if isinstance(rng_state, torch.Tensor):
            self._sampling_control_generator.set_state(rng_state.cpu())
        self._next_sampling_point_id = int(state.get("next_sampling_point_id") or 0)
        for sampler_id, points in {
            **self._latest_sampling_points,
            **self._controlled_sampling_points,
        }.items():
            if sampler_id not in self._sampling_point_metadata:
                self._sampling_point_metadata[sampler_id] = self._new_point_metadata(
                    int(points.shape[0]), origin="initial"
                )
            self._sampling_state_versions.setdefault(sampler_id, 0)

    def _serializable_constraint_cache(self) -> dict[tuple[str, int], dict[str, Any]]:
        result: dict[tuple[str, int], dict[str, Any]] = {}
        for key, values in self._constraint_cache.items():
            encoded: dict[str, Any] = {}
            for name, value in values.items():
                if isinstance(value, torch.Tensor):
                    encoded[str(name)] = {
                        "kind": "tensor",
                        "points": value.detach().clone().cpu(),
                    }
                    continue
                points = getattr(value, "points", None)
                encoded[str(name)] = {
                    "kind": "constraint",
                    "points": points.detach().clone().cpu()
                    if isinstance(points, torch.Tensor)
                    else None,
                    "pointset_values": (
                        getattr(value, "pointset_values").detach().clone().cpu()
                        if isinstance(getattr(value, "pointset_values", None), torch.Tensor)
                        else None
                    ),
                    "pointset_component": getattr(value, "pointset_component", None),
                    "location": getattr(value, "location", None),
                    "derivative_variable": getattr(value, "derivative_variable", None),
                    "derivative_order": int(getattr(value, "derivative_order", 0) or 0),
                    "category": getattr(value, "category", key[0]),
                }
            result[(str(key[0]), int(key[1]))] = encoded
        return result

    def _restore_constraint_cache(
        self, state: dict[tuple[str, int], dict[str, Any]]
    ) -> dict[tuple[str, int], dict[str, Any]]:
        restored: dict[tuple[str, int], dict[str, Any]] = {}
        for raw_key, encoded in state.items():
            key = (str(raw_key[0]), int(raw_key[1]))
            category, count = key
            if category == "legacy_wave_aggregate":
                templates = self._sample_legacy_wave_constraints(
                    count, refresh=True
                )
            else:
                templates = self._sample_constraint_category(
                    category, count, refresh=True
                )
            values: dict[str, Any] = {}
            for name, item in encoded.items():
                if item.get("kind") == "tensor":
                    values[str(name)] = item["points"].to(self.device)
                    continue
                template = templates.get(name)
                if template is None:
                    continue
                points = item.get("points")
                pointset_values = item.get("pointset_values")
                replacements = {
                    "points": points.to(self.device)
                    if isinstance(points, torch.Tensor)
                    else getattr(template, "points", None),
                    "numpy_points": (
                        points.detach().cpu().numpy()
                        if isinstance(points, torch.Tensor)
                        else getattr(template, "numpy_points", None)
                    ),
                    "pointset_values": (
                        pointset_values.to(self.device)
                        if isinstance(pointset_values, torch.Tensor)
                        else None
                    ),
                    "pointset_component": item.get("pointset_component"),
                    "location": item.get("location"),
                    "derivative_variable": item.get("derivative_variable"),
                    "derivative_order": int(item.get("derivative_order") or 0),
                }
                try:
                    values[str(name)] = replace(template, **replacements)
                except TypeError:
                    values[str(name)] = template
            restored[key] = values
        return restored

    def sample(self, overrides: dict[str, int] | None = None) -> PhysicsBatch:
        overrides = dict(overrides or {})
        interior_n = int(overrides.get("interior", self.spec["interior"]["parameters"].get("n_points", 0)))
        interior_descriptor = next(
            (
                item
                for item in getattr(self.component_registry, "sampling_components", ())
                if item.domain_role == "interior"
            ),
            None,
        )
        interior_sampler_id = (
            str(interior_descriptor.sampler_id)
            if interior_descriptor is not None
            else "sample:interior"
        )
        controlled = self._controlled_sampling_points.get(interior_sampler_id)
        if controlled is not None and int(controlled.shape[0]) == max(1, interior_n):
            domain = controlled.detach().clone().to(self.device)
        else:
            domain = self._apply_time_window_to_tensor(
                self._domain(max(1, interior_n))
            )
        self._latest_sampling_points[interior_sampler_id] = domain.detach().clone()
        if interior_sampler_id not in self._sampling_point_metadata or len(
            self._sampling_point_metadata[interior_sampler_id]
        ) != int(domain.shape[0]):
            self._sampling_point_metadata[interior_sampler_id] = self._new_point_metadata(
                int(domain.shape[0]), origin="initial"
            )
        self._sampling_state_versions.setdefault(interior_sampler_id, 0)
        interval = self.constraint_resample_interval
        refresh_constraints = (
            not self._constraint_cache
            or (interval > 0 and self._sample_call_count % interval == 0)
        )
        if self.legacy_wave_aggregate:
            boundary_n = int(overrides.get("boundary", self.spec["boundary"]["parameters"].get("n_points", 0)))
            boundary = self._sample_legacy_wave_constraints(
                max(1, boundary_n),
                refresh=refresh_constraints,
            )
            initial = {}
        elif self.component_registry is not None:
            boundary = {}
            initial = {}
            sampled_categories: set[str] = set()
            for descriptor in self.component_registry.sampling_components:
                if descriptor.domain_role == "interior":
                    continue
                category = str(
                    descriptor.metadata.get("constraint_category")
                    or descriptor.domain_role
                )
                if category in sampled_categories:
                    continue
                sampled_categories.add(category)
                runtime_key = str(
                    descriptor.metadata.get("runtime_sample_key")
                    or descriptor.sampler_id
                )
                count = int(
                    overrides.get(
                        descriptor.sampler_id,
                        overrides.get(
                            runtime_key,
                            descriptor.point_budget or 0,
                        ),
                    )
                )
                sampled = self._sample_constraint_category(
                    category,
                    max(1, count),
                    refresh=refresh_constraints,
                )
                if descriptor.domain_role == "initial_surface":
                    initial.update(sampled)
                else:
                    boundary.update(sampled)
        else:
            boundary_n = int(overrides.get("boundary", self.spec["boundary"]["parameters"].get("n_points", 0)))
            initial_n = int(overrides.get("initial", self.spec["initial"]["parameters"].get("n_points", 0)))
            boundary = self._sample_constraint_category(
                "boundary",
                max(1, boundary_n),
                refresh=refresh_constraints,
            )
            initial = self._sample_constraint_category(
                "initial",
                max(1, initial_n),
                refresh=refresh_constraints,
            )
        constraints = dict(boundary)
        constraints.update(initial)
        constraints = self._apply_time_window_to_constraints(constraints)
        self._sample_call_count += 1
        batch = PhysicsBatch(domain, constraints, self.problem.get_observations())
        self._record_sampling_snapshot(batch)
        return batch

    def _new_point_metadata(self, count: int, *, origin: str) -> list[dict[str, Any]]:
        result = []
        for _ in range(int(count)):
            result.append(
                {
                    "point_id": f"sampling_point_{self._next_sampling_point_id}",
                    "age": 0,
                    "consecutive_retention_count": 0,
                    "historical_score_ema": 0.0,
                    "last_selected_iteration": None,
                    "origin": str(origin),
                }
            )
            self._next_sampling_point_id += 1
        return result

    def sampling_runtime_capabilities(self, sampler_id: str) -> dict[str, bool]:
        descriptor = next(
            (
                item
                for item in getattr(self.component_registry, "sampling_components", ())
                if item.sampler_id == str(sampler_id)
            ),
            None,
        )
        interior = descriptor is not None and descriptor.domain_role == "interior"
        complete = bool(
            interior
            and callable(getattr(self.problem, "sample_domain", None))
            and callable(getattr(self._loss_runtime, "compute_pointwise_component_scores", None))
        )
        return {
            "pointwise_scoring_supported": complete,
            "candidate_generation_supported": complete,
            "state_snapshot_supported": True,
            "replacement_supported": complete,
        }

    def get_training_points(self, sampler_id: str) -> Any:
        from forge.pipeline.g_training_evaluation.training.sampling_runtime import SamplingBatch

        key = str(sampler_id)
        points = self._controlled_sampling_points.get(key)
        if points is None:
            points = self._latest_sampling_points.get(key)
        if points is None:
            raise KeyError(f"No training points are available for sampler {key!r}")
        metadata = self._sampling_point_metadata.get(key)
        if metadata is None or len(metadata) != int(points.shape[0]):
            metadata = self._new_point_metadata(int(points.shape[0]), origin="initial")
            self._sampling_point_metadata[key] = metadata
        return SamplingBatch(
            sampler_id=key,
            points=points.detach().clone(),
            point_metadata=tuple(deepcopy(metadata)),
            state_version=int(self._sampling_state_versions.get(key, 0)),
        )

    def score_points(
        self,
        model: nn.Module,
        component_ids: tuple[str, ...],
        points: Any,
        context: Any,
    ) -> Any:
        from forge.pipeline.g_training_evaluation.training.sampling_runtime import PointScoreResult

        if self._loss_runtime is None:
            raise RuntimeError("Sampling point scoring requires a registered loss runtime")
        raw = self._loss_runtime.compute_pointwise_component_scores(
            model, tuple(component_ids), points.points
        )
        normalized = []
        for component_id in sorted(raw):
            values = raw[component_id].detach().reshape(-1).float()
            scale = torch.quantile(values, 0.5).clamp_min(1e-12)
            normalized.append(values / scale)
        if not normalized:
            raise RuntimeError("No associated component produced pointwise scores")
        scores = torch.stack(normalized, dim=0).amax(dim=0)
        median_value = float(torch.quantile(scores, 0.5).cpu())
        mean_value = float(scores.mean().cpu())
        std_value = float(scores.std(unbiased=False).cpu())
        quantiles = {
            name: float(torch.quantile(scores, q).cpu())
            for name, q in (("p75", 0.75), ("p90", 0.90), ("p95", 0.95), ("p99", 0.99))
        }
        statistics = {
            "median": median_value,
            **quantiles,
            "maximum": float(scores.max().cpu()),
            "mean": mean_value,
            "coefficient_of_variation": std_value / max(abs(mean_value), 1e-12),
            "p95_to_median_ratio": quantiles["p95"] / max(abs(median_value), 1e-12),
        }
        return PointScoreResult(
            sampler_id=str(points.sampler_id),
            scores=scores.cpu(),
            component_scores={name: value.detach().cpu() for name, value in raw.items()},
            statistics=statistics,
            scoring_component_ids=tuple(sorted(raw)),
        )

    def generate_candidate_points(
        self, sampler_id: str, count: int, rng_state: Any | None = None
    ) -> Any:
        from forge.pipeline.g_training_evaluation.training.sampling_runtime import SamplingBatch

        if int(count) <= 0:
            raise ValueError("Candidate count must be positive")
        if rng_state is not None:
            self._sampling_control_generator.set_state(rng_state)
        seed = int(
            torch.randint(
                0, 2**31 - 1, (1,), generator=self._sampling_control_generator
            ).item()
        )
        global_state = torch.get_rng_state()
        try:
            torch.manual_seed(seed)
            generated = self._apply_time_window_to_tensor(self._domain(int(count)))
        finally:
            torch.set_rng_state(global_state)
        return SamplingBatch(
            sampler_id=str(sampler_id), points=generated.detach().clone(),
            point_metadata=tuple(self._new_point_metadata(int(count), origin="exploration")),
            state_version=int(self._sampling_state_versions.get(str(sampler_id), 0)),
        )

    def replace_subset(
        self,
        sampler_id: str,
        retained_indices: Any,
        replacement_points: Any,
        **context: Any,
    ) -> Any:
        from forge.pipeline.g_training_evaluation.training.sampling_runtime import SamplingMutationResult

        current = self.get_training_points(sampler_id)
        retained = [int(index) for index in retained_indices]
        replacement = replacement_points.points.detach().clone().to(self.device)
        if len(set(retained)) != len(retained) or any(index < 0 or index >= current.point_count for index in retained):
            raise ValueError("Retained indices must be unique and in range")
        if len(retained) + int(replacement.shape[0]) != current.point_count:
            raise ValueError("Fixed-budget replacement must preserve point count")
        before_version = int(self._sampling_state_versions.get(str(sampler_id), 0))
        retained_tensor = current.points[retained].to(self.device)
        combined = torch.cat((retained_tensor, replacement), dim=0)
        old_metadata = list(current.point_metadata)
        hard = {int(index) for index in context.get("hard_retained_indices", ())}
        iteration = int(context.get("iteration", 0))
        scores = list(context.get("current_scores", ()))
        retained_metadata = []
        for index in retained:
            item = dict(old_metadata[index])
            item["age"] = int(item.get("age", 0)) + 1
            item["consecutive_retention_count"] = (
                int(item.get("consecutive_retention_count", 0)) + 1
                if index in hard else 0
            )
            if index in hard:
                item["origin"] = "hard_retained"
            if index < len(scores):
                old_score = float(item.get("historical_score_ema", 0.0))
                item["historical_score_ema"] = 0.9 * old_score + 0.1 * float(scores[index])
            item["last_selected_iteration"] = iteration
            retained_metadata.append(item)
        new_metadata = self._new_point_metadata(int(replacement.shape[0]), origin="replacement")
        exploration_count = int(context.get("exploration_count", 0))
        for item in new_metadata[-exploration_count:] if exploration_count else ():
            item["origin"] = "exploration"
        self._controlled_sampling_points[str(sampler_id)] = combined
        self._latest_sampling_points[str(sampler_id)] = combined.detach().clone()
        self._sampling_point_metadata[str(sampler_id)] = retained_metadata + new_metadata
        self._sampling_state_versions[str(sampler_id)] = before_version + 1
        return SamplingMutationResult(
            sampler_id=str(sampler_id),
            point_count_before=current.point_count,
            point_count_after=int(combined.shape[0]),
            points_replaced=int(replacement.shape[0]),
            exploration_points=exploration_count,
            state_version_before=before_version,
            state_version_after=before_version + 1,
        )

    def snapshot_state(self) -> dict[str, Any]:
        return self.state_dict()

    def restore_state(self, state: dict[str, Any]) -> None:
        self.load_state_dict(state)

    def _sample_constraint_category(
        self,
        category: str,
        count: int,
        *,
        refresh: bool,
    ) -> dict[str, Any]:
        key = (str(category), int(count))
        if not refresh and key in self._constraint_cache:
            return self._constraint_cache[key]

        sample_constraints = self.problem.sample_constraints
        try:
            parameters = inspect.signature(sample_constraints).parameters.values()
            supports_category = any(
                parameter.name == "category"
                or parameter.kind == parameter.VAR_KEYWORD
                for parameter in parameters
            )
        except (TypeError, ValueError):
            supports_category = False
        if supports_category:
            sampled = sample_constraints(
                int(count),
                category=category,
                device=self.device,
            )
        else:
            sampled = sample_constraints(int(count), device=self.device)

        values = dict(sampled or {})
        if str(category) in values:
            values = {str(category): values[str(category)]}
        elif values and any(hasattr(value, "category") for value in values.values()):
            values = {
                name: value
                for name, value in values.items()
                if getattr(value, "category", "boundary") == category
            }
        elif category == "initial":
            values = (
                {"initial": values["initial"]}
                if "initial" in values
                else {}
            )
        self._constraint_cache[key] = values
        return values

    def _sample_legacy_wave_constraints(
        self,
        count: int,
        *,
        refresh: bool,
    ) -> dict[str, Any]:
        key = ("legacy_wave_aggregate", int(count))
        if not refresh and key in self._constraint_cache:
            return self._constraint_cache[key]
        sampled = self.problem.sample_constraints(
            int(count),
            category=None,
            device=self.device,
        )
        values = dict(sampled or {})
        self._constraint_cache[key] = values
        return values

    def _record_sampling_snapshot(self, batch: PhysicsBatch) -> None:
        interior_points = int(batch.domain_samples.shape[0])
        boundary_batches: list[torch.Tensor] = []
        initial_batches: list[torch.Tensor] = []
        constraint_breakdown: list[dict[str, Any]] = []
        for name, value in batch.constraint_samples.items():
            points = getattr(value, "points", value)
            count = int(points.shape[0]) if hasattr(points, "shape") and len(points.shape) else 0
            if getattr(value, "category", "boundary") == "initial":
                if isinstance(points, torch.Tensor):
                    initial_batches.append(points)
            elif isinstance(points, torch.Tensor):
                boundary_batches.append(points)
            constraint_breakdown.append(
                {
                    "constraint_id": str(name),
                    "constraint_type": str(
                        getattr(value, "category", "boundary")
                    ),
                    "location": getattr(value, "location", None),
                    "sample_count": count,
                    "derivative_variable": getattr(
                        value, "derivative_variable", None
                    ),
                    "derivative_order": int(
                        getattr(value, "derivative_order", 0) or 0
                    ),
                }
            )
        def unique_coordinate_count(values: list[torch.Tensor]) -> int:
            unique: list[torch.Tensor] = []
            for points in values:
                if any(
                    points.shape == existing.shape
                    and torch.equal(points, existing)
                    for existing in unique
                ):
                    continue
                unique.append(points)
            return sum(int(points.shape[0]) for points in unique)

        # Count coordinate sets, not the number of loss functions consuming
        # them. This handles Wave's shared initial coordinates without hiding
        # independent initial/boundary samplers used by other PDEs.
        boundary_points = unique_coordinate_count(boundary_batches)
        initial_points = unique_coordinate_count(initial_batches)
        constraint_residual_points = sum(
            int(points.shape[0])
            for points in boundary_batches + initial_batches
        )
        total = interior_points + boundary_points + initial_points
        planned_total = (
            int(self.sampling_budget_plan.total_training_points)
            if self.sampling_budget_plan is not None
            else None
        )
        self.peak_total_sampling_points = max(self.peak_total_sampling_points, total)
        region_snapshot = None
        region_ids_provider = getattr(
            self.problem, "training_residual_region_ids", None
        )
        region_metadata_provider = getattr(
            self.problem, "training_residual_region_metadata", None
        )
        if callable(region_ids_provider):
            region_ids = region_ids_provider(batch.domain_samples)
            if (
                isinstance(region_ids, torch.Tensor)
                and region_ids.ndim == 1
                and int(region_ids.shape[0]) == interior_points
            ):
                metadata = (
                    dict(region_metadata_provider() or {})
                    if callable(region_metadata_provider)
                    else {}
                )
                configured_count = int(metadata.get("region_count") or 0)
                observed_count = (
                    int(region_ids.max().detach().cpu()) + 1
                    if region_ids.numel()
                    else 0
                )
                region_count = max(configured_count, observed_count)
                counts = torch.bincount(
                    region_ids.detach().to(dtype=torch.long).cpu(),
                    minlength=region_count,
                ).tolist()
                region_snapshot = {
                    **metadata,
                    "sample_counts": [int(value) for value in counts],
                    "minimum_sample_count": (
                        int(min(counts)) if counts else 0
                    ),
                    "maximum_sample_count": (
                        int(max(counts)) if counts else 0
                    ),
                    "total_sample_count": int(sum(counts)),
                }
        self.last_sampling_snapshot = {
            "current_interior_points": interior_points,
            "boundary_points": boundary_points,
            "initial_condition_points": initial_points,
            "current_total_sampling_points": total,
            "effective_total_sampling_points": (
                interior_points + constraint_residual_points
            ),
            "constraint_breakdown": constraint_breakdown,
            "time_window_fraction": float(self.time_window_fraction),
            "peak_total_sampling_points": self.peak_total_sampling_points,
            "maximum_sampling_points": self.maximum_sampling_points,
            "planned_total_sampling_points": planned_total,
            "sampling_plan_delta": (
                None if planned_total is None else total - planned_total
            ),
            "sampling_budget_plan": (
                self.sampling_budget_plan.to_dict()
                if self.sampling_budget_plan is not None
                else None
            ),
            "training_residual_regions": region_snapshot,
        }
        if self.maximum_sampling_points is not None and total > self.maximum_sampling_points:
            raise RuntimeError(
                "Actual sampling budget exceeded: "
                f"{total} > {self.maximum_sampling_points}"
            )

    def _apply_time_window_to_tensor(self, points: torch.Tensor) -> torch.Tensor:
        if self.time_window_fraction >= 1.0:
            return points
        problem_spec = self.problem.get_spec()
        time_index = self._temporal_variable_index
        if time_index is None:
            return points
        variable_id = list(problem_spec.input_variables)[time_index]
        lower, upper = problem_spec.domain.bounds[variable_id]
        result = points.clone()
        result[:, time_index] = float(lower) + self.time_window_fraction * (
            result[:, time_index] - float(lower)
        )
        result[:, time_index].clamp_(
            min=float(lower),
            max=float(lower)
            + self.time_window_fraction * (float(upper) - float(lower)),
        )
        return result

    def _apply_time_window_to_constraints(
        self, constraints: dict[str, Any]
    ) -> dict[str, Any]:
        if self.time_window_fraction >= 1.0:
            return constraints
        adjusted: dict[str, Any] = {}
        for name, value in constraints.items():
            if getattr(value, "category", "boundary") == "initial":
                adjusted[name] = value
                continue
            points = getattr(value, "points", None)
            if not isinstance(points, torch.Tensor):
                adjusted[name] = value
                continue
            transformed = self._apply_time_window_to_tensor(points)
            replacement_fields: dict[str, Any] = {"points": transformed}
            if hasattr(value, "paired_points"):
                paired_points = getattr(value, "paired_points", None)
                replacement_fields["paired_points"] = (
                    self._apply_time_window_to_tensor(paired_points)
                    if isinstance(paired_points, torch.Tensor)
                    else paired_points
                )
            if hasattr(value, "numpy_points"):
                replacement_fields["numpy_points"] = (
                    transformed.detach().cpu().numpy()
                )
            try:
                adjusted[name] = replace(value, **replacement_fields)
            except (TypeError, ValueError):
                # Legacy tensor-only constraint providers have no dataclass
                # fields to preserve.
                adjusted[name] = transformed
        return adjusted

    def _domain(self, n: int) -> torch.Tensor:
        strategy = self.spec["interior"]["strategy"]
        if strategy == "uniform":
            base = self.problem.sample_domain(n, device=self.device)
        elif strategy == "latin_hypercube":
            base = self._latin_hypercube(n)
        elif strategy == "sobol":
            base = self._sobol(n)
        elif strategy == "halton":
            base = self._low_discrepancy(n, hammersley=False)
        elif strategy == "hammersley":
            base = self._low_discrepancy(n, hammersley=True)
        elif strategy == "grid":
            base = self._grid(n)
        elif strategy in CUSTOM_INTERIOR_SAMPLERS:
            base = CUSTOM_INTERIOR_SAMPLERS[strategy](self, n)
            if not isinstance(base, torch.Tensor) or base.ndim != 2 or base.shape[0] != n:
                _raise(
                    "invalid_runtime_component",
                    "sampling.interior.strategy",
                    strategy,
                    f"Runtime sampler '{strategy}' must return a tensor shaped (n, input_dim)",
                )
            base = base.to(self.device)
        else:
            _raise("unsupported_component", "sampling.interior.strategy", strategy, f"Unknown sampler '{strategy}'")
        balance = getattr(
            self.problem, "balance_training_domain_samples", None
        )
        if callable(balance):
            base = balance(base)
        if not self.adaptive_points:
            return base
        adaptive = torch.cat(self.adaptive_points, dim=0).to(self.device)
        region_ids = getattr(
            self.problem, "training_residual_region_ids", None
        )
        if callable(region_ids):
            base_regions = region_ids(base)
            adaptive_regions = region_ids(adaptive)
            if isinstance(base_regions, torch.Tensor) and isinstance(
                adaptive_regions, torch.Tensor
            ):
                result = base.clone()
                for region in torch.unique(adaptive_regions).tolist():
                    base_indices = torch.nonzero(
                        base_regions == int(region), as_tuple=False
                    ).reshape(-1)
                    region_points = adaptive[
                        adaptive_regions == int(region)
                    ]
                    count = min(
                        int(base_indices.numel()),
                        int(region_points.shape[0]),
                    )
                    if count > 0:
                        result[base_indices[-count:]] = region_points[-count:]
                return result
        count = min(n, adaptive.shape[0])
        base[-count:] = adaptive[-count:]
        return base

    def _latin_hypercube(self, n: int) -> torch.Tensor:
        lower_bounds, upper_bounds = self._bounds()
        parameters = dict(self.spec["interior"].get("parameters") or {})
        centered = bool(parameters.get("centered", False))
        columns = []
        for lower, upper in zip(lower_bounds, upper_bounds):
            offsets = torch.full((n,), 0.5, device=self.device) if centered else torch.rand(n, device=self.device)
            unit = (torch.arange(n, device=self.device) + offsets) / n
            unit = unit[torch.randperm(n, device=self.device)]
            columns.append((lower + unit * (upper - lower)).reshape(-1, 1))
        return torch.cat(columns, dim=1)

    def _sobol(self, n: int) -> torch.Tensor:
        lower, upper = self._bounds()
        dimension = len(lower)
        parameters = dict(self.spec["interior"].get("parameters") or {})
        scramble = bool(parameters.get("scramble", True))
        configured_seed = parameters.get("seed")
        seed = int(configured_seed) if configured_seed is not None else int(torch.initial_seed() % (2**31 - 1))
        if self._sobol_engine is None:
            self._sobol_engine = torch.quasirandom.SobolEngine(dimension, scramble=scramble, seed=seed)
            restored = getattr(self, "_restored_sobol_num_generated", None)
            skip = int(restored) if restored is not None else int(parameters.get("skip", 0))
            if skip > 0:
                self._sobol_engine.fast_forward(skip)
            self._restored_sobol_num_generated = None
        unit = self._sobol_engine.draw(n).to(self.device)
        lower_tensor = torch.tensor(lower, dtype=unit.dtype, device=self.device)
        upper_tensor = torch.tensor(upper, dtype=unit.dtype, device=self.device)
        return lower_tensor + unit * (upper_tensor - lower_tensor)

    def _grid(self, n: int) -> torch.Tensor:
        lower, upper = self._bounds()
        dimension = len(lower)
        parameters = dict(self.spec["interior"].get("parameters") or {})
        configured_points = parameters.get("points_per_axis")
        points_per_axis = (
            max(2, int(configured_points))
            if configured_points is not None
            else max(2, int(math.ceil(n ** (1.0 / max(1, dimension)))))
        )
        axes = [
            torch.linspace(lo, hi, points_per_axis, device=self.device)
            for lo, hi in zip(lower, upper)
        ]
        grid = torch.cartesian_prod(*axes)
        if dimension == 1:
            grid = grid.reshape(-1, 1)
        if bool(parameters.get("shuffle", False)):
            grid = grid[torch.randperm(grid.shape[0], device=grid.device)]
        if grid.shape[0] >= n:
            return grid[:n]
        repeats = int(math.ceil(n / grid.shape[0]))
        return grid.repeat((repeats, 1))[:n]

    def _low_discrepancy(self, n: int, *, hammersley: bool) -> torch.Tensor:
        lower, upper = self._bounds()
        dimension = len(lower)
        name = "hammersley" if hammersley else "halton"
        parameters = dict(self.spec["interior"].get("parameters") or {})
        configured_start = parameters.get("start_index")
        if configured_start is not None and self._sequence_offsets[name] == 1:
            self._sequence_offsets[name] = max(1, int(configured_start))
        offset = self._sequence_offsets[name]
        primes = _first_primes(dimension if not hammersley else max(0, dimension - 1))
        rows: list[list[float]] = []
        for local_index in range(n):
            index = offset + local_index
            if hammersley:
                row = [((local_index + 0.5) / n)]
                row.extend(_radical_inverse(index, base) for base in primes)
            else:
                row = [_radical_inverse(index, base) for base in primes]
            rows.append(row)
        self._sequence_offsets[name] += n
        unit = torch.tensor(rows, dtype=torch.float32, device=self.device)
        lower_tensor = torch.tensor(lower, dtype=unit.dtype, device=self.device)
        upper_tensor = torch.tensor(upper, dtype=unit.dtype, device=self.device)
        return lower_tensor + unit * (upper_tensor - lower_tensor)

    def _bounds(self) -> tuple[list[float], list[float]]:
        problem_spec = self.problem.get_spec()
        bounds = dict(problem_spec.domain.bounds or {})
        lower_bounds: list[float] = []
        upper_bounds: list[float] = []
        pilot = None
        for index, name in enumerate(problem_spec.input_variables):
            interval = bounds.get(name)
            if not isinstance(interval, (list, tuple)) or len(interval) != 2:
                pilot = pilot if pilot is not None else self.problem.sample_domain(1024, device=self.device)
                lower, upper = float(pilot[:, index].min()), float(pilot[:, index].max())
            else:
                lower, upper = map(float, interval)
            lower_bounds.append(lower)
            upper_bounds.append(upper)
        return lower_bounds, upper_bounds

    def adaptive_update(self, model: nn.Module) -> None:
        adaptive = self.spec["adaptive_refinement"]
        if not adaptive.get("enabled"):
            return
        parameters = adaptive.get("parameters") or {}
        initial_interior_points = int(
            (self.spec.get("interior", {}).get("parameters") or {}).get("n_points") or 0
        )
        adaptive_points_before = sum(int(points.shape[0]) for points in self.adaptive_points)
        candidate_count = int(parameters.get("candidate_points", 1024))
        add_points = int(parameters.get("add_points", 64))
        candidates = self.problem.sample_domain(max(candidate_count, add_points), device=self.device)
        candidates = candidates.detach().clone().requires_grad_(True)
        strategy = adaptive.get("strategy") or "residual_adaptive"
        needs_gradient = strategy in {"gradient_adaptive", "hybrid_adaptive"}
        residuals = self.problem.compute_governing_residuals(model, candidates, create_graph=needs_gradient)
        transform = getattr(
            self.problem, "training_governing_residuals", None
        )
        if callable(transform):
            residuals = dict(transform(candidates, residuals))
        residual_component = parameters.get("residual_component")
        if residual_component is None:
            if not residuals:
                raise RuntimeError(
                    "Adaptive refinement requires at least one governing residual"
                )
            # Multi-equation PDEs must not silently refine against whichever
            # residual happens to be inserted first.  Score every equation on
            # a comparable scale so a localized failure in a later component
            # (for example, Navier--Stokes momentum_y) can drive refinement.
            normalized_component_scores: list[torch.Tensor] = []
            for component_residual in residuals.values():
                flattened = component_residual.reshape(candidates.shape[0], -1)
                component_rms = torch.sqrt(
                    flattened.square().mean(dim=1).clamp_min(1.0e-24)
                )
                component_scale = component_rms.detach().mean().clamp_min(1.0e-12)
                normalized_component_scores.append(component_rms / component_scale)
            residual = torch.sqrt(
                torch.stack(normalized_component_scores, dim=1)
                .square()
                .mean(dim=1)
                .clamp_min(1.0e-24)
            ).unsqueeze(1)
            residual_target = "normalized_rms_all_components"
        else:
            component_name = str(residual_component)
            if component_name not in residuals:
                _raise(
                    "unsupported_component",
                    "sampling.adaptive_refinement.parameters.residual_component",
                    component_name,
                    f"Unknown governing residual component '{component_name}'",
                )
            residual = residuals[component_name]
            residual_target = component_name
        residual_score = residual.detach().abs().reshape(candidates.shape[0], -1).mean(dim=1)
        if needs_gradient:
            gradient = torch.autograd.grad(
                residual,
                candidates,
                torch.ones_like(residual),
                create_graph=False,
                retain_graph=False,
            )[0]
            gradient_indices = parameters.get("gradient_input_indices")
            if gradient_indices is not None:
                indices = [int(index) for index in gradient_indices]
                invalid_indices = [index for index in indices if index < 0 or index >= gradient.shape[1]]
                if invalid_indices:
                    _raise(
                        "invalid_parameter",
                        "sampling.adaptive_refinement.parameters.gradient_input_indices",
                        invalid_indices,
                        f"Gradient input indices must be in [0, {gradient.shape[1] - 1}]",
                    )
                gradient = gradient[:, indices]
            gradient_score = gradient.detach().norm(dim=1)
            if strategy == "gradient_adaptive":
                scores = gradient_score
            else:
                residual_weight = float(parameters.get("residual_weight", 1.0))
                gradient_weight = float(parameters.get("gradient_weight", 1.0))
                hybrid_mode = _component_name(parameters.get("hybrid_mode", "product"))
                if hybrid_mode == "weighted_sum":
                    residual_normalized = residual_score / residual_score.mean().clamp_min(1e-12)
                    gradient_normalized = gradient_score / gradient_score.mean().clamp_min(1e-12)
                    scores = residual_weight * residual_normalized + gradient_weight * gradient_normalized
                elif hybrid_mode == "product":
                    scores = residual_score.pow(residual_weight) * (1.0 + gradient_score).pow(gradient_weight)
                else:
                    _raise(
                        "invalid_parameter",
                        "sampling.adaptive_refinement.parameters.hybrid_mode",
                        hybrid_mode,
                        "hybrid_mode must be 'product' or 'weighted_sum'",
                    )
        else:
            scores = residual_score
        selection_count = min(add_points, candidates.shape[0])
        if strategy in {"rad", "rar_d"}:
            exponent = float(parameters.get("residual_exponent", 2.0))
            offset = float(parameters.get("distribution_offset", 1.0))
            weights = scores.clamp_min(0.0).pow(exponent)
            weights = weights / weights.mean().clamp_min(1e-12) + offset
            probabilities = weights / weights.sum().clamp_min(1e-12)
            selected = torch.multinomial(probabilities, num_samples=selection_count, replacement=False)
        else:
            selected = torch.topk(scores, k=selection_count).indices
        selected_points = candidates.detach()[selected].clone()
        replace_existing = bool(parameters.get("replace_existing", strategy == "rad"))
        if replace_existing:
            self.adaptive_points = [selected_points]
        else:
            self.adaptive_points.append(selected_points)
        max_points = int(parameters.get("max_points", 0))
        if max_points > 0:
            combined = torch.cat(self.adaptive_points, dim=0)
            if combined.shape[0] > max_points:
                self.adaptive_points = [combined[-max_points:].detach().clone()]
        adaptive_points_after = sum(int(points.shape[0]) for points in self.adaptive_points)
        current_interior_points = initial_interior_points
        boundary_points = int(
            (self.spec.get("boundary", {}).get("parameters") or {}).get("n_points") or 0
        )
        initial_condition_points = int(
            (self.spec.get("initial", {}).get("parameters") or {}).get("n_points") or 0
        )
        current_total = current_interior_points + boundary_points + initial_condition_points
        self.peak_total_sampling_points = max(self.peak_total_sampling_points, current_total)
        event = {
            "initial_interior_points": initial_interior_points,
            "adaptive_points_before": adaptive_points_before,
            "adaptive_points_added": max(0, adaptive_points_after - adaptive_points_before),
            "adaptive_points_stored": adaptive_points_after,
            "current_interior_points": current_interior_points,
            "boundary_points": boundary_points,
            "initial_condition_points": initial_condition_points,
            "current_total_sampling_points": current_total,
            "peak_total_sampling_points": self.peak_total_sampling_points,
            "candidate_pool_points": int(candidates.shape[0]),
            "residual_target": residual_target,
            "available_residual_components": [str(name) for name in residuals],
            "max_points": max_points,
            "max_points_semantics": (
                "maximum stored adaptive replacement points; adaptive points replace positions "
                "inside the fixed interior batch and do not increase current_interior_points"
            ),
            "maximum_sampling_points": self.maximum_sampling_points,
        }
        self.adaptive_refinement_events.append(event)
        if self.maximum_sampling_points is not None and current_total > self.maximum_sampling_points:
            raise RuntimeError(
                "Actual sampling budget exceeded: "
                f"{current_total} > {self.maximum_sampling_points}"
            )


def legacy_wave_split_weights(
    aggregate_weight: float,
    sample_counts: dict[str, int],
) -> dict[str, float]:
    """Return split weights exactly equivalent to concatenated legacy MSE.

    For residual blocks with ``n_i`` samples, ``MSE(concat(blocks))`` equals
    ``sum(n_i / sum(n) * MSE(block_i))``.
    """

    names = (
        "spatial_boundary",
        "initial_displacement",
        "initial_velocity",
    )
    counts = {name: int(sample_counts.get(name, 0)) for name in names}
    if any(value < 0 for value in counts.values()) or sum(counts.values()) <= 0:
        raise ValueError("Legacy Wave compatibility requires positive sample coverage")
    total = float(sum(counts.values()))
    return {
        name: float(aggregate_weight) * counts[name] / total
        for name in names
    }


_GOVERNING_LOSS_TERM_NAMES = frozenset(
    {"pde_residual", "governing_residual", "pde_component"}
)
_GOVERNING_COMPONENT_PREFIX = "pde/"


def _governing_component_id(residual_name: str) -> str:
    """Return the stable public loss ID for one governing equation."""

    return f"{_GOVERNING_COMPONENT_PREFIX}{str(residual_name)}"


def _is_governing_component_id(component_id: str) -> bool:
    name = str(component_id)
    return (
        name in _GOVERNING_LOSS_TERM_NAMES
        or name.startswith(_GOVERNING_COMPONENT_PREFIX)
        or name == "gradient_residual"
    )


def _problem_governing_residual_names(problem: Any) -> tuple[str, ...]:
    if problem is None:
        return ()
    get_spec = getattr(problem, "get_spec", None)
    if not callable(get_spec):
        return ()
    return tuple(str(law.law_id) for law in get_spec().governing_laws)


def _expand_governing_loss_terms(
    spec: dict[str, Any], problem: Any
) -> tuple[list[dict[str, Any]], dict[str, tuple[str, ...]]]:
    """Expand aggregate PDE terms into independently addressable equations.

    AlgorithmSpec remains backward compatible: a declared ``pde_residual``
    still means "all governing equations". The executable runtime exposes one
    stable ``pde/<law_id>`` component per equation so weighting, diagnostics,
    validation probes, and optimizers can address them separately.
    """

    governing_names = _problem_governing_residual_names(problem)
    runtime_terms: list[dict[str, Any]] = []
    expansion_map: dict[str, tuple[str, ...]] = {}
    for declared_index, declared in enumerate(spec.get("terms") or []):
        term = deepcopy(declared)
        source_name = str(term["name"])
        parameters = dict(term.get("parameters") or {})
        if source_name not in _GOVERNING_LOSS_TERM_NAMES or not governing_names:
            term["_source_name"] = source_name
            runtime_terms.append(term)
            expansion_map[source_name] = (source_name,)
            continue

        selected_name = parameters.get("component_name")
        selected_index = parameters.get("component_index")
        if selected_name is not None:
            selected = (str(selected_name),)
        elif selected_index is not None:
            index = int(selected_index)
            if index < 0 or index >= len(governing_names):
                _raise(
                    "invalid_loss_component_selector",
                    f"loss.terms[{declared_index}].parameters.component_index",
                    index,
                    f"Governing residual component index {index} is out of range",
                )
            selected = (governing_names[index],)
        elif source_name == "pde_component":
            _raise(
                "missing_loss_component_selector",
                f"loss.terms[{declared_index}].parameters",
                parameters,
                "pde_component requires component_name or component_index",
            )
        else:
            selected = governing_names

        expanded_ids: list[str] = []
        for residual_name in selected:
            runtime_term = deepcopy(term)
            component_id = _governing_component_id(residual_name)
            runtime_term["name"] = component_id
            runtime_term["_source_name"] = source_name
            runtime_term["_governing_residual_name"] = residual_name
            runtime_term["parameters"] = {
                **parameters,
                "component_name": residual_name,
            }
            runtime_term["parameters"].pop("component_index", None)
            runtime_terms.append(runtime_term)
            expanded_ids.append(component_id)
        expansion_map[source_name] = tuple(expanded_ids)

    runtime_ids = [str(term["name"]) for term in runtime_terms]
    duplicates = sorted(
        {component_id for component_id in runtime_ids if runtime_ids.count(component_id) > 1}
    )
    if duplicates:
        _raise(
            "duplicate_executable_loss_component",
            "loss.terms",
            spec.get("terms") or [],
            f"Duplicate executable loss component IDs: {duplicates}",
        )
    return runtime_terms, expansion_map


class OpenSpecLoss:
    def __init__(self, problem: Any, sampler: OpenSpecSampler, spec: dict[str, Any]) -> None:
        self.problem = problem
        self.sampler = sampler
        self.spec = deepcopy(spec)
        self._runtime_terms, self._term_expansion_map = _expand_governing_loss_terms(
            self.spec, problem
        )
        self._adaptive_weights: torch.Tensor | None = None
        self._ema_magnitudes: torch.Tensor | None = None
        self._ntk_weights: torch.Tensor | None = None
        self._last_ntk_diagnostics: dict[str, Any] | None = None
        self._weighting_calls = 0
        self._last_lra_update_call: int | None = None
        self.last_term_values: list[torch.Tensor] = []
        self.last_term_names: list[str] = []
        self.last_raw_losses: dict[str, float] = {}
        self.last_weighted_losses: dict[str, float] = {}
        self.last_grouped_losses: dict[str, float] = {}
        self.last_loss_groups: dict[str, list[str]] = {}
        # These are the effective weights before the first dynamic update.
        self.last_effective_loss_weights: dict[str, float] = {
            str(term["name"]): float(term["weight"])
            for term in self._runtime_terms
        }
        self.last_dynamic_weighting: dict[str, Any] | None = None
        self.capture_residual_statistics = False
        self.last_residual_statistics: dict[str, Any] = {}
        self.active_term_names: set[str] | None = None
        self._runtime_base_weights: dict[str, float] = {
            str(term["name"]): float(term["weight"])
            for term in self._runtime_terms
        }
        self._declared_base_weights = dict(self._runtime_base_weights)
        problem_metadata: dict[str, Any] = {}
        if problem is not None:
            get_spec = getattr(problem, "get_spec", None)
            if callable(get_spec):
                problem_metadata = dict(get_spec().metadata or {})
        raw_floor_ratios = dict(
            problem_metadata.get("dynamic_loss_weight_floor_ratios") or {}
        )
        self._dynamic_loss_weight_floor_ratios: dict[str, float] = {}
        for name, value in raw_floor_ratios.items():
            ratio = float(value)
            if not math.isfinite(ratio) or ratio <= 0.0:
                raise ValueError(
                    "dynamic_loss_weight_floor_ratios values must be finite "
                    "and positive"
                )
            expanded_names = self._term_expansion_map.get(str(name), (str(name),))
            for expanded_name in expanded_names:
                self._dynamic_loss_weight_floor_ratios[expanded_name] = ratio
        self._dynamic_updates_frozen = False
        self._frozen_effective_weights: dict[str, float] | None = None
        self._frozen_term_names: tuple[str, ...] | None = None

    def get_runtime_loss_weights(self) -> dict[str, float]:
        return dict(self._runtime_base_weights)

    def set_runtime_loss_weights(self, weights: dict[str, float]) -> None:
        """Apply runtime weights without mutating the normalized AlgorithmSpec."""

        weights = self._expand_legacy_weight_mapping(weights)
        if set(weights) != set(self._runtime_base_weights):
            raise ValueError("Runtime loss-weight updates cannot add or remove loss terms")
        validated: dict[str, float] = {}
        for name, value in weights.items():
            scalar = float(value)
            if not math.isfinite(scalar) or scalar <= 0.0:
                raise ValueError(f"Runtime loss weight for {name!r} must be finite and positive")
            validated[str(name)] = scalar
        self._runtime_base_weights = validated

    def _expand_legacy_weight_mapping(
        self, weights: dict[str, float]
    ) -> dict[str, float]:
        """Migrate aggregate PDE weights from pre-expansion checkpoints."""

        supplied = {str(name): float(value) for name, value in weights.items()}
        if set(supplied) == set(self._runtime_base_weights):
            return supplied
        migrated: dict[str, float] = {}
        for declared_name, runtime_names in self._term_expansion_map.items():
            if declared_name in supplied:
                for runtime_name in runtime_names:
                    migrated[runtime_name] = supplied[declared_name]
            else:
                for runtime_name in runtime_names:
                    if runtime_name in supplied:
                        migrated[runtime_name] = supplied[runtime_name]
        return migrated

    def _active_runtime_terms(self) -> list[dict[str, Any]]:
        if self.active_term_names is None:
            return list(self._runtime_terms)
        return [
            term
            for term in self._runtime_terms
            if str(term["name"]) in self.active_term_names
        ]

    def freeze_dynamic_weighting(self) -> dict[str, Any]:
        names = tuple(
            str(term["name"])
            for term in self._active_runtime_terms()
        )
        effective = {
            name: float(
                self._runtime_base_weights[name]
                if self.spec["weighting_strategy"]["name"] == "fixed"
                else self.last_effective_loss_weights.get(
                    name, self._runtime_base_weights[name]
                )
            )
            for name in names
        }
        self._dynamic_updates_frozen = True
        self._frozen_term_names = names
        self._frozen_effective_weights = effective
        return {
            "frozen_loss_weights": dict(effective),
            "frozen_loss_terms": list(names),
        }

    def unfreeze_dynamic_weighting(self) -> None:
        self._dynamic_updates_frozen = False
        self._frozen_effective_weights = None
        self._frozen_term_names = None

    @staticmethod
    def _clone_optional_tensor(value: torch.Tensor | None) -> torch.Tensor | None:
        return value.detach().clone().cpu() if isinstance(value, torch.Tensor) else None

    def state_dict(self) -> dict[str, Any]:
        return {
            "runtime_base_weights": dict(self._runtime_base_weights),
            "adaptive_weights": self._clone_optional_tensor(self._adaptive_weights),
            "ema_magnitudes": self._clone_optional_tensor(self._ema_magnitudes),
            "ntk_weights": self._clone_optional_tensor(self._ntk_weights),
            "last_ntk_diagnostics": deepcopy(self._last_ntk_diagnostics),
            "weighting_calls": int(self._weighting_calls),
            "last_lra_update_call": self._last_lra_update_call,
            "last_effective_loss_weights": dict(self.last_effective_loss_weights),
            "last_grouped_losses": dict(self.last_grouped_losses),
            "last_loss_groups": deepcopy(self.last_loss_groups),
            "last_dynamic_weighting": deepcopy(self.last_dynamic_weighting),
            "active_term_names": sorted(self.active_term_names) if self.active_term_names else None,
            "dynamic_updates_frozen": bool(self._dynamic_updates_frozen),
            "frozen_effective_weights": deepcopy(self._frozen_effective_weights),
            "frozen_term_names": list(self._frozen_term_names) if self._frozen_term_names else None,
        }

    def load_state_dict(self, state: dict[str, Any] | None) -> None:
        state = dict(state or {})
        if state.get("runtime_base_weights"):
            self.set_runtime_loss_weights(dict(state["runtime_base_weights"]))
        for attribute, key in (
            ("_adaptive_weights", "adaptive_weights"),
            ("_ema_magnitudes", "ema_magnitudes"),
            ("_ntk_weights", "ntk_weights"),
        ):
            value = state.get(key)
            setattr(self, attribute, value if isinstance(value, torch.Tensor) else None)
        self._last_ntk_diagnostics = deepcopy(state.get("last_ntk_diagnostics"))
        self._weighting_calls = int(state.get("weighting_calls") or 0)
        last_lra = state.get("last_lra_update_call")
        self._last_lra_update_call = int(last_lra) if last_lra is not None else None
        self.last_effective_loss_weights = self._expand_legacy_weight_mapping(
            dict(state.get("last_effective_loss_weights") or self._runtime_base_weights)
        )
        self.last_grouped_losses = dict(state.get("last_grouped_losses") or {})
        self.last_loss_groups = {
            str(name): [str(value) for value in values]
            for name, values in (state.get("last_loss_groups") or {}).items()
        }
        self.last_dynamic_weighting = deepcopy(state.get("last_dynamic_weighting"))
        active = state.get("active_term_names")
        self.set_active_term_names(set(active) if active else None)
        self._dynamic_updates_frozen = bool(state.get("dynamic_updates_frozen", False))
        frozen = state.get("frozen_effective_weights")
        self._frozen_effective_weights = (
            self._expand_legacy_weight_mapping(dict(frozen)) if frozen else None
        )
        names = state.get("frozen_term_names")
        if names:
            expanded_names: list[str] = []
            for name in map(str, names):
                expanded_names.extend(self._term_expansion_map.get(name, (name,)))
            self._frozen_term_names = tuple(expanded_names)
        else:
            self._frozen_term_names = None

    def set_active_term_names(self, names: set[str] | None) -> None:
        if names is None:
            self.active_term_names = None
            return
        expanded: set[str] = set()
        for name in map(str, names):
            expanded.update(self._term_expansion_map.get(name, (name,)))
        self.active_term_names = expanded

    def compute_pointwise_component_scores(
        self,
        model: nn.Module,
        component_ids: tuple[str, ...],
        points: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return unweighted point-local losses for registered domain components."""

        requested: set[str] = set()
        for component_id in map(str, component_ids):
            requested.update(
                self._term_expansion_map.get(component_id, (component_id,))
            )
        domain = points.detach().clone().to(
            next(model.parameters()).device
        ).requires_grad_(True)
        governing = self.problem.compute_governing_residuals(
            model, domain, create_graph=True
        )
        transform = getattr(
            self.problem, "training_governing_residuals", None
        )
        if callable(transform):
            governing = dict(transform(domain, governing))
        results: dict[str, torch.Tensor] = {}
        try:
            for term in self._runtime_terms:
                name = str(term["name"])
                if name not in requested:
                    continue
                residual = self._term_residual(term, domain, governing, {})
                if residual.ndim == 0 or int(residual.shape[0]) != int(domain.shape[0]):
                    continue
                flattened = residual.reshape(domain.shape[0], -1)
                results[name] = flattened.square().mean(dim=1).detach()
        finally:
            clear = getattr(self.problem, "clear_autodiff_cache", None)
            if callable(clear):
                clear()
        return results

    def __call__(self, model: nn.Module, batch: PhysicsBatch | None = None) -> tuple[torch.Tensor, dict[str, float]]:
        batch = batch or self.sampler.sample()
        # Every differential PDE needs coordinates that track gradients.  The
        # extra gradient-residual term merely differentiates the resulting PDE
        # residual once more; it is not the condition for enabling autograd.
        domain = batch.domain_samples.detach().clone().requires_grad_(True)
        raw_governing = self.problem.compute_governing_residuals(
            model, domain, create_graph=True
        )
        transform = getattr(
            self.problem, "training_governing_residuals", None
        )
        governing = (
            dict(transform(domain, raw_governing))
            if callable(transform)
            else raw_governing
        )
        if set(governing) != set(raw_governing):
            raise ValueError(
                "Training residual scaling cannot add or remove PDE components"
            )
        constraints = self.problem.compute_constraint_residuals(model, batch.constraint_samples, create_graph=True)
        components: dict[str, float] = {}
        self.last_residual_statistics = {}
        values: list[torch.Tensor] = []
        raw_residuals: list[torch.Tensor] = []
        term_names: list[str] = []
        base_weights: list[float] = []
        for index, term in enumerate(self._active_runtime_terms()):
            residual = self._term_residual(term, domain, governing, constraints)
            if (
                self.capture_residual_statistics
                and str(term.get("_source_name")) in _GOVERNING_LOSS_TERM_NAMES
            ):
                absolute = residual.detach().abs()
                raw_residual = self._term_residual(
                    term, domain, raw_governing, constraints
                ).detach().abs()
                component_statistics: dict[str, Any] = {
                    "maximum_residual": float(absolute.max().cpu()),
                    "mean_residual": float(absolute.mean().cpu()),
                    "raw_maximum_residual": float(
                        raw_residual.max().cpu()
                    ),
                    "raw_mean_residual": float(raw_residual.mean().cpu()),
                }
                region_ids_provider = getattr(
                    self.problem, "training_residual_region_ids", None
                )
                if (
                    callable(region_ids_provider)
                    and residual.ndim > 0
                    and raw_residual.ndim > 0
                    and int(residual.shape[0]) == int(domain.shape[0])
                    and int(raw_residual.shape[0]) == int(domain.shape[0])
                ):
                    region_ids = region_ids_provider(domain)
                    if (
                        isinstance(region_ids, torch.Tensor)
                        and region_ids.ndim == 1
                        and int(region_ids.shape[0]) == int(domain.shape[0])
                    ):
                        pointwise_training_mse = (
                            residual.detach()
                            .reshape(domain.shape[0], -1)
                            .square()
                            .mean(dim=1)
                        )
                        pointwise_raw_mse = (
                            raw_residual
                            .reshape(domain.shape[0], -1)
                            .square()
                            .mean(dim=1)
                        )
                        region_statistics: list[dict[str, Any]] = []
                        for region_id in torch.unique(
                            region_ids.detach().to(dtype=torch.long)
                        ).tolist():
                            mask = region_ids == int(region_id)
                            training_rmse = pointwise_training_mse[
                                mask
                            ].mean().sqrt()
                            raw_rmse = pointwise_raw_mse[mask].mean().sqrt()
                            region_statistics.append(
                                {
                                    "region_id": int(region_id),
                                    "sample_count": int(mask.sum().detach().cpu()),
                                    "training_residual_rmse": float(
                                        training_rmse.detach().cpu()
                                    ),
                                    "raw_residual_rmse": float(
                                        raw_rmse.detach().cpu()
                                    ),
                                }
                            )
                        if region_statistics:
                            training_values = [
                                item["training_residual_rmse"]
                                for item in region_statistics
                            ]
                            raw_values = [
                                item["raw_residual_rmse"]
                                for item in region_statistics
                            ]
                            component_statistics.update(
                                {
                                    "region_statistics": region_statistics,
                                    "mean_region_training_residual_rmse": float(
                                        sum(training_values)
                                        / len(training_values)
                                    ),
                                    "worst_region_training_residual_rmse": float(
                                        max(training_values)
                                    ),
                                    "mean_region_raw_residual_rmse": float(
                                        sum(raw_values) / len(raw_values)
                                    ),
                                    "worst_region_raw_residual_rmse": float(
                                        max(raw_values)
                                    ),
                                }
                            )
                component_id = str(term["name"])
                stored_components = dict(
                    self.last_residual_statistics.get("components") or {}
                )
                stored_components[component_id] = component_statistics
                maximums = [
                    float(item["maximum_residual"])
                    for item in stored_components.values()
                ]
                means = [
                    float(item["mean_residual"])
                    for item in stored_components.values()
                ]
                raw_maximums = [
                    float(item["raw_maximum_residual"])
                    for item in stored_components.values()
                ]
                raw_means = [
                    float(item["raw_mean_residual"])
                    for item in stored_components.values()
                ]
                self.last_residual_statistics = {
                    "components": stored_components,
                    "maximum_residual": max(maximums),
                    "mean_residual": sum(means) / len(means),
                    "raw_maximum_residual": max(raw_maximums),
                    "raw_mean_residual": sum(raw_means) / len(raw_means),
                }
                regional_by_id: dict[int, list[dict[str, Any]]] = {}
                for item in stored_components.values():
                    for regional in item.get("region_statistics") or []:
                        regional_by_id.setdefault(
                            int(regional["region_id"]), []
                        ).append(regional)
                if regional_by_id:
                    aggregate_regions = []
                    for region_id, region_values in sorted(regional_by_id.items()):
                        training_rmse = math.sqrt(
                            sum(
                                float(value["training_residual_rmse"]) ** 2
                                for value in region_values
                            )
                            / len(region_values)
                        )
                        raw_rmse = math.sqrt(
                            sum(
                                float(value["raw_residual_rmse"]) ** 2
                                for value in region_values
                            )
                            / len(region_values)
                        )
                        aggregate_regions.append(
                            {
                                "region_id": region_id,
                                "sample_count": min(
                                    int(value["sample_count"])
                                    for value in region_values
                                ),
                                "training_residual_rmse": training_rmse,
                                "raw_residual_rmse": raw_rmse,
                            }
                        )
                    training_values = [
                        item["training_residual_rmse"]
                        for item in aggregate_regions
                    ]
                    raw_values = [
                        item["raw_residual_rmse"] for item in aggregate_regions
                    ]
                    self.last_residual_statistics.update(
                        {
                            "region_statistics": aggregate_regions,
                            "mean_region_training_residual_rmse": sum(
                                training_values
                            )
                            / len(training_values),
                            "worst_region_training_residual_rmse": max(
                                training_values
                            ),
                            "mean_region_raw_residual_rmse": sum(raw_values)
                            / len(raw_values),
                            "worst_region_raw_residual_rmse": max(raw_values),
                        }
                    )
            reducer = LOSS_FUNCTIONS.get(term["loss_function"])
            if reducer is None:
                _raise("unsupported_component", f"loss.terms[{index}].loss_function", term["loss_function"], "Unknown loss function")
            value = reducer(residual, dict(term.get("parameters") or {}))
            values.append(value)
            raw_residuals.append(residual)
            term_names.append(str(term["name"]))
            base_weights.append(float(self._runtime_base_weights[str(term["name"])]))
            components[term["name"]] = float(value.detach().cpu())
        if not values:
            _raise(
                "invalid_training_stage",
                "training.initial_state_pretraining",
                sorted(self.active_term_names or []),
                "The active training stage selected no loss terms",
            )
        self.last_term_values = values
        self.last_term_names = term_names
        total = self._aggregate(
            model,
            values,
            base_weights,
            term_names,
            raw_residuals=raw_residuals,
        )
        components["total_loss"] = float(total.detach().cpu())
        return total, components

    def _aggregate(
        self,
        model: nn.Module,
        values: list[torch.Tensor],
        base_weights: list[float],
        term_names: list[str] | None = None,
        *,
        raw_residuals: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        stacked = torch.stack(values)
        weights = torch.tensor(base_weights, dtype=stacked.dtype, device=stacked.device)
        names = list(
            term_names
            or [f"loss_{index}" for index in range(len(values))]
        )
        weighting = self.spec["weighting_strategy"]
        weighting_name = weighting["name"]
        weighting_parameters = dict(weighting.get("parameters") or {})
        self.last_dynamic_weighting = None
        if self._dynamic_updates_frozen:
            if tuple(names) != tuple(self._frozen_term_names or ()):
                raise RuntimeError("Loss-term membership changed during frozen L-BFGS phase")
            frozen = self._frozen_effective_weights or {}
            weights = torch.tensor(
                [float(frozen[name]) for name in names],
                dtype=stacked.dtype,
                device=stacked.device,
            )
            self.last_dynamic_weighting = {
                "strategy": weighting_name,
                "updated": False,
                "frozen": True,
                "effective_weights": dict(frozen),
            }
        elif weighting_name == "normalized":
            epsilon = float(weighting_parameters.get("epsilon", 1e-12))
            weights = weights / torch.clamp(weights.abs().sum(), min=epsilon)
        elif weighting_name == "softmax":
            temperature = max(float(weighting_parameters.get("temperature", 1.0)), 1e-8)
            weights = torch.softmax(weights / temperature, dim=0)
        elif weighting_name == "inverse_magnitude":
            epsilon = float(weighting_parameters.get("epsilon", 1e-8))
            power = float(weighting_parameters.get("power", 1.0))
            weights = weights / (stacked.detach().abs() + epsilon).pow(power)
            if bool(weighting_parameters.get("normalize", True)):
                weights = weights * (len(values) / torch.clamp(weights.abs().sum(), min=epsilon))
        elif weighting_name == "lra":
            alpha = float(weighting_parameters.get("alpha", 0.1))
            epsilon = float(weighting_parameters.get("epsilon", 1e-12))
            reference_index = int(weighting_parameters.get("reference_index", 0))
            if reference_index < 0 or reference_index >= len(values):
                _raise("invalid_parameter", "loss.weighting_strategy.parameters.reference_index", reference_index, "LRA reference index is out of range")
            maximum = float(weighting_parameters.get("max_weight", 20.0))
            minimum = float(weighting_parameters.get("min_weight", 0.01))
            gradient_stats = [self._gradient_stat(value, model, mode="mean") for value in values]
            reference_stat = self._gradient_stat(values[reference_index], model, mode="max")
            targets = weights.detach().clone()
            for index, statistic in enumerate(gradient_stats):
                if index != reference_index:
                    targets[index] = torch.clamp(reference_stat / (statistic + epsilon), min=minimum, max=maximum)
            previous = (
                self._adaptive_weights.to(targets.device)
                if self._adaptive_weights is not None
                and self._adaptive_weights.shape == targets.shape
                else weights.detach().clone()
            )
            cooldown = max(
                0, int(weighting_parameters.get("cooldown_iterations", 0))
            )
            in_cooldown = (
                self._last_lra_update_call is not None
                and self._weighting_calls - self._last_lra_update_call <= cooldown
            )
            if in_cooldown:
                weights = previous
            else:
                weights = self._ema_weights(targets, alpha)
                minimum_ratio = float(
                    weighting_parameters.get("minimum_update_ratio", 0.5)
                )
                maximum_ratio = float(
                    weighting_parameters.get("maximum_update_ratio", 2.0)
                )
                weights = torch.minimum(
                    torch.maximum(weights, previous * minimum_ratio),
                    previous * maximum_ratio,
                ).clamp(min=minimum, max=maximum)
                self._adaptive_weights = weights.detach().clone()
                self._last_lra_update_call = int(self._weighting_calls)
            self.last_dynamic_weighting = {
                "strategy": "lra",
                "updated": not in_cooldown,
                "in_cooldown": in_cooldown,
                "gradient_statistics": [
                    float(value.detach().cpu()) for value in gradient_stats
                ],
                "reference_gradient_statistic": float(reference_stat.detach().cpu()),
                "effective_weights": {
                    name: float(weight.detach().cpu())
                    for name, weight in zip(names, weights)
                },
            }
        elif weighting_name == "ntk":
            weights = self._ntk_group_weights(
                model,
                values,
                names,
                weighting_parameters,
                weights,
                raw_residuals=raw_residuals,
            )
        elif weighting_name == "ntk_trace":
            alpha = float(weighting_parameters.get("alpha", 0.1))
            epsilon = float(weighting_parameters.get("epsilon", 1e-12))
            traces = torch.stack([self._gradient_stat(value, model, mode="squared") for value in values])
            targets = traces.sum() / torch.clamp(traces, min=epsilon)
            if bool(weighting_parameters.get("normalize", True)):
                targets = targets * (len(values) / torch.clamp(targets.sum(), min=epsilon))
            weights = self._ema_weights(targets, alpha)
        elif weighting_name == "ema_inverse_magnitude":
            alpha = min(max(float(weighting_parameters.get("alpha", 0.1)), 0.0), 1.0)
            epsilon = float(weighting_parameters.get("epsilon", 1e-8))
            power = float(weighting_parameters.get("power", 1.0))
            magnitudes = stacked.detach().abs()
            if self._ema_magnitudes is None or self._ema_magnitudes.shape != magnitudes.shape:
                self._ema_magnitudes = magnitudes.clone()
            else:
                self._ema_magnitudes = (
                    (1.0 - alpha) * self._ema_magnitudes.to(magnitudes.device)
                    + alpha * magnitudes
                )
            weights = weights / (self._ema_magnitudes + epsilon).pow(power)
            if bool(weighting_parameters.get("normalize", True)):
                weights = weights * (len(values) / torch.clamp(weights.abs().sum(), min=epsilon))
        elif weighting_name == "loss_softmax":
            temperature = max(float(weighting_parameters.get("temperature", 1.0)), 1e-8)
            dynamic = torch.softmax(stacked.detach().abs() / temperature, dim=0)
            weights = weights * dynamic
            if bool(weighting_parameters.get("normalize", True)):
                weights = weights * (len(values) / torch.clamp(weights.abs().sum(), min=1e-12))
        elif weighting_name == "gradient_balance":
            # The bounded Trainer controller updates the runtime base weights.
            pass
        elif weighting_name != "fixed" and not self._dynamic_updates_frozen:
            _raise("unsupported_component", "loss.weighting_strategy.name", weighting_name, "Unknown weighting strategy")

        protected_weight_floors: dict[str, float] = {}
        dynamically_scaled_strategies = {
            "lra",
            "inverse_magnitude",
            "ntk",
            "ntk_trace",
            "ema_inverse_magnitude",
            "loss_softmax",
            "gradient_balance",
        }
        if (
            not self._dynamic_updates_frozen
            and weighting_name in dynamically_scaled_strategies
            and self._dynamic_loss_weight_floor_ratios
        ):
            weights = weights.clone()
            for index, name in enumerate(names):
                ratio = self._dynamic_loss_weight_floor_ratios.get(name)
                declared = self._declared_base_weights.get(name)
                if ratio is None or declared is None:
                    continue
                floor_value = float(declared) * float(ratio)
                if float(weights[index].detach().cpu()) < floor_value:
                    weights[index] = weights.new_tensor(floor_value)
                    protected_weight_floors[name] = floor_value
            if protected_weight_floors:
                if self.last_dynamic_weighting is None:
                    self.last_dynamic_weighting = {
                        "strategy": weighting_name,
                        "updated": False,
                    }
                self.last_dynamic_weighting["protected_weight_floors"] = dict(
                    protected_weight_floors
                )
                self.last_dynamic_weighting["effective_weights"] = {
                    name: float(weight.detach().cpu())
                    for name, weight in zip(names, weights)
                }

        aggregation_values: torch.Tensor | None = None
        self.last_grouped_losses = {}
        self.last_loss_groups = {}
        if weighting_name == "fixed":
            # Fixed weights describe importance *within* the PDE and constraint
            # groups. Dividing by group size prevents a system with three PDEs
            # or nine boundary conditions from gaining weight merely because it
            # has more registered equations/constraints.
            group_indices: dict[str, list[int]] = {}
            for index, name in enumerate(names):
                group_name = (
                    "governing"
                    if _is_governing_component_id(name)
                    else "constraints"
                )
                group_indices.setdefault(group_name, []).append(index)
            effective_weights = weights.clone()
            for indices in group_indices.values():
                effective_weights[indices] = effective_weights[indices] / float(
                    len(indices)
                )
            weights = effective_weights
            self.last_loss_groups = {
                group_name: [names[index] for index in indices]
                for group_name, indices in group_indices.items()
            }
        self.last_effective_loss_weights = {
            name: float(weight.detach().cpu())
            for name, weight in zip(names, weights)
        }
        if not self._dynamic_updates_frozen:
            self._weighting_calls += 1
        weighted = weights * stacked
        self.last_raw_losses = {
            name: float(value.detach().cpu())
            for name, value in zip(names, stacked)
        }
        self.last_weighted_losses = {
            name: float(value.detach().cpu())
            for name, value in zip(names, weighted)
        }
        if weighting_name == "fixed":
            grouped_values = []
            for group_name, grouped_names in self.last_loss_groups.items():
                indices = [names.index(name) for name in grouped_names]
                group_value = weighted[indices].sum()
                grouped_values.append(group_value)
                self.last_grouped_losses[group_name] = float(
                    group_value.detach().cpu()
                )
            aggregation_values = torch.stack(grouped_values)
        aggregation = self.spec["aggregation"]
        aggregation_name = aggregation["name"]
        aggregation_parameters = dict(aggregation.get("parameters") or {})
        values_to_aggregate = (
            aggregation_values if aggregation_values is not None else weighted
        )
        if aggregation_name == "weighted_sum":
            return values_to_aggregate.sum()
        if aggregation_name == "mean":
            return values_to_aggregate.mean()
        if aggregation_name == "max":
            return values_to_aggregate.max()
        if aggregation_name == "logsumexp":
            temperature = max(float(aggregation_parameters.get("temperature", 1.0)), 1e-8)
            return temperature * torch.logsumexp(
                values_to_aggregate / temperature, dim=0
            )
        if aggregation_name == "p_norm":
            p = float(aggregation_parameters.get("p", 2.0))
            epsilon = float(aggregation_parameters.get("epsilon", 1e-12))
            return (values_to_aggregate.abs().pow(p).sum() + epsilon).pow(1.0 / p)
        if aggregation_name == "root_mean_square":
            epsilon = float(aggregation_parameters.get("epsilon", 1e-12))
            return torch.sqrt(values_to_aggregate.square().mean() + epsilon)
        _raise("unsupported_component", "loss.aggregation.name", aggregation_name, "Unknown loss aggregation")

    def _ntk_group_weights(
        self,
        model: nn.Module,
        values: list[torch.Tensor],
        names: list[str],
        parameters: dict[str, Any],
        base_weights: torch.Tensor,
        *,
        raw_residuals: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        gradient_measure = str(
            parameters.get("gradient_measure", "residual_sum")
        )
        if gradient_measure not in {"residual_sum", "mse"}:
            _raise(
                "invalid_parameter",
                "loss.weighting_strategy.parameters.gradient_measure",
                gradient_measure,
                "NTK gradient_measure must be residual_sum or mse",
            )
        if gradient_measure == "residual_sum" and raw_residuals is not None:
            if len(raw_residuals) != len(values):
                raise ValueError(
                    "NTK raw residual count must match the loss-term count"
                )
            trace_values = [residual.sum() for residual in raw_residuals]
        else:
            # The MSE path is retained only for direct historical compatibility.
            # Formal `ntk` specs omit gradient_measure and therefore use the
            # PINNacle residual-sum path whenever raw residuals are available.
            trace_values = values
        residual_values = [
            value
            for name, value in zip(names, trace_values)
            if _is_governing_component_id(name)
        ]
        constraint_values = [
            value
            for name, value in zip(names, trace_values)
            if not _is_governing_component_id(name)
        ]
        if not residual_values or not constraint_values:
            _raise(
                "invalid_loss_grouping",
                "loss.terms",
                names,
                "NTK requires both a PDE residual group and a constraint group",
            )
        update_every = max(1, int(parameters.get("update_every", 1)))
        should_update = (
            self._ntk_weights is None
            or self._weighting_calls % update_every == 0
        )
        if should_update:
            loss_r = torch.stack(residual_values).sum()
            loss_c = torch.stack(constraint_values).sum()
            raw_r = self._gradient_stat(loss_r, model, mode="squared")
            raw_c = self._gradient_stat(loss_c, model, mode="squared")
            epsilon = float(parameters.get("epsilon", 1e-12))
            maximum_safe_norm = 1.0 / epsilon

            def safe_norm(value: torch.Tensor) -> float:
                scalar = float(value.detach().cpu())
                if not math.isfinite(scalar) or scalar < 0.0:
                    return epsilon
                return min(max(scalar, epsilon), maximum_safe_norm)

            safe_r = safe_norm(raw_r)
            safe_c = safe_norm(raw_c)
            total = safe_r + safe_c
            minimum = float(parameters.get("min_weight", 1e-6))
            maximum = float(parameters.get("max_weight", 1e6))
            raw_residual_multiplier = total / safe_r
            raw_constraint_multiplier = total / safe_c
            residual_multiplier = min(
                max(raw_residual_multiplier, minimum), maximum
            )
            constraint_multiplier = min(
                max(raw_constraint_multiplier, minimum), maximum
            )
            targets = torch.tensor(
                [
                    residual_multiplier
                    if _is_governing_component_id(name)
                    else constraint_multiplier
                    for name in names
                ],
                dtype=base_weights.dtype,
                device=base_weights.device,
            )
            self._ntk_weights = targets.detach().clone()
            self._last_ntk_diagnostics = {
                "name": "ntk",
                "gradient_measure": gradient_measure,
                "residual_gradient_norm_sq": float(raw_r.detach().cpu())
                if torch.isfinite(raw_r)
                else None,
                "constraint_gradient_norm_sq": float(raw_c.detach().cpu())
                if torch.isfinite(raw_c)
                else None,
                "residual_multiplier": residual_multiplier,
                "constraint_multiplier": constraint_multiplier,
                "residual_weight_clipped": (
                    residual_multiplier != raw_residual_multiplier
                ),
                "constraint_weight_clipped": (
                    constraint_multiplier != raw_constraint_multiplier
                ),
                "updated": True,
                "update_every": update_every,
            }
            self.last_dynamic_weighting = dict(self._last_ntk_diagnostics)
        else:
            self.last_dynamic_weighting = dict(
                self._last_ntk_diagnostics
                or {
                    "name": "ntk",
                    "gradient_measure": gradient_measure,
                    "residual_gradient_norm_sq": None,
                    "constraint_gradient_norm_sq": None,
                    "residual_multiplier": None,
                    "constraint_multiplier": None,
                    "residual_weight_clipped": False,
                    "constraint_weight_clipped": False,
                    "update_every": update_every,
                }
            )
            self.last_dynamic_weighting["updated"] = False
        return (
            self._ntk_weights.to(
                dtype=base_weights.dtype, device=base_weights.device
            )
            if self._ntk_weights is not None
            else base_weights
        )

    def _ema_weights(self, targets: torch.Tensor, alpha: float) -> torch.Tensor:
        alpha = min(max(float(alpha), 0.0), 1.0)
        if self._adaptive_weights is None or self._adaptive_weights.shape != targets.shape:
            self._adaptive_weights = targets.detach().clone()
        else:
            self._adaptive_weights = (1.0 - alpha) * self._adaptive_weights.to(targets.device) + alpha * targets.detach()
        return self._adaptive_weights

    @staticmethod
    def _gradient_stat(value: torch.Tensor, model: nn.Module, *, mode: str) -> torch.Tensor:
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not value.requires_grad:
            return torch.zeros((), dtype=value.dtype, device=value.device)
        try:
            gradients = torch.autograd.grad(
                value,
                parameters,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )
        except RuntimeError:
            return torch.zeros((), dtype=value.dtype, device=value.device)
        nonempty = [gradient.detach().reshape(-1) for gradient in gradients if gradient is not None]
        if not nonempty:
            return torch.zeros((), dtype=value.dtype, device=value.device)
        flattened = torch.cat(nonempty)
        if mode == "max":
            return flattened.abs().max()
        if mode == "squared":
            return flattened.square().sum()
        return flattened.abs().mean()

    def _term_residual(
        self,
        term: dict[str, Any],
        domain: torch.Tensor,
        governing: dict[str, torch.Tensor],
        constraints: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        name = str(term.get("_source_name") or term["name"])
        parameters = dict(term.get("parameters") or {})
        if name in {"pde_residual", "governing_residual", "pde_component"}:
            return self._select_residual(governing, parameters, aggregate_default=name != "pde_component")
        if name == "boundary_condition":
            if (
                str(self.problem.get_spec().problem_id) == "wave_1d"
                and not {
                    "spatial_boundary",
                    "initial_displacement",
                    "initial_velocity",
                }.intersection(
                    {str(item.get("name")) for item in self.spec["terms"]}
                )
            ):
                legacy_values = [
                    constraints[key].reshape(-1)
                    for key in (
                        "initial_velocity",
                        "initial_displacement",
                        "spatial_boundary",
                    )
                    if key in constraints
                ]
                if len(legacy_values) == 3:
                    return torch.cat(legacy_values, dim=0)
            return constraints["boundary"]
        if name == "spatial_boundary":
            return constraints["spatial_boundary"]
        if name == "initial_condition":
            return constraints["initial"]
        if name == "initial_displacement":
            return constraints["initial_displacement"]
        if name == "initial_velocity":
            return constraints["initial_velocity"]
        if name == "conservation":
            return constraints["conservation"]
        if name == "constraint_component":
            component_name = str(parameters.get("component_name") or "")
            if not component_name or component_name not in constraints:
                _raise("unsupported_component", "loss.terms[].parameters.component_name", component_name, "Unknown constraint residual component")
            return constraints[component_name]
        if name == "gradient_residual":
            residual = self._select_residual(governing, parameters, aggregate_default=True)
            gradient = torch.autograd.grad(residual, domain, torch.ones_like(residual), create_graph=True)[0]
            indices = parameters.get("input_indices")
            if indices is not None:
                gradient = gradient[:, [int(index) for index in indices]]
            return gradient
        if name in constraints:
            return constraints[name]
        _raise("unsupported_component", "loss.terms[].name", name, f"Unsupported loss term '{name}'")

    @staticmethod
    def _select_residual(
        residuals: dict[str, torch.Tensor],
        parameters: dict[str, Any],
        *,
        aggregate_default: bool,
    ) -> torch.Tensor:
        items = list(residuals.items())
        component_name = parameters.get("component_name")
        component_index = parameters.get("component_index")
        if component_name is not None:
            name = str(component_name)
            if name not in residuals:
                _raise("unsupported_component", "loss.terms[].parameters.component_name", name, f"Unknown residual component '{name}'")
            return residuals[name]
        if component_index is not None:
            index = int(component_index)
            if index < 0 or index >= len(items):
                _raise("invalid_parameter", "loss.terms[].parameters.component_index", index, "Residual component index is out of range")
            return items[index][1]
        if not aggregate_default:
            _raise("missing_parameter", "loss.terms[].parameters", parameters, "pde_component requires component_name or component_index")
        return torch.cat([value.reshape(-1) for _, value in items], dim=0)


class PINNBuilder:
    def build(self, spec: dict, problem: Any, runtime_config: dict | None = None) -> BuiltPINN:
        original = deepcopy(spec)
        augment = getattr(problem, "augment_training_algorithm_spec", None)
        normalized = (
            augment(spec) if callable(augment) else deepcopy(spec)
        )
        runtime_config = dict(runtime_config or {})
        device = str(runtime_config.get("device") or "cpu")
        self._validate_training(normalized["training"])
        model = self._build_network(normalized["network"], problem)
        model = self._apply_constraint_enforcement(
            model,
            normalized["constraint_enforcement"],
            problem,
        ).to(device)
        term_names = {
            str(term.get("name"))
            for term in normalized["loss"].get("terms") or []
        }
        legacy_wave_aggregate = (
            str(problem.get_spec().problem_id) == "wave_1d"
            and "boundary_condition" in term_names
            and not {
                "spatial_boundary",
                "initial_displacement",
                "initial_velocity",
            }.intersection(term_names)
        )
        sampler = self._build_sampler(
            normalized["sampling"],
            problem,
            device,
            maximum_sampling_points=runtime_config.get("maximum_sampling_points"),
            constraint_resample_interval=runtime_config.get(
                "constraint_resample_interval", 50
            ),
            legacy_wave_aggregate=legacy_wave_aggregate,
        )
        loss_function = self._build_loss(normalized["loss"], problem, sampler)
        sampler.set_loss_runtime(loss_function)
        phases = self._build_optimization_phases(normalized["optimization"], model)
        try:
            sampling_budget_plan = resolve_sampling_budget_plan(
                problem, normalized
            )
        except SamplingContractError as exc:
            _raise(
                "invalid_sampling_contract",
                "sampling",
                normalized.get("sampling"),
                str(exc),
            )
        sampler.set_sampling_budget_plan(sampling_budget_plan)
        registry = self._build_training_component_registry(
            normalized, problem, phases, sampler, sampling_budget_plan
        )
        curriculum_runtimes, registry = self._register_curriculum_runtimes(
            normalized, problem, sampler, registry
        )
        sampler.set_component_registry(registry)
        gradient_controller = None
        lbfgs_controller = None
        rollback_manager = None
        audit_logger = None
        action_validator = None
        coordinator = None
        validation_probe_manager = None
        best_checkpoint_manager = None
        sampling_controller = None
        curriculum_controller = None
        training_extra = dict(normalized["training"].get("extra_parameters") or {})
        adaptive_config = dict(
            training_extra.get("adaptive_control")
            or {}
        )
        component_control = dict(training_extra.get("component_control") or {})
        physics_config = dict(training_extra.get("physics_validation") or {})
        best_config = dict(training_extra.get("best_checkpoint") or {})
        sampling_control_config = dict(
            adaptive_config.get("sampling_control") or {}
        )
        curriculum_control_config = dict(
            adaptive_config.get("curriculum_control") or {}
        )
        control_enabled = bool(
            adaptive_config.get("enabled", False)
            or component_control.get("enabled", False)
            or physics_config.get("enabled", False)
            or best_config.get("enabled", False)
            or sampling_control_config.get("enabled", False)
            or curriculum_control_config.get("enabled", False)
        )
        if control_enabled:
            from forge.pipeline.g_training_evaluation.training.controllers import (
                ActionBoundaryValidator,
                AdaptiveAuditLogger,
                CheckpointRollbackManager,
                ControllerCoordinator,
                GradientBalanceController,
                FixedBudgetSamplingController,
                CurriculumAdvanceController,
                LBFGSStallController,
            )
            from forge.pipeline.g_training_evaluation.training.best_train_loss_checkpoint import (
                BestTrainLossCheckpointManager,
            )
            from forge.pipeline.g_training_evaluation.training.validation_probe import (
                ValidationProbeManager,
            )

            weighting = normalized["loss"]["weighting_strategy"]
            if (
                weighting["name"] == "gradient_balance"
                and registry.capabilities.supports_dynamic_loss_weighting
                and registry.capabilities.supports_component_gradient_diagnostics
            ):
                gradient_controller = GradientBalanceController(
                    dict(weighting.get("parameters") or {})
                )
            stall_config = dict(
                adaptive_config.get("lbfgs_stall_recovery") or {}
            )
            if (
                adaptive_config
                and stall_config.get("enabled", True)
                and registry.capabilities.supports_stage_reallocation
            ):
                lbfgs_controller = LBFGSStallController(stall_config)
            if (
                sampling_control_config.get("enabled", False)
                and registry.capabilities.supports_resampling
                and physics_config.get("enabled", False)
            ):
                sampling_controller = FixedBudgetSamplingController(
                    sampling_control_config
                )
            if curriculum_control_config.get("enabled", False):
                if not registry.capabilities.supports_curriculum_advance:
                    _raise(
                        "unsupported_problem_capability",
                        "training.extra_parameters.adaptive_control.curriculum_control",
                        curriculum_control_config,
                        "Curriculum control requires a registered complete Curriculum Runtime",
                    )
                if not physics_config.get("enabled", False):
                    _raise(
                        "unsupported_problem_capability",
                        "training.extra_parameters.physics_validation.enabled",
                        False,
                        "Curriculum control requires fixed physics validation",
                    )
                curriculum_controller = CurriculumAdvanceController(
                    curriculum_control_config
                )
            rollback_manager = CheckpointRollbackManager(
                dict(adaptive_config.get("rollback") or {})
            )
            audit_logger = AdaptiveAuditLogger()
            action_validator = ActionBoundaryValidator()
            coordinator_config = dict(
                training_extra.get("controller_coordinator") or {}
            )
            # Preserve the v1 controller cadence unless the new component
            # control protocol explicitly opts into its global observation
            # window.
            if (
                not coordinator_config
                and (sampling_controller is not None or curriculum_controller is not None)
            ):
                coordinator_config = {
                    "observation_window_iterations": max(
                        int(sampling_control_config.get("observation_window_iterations", 0)),
                        int(curriculum_control_config.get("observation_window_iterations", 0)),
                    ),
                    "global_cooldown_iterations": max(
                        int(sampling_control_config.get("cooldown_iterations", 0)),
                        int(curriculum_control_config.get("cooldown_iterations", 0)),
                    ),
                }
            elif not coordinator_config and not component_control.get("enabled", False):
                coordinator_config = {
                    "observation_window_iterations": 0,
                    "global_cooldown_iterations": 0,
                }
            if coordinator_config.get("enabled", True):
                coordinator = ControllerCoordinator(
                    [
                        item
                        for item in (
                            gradient_controller, lbfgs_controller,
                            sampling_controller,
                            curriculum_controller,
                        )
                        if item is not None
                    ],
                    coordinator_config,
                )
            if (
                physics_config.get("enabled", False)
                and registry.capabilities.supports_validation_probes
            ):
                validation_probe_manager = ValidationProbeManager(physics_config)
            if best_config.get("enabled", False):
                best_checkpoint_manager = BestTrainLossCheckpointManager(best_config)
        if spec != original:
            _raise("builder_mutated_spec", "$", None, "Builder modified the input AlgorithmSpec")
        return BuiltPINN(
            model,
            sampler,
            loss_function,
            phases,
            normalized,
            gradient_controller,
            lbfgs_controller,
            rollback_manager,
            audit_logger,
            registry,
            action_validator,
            coordinator,
            validation_probe_manager,
            best_checkpoint_manager,
            sampling_controller,
            curriculum_runtimes,
            curriculum_controller,
        )

    def _register_curriculum_runtimes(
        self,
        spec: dict[str, Any],
        problem: Any,
        sampler: Any,
        registry: Any,
    ) -> tuple[dict[str, Any], Any]:
        """Build concrete runtimes first, then derive truthful capabilities."""

        from forge.pipeline.g_training_evaluation.training.curriculum_runtime import (
            CurriculumRuntimeProtocol,
        )
        from forge.pipeline.g_training_evaluation.training.registry import (
            TrainingComponentRegistry,
            derive_trainer_capabilities,
        )

        context = {
            "normalized_spec": deepcopy(spec),
            "sampler": sampler,
            "loss_components": registry.loss_components,
            "sampling_components": registry.sampling_components,
            "variable_roles": registry.variable_roles,
            "optimization_stages": registry.optimization_stages,
        }
        produced: Any = None
        factory = getattr(problem, "build_curriculum_runtimes", None)
        if callable(factory):
            produced = factory(context)
        runtimes: dict[str, Any] = {}
        if produced is not None:
            values = (
                tuple(produced.values())
                if isinstance(produced, dict)
                else tuple(produced)
                if isinstance(produced, (tuple, list))
                else (produced,)
            )
            for runtime in values:
                descriptor = runtime.get_descriptor()
                runtimes[descriptor.curriculum_id] = runtime

        adaptive = dict(
            ((spec.get("training") or {}).get("extra_parameters") or {}).get(
                "adaptive_control"
            )
            or {}
        )
        curriculum_control = dict(adaptive.get("curriculum_control") or {})
        time_strategy = dict((spec.get("training") or {}).get("time_strategy") or {})
        if (
            curriculum_control.get("enabled", False)
            and not runtimes
            and time_strategy.get("name") == "time_window_curriculum"
        ):
            from .curriculum_adapter import OrderedSamplerWindowCurriculumRuntime

            parameters = dict(time_strategy.get("parameters") or {})
            temporal_variables = tuple(
                item.variable_id
                for item in registry.variable_roles
                if item.role == "temporal" and item.causal_ordered
            )
            if not temporal_variables:
                _raise(
                    "unsupported_problem_capability",
                    "training.time_strategy",
                    time_strategy,
                    "The registered ordered axis has no temporal variable role",
                )
            runtime = OrderedSamplerWindowCurriculumRuntime(
                sampler=sampler,
                level_values=tuple(
                    float(value)
                    for value in parameters.get("window_fractions") or ()
                ),
                associated_loss_component_ids=tuple(registry.component_ids),
                associated_sampler_ids=tuple(
                    item.sampler_id for item in registry.sampling_components
                ),
                associated_variable_ids=temporal_variables,
                critical_component_ids=tuple(
                    item.component_id
                    for item in registry.loss_components
                    if bool(item.metadata.get("critical_for_valid_solution", False))
                ),
                structural_component_ids=tuple(registry.component_ids),
                structural_sampler_ids=tuple(
                    item.sampler_id for item in registry.sampling_components
                ),
            )
            runtimes[runtime.get_descriptor().curriculum_id] = runtime

        required_methods = (
            "get_descriptor", "get_current_state", "can_advance",
            "preview_advance", "apply_advance", "note_training_iteration",
            "record_rollback", "validate_runtime_state", "snapshot_state",
            "restore_state",
        )
        descriptors = []
        for curriculum_id, runtime in sorted(runtimes.items()):
            if not isinstance(runtime, CurriculumRuntimeProtocol) or not all(
                callable(getattr(runtime, name, None)) for name in required_methods
            ):
                _raise(
                    "invalid_runtime_component",
                    "curriculum_runtime",
                    curriculum_id,
                    "Curriculum Runtime does not implement the complete protocol",
                )
            descriptor = runtime.get_descriptor()
            if descriptor.curriculum_id != curriculum_id:
                _raise(
                    "invalid_runtime_component",
                    "curriculum_runtime.curriculum_id",
                    curriculum_id,
                    "Runtime mapping key and descriptor ID differ",
                )
            state = runtime.get_current_state()
            if (
                state.curriculum_id != curriculum_id
                or state.current_level_index != descriptor.initial_level_index
                or state.final_level_index != descriptor.final_level_index
                or state.current_level_id
                != descriptor.level_ids[state.current_level_index]
                or not runtime.validate_runtime_state()
            ):
                _raise(
                    "invalid_runtime_component",
                    "curriculum_runtime.state",
                    state.to_dict(),
                    "Initial Curriculum Runtime state is inconsistent with its descriptor",
                )
            descriptors.append(descriptor)

        descriptor_tuple = tuple(descriptors)
        capabilities = derive_trainer_capabilities(
            loss_components=registry.loss_components,
            sampling_components=registry.sampling_components,
            variable_roles=registry.variable_roles,
            optimization_stages=registry.optimization_stages,
            curricula=descriptor_tuple,
        )
        rebuilt = TrainingComponentRegistry(
            registry.loss_components,
            registry.sampling_components,
            registry.variable_roles,
            registry.optimization_stages,
            capabilities,
            descriptor_tuple,
        )
        best = dict(
            ((spec.get("training") or {}).get("extra_parameters") or {}).get(
                "best_checkpoint"
            )
            or {}
        )
        if best.get("enabled", False):
            incompatible = [
                item.curriculum_id
                for item in descriptor_tuple
                if item.metadata.get("probe_comparability_mode", "global_fixed_probe")
                != "global_fixed_probe"
                or not bool(
                    item.metadata.get(
                        "cross_level_best_checkpoint_supported", True
                    )
                )
            ]
            if incompatible:
                _raise(
                    "unsupported_problem_capability",
                    "training.extra_parameters.best_checkpoint",
                    incompatible,
                    "Cross-level best checkpointing requires a global fixed probe",
                )
        return runtimes, rebuilt

    def _build_training_component_registry(
        self,
        spec: dict[str, Any],
        problem: Any,
        phases: list[OptimizationPhase],
        sampler_runtime: Any,
        sampling_budget_plan: Any,
    ) -> Any:
        from forge.pipeline.g_training_evaluation.training.registry import (
            LossComponentDescriptor,
            OptimizationStageDescriptor,
            SamplingComponentDescriptor,
            TrainingComponentRegistry,
            VariableRoleDescriptor,
            derive_trainer_capabilities,
        )

        problem_spec = problem.get_spec()
        constraints = {
            str(item.constraint_id): item for item in problem_spec.constraints
        }
        runtime_terms, term_expansion_map = _expand_governing_loss_terms(
            spec["loss"], problem
        )
        loss_components = []
        preliminary: list[tuple[str, str, str | None, dict[str, Any]]] = []
        for term in runtime_terms:
            component_id = str(term["name"])
            parameters = dict(term.get("parameters") or {})
            source_id = str(parameters.get("component_name") or component_id)
            constraint = constraints.get(source_id) or constraints.get(component_id)
            metadata = dict(getattr(constraint, "metadata", {}) or {})
            if _is_governing_component_id(component_id):
                category = "equation_residual"
                sampler_id = "sample:interior"
                critical = True
                metadata.update(
                    {
                        "declared_loss_term": str(
                            term.get("_source_name") or component_id
                        ),
                        "governing_residual_name": str(
                            term.get("_governing_residual_name") or source_id
                        ),
                        "loss_group": "governing",
                    }
                )
            elif constraint is not None:
                category = sampling_constraint_category(
                    str(constraint.constraint_type)
                )
                role, runtime_category, sampler_id = constraint_sampling_identity(
                    constraint
                )
                critical = bool(constraint.required)
            elif component_id in INITIAL_ALIASES:
                category = "initial_constraint"
                matching_sources = [
                    item.sampler_id
                    for item in sampling_budget_plan.sources
                    if component_id in item.associated_loss_component_ids
                    and item.budget_pool == "initial"
                ]
                sampler_id = (
                    matching_sources[0]
                    if matching_sources
                    else "sample:initial_surface:initial"
                )
                runtime_category = "initial"
                role = "initial_surface"
                critical = True
            elif component_id in BOUNDARY_ALIASES:
                category = "boundary_constraint"
                matching_sources = [
                    item.sampler_id
                    for item in sampling_budget_plan.sources
                    if component_id in item.associated_loss_component_ids
                    and item.budget_pool == "boundary"
                ]
                sampler_id = (
                    matching_sources[0]
                    if matching_sources
                    else "sample:boundary:boundary"
                )
                runtime_category = "boundary"
                role = "boundary"
                critical = True
            elif component_id == "conservation":
                category = "conservation_constraint"
                sampler_id = "sample:custom_constraint_domain:conservation"
                runtime_category = "conservation"
                role = "custom_constraint_domain"
                critical = True
            elif component_id == "gradient_residual":
                category = "regularization"
                sampler_id = "sample:interior"
                critical = False
            else:
                category = "auxiliary_constraint"
                sampler_id = "sample:custom_constraint_domain:boundary"
                runtime_category = "boundary"
                role = "custom_constraint_domain"
                critical = False
            component_metadata = {
                **metadata,
                "critical_for_valid_solution": critical,
            }
            preliminary.append(
                (component_id, category, sampler_id, component_metadata)
            )

        stage_descriptors = []
        first_order_seen = False
        for index, phase in enumerate(phases):
            second_order = isinstance(phase.optimizer, torch.optim.LBFGS)
            first_order = not second_order
            recoverable = bool(second_order and first_order_seen)
            stage_descriptors.append(
                OptimizationStageDescriptor(
                    stage_id=phase.name,
                    optimizer_family=(
                        "second_order_line_search"
                        if second_order
                        else "first_order_gradient"
                    ),
                    order=index,
                    first_order=first_order,
                    second_order=second_order,
                    dynamic_objective_allowed=first_order,
                    recoverable=recoverable,
                    metadata={"runtime_optimizer": phase.optimizer_name},
                )
            )
            first_order_seen = first_order_seen or first_order
        stage_ids = tuple(item.stage_id for item in stage_descriptors)
        for component_id, category, sampler_id, metadata in preliminary:
            loss_components.append(
                LossComponentDescriptor(
                    component_id=component_id,
                    category=category,
                    sampler_id=sampler_id,
                    trainable=True,
                    dynamically_weightable=bool(
                        metadata.get("dynamically_weightable", True)
                    ),
                    validation_enabled=bool(
                        metadata.get("validation_enabled", True)
                    ),
                    active_stages=stage_ids,
                    metadata=metadata,
                )
            )

        grouped: dict[str, dict[str, Any]] = {}
        for source in sampling_budget_plan.sources:
            associated_ids: list[str] = []
            for component_id in source.associated_loss_component_ids:
                associated_ids.extend(
                    term_expansion_map.get(str(component_id), (str(component_id),))
                )
            grouped[source.sampler_id] = {
                "role": source.domain_role,
                "category": source.constraint_category,
                "runtime_key": source.budget_pool,
                "budget": source.point_budget,
                "ids": list(dict.fromkeys(associated_ids)),
                "minimum_points": source.minimum_points,
                "fixed_points": source.fixed_points,
                "allocation_weight": source.allocation_weight,
                "point_multiplier": source.point_multiplier,
            }
        sampling_components = tuple(
            SamplingComponentDescriptor(
                sampler_id=sampler_id,
                domain_role=value["role"],
                associated_loss_component_ids=tuple(value["ids"]),
                point_budget=value["budget"],
                mutable_during_first_order_stage=bool(
                    value["role"] == "interior"
                    and all(
                        callable(getattr(sampler_runtime, method, None))
                        for method in (
                            "get_training_points", "score_points",
                            "generate_candidate_points", "replace_subset",
                            "snapshot_state", "restore_state",
                        )
                    )
                ),
                fixed_during_second_order_stage=True,
                replacement_supported=bool(
                    value["role"] == "interior"
                    and all(
                        callable(getattr(sampler_runtime, method, None))
                        for method in (
                            "get_training_points", "score_points",
                            "generate_candidate_points", "replace_subset",
                            "snapshot_state", "restore_state",
                        )
                    )
                ),
                pointwise_scoring_supported=bool(value["role"] == "interior"),
                candidate_generation_supported=bool(value["role"] == "interior"),
                state_snapshot_supported=True,
                minimum_point_count=value["minimum_points"],
                maximum_point_count=(
                    value["budget"] if int(value["budget"] or 0) > 0 else None
                ),
                validation_probe_supported=True,
                metadata={
                    "constraint_category": value["category"],
                    "runtime_sample_key": value["runtime_key"],
                    "budget_pool": value["runtime_key"],
                    "fixed_points": value["fixed_points"],
                    "allocation_weight": value["allocation_weight"],
                    "point_multiplier": value["point_multiplier"],
                },
            )
            for sampler_id, value in grouped.items()
        )

        declared_roles = dict(problem_spec.domain.metadata.get("variable_roles") or {})
        spatial_dimension = int(problem_spec.metadata.get("spatial_dimension") or 0)
        time_dependent = bool(problem_spec.metadata.get("time_dependent", False))
        parameter_ids = {str(item.name) for item in problem_spec.unknown_parameters}
        variable_roles = []
        for index, variable_id in enumerate(problem_spec.input_variables):
            declaration = declared_roles.get(variable_id)
            declaration = {"role": declaration} if isinstance(declaration, str) else dict(declaration or {})
            if declaration:
                role = str(declaration.get("role", "auxiliary"))
                causal = bool(declaration.get("causal_ordered", role == "temporal"))
            elif variable_id in parameter_ids:
                role, causal = "parameter", False
            elif index < spatial_dimension:
                role, causal = "spatial", False
            elif time_dependent and index == spatial_dimension:
                role, causal = "temporal", True
            else:
                role, causal = "auxiliary", False
            bounds = problem_spec.domain.bounds.get(variable_id)
            variable_roles.append(
                VariableRoleDescriptor(
                    variable_id=str(variable_id),
                    role=role,
                    lower_bound=float(bounds[0]) if bounds else None,
                    upper_bound=float(bounds[1]) if bounds else None,
                    causal_ordered=causal,
                    metadata=declaration,
                )
            )
        loss_tuple = tuple(loss_components)
        variable_tuple = tuple(variable_roles)
        stage_tuple = tuple(stage_descriptors)
        capabilities = derive_trainer_capabilities(
            loss_components=loss_tuple,
            sampling_components=sampling_components,
            variable_roles=variable_tuple,
            optimization_stages=stage_tuple,
        )
        return TrainingComponentRegistry(
            loss_components=loss_tuple,
            sampling_components=sampling_components,
            variable_roles=variable_tuple,
            optimization_stages=stage_tuple,
            capabilities=capabilities,
        )

    def _apply_constraint_enforcement(
        self,
        model: nn.Module,
        spec: dict[str, Any],
        problem: Any,
    ) -> nn.Module:
        method = spec["method"]
        transform_id = spec["transform_id"]
        parameters = dict(spec.get("parameters") or {})
        if method == "soft_penalty":
            if transform_id != "none" or parameters:
                _raise(
                    "invalid_constraint_enforcement",
                    "constraint_enforcement",
                    spec,
                    "soft_penalty requires transform_id='none' and no parameters",
                )
            return model
        if method != "problem_hard":
            _raise(
                "unsupported_component",
                "constraint_enforcement.method",
                method,
                f"Unknown constraint enforcement method '{method}'",
            )
        factory = getattr(problem, "build_constraint_output_transform", None)
        if not callable(factory):
            _raise(
                "unsupported_problem_capability",
                "constraint_enforcement.transform_id",
                transform_id,
                "Problem does not expose a hard constraint transform factory",
            )
        try:
            transform = factory(transform_id, parameters)
        except Exception as exc:
            _raise(
                "hard_transform_build_failed",
                "constraint_enforcement.transform_id",
                transform_id,
                str(exc),
            )
        if not callable(transform):
            _raise(
                "invalid_runtime_component",
                "constraint_enforcement.transform_id",
                transform_id,
                "Problem hard transform factory must return a callable",
            )
        return ConstraintOutputWrapper(model, transform, transform_id=transform_id)

    def _build_network(self, spec: dict[str, Any], problem: Any) -> nn.Module:
        architecture = spec["architecture"]
        builder = NETWORK_BUILDERS.get(architecture)
        if builder is None:
            _raise("unsupported_component", "network.architecture", architecture, f"No registered builder exists for '{architecture}'")
        activation = _construct_component(ACTIVATIONS, spec["activation"], "network.activation")
        output_activation = _construct_component(ACTIVATIONS, spec["output_activation"], "network.output_activation")
        input_transform = self._build_input_transform(spec["input_transform"], problem)
        residual = spec["residual_connections"]
        residual_parameters = dict(residual.get("parameters") or {})
        residual_scale = float(residual_parameters.pop("scale", 1.0))
        if residual_parameters:
            _raise("unsupported_parameter", "network.residual_connections.parameters", sorted(residual_parameters), "Unsupported residual connection parameters")
        extra = dict(spec.get("extra_parameters") or {})
        bias = bool(extra.pop("bias", True))
        dropout_rate = float(extra.pop("dropout_rate", 0.0))
        input_scale = float(extra.pop("input_scale", 1.0))
        input_shift = float(extra.pop("input_shift", 0.0))
        output_scale = float(extra.pop("output_scale", 1.0))
        output_shift = float(extra.pop("output_shift", 0.0))
        activation = ActivationWithDropout(activation, dropout_rate)
        input_transform = AffineInputTransform(input_transform, input_scale, input_shift)
        physical_input_dim = len(problem.get_spec().input_variables)
        effective_input_dim = int(
            getattr(input_transform, "output_dim", physical_input_dim)
        )
        kwargs = {
            "input_dim": effective_input_dim,
            "output_dim": len(problem.get_spec().output_variables),
            "hidden_layers": list(spec["hidden_layers"]),
            "activation": activation,
            "output_activation": output_activation,
            "input_transform": input_transform,
            "residual_enabled": bool(residual.get("enabled", False)),
            "residual_scale": residual_scale,
            "bias": bias,
        }
        if architecture in {
            "fourier_mlp", "multiscale_fourier_mlp", "siren_mlp",
            "multiscale_mlp", "modified_mlp", "gated_mlp", "factorized_mlp",
            "laaf", "laaf_mlp", "gaaf", "gaaf_mlp",
        }:
            model = builder(extra_parameters=extra, **kwargs)
        else:
            allowed_extra = set(NETWORK_EXTRA_PARAMETER_CONTRACTS.get(architecture, [])) - {"bias"}
            unsupported_extra = sorted(set(extra) - allowed_extra)
            if unsupported_extra:
                _raise("unsupported_parameter", "network.extra_parameters", unsupported_extra, f"Unsupported {architecture} parameters: {unsupported_extra}")
            model = builder(extra_parameters=extra, **kwargs) if extra and allowed_extra else builder(**kwargs)
        if output_scale != 1.0 or output_shift != 0.0:
            model = OutputAffineWrapper(model, output_scale, output_shift)
        initializer = INITIALIZERS.get(spec["initialization"]["name"])
        if initializer is None:
            _raise("unsupported_component", "network.initialization.name", spec["initialization"]["name"], "Unknown initializer")
        raw_init_parameters = dict(INITIALIZER_PARAMETER_DEFAULTS.get(spec["initialization"]["name"], {}))
        raw_init_parameters.update(spec["initialization"].get("parameters") or {})
        if architecture == "siren_mlp":
            if spec["initialization"]["name"] != "siren":
                _raise(
                    "invalid_parameter",
                    "network.initialization.name",
                    spec["initialization"]["name"],
                    "siren_mlp requires the layer-aware siren initializer",
                )
            siren_model = (
                model.model if isinstance(model, OutputAffineWrapper) else model
            )
            if not isinstance(siren_model, SirenPINN):
                _raise(
                    "invalid_runtime_component",
                    "network.architecture",
                    architecture,
                    "siren_mlp did not construct a SirenPINN",
                )
            siren_model.reset_siren_parameters(
                float(raw_init_parameters.get("initialization_scale", 1.0))
            )
            return model
        if spec["initialization"]["name"] == "siren":
            _raise(
                "invalid_parameter",
                "network.initialization.name",
                "siren",
                "The siren initializer is only valid with siren_mlp",
            )
        init_parameters = _checked_kwargs(initializer, raw_init_parameters, "network.initialization.parameters", skip={"tensor"})
        for module in model.modules():
            if isinstance(module, nn.Linear):
                initializer(module.weight, **init_parameters)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        return model

    def _build_input_transform(self, spec: dict[str, Any], problem: Any) -> nn.Module:
        name = spec["name"]
        parameters = dict(spec.get("parameters") or {})
        if name == "none":
            if parameters:
                _raise("unsupported_parameter", "network.input_transform.parameters", sorted(parameters), "none input transform accepts no parameters")
            return nn.Identity()
        if name in {"normalize_bounds", "unit_bounds"}:
            if parameters:
                _raise("unsupported_parameter", "network.input_transform.parameters", sorted(parameters), f"{name} uses problem bounds and accepts no parameters")
            problem_spec = problem.get_spec()
            bounds = problem_spec.domain.bounds
            try:
                lower = torch.tensor([bounds[name][0] for name in problem_spec.input_variables], dtype=torch.float32)
                upper = torch.tensor([bounds[name][1] for name in problem_spec.input_variables], dtype=torch.float32)
            except Exception as exc:
                _raise("missing_problem_bounds", "network.input_transform", bounds, f"Cannot normalize inputs: {exc}")
            return BoundsTransform(lower, upper) if name == "normalize_bounds" else UnitBoundsTransform(lower, upper)
        if name in {
            "fourier_features",
            "multiscale_fourier_features",
            "periodic_positional_encoding",
        }:
            input_dim = len(problem.get_spec().input_variables)
            if name == "fourier_features":
                allowed = {
                    "frequency_count",
                    "frequency_scale",
                    "trainable_frequencies",
                    "include_raw_input",
                }
                unsupported = sorted(set(parameters) - allowed)
                if unsupported:
                    _raise(
                        "unsupported_parameter",
                        "network.input_transform.parameters",
                        unsupported,
                        f"Unsupported Fourier representation parameters: {unsupported}",
                    )
                return FourierFeatures(
                    input_dim=input_dim,
                    num_frequencies=int(parameters.get("frequency_count", 8)),
                    sigma=float(parameters.get("frequency_scale", 1.0)),
                    trainable=bool(parameters.get("trainable_frequencies", False)),
                    include_raw_input=bool(parameters.get("include_raw_input", False)),
                )
            allowed = {
                "frequency_count",
                "frequency_scale",
                "frequency_scales",
                "trainable_frequencies",
                "include_raw_input",
            }
            unsupported = sorted(set(parameters) - allowed)
            if unsupported:
                _raise(
                    "unsupported_parameter",
                    "network.input_transform.parameters",
                    unsupported,
                    f"Unsupported Fourier representation parameters: {unsupported}",
                )
            return MultiScaleFourierFeatures(
                input_dim=input_dim,
                frequency_count=int(parameters.get("frequency_count", 8)),
                frequency_scale=float(parameters.get("frequency_scale", 1.0)),
                frequency_scales=list(
                    parameters.get("frequency_scales")
                    or ([1.0] if name == "periodic_positional_encoding" else [1.0, 2.0, 4.0])
                ),
                trainable_frequencies=bool(
                    parameters.get("trainable_frequencies", False)
                ),
                include_raw_input=bool(
                    parameters.get(
                        "include_raw_input",
                        name == "periodic_positional_encoding",
                    )
                ),
            )
        _raise("unsupported_component", "network.input_transform.name", name, f"Unknown input transform '{name}'")

    def _build_sampler(
        self,
        spec: dict[str, Any],
        problem: Any,
        device: str,
        *,
        maximum_sampling_points: int | None = None,
        constraint_resample_interval: int = 50,
        legacy_wave_aggregate: bool = False,
    ) -> OpenSpecSampler:
        interior_parameter_contracts = {
            "uniform": {"n_points"},
            "latin_hypercube": {"n_points", "centered"},
            "sobol": {"n_points", "scramble", "seed", "skip"},
            "halton": {"n_points", "start_index"},
            "hammersley": {"n_points", "start_index"},
            "grid": {"n_points", "points_per_axis", "shuffle"},
        }
        for section in ("interior", "boundary", "initial"):
            parameters = dict(spec[section].get("parameters") or {})
            strategy = spec[section]["strategy"]
            if section == "interior":
                if strategy not in SAMPLERS:
                    _raise("unsupported_component", f"sampling.{section}.strategy", strategy, f"Unknown sampler '{strategy}'")
                allowed = interior_parameter_contracts.get(strategy, {"n_points"})
            else:
                if strategy not in CONSTRAINT_SAMPLERS:
                    _raise("unsupported_component", f"sampling.{section}.strategy", strategy, f"Constraint samplers support {sorted(CONSTRAINT_SAMPLERS)}")
                allowed = {"n_points"}
            unsupported = sorted(set(parameters) - allowed)
            if unsupported:
                _raise("unsupported_parameter", f"sampling.{section}.parameters", unsupported, f"Unsupported {strategy} sampler parameters: {unsupported}")
        adaptive = spec["adaptive_refinement"]
        if adaptive.get("enabled") and adaptive.get("strategy") not in ADAPTIVE_SAMPLERS:
            _raise("unsupported_component", "sampling.adaptive_refinement.strategy", adaptive.get("strategy"), "Unknown adaptive sampler")
        adaptive_parameters = dict(adaptive.get("parameters") or {})
        supported_adaptive = {
            "candidate_points",
            "add_points",
            "update_every",
            "max_points",
            "residual_exponent",
            "distribution_offset",
            "residual_component",
            "gradient_input_indices",
            "residual_weight",
            "gradient_weight",
            "hybrid_mode",
            "replace_existing",
        }
        unsupported_adaptive = sorted(set(adaptive_parameters) - supported_adaptive)
        if unsupported_adaptive:
            _raise("unsupported_parameter", "sampling.adaptive_refinement.parameters", unsupported_adaptive, f"Unsupported adaptive sampler parameters: {unsupported_adaptive}")
        return OpenSpecSampler(
            problem,
            spec,
            device,
            maximum_sampling_points=maximum_sampling_points,
            constraint_resample_interval=constraint_resample_interval,
            legacy_wave_aggregate=legacy_wave_aggregate,
        )

    def _build_loss(self, spec: dict[str, Any], problem: Any, sampler: OpenSpecSampler) -> OpenSpecLoss:
        weighting_name = spec["weighting_strategy"]["name"]
        aggregation_name = spec["aggregation"]["name"]
        if weighting_name not in WEIGHTING_STRATEGIES or aggregation_name not in AGGREGATIONS:
            _raise("unsupported_component", "loss", spec, "Only registered loss weighting and aggregation can be built")
        _validate_parameter_contract(
            spec["weighting_strategy"].get("parameters") or {},
            WEIGHTING_PARAMETER_CONTRACTS[weighting_name],
            "loss.weighting_strategy.parameters",
        )
        _validate_parameter_contract(
            spec["aggregation"].get("parameters") or {},
            AGGREGATION_PARAMETER_CONTRACTS[aggregation_name],
            "loss.aggregation.parameters",
        )
        _validate_positive_parameters(
            spec["weighting_strategy"].get("parameters") or {},
            {"epsilon", "temperature", "power", "min_weight", "max_weight"},
            "loss.weighting_strategy.parameters",
        )
        weighting_parameters = spec["weighting_strategy"].get("parameters") or {}
        if "alpha" in weighting_parameters and not 0.0 <= float(weighting_parameters["alpha"]) <= 1.0:
            _raise("invalid_parameter", "loss.weighting_strategy.parameters.alpha", weighting_parameters["alpha"], "alpha must be between 0 and 1")
        _validate_positive_parameters(
            spec["aggregation"].get("parameters") or {},
            {"temperature", "p", "epsilon"},
            "loss.aggregation.parameters",
        )
        problem_spec = problem.get_spec()
        constraints_by_id = {
            str(constraint.constraint_id): constraint
            for constraint in problem_spec.constraints
        }
        concrete_constraint_ids = {
            constraint_id
            for constraint_id, constraint in constraints_by_id.items()
            if not bool((constraint.metadata or {}).get("aggregate"))
        }
        supported_terms = set(LOSS_TERM_PARAMETER_CONTRACTS) | concrete_constraint_ids
        for index, term in enumerate(spec["terms"]):
            if term["name"] not in supported_terms:
                _raise("unsupported_component", f"loss.terms[{index}].name", term["name"], f"Unsupported loss term '{term['name']}'")
            loss_name = term["loss_function"]
            if loss_name not in LOSS_PARAMETER_CONTRACTS:
                _raise("unsupported_component", f"loss.terms[{index}].loss_function", loss_name, "Unknown loss function")
            term_parameter_contract = set(
                LOSS_TERM_PARAMETER_CONTRACTS.get(term["name"], set())
            )
            direct_constraint = constraints_by_id.get(str(term["name"]))
            if direct_constraint is not None:
                derivative_variable = (
                    direct_constraint.derivative_variable
                    or (direct_constraint.metadata or {}).get(
                        "derivative_variable"
                    )
                )
                derivative_order = (
                    direct_constraint.derivative_order
                    if direct_constraint.derivative_order is not None
                    else (direct_constraint.metadata or {}).get(
                        "derivative_order"
                    )
                )
                if derivative_variable is not None:
                    term_parameter_contract.add("derivative_variable")
                if derivative_order is not None:
                    term_parameter_contract.add("derivative_order")
            _validate_parameter_contract(
                term.get("parameters") or {},
                LOSS_PARAMETER_CONTRACTS[loss_name]
                | term_parameter_contract,
                f"loss.terms[{index}].parameters",
            )
            _validate_positive_loss_parameters(loss_name, term.get("parameters") or {}, index)
        return OpenSpecLoss(problem, sampler, spec)

    def _validate_training(self, spec: dict[str, Any]) -> None:
        if spec.get("batch_mode") != "full_batch":
            _raise("unsupported_component", "training.batch_mode", spec.get("batch_mode"), "The current trainer supports full_batch only")
        if spec.get("batch_size") is not None:
            _raise("unsupported_parameter", "training.batch_size", spec.get("batch_size"), "batch_size must be null in full_batch mode")
        extra = dict(spec.get("extra_parameters") or {})
        unknown_extra = sorted(
            set(extra)
            - {
                "adaptive_control",
                "component_control",
                "controller_coordinator",
                "physics_validation",
                "best_checkpoint",
            }
        )
        if unknown_extra:
            _raise("unsupported_parameter", "training.extra_parameters", unknown_extra, "Unsupported training parameters")
        adaptive_control = dict(extra.get("adaptive_control") or {})
        unknown_control = sorted(
            set(adaptive_control)
            - {
                "enabled", "freeze_lbfgs_objective", "lbfgs_stall_recovery",
                "rollback", "sampling_control", "curriculum_control",
            }
        )
        if unknown_control:
            _raise(
                "unsupported_parameter",
                "training.extra_parameters.adaptive_control",
                unknown_control,
                "Unsupported adaptive-control fields",
            )
        clipping = spec["gradient_clipping"]
        clipping_unknown = sorted(set(clipping.get("parameters") or {}) - {"max_norm", "norm_type"})
        if clipping_unknown:
            _raise("unsupported_parameter", "training.gradient_clipping.parameters", clipping_unknown, "Unsupported gradient clipping parameters")
        early = spec["early_stopping"]
        early_unknown = sorted(set(early.get("parameters") or {}) - {"threshold", "patience"})
        if early_unknown:
            _raise("unsupported_parameter", "training.early_stopping.parameters", early_unknown, "Unsupported early stopping parameters")
        time_strategy = spec.get("time_strategy") or {
            "name": "none",
            "parameters": {},
        }
        time_name = str(time_strategy.get("name") or "none")
        time_parameters = dict(time_strategy.get("parameters") or {})
        if time_name not in {"none", "time_window_curriculum"}:
            _raise(
                "unsupported_component",
                "training.time_strategy.name",
                time_name,
                "Only none and time_window_curriculum are supported",
            )
        if time_name == "none" and time_parameters:
            _raise(
                "unsupported_parameter",
                "training.time_strategy.parameters",
                sorted(time_parameters),
                "The none time strategy accepts no parameters",
            )
        if time_name == "time_window_curriculum":
            unknown = sorted(
                set(time_parameters)
                - {"window_fractions", "iterations_per_window"}
            )
            if unknown:
                _raise(
                    "unsupported_parameter",
                    "training.time_strategy.parameters",
                    unknown,
                    f"Unsupported time curriculum parameters: {unknown}",
                )
        pretraining = spec.get("initial_state_pretraining") or {
            "enabled": False,
            "parameters": {},
        }
        pretraining_parameters = dict(pretraining.get("parameters") or {})
        unknown_pretraining = sorted(set(pretraining_parameters) - {"iterations"})
        if unknown_pretraining:
            _raise(
                "unsupported_parameter",
                "training.initial_state_pretraining.parameters",
                unknown_pretraining,
                f"Unsupported initial pretraining parameters: {unknown_pretraining}",
            )

    def _build_optimization_phases(self, spec: dict[str, Any], model: nn.Module) -> list[OptimizationPhase]:
        phases: list[OptimizationPhase] = []
        for index, phase_spec in enumerate(spec["phases"]):
            name = phase_spec["optimizer"]
            optimizer_class = OPTIMIZERS.get(name)
            if optimizer_class is None:
                _raise("unsupported_component", f"optimization.phases[{index}].optimizer", name, f"Unknown optimizer '{name}'")
            parameters = dict(phase_spec.get("parameters") or {})
            if "learning_rate" in parameters:
                parameters["lr"] = parameters.pop("learning_rate")
            for tuple_parameter in ("betas", "etas", "step_sizes", "eps", "ns_coefficients"):
                if tuple_parameter in parameters and isinstance(parameters[tuple_parameter], (list, tuple)):
                    parameters[tuple_parameter] = tuple(parameters[tuple_parameter])
            if name == "lbfgs":
                # L-BFGS is executed in persistent chunks by OpenSpecTrainer.
                # These values are per-call limits, not the full phase budget.
                parameters.setdefault("max_iter", 20)
                parameters.setdefault("max_eval", 25)
            checked = _checked_kwargs(optimizer_class.__init__, parameters, f"optimization.phases[{index}].parameters", skip={"self", "params"})
            optimizer = optimizer_class(model.parameters(), **checked)
            scheduler = self._build_scheduler(phase_spec["scheduler"], optimizer, index, phase_spec["iterations"])
            phases.append(OptimizationPhase(f"phase_{index}_{name}", name, int(phase_spec["iterations"]), optimizer, scheduler, deepcopy(parameters)))
        return phases

    def _build_scheduler(self, spec: dict[str, Any], optimizer: torch.optim.Optimizer, index: int, iterations: int) -> object | None:
        name = spec["name"]
        scheduler_class = SCHEDULERS.get(name)
        if name not in SCHEDULERS:
            _raise("unsupported_component", f"optimization.phases[{index}].scheduler.name", name, f"Unknown scheduler '{name}'")
        parameters = dict(spec.get("parameters") or {})
        if scheduler_class is None:
            if parameters:
                _raise("unsupported_parameter", f"optimization.phases[{index}].scheduler.parameters", sorted(parameters), "none scheduler accepts no parameters")
            return None
        if scheduler_class is torch.optim.lr_scheduler.CosineAnnealingLR:
            parameters.setdefault("T_max", max(1, iterations))
        if scheduler_class is torch.optim.lr_scheduler.CosineAnnealingWarmRestarts:
            parameters.setdefault("T_0", max(1, iterations // 4))
            parameters.setdefault("T_mult", 1)
            parameters.setdefault("eta_min", 0.0)
        if scheduler_class is torch.optim.lr_scheduler.OneCycleLR:
            parameters.pop("epochs", None)
            parameters.pop("steps_per_epoch", None)
            parameters["total_steps"] = max(1, iterations)
            parameters.setdefault("max_lr", optimizer.param_groups[0]["lr"])
        if scheduler_class is torch.optim.lr_scheduler.LinearLR:
            parameters.setdefault("start_factor", 1.0)
            parameters.setdefault("end_factor", 0.1)
            parameters.setdefault("total_iters", max(1, iterations))
        if scheduler_class is torch.optim.lr_scheduler.PolynomialLR:
            parameters.setdefault("total_iters", max(1, iterations))
            parameters.setdefault("power", 1.0)
        if scheduler_class is torch.optim.lr_scheduler.ConstantLR:
            parameters.setdefault("factor", 0.5)
            parameters.setdefault("total_iters", max(1, iterations // 3))
        if scheduler_class is torch.optim.lr_scheduler.MultiStepLR:
            parameters.setdefault("milestones", sorted({max(1, iterations // 2), max(1, 3 * iterations // 4)}))
            parameters.setdefault("gamma", 0.1)
        if scheduler_class is torch.optim.lr_scheduler.CyclicLR:
            learning_rate = float(optimizer.param_groups[0]["lr"])
            parameters.setdefault("base_lr", learning_rate * 0.1)
            parameters.setdefault("max_lr", learning_rate)
            parameters.setdefault("cycle_momentum", False)
        checked = _checked_kwargs(scheduler_class.__init__, parameters, f"optimization.phases[{index}].scheduler.parameters", skip={"self", "optimizer"})
        return scheduler_class(optimizer, **checked)


def _construct_component(registry: dict[str, Any], spec: dict[str, Any], path: str) -> nn.Module:
    name = spec["name"]
    target = registry.get(name)
    if target is None:
        _raise("unsupported_component", f"{path}.name", name, f"No registered component exists for '{name}'")
    raw_parameters = dict(COMPONENT_PARAMETER_DEFAULTS.get(name, {}))
    raw_parameters.update(spec.get("parameters") or {})
    parameters = _checked_kwargs(target.__init__, raw_parameters, f"{path}.parameters", skip={"self"})
    return target(**parameters)


def _validate_parameter_contract(values: dict[str, Any], allowed: set[str], path: str) -> None:
    unsupported = sorted(set(values) - set(allowed))
    if unsupported:
        _raise("unsupported_parameter", path, unsupported, f"Unsupported parameters at {path}: {unsupported}")


def _validate_positive_parameters(values: dict[str, Any], names: set[str], path: str) -> None:
    invalid = {name: values[name] for name in names & set(values) if float(values[name]) <= 0.0}
    if invalid:
        _raise("invalid_parameter", path, invalid, f"Parameters must be positive at {path}: {sorted(invalid)}")


def _validate_positive_loss_parameters(loss_name: str, values: dict[str, Any], index: int) -> None:
    positive = {
        "rmse": {"epsilon"},
        "huber": {"delta"},
        "smooth_l1": {"beta"},
        "log_cosh": {"scale"},
        "cauchy": {"scale"},
        "charbonnier": {"epsilon", "alpha"},
        "pseudo_huber": {"delta"},
        "lp": {"p", "epsilon"},
        "tukey": {"c"},
        "geman_mcclure": {"scale"},
        "log_l2": {"epsilon"},
    }.get(loss_name, set())
    path = f"loss.terms[{index}].parameters"
    _validate_positive_parameters(values, positive, path)
    if loss_name == "quantile" and "quantile" in values:
        quantile = float(values["quantile"])
        if not 0.0 < quantile < 1.0:
            _raise("invalid_parameter", f"{path}.quantile", quantile, "quantile must be strictly between 0 and 1")


def _parameter_contract(callable_object: Callable[..., Any], skip: set[str] | None = None) -> list[str]:
    parameters = inspect.signature(callable_object).parameters
    return sorted(
        name
        for name, parameter in parameters.items()
        if name not in (skip or set())
        and parameter.kind in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
    )


def _required_parameter_contract(callable_object: Callable[..., Any], skip: set[str] | None = None) -> list[str]:
    parameters = inspect.signature(callable_object).parameters
    return sorted(
        name
        for name, parameter in parameters.items()
        if name not in (skip or set())
        and parameter.default is inspect.Parameter.empty
        and parameter.kind in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
    )


def _radical_inverse(index: int, base: int) -> float:
    value = 0.0
    factor = 1.0 / base
    while index > 0:
        index, remainder = divmod(index, base)
        value += remainder * factor
        factor /= base
    return value


def _first_primes(count: int) -> list[int]:
    primes: list[int] = []
    candidate = 2
    while len(primes) < count:
        if all(candidate % prime for prime in primes if prime * prime <= candidate):
            primes.append(candidate)
        candidate += 1
    return primes


def _checked_kwargs(callable_object: Callable[..., Any], values: dict[str, Any], path: str, skip: set[str] | None = None) -> dict[str, Any]:
    values = dict(values or {})
    signature = inspect.signature(callable_object)
    parameters = signature.parameters
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return values
    allowed = {
        name for name, parameter in parameters.items()
        if name not in (skip or set()) and parameter.kind in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
    }
    unsupported = sorted(set(values) - allowed)
    if unsupported:
        _raise("unsupported_parameter", path, unsupported, f"Unsupported parameters at {path}: {unsupported}")
    return values


def _raise(error_type: str, path: str, value: Any, message: str):
    raise BuildError(BuildReport("build_failed", error_type, path, value, message))

# ---------------------------------------------------------------------------
# One-call facade
# ---------------------------------------------------------------------------
class AlgorithmSpecValidationError(ValueError):
    """Raised when validate_and_build receives an invalid AlgorithmSpec."""

    def __init__(self, report: ValidationReport) -> None:
        self.report = report
        if report.errors:
            first = report.errors[0]
            message = f"{first.code} at {first.path}: {first.message}"
        else:
            message = "AlgorithmSpec validation failed"
        super().__init__(message)


@dataclass
class ValidatedBuiltPINN:
    """Combined result returned by the all-in-one validate-and-build API."""

    validation_report: ValidationReport
    built_pinn: BuiltPINN

    @property
    def model(self) -> nn.Module:
        return self.built_pinn.model

    @property
    def sampler(self) -> OpenSpecSampler:
        return self.built_pinn.sampler

    @property
    def loss_function(self) -> OpenSpecLoss:
        return self.built_pinn.loss_function

    @property
    def optimization_phases(self) -> list[OptimizationPhase]:
        return self.built_pinn.optimization_phases

    @property
    def normalized_spec(self) -> dict[str, Any]:
        return self.built_pinn.normalized_spec


class OpenAlgorithmSpecRuntime:
    """Single entry point for capabilities, validation, and PINN construction."""

    def __init__(self) -> None:
        self.validator = AlgorithmSpecValidator()
        self.builder = PINNBuilder()

    def capabilities(self) -> dict[str, Any]:
        return builder_capabilities()

    def option_registry(self) -> dict[str, Any]:
        return algorithm_spec_option_registry()

    def validate(
        self,
        spec: dict[str, Any],
        problem: Any,
        experiment_config: dict[str, Any] | None = None,
    ) -> ValidationReport:
        return self.validator.validate(
            spec,
            problem,
            dict(experiment_config or {}),
            self.capabilities(),
        )

    def build(
        self,
        normalized_spec: dict[str, Any],
        problem: Any,
        runtime_config: dict[str, Any] | None = None,
    ) -> BuiltPINN:
        """Build an already validated and normalized AlgorithmSpec."""

        return self.builder.build(
            normalized_spec,
            problem,
            dict(runtime_config or {}),
        )

    def validate_and_build(
        self,
        raw_spec: dict[str, Any],
        problem: Any,
        *,
        experiment_config: dict[str, Any] | None = None,
        runtime_config: dict[str, Any] | None = None,
    ) -> ValidatedBuiltPINN:
        """Normalize, validate, and build with one call."""

        report = self.validate(raw_spec, problem, experiment_config)
        if not report.valid or report.normalized_spec is None:
            raise AlgorithmSpecValidationError(report)
        built = self.build(report.normalized_spec, problem, runtime_config)
        return ValidatedBuiltPINN(report, built)


def get_algorithm_spec_capabilities() -> dict[str, Any]:
    """Return executable components and their expanded numeric schemas."""

    return builder_capabilities()


def validate_and_build_algorithm_spec(
    raw_spec: dict[str, Any],
    problem: Any,
    *,
    experiment_config: dict[str, Any] | None = None,
    runtime_config: dict[str, Any] | None = None,
) -> ValidatedBuiltPINN:
    """Convenience function requiring only this module to be imported."""

    return OpenAlgorithmSpecRuntime().validate_and_build(
        raw_spec,
        problem,
        experiment_config=experiment_config,
        runtime_config=runtime_config,
    )


__all__ = [
    "AlgorithmSpecValidationError",
    "AlgorithmSpecValidator",
    "BuildError",
    "BuildReport",
    "BuiltPINN",
    "OpenAlgorithmSpecRuntime",
    "OpenSpecLoss",
    "OpenSpecSampler",
    "OptimizationPhase",
    "PINNBuilder",
    "ValidatedBuiltPINN",
    "ValidationIssue",
    "ValidationReport",
    "algorithm_spec_option_registry",
    "apply_numeric_max_expansion",
    "apply_parameter_expansion",
    "builder_capabilities",
    "expanded_algorithm_spec_option_registry",
    "get_algorithm_spec_capabilities",
    "legacy_wave_split_weights",
    "normalize_algorithm_spec",
    "validate_algorithm_spec",
    "validate_and_build_algorithm_spec",
]
