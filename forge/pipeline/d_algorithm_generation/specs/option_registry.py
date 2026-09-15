"""Field-level option registry for dynamically composed AlgorithmSpec objects.

The registry describes independent choices.  It deliberately does not enumerate
complete PINN algorithms or Cartesian products of choices.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping


def _parameter(
    value_type: str,
    default: Any,
    *,
    minimum: float | int | None = None,
    maximum: float | int | None = None,
    exclusive_minimum: float | int | None = None,
    enum: list[Any] | None = None,
    items: dict[str, Any] | None = None,
    min_items: int | None = None,
    max_items: int | None = None,
    x_search: dict[str, Any] | None = None,
    fidelity_scaling: str | None = None,
    hard_safety: bool = False,
    llm_visible: bool = True,
    description: str = "",
) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": value_type, "default": deepcopy(default)}
    if minimum is not None:
        schema["minimum"] = minimum
    if maximum is not None:
        schema["maximum"] = maximum
    if exclusive_minimum is not None:
        schema["exclusiveMinimum"] = exclusive_minimum
    if enum is not None:
        schema["enum"] = deepcopy(enum)
    if items is not None:
        schema["items"] = deepcopy(items)
    if min_items is not None:
        schema["minItems"] = min_items
    if max_items is not None:
        schema["maxItems"] = max_items
    if x_search is not None:
        schema["x-search"] = deepcopy(x_search)
    if fidelity_scaling is not None:
        schema["x-fidelity-scaling"] = str(fidelity_scaling)
        schema["x-hard-safety"] = bool(hard_safety)
    elif hard_safety:
        schema["x-hard-safety"] = True
    if not llm_visible:
        schema["x-llm-visible"] = False
    if description:
        schema["description"] = description
    return schema


def _option(
    parameters: Mapping[str, dict[str, Any]] | None = None,
    *,
    description: str = "",
    compatibility: list[dict[str, Any]] | None = None,
    alias_of: str | None = None,
) -> dict[str, Any]:
    properties = deepcopy(dict(parameters or {}))
    result: dict[str, Any] = {
        "description": description,
        "parameter_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": properties,
        },
        "default_parameters": {
            name: deepcopy(schema.get("default"))
            for name, schema in properties.items()
            if "default" in schema
        },
        "compatibility": deepcopy(compatibility or []),
    }
    if alias_of:
        result["alias_of"] = alias_of
    return result


BOOL_TRUE = _parameter("boolean", True)
BOOL_FALSE = _parameter("boolean", False)
PINNSAGENT_LEARNING_RATES = [1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1]
PINNSAGENT_WIDTHS = list(range(8, 257, 4))
PINNSAGENT_DEPTHS = list(range(3, 11))
PINNSAGENT_POINT_COUNTS = list(range(100, 9601, 500))
POSITIVE_SCALE = _parameter("number", 1.0, exclusive_minimum=0.0, maximum=1.0e4)
INPLACE_FALSE = _parameter(
    "boolean",
    False,
    enum=[False],
    description="In-place activations are disabled to preserve higher-order coordinate autograd.",
)
N_POINTS = _parameter("integer", 1024, minimum=0)


NETWORK_OPTIONS = {
    "mlp": _option({"bias": BOOL_TRUE}, description="Fully connected coordinate MLP."),
    "fnn": _option({"bias": BOOL_TRUE}, description="PINNsAgent/DeepXDE name for a fully connected MLP.", alias_of="mlp"),
    "residual_mlp": _option({"bias": BOOL_TRUE}, description="MLP with projected residual connections."),
    "resnet": _option({"bias": BOOL_TRUE}, description="Alias of residual_mlp.", alias_of="residual_mlp"),
    "fourier_mlp": _option(
        {
            "bias": BOOL_TRUE,
            "num_frequencies": _parameter("integer", 8, minimum=1, maximum=256),
            "sigma": _parameter("number", 1.0, exclusive_minimum=0.0, maximum=1000.0),
            "include_raw_input": BOOL_FALSE,
            "trainable": BOOL_FALSE,
            "frequency_count": _parameter("integer", 8, minimum=1, maximum=256),
            "frequency_scale": _parameter("number", 1.0, exclusive_minimum=0.0, maximum=1000.0),
            "trainable_frequencies": BOOL_FALSE,
        },
        description="Fourier coordinate encoding followed by an MLP.",
    ),
    "multiscale_fourier_mlp": _option(
        {
            "bias": BOOL_TRUE,
            "frequency_count": _parameter("integer", 8, minimum=1, maximum=256),
            "frequency_scale": _parameter("number", 1.0, exclusive_minimum=0.0, maximum=1000.0),
            "frequency_scales": _parameter(
                "array",
                [1.0, 2.0, 4.0],
                items={"type": "number", "exclusiveMinimum": 0.0, "maximum": 1.0e4},
                min_items=1,
                max_items=32,
            ),
            "trainable_frequencies": BOOL_FALSE,
            "include_raw_input": BOOL_FALSE,
        },
        description="Multiple Fourier frequency bands followed by a tanh MLP.",
    ),
    "siren_mlp": _option(
        {
            "bias": BOOL_TRUE,
            "omega_0": _parameter("number", 30.0, exclusive_minimum=0.0, maximum=1000.0),
            "hidden_omega_0": _parameter("number", 30.0, exclusive_minimum=0.0, maximum=1000.0),
            "initialization_scale": _parameter("number", 1.0, exclusive_minimum=0.0, maximum=100.0),
        },
        description="SIREN network with layer-aware sine frequencies and initialization.",
    ),
    "multiscale_mlp": _option(
        {
            "bias": BOOL_TRUE,
            "scales": _parameter(
                "array",
                [1.0, 2.0, 4.0],
                items={"type": "number", "exclusiveMinimum": 0.0, "maximum": 1.0e4},
                min_items=1,
                max_items=32,
            ),
        },
        description="Concatenated multi-scale coordinate features followed by an MLP.",
    ),
    "modified_mlp": _option(
        {"bias": BOOL_TRUE, "gate_temperature": POSITIVE_SCALE},
        description="Modified/gated PINN MLP with two coordinate encoders.",
    ),
    "gated_mlp": _option(
        {"bias": BOOL_TRUE, "gate_temperature": POSITIVE_SCALE},
        description="Alias of modified_mlp.",
        alias_of="modified_mlp",
    ),
    "factorized_mlp": _option(
        {"bias": BOOL_TRUE, "rank": _parameter("integer", 16, minimum=1, maximum=2048)},
        description="Low-rank factorized linear layers.",
    ),
    "laaf_mlp": _option(
        {
            "bias": BOOL_TRUE,
            "activation_scale": _parameter("number", 10.0, exclusive_minimum=0.0, maximum=1000.0),
            "initial_slope": _parameter("number", 1.0, exclusive_minimum=0.0, maximum=100.0),
        },
        description="Layer-wise locally adaptive activation MLP.",
    ),
    "laaf": _option(
        {
            "bias": BOOL_TRUE,
            "activation_scale": _parameter("number", 10.0, exclusive_minimum=0.0, maximum=1000.0),
            "initial_slope": _parameter("number", 1.0, exclusive_minimum=0.0, maximum=100.0),
        },
        description="Alias of laaf_mlp.",
        alias_of="laaf_mlp",
    ),
    "gaaf_mlp": _option(
        {
            "bias": BOOL_TRUE,
            "activation_scale": _parameter("number", 10.0, exclusive_minimum=0.0, maximum=1000.0),
            "initial_slope": _parameter("number", 1.0, exclusive_minimum=0.0, maximum=100.0),
        },
        description="Globally adaptive activation MLP.",
    ),
    "gaaf": _option(
        {
            "bias": BOOL_TRUE,
            "activation_scale": _parameter("number", 10.0, exclusive_minimum=0.0, maximum=1000.0),
            "initial_slope": _parameter("number", 1.0, exclusive_minimum=0.0, maximum=100.0),
        },
        description="Alias of gaaf_mlp.",
        alias_of="gaaf_mlp",
    ),
    "parallel_fnn": _option({"bias": BOOL_TRUE}, description="One coordinate-network branch per PDE output."),
    "pfnn": _option({"bias": BOOL_TRUE}, description="Alias of parallel_fnn.", alias_of="parallel_fnn"),
}


ACTIVATION_OPTIONS = {
    name: _option(description=f"{name} activation.")
    for name in [
        "identity", "tanh", "silu", "swish", "gelu", "relu", "sigmoid", "mish",
        "softsign", "tanhshrink", "selu",
    ]
}
ACTIVATION_OPTIONS.update(
    {
        "sin": _option({"omega": _parameter("number", 1.0, exclusive_minimum=0.0, maximum=1000.0)}, description="Sine activation."),
        "sine": _option({"omega": _parameter("number", 1.0, exclusive_minimum=0.0, maximum=1000.0)}, description="Alias of sin.", alias_of="sin"),
        "gaussian": _option({"beta": _parameter("number", 1.0, exclusive_minimum=0.0, maximum=1000.0)}),
        "laaf": _option(
            {
                "initial_slope": _parameter("number", 1.0, exclusive_minimum=0.0, maximum=100.0),
                "activation_scale": _parameter("number", 10.0, exclusive_minimum=0.0, maximum=1000.0),
            },
            description="Trainable adaptive-slope tanh activation; use network.architecture=laaf_mlp for layer-wise local slopes.",
        ),
        "leaky_relu": _option({"negative_slope": _parameter("number", 0.01, minimum=0.0, maximum=1.0), "inplace": INPLACE_FALSE}),
        "elu": _option({"alpha": POSITIVE_SCALE, "inplace": INPLACE_FALSE}),
        "celu": _option({"alpha": POSITIVE_SCALE, "inplace": INPLACE_FALSE}),
        "softplus": _option({"beta": POSITIVE_SCALE, "threshold": _parameter("number", 20.0, exclusive_minimum=0.0, maximum=1000.0)}),
        "hardtanh": _option(
            {
                "min_val": _parameter("number", -1.0, minimum=-1.0e6, maximum=1.0e6),
                "max_val": _parameter("number", 1.0, minimum=-1.0e6, maximum=1.0e6),
                "inplace": INPLACE_FALSE,
            },
            compatibility=[{"rule": "parameter_order", "lower": "min_val", "upper": "max_val"}],
        ),
        "prelu": _option(
            {
                "num_parameters": _parameter("integer", 1, minimum=1, maximum=4096),
                "init": _parameter("number", 0.25, minimum=-100.0, maximum=100.0),
            }
        ),
    }
)


BASE_SAMPLER_OPTIONS = {
    name: _option({"n_points": N_POINTS}, description=f"{name} coordinate sampler.")
    for name in ["uniform", "latin_hypercube", "sobol", "halton", "hammersley", "grid"]
}
CONSTRAINT_SAMPLER_OPTIONS = {
    "uniform": _option({"n_points": N_POINTS}, description="Problem-owned boundary/initial constraint sampling."),
}
ADAPTIVE_COMMON = {
    "candidate_points": _parameter("integer", 1024, minimum=1),
    "add_points": _parameter("integer", 64, minimum=1),
    "max_points": _parameter("integer", 0, minimum=0),
    "update_every": _parameter("integer", 100, minimum=1, maximum=10_000_000),
}
RESIDUAL_COMPATIBILITY = [{"rule": "problem_capability", "name": "governing_residual"}]
ADAPTIVE_SAMPLER_OPTIONS = {
    "rar": _option(ADAPTIVE_COMMON, description="Residual-based adaptive refinement using top residual points.", compatibility=RESIDUAL_COMPATIBILITY),
    "residual_adaptive": _option(ADAPTIVE_COMMON, description="Alias of RAR.", compatibility=RESIDUAL_COMPATIBILITY, alias_of="rar"),
    "rad": _option(
        {
            **ADAPTIVE_COMMON,
            "residual_exponent": _parameter("number", 2.0, minimum=0.0, maximum=16.0),
            "distribution_offset": _parameter("number", 1.0, minimum=0.0, maximum=1.0e6),
        },
        description="Residual-based adaptive distribution that refreshes the adaptive pool.",
        compatibility=RESIDUAL_COMPATIBILITY,
    ),
    "rar_d": _option(
        {
            **ADAPTIVE_COMMON,
            "residual_exponent": _parameter("number", 2.0, minimum=0.0, maximum=16.0),
            "distribution_offset": _parameter("number", 1.0, minimum=0.0, maximum=1.0e6),
        },
        description="Distribution-based residual adaptive refinement that accumulates selected points.",
        compatibility=RESIDUAL_COMPATIBILITY,
    ),
    "gradient_adaptive": _option(ADAPTIVE_COMMON, compatibility=RESIDUAL_COMPATIBILITY),
    "hybrid_adaptive": _option(ADAPTIVE_COMMON, compatibility=RESIDUAL_COMPATIBILITY),
}


def _optimizer_common(lr: float = 1.0e-3) -> dict[str, dict[str, Any]]:
    return {
        "learning_rate": _parameter(
            "number",
            lr,
            exclusive_minimum=0.0,
            maximum=100.0,
            x_search={"scale": "log", "recommended_values": PINNSAGENT_LEARNING_RATES},
        ),
        "weight_decay": _parameter("number", 0.0, minimum=0.0, maximum=100.0),
    }


BETAS = _parameter(
    "array",
    [0.9, 0.999],
    items={"type": "number", "minimum": 0.0, "exclusiveMaximum": 1.0},
    min_items=2,
    max_items=2,
)
OPTIMIZER_OPTIONS = {
    "adam": _option({**_optimizer_common(), "betas": BETAS, "eps": _parameter("number", 1.0e-8, exclusive_minimum=0.0, maximum=1.0)}),
    "adamw": _option({**_optimizer_common(), "betas": BETAS, "eps": _parameter("number", 1.0e-8, exclusive_minimum=0.0, maximum=1.0)}),
    "radam": _option({**_optimizer_common(), "betas": BETAS, "eps": _parameter("number", 1.0e-8, exclusive_minimum=0.0, maximum=1.0)}),
    "nadam": _option({**_optimizer_common(), "betas": BETAS, "eps": _parameter("number", 1.0e-8, exclusive_minimum=0.0, maximum=1.0), "momentum_decay": _parameter("number", 0.004, minimum=0.0, maximum=1.0)}),
    "adamax": _option({**_optimizer_common(0.002), "betas": BETAS, "eps": _parameter("number", 1.0e-8, exclusive_minimum=0.0, maximum=1.0)}),
    "multiadam": _option(
        {
            **_optimizer_common(),
            "betas": _parameter("array", [0.99, 0.99], items={"type": "number", "minimum": 0.0, "exclusiveMaximum": 1.0}, min_items=2, max_items=2),
            "eps": _parameter("number", 1.0e-8, exclusive_minimum=0.0, maximum=1.0),
            "group_weights": {"type": "array", "items": {"type": "number", "minimum": 0.0}, "minItems": 2, "maxItems": 128},
            "normalize_updates": BOOL_FALSE,
            "grouping": _parameter(
                "string",
                "pde_vs_constraints",
                enum=["pde_vs_constraints", "dirichlet_vs_non_dirichlet"],
            ),
        },
        compatibility=[
            {"rule": "minimum_loss_terms", "value": 2},
            {"rule": "field_equals", "path": "training.gradient_clipping.enabled", "value": False},
        ],
    ),
    "sgd": _option({**_optimizer_common(0.01), "momentum": _parameter("number", 0.0, minimum=0.0, maximum=1.0), "dampening": _parameter("number", 0.0, minimum=0.0, maximum=1.0), "nesterov": BOOL_FALSE}),
    "rmsprop": _option({**_optimizer_common(0.01), "alpha": _parameter("number", 0.99, minimum=0.0, maximum=1.0), "eps": _parameter("number", 1.0e-8, exclusive_minimum=0.0, maximum=1.0), "momentum": _parameter("number", 0.0, minimum=0.0, maximum=1.0), "centered": BOOL_FALSE}),
    "adagrad": _option({**_optimizer_common(0.01), "lr_decay": _parameter("number", 0.0, minimum=0.0, maximum=100.0), "initial_accumulator_value": _parameter("number", 0.0, minimum=0.0, maximum=1.0e6), "eps": _parameter("number", 1.0e-10, exclusive_minimum=0.0, maximum=1.0)}),
    "adadelta": _option({**_optimizer_common(1.0), "rho": _parameter("number", 0.9, minimum=0.0, maximum=1.0), "eps": _parameter("number", 1.0e-6, exclusive_minimum=0.0, maximum=1.0)}),
    "asgd": _option({**_optimizer_common(0.01), "lambd": _parameter("number", 1.0e-4, minimum=0.0, maximum=100.0), "alpha": _parameter("number", 0.75, minimum=0.0, maximum=10.0), "t0": _parameter("number", 1.0e6, minimum=0.0, maximum=1.0e12)}),
    "rprop": _option({"learning_rate": _parameter("number", 0.01, exclusive_minimum=0.0, maximum=100.0), "etas": _parameter("array", [0.5, 1.2], items={"type": "number", "exclusiveMinimum": 0.0}, min_items=2, max_items=2), "step_sizes": _parameter("array", [1.0e-6, 50.0], items={"type": "number", "exclusiveMinimum": 0.0}, min_items=2, max_items=2)}),
    "lbfgs": _option(
        {
            "learning_rate": _parameter(
                "number",
                1.0,
                exclusive_minimum=0.0,
                maximum=100.0,
                x_search={"scale": "log", "recommended_values": PINNSAGENT_LEARNING_RATES},
            ),
            "max_iter": _parameter("integer", 20, minimum=1, maximum=1_000_000),
            "max_eval": _parameter("integer", 25, minimum=1, maximum=2_000_000),
            "tolerance_grad": _parameter("number", 1.0e-7, exclusive_minimum=0.0, maximum=1.0),
            "tolerance_change": _parameter("number", 1.0e-9, exclusive_minimum=0.0, maximum=1.0),
            "history_size": _parameter("integer", 100, minimum=1, maximum=10_000),
            "line_search_fn": _parameter("string", "strong_wolfe", enum=["strong_wolfe"]),
        },
        compatibility=[
            {"rule": "field_equals", "path": "training.batch_mode", "value": "full_batch"},
            {"rule": "optimizer_phase_position", "value": "last"},
        ],
    ),
}


INITIALIZER_OPTIONS = {
    "xavier_normal": _option({"gain": _parameter("number", 1.0, exclusive_minimum=0.0, maximum=100.0)}),
    "xavier_uniform": _option({"gain": _parameter("number", 1.0, exclusive_minimum=0.0, maximum=100.0)}),
    "glorot_normal": _option(
        {"gain": _parameter("number", 1.0, exclusive_minimum=0.0, maximum=100.0)},
        description="PINNacle/DeepXDE name for Xavier normal initialization.",
        alias_of="xavier_normal",
    ),
    "glorot_uniform": _option(
        {"gain": _parameter("number", 1.0, exclusive_minimum=0.0, maximum=100.0)},
        description="PINNacle/DeepXDE name for Xavier uniform initialization.",
        alias_of="xavier_uniform",
    ),
    "kaiming_normal": _option({"a": _parameter("number", 0.0, minimum=0.0, maximum=100.0), "mode": _parameter("string", "fan_in", enum=["fan_in", "fan_out"]), "nonlinearity": _parameter("string", "leaky_relu", enum=["linear", "conv1d", "conv2d", "conv3d", "conv_transpose1d", "conv_transpose2d", "conv_transpose3d", "sigmoid", "tanh", "relu", "leaky_relu", "selu"])}),
    "kaiming_uniform": _option({"a": _parameter("number", 0.0, minimum=0.0, maximum=100.0), "mode": _parameter("string", "fan_in", enum=["fan_in", "fan_out"]), "nonlinearity": _parameter("string", "leaky_relu", enum=["linear", "conv1d", "conv2d", "conv3d", "conv_transpose1d", "conv_transpose2d", "conv_transpose3d", "sigmoid", "tanh", "relu", "leaky_relu", "selu"])}),
    "he_normal": _option(
        {"a": _parameter("number", 0.0, minimum=0.0, maximum=100.0), "mode": _parameter("string", "fan_in", enum=["fan_in", "fan_out"]), "nonlinearity": _parameter("string", "leaky_relu", enum=["linear", "conv1d", "conv2d", "conv3d", "conv_transpose1d", "conv_transpose2d", "conv_transpose3d", "sigmoid", "tanh", "relu", "leaky_relu", "selu"])},
        description="PINNsAgent/DeepXDE name for Kaiming normal initialization.",
        alias_of="kaiming_normal",
    ),
    "he_uniform": _option(
        {"a": _parameter("number", 0.0, minimum=0.0, maximum=100.0), "mode": _parameter("string", "fan_in", enum=["fan_in", "fan_out"]), "nonlinearity": _parameter("string", "leaky_relu", enum=["linear", "conv1d", "conv2d", "conv3d", "conv_transpose1d", "conv_transpose2d", "conv_transpose3d", "sigmoid", "tanh", "relu", "leaky_relu", "selu"])},
        description="PINNsAgent/DeepXDE name for Kaiming uniform initialization.",
        alias_of="kaiming_uniform",
    ),
    "orthogonal": _option({"gain": _parameter("number", 1.0, exclusive_minimum=0.0, maximum=100.0)}),
    "normal": _option({"mean": _parameter("number", 0.0, minimum=-1.0e6, maximum=1.0e6), "std": _parameter("number", 1.0, exclusive_minimum=0.0, maximum=1.0e6)}),
    "uniform": _option({"a": _parameter("number", 0.0, minimum=-1.0e6, maximum=1.0e6), "b": _parameter("number", 1.0, minimum=-1.0e6, maximum=1.0e6)}, compatibility=[{"rule": "parameter_order", "lower": "a", "upper": "b"}]),
    "trunc_normal": _option({"mean": _parameter("number", 0.0, minimum=-1.0e6, maximum=1.0e6), "std": _parameter("number", 1.0, exclusive_minimum=0.0, maximum=1.0e6), "a": _parameter("number", -2.0, minimum=-1.0e6, maximum=1.0e6), "b": _parameter("number", 2.0, minimum=-1.0e6, maximum=1.0e6)}, compatibility=[{"rule": "parameter_order", "lower": "a", "upper": "b"}]),
    "sparse": _option({"sparsity": _parameter("number", 0.1, minimum=0.0, maximum=1.0), "std": _parameter("number", 0.01, exclusive_minimum=0.0, maximum=1.0e6)}),
    "siren": _option(
        {
            "initialization_scale": _parameter(
                "number", 1.0, exclusive_minimum=0.0, maximum=100.0
            )
        },
        description="Layer-aware SIREN initialization; valid with siren_mlp.",
    ),
    "zeros": _option(),
}


LOSS_OPTIONS = {
    "mse": _option(),
    "mae": _option(),
    "rmse": _option({"epsilon": _parameter("number", 1.0e-12, exclusive_minimum=0.0, maximum=1.0)}),
    "huber": _option({"delta": POSITIVE_SCALE}),
    "smooth_l1": _option({"beta": POSITIVE_SCALE}),
    "log_cosh": _option({"scale": POSITIVE_SCALE}),
    "cauchy": _option({"scale": POSITIVE_SCALE}),
    "charbonnier": _option({"epsilon": _parameter("number", 1.0e-3, exclusive_minimum=0.0, maximum=1.0), "alpha": _parameter("number", 0.5, exclusive_minimum=0.0, maximum=1.0)}),
    "pseudo_huber": _option({"delta": POSITIVE_SCALE}),
}


LOSS_TERM_OPTIONS = {
    "pde_residual": _option(
        description="Aggregate governing-equation residual loss."
    ),
    "governing_residual": _option(
        description="Alias-style governing residual selection."
    ),
    "pde_component": _option(
        {
            "component_name": _parameter(
                "string",
                "",
                description="Select a governing-law ID supplied by problem_spec.governing_laws.",
            ),
            "component_index": _parameter(
                "integer",
                0,
                minimum=0,
                description="Select a governing residual by zero-based component index.",
            ),
        },
        description="A named or indexed governing-equation residual component. Supply exactly one selector."
    ),
    "boundary_condition": _option(
        description="Legacy aggregate boundary-condition loss."
    ),
    "initial_condition": _option(
        description="Legacy aggregate initial-condition loss."
    ),
    "spatial_boundary": _option(
        description="Independent Wave1D spatial-boundary constraint loss."
    ),
    "initial_displacement": _option(
        description="Independent Wave1D initial-displacement constraint loss."
    ),
    "initial_velocity": _option(
        {
            "derivative_variable": _parameter(
                "string", "t", enum=["t"]
            ),
            "derivative_order": _parameter(
                "integer", 1, enum=[1]
            ),
        },
        description="Independent Wave1D first-time-derivative constraint loss.",
    ),
    "conservation": _option(),
    "constraint_component": _option(
        {
            "component_name": _parameter(
                "string",
                "",
                description=(
                    "Required constraint_id from problem_spec.constraints. The LLM transport "
                    "canonicalizes this generic selector to that concrete loss-term name."
                ),
            )
        },
        description=(
            "Select one concrete problem constraint. Emit one term per required constraint, "
            "each with a distinct non-empty component_name."
        ),
    ),
    "gradient_residual": _option(),
}


WEIGHTING_OPTIONS = {
    "fixed": _option(),
    "none": _option(description="PINNsAgent name for fixed, unadapted loss weights.", alias_of="fixed"),
    "normalized": _option({"epsilon": _parameter("number", 1.0e-12, exclusive_minimum=0.0, maximum=1.0)}),
    "softmax": _option({"temperature": POSITIVE_SCALE}),
    "inverse_magnitude": _option({"epsilon": _parameter("number", 1.0e-12, exclusive_minimum=0.0, maximum=1.0), "power": _parameter("number", 1.0, minimum=0.0, maximum=16.0), "normalize": BOOL_TRUE}),
    "lra": _option({"alpha": _parameter("number", 0.9, minimum=0.0, maximum=1.0), "epsilon": _parameter("number", 1.0e-12, exclusive_minimum=0.0, maximum=1.0), "reference_index": _parameter("integer", 0, minimum=0, maximum=1024), "min_weight": _parameter("number", 1.0e-6, exclusive_minimum=0.0, maximum=1.0e6), "max_weight": _parameter("number", 1.0e6, exclusive_minimum=0.0, maximum=1.0e12)}),
    "gradient_balance": _option(
        {
            "ema_beta": _parameter("number", 0.95, minimum=0.0, maximum=0.9999, fidelity_scaling="none", description="EMA decay for registered component gradient norms."),
            "diagnostic_interval": _parameter("integer", 50, minimum=1, maximum=1_000_000, fidelity_scaling="optimizer_steps", description="Optimizer-step interval between component-gradient diagnostics."),
            "patience": _parameter("integer", 3, minimum=1, maximum=100, fidelity_scaling="none", description="Consecutive diagnostic checks required before proposing an update."),
            "cooldown_iterations": _parameter("integer", 200, minimum=0, maximum=10_000_000, fidelity_scaling="optimizer_steps", description="Optimizer-step cooldown after an accepted update."),
            "minimum_weight": _parameter("number", 0.01, exclusive_minimum=0.0, maximum=1.0e6, fidelity_scaling="none", description="Lower dynamic component-weight bound."),
            "maximum_weight": _parameter("number", 20.0, exclusive_minimum=0.0, maximum=1.0e6, fidelity_scaling="none", description="Upper dynamic component-weight bound."),
            "minimum_update_ratio": _parameter("number", 0.5, minimum=0.5, maximum=1.0, fidelity_scaling="none", hard_safety=True, description="Hard lower per-action weight ratio; cannot be below 0.5."),
            "maximum_update_ratio": _parameter("number", 2.0, minimum=1.0, maximum=2.0, fidelity_scaling="none", hard_safety=True, description="Hard upper per-action weight ratio; cannot exceed 2.0."),
            "conflict_threshold": _parameter("number", -0.1, minimum=-1.0, maximum=1.0, fidelity_scaling="none", description="Minimum gradient cosine below which a conflicting update is withheld."),
            "imbalance_threshold": _parameter("number", 100.0, exclusive_minimum=0.0, maximum=1.0e18, fidelity_scaling="none", llm_visible=False, description="Internal imbalance trigger retained for backward compatibility."),
            "alpha": _parameter("number", 0.5, minimum=0.0, maximum=1.0, fidelity_scaling="none", llm_visible=False, description="Internal bounded update blend."),
            "epsilon": _parameter("number", 1.0e-12, exclusive_minimum=0.0, maximum=1.0, fidelity_scaling="none", llm_visible=False, description="Internal numerical guard."),
        },
        compatibility=[
            {"rule": "parameter_order", "lower": "minimum_weight", "upper": "maximum_weight"},
            {
                "rule": "field_equals",
                "path": "training.extra_parameters.adaptive_control.enabled",
                "value": True,
            },
        ],
        description="Bounded registered-component gradient balancing in first-order stages.",
    ),
    "ntk": _option(
        {
            "epsilon": _parameter("number", 1.0e-12, exclusive_minimum=0.0, maximum=1.0),
            "min_weight": _parameter("number", 1.0e-6, exclusive_minimum=0.0, maximum=1.0e6),
            "max_weight": _parameter("number", 1.0e6, exclusive_minimum=0.0, maximum=1.0e12),
            "update_every": _parameter("integer", 1, minimum=1, maximum=1_000_000),
        },
        compatibility=[
            {"rule": "parameter_order", "lower": "min_weight", "upper": "max_weight"},
        ],
    ),
    "ntk_trace": _option({"alpha": _parameter("number", 0.9, minimum=0.0, maximum=1.0), "epsilon": _parameter("number", 1.0e-12, exclusive_minimum=0.0, maximum=1.0), "normalize": BOOL_TRUE}),
}


LBFGS_RECOVERY_OPTIONS = {
    "disabled": _option(description="Disable L-BFGS stall recovery."),
    "terminate_candidate": _option(
        {
            "minimum_completed_iterations_before_detection": _parameter("integer", 50, minimum=1, maximum=10_000_000, fidelity_scaling="optimizer_steps", description="Minimum L-BFGS outer iterations before stall detection."),
            "patience": _parameter("integer", 20, minimum=1, maximum=100_000, fidelity_scaling="optimizer_steps", description="Length of the bounded L-BFGS observation window."),
            "minimum_relative_improvement": _parameter("number", 1e-4, minimum=0.0, maximum=1.0e18, fidelity_scaling="none", description="Minimum relative loss improvement within the stall window."),
        },
        description="Terminate the candidate when bounded stall evidence is met.",
    ),
    "reallocate_to_adam": _option(
        {
            "minimum_completed_iterations_before_detection": _parameter("integer", 50, minimum=1, maximum=10_000_000, fidelity_scaling="optimizer_steps", description="Minimum L-BFGS outer iterations before stall detection."),
            "patience": _parameter("integer", 20, minimum=1, maximum=100_000, fidelity_scaling="optimizer_steps", description="Length of the bounded L-BFGS observation window."),
            "minimum_relative_improvement": _parameter("number", 1e-4, minimum=0.0, maximum=1.0e18, fidelity_scaling="none", description="Minimum relative loss improvement within the stall window."),
            "adam_learning_rate_scale": _parameter("number", 0.1, exclusive_minimum=0.0, maximum=1.0, fidelity_scaling="none", description="Recovery Adam learning-rate scale relative to the first-order base rate."),
            "maximum_recovery_iterations": _parameter("integer", 1000, minimum=1, maximum=100_000_000, fidelity_scaling="optimizer_steps", description="Maximum budget reallocated to fresh recovery Adam steps."),
        },
        description="Reallocate only the remaining bounded budget to a fresh Adam recovery stage.",
    ),
}


SAMPLING_CONTROL_OPTIONS = {
    "none": _option(description="Disable runtime sampling control."),
    "fixed_budget_subset_replacement": _option(
        {
            "diagnostic_interval": _parameter("integer", 200, minimum=1, maximum=10_000_000, fidelity_scaling="optimizer_steps", description="Optimizer-step interval between sampling diagnostics."),
            "minimum_iterations_before_control": _parameter("integer", 500, minimum=0, maximum=100_000_000, fidelity_scaling="optimizer_steps", description="Minimum first-order residence before replacement is eligible."),
            "patience": _parameter("integer", 3, minimum=1, maximum=100, fidelity_scaling="none", description="Consecutive diagnostic checks before replacement."),
            "cooldown_iterations": _parameter("integer", 500, minimum=0, maximum=100_000_000, fidelity_scaling="optimizer_steps", description="Optimizer-step cooldown between replacements."),
            "maximum_updates": _parameter("integer", 10, minimum=0, maximum=10_000, fidelity_scaling="none", description="Hard maximum replacement actions."),
            "p95_to_median_threshold": _parameter("number", 10.0, minimum=0.0, maximum=1.0e6, fidelity_scaling="none", description="Residual concentration threshold."),
            "replacement_fraction": _parameter("number", 0.2, exclusive_minimum=0.0, maximum=0.9, fidelity_scaling="none", description="Fraction of the fixed sampler budget replaced per action."),
            "hard_point_retention_fraction": _parameter("number", 0.2, minimum=0.0, maximum=1.0, fidelity_scaling="none", description="Fraction of persistent hard points retained."),
            "random_exploration_fraction": _parameter("number", 0.1, minimum=0.0, maximum=1.0, fidelity_scaling="none", description="Fraction reserved for random exploration."),
            "candidate_multiplier": _parameter("integer", 4, minimum=1, maximum=64, fidelity_scaling="none", description="Candidate pool multiplier relative to replacement count."),
            "maximum_point_age": _parameter("integer", 20, minimum=1, maximum=10_000, fidelity_scaling="none", description="Maximum point age measured in controller update opportunities."),
            "maximum_consecutive_retention": _parameter("integer", 5, minimum=0, maximum=10_000, fidelity_scaling="none", description="Maximum consecutive retention actions."),
            "minimum_point_age": _parameter("integer", 2, minimum=0, maximum=10_000, fidelity_scaling="none", description="Minimum age before a point may be replaced."),
            "observation_window_iterations": _parameter("integer", 300, minimum=1, maximum=100_000_000, fidelity_scaling="optimizer_steps", description="Post-action fixed-probe observation length."),
            "minimum_validation_stability_checks": _parameter("integer", 2, minimum=1, maximum=100, fidelity_scaling="none", description="Stable fixed-probe checks required before an action."),
            "validation_degradation_ratio": _parameter("number", 1.5, exclusive_minimum=1.0, maximum=1.0e6, fidelity_scaling="none", description="Aggregate fixed-probe rollback bound."),
            "associated_component_degradation_ratio": _parameter("number", 2.0, exclusive_minimum=1.0, maximum=1.0e6, fidelity_scaling="none", description="Associated-component rollback bound."),
            "maximum_candidate_points_evaluated": _parameter("integer", 100000, minimum=1, maximum=100_000_000, fidelity_scaling="sampling_points", description="Hard candidate-scoring point cap."),
        },
        description="Replace a bounded subset while preserving every sampler point budget.",
    ),
}


CURRICULUM_CONTROL_OPTIONS = {
    "none": _option(description="Disable curriculum advancement."),
    "residual_gated_level_advance": _option(
        {
            "minimum_iterations_per_level": _parameter("integer", 500, minimum=0, maximum=100_000_000, fidelity_scaling="optimizer_steps", description="Minimum residence in a predeclared level."),
            "maximum_iterations_per_level": _parameter("integer", 3000, minimum=1, maximum=100_000_000, fidelity_scaling="optimizer_steps", description="Maximum residence before the finite forced policy."),
            "validation_interval": _parameter("integer", 100, minimum=1, maximum=10_000_000, fidelity_scaling="optimizer_steps", description="Fixed Physics Probe validation interval."),
            "patience": _parameter("integer", 3, minimum=1, maximum=100, fidelity_scaling="none", description="Consecutive stable validation checks required."),
            "cooldown_iterations": _parameter("integer", 300, minimum=0, maximum=100_000_000, fidelity_scaling="optimizer_steps", description="Cooldown after advance or rollback."),
            "maximum_advances": _parameter("integer", 20, minimum=0, maximum=10_000, fidelity_scaling="none", description="Maximum registered single-level advances."),
            "normal_advance_condition": _parameter("string", "physics_validation_stable", enum=["physics_validation_stable", "absolute_threshold", "relative_plateau", "relative_improvement", "hybrid"], fidelity_scaling="none", description="Registered internal-physics gating rule."),
            "forced_advance_policy": _parameter("string", "advance_one_level", enum=["advance_one_level", "terminate_candidate", "hold_until_budget_end"], fidelity_scaling="none", description="Finite action when maximum residence is reached."),
            "minimum_remaining_budget_after_advance": _parameter("integer", 500, minimum=0, maximum=100_000_000, fidelity_scaling="optimizer_steps", description="Budget reserved for post-advance training and observation."),
            "observation_window_iterations": _parameter("integer", 500, minimum=1, maximum=100_000_000, fidelity_scaling="optimizer_steps", description="Post-advance fixed-probe observation length."),
            "post_advance_controller_ema_decay": _parameter("number", 0.5, minimum=0.0, maximum=1.0, fidelity_scaling="none", description="Predeclared gradient-controller EMA migration factor."),
        },
        description="PDE- and axis-independent single-level advancement over a registered Runtime.",
    ),
}


PHYSICS_VALIDATION_OPTIONS = {
    "disabled": _option(description="Disable internal fixed-probe validation."),
    "fixed_component_probe": _option(
        {
            "evaluation_interval": _parameter("integer", 100, minimum=1, maximum=10_000_000, fidelity_scaling="optimizer_steps", description="Optimizer-step interval between fixed-probe evaluations."),
            "points_per_component": _parameter("integer", 128, minimum=1, maximum=1_000_000, fidelity_scaling="validation_points", description="Fixed validation point budget per registered component."),
            "maximum_total_points": _parameter("integer", 4096, minimum=1, maximum=10_000_000, fidelity_scaling="validation_points", description="Hard total fixed-probe point cap."),
            "minimum_component_scale": _parameter("number", 0.0, minimum=0.0, maximum=1_000_000_000.0, fidelity_scaling="none", description="Absolute normalization floor that prevents degenerate near-zero initial residuals from defining the probe scale."),
            "region_normalization": _parameter("string", "per_region_initial", enum=["per_region_initial", "shared_pde_rmse"], fidelity_scaling="none", description="Regional residual normalization contract."),
            "second_order_degradation_ratio": _parameter("number", 1.0, minimum=1.0, maximum=100.0, fidelity_scaling="none", description="Reference-free L-BFGS rollback threshold relative to the pre-second-order Physics Probe score; one disables the guard."),
            "second_order_guard_metric": _parameter("string", "physics_validation_score", enum=["physics_validation_score", "variational_energy"], fidelity_scaling="none", description="Label-free fixed-probe metric used by the second-order degradation guard."),
        },
        description="Reference-free validation on an immutable registered component probe.",
    ),
}


FINAL_MODEL_POLICY_OPTIONS = {
    "last": _option(description="Return the final finite model state."),
    "best_train_loss": _option(
        {
            "evaluation_interval": _parameter("integer", 100, minimum=1, maximum=10_000_000, fidelity_scaling="optimizer_steps", description="Checkpoint comparison interval matching PINNacle/DeepXDE's default display_every=100."),
        },
        description="Match PINNacle/DeepXDE by reporting metrics at the checkpoint with minimum summed training loss.",
    ),
}


CONSTRAINT_ENFORCEMENT_OPTIONS = {
    "soft_penalty": _option(
        description="Use the registered initial/boundary/other constraint loss terms without an output transform."
    ),
    "problem_hard": _option(
        description="Apply a differentiable problem-owned hard output transform selected by transform_id."
    ),
}


AGGREGATION_OPTIONS = {
    "weighted_sum": _option(),
    "mean": _option(),
    "max": _option(),
    "logsumexp": _option({"temperature": POSITIVE_SCALE}),
}


INPUT_TRANSFORM_OPTIONS = {
    "none": _option(description="Use physical coordinates directly."),
    "normalize_bounds": _option(description="Map problem bounds to [-1, 1]."),
    "unit_bounds": _option(description="Map problem bounds to [0, 1]."),
    "fourier_features": _option(
        {
            "frequency_count": _parameter("integer", 8, minimum=1, maximum=256),
            "frequency_scale": _parameter("number", 1.0, exclusive_minimum=0.0, maximum=1000.0),
            "trainable_frequencies": BOOL_FALSE,
            "include_raw_input": BOOL_FALSE,
        },
        description="Fourier representation usable with a standard Wave MLP.",
    ),
    "multiscale_fourier_features": _option(
        {
            "frequency_count": _parameter("integer", 8, minimum=1, maximum=256),
            "frequency_scale": _parameter("number", 1.0, exclusive_minimum=0.0, maximum=1000.0),
            "frequency_scales": _parameter(
                "array",
                [1.0, 2.0, 4.0],
                items={"type": "number", "exclusiveMinimum": 0.0, "maximum": 1.0e4},
                min_items=1,
                max_items=32,
            ),
            "trainable_frequencies": BOOL_FALSE,
            "include_raw_input": BOOL_FALSE,
        },
        description="Concatenated multi-band Fourier representation.",
    ),
    "periodic_positional_encoding": _option(
        {
            "frequency_count": _parameter("integer", 8, minimum=1, maximum=256),
            "frequency_scale": _parameter("number", 1.0, exclusive_minimum=0.0, maximum=1000.0),
            "trainable_frequencies": BOOL_FALSE,
            "include_raw_input": BOOL_TRUE,
        },
        description="Periodic sine/cosine positional encoding.",
    ),
}

TIME_STRATEGY_OPTIONS = {
    "none": _option(description="Train on the complete time domain."),
    "time_window_curriculum": _option(
        {
            "window_fractions": _parameter(
                "array",
                [0.2, 0.4, 0.6, 0.8, 1.0],
                items={
                    "type": "number",
                    "exclusiveMinimum": 0.0,
                    "maximum": 1.0,
                },
                min_items=1,
                max_items=64,
            ),
            "iterations_per_window": _parameter(
                "array",
                [800, 800, 800, 800, 1800],
                items={"type": "integer", "minimum": 1},
                min_items=1,
                max_items=64,
            ),
        },
        description="Expand the active time interval monotonically to the full domain.",
    ),
}


SCHEDULER_OPTIONS = {
    "none": _option(),
    "step_lr": _option(
        {
            "step_size": _parameter("integer", 100, minimum=1, maximum=10_000_000),
            "gamma": _parameter("number", 0.1, exclusive_minimum=0.0, maximum=1.0),
        }
    ),
    "steplr": _option(
        {
            "step_size": _parameter("integer", 100, minimum=1, maximum=10_000_000),
            "gamma": _parameter("number", 0.1, exclusive_minimum=0.0, maximum=1.0),
        },
        alias_of="step_lr",
    ),
    "cosine_annealing": _option(
        {
            "T_max": _parameter("integer", 1000, minimum=1, maximum=10_000_000),
            "eta_min": _parameter("number", 0.0, minimum=0.0, maximum=100.0),
        }
    ),
    "cosineannealinglr": _option(
        {
            "T_max": _parameter("integer", 1000, minimum=1, maximum=10_000_000),
            "eta_min": _parameter("number", 0.0, minimum=0.0, maximum=100.0),
        },
        alias_of="cosine_annealing",
    ),
    "exponential_lr": _option({"gamma": _parameter("number", 0.99, exclusive_minimum=0.0, maximum=1.0)}),
    "exponentiallr": _option({"gamma": _parameter("number", 0.99, exclusive_minimum=0.0, maximum=1.0)}, alias_of="exponential_lr"),
    "reduce_on_plateau": _option(
        {
            "mode": _parameter("string", "min", enum=["min", "max"]),
            "factor": _parameter("number", 0.1, exclusive_minimum=0.0, maximum=1.0),
            "patience": _parameter("integer", 10, minimum=0, maximum=1_000_000),
            "threshold": _parameter("number", 1.0e-4, minimum=0.0, maximum=1.0),
        }
    ),
    "reducelronplateau": _option(
        {
            "mode": _parameter("string", "min", enum=["min", "max"]),
            "factor": _parameter("number", 0.1, exclusive_minimum=0.0, maximum=1.0),
            "patience": _parameter("integer", 10, minimum=0, maximum=1_000_000),
            "threshold": _parameter("number", 1.0e-4, minimum=0.0, maximum=1.0),
        },
        alias_of="reduce_on_plateau",
    ),
    "one_cycle": _option(
        {
            "max_lr": _parameter("number", 1.0e-3, exclusive_minimum=0.0, maximum=100.0),
            "total_steps": _parameter("integer", 1000, minimum=1, maximum=10_000_000),
            "pct_start": _parameter("number", 0.3, exclusive_minimum=0.0, maximum=1.0),
        }
    ),
    "onecyclelr": _option(
        {
            "max_lr": _parameter("number", 1.0e-3, exclusive_minimum=0.0, maximum=100.0),
            "total_steps": _parameter("integer", 1000, minimum=1, maximum=10_000_000),
            "pct_start": _parameter("number", 0.3, exclusive_minimum=0.0, maximum=1.0),
        },
        alias_of="one_cycle",
    ),
    "linear_lr": _option(
        {
            "start_factor": _parameter("number", 1.0, exclusive_minimum=0.0, maximum=1.0),
            "end_factor": _parameter("number", 0.1, exclusive_minimum=0.0, maximum=1.0),
            "total_iters": _parameter("integer", 1000, minimum=1, maximum=10_000_000),
        }
    ),
    "polynomial_lr": _option(
        {
            "total_iters": _parameter("integer", 1000, minimum=1, maximum=10_000_000),
            "power": _parameter("number", 1.0, minimum=0.0, maximum=16.0),
        }
    ),
    "constant_lr": _option(
        {
            "factor": _parameter("number", 0.5, exclusive_minimum=0.0, maximum=1.0),
            "total_iters": _parameter("integer", 100, minimum=1, maximum=10_000_000),
        }
    ),
    "multi_step_lr": _option(
        {
            "milestones": _parameter("array", [500, 750], items={"type": "integer", "minimum": 1}, min_items=1, max_items=1024),
            "gamma": _parameter("number", 0.1, exclusive_minimum=0.0, maximum=1.0),
        }
    ),
    "cyclic_lr": _option(
        {
            "base_lr": _parameter("number", 1.0e-4, exclusive_minimum=0.0, maximum=100.0),
            "max_lr": _parameter("number", 1.0e-3, exclusive_minimum=0.0, maximum=100.0),
            "step_size_up": _parameter("integer", 2000, minimum=1, maximum=10_000_000),
            "cycle_momentum": BOOL_FALSE,
        },
        compatibility=[{"rule": "parameter_order", "lower": "base_lr", "upper": "max_lr"}],
    ),
}


def algorithm_spec_option_registry() -> dict[str, Any]:
    """Return an isolated copy of the legal independent AlgorithmSpec options."""

    return deepcopy(
        {
            "schema_version": "1.1",
            "composition_policy": "dynamic_field_composition_no_predefined_complete_algorithms",
            "fields": {
                "network.architecture": {
                    "options": NETWORK_OPTIONS,
                    "x-search": {"recommended_values": ["fnn", "laaf", "gaaf"]},
                },
                "network.activation.name": {
                    "options": ACTIVATION_OPTIONS,
                    "x-search": {
                        "recommended_values": ["elu", "selu", "sigmoid", "silu", "relu", "tanh", "swish", "sin", "gaussian"]
                    },
                },
                "network.output_activation.name": {"options": ACTIVATION_OPTIONS},
                "network.initialization.name": {
                    "options": INITIALIZER_OPTIONS,
                    "x-search": {"recommended_values": ["glorot_normal", "glorot_uniform", "he_normal", "he_uniform", "zeros"]},
                },
                "network.input_transform.name": {"options": INPUT_TRANSFORM_OPTIONS},
                "network.hidden_layers": {
                    "value_schema": _parameter(
                        "array",
                        [128, 128, 128, 128],
                        items={
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 4096,
                            "x-search": {"scale": "linear", "recommended_values": PINNSAGENT_WIDTHS},
                        },
                        min_items=1,
                        max_items=64,
                        x_search={
                            "recommended_lengths": PINNSAGENT_DEPTHS,
                            "composition": "constant_width_hidden_layers",
                        },
                    )
                },
                "network.residual_connections.enabled": {
                    "value_schema": _parameter("boolean", False)
                },
                "network.residual_connections.parameters": {
                    "value_schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {"scale": _parameter("number", 1.0, minimum=-1000.0, maximum=1000.0)},
                    }
                },
                "sampling.interior.strategy": {
                    "options": BASE_SAMPLER_OPTIONS,
                    "x-search": {"parameter_recommendations": {"n_points": PINNSAGENT_POINT_COUNTS}},
                },
                "sampling.boundary.strategy": {
                    "options": CONSTRAINT_SAMPLER_OPTIONS,
                    "x-search": {"parameter_recommendations": {"n_points": PINNSAGENT_POINT_COUNTS}},
                },
                "sampling.initial.strategy": {
                    "options": CONSTRAINT_SAMPLER_OPTIONS,
                    "x-search": {"parameter_recommendations": {"n_points": PINNSAGENT_POINT_COUNTS}},
                },
                "sampling.adaptive_refinement.strategy": {"options": ADAPTIVE_SAMPLER_OPTIONS},
                "constraint_enforcement.method": {
                    "options": CONSTRAINT_ENFORCEMENT_OPTIONS,
                    "x-search": {"recommended_values": ["soft_penalty", "problem_hard"]},
                },
                "constraint_enforcement.transform_id": {
                    "value_schema": _parameter("string", "none")
                },
                "constraint_enforcement.parameters": {
                    "value_schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {},
                    }
                },
                "loss.terms[].name": {
                    "options": LOSS_TERM_OPTIONS,
                    "description": (
                        "Stable loss-term identifiers. PDE-provided concrete "
                        "constraint_id values are also accepted contextually."
                    ),
                },
                "loss.terms[].loss_function": {"options": LOSS_OPTIONS},
                "loss.terms[].weight": {
                    "value_schema": _parameter(
                        "number",
                        1.0,
                        exclusive_minimum=0.0,
                        maximum=1.0e12,
                        x_search={
                            "scale": "log",
                            "minimum": 0.01,
                            "maximum": 100.0,
                        },
                        description=(
                            "Every included loss term must have a strictly positive weight; "
                            "omit an optional term instead of disabling it with weight 0."
                        ),
                    )
                },
                "loss.weighting_strategy.name": {
                    "options": WEIGHTING_OPTIONS,
                    "x-search": {
                        "recommended_values": ["none", "lra", "ntk"],
                        "adaptive_policy_values": ["fixed", "lra", "gradient_balance"],
                    },
                },
                "loss.aggregation.name": {"options": AGGREGATION_OPTIONS},
                "optimization.phases[].optimizer": {
                    "options": OPTIMIZER_OPTIONS,
                    "x-search": {"recommended_values": ["sgd", "adam", "multiadam", "lbfgs"]},
                },
                "optimization.phases[].iterations": {
                    "value_schema": _parameter(
                        "integer",
                        1000,
                        minimum=0,
                        maximum=100_000_000,
                        x_search={"scale": "log"},
                    )
                },
                "optimization.phases[].scheduler.name": {"options": SCHEDULER_OPTIONS},
                "training.batch_mode": {
                    "value_schema": _parameter("string", "full_batch", enum=["full_batch"])
                },
                "training.batch_size": {
                    "value_schema": _parameter(
                        "null",
                        None,
                        enum=[None],
                        description=(
                            "The current PINN trainer is full-batch only; this field must be JSON null."
                        ),
                    )
                },
                "training.gradient_clipping.enabled": {
                    "value_schema": _parameter("boolean", False)
                },
                "training.gradient_clipping.parameters": {
                    "value_schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "max_norm": _parameter("number", 1.0, exclusive_minimum=0.0, maximum=1.0e6),
                            "norm_type": _parameter("number", 2.0, exclusive_minimum=0.0, maximum=1.0e6),
                        },
                    }
                },
                "training.early_stopping.enabled": {
                    "value_schema": _parameter("boolean", False)
                },
                "training.early_stopping.parameters": {
                    "value_schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "patience": _parameter("integer", 100, minimum=1, maximum=10_000_000),
                            "threshold": _parameter("number", 1.0e-8, minimum=0.0, maximum=1.0e6),
                        },
                    }
                },
                "training.time_strategy.name": {
                    "options": TIME_STRATEGY_OPTIONS
                },
                "training.extra_parameters.adaptive_control.lbfgs_stall_recovery.action": {
                    "options": LBFGS_RECOVERY_OPTIONS,
                    "x-search": {"recommended_values": ["disabled", "terminate_candidate", "reallocate_to_adam"]},
                },
                "training.extra_parameters.adaptive_control.sampling_control.strategy": {
                    "options": SAMPLING_CONTROL_OPTIONS,
                    "x-search": {"recommended_values": ["none", "fixed_budget_subset_replacement"]},
                },
                "training.extra_parameters.adaptive_control.curriculum_control.strategy": {
                    "options": CURRICULUM_CONTROL_OPTIONS,
                    "x-search": {"recommended_values": ["none", "residual_gated_level_advance"]},
                },
                "training.extra_parameters.physics_validation.strategy": {
                    "options": PHYSICS_VALIDATION_OPTIONS,
                    "x-search": {"recommended_values": ["disabled", "fixed_component_probe"]},
                },
                "training.extra_parameters.best_checkpoint.final_model_policy": {
                    "options": FINAL_MODEL_POLICY_OPTIONS,
                    "x-search": {"recommended_values": ["last", "best_train_loss"]},
                },
                "training.initial_state_pretraining.enabled": {
                    "value_schema": _parameter("boolean", False)
                },
                "training.initial_state_pretraining.parameters": {
                    "value_schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "iterations": _parameter(
                                "integer", 0, minimum=0, maximum=10_000_000
                            )
                        },
                    }
                },
            },
        }
    )


def llm_algorithm_spec_option_registry() -> dict[str, Any]:
    """Return a compact, prompt-safe view of the AlgorithmSpec rules.

    The full runtime capability object is intentionally not part of this
    contract.  The LLM only needs legal field choices and the parameter rules
    required to compose a candidate; validation and construction continue to
    use the full local registry and builder implementation.
    """

    registry = algorithm_spec_option_registry()
    compact: dict[str, Any] = {
        "schema_version": registry["schema_version"],
        "composition_policy": registry["composition_policy"],
        "adaptive_parameter_semantics": {
            "optimizer_steps": "integer optimizer-step count or window; scale with fidelity",
            "sampling_points": "integer sampling-point cap; scale with sampling budget",
            "validation_points": "integer fixed-probe point cap; scale with validation budget",
            "none": "dimensionless threshold, ratio, count, or policy identity; do not scale",
            "x-hard-safety": "true means the LLM may not relax the registered bound",
        },
        "fields": {},
    }
    fields = compact["fields"]
    for field_path, field in registry["fields"].items():
        if "value_schema" in field:
            value_schema = _compact_parameter_schema(field["value_schema"])
            if "x-search" in field["value_schema"]:
                value_schema["x-search"] = deepcopy(
                    field["value_schema"]["x-search"]
                )
            fields[field_path] = {"value_schema": value_schema}
            continue
        options: dict[str, Any] = {}
        for option_name, option in (field.get("options") or {}).items():
            alias_of = option.get("alias_of")
            if alias_of:
                options[option_name] = {"alias_of": alias_of}
                continue
            properties = ((option.get("parameter_schema") or {}).get("properties") or {})
            compact_option: dict[str, Any] = {
                "parameters": {
                    parameter_name: _compact_parameter_schema(parameter_schema)
                    for parameter_name, parameter_schema in properties.items()
                    if parameter_schema.get("x-llm-visible") is not False
                }
            }
            if option.get("compatibility"):
                compact_option["compatibility"] = deepcopy(option["compatibility"])
            options[option_name] = compact_option
        compact_field: dict[str, Any] = {"options": options}
        if field.get("x-search"):
            compact_field["x-search"] = deepcopy(field["x-search"])
        fields[field_path] = compact_field
    return compact


def _compact_parameter_schema(value: Any) -> Any:
    """Keep only executable parameter constraints needed by the LLM."""

    if isinstance(value, list):
        return [_compact_parameter_schema(item) for item in value]
    if not isinstance(value, dict):
        return deepcopy(value)
    executable_keys = {
        "type",
        "default",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "enum",
        "items",
        "minItems",
        "maxItems",
        "required",
        "additionalProperties",
        "properties",
        "x-fidelity-scaling",
        "x-hard-safety",
    }
    compact: dict[str, Any] = {}
    for key, item in value.items():
        if key not in executable_keys:
            continue
        if key == "properties" and isinstance(item, dict):
            compact[key] = {
                property_name: _compact_parameter_schema(property_schema)
                for property_name, property_schema in item.items()
                if not (
                    isinstance(property_schema, dict)
                    and property_schema.get("x-llm-visible") is False
                )
            }
        else:
            compact[key] = _compact_parameter_schema(item)
    return compact


def option_parameter_names(registry: Mapping[str, Any], field_path: str) -> dict[str, list[str]]:
    options = ((registry.get("fields") or {}).get(field_path) or {}).get("options") or {}
    return {
        name: sorted(((option.get("parameter_schema") or {}).get("properties") or {}).keys())
        for name, option in options.items()
    }


__all__ = [
    "algorithm_spec_option_registry",
    "llm_algorithm_spec_option_registry",
    "option_parameter_names",
]
