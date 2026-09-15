"""Machine-readable coverage registry for the PINNacle PDE suite.

The 20 core profiles follow the benchmark registry bundled under
``third_party/pinnacle``. Two supplemental profiles cover additional PDE
implementations shipped with the same source tree.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


PINNACLE_ALGORITHM_PARAMETER_SPACE: dict[str, Any] = {
    "iterations": {"default": 20_000, "minimum": 1, "switch_epoch_default": 5_000},
    "network": {
        "architectures": [
            "mlp", "resnet", "modified_mlp", "factorized_mlp", "fourier_mlp",
            "multiscale_mlp", "multiscale_fourier_mlp", "siren_mlp",
            "laaf", "gaaf", "parallel_fnn",
        ],
        "width": {"minimum": 1, "pinnacle_random_search": {"start": 100, "stop": 250, "step": 4}},
        "depth": {"minimum": 1, "pinnacle_random_search": {"start": 3, "stop": 7, "step": 1}},
        "adaptive_activation": {"activation_scale_default": 10.0, "initial_slope_default": 1.0},
        "factorization": {"rank_minimum": 1},
    },
    "activation": ["elu", "relu", "selu", "sigmoid", "silu", "swish", "sine", "tanh"],
    "initializer": [
        "glorot_normal", "glorot_uniform", "xavier_normal", "xavier_uniform",
        "kaiming_normal", "kaiming_uniform", "siren", "zeros"
    ],
    "optimizer": {
        "names": ["adam", "sgd", "multiadam", "lbfgs", "adamw", "adamax", "radam", "nadam"],
        "learning_rates": [1e-5, 1e-4, 1e-3],
        "adam_lbfgs": {"representation": "two optimization phases", "switch_epoch_default": 5_000},
    },
    "sampling": {
        "interior": ["uniform", "latin_hypercube", "sobol", "halton", "hammersley", "grid"],
        "adaptive": ["residual_adaptive", "gradient_adaptive", "hybrid_adaptive"],
        "point_range": {"minimum": 100, "maximum": 10_000, "step": 500},
        "defaults": {"domain": 8_192, "boundary": 2_048, "initial": 2_048, "test": 8_192},
    },
    "loss": {
        "functions": [
            "mse", "mae", "rmse", "huber", "smooth_l1", "log_cosh", "cauchy",
            "charbonnier", "pseudo_huber",
        ],
        "terms": [
            "pde_residual", "pde_component", "gradient_residual", "boundary_condition",
            "spatial_boundary", "initial_condition", "initial_displacement", "initial_velocity",
            "constraint_component",
        ],
        "weighting": ["fixed", "normalized", "inverse_magnitude", "softmax", "lra", "ntk", "ntk_trace"],
        "aggregation": ["weighted_sum", "mean", "max", "logsumexp"],
    },
    "constraint_types": ["dirichlet", "initial", "neumann", "operator", "periodic", "pointset", "robin"],
    "constraint_enforcement": {
        "methods": ["soft_penalty", "problem_hard"],
        "fairness_policy": "Every candidate for one problem receives the same registered method and transform options.",
    },
    "general_methods": {
        "gepinn": {"representation": "gradient_residual loss term"},
        "rar": {"representation": "residual_adaptive sampler"},
        "lra": {"representation": "lra loss weighting"},
        "ntk": {"representation": "ntk two-group PINNacle loss weighting"},
        "laaf": {"representation": "laaf network"},
        "gaaf": {"representation": "gaaf network"},
        "multiadam": {"representation": "multiadam optimizer with per-group moment states"},
        "time_window_curriculum": {"representation": "expanding causal time windows"},
        "initial_state_pretraining": {"representation": "Wave initial-state-only warm start"},
    },
}


def _sampling(multiplier: int, time_dependent: bool) -> dict[str, int]:
    return {
        "domain": 8_192 * multiplier,
        "boundary": 2_048 * multiplier,
        "initial": 2_048 * multiplier if time_dependent else 0,
        "test": 8_192 * multiplier,
    }


def _requirements(
    *,
    tags: list[str],
    output_dimension: int,
    inverse: bool,
    constraints: list[str],
    wave_initial_state: bool = False,
) -> dict[str, list[str]]:
    networks = ["mlp", "resnet", "modified_mlp", "laaf", "gaaf"]
    activations = ["tanh", "silu", "swish"]
    samplers = ["uniform", "latin_hypercube", "sobol", "halton", "hammersley"]
    terms = ["pde_residual", "gradient_residual"]
    if "complex_geometry" in tags:
        samplers = ["uniform"]
    if "multiscale" in tags or "long_time" in tags or "chaotic" in tags:
        networks.extend(["fourier_mlp", "multiscale_mlp"])
        activations.append("sine")
    if "high_dimensional" in tags:
        networks.extend(["factorized_mlp", "fourier_mlp"])
    if output_dimension > 1 or inverse:
        networks.append("parallel_fnn")
        terms.extend(["pde_component", "constraint_component"])
    if wave_initial_state:
        networks.extend(["fourier_mlp", "multiscale_fourier_mlp", "siren_mlp"])
        activations.append("sine")
        terms.extend(["spatial_boundary", "initial_displacement", "initial_velocity"])
    elif "initial" in constraints:
        terms.append("initial_condition")
    if any(name != "initial" for name in constraints):
        terms.append("boundary_condition")
    optimizers = ["adam", "adamw", "lbfgs"]
    if output_dimension > 1 or "multiscale" in tags or wave_initial_state:
        optimizers.append("multiadam")
    return {
        "networks": sorted(set(networks)),
        "activations": sorted(set(activations)),
        "interior_samplers": sorted(set(samplers)),
        "adaptive_samplers": ["residual_adaptive", "gradient_adaptive", "hybrid_adaptive"],
        "loss_terms": sorted(set(terms)),
        "weighting_strategies": ["fixed", "inverse_magnitude", "lra", "normalized", "ntk", "ntk_trace"],
        "optimizers": sorted(set(optimizers)),
        "constraint_types": sorted(set(constraints)),
    }


def _profile(
    name: str,
    *,
    aliases: list[str],
    family: str,
    spatial_dimension: int,
    input_dimension: int,
    output_dimension: int,
    time_dependent: bool,
    differential_order: int,
    equations: int,
    geometry: str,
    constraints: list[str],
    tags: list[str],
    physical_parameters: dict[str, Any],
    sampling_multiplier: int = 1,
    core: bool = True,
    inverse: bool = False,
    hard_constraint_transforms: list[str] | None = None,
    wave_initial_state: bool = False,
) -> dict[str, Any]:
    hard_constraint_transforms = list(hard_constraint_transforms or [])
    return {
        "pinnacle_name": name,
        "aliases": aliases,
        "core_benchmark": core,
        "task_type": "inverse" if inverse else "forward",
        "family": family,
        "spatial_dimension": spatial_dimension,
        "input_dimension": input_dimension,
        "output_dimension": output_dimension,
        "time_dependent": time_dependent,
        "differential_order": differential_order,
        "governing_equations": equations,
        "geometry": geometry,
        "constraint_types": constraints,
        "constraint_enforcement_options": [
            {"method": "soft_penalty", "transform_id": "none"},
            *[
                {"method": "problem_hard", "transform_id": transform_id}
                for transform_id in hard_constraint_transforms
            ],
        ],
        "hard_constraint_transforms": hard_constraint_transforms,
        "challenge_tags": tags,
        "physical_parameters": physical_parameters,
        "default_sampling": _sampling(sampling_multiplier, time_dependent),
        "required_registry_parameters": _requirements(
            tags=tags,
            output_dimension=output_dimension,
            inverse=inverse,
            constraints=constraints,
            wave_initial_state=wave_initial_state,
        ),
        "mandatory_algorithm_requirements": (
            {
                "positive_initial_sampling": True,
                "required_loss_terms": [
                    "spatial_boundary",
                    "initial_displacement",
                    "initial_velocity",
                ],
                "time_derivative_constraint": {
                    "name": "initial_velocity",
                    "derivative_variable": "t",
                    "derivative_order": 1,
                },
            }
            if wave_initial_state
            else {}
        ),
    }


PINNACLE_PDE_PROFILES: dict[str, dict[str, Any]] = {
    "burgers_1d": _profile(
        "Burgers1D", aliases=["classic_burgers_1d", "Burgers1d"], family="burgers",
        spatial_dimension=1, input_dimension=2, output_dimension=1, time_dependent=True,
        differential_order=2, equations=1, geometry="interval_x_time",
        constraints=["dirichlet", "initial"], tags=["shock", "nonlinear", "conservation"],
        physical_parameters={"nu": "0.01/pi", "x": [-1, 1], "t": [0, 1]},
        hard_constraint_transforms=["burgers_1d_initial_dirichlet_exact"],
    ),
    "wave_1d": _profile(
        "Wave1D", aliases=["WaveEquation1D", "wave_c"], family="wave",
        spatial_dimension=1, input_dimension=2, output_dimension=1, time_dependent=True,
        differential_order=2, equations=1, geometry="rectangle_xt",
        constraints=["dirichlet", "neumann", "initial"], tags=["oscillatory"],
        physical_parameters={"C": 2, "a": 4, "domain": [0, 1, 0, 1]},
        wave_initial_state=True,
    ),
    "kuramoto_sivashinsky_1d": _profile(
        "KuramotoSivashinskyEquation", aliases=["ks_1d"], family="chaotic",
        spatial_dimension=1, input_dimension=2, output_dimension=1, time_dependent=True,
        differential_order=4, equations=1, geometry="periodic_interval_x_time",
        constraints=["initial"], tags=["chaotic", "stiff", "high_order", "periodic"],
        physical_parameters={"alpha": "100/16", "beta": "100/16^2", "gamma": "100/16^4", "x": [0, "2*pi"], "t": [0, 1]},
    ),
    "burgers_2d": _profile(
        "Burgers2D", aliases=["Burgers2d"], family="burgers",
        spatial_dimension=2, input_dimension=3, output_dimension=2, time_dependent=True,
        differential_order=2, equations=2, geometry="rectangle_xy_time",
        constraints=["initial", "periodic"], tags=["shock", "nonlinear", "multi_output", "conservation"],
        physical_parameters={"nu": 0.001, "L": 4, "T": 1}, sampling_multiplier=4,
    ),
    "poisson_2d_classic": _profile(
        "Poisson2D_Classic", aliases=["Poisson2d"], family="poisson",
        spatial_dimension=2, input_dimension=2, output_dimension=1, time_dependent=False,
        differential_order=2, equations=1, geometry="rectangle_with_holes",
        constraints=["dirichlet"], tags=["complex_geometry", "elliptic"],
        physical_parameters={"scale": 1},
    ),
    "poisson_boltzmann_2d": _profile(
        "PoissonBoltzmann2D", aliases=["Poisson_boltzmann2d"], family="poisson_boltzmann",
        spatial_dimension=2, input_dimension=2, output_dimension=1, time_dependent=False,
        differential_order=2, equations=1, geometry="rectangle_with_four_holes",
        constraints=["dirichlet"], tags=["complex_geometry", "nonlinear", "elliptic"],
        physical_parameters={"k": 8, "mu": [1, 4], "A": 10, "bbox": [-1, 1, -1, 1]},
    ),
    "poisson_2d_many_area": _profile(
        "Poisson2D_ManyArea", aliases=["Poisson2d_Many_subdomains"], family="poisson",
        spatial_dimension=2, input_dimension=2, output_dimension=1, time_dependent=False,
        differential_order=2, equations=1, geometry="rectangle_piecewise_subdomains",
        constraints=["robin"], tags=["multiscale", "piecewise_coefficient", "interface"],
        physical_parameters={"bbox": [-10, 10, -10, 10], "split": [5, 5], "frequency": 2},
    ),
    "heat_2d_varying_coefficient": _profile(
        "Heat2D_VaryingCoef", aliases=["Heat2d_Varying_Source"], family="heat",
        spatial_dimension=2, input_dimension=3, output_dimension=1, time_dependent=True,
        differential_order=2, equations=1, geometry="rectangle_xy_time",
        constraints=["dirichlet", "initial"], tags=["varying_coefficient", "time_dependent"],
        physical_parameters={"A": 200, "m": [1, 5, 1], "bbox": [0, 1, 0, 1, 0, 5]}, sampling_multiplier=4,
    ),
    "heat_2d_multiscale": _profile(
        "Heat2D_Multiscale", aliases=["Heat_Multi_scale"], family="heat",
        spatial_dimension=2, input_dimension=3, output_dimension=1, time_dependent=True,
        differential_order=2, equations=1, geometry="rectangle_xy_time",
        constraints=["dirichlet", "initial"], tags=["multiscale", "anisotropic", "time_dependent"],
        physical_parameters={"pde_coef": ["1/(500*pi)^2", "1/pi^2"], "init_coef": ["20*pi", "pi"]}, sampling_multiplier=4,
    ),
    "heat_2d_complex_geometry": _profile(
        "Heat2D_ComplexGeometry", aliases=["HeatComplex"], family="heat",
        spatial_dimension=2, input_dimension=3, output_dimension=1, time_dependent=True,
        differential_order=2, equations=1, geometry="rectangle_with_multiple_circular_holes_x_time",
        constraints=["initial", "robin"], tags=["complex_geometry", "time_dependent"],
        physical_parameters={"bbox": [-8, 8, -12, 12, 0, 3]}, sampling_multiplier=4,
    ),
    "heat_2d_long_time": _profile(
        "Heat2D_LongTime", aliases=["HeatLongTime"], family="heat",
        spatial_dimension=2, input_dimension=3, output_dimension=1, time_dependent=True,
        differential_order=2, equations=1, geometry="rectangle_xy_time",
        constraints=["dirichlet", "initial"], tags=["long_time", "time_dependent"],
        physical_parameters={"k": 1, "m1": 4, "m2": 2, "t": [0, 100]}, sampling_multiplier=4,
    ),
    "navier_stokes_2d_C": _profile(
        "NS2D_LidDriven", aliases=["NS2D_LidDriven"], family="navier_stokes",
        spatial_dimension=2, input_dimension=2, output_dimension=3, time_dependent=False,
        differential_order=2, equations=3, geometry="unit_square_lid_driven_cavity",
        constraints=["dirichlet", "pointset"],
        tags=["multi_output", "incompressible", "nonlinear", "lid_driven_cavity"],
        physical_parameters={"a": 4, "nu": 0.01, "reynolds": 100, "bbox": [0, 1, 0, 1]},
    ),
    "navier_stokes_2d_backstep": _profile(
        "NS2D_BackStep", aliases=["NS_Back_Step"], family="navier_stokes",
        spatial_dimension=2, input_dimension=2, output_dimension=3, time_dependent=False,
        differential_order=2, equations=3, geometry="backward_facing_step_with_optional_obstacles",
        constraints=["dirichlet"], tags=["multi_output", "incompressible", "nonlinear", "complex_geometry"],
        physical_parameters={"nu": 0.01, "bbox": [0, 4, 0, 2]},
    ),
    "navier_stokes_2d_long_time": _profile(
        "NS2D_LongTime", aliases=["NSEquation_Long"], family="navier_stokes",
        spatial_dimension=2, input_dimension=3, output_dimension=3, time_dependent=True,
        differential_order=2, equations=3, geometry="rectangle_xy_time",
        constraints=["dirichlet", "initial"], tags=["multi_output", "incompressible", "nonlinear", "long_time"],
        physical_parameters={"nu": 0.01, "bbox": [0, 2, 0, 1, 0, 5]}, sampling_multiplier=4,
    ),
    "wave_2d_heterogeneous": _profile(
        "Wave2D_Heterogeneous", aliases=["WaveHeterogeneous"], family="wave",
        spatial_dimension=2, input_dimension=3, output_dimension=1, time_dependent=True,
        differential_order=2, equations=1, geometry="box_xyt",
        constraints=["dirichlet", "neumann"], tags=["heterogeneous", "varying_coefficient", "time_dependent"],
        physical_parameters={"mu": [-0.5, 0], "sigma": 0.3, "bbox": [-1, 1, -1, 1, 0, 5]}, sampling_multiplier=4,
    ),
    "wave_2d_long_time": _profile(
        "Wave2D_LongTime", aliases=["WaveEquation2D_Long"], family="wave",
        spatial_dimension=2, input_dimension=3, output_dimension=1, time_dependent=True,
        differential_order=2, equations=1, geometry="rectangle_xy_time",
        constraints=["dirichlet", "initial"], tags=["oscillatory", "long_time"],
        physical_parameters={
            "a": "sqrt(2)",
            "t": [0, 100],
            "equation": "u_tt + u_xx + a^2*u_yy = 0",
            "temporal_cycles": 50,
            "reference_peak_scale": "approximately sinh(3*pi)",
            "initial_velocity": "u_t(x,y,0)=0",
        }, sampling_multiplier=4,
    ),
    "gray_scott_2d": _profile(
        "GrayScottEquation", aliases=["GrayScottEquation"], family="reaction_diffusion",
        spatial_dimension=2, input_dimension=3, output_dimension=2, time_dependent=True,
        differential_order=2, equations=2, geometry="rectangle_xy_time",
        constraints=["initial"], tags=["chaotic", "stiff", "multi_output", "reaction_diffusion", "long_time"],
        physical_parameters={"b": 0.04, "d": 0.1, "epsilon": [1e-5, 5e-6], "t": [0, 200]}, sampling_multiplier=4,
    ),
    "poisson_3d_complex_geometry": _profile(
        "Poisson3D_ComplexGeometry", aliases=["Poisson3d"], family="poisson",
        spatial_dimension=3, input_dimension=3, output_dimension=1, time_dependent=False,
        differential_order=2, equations=1, geometry="cube_with_spherical_holes",
        constraints=["neumann"], tags=["complex_geometry", "piecewise_coefficient", "elliptic"],
        physical_parameters={"interface_z": 0.5, "A": [20, 100], "m": [1, 10, 5], "k": [8, 10], "mu": [1, 1]}, sampling_multiplier=4,
    ),
    "poisson_5d": _profile(
        "PoissonND", aliases=["PoissonND"], family="poisson",
        spatial_dimension=5, input_dimension=5, output_dimension=1, time_dependent=False,
        differential_order=2, equations=1, geometry="five_dimensional_hypercube",
        constraints=["dirichlet"], tags=["high_dimensional", "elliptic"],
        physical_parameters={"dimension": 5, "length": 1}, sampling_multiplier=4,
    ),
    "heat_5d": _profile(
        "HeatND", aliases=["HeatND"], family="heat",
        spatial_dimension=5, input_dimension=6, output_dimension=1, time_dependent=True,
        differential_order=2, equations=1, geometry="five_dimensional_hypersphere_x_time",
        constraints=["initial", "neumann"], tags=["high_dimensional", "time_dependent", "complex_geometry"],
        physical_parameters={"dimension": 5, "T": 1}, sampling_multiplier=4,
    ),
    # Supplemental PDE implementations bundled with PINNacle.
    "poisson_inverse_2d": _profile(
        "PoissonInv", aliases=["PoissonInv"], family="inverse_poisson", core=False, inverse=True,
        spatial_dimension=2, input_dimension=2, output_dimension=2, time_dependent=False,
        differential_order=2, equations=1, geometry="unit_square", constraints=["dirichlet", "pointset"],
        tags=["inverse", "multi_output", "noisy_observations"], physical_parameters={"noise_std": 0.1},
    ),
    "heat_inverse_2d": _profile(
        "HeatInv", aliases=["HeatInv"], family="inverse_heat", core=False, inverse=True,
        spatial_dimension=2, input_dimension=3, output_dimension=2, time_dependent=True,
        differential_order=2, equations=1, geometry="rectangle_xy_time", constraints=["dirichlet", "pointset"],
        tags=["inverse", "multi_output", "noisy_observations", "time_dependent"],
        physical_parameters={"noise_std": 0.1, "bbox": [-1, 1, -1, 1, 0, 1]}, sampling_multiplier=4,
    ),
}


_ALIASES = {
    alias.lower(): problem_id
    for problem_id, profile in PINNACLE_PDE_PROFILES.items()
    for alias in [problem_id, profile["pinnacle_name"], *profile["aliases"]]
}


def list_pinnacle_problems(*, core_only: bool = False) -> list[str]:
    return sorted(
        problem_id
        for problem_id, profile in PINNACLE_PDE_PROFILES.items()
        if not core_only or profile["core_benchmark"]
    )


def get_pinnacle_profile(problem_id: str) -> dict[str, Any] | None:
    canonical = _ALIASES.get(str(problem_id).lower())
    return deepcopy(PINNACLE_PDE_PROFILES[canonical]) if canonical else None


def pinnacle_registry_summary() -> dict[str, Any]:
    return {
        "core_problem_count": len(list_pinnacle_problems(core_only=True)),
        "total_problem_count": len(PINNACLE_PDE_PROFILES),
        "problem_ids": list_pinnacle_problems(),
        "algorithm_parameter_space": deepcopy(PINNACLE_ALGORITHM_PARAMETER_SPACE),
    }


__all__ = [
    "PINNACLE_ALGORITHM_PARAMETER_SPACE",
    "PINNACLE_PDE_PROFILES",
    "get_pinnacle_profile",
    "list_pinnacle_problems",
    "pinnacle_registry_summary",
]
