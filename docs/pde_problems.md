# PDE problem overview

This document is the authoritative overview of the project's PDE problems, sampling rules, and evaluation sizes. The registry currently contains **26 executable problems**: 22 PINNacle-adapted problems and 4 external problems with analytical or manufactured solutions.

The code registry is the runtime source of truth and can be inspected at any time:

```powershell
python -m forge.experiments.run_search --list-problems
```

Unified launch entry point:

```powershell
python -m forge.run --problem <problem_id> --device cuda:0
```

Problem-configuration JSON files are optional. PINNacle problems directly reuse the equation operators, geometries, initial and boundary conditions, and reference data under `third_party/pinnacle`. External problems use the same `PhysicsProblem` registration interface. The Trainer and EvolutionController do not contain problem-ID-specific training branches.

## Complete problem list

In the table, the dimension column uses the format "spatial dimension; network input → output." Time-dependent problems include time in the network input.

| Canonical ID | Source | Equation family and task | Dimensions | Geometry/time domain | Main constraints and challenges |
| --- | --- | --- | --- | --- | --- |
| `allen_cahn_1d` | External | Allen–Cahn, forward problem | 1D; 2→1 | Interval × time, default `t∈[0,0.25]` | Initial condition; Dirichlet or periodic boundary; stiff interface |
| `burgers_1d` | PINNacle | Burgers, forward problem | 1D; 2→1 | `x∈[-1,1]`, `t∈[0,1]` | Initial condition and Dirichlet boundary; nonlinear advection and shock; `ν=0.01/π` |
| `burgers_2d` | PINNacle | Burgers system, forward problem | 2D; 3→2 | Rectangle × time | Initial and periodic constraints; two outputs and strong advection; `ν=0.001` |
| `darcy_flow_2d` | External | Variable-coefficient Darcy, forward problem | 2D; 2→1 | Unit square | Homogeneous Dirichlet boundary; heterogeneous coefficient; manufactured solution |
| `gray_scott_2d` | PINNacle | Gray–Scott reaction–diffusion, forward problem | 2D; 3→2 | Rectangle × `t∈[0,200]` | Initial condition; two outputs, stiffness, and long-time patterns |
| `heat_2d_complex_geometry` | PINNacle | Heat equation, forward problem | 2D; 3→1 | Multi-hole domain × time | Initial and Robin conditions; complex geometry |
| `heat_2d_long_time` | PINNacle | Heat equation, forward problem | 2D; 3→1 | Rectangle × `t∈[0,100]` | Initial and Dirichlet conditions; long-time dissipation |
| `heat_2d_multiscale` | PINNacle | Multiscale heat equation, forward problem | 2D; 3→1 | Rectangle × time | Initial and Dirichlet conditions; anisotropy and scale separation |
| `heat_2d_varying_coefficient` | PINNacle | Variable-coefficient heat equation, forward problem | 2D; 3→1 | Rectangle × `t∈[0,5]` | Initial and Dirichlet conditions; spatially and temporally varying coefficients |
| `heat_5d` | PINNacle | High-dimensional heat equation, forward problem | 5D; 6→1 | Five-dimensional ball × time | Initial and Neumann conditions; high-dimensional complex geometry |
| `heat_inverse_2d` | PINNacle | Heat equation, inverse problem | 2D; 3→2 | Rectangle × time | Dirichlet boundary and noisy point observations; joint field and unknown-quantity identification |
| `kovasznay_flow_2d` | External | Steady incompressible Navier–Stokes, forward problem | 2D; 2→3 | Rectangle | Analytical boundary and pressure anchor; velocity–pressure coupling |
| `kuramoto_sivashinsky_1d` | PINNacle | Kuramoto–Sivashinsky, forward problem | 1D; 2→1 | Periodic interval × time | Initial and periodic conditions; fourth order, stiff, and chaotic |
| `navier_stokes_2d_backstep` | PINNacle | Steady incompressible Navier–Stokes, forward problem | 2D; 2→3 | Backward-facing step with optional obstacle | Dirichlet boundary; complex geometry and velocity–pressure coupling |
| `navier_stokes_2d_C` | PINNacle | Steady incompressible Navier–Stokes, forward problem | 2D; 2→3 | Unit-square lid-driven cavity | Dirichlet and pointwise pressure constraints; `Re=100`, recirculation vortices, and velocity–pressure coupling |
| `navier_stokes_2d_long_time` | PINNacle | Transient incompressible Navier–Stokes, forward problem | 2D; 3→3 | Rectangle × `t∈[0,5]` | Initial and Dirichlet conditions; nonlinearity and long-time integration |
| `poisson_2d_classic` | PINNacle | Poisson, forward problem | 2D; 2→1 | Rectangle with a hole | Dirichlet boundary; global elliptic coupling and complex boundary |
| `poisson_2d_many_area` | PINNacle | Partitioned Poisson, forward problem | 2D; 2→1 | `[-10,10]²` split into 5×5 subdomains | Robin conditions; piecewise coefficients, interfaces, and multiple scales |
| `poisson_3d_complex_geometry` | PINNacle | Poisson, forward problem | 3D; 3→1 | Cube with a spherical hole | Neumann boundary; piecewise coefficients and complex geometry |
| `poisson_5d` | PINNacle | Poisson, forward problem | 5D; 5→1 | Five-dimensional unit hypercube | Dirichlet boundary; high-dimensional elliptic problem |
| `poisson_boltzmann_2d` | PINNacle | Poisson–Boltzmann, forward problem | 2D; 2→1 | Rectangle with four holes | Dirichlet boundary; nonlinear elliptic equation |
| `poisson_inverse_2d` | PINNacle supplemental | Poisson, inverse problem | 2D; 2→2 | Unit square | Dirichlet boundary and noisy point observations; parameter inversion |
| `shallow_water_2d` | External | Two-dimensional shallow-water conservation law, forward problem | 2D; 3→3 | Periodic unit square × short time | Initial and periodic conditions, positive water depth; two momentum equations and mass conservation |
| `wave_1d` | PINNacle | One-dimensional wave equation, forward problem | 1D; 2→1 | `[0,1]×[0,1]` | Spatial boundary, initial displacement, and initial velocity; oscillation and phase |
| `wave_2d_heterogeneous` | PINNacle | Heterogeneous two-dimensional wave equation, forward problem | 2D; 3→1 | Square × `t∈[0,5]` | Dirichlet and Neumann conditions; variable-coefficient propagation |
| `wave_2d_long_time` | PINNacle | Two-dimensional wave equation, forward problem | 2D; 3→1 | Rectangle × `t∈[0,100]` | Initial and Dirichlet conditions; long-time phase error |

By source, the registry contains 20 core PINNacle problems, 2 supplemental PINNacle problems, and 4 external analytical or manufactured-solution problems. By task, it contains 24 forward problems and 2 inverse problems.

## Unified sampling structure

An `AlgorithmSpec` declares three training-point pools instead of allocating a complete set of points to every individual loss:

```text
sampling.interior.n_points  → PDE interior-point pool
sampling.boundary.n_points  → boundary, periodic, observation, interface, and other non-initial constraints
sampling.initial.n_points   → initial-surface point pool
```

At runtime, the system first inspects the loss terms enabled by the current candidate, resolves each term to its unique coordinate source, and allocates each point pool across those sources exactly once. Consequently:

- Multiple PDE residuals share the same interior-point source; `interior.n_points` is not multiplied by the number of equations.
- Constraints on the same coordinate source share points. Distinct boundary, observation, or interface sources divide the boundary pool according to the weights and minimum counts declared by the problem.
- Disabled constraint losses are not sampled. Steady problems generally do not use the initial-point pool.
- The initial-displacement and initial-velocity conditions of `wave_1d` share the same initial coordinates but evaluate value and time-derivative constraints separately.
- Paired periodic points are counted by their actual tensor size. For example, a periodic source in `burgers_2d` has a point-count multiplier of 2.
- Fixed anchors such as a pressure gauge can be defined by the problem as a single point and do not consume an entire variable boundary-point pool.

This structure is shared across problems, but every PDE declares its own constraint set, sampling sources, sharing groups, fixed counts, minimum counts, and multipliers. The common element is the budget-accounting method, not a requirement that every PDE use the same number of points.

## Sampling counts and limits

All PDEs currently use the same resource defaults:

| Setting | Default | Meaning |
| --- | ---: | --- |
| `maximum_sampling_points` | `None` | No global limit on training sample points |
| `evaluation_size` | `None` | Use every valid reference row when PINNacle reference data exists; otherwise use the canonical deterministic size |
| `residual_evaluation_size` | `8192` | Independent PDE-residual evaluation points used by the CLI and PINNacle adapter |
| `low_fidelity_iterations` | `1000` | Training steps for each low-fidelity candidate |
| `final_iterations` | `10000` | Training steps for each candidate in the independent high-fidelity Top-3 review |

A run can explicitly override these values. For example, a launcher may use `--evaluation-size full --residual-evaluation-size 2048`, in which case its experiment contract records 2048 instead of the CLI default of 8192. A resumed run must retain the values in its original experiment contract.

The `n_points` fields require only non-negative integers; their schemas do not impose numerical upper bounds. By default, a larger count generated by an LLM is not rejected by a shared limit, although it may still fail because of insufficient GPU memory, system memory, or runtime. A user can set an explicit numerical limit:

```powershell
python -m forge.run --problem poisson_5d --maximum-sampling-points 50000
```

The Validator computes the actual total across every active sampling source, including periodic-pair multipliers and fixed points. A candidate above `50000` is marked as a budget-validation failure before construction or training. Use `--maximum-sampling-points unlimited` or `none` to restore an unlimited budget.

PINNacle profiles retain the original benchmark sampling sizes as problem priors and reproducibility references. They are not hard limits in this project:

| PINNacle reference group | Interior | Boundary | Initial | Original reference test points | Applicable problems |
| --- | ---: | ---: | ---: | ---: | --- |
| Base scale | 8192 | 2048 | 2048 for time-dependent / 0 for steady | 8192 | `burgers_1d`, `wave_1d`, `kuramoto_sivashinsky_1d`, `poisson_2d_classic`, `poisson_boltzmann_2d`, `poisson_2d_many_area`, `navier_stokes_2d_backstep`, `navier_stokes_2d_C`, `poisson_inverse_2d` |
| Extended scale | 32768 | 8192 | 8192 for time-dependent / 0 for steady | 32768 | The other 13 PINNacle problems |

External problems are not forced into either PINNacle reference scale. They read the same three point pools from the candidate `AlgorithmSpec`. The general sampler default is `1024` points per pool, and the LLM may select other values according to dimensionality, geometry, equation order, constraint count, and training budget.

### Poisson-5D

`poisson_5d` is a steady Dirichlet problem on a five-dimensional unit hypercube and has no initial surface:

```text
PINNacle reference: interior=32768, boundary=8192, initial=0, test=32768
Project hard limit by default: none
Default solution evaluation: all valid reference rows
Default residual evaluation points: 8192
```

Point counts for a high-dimensional problem should not be obtained by simply scaling a two-dimensional total. The LLM can balance coverage, low-discrepancy sampling, and GPU-memory cost. The project standardizes accounting and validation, but does not impose the same limit on every PDE.

## `evaluation_size` and `residual_evaluation_size`

Neither setting participates in backpropagation or belongs to the three training-point pools:

- `evaluation_size` controls solution metrics such as MSE, relative L2 error, and per-output error on reference solutions or test data. This path generally requires only a network forward pass. The default is `None`, which means all non-NaN reference rows for PINNacle problems with reference tables. When a positive integer is supplied, the adapter limits rows using deterministic evenly spaced indexes. `full`, `all`, and `none` all resolve to the default complete range.
- For a continuous analytical PINNacle domain without a finite reference table, `None` resolves to a canonical deterministic size: 2,500 points for input dimension 2 and 20,000 points for other input dimensions. Coordinates are sampled with `pseudo` interior sampling using fixed NumPy seed 0, and the exact requested count is returned.
- `residual_evaluation_size` controls recomputation of PDE residuals at independent coordinates. This may require first-, second-, or even fourth-order coordinate derivatives and usually consumes more time and GPU memory than solution evaluation at the same scale. The CLI default is 8192. The PINNacle adapter likewise uses fixed seed 0 and an exact number of pseudorandom interior points, then caches CPU coordinates on the problem instance so every candidate in the same run shares the identical residual-evaluator point set.

External analytical problems currently prefer their own fixed evaluation grids: Kovasznay `40×40`, Allen–Cahn `128×16`, Darcy `48×48`, and Shallow Water `20×20×8`. `evaluation_size` primarily controls subsampling of PINNacle reference data, while `residual_evaluation_size` continues to control the shared residual-diagnostic scale.

## Complex-geometry sampling and boundary normals

The evaluation path does not call the DeepXDE base class's `uniform_points`. This is especially important for CSG because `CSGDifference` does not implement a true uniform grid and its base class only prints a warning before falling back to random sampling. Space-time grids may also return more points than requested because of per-axis rounding. The project directly performs deterministic `random_points` sampling and validates rank, finiteness, and the exact point count, avoiding messages such as:

```text
CSGDifference.uniform_points not implemented
N points required, but M points sampled
```

Training constraints still use each geometry's own boundary sampler. For Neumann and Robin conditions, the adapter recursively traverses `GeometryXTime`, `geom1`, and `geom2`, collects the axis-aligned boundaries of nested `Rectangle`, `Cuboid`, and `Hypercube` geometries, and filters out points that lie on two or more faces. It then replenishes the batch through bounded incremental oversampling. This prevents CSG from passing vertices or edges with ambiguous normals to `boundary_normal`. Dirichlet conditions do not depend on normals and therefore skip this filtering.

## Important problem details

### Kuramoto–Sivashinsky 1D

The equation and PINNacle parameters are:

```text
u_t + α u u_x + β u_xx + γ u_xxxx = 0
α = 100/16 = 6.25
β = 100/16² = 0.390625
γ = 100/16⁴ = 0.00152587890625
x ∈ [0,2π], t ∈ [0,1]
u(x,0) = cos(x)(1+sin(x))
```

Spatial derivatives through third order satisfy periodic consistency. The reference data is stored at `third_party/pinnacle/ref/Kuramoto_Sivashinsky.dat` on an original grid of 512 spatial points × 251 time points.

## Python API, aliases, and extension

```python
from forge import get_problem

problem = get_problem("poisson_5d")
spec = problem.get_spec()
domain = problem.sample_domain(1024, seed=0, device="cuda")
constraints = problem.sample_constraints(256, seed=0, device="cuda")
```

PINNacle class names can be used as aliases, including `Burgers1D`, `Poisson2D_Classic`, and `NS2D_LidDriven`. The canonical ID for `NS2D_LidDriven` is `navier_stokes_2d_C`. Legacy project aliases such as `classic_burgers_1d`, `wave_c`, and `ks_1d` remain compatible, but new configurations and experiment records should use the canonical IDs in the table.

New problems can be integrated through `register_problem()`. A provider must define an immutable problem specification, domain sampling, constraint sampling, equation residuals, and reference/evaluation capabilities. Sampling budgets continue to use the shared three-pool allocator, so the generic Trainer does not need to be modified.
