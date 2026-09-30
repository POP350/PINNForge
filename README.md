# PINNForge

The Python package is named `forge`. The 26 currently registered PDEs, their sampling rules, and evaluation sizes are documented in the [PDE problem overview](docs/pde_problems.md). The unified entry point is `python -m forge.run --problem <id>`.

PINNForge uses an LLM to generate complete `AlgorithmSpec` objects, validates their credibility, builds and trains PyTorch PINNs, evaluates them numerically, and closes the loop through population updates and experimental knowledge feedback.

Built-in problems are provided directly by `third_party/pinnacle`. External generalization problems use the same open registry. The project no longer maintains duplicate handwritten PINNacle PDEs, legacy benchmarks, or legacy adapters; additional sources can be integrated through `register_problem()`.

## Project structure

The main code follows the explicit A–I stages under `forge/pipeline/`:

```text
forge/
├─ pipeline/
│  ├─ a_problem_definition/       # PINNacle adapter and extensible problem registry
│  ├─ b_feature_extraction/        # Retrieval and generation features from problem definitions
│  ├─ c_knowledge_retrieval/       # Prior/posterior knowledge and PINNacle metadata
│  ├─ d_algorithm_generation/      # LLM, AgenticVariationOperator, and AlgorithmSpec
│  ├─ e_credibility_assessment/    # Spec validation, normalization, and budget checks
│  ├─ f_pinn_construction/         # PyTorch networks, losses, samplers, and Builder
│  ├─ g_training_evaluation/       # Formal training and numerical metrics
│  ├─ h_evolution_reflection/      # Population, evolution control, and ReflectionAgent
│  └─ i_artifact_feedback/         # Experimental events and candidate-record feedback
├─ experiments/                    # Closed-loop entry point and CLI
└─ utils/                          # Shared utilities

third_party/pinnacle/              # Minimal PINNacle runtime, compatibility layer, and reference data
docs/                              # Usage documentation
```

Historical training output is not committed. Every run writes its artifacts to `outputs/`, which is excluded by `.gitignore`.

## Quick start

```powershell
pip install -r requirements.txt

Copy-Item .env.example .env.local
# Edit .env.local and provide the model and API key for your OpenAI-compatible service.

python -m forge.experiments.run_search `
  --benchmark burgers_1d `
  --llm-provider openai `
  --llm-model YOUR_MODEL `
  --device cuda:0
```

The same search can be started through the unified entry point without a JSON configuration:

```powershell
python -m forge.run `
  --problem burgers_1d `
  --device cuda:0
```

For reproducible server runs, use a new `--output-dir` whenever the experiment contract or runtime code changes. Use `--resume` with an existing directory only when the code, parameters, and `experiment_contract_version` are unchanged; otherwise, an old `search_state.json` may restore candidates and training state from an earlier run.

## Validation and runtime safety boundaries

- The field-level Option Registry is the sole source of executable options. The Validator rejects unregistered options before the Builder or Trainer runs instead of silently skipping unknown names. `training.time_strategy.name` currently supports only `none` and `time_window_curriculum`. For example, `linear_ramp` is recorded as `validation_failed/UNREGISTERED_OPTION` and handled by the existing candidate-correction and replenishment policy.
- Adaptive rollback preserves the state of every optimization stage, but a repeated active optimizer state can be restored only to the stage of the same name that created it. This prevents Adam parameter groups from being loaded into L-BFGS. When L-BFGS produces a non-finite loss, gradient, or parameter, the Trainer first restores the model and loss state from before that step. If an eligible Adam-family stage exists in the plan, the remaining second-order budget is converted to `adaptive_recovery_adam`. A candidate is marked `training_failed` only if the recovery stage also produces non-finite values.
- When PINNacle reference data is available, solution metrics use all valid reference rows or a user-specified limit. When no finite reference table exists, evaluation and residual diagnostics use an exact number of deterministic pseudorandom interior points with a fixed seed. This avoids the unimplemented CSG `uniform_points` path and warnings caused by requesting a different number of points than a grid returns.
- Neumann and Robin constraint sampling recursively examines `GeometryXTime` and CSG child geometries. It excludes vertices and edges with ambiguous normals from box-like geometries and then replenishes the requested batch. The project does not alter PDE equations or suppress runtime warnings to avoid this issue.

`termination_reason=escape_failed` means that a low-fidelity search detected stagnation and the escape generation did not satisfy the improvement criteria required to resume the normal search. The run still proceeds to the final high-fidelity review; this state does not indicate a process, training, or experiment failure. PyTorch may print a one-time cuBLAS CUDA-context initialization message during the first backward pass. PyTorch handles it automatically and it does not change the training result.

## Search contract and output

The experiment contract in the code defines search budgets, candidate counts, stagnation conditions, and the final review strategy. Inspect the current contract without model calls or training:

```powershell
python -m forge.experiments.run_search --dry-run --benchmark burgers_1d
```

Candidates always undergo normalization and executability validation. Low-fidelity search and final review use independently constructed models and do not reuse parent weights. Output is written to `outputs/` by default and is not committed to Git. Use `--output-detail compact` to retain search summaries and recovery state, or `--output-detail diagnostic` to retain complete diagnostic artifacts.

See [Closed-loop search](docs/direct_llm_search.md) and the [PDE problem overview](docs/pde_problems.md) for more information.

## License

PINNForge is licensed under the [Apache License 2.0](LICENSE). The vendored PINNacle subset under `third_party/pinnacle` retains its upstream MIT License; see `third_party/pinnacle/LICENSE`.
