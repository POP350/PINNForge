# Closed-loop search

PINNsForge uses an LLM to generate a complete `AlgorithmSpec`, rather than Python code or a partial patch to a fixed template. Before training, each specification undergoes field-level option validation, problem-capability checks, sampling-budget checks, and normalization.

## Main workflow

1. Load a PDE problem from the registry and extract features such as dimensionality, equation order, boundary conditions, and sampling capabilities.
2. Retrieve design context from prior knowledge and the current run's posterior memory.
3. Ask the design agent to generate a complete `AlgorithmSpec`.
4. Reject unregistered, incompatible, or over-budget options with the Validator.
5. Build the PyTorch PINN with the Builder and execute the specification with the Trainer.
6. Compute metrics at fixed evaluation points with the Evaluator, then let the evolution controller update the candidate archive and decide whether to continue, escape, or stop.
7. Write experimental events and candidate summaries to the run directory for recovery and posterior feedback within the current run.

## Executable options

The field-level option registry is located at `forge/pipeline/d_algorithm_generation/specs/option_registry.py`. Networks, activation functions, samplers, constraint enforcement, loss weighting, optimizers, and related fields are registered independently. The Validator never silently accepts an unknown name.

Each problem declares its supported geometry, constraints, reference data, and sampling capabilities. External problems can be integrated through `register_problem()` without modifying the search controller.

## Search contract

Candidate counts, low- and high-fidelity training budgets, stagnation and escape conditions, the global archive, and the final review strategy are defined by `experiment_contract.py`. These values evolve with the code, so this document does not duplicate volatile constants. Inspect the current executable contract with:

```powershell
python -m forge.experiments.run_search --dry-run --benchmark burgers_1d
```

During a formal run, every candidate is built independently and randomly initialized. Parent candidates provide structured design context only; model weights and checkpoints are not reused directly.

## Output and recovery

Run artifacts are written to `outputs/` by default:

- `metrics` retains only core metrics.
- `compact` additionally retains summaries, candidate indexes, and recovery state.
- `diagnostic` retains complete training traces, model responses, and diagnostic artifacts.

Use `--resume` with an existing directory only when the code, parameters, and experiment-contract version are identical. All run artifacts and model checkpoints are excluded by `.gitignore`.
