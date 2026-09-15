# PINNacle runtime subset

This directory contains the minimal vendored PINNacle runtime used by
Forge. It is not a standalone copy of the upstream training project.

The retained files provide:

- PINNacle PDE equations and boundary/initial-condition definitions in `src/pde`;
- the model primitive required by the inverse-PDE definitions in `src/model/fnn.py`;
- geometry, coefficient, and tensor helpers required by those PDEs in `src/utils`;
- the PyTorch DeepXDE compatibility layer in `deepxde`;
- the default reference dataset selected by every problem registered in
  `forge.pipeline.c_knowledge_retrieval.pinnacle_registry`.

Upstream experiment launchers, optimizers, plotting/reporting utilities, random-search
artifacts, and unselected reference-data variants are intentionally omitted. All
PINNacle problems must be launched through Forge's normal entry points rather
than the upstream PINNacle CLI.

The upstream license is retained in `LICENSE`.
