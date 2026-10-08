# Loop Latent Transformer

Loop Latent Transformer (LLT) is an architectural research prototype that reuses
a small stack of Transformer blocks while attending to a shared, low-rank KV
latent derived from the input embeddings. Queries and residual states evolve
through the loops; the historical KV latent stays fixed. Projection folding
avoids materializing full-width keys and values.

The current evidence is a **kernel latency and memory study on NVIDIA L40S**.
It does not yet establish trained language-model quality. Numerical kernels use
[Tensor](https://github.com/kreasof-ai/tensor), with matched PyTorch controls.

## Start here

- [Architecture](docs/ARCHITECTURE.md): equations, model variants, parameters, and cache scaling.
- [L40S loop sweep](experiments/l40s/LOOP_SWEEP.md): the main experiment, all loop counts, ranks, training/inference latency, and peak memory.
- [Checkpointing](docs/CHECKPOINTING.md): exact block AC and the preliminary LLT latent-region policy.
- [Documentation map](docs/README.md): implementation and reproduction guides, research questions, and historical work.

The sweep covers batch 4, sequence 1024, T=1..16, and LLT ranks 32/64/128 against
Naive Loop, independent stacks, and fixed-depth models that match each stack's
parameter count. It contains 672 performance records (664 passed, 8 OOM), plus
backend and checkpoint correctness checks.

At rank 64 and T=16, captured Tensor training takes **518.51 ms / 4.79 GiB with
block AC**, compared with **411.77 ms / 25.11 GiB without checkpoints**. The current
LAC checkpoints only the latent attention/output branch and remains exploratory;
it is not a completed solution for checkpointing whole loops through latent state.
See the main report for the full protocol, Torch comparisons, and inference.

## Repository map

| Path | Purpose |
|---|---|
| [model/](model/README.md) | Torch reference, Tensor backend adapter, and checkpoint policies |
| [inference/](inference/README.md) | Prepared serving weights; cache/decode implementation lives with the model |
| [experiments/l40s/](experiments/l40s/README.md) | Main study, sequential profiling harnesses, and CPU report tools |
| [experiments/l40s/results/](experiments/l40s/results/README.md) | Frozen measured records, source snapshots, and regenerated views |
| [docs/](docs/README.md) | Architecture, checkpoint boundaries, reproducibility, and research notes |
| [tests/](tests/README.md) | CPU checks for model behavior and result preservation |
| [archive/](archive/README.md) | Initial proposal, earlier studies, and retired experiment code |

## Read or reproduce

Reading the report needs no environment setup. Regenerate its audited tables and
figures on CPU with Python and Matplotlib:

```bash
python -m experiments.l40s.loop_sweep_summary
python -m experiments.l40s.latent_checkpoint_summary
python -m experiments.l40s.latent_checkpoint_report
python -m experiments.l40s.loop_sweep_manifest
python -m experiments.l40s.latent_checkpoint_manifest
```

Run the CPU model checks in a Python environment with Torch and Tensor's Torch
adapter installed:

```bash
python -m unittest discover -s tests -v
```

The [reproduction guide](docs/REPRODUCIBILITY.md) covers dependencies, pinned
measurement revisions, fresh GPU runs, and provenance. The published measurements
are preserved; new profiles use a separate `LLT_RESULTS_ROOT`.
