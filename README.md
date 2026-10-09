# Loop Latent Transformer

Loop Latent Transformer (LLT) is an architectural research prototype that reuses
Transformer blocks while attending to **one contextual low-rank memory per layer**.
Each memory is built from that layer's input during the first loop and reused on
later loops. Queries and full-width residuals evolve; historical memories remain
fixed. Projection folding avoids expanded K/V storage. KV storage is constant
in loop count, with L latent banks rather than a single global bank.

The evidence is a **kernel latency and memory study on NVIDIA L40S**, without a
trained-quality claim. Numerical kernels use [Tensor](https://github.com/kreasof-ai/tensor),
with matched PyTorch controls. The current sweep measures none and standard AC;
LAC is excluded. The globally shared embedding-latent prototype and its LAC
measurements remain [historical](archive/reports/L40S_GLOBAL_LATENT_SWEEP.md).

## Start here

- [Architecture](docs/ARCHITECTURE.md): equations, model variants, parameters, and cache scaling.
- [L40S loop sweep](experiments/l40s/LOOP_SWEEP.md): the main experiment, all loop counts, ranks, training/inference latency, and peak memory.
- [Checkpointing](docs/CHECKPOINTING.md): exact block AC and historical latent-region limitations.
- [Documentation map](docs/README.md): implementation and reproduction guides, research questions, and historical work.

The sweep covers B4/S1024, T=1..16, and LLT ranks 32/64/128 against Naive Loop,
independent stacks, and fixed-depth models matched to each stack's parameter count.
Additional controls cover U-YOCO, LPT, GRT, refreshed per-layer/per-loop latents,
and attention-only looping. The [architecture contracts](docs/RESEARCH_BASELINES.md)
identify source fidelity, precision and work-count differences.

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
python -m experiments.l40s.contextual_llt_report
```

Run the CPU model checks in a Python environment with Torch and Tensor's Torch
adapter installed:

```bash
python -m unittest discover -s tests -v
```

The [reproduction guide](docs/REPRODUCIBILITY.md) covers dependencies, pinned
measurement revisions, fresh GPU runs, and provenance. The published measurements
are preserved; new profiles use a separate `LLT_RESULTS_ROOT`.
