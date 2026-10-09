# Historical work

This archive preserves the project's earlier proposals, prototypes, and small
studies. The current architectural description is in
[docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md), and the main measured result is
[LOOP_SWEEP.md](../experiments/l40s/LOOP_SWEEP.md).

## Reading map

| Material | Contents |
|---|---|
| [Initial proposal](PROPOSAL.md) | Original motivation, hypotheses, sketches, and early CPU results |
| [Early native-model specification](research/NATIVE_MODEL.md) | Initial model design and historical controls |
| [Benchmark notes](reports/README.md), [regime notes](reports/REGIMES.md) | Small CPU/Vulkan studies and crossover explorations |
| [Initial L40S kernels](reports/L40S.md) | Early CUDA kernel experiments |
| [Native model study](reports/L40S_NATIVE.md) | Earlier implementation profiles |
| [Actual nanoGPT study](reports/L40S_NANOGPT.md) | Historical nanoGPT comparison, with its vendor source preserved |
| [Global input-latent sweep](reports/L40S_GLOBAL_LATENT_SWEEP.md) | Superseded global-cache LLT ranks and historical LAC, with all original measurements |
| [Original rank-64 loop report](reports/L40S_LOOP_SWEEP.md) | Baseline study before the combined rank/checkpoint report |
| [Earlier checkpoint report](reports/L40S_LATENT_CHECKPOINT.md) | Previous presentation of the combined study; superseded by LOOP_SWEEP.md |
| [Retired benchmark scripts](benchmarks/) | Earlier CPU/Vulkan and regime harnesses |
| [Retired L40S scripts](experiments/l40s/) | Early CUDA/actual nanoGPT harnesses and vendor snapshot |
| [Earlier raw results](results/) | Measured JSON, source snapshots, diagnostics, and exports |

Archived scripts retain their historical source bytes and may depend on the old
layout. They are retained for inspection, not maintained as current entry points.
To rerun a historical study, use its pinned measurement revision and recorded
source snapshots. See [reproducibility](../docs/REPRODUCIBILITY.md).

The active sweep's measured payloads live beside that experiment, under
[experiments/l40s/results/](../experiments/l40s/results/README.md). Their original
path prefixes and those of archived raw results are recorded in the
[relocation map](../experiments/l40s/results/relocation.json).
