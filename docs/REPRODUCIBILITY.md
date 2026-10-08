# Reproducing the L40S study

[LOOP_SWEEP.md](../experiments/l40s/LOOP_SWEEP.md) defines the measured protocol.
Use this guide to regenerate views, verify preserved payloads, or start a fresh
GPU run. Reading the committed report and raw JSON needs no GPU environment.

## Environments

The original GPU study used an NVIDIA L40S, Python 3.12, Torch 2.14.0+cu130,
CUDA 12.9 NVRTC, and Tensor's native Torch executor. Exact runtime, driver, source,
and artifact identities are retained in individual JSON records and the original
[baseline](../experiments/l40s/results/loop-baseline/run-manifest.json) and
[extension](../experiments/l40s/results/loop-sweep/run-manifest.json) manifests.
The original rank/checkpoint extension pins Tensor commit `7fc6b8cf9c7a2e58a55d9f8ba7e7dc1a990d87eb`;
the baseline has its own earlier Tensor revision and the same runtime binary hashes.

Follow the native executor installation instructions in
[Tensor](https://github.com/kreasof-ai/tensor) for its supported environment.
LLT backend imports require its `tensor_torch` adapter. The pure Torch reference
in `model.reference` requires only Torch. CPU report regeneration requires
Matplotlib and the Python standard library, without importing the GPU backend.
This repository does not install or develop the Tensor runtime.

## Regenerate published views on CPU

From the repository root, leave `LLT_RESULTS_ROOT` unset:

```bash
python -m experiments.l40s.loop_sweep_summary
python -m experiments.l40s.latent_checkpoint_summary
python -m experiments.l40s.research_summary
python -m experiments.l40s.latent_checkpoint_report
python -m experiments.l40s.loop_sweep_manifest
python -m experiments.l40s.latent_checkpoint_manifest
```

The three summary commands strictly audit the raw records, source snapshots, and
compiled artifact hashes before generating CSVs/tables/figures in `*/views/`.
The report command rebuilds the single canonical LOOP_SWEEP.md. Manifest commands
record current view-generator source hashes while preserving original manifests.
These commands perform no GPU measurements or Tensor compilation.

Strict artifact auditing needs the ignored CUDA artifact caches that accompanied
the original runs. They are present on the measurement machine and are not
included in Git. A fresh clone retains the committed raw records, source snapshots,
original audits, and generated report, but needs those artifacts to repeat the
strict audit. Missing artifacts cause a failure rather than weakening verification.

## Verify the reorganization

```bash
python -m unittest discover -s tests -v
```

Tests verify all 1023 relocated tracked payload/source files against their recorded
SHA256, check old-path resolution, compare the moved model implementation with
frozen source snapshots, and exercise exact checkpoint and cached-decode behavior
on CPU. Model checks use an environment with Torch and Tensor's Torch adapter.

Raw files still contain the absolute paths used when measured. The
[relocation map](../experiments/l40s/results/relocation.json) translates old prefixes;
[runtime.py](../experiments/l40s/runtime.py) resolves artifacts without modifying
measured JSON. New views use current paths. Original measurement manifests and
regenerated-view manifests are deliberately distinct provenance records.

## Fresh GPU sweep with the current layout

Set a separate output directory and the installed GPU/CPU Python executables.
On the existing measurement machine:

```bash
export LLT_RESULTS_ROOT=/tmp/llt-fresh-l40s
export LLT_PYTHON=/home/sagemaker-user/tensor/.venv/bin/python
export LLT_CPU_PYTHON=python
export TENSOR_CHECKOUT=/home/sagemaker-user/tensor
export TENSOR_NVRTC_HOME=/home/sagemaker-user/tensor/build/nvrtc-12.9
experiments/l40s/run.sh
```

These paths select an installation; the numerical dependency is the
[Tensor repository](https://github.com/kreasof-ai/tensor). Change the paths for
another machine. `run.sh` runs GPU jobs sequentially in fresh processes: original
256 cases and eight backend checks, separate capture retries, 416 extension cases,
12 small/12 full checkpoint checks, and BF16 diagnostics. It then runs CPU audits
and writes a report inside the new run's `loop-sweep/views/`. It leaves the
published report and measured payloads in place.

The grid uses B4/S1024, width 768, 12 heads, 12 base blocks, vocabulary 50,304,
position capacity 1025, T=1..16, and LLT ranks 32/64/128. Training uses full-token
loss and all model parameters. There are three warmups, nine timing samples,
and three graph replays per sample; optimizer counters check 42 actual updates.
Use the main report for exact precision, memory, and inference definitions.

`LLT_RESUME=1` resumes an interrupted run with the same sources and runtime.
Do not mix numerical revisions in one output directory: the strict audit checks
source-family consistency. A recorded OOM is a result; supplementary capture
retries remain separate. The published output root is rejected by GPU workers.
Do not run concurrent GPU benchmarks when reproducing isolated measurements.

## Reproduce the original source layout

The published data came from the revisions named in its manifests, before this
reorganization. For a historical rerun, create a separate worktree at the
published pre-reorganization revision:

```bash
git worktree add /tmp/llt-original d4fb963011c16528b7780867b2c21069d84a46c5
```

Inspect the study's source snapshots and manifests there and use that revision's
commands and layout. The extension's numerical source commit is
`36e40d7be01ed3be942689efa109f858fbe4a1f2`; the baseline pins its earlier source
revision. Current package/report changes do not relabel either measured source
family as current HEAD. A fresh run records its own source hashes.

## Additional architecture families

The [source/adaptation contract](RESEARCH_BASELINES.md) covers U-YOCO, LPT, GRT,
a per-layer latent control and an attention-only loop control. Reproduce their
480-case grid separately using `run_research_sweep.sh` with a fresh `LLT_RESULTS_ROOT`.
It uses the same B4/S1024, T=1..16, training none/AC and three inference operations.
The audit joins the new records with the original 672; their raw payloads and
measurement manifests remain separate. The added-family manifest pins its model,
worker and Tensor sources independently of the original rank/checkpoint study.
