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

## Regenerate the current report on CPU

Leave `LLT_RESULTS_ROOT` unset and run:

```bash
python -m experiments.l40s.contextual_llt_report
```

This audits the 288 replacement LLT records and joins them with 768 preserved
non-LLT records. It regenerates the canonical report, active CSVs and LLT figures.
Historical summary/report tools remain available for their original source families;
the report entry point delegates to the current generator when its audit exists.
Raw historical records and their source snapshots remain unchanged.

Strict artifact auditing needs the ignored CUDA artifact caches on the measurement
machine. They are not included in Git. A fresh clone can read the committed raw
records, audits, source snapshots and figures; repeating the strict audit needs
the original artifact caches. Missing artifacts cause audit failure.

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

## Fresh contextual LLT sweep

Use a new output root and run sequentially on an isolated NVIDIA L40S:

```bash
export LLT_RESULTS_ROOT=/tmp/llt-contextual-fresh-l40s
export LLT_PYTHON=/home/sagemaker-user/tensor/.venv/bin/python
export LLT_CPU_PYTHON=/opt/conda/bin/python
export TENSOR_CHECKOUT=/home/sagemaker-user/tensor
export TENSOR_NVRTC_HOME=/home/sagemaker-user/tensor/build/nvrtc-12.9
bash experiments/l40s/run_contextual_llt_sweep.sh
python -m experiments.l40s.contextual_llt_report
```

These installation paths can be changed for another machine; the dependency is
[Tensor](https://github.com/kreasof-ai/tensor). The runner first checks all ranks
at small T4/T16, measures 288 fresh performance cases, then checks every rank on
both backends at full B4/S1024 T16. No LAC worker or policy is run.

The geometry is W768/H12/L12/V50304/P1025, T1..16, rank32/64/128. Each layer
builds memory from its first-loop input and reuses it on later loops. Training
profiles none/AC. Inference profiles prompt, startup and one supplied cached token.
Three warmups, nine samples and three graph replays match the historical protocol.
Full-size numerical checks use deterministic controls separately from timing.

`LLT_RESUME=1` resumes only with identical numerical sources and runtime.
Each job uses a fresh process. OOM is retained as a measured outcome. Do not mix
source revisions in one campaign or run concurrent GPU benchmarks. GPU workers
reject the published output root. The report generator writes the canonical
report only after auditing the complete replacement grid.

For the historical global-latent sweep, use its frozen revisions and `run.sh`.

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
