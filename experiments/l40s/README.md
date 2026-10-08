# L40S experiment

[LOOP_SWEEP.md](LOOP_SWEEP.md) is the project's main measured result. Read it first
for training/inference latency and peak memory at B4/S1024, T=1..16, and LLT ranks
32/64/128. Earlier kernel and actual nanoGPT studies are in the [archive](../../archive/README.md).

## Harness map

| Files | Purpose |
|---|---|
| [loop_sweep.py](loop_sweep.py), [run_loop_sweep.sh](run_loop_sweep.sh) | Original 256-case rank-64 and conventional baseline grid |
| [latent_checkpoint_sweep.py](latent_checkpoint_sweep.py), [run_latent_checkpoint_sweep.sh](run_latent_checkpoint_sweep.sh) | 416 additional rank/AC/experimental-LAC cases and checkpoint qualification |
| [research_sweep.py](research_sweep.py), [run_research_sweep.sh](run_research_sweep.sh) | Additional recurrent/cache-sharing architecture profiles |
| [research_summary.py](research_summary.py), [research_report.py](research_report.py) | Added-family audit, joined CSV, plots and canonical report extension |
| [full_latent_qualification.py](full_latent_qualification.py) | Repeatable full-size gradient controls |
| [checkpoint_numerics_diagnostic.py](checkpoint_numerics_diagnostic.py) | Torch BF16 repeated-backward diagnostics |
| [loop_sweep_recovery.py](loop_sweep_recovery.py), [run_loop_sweep_recovery.sh](run_loop_sweep_recovery.sh) | Separate retries for graph capture after eager-gradient release |
| [loop_sweep_record.py](loop_sweep_record.py) | Preserve and classify allocator OOM failures |
| [timing.py](timing.py) | CUDA-event, synchronized wall time, and allocator sampling |
| [loop_sweep_summary.py](loop_sweep_summary.py), [latent_checkpoint_summary.py](latent_checkpoint_summary.py) | Strict CPU audits, tables, CSV exports, and figures |
| [latent_checkpoint_report.py](latent_checkpoint_report.py) | Single current report generator |
| [manifest.py](manifest.py), [loop_sweep_manifest.py](loop_sweep_manifest.py), [latent_checkpoint_manifest.py](latent_checkpoint_manifest.py) | Provenance for regenerated views |
| [runtime.py](runtime.py) | Output roots and resolution of original recorded paths |
| [run.sh](run.sh) | Sequential fresh-run pipeline |

Model code lives in [model/](../../model/README.md), and serving helpers in
[inference/](../../inference/README.md). Experiment harnesses select geometry and
measurement policy rather than defining the architecture.

The scripts work both as direct files and as `python -m experiments.l40s.<name>`.
Report tools run on CPU with Matplotlib. GPU workers use Torch and Tensor's native
Torch adapter. See [reproducibility](../../docs/REPRODUCIBILITY.md) for dependencies,
commands, and the distinction between original sources and the current layout.

## Results

The [results index](results/README.md) explains frozen raw payloads, source
snapshots, original manifests, and regenerated `views/`. No new GPU timings were
collected during the repository reorganization. New GPU runs require a separate
`LLT_RESULTS_ROOT`; the report generator writes a fresh-run report inside that
run's `views/` rather than replacing the published study.
