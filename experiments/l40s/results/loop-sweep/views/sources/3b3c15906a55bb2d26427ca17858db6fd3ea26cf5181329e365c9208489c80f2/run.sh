#!/usr/bin/env bash
# Sequential fresh-run reproduction. CPU-only regeneration is documented separately.
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${LLT_RESULTS_ROOT:?Set LLT_RESULTS_ROOT to a fresh output directory}"
export LLT_RESULTS_ROOT
llt_cpu_python="${LLT_CPU_PYTHON:-python}"
llt_python="${LLT_PYTHON:-/home/sagemaker-user/tensor/.venv/bin/python}"
LLT_RESULTS_ROOT="$("$llt_cpu_python" -c 'from experiments.l40s.runtime import results_root, require_run_directory; require_run_directory(); print(results_root())')"
export TENSOR_NVRTC_HOME="${TENSOR_NVRTC_HOME:-/home/sagemaker-user/tensor/build/nvrtc-12.9}"
experiments/l40s/run_loop_sweep.sh
experiments/l40s/run_loop_sweep_recovery.sh
experiments/l40s/run_latent_checkpoint_sweep.sh
"$llt_python" -m experiments.l40s.checkpoint_numerics_diagnostic
"$llt_python" -m experiments.l40s.checkpoint_numerics_diagnostic --deterministic
"$llt_cpu_python" -m experiments.l40s.loop_sweep_summary
"$llt_cpu_python" -m experiments.l40s.latent_checkpoint_summary
"$llt_cpu_python" -m experiments.l40s.latent_checkpoint_report
"$llt_cpu_python" -m experiments.l40s.loop_sweep_manifest
"$llt_cpu_python" -m experiments.l40s.latent_checkpoint_manifest
