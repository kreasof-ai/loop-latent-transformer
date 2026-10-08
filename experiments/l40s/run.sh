#!/usr/bin/env bash
set -euo pipefail
llt_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$llt_root"
export TENSOR_NVRTC_HOME="${TENSOR_NVRTC_HOME:-$llt_root/../tensor/build/nvrtc-12.9}"
llt_python="${LLT_PYTHON:-$llt_root/../tensor/.venv/bin/python}"
llt_results="$llt_root/benchmarks/results/l40s"
mkdir -p "$llt_results"
"$llt_python" experiments/l40s/check.py > "$llt_results/correctness.log" 2>&1
"$llt_python" experiments/l40s/systems.py attention --samples 9 > "$llt_results/attention.log" 2>&1
"$llt_python" experiments/l40s/systems.py attention --serial --samples 9 > "$llt_results/attention-serial.log" 2>&1
"$llt_python" experiments/l40s/decode_sweep.py --samples 9 > "$llt_results/decode-partitions.log" 2>&1
"$llt_python" experiments/l40s/systems.py inference --samples 9 > "$llt_results/inference.log" 2>&1
"$llt_python" experiments/l40s/prefill.py --samples 9 > "$llt_results/prefill.log" 2>&1
"$llt_python" experiments/l40s/systems.py training --samples 9 > "$llt_results/training.log" 2>&1
"$llt_python" experiments/l40s/repeat.py --samples 9 > "$llt_results/repeat.log" 2>&1
"$llt_python" experiments/l40s/summarize.py
"$llt_python" experiments/l40s/audit.py
