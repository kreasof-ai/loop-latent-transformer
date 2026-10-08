#!/usr/bin/env bash
set -euo pipefail
llt_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$llt_root"
: "${LLT_PYTHON:?Set LLT_PYTHON to the Python with the qualified Tensor adapter installed}"
: "${TENSOR_CHECKOUT:?Set TENSOR_CHECKOUT to the pinned Tensor installation used for provenance}"
: "${TENSOR_NVRTC_HOME:?Set TENSOR_NVRTC_HOME to the bootstrapped NVRTC installation}"
llt_results="$llt_root/benchmarks/results/l40s-native"
mkdir -p "$llt_results"
for llt_phase in correctness training inference audit; do
    "$LLT_PYTHON" experiments/l40s/native_study.py "$llt_phase" --samples 9 > "$llt_results/$llt_phase.log" 2>&1
done
"$LLT_PYTHON" experiments/l40s/native_summary.py
