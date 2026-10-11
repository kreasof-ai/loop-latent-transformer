#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
tensor_repo="${TENSOR_CHECKOUT:-$PWD/../tensor}"
export TENSOR_CHECKOUT="$tensor_repo"
export TENSOR_NVRTC_HOME="${TENSOR_NVRTC_HOME:-$tensor_repo/build/nvrtc-12.9}"
export TENSOR_LLT_CUDA=1
python="${LLT_PYTHON:-python}"
mkdir -p build/llt-qualification
"$python" -m pytest -o addopts= -q packages/llt-tensor/tests
"$python" -m benchmarks.llt.qualify cold
"$python" -m benchmarks.llt.qualify gradients
"$python" -m benchmarks.llt.qualify training --steps 1000 --lr 0.001
"$python" -m benchmarks.llt.leak
for phase in generation attention decode loss-memory systems; do
    "$python" -m benchmarks.llt.scaling "$phase"
done
