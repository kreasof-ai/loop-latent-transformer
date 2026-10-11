#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
repo="$PWD"
tensor_repo="${TENSOR_CHECKOUT:-$repo/../tensor}"
export TENSOR_CHECKOUT="$tensor_repo"
python="${LLT_PYTHON:-python}"
uv build --wheel --no-build-isolation "$tensor_repo" --out-dir "$repo/build/llt-wheels"
LLT_TENSOR_BUILD_NATIVE=1 MAX_JOBS=2 uv build --wheel --no-build-isolation packages/llt-tensor --out-dir build/llt-wheels
uv venv build/llt-consumer --python "$python" --clear
uv pip install --python build/llt-consumer/bin/python numpy==2.5.3 torch==2.14.0 build/llt-wheels/*.whl
mkdir -p build/llt-qualification/artifacts
cp -n build/llt-tests/*.tbin build/llt-qualification/artifacts/
mkdir -p build/llt-consumer/standalone
cp benchmarks/llt/consumer.py build/llt-consumer/standalone/consumer.py
cp benchmarks/llt/model.py build/llt-consumer/standalone/model_fixture.py
cd build/llt-consumer/standalone
../bin/python consumer.py "$repo/build/llt-qualification/artifacts" "$repo/docs/research/data/llt-readiness/consumer.json"
cd "$repo"
"$python" -m benchmarks.llt.audit
