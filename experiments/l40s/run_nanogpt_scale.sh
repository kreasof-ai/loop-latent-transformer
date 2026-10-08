#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export TENSOR_NVRTC_HOME="${TENSOR_NVRTC_HOME:-/home/sagemaker-user/tensor/build/nvrtc-12.9}"
llt_python="${LLT_PYTHON:-/home/sagemaker-user/tensor/.venv/bin/python}"
llt_out=benchmarks/results/l40s-nanogpt
mkdir -p "$llt_out"
llt_run() {
 local llt_stem="$1"
 shift
 if [[ "${LLT_RESUME:-0}" == 1 && -f "$llt_out/$llt_stem.json" ]]; then return; fi
 "$llt_python" experiments/l40s/nanogpt_scale.py "$@" > "$llt_out/$llt_stem.log" 2>&1
}
for llt_model in llt naive nanogpt; do
 for llt_loops in 1 4; do
  if [[ "$llt_model" == nanogpt && "$llt_loops" != 1 ]]; then continue; fi
  llt_run "correctness-$llt_model-t$llt_loops" correctness --model "$llt_model" --loops "$llt_loops"
 done
done
for llt_batch in 1 4; do
 for llt_model in llt naive nanogpt; do
  for llt_loops in 1 4; do
   if [[ "$llt_model" == nanogpt && "$llt_loops" != 1 ]]; then continue; fi
   for llt_policy in none loop; do
    if [[ "$llt_policy" == loop && "$llt_loops" != 4 ]]; then continue; fi
    llt_run "training-$llt_model-torch-b$llt_batch-t$llt_loops-$llt_policy-fused" training --model "$llt_model" --backend torch --loops "$llt_loops" --batch "$llt_batch" --policy "$llt_policy" --torch-optimizer fused
   done
  done
 done
done
for llt_batch in 1 4; do
 for llt_model in llt naive nanogpt; do
  for llt_loops in 1 4; do
   if [[ "$llt_model" == nanogpt && "$llt_loops" != 1 ]]; then continue; fi
   for llt_backend in torch tensor; do
    for llt_phase in inference training; do
     llt_run "$llt_phase-$llt_model-$llt_backend-b$llt_batch-t$llt_loops-none" "$llt_phase" --model "$llt_model" --backend "$llt_backend" --loops "$llt_loops" --batch "$llt_batch"
     if [[ "$llt_phase" == training && "$llt_loops" == 4 ]]; then
      llt_run "training-$llt_model-$llt_backend-b$llt_batch-t4-loop" training --model "$llt_model" --backend "$llt_backend" --loops "$llt_loops" --batch "$llt_batch" --policy loop
     fi
    done
   done
  done
 done
done
for llt_batch in 1 4; do
 for llt_backend in torch tensor; do
  if [[ "$llt_backend" == torch ]]; then
   llt_run "training-nanogpt-torch-b$llt_batch-t1-none-fused-graph" training --model nanogpt --backend torch --batch "$llt_batch" --torch-optimizer fused --training-graph
  else
   llt_run "training-nanogpt-tensor-b$llt_batch-t1-none-graph" training --model nanogpt --backend tensor --batch "$llt_batch" --training-graph
  fi
 done
done
