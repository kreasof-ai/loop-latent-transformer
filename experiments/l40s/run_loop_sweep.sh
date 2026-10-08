#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export TENSOR_NVRTC_HOME="${TENSOR_NVRTC_HOME:-/home/sagemaker-user/tensor/build/nvrtc-12.9}"
llt_python="${LLT_PYTHON:-/home/sagemaker-user/tensor/.venv/bin/python}"
llt_out=benchmarks/results/l40s-loop-sweep
mkdir -p "$llt_out"
llt_run() {
 local llt_phase="$1" llt_model="$2" llt_backend="$3" llt_loops="$4" llt_stem
 printf -v llt_stem '%s-%s-%s-t%02d' "$llt_phase" "$llt_model" "$llt_backend" "$llt_loops"
 if [[ "${LLT_RESUME:-0}" == 1 && -f "$llt_out/$llt_stem.json" ]]; then return; fi
 "$llt_python" experiments/l40s/loop_sweep.py "$llt_phase" --model "$llt_model" --backend "$llt_backend" --loops "$llt_loops" > "$llt_out/$llt_stem.log" 2>&1
 tail -n 1 "$llt_out/$llt_stem.log"
}
# Qualification first; all cases execute sequentially in isolated CUDA processes.
for llt_model in llt naive_loop stacked fixed_depth; do
 for llt_loops in 4 16; do llt_run correctness "$llt_model" tensor "$llt_loops"; done
done
for llt_loops in {1..16}; do
 for llt_model in llt naive_loop stacked fixed_depth; do
  for llt_phase in training inference; do
   for llt_backend in tensor torch; do
    llt_run "$llt_phase" "$llt_model" "$llt_backend" "$llt_loops"
   done
  done
 done
done
