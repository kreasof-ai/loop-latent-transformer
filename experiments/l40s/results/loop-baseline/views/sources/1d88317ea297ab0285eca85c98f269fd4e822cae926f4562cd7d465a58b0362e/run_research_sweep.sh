#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${LLT_RESULTS_ROOT:?Set a fresh run root}"
export LLT_RESULTS_ROOT
llt_cpu_python="${LLT_CPU_PYTHON:-python}"
llt_python="${LLT_PYTHON:-/home/sagemaker-user/tensor/.venv/bin/python}"
LLT_RESULTS_ROOT="$("$llt_cpu_python" -c 'from experiments.l40s.runtime import results_root,require_run_directory; require_run_directory(); print(results_root())')"
export TENSOR_NVRTC_HOME="${TENSOR_NVRTC_HOME:-/home/sagemaker-user/tensor/build/nvrtc-12.9}"
llt_out="$LLT_RESULTS_ROOT/research-baselines"
mkdir -p "$llt_out"
llt_run() {
 local llt_phase="$1" llt_model="$2" llt_loops="$3" llt_backend="$4" llt_policy="$5" llt_stem
 printf -v llt_stem '%s-%s-%s-t%02d-%s' "$llt_phase" "$llt_model" "$llt_backend" "$llt_loops" "$llt_policy"
 if [[ "${LLT_RESUME:-0}" == 1 && -f "$llt_out/$llt_stem.json" ]]; then
  "$llt_cpu_python" -c 'import json,sys; assert json.load(open(sys.argv[1]))["status"] in ("passed","out_of_memory")' "$llt_out/$llt_stem.json"
  return
 fi
 if ! "$llt_python" -m experiments.l40s.research_sweep "$llt_phase" --model "$llt_model" --loops "$llt_loops" --backend "$llt_backend" --policy "$llt_policy" > "$llt_out/$llt_stem.log" 2>&1; then
  "$llt_cpu_python" -m experiments.l40s.loop_sweep_record "$llt_out/$llt_stem.json"
 fi
 tail -n 1 "$llt_out/$llt_stem.log"
}
for llt_model in uyoco lpt grt per_layer_latent attention_only; do
 for llt_loops in 4 16; do llt_run qualification "$llt_model" "$llt_loops" tensor none; done
done
for llt_loops in {1..16}; do
 for llt_model in uyoco lpt grt per_layer_latent attention_only; do
  for llt_backend in tensor torch; do
   llt_run training "$llt_model" "$llt_loops" "$llt_backend" none
   llt_run training "$llt_model" "$llt_loops" "$llt_backend" ac
   llt_run inference "$llt_model" "$llt_loops" "$llt_backend" none
  done
 done
done
for llt_model in uyoco lpt grt per_layer_latent attention_only; do
 for llt_backend in tensor torch; do llt_run full_qualification "$llt_model" 16 "$llt_backend" none; done
done
