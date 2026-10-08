#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export TENSOR_NVRTC_HOME="${TENSOR_NVRTC_HOME:-/home/sagemaker-user/tensor/build/nvrtc-12.9}"
llt_python="${LLT_PYTHON:-/home/sagemaker-user/tensor/.venv/bin/python}"
llt_out=benchmarks/results/l40s-latent-checkpoint-sweep
mkdir -p "$llt_out"
llt_run() {
 local llt_phase="$1" llt_model="$2" llt_rank="$3" llt_loops="$4" llt_backend="$5" llt_policy="$6" llt_stem llt_worker
 printf -v llt_stem '%s-%s-%s-r%03d-t%02d-%s' "$llt_phase" "$llt_model" "$llt_backend" "$llt_rank" "$llt_loops" "$llt_policy"
 if [[ "${LLT_RESUME:-0}" == 1 && -f "$llt_out/$llt_stem.json" ]]; then
  "${LLT_CPU_PYTHON:-python}" -c 'import json,sys; assert json.load(open(sys.argv[1]))["status"] in ("passed","out_of_memory")' "$llt_out/$llt_stem.json"
  return
 fi
 llt_worker=experiments/l40s/latent_checkpoint_sweep.py
 if [[ "$llt_phase" == full_qualification ]]; then llt_worker=experiments/l40s/full_latent_qualification.py; fi
 if ! "$llt_python" "$llt_worker" "$llt_phase" --model "$llt_model" --rank "$llt_rank" --loops "$llt_loops" --backend "$llt_backend" --policy "$llt_policy" > "$llt_out/$llt_stem.log" 2>&1; then
  "${LLT_CPU_PYTHON:-python}" experiments/l40s/loop_sweep_record.py "$llt_out/$llt_stem.json"
 else
  tail -n 1 "$llt_out/$llt_stem.log"
 fi
}
for llt_variant in llt:32 llt:64 llt:128 naive_loop:64 stacked:64 fixed_depth:64; do
 llt_model="${llt_variant%:*}";llt_rank="${llt_variant#*:}"
 for llt_loops in 4 16; do llt_run qualification "$llt_model" "$llt_rank" "$llt_loops" tensor none; done
done
# 416 new cases. Original 256 no-checkpoint measurements remain unchanged.
for llt_loops in {1..16}; do
 for llt_variant in llt:32 llt:64 llt:128 naive_loop:64 stacked:64 fixed_depth:64; do
  llt_model="${llt_variant%:*}";llt_rank="${llt_variant#*:}"
  for llt_backend in tensor torch; do
   if [[ "$llt_model" == llt && "$llt_rank" != 64 ]]; then
    for llt_phase in training inference; do llt_run "$llt_phase" "$llt_model" "$llt_rank" "$llt_loops" "$llt_backend" none; done
   fi
   llt_run training "$llt_model" "$llt_rank" "$llt_loops" "$llt_backend" ac
   if [[ "$llt_model" == llt ]]; then llt_run training "$llt_model" "$llt_rank" "$llt_loops" "$llt_backend" lac; fi
  done
 done
done
# Full geometry, initial weights; reference gradients streamed to CPU.
for llt_variant in llt:32 llt:64 llt:128 naive_loop:64 stacked:64 fixed_depth:64; do
 llt_model="${llt_variant%:*}";llt_rank="${llt_variant#*:}"
 for llt_backend in tensor torch; do llt_run full_qualification "$llt_model" "$llt_rank" 16 "$llt_backend" none; done
done
