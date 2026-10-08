#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export TENSOR_NVRTC_HOME="${TENSOR_NVRTC_HOME:-/home/sagemaker-user/tensor/build/nvrtc-12.9}"
llt_python="${LLT_PYTHON:-/home/sagemaker-user/tensor/.venv/bin/python}"
llt_cpu_python="${LLT_CPU_PYTHON:-python}"
llt_out=benchmarks/results/l40s-loop-sweep/capture-recovery
mkdir -p "$llt_out"
"$llt_cpu_python" -c 'import pathlib; p=pathlib.Path("benchmarks/results/l40s-loop-sweep"); assert len(list(p.glob("training-*.json")))+len(list(p.glob("inference-*.json")))==256, "Complete the primary sweep before capture retries"'
while read -r llt_model llt_backend llt_loops; do
 printf -v llt_stem 'training-%s-%s-t%02d' "$llt_model" "$llt_backend" "$llt_loops"
 if [[ "${LLT_RESUME:-0}" == 1 && -f "$llt_out/$llt_stem.json" ]]; then continue; fi
 if ! "$llt_python" experiments/l40s/loop_sweep_recovery.py --model "$llt_model" --backend "$llt_backend" --loops "$llt_loops" > "$llt_out/$llt_stem.log" 2>&1; then
  "$llt_cpu_python" experiments/l40s/loop_sweep_record.py "$llt_out/$llt_stem.json"
 else
  tail -n 1 "$llt_out/$llt_stem.log"
 fi
done < <("$llt_cpu_python" - <<'PY'
import json,pathlib
root=pathlib.Path('benchmarks/results/l40s-loop-sweep')
paths=list(root.glob('training-*.json'))+list(root.glob('inference-*.json'))
assert len(paths)==256, 'Complete the primary sweep before capture retries'
for p in sorted(root.glob('training-*.json')):
 d=json.loads(p.read_text())
 if d['status']=='out_of_memory' and d.get('stage')=='training_graph':
  a=d['arguments'];print(a['model'],a['backend'],a['loops'])
PY
)
