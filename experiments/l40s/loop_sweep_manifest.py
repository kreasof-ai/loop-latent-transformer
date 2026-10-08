"""Record binaries and reproduction sources without opening a CUDA context."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'benchmarks/results/l40s-loop-sweep'
TENSOR=Path(os.environ.get('TENSOR_CHECKOUT',ROOT.parent/'tensor'))


def main():
    audit=json.loads((OUT/'audit.json').read_text())
    assert audit['status']=='passed' and audit['case_count']==256
    source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    for name,sha in audit['numerical_sources'].items():
        committed=subprocess.check_output(['git','show',f'{source_commit}:{name}'],cwd=ROOT)
        assert hashlib.sha256(committed).hexdigest()==sha, name
    binaries={}
    for name in ('_executor_2_14.cpython-312-x86_64-linux-gnu.so','_launch.abi3.so'):
        p=TENSOR/'packages/tensor-torch/src/tensor_torch'/name
        binaries[name]=hashlib.sha256(p.read_bytes()).hexdigest()
    previous=json.loads((ROOT/'benchmarks/results/l40s-nanogpt/run-manifest.json').read_text())
    assert binaries==previous['runtime_binary_sha256'], 'runtime changed from the qualified native executor'
    sources={}
    for name in ('loop_sweep.py','loop_sweep_summary.py','loop_sweep_manifest.py','loop_sweep_report.py','run_loop_sweep.sh','LOOP_SWEEP.md','model.py','tensor_model.py','prepared_inference.py','nanogpt_scale.py'):
        p=ROOT/'experiments/l40s'/name
        sha=hashlib.sha256(p.read_bytes()).hexdigest()
        sources[str(p.relative_to(ROOT))]=sha
        dest=OUT/'sources'/sha/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(p.read_bytes())
    manifest=dict(status='passed',tensor_repository='https://github.com/kreasof-ai/tensor',
                  llt_numerical_source_commit=source_commit,
                  tensor_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=TENSOR,text=True).strip(),
                  runtime_binary_sha256=binaries,sources=sources,
                  audit_sha256=hashlib.sha256((OUT/'audit.json').read_bytes()).hexdigest(),
                  case_count=256,correctness_count=8)
    (OUT/'run-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('manifest passed')


if __name__=='__main__':
    main()
