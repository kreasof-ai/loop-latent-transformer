"""Pin audited numerical sources and the unchanged native Tensor runtime."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'benchmarks/results/l40s-latent-checkpoint-sweep'
TENSOR=Path(os.environ.get('TENSOR_CHECKOUT',ROOT.parent/'tensor'))


def main():
    audit=json.loads((OUT/'audit.json').read_text())
    assert audit['status']=='passed' and audit['case_count']==672
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    for name,digest in (audit['numerical_sources'] | audit['extension_sources']).items():
        data=subprocess.check_output(['git','show',f'{commit}:{name}'],cwd=ROOT)
        assert hashlib.sha256(data).hexdigest()==digest,name
    binaries={}
    for name in ('_executor_2_14.cpython-312-x86_64-linux-gnu.so','_launch.abi3.so'):
        binaries[name]=hashlib.sha256((TENSOR/'packages/tensor-torch/src/tensor_torch'/name).read_bytes()).hexdigest()
    baseline=ROOT/'benchmarks/results/l40s-loop-sweep/run-manifest.json'
    assert binaries==json.loads(baseline.read_text())['runtime_binary_sha256']
    sources={}
    for name in ('latent_checkpoint_model.py','latent_checkpoint_sweep.py','latent_checkpoint_summary.py',
                 'latent_checkpoint_manifest.py','latent_checkpoint_report.py','run_latent_checkpoint_sweep.sh','LOOP_SWEEP.md'):
        p=ROOT/'experiments/l40s'/name;digest=hashlib.sha256(p.read_bytes()).hexdigest()
        dest=OUT/'sources'/digest/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
        sources[str(p.relative_to(ROOT))]=digest
    manifest=dict(status='passed',case_count=672,new_case_count=416,reused_case_count=256,
        qualification_count=12,full_qualification_count=12,numerical_source_commit=commit,
        tensor_repository='https://github.com/kreasof-ai/tensor',
        tensor_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=TENSOR,text=True).strip(),
        runtime_binary_sha256=binaries,sources=sources,
        audit_sha256=hashlib.sha256((OUT/'audit.json').read_bytes()).hexdigest(),
        baseline_manifest_sha256=hashlib.sha256(baseline.read_bytes()).hexdigest(),
        ac_region='one full Transformer block',
        lac_region='existing LLT latent attention and folded output projection; explicit inputs Q_r, C, output weight',
        checkpoint_exact=True,lac_applicable_models=['llt'],
        notes='Primary old no-checkpoint records reused intact; original allocator-setup retries remain separate; all residual/query/MLP state outside LAC counts toward measured memory.')
    (OUT/'run-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('manifest passed')


if __name__=='__main__':main()
