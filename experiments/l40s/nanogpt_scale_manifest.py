"""Record runtime binaries and source identity after the numerical artifact audit."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import importlib.metadata
import torch
import tensor_torch.bridge as bridge
import tensor_torch._launch as launch

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'benchmarks/results/l40s-nanogpt'
TENSOR=Path(os.environ.get('TENSOR_CHECKOUT',ROOT.parent/'tensor'))


def main():
    assert bridge._executor is not None, 'qualified C++ executor must be loaded'
    binaries={}
    for module in (bridge._executor,launch):
        p=Path(module.__file__)
        binaries[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
    audit=json.loads((OUT/'audit.json').read_text())
    assert audit['status']=='passed' and len(audit['inputs'])==71
    sources={}
    for p in (ROOT/'experiments/l40s').glob('*.py'):
        digest=hashlib.sha256(p.read_bytes()).hexdigest()
        sources[str(p.relative_to(ROOT))]=digest
        dest=OUT/'sources'/digest/p.name;dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(p.read_bytes())
    data=dict(status='passed',native_cpp_executor=True,runtime_binary_sha256=binaries,
        tensor_repository='https://github.com/kreasof-ai/tensor',
        tensor_implementation='9d03a692b26e44f23348547263a97b7f18155c7f',
        tensor_current_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=TENSOR,text=True).strip(),
        torch=torch.__version__,cuda=torch.version.cuda,tilelang=importlib.metadata.version('tilelang'),
        tvm_ffi=importlib.metadata.version('apache-tvm-ffi'),nvrtc='12.9',sources=sources,
        selected_raw_reports=71,historical_observations=len(audit['observations']),
        audited_tensor_artifacts=audit['artifact_count'],
        audit_sha256=hashlib.sha256((OUT/'audit.json').read_bytes()).hexdigest())
    (OUT/'run-manifest.json').write_text(json.dumps(data,indent=2)+'\n')
    print('runtime and report audit passed')


if __name__=='__main__':main()
