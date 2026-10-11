"""Freeze optimization sources, environment, result hashes and artifact audit."""
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path
import torch
from .optimize import ROOT, OUT
from .qualify import TENSOR, source_key


def main():
 files=list((ROOT/'packages/llt-tensor/src/llt_tensor').glob('*.py'))+list((ROOT/'packages/llt-tensor/src/llt_tensor/templates').glob('*.py'))
 files+=list((ROOT/'benchmarks/llt').glob('optim*.py'))
 files+=[TENSOR/'src/tensor/compiler/search.py',TENSOR/'src/tensor/compiler/frontend/source.py',TENSOR/'src/tensor/compiler/cuda/nvrtc.py',TENSOR/'src/tensor/compiler/cuda/build.py',TENSOR/'src/tensor/compiler/cache.py',TENSOR/'src/tensor/compiler/cuda/lowering.py',ROOT/'packages/llt-tensor/src/llt_tensor/executor.cpp']
 llt=Path(os.environ.get('LLT_CHECKOUT',ROOT))
 external=[llt/name for name in ('model/__init__.py','model/reference.py','model/tensor_backend.py','archive/experiments/l40s/nanogpt_adapter.py','archive/experiments/l40s/third_party/nanogpt/model.py','archive/experiments/l40s/third_party/nanogpt/LICENSE')]
 sources={}
 for p in files+external:
  key=source_key(p)
  digest=hashlib.sha256(p.read_bytes()).hexdigest();sources[key]=digest
  dest=OUT/'sources'/digest/p.name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
 artifacts={};results={}
 def walk(v):
  if isinstance(v,dict):
   if 'path' in v and 'sha256' in v and str(v['path']).endswith('.tbin'):
    p=Path(v['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==v['sha256'];artifacts[v['sha256']]=dict(path=str(p),target=v['target'])
   for x in v.values():walk(x)
  elif isinstance(v,list):
   for x in v:walk(x)
 for p in OUT.glob('*.json'):
  if p.name=='manifest.json':continue
  v=json.loads(p.read_text());assert v['status']=='passed',(p,v.get('status'));walk(v)
  results[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
 data=dict(status='passed',environment=dict(torch=torch.__version__,cuda=torch.version.cuda,python=platform.python_version(),
  tilelang=importlib.metadata.version('tilelang'),tvm_ffi=importlib.metadata.version('apache-tvm-ffi'),
  gpu=torch.cuda.get_device_name(),sm=torch.cuda.get_device_capability(),nvrtc='12.9',commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()),
  sources=sources,results=results,artifact_count=len(artifacts),artifacts=artifacts,
  search=dict(algorithm='ScheduleSearch beam width 4',candidates_per_case=18,time_limit_per_case_seconds=180,seed=9502,atol=.005,rtol=.035),
  frozen_search_sources={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (OUT/'search-sources').glob('*.py')})
 (OUT/'manifest.json').write_text(json.dumps(data,indent=2)+'\n')
 print('audited',len(artifacts),'artifacts')
if __name__=='__main__':main()
