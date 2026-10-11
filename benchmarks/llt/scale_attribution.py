"""Kernel attribution for the measured full-size nanoGPT training path.

This is a Tensor development diagnostic. Model latency/memory measurements live
in the LLT repository. CUPTI overhead is never included in those timings.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import torch
from llt_tensor.llt import AdamW, Operators
from .optimize import ROOT, BUILD
from archive.experiments.l40s.nanogpt_adapter import make


def fixture(a):
 torch.manual_seed(9505);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 ops=Operators(BUILD/'artifacts',cache_inference_weights=a.phase=='inference') if a.backend=='tensor' else None
 m=make(ops,n_embd=768,n_head=12,n_layer=12,vocab_size=50304,block_size=1024,dropout=0,bias=True)
 x=torch.randint(50304,(a.batch,1024),device='cuda');y=torch.randint(50304,x.shape,device='cuda')
 return m,ops,x,y


def main():
 out=ROOT/'docs/research/data/llt-scale-attribution';out.mkdir(parents=True,exist_ok=True)
 for backend in ('torch','tensor'):
  a=SimpleNamespace(model='nanogpt',backend=backend,phase='training',loops=1,batch=1,policy='none')
  m,ops,x,y=fixture(a)
  groups=[dict(params=[p for p in m.parameters() if p.ndim>=2],weight_decay=.1),dict(params=[p for p in m.parameters() if p.ndim<2],weight_decay=0)]
  opt=AdamW(groups,ops,lr=.0006,betas=(.9,.95)) if ops else torch.optim.AdamW(groups,lr=.0006,betas=(.9,.95),foreach=False,fused=False)
  def step():
   opt.zero_grad(set_to_none=True)
   with torch.autocast('cuda',dtype=torch.bfloat16):loss=m(x,y)[1]
   loss.backward()
   if not ops:torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
   opt.step()
  for _ in range(3):step()
  torch.cuda.synchronize()
  if ops:
   original=ops.call
   def traced(factory,p,outputs,*args,**kwargs):
    name=p.get('kind',factory) if isinstance(p,dict) else factory
    shape=','.join(f'{k}={p[k]}' for k in ('m','k','c') if isinstance(p,dict) and k in p)
    with torch.profiler.record_function('tensor:'+name+(':'+shape if shape else '')):
     return original(factory,p,outputs,*args,**kwargs)
   ops.call=traced
  with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as p:step()
  rows=[];kernels=[]
  for e in p.key_averages():
   row=dict(operation=e.key,count=e.count,cpu_total_us=e.cpu_time_total,device_total_us=e.device_time_total,self_device_total_us=e.self_device_time_total)
   if e.key.startswith('tensor:') or (backend=='torch' and e.key.startswith('aten:')):rows.append(row)
   if str(e.device_type).endswith('CUDA'):kernels.append(row)
  rows.sort(key=lambda r:r['device_total_us'],reverse=True);kernels.sort(key=lambda r:r['self_device_total_us'],reverse=True)
  p.export_chrome_trace(str(BUILD/('scale-'+backend+'-trace.json')))
  data=dict(status='passed',config=vars(a),operations=rows,kernels=kernels,
       protocol='one CUPTI step after three warmups; attribution only, profiler overhead excluded from latency comparisons',
       source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
  if ops:data['coverage']=ops.report
  (out/(backend+'.json')).write_text(json.dumps(data,indent=2)+'\n')
  (out/'scale_attribution.py').write_bytes(Path(__file__).read_bytes())
  print(backend,json.dumps(rows[:8]),flush=True)
  if ops:ops.call=original
  del m,ops,opt,step,x,y,p;import gc;gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':main()
