"""Aggregate retained nanoGPT-scale measurements and export shareable figures."""
import json
from pathlib import Path
import hashlib
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'benchmarks/results/l40s-nanogpt'

def main():
 rows=[];artifacts={};inputs={}
 for p in sorted(OUT.glob('*.json')):
  if p.name in ('summary.json','audit.json'):continue
  d=json.loads(p.read_text());assert d['status']=='passed',p
  inputs[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
  if 'coverage' in d:
   assert not d['coverage']['fallbacks']
   for a in d['coverage']['artifacts']:
    assert a['target']=='sm_89';ap=Path(a['path']);assert hashlib.sha256(ap.read_bytes()).hexdigest()==a['sha256']
    artifacts[a['sha256']]=a
  if d['arguments']['phase']=='correctness':continue
  a=d['arguments'];row={k:a[k] for k in ('phase','model','backend','batch','loops','policy')};row['parameter_count']=d['parameter_count']
  row['optimizer']=a.get('torch_optimizer','scalar') if a['backend']=='torch' else 'tensor'
  row['training_graph']=bool(a.get('training_graph',False))
  if a['phase']=='training':
   t=d['training'];row.update(gpu_ms=t['gpu_median_ms'],wall_ms=t['wall_median_ms'],peak_mib=t['peak_allocated_bytes']/2**20,tokens_per_second=d['tokens_per_second'])
   if 'training_graph' in d:row['graph_ms']=d['training_graph']['gpu_median_ms'];row['graph_step_counter']=d['graph_step_counter']
  else:
   for label in ('causal_last_logits','serving_startup','four_token_cached','four_token_recompute'):
    if label in d:
     row[label+'_eager_ms']=d[label]['eager']['gpu_median_ms'];row[label+'_graph_ms']=d[label]['graph']['gpu_median_ms']
     row[label+'_eager_peak_mib']=d[label]['eager']['peak_allocated_bytes']/2**20
   row['cache_mib']=d.get('cache_bytes',0)/2**20;row['fold_mib']=d.get('fold_bytes',0)/2**20
  rows.append(row)
 data=dict(status='passed',rows=rows,inputs=inputs,unique_tensor_artifacts=len(artifacts))
 (OUT/'summary.json').write_text(json.dumps(data,indent=2)+'\n')
 (OUT/'audit.json').write_text(json.dumps(dict(status='passed',inputs=inputs,artifact_count=len(artifacts),artifacts=artifacts),indent=2)+'\n')
 # Compare one-pass geometry at both microbatch sizes. Cached and recomputed
 # continuation are separate metrics in data/report rather than a false ratio.
 fig,ax=plt.subplots(2,2,figsize=(11,7))
 for col,batch in enumerate((1,4)):
  labels=['LLT','Naive','nanoGPT'];models=['llt','naive','nanogpt'];x=list(range(3))
  for backend,shift,color in [('torch',-.18,'#3b78a2'),('tensor',.18,'#d08838')]:
   tr=[next(r for r in rows if r['phase']=='training' and r['model']==m and r['backend']==backend and r['batch']==batch and r['loops']==1 and r['policy']=='none' and not r['training_graph'] and (backend!='torch' or r['optimizer']=='fused')) for m in models]
   label='Torch / fused AdamW' if backend=='torch' else 'Tensor'
   ax[0,col].bar([v+shift for v in x],[r['wall_ms'] for r in tr],width=.35,label=label,color=color)
   ax[1,col].bar([v+shift for v in x],[r['peak_mib']/1024 for r in tr],width=.35,label=backend,color=color)
  ax[0,col].set_title(f'W768 / 12 layers / S1024 / V50,304 / B{batch}')
  for a in ax[:,col]:a.set_xticks(x,labels);a.grid(axis='y',alpha=.2);a.set_axisbelow(True)
  ax[0,col].set_ylabel('Training step wall time (ms)');ax[1,col].set_ylabel('Peak allocated GPU memory (GiB)')
 ax[0,0].legend();fig.tight_layout();fig.savefig(OUT/'training.png',dpi=180);fig.savefig(OUT/'training.pdf');plt.close(fig)
 fig,ax=plt.subplots(2,2,figsize=(11,7))
 for col,batch in enumerate((1,4)):
  labels=['LLT','Naive','nanoGPT'];models=['llt','naive','nanogpt'];x=list(range(3))
  for backend,shift,color in [('torch',-.18,'#3b78a2'),('tensor',.18,'#d08838')]:
   inf=[next(r for r in rows if r['phase']=='inference' and r['model']==m and r['backend']==backend and r['batch']==batch and r['loops']==1) for m in models]
   ax[0,col].bar([v+shift for v in x],[r['causal_last_logits_graph_ms'] for r in inf],width=.35,label=backend,color=color)
   ax[1,col].bar([v+shift for v in x],[r['causal_last_logits_eager_peak_mib']/1024 for r in inf],width=.35,label=backend,color=color)
  ax[0,col].set_title(f'Full causal prefix / last logits / B{batch}')
  for a in ax[:,col]:a.set_xticks(x,labels);a.grid(axis='y',alpha=.2);a.set_axisbelow(True)
  ax[0,col].set_ylabel('CUDA graph forward (ms)');ax[1,col].set_ylabel('Eager peak allocation (GiB)')
 ax[0,0].legend();fig.tight_layout();fig.savefig(OUT/'inference.png',dpi=180);fig.savefig(OUT/'inference.pdf');plt.close(fig)
 print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
