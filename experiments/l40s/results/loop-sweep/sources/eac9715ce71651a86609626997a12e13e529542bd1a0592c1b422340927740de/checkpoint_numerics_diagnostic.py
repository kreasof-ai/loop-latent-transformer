"""Isolate BF16 autocast and checkpoint gradient differences on frozen sources."""
import json
from pathlib import Path
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from latent_checkpoint_sweep import config, gradient_metrics, OUT
from latent_checkpoint_model import CheckpointTransformer

torch.manual_seed(9505)
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32=False
c=config('llt',16,rank=32)
x=torch.randint(c.vocab,(4,1024),device='cuda')
y=torch.randint(c.vocab,x.shape,device='cuda')
model=CheckpointTransformer(c).cuda()
rows=[]
for cached in (True,False):
    reference=None
    for policy in ('none','none','ac','lac'):
        model.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=cached),sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            loss=model.loss(x,y,policy=policy)
        loss.backward()
        actual={n:p.grad.detach().cpu() for n,p in model.named_parameters()}
        if reference is None:reference=actual
        m=gradient_metrics(actual,reference)
        row=dict(cache_enabled=cached,policy=policy,loss=loss.item(),gradient=m)
        rows.append(row)
        print(cached,policy,loss.item(),m['global_relative_l2'],m['maximum_parameter_relative_l2'],flush=True)
        model.zero_grad(set_to_none=True)
        if actual is not reference:del actual
    del reference
out=OUT/'diagnostics';out.mkdir(exist_ok=True)
(out/'bf16-autocast-checkpoint.json').write_text(json.dumps(rows,indent=2)+'\n')
