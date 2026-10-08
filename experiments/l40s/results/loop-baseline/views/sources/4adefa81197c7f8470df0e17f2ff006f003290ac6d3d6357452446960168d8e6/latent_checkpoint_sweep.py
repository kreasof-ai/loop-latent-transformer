"""Reuse the frozen no-checkpoint profiler and extend ranks/exact AC/native LAC."""

# Support both direct script execution and python -m experiments.l40s.<module>.
import sys
from pathlib import Path
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.l40s.runtime import results_root, require_run_directory
import argparse
import gc
import hashlib
import time
import traceback

import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from tensor_torch.llt import Operators
from experiments.l40s import loop_sweep as base
from model.checkpointing import CheckpointTransformer

OUT=results_root()/'loop-sweep'
ORIGINAL_CONFIG=base.config
EXTENSION_SOURCES=('model/checkpointing.py','experiments/l40s/latent_checkpoint_sweep.py')


def config(model,loops,small=False,rank=64):
    c=ORIGINAL_CONFIG(model,loops,small)
    c.rank=rank if model=='llt' else 64
    return c


def gradient_metrics(actual,reference):
    error=expected_norm=actual_norm=inner=0.
    per_parameter={}
    for name,expected in reference.items():
        current=actual[name].flatten();expected=expected.flatten()
        e=r=n=d=0.
        for offset in range(0,current.numel(),1048576):
            u=current[offset:offset+1048576].double();v=expected[offset:offset+1048576].double()
            e+=float((u-v).square().sum());r+=float(v.square().sum())
            n+=float(u.square().sum());d+=float((u*v).sum())
        per_parameter[name]=(e/max(r,1e-24))**.5
        error+=e;expected_norm+=r;actual_norm+=n;inner+=d
    return dict(global_relative_l2=(error/max(expected_norm,1e-24))**.5,
                cosine=inner/max((actual_norm*expected_norm)**.5,1e-24),
                maximum_parameter_relative_l2=max(per_parameter.values()),per_parameter_relative_l2=per_parameter)


def backward(model,x,y,policy):
    model.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16),sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        loss=model.loss(x,y,policy=policy)
    loss.backward()
    assert torch.isfinite(loss).item()
    assert all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters())
    gradients={name:p.grad.detach().cpu() for name,p in model.named_parameters()}
    return loss.item(),gradients


def cache_check(model,x):
    model.eval()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16),sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        full=model(x,last_logits=True)
        _,cache=model.prefill(x[:,:-1],capacity=x.shape[1]+1,last_logits=True)
        cached=model.decode_token(x[:,-1:],cache)
        maximum=float((cached.float()-full.float()).abs().max())
        relative=float((cached.float()-full.float()).norm()/full.float().norm().clamp_min(1e-12))
        assert maximum<.05 and relative<.03,(maximum,relative)
    model.train()
    return dict(max_abs_error=maximum,relative_l2=relative)


def qualification(a,full=False):
    c=config(a.model,a.loops,small=not full,rank=a.rank)
    batch,seq=(4,1024) if full else (2,32)
    x=torch.randint(c.vocab,(batch,seq),device='cuda')
    y=torch.randint(c.vocab,x.shape,device='cuda')
    policies=('none','ac','lac') if a.model=='llt' else ('none','ac')
    checks={};baselines={};coverage=None
    backends=(a.backend,) if full else ('torch','tensor')
    state=None
    for backend in backends:
        ops=Operators(OUT/'artifacts') if backend=='tensor' else None
        model=CheckpointTransformer(c,ops).cuda()
        if state is not None:model.load_state_dict(state)
        elif not full:state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        assert sum(p.numel() for p in model.parameters())==base.parameters(c)
        if a.model=='fixed_depth':assert base.parameters(c)==base.parameters(config('stacked',a.loops,small=not full))
        cache=cache_check(model,x)
        reference=None;reference_loss=None
        for policy in policies:
            loss,gradients=backward(model,x,y,policy)
            if policy=='none':reference=gradients;reference_loss=loss
            metrics=gradient_metrics(gradients,reference)
            assert loss==reference_loss,(backend,policy,loss,reference_loss)
            assert metrics['maximum_parameter_relative_l2']<1e-5,(backend,policy,metrics)
            checks[backend+'_'+policy]=dict(loss=loss,gradient_error=metrics,all_gradients_finite=True)
            model.zero_grad(set_to_none=True)
            if policy!='none':del gradients
            gc.collect();torch.cuda.empty_cache()
        if not full:baselines[backend]=reference
        checks[backend+'_cache']=cache
        if ops:
            assert not ops.report['fallbacks'];coverage=ops.report
        del model,reference
        gc.collect();torch.cuda.empty_cache()
    agreement=None
    if not full:
        agreement=gradient_metrics(baselines['tensor'],baselines['torch'])
        assert agreement['maximum_parameter_relative_l2']<.10,agreement
    return dict(status='passed',config=base.asdict(c),parameter_count=base.parameters(c),
                batch_size=batch,sequence_length=seq,policy_checks=checks,backend_gradient_agreement=agreement,coverage=coverage)


def run(a,out):
    base.config=lambda model,loops,small=False:config(model,loops,small,rank=a.rank)
    class ProfileModel(CheckpointTransformer):
        def loss(self,tokens,targets,policy=None,chunk_size=0):
            return super().loss(tokens,targets,policy=a.policy if policy is None else policy,chunk_size=chunk_size)
    base.BackendTransformer=ProfileModel
    base.run(a,out)
    out['training_protocol']=('Full logits/loss, backward, clip=1, AdamW FP32 states; '
        f'exact checkpoint policy={a.policy}; standard primary CUDA capture setup.') if a.phase=='training' else out['inference_protocol']


def save(stem,out):
    a=out['arguments']
    out['checkpoint_policy']=a['policy']
    out['checkpoint_boundary']=dict(none=None,ac='one full Transformer block',lac='LLT latent attention and folded output projection; inputs Q_r, C, folded output weight')[a['policy']]
    out['checkpoint_exact']=True
    sources={}
    for name in EXTENSION_SOURCES:
        p=base.ROOT/name;digest=hashlib.sha256(p.read_bytes()).hexdigest()
        dest=OUT/'sources'/digest/Path(name).name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
        sources[str(p.relative_to(base.ROOT))]=digest
    out['extension_sources']=sources
    base.save(stem,out)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('phase',choices=('training','inference','qualification','full_qualification'))
    p.add_argument('--model',choices=base.MODELS,required=True)
    p.add_argument('--rank',type=int,choices=(32,64,128),default=64)
    p.add_argument('--loops',type=int,required=True)
    p.add_argument('--backend',choices=('tensor','torch'),default='tensor')
    p.add_argument('--policy',choices=('none','ac','lac'),default='none')
    p.add_argument('--samples',type=int,default=9);p.add_argument('--repeats',type=int,default=3)
    a=p.parse_args()
    require_run_directory()
    assert 1<=a.loops<=16 and a.samples>=3 and a.repeats>=1
    assert a.model=='llt' or (a.rank==64 and a.policy!='lac')
    assert a.phase!='inference' or a.policy=='none'
    assert base._bridge._executor is not None
    base.OUT=OUT
    torch.manual_seed(9505);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    stem=f'{a.phase}-{a.model}-{a.backend}-r{a.rank:03}-t{a.loops:02}-{a.policy}'
    out=dict(status='running',arguments=vars(a),started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    start=time.monotonic()
    try:
        if a.phase in ('qualification','full_qualification'):out.update(qualification(a,full=a.phase=='full_qualification'))
        else:run(a,out)
    except torch.cuda.OutOfMemoryError as error:
        out.update(status='out_of_memory',error=str(error),failed_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   failed_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except Exception as error:out.update(status='error',error=str(error),traceback=traceback.format_exc())
    out['duration_seconds']=time.monotonic()-start
    save(stem,out)
    print(stem,out['status'],out.get('stage',''),f"{out['duration_seconds']:.1f}s",flush=True)
    if out['status']=='error' or (a.phase.endswith('qualification') and out['status']!='passed'):raise SystemExit(1)


if __name__=='__main__':main()
