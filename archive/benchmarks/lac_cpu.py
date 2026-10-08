"""Exact recomputation and lossy latent-checkpoint gradient checks on CPU."""
from __future__ import annotations
import argparse, gc, hashlib, json, statistics, time
from pathlib import Path
from datetime import datetime, timezone
import torch
torch.set_num_threads(4)


def step(state,params,latent):
    wq,wk,wv,wo,w1,w2=params[:6]
    x=state@params[6].T if latent else state
    z=x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-5)
    q,k,v=(z@w.T for w in (wq,wk,wv))
    scores=q@k.T/(x.shape[-1]**.5)
    mask=torch.ones_like(scores,dtype=torch.bool).triu(1)
    x=x+torch.softmax(scores.masked_fill(mask,float('-inf')),dim=-1)@v@wo.T
    z=x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-5)
    x=x+torch.nn.functional.gelu(z@w1.T,approximate='tanh')@w2.T
    return x@params[7].T if latent else x


def unroll(initial,params,count,latent):
    state=initial
    for _ in range(count): state=step(state,params,latent)
    return state


class Replay(torch.autograd.Function):
    @staticmethod
    def forward(ctx,initial,*args):
        params,count,latent=args[:-2],args[-2],args[-1]
        ctx.save_for_backward(initial,*params);ctx.count=count;ctx.latent=latent
        return unroll(initial,params,count,latent)
    @staticmethod
    def backward(ctx,upstream):
        initial,*parameters=ctx.saved_tensors
        totals=[torch.zeros_like(p) for p in parameters]
        grad=upstream
        for index in range(ctx.count-1,-1,-1):
            # Prefix replay has no tape. Only one recurrence step has a graph.
            # This makes tape memory constant in depth, at quadratic replay work.
            with torch.no_grad(): state=unroll(initial,parameters,index,ctx.latent)
            with torch.enable_grad():
                state=state.detach().requires_grad_(True)
                params=[p.detach().requires_grad_(True) for p in parameters]
                out=step(state,params,ctx.latent)
                grads=torch.autograd.grad(out,(state,*params),grad,allow_unused=True)
            grad=grads[0]
            for total,value in zip(totals,grads[1:]):
                if value is not None: total.add_(value)
        return (grad,*totals,None,None)


class LossyReplay(torch.autograd.Function):
    @staticmethod
    def forward(ctx,initial,codec,*args):
        params,count=args[:-1],args[-1]
        ctx.save_for_backward(initial@codec,codec,*params);ctx.count=count
        return unroll(initial,params,count,False)
    @staticmethod
    def backward(ctx,upstream):
        z,codec,*params=ctx.saved_tensors
        # Deliberately mirrors the proposal's approximate checkpoint: forward
        # used x, backward substitutes Up(Down(x)). It is not an exact gradient.
        with torch.enable_grad():
            reconstructed=(z@codec.T).detach().requires_grad_(True)
            replay_params=[p.detach().requires_grad_(True) for p in params]
            output=unroll(reconstructed,replay_params,ctx.count,False)
            grads=torch.autograd.grad(output,(reconstructed,*replay_params),upstream)
        return (grads[0],None,*grads[1:],None)


class SavedMeter:
    """Peak unique storage held by autograd saved-tensor wrappers, excluding weights.

    This is a measured tape-storage ledger, not process RSS or all CPU workspace.
    """
    def __init__(self,params):
        self.excluded={p.untyped_storage().data_ptr() for p in params}
        self.active={};self.current=0;self.peak=0
    def pack(self,tensor):
        return Saved(self,tensor)
    @staticmethod
    def unpack(saved): return saved.tensor


class Saved:
    def __init__(self,meter,tensor):
        self.tensor=tensor.detach();self.meter=meter
        ptr=tensor.untyped_storage().data_ptr();self.ptr=ptr
        if ptr not in meter.excluded:
            if ptr not in meter.active:
                size=tensor.untyped_storage().nbytes();meter.active[ptr]=[0,size]
                meter.current+=size;meter.peak=max(meter.peak,meter.current)
            meter.active[ptr][0]+=1
    def __del__(self):
        meter=self.meter
        if self.ptr in meter.active:
            meter.active[self.ptr][0]-=1
            if meter.active[self.ptr][0]==0:
                meter.current-=meter.active[self.ptr][1];del meter.active[self.ptr]


def parameters(width,rank,latent):
    generator=torch.Generator().manual_seed(91)
    shapes=[(width,width)]*4+[(4*width,width),(width,4*width)]
    values=[torch.randn(shape,generator=generator,dtype=torch.float64)*(.15/shape[-1]**.5) for shape in shapes]
    if latent:
        up=torch.linalg.qr(torch.randn(width,rank,generator=generator,dtype=torch.float64)).Q
        values += [up.clone(),up.T.contiguous().clone()]
    return [p.requires_grad_(True) for p in values]


def trial(initial,params,count,latent,replay,memory=False):
    x=initial.detach().clone().requires_grad_(True)
    p=[v.detach().clone().requires_grad_(True) for v in params]
    meter=SavedMeter(p) if memory else None
    context=torch.autograd.graph.saved_tensors_hooks(meter.pack,meter.unpack) if meter else None
    if context: context.__enter__()
    started=time.perf_counter()
    try:
        out=Replay.apply(x,*p,count,latent) if replay else unroll(x,p,count,latent)
        loss=out.square().mean();loss.backward()
        elapsed=(time.perf_counter()-started)*1000
    finally:
        if context: context.__exit__(None,None,None)
    gradients=[x.grad.detach().clone()]+[v.grad.detach().clone() for v in p]
    peak=meter.peak if meter else None
    del out,loss,x,p;gc.collect()
    return gradients,elapsed,peak


def run(out):
    out.mkdir(parents=True,exist_ok=True)
    report={'schema':'llt.lac-cpu.v1','status':'running','timestamp_utc':datetime.now(timezone.utc).isoformat(),
        'torch':torch.__version__,'device':'cpu','dtype':'float64','threads':torch.get_num_threads(),
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'protocol':{'width':64,'rank':16,'sequence':16,'replays':3,
            'memory':'peak unique autograd saved storage, excludes parameter storage; not RSS or total workspace',
            'timing':'forward+backward without storage-meter hooks; clones and GC excluded',
            'scope':'toy causal attention + RMSNorm + GELU MLP; no GPU training or trained-model quality',
            'exact_replay':'one input checkpoint + prefix recomputation per backward step; quadratic forward replay count'},
        'cases':[]}
    def save(): (out/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    for latent in (False,True):
        params=parameters(64,16,latent)
        initial=torch.randn(16,16 if latent else 64,generator=torch.Generator().manual_seed(92),dtype=torch.float64)
        for count in (1,4,16,32):
            baseline,_,baseline_peak=trial(initial,params,count,latent,False,True)
            replay,_,replay_peak=trial(initial,params,count,latent,True,True)
            for actual,expected in zip(replay,baseline):
                torch.testing.assert_close(actual,expected,atol=1e-10,rtol=1e-9)
            maxerror=max(float((a-b).abs().max()) for a,b in zip(replay,baseline))
            times={}
            for name,use_replay in [('autograd',False),('prefix_replay',True)]:
                trial(initial,params,count,latent,use_replay)
                values=[trial(initial,params,count,latent,use_replay)[1] for _ in range(3)]
                times[name]={'median_ms':statistics.median(values),'samples_ms':values}
            row={'state':'latent' if latent else 'full','loops':count,'gradient_max_abs_error':maxerror,
                'peak_saved_bytes':{'autograd':baseline_peak,'prefix_replay':replay_peak},
                'checkpoint_payload_bytes':initial.numel()*initial.element_size(),
                'replayed_prefix_forward_steps':count*(count-1)//2,'timings':times,'passed':True}
            report['cases'].append(row);save();print(row,flush=True)
    params=parameters(64,16,False)
    x=torch.randn(16,64,generator=torch.Generator().manual_seed(93),dtype=torch.float64)
    codec=torch.linalg.qr(torch.randn(64,16,generator=torch.Generator().manual_seed(94),dtype=torch.float64)).Q
    exact,_,_=trial(x,params,4,False,False)
    xp=x.clone().requires_grad_(True);pp=[p.detach().clone().requires_grad_(True) for p in params]
    output=LossyReplay.apply(xp,codec,*pp,4);loss=output.square().mean();loss.backward()
    approximate=[xp.grad]+[p.grad for p in pp]
    differences=[float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b).clamp_min(1e-30)) for a,b in zip(approximate,exact)]
    report['lossy_checkpoint']={'rank':16,'width':64,'loops':4,'relative_l2_gradient_errors':differences,
        'input_reconstruction_relative_l2_error':float(torch.linalg.vector_norm(x-x@codec@codec.T)/torch.linalg.vector_norm(x)),
        'expected_exact':False,'biased_gradient_detected':max(differences)>1e-3}
    assert report['lossy_checkpoint']['biased_gradient_detected']
    # Independent directional finite difference checks the full-state replay
    # input gradient against a loss evaluation, not only another backward graph.
    direction=torch.randn(x.shape,generator=torch.Generator().manual_seed(95),dtype=torch.float64)
    direction/=torch.linalg.vector_norm(direction);eps=1e-5
    with torch.no_grad():
        finite=(unroll(x+eps*direction,params,4,False).square().mean()-unroll(x-eps*direction,params,4,False).square().mean())/(2*eps)
    replay,_,_=trial(x,params,4,False,True)
    analytic=(replay[0]*direction).sum();error=float((analytic-finite).abs())
    assert error<1e-8
    report['directional_gradient_check']={'finite_difference':float(finite),'replay_gradient_dot_direction':float(analytic),'absolute_error':error,'passed':True}
    report['status']='passed';save();print('Lossy checkpoint',report['lossy_checkpoint'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'results'/'lac-cpu')
    run(parser.parse_args().out)
