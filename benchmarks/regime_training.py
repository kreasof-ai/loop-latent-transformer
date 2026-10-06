"""CPU training regime search: native shared-latent LLT vs looped/nonloop MHA.

Exact gradients, folded differentiable projections and identical checkpoint
options. Measures all observable live tensor storage, including Adam state and
temporaries; native operator scratch/runtime RSS are outside this ledger.
LLA is a post-training cache codec and has no corresponding pretraining graph.
"""
from __future__ import annotations
import argparse, contextlib, gc, hashlib, json, statistics, time, weakref
import platform
from datetime import datetime, timezone
from pathlib import Path
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten
from torch.utils.checkpoint import checkpoint
from lac_cpu import SavedMeter
torch.set_num_threads(4)


class LiveMeter(TorchDispatchMode):
    def __init__(self):
        super().__init__();self.refs={};self.storages={};self.current=0;self.peak=0
    def track(self,tensor):
        if not isinstance(tensor,torch.Tensor) or id(tensor) in self.refs: return
        storage=tensor.untyped_storage();size=storage.nbytes()
        if not size: return
        key=(storage.data_ptr(),size);identity=id(tensor)
        if key not in self.storages:
            self.storages[key]=0;self.current+=size;self.peak=max(self.peak,self.current)
        self.storages[key]+=1
        def gone(_):
            self.refs.pop(identity,None)
            self.storages[key]-=1
            if not self.storages[key]:
                del self.storages[key];self.current-=size
        self.refs[identity]=weakref.ref(tensor,gone)
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        for value in tree_flatten((args,kwargs or {}))[0]: self.track(value)
        out=func(*args,**(kwargs or {}))
        for value in tree_flatten(out)[0]: self.track(value)
        return out


def fixture(width,heads,layers,rank,batch,seq):
    gen=torch.Generator().manual_seed(911)
    def w(n,m): return torch.randn(n,m,generator=gen)*(.15/m**.5)
    common=[]
    for _ in range(layers): common += [w(width,width) for _ in range(4)]+[w(4*width,width),w(width,4*width)]
    common += [w(128,width)]
    gen=torch.Generator().manual_seed(913)
    x=torch.randn(batch,seq,width,generator=gen)
    target=torch.randint(128,(batch,seq),generator=gen)
    gen=torch.Generator().manual_seed(917)
    extra=[w(rank,width)]
    for _ in range(layers): extra += [w(width,rank),w(width,rank)]
    return common,extra,x,target


def norm(x): return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-5)


def forward(x,p,heads,layers,rank,loops,model,policy):
    width=x.shape[-1];dh=width//heads;common_count=layers*6+1
    cache=None;folds=[]
    if model=='llt':
        cache=norm(x)@p[common_count].T
        for layer in range(layers):
            wq,_,_,wo,_,_=p[layer*6:layer*6+6]
            uk,uv=p[common_count+1+layer*2:common_count+3+layer*2]
            q=(uk.reshape(heads,dh,rank).transpose(1,2)@wq.reshape(heads,dh,width)).reshape(heads*rank,width)
            out=torch.einsum('mhd,hdr->mhr',wo.reshape(width,heads,dh),uv.reshape(heads,dh,rank)).reshape(width,heads*rank)
            folds.append((q,out))
    def mlp(z,w1,w2):
        return torch.nn.functional.gelu(norm(z)@w1.T,approximate='tanh')@w2.T
    def loop_step(state):
        for layer in range(layers):
            wq,wk,wv,wo,w1,w2=p[layer*6:layer*6+6];z=norm(state)
            if model=='llt':
                fq,fo=folds[layer]
                q=(z@fq.T).reshape(*z.shape[:2],heads,rank).transpose(1,2)
                shared=cache.unsqueeze(1).expand(-1,heads,-1,-1)
                a=torch.nn.functional.scaled_dot_product_attention(q,shared,shared,is_causal=True,scale=dh**-.5)
                projection=a.transpose(1,2).reshape(*z.shape[:2],heads*rank)@fo.T
            else:
                q,key,value=[(z@w.T).reshape(*z.shape[:2],heads,dh).transpose(1,2) for w in (wq,wk,wv)]
                a=torch.nn.functional.scaled_dot_product_attention(q,key,value,is_causal=True)
                projection=a.transpose(1,2).reshape(*z.shape[:2],width)@wo.T
            state=state+projection
            delta=checkpoint(mlp,state,w1,w2,use_reentrant=False) if policy=='mlp' else mlp(state,w1,w2)
            state=state+delta
        return state
    if policy in ('loop','pair'):
        stride=1 if policy=='loop' else 2
        for start in range(0,loops,stride):
            count=min(stride,loops-start)
            def segment(state,count=count):
                for _ in range(count): state=loop_step(state)
                return state
            x=checkpoint(segment,x,use_reentrant=False)
    else:
        for _ in range(loops): x=loop_step(x)
    return norm(x)@p[layers*6].T


def selected(common,extra,layers,model):
    if model=='naive': return common
    # Preserve list indices but K/V projection weights are inert and excluded
    # from trainable allocations for LLT. Scalars fill those unused slots.
    values=list(common)
    for layer in range(layers):
        values[layer*6+1]=torch.empty(0);values[layer*6+2]=torch.empty(0)
    return values+extra


def validate_folding():
    width,h,layers,r,loops=128,2,2,16,3
    common,extra,input_value,_=fixture(width,h,layers,r,1,8)
    params=[v.double().requires_grad_(v.numel()>0) for v in selected(common,extra,layers,'llt')]
    active=[v for v in params if v.numel()];x=input_value.double().requires_grad_(True)
    folded=forward(x,params,h,layers,r,loops,'llt','none')
    cache=norm(x)@params[layers*6+1].T;state=x;dh=width//h
    mask=torch.ones(8,8,dtype=torch.bool).triu(1)
    for _ in range(loops):
        for layer in range(layers):
            wq,_,_,wo,w1,w2=params[layer*6:layer*6+6]
            uk,uv=params[layers*6+2+layer*2:layers*6+4+layer*2]
            q=(norm(state)@wq.T).reshape(1,8,h,dh).transpose(1,2)
            key,value=[(cache@w.T).reshape(1,8,h,dh).transpose(1,2) for w in (uk,uv)]
            probability=torch.softmax((q@key.transpose(-1,-2)*dh**-.5).masked_fill(mask,float('-inf')),dim=-1)
            state=state+(probability@value).transpose(1,2).reshape(1,8,width)@wo.T
            state=state+torch.nn.functional.gelu(norm(state)@w1.T,approximate='tanh')@w2.T
    explicit=norm(state)@params[layers*6].T
    torch.testing.assert_close(folded,explicit,atol=1e-11,rtol=1e-10)
    fg=torch.autograd.grad(folded.square().mean(),(x,*active),retain_graph=True)
    eg=torch.autograd.grad(explicit.square().mean(),(x,*active))
    for actual,expected in zip(fg,eg): torch.testing.assert_close(actual,expected,atol=1e-11,rtol=1e-10)
    return {'passed':True,'dtype':'float64','reference':'explicit full K/V and masked-softmax attention, unfolded projections',
            'forward_max_absolute_error':float((folded-explicit).abs().max().detach()),
            'gradient_max_absolute_error':max(float((a-b).abs().max()) for a,b in zip(fg,eg))}


def trial(f,heads,layers,rank,loops,model,policy,memory=False,gradients=False,optimizer_step=True):
    common,extra,initial,target=f
    p=[v.detach().clone().requires_grad_(v.numel()>0) for v in selected(common,extra,layers,model)]
    active=[v for v in p if v.numel()];x=initial.detach().clone().requires_grad_(True)
    opt=torch.optim.Adam(active,lr=1e-4,foreach=False)
    for value in active:
        opt.state[value]={'step':torch.tensor(0.),'exp_avg':torch.zeros_like(value),'exp_avg_sq':torch.zeros_like(value)}
    owned=[x,target,*active,*[v for state in opt.state.values() for v in state.values()]]
    live=LiveMeter() if memory else None;saved=SavedMeter(active) if memory else None
    gradient_buffers=[];handles=[]
    if live:
        for value in owned: live.track(value)
        def accumulated(parameter):
            gradient_buffers.append(parameter.grad);live.track(parameter.grad)
        handles=[value.register_post_accumulate_grad_hook(accumulated) for value in (x,*active)]
    def pack(value):
        holder=saved.pack(value);live.track(holder.tensor);return holder
    mode=live if live else contextlib.nullcontext()
    hooks=torch.autograd.graph.saved_tensors_hooks(pack,saved.unpack) if memory else contextlib.nullcontext()
    started=time.perf_counter()
    with mode,hooks:
        logits=forward(x,p,heads,layers,rank,loops,model,policy)
        loss=torch.nn.functional.cross_entropy(logits.flatten(0,1),target.flatten())
        loss.backward()
        if optimizer_step: opt.step()
    elapsed=(time.perf_counter()-started)*1000
    result={'loss':float(loss.detach()),'elapsed_ms':elapsed,
            'parameter_bytes':sum(v.numel()*v.element_size() for v in active),
            'optimizer_state_bytes':sum(v.numel()*v.element_size() for state in opt.state.values() for v in state.values()),
            'peak_live_tensor_bytes':live.peak if live else None,'peak_saved_tensor_bytes':saved.peak if saved else None}
    if gradients: result['gradients']=[x.grad.detach().clone()]+[v.grad.detach().clone() for v in active]
    for handle in handles: handle.remove()
    del opt,p,active,x,loss,logits,owned;gc.collect()
    return result


def configurations(smoke):
    if smoke: return [(64,2,1,16,1,16,4)]
    return [(w,w//64,2,r,1,s,t) for w,r,s,ts in [(256,16,64,[1,10,16,32]),(256,32,64,[1,10,16,32]),
                                               (256,32,256,[1,10,16]),(512,32,128,[1,10,16]),(512,16,128,[1,16])]
            for t in ts]


def run(a):
    a.out.mkdir(parents=True,exist_ok=True)
    report={'schema':'llt.regime-training.v1','status':'running','timestamp_utc':datetime.now(timezone.utc).isoformat(),
        'torch':torch.__version__,'device':'cpu','processor':platform.processor(),'dtype':'float32','threads':torch.get_num_threads(),
        'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(__file__).with_name('lac_cpu.py'))},
        'protocol':{'samples':a.samples,'acceptable_tax':.2,'large_memory_reduction':.5,'comparable_nonloop_max_ratio':1.25,
            'timing':'forward+cross-entropy+backward+Adam update; allocation/clones/GC/initial optimizer state excluded; hooks absent in timing runs',
            'memory':'unique live observable CPU tensor storage, includes input, labels, weights, gradients, Adam state and operator outputs; excludes immutable fixtures/runtime/native operator scratch; not RSS/GPU VRAM',
            'attention':'PyTorch CPU scaled-dot-product attention, causal; same API for both models; explicit scale head_width**-.5',
            'llt':'fixed shared global KV latent from initial input; full residual state; differentiable query/output folding outside loops; exact autograd',
            'lla':'paper is post-training inference cache codec; no native LLA pretraining memory graph claimed; parent teacher training follows naive graph',
            'policies':['none','mlp','loop','pair'],'checkpoint':'non-reentrant PyTorch exact checkpointing; identical options for both models',
            'quality':'random weights/token targets; gradient correctness only, no learned quality comparison'},
        'folding_validation':validate_folding(),'cases':[]}
    def save(): (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    specs=configurations(a.smoke)
    if a.cases: specs=[(w,w//64,2,r,b,s,t) for w,r,b,s,t in (map(int,value.split(':')) for value in a.cases)]
    for width,h,layers,r,b,s,t in specs:
        f=fixture(width,h,layers,r,b,s)
        row={'width':width,'heads':h,'layers':layers,'rank':r,'batch':b,'sequence':s,'loops':t,'methods':[]}
        for model in ('naive','llt'):
            gold=trial(f,h,layers,r,t,model,'none',gradients=True,optimizer_step=False)
            for policy in a.policies:
                checked=trial(f,h,layers,r,t,model,policy,gradients=True,optimizer_step=False)
                errors=[]
                for actual,expected in zip(checked.pop('gradients'),gold['gradients']):
                    torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-4)
                    errors.append(float((actual-expected).abs().max()))
                measured=trial(f,h,layers,r,t,model,policy,memory=True)
                trial(f,h,layers,r,t,model,policy)
                samples=[trial(f,h,layers,r,t,model,policy)['elapsed_ms'] for _ in range(a.samples)]
                result={key:value for key,value in measured.items() if key not in ('elapsed_ms',)}
                result.update({'model':model,'policy':policy,'validation':{'passed':True,'gradient_max_absolute_error':max(errors)},
                    'timing':{'median_ms':statistics.median(samples),'samples_ms':samples}})
                row['methods'].append(result)
                print(f'S={s} R={r} T={t} {model}/{policy}',round(result['timing']['median_ms'],3),'ms',
                      round(result['peak_live_tensor_bytes']/2**20,3),'MiB',flush=True)
            del gold
        report['cases'].append(row);save()
    report['status']='passed';save();print('Saved',a.out/'report.json',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'results'/'regime-training')
    p.add_argument('--samples',type=int,default=5);p.add_argument('--smoke',action='store_true')
    p.add_argument('--cases',nargs='+',help='width:rank:batch:sequence:loops, two layers, 64D heads')
    p.add_argument('--policies',nargs='+',choices=['none','mlp','loop','pair'],default=['none','mlp','loop','pair'])
    run(p.parse_args())
