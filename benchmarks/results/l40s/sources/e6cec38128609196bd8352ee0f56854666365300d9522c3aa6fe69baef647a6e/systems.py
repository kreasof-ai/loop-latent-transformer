"""L40S Tensor/PyTorch attention, complete decoder and GPU training sweeps."""
import argparse
import gc
import json
from dataclasses import asdict

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from common import OUT, environment, graph_timing, memory, save, setup, timing
from kernels import Attention
from model import Config, Transformer


def attention_study(a):
    setup(101)
    engine = Attention(split_decode=not a.serial)
    report = dict(status='running', environment=environment(), protocol=dict(
        dtype='float16', accumulation='float32', timing='CUDA events and synchronized wall time; prepared Tensor output; repeated calls',
        reference='explicit FP32 matmul/softmax and PyTorch Flash SDPA',
        shared_kv='one stored KV head, addressed by all query heads', samples=a.samples), cases=[])
    specs = [(1,4,1,13,13,16,True), (2,4,1,37,37,32,True), (1,4,4,65,65,64,True),
             (1,4,1,1,513,32,False)] if a.smoke else [
                 (b,8,kh,m,n,d,causal) for b in (1,4) for d in (32,64,96,128)
                 for kh,m,n,causal in ((1,1,4097,False), (8,1,4097,False), (1,257,257,True), (1,1024,1024,True))]
    with torch.inference_mode():
        for b,h,kh,m,n,d,causal in specs:
            q=torch.randn(b,h,m,d,device='cuda',dtype=torch.float16)
            k=torch.randn(b,kh,n,d,device='cuda',dtype=torch.float16)
            v=torch.randn_like(k)
            kk,vv=k.expand(-1,h,-1,-1),v.expand(-1,h,-1,-1)
            scale=64**-.5
            reference=q.float()@kk.float().transpose(-1,-2)*scale
            if causal:
                qi=torch.arange(m,device='cuda')+n-m
                ki=torch.arange(n,device='cuda')
                reference.masked_fill_(ki[None,:]>qi[:,None],float('-inf'))
            reference=reference.softmax(-1)@vv.float()
            call=engine.prepare(q,k,v,causal,scale)
            call()
            actual=call.outputs[0]
            torch.testing.assert_close(actual.float(),reference,atol=2e-3,rtol=2e-3)
            torch_fn=lambda: F.scaled_dot_product_attention(q,kk,vv,is_causal=causal,scale=scale)
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                baseline=torch_fn()
                torch.testing.assert_close(actual,baseline,atol=2e-3,rtol=2e-3)
                pt=timing(torch_fn,a.samples,20)
                pg=graph_timing(torch_fn,a.samples)
            tx=timing(call,a.samples,20)
            tg=graph_timing(call,a.samples)
            row=dict(batch=b,heads=h,kv_heads=kh,query=m,context=n,dim=d,causal=causal,
                     max_absolute_error=(actual.float()-reference).abs().max().item(),
                     relative_l2_error=((actual.float()-reference).norm()/reference.norm()).item(),
                     tensor=tx,pytorch_flash=pt,tensor_graph=tg,pytorch_flash_graph=pg)
            report['cases'].append(row)
            report['artifacts']=engine.artifacts
            save(OUT/('attention-serial.json' if a.serial else 'attention.json'),report)
            print('attention',b,kh,m,n,d,'Tensor',round(tx['gpu_median_ms'],4),'Flash',round(pt['gpu_median_ms'],4),flush=True)
    report['status']='passed'
    save(OUT/('attention-serial.json' if a.serial else 'attention.json'),report)


def caches_bytes(caches):
    flat=[t for pair in caches for t in pair] if caches and isinstance(caches[0],tuple) else caches
    storages={t.untyped_storage().data_ptr(): t.untyped_storage().nbytes() for t in flat}
    return sum(storages.values())


def inference_study(a):
    setup(102)
    engine=Attention()
    report=dict(status='running',environment=environment(), protocol=dict(
        dtype='float16',history='independent synthetic cache, current token included',
        positions='learned absolute embeddings; no RoPE',
        timing='full one-token decoder, including cache concatenation, MLP, normalization, logits; folded weights prepared outside timing',
        memory='CUDA allocator peak including model, folded weights, persistent history and temporary tensors; driver/module overhead excluded',
        samples=a.samples),cases=[])
    specs=[(128,2,4,32,1,128)] if a.smoke else (
        [(512,4,t,r,1,4096) for r in (32,64,96,128) for t in (1,4,10,16)] +
        [(512,4,10,64,b,s) for b,s in ((1,512),(1,8192),(4,4096),(8,4096))] +
        [(768,12,10,r,1,4096) for r in (32,64,128)])
    with torch.inference_mode(),sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        for width,layers,loops,rank,batch,seq in specs:
            row=dict(width=width,layers=layers,heads=width//64,loops=loops,rank=rank,batch=batch,context=seq,methods=[])
            for kind in ('naive','llt','layerwise'):
                gc.collect();torch.cuda.empty_cache()
                c=Config(kind=kind,width=width,heads=width//64,layers=layers,loops=loops,rank=rank,vocab=256,max_seq=16384)
                model=Transformer(c).cuda().half().eval()
                if kind=='naive':
                    caches=[tuple(torch.randn(batch,c.heads,seq,64,device='cuda',dtype=torch.float16)*.1 for _ in range(2))
                            for _ in range(loops*layers)]
                else:
                    caches=[torch.randn(batch,1,seq,rank,device='cuda',dtype=torch.float16)*.1
                            for _ in range(layers if kind=='layerwise' else 1)]
                token=torch.zeros(batch,1,device='cuda',dtype=torch.long)
                pt=model.prepare_decode(caches,seq)
                tx=model.prepare_decode(caches,seq,engine)
                expected=pt(token)
                actual=tx(token)
                torch.testing.assert_close(actual,expected,atol=.02,rtol=.03)
                del actual,expected
                # Both closures own folded weights; drop the unused one for fair memory accounting.
                for backend in ('tensor','pytorch_flash'):
                    if backend=='tensor':
                        del pt
                        fn=lambda: tx(token)
                    else:
                        del fn,tx
                        pt=model.prepare_decode(caches,seq)
                        fn=lambda: pt(token)
                    _,mem=memory(fn)
                    del _
                    t=timing(fn,a.samples,3)
                    gt=graph_timing(fn,a.samples,3)
                    result=dict(kind=kind,backend=backend,config=asdict(c),cache_bytes=caches_bytes(caches),
                                parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),memory=mem,timing=t,graph_timing=gt,validation='passed')
                    row['methods'].append(result)
                    print('decode',seq,loops,rank,batch,kind,backend,round(t['gpu_median_ms'],3),
                          round(mem['peak_allocated_bytes']/2**20,2),'MiB',flush=True)
                del fn,pt,model,caches,token
            report['cases'].append(row);report['artifacts']=engine.artifacts;save(OUT/'inference.json',report)
    report['status']='passed';save(OUT/'inference.json',report)


def training_study(a):
    setup(103)
    report=dict(status='running',environment=environment(),protocol=dict(
        parameters='FP32',autocast='bfloat16',optimizer='AdamW FP32 states, foreach=False',
        attention='PyTorch CUDA Flash SDPA; force backend rather than silently fall back',
        timing='steady-state full forward, all-token cross entropy, backward, AdamW; initial optimizer allocation and compilation excluded',
        memory='CUDA allocator peak, model + gradients + optimizer + activations; reserved memory separately; driver overhead excluded',
        checkpoints='non-reentrant exact per-loop; identical baseline policy',samples=a.samples),cases=[])
    specs=[(128,2,32,1,64,4,128)] if a.smoke else (
        [(512,2,r,1,1024,t,4096) for r in (32,64,128) for t in (1,4,10,20)] +
        [(512,2,32,1,1024,20,50257),(512,4,64,4,512,10,4096),
         (512,2,64,1,4096,10,4096),(768,12,64,1,1024,10,4096)])
    for width,layers,rank,batch,seq,loops,vocab in specs:
        row=dict(width=width,layers=layers,rank=rank,batch=batch,sequence=seq,loops=loops,vocab=vocab,methods=[])
        for kind in ('naive','llt','layerwise'):
            for policy in ('none','loop'):
                gc.collect();torch.cuda.empty_cache();setup(103)
                c=Config(kind=kind,width=width,heads=width//64,layers=layers,rank=rank,loops=loops,vocab=vocab,max_seq=seq)
                model=Transformer(c).cuda()
                tokens=torch.randint(vocab,(batch,seq),device='cuda')
                targets=torch.randint(vocab,(batch,seq),device='cuda')
                opt=torch.optim.AdamW(model.parameters(),lr=1e-4,foreach=False)
                def step():
                    opt.zero_grad(set_to_none=True)
                    with torch.autocast('cuda',dtype=torch.bfloat16),sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                        logits=model(tokens,policy)
                        loss=F.cross_entropy(logits.flatten(0,1).float(),targets.flatten())
                    loss.backward();opt.step()
                    return loss.detach()
                for _ in range(3): step()
                loss,mem=memory(step)
                t=timing(step,a.samples,1,warmup=1)
                result=dict(kind=kind,policy=policy,loss=loss.item(),memory=mem,timing=t,
                            parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
                            optimizer_state_bytes=sum(t.numel()*t.element_size() for state in opt.state.values() for t in state.values()))
                row['methods'].append(result)
                print('train',seq,loops,rank,vocab,kind,policy,round(t['gpu_median_ms'],3),
                      round(mem['peak_allocated_bytes']/2**20,2),'MiB',flush=True)
                del step,loss,opt,model,tokens,targets
        report['cases'].append(row);save(OUT/'training.json',report)
    report['status']='passed';save(OUT/'training.json',report)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['attention','inference','training'])
    parser.add_argument('--samples',type=int,default=9)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--serial',action='store_true',help='reference serial decode kernel, attention phase only')
    args=parser.parse_args()
    globals()[args.phase+'_study'](args)
