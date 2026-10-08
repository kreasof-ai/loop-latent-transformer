"""Independent rotating-order repeat of the original W768/L12/S4K/T10 geometry."""
import argparse
import statistics
import torch
from torch.nn.attention import SDPBackend,sdpa_kernel
from common import OUT,environment,save,setup,timing
from kernels import Attention
from model import Config,Transformer


def run(a):
    setup(313);engine=Attention();owned=[];plans={};observations={}
    report=dict(status='running',environment=environment(),protocol=dict(
        width=768,layers=12,heads=12,context=4096,loops=10,batch=1,ranks=[32,64,128],
        seed=313,samples=a.samples,order='rotate one method each round',
        timing='CUDA graph replay, three repeats per sample, two warmups per sample; new process and new synthetic fixtures',
        memory='no memory comparison while all methods coexist; use isolated allocator measurements in inference.json'),methods=[])
    with torch.inference_mode(),sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        for kind,rank in [('naive',32),('llt',32),('llt',64),('llt',128)]:
            c=Config(kind=kind,width=768,heads=12,layers=12,loops=10,rank=rank,vocab=256,max_seq=16384)
            model=Transformer(c).cuda().half().eval()
            caches=[tuple(torch.randn(1,12,4097,64,device='cuda',dtype=torch.float16)*.1 for _ in range(2)) for _ in range(120)] if kind=='naive' else [torch.randn(1,1,4097,rank,device='cuda',dtype=torch.float16)*.1]
            token=torch.zeros(1,1,device='cuda',dtype=torch.long)
            pt=model.prepare_decode(caches,4096,cache_capacity=True)
            tx=model.prepare_decode(caches,4096,engine,cache_capacity=True)
            torch.testing.assert_close(tx(token),pt(token),atol=.02,rtol=.03)
            for backend,fn in [('tensor',tx),('pytorch_flash',pt)]:
                key=f'{kind}_r{rank}_{backend}'
                side=torch.cuda.Stream();side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    for _ in range(3): fn(token)
                torch.cuda.current_stream().wait_stream(side);torch.cuda.synchronize()
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph): output=fn(token)
                plans[key]=graph;observations[key]=[]
                owned.append((model,caches,token,fn,output,side))
        keys=list(plans)
        for round in range(a.samples):
            order=keys[round%len(keys):]+keys[:round%len(keys)]
            for key in order:
                t=timing(plans[key].replay,samples=1,repeats=3,warmup=2)
                observations[key].append(t['gpu_median_ms'])
            print('independent repeat round',round+1,flush=True)
        for key,samples in observations.items():
            report['methods'].append(dict(name=key,gpu_samples_ms=samples,median_gpu_ms=statistics.median(samples)))
        report['paired_ratios']={}
        for backend in ('tensor','pytorch_flash'):
            for rank in (32,64,128):
                ratios=[l/n for l,n in zip(observations[f'llt_r{rank}_{backend}'],observations[f'naive_r32_{backend}'])]
                report['paired_ratios'][f'llt_r{rank}_{backend}_vs_same_backend_naive']=dict(samples=ratios,median=statistics.median(ratios),max=max(ratios))
                ratios=[l/n for l,n in zip(observations[f'llt_r{rank}_{backend}'],observations['naive_r32_pytorch_flash'])]
                report['paired_ratios'][f'llt_r{rank}_{backend}_vs_flash_naive']=dict(samples=ratios,median=statistics.median(ratios),max=max(ratios))
    report['status']='passed';report['artifacts']=engine.artifacts;save(OUT/'repeat.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--samples',type=int,default=9);run(p.parse_args())
