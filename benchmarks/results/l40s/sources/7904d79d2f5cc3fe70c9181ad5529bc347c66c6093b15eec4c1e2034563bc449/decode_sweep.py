"""Partition-count sensitivity, long contexts, tail and aliasing correctness."""
import argparse
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend,sdpa_kernel
from common import OUT,environment,graph_timing,save,setup
from kernels import Attention


def run(a):
    setup(107)
    report=dict(status='running',environment=environment(),protocol=dict(
        dtype='float16',reference='forced PyTorch Flash SDPA, scale=1/sqrt(64)',
        query_length=1,heads=8,kv_heads=1,value='same latent tensor as keys',
        timing='prepared two-kernel partition+merge plan under CUDA graph replay',
        selection='fixed default 32 partitions; sensitivity includes 8/16/32/64; no per-case best selection in decoder'),cases=[])
    engines={p:Attention(partitions=p) for p in (8,16,32,64)}
    with torch.inference_mode(),sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        for batch,n,dim in [(1,n,d) for n in (513,4097,16385,65537) for d in (32,64,128)]+[(4,4097,64),(8,4097,64)]:
            q=torch.randn(batch,8,1,dim,device='cuda',dtype=torch.float16)
            latent=torch.randn(batch,1,n,dim,device='cuda',dtype=torch.float16)
            kv=latent.expand(-1,8,-1,-1)
            baseline=lambda:F.scaled_dot_product_attention(q,kv,kv,scale=64**-.5)
            ref=baseline()
            row=dict(batch=batch,context=n,dim=dim,pytorch_flash_graph=graph_timing(baseline,a.samples),partitions=[])
            for p,engine in engines.items():
                call=engine.prepare(q,latent,latent,False,64**-.5);call()
                actual=call.outputs[0]
                torch.testing.assert_close(actual,ref,atol=.002,rtol=.003)
                gt=graph_timing(call,a.samples)
                row['partitions'].append(dict(count=p,max_absolute_error=(actual-ref).abs().max().item(),graph_timing=gt))
                print('partition',batch,n,dim,p,round(gt['gpu_median_ms']*1e3,2),'us',flush=True)
            report['cases'].append(row);save(OUT/'decode-partitions.json',report)
        edge_checks=[]
        engine=engines[32]
        for n in (1,7,31,64,65,129):
            for magnitude in (.1,1.,10.):
                q=torch.randn(1,8,1,32,device='cuda',dtype=torch.float16)*magnitude
                latent=torch.randn(1,1,n,32,device='cuda',dtype=torch.float16)
                kv=latent.expand(-1,8,-1,-1)
                actual=engine(q,latent,latent,False,64**-.5)
                scores=q.float()@kv.float().transpose(-1,-2)*64**-.5
                ref=scores.softmax(-1)@kv.float()
                torch.testing.assert_close(actual.float(),ref,atol=.003,rtol=.003)
                assert actual.isfinite().all()
                edge_checks.append(dict(context=n,query_magnitude=magnitude,max_absolute_error=(actual.float()-ref).abs().max().item()))
    report['edge_checks']=edge_checks
    report['artifacts']=[r for engine in engines.values() for r in engine.artifacts]
    report['status']='passed';save(OUT/'decode-partitions.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--samples',type=int,default=9)
    run(p.parse_args())
