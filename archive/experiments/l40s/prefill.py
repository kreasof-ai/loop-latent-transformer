"""Complete causal prefill: Tensor attention vs forced PyTorch Flash SDPA."""
import argparse
import gc
from dataclasses import asdict
import torch
from torch.nn.attention import SDPBackend,sdpa_kernel
from common import OUT,environment,graph_timing,memory,save,setup,timing
from kernels import Attention
from model import Config,Transformer
from systems import caches_bytes


def run(a):
    setup(108);engine=Attention()
    report=dict(status='running',environment=environment(),protocol=dict(
        dtype='float16',attention='causal prefill, serial online softmax Tensor; forced Flash SDPA control',
        cache='return and retain actual prefix caches from the full causal model',
        memory='CUDA allocator includes weights, actual retained caches, operator temporaries, current-stream cuBLAS workspace',
        positions='learned absolute embeddings',samples=a.samples),cases=[])
    with torch.inference_mode(),sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        for batch,seq,loops,rank in [(1,s,t,r) for s in (256,1024,4096) for t,r in ((1,32),(4,32),(4,64))]+[(4,1024,4,64)]:
            row=dict(batch=batch,sequence=seq,loops=loops,rank=rank,methods=[])
            for kind in ('naive','llt','layerwise'):
                gc.collect();torch.cuda.empty_cache()
                c=Config(kind=kind,width=512,heads=8,layers=4,loops=loops,rank=rank,vocab=256,max_seq=4096)
                model=Transformer(c).cuda().half().eval()
                tokens=torch.randint(c.vocab,(batch,seq),device='cuda')
                expected,expected_cache=model(tokens,return_cache=True)
                actual,actual_cache=model(tokens,attention=engine,return_cache=True)
                torch.testing.assert_close(actual,expected,atol=.02,rtol=.03)
                cache_size=caches_bytes(actual_cache)
                del actual,expected,actual_cache,expected_cache
                for backend in ('tensor','pytorch_flash'):
                    fn=lambda: model(tokens,attention=engine if backend=='tensor' else None,return_cache=True)
                    result,mem=memory(fn);del result
                    tm=timing(fn,a.samples,1)
                    gt=graph_timing(fn,a.samples,1)
                    row['methods'].append(dict(kind=kind,backend=backend,config=asdict(c),memory=mem,
                                               cache_bytes=cache_size,timing=tm,graph_timing=gt,validation='passed'))
                    print('prefill',batch,seq,loops,rank,kind,backend,round(gt['gpu_median_ms'],3),
                          round(mem['peak_allocated_bytes']/2**20,2),'MiB',flush=True)
                del fn,model,tokens
            report['cases'].append(row);report['artifacts']=engine.artifacts;save(OUT/'prefill.json',report)
    report['status']='passed';save(OUT/'prefill.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--samples',type=int,default=9)
    run(p.parse_args())
