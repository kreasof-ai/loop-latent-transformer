"""One-token evolving decoder, isolated tensor allocations and Vulkan timings.

Synthetic weights/history; no training, prefill, sampling or model quality claim.
naive_cache and fused evaluate the same latent-cache model, with separate KV
buffers for every layer/loop in the baseline. mha changes current-token KV math.
"""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess
from pathlib import Path
from datetime import datetime, timezone
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import numpy as np
from extended_attention import setup, Library, memory_bytes
import extended_kernels as k
from vulkan_attention import measure, measure_gpu, warm_device


def normalize(x): return x/np.sqrt(np.mean(x*x)+1e-5)
def gelu(x): return .5*x*(1+np.tanh(.7978845608028654*(x+.044715*x**3)))


def fixture(width,heads,layers,s,rank):
    rng=np.random.default_rng(np.random.SeedSequence([191,width,layers,s,rank]))
    def w(n,m,scale=.2): return (rng.normal(size=(n,m))*scale/np.sqrt(m)).astype('float16')
    return {'x':rng.normal(size=width).astype('float32'),
            'c':(rng.normal(size=(s,rank))*.5).astype('float16'),
            'down':w(rank,width), 'vocab':w(256,width),
            'layers':[{'q':w(width,width),'o':w(width,width),
                       'up':w(2*width,rank), 'kv':w(2*width,width),
                       'w1':w(4*width,width),'w2':w(width,4*width)} for _ in range(layers)]}


def reference(f,heads,loops,method):
    x=f['x'].astype('float64');width=x.size;d=width//heads
    c=f['c'].astype('float64').copy()
    current=f['down'].astype('float64')@normalize(x)
    c[-1]=current.astype('float16').astype('float64')
    full=[(c@layer['up'].astype('float64').T).astype('float16').astype('float64') for layer in f['layers']]
    for _ in range(loops):
        for index,layer in enumerate(f['layers']):
            n=normalize(x);q=(layer['q'].astype('float64')@n).reshape(heads,d)
            if method=='absorbed':
                up=layer['up'].astype('float64').reshape(2,heads,d,-1)
                aq=np.einsum('hd,hdr->hr',q,up[0]);scores=aq@c.T*d**-.5
                p=np.exp(scores-scores.max(axis=1,keepdims=True));p/=p.sum(axis=1,keepdims=True)
                output=np.einsum('hr,hdr->hd',p@c,up[1]).ravel()
            else:
                kv=full[index].copy()
                if method=='mha': kv[-1]=(layer['kv'].astype('float64')@n).astype('float16')
                key,value=kv.reshape(-1,2,heads,d).transpose(1,2,0,3)
                scores=np.einsum('hd,hsd->hs',q,key)*d**-.5
                p=np.exp(scores-scores.max(axis=1,keepdims=True));p/=p.sum(axis=1,keepdims=True)
                output=np.einsum('hs,hsd->hd',p,value).ravel()
            x=x+layer['o'].astype('float64')@output
            x=x+layer['w2'].astype('float64')@gelu(layer['w1'].astype('float64')@normalize(x))
    return f['vocab'].astype('float64')@normalize(x),c,full


def execute(tx,device,lib,period,f,heads,loops,method,samples,repeats):
    width=f['x'].size;d=width//heads;s,rank=f['c'].shape;p=16
    expected,c,full=reference(f,heads,loops,method)
    resources=[];categories={};calls=[]
    def put(x,category):
        buffer=device.from_numpy(np.ascontiguousarray(x).ravel())
        resources.append(buffer);categories.setdefault(category,[]).append(buffer);return buffer
    def empty(n,category='workspace',dtype='float32'): return put(np.zeros(n,dtype),category)
    def call(source,*args): calls.append(lib.call(source,args))
    original=put(f['x'],'input');x=empty(width);residual=empty(width);updated=empty(width)
    n=empty(width);q=empty(width);a=empty(width);projection=empty(width)
    hidden=empty(4*width);activation=empty(4*width);mlp=empty(width);logits=empty(256)
    down=put(f['down'],'weights') if method!='mha' else None
    vocab=put(f['vocab'],'weights');current=empty(rank)
    current_half=empty(rank,dtype='float16');rounded=empty(rank)
    latent=None
    if method in ('fused','absorbed'):
        latent=put(f['c'],'cache')
    else:
        # Needed only to form the current token, not a historical latent buffer.
        pass
    weights=[{name:put(array,'weights') for name,array in layer.items()
              if (name!='kv' or method=='mha') and (name!='up' or method!='mha')} for layer in f['layers']]
    ep=empty(heads*p*d);es=empty(heads*p*2)
    aq=al=ap=ats=None
    if method=='absorbed':
        aq=empty(heads*rank);al=empty(heads*rank);ap=empty(heads*p*rank);ats=empty(heads*p*2)
    call(k.pointwise(width,'copy'),original,x)
    if method!='mha':
        call(k.norm(1,width),x,n);call(k.gemv(1,width,rank),n,down,current)
        for input_dtype,output_dtype,src,dst in [('float32','float16',current,current_half),('float16','float32',current_half,rounded)]:
            call(k.emit([('x',rank,input_dtype),('out',rank,output_dtype)],f'''with T.Kernel(1, threads=128):
    for j in T.Parallel({rank}):
        out[j] = x[j]'''),src,dst)
    current=rounded
    if latent is not None: call(k.append(1,s,rank,'float16'),current,latent)
    current_kv=empty(2*width) if method in ('naive_cache','reuse_cache','mha') else None
    layer_caches={}
    for loop in range(loops):
        for index,layer in enumerate(weights):
            call(k.norm(1,width),x,n);call(k.gemv(1,width,width),n,layer['q'],q)
            if method in ('naive_cache','reuse_cache','mha'):
                if method=='reuse_cache':
                    if index not in layer_caches: layer_caches[index]=put(full[index].astype('float16'),'cache')
                    cache=layer_caches[index]
                else: cache=put(full[index].astype('float16'),'cache')
                if method=='mha': call(k.gemv(1,width,2*width),n,layer['kv'],current_kv)
                else: call(k.gemv(1,rank,2*width),current,layer['up'],current_kv)
                call(k.append(1,s,2*width,'float16'),current_kv,cache)
                call(k.attention(s,heads,d,rank,1,'float16',p,True),q,cache,ep,es)
                call(k.merge_source(heads,d,p),ep,es,a)
            elif method=='fused':
                call(k.fused(s,heads,d,rank,1,'float16',p),q,latent,layer['up'],ep,es)
                call(k.merge_source(heads,d,p),ep,es,a)
            else:
                call(k.query(heads,d,rank,1,'float16'),q,layer['up'],aq)
                call(k.attention(s,heads,d,rank,1,'float16',p),aq,latent,ap,ats)
                call(k.merge_source(heads,rank,p),ap,ats,al)
                call(k.output(heads,d,rank,1,'float16'),al,layer['up'],a)
            call(k.gemv(1,width,width),a,layer['o'],projection)
            call(k.pointwise(width,'add'),x,projection,residual)
            call(k.norm(1,width),residual,n)
            call(k.gemv(1,width,4*width),n,layer['w1'],hidden)
            call(k.pointwise(4*width,'gelu'),hidden,activation)
            call(k.gemv(1,4*width,width),activation,layer['w2'],mlp)
            call(k.pointwise(width,'add'),residual,mlp,updated)
            x,updated=updated,x
    call(k.norm(1,width),x,n);call(k.gemv(1,width,256),n,vocab,logits)
    plan=device.prepare_plan(calls);errors=[]
    # Replay resets residual state; every cache's final row is overwritten.
    for _ in range(5):
        plan.launch();actual=logits.to_numpy()
        np.testing.assert_allclose(actual,expected,atol=2e-4,rtol=.003)
        errors.append(float(np.max(np.abs(actual-expected))))
    warm_device(plan,device)
    timing=measure_gpu(plan,device,samples,repeats,period)
    timing['host']=measure(plan,device,samples,repeats)
    result={'method':method,'timing':timing,'dispatches':len(calls),
            'validation':{'passed':True,'replays':5,'maximum_absolute_error':max(errors)},
            'peak_live_tensor_bytes':memory_bytes(resources),
            'allocation_bytes':{name:memory_bytes(buffers) for name,buffers in categories.items()},
            'allocation_count':len(resources)}
    plan.close()
    for buffer in resources: buffer.release()
    return result


def run(a):
    tx,device,period=setup(a.tensor_root.resolve());a.out.mkdir(parents=True,exist_ok=True)
    report={'schema':'llt.decoder-forward.v1','status':'running','timestamp_utc':datetime.now(timezone.utc).isoformat(),
        'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                         (Path(__file__),Path(__file__).with_name('extended_kernels.py'))},
        'tensor_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.tensor_root,text=True).strip(),
        'protocol':{'samples':a.samples,'repeats':a.repeats,'gpu_timestamp_period_ns':period,
                    'precision':'FP16 weights/history; FP32 residual, accumulators, output',
                    'scope':'one input embedding to 256 logits; evolving residual/query across all layers and loops',
                    'architecture':'RMSNorm, attention, residual, RMSNorm, 4x GELU MLP, residual; final RMSNorm/vocabulary',
                    'baseline':'naive_cache: same LLT graph, distinct full KV cache for each layer/loop; reuse_cache: one per layer; mha: current KV from evolving hidden, different model math',
                    'memory':'all tensor allocations live at once, one method at a time; weights/cache/workspace included; driver and pipeline overhead excluded',
                    'history':'deterministic synthetic prefixes, not trained-model prefill',
                    'excluded':['compilation','allocation','uploads','readbacks','prefill','embedding lookup','sampling']},'cases':[]}
    def save(): (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    with device:
        report['device']=device.info;lib=Library(tx,device,a.out)
        specs=[(128,2,2,64,2)] if a.smoke else [(768,12,12,r,t) for r in a.ranks for t in a.loops]
        for width,heads,layers,rank,loops in specs:
            f=fixture(width,heads,layers,a.context,rank)
            row={'width':width,'heads':heads,'layers':layers,'rank':rank,'loops':loops,'context':a.context,'methods':[]}
            for method in ('naive_cache','reuse_cache','mha','absorbed','fused'):
                result=execute(tx,device,lib,period,f,heads,loops,method,a.samples,a.repeats)
                row['methods'].append(result)
                print(f'S={a.context} R={rank} T={loops} {method}',round(result['timing']['median_ms'],3),
                      'ms',round(result['peak_live_tensor_bytes']/2**20,3),'MiB',flush=True)
            report['cases'].append(row);report['artifacts']=lib.records;save()
        report['status']='passed';save();lib.close()
    print('Saved',a.out/'report.json',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tensor-root',type=Path,default=Path(__file__).resolve().parents[2]/'tensor')
    p.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'results'/'decoder')
    p.add_argument('--samples',type=int,default=5);p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--context',type=int,default=512);p.add_argument('--smoke',action='store_true')
    p.add_argument('--ranks',type=int,nargs='+',default=[64,128]);p.add_argument('--loops',type=int,nargs='+',default=[1,4,10])
    run(p.parse_args())
