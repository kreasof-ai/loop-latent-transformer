"""Search decoder memory/latency regimes: naive MHA, per-head LLA geometry, LLT.

Random codecs/synthetic history, no RoPE or quality claim. LLA follows the paper's
per-head separate K/V store and loop maps, including current-token encoding.
LLT folds the fixed up-projections into query/output weights for inference.
"""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess
from pathlib import Path
from datetime import datetime, timezone
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import numpy as np
from decoder_forward import fixture, normalize, gelu
from extended_attention import setup, Library, memory_bytes
import extended_kernels as k
from vulkan_attention import attention_source, measure, measure_gpu, warm_device


def self_merge(h,width,scale_width,p,shared=False):
    key=f'kv[j]' if shared else f'kv[head*{width}+j]'
    value=f'kv[j]' if shared else f'kv[{h*width}+head*{width}+j]'
    return k.emit([('q',h*width,'float32'),('kv',width if shared else 2*h*width,'float32'),
                   ('prefix',h*width,'float32'),('stats',h*p*2,'float32'),('out',h*width,'float32')],f'''with T.Kernel({h},threads=128) as head:
    dot = T.alloc_fragment(({width},), "float32")
    score = T.alloc_fragment((1,), "float32")
    maxima = T.alloc_fragment(({p},), "float32")
    maximum = T.alloc_fragment((1,), "float32")
    sums = T.alloc_fragment(({p},), "float32")
    total = T.alloc_fragment((1,), "float32")
    for j in T.Parallel({width}):
        dot[j] = q[head*{width}+j]*{key}
    T.reduce_sum(dot,score,dim=0)
    for part in T.Parallel({p}):
        maxima[part] = stats[(head*{p}+part)*2]
    T.reduce_max(maxima,maximum,dim=0)
    for part in T.Parallel({p}):
        sums[part] = stats[(head*{p}+part)*2+1]*T.exp(maxima[part]-maximum[0])
    T.reduce_sum(sums,total,dim=0)
    for j in T.Parallel({width}):
        high = T.max(maximum[0],score[0]*{scale_width**-.5})
        old = total[0]*T.exp(maximum[0]-high)
        fresh = T.exp(score[0]*{scale_width**-.5}-high)
        out[head*{width}+j] = (old*prefix[head*{width}+j]+fresh*{value})/(old+fresh)''')


def cast_copy(size,src_dtype,dst_dtype):
    return k.emit([('x',size,src_dtype),('out',size,dst_dtype)],f'''with T.Kernel(T.ceildiv({size},128),threads=128) as block:
    for j in T.Parallel(128):
        if block*128+j<{size}:
            out[block*128+j] = x[block*128+j]''')


def trajectory_write(t,width,loop):
    return k.emit([('x',2*width,'float32'),('out',t*2*width,'float32')],f'''with T.Kernel(T.ceildiv({2*width},128),threads=128) as block:
    for j in T.Parallel(128):
        if block*128+j<{2*width}:
            out[{loop*2*width}+block*128+j] = x[block*128+j]''')


def encode(t,h,d,r):
    from tensor.compiler.webgpu_lowering import partitioned_matmul_schedule
    depth=t*d
    body=partitioned_matmul_schedule(2*h,depth,r,
        f'x[(({{k}})//{d})*{2*h*d}+({{row}})*{d}+({{k}})%{d}]',
        f'T.cast(w[(by*{r}+({{column}}))*{depth}+({{k}})], "float32")',
        tile_m=1,tile_n=16,threads=128,partitions=8,unroll=4,dot_width=4)
    return k.emit([('x',t*2*h*d,'float32'),('w',2*h*r*depth,'float16'),('out',2*h*r,'float32')],body)


def data(s,r,t,width=768,h=12,layers=12):
    f=fixture(width,h,layers,s,r);d=width//h
    f['lla']=[]
    for index in range(layers):
        rng=np.random.default_rng(np.random.SeedSequence([109,s,r,index]))
        c=(rng.normal(size=(s,2,h,r))*.5).astype('float16')
        # Separate RNG streams make the prefix of the loop maps invariant in T.
        rng=np.random.default_rng(np.random.SeedSequence([113,s,r,index]))
        up=(rng.normal(size=(t,2,h,d,r))*.2/r**.5).astype('float16')
        down=(up.transpose(1,2,4,0,3).reshape(2,h,r,t*d)/t).astype('float16')
        f['lla'].append({'c':c,'up':up,'down':down})
    f['folded']=[]
    for layer in f['layers']:
        up=layer['up'].astype('float64').reshape(2,h,d,r)
        q=np.einsum('hdr,hdm->hrm',up[0],layer['q'].astype('float64').reshape(h,d,width)).reshape(h*r,width)
        out=np.einsum('mhd,hdr->mhr',layer['o'].astype('float64').reshape(width,h,d),up[1]).reshape(width,h*r)
        f['folded'].append({'q':q.astype('float32'),'o':out.astype('float32')})
    return f


def history(f,index,loop,h,method):
    if method=='llt':
        return f['c'].astype('float64')@f['layers'][index]['up'].astype('float64').T
    codec=f['lla'][index]
    full=np.einsum('sahr,ahdr->sahd',codec['c'].astype('float64'),codec['up'][loop].astype('float64')).reshape(-1,2*h*(f['x'].size//h))
    return full.astype('float16').astype('float64') if method=='naive' else full


def cpu_reference(f,h,t,method):
    width=f['x'].size;d=width//h;r=f['c'].shape[1];x=f['x'].astype('float64')
    current=(f['down'].astype('float64')@normalize(x)).astype('float16').astype('float64')
    trajectory=[[] for _ in f['layers']]
    for loop in range(t):
        for index,layer in enumerate(f['layers']):
            n=normalize(x)
            if method=='llt':
                q=(f['folded'][index]['q'].astype('float64')@n).reshape(h,r)
                c=np.concatenate([f['c'].astype('float64'),current[None,:]])
                scores=q@c.T*d**-.5
                probabilities=np.exp(scores-scores.max(axis=1,keepdims=True));probabilities/=probabilities.sum(axis=1,keepdims=True)
                projection=f['folded'][index]['o'].astype('float64')@(probabilities@c).ravel()
            else:
                q=(layer['q'].astype('float64')@n).reshape(h,d)
                new=(layer['kv'].astype('float64')@n).astype('float16').astype('float64')
                trajectory[index].append(new)
                full=np.concatenate([history(f,index,loop,h,method),new[None,:]])
                key,value=full.reshape(-1,2,h,d).transpose(1,2,0,3)
                scores=np.einsum('hd,hsd->hs',q,key)*d**-.5
                probabilities=np.exp(scores-scores.max(axis=1,keepdims=True));probabilities/=probabilities.sum(axis=1,keepdims=True)
                projection=layer['o'].astype('float64')@np.einsum('hs,hsd->hd',probabilities,value).ravel()
            x=x+projection
            x=x+layer['w2'].astype('float64')@gelu(layer['w1'].astype('float64')@normalize(x))
    encoded=[]
    if method=='lla':
        for index,values in enumerate(trajectory):
            stack=np.array(values).reshape(t,2,h,d).transpose(1,2,0,3).reshape(2,h,t*d)
            encoded.append(np.einsum('ahrk,ahk->ahr',f['lla'][index]['down'].astype('float64'),stack))
    return f['vocab'].astype('float64')@normalize(x),encoded


def execute(tx,device,lib,period,f,h,t,method,samples,repeats):
    width=f['x'].size;d=width//h;s,r=f['c'].shape;p=16
    expected,expected_codec=cpu_reference(f,h,t,method)
    resources=[];categories={};calls=[]
    def put(x,category='workspace'):
        buffer=device.from_numpy(np.ascontiguousarray(x).ravel());resources.append(buffer)
        categories.setdefault(category,[]).append(buffer);return buffer
    def empty(n,dtype='float32',category='workspace'): return put(np.zeros(n,dtype),category)
    def call(source,*args): calls.append(lib.call(source,args))
    inp=put(f['x'],'input');x=empty(width);updated=empty(width);residual=empty(width)
    n=empty(width);projection=empty(width);hidden=empty(4*width);activation=empty(4*width);mlp=empty(width)
    vocab=put(f['vocab'],'weights');logits=empty(256)
    weights=[]
    for index,layer in enumerate(f['layers']):
        selected={name:array for name,array in layer.items() if name in ('w1','w2') or (method!='llt' and name in ('q','o','kv'))}
        if method=='llt': selected.update(f['folded'][index])
        weights.append({name:put(array,'weights') for name,array in selected.items()})
    call(k.pointwise(width,'copy'),inp,x)
    current=current_half=None
    if method=='llt':
        cb=put(f['c'],'cache');down=put(f['down'],'weights');current=empty(r);raw=empty(r)
        current_half=empty(r,'float16','pending_cache_entry')
        call(k.norm(1,width),x,n);call(k.gemv(1,width,r),n,down,raw)
        call(cast_copy(r,'float32','float16'),raw,current_half)
        call(cast_copy(r,'float16','float32'),current_half,current)
    else:
        current=empty(2*width);raw=empty(2*width);current_half=empty(2*width,'float16')
    att_width=r if method!='naive' else d
    q=empty(h*att_width);prefix=empty(h*att_width);combined=empty(h*att_width)
    ep=empty(h*p*att_width);es=empty(h*p*2)
    fullq=empty(width) if method=='lla' else None
    out=empty(width) if method!='llt' else None
    merged=empty(width) if method=='lla' else None
    codec=[];encoded=[]
    if method=='lla':
        for index,values in enumerate(f['lla']):
            codec.append({'c':put(values['c'],'cache'),
                'up':[put(values['up'][loop].reshape(2*width,r),'codec_weights') for loop in range(t)],
                'down':put(values['down'],'codec_weights'),'trajectory':empty(t*2*width)})
            encoded.append(empty(2*h*r,'float16','pending_cache_entry'))
    for loop in range(t):
        for index,layer in enumerate(weights):
            call(k.norm(1,width),x,n)
            call(k.gemv(1,width,h*r if method=='llt' else width,'float32' if method=='llt' else 'float16'),n,layer['q'],fullq if method=='lla' else q)
            if method!='llt':
                call(k.gemv(1,width,2*width),n,layer['kv'],raw)
                call(cast_copy(2*width,'float32','float16'),raw,current_half)
                call(cast_copy(2*width,'float16','float32'),current_half,current)
            if method=='naive':
                cache=put(history(f,index,loop,h,'naive').astype('float16'),'cache')
                pending=empty(2*width,'float16','pending_cache_entry')
                call(cast_copy(2*width,'float32','float16'),current,pending)
                call(k.attention(s,h,d,r,1,'float16',p,True),q,cache,ep,es)
            elif method=='llt':
                call(k.attention(s,h,d,r,1,'float16',p),q,cb,ep,es)
            else:
                call(trajectory_write(t,width,loop),current,codec[index]['trajectory'])
                call(k.query(h,d,r,1,'float16'),fullq,codec[index]['up'][loop],q)
                call(attention_source(s,h,r,d,explicit=True,partitions=p),q,codec[index]['c'],ep,es)
            call(k.merge_source(h,att_width,p),ep,es,prefix)
            if method=='lla':
                # Decode prefix in full head width, then add exact current-token KV.
                call(k.output(h,d,r,1,'float16'),prefix,codec[index]['up'][loop],out)
                call(self_merge(h,d,d,p),fullq,current,out,es,merged)
                call(k.gemv(1,width,width),merged,layer['o'],projection)
            else:
                call(self_merge(h,att_width,d,p,method=='llt'),q,current,prefix,es,combined)
                call(k.gemv(1,h*att_width,width,'float32' if method=='llt' else 'float16'),combined,layer['o'],projection)
            call(k.pointwise(width,'add'),x,projection,residual)
            call(k.norm(1,width),residual,n);call(k.gemv(1,width,4*width),n,layer['w1'],hidden)
            call(k.pointwise(4*width,'gelu'),hidden,activation);call(k.gemv(1,4*width,width),activation,layer['w2'],mlp)
            call(k.pointwise(width,'add'),residual,mlp,updated);x,updated=updated,x
    if method=='lla':
        encoding=empty(2*h*r)
        for index in range(len(weights)):
            call(encode(t,h,d,r),codec[index]['trajectory'],codec[index]['down'],encoding)
            call(cast_copy(2*h*r,'float32','float16'),encoding,encoded[index])
    call(k.norm(1,width),x,n);call(k.gemv(1,width,256),n,vocab,logits)
    plan=device.prepare_plan(calls);errors=[];codec_errors=[];trajectory_errors=[]
    for _ in range(5):
        plan.launch();actual=logits.to_numpy()
        np.testing.assert_allclose(actual,expected,atol=2e-4,rtol=.003)
        errors.append(float(np.max(np.abs(actual-expected))))
        for index,buffer in enumerate(encoded):
            observed=buffer.to_numpy().reshape(2,h,r);target=expected_codec[index].astype('float16').astype('float64')
            supplied=codec[index]['trajectory'].to_numpy().reshape(t,2,h,d).transpose(1,2,0,3).reshape(2,h,t*d)
            oracle=np.einsum('ahrk,ahk->ahr',f['lla'][index]['down'].astype('float64'),supplied.astype('float64')).astype('float16').astype('float64')
            np.testing.assert_allclose(observed,oracle,atol=3e-6,rtol=.003)
            np.testing.assert_allclose(observed,target,atol=2e-4,rtol=.003)
            codec_errors.append(float(np.max(np.abs(observed-oracle))))
            trajectory_errors.append(float(np.max(np.abs(observed-target))))
    warm_device(plan,device)
    timing=measure_gpu(plan,device,samples,repeats,period);timing['host']=measure(plan,device,samples,repeats)
    result={'method':method,'timing':timing,'dispatches':len(calls),
        'validation':{'passed':True,'replays':5,'maximum_absolute_error':max(errors),
                      'codec_maximum_absolute_error':max(codec_errors,default=0),
                      'codec_end_to_end_maximum_absolute_error':max(trajectory_errors,default=0)},
        'peak_live_tensor_bytes':memory_bytes(resources),'allocation_count':len(resources),
        'allocation_bytes':{name:memory_bytes(buffers) for name,buffers in categories.items()}}
    plan.close()
    for buffer in resources: buffer.release()
    return result


def configurations(smoke):
    if smoke: return [(256,32,2,128,2,2)]
    return [(s,r,t,768,12,12) for s,r,ts in [(512,32,[1,4,10,16]),(4096,32,[1,4,10,16]),
                                            (512,64,[1,10]),(4096,64,[1,10]),(8192,32,[1,10]),(8192,64,[1,10])]
            for t in ts]


def run(a):
    tx,device,period=setup(a.tensor_root.resolve());a.out.mkdir(parents=True,exist_ok=True)
    report={'schema':'llt.regime-inference.v1','status':'running','timestamp_utc':datetime.now(timezone.utc).isoformat(),
        'tensor_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.tensor_root,text=True).strip(),
        'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (Path(__file__),Path(__file__).with_name('extended_kernels.py'),Path(__file__).with_name('decoder_forward.py'))},
        'protocol':{'samples':a.samples,'repeats':a.repeats,'gpu_timestamp_period_ns':period,'acceptable_tax':.2,
            'large_memory_reduction':.5,'comparable_nonloop_max_ratio':1.25,
            'models':'naive MHA (distinct caches); LLA per-head geometry (separate K/V, loop maps, encode new trajectory); LLT global latent, folded projections',
            'lla_source':'https://arxiv.org/html/2607.15456v2#S3','lla_scope':'random zero-mean codec geometry, no pretrained teacher/SVD/distillation/RoPE; not official LLA quality or serving reproduction',
            'precision':'FP16 cache/base weights/codec; FP32 residual, accumulations, folded LLT query/output weights',
            'memory':'peak live tensor allocation including weights, codecs, cache, pending token entries and workspace; excludes driver overhead',
            'token':'S historical tokens plus exact current-token attention; LLA encodes current loop trajectory after all loops',
            'excluded':['compilation','allocation','uploads','readback','prefill','token lookup','sampling'],
            'quality':'none; LLT is a different forward model; comparison is geometry and resource use'},'cases':[]}
    def save(): (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    with device:
        report['device']=device.info;lib=Library(tx,device,a.out)
        for s,r,t,width,h,layers in configurations(a.smoke):
            f=data(s,r,t,width,h,layers)
            row={'context_history':s,'rank':r,'loops':t,'width':width,'heads':h,'layers':layers,'methods':[]}
            for method in ('naive','lla','llt'):
                result=execute(tx,device,lib,period,f,h,t,method,a.samples,a.repeats);row['methods'].append(result)
                print(f'S={s} R={r} T={t} {method}',round(result['timing']['median_ms'],3),'ms',
                      round(result['peak_live_tensor_bytes']/2**20,3),'MiB',flush=True)
            report['cases'].append(row);report['artifacts']=lib.records;save()
        report['status']='passed';save();lib.close()
    print('Saved',a.out/'report.json',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tensor-root',type=Path,default=Path(__file__).resolve().parents[2]/'tensor')
    p.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'results'/'regime-inference')
    p.add_argument('--samples',type=int,default=5);p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--smoke',action='store_true');run(p.parse_args())
