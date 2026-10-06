"""Nine-suite follow-up: precision/batch/rank/context/depth and fused attention."""
from __future__ import annotations
import argparse, hashlib, json, os, re, statistics, subprocess, sys, time
from pathlib import Path
from datetime import datetime, timezone
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
os.environ.setdefault('WGPU_BACKEND_TYPE','Vulkan')
import numpy as np
from vulkan_attention import bound_call, measure, measure_gpu, TimestampAdapter, warm_device, oracle
import extended_kernels as kernels


def setup(root):
    sys.path.insert(0,str(root/'src'))
    import tensor as tx
    from tensor.providers.webgpu import Device
    vk=subprocess.run(['vulkaninfo'],capture_output=True,text=True,check=True)
    values=re.findall(r'timestampPeriod\s*=\s*([0-9.]+)',vk.stdout)
    if len(values)!=1: raise RuntimeError('one Vulkan GPU required for timestamp calibration')
    device=Device();device._adapter=TimestampAdapter(device._adapter)
    return tx,device,float(values[0])


class Library:
    def __init__(self,tx,device,out):
        self.tx,self.device,self.out=tx,device,out
        self.loaded={};self.records=[]
        (out/'artifacts').mkdir(parents=True,exist_ok=True)
    def get(self,source):
        sha=hashlib.sha256(source.encode()).hexdigest()
        if sha not in self.loaded:
            path=self.out/'artifacts'/f'{sha[:20]}.py';binpath=path.with_suffix('.tbin')
            path.write_text(source,encoding='utf-8')
            if not binpath.exists(): self.tx.build(path,binpath,provider='webgpu',cache_dir=self.out/'cache')
            started=time.perf_counter();kernel=self.device.load(binpath)
            self.loaded[sha]=kernel
            self.records.append({'source_sha256':sha,'artifact':str(binpath.relative_to(self.out)),
                'artifact_sha256':hashlib.sha256(binpath.read_bytes()).hexdigest(),
                'pipeline_ms':(time.perf_counter()-started)*1000,
                'workgroup_bytes':kernel.manifest['webgpu']['workgroup_storage_bytes']})
        return self.loaded[sha]
    def call(self,source,args): return bound_call(self.device,self.get(source),args)
    def close(self):
        for kernel in self.loaded.values(): kernel._dispose()
        self.loaded.clear()


def memory_bytes(resources):
    return sum(b.nbytes for b in {id(v):v for v in resources}.values())


def configurations(smoke=False):
    if smoke:
        return [{'suite':'smoke','context':256,'rank':r,'batch':b,'dtype':dt,'loops':[1,4]}
                for r,b,dt in [(64,1,'float16'),(128,2,'float16'),(64,2,'float32')]]
    result=[]
    for r in (32,64,96,128):
        result.append(dict(suite='rank',context=4096,rank=r,batch=1,dtype='float16',loops=[10]))
    for s in (8192,16384):
        for r in (64,128): result.append(dict(suite='context',context=s,rank=r,batch=1,dtype='float16',loops=[10]))
    for r in (64,128): result.append(dict(suite='depth',context=4096,rank=r,batch=1,dtype='float16',loops=[16,32,64]))
    for b in (2,4,8):
        for r in (64,128): result.append(dict(suite='batch',context=1024,rank=r,batch=b,dtype='float16',loops=[10]))
    for dt in ('float16','float32'):
        for r in (64,128): result.append(dict(suite='precision',context=4096,rank=r,batch=1,dtype=dt,loops=[10]))
    return result


def run(a):
    tx,device,period=setup(a.tensor_root.resolve());a.out.mkdir(parents=True,exist_ok=True)
    report={'schema':'llt.extended-attention.v1','status':'running','timestamp_utc':datetime.now(timezone.utc).isoformat(),
        'tensor_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.tensor_root,text=True).strip(),
        'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                         (Path(__file__),Path(__file__).with_name('extended_kernels.py'),Path(__file__).with_name('vulkan_attention.py'))},
        'protocol':{'samples':a.samples,'repeats':a.repeats,'gpu_timestamp_period_ns':period,
            'cache_and_weights':'selected FP16/FP32 storage','query_accumulator_output':'FP32',
            'correctness_replays':10,'reference':'independent float64; explicit/fused storage-rounded K/V vs unrounded absorbed',
            'excluded':['pipeline compilation','allocation','uploads','readback','CPU oracle'],
            'scope':'fixed-query attention sequences, not evolving model; live bytes are method resource ledgers, not process VRAM',
            'fused':'16-token workgroup tile; no global expanded KV; partition merge retained'},'cases':[]}
    def save(): (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    with device:
        if device.info['adapter']['backend_type']!='Vulkan' or device.info['adapter']['adapter_type']!='DiscreteGPU':
            raise RuntimeError('physical discrete Vulkan GPU required')
        report['device']=device.info
        lib=Library(tx,device,a.out)
        h,d=12,64
        for spec in configurations(a.smoke):
            s,r,b,dt=(spec[n] for n in ('context','rank','batch','dtype'));p=16
            rng=np.random.default_rng(np.random.SeedSequence([71,s,r,b]))
            c=rng.normal(size=(b,s,r)).astype(dt)
            w=(rng.normal(size=(2*h*d,r))/r**.5).astype(dt)
            q=(rng.normal(size=(b,h,d))*.5).astype(np.float32)
            expected_e=[];expected_a=[]
            for index in range(b):
                full,_,ea=oracle(c[index],w,q[index],h,d)
                kv=full.astype(dt).astype(np.float64).reshape(s,2,h,d)
                k,v=kv[:,0].transpose(1,0,2),kv[:,1].transpose(1,0,2)
                scores=np.einsum('hd,hsd->hs',q[index].astype(np.float64),k)*d**-.5
                probabilities=np.exp(scores-scores.max(axis=1,keepdims=True));probabilities/=probabilities.sum(axis=1,keepdims=True)
                expected_e.append(np.einsum('hs,hsd->hd',probabilities,v));expected_a.append(ea)
            expected_e,expected_a=np.array(expected_e),np.array(expected_a)
            resources=[]
            def upload(x):
                value=device.from_numpy(np.ascontiguousarray(x).ravel());resources.append(value);return value
            def empty(n,dtype='float32'):
                value=device.full(n,np.nan,dtype=dtype);resources.append(value);return value
            cb,wb,qb=upload(c),upload(w),upload(q)
            kvb=empty(b*s*2*h*d,dt);out=empty(b*h*d)
            ep,es=empty(b*h*p*d),empty(b*h*p*2)
            aq,al=empty(b*h*r),empty(b*h*r)
            ap,ats=empty(b*h*p*r),empty(b*h*p*2)
            expand=lib.call(kernels.matrix(b*s,2*h*d,r,dt),(cb,wb,kvb))
            ec=[lib.call(kernels.attention(s,h,d,r,b,dt,p,True),(qb,kvb,ep,es)),
                lib.call(kernels.merge_source(b*h,d,p),(ep,es,out))]
            ac=[lib.call(kernels.query(h,d,r,b,dt),(qb,wb,aq)),
                lib.call(kernels.attention(s,h,d,r,b,dt,p),(aq,cb,ap,ats)),
                lib.call(kernels.merge_source(b*h,r,p),(ap,ats,al)),
                lib.call(kernels.output(h,d,r,b,dt),(al,wb,out))]
            fc=[lib.call(kernels.fused(s,h,d,r,b,dt,p),(qb,cb,wb,ep,es)),ec[-1]]
            validation={}
            for name,sequence,expected in [('explicit',[expand]+ec,expected_e),('absorbed',ac,expected_a),('fused',fc,expected_e)]:
                plan=device.prepare_plan(sequence);errors=[]
                for _ in range(10):
                    plan.launch();actual=out.to_numpy().reshape(b,h,d)
                    np.testing.assert_allclose(actual,expected,atol=.0005 if dt=='float16' else 5e-6,rtol=.003 if dt=='float16' else .0001)
                    errors.append(float(np.max(np.abs(actual-expected))))
                plan.close();validation[name]={'passed':True,'maximum_absolute_error':max(errors)}
            warm=device.prepare_plan(ec);warm_device(warm,device);warm.close()
            resource_sets={'cached':[wb,qb,kvb,out,ep,es], 'reexpand':[cb,wb,qb,kvb,out,ep,es],
                'reuse':[cb,wb,qb,kvb,out,ep,es], 'absorbed':[cb,wb,qb,out,aq,al,ap,ats], 'fused':[cb,wb,qb,out,ep,es]}
            for t in spec['loops']:
                sequences={'cached':ec*t,'reexpand':([expand]+ec)*t,'reuse':[expand]+ec*t,'absorbed':ac*t,'fused':fc*t}
                row={**spec,'loops':t,'validation':validation,'timings':{},'live_tensor_bytes':{k:memory_bytes(v) for k,v in resource_sets.items()},
                    'cache_bytes':{'explicit_one_layer_loop':kvb.nbytes,'shared_latent':cb.nbytes},
                    'fused_workgroup_bytes':lib.get(kernels.fused(s,h,d,r,b,dt,p)).manifest['webgpu']['workgroup_storage_bytes']}
                for name,sequence in sequences.items():
                    plan=device.prepare_plan(sequence)
                    timing=measure_gpu(plan,device,a.samples,a.repeats,period)
                    timing['host']=measure(plan,device,a.samples,a.repeats)
                    timing['attention_calls_per_second']=b*t*1000/timing['median_ms']
                    timing['dispatches']=len(sequence)
                    row['timings'][name]=timing;plan.close()
                report['cases'].append(row);report['artifacts']=lib.records;save()
                print(spec['suite'],f'S={s} R={r} B={b} T={t} {dt}',
                    {n:round(v['median_ms'],3) for n,v in row['timings'].items()},flush=True)
            for resource in resources: resource.release()
        report['status']='passed';save();lib.close()
    print('Saved',a.out/'report.json',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tensor-root',type=Path,default=Path(__file__).resolve().parents[2]/'tensor')
    parser.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'results'/'extended')
    parser.add_argument('--samples',type=int,default=5);parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--smoke',action='store_true');run(parser.parse_args())
